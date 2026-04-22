# Dokumentacja modułu `exploration`

Moduł odpowiedzialny za eksploracyjną analizę danych (EDA) w ramach API FastAPI. Umożliwia przesyłanie plików CSV i wykonywanie podstawowych analiz statystycznych, analizy typów kolumn, korelacji oraz redukcji wymiarowości (PCA).

## Struktura modułu

```
backend/exploration/
├── core.py      – funkcje analityczne (statystyki, korelacja, PCA)
├── routes.py    – endpointy API (FastAPI)
├── utils.py     – funkcje pomocnicze (odczyt CSV, walidacja kolumn)

```




---

## Plik `utils.py`

Zawiera funkcje pomocnicze do obsługi plików i walidacji danych.

### `read_csv_safe(file: UploadFile) -> pd.DataFrame`

Odczytuje przesłany plik CSV i zwraca obiekt `pandas.DataFrame`. W przypadku błędu kodowania lub formatu zwraca `HTTPException`.

**Parametry:**
- `file` – obiekt `UploadFile` z FastAPI.

**Wyjątki:**
- `HTTPException(400)` – nieprawidłowe kodowanie (brak UTF-8) lub błąd parsowania CSV.

**Przykład użycia (w endpointzie):**
```python
df = read_csv_safe(file)
```

### `get_numeric_columns(df: pd.DataFrame) -> List[str]`

Zwraca listę nazw kolumn numerycznych (typy `int`, `float`).

### `get_categorical_columns(df: pd.DataFrame, threshold: int = 10) -> List[str]`

Zwraca listę kolumn, które są typu `object` lub `category` i mają liczbę unikalnych wartości nie większą niż `threshold`. Domyślnie `10`.

### `validate_columns(df: pd.DataFrame, columns: Optional[str], numeric_only: bool = True) -> List[str]`

Waliduje i przetwarza parametr `columns` przekazany jako string z listą nazw rozdzielonych przecinkami.

**Parametry:**
- `df` – ramka danych.
- `columns` – string np. `"Age, Fare"` lub `None` (wtedy wybiera wszystkie odpowiednie kolumny).
- `numeric_only` – czy wymagać, aby kolumny były numeryczne (domyślnie `True`).

**Zwraca:** lista nazw kolumn (po walidacji).

**Wyjątki:**
- `HTTPException(400)` – brak kolumn, nieistniejące kolumny, lub (gdy `numeric_only=True`) kolumny nienumeryczne.

---

## Plik `core.py`

Zawiera funkcje realizujące obliczenia statystyczne i analizy.

### `basic_stats(df: pd.DataFrame, columns: List[str]) -> Dict[str, Dict[str, Any]]`

Oblicza podstawowe statystyki opisowe dla podanych kolumn numerycznych.

**Parametry:**
- `df` – ramka danych.
- `columns` – lista nazw kolumn numerycznych.

**Zwraca:** słownik, gdzie klucz = nazwa kolumny, wartość = słownik z następującymi polami:
```json
{
  "count": 714,
  "mean": 29.699,
  "median": 28.0,
  "std": 14.526,
  "min": 0.42,
  "max": 80.0,
  "skewness": 0.389,
  "kurtosis": -0.553,
  "q25": 20.125,
  "q75": 38.0
}
```
Jeśli kolumna nie zawiera danych, zwraca `{"error": "Brak danych w kolumnie"}`.


### `column_types_report(df: pd.DataFrame) -> Dict[str, Any]`

Generuje raport podziału kolumn na typy.

**Zwraca:**
```json
{
  "numeric_columns": ["Age", "Fare", "SibSp", "Parch"],
  "categorical_columns": ["Sex", "Embarked", "Pclass"],
  "text_columns": ["Name", "Ticket", "Cabin"],
  "total_columns": 12,
  "numeric_count": 4,
  "categorical_count": 3
}
```
- `numeric_columns` – wszystkie kolumny numeryczne.
- `categorical_columns` – kolumny obiektowe/kategoryczne z liczbą unikalnych wartości ≤ 10.
- `text_columns` – pozostałe kolumny obiektowe (unikalnych > 10).

### `correlation_matrix(df: pd.DataFrame, columns: List[str], method: str = 'pearson') -> Dict[str, Any]`

Oblicza macierz korelacji dla podanych kolumn numerycznych.

**Parametry:**
- `method` – `'pearson'`, `'spearman'` lub `'kendall'`. Domyślnie `'pearson'`.

**Zwraca:**
```json
{
  "columns": ["Age", "Fare", "SibSp"],
  "correlation_matrix": [[1.0, 0.1, -0.2], [0.1, 1.0, 0.3], [-0.2, 0.3, 1.0]],
  "method": "pearson"
}
```
Wartości `NaN` są zastępowane `null`.

### `perform_pca(df: pd.DataFrame, columns: List[str], n_components: Optional[int] = None, standardize: bool = True) -> Dict[str, Any]`

Wykonuje PCA na wybranych kolumnach numerycznych. Wiersze z brakującymi wartościami są automatycznie usuwane.

**Parametry:**
- `n_components` – liczba składowych głównych. Jeśli `None`, ustawiana jako `min(liczba_wierszy, liczba_kolumn)`.
- `standardize` – czy standaryzować dane (średnia 0, wariancja 1). Domyślnie `True`.

**Zwraca:**
```json
{
  "explained_variance_ratio": [0.45, 0.23, 0.12],
  "cumulative_variance": [0.45, 0.68, 0.80],
  "loadings": {
    "PC1": {"Age": -0.55, "Fare": 0.78, "SibSp": 0.29},
    "PC2": {"Age": 0.62, "Fare": 0.45, "SibSp": -0.63}
  },
  "n_components": 3,
  "n_samples_used": 714,
  "standardized": true
}
```
- `loadings` – tylko dla pierwszych 5 komponentów (`PC1`…`PC5`). Każdy komponent zawiera ładunki czynnikowe dla oryginalnych kolumn.

**Wyjątki:**
- `ValueError` – gdy po usunięciu braków danych pozostanie < 2 wiersze.

---

## Plik `routes.py`

Definiuje endpointy FastAPI (router o nazwie `router`). Wszystkie endpointy przyjmują plik CSV jako `multipart/form-data`.

### `POST /basic-stats`

Zwraca podstawowe statystyki dla wybranych kolumn numerycznych.

**Parametry formularza:**
- `file` (wymagany) – plik CSV.
- `columns` (opcjonalny) – string z listą nazw kolumn rozdzielonych przecinkami. Jeśli brak – brane są wszystkie kolumny numeryczne.

**Odpowiedź (200 OK):** obiekt JSON w formacie opisanym w `basic_stats`.

**Błędy:**
- `400` – brak kolumn numerycznych, podane kolumny nie istnieją lub nie są numeryczne.
- `500` – błąd podczas obliczeń.

**Przykład `curl`:**
```bash
curl -X POST http://localhost:8000/explore/basic-stats \
  -F "file=@Titanic-Dataset.csv" \
  -F "columns=Age,Fare"
```

### `POST /column-types`

Analizuje typy wszystkich kolumn w pliku CSV.

**Parametry formularza:**
- `file` (wymagany)

**Odpowiedź (200):** JSON z raportem (`column_types_report`).

**Przykład:**
```bash
curl -X POST http://localhost:8000/explore/column-types -F "file=@Titanic-Dataset.csv"
```

### `POST /correlation`

Oblicza macierz korelacji dla kolumn numerycznych.

**Parametry formularza:**
- `file` (wymagany)
- `method` (opcjonalny) – `pearson`, `spearman`, `kendall`. Domyślnie `pearson`.
- `columns` (opcjonalny) – lista kolumn. Jeśli brak – wszystkie numeryczne.

**Odpowiedź (200):** JSON (`correlation_matrix`).

**Błędy:**
- `400` – mniej niż 2 kolumny numeryczne lub nieprawidłowa metoda.

**Przykład:**
```bash
curl -X POST http://localhost:8000/explore/correlation \
  -F "file=@Titanic-Dataset.csv" \
  -F "method=spearman" \
  -F "columns=Age,Fare,SibSp"
```

### `POST /pca`

Wykonuje analizę głównych składowych.

**Parametry formularza:**
- `file` (wymagany)
- `columns` (opcjonalny) – lista kolumn numerycznych. Wymagane co najmniej 2.
- `n_components` (opcjonalny) – liczba składowych. Jeśli brak – `min(wiersze, kolumny)`.
- `standardize` (opcjonalny) – `true` lub `false`. Domyślnie `true`.

**Odpowiedź (200):** JSON (`perform_pca`).

**Błędy:**
- `400` – za mało kolumn numerycznych (<2), lub po usunięciu braków za mało wierszy (<2).
- `500` – inne błędy PCA.

**Przykład:**
```bash
curl -X POST http://localhost:8000/explore/pca \
  -F "file=@Titanic-Dataset.csv" \
  -F "columns=Age,Fare,SibSp,Parch" \
  -F "n_components=2" \
  -F "standardize=true"
```

### `POST /describe-all`

Zwraca kompleksowy raport: typy kolumn, podstawowe statystyki dla wszystkich kolumn numerycznych oraz macierz korelacji Pearsona (jeśli istnieją co najmniej 2 kolumny numeryczne).

**Parametry formularza:**
- `file` (wymagany)

**Odpowiedź (200):**
```json
{
  "column_types": { ... },
  "basic_statistics": { ... },
  "correlation_pearson": { ... }   // lub null, gdy brak wystarczającej liczby kolumn
}
```

**Przykład:**
```bash
curl -X POST http://localhost:8000/explore/describe-all -F "file=@Titanic-Dataset.csv"
```

---

## Przykład użycia 

1. Uruchom serwer:
   ```bash
   uvicorn main:app --reload
   ```

2. Sprawdź typy kolumn:
   ```bash
   curl -X POST http://localhost:8000/explore/column-types -F "file=@Titanic-Dataset.csv"
   ```

3. Oblicz statystyki dla wieku i ceny biletu:
   ```bash
   curl -X POST http://localhost:8000/explore/basic-stats -F "file=@Titanic-Dataset.csv" -F "columns=Age,Fare"
   ```

4. Wykonaj PCA dla cech numerycznych:
   ```bash
   curl -X POST http://localhost:8000/explore/pca -F "file=@Titanic-Dataset.csv" -F "columns=Age,Fare,SibSp,Parch" -F "n_components=2"
   ```

5. Pobierz pełny raport:
   ```bash
   curl -X POST http://localhost:8000/explore/describe-all -F "file=@Titanic-Dataset.csv"
   ```

---


