# Dokumentacja modułu `exploration` (wersja obiektowa)

Moduł odpowiedzialny za eksploracyjną analizę danych (EDA) w ramach API FastAPI. Umożliwia przesyłanie plików CSV i wykonywanie podstawowych analiz statystycznych, analizy typów kolumn, korelacji, redukcji wymiarowości (PCA), generowania wykresów, inteligentnego uzupełniania braków oraz badania rozkładu wartości kolumn.

Kod został zorganizowany obiektowo: logika analityczna znajduje się w klasie `DataExplorer`, operacje na plikach w `FileHandler`, a endpointy w klasie `ExplorationRouter`.

## Struktura modułu

```
backend/exploration/
├── __init__.py          – oznacza pakiet
├── core.py              – klasa DataExplorer (wszystkie obliczenia, wykresy, imputacja)
├── utils.py             – klasa FileHandler (odczyt CSV, walidacja kolumn)
├── routes.py            – klasa ExplorationRouter (endpointy FastAPI)
```

## Plik `utils.py` – klasa `FileHandler`

Zawiera statyczne metody do obsługi plików i walidacji.

### `FileHandler.read_csv_safe(file: UploadFile) -> pd.DataFrame`

Odczytuje przesłany plik CSV i zwraca obiekt `pandas.DataFrame`. W przypadku błędu kodowania lub formatu zwraca `HTTPException`.

**Parametry:**
- `file` – obiekt `UploadFile` z FastAPI.

**Wyjątki:**
- `HTTPException(400)` – nieprawidłowe kodowanie (brak UTF-8) lub błąd parsowania CSV.

### `FileHandler.validate_columns(df: pd.DataFrame, columns: Optional[str], numeric_only: bool = True) -> List[str]`

Waliduje i przetwarza parametr `columns` (string z listą nazw rozdzielonych przecinkami). Zwraca listę nazw kolumn po walidacji. Jeśli `columns=None`, zwraca wszystkie odpowiednie kolumny (numeryczne lub wszystkie, w zależności od `numeric_only`).

**Parametry:**
- `df` – ramka danych.
- `columns` – string np. `"Age, Fare"` lub `None`.
- `numeric_only` – czy wymagać, aby kolumny były numeryczne (domyślnie `True`).

**Wyjątki:**
- `HTTPException(400)` – brak kolumn, nieistniejące kolumny, lub (gdy `numeric_only=True`) kolumny nienumeryczne.

## Plik `core.py` – klasa `DataExplorer`

Główna klasa zawierająca logikę analizy danych. Przyjmuje w konstruktorze `DataFrame` oraz opcjonalny próg dla kolumn kategorycznych (`cat_threshold`, domyślnie 10).

### Metody analityczne

#### `__init__(df: pd.DataFrame, cat_threshold: int = 10)`
- Tworzy kopię danych i automatycznie wykrywa typy kolumn (numeryczne, kategoryczne, tekstowe) za pomocą `_detect_column_types()`.

#### `column_types_report() -> Dict[str, Any]`
Zwiera raport o typach kolumn – taki sam jak poprzednio.

#### `basic_stats(columns: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]`
Oblicza podstawowe statystyki dla podanych kolumn numerycznych (domyślnie wszystkich numerycznych). Format odpowiedzi bez zmian.

#### `correlation_matrix(columns: Optional[List[str]] = None, method: str = 'pearson') -> Dict[str, Any]`
Oblicza macierz korelacji dla podanych kolumn (domyślnie wszystkie numeryczne). Metoda: `pearson`, `spearman`, `kendall`.

#### `pca(columns: Optional[List[str]] = None, n_components: Optional[int] = None, standardize: bool = True) -> Dict[str, Any]`
Wykonuje PCA na wybranych kolumnach numerycznych. Działa identycznie jak poprzednia funkcja `perform_pca`, ale jest metodą klasy.

#### `describe_all() -> Dict[str, Any]`
Zwraca kompleksowy raport: typy kolumn, statystyki dla wszystkich kolumn numerycznych oraz macierz korelacji Pearsona (jeśli dostępne co najmniej 2 kolumny numeryczne).

### Nowe metody: rozkład wartości i wykresy

#### `value_distribution(column: str) -> Dict[str, Any]`
Zwraca rozkład wartości w kolumnie. Dla kolumn numerycznych – histogram z automatycznymi przedziałami (`bins='auto'`) oraz podstawowe statystyki (min, max, średnia, mediana). Dla kolumn kategorycznych – liczebności każdej wartości.
**Przykład odpowiedzi (numeryczna):**
```json
{
  "column": "Age",
  "type": "numeric",
  "bins": [0.0, 10.0, 20.0, ...],
  "counts": [5, 42, ...],
  "statistics": {"min": 0.42, "max": 80.0, "mean": 29.7, "median": 28.0}
}
```
**Przykład (kategoryczna):**
```json
{
  "column": "Sex",
  "type": "categorical",
  "values": {"male": 577, "female": 314},
  "unique_count": 2
}
```

#### `plot_histogram(column: str, bins: int = 10) -> bytes`
Generuje histogram dla kolumny numerycznej i zwraca obraz PNG jako obiekt `bytes`. Wykorzystuje `matplotlib`.

#### `plot_boxplot(column: str) -> bytes`
Generuje boxplot (wykres pudełkowy) dla kolumny numerycznej i zwraca PNG.

#### `plot_correlation_heatmap(columns: Optional[List[str]] = None) -> bytes`
Generuje heatmapę korelacji dla podanych kolumn numerycznych (domyślnie wszystkich numerycznych) i zwraca PNG.

### Nowa metoda: inteligentne uzupełnianie braków

#### `smart_impute(threshold_drop_col: float = 0.2, num_strategy: str = 'median', cat_strategy: str = 'mode') -> Dict[str, Any]`
Analizuje braki danych i automatycznie podejmuje decyzje:
- Jeśli brakuje więcej niż `threshold_drop_col` (domyślnie 20%) wartości w kolumnie – kolumna zostaje usunięta (odnotowana w raporcie).
- W przeciwnym razie:
  - Dla kolumn numerycznych: uzupełnia wartością `mean`, `median` lub `knn` (na razie `knn` nie zaimplementowane w tej wersji, przyjmuje medianę). Domyślnie `median`.
  - Dla kolumn kategorycznych: uzupełnia najczęstszą wartością (`mode`) lub stałą `"MISSING"`. Domyślnie `mode`.

**Zwraca** słownik z raportem:
```json
{
  "dropped_columns": ["Cabin"],
  "imputed_columns": {"Age": {"strategy": "median", "value": 28.0}, ...},
  "new_shape": [891, 11]
}
```
Metoda modyfikuje wewnętrzny `DataFrame` (`self.df`) i ponownie wykrywa typy kolumn. Można potem wyeksportować zmodyfikowane dane (np. przez endpoint zwracający CSV).

## Plik `routes.py` – klasa `ExplorationRouter`

Definiuje wszystkie endpointy API. Router jest tworzony jako instancja klasy, a następnie dołączany do głównej aplikacji FastAPI.

### Endpointy (wszystkie pod `/exploration`):

| Metoda | Ścieżka | Opis |
|--------|---------|------|
| POST | `/basic-stats` | Podstawowe statystyki dla wybranych kolumn numerycznych |
| POST | `/column-types` | Raport typów kolumn |
| POST | `/correlation` | Macierz korelacji |
| POST | `/pca` | Analiza głównych składowych |
| POST | `/describe-all` | Kompleksowy raport |
| POST | `/distribution` | Rozkład wartości w kolumnie (JSON) |
| POST | `/plot/histogram` | Histogram (PNG) |
| POST | `/plot/boxplot` | Boxplot (PNG) |
| POST | `/plot/correlation-heatmap` | Heatmapa korelacji (PNG) |
| POST | `/smart-impute` | Inteligentne uzupełnianie braków – zwraca CSV lub raport JSON |

### Opis nowych endpointów

#### `POST /distribution`

**Parametry formularza:**
- `file` (wymagany) – plik CSV.
- `column` (wymagany) – nazwa kolumny.

**Odpowiedź (200 OK):** JSON z rozkładem (szczegóły w `value_distribution`).

**Błędy:**
- `400` – kolumna nie istnieje.

**Przykład `curl`:**
```bash
curl -X POST http://localhost:8000/exploration/distribution \
  -F "file=@Titanic-Dataset.csv" \
  -F "column=Age"
```

#### `POST /plot/histogram`

**Parametry formularza:**
- `file` (wymagany)
- `column` (wymagany) – kolumna numeryczna.
- `bins` (opcjonalny) – liczba przedziałów, domyślnie 10.

**Odpowiedź (200 OK):** obraz PNG (`image/png`).

**Przykład:**
```bash
curl -X POST http://localhost:8000/exploration/plot/histogram \
  -F "file=@Titanic-Dataset.csv" \
  -F "column=Age" \
  -F "bins=20" --output histogram.png
```

#### `POST /plot/boxplot`

**Parametry formularza:**
- `file` (wymagany)
- `column` (wymagany)

**Odpowiedź:** PNG.

#### `POST /plot/correlation-heatmap`

**Parametry formularza:**
- `file` (wymagany)
- `columns` (opcjonalny) – lista kolumn numerycznych (rozdzielona przecinkami). Jeśli brak – używane są wszystkie numeryczne.

**Odpowiedź:** PNG.

#### `POST /smart-impute`

**Parametry formularza:**
- `file` (wymagany)
- `threshold_drop_col` (opcjonalny, float, domyślnie 0.2) – próg usuwania kolumny (20% braków).
- `num_strategy` (opcjonalny, `median`/`mean`/`knn`, domyślnie `median`) – strategia dla numerycznych.
- `cat_strategy` (opcjonalny, `mode`/`constant`, domyślnie `mode`) – strategia dla kategorycznych.
- `return_csv` (opcjonalny, boolean, domyślnie `True`) – jeśli `True`, zwraca nowy plik CSV z uzupełnionymi danymi; jeśli `False`, zwraca raport JSON.

**Odpowiedź:** 
- Jeśli `return_csv=true` – CSV (`text/csv`).
- Jeśli `return_csv=false` – JSON z raportem.

**Przykład (zwrócenie CSV):**
```bash
curl -X POST http://localhost:8000/exploration/smart-impute \
  -F "file=@Titanic-Dataset.csv" \
  -F "threshold_drop_col=0.3" \
  -F "num_strategy=mean" \
  -F "return_csv=true" --output imputed.csv
```

**Przykład (zwrócenie raportu):**
```bash
curl -X POST http://localhost:8000/exploration/smart-impute \
  -F "file=@Titanic-Dataset.csv" \
  -F "return_csv=false"
```

## Uruchomienie i przykład użycia (aktualizacja)

1. Uruchom serwer w katalogu `backend`:
   ```bash
   uvicorn main:app --reload
   ```

2. Sprawdzenie rozkładu wartości:
   ```bash
   curl -X POST http://localhost:8000/exploration/distribution -F "file=@Titanic-Dataset.csv" -F "column=Sex"
   ```

3. Wygenerowanie histogramu:
   ```bash
   curl -X POST http://localhost:8000/exploration/plot/histogram -F "file=@Titanic-Dataset.csv" -F "column=Age" -F "bins=15" --output wiek.png
   ```

4. Inteligentne uzupełnienie braków (od razu zapis do pliku):
   ```bash
   curl -X POST http://localhost:8000/exploration/smart-impute -F "file=@Titanic-Dataset.csv" -F "return_csv=true" --output Titanic_clean.csv
   ```

## Uwagi końcowe

- Wszystkie funkcjonalności z pierwszej wersji (statystyki, korelacje, PCA) są nadal dostępne.
- Kod w pełni obiektowy – klasy `DataExplorer`, `FileHandler`, `ExplorationRouter`.
- Wykresy są generowane przy użyciu `matplotlib` i `seaborn`, pamiętaj o instalacji tych bibliotek.
- Inteligentna imputacja działa automatycznie; można rozszerzyć o bardziej zaawansowane metody (np. KNNImputer z scikit-learn).
```

Dokumentacja została zaktualizowana o nowe endpointy, opisy klas i metod, a także przykłady użycia. Zachowano spójność z poprzednią wersją, jednocześnie podkreślając zmiany obiektowe.