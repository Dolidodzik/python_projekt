# Dokumentacja modułu `exploration` 

Moduł odpowiada za eksploracyjną analizę danych (EDA) w API FastAPI. Obsługuje przesyłanie plików CSV i zwraca wyniki analiz w formacie JSON: statystyki, typy kolumn, korelacje, PCA, raport zbiorczy oraz rozkład wartości.

Kod jest zorganizowany obiektowo: logika analityczna znajduje się w klasie `DataExplorer`, operacje na plikach w `FileHandler`, a endpointy w klasie `ExplorationRouter`.

## Struktura modułu

```
backend/exploration/
├── core.py              – obliczenia i raporty
├── utils.py             – odczyt CSV, walidacja kolumn
├── routes.py            – endpointy FastAPI
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
Zwraca raport o typach kolumn.

#### `basic_stats(columns: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]`
Oblicza podstawowe statystyki dla podanych kolumn numerycznych (domyślnie wszystkich numerycznych).

#### `correlation_matrix(columns: Optional[List[str]] = None, method: str = 'pearson') -> Dict[str, Any]`
Oblicza macierz korelacji dla podanych kolumn (domyślnie wszystkie numeryczne). Metoda: `pearson`, `spearman`, `kendall`.

#### `pca(columns: Optional[List[str]] = None, n_components: Optional[int] = None, standardize: bool = True) -> Dict[str, Any]`
Wykonuje PCA na wybranych kolumnach numerycznych.

#### `value_distribution(column: str) -> Dict[str, Any]`
Zwraca rozkład wartości w kolumnie. Dla kolumn numerycznych podaje przedziały histogramu (`bins='auto'`) i liczebności. Dla kolumn kategorycznych zwraca liczebności wartości.

#### `describe_all() -> Dict[str, Any]`
Zwraca kompleksowy raport: typy kolumn, statystyki dla wszystkich kolumn numerycznych oraz macierz korelacji Pearsona (jeśli dostępne co najmniej 2 kolumny numeryczne).

## Plik `routes.py` – klasa `ExplorationRouter`

Definiuje endpointy API. Router jest tworzony jako instancja klasy, a następnie dołączany do głównej aplikacji FastAPI.

### Endpointy (wszystkie pod `/exploration`)

| Metoda | Ścieżka | Opis |
|--------|---------|------|
| POST | `/basic-stats` | Podstawowe statystyki dla wybranych kolumn numerycznych |
| POST | `/column-types` | Raport typów kolumn |
| POST | `/correlation` | Macierz korelacji |
| POST | `/pca` | Analiza głównych składowych |
| POST | `/describe-all` | Kompleksowy raport |
| POST | `/distribution` | Rozkład wartości w kolumnie (JSON) |

### `POST /distribution`

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

