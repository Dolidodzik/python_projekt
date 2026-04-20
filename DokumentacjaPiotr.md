# !!!  DO EDYCJI  !!!

## Opis modułu

Moduł `exploration` dostarcza narzędzia do wstępnej analizy zbiorów danych w formacie CSV. Umożliwia szybkie obliczenie podstawowych statystyk opisowych, identyfikację typów kolumn, badanie korelacji między zmiennymi numerycznymi oraz wykonanie analizy głównych składowych (PCA).

## Wymagania

Do poprawnego działania modułu niezbędne są następujące biblioteki Pythona (ujęte w pliku `requirements.txt`):

- fastapi
- uvicorn
- pandas
- numpy
- scikit-learn
- python-multipart

Przed uruchomieniem należy zainstalować zależności poleceniem:

```bash
pip install -r requirements.txt
```

## Struktura katalogów

```
exploration/
├── __init__.py          # Plik inicjalizacyjny (może być pusty)
├── utils.py             # Funkcje pomocnicze (odczyt pliku, walidacja kolumn)
├── core.py              # Właściwe obliczenia statystyczne i PCA
├── routes.py            # Definicje endpointów FastAPI
└── README.md            # Niniejsza dokumentacja
```

## Uruchomienie serwera

Należy przejść do katalogu `backend` i uruchomić serwer Uvicorn:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Po starcie API będzie dostępne pod adresem `http://localhost:8000`. Interaktywna dokumentacja Swagger UI znajduje się pod adresem `http://localhost:8000/docs`.

## Endpointy

Wszystkie endpointy dostępne są pod wspólnym prefiksem `/exploration`. Poniżej znajduje się szczegółowy opis każdego z nich.

### 1. `/exploration/basic-stats` (POST)

**Cel:** Obliczenie podstawowych statystyk opisowych dla wybranych kolumn numerycznych.

**Parametry formularza (`multipart/form-data`):**

| Nazwa    | Typ           | Wymagany | Opis                                                                 |
|----------|---------------|----------|----------------------------------------------------------------------|
| `file`   | plik (.csv)   | Tak      | Plik CSV z danymi.                                                   |
| `columns`| string        | Nie      | Lista kolumn oddzielonych przecinkami, np. `"Age,Fare"`. Jeśli brak, analizowane są wszystkie kolumny numeryczne. |

**Przykładowe żądanie (Python):**

```python
import requests

url = "http://localhost:8000/exploration/basic-stats"
files = {"file": open("Titanic-Dataset.csv", "rb")}
data = {"columns": "Age,Fare,SibSp"}

response = requests.post(url, files=files, data=data)
print(response.json())
```

**Przykład odpowiedzi (JSON):**

```json
{
  "Age": {
    "count": 714,
    "mean": 29.69911764705882,
    "median": 28.0,
    "std": 14.526497,
    "min": 0.42,
    "max": 80.0,
    "skewness": 0.389108,
    "kurtosis": 0.178274,
    "q25": 20.125,
    "q75": 38.0
  },
  "Fare": {
    "count": 891,
    "mean": 32.204208,
    "median": 14.4542,
    "std": 49.693429,
    "min": 0.0,
    "max": 512.3292,
    "skewness": 4.787316,
    "kurtosis": 33.148267,
    "q25": 7.9104,
    "q75": 31.0
  }
}
```

**Kody błędów:**

- `400 Bad Request` – nieprawidłowy plik CSV, podana kolumna nie istnieje lub nie jest numeryczna.
- `500 Internal Server Error` – błąd podczas obliczeń.

---

### 2. `/exploration/column-types` (POST)

**Cel:** Analiza typów kolumn w dostarczonym pliku CSV.

**Parametry formularza:**

| Nazwa  | Typ         | Wymagany | Opis           |
|--------|-------------|----------|----------------|
| `file` | plik (.csv) | Tak      | Plik CSV z danymi. |

**Przykładowe żądanie:**

```python
url = "http://localhost:8000/exploration/column-types"
files = {"file": open("Titanic-Dataset.csv", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

**Przykład odpowiedzi:**

```json
{
  "numeric_columns": ["PassengerId", "Survived", "Pclass", "Age", "SibSp", "Parch", "Fare"],
  "categorical_columns": ["Sex", "Embarked"],
  "text_columns": ["Name", "Ticket", "Cabin"],
  "total_columns": 12,
  "numeric_count": 7,
  "categorical_count": 2
}
```

**Uwaga:** Kolumny kategoryczne rozpoznawane są jako kolumny tekstowe (`object`) z liczbą unikalnych wartości nie większą niż 10.

**Kody błędów:**

- `400 Bad Request` – plik nie jest poprawnym CSV.

---

### 3. `/exploration/correlation` (POST)

**Cel:** Obliczenie macierzy korelacji dla wybranych kolumn numerycznych.

**Parametry formularza:**

| Nazwa     | Typ           | Wymagany | Opis                                                                                        |
|-----------|---------------|----------|---------------------------------------------------------------------------------------------|
| `file`    | plik (.csv)   | Tak      | Plik CSV z danymi.                                                                          |
| `method`  | string        | Nie      | Metoda korelacji: `"pearson"` (domyślnie), `"spearman"` lub `"kendall"`.                     |
| `columns` | string        | Nie      | Lista kolumn oddzielonych przecinkami. Jeśli brak, użyte zostaną wszystkie kolumny numeryczne. |

**Przykład żądania:**

```python
url = "http://localhost:8000/exploration/correlation"
files = {"file": open("Titanic-Dataset.csv", "rb")}
data = {"method": "pearson", "columns": "Age,Fare,SibSp,Parch"}

response = requests.post(url, files=files, data=data)
print(response.json())
```

**Przykład odpowiedzi:**

```json
{
  "columns": ["Age", "Fare", "SibSp", "Parch"],
  "correlation_matrix": [
    [1.0, 0.096688, -0.308246, -0.189119],
    [0.096688, 1.0, 0.159651, 0.216225],
    [-0.308246, 0.159651, 1.0, 0.414838],
    [-0.189119, 0.216225, 0.414838, 1.0]
  ],
  "method": "pearson"
}
```

**Kody błędów:**

- `400 Bad Request` – nieprawidłowa metoda, mniej niż 2 kolumny numeryczne lub kolumna nie istnieje.
- `500 Internal Server Error` – błąd podczas obliczeń.

---

### 4. `/exploration/pca` (POST)

**Cel:** Przeprowadzenie analizy głównych składowych (PCA) na wybranych kolumnach numerycznych.

**Parametry formularza:**

| Nazwa          | Typ           | Wymagany | Opis                                                                                     |
|----------------|---------------|----------|------------------------------------------------------------------------------------------|
| `file`         | plik (.csv)   | Tak      | Plik CSV z danymi.                                                                       |
| `columns`      | string        | Nie      | Lista kolumn oddzielonych przecinkami. Domyślnie wszystkie numeryczne.                    |
| `n_components` | integer       | Nie      | Liczba składowych do wyznaczenia. Domyślnie tyle, ile jest kolumn (lub wierszy bez braków). |
| `standardize`  | boolean       | Nie      | Czy standaryzować dane przed PCA (`true`/`false`). Domyślnie `true`.                      |

**Przykład żądania:**

```python
url = "http://localhost:8000/exploration/pca"
files = {"file": open("Titanic-Dataset.csv", "rb")}
data = {
    "columns": "Age,Fare,SibSp,Parch",
    "n_components": 3,
    "standardize": True
}

response = requests.post(url, files=files, data=data)
print(response.json())
```

**Przykład odpowiedzi (fragment):**

```json
{
  "explained_variance_ratio": [0.4143, 0.2621, 0.2015],
  "cumulative_variance": [0.4143, 0.6764, 0.8779],
  "loadings": {
    "PC1": {
      "Age": -0.35,
      "Fare": 0.55,
      "SibSp": 0.62,
      "Parch": 0.44
    },
    "PC2": {
      "Age": 0.82,
      "Fare": 0.11,
      "SibSp": -0.08,
      "Parch": 0.13
    },
    "PC3": {
      "Age": 0.02,
      "Fare": -0.72,
      "SibSp": 0.11,
      "Parch": 0.67
    }
  },
  "n_components": 3,
  "n_samples_used": 714,
  "standardized": true
}
```

**Uwaga:** Przed wykonaniem PCA wiersze zawierające braki danych w analizowanych kolumnach są usuwane.

**Kody błędów:**

- `400 Bad Request` – mniej niż 2 kolumny numeryczne, zbyt mało danych po usunięciu braków, nieistniejąca kolumna.
- `500 Internal Server Error` – błąd wewnętrzny biblioteki scikit-learn.

---

### 5. `/exploration/describe-all` (POST)

**Cel:** Wygenerowanie skróconego raportu łączącego informacje o typach kolumn, podstawowych statystykach wszystkich kolumn numerycznych oraz macierzy korelacji Pearsona.

**Parametry formularza:**

| Nazwa  | Typ         | Wymagany | Opis           |
|--------|-------------|----------|----------------|
| `file` | plik (.csv) | Tak      | Plik CSV z danymi. |

**Przykład żądania:**

```python
url = "http://localhost:8000/exploration/describe-all"
files = {"file": open("Titanic-Dataset.csv", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

**Struktura odpowiedzi:**

```json
{
  "column_types": { ... },        // wynik identyczny z /column-types
  "basic_statistics": { ... },    // wynik /basic-stats dla wszystkich kolumn numerycznych
  "correlation_pearson": { ... }  // wynik /correlation z metodą 'pearson' (lub null, jeśli mniej niż 2 kolumny numeryczne)
}
```

## Obsługa błędów

W przypadku wystąpienia błędu walidacji (np. brak wymaganej kolumny, niepoprawny plik) serwer zwróci odpowiedź z kodem HTTP 400 oraz komunikatem w formacie:

```json
{
  "detail": "Opis błędu"
}
```

Przykładowy błąd:

```json
{
  "detail": "Kolumny nie istnieją w pliku: ['Weight']"
}
```
