# Dokumentacja modułu `statistic_tests` 

Projekt umożliwia automatyczny dobór i wykonanie odpowiednich testów statystycznych na podstawie danych przekazanych przez użytkownika.

## Struktura modułu

```
backend/statistic_tests/
├── anova.py                  -Analiza wariancji
├── chi_square.py             -Test Chi-kwadrat
├── correlation.py            -Korelacja  
├── covariance.py             -Kowariancja
├── mann_whitney.py           -Test Manna-Whitneya  
├── normality.py              -Test normalności  
├── statistical_engine.py     -Automatyczne dobieranie testów statystycznych         
├── t_test.py                 -Test t-Studenta    
```

## Testy statystyczne

## `anova.py`

Analiza wariancji (anova) służy do porównywania średnich pomiędzy więcej niż dwiema grupami. Pozwala jednym testem ustalić, czy przynajmniej jedna z porównywanych grup znacząco różni się od pozostałych pod kątem badanej cechy.

![ANOVA](screenshots/anova.png)

Działanie testu na przykładzie `anova_Embarked_Fare`:

Do przeprowadzenia analizy wariancji wybrano kolumny:
- `Embarked`- określa port, w którym pasażer wszedł na pokład statku i zawiera trzy grupy: S (Southampton), C (Cherbourg), Q (Queenstown).
- `Fare`- zawiera cenę biletu pasażera i jest zmienną liczbową.

Celem analizy jest sprawdzenie, czy średnia cena biletu różni się pomiędzy pasażerami rozpoczynającymi podróż w różnych portach.

Najpierw wyznaczane są:
- średnia dla każdej grupy,
- średnia dla całego zbioru danych,
- wariancja pomiędzy grupami,
- wariancja wewnątrz grup.

Następnie obliczane jest f_value = (wariancja pomiędzy grupami) / (wariancja wewnątrz grup)
oraz p_value obliczenia wykonywane są przy użyciu funkcji `f_oneway()` z biblioteki SciPy. 

Otrzymane wartości:
- "f_value": 38.140305200112635, 
- "p_value": 1.2896450252637473e-16

Wartość F jest stosunkowo wysoka co oznacza, że port wejścia pasażera na statek ma wyraźny związek z ceną zakupionego biletu.
Wartość P jest niska co oznacza, że prawdopodobieństwo uzyskania tak dużych różnic pomiędzy grupami wyłącznie przez przypadek jest praktycznie zerowe.

## `chi_square.py`

Test Chi-kwadrat sprawdza, czy pomiędzy dwiema zmiennymi kategorycznymi istnieje statystycznie istotna zależność, czy też są one od siebie niezależne. Najczęściej odpowiada na pytanie, czy zaobserwowane różnice w grupach są wynikiem przypadku, czy rzeczywistej relacji.

![chi_square](screenshots/chi_square.png)

Działanie testu na przykładzie `chi_square_Sex_Embarked`:

Do przeprowadzenia testu Chi-kwadrat wybrano kolumny:
- `Sex`-płeć,
- `Embarked`-port wejścia na statek.

Celem analizy jest sprawdzenie, czy istnieje zależność pomiędzy płcią pasażera a portem wejścia na statek.

Działanie:
- Najpierw sprawdzane jest, ilu pasażerów należy do poszczególnych kombinacji badanych grup, np. ilu mężczyzn i ile kobiet wsiadło w każdym porcie.
- Następnie wyznaczane są wartości oczekiwane (expected) przy założeniu, że zmienne są od siebie niezależne.
- Następnie obliczane jest p_value (przy użyciu funkcji chi2_contingency() z biblioteki SciPy).

Otrzymane wartości:
- "chi2": 13.355630515001746, 
- "p_value": 0.0012585245232290142, 
- "dof": 2

Wartość chi2 jest stosunkowo wysoka, co sugeruje istnienie zależności pomiędzy płcią pasażera a portem wejścia na statek.
Wartość p jest niska co oznacza, że prawdopodobieństwo uzyskania takich różnic wyłącznie przez przypadek jest bardzo małe.
Wartość dof oznacza liczbę stopni swobody wykorzystywaną podczas obliczania wartości p-value. Dla analizowanych zmiennych Sex (2 grupy) oraz Embarked (3 grupy) liczba stopni swobody wynosi 2.
Wartości expected pokazują, ilu pasażerów powinno znaleźć się w każdej grupie, gdyby płeć pasażera nie miała żadnego związku z portem wejścia na statek.

## `correlation.py`

Korelacja służy do określania siły oraz kierunku zależności pomiędzy dwiema zmiennymi liczbowymi.

![correlation](screenshots/correlation.png)

Działanie testu na przykładzie `correlation_Pclass_Fare`:

Do przeprowadzenia analizy wybrano kolumny:
- `Pclass`–klasa pasażera (1, 2 lub 3 klasa),
- `Fare`–cena biletu pasażera.

Celem analizy jest sprawdzenie, czy istnieje zależność pomiędzy klasą pasażera a ceną biletu.

Działanie:
- Najpierw pobierane są wartości z obu analizowanych kolumn.
- Następnie obliczany jest współczynnik korelacji Pearsona określający siłę oraz kierunek zależności pomiędzy zmiennymi.
- Obliczenia wykonywane są przy użyciu funkcji corr() z biblioteki Pandas.

Otrzymana wartość:
- Wynik został zwrócony w postaci macierzy korelacji. Wartości na przekątnej są równe 1, ponieważ każda zmienna jest w pełni skorelowana sama ze sobą.
Interesuje nas "correlation": -0.5494996199439076

Otrzymana wartość korelacji wynosi około -0.55, co oznacza umiarkowaną ujemną zależność pomiędzy analizowanymi zmiennymi.Ujemny znak oznacza, że wraz ze wzrostem jednej zmiennej druga zmienna ma tendencję do spadku.
Pasażerowie podróżujący w 1 klasie płacili średnio więcej za bilety niż pasażerowie podróżujący w 2 i 3 klasie.

## `covariance.py`

Kowariancja pozwala sprawdzić, czy zmienne mają tendencję do wzrostu razem, czy też wzrost jednej zmiennej wiąże się ze spadkiem drugiej.

![covariance](screenshots/covariance.png)

Działanie testu na przykładzie `covariance_Pclass_Fare`:

Do przeprowadzenia analizy wybrano kolumny:
- `Pclass` - klasa pasażera,
- `Fare` - cena biletu.

Celem analizy jest sprawdzenie, jak zmienia się cena biletu w zależności od klasy pasażera.

Działanie:
- Najpierw pobierane są wartości z obu analizowanych kolumn.
- Następnie obliczana jest kowariancja pomiędzy zmiennymi.
- Obliczenia wykonywane są przy użyciu funkcji cov() z biblioteki Pandas.

Otrzymana wartość (wynik został zwrócony w postaci macierzy korelacji):
- "Pclass": 0.6990151199889029- pokazuje, jak bardzo wartości są rozproszone wokół średniej.
- "Fare": 2469.436845743115- pokazuje, jak bardzo ceny biletów różnią się od średniej ceny biletu.
- "Fare": -22.830196170065186- właściwa kowariancja między badanymi zmiennymi.

Ujemna wartość oznacza, że wraz ze wzrostem jednej zmiennej druga zmienna ma tendencję do spadku. Wraz ze wzrostem numeru klasy pasażera średnia cena biletu maleje.


## `mann_whitney.py`

Test Manna-Whitneya służy do sprawdzenia, czy istnieją istotne statystycznie różnice między dwiema niezależnymi grupami.

![mann_whitney](screenshots/mann_whitney.png)

Działanie testu na przykładzie `mann_whitney_Sex_Fare`:

Do przeprowadzenia analizy wybrano kolumny:
- `Sex`-płeć pasażera,
- `Fare`-cena biletu.

Celem analizy było sprawdzenie, czy ceny biletów różnią się pomiędzy kobietami i mężczyznami.

Działanie:
- Najpierw dane są dzielone na dwie grupy na podstawie kolumny `Sex`.
- Następnie wszystkie wartości z obu grup są porządkowane rosnąco i nadawane są im rangi.
- Na podstawie rang obliczana jest statystyka U.
- Następnie obliczane jest `p_value`.
- Obliczenia wykonywane są przy użyciu funkcji `mannwhitneyu()` z biblioteki `SciPy`.

Otrzymane wartości:
- "u_statistic": 62175,
- "p_value": 9.61232696290926e-15

Wartość u_statistic określa różnicę pomiędzy rozkładami porównywanych grup.
Wartość p_value jest bardzo mała co oznacza, że prawdopodobieństwo uzyskania takiej różnicy pomiędzy grupami wyłącznie przez przypadek jest bardzo małe.

Wniosek: istnieje istotna statystycznie różnica pomiędzy cenami biletów płaconymi przez kobiety i mężczyzn.

## `normality.py`

Test normalności służy do sprawdzenia, czy analizowane dane mają rozkład normalny (są zgodne z krzywą Gaussa).

![normality](screenshots/normality.png)

Działanie testu na przykładzie `normality_Fare`:

Do przeprowadzenia testu wybrano kolumnę:
- `Fare`-cena biletu pasażera.

Celem analizy było sprawdzenie, czy wartości w kolumnie `Fare` mają rozkład normalny.

Działanie:
- Najpierw pobierane są wszystkie wartości z analizowanej kolumny.
- Następnie wykonywany jest test Shapiro-Wilka.
- Obliczana jest statystyka testowa oraz `p_value`.
- Obliczenia wykonywane są przy użyciu funkcji `shapiro()` z biblioteki `SciPy`.

Otrzymane wartości:
"statistic": 0.5218913010396559,
"p_value": 1.0840444395829658e-43

Wartość statistic jest znacznie mniejsza od 1 co oznacza wyraźne odstępstwo od rozkładu normalnego.
Wartość p_value jest bardzo mała co oznacza, że prawdopodobieństwo uzyskania takiego wyniku przy założeniu normalności rozkładu jest bardzo małe.

Dane w kolumnie `Fare` nie mają rozkładu normalnego.

## `t_test.py`

Test t-Studenta to metoda statystyczna, która pozwala sprawdzić, czy dwie grupy lub dwie sytuacje znacząco różnią się od siebie pod kątem badanej cechy. Pozwala sprawdzić, czy różnice pomiędzy grupami są istotne statystycznie, czy też wynikają z przypadku.

Do przeprowadzenia testu wymagane są:
- kolumna kategoryczna zawierająca dokładnie dwie grupy,
- kolumna liczbowa,
- spełnione założenie normalności rozkładu analizowanych danych.

Działanie:
- Dane są dzielone na dwie grupy na podstawie kolumny kategorycznej.
- Następnie obliczane są średnie wartości w obu grupach.
- Wyznaczana jest statystyka t określająca wielkość różnicy pomiędzy grupami.
- Następnie obliczane jest p_value.
- Obliczenia wykonywane są przy użyciu funkcji `ttest_ind()` z biblioteki `SciPy`.

Interpretacja wyników:
- wysoka bezwzględna wartość t_statistic oznacza większą różnicę pomiędzy średnimi grup,
- niska wartość p_value oznacza istotną statystycznie różnicę pomiędzy grupami.


## `statistical_engine.py`

Klasa `StatisticalTestEngine` odpowiada za automatyczny dobór oraz wykonywanie testów statystycznych na podstawie przekazanego zbioru danych bez konieczności ręcznego wybierania odpowiednich testów statystycznych.

Działanie:
- Po utworzeniu obiektu przekazywany jest zbiór danych w postaci obiektu DataFrame.
- Klasa analizuje typy kolumn znajdujących się w zbiorze danych.
  - Dla kolumn liczbowych wykonywane są testy normalności rozkładu.
  - Dla odpowiednich par kolumn liczbowych obliczane są korelacje oraz kowariancje.
  - Dla kolumn kategorycznych i liczbowych klasa może automatycznie wybrać test anova, t-test lub test Manna-Whitneya.
  - Dla par kolumn kategorycznych wykonywany jest test Chi-kwadrat.
- Wyniki wszystkich wykonanych analiz są zapisywane w jednym słowniku i zwracane jako odpowiedź api.

System analizuje strukturę danych i automatycznie uruchamia testy, które mają sens dla danego zestawu danych.


