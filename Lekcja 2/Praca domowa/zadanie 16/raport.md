# Energy Benchmark – raport

Procesor: 32 wątków, założona moc 65 W. Każda operacja uruchomiona 3 razy (średnia).
Energia = czas CPU (wszystkie wątki) × moc jednego wątku – to szacunek, nie pomiar z gniazdka.

## Wyniki

| Grupa | Metoda | Czas [s] | Czas CPU [s] | RAM [MB] | Energia [mWh] |
|---|---|---|---|---|---|
| Suma kwadratów (1 mln) | Pętla Python | 0.0244 | 0.0244 | 0.0 | 0.01378 |
| Suma kwadratów (1 mln) | NumPy | 0.0010 | 0.0010 | 7.6 | 0.00057 |
| Mnożenie macierzy 120×120 | Pętle Python | 0.0515 | 0.0515 | 0.4 | 0.02905 |
| Mnożenie macierzy 120×120 | NumPy | 0.0001 | 0.0001 | 0.1 | 0.00006 |
| Nowa kolumna (20 tys. wierszy) | iterrows | 0.1906 | 0.3779 | 1.5 | 0.21325 |
| Nowa kolumna (20 tys. wierszy) | Wektoryzacja Pandas | 0.0001 | 0.0001 | 0.2 | 0.00008 |
| Średnia w grupach | iterrows + słownik | 0.1941 | 0.1940 | 1.4 | 0.10945 |
| Średnia w grupach | groupby | 0.0009 | 0.0009 | 0.3 | 0.00054 |
| Wykres 10 tys. punktów | Matplotlib (PNG) | 0.0338 | 0.0338 | 1.1 | 0.01905 |
| Wykres 10 tys. punktów | Plotly (HTML) | 0.0292 | 0.0292 | 0.9 | 0.01648 |
| Trening modelu (20 tys. próbek) | sklearn GradientBoosting | 7.7838 | 7.7813 | 3.6 | 4.39049 |
| Trening modelu (20 tys. próbek) | XGBoost | 0.2618 | 5.5437 | 0.2 | 3.12798 |

## Co jest bardziej green?

- **Suma kwadratów (1 mln)**: wybierz **NumPy** – zużywa ok. 24.1× mniej energii niż Pętla Python.
- **Mnożenie macierzy 120×120**: wybierz **NumPy** – zużywa ok. 487.1× mniej energii niż Pętle Python.
- **Nowa kolumna (20 tys. wierszy)**: wybierz **Wektoryzacja Pandas** – zużywa ok. 2800.9× mniej energii niż iterrows.
- **Średnia w grupach**: wybierz **groupby** – zużywa ok. 204.4× mniej energii niż iterrows + słownik.
- **Wykres 10 tys. punktów**: wybierz **Plotly (HTML)** – zużywa ok. 1.2× mniej energii niż Matplotlib (PNG).
- **Trening modelu (20 tys. próbek)**: wybierz **XGBoost** – zużywa ok. 1.4× mniej energii niż sklearn GradientBoosting.

## Rekomendacje

- Unikaj pętli Pythona na dużych danych – NumPy i Pandas liczą w C i są wielokrotnie oszczędniejsze.
- `iterrows()` to najgorszy sposób na przechodzenie po DataFrame – używaj operacji na kolumnach i `groupby`.
- Przy treningu patrz na czas CPU, a nie tylko na czas zegarowy – biblioteka używająca wielu wątków kończy szybciej, ale zużywa energię na wszystkich rdzeniach.
- RAM mierzony przez tracemalloc (pamięć Pythona i NumPy) – nie obejmuje pamięci natywnej np. XGBoost.
- Przy wykresach Plotly część pracy przenosi się do przeglądarki, więc pomiar w Pythonie nie pokazuje całego kosztu.
