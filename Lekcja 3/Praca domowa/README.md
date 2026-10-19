# Titanic – projekt ML z Git Flow

Projekt przewidujący przeżycie pasażerów Titanica (klasyfikacja binarna).

## Struktura

```
data/raw/        surowe dane (nie są w repozytorium)
data/processed/  dane przetworzone
notebooks/       notebooki z analizą
src/             kod źródłowy
models/          wytrenowane modele (nie są w repozytorium)
```

## Uruchomienie

1. Pobierz `Titanic-Dataset.csv` (np. z Kaggle) do `data/raw/`.
2. Zainstaluj zależności: `pip install -r requirements.txt`.

## Git Flow

- `main` – wersje wydane (tagi `v*`)
- `develop` – bieżąca praca
- `feature-*` – nowe funkcjonalności, łączone do `develop` przez Pull Request
