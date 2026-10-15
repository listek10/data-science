# Raport zależności – przyklad.ipynb

Data: 2026-10-12 16:36  
Python: 3.11.17

## Wykryte importy

| Import | Pakiet | Status | Wersja |
|---|---|---|---|
| `bs4` | beautifulsoup4 | ✅ zainstalowany | 4.15.0 |
| `matplotlib` | matplotlib | ✅ zainstalowany | 3.11.2 |
| `numpy` | numpy | ✅ zainstalowany | 2.4.6 |
| `pandas` | pandas | ✅ zainstalowany | 3.0.6 |
| `requests` | requests | ✅ zainstalowany | 2.34.2 |
| `seaborn` | seaborn | ❌ brak | - |
| `sklearn` | scikit-learn | ✅ zainstalowany | 1.9.1 |
| `tensorflow` | tensorflow | ❌ brak | - |
| `xgboost` | xgboost | ✅ zainstalowany | 3.2.0 |

## Brakujące pakiety

```bash
pip install seaborn tensorflow
```

## Konflikty wersji

- Notebook instaluje pandas==2.2.0, a w środowisku jest 3.0.6

## Sugerowane lżejsze alternatywy

- **beautifulsoup4** → selectolax – wielokrotnie szybsze parsowanie HTML
- **pandas** → polars – szybszy i zużywa mniej RAM przy dużych danych
- **requests** → urllib z biblioteki standardowej przy prostych zapytaniach
- **seaborn** → matplotlib – seaborn i tak z niego korzysta
- **tensorflow** → onnxruntime lub tflite-runtime do samej predykcji; scikit-learn dla prostych modeli
- **xgboost** → lightgbm – zwykle szybszy trening i mniej pamięci

## Minimalny requirements.txt

```
beautifulsoup4==4.15.0
matplotlib==3.11.2
numpy==2.4.6
pandas==3.0.6
requests==2.34.2
scikit-learn==1.9.1
xgboost==3.2.0
seaborn  # nie zainstalowany - brak wersji
tensorflow  # nie zainstalowany - brak wersji
```
