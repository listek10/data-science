from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "Titanic-Dataset.csv"


def load_data(path=DATA_PATH):
    return pd.read_csv(path)
