from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from src.data import load_data
from src.preprocessing import preprocess

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "baseline_logreg.joblib"


def split_data():
    X, y = preprocess(load_data())
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def train():
    X_train, X_test, y_train, y_test = split_data()

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    return model, accuracy


if __name__ == "__main__":
    _, accuracy = train()
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Model zapisany: {MODEL_PATH}")
