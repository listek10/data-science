import pandas as pd

FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
TARGET = "Survived"


def preprocess(df):
    X = df[FEATURES].copy()
    X["Age"] = X["Age"].fillna(X["Age"].median())
    X["Embarked"] = X["Embarked"].fillna(X["Embarked"].mode()[0])
    X["Sex"] = X["Sex"].map({"male": 0, "female": 1})
    X = pd.get_dummies(X, columns=["Embarked"], drop_first=True, dtype=int)
    return X, df[TARGET]
