# Przykładowy skrypt do testu dashboardu - celowo ma kilka "nieekologicznych" rzeczy
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

df = pd.read_csv("dane.csv")

# liczenie wartości wiersz po wierszu
totals = []
for _, row in df.iterrows():
    totals.append(row["cena"] * row["ilosc"])
df["suma"] = totals

df["kategoria_dl"] = df["kategoria"].apply(lambda x: len(x))

# ręczne mnożenie macierzy
a = np.random.rand(100, 100)
b = np.random.rand(100, 100)
wynik = np.zeros((100, 100))
for i in range(100):
    for j in range(100):
        for k in range(100):
            wynik[i, j] += a[i, k] * b[k, j]

X = df[["cena", "ilosc"]]
y = df["kupil"]

params = {"n_estimators": [100, 200, 500], "max_depth": [3, 5, 10, None]}
search = GridSearchCV(RandomForestClassifier(), params, cv=5)
search.fit(X, y)
print(search.best_params_)
