import io
import time
import tracemalloc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import psutil
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

# Szacunek energii: czas pracy CPU (suma wszystkich wątków) × moc jednego wątku.
# Zakładamy procesor o mocy 65 W podzielonej na wszystkie wątki logiczne.
CPU_POWER_W = 65
WATTS_PER_THREAD = CPU_POWER_W / psutil.cpu_count(logical=True)
REPEATS = 3

rng = np.random.default_rng(42)


# ---------- dane testowe ----------
N = 1_000_000
numbers = list(range(N))
numbers_np = np.arange(N, dtype=np.float64)

M = 120
mat_a = rng.random((M, M))
mat_b = rng.random((M, M))
mat_a_list, mat_b_list = mat_a.tolist(), mat_b.tolist()

df = pd.DataFrame({
    "cena": rng.uniform(10, 500, 20_000),
    "ilosc": rng.integers(1, 20, 20_000),
    "kategoria": rng.choice(["A", "B", "C", "D"], 20_000),
})

plot_x = np.arange(10_000)
plot_y = rng.normal(size=10_000).cumsum()

X, y = make_classification(n_samples=20_000, n_features=20, random_state=42)


# ---------- operacje ----------
def squares_loop():
    total = 0
    for n in numbers:
        total += n * n
    return total


def squares_numpy():
    return np.sum(numbers_np ** 2)


def matmul_loop():
    result = [[0.0] * M for _ in range(M)]
    for i in range(M):
        for j in range(M):
            s = 0.0
            for k in range(M):
                s += mat_a_list[i][k] * mat_b_list[k][j]
            result[i][j] = s
    return result


def matmul_numpy():
    return mat_a @ mat_b


def column_iterrows():
    values = []
    for _, row in df.iterrows():
        values.append(row["cena"] * row["ilosc"])
    return values


def column_vectorized():
    return df["cena"] * df["ilosc"]


def groupby_loop():
    sums, counts = {}, {}
    for _, row in df.iterrows():
        k = row["kategoria"]
        sums[k] = sums.get(k, 0) + row["cena"]
        counts[k] = counts.get(k, 0) + 1
    return {k: sums[k] / counts[k] for k in sums}


def groupby_pandas():
    return df.groupby("kategoria")["cena"].mean()


def plot_matplotlib():
    fig, ax = plt.subplots()
    ax.plot(plot_x, plot_y)
    fig.savefig(io.BytesIO(), format="png")
    plt.close(fig)


def plot_plotly():
    # plotly rysuje w przeglądarce, więc mierzymy zbudowanie wykresu i wygenerowanie HTML
    fig = go.Figure(go.Scatter(x=plot_x, y=plot_y, mode="lines"))
    fig.to_html(include_plotlyjs=False)


def train_sklearn():
    GradientBoostingClassifier(n_estimators=100, random_state=42).fit(X, y)


def train_xgboost():
    XGBClassifier(n_estimators=100, random_state=42).fit(X, y)


# (grupa, nazwa, funkcja)
BENCHMARKS = [
    ("Suma kwadratów (1 mln)", "Pętla Python", squares_loop),
    ("Suma kwadratów (1 mln)", "NumPy", squares_numpy),
    ("Mnożenie macierzy 120×120", "Pętle Python", matmul_loop),
    ("Mnożenie macierzy 120×120", "NumPy", matmul_numpy),
    ("Nowa kolumna (20 tys. wierszy)", "iterrows", column_iterrows),
    ("Nowa kolumna (20 tys. wierszy)", "Wektoryzacja Pandas", column_vectorized),
    ("Średnia w grupach", "iterrows + słownik", groupby_loop),
    ("Średnia w grupach", "groupby", groupby_pandas),
    ("Wykres 10 tys. punktów", "Matplotlib (PNG)", plot_matplotlib),
    ("Wykres 10 tys. punktów", "Plotly (HTML)", plot_plotly),
    ("Trening modelu (20 tys. próbek)", "sklearn GradientBoosting", train_sklearn),
    ("Trening modelu (20 tys. próbek)", "XGBoost", train_xgboost),
]


# ---------- pomiar ----------
def peak_ram_during(func):
    """Maksymalna ilość pamięci zaalokowanej w trakcie działania funkcji [MB]."""
    tracemalloc.start()
    func()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 1024**2


def measure(func):
    times, cpu_times = [], []
    for _ in range(REPEATS):
        cpu_start = time.process_time()  # czas CPU całego procesu (wszystkie wątki)
        t_start = time.perf_counter()
        func()
        times.append(time.perf_counter() - t_start)
        cpu_times.append(time.process_time() - cpu_start)

    cpu_time = float(np.mean(cpu_times))
    return {
        "czas_s": float(np.mean(times)),
        "czas_cpu_s": cpu_time,
        "ram_mb": peak_ram_during(func),
        "energia_mwh": cpu_time * WATTS_PER_THREAD / 3.6,  # J -> mWh
    }


def run_benchmarks():
    rows = []
    for group, name, func in BENCHMARKS:
        print(f"{group:32} | {name:25} ...", end=" ", flush=True)
        result = measure(func)
        print(f"{result['czas_s']:.4f} s")
        rows.append({"grupa": group, "metoda": name, **result})
    return pd.DataFrame(rows)


# ---------- wykres i raport ----------
def plot_results(results, filename="benchmark.png"):
    groups = list(dict.fromkeys(results["grupa"]))
    fig, ax = plt.subplots(figsize=(11, 7))

    bar_h = 0.38
    for i, group in enumerate(groups):
        part = results[results["grupa"] == group].sort_values("energia_mwh", ascending=False)
        for j, (_, row) in enumerate(part.iterrows()):
            # na niebiesko wersja bardziej "green", na pomarańczowo mniej
            color = "#2a78d6" if j == 1 else "#eb6834"
            y_pos = i + (j - 0.5) * (bar_h + 0.02)
            ax.barh(y_pos, row["energia_mwh"], height=bar_h, color=color)
            ax.text(row["energia_mwh"] * 1.15, y_pos, f"{row['metoda']}  ({row['energia_mwh']:.3g} mWh)",
                    va="center", fontsize=9)

    ax.set_xscale("log")
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels(groups)
    ax.invert_yaxis()
    ax.set_xlim(right=results["energia_mwh"].max() * 200)
    ax.set_xlabel("Szacowane zużycie energii [mWh] – skala logarytmiczna")
    ax.set_title("Porównanie zużycia energii operacji Data Science")
    ax.spines[["top", "right"]].set_visible(False)

    handles = [plt.Rectangle((0, 0), 1, 1, color="#2a78d6"), plt.Rectangle((0, 0), 1, 1, color="#eb6834")]
    ax.legend(handles, ["bardziej green", "mniej green"], loc="upper right")

    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    print(f"Wykres zapisany: {filename}")


def save_report(results, filename="raport.md"):
    lines = [
        "# Energy Benchmark – raport",
        "",
        f"Procesor: {psutil.cpu_count(logical=True)} wątków, założona moc {CPU_POWER_W} W. "
        f"Każda operacja uruchomiona {REPEATS} razy (średnia).",
        "Energia = czas CPU (wszystkie wątki) × moc jednego wątku – to szacunek, nie pomiar z gniazdka.",
        "",
        "## Wyniki",
        "",
        "| Grupa | Metoda | Czas [s] | Czas CPU [s] | RAM [MB] | Energia [mWh] |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in results.iterrows():
        lines.append(f"| {r['grupa']} | {r['metoda']} | {r['czas_s']:.4f} | {r['czas_cpu_s']:.4f} "
                     f"| {r['ram_mb']:.1f} | {r['energia_mwh']:.5f} |")

    lines += ["", "## Co jest bardziej green?", ""]
    for group, part in results.groupby("grupa", sort=False):
        best = part.loc[part["energia_mwh"].idxmin()]
        worst = part.loc[part["energia_mwh"].idxmax()]
        ratio = worst["energia_mwh"] / best["energia_mwh"]
        lines.append(f"- **{group}**: wybierz **{best['metoda']}** – zużywa ok. {ratio:.1f}× mniej energii "
                     f"niż {worst['metoda']}.")

    lines += [
        "",
        "## Rekomendacje",
        "",
        "- Unikaj pętli Pythona na dużych danych – NumPy i Pandas liczą w C i są wielokrotnie oszczędniejsze.",
        "- `iterrows()` to najgorszy sposób na przechodzenie po DataFrame – używaj operacji na kolumnach i `groupby`.",
        "- Przy treningu patrz na czas CPU, a nie tylko na czas zegarowy – biblioteka używająca wielu wątków "
        "kończy szybciej, ale zużywa energię na wszystkich rdzeniach.",
        "- RAM mierzony przez tracemalloc (pamięć Pythona i NumPy) – nie obejmuje pamięci natywnej np. XGBoost.",
        "- Przy wykresach Plotly część pracy przenosi się do przeglądarki, więc pomiar w Pythonie nie pokazuje całego kosztu.",
    ]
    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Raport zapisany: {filename}")


if __name__ == "__main__":
    results = run_benchmarks()
    results.to_csv("wyniki.csv", index=False)
    plot_results(results)
    save_report(results)
