# Uruchomienie:  streamlit run zadanie_20.py
import ast
import io
import json
import re
import textwrap
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import plotly.graph_objects as go
import streamlit as st

# średni pobór mocy [kW] i względna szybkość (1.0 = laptop)
DEVICES = {
    "laptop": {"kw": 0.05, "speed": 1.0},
    "desktop": {"kw": 0.20, "speed": 0.6},
    "cloud GPU": {"kw": 0.40, "speed": 0.6},
}

# emisja CO2 [g/kWh] - wartości przybliżone
REGIONS = {
    "Polska": 650,
    "Niemcy": 380,
    "USA": 370,
    "UE (średnia)": 250,
    "Francja": 55,
    "Norwegia": 30,
}

CAR_G_PER_KM = 150

# wzorce w kodzie: (nazwa, regex, minuty na wystąpienie, sugestia, oszczędność %)
PATTERNS = [
    ("iterrows()", r"\.iterrows\(", 0.5,
     "Zamień iterrows() na operacje na kolumnach (wektoryzacja Pandas).", 30),
    (".apply()", r"\.apply\(", 0.2,
     "Sprawdź, czy .apply() da się zastąpić wbudowaną operacją Pandas/NumPy.", 10),
    ("trening modelu (.fit)", r"\.fit\(", 1.0,
     "Trenuj najpierw na próbce danych i zapisuj model (joblib), zamiast trenować go od nowa przy każdym uruchomieniu.", 15),
    ("GridSearchCV", r"GridSearchCV", 5.0,
     "Zamiast GridSearchCV użyj RandomizedSearchCV albo mniejszej siatki parametrów.", 40),
    ("read_csv bez usecols", r"read_csv\((?![^)]*usecols)", 0.1,
     "Wczytuj tylko potrzebne kolumny (usecols) i ustawiaj dtype – mniej RAM i szybsze wczytywanie.", 5),
]

DL_LIBS = {"torch", "tensorflow", "keras"}


# ---------- analiza kodu ----------
def load_code(name, raw):
    text = raw.decode("utf-8")
    if name.endswith(".ipynb"):
        nb = json.loads(text)
        cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
        text = "\n\n".join(cells)
    # bez magic commands i poleceń shella z notebooków
    return "\n".join(l for l in text.splitlines() if not l.strip().startswith(("%", "!")))


def count_loops(code):
    """Zwraca (liczba pętli, liczba pętli zagnieżdżonych)."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return len(re.findall(r"^\s*(for|while) ", code, re.M)), 0

    loops = nested = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.For, ast.While)):
            loops += 1
            inner = [n for n in ast.walk(node) if isinstance(n, (ast.For, ast.While)) and n is not node]
            if inner:
                nested += 1
    return loops, nested


def find_imports(code):
    return set(re.findall(r"^\s*(?:import|from)\s+(\w+)", code, re.M))


def analyze_code(code):
    findings = []
    minutes = 0.5  # sam start skryptu, importy
    # wzorce szukamy poza liniami z importami
    body = "\n".join(l for l in code.splitlines() if not re.match(r"\s*(import|from)\s", l))

    for name, pattern, cost, tip, saving in PATTERNS:
        count = len(re.findall(pattern, body))
        if count:
            findings.append({"name": name, "count": count, "minutes": count * cost,
                             "tip": tip, "saving": saving})
            minutes += count * cost

    loops, nested = count_loops(code)
    if loops:
        minutes += loops * 0.05
    if nested:
        findings.append({"name": "zagnieżdżone pętle", "count": nested, "minutes": nested * 1.0,
                         "tip": "Zagnieżdżone pętle zastąp operacjami NumPy (np. mnożenie macierzy, broadcasting).",
                         "saving": 25})
        minutes += nested * 1.0

    imports = find_imports(code)
    dl_minutes = 0
    if imports & DL_LIBS:
        epochs = sum(int(e) for e in re.findall(r"epochs\s*=\s*(\d+)", code)) or 10
        dl_minutes = 5 + epochs * 0.5
        findings.append({"name": "deep learning", "count": epochs, "minutes": dl_minutes,
                         "tip": f"Sieć trenowana przez {epochs} epok – dodaj EarlyStopping i sprawdź, "
                                "czy prostszy model (np. z scikit-learn) nie wystarczy.",
                         "saving": 35})

    ram_gb = 0.3 + 0.5 * bool(imports & {"pandas", "polars"}) + 3 * bool(imports & DL_LIBS)
    return {
        "lines": len([l for l in code.splitlines() if l.strip()]),
        "imports": sorted(imports),
        "loops": loops,
        "base_minutes": minutes,
        "dl_minutes": dl_minutes,
        "ram_gb": ram_gb,
        "findings": findings,
    }


# ---------- obliczenia śladu węglowego ----------
def footprint(analysis, device, region, runs, optimized=False):
    speed = DEVICES[device]["speed"]
    dl_speed = 0.2 if device == "cloud GPU" else speed  # GPU mocno przyspiesza tylko deep learning
    minutes = analysis["base_minutes"] * speed + analysis["dl_minutes"] * dl_speed

    if optimized:
        # każda sugestia zmniejsza czas o swój procent (efekty się nie sumują liniowo)
        for f in analysis["findings"]:
            minutes *= 1 - f["saving"] / 100

    kwh = minutes / 60 * DEVICES[device]["kw"] * runs
    co2 = kwh * REGIONS[region]
    return {"minutes": minutes, "kwh": kwh, "co2_g": co2, "car_km": co2 / CAR_G_PER_KM}


def green_score(analysis):
    score = 100
    for f in analysis["findings"]:
        score -= min(f["saving"] / 2 * f["count"], f["saving"])
    return max(0, round(score))


# ---------- wykresy ----------
def gauge(value, title, max_value, suffix="", good_is_low=True):
    green, yellow, red = "#c8ebd2", "#fdeab5", "#f6c9c9"
    steps_colors = [green, yellow, red] if good_is_low else [red, yellow, green]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix": suffix},
        title={"text": title},
        gauge={
            "axis": {"range": [0, max_value]},
            "bar": {"color": "#2a78d6"},
            "steps": [
                {"range": [0, max_value / 3], "color": steps_colors[0]},
                {"range": [max_value / 3, 2 * max_value / 3], "color": steps_colors[1]},
                {"range": [2 * max_value / 3, max_value], "color": steps_colors[2]},
            ],
        },
    ))
    fig.update_layout(height=260, margin=dict(t=60, b=10, l=30, r=30))
    return fig


def devices_chart(analysis, region, runs):
    names = list(DEVICES)
    now = [footprint(analysis, d, region, runs)["co2_g"] for d in names]
    opt = [footprint(analysis, d, region, runs, optimized=True)["co2_g"] for d in names]
    fig = go.Figure([
        go.Bar(name="Obecny kod", x=names, y=now, marker_color="#eb6834",
               text=[f"{v:.0f} g" for v in now], textposition="outside"),
        go.Bar(name="Po optymalizacji", x=names, y=opt, marker_color="#2a78d6",
               text=[f"{v:.0f} g" for v in opt], textposition="outside"),
    ])
    fig.update_layout(title=f"CO2 na miesiąc – urządzenia ({region})", yaxis_title="CO2 [g]",
                      barmode="group", height=380, legend=dict(orientation="h", y=-0.15))
    return fig


def regions_chart(analysis, device, runs, current_region):
    names = sorted(REGIONS, key=REGIONS.get, reverse=True)
    values = [footprint(analysis, device, r, runs)["co2_g"] for r in names]
    colors = ["#2a78d6" if r == current_region else "#b7b6b0" for r in names]
    fig = go.Figure(go.Bar(x=values, y=names, orientation="h", marker_color=colors,
                           text=[f"{v:.0f} g" for v in values], textposition="outside"))
    fig.update_layout(title=f"CO2 na miesiąc – regiony ({device})", xaxis_title="CO2 [g]",
                      yaxis=dict(autorange="reversed"), height=380)
    return fig


# ---------- eksport PDF ----------
def make_pdf(file_name, analysis, device, region, runs, now, opt, score):
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        # strona 1 - podsumowanie
        fig = plt.figure(figsize=(8.27, 11.69))  # A4
        lines = [
            ("Green Data Science – raport", 18, "bold"),
            (f"Plik: {file_name}", 11, "normal"),
            (f"Data: {datetime.now():%Y-%m-%d %H:%M}", 11, "normal"),
            ("", 11, "normal"),
            ("Ustawienia", 14, "bold"),
            (f"Urządzenie: {device}   Region: {region}   Uruchomień w miesiącu: {runs}", 11, "normal"),
            ("", 11, "normal"),
            ("Wyniki (miesięcznie)", 14, "bold"),
            (f"Szacowany czas jednego uruchomienia: {now['minutes']:.1f} min", 11, "normal"),
            (f"Zużycie energii: {now['kwh']:.3f} kWh", 11, "normal"),
            (f"Emisja CO2: {now['co2_g']:.0f} g  (≈ {now['car_km']:.1f} km jazdy autem)", 11, "normal"),
            (f"Po optymalizacji: {opt['co2_g']:.0f} g CO2  "
             f"(oszczędność {now['co2_g'] - opt['co2_g']:.0f} g)", 11, "normal"),
            (f"Green score: {score}/100", 11, "normal"),
            ("", 11, "normal"),
            ("Sugestie optymalizacji", 14, "bold"),
        ]
        for f in analysis["findings"]:
            lines.append((f"• {f['name']} ({f['count']}×): {f['tip']}", 10, "normal"))
        if not analysis["findings"]:
            lines.append(("Nie znaleziono problemów – kod wygląda na oszczędny.", 10, "normal"))

        y = 0.95
        for text, size, weight in lines:
            for part in textwrap.wrap(text, 90) or [""]:
                fig.text(0.07, y, part, fontsize=size, weight=weight)
                y -= 0.028 if size < 14 else 0.035
        pdf.savefig(fig)
        plt.close(fig)

        # strona 2 - wykresy
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.27, 11.69))
        names = list(DEVICES)
        x = range(len(names))
        v_now = [footprint(analysis, d, region, runs)["co2_g"] for d in names]
        v_opt = [footprint(analysis, d, region, runs, optimized=True)["co2_g"] for d in names]
        ax1.bar([i - 0.2 for i in x], v_now, width=0.4, color="#eb6834", label="Obecny kod")
        ax1.bar([i + 0.2 for i in x], v_opt, width=0.4, color="#2a78d6", label="Po optymalizacji")
        ax1.set_xticks(list(x))
        ax1.set_xticklabels(names)
        ax1.set_ylabel("CO2 [g / miesiąc]")
        ax1.set_title(f"Porównanie urządzeń ({region})")
        ax1.legend()

        reg = sorted(REGIONS, key=REGIONS.get, reverse=True)
        v_reg = [footprint(analysis, device, r, runs)["co2_g"] for r in reg]
        ax2.barh(reg, v_reg, color=["#2a78d6" if r == region else "#b7b6b0" for r in reg])
        ax2.invert_yaxis()
        ax2.set_xlabel("CO2 [g / miesiąc]")
        ax2.set_title(f"Porównanie regionów ({device})")

        for ax in (ax1, ax2):
            ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout(pad=3)
        pdf.savefig(fig)
        plt.close(fig)
    return buf.getvalue()


# ---------- interfejs ----------
st.set_page_config(page_title="Green DS Dashboard", page_icon="🌱", layout="wide")
st.title("🌱 Green Data Science Dashboard")
st.caption("Szacuje zużycie energii i ślad węglowy kodu na podstawie jego analizy (symulacja, nie pomiar).")

with st.sidebar:
    st.header("Plik z kodem")
    uploaded = st.file_uploader("Wgraj skrypt z VS Code (.py) lub notebook (.ipynb)", type=["py", "ipynb"])
    use_example = st.checkbox("Użyj przykładowego pliku", value=uploaded is None)

    st.header("Ustawienia")
    device = st.selectbox("Urządzenie", list(DEVICES))
    region = st.selectbox("Region", list(REGIONS))
    runs = st.slider("Ile razy w miesiącu uruchamiasz kod?", 1, 200, 30)

if uploaded is not None and not use_example:
    file_name, raw = uploaded.name, uploaded.getvalue()
elif use_example:
    example = Path(__file__).parent / "przyklad.py"
    file_name, raw = example.name, example.read_bytes()
else:
    st.info("Wgraj plik .py lub .ipynb w panelu po lewej.")
    st.stop()

try:
    code = load_code(file_name, raw)
except (UnicodeDecodeError, json.JSONDecodeError, KeyError):
    st.error("Nie udało się odczytać pliku.")
    st.stop()

analysis = analyze_code(code)
now = footprint(analysis, device, region, runs)
opt = footprint(analysis, device, region, runs, optimized=True)
score = green_score(analysis)

st.subheader(f"Analiza: {file_name}")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Linie kodu", analysis["lines"])
c2.metric("Pętle", analysis["loops"])
c3.metric("Czas uruchomienia (szac.)", f"{now['minutes']:.1f} min")
c4.metric("RAM (szac.)", f"{analysis['ram_gb']:.1f} GB")

g1, g2, g3 = st.columns(3)
g1.plotly_chart(gauge(now["co2_g"], "CO2 / miesiąc [g]", max(now["co2_g"] * 1.5, 500)), width="stretch")
g2.plotly_chart(gauge(now["kwh"], "Energia / miesiąc [kWh]", max(now["kwh"] * 1.5, 1)), width="stretch")
g3.plotly_chart(gauge(score, "Green score", 100, good_is_low=False), width="stretch")

st.markdown(f"To tyle CO2 co **{now['car_km']:.1f} km jazdy autem**. "
            f"Po optymalizacji: **{opt['co2_g']:.0f} g** "
            f"(mniej o {now['co2_g'] - opt['co2_g']:.0f} g miesięcznie).")

st.subheader("Porównanie z alternatywami")
ch1, ch2 = st.columns(2)
ch1.plotly_chart(devices_chart(analysis, region, runs), width="stretch")
ch2.plotly_chart(regions_chart(analysis, device, runs, region), width="stretch")

st.subheader("Sugestie optymalizacji")
if analysis["findings"]:
    for f in sorted(analysis["findings"], key=lambda f: -f["saving"]):
        st.markdown(f"- **{f['name']}** ({f['count']}×, ok. −{f['saving']}% czasu): {f['tip']}")
else:
    st.success("Nie znaleziono problemów – kod wygląda na oszczędny.")

with st.expander("Wykryte biblioteki"):
    st.write(", ".join(analysis["imports"]) or "brak")

st.download_button(
    "📄 Pobierz raport PDF",
    data=make_pdf(file_name, analysis, device, region, runs, now, opt, score),
    file_name=f"raport_green_{Path(file_name).stem}.pdf",
    mime="application/pdf",
)
