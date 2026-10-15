import argparse
import ast
import json
import re
import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version

# pkg_resources zostało usunięte z nowych wersji setuptools,
# dlatego do sprawdzania pakietów używam wbudowanego importlib.metadata

# nazwy importów, które różnią się od nazw pakietów w pip
IMPORT_TO_PIP = {
    "sklearn": "scikit-learn",
    "cv2": "opencv-python",
    "bs4": "beautifulsoup4",
    "PIL": "pillow",
    "yaml": "pyyaml",
    "dotenv": "python-dotenv",
}

# lżejsze / szybsze alternatywy
ALTERNATIVES = {
    "pandas": "polars – szybszy i zużywa mniej RAM przy dużych danych",
    "tensorflow": "onnxruntime lub tflite-runtime do samej predykcji; scikit-learn dla prostych modeli",
    "torch": "onnxruntime do samej predykcji; scikit-learn dla prostych modeli",
    "keras": "scikit-learn (MLPClassifier) dla małych sieci",
    "xgboost": "lightgbm – zwykle szybszy trening i mniej pamięci",
    "seaborn": "matplotlib – seaborn i tak z niego korzysta",
    "plotly": "matplotlib, jeśli wystarczą statyczne wykresy",
    "opencv-python": "opencv-python-headless (bez GUI) albo pillow do prostych operacji",
    "beautifulsoup4": "selectolax – wielokrotnie szybsze parsowanie HTML",
    "nltk": "spacy z małym modelem lub proste wyrażenia regularne",
    "requests": "urllib z biblioteki standardowej przy prostych zapytaniach",
}


def read_code(path):
    """Zwraca listę fragmentów kodu: komórki z .ipynb albo cały plik .py."""
    if path.suffix == ".ipynb":
        nb = json.loads(path.read_text(encoding="utf-8"))
        return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    return [path.read_text(encoding="utf-8")]


def find_pip_pins(cells):
    """Szuka w notebooku linii typu '!pip install pandas==1.5.0'."""
    pins = {}
    for cell in cells:
        for line in cell.splitlines():
            line = line.strip()
            if re.match(r"^[!%]pip install", line):
                for part in line.split()[2:]:
                    if "==" in part:
                        name, ver = part.split("==", 1)
                        pins[name.lower()] = ver
    return pins


def find_imports(cells):
    imports = set()
    for cell in cells:
        # usuwamy magic commands i polecenia shella - ast ich nie rozumie
        lines = [l for l in cell.splitlines() if not l.strip().startswith(("%", "!"))]
        try:
            tree = ast.parse("\n".join(lines))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                imports.add(node.module.split(".")[0])
    # pomijamy bibliotekę standardową (os, json, ...)
    return sorted(i for i in imports if i not in sys.stdlib_module_names)


def import_to_package(name):
    dists = metadata.packages_distributions().get(name)
    if dists:
        return dists[0]
    return IMPORT_TO_PIP.get(name, name)


def installed_version(package):
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def find_conflicts(packages):
    """Sprawdza, czy zależności używanych pakietów mają zgodne wersje."""
    conflicts = []
    checked = set()
    to_check = list(packages)
    while to_check:
        pkg = to_check.pop()
        if pkg.lower() in checked or installed_version(pkg) is None:
            continue
        checked.add(pkg.lower())
        for req_text in metadata.requires(pkg) or []:
            req = Requirement(req_text)
            if req.marker and not req.marker.evaluate({"extra": ""}):
                continue  # zależność opcjonalna albo dla innego systemu
            dep_version = installed_version(req.name)
            if dep_version is None:
                conflicts.append(f"{pkg} wymaga {req}, ale {req.name} nie jest zainstalowany")
            elif req.specifier and Version(dep_version) not in req.specifier:
                conflicts.append(f"{pkg} wymaga {req}, a zainstalowano {req.name} {dep_version}")
            else:
                to_check.append(req.name)
    return conflicts


def analyze(file_path):
    path = Path(file_path).resolve()
    cells = read_code(path)

    rows = []
    for imp in find_imports(cells):
        package = import_to_package(imp)
        rows.append({"import": imp, "package": package, "version": installed_version(package)})

    installed = [r for r in rows if r["version"]]
    missing = [r for r in rows if not r["version"]]

    # konflikty: wersje przypięte w notebooku vs zainstalowane
    conflicts = []
    for name, wanted in find_pip_pins(cells).items():
        have = installed_version(name)
        if have and have != wanted:
            conflicts.append(f"Notebook instaluje {name}=={wanted}, a w środowisku jest {have}")
    conflicts += find_conflicts([r["package"] for r in installed])

    # minimalny requirements.txt
    req_file = path.parent / "requirements.txt"
    req_lines = [f"{r['package']}=={r['version']}" for r in installed]
    req_lines += [f"{r['package']}  # nie zainstalowany - brak wersji" for r in missing]
    req_file.write_text("\n".join(req_lines) + "\n", encoding="utf-8")

    # raport markdown
    md = [
        f"# Raport zależności – {path.name}",
        "",
        f"Data: {datetime.now():%Y-%m-%d %H:%M}  ",
        f"Python: {sys.version.split()[0]}",
        "",
        "## Wykryte importy",
        "",
        "| Import | Pakiet | Status | Wersja |",
        "|---|---|---|---|",
    ]
    for r in rows:
        status = "✅ zainstalowany" if r["version"] else "❌ brak"
        md.append(f"| `{r['import']}` | {r['package']} | {status} | {r['version'] or '-'} |")

    md += ["", "## Brakujące pakiety", ""]
    if missing:
        md.append("```bash")
        md.append("pip install " + " ".join(r["package"] for r in missing))
        md.append("```")
    else:
        md.append("Wszystkie pakiety są zainstalowane.")

    md += ["", "## Konflikty wersji", ""]
    md += [f"- {c}" for c in conflicts] or ["Nie znaleziono konfliktów."]

    md += ["", "## Sugerowane lżejsze alternatywy", ""]
    tips = [f"- **{r['package']}** → {ALTERNATIVES[r['package']]}" for r in rows if r["package"] in ALTERNATIVES]
    md += tips or ["Brak sugestii."]

    md += ["", "## Minimalny requirements.txt", "", "```", *req_lines, "```", ""]

    report_file = path.parent / f"raport_{path.stem}.md"
    report_file.write_text("\n".join(md), encoding="utf-8")

    print(f"Importy: {len(rows)}, zainstalowane: {len(installed)}, brakujące: {len(missing)}, "
          f"konflikty: {len(conflicts)}")
    print(f"Zapisano: {req_file.name}, {report_file.name}")
    return report_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analiza zależności notebooka / skryptu")
    parser.add_argument("file", help="plik .ipynb lub .py")
    args = parser.parse_args()
    analyze(args.file)
