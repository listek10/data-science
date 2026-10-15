import json
import subprocess
import sys
from pathlib import Path

FOLDERS = [
    "data/raw",
    "data/processed",
    "notebooks",
    "src",
    "models",
    "reports/figures",
]

PACKAGES = ["numpy", "pandas", "matplotlib", "scikit-learn", "ipykernel"]

GITIGNORE = """\
# środowisko
venv/
.venv/
__pycache__/
*.pyc

# jupyter / vs code
.ipynb_checkpoints/
.vscode/

# dane i modele (zwykle za duże do gita)
data/raw/
data/processed/
models/

.DS_Store
.env
"""


def notebook_template(project_name):
    """Prosty szablon notebooka w formacie .ipynb (JSON)."""
    def md(text):
        return {"cell_type": "markdown", "metadata": {}, "source": text}

    def code(text):
        return {"cell_type": "code", "metadata": {}, "execution_count": None,
                "outputs": [], "source": text}

    cells = [
        md(f"# {project_name}\n\nCel analizy: ..."),
        md("## 1. Importy i ustawienia"),
        code("import numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\n\n"
             "SEED = 42\nnp.random.seed(SEED)"),
        md("## 2. Wczytanie danych"),
        code("# df = pd.read_csv('../data/raw/dane.csv')"),
        md("## 3. Eksploracja danych"),
        code(""),
        md("## 4. Wnioski"),
    ]
    return {
        "cells": cells,
        "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3",
                                    "language": "python"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def setup_project(project_name, install_packages=True):
    """Tworzy gotowy szkielet projektu Data Science i zwraca listę kroków."""
    root = Path(project_name).resolve()
    report = []

    def step(name, func):
        try:
            func()
            report.append(("OK", name))
        except Exception as e:
            report.append(("BŁĄD", f"{name} -> {e}"))

    def run(cmd):
        subprocess.run(cmd, cwd=root, check=True, capture_output=True, text=True)

    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Folder {root} już istnieje i nie jest pusty")

    # 1. struktura folderów
    def make_folders():
        for folder in FOLDERS:
            (root / folder).mkdir(parents=True, exist_ok=True)
            (root / folder / ".gitkeep").touch()
    step("Struktura folderów", make_folders)

    # 2. README i .gitignore
    def make_readme():
        tree = "\n".join(f"    {f}/" for f in FOLDERS)
        text = (
            f"# {project_name}\n\n"
            "Projekt Data Science.\n\n"
            "## Struktura\n\n"
            f"{tree}\n\n"
            "## Uruchomienie\n\n"
            "```bash\n"
            "source venv/bin/activate\n"
            "pip install -r requirements.txt\n"
            "```\n"
        )
        (root / "README.md").write_text(text, encoding="utf-8")
    step("README.md", make_readme)
    step(".gitignore", lambda: (root / ".gitignore").write_text(GITIGNORE, encoding="utf-8"))

    # 3. szablon notebooka
    def make_notebook():
        nb_path = root / "notebooks" / "01_eksploracja.ipynb"
        nb_path.write_text(json.dumps(notebook_template(project_name), indent=1, ensure_ascii=False),
                           encoding="utf-8")
    step("Szablon notebooka (notebooks/01_eksploracja.ipynb)", make_notebook)

    # 4. venv + pakiety
    step("Wirtualne środowisko (venv)", lambda: run([sys.executable, "-m", "venv", "venv"]))
    pip = root / "venv" / "bin" / "pip"
    if install_packages:
        step(f"Instalacja pakietów: {', '.join(PACKAGES)}", lambda: run([str(pip), "install", *PACKAGES]))

    def freeze():
        out = subprocess.run([str(pip), "freeze"], capture_output=True, text=True, check=True)
        (root / "requirements.txt").write_text(out.stdout, encoding="utf-8")
    step("requirements.txt", freeze)

    # 5. git
    step("git init", lambda: run(["git", "init"]))

    def first_commit():
        run(["git", "add", "."])
        run(["git", "commit", "-m", "Początkowa struktura projektu"])
    step("Pierwszy commit", first_commit)

    # raport
    print(f"\nRaport z tworzenia projektu: {root}")
    for status, name in report:
        print(f"  [{status}] {name}")
    print(f"\nGotowe: {sum(s == 'OK' for s, _ in report)}/{len(report)} kroków")
    return report


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Użycie: python zadanie_13.py <nazwa_projektu>")
        sys.exit(1)
    setup_project(sys.argv[1])
