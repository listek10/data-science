import argparse
import base64
import io
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import psutil


class ResourceMonitor(threading.Thread):
    """Wątek, który co `interval` sekund zapisuje zużycie CPU i RAM."""

    def __init__(self, pid, ram_limit=80, interval=0.5):
        super().__init__(daemon=True)
        self.process = psutil.Process(pid)
        self.ram_limit = ram_limit
        self.interval = interval
        self.samples = []   # (czas [s], CPU %, RAM systemu %, RAM procesu MB)
        self.warnings = []
        self._stop_event = threading.Event()

    def process_ram_mb(self):
        # RAM procesu + jego dzieci (np. kernel Jupytera)
        total = 0
        try:
            procs = [self.process] + self.process.children(recursive=True)
        except psutil.NoSuchProcess:
            return 0
        for p in procs:
            try:
                total += p.memory_info().rss
            except psutil.NoSuchProcess:
                pass
        return total / 1024**2

    def run(self):
        start = time.time()
        over_limit = False
        psutil.cpu_percent()  # pierwszy odczyt zawsze zwraca 0
        while not self._stop_event.is_set():
            t = time.time() - start
            cpu = psutil.cpu_percent()
            ram = psutil.virtual_memory().percent
            self.samples.append((t, cpu, ram, self.process_ram_mb()))

            # ostrzeżenie tylko w momencie przekroczenia progu
            if ram > self.ram_limit and not over_limit:
                msg = f"[{t:.1f} s] UWAGA: RAM {ram:.1f}% > {self.ram_limit}%"
                print(msg)
                self.warnings.append(msg)
            over_limit = ram > self.ram_limit

            time.sleep(self.interval)

    def stop(self):
        self._stop_event.set()


def build_command(path):
    if path.suffix == ".ipynb":
        out_name = f"{path.stem}_wykonany.ipynb"
        return [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook",
                "--execute", str(path), "--output", out_name]
    return [sys.executable, str(path)]


def make_plot(monitor):
    t = [s[0] for s in monitor.samples]
    cpu = [s[1] for s in monitor.samples]
    ram = [s[2] for s in monitor.samples]
    proc_ram = [s[3] for s in monitor.samples]

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    axes[0].plot(t, cpu, color="#2a78d6", linewidth=2)
    axes[0].set_ylabel("CPU [%]")
    axes[0].set_title("Obciążenie CPU (cały system)")

    axes[1].plot(t, ram, color="#eb6834", linewidth=2)
    axes[1].axhline(monitor.ram_limit, color="#d03b3b", linestyle="--", label=f"próg {monitor.ram_limit}%")
    axes[1].set_ylabel("RAM [%]")
    axes[1].set_title("Zużycie RAM (cały system)")
    axes[1].legend(loc="upper left")

    axes[2].plot(t, proc_ram, color="#1baf7a", linewidth=2)
    axes[2].set_ylabel("RAM [MB]")
    axes[2].set_title("Pamięć uruchomionego notebooka / skryptu")
    axes[2].set_xlabel("Czas [s]")

    for ax in axes:
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylim(0, 100)
    axes[1].set_ylim(0, 100)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def save_html_report(path, monitor, duration, return_code, output_file):
    cpu = [s[1] for s in monitor.samples]
    proc_ram = [s[3] for s in monitor.samples]
    img = make_plot(monitor)

    status = "zakończony poprawnie" if return_code == 0 else f"błąd (kod {return_code})"
    warnings_html = "".join(f"<li>{w}</li>" for w in monitor.warnings) or "<li>Brak ostrzeżeń</li>"

    html = f"""<!DOCTYPE html>
<html lang="pl">
<head>
<meta charset="utf-8">
<title>Raport zasobów – {path.name}</title>
<style>
  body {{ font-family: sans-serif; max-width: 900px; margin: 30px auto; color: #222; }}
  table {{ border-collapse: collapse; }}
  td {{ padding: 4px 12px; border-bottom: 1px solid #ddd; }}
  img {{ max-width: 100%; }}
</style>
</head>
<body>
<h1>Raport zużycia zasobów</h1>
<table>
  <tr><td>Plik</td><td>{path.name}</td></tr>
  <tr><td>Data</td><td>{datetime.now():%Y-%m-%d %H:%M:%S}</td></tr>
  <tr><td>Status</td><td>{status}</td></tr>
  <tr><td>Czas wykonania</td><td>{duration:.1f} s</td></tr>
  <tr><td>Średnie CPU</td><td>{sum(cpu) / len(cpu):.1f} %</td></tr>
  <tr><td>Maks. CPU</td><td>{max(cpu):.1f} %</td></tr>
  <tr><td>Maks. RAM procesu</td><td>{max(proc_ram):.0f} MB</td></tr>
  <tr><td>Próg ostrzeżenia RAM</td><td>{monitor.ram_limit} %</td></tr>
  <tr><td>Liczba pomiarów</td><td>{len(monitor.samples)}</td></tr>
</table>
<h2>Ostrzeżenia</h2>
<ul>{warnings_html}</ul>
<h2>Zużycie zasobów w czasie</h2>
<img src="data:image/png;base64,{img}" alt="Wykres zużycia zasobów">
</body>
</html>"""
    output_file.write_text(html, encoding="utf-8")


def run_with_monitoring(file_path, ram_limit=80, interval=0.5):
    path = Path(file_path).resolve()
    print(f"Uruchamiam: {path.name} (próg RAM: {ram_limit}%)", flush=True)

    start = time.time()
    proc = subprocess.Popen(build_command(path), cwd=path.parent)
    monitor = ResourceMonitor(proc.pid, ram_limit, interval)
    monitor.start()

    return_code = proc.wait()
    monitor.stop()
    monitor.join()
    duration = time.time() - start

    report = path.parent / f"raport_{path.stem}.html"
    save_html_report(path, monitor, duration, return_code, report)
    print(f"Zakończono w {duration:.1f} s, ostrzeżeń: {len(monitor.warnings)}")
    print(f"Raport: {report}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Uruchamia notebook/skrypt i monitoruje zasoby")
    parser.add_argument("file", help="plik .ipynb lub .py")
    parser.add_argument("--ram-limit", type=float, default=80, help="próg RAM w %% (domyślnie 80)")
    parser.add_argument("--interval", type=float, default=0.5, help="co ile sekund pomiar")
    args = parser.parse_args()

    run_with_monitoring(args.file, args.ram_limit, args.interval)
