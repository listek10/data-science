import datetime
import json
import os
import platform
import sys
import psutil

try:
    import GPUtil
except ImportError:
    GPUtil = None


def generate_environment_report(output_dir: str = None) -> str:
    """Generuje raport środowiskowy i sprzętowy, zapisując go do pliku JSON.

    Argumenty:
        output_dir (str): Katalog zapisu. Jeśli None lub niedostępny,
                          używa katalogu, w którym znajduje się plik .py.

    Zwraca:
        str: Absolutna ścieżka do wygenerowanego pliku JSON.
    """
   
    if output_dir is None:
        try:           
            output_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:            
            output_dir = os.getcwd()

    os.makedirs(output_dir, exist_ok=True)

    report_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "python_version": sys.version,
        "os_platform": platform.platform(),
        "installed_libraries": sorted(
            [f"{pkg.key}=={pkg.version}" for pkg in psutil.Process().iter_modules()]
            if hasattr(psutil.Process(), "iter_modules")
            else list(sys.modules.keys())[:50]
        ),  
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "ram_available_gb": round(
            psutil.virtual_memory().available / (1024**3), 2
        ),
        "gpu_info": "Niedostępne (Brak biblioteki GPUtil lub karty NVIDIA)",
    }

    if GPUtil:
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_list = []
                for gpu in gpus:
                    gpu_list.append(
                        {
                            "name": gpu.name,
                            "memory_total_mb": gpu.memoryTotal,
                            "driver": gpu.driver,
                        }
                    )
                report_data["gpu_info"] = gpu_list
            else:
                report_data["gpu_info"] = "Brak dedykowanej karty GPU NVIDIA"
        except Exception:
            report_data["gpu_info"] = "Błąd podczas odczytu danych GPU"

    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"env_report_{timestamp_str}.json"
    full_path = os.path.join(output_dir, file_name)

    with open(full_path, "w", encoding="utf-8") as json_file:
        json.dump(report_data, json_file, indent=4, ensure_ascii=False)

    return os.path.abspath(full_path)


if __name__ == "__main__":
    path_to_report = generate_environment_report()
    print(f"Raport został pomyślnie zapisany w: {path_to_report}")
