import matplotlib.pyplot as plt

# średni pobór mocy urządzenia podczas pracy [kW]
DEVICE_POWER_KW = {
    "laptop": 0.05,
    "desktop": 0.20,
    "cloud GPU": 0.40,  # GPU + chłodzenie serwerowni
}

# emisja CO2 na 1 kWh energii elektrycznej [g/kWh] - wartości przybliżone
REGION_CO2 = {
    "Polska": 650,
    "Niemcy": 380,
    "USA": 370,
    "UE (średnia)": 250,
    "Francja": 55,
    "Norwegia": 30,
}

# do porównań z codziennymi czynnościami
CAR_G_PER_KM = 150          # g CO2 na 1 km jazdy autem
PHONE_CHARGE_KWH = 0.015    # jedno pełne ładowanie smartfona
KETTLE_KWH = 0.1            # zagotowanie czajnika wody


def carbon_footprint(hours, device, region):
    """Liczy zużycie energii i emisję CO2 dla sesji Data Science."""
    if device not in DEVICE_POWER_KW:
        raise ValueError(f"Nieznane urządzenie: {device}. Dostępne: {list(DEVICE_POWER_KW)}")
    if region not in REGION_CO2:
        raise ValueError(f"Nieznany region: {region}. Dostępne: {list(REGION_CO2)}")

    energy_kwh = hours * DEVICE_POWER_KW[device]
    co2_g = energy_kwh * REGION_CO2[region]

    return {
        "hours": hours,
        "device": device,
        "region": region,
        "energy_kwh": energy_kwh,
        "co2_g": co2_g,
        "car_km": co2_g / CAR_G_PER_KM,
        "phone_charges": energy_kwh / PHONE_CHARGE_KWH,
        "kettles": energy_kwh / KETTLE_KWH,
    }


def print_result(r):
    print(f"Sesja: {r['hours']} h, {r['device']}, {r['region']}")
    print(f"  Zużycie energii: {r['energy_kwh']:.2f} kWh")
    print(f"  Emisja CO2:      {r['co2_g']:.0f} g")
    print("  To tyle co:")
    print(f"    - {r['car_km']:.1f} km jazdy autem")
    print(f"    - {r['phone_charges']:.0f} ładowań smartfona")
    print(f"    - {r['kettles']:.0f} zagotowań czajnika")


def plot_result(r, filename="carbon_footprint.png"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 1. to samo urządzenie w różnych regionach
    regions = list(REGION_CO2)
    co2_regions = [carbon_footprint(r["hours"], r["device"], reg)["co2_g"] for reg in regions]
    colors = ["#2a78d6" if reg == r["region"] else "#b7b6b0" for reg in regions]
    bars = ax1.barh(regions, co2_regions, color=colors)
    ax1.bar_label(bars, labels=[f"{v:.0f} g" for v in co2_regions], padding=3)
    ax1.invert_yaxis()
    ax1.set_xlim(0, max(co2_regions) * 1.2)
    ax1.set_title(f"Emisja CO2 – {r['device']}, {r['hours']} h, różne regiony")
    ax1.set_xlabel("CO2 [g]")

    # 2. różne urządzenia w wybranym regionie
    devices = list(DEVICE_POWER_KW)
    co2_devices = [carbon_footprint(r["hours"], dev, r["region"])["co2_g"] for dev in devices]
    colors = ["#2a78d6" if dev == r["device"] else "#b7b6b0" for dev in devices]
    bars = ax2.bar(devices, co2_devices, color=colors)
    ax2.bar_label(bars, labels=[f"{v:.0f} g" for v in co2_devices], padding=3)
    ax2.set_ylim(0, max(co2_devices) * 1.2)
    ax2.set_title(f"Emisja CO2 – {r['region']}, {r['hours']} h, różne urządzenia")
    ax2.set_ylabel("CO2 [g]")

    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        f"Twoja sesja: {r['energy_kwh']:.2f} kWh, {r['co2_g']:.0f} g CO2 "
        f"≈ {r['car_km']:.1f} km autem",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    print(f"Wykres zapisany: {filename}")


if __name__ == "__main__":
    result = carbon_footprint(hours=6, device="cloud GPU", region="Polska")
    print_result(result)
    plot_result(result)
