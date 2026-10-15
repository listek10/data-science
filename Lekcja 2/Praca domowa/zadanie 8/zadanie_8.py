import matplotlib.pyplot as plt

devices = ["Laptop", "Desktop", "Server"]
energy_wh = [30, 150, 400]
colors = ["#4C9F70", "#F2A541", "#D1495B"]

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(devices, energy_wh, color=colors)
ax.bar_label(bars, labels=[f"{v} Wh" for v in energy_wh], padding=3)

ax.set_title("Zużycie energii przez urządzenia")
ax.set_xlabel("Urządzenie")
ax.set_ylabel("Zużycie energii [Wh]")
ax.set_ylim(0, max(energy_wh) * 1.15)

fig.tight_layout()
fig.savefig("energy_comparison.png", dpi=150)
