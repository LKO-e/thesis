import sys
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import plot_style

plot_style.set_plot_style()


def read_csv_columns(filename):
    columns = {}
    with open(filename, mode="r", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile, delimiter="\t")  # first row is header
        # Initialize empty lists for each column
        for field in reader.fieldnames:
            columns[field] = []
        # Fill the columns
        for row in reader:
            for field in reader.fieldnames:
                columns[field].append(float(row[field]))
        for field in columns:
            columns[field] = np.array(columns[field])
    return columns


if len(sys.argv) < 2:
    print("Provide a source file from an oscillograph")
    sys.exit(1)
to_save = False
if len(sys.argv) == 3:
    plot_dir_path = Path(sys.argv[2])
    if not plot_dir_path.is_dir():
        plot_dir_path.mkdir()
    to_save = True

sim = read_csv_columns(sys.argv[1])
sim["Vg"] = sim["V(n005)"]
sim["Vs"] = sim["V(n003)"]
sim["time"] = sim["time"] * 1e6

fig = plt.figure(figsize=(7.0, 2.8))
ax1 = plt.subplot(121)
t = sim["time"][(sim["time"] >= 1.015) & (sim["time"] <= 1.13)]
t = (t - t[0]) * 1e3
Vs = sim["Vs"][(sim["time"] >= 1.015) & (sim["time"] <= 1.13)]
Vg = sim["Vg"][(sim["time"] >= 1.015) & (sim["time"] <= 1.13)]
ax1.plot(t, Vs, "k-", label=r"$V_{S}$")
ax1.plot(t, Vg, "k--", label="$V_{G}$")
ax1.set_ylabel(r"$V, \  \mathrm{В}$")
ax1.set_xlabel(r"$t, \ \mathrm{нс}$")
ax1.legend()
ax1.set_title("а)")

ax2 = plt.subplot(122)
t = sim["time"][(sim["time"] >= 2.01) & (sim["time"] <= 2.17)]
t = (t - t[0]) * 1e3
Vs = sim["Vs"][(sim["time"] >= 2.01) & (sim["time"] <= 2.17)]
Vg = sim["Vg"][(sim["time"] >= 2.01) & (sim["time"] <= 2.17)]
ax2.plot(t, Vs, "k-", label=r"$V_{S}$")
ax2.plot(t, Vg, "k--", label="$V_{G}$")
ax2.set_ylabel(r"$V, \  \mathrm{В}$")
ax2.set_xlabel(r"$t, \ \mathrm{нс}$")
ax2.legend()
ax2.set_title("б)")
if to_save:
    plt.savefig(plot_dir_path / "pwm_sim.svg")
else:
    plt.show()
