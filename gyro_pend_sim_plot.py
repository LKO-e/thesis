import sys
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
import plot_style

plot_style.set_plot_style()


def read_csv_columns(filename):
    columns = {}
    with open(filename, mode="r", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)  # first row is header
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


if len(sys.argv) > 1:
    data = read_csv_columns(sys.argv[1])
else:
    print("Provide a simulation log file")
    sys.exit(1)

plt.figure(figsize=(7.0, 2.9))
ax1 = plt.subplot(121)
ax2 = plt.subplot(122)
ax1.plot(
    data["time"],
    data["omega"],
    "k-",
    label=r"$\Omega, \ \mathrm{с}^{-1}$",
)
ax2.plot(
    data["time"], data["motor_current"], "k-", label=r"$i_\mathrm{гд}, \ \mathrm{А}$"
)
ax2.set_ylim([-0.05, 0.6])
axins = ax1.inset_axes(
    [0.35, 0.15, 0.45, 0.30],
    xlim=(6.2, 7.7),
    ylim=(990, 1015),
)
start, end = axins.get_ylim()
axins.yaxis.set_ticks(np.arange(start, end, 10))
axins.plot(data["time"], data["omega"], "k-")
ax1.indicate_inset_zoom(axins, edgecolor="black")
axins.yaxis.tick_right()
ax1.set_ylabel(r"$\Omega, \ \mathrm{с}^{-1}$")
ax1.set_xlabel(r"$t, \ \mathrm{c}$")
ax2.set_ylabel(r"$i_\mathrm{гд}, \ \mathrm{А}$")
ax2.set_xlabel(r"$t, \ \mathrm{c}$")
ax1.set_title("a)")
ax2.set_title("б)")
if len(sys.argv) == 3:
    plot_dir_path = Path(sys.argv[2])
    if not plot_dir_path.is_dir():
        plot_dir_path.mkdir()
    plt.savefig(plot_dir_path / "gyro_pend_step_nonlin.svg")
else:
    plt.show()
