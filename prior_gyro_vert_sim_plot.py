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


to_highlight = False
if len(sys.argv) >= 3:
    plot_dir_path = Path(sys.argv[2])
    if not plot_dir_path.is_dir():
        plot_dir_path.mkdir()
    if len(sys.argv) == 4 and sys.argv[3] == "--highlight":
        to_highlight = True
data = read_csv_columns(sys.argv[1])

l0 = 1600

# Yaw plot
fig = plt.figure(figsize=(7.0, 2.8))
ax = plt.subplot(111)
ax.plot(data["time"], data["psi"] * 57.3, "k-")
ax.set_ylabel(r"$\mathrm{ψ, \ град.}$")
ax.set_xlabel(r"$t, \  \mathrm{с}$")
axins = ax.inset_axes(
    [0.6, 0.15, 0.3, 0.3],
    xlim=(87.0, 90.0),
    ylim=(93.85, 94.25),
)
axins.plot(data["time"], data["psi"] * 57.3, "k-")
ax.indicate_inset_zoom(axins, edgecolor="black")
start, end = axins.get_ylim()
axins.yaxis.tick_right()
axins1 = ax.inset_axes(
    [0.1, 0.6, 0.3, 0.3],
    xlim=(0, 25.2),
    ylim=(-0.1, 0.1),
)
axins1.plot(data["time"], data["psi"] * 57.3, "k-")
ax.indicate_inset_zoom(axins1, edgecolor="black")
start, end = axins1.get_ylim()
axins1.yaxis.tick_right()
if len(sys.argv) >= 3:
    plt.savefig(plot_dir_path / "sim_gv_psi.svg")
# Pitch plot
fig = plt.figure(figsize=(7.0, 2.8))
ax = plt.subplot(111)
ax.plot(data["time"], data["theta"] * 57.3 * 60.0, "k-")
ax.set_ylabel(r"$\mathrm{\vartheta,\ угл. \ мин}$")
ax.set_xlabel(r"$t, \  \mathrm{с}$")
axins = ax.inset_axes(
    [0.60, 0.17, 0.3, 0.3],
    xlim=(85.0, 90.0),
    ylim=(84.0, 88.0),
)
axins.plot(data["time"], data["theta"] * 57.3 * 60.0, "k-")
ax.indicate_inset_zoom(axins, edgecolor="black")
start, end = axins.get_ylim()
# axins.yaxis.set_ticks(np.arange(start, end, 10))
axins.yaxis.tick_right()
if len(sys.argv) >= 3:
    plt.savefig(plot_dir_path / "sim_gv_theta.svg")
# Roll plot
fig = plt.figure(figsize=(7.0, 2.8))
ax = plt.subplot(111)
ax.plot(data["time"], data["gam"] * 57.3, "k-")
ax.set_ylabel(r"$\mathrm{γ, \ град.}$")
ax.set_xlabel(r"$t, \  \mathrm{с}$")
axins = ax.inset_axes(
    [0.35, 0.18, 0.3, 0.3],
    xlim=(45.0, 50.0),
    ylim=(4.0, 4.6),
)
axins.plot(data["time"], data["gam"] * 57.3, "k-")
ax.indicate_inset_zoom(axins, edgecolor="black")
start, end = axins.get_ylim()
# axins.yaxis.set_ticks(np.arange(start, end, 10))
axins.yaxis.tick_right()
if len(sys.argv) >= 3:
    plt.savefig(plot_dir_path / "sim_gv_gamma.svg")
# Roll measurement error plot
fig = plt.figure(figsize=(7.0, 2.8))
ax = plt.subplot(111)
# ax.plot(data["time"], (data["gam"] + data["alpha"]) * 57.3 * 60.0, "k-")
ax.plot(data["time"], l0 * np.sin(data["gam"] + data["alpha"]), "k-")
# ax.axhline(-2.1, c="k", ls="--")
# ax.axhline(2.1, c="k", ls="--")
ax.set_ylabel(r"$Δ\hat{h}, \ \mathrm{мм}$")
ax.set_xlabel(r"$t, \  \mathrm{с}$")
if to_highlight:
    ax.axvspan(22, 27, alpha=0.5, color="grey")
    ax.axvspan(77, 82, alpha=0.5, color="grey")
if len(sys.argv) >= 3:
    plt.savefig(plot_dir_path / "sim_gv_crosslevel_err.svg")
# Delta beta plot
fig = plt.figure(figsize=(7.0, 2.8))
ax = plt.subplot(111)
ax.plot(
    data["time"],
    data["beta"] * 57.3 * 60.0,
    "k-",
)
# Areas when rolling
if to_highlight:
    ax.axvspan(22, 27, alpha=0.5, color="grey")
    ax.axvspan(77, 82, alpha=0.5, color="grey")
ax.set_ylabel(r"$\mathrm{Δ{β}, \ угл. \ мин}$")
ax.set_xlabel(r"$t, \  \mathrm{с}$")
if len(sys.argv) >= 3:
    plt.savefig(plot_dir_path / "sim_gv_delta_beta.svg")
# Theta 1 plot
fig = plt.figure(figsize=(7.0, 2.8))
ax = plt.subplot(111)
ax.plot(data["time"], (data["theta"] + data["beta"]) * 57.3 * 60.0, "k-")
ax.set_ylabel(r"$\mathrm{\vartheta_1, \ угл. \ мин}$")
ax.set_xlabel(r"$t, \  \mathrm{с}$")
if len(sys.argv) >= 3:
    plt.savefig(plot_dir_path / "sim_gv_theta_1.svg")
# Winding current of the stabilization motor and voltage on correction motor's windings
fig = plt.figure(figsize=(7.0, 2.8))
ax1 = plt.subplot(121)
ax2 = plt.subplot(122)
ax1.set_title("а)")
ax1.plot(data["time"], (data["u_s"]), "k-")
ax1.set_ylabel(r"$u_\mathrm{с} \mathrm{, \ В}$")
ax1.set_xlabel(r"$t, \  \mathrm{с}$")
ax1.set_xlim([50.0, 52.0])
ax2.set_title("б)")
ax2.plot(data["time"], (data["u_k"]), "k-")
ax2.set_ylabel(r"$u_\mathrm{к} \mathrm{, \ В}$")
ax2.set_xlabel(r"$t, \  \mathrm{с}$")
ax2.set_xlim([50.0, 52.0])
if len(sys.argv) >= 3:
    plt.savefig(plot_dir_path / "sim_gv_motors.svg")
# Gyro pendulum's float roll angle
fig = plt.figure(figsize=(7.0, 2.8))
ax = plt.subplot(111)
ax.plot(data["time"], (data["alpha_m"] + data["gam"]) * 57.3, "k-")
ax.set_ylabel(r"$\mathrm{{γ_{м}}, \ град.}$")
ax.set_xlabel(r"$t, \  \mathrm{с}$")
if len(sys.argv) >= 3:
    plt.savefig(plot_dir_path / "sim_gv_gam_m.svg")
if len(sys.argv) == 2:
    plt.show()
