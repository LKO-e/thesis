import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import plot_style
from src import bode_tf, s

plot_style.set_plot_style()

dpsi = 0.1
Tk = 8.6
delta_gamma = 0.5 / 57.3 / 60
theta = 40e-3
dtheta = 1.1e-2

T_theta = theta / dtheta
W = delta_gamma / (dpsi * Tk) * (Tk * s + 1) * (T_theta * s + 1) / theta

f1 = 5
T1 = 1 / 2 / np.pi / f1
Wf1 = T1 * s / (T1 * s + 1)
Wf2 = (T1**2 * s**2 + 2 * 0.707 * T1 * s) / (T1**2 * s**2 + 2 * 0.707 * T1 * s + 1)

w = np.logspace(-2, 4, 1000)
_, mag, _ = bode_tf(W, w)
_, mag1, _ = bode_tf(Wf1, w)
_, mag2, _ = bode_tf(Wf2, w)

fig, ax = plt.subplots(1, 1, figsize=(4.5, 3.1))

ax.semilogx(w, mag, ls="-", c="k", label=r"$\mathrm{Запр. \  зона}$")
ax.semilogx(w, mag1, ls="-.", c="k", label=r"$\mathrm{ФНЧ \ 1 \ п.}$")
ax.semilogx(w, mag2, ls="--", c="k", label=r"$\mathrm{ФНЧ \ 2 \ п.}$")
ax.grid(which="both")
ax.set_ylabel(r"$L \mathrm{,\  дБ}$")
ax.set_xlabel(r"$\mathrm{\omega, \  с^{-1}}$")
ax.legend()

if len(sys.argv) == 2:
    plot_dir_path = Path(sys.argv[1])
    if not plot_dir_path.is_dir():
        plot_dir_path.mkdir()
    plt.savefig(plot_dir_path / "pitch_filter.svg")
else:
    plt.show()
