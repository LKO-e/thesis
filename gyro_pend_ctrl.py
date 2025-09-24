import sys
from pathlib import Path
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import plot_style
from src import s, tf2ss

plot_style.set_plot_style()

# Filter in feedback
T = 1 / (2 * 3.14 * 40)
ksi = 0.707
# PI-controller
k = 2 * 3.14 * 10
Ti = 0.3
K = k / (T**2 * s**2 + 2 * ksi * T * s + 1) * (1 + 1 / Ti / s)
# Plant transfer function
G = 1 / s
# Open-loop transfer function
print(K * G)
# Closed loop
W = (K * G) / (1 + K * G)
# Bode plot
omega = np.logspace(0, 3.5, 1000)
omega, mag, phase = sp.signal.bode(tf2ss(K / s), omega)
# Step plot
t = np.linspace(0, 0.8, 1000)
u = np.ones_like(t) * 10
t, y_out, _ = sp.signal.lsim(tf2ss(W), u, t)
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7), frameon=False)
ax01 = ax[0].twinx()
(line1,) = ax[0].semilogx(omega, mag, "k-", label=r"$L, \  \mathrm{дБ}$")
(line2,) = ax01.semilogx(omega, phase, "k--", label=r"$\mathrm{\phi, \ град.}$")
ax[1].plot(t, y_out, "k-")
# Decoration
ax[0].set_title(r"a)")
ax[1].set_title(r"б)")
ax01.set_ylabel(r"$\mathrm{\phi, \ град.}$")
ax[0].set_ylabel(r"$L, \mathrm{ \ дБ}$")
ax[0].set_xlabel(r"$\mathrm{\omega, \  с^{-1}}$")
ax[1].set_xlabel(r"$t, \ \mathrm{с}$")
ax[1].set_ylabel(r"$\Omega_\mathrm{гд}, \ \mathrm{c}^{-1} $")
ax01.set_yticks([-90, -180, -270])
ax01.legend(handles=[line1, line2], loc="lower left")
ax[0].grid(True, which="both")
# Saving plot
if len(sys.argv) != 2:
    plt.show()
else:
    plot_dir_path = Path(sys.argv[1])
    if not plot_dir_path.is_dir():
        plot_dir_path.mkdir()
        plt.savefig(plot_dir_path / "bode_step_gyro_pend.svg")
