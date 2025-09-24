import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import plot_style
from src import (
    a_dgam,
    D_dgam,
    b_dgam,
    D_dtheta,
    T_dtheta,
    D_trq,
    T_trq,
    a_la,
    b_la,
    D_la,
)

plot_style.set_plot_style()

to_save = False
if len(sys.argv) == 2:
    plot_dir_path = Path(sys.argv[1])
    if not plot_dir_path.is_dir():
        plot_dir_path.mkdir()
    to_save = True


def F_dgam(p):
    return np.sqrt((2 * a_dgam * D_dgam) / (1 + a_dgam * p + b_dgam * p**2))


def S_dgam(ω):
    p = 1j * ω
    return np.real(F_dgam(p) * F_dgam(-p))


def F_dtheta(p):
    return np.sqrt(2 * D_dtheta * T_dtheta) / (T_dtheta * p + 1)


def S_dtheta(ω):
    p = 1j * ω
    return np.real(F_dtheta(p) * F_dtheta(-p))


def F_trq(p):
    return np.sqrt(2 * D_trq * T_trq) / (T_trq * p + 1)


def S_trq(ω):
    p = 1j * ω
    return np.real(F_trq(p) * F_trq(-p))


def F_la(p):
    return np.sqrt(2 * a_la * b_la * D_la) * p / (b_la * p**2 + a_la * p + 1)


def S_la(ω):
    p = 1j * ω
    return np.real(F_la(p) * F_la(-p))


fig, ax = plt.subplots(1, 2, dpi=100, figsize=(7.0, 2.9))
f = np.linspace(0, 20, 1000)
(line1,) = ax[0].plot(
    f, S_dgam(2 * np.pi * f), "k-", label=r"$S_\mathrm{\dot{\gamma}}$"
)
ax[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
ax[0].set_ylabel(r"$S_\mathrm{\dot{\gamma}}, \  \mathrm{(рад/с)^2/Гц}$")
ax[0].set_xlabel(r"$f \mathrm{, \  Гц}$")
ax[0].set_title(r"$\mathrm{а)}$")
ax[0].set_ylim(bottom=0)
ax11 = ax[0].twinx()
(line2,) = ax11.plot(f, S_la(2 * np.pi * f), "k--", label=r"$S_\mathrm{бо}$")
ax11.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
ax11.set_ylabel(r"$S_\mathrm{бо}, \ \mathrm{(м/с^2)^2/Гц}$")
ax11.set_xlabel(r"$f \mathrm{, \  Гц}$")
ax11.set_ylim(bottom=0)
ax11.legend(handles=[line1, line2])
f = np.linspace(0, 20, 1000)
ax[1].plot(f, S_dtheta(2 * np.pi * f), "k-")
ax[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
ax[1].set_ylabel(r"$S_\mathrm{\dot{\vartheta}}, \ \mathrm{(рад/c)^2 / Гц}$")
ax[1].set_xlabel(r"$f \mathrm{, \  Гц}$")
ax[1].set_title(r"$\mathrm{б)}$")
ax[1].set_ylim(bottom=0)
if to_save:
    plt.savefig(plot_dir_path / "PSD1.svg")

fig, ax = plt.subplots(1, 1, dpi=100, figsize=(3.6, 2.9))

f = np.linspace(0, 0.5, 1000)
ax.plot(f, S_trq(2 * np.pi * f), "k-")
ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
ax.set_ylabel(r"$S_\mathrm{м}, \   \mathrm{(Н \cdot м)^2/Гц}$")
ax.set_xlabel(r"$f \mathrm{, \  Гц}$")
ax.set_ylim(bottom=0)
if to_save:
    plt.savefig(plot_dir_path / "PSD2.svg")
if not to_save:
    plt.show()
