import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
from src import b_la, a_la, D_la, g
import plot_style

plot_style.set_plot_style()


def contour_line_W(T, omega):
    return (
        np.sqrt(
            (omega / 3600 / 57.3) ** 2 * T**2
            + b_la * D_la / (T**2 + T * a_la + b_la) / g**2
        )
        * 57.3
        * 60
    )


def opt_line_W(T):
    return (
        np.sqrt(
            b_la
            * D_la
            * (2 * T + a_la)
            / 2
            / T
            / (T**2 + T * a_la + b_la) ** 2
            / g**2
            * 57.3**2
        )
        * 3600
    )


def contour_line_N(T, N):
    return (
        np.sqrt(
            (N / 60 / 57.3) ** 2 * T / 2 + b_la * D_la / (T**2 + T * a_la + b_la) / g**2
        )
        * 57.3
        * 60
    )


def opt_line_N(T):
    return (
        np.sqrt(2 * b_la * D_la * (2 * T + a_la) / (T**2 + T * a_la + b_la) ** 2 / g**2)
        * 57.3
        * 60
    )


fig, ax = plt.subplots(1, 2, dpi=150, figsize=(7.5, 4))


def formatter(x, pos):
    return rf"{x:.1f}$\prime$".replace(".", ",")


T = np.linspace(0.1, 40, 1000)
N = np.linspace(0, 2, 1000)
T1, N1 = np.meshgrid(T, N)
z = contour_line_N(T1, N1)
cs = ax[0].contour(
    T1,
    N1,
    z,
    levels=[1, 1.5, 2, 3, 5, 8, 10, 15, 30],
    linestyles="dashed",
    colors="0.3",
    extend="both",
)
ax[0].clabel(cs, inline=True, fontsize=12, manual=True, fmt=FuncFormatter(formatter))
ax[0].plot(T, opt_line_N(T), ls="solid", c="black")
ax[0].set_ylim(0, 2)
ax[0].set_xlabel(r"$T_\mathrm{к}, \mathrm{с}$")
ax[0].set_ylabel(r"$N, \  \mathrm{°/ \sqrt{ч}}$")
ax[0].set_title("a)")

w = np.linspace(0.1, 20, 1000)
T1, w1 = np.meshgrid(T, w)
z = contour_line_W(T1, w1)
cs = ax[1].contour(
    T1,
    w1,
    z,
    levels=[1, 1.5, 2, 3, 5, 8, 10, 15, 30],
    linestyles="dashed",
    colors="0.3",
    extend="both",
)
ax[1].clabel(cs, inline=True, fontsize=12, manual=True, fmt=FuncFormatter(formatter))
ax[1].plot(T, opt_line_W(T), ls="solid", c="black")
ax[1].set_ylim(0, 20)
ax[1].set_xlabel(r"$T_\mathrm{к}, \mathrm{с}$")
ax[1].set_ylabel(r"$\mathrm{\sigma_{нн}(\omega), \ °/ ч}$")
ax[1].set_title("б)")
plt.show()
