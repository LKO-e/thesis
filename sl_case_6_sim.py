import sys
import copy
from pathlib import Path
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from src import A, B2, C2, J1, S
import plot_style

plot_style.set_plot_style()

C2 = C2[:, :4]
B2 = B2[:4, :]


class StabLoop:
    def __init__(self):
        self.t = 0.0
        self.last_t = 0.0
        self.last_u = 0.0
        self.beta_ = 1 / 57.3 / 60
        self.dt = 13e-3
        self.k_c = 1940
        self.lu = []

    def step_1(self, t, x):
        if t >= self.t:
            self.t = t
        else:
            print("Ooops!")
        X = np.array([x]).T
        if t - self.last_t >= self.dt:
            self.last_t = t
            self.last_u = self.k_c * (C2 @ X + self.beta_)
        u = self.last_u
        self.lu.append(float(u))
        dx = A @ X + B2 * u
        dx = dx[:, 0].tolist()
        return dx

    def step_2(self, t, x):
        if t >= self.t:
            self.t = t
        else:
            print("Ooops!")
        X = np.array([x]).T
        if t - self.last_t >= self.dt:
            self.last_t = t
            self.last_u = self.k_c * (C2 @ X)
        u = self.last_u
        self.lu.append(float(u))
        dx = A @ X + B2 * u + S[:4, :4] @ np.array([[0, 0, 1 / J1, 0]]).T * 0.01
        dx = dx[:, 0].tolist()
        return dx


stab_loop = StabLoop()
sol = sp.integrate.RK45(
    stab_loop.step_1,
    0.0,
    [0, 0, 0, 0],
    0.4,
    first_step=1e-5,
    max_step=1e-5,
    rtol=10,
    atol=10,
)

t_values1 = []
y_values1 = []
for i in range(1000000):
    sol.step()
    t_values1.append(sol.t)
    y_values1.append(sol.y[0] * 10)
    if sol.status == "finished":
        break
u_values1 = copy.deepcopy(stab_loop.lu[1::6])
stab_loop.lu = []
del stab_loop
del sol
stab_loop1 = StabLoop()
sol1 = sp.integrate.RK45(
    stab_loop1.step_2,
    0.0,
    [0, 0, 0, 0],
    0.7,
    first_step=1e-5,
    max_step=1e-5,
    rtol=10,
    atol=10,
)

t_values2 = []
y_values2 = []
for i in range(1000000):
    sol1.step()
    t_values2.append(sol1.t)
    y_values2.append(sol1.y[0] * 10)
    if sol1.status == "finished":
        break
print(len(stab_loop1.lu[1::6]))
u_values2 = copy.deepcopy(stab_loop1.lu[1::6])

fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.9))
# without H inf
ax[0].plot(
    t_values1, y_values1, ls="--", c="k", label=r"$\mathrm{\vartheta_1, \ угл. \ мин.}$"
)
ax[0].plot(t_values1, u_values1, ls="-", c="k", label=r"$u_\mathrm{ус}$, В")
# with H inf
ax[1].plot(
    t_values2, y_values2, ls="--", c="k", label=r"$\mathrm{\vartheta_1, \ угл. \ мин.}$"
)
ax[1].plot(t_values2, u_values2, ls="-", c="k", label=r"$u_\mathrm{ус}$, В")
# Decoration
ax[0].set_xlabel(r"$t, \mathrm{ \  с}$")
ax[0].grid(True)
ax[0].legend(loc="best")
ax[0].set_title("a)")
ax[1].set_xlabel(r"$t, \mathrm{ \  с}$")
ax[1].grid(True)
ax[1].legend()
ax[1].set_title("б)")
if len(sys.argv) != 2:
    plt.show()
else:
    plot_dir_path = Path(sys.argv[1])
    if not plot_dir_path.is_dir():
        plot_dir_path.mkdir()
    plt.savefig(plot_dir_path / "sl_case_6_step.svg")
