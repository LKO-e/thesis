import numpy as np
import scipy as sp
from src import s, calc_hinf, tf2ss

# Parameters
w_v = 0.1  # Vertical angular velocity
g = 9.81  # Free-fall acceleration
Jgd = 1.5e-5  # Motor's moment of inertia
Cm = 6e-3  # Torque constant
Ce = Cm  # Back-EMF constant
R = 62.0  # Winding resistance
L = 25e-3  # Winding inductance
Tk = 8.9  # Correction loop time constant
dV = 1.5  # Linear acceleration
k = 9.73  # Controller gain
Ti = 0.32  # Integrator time constant
T = 4e-3  # Dynamic tuning loop's feedback filter time constant
ksi = 0.707  # Damping factor
# Dynamic tuning loop's transfer function
Wdn = (
    k
    * (1 + 1 / Ti / s)
    * Cm
    / (L * s + R)
    / Jgd
    / s
    / (T**2 * s**2 + 2 * ksi * T * s + 1)
)
# Transfer function of angle measurement error
Werr = 1 / s / (1 + Wdn) * w_v / g * 1 / (Tk * s + 1)
Werr = tf2ss(Werr)
# H inifinity norm
h_inf = calc_hinf(Werr)
print(f"H∞ norm of the error transfer function: {h_inf: .2E}")
print(f"Error for 1.5 m/s acceleration: {h_inf * dV * 57.3 * 60.0: .2E} arcmin")
# Ellipsoid estimation
nu_max = -2 * np.max(sp.linalg.eigvals(Werr[0]))
print(nu_max)
ss_size = Werr[0].shape


def f(nu):
    vQ = sp.linalg.solve_continuous_lyapunov(
        Werr[0] + nu / 2 * np.eye(ss_size[0]), -1 / nu * Werr[1] @ Werr[1].T
    )
    if (sp.linalg.eigvals(vQ) < 0.0).any():
        return 1e5
    return np.trace(Werr[2] @ vQ @ Werr[2].T)


x = np.real(np.linspace(1e-4, nu_max, 1000, endpoint=False))
y = [f(i) for i in x]
nu = x[y.index(min(y))]
print("Observer minimizer η = %.2E" % nu)
vQ = sp.linalg.solve_continuous_lyapunov(
    Werr[0] + nu / 2 * np.eye(ss_size[0]), -1 / nu * Werr[1] @ Werr[1].T
)
# Results
Δγ = np.trace(Werr[2] @ vQ @ Werr[2].T)
Δγ = np.sqrt(Δγ)
print("Δγ= %.2E" % (dV * Δγ * 60.0 * 57.3))
