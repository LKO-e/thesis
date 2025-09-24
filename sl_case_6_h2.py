import numpy as np
import scipy as sp
from src import check_hinf_upper_bound, calc_hinf, ss_d2c
from src import Aa, v, C2, B2, C1, B1

# State space
C = np.vstack([C1, C2])
B = np.hstack([B1, B2])
# Constraints on contol effort and h inf norm
u_max = 24 / 2
hinf_max = 1.2
# Large number used as a constraint
Y = 1e5


def penalty_fnc_h2_dscr(x, to_print=False):
    kc, Ts = x
    Ts = Ts * 1e-3
    # Discretization
    Ad, Bd, Cd, Dd = Aa[:4, :4], B[:4, :], C[:, :4], np.zeros((C.shape[0], B.shape[1]))
    Ad, Bd, Cd, Dd, _ = sp.signal.cont2discrete((Ad, Bd, Cd, Dd), Ts, "zoh")
    Acld, Bcld, Ccld, Dcld = (
        Ad + Bd[:, 3:] @ (kc * Cd[[2], :]),
        Bd[:, 3:] * kc,
        Cd[2, :],
        Dd[2, 3:],
    )
    # Check poles of the system
    poles = sp.linalg.eigvals(Acld)
    if to_print:
        print("Poles of closed loop system")
        print(poles)
    if not to_print and np.any(np.abs(poles) >= 0.95):
        return Y
    # Check h inf norm
    if to_print:
        hinf = calc_hinf(ss_d2c((Acld, Bcld, Ccld, Dcld)))
        print(f"H∞ norm = {hinf:.2f}")
    if not check_hinf_upper_bound(ss_d2c((Acld, Bcld, Ccld, Dcld)), hinf_max):
        return Y
    # Calculate variances
    Ad, Bd, Cd, Dd = Aa, B, C, np.zeros((C.shape[0], B.shape[1]))
    Ad, Bd, Cd, Dd, _ = sp.signal.cont2discrete((Ad, Bd, Cd, Dd), Ts, "zoh")
    Acld, Bcld, Ccld, Dcld = (
        Ad + Bd[:, 3:] @ (kc * Cd[[2], :]),
        Bd[:, :3],
        Cd[0:2, :],
        Dd[0, :3],
    )
    # Check control effort
    P = sp.linalg.solve_discrete_lyapunov(Acld, Bcld @ Bcld.T)
    P = np.atleast_2d(Ccld @ P @ Ccld.T + Dcld @ Dcld.T)
    s_beta = np.sqrt(P[0, 0]) / np.sqrt(Ts) / v
    s_gamma = np.sqrt(P[1, 1]) / np.sqrt(Ts)
    # Check control effort
    if s_beta * kc > u_max:
        print(s_beta * kc)
        return Y
    if to_print:
        print(f"σ[β]= {s_beta * 57.3 * 60:.2f} arch. min")
        print(f"σ[u]= {s_beta * kc:.2f} V")
    if to_print:
        print(f"σ[γ]= {s_gamma * 57.3 * 60:.2f} arch. min.")
        print(
            f"F_1= ({np.sqrt(v**2 * s_beta**2 + s_gamma**2) * 57.3 * 60:.2f} arch.min)**2"
        )
    return np.sqrt(v**2 * s_beta**2 + s_gamma**2) * 57.3 * 60


sp.optimize.bounds = [(1, 30e3), (1, 20)]
result = sp.optimize.differential_evolution(
    penalty_fnc_h2_dscr,
    sp.optimize.bounds,
    strategy="best2bin",
    popsize=250,
)

print(result.message)
print(result.x)
print(result.fun)
penalty_fnc_h2_dscr(result.x, to_print=True)
