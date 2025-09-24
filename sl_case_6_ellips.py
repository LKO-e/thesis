import numpy as np
import scipy as sp
from src import A, v, C2, B2, C1, D_trq, M_f, D_dθ, J1, D_dgam, Ce, L, S
from src import check_hinf_upper_bound, calc_hinf, ss_d2c

# State space
C2 = C2[:, :4]
B2 = B2[:4, :]
Md_max = M_f + 2 * np.sqrt(D_trq)
B1 = np.array(
    [
        [2 * np.sqrt(D_dθ), 0, 0],
        [0, 0, 0],
        [0, Md_max / J1, 0],
        [0, 0, 2 * np.sqrt(D_dgam) * Ce / L],
    ]
)
B1 = S[:4, :4] @ B1
C = np.vstack([C1[:, :4], C2])
B = np.hstack([B1, B2])
# Constraiints on control effort and h inf norm
u_max = 24
hinf_max = 1.2
# Large number used as a constraint
Y = 1e5


def ellips_estim(A, B, C, n=100):
    rho2 = np.max(np.abs(sp.linalg.eigvals(A))) ** 2
    x = np.linspace(rho2 + 0.001, 1, n, endpoint=False)
    y_min = np.inf
    i_min = 0
    for i in x:
        P = sp.linalg.solve_discrete_lyapunov(A / np.sqrt(i), B @ B.T / (1 - i))
        y = np.trace(np.atleast_2d(C @ P @ C.T))
        if y_min > y:
            y_min = y
            i_min = i
    P = sp.linalg.solve_discrete_lyapunov(A / np.sqrt(i_min), B @ B.T / (1 - i_min))
    return np.atleast_2d(P)


#
def penalty_fnc_ellips_dscr(x, to_print=False):
    kc, Ts = x
    Ts = Ts * 1e-3
    # Discretization
    Ad, Bd, Cd, Dd = A[:4, :4], B[:4, :], C[:, :4], np.zeros((C.shape[0], B.shape[1]))
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
    # Check H inf norm
    if to_print:
        hinf = calc_hinf(ss_d2c((Acld, Bcld, Ccld, Dcld)))
        print(f"H∞ norm = {hinf:.2f}")
    if not check_hinf_upper_bound(ss_d2c((Acld, Bcld, Ccld, Dcld)), hinf_max):
        return Y
    # Calculate ellipsoid estimation
    Ad, Bd, Cd, Dd = A, B, C, np.zeros((C.shape[0], B.shape[1]))
    Ad, Bd, Cd, Dd, _ = sp.signal.cont2discrete((Ad, Bd, Cd, Dd), Ts, "zoh")
    Acld, Bcld, Ccld, Dcld = (
        Ad + Bd[:, 3:] @ (kc * Cd[[2], :]),
        Bd[:, :3],
        Cd[0:2, :],
        Dd[0, :3],
    )
    P = ellips_estim(Acld, Bcld, Ccld)
    P = np.atleast_2d(Ccld @ P @ Ccld.T)
    s_beta = np.sqrt(P[0, 0]) / v
    s_gamma = np.sqrt(P[1, 1])
    # Check control effort
    if to_print:
        print(f"σ[β]= {s_beta * 57.3 * 60:.2f} arch. min")
        print(f"σ[u]= {s_beta * kc:.2f} V")
    if s_beta * kc > 2 * u_max:
        print(s_beta * kc)
        return Y
    if to_print:
        print(f"σ[γ]= {s_gamma * 57.3 * 60:.2f} arch. min.")
        print(
            f"F_2= ({np.sqrt(v**2 * s_beta**2 + s_gamma**2) * 57.3 * 60:.2f} arch.min)**2"
        )
    return np.sqrt(v**2 * s_beta**2 + s_gamma**2) * 57.3 * 60


sp.optimize.bounds = [(0, 30e3), (0, 30)]
result = sp.optimize.differential_evolution(
    penalty_fnc_ellips_dscr,
    sp.optimize.bounds,
    strategy="best2bin",
    popsize=250,
    mutation=(0.5, 1.5),
)

print(result.message)
if result.success:
    print(result.x)
    print(result.fun)
print(penalty_fnc_ellips_dscr(result.x, True))
