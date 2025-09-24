from pathlib import Path
import numpy as np
import sympy as sy
import scipy as sp
import json

s = sy.symbols("s")

# Parameters
g = 9.81
v = 0.1
u_max = 12
H = 0.2
Cm = 0.11
Ce = Cm
J1 = 1.1e-3
J2 = 2.5e-4
Jr = 5e-6
L = 9e-3
R = 18
# ϑ parameters
D_dtheta = (5.5e-2) ** 2
T_dtheta = 1.67e-2
fp_0 = np.sqrt(2 * D_dtheta / T_dtheta)
fp_1 = 1 / T_dtheta
# Ɣ parameters
beta_dgam = 5 * 2 * np.pi
mu_dgam = 30
D_dgam = (3 / 57.3 * 2 * np.pi) ** 2
a_dgam = 2 * mu_dgam / (mu_dgam**2 + beta_dgam**2)
b_dgam = 1 / (mu_dgam**2 + beta_dgam**2)
fr_0 = np.sqrt(2 * a_dgam * D_dgam) / b_dgam
fr_1 = a_dgam / b_dgam
fr_2 = 1 / b_dgam
# Friction toqrue along the stabilization axis
M_f = 150e-4
kl = M_f / np.sqrt(D_dgam) * (np.sqrt(2 * np.pi) + 2) / 2 / np.sqrt(2 * np.pi)
# Stochastic paramters of the rest of disturbance torques
D_trq = (20e-4) ** 2
dM = 30e-4 / 2
k_trq = 2 * D_trq / dM
T_trq = k_trq / dM
fm_0 = np.sqrt(2 * D_trq / T_trq)
fm_1 = 1 / T_trq
#  Lateral acceleration of swinging
D_la = (0.15) ** 2
a_la = 40e-2
b_la = 150e-4
# State space
A = np.array(
    [
        [0, -1, 0, 0],
        [0, 0, -H / J2, 0],
        [0, H / J1, 0, Cm / J1],
        [0, 0, -Ce / L, -R / L],
    ],
)
Asf = np.array(
    [
        [-1 / T_dtheta, 0, 0, 0],
        [0, -1 / T_trq, 0, 0],
        [0, 0, 0, 1],
        [0, 0, -1 / b_dgam, -a_dgam / b_dgam],
    ]
)
Csf = np.array(
    [
        [1, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 1 / J1, kl / J1, 0],
        [0, 0, Ce / L, 0],
    ]
)
Aa = np.block(
    [
        [A, Csf],
        [np.zeros((4, 4)), Asf],
    ]
)
B1 = np.array(
    [
        [0, 0, 0, 0, 0, np.sqrt(2 * D_trq / T_trq), 0, 0],
        [0, 0, 0, 0, np.sqrt(2 * D_dtheta / T_dtheta), 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, np.sqrt(2 * a_dgam * D_dgam) / b_dgam],
    ]
).T
B2 = np.array([[0, 0, 0, 1 / L, 0, 0, 0, 0]]).T
C1 = np.array(
    [
        [v, 0, 0, 0, 0, 0, 0, 0],
        [0, -J2 / H, 0, 0, 0, 0, 0, 0],
    ]
)
C2 = np.array([[-1, 0, 0, 0, 0, 0, 0, 0]])
D2 = np.zeros((1, 1))

Si = [10 / 57.3 / 60.0, 0.1, 0.3, 1, 0.1, 1e-3, 1, 0.33 / 5e-4]
S = np.diag([1 / i for i in Si])
Si = np.diag(Si)

# S = np.eye(*S.shape)
# Si = np.eye(*Si.shape)

A, Aa, B1, B2, C1, C2 = (
    S[:4, :4] @ A @ Si[:4, :4],
    S @ Aa @ Si,
    S @ B1,
    S @ B2,
    C1 @ Si,
    C2 @ Si,
)

W_ol = (A, B2[:4, :], C2[:, :4], np.array([[0.0]]))


def check_hinf_upper_bound(ss, gam):
    """Checking the H inf norm  upper bound via Hamiltonian matrix"""
    A, B, C, D = ss
    Ri = sp.linalg.inv(gam**2 * np.eye(D.shape[1], D.shape[1]) - D.T @ D)
    Ham = np.block(
        [
            [A + B @ Ri @ D.T @ C, B @ Ri @ B.T],
            [
                -C.T @ (np.eye(D.shape[0]) + D @ Ri @ D.T) @ C,
                -(A + B @ Ri @ D.T @ C).T,
            ],
        ]
    )
    return not any(np.isclose(np.real(sp.linalg.eigvals(Ham)), 0.0))


def penalty_fnc_h2(Ak, Bk, Ck, Dk, A, By, Cy, gam, u_max, to_print=False):
    """Penalty function for fixed structure optimal control synthesis
    based on H2 norm criterion"""
    Y = 1e3
    # Check closed loop poles
    if Ak is not None:
        Acl = np.block([[A + By @ Dk @ Cy, By @ Ck], [Bk @ Cy, Ak]])
    else:
        Acl = A + By @ Dk @ Cy
    if (max_eig := np.max(np.real(sp.linalg.eigvals(Acl)))) > -1e-2:
        return Y + max_eig
    # Check H_inf norm
    Bw = np.array([[0, 0, 0.0, 0, 0, 0, 0, 0]]).T
    Dwy = np.zeros((Cy.shape[0], Bw.shape[1]))
    Dwy[0, 0] = 1
    Cz = np.array([[1, 0, 0, 0, 0, 0, 0, 0]])
    Cz = Cz @ Si
    Dwz = np.array([[0]])
    Duz = np.array([[0]])
    if Ak is not None:
        Bcl = np.block([[Bw + By @ Dk @ Dwy], [Bk @ Dwy]])
        Ccl = np.block([[Cz + Duz @ Dk @ Cy, Duz @ Ck]])
    else:
        Bcl = Bw + By @ Dk @ Dwy
        Ccl = Cz + Duz @ Dk @ Cy
    Dcl = Dwz + Duz @ Dk @ Dwy
    if to_print:
        hinf = calc_hinf((Acl, Bcl, Ccl, Dcl))
        print(f"H∞ norm = {hinf:.2f}")
    if not check_hinf_upper_bound((Acl, Bcl, Ccl, Dcl), gam):
        return Y
    # Covariance calculation
    Bw = B1
    Dwy = np.zeros((Cy.shape[0], Bw.shape[1]))
    if Ak is not None:
        Bcl = np.block([[Bw + By @ Dk @ Dwy], [Bk @ Dwy]])
    else:
        Bcl = Bw + By @ Dk @ Dwy
    P = sp.linalg.solve_continuous_lyapunov(Acl, -Bcl @ Bcl.T)
    Duz = np.array([[1.0]])
    Cz = np.array([[0.0]])
    if Ak is not None:
        Ccl = np.block([[Cz + Duz @ Dk @ Cy, Duz @ Ck]])
    else:
        Ccl = Cz + Duz @ Dk @ Cy
    # Check control effort
    if (U := Ccl @ P @ Ccl.T) > u_max**2:
        return Y + U[0, 0] - u_max**2
    if to_print:
        print(f"U = {np.sqrt(U[0, 0]):.2f} V")
    # Compute covar matrix of the angles
    Duz = np.array([[0.0, 0.0]]).T
    Cz = C1
    if Ak is not None:
        Ccl = np.block([[Cz + Duz @ Dk @ Cy, Duz @ Ck]])
    else:
        Ccl = Cz + Duz @ Dk @ Cy
    Z = Ccl @ P @ Ccl.T
    if to_print:
        print(f"σ[β] = {np.sqrt(Z[0, 0]) * 57.3 * 60 / v:.2f} arc min")
        print(f"σ[γ1] = {np.sqrt(Z[1, 1]) * 57.3 * 60:.2f} arc min")
        print(f"F = ({np.sqrt(np.trace(Z)) * 57.3 * 60:.2f} arc min)**2")
    # Return H_2 norm
    return np.trace(Z)


def penalty_fnc_ellips(Ak, Bk, Ck, Dk, A, B1, By, Cy, Dw, gam, u_max, to_print=False):
    """Penalty function for fixed structure optimal control synthesis
    based on ellipsoid criterion"""
    Y = 1000
    # Check closed loop poles
    if Ak is not None:
        Acl = np.block([[A + By @ Dk @ Cy, By @ Ck], [Bk @ Cy, Ak]])
    else:
        Acl = A + By @ Dk @ Cy
    if (max_eig := np.max(np.real(sp.linalg.eigvals(Acl)))) > -1e-3:
        return Y + max_eig
    x_max = -2 * max_eig
    # Check H_inf norm
    Bw = np.array([[0.0, 0.0, 0.0, 0.0]]).T
    Dwy = np.zeros((Cy.shape[0], Bw.shape[1]))
    Dwy[0, 0] = 1
    Cz = np.array([[1, 0, 0, 0]])
    Cz = Cz @ Si[:4, :4]
    Dwz = np.array([[0]])
    Duz = np.array([[0]])
    if Ak is not None:
        Bcl = np.block([[Bw + By @ Dk @ Dwy], [Bk @ Dwy]])
        Ccl = np.block([[Cz + Duz @ Dk @ Cy, Duz @ Ck]])
    else:
        Bcl = Bw + By @ Dk @ Dwy
        Ccl = Cz + Duz @ Dk @ Cy
    Dcl = Dwz + Duz @ Dk @ Dwy
    if to_print:
        hinf = calc_hinf((Acl, Bcl, Ccl, Dcl))
        print(f"H∞ norm = {hinf:.2f}")
    if not check_hinf_upper_bound((Acl, Bcl, Ccl, Dcl), gam):
        return Y
    # Ellispsoid estimation
    Bw = B1
    Dwy = Dw
    if Ak is not None:
        Bcl = np.block([[Bw + By @ Dk @ Dwy], [Bk @ Dwy]])
    else:
        Bcl = Bw + By @ Dk @ Dwy
    Duz = np.array([[0.0, 0.0]]).T
    Cz = C1[:, :4]
    if Ak is not None:
        Ccl = np.block([[Cz + Duz @ Dk @ Cy, Duz @ Ck]])
    else:
        Ccl = Cz + Duz @ Dk @ Cy

    def f(x):
        P = sp.linalg.solve_continuous_lyapunov(
            Acl + x / 2 * np.eye(*Acl.shape), -Bcl @ Bcl.T / x
        )
        # Check if P is PSD
        Z = Ccl @ P @ Ccl.T
        if np.any(np.diag(Z) < 0):
            return np.inf
        else:
            return np.trace(Z)

    # Minimize by zeta
    x = np.linspace(0.01, x_max, 100)
    y = [f(i) for i in x]
    y_min = min(y)
    if y_min == np.inf:
        return Y
    x_ = x[y.index(min(y))]
    P = sp.linalg.solve_continuous_lyapunov(
        Acl + x_ / 2 * np.eye(*Acl.shape), -Bcl @ Bcl.T / x_
    )
    # Check control effort
    Duz = np.array([[1.0]])
    Cz = np.array([[0.0]])
    if Ak is not None:
        Ccl = np.block([[Cz + Duz @ Dk @ Cy, Duz @ Ck]])
    else:
        Ccl = Cz + Duz @ Dk @ Cy
    if (U := Ccl @ P @ Ccl.T) > u_max**2:
        return Y + U[0, 0] - u_max**2
    if to_print:
        print(f"|u|max = {np.sqrt(U[0, 0]):.2f} V")
    if to_print:
        Duz = np.array([[0.0, 0.0]]).T
        Cz = C1[:, :4]
        if Ak is not None:
            Ccl = np.block([[Cz + Duz @ Dk @ Cy, Duz @ Ck]])
        else:
            Ccl = Cz + Duz @ Dk @ Cy
        Z = Ccl @ P @ Ccl.T
        print(f"|β|max = {np.sqrt(Z[0, 0]) * 57.3 * 60 / v:.2f} arc min")
        print(f"|γ1|max = {np.sqrt(Z[1, 1]) * 57.3 * 60:.2f} arc min")
        print(f"F = ({np.sqrt(y_min) * 57.3 * 60:.2f} arc min)**2")
        print(f"ζ = {x_:.2e}")
    return y_min


def sqr_FWBT(Wo, G, n: int):
    """Square root frequency-weighted balanced truncation"""
    # Augmented state space
    aA = np.block(
        [[G[0], np.zeros((G[0].shape[0], Wo[0].shape[1]))], [Wo[1] @ G[2], Wo[0]]]
    )
    aB = np.block([[G[1]], [Wo[1] @ G[3]]])
    aC = np.block([Wo[3] @ G[2], Wo[2]])
    aD = Wo[3] @ G[3]
    Ga = (aA, aB, aC, aD)
    # Gramians
    W = sp.linalg.solve_continuous_lyapunov(G[0], -G[1] @ G[1].T)
    W = 0.5 * (W + W.T)
    S = sp.linalg.cholesky(W + 1e-15 * np.eye(W.shape[0]), lower=False)
    S = S.T
    R = sp.linalg.solve_continuous_lyapunov(Ga[0].T, -Ga[2].T @ Ga[2])
    R = 0.5 * (R + R.T)
    R = sp.linalg.cholesky(R + 1e-15 * np.eye(R.shape[0]), lower=False)
    R = R[: G[0].shape[0], : G[0].shape[1]]
    # Hankel singlular values calculation
    U, E, V = sp.linalg.svd(R @ S, compute_uv=True)
    V = V.T
    r = sum([i > 1e-8 * E[0] for i in E])
    hsv = np.diag(E ** (-1 / 2))
    L = R.T @ U @ hsv
    T = S @ V @ hsv
    # Balance and transform state space
    bA = L.T @ G[0] @ T
    bB = L.T @ G[1]
    bC = G[2] @ T
    bD = Ga[3]
    # Truncate state space
    A11 = bA[:n, :n]
    A12 = bA[:n, n:r]
    A21 = bA[n:r, :n]
    A22 = bA[n:r, n:r]
    B1 = bB[:n, :r]
    B2 = bB[n:r, :r]
    C1 = bC[:r, :n]
    C2 = bC[:r, n:r]
    D = bD
    # Singular perturbation approximation to match dc-gain
    Ar = A11 - A12 @ sp.linalg.solve(A22, A21)
    Br = B1 - A12 @ sp.linalg.solve(A22, B2)
    Cr = C1 - C2 @ sp.linalg.solve(A22, A21)
    Dr = D - C2 @ sp.linalg.solve(A22, B2)

    return Ar, Br, Cr, Dr


def connect_series_siso(ss1, ss2):
    A1, B1, C1, D1 = ss1
    A2, B2, C2, D2 = ss2
    # Dimensions
    n1 = A1.shape[0] if A1 is not None else 0
    n2 = A2.shape[0] if A2 is not None else 0

    if n1 > 0 and n2 > 0:
        A = np.block([[A1, np.zeros((n1, n2))], [B2 @ C1, A2]])
        B = np.vstack([B1, B2 @ D1])
        C = np.hstack([D2 @ C1, C2])
        D = D2 @ D1
    elif n1 == 0 and n2 > 0:
        A = A2
        B = B2 @ D1
        C = C2
        D = D2 @ D1
    elif n2 == 0 and n1 > 0:
        A = A1
        B = B1
        C = D2 @ C1
        D = D2 @ D1
    else:
        A = np.zeros((0, 0))
        B = np.zeros((0, B1.shape[1]))
        C = np.zeros((D2.shape[0], 0))
        D = D2 @ D1

    return A, B, C, D


def save_ss(name: str, W: tuple, file_path=None):
    """Save controller's state space matrices in a json file"""
    if not file_path:
        file_path = Path(__file__).parent.resolve() / "sl_controller.json"
    if type(W) is tuple:
        A, B, C, D = W
    else:
        raise Exception("Wrong type of W")
    if A is None and D is None:
        raise Exception("Bad controller structure")
    try:
        file = open(file_path, "r+", newline="")
    except:
        file = open(file_path, "w+", newline="")
    finally:
        with file:
            try:
                controller_dict = json.load(file)
                if controller_dict is None:
                    controller_dict = {}
            except:
                controller_dict = {}
    controller_dict[name] = {
        "A": A.tolist() if A is not None else None,
        "B": B.tolist() if B is not None else None,
        "C": C.tolist() if C is not None else None,
        "D": D.tolist() if D is not None else None,
    }
    with open(file_path, "w", newline="") as file:
        json.dump(controller_dict, file, indent=4)


def load_ss(name: str, file_path=None):
    """Load controller's state space matrices from a json file"""
    if not file_path:
        file_path = Path(__file__).parent.resolve() / "sl_controller.json"
    with open(file_path, "r", newline="") as file:
        try:
            controller_dict = json.load(file)
            ss = controller_dict[name]
            return ss["A"], ss["B"], ss["C"], ss["D"]
        except:
            return None


def LQR_synthesis(A, B, Q, R, method="scipy"):
    """LQR synthesis via Algebraic Riccati equation"""
    if method == "scipy":
        R = np.atleast_2d(R)
        Po = sp.linalg.solve_continuous_are(A, B, Q, R)
        oL = Po @ sp.linalg.solve(R.T, B.T).T
        return oL
    elif method == "custom":
        Ham = np.block([[A, -B @ B.T / R], [-Q, -A.T]])
        _, U, _ = sp.linalg.schur(Ham, sort="lhp")
        m, n = U.shape
        U11 = U[0 : m // 2, 0 : n // 2]
        U21 = U[m // 2 : m, 0 : n // 2]
        Po = sp.linalg.solve(U11.T, U21.T).T
        oL = Po @ sp.linalg.solve(R.T, B.T).T
        return oL


def calc_hinf(ss, max_iter=200, gam_tol=1e-6):
    """Bisection algorithm for H inf norm calculation"""
    A, B, C, D = ss
    gam_upper = 1.0
    while True:
        if check_hinf_upper_bound((A, B, C, D), gam_upper):
            break
        gam_upper *= 2
    gam_lower = 0
    for _ in range(max_iter):
        gam = (gam_lower + gam_upper) / 2
        if check_hinf_upper_bound((A, B, C, D), gam):
            gam_upper = gam
        else:
            gam_lower = gam
        if gam_upper - gam_lower < gam_tol:
            return gam_upper
    return -1


def ss_d2c(ss):
    """Convert discrete state space to continuous via Tustin transform"""
    Ad, Bd, Cd, Dd = ss
    In = np.eye(len(Ad))
    Adinv = sp.linalg.inv(Ad + In)
    A = 2 * (Ad - In) @ Adinv
    B = 2 * Adinv @ Bd
    C = 2 * Cd @ Adinv
    D = Dd - Cd @ Adinv @ Bd
    return (np.atleast_2d(A), np.atleast_2d(B), np.atleast_2d(C), np.atleast_2d(D))


def ss2tf(ss):
    """Convert tate space to transfer function"""
    A, B, C, D = ss
    return sy.simplify(C * (s * sy.eye(A.shape[0]) - A).inv() * B + D)


def tf_to_num_den(W):
    num = sy.Poly(sy.together(W).as_numer_denom()[0], s).all_coeffs()
    den = sy.Poly(sy.together(W).as_numer_denom()[1], s).all_coeffs()
    return [float(i) for i in num], [float(i) for i in den]


def tf2ss(W):
    num, den = tf_to_num_den(W)
    num = np.asarray(num)
    den = np.asarray(den)
    num = num / den[0]
    den = den / den[0]
    n = len(den) - 1  # system order
    # Pad numerator
    m = len(num) - 1
    if m < n:
        num = np.hstack([np.zeros(n - m), num])
    A = np.zeros((n, n))
    A[0, :] = -den[1:]
    if n > 1:
        A[1:, :-1] = np.eye(n - 1)
    B = np.zeros((n, 1))
    B[0, 0] = 1.0
    D = num[0]
    C = (num[1:] - num[0] * den[1:]).reshape(1, -1)
    return np.atleast_2d(A), np.atleast_2d(B), np.atleast_2d(C), np.atleast_2d(D)


def dc_gain_ss(ss):
    A, B, C, D = ss
    if any(sp.linalg.eigvals(A) >= -1e-8):
        return np.inf
    return -C @ sp.linalg.inv(A) @ B + D


def bode_tf(W, w):
    num, den = tf_to_num_den(W)
    vec = []
    for w_ in w:
        num_ = [a * (1j * w_) ** i for i, a in enumerate(num[::-1])]
        den_ = [a * (1j * w_) ** i for i, a in enumerate(den[::-1])]
        vec.append(sum(num_) / sum(den_))
    mag = 20 * np.log10(np.abs(vec))
    phase = np.angle(vec, deg=True)
    phase = np.unwrap(phase)
    return w, mag, phase


def feedback_connect(sys1, sys2=None, sign=-1):
    A1, B1, C1, D1 = sys1

    if sys2 is None:
        # Only states from sys1
        A_cl = A1 + sign * (-B1 @ C1)
        B_cl = B1
        C_cl = C1 + sign * (-D1 @ C1)
        D_cl = D1
        return A_cl, B_cl, C_cl, D_cl

    A2, B2, C2, D2 = sys2
    # Check matrix invertibility
    M = np.eye(D2.shape[0]) + sign * (D2 @ D1)
    if np.linalg.matrix_rank(M) < M.shape[0]:
        raise ValueError("I + sign*D2*D1 is singular, feedback undefined.")
    Minv = np.linalg.inv(M)
    # Closed-loop matrices
    A_cl = np.block(
        [
            [A1 - sign * B1 @ Minv @ D2 @ C1, -sign * B1 @ Minv @ C2],
            [B2 @ (C1 - sign * D1 @ Minv @ D2 @ C1), A2 - sign * B2 @ D1 @ Minv @ C2],
        ]
    )
    B_cl = np.vstack([B1 @ Minv, B2 @ D1 @ Minv])
    C_cl = np.hstack([C1 - sign * D1 @ Minv @ D2 @ C1, -sign * D1 @ Minv @ C2])
    D_cl = D1 @ Minv

    return A_cl, B_cl, C_cl, D_cl


def get_obsvb_matrix(A, C):
    Obsv = C
    for i in range(1, A.shape[0]):
        Obsv = np.vstack((Obsv, C @ np.linalg.matrix_power(A, i)))
    return Obsv


def get_ctrlb_matrix(A, B):
    Ctrl = B
    for i in range(1, A.shape[1]):
        Ctrl = np.hstack((Ctrl, np.linalg.matrix_power(A, i) @ B))
    return Ctrl
