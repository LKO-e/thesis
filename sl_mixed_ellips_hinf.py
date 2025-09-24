import sys
import numpy as np
import scipy as sp
import cvxpy as cvx
import matplotlib.pyplot as plt
from src import A, C2, B2, C1, v, W_ol, J1, Ce, L, S, Si
from src import M_f, D_trq, D_dθ, D_dgam
from src import save_ss, connect_series_siso, ss2tf, dc_gain_ss, calc_hinf, feedback_connect
import plot_style

plot_style.set_plot_style()

match input("Use h inf constraint? Y/n: ").capitalize():
    case "Y" | "":
        is_h_inf_constr = True
    case _:
        is_h_inf_constr = False

print("".center(79, "#"))
print("Mixed ellips/h_inf synthesis via LMI")


def lmi_ellipsoid(constr, ζ, X, Y, Ah, Bh, Ch, Dh, A, B, C, Cj, Bj, Fj, Ej):
    """
    Add a constraint in the form of a limit ellipsoid with the matrix Z
    as LMIs with the paramter ζ

    :param: list constr: A list of LMIs of the problem
    :param: float ζ: A variable scalar parameter paramter
    :param: cvx.Variable X: An auxiliary matrix for optimization
    :param: cvx.Variable Y: An auxiliary matrix for optimization
    :param: cvx.Variable Ah: A transformed optimization variable
    :param: cvx.Variable Bh: A transformed optimization variable
    :param: cvx.Variable Ch: A transformed optimization variable
    :param: cvx.Variable Dh: A transformed optimization variable
    :param: cvx.Variable A: System's dynamic matrix
    :param: cvx.Variable B: Open loop input control matrix
    :param: cvx.Variable C: System's measurable output
    :param: cvx.Variable Bj: Closed loop dinturbance matrix
    :param: cvx.Variable Cj: Closed loop system quality output matrix
    :param: cvx.Variable Dj: Closed loop disturbance to quality throughput
    :param: cvx.Variable Ej: Control action to quality vector matrix
    :param: cvx.Variable Fj: Measurement noise matrix
    :return: cvx.Variable Z: A matrix of the limit ellipsoid

    dx/dt = A * x + Bj * w + B * u
    z = Cj * x + Dj * w + Ej * u
    y = C * x + Fj * w
    """
    Z = cvx.Variable((Cj.shape[0], Cj.shape[0]), name="Z", symmetric=True)
    constr += [
        cvx.bmat(
            [
                [X, np.eye(*A.shape), (Cj @ X + Ej @ Ch).T],
                [np.eye(*A.shape), Y, (Cj + Ej @ Dh @ C).T],
                [Cj @ X + Ej @ Ch, Cj + Ej @ Dh @ C, Z],
            ]
        )
        >> 0
    ]
    Ej = np.array([[1.0]])
    u_max2 = np.array([[24**2]])
    constr += [
        cvx.bmat(
            [
                [X, np.eye(*A.shape), (Ej @ Ch).T],
                [np.eye(*A.shape), Y, (Ej @ Dh @ C).T],
                [Ej @ Ch, Ej @ Dh @ C, u_max2],
            ]
        )
        >> 0
    ]
    constr += [
        cvx.bmat(
            [
                [
                    (A @ X + B @ Ch) + (A @ X + B @ Ch).T + ζ * X,
                    Ah.T + (A + B @ Dh @ C) + ζ * np.eye(*A.shape),
                    Bj + B @ Dh @ Fj,
                ],
                [
                    Ah + (A + B @ Dh @ C).T + ζ * np.eye(*A.shape),
                    (Y @ A + Bh @ C) + (Y @ A + Bh @ C).T + ζ * Y,
                    Y @ Bj + Bh @ Fj,
                ],
                [
                    (Bj + B @ Dh @ Fj).T,
                    (Y @ Bj + Bh @ Fj).T,
                    -ζ * np.eye(Bj.shape[1], Bj.shape[1]),
                ],
            ]
        )
        << 0
    ]
    return Z


def lmi_hinf_norm(constr, μ, X, Y, Ah, Bh, Ch, Dh, A, B, C, Cj, Bj, Dj, Fj, Ej):
    """
    Add a constraint in the form of an H inf norm μ

    :param: list constr: A list of LMIs of the problem
    :param: float μ: An H inf norm value
    :param: nd.array X: An auxiliary matrix for optimization
    :param: nd.array Y: An auxiliary matrix for optimization
    :param: matrix Ah: A transformed optimization variable
    :param: matrix Bh: A transformed optimization variable
    :param: matrix Ch: A transformed optimization variable
    :param: matrix Dh: A transformed optimization variable
    :param: nd.array A: System's dynamic matrix
    :param: nd.array B: Open loop input control matrix
    :param: nd.array C: System's measurable output
    :param: nd.array Bj: Closed loop dinturbance matrix
    :param: nd.array Cj: Closed loop system quality output matrix
    :param: nd.array Dj: Closed loop disturbance to quality throughput
    :param: nd.array Ej: Control action to quality vector matrix
    :param: nd.array Fj: Measurement noise matix

    dx/dt = A * x + Bj * w + B * u
    z = Cj * x + Dj * w + Ej * u
    y = C * x + Fj * w
    """
    constr += [
        cvx.bmat(
            [
                [
                    (A @ X + B @ Ch) + (A @ X + B @ Ch).T,
                    Ah.T + (A + B @ Dh @ C),
                    (Bj + B @ Dh @ Fj),
                    (Cj @ X + Ej @ Ch).T,
                ],
                [
                    Ah + (A + B @ Dh @ C).T,
                    (Y @ A + Bh @ C) + (Y @ A + Bh @ C).T,
                    (Y @ Bj + Bh @ Fj),
                    (Cj + Ej @ Dh @ C).T,
                ],
                [
                    (Bj + B @ Dh @ Fj).T,
                    (Y @ Bj + Bh @ Fj).T,
                    -(μ**2) * np.eye(Bj.shape[1], Bj.shape[1]),
                    (Dj + Ej @ Dh @ Fj).T,
                ],
                [
                    (Cj @ X + Ej @ Ch),
                    (Cj + Ej @ Dh @ C),
                    (Dj + Ej @ Dh @ Fj),
                    -np.eye(Cj.shape[0], Cj.shape[0]),
                ],
            ]
        )
        << 0
    ]


def controller_recovery(problem, A, B, C):
    """
    Recovers controller from optimization solution
    :param: cvx.Problem problem: A cxv optimization problem
    :param: np.ndarray A: A dynamics matrix of the plant
    :param: np.ndarray B: An input matrix of the plant
    :param: np.ndarray C: A measurement matrix of the plant
    :return (Ak, Bk, Ck, Dk): State space matrices of the controller
    """

    X = problem.var_dict["X"].value
    Y = problem.var_dict["Y"].value
    Ah = problem.var_dict["Ah"].value
    Bh = problem.var_dict["Bh"].value
    Ch = problem.var_dict["Ch"].value
    Dh = np.zeros((Ej.shape[1], C.shape[0]))
    if "Dk" in problem.var_dict:
        Dh = problem.var_dict["Dh"].value
    else:
        Dh = np.zeros((Ej.shape[1], C.shape[0]))
    u, s, vh = sp.linalg.svd(np.eye(*X.shape) - X @ Y)
    Mti = u @ np.diag(np.sqrt(1.0 / s))
    Ni = np.diag(np.sqrt(1.0 / s)) @ vh
    Dk = Dh
    Ck = (Ch - Dh @ C @ X) @ Mti
    Bk = Ni @ (Bh - Y.T @ B @ Dk)
    Ak = Ni @ (Ah - Bh @ C @ X - Y @ A @ X - Y @ B @ Ch + Y @ B @ Dh @ C @ X) @ Mti
    return Ak, Bk, Ck, Dk


# State space
Md_max = M_f + 2 * np.sqrt(D_trq)
B1 = np.array(
    [
        [2 * np.sqrt(D_dθ), 0, 0, 0],
        [0, 0, 0, 0],
        [0, Md_max / J1, 0, 0],
        [0, 0, 2 * np.sqrt(D_dgam) * Ce / L, 0],
    ]
)
B1 = S[:4, :4] @ B1
B2 = B2[:4, :]
C1 = C1[:, :4]
C2 = C2[:, :4]

# Optimization problem statement
X = cvx.Variable(A.shape, name="X", symmetric=True)
Y = cvx.Variable(A.shape, name="Y", symmetric=True)
Ah = cvx.Variable(A.shape, name="Ah")
Bh = cvx.Variable((A.shape[0], C2.shape[0]), name="Bh")
Ch = cvx.Variable((B2.shape[1], Ah.shape[1]), name="Ch")
Dh = np.zeros((Ch.shape[0], Bh.shape[1]))
# Dh = cvx.Variable((Ch.shape[0], Bh.shape[1]), name="Dh")
ζ = cvx.Parameter(name="ζ")
t = 1
constr = []
constr += [cvx.bmat([[X, t * np.eye(*A.shape)], [t * np.eye(*A.shape), Y]]) >> 0]
Fj = np.array([[0.0, 0.0, 0.0, 0.5]]) / 57.3 / 60.0
Ej = np.array([[0.0]])
Z = lmi_ellipsoid(constr, ζ, X, Y, Ah, Bh, Ch, Dh, A, B2, C2, C1, B1, Fj, Ej)
r_pole = 3000
Lp = -r_pole * np.eye(2, 2)
Mp = np.array([[0, 1], [0, 0]])
constr += [
    cvx.kron(
        Lp,
        cvx.bmat(
            [[X, np.eye(X.shape[0], X.shape[0])], [np.eye(Y.shape[0], Y.shape[0]), Y]]
        ),
    )
    + cvx.kron(
        Mp, cvx.bmat([[A @ X + B2 @ Ch, A + B2 @ Dh @ C2], [Ah, Y @ A + Bh @ C2]])
    )
    + cvx.kron(
        Mp.T, cvx.bmat([[A @ X + B2 @ Ch, A + B2 @ Dh @ C2], [Ah, Y @ A + Bh @ C2]]).T
    )
    << 0
]
μ = 1.5
Cj1 = np.array([[1, 0, 0.0, 0.0]])
Bj1 = np.array([[0.0, 0.0, 0.0, 0.0]]).T
Fj1 = np.array([[10]]) / 57.3 / 60
Ej1 = np.array([[0.0]])
Dj1 = np.array([[0.0]])
if is_h_inf_constr:
    constr += [
        cvx.bmat(
            [
                [
                    (A @ X + B2 @ Ch) + (A @ X + B2 @ Ch).T,
                    Ah.T + (A + B2 @ Dh @ C2),
                    (Bj1 + B2 @ Dh @ Fj1),
                    (Cj1 @ X + Ej1 @ Ch).T,
                ],
                [
                    Ah + (A + B2 @ Dh @ C2).T,
                    (Y @ A + Bh @ C2) + (Y @ A + Bh @ C2).T,
                    (Y @ Bj1 + Bh @ Fj1),
                    (Cj1 + Ej1 @ Dh @ C2).T,
                ],
                [
                    (Bj1 + B2 @ Dh @ Fj1).T,
                    (Y @ Bj1 + Bh @ Fj1).T,
                    -(μ**2) * np.eye(Bj1.shape[1], Bj1.shape[1]),
                    (Dj1 + Ej1 @ Dh @ Fj1).T,
                ],
                [
                    (Cj1 @ X + Ej1 @ Ch),
                    (Cj1 + Ej1 @ Dh @ C2),
                    (Dj1 + Ej1 @ Dh @ Fj1),
                    -np.eye(Cj1.shape[0], Cj1.shape[0]),
                ],
            ]
        )
        << 0
    ]
problem = cvx.Problem(cvx.Minimize(1e5 * cvx.trace(Z)), constraints=constr)
# Solving the optimization problem
if len(sys.argv) == 2:
    solver_ = sys.argv[1].upper()
    if solver_ == "MOSEK":
        solver = cvx.MOSEK
    elif solver_ == "CVXOPT":
        solver = cvx.CVXOPT
    elif solver_ == "CLARABEL":
        solver = cvx.CLARABEL
    elif solver_ == "SDPA":
        solver = cvx.SDPA
    else:
        print("Defaulting to CVXOPT")
        solver = cvx.CVXOPT
else:
    print("Defaulting to CVXOPT")
    solver = cvx.CVXOPT
x = np.arange(0.1, 500, 5)
opt_f_vals = np.ones_like(x) * np.inf
for i, ζ in enumerate(x):
    problem.param_dict["ζ"].value = ζ
    try:
        problem.solve(solver=solver)
    except:
        opt_f_vals[i] = np.inf
    if problem.status != "optimal":
        opt_f_vals[i] = np.inf
    else:
        opt_f_vals[i] = problem.value
if np.argmin(opt_f_vals) == np.inf:
    print("No solution found")
    sys.exit()
    quit()
ζ = x[np.argmin(opt_f_vals)]
print(f"ζ= {ζ:0.2E}")
problem.param_dict["ζ"].value = ζ
problem.solve(solver=solver)
print(f"Optimization result=  {problem.value:0.2E}")
Z = problem.var_dict["Z"].value
print(f"β= {Z[0, 0] ** 0.5:.2f} arc min")
print(f"γ= {Z[1, 1] ** 0.5 / v:.2f} arc min")
# print(f"u= {Z[2, 2]**0.5:.2f} V")
np.set_printoptions(formatter={"float": lambda x: "{0:0.2e}".format(x)})

# State space open loop
print(sp.linalg.eigvals(W_ol[0]))
# Controller recovery
Ak, Bk, Ck, Dk = controller_recovery(problem, A, B2, C2)
print("eig(Ak) =")
print(sp.linalg.eigvals(Ak))
Wf = (Ak, Bk, Ck, Dk)
print(ss2tf(Wf))
print(dc_gain_ss(Wf))

K = Ak, Bk, Ck, Dk
if is_h_inf_constr:
    save_ss("mixed_syn_w_hinf_k", K)
    save_ss(
        "mixed_syn_w_hinf_g",
        (A[:4, :4], B2[:4, :], C2[:, :4], np.zeros((1, 1))),
    )
else:
    save_ss("mixed_syn_w_o_hinf_k", K)
    save_ss(
        "mixed_syn_w_o_hinf_g",
        (A[:4, :4], B2[:4, :], C2[:, :4], np.zeros((1, 1))),
    )

if __name__ == "__main__":
    Wf = (Ak, Bk, Ck, Dk)
    w = np.logspace(1, 3.6, 500)
    if Ak is not None:
        print("Dc-gain = ", dc_gain_ss(Wf)[0])
    # Open-loop Bode plots
    fig, ax = plt.subplots(2, 1)
    fig.canvas.manager.set_window_title("Open-loop bode plots")
    ax[0].grid("on")
    ax[1].grid("on")
    if Ak is not None:
        _, mag, phase = sp.signal.bode(Wf, w)
        ax[0].semilogx(w, mag, label="K")
        ax[1].semilogx(w, phase, label="K")
    W_ol1 = (W_ol[0], W_ol[1], -W_ol[2], W_ol[3])
    _, mag, phase = sp.signal.bode(W_ol1, w)
    ax[0].semilogx(w, mag, label="-G")
    ax[1].semilogx(w, phase, label="-G")
    W_ol1 = connect_series_siso(Wf, W_ol)
    _, mag, phase = sp.signal.bode(W_ol1, w)
    ax[0].semilogx(w, mag, label="-GK")
    ax[1].semilogx(w, phase, label="-GK")
    ax[0].legend()
    ax[1].legend()
    # Closed-loop Bode plots
    W_cl = feedback_connect(W_ol1, sign=-1)
    fig, ax = plt.subplots(2, 1)
    fig.canvas.manager.set_window_title("Closed-loop bode plot")
    ax[0].grid("on")
    ax[1].grid("on")
    _, mag, phase = sp.signal.bode(W_cl, w)
    ax[0].semilogx(w, mag)
    ax[1].semilogx(w, phase)
    print(f"H∞ norm = {calc_hinf(W_cl):.2e}")
    # Step plot
    fig, ax = plt.subplots(1, 1)
    fig.canvas.manager.set_window_title("Unit step response plot")
    ax.grid("on")
    t = np.arange(0, 7.0, 1e-5)
    u = np.ones_like(t)
    _, y, _ = sp.signal.lsim(W_cl, u, t)
    ax.plot(t, y)
    ax.grid("on")
    # An optimization curve plot
    fig, ax = plt.subplots(1, 1, figsize=(4.1, 3.1))
    ax.set_yscale("log")
    ax.plot(x, opt_f_vals, c="k")
    ax.set_ylabel(r"$F$")
    ax.set_xlabel(r"$\mathrm{\zeta}$")
    ax.grid()
    ax.grid(which="both", axis="both")
    plt.show()
