import sys
from pathlib import Path
import numpy as np
import cvxpy as cvx
import scipy as sp
import matplotlib.pyplot as plt
from src import A, C2, B2, C1, v, W_ol, H, J2, J1, Ce, L, S, Si
from src import M_f, D_trq, D_dtheta, D_dgam
from src import sqr_FWBT, save_ss, LQR_synthesis, feedback_connect
from src import dc_gain_ss, ss2tf, tf_to_num_den, get_obsvb_matrix
from src import get_ctrlb_matrix, connect_series_siso, calc_hinf
import plot_style

plot_style.set_plot_style()

# State space

Md_max = M_f + 2 * np.sqrt(D_trq)
B1 = np.array(
    [
        [2 * np.sqrt(D_dtheta), 0, 0],
        [0, 0, 0],
        [0, Md_max / J1, 0],
        [0, 0, 2 * np.sqrt(D_dgam) * Ce / L],
    ]
)

B2 = B2[:4, :]

C1 = C1[:, :4]

C2 = C2[:, :4]

S = S[:4, :4]
Si = Si[:4, :4]

B1 = S @ B1

##### Control synthesis #####


print("".center(79, "#"))
print("Ellipsoid-controller via LMI")
eigs_obsvb = sp.linalg.eigvals(get_obsvb_matrix(A, C2))
eigs_ctrlb = sp.linalg.eigvals(get_ctrlb_matrix(A, B2))
print("Obsvb matrix rank= ", np.sum(np.abs(eigs_obsvb) > 1e-10))
print("Ctrlb matrix rank= ", np.sum(np.abs(eigs_ctrlb) > 1e-10))

λmin = 600
u_max = np.array([[24]])


def f(ζ):
    P = cvx.Variable((4, 4), symmetric=True)
    Z = cvx.Variable((2, 2), symmetric=True)
    W = cvx.Variable((1, 4))

    constr = [P >> 0]
    constr += [Z >> 0]
    constr += [(A @ P + B2 @ W) + (A @ P + B2 @ W).T + ζ * P + B1 @ B1.T / ζ << 0]
    constr += [cvx.bmat([[u_max**2, W], [W.T, P]]) >> 0]
    constr += [(A @ P + B2 @ W) + (A @ P + B2 @ W).T + 2 * λmin * P >> 0]
    constr += [cvx.bmat([[Z, C1 @ P], [(C1 @ P).T, P]]) >> 0]
    opt = cvx.Problem(cvx.Minimize(1e5 * cvx.trace(Z)), constr)
    try:
        opt.solve(solver=cvx.CVXOPT)
    except:
        return np.inf
    if opt.status == "optimal":
        return opt.value * 1e-5
    return np.inf


x = range(1, λmin, 1)
y = [f(i) for i in x]
ζ = x[y.index(min(y))]
fig, ax = plt.subplots(1, 1, figsize=(4.1, 3.1))
ax.set_yscale("log")
ax.plot(x, y, c="k")
ax.set_ylabel(r"$F_2$")
ax.set_xlabel(r"$\mathrm{\zeta}$")
ax.grid()
ax.grid(which="both", axis="both")
if len(sys.argv) == 2:
    plot_dir_path = Path(sys.argv[1])
    if not plot_dir_path.is_dir():
        plot_dir_path.mkdir()
    plt.savefig(plot_dir_path / "ellips_zeta.svg")

print("Control minimizer ζ = %i" % ζ)


P = cvx.Variable((4, 4), symmetric=True)
Z = cvx.Variable((2, 2), symmetric=True)
W = cvx.Variable((1, 4))
constr = [P >> 0]
constr += [Z >> 0]
constr += [(A @ P + B2 @ W) + (A @ P + B2 @ W).T + ζ * P + B1 @ B1.T / ζ << 0]
constr += [cvx.bmat([[u_max**2, W], [W.T, P]]) >> 0]
constr += [(A @ P + B2 @ W) + (A @ P + B2 @ W).T + 2 * λmin * P >> 0]
constr += [cvx.bmat([[Z, C1 @ P], [(C1 @ P).T, P]]) >> 0]
opt = cvx.Problem(cvx.Minimize(1e5 * cvx.trace(Z)), constr)
opt.solve(solver=cvx.CVXOPT)
K = sp.linalg.solve(P.value.T, W.value.T).T
print(f"K= {str(K @ S)}")

##### Observer synthesis #####
oR = 5e-14
oL = LQR_synthesis(A.T, C2.T, B1 @ B1.T, oR)
Ak = A + B2 @ K - oL @ C2
Bk = oL
Ck = K
Dk = np.zeros((1, 1))
Wf = (Ak, Bk, Ck, Dk)
##### Reduce order #####
# Frequency weighting filter
Wfw = feedback_connect(W_ol, Wf, -1)
# Reduced order controller
Wfr = sqr_FWBT(Wfw, Wf, 2)

##### Variance calculation #####
print("Limit of variances w/ state feedback")
print("Limit variances w/ state feedback")
print(f"F= {opt.value: .2e}")
print(f"β=  {np.sqrt(Z.value[0, 0]) * 60 * 57.3 / v:.2f}")
print(f"γ1= {np.sqrt(Z.value[1, 1]) * 60 * 57.3:.2f}")
print(f"u= {np.sqrt(K @ P.value @ K.T)[0][0]:.2f}")
print(f"kdc = {dc_gain_ss(Wf)[0][0]:.2f}")
vA = np.block([[A, B2 @ K], [oL @ C2, A + B2 @ K - oL @ C2]])
vD = np.block([[B1], [np.zeros((4, 3))]])
vC = np.array([[1, 0, 0, 0], [0, -J2 / H, 0, 0]])
vC = vC @ Si
vC = np.block([[vC], [K]])
vC = np.block([vC, np.zeros((3, 4))])
print("Real variances w/ observer")


def f(ζ):
    vQ = sp.linalg.solve_continuous_lyapunov(vA + ζ / 2 * np.eye(*vA.shape), -1 / ζ * vD @ vD.T)
    return np.trace(vC @ vQ @ vC.T)


x = range(1, int(-2 * np.max(np.real(sp.linalg.eigvals(vA)))), 1)
y = [f(i) for i in x]
ζ = x[y.index(min(y))]
print("Observer minimizer η = %i" % ζ)
vQ = sp.linalg.solve_continuous_lyapunov(vA + ζ / 2 * np.eye(*vA.shape), -1 / ζ * vD @ vD.T)
z = vC @ vQ @ vC.T
print(f"F= {z[0, 0] + z[1, 1] * v**2:.2e}")
print("|β|max= %.2f'" % (np.sqrt(z[0, 0]) * 60 * 57.3))
print("|γ1|max= %.2f'" % (np.sqrt(z[1, 1]) * 60 * 57.3))
print("|u|max= %.2f V" % (np.sqrt(z[2, 2])))
# Variances with reduced order
print("Real variances w/ observer for reduced order")
vA = np.block([[A + B2 @ Wfr[3] @ C2, B2 @ Wfr[2]], [Wfr[1] @ C2, Wfr[0]]])
vD = np.block([[B1], [np.zeros((2, 3))]])
vC = np.array([[1, 0, 0, 0], [0, -J2 / H, 0, 0]])
vC = vC @ Si
vC = np.block([[vC], [Wfr[3] @ C2]])
vC = np.block([vC, np.zeros((3, 2))])
vC[2, 4:] = Wfr[2]


def f1(ζ):
    vQ = sp.linalg.solve_continuous_lyapunov(vA + ζ / 2 * np.eye(*vA.shape), -1 / ζ * vD @ vD.T)
    Z = np.trace(vC @ vQ @ vC.T)
    return Z


x = range(1, int(-2 * np.max(np.real(sp.linalg.eigvals(vA)))), 1)
y = [f1(i) for i in x]
ζ = x[y.index(min(y))]
print("Observer minimizer η = %i" % ζ)
vQ = sp.linalg.solve_continuous_lyapunov(vA + ζ / 2 * np.eye(*vA.shape), -1 / ζ * vD @ vD.T)
Z = np.trace(vC @ vQ @ vC.T)
print(f"F= {z[0, 0] + z[1, 1] * v**2:.2e}")
print(f"kdc = {dc_gain_ss(Wfr)[0][0]:.2f}")
print("|β|max= %.2f'" % (np.sqrt(z[0, 0]) * 60 * 57.3))
print("|γ1|max= %.2f'" % (np.sqrt(z[1, 1]) * 60 * 57.3))
print("|u|max= %.2f V" % (np.sqrt(z[2, 2])))

save_ss("ellips_obsv_k", Wf)
save_ss("ellips_obsv_k_r", Wfr)
save_ss(
    "ellips_obsv_g",
    (A, B2, C2, np.zeros((1, 1))),
)

Wfr = ss2tf(Wfr)[0]
num, den = tf_to_num_den(Wfr)
kdc = num[2] / den[2]
T1 = (num[0] / num[2]) ** 0.5
ksi1 = num[1] / num[2] / T1 / 2
T2 = (den[0] / den[2]) ** 0.5
ksi2 = den[1] / den[2] / T2 / 2
print(f"kdc= {kdc:.2e}; T1= {T1:.2e}; ξ1= {ksi1:.2e}; T2= {T2:.2e}; ξ2= {ksi2:.2e}")

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
    plt.show()
