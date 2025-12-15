import numpy as np
import cvxpy as cvx
import scipy as sp
import matplotlib.pyplot as plt
from src import H, J2, J1, Ce, L, Cm, T_dtheta, T_trq, W_ol, v
from src import kl, b_dgam, a_dgam, R
from src import D_trq, D_dtheta, D_dgam
from src import sqr_FWBT, save_ss, LQR_synthesis, feedback_connect
from src import dc_gain_ss, ss2tf, tf_to_num_den, get_obsvb_matrix
from src import get_ctrlb_matrix, connect_series_siso, calc_hinf

T = 8.6
print(T)
A = np.array(
    [
        [-1 / T, -1 / T, 0, 0, 0],
        [0, 0, -1, 0, 0],
        [0, 0, 0, -H / J2, 0],
        [0, 0, H / J1, 0, Cm / J1],
        [0, 0, 0, -Ce / L, -R / L],
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
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 1 / J1, kl / J1, 0],
        [0, 0, Ce / L, 0],
    ]
)
Aa = np.block(
    [
        [A, Csf],
        [np.zeros((4, 5)), Asf],
    ]
)
B1 = np.array(
    [
        [0, 0, 0, 0, 0, 0, np.sqrt(2 * D_trq / T_trq), 0, 0],
        [0, 0, 0, 0, 0, np.sqrt(2 * D_dtheta / T_dtheta), 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, np.sqrt(2 * a_dgam * D_dgam) / b_dgam],
    ]
).T
B2 = np.array([[0, 0, 0, 0, 1 / L, 0, 0, 0, 0]]).T
C1 = np.array(
    [
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, -J2 / H, 0, 0, 0, 0, 0, 0],
    ]
)
C2 = np.array([[0, -1, 0, 0, 0, 0, 0, 0, 0]])
D2 = np.zeros((1, 1))

Si = [10 / 57.3 / 60.0, 10 / 57.3 / 60.0, 0.1, 0.3, 1, 0.1, 1e-3, 1, 0.33 / 5e-4]
S = np.diag([1 / i for i in Si])
Si = np.diag(Si)

A, Aa, B1, B2, C1, C2 = (
    S[:5, :5] @ A @ Si[:5, :5],
    S @ Aa @ Si,
    S @ B1,
    S @ B2,
    C1 @ Si,
    C2 @ Si,
)

print("".center(79, "#"))
print("H2-controller via LMI")
eigs_obsvb = sp.linalg.eigvals(get_obsvb_matrix(Aa, C2))
eigs_ctrlb = sp.linalg.eigvals(get_ctrlb_matrix(Aa, B2))
print("Obsvb matrix rank= ", np.sum(np.abs(eigs_obsvb) > 1e-10))
print("Ctrlb matrix rank= ", np.sum(np.abs(eigs_ctrlb) > 1e-10))

##### Control synthesis #####
λmin = 600
u_max = np.array([[24 / 2]])

P = cvx.Variable((9, 9), symmetric=True)
Z = cvx.Variable((2, 2), symmetric=True)
W = cvx.Variable((1, 9))

constr = [P >> 0]
constr += [Z >> 0]
constr += [(Aa @ P + B2 @ W) + (Aa @ P + B2 @ W).T + B1 @ B1.T << 0]
constr += [cvx.bmat([[Z, C1 @ P], [(C1 @ P).T, P]]) >> 0]
constr += [cvx.bmat([[u_max**2, W], [W.T, P]]) >> 0]
constr += [(Aa @ P + B2 @ W) + (Aa @ P + B2 @ W).T + 2 * λmin * P >> 0]
opt = cvx.Problem(cvx.Minimize(1e5 * cvx.trace(Z)), constr)
opt.solve(
    solver=cvx.CVXOPT,
)

K = sp.linalg.solve(P.value.T, W.value.T).T
print(f"K= {str(K @ S)}")

##### Observer synthesis #####
oR = 5e-14
oL = LQR_synthesis(Aa.T, C2.T, B1 @ B1.T, oR)
Ak = Aa + B2 @ K - oL @ C2
Bk = oL
Ck = K
Dk = np.zeros((1, 1))
Wf = (Ak, Bk, Ck, Dk)

##### Variance calculation #####
print("Limit variances w/ state feedback")
print(f"F= {opt.value: .2e}")
print(f"σ[β]=  {np.sqrt(Z.value[0, 0]) * 60 * 57.3 / v:.2f}")
print(f"σ[γ1]= {np.sqrt(Z.value[1, 1]) * 60 * 57.3:.2f}")
print(f"σ[U]= {np.sqrt(K @ P.value @ K.T)[0][0]:.2f}")
vA = np.block([[Aa, B2 @ K], [oL @ C2, Aa + B2 @ K - oL @ C2]])
vD = np.block([[B1], [np.zeros((9, 3))]])
vC = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, -J2 / H, 0, 0, 0, 0, 0, 0]])
vC = vC @ Si
vC = np.block([[vC], [K]])
vC = np.block([vC, np.zeros((3, 9))])
vQ = sp.linalg.solve_continuous_lyapunov(vA, -vD @ vD.T)
z = vC @ vQ @ vC.T
print("Real variances w/ observer")
print(f"F= {z[0, 0] + z[1, 1] * v**2:.2e}")
print(f"kdc = {dc_gain_ss(Wf)[0][0]:.2f}")
print("β= %.2f'" % (np.sqrt(z[0, 0]) * 60 * 57.3))
print("γ1= %.2f'" % (np.sqrt(z[1, 1]) * 60 * 57.3))
print("U= %.2f V" % (np.sqrt(z[2, 2])))


save_ss("h2_corr_obsv_k", Wf)
save_ss(
    "h2_corr_obsv_g",
    (Aa[:5, :5], B2[:5, :], C2[:, :5], np.zeros((1, 1))),
)


if __name__ == "__main__":
    Wf = (Ak, Bk, Ck, Dk)
    w = np.logspace(-1, 3.6, 500)
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
