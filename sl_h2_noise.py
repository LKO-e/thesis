import numpy as np
import cvxpy as cvx
import scipy as sp
import matplotlib.pyplot as plt
from src import Aa, C2, B2, v, C1, B1, W_ol, H, J2, Si, S
from src import sqr_FWBT, save_ss, LQR_synthesis, feedback_connect
from src import dc_gain_ss, ss2tf, tf_to_num_den, get_obsvb_matrix
from src import get_ctrlb_matrix, connect_series_siso, calc_hinf

print("".center(79, "#"))
print("H2-controller via LMI")
eigs_obsvb = sp.linalg.eigvals(get_obsvb_matrix(Aa, C2))
eigs_ctrlb = sp.linalg.eigvals(get_ctrlb_matrix(Aa, B2))
print("Obsvb matrix rank= ", np.sum(np.abs(eigs_obsvb) > 1e-10))
print("Ctrlb matrix rank= ", np.sum(np.abs(eigs_ctrlb) > 1e-10))

##### Control synthesis #####
λmin = 600
u_max = np.array([[24 / 2]])

P = cvx.Variable((8, 8), symmetric=True)
Z = cvx.Variable((2, 2), symmetric=True)
W = cvx.Variable((1, 8))

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
oR = 5e-11
oL = LQR_synthesis(Aa.T, C2.T, B1 @ B1.T, oR)
Ak = Aa + B2 @ K - oL @ C2
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
print("Limit variances w/ state feedback")
print(f"F= {opt.value: .2e}")
print(f"σ[β]=  {np.sqrt(Z.value[0, 0]) * 60 * 57.3 / v:.2f}")
print(f"σ[γ1]= {np.sqrt(Z.value[1, 1]) * 60 * 57.3:.2f}")
print(f"σ[U]= {np.sqrt(K @ P.value @ K.T)[0][0]:.2f}")
vA = np.block([[Aa, B2 @ K], [oL @ C2, Aa + B2 @ K - oL @ C2]])
vD = np.block([[B1], [np.zeros((8, 3))]])
vC = np.array([[1, 0, 0, 0, 0, 0, 0, 0], [0, -J2 / H, 0, 0, 0, 0, 0, 0]])
vC = vC @ Si
vC = np.block([[vC], [K]])
vC = np.block([vC, np.zeros((3, 8))])
vQ = sp.linalg.solve_continuous_lyapunov(vA, -vD @ vD.T)
z = vC @ vQ @ vC.T
print("Real variances w/ observer")
print(f"F= {z[0, 0] + z[1, 1] * v**2:.2e}")
print(f"kdc = {dc_gain_ss(Wf)[0][0]:.2f}")
print("β= %.2f'" % (np.sqrt(z[0, 0]) * 60 * 57.3))
print("γ1= %.2f'" % (np.sqrt(z[1, 1]) * 60 * 57.3))
print("U= %.2f V" % (np.sqrt(z[2, 2])))
# Variances with reduced order
print("Real variances w/ observer for reduced order")
vA = np.block([[Aa + B2 @ Wfr[3] @ C2, B2 @ Wfr[2]], [Wfr[1] @ C2, Wfr[0]]])
vD = np.block([[B1], [np.zeros((2, 3))]])
vC = np.array([[1, 0, 0, 0, 0, 0, 0, 0], [0, -J2 / H, 0, 0, 0, 0, 0, 0]])
vC = vC @ Si
vC = np.block([[vC], [Wfr[3] @ C2]])
vC = np.block([vC, np.zeros((3, 2))])
vC[2, 8:] = Wfr[2]
vQ = sp.linalg.solve_continuous_lyapunov(vA, -vD @ vD.T)
z = vC @ vQ @ vC.T
print(f"F= {z[0, 0] + z[1, 1] * v**2:.2e}")
print(f"kdc = {dc_gain_ss(Wfr)[0][0]:.2f}")
print("β= %.2f'" % (np.sqrt(z[0, 0]) * 60 * 57.3))
print("γ1= %.2f'" % (np.sqrt(z[1, 1]) * 60 * 57.3))
print("U= %.2f V" % (np.sqrt(z[2, 2])))

save_ss("h2_obsv_noise_k", Wf)
save_ss("h2_obsv_noise_k_r", Wfr)
save_ss(
    "h2_obsv_noise_g",
    (Aa[:4, :4], B2[:4, :], C2[:, :4], np.zeros((1, 1))),
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
