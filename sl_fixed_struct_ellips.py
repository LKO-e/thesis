import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from src import A, C2, B2, W_ol, H, J2, J1, Jr, Cm, Ce, L, R, S, Si
from src import M_f, D_trq, D_dθ, D_dgam
from src import penalty_fnc_ellips, save_ss, connect_series_siso, ss2tf
from src import dc_gain_ss, feedback_connect, calc_hinf

# Constraints on h inf norm and control effort
hinf_max = 1.2
u_max = 24
# State space
C2 = C2[:, :4]
B2 = B2[:4, :]

Md_max = M_f + 2 * np.sqrt(D_trq)
B1_ = np.array(
    [
        [2 * np.sqrt(D_dθ), 0, 0, 0],
        [0, 0, 0, 0],
        [0, Md_max / J1, 0, 0],
        [0, 0, 2 * np.sqrt(D_dgam) * Ce / L, 0],
    ]
)
B1_ = S[:4, :4] @ B1_
Dw_ = np.zeros((C2.shape[0], B1_.shape[1]))


def solve_task(task, to_show_plots):
    name = f"case_{task:d}"
    match task:
        case 0:
            print(f"{name:s}: Without correction")

            def get_ss_matrices(kdc):
                Ag = A
                Cy = C2
                By = B2
                Dw = Dw_
                B1 = B1_
                Ak = None
                Bk = None
                Ck = None
                Dk = np.array([[kdc]])
                return Ak, Bk, Ck, Dk, Ag, B1, By, Cy, Dw

            sp.optimize.bounds = [(0, 30000)]

        case 1:
            print(f"{name:s}: Tachogenerator feedback")

            def get_ss_matrices(kdc, ka):
                Ag = A
                Cy = np.array([[-1, 0, 0, 0], [0, 0, 1, 0]])
                Cy = Cy @ Si[:4, :4]
                By = B2
                B1 = B1_
                Dw = np.array([[0, 0, 0, 0], [0, 0, -2 * np.sqrt(D_dgam), 0]])
                Ak = None
                Bk = None
                Ck = None
                Dk = np.array([[kdc, ka]])
                return Ak, Bk, Ck, Dk, Ag, B1, By, Cy, Dw

            sp.optimize.bounds = [(0, 30000), (-100, 100)]
        case 2:
            print(f"{name:s}: Gear ratio")

            def get_ss_matrices(kdc, q):
                A = np.array(
                    [
                        [0, -1, 0, 0],
                        [0, 0, -H / J2, 0],
                        [0, H / (J1 + Jr * q**2), 0, q * Cm / (J1 + Jr * q**2)],
                        [0, 0, -q * Ce / L, -R / L],
                    ]
                )
                Ag = S[:4, :4] @ A @ Si[:4, :4]
                B1 = np.array(
                    [
                        [2 * np.sqrt(D_dθ), 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, Md_max / J1, 0, 0],
                        [0, 0, 2 * np.sqrt(D_dgam) * q * Ce / L, 0],
                    ]
                )
                B1 = S[:4, :4] @ B1
                Cy = C2
                By = B2
                Dw = Dw_
                Ak = None
                Bk = None
                Ck = None
                Dk = np.array([[kdc]])
                return Ak, Bk, Ck, Dk, Ag, B1, By, Cy, Dw

            sp.optimize.bounds = [(0, 30000), (0, 100)]
        case 3:
            print(f"{name:s}: Notch-filter")

            def get_ss_matrices(kdc, ksi1, ksi2):
                Ag = A
                B1 = B1_
                Cy = C2
                By = B2
                Dw = Dw_
                wn = 381
                Ak = np.array([[0, 1], [-(wn**2), -2 * ksi2 * wn]])
                Bk = np.array([[0, kdc]]).T
                Ck = np.array([[0, 2 * wn * (ksi1 - ksi2)]])
                Dk = np.array([[kdc]])
                return Ak, Bk, Ck, Dk, Ag, B1, By, Cy, Dw

            sp.optimize.bounds = [(0, 30000), (0, 1), (0, 1)]
        case 4:
            print(f"{name:s}: First-order all-pass filter")

            def get_ss_matrices(kdc, w):
                Ag = A
                B1 = B1_
                Cy = C2
                By = B2
                Dw = Dw_
                Ak = np.array([[-w]])
                Bk = np.array([[kdc * w]])
                Ck = np.array([[2]])
                Dk = np.array([[-kdc]])
                return Ak, Bk, Ck, Dk, Ag, B1, By, Cy, Dw

            sp.optimize.bounds = [(0, 30000), (0, 1000)]
        case 5:
            print(f"{name:s}: Second-order low-pass filter")

            def get_ss_matrices(kdc, ksi, w):
                Ag = A
                B1 = B1_
                Cy = C2
                By = B2
                Dw = Dw_
                Ak = np.array([[0, 1], [-(w**2), -2 * ksi * w]])
                Bk = np.array([[0, kdc]]).T
                Ck = np.array([[w**2, 0]])
                Dk = np.array([[0]])
                return Ak, Bk, Ck, Dk, Ag, B1, By, Cy, Dw

            sp.optimize.bounds = [(0, 30000), (0, 10), (0, 500)]
        case 7:
            print(f"{name:s}: Second-order high-pass filter")

            def get_ss_matrices(kdc, w1, w2, ksi1, ksi2):
                Ag = A
                B1 = B1_
                Cy = C2
                By = B2
                Dw = Dw_
                Ak = np.array([[0, 1], [-(w2**2), -2 * ksi2 * w2]])
                Bk = np.array([[0, kdc * w2**2 / w1**2]]).T
                Ck = np.array([[w1**2 - w2**2, 2 * (ksi1 * w1 - ksi2 * w2)]])
                Dk = np.array([[kdc * w2**2 / w1**2]])
                return Ak, Bk, Ck, Dk, Ag, B1, By, Cy, Dw

            sp.optimize.bounds = [
                (0, 30000),
                (0, 1000),
                (1000, 3000),
                (-10, 10),
                (0, 10),
            ]
        case _:
            print("Case does not exist")
            return
    print(sp.optimize.bounds)

    # Penalty function
    def f(x, get_ss_matrices):
        Ak, Bk, Ck, Dk, Ag, B1, By, Cy, Dw = get_ss_matrices(*x)
        return penalty_fnc_ellips(Ak, Bk, Ck, Dk, Ag, B1, By, Cy, Dw, hinf_max, u_max)

    result = sp.optimize.differential_evolution(
        f,
        sp.optimize.bounds,
        args=(get_ss_matrices,),
        seed=None,
        popsize=150,
        polish=False,
    )
    print(result.message)
    if result.success:
        print(result.x)
        print(result.fun)
        Ak, Bk, Ck, Dk, Ag, B1, By, Cy, Dw = get_ss_matrices(*result.x)
        _ = penalty_fnc_ellips(
            Ak, Bk, Ck, Dk, Ag, B1, By, Cy, Dw, hinf_max, u_max, to_print=True
        )
        save_ss(f"ellips_{name:s}_k", (Ak, Bk, Ck, Dk))
        save_ss(
            f"ellips_{name:s}_g",
            (Ag[:4, :4], By[:4, :], Cy[:, :4], np.zeros((Cy.shape[0], By.shape[1]))),
        )
        if to_show_plots:
            Wf = (Ak, Bk, Ck, Dk)
            w = np.logspace(1, 3.6, 500)
            if Ak is not None:
                print("K = \n", ss2tf(Wf)[0])
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
            W_ol1 = connect_series_siso(
                Wf,
                (
                    Ag[:4, :4],
                    By[:4, :],
                    Cy[:, :4],
                    np.zeros((Cy.shape[0], By.shape[1])),
                ),
            )
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


if __name__ == "__main__":
    message = input("Enter which cases to solve: \n")
    tasks_to_solve = []
    for part in message.split(","):
        if "-" in part:
            start, end = [int(i) for i in part.split("-")]
            tasks_to_solve.extend(range(start, end + 1))
        elif part != "":
            tasks_to_solve.append(int(part))
    to_show_plots = len(tasks_to_solve) == 1
    for task in tasks_to_solve:
        print("".center(79, "#"))
        if task == 6:
            print(
                "case_6: The case concerns digital control. It's done in a separate file."
            )
            continue
        solve_task(task, to_show_plots)
