import sys
from pathlib import Path
import copy
import json
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from src import connect_series_siso, S, Si, J1, tf_to_num_den, ss2tf, feedback_connect
import plot_style

plot_style.set_plot_style()


def bode_fixed_struct_h2(ss_dict):
    # All h2-controllers' Bode plot
    fig, ax = plt.subplots(2, 2, sharex="col", figsize=(7.0, 4.5), frameon=False)
    fig.canvas.manager.set_window_title("H2 fixed-struct controllers")
    w0 = np.logspace(1, 3, 500)
    w1 = np.logspace(1, 3.6, 500)
    # Tachogenerator
    A, B, C, D = copy.deepcopy(list(ss_dict["h2_case_1_g"].values()))
    Ak, Bk, Ck, Dk = copy.deepcopy(list(ss_dict["h2_case_1_k"].values()))
    A, B, _, _ = feedback_connect((A, B, C[1:2, :], D[1:2, :]), sign=-Dk[0, 1])
    C[0, :] = C[0, :] * -1
    As, Bs, Cs, Ds = connect_series_siso(
        (A, B, C[0:1, :], D[0:1, :]), (Ak, Bk, Ck, Dk[:, 0:1])
    )
    _, mag, phase = sp.signal.bode((As, Bs, Cs, Ds), w0)
    ax[0, 0].semilogx(w0, mag, ls="-", c="k", label=r"$k_\mathrm{\dot{\alpha}}$")
    ax[1, 0].semilogx(w0, phase, ls="-", c="k", label=r"$k_\mathrm{\dot{\alpha}}$")
    # Gear ratio
    A, B, C, D = copy.deepcopy(list(ss_dict["h2_case_2_g"].values()))
    Ak, Bk, Ck, Dk = copy.deepcopy(list(ss_dict["h2_case_2_k"].values()))
    C[0, :] = C[0, :] * -1
    As, Bs, Cs, Ds = connect_series_siso((A, B, C, D), (Ak, Bk, Ck, Dk))
    _, mag, phase = sp.signal.bode((As, Bs, Cs, Ds), w0)
    ax[0, 0].semilogx(w0, mag, ls="--", c="k", label=r"$q$")
    ax[1, 0].semilogx(w0, phase, ls="--", c="k", label=r"$q$")
    # Notch filter
    A, B, C, D = copy.deepcopy(list(ss_dict["h2_case_3_g"].values()))
    Ak, Bk, Ck, Dk = copy.deepcopy(list(ss_dict["h2_case_3_k"].values()))
    C[0, :] = C[0, :] * -1
    As, Bs, Cs, Ds = connect_series_siso((A, B, C, D), (Ak, Bk, Ck, Dk))
    _, mag, phase = sp.signal.bode((As, Bs, Cs, Ds), w0)
    ax[0, 0].semilogx(w0, mag, ls="-.", c="k", label=r"РЖ")
    ax[1, 0].semilogx(w0, phase, ls="-.", c="k", label=r"РЖ")
    # All-pass filter
    A, B, C, D = copy.deepcopy(list(ss_dict["h2_case_4_g"].values()))
    Ak, Bk, Ck, Dk = copy.deepcopy(list(ss_dict["h2_case_4_k"].values()))
    C[0, :] = C[0, :] * -1
    As, Bs, Cs, Ds = connect_series_siso((A, B, C, D), (Ak, Bk, Ck, Dk))
    _, mag, phase = sp.signal.bode((As, Bs, Cs, Ds), w1)
    ax[0, 1].semilogx(w1, mag, ls="-", c="k", label=r"ФФ")
    ax[1, 1].semilogx(w1, phase, ls="-", c="k", label=r"ФФ")
    # Low-pass filter
    A, B, C, D = copy.deepcopy(list(ss_dict["h2_case_5_g"].values()))
    Ak, Bk, Ck, Dk = copy.deepcopy(list(ss_dict["h2_case_5_k"].values()))
    C[0, :] = C[0, :] * -1
    As, Bs, Cs, Ds = connect_series_siso((A, B, C, D), (Ak, Bk, Ck, Dk))
    _, mag, phase = sp.signal.bode((As, Bs, Cs, Ds), w1)
    ax[0, 1].semilogx(w1, mag, ls="--", c="k", label=r"ФНЧ")
    ax[1, 1].semilogx(w1, phase, ls="--", c="k", label=r"ФНЧ")
    # High-pass filter
    A, B, C, D = copy.deepcopy(list(ss_dict["h2_case_7_g"].values()))
    Ak, Bk, Ck, Dk = copy.deepcopy(list(ss_dict["h2_case_7_k"].values()))
    C[0, :] = C[0, :] * -1
    As, Bs, Cs, Ds = connect_series_siso((A, B, C, D), (Ak, Bk, Ck, Dk))
    _, mag, phase = sp.signal.bode((As, Bs, Cs, Ds), w1)
    ax[0, 1].semilogx(w1, mag, ls="-.", c="k", label=r"ФВЧ")
    ax[1, 1].semilogx(w1, phase, ls="-.", c="k", label=r"ФВЧ")
    # Margins
    ax[0, 0].axhline(0, c="k", ls=":", label="")
    ax[1, 0].axhline(-180, c="k", ls=":")
    ax[0, 1].axhline(0, c="k", ls=":", label="")
    ax[1, 1].axhline(-180, c="k", ls=":")
    # Legends
    ax[0, 0].legend()
    ax[0, 1].legend()
    ax[1, 0].legend()
    ax[1, 1].legend()
    ax[0, 0].set_ylabel(
        r"$\vert W_\mathrm{р}(\mathrm{\vartheta}, \mathrm{\vartheta_1})\vert \mathrm{, \ дБ}$"
    )
    ax[1, 0].set_ylabel(
        r"$\angle W_\mathrm{р}(\mathrm{\vartheta}, \mathrm{\vartheta_1}) \mathrm{, \ град.}$"
    )
    ax[0, 0].grid(True, which="both")
    ax[0, 0].set_title("a)")
    ax[0, 1].set_title("б)")
    ax[1, 0].set_xlabel(r"$\mathrm{\omega, \  с^{-1}}$")
    ax[1, 1].set_xlabel(r"$\mathrm{\omega, \  с^{-1}}$")
    ax[1, 0].grid(True, which="both")
    return fig


def bode_fixed_struct_ellips(ss_dict):
    fig, ax = plt.subplots(2, 2, sharex="col", figsize=(7.0, 4.5), frameon=False)
    fig.canvas.manager.set_window_title("Ellipsoid fixed-struct controllers")
    w0 = np.logspace(1, 3, 1000)
    w1 = np.logspace(1, 3.5, 1000)
    # Tachogenerator
    A, B, C, D = copy.deepcopy(list(ss_dict["h2_case_1_g"].values()))
    Ak, Bk, Ck, Dk = copy.deepcopy(list(ss_dict["h2_case_1_k"].values()))
    A, B, _, _ = feedback_connect((A, B, C[1:2, :], D[1:2, :]), sign=-Dk[0, 1])
    C[0, :] = C[0, :] * -1
    As, Bs, Cs, Ds = connect_series_siso(
        (A, B, C[0:1, :], D[0:1, :]), (Ak, Bk, Ck, Dk[:, 0:1])
    )
    _, mag, phase = sp.signal.bode((As, Bs, Cs, Ds), w0)
    ax[0, 0].semilogx(w0, mag, ls="-", c="k", label=r"$k_\mathrm{\dot{\alpha}}$")
    ax[1, 0].semilogx(w0, phase, ls="-", c="k", label=r"$k_\mathrm{\dot{\alpha}}$")
    # Gear ratio
    A, B, C, D = copy.deepcopy(list(ss_dict["ellips_case_2_g"].values()))
    Ak, Bk, Ck, Dk = copy.deepcopy(list(ss_dict["ellips_case_2_k"].values()))
    C[0, :] = C[0, :] * -1
    As, Bs, Cs, Ds = connect_series_siso((A, B, C, D), (Ak, Bk, Ck, Dk))
    _, mag, phase = sp.signal.bode((As, Bs, Cs, Ds), w0)
    ax[0, 0].semilogx(w0, mag, ls="--", c="k", label=r"$q$")
    ax[1, 0].semilogx(w0, phase, ls="--", c="k", label=r"$q$")
    # Notch filter
    A, B, C, D = copy.deepcopy(list(ss_dict["ellips_case_3_g"].values()))
    Ak, Bk, Ck, Dk = copy.deepcopy(list(ss_dict["ellips_case_3_k"].values()))
    C[0, :] = C[0, :] * -1
    As, Bs, Cs, Ds = connect_series_siso((A, B, C, D), (Ak, Bk, Ck, Dk))
    _, mag, phase = sp.signal.bode((As, Bs, Cs, Ds), w0)
    ax[0, 0].semilogx(w0, mag, ls="-.", c="k", label=r"РЖ")
    ax[1, 0].semilogx(w0, phase, ls="-.", c="k", label=r"РЖ")
    # All-pass filter
    A, B, C, D = copy.deepcopy(list(ss_dict["ellips_case_4_g"].values()))
    Ak, Bk, Ck, Dk = copy.deepcopy(list(ss_dict["ellips_case_4_k"].values()))
    C[0, :] = C[0, :] * -1
    As, Bs, Cs, Ds = connect_series_siso((A, B, C, D), (Ak, Bk, Ck, Dk))
    _, mag, phase = sp.signal.bode((As, Bs, Cs, Ds), w1)
    ax[0, 1].semilogx(w1, mag, ls="-", c="k", label=r"ФФ")
    ax[1, 1].semilogx(w1, phase, ls="-", c="k", label=r"ФФ")
    # Low-pass filter
    A, B, C, D = copy.deepcopy(list(ss_dict["ellips_case_5_g"].values()))
    Ak, Bk, Ck, Dk = copy.deepcopy(list(ss_dict["ellips_case_5_k"].values()))
    C[0, :] = C[0, :] * -1
    As, Bs, Cs, Ds = connect_series_siso((A, B, C, D), (Ak, Bk, Ck, Dk))
    _, mag, phase = sp.signal.bode((As, Bs, Cs, Ds), w1)
    ax[0, 1].semilogx(w1, mag, ls="--", c="k", label=r"ФНЧ")
    ax[1, 1].semilogx(w1, phase, ls="--", c="k", label=r"ФНЧ")
    # High-pass filter
    A, B, C, D = copy.deepcopy(list(ss_dict["ellips_case_7_g"].values()))
    Ak, Bk, Ck, Dk = copy.deepcopy(list(ss_dict["ellips_case_7_k"].values()))
    C[0, :] = C[0, :] * -1
    As, Bs, Cs, Ds = connect_series_siso((A, B, C, D), (Ak, Bk, Ck, Dk))
    _, mag, phase = sp.signal.bode((As, Bs, Cs, Ds), w1)
    ax[0, 1].semilogx(w1, mag, ls="-.", c="k", label=r"ФВЧ")
    ax[1, 1].semilogx(w1, phase, ls="-.", c="k", label=r"ФВЧ")
    # Margins
    ax[0, 0].axhline(0, c="k", ls=":", label="")
    ax[1, 0].axhline(-180, c="k", ls=":")
    ax[0, 1].axhline(0, c="k", ls=":", label="")
    ax[1, 1].axhline(-180, c="k", ls=":")
    ax[1, 1].axhline(-180 - 360, c="k", ls=":")
    # Legends
    ax[0, 0].legend()
    ax[0, 1].legend()
    ax[1, 0].legend()
    ax[1, 1].legend()
    ax[0, 0].set_ylabel(
        r"$\vert W_\mathrm{р}(\mathrm{\vartheta}, \mathrm{\vartheta_1})\vert \mathrm{, \ дБ}$"
    )
    ax[1, 0].set_ylabel(
        r"$\angle W_\mathrm{р}(\mathrm{\vartheta}, \mathrm{\vartheta_1}) \mathrm{, \ град.}$"
    )
    ax[0, 0].grid(True, which="both")
    ax[0, 0].set_title("a)")
    ax[0, 1].set_title("б)")
    ax[1, 0].set_xlabel(r"$\mathrm{\omega, \  с^{-1}}$")
    ax[1, 1].set_xlabel(r"$\mathrm{\omega, \  с^{-1}}$")
    ax[1, 0].grid(True, which="both")
    return fig


def bode_h2_obsv_noise(ss_dict):
    fig, ax = plt.subplots(2, 2, sharex="col", figsize=(7.0, 4.5), frameon=False)
    fig.canvas.manager.set_window_title("H2 controller with noise")
    w0 = np.logspace(-1, 5, 500)
    w1 = np.logspace(1, 3.5, 500)
    # Mininal noise controller
    A, B, C, D = copy.deepcopy(list(ss_dict["h2_obsv_g"].values()))
    Ak, Bk, Ck, Dk = copy.deepcopy(list(ss_dict["h2_obsv_k"].values()))
    C[0, :] = C[0, :] * -1
    As, Bs, Cs, Ds = connect_series_siso((A, B, C, D), (Ak, Bk, Ck, Dk))
    _, mag, phase = sp.signal.bode((Ak, Bk, Ck, Dk), w0)
    ax[0, 0].semilogx(w0, mag, ls="-", c="k", label=r"$R_\mathrm{н1}$")
    ax[1, 0].semilogx(w0, phase, ls="-", c="k", label=r"$R_\mathrm{н1}$")
    _, mag, phase = sp.signal.bode((As, Bs, Cs, Ds), w1)
    ax[0, 1].semilogx(w1, mag, ls="-", c="k", label=r"$R_\mathrm{н1}$")
    ax[1, 1].semilogx(w1, phase, ls="-", c="k", label=r"$R_\mathrm{н1}$")
    # Increased noise
    A, B, C, D = copy.deepcopy(list(ss_dict["h2_obsv_noise_g"].values()))
    Ak, Bk, Ck, Dk = copy.deepcopy(list(ss_dict["h2_obsv_noise_k"].values()))
    C[0, :] = C[0, :] * -1
    As, Bs, Cs, Ds = connect_series_siso((A, B, C, D), (Ak, Bk, Ck, Dk))
    _, mag, phase = sp.signal.bode((Ak, Bk, Ck, Dk), w0)
    ax[0, 0].semilogx(w0, mag, ls="--", c="k", label=r"$R_\mathrm{н2}$")
    ax[1, 0].semilogx(w0, phase, ls="--", c="k", label=r"$R_\mathrm{н2}$")
    _, mag, phase = sp.signal.bode((As, Bs, Cs, Ds), w1)
    ax[0, 1].semilogx(w1, mag, ls="--", c="k", label=r"$R_\mathrm{н2}$")
    ax[1, 1].semilogx(w1, phase, ls="--", c="k", label=r"$R_\mathrm{н2}$")
    # Margins
    ax[1, 0].axhline(-180, c="k", ls=":")
    ax[0, 1].axhline(0, c="k", ls=":", label="")
    ax[1, 1].axhline(-180, c="k", ls=":")
    # Legends
    ax[0, 0].legend()
    ax[0, 1].legend(loc=3)
    ax[1, 0].legend()
    ax[1, 1].legend()
    ax[0, 1].set_ylabel(
        r"$\vert W_\mathrm{р}(\mathrm{\vartheta}, \mathrm{\vartheta_1})\vert \mathrm{, \ дБ}$"
    )
    ax[0, 0].set_ylabel(r"$\vert k_\mathrm{с} W_\mathrm{ф}\vert \mathrm{, \ дБ}$")
    ax[1, 1].set_ylabel(
        r"$\angle W_\mathrm{р}(\mathrm{\vartheta}, \mathrm{\vartheta_1}) \mathrm{, \ град.}$"
    )
    ax[1, 0].set_ylabel(r"$\angle k_\mathrm{с} W_\mathrm{ф} \mathrm{, \ град.}$")
    ax[0, 0].grid(True, which="both")
    ax[0, 0].set_title("a)")
    ax[0, 1].set_title("б)")
    ax[1, 0].set_xlabel(r"$\mathrm{\omega, \  с^{-1}}$")
    ax[1, 1].set_xlabel(r"$\mathrm{\omega, \  с^{-1}}$")
    ax[1, 0].grid(True, which="both")
    return fig

def bode_h2_test_with_corr_loop(ss_dict):
    fig, ax = plt.subplots(2, 2, sharex="col", figsize=(7.0, 4.5), frameon=False)
    fig.canvas.manager.set_window_title("H2 controller with corr loop")
    w0 = np.logspace(-2, 5, 500)
    w1 = np.logspace(1, 4, 500)
    # Mininal noise controller
    A, B, C, D = copy.deepcopy(list(ss_dict["h2_obsv_g"].values()))
    Ak, Bk, Ck, Dk = copy.deepcopy(list(ss_dict["h2_corr_obsv_k"].values()))
    C[0, :] = C[0, :] * -1
    As, Bs, Cs, Ds = connect_series_siso((A, B, C, D), (Ak, Bk, Ck, Dk))
    _, mag, phase = sp.signal.bode((Ak, Bk, Ck, Dk), w0)
    ax[0, 0].semilogx(w0, mag, ls="-", c="k")
    ax[1, 0].semilogx(w0, phase, ls="-", c="k")
    _, mag, phase = sp.signal.bode((As, Bs, Cs, Ds), w1)
    ax[0, 1].semilogx(w1, mag, ls="-", c="k")
    ax[1, 1].semilogx(w1, phase, ls="-", c="k")
    # Margins
    ax[1, 0].axhline(-180, c="k", ls=":")
    ax[0, 1].axhline(0, c="k", ls=":", label="")
    ax[1, 1].axhline(-180, c="k", ls=":")
    # Legends
    ax[0, 1].set_ylabel(
        r"$\vert W_\mathrm{р}(\mathrm{\vartheta}, \mathrm{\vartheta_1})\vert \mathrm{, \ дБ}$"
    )
    ax[0, 0].set_ylabel(r"$\vert k_\mathrm{с} W_\mathrm{ф}\vert \mathrm{, \ дБ}$")
    ax[1, 1].set_ylabel(
        r"$\angle W_\mathrm{р}(\mathrm{\vartheta}, \mathrm{\vartheta_1}) \mathrm{, \ град.}$"
    )
    ax[1, 0].set_ylabel(r"$\angle k_\mathrm{с} W_\mathrm{ф} \mathrm{, \ град.}$")
    ax[0, 0].grid(True, which="both")
    ax[0, 0].set_title("a)")
    ax[0, 1].set_title("б)")
    ax[1, 0].set_xlabel(r"$\mathrm{\omega, \  с^{-1}}$")
    ax[1, 1].set_xlabel(r"$\mathrm{\omega, \  с^{-1}}$")
    ax[1, 0].grid(True, which="both")
    return fig

def bode_h2_ellips(ss_dict):
    fig, ax = plt.subplots(2, 2, sharex="col", figsize=(7.0, 4.5), frameon=False)
    fig.canvas.manager.set_window_title("Bode H2 and ellipsoid")
    w0 = np.logspace(-1, 5, 500)
    w1 = np.logspace(1, 3.5, 500)
    # H2 controller
    A, B, C, D = copy.deepcopy(list(ss_dict["h2_obsv_g"].values()))
    Ak, Bk, Ck, Dk = copy.deepcopy(list(ss_dict["h2_obsv_k"].values()))
    C[0, :] = C[0, :] * -1
    As, Bs, Cs, Ds = connect_series_siso((A, B, C, D), (Ak, Bk, Ck, Dk))
    _, mag, phase = sp.signal.bode((Ak, Bk, Ck, Dk), w0)
    ax[0, 0].semilogx(w0, mag, ls="-", c="k", label=r"$H_2$")
    ax[1, 0].semilogx(w0, phase, ls="-", c="k", label=r"$H_2$")
    _, mag, phase = sp.signal.bode((As, Bs, Cs, Ds), w1)
    ax[0, 1].semilogx(w1, mag, ls="-", c="k", label=r"$H_2$")
    ax[1, 1].semilogx(w1, phase, ls="-", c="k", label=r"$H_2$")
    # Ellipsoid controller
    A, B, C, D = copy.deepcopy(list(ss_dict["ellips_obsv_g"].values()))
    Ak, Bk, Ck, Dk = copy.deepcopy(list(ss_dict["ellips_obsv_k"].values()))
    C[0, :] = C[0, :] * -1
    As, Bs, Cs, Ds = connect_series_siso((A, B, C, D), (Ak, Bk, Ck, Dk))
    _, mag, phase = sp.signal.bode((Ak, Bk, Ck, Dk), w0)
    ax[0, 0].semilogx(w0, mag, ls="--", c="k", label=r"$\mathrm{Эл.}$")
    ax[1, 0].semilogx(w0, phase, ls="--", c="k", label=r"$\mathrm{Эл.}$")
    _, mag, phase = sp.signal.bode((As, Bs, Cs, Ds), w1)
    ax[0, 1].semilogx(w1, mag, ls="--", c="k", label=r"$\mathrm{Эл.}$")
    ax[1, 1].semilogx(w1, phase, ls="--", c="k", label=r"$\mathrm{Эл.}$")
    # Margins
    ax[1, 0].axhline(-180, c="k", ls=":")
    ax[0, 1].axhline(0, c="k", ls=":", label="")
    ax[1, 1].axhline(-180, c="k", ls=":")
    # Legends
    ax[0, 0].legend()
    ax[0, 1].legend(loc=3)
    ax[1, 0].legend()
    ax[1, 1].legend(loc=3)
    ax[0, 1].set_ylabel(
        r"$\vert W_\mathrm{р}(\mathrm{\vartheta}, \mathrm{\vartheta_1})\vert \mathrm{, \ дБ}$"
    )
    ax[0, 0].set_ylabel(r"$\vert k_\mathrm{с} W_\mathrm{ф}\vert \mathrm{, \ дБ}$")
    ax[1, 1].set_ylabel(
        r"$\angle W_\mathrm{р}(\mathrm{\vartheta}, \mathrm{\vartheta_1}) \mathrm{, \ град.}$"
    )
    ax[1, 0].set_ylabel(r"$\angle k_\mathrm{с} W_\mathrm{ф} \mathrm{, \ град.}$")
    ax[0, 0].grid(True, which="both")
    ax[0, 0].set_title("a)")
    ax[0, 1].set_title("б)")
    ax[1, 0].set_xlabel(r"$\mathrm{\omega, \  с^{-1}}$")
    ax[1, 1].set_xlabel(r"$\mathrm{\omega, \  с^{-1}}$")
    ax[1, 0].grid(True, which="both")
    return fig


def bode_h2_ellips_reduced(ss_dict):
    fig, ax = plt.subplots(2, 2, sharex="col", figsize=(7.0, 4.5), frameon=False)
    fig.canvas.manager.set_window_title("Bode H2 and ellipsoid reduced")
    w0 = np.logspace(-1, 5, 500)
    w1 = np.logspace(1, 3.5, 500)
    # H2 controller
    A, B, C, D = copy.deepcopy(list(ss_dict["h2_obsv_g"].values()))
    Ak, Bk, Ck, Dk = copy.deepcopy(list(ss_dict["h2_obsv_k_r"].values()))
    C[0, :] = C[0, :] * -1
    As, Bs, Cs, Ds = connect_series_siso((A, B, C, D), (Ak, Bk, Ck, Dk))
    _, mag, phase = sp.signal.bode((Ak, Bk, Ck, Dk), w0)
    ax[0, 0].semilogx(w0, mag, ls="-", c="k", label=r"$H_2$")
    ax[1, 0].semilogx(w0, phase, ls="-", c="k", label=r"$H_2$")
    _, mag, phase = sp.signal.bode((As, Bs, Cs, Ds), w1)
    ax[0, 1].semilogx(w1, mag, ls="-", c="k", label=r"$H_2$")
    ax[1, 1].semilogx(w1, phase, ls="-", c="k", label=r"$H_2$")
    # Ellipsoid controller
    A, B, C, D = copy.deepcopy(list(ss_dict["ellips_obsv_g"].values()))
    Ak, Bk, Ck, Dk = copy.deepcopy(list(ss_dict["ellips_obsv_k_r"].values()))
    C[0, :] = C[0, :] * -1
    As, Bs, Cs, Ds = connect_series_siso((A, B, C, D), (Ak, Bk, Ck, Dk))
    _, mag, phase = sp.signal.bode((Ak, Bk, Ck, Dk), w0)
    ax[0, 0].semilogx(w0, mag, ls="--", c="k", label=r"$\mathrm{Эл.}$")
    ax[1, 0].semilogx(w0, phase, ls="--", c="k", label=r"$\mathrm{Эл.}$")
    _, mag, phase = sp.signal.bode((As, Bs, Cs, Ds), w1)
    ax[0, 1].semilogx(w1, mag, ls="--", c="k", label=r"$\mathrm{Эл.}$")
    ax[1, 1].semilogx(w1, phase, ls="--", c="k", label=r"$\mathrm{Эл.}$")
    # Decorations
    ax[1, 0].axhline(-180, c="k", ls=":")
    ax[0, 1].axhline(0, c="k", ls=":", label="")
    ax[1, 1].axhline(-180, c="k", ls=":")
    ax[0, 0].legend()
    ax[0, 1].legend(loc=3)
    ax[1, 0].legend()
    ax[1, 1].legend()
    ax[0, 1].set_ylabel(
        r"$\vert W_\mathrm{р}(\mathrm{\vartheta}, \mathrm{\vartheta_1})\vert \mathrm{, \ дБ}$"
    )
    ax[0, 0].set_ylabel(r"$\vert k_\mathrm{с} \bar{W}_\mathrm{ф}\vert \mathrm{, \ дБ}$")
    ax[1, 1].set_ylabel(
        r"$\angle W_\mathrm{р}(\mathrm{\vartheta}, \mathrm{\vartheta_1}) \mathrm{, \ град.}$"
    )
    ax[1, 0].set_ylabel(r"$\angle k_\mathrm{с} \bar{W}_\mathrm{ф} \mathrm{, \ град.}$")
    ax[0, 0].grid(True, which="both")
    ax[0, 0].set_title("a)")
    ax[0, 1].set_title("б)")
    ax[1, 0].set_xlabel(r"$\mathrm{\omega, \  с^{-1}}$")
    ax[1, 1].set_xlabel(r"$\mathrm{\omega, \  с^{-1}}$")
    ax[1, 0].grid(True, which="both")
    return fig


def bode_mixed_ellips_hinf(ss_dict):
    fig, ax = plt.subplots(2, 2, sharex="col", figsize=(7.0, 4.5), frameon=False)
    fig.canvas.manager.set_window_title("Mixed ellipsoid & H inf synthesis")
    w0 = np.logspace(-1, 5, 500)
    w1 = np.logspace(1, 3.5, 500)
    # with the H inf constr
    A, B, C, D = copy.deepcopy(list(ss_dict["mixed_syn_w_hinf_g"].values()))
    Ak, Bk, Ck, Dk = copy.deepcopy(list(ss_dict["mixed_syn_w_hinf_k"].values()))
    C[0, :] = C[0, :] * -1
    As, Bs, Cs, Ds = connect_series_siso((A, B, C, D), (Ak, Bk, Ck, Dk))
    _, mag, phase = sp.signal.bode((Ak, Bk, Ck, Dk), w0)
    ax[0, 0].semilogx(w0, mag, ls="-", c="k", label=r"$\mathrm{c} \ H_\infty$")
    ax[1, 0].semilogx(w0, phase, ls="-", c="k", label=r"$\mathrm{c} \ H_\infty$")
    _, mag, phase = sp.signal.bode((As, Bs, Cs, Ds), w1)
    ax[0, 1].semilogx(w1, mag, ls="-", c="k", label=r"$\mathrm{c} \ H_\infty$")
    ax[1, 1].semilogx(w1, phase, ls="-", c="k", label=r"$\mathrm{c} \ H_\infty$")
    # without the H inf constraint
    A, B, C, D = copy.deepcopy(list(ss_dict["mixed_syn_w_o_hinf_g"].values()))
    Ak, Bk, Ck, Dk = copy.deepcopy(list(ss_dict["mixed_syn_w_o_hinf_k"].values()))
    C[0, :] = C[0, :] * -1
    As, Bs, Cs, Ds = connect_series_siso((A, B, C, D), (Ak, Bk, Ck, Dk))
    _, mag, phase = sp.signal.bode((Ak, Bk, Ck, Dk), w0)
    ax[0, 0].semilogx(w0, mag, ls="--", c="k", label=r"$\mathrm{без} \ H_\infty$")
    ax[1, 0].semilogx(w0, phase, ls="--", c="k", label=r"$\mathrm{без} \ H_\infty$")
    _, mag, phase = sp.signal.bode((As, Bs, Cs, Ds), w1)
    ax[0, 1].semilogx(w1, mag, ls="--", c="k", label=r"$\mathrm{без} \ H_\infty$")
    ax[1, 1].semilogx(w1, phase, ls="--", c="k", label=r"$\mathrm{без} \ H_\infty$")
    # Decoration
    ax[1, 0].axhline(-180, c="k", ls=":")
    ax[0, 1].axhline(0, c="k", ls=":", label="")
    ax[1, 1].axhline(-180, c="k", ls=":")
    ax[0, 0].legend()
    ax[0, 1].legend()
    ax[1, 0].legend(loc="lower left")
    ax[1, 1].legend()
    ax[0, 1].set_ylabel(
        r"$\vert W_\mathrm{р}(\mathrm{\vartheta}, \mathrm{\vartheta_1})\vert \mathrm{, \ дБ}$"
    )
    ax[0, 0].set_ylabel(r"$\vert k_\mathrm{с} W_\mathrm{ф}\vert \mathrm{, \ дБ}$")
    ax[1, 1].set_ylabel(
        r"$\angle W_\mathrm{р}(\mathrm{\vartheta}, \mathrm{\vartheta_1}) \mathrm{, \ град.}$"
    )
    ax[1, 0].set_ylabel(r"$\angle k_\mathrm{с} W_\mathrm{ф} \mathrm{, \ град.}$")
    ax[0, 0].grid(True, which="both")
    ax[0, 0].set_title("a)")
    ax[0, 1].set_title("б)")
    ax[1, 0].set_xlabel(r"$\mathrm{\omega, \  с^{-1}}$")
    ax[1, 1].set_xlabel(r"$\mathrm{\omega, \  с^{-1}}$")
    ax[1, 0].grid(True, which="both")
    return fig


def mag_dist_to_theta(ss_dict):
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.9))
    fig.canvas.manager.set_window_title("Closed-loop mag plot for T dist to theta")
    w0 = np.logspace(0.5, 3, 1000)
    # Ellipsiod
    # Notch-filter
    A, B, C, D = copy.deepcopy(list(ss_dict["ellips_case_3_g"].values()))
    Ak, Bk, Ck, Dk = copy.deepcopy(list(ss_dict["ellips_case_3_k"].values()))
    Acl = np.block([[A + B @ Dk @ C, B @ Ck], [Bk @ C, Ak]])
    Bcl = S[:4, :4] @ np.array([[0, 0, 1 / J1, 0]]).T
    Bcl = np.block([[Bcl], [np.zeros((Ak.shape[0], 1))]])
    Ccl = np.array([[1, 0, 0, 0]]) @ Si[:4, :4]
    Ccl = np.block([[Ccl, np.zeros((1, Ak.shape[1]))]])
    Dcl = D @ Dk
    _, mag, _ = sp.signal.bode((Acl, Bcl, Ccl, Dcl), w0)
    ax[0].semilogx(w0, mag, ls="--", c="k", label=r"РЖ")
    # High-pass filter
    A, B, C, D = copy.deepcopy(list(ss_dict["ellips_case_7_g"].values()))
    Ak, Bk, Ck, Dk = copy.deepcopy(list(ss_dict["ellips_case_7_k"].values()))
    Acl = np.block([[A + B @ Dk @ C, B @ Ck], [Bk @ C, Ak]])
    Bcl = S[:4, :4] @ np.array([[0, 0, 1 / J1, 0]]).T
    Bcl = np.block([[Bcl], [np.zeros((Ak.shape[0], 1))]])
    Ccl = np.array([[1, 0, 0, 0]]) @ Si[:4, :4]
    Ccl = np.block([[Ccl, np.zeros((1, Ak.shape[1]))]])
    Dcl = D @ Dk
    _, mag, _ = sp.signal.bode((Acl, Bcl, Ccl, Dcl), w0)
    ax[0].semilogx(w0, mag, ls="-.", c="k", label="ФВЧ")
    # Low-pass filter
    A, B, C, D = copy.deepcopy(list(ss_dict["ellips_case_5_g"].values()))
    Ak, Bk, Ck, Dk = copy.deepcopy(list(ss_dict["ellips_case_5_k"].values()))
    Acl = np.block([[A + B @ Dk @ C, B @ Ck], [Bk @ C, Ak]])
    Bcl = S[:4, :4] @ np.array([[0, 0, 1 / J1, 0]]).T
    Bcl = np.block([[Bcl], [np.zeros((Ak.shape[0], 1))]])
    Ccl = np.array([[1, 0, 0, 0]]) @ Si[:4, :4]
    Ccl = np.block([[Ccl, np.zeros((1, Ak.shape[1]))]])
    Dcl = D @ Dk
    _, mag, _ = sp.signal.bode((Acl, Bcl, Ccl, Dcl), w0)
    ax[0].semilogx(w0, mag, ls="-", c="k", label=r"ФНЧ")
    # H2
    # Notch-filter
    A, B, C, D = copy.deepcopy(list(ss_dict["h2_case_3_g"].values()))
    Ak, Bk, Ck, Dk = copy.deepcopy(list(ss_dict["h2_case_3_k"].values()))
    Acl = np.block([[A + B @ Dk @ C, B @ Ck], [Bk @ C, Ak]])
    Bcl = S[:4, :4] @ np.array([[0, 0, 1 / J1, 0]]).T
    Bcl = np.block([[Bcl], [np.zeros((Ak.shape[0], 1))]])
    Ccl = np.array([[1, 0, 0, 0]]) @ Si[:4, :4]
    Ccl = np.block([[Ccl, np.zeros((1, Ak.shape[1]))]])
    Dcl = D @ Dk
    _, mag, _ = sp.signal.bode((Acl, Bcl, Ccl, Dcl), w0)
    ax[1].semilogx(w0, mag, ls="--", c="k", label=r"РЖ")
    # High-pass filter
    A, B, C, D = copy.deepcopy(list(ss_dict["h2_case_7_g"].values()))
    Ak, Bk, Ck, Dk = copy.deepcopy(list(ss_dict["h2_case_7_k"].values()))
    Acl = np.block([[A + B @ Dk @ C, B @ Ck], [Bk @ C, Ak]])
    Bcl = S[:4, :4] @ np.array([[0, 0, 1 / J1, 0]]).T
    Bcl = np.block([[Bcl], [np.zeros((Ak.shape[0], 1))]])
    Ccl = np.array([[1, 0, 0, 0]]) @ Si[:4, :4]
    Ccl = np.block([[Ccl, np.zeros((1, Ak.shape[1]))]])
    Dcl = D @ Dk
    _, mag, _ = sp.signal.bode((Acl, Bcl, Ccl, Dcl), w0)
    ax[1].semilogx(w0, mag, ls="-.", c="k", label="ФВЧ")
    # Low-pass filter
    A, B, C, D = copy.deepcopy(list(ss_dict["h2_case_5_g"].values()))
    Ak, Bk, Ck, Dk = copy.deepcopy(list(ss_dict["h2_case_5_k"].values()))
    Acl = np.block([[A + B @ Dk @ C, B @ Ck], [Bk @ C, Ak]])
    Bcl = S[:4, :4] @ np.array([[0, 0, 1 / J1, 0]]).T
    Bcl = np.block([[Bcl], [np.zeros((Ak.shape[0], 1))]])
    Ccl = np.array([[1, 0, 0, 0]]) @ Si[:4, :4]
    Ccl = np.block([[Ccl, np.zeros((1, Ak.shape[1]))]])
    Dcl = D @ Dk
    _, mag, _ = sp.signal.bode((Acl, Bcl, Ccl, Dcl), w0)
    ax[1].semilogx(w0, mag, ls="-", c="k", label=r"ФНЧ")
    # Decoration
    ax[0].legend()
    ax[0].set_ylabel(
        r"$\vert W_\mathrm{з}(M_\mathrm{в}^x, \mathrm{\vartheta_1})\vert \mathrm{, \ дБ}$"
    )
    ax[0].grid(True, which="both")
    ax[0].set_title("a)")
    ax[0].set_xlabel(r"$\mathrm{\omega, \  с^{-1}}$")
    ax[1].legend()
    ax[1].grid(True, which="both")
    ax[1].set_title("б)")
    ax[1].set_xlabel(r"$\mathrm{\omega, \  с^{-1}}$")
    return fig


def mag_cmd_der_to_u(ss_dict):
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.9))
    fig.canvas.manager.set_window_title("Closed-loop mag plot for cmd der to u")
    w0 = np.logspace(-1, 5, 1000)
    # Ellipsiod
    # Notch-filter
    A, B, C, D = copy.deepcopy(list(ss_dict["ellips_case_3_g"].values()))
    Ak, Bk, Ck, Dk = copy.deepcopy(list(ss_dict["ellips_case_3_k"].values()))
    Acl = np.block([[A + B @ Dk @ C, B @ Ck], [Bk @ C, Ak]])
    Bcl = S[:4, :4] @ np.array([[1, 0, 0, 0]]).T
    Bcl = np.block([[Bcl], [np.zeros((Ak.shape[0], 1))]])
    Ccl = np.block([[Dk @ C, Ck]])
    Dcl = D @ Dk
    _, mag, _ = sp.signal.bode((Acl, Bcl, Ccl, Dcl), w0)
    ax[0].semilogx(w0, mag, ls="--", c="k", label=r"РЖ")
    # High-pass filter
    A, B, C, D = copy.deepcopy(list(ss_dict["ellips_case_7_g"].values()))
    Ak, Bk, Ck, Dk = copy.deepcopy(list(ss_dict["ellips_case_7_k"].values()))
    Acl = np.block([[A + B @ Dk @ C, B @ Ck], [Bk @ C, Ak]])
    Bcl = S[:4, :4] @ np.array([[1, 0, 0, 0]]).T
    Bcl = np.block([[Bcl], [np.zeros((Ak.shape[0], 1))]])
    Ccl = np.block([[Dk @ C, Ck]])
    Dcl = D @ Dk
    _, mag, _ = sp.signal.bode((Acl, Bcl, Ccl, Dcl), w0)
    ax[0].semilogx(w0, mag, ls="-.", c="k", label="ФВЧ")
    # Low-pass filter
    A, B, C, D = copy.deepcopy(list(ss_dict["ellips_case_5_g"].values()))
    Ak, Bk, Ck, Dk = copy.deepcopy(list(ss_dict["ellips_case_5_k"].values()))
    Acl = np.block([[A + B @ Dk @ C, B @ Ck], [Bk @ C, Ak]])
    Bcl = S[:4, :4] @ np.array([[1, 0, 0, 0]]).T
    Bcl = np.block([[Bcl], [np.zeros((Ak.shape[0], 1))]])
    Ccl = np.block([[Dk @ C, Ck]])
    Dcl = D @ Dk
    _, mag, _ = sp.signal.bode((Acl, Bcl, Ccl, Dcl), w0)
    ax[0].semilogx(w0, mag, ls="-", c="k", label=r"ФНЧ")
    # H2
    # Notch-filter
    A, B, C, D = copy.deepcopy(list(ss_dict["h2_case_3_g"].values()))
    Ak, Bk, Ck, Dk = copy.deepcopy(list(ss_dict["h2_case_3_k"].values()))
    Acl = np.block([[A + B @ Dk @ C, B @ Ck], [Bk @ C, Ak]])
    Bcl = S[:4, :4] @ np.array([[1, 0, 0, 0]]).T
    Bcl = np.block([[Bcl], [np.zeros((Ak.shape[0], 1))]])
    Ccl = np.block([[Dk @ C, Ck]])
    Dcl = D @ Dk
    _, mag, _ = sp.signal.bode((Acl, Bcl, Ccl, Dcl), w0)
    ax[1].semilogx(w0, mag, ls="--", c="k", label=r"РЖ")
    # High-pass filter
    A, B, C, D = copy.deepcopy(list(ss_dict["h2_case_7_g"].values()))
    Ak, Bk, Ck, Dk = copy.deepcopy(list(ss_dict["h2_case_7_k"].values()))
    Acl = np.block([[A + B @ Dk @ C, B @ Ck], [Bk @ C, Ak]])
    Bcl = S[:4, :4] @ np.array([[1, 0, 0, 0]]).T
    Bcl = np.block([[Bcl], [np.zeros((Ak.shape[0], 1))]])
    Ccl = np.block([[Dk @ C, Ck]])
    Dcl = D @ Dk
    _, mag, _ = sp.signal.bode((Acl, Bcl, Ccl, Dcl), w0)
    ax[1].semilogx(w0, mag, ls="-.", c="k", label="ФВЧ")
    # Low-pass filter
    A, B, C, D = copy.deepcopy(list(ss_dict["h2_case_5_g"].values()))
    Ak, Bk, Ck, Dk = copy.deepcopy(list(ss_dict["h2_case_5_k"].values()))
    Acl = np.block([[A + B @ Dk @ C, B @ Ck], [Bk @ C, Ak]])
    Bcl = S[:4, :4] @ np.array([[1, 0, 0, 0]]).T
    Bcl = np.block([[Bcl], [np.zeros((Ak.shape[0], 1))]])
    Ccl = np.block([[Dk @ C, Ck]])
    Dcl = D @ Dk
    _, mag, _ = sp.signal.bode((Acl, Bcl, Ccl, Dcl), w0)
    ax[1].semilogx(w0, mag, ls="-", c="k", label=r"ФНЧ")
    # Decoration
    ax[0].legend()
    ax[0].set_ylabel(
        r"$\vert W_\mathrm{з}(\dot{\mathrm{\vartheta}}, u_\mathrm{с})\vert \mathrm{, \ дБ}$"
    )
    ax[0].grid(True, which="both")
    ax[0].set_title("a)")
    ax[0].set_xlabel(r"$\mathrm{\omega, \  с^{-1}}$")
    ax[1].legend()
    ax[1].grid(True, which="both")
    ax[1].set_title("б)")
    ax[1].set_xlabel(r"$\mathrm{\omega, \  с^{-1}}$")
    return fig


def step_tacho_and_gear(ss_dict):
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.9))
    fig.canvas.manager.set_window_title("step case tacho and gear")
    # Parameters
    t0 = np.arange(0, 0.4, 5e-4)
    t1 = np.arange(0, 0.4, 5e-4)
    w0 = np.ones_like(t0) * -1 / 57.3 / 60
    w1 = np.ones_like(t1) * -1e-2
    # Tachogenerator
    A, B, C, D = copy.deepcopy(list(ss_dict["h2_case_1_g"].values()))
    Ak, Bk, Ck, Dk = copy.deepcopy(list(ss_dict["h2_case_1_k"].values()))
    Acl = A + B @ Dk @ C
    Bcl = B * Dk[:, 0]
    Ccl = C[0, :]
    Dcl = Dk @ D
    _, y0, _ = sp.signal.lsim((Acl, Bcl, Ccl, Dcl), w0, t0)
    ax[0].plot(t0, y0 * 57.3 * 60, ls="-", c="k", label=r"$k_\mathrm{\dot{\alpha}}$")
    Bcl = S[:4, :4] @ np.array([[0, 0, 1 / J1, 0]]).T
    t0 = np.arange(0, 0.4, 5e-4)
    _, y0, _ = sp.signal.lsim((Acl, Bcl, Ccl, Dcl), w1, t1)
    ax[1].plot(t0, y0 * 57.3 * 60, ls="-", c="k", label=r"$k_\mathrm{\dot{\alpha}}$")
    # Gear
    A, B, C, D = copy.deepcopy(list(ss_dict["h2_case_2_g"].values()))
    Ak, Bk, Ck, Dk = copy.deepcopy(list(ss_dict["h2_case_2_k"].values()))
    Acl = A + B @ Dk @ C
    Bcl = B * Dk[:, 0]
    Ccl = C[0, :]
    Dcl = Dk @ D
    _, y0, _ = sp.signal.lsim((Acl, Bcl, Ccl, Dcl), w0, t0)
    ax[0].plot(t0, y0 * 57.3 * 60, ls="--", c="k", label=r"$q$")
    ax[0].set_ylabel(r"$\mathrm{\vartheta_1, \ угл. \ мин.}$")
    ax[0].set_xlabel(r"$t, \mathrm{ \  с}$")
    ax[0].grid(True)
    ax[0].set_title("a)")
    Bcl = S[:4, :4] @ np.array([[0, 0, 1 / J1, 0]]).T
    _, y0, _ = sp.signal.lsim((Acl, Bcl, Ccl, Dcl), w1, t1)
    ax[1].plot(t0, y0 * 57.3 * 60, ls="--", c="k", label=r"$q$")
    # Decoration
    ax[0].set_ylabel(r"$\mathrm{\vartheta_1, \ угл. \ мин.}$")
    ax[0].set_xlabel(r"$t, \mathrm{ \  с}$")
    ax[0].grid(True)
    ax[0].legend()
    ax[0].set_title("a)")
    ax[1].set_ylabel(r"$\mathrm{\vartheta_1, \ угл. \ мин.}$")
    ax[1].set_xlabel(r"$t, \mathrm{ \  с}$")
    ax[1].grid(True)
    ax[1].legend()
    ax[1].set_title("б)")
    return fig


def step_notch(ss_dict):
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.9))
    fig.canvas.manager.set_window_title("step notch")
    # Parameters
    t0 = np.arange(0, 0.1, 5e-4)
    t1 = np.arange(0, 10, 5e-4)
    w0 = np.ones_like(t0) * -1 / 57.3 / 60
    w1 = np.ones_like(t1) * -1e-2
    # Notch-filter
    A, B, C, D = copy.deepcopy(list(ss_dict["h2_case_3_g"].values()))
    Ak, Bk, Ck, Dk = copy.deepcopy(list(ss_dict["h2_case_3_k"].values()))
    Acl = np.block([[A + B @ Dk @ C, B @ Ck], [Bk @ C, Ak]])
    Bcl = np.block([[np.zeros((A.shape[0], 1)) + B @ Dk], [Bk]])
    Ccl = np.block([[C, np.zeros((C.shape[0], Ak.shape[1]))]])
    Dcl = Dk @ D
    _, y0, _ = sp.signal.lsim((Acl, Bcl, Ccl, Dcl), w0, t0)
    ax[0].plot(t0 * 1e3, y0 * 57.3 * 60, ls="-", c="k", label=r"РЖ")
    Bcl = S[:4, :4] @ np.array([[0, 0, 1 / J1, 0]]).T
    Bcl = np.block([[Bcl], [np.zeros((Ak.shape[0], 1))]])
    _, y1, _ = sp.signal.lsim((Acl, Bcl, Ccl, Dcl), w1, t1)
    ax[1].plot(t1, y1 * 57.3 * 60, ls="-", c="k", label=r"РЖ")
    axins = ax[1].inset_axes([0.6, 0.15, 0.3, 0.3], xlim=(9.9, 10), ylim=(1.19, 1.25))
    axins.set_yticks([1.20, 1.25])
    axins.plot(t1, y1 * 57.3 * 60, "k-")
    ax[1].indicate_inset_zoom(axins, edgecolor="black")
    axins.grid()
    # Decoration
    ax[0].set_ylabel(r"$\mathrm{\vartheta_1, \ угл. \ мин.}$")
    ax[0].set_xlabel(r"$t, \mathrm{ \  мс}$")
    ax[0].grid(True)
    ax[0].legend(loc="lower right")
    ax[0].set_title("a)")
    ax[1].set_ylabel(r"$\mathrm{\vartheta_1, \ угл. \ мин.}$")
    ax[1].set_xlabel(r"$t, \mathrm{ \  с}$")
    ax[1].grid(True)
    ax[1].legend()
    ax[1].set_title("б)")
    return fig


def step_apf_lpf_hpf(ss_dict):
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.9))
    fig.canvas.manager.set_window_title("step all-pass, low-pass, high-pass filters")
    # Parameters
    t0 = np.arange(0, 0.075, 5e-4)
    t1 = np.arange(0, 0.2, 5e-4)
    w0 = np.ones_like(t0) * -1 / 57.3 / 60
    w1 = np.ones_like(t1) * -1e-2
    # All-pass filter
    A, B, C, D = copy.deepcopy(list(ss_dict["h2_case_4_g"].values()))
    Ak, Bk, Ck, Dk = copy.deepcopy(list(ss_dict["h2_case_4_k"].values()))
    Acl = np.block([[A + B @ Dk @ C, B @ Ck], [Bk @ C, Ak]])
    Bcl = np.block([[np.zeros((A.shape[0], 1)) + B @ Dk], [Bk]])
    Ccl = np.block([[C, np.zeros((C.shape[0], Ak.shape[1]))]])
    Dcl = Dk @ D
    _, y0, _ = sp.signal.lsim((Acl, Bcl, Ccl, Dcl), w0, t0)
    ax[0].plot(t0 * 1e3, y0 * 57.3 * 60, ls="-.", c="k", label=r"$\mathrm{ФФ}$")
    Bcl = S[:4, :4] @ np.array([[0, 0, 1 / J1, 0]]).T
    Bcl = np.block([[Bcl], [np.zeros((Ak.shape[0], 1))]])
    _, y1, _ = sp.signal.lsim((Acl, Bcl, Ccl, Dcl), w1, t1)
    ax[1].plot(t1 * 1e3, y1 * 57.3 * 60, ls="-.", c="k", label=r"$\mathrm{ФФ}$")
    # Low-pass filter
    A, B, C, D = copy.deepcopy(list(ss_dict["h2_case_5_g"].values()))
    Ak, Bk, Ck, Dk = copy.deepcopy(list(ss_dict["h2_case_5_k"].values()))
    Acl = np.block([[A + B @ Dk @ C, B @ Ck], [Bk @ C, Ak]])
    Bcl = np.block([[np.zeros((A.shape[0], 1)) + B @ Dk], [Bk]])
    Ccl = np.block([[C, np.zeros((C.shape[0], Ak.shape[1]))]])
    Dcl = Dk @ D
    _, y0, _ = sp.signal.lsim((Acl, Bcl, Ccl, Dcl), w0, t0)
    ax[0].plot(t0 * 1e3, y0 * 57.3 * 60, ls="--", c="k", label=r"$\mathrm{ФНЧ}$")
    Bcl = S[:4, :4] @ np.array([[0, 0, 1 / J1, 0]]).T
    Bcl = np.block([[Bcl], [np.zeros((Ak.shape[0], 1))]])
    _, y1, _ = sp.signal.lsim((Acl, Bcl, Ccl, Dcl), w1, t1)
    ax[1].plot(t1 * 1e3, y1 * 57.3 * 60, ls="--", c="k", label=r"$\mathrm{ФНЧ}$")
    # High-pass filter
    A, B, C, D = copy.deepcopy(list(ss_dict["h2_case_7_g"].values()))
    Ak, Bk, Ck, Dk = copy.deepcopy(list(ss_dict["h2_case_7_k"].values()))
    Acl = np.block([[A + B @ Dk @ C, B @ Ck], [Bk @ C, Ak]])
    Bcl = np.block([[np.zeros((A.shape[0], 1)) + B @ Dk], [Bk]])
    Ccl = np.block([[C, np.zeros((C.shape[0], Ak.shape[1]))]])
    Dcl = Dk @ D
    _, y0, _ = sp.signal.lsim((Acl, Bcl, Ccl, Dcl), w0, t0)
    ax[0].plot(t0 * 1e3, y0 * 57.3 * 60, ls="-", c="k", label=r"$\mathrm{ФВЧ}$")
    Bcl = S[:4, :4] @ np.array([[0, 0, 1 / J1, 0]]).T
    Bcl = np.block([[Bcl], [np.zeros((Ak.shape[0], 1))]])
    _, y1, _ = sp.signal.lsim((Acl, Bcl, Ccl, Dcl), w1, t1)
    ax[1].plot(t1 * 1e3, y1 * 57.3 * 60, ls="-", c="k", label=r"$\mathrm{ФВЧ}$")
    # Decoration
    ax[0].set_ylabel(r"$\mathrm{\vartheta_1, \ угл. \ мин.}$")
    ax[0].set_xlabel(r"$t, \mathrm{ \  мс}$")
    ax[0].grid(True)
    ax[0].legend()
    ax[0].set_title("a)")
    ax[1].set_ylabel(r"$\mathrm{\vartheta_1, \ угл. \ мин.}$")
    ax[1].set_xlabel(r"$t, \mathrm{ \  мс}$")
    ax[1].grid(True)
    ax[1].legend(loc="lower right")
    ax[1].set_title("б)")
    return fig


def step_mixed_synth(ss_dict):
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.9))
    fig.canvas.manager.set_window_title("step mixed synth")
    # Paramters
    t0 = np.arange(0, 0.15, 5e-4)
    t1 = np.arange(0, 0.4, 5e-4)
    w0 = np.ones_like(t0) * -1 / 57.3 / 60
    w1 = np.ones_like(t1) * -1e-2
    # without H inf
    A, B, C, D = copy.deepcopy(list(ss_dict["mixed_syn_w_o_hinf_g"].values()))
    Ak, Bk, Ck, Dk = copy.deepcopy(list(ss_dict["mixed_syn_w_o_hinf_k"].values()))
    Acl = np.block([[A + B @ Dk @ C, B @ Ck], [Bk @ C, Ak]])
    Bcl = np.block([[np.zeros((A.shape[0], 1)) + B @ Dk], [Bk]])
    Ccl = np.block([[C, np.zeros((C.shape[0], Ak.shape[1]))]])
    Dcl = Dk @ D
    _, y0, _ = sp.signal.lsim((Acl, Bcl, Ccl, Dcl), w0, t0)
    ax[0].plot(t0 * 1e3, y0 * 57.3 * 60, ls="-", c="k", label=r"без $H_\infty$")
    Bcl = S[:4, :4] @ np.array([[0, 0, 1 / J1, 0]]).T
    Bcl = np.block([[Bcl], [np.zeros((Ak.shape[0], 1))]])
    _, y1, _ = sp.signal.lsim((Acl, Bcl, Ccl, Dcl), w1, t1)
    ax[1].plot(t1, y1 * 57.3 * 60, ls="-", c="k", label=r"без $H_\infty$")
    # with H inf
    A, B, C, D = copy.deepcopy(list(ss_dict["mixed_syn_w_hinf_g"].values()))
    Ak, Bk, Ck, Dk = copy.deepcopy(list(ss_dict["mixed_syn_w_hinf_k"].values()))
    Acl = np.block([[A + B @ Dk @ C, B @ Ck], [Bk @ C, Ak]])
    Bcl = np.block([[np.zeros((A.shape[0], 1)) + B @ Dk], [Bk]])
    Ccl = np.block([[C, np.zeros((C.shape[0], Ak.shape[1]))]])
    Dcl = Dk @ D
    _, y0, _ = sp.signal.lsim((Acl, Bcl, Ccl, Dcl), w0, t0)
    ax[0].plot(t0 * 1e3, y0 * 57.3 * 60, ls="--", c="k", label=r"c $H_\infty$")
    Bcl = S[:4, :4] @ np.array([[0, 0, 1 / J1, 0]]).T
    Bcl = np.block([[Bcl], [np.zeros((Ak.shape[0], 1))]])
    _, y1, _ = sp.signal.lsim((Acl, Bcl, Ccl, Dcl), w1, t1)
    ax[1].plot(t1, y1 * 57.3 * 60, ls="--", c="k", label=r"c $H_\infty$")
    # Decoration
    ax[0].set_ylabel(r"$\mathrm{\vartheta_1, \ угл. \ мин.}$")
    ax[0].set_xlabel(r"$t, \mathrm{ \  мс}$")
    ax[0].grid(True)
    ax[0].legend(loc="lower right")
    ax[0].set_title("a)")
    ax[1].set_ylabel(r"$\mathrm{\vartheta_1, \ угл. \ мин.}$")
    ax[1].set_xlabel(r"$t, \mathrm{ \  с}$")
    ax[1].grid(True)
    ax[1].legend()
    ax[1].set_title("б)")
    return fig


def bode_result_controller(ss_dict):
    fig, ax = plt.subplots(1, 2, sharex="col", figsize=(7.2, 2.6), frameon=False)
    ax0 = ax[0].twinx()
    ax1 = ax[1].twinx()
    fig.canvas.manager.set_window_title("bode_result_controller")
    w0 = np.logspace(-1, 3, 500)
    w1 = np.logspace(0, 3, 500)
    # with the H inf constr
    A, B, C, D = copy.deepcopy(list(ss_dict["h2_case_5_i_g"].values()))
    Ak, Bk, Ck, Dk = copy.deepcopy(list(ss_dict["h2_case_5_i_k"].values()))
    C[0, :] = C[0, :] * -1
    As, Bs, Cs, Ds = connect_series_siso((A, B, C, D), (Ak, Bk, Ck, Dk))
    _, mag, phase = sp.signal.bode((Ak, Bk, Ck, Dk), w0)
    (line1,) = ax[0].semilogx(
        w0, mag, ls="-", c="k", label=r"$\vert k_\mathrm{с} W_\mathrm{ф}\vert$"
    )
    (line2,) = ax0.semilogx(
        w0, phase, ls="--", c="k", label=r"$\angle k_\mathrm{с} W_\mathrm{ф}$"
    )
    _, mag, phase = sp.signal.bode((As, Bs, Cs, Ds), w1)
    (line3,) = ax[1].semilogx(
        w1,
        mag,
        ls="-",
        c="k",
        label=r"$\vert W_\mathrm{р}(\mathrm{\vartheta}, \mathrm{\vartheta_1})\vert$",
    )
    (line4,) = ax1.semilogx(
        w1,
        phase,
        ls="--",
        c="k",
        label=r"$\angle W_\mathrm{р}(\mathrm{\vartheta}, \mathrm{\vartheta_1})$",
    )
    # Decoration
    ax[1].set_ylabel(
        r"$\vert W_\mathrm{р}(\mathrm{\vartheta}, \mathrm{\vartheta_1})\vert \mathrm{, \ дБ}$"
    )
    ax[0].set_ylabel(r"$\vert k_\mathrm{с} W_\mathrm{ф}\vert \mathrm{, \ дБ}$")
    ax1.set_ylabel(
        r"$\angle W_\mathrm{р}(\mathrm{\vartheta}, \mathrm{\vartheta_1}) \mathrm{, \ град.}$"
    )
    ax0.set_ylabel(r"$\angle k_\mathrm{с} W_\mathrm{ф} \mathrm{, \ град.}$")
    ax0.set_yticks([0, -90, -180])
    ax1.set_yticks([0, -90, -180, -270, -360, -450])
    ax0.legend(handles=[line1, line2], loc=3)
    ax1.legend(handles=[line3, line4], loc=3)
    ax[0].set_title("a)")
    ax[1].set_title("б)")
    return fig


def digi_bode(ss_dict):
    A, B, C, D = copy.deepcopy(list(ss_dict["h2_case_5_g"].values()))
    num, den = tf_to_num_den(ss2tf((A, B, C * -1945, D))[0])
    vec = []
    w = np.logspace(0, 3, 10000)
    Ts = 13e-3
    for w_ in w:
        num_ = [a * (1j * w_) ** i for i, a in enumerate(num[::-1])]
        den_ = [a * (1j * w_) ** i for i, a in enumerate(den[::-1])]
        vec.append(sum(num_) / sum(den_) * (1 - np.exp(-1j * w_ * Ts)) / (1j * w_ * Ts))
    mag = 20 * np.log10(np.abs(vec))
    phase = np.angle(vec, deg=True)
    phase = np.unwrap(phase)
    fig, ax = plt.subplots(1, 1, figsize=(7.0, 2.6))
    fig.canvas.manager.set_window_title("digi_bode")
    ax2 = ax.twinx()
    (line1,) = ax.semilogx(
        w,
        mag,
        "k-",
        label=r"$\vert W_\mathrm{р}(\mathrm{\vartheta}, \mathrm{\vartheta_1})\vert$",
    )
    (line2,) = ax2.semilogx(
        w,
        phase,
        "k--",
        label=r"$\angle W_\mathrm{р}(\mathrm{\vartheta}, \mathrm{\vartheta_1})$",
    )
    ax2.set_yticks([-90, -180, -270, -360, -450])
    ax2.set_ylabel(r"$\angle W_\mathrm{р}(\mathrm{\vartheta}, \mathrm{\vartheta_1}) \mathrm{, \ град.}$")
    ax.set_ylabel(r"$\vert W_\mathrm{р}(\mathrm{\vartheta}, \mathrm{\vartheta_1})\vert \mathrm{, \ дБ}$")
    ax.set_xlabel(r"$\mathrm{\omega, \  с^{-1}}$")
    ax2.legend(handles=[line1, line2], loc="lower left")
    return fig


def step_result_controller(ss_dict):
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.9))
    fig.canvas.manager.set_window_title("step_result_controller")
    # Paramters
    t0 = np.arange(0, 0.15, 5e-4)
    t1 = np.arange(0, 0.4, 5e-4)
    w0 = np.ones_like(t0) * -1 / 57.3 / 60
    w1 = np.ones_like(t1) * -1e-2
    # without H inf
    A, B, C, D = copy.deepcopy(list(ss_dict["h2_case_5_i_g"].values()))
    Ak, Bk, Ck, Dk = copy.deepcopy(list(ss_dict["h2_case_5_i_k"].values()))
    Acl = np.block([[A + B @ Dk @ C, B @ Ck], [Bk @ C, Ak]])
    Bcl = np.block([[np.zeros((A.shape[0], 1)) + B @ Dk], [Bk]])
    Ccl = np.block([[C, np.zeros((C.shape[0], Ak.shape[1]))]])
    Dcl = Dk @ D
    _, y0, _ = sp.signal.lsim((Acl, Bcl, Ccl, Dcl), w0, t0)
    ax[0].plot(t0 * 1e3, y0 * 57.3 * 60, ls="-", c="k")
    Bcl = S[:4, :4] @ np.array([[0, 0, 1 / J1, 0]]).T
    Bcl = np.block([[Bcl], [np.zeros((Ak.shape[0], 1))]])
    _, y1, _ = sp.signal.lsim((Acl, Bcl, Ccl, Dcl), w1, t1)
    ax[1].plot(t1, y1 * 57.3 * 60, ls="-", c="k")
    # Decoration
    ax[0].set_ylabel(r"$\mathrm{\vartheta_1, \ угл. \ мин.}$")
    ax[0].set_xlabel(r"$t, \mathrm{ \  мс}$")
    ax[0].grid(True)
    ax[0].set_title("a)")
    ax[1].set_ylabel(r"$\mathrm{\vartheta_1, \ угл. \ мин.}$")
    ax[1].set_xlabel(r"$t, \mathrm{ \  с}$")
    ax[1].grid(True)
    ax[1].set_title("б)")
    return fig


if __name__ == "__main__":
    # Load state space matrices
    file_path = Path(__file__).parent.resolve() / "sl_controller.json"
    with open(file_path, "r", newline="") as file:
        ss_dict = json.load(file)
    for ss in ss_dict:
        for key, val in ss_dict[ss].items():
            if val is not None:
                ss_dict[ss][key] = np.asarray(val)
    # Make plots
    figures = {}
    figures["bode_h2_obsv_noise"] = bode_h2_obsv_noise(ss_dict)
    figures["bode_h2_test_with_corr_loop"] = bode_h2_test_with_corr_loop(ss_dict)
    figures["bode_h2_ellips"] = bode_h2_ellips(ss_dict)
    figures["bode_h2_ellips_reduced"] = bode_h2_ellips_reduced(ss_dict)
    figures["bode_mixed_ellips_hinf"] = bode_mixed_ellips_hinf(ss_dict)
    figures["bode_fixed_struct_h2"] = bode_fixed_struct_h2(ss_dict)
    figures["bode_fixed_struct_ellips"] = bode_fixed_struct_ellips(ss_dict)
    figures["mag_dist_to_theta"] = mag_dist_to_theta(ss_dict)
    figures["mag_cmd_der_to_u"] = mag_cmd_der_to_u(ss_dict)
    figures["step_tacho_and_gear"] = step_tacho_and_gear(ss_dict)
    figures["step_notch"] = step_notch(ss_dict)
    figures["step_apf_lpf_hpf"] = step_apf_lpf_hpf(ss_dict)
    figures["step_mixed_synth"] = step_mixed_synth(ss_dict)
    figures["digi_bode"] = digi_bode(ss_dict)
    figures["bode_result_controller"] = bode_result_controller(ss_dict)
    figures["step_result_controller"] = step_result_controller(ss_dict)
    # Save or show
    if len(sys.argv) != 2:
        plt.show()
    else:
        plot_dir_path = Path(sys.argv[1])
        if not plot_dir_path.is_dir():
            plot_dir_path.mkdir()
        for name in figures:
            figures[name].savefig(plot_dir_path / f"{name:s}.svg")
