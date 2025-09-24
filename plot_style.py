import locale
import matplotlib as mpl


def set_plot_style():
    locale.setlocale(locale.LC_ALL, "ru_RU.UTF-8")
    mpl.rcParams["mathtext.fontset"] = "custom"
    mpl.rcParams["font.serif"] = "XITS"
    mpl.rcParams["mathtext.rm"] = "serif"
    mpl.rcParams["mathtext.it"] = "serif:italic"
    mpl.rcParams["axes.formatter.use_locale"] = True
    mpl.rcParams["text.usetex"] = False
    mpl.rcParams["font.family"] = "XITS"
    mpl.rcParams["font.size"] = 12
    mpl.rcParams["axes.labelsize"] = 13
    mpl.rcParams["grid.linestyle"] = ":"
    mpl.rcParams["grid.color"] = "k"
    mpl.rcParams["grid.linewidth"] = 0.25 * 72 / 25.4
    mpl.rcParams["lines.linewidth"] = 0.37 * 72 / 25.4
    mpl.rcParams["axes.linewidth"] = 0.37 * 72 / 25.4
    mpl.rcParams["legend.framealpha"] = 1
    mpl.rcParams["savefig.bbox"] = "tight"
    mpl.rcParams["savefig.pad_inches"] = 0.05
    mpl.rcParams["figure.constrained_layout.use"] = True
    mpl.rcParams["axes.grid"] = True
