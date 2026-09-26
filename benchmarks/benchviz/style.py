"""The house figure style: plotting/stylesheet.py, as used for the PASC'24
dualization paper and the MSc thesis.

Figures are drawn large (20 x 10 in, 30 pt) and scaled down by LaTeX, which
is what gives the thin lines, big markers and full-box frames their look.
Line plots: dotted connectors, markers o ^ s * D, a gray 2-sigma band, a
frameless legend with enlarged markers, red dashed reference lines. Maps:
viridis, a faint cell grid, bold column titles, a bracketed colorbar label.
"""
from __future__ import annotations

import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from cycler import cycler  # noqa: E402

FONTSIZE = 30
PANEL = (20, 10)          # one line-plot panel
MAP_PANEL = (10, 10)      # one 2D-map panel

# stylesheet.py's colour dictionary, in its order.
CD = ["#1f77b4", "#e377c2", "#0D9276", "#8c564b", "#7570b3", "#d95f02", "#e7298a", "#66a61e", "#8931EF"]
MARKERS = ["o", "^", "s", "*", "D"]
MARKER_SCALES = [1.1, 1.25, 1.0, 1.5, 1.0]

PREC_ORDER = ["float", "double", "cfloat", "cdouble"]
PREC_COLOR = dict(zip(PREC_ORDER, CD))
PREC_MARKER = dict(zip(PREC_ORDER, MARKERS))
PREC_MSCALE = dict(zip(PREC_ORDER, MARKER_SCALES))
LIB_COLOR = {"batchlas": CD[0], "vendor": CD[3]}
LIB_MARKER = {"batchlas": "o", "vendor": "*"}
LIB_MSCALE = {"batchlas": 1.1, "vendor": 1.5}
BAND = "#d3d3d3"
REF = "red"
CMAP = "viridis"


def usetex_available() -> bool:
    return all(shutil.which(b) for b in ("latex", "dvipng", "kpsewhich"))


def apply(usetex: bool | None = None) -> bool:
    if usetex is None:
        usetex = usetex_available()
    rc = plt.rcParams
    rc.update({
        "text.usetex": usetex,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "text.latex.preamble": r"\usepackage{amssymb}\usepackage{amsmath}",
        "font.size": FONTSIZE,
        "figure.figsize": PANEL,
        "lines.markersize": 10,
        "lines.linewidth": 1.5,
        "lines.markeredgecolor": matplotlib.colors.to_rgba("black", 0.5),
        "lines.markeredgewidth": 0.01,
        "legend.markerscale": 2.0,
        "legend.framealpha": 0,
        "legend.frameon": False,
        "legend.labelspacing": 0.1,
        "legend.fontsize": int(FONTSIZE * 0.9),
        "legend.loc": "upper left",
        "axes.autolimit_mode": "data",
        "axes.xmargin": 0,
        "axes.ymargin": 0.10,
        "axes.titlesize": FONTSIZE,
        "axes.labelsize": FONTSIZE,
        "axes.linewidth": 1.0,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.prop_cycle": cycler(color=CD),
        "grid.linestyle": "-",
        "grid.alpha": 0.2,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.labelsize": FONTSIZE,
        "ytick.labelsize": FONTSIZE,
        "figure.autolayout": False,
        "figure.constrained_layout.use": True,
        "savefig.dpi": 100,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    return usetex


def tex(s: str) -> str:
    if not plt.rcParams["text.usetex"]:
        return s
    for a, b in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"), ("&", r"\&"), ("#", r"\#")):
        s = s.replace(a, b)
    return s


def bold(s: str) -> str:
    return rf"\textbf{{{tex(s)}}}" if plt.rcParams["text.usetex"] else s


def mono(s: str) -> str:
    return rf"\texttt{{{tex(s)}}}" if plt.rcParams["text.usetex"] else s


def times(x: float) -> str:
    v = f"{x:g}" if x >= 1 else f"{x:.3g}"
    return rf"{v}$\times$"


def series(ax, x, y, color, marker, mscale=1.0, label=None, band=None, ls=":", zorder=3):
    """One series in the house style: dotted connector, big marker, optional
    band = (lo, hi) drawn as the gray 2-sigma fill."""
    if band is not None:
        ax.fill_between(x, band[0], band[1], color=BAND, alpha=0.6, lw=0, zorder=1)
    ax.plot(x, y, ls=ls, color=color, marker=marker, ms=10 * mscale, label=label, zorder=zorder)


def band_handle():
    from matplotlib.patches import Patch
    return Patch(facecolor=BAND, alpha=0.6, edgecolor="none", label=r"$2\sigma$")


def outline(ax):
    """Full box, as every panel in the house style has."""
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(1.0)
        s.set_color("black")


def log2_ticks(lo: float, hi: float, max_ticks: int = 8):
    k0, k1 = int(np.floor(np.log2(lo))), int(np.ceil(np.log2(hi)))
    step = max(1, int(np.ceil((k1 - k0 + 1) / max_ticks)))
    return [2.0 ** k for k in range(k0, k1 + 1, step)]
