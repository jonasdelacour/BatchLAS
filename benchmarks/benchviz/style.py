"""Journal figure style: LaTeX Computer Modern, real column widths, 8 pt text.

Sizes are the printed sizes. A figure made at 3.5 in and \\includegraphics'd at
\\columnwidth is typeset 1:1, so its 8 pt labels match an 8-9 pt caption -- the
opposite of the 20 x 10 in, 30 pt figures that shrink to illegibility.

Colour roles (validated with the dataviz palette validator, light surface):
  precisions  S/D/C/Z  categorical slots 1-4, always with a distinct marker AND a
                       direct label (slots 3-4 are under 3:1 on white; the label
                       is the required relief)
  libraries            BatchLAS in blue; the comparator in secondary ink, dashed,
                       open markers -- the subject is the one in colour
  speedup              diverging blue (BatchLAS faster) / red (vendor faster)
                       around a neutral gray at exactly 1x, on a log2 scale
"""
from __future__ import annotations

import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SINGLE_COL = 3.5    # in; IEEE/ACM/SIAM single column is 3.3-3.5
DOUBLE_COL = 7.16   # in; IEEE double column

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
MISSING = "#f4f3f0"

PREC_COLOR = {"float": "#2a78d6", "double": "#eb6834", "cfloat": "#1baf7a", "cdouble": "#eda100"}
PREC_MARKER = {"float": "o", "double": "s", "cfloat": "^", "cdouble": "D"}
LIB_COLOR = {"batchlas": "#2a78d6", "vendor": INK_2}

SPEEDUP_CMAP = LinearSegmentedColormap.from_list(
    "speedup",
    ["#7a1f1f", "#c93a39", "#ee9491", "#f0efec", "#9ec5f4", "#2a78d6", "#0d366b"],
)

_applied = False


def usetex_available() -> bool:
    return all(shutil.which(b) for b in ("latex", "dvipng")) and bool(shutil.which("kpsewhich"))


def apply(usetex: bool | None = None) -> bool:
    """Set rcParams. Returns whether LaTeX is in use."""
    global _applied
    if usetex is None:
        usetex = usetex_available()
    rc = plt.rcParams
    rc.update({
        "text.usetex": usetex,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.6,
        "axes.edgecolor": INK_2,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.which": "major",
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.4,
        "grid.linestyle": "-",
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.labelcolor": INK,
        "ytick.labelcolor": INK,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "lines.linewidth": 1.1,
        "lines.markersize": 3.6,
        "lines.markeredgewidth": 0.7,
        "legend.frameon": False,
        "legend.handlelength": 1.8,
        "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.2,
        "legend.borderaxespad": 0.2,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "savefig.transparent": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "hatch.linewidth": 0.4,
        "hatch.color": GRID,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 0.02,
        "figure.constrained_layout.w_pad": 0.02,
    })
    _applied = True
    return usetex


def tex(s: str) -> str:
    """Escape the characters LaTeX treats specially in running text."""
    if not plt.rcParams["text.usetex"]:
        return s
    for a, b in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"), ("&", r"\&"), ("#", r"\#")):
        s = s.replace(a, b)
    return s


def mono(s: str) -> str:
    return rf"\texttt{{{tex(s)}}}" if plt.rcParams["text.usetex"] else s


def times(x: float) -> str:
    """A speedup tick label: 0.25x, 1x, 4x."""
    if x >= 1:
        v = f"{x:g}"
    else:
        v = f"{x:.3g}".rstrip("0").rstrip(".")
    return rf"{v}$\times$"
