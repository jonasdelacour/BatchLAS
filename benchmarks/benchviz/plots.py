"""Figures for one campaign. Every figure is derived from results.jsonl alone.

Per op:
  speedup_n     vendor time / BatchLAS time at the saturated batch, vs n,
                one line per precision                         (single column)
  throughput_n  both libraries' throughput vs n at the saturated batch, one
                panel per precision                            (double column)
  heatmap       the speedup over the whole n x batch grid, one panel per
                precision                                      (double column)
Per campaign:
  summary_<t>   op x n speedup table at saturation              (single column)

"Saturated batch" is the largest batch at which BOTH arms produced a verified
result for that (op, precision, n). An unsaturated ratio is a ratio of launch
overheads (docs/perf/README.md), which is why the n-figures never average
across batches.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

import style
from ops import OPS, TYPE_LABEL, TYPE_PREFIX, TYPES

style.apply()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import BoundaryNorm, Normalize  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter, NullLocator  # noqa: E402

KEYS = ["op", "dtype", "m", "n", "nrhs", "batch"]
SPEEDUP_LIM = 3.0  # log2: the colour scale saturates at 8x either way


# ----------------------------------------------------------------- data
def paired(rows: List[dict]) -> pd.DataFrame:
    """One row per cell with both arms side by side; only verified pairs."""
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df[df.get("ok", False) == True]  # noqa: E712
    if df.empty or "time_ms" not in df:
        return pd.DataFrame()
    # A resumed campaign can hold a cell twice; the newest measurement wins.
    df = df.sort_values("t").drop_duplicates(KEYS + ["arm"], keep="last")
    cols = ["time_ms", "rel_sd", "route", "vendor_free"]
    for c in cols:
        if c not in df:
            df[c] = None
    w = df.pivot_table(index=KEYS, columns="arm", values=cols, aggfunc="first")
    w.columns = [f"{a}_{b}" for a, b in w.columns]
    w = w.reset_index()
    need = ["time_ms_batchlas", "time_ms_vendor"]
    if not all(c in w for c in need):
        return pd.DataFrame()
    w = w.dropna(subset=need)
    w["speedup"] = w["time_ms_vendor"].astype(float) / w["time_ms_batchlas"].astype(float)
    return w


def throughput(op: str, dtype: str, m, n, nrhs, batch, time_ms) -> float:
    spec = OPS[op]
    per_s = batch / (np.asarray(time_ms, float) * 1e-3)
    if spec.flops is None:
        return per_s
    return per_s * spec.flops(m, n, nrhs, dtype) * 1e-9


def throughput_unit(op: str) -> str:
    return "matrices/s" if OPS[op].flops is None else "GFLOP/s"


def saturated(w: pd.DataFrame) -> pd.DataFrame:
    if w.empty:
        return w
    idx = w.groupby(["op", "dtype", "n"])["batch"].idxmax()
    return w.loc[idx].sort_values(["op", "dtype", "n"])


# ----------------------------------------------------------------- axes helpers
def _pow2_axis(ax, values: Sequence[int], axis="x"):
    """Log2 axis: powers of two labelled, every other measured order a minor tick."""
    vals = sorted(set(int(v) for v in values))
    set_scale = ax.set_xscale if axis == "x" else ax.set_yscale
    set_scale("log", base=2)
    a = ax.xaxis if axis == "x" else ax.yaxis
    lo, hi = min(vals), max(vals)
    pows = [2 ** k for k in range(math.floor(math.log2(lo)), math.ceil(math.log2(hi)) + 1)
            if lo <= 2 ** k <= hi] or vals
    if len(pows) > 8:  # thin to every other power so labels never touch
        pows = pows[::2]
    a.set_major_locator(FixedLocator(pows))
    a.set_major_formatter(FuncFormatter(lambda v, _: f"{int(round(v))}"))
    a.set_minor_locator(FixedLocator([v for v in vals if v not in pows]))
    a.set_minor_formatter(NullFormatter())
    (ax.set_xlim if axis == "x" else ax.set_ylim)(lo / 2 ** 0.25, hi * 2 ** 0.25)


def _speedup_yaxis(ax, lo: float, hi: float):
    lo = min(lo, 0.9)
    hi = max(hi, 1.1)
    ax.set_yscale("log", base=2)
    k0, k1 = math.floor(math.log2(lo)), math.ceil(math.log2(hi))
    if k1 - k0 > 6:  # keep the tick count readable on a wide range
        ticks = [2.0 ** k for k in range(k0, k1 + 1, 2)]
        if 1.0 not in ticks:
            ticks.append(1.0)
    else:
        ticks = [2.0 ** k for k in range(k0, k1 + 1)]
    ax.yaxis.set_major_locator(FixedLocator(sorted(ticks)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: style.times(v)))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_ylim(lo / 1.15, hi * 1.15)


def _caption(fig, text: str):
    fig.text(1.0, -0.005, text, ha="right", va="top", fontsize=6, color=style.MUTED)


def _title(ax_or_fig, op: str, extra: str = "", fig_level=False):
    s = OPS[op]
    t = rf"{style.mono(op)}\,\,{style.tex(s.title)}" if plt.rcParams["text.usetex"] else f"{op}  {s.title}"
    if extra:
        t += f"  {extra}"
    if fig_level:
        ax_or_fig.suptitle(t, x=0.0, ha="left", fontsize=8)
    else:
        ax_or_fig.set_title(t, loc="left", fontsize=8, pad=4)


def _dedupe_labels(ys: List[float], min_gap: float) -> List[float]:
    """Nudge direct labels apart (in log2 units) so none overlap."""
    order = np.argsort(ys)
    out = list(ys)
    for i in range(1, len(order)):
        a, b = order[i - 1], order[i]
        if out[b] - out[a] < min_gap:
            out[b] = out[a] + min_gap
    return out


# ----------------------------------------------------------------- figures
def fig_speedup_n(w: pd.DataFrame, op: str, meta: dict, titles=True):
    s = saturated(w[w.op == op])
    if s.empty:
        return None
    fig, ax = plt.subplots(figsize=(style.SINGLE_COL, 2.35))
    vendor = meta.get("vendor_label", "vendor")
    ax.axhline(1.0, color=style.AXIS, lw=0.9, zorder=1)
    ends = []
    any_hollow = False
    for t in TYPES:
        d = s[s.dtype == t]
        if d.empty:
            continue
        c, mk = style.PREC_COLOR[t], style.PREC_MARKER[t]
        ax.plot(d.n, d.speedup, color=c, lw=1.1, zorder=3)
        vf = d.get("vendor_free_batchlas", pd.Series(True, index=d.index)).fillna(True).astype(bool)
        if OPS[op].setup_ops:  # undetermined, see OpSpec.setup_ops; also fixes rows recorded before it
            vf[:] = True
        any_hollow |= bool((~vf).any())
        ax.plot(d.n[vf], d.speedup[vf], ls="none", marker=mk, color=c, mec="white", mew=0.5, zorder=4)
        ax.plot(d.n[~vf], d.speedup[~vf], ls="none", marker=mk, mfc="white", mec=c, mew=0.8, zorder=4)
        ends.append((t, d.n.iloc[-1], d.speedup.iloc[-1]))
    lo, hi = float(s.speedup.min()), float(s.speedup.max())
    _pow2_axis(ax, s.n)
    _speedup_yaxis(ax, lo, hi)
    # Direct labels at each line's right end, nudged apart.
    if ends:
        ys = _dedupe_labels([math.log2(e[2]) for e in ends],
                            min_gap=0.075 * (math.log2(ax.get_ylim()[1]) - math.log2(ax.get_ylim()[0])))
        for (t, x, _), y in zip(ends, ys):
            ax.annotate(TYPE_PREFIX[t], (x, 2 ** y), xytext=(5, 0), textcoords="offset points",
                        va="center", ha="left", fontsize=7, color=style.INK, annotation_clip=False)
    ax.set_xlabel(r"matrix order $n$" if plt.rcParams["text.usetex"] else "matrix order n")
    ax.set_ylabel(style.tex(f"speedup over {vendor}"))
    ax.grid(axis="x", visible=False)
    handles = [Line2D([], [], color=style.PREC_COLOR[t], marker=style.PREC_MARKER[t], mec="white", mew=0.5,
                      label=f"{TYPE_PREFIX[t]}  {TYPE_LABEL[t]}") for t in TYPES if (s.dtype == t).any()]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=2, fontsize=6.5,
              handlelength=1.6, columnspacing=1.0, borderaxespad=0.0)
    if titles:
        _title(fig, op, fig_level=True)
    notes = " / ".join(n for n in (meta.get("device", ""), "batch at saturation", "above 1: BatchLAS faster") if n)
    if any_hollow:
        notes += "\nopen marker: the BatchLAS arm calls a vendor sub-op"
    _caption(fig, style.tex(notes))
    return fig


def fig_throughput_n(w: pd.DataFrame, op: str, meta: dict, titles=True):
    s = saturated(w[w.op == op])
    if s.empty:
        return None
    types = [t for t in TYPES if (s.dtype == t).any()]
    fig, axes = plt.subplots(1, len(types), figsize=(style.DOUBLE_COL if len(types) > 2 else style.SINGLE_COL * 1.6, 2.0),
                             squeeze=False)
    vendor = meta.get("vendor_label", "vendor")
    unit = throughput_unit(op)
    for ax, t in zip(axes[0], types):
        d = s[s.dtype == t]
        yb = throughput(op, t, d.m, d.n, d.nrhs, d.batch, d.time_ms_batchlas)
        yv = throughput(op, t, d.m, d.n, d.nrhs, d.batch, d.time_ms_vendor)
        ax.plot(d.n, yv, color=style.LIB_COLOR["vendor"], ls=(0, (3.5, 1.8)), lw=1.0,
                marker="o", mfc="white", mec=style.LIB_COLOR["vendor"], mew=0.8, ms=3.2, zorder=3)
        ax.plot(d.n, yb, color=style.LIB_COLOR["batchlas"], lw=1.2,
                marker="o", mec="white", mew=0.5, ms=3.6, zorder=4)
        _pow2_axis(ax, d.n)
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10))
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.grid(axis="x", visible=False)
        ax.set_title(style.tex(TYPE_LABEL[t]) + rf" ({TYPE_PREFIX[t]})", fontsize=7.5, loc="left", pad=3)
        ax.set_xlabel(r"$n$" if plt.rcParams["text.usetex"] else "n")
        if len(d.n) > 5:
            ax.tick_params(axis="x", labelsize=6)
    axes[0][0].set_ylabel(style.tex(unit))
    handles = [Line2D([], [], color=style.LIB_COLOR["batchlas"], lw=1.2, marker="o", mec="white", mew=0.5,
                      label="BatchLAS"),
               Line2D([], [], color=style.LIB_COLOR["vendor"], lw=1.0, ls=(0, (3.5, 1.8)), marker="o",
                      mfc="white", mec=style.LIB_COLOR["vendor"], label=style.tex(vendor))]
    fig.legend(handles=handles, loc="outside upper right", ncol=2, fontsize=7)
    if titles:
        _title(fig, op, fig_level=True)
    _caption(fig, style.tex(" / ".join(x for x in (meta.get("device", ""), "batch at saturation") if x)))
    return fig


def _cell_text(v: float) -> str:
    """At most three characters, so a value always fits its cell: 210, 13, 2.1, .43"""
    if v >= 9.95:
        return f"{v:.0f}"
    if v >= 0.995:
        return f"{v:.1f}"
    return f"{v:.2f}".lstrip("0")


def _hatch_background(ax):
    """Unmeasured cells: white with a hairline hatch. A flat pale gray would be
    indistinguishable from the diverging map's 1x midpoint."""
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor="white",
                           edgecolor=style.GRID, hatch="//////", lw=0, zorder=0))


def _batch_label(b: int) -> str:
    k = int(round(math.log2(b)))
    if 2 ** k == b:
        return rf"$2^{{{k}}}$"
    return f"{b}"


def fig_heatmap(w: pd.DataFrame, op: str, meta: dict, titles=True):
    d0 = w[w.op == op]
    if d0.empty:
        return None
    types = [t for t in TYPES if (d0.dtype == t).any()]
    vendor = meta.get("vendor_label", "vendor")
    width = style.DOUBLE_COL if len(types) > 2 else style.SINGLE_COL * 1.6
    fig, axes = plt.subplots(1, len(types), figsize=(width, 2.25), squeeze=False)
    norm = Normalize(vmin=-SPEEDUP_LIM, vmax=SPEEDUP_LIM)
    im = None
    for ax, t in zip(axes[0], types):
        d = d0[d0.dtype == t]
        ns = sorted(d.n.unique())
        bs = sorted(d.batch.unique())
        Z = np.full((len(ns), len(bs)), np.nan)
        for _, r in d.iterrows():
            Z[ns.index(r.n), bs.index(r.batch)] = math.log2(r.speedup)
        _hatch_background(ax)
        im = ax.pcolormesh(np.arange(len(bs) + 1), np.arange(len(ns) + 1), np.ma.masked_invalid(Z),
                           cmap=style.SPEEDUP_CMAP, norm=norm, edgecolors="white", linewidth=0.6)
        # Values in the cells when there is room for them.
        if Z.size <= 72 and len(bs) <= 9:
            fs = 5.0 if len(bs) > 6 else 5.6
            for i in range(len(ns)):
                for j in range(len(bs)):
                    if np.isnan(Z[i, j]):
                        continue
                    v = 2 ** Z[i, j]
                    txt = _cell_text(v)
                    ax.text(j + 0.5, i + 0.5, txt, ha="center", va="center", fontsize=fs,
                            color="white" if abs(Z[i, j]) > 1.6 else style.INK)
        ax.set_xticks(np.arange(len(bs)) + 0.5)
        ax.set_xticklabels([_batch_label(b) for b in bs], fontsize=5.8 if len(bs) > 7 else 6.5)
        ax.set_yticks(np.arange(len(ns)) + 0.5)
        ax.set_yticklabels([str(n) for n in ns], fontsize=6.5)
        ax.tick_params(length=0, pad=2)
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(style.tex(TYPE_LABEL[t]) + rf" ({TYPE_PREFIX[t]})", fontsize=7.5, loc="left", pad=3)
        ax.set_xlabel("batch size")
    axes[0][0].set_ylabel(r"matrix order $n$" if plt.rcParams["text.usetex"] else "matrix order n")
    ticks = np.arange(-SPEEDUP_LIM, SPEEDUP_LIM + 1)
    cb = fig.colorbar(im, ax=axes[0].tolist(), location="right", shrink=0.92, aspect=22, pad=0.015, ticks=ticks)
    cb.ax.set_yticklabels([style.times(2.0 ** k) for k in ticks], fontsize=6)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=2, width=0.4)
    cb.set_label(style.tex(f"speedup over {vendor}"), fontsize=7)
    if titles:
        _title(fig, op, fig_level=True)
    _caption(fig, style.tex(" / ".join(x for x in (meta.get("device", ""),
                                                   "blue: BatchLAS faster, red: vendor faster, hatched: not measured") if x)))
    return fig


def fig_summary(w: pd.DataFrame, dtype: str, meta: dict, titles=True):
    s = saturated(w[w.dtype == dtype])
    if s.empty:
        return None
    ops = [o for o in OPS if (s.op == o).any()]
    ns = sorted(s.n.unique())
    Z = np.full((len(ops), len(ns)), np.nan)
    for _, r in s.iterrows():
        Z[ops.index(r.op), ns.index(r.n)] = math.log2(r.speedup)
    fig, ax = plt.subplots(figsize=(style.SINGLE_COL * (1.0 if len(ns) <= 8 else 1.3), 0.35 + 0.2 * len(ops)))
    _hatch_background(ax)
    im = ax.pcolormesh(np.arange(len(ns) + 1), np.arange(len(ops) + 1), np.ma.masked_invalid(Z),
                       cmap=style.SPEEDUP_CMAP, norm=Normalize(-SPEEDUP_LIM, SPEEDUP_LIM),
                       edgecolors="white", linewidth=0.6)
    for i in range(len(ops)):
        for j in range(len(ns)):
            if not np.isnan(Z[i, j]):
                v = 2 ** Z[i, j]
                ax.text(j + 0.5, i + 0.5, _cell_text(v),
                        ha="center", va="center", fontsize=5.4,
                        color="white" if abs(Z[i, j]) > 1.6 else style.INK)
    ax.set_xticks(np.arange(len(ns)) + 0.5)
    ax.set_xticklabels([str(n) for n in ns], fontsize=6.5)
    ax.set_yticks(np.arange(len(ops)) + 0.5)
    ax.set_yticklabels([style.mono(o) for o in ops], fontsize=7)
    ax.invert_yaxis()
    ax.tick_params(length=0, pad=2)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xlabel(r"matrix order $n$" if plt.rcParams["text.usetex"] else "matrix order n")
    cb = fig.colorbar(im, ax=ax, location="right", shrink=0.95, aspect=16, pad=0.02,
                      ticks=np.arange(-SPEEDUP_LIM, SPEEDUP_LIM + 1))
    cb.ax.set_yticklabels([style.times(2.0 ** k) for k in cb.get_ticks()], fontsize=5.5)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=2, width=0.4)
    if titles:
        fig.suptitle(style.tex(f"Speedup over {meta.get('vendor_label', 'vendor')}, {TYPE_LABEL[dtype]} "
                               f"precision, batch at saturation"), x=0.0, ha="left", fontsize=8)
    return fig


# ----------------------------------------------------------------- export
def save(fig, stem: Path, formats=("pdf", "png")) -> List[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    out = []
    for f in formats:
        p = stem.with_suffix(f".{f}")
        tmp = p.with_name(p.stem + ".tmp." + f)
        fig.savefig(tmp)
        tmp.replace(p)  # atomic, so the dashboard never serves half a PNG
        out.append(p)
    plt.close(fig)
    return out


def render_op(camp, op: str, w: Optional[pd.DataFrame] = None, titles=True) -> List[Path]:
    meta = camp.config.get("provenance", {})
    if w is None:
        w = paired(camp.rows())
    if w.empty or not (w.op == op).any():
        return []
    out = []
    for name, fn in (("speedup_n", fig_speedup_n), ("throughput_n", fig_throughput_n), ("heatmap", fig_heatmap)):
        fig = fn(w, op, meta, titles=titles)
        if fig is not None:
            out += save(fig, camp.figures / op / name)
    return out


def render_all(camp, titles=True, ops: Optional[Sequence[str]] = None) -> List[Path]:
    w = paired(camp.rows())
    if w.empty:
        return []
    out = []
    for op in ops or [o for o in OPS if (w.op == o).any()]:
        out += render_op(camp, op, w, titles=titles)
    meta = camp.config.get("provenance", {})
    for t in TYPES:
        fig = fig_summary(w, t, meta, titles=titles)
        if fig is not None:
            out += save(fig, camp.figures / "_summary" / f"summary_{t}")
    return out
