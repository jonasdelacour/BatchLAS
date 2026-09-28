"""Figures for one campaign, derived from results.jsonl alone, in the house
style (style.py). Per op: speedup_n, throughput_n, heatmap, and for the
rectangular ops speedup_2d and throughput_2d. Per precision: summary_<t>.
What "saturated batch" means and why: README.md, "Figures"."""
from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

import style
from ops import OPS, TYPE_PREFIX, TYPES, vendor_name

style.apply()
# A campaign's first render can hold a single point per panel; log axes warn on it.
warnings.filterwarnings("ignore", message="Data has no positive values")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm, Normalize  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter, NullLocator  # noqa: E402

KEYS = ["op", "dtype", "m", "n", "nrhs", "batch"]
PREC_TITLE = {"float": "Single", "double": "Double", "cfloat": "Complex Single", "cdouble": "Complex Double"}
FONT_CELL = 22


# ----------------------------------------------------------------- data
def paired(rows: List[dict]) -> pd.DataFrame:
    """One row per cell with both arms side by side; only verified pairs. A
    single-arm op's cells are kept alone, with no vendor time and no speedup."""
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "ok" not in df or "time_ms" not in df:
        return pd.DataFrame()
    df = df[df["ok"] == True]  # noqa: E712
    if df.empty:
        return pd.DataFrame()
    df = df.sort_values("t").drop_duplicates(KEYS + ["arm"], keep="last")
    for c in ("rel_sd", "route", "vendor_free"):
        if c not in df:
            df[c] = None
    w = df.pivot_table(index=KEYS, columns="arm", values=["time_ms", "rel_sd", "route", "vendor_free"],
                       aggfunc="first")
    w.columns = [f"{a}_{b}" for a, b in w.columns]
    w = w.reset_index()
    if "time_ms_batchlas" not in w:
        return pd.DataFrame()
    for c in ("time_ms_vendor", "rel_sd_vendor", "route_vendor", "vendor_free_vendor"):
        if c not in w:
            w[c] = np.nan
    single = w.op.map(lambda o: o in OPS and OPS[o].single)
    w = w[w.time_ms_batchlas.notna() & (w.time_ms_vendor.notna() | single)].copy()
    if w.empty:
        return pd.DataFrame()
    tb, tv = w["time_ms_batchlas"].astype(float), w["time_ms_vendor"].astype(float)
    w["speedup"] = tv / tb
    rb = pd.to_numeric(w.get("rel_sd_batchlas"), errors="coerce").fillna(0.0)
    rv = pd.to_numeric(w.get("rel_sd_vendor"), errors="coerce").fillna(0.0)
    w["speedup_sd"] = w["speedup"] * np.sqrt(rb ** 2 + rv ** 2)
    # The n-figures plot square cells only; the rectangular ones go to the 2-D maps.
    w["square"] = [OPS[o].is_square(int(m), int(n), int(k)) if o in OPS else True
                   for o, m, n, k in zip(w.op, w.m, w.n, w.nrhs)]
    return w


def throughput(op: str, dtype: str, m, n, nrhs, batch, time_ms):
    per_s = np.asarray(batch, float) / (np.asarray(time_ms, float) * 1e-3)
    spec = OPS[op]
    if spec.flops is None:
        return per_s
    f = spec.flops(np.asarray(m, float), np.asarray(n, float), np.asarray(nrhs, float), dtype)
    return per_s * np.asarray(f) * 1e-9


def throughput_label(op: str) -> str:
    return "Throughput [Matrices / s]" if OPS[op].flops is None else "Throughput [GFLOP / s]"


def square(w: pd.DataFrame) -> pd.DataFrame:
    return w[w.square.astype(bool)] if "square" in w else w


def saturated(w: pd.DataFrame) -> pd.DataFrame:
    w = square(w)
    if w.empty:
        return w
    return w.loc[w.groupby(["op", "dtype", "n"])["batch"].idxmax()].sort_values(["op", "dtype", "n"])


def plane_points(w: pd.DataFrame, op: str) -> pd.DataFrame:
    """The op's cells placed on its Plane as (x, y), one per (dtype, x, y): the
    largest batch measured there."""
    plane = OPS[op].plane
    d = w[w.op == op] if not w.empty else w
    if plane is None or d.empty:
        return d.iloc[0:0]
    xy = [plane.coords(int(m), int(n), int(k)) for m, n, k in zip(d.m, d.n, d.nrhs)]
    d = d.assign(x=[p[0] for p in xy], y=[p[1] for p in xy])
    return d.loc[d.groupby(["dtype", "x", "y"])["batch"].idxmax()]


def vendor_label(meta: dict, op: str = None) -> str:
    """The library the vendor arm of `op` reaches; with no op (the cross-op
    summary) the generic word, since that figure mixes cuSOLVER and cuBLAS. In a
    build comparison the reference slot holds the baseline build instead."""
    if meta.get("ref_label"):
        return meta["ref_label"]
    if op is None:
        return "Vendor"
    return vendor_name(op, "rocm" if "roc" in meta.get("vendor_label", "").lower() else "cuda")


def batchlas_label(meta: dict) -> str:
    return meta.get("new_label") or "BatchLAS"


def _label(lab: str) -> str:
    return lab if plt.rcParams["text.usetex"] else lab.replace("$", "")


def _n_label(op: str = None) -> str:
    return _label(OPS[op].n_label if op in OPS else "Matrix Order $n$ [1]")


# ----------------------------------------------------------------- axes
def _n_axis(ax, ns: Sequence[int], label=True, op=None):
    ns = sorted(set(int(v) for v in ns))
    ax.set_xscale("log", base=2)
    lo, hi = ns[0], ns[-1]
    pows = [2 ** k for k in range(int(math.log2(lo)), int(math.ceil(math.log2(hi))) + 1)
            if lo <= 2 ** k <= hi] or ns
    ax.xaxis.set_major_locator(FixedLocator(pows))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(round(v))}"))
    ax.xaxis.set_minor_locator(FixedLocator([v for v in ns if v not in pows]))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(lo / 2 ** 0.15, hi * 2 ** 0.15)
    if label:
        ax.set_xlabel(_n_label(op))


def _speedup_axis(ax, lo: float, hi: float):
    lo, hi = min(lo, 1.0), max(hi, 1.0)
    if hi / lo > 8:  # linear, as in the house style, unless the range needs log
        ax.set_yscale("log", base=2)
        ax.yaxis.set_major_locator(FixedLocator(style.log2_ticks(lo, hi)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: style.times(v)))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.set_ylim(lo / 1.25, hi * 1.25)
    else:
        ax.set_ylim(0, hi * 1.12)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: style.times(v) if v > 0 else "0"))


# ----------------------------------------------------------------- figures
def fig_speedup_n(w: pd.DataFrame, op: str, meta: dict):
    s = saturated(w[w.op == op]).dropna(subset=["speedup"])
    if s.empty:
        return None
    fig, ax = plt.subplots(figsize=style.PANEL)
    handles = [style.band_handle()]
    hollow = False
    for t in style.PREC_ORDER:
        d = s[s.dtype == t]
        if d.empty:
            continue
        c, mk, sc = style.PREC_COLOR[t], style.PREC_MARKER[t], style.PREC_MSCALE[t]
        sd = d.speedup_sd.to_numpy()
        style.series(ax, d.n, d.speedup, c, mk, sc, band=(d.speedup - 2 * sd, d.speedup + 2 * sd))
        vf = d.get("vendor_free_batchlas", pd.Series(True, index=d.index)).fillna(True).astype(bool)
        if OPS[op].setup_ops:
            vf[:] = True
        if (~vf).any():
            hollow = True
            ax.plot(d.n[~vf], d.speedup[~vf], ls="none", marker=mk, ms=10 * sc, mfc="white", mec=c,
                    mew=2, zorder=4)
        handles.append(Line2D([], [], ls=":", color=c, marker=mk, ms=10 * sc,
                              label=f"{PREC_TITLE[t]} [{TYPE_PREFIX[t]}]"))
    ax.axhline(1.0, color=style.REF, ls="--", lw=1.5, zorder=2)
    handles.append(Line2D([], [], color=style.REF, ls="--", label=style.tex(f"Parity with {vendor_label(meta, op)}")))
    if hollow:
        handles.append(Line2D([], [], ls="none", marker="o", ms=10, mfc="white", mec="black", mew=2,
                              label="Calls a vendor sub-op"))
    _n_axis(ax, s.n, op=op)
    lo = float((s.speedup - 2 * s.speedup_sd).clip(lower=1e-3).min())
    _speedup_axis(ax, lo, float((s.speedup + 2 * s.speedup_sd).max()))
    ax.set_ylabel(style.tex(f"Speedup vs. {vendor_label(meta, op)}"))
    ax.legend(handles=handles, loc="best")
    style.outline(ax)
    return fig


def fig_throughput_n(w: pd.DataFrame, op: str, meta: dict):
    s = saturated(w[w.op == op])
    if s.empty:
        return None
    types = [t for t in style.PREC_ORDER if (s.dtype == t).any()]
    ncol = 2 if len(types) > 1 else 1
    nrow = int(math.ceil(len(types) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(10 * ncol, 7.5 * nrow), squeeze=False)
    for ax in axes.flat[len(types):]:
        ax.set_visible(False)
    arms = [a for a in ("vendor", "batchlas") if s[f"time_ms_{a}"].notna().any()]
    hollow = False
    for i, (ax, t) in enumerate(zip(axes.flat, types)):
        d = s[s.dtype == t]
        for arm in arms:
            tm = d[f"time_ms_{arm}"].astype(float).to_numpy()
            rs = pd.to_numeric(d.get(f"rel_sd_{arm}"), errors="coerce").fillna(0).to_numpy()
            y = throughput(op, t, d.m, d.n, d.nrhs, d.batch, tm)
            band = (y / (1 + 2 * rs), y / np.maximum(1 - 2 * rs, 0.05))
            c, mk, sc = style.LIB_COLOR[arm], style.LIB_MARKER[arm], style.LIB_MSCALE[arm]
            style.series(ax, d.n, y, c, mk, sc, band=band)
            if arms == ["batchlas"]:
                # With no speedup figure, this is where a vendor sub-op call shows.
                vf = d.vendor_free_batchlas.fillna(True).astype(bool).to_numpy()
                if (~vf).any():
                    hollow = True
                    ax.plot(d.n[~vf], y[~vf], ls="none", marker=mk, ms=10 * sc, mfc="white", mec=c, mew=2, zorder=4)
        ax.set_yscale("log")
        ax.set_title(style.bold(f"{PREC_TITLE[t]} [{TYPE_PREFIX[t]}]"), pad=12)
        _n_axis(ax, d.n, label=i + ncol >= len(types), op=op)
        if i % ncol == 0:
            ax.set_ylabel(throughput_label(op))
        style.outline(ax)
    handles = [Line2D([], [], ls=":", color=style.LIB_COLOR[a], marker=style.LIB_MARKER[a],
                      ms=10 * style.LIB_MSCALE[a], label=l)
               for a, l in (("batchlas", style.tex(batchlas_label(meta))), ("vendor", style.tex(vendor_label(meta, op))))
               if a in arms]
    if hollow:
        handles.append(Line2D([], [], ls="none", marker="o", ms=10, mfc="white", mec="black", mew=2,
                              label="Calls a vendor sub-op"))
    axes.flat[0].legend(handles=handles + [style.band_handle()], loc="upper left" if len(arms) > 1 else "best")
    return fig


def _batch_label(b: int) -> str:
    k = int(round(math.log2(b)))
    return rf"$2^{{{k}}}$" if 2 ** k == b else f"{b}"


def _parity_edges(ax, Z: np.ndarray):
    """Red boundary along every cell edge where the speedup crosses 1x."""
    rows, cols = Z.shape
    for i in range(rows):
        for j in range(cols):
            if np.isnan(Z[i, j]):
                continue
            if j + 1 < cols and not np.isnan(Z[i, j + 1]) and (Z[i, j] > 0) != (Z[i, j + 1] > 0):
                ax.plot([j + 1, j + 1], [i, i + 1], color=style.REF, lw=3, zorder=4)
            if i + 1 < rows and not np.isnan(Z[i + 1, j]) and (Z[i, j] > 0) != (Z[i + 1, j] > 0):
                ax.plot([j, j + 1], [i + 1, i + 1], color=style.REF, lw=3, zorder=4)


def _map_norm(values: np.ndarray) -> Normalize:
    v = values[~np.isnan(values)]
    lo = max(-4.0, math.floor(min(v.min(), -1.0))) if v.size else -1.0
    hi = min(4.0, math.ceil(max(v.max(), 1.0))) if v.size else 1.0
    return Normalize(vmin=lo, vmax=hi)


def _speedup_colorbar(fig, im, axes, meta, norm, op=None):
    ticks = np.arange(norm.vmin, norm.vmax + 1)
    cb = fig.colorbar(im, ax=axes, location="right", ticks=ticks, pad=0.02, fraction=0.05, aspect=25)
    cb.ax.set_yticklabels([style.times(2.0 ** k) for k in ticks])
    cb.set_label(style.tex(f"Speedup vs. {vendor_label(meta, op)}") + r" [$\times$]")
    cb.ax.axhline(0.0, color=style.REF, lw=3)
    return cb


def _map_axes(ax, xlabels, ylabels):
    ax.set_xticks(np.arange(len(xlabels)) + 0.5)
    ax.set_xticklabels(xlabels)
    ax.set_yticks(np.arange(len(ylabels)) + 0.5)
    ax.set_yticklabels(ylabels)
    ax.grid(False)
    style.outline(ax)


def _fmt_speedup(v: float) -> str:
    return f"{v:.0f}" if v >= 9.95 else (f"{v:.1f}" if v >= 0.995 else f"{v:.2f}".lstrip("0"))


def _fmt_rate(v: float) -> str:
    if v >= 9995:
        return f"{v / 1e3:.0f}k"
    if v >= 999.5:
        return f"{v / 1e3:.1f}k"
    return f"{v:.0f}" if v >= 9.95 else f"{v:.2g}"


def _annotate(ax, Z: np.ndarray, norm, text):
    """The value in each cell, black or white by the cell colour's luminance."""
    cmap = plt.get_cmap(style.CMAP)
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            if np.isnan(Z[i, j]):
                continue
            lum = np.dot(cmap(norm(Z[i, j]))[:3], [0.299, 0.587, 0.114])
            ax.text(j + 0.5, i + 0.5, text(Z[i, j]), ha="center", va="center", fontsize=FONT_CELL,
                    color="black" if lum > 0.5 else "white")


def fig_heatmap(w: pd.DataFrame, op: str, meta: dict):
    """Speedup over n x batch; for an op with no reference arm, BatchLAS's throughput."""
    d0 = square(w[w.op == op])
    if d0.empty:
        return None
    rate = d0.speedup.isna().all()
    types = [t for t in style.PREC_ORDER if (d0.dtype == t).any()]
    ns, bs = sorted(d0.n.unique()), sorted(d0.batch.unique())
    grids = {}
    for t in types:
        Z = np.full((len(ns), len(bs)), np.nan)
        dt = d0[d0.dtype == t]
        v = throughput(op, t, dt.m, dt.n, dt.nrhs, dt.batch, dt.time_ms_batchlas.astype(float)) if rate \
            else np.log2(dt.speedup.astype(float))
        for r, z in zip(dt.itertuples(), v):
            Z[ns.index(r.n), bs.index(r.batch)] = z
        grids[t] = Z
    allz = np.concatenate([g.ravel() for g in grids.values()])
    if rate:
        pos = allz[np.isfinite(allz) & (allz > 0)]
        norm = LogNorm(vmin=pos.min(), vmax=max(pos.max(), pos.min() * 1.01)) if pos.size else LogNorm(1, 10)
    else:
        norm = _map_norm(allz)
    fig, axes = plt.subplots(1, len(types), figsize=(style.MAP_PANEL[0] * len(types) + 2, style.MAP_PANEL[1]),
                             squeeze=False)
    im = None
    for i, (ax, t) in enumerate(zip(axes[0], types)):
        im = ax.pcolormesh(np.arange(len(bs) + 1), np.arange(len(ns) + 1), np.ma.masked_invalid(grids[t]),
                           cmap=style.CMAP, norm=norm, edgecolors=(1, 1, 1, 0.25), linewidth=0.5)
        if rate:
            _annotate(ax, grids[t], norm, _fmt_rate)
        else:
            _parity_edges(ax, grids[t])
        _map_axes(ax, [_batch_label(b) for b in bs], [str(n) if i == 0 else "" for n in ns])
        ax.set_title(style.bold(f"{PREC_TITLE[t]} [{TYPE_PREFIX[t]}]"), pad=12)
        ax.set_xlabel("Batch Size [1]")
        if i == 0:
            ax.set_ylabel(_n_label(op))
    if rate:
        cb = fig.colorbar(im, ax=axes[0].tolist(), location="right", pad=0.02, fraction=0.05, aspect=25)
        cb.ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))  # a narrow range still gets ticks
        cb.ax.yaxis.set_minor_locator(NullLocator())
        cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
        cb.set_label(throughput_label(op))
    else:
        _speedup_colorbar(fig, im, axes[0].tolist(), meta, norm, op)
    return fig


def fig_summary(w: pd.DataFrame, dtype: str, meta: dict):
    s = saturated(w[w.dtype == dtype]).dropna(subset=["speedup"])
    if s.empty:
        return None
    ops = [o for o in OPS if (s.op == o).any()]
    ns = sorted(s.n.unique())
    Z = np.full((len(ops), len(ns)), np.nan)
    for r in s.itertuples():
        Z[ops.index(r.op), ns.index(r.n)] = math.log2(r.speedup)
    norm = _map_norm(Z)
    fig, ax = plt.subplots(figsize=(max(14, 1.6 * len(ns) + 5), 0.8 * len(ops) + 2.5))
    im = ax.pcolormesh(np.arange(len(ns) + 1), np.arange(len(ops) + 1), np.ma.masked_invalid(Z),
                       cmap=style.CMAP, norm=norm, edgecolors=(1, 1, 1, 0.25), linewidth=0.5)
    _parity_edges(ax, Z)
    _annotate(ax, Z, norm, lambda z: _fmt_speedup(2 ** z))
    _map_axes(ax, [str(n) for n in ns], [style.mono(o) for o in ops])
    ax.invert_yaxis()
    ax.set_xlabel(_n_label())
    ax.set_title(style.bold(f"{PREC_TITLE[dtype]} Precision [{TYPE_PREFIX[dtype]}]"), pad=12)
    _speedup_colorbar(fig, im, ax, meta, norm)
    return fig


def _plane_grid(d: pd.DataFrame, xs, ys, value) -> np.ndarray:
    Z = np.full((len(ys), len(xs)), np.nan)
    for r, v in zip(d.itertuples(), value(d)):
        Z[ys.index(r.y), xs.index(r.x)] = v
    return Z


def _plane_axes(ax, plane, xs, ys, ylabels=True):
    """Tick labels, axis labels, and a faint hatch over shapes the op does not define
    (n > m for QR), so they read differently from shapes not yet measured."""
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            if not plane.valid(x, y):
                ax.add_patch(Rectangle((j, i), 1, 1, fill=False, hatch="//", lw=0,
                                       edgecolor=(0, 0, 0, 0.18), zorder=0))
    _map_axes(ax, [str(x) for x in xs], [str(y) if ylabels else "" for y in ys])
    ax.set_xlabel(_label(plane.x_label))
    if ylabels:
        ax.set_ylabel(_label(plane.y_label))


def _plane_data(w: pd.DataFrame, op: str):
    """(points, types, xs, ys), or None when the op has no Plane or only its square
    cells were measured: a map of the diagonal alone says nothing the n-figures do not."""
    d = plane_points(w, op)
    if d.empty or d.square.astype(bool).all():
        return None
    return d, [t for t in style.PREC_ORDER if (d.dtype == t).any()], sorted(d.x.unique()), sorted(d.y.unique())


def fig_speedup_2d(w: pd.DataFrame, op: str, meta: dict):
    got = _plane_data(w, op)
    if got is None:
        return None
    d, types, xs, ys = got
    plane = OPS[op].plane
    grids = {t: _plane_grid(d[d.dtype == t], xs, ys, lambda e: np.log2(e.speedup.astype(float))) for t in types}
    norm = _map_norm(np.concatenate([g.ravel() for g in grids.values()]))
    fig, axes = plt.subplots(1, len(types), figsize=(style.MAP_PANEL[0] * len(types) + 2, style.MAP_PANEL[1]),
                             squeeze=False)
    im = None
    for i, (ax, t) in enumerate(zip(axes[0], types)):
        Z = grids[t]
        im = ax.pcolormesh(np.arange(len(xs) + 1), np.arange(len(ys) + 1), np.ma.masked_invalid(Z),
                           cmap=style.CMAP, norm=norm, edgecolors=(1, 1, 1, 0.25), linewidth=0.5)
        _parity_edges(ax, Z)
        _annotate(ax, Z, norm, lambda z: _fmt_speedup(2 ** z))
        _plane_axes(ax, plane, xs, ys, ylabels=i == 0)
        ax.set_title(style.bold(f"{PREC_TITLE[t]} [{TYPE_PREFIX[t]}]"), pad=12)
    _speedup_colorbar(fig, im, axes[0].tolist(), meta, norm, op)
    return fig


def fig_throughput_2d(w: pd.DataFrame, op: str, meta: dict):
    """One row per precision, BatchLAS and vendor side by side on a shared log scale."""
    got = _plane_data(w, op)
    if got is None:
        return None
    d, types, xs, ys = got
    plane = OPS[op].plane
    libs = (("batchlas", batchlas_label(meta)), ("vendor", vendor_label(meta, op)))
    fig, axes = plt.subplots(len(types), 2, figsize=(style.MAP_PANEL[0] * 2 + 3, style.MAP_PANEL[1] * len(types)),
                             squeeze=False)
    for row, t in enumerate(types):
        dt = d[d.dtype == t]
        grids = {a: _plane_grid(dt, xs, ys, lambda e, a=a: throughput(
            op, t, e.m, e.n, e.nrhs, e.batch, e[f"time_ms_{a}"].astype(float))) for a, _ in libs}
        v = np.concatenate([g.ravel() for g in grids.values()])
        v = v[np.isfinite(v) & (v > 0)]
        norm = LogNorm(vmin=v.min(), vmax=max(v.max(), v.min() * 1.01)) if v.size else LogNorm(1, 10)
        im = None
        for col, (a, lab) in enumerate(libs):
            ax, Z = axes[row, col], grids[a]
            im = ax.pcolormesh(np.arange(len(xs) + 1), np.arange(len(ys) + 1), np.ma.masked_invalid(Z),
                               cmap=style.CMAP, norm=norm, edgecolors=(1, 1, 1, 0.25), linewidth=0.5)
            _annotate(ax, Z, norm, _fmt_rate)
            _plane_axes(ax, plane, xs, ys, ylabels=col == 0)
            if row + 1 < len(types):
                ax.set_xlabel("")
            ax.set_title(style.bold(f"{lab}, {PREC_TITLE[t]} [{TYPE_PREFIX[t]}]"), pad=12)
        cb = fig.colorbar(im, ax=axes[row].tolist(), location="right", pad=0.02, fraction=0.05, aspect=20)
        cb.ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))  # a narrow range still gets ticks
        cb.ax.yaxis.set_minor_locator(NullLocator())
        cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
        cb.set_label(throughput_label(op))
    return fig


# ----------------------------------------------------------------- export
FIGURES = (("speedup_n", fig_speedup_n), ("throughput_n", fig_throughput_n), ("heatmap", fig_heatmap),
           ("speedup_2d", fig_speedup_2d), ("throughput_2d", fig_throughput_2d))


def save(fig, stem: Path, formats=("pdf", "png")) -> List[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    out = []
    for f in formats:
        p = stem.with_suffix(f".{f}")
        tmp = p.with_name(p.stem + ".tmp." + f)
        fig.savefig(tmp)
        tmp.replace(p)  # atomic: the dashboard never serves half a PNG
        out.append(p)
    plt.close(fig)
    return out


def render_op(camp, op: str, w: Optional[pd.DataFrame] = None) -> List[Path]:
    meta = camp.config.get("provenance", {})
    if w is None:
        w = paired(camp.rows())
    if w.empty or not (w.op == op).any():
        return []
    out = []
    for name, fn in FIGURES:
        fig = fn(w, op, meta)
        if fig is not None:
            out += save(fig, camp.figures / op / name)
    return out


def render_all(camp, ops: Optional[Sequence[str]] = None) -> List[Path]:
    w = paired(camp.rows())
    if w.empty:
        return []
    out = []
    for op in ops or [o for o in OPS if (w.op == o).any()]:
        out += render_op(camp, op, w)
    meta = camp.config.get("provenance", {})
    for t in TYPES:
        fig = fig_summary(w, t, meta)
        if fig is not None:
            out += save(fig, camp.figures / "_summary" / f"summary_{t}")
    return out
