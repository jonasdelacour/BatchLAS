#!/usr/bin/env python3
"""plot_bandr1_evolution

Plot a dense reconstruction of the BANDR1 working band matrix (ABw) as it
changes over steps/sweeps.

This script consumes the CSV dumps produced by BatchLAS' BANDR1 debug dumping.
In particular it looks for files named:
  ABw_band_before_i1<i1>_i2<i2>_j1<j1>_j2<j2>_b<batch>.csv
  ABw_band_after_i1<i1>_i2<i2>_j1<j1>_j2<j2>_b<batch>.csv

Those dumps contain the band-storage representation (kd_work+1, n) where
ABw[r, j] stores A[j+r, j]. We reconstruct a dense Hermitian matrix and plot a
heatmap.

Recommended dump organization
----------------------------
To avoid filename collisions and to preserve a clear ordering, dump each step
into its own subdirectory, e.g.:
  output/bandr1_dumps/onesweep_steps/step_0000/
  output/bandr1_dumps/onesweep_steps/step_0001/
  ...

Then point this script at the root:
  python3 playground/plot_bandr1_evolution.py \
      --dump-root output/bandr1_dumps/onesweep_steps \
      --out-dir   output/bandr1_plots/onesweep \
      --which     after \
      --mode      abs

You can also create a single figure containing many steps (a mosaic):
    python3 playground/plot_bandr1_evolution.py \
            --dump-root output/bandr1_dumps/onesweep_steps \
            --mosaic-out output/bandr1_plots/onesweep_mosaic.png \
            --which after --mode abs --cols 6

Important: a mosaic is intended to show the *same* matrix evolving. If your
dump root contains multiple runs (different n/kd_work), use a more specific
--dump-root or pass --n / --kd-work to select the run you want.

"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "matplotlib is required for plotting. "
        "Install it (e.g. `python3 -m pip install matplotlib`) and retry.\n"
        f"Original import error: {exc}"
    )


_STEP_RE = re.compile(
    r"ABw_band_(?P<which>before|after)"
    r"(?:_sw(?P<sw>-?\d+))?"
    r"(?:_st(?P<st>-?\d+))?"
    r"(?:_g(?P<g>-?\d+))?"
    r"_i1(?P<i1>-?\d+)_i2(?P<i2>-?\d+)_j1(?P<j1>-?\d+)_j2(?P<j2>-?\d+)_b(?P<b>-?\d+)\.csv$"
)

_DIR_STEP_RE = re.compile(r"(?:^|/)step_(?P<step>\d+)(?:/|$)")
_DIR_SWEEP_RE = re.compile(r"(?:^|/)sweep_(?P<sweep>\d+)(?:/|$)")


def load_csv_matrix(path: Path) -> np.ndarray:
    rows = None
    cols = None
    data: list[tuple[int, int, float, float]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                if line.startswith("# rows,"):
                    parts = line.split(",")
                    rows = int(parts[1])
                    cols = int(parts[3])
                continue
            i_s, j_s, re_s, im_s = line.split(",")
            data.append((int(i_s), int(j_s), float(re_s), float(im_s)))

    if rows is None or cols is None:
        raise ValueError(f"missing header rows/cols in {path}")

    any_im = any(abs(im) != 0.0 for (_, _, _, im) in data)
    dtype = np.complex128 if any_im else np.float64
    out = np.zeros((rows, cols), dtype=dtype)
    for i, j, re_v, im_v in data:
        out[i, j] = re_v + (1j * im_v if any_im else 0.0)
    return out


def band_storage_to_dense_hermitian(ab: np.ndarray) -> np.ndarray:
    """Reconstruct dense Hermitian matrix from lower-band storage (kd+1, n)."""
    if ab.ndim != 2:
        raise ValueError(f"expected 2D band storage, got shape={ab.shape}")
    kd = ab.shape[0] - 1
    n = ab.shape[1]

    dense = np.zeros((n, n), dtype=ab.dtype)
    for j in range(n):
        max_r = min(kd, (n - 1) - j)
        for r in range(max_r + 1):
            i = j + r
            v = ab[r, j]
            if i == j:
                dense[i, j] = np.real(v)
            else:
                dense[i, j] = v
                dense[j, i] = np.conjugate(v)
    return dense


@dataclass(frozen=True)
class Snapshot:
    path: Path
    which: str  # "before" or "after"
    i1: int
    i2: int
    j1: int
    j2: int
    batch: int
    sweep: Optional[int]
    step: Optional[int]


def infer_step_index(path: Path) -> Optional[int]:
    # Look for .../step_0007/... anywhere in the parents.
    s = "/".join(path.parts)
    m = _DIR_STEP_RE.search(s)
    if not m:
        return None
    return int(m.group("step"))


def infer_sweep_index(path: Path) -> Optional[int]:
    s = "/".join(path.parts)
    m = _DIR_SWEEP_RE.search(s)
    if not m:
        return None
    return int(m.group("sweep"))


def discover_snapshots(dump_root: Path) -> list[Snapshot]:
    snaps: list[Snapshot] = []
    for p in dump_root.rglob("ABw_band_*.csv"):
        m = _STEP_RE.match(p.name)
        if not m:
            continue
        snaps.append(
            Snapshot(
                path=p,
                which=m.group("which"),
                i1=int(m.group("i1")),
                i2=int(m.group("i2")),
                j1=int(m.group("j1")),
                j2=int(m.group("j2")),
                batch=int(m.group("b")),
                sweep=infer_sweep_index(p),
                step=infer_step_index(p),
            )
        )

    # Primary order: inferred step index if available. Otherwise fall back to (i1,j1).
    snaps.sort(
        key=lambda s: (
            10**9 if s.sweep is None else s.sweep,
            10**9 if s.step is None else s.step,
            s.i1,
            s.j1,
            s.batch,
            s.which,
        )
    )
    return snaps


def compute_field(A: np.ndarray, mode: str, *, eps: float) -> np.ndarray:
    if mode == "abs":
        return np.abs(A)
    if mode == "real":
        return np.real(A)
    if mode == "imag":
        return np.imag(A)
    if mode in ("nz", "mask"):
        # Binary mask: 1 if |A_ij| > eps else 0.
        return (np.abs(A) > eps).astype(np.float32)
    raise ValueError(f"unknown mode={mode}")


def plot_heatmap(field: np.ndarray, *, title: str, out_path: Path, cmap: str, vmin: Optional[float], vmax: Optional[float]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.5, 6.5), dpi=140)
    # Mathematical convention: row 0 at top.
    im = ax.imshow(field, origin="upper", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_title(title)
    ax.set_xlabel("j")
    ax.set_ylabel("i")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_mosaic(
    fields: list[np.ndarray],
    titles: list[str],
    *,
    out_path: Path,
    cmap: str,
    cols: int,
    vmin: Optional[float],
    vmax: Optional[float],
) -> None:
    if len(fields) != len(titles):
        raise ValueError("fields/titles length mismatch")
    if not fields:
        raise ValueError("no fields to plot")
    cols = max(1, int(cols))
    rows = int(np.ceil(len(fields) / cols))

    # Size heuristic: keep each panel readable.
    # Allocate an extra narrow column for the colorbar so it never overlaps.
    fig_w = max(8.0, 2.5 * cols + 0.9)
    fig_h = max(6.0, 2.2 * rows)
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=140)
    gs = GridSpec(
        rows,
        cols + 1,
        figure=fig,
        width_ratios=[1.0] * cols + [0.06],
        wspace=0.05,
        hspace=0.18,
    )

    im = None
    for idx, (field, title) in enumerate(zip(fields, titles)):
        r = idx // cols
        c = idx % cols
        ax = fig.add_subplot(gs[r, c])
        # Mathematical convention: row 0 at top.
        im = ax.imshow(
            field,
            origin="upper",
            interpolation="nearest",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect="equal",
        )
        ax.set_title(title, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide unused axes.
    for idx in range(len(fields), rows * cols):
        r = idx // cols
        c = idx % cols
        ax = fig.add_subplot(gs[r, c])
        ax.axis("off")

    if im is not None:
        cax = fig.add_subplot(gs[:, -1])
        fig.colorbar(im, cax=cax)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.03)
    fig.savefig(out_path)
    plt.close(fig)


def plot_sweep_step_grid(
    frames: list[tuple[Snapshot, np.ndarray]],
    *,
    out_path: Path,
    cmap: str,
    vmin: float,
    vmax: float,
    super_title: Optional[str],
) -> None:
    grid_frames = [
        (snap, field)
        for (snap, field) in frames
        if snap.sweep is not None and snap.step is not None
    ]
    if not grid_frames:
        raise ValueError(
            "No frames have sweep/step indices. "
            "Expected dumps organized under sweep_XXX/step_YYYY directories."
        )

    sweeps = sorted({snap.sweep for (snap, _) in grid_frames if snap.sweep is not None})
    steps = sorted({snap.step for (snap, _) in grid_frames if snap.step is not None})
    sweep_to_row = {s: i for i, s in enumerate(sweeps)}
    step_to_col = {s: j for j, s in enumerate(steps)}

    nrows = len(sweeps)
    ncols = len(steps)

    # Shared colorbar column.
    fig_w = max(8.0, 2.5 * ncols + 0.9)
    fig_h = max(6.0, 2.2 * nrows)
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=140)
    gs = GridSpec(
        nrows,
        ncols + 1,
        figure=fig,
        width_ratios=[1.0] * ncols + [0.06],
        wspace=0.05,
        hspace=0.18,
    )

    if super_title:
        fig.suptitle(super_title)

    im = None
    for snap, field in grid_frames:
        r = sweep_to_row[snap.sweep]
        c = step_to_col[snap.step]
        ax = fig.add_subplot(gs[r, c])
        im = ax.imshow(
            field,
            origin="upper",
            interpolation="nearest",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect="equal",
        )
        if r == 0:
            ax.set_title(f"step {snap.step}", fontsize=8)
        if c == 0:
            ax.set_ylabel(f"sweep {snap.sweep}")
        ax.set_xticks([])
        ax.set_yticks([])

    if im is not None:
        cax = fig.add_subplot(gs[:, -1])
        fig.colorbar(im, cax=cax)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.03)
    fig.savefig(out_path)
    plt.close(fig)


def main(argv: Optional[Iterable[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-root", required=True, help="Root directory containing ABw_band_*.csv (may have step_XXXX subdirs)")
    ap.add_argument(
        "--generate",
        action="store_true",
        help="If set, run the C++ bandr1_dump driver to populate --dump-root before plotting.",
    )
    ap.add_argument(
        "--driver",
        default="build/benchmarks/bandr1_dump",
        help="Path to the bandr1_dump executable (workspace-relative or absolute).",
    )
    ap.add_argument("--kd", type=int, default=8, help="(generate) input band semibandwidth kd")
    ap.add_argument("--d", type=int, default=0, help="(generate) diagonals to eliminate per sweep (0 => impl default)")
    ap.add_argument(
        "--gen-max-sweeps",
        type=int,
        default=None,
        help="(generate) max sweeps to run (default: driver default; if unset and --sweep-max is set, uses sweep-max+1)",
    )
    ap.add_argument("--block-size", type=int, default=16, help="(generate) BANDR1 panel size")
    ap.add_argument("--seed", type=int, default=123, help="(generate) RNG seed")
    ap.add_argument("--type", dest="scalar_type", default="f32", help="(generate) f32|f64|c64|c128")
    ap.add_argument("--device", default="gpu", help="(generate) device string for Queue (default gpu)")
    ap.add_argument(
        "--clean-dump-root",
        action="store_true",
        help="(generate) delete existing --dump-root before generating (default behavior when --generate is set)",
    )
    ap.add_argument("--out-dir", default=None, help="Directory to write per-frame PNGs")
    ap.add_argument("--mosaic-out", default=None, help="Write a single mosaic PNG instead of per-frame images")
    ap.add_argument(
        "--layout",
        choices=["linear", "sweep-step"],
        default="linear",
        help="When using --mosaic-out: 'linear' places frames in discovery order; 'sweep-step' uses rows=sweeps, cols=steps.",
    )
    ap.add_argument("--which", choices=["before", "after", "both"], default="after")
    ap.add_argument("--batch", type=int, default=0, help="Batch index to plot")
    ap.add_argument("--mode", choices=["abs", "real", "imag", "nz"], default="abs")
    ap.add_argument(
        "--eps",
        type=float,
        default=0.0,
        help="Threshold for --mode nz (treat |A_ij| <= eps as zero). Default 0.",
    )
    ap.add_argument("--cmap", default=None, help="Matplotlib colormap (defaults based on mode)")
    ap.add_argument("--cols", type=int, default=6, help="Mosaic columns (when using --mosaic-out)")
    ap.add_argument("--n", type=int, default=None, help="Only include dumps with this matrix size n")
    ap.add_argument("--kd-work", type=int, default=None, help="Only include dumps with this kd_work")
    ap.add_argument(
        "--allow-mixed-sizes",
        action="store_true",
        help="Allow mosaics that mix different n/kd_work (usually not what you want)",
    )
    ap.add_argument("--vmax", type=float, default=None, help="Fixed vmax for consistent scaling across frames")
    ap.add_argument("--vmin", type=float, default=None, help="Fixed vmin for consistent scaling across frames")
    ap.add_argument("--limit", type=int, default=None, help="Only plot the first N snapshots (debug convenience)")
    ap.add_argument("--sweep-max", type=int, default=None, help="Only include sweeps <= this value (requires sweep_XXX dirs)")
    ap.add_argument("--step-max", type=int, default=None, help="Only include steps <= this value (requires step_YYYY dirs)")
    ap.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="For --layout sweep-step: downsample to at most this many sweeps (rows)",
    )
    ap.add_argument(
        "--max-cols",
        type=int,
        default=None,
        help="For --layout sweep-step: downsample to at most this many steps (columns)",
    )
    ap.add_argument(
        "--scale",
        choices=["global", "per-frame"],
        default="global",
        help="Color scaling: 'global' uses one vmin/vmax for all frames; 'per-frame' rescales each frame",
    )

    args = ap.parse_args(list(argv) if argv is not None else None)

    dump_root = Path(args.dump_root)

    if args.generate:
        driver = Path(args.driver)
        if not driver.is_absolute():
            driver = (Path.cwd() / driver).resolve()
        if not driver.exists():
            raise SystemExit(f"bandr1_dump driver not found at {driver}. Build it and retry.")

        if dump_root.exists() and any(dump_root.iterdir()):
            # Default behavior: avoid the stale-dumps footgun.
            import shutil

            if not args.clean_dump_root:
                print(f"--generate: cleaning existing non-empty --dump-root {dump_root}")
            shutil.rmtree(dump_root)
        dump_root.mkdir(parents=True, exist_ok=True)

        gen_n = args.n if args.n is not None else 64

        gen_max_sweeps = args.gen_max_sweeps
        if gen_max_sweeps is None and args.sweep_max is not None:
            # sweep indices are 0-based; if user wants to plot up to sweep_max, generate at least that many.
            gen_max_sweeps = int(args.sweep_max) + 1

        cmd = [
            str(driver),
            "--dump-dir",
            str(dump_root),
            "--n",
            str(gen_n),
            "--kd",
            str(args.kd),
            "--kd-work",
            str(args.kd_work or 0),
            "--block-size",
            str(args.block_size),
            "--d",
            str(args.d),
            "--batch",
            "1",
            "--seed",
            str(args.seed),
            "--type",
            str(args.scalar_type),
            "--device",
            str(args.device),
            "--abw-only",
            "--dump-step",
        ]
        if gen_max_sweeps is not None:
            cmd += ["--max-sweeps", str(gen_max_sweeps)]
        print("Generating dumps via:\n  " + " ".join(cmd))
        subprocess.run(cmd, check=True)
    out_dir = Path(args.out_dir) if args.out_dir is not None else None
    mosaic_out = Path(args.mosaic_out) if args.mosaic_out is not None else None
    if out_dir is None and mosaic_out is None:
        raise SystemExit("Provide either --out-dir (per-frame) or --mosaic-out (single figure).")
    if out_dir is not None and mosaic_out is not None:
        raise SystemExit("Choose only one of --out-dir or --mosaic-out.")

    snaps = [s for s in discover_snapshots(dump_root) if s.batch == args.batch]
    if args.which != "both":
        snaps = [s for s in snaps if s.which == args.which]

    if args.sweep_max is not None:
        snaps = [s for s in snaps if s.sweep is not None and s.sweep <= args.sweep_max]
    if args.step_max is not None:
        snaps = [s for s in snaps if s.step is not None and s.step <= args.step_max]

    # Optional downsampling for sweep-step grids.
    if mosaic_out is not None and args.layout == "sweep-step":
        sweeps = sorted({s.sweep for s in snaps if s.sweep is not None})
        steps = sorted({s.step for s in snaps if s.step is not None})

        if args.max_rows is not None and args.max_rows > 0 and len(sweeps) > args.max_rows:
            idxs = np.linspace(0, len(sweeps) - 1, args.max_rows, dtype=int)
            keep_sweeps = {sweeps[i] for i in idxs.tolist()}
            snaps = [s for s in snaps if s.sweep in keep_sweeps]

        if args.max_cols is not None and args.max_cols > 0 and len(steps) > args.max_cols:
            idxs = np.linspace(0, len(steps) - 1, args.max_cols, dtype=int)
            keep_steps = {steps[i] for i in idxs.tolist()}
            snaps = [s for s in snaps if s.step in keep_steps]

    if not snaps:
        raise SystemExit(f"No matching ABw dumps found under {dump_root} for batch={args.batch}.")

    if args.limit is not None:
        snaps = snaps[: args.limit]

    cmap = args.cmap
    if cmap is None:
        if args.mode == "abs":
            cmap = "magma"
        elif args.mode == "nz":
            cmap = "Greys"
        else:
            cmap = "coolwarm"

    # Load all fields up-front so we can compute global scaling if requested, and
    # so we can filter by (n, kd_work).
    fields: list[np.ndarray] = []
    titles: list[str] = []
    kept_snaps: list[Snapshot] = []
    meta: list[tuple[int, str, int, int, int, int, int, int]] = []
    # (idx, step_s, n, kd_work, i1, i2, j1, j2)

    # Track which problem sizes appear under dump_root (useful error message).
    seen_sizes: dict[tuple[int, int], int] = {}

    for idx, snap in enumerate(snaps):
        ab = load_csv_matrix(snap.path)
        A = band_storage_to_dense_hermitian(ab)
        field = compute_field(A, args.mode, eps=float(args.eps))

        n = A.shape[0]
        kd_work = ab.shape[0] - 1
        seen_sizes[(n, kd_work)] = seen_sizes.get((n, kd_work), 0) + 1

        if args.n is not None and n != args.n:
            continue
        if args.kd_work is not None and kd_work != args.kd_work:
            continue

        fields.append(field)
        kept_snaps.append(snap)

        step_s = f"{snap.step:06d}" if snap.step is not None else f"{idx:06d}"
        titles.append(f"{snap.which} step={step_s}  n={n} kd={kd_work}")
        meta.append((idx, step_s, n, kd_work, snap.i1, snap.i2, snap.j1, snap.j2))

    if not fields:
        if args.n is not None or args.kd_work is not None:
            raise SystemExit(
                "No matching dumps after filtering. "
                f"Try removing filters or adjust --n/--kd-work. Seen sizes: {sorted(seen_sizes.keys())}"
            )
        raise SystemExit(f"No matching ABw dumps found under {dump_root} for batch={args.batch}.")

    if args.scale == "global":
        vmin = args.vmin
        vmax = args.vmax
        if args.mode == "nz":
            # Fixed scale for binary masks.
            vmin = 0.0
            vmax = 1.0
        elif vmin is None or vmax is None:
            all_min = float(min(np.min(f) for f in fields))
            all_max = float(max(np.max(f) for f in fields))
            if vmin is None:
                vmin = all_min
            if vmax is None:
                vmax = all_max
    else:
        vmin = args.vmin
        vmax = args.vmax

    # sweep-step layout always uses a shared color scale.
    if mosaic_out is not None and args.layout == "sweep-step":
        if vmin is None or vmax is None:
            all_min = float(min(np.min(f) for f in fields))
            all_max = float(max(np.max(f) for f in fields))
            if vmin is None:
                vmin = all_min
            if vmax is None:
                vmax = all_max

    if mosaic_out is not None:
        if args.layout == "sweep-step" and args.which == "both":
            raise SystemExit("--layout sweep-step requires --which before or --which after (not both).")

        # Mosaics are meant to show a single matrix evolving.
        uniq_sizes = {(n, kd_work) for (_, _, n, kd_work, _, _, _, _) in meta}
        if len(uniq_sizes) > 1 and not args.allow_mixed_sizes:
            msg = (
                "dump-root contains multiple matrix sizes (n, kd_work), so a single evolution mosaic would mix runs:\n"
                + "\n".join([f"  n={n} kd_work={kd}  (files={seen_sizes.get((n, kd), 0)})" for (n, kd) in sorted(uniq_sizes)])
                + "\n\n"
                + "Fix: use a more specific --dump-root (e.g. output/bandr1_dumps/onesweep) "
                + "or pass --n <n> and/or --kd-work <k> to select one run."
            )
            raise SystemExit(msg)

        # Richer titles for mosaic.
        mosaic_titles: list[str] = []
        for (idx, step_s, n, kd_work, i1, i2, j1, j2), snap in zip(meta, kept_snaps):
            if args.layout == "linear":
                mosaic_titles.append(f"{snap.which} step={step_s}  n={n}  (i1={i1},j1={j1})")
            else:
                sw = "?" if snap.sweep is None else str(snap.sweep)
                st = "?" if snap.step is None else str(snap.step)
                mosaic_titles.append(f"{snap.which} sweep={sw} step={st}  n={n}")

        if args.layout == "linear":
            plot_mosaic(fields, mosaic_titles, out_path=mosaic_out, cmap=cmap, cols=args.cols, vmin=vmin, vmax=vmax)
        else:
            frames = list(zip(kept_snaps, fields))
            plot_sweep_step_grid(
                frames,
                out_path=mosaic_out,
                cmap=cmap,
                vmin=float(vmin),
                vmax=float(vmax),
                super_title=os.environ.get("BATCHLAS_PLOT_TITLE", None),
            )
        print(f"Wrote mosaic with {len(fields)} frame(s) to {mosaic_out}")
        return 0

    assert out_dir is not None
    for (idx, step_s, n, kd_work, i1, i2, j1, j2), snap, field in zip(meta, kept_snaps, fields):
        base = f"{idx:06d}_{snap.which}_step{step_s}_i1{i1}_i2{i2}_j1{j1}_j2{j2}_b{snap.batch}.png"
        title = (
            f"ABw ({snap.which})  step={step_s}  batch={snap.batch}\\n"
            f"i1={i1} i2={i2} j1={j1} j2={j2}  n={n} kd_work={kd_work}  mode={args.mode}"
        )
        if args.scale == "per-frame" and (args.vmin is None or args.vmax is None):
            local_vmin = float(np.min(field)) if args.vmin is None else args.vmin
            local_vmax = float(np.max(field)) if args.vmax is None else args.vmax
        else:
            local_vmin = vmin
            local_vmax = vmax

        plot_heatmap(field, title=title, out_path=out_dir / base, cmap=cmap, vmin=local_vmin, vmax=local_vmax)

    print(f"Wrote {len(snaps)} frame(s) to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
