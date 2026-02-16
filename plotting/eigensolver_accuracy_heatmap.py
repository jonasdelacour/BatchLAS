from __future__ import annotations

import argparse
import os
import subprocess
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

from bench_common import save_figure
import stylesheet


METRIC_INFO = {
    "R": {
        "raw": "R",
        "log": "log10_R",
        "ylabel": r"$\log_{10}(R)$",
        "name": "residual",
    },
    "O": {
        "raw": "O",
        "log": "log10_O",
        "ylabel": r"$\log_{10}(O)$",
        "name": "orthogonality",
    },
    "relerr": {
        "raw": "max_relerr",
        "log": "log10_relerr",
        "ylabel": r"$\log_{10}(\mathrm{max\ relative\ error})$",
        "name": "relerr",
    },
    "res_num": {
        "raw": "res_num",
        "log": "log10_res_num",
        "ylabel": r"$\log_{10}(\|AZ-Z\Lambda\|_F)$",
        "name": "residual_raw",
    },
    "ortho_num": {
        "raw": "ortho_num",
        "log": "log10_ortho_num",
        "ylabel": r"$\log_{10}(\|Z^TZ-I\|_F)$",
        "name": "orthogonality_raw",
    },
}


def _default_csv_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "..", "output", "accuracy", "eigensolver_accuracy.csv")


def _default_plot_path(metric: str) -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    suffix = METRIC_INFO[metric]["name"]
    return os.path.join(here, "..", "output", "plots", f"eigensolver_accuracy_{suffix}_heatmap.png")


def _default_mean_plot_path(metric: str) -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    suffix = METRIC_INFO[metric]["name"]
    return os.path.join(here, "..", "output", "plots", f"eigensolver_accuracy_{suffix}_mean_lines.png")


def _derive_output_paths(base_output: Optional[str], metric: str) -> tuple[str, str]:
    if not base_output:
        return _default_plot_path(metric), _default_mean_plot_path(metric)
    root, ext = os.path.splitext(base_output)
    if not ext:
        ext = ".png"
    suffix = METRIC_INFO[metric]["name"]
    heatmap = f"{root}_{suffix}_heatmap{ext}"
    mean = f"{root}_{suffix}_mean_lines{ext}"
    return heatmap, mean


def _default_bench_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "..", "build", "benchmarks", "eigensolver_accuracy")


def _unique_or_none(df: pd.DataFrame, col: str) -> Optional[str]:
    if col not in df.columns:
        return None
    vals = sorted(set(df[col].astype(str).tolist()))
    if len(vals) == 1:
        return vals[0]
    return None


def _prepare_dataframe(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    info = METRIC_INFO[metric]
    raw_col = info["raw"]
    log_col = info["log"]

    df = df.copy()
    if "cond" in df.columns and "log10_cond" not in df.columns:
        df["log10_cond"] = np.log10(np.maximum(df["cond"].astype(float), np.finfo(float).tiny))
    if "target_log10_cond" in df.columns and "log10_cond" not in df.columns:
        df["log10_cond"] = df["target_log10_cond"].astype(float)

    if raw_col in df.columns and log_col not in df.columns:
        df[log_col] = np.log10(np.maximum(np.abs(df[raw_col].astype(float)), np.finfo(float).tiny))

    for col in ("log10_cond", log_col):
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in CSV")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["log10_cond"])
    return df


def _parse_bench_schemes(value: str) -> list[str]:
    key = (value or "").strip().lower()
    if key in {"", "both"}:
        return ["pg", "exp"]
    if "," in key:
        schemes = [s.strip() for s in key.split(",") if s.strip()]
        valid_schemes = {"pg", "exp"}
        for scheme in schemes:
            if scheme not in valid_schemes:
                raise ValueError(f"Invalid scheme '{scheme}'. Must be one of: pg, exp")
        return schemes
    if key in {"pg", "exp"}:
        return [key]
    raise ValueError("--bench-scheme must be one of: pg, exp, both, or comma-separated list")


def _make_bins(values: np.ndarray, bins: int, *, clamp: Optional[Tuple[float, float]] = None) -> np.ndarray:
    if clamp is not None:
        vmin, vmax = clamp
    else:
        vmin = float(np.nanmin(values))
        vmax = float(np.nanmax(values))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        vmin, vmax = -1.0, 1.0
    elif vmin == vmax:
        pad = 1.0 if vmin == 0.0 else max(1.0, abs(vmin) * 0.1)
        vmin, vmax = vmin - pad, vmin + pad
    return np.linspace(vmin, vmax, bins + 1)


def _hist2d(df: pd.DataFrame, metric_log_col: str, *, xedges: np.ndarray, yedges: np.ndarray) -> np.ndarray:
    x = df["log10_cond"].to_numpy()
    y = df[metric_log_col].to_numpy()
    H, _, _ = np.histogram2d(x, y, bins=[xedges, yedges], density=False)
    return H.T


def _normalize_histogram(H: np.ndarray, xedges: np.ndarray, yedges: np.ndarray) -> np.ndarray:
    H = np.array(H, dtype=float, copy=True)
    total = float(np.sum(H))
    if total <= 0.0:
        return H
    area = np.outer(np.diff(yedges), np.diff(xedges))
    H /= (total * area)
    return H


def _bin_means(x: np.ndarray, y: np.ndarray, xedges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    inds = np.digitize(x, xedges) - 1
    centers = 0.5 * (xedges[:-1] + xedges[1:])
    means = np.full_like(centers, np.nan, dtype=float)
    for i in range(len(centers)):
        in_bin = inds == i
        if np.any(in_bin):
            yi = y[in_bin]
            yi = yi[np.isfinite(yi)]
            if yi.size:
                means[i] = float(np.mean(yi))
    return centers, means


def _bin_failure_rate(x: np.ndarray, y: np.ndarray, xedges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    inds = np.digitize(x, xedges) - 1
    centers = 0.5 * (xedges[:-1] + xedges[1:])
    rates = np.full_like(centers, np.nan, dtype=float)
    for i in range(len(centers)):
        in_bin = inds == i
        total = int(np.count_nonzero(in_bin))
        if total > 0:
            failures = int(np.count_nonzero(~np.isfinite(y[in_bin])))
            rates[i] = failures / float(total)
    return centers, rates


def plot_multi_heatmap(
    df: pd.DataFrame,
    metric: str,
    *,
    impls: list[str],
    ns: list[int],
    x_bins: int,
    y_bins: int,
    clamp_x: Optional[Tuple[float, float]],
    clamp_y: Optional[Tuple[float, float]],
    log_color: bool,
    output: Optional[str],
) -> None:
    info = METRIC_INFO[metric]
    metric_log_col = info["log"]
    y_label = info["ylabel"]

    df = _prepare_dataframe(df, metric)
    if "impl" not in df.columns:
        raise ValueError("multi-plot requires 'impl' column")
    if "n" not in df.columns:
        raise ValueError("multi-plot requires 'n' column")

    impls = [imp for imp in impls if imp]
    if not impls:
        default_impls = [
            "steqr_cta_exp",
            "steqr_cta_pg",
            "stedc",
            "syev_cta_exp",
            "syev_cta_pg",
            "syev_blocked",
            "syevx",
        ]
        available_impls = set(df["impl"].astype(str).tolist())
        impls = [imp for imp in default_impls if imp in available_impls]
        if not impls:
            impls = sorted(available_impls)
    else:
        missing_impls = [imp for imp in impls if imp not in set(df["impl"].astype(str).tolist())]
        if missing_impls:
            raise ValueError(f"No rows matched impl(s) {missing_impls}")

    ns = [int(n) for n in ns]
    available_ns = sorted(set(df["n"].astype(int).tolist()))
    ns = [n for n in ns if n in available_ns]
    if not ns:
        raise ValueError("No rows matched requested N values")

    df = df[df["impl"].astype(str).isin(impls) & df["n"].astype(int).isin(ns)].copy()
    if df.empty:
        raise ValueError("No rows matched impl/N filters")

    success_df = df[np.isfinite(df[metric_log_col].to_numpy())].copy()
    if success_df.empty:
        raise ValueError(f"No successful rows (finite {metric_log_col}) matched impl/N filters")

    xedges = _make_bins(df["log10_cond"].to_numpy(), x_bins, clamp=clamp_x)
    yedges = _make_bins(success_df[metric_log_col].to_numpy(), y_bins, clamp=clamp_y)

    hists = []
    failure_curves = []
    for n in ns:
        row = []
        failure_row = []
        for impl in impls:
            dfi = df[(df["impl"].astype(str) == impl) & (df["n"].astype(int) == n)].copy()
            if dfi.empty:
                raise ValueError(f"No rows matched impl='{impl}' with n={n}")
            dfi_success = dfi[np.isfinite(dfi[metric_log_col].to_numpy())].copy()
            H = _hist2d(dfi_success, metric_log_col, xedges=xedges, yedges=yedges)
            H = _normalize_histogram(H, xedges, yedges)
            row.append(H)
            fcenters, frates = _bin_failure_rate(
                dfi["log10_cond"].to_numpy(),
                dfi[metric_log_col].to_numpy(),
                xedges,
            )
            failure_row.append((fcenters, frates))
        hists.append(row)
        failure_curves.append(failure_row)

    all_vals = np.concatenate([h.ravel() for row in hists for h in row]) if hists else np.array([0.0])
    all_vals = np.nan_to_num(all_vals, nan=0.0, posinf=0.0, neginf=0.0)
    max_val = float(np.max(all_vals)) if all_vals.size else 0.0
    min_pos = float(np.min(all_vals[all_vals > 0])) if np.any(all_vals > 0) else None
    if log_color and (min_pos is None or max_val <= 0.0):
        log_color = False

    norm = LogNorm(vmin=min_pos, vmax=max_val) if log_color else None

    rows = len(ns)
    cols = len(impls)
    fig, axes = plt.subplots(rows, cols, sharex=True, sharey=True, figsize=(5 * cols, 4 * rows), constrained_layout=True)
    axes = np.array(axes).reshape(rows, cols)

    mesh = None
    for r, n in enumerate(ns):
        for c, impl in enumerate(impls):
            ax = axes[r, c]
            H = np.nan_to_num(hists[r][c], nan=0.0, posinf=0.0, neginf=0.0)
            mesh = ax.pcolormesh(xedges, yedges, H, shading="auto", norm=norm)

            fcenters, frates = failure_curves[r][c]
            finite_fail = np.isfinite(frates)
            ax_fail = ax.twinx()
            if np.any(finite_fail):
                ax_fail.plot(fcenters[finite_fail], frates[finite_fail], color="C3", linewidth=1.6, alpha=0.9)
            ax_fail.set_ylim(0.0, 1.0)
            ax_fail.set_yticks([0.0, 0.5, 1.0])
            if c == cols - 1:
                ax_fail.set_ylabel("Failure probability", color="C3")
                ax_fail.tick_params(axis="y", colors="C3")
            else:
                ax_fail.set_yticklabels([])
                ax_fail.tick_params(axis="y", length=0)

            if r == 0:
                ax.set_title(impl)
            if c == 0:
                ax.set_ylabel(f"N={n}\n" + y_label)
            if r == rows - 1:
                ax.set_xlabel(r"$\log_{10}(\kappa(A))$")
            ax.grid(True, alpha=0.2)

    if mesh is not None:
        fig.colorbar(mesh, ax=axes.ravel().tolist(), label="Probability density", fraction=0.046, pad=0.04)

    backend = _unique_or_none(df, "backend")
    dtype = _unique_or_none(df, "dtype")
    subtitle = ", ".join([v for v in [backend, dtype] if v])
    title = f"Eigensolver accuracy heatmaps ({METRIC_INFO[metric]['name']})"
    if subtitle:
        fig.suptitle(f"{title} ({subtitle})")
    else:
        fig.suptitle(title)
    save_figure(fig, output or _default_plot_path(metric))


def plot_mean_lines_by_n(
    df: pd.DataFrame,
    metric: str,
    *,
    impls: list[str],
    ns: list[int],
    x_bins: int,
    clamp_x: Optional[Tuple[float, float]],
    output: Optional[str],
) -> None:
    info = METRIC_INFO[metric]
    metric_log_col = info["log"]
    y_label = info["ylabel"]

    df = _prepare_dataframe(df, metric)
    if "impl" not in df.columns:
        raise ValueError("mean-lines plot requires 'impl' column")
    if "n" not in df.columns:
        raise ValueError("mean-lines plot requires 'n' column")

    impls = [imp for imp in impls if imp]
    if not impls:
        impls = sorted(set(df["impl"].astype(str).tolist()))
    else:
        missing_impls = [imp for imp in impls if imp not in set(df["impl"].astype(str).tolist())]
        if missing_impls:
            raise ValueError(f"No rows matched impl(s) {missing_impls}")

    ns = [int(n) for n in ns]
    available_ns = sorted(set(df["n"].astype(int).tolist()))
    ns = [n for n in ns if n in available_ns]
    if not ns:
        raise ValueError("No rows matched requested N values")

    df = df[df["impl"].astype(str).isin(impls) & df["n"].astype(int).isin(ns)].copy()
    if df.empty:
        raise ValueError("No rows matched impl/N filters")

    xedges = _make_bins(df["log10_cond"].to_numpy(), x_bins, clamp=clamp_x)
    x_min = xedges[0]
    x_max = np.min([xedges[-1], 12.0])

    fig, axes = plt.subplots(1, len(ns), sharex=True, sharey=True, figsize=(3.4 * len(ns), 3.2))
    if len(ns) == 1:
        axes = [axes]

    handles = []
    labels = []
    for ax, n in zip(axes, ns):
        dfn = df[df["n"].astype(int) == n]
        for impl in impls:
            dfi = dfn[dfn["impl"].astype(str) == impl]
            centers, means = _bin_means(dfi["log10_cond"].to_numpy(), dfi[metric_log_col].to_numpy(), xedges)
            mask = np.isfinite(means)
            line, = ax.plot(centers[mask], means[mask], label=impl, linewidth=1.6)
            if len(handles) < len(impls):
                handles.append(line)
                labels.append(impl)
        ax.set_title(f"N={n}", fontsize=12)
        if ax is axes[0]:
            ax.set_ylabel(y_label, fontsize=12)
        ax.set_xlim(x_min, x_max)
        ax.grid(True, alpha=0.2)
        ax.set_xlabel(r"$\log_{10}(\kappa(A))$", fontsize=12)
        ax.tick_params(axis="both", which="major", labelsize=10)

    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=max(1, len(impls)), frameon=False, fontsize=10, bbox_to_anchor=(0.5, 1.02))
    plt.subplots_adjust(top=0.88)
    save_figure(fig, output or _default_mean_plot_path(metric))


def _parse_clamp(values: Optional[str]) -> Optional[Tuple[float, float]]:
    if not values:
        return None
    parts = [p.strip() for p in values.split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError("Clamp must be 'min,max'")
    v0 = float(parts[0])
    v1 = float(parts[1])
    return (v0, v1) if v0 <= v1 else (v1, v0)


def _resolve_clamp(
    clamp_csv: Optional[str],
    clamp_min: Optional[float],
    clamp_max: Optional[float],
    *,
    name: str,
) -> Optional[Tuple[float, float]]:
    if clamp_csv is not None and (clamp_min is not None or clamp_max is not None):
        raise ValueError(f"Use either --{name} or --{name}-min/--{name}-max, not both")

    if clamp_csv is not None:
        return _parse_clamp(clamp_csv)

    if clamp_min is None and clamp_max is None:
        return None
    if clamp_min is None or clamp_max is None:
        raise ValueError(f"Both --{name}-min and --{name}-max must be provided together")

    return (clamp_min, clamp_max) if clamp_min <= clamp_max else (clamp_max, clamp_min)


def _parse_ns(values: Optional[str]) -> list[int]:
    if not values:
        return [4, 8, 16, 32]
    parts = [p.strip() for p in values.split(",") if p.strip()]
    if not parts:
        return [4, 8, 16, 32]
    return [int(p) for p in parts]


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot eigensolver accuracy heatmaps")
    parser.add_argument("--csv", default=_default_csv_path(), help="input CSV from eigensolver_accuracy")
    parser.add_argument("--output", default=None, help="output image base path (metric-specific suffixes added)")
    parser.add_argument(
        "--metric",
        default="all",
        choices=["all", "R", "O", "relerr", "res_num", "ortho_num"],
        help="which metric to plot",
    )

    parser.add_argument("--run", action="store_true", help="run eigensolver_accuracy before plotting")
    parser.add_argument("--bench-bin", default=_default_bench_path(), help="path to eigensolver_accuracy binary")
    parser.add_argument("--bench-impl", default="all", help="eigensolver_accuracy --impl value")
    parser.add_argument("--bench-backend", default="CUDA", help="eigensolver_accuracy --backend value")
    parser.add_argument("--bench-type", default="float", help="eigensolver_accuracy --type value (float|double)")
    parser.add_argument("--bench-scheme", default="both", help="eigensolver_accuracy --scheme for CTA variants (pg|exp|both)")
    parser.add_argument("--bench-samples", type=int, default=20000, help="eigensolver_accuracy --samples value")
    parser.add_argument("--bench-batch", type=int, default=256, help="eigensolver_accuracy --batch value")
    parser.add_argument("--bench-log10-cond-min", type=float, default=0.0, help="eigensolver_accuracy --log10-cond-min value")
    parser.add_argument("--bench-log10-cond-max", type=float, default=12.0, help="eigensolver_accuracy --log10-cond-max value")
    parser.add_argument("--bench-seed", type=int, default=1234, help="eigensolver_accuracy --seed value")
    parser.add_argument("--bench-max-sweeps", type=int, default=None, help="eigensolver_accuracy --max-sweeps value")
    parser.add_argument("--bench-cta-shift", default=None, help="eigensolver_accuracy --cta-shift value (lapack|wilkinson)")
    parser.add_argument("--bench-sytrd-block-size", type=int, default=None, help="eigensolver_accuracy --sytrd-block-size value")
    parser.add_argument("--bench-ormqr-block-size", type=int, default=None, help="eigensolver_accuracy --ormqr-block-size value")
    parser.add_argument("--bench-syevx-iterations", type=int, default=None, help="eigensolver_accuracy --syevx-iterations value")
    parser.add_argument("--bench-syevx-extra-directions", type=int, default=None, help="eigensolver_accuracy --syevx-extra-directions value")
    parser.add_argument("--bench-syevx-neigs", type=int, default=None, help="eigensolver_accuracy --syevx-neigs value")
    parser.add_argument("--bench-syevx-find-largest", default=None, help="eigensolver_accuracy --syevx-find-largest value (0|1|true|false)")

    parser.add_argument("--impls", default=None, help="comma-separated impl list for plots")
    parser.add_argument("--ns", default=None, help="comma-separated N list for plots (default: 4,8,16,32)")
    parser.add_argument("--x-bins", type=int, default=60, help="number of bins for log10(cond)")
    parser.add_argument("--y-bins", type=int, default=60, help="number of bins for metric")
    parser.add_argument("--clamp-x", default=None, help="clamp log10(cond) to min,max (e.g. --clamp-x=-2,8)")
    parser.add_argument("--clamp-y", default=None, help="clamp metric to min,max (e.g. --clamp-y=-10,2)")
    parser.add_argument("--clamp-x-min", type=float, default=None, help="minimum clamp for log10(cond)")
    parser.add_argument("--clamp-x-max", type=float, default=None, help="maximum clamp for log10(cond)")
    parser.add_argument("--clamp-y-min", type=float, default=None, help="minimum clamp for metric")
    parser.add_argument("--clamp-y-max", type=float, default=None, help="maximum clamp for metric")
    parser.add_argument("--linear-color", action="store_true", help="use linear color scale instead of log")

    args = parser.parse_args()
    args.bench_type = args.bench_type.lower()
    if args.bench_type not in {"float", "double"}:
        raise ValueError("--bench-type must be 'float' or 'double'")

    if args.run:
        run_ns = _parse_ns(args.ns)
        schemes = _parse_bench_schemes(args.bench_scheme)
        temp_paths = []

        for n in run_ns:
            for scheme in schemes:
                single_output = (len(run_ns) == 1) and (len(schemes) == 1)
                out_path = args.csv if single_output else f"{args.csv}.n{n}.{scheme}.tmp"
                cmd = [
                    args.bench_bin,
                    f"--impl={args.bench_impl}",
                    f"--backend={args.bench_backend}",
                    f"--type={args.bench_type}",
                    f"--scheme={scheme}",
                    f"--n={n}",
                    f"--samples={args.bench_samples}",
                    f"--batch={args.bench_batch}",
                    f"--log10-cond-min={args.bench_log10_cond_min}",
                    f"--log10-cond-max={args.bench_log10_cond_max}",
                    f"--seed={args.bench_seed}",
                    f"--output={out_path}",
                ]
                if args.bench_max_sweeps is not None:
                    cmd.append(f"--max-sweeps={args.bench_max_sweeps}")
                if args.bench_cta_shift:
                    cmd.append(f"--cta-shift={args.bench_cta_shift}")
                if args.bench_sytrd_block_size is not None:
                    cmd.append(f"--sytrd-block-size={args.bench_sytrd_block_size}")
                if args.bench_ormqr_block_size is not None:
                    cmd.append(f"--ormqr-block-size={args.bench_ormqr_block_size}")
                if args.bench_syevx_iterations is not None:
                    cmd.append(f"--syevx-iterations={args.bench_syevx_iterations}")
                if args.bench_syevx_extra_directions is not None:
                    cmd.append(f"--syevx-extra-directions={args.bench_syevx_extra_directions}")
                if args.bench_syevx_neigs is not None:
                    cmd.append(f"--syevx-neigs={args.bench_syevx_neigs}")
                if args.bench_syevx_find_largest is not None:
                    cmd.append(f"--syevx-find-largest={args.bench_syevx_find_largest}")

                print("Running:", " ".join(cmd))
                subprocess.run(cmd, check=True)
                if not single_output:
                    temp_paths.append(out_path)

        if temp_paths:
            frames = [pd.read_csv(path) for path in temp_paths]
            df_concat = pd.concat(frames, ignore_index=True)
            df_concat.to_csv(args.csv, index=False)
            for path in temp_paths:
                try:
                    os.remove(path)
                except OSError:
                    pass

    if not os.path.isfile(args.csv):
        raise FileNotFoundError(f"CSV not found: {args.csv}")

    df = pd.read_csv(args.csv)
    impls = [s.strip() for s in args.impls.split(",")] if args.impls else []
    ns = _parse_ns(args.ns)

    clamp_x = _resolve_clamp(args.clamp_x, args.clamp_x_min, args.clamp_x_max, name="clamp-x")
    clamp_y = _resolve_clamp(args.clamp_y, args.clamp_y_min, args.clamp_y_max, name="clamp-y")

    metrics = ["R", "O", "relerr", "res_num", "ortho_num"] if args.metric == "all" else [args.metric]

    for metric in metrics:
        heatmap_output, mean_output = _derive_output_paths(args.output, metric)
        plot_multi_heatmap(
            df,
            metric,
            impls=impls,
            ns=ns,
            x_bins=args.x_bins,
            y_bins=args.y_bins,
            clamp_x=clamp_x,
            clamp_y=clamp_y,
            log_color=not args.linear_color,
            output=heatmap_output,
        )
        plot_mean_lines_by_n(
            df,
            metric,
            impls=impls,
            ns=ns,
            x_bins=args.x_bins,
            clamp_x=clamp_x,
            output=mean_output,
        )


if __name__ == "__main__":
    main()
