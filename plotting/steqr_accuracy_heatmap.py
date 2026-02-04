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



def _default_csv_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "..", "output", "accuracy", "steqr_accuracy.csv")


def _default_plot_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "..", "output", "plots", "steqr_accuracy_heatmap.png")


def _default_mean_plot_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "..", "output", "plots", "steqr_accuracy_mean_lines.png")


def _derive_output_paths(base_output: Optional[str]) -> tuple[str, str]:
    if not base_output:
        return _default_plot_path(), _default_mean_plot_path()
    root, ext = os.path.splitext(base_output)
    if not ext:
        ext = ".png"
    heatmap = f"{root}_heatmap{ext}"
    mean = f"{root}_mean_lines{ext}"
    return heatmap, mean


def _default_bench_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "..", "build", "benchmarks", "steqr_accuracy")


def _unique_or_none(df: pd.DataFrame, col: str) -> Optional[str]:
    if col not in df.columns:
        return None
    vals = sorted(set(df[col].astype(str).tolist()))
    if len(vals) == 1:
        return vals[0]
    return None


def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "dtype" in df.columns:
        bad_types = {"cfloat", "cdouble", "complex<float>", "complex<double>"}
        if any(str(v) in bad_types for v in df["dtype"].astype(str).tolist()):
            raise ValueError("Complex dtypes are not supported for STEQR accuracy plots")
    if "cond" in df.columns and "log10_cond" not in df.columns:
        df["log10_cond"] = np.log10(np.maximum(df["cond"].astype(float), np.finfo(float).tiny))
    if "target_log10_cond" in df.columns and "log10_cond" not in df.columns:
        df["log10_cond"] = df["target_log10_cond"].astype(float)
    if "relerr" in df.columns and "log10_relerr" not in df.columns:
        df["log10_relerr"] = np.log10(np.maximum(df["relerr"].astype(float), np.finfo(float).tiny))

    for col in ("log10_cond", "log10_relerr"):
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in CSV")

    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["log10_cond", "log10_relerr"])
    return df


def _sanitize_bench_impl(bench_impl: str, bench_type: str, bench_backend: str) -> str:
    if bench_type not in {"float", "double"}:
        raise ValueError("--bench-type must be 'float' or 'double'")
    impl = bench_impl
    if bench_type == "float":
        if impl == "both" and bench_backend in {"CUDA", "ROCM", "MKL"}:
            impl = "all"
    if bench_type == "double":
        if impl == "netlib_syev32":
            raise ValueError("netlib_syev32 requires --bench-type=float")
    if bench_backend == "NETLIB" and impl in {"both", "all", "steqr_cta", "cuda_syev"}:
        if impl != "all":
            impl = "steqr"
    return impl


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


def _hist2d(df: pd.DataFrame, *, xedges: np.ndarray, yedges: np.ndarray) -> np.ndarray:
    x = df["log10_cond"].to_numpy()
    y = df["log10_relerr"].to_numpy()
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


def plot_heatmap(
    df: pd.DataFrame,
    *,
    x_bins: int,
    y_bins: int,
    clamp_x: Optional[Tuple[float, float]],
    clamp_y: Optional[Tuple[float, float]],
    log_color: bool,
    impl_filter: Optional[str],
    output: Optional[str],
) -> None:
    df = _prepare_dataframe(df)
    if impl_filter:
        if "impl" not in df.columns:
            raise ValueError("impl filter requested but 'impl' column is missing")
        df = df[df["impl"].astype(str) == impl_filter].copy()
        if df.empty:
            raise ValueError(f"No rows matched impl='{impl_filter}'")
    xedges = _make_bins(df["log10_cond"].to_numpy(), x_bins, clamp=clamp_x)
    yedges = _make_bins(df["log10_relerr"].to_numpy(), y_bins, clamp=clamp_y)

    H = _hist2d(df, xedges=xedges, yedges=yedges)
    H = _normalize_histogram(H, xedges, yedges)

    fig, ax = plt.subplots(constrained_layout=True)
    H = np.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0)
    max_val = float(np.max(H)) if H.size else 0.0
    min_pos = float(np.min(H[H > 0])) if np.any(H > 0) else None
    if log_color and (min_pos is None or max_val <= 0.0):
        log_color = False

    norm = LogNorm(vmin=min_pos, vmax=max_val) if log_color else None
    mesh = ax.pcolormesh(xedges, yedges, H, shading="auto", norm=norm)
    fig.colorbar(mesh, ax=ax, label="Probability density", fraction=0.046, pad=0.04)

    ax.set_xlabel(r"$\log_{10}(\kappa(A))$")
    ax.set_ylabel(r"$\log_{10}(\mathrm{relative\ error})$")

    impl = _unique_or_none(df, "impl")
    backend = _unique_or_none(df, "backend")
    dtype = _unique_or_none(df, "dtype")
    title = "STEQR accuracy heatmap"
    subtitle = ", ".join([v for v in [impl, backend, dtype] if v])
    if subtitle:
        title = f"{title} ({subtitle})"
    ax.set_title(title)
    ax.grid(True, alpha=0.2)

    save_figure(fig, output or _default_plot_path())


def plot_multi_heatmap(
    df: pd.DataFrame,
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
    df = _prepare_dataframe(df)
    if "impl" not in df.columns:
        raise ValueError("multi-plot requires 'impl' column")
    if "n" not in df.columns:
        raise ValueError("multi-plot requires 'n' column")

    impls = [imp for imp in impls if imp]
    if not impls:
        # Default impl list for comparison
        default_impls = ["steqr_cta_exp", "steqr_cta_pg", "netlib_steqr", "cuda_syev"]
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

    xedges = _make_bins(df["log10_cond"].to_numpy(), x_bins, clamp=clamp_x)
    yedges = _make_bins(df["log10_relerr"].to_numpy(), y_bins, clamp=clamp_y)

    hists = []
    for n in ns:
        row = []
        for impl in impls:
            dfi = df[(df["impl"].astype(str) == impl) & (df["n"].astype(int) == n)].copy()
            if dfi.empty:
                raise ValueError(f"No rows matched impl='{impl}' with n={n}")
            H = _hist2d(dfi, xedges=xedges, yedges=yedges)
            H = _normalize_histogram(H, xedges, yedges)
            row.append(H)
        hists.append(row)

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
            if r == 0:
                ax.set_title(impl)
            if c == 0:
                ax.set_ylabel(f"N={n}\n" + r"$\log_{10}(\mathrm{relative\ error})$")
            if r == rows - 1:
                ax.set_xlabel(r"$\log_{10}(\kappa(A))$")
            ax.grid(True, alpha=0.2)

    if mesh is not None:
        fig.colorbar(mesh, ax=axes.ravel().tolist(), label="Probability density", fraction=0.046, pad=0.04)
    save_figure(fig, output or _default_plot_path())


def _bin_means(x: np.ndarray, y: np.ndarray, xedges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    inds = np.digitize(x, xedges) - 1
    centers = 0.5 * (xedges[:-1] + xedges[1:])
    means = np.full_like(centers, np.nan, dtype=float)
    for i in range(len(centers)):
        mask = inds == i
        if np.any(mask):
            means[i] = float(np.mean(y[mask]))
    return centers, means


def plot_mean_lines_by_n(
    df: pd.DataFrame,
    *,
    impls: list[str],
    ns: list[int],
    x_bins: int,
    clamp_x: Optional[Tuple[float, float]],
    output: Optional[str],
) -> None:
    df = _prepare_dataframe(df)
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
            centers, means = _bin_means(dfi["log10_cond"].to_numpy(), dfi["log10_relerr"].to_numpy(), xedges)
            mask = np.isfinite(means)
            line, = ax.plot(centers[mask], means[mask], label=impl, linewidth=1.6)
            if len(handles) < len(impls):
                handles.append(line)
                labels.append(impl)
        ax.set_title(f"N={n}", fontsize=12)
        if ax is axes[0]:
            ax.set_ylabel(r"$\log_{10}(\mathrm{relative\ error})$", fontsize=12)
        ax.set_xlim(x_min, x_max)
        ax.grid(True, alpha=0.2)
        ax.set_xlabel(r"$\log_{10}(\kappa(A))$", fontsize=12)
        ax.tick_params(axis="both", which="major", labelsize=10)

    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=len(impls), frameon=False, fontsize=10, bbox_to_anchor=(0.5, 1.02))
    plt.subplots_adjust(top=0.88)
    save_figure(fig, output or _default_mean_plot_path())


def _parse_clamp(values: Optional[str]) -> Optional[Tuple[float, float]]:
    if not values:
        return None
    parts = [p.strip() for p in values.split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError("Clamp must be 'min,max'")
    return float(parts[0]), float(parts[1])


def _parse_ns(values: Optional[str]) -> list[int]:
    if not values:
        return [4, 8, 16, 32]
    parts = [p.strip() for p in values.split(",") if p.strip()]
    if not parts:
        return [4, 8, 16, 32]
    return [int(p) for p in parts]


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot STEQR accuracy heatmap")
    parser.add_argument("--csv", default=_default_csv_path(), help="input CSV from steqr_accuracy")
    parser.add_argument("--output", default=None, help="output image path base (suffixes added for heatmap/mean plots)")
    parser.add_argument(
        "--impl",
        default=None,
        help="filter to a specific impl (e.g. steqr, steqr_cta_pg, steqr_cta_exp, cuda_syev, netlib_syev32)",
    )
    parser.add_argument("--run", action="store_true", help="run steqr_accuracy to generate CSV before plotting")
    parser.add_argument("--bench-bin", default=_default_bench_path(), help="path to steqr_accuracy binary")
    parser.add_argument("--bench-impl", default="all", help="steqr_accuracy --impl value")
    parser.add_argument("--bench-backend", default="CUDA", help="steqr_accuracy --backend value")
    parser.add_argument("--bench-type", default="float", help="steqr_accuracy --type value (float|double)")
    parser.add_argument(
        "--bench-scheme",
        default="both",
        help="steqr_accuracy --scheme value for steqr_cta (pg|exp|both)",
    )
    parser.add_argument("--bench-n", type=int, default=32, help="steqr_accuracy --n value")
    parser.add_argument("--bench-samples", type=int, default=20000, help="steqr_accuracy --samples value")
    parser.add_argument("--bench-batch", type=int, default=256, help="steqr_accuracy --batch value")
    parser.add_argument("--bench-log10-cond-min", type=float, default=0.0, help="steqr_accuracy --log10-cond-min value")
    parser.add_argument("--bench-log10-cond-max", type=float, default=12.0, help="steqr_accuracy --log10-cond-max value")
    parser.add_argument("--bench-seed", type=int, default=1234, help="steqr_accuracy --seed value")
    parser.add_argument("--bench-max-sweeps", type=int, default=None, help="steqr_accuracy --max-sweeps value")
    parser.add_argument("--bench-cta-shift", default=None, help="steqr_accuracy --cta-shift value (lapack|wilkinson)")
    parser.add_argument("--multi", action="store_true", help="plot impl x N grid in a single figure")
    parser.add_argument("--mean-lines", action="store_true", help="plot mean log10(relerr) lines per impl, one subplot per N")
    parser.add_argument(
        "--impls",
        default=None,
        help="comma-separated impl list for plots (default: steqr_cta_exp,steqr_cta_pg,netlib_steqr,cuda_syev)",
    )
    parser.add_argument("--ns", default=None, help="comma-separated N list for plots (default: 4,8,16,32)")
    parser.add_argument("--x-bins", type=int, default=60, help="number of bins for log10(cond)")
    parser.add_argument("--y-bins", type=int, default=60, help="number of bins for log10(relerr)")
    parser.add_argument("--clamp-x", default=None, help="clamp log10(cond) to min,max")
    parser.add_argument("--clamp-y", default=None, help="clamp log10(relerr) to min,max")
    parser.add_argument("--linear-color", action="store_true", help="use linear color scale instead of log")

    args = parser.parse_args()
    args.bench_type = args.bench_type.lower()
    if args.bench_type not in {"float", "double"}:
        raise ValueError("--bench-type must be 'float' or 'double'")

    if args.run:
        bench_impl = _sanitize_bench_impl(args.bench_impl, args.bench_type, args.bench_backend)
        run_ns = _parse_ns(args.ns)
        schemes = _parse_bench_schemes(args.bench_scheme)
        temp_paths = []

        for n in run_ns:
            for scheme in schemes:
                single_output = (len(run_ns) == 1) and (len(schemes) == 1)
                out_path = args.csv if single_output else f"{args.csv}.n{n}.{scheme}.tmp"
                cmd = [
                    args.bench_bin,
                    f"--impl={bench_impl}",
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
    heatmap_output, mean_output = _derive_output_paths(args.output)

    plot_multi_heatmap(
        df,
        impls=impls,
        ns=ns,
        x_bins=args.x_bins,
        y_bins=args.y_bins,
        clamp_x=_parse_clamp(args.clamp_x),
        clamp_y=_parse_clamp(args.clamp_y),
        log_color=not args.linear_color,
        output=heatmap_output,
    )
    plot_mean_lines_by_n(
        df,
        impls=impls,
        ns=ns,
        x_bins=args.x_bins,
        clamp_x=_parse_clamp(args.clamp_x),
        output=mean_output,
    )


if __name__ == "__main__":
    main()
