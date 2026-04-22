from __future__ import annotations

import argparse
import os
import subprocess
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

from bench_common import save_figure, with_device_title
import stylesheet


METRIC_INFO = {
    "R": {
        "raw": "R",
        "log": "log10_R",
        "ylabel": r"$\log_{10}(\|AZ - Z\Lambda\|_F)$",
        "name": "residual",
    },
    "O": {
        "raw": "O",
        "log": "log10_O",
        "ylabel": r"$\log_{10}(\|Z^T Z - I\|_F)$",
        "name": "orthogonality",
    },
    "relerr": {
        "raw": "max_relerr",
        "log": "log10_relerr",
        "ylabel": r"$\log_{10}\left(\max_i |\hat{\lambda}_i - \lambda_i^{\mathrm{ref}}|\right)$",
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

DEFAULT_IMPL_ORDER = [
    "steqr_cta_exp",
    "steqr_cta_pg",
    "stedc",
    "syev_vendor",
    "syev_cta_exp",
    "syev_cta_pg",
    "syev_blocked",
    "syevx",
]

IMPL_ALIASES = {
    "cuda": "syev_vendor",
    "vendor": "syev_vendor",
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


def _normalize_impl_name(value: str) -> str:
    key = (value or "").strip()
    if not key:
        return key
    return IMPL_ALIASES.get(key.lower(), key)


def _default_impls(df: pd.DataFrame) -> list[str]:
    available_impls = set(df["impl"].astype(str).tolist())
    impls = [impl for impl in DEFAULT_IMPL_ORDER if impl in available_impls]
    if impls:
        return impls
    return sorted(available_impls)


def _wants_vendor_syev(explicit_impls: list[str]) -> bool:
    if explicit_impls:
        return "syev_vendor" in explicit_impls
    return True


def _run_command(cmd: list[str], *, env: Optional[dict[str, str]] = None) -> None:
    run_env = os.environ.copy()
    env_prefix = ""
    if env:
        run_env.update(env)
        env_prefix = " ".join(f"{k}={v}" for k, v in env.items()) + " "
    print(f"Running: {env_prefix}{' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=run_env)


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
    if "impl" not in df.columns and "tag_impl" in df.columns:
        df["impl"] = df["tag_impl"]
    if "backend" not in df.columns and "tag_backend" in df.columns:
        df["backend"] = df["tag_backend"]
    if "dtype" not in df.columns and "tag_dtype" in df.columns:
        df["dtype"] = df["tag_dtype"]
    if "n" not in df.columns and "arg0" in df.columns:
        df["n"] = pd.to_numeric(df["arg0"], errors="coerce")
    if raw_col == "max_relerr" and "max_relerr" not in df.columns and "relerr" in df.columns:
        df["max_relerr"] = df["relerr"]
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


def _normalize_optional_str_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.lower().replace({"nan": ""})


def _tag_benchmark_run(
    df: pd.DataFrame,
    *,
    scheme: Optional[str] = None,
    cta_shift: Optional[str] = None,
) -> pd.DataFrame:
    df = df.copy()

    if "scheme" not in df.columns:
        df["scheme"] = ""

    if "cta_shift" not in df.columns:
        df["cta_shift"] = ""

    if scheme is None and cta_shift is None:
        return df

    impl_series = df["impl"].astype(str)
    steqr_sensitive = (
        (impl_series == "stedc")
        | impl_series.str.startswith("steqr_cta_")
        | impl_series.str.startswith("syev_cta_")
        | (impl_series == "syev_cta_dispatch")
    )

    if scheme is not None:
        missing = df["scheme"].isna() | (df["scheme"].astype(str).str.strip() == "")
        df.loc[missing & steqr_sensitive, "scheme"] = scheme

    if cta_shift is not None:
        missing = df["cta_shift"].isna() | (df["cta_shift"].astype(str).str.strip() == "")
        df.loc[missing & steqr_sensitive, "cta_shift"] = cta_shift

    return df


def _variant_impl_label(base_impl: str, scheme: str, cta_shift: str) -> str:
    if base_impl == "stedc":
        parts = [base_impl]
        if scheme:
            parts.append(scheme)
        if cta_shift:
            parts.append(cta_shift)
        return "_".join(parts)

    if base_impl.startswith("steqr_cta_") or base_impl.startswith("syev_cta_"):
        if cta_shift:
            return f"{base_impl}_{cta_shift}"

    return base_impl


def _annotate_impl_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    impl_series = df["impl"].astype(str)
    scheme_series = _normalize_optional_str_series(df["scheme"]) if "scheme" in df.columns else pd.Series("", index=df.index)
    shift_series = _normalize_optional_str_series(df["cta_shift"]) if "cta_shift" in df.columns else pd.Series("", index=df.index)

    df["impl_variant"] = [
        _variant_impl_label(base_impl, scheme, cta_shift)
        for base_impl, scheme, cta_shift in zip(impl_series, scheme_series, shift_series)
    ]
    df["impl_plot"] = impl_series

    variant_counts = df.groupby("impl", dropna=False)["impl_variant"].nunique()
    ambiguous_impls = {impl for impl, count in variant_counts.items() if count > 1}
    if ambiguous_impls:
        mask = df["impl"].astype(str).isin(ambiguous_impls)
        df.loc[mask, "impl_plot"] = df.loc[mask, "impl_variant"]

    return df


def _impl_plot_sort_key(value: str) -> tuple[int, int, int, str]:
    scheme_rank = 99
    if "_exp" in value:
        scheme_rank = 0
    elif "_pg" in value:
        scheme_rank = 1

    shift_rank = 99
    if value.endswith("_lapack"):
        shift_rank = 0
    elif value.endswith("_wilkinson"):
        shift_rank = 1

    family_rank = 1 if value.startswith("stedc") else 0
    return (family_rank, scheme_rank, shift_rank, value)


def _resolve_plot_impls(df: pd.DataFrame, explicit_impls: list[str]) -> list[str]:
    available_impls = set(df["impl"].astype(str).tolist())
    available_plot_impls = set(df["impl_plot"].astype(str).tolist())

    requested_impls = explicit_impls if explicit_impls else _default_impls(df)
    resolved_impls: list[str] = []
    missing_impls: list[str] = []

    for requested in requested_impls:
        if requested in available_plot_impls:
            if requested not in resolved_impls:
                resolved_impls.append(requested)
            continue

        if requested not in available_impls:
            missing_impls.append(requested)
            continue

        variants = sorted(
            set(df.loc[df["impl"].astype(str) == requested, "impl_plot"].astype(str).tolist()),
            key=_impl_plot_sort_key,
        )
        for variant in variants:
            if variant not in resolved_impls:
                resolved_impls.append(variant)

    if missing_impls:
        raise ValueError(f"No rows matched impl(s) {missing_impls}")

    return resolved_impls


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
    df = _annotate_impl_labels(df)
    if "impl" not in df.columns:
        raise ValueError("multi-plot requires 'impl' column")
    if "n" not in df.columns:
        raise ValueError("multi-plot requires 'n' column")

    impls = _resolve_plot_impls(df, [imp for imp in impls if imp])

    ns = [int(n) for n in ns]
    available_ns = sorted(set(df["n"].astype(int).tolist()))
    ns = [n for n in ns if n in available_ns]
    if not ns:
        raise ValueError("No rows matched requested N values")

    df = df[df["impl_plot"].astype(str).isin(impls) & df["n"].astype(int).isin(ns)].copy()
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
            dfi = df[(df["impl_plot"].astype(str) == impl) & (df["n"].astype(int) == n)].copy()
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
        fig.suptitle(with_device_title(f"{title} ({subtitle})"))
    else:
        fig.suptitle(with_device_title(title))
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
    df = _annotate_impl_labels(df)
    if "impl" not in df.columns:
        raise ValueError("mean-lines plot requires 'impl' column")
    if "n" not in df.columns:
        raise ValueError("mean-lines plot requires 'n' column")

    impls = _resolve_plot_impls(df, [imp for imp in impls if imp])

    ns = [int(n) for n in ns]
    available_ns = sorted(set(df["n"].astype(int).tolist()))
    ns = [n for n in ns if n in available_ns]
    if not ns:
        raise ValueError("No rows matched requested N values")

    df = df[df["impl_plot"].astype(str).isin(impls) & df["n"].astype(int).isin(ns)].copy()
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
            dfi = dfn[dfn["impl_plot"].astype(str) == impl]
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
    backend = _unique_or_none(df, "backend")
    dtype = _unique_or_none(df, "dtype")
    subtitle = ", ".join([v for v in [backend, dtype] if v])
    title = f"Eigensolver mean {METRIC_INFO[metric]['name']} vs conditioning"
    if subtitle:
        title = f"{title} ({subtitle})"
    fig.suptitle(with_device_title(title))
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
    parser.add_argument(
        "--bench-run-vendor-syev",
        action="store_true",
        help="run an additional eigensolver_accuracy --impl=syev pass with BATCHLAS_SYEV_PROVIDER=VENDOR",
    )

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

    impls = [_normalize_impl_name(s) for s in args.impls.split(",")] if args.impls else []
    wants_vendor_syev = _wants_vendor_syev(impls)

    if args.run:
        run_ns = _parse_ns(args.ns)
        schemes = _parse_bench_schemes(args.bench_scheme)
        temp_paths = []
        run_vendor_syev = args.bench_run_vendor_syev or (
            wants_vendor_syev and args.bench_backend.upper() == "CUDA"
        )

        for n in run_ns:
            for scheme in schemes:
                out_path = f"{args.csv}.n{n}.{scheme}.tmp"
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

                _run_command(cmd)
                temp_paths.append(out_path)

            if run_vendor_syev:
                out_path = f"{args.csv}.n{n}.vendor.tmp"
                cmd = [
                    args.bench_bin,
                    "--impl=syev",
                    f"--backend={args.bench_backend}",
                    f"--type={args.bench_type}",
                    f"--n={n}",
                    f"--samples={args.bench_samples}",
                    f"--batch={args.bench_batch}",
                    f"--log10-cond-min={args.bench_log10_cond_min}",
                    f"--log10-cond-max={args.bench_log10_cond_max}",
                    f"--seed={args.bench_seed}",
                    f"--output={out_path}",
                ]
                _run_command(cmd, env={"BATCHLAS_SYEV_PROVIDER": "VENDOR"})
                temp_paths.append(out_path)

        if temp_paths:
            frames = []
            for path in temp_paths:
                frame = pd.read_csv(path)
                scheme = None
                if ".pg.tmp" in path:
                    scheme = "pg"
                elif ".exp.tmp" in path:
                    scheme = "exp"
                frame = _tag_benchmark_run(frame, scheme=scheme, cta_shift=args.bench_cta_shift)
                frames.append(frame)
            df_concat = pd.concat(frames, ignore_index=True)
            df_concat.to_csv(args.csv, index=False)
            for path in temp_paths:
                try:
                    os.remove(path)
                except OSError:
                    pass

    if not os.path.isfile(args.csv):
        raise FileNotFoundError(f"CSV not found: {args.csv}")

    df = pd.read_csv(args.csv, low_memory=False)
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
