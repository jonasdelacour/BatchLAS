from __future__ import annotations

import argparse
import os
import tempfile
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from bench_common import load_results, plot_metric, run_benchmark, save_figure


def _here() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _default_bench_path() -> str:
    return os.path.join(_here(), "..", "build", "benchmarks", "syev_benchmark")


def _dtype_to_filesafe_suffix(dtype: str) -> str:
    """Convert minibench dtype to filesystem-safe suffix."""
    dt = dtype.lower().strip()
    if dt in ("float",):
        return "float32"
    elif dt in ("double",):
        return "float64"
    elif dt in ("cfloat", "complex<float>"):
        return "cfloat32"
    elif dt in ("cdouble", "complex<double>"):
        return "cfloat64"
    else:
        return dt.replace(" ", "").replace("<", "_").replace(">", "_")


def _default_csv_path(dtype: str = "float") -> str:
    suffix = _dtype_to_filesafe_suffix(dtype)
    return os.path.join(_here(), "..", "build", f"syev_vendor_{suffix}.csv")


def _default_csv_compare_path(dtype: str = "float") -> str:
    suffix = _dtype_to_filesafe_suffix(dtype)
    return os.path.join(_here(), "..", "build", f"syev_compare_{suffix}.csv")


def _default_plot_path(dtype: str = "float") -> str:
    suffix = _dtype_to_filesafe_suffix(dtype)
    return os.path.join(_here(), "..", "output", "plots", f"syev_provider_compare_{suffix}.png")


def _default_n_values() -> list[int]:
    fine = list(range(4, 33))
    coarse = list(range(40, 513, 8))
    return [*fine, *coarse]


def _default_batches_for_n(n_values: Sequence[int], *, batch_at_n4: int = 65536) -> list[int]:
    """Choose batch sizes with inverse-linear scaling in N.

    The schedule is anchored at N=4 with batch=65536 and then uses:
        batch(N) ~= (4 * batch_at_n4) / N

    Example: 4 -> 65536, 8 -> 32768.
    """
    target = float(batch_at_n4) * 4.0
    batches: list[int] = []
    for n in n_values:
        b = int(round(target / float(n)))
        batches.append(max(1, b))
    return batches


def _run_benchmark_pairs(
    *,
    binary: str,
    csv_out: str,
    n_values: Sequence[int],
    batches: Sequence[int],
    common_args: Sequence[str],
    bench_args: Sequence[str],
    env: Optional[dict[str, str]] = None,
) -> None:
    """Run one benchmark point per (N,batch) pair and aggregate into csv_out.

    This avoids the minibench Cartesian-product behavior when both arg vectors
    are passed as comma-separated lists in a single invocation.
    """
    if len(n_values) != len(batches):
        raise ValueError("n_values and batches must have identical lengths")

    frames: list[pd.DataFrame] = []
    tmp_paths: list[str] = []

    try:
        for n, b in zip(n_values, batches):
            with tempfile.NamedTemporaryFile(prefix="syev_point_", suffix=".csv", delete=False) as tmp:
                tmp_path = tmp.name
            tmp_paths.append(tmp_path)

            run_benchmark(
                binary,
                tmp_path,
                [*common_args, str(n), str(b), *bench_args],
                env=env,
            )
            frames.append(load_results(tmp_path))

        if not frames:
            raise ValueError("No benchmark points were executed")

        out = pd.concat(frames, ignore_index=True)
        os.makedirs(os.path.dirname(os.path.abspath(csv_out)), exist_ok=True)
        out.to_csv(csv_out, index=False)
        print(f"Wrote aggregated paired benchmark CSV: {csv_out}")
    finally:
        for p in tmp_paths:
            try:
                os.remove(p)
            except OSError:
                pass


def _readable_n_ticks(n_values: Sequence[int]) -> list[int]:
    """Pick sparse ticks for dense N grids (4..32 then step-8)."""
    nset = set(int(v) for v in n_values)
    ticks: list[int] = []
    for n in sorted(nset):
        if n <= 32:
            if (n - 4) % 4 == 0:
                ticks.append(n)
        else:
            if n % 32 == 0:
                ticks.append(n)
    # Ensure anchors are always present when available.
    for anchor in (4, 32, 64, 128, 256, 512):
        if anchor in nset and anchor not in ticks:
            ticks.append(anchor)
    return sorted(set(ticks))


def _dtype_to_label(dtype: str) -> str:
    """Map minibench dtype string to readable scalar type label."""
    dt = dtype.lower().strip()
    if dt in ("float",):
        return "Float32"
    elif dt in ("double",):
        return "Float64"
    elif dt in ("cfloat", "complex<float>"):
        return "Cfloat32"
    elif dt in ("cdouble", "complex<double>"):
        return "Cfloat64"
    else:
        return dt


def _require_columns(df: pd.DataFrame, cols: Sequence[str], *, label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{label}: missing columns: {missing}. Available: {list(df.columns)}")


def _pick_metric_column(df: pd.DataFrame) -> str:
    # BatchLAS benchmarks have both styles in the tree.
    for name in ("Time (µs) / matrix", "T(µs)/matrix"):
        if name in df.columns:
            return name
    raise ValueError(
        f"No known time metric column found. Available: {list(df.columns)}"
    )


def _metric_std_column(df: pd.DataFrame, metric: str) -> Optional[str]:
    candidate = f"{metric}_std"
    if candidate in df.columns:
        return candidate
    return None


def _add_derived_metrics(
    df: pd.DataFrame,
    *,
    metric_time: str,
    gflops_factor: float,
    n_field: str = "arg0",
) -> pd.DataFrame:
    """Add derived throughput and effective GFLOPS metrics.

    Throughput assumes the benchmark reports time per matrix in microseconds.
    Effective GFLOPS is `gflops_factor * N^3 / time_us` scaled to 1e9 flop/s.
    """
    _require_columns(df, [n_field, metric_time], label="derived metrics")
    out = df.copy()
    n = out[n_field].astype(float)
    time_us = out[metric_time].astype(float)

    out["Throughput (matrices/s)"] = 1.0e6 / time_us
    out["GFLOPS"] = (gflops_factor * n * n * n) / (time_us * 1.0e3)

    metric_std = _metric_std_column(out, metric_time)
    if metric_std is not None:
        std_us = out[metric_std].astype(float)
        out["Throughput (matrices/s)_std"] = (1.0e6 * std_us) / (time_us * time_us)
        out["GFLOPS_std"] = out["GFLOPS"] * (std_us / time_us)

    return out


def _filter_by_scalar_type(df: pd.DataFrame, scalar_type: str, *, label: str) -> pd.DataFrame:
    """Filter a minibench CSV DataFrame to a specific scalar type.

    Note: minibench's `--type=float` intentionally matches any benchmark name
    containing the substring "float", which includes `std::complex<float>`.
    For plotting, we usually want to disambiguate.
    """
    if df.empty:
        return df
    if "name" not in df.columns:
        return df

    name = df["name"].astype(str)
    st = scalar_type.strip()

    if st in {"float", "double"}:
        # Match the real scalar, exclude complex specializations.
        mask = name.str.contains(f"<{st}") & ~name.str.contains("complex<")
    elif st in {"cfloat", "complex<float>"}:
        mask = name.str.contains("complex<float>")
    elif st in {"cdouble", "complex<double>"}:
        mask = name.str.contains("complex<double>")
    else:
        # Unknown token; leave unfiltered.
        return df

    out = df[mask].copy()
    if out.empty:
        # Help diagnose cases like --type=float but only complex results exist.
        raise ValueError(
            f"{label}: no rows matched scalar type '{scalar_type}'. "
            f"Available benchmark names (sample): {sorted(set(name.tolist()))[:5]}"
        )
    return out


def _require_unique_points(df: pd.DataFrame, *, label: str, keys: Sequence[str]) -> None:
    if df.empty:
        return
    dup = df.duplicated(subset=list(keys), keep=False)
    if dup.any():
        sample = df.loc[dup, list(keys) + ["name"]].head(10)
        raise ValueError(
            f"{label}: multiple rows map to the same plotted point for keys={list(keys)}. "
            "This typically means multiple scalar types are mixed (e.g. float + complex<float>) or the CSV contains extra argument variants. "
            f"Sample duplicates:\n{sample.to_string(index=False)}"
        )


def _compute_speedup(
    df_all: pd.DataFrame,
    *,
    base_impl: str,
    compare_impl: str,
    metric: str,
    metric_std: Optional[str],
    higher_is_better: bool,
    keys: Sequence[str],
) -> pd.DataFrame:
    key_fields = list(keys)
    base = df_all[df_all["impl"] == base_impl][key_fields + [metric]].copy()
    cmp_ = df_all[df_all["impl"] == compare_impl][key_fields + [metric]].copy()

    base = base.rename(columns={metric: "metric_base"})
    cmp_ = cmp_.rename(columns={metric: "metric_cmp"})

    if metric_std is not None:
        base_std = df_all[df_all["impl"] == base_impl][key_fields + [metric_std]].copy()
        cmp_std = df_all[df_all["impl"] == compare_impl][key_fields + [metric_std]].copy()
        base_std = base_std.rename(columns={metric_std: "std_base"})
        cmp_std = cmp_std.rename(columns={metric_std: "std_cmp"})
    else:
        base_std = None
        cmp_std = None

    merged = pd.merge(base, cmp_, on=key_fields, how="inner")
    if base_std is not None and cmp_std is not None:
        merged = pd.merge(merged, base_std, on=key_fields, how="left")
        merged = pd.merge(merged, cmp_std, on=key_fields, how="left")

    if merged.empty:
        raise ValueError("No overlapping points between providers for speedup")

    if higher_is_better:
        merged["speedup"] = merged["metric_cmp"] / merged["metric_base"]
    else:
        merged["speedup"] = merged["metric_base"] / merged["metric_cmp"]

    if "std_base" in merged.columns and "std_cmp" in merged.columns:
        rel_base = np.where(merged["metric_base"] != 0.0, merged["std_base"] / merged["metric_base"], 0.0)
        rel_cmp = np.where(merged["metric_cmp"] != 0.0, merged["std_cmp"] / merged["metric_cmp"], 0.0)
        merged["speedup_std"] = merged["speedup"] * np.sqrt(rel_base * rel_base + rel_cmp * rel_cmp)

    return merged


def plot_provider_compare_vs_n(
    df_base: pd.DataFrame,
    df_compare: pd.DataFrame,
    *,
    base_provider: str,
    compare_provider: str,
    base_label: str,
    compare_label: str,
    n_values: Sequence[int],
    dtype: str = "float",
    savepath: Optional[str] = None,
    y_metric: str = "gflops",
    gflops_factor: float = 9.0,
) -> None:
    metric_time = _pick_metric_column(df_base)
    metric_time_cmp = _pick_metric_column(df_compare)
    if metric_time != metric_time_cmp:
        df_compare = df_compare.rename(columns={metric_time_cmp: metric_time})

    _require_columns(df_base, ["arg0", "arg1", metric_time], label=base_provider)
    _require_columns(df_compare, ["arg0", "arg1", metric_time], label=compare_provider)

    df_base = _add_derived_metrics(df_base, metric_time=metric_time, gflops_factor=gflops_factor)
    df_compare = _add_derived_metrics(df_compare, metric_time=metric_time, gflops_factor=gflops_factor)

    df_base = df_base.copy()
    df_base["impl"] = base_label
    df_compare = df_compare.copy()
    df_compare["impl"] = compare_label

    # This script is intended for one effective point per N and provider.
    _require_unique_points(df_base, label=base_label, keys=["arg0"])
    _require_unique_points(df_compare, label=compare_label, keys=["arg0"])

    df_all = pd.concat([df_base, df_compare], ignore_index=True)

    if y_metric == "gflops":
        metric = "GFLOPS"
        higher_is_better = True
        logy = False
    elif y_metric == "throughput":
        metric = "Throughput (matrices/s)"
        higher_is_better = True
        logy = False
    else:
        metric = metric_time
        higher_is_better = False
        logy = True

    metric_std = _metric_std_column(df_all, metric)
    speedup = _compute_speedup(
        df_all,
        base_impl=base_label,
        compare_impl=compare_label,
        metric=metric,
        metric_std=metric_std,
        higher_is_better=higher_is_better,
        keys=["arg0"],
    )

    df_all = df_all[df_all["arg0"].isin(n_values)].copy()
    speedup = speedup[speedup["arg0"].isin(n_values)].copy()

    # Keep the global stylesheet identity (TeX fonts, palette, etc.) and only
    # compact the typography for this dense comparison figure.
    local_style = {
        "axes.titlesize": 18,
        "axes.labelsize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "legend.markerscale": 0.8,
    }

    with plt.rc_context(local_style):
        fig, (ax_top, ax_mid, ax_bottom) = plt.subplots(3, 1, sharex=True, figsize=(11.5, 9.5))

        plot_metric(
            df_all,
            metric,
            x_field="arg0",
            group_by="impl",
            metric_std=metric_std,
            label_fmt="{group}",
            xlabel="",
            ylabel=metric,
            title=None,
            logx=True,
            logx_base=2,
            logy=logy,
            set_xticks=False,
            show_errorbars=True,
            ax=ax_top,
        )

        # Tone down markers for dense x-grids.
        for line in ax_top.get_lines():
            line.set_markersize(4)
            line.set_linewidth(1.4)

        # Middle panel: Throughput (matrices/s) with log scale
        plot_metric(
            df_all,
            "Throughput (matrices/s)",
            x_field="arg0",
            group_by="impl",
            metric_std=_metric_std_column(df_all, "Throughput (matrices/s)"),
            label_fmt="{group}",
            xlabel="",
            ylabel="Throughput (matrices/s)",
            title=None,
            logx=True,
            logx_base=2,
            logy=True,
            set_xticks=False,
            show_errorbars=True,
            ax=ax_mid,
        )

        # Tone down markers for middle panel.
        for line in ax_mid.get_lines():
            line.set_markersize(4)
            line.set_linewidth(1.4)

        speedup = speedup.sort_values("arg0")
        ax_bottom.plot(
            speedup["arg0"],
            speedup["speedup"],
            marker="o",
            linestyle=":",
            label=f"{compare_label} / {base_label}",
            markersize=4,
            linewidth=1.4,
        )
        if "speedup_std" in speedup.columns:
            lower = np.maximum(speedup["speedup"] - speedup["speedup_std"], np.finfo(float).tiny)
            upper = np.maximum(speedup["speedup"] + speedup["speedup_std"], np.finfo(float).tiny)
            ax_bottom.fill_between(speedup["arg0"], lower, upper, alpha=0.12)
            ax_bottom.errorbar(
                speedup["arg0"],
                speedup["speedup"],
                yerr=speedup["speedup_std"],
                fmt="none",
                elinewidth=1.0,
                alpha=0.35,
                capsize=2.0,
            )

        ax_bottom.axhline(1.0, color="black", linewidth=1.0, linestyle="--")
        ax_bottom.set_xscale("log", base=2)
        ax_bottom.set_ylabel("Speedup")
        ax_bottom.set_xlabel("Matrix size N")
        ax_bottom.grid(True)
        ax_bottom.legend(loc="upper right")

        ticks = _readable_n_ticks(n_values)
        formatter = FuncFormatter(lambda v, pos: f"{int(v)}" if v >= 1 else f"{v:g}")
        ax_top.set_xticks(ticks)
        ax_mid.set_xticks(ticks)
        ax_bottom.set_xticks(ticks)
        ax_top.xaxis.set_major_formatter(formatter)
        ax_mid.xaxis.set_major_formatter(formatter)
        ax_bottom.xaxis.set_major_formatter(formatter)
        ax_bottom.tick_params(axis="x", labelrotation=30)

        dtype_label = _dtype_to_label(dtype)
        fig.suptitle(f"SYEV Performance Comparison ({dtype_label})", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.95])

    target_path = savepath or _default_plot_path()
    save_figure(fig, target_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run and plot SYEV provider comparison")
    parser.add_argument("--run", action="store_true", help="run benchmarks before plotting")
    parser.add_argument("--bench-bin", default=_default_bench_path(), help="path to syev_benchmark binary")
    parser.add_argument("--csv-base", default=None, help="CSV output path for base provider (defaults based on --type)")
    parser.add_argument("--csv-compare", default=None, help="CSV output path for compared provider (defaults based on --type)")
    parser.add_argument("--output", default=None, help="optional path to save the plot (defaults based on --type)")

    parser.add_argument("--provider-base", default="VENDOR", help="baseline BATCHLAS_SYEV_PROVIDER")
    parser.add_argument("--provider-compare", default="CTA", help="compared BATCHLAS_SYEV_PROVIDER")
    parser.add_argument("--label-base", default="cuSOLVER", help="display label for baseline series")
    parser.add_argument("--label-compare", default="BatchLAS", help="display label for compared series")

    parser.add_argument("--backend", default="CUDA", help="minibench backend filter")
    parser.add_argument("--type", dest="dtype", default="float", help="minibench type filter")
    parser.add_argument("--warmup", type=int, default=10, help="benchmark warmup iterations")
    parser.add_argument("--no-metric-stddev", action="store_true", help="do not request metric stddev columns")
    parser.add_argument(
        "--y-metric",
        choices=["gflops", "throughput", "time"],
        default="gflops",
        help="plot effective gflops, throughput (matrices/s), or time",
    )
    parser.add_argument(
        "--gflops-factor",
        type=float,
        default=9.0,
        help="operation model factor in effective GFLOPS ~= factor * N^3 / time",
    )

    parser.add_argument("--n", type=int, nargs="+", default=_default_n_values(), help="matrix sizes N to compare")
    parser.add_argument(
        "--batches",
        type=int,
        nargs="+",
        default=None,
        help="optional explicit batch sizes (must match --n length); default uses inverse-linear scaling with N anchored at N=4 -> 65536",
    )

    parser.add_argument(
        "--bench-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="extra args forwarded to benchmark runs (prefix with --)",
    )

    args = parser.parse_args()

    # Apply dtype-aware defaults for file paths
    if args.csv_base is None:
        args.csv_base = _default_csv_path(args.dtype)
    if args.csv_compare is None:
        args.csv_compare = _default_csv_compare_path(args.dtype)
    if args.output is None:
        args.output = _default_plot_path(args.dtype)

    if args.batches is None:
        args.batches = _default_batches_for_n(args.n, batch_at_n4=65536)
    if len(args.batches) != len(args.n):
        raise ValueError("--batches must have the same number of entries as --n")
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0")

    if args.run:
        common_args = [f"--backend={args.backend}", f"--type={args.dtype}"]
        common_args.append(f"--warmup={args.warmup}")
        if not args.no_metric_stddev:
            common_args.append("--metric_stddev")

        _run_benchmark_pairs(
            binary=args.bench_bin,
            csv_out=args.csv_base,
            n_values=args.n,
            batches=args.batches,
            common_args=common_args,
            bench_args=args.bench_args,
            env={"BATCHLAS_SYEV_PROVIDER": args.provider_base},
        )
        _run_benchmark_pairs(
            binary=args.bench_bin,
            csv_out=args.csv_compare,
            n_values=args.n,
            batches=args.batches,
            common_args=common_args,
            bench_args=args.bench_args,
            env={"BATCHLAS_SYEV_PROVIDER": args.provider_compare},
        )

    if not os.path.isfile(args.csv_base):
        raise FileNotFoundError(f"CSV not found: {args.csv_base}")
    if not os.path.isfile(args.csv_compare):
        raise FileNotFoundError(f"CSV not found: {args.csv_compare}")

    df_base = load_results(args.csv_base)
    df_compare = load_results(args.csv_compare)

    # Disambiguate real vs complex results for plotting.
    df_base = _filter_by_scalar_type(df_base, args.dtype, label=args.provider_base)
    df_compare = _filter_by_scalar_type(df_compare, args.dtype, label=args.provider_compare)

    plot_provider_compare_vs_n(
        df_base,
        df_compare,
        base_provider=args.provider_base,
        compare_provider=args.provider_compare,
        base_label=args.label_base,
        compare_label=args.label_compare,
        n_values=args.n,
        dtype=args.dtype,
        savepath=args.output,
        y_metric=args.y_metric,
        gflops_factor=args.gflops_factor,
    )


if __name__ == "__main__":
    main()
