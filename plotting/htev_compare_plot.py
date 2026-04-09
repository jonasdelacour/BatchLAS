"""htev_compare_plot.py – Run and plot BatchLAS STEQR / STEDC vs cuSolverDx HTEV.

Usage examples
--------------
# Just plot from an existing CSV:
python plotting/htev_compare_plot.py --csv build/htev_compare.csv

# Run the benchmark first, then plot:
python plotting/htev_compare_plot.py --run --csv build/htev_compare.csv

# Select a different dtype and matrix sizes:
python plotting/htev_compare_plot.py --run --type double --n 8 12 16 24 32 48 64
"""
from __future__ import annotations

import argparse
import os
import tempfile
from typing import Iterable, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from bench_common import load_results, plot_metric, run_benchmark, save_figure, with_device_title


# ── Helpers ───────────────────────────────────────────────────────────────────

def _here() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _default_bench_path() -> str:
    return os.path.join(_here(), "..", "build", "benchmarks", "htev_compare_benchmark")


def _default_csv_path() -> str:
    return os.path.join(_here(), "..", "build", "htev_compare.csv")


def _default_plot_path(dtype: str = "float") -> str:
    suffix = dtype.lower().replace(" ", "").replace("<", "_").replace(">", "")
    return os.path.join(_here(), "..", "output", "plots", f"htev_compare_{suffix}.png")


def _default_n_values() -> list[int]:
    return list(range(8, 65))


def _default_batches_for_n(n_values: Sequence[int], *, batch_at_n8: int = 8192) -> list[int]:
    target = float(batch_at_n8) * 8.0
    batches: list[int] = []
    for n in n_values:
        b = int(round(target / float(n)))
        batches.append(max(1, b))
    return batches


def _default_batches() -> list[int]:
    return [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]


def _as_csv_arg(values: Iterable[int]) -> str:
    return ",".join(str(v) for v in values)


def _run_benchmark_pairs(
    *,
    binary: str,
    csv_out: str,
    n_values: Sequence[int],
    batches: Sequence[int],
    common_args: Sequence[str],
    bench_args: Sequence[str],
) -> None:
    if len(n_values) != len(batches):
        raise ValueError("n_values and batches must have identical lengths")

    frames: list[pd.DataFrame] = []
    tmp_paths: list[str] = []

    try:
        for n, b in zip(n_values, batches):
            with tempfile.NamedTemporaryFile(prefix="htev_point_", suffix=".csv", delete=False) as tmp:
                tmp_path = tmp.name
            tmp_paths.append(tmp_path)

            run_benchmark(binary, tmp_path, [*common_args, str(n), str(b), *bench_args])
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
    nset = set(int(v) for v in n_values)
    ticks: list[int] = []
    for n in sorted(nset):
        if n <= 32:
            if n % 4 == 0:
                ticks.append(n)
        else:
            if n % 8 == 0:
                ticks.append(n)
    for anchor in (8, 16, 32, 48, 64):
        if anchor in nset and anchor not in ticks:
            ticks.append(anchor)
    return sorted(set(ticks))


def _require_columns(df: pd.DataFrame, cols: Sequence[str], *, label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"{label}: missing columns {missing}. Available: {list(df.columns)}"
        )


def _filter_by_scalar_type(df: pd.DataFrame, scalar_type: str) -> pd.DataFrame:
    """Keep only rows whose benchmark name matches the requested scalar type."""
    if "name" not in df.columns or df.empty:
        return df
    name = df["name"].astype(str)
    st = scalar_type.strip().lower()
    if st == "float":
        mask = name.str.contains("<float") & ~name.str.contains("complex<")
    elif st == "double":
        mask = name.str.contains("<double") & ~name.str.contains("complex<")
    elif st in {"cfloat", "complex<float>"}:
        mask = name.str.contains("complex<float>")
    elif st in {"cdouble", "complex<double>"}:
        mask = name.str.contains("complex<double>")
    else:
        return df
    return df[mask].copy()


def _pick_metric_column(df: pd.DataFrame) -> str:
    for name in ("Time (µs) / matrix", "T(µs)/matrix"):
        if name in df.columns:
            return name
    raise ValueError(f"No known time metric column found. Available: {list(df.columns)}")


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


def _impl_label(name: str) -> str:
    """Map benchmark function name (from the CSV 'name' column) to a short label."""
    n = name.lower()
    if "steqr" in n:
        return "STEQR"
    if "stedc" in n:
        return "STEDC"
    if "htev_dx" in n or "htev_cusolverdx" in n or "cusolverdx_htev" in n:
        return "cuSolverDx HTEV"
    return name


def _require_unique_points(df: pd.DataFrame, *, label: str, keys: Sequence[str]) -> None:
    if df.empty:
        return
    dup = df.duplicated(subset=list(keys), keep=False)
    if dup.any():
        sample = df.loc[dup, list(keys) + ["name"]].head(10)
        raise ValueError(
            f"{label}: multiple rows map to the same plotted point for keys={list(keys)}. "
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
        raise ValueError(f"No overlapping points between {base_impl} and {compare_impl} for speedup")

    if higher_is_better:
        merged["speedup"] = merged["metric_cmp"] / merged["metric_base"]
    else:
        merged["speedup"] = merged["metric_base"] / merged["metric_cmp"]

    if "std_base" in merged.columns and "std_cmp" in merged.columns:
        rel_base = np.where(merged["metric_base"] != 0.0, merged["std_base"] / merged["metric_base"], 0.0)
        rel_cmp = np.where(merged["metric_cmp"] != 0.0, merged["std_cmp"] / merged["metric_cmp"], 0.0)
        merged["speedup_std"] = merged["speedup"] * np.sqrt(rel_base * rel_base + rel_cmp * rel_cmp)

    return merged


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_htev_compare(
    df: pd.DataFrame,
    *,
    n_values: Sequence[int],
    savepath: Optional[str] = None,
    dtype: str = "float",
    y_metric: str = "gflops",
    gflops_factor: float = 9.0,
    base_label: str = "cuSolverDx HTEV",
    compare_labels: Sequence[str] = ("STEQR", "STEDC"),
) -> None:
    metric_time = _pick_metric_column(df)
    _require_columns(df, ["arg0", "arg1", metric_time], label="htev_compare")

    df = _add_derived_metrics(df, metric_time=metric_time, gflops_factor=gflops_factor)

    if "name" in df.columns:
        df = df.copy()
        df["impl"] = df["name"].apply(_impl_label)
    else:
        df = df.copy()
        df["impl"] = "unknown"

    df = df[df["arg0"].isin(n_values)].copy()
    if df.empty:
        raise ValueError("No rows remain after filtering to requested --n values")

    present_impls = set(df["impl"].tolist())
    missing = [name for name in [base_label, *compare_labels] if name not in present_impls]
    if missing:
        raise ValueError(
            f"Missing implementation rows in CSV: {missing}. "
            f"Present implementations: {sorted(present_impls)}"
        )

    # One point per (N, impl) for this compare figure.
    for impl in set(df["impl"].tolist()):
        _require_unique_points(df[df["impl"] == impl], label=impl, keys=["arg0"])

    if y_metric == "gflops":
        metric = "GFLOPS"
        higher_is_better = True
        logy_top = False
    elif y_metric == "throughput":
        metric = "Throughput (matrices/s)"
        higher_is_better = True
        logy_top = False
    else:
        metric = metric_time
        higher_is_better = False
        logy_top = True

    metric_std = _metric_std_column(df, metric)

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
            df,
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
            logy=logy_top,
            set_xticks=False,
            show_errorbars=True,
            ax=ax_top,
        )

        for line in ax_top.get_lines():
            line.set_markersize(4)
            line.set_linewidth(1.4)

        plot_metric(
            df,
            "Throughput (matrices/s)",
            x_field="arg0",
            group_by="impl",
            metric_std=_metric_std_column(df, "Throughput (matrices/s)"),
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

        for line in ax_mid.get_lines():
            line.set_markersize(4)
            line.set_linewidth(1.4)

        for cmp_label in compare_labels:
            speedup = _compute_speedup(
                df,
                base_impl=base_label,
                compare_impl=cmp_label,
                metric=metric,
                metric_std=metric_std,
                higher_is_better=higher_is_better,
                keys=["arg0"],
            ).sort_values("arg0")

            ax_bottom.plot(
                speedup["arg0"],
                speedup["speedup"],
                marker="o",
                linestyle=":",
                label=f"{cmp_label} / {base_label}",
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

    dtype_label = dtype.replace("complex<float>", "cfloat").replace("complex<double>", "cdouble")
    fig.suptitle(with_device_title(f"HTEV Performance Comparison ({dtype_label})"), fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    target = savepath or _default_plot_path(dtype)
    save_figure(fig, target)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run and plot STEQR / STEDC vs cuSolverDx HTEV"
    )
    parser.add_argument(
        "--run", action="store_true", help="run the benchmark before plotting"
    )
    parser.add_argument(
        "--bench-bin",
        default=_default_bench_path(),
        help="path to htev_compare_benchmark binary",
    )
    parser.add_argument("--csv", default=_default_csv_path(), help="CSV output path")
    parser.add_argument("--output", default=None, help="path to save the plot")
    parser.add_argument(
        "--backend", default="CUDA", help="minibench backend filter (default: CUDA)"
    )
    parser.add_argument(
        "--type",
        dest="dtype",
        default="float",
        help="scalar type filter: float, double, cfloat, cdouble",
    )
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
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="number of warmup iterations (default: 5)",
    )
    parser.add_argument(
        "--n",
        type=int,
        nargs="+",
        default=_default_n_values(),
        help="matrix sizes N to compare",
    )
    parser.add_argument(
        "--batches",
        type=int,
        nargs="+",
        default=None,
        help="optional explicit batch sizes (must match --n length); default uses inverse-linear scaling with N anchored at N=8 -> 8192",
    )
    parser.add_argument(
        "--no-metric-stddev",
        action="store_true",
        help="do not request metric stddev columns",
    )
    args = parser.parse_args()

    if args.batches is None:
        args.batches = _default_batches_for_n(args.n, batch_at_n8=8192)
    if len(args.batches) != len(args.n):
        raise ValueError("--batches must have the same number of entries as --n")
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0")

    if args.run:
        common_args = [
            f"--backend={args.backend}",
            f"--type={args.dtype}",
            f"--warmup={args.warmup}",
        ]
        bench_args: list[str] = []
        if not args.no_metric_stddev:
            common_args.append("--metric_stddev")

        _run_benchmark_pairs(
            binary=args.bench_bin,
            csv_out=args.csv,
            n_values=args.n,
            batches=args.batches,
            common_args=common_args,
            bench_args=bench_args,
        )

    if not os.path.isfile(args.csv):
        raise FileNotFoundError(
            f"CSV not found: {args.csv}. Run with --run to generate it."
        )

    df = load_results(args.csv)
    df = _filter_by_scalar_type(df, args.dtype)
    if df.empty:
        raise ValueError(
            f"No rows matched scalar type '{args.dtype}' in {args.csv}."
        )

    plot_htev_compare(
        df,
        n_values=args.n,
        savepath=args.output,
        dtype=args.dtype,
        y_metric=args.y_metric,
        gflops_factor=args.gflops_factor,
    )


if __name__ == "__main__":
    main()
