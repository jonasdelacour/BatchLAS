from __future__ import annotations

import argparse
import os
from typing import Iterable, Optional, Sequence

import pandas as pd

from bench_common import load_results, plot_metric, run_benchmark, save_figure


def _here() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _default_bench_path() -> str:
    return os.path.join(_here(), "..", "build", "benchmarks", "syev_benchmark")


def _default_bench_blocked_path() -> str:
    return os.path.join(_here(), "..", "build", "benchmarks", "syev_blocked_benchmark")


def _default_csv_path() -> str:
    return os.path.join(_here(), "..", "build", "syev_out.csv")


def _default_csv_blocked_path() -> str:
    return os.path.join(_here(), "..", "build", "syev_blocked_out.csv")


def _default_plot_path(kind: str, dtype: str, batch: int) -> str:
    return os.path.join(_here(), "..", "output", "plots", f"syev_blocked_{kind}_by_n_batch{batch}_{dtype}.png")


def _as_csv_arg(values: Iterable[int]) -> str:
    return ",".join(str(v) for v in values)


def _parse_csv_ints(s: str) -> list[int]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    out: list[int] = []
    for p in parts:
        try:
            out.append(int(p))
        except ValueError as e:
            raise ValueError(f"Invalid integer token '{p}' in '{s}'") from e
    return out


def _unique_sorted_ints(values: Iterable[int]) -> list[int]:
    return sorted(set(int(v) for v in values))


def _resolve_output_template(template: Optional[str], *, dtype: str) -> Optional[str]:
    if template is None:
        return None
    # Allow a simple format string like "..._{dtype}.png".
    try:
        return template.format(dtype=dtype)
    except Exception as e:
        raise ValueError(f"Invalid output template '{template}'. Use '{{dtype}}' for substitution.") from e


def _pick_time_metric_column(df: pd.DataFrame) -> str:
    for name in ("Time (µs) / Batch", "T(µs)/Batch"):
        if name in df.columns:
            return name
    raise ValueError(f"No known time-per-batch metric column found. Available: {list(df.columns)}")


def _add_throughput_columns(df: pd.DataFrame, *, batch_field: str = "arg1") -> pd.DataFrame:
    """Add throughput columns to a minibench result DataFrame.

    Throughput is computed as matrices per second:
        throughput = batch / (avg_ms / 1000) = batch * 1000 / avg_ms

    Error is propagated from stddev_ms (1-sigma):
        sigma = batch * 1000 * stddev_ms / avg_ms^2
    """
    needed = {batch_field, "avg_ms", "stddev_ms"}
    missing = sorted(needed.difference(df.columns))
    if missing:
        raise ValueError(f"Missing columns for throughput: {missing}. Available: {list(df.columns)}")

    out = df.copy()
    batch = out[batch_field].astype(float)
    avg_ms = out["avg_ms"].astype(float)
    std_ms = out["stddev_ms"].astype(float)
    k = batch * 1000.0
    out["Throughput (matrices/s)"] = k / avg_ms
    out["Throughput (matrices/s)_std"] = (k * std_ms) / (avg_ms * avg_ms)
    return out


def _filter_by_scalar_type(df: pd.DataFrame, scalar_type: str, *, label: str) -> pd.DataFrame:
    if df.empty:
        return df
    if "name" not in df.columns:
        return df

    name = df["name"].astype(str)
    st = scalar_type.strip()

    if st in {"float", "double"}:
        mask = name.str.contains(f"<{st}") & ~name.str.contains("complex<")
    elif st in {"cfloat", "complex<float>"}:
        mask = name.str.contains("complex<float>")
    elif st in {"cdouble", "complex<double>"}:
        mask = name.str.contains("complex<double>")
    else:
        return df

    out = df[mask].copy()
    if out.empty:
        raise ValueError(f"{label}: no rows matched scalar type '{scalar_type}'.")
    return out


def _filter_fixed_batch(df: pd.DataFrame, *, batch: int, label: str) -> pd.DataFrame:
    if "arg1" not in df.columns:
        raise ValueError(f"{label}: expected column arg1 (batch size). Available: {list(df.columns)}")
    out = df[df["arg1"] == batch].copy()
    if out.empty:
        raise ValueError(f"{label}: no rows found for batch={batch}.")
    return out


def _select_blocked_best_blocks(df_blocked: pd.DataFrame, time_metric: str) -> pd.DataFrame:
    # syev_blocked args:
    # arg0=n, arg1=batch, arg2=jobz, arg3=sytrd_block_size, arg4=ormqr_block_size
    needed = {"arg0", "arg1", "arg2", "arg3", "arg4", time_metric}
    missing = sorted(needed.difference(df_blocked.columns))
    if missing:
        raise ValueError(f"syev_blocked: missing columns: {missing}. Available: {list(df_blocked.columns)}")

    idx = df_blocked.groupby(["arg0", "arg1", "arg2"])[time_metric].idxmin()
    return df_blocked.loc[idx].copy()


def _compute_speedup(
    df_baseline: pd.DataFrame,
    df_impl: pd.DataFrame,
    *,
    time_metric: str,
    label: str,
) -> pd.DataFrame:
    """Compute speedup = baseline_time / impl_time on matching (N,batch).

    If stddev columns exist (metric_std), propagate 1-sigma.
    """
    for col in ("arg0", "arg1", time_metric):
        if col not in df_baseline.columns:
            raise ValueError(f"baseline missing '{col}'. Available: {list(df_baseline.columns)}")
        if col not in df_impl.columns:
            raise ValueError(f"{label} missing '{col}'. Available: {list(df_impl.columns)}")

    base = df_baseline[["arg0", "arg1", time_metric]].copy()
    impl = df_impl[["arg0", "arg1", time_metric]].copy()

    base = base.rename(columns={time_metric: "baseline_time"})
    impl = impl.rename(columns={time_metric: "impl_time"})

    if f"{time_metric}_std" in df_baseline.columns:
        base[f"baseline_time_std"] = df_baseline[f"{time_metric}_std"].astype(float)
    if f"{time_metric}_std" in df_impl.columns:
        impl[f"impl_time_std"] = df_impl[f"{time_metric}_std"].astype(float)

    merged = base.merge(impl, on=["arg0", "arg1"], how="inner")
    if merged.empty:
        raise ValueError("No overlapping points to compute speedup. Check N/batch filters.")

    merged["Speedup (x)"] = merged["baseline_time"].astype(float) / merged["impl_time"].astype(float)

    # Error propagation for a/b.
    if "baseline_time_std" in merged.columns and "impl_time_std" in merged.columns:
        a = merged["baseline_time"].astype(float)
        b = merged["impl_time"].astype(float)
        sa = merged["baseline_time_std"].astype(float)
        sb = merged["impl_time_std"].astype(float)
        merged["Speedup (x)_std"] = merged["Speedup (x)"] * ((sa / a) ** 2 + (sb / b) ** 2) ** 0.5

    merged["impl"] = label
    return merged


def plot_time_vs_n_fixed_batch(
    df_syev: pd.DataFrame,
    df_syev_blocked: pd.DataFrame,
    *,
    batch: int,
    n_values: Sequence[int],
    dtype: str,
    jobz: int,
    sytrd_block_size: Optional[int],
    ormqr_block_size: Optional[int],
    best_blocks: bool,
    savepath: Optional[str] = None,
    y_metric: str = "throughput",
) -> None:
    time_metric = _pick_time_metric_column(df_syev)
    time_metric_blocked = _pick_time_metric_column(df_syev_blocked)
    if time_metric != time_metric_blocked:
        df_syev_blocked = df_syev_blocked.rename(columns={time_metric_blocked: time_metric})

    df_syev = _add_throughput_columns(df_syev)
    df_syev_blocked = _add_throughput_columns(df_syev_blocked)

    df_syev = _filter_fixed_batch(df_syev, batch=batch, label="syev")
    df_blk = _filter_fixed_batch(df_syev_blocked, batch=batch, label="syev_blocked")

    # Filter blocked to the desired jobz.
    if "arg2" not in df_blk.columns:
        raise ValueError(f"syev_blocked: expected arg2 (jobz). Available: {list(df_blk.columns)}")
    df_blk = df_blk[df_blk["arg2"] == jobz].copy()
    if df_blk.empty:
        raise ValueError("syev_blocked: no rows after filtering for jobz")

    if best_blocks:
        df_blk = _select_blocked_best_blocks(df_blk, time_metric)
        blocked_label = "syev_blocked(best blocks)"
    else:
        if sytrd_block_size is None or ormqr_block_size is None:
            raise ValueError("Provide both --sytrd-block-size and --ormqr-block-size when not using --best-blocks")
        df_blk = df_blk[(df_blk["arg3"] == sytrd_block_size) & (df_blk["arg4"] == ormqr_block_size)].copy()
        if df_blk.empty:
            raise ValueError("syev_blocked: no rows after filtering for sytrd/ormqr block sizes")
        blocked_label = f"syev_blocked(sytrd={sytrd_block_size}, ormqr={ormqr_block_size})"

    # Filter to requested N list.
    df_syev = df_syev[df_syev["arg0"].isin(n_values)].copy()
    df_blk = df_blk[df_blk["arg0"].isin(n_values)].copy()

    missing_base = sorted(set(n_values).difference(set(df_syev["arg0"].tolist())))
    missing_blk = sorted(set(n_values).difference(set(df_blk["arg0"].tolist())))
    if missing_base:
        raise ValueError(f"syev: missing N values for batch={batch}: {missing_base}")
    if missing_blk:
        raise ValueError(f"syev_blocked: missing N values for batch={batch}: {missing_blk}")

    df_syev["impl"] = "syev (baseline)"
    df_blk["impl"] = blocked_label
    df = pd.concat([df_syev, df_blk], ignore_index=True)

    if y_metric == "throughput":
        y_col = "Throughput (matrices/s)"
        y_col_std = f"{y_col}_std"
        ylabel = y_col
        logy = True
        title = f"SYEV_BLOCKED vs SYEV throughput (batch={batch}, type={dtype})"
    else:
        y_col = time_metric
        y_col_std = f"{y_col}_std"
        ylabel = y_col
        logy = False
        title = f"SYEV_BLOCKED vs SYEV time/batch (batch={batch}, type={dtype})"

    fig, _ = plot_metric(
        df,
        y_col,
        x_field="arg0",
        group_by="impl",
        metric_std=y_col_std,
        label_fmt="{group}",
        xlabel="Matrix Size (N)",
        ylabel=ylabel,
        title=title,
        logy=logy,
        logx=False,
        show_errorbars=True,
    )

    target_path = savepath or _default_plot_path("compare", dtype, batch)
    save_figure(fig, target_path)


def plot_speedup_vs_n_fixed_batch(
    df_syev: pd.DataFrame,
    df_syev_blocked: pd.DataFrame,
    *,
    batch: int,
    n_values: Sequence[int],
    dtype: str,
    jobz: int,
    sytrd_block_size: Optional[int],
    ormqr_block_size: Optional[int],
    best_blocks: bool,
    savepath: Optional[str] = None,
) -> None:
    time_metric = _pick_time_metric_column(df_syev)
    time_metric_blocked = _pick_time_metric_column(df_syev_blocked)
    if time_metric != time_metric_blocked:
        df_syev_blocked = df_syev_blocked.rename(columns={time_metric_blocked: time_metric})

    df_syev = _filter_fixed_batch(df_syev, batch=batch, label="syev")
    df_blk = _filter_fixed_batch(df_syev_blocked, batch=batch, label="syev_blocked")

    df_syev = df_syev[df_syev["arg0"].isin(n_values)].copy()
    df_blk = df_blk[df_blk["arg0"].isin(n_values)].copy()

    if "arg2" not in df_blk.columns:
        raise ValueError(f"syev_blocked: expected arg2 (jobz). Available: {list(df_blk.columns)}")
    df_blk = df_blk[df_blk["arg2"] == jobz].copy()
    if df_blk.empty:
        raise ValueError("syev_blocked: no rows after filtering for jobz")

    if best_blocks:
        df_blk = _select_blocked_best_blocks(df_blk, time_metric)
        blocked_label = "syev_blocked(best blocks)"
    else:
        if sytrd_block_size is None or ormqr_block_size is None:
            raise ValueError("Provide both --sytrd-block-size and --ormqr-block-size when not using --best-blocks")
        df_blk = df_blk[(df_blk["arg3"] == sytrd_block_size) & (df_blk["arg4"] == ormqr_block_size)].copy()
        if df_blk.empty:
            raise ValueError("syev_blocked: no rows after filtering for sytrd/ormqr block sizes")
        blocked_label = f"syev_blocked(sytrd={sytrd_block_size}, ormqr={ormqr_block_size})"

    speed = _compute_speedup(df_syev, df_blk, time_metric=time_metric, label=blocked_label)

    fig, _ = plot_metric(
        speed,
        "Speedup (x)",
        x_field="arg0",
        group_by="impl",
        metric_std="Speedup (x)_std" if "Speedup (x)_std" in speed.columns else None,
        label_fmt="{group}",
        xlabel="Matrix Size (N)",
        ylabel="Speedup vs syev (x)",
        title=f"SYEV_BLOCKED speedup vs SYEV baseline (batch={batch}, type={dtype})",
        logy=False,
        logx=False,
        show_errorbars=True,
    )

    target_path = savepath or _default_plot_path("speedup", dtype, batch)
    save_figure(fig, target_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare SYEV_BLOCKED vs SYEV at fixed batch size and plot speedup vs baseline (syev CUDA)"
    )
    parser.add_argument("--run", action="store_true", help="run benchmarks before plotting")
    parser.add_argument("--bench-bin", default=_default_bench_path(), help="path to syev_benchmark binary")
    parser.add_argument(
        "--bench-bin-blocked",
        default=_default_bench_blocked_path(),
        help="path to syev_blocked_benchmark binary",
    )
    parser.add_argument("--csv", default=_default_csv_path(), help="CSV output path for syev")
    parser.add_argument("--csv-blocked", default=_default_csv_blocked_path(), help="CSV output path for syev_blocked")

    parser.add_argument("--backend", default="CUDA", help="minibench backend filter")
    parser.add_argument("--no-metric-stddev", action="store_true", help="do not request metric stddev columns")

    parser.add_argument("--batch", type=int, default=16384, help="fixed batch size")

    n_group = parser.add_mutually_exclusive_group()
    n_group.add_argument(
        "--n-values",
        default=None,
        help="explicit comma-separated N values (overrides --n-min/--n-max/--n-step), e.g. '32,64,96,128'",
    )
    n_group.add_argument(
        "--n",
        type=int,
        nargs="+",
        default=None,
        help="explicit N values as a list (overrides --n-min/--n-max/--n-step), e.g. --n 32 64 128",
    )
    parser.add_argument("--n-min", type=int, default=1, help="min N (inclusive) for N sweep")
    parser.add_argument("--n-max", type=int, default=32, help="max N (inclusive) for N sweep")
    parser.add_argument("--n-step", type=int, default=1, help="stride for N sweep within [n-min,n-max]")

    parser.add_argument(
        "--types",
        nargs="+",
        default=["float", "cfloat"],
        choices=["float", "double", "cfloat", "cdouble"],
        help="scalar types to benchmark/plot (default: float cfloat)",
    )

    parser.add_argument("--jobz", type=int, choices=[0, 1], default=1, help="blocked jobz: 0=no vecs, 1=vecs")

    blk_group = parser.add_mutually_exclusive_group()
    blk_group.add_argument(
        "--best-blocks",
        action="store_true",
        help="pick best (sytrd_block_size, ormqr_block_size) per N from the CSV (requires multiple block sizes)",
    )
    blk_group.add_argument(
        "--sytrd-block-size",
        type=int,
        default=None,
        help="sytrd_block_size to filter to (requires --ormqr-block-size and not using --best-blocks)",
    )
    parser.add_argument(
        "--ormqr-block-size",
        type=int,
        default=None,
        help="ormqr_block_size to filter to (requires --sytrd-block-size and not using --best-blocks)",
    )

    parser.add_argument(
        "--sytrd-block-sizes",
        default="16,32,64",
        help="comma-separated sytrd_block_size sweep when using --run",
    )
    parser.add_argument(
        "--ormqr-block-sizes",
        default="1,2,4,8,16,32,64,128",
        help="comma-separated ormqr_block_size sweep when using --run",
    )

    parser.add_argument(
        "--output-float",
        default=None,
        help="optional output path for float compare plot",
    )
    parser.add_argument(
        "--output-cfloat",
        default=None,
        help="optional output path for complex<float> compare plot",
    )
    parser.add_argument(
        "--output-speedup-float",
        default=None,
        help="optional output path for float speedup plot",
    )
    parser.add_argument(
        "--output-speedup-cfloat",
        default=None,
        help="optional output path for complex<float> speedup plot",
    )

    parser.add_argument(
        "--output",
        default=None,
        help="optional compare-plot output path template (supports '{dtype}'), e.g. '.../compare_{dtype}.png'",
    )
    parser.add_argument(
        "--output-speedup",
        default=None,
        help="optional speedup-plot output path template (supports '{dtype}'), e.g. '.../speedup_{dtype}.png'",
    )

    parser.add_argument(
        "--y-metric",
        choices=["throughput", "time"],
        default="throughput",
        help="y-axis metric for compare plot: throughput (matrices/s) or time (µs)/batch",
    )

    parser.add_argument(
        "--bench-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="extra args forwarded to both benchmarks (prefix with --)",
    )

    args = parser.parse_args()

    # Decide blocked block-size selection strategy early, because --run needs it.
    if args.best_blocks:
        best_blocks = True
    elif args.sytrd_block_size is not None or args.ormqr_block_size is not None:
        if args.sytrd_block_size is None or args.ormqr_block_size is None:
            raise ValueError(
                "When providing fixed block sizes, both --sytrd-block-size and --ormqr-block-size are required"
            )
        best_blocks = False
    else:
        # Default behavior: pick best blocks from the CSV sweep.
        best_blocks = True

    if args.n_values is not None:
        n_values = _unique_sorted_ints(_parse_csv_ints(args.n_values))
    elif args.n is not None:
        n_values = _unique_sorted_ints(args.n)
    else:
        if args.n_step <= 0:
            raise ValueError("--n-step must be positive")
        n_values = list(range(args.n_min, args.n_max + 1, args.n_step))

    if not n_values:
        raise ValueError("No N values selected")
    n_arg = _as_csv_arg(n_values)
    b_arg = str(args.batch)

    common_args = [f"--backend={args.backend}"]
    if not args.no_metric_stddev:
        common_args.append("--metric_stddev")

    if args.run:
        # When the user provided fixed block sizes, run only those.
        # Otherwise (default / best-blocks mode), sweep the configured grids.
        if best_blocks:
            run_sytrd_arg = args.sytrd_block_sizes
            run_ormqr_arg = args.ormqr_block_sizes
        else:
            run_sytrd_arg = str(args.sytrd_block_size)
            run_ormqr_arg = str(args.ormqr_block_size)

        for dtype in args.types:
            type_arg = f"--type={dtype}"
            run_benchmark(args.bench_bin, args.csv, [*common_args, type_arg, n_arg, b_arg, *args.bench_args])

            jobz_arg = str(args.jobz)
            run_benchmark(
                args.bench_bin_blocked,
                args.csv_blocked,
                [
                    *common_args,
                    type_arg,
                    n_arg,
                    b_arg,
                    jobz_arg,
                    run_sytrd_arg,
                    run_ormqr_arg,
                    *args.bench_args,
                ],
            )

    if not os.path.isfile(args.csv):
        raise FileNotFoundError(f"CSV not found: {args.csv}")
    if not os.path.isfile(args.csv_blocked):
        raise FileNotFoundError(f"CSV not found: {args.csv_blocked}")

    df_syev_all = load_results(args.csv)
    df_blk_all = load_results(args.csv_blocked)

    for dtype in args.types:
        df_syev = _filter_by_scalar_type(df_syev_all, dtype, label="syev")
        df_blk = _filter_by_scalar_type(df_blk_all, dtype, label="syev_blocked")

        # Back-compat outputs for the common float/cfloat case.
        compare_out = _resolve_output_template(args.output, dtype=dtype)
        speedup_out = _resolve_output_template(args.output_speedup, dtype=dtype)
        if dtype == "float" and args.output_float is not None:
            compare_out = args.output_float
        if dtype == "cfloat" and args.output_cfloat is not None:
            compare_out = args.output_cfloat
        if dtype == "float" and args.output_speedup_float is not None:
            speedup_out = args.output_speedup_float
        if dtype == "cfloat" and args.output_speedup_cfloat is not None:
            speedup_out = args.output_speedup_cfloat

        plot_time_vs_n_fixed_batch(
            df_syev,
            df_blk,
            batch=args.batch,
            n_values=n_values,
            dtype=dtype,
            jobz=args.jobz,
            sytrd_block_size=args.sytrd_block_size,
            ormqr_block_size=args.ormqr_block_size,
            best_blocks=best_blocks,
            savepath=compare_out,
            y_metric=args.y_metric,
        )
        plot_speedup_vs_n_fixed_batch(
            df_syev,
            df_blk,
            batch=args.batch,
            n_values=n_values,
            dtype=dtype,
            jobz=args.jobz,
            sytrd_block_size=args.sytrd_block_size,
            ormqr_block_size=args.ormqr_block_size,
            best_blocks=best_blocks,
            savepath=speedup_out,
        )


if __name__ == "__main__":
    main()
