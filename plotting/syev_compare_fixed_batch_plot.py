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


def _default_bench_cta_path() -> str:
    return os.path.join(_here(), "..", "build", "benchmarks", "syev_cta_benchmark")


def _default_csv_path() -> str:
    return os.path.join(_here(), "..", "build", "syev_out.csv")


def _default_csv_cta_path() -> str:
    return os.path.join(_here(), "..", "build", "syev_cta_out.csv")


def _default_plot_path(dtype: str, batch: int) -> str:
    return os.path.join(_here(), "..", "output", "plots", f"syev_compare_by_n_batch{batch}_{dtype}.png")


def _as_csv_arg(values: Iterable[int]) -> str:
    return ",".join(str(v) for v in values)


def _pick_metric_column(df: pd.DataFrame) -> str:
    for name in ("Time (µs) / matrix", "T(µs)/matrix"):
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


def _select_cta_best_wg(df_cta: pd.DataFrame, metric: str) -> pd.DataFrame:
    # CTA args: arg0=n, arg1=batch, arg2=jobz, arg3=uplo, arg4=wg_mult
    needed = {"arg0", "arg1", "arg2", "arg3", "arg4", metric}
    missing = sorted(needed.difference(df_cta.columns))
    if missing:
        raise ValueError(f"syev_cta: missing columns: {missing}. Available: {list(df_cta.columns)}")

    idx = df_cta.groupby(["arg0", "arg1"])[metric].idxmin()
    return df_cta.loc[idx].copy()


def plot_time_vs_n_fixed_batch(
    df_syev: pd.DataFrame,
    df_syev_cta: pd.DataFrame,
    *,
    batch: int,
    n_values: Sequence[int],
    dtype: str,
    jobz: int,
    uplo: int,
    wg_mult: int,
    best_wg_mult: bool,
    savepath: Optional[str] = None,
    y_metric: str = "throughput",
) -> None:
    metric = _pick_metric_column(df_syev)
    metric_cta = _pick_metric_column(df_syev_cta)
    if metric != metric_cta:
        df_syev_cta = df_syev_cta.rename(columns={metric_cta: metric})

    df_syev = _add_throughput_columns(df_syev)
    df_syev_cta = _add_throughput_columns(df_syev_cta)

    metric_std = f"{metric}_std"

    df_syev = _filter_fixed_batch(df_syev, batch=batch, label="syev")
    df_cta = _filter_fixed_batch(df_syev_cta, batch=batch, label="syev_cta")

    # Filter CTA by jobz/uplo to match labels.
    df_cta = df_cta[(df_cta["arg2"] == jobz) & (df_cta["arg3"] == uplo)].copy()
    if df_cta.empty:
        raise ValueError("syev_cta: no rows after filtering for jobz/uplo")

    if best_wg_mult:
        df_cta = _select_cta_best_wg(df_cta, metric)
        cta_label = "syev_cta(best wg_mult)"
    else:
        df_cta = df_cta[df_cta["arg4"] == wg_mult].copy()
        if df_cta.empty:
            raise ValueError("syev_cta: no rows after filtering for wg_mult")
        cta_label = f"syev_cta(wg_mult={wg_mult})"

    # Filter to requested N list.
    df_syev = df_syev[df_syev["arg0"].isin(n_values)].copy()
    df_cta = df_cta[df_cta["arg0"].isin(n_values)].copy()

    missing_syev = sorted(set(n_values).difference(set(df_syev["arg0"].tolist())))
    missing_cta = sorted(set(n_values).difference(set(df_cta["arg0"].tolist())))
    if missing_syev:
        raise ValueError(f"syev: missing N values for batch={batch}: {missing_syev}")
    if missing_cta:
        raise ValueError(f"syev_cta: missing N values for batch={batch}: {missing_cta}")

    df_syev["impl"] = "syev"
    df_cta["impl"] = cta_label
    df = pd.concat([df_syev, df_cta], ignore_index=True)

    if y_metric == "throughput":
        y_col = "Throughput (matrices/s)"
        y_col_std = f"{y_col}_std"
        ylabel = y_col
        logy = True
        title = f"SYEV vs SYEV_CTA throughput (batch={batch}, type={dtype})"
    else:
        y_col = metric
        y_col_std = metric_std
        ylabel = metric
        logy = False
        title = f"SYEV vs SYEV_CTA (batch={batch}, type={dtype})"

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

    target_path = savepath or _default_plot_path(dtype, batch)
    save_figure(fig, target_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare SYEV vs SYEV_CTA at fixed batch size (two plots: float and cfloat)"
    )
    parser.add_argument("--run", action="store_true", help="run benchmarks before plotting")
    parser.add_argument("--bench-bin", default=_default_bench_path(), help="path to syev_benchmark binary")
    parser.add_argument(
        "--bench-bin-cta",
        default=_default_bench_cta_path(),
        help="path to syev_cta_benchmark binary",
    )
    parser.add_argument("--csv", default=_default_csv_path(), help="CSV output path for syev")
    parser.add_argument("--csv-cta", default=_default_csv_cta_path(), help="CSV output path for syev_cta")

    parser.add_argument("--backend", default="CUDA", help="minibench backend filter")
    parser.add_argument("--no-metric-stddev", action="store_true", help="do not request metric stddev columns")

    parser.add_argument("--batch", type=int, default=16384, help="fixed batch size")
    parser.add_argument(
        "--n-min",
        type=int,
        default=1,
        help="min N (inclusive) for N sweep",
    )
    parser.add_argument(
        "--n-max",
        type=int,
        default=32,
        help="max N (inclusive) for N sweep",
    )

    # These correspond to BM_SYEV_CTA args; syev_benchmark is fixed jobz=EigenVectors, uplo=Lower.
    parser.add_argument("--jobz", type=int, choices=[0, 1], default=1, help="CTA jobz: 0=no vecs, 1=vecs")
    parser.add_argument("--uplo", type=int, choices=[0, 1], default=0, help="CTA uplo: 0=lower, 1=upper")

    wg_group = parser.add_mutually_exclusive_group()
    wg_group.add_argument(
        "--best-wg-mult",
        action="store_true",
        help="for CTA, pick best wg_mult per N from the CSV (requires multiple wg_mult)",
    )
    wg_group.add_argument(
        "--wg-mult",
        type=int,
        default=1,
        help="CTA wg_mult to filter to (used when not using --best-wg-mult)",
    )

    parser.add_argument(
        "--output-float",
        default=None,
        help="optional output path for float plot",
    )
    parser.add_argument(
        "--output-cfloat",
        default=None,
        help="optional output path for complex<float> plot",
    )

    parser.add_argument(
        "--y-metric",
        choices=["throughput", "time"],
        default="throughput",
        help="y-axis metric: throughput (matrices/s) or time (µs)/matrix",
    )

    parser.add_argument(
        "--bench-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="extra args forwarded to both benchmarks (prefix with --)",
    )

    args = parser.parse_args()

    n_values = list(range(args.n_min, args.n_max + 1))
    n_arg = _as_csv_arg(n_values)

    # Fixed batch argument is a single integer.
    b_arg = str(args.batch)

    common_args = [f"--backend={args.backend}"]
    if not args.no_metric_stddev:
        common_args.append("--metric_stddev")

    if args.run:
        # Run once with float filter and once with cfloat filter.
        # Note: we still disambiguate at plot time using the CSV 'name' column.
        for dtype in ("float", "cfloat"):
            type_arg = f"--type={dtype}"
            run_benchmark(args.bench_bin, args.csv, [*common_args, type_arg, n_arg, b_arg, *args.bench_args])
            wg_arg = "1,2,4,8" if args.best_wg_mult else str(args.wg_mult)
            run_benchmark(
                args.bench_bin_cta,
                args.csv_cta,
                [
                    *common_args,
                    type_arg,
                    n_arg,
                    b_arg,
                    str(args.jobz),
                    str(args.uplo),
                    wg_arg,
                    *args.bench_args,
                ],
            )

    if not os.path.isfile(args.csv):
        raise FileNotFoundError(f"CSV not found: {args.csv}")
    if not os.path.isfile(args.csv_cta):
        raise FileNotFoundError(f"CSV not found: {args.csv_cta}")

    df_syev_all = load_results(args.csv)
    df_cta_all = load_results(args.csv_cta)

    # Plot float
    df_syev = _filter_by_scalar_type(df_syev_all, "float", label="syev")
    df_cta = _filter_by_scalar_type(df_cta_all, "float", label="syev_cta")
    plot_time_vs_n_fixed_batch(
        df_syev,
        df_cta,
        batch=args.batch,
        n_values=n_values,
        dtype="float",
        jobz=args.jobz,
        uplo=args.uplo,
        wg_mult=args.wg_mult,
        best_wg_mult=args.best_wg_mult,
        savepath=args.output_float,
        y_metric=args.y_metric,
    )

    # Plot complex<float>
    df_syev = _filter_by_scalar_type(df_syev_all, "cfloat", label="syev")
    df_cta = _filter_by_scalar_type(df_cta_all, "cfloat", label="syev_cta")
    plot_time_vs_n_fixed_batch(
        df_syev,
        df_cta,
        batch=args.batch,
        n_values=n_values,
        dtype="cfloat",
        jobz=args.jobz,
        uplo=args.uplo,
        wg_mult=args.wg_mult,
        best_wg_mult=args.best_wg_mult,
        savepath=args.output_cfloat,
        y_metric=args.y_metric,
    )


if __name__ == "__main__":
    main()
