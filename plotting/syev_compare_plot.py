from __future__ import annotations

import argparse
import os
from typing import Iterable, Optional, Sequence

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

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


def _default_plot_path() -> str:
    return os.path.join(_here(), "..", "output", "plots", "syev_compare_by_batch.png")


def _default_batches() -> list[int]:
    return [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]


def _as_csv_arg(values: Iterable[int]) -> str:
    return ",".join(str(v) for v in values)


def _require_columns(df: pd.DataFrame, cols: Sequence[str], *, label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{label}: missing columns: {missing}. Available: {list(df.columns)}")


def _pick_metric_column(df: pd.DataFrame) -> str:
    # BatchLAS benchmarks have both styles in the tree.
    for name in ("Time (µs) / Batch", "T(µs)/Batch"):
        if name in df.columns:
            return name
    raise ValueError(
        f"No known time-per-batch metric column found. Available: {list(df.columns)}"
    )


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


def _load_with_impl(csv_path: str, *, impl: str) -> pd.DataFrame:
    df = load_results(csv_path)
    df = df.copy()
    df["impl"] = impl
    return df


def _filter_cta(
    df: pd.DataFrame,
    *,
    jobz: int,
    uplo: int,
    wg_mult: Optional[int],
    best_wg_mult: bool,
    metric: str,
) -> pd.DataFrame:
    # CTA benchmark args:
    # arg0=n, arg1=batch, arg2=jobz, arg3=uplo, arg4=wg_mult
    _require_columns(df, ["arg0", "arg1", "arg2", "arg3", "arg4", metric], label="syev_cta")

    df = df[(df["arg2"] == jobz) & (df["arg3"] == uplo)]
    if df.empty:
        raise ValueError("No syev_cta data after filtering for jobz/uplo")

    if best_wg_mult:
        # Keep the best wg_mult (minimum time) per (N, batch).
        idx = df.groupby(["arg0", "arg1"])[metric].idxmin()
        df = df.loc[idx].copy()
    else:
        if wg_mult is None:
            raise ValueError("wg_mult must be provided when best_wg_mult is False")
        df = df[df["arg4"] == wg_mult].copy()

    if df.empty:
        raise ValueError("No syev_cta data after wg_mult selection")
    return df


def plot_compare_time_vs_batch(
    df_syev: pd.DataFrame,
    df_syev_cta: pd.DataFrame,
    *,
    n_values: Sequence[int],
    jobz: int,
    uplo: int,
    wg_mult: Optional[int],
    best_wg_mult: bool,
    savepath: Optional[str] = None,
    y_metric: str = "throughput",
) -> None:
    metric_time = _pick_metric_column(df_syev)
    metric_time_cta = _pick_metric_column(df_syev_cta)
    if metric_time != metric_time_cta:
        # Normalize to one name so concat + plotting is easy.
        df_syev_cta = df_syev_cta.rename(columns={metric_time_cta: metric_time})

    df_syev = _add_throughput_columns(df_syev)
    df_syev_cta = _add_throughput_columns(df_syev_cta)

    _require_columns(df_syev, ["arg0", "arg1", metric_time], label="syev")
    df_cta = _filter_cta(
        df_syev_cta,
        jobz=jobz,
        uplo=uplo,
        wg_mult=wg_mult,
        best_wg_mult=best_wg_mult,
        metric=metric_time,
    )

    # syev_benchmark is hardcoded to jobz=EigenVectors and uplo=Lower.
    # Keep the plot honest by encoding CTA selection in the label.
    if best_wg_mult:
        cta_label = "syev_cta(best wg_mult)"
    else:
        cta_label = f"syev_cta(wg_mult={wg_mult})"

    df_syev = df_syev.copy()
    df_syev["impl"] = "syev"
    df_cta = df_cta.copy()
    df_cta["impl"] = cta_label

    df = pd.concat([df_syev, df_cta], ignore_index=True)

    # Guard against accidentally plotting multiple series collapsed into one group.
    _require_unique_points(df[df["impl"] == "syev"], label="syev", keys=["arg0", "arg1"])
    _require_unique_points(df[df["impl"].str.startswith("syev_cta")], label="syev_cta", keys=["arg0", "arg1"])

    fig, axes = plt.subplots(1, len(n_values), sharey=True)
    if len(n_values) == 1:
        axes = [axes]

    for ax, n in zip(axes, n_values):
        dfn = df[df["arg0"] == n]
        if dfn.empty:
            raise ValueError(f"No data for N={n}. Check your CSVs / benchmark args.")

        if y_metric == "throughput":
            metric = "Throughput (matrices/s)"
            metric_std = f"{metric}_std"
            ylabel = metric if ax is axes[0] else None
            logy = False
        else:
            metric = metric_time
            metric_std = f"{metric}_std"
            ylabel = metric if ax is axes[0] else None
            logy = True

        plot_metric(
            dfn,
            metric,
            x_field="arg1",
            group_by="impl",
            metric_std=metric_std,
            label_fmt="{group}",
            xlabel="Batch Size",
            ylabel=ylabel,
            title=None,
            logx=True,
            logx_base=2,
            logy=logy,
            set_xticks=False,
            show_errorbars=True,
            ax=ax,
        )

        ax.set_title(f"N={n}", fontsize=22)

        ticks = sorted(set(int(v) for v in dfn["arg1"].tolist()))
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{int(v)}" if v >= 1 else f"{v:g}"))
        ax.tick_params(axis="x", labelrotation=45)

    subtitle = "jobz=EigenVectors, uplo=Lower"
    if best_wg_mult:
        subtitle += " (CTA picks best wg_mult)"
    else:
        subtitle += f" (CTA wg_mult={wg_mult})"

    metric_title = "throughput" if y_metric == "throughput" else "time per batch"
    fig.suptitle(f"SYEV vs SYEV_CTA ({metric_title})\n{subtitle}", fontsize=22)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    target_path = savepath or _default_plot_path()
    save_figure(fig, target_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run and plot SYEV vs SYEV_CTA (small N)")
    parser.add_argument("--run", action="store_true", help="run benchmarks before plotting")
    parser.add_argument("--bench-bin", default=_default_bench_path(), help="path to syev_benchmark binary")
    parser.add_argument(
        "--bench-bin-cta",
        default=_default_bench_cta_path(),
        help="path to syev_cta_benchmark binary",
    )
    parser.add_argument("--csv", default=_default_csv_path(), help="CSV output path for syev")
    parser.add_argument("--csv-cta", default=_default_csv_cta_path(), help="CSV output path for syev_cta")
    parser.add_argument("--output", default=None, help="optional path to save the plot")

    parser.add_argument("--backend", default="CUDA", help="minibench backend filter")
    parser.add_argument("--type", dest="dtype", default="float", help="minibench type filter")
    parser.add_argument("--no-metric-stddev", action="store_true", help="do not request metric stddev columns")
    parser.add_argument(
        "--y-metric",
        choices=["throughput", "time"],
        default="throughput",
        help="plot throughput (matrices/s) or time per batch",
    )

    parser.add_argument("--n", type=int, nargs="+", default=[8, 16, 32], help="matrix sizes N to compare")
    parser.add_argument(
        "--batches",
        type=int,
        nargs="+",
        default=_default_batches(),
        help="batch sizes to run (only used with --run)",
    )

    # These correspond to BM_SYEV_CTA arguments; syev_benchmark itself is fixed.
    parser.add_argument("--jobz", type=int, choices=[0, 1], default=1, help="CTA jobz: 0=no vecs, 1=vecs")
    parser.add_argument("--uplo", type=int, choices=[0, 1], default=0, help="CTA uplo: 0=lower, 1=upper")

    wg_group = parser.add_mutually_exclusive_group()
    wg_group.add_argument(
        "--best-wg-mult",
        action="store_true",
        help="for CTA, pick best wg_mult per (N,batch) from the CSV (requires CSV to contain multiple wg_mult)",
    )
    wg_group.add_argument(
        "--wg-mult",
        type=int,
        default=1,
        help="CTA wg_mult to filter to (used when not using --best-wg-mult)",
    )

    parser.add_argument(
        "--bench-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="extra args forwarded to both benchmarks (prefix with --)",
    )

    args = parser.parse_args()

    if args.run:
        n_arg = _as_csv_arg(args.n)
        b_arg = _as_csv_arg(args.batches)

        common_args = [f"--backend={args.backend}", f"--type={args.dtype}"]
        if not args.no_metric_stddev:
            common_args.append("--metric_stddev")

        # syev_benchmark is (N, batch).
        run_benchmark(args.bench_bin, args.csv, [*common_args, n_arg, b_arg, *args.bench_args])

        # syev_cta_benchmark is (N, batch, jobz, uplo, wg_mult).
        if args.best_wg_mult:
            wg_arg = "1,2,4,8"
        else:
            wg_arg = str(args.wg_mult if args.wg_mult is not None else 1)

        run_benchmark(
            args.bench_bin_cta,
            args.csv_cta,
            [
                *common_args,
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

    df_syev = load_results(args.csv)
    df_syev_cta = load_results(args.csv_cta)

    # Disambiguate real vs complex results for plotting.
    df_syev = _filter_by_scalar_type(df_syev, args.dtype, label="syev")
    df_syev_cta = _filter_by_scalar_type(df_syev_cta, args.dtype, label="syev_cta")

    plot_compare_time_vs_batch(
        df_syev,
        df_syev_cta,
        n_values=args.n,
        jobz=args.jobz,
        uplo=args.uplo,
        wg_mult=args.wg_mult,
        best_wg_mult=args.best_wg_mult,
        savepath=args.output,
        y_metric=args.y_metric,
    )


if __name__ == "__main__":
    main()
