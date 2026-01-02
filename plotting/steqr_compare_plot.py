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
    return os.path.join(_here(), "..", "build", "benchmarks", "steqr_benchmark")


def _default_bench_cta_path() -> str:
    return os.path.join(_here(), "..", "build", "benchmarks", "steqr_cta_benchmark")


def _default_csv_path() -> str:
    return os.path.join(_here(), "..", "build", "steqr_out.csv")


def _default_csv_cta_path() -> str:
    return os.path.join(_here(), "..", "build", "steqr_cta_out.csv")


def _default_plot_path() -> str:
    return os.path.join(_here(), "..", "output", "plots", "steqr_compare_by_batch.png")


def _default_batches() -> list[int]:
    return [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]


def _as_csv_arg(values: Iterable[int]) -> str:
    return ",".join(str(v) for v in values)


def _require_columns(df: pd.DataFrame, cols: Sequence[str], *, label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{label}: missing columns: {missing}. Available: {list(df.columns)}")


def _load_with_impl(csv_path: str, *, impl: str) -> pd.DataFrame:
    df = load_results(csv_path)
    df = df.copy()
    df["impl"] = impl
    return df


def plot_compare_time_vs_batch(
    df_steqr: pd.DataFrame,
    df_steqr_cta: pd.DataFrame,
    *,
    n_values: Sequence[int],
    savepath: Optional[str] = None,
) -> None:
    metric = "T(µs)/Batch"
    metric_std = f"{metric}_std"

    _require_columns(df_steqr, ["arg0", "arg1", metric], label="steqr")
    _require_columns(df_steqr_cta, ["arg0", "arg1", metric], label="steqr_cta")

    df = pd.concat([df_steqr, df_steqr_cta], ignore_index=True)

    fig, axes = plt.subplots(1, len(n_values), sharey=True)
    if len(n_values) == 1:
        axes = [axes]

    for ax, n in zip(axes, n_values):
        dfn = df[df["arg0"] == n]
        if dfn.empty:
            raise ValueError(f"No data for N={n}. Check your CSVs / benchmark args.")

        plot_metric(
            dfn,
            metric,
            x_field="arg1",
            group_by="impl",
            metric_std=metric_std,
            label_fmt="{group}",
            xlabel="Batch Size",
            ylabel=metric if ax is axes[0] else None,
            title=None,
            logx=True,
            logx_base=2,
            logy=True,
            set_xticks=False,
            show_errorbars=True,
            ax=ax,
        )

        # Avoid huge titles on small subplots with the global stylesheet.
        ax.set_title(f"N={n}", fontsize=22)

        # Even on log2 x-scale, prefer base-10 integer tick labels.
        ticks = sorted(set(int(v) for v in dfn["arg1"].tolist()))
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{int(v)}" if v >= 1 else f"{v:g}"))
        ax.tick_params(axis="x", labelrotation=45)

    fig.suptitle("STEQR vs STEQR_CTA (time per batch)", fontsize=22)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    target_path = savepath or _default_plot_path()
    save_figure(fig, target_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run and plot STEQR vs STEQR_CTA")
    parser.add_argument("--run", action="store_true", help="run benchmarks before plotting")
    parser.add_argument("--bench-bin", default=_default_bench_path(), help="path to steqr_benchmark binary")
    parser.add_argument(
        "--bench-bin-cta",
        default=_default_bench_cta_path(),
        help="path to steqr_cta_benchmark binary",
    )
    parser.add_argument("--csv", default=_default_csv_path(), help="CSV output path for steqr")
    parser.add_argument("--csv-cta", default=_default_csv_cta_path(), help="CSV output path for steqr_cta")
    parser.add_argument("--output", default=None, help="optional path to save the plot")

    parser.add_argument("--backend", default="CUDA", help="minibench backend filter")
    parser.add_argument("--type", dest="dtype", default="float", help="minibench type filter")
    parser.add_argument("--no-metric-stddev", action="store_true", help="do not request metric stddev columns")

    parser.add_argument("--n", type=int, nargs="+", default=[8, 16, 32], help="matrix sizes N to compare")
    parser.add_argument(
        "--batches",
        type=int,
        nargs="+",
        default=_default_batches(),
        help="batch sizes to run (only used with --run)",
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

        # Provide explicit args so we only run the sizes we care about.
        # minibench will take the cartesian product of provided arg lists.
        run_args = [*common_args, n_arg, b_arg, *args.bench_args]
        run_benchmark(args.bench_bin, args.csv, run_args)
        run_benchmark(args.bench_bin_cta, args.csv_cta, run_args)

    if not os.path.isfile(args.csv):
        raise FileNotFoundError(f"CSV not found: {args.csv}")
    if not os.path.isfile(args.csv_cta):
        raise FileNotFoundError(f"CSV not found: {args.csv_cta}")

    df_steqr = _load_with_impl(args.csv, impl="steqr")
    df_steqr_cta = _load_with_impl(args.csv_cta, impl="steqr_cta")

    plot_compare_time_vs_batch(df_steqr, df_steqr_cta, n_values=args.n, savepath=args.output)


if __name__ == "__main__":
    main()
