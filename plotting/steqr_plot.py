from __future__ import annotations
import argparse
import os
from typing import Optional, Sequence
import pandas as pd
from bench_common import load_results, plot_metric, run_benchmark, save_figure


def _default_bench_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "..", "build", "benchmarks", "steqr_benchmark")

def _default_csv_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "..", "build", "steqr_out.csv")


def _default_plot_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "..", "output", "plots", "steqr.png")


def _default_plot_batch_path(n_values: Optional[Sequence[int]]) -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    if n_values and len(n_values) == 1:
        fname = f"steqr_by_batch_N{n_values[0]}.png"
    else:
        fname = "steqr_by_batch.png"
    return os.path.join(here, "..", "output", "plots", fname)


def plot_steqr_benchmark(df: pd.DataFrame, savepath: Optional[str] = None) -> None:
    metric = "T(µs)/Batch"
    metric_std = f"{metric}_std"

    fig, _ = plot_metric(
        df,
        metric,
        x_field="arg0",
        group_by="arg1",
        metric_std=metric_std,
        label_fmt="batch={group}",
        xlabel="Matrix Size (N)",
        ylabel=metric,
        title="STEQR Benchmark",
        logy=True,
    )

    target_path = savepath or _default_plot_path()
    save_figure(fig, target_path)


def plot_time_vs_batch(df: pd.DataFrame, n_values: Optional[Sequence[int]], savepath: Optional[str] = None) -> None:
    metric = "T(µs)/Batch"
    metric_std = f"{metric}_std"

    fig, _ = plot_metric(
        df,
        metric,
        x_field="arg1",
        group_by="arg0",
        group_filter=n_values,
        metric_std=metric_std,
        label_fmt="N={group}",
        xlabel="Batch Size",
        ylabel=metric,
        title="STEQR Benchmark (fixed N)",
        logy=True,
    )

    target_path = savepath or _default_plot_batch_path(n_values)
    save_figure(fig, target_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run STEQR benchmark and plot results")
    parser.add_argument("--run", action="store_true", help="run benchmark before plotting")
    parser.add_argument("--bench-bin", default=_default_bench_path(), help="path to steqr_benchmark binary")
    parser.add_argument("--csv", default=_default_csv_path(), help="CSV output path")
    parser.add_argument("--output", default=None, help="optional path to save the plot (default: build/plots/steqr.png)")
    parser.add_argument("--output-batch", default=None, help="optional path for batch-size plot (default: build/plots/steqr_by_batch*.png)")
    parser.add_argument("--n", type=int, nargs="+", default=None, help="one or more matrix sizes N for batch-size plot (default: all available)")
    parser.add_argument(
        "--bench-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="extra args forwarded to steqr_benchmark (prefix with --)",
    )

    args = parser.parse_args()

    if args.run:
        run_benchmark(args.bench_bin, args.csv, args.bench_args)

    if not os.path.isfile(args.csv):
        raise FileNotFoundError(f"CSV not found: {args.csv}")

    df = load_results(args.csv)
    plot_steqr_benchmark(df, savepath=args.output)
    plot_time_vs_batch(df, n_values=args.n, savepath=args.output_batch)


if __name__ == "__main__":
    main()

