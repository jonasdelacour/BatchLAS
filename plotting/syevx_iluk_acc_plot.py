from __future__ import annotations

import argparse
import os
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bench_common import load_results, run_benchmark, save_figure


def _here() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _default_bench_path() -> str:
    return os.path.join(_here(), "..", "build", "benchmarks", "syevx_iluk_acc")


def _default_csv_path() -> str:
    return os.path.join(_here(), "..", "output", "accuracy", "syevx_iluk_acc.csv")


def _default_summary_plot_path() -> str:
    return os.path.join(_here(), "..", "output", "plots", "syevx_iluk_acc_summary.png")


def _default_heatmap_plot_path() -> str:
    return os.path.join(_here(), "..", "output", "plots", "syevx_iluk_acc_heatmaps.png")


def _parse_impl_order(df: pd.DataFrame) -> list[str]:
    impls = sorted(df["tag_impl"].dropna().unique().tolist())
    if "syevx_sparse_baseline" in impls:
        impls.remove("syevx_sparse_baseline")
        impls = ["syevx_sparse_baseline"] + impls
    return impls


def _impl_label(impl: str) -> str:
    if impl == "syevx_sparse_baseline":
        return "Baseline"
    if impl.startswith("syevx_sparse_iluk_k"):
        return "ILUK k=" + impl.rsplit("k", 1)[-1]
    return impl


def _require_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = set(required).difference(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")


def _coerce_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _filter_ok(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ok"] = pd.to_numeric(out["ok"], errors="coerce").fillna(0).astype(int)
    return out[out["ok"] == 1].copy()


def plot_summary(df: pd.DataFrame, output: str) -> None:
    impl_order = _parse_impl_order(df)
    summary = (
        df.groupby("tag_impl", as_index=False)
        .agg(
            iterations_done=("iterations_done", "median"),
            total_time_sec=("total_time_sec", "median"),
            final_best_max=("final_best_max", "median"),
            converged_fraction=("converged_fraction", "mean"),
            iter_ratio_vs_baseline=("iter_ratio_vs_baseline", "median"),
            time_ratio_vs_baseline=("time_ratio_vs_baseline", "median"),
            residual_ratio_vs_baseline=("residual_ratio_vs_baseline", "median"),
            fill_ratio=("fill_ratio", "median"),
        )
    )
    summary["tag_impl"] = pd.Categorical(summary["tag_impl"], categories=impl_order, ordered=True)
    summary = summary.sort_values("tag_impl")

    x = np.arange(len(summary))
    labels = [_impl_label(v) for v in summary["tag_impl"].astype(str)]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    ax_iter = axes[0, 0]
    ax_time = axes[0, 1]
    ax_resid = axes[1, 0]
    ax_conv = axes[1, 1]

    ax_iter.bar(x, summary["iterations_done"])
    ax_iter.set_title("Median iteration count")
    ax_iter.set_ylabel("Iterations")
    ax_iter.set_xticks(x, labels, rotation=20)

    ax_time.bar(x, summary["total_time_sec"])
    ax_time.set_title("Median total time")
    ax_time.set_ylabel("Seconds / matrix")
    ax_time.set_xticks(x, labels, rotation=20)

    iluk_only = summary[summary["tag_impl"].astype(str) != "syevx_sparse_baseline"].copy()
    if not iluk_only.empty:
        x_iluk = np.arange(len(iluk_only))
        labels_iluk = [_impl_label(v) for v in iluk_only["tag_impl"].astype(str)]
        ax_resid.bar(x_iluk - 0.25, iluk_only["iter_ratio_vs_baseline"], width=0.25, label="Iter ratio")
        ax_resid.bar(x_iluk, iluk_only["time_ratio_vs_baseline"], width=0.25, label="Time ratio")
        ax_resid.bar(x_iluk + 0.25, iluk_only["residual_ratio_vs_baseline"], width=0.25, label="Residual ratio")
        ax_resid.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
        ax_resid.set_title("Median paired ratios vs baseline")
        ax_resid.set_ylabel("Ratio")
        ax_resid.set_xticks(x_iluk, labels_iluk, rotation=20)
        ax_resid.legend(frameon=False)
    else:
        ax_resid.set_visible(False)

    ax_conv.bar(x, summary["converged_fraction"])
    ax_conv.set_ylim(0.0, 1.0)
    ax_conv.set_title("Mean converged eigenpair fraction")
    ax_conv.set_ylabel("Fraction")
    ax_conv.set_xticks(x, labels, rotation=20)

    fig.tight_layout()
    save_figure(fig, output)


def plot_heatmaps(df: pd.DataFrame, output: str, metric: str = "iter_ratio_vs_baseline") -> None:
    iluk_df = df[df["tag_impl"].astype(str) != "syevx_sparse_baseline"].copy()
    if iluk_df.empty:
        raise ValueError("No ILUK rows available for heatmap plot")

    impl_order = [impl for impl in _parse_impl_order(iluk_df) if impl != "syevx_sparse_baseline"]
    fig, axes = plt.subplots(
        1,
        len(impl_order),
        figsize=(5.5 * len(impl_order), 4.8),
        squeeze=False,
        constrained_layout=True,
    )

    image = None
    for ax, impl in zip(axes[0], impl_order):
        subset = iluk_df[iluk_df["tag_impl"] == impl]
        pivot = subset.pivot_table(
            index="drop_tolerance",
            columns="fill_factor",
            values=metric,
            aggfunc="median",
        ).sort_index(ascending=False).sort_index(axis=1)
        image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", origin="upper", cmap="coolwarm")
        ax.set_title(_impl_label(impl))
        ax.set_xlabel("Fill factor")
        ax.set_ylabel("Drop tolerance")
        ax.set_xticks(np.arange(len(pivot.columns)), [f"{c:g}" for c in pivot.columns])
        ax.set_yticks(np.arange(len(pivot.index)), [f"{r:.0e}" for r in pivot.index])
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                value = pivot.to_numpy(dtype=float)[i, j]
                if np.isfinite(value):
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=9, color="black")

    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.9, label=f"Median {metric}")
    save_figure(fig, output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot sparse SYEVX + ILUK miniacc results")
    parser.add_argument("--run", action="store_true", help="run the benchmark before plotting")
    parser.add_argument("--bench-bin", default=_default_bench_path(), help="path to syevx_iluk_acc binary")
    parser.add_argument("--csv", default=_default_csv_path(), help="CSV output/input path")
    parser.add_argument("--summary-output", default=_default_summary_plot_path(), help="summary plot output path")
    parser.add_argument("--heatmap-output", default=_default_heatmap_plot_path(), help="heatmap output path")
    parser.add_argument("--samples", type=int, default=32, help="samples for benchmark run")
    parser.add_argument("--backend", default="CUDA", help="backend filter for benchmark run")
    parser.add_argument("--type", dest="dtype", default="float", help="type filter for benchmark run")
    parser.add_argument("--benchmark-filter", default="*SPARSE*", help="benchmark filter for benchmark run")
    parser.add_argument("--bench-args", nargs=argparse.REMAINDER, default=[], help="extra benchmark args")
    args = parser.parse_args()

    if args.run:
        run_benchmark(
            args.bench_bin,
            args.csv,
            [
                f"--samples={args.samples}",
                f"--backend={args.backend}",
                f"--type={args.dtype}",
                f"--benchmark_filter={args.benchmark_filter}",
                *args.bench_args,
            ],
        )

    df = load_results(args.csv)
    _require_columns(
        df,
        [
            "ok",
            "tag_impl",
            "iterations_done",
            "total_time_sec",
            "final_best_max",
            "converged_fraction",
            "fill_factor",
            "drop_tolerance",
        ],
    )
    df = _coerce_numeric(
        df,
        [
            "ok",
            "iterations_done",
            "total_time_sec",
            "final_best_max",
            "converged_fraction",
            "iter_ratio_vs_baseline",
            "time_ratio_vs_baseline",
            "residual_ratio_vs_baseline",
            "fill_ratio",
            "fill_factor",
            "drop_tolerance",
        ],
    )
    df_ok = _filter_ok(df)

    plot_summary(df_ok, args.summary_output)
    plot_heatmaps(df_ok, args.heatmap_output)


if __name__ == "__main__":
    main()