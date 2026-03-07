from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def _default_csv() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "..", "output", "iluk_convergence_trace.csv")


def _default_png() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "..", "output", "plots", "iluk_convergence_trace.png")


def _default_summary_png() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "..", "output", "plots", "iluk_convergence_summary.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot ILU(k) LOBPCG convergence traces")
    parser.add_argument("--csv", default=_default_csv(), help="Input convergence CSV")
    parser.add_argument("--output", default=_default_png(), help="Output trace PNG path")
    parser.add_argument("--summary-output", default=_default_summary_png(), help="Output summary PNG path")
    parser.add_argument("--batch", type=int, default=0, help="Batch index to plot")
    parser.add_argument("--case", default="", help="Case label to plot trace for (default: first in CSV)")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    required_cols = {"batch", "iter", "eig", "label", "best_res", "ritz"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")

    if "case_label" not in df.columns:
        df["case_label"] = "single_case"

    df = df[df["batch"] == args.batch].copy()
    if df.empty:
        raise ValueError(f"No rows for batch={args.batch}")

    case_label = args.case or str(df["case_label"].iloc[0])
    dft = df[df["case_label"] == case_label].copy()
    if dft.empty:
        raise ValueError(f"No rows for case_label={case_label!r}")

    labels = list(dft["label"].drop_duplicates())
    eigs = sorted(dft["eig"].drop_duplicates().tolist())

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    ax_res = axes[0]
    ax_ritz = axes[1]

    cmap = plt.get_cmap("tab10")

    for ei, eig in enumerate(eigs):
        dfe = dft[dft["eig"] == eig]
        for li, label in enumerate(labels):
            dfl = dfe[dfe["label"] == label].sort_values("iter")
            color = cmap(li % 10)
            ax_res.plot(
                dfl["iter"],
                dfl["best_res"],
                color=color,
                linestyle=["-", "--", ":", "-."][ei % 4],
                linewidth=1.5,
                label=f"eig={eig} {label}" if ei == 0 else None,
                alpha=0.9,
            )
            ax_ritz.plot(
                dfl["iter"],
                dfl["ritz"],
                color=color,
                linestyle=["-", "--", ":", "-."][ei % 4],
                linewidth=1.3,
                label=f"eig={eig} {label}" if ei == 0 else None,
                alpha=0.9,
            )

    ax_res.set_yscale("log")
    ax_res.set_ylabel("Best residual")
    ax_res.set_title(f"LOBPCG residual convergence per eigenpair ({case_label})")
    ax_res.grid(True, alpha=0.3)

    ax_ritz.set_ylabel("Ritz value")
    ax_ritz.set_xlabel("Iteration")
    ax_ritz.set_title(f"Ritz trajectories per eigenpair ({case_label})")
    ax_ritz.grid(True, alpha=0.3)

    handles, labels = ax_res.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(4, len(handles)), frameon=False)

    fig.tight_layout(rect=[0, 0, 1, 0.94])

    trace_out = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(trace_out), exist_ok=True)
    fig.savefig(trace_out, dpi=160)
    print(f"Saved trace plot: {trace_out}")

    # Summary plot: average final best residual per case and level label.
    final_rows = (
        df.sort_values("iter")
        .groupby(["case_label", "label", "eig"], as_index=False)
        .tail(1)
        .copy()
    )
    summary = (
        final_rows.groupby(["case_label", "label"], as_index=False)["best_res"]
        .mean()
        .rename(columns={"best_res": "avg_final_best"})
    )

    fig2, ax2 = plt.subplots(figsize=(12, 5))
    for label in sorted(summary["label"].drop_duplicates().tolist()):
        dfl = summary[summary["label"] == label].copy()
        ax2.plot(
            dfl["case_label"],
            dfl["avg_final_best"],
            marker="o",
            linewidth=1.4,
            label=label,
        )
    ax2.set_yscale("log")
    ax2.set_ylabel("Average final best residual (log)")
    ax2.set_xlabel("Sweep case")
    ax2.set_title("ILU(k) sweep summary across density/conditioning cases")
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis="x", rotation=45)
    ax2.legend(frameon=False)
    fig2.tight_layout()

    summary_out = os.path.abspath(args.summary_output)
    os.makedirs(os.path.dirname(summary_out), exist_ok=True)
    fig2.savefig(summary_out, dpi=160)
    print(f"Saved summary plot: {summary_out}")


if __name__ == "__main__":
    main()
