from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _default_summary_csv() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "..", "output", "accuracy", "iluk_convergence_stats_summary.csv")


def _default_hist_png() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "..", "output", "plots", "iluk_convergence_histograms.png")


def _default_heatmap_png() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "..", "output", "plots", "iluk_convergence_bucket_heatmaps.png")


def _label_order(df: pd.DataFrame) -> list[str]:
    labels = sorted(df["label"].dropna().unique().tolist())
    if "baseline" in labels:
        labels.remove("baseline")
        labels = ["baseline"] + labels
    return labels


def _finite_log10(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values) & (values > 0.0)]
    if values.size == 0:
        return values
    return np.log10(values)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot ILU(k) convergence histograms from benchmark summary CSV")
    parser.add_argument("--summary-csv", default=_default_summary_csv(), help="Summary CSV emitted by iluk_convergence_stats")
    parser.add_argument("--hist-output", default=_default_hist_png(), help="Output path for histogram summary PNG")
    parser.add_argument("--heatmap-output", default=_default_heatmap_png(), help="Output path for bucket heatmap PNG")
    args = parser.parse_args()

    df = pd.read_csv(args.summary_csv)
    required = {
        "label",
        "final_best",
        "first_tol_iter",
        "converged",
        "ratio_vs_baseline",
        "log10_ratio_vs_baseline",
        "density",
        "diag_boost",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")

    df = df.copy()
    df["converged"] = pd.to_numeric(df["converged"], errors="coerce").fillna(0).astype(int)
    df["first_tol_iter"] = pd.to_numeric(df["first_tol_iter"], errors="coerce")
    label_order = _label_order(df)
    colors = {label: plt.get_cmap("tab10")(idx % 10) for idx, label in enumerate(label_order)}

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    ax_final = axes[0, 0]
    ax_tol = axes[0, 1]
    ax_ratio = axes[1, 0]
    ax_conv = axes[1, 1]

    all_final = _finite_log10(df["final_best"])
    final_bins = np.linspace(all_final.min(), all_final.max(), 36) if all_final.size else 30
    for label in label_order:
        subset = df[df["label"] == label]
        values = _finite_log10(subset["final_best"])
        if values.size == 0:
            continue
        ax_final.hist(values, bins=final_bins, histtype="step", linewidth=1.8, label=label, color=colors[label])
    ax_final.set_title("Final residual distribution across all eigenpairs")
    ax_final.set_xlabel("log10(final best residual)")
    ax_final.set_ylabel("Count")
    ax_final.grid(True, alpha=0.25)

    tol_values = df.copy()
    valid_tol_iters = pd.to_numeric(tol_values["first_tol_iter"], errors="coerce").replace(-1, np.nan)
    if tol_values.empty or valid_tol_iters.dropna().empty:
        max_iter = 0
    else:
        max_iter = int(valid_tol_iters.max())
    tol_values["tol_iter_hist"] = tol_values["first_tol_iter"].where(tol_values["first_tol_iter"] >= 0, max_iter + 2)
    tol_bins = np.arange(-0.5, max_iter + 3.5, 1.0)
    for label in label_order:
        subset = tol_values[tol_values["label"] == label]
        values = pd.to_numeric(subset["tol_iter_hist"], errors="coerce").dropna().to_numpy(dtype=float)
        if values.size == 0:
            continue
        ax_tol.hist(values, bins=tol_bins, histtype="step", linewidth=1.8, label=label, color=colors[label])
    ax_tol.set_title("Iterations to reach tolerance")
    ax_tol.set_xlabel("Iteration (not converged shown in final bin)")
    ax_tol.set_ylabel("Count")
    ax_tol.grid(True, alpha=0.25)

    ratio_df = df[df["label"] != "baseline"].copy()
    for label in [label for label in label_order if label != "baseline"]:
        subset = ratio_df[ratio_df["label"] == label]
        values = pd.to_numeric(subset["log10_ratio_vs_baseline"], errors="coerce")
        values = values[np.isfinite(values)]
        if values.empty:
            continue
        bins = np.linspace(values.min(), values.max(), 36) if len(values) > 1 else 15
        ax_ratio.hist(values, bins=bins, histtype="step", linewidth=1.8, label=label, color=colors[label])
    ax_ratio.axvline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.7)
    ax_ratio.set_title("Paired improvement vs baseline")
    ax_ratio.set_xlabel("log10(final residual / baseline final residual)")
    ax_ratio.set_ylabel("Count")
    ax_ratio.grid(True, alpha=0.25)

    convergence = (
        df.groupby("label", as_index=False)["converged"]
        .mean()
        .rename(columns={"converged": "converged_fraction"})
    )
    convergence["label"] = pd.Categorical(convergence["label"], categories=label_order, ordered=True)
    convergence = convergence.sort_values("label")
    ax_conv.bar(convergence["label"], convergence["converged_fraction"], color=[colors[label] for label in convergence["label"]])
    ax_conv.set_ylim(0.0, 1.0)
    ax_conv.set_title("Converged eigenpair fraction")
    ax_conv.set_xlabel("Method")
    ax_conv.set_ylabel("Fraction converged")
    ax_conv.grid(True, axis="y", alpha=0.25)
    ax_conv.tick_params(axis="x", rotation=30)

    handles, labels = ax_final.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(5, len(handles)), frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    hist_out = os.path.abspath(args.hist_output)
    os.makedirs(os.path.dirname(hist_out), exist_ok=True)
    fig.savefig(hist_out, dpi=170)
    print(f"Saved histogram summary: {hist_out}")

    nonbaseline = [label for label in label_order if label != "baseline"]
    if nonbaseline:
        fig2, axes2 = plt.subplots(1, len(nonbaseline), figsize=(5.5 * len(nonbaseline), 4.8), squeeze=False)
        for ax, label in zip(axes2[0], nonbaseline):
            subset = df[df["label"] == label].copy()
            pivot = subset.pivot_table(
                index="density",
                columns="diag_boost",
                values="log10_ratio_vs_baseline",
                aggfunc="median",
            ).sort_index().sort_index(axis=1)
            im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", origin="lower", cmap="coolwarm", vmin=-1.0, vmax=1.0)
            ax.set_title(f"{label} median log10 ratio")
            ax.set_xlabel("Diagonal boost")
            ax.set_ylabel("Density")
            ax.set_xticks(np.arange(len(pivot.columns)), [f"{c:g}" for c in pivot.columns])
            ax.set_yticks(np.arange(len(pivot.index)), [f"{r:g}" for r in pivot.index])
            for i in range(len(pivot.index)):
                for j in range(len(pivot.columns)):
                    value = pivot.to_numpy(dtype=float)[i, j]
                    if np.isfinite(value):
                        ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=9, color="black")
        fig2.colorbar(im, ax=axes2.ravel().tolist(), shrink=0.9, label="Median log10 ratio vs baseline")
        fig2.tight_layout()

        heatmap_out = os.path.abspath(args.heatmap_output)
        os.makedirs(os.path.dirname(heatmap_out), exist_ok=True)
        fig2.savefig(heatmap_out, dpi=170)
        print(f"Saved bucket heatmaps: {heatmap_out}")


if __name__ == "__main__":
    main()