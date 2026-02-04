from __future__ import annotations
import stylesheet
import os
import subprocess
from typing import Iterable, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


_MARKERS = stylesheet.Markers
_MARKERS_SCALES = stylesheet.MarkerScales
from matplotlib import rcParams as rc


def _ensure_dir_for(path: str) -> str:
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    return path


def run_benchmark(
    binary: str,
    csv_path: str,
    bench_args: Iterable[str],
    env: Optional[Mapping[str, str]] = None,
) -> None:
    """Run a benchmark binary, writing results to csv_path."""
    _ensure_dir_for(csv_path)
    cmd = [binary, f"--csv={csv_path}", *bench_args]
    print(f"Running benchmark: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=None if env is None else {**os.environ, **env})


def load_results(csv_path: str) -> pd.DataFrame:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    return pd.read_csv(csv_path)


def save_figure(fig: plt.Figure, path: str) -> None:
    _ensure_dir_for(path)
    # Many plots use figure-level legends placed outside the axes.
    # Using a tight bounding box prevents them from being clipped.
    fig.savefig(path, bbox_inches="tight")
    print(f"Saved plot to {path}")


def plot_metric(
    df: pd.DataFrame,
    metric: str,
    *,
    x_field: str,
    group_by: str,
    metric_std: Optional[str] = None,
    group_filter: Optional[Sequence] = None,
    label_fmt: str = "group={group}",
    xlabel: str = "X",
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    logx: bool = False,
    logx_base: float = 10,
    logy: bool = False,
    set_xticks: bool = True,
    show_errorbars: bool = False,
    errorbar_capsize: float = 2.0,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    if df.empty:
        raise ValueError("No data to plot")

    if group_filter is not None:
        df = df[df[group_by].isin(group_filter)]
    if df.empty:
        raise ValueError("No data after filtering")

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    x_ticks: list = []
    for idx, (group_value, g) in enumerate(df.groupby(group_by)):
        g_sorted = g.sort_values(x_field)
        x = g_sorted[x_field].to_numpy()
        y = g_sorted[metric].to_numpy()
        yerr = g_sorted[metric_std].to_numpy() if metric_std and metric_std in g_sorted else None

        (line,) = ax.plot(
            x,
            y,
            marker=_MARKERS[idx % len(_MARKERS)],
            markersize=_MARKERS_SCALES[idx % len(_MARKERS_SCALES)] * rc["lines.markersize"],
            linestyle=":",
            label=label_fmt.format(group=group_value, size=group_value),
        )
        if yerr is not None:
            lower = y - yerr
            upper = y + yerr
            if logy:
                # Avoid <=0 values which break log-scale fills.
                tiny = np.finfo(float).tiny
                lower = np.maximum(lower, tiny)
                upper = np.maximum(upper, tiny)

            ax.fill_between(
                x,
                lower,
                upper,
                alpha=0.12,
                color=line.get_color(),
                linewidth=0.0,
            )
            if show_errorbars:
                ax.errorbar(
                    x,
                    y,
                    yerr=yerr,
                    fmt="none",
                    ecolor=line.get_color(),
                    elinewidth=1.0,
                    alpha=0.35,
                    capsize=errorbar_capsize,
                )
        x_ticks.extend(x.tolist())

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel or metric)
    if set_xticks and x_ticks:
        # Use unique, sorted x positions to align ticks with data points.
        xs_sorted = sorted(set(x_ticks))
        ax.set_xticks(xs_sorted)
    if logy:
        ax.set_yscale("log")
    if title:
        ax.set_title(title)
    if logx:
        # Matplotlib changed the keyword from basex -> base.
        try:
            ax.set_xscale("log", base=logx_base)
        except TypeError:
            ax.set_xscale("log", basex=logx_base)
    ax.legend()
    ax.grid(True)
    ax.set_ylim(bottom=0)
    return fig, ax
