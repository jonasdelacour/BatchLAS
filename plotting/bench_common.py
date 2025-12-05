from __future__ import annotations
import stylesheet
import os
import subprocess
from typing import Iterable, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
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
    fig.savefig(path)
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
    logy: bool = False,
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
        yerr = g_sorted[metric_std] if metric_std and metric_std in g_sorted else None
        ax.plot(
            g_sorted[x_field],
            g_sorted[metric],
            marker=_MARKERS[idx % len(_MARKERS)],
            markersize=_MARKERS_SCALES[idx % len(_MARKERS_SCALES)]*rc["lines.markersize"],
            linestyle=":",
            label=label_fmt.format(group=group_value, size=group_value),
        )
        if yerr is not None:
            ax.fill_between(
                g_sorted[x_field],
                g_sorted[metric] - yerr,
                g_sorted[metric] + yerr,
                alpha=0.15,
                joinstyle='bevel',
            )
        x_ticks.extend(g_sorted[x_field].tolist())

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel or metric)
    if x_ticks:
        # Use unique, sorted x positions to align ticks with data points.
        xs_sorted = sorted(set(x_ticks))
        ax.set_xticks(xs_sorted)
    if logy:
        ax.set_yscale("log")
    if title:
        ax.set_title(title)
    ax.legend()
    ax.grid(True)
    return fig, ax
