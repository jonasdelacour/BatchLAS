#!/usr/bin/env python3
"""Plot syevx Ritz-value relative-error history on a batch of symmetric sparse matrices.

This playground example:
  1) imports the build-tree `batchlas` package if it is available
  2) constructs a batch of non-diagonal symmetric CSR matrices
  3) optionally builds an ILUK preconditioner for the batch
  4) runs `batchlas.syevx(..., return_history=True)` with Ritz-value tracking
  5) plots the per-iteration relative error of each Ritz value

Example:
  python3 playground/syevx_history_batch.py \
      --n 48 --batch-size 3 --neigs 4 --iterations 40 \
      --compare-scipy-lobpcg \
      --backend auto --device gpu \
      --out output/playground/syevx_history_batch.png
"""

from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "matplotlib is required for this example. "
        "Install it with `python3 -m pip install matplotlib`.\n"
        f"Original import error: {exc}"
    )


REPO_ROOT = Path(__file__).resolve().parents[1]
BUILD_PYTHON = REPO_ROOT / "build" / "python"
if str(BUILD_PYTHON) not in sys.path:
    sys.path.insert(0, str(BUILD_PYTHON))

import batchlas as bl  # noqa: E402


@dataclass
class SolverTrace:
    name: str
    final_values: np.ndarray
    histories: list[np.ndarray]
    line_style: str = "-"


def make_symmetric_csr_batch(
    n: int,
    batch_size: int,
    density: float,
    seed: int,
) -> list[sp.csr_matrix]:
    rng = np.random.default_rng(seed)
    matrices: list[sp.csr_matrix] = []
    density = float(np.clip(density, 0.0, 1.0))
    upper_mask = rng.random((n, n)) < density
    upper_mask = np.triu(upper_mask, k=2)

    for batch in range(batch_size):
        dense = np.zeros((n, n), dtype=np.float64)

        # Always include a strong tridiagonal backbone so the matrix is visibly
        # non-diagonal and well-conditioned enough for the iterative solver.
        diagonal = 4.0 + rng.random(n)
        offdiag = 0.15 + 0.35 * rng.random(n - 1)
        dense[np.arange(n), np.arange(n)] = diagonal
        dense[np.arange(n - 1), np.arange(1, n)] = offdiag
        dense[np.arange(1, n), np.arange(n - 1)] = offdiag

        # Add a few random long-range symmetric couplings.
        upper_values = rng.normal(scale=0.25, size=(n, n)) * upper_mask
        dense += upper_values + upper_values.T

        # Make the matrix more strongly diagonally dominant per batch.
        dense[np.arange(n), np.arange(n)] += 0.5 * np.sum(np.abs(dense), axis=1)

        matrices.append(sp.csr_matrix(dense))

    return matrices


def select_reference_eigenvalues(matrix: sp.csr_matrix, neigs: int, find_largest: bool) -> np.ndarray:
    values = np.linalg.eigvalsh(matrix.toarray())
    if find_largest:
        return values[::-1][:neigs]
    return values[:neigs]


def relative_error(approx: np.ndarray, exact: np.ndarray) -> np.ndarray:
    scale = np.maximum(np.abs(exact), np.finfo(np.float64).eps)
    return np.abs(approx - exact) / scale


def batchlas_histories(ritz_history: np.ndarray, iterations_done: np.ndarray) -> list[np.ndarray]:
    stored_iters = ritz_history.shape[0]
    batch_size = ritz_history.shape[1]
    histories: list[np.ndarray] = []
    for batch in range(batch_size):
        history_iters = int(iterations_done[batch]) if iterations_done.size > batch else stored_iters
        history_iters = max(1, min(history_iters, stored_iters))
        histories.append(np.asarray(ritz_history[:history_iters, batch, :], dtype=np.float64))
    return histories


def run_scipy_lobpcg_batch(
    matrices: list[sp.csr_matrix],
    neigs: int,
    find_largest: bool,
    maxiter: int,
    seed: int,
    tol: Optional[float],
) -> SolverTrace:
    values: list[np.ndarray] = []
    histories: list[np.ndarray] = []

    for batch, matrix in enumerate(matrices):
        rng = np.random.default_rng(seed + batch)
        initial_guess = rng.normal(size=(matrix.shape[0], neigs))
        initial_guess, _ = np.linalg.qr(initial_guess, mode="reduced")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = spla.lobpcg(
                matrix,
                initial_guess,
                largest=find_largest,
                maxiter=maxiter,
                tol=tol,
                retLambdaHistory=True,
            )

        eigenvalues = np.asarray(result[0], dtype=np.float64)
        lambda_history = np.asarray(result[2], dtype=np.float64) if len(result) >= 3 else eigenvalues[None, :]
        if lambda_history.ndim == 1:
            lambda_history = lambda_history[:, None]

        values.append(eigenvalues)
        histories.append(lambda_history)

        for warning in caught:
            message = str(warning.message).strip().replace("\n", " ")
            print(f"SciPy lobpcg batch {batch} warning: {message}")

    return SolverTrace(
        name="SciPy lobpcg",
        final_values=np.stack(values, axis=0),
        histories=histories,
        line_style="--",
    )


def plot_history(
    references: np.ndarray,
    traces: list[SolverTrace],
    used_iluk: bool,
    out_path: Path,
) -> None:
    batch_size, neigs = references.shape
    fig, axes = plt.subplots(batch_size, 1, figsize=(9, 3.5 * batch_size), sharex=True, squeeze=False)
    axes_flat = axes[:, 0]
    colors = plt.get_cmap("tab10")(np.arange(neigs) % 10)

    for batch, ax in enumerate(axes_flat):
        title_lines = [f"Batch {batch}"]
        for trace in traces:
            history = trace.histories[batch]
            x = np.arange(history.shape[0])
            history_errors = relative_error(history, references[batch][None, :])
            final_errors = relative_error(trace.final_values[batch], references[batch])

            for eig in range(neigs):
                ax.semilogy(
                    x,
                    history_errors[:, eig],
                    color=colors[eig],
                    linestyle=trace.line_style,
                    linewidth=2.0,
                )

            title_lines.append(
                f"{trace.name} relerr={np.array2string(final_errors, precision=3, suppress_small=False)}"
            )

        ax.set_title("\n".join(title_lines))
        ax.set_ylabel("Relative error")
        ax.grid(True, alpha=0.25)

    axes_flat[-1].set_xlabel("Iteration")
    eig_handles = [
        Line2D([0], [0], color=colors[eig], linewidth=2.0, label=f"eig {eig}")
        for eig in range(neigs)
    ]
    eig_legend = axes_flat[0].legend(handles=eig_handles, ncols=min(neigs, 4), loc="upper right", title="Eigenvalue")
    if len(traces) > 1:
        solver_handles = [
            Line2D([0], [0], color="black", linestyle=trace.line_style, linewidth=2.0, label=trace.name)
            for trace in traces
        ]
        axes_flat[0].add_artist(eig_legend)
        axes_flat[0].legend(handles=solver_handles, loc="upper left", title="Solver")
    title = "BatchLAS syevx Ritz-value relative error on symmetric CSR matrices"
    if used_iluk:
        title += " with ILUK preconditioning"
    if len(traces) > 1:
        title += " compared with SciPy lobpcg"
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def print_solver_summary(name: str, final_values: np.ndarray, references: np.ndarray) -> None:
    print(f"{name} final eigenvalues:")
    for batch, batch_values in enumerate(final_values):
        print(f"  batch {batch}: {np.array2string(batch_values, precision=6, suppress_small=True)}")
        print(
            "  relative error vs. exact: "
            f"{np.array2string(relative_error(batch_values, references[batch]), precision=6, suppress_small=False)}"
        )


def main(argv: Optional[Iterable[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--batch-size", type=int, default=3)
    ap.add_argument("--neigs", type=int, default=4)
    ap.add_argument("--iterations", type=int, default=40)
    ap.add_argument("--density", type=float, default=0.06, help="Extra off-diagonal coupling density above the tridiagonal backbone")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--device", default="gpu")
    ap.add_argument("--extra-directions", type=int, default=2, help="Number of extra Lanczos directions to store for Ritz-value tracking")
    ap.add_argument("--find-smallest", action="store_true", help="Track the smallest eigenvalues instead of the largest ones")
    ap.add_argument("--use-iluk", action="store_true", help="Build and use an ILUK preconditioner")
    ap.add_argument("--iluk-levels-of-fill", type=int, default=0)
    ap.add_argument("--iluk-drop-tolerance", type=float, default=1e-4)
    ap.add_argument("--iluk-fill-factor", type=float, default=10.0)
    ap.add_argument("--iluk-diagonal-shift", type=float, default=1e-8)
    ap.add_argument("--compare-scipy-lobpcg", action="store_true", help="Overlay SciPy lobpcg convergence traces on the same plots")
    ap.add_argument("--scipy-seed", type=int, default=None, help="Seed for SciPy lobpcg initial guesses (defaults to seed + 1000)")
    ap.add_argument("--scipy-tol", type=float, default=None, help="Tolerance passed to SciPy lobpcg")
    ap.add_argument("--scipy-maxiter", type=int, default=None, help="Maximum iterations for SciPy lobpcg (defaults to --iterations)")
    ap.add_argument("--out", default="output/playground/syevx_history_batch.png")
    args = ap.parse_args(list(argv) if argv is not None else None)

    matrices = make_symmetric_csr_batch(args.n, args.batch_size, args.density, args.seed)
    references = np.stack(
        [select_reference_eigenvalues(matrix, args.neigs, not args.find_smallest) for matrix in matrices],
        axis=0,
    )
    options = bl.SyevxOptions(
        iterations=args.iterations,
        find_largest=not args.find_smallest,
        store_every=1,
        store_ritz_values=True,
        extra_directions=args.extra_directions,
    )
    preconditioner = None
    if args.use_iluk:
        iluk_options = bl.ILUKOptions(
            levels_of_fill=args.iluk_levels_of_fill,
            drop_tolerance=args.iluk_drop_tolerance,
            fill_factor=args.iluk_fill_factor,
            diagonal_shift=args.iluk_diagonal_shift,
        )
        print(
            "Building ILUK preconditioner "
            f"(level={iluk_options.levels_of_fill}, "
            f"drop_tolerance={iluk_options.drop_tolerance}, "
            f"fill_factor={iluk_options.fill_factor}, "
            f"diagonal_shift={iluk_options.diagonal_shift})"
        )
        preconditioner = bl.iluk_factorize(
            matrices,
            options=iluk_options,
            backend=args.backend,
            device=args.device,
        )

    values, history = bl.syevx(
        matrices,
        args.neigs,
        compute_vectors=False,
        options=options,
        backend=args.backend,
        device=args.device,
        return_history=True,
        preconditioner=preconditioner,
    )

    values = np.asarray(values)
    if values.ndim == 1:
        values = values[None, :]

    ritz_history = history["ritz_value_history"]
    if ritz_history is None:
        raise RuntimeError("syevx did not return Ritz-value history; ensure store_ritz_values=True")
    ritz_history = np.asarray(ritz_history)
    iterations_done = np.asarray(history["iterations_done"])
    traces = [
        SolverTrace(
            name="BatchLAS syevx",
            final_values=values,
            histories=batchlas_histories(ritz_history, iterations_done),
        )
    ]

    if args.compare_scipy_lobpcg:
        scipy_seed = args.scipy_seed if args.scipy_seed is not None else args.seed + 1000
        scipy_maxiter = args.scipy_maxiter if args.scipy_maxiter is not None else args.iterations
        if args.use_iluk:
            print("Note: BatchLAS uses ILUK here; SciPy lobpcg is run without a matching preconditioner.")
        traces.append(
            run_scipy_lobpcg_batch(
                matrices,
                args.neigs,
                find_largest=not args.find_smallest,
                maxiter=scipy_maxiter,
                seed=scipy_seed,
                tol=args.scipy_tol,
            )
        )

    out_path = Path(args.out)
    plot_history(
        references=references,
        traces=traces,
        used_iluk=args.use_iluk,
        out_path=out_path,
    )

    print(f"Wrote plot to {out_path}")
    for trace in traces:
        print_solver_summary(trace.name, trace.final_values, references)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
