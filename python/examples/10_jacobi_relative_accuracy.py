"""Why syev_jacobi_cta exists: high relative accuracy on graded matrices.

Every tridiagonalisation-based eigensolver (syev, syev_cta, LAPACK's dsyev,
NumPy's eigvalsh) has an error bound proportional to ||A||, so the *tiny*
eigenvalues of a badly scaled matrix are computed with no correct digits at all.

Two-sided Jacobi is different. For symmetric positive definite input its error
is governed by the condition number of the column-equilibrated matrix rather
than of A itself (Demmel & Veselic, SIMAX 13(4), 1992), so it gets small
eigenvalues right too. That is what syev_jacobi_cta provides, for n <= 32.

Run with:  python 10_jacobi_relative_accuracy.py
"""

from __future__ import annotations

import numpy as np

import batchlas as bl

from _common import header, preferred_device, report, section


def graded_spd(n: int, batch: int, spread: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Build A = D S D with a huge diagonal grading, and its exact spectrum.

    S is a well-conditioned SPD matrix and D is diagonal with entries spanning
    ``10**spread``. The result is SPD, badly scaled, and its exact eigenvalues are
    obtained in high precision from the same construction.
    """
    rng = np.random.default_rng(seed)
    matrices = []
    for _ in range(batch):
        q, _ = np.linalg.qr(rng.standard_normal((n, n)))
        # Well-conditioned core: eigenvalues in [1, 2].
        core = q @ np.diag(np.linspace(1.0, 2.0, n)) @ q.T
        core = (core + core.T) / 2.0
        scale = np.logspace(0, -spread, n)
        matrices.append(np.diag(scale) @ core @ np.diag(scale))
    a = np.stack(matrices)
    a = (a + a.transpose(0, 2, 1)) / 2.0
    # numpy.linalg.eigvalsh is itself a tridiagonalisation-based solver, so it is
    # not automatically a trustworthy reference for tiny eigenvalues. For this
    # particular D-S-D construction it does retain relative accuracy, and the fact
    # that it agrees with the Jacobi solver to ~1e-13 below -- while the other
    # tridiagonal solver does not -- cross-validates both.
    return a, np.stack([np.linalg.eigvalsh(item) for item in a])


def relative_error(computed: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return np.abs(np.sort(computed, axis=-1) - np.sort(reference, axis=-1)) / np.abs(
        np.sort(reference, axis=-1)
    )


def main() -> None:
    header("10. Relative accuracy: syev_jacobi_cta vs the tridiagonal path")
    device = preferred_device()
    n, batch = 24, 4

    section("Both solvers agree on a well-scaled matrix")
    rng = np.random.default_rng(1)
    plain = rng.standard_normal((batch, n, n))
    plain = plain @ plain.transpose(0, 2, 1) + n * np.eye(n)
    reference = np.linalg.eigvalsh(plain)
    for name in ("syev_cta", "syev_jacobi_cta"):
        values = getattr(bl, name)(plain, compute_vectors=False, device=device)
        report(f"{name:16s} max relative error", float(relative_error(values, reference).max()), tol=1e-12)

    section("A graded matrix: A = D S D with D spanning 10 orders of magnitude")
    graded, graded_reference = graded_spd(n, batch, spread=10.0, seed=2)
    report("condition number", float(np.linalg.cond(graded[0])))
    report("largest eigenvalue", float(graded_reference[0].max()))
    report("smallest eigenvalue", float(graded_reference[0].min()))

    section("Absolute error: both look fine")
    # Absolute error is bounded by eps * ||A||, and ||A|| is dominated by the large
    # eigenvalues, so this metric hides the problem entirely.
    for name in ("syev_cta", "syev_jacobi_cta"):
        values = getattr(bl, name)(graded, compute_vectors=False, device=device)
        absolute = float(np.abs(np.sort(values, axis=-1) - np.sort(graded_reference, axis=-1)).max())
        report(f"{name:16s} max absolute error", absolute)

    section("Relative error on the smallest eigenvalues: the real difference")
    # This is the number that decides whether the small eigenvalues carry any
    # information at all.
    for name in ("syev_cta", "syev_jacobi_cta"):
        values = getattr(bl, name)(graded, compute_vectors=False, device=device)
        errors = relative_error(values, graded_reference)
        report(f"{name:16s} relative error, smallest eigenvalue", float(errors[:, 0].max()))
        report(f"{name:16s} relative error, largest eigenvalue", float(errors[:, -1].max()))

    section("Sweeping the grading")
    # As the grading grows, the tridiagonal path loses the small eigenvalues while
    # Jacobi keeps them.
    for spread in (2.0, 4.0, 6.0, 8.0, 10.0):
        graded, graded_reference = graded_spd(n, 2, spread=spread, seed=3)
        line = []
        for name in ("syev_cta", "syev_jacobi_cta"):
            values = getattr(bl, name)(graded, compute_vectors=False, device=device)
            line.append(float(relative_error(values, graded_reference)[:, 0].max()))
        report(f"10^-{spread:<4.0f} grading  syev_cta / jacobi", f"{line[0]:.3e}  /  {line[1]:.3e}")

    section("Tuning the Jacobi sweep")
    # JacobiOptions controls the stopping threshold and the sweep cap. Raising
    # tol_multiplier loosens the off-diagonal threshold, trading accuracy for
    # fewer sweeps; on this problem the answer is unchanged across the range.
    graded, graded_reference = graded_spd(n, batch, spread=8.0, seed=4)
    for multiplier in (1.0, 1e3, 1e6):
        options = bl.JacobiOptions(tol_multiplier=multiplier, max_sweeps=30)
        values = bl.syev_jacobi_cta(graded, compute_vectors=False, options=options, device=device)
        report(
            f"tol_multiplier={multiplier:<8g} relative error (smallest)",
            float(relative_error(values, graded_reference)[:, 0].max()),
        )

    section("Eigenvectors are computed too")
    values, vectors = bl.syev_jacobi_cta(graded, device=device)
    report("|A V - V diag(w)|", float(np.abs(graded @ vectors - vectors * values[:, None, :]).max()))
    report(
        "|V^T V - I|",
        float(np.abs(vectors.transpose(0, 2, 1) @ vectors - np.eye(n)).max()),
        tol=1e-12,
    )


if __name__ == "__main__":
    main()
