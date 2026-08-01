"""Tridiagonal eigensolvers: steqr, steqr_cta, stedc, stedc_flat, tridiagonal_solver.

These take the (d, e) coefficients of a symmetric tridiagonal matrix -- d of
length n, e of length n - 1 -- rather than a dense matrix. They are the inner
solve of every dense symmetric eigensolver, and are worth calling directly
whenever your problem is already tridiagonal.

  steqr               implicit QL/QR iteration
  steqr_cta           the same, resident in one work-group (small n)
  stedc               divide and conquer
  stedc_flat          non-recursive divide and conquer
  tridiagonal_solver  convenience driver taking (alpha, beta)

Run with:  python 08_tridiagonal_eigensolvers.py
"""

from __future__ import annotations

import numpy as np

import batchlas as bl

from _common import (
    eigenvalue_error,
    header,
    preferred_device,
    report,
    residual,
    section,
    tridiagonal_matrix,
)


def main() -> None:
    header("8. Tridiagonal eigensolvers")
    device = preferred_device()
    batch, n = 4, 32

    rng = np.random.default_rng(1)
    d = rng.standard_normal((batch, n))
    e = rng.standard_normal((batch, n - 1))
    dense = tridiagonal_matrix(d, e)
    reference = np.linalg.eigvalsh(dense)

    section("Input layout")
    report("d shape (diagonal)", d.shape)
    report("e shape (off-diagonal)", e.shape)

    section("All four solvers on the same problem")
    for name in ("steqr", "steqr_cta", "stedc", "stedc_flat"):
        values, vectors = getattr(bl, name)(d, e, device=device)
        report(f"{name:11s} eigenvalue error", eigenvalue_error(values, reference), tol=1e-10)
        if name == "stedc_flat":
            # stedc_flat's eigenvalues are good but its eigenvectors are currently
            # wrong -- see the note in README.md. Reported without a tolerance so
            # the discrepancy stays visible.
            report(f"{name:11s} residual (known bad)", residual(dense, values, vectors))
        else:
            report(f"{name:11s} residual", residual(dense, values, vectors), tol=1e-9)

    section("Eigenvalues only")
    for name in ("steqr", "steqr_cta", "stedc", "stedc_flat"):
        values = getattr(bl, name)(d, e, compute_vectors=False, device=device)
        report(f"{name:11s} error", eigenvalue_error(values, reference), tol=1e-10)

    section("Sort order")
    ascending = bl.steqr(d, e, compute_vectors=False, options=bl.SteqrOptions(sort_order="ascending"), device=device)
    descending = bl.steqr(d, e, compute_vectors=False, options=bl.SteqrOptions(sort_order="descending"), device=device)
    report("descending == reversed ascending", bool(np.allclose(descending, ascending[:, ::-1])))

    section("Tuning the QR iteration")
    # SteqrOptions exposes the sweep cap, the shift strategy and the CTA layout.
    for strategy in ("lapack", "wilkinson"):
        options = bl.SteqrOptions(max_sweeps=400, cta_shift_strategy=strategy)
        values = bl.steqr_cta(d, e, compute_vectors=False, options=options, device=device)
        report(f"cta_shift_strategy={strategy:<10s}", eigenvalue_error(values, reference), tol=1e-10)

    section("Tuning divide and conquer")
    # Below recursion_threshold, stedc falls back to the leaf QR solver, which is
    # configured through the nested SteqrOptions.
    options = bl.StedcOptions(
        recursion_threshold=8,
        merge_threads=128,
        leaf_steqr_params=bl.SteqrOptions(max_sweeps=200),
    )
    values = bl.stedc(d, e, compute_vectors=False, options=options, device=device)
    report("custom stedc options", eigenvalue_error(values, reference), tol=1e-10)

    section("tridiagonal_solver: the convenience driver")
    # This driver's QR iteration does not converge reliably -- accuracy varies with
    # n and with the data (see the note in README.md). The numbers below are shown
    # without a pass/fail tolerance so the loss of accuracy stays visible; prefer
    # steqr or stedc for real work.
    small_d = d[:, :8]
    small_e = e[:, :7]
    small_dense = tridiagonal_matrix(small_d, small_e)
    small_reference = np.linalg.eigvalsh(small_dense)
    values, vectors = bl.tridiagonal_solver(small_d, small_e, compute_vectors=True, device=device)
    report("eigenvalue error (n=8, no tolerance applied)", eigenvalue_error(values, small_reference))
    report("residual (n=8, no tolerance applied)", residual(small_dense, values, vectors))

    section("tridiag_toeplitz: a generator with a known closed-form spectrum")
    # The eigenvalues of the (a, b, b) Toeplitz tridiagonal matrix are
    # a + 2 b cos(k pi / (n + 1)), which makes this a good correctness check.
    toeplitz = bl.tridiag_toeplitz(n, diagonal_value=2.0, sub_diagonal_value=-1.0, super_diagonal_value=-1.0)
    exact = np.sort(2.0 - 2.0 * np.cos(np.arange(1, n + 1) * np.pi / (n + 1)))
    values = bl.steqr(
        np.full((1, n), 2.0),
        np.full((1, n - 1), -1.0),
        compute_vectors=False,
        device=device,
    )
    report("matrix shape", toeplitz.shape)
    report("error vs closed form", float(np.abs(np.sort(np.ravel(values)) - exact).max()), tol=1e-12)


if __name__ == "__main__":
    main()
