"""Factorizations and linear solves: potrf, getrf/getrs, getri, inv, trsm.

Run with:  python 03_linear_solvers.py
"""

from __future__ import annotations

import numpy as np

import batchlas as bl

from _common import batched_general, batched_spd, header, preferred_device, report, section


def main() -> None:
    header("3. Factorizations and linear solves")
    device = preferred_device()
    batch, n, nrhs = 4, 24, 3

    section("potrf: Cholesky factorization of an SPD batch")
    spd = batched_spd(batch, n, seed=1)
    lower = bl.potrf(spd, uplo="lower", device=device)
    # Only the requested triangle is meaningful; mask the other half before checking.
    lower = np.tril(lower)
    report("|L L^T - A|", float(np.abs(lower @ lower.transpose(0, 2, 1) - spd).max()), tol=1e-8)

    section("Solving with the Cholesky factor via two triangular solves")
    rhs = batched_general(batch, n, nrhs, seed=2)
    y = bl.trsm(lower, rhs, side="left", uplo="lower", trans_a="n", device=device)
    x = bl.trsm(lower, y, side="left", uplo="lower", trans_a="t", device=device)
    report("|A x - b|", float(np.abs(spd @ x - rhs).max()), tol=1e-6)

    section("getrf: LU factorization with partial pivoting")
    a = batched_general(batch, n, n, seed=3) + n * np.eye(n)
    lu, pivots = bl.getrf(a, device=device)
    report("pivots shape", pivots.shape)

    section("getrs: reuse one factorization for many right-hand sides")
    # This is the main reason to keep getrf and getrs separate: factor once, solve often.
    x = bl.getrs(lu, rhs, pivots, device=device)
    report("|A x - b|", float(np.abs(a @ x - rhs).max()), tol=1e-8)

    section("getrs with the transposed system")
    xt = bl.getrs(lu, rhs, pivots, trans_a="t", device=device)
    report("|A^T x - b|", float(np.abs(a.transpose(0, 2, 1) @ xt - rhs).max()), tol=1e-8)

    section("getri: explicit inverse from an existing LU")
    a_inv = bl.getri(lu, pivots, device=device)
    report("|A A^-1 - I|", float(np.abs(a @ a_inv - np.eye(n)).max()), tol=1e-8)

    section("inv: the one-shot convenience wrapper")
    # Prefer getrf + getrs over inv when you actually want to solve a system --
    # forming the inverse is both slower and less accurate.
    report("|inv(A) - getri(...)|", float(np.abs(bl.inv(a, device=device) - a_inv).max()), tol=1e-8)

    section("Complex input works the same way")
    rng = np.random.default_rng(4)
    az = (rng.standard_normal((batch, n, n)) + 1j * rng.standard_normal((batch, n, n))) + n * np.eye(n)
    bz = rng.standard_normal((batch, n, nrhs)) + 1j * rng.standard_normal((batch, n, nrhs))
    lu_z, piv_z = bl.getrf(az, device=device)
    xz = bl.getrs(lu_z, bz, piv_z, device=device)
    report("complex |A x - b|", float(np.abs(az @ xz - bz).max()), tol=1e-8)


if __name__ == "__main__":
    main()
