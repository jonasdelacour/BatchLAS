"""Reduction to tridiagonal form: the sytrd family, one stage at a time.

Every dense symmetric eigensolver is built on the same idea -- reduce A to a
symmetric tridiagonal matrix with an orthogonal similarity transform, solve the
much cheaper tridiagonal problem, then transform the vectors back. BatchLAS
exposes each stage so you can benchmark or recombine them:

  sytrd_cta             dense -> tridiagonal in one work-group (n <= 32)
  sytrd_blocked         dense -> tridiagonal, blocked panel + BLAS-3 update
  sytrd_sy2sb           stage 1 of two-stage: dense -> band
  sytrd_sb2st           stage 2 of two-stage: band -> tridiagonal (bulge chasing)
  hetrd_hb2st           LAPACK-style alias of sytrd_sb2st
  sytrd_band_reduction  band -> tridiagonal via the BANDR1 blocked schedule

A similarity transform preserves the spectrum, so every stage is checked by
comparing eigenvalues against the original matrix.

Run with:  python 07_tridiagonal_reduction.py
"""

from __future__ import annotations

import numpy as np

import batchlas as bl

from _common import (
    band_to_dense,
    batched_symmetric,
    eigenvalue_error,
    header,
    preferred_device,
    report,
    section,
    tridiagonal_matrix,
)


def main() -> None:
    header("7. Reduction to tridiagonal form")
    device = preferred_device()
    batch = 4

    section("sytrd_cta: one work-group per matrix (n <= 32)")
    # Returns (a, d, e, tau): d and e are the tridiagonal coefficients, while a
    # and tau hold the Householder reflectors in the packed SYTD2 layout.
    small = batched_symmetric(batch, 16, seed=1)
    small_reference = np.linalg.eigvalsh(small)
    reduced, d, e, tau = bl.sytrd_cta(small, uplo="lower", device=device)
    report("d shape / e shape", (d.shape, e.shape))
    report(
        "spectrum error",
        eigenvalue_error(np.linalg.eigvalsh(tridiagonal_matrix(d, e)), small_reference),
        tol=1e-10,
    )

    section("sytrd_blocked: blocked panel plus BLAS-3 trailing update")
    medium = batched_symmetric(batch, 96, seed=2)
    medium_reference = np.linalg.eigvalsh(medium)
    for block_size in (16, 32):
        _, d, e, tau = bl.sytrd_blocked(medium, uplo="lower", block_size=block_size, device=device)
        report(
            f"block_size={block_size:<3d} spectrum error",
            eigenvalue_error(np.linalg.eigvalsh(tridiagonal_matrix(d, e)), medium_reference),
            tol=1e-8,
        )

    section("Two-stage, step 1 -- sytrd_sy2sb: dense to band")
    # The band is returned in LAPACK band storage: shape (kd + 1, n), where
    # AB[i, j] holds A[j + i, j] for the lower triangle.
    kd = 8
    reduced, ab, tau = bl.sytrd_sy2sb(medium, kd, uplo="lower", device=device)
    report("band storage shape", ab.shape)
    report(
        "spectrum error",
        eigenvalue_error(np.linalg.eigvalsh(band_to_dense(ab)), medium_reference),
        tol=1e-8,
    )

    section("Two-stage, step 2 -- sytrd_sb2st: band to tridiagonal")
    d, e, tau = bl.sytrd_sb2st(ab, kd, uplo="lower", block_size=16, device=device)
    report(
        "spectrum error",
        eigenvalue_error(np.linalg.eigvalsh(tridiagonal_matrix(d, e)), medium_reference),
        tol=1e-8,
    )

    section("hetrd_hb2st: the same routine under its LAPACK name")
    d_alias, e_alias, _ = bl.hetrd_hb2st(ab, kd, uplo="lower", block_size=16, device=device)
    report("identical to sytrd_sb2st", bool(np.array_equal(d, d_alias) and np.array_equal(e, e_alias)))

    section("sytrd_band_reduction: the BANDR1 blocked schedule")
    # This alternative band -> tridiagonal path takes a per-sweep schedule:
    # d_seq is how many diagonals to eliminate per sweep, block_size_seq the
    # block size to use. A 0 or an omitted entry means "use the default".
    d_b, e_b, _ = bl.sytrd_band_reduction(ab, kd, uplo="lower", device=device)
    report(
        "default schedule",
        eigenvalue_error(np.linalg.eigvalsh(tridiagonal_matrix(d_b, e_b)), medium_reference),
        tol=1e-8,
    )

    options = bl.SytrdBandReductionOptions(d_seq=[4, 2], block_size_seq=[16, 16], max_sweeps=4)
    d_b, e_b, _ = bl.sytrd_band_reduction(ab, kd, uplo="lower", options=options, device=device)
    report(
        "explicit schedule",
        eigenvalue_error(np.linalg.eigvalsh(tridiagonal_matrix(d_b, e_b)), medium_reference),
        tol=1e-8,
    )

    section("Putting it together: two-stage reduction + tridiagonal solve")
    # This is syev_two_stage spelled out by hand.
    _, ab, _ = bl.sytrd_sy2sb(medium, kd, uplo="lower", device=device)
    d, e, _ = bl.sytrd_sb2st(ab, kd, uplo="lower", block_size=16, device=device)
    values = bl.stedc(d, e, compute_vectors=False, device=device)
    report("eigenvalue error", eigenvalue_error(values, medium_reference), tol=1e-8)

    section("Complex Hermitian input")
    rng = np.random.default_rng(3)
    z = rng.standard_normal((batch, 16, 16)) + 1j * rng.standard_normal((batch, 16, 16))
    z = (z + z.conj().transpose(0, 2, 1)) / 2.0
    _, dz, ez, _ = bl.sytrd_cta(z, uplo="lower", device=device)
    # For Hermitian input d is real, but the subdiagonal e carries a phase that is
    # absorbed into the Householder reflectors. The real tridiagonal matrix whose
    # spectrum matches A is built from |e|, not from e.real.
    report("max |imag(d)|", float(np.abs(dz.imag).max()))
    report("max |imag(e)|", float(np.abs(ez.imag).max()))
    report(
        "spectrum error using |e|",
        eigenvalue_error(np.linalg.eigvalsh(tridiagonal_matrix(dz.real, np.abs(ez))), np.linalg.eigvalsh(z)),
        tol=1e-10,
    )


if __name__ == "__main__":
    main()
