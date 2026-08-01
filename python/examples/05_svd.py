"""Singular value decomposition: gesvd and its variants, plus the gebrd/bdsqr pipeline.

BatchLAS exposes three SVD drivers with different sweet spots:

  gesvd          general-purpose driver (vendor path where available)
  gesvd_blocked  native blocked path for medium and large matrices
  gesvd_cta      one work-group per matrix, for very small matrices (n <= 32)

The underlying two-step pipeline (bidiagonal reduction, then bidiagonal QR) is
available separately as gebrd_* and bdsqr, which is what the last sections use.

Run with:  python 05_svd.py
"""

from __future__ import annotations

import numpy as np

import batchlas as bl

from _common import batched_general, header, preferred_device, report, section


def bidiagonal_singular_values(d: np.ndarray, e: np.ndarray) -> np.ndarray:
    """Singular values of the upper bidiagonal matrices described by (d, e)."""
    return np.stack([
        np.linalg.svd(np.diag(d[i]) + np.diag(e[i], 1), compute_uv=False)
        for i in range(d.shape[0])
    ])


def main() -> None:
    header("5. Singular value decomposition")
    device = preferred_device()
    batch = 4

    section("gesvd: A = U diag(s) Vh")
    a = batched_general(batch, 24, 24, seed=1)
    u, s, vh = bl.gesvd(a, device=device)
    reference = np.linalg.svd(a, compute_uv=False)
    report("singular value error", float(np.abs(np.sort(s)[:, ::-1] - reference).max()), tol=1e-9)
    report("|U diag(s) Vh - A|", float(np.abs(u @ (s[:, :, None] * vh) - a).max()), tol=1e-9)
    report("|U^T U - I|", float(np.abs(u.transpose(0, 2, 1) @ u - np.eye(24)).max()), tol=1e-9)

    section("Values only: skip the vectors when you do not need them")
    values = bl.gesvd(a, compute_vectors=False, device=device)
    report("shape", values.shape)
    report("error", float(np.abs(np.sort(values)[:, ::-1] - reference).max()), tol=1e-9)

    section("gesvd_blocked: native blocked path")
    u_b, s_b, vh_b = bl.gesvd_blocked(a, device=device)
    report("singular value error", float(np.abs(np.sort(s_b)[:, ::-1] - reference).max()), tol=1e-9)
    report("|U diag(s) Vh - A|", float(np.abs(u_b @ (s_b[:, :, None] * vh_b) - a).max()), tol=1e-9)

    section("gesvd_cta: one work-group per matrix, for n <= 32")
    small = batched_general(batch, 16, 16, seed=2)
    small_reference = np.linalg.svd(small, compute_uv=False)
    u_c, s_c, vh_c = bl.gesvd_cta(small, device=device)
    report("singular value error", float(np.abs(np.sort(s_c)[:, ::-1] - small_reference).max()), tol=1e-9)
    report("|U diag(s) Vh - A|", float(np.abs(u_c @ (s_c[:, :, None] * vh_c) - small).max()), tol=1e-9)

    section("Symmetric input: declare uplo to take the Hermitian fast path")
    sym = batched_general(batch, 16, 16, seed=3)
    sym = (sym + sym.transpose(0, 2, 1)) / 2.0
    values = bl.gesvd(sym, compute_vectors=False, uplo="lower", device=device)
    report(
        "error vs |eigenvalues|",
        float(np.abs(np.sort(values)[:, ::-1] - np.sort(np.abs(np.linalg.eigvalsh(sym)))[:, ::-1]).max()),
        tol=1e-9,
    )

    section("gebrd_unblocked / gebrd_cta / gebrd_blocked: reduction to bidiagonal form")
    # All three return (a, d, e, tauq, taup): the bidiagonal coefficients d and e,
    # plus the reflectors that generate the orthogonal factors Q and P.
    for label, call, matrix in (
        ("unblocked", lambda m: bl.gebrd_unblocked(m, device=device), small),
        ("cta      ", lambda m: bl.gebrd_cta(m, device=device), small),
        ("blocked  ", lambda m: bl.gebrd_blocked(m, block_size=16, device=device), a),
    ):
        _, d, e, tauq, taup = call(matrix)
        expected = np.linalg.svd(matrix, compute_uv=False)
        got = np.sort(bidiagonal_singular_values(d, e))[:, ::-1]
        report(f"{label} singular value error", float(np.abs(got - expected).max()), tol=1e-9)

    section("bdsqr: bidiagonal QR iteration on the (d, e) coefficients")
    # gebrd + bdsqr is the SVD pipeline spelled out: reduce, then iterate.
    reduced, d, e, tauq, taup = bl.gebrd_cta(small, device=device)
    singular = bl.bdsqr(d, e, sort_desc=True, device=device)
    report("error", float(np.abs(singular - small_reference).max()), tol=1e-9)

    section("ormbr: apply the Q or P factor from gebrd")
    # This is how you turn the bidiagonal problem's vectors back into A's vectors.
    # Pass the *reduced* matrix returned by gebrd -- that is where the reflectors live.
    c = np.tile(np.eye(16), (batch, 1, 1))
    q_applied = bl.ormbr(reduced, tauq, c, vect="Q", side="left", trans="n", device=device)
    report("|Q^T Q - I|", float(np.abs(q_applied.transpose(0, 2, 1) @ q_applied - np.eye(16)).max()), tol=1e-9)


if __name__ == "__main__":
    main()
