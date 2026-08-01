"""QR factorization and orthogonalization: geqrf, orgqr, ormqr, ortho, ortho_metric.

Run with:  python 04_qr_and_orthogonalization.py
"""

from __future__ import annotations

import numpy as np

import batchlas as bl

from _common import (
    batched_general,
    header,
    orthogonality_error,
    preferred_device,
    report,
    section,
)


def main() -> None:
    header("4. QR factorization and orthogonalization")
    device = preferred_device()
    batch, m, n = 4, 32, 16

    a = batched_general(batch, m, n, seed=1)

    section("geqrf: QR in packed reflector form")
    # geqrf returns (qr, tau): R in the upper triangle of qr, and the Householder
    # reflectors that generate Q stored below it plus the tau scalars.
    qr, tau = bl.geqrf(a, device=device)
    r = np.triu(qr[:, :n, :])
    report("qr shape", qr.shape)
    report("tau shape", tau.shape)

    section("orgqr: materialise Q from the packed form")
    q = bl.orgqr(qr, tau, device=device)[:, :, :n]
    report("|Q^T Q - I|", orthogonality_error(q), tol=1e-10)
    report("|Q R - A|", float(np.abs(q @ r - a).max()), tol=1e-10)

    section("ormqr: apply Q without ever forming it")
    # For tall matrices this is much cheaper than orgqr followed by a gemm.
    c = batched_general(batch, m, 5, seed=2)
    # ormqr applies the full m-by-m Q, so its result has m rows; orgqr above only
    # materialised the first n columns of Q, which reproduce the first n rows.
    applied = bl.ormqr(qr, c, tau, side="left", trans="t", device=device)
    report("ormqr result shape", applied.shape)
    report("|Q^T C - ormqr(...)|", float(np.abs(q.transpose(0, 2, 1) @ c - applied[:, :n, :]).max()), tol=1e-10)

    section("ortho: orthonormalise a block of vectors")
    # Several algorithms are available; they trade robustness against speed.
    # "householder" is also accepted but is currently unreliable on the CUDA
    # backend when an earlier call in the same process used geqrf -- see the note
    # in README.md.
    block = batched_general(batch, m, 8, seed=3)
    for algorithm in ("chol2", "cgs2", "svqb", "svqb2", "shiftchol3"):
        try:
            q_block = bl.ortho(block, algorithm=algorithm, device=device)
            report(f"{algorithm:8s} |Q^T Q - I|", orthogonality_error(q_block), tol=1e-8)
        except (RuntimeError, NotImplementedError, ValueError) as exc:
            report(f"{algorithm:8s}", f"unavailable ({type(exc).__name__})")

    section("ortho on an ill-conditioned block")
    # A block whose columns nearly coincide is where the cheap algorithms lose
    # orthogonality first -- worth checking before picking one.
    near_singular = batched_general(batch, m, 4, seed=4)
    near_singular[:, :, 1] = near_singular[:, :, 0] + 1e-8 * near_singular[:, :, 1]
    # chol2 is expected to break down here (it squares the condition number);
    # that is exactly the trade-off this section is meant to show.
    for algorithm in ("chol2", "cgs2", "svqb2", "shiftchol3"):
        try:
            q_block = bl.ortho(near_singular, algorithm=algorithm, device=device)
            report(f"{algorithm:8s} |Q^T Q - I|", orthogonality_error(q_block))
        except (RuntimeError, NotImplementedError, ValueError) as exc:
            report(f"{algorithm:8s}", f"unavailable ({type(exc).__name__})")

    section("ortho_metric: orthogonalise against an existing basis")
    # M holds vectors you already trust (say, converged eigenvectors). The result
    # is orthonormal *and* orthogonal to every column of M -- the block step at
    # the heart of iterative eigensolvers. Note cols(A) + cols(M) <= dimension.
    basis = bl.ortho(batched_general(batch, m, 6, seed=5), algorithm="chol2", device=device)
    q_m = bl.ortho_metric(block, basis, algorithm="chol2", iterations=2, device=device)
    report("|Q^T Q - I|", orthogonality_error(q_m), tol=1e-8)
    report("|M^T Q| (should be ~0)", float(np.abs(basis.transpose(0, 2, 1) @ q_m).max()), tol=1e-8)


if __name__ == "__main__":
    main()
