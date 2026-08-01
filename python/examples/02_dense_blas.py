"""Batched dense BLAS: gemm, gemv, symm, syrk, syr2k, trmm, trsm.

Also shows heterogeneous batches, where every matrix in the batch may have a
different shape -- something plain NumPy broadcasting cannot express.

Run with:  python 02_dense_blas.py
"""

from __future__ import annotations

import numpy as np

import batchlas as bl

from _common import batched_general, batched_symmetric, header, preferred_device, report, section


def main() -> None:
    header("2. Batched dense BLAS")
    device = preferred_device()
    batch, n = 4, 32

    a = batched_general(batch, n, n, seed=1)
    b = batched_general(batch, n, n, seed=2)
    c = batched_general(batch, n, n, seed=3)

    section("gemm: C <- alpha * op(A) op(B) + beta * C")
    # Note beta needs an existing C, which you supply through out=.
    got = bl.gemm(a, b, alpha=1.5, beta=-0.25, out=c.copy(), device=device)
    report("error", float(np.abs(got - (1.5 * (a @ b) - 0.25 * c)).max()), tol=1e-10)

    section("gemv: y <- alpha * op(A) x + beta * y")
    x = batched_general(batch, n, 1, seed=4)[:, :, 0]
    y = batched_general(batch, n, 1, seed=5)[:, :, 0]
    got = bl.gemv(a, x, alpha=2.0, beta=1.0, out=y.copy(), device=device)
    report("error", float(np.abs(got - (2.0 * np.einsum("bij,bj->bi", a, x) + y)).max()), tol=1e-10)

    section("symm: C <- alpha * A B + beta * C with symmetric A")
    # Only the referenced triangle of A is read, so the strict other half is free
    # to hold garbage -- here we verify against the symmetrised matrix.
    s = batched_symmetric(batch, n, seed=6)
    got = bl.symm(s, b, alpha=1.0, side="left", uplo="lower", device=device)
    report("error", float(np.abs(got - s @ b).max()), tol=1e-10)

    section("syrk: C <- alpha * A A^T + beta * C (lower triangle)")
    got = bl.syrk(a, alpha=1.0, uplo="lower", device=device)
    report("error", float(np.abs(np.tril(got) - np.tril(a @ a.transpose(0, 2, 1))).max()), tol=1e-10)

    section("syr2k: C <- alpha (A B^T + B A^T) + beta * C (lower triangle)")
    got = bl.syr2k(a, b, alpha=1.0, uplo="lower", device=device)
    expected = a @ b.transpose(0, 2, 1) + b @ a.transpose(0, 2, 1)
    report("error", float(np.abs(np.tril(got) - np.tril(expected)).max()), tol=1e-10)

    section("trmm: B <- alpha * op(A) B with triangular A")
    lower = np.tril(a)
    got = bl.trmm(lower, b, alpha=1.0, side="left", uplo="lower", diag="non_unit", device=device)
    report("error", float(np.abs(got - lower @ b).max()), tol=1e-10)

    section("trsm: solve op(A) X = alpha * B for triangular A")
    # Bias the diagonal so the triangular system is well conditioned.
    lower = np.tril(a) + n * np.eye(n)
    x_solved = bl.trsm(lower, b, alpha=1.0, side="left", uplo="lower", diag="non_unit", device=device)
    report("residual", float(np.abs(lower @ x_solved - b).max()), tol=1e-9)

    section("Heterogeneous batches: one call, differently shaped matrices")
    # Pass a *list* of 2-D arrays instead of a 3-D array. Results come back as a
    # list too, one entry per problem.
    rng = np.random.default_rng(7)
    lhs = [rng.standard_normal((m, k)) for m, k in ((3, 5), (7, 2), (4, 4))]
    rhs = [rng.standard_normal((k, p)) for k, p in ((5, 6), (2, 3), (4, 4))]
    products = bl.gemm_heterogeneous(lhs, rhs, device=device)
    worst = max(float(np.abs(products[i][: l.shape[0], : r.shape[1]] - l @ r).max())
                for i, (l, r) in enumerate(zip(lhs, rhs)))
    report("shapes in", [f"{l.shape}x{r.shape}" for l, r in zip(lhs, rhs)])
    report("error", worst, tol=1e-10)

    section("Mixed precision: float32 storage, higher-precision accumulation")
    a32 = a.astype(np.float32)
    b32 = b.astype(np.float32)
    for precision in ("default", "tf32"):
        try:
            got = bl.gemm(a32, b32, compute_precision=precision, device=device)
            report(f"{precision} error vs float64", float(np.abs(got - a @ b).max()))
        except (RuntimeError, NotImplementedError, ValueError) as exc:
            report(f"{precision}", f"unavailable ({type(exc).__name__})")


if __name__ == "__main__":
    main()
