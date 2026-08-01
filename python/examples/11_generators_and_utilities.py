"""Matrix generators and utilities: constructors, norms, condition numbers, scaling.

BatchLAS can build its own test matrices on the device, which saves a host-to-
device round trip and makes it easy to request a specific condition number.

Run with:  python 11_generators_and_utilities.py
"""

from __future__ import annotations

import numpy as np

import batchlas as bl

from _common import batched_general, header, preferred_device, report, section


def main() -> None:
    header("11. Generators and utilities")
    device = preferred_device()
    n = 8

    section("Structured constructors")
    report("zeros", bl.zeros(3, 4).shape)
    report("ones sum", float(bl.ones(3, 4).sum()))
    report("identity error", float(np.abs(bl.identity(n) - np.eye(n)).max()), tol=0.0)
    report("diagonal", np.diag(bl.diagonal(np.arange(1.0, 5.0))).tolist())

    triangular = bl.triangular(n, uplo="lower", diagonal_value=2.0, non_diagonal_value=1.0)
    report("triangular is lower", bool(np.allclose(triangular, np.tril(triangular))))
    report("triangular diagonal", sorted(set(np.diag(triangular).tolist())))
    report("triangular off-diagonal", sorted(set(triangular[np.tril_indices(n, -1)].tolist())))

    toeplitz = bl.tridiag_toeplitz(n, diagonal_value=2.0, sub_diagonal_value=-1.0, super_diagonal_value=-1.0)
    report("tridiag_toeplitz bandwidth", int(np.max(np.abs(np.nonzero(toeplitz)[0] - np.nonzero(toeplitz)[1]))))

    section("Batched constructors")
    report("identity batch shape", bl.identity(n, batch_size=5).shape)
    report("zeros batch shape", bl.zeros(3, 4, batch_size=5).shape)

    section("random: reproducible pseudo-random matrices")
    first = bl.random(n, n, seed=123)
    again = bl.random(n, n, seed=123)
    report("same seed gives same matrix", bool(np.array_equal(first, again)))
    report("different seed differs", not np.array_equal(first, bl.random(n, n, seed=124)))
    hermitian = bl.random(n, n, hermitian=True, seed=5)
    report("hermitian=True is symmetric", float(np.abs(hermitian - hermitian.T).max()), tol=1e-12)

    section("Generators with a requested condition number")
    # These build matrices whose log10 condition number (in a chosen norm) is the
    # value you ask for -- exactly what you want when studying numerical stability.
    for log10_kappa in (2.0, 6.0, 10.0):
        matrix = bl.random_with_log10_cond_metric(
            n=32, log10_kappa=log10_kappa, metric="spectral", seed=11, device=device
        )
        report(f"requested 10^{log10_kappa:<4.0f} -> actual", float(np.log10(np.linalg.cond(matrix))))

    section("The other conditioned generators")
    for name, extra in (
        ("random_hermitian_with_log10_cond_metric", {}),
        ("random_banded_with_log10_cond_metric", {"kd": 3}),
        ("random_hermitian_banded_with_log10_cond_metric", {"kd": 3}),
        ("random_tridiagonal_with_log10_cond_metric", {}),
        ("random_hermitian_tridiagonal_with_log10_cond_metric", {}),
    ):
        matrix = getattr(bl, name)(n=32, log10_kappa=4.0, seed=13, device=device, **extra)
        report(f"{name[7:40]:36s} log10(cond)", float(np.log10(np.linalg.cond(matrix))))

    section("norm: matrix norms, one value per batch entry")
    batch = batched_general(4, 6, 6, seed=2)
    for norm_type in ("fro", "1", "inf", "max"):
        values = bl.norm(batch, norm_type)
        report(f"norm '{norm_type}' shape", values.shape)

    expected = np.stack([np.linalg.norm(item, ord="fro") for item in batch])
    report("frobenius error", float(np.abs(bl.norm(batch, "fro") - expected).max()), tol=1e-12)

    section("cond: condition numbers")
    # The 'fro', '1' and 'inf' norms work for any square matrix.
    for norm_type, order in (("fro", "fro"), ("1", 1), ("inf", np.inf)):
        values = bl.cond(batch, norm_type, device=device)
        expected = np.stack([np.linalg.cond(item, p=order) for item in batch])
        report(f"cond '{norm_type}' relative error", float((np.abs(values - expected) / expected).max()), tol=1e-10)

    # 'spectral' is computed through a symmetric eigensolve, so it is only valid
    # for symmetric/Hermitian input -- it does not raise on a general matrix, it
    # just returns |lambda_max| / |lambda_min| of the assumed-symmetric matrix.
    symmetric = (batch + batch.transpose(0, 2, 1)) / 2.0
    values = bl.cond(symmetric, "spectral", device=device)
    expected = np.stack([np.linalg.cond(item, p=2) for item in symmetric])
    report("cond 'spectral' relative error", float((np.abs(values - expected) / expected).max()), tol=1e-8)

    section("transpose")
    report("error", float(np.abs(bl.transpose(batch) - batch.transpose(0, 2, 1)).max()), tol=0.0)

    section("lascl: rescale a matrix by cto / cfrom")
    scaled = bl.lascl(batch, 2.0, 6.0)
    report("error", float(np.abs(scaled - 3.0 * batch).max()), tol=1e-12)

    section("Utilities accept SciPy sparse input too")
    try:
        import scipy.sparse as sp

        sparse = bl.random_sparse_hermitian(64, density=0.1, seed=3)
        report("sparse frobenius norm", float(np.ravel(bl.norm(sparse, "fro"))[0]))
        report(
            "matches scipy",
            float(abs(np.ravel(bl.norm(sparse, "fro"))[0] - sp.linalg.norm(sparse, "fro"))),
            tol=1e-10,
        )
    except ImportError:  # pragma: no cover
        report("scipy", "not installed")


if __name__ == "__main__":
    main()
