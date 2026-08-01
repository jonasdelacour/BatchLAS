"""Sparse matrices and iterative eigensolvers: spmm, syevx, lanczos, ritz_values, ILU(k).

Sparse input is given as SciPy CSR matrices -- one matrix, or a list of matrices
for a batch. Iterative solvers compute a few extreme eigenpairs rather than the
whole spectrum, which is what you want when n is large.

One section is informational rather than pass/fail: unpreconditioned syevx can
stagnate on hard problems, and that is shown deliberately. See README.md.

Run with:  python 09_sparse_and_iterative.py
"""

from __future__ import annotations

import numpy as np

import batchlas as bl

from _common import header, preferred_device, report, section

try:
    import scipy.sparse as sp
except ImportError:  # pragma: no cover
    sp = None


def well_separated_operator(n: int, batch: int) -> list:
    """A batch of sparse matrices with a clearly separated, known spectrum.

    Iterative eigensolvers converge on eigenvalue *gaps*, so a diagonal-dominant
    operator like this is the right thing to check a solver against.
    """
    matrices = []
    for index in range(batch):
        diagonal = np.arange(1, n + 1, dtype=np.float64) + index
        off = 0.05 * np.ones(n - 1)
        dense = np.diag(diagonal) + np.diag(off, 1) + np.diag(off, -1)
        matrices.append(sp.csr_matrix(dense))
    return matrices


def main() -> None:
    header("9. Sparse matrices and iterative eigensolvers")
    if sp is None:
        report("scipy", "not installed -- this example needs SciPy")
        return

    device = preferred_device()
    n, batch = 128, 3

    section("Building a batch of sparse symmetric matrices")
    # random_sparse_hermitian returns SciPy CSR matrices (a list when batched).
    # shared_pattern makes every matrix in the batch use the same CSR structure,
    # which is what the ILU(k) section below requires.
    matrices = bl.random_sparse_hermitian(
        n, density=0.05, batch_size=batch, seed=7, diagonal_boost=8.0, shared_pattern=True
    )
    report("batch size", len(matrices))
    report("shape / nnz", (matrices[0].shape, matrices[0].nnz))

    section("spmm: sparse-times-dense product")
    dense_block = np.random.default_rng(1).standard_normal((batch, n, 4))
    product = bl.spmm(matrices, dense_block, device=device)
    expected = np.stack([matrices[i] @ dense_block[i] for i in range(batch)])
    report("error", float(np.abs(product - expected).max()), tol=1e-10)

    section("spmm with alpha and beta")
    c = np.random.default_rng(2).standard_normal((batch, n, 4))
    got = bl.spmm(matrices, dense_block, alpha=2.0, beta=0.5, out=c.copy(), device=device)
    report("error", float(np.abs(got - (2.0 * expected + 0.5 * c)).max()), tol=1e-10)

    section("syevx: a few extreme eigenpairs of a sparse matrix")
    # LOBPCG-style solver; find_largest picks which end of the spectrum you get.
    separated = well_separated_operator(n, batch)
    spectrum = np.linalg.eigvalsh(np.stack([m.toarray() for m in separated]))
    neigs = 4
    options = bl.SyevxOptions(iterations=300, find_largest=True, algorithm="chol2")
    values, vectors = bl.syevx(separated, neigs, options=options, device=device)
    report("values shape / vectors shape", (values.shape, vectors.shape))
    report(
        "largest eigenvalue error",
        float(np.abs(np.sort(values, axis=-1) - spectrum[:, -neigs:]).max()),
        tol=1e-6,
    )
    residual_norm = np.abs(
        np.stack([separated[i] @ vectors[i] for i in range(batch)]) - vectors * values[:, None, :]
    ).max()
    report("residual", float(residual_norm), tol=1e-5)

    section("syevx at the other end of the spectrum")
    options = bl.SyevxOptions(iterations=300, find_largest=False)
    values_small = bl.syevx(separated, neigs, compute_vectors=False, options=options, device=device)
    report(
        "smallest eigenvalue error",
        float(np.abs(np.sort(values_small, axis=-1) - spectrum[:, :neigs]).max()),
        tol=1e-6,
    )

    section("syevx convergence history (informational)")
    # return_history gives per-iteration residuals, which is how you tell a slow
    # problem from a stalled one. The random sparse batch above is a hard problem
    # for an unpreconditioned solver, and this is what stalling looks like.
    options = bl.SyevxOptions(
        iterations=100,
        find_largest=True,
        store_every=10,
        store_convergence_rate=True,
    )
    _, _, history = bl.syevx(matrices, neigs, options=options, return_history=True, device=device)
    report("history keys", sorted(k for k, v in history.items() if v is not None))
    report("iterations actually run", history["iterations_done"])
    best = history["best_residual_history"]
    report("residual history shape (iters, batch, neigs)", best.shape)
    report("first stored residual", float(best[0].max()))
    report("last stored residual", float(best[-1].max()))

    section("ritz_values: Rayleigh quotients for a given subspace")
    # Given a trial subspace V, this returns the eigenvalues of V^T A V. Feeding it
    # converged eigenvectors must reproduce the corresponding eigenvalues.
    ritz = bl.ritz_values(separated, vectors, device=device)
    report("ritz values shape", ritz.shape)
    report(
        "matches syevx values",
        float(np.abs(np.sort(ritz, axis=-1) - np.sort(values, axis=-1)).max()),
        tol=1e-6,
    )

    section("lanczos: Krylov subspace eigenvalues")
    # Lanczos returns n Ritz values; the extreme ones converge first and fastest.
    # Accuracy depends strongly on the problem: it is good on this well-separated
    # operator, but degrades badly on the random sparse batch (see README.md).
    options = bl.LanczosOptions(ortho_algorithm="cgs2", reorthogonalization_iterations=2)
    lanczos_values = np.asarray(bl.lanczos(separated, compute_vectors=False, options=options, device=device))
    report("values shape", lanczos_values.shape)
    report(
        "largest Ritz value error",
        float(np.abs(lanczos_values.max(axis=-1) - spectrum[:, -1]).max()),
        tol=1e-1,
    )
    report(
        "smallest Ritz value error",
        float(np.abs(lanczos_values.min(axis=-1) - spectrum[:, 0]).max()),
        tol=1e-1,
    )

    section("iluk_factorize / iluk_apply: ILU(k) preconditioning")
    # Every matrix in the batch must share one CSR pattern. M ~= A, so applying
    # M^-1 to b should land close to the true solution of A x = b.
    preconditioner = bl.iluk_factorize(separated, options=bl.ILUKOptions(levels_of_fill=1), device=device)
    report("dtype / n / batch", (preconditioner.dtype, preconditioner.n, preconditioner.batch_size))
    rhs = np.random.default_rng(3).standard_normal((batch, n, 1))
    solution = bl.iluk_apply(preconditioner, rhs, device=device)
    report("solution shape", solution.shape)
    # M ~= A, so A (M^-1 b) should be much closer to b than an unpreconditioned guess.
    approximated = np.stack([separated[i] @ solution[i] for i in range(batch)])
    exact = np.stack([np.linalg.solve(separated[i].toarray(), rhs[i]) for i in range(batch)])
    report("|b|", float(np.abs(rhs).max()))
    report("|A M^-1 b - b|", float(np.abs(approximated - rhs).max()), tol=1e-1)
    report("|M^-1 b - A^-1 b|", float(np.abs(solution - exact).max()), tol=1e-2)

    section("syevx with an ILU(k) preconditioner")
    # Pass the handle straight through; syevx uses it inside its inner solves.
    options = bl.SyevxOptions(iterations=200, find_largest=False)
    values = bl.syevx(
        separated,
        neigs,
        compute_vectors=False,
        options=options,
        preconditioner=preconditioner,
        device=device,
    )
    report(
        "smallest eigenvalue error",
        float(np.abs(np.sort(values, axis=-1) - spectrum[:, :neigs]).max()),
        tol=1e-6,
    )


if __name__ == "__main__":
    main()
