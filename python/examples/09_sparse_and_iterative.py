# %% [markdown]
# # 9. Sparse matrices and iterative eigensolvers
#
# Sparse input is given as SciPy CSR matrices — one matrix, or a list of matrices
# for a batch. Iterative solvers compute a *few* extreme eigenpairs rather than
# the whole spectrum, which is what you want when $n$ is large.
#
# **Covered:** `spmm`, `syevx` (with convergence history), `lanczos`,
# `ritz_values`, and ILU(k) preconditioning.
#
# One section is deliberately informational rather than pass/fail:
# unpreconditioned `syevx` can stagnate on hard problems, and seeing what that
# looks like is the point.

# %%
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


header("9. Sparse matrices and iterative eigensolvers")

if sp is None:
    report("scipy", "not installed -- this notebook needs SciPy")

device = preferred_device()
n, batch = 128, 3

# %% [markdown]
# ## Building a batch of sparse symmetric matrices
#
# `random_sparse_hermitian` returns SciPy CSR matrices — a list when batched.
# `shared_pattern=True` makes every matrix in the batch use the same CSR
# structure, which is what the ILU(k) section below requires.

# %%
section("Building a batch of sparse symmetric matrices")

matrices = bl.random_sparse_hermitian(
    n, density=0.05, batch_size=batch, seed=7, diagonal_boost=8.0, shared_pattern=True
)
report("batch size", len(matrices))
report("shape / nnz", (matrices[0].shape, matrices[0].nnz))

# %% [markdown]
# ## `spmm` — sparse times dense
#
# Same $\alpha$ / $\beta$ contract as the dense BLAS routines: $\beta$ needs an
# existing `C`, supplied through `out=`.

# %%
section("spmm: sparse-times-dense product")

dense_block = np.random.default_rng(1).standard_normal((batch, n, 4))
product = bl.spmm(matrices, dense_block, device=device)
expected = np.stack([matrices[i] @ dense_block[i] for i in range(batch)])
report("error", float(np.abs(product - expected).max()), tol=1e-10)

section("spmm with alpha and beta")

c = np.random.default_rng(2).standard_normal((batch, n, 4))
got = bl.spmm(matrices, dense_block, alpha=2.0, beta=0.5, out=c.copy(), device=device)
report("error", float(np.abs(got - (2.0 * expected + 0.5 * c)).max()), tol=1e-10)

# %% [markdown]
# ## `syevx` — a few extreme eigenpairs
#
# A LOBPCG-style solver. `find_largest` picks which end of the spectrum you get.
# We check it against an operator with well-separated eigenvalues, since that is
# what iterative methods converge on.

# %%
section("syevx: a few extreme eigenpairs of a sparse matrix")

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

# %% [markdown]
# ## Convergence history
#
# `return_history=True` adds a third return value with per-iteration residuals,
# which is how you tell a slow problem from a stalled one.
#
# The random sparse batch built at the top is a hard problem for an
# unpreconditioned solver — the residual below drops quickly at first and then
# flattens out. That is what stalling looks like.

# %%
section("syevx convergence history (informational)")

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

# %% [markdown]
# ## `ritz_values` — Rayleigh quotients for a given subspace
#
# Given a trial subspace $V$, this returns the eigenvalues of $V^{T} A V$.
# Feeding it converged eigenvectors must reproduce the corresponding eigenvalues.

# %%
section("ritz_values: Rayleigh quotients for a given subspace")

ritz = bl.ritz_values(separated, vectors, device=device)
report("ritz values shape", ritz.shape)
report(
    "matches syevx values",
    float(np.abs(np.sort(ritz, axis=-1) - np.sort(values, axis=-1)).max()),
    tol=1e-6,
)

# %% [markdown]
# ## `lanczos` — Krylov subspace eigenvalues
#
# Returns $n$ Ritz values; the extreme ones converge first and fastest. Accuracy
# depends strongly on the problem — good on this well-separated operator, much
# worse on the random sparse batch.

# %%
section("lanczos: Krylov subspace eigenvalues")

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

# %% [markdown]
# ## ILU(k) preconditioning
#
# `iluk_factorize` builds an incomplete LU factorization $M \approx A$; every
# matrix in the batch must share one CSR pattern. `iluk_apply` then applies
# $M^{-1}$, which should land close to the true solution of $A x = b$.

# %%
section("iluk_factorize / iluk_apply: ILU(k) preconditioning")

preconditioner = bl.iluk_factorize(separated, options=bl.ILUKOptions(levels_of_fill=1), device=device)
report("dtype / n / batch", (preconditioner.dtype, preconditioner.n, preconditioner.batch_size))

rhs = np.random.default_rng(3).standard_normal((batch, n, 1))
solution = bl.iluk_apply(preconditioner, rhs, device=device)
report("solution shape", solution.shape)

approximated = np.stack([separated[i] @ solution[i] for i in range(batch)])
exact = np.stack([np.linalg.solve(separated[i].toarray(), rhs[i]) for i in range(batch)])
report("|b|", float(np.abs(rhs).max()))
report("|A M^-1 b - b|", float(np.abs(approximated - rhs).max()), tol=1e-1)
report("|M^-1 b - A^-1 b|", float(np.abs(solution - exact).max()), tol=1e-2)

# %% [markdown]
# ## Combining the two
#
# The preconditioner handle passes straight into `syevx`, which uses it inside
# its inner solves.

# %%
section("syevx with an ILU(k) preconditioner")

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
