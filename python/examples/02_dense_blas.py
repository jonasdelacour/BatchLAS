# %% [markdown]
# # 2. Batched dense BLAS
#
# The classic BLAS-2 and BLAS-3 operations, applied to a whole batch per call.
#
# | Routine | Computes |
# |---|---|
# | `gemm` | $C \leftarrow \alpha\,\mathrm{op}(A)\,\mathrm{op}(B) + \beta C$ |
# | `gemv` | $y \leftarrow \alpha\,\mathrm{op}(A)\,x + \beta y$ |
# | `symm` | $C \leftarrow \alpha A B + \beta C$, $A$ symmetric |
# | `syrk` | $C \leftarrow \alpha A A^{T} + \beta C$ |
# | `syr2k` | $C \leftarrow \alpha (A B^{T} + B A^{T}) + \beta C$ |
# | `trmm` | $B \leftarrow \alpha\,\mathrm{op}(A)\,B$, $A$ triangular |
# | `trsm` | solve $\mathrm{op}(A) X = \alpha B$, $A$ triangular |
#
# The last two sections cover heterogeneous batches and mixed precision.

# %%
import numpy as np

import batchlas as bl

from _common import batched_general, batched_symmetric, header, preferred_device, report, section

header("2. Batched dense BLAS")

device = preferred_device()
batch, n = 4, 32

a = batched_general(batch, n, n, seed=1)
b = batched_general(batch, n, n, seed=2)
c = batched_general(batch, n, n, seed=3)

# %% [markdown]
# ## `gemm` — general matrix product
#
# Remember that $\beta$ multiplies an existing `C`, which you supply through `out=`.

# %%
section("gemm: C <- alpha * op(A) op(B) + beta * C")

got = bl.gemm(a, b, alpha=1.5, beta=-0.25, out=c.copy(), device=device)
report("error", float(np.abs(got - (1.5 * (a @ b) - 0.25 * c)).max()), tol=1e-10)

# %% [markdown]
# ## `gemv` — matrix-vector product
#
# Vectors follow the same batching rule: `(n,)` is one vector, `(batch, n)` is a batch.

# %%
section("gemv: y <- alpha * op(A) x + beta * y")

x = batched_general(batch, n, 1, seed=4)[:, :, 0]
y = batched_general(batch, n, 1, seed=5)[:, :, 0]

got = bl.gemv(a, x, alpha=2.0, beta=1.0, out=y.copy(), device=device)
report("error", float(np.abs(got - (2.0 * np.einsum("bij,bj->bi", a, x) + y)).max()), tol=1e-10)

# %% [markdown]
# ## `symm` — symmetric matrix product
#
# Only the triangle named by `uplo` is read, so the other half never has to be
# filled in. Here we verify against the fully symmetrised matrix.

# %%
section("symm: C <- alpha * A B + beta * C with symmetric A")

s = batched_symmetric(batch, n, seed=6)
got = bl.symm(s, b, alpha=1.0, side="left", uplo="lower", device=device)
report("error", float(np.abs(got - s @ b).max()), tol=1e-10)

# %% [markdown]
# ## `syrk` and `syr2k` — symmetric rank-k updates
#
# Both write only the triangle named by `uplo`; the opposite triangle of the
# result is left untouched, which is why the checks mask with `np.tril`.

# %%
section("syrk: C <- alpha * A A^T + beta * C (lower triangle)")

got = bl.syrk(a, alpha=1.0, uplo="lower", device=device)
report("error", float(np.abs(np.tril(got) - np.tril(a @ a.transpose(0, 2, 1))).max()), tol=1e-10)

section("syr2k: C <- alpha (A B^T + B A^T) + beta * C (lower triangle)")

got = bl.syr2k(a, b, alpha=1.0, uplo="lower", device=device)
expected = a @ b.transpose(0, 2, 1) + b @ a.transpose(0, 2, 1)
report("error", float(np.abs(np.tril(got) - np.tril(expected)).max()), tol=1e-10)

# %% [markdown]
# ## `trmm` and `trsm` — triangular multiply and solve
#
# `trsm` solves rather than multiplies, so the triangular matrix needs a
# well-conditioned diagonal; we bias it before solving.

# %%
section("trmm: B <- alpha * op(A) B with triangular A")

lower = np.tril(a)
got = bl.trmm(lower, b, alpha=1.0, side="left", uplo="lower", diag="non_unit", device=device)
report("error", float(np.abs(got - lower @ b).max()), tol=1e-10)

section("trsm: solve op(A) X = alpha * B for triangular A")

lower = np.tril(a) + n * np.eye(n)
x_solved = bl.trsm(lower, b, alpha=1.0, side="left", uplo="lower", diag="non_unit", device=device)
report("residual", float(np.abs(lower @ x_solved - b).max()), tol=1e-9)

# %% [markdown]
# ## Heterogeneous batches
#
# Pass a **list** of 2-D arrays instead of a 3-D array and every problem in the
# batch may have its own shape. Results come back as a list too. This is
# something plain NumPy broadcasting cannot express.

# %%
section("Heterogeneous batches: one call, differently shaped matrices")

rng = np.random.default_rng(7)
lhs = [rng.standard_normal((m, k)) for m, k in ((3, 5), (7, 2), (4, 4))]
rhs = [rng.standard_normal((k, p)) for k, p in ((5, 6), (2, 3), (4, 4))]

products = bl.gemm_heterogeneous(lhs, rhs, device=device)
worst = max(
    float(np.abs(products[i][: l.shape[0], : r.shape[1]] - l @ r).max())
    for i, (l, r) in enumerate(zip(lhs, rhs))
)
report("shapes in", [f"{l.shape}x{r.shape}" for l, r in zip(lhs, rhs)])
report("error", worst, tol=1e-10)

# %% [markdown]
# ## Mixed precision
#
# `compute_precision` selects the accumulation mode independently of the storage
# dtype. `tf32` trades accuracy for throughput on hardware that supports it — the
# error below is measured against the float64 result.

# %%
section("Mixed precision: float32 storage, higher-precision accumulation")

a32 = a.astype(np.float32)
b32 = b.astype(np.float32)

for precision in ("default", "tf32"):
    try:
        got = bl.gemm(a32, b32, compute_precision=precision, device=device)
        report(f"{precision} error vs float64", float(np.abs(got - a @ b).max()))
    except (RuntimeError, NotImplementedError, ValueError) as exc:
        report(f"{precision}", f"unavailable ({type(exc).__name__})")
