# %% [markdown]
# # 8. Tridiagonal eigensolvers
#
# These solvers take the coefficients of a symmetric tridiagonal matrix rather
# than a dense matrix: `d` of length $n$ on the diagonal, `e` of length $n - 1$
# on the off-diagonal.
#
# | Solver | Algorithm |
# |---|---|
# | `steqr` | implicit QL/QR iteration |
# | `steqr_cta` | the same, resident in one work-group (small $n$) |
# | `stedc` | divide and conquer |
# | `stedc_flat` | non-recursive divide and conquer |
# | `tridiagonal_solver` | convenience driver taking `(alpha, beta)` |
#
# They are the inner solve of every dense symmetric eigensolver, and worth
# calling directly whenever your problem is already tridiagonal.

# %%
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

# %% [markdown]
# ## All four solvers on the same problem
#
# > `stedc_flat` produces correct eigenvalues but its **eigenvectors** do not
# > currently satisfy $A V = V \, \mathrm{diag}(w)$. That residual is reported
# > without a tolerance so the discrepancy stays visible. See the README.

# %%
section("All four solvers on the same problem")

for name in ("steqr", "steqr_cta", "stedc", "stedc_flat"):
    values, vectors = getattr(bl, name)(d, e, device=device)
    report(f"{name:11s} eigenvalue error", eigenvalue_error(values, reference), tol=1e-10)
    if name == "stedc_flat":
        report(f"{name:11s} residual (known bad)", residual(dense, values, vectors))
    else:
        report(f"{name:11s} residual", residual(dense, values, vectors), tol=1e-9)

# %% [markdown]
# ### Eigenvalues only

# %%
section("Eigenvalues only")

for name in ("steqr", "steqr_cta", "stedc", "stedc_flat"):
    values = getattr(bl, name)(d, e, compute_vectors=False, device=device)
    report(f"{name:11s} error", eigenvalue_error(values, reference), tol=1e-10)

# %% [markdown]
# ## Sort order
#
# Eigenvalues come back ascending by default; `sort_order="descending"` reverses
# them (and permutes the eigenvectors to match).

# %%
section("Sort order")

ascending = bl.steqr(d, e, compute_vectors=False, options=bl.SteqrOptions(sort_order="ascending"), device=device)
descending = bl.steqr(d, e, compute_vectors=False, options=bl.SteqrOptions(sort_order="descending"), device=device)
report("descending == reversed ascending", bool(np.allclose(descending, ascending[:, ::-1])))

# %% [markdown]
# ## Tuning the QR iteration
#
# `SteqrOptions` exposes the sweep cap, the shift strategy and the CTA layout.
# Wilkinson shifts tend to converge faster on hard small problems.

# %%
section("Tuning the QR iteration")

for strategy in ("lapack", "wilkinson"):
    options = bl.SteqrOptions(max_sweeps=400, cta_shift_strategy=strategy)
    values = bl.steqr_cta(d, e, compute_vectors=False, options=options, device=device)
    report(f"cta_shift_strategy={strategy:<10s}", eigenvalue_error(values, reference), tol=1e-10)

# %% [markdown]
# ## Tuning divide and conquer
#
# Below `recursion_threshold`, `stedc` falls back to the leaf QR solver, which is
# configured through a **nested** `SteqrOptions`.

# %%
section("Tuning divide and conquer")

options = bl.StedcOptions(
    recursion_threshold=8,
    merge_threads=128,
    leaf_steqr_params=bl.SteqrOptions(max_sweeps=200),
)
values = bl.stedc(d, e, compute_vectors=False, options=options, device=device)
report("custom stedc options", eigenvalue_error(values, reference), tol=1e-10)

# %% [markdown]
# ## `tridiagonal_solver` — the convenience driver
#
# > This driver's QR iteration does not converge reliably: accuracy varies with
# > $n$ and with the data. The numbers below are shown **without** a pass/fail
# > tolerance so the loss of accuracy stays visible. Prefer `steqr` or `stedc`
# > for real work. See the README.

# %%
section("tridiagonal_solver: the convenience driver")

small_d = d[:, :8]
small_e = e[:, :7]
small_dense = tridiagonal_matrix(small_d, small_e)
small_reference = np.linalg.eigvalsh(small_dense)

values, vectors = bl.tridiagonal_solver(small_d, small_e, compute_vectors=True, device=device)
report("eigenvalue error (n=8, no tolerance applied)", eigenvalue_error(values, small_reference))
report("residual (n=8, no tolerance applied)", residual(small_dense, values, vectors))

# %% [markdown]
# ## A generator with a known closed-form spectrum
#
# The eigenvalues of the $(a, b, b)$ Toeplitz tridiagonal matrix are
#
# $$\lambda_k = a + 2 b \cos\!\left(\frac{k \pi}{n + 1}\right), \qquad k = 1 \dots n$$
#
# which makes it a good exact correctness check.

# %%
section("tridiag_toeplitz: a generator with a known closed-form spectrum")

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
