# %% [markdown]
# # 6. Symmetric and Hermitian eigensolvers
#
# BatchLAS ships several full-spectrum symmetric eigensolvers. They all compute
# the same thing — they differ in how the work is mapped onto the device.
#
# | Variant | Approach | Best for |
# |---|---|---|
# | `syev` | general driver, vendor path where available | anything |
# | `syev_cta` | `sytrd_cta` → `steqr_cta` → `ormqx_cta`, one work-group per matrix | $n \le 32$ |
# | `syev_jacobi_cta` | two-sided Jacobi in one work-group | $n \le 32$, graded input |
# | `syev_blocked` | blocked reduction + divide and conquer | medium/large $n$ |
# | `syev_two_stage` | dense → band → tridiagonal reduction | large $n$ |
#
# `syev_variant_support()` asks the device which of these it can actually run,
# instead of guessing from the matrix size.
#
# Notebook **10** shows why `syev_jacobi_cta` is worth having; notebook **12**
# measures which variant is fastest where.

# %%
import numpy as np

import batchlas as bl

from _common import (
    batched_symmetric,
    eigenvalue_error,
    header,
    preferred_device,
    report,
    residual,
    section,
)

header("6. Symmetric eigensolvers")

device = preferred_device()
batch = 8

# %% [markdown]
# ## Which variants does this device support?

# %%
section("Which variants does this device support?")

small = batched_symmetric(batch, 16, seed=1)
for key, value in bl.syev_variant_support(small, uplo="lower", device=device).items():
    report(key, value)

# %% [markdown]
# ## `syev` — eigenvalues and eigenvectors
#
# Returns `(w, V)` with eigenvalues ascending. The input is read from the lower
# triangle by default. The residual $\lVert A V - V \, \mathrm{diag}(w) \rVert$
# is the check that matters.

# %%
section("syev: eigenvalues and eigenvectors")

reference = np.linalg.eigvalsh(small)
values, vectors = bl.syev(small, device=device)

report("eigenvalue error", eigenvalue_error(values, reference), tol=1e-10)
report("|A V - V diag(w)|", residual(small, values, vectors), tol=1e-10)

# %% [markdown]
# ### Eigenvalues only
#
# Pass `compute_vectors=False` to skip the back-transform entirely; the call then
# returns just the eigenvalue array.

# %%
section("Eigenvalues only")

values = bl.syev(small, compute_vectors=False, device=device)
report("shape", values.shape)
report("error", eigenvalue_error(values, reference), tol=1e-10)

# %% [markdown]
# ## The small-matrix variants
#
# Both keep one matrix resident in a single work-group. They need a GPU with a
# sub-group width of 32 and are limited to $n \le 32$.

# %%
section("The small-matrix variants (n <= 32): one work-group per problem")

for name in ("syev_cta", "syev_jacobi_cta"):
    values, vectors = getattr(bl, name)(small, device=device)
    report(f"{name:16s} eigenvalue error", eigenvalue_error(values, reference), tol=1e-10)
    report(f"{name:16s} residual", residual(small, values, vectors), tol=1e-10)

# %% [markdown]
# ## The medium and large variants

# %%
section("The medium/large variants")

medium = batched_symmetric(4, 128, seed=2)
medium_reference = np.linalg.eigvalsh(medium)

for name in ("syev", "syev_blocked", "syev_two_stage"):
    try:
        values, vectors = getattr(bl, name)(medium, device=device)
        report(f"{name:16s} eigenvalue error", eigenvalue_error(values, medium_reference), tol=1e-8)
        report(f"{name:16s} residual", residual(medium, values, vectors), tol=1e-8)
    except (RuntimeError, NotImplementedError) as exc:
        report(f"{name:16s}", f"unavailable ({type(exc).__name__}: {exc})")

# %% [markdown]
# ## Tuning a variant through its options object
#
# Each variant is driven by a different inner algorithm, and each has its own
# options dataclass:
#
# - `syev_cta` → `SteqrOptions` (a CTA-resident QR iteration)
# - `syev_blocked` / `syev_two_stage` → `StedcOptions` (divide and conquer)
# - `syev_jacobi_cta` → `JacobiOptions`
#
# A plain `dict` works anywhere an options object does.

# %%
section("Tuning a variant through its options object")

options = bl.SteqrOptions(max_sweeps=200, cta_shift_strategy="wilkinson", sort_order="ascending")
values, _ = bl.syev_cta(small, options=options, device=device)
report("eigenvalue error", eigenvalue_error(values, reference), tol=1e-10)

options = bl.StedcOptions(recursion_threshold=32, max_sec_iter=80)
try:
    values, _ = bl.syev_blocked(medium, options=options, device=device)
    report("blocked with custom stedc options", eigenvalue_error(values, medium_reference), tol=1e-8)
except (RuntimeError, NotImplementedError) as exc:
    report("blocked with custom stedc options", f"unavailable ({type(exc).__name__})")

# %% [markdown]
# ## Hermitian (complex) input
#
# Complex input works through the same calls; the eigenvalues come back real.

# %%
section("Hermitian (complex) input")

rng = np.random.default_rng(3)
z = rng.standard_normal((batch, 16, 16)) + 1j * rng.standard_normal((batch, 16, 16))
z = (z + z.conj().transpose(0, 2, 1)) / 2.0
z_reference = np.linalg.eigvalsh(z)

values, vectors = bl.syev(z, device=device)
report("eigenvalue error", eigenvalue_error(values, z_reference), tol=1e-10)
report("|A V - V diag(w)|", residual(z, values.astype(z.dtype), vectors), tol=1e-10)

# %% [markdown]
# ## `uplo` — which triangle holds your data
#
# Both triangles of a full symmetric matrix give the same spectrum. Supplying
# only the lower triangle also works, which confirms the strict upper half really
# is ignored for `uplo="lower"`.
#
# > The mirror image — upper triangle only with `uplo="upper"` — is **not**
# > reliable on the CUDA path today. See the README. Pass the full matrix instead.

# %%
section("uplo: which triangle holds your data")

for side in ("lower", "upper"):
    values = bl.syev(small, compute_vectors=False, uplo=side, device=device)
    report(f"uplo={side}", eigenvalue_error(values, reference), tol=1e-10)

values = bl.syev(np.tril(small), compute_vectors=False, uplo="lower", device=device)
report("lower triangle only", eigenvalue_error(values, reference), tol=1e-10)
