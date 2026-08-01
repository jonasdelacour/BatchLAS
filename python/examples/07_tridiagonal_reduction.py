# %% [markdown]
# # 7. Reduction to tridiagonal form
#
# Every dense symmetric eigensolver rests on the same idea: reduce $A$ to a
# symmetric tridiagonal matrix with an orthogonal similarity transform, solve the
# much cheaper tridiagonal problem, then transform the vectors back.
#
# BatchLAS exposes each stage separately so you can benchmark or recombine them:
#
# | Routine | Stage |
# |---|---|
# | `sytrd_cta` | dense → tridiagonal in one work-group ($n \le 32$) |
# | `sytrd_blocked` | dense → tridiagonal, blocked panel + BLAS-3 update |
# | `sytrd_sy2sb` | two-stage step 1: dense → band |
# | `sytrd_sb2st` | two-stage step 2: band → tridiagonal (bulge chasing) |
# | `hetrd_hb2st` | LAPACK-style alias of `sytrd_sb2st` |
# | `sytrd_band_reduction` | band → tridiagonal via the BANDR1 blocked schedule |
#
# A similarity transform preserves the spectrum, so every stage below is checked
# by comparing eigenvalues against the original matrix.

# %%
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

header("7. Reduction to tridiagonal form")

device = preferred_device()
batch = 4

# %% [markdown]
# ## `sytrd_cta` — one work-group per matrix
#
# Returns `(a, d, e, tau)`. `d` and `e` are the tridiagonal coefficients; `a` and
# `tau` hold the Householder reflectors in the packed SYTD2 layout.

# %%
section("sytrd_cta: one work-group per matrix (n <= 32)")

small = batched_symmetric(batch, 16, seed=1)
small_reference = np.linalg.eigvalsh(small)

reduced, d, e, tau = bl.sytrd_cta(small, uplo="lower", device=device)
report("d shape / e shape", (d.shape, e.shape))
report(
    "spectrum error",
    eigenvalue_error(np.linalg.eigvalsh(tridiagonal_matrix(d, e)), small_reference),
    tol=1e-10,
)

# %% [markdown]
# ## `sytrd_blocked` — blocked panel plus BLAS-3 update
#
# The `block_size` parameter controls the panel width, trading panel work against
# trailing-update efficiency.

# %%
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

# %% [markdown]
# ## Two-stage reduction, step 1 — `sytrd_sy2sb`
#
# Dense to band. The band comes back in LAPACK band storage: shape
# $(k_d + 1) \times n$, where `AB[i, j]` holds `A[j + i, j]` for the lower
# triangle. `_common.band_to_dense` expands it back for checking.

# %%
section("Two-stage, step 1 -- sytrd_sy2sb: dense to band")

kd = 8
reduced, ab, tau = bl.sytrd_sy2sb(medium, kd, uplo="lower", device=device)

report("band storage shape", ab.shape)
report(
    "spectrum error",
    eigenvalue_error(np.linalg.eigvalsh(band_to_dense(ab)), medium_reference),
    tol=1e-8,
)

# %% [markdown]
# ## Two-stage reduction, step 2 — `sytrd_sb2st`
#
# Band to tridiagonal by bulge chasing. `hetrd_hb2st` is the same routine under
# its LAPACK name, for Hermitian band input.

# %%
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

# %% [markdown]
# ## `sytrd_band_reduction` — the BANDR1 blocked schedule
#
# An alternative band → tridiagonal path that takes a per-sweep schedule:
#
# - `d_seq` — how many diagonals to eliminate per sweep
# - `block_size_seq` — the block size to use per sweep
#
# A `0` entry, or a sequence shorter than the sweep count, falls back to the
# implementation default (the last entry is reused).

# %%
section("sytrd_band_reduction: the BANDR1 blocked schedule")

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

# %% [markdown]
# ## Putting it together
#
# Dense → band → tridiagonal → eigenvalues. This is `syev_two_stage` spelled out
# by hand.

# %%
section("Putting it together: two-stage reduction + tridiagonal solve")

_, ab, _ = bl.sytrd_sy2sb(medium, kd, uplo="lower", device=device)
d, e, _ = bl.sytrd_sb2st(ab, kd, uplo="lower", block_size=16, device=device)
values = bl.stedc(d, e, compute_vectors=False, device=device)

report("eigenvalue error", eigenvalue_error(values, medium_reference), tol=1e-8)

# %% [markdown]
# ## Complex Hermitian input
#
# For Hermitian input `d` is real, but the subdiagonal `e` carries a phase that
# gets absorbed into the Householder reflectors. The real tridiagonal matrix
# whose spectrum matches $A$ is built from $\lvert e \rvert$, **not** from
# `e.real`.

# %%
section("Complex Hermitian input")

rng = np.random.default_rng(3)
z = rng.standard_normal((batch, 16, 16)) + 1j * rng.standard_normal((batch, 16, 16))
z = (z + z.conj().transpose(0, 2, 1)) / 2.0

_, dz, ez, _ = bl.sytrd_cta(z, uplo="lower", device=device)

report("max |imag(d)|", float(np.abs(dz.imag).max()))
report("max |imag(e)|", float(np.abs(ez.imag).max()))
report(
    "spectrum error using |e|",
    eigenvalue_error(np.linalg.eigvalsh(tridiagonal_matrix(dz.real, np.abs(ez))), np.linalg.eigvalsh(z)),
    tol=1e-10,
)
