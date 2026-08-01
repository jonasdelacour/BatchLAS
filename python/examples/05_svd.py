# %% [markdown]
# # 5. Singular value decomposition
#
# BatchLAS exposes three SVD drivers with different sweet spots:
#
# | Driver | Intended for |
# |---|---|
# | `gesvd` | general-purpose, uses the vendor path where available |
# | `gesvd_blocked` | native blocked path, medium and large matrices |
# | `gesvd_cta` | one work-group per matrix, very small matrices ($n \le 32$) |
#
# All three compute $A = U \, \mathrm{diag}(s) \, V^{H}$.
#
# The underlying two-step pipeline — reduce to bidiagonal form, then run the
# bidiagonal QR iteration — is also available separately as `gebrd_*` and
# `bdsqr`, which the later sections use directly.

# %%
import numpy as np

import batchlas as bl

from _common import batched_general, header, preferred_device, report, section


def bidiagonal_singular_values(d: np.ndarray, e: np.ndarray) -> np.ndarray:
    """Singular values of the upper bidiagonal matrices described by (d, e)."""
    return np.stack([
        np.linalg.svd(np.diag(d[i]) + np.diag(e[i], 1), compute_uv=False)
        for i in range(d.shape[0])
    ])


header("5. Singular value decomposition")

device = preferred_device()
batch = 4

# %% [markdown]
# ## `gesvd` — the general driver

# %%
section("gesvd: A = U diag(s) Vh")

a = batched_general(batch, 24, 24, seed=1)
u, s, vh = bl.gesvd(a, device=device)
reference = np.linalg.svd(a, compute_uv=False)

report("singular value error", float(np.abs(np.sort(s)[:, ::-1] - reference).max()), tol=1e-9)
report("|U diag(s) Vh - A|", float(np.abs(u @ (s[:, :, None] * vh) - a).max()), tol=1e-9)
report("|U^T U - I|", float(np.abs(u.transpose(0, 2, 1) @ u - np.eye(24)).max()), tol=1e-9)

# %% [markdown]
# ### Values only
#
# Skip the vectors when you do not need them — the call returns just `s`.

# %%
section("Values only: skip the vectors when you do not need them")

values = bl.gesvd(a, compute_vectors=False, device=device)
report("shape", values.shape)
report("error", float(np.abs(np.sort(values)[:, ::-1] - reference).max()), tol=1e-9)

# %% [markdown]
# ## The blocked and CTA drivers
#
# `gesvd_cta` is limited to $n \le 32$ and needs a GPU, but for a large batch of
# small matrices it keeps the whole problem resident in one work-group.

# %%
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

# %% [markdown]
# ## Symmetric input
#
# Passing `uplo` tells the driver the matrix is Hermitian so it can take the
# eigensolver fast path. For a symmetric matrix the singular values are the
# absolute eigenvalues.

# %%
section("Symmetric input: declare uplo to take the Hermitian fast path")

sym = batched_general(batch, 16, 16, seed=3)
sym = (sym + sym.transpose(0, 2, 1)) / 2.0

values = bl.gesvd(sym, compute_vectors=False, uplo="lower", device=device)
report(
    "error vs |eigenvalues|",
    float(np.abs(np.sort(values)[:, ::-1] - np.sort(np.abs(np.linalg.eigvalsh(sym)))[:, ::-1]).max()),
    tol=1e-9,
)

# %% [markdown]
# ## `gebrd_*` — reduction to bidiagonal form
#
# All three variants return `(a, d, e, tauq, taup)`: the bidiagonal coefficients
# `d` and `e`, plus the reflectors generating the orthogonal factors $Q$ and $P$.
# The bidiagonal matrix has the same singular values as the input.

# %%
section("gebrd_unblocked / gebrd_cta / gebrd_blocked: reduction to bidiagonal form")

for label, call, matrix in (
    ("unblocked", lambda mat: bl.gebrd_unblocked(mat, device=device), small),
    ("cta      ", lambda mat: bl.gebrd_cta(mat, device=device), small),
    ("blocked  ", lambda mat: bl.gebrd_blocked(mat, block_size=16, device=device), a),
):
    _, d, e, tauq, taup = call(matrix)
    expected = np.linalg.svd(matrix, compute_uv=False)
    got = np.sort(bidiagonal_singular_values(d, e))[:, ::-1]
    report(f"{label} singular value error", float(np.abs(got - expected).max()), tol=1e-9)

# %% [markdown]
# ## `bdsqr` — bidiagonal QR iteration
#
# `gebrd` followed by `bdsqr` is the SVD pipeline spelled out: reduce, then iterate.

# %%
section("bdsqr: bidiagonal QR iteration on the (d, e) coefficients")

reduced, d, e, tauq, taup = bl.gebrd_cta(small, device=device)
singular = bl.bdsqr(d, e, sort_desc=True, device=device)
report("error", float(np.abs(singular - small_reference).max()), tol=1e-9)

# %% [markdown]
# ## `ormbr` — apply the $Q$ or $P$ factor from `gebrd`
#
# This is how the bidiagonal problem's vectors are turned back into vectors of
# the original $A$. Pass the *reduced* matrix returned by `gebrd` — that is where
# the reflectors live.

# %%
section("ormbr: apply the Q or P factor from gebrd")

c = np.tile(np.eye(16), (batch, 1, 1))
q_applied = bl.ormbr(reduced, tauq, c, vect="Q", side="left", trans="n", device=device)
report("|Q^T Q - I|", float(np.abs(q_applied.transpose(0, 2, 1) @ q_applied - np.eye(16)).max()), tol=1e-9)
