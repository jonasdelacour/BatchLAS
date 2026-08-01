# %% [markdown]
# # 4. QR factorization and orthogonalization
#
# Two related jobs:
#
# - **`geqrf` / `orgqr` / `ormqr`** — the LAPACK-style QR factorization, where $Q$
#   stays in packed Householder form until you ask for it.
# - **`ortho` / `ortho_metric`** — orthonormalise a block of vectors, the inner
#   step of every iterative eigensolver.

# %%
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

header("4. QR factorization and orthogonalization")

device = preferred_device()
batch, m, n = 4, 32, 16

a = batched_general(batch, m, n, seed=1)

# %% [markdown]
# ## `geqrf` — QR in packed reflector form
#
# Returns `(qr, tau)`: $R$ sits in the upper triangle of `qr`, the Householder
# reflectors that generate $Q$ are stored below it, and `tau` holds the
# corresponding scalars.

# %%
section("geqrf: QR in packed reflector form")

qr, tau = bl.geqrf(a, device=device)
r = np.triu(qr[:, :n, :])
report("qr shape", qr.shape)
report("tau shape", tau.shape)

# %% [markdown]
# ## `orgqr` — materialise $Q$
#
# Expands the packed form into an explicit (economy-sized) $Q$.

# %%
section("orgqr: materialise Q from the packed form")

q = bl.orgqr(qr, tau, device=device)[:, :, :n]
report("|Q^T Q - I|", orthogonality_error(q), tol=1e-10)
report("|Q R - A|", float(np.abs(q @ r - a).max()), tol=1e-10)

# %% [markdown]
# ## `ormqr` — apply $Q$ without forming it
#
# For tall matrices this is much cheaper than `orgqr` followed by a `gemm`.
#
# Note the shapes: `ormqr` applies the full $m \times m$ $Q$, so its result has
# $m$ rows, whereas `orgqr` above materialised only the first $n$ columns of $Q$
# — which reproduce the first $n$ rows of the product.

# %%
section("ormqr: apply Q without ever forming it")

c = batched_general(batch, m, 5, seed=2)
applied = bl.ormqr(qr, c, tau, side="left", trans="t", device=device)
report("ormqr result shape", applied.shape)
report("|Q^T C - ormqr(...)|", float(np.abs(q.transpose(0, 2, 1) @ c - applied[:, :n, :]).max()), tol=1e-10)

# %% [markdown]
# ## `ortho` — orthonormalise a block of vectors
#
# Several algorithms are available and they trade robustness against speed:
#
# | Algorithm | Note |
# |---|---|
# | `chol2` | two Cholesky-QR passes; fastest, but squares the condition number |
# | `shiftchol3` | shifted variant, survives much worse conditioning |
# | `cgs2` | classical Gram-Schmidt, twice |
# | `svqb`, `svqb2` | eigen-decomposition of the Gram matrix |
# | `householder` | currently unreliable on CUDA — see the README |

# %%
section("ortho: orthonormalise a block of vectors")

block = batched_general(batch, m, 8, seed=3)

for algorithm in ("chol2", "cgs2", "svqb", "svqb2", "shiftchol3"):
    try:
        q_block = bl.ortho(block, algorithm=algorithm, device=device)
        report(f"{algorithm:8s} |Q^T Q - I|", orthogonality_error(q_block), tol=1e-8)
    except (RuntimeError, NotImplementedError, ValueError) as exc:
        report(f"{algorithm:8s}", f"unavailable ({type(exc).__name__})")

# %% [markdown]
# ## Where the cheap algorithms break down
#
# Here two columns of the block nearly coincide. `chol2` is expected to fail —
# it forms $A^{T}A$, squaring the condition number — while the shifted and
# Gram-Schmidt variants hold up. This is exactly the trade-off to weigh when
# picking an algorithm, so these values are printed without a tolerance.

# %%
section("ortho on an ill-conditioned block")

near_singular = batched_general(batch, m, 4, seed=4)
near_singular[:, :, 1] = near_singular[:, :, 0] + 1e-8 * near_singular[:, :, 1]

for algorithm in ("chol2", "cgs2", "svqb2", "shiftchol3"):
    try:
        q_block = bl.ortho(near_singular, algorithm=algorithm, device=device)
        report(f"{algorithm:8s} |Q^T Q - I|", orthogonality_error(q_block))
    except (RuntimeError, NotImplementedError, ValueError) as exc:
        report(f"{algorithm:8s}", f"unavailable ({type(exc).__name__})")

# %% [markdown]
# ## `ortho_metric` — orthogonalise against an existing basis
#
# `M` holds vectors you already trust — say, converged eigenvectors. The result
# is orthonormal **and** orthogonal to every column of `M`. This is the block
# step at the heart of iterative eigensolvers.
#
# Note the constraint: `cols(A) + cols(M)` must not exceed the vector dimension.

# %%
section("ortho_metric: orthogonalise against an existing basis")

basis = bl.ortho(batched_general(batch, m, 6, seed=5), algorithm="chol2", device=device)
q_m = bl.ortho_metric(block, basis, algorithm="chol2", iterations=2, device=device)

report("|Q^T Q - I|", orthogonality_error(q_m), tol=1e-8)
report("|M^T Q| (should be ~0)", float(np.abs(basis.transpose(0, 2, 1) @ q_m).max()), tol=1e-8)
