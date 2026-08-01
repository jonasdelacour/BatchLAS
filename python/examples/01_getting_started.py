# %% [markdown]
# # 1. Getting started with BatchLAS
#
# This notebook covers the basics: discovering what the runtime can do, running
# your first batched operation, and the conventions every other notebook builds on.
#
# **Covered:** `available_backends`, `available_devices`, `compiled_features`,
# `gemm`, the batching convention, dtypes, and the `out=` contract.
#
# Every check prints `[ok ]` or `[FAIL]`, so a clean run doubles as a smoke test.

# %%
import numpy as np

import batchlas as bl

from _common import header, preferred_device, report, section

header("1. Getting started with BatchLAS")

# %% [markdown]
# ## What is available at runtime
#
# `available_backends()` lists the vendor and reference backends compiled into
# this build. Passing `backend="auto"` (the default) lets the library choose one
# based on the device you target.

# %%
section("What is available at runtime")

report("backends", bl.available_backends())
for device_info in bl.available_devices():
    report("device", f"{device_info['name']} (type={device_info['type']})")
report("compiled features", sorted(bl.compiled_features()))

device = preferred_device()
report("device used below", device or "library default")

# %% [markdown]
# ## A single matrix product
#
# Every routine accepts plain NumPy arrays and returns plain NumPy arrays. There
# is no separate device-array type to manage from Python.

# %%
section("A single matrix product")

a = np.array([[1.0, 2.0], [3.0, 4.0]])
b = np.array([[5.0, 6.0], [7.0, 8.0]])

report("gemm error vs numpy", float(np.abs(bl.gemm(a, b, device=device) - a @ b).max()), tol=1e-12)

# %% [markdown]
# ## The batching convention
#
# This is the single most important convention in the library:
#
# | Input shape | Meaning |
# |---|---|
# | `(rows, cols)` | one matrix |
# | `(batch, rows, cols)` | a batch of equally shaped matrices |
# | `list` of 2-D arrays | a *heterogeneous* batch, shapes may differ |
#
# The same call handles all three — that is the point of the library.

# %%
section("The batching convention")

batch = np.stack([a, 2.0 * a, 3.0 * a])
rhs = np.stack([b, b, b])

got = bl.gemm(batch, rhs, device=device)
report("batched gemm shape", got.shape)
report("batched gemm error", float(np.abs(got - batch @ rhs).max()), tol=1e-12)

# %% [markdown]
# ## Scaling factors
#
# BLAS-style routines compute
#
# $$C \leftarrow \alpha \, \mathrm{op}(A)\,\mathrm{op}(B) + \beta \, C$$
#
# Because $\beta$ multiplies an *existing* `C`, you must supply that operand.
# `out=` serves as both the `C` operand and the destination buffer. Passing
# `beta != 0` without `out=` raises a `ValueError` rather than silently ignoring it.

# %%
section("Scaling factors: C = alpha * A @ B + beta * C")

c = np.ones((2, 2))
got = bl.gemm(a, b, alpha=2.0, beta=0.5, out=c.copy(), device=device)
report("alpha/beta error", float(np.abs(got - (2.0 * (a @ b) + 0.5 * c)).max()), tol=1e-12)

try:
    bl.gemm(a, b, beta=1.0, device=device)
    report("beta without out=", "no error raised (unexpected)")
except ValueError as exc:
    report("beta without out= raises", str(exc))

# %% [markdown]
# ## Writing into a preallocated buffer
#
# Passing `out=` avoids an allocation and returns that same array object, so you
# can preallocate once and reuse it across a loop.

# %%
section("Writing into a preallocated buffer")

out = np.empty((2, 2))
result = bl.gemm(a, b, out=out, device=device)
report("out= returns the same object", result is out)

# %% [markdown]
# ## Supported dtypes
#
# `float32`, `float64`, `complex64` and `complex128` are all supported, and the
# output dtype follows the input dtype.

# %%
section("Supported dtypes")

for dtype in (np.float32, np.float64, np.complex64, np.complex128):
    scale = 2.0 + 1.0j if np.issubdtype(dtype, np.complexfloating) else 2.0
    x = np.eye(3, dtype=dtype) * scale
    report(f"{np.dtype(dtype).name} ->", bl.gemm(x, x, device=device).dtype.name)

# %% [markdown]
# ## Transposes without materialising a copy
#
# `trans_a` / `trans_b` accept `"n"` (none), `"t"` (transpose) and `"c"`
# (conjugate transpose). The transpose is folded into the kernel rather than
# built as a separate array.

# %%
section("Transposes without materialising a copy")

m = np.arange(6.0).reshape(2, 3)
report(
    "trans_a='t' error",
    float(np.abs(bl.gemm(m, m, trans_a="t", device=device) - m.T @ m).max()),
    tol=1e-12,
)

# %% [markdown]
# ## Where to go next
#
# - **02** — the rest of the dense BLAS surface, including heterogeneous batches
# - **06** — the symmetric eigensolvers, the richest part of the library
# - **12** — measuring the batching speed-up and picking between variants
