"""Getting started: discovering the runtime and running your first batched op.

Covers: available_backends, available_devices, compiled_features, gemm,
the batching convention, dtypes, and the ``out=`` contract.

Run with:  python 01_getting_started.py
"""

from __future__ import annotations

import numpy as np

import batchlas as bl

from _common import header, preferred_device, report, section


def main() -> None:
    header("1. Getting started with BatchLAS")

    section("What is available at runtime")
    # available_backends() lists the vendor/reference backends compiled in.
    # "auto" lets the library choose based on the device you target.
    report("backends", bl.available_backends())
    for device in bl.available_devices():
        report("device", f"{device['name']} (type={device['type']})")
    features = bl.compiled_features()
    report("compiled features", sorted(features))

    device = preferred_device()
    report("device used below", device or "library default")

    section("A single matrix product")
    # Every routine accepts plain NumPy arrays and returns plain NumPy arrays.
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([[5.0, 6.0], [7.0, 8.0]])
    report("gemm error vs numpy", float(np.abs(bl.gemm(a, b, device=device) - a @ b).max()), tol=1e-12)

    section("The batching convention")
    # A 2-D array is one matrix; a 3-D array of shape (batch, rows, cols) is a
    # batch. The same call handles both -- that is the point of the library.
    batch = np.stack([a, 2.0 * a, 3.0 * a])
    rhs = np.stack([b, b, b])
    got = bl.gemm(batch, rhs, device=device)
    report("batched gemm shape", got.shape)
    report("batched gemm error", float(np.abs(got - batch @ rhs).max()), tol=1e-12)

    section("Scaling factors: C = alpha * A @ B + beta * C")
    c = np.ones((2, 2))
    got = bl.gemm(a, b, alpha=2.0, beta=0.5, out=c.copy(), device=device)
    report("alpha/beta error", float(np.abs(got - (2.0 * (a @ b) + 0.5 * c)).max()), tol=1e-12)

    section("Writing into a preallocated buffer")
    # Passing out= avoids an allocation and returns that same array object.
    out = np.empty((2, 2))
    result = bl.gemm(a, b, out=out, device=device)
    report("out= returns the same object", result is out)

    section("Supported dtypes")
    # float32, float64, complex64 and complex128 are all supported. The output
    # dtype follows the input dtype.
    for dtype in (np.float32, np.float64, np.complex64, np.complex128):
        x = np.eye(3, dtype=dtype) * (2.0 + 1.0j if np.issubdtype(dtype, np.complexfloating) else 2.0)
        report(f"{np.dtype(dtype).name} -> ", bl.gemm(x, x, device=device).dtype.name)

    section("Transposes without materialising a copy")
    m = np.arange(6.0).reshape(2, 3)
    report(
        "trans_a='t' error",
        float(np.abs(bl.gemm(m, m, trans_a="t", device=device) - m.T @ m).max()),
        tol=1e-12,
    )


if __name__ == "__main__":
    main()
