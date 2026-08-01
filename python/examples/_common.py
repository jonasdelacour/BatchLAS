"""Small helpers shared by the BatchLAS examples.

Nothing here is required to use BatchLAS -- these are just utilities that keep
the example scripts short and give them a consistent, verifiable output format.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

import batchlas as bl


def preferred_device() -> str | None:
    """Return ``"gpu"`` when a GPU is visible, otherwise ``None`` (library default).

    Every BatchLAS entry point takes ``device=`` and ``backend=`` keywords. Passing
    ``None`` lets the library pick, which is what most user code should do.
    """
    for device in bl.available_devices():
        if isinstance(device, dict) and str(device.get("type", "")).lower() == "gpu":
            return "gpu"
    return None


def header(title: str) -> None:
    print()
    print(title)
    print("=" * len(title))


def section(title: str) -> None:
    print()
    print(f"-- {title}")


def report(label: str, value: Any, tol: float | None = None) -> None:
    """Print a labelled value; when ``tol`` is given, treat it as a pass/fail check."""
    if isinstance(value, float) and tol is not None:
        status = "ok  " if value <= tol else "FAIL"
        print(f"   [{status}] {label}: {value:.3e}  (tol {tol:.1e})")
    elif isinstance(value, float):
        print(f"          {label}: {value:.6g}")
    else:
        print(f"          {label}: {value}")


def batched_symmetric(batch: int, n: int, seed: int = 0) -> np.ndarray:
    """A batch of real symmetric matrices, shape ``(batch, n, n)``."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((batch, n, n))
    return (a + a.transpose(0, 2, 1)) / 2.0


def batched_general(batch: int, rows: int, cols: int | None = None, seed: int = 0) -> np.ndarray:
    """A batch of general real matrices, shape ``(batch, rows, cols)``."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((batch, rows, cols if cols is not None else rows))


def batched_spd(batch: int, n: int, seed: int = 0, shift: float = 1.0) -> np.ndarray:
    """A batch of symmetric positive definite matrices."""
    a = batched_general(batch, n, n, seed)
    return a @ a.transpose(0, 2, 1) + shift * n * np.eye(n)


def eigenvalue_error(computed: np.ndarray, reference: np.ndarray) -> float:
    """Max absolute difference between two ascending-sorted spectra."""
    return float(np.abs(np.sort(computed, axis=-1) - np.sort(reference, axis=-1)).max())


def residual(a: np.ndarray, values: np.ndarray, vectors: np.ndarray) -> float:
    """Max ``|A V - V diag(w)|`` over the batch."""
    return float(np.abs(a @ vectors - vectors * values[..., None, :]).max())


def orthogonality_error(q: np.ndarray) -> float:
    """Max ``|Q^T Q - I|`` over the batch."""
    gram = q.transpose(0, 2, 1) @ q if q.ndim == 3 else q.T @ q
    return float(np.abs(gram - np.eye(q.shape[-1])).max())


def tridiagonal_matrix(d: np.ndarray, e: np.ndarray) -> np.ndarray:
    """Rebuild dense symmetric tridiagonal matrices from ``(d, e)`` coefficients."""
    single = d.ndim == 1
    d2 = d[None, :] if single else d
    e2 = e[None, :] if single else e
    out = np.stack([
        np.diag(d2[i]) + np.diag(e2[i], -1) + np.diag(e2[i], 1)
        for i in range(d2.shape[0])
    ])
    return out[0] if single else out


def band_to_dense(ab: np.ndarray) -> np.ndarray:
    """Expand lower LAPACK band storage ``(kd + 1, n)`` into dense symmetric matrices.

    ``AB[i, j] == A[j + i, j]`` for the lower triangle.
    """
    single = ab.ndim == 2
    ab2 = ab[None, ...] if single else ab
    kd1, n = ab2.shape[1], ab2.shape[2]
    out = np.zeros((ab2.shape[0], n, n), dtype=ab2.dtype)
    for b in range(ab2.shape[0]):
        lower = np.zeros((n, n), dtype=ab2.dtype)
        for j in range(n):
            for i in range(kd1):
                if j + i < n:
                    lower[j + i, j] = ab2[b, i, j]
        out[b] = np.tril(lower) + np.tril(lower, -1).T
    return out[0] if single else out


def timed(fn, *args, repeats: int = 3, **kwargs) -> tuple[Any, float]:
    """Run ``fn`` a few times and return ``(last_result, best_seconds)``."""
    best = float("inf")
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        best = min(best, time.perf_counter() - start)
    return result, best
