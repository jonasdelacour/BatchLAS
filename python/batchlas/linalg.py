# SPDX-License-Identifier: MIT
"""NumPy/SciPy-flavoured front end of the C++ ``batchlas::linalg`` layer.

Every function runs the matching ``batchlas::linalg`` wrapper on the device and
returns NumPy arrays. Conventions:

- **Batching.** The last two axes are the matrix; any leading axes are the batch
  (``(m, n)``, ``(batch, m, n)`` or ``(b1, b2, m, n)``). Outputs keep the input's
  leading axes, and per-item scalars (``info``, ``count``, ``norm``, ``cond``) have
  exactly the batch shape, so a 2-D input gives a NumPy scalar.
- **dtypes.** float32, float64, complex64, complex128. Integer and bool input is
  promoted to float64; two operands are promoted to their common dtype.
- **info is always returned.** The C++ layer always fills a per-item status, so
  every factorisation and solve returns a named tuple whose last field is ``info``
  (``0`` = success, ``i > 0`` as LAPACK) rather than raising: one bad item does not
  discard the rest of the batch. Check it, e.g. ``np.all(result.info == 0)``.
- **Triangles.** ``lower=True`` (the default) reads the lower triangle.
- ``backend=`` and ``device=`` mean what they mean everywhere else in ``batchlas``.

Elementwise ``add``, ``subtract``, ``multiply``, ``divide``, ``axpby`` and ``scale``
are the top-level functions, re-exported here.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np

from . import _batchlas as _ext
from ._api import (
    _normalize_backend,
    _normalize_device,
    add,
    axpby,
    divide,
    multiply,
    scale,
    subtract,
)

__all__ = [
    "CholeskyResult",
    "EighResult",
    "EighSubsetResult",
    "EigvalshSubsetResult",
    "InvResult",
    "LUResult",
    "SVDResult",
    "SolveResult",
    "add",
    "axpby",
    "cholesky",
    "cond",
    "divide",
    "eigh",
    "eigvalsh",
    "gram",
    "inv",
    "lu",
    "matmul",
    "matvec",
    "multiply",
    "norm",
    "scale",
    "solve",
    "solve_spd",
    "solve_triangular",
    "subtract",
    "svd",
    "transpose",
    "tril",
    "triu",
]


class CholeskyResult(NamedTuple):
    factor: np.ndarray
    info: Any


class EighResult(NamedTuple):
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    info: Any


class EighSubsetResult(NamedTuple):
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    count: Any
    info: Any


class EigvalshSubsetResult(NamedTuple):
    eigenvalues: np.ndarray
    count: Any
    info: Any


class SolveResult(NamedTuple):
    x: np.ndarray
    info: Any


class InvResult(NamedTuple):
    inverse: np.ndarray
    info: Any


class SVDResult(NamedTuple):
    U: np.ndarray | None
    S: np.ndarray
    Vh: np.ndarray | None
    info: Any


class LUResult(NamedTuple):
    lu: np.ndarray
    pivots: np.ndarray
    info: Any


_SUPPORTED = (np.float32, np.float64, np.complex64, np.complex128)


def _dtype(*arrays: np.ndarray) -> np.dtype:
    dtype = np.result_type(*arrays)
    if dtype.kind in "biu":
        return np.dtype(np.float64)
    if dtype == np.float16:
        return np.dtype(np.float32)
    if dtype.type not in _SUPPORTED:
        raise TypeError(f"unsupported dtype {dtype}; expected float32, float64, complex64 or complex128")
    return dtype


class _Batched:
    """One operand flattened to the ``(batch, m, n)`` layout the extension takes."""

    def __init__(self, value: Any, dtype: np.dtype | None = None, name: str = "a"):
        array = np.asarray(value)
        if array.ndim < 2:
            raise ValueError(f"{name} must have at least 2 dimensions, got shape {array.shape}")
        array = array.astype(dtype if dtype is not None else _dtype(array), copy=False)
        self.batch_shape = array.shape[:-2]
        self.core = array.shape[-2:]
        if array.ndim > 3:
            array = array.reshape((-1,) + self.core)
        if array.ndim == 3 and array.shape[0] == 0:
            raise ValueError(f"{name} has an empty batch")
        self.array = array
        self.handle = _ext._dense_from_numpy(array)

    def out(self, value: Any, core_ndim: int) -> Any:
        if value is None:
            return None
        array = np.asarray(value)
        reshaped = array.reshape(self.batch_shape + array.shape[array.ndim - core_ndim:])
        return reshaped[()] if core_ndim == 0 else reshaped


def _pair(a: Any, b: Any) -> tuple[_Batched, _Batched]:
    a_array, b_array = np.asarray(a), np.asarray(b)
    dtype = _dtype(a_array, b_array)
    left, right = _Batched(a_array, dtype, "a"), _Batched(b_array, dtype, "b")
    if left.batch_shape != right.batch_shape:
        raise ValueError(f"batch shapes differ: a {left.batch_shape}, b {right.batch_shape}")
    return left, right


def _rhs(a: Any, b: Any) -> tuple[_Batched, _Batched, bool]:
    """``b`` as a matrix; a vector right-hand side (``b.ndim == a.ndim - 1``) becomes one column."""
    a_array, b_array = np.asarray(a), np.asarray(b)
    vector = b_array.ndim == a_array.ndim - 1
    if vector:
        b_array = b_array[..., None]
    left, right = _pair(a_array, b_array)
    return left, right, vector


def _squeeze_column(x: np.ndarray, vector: bool) -> np.ndarray:
    return x[..., 0] if vector else x


def _queue_args(backend: str, device: str | None) -> tuple[str, str | None]:
    return _normalize_backend(backend, device), _normalize_device(device)


def _trans(value: Any) -> str:
    return {0: "n", 1: "t", 2: "c"}.get(value, value) if isinstance(value, int) else str(value)


def _uplo(lower: bool) -> str:
    return "lower" if lower else "upper"


def matmul(
    a: Any,
    b: Any,
    *,
    alpha: Any = 1,
    trans_a: str = "n",
    trans_b: str = "n",
    compute_precision: str = "default",
    backend: str = "auto",
    device: str | None = None,
) -> np.ndarray:
    """``alpha * op(a) @ op(b)`` per item; ``trans_*`` is ``"n"``, ``"t"`` or ``"c"``."""
    left, right = _pair(a, b)
    return left.out(
        _ext._linalg_matmul(left.handle, right.handle, alpha, trans_a, trans_b, compute_precision,
                            *_queue_args(backend, device)),
        2,
    )


def matvec(a: Any, x: Any, *, trans: str = "n", backend: str = "auto", device: str | None = None) -> np.ndarray:
    """``op(a) @ x`` per item; ``x`` has the batch shape of ``a`` plus one axis."""
    a_array, x_array = np.asarray(a), np.asarray(x)
    dtype = _dtype(a_array, x_array)
    left = _Batched(a_array, dtype)
    if x_array.shape[:-1] != left.batch_shape:
        raise ValueError(f"x has shape {x_array.shape}; expected batch shape {left.batch_shape} plus one axis")
    x_flat = x_array.astype(dtype, copy=False)
    if x_flat.ndim > 2:
        x_flat = x_flat.reshape(-1, x_flat.shape[-1])
    vector = _ext._vector_from_numpy(x_flat)
    return left.out(_ext._linalg_matvec(left.handle, vector, trans, *_queue_args(backend, device)), 1)


def gram(a: Any, *, trans: str = "c", backend: str = "auto", device: str | None = None) -> np.ndarray:
    """Gram matrix with both triangles filled: ``a^H a`` (``trans="c"`` or ``"t"``, n x n) or
    ``a a^H`` (``trans="n"``, m x m). Complex input needs the vendor ``herk``."""
    left = _Batched(a)
    return left.out(_ext._linalg_gram(left.handle, trans, *_queue_args(backend, device)), 2)


def solve_triangular(
    a: Any,
    b: Any,
    *,
    lower: bool = True,
    trans: Any = "n",
    unit_diagonal: bool = False,
    side: str = "left",
    alpha: Any = 1,
    backend: str = "auto",
    device: str | None = None,
) -> np.ndarray:
    """``x`` with ``op(a) x = alpha b`` (``side="left"``) or ``x op(a) = alpha b`` (``"right"``).

    Only the ``lower`` triangle of ``a`` is read. ``trans`` is ``"n"``/``"t"``/``"c"`` or SciPy's
    0/1/2. No status: a zero diagonal gives infinities in that item. A vector ``b`` is a column.
    """
    left, right, vector = _rhs(a, b)
    x = _ext._linalg_solve_triangular(
        left.handle, right.handle, alpha, side, _uplo(lower), _trans(trans),
        "unit" if unit_diagonal else "non_unit", *_queue_args(backend, device),
    )
    return _squeeze_column(right.out(x, 2), vector)


def cholesky(a: Any, *, lower: bool = True, backend: str = "auto", device: str | None = None) -> CholeskyResult:
    """Cholesky factor (``a = L L^H`` or ``U^H U``), the other triangle zero.

    ``info[b] = i > 0``: the leading minor of order ``i`` is not positive definite.
    """
    left = _Batched(a)
    factor, info = _ext._linalg_cholesky(left.handle, _uplo(lower), *_queue_args(backend, device))
    return CholeskyResult(left.out(factor, 2), left.out(info, 0))


def _subset(subset_by_index: Any, subset_by_value: Any, lower: bool) -> dict[str, Any] | None:
    if subset_by_index is None and subset_by_value is None:
        return None
    if subset_by_index is not None and subset_by_value is not None:
        raise ValueError("give subset_by_index or subset_by_value, not both")
    if not lower:
        raise ValueError("a subset reads the lower triangle only; pass lower=True")
    if subset_by_index is not None:
        il, iu = subset_by_index
        return {"select": "index", "il": int(il), "iu": int(iu)}
    vl, vu = subset_by_value
    return {"select": "value", "vl": float(vl), "vu": float(vu)}


def _subset_call(left: _Batched, subset: dict[str, Any], vectors: bool, backend: str, device: str | None):
    values, vecs, count, info = _ext._linalg_eigh_subset(left.handle, subset, vectors,
                                                         *_queue_args(backend, device))
    count_flat = np.asarray(count).reshape(-1)
    values = np.array(values, copy=True).reshape(count_flat.size, np.size(values) // count_flat.size)
    values[np.arange(values.shape[1])[None, :] >= count_flat[:, None]] = np.nan
    return left.out(values, 1), left.out(vecs, 2), left.out(count, 0), left.out(info, 0)


def eigh(
    a: Any,
    *,
    lower: bool = True,
    subset_by_index: tuple[int, int] | None = None,
    subset_by_value: tuple[float, float] | None = None,
    backend: str = "auto",
    device: str | None = None,
) -> EighResult | EighSubsetResult:
    """Eigenvalues (ascending) and eigenvectors (columns) of Hermitian ``a``.

    ``subset_by_index=(il, iu)`` (0-based, inclusive) or ``subset_by_value=(vl, vu)``
    (half-open ``(vl, vu]``) runs syevx and returns ``EighSubsetResult``: ``k`` value slots per
    item (``iu - il + 1``, or ``n`` for a value range), of which the first ``count`` are
    found; the rest are NaN and their vector columns zero.
    """
    left = _Batched(a)
    subset = _subset(subset_by_index, subset_by_value, lower)
    if subset is not None:
        return EighSubsetResult(*_subset_call(left, subset, True, backend, device))
    values, vectors, info = _ext._linalg_eigh(left.handle, _uplo(lower), *_queue_args(backend, device))
    return EighResult(left.out(values, 1), left.out(vectors, 2), left.out(info, 0))


def eigvalsh(
    a: Any,
    *,
    lower: bool = True,
    subset_by_index: tuple[int, int] | None = None,
    subset_by_value: tuple[float, float] | None = None,
    backend: str = "auto",
    device: str | None = None,
) -> np.ndarray | EigvalshSubsetResult:
    """Eigenvalues of Hermitian ``a``, ascending. The full form returns an array (C++
    ``linalg::eigvalsh`` reports no status); a subset returns ``EigvalshSubsetResult``."""
    left = _Batched(a)
    subset = _subset(subset_by_index, subset_by_value, lower)
    if subset is not None:
        values, _, count, info = _subset_call(left, subset, False, backend, device)
        return EigvalshSubsetResult(values, count, info)
    return left.out(_ext._linalg_eigvalsh(left.handle, _uplo(lower), *_queue_args(backend, device)), 1)


def solve(a: Any, b: Any, *, trans: Any = "n", backend: str = "auto", device: str | None = None) -> SolveResult:
    """``x`` with ``op(a) x = b`` by LU with partial pivoting. ``info[b] = i > 0``: ``U[i-1, i-1]``
    is exactly zero and that item's ``x`` is meaningless. A vector ``b`` is a column."""
    left, right, vector = _rhs(a, b)
    x, info = _ext._linalg_solve(left.handle, right.handle, _trans(trans), *_queue_args(backend, device))
    return SolveResult(_squeeze_column(right.out(x, 2), vector), left.out(info, 0))


def solve_spd(a: Any, b: Any, *, lower: bool = True, backend: str = "auto", device: str | None = None) -> SolveResult:
    """``x`` with ``a x = b`` for Hermitian positive-definite ``a`` by Cholesky; ``info`` as cholesky()."""
    left, right, vector = _rhs(a, b)
    x, info = _ext._linalg_solve_spd(left.handle, right.handle, _uplo(lower), *_queue_args(backend, device))
    return SolveResult(_squeeze_column(right.out(x, 2), vector), left.out(info, 0))


def inv(a: Any, *, backend: str = "auto", device: str | None = None) -> InvResult:
    """Inverse by LU (getrf + getri); ``info`` as solve()."""
    left = _Batched(a)
    inverse, info = _ext._linalg_inv(left.handle, *_queue_args(backend, device))
    return InvResult(left.out(inverse, 2), left.out(info, 0))


def svd(
    a: Any,
    *,
    full_matrices: bool = True,
    compute_uv: bool = True,
    hermitian_uplo: str | None = None,
    jobu: str | None = None,
    jobvh: str | None = None,
    backend: str = "auto",
    device: str | None = None,
) -> SVDResult:
    """``a = U diag(S) Vh``, ``S`` descending, ``k = min(m, n)`` values per item.

    ``full_matrices`` picks ``"all"`` (U m x m, Vh n x n) or ``"thin"`` (m x k, k x n);
    ``compute_uv=False`` computes values only (U and Vh are None). ``jobu`` / ``jobvh``
    (``"all"``, ``"thin"``, ``"none"``) override either side independently.
    ``hermitian_uplo="lower"``/``"upper"`` treats square ``a`` as Hermitian and reads that triangle.
    """
    default = ("all" if full_matrices else "thin") if compute_uv else "none"
    left = _Batched(a)
    u, s, vh, info = _ext._linalg_svd(left.handle, jobu or default, jobvh or default, hermitian_uplo,
                                     *_queue_args(backend, device))
    return SVDResult(left.out(u, 2), left.out(s, 1), left.out(vh, 2), left.out(info, 0))


def lu(a: Any, *, backend: str = "auto", device: str | None = None) -> LUResult:
    """Packed LU factors (L's unit diagonal implicit) and 1-based LAPACK pivots (int64), the
    format ``batchlas.getrs``/``getri`` take. ``info[b] = i > 0``: ``U[i-1, i-1]`` is zero."""
    left = _Batched(a)
    factors, pivots, info = _ext._linalg_lu(left.handle, *_queue_args(backend, device))
    return LUResult(left.out(factors, 2), left.out(pivots, 1), left.out(info, 0))


def triu(a: Any, k: int = 0, *, device: str | None = None) -> np.ndarray:
    """Copy of ``a`` with everything below diagonal ``k`` zeroed (``numpy.triu``)."""
    left = _Batched(a)
    return left.out(_ext._linalg_triu(left.handle, int(k), _normalize_device(device)), 2)


def tril(a: Any, k: int = 0, *, device: str | None = None) -> np.ndarray:
    """Copy of ``a`` with everything above diagonal ``k`` zeroed (``numpy.tril``)."""
    left = _Batched(a)
    return left.out(_ext._linalg_tril(left.handle, int(k), _normalize_device(device)), 2)


def transpose(a: Any, *, device: str | None = None) -> np.ndarray:
    """``a^T`` per item. Real dtypes only: nothing conjugates, so complex raises NotImplementedError."""
    left = _Batched(a)
    return left.out(_ext._linalg_transpose(left.handle, _normalize_device(device)), 2)


def _norm_name(ord: Any) -> str:
    if ord is None or ord == "fro":
        return "fro"
    if ord in (1, "1", "one"):
        return "one"
    if ord in (np.inf, "inf"):
        return "inf"
    if ord in (2, "2", "spectral"):
        return "spectral"
    if ord == "max":
        return "max"
    raise ValueError(f"unsupported ord {ord!r}; expected 'fro', 1, inf, 2 or 'max'")


def norm(a: Any, ord: Any = "fro", *, device: str | None = None) -> Any:
    """Per-item matrix norm: ``"fro"``, ``1`` (max column sum), ``inf`` (max row sum), ``2`` or ``"max"``."""
    left = _Batched(a)
    return left.out(_ext._linalg_norm(left.handle, _norm_name(ord), _normalize_device(device)), 0)


def cond(a: Any, ord: Any = "fro", *, backend: str = "auto", device: str | None = None) -> Any:
    """Per-item ``||a|| * ||a^-1||`` (real dtypes). ``ord=2`` is the largest/smallest eigenvalue
    magnitude ratio of a *symmetric* ``a`` (lower triangle read), not the general 2-norm
    condition number."""
    left = _Batched(a)
    return left.out(_ext._linalg_cond(left.handle, _norm_name(ord), *_queue_args(backend, device)), 0)
