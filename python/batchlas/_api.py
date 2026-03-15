from __future__ import annotations

from typing import Any

import numpy as np

try:
    import scipy.sparse as sp
except ImportError:  # pragma: no cover - optional at import time
    sp = None

from . import _batchlas as _ext
from ._options import (
    ILUKOptions,
    LanczosOptions,
    StedcOptions,
    SteqrOptions,
    SyevxOptions,
    SytrdBandReductionOptions,
    options_to_dict,
)

ILUKPreconditioner = _ext._ILUKPreconditioner

_DENSE_MATRIX_TYPE = _ext._DenseMatrix
_DENSE_VECTOR_TYPE = _ext._DenseVector
_SPARSE_MATRIX_TYPE = _ext._SparseMatrix


def _available_backend_set() -> set[str]:
    return {str(name).lower() for name in _ext.available_backends()}


def _normalize_backend(backend: str | None, device: str | None = None) -> str:
    if backend is not None and str(backend).lower() != "auto":
        return str(backend)
    available = _available_backend_set()
    preferred: list[str]
    normalized_device = None if device is None else str(device).lower()
    if normalized_device == "cpu":
        preferred = ["netlib", "mkl"]
    elif normalized_device == "gpu":
        preferred = ["cuda", "rocm"]
    elif normalized_device == "accelerator":
        preferred = ["cuda", "rocm", "mkl"]
    else:
        preferred = ["cuda", "rocm", "mkl", "netlib"]
    for candidate in preferred:
        if candidate in available:
            return candidate
    return "auto"


def _normalize_device(device: str | None) -> str | None:
    return None if device is None else str(device)


def _normalize_dtype(dtype: Any) -> str:
    name = np.dtype(dtype).name
    if name not in {"float32", "float64", "complex64", "complex128"}:
        raise ValueError("dtype must be float32, float64, complex64, or complex128")
    return name


def _normalize_options(options: Any) -> dict[str, Any]:
    return options_to_dict(options)


def _normalize_norm_name(norm_type: str) -> str:
    value = str(norm_type).lower()
    aliases = {
        "frobenius": "fro",
        "one": "1",
        "infinity": "inf",
        "spectral": "spectral",
    }
    return aliases.get(value, value)


def _is_complex_dtype_name(dtype_name: str) -> bool:
    return np.issubdtype(np.dtype(dtype_name), np.complexfloating)


def _input_dtype_name(value: Any) -> str:
    if isinstance(value, (_DENSE_MATRIX_TYPE, _DENSE_VECTOR_TYPE, _SPARSE_MATRIX_TYPE, ILUKPreconditioner)):
        return str(value.dtype)
    if isinstance(value, (list, tuple)) and len(value) > 0:
        return _input_dtype_name(value[0])
    if _is_sparse_object(value):
        return np.dtype(value.dtype).name
    return np.asarray(value).dtype.name


def _scalar_for_dtype(value: Any, dtype: np.dtype[Any]) -> Any:
    scalar = np.asarray(value)
    if scalar.ndim != 0:
        raise ValueError("expected a scalar value")
    if np.iscomplexobj(scalar) and not np.issubdtype(dtype, np.complexfloating):
        raise ValueError("complex scaling values require a complex matrix dtype")
    return np.asarray(value, dtype=dtype).item()


def _dense_to_numpy(value: Any) -> np.ndarray:
    dense = _unwrap(_coerce_dense_matrix(value))
    return np.asarray(dense)


def _dense_batch_list(value: Any) -> tuple[list[np.ndarray], bool]:
    dense = _dense_to_numpy(value)
    if dense.ndim == 2:
        return [dense], True
    if dense.ndim == 3:
        return [dense[index] for index in range(dense.shape[0])], False
    raise ValueError("dense arrays must have shape (m, n) or (batch, m, n)")


def _sparse_batch_list(value: Any):
    _require_scipy()
    sparse_value = _unwrap(_coerce_sparse_matrix(value))
    if isinstance(sparse_value, list):
        return [sp.csr_matrix(item) for item in sparse_value], False
    return [sp.csr_matrix(sparse_value)], True


def _dense_transpose_fallback(value: Any) -> np.ndarray:
    dense = _dense_to_numpy(value)
    if dense.ndim not in {2, 3}:
        raise ValueError("dense arrays must have shape (m, n) or (batch, m, n)")
    return np.swapaxes(dense, -1, -2).copy()


def _lascl_fallback(value: Any, cfrom: Any, cto: Any):
    if _is_sparse_object(value) or _is_sparse_batch(value):
        matrices, single = _sparse_batch_list(value)
        dtype = matrices[0].dtype
        cfrom_scalar = _scalar_for_dtype(cfrom, dtype)
        cto_scalar = _scalar_for_dtype(cto, dtype)
        scale = cto_scalar / cfrom_scalar
        result = []
        for matrix in matrices:
            scaled = matrix.tocsr(copy=True)
            scaled.data = scaled.data * scale
            result.append(scaled)
        return result[0] if single else result

    dense = _dense_to_numpy(value).copy()
    cfrom_scalar = _scalar_for_dtype(cfrom, dense.dtype)
    cto_scalar = _scalar_for_dtype(cto, dense.dtype)
    dense *= cto_scalar / cfrom_scalar
    return dense


def _matrix_norm_value(matrix: np.ndarray, norm_name: str) -> float:
    if norm_name == "fro":
        return float(np.linalg.norm(matrix, ord="fro"))
    if norm_name == "1":
        return float(np.linalg.norm(matrix, ord=1))
    if norm_name == "inf":
        return float(np.linalg.norm(matrix, ord=np.inf))
    if norm_name == "max":
        return float(np.abs(matrix).max(initial=0.0))
    if norm_name in {"2", "spectral"}:
        return float(np.linalg.norm(matrix, ord=2))
    raise ValueError("invalid norm type")


def _norm_fallback(value: Any, norm_type: str) -> np.ndarray:
    norm_name = _normalize_norm_name(norm_type)
    if _is_sparse_object(value) or _is_sparse_batch(value):
        matrices, _ = _sparse_batch_list(value)
        values = [_matrix_norm_value(matrix.toarray(), norm_name) for matrix in matrices]
    else:
        matrices, _ = _dense_batch_list(value)
        values = [_matrix_norm_value(matrix, norm_name) for matrix in matrices]
    return np.asarray(values, dtype=np.float64)


def _cond_fallback(value: Any, norm_type: str) -> np.ndarray:
    norm_name = _normalize_norm_name(norm_type)
    if norm_name == "fro":
        ord_value: Any = "fro"
    elif norm_name == "1":
        ord_value = 1
    elif norm_name == "inf":
        ord_value = np.inf
    elif norm_name in {"2", "spectral"}:
        ord_value = 2
    else:
        raise ValueError("cond only supports norm_type in {'fro', '1', 'inf', 'spectral'}")

    if _is_sparse_object(value) or _is_sparse_batch(value):
        matrices, _ = _sparse_batch_list(value)
        dense_matrices = [matrix.toarray() for matrix in matrices]
    else:
        dense_matrices, _ = _dense_batch_list(value)
    return np.asarray([np.linalg.cond(matrix, p=ord_value) for matrix in dense_matrices], dtype=np.float64)


def _is_sparse_object(value: Any) -> bool:
    if isinstance(value, _SPARSE_MATRIX_TYPE):
        return True
    if sp is None:
        return False
    if sp.issparse(value):
        return True
    if hasattr(sp, "sparray") and isinstance(value, sp.sparray):
        return True
    return hasattr(value, "tocsr")


def _is_sparse_batch(value: Any) -> bool:
    return isinstance(value, (list, tuple)) and len(value) > 0 and _is_sparse_object(value[0])


def _coerce_dense_matrix(value: Any, *, heterogeneous: bool = False):
    if isinstance(value, _DENSE_MATRIX_TYPE):
        return value
    if heterogeneous:
        if not isinstance(value, (list, tuple)):
            raise ValueError("heterogeneous dense batches must be provided as a sequence of 2D arrays")
        return _ext._dense_from_sequence(value)
    return _ext._dense_from_numpy(np.asarray(value))


def _coerce_dense_vector(value: Any):
    if isinstance(value, _DENSE_VECTOR_TYPE):
        return value
    return _ext._vector_from_numpy(np.asarray(value))


def _require_scipy() -> None:
    if sp is None:
        raise ImportError("SciPy is required for sparse BatchLAS wrappers")


def _coerce_sparse_matrix(value: Any):
    if isinstance(value, _SPARSE_MATRIX_TYPE):
        return value
    if isinstance(value, (list, tuple)):
        _require_scipy()
        return _ext._sparse_from_sequence(value)
    _require_scipy()
    return _ext._sparse_from_python(value)


def _unwrap(value: Any):
    if isinstance(value, (_DENSE_MATRIX_TYPE, _DENSE_VECTOR_TYPE, _SPARSE_MATRIX_TYPE)):
        return value.to_python()
    if isinstance(value, tuple):
        return tuple(_unwrap(item) for item in value)
    if isinstance(value, list):
        return [_unwrap(item) for item in value]
    return value


def _copy_out(result: Any, out: Any):
    if out is None:
        return result
    if not isinstance(result, np.ndarray):
        raise ValueError("out is only supported for dense ndarray outputs")
    out_array = np.asarray(out)
    if out_array.shape != result.shape:
        raise ValueError(f"out has shape {out_array.shape}, expected {result.shape}")
    if out_array.dtype != result.dtype:
        raise ValueError(f"out has dtype {out_array.dtype}, expected {result.dtype}")
    out_array[...] = result
    return out


def _unwrap_with_out(value: Any, out: Any = None):
    return _copy_out(_unwrap(value), out)


def available_backends() -> list[str]:
    return list(_ext.available_backends())


def available_devices() -> list[dict[str, Any]]:
    return list(_ext.available_devices())


def compiled_features() -> dict[str, Any]:
    return dict(_ext.compiled_features())


def gemm(
    a: Any,
    b: Any,
    *,
    alpha: Any = 1,
    beta: Any = 0,
    trans_a: str = "n",
    trans_b: str = "n",
    compute_precision: str = "default",
    backend: str = "auto",
    device: str | None = None,
    out: Any = None,
):
    return _unwrap_with_out(
        _ext._gemm(
            _coerce_dense_matrix(a),
            _coerce_dense_matrix(b),
            alpha,
            beta,
            trans_a,
            trans_b,
            compute_precision,
            _normalize_backend(backend, device),
            _normalize_device(device),
        ),
        out,
    )


def gemm_heterogeneous(
    a: Any,
    b: Any,
    *,
    alpha: Any = 1,
    beta: Any = 0,
    trans_a: str = "n",
    trans_b: str = "n",
    compute_precision: str = "default",
    backend: str = "auto",
    device: str | None = None,
    out: Any = None,
):
    return _unwrap_with_out(
        _ext._gemm_heterogeneous(
            _coerce_dense_matrix(a, heterogeneous=True),
            _coerce_dense_matrix(b, heterogeneous=True),
            alpha,
            beta,
            trans_a,
            trans_b,
            compute_precision,
            _normalize_backend(backend, device),
            _normalize_device(device),
        ),
        out,
    )


def gemv(
    a: Any,
    x: Any,
    *,
    alpha: Any = 1,
    beta: Any = 0,
    trans_a: str = "n",
    backend: str = "auto",
    device: str | None = None,
    out: Any = None,
):
    return _unwrap_with_out(
        _ext._gemv(
            _coerce_dense_matrix(a),
            _coerce_dense_vector(x),
            alpha,
            beta,
            trans_a,
            _normalize_backend(backend, device),
            _normalize_device(device),
        ),
        out,
    )


def symm(
    a: Any,
    b: Any,
    *,
    alpha: Any = 1,
    beta: Any = 0,
    side: str = "left",
    uplo: str = "lower",
    backend: str = "auto",
    device: str | None = None,
    out: Any = None,
):
    return _unwrap_with_out(
        _ext._symm(
            _coerce_dense_matrix(a),
            _coerce_dense_matrix(b),
            alpha,
            beta,
            side,
            uplo,
            _normalize_backend(backend, device),
            _normalize_device(device),
        ),
        out,
    )


def syrk(
    a: Any,
    *,
    alpha: Any = 1,
    beta: Any = 0,
    uplo: str = "lower",
    trans_a: str = "n",
    backend: str = "auto",
    device: str | None = None,
    out: Any = None,
):
    return _unwrap_with_out(
        _ext._syrk(
            _coerce_dense_matrix(a),
            alpha,
            beta,
            uplo,
            trans_a,
            _normalize_backend(backend, device),
            _normalize_device(device),
        ),
        out,
    )


def syr2k(
    a: Any,
    b: Any,
    *,
    alpha: Any = 1,
    beta: Any = 0,
    uplo: str = "lower",
    trans_a: str = "n",
    backend: str = "auto",
    device: str | None = None,
    out: Any = None,
):
    return _unwrap_with_out(
        _ext._syr2k(
            _coerce_dense_matrix(a),
            _coerce_dense_matrix(b),
            alpha,
            beta,
            uplo,
            trans_a,
            _normalize_backend(backend, device),
            _normalize_device(device),
        ),
        out,
    )


def trmm(
    a: Any,
    b: Any,
    *,
    alpha: Any = 1,
    side: str = "left",
    uplo: str = "lower",
    trans_a: str = "n",
    diag: str = "non_unit",
    backend: str = "auto",
    device: str | None = None,
    out: Any = None,
):
    return _unwrap_with_out(
        _ext._trmm(
            _coerce_dense_matrix(a),
            _coerce_dense_matrix(b),
            alpha,
            side,
            uplo,
            trans_a,
            diag,
            _normalize_backend(backend, device),
            _normalize_device(device),
        ),
        out,
    )


def trsm(
    a: Any,
    b: Any,
    *,
    alpha: Any = 1,
    side: str = "left",
    uplo: str = "lower",
    trans_a: str = "n",
    diag: str = "non_unit",
    backend: str = "auto",
    device: str | None = None,
    out: Any = None,
):
    return _unwrap_with_out(
        _ext._trsm(
            _coerce_dense_matrix(a),
            _coerce_dense_matrix(b),
            alpha,
            side,
            uplo,
            trans_a,
            diag,
            _normalize_backend(backend, device),
            _normalize_device(device),
        ),
        out,
    )


def spmm(
    a: Any,
    b: Any,
    *,
    alpha: Any = 1,
    beta: Any = 0,
    trans_a: str = "n",
    trans_b: str = "n",
    backend: str = "auto",
    device: str | None = None,
    out: Any = None,
):
    return _unwrap_with_out(
        _ext._spmm(
            _coerce_sparse_matrix(a),
            _coerce_dense_matrix(b),
            alpha,
            beta,
            trans_a,
            trans_b,
            _normalize_backend(backend, device),
            _normalize_device(device),
        ),
        out,
    )


def potrf(a: Any, *, uplo: str = "lower", backend: str = "auto", device: str | None = None):
    return _unwrap(
        _ext._potrf(_coerce_dense_matrix(a), uplo, _normalize_backend(backend, device), _normalize_device(device))
    )


def getrf(a: Any, *, backend: str = "auto", device: str | None = None):
    return _unwrap(_ext._getrf(_coerce_dense_matrix(a), _normalize_backend(backend, device), _normalize_device(device)))


def getrs(
    lu: Any,
    b: Any,
    pivots: Any,
    *,
    trans_a: str = "n",
    backend: str = "auto",
    device: str | None = None,
):
    return _unwrap(
        _ext._getrs(
            _coerce_dense_matrix(lu),
            _coerce_dense_matrix(b),
            np.asarray(pivots, dtype=np.int64),
            trans_a,
            _normalize_backend(backend, device),
            _normalize_device(device),
        )
    )


def getri(lu: Any, pivots: Any, *, backend: str = "auto", device: str | None = None):
    return _unwrap(
        _ext._getri(
            _coerce_dense_matrix(lu),
            np.asarray(pivots, dtype=np.int64),
            _normalize_backend(backend, device),
            _normalize_device(device),
        )
    )


def inv(a: Any, *, backend: str = "auto", device: str | None = None):
    return _unwrap(_ext._inv(_coerce_dense_matrix(a), _normalize_backend(backend, device), _normalize_device(device)))


def geqrf(a: Any, *, backend: str = "auto", device: str | None = None):
    return _unwrap(_ext._geqrf(_coerce_dense_matrix(a), _normalize_backend(backend, device), _normalize_device(device)))


def orgqr(qr: Any, tau: Any, *, backend: str = "auto", device: str | None = None):
    return _unwrap(
        _ext._orgqr(
            _coerce_dense_matrix(qr),
            _coerce_dense_vector(tau),
            _normalize_backend(backend, device),
            _normalize_device(device),
        )
    )


def ormqr(
    qr: Any,
    c: Any,
    tau: Any,
    *,
    side: str = "left",
    trans: str = "n",
    backend: str = "auto",
    device: str | None = None,
):
    return _unwrap(
        _ext._ormqr(
            _coerce_dense_matrix(qr),
            _coerce_dense_matrix(c),
            _coerce_dense_vector(tau),
            side,
            trans,
            _normalize_backend(backend, device),
            _normalize_device(device),
        )
    )


def gesvd(a: Any, *, compute_vectors: bool = True, backend: str = "auto", device: str | None = None):
    return _unwrap(
        _ext._gesvd(
            _coerce_dense_matrix(a),
            compute_vectors,
            _normalize_backend(backend, device),
            _normalize_device(device),
        )
    )


def gesvd_blocked(a: Any, *, compute_vectors: bool = True, backend: str = "auto", device: str | None = None):
    return _unwrap(
        _ext._gesvd_blocked(
            _coerce_dense_matrix(a),
            compute_vectors,
            _normalize_backend(backend, device),
            _normalize_device(device),
        )
    )


def gebrd_unblocked(a: Any, *, backend: str = "auto", device: str | None = None):
    return _unwrap(
        _ext._gebrd_unblocked(_coerce_dense_matrix(a), _normalize_backend(backend, device), _normalize_device(device))
    )


def bdsqr(d: Any, e: Any, *, sort_desc: bool = False, backend: str = "auto", device: str | None = None):
    return _unwrap(
        _ext._bdsqr(
            _coerce_dense_vector(d),
            _coerce_dense_vector(e),
            sort_desc,
            _normalize_backend(backend, device),
            _normalize_device(device),
        )
    )


def ormbr(
    a: Any,
    tau: Any,
    c: Any,
    *,
    vect: str = "Q",
    side: str = "left",
    trans: str = "n",
    block_size: int = 32,
    backend: str = "auto",
    device: str | None = None,
):
    return _unwrap(
        _ext._ormbr(
            _coerce_dense_matrix(a),
            _coerce_dense_vector(tau),
            _coerce_dense_matrix(c),
            vect,
            side,
            trans,
            int(block_size),
            _normalize_backend(backend, device),
            _normalize_device(device),
        )
    )


def syev(
    a: Any,
    *,
    compute_vectors: bool = True,
    uplo: str = "lower",
    options: dict[str, Any] | None = None,
    backend: str = "auto",
    device: str | None = None,
):
    return _unwrap(
        _ext._syev(
            _coerce_dense_matrix(a),
            compute_vectors,
            uplo,
            _normalize_options(options),
            _normalize_backend(backend, device),
            _normalize_device(device),
        )
    )


def syev_cta(
    a: Any,
    *,
    compute_vectors: bool = True,
    uplo: str = "lower",
    options: SteqrOptions | dict[str, Any] | None = None,
    backend: str = "auto",
    device: str | None = None,
):
    return _unwrap(
        _ext._syev_cta(
            _coerce_dense_matrix(a),
            compute_vectors,
            uplo,
            _normalize_options(options),
            _normalize_backend(backend, device),
            _normalize_device(device),
        )
    )


def syev_blocked(
    a: Any,
    *,
    compute_vectors: bool = True,
    uplo: str = "lower",
    options: StedcOptions | dict[str, Any] | None = None,
    backend: str = "auto",
    device: str | None = None,
):
    return _unwrap(
        _ext._syev_blocked(
            _coerce_dense_matrix(a),
            compute_vectors,
            uplo,
            _normalize_options(options),
            _normalize_backend(backend, device),
            _normalize_device(device),
        )
    )


def syev_two_stage(
    a: Any,
    *,
    compute_vectors: bool = True,
    uplo: str = "lower",
    options: StedcOptions | dict[str, Any] | None = None,
    backend: str = "auto",
    device: str | None = None,
):
    return _unwrap(
        _ext._syev_two_stage(
            _coerce_dense_matrix(a),
            compute_vectors,
            uplo,
            _normalize_options(options),
            _normalize_backend(backend, device),
            _normalize_device(device),
        )
    )


def syevx(
    a: Any,
    neigs: int,
    *,
    compute_vectors: bool = True,
    options: SyevxOptions | dict[str, Any] | None = None,
    backend: str = "auto",
    device: str | None = None,
    return_history: bool = False,
    preconditioner: ILUKPreconditioner | None = None,
):
    normalized = _normalize_options(options)
    if _is_sparse_object(a) or _is_sparse_batch(a):
        raw = _ext._syevx_sparse(
            _coerce_sparse_matrix(a),
            int(neigs),
            compute_vectors,
            normalized,
            _normalize_backend(backend, device),
            _normalize_device(device),
            return_history,
            preconditioner,
        )
    else:
        raw = _ext._syevx_dense(
            _coerce_dense_matrix(a),
            int(neigs),
            compute_vectors,
            normalized,
            _normalize_backend(backend, device),
            _normalize_device(device),
            return_history,
            preconditioner,
        )
    return _unwrap(raw)


def lanczos(
    a: Any,
    *,
    compute_vectors: bool = True,
    options: LanczosOptions | dict[str, Any] | None = None,
    backend: str = "auto",
    device: str | None = None,
):
    if _is_complex_dtype_name(_input_dtype_name(a)):
        raise NotImplementedError("lanczos only supports float32 and float64")
    normalized = _normalize_options(options)
    if _is_sparse_object(a) or _is_sparse_batch(a):
        raw = _ext._lanczos_sparse(
            _coerce_sparse_matrix(a),
            compute_vectors,
            normalized,
            _normalize_backend(backend, device),
            _normalize_device(device),
        )
    else:
        raw = _ext._lanczos_dense(
            _coerce_dense_matrix(a),
            compute_vectors,
            normalized,
            _normalize_backend(backend, device),
            _normalize_device(device),
        )
    return _unwrap(raw)


def steqr(
    d: Any,
    e: Any,
    *,
    compute_vectors: bool = True,
    options: SteqrOptions | dict[str, Any] | None = None,
    backend: str = "auto",
    device: str | None = None,
):
    return _unwrap(
        _ext._steqr(
            _coerce_dense_vector(d),
            _coerce_dense_vector(e),
            compute_vectors,
            _normalize_options(options),
            _normalize_backend(backend, device),
            _normalize_device(device),
        )
    )


def steqr_cta(
    d: Any,
    e: Any,
    *,
    compute_vectors: bool = True,
    options: SteqrOptions | dict[str, Any] | None = None,
    backend: str = "auto",
    device: str | None = None,
):
    return _unwrap(
        _ext._steqr_cta(
            _coerce_dense_vector(d),
            _coerce_dense_vector(e),
            compute_vectors,
            _normalize_options(options),
            _normalize_backend(backend, device),
            _normalize_device(device),
        )
    )


def stedc(
    d: Any,
    e: Any,
    *,
    compute_vectors: bool = True,
    options: StedcOptions | dict[str, Any] | None = None,
    backend: str = "auto",
    device: str | None = None,
):
    return _unwrap(
        _ext._stedc(
            _coerce_dense_vector(d),
            _coerce_dense_vector(e),
            compute_vectors,
            _normalize_options(options),
            _normalize_backend(backend, device),
            _normalize_device(device),
        )
    )


def stedc_flat(
    d: Any,
    e: Any,
    *,
    compute_vectors: bool = True,
    options: StedcOptions | dict[str, Any] | None = None,
    backend: str = "auto",
    device: str | None = None,
):
    return _unwrap(
        _ext._stedc_flat(
            _coerce_dense_vector(d),
            _coerce_dense_vector(e),
            compute_vectors,
            _normalize_options(options),
            _normalize_backend(backend, device),
            _normalize_device(device),
        )
    )


def tridiagonal_solver(
    alpha: Any,
    beta: Any,
    *,
    compute_vectors: bool = True,
    backend: str = "auto",
    device: str | None = None,
):
    if _is_complex_dtype_name(_input_dtype_name(alpha)):
        raise NotImplementedError("tridiagonal_solver only supports float32 and float64")
    return _unwrap(
        _ext._tridiagonal_solver(
            _coerce_dense_vector(alpha),
            _coerce_dense_vector(beta),
            compute_vectors,
            _normalize_backend(backend, device),
            _normalize_device(device),
        )
    )


def ritz_values(a: Any, vectors: Any, *, backend: str = "auto", device: str | None = None):
    if _is_sparse_object(a) or _is_sparse_batch(a):
        raw = _ext._ritz_values_sparse(
            _coerce_sparse_matrix(a),
            _coerce_dense_matrix(vectors),
            _normalize_backend(backend, device),
            _normalize_device(device),
        )
    else:
        raw = _ext._ritz_values_dense(
            _coerce_dense_matrix(a),
            _coerce_dense_matrix(vectors),
            _normalize_backend(backend, device),
            _normalize_device(device),
        )
    return _unwrap(raw)


def transpose(a: Any, *, out: Any = None):
    if _is_sparse_object(a) or _is_sparse_batch(a):
        if out is not None:
            raise ValueError("out is not supported for sparse transpose")
        matrices, single = _sparse_batch_list(a)
        transposed = [matrix.transpose().tocsr() for matrix in matrices]
        return transposed[0] if single else transposed
    if _is_complex_dtype_name(_input_dtype_name(a)):
        return _copy_out(_dense_transpose_fallback(a), out)
    return _unwrap_with_out(_ext._transpose_dense(_coerce_dense_matrix(a)), out)


def lascl(a: Any, cfrom: Any, cto: Any, *, out: Any = None):
    if _is_sparse_object(a) or _is_sparse_batch(a):
        if out is not None:
            raise ValueError("out is not supported for sparse lascl")
        return _lascl_fallback(a, cfrom, cto)
    if _is_complex_dtype_name(_input_dtype_name(a)):
        return _copy_out(_lascl_fallback(a, cfrom, cto), out)
    return _unwrap_with_out(_ext._lascl_dense(_coerce_dense_matrix(a), cfrom, cto), out)


def norm(a: Any, norm_type: str = "fro"):
    if _is_sparse_object(a) or _is_sparse_batch(a):
        return _norm_fallback(a, norm_type)
    return _unwrap(_ext._norm_dense(_coerce_dense_matrix(a), norm_type))


def cond(a: Any, norm_type: str = "spectral", *, backend: str = "auto", device: str | None = None):
    if _is_sparse_object(a) or _is_sparse_batch(a):
        return _cond_fallback(a, norm_type)
    if _is_complex_dtype_name(_input_dtype_name(a)):
        return _cond_fallback(a, norm_type)
    return _unwrap(
        _ext._cond_dense(
            _coerce_dense_matrix(a), norm_type, _normalize_backend(backend, device), _normalize_device(device)
        )
    )


def ortho(
    a: Any,
    *,
    trans_a: str = "n",
    algorithm: str = "chol2",
    backend: str = "auto",
    device: str | None = None,
):
    options = {"trans_a": trans_a, "algorithm": algorithm}
    return _unwrap(
        _ext._ortho(_coerce_dense_matrix(a), options, _normalize_backend(backend, device), _normalize_device(device))
    )


def ortho_metric(
    a: Any,
    metric: Any,
    *,
    trans_a: str = "n",
    trans_m: str = "n",
    algorithm: str = "chol2",
    iterations: int = 2,
    backend: str = "auto",
    device: str | None = None,
):
    options = {
        "trans_a": trans_a,
        "trans_m": trans_m,
        "algorithm": algorithm,
        "iterations": int(iterations),
    }
    return _unwrap(
        _ext._ortho_metric(
            _coerce_dense_matrix(a),
            _coerce_dense_matrix(metric),
            options,
            _normalize_backend(backend, device),
            _normalize_device(device),
        )
    )


def iluk_factorize(
    a: Any,
    *,
    options: ILUKOptions | dict[str, Any] | None = None,
    backend: str = "auto",
    device: str | None = None,
) -> ILUKPreconditioner:
    return _ext._iluk_factorize(
        _coerce_sparse_matrix(a),
        _normalize_options(options),
        _normalize_backend(backend, device),
        _normalize_device(device),
    )


def iluk_apply(
    preconditioner: ILUKPreconditioner,
    rhs: Any,
    *,
    backend: str = "auto",
    device: str | None = None,
):
    return _unwrap(
        _ext._iluk_apply(
            preconditioner,
            _coerce_dense_matrix(rhs),
            _normalize_backend(backend, device),
            _normalize_device(device),
        )
    )


def identity(n: int, *, dtype: Any = np.float64, batch_size: int = 1):
    return _unwrap(_ext._identity_dense(_normalize_dtype(dtype), int(n), int(batch_size)))


def random(
    rows: int,
    cols: int | None = None,
    *,
    dtype: Any = np.float64,
    hermitian: bool = False,
    batch_size: int = 1,
    seed: int = 42,
):
    cols = rows if cols is None else cols
    return _unwrap(
        _ext._random_dense(
            _normalize_dtype(dtype), int(rows), int(cols), bool(hermitian), int(batch_size), int(seed)
        )
    )


def zeros(rows: int, cols: int | None = None, *, dtype: Any = np.float64, batch_size: int = 1):
    cols = rows if cols is None else cols
    return _unwrap(_ext._zeros_dense(_normalize_dtype(dtype), int(rows), int(cols), int(batch_size)))


def ones(rows: int, cols: int | None = None, *, dtype: Any = np.float64, batch_size: int = 1):
    cols = rows if cols is None else cols
    return _unwrap(_ext._ones_dense(_normalize_dtype(dtype), int(rows), int(cols), int(batch_size)))


def diagonal(diagonal_values: Any, *, batch_size: int | None = None):
    vector = _coerce_dense_vector(diagonal_values)
    inferred = int(vector.batch_size)
    if batch_size is None:
        batch_size = inferred
    elif inferred != 1 and inferred != batch_size:
        raise ValueError("batch_size does not match diagonal_values batch dimension")
    return _unwrap(_ext._diagonal_dense(vector, int(batch_size)))


def triangular(
    n: int,
    *,
    uplo: str = "lower",
    diagonal_value: Any = 1,
    non_diagonal_value: Any = 0,
    dtype: Any = np.float64,
    batch_size: int = 1,
):
    return _unwrap(
        _ext._triangular_dense(
            _normalize_dtype(dtype),
            int(n),
            uplo,
            diagonal_value,
            non_diagonal_value,
            int(batch_size),
        )
    )


def tridiag_toeplitz(
    n: int,
    *,
    diagonal_value: Any = 1,
    sub_diagonal_value: Any = -1,
    super_diagonal_value: Any = -1,
    dtype: Any = np.float64,
    batch_size: int = 1,
):
    return _unwrap(
        _ext._tridiag_toeplitz_dense(
            _normalize_dtype(dtype),
            int(n),
            diagonal_value,
            sub_diagonal_value,
            super_diagonal_value,
            int(batch_size),
        )
    )


def random_sparse_hermitian(
    n: int,
    *,
    density: float = 0.05,
    dtype: Any = np.float64,
    batch_size: int = 1,
    seed: int = 42,
    diagonal_boost: Any = 1.0,
    shared_pattern: bool = False,
):
    return _unwrap(
        _ext._random_sparse_hermitian(
            _normalize_dtype(dtype),
            int(n),
            float(density),
            int(batch_size),
            int(seed),
            diagonal_boost,
            bool(shared_pattern),
        )
    )


def _conditioned_random(
    which: str,
    *,
    n: int,
    log10_kappa: float,
    metric: str = "spectral",
    dtype: Any = np.float64,
    batch_size: int = 1,
    seed: int = 42,
    kd: int = 0,
    algorithm: str = "chol2",
    backend: str = "auto",
    device: str | None = None,
):
    return _unwrap(
        _ext._conditioned_random_dense(
            which,
            _normalize_dtype(dtype),
            int(n),
            float(log10_kappa),
            metric,
            int(batch_size),
            int(seed),
            int(kd),
            algorithm,
            _normalize_backend(backend, device),
            _normalize_device(device),
        )
    )


def random_with_log10_cond_metric(**kwargs):
    return _conditioned_random("random_with_log10_cond_metric", **kwargs)


def random_hermitian_with_log10_cond_metric(**kwargs):
    return _conditioned_random("random_hermitian_with_log10_cond_metric", **kwargs)


def random_banded_with_log10_cond_metric(**kwargs):
    return _conditioned_random("random_banded_with_log10_cond_metric", **kwargs)


def random_hermitian_banded_with_log10_cond_metric(**kwargs):
    return _conditioned_random("random_hermitian_banded_with_log10_cond_metric", **kwargs)


def random_tridiagonal_with_log10_cond_metric(**kwargs):
    return _conditioned_random("random_tridiagonal_with_log10_cond_metric", **kwargs)


def random_hermitian_tridiagonal_with_log10_cond_metric(**kwargs):
    return _conditioned_random("random_hermitian_tridiagonal_with_log10_cond_metric", **kwargs)
