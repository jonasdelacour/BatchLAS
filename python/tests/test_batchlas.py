from __future__ import annotations

import numpy as np
import pytest

import batchlas as bl

sp = pytest.importorskip("scipy.sparse")


def _skip_if_unavailable(exc: Exception) -> None:
    if isinstance(exc, (NotImplementedError, RuntimeError)):
        pytest.skip(str(exc))
    raise exc


def _available_device_types() -> set[str]:
    return {
        str(device.get("type", "")).lower()
        for device in bl.available_devices()
        if isinstance(device, dict)
    }


def test_import_surface():
    features = bl.compiled_features()
    assert isinstance(bl.available_backends(), list)
    assert isinstance(bl.available_devices(), list)
    assert isinstance(features, dict)
    assert "backends" in features


def test_gemm_and_out_contract():
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64)
    out = np.empty((2, 2), dtype=np.float64)

    result = bl.gemm(a, b, backend="netlib", out=out)

    assert result is out
    np.testing.assert_allclose(out, a @ b)
    np.testing.assert_allclose(a, np.array([[1.0, 2.0], [3.0, 4.0]]))
    np.testing.assert_allclose(b, np.array([[5.0, 6.0], [7.0, 8.0]]))


def test_norm_batch_shape_is_vector():
    batch = np.stack([np.eye(2), 2.0 * np.eye(2)]).astype(np.float64)

    values = bl.norm(batch, "fro")

    assert values.shape == (2,)
    np.testing.assert_allclose(values, np.array([np.sqrt(2.0), np.sqrt(8.0)]))


def test_getrf_getrs_roundtrip():
    a = np.array([[4.0, 1.0], [2.0, 3.0]], dtype=np.float64)
    b = np.array([[1.0], [0.0]], dtype=np.float64)

    lu, pivots = bl.getrf(a, backend="netlib")
    x = bl.getrs(lu, b, pivots, backend="netlib")

    np.testing.assert_allclose(a @ x, b, rtol=1e-10, atol=1e-10)


def test_gemm_heterogeneous_returns_python_list():
    a = [
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
        np.array([[2.0, 0.0, 1.0]], dtype=np.float64),
    ]
    b = [
        np.array([[1.0], [2.0]], dtype=np.float64),
        np.array([[1.0], [0.0], [1.0]], dtype=np.float64),
    ]

    result = bl.gemm_heterogeneous(a, b, backend="netlib")

    assert isinstance(result, list)
    assert len(result) == 2
    np.testing.assert_allclose(result[0], a[0] @ b[0])
    np.testing.assert_allclose(result[1], a[1] @ b[1])


def test_sparse_spmm_and_transpose():
    a = sp.csr_matrix(np.array([[2.0, 0.0], [1.0, 3.0]], dtype=np.float64))
    b = np.array([[1.0], [2.0]], dtype=np.float64)

    try:
        c = bl.spmm(a, b, backend="auto")
        at = bl.transpose(a)
    except Exception as exc:  # pragma: no cover - backend/runtime dependent
        _skip_if_unavailable(exc)

    np.testing.assert_allclose(c, a.toarray() @ b)
    np.testing.assert_allclose(at.toarray(), a.toarray().T)


def test_syevx_history_contract():
    a = np.array([[3.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    options = bl.SyevxOptions(iterations=4, store_every=1)

    try:
        values, vectors, history = bl.syevx(
            a,
            1,
            compute_vectors=True,
            options=options,
            backend="auto",
            return_history=True,
        )
    except Exception as exc:  # pragma: no cover - backend/runtime dependent
        _skip_if_unavailable(exc)

    assert values.shape == (1,)
    assert vectors.shape == (2, 1)
    assert history["best_residual_history"].shape == (4, 1, 1)
    assert history["iterations_done"].shape == (1,)


def test_iluk_identity_apply():
    a = sp.csr_matrix(np.eye(3, dtype=np.float64))
    rhs = np.array([[1.0], [2.0], [3.0]], dtype=np.float64)

    try:
        handle = bl.iluk_factorize(a, backend="auto")
        out = bl.iluk_apply(handle, rhs, backend="auto")
    except Exception as exc:  # pragma: no cover - backend/runtime dependent
        _skip_if_unavailable(exc)

    np.testing.assert_allclose(out, rhs, rtol=1e-10, atol=1e-10)


def test_sparse_syevx_accepts_iluk_preconditioner():
    if "cuda" not in {name.lower() for name in bl.available_backends()}:
        pytest.skip("CUDA backend unavailable")
    if "gpu" not in _available_device_types():
        pytest.skip("GPU device unavailable")

    n = 16
    base = 4.0 * np.eye(n, dtype=np.float64)
    offdiag = -0.25 * np.ones(n - 1, dtype=np.float64)
    base[np.arange(n - 1), np.arange(1, n)] = offdiag
    base[np.arange(1, n), np.arange(n - 1)] = offdiag

    matrices = [sp.csr_matrix(base), sp.csr_matrix(base + 0.1 * np.eye(n, dtype=np.float64))]
    options = bl.SyevxOptions(iterations=2, extra_directions=1, find_largest=True)

    try:
        handle = bl.iluk_factorize(matrices, backend="cuda", device="gpu")
        values = bl.syevx(
            matrices,
            2,
            compute_vectors=False,
            options=options,
            backend="cuda",
            device="gpu",
            preconditioner=handle,
        )
    except Exception as exc:  # pragma: no cover - backend/runtime dependent
        _skip_if_unavailable(exc)

    values = np.asarray(values)
    assert values.shape == (2, 2)
    assert np.isfinite(values).all()
