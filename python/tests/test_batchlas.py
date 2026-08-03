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
    # ILU(k) approximates A^-1, so it is only a valid preconditioner for the
    # smallest eigenpairs; syevx rejects find_largest=True with a preconditioner.
    options = bl.SyevxOptions(iterations=2, extra_directions=1, find_largest=False)

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


def test_gesvd_accepts_hermitian_uplo_for_complex_input():
    a = np.array(
        [
            [2.0 + 0.0j, 1.0 - 2.0j],
            [1.0 + 2.0j, -3.0 + 0.0j],
        ],
        dtype=np.complex64,
    )

    try:
        u, s, vh = bl.gesvd(a, uplo="lower", backend="auto")
    except Exception as exc:  # pragma: no cover - backend/runtime dependent
        _skip_if_unavailable(exc)

    assert s.shape == (2,)
    assert u.shape == (2, 2)
    assert vh.shape == (2, 2)

    _, ref_s, _ = np.linalg.svd(a, full_matrices=True)
    np.testing.assert_allclose(np.asarray(s), ref_s, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(np.asarray(u) @ np.diag(np.asarray(s)) @ np.asarray(vh), a, rtol=5e-4, atol=5e-4)


def _symmetric_batch(batch: int, n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((batch, n, n))
    return (a + a.transpose(0, 2, 1)) / 2.0


def _tridiagonal(d: np.ndarray, e: np.ndarray) -> np.ndarray:
    return np.stack(
        [np.diag(d[i]) + np.diag(e[i], -1) + np.diag(e[i], 1) for i in range(d.shape[0])]
    )


def _gpu_device_or_skip() -> str:
    if "gpu" not in _available_device_types():
        pytest.skip("CTA routines require a GPU with sub-group width 32")
    return "gpu"


def test_beta_requires_an_out_operand():
    a = np.eye(2)
    with pytest.raises(ValueError):
        bl.gemm(a, a, beta=1.0)


def test_gemm_applies_beta_to_the_out_operand():
    rng = np.random.default_rng(0)
    a, b, c = (rng.standard_normal((2, 3, 3)) for _ in range(3))

    got = bl.gemm(a, b, alpha=2.0, beta=0.5, out=c.copy())

    np.testing.assert_allclose(got, 2.0 * (a @ b) + 0.5 * c, rtol=1e-10, atol=1e-10)


def test_gemv_applies_beta_to_the_out_operand():
    rng = np.random.default_rng(1)
    a = rng.standard_normal((2, 3, 3))
    x = rng.standard_normal((2, 3))
    y = rng.standard_normal((2, 3))

    got = bl.gemv(a, x, alpha=2.0, beta=-1.0, out=y.copy())

    expected = 2.0 * np.einsum("bij,bj->bi", a, x) - y
    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-10)


def test_syev_without_eigenvectors_returns_the_full_spectrum():
    a = _symmetric_batch(3, 8, seed=2)

    try:
        values = bl.syev(a, compute_vectors=False)
    except Exception as exc:  # pragma: no cover - backend/runtime dependent
        _skip_if_unavailable(exc)

    assert np.asarray(values).shape == (3, 8)
    np.testing.assert_allclose(np.sort(values, axis=-1), np.linalg.eigvalsh(a), rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize("name", ["stedc", "stedc_flat", "steqr", "steqr_cta"])
def test_tridiagonal_solvers_without_eigenvectors(name):
    rng = np.random.default_rng(3)
    d = rng.standard_normal((2, 16))
    e = rng.standard_normal((2, 15))
    reference = np.linalg.eigvalsh(_tridiagonal(d, e))

    try:
        values = getattr(bl, name)(d, e, compute_vectors=False)
    except Exception as exc:  # pragma: no cover - backend/runtime dependent
        _skip_if_unavailable(exc)

    np.testing.assert_allclose(np.sort(values, axis=-1), reference, rtol=1e-8, atol=1e-8)


def test_syev_two_stage_without_eigenvectors():
    a = _symmetric_batch(2, 48, seed=4)

    try:
        values = bl.syev_two_stage(a, compute_vectors=False)
    except Exception as exc:  # pragma: no cover - backend/runtime dependent
        _skip_if_unavailable(exc)

    np.testing.assert_allclose(np.sort(values, axis=-1), np.linalg.eigvalsh(a), rtol=1e-8, atol=1e-8)


def test_nested_option_dataclasses_are_accepted():
    a = _symmetric_batch(2, 32, seed=5)
    options = bl.StedcOptions(recursion_threshold=8, leaf_steqr_params=bl.SteqrOptions(max_sweeps=200))

    try:
        values = bl.syev_blocked(a, compute_vectors=False, options=options)
    except Exception as exc:  # pragma: no cover - backend/runtime dependent
        _skip_if_unavailable(exc)

    np.testing.assert_allclose(np.sort(values, axis=-1), np.linalg.eigvalsh(a), rtol=1e-8, atol=1e-8)


def test_syev_jacobi_cta_matches_reference():
    device = _gpu_device_or_skip()
    a = _symmetric_batch(4, 16, seed=6)

    try:
        values, vectors = bl.syev_jacobi_cta(a, device=device)
    except Exception as exc:  # pragma: no cover - backend/runtime dependent
        _skip_if_unavailable(exc)

    np.testing.assert_allclose(np.sort(values, axis=-1), np.linalg.eigvalsh(a), rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(a @ vectors, vectors * values[:, None, :], rtol=1e-9, atol=1e-9)


def test_syev_jacobi_cta_keeps_relative_accuracy_on_graded_input():
    device = _gpu_device_or_skip()
    rng = np.random.default_rng(7)
    n = 16
    q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    core = q @ np.diag(np.linspace(1.0, 2.0, n)) @ q.T
    scale = np.diag(np.logspace(0, -8, n))
    a = (scale @ ((core + core.T) / 2.0) @ scale)[None, ...]
    reference = np.linalg.eigvalsh(a)

    try:
        values = bl.syev_jacobi_cta(a, compute_vectors=False, device=device)
    except Exception as exc:  # pragma: no cover - backend/runtime dependent
        _skip_if_unavailable(exc)

    relative = np.abs(np.sort(values, axis=-1) - reference) / np.abs(reference)
    assert relative.max() < 1e-10


def test_syev_variant_support_reports_capabilities():
    a = _symmetric_batch(2, 16, seed=8)

    support = bl.syev_variant_support(a)

    for key in ("device", "is_gpu", "max_sub_group", "cta", "blocked", "two_stage"):
        assert key in support
    assert isinstance(support["cta"], bool)


@pytest.mark.parametrize(
    "call",
    [
        lambda a, device: bl.sytrd_cta(a, device=device),
        lambda a, device: bl.sytrd_blocked(a, block_size=16, device=device),
    ],
)
def test_sytrd_preserves_the_spectrum(call):
    device = _gpu_device_or_skip()
    a = _symmetric_batch(2, 32, seed=9)

    try:
        _, d, e, _ = call(a, device)
    except Exception as exc:  # pragma: no cover - backend/runtime dependent
        _skip_if_unavailable(exc)

    got = np.linalg.eigvalsh(_tridiagonal(np.asarray(d), np.asarray(e)))
    np.testing.assert_allclose(got, np.linalg.eigvalsh(a), rtol=1e-8, atol=1e-8)


def test_two_stage_band_reduction_pipeline():
    device = _gpu_device_or_skip()
    a = _symmetric_batch(2, 64, seed=10)
    kd = 8
    reference = np.linalg.eigvalsh(a)

    try:
        _, ab, _ = bl.sytrd_sy2sb(a, kd, device=device)
        d, e, _ = bl.sytrd_sb2st(ab, kd, block_size=16, device=device)
    except Exception as exc:  # pragma: no cover - backend/runtime dependent
        _skip_if_unavailable(exc)

    assert ab.shape == (2, kd + 1, 64)
    got = np.linalg.eigvalsh(_tridiagonal(np.asarray(d), np.asarray(e)))
    np.testing.assert_allclose(got, reference, rtol=1e-7, atol=1e-7)

    d_alias, e_alias, _ = bl.hetrd_hb2st(ab, kd, block_size=16, device=device)
    np.testing.assert_array_equal(np.asarray(d), np.asarray(d_alias))
    np.testing.assert_array_equal(np.asarray(e), np.asarray(e_alias))

    d_band, e_band, _ = bl.sytrd_band_reduction(
        ab, kd, options=bl.SytrdBandReductionOptions(block_size_seq=[16]), device=device
    )
    got = np.linalg.eigvalsh(_tridiagonal(np.asarray(d_band), np.asarray(e_band)))
    np.testing.assert_allclose(got, reference, rtol=1e-7, atol=1e-7)


@pytest.mark.parametrize("name", ["gebrd_cta", "gebrd_blocked"])
def test_gebrd_variants_preserve_singular_values(name):
    device = _gpu_device_or_skip()
    rng = np.random.default_rng(11)
    a = rng.standard_normal((2, 16, 16))

    try:
        _, d, e, tauq, taup = getattr(bl, name)(a, device=device)
    except Exception as exc:  # pragma: no cover - backend/runtime dependent
        _skip_if_unavailable(exc)

    assert np.asarray(tauq).shape == (2, 16)
    assert np.asarray(taup).shape == (2, 16)
    bidiagonal = np.stack(
        [np.diag(np.asarray(d)[i]) + np.diag(np.asarray(e)[i], 1) for i in range(2)]
    )
    got = np.sort(np.linalg.svd(bidiagonal, compute_uv=False))[:, ::-1]
    np.testing.assert_allclose(got, np.linalg.svd(a, compute_uv=False), rtol=1e-8, atol=1e-8)


def test_gesvd_cta_reconstructs_the_input():
    device = _gpu_device_or_skip()
    rng = np.random.default_rng(12)
    a = rng.standard_normal((2, 16, 16))

    try:
        u, s, vh = bl.gesvd_cta(a, device=device)
    except Exception as exc:  # pragma: no cover - backend/runtime dependent
        _skip_if_unavailable(exc)

    np.testing.assert_allclose(
        np.sort(s, axis=-1)[:, ::-1], np.linalg.svd(a, compute_uv=False), rtol=1e-8, atol=1e-8
    )
    np.testing.assert_allclose(u @ (s[:, :, None] * vh), a, rtol=1e-8, atol=1e-8)

    values_only = bl.gesvd_cta(a, compute_vectors=False, device=device)
    np.testing.assert_allclose(np.sort(values_only, axis=-1), np.sort(s, axis=-1), rtol=1e-8, atol=1e-8)


def test_triangular_generator_fills_every_element():
    n = 6
    lower = bl.triangular(n, uplo="lower", diagonal_value=2.0, non_diagonal_value=1.0)

    np.testing.assert_array_equal(lower, np.tril(lower))
    np.testing.assert_array_equal(np.diag(lower), np.full(n, 2.0))
    np.testing.assert_array_equal(lower[np.tril_indices(n, -1)], np.ones(n * (n - 1) // 2))

    upper = bl.triangular(n, uplo="upper", diagonal_value=2.0, non_diagonal_value=1.0)
    np.testing.assert_array_equal(upper, np.triu(upper))
    np.testing.assert_array_equal(upper[np.triu_indices(n, 1)], np.ones(n * (n - 1) // 2))

    batched = bl.triangular(n, batch_size=3, diagonal_value=2.0, non_diagonal_value=1.0)
    assert batched.shape == (3, n, n)
    np.testing.assert_array_equal(batched[0], batched[2])


def test_tridiagonal_solver_handles_batches_consistently():
    rng = np.random.default_rng(13)
    n = 8
    d = rng.standard_normal((1, n))
    e = rng.standard_normal((1, n - 1))
    batched_d = np.repeat(d, 3, axis=0)
    batched_e = np.repeat(e, 3, axis=0)

    try:
        single = bl.tridiagonal_solver(d, e, compute_vectors=False)
        batched = bl.tridiagonal_solver(batched_d, batched_e, compute_vectors=False)
    except Exception as exc:  # pragma: no cover - backend/runtime dependent
        _skip_if_unavailable(exc)

    # Every batch entry solves the same problem, so they must agree with the
    # single-matrix call; a stride bug in the beta buffer shows up here.
    for index in range(3):
        np.testing.assert_allclose(
            np.sort(np.asarray(batched)[index]), np.sort(np.ravel(single)), rtol=1e-8, atol=1e-8
        )


def test_lanczos_without_eigenvectors_runs():
    a = _symmetric_batch(2, 32, seed=14)

    try:
        values = bl.lanczos(a, compute_vectors=False)
    except Exception as exc:  # pragma: no cover - backend/runtime dependent
        _skip_if_unavailable(exc)

    assert np.asarray(values).shape == (2, 32)
    assert np.isfinite(np.asarray(values)).all()


# --- elementwise arithmetic (batchlas::linalg) ------------------------------


@pytest.mark.parametrize(
    "op, reference",
    [
        (bl.add, lambda a, b: a + b),
        (bl.subtract, lambda a, b: a - b),
        (bl.multiply, lambda a, b: a * b),
        (bl.divide, lambda a, b: a / b),
    ],
)
def test_elementwise_matches_numpy(op, reference):
    rng = np.random.default_rng(0)
    # Non-square and batched: a shape bug that only shows up off the diagonal
    # would pass unnoticed on square single-batch inputs.
    a = rng.random((4, 3, 5)).astype(np.float64) + 1.0
    b = rng.random((4, 3, 5)).astype(np.float64) + 1.0
    np.testing.assert_allclose(op(a, b), reference(a, b), rtol=1e-12)


def test_elementwise_two_dimensional_keeps_shape():
    a = np.arange(12, dtype=np.float32).reshape(3, 4) + 1.0
    b = np.full((3, 4), 2.0, dtype=np.float32)
    result = bl.add(a, b)
    assert result.shape == (3, 4)
    np.testing.assert_allclose(result, a + b, rtol=1e-6)


def test_multiply_is_hadamard_not_matmul():
    # Square operands make both interpretations shape-valid, so only the values
    # distinguish them.
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64)
    np.testing.assert_allclose(bl.multiply(a, b), a * b)
    assert not np.allclose(bl.multiply(a, b), a @ b)


def test_axpby_and_scale():
    rng = np.random.default_rng(1)
    a = rng.random((2, 4, 4)).astype(np.float64)
    b = rng.random((2, 4, 4)).astype(np.float64)
    np.testing.assert_allclose(bl.axpby(2.0, a, -3.0, b), 2.0 * a - 3.0 * b, rtol=1e-12)
    np.testing.assert_allclose(bl.scale(a, 0.25), 0.25 * a, rtol=1e-12)


def test_scale_leaves_the_caller_array_untouched():
    a = np.ones((3, 3), dtype=np.float64)
    original = a.copy()
    bl.scale(a, 5.0)
    np.testing.assert_array_equal(a, original)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_elementwise_supports_all_dtypes(dtype):
    rng = np.random.default_rng(2)
    shape = (3, 3)
    if np.issubdtype(dtype, np.complexfloating):
        a = (rng.random(shape) + 1j * rng.random(shape)).astype(dtype)
        b = (rng.random(shape) + 1j * rng.random(shape)).astype(dtype)
    else:
        a = rng.random(shape).astype(dtype)
        b = rng.random(shape).astype(dtype)
    tol = 1e-6 if np.dtype(dtype).itemsize <= 8 else 1e-12
    np.testing.assert_allclose(bl.multiply(a, b), a * b, rtol=tol)


def test_elementwise_rejects_mismatched_shapes():
    a = np.ones((3, 4), dtype=np.float32)
    b = np.ones((3, 5), dtype=np.float32)
    with pytest.raises(ValueError):
        bl.add(a, b)


def test_elementwise_out_parameter_is_written():
    a = np.ones((2, 2), dtype=np.float64)
    b = np.full((2, 2), 3.0, dtype=np.float64)
    out = np.zeros((2, 2), dtype=np.float64)
    bl.add(a, b, out=out)
    np.testing.assert_allclose(out, a + b)
