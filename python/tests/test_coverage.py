# SPDX-License-Identifier: MIT
"""NumPy-referenced tests for bound routines that test_batchlas.py does not reach.

Every case runs batch > 1 with distinct items, feeds at least one non-contiguous operand (the
binding copies through the array's own strides), and, where the routine reads one triangle,
poisons the other with large finite values. Complex cases carry nonzero imaginary parts.
"""

from __future__ import annotations

import numpy as np
import pytest

import batchlas as bl

try:
    import scipy.sparse as sp
except ImportError:  # pragma: no cover - only the sparse generator needs SciPy
    sp = None

REAL = [np.float32, np.float64]
ALL = [np.float32, np.float64, np.complex64, np.complex128]
POISON = 1.0e6


def _tol(dtype) -> float:
    return 2e-4 if np.dtype(dtype) in (np.float32, np.complex64) else 1e-10


def _rand(rng, shape, dtype):
    out = rng.standard_normal(shape)
    if np.issubdtype(dtype, np.complexfloating):
        out = out + 1j * rng.standard_normal(shape)
    return out.astype(dtype)


def _strided(a):
    """The same values as `a`, as a view whose last axis has stride 2."""
    wide = np.zeros(a.shape[:-1] + (2 * a.shape[-1],), dtype=a.dtype)
    wide[..., ::2] = a
    view = wide[..., ::2]
    assert not view.flags.c_contiguous
    return view


def _h(a):
    return np.conj(np.swapaxes(a, -1, -2))


def _op(a, trans):
    return {"n": a, "t": np.swapaxes(a, -1, -2), "c": _h(a)}[trans]


def _tri(a, uplo, k=0):
    return np.tril(a, k) if uplo == "lower" else np.triu(a, -k if k else 0)


def _poison_other_triangle(a, uplo):
    """Keep `uplo` (with diagonal); fill the strict other triangle with POISON."""
    keep = np.tril(np.ones(a.shape[-2:], dtype=bool)) if uplo == "lower" else np.triu(np.ones(a.shape[-2:], dtype=bool))
    return np.where(keep, a, np.asarray(POISON, dtype=a.dtype))


def _assert_close(got, want, dtype, scale=1.0):
    tol = _tol(dtype) * scale
    np.testing.assert_allclose(np.asarray(got), want, rtol=tol, atol=tol * max(1.0, float(np.abs(want).max())))


def _hpd(rng, batch, n, dtype):
    x = _rand(rng, (batch, n, n), dtype)
    return (x @ _h(x) + n * np.eye(n)).astype(dtype)


def _require_gpu() -> str:
    if "gpu" not in {str(d.get("type", "")).lower() for d in bl.available_devices() if isinstance(d, dict)}:
        pytest.skip("CTA routines require a GPU with sub-group width 32")
    return "gpu"


# --- level 3 -----------------------------------------------------------------


@pytest.mark.parametrize("dtype", REAL)
@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("uplo", ["lower", "upper"])
def test_symm_reads_one_triangle_and_applies_beta(dtype, side, uplo):
    rng = np.random.default_rng(1)
    batch, m, n = 3, 5, 4
    order = m if side == "left" else n
    full = _rand(rng, (batch, order, order), dtype)
    full = (full + np.swapaxes(full, -1, -2)) / 2
    b = _rand(rng, (batch, m, n), dtype)
    c = _rand(rng, (batch, m, n), dtype)

    got = bl.symm(_poison_other_triangle(full, uplo), _strided(b), alpha=1.5, beta=-0.5, side=side, uplo=uplo,
                  out=c.copy())

    want = 1.5 * (full @ b if side == "left" else b @ full) - 0.5 * c
    _assert_close(got, want, dtype, scale=10)


@pytest.mark.parametrize("dtype", REAL)
@pytest.mark.parametrize("trans", ["n", "t"])
@pytest.mark.parametrize("uplo", ["lower", "upper"])
def test_syrk_writes_the_requested_triangle(dtype, trans, uplo):
    rng = np.random.default_rng(2)
    batch, n, k = 3, 6, 4
    a = _rand(rng, (batch, n, k) if trans == "n" else (batch, k, n), dtype)
    c = _rand(rng, (batch, n, n), dtype)
    c = (c + np.swapaxes(c, -1, -2)) / 2

    got = bl.syrk(_strided(a), alpha=2.0, beta=0.25, uplo=uplo, trans_a=trans, out=c.copy())

    product = a @ np.swapaxes(a, -1, -2) if trans == "n" else np.swapaxes(a, -1, -2) @ a
    want = 2.0 * product + 0.25 * c
    _assert_close(_tri(np.asarray(got), uplo), _tri(want, uplo), dtype, scale=10)


@pytest.mark.parametrize("dtype", REAL)
@pytest.mark.parametrize("trans", ["n", "t"])
@pytest.mark.parametrize("uplo", ["lower", "upper"])
def test_syr2k_writes_the_requested_triangle(dtype, trans, uplo):
    rng = np.random.default_rng(3)
    batch, n, k = 3, 6, 4
    shape = (batch, n, k) if trans == "n" else (batch, k, n)
    a, b = _rand(rng, shape, dtype), _rand(rng, shape, dtype)
    c = _rand(rng, (batch, n, n), dtype)

    got = bl.syr2k(a, _strided(b), alpha=0.5, beta=1.5, uplo=uplo, trans_a=trans, out=c.copy())

    at, bt = (a, b) if trans == "n" else (np.swapaxes(a, -1, -2), np.swapaxes(b, -1, -2))
    want = 0.5 * (at @ np.swapaxes(bt, -1, -2) + bt @ np.swapaxes(at, -1, -2)) + 1.5 * c
    _assert_close(_tri(np.asarray(got), uplo), _tri(want, uplo), dtype, scale=10)


def _triangular_operand(rng, batch, n, dtype, uplo, diag):
    a = _rand(rng, (batch, n, n), dtype) * 0.3
    a = a + 2.0 * np.eye(n, dtype=dtype)
    poisoned = _poison_other_triangle(a, uplo)
    if diag == "unit":
        idx = np.arange(n)
        poisoned = poisoned.copy()
        poisoned[:, idx, idx] = POISON
    effective = _tri(a, uplo)
    if diag == "unit":
        effective = effective.copy()
        effective[:, np.arange(n), np.arange(n)] = 1.0
    return poisoned, effective


@pytest.mark.parametrize("dtype", ALL)
@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("uplo", ["lower", "upper"])
@pytest.mark.parametrize("trans", ["n", "t", "c"])
@pytest.mark.parametrize("diag", ["non_unit", "unit"])
def test_trmm_matches_numpy(dtype, side, uplo, trans, diag):
    rng = np.random.default_rng(4)
    batch, m, n = 3, 5, 4
    order = m if side == "left" else n
    a, effective = _triangular_operand(rng, batch, order, dtype, uplo, diag)
    b = _rand(rng, (batch, m, n), dtype)
    alpha = 1.5 - 0.5j if np.issubdtype(dtype, np.complexfloating) else 1.5

    got = bl.trmm(a, _strided(b), alpha=alpha, side=side, uplo=uplo, trans_a=trans, diag=diag)

    op = _op(effective, trans)
    want = alpha * (op @ b if side == "left" else b @ op)
    _assert_close(got, want, dtype, scale=10)


@pytest.mark.parametrize("dtype", ALL)
@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("uplo", ["lower", "upper"])
@pytest.mark.parametrize("trans", ["n", "t", "c"])
@pytest.mark.parametrize("diag", ["non_unit", "unit"])
def test_trsm_matches_numpy(dtype, side, uplo, trans, diag):
    rng = np.random.default_rng(5)
    batch, m, n = 3, 5, 4
    order = m if side == "left" else n
    a, effective = _triangular_operand(rng, batch, order, dtype, uplo, diag)
    b = _rand(rng, (batch, m, n), dtype)
    alpha = 0.5 + 0.25j if np.issubdtype(dtype, np.complexfloating) else 0.5

    got = bl.trsm(a, _strided(b), alpha=alpha, side=side, uplo=uplo, trans_a=trans, diag=diag)

    op = _op(effective, trans).astype(np.complex128 if np.issubdtype(dtype, np.complexfloating) else np.float64)
    want = alpha * (np.linalg.solve(op, b) if side == "left" else np.swapaxes(
        np.linalg.solve(np.swapaxes(op, -1, -2), np.swapaxes(b, -1, -2)), -1, -2))
    _assert_close(got, want, dtype, scale=10)


# --- factorizations ------------------------------------------------------------


@pytest.mark.parametrize("dtype", ALL)
@pytest.mark.parametrize("uplo", ["lower", "upper"])
def test_potrf_factors_the_referenced_triangle(dtype, uplo):
    rng = np.random.default_rng(6)
    batch, n = 3, 6
    a = _hpd(rng, batch, n, dtype)

    got = np.asarray(bl.potrf(_strided(_poison_other_triangle(a, uplo)), uplo=uplo))

    factor = _tri(got, uplo)
    rebuilt = factor @ _h(factor) if uplo == "lower" else _h(factor) @ factor
    _assert_close(rebuilt, a, dtype, scale=10)
    assert np.all(np.diagonal(factor, axis1=-2, axis2=-1).real > 0)


@pytest.mark.parametrize("dtype", ALL)
def test_getri_and_inv_invert(dtype):
    rng = np.random.default_rng(7)
    batch, n = 3, 6
    a = (_rand(rng, (batch, n, n), dtype) + n * np.eye(n)).astype(dtype)
    want = np.linalg.inv(a.astype(np.complex128 if np.issubdtype(dtype, np.complexfloating) else np.float64))

    lu, pivots = bl.getrf(_strided(a))
    _assert_close(bl.getri(lu, pivots), want, dtype, scale=10)
    _assert_close(bl.inv(_strided(a)), want, dtype, scale=10)


@pytest.mark.parametrize("dtype", ALL)
def test_one_dimensional_results_are_not_aliased(dtype):
    # pybind11 < 2.12 under NumPy 2 gave every count-constructed 1-D result stride 0, so all
    # entries read back as the last one written. A 2-D input returns 1-D values.
    rng = np.random.default_rng(17)
    a = _rand(rng, (6, 6), dtype)
    a = ((a + _h(a)) / 2).astype(dtype)

    values = np.asarray(bl.syev(a, compute_vectors=False))

    assert values.ndim == 1 and values.strides[0] == values.itemsize
    _assert_close(np.sort(values), np.linalg.eigvalsh(a), dtype, scale=10)


@pytest.mark.parametrize("dtype", REAL)
def test_getrf_pivots_are_per_item_and_one_based(dtype):
    # Item 0 needs no interchange, item 1 needs one at step 0: a pivot array that repeats one value
    # (the stride-0 output bug) or is 0-based cannot satisfy both.
    a = np.array([[[4.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [5.0, 1.0]]], dtype=dtype)

    _, pivots = bl.getrf(a)

    np.testing.assert_array_equal(np.asarray(pivots)[:, 0], [1, 2])


def _householder_q(qr, tau):
    """Full m x m Q = H_0 H_1 ... H_{k-1} from LAPACK-format reflectors, in complex128."""
    m = qr.shape[-2]
    q = np.eye(m, dtype=np.complex128)
    for i in range(len(tau)):
        v = np.zeros(m, dtype=np.complex128)
        v[i] = 1.0
        v[i + 1:] = qr[i + 1:, i]
        q = q @ (np.eye(m) - tau[i] * np.outer(v, np.conj(v)))
    return q


@pytest.mark.parametrize("dtype", ALL)
def test_geqrf_orgqr_ormqr_agree_with_explicit_reflectors(dtype):
    rng = np.random.default_rng(8)
    batch, m, n = 3, 7, 4
    a = _rand(rng, (batch, m, n), dtype)

    qr, tau = bl.geqrf(_strided(a))
    qr, tau = np.asarray(qr), np.asarray(tau)
    assert tau.shape == (batch, n)
    q_thin = np.asarray(bl.orgqr(qr, tau))

    for item in range(batch):
        q_full = _householder_q(qr[item], tau[item])
        _assert_close(q_full.conj().T @ q_full, np.eye(m), dtype, scale=10)
        _assert_close(q_thin[item], q_full[:, :n], dtype, scale=10)
        _assert_close(q_full[:, :n] @ np.triu(qr[item][:n, :]), a[item], dtype, scale=10)

    c_left = _rand(rng, (batch, m, 3), dtype)
    c_right = _rand(rng, (batch, 3, m), dtype)
    conj_trans = "c" if np.issubdtype(dtype, np.complexfloating) else "t"
    q_all = np.stack([_householder_q(qr[i], tau[i]) for i in range(batch)])
    _assert_close(bl.ormqr(qr, _strided(c_left), tau, side="left", trans="n"), q_all @ c_left, dtype, scale=10)
    _assert_close(bl.ormqr(qr, c_left, tau, side="left", trans=conj_trans), _h(q_all) @ c_left, dtype, scale=10)
    _assert_close(bl.ormqr(qr, c_right, tau, side="right", trans="n"), c_right @ q_all, dtype, scale=10)


# --- SVD pieces ----------------------------------------------------------------


# Complex has no native gesvd_blocked path; the library raises batchlas::unsupported for it.
@pytest.mark.parametrize("dtype", REAL)
def test_gesvd_blocked_reconstructs(dtype):
    rng = np.random.default_rng(9)
    batch, n = 3, 12
    a = _rand(rng, (batch, n, n), dtype)

    u, s, vh = bl.gesvd_blocked(_strided(a))

    _assert_close(np.sort(np.asarray(s), axis=-1)[:, ::-1], np.linalg.svd(a, compute_uv=False), dtype, scale=10)
    _assert_close(np.asarray(u) @ (np.asarray(s)[:, :, None] * np.asarray(vh)), a, dtype, scale=10)


# gebrd_unblocked is real-only in the library (batchlas::unsupported for complex).
@pytest.mark.parametrize("dtype", REAL)
def test_gebrd_unblocked_and_ormbr_reduce_to_the_bidiagonal(dtype):
    rng = np.random.default_rng(10)
    batch, n = 3, 8
    a = _rand(rng, (batch, n, n), dtype)

    reflectors, d, e, tauq, taup = bl.gebrd_unblocked(_strided(a))
    d, e = np.asarray(d), np.asarray(e)
    assert np.asarray(tauq).shape == (batch, n) and np.asarray(taup).shape == (batch, n)
    bidiagonal = np.stack([np.diag(d[i]) + np.diag(e[i], 1) for i in range(batch)])
    _assert_close(np.linalg.svd(bidiagonal, compute_uv=False), np.linalg.svd(a, compute_uv=False), dtype, scale=10)

    conj_trans = "c" if np.issubdtype(dtype, np.complexfloating) else "t"
    qha = bl.ormbr(reflectors, np.asarray(tauq), a, vect="Q", side="left", trans=conj_trans)
    qhap = bl.ormbr(reflectors, np.asarray(taup), qha, vect="P", side="right", trans="n")
    _assert_close(np.asarray(qhap), bidiagonal.astype(dtype), dtype, scale=100)


@pytest.mark.parametrize("dtype", REAL)
@pytest.mark.parametrize("sort_desc", [False, True])
def test_bdsqr_returns_the_bidiagonal_singular_values(dtype, sort_desc):
    rng = np.random.default_rng(11)
    batch, n = 3, 9
    d = rng.standard_normal((batch, n)).astype(dtype)
    e = rng.standard_normal((batch, n - 1)).astype(dtype)
    bidiagonal = np.stack([np.diag(d[i]) + np.diag(e[i], 1) for i in range(batch)])

    got = np.asarray(bl.bdsqr(_strided(d), e, sort_desc=sort_desc))

    want = np.linalg.svd(bidiagonal, compute_uv=False)
    # sort_desc=False means "no final sort", not ascending: compare the multiset then.
    _assert_close(got if sort_desc else np.sort(got, axis=-1)[:, ::-1], want, dtype, scale=10)


# --- eigen pieces ----------------------------------------------------------------


@pytest.mark.parametrize("dtype", ALL)
@pytest.mark.parametrize(
    "uplo",
    [
        "lower",
        pytest.param("upper", marks=pytest.mark.xfail(strict=True, reason=(
            "syev_cta Upper reads the lower triangle: "
            "docs/design/known-defects.md#defect-14-the-hermitian-drivers-read-the-unreferenced-triangle"))),
    ],
)
def test_syev_cta_eigenpairs(dtype, uplo):
    device = _require_gpu()
    rng = np.random.default_rng(12)
    batch, n = 3, 16
    a = _rand(rng, (batch, n, n), dtype)
    a = ((a + _h(a)) / 2).astype(dtype)

    values, vectors = bl.syev_cta(_poison_other_triangle(a, uplo), uplo=uplo, device=device)
    values, vectors = np.asarray(values), np.asarray(vectors)

    _assert_close(np.sort(values, axis=-1), np.linalg.eigvalsh(a), dtype, scale=10)
    _assert_close(a @ vectors, vectors * values[:, None, :], dtype, scale=100)


@pytest.mark.parametrize("dtype", ALL)
def test_ritz_values_are_rayleigh_quotients(dtype):
    rng = np.random.default_rng(13)
    batch, n, k = 3, 10, 3
    a = _rand(rng, (batch, n, n), dtype)
    a = ((a + _h(a)) / 2).astype(dtype)
    v, _ = np.linalg.qr(_rand(rng, (batch, n, k), dtype))
    v = v.astype(dtype)

    got = np.asarray(bl.ritz_values(a, _strided(v)))

    want = np.real(np.einsum("bik,bij,bjk->bk", np.conj(v), a, v))
    _assert_close(got, want, dtype, scale=10)


# --- utilities -----------------------------------------------------------------


@pytest.mark.parametrize("dtype", ALL)
def test_lascl_scales_by_cto_over_cfrom(dtype):
    rng = np.random.default_rng(14)
    a = _rand(rng, (3, 4, 5), dtype)

    got = bl.lascl(_strided(a), 4.0, 3.0)

    _assert_close(got, a * (3.0 / 4.0), dtype)


@pytest.mark.parametrize("dtype", REAL)
def test_cond_matches_numpy(dtype):
    rng = np.random.default_rng(15)
    batch, n = 3, 6
    a = _hpd(rng, batch, n, dtype) * np.array([1.0, 10.0, 100.0], dtype=dtype)[:, None, None]
    a[2] = a[2] + np.diag(np.logspace(0, 3, n)).astype(dtype)

    spectral = np.asarray(bl.cond(_strided(a), "spectral"))
    frobenius = np.asarray(bl.cond(a, "fro"))

    _assert_close(spectral, np.linalg.cond(a.astype(np.float64), 2), dtype, scale=100)
    _assert_close(frobenius, np.linalg.cond(a.astype(np.float64), "fro"), dtype, scale=100)


@pytest.mark.parametrize("dtype", ALL)
@pytest.mark.parametrize("algorithm", ["chol2", "cgs2", "svqb", "shiftchol3", "householder"])
def test_ortho_returns_an_orthonormal_basis_of_the_same_span(dtype, algorithm):
    rng = np.random.default_rng(16)
    batch, m, k = 3, 12, 4
    a = _rand(rng, (batch, m, k), dtype)

    q = np.asarray(bl.ortho(_strided(a), algorithm=algorithm))

    _assert_close(_h(q) @ q, np.broadcast_to(np.eye(k), (batch, k, k)), dtype, scale=100)
    _assert_close(q @ (_h(q) @ a), a, dtype, scale=100)


# --- generators ----------------------------------------------------------------


@pytest.mark.parametrize("dtype", ALL)
def test_dense_generators(dtype):
    n, batch = 5, 3
    np.testing.assert_array_equal(bl.identity(n, dtype=dtype, batch_size=batch), np.broadcast_to(np.eye(n), (batch, n, n)))
    np.testing.assert_array_equal(bl.zeros(n, 3, dtype=dtype, batch_size=batch), np.zeros((batch, n, 3)))
    np.testing.assert_array_equal(bl.ones(n, 3, dtype=dtype, batch_size=batch), np.ones((batch, n, 3)))
    assert np.asarray(bl.identity(n, dtype=dtype, batch_size=batch)).dtype == np.dtype(dtype)

    values = np.arange(1, batch * n + 1, dtype=np.float64).reshape(batch, n).astype(dtype)
    np.testing.assert_array_equal(bl.diagonal(values), np.stack([np.diag(values[i]) for i in range(batch)]))

    toeplitz = np.asarray(bl.tridiag_toeplitz(n, diagonal_value=2.0, sub_diagonal_value=-1.0,
                                              super_diagonal_value=3.0, dtype=dtype, batch_size=batch))
    want = 2.0 * np.eye(n) - np.eye(n, k=-1) + 3.0 * np.eye(n, k=1)
    np.testing.assert_array_equal(toeplitz, np.broadcast_to(want, (batch, n, n)))


@pytest.mark.parametrize("dtype", ALL)
def test_random_generator_is_seeded_and_hermitian_on_request(dtype):
    first = np.asarray(bl.random(6, dtype=dtype, hermitian=True, batch_size=3, seed=7))
    again = np.asarray(bl.random(6, dtype=dtype, hermitian=True, batch_size=3, seed=7))
    other = np.asarray(bl.random(6, dtype=dtype, hermitian=True, batch_size=3, seed=8))

    np.testing.assert_array_equal(first, again)
    assert not np.array_equal(first, other)
    np.testing.assert_allclose(first, _h(first), rtol=0, atol=0)
    if np.issubdtype(dtype, np.complexfloating):
        assert np.abs(first.imag).max() > 0
    assert not np.array_equal(first[0], first[1])


@pytest.mark.parametrize("log10_kappa", [1.0, 4.0])
def test_conditioned_generator_hits_the_requested_condition(log10_kappa):
    a = np.asarray(bl.random_with_log10_cond_metric(n=12, log10_kappa=log10_kappa, batch_size=3, seed=3))

    np.testing.assert_allclose(np.log10(np.linalg.cond(a, 2)), log10_kappa, atol=0.05)

    h = np.asarray(bl.random_hermitian_with_log10_cond_metric(n=12, log10_kappa=log10_kappa, batch_size=3,
                                                              dtype=np.complex128, seed=4))
    np.testing.assert_allclose(h, _h(h), atol=1e-12)
    np.testing.assert_allclose(np.log10(np.linalg.cond(h, 2)), log10_kappa, atol=0.05)


@pytest.mark.skipif(sp is None, reason="SciPy is required for the sparse wrappers")
def test_random_sparse_hermitian_is_hermitian():
    matrices = bl.random_sparse_hermitian(20, density=0.2, dtype=np.complex128, batch_size=2, seed=5)
    matrices = matrices if isinstance(matrices, list) else [matrices]

    assert len(matrices) == 2
    for m in matrices:
        dense = m.toarray()
        np.testing.assert_allclose(dense, dense.conj().T, atol=1e-12)
        assert np.abs(dense.imag).max() > 0
