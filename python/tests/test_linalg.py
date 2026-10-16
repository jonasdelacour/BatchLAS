"""NumPy-referenced tests for the batchlas.linalg submodule (the C++ batchlas::linalg layer).

Every case runs batch 3 with distinct items and at least one non-contiguous operand; a routine
that reads one triangle gets the other poisoned with large finite values; complex cases carry
nonzero imaginary parts; status tests break exactly one item and expect `info` nonzero there only.
"""

from __future__ import annotations

import numpy as np
import pytest

import batchlas as bl

L = bl.linalg
REAL = [np.float32, np.float64]
ALL = [np.float32, np.float64, np.complex64, np.complex128]
POISON = 1.0e6
BATCH = 3


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


def _keep(a, lower):
    return np.tril(np.ones(a.shape[-2:], dtype=bool)) if lower else np.triu(np.ones(a.shape[-2:], dtype=bool))


def _poison(a, lower):
    """Keep the `lower`/upper triangle with its diagonal; the strict other triangle becomes POISON."""
    return np.where(_keep(a, lower), a, np.asarray(POISON, dtype=a.dtype))


def _close(got, want, dtype, scale=1.0):
    tol = _tol(dtype) * scale
    want = np.asarray(want)
    np.testing.assert_allclose(np.asarray(got), want, rtol=tol, atol=tol * max(1.0, float(np.abs(want).max())))


def _hpd(rng, n, dtype, batch=BATCH):
    x = _rand(rng, (batch, n, n), dtype)
    return (x @ _h(x) + n * np.eye(n)).astype(dtype)


def _hermitian_with_spectrum(rng, spectra, dtype):
    """Items Q diag(d) Q^H with prescribed eigenvalues d (one row of `spectra` per item)."""
    spectra = np.asarray(spectra, dtype=np.float64)
    q, _ = np.linalg.qr(_rand(rng, spectra.shape + (spectra.shape[-1],), np.complex128
                              if np.issubdtype(dtype, np.complexfloating) else np.float64))
    return (q @ (spectra[..., None] * _h(q))).astype(dtype)


# --- products ----------------------------------------------------------------


@pytest.mark.parametrize("dtype", ALL)
@pytest.mark.parametrize("trans_a", ["n", "t", "c"])
@pytest.mark.parametrize("trans_b", ["n", "c"])
def test_matmul_matches_numpy(dtype, trans_a, trans_b):
    rng = np.random.default_rng(1)
    m, k, n = 5, 4, 3
    a = _rand(rng, (BATCH,) + ((m, k) if trans_a == "n" else (k, m)), dtype)
    b = _rand(rng, (BATCH,) + ((k, n) if trans_b == "n" else (n, k)), dtype)
    alpha = (1.5 - 0.5j) if np.issubdtype(dtype, np.complexfloating) else 1.5
    got = L.matmul(_strided(a), b, alpha=alpha, trans_a=trans_a, trans_b=trans_b)
    assert got.shape == (BATCH, m, n) and got.dtype == dtype
    _close(got, alpha * (_op(a, trans_a) @ _op(b, trans_b)), dtype, scale=10)


@pytest.mark.parametrize("dtype", ALL)
@pytest.mark.parametrize("trans", ["n", "t", "c"])
def test_matvec_matches_numpy(dtype, trans):
    rng = np.random.default_rng(2)
    a = _rand(rng, (BATCH, 6, 4), dtype)
    x = _rand(rng, (BATCH, 4 if trans == "n" else 6), dtype)
    got = L.matvec(_strided(a), _strided(x), trans=trans)
    _close(got, np.einsum("bij,bj->bi", _op(a, trans), x), dtype, scale=10)


@pytest.mark.parametrize("dtype", ALL)
@pytest.mark.parametrize("trans", ["c", "n"])
def test_gram_fills_both_triangles(dtype, trans):
    rng = np.random.default_rng(3)
    a = _rand(rng, (BATCH, 6, 4), dtype)
    got = L.gram(_strided(a), trans=trans)
    want = _h(a) @ a if trans == "c" else a @ _h(a)
    assert got.shape == want.shape
    _close(got, want, dtype, scale=10)


def test_gram_real_trans_is_the_inner_product():
    rng = np.random.default_rng(4)
    a = _rand(rng, (BATCH, 5, 3), np.float64)
    _close(L.gram(a, trans="t"), np.swapaxes(a, -1, -2) @ a, np.float64)


# --- triangular solve ----------------------------------------------------------


@pytest.mark.parametrize("dtype", ALL)
@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("lower", [True, False])
@pytest.mark.parametrize("trans", ["n", "t", "c"])
@pytest.mark.parametrize("unit", [False, True])
def test_solve_triangular_every_combination(dtype, side, lower, trans, unit):
    rng = np.random.default_rng(5)
    m, n = 5, 3
    order = m if side == "left" else n
    tri = _rand(rng, (BATCH, order, order), dtype) + 4 * np.eye(order, dtype=dtype)
    tri = np.where(_keep(tri, lower), tri, 0).astype(dtype)
    if unit:
        tri[:, np.arange(order), np.arange(order)] = 1
    stored = _poison(tri, lower)
    if unit:  # the diagonal must not be read either
        stored[:, np.arange(order), np.arange(order)] = 7.0
    b = _rand(rng, (BATCH, m, n), dtype)
    alpha = (0.5 + 0.25j) if np.issubdtype(dtype, np.complexfloating) else 0.5
    x = L.solve_triangular(_strided(stored), b, lower=lower, trans=trans, unit_diagonal=unit, side=side,
                           alpha=alpha)
    op = _op(tri, trans)
    lhs = op @ x if side == "left" else x @ op
    _close(lhs, alpha * b, dtype, scale=100)


def test_solve_triangular_accepts_scipy_trans_codes_and_vector_rhs():
    rng = np.random.default_rng(6)
    tri = np.tril(_rand(rng, (4, 4), np.float64)) + 4 * np.eye(4)
    b = _rand(rng, (4,), np.float64)
    x = L.solve_triangular(tri, b, trans=1)
    assert x.shape == (4,)
    _close(tri.T @ x, b, np.float64)


# --- factorisations and solves ---------------------------------------------------


@pytest.mark.parametrize("dtype", ALL)
@pytest.mark.parametrize("lower", [True, False])
def test_cholesky_factor_and_per_item_info(dtype, lower):
    rng = np.random.default_rng(7)
    a = _hpd(rng, 5, dtype)
    a[1] = -a[1]  # not positive definite: fails at the first leading minor
    r = L.cholesky(_strided(_poison(a, lower)), lower=lower)
    assert r.info.dtype == np.int32
    assert list(r.info) == [0, 1, 0]
    for b in (0, 2):
        f = r.factor[b]
        assert np.all(f[~_keep(f, lower)] == 0)
        _close(f @ _h(f) if lower else _h(f) @ f, a[b], dtype, scale=10)


@pytest.mark.parametrize("dtype", ALL)
@pytest.mark.parametrize("trans", ["n", "t", "c"])
def test_solve_and_singular_item_info(dtype, trans):
    rng = np.random.default_rng(8)
    a = _rand(rng, (BATCH, 5, 5), dtype) + 3 * np.eye(5, dtype=dtype)
    a[2, :, 0] = 0  # exactly singular: U[0, 0] == 0
    b = _rand(rng, (BATCH, 5, 2), dtype)
    r = L.solve(_strided(a), b, trans=trans)
    assert list(r.info) == [0, 0, 1]
    _close(_op(a[:2], trans) @ r.x[:2], b[:2], dtype, scale=100)


def test_solve_vector_rhs_and_scalar_info():
    rng = np.random.default_rng(9)
    a = _rand(rng, (4, 4), np.float64) + 3 * np.eye(4)
    b = _rand(rng, (4,), np.float64)
    r = L.solve(a, b)
    assert r.x.shape == (4,) and np.ndim(r.info) == 0 and r.info == 0
    _close(r.x, np.linalg.solve(a, b), np.float64)


@pytest.mark.parametrize("dtype", ALL)
@pytest.mark.parametrize("lower", [True, False])
def test_solve_spd_and_non_spd_item_info(dtype, lower):
    rng = np.random.default_rng(10)
    a = _hpd(rng, 5, dtype)
    a[0] = -a[0]
    b = _rand(rng, (BATCH, 5, 3), dtype)
    r = L.solve_spd(_strided(_poison(a, lower)), b, lower=lower)
    assert r.info[0] > 0 and list(r.info[1:]) == [0, 0]
    _close(a[1:] @ r.x[1:], b[1:], dtype, scale=100)


@pytest.mark.parametrize("dtype", ALL)
def test_inv_and_singular_item_info(dtype):
    rng = np.random.default_rng(11)
    a = _rand(rng, (BATCH, 5, 5), dtype) + 3 * np.eye(5, dtype=dtype)
    a[1, :, 0] = 0
    r = L.inv(_strided(a))
    assert list(r.info) == [0, 1, 0]
    for b in (0, 2):
        _close(r.inverse[b] @ a[b], np.eye(5), dtype, scale=100)


def _permutation_from_pivots(pivots):
    n = pivots.shape[-1]
    perm = np.arange(n)
    for i, p in enumerate(pivots - 1):
        perm[[i, p]] = perm[[p, i]]
    return perm


@pytest.mark.parametrize("dtype", ALL)
def test_lu_reconstructs_and_pivots_match_getrf(dtype):
    rng = np.random.default_rng(12)
    a = _rand(rng, (BATCH, 5, 5), dtype)
    a[0, :, 0] = 0
    r = L.lu(_strided(a))
    assert r.pivots.dtype == np.int64 and r.pivots.shape == (BATCH, 5)
    assert list(r.info) == [1, 0, 0]
    _, getrf_pivots = bl.getrf(a)
    np.testing.assert_array_equal(r.pivots, getrf_pivots)
    for b in (1, 2):
        low = np.tril(r.lu[b], -1) + np.eye(5)
        up = np.triu(r.lu[b])
        _close((low @ up), a[b][_permutation_from_pivots(r.pivots[b])], dtype, scale=10)


# --- eigenvalues -------------------------------------------------------------------


@pytest.mark.parametrize("dtype", ALL)
@pytest.mark.parametrize("lower", [True, False])
def test_eigh_and_eigvalsh_read_one_triangle(dtype, lower):
    rng = np.random.default_rng(13)
    x = _rand(rng, (BATCH, 6, 6), dtype)
    a = ((x + _h(x)) / 2).astype(dtype)
    stored = _strided(_poison(a, lower))
    want = np.linalg.eigvalsh(a.astype(np.complex128))
    r = L.eigh(stored, lower=lower)
    assert list(r.info) == [0, 0, 0]
    _close(r.eigenvalues, want, dtype, scale=10)
    _close(a @ r.eigenvectors, r.eigenvectors * r.eigenvalues[:, None, :], dtype, scale=100)
    _close(L.eigvalsh(stored, lower=lower), want, dtype, scale=10)


@pytest.mark.parametrize("dtype", ALL)
def test_eigh_subset_by_index(dtype):
    rng = np.random.default_rng(14)
    spectra = np.array([[1, 2, 3, 4, 5, 6], [-3, -1, 0.5, 2, 7, 9], [0.1, 0.2, 0.4, 0.8, 1.6, 3.2]])
    a = _hermitian_with_spectrum(rng, spectra, dtype)
    r = L.eigh(_strided(_poison(a, True)), subset_by_index=(1, 3))
    assert list(r.count) == [3, 3, 3] and list(r.info) == [0, 0, 0]
    assert r.eigenvectors.shape == (BATCH, 6, 3)
    _close(r.eigenvalues, spectra[:, 1:4], dtype, scale=10)
    _close(a @ r.eigenvectors, r.eigenvectors * r.eigenvalues[:, None, :], dtype, scale=100)
    v = L.eigvalsh(a, subset_by_index=(1, 3))
    _close(v.eigenvalues, spectra[:, 1:4], dtype, scale=10)
    assert list(v.count) == [3, 3, 3]


@pytest.mark.parametrize("dtype", ALL)
def test_eigh_subset_by_value_counts_differ_per_item(dtype):
    rng = np.random.default_rng(15)
    spectra = np.array([[1, 2, 3, 4, 5, 6], [-3, -1, 0.5, 2, 7, 9], [0.1, 0.2, 0.4, 0.8, 1.6, 3.2]])
    a = _hermitian_with_spectrum(rng, spectra, dtype)
    vl, vu = 0.45, 4.5  # (vl, vu]: 4, 2 and 3 eigenvalues
    r = L.eigh(_poison(a, True), subset_by_value=(vl, vu))
    expected = [s[(s > vl) & (s <= vu)] for s in spectra]
    assert list(r.count) == [len(e) for e in expected] == [4, 2, 3]
    assert r.eigenvalues.shape == (BATCH, 6)
    for b, e in enumerate(expected):
        c = len(e)
        _close(r.eigenvalues[b, :c], e, dtype, scale=10)
        assert np.all(np.isnan(r.eigenvalues[b, c:]))
        assert np.all(r.eigenvectors[b, :, c:] == 0)
        vec = r.eigenvectors[b, :, :c]
        _close(a[b] @ vec, vec * e[None, :], dtype, scale=100)
    v = L.eigvalsh(a, subset_by_value=(vl, vu))
    assert list(v.count) == [4, 2, 3]


def test_eigh_subset_rejects_upper_and_both_ranges():
    a = np.eye(4)
    with pytest.raises(ValueError):
        L.eigh(a, lower=False, subset_by_index=(0, 1))
    with pytest.raises(ValueError):
        L.eigh(a, subset_by_index=(0, 1), subset_by_value=(0, 1))


# --- SVD ------------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", ALL)
@pytest.mark.parametrize("shape", [(6, 4), (4, 6)])
@pytest.mark.parametrize("full", [True, False])
def test_svd_full_and_thin(dtype, shape, full):
    rng = np.random.default_rng(16)
    m, n = shape
    k = min(m, n)
    a = _rand(rng, (BATCH,) + shape, dtype)
    r = L.svd(_strided(a), full_matrices=full)
    assert r.U.shape == (BATCH, m, m if full else k) and r.Vh.shape == (BATCH, n if full else k, n)
    assert list(r.info) == [0, 0, 0]
    _close(r.S, np.linalg.svd(a, compute_uv=False), dtype, scale=10)
    _close((r.U[..., :k] * r.S[:, None, :]) @ r.Vh[:, :k, :], a, dtype, scale=100)
    if full:
        _close(_h(r.U) @ r.U, np.broadcast_to(np.eye(m), (BATCH, m, m)), dtype, scale=100)


@pytest.mark.parametrize("dtype", ALL)
def test_svd_values_only_and_independent_jobs(dtype):
    rng = np.random.default_rng(17)
    a = _rand(rng, (BATCH, 6, 4), dtype)
    want = np.linalg.svd(a, compute_uv=False)
    r = L.svd(a, compute_uv=False)
    assert r.U is None and r.Vh is None and list(r.info) == [0, 0, 0]
    _close(r.S, want, dtype, scale=10)
    r = L.svd(a, jobu="thin", jobvh="none")
    assert r.U.shape == (BATCH, 6, 4) and r.Vh is None
    _close(np.abs(_h(r.U) @ a), np.abs(r.S[:, :, None] * np.linalg.svd(a)[2][:, :4, :]), dtype, scale=100)


@pytest.mark.parametrize("dtype", ALL)
@pytest.mark.parametrize("uplo", ["lower", "upper"])
def test_svd_hermitian_uplo_reads_one_triangle(dtype, uplo):
    rng = np.random.default_rng(18)
    spectra = np.array([[-4, -1, 0.5, 2, 3], [1, 2, 3, 4, 5], [-6, -5, 1, 1.5, 8]])
    a = _hermitian_with_spectrum(rng, spectra, dtype)
    r = L.svd(_poison(a, uplo == "lower"), hermitian_uplo=uplo)
    _close(r.S, np.sort(np.abs(spectra), axis=-1)[:, ::-1], dtype, scale=10)
    _close((r.U * r.S[:, None, :]) @ r.Vh, a, dtype, scale=100)


# --- masks, transpose, norms --------------------------------------------------------


@pytest.mark.parametrize("dtype", ALL)
@pytest.mark.parametrize("k", [-1, 0, 2])
def test_triu_tril_match_numpy(dtype, k):
    rng = np.random.default_rng(19)
    a = _rand(rng, (BATCH, 4, 6), dtype)
    np.testing.assert_array_equal(L.triu(_strided(a), k), np.triu(a, k))
    np.testing.assert_array_equal(L.tril(_strided(a), k), np.tril(a, k))


@pytest.mark.parametrize("dtype", REAL)
def test_transpose_real(dtype):
    rng = np.random.default_rng(20)
    a = _rand(rng, (BATCH, 4, 6), dtype)
    np.testing.assert_array_equal(L.transpose(_strided(a)), np.swapaxes(a, -1, -2))


def test_transpose_and_cond_reject_complex():
    a = np.eye(3, dtype=np.complex128)
    with pytest.raises(NotImplementedError):
        L.transpose(a)
    with pytest.raises(NotImplementedError):
        L.cond(a)


@pytest.mark.parametrize("dtype", ALL)
@pytest.mark.parametrize("ord", ["fro", 1, np.inf, "max"])
def test_norm_matches_numpy(dtype, ord):
    rng = np.random.default_rng(21)
    a = _rand(rng, (BATCH, 4, 6), dtype)
    got = L.norm(_strided(a), ord)
    want = np.abs(a).max(axis=(-2, -1)) if ord == "max" else np.linalg.norm(a, ord=ord, axis=(-2, -1))
    assert got.shape == (BATCH,)
    _close(got, want, dtype, scale=10)


@pytest.mark.parametrize("dtype", REAL)
@pytest.mark.parametrize("ord", ["fro", 1, np.inf])
def test_cond_matches_numpy(dtype, ord):
    rng = np.random.default_rng(22)
    a = _rand(rng, (BATCH, 5, 5), dtype) + 3 * np.eye(5, dtype=dtype)
    _close(L.cond(_strided(a), ord), np.linalg.cond(a.astype(np.float64), ord), dtype, scale=100)


@pytest.mark.parametrize("dtype", REAL)
def test_cond_2_is_the_symmetric_eigenvalue_ratio(dtype):
    rng = np.random.default_rng(23)
    spectra = np.array([[1, 2, 3, 4], [-8, -1, 0.5, 2], [0.1, 1, 10, 100]])
    a = _hermitian_with_spectrum(rng, spectra, dtype)
    want = np.abs(spectra).max(-1) / np.abs(spectra).min(-1)
    np.testing.assert_allclose(L.cond(a, 2), want, rtol=100 * _tol(dtype))


# --- shapes and dtypes ------------------------------------------------------------------


def test_batch_shape_follows_the_input():
    rng = np.random.default_rng(24)
    a = _hpd(rng, 4, np.float64, batch=6).reshape(2, 3, 4, 4)
    r = L.eigh(a)
    assert r.eigenvalues.shape == (2, 3, 4) and r.eigenvectors.shape == (2, 3, 4, 4) and r.info.shape == (2, 3)
    _close(r.eigenvalues, np.linalg.eigvalsh(a), np.float64)
    single = L.cholesky(a[0, 0])
    assert single.factor.shape == (4, 4) and np.ndim(single.info) == 0 and single.info == 0
    one = L.inv(a[:1, 0])
    assert one.inverse.shape == (1, 4, 4) and one.info.shape == (1,)
    assert L.norm(a).shape == (2, 3) and np.ndim(L.norm(a[0, 0])) == 0


def test_dtype_promotion():
    a = np.array([[2, 1], [1, 3]])
    assert L.matmul(a, a).dtype == np.float64
    assert L.matmul(a.astype(np.float32), a.astype(np.float64)).dtype == np.float64
    _close(L.solve(a, np.array([1, 2])).x, np.linalg.solve(a, [1, 2]), np.float64)


def test_elementwise_family_is_reexported():
    a = np.arange(6.0).reshape(2, 3)
    np.testing.assert_array_equal(L.add(a, a), 2 * a)
    np.testing.assert_array_equal(L.scale(a, 3.0), 3 * a)
    assert L.multiply is bl.multiply and "linalg" in bl.__all__
