# SPDX-License-Identifier: MIT
"""gesv, posv, the Hermitian level-3 ops, per-item info, thin SVD and the newer option fields."""

from __future__ import annotations

import numpy as np
import pytest

import batchlas as bl

COMPLEX_TYPES = [(np.complex64, 2e-4), (np.complex128, 1e-11)]
ALL_TYPES = [(np.float32, 2e-4), (np.float64, 1e-11)] + COMPLEX_TYPES


def _random(rng, shape, dtype):
    values = rng.standard_normal(shape)
    if np.issubdtype(dtype, np.complexfloating):
        values = values + 1j * rng.standard_normal(shape)
    return values.astype(dtype)


def _ct(x):
    return np.conj(np.swapaxes(x, -1, -2))


def _hpd(rng, batch, n, dtype):
    g = _random(rng, (batch, n, n), dtype)
    return (g @ _ct(g) + n * np.eye(n)).astype(dtype)


def _rel(x, ref):
    return np.linalg.norm(x - ref) / np.linalg.norm(ref)


def _poison(a, keep):
    """Large finite values in the triangle the op must not read; the diagonal is kept."""
    out = a.copy()
    n = a.shape[-1]
    mask = np.triu(np.ones((n, n), bool), 1) if keep == "lower" else np.tril(np.ones((n, n), bool), -1)
    out[..., mask] = 1e6 * (1 + 1j) if np.iscomplexobj(a) else 1e6
    return out


@pytest.mark.parametrize("dtype,tol", ALL_TYPES)
def test_gesv_solves_a_batch_and_returns_reusable_factors(dtype, tol):
    rng = np.random.default_rng(10)
    batch, n, nrhs = 5, 9, 3
    a = (_random(rng, (batch, n, n), dtype) + 4 * np.eye(n)).astype(dtype)
    b = _random(rng, (batch, n, nrhs), dtype)

    x = bl.gesv(a, b)
    x2, lu, pivots = bl.gesv(a, b, return_factors=True)

    assert x.shape == (batch, n, nrhs) and x.dtype == dtype
    assert _rel(x, np.linalg.solve(a, b)) < tol
    np.testing.assert_array_equal(x, x2)
    assert pivots.shape == (batch, n)
    assert _rel(bl.getrs(lu, b, pivots), np.linalg.solve(a, b)) < tol


@pytest.mark.parametrize("dtype,tol", COMPLEX_TYPES)
def test_gesv_info_flags_exactly_the_singular_item(dtype, tol):
    rng = np.random.default_rng(11)
    batch, n = 4, 7
    a = (_random(rng, (batch, n, n), dtype) + 4 * np.eye(n)).astype(dtype)
    a[2, :, 3] = 0
    b = _random(rng, (batch, n, 2), dtype)

    x, info = bl.gesv(a, b, return_info=True)

    assert info.dtype == np.int32 and info.shape == (batch,)
    assert info[2] > 0
    np.testing.assert_array_equal(np.delete(info, 2), 0)
    good = [0, 1, 3]
    assert _rel(x[good], np.linalg.solve(a[good], b[good])) < tol
    x3, lu, pivots, info3 = bl.gesv(a, b, return_factors=True, return_info=True)
    np.testing.assert_array_equal(info3, info)


@pytest.mark.parametrize("uplo", ["lower", "upper"])
@pytest.mark.parametrize("dtype,tol", ALL_TYPES)
def test_posv_reads_only_the_uplo_triangle(dtype, tol, uplo):
    rng = np.random.default_rng(12)
    batch, n, nrhs = 6, 10, 4
    a = _hpd(rng, batch, n, dtype)
    b = _random(rng, (batch, n, nrhs), dtype)

    x, factor = bl.posv(_poison(a, uplo), b, uplo=uplo, return_factors=True)

    assert _rel(x, np.linalg.solve(a, b)) < tol
    tri = np.tril if uplo == "lower" else np.triu
    f = tri(factor)
    rebuilt = f @ _ct(f) if uplo == "lower" else _ct(f) @ f
    assert _rel(rebuilt, a) < tol


@pytest.mark.parametrize("dtype,tol", COMPLEX_TYPES)
def test_posv_and_potrf_info_flag_exactly_the_indefinite_item(dtype, tol):
    rng = np.random.default_rng(13)
    batch, n = 5, 8
    a = _hpd(rng, batch, n, dtype)
    a[3, 4, 4] = -50
    b = _random(rng, (batch, n, 2), dtype)

    x, info = bl.posv(a, b, return_info=True)
    factor, potrf_info = bl.potrf(a, return_info=True)

    for status in (info, potrf_info):
        assert status.dtype == np.int32 and status.shape == (batch,)
        assert status[3] > 0
        np.testing.assert_array_equal(np.delete(status, 3), 0)
    good = [0, 1, 2, 4]
    assert _rel(x[good], np.linalg.solve(a[good], b[good])) < tol
    np.testing.assert_array_equal(bl.potrf(a), factor)


@pytest.mark.parametrize("dtype,tol", COMPLEX_TYPES)
def test_getri_info_flags_exactly_the_singular_item(dtype, tol):
    rng = np.random.default_rng(14)
    batch, n = 4, 6
    a = (_random(rng, (batch, n, n), dtype) + 4 * np.eye(n)).astype(dtype)
    a[1, :, 2] = 0
    lu, pivots = bl.getrf(a)

    inverse, info = bl.getri(lu, pivots, return_info=True)

    assert info.dtype == np.int32 and info[1] > 0
    np.testing.assert_array_equal(np.delete(info, 1), 0)
    good = [0, 2, 3]
    assert _rel(inverse[good], np.linalg.inv(a[good])) < tol
    np.testing.assert_array_equal(bl.getri(lu, pivots)[good], inverse[good])


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("uplo", ["lower", "upper"])
@pytest.mark.parametrize("dtype,tol", COMPLEX_TYPES)
def test_hemm_matches_numpy_with_beta_and_a_poisoned_triangle(dtype, tol, uplo, side):
    rng = np.random.default_rng(15)
    batch, m, n = 3, 7, 5
    k = m if side == "left" else n
    h = _random(rng, (batch, k, k), dtype)
    h = (h + _ct(h)).astype(dtype)
    stored = _poison(h, uplo)
    stored[..., np.arange(k), np.arange(k)] += 1e3j
    b = _random(rng, (batch, m, n), dtype)
    c = _random(rng, (batch, m, n), dtype)
    alpha, beta = 0.5 - 1.25j, 0.75 + 0.5j

    result = bl.hemm(stored, b, alpha=alpha, beta=beta, side=side, uplo=uplo, out=c.copy())

    product = h @ b if side == "left" else b @ h
    assert _rel(result, alpha * product + beta * c) < tol


@pytest.mark.parametrize("trans", ["n", "c"])
@pytest.mark.parametrize("uplo", ["lower", "upper"])
@pytest.mark.parametrize("dtype,tol", COMPLEX_TYPES)
def test_herk_matches_numpy_in_the_written_triangle(dtype, tol, uplo, trans):
    rng = np.random.default_rng(16)
    batch, n, k = 4, 6, 9
    a = _random(rng, (batch, n, k) if trans == "n" else (batch, k, n), dtype)
    c = _random(rng, (batch, n, n), dtype)
    c = (c + _ct(c)).astype(dtype)
    alpha, beta = 1.5, -0.5

    result = bl.herk(a, alpha=alpha, beta=beta, uplo=uplo, trans_a=trans, out=c.copy())

    op = a if trans == "n" else _ct(a)
    expected = alpha * op @ _ct(op) + beta * c
    tri = np.tril if uplo == "lower" else np.triu
    other = np.triu if uplo == "lower" else np.tril
    assert _rel(tri(result), tri(expected)) < tol
    np.testing.assert_array_equal(other(result, 1 if uplo == "lower" else -1), other(c, 1 if uplo == "lower" else -1))


@pytest.mark.parametrize("trans", ["n", "c"])
@pytest.mark.parametrize("uplo", ["lower", "upper"])
@pytest.mark.parametrize("dtype,tol", COMPLEX_TYPES)
def test_her2k_matches_numpy_with_complex_alpha(dtype, tol, uplo, trans):
    rng = np.random.default_rng(17)
    batch, n, k = 3, 5, 8
    shape = (batch, n, k) if trans == "n" else (batch, k, n)
    a = _random(rng, shape, dtype)
    b = _random(rng, shape, dtype)
    c = _random(rng, (batch, n, n), dtype)
    c = (c + _ct(c)).astype(dtype)
    alpha, beta = 0.25 + 1.5j, 2.0

    result = bl.her2k(a, b, alpha=alpha, beta=beta, uplo=uplo, trans_a=trans, out=c.copy())

    op_a = a if trans == "n" else _ct(a)
    op_b = b if trans == "n" else _ct(b)
    expected = alpha * op_a @ _ct(op_b) + np.conj(alpha) * op_b @ _ct(op_a) + beta * c
    tri = np.tril if uplo == "lower" else np.triu
    assert _rel(tri(result), tri(expected)) < tol


@pytest.mark.parametrize("op", ["hemm", "herk", "her2k"])
def test_hermitian_level3_rejects_real_input(op):
    a = np.eye(4)
    with pytest.raises(NotImplementedError):
        if op == "herk":
            bl.herk(a)
        else:
            getattr(bl, op)(a, a)


@pytest.mark.parametrize("shape", [(3, 9, 5), (3, 5, 9)])
@pytest.mark.parametrize("dtype,tol", [(np.float64, 1e-11), (np.complex128, 1e-11)])
def test_gesvd_thin_returns_economy_factors(dtype, tol, shape):
    rng = np.random.default_rng(18)
    a = _random(rng, shape, dtype)
    batch, m, n = shape
    k = min(m, n)

    u, s, vh = bl.gesvd(a, compute_vectors="thin")
    u_all, s_all, vh_all = bl.gesvd(a, compute_vectors="all")

    assert u.shape == (batch, m, k) and s.shape == (batch, k) and vh.shape == (batch, k, n)
    assert u_all.shape == (batch, m, m) and vh_all.shape == (batch, n, n)
    np.testing.assert_allclose(s, np.linalg.svd(a, compute_uv=False), rtol=tol, atol=tol)
    assert _rel((u * s[:, None, :]) @ vh, a) < tol
    assert _rel(_ct(u) @ u, np.broadcast_to(np.eye(k), (batch, k, k))) < tol
    np.testing.assert_allclose(s_all, s, rtol=tol, atol=tol)


def test_gesvd_variants_accept_the_vector_policy_words():
    rng = np.random.default_rng(19)
    a = rng.standard_normal((2, 6, 6))
    for fn in (bl.gesvd, bl.gesvd_blocked, bl.gesvd_cta):
        u, s, vh = fn(a, compute_vectors="thin")
        assert u.shape == (2, 6, 6) and vh.shape == (2, 6, 6)
        assert _rel((u * s[:, None, :]) @ vh, a) < 1e-10
        assert fn(a, compute_vectors="none").shape == (2, 6)
    with pytest.raises(ValueError):
        bl.gesvd(a, compute_vectors="economy")


def _symmetric(rng, batch, n):
    g = rng.standard_normal((batch, n, n))
    return g + np.swapaxes(g, -1, -2)


def test_syevx_method_reaches_the_solver():
    rng = np.random.default_rng(20)
    a = _symmetric(rng, 2, 40)
    ref = np.linalg.eigvalsh(a)[:, -3:]

    def err(**fields):
        options = bl.SyevxOptions(iterations=1, **fields)
        values = bl.syevx(a, 3, compute_vectors=False, options=options)
        return np.abs(np.sort(values, axis=-1) - ref).max()

    assert err(method="direct") < 1e-10
    assert err(method="direct_subset") < 1e-10
    assert err(method="lobpcg") > 1e-6
    with pytest.raises(ValueError):
        err(method="qr")


def test_syevx_preconditioner_type_reaches_the_solver():
    rng = np.random.default_rng(21)
    a = _symmetric(rng, 2, 32) + 80 * np.eye(32)
    ref = np.linalg.eigvalsh(a)[:, :3]
    options = bl.SyevxOptions(method="lobpcg", preconditioner_type="jacobi", find_largest=False)

    values = bl.syevx(a, 3, compute_vectors=False, options=options)

    assert np.abs(np.sort(values, axis=-1) - ref).max() < 1e-8
    with pytest.raises(bl.InvalidArgumentError, match="smallest"):
        bl.syevx(a, 3, compute_vectors=False,
                 options=bl.SyevxOptions(method="lobpcg", preconditioner_type="jacobi", find_largest=True))
    with pytest.raises(ValueError):
        bl.syevx(a, 3, compute_vectors=False, options=bl.SyevxOptions(preconditioner_type="ilu"))


@pytest.mark.parametrize(
    "fields",
    [
        {"method": "lobpcg", "init_power_iterations": (0, 12)},
        {"method": "filtered", "filter_degree": (2, 24)},
    ],
)
def test_syevx_iteration_knobs_change_the_iterate(fields):
    rng = np.random.default_rng(22)
    a = _symmetric(rng, 2, 48)
    (knob, (low, high)), = [(k, v) for k, v in fields.items() if k != "method"]

    def run(value):
        options = bl.SyevxOptions(method=fields["method"], iterations=2, **{knob: value})
        return bl.syevx(a, 4, compute_vectors=False, options=options)

    np.testing.assert_array_equal(run(low), run(low))
    assert not np.array_equal(run(low), run(high))


@pytest.mark.parametrize("algorithm", ["auto", "levels", "recursive"])
def test_stedc_algorithm_option(algorithm):
    rng = np.random.default_rng(23)
    batch, n = 3, 90
    d = rng.standard_normal((batch, n))
    e = rng.standard_normal((batch, n - 1))
    t = np.stack([np.diag(d[b]) + np.diag(e[b], 1) + np.diag(e[b], -1) for b in range(batch)])

    values = bl.stedc(d, e, compute_vectors=False, options=bl.StedcOptions(algorithm=algorithm, recursion_threshold=16))

    np.testing.assert_allclose(np.sort(values, axis=-1), np.linalg.eigvalsh(t), atol=1e-11)


def test_stedc_rejects_an_unknown_algorithm():
    with pytest.raises(ValueError):
        bl.stedc(np.ones((1, 8)), np.ones((1, 7)), options={"algorithm": "bfs"})
