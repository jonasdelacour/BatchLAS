"""Exception classes, route pins, settings reload and the build-feature report."""

from __future__ import annotations

import os

import numpy as np
import pytest

import batchlas as bl


def test_exception_hierarchy_mirrors_error_hh():
    assert issubclass(bl.BatchLASError, RuntimeError)
    assert issubclass(bl.BatchLASError, bl.BatchLASException)
    for leaf in (bl.UnsupportedError, bl.DeviceError, bl.WorkspaceError, bl.ConvergenceError,
                 bl.InternalError, bl.ApiMisuseError):
        assert issubclass(leaf, bl.BatchLASError)
    assert issubclass(bl.NoRouteError, bl.UnsupportedError)
    assert issubclass(bl.InvalidArgumentError, ValueError)
    assert issubclass(bl.InvalidArgumentError, bl.BatchLASException)
    assert not issubclass(bl.InvalidArgumentError, RuntimeError)
    assert issubclass(bl.OutOfRangeError, IndexError)
    assert bl.NoRouteError.__module__ == "batchlas"
    for name in ("BatchLASException", "BatchLASError", "NoRouteError", "InvalidArgumentError"):
        assert name in bl.__all__


def test_invalid_argument_is_translated():
    with pytest.raises(bl.InvalidArgumentError, match="GESV: A must be square"):
        bl.gesv(np.ones((2, 3, 4)), np.ones((2, 3, 1)))


def test_unsupported_is_translated_and_still_a_runtime_error():
    a = np.eye(6, dtype=np.complex128) * (1 + 1j)
    with pytest.raises(bl.UnsupportedError) as caught:
        bl.gesvd_cta(a)
    assert isinstance(caught.value, RuntimeError)
    assert not isinstance(caught.value, bl.NoRouteError)


@pytest.mark.skipif(not bl.compiled_features()["vendor_free"], reason="hemm has a vendor route in this build")
@pytest.mark.parametrize("dtype,name", [(np.complex64, "complex64"), (np.complex128, "complex128")])
def test_no_route_error_carries_op_backend_and_dtype(dtype, name):
    a = np.eye(4, dtype=dtype)
    with pytest.raises(bl.NoRouteError) as caught:
        bl.hemm(a, a)
    assert caught.value.op == "hemm"
    assert caught.value.dtype == name
    assert caught.value.backend in {b for b in bl.available_backends() if b != "auto"}
    assert isinstance(caught.value, RuntimeError)


def test_bad_route_pin_raises_invalid_argument_and_is_restored():
    a = np.ones((4, 8, 8), dtype=np.float32)
    expected = bl.gemm(a, a)
    with bl.route("gemm", "bogus"):
        assert os.environ["BATCHLAS_GEMM_ROUTE"] == "bogus"
        assert bl.route_pin("gemm") == "bogus"
        with pytest.raises(bl.InvalidArgumentError, match="BATCHLAS_GEMM_ROUTE"):
            bl.gemm(a, a)
        with pytest.raises(ValueError):
            bl.gemm(a, a)
    assert "BATCHLAS_GEMM_ROUTE" not in os.environ
    assert bl.route_pin("gemm") is None
    np.testing.assert_array_equal(bl.gemm(a, a), expected)


def test_route_rejects_an_op_without_a_route_variable():
    with pytest.raises(bl.InvalidArgumentError, match="BATCHLAS_EXPAND_ROUTE"):
        bl.set_route("herk", "native")
    with pytest.raises(bl.InvalidArgumentError):
        bl.route_pin("not_an_op")


def test_valid_route_pin_changes_the_chosen_route(capfd):
    a = np.ones((4, 16, 16), dtype=np.float32)
    with bl.route("gemm", None):
        os.environ["BATCHLAS_SELECT_TRACE"] = "1"
        bl.reload_settings()
        try:
            capfd.readouterr()
            bl.gemm(a, a)
            auto_trace = capfd.readouterr().err
            with bl.route("gemm", "direct"):
                pinned = bl.gemm(a, a)
            direct_trace = capfd.readouterr().err
            with bl.route("gemm", "small"):
                bl.gemm(a, a)
            small_trace = capfd.readouterr().err
        finally:
            del os.environ["BATCHLAS_SELECT_TRACE"]
            bl.reload_settings()
    assert "gemm float" in auto_trace and "[pinned]" not in auto_trace
    assert "-> direct  [pinned]" in direct_trace
    assert "-> small  [pinned]" in small_trace
    np.testing.assert_allclose(pinned, a @ a)


def test_nested_route_restores_the_outer_pin():
    with bl.route("potrf", "native"):
        with bl.route("potrf", "vendor"):
            assert bl.route_pin("potrf") == "vendor"
        assert bl.route_pin("potrf") == "native"
        assert os.environ["BATCHLAS_POTRF_ROUTE"] == "native"
    assert bl.route_pin("potrf") is None


def test_reload_settings_picks_up_a_raw_environment_change():
    os.environ["BATCHLAS_SYRK_ROUTE"] = "vendor"
    try:
        assert bl.route_pin("syrk") is None or bl.route_pin("syrk") == "vendor"
        bl.reload_settings()
        assert bl.route_pin("syrk") == "vendor"
    finally:
        del os.environ["BATCHLAS_SYRK_ROUTE"]
        bl.reload_settings()
    assert bl.route_pin("syrk") is None


def test_compiled_features_reports_vendor_status_and_sycl_implementation():
    features = bl.compiled_features()
    assert features["sycl_implementation"] in {"dpcpp", "acpp"}
    assert isinstance(features["vendor_libraries"], list)
    assert features["vendor_free"] == (len(features["vendor_libraries"]) == 0)
    assert features["backends"] == bl.available_backends()


def test_every_listed_backend_is_accepted():
    backends = bl.available_backends()
    assert backends[0] == "auto"
    assert set(backends) <= {"auto", "cuda", "rocm", "mkl", "sycl", "magma", "netlib"}
    assert "magma" not in backends and "mkl" not in backends
    features = bl.compiled_features()
    assert ("cuda" in backends) == features["has_cuda_backend"]
    assert ("sycl" in backends) == features["has_sycl_backend"]
    assert ("netlib" in backends) == features["has_host_backend"]
