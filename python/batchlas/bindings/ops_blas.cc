// SPDX-License-Identifier: MIT
#include <python/batchlas/bindings/init.hh>
#include <python/batchlas/bindings/support.hh>

namespace batchlas::python {

namespace {

template <typename Scalar>
DenseMatrixT<Scalar> make_gemm_output(const DenseMatrixT<Scalar>& a,
                                      const DenseMatrixT<Scalar>& b,
                                      Transpose trans_a,
                                      Transpose trans_b) {
    if (a.batch_size() != b.batch_size()) {
        throw py::value_error("dense batch sizes must match");
    }

    if (!a.is_heterogeneous() && !b.is_heterogeneous()) {
        const int m = trans_a == Transpose::NoTrans ? a.rows() : a.cols();
        const int k_a = trans_a == Transpose::NoTrans ? a.cols() : a.rows();
        const int k_b = trans_b == Transpose::NoTrans ? b.rows() : b.cols();
        const int n = trans_b == Transpose::NoTrans ? b.cols() : b.rows();
        if (k_a != k_b) {
            throw py::value_error("inner dimensions do not match for gemm");
        }
        return DenseMatrixT<Scalar>(m, n, a.batch_size());
    }

    const int batch_size = a.batch_size();
    std::vector<int> active_rows(static_cast<std::size_t>(batch_size));
    std::vector<int> active_cols(static_cast<std::size_t>(batch_size));
    int max_rows = 0;
    int max_cols = 0;
    for (int batch = 0; batch < batch_size; ++batch) {
        const int a_rows = trans_a == Transpose::NoTrans ? a.rows(batch) : a.cols(batch);
        const int a_k = trans_a == Transpose::NoTrans ? a.cols(batch) : a.rows(batch);
        const int b_k = trans_b == Transpose::NoTrans ? b.rows(batch) : b.cols(batch);
        const int b_cols = trans_b == Transpose::NoTrans ? b.cols(batch) : b.rows(batch);
        if (a_k != b_k) {
            throw py::value_error("inner dimensions do not match for heterogeneous gemm");
        }
        active_rows[static_cast<std::size_t>(batch)] = a_rows;
        active_cols[static_cast<std::size_t>(batch)] = b_cols;
        max_rows = std::max(max_rows, a_rows);
        max_cols = std::max(max_cols, b_cols);
    }

    DenseMatrixT<Scalar> result(max_rows, max_cols, batch_size);
    UnifiedVector<int> rows_meta(static_cast<std::size_t>(batch_size));
    UnifiedVector<int> cols_meta(static_cast<std::size_t>(batch_size));
    for (int batch = 0; batch < batch_size; ++batch) {
        rows_meta[static_cast<std::size_t>(batch)] = active_rows[static_cast<std::size_t>(batch)];
        cols_meta[static_cast<std::size_t>(batch)] = active_cols[static_cast<std::size_t>(batch)];
    }
    result.set_active_dims(rows_meta.to_span(), cols_meta.to_span());
    return result;
}

// BLAS updates read C as well as writing it (C <- alpha * op(A) op(B) + beta * C).
// When the caller supplies an accumulator we start from a copy of it; otherwise we
// allocate a fresh buffer, which is only meaningful when beta == 0.
template <typename Scalar>
DenseMatrixT<Scalar> resolve_accumulator(const py::object& c_object,
                                         DenseMatrixT<Scalar>&& fallback,
                                         const Scalar& beta,
                                         const char* where) {
    if (c_object.is_none()) {
        if (beta != Scalar(0)) {
            throw py::value_error(std::string(where) +
                                  ": beta != 0 requires an existing C; pass it via out=");
        }
        return std::move(fallback);
    }
    const DenseMatrix& wrapper = py::cast<const DenseMatrix&>(c_object);
    const auto* typed = std::get_if<DenseMatrixT<Scalar>>(&wrapper.storage);
    if (typed == nullptr) {
        throw py::value_error(std::string(where) + ": out dtype must match the operand dtype");
    }
    if (fallback.is_heterogeneous() || typed->is_heterogeneous()) {
        // A cloned accumulator would carry its own per-batch active dimensions,
        // which need not agree with the ones this product implies.
        throw py::value_error(std::string(where) +
                              ": out is not supported for heterogeneous batches");
    }
    if (typed->rows() != fallback.rows() || typed->cols() != fallback.cols() ||
        typed->batch_size() != fallback.batch_size()) {
        throw py::value_error(std::string(where) + ": out has the wrong shape for this operation");
    }
    return typed->clone();
}

template <typename Scalar>
Vector<Scalar> resolve_accumulator_vector(const py::object& y_object,
                                          Vector<Scalar>&& fallback,
                                          const Scalar& beta,
                                          const char* where) {
    if (y_object.is_none()) {
        if (beta != Scalar(0)) {
            throw py::value_error(std::string(where) +
                                  ": beta != 0 requires an existing y; pass it via out=");
        }
        return std::move(fallback);
    }
    const DenseVector& wrapper = py::cast<const DenseVector&>(y_object);
    const auto* typed = std::get_if<Vector<Scalar>>(&wrapper.storage);
    if (typed == nullptr) {
        throw py::value_error(std::string(where) + ": out dtype must match the operand dtype");
    }
    if (typed->size() != fallback.size() || typed->batch_size() != fallback.batch_size()) {
        throw py::value_error(std::string(where) + ": out has the wrong shape for this operation");
    }
    Vector<Scalar> copy(typed->size(), typed->batch_size());
    const auto source = typed->data();
    for (int batch = 0; batch < typed->batch_size(); ++batch) {
        for (int index = 0; index < typed->size(); ++index) {
            copy[static_cast<std::size_t>(batch * copy.stride() + index * copy.inc())] =
                source[static_cast<std::size_t>(batch * typed->stride() + index * typed->inc())];
        }
    }
    return copy;
}

template <typename Scalar>
DenseVector dense_gemv_impl(const DenseMatrix& a_wrapper,
                            const DenseVector& x_wrapper,
                            const py::object& alpha_object,
                            const py::object& beta_object,
                            Transpose trans_a,
                            Backend backend,
                            const std::optional<std::string>& device_name,
                            const py::object& y_object) {
    ensure_same_dtype(a_wrapper, x_wrapper, "matrix and vector dtypes must match");
    const auto& a = std::get<DenseMatrixT<Scalar>>(a_wrapper.storage);
    const auto& x = std::get<Vector<Scalar>>(x_wrapper.storage);
    if (a.batch_size() != x.batch_size()) {
        throw py::value_error("matrix and vector batch sizes must match");
    }

    const int expected_x = trans_a == Transpose::NoTrans ? a.cols() : a.rows();
    const int y_size = trans_a == Transpose::NoTrans ? a.rows() : a.cols();
    if (x.size() != expected_x) {
        throw py::value_error("matrix/vector dimensions do not match for gemv");
    }

    const Scalar alpha = scalar_from_object<Scalar>(alpha_object);
    const Scalar beta = scalar_from_object<Scalar>(beta_object);
    Vector<Scalar> y = resolve_accumulator_vector<Scalar>(y_object, Vector<Scalar>(y_size, a.batch_size()), beta,
                                                          "gemv");
    Queue& queue = acquire_queue(device_name, backend);
    visit_backend<Scalar>(queue, [&](auto backend_tag) {
        constexpr Backend B = decltype(backend_tag)::value;
        batchlas::gemv<B, Scalar>(queue, a.view(), x, y, alpha, beta, trans_a);
    });
    queue.wait();
    return wrap_vector(std::move(y));
}

template <typename Real>
DenseMatrix dense_symm_impl(const DenseMatrix& a_wrapper,
                            const DenseMatrix& b_wrapper,
                            const py::object& alpha_object,
                            const py::object& beta_object,
                            Side side,
                            Uplo uplo,
                            Backend backend,
                            const std::optional<std::string>& device_name,
                            const py::object& c_object) {
    static_assert(std::is_floating_point_v<Real>, "symm only supports real dtypes");
    ensure_same_dtype(a_wrapper, b_wrapper, "matrix dtypes must match");
    const auto& a = std::get<DenseMatrixT<Real>>(a_wrapper.storage);
    const auto& b = std::get<DenseMatrixT<Real>>(b_wrapper.storage);
    if (a.batch_size() != b.batch_size()) {
        throw py::value_error("matrix batch sizes must match");
    }

    const Real alpha = scalar_from_object<Real>(alpha_object);
    const Real beta = scalar_from_object<Real>(beta_object);
    DenseMatrixT<Real> c =
        resolve_accumulator<Real>(c_object, DenseMatrixT<Real>(b.rows(), b.cols(), b.batch_size()), beta, "symm");
    Queue& queue = acquire_queue(device_name, backend);
    visit_backend<Real>(queue, [&](auto backend_tag) {
        constexpr Backend B = decltype(backend_tag)::value;
        batchlas::symm<B, Real>(queue, a.view(), b.view(), c.view(), alpha, beta, side, uplo);
    });
    queue.wait();
    return wrap_dense(std::move(c));
}

template <typename Real>
DenseMatrix dense_syrk_impl(const DenseMatrix& a_wrapper,
                            const py::object& alpha_object,
                            const py::object& beta_object,
                            Uplo uplo,
                            Transpose trans_a,
                            Backend backend,
                            const std::optional<std::string>& device_name,
                            const py::object& c_object) {
    static_assert(std::is_floating_point_v<Real>, "syrk only supports real dtypes");
    const auto& a = std::get<DenseMatrixT<Real>>(a_wrapper.storage);
    const int n = trans_a == Transpose::NoTrans ? a.rows() : a.cols();
    const Real alpha = scalar_from_object<Real>(alpha_object);
    const Real beta = scalar_from_object<Real>(beta_object);
    DenseMatrixT<Real> c = resolve_accumulator<Real>(c_object, DenseMatrixT<Real>(n, n, a.batch_size()), beta, "syrk");
    Queue& queue = acquire_queue(device_name, backend);
    visit_backend<Real>(queue, [&](auto backend_tag) {
        constexpr Backend B = decltype(backend_tag)::value;
        batchlas::syrk<B, Real>(queue, a.view(), c.view(), alpha, beta, uplo, trans_a);
    });
    queue.wait();
    return wrap_dense(std::move(c));
}

template <typename Real>
DenseMatrix dense_syr2k_impl(const DenseMatrix& a_wrapper,
                             const DenseMatrix& b_wrapper,
                             const py::object& alpha_object,
                             const py::object& beta_object,
                             Uplo uplo,
                             Transpose trans_a,
                             Backend backend,
                             const std::optional<std::string>& device_name,
                             const py::object& c_object) {
    static_assert(std::is_floating_point_v<Real>, "syr2k only supports real dtypes");
    ensure_same_dtype(a_wrapper, b_wrapper, "matrix dtypes must match");
    const auto& a = std::get<DenseMatrixT<Real>>(a_wrapper.storage);
    const auto& b = std::get<DenseMatrixT<Real>>(b_wrapper.storage);
    if (a.batch_size() != b.batch_size()) {
        throw py::value_error("matrix batch sizes must match");
    }
    const int n = trans_a == Transpose::NoTrans ? a.rows() : a.cols();
    const Real alpha = scalar_from_object<Real>(alpha_object);
    const Real beta = scalar_from_object<Real>(beta_object);
    DenseMatrixT<Real> c = resolve_accumulator<Real>(c_object, DenseMatrixT<Real>(n, n, a.batch_size()), beta, "syr2k");
    Queue& queue = acquire_queue(device_name, backend);
    visit_backend<Real>(queue, [&](auto backend_tag) {
        constexpr Backend B = decltype(backend_tag)::value;
        batchlas::syr2k<B, Real>(queue, a.view(), b.view(), c.view(), alpha, beta, uplo, trans_a);
    });
    queue.wait();
    return wrap_dense(std::move(c));
}

template <typename Scalar>
DenseMatrix dense_hemm_impl(const DenseMatrix& a_wrapper,
                            const DenseMatrix& b_wrapper,
                            const py::object& alpha_object,
                            const py::object& beta_object,
                            Side side,
                            Uplo uplo,
                            Backend backend,
                            const std::optional<std::string>& device_name,
                            const py::object& c_object) {
    ensure_same_dtype(a_wrapper, b_wrapper, "matrix dtypes must match");
    const auto& a = std::get<DenseMatrixT<Scalar>>(a_wrapper.storage);
    const auto& b = std::get<DenseMatrixT<Scalar>>(b_wrapper.storage);
    if (a.batch_size() != b.batch_size()) {
        throw py::value_error("matrix batch sizes must match");
    }
    const Scalar alpha = scalar_from_object<Scalar>(alpha_object);
    const Scalar beta = scalar_from_object<Scalar>(beta_object);
    DenseMatrixT<Scalar> c =
        resolve_accumulator<Scalar>(c_object, DenseMatrixT<Scalar>(b.rows(), b.cols(), b.batch_size()), beta, "hemm");
    Queue& queue = acquire_queue(device_name, backend);
    visit_backend<Scalar>(queue, [&](auto backend_tag) {
        constexpr Backend B = decltype(backend_tag)::value;
        batchlas::hemm<B, Scalar>(queue, a.view(), b.view(), c.view(), alpha, beta, side, uplo);
    });
    queue.wait();
    return wrap_dense(std::move(c));
}

template <typename Scalar>
DenseMatrix dense_herk_impl(const DenseMatrix& a_wrapper,
                            const py::object& alpha_object,
                            const py::object& beta_object,
                            Uplo uplo,
                            Transpose trans_a,
                            Backend backend,
                            const std::optional<std::string>& device_name,
                            const py::object& c_object) {
    using real_type = typename base_type<Scalar>::type;
    const auto& a = std::get<DenseMatrixT<Scalar>>(a_wrapper.storage);
    const int n = trans_a == Transpose::NoTrans ? a.rows() : a.cols();
    const real_type alpha = scalar_from_object<real_type>(alpha_object);
    const real_type beta = scalar_from_object<real_type>(beta_object);
    DenseMatrixT<Scalar> c = resolve_accumulator<Scalar>(c_object, DenseMatrixT<Scalar>(n, n, a.batch_size()),
                                                         Scalar(beta), "herk");
    Queue& queue = acquire_queue(device_name, backend);
    visit_backend<Scalar>(queue, [&](auto backend_tag) {
        constexpr Backend B = decltype(backend_tag)::value;
        batchlas::herk<B, Scalar>(queue, a.view(), c.view(), alpha, beta, uplo, trans_a);
    });
    queue.wait();
    return wrap_dense(std::move(c));
}

template <typename Scalar>
DenseMatrix dense_her2k_impl(const DenseMatrix& a_wrapper,
                             const DenseMatrix& b_wrapper,
                             const py::object& alpha_object,
                             const py::object& beta_object,
                             Uplo uplo,
                             Transpose trans_a,
                             Backend backend,
                             const std::optional<std::string>& device_name,
                             const py::object& c_object) {
    using real_type = typename base_type<Scalar>::type;
    ensure_same_dtype(a_wrapper, b_wrapper, "matrix dtypes must match");
    const auto& a = std::get<DenseMatrixT<Scalar>>(a_wrapper.storage);
    const auto& b = std::get<DenseMatrixT<Scalar>>(b_wrapper.storage);
    if (a.batch_size() != b.batch_size()) {
        throw py::value_error("matrix batch sizes must match");
    }
    const int n = trans_a == Transpose::NoTrans ? a.rows() : a.cols();
    const Scalar alpha = scalar_from_object<Scalar>(alpha_object);
    const real_type beta = scalar_from_object<real_type>(beta_object);
    DenseMatrixT<Scalar> c = resolve_accumulator<Scalar>(c_object, DenseMatrixT<Scalar>(n, n, a.batch_size()),
                                                         Scalar(beta), "her2k");
    Queue& queue = acquire_queue(device_name, backend);
    visit_backend<Scalar>(queue, [&](auto backend_tag) {
        constexpr Backend B = decltype(backend_tag)::value;
        batchlas::her2k<B, Scalar>(queue, a.view(), b.view(), c.view(), alpha, beta, uplo, trans_a);
    });
    queue.wait();
    return wrap_dense(std::move(c));
}

template <typename Scalar>
DenseMatrix dense_trmm_impl(const DenseMatrix& a_wrapper,
                            const DenseMatrix& b_wrapper,
                            const py::object& alpha_object,
                            Side side,
                            Uplo uplo,
                            Transpose trans_a,
                            Diag diag,
                            Backend backend,
                            const std::optional<std::string>& device_name) {
    ensure_same_dtype(a_wrapper, b_wrapper, "matrix dtypes must match");
    const auto& a = std::get<DenseMatrixT<Scalar>>(a_wrapper.storage);
    const auto& b = std::get<DenseMatrixT<Scalar>>(b_wrapper.storage);
    if (a.batch_size() != b.batch_size()) {
        throw py::value_error("matrix batch sizes must match");
    }

    DenseMatrixT<Scalar> c(b.rows(), b.cols(), b.batch_size());
    const Scalar alpha = scalar_from_object<Scalar>(alpha_object);
    Queue& queue = acquire_queue(device_name, backend);
    visit_backend<Scalar>(queue, [&](auto backend_tag) {
        constexpr Backend B = decltype(backend_tag)::value;
        batchlas::trmm<B, Scalar>(queue, a.view(), b.view(), c.view(), alpha, side, uplo, trans_a, diag);
    });
    queue.wait();
    return wrap_dense(std::move(c));
}

template <typename Scalar>
DenseMatrix dense_trsm_impl(const DenseMatrix& a_wrapper,
                            const DenseMatrix& b_wrapper,
                            const py::object& alpha_object,
                            Side side,
                            Uplo uplo,
                            Transpose trans_a,
                            Diag diag,
                            Backend backend,
                            const std::optional<std::string>& device_name) {
    ensure_same_dtype(a_wrapper, b_wrapper, "matrix dtypes must match");
    const auto& a = std::get<DenseMatrixT<Scalar>>(a_wrapper.storage);
    DenseMatrixT<Scalar> out = std::get<DenseMatrixT<Scalar>>(b_wrapper.storage).clone();
    const Scalar alpha = scalar_from_object<Scalar>(alpha_object);
    Queue& queue = acquire_queue(device_name, backend);
    visit_backend<Scalar>(queue, [&](auto backend_tag) {
        constexpr Backend B = decltype(backend_tag)::value;
        batchlas::trsm<B, Scalar>(queue, a.view(), out.view(), alpha, side, uplo, trans_a, diag);
    });
    queue.wait();
    return wrap_dense(std::move(out));
}

template <typename Scalar>
DenseMatrix sparse_spmm_impl(const SparseMatrix& a_wrapper,
                             const DenseMatrix& b_wrapper,
                             const py::object& alpha_object,
                             const py::object& beta_object,
                             Transpose trans_a,
                             Transpose trans_b,
                             Backend backend,
                             const std::optional<std::string>& device_name,
                             const py::object& c_object) {
    ensure_same_dtype(a_wrapper, b_wrapper, "sparse and dense dtypes must match");
    const auto& a = std::get<SparseMatrixT<Scalar>>(a_wrapper.storage);
    const auto& b = std::get<DenseMatrixT<Scalar>>(b_wrapper.storage);
    if (a.batch_size() != b.batch_size()) {
        throw py::value_error("sparse and dense batch sizes must match");
    }
    const int m = trans_a == Transpose::NoTrans ? a.rows() : a.cols();
    const int k_a = trans_a == Transpose::NoTrans ? a.cols() : a.rows();
    const int k_b = trans_b == Transpose::NoTrans ? b.rows() : b.cols();
    const int n = trans_b == Transpose::NoTrans ? b.cols() : b.rows();
    if (k_a != k_b) {
        throw py::value_error("inner dimensions do not match for spmm");
    }

    const Scalar alpha = scalar_from_object<Scalar>(alpha_object);
    const Scalar beta = scalar_from_object<Scalar>(beta_object);
    DenseMatrixT<Scalar> c = resolve_accumulator<Scalar>(c_object, DenseMatrixT<Scalar>(m, n, b.batch_size()), beta,
                                                         "spmm");
    Queue& queue = acquire_queue(device_name, backend);
    run_backend_with_workspace<Scalar>(
        queue,
        [&](auto backend_tag) {
            constexpr Backend B = decltype(backend_tag)::value;
            return batchlas::spmm_buffer_size<B, Scalar, MatrixFormat::CSR>(queue, a.view(), b.view(), c.view(), alpha,
                                                                            beta, trans_a, trans_b);
        },
        [&](auto backend_tag, Span<std::byte> workspace) {
            constexpr Backend B = decltype(backend_tag)::value;
            batchlas::spmm<B, Scalar, MatrixFormat::CSR>(queue, a.view(), b.view(), c.view(), alpha, beta, trans_a,
                                                         trans_b, workspace);
        });
    queue.wait();
    return wrap_dense(std::move(c));
}

}  // namespace

void init_blas_ops(py::module_& module) {
    module.def("_gemm", [](const DenseMatrix& a,
                            const DenseMatrix& b,
                            const py::object& alpha,
                            const py::object& beta,
                            const std::string& trans_a_name,
                            const std::string& trans_b_name,
                            const std::string& precision_name,
                            const std::string& backend_name,
                            const py::object& device_name_obj,
                            const py::object& out_object) {
        ensure_same_dtype(a, b, "matrix dtypes must match");
        const Backend backend = parse_backend(backend_name);
        const Transpose trans_a = parse_transpose(trans_a_name);
        const Transpose trans_b = parse_transpose(trans_b_name);
        const ComputePrecision precision = parse_compute_precision(precision_name);
        const auto device_name = optional_string_from_obj(device_name_obj);
        return visit_dense(a, [&](auto tag, const auto& typed_a) {
            using scalar_type = typename decltype(tag)::type;
            const auto& typed_b = std::get<DenseMatrixT<scalar_type>>(b.storage);
            const scalar_type typed_alpha = scalar_from_object<scalar_type>(alpha);
            const scalar_type typed_beta = scalar_from_object<scalar_type>(beta);
            DenseMatrixT<scalar_type> c = resolve_accumulator<scalar_type>(
                out_object, make_gemm_output(typed_a, typed_b, trans_a, trans_b), typed_beta, "gemm");
            Queue& queue = acquire_queue(device_name, backend);
            visit_backend<scalar_type>(queue, [&](auto backend_tag) {
                constexpr Backend B = decltype(backend_tag)::value;
                // One call for both shapes: gemm dispatches on
                // gemm_has_heterogeneous_batch internally, on every backend.
                batchlas::gemm<B, scalar_type>(queue, typed_a.view(), typed_b.view(), c.view(), typed_alpha,
                                               typed_beta, trans_a, trans_b, precision);
            });
            queue.wait();
            return wrap_dense(std::move(c));
        });
    });

    module.def("_gemm_heterogeneous", [](const DenseMatrix& a,
                                          const DenseMatrix& b,
                                          const py::object& alpha,
                                          const py::object& beta,
                                          const std::string& trans_a_name,
                                          const std::string& trans_b_name,
                                          const std::string& precision_name,
                                          const std::string& backend_name,
                                          const py::object& device_name_obj,
                            const py::object& out_object) {
        ensure_same_dtype(a, b, "matrix dtypes must match");
        const Backend backend = parse_backend(backend_name);
        const Transpose trans_a = parse_transpose(trans_a_name);
        const Transpose trans_b = parse_transpose(trans_b_name);
        const ComputePrecision precision = parse_compute_precision(precision_name);
        const auto device_name = optional_string_from_obj(device_name_obj);
        return visit_dense(a, [&](auto tag, const auto& typed_a) {
            using scalar_type = typename decltype(tag)::type;
            const auto& typed_b = std::get<DenseMatrixT<scalar_type>>(b.storage);
            const scalar_type typed_alpha = scalar_from_object<scalar_type>(alpha);
            const scalar_type typed_beta = scalar_from_object<scalar_type>(beta);
            DenseMatrixT<scalar_type> c = resolve_accumulator<scalar_type>(
                out_object, make_gemm_output(typed_a, typed_b, trans_a, trans_b), typed_beta,
                "gemm_heterogeneous");
            Queue& queue = acquire_queue(device_name, backend);
            visit_backend<scalar_type>(queue, [&](auto backend_tag) {
                constexpr Backend B = decltype(backend_tag)::value;
                batchlas::gemm<B, scalar_type>(queue, typed_a.view(), typed_b.view(), c.view(),
                                                             typed_alpha, typed_beta, trans_a, trans_b, precision);
            });
            queue.wait();
            return wrap_dense(std::move(c));
        });
    });

    module.def("_gemv", [](const DenseMatrix& a,
                            const DenseVector& x,
                            const py::object& alpha,
                            const py::object& beta,
                            const std::string& trans_a_name,
                            const std::string& backend_name,
                            const py::object& device_name_obj,
                            const py::object& out_object) {
        const Backend backend = parse_backend(backend_name);
        const Transpose trans_a = parse_transpose(trans_a_name);
        const auto device_name = optional_string_from_obj(device_name_obj);
        return visit_dense(a, [&](auto tag, const auto&) {
            using scalar_type = typename decltype(tag)::type;
            return dense_gemv_impl<scalar_type>(a, x, alpha, beta, trans_a, backend, device_name, out_object);
        });
    });

    module.def("_symm", [](const DenseMatrix& a,
                            const DenseMatrix& b,
                            const py::object& alpha,
                            const py::object& beta,
                            const std::string& side_name,
                            const std::string& uplo_name,
                            const std::string& backend_name,
                            const py::object& device_name_obj,
                            const py::object& out_object) {
        const Backend backend = parse_backend(backend_name);
        const Side side = parse_side(side_name);
        const Uplo uplo = parse_uplo(uplo_name);
        const auto device_name = optional_string_from_obj(device_name_obj);
        return visit_dense(a, [&](auto tag, const auto&) -> DenseMatrix {
            using scalar_type = typename decltype(tag)::type;
            if constexpr (!std::is_floating_point_v<scalar_type>) {
                throw_not_implemented("symm only supports float32 and float64");
            } else {
                return dense_symm_impl<scalar_type>(a, b, alpha, beta, side, uplo, backend, device_name, out_object);
            }
        });
    });

    module.def("_syrk", [](const DenseMatrix& a,
                            const py::object& alpha,
                            const py::object& beta,
                            const std::string& uplo_name,
                            const std::string& trans_a_name,
                            const std::string& backend_name,
                            const py::object& device_name_obj,
                            const py::object& out_object) {
        const Backend backend = parse_backend(backend_name);
        const Uplo uplo = parse_uplo(uplo_name);
        const Transpose trans_a = parse_transpose(trans_a_name);
        const auto device_name = optional_string_from_obj(device_name_obj);
        return visit_dense(a, [&](auto tag, const auto&) -> DenseMatrix {
            using scalar_type = typename decltype(tag)::type;
            if constexpr (!std::is_floating_point_v<scalar_type>) {
                throw_not_implemented("syrk only supports float32 and float64");
            } else {
                return dense_syrk_impl<scalar_type>(a, alpha, beta, uplo, trans_a, backend, device_name, out_object);
            }
        });
    });

    module.def("_syr2k", [](const DenseMatrix& a,
                             const DenseMatrix& b,
                             const py::object& alpha,
                             const py::object& beta,
                             const std::string& uplo_name,
                             const std::string& trans_a_name,
                             const std::string& backend_name,
                             const py::object& device_name_obj,
                            const py::object& out_object) {
        const Backend backend = parse_backend(backend_name);
        const Uplo uplo = parse_uplo(uplo_name);
        const Transpose trans_a = parse_transpose(trans_a_name);
        const auto device_name = optional_string_from_obj(device_name_obj);
        return visit_dense(a, [&](auto tag, const auto&) -> DenseMatrix {
            using scalar_type = typename decltype(tag)::type;
            if constexpr (!std::is_floating_point_v<scalar_type>) {
                throw_not_implemented("syr2k only supports float32 and float64");
            } else {
                return dense_syr2k_impl<scalar_type>(a, b, alpha, beta, uplo, trans_a, backend, device_name, out_object);
            }
        });
    });

    module.def("_hemm", [](const DenseMatrix& a,
                            const DenseMatrix& b,
                            const py::object& alpha,
                            const py::object& beta,
                            const std::string& side_name,
                            const std::string& uplo_name,
                            const std::string& backend_name,
                            const py::object& device_name_obj,
                            const py::object& out_object) {
        const Backend backend = parse_backend(backend_name);
        const Side side = parse_side(side_name);
        const Uplo uplo = parse_uplo(uplo_name);
        const auto device_name = optional_string_from_obj(device_name_obj);
        return visit_dense(a, [&](auto tag, const auto&) -> DenseMatrix {
            using scalar_type = typename decltype(tag)::type;
            if constexpr (std::is_floating_point_v<scalar_type>) {
                throw_not_implemented("hemm only supports complex64 and complex128; use symm");
            } else {
                return dense_hemm_impl<scalar_type>(a, b, alpha, beta, side, uplo, backend, device_name, out_object);
            }
        });
    });

    module.def("_herk", [](const DenseMatrix& a,
                            const py::object& alpha,
                            const py::object& beta,
                            const std::string& uplo_name,
                            const std::string& trans_a_name,
                            const std::string& backend_name,
                            const py::object& device_name_obj,
                            const py::object& out_object) {
        const Backend backend = parse_backend(backend_name);
        const Uplo uplo = parse_uplo(uplo_name);
        const Transpose trans_a = parse_transpose(trans_a_name);
        const auto device_name = optional_string_from_obj(device_name_obj);
        return visit_dense(a, [&](auto tag, const auto&) -> DenseMatrix {
            using scalar_type = typename decltype(tag)::type;
            if constexpr (std::is_floating_point_v<scalar_type>) {
                throw_not_implemented("herk only supports complex64 and complex128; use syrk");
            } else {
                return dense_herk_impl<scalar_type>(a, alpha, beta, uplo, trans_a, backend, device_name, out_object);
            }
        });
    });

    module.def("_her2k", [](const DenseMatrix& a,
                             const DenseMatrix& b,
                             const py::object& alpha,
                             const py::object& beta,
                             const std::string& uplo_name,
                             const std::string& trans_a_name,
                             const std::string& backend_name,
                             const py::object& device_name_obj,
                             const py::object& out_object) {
        const Backend backend = parse_backend(backend_name);
        const Uplo uplo = parse_uplo(uplo_name);
        const Transpose trans_a = parse_transpose(trans_a_name);
        const auto device_name = optional_string_from_obj(device_name_obj);
        return visit_dense(a, [&](auto tag, const auto&) -> DenseMatrix {
            using scalar_type = typename decltype(tag)::type;
            if constexpr (std::is_floating_point_v<scalar_type>) {
                throw_not_implemented("her2k only supports complex64 and complex128; use syr2k");
            } else {
                return dense_her2k_impl<scalar_type>(a, b, alpha, beta, uplo, trans_a, backend, device_name, out_object);
            }
        });
    });

    module.def("_trmm", [](const DenseMatrix& a,
                            const DenseMatrix& b,
                            const py::object& alpha,
                            const std::string& side_name,
                            const std::string& uplo_name,
                            const std::string& trans_a_name,
                            const std::string& diag_name,
                            const std::string& backend_name,
                            const py::object& device_name_obj) {
        const Backend backend = parse_backend(backend_name);
        const Side side = parse_side(side_name);
        const Uplo uplo = parse_uplo(uplo_name);
        const Transpose trans_a = parse_transpose(trans_a_name);
        const Diag diag = parse_diag(diag_name);
        const auto device_name = optional_string_from_obj(device_name_obj);
        return visit_dense(a, [&](auto tag, const auto&) {
            using scalar_type = typename decltype(tag)::type;
            return dense_trmm_impl<scalar_type>(a, b, alpha, side, uplo, trans_a, diag, backend, device_name);
        });
    });

    module.def("_trsm", [](const DenseMatrix& a,
                            const DenseMatrix& b,
                            const py::object& alpha,
                            const std::string& side_name,
                            const std::string& uplo_name,
                            const std::string& trans_a_name,
                            const std::string& diag_name,
                            const std::string& backend_name,
                            const py::object& device_name_obj) {
        const Backend backend = parse_backend(backend_name);
        const Side side = parse_side(side_name);
        const Uplo uplo = parse_uplo(uplo_name);
        const Transpose trans_a = parse_transpose(trans_a_name);
        const Diag diag = parse_diag(diag_name);
        const auto device_name = optional_string_from_obj(device_name_obj);
        return visit_dense(a, [&](auto tag, const auto&) {
            using scalar_type = typename decltype(tag)::type;
            return dense_trsm_impl<scalar_type>(a, b, alpha, side, uplo, trans_a, diag, backend, device_name);
        });
    });

    module.def("_spmm", [](const SparseMatrix& a,
                            const DenseMatrix& b,
                            const py::object& alpha,
                            const py::object& beta,
                            const std::string& trans_a_name,
                            const std::string& trans_b_name,
                            const std::string& backend_name,
                            const py::object& device_name_obj,
                            const py::object& out_object) {
        const Backend backend = parse_backend(backend_name);
        const Transpose trans_a = parse_transpose(trans_a_name);
        const Transpose trans_b = parse_transpose(trans_b_name);
        const auto device_name = optional_string_from_obj(device_name_obj);
        return visit_sparse(a, [&](auto tag, const auto&) {
            using scalar_type = typename decltype(tag)::type;
            return sparse_spmm_impl<scalar_type>(a, b, alpha, beta, trans_a, trans_b, backend, device_name, out_object);
        });
    });
}

}  // namespace batchlas::python
