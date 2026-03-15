#include "init.hh"
#include "support.hh"

namespace batchlas::python {

namespace {

template <typename T>
DenseMatrixT<T> make_gemm_output(const DenseMatrixT<T>& a,
                                 const DenseMatrixT<T>& b,
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
        return DenseMatrixT<T>(m, n, a.batch_size());
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

    DenseMatrixT<T> result(max_rows, max_cols, batch_size);
    UnifiedVector<int> rows_meta(static_cast<std::size_t>(batch_size));
    UnifiedVector<int> cols_meta(static_cast<std::size_t>(batch_size));
    for (int batch = 0; batch < batch_size; ++batch) {
        rows_meta[static_cast<std::size_t>(batch)] = active_rows[static_cast<std::size_t>(batch)];
        cols_meta[static_cast<std::size_t>(batch)] = active_cols[static_cast<std::size_t>(batch)];
    }
    result.set_active_dims(rows_meta.to_span(), cols_meta.to_span());
    return result;
}

template <typename T>
DenseVector dense_gemv_impl(const DenseMatrix& a_wrapper,
                            const DenseVector& x_wrapper,
                            const py::object& alpha_object,
                            const py::object& beta_object,
                            Transpose trans_a,
                            Backend backend,
                            const std::optional<std::string>& device_name) {
    ensure_same_dtype(a_wrapper, x_wrapper, "matrix and vector dtypes must match");
    const auto& a = std::get<DenseMatrixT<T>>(a_wrapper.storage);
    const auto& x = std::get<Vector<T>>(x_wrapper.storage);
    if (a.batch_size() != x.batch_size()) {
        throw py::value_error("matrix and vector batch sizes must match");
    }

    const int expected_x = trans_a == Transpose::NoTrans ? a.cols() : a.rows();
    const int y_size = trans_a == Transpose::NoTrans ? a.rows() : a.cols();
    if (x.size() != expected_x) {
        throw py::value_error("matrix/vector dimensions do not match for gemv");
    }

    Vector<T> y(y_size, a.batch_size());
    const T alpha = scalar_from_object<T>(alpha_object);
    const T beta = scalar_from_object<T>(beta_object);
    Queue queue = make_queue(device_name);
    visit_backend(backend, [&](auto backend_tag) {
        constexpr Backend B = decltype(backend_tag)::value;
        batchlas::gemv<B, T>(queue, a.view(), x, y, alpha, beta, trans_a);
    });
    queue.wait();
    return wrap_vector(std::move(y));
}

template <typename T>
DenseMatrix dense_symm_impl(const DenseMatrix& a_wrapper,
                            const DenseMatrix& b_wrapper,
                            const py::object& alpha_object,
                            const py::object& beta_object,
                            Side side,
                            Uplo uplo,
                            Backend backend,
                            const std::optional<std::string>& device_name) {
    static_assert(std::is_floating_point_v<T>, "symm only supports real dtypes");
    ensure_same_dtype(a_wrapper, b_wrapper, "matrix dtypes must match");
    const auto& a = std::get<DenseMatrixT<T>>(a_wrapper.storage);
    const auto& b = std::get<DenseMatrixT<T>>(b_wrapper.storage);
    if (a.batch_size() != b.batch_size()) {
        throw py::value_error("matrix batch sizes must match");
    }

    DenseMatrixT<T> c(b.rows(), b.cols(), b.batch_size());
    const T alpha = scalar_from_object<T>(alpha_object);
    const T beta = scalar_from_object<T>(beta_object);
    Queue queue = make_queue(device_name);
    visit_backend(backend, [&](auto backend_tag) {
        constexpr Backend B = decltype(backend_tag)::value;
        batchlas::symm<B, T>(queue, a.view(), b.view(), c.view(), alpha, beta, side, uplo);
    });
    queue.wait();
    return wrap_dense(std::move(c));
}

template <typename T>
DenseMatrix dense_syrk_impl(const DenseMatrix& a_wrapper,
                            const py::object& alpha_object,
                            const py::object& beta_object,
                            Uplo uplo,
                            Transpose trans_a,
                            Backend backend,
                            const std::optional<std::string>& device_name) {
    static_assert(std::is_floating_point_v<T>, "syrk only supports real dtypes");
    const auto& a = std::get<DenseMatrixT<T>>(a_wrapper.storage);
    const int n = trans_a == Transpose::NoTrans ? a.rows() : a.cols();
    DenseMatrixT<T> c(n, n, a.batch_size());
    const T alpha = scalar_from_object<T>(alpha_object);
    const T beta = scalar_from_object<T>(beta_object);
    Queue queue = make_queue(device_name);
    visit_backend(backend, [&](auto backend_tag) {
        constexpr Backend B = decltype(backend_tag)::value;
        batchlas::syrk<B, T>(queue, a.view(), c.view(), alpha, beta, uplo, trans_a);
    });
    queue.wait();
    return wrap_dense(std::move(c));
}

template <typename T>
DenseMatrix dense_syr2k_impl(const DenseMatrix& a_wrapper,
                             const DenseMatrix& b_wrapper,
                             const py::object& alpha_object,
                             const py::object& beta_object,
                             Uplo uplo,
                             Transpose trans_a,
                             Backend backend,
                             const std::optional<std::string>& device_name) {
    static_assert(std::is_floating_point_v<T>, "syr2k only supports real dtypes");
    ensure_same_dtype(a_wrapper, b_wrapper, "matrix dtypes must match");
    const auto& a = std::get<DenseMatrixT<T>>(a_wrapper.storage);
    const auto& b = std::get<DenseMatrixT<T>>(b_wrapper.storage);
    if (a.batch_size() != b.batch_size()) {
        throw py::value_error("matrix batch sizes must match");
    }
    const int n = trans_a == Transpose::NoTrans ? a.rows() : a.cols();
    DenseMatrixT<T> c(n, n, a.batch_size());
    const T alpha = scalar_from_object<T>(alpha_object);
    const T beta = scalar_from_object<T>(beta_object);
    Queue queue = make_queue(device_name);
    visit_backend(backend, [&](auto backend_tag) {
        constexpr Backend B = decltype(backend_tag)::value;
        batchlas::syr2k<B, T>(queue, a.view(), b.view(), c.view(), alpha, beta, uplo, trans_a);
    });
    queue.wait();
    return wrap_dense(std::move(c));
}

template <typename T>
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
    const auto& a = std::get<DenseMatrixT<T>>(a_wrapper.storage);
    const auto& b = std::get<DenseMatrixT<T>>(b_wrapper.storage);
    if (a.batch_size() != b.batch_size()) {
        throw py::value_error("matrix batch sizes must match");
    }

    DenseMatrixT<T> c(b.rows(), b.cols(), b.batch_size());
    const T alpha = scalar_from_object<T>(alpha_object);
    Queue queue = make_queue(device_name);
    visit_backend(backend, [&](auto backend_tag) {
        constexpr Backend B = decltype(backend_tag)::value;
        batchlas::trmm<B, T>(queue, a.view(), b.view(), c.view(), alpha, side, uplo, trans_a, diag);
    });
    queue.wait();
    return wrap_dense(std::move(c));
}

template <typename T>
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
    const auto& a = std::get<DenseMatrixT<T>>(a_wrapper.storage);
    DenseMatrixT<T> out = std::get<DenseMatrixT<T>>(b_wrapper.storage).clone();
    const T alpha = scalar_from_object<T>(alpha_object);
    Queue queue = make_queue(device_name);
    visit_backend(backend, [&](auto backend_tag) {
        constexpr Backend B = decltype(backend_tag)::value;
        batchlas::trsm<B, T>(queue, a.view(), out.view(), side, uplo, trans_a, diag, alpha);
    });
    queue.wait();
    return wrap_dense(std::move(out));
}

template <typename T>
DenseMatrix sparse_spmm_impl(const SparseMatrix& a_wrapper,
                             const DenseMatrix& b_wrapper,
                             const py::object& alpha_object,
                             const py::object& beta_object,
                             Transpose trans_a,
                             Transpose trans_b,
                             Backend backend,
                             const std::optional<std::string>& device_name) {
    ensure_same_dtype(a_wrapper, b_wrapper, "sparse and dense dtypes must match");
    const auto& a = std::get<SparseMatrixT<T>>(a_wrapper.storage);
    const auto& b = std::get<DenseMatrixT<T>>(b_wrapper.storage);
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

    DenseMatrixT<T> c(m, n, b.batch_size());
    const T alpha = scalar_from_object<T>(alpha_object);
    const T beta = scalar_from_object<T>(beta_object);
    Queue queue = make_queue(device_name);
    run_backend_with_workspace(
        backend, queue,
        [&](auto backend_tag) {
            constexpr Backend B = decltype(backend_tag)::value;
            return batchlas::spmm_buffer_size<B, T, MatrixFormat::CSR>(queue, a.view(), b.view(), c.view(), alpha,
                                                                       beta, trans_a, trans_b);
        },
        [&](auto backend_tag, Span<std::byte> workspace) {
            constexpr Backend B = decltype(backend_tag)::value;
            batchlas::spmm<B, T, MatrixFormat::CSR>(queue, a.view(), b.view(), c.view(), alpha, beta, trans_a,
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
                            const py::object& device_name_obj) {
        ensure_same_dtype(a, b, "matrix dtypes must match");
        const Backend backend = parse_backend(backend_name);
        const Transpose trans_a = parse_transpose(trans_a_name);
        const Transpose trans_b = parse_transpose(trans_b_name);
        const ComputePrecision precision = parse_compute_precision(precision_name);
        const auto device_name = optional_string_from_obj(device_name_obj);
        return visit_dense(a, [&](auto tag, const auto& typed_a) {
            using scalar_type = typename decltype(tag)::type;
            const auto& typed_b = std::get<DenseMatrixT<scalar_type>>(b.storage);
            DenseMatrixT<scalar_type> c = make_gemm_output(typed_a, typed_b, trans_a, trans_b);
            const scalar_type typed_alpha = scalar_from_object<scalar_type>(alpha);
            const scalar_type typed_beta = scalar_from_object<scalar_type>(beta);
            Queue queue = make_queue(device_name);
            visit_backend(backend, [&](auto backend_tag) {
                constexpr Backend B = decltype(backend_tag)::value;
                if (typed_a.is_heterogeneous() || typed_b.is_heterogeneous()) {
                    batchlas::gemm_heterogeneous<B, scalar_type>(queue, typed_a.view(), typed_b.view(), c.view(),
                                                                 typed_alpha, typed_beta, trans_a, trans_b, precision);
                } else {
                    batchlas::gemm<B, scalar_type>(queue, typed_a.view(), typed_b.view(), c.view(), typed_alpha,
                                                   typed_beta, trans_a, trans_b, precision);
                }
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
                                          const py::object& device_name_obj) {
        ensure_same_dtype(a, b, "matrix dtypes must match");
        const Backend backend = parse_backend(backend_name);
        const Transpose trans_a = parse_transpose(trans_a_name);
        const Transpose trans_b = parse_transpose(trans_b_name);
        const ComputePrecision precision = parse_compute_precision(precision_name);
        const auto device_name = optional_string_from_obj(device_name_obj);
        return visit_dense(a, [&](auto tag, const auto& typed_a) {
            using scalar_type = typename decltype(tag)::type;
            const auto& typed_b = std::get<DenseMatrixT<scalar_type>>(b.storage);
            DenseMatrixT<scalar_type> c = make_gemm_output(typed_a, typed_b, trans_a, trans_b);
            const scalar_type typed_alpha = scalar_from_object<scalar_type>(alpha);
            const scalar_type typed_beta = scalar_from_object<scalar_type>(beta);
            Queue queue = make_queue(device_name);
            visit_backend(backend, [&](auto backend_tag) {
                constexpr Backend B = decltype(backend_tag)::value;
                batchlas::gemm_heterogeneous<B, scalar_type>(queue, typed_a.view(), typed_b.view(), c.view(),
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
                            const py::object& device_name_obj) {
        const Backend backend = parse_backend(backend_name);
        const Transpose trans_a = parse_transpose(trans_a_name);
        const auto device_name = optional_string_from_obj(device_name_obj);
        return visit_dense(a, [&](auto tag, const auto&) {
            using scalar_type = typename decltype(tag)::type;
            return dense_gemv_impl<scalar_type>(a, x, alpha, beta, trans_a, backend, device_name);
        });
    });

    module.def("_symm", [](const DenseMatrix& a,
                            const DenseMatrix& b,
                            const py::object& alpha,
                            const py::object& beta,
                            const std::string& side_name,
                            const std::string& uplo_name,
                            const std::string& backend_name,
                            const py::object& device_name_obj) {
        const Backend backend = parse_backend(backend_name);
        const Side side = parse_side(side_name);
        const Uplo uplo = parse_uplo(uplo_name);
        const auto device_name = optional_string_from_obj(device_name_obj);
        return visit_dense(a, [&](auto tag, const auto&) -> DenseMatrix {
            using scalar_type = typename decltype(tag)::type;
            if constexpr (!std::is_floating_point_v<scalar_type>) {
                throw_not_implemented("symm only supports float32 and float64");
            } else {
                return dense_symm_impl<scalar_type>(a, b, alpha, beta, side, uplo, backend, device_name);
            }
        });
    });

    module.def("_syrk", [](const DenseMatrix& a,
                            const py::object& alpha,
                            const py::object& beta,
                            const std::string& uplo_name,
                            const std::string& trans_a_name,
                            const std::string& backend_name,
                            const py::object& device_name_obj) {
        const Backend backend = parse_backend(backend_name);
        const Uplo uplo = parse_uplo(uplo_name);
        const Transpose trans_a = parse_transpose(trans_a_name);
        const auto device_name = optional_string_from_obj(device_name_obj);
        return visit_dense(a, [&](auto tag, const auto&) -> DenseMatrix {
            using scalar_type = typename decltype(tag)::type;
            if constexpr (!std::is_floating_point_v<scalar_type>) {
                throw_not_implemented("syrk only supports float32 and float64");
            } else {
                return dense_syrk_impl<scalar_type>(a, alpha, beta, uplo, trans_a, backend, device_name);
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
                             const py::object& device_name_obj) {
        const Backend backend = parse_backend(backend_name);
        const Uplo uplo = parse_uplo(uplo_name);
        const Transpose trans_a = parse_transpose(trans_a_name);
        const auto device_name = optional_string_from_obj(device_name_obj);
        return visit_dense(a, [&](auto tag, const auto&) -> DenseMatrix {
            using scalar_type = typename decltype(tag)::type;
            if constexpr (!std::is_floating_point_v<scalar_type>) {
                throw_not_implemented("syr2k only supports float32 and float64");
            } else {
                return dense_syr2k_impl<scalar_type>(a, b, alpha, beta, uplo, trans_a, backend, device_name);
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
                            const py::object& device_name_obj) {
        const Backend backend = parse_backend(backend_name);
        const Transpose trans_a = parse_transpose(trans_a_name);
        const Transpose trans_b = parse_transpose(trans_b_name);
        const auto device_name = optional_string_from_obj(device_name_obj);
        return visit_sparse(a, [&](auto tag, const auto&) {
            using scalar_type = typename decltype(tag)::type;
            return sparse_spmm_impl<scalar_type>(a, b, alpha, beta, trans_a, trans_b, backend, device_name);
        });
    });
}

}  // namespace batchlas::python
