#include "init.hh"
#include "support.hh"

namespace batchlas::python {

namespace {

template <typename T, MatrixFormat MF>
DenseMatrix transpose_dense_like(const Matrix<T, MF>& input) {
    Queue queue = Queue(Device::default_device());
    auto out = batchlas::transpose(queue, input.view());
    queue.wait();
    if constexpr (MF == MatrixFormat::Dense) {
        return wrap_dense(std::move(out));
    } else {
        throw py::value_error("transpose_dense_like called with sparse matrix");
    }
}

template <typename T>
DenseMatrix dense_lascl_impl(const DenseMatrix& matrix_wrapper, T cfrom, T cto) {
    DenseMatrixT<T> out = std::get<DenseMatrixT<T>>(matrix_wrapper.storage).clone();
    Queue queue = Queue(Device::default_device());
    batchlas::lascl(queue, out.view(), cfrom, cto);
    queue.wait();
    return wrap_dense(std::move(out));
}

template <typename T>
SparseMatrix sparse_lascl_impl(const SparseMatrix& matrix_wrapper, T cfrom, T cto) {
    SparseMatrixT<T> out = std::get<SparseMatrixT<T>>(matrix_wrapper.storage).clone();
    Queue queue = Queue(Device::default_device());
    batchlas::lascl(queue, out.view(), cfrom, cto);
    queue.wait();
    return wrap_sparse(std::move(out));
}

template <typename T>
DenseVector dense_norm_impl(const DenseMatrix& matrix_wrapper, NormType norm_type) {
    const auto& matrix = std::get<DenseMatrixT<T>>(matrix_wrapper.storage);
    Queue queue = Queue(Device::default_device());
    auto values = batchlas::norm<T, MatrixFormat::Dense>(queue, matrix.view(), norm_type);
    queue.wait();
    return wrap_vector_copy(values, matrix.batch_size(), 1);
}

template <typename T>
DenseVector sparse_norm_impl(const SparseMatrix& matrix_wrapper, NormType norm_type) {
    const auto& matrix = std::get<SparseMatrixT<T>>(matrix_wrapper.storage);
    Queue queue = Queue(Device::default_device());
    auto values = batchlas::norm<T, MatrixFormat::CSR>(queue, matrix.view(), norm_type);
    queue.wait();
    return wrap_vector_copy(values, matrix.batch_size(), 1);
}

template <typename T, MatrixFormat MF>
DenseVector cond_impl(const Matrix<T, MF>& matrix, NormType norm_type, Backend backend,
                      const std::optional<std::string>& device_name) {
    Queue queue = make_queue(device_name);
    DenseVector result = visit_backend(backend, [&](auto backend_tag) {
        constexpr Backend B = decltype(backend_tag)::value;
        auto values = batchlas::cond<B, T, MF>(queue, matrix.view(), norm_type);
        queue.wait();
        return wrap_vector_copy(values, matrix.batch_size(), 1);
    });
    return result;
}

template <typename T>
DenseMatrix ortho_impl(const DenseMatrix& a_wrapper,
                       const py::dict& options,
                       Backend backend,
                       const std::optional<std::string>& device_name) {
    DenseMatrixT<T> out = std::get<DenseMatrixT<T>>(a_wrapper.storage).clone();
    const Transpose trans_a =
        options.contains("trans_a") ? parse_transpose(py::cast<std::string>(options["trans_a"])) : Transpose::NoTrans;
    const OrthoAlgorithm algorithm =
        options.contains("algorithm") ? parse_ortho_algorithm(py::cast<std::string>(options["algorithm"]))
                                      : OrthoAlgorithm::Chol2;
    Queue queue = make_queue(device_name);
    run_backend_with_workspace(
        backend, queue,
        [&](auto backend_tag) {
            constexpr Backend B = decltype(backend_tag)::value;
            return batchlas::ortho_buffer_size<B, T>(queue, out.view(), trans_a, algorithm);
        },
        [&](auto backend_tag, Span<std::byte> workspace) {
            constexpr Backend B = decltype(backend_tag)::value;
            batchlas::ortho<B, T>(queue, out.view(), trans_a, workspace, algorithm);
        });
    queue.wait();
    return wrap_dense(std::move(out));
}

template <typename T>
DenseMatrix ortho_metric_impl(const DenseMatrix& a_wrapper,
                              const DenseMatrix& m_wrapper,
                              const py::dict& options,
                              Backend backend,
                              const std::optional<std::string>& device_name) {
    ensure_same_dtype(a_wrapper, m_wrapper, "matrix and metric dtypes must match");
    DenseMatrixT<T> out = std::get<DenseMatrixT<T>>(a_wrapper.storage).clone();
    const auto& metric = std::get<DenseMatrixT<T>>(m_wrapper.storage);
    const Transpose trans_a =
        options.contains("trans_a") ? parse_transpose(py::cast<std::string>(options["trans_a"])) : Transpose::NoTrans;
    const Transpose trans_m =
        options.contains("trans_m") ? parse_transpose(py::cast<std::string>(options["trans_m"])) : Transpose::NoTrans;
    const OrthoAlgorithm algorithm =
        options.contains("algorithm") ? parse_ortho_algorithm(py::cast<std::string>(options["algorithm"]))
                                      : OrthoAlgorithm::Chol2;
    const std::size_t iterations = py_scalar_or_default<std::size_t>(options, "iterations", 2);
    Queue queue = make_queue(device_name);
    run_backend_with_workspace(
        backend, queue,
        [&](auto backend_tag) {
            constexpr Backend B = decltype(backend_tag)::value;
            return batchlas::ortho_buffer_size<B, T>(queue, out.view(), metric.view(), trans_a, trans_m, algorithm,
                                                     iterations);
        },
        [&](auto backend_tag, Span<std::byte> workspace) {
            constexpr Backend B = decltype(backend_tag)::value;
            batchlas::ortho<B, T>(queue, out.view(), metric.view(), trans_a, trans_m, workspace, algorithm,
                                  iterations);
        });
    queue.wait();
    return wrap_dense(std::move(out));
}

template <typename T>
ILUKHandle iluk_factorize_impl(const SparseMatrix& matrix_wrapper,
                               const py::dict& options,
                               Backend backend,
                               const std::optional<std::string>& device_name) {
    const auto& matrix = std::get<SparseMatrixT<T>>(matrix_wrapper.storage);
    const ILUKParams<T> params = parse_iluk_params<T>(options);
    Queue queue = make_queue(device_name);
    ILUKPreconditioner<T> handle{};
    visit_backend(backend, [&](auto backend_tag) {
        constexpr Backend B = decltype(backend_tag)::value;
        handle = batchlas::iluk_factorize<B, T>(queue, matrix.view(), params);
    });
    queue.wait();
    return wrap_iluk(std::move(handle));
}

template <typename T>
DenseMatrix iluk_apply_impl(const ILUKHandle& handle_wrapper,
                            const DenseMatrix& rhs_wrapper,
                            Backend backend,
                            const std::optional<std::string>& device_name) {
    ensure_same_dtype(handle_wrapper, rhs_wrapper, "ILUK preconditioner and RHS dtypes must match");
    const auto& handle = std::get<ILUKPreconditioner<T>>(handle_wrapper.storage);
    DenseMatrixT<T> out = std::get<DenseMatrixT<T>>(rhs_wrapper.storage).clone();
    Queue queue = make_queue(device_name);
    run_backend_with_workspace(
        backend, queue,
        [&](auto backend_tag) {
            constexpr Backend B = decltype(backend_tag)::value;
            return batchlas::iluk_apply_buffer_size<B, T>(queue, handle, out.view(), out.view());
        },
        [&](auto backend_tag, Span<std::byte> workspace) {
            constexpr Backend B = decltype(backend_tag)::value;
            batchlas::iluk_apply<B, T>(queue, handle, std::get<DenseMatrixT<T>>(rhs_wrapper.storage).view(),
                                       out.view(), workspace);
        });
    queue.wait();
    return wrap_dense(std::move(out));
}

template <typename T>
DenseMatrix conditioned_random_impl(const std::string& which,
                                    int n,
                                    typename base_type<T>::type log10_kappa,
                                    NormType metric,
                                    int batch_size,
                                    unsigned int seed,
                                    int kd,
                                    OrthoAlgorithm algorithm,
                                    Backend backend,
                                    const std::optional<std::string>& device_name) {
    Queue queue = make_queue(device_name);
    return visit_backend(backend, [&](auto backend_tag) {
        constexpr Backend B = decltype(backend_tag)::value;
        if (which == "random_with_log10_cond_metric") {
            return wrap_dense(batchlas::random_with_log10_cond_metric<B, T>(queue, n, log10_kappa, metric, batch_size,
                                                                            seed, algorithm));
        }
        if (which == "random_hermitian_with_log10_cond_metric") {
            return wrap_dense(batchlas::random_hermitian_with_log10_cond_metric<B, T>(queue, n, log10_kappa, metric,
                                                                                      batch_size, seed, algorithm));
        }
        if (which == "random_banded_with_log10_cond_metric") {
            return wrap_dense(batchlas::random_banded_with_log10_cond_metric<B, T>(queue, n, kd, log10_kappa, metric,
                                                                                   batch_size, seed));
        }
        if (which == "random_hermitian_banded_with_log10_cond_metric") {
            return wrap_dense(batchlas::random_hermitian_banded_with_log10_cond_metric<B, T>(
                queue, n, kd, log10_kappa, metric, batch_size, seed));
        }
        if (which == "random_tridiagonal_with_log10_cond_metric") {
            return wrap_dense(batchlas::random_tridiagonal_with_log10_cond_metric<B, T>(queue, n, log10_kappa, metric,
                                                                                        batch_size, seed));
        }
        if (which == "random_hermitian_tridiagonal_with_log10_cond_metric") {
            return wrap_dense(batchlas::random_hermitian_tridiagonal_with_log10_cond_metric<B, T>(
                queue, n, log10_kappa, metric, batch_size, seed));
        }
        throw py::value_error("unknown conditioned random helper");
    });
}

}  // namespace

void init_misc_ops(py::module_& module) {
    module.def("_transpose_dense", [](const DenseMatrix& matrix) {
        return visit_dense(matrix, [&](auto tag, const auto& typed_matrix) -> DenseMatrix {
            using scalar_type = typename decltype(tag)::type;
            if constexpr (!std::is_floating_point_v<scalar_type>) {
                return not_implemented<DenseMatrix>(
                    "dense transpose is currently available for float32 and float64 only");
            } else {
                Queue queue = Queue(Device::default_device());
                auto out = batchlas::transpose(queue, typed_matrix.view());
                queue.wait();
                return wrap_dense(std::move(out));
            }
        });
    });

    module.def("_transpose_sparse", [](const SparseMatrix& matrix) {
        (void)matrix;
        throw_not_implemented("sparse transpose is not provided by the native BatchLAS library");
    });

    module.def("_lascl_dense", [](const DenseMatrix& matrix, const py::object& cfrom, const py::object& cto) {
        return visit_dense(matrix, [&](auto tag, const auto&) -> DenseMatrix {
            using scalar_type = typename decltype(tag)::type;
            if constexpr (!std::is_floating_point_v<scalar_type>) {
                return not_implemented<DenseMatrix>("lascl is currently available for float32 and float64 only");
            } else {
                return dense_lascl_impl<scalar_type>(matrix, scalar_from_object<scalar_type>(cfrom),
                                                     scalar_from_object<scalar_type>(cto));
            }
        });
    });

    module.def("_lascl_sparse", [](const SparseMatrix& matrix, const py::object& cfrom, const py::object& cto) {
        (void)matrix;
        (void)cfrom;
        (void)cto;
        throw_not_implemented("sparse lascl is not provided by the native BatchLAS library");
    });

    module.def("_norm_dense", [](const DenseMatrix& matrix, const std::string& norm_name) {
        const NormType norm_type = parse_norm_type(norm_name);
        return visit_dense(matrix, [&](auto tag, const auto&) {
            using scalar_type = typename decltype(tag)::type;
            return dense_norm_impl<scalar_type>(matrix, norm_type);
        });
    });

    module.def("_norm_sparse", [](const SparseMatrix& matrix, const std::string& norm_name) {
        (void)matrix;
        (void)norm_name;
        throw_not_implemented("sparse norm is not provided by the native BatchLAS library");
    });

    module.def("_cond_dense", [](const DenseMatrix& matrix,
                                  const std::string& norm_name,
                                  const std::string& backend_name,
                                  const py::object& device_name_obj) {
        const NormType norm_type = parse_norm_type(norm_name);
        const Backend backend = parse_backend(backend_name);
        const auto device_name = optional_string_from_obj(device_name_obj);
        return visit_dense(matrix, [&](auto tag, const auto& typed_matrix) -> DenseVector {
            using scalar_type = typename decltype(tag)::type;
            if constexpr (!std::is_floating_point_v<scalar_type>) {
                return not_implemented<DenseVector>("cond is currently available for float32 and float64 only");
            } else {
                return cond_impl<scalar_type, MatrixFormat::Dense>(typed_matrix, norm_type, backend, device_name);
            }
        });
    });

    module.def("_cond_sparse", [](const SparseMatrix& matrix,
                                   const std::string& norm_name,
                                   const std::string& backend_name,
                                   const py::object& device_name_obj) {
        (void)matrix;
        (void)norm_name;
        (void)backend_name;
        (void)device_name_obj;
        throw_not_implemented("sparse cond is not provided by the native BatchLAS library");
    });

    module.def("_ortho", [](const DenseMatrix& matrix,
                             const py::dict& options,
                             const std::string& backend_name,
                             const py::object& device_name_obj) {
        const Backend backend = parse_backend(backend_name);
        const auto device_name = optional_string_from_obj(device_name_obj);
        return visit_dense(matrix, [&](auto tag, const auto&) {
            using scalar_type = typename decltype(tag)::type;
            return ortho_impl<scalar_type>(matrix, options, backend, device_name);
        });
    });

    module.def("_ortho_metric", [](const DenseMatrix& matrix,
                                    const DenseMatrix& metric,
                                    const py::dict& options,
                                    const std::string& backend_name,
                                    const py::object& device_name_obj) {
        const Backend backend = parse_backend(backend_name);
        const auto device_name = optional_string_from_obj(device_name_obj);
        return visit_dense(matrix, [&](auto tag, const auto&) {
            using scalar_type = typename decltype(tag)::type;
            return ortho_metric_impl<scalar_type>(matrix, metric, options, backend, device_name);
        });
    });

    module.def("_iluk_factorize", [](const SparseMatrix& matrix,
                                      const py::dict& options,
                                      const std::string& backend_name,
                                      const py::object& device_name_obj) {
        const Backend backend = parse_backend(backend_name);
        const auto device_name = optional_string_from_obj(device_name_obj);
        return visit_sparse(matrix, [&](auto tag, const auto&) {
            using scalar_type = typename decltype(tag)::type;
            return iluk_factorize_impl<scalar_type>(matrix, options, backend, device_name);
        });
    });

    module.def("_iluk_apply", [](const ILUKHandle& handle,
                                  const DenseMatrix& rhs,
                                  const std::string& backend_name,
                                  const py::object& device_name_obj) {
        const Backend backend = parse_backend(backend_name);
        const auto device_name = optional_string_from_obj(device_name_obj);
        return visit_iluk(handle, [&](auto tag, const auto&) {
            using scalar_type = typename decltype(tag)::type;
            return iluk_apply_impl<scalar_type>(handle, rhs, backend, device_name);
        });
    });

    module.def("_identity_dense", [](const std::string& dtype_name_value, int n, int batch_size) {
        const DTypeCode dtype = lower_copy(dtype_name_value) == "float32" ? DTypeCode::Float32
                                    : lower_copy(dtype_name_value) == "float64" ? DTypeCode::Float64
                                    : lower_copy(dtype_name_value) == "complex64" ? DTypeCode::Complex64
                                    : lower_copy(dtype_name_value) == "complex128" ? DTypeCode::Complex128
                                    : throw py::value_error("invalid dtype");
        return visit_dtype(dtype, [&](auto tag) {
            using scalar_type = typename decltype(tag)::type;
            return wrap_dense(DenseMatrixT<scalar_type>::Identity(n, batch_size));
        });
    });

    module.def("_random_dense", [](const std::string& dtype_name_value,
                                    int rows,
                                    int cols,
                                    bool hermitian,
                                    int batch_size,
                                    unsigned int seed) {
        const std::string dtype = lower_copy(dtype_name_value);
        const DTypeCode code = dtype == "float32"   ? DTypeCode::Float32
                               : dtype == "float64" ? DTypeCode::Float64
                               : dtype == "complex64" ? DTypeCode::Complex64
                               : dtype == "complex128" ? DTypeCode::Complex128
                               : throw py::value_error("invalid dtype");
        return visit_dtype(code, [&](auto tag) {
            using scalar_type = typename decltype(tag)::type;
            return wrap_dense(DenseMatrixT<scalar_type>::Random(rows, cols, hermitian, batch_size, seed));
        });
    });

    module.def("_zeros_dense", [](const std::string& dtype_name_value, int rows, int cols, int batch_size) {
        const std::string dtype = lower_copy(dtype_name_value);
        const DTypeCode code = dtype == "float32"   ? DTypeCode::Float32
                               : dtype == "float64" ? DTypeCode::Float64
                               : dtype == "complex64" ? DTypeCode::Complex64
                               : dtype == "complex128" ? DTypeCode::Complex128
                               : throw py::value_error("invalid dtype");
        return visit_dtype(code, [&](auto tag) {
            using scalar_type = typename decltype(tag)::type;
            return wrap_dense(DenseMatrixT<scalar_type>::Zeros(rows, cols, batch_size));
        });
    });

    module.def("_ones_dense", [](const std::string& dtype_name_value, int rows, int cols, int batch_size) {
        const std::string dtype = lower_copy(dtype_name_value);
        const DTypeCode code = dtype == "float32"   ? DTypeCode::Float32
                               : dtype == "float64" ? DTypeCode::Float64
                               : dtype == "complex64" ? DTypeCode::Complex64
                               : dtype == "complex128" ? DTypeCode::Complex128
                               : throw py::value_error("invalid dtype");
        return visit_dtype(code, [&](auto tag) {
            using scalar_type = typename decltype(tag)::type;
            return wrap_dense(DenseMatrixT<scalar_type>::Ones(rows, cols, batch_size));
        });
    });

    module.def("_diagonal_dense", [](const DenseVector& diagonal, int batch_size) {
        return visit_vector(diagonal, [&](auto tag, const auto& typed_diagonal) {
            using scalar_type = typename decltype(tag)::type;
            return wrap_dense(DenseMatrixT<scalar_type>::Diagonal(typed_diagonal.data(), batch_size));
        });
    });

    module.def("_triangular_dense", [](const std::string& dtype_name_value,
                                        int n,
                                        const std::string& uplo_name,
                                        const py::object& diagonal_value,
                                        const py::object& non_diagonal_value,
                                        int batch_size) {
        const std::string dtype = lower_copy(dtype_name_value);
        const DTypeCode code = dtype == "float32"   ? DTypeCode::Float32
                               : dtype == "float64" ? DTypeCode::Float64
                               : dtype == "complex64" ? DTypeCode::Complex64
                               : dtype == "complex128" ? DTypeCode::Complex128
                               : throw py::value_error("invalid dtype");
        const Uplo uplo = parse_uplo(uplo_name);
        return visit_dtype(code, [&](auto tag) {
            using scalar_type = typename decltype(tag)::type;
            return wrap_dense(DenseMatrixT<scalar_type>::Triangular(
                n, uplo, scalar_from_object<scalar_type>(diagonal_value),
                scalar_from_object<scalar_type>(non_diagonal_value), batch_size));
        });
    });

    module.def("_tridiag_toeplitz_dense", [](const std::string& dtype_name_value,
                                              int n,
                                              const py::object& diagonal_value,
                                              const py::object& sub_diagonal_value,
                                              const py::object& super_diagonal_value,
                                              int batch_size) {
        const std::string dtype = lower_copy(dtype_name_value);
        const DTypeCode code = dtype == "float32"   ? DTypeCode::Float32
                               : dtype == "float64" ? DTypeCode::Float64
                               : dtype == "complex64" ? DTypeCode::Complex64
                               : dtype == "complex128" ? DTypeCode::Complex128
                               : throw py::value_error("invalid dtype");
        return visit_dtype(code, [&](auto tag) {
            using scalar_type = typename decltype(tag)::type;
            return wrap_dense(DenseMatrixT<scalar_type>::TriDiagToeplitz(
                n, scalar_from_object<scalar_type>(diagonal_value), scalar_from_object<scalar_type>(sub_diagonal_value),
                scalar_from_object<scalar_type>(super_diagonal_value), batch_size));
        });
    });

    module.def("_random_sparse_hermitian", [](const std::string& dtype_name_value,
                                               int n,
                                               float density,
                                               int batch_size,
                                               unsigned int seed,
                                               const py::object& diagonal_boost,
                                               bool shared_pattern) {
        const std::string dtype = lower_copy(dtype_name_value);
        const DTypeCode code = dtype == "float32"   ? DTypeCode::Float32
                               : dtype == "float64" ? DTypeCode::Float64
                               : dtype == "complex64" ? DTypeCode::Complex64
                               : dtype == "complex128" ? DTypeCode::Complex128
                               : throw py::value_error("invalid dtype");
        return visit_dtype(code, [&](auto tag) {
            using scalar_type = typename decltype(tag)::type;
            using real_type = typename base_type<scalar_type>::type;
            return wrap_sparse(SparseMatrixT<scalar_type>::RandomSparseHermitian(
                n, density, batch_size, seed, diagonal_boost.cast<real_type>(), shared_pattern));
        });
    });

    module.def("_conditioned_random_dense", [](const std::string& which,
                                                const std::string& dtype_name_value,
                                                int n,
                                                double log10_kappa,
                                                const std::string& metric_name,
                                                int batch_size,
                                                unsigned int seed,
                                                int kd,
                                                const std::string& algorithm_name,
                                                const std::string& backend_name,
                                                const py::object& device_name_obj) {
        const std::string dtype = lower_copy(dtype_name_value);
        const DTypeCode code = dtype == "float32"   ? DTypeCode::Float32
                               : dtype == "float64" ? DTypeCode::Float64
                               : dtype == "complex64" ? DTypeCode::Complex64
                               : dtype == "complex128" ? DTypeCode::Complex128
                               : throw py::value_error("invalid dtype");
        const NormType metric = parse_norm_type(metric_name);
        const OrthoAlgorithm algorithm = parse_ortho_algorithm(algorithm_name);
        const Backend backend = parse_backend(backend_name);
        const auto device_name = optional_string_from_obj(device_name_obj);
        return visit_dtype(code, [&](auto tag) {
            using scalar_type = typename decltype(tag)::type;
            using real_type = typename base_type<scalar_type>::type;
            return conditioned_random_impl<scalar_type>(which, n, static_cast<real_type>(log10_kappa), metric,
                                                        batch_size, seed, kd, algorithm, backend, device_name);
        });
    });
}

}  // namespace batchlas::python
