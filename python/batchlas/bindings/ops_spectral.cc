#include "init.hh"
#include "support.hh"

namespace batchlas::python {

namespace {

template <typename T>
DenseMatrix dense_syev_common(const DenseMatrix& a_wrapper,
                              bool compute_vectors,
                              Uplo uplo,
                              Backend backend,
                              const std::optional<std::string>& device_name,
                              const py::dict& options,
                              const std::string& variant) {
    DenseMatrixT<T> out = std::get<DenseMatrixT<T>>(a_wrapper.storage).clone();
    Vector<typename base_type<T>::type> eigenvalues(out.rows(), out.batch_size());
    const JobType jobz = compute_vectors ? JobType::EigenVectors : JobType::NoEigenVectors;
    Queue queue = make_queue(device_name);
    run_backend_with_workspace(
        backend, queue,
        [&](auto backend_tag) {
            constexpr Backend B = decltype(backend_tag)::value;
            if (variant == "syev") {
                return batchlas::syev_buffer_size<B, T>(queue, out.view(), eigenvalues.data(), jobz, uplo);
            }
            if (variant == "syev_cta") {
                return batchlas::syev_cta_buffer_size<B, T>(queue, out.view(), jobz, parse_steqr_params<T>(options));
            }
            if (variant == "syev_blocked") {
                return batchlas::syev_blocked_buffer_size<B, T>(
                    queue, out.view(), jobz, uplo, parse_stedc_params<typename base_type<T>::type>(options));
            }
            if (variant == "syev_two_stage") {
                return batchlas::syev_two_stage_buffer_size<B, T>(
                    queue, out.view(), jobz, uplo, parse_stedc_params<typename base_type<T>::type>(options));
            }
            throw py::value_error("unknown syev variant");
        },
        [&](auto backend_tag, Span<std::byte> workspace) {
            constexpr Backend B = decltype(backend_tag)::value;
            if (variant == "syev") {
                batchlas::syev<B, T>(queue, out.view(), eigenvalues.data(), jobz, uplo, workspace);
            } else if (variant == "syev_cta") {
                batchlas::syev_cta<B, T>(queue, out.view(), eigenvalues.data(), jobz, uplo, workspace,
                                         parse_steqr_params<T>(options),
                                         py_scalar_or_default<std::size_t>(options, "cta_wg_size_multiplier", 1));
            } else if (variant == "syev_blocked") {
                batchlas::syev_blocked<B, T>(queue, out.view(), eigenvalues.data(), jobz, uplo, workspace,
                                             parse_stedc_params<typename base_type<T>::type>(options));
            } else {
                batchlas::syev_two_stage<B, T>(queue, out.view(), eigenvalues.data(), jobz, uplo, workspace,
                                               parse_stedc_params<typename base_type<T>::type>(options));
            }
        });
    queue.wait();
    if (compute_vectors) {
        return wrap_dense(std::move(out));
    }
    return wrap_dense(DenseMatrixT<T>(1, 1, 1));
}

template <typename T, MatrixFormat MF>
py::object sparse_iterative_eigensolver(const Matrix<T, MF>& matrix,
                                        std::size_t neigs,
                                        bool compute_vectors,
                                        const py::dict& options,
                                        Backend backend,
                                        const std::optional<std::string>& device_name,
                                        bool use_lanczos,
                                        bool return_history,
                                        const ILUKPreconditioner<T>* preconditioner = nullptr) {
    Queue queue = make_queue(device_name);
    Vector<typename base_type<T>::type> values(static_cast<int>(neigs), matrix.batch_size());
    const JobType jobz = compute_vectors ? JobType::EigenVectors : JobType::NoEigenVectors;
    std::optional<DenseMatrixT<T>> vectors;
    if (compute_vectors) {
        vectors.emplace(matrix.rows(), static_cast<int>(neigs), matrix.batch_size());
    }

    using real_type = typename base_type<T>::type;
    UnifiedVector<real_type> best_history;
    UnifiedVector<real_type> current_history;
    UnifiedVector<real_type> rate_history;
    UnifiedVector<real_type> ritz_history;
    std::vector<int32_t> iterations_done(static_cast<std::size_t>(matrix.batch_size()), 0);
    std::optional<SyevxInstrumentation<T>> instrumentation;
    SyevxParams<T> syevx_params = parse_syevx_params<T>(options, preconditioner, nullptr);
    if (!use_lanczos && return_history) {
        const std::size_t store_every = py_scalar_or_default<std::size_t>(options, "store_every", 1);
        const bool store_current = py_scalar_or_default<bool>(options, "store_current_residual", false);
        const bool store_rate = py_scalar_or_default<bool>(options, "store_convergence_rate", true);
        const bool store_ritz = py_scalar_or_default<bool>(options, "store_ritz_values", false);
        const std::size_t stored_iters = (syevx_params.iterations + store_every - 1) / store_every;
        best_history.resize(stored_iters * matrix.batch_size() * neigs);
        if (store_current) {
            current_history.resize(stored_iters * matrix.batch_size() * neigs);
        }
        if (store_rate) {
            rate_history.resize(stored_iters * matrix.batch_size() * neigs);
        }
        if (store_ritz) {
            ritz_history.resize(stored_iters * matrix.batch_size() * neigs);
        }
        instrumentation = SyevxInstrumentation<T>{};
        instrumentation->best_residual_history = best_history.to_span();
        instrumentation->current_residual_history = current_history.to_span();
        instrumentation->convergence_rate_history = rate_history.to_span();
        instrumentation->ritz_value_history = ritz_history.to_span();
        instrumentation->iterations_done = iterations_done.data();
        instrumentation->max_iterations = syevx_params.iterations;
        instrumentation->store_every = store_every;
        instrumentation->store_current_residual = store_current;
        instrumentation->store_convergence_rate = store_rate;
        instrumentation->store_ritz_values = store_ritz;
        syevx_params.instrumentation = &*instrumentation;
    }

    run_backend_with_workspace(
        backend, queue,
        [&](auto backend_tag) {
            constexpr Backend B = decltype(backend_tag)::value;
            if (use_lanczos) {
                if (compute_vectors) {
                    return batchlas::lanczos_buffer_size<B, T, MF>(
                        queue, matrix.view(), values.data(), jobz, vectors->view(), parse_lanczos_params<T>(options));
                }
                return batchlas::lanczos_buffer_size<B, T, MF>(queue, matrix.view(), values.data(), jobz,
                                                               MatrixView<T, MatrixFormat::Dense>(),
                                                               parse_lanczos_params<T>(options));
            }
            if (compute_vectors) {
                return batchlas::syevx_buffer_size<B, T, MF>(queue, matrix.view(), values.data(), neigs, jobz,
                                                             vectors->view(), syevx_params);
            }
            return batchlas::syevx_buffer_size<B, T, MF>(queue, matrix.view(), values.data(), neigs, jobz,
                                                         MatrixView<T, MatrixFormat::Dense>(), syevx_params);
        },
        [&](auto backend_tag, Span<std::byte> workspace) {
            constexpr Backend B = decltype(backend_tag)::value;
            if (use_lanczos) {
                if (compute_vectors) {
                    batchlas::lanczos<B, T, MF>(queue, matrix.view(), values.data(), workspace, jobz, vectors->view(),
                                                parse_lanczos_params<T>(options));
                } else {
                    batchlas::lanczos<B, T, MF>(queue, matrix.view(), values.data(), workspace, jobz,
                                                MatrixView<T, MatrixFormat::Dense>(),
                                                parse_lanczos_params<T>(options));
                }
            } else {
                if (compute_vectors) {
                    batchlas::syevx<B, T, MF>(queue, matrix.view(), values.data(), neigs, workspace, jobz,
                                              vectors->view(), syevx_params);
                } else {
                    batchlas::syevx<B, T, MF>(queue, matrix.view(), values.data(), neigs, workspace, jobz,
                                              MatrixView<T, MatrixFormat::Dense>(), syevx_params);
                }
            }
        });
    queue.wait();

    py::object values_object = dense_vector_to_python(wrap_vector(std::move(values)));
    py::object vectors_object =
        compute_vectors ? dense_matrix_to_python(wrap_dense(std::move(*vectors))) : py::none();
    if (!return_history || use_lanczos) {
        if (compute_vectors) {
            return py::make_tuple(values_object, vectors_object);
        }
        return values_object;
    }

    const std::size_t store_every = instrumentation->store_every;
    const std::size_t stored_iters = (syevx_params.iterations + store_every - 1) / store_every;
    auto history_array = [&](const UnifiedVector<real_type>& buffer) -> py::object {
        if (buffer.size() == 0) {
            return py::none();
        }
        py::array_t<real_type> out(
            {static_cast<py::ssize_t>(stored_iters), static_cast<py::ssize_t>(matrix.batch_size()),
             static_cast<py::ssize_t>(neigs)});
        auto view = out.template mutable_unchecked<3>();
        for (std::size_t iter = 0; iter < stored_iters; ++iter) {
            for (int batch = 0; batch < matrix.batch_size(); ++batch) {
                for (std::size_t eig = 0; eig < neigs; ++eig) {
                    const std::size_t index =
                        iter * static_cast<std::size_t>(matrix.batch_size()) * neigs +
                        static_cast<std::size_t>(batch) * neigs + eig;
                    view(iter, static_cast<std::size_t>(batch), eig) = buffer[index];
                }
            }
        }
        return out;
    };

    py::dict history;
    history["best_residual_history"] = history_array(best_history);
    history["current_residual_history"] = history_array(current_history);
    history["convergence_rate_history"] = history_array(rate_history);
    history["ritz_value_history"] = history_array(ritz_history);
    py::array_t<int32_t> iterations_out({static_cast<py::ssize_t>(matrix.batch_size())});
    auto iterations_view = iterations_out.mutable_unchecked<1>();
    for (int batch = 0; batch < matrix.batch_size(); ++batch) {
        iterations_view(static_cast<std::size_t>(batch)) = iterations_done[static_cast<std::size_t>(batch)];
    }
    history["iterations_done"] = std::move(iterations_out);

    if (compute_vectors) {
        return py::make_tuple(values_object, vectors_object, history);
    }
    return py::make_tuple(values_object, history);
}

template <typename T>
py::object steqr_common(const DenseVector& d_wrapper,
                        const DenseVector& e_wrapper,
                        bool compute_vectors,
                        const py::dict& options,
                        Backend backend,
                        const std::optional<std::string>& device_name,
                        bool cta) {
    ensure_same_dtype(d_wrapper, e_wrapper, "d and e dtypes must match");
    const auto& d = std::get<Vector<T>>(d_wrapper.storage);
    const auto& e = std::get<Vector<T>>(e_wrapper.storage);
    Vector<T> eigenvalues(d.size(), d.batch_size());
    DenseMatrixT<T> vectors = compute_vectors ? DenseMatrixT<T>(d.size(), d.size(), d.batch_size())
                                              : DenseMatrixT<T>(1, 1, d.batch_size());
    const JobType jobz = compute_vectors ? JobType::EigenVectors : JobType::NoEigenVectors;
    const auto params = parse_steqr_params<T>(options);
    Queue queue = make_queue(device_name);
    const std::size_t workspace_size =
        cta ? batchlas::steqr_cta_buffer_size<T>(queue, VectorView<T>(d), VectorView<T>(e),
                                                 VectorView<T>(eigenvalues), jobz, params)
            : batchlas::steqr_buffer_size<T>(queue, VectorView<T>(d), VectorView<T>(e), VectorView<T>(eigenvalues),
                                             jobz, params);
    UnifiedVector<std::byte> workspace(workspace_size);
    visit_backend(backend, [&](auto backend_tag) {
        constexpr Backend B = decltype(backend_tag)::value;
        if (cta) {
            batchlas::steqr_cta<B, T>(queue, VectorView<T>(d), VectorView<T>(e), VectorView<T>(eigenvalues),
                                      workspace.to_span(), jobz, params, vectors.view());
        } else {
            batchlas::steqr<B, T>(queue, VectorView<T>(d), VectorView<T>(e), VectorView<T>(eigenvalues),
                                  workspace.to_span(), jobz, params, vectors.view());
        }
    });
    queue.wait();
    if (compute_vectors) {
        return py::make_tuple(wrap_vector(std::move(eigenvalues)), wrap_dense(std::move(vectors)));
    }
    return py::cast(wrap_vector(std::move(eigenvalues)));
}

template <typename T>
py::object stedc_common(const DenseVector& d_wrapper,
                        const DenseVector& e_wrapper,
                        bool compute_vectors,
                        const py::dict& options,
                        Backend backend,
                        const std::optional<std::string>& device_name,
                        bool flat) {
    ensure_same_dtype(d_wrapper, e_wrapper, "d and e dtypes must match");
    const auto& d = std::get<Vector<T>>(d_wrapper.storage);
    const auto& e = std::get<Vector<T>>(e_wrapper.storage);
    Vector<T> eigenvalues(d.size(), d.batch_size());
    DenseMatrixT<T> vectors = compute_vectors ? DenseMatrixT<T>(d.size(), d.size(), d.batch_size())
                                              : DenseMatrixT<T>(1, 1, d.batch_size());
    const JobType jobz = compute_vectors ? JobType::EigenVectors : JobType::NoEigenVectors;
    const auto params = parse_stedc_params<T>(options);
    Queue queue = make_queue(device_name);
    const std::size_t workspace_size = visit_backend(backend, [&](auto backend_tag) {
        constexpr Backend B = decltype(backend_tag)::value;
        if (flat) {
            return batchlas::stedc_flat_workspace_size<B, T>(queue, d.size(), d.batch_size(), jobz, params);
        }
        return batchlas::stedc_workspace_size<B, T>(queue, d.size(), d.batch_size(), jobz, params);
    });
    UnifiedVector<std::byte> workspace(workspace_size);
    visit_backend(backend, [&](auto backend_tag) {
        constexpr Backend B = decltype(backend_tag)::value;
        if (flat) {
            batchlas::stedc_flat<B, T>(queue, VectorView<T>(d), VectorView<T>(e), VectorView<T>(eigenvalues),
                                       workspace.to_span(), jobz, params, vectors.view());
        } else {
            batchlas::stedc<B, T>(queue, VectorView<T>(d), VectorView<T>(e), VectorView<T>(eigenvalues),
                                  workspace.to_span(), jobz, params, vectors.view());
        }
    });
    queue.wait();
    if (compute_vectors) {
        return py::make_tuple(wrap_vector(std::move(eigenvalues)), wrap_dense(std::move(vectors)));
    }
    return py::cast(wrap_vector(std::move(eigenvalues)));
}

template <typename T>
py::object tridiagonal_solver_impl(const DenseVector& alpha_wrapper,
                                   const DenseVector& beta_wrapper,
                                   bool compute_vectors,
                                   Backend backend,
                                   const std::optional<std::string>& device_name) {
    ensure_same_dtype(alpha_wrapper, beta_wrapper, "alpha and beta dtypes must match");
    const auto& alpha = std::get<Vector<T>>(alpha_wrapper.storage);
    const auto& beta = std::get<Vector<T>>(beta_wrapper.storage);
    Vector<typename base_type<T>::type> eigenvalues(alpha.size(), alpha.batch_size());
    DenseMatrixT<T> q = compute_vectors ? DenseMatrixT<T>(alpha.size(), alpha.size(), alpha.batch_size())
                                        : DenseMatrixT<T>(1, 1, alpha.batch_size());
    const JobType jobz = compute_vectors ? JobType::EigenVectors : JobType::NoEigenVectors;
    Queue queue = make_queue(device_name);
    run_backend_with_workspace(
        backend, queue,
        [&](auto backend_tag) {
            constexpr Backend B = decltype(backend_tag)::value;
            return batchlas::tridiagonal_solver_buffer_size<B, T>(queue, alpha.size(), alpha.batch_size(), jobz);
        },
        [&](auto backend_tag, Span<std::byte> workspace) {
            constexpr Backend B = decltype(backend_tag)::value;
            batchlas::tridiagonal_solver<B, T>(queue, alpha.data(), beta.data(), eigenvalues.data(), workspace, jobz,
                                               q.view(), alpha.size(), alpha.batch_size());
        });
    queue.wait();
    if (compute_vectors) {
        return py::make_tuple(wrap_vector(std::move(eigenvalues)), wrap_dense(std::move(q)));
    }
    return py::cast(wrap_vector(std::move(eigenvalues)));
}

template <typename T, MatrixFormat MF>
DenseVector ritz_values_impl(const Matrix<T, MF>& matrix,
                             const DenseMatrix& vectors_wrapper,
                             Backend backend,
                             const std::optional<std::string>& device_name) {
    const auto& vectors = std::get<DenseMatrixT<T>>(vectors_wrapper.storage);
    Queue queue = make_queue(device_name);
    return visit_backend(backend, [&](auto backend_tag) {
        constexpr Backend B = decltype(backend_tag)::value;
        auto values = batchlas::ritz_values<B, T, MF>(queue, matrix.view(), vectors.view());
        queue.wait();
        return wrap_vector(std::move(values));
    });
}

}  // namespace

void init_spectral_ops(py::module_& module) {
    module.def("_syev", [](const DenseMatrix& matrix,
                            bool compute_vectors,
                            const std::string& uplo_name,
                            const py::dict& options,
                            const std::string& backend_name,
                            const py::object& device_name_obj) {
        const Uplo uplo = parse_uplo(uplo_name);
        const Backend backend = parse_backend(backend_name);
        const auto device_name = optional_string_from_obj(device_name_obj);
        return visit_dense(matrix, [&](auto tag, const auto&) -> py::object {
            using scalar_type = typename decltype(tag)::type;
            auto vectors = dense_syev_common<scalar_type>(matrix, compute_vectors, uplo, backend, device_name,
                                                          options, "syev");
            const auto& typed_vectors = std::get<DenseMatrixT<scalar_type>>(vectors.storage);
            Vector<typename base_type<scalar_type>::type> values(typed_vectors.rows(), typed_vectors.batch_size());
            DenseMatrixT<scalar_type> a_copy = std::get<DenseMatrixT<scalar_type>>(matrix.storage).clone();
            Queue queue = make_queue(device_name);
            run_backend_with_workspace(
                backend, queue,
                [&](auto backend_tag) {
                    constexpr Backend B = decltype(backend_tag)::value;
                    const JobType jobz = compute_vectors ? JobType::EigenVectors : JobType::NoEigenVectors;
                    return batchlas::syev_buffer_size<B, scalar_type>(queue, a_copy.view(), values.data(), jobz, uplo);
                },
                [&](auto backend_tag, Span<std::byte> workspace) {
                    constexpr Backend B = decltype(backend_tag)::value;
                    const JobType jobz = compute_vectors ? JobType::EigenVectors : JobType::NoEigenVectors;
                    batchlas::syev<B, scalar_type>(queue, a_copy.view(), values.data(), jobz, uplo, workspace);
                });
            queue.wait();
            if (compute_vectors) {
                return py::make_tuple(wrap_vector(std::move(values)), wrap_dense(std::move(a_copy)));
            }
            return py::cast(wrap_vector(std::move(values)));
        });
    });

    module.def("_syev_cta", [](const DenseMatrix& matrix, bool compute_vectors, const std::string& uplo_name,
                                const py::dict& options, const std::string& backend_name,
                                const py::object& device_name_obj) {
        const Uplo uplo = parse_uplo(uplo_name);
        const Backend backend = parse_backend(backend_name);
        const auto device_name = optional_string_from_obj(device_name_obj);
        return visit_dense(matrix, [&](auto tag, const auto&) -> py::object {
            using scalar_type = typename decltype(tag)::type;
            DenseMatrixT<scalar_type> out = std::get<DenseMatrixT<scalar_type>>(matrix.storage).clone();
            Vector<typename base_type<scalar_type>::type> values(out.rows(), out.batch_size());
            const JobType jobz = compute_vectors ? JobType::EigenVectors : JobType::NoEigenVectors;
            Queue queue = make_queue(device_name);
            run_backend_with_workspace(
                backend, queue,
                [&](auto backend_tag) {
                    constexpr Backend B = decltype(backend_tag)::value;
                    const auto params = parse_steqr_params<scalar_type>(options);
                    return batchlas::syev_cta_buffer_size<B, scalar_type>(queue, out.view(), jobz, params);
                },
                [&](auto backend_tag, Span<std::byte> workspace) {
                    constexpr Backend B = decltype(backend_tag)::value;
                    const auto params = parse_steqr_params<scalar_type>(options);
                    batchlas::syev_cta<B, scalar_type>(
                        queue, out.view(), values.data(), jobz, uplo, workspace, params,
                        py_scalar_or_default<std::size_t>(options, "cta_wg_size_multiplier", 1));
                });
            queue.wait();
            if (compute_vectors) {
                return py::make_tuple(wrap_vector(std::move(values)), wrap_dense(std::move(out)));
            }
            return py::cast(wrap_vector(std::move(values)));
        });
    });

    module.def("_syev_blocked", [](const DenseMatrix& matrix, bool compute_vectors, const std::string& uplo_name,
                                    const py::dict& options, const std::string& backend_name,
                                    const py::object& device_name_obj) {
        const Uplo uplo = parse_uplo(uplo_name);
        const Backend backend = parse_backend(backend_name);
        const auto device_name = optional_string_from_obj(device_name_obj);
        return visit_dense(matrix, [&](auto tag, const auto&) -> py::object {
            using scalar_type = typename decltype(tag)::type;
            DenseMatrixT<scalar_type> out = std::get<DenseMatrixT<scalar_type>>(matrix.storage).clone();
            Vector<typename base_type<scalar_type>::type> values(out.rows(), out.batch_size());
            const JobType jobz = compute_vectors ? JobType::EigenVectors : JobType::NoEigenVectors;
            const auto params = parse_stedc_params<typename base_type<scalar_type>::type>(options);
            Queue queue = make_queue(device_name);
            run_backend_with_workspace(
                backend, queue,
                [&](auto backend_tag) {
                    constexpr Backend B = decltype(backend_tag)::value;
                    return batchlas::syev_blocked_buffer_size<B, scalar_type>(queue, out.view(), jobz, uplo, params);
                },
                [&](auto backend_tag, Span<std::byte> workspace) {
                    constexpr Backend B = decltype(backend_tag)::value;
                    batchlas::syev_blocked<B, scalar_type>(queue, out.view(), values.data(), jobz, uplo, workspace,
                                                           params);
                });
            queue.wait();
            if (compute_vectors) {
                return py::make_tuple(wrap_vector(std::move(values)), wrap_dense(std::move(out)));
            }
            return py::cast(wrap_vector(std::move(values)));
        });
    });

    module.def("_syev_two_stage", [](const DenseMatrix& matrix, bool compute_vectors, const std::string& uplo_name,
                                      const py::dict& options, const std::string& backend_name,
                                      const py::object& device_name_obj) {
        const Uplo uplo = parse_uplo(uplo_name);
        const Backend backend = parse_backend(backend_name);
        const auto device_name = optional_string_from_obj(device_name_obj);
        return visit_dense(matrix, [&](auto tag, const auto&) -> py::object {
            using scalar_type = typename decltype(tag)::type;
            DenseMatrixT<scalar_type> out = std::get<DenseMatrixT<scalar_type>>(matrix.storage).clone();
            Vector<typename base_type<scalar_type>::type> values(out.rows(), out.batch_size());
            const JobType jobz = compute_vectors ? JobType::EigenVectors : JobType::NoEigenVectors;
            const auto params = parse_stedc_params<typename base_type<scalar_type>::type>(options);
            Queue queue = make_queue(device_name);
            run_backend_with_workspace(
                backend, queue,
                [&](auto backend_tag) {
                    constexpr Backend B = decltype(backend_tag)::value;
                    return batchlas::syev_two_stage_buffer_size<B, scalar_type>(queue, out.view(), jobz, uplo, params);
                },
                [&](auto backend_tag, Span<std::byte> workspace) {
                    constexpr Backend B = decltype(backend_tag)::value;
                    batchlas::syev_two_stage<B, scalar_type>(queue, out.view(), values.data(), jobz, uplo, workspace,
                                                             params);
                });
            queue.wait();
            if (compute_vectors) {
                return py::make_tuple(wrap_vector(std::move(values)), wrap_dense(std::move(out)));
            }
            return py::cast(wrap_vector(std::move(values)));
        });
    });

    module.def("_syevx_dense", [](const DenseMatrix& matrix, std::size_t neigs, bool compute_vectors,
                                   const py::dict& options, const std::string& backend_name,
                                   const py::object& device_name_obj, bool return_history, const py::object& preconditioner) {
        const Backend backend = parse_backend(backend_name);
        const auto device_name = optional_string_from_obj(device_name_obj);
        if (!preconditioner.is_none()) {
            throw py::value_error("dense syevx does not accept an ILUK preconditioner");
        }
        return visit_dense(matrix, [&](auto tag, const auto& typed_matrix) -> py::object {
            using scalar_type = typename decltype(tag)::type;
            return sparse_iterative_eigensolver<scalar_type, MatrixFormat::Dense>(typed_matrix, neigs, compute_vectors,
                                                                                  options, backend, device_name, false,
                                                                                  return_history);
        });
    });

    module.def("_syevx_sparse", [](const SparseMatrix& matrix, std::size_t neigs, bool compute_vectors,
                                    const py::dict& options, const std::string& backend_name,
                                    const py::object& device_name_obj, bool return_history, const py::object& preconditioner) {
        const Backend backend = parse_backend(backend_name);
        const auto device_name = optional_string_from_obj(device_name_obj);
        return visit_sparse(matrix, [&](auto tag, const auto& typed_matrix) -> py::object {
            using scalar_type = typename decltype(tag)::type;
            const ILUKPreconditioner<scalar_type>* handle_ptr = nullptr;
            if (!preconditioner.is_none()) {
                const auto& handle = preconditioner.cast<const ILUKHandle&>();
                ensure_same_dtype(handle, matrix, "preconditioner and sparse matrix dtypes must match");
                handle_ptr = &std::get<ILUKPreconditioner<scalar_type>>(handle.storage);
            }
            return sparse_iterative_eigensolver<scalar_type, MatrixFormat::CSR>(typed_matrix, neigs, compute_vectors,
                                                                                options, backend, device_name, false,
                                                                                return_history, handle_ptr);
        });
    });

    module.def("_lanczos_dense", [](const DenseMatrix& matrix, bool compute_vectors, const py::dict& options,
                                     const std::string& backend_name, const py::object& device_name_obj) {
        const Backend backend = parse_backend(backend_name);
        const auto device_name = optional_string_from_obj(device_name_obj);
        return visit_dense(matrix, [&](auto tag, const auto& typed_matrix) -> py::object {
            using scalar_type = typename decltype(tag)::type;
            if constexpr (!std::is_floating_point_v<scalar_type>) {
                return not_implemented<py::object>("lanczos only supports float32 and float64");
            } else {
                return sparse_iterative_eigensolver<scalar_type, MatrixFormat::Dense>(
                    typed_matrix, typed_matrix.rows(), compute_vectors, options, backend, device_name, true, false);
            }
        });
    });

    module.def("_lanczos_sparse", [](const SparseMatrix& matrix, bool compute_vectors, const py::dict& options,
                                      const std::string& backend_name, const py::object& device_name_obj) {
        const Backend backend = parse_backend(backend_name);
        const auto device_name = optional_string_from_obj(device_name_obj);
        return visit_sparse(matrix, [&](auto tag, const auto& typed_matrix) -> py::object {
            using scalar_type = typename decltype(tag)::type;
            if constexpr (!std::is_floating_point_v<scalar_type>) {
                return not_implemented<py::object>("lanczos only supports float32 and float64");
            } else {
                return sparse_iterative_eigensolver<scalar_type, MatrixFormat::CSR>(
                    typed_matrix, typed_matrix.rows(), compute_vectors, options, backend, device_name, true, false);
            }
        });
    });

    module.def("_steqr", [](const DenseVector& d, const DenseVector& e, bool compute_vectors, const py::dict& options,
                             const std::string& backend_name, const py::object& device_name_obj) {
        const Backend backend = parse_backend(backend_name);
        const auto device_name = optional_string_from_obj(device_name_obj);
        return visit_vector(d, [&](auto tag, const auto&) -> py::object {
            using scalar_type = typename decltype(tag)::type;
            if constexpr (!std::is_floating_point_v<scalar_type>) {
                throw_not_implemented("steqr only supports float32 and float64");
            } else {
                return steqr_common<scalar_type>(d, e, compute_vectors, options, backend, device_name, false);
            }
        });
    });

    module.def("_steqr_cta", [](const DenseVector& d, const DenseVector& e, bool compute_vectors, const py::dict& options,
                                 const std::string& backend_name, const py::object& device_name_obj) {
        const Backend backend = parse_backend(backend_name);
        const auto device_name = optional_string_from_obj(device_name_obj);
        return visit_vector(d, [&](auto tag, const auto&) -> py::object {
            using scalar_type = typename decltype(tag)::type;
            if constexpr (!std::is_floating_point_v<scalar_type>) {
                throw_not_implemented("steqr_cta only supports float32 and float64");
            } else {
                return steqr_common<scalar_type>(d, e, compute_vectors, options, backend, device_name, true);
            }
        });
    });

    module.def("_stedc", [](const DenseVector& d, const DenseVector& e, bool compute_vectors, const py::dict& options,
                             const std::string& backend_name, const py::object& device_name_obj) {
        const Backend backend = parse_backend(backend_name);
        const auto device_name = optional_string_from_obj(device_name_obj);
        return visit_vector(d, [&](auto tag, const auto&) -> py::object {
            using scalar_type = typename decltype(tag)::type;
            if constexpr (!std::is_floating_point_v<scalar_type>) {
                throw_not_implemented("stedc only supports float32 and float64");
            } else {
                return stedc_common<scalar_type>(d, e, compute_vectors, options, backend, device_name, false);
            }
        });
    });

    module.def("_stedc_flat", [](const DenseVector& d, const DenseVector& e, bool compute_vectors, const py::dict& options,
                                  const std::string& backend_name, const py::object& device_name_obj) {
        const Backend backend = parse_backend(backend_name);
        const auto device_name = optional_string_from_obj(device_name_obj);
        return visit_vector(d, [&](auto tag, const auto&) -> py::object {
            using scalar_type = typename decltype(tag)::type;
            if constexpr (!std::is_floating_point_v<scalar_type>) {
                throw_not_implemented("stedc_flat only supports float32 and float64");
            } else {
                return stedc_common<scalar_type>(d, e, compute_vectors, options, backend, device_name, true);
            }
        });
    });

    module.def("_tridiagonal_solver", [](const DenseVector& alpha, const DenseVector& beta, bool compute_vectors,
                                          const std::string& backend_name, const py::object& device_name_obj) {
        const Backend backend = parse_backend(backend_name);
        const auto device_name = optional_string_from_obj(device_name_obj);
        return visit_vector(alpha, [&](auto tag, const auto&) -> py::object {
            using scalar_type = typename decltype(tag)::type;
            if constexpr (!std::is_floating_point_v<scalar_type>) {
                return not_implemented<py::object>("tridiagonal_solver only supports float32 and float64");
            } else {
                return tridiagonal_solver_impl<scalar_type>(alpha, beta, compute_vectors, backend, device_name);
            }
        });
    });

    module.def("_ritz_values_dense", [](const DenseMatrix& matrix, const DenseMatrix& vectors, const std::string& backend_name,
                                         const py::object& device_name_obj) {
        const Backend backend = parse_backend(backend_name);
        const auto device_name = optional_string_from_obj(device_name_obj);
        ensure_same_dtype(matrix, vectors, "matrix and trial vectors dtypes must match");
        return visit_dense(matrix, [&](auto tag, const auto& typed_matrix) {
            using scalar_type = typename decltype(tag)::type;
            return ritz_values_impl<scalar_type, MatrixFormat::Dense>(typed_matrix, vectors, backend, device_name);
        });
    });

    module.def("_ritz_values_sparse", [](const SparseMatrix& matrix, const DenseMatrix& vectors, const std::string& backend_name,
                                          const py::object& device_name_obj) {
        const Backend backend = parse_backend(backend_name);
        const auto device_name = optional_string_from_obj(device_name_obj);
        ensure_same_dtype(matrix, vectors, "matrix and trial vectors dtypes must match");
        return visit_sparse(matrix, [&](auto tag, const auto& typed_matrix) {
            using scalar_type = typename decltype(tag)::type;
            return ritz_values_impl<scalar_type, MatrixFormat::CSR>(typed_matrix, vectors, backend, device_name);
        });
    });
}

}  // namespace batchlas::python
