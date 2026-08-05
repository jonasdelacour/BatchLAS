#include "init.hh"
#include "support.hh"

#include <blas/dispatch/context.hh>

namespace batchlas::python {

namespace {

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
    // lanczos sorts its Ritz values through a helper that derives the batch size
    // from the eigenvector matrix, so that buffer must exist even when the caller
    // only wants values.
    if (compute_vectors || use_lanczos) {
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

    // Per-batch-item count of eigenpairs actually found (LAPACK's M).
    //
    // For SyevxSelect::Value this is the ONLY way to know how much of `values` /
    // `vectors` is meaningful: the count depends on the data, differs from one
    // batch item to the next, and `neigs` is merely the capacity the caller
    // declared. m[b] > neigs is therefore the caller's overflow signal, and
    // w[b, m[b]:] is undefined -- syevx leaves those slots of W untouched (it
    // does zero the corresponding columns of V, but do not rely on that here).
    // For Extremal and Index the count is static and syevx fills `m` with it.
    //
    // Allocated for every syevx call -- batch_size int32s is nothing next to the
    // workspace -- but RETURNED only when a range was actually requested, so no
    // existing caller's return shape changes.
    const bool report_counts = !use_lanczos && syevx_params.select != SyevxSelect::Extremal;
    // Default-constructed then resized, exactly like the history buffers below:
    // the lanczos path never calls syevx and must not allocate a zero-length
    // unified allocation just to leave it unused.
    UnifiedVector<int32_t> counts;
    if (!use_lanczos) {
        counts.resize(static_cast<std::size_t>(matrix.batch_size()));
    }

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
                return batchlas::lanczos_buffer_size<B, T, MF>(
                    queue, matrix.view(), values.data(), jobz, vectors->view(), parse_lanczos_params<T>(options));
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
                batchlas::lanczos<B, T, MF>(queue, matrix.view(), values.data(), workspace, jobz, vectors->view(),
                                            parse_lanczos_params<T>(options));
            } else {
                // Always the `m`-taking overload: SyevxSelect::Value REQUIRES it
                // (the m-less form throws for a value range, by design, because it
                // has nowhere to report a data-dependent count), and for Extremal
                // and Index it costs one extra batch_size-wide fill.
                if (compute_vectors) {
                    batchlas::syevx<B, T, MF>(queue, matrix.view(), values.data(), counts.to_span(), neigs,
                                              workspace, jobz, vectors->view(), syevx_params);
                } else {
                    batchlas::syevx<B, T, MF>(queue, matrix.view(), values.data(), counts.to_span(), neigs,
                                              workspace, jobz, MatrixView<T, MatrixFormat::Dense>(),
                                              syevx_params);
                }
            }
        });
    queue.wait();

    py::object values_object = dense_vector_to_python(wrap_vector(std::move(values)));
    py::object vectors_object =
        compute_vectors ? dense_matrix_to_python(wrap_dense(std::move(*vectors))) : py::none();

    // Return layout: (values[, vectors][, m][, history]). `m` sits immediately
    // after the eigenvectors, mirroring the C++ argument order where the count
    // span follows W, and `history` stays last.
    py::object counts_object = py::none();
    if (report_counts) {
        py::array_t<int32_t> counts_out({static_cast<py::ssize_t>(matrix.batch_size())});
        auto counts_view = counts_out.mutable_unchecked<1>();
        for (int batch = 0; batch < matrix.batch_size(); ++batch) {
            counts_view(static_cast<py::ssize_t>(batch)) = counts[static_cast<std::size_t>(batch)];
        }
        counts_object = std::move(counts_out);
    }

    if (!return_history || use_lanczos) {
        if (compute_vectors && report_counts) {
            return py::make_tuple(values_object, vectors_object, counts_object);
        }
        if (compute_vectors) {
            return py::make_tuple(values_object, vectors_object);
        }
        if (report_counts) {
            return py::make_tuple(values_object, counts_object);
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

    if (compute_vectors && report_counts) {
        return py::make_tuple(values_object, vectors_object, counts_object, history);
    }
    if (compute_vectors) {
        return py::make_tuple(values_object, vectors_object, history);
    }
    if (report_counts) {
        return py::make_tuple(values_object, counts_object, history);
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
    // stedc's NoEigenVectors path is not currently correct (it still slices the
    // eigenvector output, and returns wrong eigenvalues), so we always drive it
    // the way syev_blocked does -- request vectors, then discard them if the
    // caller did not ask for them.
    DenseMatrixT<T> vectors(d.size(), d.size(), d.batch_size());
    const JobType jobz = JobType::EigenVectors;
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
    const auto& beta_in = std::get<Vector<T>>(beta_wrapper.storage);
    const int n = alpha.size();
    if (beta_in.size() != n && beta_in.size() != n - 1) {
        throw py::value_error("beta must have n or n - 1 entries");
    }
    // The kernel indexes betas with stride n, so a caller-supplied n-1 vector has
    // to be repacked; otherwise every batch after the first reads the wrong data.
    Vector<T> beta(n, alpha.batch_size());
    for (int batch = 0; batch < alpha.batch_size(); ++batch) {
        for (int index = 0; index < n; ++index) {
            const bool in_range = index < beta_in.size();
            beta[static_cast<std::size_t>(batch * beta.stride() + index * beta.inc())] =
                in_range ? beta_in.data()[static_cast<std::size_t>(batch * beta_in.stride() +
                                                                   index * beta_in.inc())]
                         : T(0);
        }
    }
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

template <typename T>
py::object syev_jacobi_cta_impl(const DenseMatrix& a_wrapper,
                                bool compute_vectors,
                                Uplo uplo,
                                const py::dict& options,
                                Backend backend,
                                const std::optional<std::string>& device_name) {
    DenseMatrixT<T> out = std::get<DenseMatrixT<T>>(a_wrapper.storage).clone();
    Vector<typename base_type<T>::type> values(out.rows(), out.batch_size());
    const JobType jobz = compute_vectors ? JobType::EigenVectors : JobType::NoEigenVectors;
    const auto params = parse_jacobi_params<T>(options);
    Queue queue = make_queue(device_name);
    run_backend_with_workspace(
        backend, queue,
        [&](auto backend_tag) {
            constexpr Backend B = decltype(backend_tag)::value;
            return batchlas::syev_jacobi_cta_buffer_size<B, T>(queue, out.view(), jobz, params);
        },
        [&](auto backend_tag, Span<std::byte> workspace) {
            constexpr Backend B = decltype(backend_tag)::value;
            batchlas::syev_jacobi_cta<B, T>(queue, out.view(), values.data(), jobz, uplo, workspace, params);
        });
    queue.wait();
    if (compute_vectors) {
        return py::make_tuple(wrap_vector(std::move(values)), wrap_dense(std::move(out)));
    }
    return py::cast(wrap_vector(std::move(values)));
}

// sytrd_cta and sytrd_blocked share the same (A, d, e, tau) contract; `blocked`
// selects which kernel runs and how the workspace is sized.
template <typename T>
py::tuple sytrd_dense_impl(const DenseMatrix& a_wrapper,
                           Uplo uplo,
                           bool blocked,
                           int32_t block_size,
                           std::size_t cta_wg_size_multiplier,
                           Backend backend,
                           const std::optional<std::string>& device_name) {
    DenseMatrixT<T> out = std::get<DenseMatrixT<T>>(a_wrapper.storage).clone();
    if (out.rows() != out.cols()) {
        throw py::value_error("sytrd requires square matrices");
    }
    const int n = out.rows();
    const int off = std::max(0, n - 1);
    Vector<T> d(n, out.batch_size());
    Vector<T> e(off, out.batch_size());
    Vector<T> tau(off, out.batch_size());
    Queue queue = make_queue(device_name);
    if (blocked) {
        run_backend_with_workspace(
            backend, queue,
            [&](auto backend_tag) {
                constexpr Backend B = decltype(backend_tag)::value;
                return batchlas::sytrd_blocked_buffer_size<B, T>(queue, out.view(), d, e, tau, uplo, block_size);
            },
            [&](auto backend_tag, Span<std::byte> workspace) {
                constexpr Backend B = decltype(backend_tag)::value;
                batchlas::sytrd_blocked<B, T>(queue, out.view(), d, e, tau, uplo, workspace, block_size);
            });
    } else {
        visit_backend(backend, [&](auto backend_tag) {
            constexpr Backend B = decltype(backend_tag)::value;
            // sytrd_cta needs no global workspace; the span is accepted for API symmetry.
            batchlas::sytrd_cta<B, T>(queue, out.view(), d, e, tau, uplo, Span<std::byte>(),
                                      cta_wg_size_multiplier);
        });
    }
    queue.wait();
    return py::make_tuple(wrap_dense(std::move(out)),
                          wrap_vector(std::move(d)),
                          wrap_vector(std::move(e)),
                          wrap_vector(std::move(tau)));
}

template <typename T>
py::tuple sytrd_sy2sb_impl(const DenseMatrix& a_wrapper,
                           Uplo uplo,
                           int32_t kd,
                           Backend backend,
                           const std::optional<std::string>& device_name) {
    DenseMatrixT<T> out = std::get<DenseMatrixT<T>>(a_wrapper.storage).clone();
    if (out.rows() != out.cols()) {
        throw py::value_error("sytrd_sy2sb requires square matrices");
    }
    const int n = out.rows();
    if (kd < 1 || kd >= n) {
        throw py::value_error("kd must satisfy 1 <= kd < n");
    }
    DenseMatrixT<T> ab(kd + 1, n, out.batch_size());
    Vector<T> tau(n - kd, out.batch_size());
    Queue queue = make_queue(device_name);
    run_backend_with_workspace(
        backend, queue,
        [&](auto backend_tag) {
            constexpr Backend B = decltype(backend_tag)::value;
            return batchlas::sytrd_sy2sb_buffer_size<B, T>(queue, out.view(), ab.view(), tau, uplo, kd);
        },
        [&](auto backend_tag, Span<std::byte> workspace) {
            constexpr Backend B = decltype(backend_tag)::value;
            batchlas::sytrd_sy2sb<B, T>(queue, out.view(), ab.view(), tau, uplo, kd, workspace);
        });
    queue.wait();
    return py::make_tuple(wrap_dense(std::move(out)), wrap_dense(std::move(ab)), wrap_vector(std::move(tau)));
}

// Band -> tridiagonal. `bandr1` picks the BANDR1-style schedule
// (sytrd_band_reduction) over the bulge-chasing sb2st/hb2st path.
template <typename T>
py::tuple sytrd_band_to_tridiagonal_impl(const DenseMatrix& ab_wrapper,
                                         Uplo uplo,
                                         int32_t kd,
                                         int32_t block_size,
                                         bool bandr1,
                                         const py::dict& options,
                                         Backend backend,
                                         const std::optional<std::string>& device_name) {
    using real_type = typename base_type<T>::type;
    const auto& ab = std::get<DenseMatrixT<T>>(ab_wrapper.storage);
    if (ab.rows() != kd + 1) {
        throw py::value_error("band storage must have exactly kd + 1 rows");
    }
    const int n = ab.cols();
    const int off = std::max(0, n - 1);
    Vector<real_type> d(n, ab.batch_size());
    Vector<real_type> e(off, ab.batch_size());
    Vector<T> tau(off, ab.batch_size());
    const auto params = parse_sytrd_band_reduction_params(options);
    Queue queue = make_queue(device_name);
    run_backend_with_workspace(
        backend, queue,
        [&](auto backend_tag) {
            constexpr Backend B = decltype(backend_tag)::value;
            if (bandr1) {
                return batchlas::sytrd_band_reduction_buffer_size<B, T>(queue, ab.view(), d, e, tau, uplo, kd, params);
            }
            return batchlas::sytrd_sb2st_buffer_size<B, T>(queue, ab.view(), d, e, tau, uplo, kd, block_size);
        },
        [&](auto backend_tag, Span<std::byte> workspace) {
            constexpr Backend B = decltype(backend_tag)::value;
            if (bandr1) {
                batchlas::sytrd_band_reduction<B, T>(queue, ab.view(), d, e, tau, uplo, kd, workspace, params);
            } else {
                batchlas::sytrd_sb2st<B, T>(queue, ab.view(), d, e, tau, uplo, kd, workspace, block_size);
            }
        });
    queue.wait();
    return py::make_tuple(wrap_vector(std::move(d)), wrap_vector(std::move(e)), wrap_vector(std::move(tau)));
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
            DenseMatrixT<scalar_type> a_copy = std::get<DenseMatrixT<scalar_type>>(matrix.storage).clone();
            // Size the eigenvalue buffer from the input, not from the output vectors:
            // with compute_vectors=false there are no output vectors to measure.
            Vector<typename base_type<scalar_type>::type> values(a_copy.rows(), a_copy.batch_size());
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
            // Its eigenvalues-only path forwards NoEigenVectors to stedc, whose
            // NoEigenVectors path is not currently correct, so always ask for
            // vectors and drop them if the caller did not want them.
            const JobType jobz = JobType::EigenVectors;
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

    module.def("_syev_jacobi_cta", [](const DenseMatrix& matrix, bool compute_vectors, const std::string& uplo_name,
                                       const py::dict& options, const std::string& backend_name,
                                       const py::object& device_name_obj) {
        const Uplo uplo = parse_uplo(uplo_name);
        const Backend backend = parse_backend(backend_name);
        const auto device_name = optional_string_from_obj(device_name_obj);
        return visit_dense(matrix, [&](auto tag, const auto&) -> py::object {
            using scalar_type = typename decltype(tag)::type;
            return syev_jacobi_cta_impl<scalar_type>(matrix, compute_vectors, uplo, options, backend, device_name);
        });
    });

    module.def("_sytrd_cta", [](const DenseMatrix& matrix, const std::string& uplo_name,
                                 std::size_t cta_wg_size_multiplier, const std::string& backend_name,
                                 const py::object& device_name_obj) {
        const Uplo uplo = parse_uplo(uplo_name);
        const Backend backend = parse_backend(backend_name);
        const auto device_name = optional_string_from_obj(device_name_obj);
        return visit_dense(matrix, [&](auto tag, const auto&) {
            using scalar_type = typename decltype(tag)::type;
            return sytrd_dense_impl<scalar_type>(matrix, uplo, false, 0, cta_wg_size_multiplier, backend, device_name);
        });
    });

    module.def("_sytrd_blocked", [](const DenseMatrix& matrix, const std::string& uplo_name, int32_t block_size,
                                     const std::string& backend_name, const py::object& device_name_obj) {
        const Uplo uplo = parse_uplo(uplo_name);
        const Backend backend = parse_backend(backend_name);
        const auto device_name = optional_string_from_obj(device_name_obj);
        return visit_dense(matrix, [&](auto tag, const auto&) {
            using scalar_type = typename decltype(tag)::type;
            return sytrd_dense_impl<scalar_type>(matrix, uplo, true, block_size, 1, backend, device_name);
        });
    });

    module.def("_sytrd_sy2sb", [](const DenseMatrix& matrix, const std::string& uplo_name, int32_t kd,
                                   const std::string& backend_name, const py::object& device_name_obj) {
        const Uplo uplo = parse_uplo(uplo_name);
        const Backend backend = parse_backend(backend_name);
        const auto device_name = optional_string_from_obj(device_name_obj);
        return visit_dense(matrix, [&](auto tag, const auto&) {
            using scalar_type = typename decltype(tag)::type;
            return sytrd_sy2sb_impl<scalar_type>(matrix, uplo, kd, backend, device_name);
        });
    });

    module.def("_sytrd_sb2st", [](const DenseMatrix& ab, const std::string& uplo_name, int32_t kd, int32_t block_size,
                                   const std::string& backend_name, const py::object& device_name_obj) {
        const Uplo uplo = parse_uplo(uplo_name);
        const Backend backend = parse_backend(backend_name);
        const auto device_name = optional_string_from_obj(device_name_obj);
        return visit_dense(ab, [&](auto tag, const auto&) {
            using scalar_type = typename decltype(tag)::type;
            return sytrd_band_to_tridiagonal_impl<scalar_type>(ab, uplo, kd, block_size, false, py::dict(), backend,
                                                               device_name);
        });
    });

    module.def("_sytrd_band_reduction", [](const DenseMatrix& ab, const std::string& uplo_name, int32_t kd,
                                            const py::dict& options, const std::string& backend_name,
                                            const py::object& device_name_obj) {
        const Uplo uplo = parse_uplo(uplo_name);
        const Backend backend = parse_backend(backend_name);
        const auto device_name = optional_string_from_obj(device_name_obj);
        return visit_dense(ab, [&](auto tag, const auto&) {
            using scalar_type = typename decltype(tag)::type;
            return sytrd_band_to_tridiagonal_impl<scalar_type>(ab, uplo, kd, 0, true, options, backend, device_name);
        });
    });

    module.def("_syev_variant_support", [](const DenseMatrix& matrix, const std::string& uplo_name,
                                            const py::object& device_name_obj) {
        const Uplo uplo = parse_uplo(uplo_name);
        const auto device_name = optional_string_from_obj(device_name_obj);
        Queue queue = make_queue(device_name);
        namespace dispatch = batchlas::blas::dispatch;
        const dispatch::DeviceCaps caps = dispatch::query_caps(queue);
        return visit_dense(matrix, [&](auto tag, const auto& typed_matrix) {
            using scalar_type = typename decltype(tag)::type;
            const auto view = typed_matrix.view();
            py::dict out;
            out["device"] = caps.name;
            out["is_gpu"] = caps.is_gpu;
            out["max_sub_group"] = caps.max_sub_group;
            out["cta"] = dispatch::detail::syev_supports_cta<scalar_type>(caps, view);
            out["blocked"] = dispatch::detail::syev_supports_blocked<scalar_type>(caps, view, uplo);
            out["two_stage"] = dispatch::detail::syev_supports_two_stage<scalar_type>(caps, view, uplo);
            return out;
        });
    });
}

}  // namespace batchlas::python
