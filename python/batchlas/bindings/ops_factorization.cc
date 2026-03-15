#include "init.hh"
#include "support.hh"

namespace batchlas::python {

namespace {

py::array_t<int64_t> pivots_to_numpy(const UnifiedVector<int64_t>& pivots, int pivots_per_batch, int batch_size) {
    if (batch_size == 1) {
        py::array_t<int64_t> out({pivots_per_batch});
        auto view = out.mutable_unchecked<1>();
        for (int index = 0; index < pivots_per_batch; ++index) {
            view(index) = pivots[static_cast<std::size_t>(index)];
        }
        return out;
    }

    py::array_t<int64_t> out({batch_size, pivots_per_batch});
    auto view = out.mutable_unchecked<2>();
    for (int batch = 0; batch < batch_size; ++batch) {
        for (int index = 0; index < pivots_per_batch; ++index) {
            view(batch, index) = pivots[static_cast<std::size_t>(batch * pivots_per_batch + index)];
        }
    }
    return out;
}

template <typename T>
DenseMatrix dense_potrf_impl(const DenseMatrix& a_wrapper,
                             Uplo uplo,
                             Backend backend,
                             const std::optional<std::string>& device_name) {
    DenseMatrixT<T> out = std::get<DenseMatrixT<T>>(a_wrapper.storage).clone();
    Queue queue = make_queue(device_name);
    run_backend_with_workspace(
        backend, queue,
        [&](auto backend_tag) {
            constexpr Backend B = decltype(backend_tag)::value;
            return batchlas::potrf_buffer_size<B, T>(queue, out.view(), uplo);
        },
        [&](auto backend_tag, Span<std::byte> workspace) {
            constexpr Backend B = decltype(backend_tag)::value;
            batchlas::potrf<B, T>(queue, out.view(), uplo, workspace);
        });
    queue.wait();
    return wrap_dense(std::move(out));
}

template <typename T>
py::tuple dense_getrf_impl(const DenseMatrix& a_wrapper,
                           Backend backend,
                           const std::optional<std::string>& device_name) {
    DenseMatrixT<T> out = std::get<DenseMatrixT<T>>(a_wrapper.storage).clone();
    const int pivots_per_batch = out.rows();
    UnifiedVector<int64_t> pivots(static_cast<std::size_t>(pivots_per_batch * out.batch_size()));
    Queue queue = make_queue(device_name);
    run_backend_with_workspace(
        backend, queue,
        [&](auto backend_tag) {
            constexpr Backend B = decltype(backend_tag)::value;
            return batchlas::getrf_buffer_size<B, T>(queue, out.view());
        },
        [&](auto backend_tag, Span<std::byte> workspace) {
            constexpr Backend B = decltype(backend_tag)::value;
            batchlas::getrf<B, T>(queue, out.view(), pivots.to_span(), workspace);
        });
    queue.wait();
    return py::make_tuple(wrap_dense(std::move(out)),
                          pivots_to_numpy(pivots, pivots_per_batch, std::get<DenseMatrixT<T>>(a_wrapper.storage).batch_size()));
}

template <typename T>
DenseMatrix dense_getrs_impl(const DenseMatrix& lu_wrapper,
                             const DenseMatrix& b_wrapper,
                             const py::array_t<int64_t, py::array::forcecast>& pivots_array,
                             Transpose trans_a,
                             Backend backend,
                             const std::optional<std::string>& device_name) {
    ensure_same_dtype(lu_wrapper, b_wrapper, "matrix dtypes must match");
    const auto& lu = std::get<DenseMatrixT<T>>(lu_wrapper.storage);
    DenseMatrixT<T> out = std::get<DenseMatrixT<T>>(b_wrapper.storage).clone();

    const py::buffer_info pivots_info = pivots_array.request();
    const int batch_size = lu.batch_size();
    const int pivots_per_batch = lu.rows();
    if ((pivots_info.ndim == 1 && batch_size != 1) ||
        (pivots_info.ndim == 1 && static_cast<int>(pivots_info.shape[0]) != pivots_per_batch) ||
        (pivots_info.ndim == 2 && (static_cast<int>(pivots_info.shape[0]) != batch_size ||
                                   static_cast<int>(pivots_info.shape[1]) != pivots_per_batch))) {
        throw py::value_error("pivots shape does not match LU factors");
    }

    UnifiedVector<int64_t> pivots(static_cast<std::size_t>(pivots_per_batch * batch_size));
    if (pivots_info.ndim == 1) {
        auto view = pivots_array.unchecked<1>();
        for (int index = 0; index < pivots_per_batch; ++index) {
            pivots[static_cast<std::size_t>(index)] = view(index);
        }
    } else {
        auto view = pivots_array.unchecked<2>();
        for (int batch = 0; batch < batch_size; ++batch) {
            for (int index = 0; index < pivots_per_batch; ++index) {
                pivots[static_cast<std::size_t>(batch * pivots_per_batch + index)] = view(batch, index);
            }
        }
    }

    Queue queue = make_queue(device_name);
    run_backend_with_workspace(
        backend, queue,
        [&](auto backend_tag) {
            constexpr Backend B = decltype(backend_tag)::value;
            return batchlas::getrs_buffer_size<B, T>(queue, lu.view(), out.view(), trans_a);
        },
        [&](auto backend_tag, Span<std::byte> workspace) {
            constexpr Backend B = decltype(backend_tag)::value;
            batchlas::getrs<B, T>(queue, lu.view(), out.view(), trans_a, pivots.to_span(), workspace);
        });
    queue.wait();
    return wrap_dense(std::move(out));
}

template <typename T>
DenseMatrix dense_getri_impl(const DenseMatrix& lu_wrapper,
                             const py::array_t<int64_t, py::array::forcecast>& pivots_array,
                             Backend backend,
                             const std::optional<std::string>& device_name) {
    const auto& lu = std::get<DenseMatrixT<T>>(lu_wrapper.storage);
    DenseMatrixT<T> out(lu.rows(), lu.cols(), lu.batch_size());
    const py::buffer_info pivots_info = pivots_array.request();
    const int pivots_per_batch = lu.rows();
    const int batch_size = lu.batch_size();
    if ((pivots_info.ndim == 1 && batch_size != 1) ||
        (pivots_info.ndim == 1 && static_cast<int>(pivots_info.shape[0]) != pivots_per_batch) ||
        (pivots_info.ndim == 2 && (static_cast<int>(pivots_info.shape[0]) != batch_size ||
                                   static_cast<int>(pivots_info.shape[1]) != pivots_per_batch))) {
        throw py::value_error("pivots shape does not match LU factors");
    }

    UnifiedVector<int64_t> pivots(static_cast<std::size_t>(pivots_per_batch * batch_size));
    if (pivots_info.ndim == 1) {
        auto view = pivots_array.unchecked<1>();
        for (int index = 0; index < pivots_per_batch; ++index) {
            pivots[static_cast<std::size_t>(index)] = view(index);
        }
    } else {
        auto view = pivots_array.unchecked<2>();
        for (int batch = 0; batch < batch_size; ++batch) {
            for (int index = 0; index < pivots_per_batch; ++index) {
                pivots[static_cast<std::size_t>(batch * pivots_per_batch + index)] = view(batch, index);
            }
        }
    }

    Queue queue = make_queue(device_name);
    run_backend_with_workspace(
        backend, queue,
        [&](auto backend_tag) {
            constexpr Backend B = decltype(backend_tag)::value;
            return batchlas::getri_buffer_size<B, T>(queue, lu.view());
        },
        [&](auto backend_tag, Span<std::byte> workspace) {
            constexpr Backend B = decltype(backend_tag)::value;
            batchlas::getri<B, T>(queue, lu.view(), out.view(), pivots.to_span(), workspace);
        });
    queue.wait();
    return wrap_dense(std::move(out));
}

template <typename T>
py::tuple dense_geqrf_impl(const DenseMatrix& a_wrapper,
                           Backend backend,
                           const std::optional<std::string>& device_name) {
    DenseMatrixT<T> out = std::get<DenseMatrixT<T>>(a_wrapper.storage).clone();
    Vector<T> tau(std::min(out.rows(), out.cols()), out.batch_size());
    Queue queue = make_queue(device_name);
    run_backend_with_workspace(
        backend, queue,
        [&](auto backend_tag) {
            constexpr Backend B = decltype(backend_tag)::value;
            return batchlas::geqrf_buffer_size<B, T>(queue, out.view(), tau.data());
        },
        [&](auto backend_tag, Span<std::byte> workspace) {
            constexpr Backend B = decltype(backend_tag)::value;
            batchlas::geqrf<B, T>(queue, out.view(), tau.data(), workspace);
        });
    queue.wait();
    return py::make_tuple(wrap_dense(std::move(out)), wrap_vector(std::move(tau)));
}

template <typename T>
DenseMatrix dense_orgqr_impl(const DenseMatrix& qr_wrapper,
                             const DenseVector& tau_wrapper,
                             Backend backend,
                             const std::optional<std::string>& device_name) {
    ensure_same_dtype(qr_wrapper, tau_wrapper, "QR factors and tau dtypes must match");
    DenseMatrixT<T> out = std::get<DenseMatrixT<T>>(qr_wrapper.storage).clone();
    const auto& tau = std::get<Vector<T>>(tau_wrapper.storage);
    Queue queue = make_queue(device_name);
    run_backend_with_workspace(
        backend, queue,
        [&](auto backend_tag) {
            constexpr Backend B = decltype(backend_tag)::value;
            return batchlas::orgqr_buffer_size<B, T>(queue, out.view(), tau.data());
        },
        [&](auto backend_tag, Span<std::byte> workspace) {
            constexpr Backend B = decltype(backend_tag)::value;
            batchlas::orgqr<B, T>(queue, out.view(), tau.data(), workspace);
        });
    queue.wait();
    return wrap_dense(std::move(out));
}

template <typename T>
DenseMatrix dense_ormqr_impl(const DenseMatrix& qr_wrapper,
                             const DenseMatrix& c_wrapper,
                             const DenseVector& tau_wrapper,
                             Side side,
                             Transpose trans,
                             Backend backend,
                             const std::optional<std::string>& device_name) {
    ensure_same_dtype(qr_wrapper, c_wrapper, "QR factors and C dtypes must match");
    ensure_same_dtype(qr_wrapper, tau_wrapper, "QR factors and tau dtypes must match");
    const auto& qr = std::get<DenseMatrixT<T>>(qr_wrapper.storage);
    DenseMatrixT<T> out = std::get<DenseMatrixT<T>>(c_wrapper.storage).clone();
    const auto& tau = std::get<Vector<T>>(tau_wrapper.storage);
    Queue queue = make_queue(device_name);
    run_backend_with_workspace(
        backend, queue,
        [&](auto backend_tag) {
            constexpr Backend B = decltype(backend_tag)::value;
            return batchlas::ormqr_buffer_size<B, T>(queue, qr.view(), out.view(), side, trans, tau.data());
        },
        [&](auto backend_tag, Span<std::byte> workspace) {
            constexpr Backend B = decltype(backend_tag)::value;
            batchlas::ormqr<B, T>(queue, qr.view(), out.view(), side, trans, tau.data(), workspace);
        });
    queue.wait();
    return wrap_dense(std::move(out));
}

template <typename T>
py::object dense_gesvd_impl(const DenseMatrix& a_wrapper,
                            bool compute_vectors,
                            Backend backend,
                            const std::optional<std::string>& device_name,
                            bool blocked) {
    DenseMatrixT<T> out = std::get<DenseMatrixT<T>>(a_wrapper.storage).clone();
    const int n = out.rows();
    Vector<typename base_type<T>::type> singular_values(n, out.batch_size());
    DenseMatrixT<T> u = compute_vectors ? DenseMatrixT<T>(n, n, out.batch_size()) : DenseMatrixT<T>(1, 1, out.batch_size());
    DenseMatrixT<T> vh = compute_vectors ? DenseMatrixT<T>(n, n, out.batch_size()) : DenseMatrixT<T>(1, 1, out.batch_size());
    const SvdVectors job = compute_vectors ? SvdVectors::All : SvdVectors::None;
    Queue queue = make_queue(device_name);
    run_backend_with_workspace(
        backend, queue,
        [&](auto backend_tag) {
            constexpr Backend B = decltype(backend_tag)::value;
            if (blocked) {
                return batchlas::gesvd_blocked_buffer_size<B, T>(queue, out.view(), singular_values.data(), u.view(),
                                                                 vh.view(), job, job);
            }
            return batchlas::gesvd_buffer_size<B, T>(queue, out.view(), singular_values.data(), u.view(), vh.view(),
                                                     job, job);
        },
        [&](auto backend_tag, Span<std::byte> workspace) {
            constexpr Backend B = decltype(backend_tag)::value;
            if (blocked) {
                batchlas::gesvd_blocked<B, T>(queue, out.view(), singular_values.data(), u.view(), vh.view(), job,
                                              job, workspace);
            } else {
                batchlas::gesvd<B, T>(queue, out.view(), singular_values.data(), u.view(), vh.view(), job, job,
                                      workspace);
            }
        });
    queue.wait();

    if (!compute_vectors) {
        return dense_vector_to_python(wrap_vector(std::move(singular_values)));
    }
    return py::make_tuple(wrap_dense(std::move(u)),
                          wrap_vector(std::move(singular_values)),
                          wrap_dense(std::move(vh)));
}

template <typename T>
py::tuple dense_gebrd_impl(const DenseMatrix& a_wrapper,
                           Backend backend,
                           const std::optional<std::string>& device_name) {
    DenseMatrixT<T> out = std::get<DenseMatrixT<T>>(a_wrapper.storage).clone();
    const int n = out.rows();
    Vector<typename base_type<T>::type> d(n, out.batch_size());
    Vector<typename base_type<T>::type> e(std::max(0, n - 1), out.batch_size());
    Vector<T> tauq(std::max(0, n - 1), out.batch_size());
    Vector<T> taup(std::max(0, n - 1), out.batch_size());
    Queue queue = make_queue(device_name);
    visit_backend(backend, [&](auto backend_tag) {
        constexpr Backend B = decltype(backend_tag)::value;
        batchlas::gebrd_unblocked<B, T>(queue, out.view(), VectorView<typename base_type<T>::type>(d),
                                        VectorView<typename base_type<T>::type>(e), VectorView<T>(tauq),
                                        VectorView<T>(taup));
    });
    queue.wait();
    return py::make_tuple(wrap_dense(std::move(out)),
                          wrap_vector(std::move(d)),
                          wrap_vector(std::move(e)),
                          wrap_vector(std::move(tauq)),
                          wrap_vector(std::move(taup)));
}

template <typename T>
DenseVector dense_bdsqr_impl(const DenseVector& d_wrapper,
                             const DenseVector& e_wrapper,
                             bool sort_desc,
                             Backend backend,
                             const std::optional<std::string>& device_name) {
    ensure_same_dtype(d_wrapper, e_wrapper, "d and e dtypes must match");
    const auto& d = std::get<Vector<T>>(d_wrapper.storage);
    const auto& e = std::get<Vector<T>>(e_wrapper.storage);
    Vector<T> singular_values(d.size(), d.batch_size());
    Queue queue = make_queue(device_name);
    const std::size_t workspace_size =
        batchlas::bdsqr_buffer_size(queue, VectorView<T>(d), VectorView<T>(e), singular_values.data());
    UnifiedVector<std::byte> workspace(workspace_size);
    visit_backend(backend, [&](auto backend_tag) {
        constexpr Backend B = decltype(backend_tag)::value;
        batchlas::bdsqr<B, T>(queue, VectorView<T>(d), VectorView<T>(e), singular_values.data(), workspace.to_span(),
                              sort_desc);
    });
    queue.wait();
    return wrap_vector(std::move(singular_values));
}

template <typename T>
DenseMatrix dense_ormbr_impl(const DenseMatrix& a_wrapper,
                             const DenseVector& tau_wrapper,
                             const DenseMatrix& c_wrapper,
                             char vect,
                             Side side,
                             Transpose trans,
                             int32_t block_size,
                             Backend backend,
                             const std::optional<std::string>& device_name) {
    ensure_same_dtype(a_wrapper, tau_wrapper, "A and tau dtypes must match");
    ensure_same_dtype(a_wrapper, c_wrapper, "A and C dtypes must match");
    const auto& a = std::get<DenseMatrixT<T>>(a_wrapper.storage);
    const auto& tau = std::get<Vector<T>>(tau_wrapper.storage);
    DenseMatrixT<T> out = std::get<DenseMatrixT<T>>(c_wrapper.storage).clone();
    Queue queue = make_queue(device_name);
    run_backend_with_workspace(
        backend, queue,
        [&](auto backend_tag) {
            constexpr Backend B = decltype(backend_tag)::value;
            return batchlas::ormbr_buffer_size<B, T>(queue, a.view(), VectorView<T>(tau), out.view(), vect, side,
                                                     trans, block_size);
        },
        [&](auto backend_tag, Span<std::byte> workspace) {
            constexpr Backend B = decltype(backend_tag)::value;
            batchlas::ormbr<B, T>(queue, a.view(), VectorView<T>(tau), out.view(), vect, side, trans, workspace,
                                  block_size);
        });
    queue.wait();
    return wrap_dense(std::move(out));
}

}  // namespace

void init_factorization_ops(py::module_& module) {
    module.def("_potrf", [](const DenseMatrix& a,
                             const std::string& uplo_name,
                             const std::string& backend_name,
                             const py::object& device_name_obj) {
        const Uplo uplo = parse_uplo(uplo_name);
        const Backend backend = parse_backend(backend_name);
        const auto device_name = optional_string_from_obj(device_name_obj);
        return visit_dense(a, [&](auto tag, const auto&) {
            using scalar_type = typename decltype(tag)::type;
            return dense_potrf_impl<scalar_type>(a, uplo, backend, device_name);
        });
    });

    module.def("_getrf", [](const DenseMatrix& a,
                             const std::string& backend_name,
                             const py::object& device_name_obj) {
        const Backend backend = parse_backend(backend_name);
        const auto device_name = optional_string_from_obj(device_name_obj);
        return visit_dense(a, [&](auto tag, const auto&) {
            using scalar_type = typename decltype(tag)::type;
            return dense_getrf_impl<scalar_type>(a, backend, device_name);
        });
    });

    module.def("_getrs", [](const DenseMatrix& lu,
                             const DenseMatrix& b,
                             const py::array_t<int64_t, py::array::forcecast>& pivots,
                             const std::string& trans_a_name,
                             const std::string& backend_name,
                             const py::object& device_name_obj) {
        const Transpose trans_a = parse_transpose(trans_a_name);
        const Backend backend = parse_backend(backend_name);
        const auto device_name = optional_string_from_obj(device_name_obj);
        return visit_dense(lu, [&](auto tag, const auto&) {
            using scalar_type = typename decltype(tag)::type;
            return dense_getrs_impl<scalar_type>(lu, b, pivots, trans_a, backend, device_name);
        });
    });

    module.def("_getri", [](const DenseMatrix& lu,
                             const py::array_t<int64_t, py::array::forcecast>& pivots,
                             const std::string& backend_name,
                             const py::object& device_name_obj) {
        const Backend backend = parse_backend(backend_name);
        const auto device_name = optional_string_from_obj(device_name_obj);
        return visit_dense(lu, [&](auto tag, const auto&) {
            using scalar_type = typename decltype(tag)::type;
            return dense_getri_impl<scalar_type>(lu, pivots, backend, device_name);
        });
    });

    module.def("_inv", [](const DenseMatrix& a,
                           const std::string& backend_name,
                           const py::object& device_name_obj) {
        const Backend backend = parse_backend(backend_name);
        const auto device_name = optional_string_from_obj(device_name_obj);
        return visit_dense(a, [&](auto tag, const auto& typed_a) {
            using scalar_type = typename decltype(tag)::type;
            Queue queue = make_queue(device_name);
            auto out = visit_backend(backend, [&](auto backend_tag) {
                constexpr Backend B = decltype(backend_tag)::value;
                return batchlas::inv<B, scalar_type>(queue, typed_a.view());
            });
            queue.wait();
            return wrap_dense(std::move(out));
        });
    });

    module.def("_geqrf", [](const DenseMatrix& a,
                             const std::string& backend_name,
                             const py::object& device_name_obj) {
        const Backend backend = parse_backend(backend_name);
        const auto device_name = optional_string_from_obj(device_name_obj);
        return visit_dense(a, [&](auto tag, const auto&) {
            using scalar_type = typename decltype(tag)::type;
            return dense_geqrf_impl<scalar_type>(a, backend, device_name);
        });
    });

    module.def("_orgqr", [](const DenseMatrix& qr,
                             const DenseVector& tau,
                             const std::string& backend_name,
                             const py::object& device_name_obj) {
        const Backend backend = parse_backend(backend_name);
        const auto device_name = optional_string_from_obj(device_name_obj);
        return visit_dense(qr, [&](auto tag, const auto&) {
            using scalar_type = typename decltype(tag)::type;
            return dense_orgqr_impl<scalar_type>(qr, tau, backend, device_name);
        });
    });

    module.def("_ormqr", [](const DenseMatrix& qr,
                             const DenseMatrix& c,
                             const DenseVector& tau,
                             const std::string& side_name,
                             const std::string& trans_name,
                             const std::string& backend_name,
                             const py::object& device_name_obj) {
        const Side side = parse_side(side_name);
        const Transpose trans = parse_transpose(trans_name);
        const Backend backend = parse_backend(backend_name);
        const auto device_name = optional_string_from_obj(device_name_obj);
        return visit_dense(qr, [&](auto tag, const auto&) {
            using scalar_type = typename decltype(tag)::type;
            return dense_ormqr_impl<scalar_type>(qr, c, tau, side, trans, backend, device_name);
        });
    });

    module.def("_gesvd", [](const DenseMatrix& a,
                             bool compute_vectors,
                             const std::string& backend_name,
                             const py::object& device_name_obj) {
        const Backend backend = parse_backend(backend_name);
        const auto device_name = optional_string_from_obj(device_name_obj);
        return visit_dense(a, [&](auto tag, const auto&) -> py::object {
            using scalar_type = typename decltype(tag)::type;
            return dense_gesvd_impl<scalar_type>(a, compute_vectors, backend, device_name, false);
        });
    });

    module.def("_gesvd_blocked", [](const DenseMatrix& a,
                                     bool compute_vectors,
                                     const std::string& backend_name,
                                     const py::object& device_name_obj) {
        const Backend backend = parse_backend(backend_name);
        const auto device_name = optional_string_from_obj(device_name_obj);
        return visit_dense(a, [&](auto tag, const auto&) -> py::object {
            using scalar_type = typename decltype(tag)::type;
            return dense_gesvd_impl<scalar_type>(a, compute_vectors, backend, device_name, true);
        });
    });

    module.def("_gebrd_unblocked", [](const DenseMatrix& a,
                                       const std::string& backend_name,
                                       const py::object& device_name_obj) {
        const Backend backend = parse_backend(backend_name);
        const auto device_name = optional_string_from_obj(device_name_obj);
        return visit_dense(a, [&](auto tag, const auto&) {
            using scalar_type = typename decltype(tag)::type;
            return dense_gebrd_impl<scalar_type>(a, backend, device_name);
        });
    });

    module.def("_bdsqr", [](const DenseVector& d,
                             const DenseVector& e,
                             bool sort_desc,
                             const std::string& backend_name,
                             const py::object& device_name_obj) {
        const Backend backend = parse_backend(backend_name);
        const auto device_name = optional_string_from_obj(device_name_obj);
        return visit_vector(d, [&](auto tag, const auto&) -> DenseVector {
            using scalar_type = typename decltype(tag)::type;
            if constexpr (!std::is_floating_point_v<scalar_type>) {
                throw_not_implemented("bdsqr only supports float32 and float64");
            } else {
                return dense_bdsqr_impl<scalar_type>(d, e, sort_desc, backend, device_name);
            }
        });
    });

    module.def("_ormbr", [](const DenseMatrix& a,
                             const DenseVector& tau,
                             const DenseMatrix& c,
                             const std::string& vect_name,
                             const std::string& side_name,
                             const std::string& trans_name,
                             int32_t block_size,
                             const std::string& backend_name,
                             const py::object& device_name_obj) {
        const Backend backend = parse_backend(backend_name);
        const auto device_name = optional_string_from_obj(device_name_obj);
        const Side side = parse_side(side_name);
        const Transpose trans = parse_transpose(trans_name);
        if (vect_name.size() != 1) {
            throw py::value_error("vect must be 'Q' or 'P'");
        }
        const char vect = static_cast<char>(std::toupper(static_cast<unsigned char>(vect_name[0])));
        return visit_dense(a, [&](auto tag, const auto&) {
            using scalar_type = typename decltype(tag)::type;
            return dense_ormbr_impl<scalar_type>(a, tau, c, vect, side, trans, block_size, backend, device_name);
        });
    });
}

}  // namespace batchlas::python
