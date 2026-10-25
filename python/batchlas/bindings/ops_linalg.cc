// SPDX-License-Identifier: MIT
#include <python/batchlas/bindings/init.hh>
#include <python/batchlas/bindings/support.hh>

#include <batchlas/blas/linalg.hh>

// Bindings for the batchlas::linalg layer: the elementwise operations (top-level
// add/subtract/multiply/divide/axpby/scale) and the value-returning wrappers behind
// the batchlas.linalg submodule (python/batchlas/linalg.py). Every `_linalg_*` call
// runs the linalg:: wrapper itself, waits, and returns NumPy arrays with the batch
// leading; linalg.py restores the caller's batch shape. Elementwise ops take no
// backend: they are SYCL kernels, so only the device selects anything.
namespace batchlas::python {

namespace {

template <typename T>
const DenseMatrixT<T>& typed(const DenseMatrix& wrapper) {
    return std::get<DenseMatrixT<T>>(wrapper.storage);
}

template <typename R>
py::object values_to_python(const UnifiedVector<R>& values, int per_item, int batch_size) {
    return dense_vector_to_python(wrap_vector_copy(values, per_item, batch_size));
}

template <typename F>
py::object visit_real_only(const DenseMatrix& a, const char* what, F&& fn) {
    return visit_dense(a, [&](auto tag, const auto&) -> py::object {
        using T = typename decltype(tag)::type;
        if constexpr (!RealScalar<T>) {
            return not_implemented<py::object>(what);
        } else {
            return fn(tag);
        }
    });
}

template <typename T, linalg::BinaryOp Op>
DenseMatrix elementwise_impl(const DenseMatrix& a_wrapper,
                             const DenseMatrix& b_wrapper,
                             const std::optional<std::string>& device_name) {
    const auto& a = typed<T>(a_wrapper);
    const auto& b = typed<T>(b_wrapper);
    DenseMatrixT<T> c(a.rows(), a.cols(), a.batch_size());
    Queue& queue = acquire_queue(device_name, Backend::Auto);
    (void)linalg::elementwise_into<T, Op>(queue, a.view(), b.view(), c.view());
    queue.wait();
    return wrap_dense(std::move(c));
}

template <linalg::BinaryOp Op>
DenseMatrix elementwise_dispatch(const DenseMatrix& a,
                                 const DenseMatrix& b,
                                 const std::optional<std::string>& device_name) {
    ensure_same_dtype(a, b, "matrix dtypes must match");
    return visit_dense(a, [&](auto tag, const auto&) {
        using scalar_type = typename decltype(tag)::type;
        return elementwise_impl<scalar_type, Op>(a, b, device_name);
    });
}

template <typename T>
DenseMatrix axpby_impl(const DenseMatrix& a_wrapper,
                       const DenseMatrix& b_wrapper,
                       const py::object& alpha_object,
                       const py::object& beta_object,
                       const std::optional<std::string>& device_name) {
    const auto& a = typed<T>(a_wrapper);
    const auto& b = typed<T>(b_wrapper);
    DenseMatrixT<T> c(a.rows(), a.cols(), a.batch_size());
    Queue& queue = acquire_queue(device_name, Backend::Auto);
    (void)linalg::axpby_into<T>(queue, scalar_from_object<T>(alpha_object), a.view(),
                          scalar_from_object<T>(beta_object), b.view(), c.view());
    queue.wait();
    return wrap_dense(std::move(c));
}

// linalg::scale is in place, which cannot reach the caller's NumPy array (it was copied
// on the way in), so this returns alpha * A instead.
template <typename T>
DenseMatrix scale_impl(const DenseMatrix& a_wrapper,
                       const py::object& alpha_object,
                       const std::optional<std::string>& device_name) {
    const auto& a = typed<T>(a_wrapper);
    Queue& queue = acquire_queue(device_name, Backend::Auto);
    auto c = linalg::scaled<T>(queue, a.view(), scalar_from_object<T>(alpha_object));
    queue.wait();
    return wrap_dense(std::move(c));
}

template <typename T>
py::object svd_impl(const DenseMatrixT<T>& a,
                    SvdVectors jobu,
                    SvdVectors jobvh,
                    std::optional<Uplo> hermitian_uplo,
                    Queue& queue) {
    using real_type = typename base_type<T>::type;
    const int batch = a.batch_size();
    const int k = std::min(a.rows(), a.cols());
    if (jobu != SvdVectors::None || jobvh != SvdVectors::None) {
        auto r = linalg::svd(queue, a.view(),
                             GesvdOptions{.jobu = jobu, .jobvh = jobvh, .hermitian_uplo = hermitian_uplo});
        queue.wait();
        py::object u = jobu == SvdVectors::None ? py::object(py::none()) : py::object(dense_matrix_to_numpy_t(r.U));
        py::object vh = jobvh == SvdVectors::None ? py::object(py::none()) : py::object(dense_matrix_to_numpy_t(r.Vh));
        return py::make_tuple(u, values_to_python(r.values, k, batch), vh, info_to_python(r.info, batch));
    }
    // linalg::svd rejects a values-only call and names gesvd as the spelling for it.
    if (hermitian_uplo && a.rows() != a.cols()) {
        throw py::value_error("svd: hermitian_uplo needs square matrices");
    }
    DenseMatrixT<T> work = a.clone();
    UnifiedVector<real_type> s(static_cast<std::size_t>(k) * static_cast<std::size_t>(batch));
    UnifiedVector<int32_t> info(static_cast<std::size_t>(batch));
    const MatrixView<T, MatrixFormat::Dense> none;
    if (hermitian_uplo) {
        (void)work.hermitize(queue, *hermitian_uplo);
        auto lease = queue.workspace(batchlas::gesvd_buffer_size(queue, work.view(), s.to_span(), none, none,
                                                                 jobu, jobvh, *hermitian_uplo));
        (void)batchlas::gesvd(queue, work.view(), s.to_span(), none, none, jobu, jobvh, *hermitian_uplo,
                        lease.span(), info.to_span());
        queue.wait();
    } else {
        auto lease = queue.workspace(
            batchlas::gesvd_buffer_size(queue, work.view(), s.to_span(), none, none, jobu, jobvh));
        (void)batchlas::gesvd(queue, work.view(), s.to_span(), none, none, jobu, jobvh, lease.span(), info.to_span());
        queue.wait();
    }
    return py::make_tuple(py::none(), values_to_python(s, k, batch), py::none(), info_to_python(info, batch));
}

// Every binding below: (dtype-dispatched) linalg call on the cached queue, wait, convert.
template <typename F>
py::object with_queue(const DenseMatrix& a, const std::string& backend_name, const py::object& device_obj,
                      F&& fn) {
    Queue& queue = acquire_queue(optional_string_from_obj(device_obj), parse_backend(backend_name));
    return visit_dense(a, [&](auto tag, const auto&) -> py::object { return fn(tag, queue); });
}

}  // namespace

void init_linalg_ops(py::module_& module) {
    const auto bind_elementwise = [&module](const char* name, auto op_tag) {
        constexpr linalg::BinaryOp Op = decltype(op_tag)::value;
        module.def(name, [](const DenseMatrix& a, const DenseMatrix& b,
                            const py::object& device_name_obj) {
            return elementwise_dispatch<Op>(a, b, optional_string_from_obj(device_name_obj));
        });
    };

    bind_elementwise("_elementwise_add",
                     std::integral_constant<linalg::BinaryOp, linalg::BinaryOp::Add>{});
    bind_elementwise("_elementwise_subtract",
                     std::integral_constant<linalg::BinaryOp, linalg::BinaryOp::Subtract>{});
    bind_elementwise("_elementwise_multiply",
                     std::integral_constant<linalg::BinaryOp, linalg::BinaryOp::Multiply>{});
    bind_elementwise("_elementwise_divide",
                     std::integral_constant<linalg::BinaryOp, linalg::BinaryOp::Divide>{});

    module.def("_axpby", [](const DenseMatrix& a, const DenseMatrix& b,
                            const py::object& alpha, const py::object& beta,
                            const py::object& device_name_obj) {
        ensure_same_dtype(a, b, "matrix dtypes must match");
        const auto device_name = optional_string_from_obj(device_name_obj);
        return visit_dense(a, [&](auto tag, const auto&) {
            using scalar_type = typename decltype(tag)::type;
            return axpby_impl<scalar_type>(a, b, alpha, beta, device_name);
        });
    });

    module.def("_scale", [](const DenseMatrix& a, const py::object& alpha,
                            const py::object& device_name_obj) {
        const auto device_name = optional_string_from_obj(device_name_obj);
        return visit_dense(a, [&](auto tag, const auto&) {
            using scalar_type = typename decltype(tag)::type;
            return scale_impl<scalar_type>(a, alpha, device_name);
        });
    });

    module.def("_linalg_matmul", [](const DenseMatrix& a, const DenseMatrix& b, const py::object& alpha,
                                    const std::string& trans_a, const std::string& trans_b,
                                    const std::string& precision, const std::string& backend,
                                    const py::object& device) {
        ensure_same_dtype(a, b, "matrix dtypes must match");
        return with_queue(a, backend, device, [&](auto tag, Queue& queue) -> py::object {
            using T = typename decltype(tag)::type;
            auto c = linalg::matmul(queue, typed<T>(a).view(), typed<T>(b).view(),
                                    linalg::MatmulOptions<T>{.alpha = scalar_from_object<T>(alpha),
                                                             .transA = parse_transpose(trans_a),
                                                             .transB = parse_transpose(trans_b),
                                                             .precision = parse_compute_precision(precision)});
            queue.wait();
            return dense_matrix_to_numpy_t(c);
        });
    });

    module.def("_linalg_matvec", [](const DenseMatrix& a, const DenseVector& x, const std::string& trans,
                                    const std::string& backend, const py::object& device) {
        ensure_same_dtype(a, x, "matrix and vector dtypes must match");
        const Transpose op = parse_transpose(trans);
        if (a.batch_size() != x.batch_size()) throw py::value_error("matvec: batch sizes of a and x differ");
        if (x.size() != (op == Transpose::NoTrans ? a.cols() : a.rows())) {
            throw py::value_error("matvec: x has the wrong length for op(a)");
        }
        return with_queue(a, backend, device, [&](auto tag, Queue& queue) -> py::object {
            using T = typename decltype(tag)::type;
            auto y = linalg::matvec(queue, typed<T>(a).view(), std::get<Vector<T>>(x.storage), op);
            queue.wait();
            return dense_vector_to_numpy_t(y);
        });
    });

    module.def("_linalg_gram", [](const DenseMatrix& a, const std::string& trans, const std::string& backend,
                                  const py::object& device) {
        return with_queue(a, backend, device, [&](auto tag, Queue& queue) -> py::object {
            using T = typename decltype(tag)::type;
            auto c = linalg::gram(queue, typed<T>(a).view(), parse_transpose(trans));
            queue.wait();
            return dense_matrix_to_numpy_t(c);
        });
    });

    module.def("_linalg_solve_triangular", [](const DenseMatrix& a, const DenseMatrix& b,
                                              const py::object& alpha, const std::string& side,
                                              const std::string& uplo, const std::string& trans,
                                              const std::string& diag, const std::string& backend,
                                              const py::object& device) {
        ensure_same_dtype(a, b, "matrix dtypes must match");
        return with_queue(a, backend, device, [&](auto tag, Queue& queue) -> py::object {
            using T = typename decltype(tag)::type;
            auto x = linalg::solve_triangular(queue, typed<T>(a).view(), typed<T>(b).view(),
                                              TrsmOptions<T>{.alpha = scalar_from_object<T>(alpha),
                                                             .side = parse_side(side),
                                                             .uplo = parse_uplo(uplo),
                                                             .trans = parse_transpose(trans),
                                                             .diag = parse_diag(diag)});
            queue.wait();
            return dense_matrix_to_numpy_t(x);
        });
    });

    module.def("_linalg_cholesky", [](const DenseMatrix& a, const std::string& uplo, const std::string& backend,
                                      const py::object& device) {
        return with_queue(a, backend, device, [&](auto tag, Queue& queue) -> py::object {
            using T = typename decltype(tag)::type;
            auto r = linalg::cholesky(queue, typed<T>(a).view(), parse_uplo(uplo));
            queue.wait();
            return py::make_tuple(dense_matrix_to_numpy_t(r.factor), info_to_python(r.info, a.batch_size()));
        });
    });

    module.def("_linalg_eigvalsh", [](const DenseMatrix& a, const std::string& uplo, const std::string& backend,
                                      const py::object& device) {
        return with_queue(a, backend, device, [&](auto tag, Queue& queue) -> py::object {
            using T = typename decltype(tag)::type;
            auto w = linalg::eigvalsh(queue, typed<T>(a).view(), parse_uplo(uplo));
            return values_to_python(w, a.rows(), a.batch_size());
        });
    });

    module.def("_linalg_eigh", [](const DenseMatrix& a, const std::string& uplo, const std::string& backend,
                                  const py::object& device) {
        return with_queue(a, backend, device, [&](auto tag, Queue& queue) -> py::object {
            using T = typename decltype(tag)::type;
            auto r = linalg::eigh(queue, typed<T>(a).view(), parse_uplo(uplo));
            queue.wait();
            return py::make_tuple(values_to_python(r.values, a.rows(), a.batch_size()),
                                  dense_matrix_to_numpy_t(r.vectors), info_to_python(r.info, a.batch_size()));
        });
    });

    // Returns (values, vectors or None, count, info); `values` has k slots per item and the
    // ones past count[b] are unwritten, which linalg.py masks.
    module.def("_linalg_eigh_subset", [](const DenseMatrix& a, const py::dict& subset, bool vectors,
                                         const std::string& backend, const py::object& device) {
        return with_queue(a, backend, device, [&](auto tag, Queue& queue) -> py::object {
            using T = typename decltype(tag)::type;
            const auto params = parse_syevx_params<T>(subset);
            auto r = vectors ? linalg::eigh(queue, typed<T>(a).view(), params)
                             : linalg::eigvalsh(queue, typed<T>(a).view(), params);
            queue.wait();
            const int batch = a.batch_size();
            const int k = static_cast<int>(r.values.size()) / std::max(batch, 1);
            return py::make_tuple(values_to_python(r.values, k, batch),
                                  vectors ? py::object(dense_matrix_to_numpy_t(r.vectors)) : py::none(),
                                  info_to_python(r.count, batch), info_to_python(r.info, batch));
        });
    });

    module.def("_linalg_solve", [](const DenseMatrix& a, const DenseMatrix& b, const std::string& trans,
                                   const std::string& backend, const py::object& device) {
        ensure_same_dtype(a, b, "matrix dtypes must match");
        return with_queue(a, backend, device, [&](auto tag, Queue& queue) -> py::object {
            using T = typename decltype(tag)::type;
            auto r = linalg::solve(queue, typed<T>(a).view(), typed<T>(b).view(), parse_transpose(trans));
            return py::make_tuple(dense_matrix_to_numpy_t(r.x), info_to_python(r.info, a.batch_size()));
        });
    });

    module.def("_linalg_solve_spd", [](const DenseMatrix& a, const DenseMatrix& b, const std::string& uplo,
                                       const std::string& backend, const py::object& device) {
        ensure_same_dtype(a, b, "matrix dtypes must match");
        return with_queue(a, backend, device, [&](auto tag, Queue& queue) -> py::object {
            using T = typename decltype(tag)::type;
            auto r = linalg::solve_spd(queue, typed<T>(a).view(), typed<T>(b).view(), parse_uplo(uplo));
            return py::make_tuple(dense_matrix_to_numpy_t(r.x), info_to_python(r.info, a.batch_size()));
        });
    });

    module.def("_linalg_inv", [](const DenseMatrix& a, const std::string& backend, const py::object& device) {
        return with_queue(a, backend, device, [&](auto tag, Queue& queue) -> py::object {
            using T = typename decltype(tag)::type;
            auto r = linalg::inv(queue, typed<T>(a).view());
            return py::make_tuple(dense_matrix_to_numpy_t(r.inverse), info_to_python(r.info, a.batch_size()));
        });
    });

    module.def("_linalg_svd", [](const DenseMatrix& a, const std::string& jobu, const std::string& jobvh,
                                 const py::object& hermitian_uplo, const std::string& backend,
                                 const py::object& device) {
        const auto uplo_name = optional_string_from_obj(hermitian_uplo);
        const std::optional<Uplo> uplo = uplo_name ? std::optional<Uplo>(parse_uplo(*uplo_name)) : std::nullopt;
        return with_queue(a, backend, device, [&](auto tag, Queue& queue) -> py::object {
            using T = typename decltype(tag)::type;
            return svd_impl<T>(typed<T>(a), parse_svd_vectors(jobu), parse_svd_vectors(jobvh), uplo, queue);
        });
    });

    module.def("_linalg_lu", [](const DenseMatrix& a, const std::string& backend, const py::object& device) {
        return with_queue(a, backend, device, [&](auto tag, Queue& queue) -> py::object {
            using T = typename decltype(tag)::type;
            auto r = linalg::lu(queue, typed<T>(a).view());
            queue.wait();
            // linalg::lu runs getrf on this queue, so the pivots are in getrf's packing.
            return py::make_tuple(dense_matrix_to_numpy_t(r.factors),
                                  pivots_to_numpy(r.pivots, pivots_packed_int32(queue), a.rows(), a.batch_size()),
                                  info_to_python(r.info, a.batch_size()));
        });
    });

    const auto bind_mask = [&module](const char* name, bool upper) {
        module.def(name, [upper](const DenseMatrix& a, std::int64_t k, const py::object& device) {
            return with_queue(a, "auto", device, [&](auto tag, Queue& queue) -> py::object {
                using T = typename decltype(tag)::type;
                auto c = upper ? linalg::triu(queue, typed<T>(a).view(), k)
                               : linalg::tril(queue, typed<T>(a).view(), k);
                queue.wait();
                return dense_matrix_to_numpy_t(c);
            });
        });
    };
    bind_mask("_linalg_triu", true);
    bind_mask("_linalg_tril", false);

    module.def("_linalg_transpose", [](const DenseMatrix& a, const py::object& device) {
        Queue& queue = acquire_queue(optional_string_from_obj(device), Backend::Auto);
        return visit_real_only(a, "linalg.transpose is real-only (nothing conjugates)", [&](auto tag) -> py::object {
            using T = typename decltype(tag)::type;
            auto c = linalg::transpose(queue, typed<T>(a).view());
            queue.wait();
            return dense_matrix_to_numpy_t(c);
        });
    });

    module.def("_linalg_norm", [](const DenseMatrix& a, const std::string& norm_type, const py::object& device) {
        return with_queue(a, "auto", device, [&](auto tag, Queue& queue) -> py::object {
            using T = typename decltype(tag)::type;
            auto v = linalg::norm(queue, typed<T>(a).view(), parse_norm_type(norm_type));
            return values_to_python(v, a.batch_size(), 1);
        });
    });

    module.def("_linalg_cond", [](const DenseMatrix& a, const std::string& norm_type, const std::string& backend,
                                  const py::object& device) {
        Queue& queue = acquire_queue(optional_string_from_obj(device), parse_backend(backend));
        return visit_real_only(a, "linalg.cond is real-only", [&](auto tag) -> py::object {
            using T = typename decltype(tag)::type;
            auto v = linalg::cond(queue, typed<T>(a).view(), parse_norm_type(norm_type));
            return values_to_python(v, a.batch_size(), 1);
        });
    });
}

}  // namespace batchlas::python
