#include "init.hh"
#include "support.hh"

#include <batchlas/blas/linalg.hh>

// Bindings for the batchlas::linalg convenience layer (P5).
//
// Only the elementwise operations are exposed. The value-returning wrappers in
// batchlas::linalg -- matmul, solve, cholesky, eigvalsh, eigh -- exist to give
// C++ callers the allocate-and-return shape that the Python bindings already
// had all along, so re-exporting them would add a second spelling of gemm,
// getrs, potrf and syev without adding capability. Elementwise arithmetic is
// genuinely new here: there was no way to express it from Python before.
//
// These take no backend argument. They are plain SYCL kernels rather than
// vendor-library calls, so there is nothing for a backend to select; the queue
// is still cached per device so the arithmetic runs on the device the caller
// asked for.
namespace batchlas::python {

namespace {

template <typename T, linalg::BinaryOp Op>
DenseMatrix elementwise_impl(const DenseMatrix& a_wrapper,
                             const DenseMatrix& b_wrapper,
                             const std::optional<std::string>& device_name) {
    const auto& a = std::get<DenseMatrixT<T>>(a_wrapper.storage);
    const auto& b = std::get<DenseMatrixT<T>>(b_wrapper.storage);

    DenseMatrixT<T> c(a.rows(), a.cols(), a.batch_size());
    Queue& queue = acquire_queue(device_name, Backend::AUTO);
    // Shape agreement is checked inside elementwise_into, which throws
    // std::invalid_argument -- pybind11 surfaces that as a Python ValueError.
    linalg::elementwise_into<T, Op>(queue, a.view(), b.view(), c.view());
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
    const auto& a = std::get<DenseMatrixT<T>>(a_wrapper.storage);
    const auto& b = std::get<DenseMatrixT<T>>(b_wrapper.storage);
    const T alpha = scalar_from_object<T>(alpha_object);
    const T beta = scalar_from_object<T>(beta_object);

    DenseMatrixT<T> c(a.rows(), a.cols(), a.batch_size());
    Queue& queue = acquire_queue(device_name, Backend::AUTO);
    linalg::axpby_into<T>(queue, alpha, a.view(), beta, b.view(), c.view());
    queue.wait();
    return wrap_dense(std::move(c));
}

template <typename T>
DenseMatrix scale_impl(const DenseMatrix& a_wrapper,
                       const py::object& alpha_object,
                       const std::optional<std::string>& device_name) {
    const auto& a = std::get<DenseMatrixT<T>>(a_wrapper.storage);
    const T alpha = scalar_from_object<T>(alpha_object);

    // linalg::scale is in-place, but in-place is meaningless across this
    // boundary: the caller's numpy array was copied into device memory on the
    // way in, so mutating it would not be visible in Python. Returning
    // alpha * A is the honest spelling of the same operation.
    DenseMatrixT<T> c(a.rows(), a.cols(), a.batch_size());
    Queue& queue = acquire_queue(device_name, Backend::AUTO);
    linalg::axpby_into<T>(queue, alpha, a.view(), T(0), a.view(), c.view());
    queue.wait();
    return wrap_dense(std::move(c));
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
}

}  // namespace batchlas::python
