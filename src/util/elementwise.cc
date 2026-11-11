// SPDX-License-Identifier: MIT
#include <batchlas/blas/linalg-ops.hh>
#include <batchlas/blas/queue-dispatch.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/kernel-heuristics.hh>
#include <batchlas/util/sycl-device-queue.hh>

#include <algorithm>
#include <complex>
#include <stdexcept>

#include <src/kernels/common/elementwise_glue_v2.hh>
#include <src/queue.hh>
#include <src/util/batch_mask.hh>
#include <src/util/template-instantiations.hh>

namespace batchlas::linalg {

namespace {

template <typename T, BinaryOp Op>
struct ElementwiseBinaryKernel {};

template <typename T>
struct AxpbyKernel {};

template <typename T, Uplo U>
struct TriangularMaskKernel {};

template <typename T, BinaryOp Op>
inline T apply(const T& a, const T& b) {
    if constexpr (Op == BinaryOp::Add) return a + b;
    else if constexpr (Op == BinaryOp::Subtract) return a - b;
    else if constexpr (Op == BinaryOp::Multiply) return a * b;
    else return a / b;
}

template <typename T>
void check_same_shape(const MatrixView<T, MatrixFormat::Dense>& A,
                      const MatrixView<T, MatrixFormat::Dense>& B,
                      const char* where) {
    if (A.rows() != B.rows() || A.cols() != B.cols() || A.batch_size() != B.batch_size()) {
        throw batchlas::invalid_argument(std::string(where) + ": operands must have the same shape");
    }
}


// One batched 2-D launch over a rows x cols shape (no flat index, no % or /); f(ex, b) runs
// for every active item b. Views are indexed per item, so submatrices (ld > rows) work, and
// operands may alias: each item touches one element of each. Not device::batched2d: that takes a
// sycl::queue&, which would bypass Queue's event recording.
template <typename Name, typename F>
Event launch_2d(Queue& ctx, int rows, int cols, int batch, const std::uint8_t* mask, F f) {
    if (rows == 0 || cols == 0 || batch == 0) return ctx.get_event();
    ctx->parallel_for<Name>(device::batched2d_range(batch, rows, cols, device::batched2d_shape(rows, cols)),
                            [=](sycl::nd_item<3> item) {
                                const device::Grid2D<> ex{item};
                                ex.for_each_batch(batch, [&](int b) {
                                    if (mask && !mask[b]) return;
                                    f(ex, b);
                                });
                            });
    return ctx.get_event();
}

}  // namespace

template <typename T, BinaryOp Op>
Event elementwise_into(Queue& ctx,
                       const MatrixView<T, MatrixFormat::Dense>& A,
                       const MatrixView<T, MatrixFormat::Dense>& B,
                       const MatrixView<T, MatrixFormat::Dense>& C) {
    ::batchlas::detail::require_device_fp64<T>(ctx, "linalg::elementwise");
    check_same_shape(A, B, "linalg::elementwise_into");
    check_same_shape(A, C, "linalg::elementwise_into");
    const std::uint8_t* mask = ::batchlas::detail::mask_ptr(::batchlas::detail::call_mask("linalg::elementwise_into", A, B, C));

    const int m = A.rows(), n = A.cols();
    auto Av = A.kernel_view();
    auto Bv = B.kernel_view();
    auto Cv = C.kernel_view();
    return launch_2d<ElementwiseBinaryKernel<T, Op>>(ctx, m, n, A.batch_size(), mask, [=](const auto& ex, int b) {
        device::glue::map2d(ex, device::item(Cv, b, m, n), [](T a, T bb) { return apply<T, Op>(a, bb); }, device::item(Av, b, m, n),
                        device::item(Bv, b, m, n));
    });
}

template <typename T>
Event axpby_into(Queue& ctx,
                 T alpha,
                 const MatrixView<T, MatrixFormat::Dense>& A,
                 T beta,
                 const MatrixView<T, MatrixFormat::Dense>& B,
                 const MatrixView<T, MatrixFormat::Dense>& C) {
    ::batchlas::detail::require_device_fp64<T>(ctx, "linalg::axpby");
    check_same_shape(A, B, "linalg::axpby_into");
    check_same_shape(A, C, "linalg::axpby_into");
    const std::uint8_t* mask = ::batchlas::detail::mask_ptr(::batchlas::detail::call_mask("linalg::axpby_into", A, B, C));

    const int m = A.rows(), n = A.cols();
    auto Av = A.kernel_view();
    auto Bv = B.kernel_view();
    auto Cv = C.kernel_view();
    return launch_2d<AxpbyKernel<T>>(ctx, m, n, A.batch_size(), mask, [=](const auto& ex, int b) {
        device::glue::geam(ex, alpha, device::item(Av, b, m, n), beta, device::item(Bv, b, m, n), device::item(Cv, b, m, n));
    });
}

template <typename T>
Event triangular_mask_into(Queue& ctx,
                           const MatrixView<T, MatrixFormat::Dense>& A,
                           const MatrixView<T, MatrixFormat::Dense>& C,
                           Uplo uplo,
                           int64_t k) {
    ::batchlas::detail::require_device_fp64<T>(ctx, "linalg::triangular_mask");
    check_same_shape(A, C, "linalg::triangular_mask_into");
    const std::uint8_t* mask = ::batchlas::detail::mask_ptr(::batchlas::detail::call_mask("linalg::triangular_mask_into", A, C));

    const int m = A.rows(), n = A.cols();
    // Signed and clamped: below the main diagonal the offset is negative, and any |k| past the
    // shape keeps or drops everything, so int32 holds every distinct case.
    const int kc = static_cast<int>(std::clamp<int64_t>(k, -static_cast<int64_t>(m), n));
    auto Av = A.kernel_view();
    auto Cv = C.kernel_view();
    const auto run = [&](auto uplo_tag) {
        return launch_2d<TriangularMaskKernel<T, decltype(uplo_tag)::value>>(ctx, m, n, A.batch_size(), mask,
                                                                            [=](const auto& ex, int b) {
            device::glue::trimask(ex, device::glue::tri(uplo_tag, kc), device::item(Av, b, m, n), device::item(Cv, b, m, n));
        });
    };
    return uplo == Uplo::Upper ? run(device::glue::UpperC{}) : run(device::glue::LowerC{});
}

template <typename T>
Event scale(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, T alpha) {
    // beta = 0 with B = A, so the second term contributes nothing and A is
    // read and written in place by the same work-item.
    return axpby_into<T>(ctx, alpha, A, T(0), A, A);
}

#define ELEMENTWISE_INSTANTIATE(fp)                                                        \
    BATCHLAS_INSTANTIATE(sig::elementwise_into<BATCHLAS_UNPAREN fp BATCHLAS_COMMA           \
                                              BinaryOp::Add>,                              \
                         elementwise_into, BATCHLAS_UNPAREN fp, BinaryOp::Add)             \
    BATCHLAS_INSTANTIATE(sig::elementwise_into<BATCHLAS_UNPAREN fp BATCHLAS_COMMA           \
                                              BinaryOp::Subtract>,                         \
                         elementwise_into, BATCHLAS_UNPAREN fp, BinaryOp::Subtract)        \
    BATCHLAS_INSTANTIATE(sig::elementwise_into<BATCHLAS_UNPAREN fp BATCHLAS_COMMA           \
                                              BinaryOp::Multiply>,                         \
                         elementwise_into, BATCHLAS_UNPAREN fp, BinaryOp::Multiply)        \
    BATCHLAS_INSTANTIATE(sig::elementwise_into<BATCHLAS_UNPAREN fp BATCHLAS_COMMA           \
                                              BinaryOp::Divide>,                           \
                         elementwise_into, BATCHLAS_UNPAREN fp, BinaryOp::Divide)          \
    BATCHLAS_INSTANTIATE(sig::axpby_into<BATCHLAS_UNPAREN fp>, axpby_into,                 \
                         BATCHLAS_UNPAREN fp)                                              \
    BATCHLAS_INSTANTIATE(sig::scale<BATCHLAS_UNPAREN fp>, scale, BATCHLAS_UNPAREN fp)         \
    BATCHLAS_INSTANTIATE(sig::triangular_mask_into<BATCHLAS_UNPAREN fp>,                      \
                         triangular_mask_into, BATCHLAS_UNPAREN fp)

BATCHLAS_FOR_EACH_SCALAR_TYPE(ELEMENTWISE_INSTANTIATE)

#undef ELEMENTWISE_INSTANTIATE

}  // namespace batchlas::linalg
