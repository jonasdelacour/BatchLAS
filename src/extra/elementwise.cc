#include <blas/linalg-ops.hh>
#include <blas/matrix.hh>
#include <util/kernel-heuristics.hh>
#include <util/sycl-device-queue.hh>

#include <complex>
#include <stdexcept>

#include "../queue.hh"
#include "../util/template-instantiations.hh"

namespace batchlas::linalg {

namespace {

template <typename T, BinaryOp Op>
struct ElementwiseBinaryKernel {};

template <typename T>
struct AxpbyKernel {};

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
        throw std::invalid_argument(std::string(where) + ": operands must have the same shape");
    }
}

}  // namespace

template <typename T, BinaryOp Op>
Event elementwise_into(Queue& ctx,
                       const MatrixView<T, MatrixFormat::Dense>& A,
                       const MatrixView<T, MatrixFormat::Dense>& B,
                       const MatrixView<T, MatrixFormat::Dense>& C) {
    check_same_shape(A, B, "linalg::elementwise_into");
    check_same_shape(A, C, "linalg::elementwise_into");

    const size_t rows = static_cast<size_t>(A.rows());
    const size_t cols = static_cast<size_t>(A.cols());
    const size_t total = rows * cols * static_cast<size_t>(A.batch_size());
    if (total == 0) return ctx.get_event();

    auto [global_size, local_size] =
        compute_nd_range_sizes(total, ctx.device(), KernelType::ELEMENTWISE);

    // Indexed through kernel views rather than raw pointers so that strided and
    // non-contiguous views (submatrices, in particular) work the same as dense
    // ones. The views may alias: every work-item touches one element of each.
    auto Av = A.kernel_view();
    auto Bv = B.kernel_view();
    auto Cv = C.kernel_view();

    ctx->parallel_for<ElementwiseBinaryKernel<T, Op>>(
        sycl::nd_range<1>(global_size, local_size), [=](sycl::nd_item<1> item) {
            const size_t stride = item.get_global_range(0);
            for (size_t flat = item.get_global_id(0); flat < total; flat += stride) {
                const size_t b = flat / (rows * cols);
                const size_t rem = flat % (rows * cols);
                const size_t col = rem / rows;
                const size_t row = rem % rows;
                Cv(row, col, b) = apply<T, Op>(Av(row, col, b), Bv(row, col, b));
            }
        });
    return ctx.get_event();
}

template <typename T>
Event axpby_into(Queue& ctx,
                 T alpha,
                 const MatrixView<T, MatrixFormat::Dense>& A,
                 T beta,
                 const MatrixView<T, MatrixFormat::Dense>& B,
                 const MatrixView<T, MatrixFormat::Dense>& C) {
    check_same_shape(A, B, "linalg::axpby_into");
    check_same_shape(A, C, "linalg::axpby_into");

    const size_t rows = static_cast<size_t>(A.rows());
    const size_t cols = static_cast<size_t>(A.cols());
    const size_t total = rows * cols * static_cast<size_t>(A.batch_size());
    if (total == 0) return ctx.get_event();

    auto [global_size, local_size] =
        compute_nd_range_sizes(total, ctx.device(), KernelType::ELEMENTWISE);

    auto Av = A.kernel_view();
    auto Bv = B.kernel_view();
    auto Cv = C.kernel_view();

    ctx->parallel_for<AxpbyKernel<T>>(
        sycl::nd_range<1>(global_size, local_size), [=](sycl::nd_item<1> item) {
            const size_t stride = item.get_global_range(0);
            for (size_t flat = item.get_global_id(0); flat < total; flat += stride) {
                const size_t b = flat / (rows * cols);
                const size_t rem = flat % (rows * cols);
                const size_t col = rem / rows;
                const size_t row = rem % rows;
                Cv(row, col, b) = alpha * Av(row, col, b) + beta * Bv(row, col, b);
            }
        });
    return ctx.get_event();
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
    BATCHLAS_INSTANTIATE(sig::scale<BATCHLAS_UNPAREN fp>, scale, BATCHLAS_UNPAREN fp)

BATCHLAS_FOR_EACH_SCALAR_TYPE(ELEMENTWISE_INSTANTIATE)

#undef ELEMENTWISE_INSTANTIATE

}  // namespace batchlas::linalg
