#include <blas/linalg.hh>
#include <batchlas/backend_config.h>
#include <stdexcept>
#include <type_traits>

namespace batchlas {

namespace {

template <typename T>
void validate_symm_arguments(const MatrixView<T, MatrixFormat::Dense>& A,
                             const MatrixView<T, MatrixFormat::Dense>& B,
                             const MatrixView<T, MatrixFormat::Dense>& C,
                             Side side) {
    if (A.rows() != A.cols()) {
        throw std::invalid_argument("symm requires A to be square");
    }
    if (A.batch_size() != B.batch_size() || B.batch_size() != C.batch_size()) {
        throw std::invalid_argument("symm requires matching batch sizes");
    }

    const int expected_rows = side == Side::Left ? A.rows() : B.rows();
    const int expected_cols = side == Side::Left ? B.cols() : A.cols();
    const int shared_dim = side == Side::Left ? B.rows() : B.cols();

    if (shared_dim != A.rows()) {
        throw std::invalid_argument("symm dimension mismatch between A and B");
    }
    if (C.rows() != expected_rows || C.cols() != expected_cols) {
        throw std::invalid_argument("symm output matrix has incompatible dimensions");
    }
}

}  // namespace

template <Backend Ba, typename T, typename std::enable_if<std::is_floating_point_v<T>, int>::type>
Event symm(Queue& ctx,
           const MatrixView<T, MatrixFormat::Dense>& A,
           const MatrixView<T, MatrixFormat::Dense>& B,
           const MatrixView<T, MatrixFormat::Dense>& C,
           T alpha,
           T beta,
           Side side,
           Uplo uplo) {
    validate_symm_arguments(A, B, C, side);

    Matrix<T, MatrixFormat::Dense> symmetric_a(A.rows(), A.cols(), A.batch_size(), A.ld(), A.stride());
    auto symmetric_a_view = symmetric_a.view();

    MatrixView<T, MatrixFormat::Dense>::copy(ctx, symmetric_a_view, A).wait();
    symmetric_a_view.symmetrize(ctx, uplo).wait();

    if (side == Side::Left) {
        return gemm<Ba>(ctx,
                        symmetric_a_view,
                        B,
                        C,
                        alpha,
                        beta,
                        Transpose::NoTrans,
                        Transpose::NoTrans);
    }

    return gemm<Ba>(ctx,
                    B,
                    symmetric_a_view,
                    C,
                    alpha,
                    beta,
                    Transpose::NoTrans,
                    Transpose::NoTrans);
}

#define SYMM_INSTANTIATE(back, fp) \
    template Event symm<back, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        fp, \
        fp, \
        Side, \
        Uplo);

#define INSTANTIATE_SYMM_FOR_BACKEND(back) \
    SYMM_INSTANTIATE(back, float) \
    SYMM_INSTANTIATE(back, double)

#if BATCHLAS_HAS_ROCM_BACKEND
INSTANTIATE_SYMM_FOR_BACKEND(Backend::ROCM)
#endif
#if BATCHLAS_HAS_MKL_BACKEND
INSTANTIATE_SYMM_FOR_BACKEND(Backend::MKL)
#endif

#undef INSTANTIATE_SYMM_FOR_BACKEND
#undef SYMM_INSTANTIATE

}  // namespace batchlas