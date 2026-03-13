#include <blas/linalg.hh>
#include <batchlas/backend_config.h>

#include "../util/template-instantiations.hh"

#include <stdexcept>
#include <type_traits>

namespace batchlas {

namespace {

template <typename T>
void validate_syr2k_arguments(const MatrixView<T, MatrixFormat::Dense>& A,
                              const MatrixView<T, MatrixFormat::Dense>& B,
                              const MatrixView<T, MatrixFormat::Dense>& C,
                              Transpose transA) {
    if (transA == Transpose::ConjTrans) {
        throw std::invalid_argument("syr2k does not support ConjTrans for real-valued inputs");
    }
    if (C.rows() != C.cols()) {
        throw std::invalid_argument("syr2k requires C to be square");
    }
    if (A.batch_size() != B.batch_size() || B.batch_size() != C.batch_size()) {
        throw std::invalid_argument("syr2k requires matching batch sizes");
    }

    const int expected_n = transA == Transpose::NoTrans ? A.rows() : A.cols();
    const int other_n = transA == Transpose::NoTrans ? B.rows() : B.cols();
    const int expected_k = transA == Transpose::NoTrans ? A.cols() : A.rows();
    const int other_k = transA == Transpose::NoTrans ? B.cols() : B.rows();
    if (expected_n != C.rows() || other_n != C.rows() || expected_k != other_k) {
        throw std::invalid_argument("syr2k dimension mismatch between A, B, and C");
    }
}

} // namespace

template <Backend Ba, typename T, typename std::enable_if<std::is_floating_point_v<T>, int>::type>
Event syr2k(Queue& ctx,
            const MatrixView<T, MatrixFormat::Dense>& A,
            const MatrixView<T, MatrixFormat::Dense>& B,
            const MatrixView<T, MatrixFormat::Dense>& C,
            T alpha,
            T beta,
            Uplo,
            Transpose transA) {
    validate_syr2k_arguments(A, B, C, transA);

    const Transpose transB = transA == Transpose::NoTrans ? Transpose::Trans : Transpose::NoTrans;
    gemm<Ba>(ctx, A, B, C, alpha, beta, transA, transB).wait();
    return gemm<Ba>(ctx, B, A, C, alpha, T(1), transA, transB);
}

#define SYR2K_INSTANTIATE(back, fp) \
    template Event syr2k<back, BATCHLAS_UNPAREN fp>( \
        Queue&, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        BATCHLAS_UNPAREN fp, \
        BATCHLAS_UNPAREN fp, \
        Uplo, \
        Transpose);

#if BATCHLAS_HAS_MKL_BACKEND
BATCHLAS_FOR_EACH_REAL_TYPE_1(SYR2K_INSTANTIATE, Backend::MKL)
#endif

#undef SYR2K_INSTANTIATE

} // namespace batchlas