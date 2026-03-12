#include <blas/linalg.hh>
#include <batchlas/backend_config.h>

#include <stdexcept>
#include <type_traits>

namespace batchlas {

namespace {

template <typename T>
void validate_syrk_arguments(const MatrixView<T, MatrixFormat::Dense>& A,
                             const MatrixView<T, MatrixFormat::Dense>& C,
                             Transpose transA) {
    if (transA == Transpose::ConjTrans) {
        throw std::invalid_argument("syrk does not support ConjTrans for real-valued inputs");
    }
    if (C.rows() != C.cols()) {
        throw std::invalid_argument("syrk requires C to be square");
    }
    if (A.batch_size() != C.batch_size()) {
        throw std::invalid_argument("syrk requires matching batch sizes");
    }

    const int expected_n = transA == Transpose::NoTrans ? A.rows() : A.cols();
    if (expected_n != C.rows()) {
        throw std::invalid_argument("syrk dimension mismatch between A and C");
    }
}

} // namespace

template <Backend Ba, typename T, typename std::enable_if<std::is_floating_point_v<T>, int>::type>
Event syrk(Queue& ctx,
           const MatrixView<T, MatrixFormat::Dense>& A,
           const MatrixView<T, MatrixFormat::Dense>& C,
           T alpha,
           T beta,
           Uplo,
           Transpose transA) {
    validate_syrk_arguments(A, C, transA);

    const Transpose transB = transA == Transpose::NoTrans ? Transpose::Trans : Transpose::NoTrans;
    return gemm<Ba>(ctx, A, A, C, alpha, beta, transA, transB);
}

#define SYRK_INSTANTIATE(back, fp) \
    template Event syrk<back, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        fp, \
        fp, \
        Uplo, \
        Transpose);

#if BATCHLAS_HAS_MKL_BACKEND
SYRK_INSTANTIATE(Backend::MKL, float)
SYRK_INSTANTIATE(Backend::MKL, double)
#endif

#undef SYRK_INSTANTIATE

} // namespace batchlas