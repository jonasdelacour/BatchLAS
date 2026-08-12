#include <batchlas/blas/linalg.hh>
#include <batchlas/backend_config.h>

#include "../util/template-instantiations.hh"
#include "symmetric_product_fold.hh"

#include <stdexcept>

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

template <Backend Ba, RealScalar T>
Event syr2k(Queue& ctx,
            const MatrixView<T, MatrixFormat::Dense>& A,
            const MatrixView<T, MatrixFormat::Dense>& B,
            const MatrixView<T, MatrixFormat::Dense>& C,
            T alpha,
            T beta,
            Uplo uplo,
            Transpose transA) {
    validate_syr2k_arguments(A, B, C, transA);

    const Transpose transB = transA == Transpose::NoTrans ? Transpose::Trans : Transpose::NoTrans;

    // As in syrk.cc: the two gemms compute the whole symmetric product, so they
    // cannot be aimed at C -- that wrote both triangles and ignored `uplo`, which
    // was an unnamed parameter here. Product to scratch, named triangle folded
    // back. See symmetric_product_fold.hh.
    auto product = Matrix<T, MatrixFormat::Dense>::Zeros(C.rows(), C.cols(), C.batch_size());
    auto product_view = product.view();
    gemm<Ba>(ctx, A, B, product_view,
             {.alpha = alpha, .beta = T(0), .transA = transA, .transB = transB}).wait();
    gemm<Ba>(ctx, B, A, product_view,
             {.alpha = alpha, .beta = T(1), .transA = transA, .transB = transB}).wait();

    auto folded = detail::fold_symmetric_product_into_triangle<T>(ctx, C, product_view, beta, uplo);
    folded.wait();  // the scratch product dies with this scope
    return folded;
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