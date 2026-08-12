#include <batchlas/blas/linalg.hh>
#include <batchlas/backend_config.h>

#include "../util/template-instantiations.hh"
#include "symmetric_product_fold.hh"

#include <stdexcept>

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

template <Backend Ba, RealScalar T>
Event syrk(Queue& ctx,
           const MatrixView<T, MatrixFormat::Dense>& A,
           const MatrixView<T, MatrixFormat::Dense>& C,
           T alpha,
           T beta,
           Uplo uplo,
           Transpose transA) {
    validate_syrk_arguments(A, C, transA);

    const Transpose transB = transA == Transpose::NoTrans ? Transpose::Trans : Transpose::NoTrans;

    // This used to be a single gemm straight into C, with `uplo` an unnamed
    // parameter: it wrote *both* triangles of C and so silently overwrote the
    // half SYRK does not own. The product is computed into scratch and only the
    // named triangle is folded back. See symmetric_product_fold.hh.
    auto product = Matrix<T, MatrixFormat::Dense>::Zeros(C.rows(), C.cols(), C.batch_size());
    auto product_view = product.view();
    gemm<Ba>(ctx, A, A, product_view,
             {.alpha = alpha, .beta = T(0), .transA = transA, .transB = transB}).wait();

    auto folded = detail::fold_symmetric_product_into_triangle<T>(ctx, C, product_view, beta, uplo);
    folded.wait();  // the scratch product dies with this scope
    return folded;
}

#define SYRK_INSTANTIATE(back, fp) \
    template Event syrk<back, BATCHLAS_UNPAREN fp>( \
        Queue&, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        BATCHLAS_UNPAREN fp, \
        BATCHLAS_UNPAREN fp, \
        Uplo, \
        Transpose);

#if BATCHLAS_HAS_MKL_BACKEND
BATCHLAS_FOR_EACH_REAL_TYPE_1(SYRK_INSTANTIATE, Backend::MKL)
#endif

#undef SYRK_INSTANTIATE

} // namespace batchlas