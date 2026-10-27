// The solve with a Cholesky factor (potrs.hh), shared by potrs, posv's cta and blocked and potri.
// evidence: docs/perf/potri.md#potrs-and-potri

#include <batchlas/backend_config.h>

#include <batchlas/blas/functions/trsm.hh>

#include <src/ops/getrs/getrs_native.hh>
#include <src/ops/potrs/potrs.hh>
#include <src/util/template-instantiations.hh>

#include <complex>
#include <type_traits>

namespace batchlas::ops::potrs {

// B := A^-1 B from the factor in the uplo triangle of A; fused (caller checked fused_fits) or two trsm.
template <Backend B, class T>
Event solve(Queue& q, const MatrixView<T, MatrixFormat::Dense>& A, const MatrixView<T, MatrixFormat::Dense>& Bm,
            Uplo uplo, bool fused) {
    if (fused) return sycl_getrs::potrs_fused_dispatch<T>(q, A, Bm, uplo);
    // ConjTrans only for a complex scalar: a real backend may refuse ConjTrans, and
    // for a real matrix Trans is the same operation. For a real T, base_type<T> IS T.
    constexpr bool kReal = std::is_same_v<T, typename base_type<T>::type>;
    constexpr Transpose kAdj = kReal ? Transpose::Trans : Transpose::ConjTrans;
    const T one = T(1);
    if (uplo == Uplo::Lower) {
        (void)trsm<B, T>(q, A, Bm, one, Side::Left, Uplo::Lower, Transpose::NoTrans, Diag::NonUnit);
        return trsm<B, T>(q, A, Bm, one, Side::Left, Uplo::Lower, kAdj, Diag::NonUnit);
    }
    (void)trsm<B, T>(q, A, Bm, one, Side::Left, Uplo::Upper, kAdj, Diag::NonUnit);
    return trsm<B, T>(q, A, Bm, one, Side::Left, Uplo::Upper, Transpose::NoTrans, Diag::NonUnit);
}

#define POTRS_SOLVE_INSTANTIATE(back, fp)                                                                   \
    template Event solve<back, BATCHLAS_UNPAREN fp>(Queue&, const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
                                                    const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, Uplo, bool);
BATCHLAS_INSTANTIATE_SCALAR_ALL_BACKENDS(POTRS_SOLVE_INSTANTIATE)
#undef POTRS_SOLVE_INSTANTIATE

}  // namespace batchlas::ops::potrs
