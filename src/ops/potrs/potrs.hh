#pragma once

// The solve with a Cholesky factor, shared by potrs, posv's cta and blocked and potri.
// evidence: docs/perf/potri.md#potrs-and-potri

#include <batchlas/blas/functions/trsm.hh>

#include <src/ops/getrs/getrs_native.hh>
#include <src/select/select.hh>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace batchlas::ops::potrs {

// Correctness only: can the one-launch fused solve take an n x nrhs system here? Never a speed term.
template <class T>
bool fused_fits(const select::Device& d, int64_t n, int64_t nrhs) {
    const auto cap = sycl_getrs::getrs_fused_max_rhs_elems<T>(static_cast<std::size_t>(d.slm_budget));
    return d.is_gpu && d.has_sg32 && n >= 1 && nrhs >= 1 && nrhs <= sycl_getrs::kGetrsFusedMaxRhs &&
           static_cast<std::size_t>(n) * static_cast<std::size_t>(nrhs) <= cap;
}

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

}  // namespace batchlas::ops::potrs
