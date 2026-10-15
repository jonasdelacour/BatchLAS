#pragma once

namespace batchlas::sycl_gemm {

template <typename T>
struct LinearEpilogue {
    static T apply(T alpha, T beta, T accum, T prior) {
        return alpha * accum + beta * prior;
    }

    // beta == 0 never reads C: 0 * NaN from a dirty workspace is NaN, and reference BLAS skips it.
    // evidence: docs/design/known-defects.md#defect-11-native-gemm-reads-c-at-beta-zero
    static void store(T* dst, T alpha, T beta, T accum) {
        *dst = beta == T(0) ? alpha * accum : apply(alpha, beta, accum, *dst);
    }
};

} // namespace batchlas::sycl_gemm
