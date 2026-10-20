#pragma once

namespace batchlas::sycl_gemm {

template <typename T>
struct LinearEpilogue {
    // beta = 0 is alpha * accum alone: + 0 * 0 would turn -0 into +0.
    // evidence: docs/perf/gemm.md#gemm-wp-x-review-beta-zero-keeps-negative-zero
    static T apply(T alpha, T beta, T accum, T prior) {
        return beta == T(0) ? alpha * accum : alpha * accum + beta * prior;
    }

    // beta == 0 never reads C: 0 * NaN from a dirty workspace is NaN, and reference BLAS skips it.
    // evidence: docs/design/known-defects.md#defect-11-native-gemm-reads-c-at-beta-zero
    static void store(T* dst, T alpha, T beta, T accum) {
        *dst = beta == T(0) ? alpha * accum : apply(alpha, beta, accum, *dst);
    }
};

} // namespace batchlas::sycl_gemm
