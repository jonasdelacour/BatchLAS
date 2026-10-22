// SPDX-License-Identifier: MIT
#pragma once

namespace batchlas::sycl_gemm {

// beta == 0 is alpha * accum alone and never reads C: 0 * NaN is NaN, and + 0 * 0 turns -0 into +0.
// evidence: docs/design/known-defects.md#defect-11-native-gemm-reads-c-at-beta-zero
template <typename T>
struct LinearEpilogue {
    static T apply(T alpha, T beta, T accum, T prior) {
        return beta == T(0) ? alpha * accum : alpha * accum + beta * prior;
    }

    static void store(T* dst, T alpha, T beta, T accum) {
        *dst = beta == T(0) ? alpha * accum : apply(alpha, beta, accum, *dst);
    }
};

} // namespace batchlas::sycl_gemm
