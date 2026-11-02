// SPDX-License-Identifier: MIT
#pragma once

namespace batchlas::sycl_gemm {

// \f$\beta = 0\f$ is \f$\alpha \cdot \mathrm{accum}\f$ alone and never reads C: \f$0 \cdot \mathrm{NaN}\f$ is NaN, and \f$-0 + 0 \cdot 0 = +0\f$.
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
