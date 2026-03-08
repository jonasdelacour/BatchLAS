#pragma once

namespace batchlas::sycl_gemm {

template <typename T>
struct LinearEpilogue {
    static T apply(T alpha, T beta, T accum, T prior) {
        return alpha * accum + beta * prior;
    }
};

} // namespace batchlas::sycl_gemm