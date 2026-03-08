#pragma once

#include "register_tiled_common.hh"

namespace batchlas::sycl_gemm {

template <typename T>
Event launch_register_32x32(Queue& ctx,
                            const MatrixView<T, MatrixFormat::Dense>& A,
                            const MatrixView<T, MatrixFormat::Dense>& B,
                            const MatrixView<T, MatrixFormat::Dense>& C,
                            T alpha,
                            T beta,
                            const char* (*kernel_trace_name)(KernelVariant)) {
    return launch_register_tiled<T, 32, 32, 8, 2>(ctx, A, B, C, alpha, beta, kernel_trace_name);
}

template <typename T>
Event launch_register_64x64(Queue& ctx,
                            const MatrixView<T, MatrixFormat::Dense>& A,
                            const MatrixView<T, MatrixFormat::Dense>& B,
                            const MatrixView<T, MatrixFormat::Dense>& C,
                            T alpha,
                            T beta,
                            const char* (*kernel_trace_name)(KernelVariant)) {
    return launch_register_tiled<T, 64, 64, 8, 4>(ctx, A, B, C, alpha, beta, kernel_trace_name);
}

template <typename T>
Event launch_register_64x64_k16(Queue& ctx,
                                const MatrixView<T, MatrixFormat::Dense>& A,
                                const MatrixView<T, MatrixFormat::Dense>& B,
                                const MatrixView<T, MatrixFormat::Dense>& C,
                                T alpha,
                                T beta,
                                const char* (*kernel_trace_name)(KernelVariant)) {
    return launch_register_tiled<T, 64, 64, 16, 4>(ctx, A, B, C, alpha, beta, kernel_trace_name);
}

template <typename T>
Event launch_register_64x64_k16_tn(Queue& ctx,
                                   const MatrixView<T, MatrixFormat::Dense>& A,
                                   const MatrixView<T, MatrixFormat::Dense>& B,
                                   const MatrixView<T, MatrixFormat::Dense>& C,
                                   T alpha,
                                   T beta,
                                   const char* (*kernel_trace_name)(KernelVariant)) {
    return launch_register_tiled<T, 64, 64, 16, 4, 4, 4, 4, 1, 1, Transpose::Trans, Transpose::NoTrans>(
        ctx, A, B, C, alpha, beta, kernel_trace_name);
}

template <typename T>
Event launch_register_64x64_k16_nt(Queue& ctx,
                                   const MatrixView<T, MatrixFormat::Dense>& A,
                                   const MatrixView<T, MatrixFormat::Dense>& B,
                                   const MatrixView<T, MatrixFormat::Dense>& C,
                                   T alpha,
                                   T beta,
                                   const char* (*kernel_trace_name)(KernelVariant)) {
    return launch_register_tiled<T, 64, 64, 16, 4, 4, 4, 4, 1, 1, Transpose::NoTrans, Transpose::Trans>(
        ctx, A, B, C, alpha, beta, kernel_trace_name);
}

template <typename T>
Event launch_register_64x64_k16_tt(Queue& ctx,
                                   const MatrixView<T, MatrixFormat::Dense>& A,
                                   const MatrixView<T, MatrixFormat::Dense>& B,
                                   const MatrixView<T, MatrixFormat::Dense>& C,
                                   T alpha,
                                   T beta,
                                   const char* (*kernel_trace_name)(KernelVariant)) {
    return launch_register_tiled<T, 64, 64, 16, 4, 4, 4, 4, 1, 1, Transpose::Trans, Transpose::Trans>(
        ctx, A, B, C, alpha, beta, kernel_trace_name);
}

template <typename T>
Event launch_register_128x32_k16(Queue& ctx,
                                 const MatrixView<T, MatrixFormat::Dense>& A,
                                 const MatrixView<T, MatrixFormat::Dense>& B,
                                 const MatrixView<T, MatrixFormat::Dense>& C,
                                 T alpha,
                                 T beta,
                                 const char* (*kernel_trace_name)(KernelVariant)) {
    return launch_register_tiled<T, 128, 32, 16, 4, 4, 4, 4, 1, 2>(ctx, A, B, C, alpha, beta, kernel_trace_name);
}

template <typename T>
Event launch_register_128x32_k16_tn(Queue& ctx,
                                    const MatrixView<T, MatrixFormat::Dense>& A,
                                    const MatrixView<T, MatrixFormat::Dense>& B,
                                    const MatrixView<T, MatrixFormat::Dense>& C,
                                    T alpha,
                                    T beta,
                                    const char* (*kernel_trace_name)(KernelVariant)) {
    return launch_register_tiled<T, 128, 32, 16, 4, 4, 4, 4, 1, 2, Transpose::Trans, Transpose::NoTrans>(
        ctx, A, B, C, alpha, beta, kernel_trace_name);
}

template <typename T>
Event launch_register_128x32_k16_nt(Queue& ctx,
                                    const MatrixView<T, MatrixFormat::Dense>& A,
                                    const MatrixView<T, MatrixFormat::Dense>& B,
                                    const MatrixView<T, MatrixFormat::Dense>& C,
                                    T alpha,
                                    T beta,
                                    const char* (*kernel_trace_name)(KernelVariant)) {
    return launch_register_tiled<T, 128, 32, 16, 4, 4, 4, 4, 1, 2, Transpose::NoTrans, Transpose::Trans>(
        ctx, A, B, C, alpha, beta, kernel_trace_name);
}

template <typename T>
Event launch_register_128x32_k16_tt(Queue& ctx,
                                    const MatrixView<T, MatrixFormat::Dense>& A,
                                    const MatrixView<T, MatrixFormat::Dense>& B,
                                    const MatrixView<T, MatrixFormat::Dense>& C,
                                    T alpha,
                                    T beta,
                                    const char* (*kernel_trace_name)(KernelVariant)) {
    return launch_register_tiled<T, 128, 32, 16, 4, 4, 4, 4, 1, 2, Transpose::Trans, Transpose::Trans>(
        ctx, A, B, C, alpha, beta, kernel_trace_name);
}

template <typename T>
Event launch_register_128x32_k32(Queue& ctx,
                                 const MatrixView<T, MatrixFormat::Dense>& A,
                                 const MatrixView<T, MatrixFormat::Dense>& B,
                                 const MatrixView<T, MatrixFormat::Dense>& C,
                                 T alpha,
                                 T beta,
                                 const char* (*kernel_trace_name)(KernelVariant)) {
    return launch_register_tiled<T, 128, 32, 32, 4, 4, 4, 4, 1, 2>(ctx, A, B, C, alpha, beta, kernel_trace_name);
}

template <typename T>
Event launch_register_128x32_k32_s2_u1(Queue& ctx,
                                       const MatrixView<T, MatrixFormat::Dense>& A,
                                       const MatrixView<T, MatrixFormat::Dense>& B,
                                       const MatrixView<T, MatrixFormat::Dense>& C,
                                       T alpha,
                                       T beta,
                                       const char* (*kernel_trace_name)(KernelVariant)) {
    return launch_register_tiled<T, 128, 32, 32, 4, 4, 4, 4, 1, 2>(ctx, A, B, C, alpha, beta, kernel_trace_name);
}

template <typename T>
Event launch_register_128x32_k32_s2_u2(Queue& ctx,
                                       const MatrixView<T, MatrixFormat::Dense>& A,
                                       const MatrixView<T, MatrixFormat::Dense>& B,
                                       const MatrixView<T, MatrixFormat::Dense>& C,
                                       T alpha,
                                       T beta,
                                       const char* (*kernel_trace_name)(KernelVariant)) {
    return launch_register_tiled<T, 128, 32, 32, 4, 4, 4, 4, 2, 2>(ctx, A, B, C, alpha, beta, kernel_trace_name);
}

template <typename T>
Event launch_register_128x32_k32_s1_u4(Queue& ctx,
                                       const MatrixView<T, MatrixFormat::Dense>& A,
                                       const MatrixView<T, MatrixFormat::Dense>& B,
                                       const MatrixView<T, MatrixFormat::Dense>& C,
                                       T alpha,
                                       T beta,
                                       const char* (*kernel_trace_name)(KernelVariant)) {
    return launch_register_tiled<T, 128, 32, 32, 4, 4, 4, 4, 4, 1>(ctx, A, B, C, alpha, beta, kernel_trace_name);
}

template <typename T>
Event launch_register_128x32_k32_tn(Queue& ctx,
                                    const MatrixView<T, MatrixFormat::Dense>& A,
                                    const MatrixView<T, MatrixFormat::Dense>& B,
                                    const MatrixView<T, MatrixFormat::Dense>& C,
                                    T alpha,
                                    T beta,
                                    const char* (*kernel_trace_name)(KernelVariant)) {
    return launch_register_tiled<T, 128, 32, 32, 4, 4, 4, 4, 1, 2, Transpose::Trans, Transpose::NoTrans>(
        ctx, A, B, C, alpha, beta, kernel_trace_name);
}

template <typename T>
Event launch_register_128x32_k32_nt(Queue& ctx,
                                    const MatrixView<T, MatrixFormat::Dense>& A,
                                    const MatrixView<T, MatrixFormat::Dense>& B,
                                    const MatrixView<T, MatrixFormat::Dense>& C,
                                    T alpha,
                                    T beta,
                                    const char* (*kernel_trace_name)(KernelVariant)) {
    return launch_register_tiled<T, 128, 32, 32, 4, 4, 4, 4, 1, 2, Transpose::NoTrans, Transpose::Trans>(
        ctx, A, B, C, alpha, beta, kernel_trace_name);
}

template <typename T>
Event launch_register_128x32_k32_tt(Queue& ctx,
                                    const MatrixView<T, MatrixFormat::Dense>& A,
                                    const MatrixView<T, MatrixFormat::Dense>& B,
                                    const MatrixView<T, MatrixFormat::Dense>& C,
                                    T alpha,
                                    T beta,
                                    const char* (*kernel_trace_name)(KernelVariant)) {
    return launch_register_tiled<T, 128, 32, 32, 4, 4, 4, 4, 1, 2, Transpose::Trans, Transpose::Trans>(
        ctx, A, B, C, alpha, beta, kernel_trace_name);
}

template <typename T>
Event launch_register_128x64_k16_tn(Queue& ctx,
                                    const MatrixView<T, MatrixFormat::Dense>& A,
                                    const MatrixView<T, MatrixFormat::Dense>& B,
                                    const MatrixView<T, MatrixFormat::Dense>& C,
                                    T alpha,
                                    T beta,
                                    const char* (*kernel_trace_name)(KernelVariant)) {
    return launch_register_tiled<T, 128, 64, 16, 4, 4, 4, 4, 1, 1, Transpose::Trans, Transpose::NoTrans>(
        ctx, A, B, C, alpha, beta, kernel_trace_name);
}

template <typename T>
Event launch_register_128x64_k16_nt(Queue& ctx,
                                    const MatrixView<T, MatrixFormat::Dense>& A,
                                    const MatrixView<T, MatrixFormat::Dense>& B,
                                    const MatrixView<T, MatrixFormat::Dense>& C,
                                    T alpha,
                                    T beta,
                                    const char* (*kernel_trace_name)(KernelVariant)) {
    return launch_register_tiled<T, 128, 64, 16, 4, 4, 4, 4, 1, 1, Transpose::NoTrans, Transpose::Trans>(
        ctx, A, B, C, alpha, beta, kernel_trace_name);
}

template <typename T>
Event launch_register_128x64_k16_tt(Queue& ctx,
                                    const MatrixView<T, MatrixFormat::Dense>& A,
                                    const MatrixView<T, MatrixFormat::Dense>& B,
                                    const MatrixView<T, MatrixFormat::Dense>& C,
                                    T alpha,
                                    T beta,
                                    const char* (*kernel_trace_name)(KernelVariant)) {
    return launch_register_tiled<T, 128, 64, 16, 4, 4, 4, 4, 1, 1, Transpose::Trans, Transpose::Trans>(
        ctx, A, B, C, alpha, beta, kernel_trace_name);
}

template <typename T>
Event launch_register_128x64_k32_large(Queue& ctx,
                                       const MatrixView<T, MatrixFormat::Dense>& A,
                                       const MatrixView<T, MatrixFormat::Dense>& B,
                                       const MatrixView<T, MatrixFormat::Dense>& C,
                                       T alpha,
                                       T beta,
                                       const char* (*kernel_trace_name)(KernelVariant)) {
    return launch_register_tiled<T, 128, 64, 32, 8, 4, 4, 4, 4, 2>(ctx, A, B, C, alpha, beta, kernel_trace_name);
}

template <typename T>
Event launch_register_32x128_k16(Queue& ctx,
                                 const MatrixView<T, MatrixFormat::Dense>& A,
                                 const MatrixView<T, MatrixFormat::Dense>& B,
                                 const MatrixView<T, MatrixFormat::Dense>& C,
                                 T alpha,
                                 T beta,
                                 const char* (*kernel_trace_name)(KernelVariant)) {
    return launch_register_tiled<T, 32, 128, 16, 4>(ctx, A, B, C, alpha, beta, kernel_trace_name);
}

template <typename T>
Event launch_register_32x128_k16_tn(Queue& ctx,
                                    const MatrixView<T, MatrixFormat::Dense>& A,
                                    const MatrixView<T, MatrixFormat::Dense>& B,
                                    const MatrixView<T, MatrixFormat::Dense>& C,
                                    T alpha,
                                    T beta,
                                    const char* (*kernel_trace_name)(KernelVariant)) {
    return launch_register_tiled<T, 32, 128, 16, 4, 4, 4, 4, 1, 1, Transpose::Trans, Transpose::NoTrans>(
        ctx, A, B, C, alpha, beta, kernel_trace_name);
}

template <typename T>
Event launch_register_32x128_k16_tt(Queue& ctx,
                                    const MatrixView<T, MatrixFormat::Dense>& A,
                                    const MatrixView<T, MatrixFormat::Dense>& B,
                                    const MatrixView<T, MatrixFormat::Dense>& C,
                                    T alpha,
                                    T beta,
                                    const char* (*kernel_trace_name)(KernelVariant)) {
    return launch_register_tiled<T, 32, 128, 16, 4, 4, 4, 4, 1, 1, Transpose::Trans, Transpose::Trans>(
        ctx, A, B, C, alpha, beta, kernel_trace_name);
}

} // namespace batchlas::sycl_gemm