#pragma once

#include <src/util/internal-api.hh>
#include <batchlas/blas/enums.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>

namespace batchlas::sycl_gemm {

// Trace identities only (BATCHLAS_KERNEL_TRACE names). Which kernel runs is decided by
// src/ops/gemm/gemm.cc; each launcher below names the instantiation it took.
enum class KernelVariant {
    Direct,
    Tiled16,
    Tiled32x32Register,
    Tiled64x64Register,
    Tiled64x64RegisterK16,
    Tiled64x64RegisterK16TN,
    Tiled64x64RegisterK16NT,
    Tiled64x64RegisterK16TT,
    Tiled128x32RegisterK16,
    Tiled128x32RegisterK16TN,
    Tiled128x32RegisterK16NT,
    Tiled128x32RegisterK16TT,
    Tiled128x32RegisterK32TN,
    Tiled128x32RegisterK32NT,
    Tiled128x32RegisterK32TT,
    Tiled128x64RegisterK16TN,
    Tiled128x64RegisterK16NT,
    Tiled128x64RegisterK16TT,
    Tiled128x32RegisterK32S2U1Aligned,  // the NN 128x32x32 tile's two legs
    Tiled128x32RegisterK32S2U1Generic,
    Tiled128x64RegisterK32Large,
    Tiled128x64RegisterK32LargeU2,
    Tiled128x128RegisterK8,
    Tiled64x64RegisterK16Wide,
    Tiled64x64RegisterK16WideCN,
    Tiled64x64RegisterK16WideNC,
    Tiled128x32RegisterK16WideNC,
    Tiled32x128RegisterK16WideCN,
    Tiled32x128RegisterK16,
    Tiled32x128RegisterK16TN,
    Tiled32x128RegisterK16TT,
    Tiled32x32RegisterK16Wide,
    Tiled16x16RegisterK16Wide,
    Tiled64x64RegisterK16WideNT, Tiled64x64RegisterK16WideTN, Tiled64x64RegisterK16WideTT,
    Tiled64x64RegisterK16WideTC, Tiled64x64RegisterK16WideCT, Tiled64x64RegisterK16WideCC,
    Tiled128x32RegisterK16WideNT, Tiled128x32RegisterK16WideTN, Tiled128x32RegisterK16WideCN,
    Tiled32x128RegisterK16WideNT, Tiled32x128RegisterK16WideTN, Tiled32x128RegisterK16WideNC,
    Tiled32x32RegisterK16WideNT, Tiled32x32RegisterK16WideNC, Tiled32x32RegisterK16WideTN, Tiled32x32RegisterK16WideCN,
    Tiled32x32RegisterK16WideCT,
    Tiled16x16RegisterK16WideNT, Tiled16x16RegisterK16WideNC, Tiled16x16RegisterK16WideTN, Tiled16x16RegisterK16WideCN,
    Tiled16x16RegisterK16WideCT,
    Tiled64x32RegisterK16Wide, Tiled64x32RegisterK16WideNT, Tiled64x32RegisterK16WideNC, Tiled64x32RegisterK16WideTN,
    Tiled64x32RegisterK16WideCN, Tiled64x32RegisterK16WideCT,
    Tiled32x64RegisterK16Wide, Tiled32x64RegisterK16WideNT, Tiled32x64RegisterK16WideNC, Tiled32x64RegisterK16WideTN,
    Tiled32x64RegisterK16WideCN, Tiled32x64RegisterK16WideCT,
    Tiled128x128RegisterK8NT, Tiled128x128RegisterK8TN, Tiled128x128RegisterK8TT,
    Tiled64x64RegisterK8Wide, Tiled64x64RegisterK8WideNT, Tiled64x64RegisterK8WideNC, Tiled64x64RegisterK8WideTN,
    Tiled64x64RegisterK8WideTT, Tiled64x64RegisterK8WideTC, Tiled64x64RegisterK8WideCN, Tiled64x64RegisterK8WideCT,
    Tiled64x64RegisterK8WideCC,
    Tiled128x128RegisterK8S4, Tiled128x128RegisterK8S4NT, Tiled128x128RegisterK8S4TN, Tiled128x128RegisterK8S4TT,
    Tiled128x128RegisterK8Wide, Tiled128x128RegisterK8WideNT, Tiled128x128RegisterK8WideNC,
    Tiled128x128RegisterK8WideTN, Tiled128x128RegisterK8WideTT, Tiled128x128RegisterK8WideTC,
    Tiled128x128RegisterK8WideCN, Tiled128x128RegisterK8WideCT, Tiled128x128RegisterK8WideCC,
    Tiled32x32RegisterK16WideTT, Tiled32x32RegisterK16WideTC, Tiled32x32RegisterK16WideCC,
    Tiled16x16RegisterK16WideTT, Tiled16x16RegisterK16WideTC, Tiled16x16RegisterK16WideCC,
    Tiled64x32RegisterK16WideCC, Tiled32x64RegisterK16WideCC,
    Tiled128x32RegisterK16WideCC, Tiled32x128RegisterK16WideCC,
    Tiled256x128RegisterK8S4, Tiled256x128RegisterK8S4NT, Tiled256x128RegisterK8S4TN, Tiled256x128RegisterK8S4TT,
    Tiled128x256RegisterK8S4, Tiled128x256RegisterK8S4NT, Tiled128x256RegisterK8S4TN, Tiled128x256RegisterK8S4TT,
    Count,  // not a kernel: splitk_trace_name's table size, so append entries above it
};

// One launcher per gemm family (src/ops/gemm/choice.hh). Each serves exactly what its family's
// can_run admits and throws batchlas::unsupported otherwise -- never a silent fallback to another
// kernel. The transpose instantiation and the aligned/predicated leg are derived here.
template <typename T>
BATCHLAS_INTERNAL_API Event gemm_direct(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A,
                                        const MatrixView<T, MatrixFormat::Dense>& B,
                                        const MatrixView<T, MatrixFormat::Dense>& C, T alpha, T beta,
                                        Transpose transA, Transpose transB);

template <typename T>
BATCHLAS_INTERNAL_API Event gemm_tiled(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A,
                                       const MatrixView<T, MatrixFormat::Dense>& B,
                                       const MatrixView<T, MatrixFormat::Dense>& C, T alpha, T beta,
                                       Transpose transA, Transpose transB);

// Every scalar; max(m, n, k) <= 64, or 32 for complex<double> (ops::gemm::small_max_dim).
template <typename T>
BATCHLAS_INTERNAL_API Event gemm_small(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A,
                                       const MatrixView<T, MatrixFormat::Dense>& B,
                                       const MatrixView<T, MatrixFormat::Dense>& C, T alpha, T beta,
                                       Transpose transA, Transpose transB);

// float only; (tm, tn, tk, u) must be a row of ops::gemm::reg_configs instantiated for the form.
BATCHLAS_INTERNAL_API Event gemm_reg(Queue& ctx, int tm, int tn, int tk, int u,
                                     const MatrixView<float, MatrixFormat::Dense>& A,
                                     const MatrixView<float, MatrixFormat::Dense>& B,
                                     const MatrixView<float, MatrixFormat::Dense>& C, float alpha, float beta,
                                     Transpose transA, Transpose transB);

// (tm, tn, tk) must be a row of ops::gemm::wide_configs instantiated for the form.
template <typename T>
BATCHLAS_INTERNAL_API Event gemm_wide(Queue& ctx, int tm, int tn, int tk, const MatrixView<T, MatrixFormat::Dense>& A,
                                      const MatrixView<T, MatrixFormat::Dense>& B,
                                      const MatrixView<T, MatrixFormat::Dense>& C, T alpha, T beta,
                                      Transpose transA, Transpose transB);

// (tm, tn, tk) must be a split-k host row (ops::gemm::splitk_host) instantiated for the form and
// splits one of ops::gemm::kSplitKSlices; the partials are leased from ctx's workspace arena.
template <typename T>
BATCHLAS_INTERNAL_API Event gemm_splitk(Queue& ctx, int tm, int tn, int tk, int splits,
                                        const MatrixView<T, MatrixFormat::Dense>& A,
                                        const MatrixView<T, MatrixFormat::Dense>& B,
                                        const MatrixView<T, MatrixFormat::Dense>& C, T alpha, T beta,
                                        Transpose transA, Transpose transB);

} // namespace batchlas::sycl_gemm
