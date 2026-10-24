// The definitions behind gemm_launch.hh's row declarations, and the macros a shard TU
// instantiates them with. Include it only from a shard TU: a row instantiated twice is a
// duplicate-symbol link error, a row instantiated nowhere an undefined symbol at load.
#pragma once

#include <src/ops/gemm/kernels/gemm_launch.hh>

#include <optional>

namespace batchlas::sycl_gemm {

template <typename T, int I, bool Split, int Half>
void gemm_wide_row(std::optional<Event>& out, Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A,
                   const MatrixView<T, MatrixFormat::Dense>& B,
                   const MatrixView<T, MatrixFormat::Dense>& C, T alpha, T beta, Transpose transA, Transpose transB,
                   WideSplit split) {
    static_for<9>([&](auto F) {
        if constexpr ((decltype(F)::value < 5) == (Half == 0))
            launch_wide_form<T, I, decltype(F)::value, Split>(out, ctx, A, B, C, alpha, beta, transA, transB, split);
    });
}

template <int I>
void gemm_reg_row(std::optional<Event>& out, Queue& ctx, const MatrixView<float, MatrixFormat::Dense>& A,
                  const MatrixView<float, MatrixFormat::Dense>& B,
                  const MatrixView<float, MatrixFormat::Dense>& C, float alpha, float beta, bool ta, bool tb) {
    constexpr og::RegCfg c = og::reg_configs[I];
    if constexpr (c.forms.nn)
        if (!ta && !tb) out = launch_reg_cfg<I, kNN>(ctx, A, B, C, alpha, beta);
    if constexpr (c.forms.nt)
        if (!ta && tb) out = launch_reg_cfg<I, kNT>(ctx, A, B, C, alpha, beta);
    if constexpr (c.forms.tn)
        if (ta && !tb) out = launch_reg_cfg<I, kTN>(ctx, A, B, C, alpha, beta);
    if constexpr (c.forms.tt)
        if (ta && tb) out = launch_reg_cfg<I, kTT>(ctx, A, B, C, alpha, beta);
}

} // namespace batchlas::sycl_gemm

#define BATCHLAS_GEMM_WIDE_ROW_SPLIT(T, I, S, H)                                                       \
    template void batchlas::sycl_gemm::gemm_wide_row<T, I, S, H>(                                      \
        std::optional<batchlas::Event>&, batchlas::Queue&, const batchlas::MatrixView<T, batchlas::MatrixFormat::Dense>&, \
        const batchlas::MatrixView<T, batchlas::MatrixFormat::Dense>&,                                 \
        const batchlas::MatrixView<T, batchlas::MatrixFormat::Dense>&, T, T, batchlas::Transpose,      \
        batchlas::Transpose, batchlas::sycl_gemm::WideSplit);
#define BATCHLAS_GEMM_WIDE_HALF(T, I, H) BATCHLAS_GEMM_WIDE_ROW_SPLIT(T, I, false, H) BATCHLAS_GEMM_WIDE_ROW_SPLIT(T, I, true, H)
#define BATCHLAS_GEMM_WIDE_ROW(T, I) BATCHLAS_GEMM_WIDE_HALF(T, I, 0) BATCHLAS_GEMM_WIDE_HALF(T, I, 1)
#define BATCHLAS_GEMM_REG_ROW(I)                                                                       \
    template void batchlas::sycl_gemm::gemm_reg_row<I>(                                                \
        std::optional<batchlas::Event>&, batchlas::Queue&,                                             \
        const batchlas::MatrixView<float, batchlas::MatrixFormat::Dense>&,                            \
        const batchlas::MatrixView<float, batchlas::MatrixFormat::Dense>&,                            \
        const batchlas::MatrixView<float, batchlas::MatrixFormat::Dense>&, float, float, bool, bool);
