// The gemm kernels' launchers, one per family of src/ops/gemm/choice.hh. Which family runs is
// decided only in src/ops/gemm/gemm.cc; nothing here selects, and nothing falls back to another
// kernel: a launcher handed a form or config it does not instantiate throws.

#include "gemm_kernels.hh"

#include "gemm/accessors.hh"
#include "gemm/register_128x128.hh"
#include "gemm/register_64x64_k16_wide.hh"
#include "gemm/register_launchers.hh"
#include "gemm/register_wide_transposed.hh"
#include "gemm/small_batched.hh"
#include "gemm/tiled_general.hh"

#include "../linalg-impl.hh"
#include "../ops/gemm/choice.hh"
#include "../queue.hh"

#include <algorithm>
#include <optional>
#include <string>
#include <sycl/sycl.hpp>

namespace batchlas::sycl_gemm {

namespace {

namespace og = ::batchlas::ops::gemm;

static_assert(og::kSmallMaxDim == sycl_gemm_small::kSmallMaxDim && og::kSmallWg == sycl_gemm_small::kSmallWg &&
                  og::kSmallTiledMaxDim == sycl_gemm_small::kSmallTiledMaxDim,
              "choice.hh's small limits must match small_batched.hh");

template <typename T>
class GemmDirectKernel;

template <typename T>
inline int ceil_div(int value, int divisor) {
    return (value + divisor - 1) / divisor;
}

inline const char* kernel_trace_name(KernelVariant variant) {
    switch (variant) {
    case KernelVariant::Direct: return "gemm_sycl_direct";
    case KernelVariant::Tiled16: return "gemm_sycl_tiled16";
    case KernelVariant::Tiled32x32Register: return "gemm_sycl_register_32x32";
    case KernelVariant::Tiled64x64Register: return "gemm_sycl_register_64x64";
    case KernelVariant::Tiled64x64RegisterK16: return "gemm_sycl_register_64x64_k16";
    case KernelVariant::Tiled64x64RegisterK16TN: return "gemm_sycl_register_64x64_k16_tn";
    case KernelVariant::Tiled64x64RegisterK16NT: return "gemm_sycl_register_64x64_k16_nt";
    case KernelVariant::Tiled64x64RegisterK16TT: return "gemm_sycl_register_64x64_k16_tt";
    case KernelVariant::Tiled128x32RegisterK16: return "gemm_sycl_register_128x32_k16";
    case KernelVariant::Tiled128x32RegisterK16TN: return "gemm_sycl_register_128x32_k16_tn";
    case KernelVariant::Tiled128x32RegisterK16NT: return "gemm_sycl_register_128x32_k16_nt";
    case KernelVariant::Tiled128x32RegisterK16TT: return "gemm_sycl_register_128x32_k16_tt";
    case KernelVariant::Tiled128x32RegisterK32TN: return "gemm_sycl_register_128x32_k32_tn";
    case KernelVariant::Tiled128x32RegisterK32NT: return "gemm_sycl_register_128x32_k32_nt";
    case KernelVariant::Tiled128x32RegisterK32TT: return "gemm_sycl_register_128x32_k32_tt";
    case KernelVariant::Tiled128x64RegisterK16TN: return "gemm_sycl_register_128x64_k16_tn";
    case KernelVariant::Tiled128x64RegisterK16NT: return "gemm_sycl_register_128x64_k16_nt";
    case KernelVariant::Tiled128x64RegisterK16TT: return "gemm_sycl_register_128x64_k16_tt";
    case KernelVariant::Tiled128x32RegisterK32S2U1Aligned: return "gemm_sycl_register_128x32_k32_s2_u1_aligned";
    case KernelVariant::Tiled128x32RegisterK32S2U1Generic: return "gemm_sycl_register_128x32_k32_s2_u1_generic";
    case KernelVariant::Tiled128x64RegisterK32Large: return "gemm_sycl_register_128x64_k32_large";
    case KernelVariant::Tiled128x64RegisterK32LargeU2: return "gemm_sycl_register_128x64_k32_large_u2";
    case KernelVariant::Tiled128x128RegisterK8: return "gemm_sycl_register_128x128_k8";
    case KernelVariant::Tiled64x64RegisterK16Wide: return "gemm_sycl_register_64x64_k16_wide";
    case KernelVariant::Tiled64x64RegisterK16WideCN: return "gemm_sycl_register_64x64_k16_wide_cn";
    case KernelVariant::Tiled64x64RegisterK16WideNC: return "gemm_sycl_register_64x64_k16_wide_nc";
    case KernelVariant::Tiled128x32RegisterK16WideNC: return "gemm_sycl_register_128x32_k16_wide_nc";
    case KernelVariant::Tiled32x128RegisterK16WideCN: return "gemm_sycl_register_32x128_k16_wide_cn";
    case KernelVariant::Tiled32x128RegisterK16: return "gemm_sycl_register_32x128_k16";
    case KernelVariant::Tiled32x128RegisterK16TN: return "gemm_sycl_register_32x128_k16_tn";
    case KernelVariant::Tiled32x128RegisterK16TT: return "gemm_sycl_register_32x128_k16_tt";
    case KernelVariant::SmallBatched: return "gemm_sycl_small_batched";
    case KernelVariant::Tiled32x32RegisterK16Wide: return "gemm_sycl_register_32x32_k16_wide";
    case KernelVariant::Tiled16x16RegisterK16Wide: return "gemm_sycl_register_16x16_k16_wide";
    }
    return "gemm_sycl_unknown";
}

template <typename T>
Event launch_direct(Queue& ctx,
                    const MatrixView<T, MatrixFormat::Dense>& A,
                    const MatrixView<T, MatrixFormat::Dense>& B,
                    const MatrixView<T, MatrixFormat::Dense>& C,
                    T alpha,
                    T beta,
                    Transpose transA,
                    Transpose transB) {
    BATCHLAS_KERNEL_TRACE_SCOPE("gemm_sycl_direct");

    const auto [m, k] = get_effective_dims(A, transA);
    const auto [_, n] = get_effective_dims(B, transB);
    static_cast<void>(_);
    constexpr int workgroup = 8;

    const sycl::range<3> local(1, workgroup, workgroup);
    const sycl::range<3> global(static_cast<size_t>(A.batch_size()),
                                static_cast<size_t>(ceil_div<T>(m, workgroup) * workgroup),
                                static_cast<size_t>(ceil_div<T>(n, workgroup) * workgroup));

    ctx->submit([&](sycl::handler& h) {
        const T* a_ptr = A.data_ptr();
        const T* b_ptr = B.data_ptr();
        T* c_ptr = C.data_ptr();
        const int lda = A.ld();
        const int ldb = B.ld();
        const int ldc = C.ld();
        const int stride_a = A.stride();
        const int stride_b = B.stride();
        const int stride_c = C.stride();
        const int batch = A.batch_size();
        const Transpose op_a = transA;
        const Transpose op_b = transB;

        h.parallel_for<GemmDirectKernel<T>>(sycl::nd_range<3>(global, local), [=](sycl::nd_item<3> item) {
            const int bid = static_cast<int>(item.get_group(0));
            const int row = static_cast<int>(item.get_global_id(1));
            const int col = static_cast<int>(item.get_global_id(2));
            if (bid >= batch || row >= m || col >= n) {
                return;
            }

            T sum = T(0);
            const int batch_a = bid * stride_a;
            const int batch_b = bid * stride_b;
            const int batch_c = bid * stride_c;
            for (int kk = 0; kk < k; ++kk) {
                const T a_val = operand_value(a_ptr, lda, batch_a, row, kk, op_a);
                const T b_val = operand_value(b_ptr, ldb, batch_b, kk, col, op_b);
                sum += a_val * b_val;
            }
            c_ptr[batch_c + col * ldc + row] = alpha * sum + beta * c_ptr[batch_c + col * ldc + row];
        });
    });

    return ctx.get_event();
}

template <typename T, int Tile>
Event launch_tiled(Queue& ctx,
                   const MatrixView<T, MatrixFormat::Dense>& A,
                   const MatrixView<T, MatrixFormat::Dense>& B,
                   const MatrixView<T, MatrixFormat::Dense>& C,
                   T alpha,
                   T beta,
                   Transpose transA,
                   Transpose transB) {
    if (transA == Transpose::NoTrans && transB == Transpose::NoTrans) {
        return launch_tiled_general<T, Tile, Transpose::NoTrans, Transpose::NoTrans>(
            ctx, A, B, C, alpha, beta, kernel_trace_name);
    }
    if (transA == Transpose::NoTrans && transB == Transpose::Trans) {
        return launch_tiled_general<T, Tile, Transpose::NoTrans, Transpose::Trans>(
            ctx, A, B, C, alpha, beta, kernel_trace_name);
    }
    if (transA == Transpose::NoTrans && transB == Transpose::ConjTrans) {
        return launch_tiled_general<T, Tile, Transpose::NoTrans, Transpose::ConjTrans>(
            ctx, A, B, C, alpha, beta, kernel_trace_name);
    }
    if (transA == Transpose::Trans && transB == Transpose::NoTrans) {
        return launch_tiled_general<T, Tile, Transpose::Trans, Transpose::NoTrans>(
            ctx, A, B, C, alpha, beta, kernel_trace_name);
    }
    if (transA == Transpose::Trans && transB == Transpose::Trans) {
        return launch_tiled_general<T, Tile, Transpose::Trans, Transpose::Trans>(
            ctx, A, B, C, alpha, beta, kernel_trace_name);
    }
    if (transA == Transpose::Trans && transB == Transpose::ConjTrans) {
        return launch_tiled_general<T, Tile, Transpose::Trans, Transpose::ConjTrans>(
            ctx, A, B, C, alpha, beta, kernel_trace_name);
    }
    if (transA == Transpose::ConjTrans && transB == Transpose::NoTrans) {
        return launch_tiled_general<T, Tile, Transpose::ConjTrans, Transpose::NoTrans>(
            ctx, A, B, C, alpha, beta, kernel_trace_name);
    }
    if (transA == Transpose::ConjTrans && transB == Transpose::Trans) {
        return launch_tiled_general<T, Tile, Transpose::ConjTrans, Transpose::Trans>(
            ctx, A, B, C, alpha, beta, kernel_trace_name);
    }

    return launch_tiled_general<T, Tile, Transpose::ConjTrans, Transpose::ConjTrans>(
        ctx, A, B, C, alpha, beta, kernel_trace_name);
}


// ---- register-tiled (float) ------------------------------------------------------------

enum Form { kNN, kNT, kTN, kTT };

// The trace identity of reg_configs[I] at a form: every (config, form) pair the table
// instantiates is one of the old named variants. `aligned` names the NN unpredicated leg.
constexpr KernelVariant reg_variant(std::size_t i, Form f, bool aligned) {
    using K = KernelVariant;
    switch (i) {
    case 0: return K::Tiled32x32Register;
    case 1: return K::Tiled64x64Register;
    case 2: return f == kNN ? K::Tiled64x64RegisterK16 : f == kTN ? K::Tiled64x64RegisterK16TN
                   : f == kNT ? K::Tiled64x64RegisterK16NT : K::Tiled64x64RegisterK16TT;
    case 3: return f == kNN ? K::Tiled128x32RegisterK16 : f == kTN ? K::Tiled128x32RegisterK16TN
                   : f == kNT ? K::Tiled128x32RegisterK16NT : K::Tiled128x32RegisterK16TT;
    case 4: return f == kNN ? (aligned ? K::Tiled128x32RegisterK32S2U1Aligned : K::Tiled128x32RegisterK32S2U1Generic)
                   : f == kTN ? K::Tiled128x32RegisterK32TN : f == kNT ? K::Tiled128x32RegisterK32NT
                              : K::Tiled128x32RegisterK32TT;
    case 5: return f == kTN ? K::Tiled128x64RegisterK16TN : f == kNT ? K::Tiled128x64RegisterK16NT
                            : K::Tiled128x64RegisterK16TT;
    case 6: return f == kNN ? K::Tiled32x128RegisterK16 : f == kTN ? K::Tiled32x128RegisterK16TN
                            : K::Tiled32x128RegisterK16TT;
    case 7: return K::Tiled128x64RegisterK32Large;
    case 8: return K::Tiled128x64RegisterK32LargeU2;
    default: return K::Tiled128x128RegisterK8;
    }
}

template <int I, Form F>
Event launch_reg_cfg(Queue& ctx, const MatrixView<float, MatrixFormat::Dense>& A,
                     const MatrixView<float, MatrixFormat::Dense>& B, const MatrixView<float, MatrixFormat::Dense>& C,
                     float alpha, float beta) {
    constexpr og::RegCfg c = og::reg_configs[I];
    constexpr Transpose OA = (F == kTN || F == kTT) ? Transpose::Trans : Transpose::NoTrans;
    constexpr Transpose OB = (F == kNT || F == kTT) ? Transpose::Trans : Transpose::NoTrans;
    if constexpr (c.m == 128 && c.n == 128) {
        static_assert(F == kNN && c.threads() == 256, "the 128x128 kernel is NN only, 256 threads");
        // The leg is derived: the unpredicated path whenever the layout allows it.
        if (can_use_128x128_fast_path<float>(A, B, C))
            return launch_register_128x128_k8<float, true>(ctx, A, B, C, alpha, beta, kernel_trace_name);
        return launch_register_128x128_k8<float, false>(ctx, A, B, C, alpha, beta, kernel_trace_name);
    } else {
        constexpr bool aligned_leg = F == kNN && c.aligned_leg;
        constexpr RegTile P{c.m, c.n, c.k, c.tr, c.tc, 4, 4, c.u, c.stages, OA, OB, aligned_leg};
        static_assert(RegisterTilePolicy<P.M, P.N, P.K, P.TR, P.TC>::ThreadsPerGroup == c.threads(),
                      "choice.hh's thread count must match the tile");
        return launch_reg<float, P>(ctx, A, B, C, alpha, beta, kernel_trace_name(reg_variant(I, F, false)),
                                    aligned_leg ? kernel_trace_name(reg_variant(I, F, true)) : nullptr);
    }
}

// ---- wide (every scalar) ---------------------------------------------------------------

constexpr KernelVariant wide_variant(std::size_t i, char form) {
    using K = KernelVariant;
    switch (i) {
    case 0: return form == 'N' ? K::Tiled64x64RegisterK16Wide : form == 'A' ? K::Tiled64x64RegisterK16WideCN
                                                                            : K::Tiled64x64RegisterK16WideNC;
    case 1: return K::Tiled128x32RegisterK16WideNC;
    case 2: return K::Tiled32x128RegisterK16WideCN;
    case 3: return K::Tiled32x32RegisterK16Wide;
    default: return K::Tiled16x16RegisterK16Wide;
    }
}

// form: 'N' = NN, 'A' = ConjTrans A (CN), 'B' = ConjTrans B (NC).
template <typename T, int I, char Fm>
Event launch_wide_cfg(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A,
                      const MatrixView<T, MatrixFormat::Dense>& B, const MatrixView<T, MatrixFormat::Dense>& C,
                      T alpha, T beta) {
    constexpr og::WideCfg c = og::wide_configs[I];
    if constexpr (Fm == 'N' && c.m == 64 && c.n == 64) {
        // The NN 64x64 tile is its own kernel; its leg is derived from the layout.
        if (can_use_64x64_k16_wide_fast_path<T>(A, B, C))
            return launch_register_64x64_k16_wide<T, true>(ctx, A, B, C, alpha, beta, kernel_trace_name);
        return launch_register_64x64_k16_wide<T, false>(ctx, A, B, C, alpha, beta, kernel_trace_name);
    } else {
        constexpr Transpose OA = Fm == 'A' ? Transpose::ConjTrans : Transpose::NoTrans;
        constexpr Transpose OB = Fm == 'B' ? Transpose::ConjTrans : Transpose::NoTrans;
        return launch_wide_transposed<T, WideTile{c.m, c.n, c.k, c.ttm, c.ttn, OA, OB}>(
            ctx, A, B, C, alpha, beta, kernel_trace_name(wide_variant(I, Fm)));
    }
}

[[noreturn]] void refuse(const std::string& what) {
    throw batchlas::unsupported("gemm: " + what + " (the choice's can_run should have refused it)");
}

} // namespace

template <typename T>
Event gemm_direct(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, const MatrixView<T, MatrixFormat::Dense>& B,
                  const MatrixView<T, MatrixFormat::Dense>& C, T alpha, T beta, Transpose transA, Transpose transB) {
    return launch_direct<T>(ctx, A, B, C, alpha, beta, transA, transB);
}

template <typename T>
Event gemm_tiled(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, const MatrixView<T, MatrixFormat::Dense>& B,
                 const MatrixView<T, MatrixFormat::Dense>& C, T alpha, T beta, Transpose transA, Transpose transB) {
    return launch_tiled<T, 16>(ctx, A, B, C, alpha, beta, transA, transB);
}

template <typename T>
Event gemm_small(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, const MatrixView<T, MatrixFormat::Dense>& B,
                 const MatrixView<T, MatrixFormat::Dense>& C, T alpha, T beta, Transpose transA, Transpose transB) {
    const auto [m, k] = get_effective_dims(A, transA);
    const auto [k_b, n] = get_effective_dims(B, transB);
    static_cast<void>(k_b);
    if (std::max({m, n, k}) > sycl_gemm_small::kSmallMaxDim) refuse("small serves max(m, n, k) <= 64");
    BATCHLAS_KERNEL_TRACE_SCOPE("gemm_sycl_small_batched");
    return sycl_gemm_small::small_batched<T>(ctx, A, B, C, alpha, beta, transA, transB, m, n, k);
}

Event gemm_reg(Queue& ctx, int tm, int tn, int tk, int u, const MatrixView<float, MatrixFormat::Dense>& A,
               const MatrixView<float, MatrixFormat::Dense>& B, const MatrixView<float, MatrixFormat::Dense>& C,
               float alpha, float beta, Transpose transA, Transpose transB) {
    // A real scalar's ConjTrans is its Trans, so the forms fold C -> T.
    const bool ta = transA != Transpose::NoTrans, tb = transB != Transpose::NoTrans;
    std::optional<Event> out;
    static_for<static_cast<int>(og::reg_configs.size())>([&](auto I) {
        constexpr og::RegCfg c = og::reg_configs[I];
        if (out || c.m != tm || c.n != tn || c.k != tk || c.u != u) return;
        if constexpr (c.forms.nn)
            if (!ta && !tb) out = launch_reg_cfg<I, kNN>(ctx, A, B, C, alpha, beta);
        if constexpr (c.forms.nt)
            if (!ta && tb) out = launch_reg_cfg<I, kNT>(ctx, A, B, C, alpha, beta);
        if constexpr (c.forms.tn)
            if (ta && !tb) out = launch_reg_cfg<I, kTN>(ctx, A, B, C, alpha, beta);
        if constexpr (c.forms.tt)
            if (ta && tb) out = launch_reg_cfg<I, kTT>(ctx, A, B, C, alpha, beta);
    });
    if (!out)
        refuse("reg:m=" + std::to_string(tm) + ":n=" + std::to_string(tn) + ":k=" + std::to_string(tk) +
               ":u=" + std::to_string(u) + " has no instantiation for this transpose form");
    return std::move(*out);
}

template <typename T>
Event gemm_wide(Queue& ctx, int tm, int tn, int tk, const MatrixView<T, MatrixFormat::Dense>& A,
                const MatrixView<T, MatrixFormat::Dense>& B, const MatrixView<T, MatrixFormat::Dense>& C, T alpha,
                T beta, Transpose transA, Transpose transB) {
    const bool nn = transA == Transpose::NoTrans && transB == Transpose::NoTrans;
    const bool cn = wide_trans_matches<T>(transA, Transpose::ConjTrans) && transB == Transpose::NoTrans;
    const bool nc = transA == Transpose::NoTrans && wide_trans_matches<T>(transB, Transpose::ConjTrans);
    std::optional<Event> out;
    static_for<static_cast<int>(og::wide_configs.size())>([&](auto I) {
        constexpr og::WideCfg c = og::wide_configs[I];
        if (out || c.m != tm || c.n != tn || c.k != tk) return;
        if constexpr (c.nn)
            if (nn) out = launch_wide_cfg<T, I, 'N'>(ctx, A, B, C, alpha, beta);
        if constexpr (c.cn)
            if (cn) out = launch_wide_cfg<T, I, 'A'>(ctx, A, B, C, alpha, beta);
        if constexpr (c.nc)
            if (nc) out = launch_wide_cfg<T, I, 'B'>(ctx, A, B, C, alpha, beta);
    });
    if (!out)
        refuse("wide:m=" + std::to_string(tm) + ":n=" + std::to_string(tn) + ":k=" + std::to_string(tk) +
               " has no instantiation for this transpose form");
    return std::move(*out);
}

#define GEMM_LAUNCHERS(T)                                                                                       \
    template Event gemm_direct<T>(Queue&, const MatrixView<T, MatrixFormat::Dense>&,                           \
                                  const MatrixView<T, MatrixFormat::Dense>&, const MatrixView<T, MatrixFormat::Dense>&, \
                                  T, T, Transpose, Transpose);                                                  \
    template Event gemm_tiled<T>(Queue&, const MatrixView<T, MatrixFormat::Dense>&,                            \
                                 const MatrixView<T, MatrixFormat::Dense>&, const MatrixView<T, MatrixFormat::Dense>&, \
                                 T, T, Transpose, Transpose);                                                   \
    template Event gemm_wide<T>(Queue&, int, int, int, const MatrixView<T, MatrixFormat::Dense>&,             \
                                const MatrixView<T, MatrixFormat::Dense>&, const MatrixView<T, MatrixFormat::Dense>&, \
                                T, T, Transpose, Transpose);

GEMM_LAUNCHERS(float)
GEMM_LAUNCHERS(double)
GEMM_LAUNCHERS(std::complex<float>)
GEMM_LAUNCHERS(std::complex<double>)
#undef GEMM_LAUNCHERS

template Event gemm_small<float>(Queue&, const MatrixView<float, MatrixFormat::Dense>&,
                                 const MatrixView<float, MatrixFormat::Dense>&,
                                 const MatrixView<float, MatrixFormat::Dense>&, float, float, Transpose, Transpose);
template Event gemm_small<double>(Queue&, const MatrixView<double, MatrixFormat::Dense>&,
                                  const MatrixView<double, MatrixFormat::Dense>&,
                                  const MatrixView<double, MatrixFormat::Dense>&, double, double, Transpose,
                                  Transpose);

} // namespace batchlas::sycl_gemm
