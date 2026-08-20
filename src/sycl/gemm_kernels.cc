#include "gemm_kernels.hh"

#include "gemm/accessors.hh"
#include "gemm/persistent.hh"
#include "gemm/register_128x128.hh"
#include "gemm/register_64x64_k16_wide.hh"
#include "gemm/register_launchers.hh"
#include "gemm/split_k.hh"
#include "gemm/tiled_general.hh"

#include "../linalg-impl.hh"
#include "../queue.hh"

#include <algorithm>
#include <cstdlib>
#include <string>
#include <sycl/sycl.hpp>

namespace batchlas::sycl_gemm {

namespace {

inline bool experimental_kernel_variants_enabled() {
    const char* raw = std::getenv("BATCHLAS_GEMM_EXPERIMENTAL");
    if (!raw) {
        return false;
    }

    std::string value(raw);
    for (char& ch : value) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }

    return value == "1" || value == "true" || value == "on" || value == "yes";
}

inline bool is_experimental_kernel_variant(KernelVariant variant) {
    switch (variant) {
    case KernelVariant::Tiled128x32RegisterK32Persistent:
    case KernelVariant::Tiled128x32RegisterK32SplitK4:
    case KernelVariant::Tiled128x32RegisterK32S1U4:
    case KernelVariant::Tiled128x64RegisterK32LargeTT4x8:
    case KernelVariant::Tiled128x64RegisterK32LargeTT4x8U2:
        return true;
    default:
        return false;
    }
}

inline bool has_forced_kernel_variant();

inline bool is_squareish_shape(int m, int n, int k) {
    const int max_dim = std::max({m, n, k});
    const int min_dim = std::min({m, n, k});
    return min_dim * 2 >= max_dim;
}

inline bool is_large_square_bucket(int m, int n, int k) {
    return is_squareish_shape(m, n, k) && std::max({m, n, k}) >= 512 && std::min({m, n, k}) >= 256;
}

inline bool is_full_512_square_bucket(int m, int n, int k) {
    return std::min({m, n, k}) >= 512;
}

template <typename T>
KernelVariant choose_runtime_kernel_variant(const Queue& ctx,
                                           const MatrixView<T, MatrixFormat::Dense>& A,
                                           const MatrixView<T, MatrixFormat::Dense>& B,
                                           const MatrixView<T, MatrixFormat::Dense>& C,
                                           Transpose transA,
                                           Transpose transB) {
    static_cast<void>(ctx);
    const KernelVariant selected = select_kernel_variant(A, B, C, transA, transB);
    if (has_forced_kernel_variant()) {
        return selected;
    }

    return selected;
}

template <typename T>
class GemmDirectKernel;

template <typename T, int Tile>
class GemmTiledKernel;

template <typename T, int TileM, int TileN, int TileK, int WorkPerThread>
class GemmRegisterTiledKernel;

template <typename T>
inline int ceil_div(int value, int divisor) {
    return (value + divisor - 1) / divisor;
}

inline const char* kernel_trace_name(KernelVariant variant) {
    switch (variant) {
    case KernelVariant::Direct:
        return "gemm_sycl_direct";
    case KernelVariant::Tiled16:
        return "gemm_sycl_tiled16";
    case KernelVariant::Tiled32x32Register:
        return "gemm_sycl_register_32x32";
    case KernelVariant::Tiled64x64Register:
        return "gemm_sycl_register_64x64";
    case KernelVariant::Tiled64x64RegisterK16:
        return "gemm_sycl_register_64x64_k16";
    // GUARD: these launchers hard-wire OpA/OpB (register_launchers.hh), so an
    // unguarded case computes the WRONG ANSWER for any transpose combination
    // other than the one it was instantiated for -- ConjTrans most of all, which
    // silently drops the conjugation. Transpose::ConjTrans is a distinct enum
    // value (enums.hh: NoTrans=0, Trans=1, ConjTrans=2), so `Trans` does not
    // cover it. The Tiled128x32RegisterK16 family below already had this guard;
    // nine other variants did not. Not reachable from select_kernel_variant
    // today, but all of them are FORCEABLE by name via
    // BATCHLAS_GEMM_SYCL_KERNEL, which is exactly how a benchmark would compare
    // them -- producing a valid-looking timing for an incorrect result.
    case KernelVariant::Tiled64x64RegisterK16TN:
        return "gemm_sycl_register_64x64_k16_tn";
    case KernelVariant::Tiled64x64RegisterK16NT:
        return "gemm_sycl_register_64x64_k16_nt";
    case KernelVariant::Tiled64x64RegisterK16TT:
        return "gemm_sycl_register_64x64_k16_tt";
    case KernelVariant::Tiled128x32RegisterK16:
        return "gemm_sycl_register_128x32_k16";
    case KernelVariant::Tiled128x32RegisterK16TN:
        return "gemm_sycl_register_128x32_k16_tn";
    case KernelVariant::Tiled128x32RegisterK16NT:
        return "gemm_sycl_register_128x32_k16_nt";
    case KernelVariant::Tiled128x32RegisterK16TT:
        return "gemm_sycl_register_128x32_k16_tt";
    case KernelVariant::Tiled128x32RegisterK32TN:
        return "gemm_sycl_register_128x32_k32_tn";
    case KernelVariant::Tiled128x32RegisterK32NT:
        return "gemm_sycl_register_128x32_k32_nt";
    case KernelVariant::Tiled128x32RegisterK32TT:
        return "gemm_sycl_register_128x32_k32_tt";
    case KernelVariant::Tiled128x64RegisterK16TN:
        return "gemm_sycl_register_128x64_k16_tn";
    case KernelVariant::Tiled128x64RegisterK16NT:
        return "gemm_sycl_register_128x64_k16_nt";
    case KernelVariant::Tiled128x64RegisterK16TT:
        return "gemm_sycl_register_128x64_k16_tt";
    case KernelVariant::Tiled128x32RegisterK32:
        return "gemm_sycl_register_128x32_k32";
    case KernelVariant::Tiled128x32RegisterK32S1U1:
        return "gemm_sycl_register_128x32_k32_s1_u1";
    case KernelVariant::Tiled128x32RegisterK32S2U1:
        return "gemm_sycl_register_128x32_k32_s2_u1";
    case KernelVariant::Tiled128x32RegisterK32S2U1Aligned:
        return "gemm_sycl_register_128x32_k32_s2_u1_aligned";
    case KernelVariant::Tiled128x32RegisterK32S2U1Generic:
        return "gemm_sycl_register_128x32_k32_s2_u1_generic";
    case KernelVariant::Tiled128x32RegisterK32S2U2:
        return "gemm_sycl_register_128x32_k32_s2_u2";
    case KernelVariant::Tiled128x32RegisterK32S2U2TT8x4:
        return "gemm_sycl_register_128x32_k32_s2_u2_tt8x4";
    case KernelVariant::Tiled128x32RegisterK32S2U2TT4x8:
        return "gemm_sycl_register_128x32_k32_s2_u2_tt4x8";
    case KernelVariant::Tiled128x32RegisterK32Persistent:
        return "gemm_sycl_register_128x32_k32_persistent";
    case KernelVariant::Tiled128x32RegisterK32SplitK4:
        return "gemm_sycl_register_128x32_k32_splitk4";
    case KernelVariant::Tiled128x32RegisterK32S1U4:
        return "gemm_sycl_register_128x32_k32_s1_u4";
    case KernelVariant::Tiled128x64RegisterK32Large:
        return "gemm_sycl_register_128x64_k32_large";
    case KernelVariant::Tiled128x64RegisterK32LargeU2:
        return "gemm_sycl_register_128x64_k32_large_u2";
    case KernelVariant::Tiled128x64RegisterK32LargeTT4x8:
        return "gemm_sycl_register_128x64_k32_large_tt4x8";
    case KernelVariant::Tiled128x64RegisterK32LargeTT4x8U2:
        return "gemm_sycl_register_128x64_k32_large_tt4x8_u2";
    case KernelVariant::Tiled128x128RegisterK8:
        return "gemm_sycl_register_128x128_k8";
    case KernelVariant::Tiled64x64RegisterK16Wide:
        return "gemm_sycl_register_64x64_k16_wide";
    case KernelVariant::Tiled32x128RegisterK16:
        return "gemm_sycl_register_32x128_k16";
    case KernelVariant::Tiled32x128RegisterK16TN:
        return "gemm_sycl_register_32x128_k16_tn";
    case KernelVariant::Tiled32x128RegisterK16TT:
        return "gemm_sycl_register_32x128_k16_tt";
    }

    return "gemm_sycl_unknown";
}

inline bool kernel_variant_matches_name(KernelVariant variant, const std::string& name) {
    switch (variant) {
    case KernelVariant::Direct:
        return name == "direct";
    case KernelVariant::Tiled16:
        return name == "tiled16" || name == "tile16";
    case KernelVariant::Tiled32x32Register:
        return name == "register32" || name == "reg32" || name == "32x32";
    case KernelVariant::Tiled64x64Register:
        return name == "register64" || name == "reg64" || name == "64x64";
    case KernelVariant::Tiled64x64RegisterK16:
        return name == "register64k16" || name == "reg64k16" || name == "64x64x16";
    case KernelVariant::Tiled64x64RegisterK16TN:
        return name == "register64k16tn" || name == "reg64k16tn" || name == "64x64x16tn";
    case KernelVariant::Tiled64x64RegisterK16NT:
        return name == "register64k16nt" || name == "reg64k16nt" || name == "64x64x16nt";
    case KernelVariant::Tiled64x64RegisterK16TT:
        return name == "register64k16tt" || name == "reg64k16tt" || name == "64x64x16tt";
    case KernelVariant::Tiled128x32RegisterK16:
        return name == "register128x32k16" || name == "reg128x32k16" || name == "128x32x16";
    case KernelVariant::Tiled128x32RegisterK16TN:
        return name == "register128x32k16tn" || name == "reg128x32k16tn" || name == "128x32x16tn";
    case KernelVariant::Tiled128x32RegisterK16NT:
        return name == "register128x32k16nt" || name == "reg128x32k16nt" || name == "128x32x16nt";
    case KernelVariant::Tiled128x32RegisterK16TT:
        return name == "register128x32k16tt" || name == "reg128x32k16tt" || name == "128x32x16tt";
    case KernelVariant::Tiled128x32RegisterK32TN:
        return name == "register128x32k32tn" || name == "reg128x32k32tn" || name == "128x32x32tn" ||
            name == "128x32x32_s2_u1_tn";
    case KernelVariant::Tiled128x32RegisterK32NT:
        return name == "register128x32k32nt" || name == "reg128x32k32nt" || name == "128x32x32nt" ||
            name == "128x32x32_s2_u1_nt";
    case KernelVariant::Tiled128x32RegisterK32TT:
        return name == "register128x32k32tt" || name == "reg128x32k32tt" || name == "128x32x32tt" ||
            name == "128x32x32_s2_u1_tt";
    case KernelVariant::Tiled128x64RegisterK16TN:
        return name == "register128x64k16tn" || name == "reg128x64k16tn" || name == "128x64x16tn";
    case KernelVariant::Tiled128x64RegisterK16NT:
        return name == "register128x64k16nt" || name == "reg128x64k16nt" || name == "128x64x16nt";
    case KernelVariant::Tiled128x64RegisterK16TT:
        return name == "register128x64k16tt" || name == "reg128x64k16tt" || name == "128x64x16tt";
    case KernelVariant::Tiled128x32RegisterK32:
        return false;
    case KernelVariant::Tiled128x32RegisterK32S1U1:
        return name == "register128x32k32s1u1" || name == "reg128x32k32s1u1" || name == "128x32x32_s1_u1";
    case KernelVariant::Tiled128x32RegisterK32S2U1:
        return name == "register128x32k32" || name == "reg128x32k32" || name == "128x32x32" ||
            name == "register128x32k32s2u1" || name == "reg128x32k32s2u1" || name == "128x32x32_s2_u1";
    case KernelVariant::Tiled128x32RegisterK32S2U1Aligned:
        return name == "register128x32k32s2u1aligned" || name == "reg128x32k32s2u1aligned" ||
            name == "128x32x32_s2_u1_aligned";
    case KernelVariant::Tiled128x32RegisterK32S2U1Generic:
        return name == "register128x32k32s2u1generic" || name == "reg128x32k32s2u1generic" ||
            name == "128x32x32_s2_u1_generic";
    case KernelVariant::Tiled128x32RegisterK32S2U2:
        return name == "register128x32k32s2u2" || name == "reg128x32k32s2u2" || name == "128x32x32_s2_u2";
    case KernelVariant::Tiled128x32RegisterK32S2U2TT8x4:
        return name == "register128x32k32s2u2tt8x4" || name == "reg128x32k32s2u2tt8x4" ||
            name == "128x32x32_s2_u2_tt8x4";
    case KernelVariant::Tiled128x32RegisterK32S2U2TT4x8:
        return name == "register128x32k32s2u2tt4x8" || name == "reg128x32k32s2u2tt4x8" ||
            name == "128x32x32_s2_u2_tt4x8";
    case KernelVariant::Tiled128x32RegisterK32Persistent:
        return name == "register128x32k32persistent" || name == "reg128x32k32persistent" ||
            name == "128x32x32_persistent";
    case KernelVariant::Tiled128x32RegisterK32SplitK4:
        return name == "register128x32k32splitk4" || name == "reg128x32k32splitk4" ||
            name == "128x32x32_splitk4";
    case KernelVariant::Tiled128x32RegisterK32S1U4:
        return name == "register128x32k32s1u4" || name == "reg128x32k32s1u4" || name == "reg128x32k32u4" ||
            name == "128x32x32_s1_u4";
    case KernelVariant::Tiled128x64RegisterK32Large:
        return name == "register128x64k32large" || name == "reg128x64k32large" || name == "128x64x32large";
    case KernelVariant::Tiled128x64RegisterK32LargeU2:
        return name == "register128x64k32largeu2" || name == "reg128x64k32largeu2" || name == "128x64x32large_u2";
    case KernelVariant::Tiled128x64RegisterK32LargeTT4x8:
        return name == "register128x64k32largett4x8" || name == "reg128x64k32largett4x8" ||
            name == "128x64x32large_tt4x8";
    case KernelVariant::Tiled128x64RegisterK32LargeTT4x8U2:
        return name == "register128x64k32largett4x8u2" || name == "reg128x64k32largett4x8u2" ||
            name == "128x64x32large_tt4x8_u2";
    case KernelVariant::Tiled128x128RegisterK8:
        return name == "register128x128k8" || name == "reg128x128k8" || name == "128x128x8";
    case KernelVariant::Tiled64x64RegisterK16Wide:
        return name == "register64x64k16wide" || name == "reg64x64k16wide" ||
            name == "64x64x16wide";
    case KernelVariant::Tiled32x128RegisterK16:
        return name == "register32x128k16" || name == "reg32x128k16" || name == "32x128x16";
    case KernelVariant::Tiled32x128RegisterK16TN:
        return name == "register32x128k16tn" || name == "reg32x128k16tn" || name == "32x128x16tn";
    case KernelVariant::Tiled32x128RegisterK16TT:
        return name == "register32x128k16tt" || name == "reg32x128k16tt" || name == "32x128x16tt";
    }

    return false;
}

inline KernelVariant forced_kernel_variant() {
    const char* raw = std::getenv("BATCHLAS_GEMM_SYCL_KERNEL");
    if (!raw || raw[0] == '\0') {
        return KernelVariant::Direct;
    }

    std::string name(raw);
    for (char& ch : name) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }

    for (KernelVariant variant : {KernelVariant::Direct,
                                  KernelVariant::Tiled16,
                                  KernelVariant::Tiled32x32Register,
                                  KernelVariant::Tiled64x64Register,
                                  KernelVariant::Tiled64x64RegisterK16,
                                  KernelVariant::Tiled64x64RegisterK16TN,
                                  KernelVariant::Tiled64x64RegisterK16NT,
                                  KernelVariant::Tiled64x64RegisterK16TT,
                                  KernelVariant::Tiled128x32RegisterK16,
                                  KernelVariant::Tiled128x32RegisterK16TN,
                                  KernelVariant::Tiled128x32RegisterK16NT,
                                  KernelVariant::Tiled128x32RegisterK16TT,
                                  KernelVariant::Tiled128x32RegisterK32TN,
                                  KernelVariant::Tiled128x32RegisterK32NT,
                                  KernelVariant::Tiled128x32RegisterK32TT,
                                  KernelVariant::Tiled128x64RegisterK16TN,
                                  KernelVariant::Tiled128x64RegisterK16NT,
                                  KernelVariant::Tiled128x64RegisterK16TT,
                                  KernelVariant::Tiled128x32RegisterK32,
                                  KernelVariant::Tiled128x32RegisterK32S1U1,
                                  KernelVariant::Tiled128x32RegisterK32S2U1,
                                  KernelVariant::Tiled128x32RegisterK32S2U1Aligned,
                                  KernelVariant::Tiled128x32RegisterK32S2U1Generic,
                                  KernelVariant::Tiled128x32RegisterK32S2U2,
                                  KernelVariant::Tiled128x32RegisterK32S2U2TT8x4,
                                  KernelVariant::Tiled128x32RegisterK32S2U2TT4x8,
                                  KernelVariant::Tiled128x32RegisterK32Persistent,
                                  KernelVariant::Tiled128x32RegisterK32SplitK4,
                                  KernelVariant::Tiled128x32RegisterK32S1U4,
                                  KernelVariant::Tiled128x64RegisterK32Large,
                                  KernelVariant::Tiled128x64RegisterK32LargeU2,
                                  KernelVariant::Tiled128x64RegisterK32LargeTT4x8,
                                  KernelVariant::Tiled128x64RegisterK32LargeTT4x8U2,
                                  KernelVariant::Tiled128x128RegisterK8,
                                  KernelVariant::Tiled64x64RegisterK16Wide,
                                  KernelVariant::Tiled32x128RegisterK16,
                                  KernelVariant::Tiled32x128RegisterK16TN,
                                  KernelVariant::Tiled32x128RegisterK16TT}) {
        if (kernel_variant_matches_name(variant, name)) {
            return variant;
        }
    }

    return KernelVariant::Direct;
}

inline bool has_forced_kernel_variant() {
    const char* raw = std::getenv("BATCHLAS_GEMM_SYCL_KERNEL");
    return raw && raw[0] != '\0';
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

} // namespace

template <typename T>
KernelVariant select_kernel_variant(const MatrixView<T, MatrixFormat::Dense>& A,
                                    const MatrixView<T, MatrixFormat::Dense>& B,
                                    const MatrixView<T, MatrixFormat::Dense>& C,
                                    Transpose transA,
                                    Transpose transB) {
    static_cast<void>(C);
    if (has_forced_kernel_variant()) {
        return forced_kernel_variant();
    }
    const auto [m, k] = get_effective_dims(A, transA);
    const auto [_, n] = get_effective_dims(B, transB);
    static_cast<void>(_);
    const int max_dim = std::max({m, n, k});
    const int min_dim = std::min({m, n, k});
    if (transA != Transpose::NoTrans || transB != Transpose::NoTrans) {
        if constexpr (std::is_same_v<T, float>) {
            if (transA == Transpose::Trans && transB == Transpose::NoTrans && m >= 128 && n >= 32 && k >= 128) {
                return KernelVariant::Tiled128x32RegisterK32TN;
            }
            if (transA == Transpose::NoTrans && transB == Transpose::Trans && m >= 128 && n >= 32 && k >= 128) {
                return KernelVariant::Tiled128x32RegisterK32NT;
            }
            if (transA == Transpose::Trans && transB == Transpose::Trans && m >= 128 && n >= 32 && k >= 128) {
                return KernelVariant::Tiled128x32RegisterK32TT;
            }
        }
        return max_dim <= 32 ? KernelVariant::Direct : KernelVariant::Tiled16;
    }
    if constexpr (std::is_same_v<T, float>) {
        // Anything with a full 128x128 output tile, a deep enough k, and
        // operands the unpredicated path can use goes to the 64-accumulator
        // kernel. It beats the whole 128x32/128x64 family by 69-97% on every
        // shape in this bucket and lands at 88-102% of cuBLAS; see
        // experiments/sycl_vs_cuda/FINDINGS.md.
        //
        // This used to be gated on the fast path rather than on shape alone,
        // with the note: "the kernel's predicated path is correct for ragged
        // shapes but has not been benchmarked against the generic route below,
        // so misaligned work keeps its existing kernel until that measurement
        // exists." THAT MEASUREMENT NOW EXISTS (WP2 E4), and the predicated
        // path wins by a wide margin on every shape tried. Square NN float,
        // batch 512 (96 for n >= 544), both betas, RTX 4090:
        //
        //   n     generic 128x32x32   predicated 128x128   gain
        //   160          7 892              13 170        1.67x
        //   192          9 781              18 000        1.84x
        //   224         11 611              22 467        1.93x
        //   320         12 188              25 288        2.07x
        //   544         13 372              27 101        2.03x
        //   672         14 107              29 715        2.11x
        //   800         14 654              31 354        2.14x
        //  1056         15 065              33 314        2.21x
        //
        // The gain GROWS with n, which is what a predication cost that is
        // constant per tile looks like against a route whose throughput has
        // plateaued. Against cuBLAS this moves the bucket from 0.36-0.51x to
        // 0.72-0.84x -- still a loss, which is why preferred() no longer
        // claims this window for float, but it halves the damage for anyone on
        // a vendor-free build, who has no cuBLAS to fall back to.
        //
        // Scope, deliberately narrow: only the GENERIC leg changes. The
        // aligned leg below is a different tuned route and was never in the
        // measurement, so it stays. See experiments/wp2_e4/.
        if (m >= 128 && n >= 128 && k >= 128 && can_use_128x128_fast_path<T>(A, B, C)) {
            return KernelVariant::Tiled128x128RegisterK8;
        }
        if (m >= 128 && n >= 128 && k >= 128 && is_squareish_shape(m, n, k)) {
            if (can_use_aligned_nn_fast_path<T, 128, 32, 32, 4, 4>(A, B, C)) {
                if (is_large_square_bucket(m, n, k)) {
                    return is_full_512_square_bucket(m, n, k)
                        ? KernelVariant::Tiled128x64RegisterK32Large
                        : KernelVariant::Tiled128x64RegisterK32LargeU2;
                }
                return KernelVariant::Tiled128x32RegisterK32S2U1Aligned;
            }
            return KernelVariant::Tiled128x128RegisterK8;
        }
        // can_use_128x128_fast_path is a LEG predicate, not a KERNEL predicate.
        // The dispatcher at :737-741 evaluates the identical predicate again and
        // picks <true>/<false> itself, so a call that fails it still runs this
        // kernel -- on its predicated leg. Used as a ROUTING gate above (:509),
        // failing it did not demote the call to that leg; it handed the call to
        // an entirely different, much slower kernel:
        //
        //   1000x1024x128  auto -> register_128x32_k16   forced -> register_128x128_k8
        //   1024x1024x64   auto -> register_64x64        forced -> register_128x128_k8
        //   1024x1024x128  auto -> register_128x128_k8   forced -> same  (null control)
        //
        // Measured native-vs-native (forced 128x128 vs the route it replaces),
        // float NN beta=1, RTX 4090, 12 shapes, at pad 0 and pad 384:
        // geomean 1.74x / 1.75x. Native goes 0.58x -> 0.99x of cuBLAS at
        // ld==rows and 0.54x -> 0.93x strided.
        //   1024x1024x64  b128 ld1408  3.187 -> 1.337 ms  2.38x
        //   1000x1024x128 b128 ld1384  2.954 -> 1.569 ms  1.88x
        //   1024x1024x16  b128         2.622 -> 1.232 ms  2.13x
        //   128x128x8     b512         0.074 -> 0.030 ms  2.43x
        // This also subsumes the ld%4 != 0 cliff (pad 1: 1.874 -> 1.003 ms),
        // because this branch does not consult the alignment predicate at all.
        // See experiments/wp4_gemm_ld/routing/.
        //
        // EVERY BOUND BELOW HAS A MEASURED COUNTEREXAMPLE. Do not round them.
        //  * mn_min >= 64 : m=32 is a wash-to-loss (32x1024x32 0.97x).
        //  * mn_min >= 128 when k >= 128 : the grid at register_128x128.hh:124-130
        //    fixes a 128x128 output tile, so a 64-wide output wastes 2-4x of
        //    every CTA. At k >= 128 the routes this would displace
        //    (Tiled64x64RegisterK16 :529, Tiled32x128RegisterK16 :526) are tuned
        //    and WIN -- forced/auto at pad0 and pad384:
        //        64x64x512   b512  0.77 / 0.69   (forced 1.3-1.45x SLOWER)
        //        64x64x1024  b256  0.58 / 0.62   (1.6-1.7x SLOWER)
        //        64x1024x512 b256  0.64 / 0.62   (1.6x SLOWER)
        //    At k < 128 those routes are not tuned and 128x128 wins even at
        //    mn_min = 64: 1024x64x64 1.80x, 64x1024x64 1.59-1.81x.
        //  * max_dim >= 128 : 64x64x64 is a wash (1.02x).
        //  * k >= 8 : it is the kernel's TileK; 1024x1024x8 wins 2.00x.
        //
        // NOTE ON REACH: with cuBLAS present this changes no runtime at all --
        // route_gemm.hh's float NN window requires m==n==k, so every shape this
        // gate captures resolves to the vendor (coverage: 79 native float gemm
        // calls against 102 791 vendor). The deliverable is the vendor-free and
        // ROCm builds, and making a future preferred() flip arguable. At 0.93x
        // it is not yet arguable.
        const int mn_min = std::min(m, n);
        if (max_dim >= 128 && k >= 8 && mn_min >= 64 && (mn_min >= 128 || k < 128)) {
            return KernelVariant::Tiled128x128RegisterK8;
        }
        if (m >= 128 && n >= 32 && k >= 128) {
            return KernelVariant::Tiled128x32RegisterK16;
        }
        if (n >= 128 && m >= 32 && k >= 128) {
            return KernelVariant::Tiled32x128RegisterK16;
        }
        if (min_dim >= 64 && k >= 128) {
            return KernelVariant::Tiled64x64RegisterK16;
        }
        if (min_dim >= 64 && max_dim >= 128) {
            return KernelVariant::Tiled64x64Register;
        }
        if (min_dim >= 32 && max_dim >= 64) {
            return KernelVariant::Tiled32x32Register;
        }
        return max_dim <= 48 ? KernelVariant::Direct : KernelVariant::Tiled16;
    }

    // Wide scalars (double, complex<float>, complex<double>) have no register
    // kernel at all below this point: they fall off the float ladder above
    // straight to Tiled16 -- one accumulator per thread, std::complex
    // operator* (and its isnan branch plus __mulsc3 call) in the inner loop,
    // and a scattered epilogue. Measured on RTX 4090 / sm_89 at 256^3 b512,
    // 512^3 b128 and 1024^3 b32, at beta 0 and beta 1, against a standalone
    // replica of Tiled16 and against cuBLAS:
    //
    //   complex<float>  : 7.0-7.7x Tiled16, 0.98-1.08x cuBLAS CGEMM
    //   complex<double> : 3.56-3.60x Tiled16, 1.12x cuBLAS ZGEMM
    //   double          : 1.01-1.08x Tiled16, 1.07-1.15x cuBLAS DGEMM
    //
    // double is small on purpose: FP64 on a 4090 is 1/64 of FP32, the ceiling
    // is ~1.44 TFLOP/s, and Tiled16 already reaches 92% of it. Do not read the
    // double row as a win for the tile design; it is not, on this part.
    // See experiments/wide_scalar_gemm/measure/.
    //
    // Two gates, both deliberate and both conservative:
    //   * The unpredicated path only, exactly like the 128x128 float kernel
    //     above. The predicated path is correct (round-off on 70x53x37) but
    //     has never been timed against Tiled16.
    //   * min_dim >= 256, the smallest dimension in the measured grid.
    //
    // HOW OFTEN THIS FIRES, MEASURED RATHER THAN ASSUMED. On the whole test
    // suite's coverage capture, after removing the 2312 synthetic probe rows
    // that route_gemm_equivalence_tests.cc feeds straight to the resolver:
    // 46 of 7223 real non-float gemm calls, 0.64%. Restricted to calls where
    // the problem is not small (max(m,n) >= 128), 91.6% are blocked by
    // k < 256 and 69% by a transpose.
    //
    // That is structural, not an artefact of test sizing. The dominant
    // internal GEMM here is a PANEL UPDATE -- large m, large n, small k --
    // and k is the blocking factor, a tuning constant clustered at 1/8/32/48/
    // 96/136 that does not grow with the problem. min_dim takes the min over
    // k, so for that population this gate cannot fire at any problem size.
    //
    // Do not "fix" that by dropping the floor alone: zero calls are blocked
    // by the k floor by itself. Every large-m,n small-k call is ALSO
    // transposed or ragged, so serving them needs a transposed and predicated
    // variant, not a wider predicate. What this kernel does serve is the
    // direct public-API call -- large, square, aligned, NN -- where it is
    // worth 3.6-7.7x over the fallback in a vendor-free build.
    if constexpr (!std::is_same_v<T, float>) {
        if (min_dim >= 256 && can_use_64x64_k16_wide_fast_path<T>(A, B, C)) {
            return KernelVariant::Tiled64x64RegisterK16Wide;
        }
    }

    if constexpr (std::is_same_v<T, double>) {
        // The Direct/Tiled16 crossover for double is at 24, not 32. Measured on
        // RTX 4090 / sm_89, median of 3, both betas, at saturation:
        //
        //   n   batch   Direct   Tiled16   winner
        //   24    512     708      518     Direct,  1.37x
        //   24   4096    1126      687     Direct,  1.64x
        //   25   4096     750      746     a wash (inside spread)
        //   28   4096     903      937     Tiled16, 1.04x
        //   32    512     903      973     Tiled16, 1.08x
        //   32   4096     938     1211     Tiled16, 1.29x
        //
        // This was worth finding rather than tidying: n=32 was the ONLY cell in
        // the entire window preferred() accepts for double where the native
        // route lost to cuBLAS (0.92-0.96x at batch 4096). Picking Tiled16
        // there turns it into a 1.15-1.23x win, so the whole double window is
        // now a native win from n=4 to n=512. See WP2_GEMM_SPEC.md E3.
        //
        // 24 rather than 25 because 25 is inside the run-to-run spread and a
        // boundary should sit where the evidence is unambiguous.
        return max_dim <= 24 ? KernelVariant::Direct : KernelVariant::Tiled16;
    }

    return max_dim <= 64 ? KernelVariant::Direct : KernelVariant::Tiled16;
}

template <typename T>
Event gemm_custom(Queue& ctx,
                  const MatrixView<T, MatrixFormat::Dense>& A,
                  const MatrixView<T, MatrixFormat::Dense>& B,
                  const MatrixView<T, MatrixFormat::Dense>& C,
                  T alpha,
                  T beta,
                  Transpose transA,
                  Transpose transB,
                  ComputePrecision precision) {
    static_cast<void>(precision);
    if (A.batch_size() != B.batch_size() || A.batch_size() != C.batch_size()) {
        throw std::runtime_error("GEMM SYCL custom path requires matching batch sizes");
    }

    const auto [m, k] = get_effective_dims(A, transA);
    const auto [k_b, n] = get_effective_dims(B, transB);
    if (k != k_b || C.rows() != m || C.cols() != n) {
        throw std::runtime_error("GEMM SYCL custom path received incompatible matrix dimensions");
    }

    const KernelVariant variant = choose_runtime_kernel_variant(ctx, A, B, C, transA, transB);
    if (is_experimental_kernel_variant(variant) && !experimental_kernel_variants_enabled()) {
        throw std::runtime_error(
            "Requested experimental GEMM SYCL kernel variant without BATCHLAS_GEMM_EXPERIMENTAL enabled");
    }

    switch (variant) {
    case KernelVariant::Direct:
        return launch_direct(ctx, A, B, C, alpha, beta, transA, transB);
    case KernelVariant::Tiled16:
        return launch_tiled<T, 16>(ctx, A, B, C, alpha, beta, transA, transB);
    case KernelVariant::Tiled32x32Register:
        return launch_register_32x32(ctx, A, B, C, alpha, beta, kernel_trace_name);
    case KernelVariant::Tiled64x64Register:
        return launch_register_64x64(ctx, A, B, C, alpha, beta, kernel_trace_name);
    case KernelVariant::Tiled64x64RegisterK16:
        return launch_register_64x64_k16(ctx, A, B, C, alpha, beta, kernel_trace_name);
    case KernelVariant::Tiled64x64RegisterK16TN:
        if (transA == Transpose::Trans && transB == Transpose::NoTrans) {
            return launch_register_64x64_k16_tn(ctx, A, B, C, alpha, beta, kernel_trace_name);
        }
        return launch_tiled<T, 16>(ctx, A, B, C, alpha, beta, transA, transB);
    case KernelVariant::Tiled64x64RegisterK16NT:
        if (transA == Transpose::NoTrans && transB == Transpose::Trans) {
            return launch_register_64x64_k16_nt(ctx, A, B, C, alpha, beta, kernel_trace_name);
        }
        return launch_tiled<T, 16>(ctx, A, B, C, alpha, beta, transA, transB);
    case KernelVariant::Tiled64x64RegisterK16TT:
        if (transA == Transpose::Trans && transB == Transpose::Trans) {
            return launch_register_64x64_k16_tt(ctx, A, B, C, alpha, beta, kernel_trace_name);
        }
        return launch_tiled<T, 16>(ctx, A, B, C, alpha, beta, transA, transB);
    case KernelVariant::Tiled128x32RegisterK16:
        return launch_register_128x32_k16(ctx, A, B, C, alpha, beta, kernel_trace_name);
    case KernelVariant::Tiled128x32RegisterK16TN:
        if (transA == Transpose::Trans && transB == Transpose::NoTrans) {
            return launch_register_128x32_k16_tn(ctx, A, B, C, alpha, beta, kernel_trace_name);
        }
        return launch_tiled<T, 16>(ctx, A, B, C, alpha, beta, transA, transB);
    case KernelVariant::Tiled128x32RegisterK16NT:
        if (transA == Transpose::NoTrans && transB == Transpose::Trans) {
            return launch_register_128x32_k16_nt(ctx, A, B, C, alpha, beta, kernel_trace_name);
        }
        return launch_tiled<T, 16>(ctx, A, B, C, alpha, beta, transA, transB);
    case KernelVariant::Tiled128x32RegisterK16TT:
        if (transA == Transpose::Trans && transB == Transpose::Trans) {
            return launch_register_128x32_k16_tt(ctx, A, B, C, alpha, beta, kernel_trace_name);
        }
        return launch_tiled<T, 16>(ctx, A, B, C, alpha, beta, transA, transB);
    case KernelVariant::Tiled128x32RegisterK32TN:
        if (transA == Transpose::Trans && transB == Transpose::NoTrans) {
            return launch_register_128x32_k32_tn(ctx, A, B, C, alpha, beta, kernel_trace_name);
        }
        return launch_tiled<T, 16>(ctx, A, B, C, alpha, beta, transA, transB);
    case KernelVariant::Tiled128x32RegisterK32NT:
        if (transA == Transpose::NoTrans && transB == Transpose::Trans) {
            return launch_register_128x32_k32_nt(ctx, A, B, C, alpha, beta, kernel_trace_name);
        }
        return launch_tiled<T, 16>(ctx, A, B, C, alpha, beta, transA, transB);
    case KernelVariant::Tiled128x32RegisterK32TT:
        if (transA == Transpose::Trans && transB == Transpose::Trans) {
            return launch_register_128x32_k32_tt(ctx, A, B, C, alpha, beta, kernel_trace_name);
        }
        return launch_tiled<T, 16>(ctx, A, B, C, alpha, beta, transA, transB);
    case KernelVariant::Tiled128x64RegisterK16TN:
        if (transA == Transpose::Trans && transB == Transpose::NoTrans) {
            return launch_register_128x64_k16_tn(ctx, A, B, C, alpha, beta, kernel_trace_name);
        }
        return launch_tiled<T, 16>(ctx, A, B, C, alpha, beta, transA, transB);
    case KernelVariant::Tiled128x64RegisterK16NT:
        if (transA == Transpose::NoTrans && transB == Transpose::Trans) {
            return launch_register_128x64_k16_nt(ctx, A, B, C, alpha, beta, kernel_trace_name);
        }
        return launch_tiled<T, 16>(ctx, A, B, C, alpha, beta, transA, transB);
    case KernelVariant::Tiled128x64RegisterK16TT:
        if (transA == Transpose::Trans && transB == Transpose::Trans) {
            return launch_register_128x64_k16_tt(ctx, A, B, C, alpha, beta, kernel_trace_name);
        }
        return launch_tiled<T, 16>(ctx, A, B, C, alpha, beta, transA, transB);
    case KernelVariant::Tiled128x32RegisterK32:
        return launch_register_128x32_k32(ctx, A, B, C, alpha, beta, kernel_trace_name);
    case KernelVariant::Tiled128x32RegisterK32S1U1:
        return launch_register_128x32_k32_s1_u1(ctx, A, B, C, alpha, beta, kernel_trace_name);
    case KernelVariant::Tiled128x32RegisterK32S2U1:
        return launch_register_128x32_k32_s2_u1(ctx, A, B, C, alpha, beta, kernel_trace_name);
    case KernelVariant::Tiled128x32RegisterK32S2U1Aligned:
        return launch_register_128x32_k32_s2_u1_aligned(ctx, A, B, C, alpha, beta, kernel_trace_name);
    case KernelVariant::Tiled128x32RegisterK32S2U1Generic:
        return launch_register_128x32_k32_s2_u1_generic(ctx, A, B, C, alpha, beta, kernel_trace_name);
    case KernelVariant::Tiled128x32RegisterK32S2U2:
        return launch_register_128x32_k32_s2_u2(ctx, A, B, C, alpha, beta, kernel_trace_name);
    case KernelVariant::Tiled128x32RegisterK32S2U2TT8x4:
        return launch_register_128x32_k32_s2_u2_tt8x4(ctx, A, B, C, alpha, beta, kernel_trace_name);
    case KernelVariant::Tiled128x32RegisterK32S2U2TT4x8:
        return launch_register_128x32_k32_s2_u2_tt4x8(ctx, A, B, C, alpha, beta, kernel_trace_name);
    case KernelVariant::Tiled128x32RegisterK32Persistent:
        return launch_register_128x32_k32_persistent(ctx, A, B, C, alpha, beta, transA, transB, kernel_trace_name);
    case KernelVariant::Tiled128x32RegisterK32SplitK4:
        return launch_register_128x32_k32_split_k4(ctx, A, B, C, alpha, beta, transA, transB, kernel_trace_name);
    case KernelVariant::Tiled128x32RegisterK32S1U4:
        return launch_register_128x32_k32_s1_u4(ctx, A, B, C, alpha, beta, kernel_trace_name);
    case KernelVariant::Tiled128x64RegisterK32Large:
        return launch_register_128x64_k32_large(ctx, A, B, C, alpha, beta, kernel_trace_name);
    case KernelVariant::Tiled128x64RegisterK32LargeU2:
        return launch_register_128x64_k32_large_u2(ctx, A, B, C, alpha, beta, kernel_trace_name);
    case KernelVariant::Tiled128x64RegisterK32LargeTT4x8:
        return launch_register_128x64_k32_large_tt4x8(ctx, A, B, C, alpha, beta, kernel_trace_name);
    case KernelVariant::Tiled128x64RegisterK32LargeTT4x8U2:
        return launch_register_128x64_k32_large_tt4x8_u2(ctx, A, B, C, alpha, beta, kernel_trace_name);
    case KernelVariant::Tiled128x128RegisterK8:
        // float only, and NN only. A 64-accumulator thread tile is 64
        // registers for float but 128 for double and 256 for complex<double>;
        // and the kernel reads A as m x k and B as k x n directly, so it
        // cannot serve a transposed operand. The selector never picks it
        // outside those bounds, but it can also be forced by name, so fall
        // back rather than compute the wrong thing.
        //
        // The reason for the float restriction is LAUNCHABILITY, not spilling
        // -- this comment used to say "which spills" and that is measured
        // false. At an 8x8 tile, double compiles to 208 total registers and
        // complex<float> to 247, both with ZERO spill bytes on sm_89; only
        // complex<double> spills, and only 3.4 KB. What actually fails is the
        // hard limit of 65,536 registers per block: 208 x 512 threads
        // overruns it and complex<double> throws at launch. Non-float goes to
        // Tiled64x64RegisterK16Wide above, whose 4x4 tile fits every scalar.
        // See WP2_WIDE_SCALAR_GEMM_VERDICT.md section 2.
        if constexpr (std::is_same_v<T, float>) {
            if (transA == Transpose::NoTrans && transB == Transpose::NoTrans) {
                if (can_use_128x128_fast_path<T>(A, B, C)) {
                    return launch_register_128x128_k8<T, true>(
                        ctx, A, B, C, alpha, beta, kernel_trace_name);
                }
                return launch_register_128x128_k8<T, false>(
                    ctx, A, B, C, alpha, beta, kernel_trace_name);
            }
        }
        return launch_tiled<T, 16>(ctx, A, B, C, alpha, beta, transA, transB);
    case KernelVariant::Tiled64x64RegisterK16Wide:
        // NN only: the kernel reads A as m x k and B as k x n directly, so it
        // cannot serve a transposed operand. Unlike the 128x128 float kernel
        // every scalar is supported -- a 4x4 thread tile is 16 accumulators,
        // i.e. 32 registers for double and complex<float> and 64 for
        // complex<double>, measured at 72-134 total with zero spill bytes on
        // sm_89. The selector never picks it outside these bounds, but it can
        // be forced by name, so fall back rather than compute the wrong thing.
        if (transA == Transpose::NoTrans && transB == Transpose::NoTrans) {
            if (can_use_64x64_k16_wide_fast_path<T>(A, B, C)) {
                return launch_register_64x64_k16_wide<T, true>(
                    ctx, A, B, C, alpha, beta, kernel_trace_name);
            }
            return launch_register_64x64_k16_wide<T, false>(
                ctx, A, B, C, alpha, beta, kernel_trace_name);
        }
        return launch_tiled<T, 16>(ctx, A, B, C, alpha, beta, transA, transB);
    case KernelVariant::Tiled32x128RegisterK16:
        return launch_register_32x128_k16(ctx, A, B, C, alpha, beta, kernel_trace_name);
    case KernelVariant::Tiled32x128RegisterK16TN:
        if (transA == Transpose::Trans && transB == Transpose::NoTrans) {
            return launch_register_32x128_k16_tn(ctx, A, B, C, alpha, beta, kernel_trace_name);
        }
        return launch_tiled<T, 16>(ctx, A, B, C, alpha, beta, transA, transB);
    case KernelVariant::Tiled32x128RegisterK16TT:
        if (transA == Transpose::Trans && transB == Transpose::Trans) {
            return launch_register_32x128_k16_tt(ctx, A, B, C, alpha, beta, kernel_trace_name);
        }
        return launch_tiled<T, 16>(ctx, A, B, C, alpha, beta, transA, transB);
    }

    return ctx.get_event();
}

template KernelVariant select_kernel_variant<float>(const MatrixView<float, MatrixFormat::Dense>&,
                                                    const MatrixView<float, MatrixFormat::Dense>&,
                                                    const MatrixView<float, MatrixFormat::Dense>&,
                                                    Transpose,
                                                    Transpose);
template KernelVariant select_kernel_variant<double>(const MatrixView<double, MatrixFormat::Dense>&,
                                                     const MatrixView<double, MatrixFormat::Dense>&,
                                                     const MatrixView<double, MatrixFormat::Dense>&,
                                                     Transpose,
                                                     Transpose);
template KernelVariant select_kernel_variant<std::complex<float>>(const MatrixView<std::complex<float>, MatrixFormat::Dense>&,
                                                                  const MatrixView<std::complex<float>, MatrixFormat::Dense>&,
                                                                  const MatrixView<std::complex<float>, MatrixFormat::Dense>&,
                                                                  Transpose,
                                                                  Transpose);
template KernelVariant select_kernel_variant<std::complex<double>>(const MatrixView<std::complex<double>, MatrixFormat::Dense>&,
                                                                   const MatrixView<std::complex<double>, MatrixFormat::Dense>&,
                                                                   const MatrixView<std::complex<double>, MatrixFormat::Dense>&,
                                                                   Transpose,
                                                                   Transpose);

template Event gemm_custom<float>(Queue&,
                                  const MatrixView<float, MatrixFormat::Dense>&,
                                  const MatrixView<float, MatrixFormat::Dense>&,
                                  const MatrixView<float, MatrixFormat::Dense>&,
                                  float,
                                  float,
                                  Transpose,
                                  Transpose,
                                  ComputePrecision);
template Event gemm_custom<double>(Queue&,
                                   const MatrixView<double, MatrixFormat::Dense>&,
                                   const MatrixView<double, MatrixFormat::Dense>&,
                                   const MatrixView<double, MatrixFormat::Dense>&,
                                   double,
                                   double,
                                   Transpose,
                                   Transpose,
                                   ComputePrecision);
template Event gemm_custom<std::complex<float>>(Queue&,
                                                const MatrixView<std::complex<float>, MatrixFormat::Dense>&,
                                                const MatrixView<std::complex<float>, MatrixFormat::Dense>&,
                                                const MatrixView<std::complex<float>, MatrixFormat::Dense>&,
                                                std::complex<float>,
                                                std::complex<float>,
                                                Transpose,
                                                Transpose,
                                                ComputePrecision);
template Event gemm_custom<std::complex<double>>(Queue&,
                                                 const MatrixView<std::complex<double>, MatrixFormat::Dense>&,
                                                 const MatrixView<std::complex<double>, MatrixFormat::Dense>&,
                                                 const MatrixView<std::complex<double>, MatrixFormat::Dense>&,
                                                 std::complex<double>,
                                                 std::complex<double>,
                                                 Transpose,
                                                 Transpose,
                                                 ComputePrecision);

} // namespace batchlas::sycl_gemm