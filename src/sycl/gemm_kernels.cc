#include "gemm_kernels.hh"

#include "gemm/accessors.hh"
#include "gemm/register_launchers.hh"
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
    case KernelVariant::Tiled128x32RegisterK32S1U4:
        return true;
    default:
        return false;
    }
}

inline bool has_forced_kernel_variant();

template <typename T>
KernelVariant choose_runtime_kernel_variant(const Queue& ctx,
                                           const MatrixView<T, MatrixFormat::Dense>& A,
                                           const MatrixView<T, MatrixFormat::Dense>& B,
                                           const MatrixView<T, MatrixFormat::Dense>& C,
                                           Transpose transA,
                                           Transpose transB) {
    const KernelVariant selected = select_kernel_variant(A, B, C, transA, transB);
    if (has_forced_kernel_variant()) {
        return selected;
    }

    if constexpr (std::is_same_v<T, float>) {
        if (ctx.device().type == DeviceType::GPU && ctx.device().get_vendor() == Vendor::NVIDIA) {
            const auto [m, k] = get_effective_dims(A, transA);
            const auto [k_b, n] = get_effective_dims(B, transB);
            static_cast<void>(k_b);

            if (transA == Transpose::NoTrans && transB == Transpose::NoTrans && m >= 512 && n >= 512 && k >= 512) {
                return KernelVariant::Tiled128x64RegisterK32Large;
            }

            switch (selected) {
            case KernelVariant::Tiled128x32RegisterK32:
                return KernelVariant::Tiled128x32RegisterK16;
            case KernelVariant::Tiled128x32RegisterK32TN:
                return KernelVariant::Tiled128x32RegisterK16TN;
            case KernelVariant::Tiled128x32RegisterK32NT:
                return KernelVariant::Tiled128x32RegisterK16NT;
            case KernelVariant::Tiled128x32RegisterK32TT:
                return KernelVariant::Tiled128x32RegisterK16TT;
            default:
                break;
            }
        }
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
    case KernelVariant::Tiled128x32RegisterK32S2U1:
        return "gemm_sycl_register_128x32_k32_s2_u1";
    case KernelVariant::Tiled128x32RegisterK32S2U2:
        return "gemm_sycl_register_128x32_k32_s2_u2";
    case KernelVariant::Tiled128x32RegisterK32S1U4:
        return "gemm_sycl_register_128x32_k32_s1_u4";
    case KernelVariant::Tiled128x64RegisterK32Large:
        return "gemm_sycl_register_128x64_k32_large";
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
        return name == "register128x32k32tn" || name == "reg128x32k32tn" || name == "128x32x32tn";
    case KernelVariant::Tiled128x32RegisterK32NT:
        return name == "register128x32k32nt" || name == "reg128x32k32nt" || name == "128x32x32nt";
    case KernelVariant::Tiled128x32RegisterK32TT:
        return name == "register128x32k32tt" || name == "reg128x32k32tt" || name == "128x32x32tt";
    case KernelVariant::Tiled128x64RegisterK16TN:
        return name == "register128x64k16tn" || name == "reg128x64k16tn" || name == "128x64x16tn";
    case KernelVariant::Tiled128x64RegisterK16NT:
        return name == "register128x64k16nt" || name == "reg128x64k16nt" || name == "128x64x16nt";
    case KernelVariant::Tiled128x64RegisterK16TT:
        return name == "register128x64k16tt" || name == "reg128x64k16tt" || name == "128x64x16tt";
    case KernelVariant::Tiled128x32RegisterK32:
        return name == "register128x32k32" || name == "reg128x32k32" || name == "128x32x32";
    case KernelVariant::Tiled128x32RegisterK32S2U1:
        return name == "register128x32k32s2u1" || name == "reg128x32k32s2u1" || name == "128x32x32_s2_u1";
    case KernelVariant::Tiled128x32RegisterK32S2U2:
        return name == "register128x32k32s2u2" || name == "reg128x32k32s2u2" || name == "128x32x32_s2_u2";
    case KernelVariant::Tiled128x32RegisterK32S1U4:
        return name == "register128x32k32s1u4" || name == "reg128x32k32s1u4" || name == "reg128x32k32u4" ||
            name == "128x32x32_s1_u4";
    case KernelVariant::Tiled128x64RegisterK32Large:
        return name == "register128x64k32large" || name == "reg128x64k32large" || name == "128x64x32large";
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
                                  KernelVariant::Tiled128x32RegisterK32S2U1,
                                  KernelVariant::Tiled128x32RegisterK32S2U2,
                                  KernelVariant::Tiled128x32RegisterK32S1U4,
                                  KernelVariant::Tiled128x64RegisterK32Large,
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
        if (m >= 128 && n >= 32 && k >= 128) {
            return KernelVariant::Tiled128x32RegisterK32;
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

    if constexpr (std::is_same_v<T, double>) {
        return max_dim <= 32 ? KernelVariant::Direct : KernelVariant::Tiled16;
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
        return launch_register_64x64_k16_tn(ctx, A, B, C, alpha, beta, kernel_trace_name);
    case KernelVariant::Tiled64x64RegisterK16NT:
        return launch_register_64x64_k16_nt(ctx, A, B, C, alpha, beta, kernel_trace_name);
    case KernelVariant::Tiled64x64RegisterK16TT:
        return launch_register_64x64_k16_tt(ctx, A, B, C, alpha, beta, kernel_trace_name);
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
        return launch_register_128x32_k32_tn(ctx, A, B, C, alpha, beta, kernel_trace_name);
    case KernelVariant::Tiled128x32RegisterK32NT:
        return launch_register_128x32_k32_nt(ctx, A, B, C, alpha, beta, kernel_trace_name);
    case KernelVariant::Tiled128x32RegisterK32TT:
        return launch_register_128x32_k32_tt(ctx, A, B, C, alpha, beta, kernel_trace_name);
    case KernelVariant::Tiled128x64RegisterK16TN:
        return launch_register_128x64_k16_tn(ctx, A, B, C, alpha, beta, kernel_trace_name);
    case KernelVariant::Tiled128x64RegisterK16NT:
        return launch_register_128x64_k16_nt(ctx, A, B, C, alpha, beta, kernel_trace_name);
    case KernelVariant::Tiled128x64RegisterK16TT:
        return launch_register_128x64_k16_tt(ctx, A, B, C, alpha, beta, kernel_trace_name);
    case KernelVariant::Tiled128x32RegisterK32:
        return launch_register_128x32_k32(ctx, A, B, C, alpha, beta, kernel_trace_name);
    case KernelVariant::Tiled128x32RegisterK32S2U1:
        return launch_register_128x32_k32_s2_u1(ctx, A, B, C, alpha, beta, kernel_trace_name);
    case KernelVariant::Tiled128x32RegisterK32S2U2:
        return launch_register_128x32_k32_s2_u2(ctx, A, B, C, alpha, beta, kernel_trace_name);
    case KernelVariant::Tiled128x32RegisterK32S1U4:
        return launch_register_128x32_k32_s1_u4(ctx, A, B, C, alpha, beta, kernel_trace_name);
    case KernelVariant::Tiled128x64RegisterK32Large:
        return launch_register_128x64_k32_large(ctx, A, B, C, alpha, beta, kernel_trace_name);
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