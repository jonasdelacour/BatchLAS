#include "syr2k_custom_dispatch.hh"

#include "gemm_cublasdx_dispatch.hh"
#include "gemm_variant.hh"
#include "syr2k_cublasdx_fused.hh"
#include "cublasdx_dispatch_common.hh"

#include "../util/kernel-trace.hh"

#include <algorithm>
#include <stdexcept>

namespace batchlas::backend {

namespace {

constexpr int kSyr2kCublasDxTile = 32;

enum class Syr2kVariantRequest {
    Vendor,
    CuBLASDx,
    Auto,
};

Syr2kVariantRequest syr2k_variant_request() {
    return detail::parse_cublasdx_variant_request("BATCHLAS_SYR2K_VARIANT",
                                                  Syr2kVariantRequest::Vendor,
                                                  Syr2kVariantRequest::CuBLASDx,
                                                  Syr2kVariantRequest::Auto);
}

bool syr2k_problem_supported(const MatrixView<float, MatrixFormat::Dense>& A,
                             const MatrixView<float, MatrixFormat::Dense>& B,
                             const MatrixView<float, MatrixFormat::Dense>& C,
                             Transpose transA) {
    if (transA == Transpose::ConjTrans) {
        return false;
    }
    if (C.rows() != C.cols()) {
        return false;
    }
    if (A.batch_size() != B.batch_size() || B.batch_size() != C.batch_size()) {
        return false;
    }

    const int n = C.rows();
    const int a_n = transA == Transpose::NoTrans ? A.rows() : A.cols();
    const int b_n = transA == Transpose::NoTrans ? B.rows() : B.cols();
    const int a_k = transA == Transpose::NoTrans ? A.cols() : A.rows();
    const int b_k = transA == Transpose::NoTrans ? B.cols() : B.rows();
    return a_n == n && b_n == n && a_k == b_k && n > 0 && a_k > 0;
}

bool syr2k_prefer_cuda_custom_heuristic(const MatrixView<float, MatrixFormat::Dense>& A,
                                        const MatrixView<float, MatrixFormat::Dense>& C,
                                        Transpose transA) {
    const int n = C.rows();
    const int k = transA == Transpose::NoTrans ? A.cols() : A.rows();
    const int max_dim = std::max(n, k);
    const int min_dim = std::min(n, k);
    if (n < kSyr2kCublasDxTile) {
        return false;
    }

    const int output_tile_rows = detail::ceil_div(n, kSyr2kCublasDxTile);
    const int reduction_tiles = detail::ceil_div(k, kSyr2kCublasDxTile);
    const int tiled_work = A.batch_size() * output_tile_rows * output_tile_rows * reduction_tiles;
    return min_dim * 2 >= max_dim && tiled_work >= 8;
}

Event syr2k_cublasdx_fallback(Queue& ctx,
                              const MatrixView<float, MatrixFormat::Dense>& A,
                              const MatrixView<float, MatrixFormat::Dense>& B,
                              const MatrixView<float, MatrixFormat::Dense>& C,
                              float alpha,
                              float beta,
                              Transpose transA) {
    const Transpose transB = transA == Transpose::NoTrans ? Transpose::Trans : Transpose::NoTrans;
    BATCHLAS_KERNEL_TRACE_SCOPE("syr2k_cuda_custom.gemm_fallback");
    gemm_cublasdx(ctx, A, B, C, alpha, beta, transA, transB, ComputePrecision::Default).wait();
    return gemm_cublasdx(ctx, B, A, C, alpha, 1.0f, transA, transB, ComputePrecision::Default);
}

[[noreturn]] void throw_forced_syr2k_unavailable(const std::string& reason) {
    detail::throw_forced_cublasdx_unavailable("BATCHLAS_SYR2K_VARIANT", "SYR2K", reason);
}

} // namespace

bool syr2k_cuda_custom_forced() {
    return syr2k_variant_request() == Syr2kVariantRequest::CuBLASDx;
}

bool syr2k_use_cuda_custom(const Queue& ctx,
                           const MatrixView<float, MatrixFormat::Dense>& A,
                           const MatrixView<float, MatrixFormat::Dense>& B,
                           const MatrixView<float, MatrixFormat::Dense>& C,
                           Uplo,
                           Transpose transA) {
    const auto request = syr2k_variant_request();
    const bool problem_supported = syr2k_problem_supported(A, B, C, transA);
    return detail::should_use_cublasdx(ctx,
                                       request,
                                       Syr2kVariantRequest::Vendor,
                                       Syr2kVariantRequest::CuBLASDx,
                                       problem_supported,
                                       problem_supported && syr2k_prefer_cuda_custom_heuristic(A, C, transA));
}

Event syr2k_cuda_custom(Queue& ctx,
                        const MatrixView<float, MatrixFormat::Dense>& A,
                        const MatrixView<float, MatrixFormat::Dense>& B,
                        const MatrixView<float, MatrixFormat::Dense>& C,
                        float alpha,
                        float beta,
                        Uplo uplo,
                        Transpose transA) {
    const bool forced = syr2k_cuda_custom_forced();
    if (!detail::is_gpu_queue(ctx)) {
        if (forced) {
            throw_forced_syr2k_unavailable("the active queue is not a GPU queue");
        }
        return syr2k_vendor_cuda_raw(ctx, A, B, C, alpha, beta, uplo, transA);
    }
    if (!syr2k_problem_supported(A, B, C, transA)) {
        if (forced) {
            throw_forced_syr2k_unavailable("the problem shape or transpose mode is unsupported");
        }
        return syr2k_vendor_cuda_raw(ctx, A, B, C, alpha, beta, uplo, transA);
    }

    const Transpose transB = transA == Transpose::NoTrans ? Transpose::Trans : Transpose::NoTrans;
    const auto variant = cublasdx_gemm_select_variant(A, B, C, transA, transB);
    if (detail::cublasdx_variant_needs_fallback(variant, syr2k_cublasdx::available())) {
        if (forced) {
            throw_forced_syr2k_unavailable("no compatible fused kernel is available in this build for the requested problem");
        }
        return syr2k_cublasdx_fallback(ctx, A, B, C, alpha, beta, transA);
    }

    syr2k_cublasdx::Syr2kLaunchDescriptor desc{};
    desc.a_ptr = A.data_ptr();
    desc.b_ptr = B.data_ptr();
    desc.c_ptr = C.data_ptr();
    desc.lda = A.ld();
    desc.ldb = B.ld();
    desc.ldc = C.ld();
    desc.stride_a = A.stride();
    desc.stride_b = B.stride();
    desc.stride_c = C.stride();
    desc.n = C.rows();
    desc.k = transA == Transpose::NoTrans ? A.cols() : A.rows();
    desc.batch = A.batch_size();
    desc.alpha = alpha;
    desc.beta = beta;

    BATCHLAS_KERNEL_TRACE_SCOPE("syr2k_cuda_custom.fused");
    const cudaError_t status = syr2k_cublasdx::launch_float(variant,
                                                            desc,
                                                            uplo,
                                                            transA,
                                                            detail::cuda_stream_from_queue(ctx));
    if (status == cudaErrorNotSupported) {
        if (forced) {
            throw_forced_syr2k_unavailable("the current device or matrix layout does not satisfy the fused kernel requirements");
        }
        return syr2k_cublasdx_fallback(ctx, A, B, C, alpha, beta, transA);
    }
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("cuBLASDx fused SYR2K launch failed: ") + cudaGetErrorString(status));
    }

    return ctx.create_event_after_external_work();
}

} // namespace batchlas::backend