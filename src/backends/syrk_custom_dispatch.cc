#include "syrk_custom_dispatch.hh"

#include "gemm_cublasdx_dispatch.hh"
#include "gemm_variant.hh"
#include "syrk_cublasdx_fused.hh"
#include "cublasdx_dispatch_common.hh"

#include "../util/kernel-trace.hh"

#include <algorithm>
#include <stdexcept>

namespace batchlas::backend {

namespace {

constexpr int kSyrkCublasDxTile = 32;

enum class SyrkVariantRequest {
    Vendor,
    CuBLASDx,
    Auto,
};

SyrkVariantRequest syrk_variant_request() {
    return detail::parse_cublasdx_variant_request("BATCHLAS_SYRK_VARIANT",
                                                  SyrkVariantRequest::Vendor,
                                                  SyrkVariantRequest::CuBLASDx,
                                                  SyrkVariantRequest::Auto);
}

bool syrk_problem_supported(const MatrixView<float, MatrixFormat::Dense>& A,
                            const MatrixView<float, MatrixFormat::Dense>& C,
                            Transpose transA) {
    if (transA == Transpose::ConjTrans) {
        return false;
    }
    if (C.rows() != C.cols()) {
        return false;
    }
    if (A.batch_size() != C.batch_size()) {
        return false;
    }

    const int n = C.rows();
    const int k = transA == Transpose::NoTrans ? A.cols() : A.rows();
    const int expected_n = transA == Transpose::NoTrans ? A.rows() : A.cols();
    return expected_n == n && n > 0 && k > 0;
}

bool syrk_prefer_cuda_custom_heuristic(const MatrixView<float, MatrixFormat::Dense>& A,
                                       const MatrixView<float, MatrixFormat::Dense>& C,
                                       Transpose transA) {
    const int n = C.rows();
    const int k = transA == Transpose::NoTrans ? A.cols() : A.rows();
    const int max_dim = std::max(n, k);
    const int min_dim = std::min(n, k);
    if (n < 16) {
        return false;
    }

    const int output_tile_rows = detail::ceil_div(n, kSyrkCublasDxTile);
    const int reduction_tiles = detail::ceil_div(k, kSyrkCublasDxTile);
    const int tiled_work = A.batch_size() * output_tile_rows * output_tile_rows * reduction_tiles;
    return min_dim * 2 >= max_dim && tiled_work >= 8;
}

Event syrk_cublasdx_fallback_gemm(Queue& ctx,
                                  const MatrixView<float, MatrixFormat::Dense>& A,
                                  const MatrixView<float, MatrixFormat::Dense>& C,
                                  float alpha,
                                  float beta,
                                  Transpose transA) {
    const Transpose transB = transA == Transpose::NoTrans ? Transpose::Trans : Transpose::NoTrans;
    BATCHLAS_KERNEL_TRACE_SCOPE("syrk_cuda_custom.gemm_fallback");
    return gemm_cublasdx(ctx, A, A, C, alpha, beta, transA, transB, ComputePrecision::Default);
}

} // namespace

bool syrk_use_cuda_custom(const Queue& ctx,
                          const MatrixView<float, MatrixFormat::Dense>& A,
                          const MatrixView<float, MatrixFormat::Dense>& C,
                          Uplo,
                          Transpose transA) {
    const auto request = syrk_variant_request();
    const bool problem_supported = syrk_problem_supported(A, C, transA);
    return detail::should_use_cublasdx(ctx,
                                       request,
                                       SyrkVariantRequest::Vendor,
                                       SyrkVariantRequest::CuBLASDx,
                                       problem_supported,
                                       problem_supported && syrk_prefer_cuda_custom_heuristic(A, C, transA));
}

Event syrk_cuda_custom(Queue& ctx,
                       const MatrixView<float, MatrixFormat::Dense>& A,
                       const MatrixView<float, MatrixFormat::Dense>& C,
                       float alpha,
                       float beta,
                       Uplo uplo,
                       Transpose transA) {
    if (!syrk_problem_supported(A, C, transA)) {
        return syrk_vendor_cuda_raw(ctx, A, C, alpha, beta, uplo, transA);
    }

    const Transpose transB = transA == Transpose::NoTrans ? Transpose::Trans : Transpose::NoTrans;
    const auto variant = cublasdx_gemm_select_variant(A, A, C, transA, transB);
    if (detail::cublasdx_variant_needs_fallback(variant, syrk_cublasdx::available())) {
        return syrk_cublasdx_fallback_gemm(ctx, A, C, alpha, beta, transA);
    }

    syrk_cublasdx::SyrkLaunchDescriptor desc{};
    desc.a_ptr = A.data_ptr();
    desc.c_ptr = C.data_ptr();
    desc.lda = A.ld();
    desc.ldc = C.ld();
    desc.stride_a = A.stride();
    desc.stride_c = C.stride();
    desc.n = C.rows();
    desc.k = transA == Transpose::NoTrans ? A.cols() : A.rows();
    desc.batch = A.batch_size();
    desc.alpha = alpha;
    desc.beta = beta;

    BATCHLAS_KERNEL_TRACE_SCOPE("syrk_cuda_custom.fused");
    const cudaError_t status = syrk_cublasdx::launch_float(variant,
                                                           desc,
                                                           uplo,
                                                           transA,
                                                           detail::cuda_stream_from_queue(ctx));
    if (status == cudaErrorNotSupported) {
        return syrk_cublasdx_fallback_gemm(ctx, A, C, alpha, beta, transA);
    }
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("cuBLASDx fused SYRK launch failed: ") + cudaGetErrorString(status));
    }

    return ctx.create_event_after_external_work();
}

} // namespace batchlas::backend