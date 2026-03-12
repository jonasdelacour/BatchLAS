#include "syrk_custom_dispatch.hh"

#include "gemm_cublasdx_dispatch.hh"
#include "gemm_variant.hh"
#include "syrk_cublasdx_fused.hh"

#include "../util/kernel-trace.hh"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <sycl/sycl.hpp>

namespace batchlas::backend {

namespace {

constexpr int kSyrkCublasDxTile = 32;

int ceil_div(int value, int divisor) {
    return (value + divisor - 1) / divisor;
}

enum class SyrkVariantRequest {
    Vendor,
    CuBLASDx,
    Auto,
};

SyrkVariantRequest syrk_variant_request() {
    const char* raw = std::getenv("BATCHLAS_SYRK_VARIANT");
    if (!raw) {
        return SyrkVariantRequest::Auto;
    }

    std::string value(raw);
    for (char& ch : value) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }

    if (value == "vendor") {
        return SyrkVariantRequest::Vendor;
    }
    if (value == "cublasdx" || value == "dx" || value == "custom") {
        return SyrkVariantRequest::CuBLASDx;
    }
    return SyrkVariantRequest::Auto;
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

    const int output_tile_rows = ceil_div(n, kSyrkCublasDxTile);
    const int reduction_tiles = ceil_div(k, kSyrkCublasDxTile);
    const int tiled_work = A.batch_size() * output_tile_rows * output_tile_rows * reduction_tiles;
    return min_dim * 2 >= max_dim && tiled_work >= 8;
}

inline cudaStream_t cuda_stream_from_queue(const Queue& ctx) {
    return sycl::get_native<sycl::backend::ext_oneapi_cuda>(*ctx);
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
    if (ctx.device().type != DeviceType::GPU) {
        return false;
    }
    if (!syrk_problem_supported(A, C, transA)) {
        return false;
    }

    const auto request = syrk_variant_request();
    if (request == SyrkVariantRequest::Vendor) {
        return false;
    }
    if (request == SyrkVariantRequest::CuBLASDx) {
        return true;
    }
    return syrk_prefer_cuda_custom_heuristic(A, C, transA);
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
    if (variant == cublasdx_gemm::CuBLASDxGemmVariant::VendorFallback ||
        !cublasdx_gemm_variant_available(variant) || !syrk_cublasdx::available()) {
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
                                                           cuda_stream_from_queue(ctx));
    if (status == cudaErrorNotSupported) {
        return syrk_cublasdx_fallback_gemm(ctx, A, C, alpha, beta, transA);
    }
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("cuBLASDx fused SYRK launch failed: ") + cudaGetErrorString(status));
    }

    return ctx.create_event_after_external_work();
}

} // namespace batchlas::backend