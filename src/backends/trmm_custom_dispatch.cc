#include "trmm_custom_dispatch.hh"

#include "gemm_cublasdx_dispatch.hh"
#include "gemm_variant.hh"
#include "trmm_cublasdx_fused.hh"

#include "../util/kernel-trace.hh"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <sycl/sycl.hpp>

namespace batchlas::backend {

namespace {

constexpr int kTrmmCublasDxTile = 32;

int ceil_div(int value, int divisor) {
    return (value + divisor - 1) / divisor;
}

enum class TrmmVariantRequest {
    Vendor,
    CuBLASDx,
    Auto,
};

TrmmVariantRequest trmm_variant_request() {
    const char* raw = std::getenv("BATCHLAS_TRMM_VARIANT");
    if (!raw) {
        return TrmmVariantRequest::Auto;
    }

    std::string value(raw);
    for (char& ch : value) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }

    if (value == "vendor") {
        return TrmmVariantRequest::Vendor;
    }
    if (value == "cublasdx" || value == "dx" || value == "custom") {
        return TrmmVariantRequest::CuBLASDx;
    }
    return TrmmVariantRequest::Auto;
}

bool trmm_problem_supported(const MatrixView<float, MatrixFormat::Dense>& A,
                            const MatrixView<float, MatrixFormat::Dense>& B,
                            const MatrixView<float, MatrixFormat::Dense>& C,
                            Side side,
                            Uplo uplo,
                            Transpose transA) {
    if (side != Side::Left || uplo != Uplo::Lower || transA != Transpose::NoTrans) {
        return false;
    }
    if (A.rows() != A.cols()) {
        return false;
    }
    if (A.batch_size() != B.batch_size() || B.batch_size() != C.batch_size()) {
        return false;
    }
    return A.rows() == B.rows() && B.rows() == C.rows() && B.cols() == C.cols() && A.rows() > 0 && B.cols() > 0;
}

bool trmm_prefer_cuda_custom_heuristic(const MatrixView<float, MatrixFormat::Dense>& A,
                                       const MatrixView<float, MatrixFormat::Dense>& B) {
    if (A.rows() < kTrmmCublasDxTile || B.cols() < kTrmmCublasDxTile) {
        return false;
    }

    const int output_tile_rows = ceil_div(A.rows(), kTrmmCublasDxTile);
    const int output_tile_cols = ceil_div(B.cols(), kTrmmCublasDxTile);
    const int tiled_work = A.batch_size() * output_tile_rows * output_tile_cols;
    return tiled_work >= 8;
}

inline cudaStream_t cuda_stream_from_queue(const Queue& ctx) {
    return sycl::get_native<sycl::backend::ext_oneapi_cuda>(*ctx);
}

[[noreturn]] void throw_forced_trmm_unavailable(const std::string& reason) {
    throw std::runtime_error("BATCHLAS_TRMM_VARIANT=cublasdx requested, but fused cuBLASDx TRMM is unavailable: " + reason);
}

} // namespace

bool trmm_cuda_custom_forced() {
    return trmm_variant_request() == TrmmVariantRequest::CuBLASDx;
}

bool trmm_use_cuda_custom(const Queue& ctx,
                          const MatrixView<float, MatrixFormat::Dense>& A,
                          const MatrixView<float, MatrixFormat::Dense>& B,
                          const MatrixView<float, MatrixFormat::Dense>& C,
                          Side side,
                          Uplo uplo,
                          Transpose transA,
                          Diag) {
    const auto request = trmm_variant_request();
    if (request == TrmmVariantRequest::CuBLASDx) {
        return true;
    }
    if (ctx.device().type != DeviceType::GPU) {
        return false;
    }
    if (!trmm_problem_supported(A, B, C, side, uplo, transA)) {
        return false;
    }

    if (request == TrmmVariantRequest::Vendor) {
        return false;
    }
    return trmm_prefer_cuda_custom_heuristic(A, B);
}

Event trmm_cuda_custom(Queue& ctx,
                       const MatrixView<float, MatrixFormat::Dense>& A,
                       const MatrixView<float, MatrixFormat::Dense>& B,
                       const MatrixView<float, MatrixFormat::Dense>& C,
                       float alpha,
                       Side side,
                       Uplo uplo,
                       Transpose transA,
                       Diag diag) {
    const bool forced = trmm_cuda_custom_forced();
    if (ctx.device().type != DeviceType::GPU) {
        if (forced) {
            throw_forced_trmm_unavailable("the active queue is not a GPU queue");
        }
        return trmm_vendor_cuda_raw(ctx, A, B, C, alpha, side, uplo, transA, diag);
    }
    if (!trmm_problem_supported(A, B, C, side, uplo, transA)) {
        if (forced) {
            throw_forced_trmm_unavailable("only left/lower/notrans float problems with matching dense batches are currently supported");
        }
        return trmm_vendor_cuda_raw(ctx, A, B, C, alpha, side, uplo, transA, diag);
    }

    const auto variant = cublasdx_gemm_select_variant(A,
                                                      B,
                                                      C,
                                                      Transpose::NoTrans,
                                                      Transpose::NoTrans);
    if (variant == cublasdx_gemm::CuBLASDxGemmVariant::VendorFallback ||
        !cublasdx_gemm_variant_available(variant) || !trmm_cublasdx::available()) {
        if (forced) {
            throw_forced_trmm_unavailable("no compatible fused kernel is available in this build for the requested problem");
        }
        return trmm_vendor_cuda_raw(ctx, A, B, C, alpha, side, uplo, transA, diag);
    }

    trmm_cublasdx::TrmmLaunchDescriptor desc{};
    desc.a_ptr = A.data_ptr();
    desc.b_ptr = B.data_ptr();
    desc.c_ptr = C.data_ptr();
    desc.lda = A.ld();
    desc.ldb = B.ld();
    desc.ldc = C.ld();
    desc.stride_a = A.stride();
    desc.stride_b = B.stride();
    desc.stride_c = C.stride();
    desc.m = C.rows();
    desc.n = C.cols();
    desc.batch = A.batch_size();
    desc.alpha = alpha;

    BATCHLAS_KERNEL_TRACE_SCOPE("trmm_cuda_custom.fused");
    const cudaError_t status = trmm_cublasdx::launch_float(variant,
                                                           desc,
                                                           diag,
                                                           cuda_stream_from_queue(ctx));
    if (status == cudaErrorNotSupported) {
        if (forced) {
            throw_forced_trmm_unavailable("the current device or matrix layout does not satisfy the fused kernel requirements");
        }
        return trmm_vendor_cuda_raw(ctx, A, B, C, alpha, side, uplo, transA, diag);
    }
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("cuBLASDx fused TRMM launch failed: ") + cudaGetErrorString(status));
    }

    return ctx.create_event_after_external_work();
}

} // namespace batchlas::backend