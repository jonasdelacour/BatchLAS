#include "trmm_custom_dispatch.hh"

#include "gemm_cublasdx_dispatch.hh"
#include "gemm_variant.hh"
#include "trmm_cublasdx_fused.hh"
#include "cublasdx_dispatch_common.hh"

#include "../util/kernel-trace.hh"

#include <algorithm>
#include <stdexcept>

namespace batchlas::backend {

namespace {

constexpr int kTrmmCublasDxTile = 32;

enum class TrmmVariantRequest {
    Vendor,
    CuBLASDx,
    Auto,
};

TrmmVariantRequest trmm_variant_request() {
    return detail::parse_cublasdx_variant_request("BATCHLAS_TRMM_VARIANT",
                                                  TrmmVariantRequest::Vendor,
                                                  TrmmVariantRequest::CuBLASDx,
                                                  TrmmVariantRequest::Auto);
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

    const int output_tile_rows = detail::ceil_div(A.rows(), kTrmmCublasDxTile);
    const int output_tile_cols = detail::ceil_div(B.cols(), kTrmmCublasDxTile);
    const int tiled_work = A.batch_size() * output_tile_rows * output_tile_cols;
    return tiled_work >= 8;
}

[[noreturn]] void throw_forced_trmm_unavailable(const std::string& reason) {
    detail::throw_forced_cublasdx_unavailable("BATCHLAS_TRMM_VARIANT", "TRMM", reason);
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
    const bool problem_supported = trmm_problem_supported(A, B, C, side, uplo, transA);
    return detail::should_use_cublasdx(ctx,
                                       request,
                                       TrmmVariantRequest::Vendor,
                                       TrmmVariantRequest::CuBLASDx,
                                       problem_supported,
                                       problem_supported && trmm_prefer_cuda_custom_heuristic(A, B));
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
    if (!detail::is_gpu_queue(ctx)) {
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
    if (detail::cublasdx_variant_needs_fallback(variant, trmm_cublasdx::available())) {
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
                                                           detail::cuda_stream_from_queue(ctx));
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