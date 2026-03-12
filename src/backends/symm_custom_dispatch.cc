#include "symm_custom_dispatch.hh"

#include "gemm_cublasdx_dispatch.hh"
#include "gemm_variant.hh"
#include "symm_cublasdx_fused.hh"

#include "../util/kernel-trace.hh"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <sycl/sycl.hpp>

namespace batchlas::backend {

namespace {

constexpr int kSymmCublasDxTile = 32;

int ceil_div(int value, int divisor) {
    return (value + divisor - 1) / divisor;
}

enum class SymmVariantRequest {
    Vendor,
    CuBLASDx,
    Auto,
};

SymmVariantRequest symm_variant_request() {
    const char* raw = std::getenv("BATCHLAS_SYMM_VARIANT");
    if (!raw) {
        return SymmVariantRequest::Auto;
    }

    std::string value(raw);
    for (char& ch : value) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }

    if (value == "vendor") {
        return SymmVariantRequest::Vendor;
    }
    if (value == "cublasdx" || value == "dx" || value == "custom") {
        return SymmVariantRequest::CuBLASDx;
    }
    if (value == "auto") {
        return SymmVariantRequest::Auto;
    }

    return SymmVariantRequest::Auto;
}

bool symm_problem_supported(const MatrixView<float, MatrixFormat::Dense>& A,
                            const MatrixView<float, MatrixFormat::Dense>& B,
                            const MatrixView<float, MatrixFormat::Dense>& C,
                            Side side) {
    if (A.rows() != A.cols()) {
        return false;
    }

    if (A.batch_size() != B.batch_size() || A.batch_size() != C.batch_size()) {
        return false;
    }

    const int m = C.rows();
    const int n = C.cols();
    const int expected_a = side == Side::Left ? B.rows() : B.cols();
    return A.rows() == expected_a && B.rows() == m && B.cols() == n && m > 0 && n > 0;
}

bool symm_prefer_cuda_custom_heuristic(const MatrixView<float, MatrixFormat::Dense>& A,
                                       const MatrixView<float, MatrixFormat::Dense>& B,
                                       const MatrixView<float, MatrixFormat::Dense>& C,
                                       Side side) {
    const int m = C.rows();
    const int n = C.cols();
    const int k = A.rows();
    const int max_dim = std::max({m, n, k});
    const int min_dim = std::min({m, n, k});
    const bool squareish = min_dim * 2 >= max_dim;
    const int shared_dim = side == Side::Left ? B.rows() : B.cols();
    if (!squareish || shared_dim != k || max_dim < 32) {
        return false;
    }

    const int output_tile_rows = ceil_div(m, kSymmCublasDxTile);
    const int output_tile_cols = ceil_div(n, kSymmCublasDxTile);
    const int reduction_tiles = ceil_div(k, kSymmCublasDxTile);
    const int tiled_work = A.batch_size() * output_tile_rows * output_tile_cols * reduction_tiles;
    return tiled_work >= 8;
}

inline cudaStream_t cuda_stream_from_queue(const Queue& ctx) {
    return sycl::get_native<sycl::backend::ext_oneapi_cuda>(*ctx);
}

Event symm_cublasdx_fallback_gemm(Queue& ctx,
                                  const MatrixView<float, MatrixFormat::Dense>& A,
                                  const MatrixView<float, MatrixFormat::Dense>& B,
                                  const MatrixView<float, MatrixFormat::Dense>& C,
                                  float alpha,
                                  float beta,
                                  Side side,
                                  Uplo uplo) {
    Matrix<float, MatrixFormat::Dense> symmetric_a(A.rows(), A.cols(), A.batch_size(), A.ld(), A.stride());
    auto symmetric_a_view = symmetric_a.view();

    BATCHLAS_KERNEL_TRACE_SCOPE("symm_cuda_custom.expand");
    MatrixView<float, MatrixFormat::Dense>::copy(ctx, symmetric_a_view, A).wait();
    symmetric_a_view.symmetrize(ctx, uplo).wait();

    if (side == Side::Left) {
        return gemm_cublasdx(ctx,
                             symmetric_a_view,
                             B,
                             C,
                             alpha,
                             beta,
                             Transpose::NoTrans,
                             Transpose::NoTrans,
                             ComputePrecision::Default);
    }

    return gemm_cublasdx(ctx,
                         B,
                         symmetric_a_view,
                         C,
                         alpha,
                         beta,
                         Transpose::NoTrans,
                         Transpose::NoTrans,
                         ComputePrecision::Default);
}

} // namespace

bool symm_use_cuda_custom(const Queue& ctx,
                          const MatrixView<float, MatrixFormat::Dense>& A,
                          const MatrixView<float, MatrixFormat::Dense>& B,
                          const MatrixView<float, MatrixFormat::Dense>& C,
                          Side side,
                          Uplo) {
    if (ctx.device().type != DeviceType::GPU) {
        return false;
    }

    if (!symm_problem_supported(A, B, C, side)) {
        return false;
    }

    const auto request = symm_variant_request();
    if (request == SymmVariantRequest::Vendor) {
        return false;
    }

    if (request == SymmVariantRequest::Auto) {
        return symm_prefer_cuda_custom_heuristic(A, B, C, side);
    }

    return request == SymmVariantRequest::CuBLASDx;
}

Event symm_cuda_custom(Queue& ctx,
                       const MatrixView<float, MatrixFormat::Dense>& A,
                       const MatrixView<float, MatrixFormat::Dense>& B,
                       const MatrixView<float, MatrixFormat::Dense>& C,
                       float alpha,
                       float beta,
                       Side side,
                       Uplo uplo) {
    if (!symm_problem_supported(A, B, C, side)) {
        return symm_vendor_cuda_raw(ctx, A, B, C, alpha, beta, side, uplo);
    }

    const auto variant = cublasdx_gemm_select_variant(side == Side::Left ? A : B,
                                                      side == Side::Left ? B : A,
                                                      C,
                                                      Transpose::NoTrans,
                                                      Transpose::NoTrans);
    if (variant == cublasdx_gemm::CuBLASDxGemmVariant::VendorFallback ||
        !cublasdx_gemm_variant_available(variant) || !symm_cublasdx::available()) {
        return symm_cublasdx_fallback_gemm(ctx, A, B, C, alpha, beta, side, uplo);
    }

    symm_cublasdx::SymmLaunchDescriptor desc{};
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
    desc.k = A.rows();
    desc.batch = A.batch_size();
    desc.alpha = alpha;
    desc.beta = beta;

    BATCHLAS_KERNEL_TRACE_SCOPE("symm_cuda_custom.fused");
    const cudaError_t status = symm_cublasdx::launch_float(variant,
                                                            desc,
                                                            side,
                                                            uplo,
                                                            cuda_stream_from_queue(ctx));
    if (status == cudaErrorNotSupported) {
        return symm_cublasdx_fallback_gemm(ctx, A, B, C, alpha, beta, side, uplo);
    }
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("cuBLASDx fused SYMM launch failed: ") + cudaGetErrorString(status));
    }

    return ctx.create_event_after_external_work();
}

} // namespace batchlas::backend