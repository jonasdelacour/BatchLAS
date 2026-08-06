#include "syrk_custom_dispatch.hh"

#include "gemm_cublasdx_dispatch.hh"
#include "gemm_variant.hh"
#include "syrk_cublasdx_fused.hh"
#include "syrk_triangular_tiles.hh"
#include "cublasdx_dispatch_common.hh"

#include "../util/kernel-trace.hh"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace batchlas::backend {

namespace {

constexpr int kSyrkCublasDxTile = 32;

// BATCHLAS_SYRK_VARIANT selects a route. The two custom routes are named
// separately so each stays independently measurable and testable: `triangular`
// is the tile-masked kernel that computes only the requested half of C,
// `cublasdx` the fused kernel, and `gemm` the full n x n batched GEMM the
// triangular route replaces.
//
// `gemm` computes and stores both triangles, which is not what SYRK means: the
// half the caller did not name is the caller's storage. It exists to measure
// the arithmetic the triangular route saves, and the automatic choice never
// selects it -- reaching it takes naming it here.
enum class SyrkRoute {
    Auto,
    Vendor,
    Fused,
    Triangular,
    Gemm,
};

SyrkRoute syrk_route_request() {
    const char* raw = std::getenv("BATCHLAS_SYRK_VARIANT");
    if (!raw) {
        return SyrkRoute::Auto;
    }

    std::string value(raw);
    for (char& ch : value) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }

    if (value == "vendor") {
        return SyrkRoute::Vendor;
    }
    if (value == "cublasdx" || value == "dx" || value == "custom") {
        return SyrkRoute::Fused;
    }
    if (value == "triangular" || value == "tiles") {
        return SyrkRoute::Triangular;
    }
    if (value == "gemm") {
        return SyrkRoute::Gemm;
    }
    return SyrkRoute::Auto;
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

// The tile-masked kernel indexes both operands as base + batch * stride, so a
// batch whose members differ in shape or live at unrelated pointers is out of
// reach.
bool syrk_triangular_supported(const MatrixView<float, MatrixFormat::Dense>& A,
                               const MatrixView<float, MatrixFormat::Dense>& C) {
    return !A.is_heterogeneous() && !C.is_heterogeneous();
}

// Where skipping the tiles outside the triangle starts paying for the
// tile-masked kernel's lower per-tile rate. Two conditions, both measured on
// RTX 4090 / sm_89 in float over n in 64..2048 x batch in 1..512, against the
// full n x n batched GEMM this replaces:
//
//   - n has to be past 256. A tile grid narrower than three 128-wide tiles a
//     side is more than half diagonal, and a diagonal tile is computed whole
//     and then masked, so at n = 256 only one tile in four is saved. That does
//     not cover the gap to cuBLAS per tile: n = 256 measured anywhere between
//     0.84x and 1.22x depending on where its grid happened to fall against a
//     wave boundary, which is no win at all. From n = 384 up every saturated
//     shape won, and the win grows with n as the diagonal thins out -- 1.45x
//     at n = 512 batch 512, 1.63x at n = 1024 batch 64, 1.71x at n = 2048
//     batch 16.
//   - the grid has to fill the device. The 128 SMs hold two of these
//     256-thread blocks apiece, and below ~160 blocks the triangular route
//     lost (1.14x slower at 144 blocks, 1.25x at 136) where from 168 up it won
//     (0.71x).
//
// k does not enter: it only deepens each block's reduction, which moves both
// routes together.
bool syrk_prefer_triangular_tiles(const MatrixView<float, MatrixFormat::Dense>& A,
                                  const MatrixView<float, MatrixFormat::Dense>& C,
                                  Transpose transA) {
    const int n = C.rows();
    const int k = transA == Transpose::NoTrans ? A.cols() : A.rows();
    if (detail::triangular_tiles_per_side(n) < 3 || k < detail::kTriangularTileK) {
        return false;
    }
    return static_cast<long long>(A.batch_size()) * detail::triangular_tile_count(n) >= 160;
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
    const auto route = syrk_route_request();
    if (route != SyrkRoute::Auto && route != SyrkRoute::Vendor) {
        return true;
    }
    if (route == SyrkRoute::Vendor || !detail::is_gpu_queue(ctx) ||
        !syrk_problem_supported(A, C, transA) || !syrk_triangular_supported(A, C)) {
        return false;
    }
    // The tile-masked kernel is the only custom route that respects the
    // triangle, so it is the only one the automatic choice may leave the vendor
    // for. Its own threshold says where it beats the full n x n GEMM; below
    // that the question is instead whether it beats a host loop over
    // cublasSsyrk, which the cuBLASDx heuristic already answers -- one launch
    // per batch member costs about 9 us, so anything with a batch at all is
    // better off here even where the tile grid is half diagonal.
    return syrk_prefer_triangular_tiles(A, C, transA) ||
        syrk_prefer_cuda_custom_heuristic(A, C, transA);
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

    const auto route = syrk_route_request();
    if (route == SyrkRoute::Gemm) {
        return syrk_cublasdx_fallback_gemm(ctx, A, C, alpha, beta, transA);
    }
    const bool triangular = route == SyrkRoute::Triangular || route == SyrkRoute::Auto;
    if (triangular && syrk_triangular_supported(A, C)) {
        return detail::syrk_triangular_tiles(ctx, A, C, alpha, beta, uplo, transA);
    }
    if (route == SyrkRoute::Auto) {
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
