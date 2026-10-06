#include "syrk_custom_dispatch.hh"

#include "syrk_gram_tiles.hh"
#include "syrk_triangular_tiles.hh"
#include "route_common.hh"
#include "level3_coverage.hh"
#include "level3_fused.hh"
#include "level3_vendor_fallback.hh"

// The terminal GEMM is the PUBLIC entry point, whose header is vendor-free.
#include <batchlas/blas/functions/gemm.hh>

#include <batchlas/blas/dispatch/route.hh>
#include <batchlas/blas/dispatch/route_env.hh>

#include "../util/kernel-trace.hh"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace batchlas::backend {

namespace {

constexpr int kSyrkCublasDxTile = 32;

// BATCHLAS_SYRK_VARIANT: `triangular` (wide C), `gram` (C within one tile),
// `cublasdx` (fused), `gemm` (full GEMM that writes BOTH triangles; measurement
// only, never Auto). Legacy "custom" is the fused kernel, "gemm" a vendor route.
// evidence: docs/perf/level3.md#level-3-one-route-parse-per-variable
dispatch::Route syrk_route_request() {
    const auto parsed = dispatch::parse_route_env(dispatch::Op::syrk);
    return parsed.found ? parsed.route
                        : dispatch::legacy_unset_default(dispatch::Op::syrk);
}

bool syrk_route_is(dispatch::Algorithm a) {
    return syrk_route_request().algo == a;
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

// Where skipping off-triangle tiles beats the full GEMM: at least three tiles a
// side (n >= 257) and >= 160 blocks to fill the device; k does not enter.
// Gate-only: see syrk_use_cuda_custom for why n = 256 still reaches the kernel.
// evidence: docs/perf/level3.md#syrk-triangular-tiles
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

// The Gram kernel serves n within one tile, where the alternative is a host
// loop over cublasSsyrk; no threshold to tune. evidence: docs/perf/level3.md#syrk-gram-tiles
bool syrk_prefer_gram_tiles(const MatrixView<float, MatrixFormat::Dense>& C) {
    return C.rows() <= detail::kGramMaxTile;
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
    return ::batchlas::gemm<Backend::CUDA, float>(ctx, A, A, C, alpha, beta, transA, transB, ComputePrecision::Default);
}

} // namespace

bool syrk_route_prefers_vendor() {
    const auto r = syrk_route_request();
    // DiagFullGemm is a vendor route but NOT "prefer the vendor syrk".
    return dispatch::is_plain_vendor(r);
}

bool syrk_route_requests_gram() {
    return syrk_route_is(dispatch::Algorithm::GramTiles);
}

bool syrk_use_cuda_custom(const Queue& ctx,
                          const MatrixView<float, MatrixFormat::Dense>& A,
                          const MatrixView<float, MatrixFormat::Dense>& C,
                          Uplo,
                          Transpose transA) {
    const auto route = syrk_route_request();
    if (route.origin != dispatch::Origin::Auto && !dispatch::is_plain_vendor(route)) {
        return true;
    }
    if (dispatch::is_plain_vendor(route) || !detail::is_gpu_queue(ctx) ||
        !syrk_problem_supported(A, C, transA) || !syrk_triangular_supported(A, C)) {
        return false;
    }
    // Only the two tile kernels respect the triangle, so only they may replace
    // the vendor. The third disjunct deliberately admits shapes below the
    // triangular window (e.g. n = 256): there the rival is a ~9 us-per-item loop.
    // evidence: docs/perf/level3.md#where-the-decision-actually-happens
    return syrk_prefer_gram_tiles(C) ||
        syrk_prefer_triangular_tiles(A, C, transA) ||
        syrk_prefer_cuda_custom_heuristic(A, C, transA);
}

Event syrk_cuda_custom(Queue& ctx,
                       const MatrixView<float, MatrixFormat::Dense>& A,
                       const MatrixView<float, MatrixFormat::Dense>& C,
                       float alpha,
                       float beta,
                       Uplo uplo,
                       Transpose transA) {
    // Coverage record: beside every return, never in place of one (level3_coverage.hh).
    const auto rec = [&](dispatch::Route taken, bool native_supported) {
        detail::record_level3_route(dispatch::Op::syrk, taken,
                                    C.rows(), C.cols(),
                                    transA == Transpose::NoTrans ? A.cols() : A.rows(),
                                    A.batch_size(), native_supported,
                                    {uplo, Side::Left, Diag::NonUnit, transA});
    };

    if (!syrk_problem_supported(A, C, transA)) {
        rec(dispatch::Route{dispatch::Origin::Vendor, dispatch::Algorithm::Auto}, false);
        return detail::syrk_vendor_fallback(ctx, A, C, alpha, beta, uplo, transA);
    }

    const auto route = syrk_route_request();
    if (route.algo == dispatch::Algorithm::DiagFullGemm) {
        rec(dispatch::Route{dispatch::Origin::Vendor, dispatch::Algorithm::DiagFullGemm}, true);
        return syrk_cublasdx_fallback_gemm(ctx, A, C, alpha, beta, transA);
    }
    if (syrk_triangular_supported(A, C)) {
        // A C within one tile goes to the Gram kernel; either can be pinned by name.
        const bool gram = route.algo == dispatch::Algorithm::GramTiles ||
            (route.origin == dispatch::Origin::Auto && syrk_prefer_gram_tiles(C));
        if (gram) {
            rec(dispatch::Route{dispatch::Origin::Native, dispatch::Algorithm::GramTiles}, true);
            return detail::syrk_gram_tiles(ctx, A, C, alpha, beta, uplo, transA);
        }
        if (route.algo == dispatch::Algorithm::TriangularTiles ||
            route.origin == dispatch::Origin::Auto) {
            rec(dispatch::Route{dispatch::Origin::Native, dispatch::Algorithm::TriangularTiles}, true);
            return detail::syrk_triangular_tiles(ctx, A, C, alpha, beta, uplo, transA);
        }
    }
    if (route.origin == dispatch::Origin::Auto) {
        // Heterogeneous batch: no tile kernel can serve it, so native_supported is false.
        rec(dispatch::Route{dispatch::Origin::Vendor, dispatch::Algorithm::Auto}, false);
        return detail::syrk_vendor_fallback(ctx, A, C, alpha, beta, uplo, transA);
    }

    auto fused = detail::syrk_fused_try(ctx, A, C, alpha, beta, uplo, transA);
    if (fused.outcome == detail::FusedResult::Outcome::Ran) {
        rec(dispatch::Route{dispatch::Origin::Vendor, dispatch::Algorithm::FusedDevice}, true);
        return std::move(fused.event);
    }

    // Record the route TAKEN: without MathDx every forced FusedDevice lands here.
    rec(dispatch::Route{dispatch::Origin::Vendor, dispatch::Algorithm::DiagFullGemm}, true);
    return syrk_cublasdx_fallback_gemm(ctx, A, C, alpha, beta, transA);
}

} // namespace batchlas::backend
