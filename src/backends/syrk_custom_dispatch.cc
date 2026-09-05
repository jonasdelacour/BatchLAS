#include "syrk_custom_dispatch.hh"

#include "syrk_gram_tiles.hh"
#include "syrk_triangular_tiles.hh"
#include "route_common.hh"
#include "level3_coverage.hh"
#include "level3_fused.hh"
#include "level3_vendor_fallback.hh"

// WP1 S2: the expansions' terminal GEMM is the PUBLIC entry point, not
// gemm_cublasdx. Vendor-free by inspection -- gemm.hh reaches only
// sycl-device-queue.hh, sycl-span.hh, matrix.hh, enums.hh and
// queue-dispatch.hh.
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

// BATCHLAS_SYRK_VARIANT selects a route. The custom routes are named
// separately so each stays independently measurable and testable: `triangular`
// is the tile-masked kernel that computes only the requested half of a wide C,
// `gram` the single-tile kernel for a narrow C over a long reduction,
// `cublasdx` the fused kernel, and `gemm` the full n x n batched GEMM the
// triangular route replaces.
//
// `gemm` computes and stores both triangles, which is not what SYRK means: the
// half the caller did not name is the caller's storage. It exists to measure
// the arithmetic the triangular route saves, and the automatic choice never
// selects it -- reaching it takes naming it here.
// The private SyrkRoute enum this used to declare is gone: it named the same
// six things dispatch::Route names, in a spelling only this file understood.
// The legacy values are unchanged and pinned by tests/route_vocabulary_tests.cc
// -- including the two that do NOT mean what the canonical vocabulary would
// read them as: "custom" is the fused cuBLASDx kernel here (not the
// register-tiled GEMM family), and "gemm" is a vendor route (it runs through
// gemm_cublasdx). See parse_legacy_route_value.
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

// The single-tile kernel's whole premise is that the tile is sized to n, so it
// serves exactly the range the triangular grid cannot: n no wider than one
// tile. Inside that range it is not a close call and there is no threshold to
// tune -- the alternative is a host loop over cublasSsyrk, which at large batch
// is one to two orders of magnitude off anything batched.
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
    // The DiagFullGemm measurement route is a vendor route but is emphatically
    // NOT "prefer the vendor syrk": it exists to run the full n x n GEMM.
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
    // The two tile-masked kernels are the only custom routes that respect the
    // triangle, so they are the only ones the automatic choice may leave the
    // vendor for. Between them they cover the range: `gram` below one tile,
    // `triangular` from three tiles a side up. Its own threshold says where it
    // beats the full n x n GEMM; below that the question is instead whether it
    // beats a host loop over cublasSsyrk, which the cuBLASDx heuristic already
    // answers -- one launch per batch member costs about 9 us, so anything with
    // a batch at all is better off here even where the tile grid is half
    // diagonal.
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
    // WP1 S0 instrumentation. Every `record` below sits BESIDE a return, never
    // in place of one, and is a no-op unless BATCHLAS_COVERAGE_OUT is set --
    // so this function's decisions are unchanged by construction. See
    // level3_coverage.hh for why these four ops cannot simply use
    // dispatch::resolve_route.
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
        // A narrow C is one tile wide, so the triangular grid has nothing to
        // skip and would charge a full 128-wide tile for it. Auto splits the
        // range at that point; either kernel can still be pinned by name.
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
        // Reached when the batch is heterogeneous, so the tile kernels cannot
        // serve it -- native_supported is false, and that is the distinction
        // the row exists to record.
        rec(dispatch::Route{dispatch::Origin::Vendor, dispatch::Algorithm::Auto}, false);
        return detail::syrk_vendor_fallback(ctx, A, C, alpha, beta, uplo, transA);
    }

    // The fused tail lives in level3_fused_cuda.cc now (WP1 S3).
    auto fused = detail::syrk_fused_try(ctx, A, C, alpha, beta, uplo, transA);
    if (fused.outcome == detail::FusedResult::Outcome::Ran) {
        rec(dispatch::Route{dispatch::Origin::Vendor, dispatch::Algorithm::FusedDevice}, true);
        return std::move(fused.event);
    }

    // The route TAKEN, not the one requested: with MathDx absent the fused
    // kernel is never available, so every forced FusedDevice request lands
    // here. Recording FusedDevice would make the table lie about what ran.
    rec(dispatch::Route{dispatch::Origin::Vendor, dispatch::Algorithm::DiagFullGemm}, true);
    return syrk_cublasdx_fallback_gemm(ctx, A, C, alpha, beta, transA);
}

} // namespace batchlas::backend
