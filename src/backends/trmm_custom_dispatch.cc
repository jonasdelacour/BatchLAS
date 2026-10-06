#include "trmm_custom_dispatch.hh"

#include "trmm_triangular_tiles.hh"
#include "route_common.hh"
#include "level3_coverage.hh"
#include "level3_fused.hh"
#include "level3_vendor_fallback.hh"

#include <batchlas/blas/dispatch/route.hh>
#include <batchlas/blas/dispatch/route_env.hh>

#include "../util/kernel-trace.hh"

#include <cctype>
#include <cstdlib>
#include <string>
#include <algorithm>
#include <stdexcept>

namespace batchlas::backend {

namespace {

constexpr int kTrmmCublasDxTile = 32;

// The ONE parse of BATCHLAS_TRMM_VARIANT; do not add a second reader.
// evidence: docs/perf/level3.md#level-3-one-route-parse-per-variable
dispatch::Route trmm_route_request() {
    const auto parsed = dispatch::parse_route_env(dispatch::Op::trmm);
    return parsed.found ? parsed.route
                        : dispatch::legacy_unset_default(dispatch::Op::trmm);
}

// BATCHLAS_TRMM_VARIANT=triangular pins the tile kernel, =vendor the expansion
// plus GEMM it replaces, so the two stay independently measurable.
bool trmm_triangular_requested() {
    return trmm_route_request().algo == dispatch::Algorithm::TriangularTiles;
}

// Every left-side float problem with a homogeneous batch (base + batch*stride
// indexing); uplo, trans and diag are loop bounds and a staging mask.
bool trmm_triangular_supported(const MatrixView<float, MatrixFormat::Dense>& A,
                               const MatrixView<float, MatrixFormat::Dense>& B,
                               const MatrixView<float, MatrixFormat::Dense>& C,
                               Side side) {
    if (side != Side::Left) {
        return false;
    }
    if (A.rows() != A.cols() || A.rows() != C.rows()) {
        return false;
    }
    if (A.batch_size() != B.batch_size() || B.batch_size() != C.batch_size()) {
        return false;
    }
    if (B.rows() != C.rows() || B.cols() != C.cols()) {
        return false;
    }
    if (A.is_heterogeneous() || B.is_heterogeneous() || C.is_heterogeneous()) {
        return false;
    }
    return C.rows() > 0 && C.cols() > 0;
}

// No m threshold, deliberately: the rival here is the expansion, not a GEMM.
// evidence: docs/perf/level3.md#trmm-tiles-have-no-threshold

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

bool trmm_route_prefers_vendor() {
    const auto r = trmm_route_request();
    return dispatch::is_plain_vendor(r);
}

bool trmm_cuda_custom_forced() {
    return trmm_route_request().algo == dispatch::Algorithm::FusedDevice;
}

bool trmm_use_cuda_custom(const Queue& ctx,
                          const MatrixView<float, MatrixFormat::Dense>& A,
                          const MatrixView<float, MatrixFormat::Dense>& B,
                          const MatrixView<float, MatrixFormat::Dense>& C,
                          Side side,
                          Uplo uplo,
                          Transpose transA,
                          Diag) {
    // `=vendor` must keep meaning the vendor: it is the only "before" to measure against.
    if (detail::is_gpu_queue(ctx) && trmm_triangular_supported(A, B, C, side) &&
        (trmm_triangular_requested() ||
         !dispatch::is_plain_vendor(trmm_route_request()))) {
        return true;
    }
    const auto request = trmm_route_request();
    const bool problem_supported = trmm_problem_supported(A, B, C, side, uplo, transA);
    return detail::should_use_cublasdx(ctx,
                                       request,
                                       dispatch::Route{dispatch::Origin::Vendor, dispatch::Algorithm::Auto},
                                       dispatch::Route{dispatch::Origin::Vendor, dispatch::Algorithm::FusedDevice},
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
    // Coverage record: beside every return, never in place of one. uplo/diag are
    // in the key on purpose. evidence: docs/perf/level3.md#the-trmm-poison-test
    const auto rec = [&](dispatch::Route taken, bool native_supported) {
        detail::record_level3_route(dispatch::Op::trmm, taken,
                                    C.rows(), C.cols(), A.rows(),
                                    A.batch_size(), native_supported,
                                    {uplo, side, diag, transA});
    };

    const bool forced = trmm_cuda_custom_forced();
    if (!detail::is_gpu_queue(ctx)) {
        if (forced) {
            throw_forced_trmm_unavailable("the active queue is not a GPU queue");
        }
        rec(dispatch::Route{dispatch::Origin::Vendor, dispatch::Algorithm::Auto}, false);
        return detail::trmm_vendor_fallback(ctx, A, B, C, alpha, side, uplo, transA, diag);
    }
    // Auto takes the tile kernel wherever it fits.
    if (trmm_triangular_supported(A, B, C, side) &&
        (trmm_triangular_requested() ||
         (!forced && !dispatch::is_plain_vendor(trmm_route_request())))) {
        rec(dispatch::Route{dispatch::Origin::Native, dispatch::Algorithm::TriangularTiles}, true);
        return detail::trmm_triangular_tiles(ctx, A, B, C, alpha, uplo, transA, diag);
    }
    if (!trmm_problem_supported(A, B, C, side, uplo, transA)) {
        if (forced) {
            throw_forced_trmm_unavailable("only left/lower/notrans float problems with matching dense batches are currently supported");
        }
        rec(dispatch::Route{dispatch::Origin::Vendor, dispatch::Algorithm::Auto}, false);
        return detail::trmm_vendor_fallback(ctx, A, B, C, alpha, side, uplo, transA, diag);
    }

    // Both non-Ran outcomes fall back to the vendor, with distinct messages when forced.
    auto fused = detail::trmm_fused_try(ctx, A, B, C, alpha, side, uplo, transA, diag);
    if (fused.outcome == detail::FusedResult::Outcome::Ran) {
        rec(dispatch::Route{dispatch::Origin::Vendor, dispatch::Algorithm::FusedDevice}, true);
        return std::move(fused.event);
    }
    if (forced) {
        if (fused.outcome == detail::FusedResult::Outcome::NoKernel) {
            throw_forced_trmm_unavailable("no compatible fused kernel is available in this build for the requested problem");
        }
        throw_forced_trmm_unavailable("the current device or matrix layout does not satisfy the fused kernel requirements");
    }

    rec(dispatch::Route{dispatch::Origin::Vendor, dispatch::Algorithm::Auto}, true);
    return detail::trmm_vendor_fallback(ctx, A, B, C, alpha, side, uplo, transA, diag);
}

} // namespace batchlas::backend