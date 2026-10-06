#include "trmm_custom_dispatch.hh"

#include "trmm_triangular_tiles.hh"
#include "route_common.hh"
#include "level3_coverage.hh"
#include "level3_fused.hh"
#include "level3_vendor_fallback.hh"

#include "../util/kernel-trace.hh"

#include <cctype>
#include <cstdlib>
#include <string>
#include <algorithm>
#include <stdexcept>

namespace batchlas::backend {

namespace {

constexpr int kTrmmCublasDxTile = 32;

// BATCHLAS_TRMM_ROUTE: vendor, the expansion plus GEMM; triangular or native, the
// tile kernel; cublasdx, the fused MathDx kernel (throws when it cannot run).
detail::Level3Pin trmm_pin() {
    using detail::Level3Pin;
    return detail::level3_pin("trmm", {Level3Pin::Native, Level3Pin::Vendor, Level3Pin::Triangular,
                                       Level3Pin::Cublasdx});
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
    detail::throw_forced_cublasdx_unavailable("trmm", reason);
}

} // namespace

bool trmm_route_prefers_vendor() {
    return trmm_pin() == detail::Level3Pin::Vendor;
}

bool trmm_cuda_custom_forced() {
    return trmm_pin() == detail::Level3Pin::Cublasdx;
}

bool trmm_use_cuda_custom(const Queue& ctx,
                          const MatrixView<float, MatrixFormat::Dense>& A,
                          const MatrixView<float, MatrixFormat::Dense>& B,
                          const MatrixView<float, MatrixFormat::Dense>& C,
                          Side side,
                          Uplo uplo,
                          Transpose transA,
                          Diag) {
    // `vendor` keeps meaning the vendor even though the tile kernel is the default:
    // it is the "before" a measurement is taken against.
    using detail::Level3Pin;
    const Level3Pin pin = trmm_pin();
    if (pin == Level3Pin::Cublasdx || pin == Level3Pin::Triangular) return true;
    if (pin == Level3Pin::Vendor || !detail::is_gpu_queue(ctx)) return false;
    if (trmm_triangular_supported(A, B, C, side)) return true;
    return pin == Level3Pin::Auto && trmm_problem_supported(A, B, C, side, uplo, transA) &&
           trmm_prefer_cuda_custom_heuristic(A, B);
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
    // uplo/diag are in the coverage key on purpose. evidence: docs/perf/level3.md#the-trmm-poison-test
    const auto rec = [&](const char* taken, bool native_supported) {
        detail::record_level3_route(Op::trmm, taken,
                                    C.rows(), C.cols(), A.rows(),
                                    A.batch_size(), native_supported,
                                    {uplo, side, diag, transA});
    };

    using detail::Level3Pin;
    const Level3Pin pin = trmm_pin();
    const bool forced = pin == Level3Pin::Cublasdx;
    if (!detail::is_gpu_queue(ctx)) {
        if (forced) {
            throw_forced_trmm_unavailable("the active queue is not a GPU queue");
        }
        if (pin == Level3Pin::Triangular) {
            throw std::invalid_argument("trmm: BATCHLAS_TRMM_ROUTE=triangular needs a GPU queue");
        }
        rec("vendor", false);
        return detail::trmm_vendor_fallback(ctx, A, B, C, alpha, side, uplo, transA, diag);
    }
    // The tile kernel is the only route that respects the triangle rather than
    // expanding it, so it is what the automatic choice takes wherever it fits.
    if (!forced && trmm_triangular_supported(A, B, C, side)) {
        rec("triangular", true);
        return detail::trmm_triangular_tiles(ctx, A, B, C, alpha, uplo, transA, diag);
    }
    if (pin == Level3Pin::Triangular) {
        throw std::invalid_argument("trmm: BATCHLAS_TRMM_ROUTE=triangular serves only left-side "
                                    "problems with a homogeneous batch");
    }
    if (!trmm_problem_supported(A, B, C, side, uplo, transA)) {
        if (forced) {
            throw_forced_trmm_unavailable("only left/lower/notrans float problems with matching dense batches are currently supported");
        }
        rec("vendor", false);
        return detail::trmm_vendor_fallback(ctx, A, B, C, alpha, side, uplo, transA, diag);
    }

    if (pin == Level3Pin::Auto || forced) {
        auto fused = detail::trmm_fused_try(ctx, A, B, C, alpha, side, uplo, transA, diag);
        if (fused.outcome == detail::FusedResult::Outcome::Ran) {
            rec("cublasdx", true);
            return std::move(fused.event);
        }
        if (forced) {
            if (fused.outcome == detail::FusedResult::Outcome::NoKernel) {
                throw_forced_trmm_unavailable("no compatible fused kernel is available in this build for the requested problem");
            }
            throw_forced_trmm_unavailable("the current device or matrix layout does not satisfy the fused kernel requirements");
        }
    }

    rec("vendor", true);
    return detail::trmm_vendor_fallback(ctx, A, B, C, alpha, side, uplo, transA, diag);
}

} // namespace batchlas::backend