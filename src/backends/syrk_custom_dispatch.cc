#include "syrk_custom_dispatch.hh"

#include "syrk_gram_tiles.hh"
#include "syrk_triangular_tiles.hh"
#include "route_common.hh"
#include "level3_coverage.hh"
#include "level3_fused.hh"
#include "level3_vendor_fallback.hh"

#include "../util/kernel-trace.hh"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace batchlas::backend {

namespace {

constexpr int kSyrkCublasDxTile = 32;

// BATCHLAS_SYRK_ROUTE: vendor; triangular, the tile-masked kernel that computes only
// the requested half of a wide C; gram, the single-tile kernel for a narrow C over a
// long reduction; native, whichever of the two Auto would take; cublasdx, the fused
// MathDx kernel (throws when it cannot run).
detail::Level3Pin syrk_pin() {
    using detail::Level3Pin;
    return detail::level3_pin("syrk", {Level3Pin::Native, Level3Pin::Vendor, Level3Pin::Triangular,
                                       Level3Pin::Gram, Level3Pin::Cublasdx});
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

} // namespace

bool syrk_route_prefers_vendor() {
    return syrk_pin() == detail::Level3Pin::Vendor;
}

bool syrk_route_requests_gram() {
    return syrk_pin() == detail::Level3Pin::Gram;
}

bool syrk_use_cuda_custom(const Queue& ctx,
                          const MatrixView<float, MatrixFormat::Dense>& A,
                          const MatrixView<float, MatrixFormat::Dense>& C,
                          Uplo,
                          Transpose transA) {
    using detail::Level3Pin;
    const Level3Pin pin = syrk_pin();
    if (pin != Level3Pin::Auto && pin != Level3Pin::Vendor) {
        return true;
    }
    if (pin == Level3Pin::Vendor || !detail::is_gpu_queue(ctx) ||
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
    const auto rec = [&](const char* taken, bool native_supported) {
        detail::record_level3_route(Op::syrk, taken,
                                    C.rows(), C.cols(),
                                    transA == Transpose::NoTrans ? A.cols() : A.rows(),
                                    A.batch_size(), native_supported,
                                    {uplo, Side::Left, Diag::NonUnit, transA});
    };

    using detail::Level3Pin;
    const Level3Pin pin = syrk_pin();
    if (!syrk_problem_supported(A, C, transA)) {
        if (pin == Level3Pin::Cublasdx) {
            detail::throw_forced_cublasdx_unavailable("syrk", "the problem shape or transpose mode is unsupported");
        }
        rec("vendor", false);
        return detail::syrk_vendor_fallback(ctx, A, C, alpha, beta, uplo, transA);
    }

    if (pin == Level3Pin::Cublasdx) {
        auto fused = detail::syrk_fused_try(ctx, A, C, alpha, beta, uplo, transA);
        if (fused.outcome == detail::FusedResult::Outcome::Ran) {
            rec("cublasdx", true);
            return std::move(fused.event);
        }
        detail::throw_forced_cublasdx_unavailable("syrk", "no fused kernel ran for this problem");
    }

    const bool any_tile = pin == Level3Pin::Auto || pin == Level3Pin::Native;
    if (syrk_triangular_supported(A, C)) {
        // A narrow C is one tile wide, so the triangular grid has nothing to skip and
        // would charge a full 128-wide tile for it. Auto splits the range there.
        if (pin == Level3Pin::Gram || (any_tile && syrk_prefer_gram_tiles(C))) {
            rec("gram", true);
            return detail::syrk_gram_tiles(ctx, A, C, alpha, beta, uplo, transA);
        }
        rec("triangular", true);
        return detail::syrk_triangular_tiles(ctx, A, C, alpha, beta, uplo, transA);
    }
    if (!any_tile) {
        throw std::invalid_argument("syrk: BATCHLAS_SYRK_ROUTE=" + std::string(detail::level3_pin_word(pin)) +
                                    " cannot run a heterogeneous batch");
    }
    // A heterogeneous batch: the tile kernels index base + batch * stride.
    rec("vendor", false);
    return detail::syrk_vendor_fallback(ctx, A, C, alpha, beta, uplo, transA);
}

} // namespace batchlas::backend
