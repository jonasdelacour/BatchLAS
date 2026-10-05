#include "syr2k_custom_dispatch.hh"

#include "syr2k_triangular_tiles.hh"
#include "route_common.hh"
#include "level3_coverage.hh"
#include "level3_fused.hh"
#include "level3_vendor_fallback.hh"

#include "../util/kernel-trace.hh"

#include <cctype>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace batchlas::backend {

namespace {

// BATCHLAS_SYR2K_ROUTE: vendor; triangular or native, the tile-masked kernel that
// computes only the requested half of C; cublasdx, the fused MathDx kernel (throws
// when it cannot run).
detail::Level3Pin syr2k_pin() {
    using detail::Level3Pin;
    return detail::level3_pin("syr2k", {Level3Pin::Native, Level3Pin::Vendor, Level3Pin::Triangular,
                                        Level3Pin::Cublasdx});
}

bool syr2k_problem_supported(const MatrixView<float, MatrixFormat::Dense>& A,
                             const MatrixView<float, MatrixFormat::Dense>& B,
                             const MatrixView<float, MatrixFormat::Dense>& C,
                             Transpose transA) {
    if (transA == Transpose::ConjTrans) {
        return false;
    }
    if (C.rows() != C.cols()) {
        return false;
    }
    if (A.batch_size() != B.batch_size() || B.batch_size() != C.batch_size()) {
        return false;
    }

    const int n = C.rows();
    const int a_n = transA == Transpose::NoTrans ? A.rows() : A.cols();
    const int b_n = transA == Transpose::NoTrans ? B.rows() : B.cols();
    const int a_k = transA == Transpose::NoTrans ? A.cols() : A.rows();
    const int b_k = transA == Transpose::NoTrans ? B.cols() : B.rows();
    return a_n == n && b_n == n && a_k == b_k && n > 0 && a_k > 0;
}

// The tile-masked kernel indexes every operand as base + batch * stride, so a
// batch whose members differ in shape or live at unrelated pointers is out of
// reach.
bool syr2k_triangular_supported(const MatrixView<float, MatrixFormat::Dense>& A,
                                const MatrixView<float, MatrixFormat::Dense>& B,
                                const MatrixView<float, MatrixFormat::Dense>& C) {
    return !A.is_heterogeneous() && !B.is_heterogeneous() && !C.is_heterogeneous();
}

// Where the fused kernel beats the vendor. The vendor route is a host loop over
// cublasSsyr2k, one launch per batch member, against one launch for the whole
// batch here, so the two are only ever close at a batch of one and the vendor
// pays double from two members up.
//
// Measured on RTX 4090 / sm_89 in float over n in 8..3072 x k in 4..2048 x
// batch in 1..1024. From batch 2 the kernel won every shape in the grid: 1.06x
// at n = 3072, 1.12x at n = 1024, 1.3-1.4x through the middle, and up to 226x
// where n is small enough that the whole cost is the launch. Neither n nor k
// nor the tile count enters, because none of them changes which side of that
// per-launch difference a shape falls on.
//
// A batch of one does not sort by anything: the vendor wins by 1.18-1.60x below
// n = 1280 and again by 1.16x at n = 3072, the kernel wins by 1.02-1.71x
// between, and by 4-10x the vendor wins on a deep k with a small n, where the
// kernel has a single block and cuBLAS splits the reduction. There is no
// threshold in n to be had, so the batch of one is left with the vendor.
bool syr2k_prefer_triangular_tiles(const MatrixView<float, MatrixFormat::Dense>& A) {
    return A.batch_size() >= 2;
}

} // namespace

bool syr2k_cuda_custom_forced() {
    return syr2k_pin() == detail::Level3Pin::Cublasdx;
}

bool syr2k_use_cuda_custom(const Queue& ctx,
                           const MatrixView<float, MatrixFormat::Dense>& A,
                           const MatrixView<float, MatrixFormat::Dense>& B,
                           const MatrixView<float, MatrixFormat::Dense>& C,
                           Uplo,
                           Transpose transA) {
    using detail::Level3Pin;
    const Level3Pin pin = syr2k_pin();
    if (pin != Level3Pin::Auto && pin != Level3Pin::Vendor) {
        return true;
    }
    if (pin == Level3Pin::Vendor || !detail::is_gpu_queue(ctx) ||
        !syr2k_problem_supported(A, B, C, transA) || !syr2k_triangular_supported(A, B, C)) {
        return false;
    }
    // The tile-masked kernel is the only custom route that respects the
    // triangle, so it is the only one the automatic choice may leave the vendor
    // for, and its own threshold is the whole decision.
    return syr2k_prefer_triangular_tiles(A);
}

Event syr2k_cuda_custom(Queue& ctx,
                        const MatrixView<float, MatrixFormat::Dense>& A,
                        const MatrixView<float, MatrixFormat::Dense>& B,
                        const MatrixView<float, MatrixFormat::Dense>& C,
                        float alpha,
                        float beta,
                        Uplo uplo,
                        Transpose transA) {
    const auto rec = [&](const char* taken, bool native_supported) {
        detail::record_level3_route(Op::syr2k, taken,
                                    C.rows(), C.cols(),
                                    transA == Transpose::NoTrans ? A.cols() : A.rows(),
                                    A.batch_size(), native_supported,
                                    {uplo, Side::Left, Diag::NonUnit, transA});
    };

    using detail::Level3Pin;
    const Level3Pin pin = syr2k_pin();
    const bool forced = pin == Level3Pin::Cublasdx;
    if (!detail::is_gpu_queue(ctx)) {
        if (forced) {
            detail::throw_forced_cublasdx_unavailable("syr2k", "the active queue is not a GPU queue");
        }
        rec("vendor", false);
        return detail::syr2k_vendor_fallback(ctx, A, B, C, alpha, beta, uplo, transA);
    }
    if (!syr2k_problem_supported(A, B, C, transA)) {
        if (forced) {
            detail::throw_forced_cublasdx_unavailable("syr2k", "the problem shape or transpose mode is unsupported");
        }
        rec("vendor", false);
        return detail::syr2k_vendor_fallback(ctx, A, B, C, alpha, beta, uplo, transA);
    }

    if (!forced) {
        if (syr2k_triangular_supported(A, B, C)) {
            rec("triangular", true);
            return detail::syr2k_triangular_tiles(ctx, A, B, C, alpha, beta, uplo, transA);
        }
        if (pin == Level3Pin::Triangular) {
            throw std::invalid_argument("syr2k: BATCHLAS_SYR2K_ROUTE=triangular cannot run a heterogeneous batch");
        }
        rec("vendor", false);
        return detail::syr2k_vendor_fallback(ctx, A, B, C, alpha, beta, uplo, transA);
    }

    // There is no uplo-respecting fallback for a fused kernel that did not run, so a
    // cublasdx pin it cannot serve throws rather than overwrite the caller's triangle.
    auto fused = detail::syr2k_fused_try(ctx, A, B, C, alpha, beta, uplo, transA);
    if (fused.outcome == detail::FusedResult::Outcome::Ran) {
        rec("cublasdx", true);
        return std::move(fused.event);
    }
    if (fused.outcome == detail::FusedResult::Outcome::NoKernel) {
        detail::throw_forced_cublasdx_unavailable(
            "syr2k", "no compatible fused kernel is available in this build for the requested problem");
    }
    detail::throw_forced_cublasdx_unavailable(
        "syr2k", "the fused kernel exists but this device refused to launch it");
}

} // namespace batchlas::backend
