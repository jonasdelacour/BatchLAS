#include "syr2k_custom_dispatch.hh"

#include "syr2k_triangular_tiles.hh"
#include "route_common.hh"
#include "level3_coverage.hh"
#include "level3_fused.hh"
#include "level3_vendor_fallback.hh"

// The terminal GEMM is the PUBLIC entry point, whose header is vendor-free.
#include <batchlas/blas/functions/gemm.hh>

#include <batchlas/blas/dispatch/route.hh>
#include <batchlas/blas/dispatch/route_env.hh>

#include "../util/kernel-trace.hh"

#include <cctype>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace batchlas::backend {

namespace {

// BATCHLAS_SYR2K_VARIANT: `triangular` (tile kernel), `cublasdx` (fused), `gemm`
// (two full GEMMs that write BOTH triangles; measurement only, never Auto).
// evidence: docs/perf/level3.md#level-3-one-route-parse-per-variable
dispatch::Route syr2k_route_request() {
    const auto parsed = dispatch::parse_route_env(dispatch::Op::syr2k);
    return parsed.found ? parsed.route
                        : dispatch::legacy_unset_default(dispatch::Op::syr2k);
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

// The vendor route is one cublasSsyr2k launch per batch member: from batch 2
// the kernel won every measured shape, and batch 1 has no threshold in n.
// evidence: docs/perf/level3.md#syr2k-triangular-tiles
bool syr2k_prefer_triangular_tiles(const MatrixView<float, MatrixFormat::Dense>& A) {
    return A.batch_size() >= 2;
}

Event syr2k_cublasdx_fallback_gemm(Queue& ctx,
                                   const MatrixView<float, MatrixFormat::Dense>& A,
                                   const MatrixView<float, MatrixFormat::Dense>& B,
                                   const MatrixView<float, MatrixFormat::Dense>& C,
                                   float alpha,
                                   float beta,
                                   Transpose transA) {
    const Transpose transB = transA == Transpose::NoTrans ? Transpose::Trans : Transpose::NoTrans;
    BATCHLAS_KERNEL_TRACE_SCOPE("syr2k_cuda_custom.gemm_fallback");

    // The second product accumulates into the first's C; out-of-order queues
    // order nothing across the SYCL/native boundary, so wait there.
    Event first = ::batchlas::gemm<Backend::CUDA, float>(ctx, A, B, C, alpha, beta, transA, transB, ComputePrecision::Default);
    if (!ctx.in_order()) {
        first.wait();
    }
    return ::batchlas::gemm<Backend::CUDA, float>(ctx, B, A, C, alpha, 1.0f, transA, transB, ComputePrecision::Default);
}

[[noreturn]] void throw_forced_syr2k_unavailable(const std::string& reason) {
    detail::throw_forced_cublasdx_unavailable("BATCHLAS_SYR2K_VARIANT", "SYR2K", reason);
}

} // namespace

bool syr2k_cuda_custom_forced() {
    return syr2k_route_request().algo == dispatch::Algorithm::FusedDevice;
}

bool syr2k_use_cuda_custom(const Queue& ctx,
                           const MatrixView<float, MatrixFormat::Dense>& A,
                           const MatrixView<float, MatrixFormat::Dense>& B,
                           const MatrixView<float, MatrixFormat::Dense>& C,
                           Uplo,
                           Transpose transA) {
    const auto route = syr2k_route_request();
    if (route.origin != dispatch::Origin::Auto && !dispatch::is_plain_vendor(route)) {
        return true;
    }
    if (dispatch::is_plain_vendor(route) || !detail::is_gpu_queue(ctx) ||
        !syr2k_problem_supported(A, B, C, transA) || !syr2k_triangular_supported(A, B, C)) {
        return false;
    }
    // Only the tile kernel respects the triangle, so it is Auto's only non-vendor choice.
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
    // Coverage record: beside every return, never in place of one (level3_coverage.hh).
    const auto rec = [&](dispatch::Route taken, bool native_supported) {
        detail::record_level3_route(dispatch::Op::syr2k, taken,
                                    C.rows(), C.cols(),
                                    transA == Transpose::NoTrans ? A.cols() : A.rows(),
                                    A.batch_size(), native_supported,
                                    {uplo, Side::Left, Diag::NonUnit, transA});
    };

    const auto route = syr2k_route_request();
    const bool forced = route.algo == dispatch::Algorithm::FusedDevice;
    if (!detail::is_gpu_queue(ctx)) {
        if (forced) {
            throw_forced_syr2k_unavailable("the active queue is not a GPU queue");
        }
        rec(dispatch::Route{dispatch::Origin::Vendor, dispatch::Algorithm::Auto}, false);
        return detail::syr2k_vendor_fallback(ctx, A, B, C, alpha, beta, uplo, transA);
    }
    if (!syr2k_problem_supported(A, B, C, transA)) {
        if (forced) {
            throw_forced_syr2k_unavailable("the problem shape or transpose mode is unsupported");
        }
        rec(dispatch::Route{dispatch::Origin::Vendor, dispatch::Algorithm::Auto}, false);
        return detail::syr2k_vendor_fallback(ctx, A, B, C, alpha, beta, uplo, transA);
    }

    if (route.algo == dispatch::Algorithm::DiagFullGemm) {
        rec(dispatch::Route{dispatch::Origin::Vendor, dispatch::Algorithm::DiagFullGemm}, true);
        return syr2k_cublasdx_fallback_gemm(ctx, A, B, C, alpha, beta, transA);
    }
    if (route.algo == dispatch::Algorithm::TriangularTiles ||
        route.origin == dispatch::Origin::Auto) {
        if (syr2k_triangular_supported(A, B, C)) {
            rec(dispatch::Route{dispatch::Origin::Native, dispatch::Algorithm::TriangularTiles}, true);
            return detail::syr2k_triangular_tiles(ctx, A, B, C, alpha, beta, uplo, transA);
        }
        rec(dispatch::Route{dispatch::Origin::Vendor, dispatch::Algorithm::Auto}, false);
        return detail::syr2k_vendor_fallback(ctx, A, B, C, alpha, beta, uplo, transA);
    }

    // Looks wrong, is preserved: syr2k throws on NoKernel even when not forced.
    // evidence: docs/perf/level3.md#forced-route-defects
    auto fused = detail::syr2k_fused_try(ctx, A, B, C, alpha, beta, uplo, transA);
    if (fused.outcome == detail::FusedResult::Outcome::Ran) {
        rec(dispatch::Route{dispatch::Origin::Vendor, dispatch::Algorithm::FusedDevice}, true);
        return std::move(fused.event);
    }
    if (fused.outcome == detail::FusedResult::Outcome::NoKernel) {
        throw_forced_syr2k_unavailable("no compatible fused kernel is available in this build for the requested problem");
    }

    // DeviceUnsupported: must NOT fall back to syr2k_cublasdx_fallback_gemm,
    // which writes BOTH triangles and no single-uplo test would catch it.
    throw_forced_syr2k_unavailable(
        "the fused kernel exists but this device refused to launch it; "
        "no uplo-respecting fused fallback exists, so the request cannot be served");
}

} // namespace batchlas::backend
