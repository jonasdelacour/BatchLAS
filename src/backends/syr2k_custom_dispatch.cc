#include "syr2k_custom_dispatch.hh"

#include "gemm_cublasdx_dispatch.hh"
#include "gemm_variant.hh"
#include "syr2k_cublasdx_fused.hh"
#include "syr2k_triangular_tiles.hh"
#include "cublasdx_dispatch_common.hh"
#include "level3_coverage.hh"

#include <batchlas/blas/dispatch/route.hh>
#include <batchlas/blas/dispatch/route_env.hh>

#include "../util/kernel-trace.hh"

#include <cctype>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace batchlas::backend {

namespace {

// BATCHLAS_SYR2K_VARIANT selects a route. The custom routes are named
// separately so each stays independently measurable and testable: `triangular`
// is the tile-masked kernel that computes only the requested half of C,
// `cublasdx` the fused kernel, and `gemm` the pair of full n x n batched GEMMs
// the triangular route replaces.
//
// `gemm` computes and stores both triangles, which is not what SYR2K means: the
// half the caller did not name is the caller's storage. It exists to measure
// what the triangular route saves, and the automatic choice never selects it --
// reaching it takes naming it here.
// The private Syr2kRoute enum is gone; see the note in syrk_custom_dispatch.cc.
// Legacy spellings are unchanged and pinned by tests/route_vocabulary_tests.cc.
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

Event syr2k_cublasdx_fallback_gemm(Queue& ctx,
                                   const MatrixView<float, MatrixFormat::Dense>& A,
                                   const MatrixView<float, MatrixFormat::Dense>& B,
                                   const MatrixView<float, MatrixFormat::Dense>& C,
                                   float alpha,
                                   float beta,
                                   Transpose transA) {
    const Transpose transB = transA == Transpose::NoTrans ? Transpose::Trans : Transpose::NoTrans;
    BATCHLAS_KERNEL_TRACE_SCOPE("syr2k_cuda_custom.gemm_fallback");

    // The second product accumulates into the C the first one wrote, so the two
    // have to be ordered. An in-order queue already orders them: both run on
    // its native stream. An out-of-order queue orders nothing across the
    // SYCL/native boundary, so there the first has to be waited out.
    Event first = gemm_cublasdx(ctx, A, B, C, alpha, beta, transA, transB, ComputePrecision::Default);
    if (!ctx.in_order()) {
        first.wait();
    }
    return gemm_cublasdx(ctx, B, A, C, alpha, 1.0f, transA, transB, ComputePrecision::Default);
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
    // WP1 S0 instrumentation -- beside every return, never in place of one, and
    // inert unless BATCHLAS_COVERAGE_OUT is set. See level3_coverage.hh.
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
        return syr2k_vendor_cuda_raw(ctx, A, B, C, alpha, beta, uplo, transA);
    }
    if (!syr2k_problem_supported(A, B, C, transA)) {
        if (forced) {
            throw_forced_syr2k_unavailable("the problem shape or transpose mode is unsupported");
        }
        rec(dispatch::Route{dispatch::Origin::Vendor, dispatch::Algorithm::Auto}, false);
        return syr2k_vendor_cuda_raw(ctx, A, B, C, alpha, beta, uplo, transA);
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
        return syr2k_vendor_cuda_raw(ctx, A, B, C, alpha, beta, uplo, transA);
    }

    const Transpose transB = transA == Transpose::NoTrans ? Transpose::Trans : Transpose::NoTrans;
    const auto variant = cublasdx_gemm_select_variant(A, B, C, transA, transB);
    if (detail::cublasdx_variant_needs_fallback(variant, syr2k_cublasdx::available())) {
        // Note this THROWS rather than falling back, and is not guarded by
        // `forced` -- unlike the two throws above. Pre-existing (a non-fused
        // named route reaching here gets a cuBLASDx message it did not ask
        // for); recorded in WP1_LEVEL3_SPEC.md as out of scope, not fixed in
        // passing.
        throw_forced_syr2k_unavailable("no compatible fused kernel is available in this build for the requested problem");
    }

    rec(dispatch::Route{dispatch::Origin::Vendor, dispatch::Algorithm::FusedDevice}, true);

    syr2k_cublasdx::Syr2kLaunchDescriptor desc{};
    desc.a_ptr = A.data_ptr();
    desc.b_ptr = B.data_ptr();
    desc.c_ptr = C.data_ptr();
    desc.lda = A.ld();
    desc.ldb = B.ld();
    desc.ldc = C.ld();
    desc.stride_a = A.stride();
    desc.stride_b = B.stride();
    desc.stride_c = C.stride();
    desc.n = C.rows();
    desc.k = transA == Transpose::NoTrans ? A.cols() : A.rows();
    desc.batch = A.batch_size();
    desc.alpha = alpha;
    desc.beta = beta;

    BATCHLAS_KERNEL_TRACE_SCOPE("syr2k_cuda_custom.fused");
    const cudaError_t status = syr2k_cublasdx::launch_float(variant,
                                                            desc,
                                                            uplo,
                                                            transA,
                                                            detail::cuda_stream_from_queue(ctx));
    if (status == cudaErrorNotSupported) {
        throw_forced_syr2k_unavailable("the current device or matrix layout does not satisfy the fused kernel requirements");
    }
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("cuBLASDx fused SYR2K launch failed: ") + cudaGetErrorString(status));
    }

    return ctx.create_event_after_external_work();
}

} // namespace batchlas::backend
