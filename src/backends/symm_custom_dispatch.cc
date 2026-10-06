#include "symm_custom_dispatch.hh"

#include "route_common.hh"
#include "level3_coverage.hh"
#include "level3_fused.hh"
#include "level3_vendor_fallback.hh"

// The terminal GEMM is the PUBLIC entry point, whose header is vendor-free.
#include <batchlas/blas/functions/gemm.hh>
#include "triangular_expand.hh"

#include "../util/kernel-trace.hh"

#include <batchlas/util/mempool.hh>

#include <algorithm>
#include <cstddef>
#include <stdexcept>

namespace batchlas::backend {

namespace {

// BATCHLAS_SYMM_ROUTE: vendor, cublasdx (the fused MathDx kernel; throws when it
// cannot run), expand or native (the mirrored expansion plus the public gemm, symm's
// only native route).
detail::Level3Pin symm_pin() {
    using detail::Level3Pin;
    return detail::level3_pin("symm", {Level3Pin::Native, Level3Pin::Vendor, Level3Pin::Expand,
                                       Level3Pin::Cublasdx});
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
    if (!squareish || shared_dim != k) {
        return false;
    }

    // Skewed shapes excluded: the k x k expansion stops paying once k dwarfs m, n.
    // evidence: docs/perf/level3.md#symm-and-hemm-expansion-crossover
    return detail::expansion_preferred(max_dim, A.batch_size());
}

Event symm_expand_gemm(Queue& ctx,
                                  const MatrixView<float, MatrixFormat::Dense>& A,
                                  const MatrixView<float, MatrixFormat::Dense>& B,
                                  const MatrixView<float, MatrixFormat::Dense>& C,
                                  float alpha,
                                  float beta,
                                  Side side,
                                  Uplo uplo) {
    const int n = A.rows();
    const int ld = detail::expanded_ld<float>(n);

    // Arena scratch, never a local Matrix: it would be freed while still enqueued.
    // evidence: docs/perf/level3.md#level-3-scratch-expansions-and-their-ceilings
    auto ws = ctx.workspace(detail::expanded_workspace_bytes<float>(ctx, n, A.batch_size()));
    BumpAllocator pool(ws.span());
    auto storage = pool.allocate<float>(ctx, static_cast<std::size_t>(ld) *
                                                 static_cast<std::size_t>(n) *
                                                 static_cast<std::size_t>(A.batch_size()));

    MatrixView<float, MatrixFormat::Dense> expanded(storage.data(), n, n, ld, ld * n, A.batch_size());

    Event expansion;
    {
        BATCHLAS_KERNEL_TRACE_SCOPE("symm_cuda_custom.expand");
        expansion = detail::expand_mirrored<float, /*Conjugate=*/false>(ctx, expanded, A, uplo);
    }

    // Out-of-order queues order nothing across the SYCL/native boundary.
    if (!ctx.in_order()) {
        expansion.wait();
    }

    if (side == Side::Left) {
        return ::batchlas::gemm<Backend::CUDA, float>(ctx,
                             expanded,
                             B,
                             C,
                             alpha,
                             beta,
                             Transpose::NoTrans,
                             Transpose::NoTrans,
                             ComputePrecision::Default);
    }

    return ::batchlas::gemm<Backend::CUDA, float>(ctx,
                         B,
                         expanded,
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
    using detail::Level3Pin;
    const Level3Pin pin = symm_pin();
    if (pin == Level3Pin::Cublasdx) return true;
    if (pin == Level3Pin::Vendor || !detail::is_gpu_queue(ctx) || !symm_problem_supported(A, B, C, side)) {
        return false;
    }
    return pin != Level3Pin::Auto || symm_prefer_cuda_custom_heuristic(A, B, C, side);
}

Event symm_cuda_custom(Queue& ctx,
                       const MatrixView<float, MatrixFormat::Dense>& A,
                       const MatrixView<float, MatrixFormat::Dense>& B,
                       const MatrixView<float, MatrixFormat::Dense>& C,
                       float alpha,
                       float beta,
                       Side side,
                       Uplo uplo) {
    const auto rec = [&](const char* taken, bool native_supported) {
        detail::record_level3_route(Op::symm, taken,
                                    C.rows(), C.cols(), A.rows(),
                                    A.batch_size(), native_supported,
                                    {uplo, side, Diag::NonUnit, Transpose::NoTrans});
    };

    using detail::Level3Pin;
    const Level3Pin pin = symm_pin();
    const bool forced = pin == Level3Pin::Cublasdx;
    if (!symm_problem_supported(A, B, C, side)) {
        if (forced) {
            detail::throw_forced_cublasdx_unavailable("symm", "the problem shape is unsupported");
        }
        rec("vendor", false);
        return detail::symm_vendor_fallback(ctx, A, B, C, alpha, beta, side, uplo);
    }

    if (pin == Level3Pin::Auto || forced) {
        auto fused = detail::symm_fused_try(ctx, A, B, C, alpha, beta, side, uplo);
        if (fused.outcome == detail::FusedResult::Outcome::Ran) {
            rec("cublasdx", true);
            return std::move(fused.event);
        }
        if (forced) {
            detail::throw_forced_cublasdx_unavailable("symm", "no fused kernel ran for this problem");
        }
    }

    rec("expand", true);
    return symm_expand_gemm(ctx, A, B, C, alpha, beta, side, uplo);
}

} // namespace batchlas::backend
