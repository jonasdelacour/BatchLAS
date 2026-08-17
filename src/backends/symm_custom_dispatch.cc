#include "symm_custom_dispatch.hh"

#include "gemm_cublasdx_dispatch.hh"
#include "gemm_variant.hh"
#include "symm_cublasdx_fused.hh"
#include "cublasdx_dispatch_common.hh"
#include "level3_coverage.hh"
#include "level3_vendor_fallback.hh"

// WP1 S2: the expansions' terminal GEMM is the PUBLIC entry point, not
// gemm_cublasdx. Vendor-free by inspection -- gemm.hh reaches only
// sycl-device-queue.hh, sycl-span.hh, matrix.hh, enums.hh and
// queue-dispatch.hh.
#include <batchlas/blas/functions/gemm.hh>
#include "triangular_expand.hh"

#include <batchlas/blas/dispatch/route.hh>
#include <batchlas/blas/dispatch/route_env.hh>

#include "../util/kernel-trace.hh"

#include <batchlas/util/mempool.hh>

#include <algorithm>
#include <cstddef>
#include <stdexcept>

namespace batchlas::backend {

namespace {

// The private three-value enum is gone; the same three states are Route's
// Origin::Vendor / {Vendor, FusedDevice} / Origin::Auto. Legacy spellings are
// unchanged and pinned by tests/route_vocabulary_tests.cc -- note that "custom"
// means the FUSED kernel here, not the register-tiled GEMM family the canonical
// vocabulary reads it as. See parse_legacy_route_value.
dispatch::Route symm_route_request() {
    const auto parsed = dispatch::parse_route_env(dispatch::Op::symm);
    return parsed.found ? parsed.route
                        : dispatch::legacy_unset_default(dispatch::Op::symm);
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

    // Skewed shapes are excluded above because the expansion always costs a
    // full k x k pass, which stops paying for itself once k dwarfs m and n.
    return detail::expansion_preferred(max_dim, A.batch_size());
}

Event symm_cublasdx_fallback_gemm(Queue& ctx,
                                  const MatrixView<float, MatrixFormat::Dense>& A,
                                  const MatrixView<float, MatrixFormat::Dense>& B,
                                  const MatrixView<float, MatrixFormat::Dense>& C,
                                  float alpha,
                                  float beta,
                                  Side side,
                                  Uplo uplo) {
    const int n = A.rows();
    const int ld = detail::expanded_ld<float>(n);

    // Scratch comes from the queue's arena rather than a local Matrix. A Matrix
    // is a fresh managed allocation whose pages are migrated to the device on
    // first touch, which at n=512 batch=512 costs an order of magnitude more
    // than the GEMM it feeds, and it would be freed on return while the kernels
    // reading it have only been enqueued.
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

    // The GEMM runs on the queue's native stream, which an in-order queue shares
    // with the expansion kernel. An out-of-order queue orders nothing across the
    // SYCL/native boundary and offers no event to hang the vendor launch off, so
    // there the dependency has to be waited out.
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
    const auto request = symm_route_request();
    const bool problem_supported = symm_problem_supported(A, B, C, side);
    return detail::should_use_cublasdx(ctx,
                                       request,
                                       dispatch::Route{dispatch::Origin::Vendor, dispatch::Algorithm::Auto},
                                       dispatch::Route{dispatch::Origin::Vendor, dispatch::Algorithm::FusedDevice},
                                       problem_supported,
                                       problem_supported && symm_prefer_cuda_custom_heuristic(A, B, C, side));
}

Event symm_cuda_custom(Queue& ctx,
                       const MatrixView<float, MatrixFormat::Dense>& A,
                       const MatrixView<float, MatrixFormat::Dense>& B,
                       const MatrixView<float, MatrixFormat::Dense>& C,
                       float alpha,
                       float beta,
                       Side side,
                       Uplo uplo) {
    // WP1 S0 instrumentation -- beside every return, never in place of one, and
    // inert unless BATCHLAS_COVERAGE_OUT is set. See level3_coverage.hh.
    const auto rec = [&](dispatch::Route taken, bool native_supported) {
        detail::record_level3_route(dispatch::Op::symm, taken,
                                    C.rows(), C.cols(), A.rows(),
                                    A.batch_size(), native_supported,
                                    {uplo, side, Diag::NonUnit, Transpose::NoTrans});
    };

    if (!symm_problem_supported(A, B, C, side)) {
        rec(dispatch::Route{dispatch::Origin::Vendor, dispatch::Algorithm::Auto}, false);
        return detail::symm_vendor_fallback(ctx, A, B, C, alpha, beta, side, uplo);
    }

    const auto variant = cublasdx_gemm_select_variant(side == Side::Left ? A : B,
                                                      side == Side::Left ? B : A,
                                                      C,
                                                      Transpose::NoTrans,
                                                      Transpose::NoTrans);
    if (detail::cublasdx_variant_needs_fallback(variant, symm_cublasdx::available())) {
        // symm's only portable kernel is the mirrored expansion; everything
        // after it is a GEMM. ExpandGemm names that honestly -- and note it is
        // NOT a claim that the GEMM is native, which is exactly what WP1 S2
        // and S5 have to make true.
        rec(dispatch::Route{dispatch::Origin::Native, dispatch::Algorithm::ExpandGemm}, true);
        return symm_cublasdx_fallback_gemm(ctx, A, B, C, alpha, beta, side, uplo);
    }

    rec(dispatch::Route{dispatch::Origin::Vendor, dispatch::Algorithm::FusedDevice}, true);

    symm_cublasdx::SymmLaunchDescriptor desc{};
    desc.a_ptr = A.data_ptr();
    desc.b_ptr = B.data_ptr();
    desc.c_ptr = C.data_ptr();
    desc.lda = A.ld();
    desc.ldb = B.ld();
    desc.ldc = C.ld();
    desc.stride_a = A.stride();
    desc.stride_b = B.stride();
    desc.stride_c = C.stride();
    desc.m = C.rows();
    desc.n = C.cols();
    desc.k = A.rows();
    desc.batch = A.batch_size();
    desc.alpha = alpha;
    desc.beta = beta;

    BATCHLAS_KERNEL_TRACE_SCOPE("symm_cuda_custom.fused");
    const cudaError_t status = symm_cublasdx::launch_float(variant,
                                                            desc,
                                                            side,
                                                            uplo,
                                                            detail::cuda_stream_from_queue(ctx));
    if (status == cudaErrorNotSupported) {
        return symm_cublasdx_fallback_gemm(ctx, A, B, C, alpha, beta, side, uplo);
    }
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("cuBLASDx fused SYMM launch failed: ") + cudaGetErrorString(status));
    }

    return ctx.create_event_after_external_work();
}

} // namespace batchlas::backend