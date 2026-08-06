#include "symm_custom_dispatch.hh"

#include "gemm_cublasdx_dispatch.hh"
#include "gemm_variant.hh"
#include "symm_cublasdx_fused.hh"
#include "cublasdx_dispatch_common.hh"
#include "triangular_expand.hh"

#include "../util/kernel-trace.hh"

#include <util/mempool.hh>

#include <algorithm>
#include <cstddef>
#include <stdexcept>

namespace batchlas::backend {

namespace {

// Tile edge of the expansion kernel below, and the number of columns a work
// group covers per pass over it.
constexpr int kSymmExpandTile = 32;
constexpr int kSymmExpandGroupCols = 8;

// Where the expansion route starts beating the vendor's per-batch cublasSsymm
// loop. Measured on sm_89 in float over square shapes, n in 16..2048 and batch
// in 1..512: the expansion wins by 1.2x to 72x everywhere except batch <= 2
// with n <= 128, where the call is launch-bound and the two extra launches cost
// more than the loop they replace -- there it loses by up to 1.9x. At n = 256
// even batch 1 goes the other way (1.26x).
constexpr int kSymmExpandMinBatch = 4;
constexpr int kSymmExpandMinDim = 256;

enum class SymmVariantRequest {
    Vendor,
    CuBLASDx,
    Auto,
};

SymmVariantRequest symm_variant_request() {
    return detail::parse_cublasdx_variant_request("BATCHLAS_SYMM_VARIANT",
                                                  SymmVariantRequest::Vendor,
                                                  SymmVariantRequest::CuBLASDx,
                                                  SymmVariantRequest::Auto);
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
    return A.batch_size() >= kSymmExpandMinBatch || max_dim >= kSymmExpandMinDim;
}

// Materialise the full symmetric matrix that A's referenced triangle stands
// for, so a plain batched GEMM can read it as an ordinary dense operand.
//
// One tile pair per work group, staged through local memory. The mirrored half
// is the reason: the mirror of a coalesced column read is a row write, one
// cache line per element, and at these sizes the expansion is pure bandwidth.
// Going through a tile keeps the read and both writes coalesced and moves
// 1.5 n^2 of traffic, against the 3 n^2 of a copy followed by an in-place
// symmetrize.
Event symm_expand_symmetric(Queue& ctx,
                            const MatrixView<float, MatrixFormat::Dense>& out,
                            const MatrixView<float, MatrixFormat::Dense>& A,
                            Uplo uplo) {
    const int n = A.rows();
    const int batch = A.batch_size();
    const int tiles = detail::ceil_div(n, kSymmExpandTile);
    const bool lower = uplo == Uplo::Lower;

    const float* src = A.data_ptr();
    float* dst = out.data_ptr();
    const int lda = A.ld();
    const int ldo = out.ld();
    const std::size_t stride_a = static_cast<std::size_t>(A.stride());
    const std::size_t stride_o = static_cast<std::size_t>(out.stride());

    // The tile grid covers both triangles and the groups on the unreferenced
    // side exit before their first barrier. Half the groups retire empty, which
    // is cheaper than putting the integer square root of an unranked triangular
    // index in front of every work item.
    const sycl::range<3> global(static_cast<std::size_t>(batch),
                                static_cast<std::size_t>(tiles) * kSymmExpandGroupCols,
                                static_cast<std::size_t>(tiles) * kSymmExpandTile);
    const sycl::range<3> local(1, kSymmExpandGroupCols, kSymmExpandTile);

    ctx->submit([&](sycl::handler& cgh) {
        // Padded by one column so the transposed read strides across all banks.
        auto tile = sycl::local_accessor<float, 1>(
            sycl::range<1>(kSymmExpandTile * (kSymmExpandTile + 1)), cgh);

        cgh.parallel_for(sycl::nd_range<3>(global, local), [=](sycl::nd_item<3> item) {
            const int ti = static_cast<int>(item.get_group(1));
            const int tj = static_cast<int>(item.get_group(2));
            if (ti < tj) {
                return;
            }

            const int b = static_cast<int>(item.get_group(0));
            const int r = static_cast<int>(item.get_local_id(2));
            const int c0 = static_cast<int>(item.get_local_id(1));

            // Row/column origin of the tile inside the referenced triangle.
            const int src_row0 = (lower ? ti : tj) * kSymmExpandTile;
            const int src_col0 = (lower ? tj : ti) * kSymmExpandTile;

            const float* src_batch = src + static_cast<std::size_t>(b) * stride_a;
            float* dst_batch = dst + static_cast<std::size_t>(b) * stride_o;

            for (int c = c0; c < kSymmExpandTile; c += kSymmExpandGroupCols) {
                const int i = src_row0 + r;
                const int j = src_col0 + c;
                tile[c * (kSymmExpandTile + 1) + r] =
                    (i < n && j < n) ? src_batch[static_cast<std::size_t>(j) * lda + i] : 0.0f;
            }

            sycl::group_barrier(item.get_group());

            if (ti == tj) {
                // The two writes below would collide on a diagonal tile, so pick
                // the referenced member of each mirrored pair instead.
                for (int c = c0; c < kSymmExpandTile; c += kSymmExpandGroupCols) {
                    const int i = src_row0 + r;
                    const int j = src_col0 + c;
                    if (i >= n || j >= n) {
                        continue;
                    }
                    const bool referenced = lower ? (r >= c) : (r <= c);
                    dst_batch[static_cast<std::size_t>(j) * ldo + i] =
                        referenced ? tile[c * (kSymmExpandTile + 1) + r]
                                   : tile[r * (kSymmExpandTile + 1) + c];
                }
                return;
            }

            for (int c = c0; c < kSymmExpandTile; c += kSymmExpandGroupCols) {
                if (src_row0 + r < n && src_col0 + c < n) {
                    dst_batch[static_cast<std::size_t>(src_col0 + c) * ldo + (src_row0 + r)] =
                        tile[c * (kSymmExpandTile + 1) + r];
                }
                if (src_col0 + r < n && src_row0 + c < n) {
                    dst_batch[static_cast<std::size_t>(src_row0 + c) * ldo + (src_col0 + r)] =
                        tile[r * (kSymmExpandTile + 1) + c];
                }
            }
        });
    });

    return ctx.get_event();
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
        expansion = symm_expand_symmetric(ctx, expanded, A, uplo);
    }

    // The GEMM runs on the queue's native stream, which an in-order queue shares
    // with the expansion kernel. An out-of-order queue orders nothing across the
    // SYCL/native boundary and offers no event to hang the vendor launch off, so
    // there the dependency has to be waited out.
    if (!ctx.in_order()) {
        expansion.wait();
    }

    if (side == Side::Left) {
        return gemm_cublasdx(ctx,
                             expanded,
                             B,
                             C,
                             alpha,
                             beta,
                             Transpose::NoTrans,
                             Transpose::NoTrans,
                             ComputePrecision::Default);
    }

    return gemm_cublasdx(ctx,
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
    const auto request = symm_variant_request();
    const bool problem_supported = symm_problem_supported(A, B, C, side);
    return detail::should_use_cublasdx(ctx,
                                       request,
                                       SymmVariantRequest::Vendor,
                                       SymmVariantRequest::CuBLASDx,
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
    if (!symm_problem_supported(A, B, C, side)) {
        return symm_vendor_cuda_raw(ctx, A, B, C, alpha, beta, side, uplo);
    }

    const auto variant = cublasdx_gemm_select_variant(side == Side::Left ? A : B,
                                                      side == Side::Left ? B : A,
                                                      C,
                                                      Transpose::NoTrans,
                                                      Transpose::NoTrans);
    if (detail::cublasdx_variant_needs_fallback(variant, symm_cublasdx::available())) {
        return symm_cublasdx_fallback_gemm(ctx, A, B, C, alpha, beta, side, uplo);
    }

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