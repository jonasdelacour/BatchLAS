// Native batched ORGQR, blocked tier: Q = H_1 H_2 ... H_k I_{m x n}, i.e. the
// routed ormqr applied to an identity. evidence: docs/perf/qr.md#qr-the-vendor-baseline
//
// Both apply-Q seams must be injected, never defaulted: this driver is
// instantiated per scalar type with no Backend parameter, so only the facade
// can name the routed ormqr. An empty seam throws.

#include <src/ops/orgqr/orgqr_native.hh>

#include <src/queue.hh>
#include <src/util/template-instantiations.hh>

#include <batchlas/util/mempool.hh>
#include <batchlas/tuning_params.hh>

#include <sycl/sycl.hpp>

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace batchlas {
namespace sycl_orgqr {

namespace {

template <typename T> class OrgqrIdentityKernel;
template <typename T> class OrgqrCopyBackKernel;

// The WY block width handed to the apply. Deliberately NOT
// tuning::ormqr_block_size_for_n. Multiple of 16, never below 32 for complex
// (the wide-scalar gemm gate). evidence: docs/perf/qr.md#block-width-evidence
template <typename T>
constexpr int32_t orgqr_nb_for_type() {
    if constexpr (std::is_same_v<T, double>) {
        return 16;
    } else {
        return 32;
    }
}

template <typename T>
inline int32_t orgqr_nb(int m, int n) {
    const int32_t k = static_cast<int32_t>(std::min(m, n));
    return std::max<int32_t>(1, std::min<int32_t>(orgqr_nb_for_type<T>(), std::max(1, k)));
}

// The view the apply writes Q into. The size query passes a null pointer: a
// workspace query may read a view's metadata but never its data.
template <typename T>
inline MatrixView<T, MatrixFormat::Dense> orgqr_c_view(T* p, int m, int n, int batch) {
    return MatrixView<T, MatrixFormat::Dense>(p, m, n, /*ld=*/m,
                                              /*stride=*/static_cast<int>(
                                                  static_cast<std::size_t>(m) *
                                                  static_cast<std::size_t>(n)),
                                              batch, nullptr);
}

// `apply_bytes` is computed outside, against the CALLER's views: a nested size
// query may not be asked about workspace-derived views.
template <typename T>
struct OrgqrWs {
    Span<T> c;
    Span<std::byte> apply;
};

template <typename T>
OrgqrWs<T> orgqr_blocked_layout(Queue& ctx, BumpAllocator& pool,
                                int m, int n, int batch, std::size_t apply_bytes) {
    OrgqrWs<T> ws;
    ws.c = pool.allocate<T>(ctx, static_cast<std::size_t>(m) *
                                     static_cast<std::size_t>(n) *
                                     static_cast<std::size_t>(batch));
    ws.apply = pool.allocate<std::byte>(ctx, apply_bytes);
    return ws;
}

template <typename T>
std::size_t orgqr_apply_bytes(Queue& ctx,
                              const MatrixView<T, MatrixFormat::Dense>& A,
                              Span<T> tau,
                              int m, int n, int batch,
                              const OrgqrApplyQBufferSize<T>& q) {
    if (!q) {
        throw batchlas::internal_error(
            "sycl_orgqr: the apply-Q workspace query was not injected. orgqr's native arm "
            "is ormqr applied to an identity, and only the facade can name the ROUTED "
            "ormqr_buffer_size -- this driver is instantiated per scalar type with no "
            "Backend parameter (see the note at the top of orgqr_blocked.cc).");
    }
    const auto C = orgqr_c_view<T>(nullptr, m, n, batch);
    return q(ctx, A, C, Side::Left, Transpose::NoTrans, tau, orgqr_nb<T>(m, n));
}

// Items per pass: the identity, the apply and the copy-back pass C between kernels through L2,
// and the apply is routed at this batch. An item is A, C and the apply's V, W1 and W2.
// evidence: docs/perf/qr.md#qr-l2-batch-chunking
template <typename T>
int orgqr_blocked_chunk(Queue& ctx, int m, int n, int batch) {
    const int64_t nb = orgqr_nb<T>(m, n);
    const int64_t item = (2 * int64_t(m) * n + int64_t(m) * nb + 2 * nb * n) * static_cast<int64_t>(sizeof(T));
    const auto dev = ctx.device();
    return static_cast<int>(tuning::blocked_chunk_items(
        batch, item, static_cast<int64_t>(dev.get_property(DeviceProperty::GLOBAL_MEM_CACHE_SIZE)),
        static_cast<int64_t>(dev.get_property(DeviceProperty::MAX_COMPUTE_UNITS))));
}

// The largest pass: the last part takes the remainder, and under two chunks there is one.
inline int orgqr_blocked_part(int chunk, int batch) {
    return batch < 2 * chunk ? batch : chunk + batch % chunk;
}

// One part's workspace: its C and the routed apply's own need, asked of the caller's views.
template <typename T>
std::size_t orgqr_part_bytes(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, Span<T> tau,
                             int part, const OrgqrApplyQBufferSize<T>& q) {
    const int m = static_cast<int>(A.rows()), n = static_cast<int>(A.cols());
    const std::size_t apply_bytes = orgqr_apply_bytes<T>(ctx, A.batch_range(0, part), tau, m, n, part, q);
    return workspace_bytes([&](BumpAllocator& pool) {
        return orgqr_blocked_layout<T>(ctx, pool, m, n, part, apply_bytes);
    });
}

}  // namespace

// True for all four types; Auto takes this driver where tuned/orgqr.*.txt ranks
// blocked first (m, n <= 512). evidence: docs/perf/qr.md#qr-route-arms
template <> bool orgqr_blocked_available<float>()                { return true; }
template <> bool orgqr_blocked_available<double>()               { return true; }
template <> bool orgqr_blocked_available<std::complex<float>>()  { return true; }
template <> bool orgqr_blocked_available<std::complex<double>>() { return true; }

template <typename T>
std::size_t orgqr_blocked_buffer_size(Queue& ctx,
                                      const MatrixView<T, MatrixFormat::Dense>& A,
                                      Span<T> tau,
                                      OrgqrApplyQBufferSize<T> apply_q_buffer_size) {
    const int m = static_cast<int>(A.rows());
    const int n = static_cast<int>(A.cols());
    const int batch = static_cast<int>(A.batch_size());
    if (m < 1 || n < 1 || batch < 1) return 0;

    return orgqr_part_bytes<T>(ctx, A, tau, orgqr_blocked_part(orgqr_blocked_chunk<T>(ctx, m, n, batch), batch),
                               apply_q_buffer_size);
}

template <typename T>
Event orgqr_blocked_dispatch(Queue& ctx,
                             const MatrixView<T, MatrixFormat::Dense>& A,
                             Span<T> tau,
                             Span<std::byte> workspace,
                             OrgqrApplyQ<T> apply_q,
                             OrgqrApplyQBufferSize<T> apply_q_buffer_size) {
    const int m = static_cast<int>(A.rows());
    const int n = static_cast<int>(A.cols());
    const int batch = static_cast<int>(A.batch_size());
    const int k = std::min(m, n);

    // can_run's gates are re-applied here: direct callers bypass select::choose.
    if (m < 1 || n < 1 || batch < 1) {
        throw batchlas::invalid_argument("orgqr_blocked: degenerate extents");
    }
    if (m < n) {
        throw batchlas::invalid_argument(
            "orgqr_blocked: n > m is not supported (orgqr's can_run refuses it)");
    }
    if (A.is_heterogeneous()) {
        throw batchlas::invalid_argument("orgqr_blocked: heterogeneous batch is not supported");
    }
    if (ctx.device().type != DeviceType::GPU) {
        throw batchlas::invalid_argument("orgqr_blocked: GPU queues only");
    }
    if (tau.size() < static_cast<std::size_t>(k) * static_cast<std::size_t>(batch)) {
        throw batchlas::invalid_argument("orgqr_blocked: tau span is shorter than k * batch");
    }
    if (!apply_q) {
        throw batchlas::internal_error(
            "sycl_orgqr::orgqr_blocked_dispatch: the apply-Q seam was not injected. orgqr's "
            "native arm is ormqr applied to an identity, and only a layer that can name a "
            "Backend can reach the ROUTED ormqr (see the note at the top of "
            "orgqr_blocked.cc).");
    }

    const int32_t nb = orgqr_nb<T>(m, n);

    // Chunk by chunk: each pass forms its items' Q completely, and the workspace is one part's.
    // A caller that sized a larger shape may hold less than this shape's part needs, so the
    // chunk halves until it fits. evidence: docs/perf/qr.md#qr-l2-batch-chunking
    int chunk = orgqr_blocked_chunk<T>(ctx, m, n, batch);
    while (chunk > 1 && orgqr_part_bytes<T>(ctx, A, tau, orgqr_blocked_part(chunk, batch), apply_q_buffer_size) >
                            workspace.size())
        chunk = (chunk + 1) / 2;
    const int part = orgqr_blocked_part(chunk, batch);
    const std::size_t apply_bytes =
        orgqr_apply_bytes<T>(ctx, A.batch_range(0, part), tau, m, n, part, apply_q_buffer_size);

    BumpAllocator pool(workspace);
    auto ws = orgqr_blocked_layout<T>(ctx, pool, m, n, part, apply_bytes);

    for (int b0 = 0, cb = 0; b0 < batch; b0 += cb) {
        cb = batch - b0 < 2 * chunk ? batch - b0 : chunk;   // the last part takes the remainder
        const auto Apart = A.batch_range(b0, cb);
        const Span<T> tau_part = tau.subspan(static_cast<std::size_t>(b0) * k, static_cast<std::size_t>(cb) * k);
        const auto C = orgqr_c_view<T>(ws.c.data(), m, n, cb);

        // Out of order, the previous part's copy-back may still be reading C.
        if (b0 > 0 && !ctx.in_order()) ctx.wait();

        {
            T* const cp = ws.c.data();
            const std::size_t stride_c =
                static_cast<std::size_t>(m) * static_cast<std::size_t>(n);
            // C := I. Dim 2 is the ROW, not the column: sycl::id<3> makes dim 2 the
            // fastest-varying index and C is column-major with ld m, so indexing the
            // column there would stride m*sizeof(T).
            ctx->parallel_for<OrgqrIdentityKernel<T>>(
                sycl::range<3>(static_cast<std::size_t>(cb), static_cast<std::size_t>(n),
                               static_cast<std::size_t>(m)),
                [=](sycl::id<3> idx) {
                    const std::size_t b = idx[0];
                    const int c = static_cast<int>(idx[1]);
                    const int r = static_cast<int>(idx[2]);
                    cp[b * stride_c + static_cast<std::size_t>(r) +
                       static_cast<std::size_t>(c) * static_cast<std::size_t>(m)] =
                        (r == c) ? T(1) : T(0);
                });
        }

        // The apply reads the identity just written; out-of-order queues need this wait.
        if (!ctx.in_order()) ctx.wait();

        // C := Q C through the injected routed ormqr. Argument order is the POSITIONAL
        // ormqr entry point's; an option struct orders its fields differently.
        (void)apply_q(ctx, Apart, C, Side::Left, Transpose::NoTrans, tau_part, ws.apply, nb);

        if (!ctx.in_order()) ctx.wait();

        // A := C -- orgqr overwrites its input, ormqr writes a separate C.
        {
            const T* const cp = ws.c.data();
            T* const ap = Apart.data_ptr();
            const int lda = A.ld();
            const int stride_a = A.stride();
            const std::size_t stride_c =
                static_cast<std::size_t>(m) * static_cast<std::size_t>(n);
            // Dim 2 is the ROW, as in the fill above.
            ctx->parallel_for<OrgqrCopyBackKernel<T>>(
                sycl::range<3>(static_cast<std::size_t>(cb), static_cast<std::size_t>(n),
                               static_cast<std::size_t>(m)),
                [=](sycl::id<3> idx) {
                    const std::size_t b = idx[0];
                    const int c = static_cast<int>(idx[1]);
                    const int r = static_cast<int>(idx[2]);
                    ap[static_cast<std::ptrdiff_t>(b) * stride_a +
                       static_cast<std::ptrdiff_t>(r) +
                       static_cast<std::ptrdiff_t>(c) * lda] =
                        cp[b * stride_c + static_cast<std::size_t>(r) +
                           static_cast<std::size_t>(c) * static_cast<std::size_t>(m)];
                });
        }
    }

    return ctx.get_event();
}

#define BATCHLAS_ORGQR_BLOCKED_INSTANTIATE(T)                                                 \
    template std::size_t orgqr_blocked_buffer_size<T>(                                        \
        Queue&, const MatrixView<T, MatrixFormat::Dense>&, Span<T>,                           \
        OrgqrApplyQBufferSize<T>);                                                            \
    template Event orgqr_blocked_dispatch<T>(Queue&,                                          \
                                             const MatrixView<T, MatrixFormat::Dense>&,       \
                                             Span<T>, Span<std::byte>, OrgqrApplyQ<T>,        \
                                             OrgqrApplyQBufferSize<T>);

BATCHLAS_ORGQR_BLOCKED_INSTANTIATE(float)
BATCHLAS_ORGQR_BLOCKED_INSTANTIATE(double)
BATCHLAS_ORGQR_BLOCKED_INSTANTIATE(std::complex<float>)
BATCHLAS_ORGQR_BLOCKED_INSTANTIATE(std::complex<double>)

#undef BATCHLAS_ORGQR_BLOCKED_INSTANTIATE

}  // namespace sycl_orgqr
}  // namespace batchlas
