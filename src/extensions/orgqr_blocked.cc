// Native batched ORGQR, blocked tier: Q = H_1 H_2 ... H_k I_{m x n}, i.e. the
// routed ormqr applied to an identity. evidence: docs/perf/qr.md#the-vendor-baseline
//
// Both apply-Q seams must be injected, never defaulted: this driver is
// instantiated per scalar type with no Backend parameter, so only the facade
// can name the routed ormqr. An empty seam throws.

#include "orgqr_native.hh"

#include "../queue.hh"
#include "../util/template-instantiations.hh"

#include <batchlas/util/mempool.hh>

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
        throw std::logic_error(
            "sycl_orgqr: the apply-Q workspace query was not injected. orgqr's native arm "
            "is ormqr applied to an identity, and only the facade can name the ROUTED "
            "ormqr_buffer_size -- this driver is instantiated per scalar type with no "
            "Backend parameter (see the note at the top of orgqr_blocked.cc).");
    }
    const auto C = orgqr_c_view<T>(nullptr, m, n, batch);
    return q(ctx, A, C, Side::Left, Transpose::NoTrans, tau, orgqr_nb<T>(m, n));
}

}  // namespace

// True for all four types, but RouteTable<Op::orgqr,T>::preferred() is still
// false: only a vendor-free build or an explicit BATCHLAS_ORGQR_ROUTE lands
// here. evidence: docs/perf/qr.md#route-arms
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

    const std::size_t apply_bytes =
        orgqr_apply_bytes<T>(ctx, A, tau, m, n, batch, apply_q_buffer_size);

    return workspace_bytes([&](BumpAllocator& pool) {
        return orgqr_blocked_layout<T>(ctx, pool, m, n, batch, apply_bytes);
    });
}

template <typename T>
int orgqr_blocked_debug_block_size(Queue& ctx, int m, int n) {
    static_cast<void>(ctx);
    if (m < 1 || n < 1) return 0;
    return static_cast<int>(orgqr_nb<T>(m, n));
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

    // supports()'s gates are re-applied here: a forced route that is unsupported
    // falls through to automatic(), so a wrong gate silently measures the vendor.
    if (m < 1 || n < 1 || batch < 1) {
        throw std::invalid_argument("orgqr_blocked: degenerate extents");
    }
    if (m < n) {
        throw std::invalid_argument(
            "orgqr_blocked: n > m is not supported (route_orgqr.hh's supports() refuses it)");
    }
    if (A.is_heterogeneous()) {
        throw std::invalid_argument("orgqr_blocked: heterogeneous batch is not supported");
    }
    if (ctx.device().type != DeviceType::GPU) {
        throw std::invalid_argument("orgqr_blocked: GPU queues only");
    }
    if (tau.size() < static_cast<std::size_t>(k) * static_cast<std::size_t>(batch)) {
        throw std::invalid_argument("orgqr_blocked: tau span is shorter than k * batch");
    }
    if (!apply_q) {
        throw std::logic_error(
            "sycl_orgqr::orgqr_blocked_dispatch: the apply-Q seam was not injected. orgqr's "
            "native arm is ormqr applied to an identity, and only a layer that can name a "
            "Backend can reach the ROUTED ormqr (see the note at the top of "
            "orgqr_blocked.cc).");
    }

    const int32_t nb = orgqr_nb<T>(m, n);
    const std::size_t apply_bytes =
        orgqr_apply_bytes<T>(ctx, A, tau, m, n, batch, apply_q_buffer_size);

    BumpAllocator pool(workspace);
    auto ws = orgqr_blocked_layout<T>(ctx, pool, m, n, batch, apply_bytes);

    const auto C = orgqr_c_view<T>(ws.c.data(), m, n, batch);

    {
        T* const cp = ws.c.data();
        const std::size_t stride_c =
            static_cast<std::size_t>(m) * static_cast<std::size_t>(n);
        // C := I. Dim 2 is the ROW, not the column: sycl::id<3> makes dim 2 the
        // fastest-varying index and C is column-major with ld m, so indexing the
        // column there would stride m*sizeof(T).
        ctx->parallel_for<OrgqrIdentityKernel<T>>(
            sycl::range<3>(static_cast<std::size_t>(batch), static_cast<std::size_t>(n),
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
    (void)apply_q(ctx, A, C, Side::Left, Transpose::NoTrans, tau, ws.apply, nb);

    if (!ctx.in_order()) ctx.wait();

    // A := C -- orgqr overwrites its input, ormqr writes a separate C.
    {
        const T* const cp = ws.c.data();
        T* const ap = A.data_ptr();
        const int lda = A.ld();
        const int stride_a = A.stride();
        const std::size_t stride_c =
            static_cast<std::size_t>(m) * static_cast<std::size_t>(n);
        // Dim 2 is the ROW, as in the fill above.
        ctx->parallel_for<OrgqrCopyBackKernel<T>>(
            sycl::range<3>(static_cast<std::size_t>(batch), static_cast<std::size_t>(n),
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

    return ctx.get_event();
}

#define BATCHLAS_ORGQR_BLOCKED_INSTANTIATE(T)                                                 \
    template std::size_t orgqr_blocked_buffer_size<T>(                                        \
        Queue&, const MatrixView<T, MatrixFormat::Dense>&, Span<T>,                           \
        OrgqrApplyQBufferSize<T>);                                                            \
    template int orgqr_blocked_debug_block_size<T>(Queue&, int, int);                         \
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
