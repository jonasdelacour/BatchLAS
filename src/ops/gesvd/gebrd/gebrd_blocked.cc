// SPDX-License-Identifier: MIT
#include <batchlas/blas/extensions.hh>
#include <batchlas/blas/functions.hh>
#include <batchlas/backend_config.h>
#include <batchlas/util/mempool.hh>
#include <batchlas/device.hh>
#include <batchlas/device/lapack/gebrd.hh>

#include <src/math-helpers.hh>
#include <src/queue.hh>
#include <src/util/template-instantiations.hh>
#include <src/select/vendor.hh>
#include <src/util/batch_mask.hh>

#include <algorithm>
#include <stdexcept>

namespace batchlas {

namespace {

inline int32_t gebrd_blocked_resolved_nb(int32_t block_size) {
    return std::max<int32_t>(1, block_size);
}

template <typename T>
struct GebrdBlockedWorkspace {
    MatrixView<T, MatrixFormat::Dense> x;
    MatrixView<T, MatrixFormat::Dense> y;
};

// Single description of gebrd_blocked's workspace; see workspace_bytes() in
// util/mempool.hh.
template <typename T>
GebrdBlockedWorkspace<T> gebrd_blocked_layout(Queue& ctx,
                                              BumpAllocator& pool,
                                              int32_t m,
                                              int32_t n,
                                              int32_t nb,
                                              int32_t batch) {
    auto x_buf = pool.allocate<T>(ctx, static_cast<size_t>(m) * static_cast<size_t>(nb) * static_cast<size_t>(batch));
    auto y_buf = pool.allocate<T>(ctx, static_cast<size_t>(n) * static_cast<size_t>(nb) * static_cast<size_t>(batch));
    return {
        MatrixView<T, MatrixFormat::Dense>(x_buf.data(), m, nb, m, static_cast<int64_t>(m) * static_cast<int64_t>(nb), batch),
        MatrixView<T, MatrixFormat::Dense>(y_buf.data(), n, nb, n, static_cast<int64_t>(n) * static_cast<int64_t>(nb), batch),
    };
}

template <typename T>
inline void validate_gebrd_dims(const MatrixView<T, MatrixFormat::Dense>& a,
                                const VectorView<typename base_type<T>::type>& d,
                                const VectorView<typename base_type<T>::type>& e,
                                const VectorView<T>& tauq,
                                const VectorView<T>& taup,
                                const char* where) {
    if (a.rows() < a.cols()) {
        throw batchlas::invalid_argument(std::string(where) + ": A must satisfy rows >= cols");
    }

    const int32_t k = static_cast<int32_t>(a.cols());
    const int32_t need_e = std::max<int32_t>(0, k - 1);
    if (d.size() != k || e.size() != need_e || tauq.size() != k || taup.size() != k) {
        throw batchlas::invalid_argument(std::string(where) + ": invalid d/e/tau sizes");
    }
    if (a.batch_size() != d.batch_size() || a.batch_size() != e.batch_size() ||
        a.batch_size() != tauq.batch_size() || a.batch_size() != taup.batch_size()) {
        throw batchlas::invalid_argument(std::string(where) + ": batch size mismatch");
    }
    if (a.batch_size() < 1) {
        throw batchlas::invalid_argument(std::string(where) + ": invalid batch size");
    }
}

template <typename T>
Event gebrd_restore_bidiag_upper(Queue& ctx,
                                 const MatrixView<T, MatrixFormat::Dense>& a,
                                 const VectorView<typename base_type<T>::type>& d,
                                 const VectorView<typename base_type<T>::type>& e,
                                 const std::uint8_t* mask) {
    auto& a_mut = const_cast<MatrixView<T, MatrixFormat::Dense>&>(a);
    const int32_t k = static_cast<int32_t>(d.size());
    const int32_t batch = static_cast<int32_t>(a.batch_size());

    ctx->submit([&](sycl::handler& h) {
        auto A = a_mut.kernel_view();
        auto D = d;
        auto E = e;

        h.parallel_for(
            sycl::range<2>(static_cast<size_t>(batch), static_cast<size_t>(std::max<int32_t>(1, k))),
            [=](sycl::id<2> idx) {
                const int32_t b = static_cast<int32_t>(idx[0]);
                const int32_t i = static_cast<int32_t>(idx[1]);
                if (i >= k || (mask != nullptr && mask[b] == 0)) return;

                A(i, i, b) = static_cast<T>(D(i, b));
                if (i < k - 1) {
                    A(i, i + 1, b) = static_cast<T>(E(i, b));
                }
            });
    });

    return ctx.get_event();
}


// The labrd panel's work-group: a type, so every Cyclic loop strides by a constant.
constexpr int32_t kPanelWG = 128;
using PanelEx = device::WorkGroup<kPanelWG>;

// device::labrd on panel columns j0 .. j0+ib-1 of item b of the host's batched views (A, X, Y
// rebased by b, d / e / tau through item(), the host view's address). A function, not the kernel
// lambda's body, so the lambda calls no view operator() (SemaSYCL entry quirk).
template <typename T, typename AV, typename RV, typename TV>
BATCHLAS_DEVICE_INLINE void labrd_item(const PanelEx& ex, int32_t b, const AV& A, const AV& X, const AV& Y, const RV& D, const RV& E,
                                       const TV& TAUQ, const TV& TAUP, const sycl::local_accessor<T, 1>& work, int32_t m, int32_t n,
                                       int32_t nb, int32_t j0, int32_t ib) {
    const auto Ab = device::mat(A.data() + static_cast<int64_t>(b) * A.stride(), m, n, A.ld());
    const auto Xb = device::mat(X.data() + static_cast<int64_t>(b) * X.stride(), m, nb, X.ld());
    const auto Yb = device::mat(Y.data() + static_cast<int64_t>(b) * Y.stride(), n, nb, Y.ld());
    device::labrd<T>(ex, Ab, Xb, Yb, device::item(D, b), device::item(E, b), device::item(TAUQ, b), device::item(TAUP, b),
                     device::vec(work, nb, 0), device::vec(work, nb, nb), j0, ib);
}

// One panel launch (one work-group of kPanelWG per item), every T.
template <typename T>
void labrd_panel(Queue& ctx,
                 MatrixView<T, MatrixFormat::Dense>& a,
                 const VectorView<typename base_type<T>::type>& d_out,
                 const VectorView<typename base_type<T>::type>& e_out,
                 const VectorView<T>& tauq_out,
                 const VectorView<T>& taup_out,
                 const MatrixView<T, MatrixFormat::Dense>& x_mat,
                 const MatrixView<T, MatrixFormat::Dense>& y_mat,
                 const std::uint8_t* mask,
                 int32_t nb, int32_t j0, int32_t ib) {
    if (static_cast<int64_t>(ctx->get_device().get_info<sycl::info::device::max_work_group_size>()) < kPanelWG) {
        throw batchlas::unsupported("gebrd_blocked: the labrd panel kernel needs a work-group of 128");
    }
    const int32_t m = static_cast<int32_t>(a.rows());
    const int32_t n = static_cast<int32_t>(a.cols());
    const int32_t batch = static_cast<int32_t>(a.batch_size());
    ctx->submit([&](sycl::handler& h) {
        auto A = a.kernel_view();
        auto D = d_out;
        auto E = e_out;
        auto TAUQ = tauq_out;
        auto TAUP = taup_out;
        auto X = x_mat.kernel_view();
        auto Y = y_mat.kernel_view();
        auto work_local = sycl::local_accessor<T, 1>(sycl::range<1>(static_cast<size_t>(2 * nb)), h);

        batchlas::impl::parallel_for(h, sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(batch) * kPanelWG), sycl::range<1>(kPanelWG)),
                                     [=](sycl::nd_item<1> item) {
            const int32_t b = static_cast<int32_t>(item.get_group_linear_id());
            if (b >= batch || (mask != nullptr && mask[b] == 0)) return;  // I1: one work-group per item
            labrd_item<T>(PanelEx{item}, b, A, X, Y, D, E, TAUQ, TAUP, work_local, m, n, nb, j0, ib);
        });
    });
}

// LAPACK xGEBRD: a device::labrd panel (zlabrd's conventions: rows hold conj(u) on exit, unmbr 'P'
// order; d and e real; for real T every conj is the identity), then the trailing gemm.
template <Backend B, typename T>
Event gebrd_blocked_impl(Queue& ctx,
                         const MatrixView<T, MatrixFormat::Dense>& a_in,
                         const VectorView<typename base_type<T>::type>& d_out,
                         const VectorView<typename base_type<T>::type>& e_out,
                         const VectorView<T>& tauq_out,
                         const VectorView<T>& taup_out,
                         const Span<std::byte>& ws,
                         int32_t block_size,
                         Span<const std::uint8_t> mask_span) {
    validate_gebrd_dims(a_in, d_out, e_out, tauq_out, taup_out, "gebrd_blocked");

    if (!ctx.in_order()) {
        throw batchlas::invalid_argument("gebrd_blocked: requires an in-order Queue");
    }

    constexpr int32_t MaxPanelNB = 64;
    const int32_t nb = gebrd_blocked_resolved_nb(block_size);
    if (nb > MaxPanelNB) {
        throw batchlas::invalid_argument("gebrd_blocked: block_size > 64 is not supported");
    }

    auto& a = const_cast<MatrixView<T, MatrixFormat::Dense>&>(a_in);
    const int32_t m = static_cast<int32_t>(a.rows());
    const int32_t n = static_cast<int32_t>(a.cols());
    const int32_t k_total = std::min(m, n);
    const int32_t batch = static_cast<int32_t>(a.batch_size());

    Span<std::byte> ws_mut(const_cast<std::byte*>(ws.data()), ws.size());
    BumpAllocator pool(ws_mut);

    auto ws_layout = gebrd_blocked_layout<T>(ctx, pool, m, n, nb, batch);
    // Rebuilt from raw pointers: re-attach the call's mask, so the trailing gemm sees one span.
    MatrixView<T, MatrixFormat::Dense> x_mat = ws_layout.x.with_batch_mask(mask_span);
    MatrixView<T, MatrixFormat::Dense> y_mat = ws_layout.y.with_batch_mask(mask_span);
    const std::uint8_t* const mask = ::batchlas::detail::mask_ptr(mask_span);

    for (int32_t j0 = 0; j0 < k_total; j0 += nb) {
        const int32_t ib = std::min(nb, k_total - j0);
        auto x_panel = x_mat({j0, SliceEnd()}, {0, ib});
        auto y_panel = y_mat({j0, SliceEnd()}, {0, ib});
        // (void) on an Event: deliberate. This Queue is in-order, so the next submission
        // is already ordered after this one and the Event carries nothing the caller needs.
        (void)x_panel.fill_zeros(ctx);
        (void)y_panel.fill_zeros(ctx);

        labrd_panel<T>(ctx, a, d_out, e_out, tauq_out, taup_out, x_mat, y_mat, mask, nb, j0, ib);

        const int32_t j2 = j0 + ib;
        if (j2 < n) {
            auto a22 = a({j2, SliceEnd()}, {j2, SliceEnd()});
            auto v2 = a({j2, SliceEnd()}, {j0, j2});
            auto y2 = y_mat({j2, SliceEnd()}, {0, ib});
            auto x2 = x_mat({j2, SliceEnd()}, {0, ib});
            auto u2 = a({j0, j2}, {j2, SliceEnd()});

            (void)gemm<B>(ctx, v2, y2, a22, {.alpha = T(-1), .beta = T(1), .transB = Transpose::ConjTrans});
            (void)gemm<B>(ctx, x2, u2, a22, {.alpha = T(-1), .beta = T(1)});
        }
    }

    (void)gebrd_restore_bidiag_upper(ctx, a, d_out, e_out, mask);
    return ctx.get_event();
}

} // namespace

template <Backend B, typename T>
Event gebrd_blocked(Queue& ctx,
                    const MatrixView<T, MatrixFormat::Dense>& a_in,
                    const VectorView<typename base_type<T>::type>& d_out,
                    const VectorView<typename base_type<T>::type>& e_out,
                    const VectorView<T>& tauq_out,
                    const VectorView<T>& taup_out,
                    const Span<std::byte>& ws,
                    int32_t block_size) {
    // NETLIB's gemm is its host BLAS, which cannot skip an item: refuse before the panel writes A.
    if constexpr (!select::native_gpu_backend<B>)
        ::batchlas::detail::refuse_batch_mask("gebrd_blocked", a_in, d_out, e_out, tauq_out, taup_out);
    // A carries the call's mask whichever operand brought it, so the trailing gemm sees it.
    const auto mask = ::batchlas::detail::call_mask("gebrd_blocked", a_in, d_out, e_out, tauq_out, taup_out);
    auto a = mask.empty() || a_in.has_batch_mask() ? a_in : a_in.with_batch_mask(mask);
    return gebrd_blocked_impl<B, T>(ctx, a, d_out, e_out, tauq_out, taup_out, ws, block_size, mask);
}

template <Backend B, typename T>
size_t gebrd_blocked_buffer_size(Queue& ctx,
                                 const MatrixView<T, MatrixFormat::Dense>& a,
                                 const VectorView<typename base_type<T>::type>& d,
                                 const VectorView<typename base_type<T>::type>& e,
                                 const VectorView<T>& tauq,
                                 const VectorView<T>& taup,
                                 int32_t block_size) {
    validate_gebrd_dims(a, d, e, tauq, taup, "gebrd_blocked_buffer_size");

    const int32_t m = static_cast<int32_t>(a.rows());
    const int32_t n = static_cast<int32_t>(a.cols());
    const int32_t batch = static_cast<int32_t>(a.batch_size());
    const int32_t nb = gebrd_blocked_resolved_nb(block_size);

    return workspace_bytes([&](BumpAllocator& pool) {
        return gebrd_blocked_layout<T>(ctx, pool, m, n, nb, batch);
    });
}

#define GEBRD_BLOCKED_INSTANTIATE(back, fp) \
    template Event gebrd_blocked<back, BATCHLAS_UNPAREN fp>( \
        Queue&, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        const VectorView<typename base_type<BATCHLAS_UNPAREN fp>::type>&, \
        const VectorView<typename base_type<BATCHLAS_UNPAREN fp>::type>&, \
        const VectorView<BATCHLAS_UNPAREN fp>&, \
        const VectorView<BATCHLAS_UNPAREN fp>&, \
        const Span<std::byte>&, \
        int32_t); \
    template size_t gebrd_blocked_buffer_size<back, BATCHLAS_UNPAREN fp>( \
        Queue&, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        const VectorView<typename base_type<BATCHLAS_UNPAREN fp>::type>&, \
        const VectorView<typename base_type<BATCHLAS_UNPAREN fp>::type>&, \
        const VectorView<BATCHLAS_UNPAREN fp>&, \
        const VectorView<BATCHLAS_UNPAREN fp>&, \
        int32_t);

BATCHLAS_INSTANTIATE_SCALAR_ALL_BACKENDS(GEBRD_BLOCKED_INSTANTIATE)

#undef GEBRD_BLOCKED_INSTANTIATE

} // namespace batchlas
