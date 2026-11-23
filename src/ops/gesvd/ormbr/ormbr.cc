// SPDX-License-Identifier: MIT
#include <batchlas/blas/extensions.hh>
#include <batchlas/blas/linalg.hh>
#include <batchlas/internal/ormqr_blocked.hh>
#include <batchlas/backend_config.h>
#include <batchlas/util/mempool.hh>

#include <src/math-helpers.hh>
#include <src/queue.hh>
#include <src/select/vendor.hh>
#include <src/util/template-instantiations.hh>
#include <src/kernels/common/larft_wy.hh>
#include <batchlas/device.hh>
#include <batchlas/device/lapack/gesvd.hh>
#include <src/util/batch_mask.hh>

#include <algorithm>
#include <cctype>
#include <complex>
#include <cstdint>
#include <stdexcept>

namespace batchlas {

namespace {

inline char upper_ascii(char c) {
    return static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
}

template <typename T>
class OrmbrPKernel;

// Kernel-name tag for this TU's instantiations of the shared wy:: LARFT.
struct OrmbrPWyTag;

template <typename Scalar>
class PackPPanelKernel;

template <Backend B, typename Scalar>
bool ormbr_q_use_cta(const MatrixView<Scalar, MatrixFormat::Dense>& a,
                     const MatrixView<Scalar, MatrixFormat::Dense>& c) {
    if constexpr (!select::native_gpu_backend<B>) {
        static_cast<void>(a);
        static_cast<void>(c);
        return false;
    } else {
        return a.rows() == a.cols() && a.rows() <= 32 && c.rows() == c.cols() && c.rows() == a.rows();
    }
}

template <Backend B, typename Scalar>
bool ormbr_p_use_cta(const MatrixView<Scalar, MatrixFormat::Dense>& a,
                     const MatrixView<Scalar, MatrixFormat::Dense>& c,
                     Side side) {
    if constexpr (!select::native_gpu_backend<B>) {
        static_cast<void>(a);
        static_cast<void>(c);
        static_cast<void>(side);
        return false;
    } else {
        return side == Side::Right &&
               a.rows() == a.cols() &&
               a.rows() <= 32 &&
               c.rows() == c.cols() &&
               c.rows() == a.rows();
    }
}

template <Backend B, typename Scalar>
Event ormbr_apply_p_cta(Queue& ctx,
                        const MatrixView<Scalar, MatrixFormat::Dense>& a,
                        const VectorView<Scalar>& tau,
                        const MatrixView<Scalar, MatrixFormat::Dense>& c,
                        Side side,
                        Transpose trans,
                        const Span<std::byte>& ws) {
    const int32_t n = static_cast<int32_t>(a.cols());
    const int32_t batch = static_cast<int32_t>(a.batch_size());

    Span<std::byte> ws_mut(const_cast<std::byte*>(ws.data()), ws.size());
    BumpAllocator pool(ws_mut);

    auto aq_buf = pool.allocate<Scalar>(ctx,
                                        static_cast<size_t>(n) * static_cast<size_t>(n) * static_cast<size_t>(batch));
    auto tauq_buf = pool.allocate<Scalar>(ctx, static_cast<size_t>(n) * static_cast<size_t>(batch));

    // Rebuilt from raw pointers: re-attach the call's mask, which ormqx_cta then honours.
    const auto mask_span = ::batchlas::detail::call_mask("ormbr", a, tau, c);
    const std::uint8_t* const mask = ::batchlas::detail::mask_ptr(mask_span);
    const auto aq_view = MatrixView<Scalar, MatrixFormat::Dense>(aq_buf.data(), n, n, n, static_cast<int64_t>(n) * n,
                                                                 batch)
                             .with_batch_mask(mask_span);
    const auto tauq_view = VectorView<Scalar>(tauq_buf, n, batch, 1, n).with_batch_mask(mask_span);

    ctx->submit([&](sycl::handler& cgh) {
        auto A = a.kernel_view();
        auto AQ = aq_view.kernel_view();
        const int32_t nb = batch;
        const int32_t nn = n;
        const int64_t total = static_cast<int64_t>(nb) * static_cast<int64_t>(nn) * static_cast<int64_t>(nn);

        batchlas::impl::parallel_for(cgh, sycl::range<1>(static_cast<size_t>(total)), [=](sycl::id<1> tid) {
            const int64_t idx = static_cast<int64_t>(tid[0]);
            const int32_t b = static_cast<int32_t>(idx / (static_cast<int64_t>(nn) * nn));
            const int64_t rem = idx - static_cast<int64_t>(b) * nn * nn;
            const int32_t row = static_cast<int32_t>(rem % nn);
            const int32_t col = static_cast<int32_t>(rem / nn);
            if (mask != nullptr && mask[b] == 0) return;  // I3

            // AQ := 0, then AQ(1:, 1:) strictly lower := A(0:, 1:)^H: P's row reflectors as Q-form columns.
            const device::ElemAt<true> ex{row, col};
            const auto aq = device::item(AQ, b, nn, nn);
            device::laset(ex, device::Zero{}, device::Zero{}, aq);
            device::lacpy<device::Part::StrictLower, Transpose::ConjTrans>(
                ex.sub(1, 1), device::window(device::item(A, b, nn, nn), 0, 1, nn - 1, nn - 1),
                device::window(aq, 1, 1, nn - 1, nn - 1));
        });
    });

    ctx->submit([&](sycl::handler& cgh) {
        auto TAU = tau;
        auto TAUQ = tauq_view;
        const int32_t nb = batch;
        const int32_t nn = n;
        const int64_t total = static_cast<int64_t>(nb) * static_cast<int64_t>(nn);

        batchlas::impl::parallel_for(cgh, sycl::range<1>(static_cast<size_t>(total)), [=](sycl::id<1> tid) {
            const int64_t idx = static_cast<int64_t>(tid[0]);
            const int32_t b = static_cast<int32_t>(idx / nn);
            const int32_t j = static_cast<int32_t>(idx - static_cast<int64_t>(b) * nn);
            if (mask != nullptr && mask[b] == 0) return;  // I3
            const auto tauq = device::item(TAUQ, b, nn);
            device::copy(device::Grid1D<true>{j}, device::shifted(device::item(TAU, b, nn - 1), 1, nn), tauq);
        });
    });

    return ormqx_cta<B, Scalar>(ctx,
                                aq_view,
                                tauq_view,
                                c,
                                Uplo::Upper,
                                side,
                                trans,
                                n,
                                Span<std::byte>(),
                                1);
}

template <typename Scalar>
size_t ormbr_p_cta_buffer_size(Queue& ctx,
                               const MatrixView<Scalar, MatrixFormat::Dense>& a) {
    const size_t n = static_cast<size_t>(a.rows());
    const size_t batch = static_cast<size_t>(a.batch_size());
    size_t size = 0;
    size += BumpAllocator::allocation_size<Scalar>(ctx, n * n * batch);
    size += BumpAllocator::allocation_size<Scalar>(ctx, n * batch);
    return size;
}

template <typename Scalar>
sycl::event pack_p_panel_batched(Queue& q,
                                 Scalar* v_out,
                                 int ld_v_out,
                                 int stride_v_out,
                                 const MatrixView<Scalar, MatrixFormat::Dense>& a,
                                 int i0,
                                 int ib,
                                 int nq,
                                 const std::uint8_t* mask) {
    const int m = std::max(0, nq - i0 - 1);
    const int ld_a = a.ld();
    const int stride_a = a.stride();
    const Scalar* a_ptr = a.data_ptr();
    const int batch = a.batch_size();

    return q->submit([&](sycl::handler& h) {
        batchlas::impl::parallel_for<PackPPanelKernel<Scalar>>(h, sycl::range<3>(static_cast<size_t>(batch),
                                                           static_cast<size_t>(m),
                                                           static_cast<size_t>(ib)),
                                            [=](sycl::id<3> idx) {
                                                const int b = static_cast<int>(idx[0]);
                                                const int r = static_cast<int>(idx[1]);
                                                const int c = static_cast<int>(idx[2]);
                                                if (mask != nullptr && mask[b] == 0) return;  // I3
                                                // V := unit lower part of A(i0:, i0+1:)^H (row reflectors as columns).
                                                const device::ElemAt<true> ex{r, c};
                                                const auto ap = device::mat(a_ptr + static_cast<std::ptrdiff_t>(b) * stride_a,
                                                                        i0 + ib, i0 + 1 + m, ld_a);
                                                const auto v = device::mat(v_out + static_cast<std::ptrdiff_t>(b) * stride_v_out,
                                                                       m, ib, ld_v_out);
                                                // One tag per laset call: see set_identity in gesvd_blocked.cc (complex<double> stack).
                                                if (r > c) device::lacpy<device::Part::StrictLower, Transpose::ConjTrans>(ex, ap.sub(i0, i0 + 1, ib, m), v);
                                                else if (r == c) device::laset(ex, device::One{}, device::One{}, v);
                                                else device::laset(ex, device::Zero{}, device::Zero{}, v);
                                            });
    });
}

template <int NB>
inline int ormbr_resolved_nb(int32_t block_size) {
    if constexpr (NB > 0) {
        return NB;
    }
    return std::max<int>(1, block_size);
}

template <Backend B, typename Scalar, int NB>
size_t ormbr_p_blocked_buffer_size_impl(Queue& ctx,
                                        const MatrixView<Scalar, MatrixFormat::Dense>& a,
                                        const MatrixView<Scalar, MatrixFormat::Dense>& c,
                                        Side side,
                                        int32_t block_size) {
    const int nq = std::max<int>(0, static_cast<int>(a.cols()) - 1);
    const int k = std::max<int>(0, std::min<int>(a.rows(), a.cols()) - 1);
    if (k == 0 || nq == 0) {
        return 0;
    }

    const int m = c.rows();
    const int n = c.cols();
    const int batch = a.batch_size();
    const int nb = std::min(ormbr_resolved_nb<NB>(block_size), k);

    size_t size = 0;
    size += BumpAllocator::allocation_size<Scalar>(ctx, static_cast<size_t>(nq) * static_cast<size_t>(nb) * static_cast<size_t>(batch));
    size += BumpAllocator::allocation_size<Scalar>(ctx, static_cast<size_t>(nb) * static_cast<size_t>(nb) * static_cast<size_t>(batch));

    const size_t w_elems = (side == Side::Left)
                               ? static_cast<size_t>(nb) * static_cast<size_t>(n)
                               : static_cast<size_t>(m) * static_cast<size_t>(nb);
    size += 2 * BumpAllocator::allocation_size<Scalar>(ctx, w_elems * static_cast<size_t>(batch));

    return size;
}

template <Backend B, typename Scalar, int NB>
Event ormbr_apply_p_blocked_impl(Queue& ctx,
                                 const MatrixView<Scalar, MatrixFormat::Dense>& a,
                                 const VectorView<Scalar>& tau,
                                 const MatrixView<Scalar, MatrixFormat::Dense>& c,
                                 Side side,
                                 Transpose trans,
                                 const Span<std::byte>& ws,
                                 int32_t block_size) {
    const int nq = static_cast<int>(a.cols());
    const int k = std::max<int>(0, std::min<int>(a.rows(), a.cols()) - 1);
    if (k == 0 || nq <= 1) {
        return ctx.get_event();
    }

    const int mC = c.rows();
    const int nC = c.cols();
    const int batch = a.batch_size();
    const int nb = std::min(ormbr_resolved_nb<NB>(block_size), k);
    const bool transpose_apply = (trans != Transpose::NoTrans);

    Span<std::byte> ws_mut(const_cast<std::byte*>(ws.data()), ws.size());
    BumpAllocator pool(ws_mut);
    auto Vbuf = pool.allocate<Scalar>(ctx, static_cast<size_t>(nq - 1) * static_cast<size_t>(nb) * static_cast<size_t>(batch));
    auto Tbuf = pool.allocate<Scalar>(ctx,
                                      static_cast<size_t>(nb) * static_cast<size_t>(nb) * static_cast<size_t>(batch));

    const size_t w_elems = (side == Side::Left)
                               ? static_cast<size_t>(nb) * static_cast<size_t>(nC)
                               : static_cast<size_t>(mC) * static_cast<size_t>(nb);
    auto W1buf = pool.allocate<Scalar>(ctx, w_elems * static_cast<size_t>(batch));
    auto W2buf = pool.allocate<Scalar>(ctx, w_elems * static_cast<size_t>(batch));

    // Workspace views are rebuilt from raw pointers: each re-attaches the call's mask for gemm.
    const auto mask_span = ::batchlas::detail::call_mask("ormbr", a, tau, c);
    const std::uint8_t* const mask = ::batchlas::detail::mask_ptr(mask_span);
    const auto Vmat = MatrixView<Scalar, MatrixFormat::Dense>(Vbuf.data(), nq - 1, nb, nq - 1,
                                                              static_cast<int64_t>(nq - 1) * nb, batch).with_batch_mask(mask_span);
    const auto Tmat = MatrixView<Scalar, MatrixFormat::Dense>(Tbuf.data(), nb, nb, nb, static_cast<int64_t>(nb) * nb,
                                                              batch)
                          .with_batch_mask(mask_span);

    auto apply_block = [&](int i0) {
        const int ib = std::min(nb, k - i0);
        const int m = nq - i0 - 1;
        if (ib <= 0 || m <= 0) {
            return;
        }

        (void)pack_p_panel_batched<Scalar>(ctx,
                                           Vmat.data_ptr(), Vmat.ld(), Vmat.stride(),
                                           a, i0, ib, nq, mask);

        (void)wy::larft_forward_columnwise_batched<OrmbrPWyTag, Scalar>(ctx,
                                                                              Tmat.data_ptr(), Tmat.ld(), Tmat.stride(),
                                                                              Vmat.data_ptr(), Vmat.ld(), Vmat.stride(),
                                                                              m, ib,
                                                                              tau.data_ptr(), tau.stride(), i0,
                                                                              batch, mask);

        auto Ublk = Vmat({0, m}, {0, ib});
        auto Tblk = Tmat({0, ib}, {0, ib});
        const Transpose t_eff = transpose_apply ? Transpose::ConjTrans : Transpose::NoTrans;

        if (side == Side::Left) {
            auto Csub = c({i0 + 1, SliceEnd()}, Slice());
            const auto W1full = MatrixView<Scalar, MatrixFormat::Dense>(W1buf.data(), nb, nC, nb, static_cast<int64_t>(nb) * nC, batch)
                                    .with_batch_mask(mask_span);
            const auto W2full = MatrixView<Scalar, MatrixFormat::Dense>(W2buf.data(), nb, nC, nb, static_cast<int64_t>(nb) * nC, batch)
                                    .with_batch_mask(mask_span);
            auto W1 = W1full({0, ib}, Slice());
            auto W2 = W2full({0, ib}, Slice());

            // (void) on an Event: deliberate. This Queue is in-order, so the next submission
            // is already ordered after this one and the Event carries nothing the caller needs.
            (void)gemm<B>(ctx, Ublk, Csub, W1, {.transA = Transpose::ConjTrans});
            (void)gemm<B>(ctx, Tblk, W1, W2, {.transA = t_eff});
            (void)gemm<B>(ctx, Ublk, W2, Csub, {.alpha = Scalar(-1), .beta = Scalar(1)});
        } else {
            auto Csub = c(Slice(), {i0 + 1, SliceEnd()});
            const auto W1full = MatrixView<Scalar, MatrixFormat::Dense>(W1buf.data(), mC, nb, mC, static_cast<int64_t>(mC) * nb, batch)
                                    .with_batch_mask(mask_span);
            const auto W2full = MatrixView<Scalar, MatrixFormat::Dense>(W2buf.data(), mC, nb, mC, static_cast<int64_t>(mC) * nb, batch)
                                    .with_batch_mask(mask_span);
            auto W1 = W1full(Slice(), {0, ib});
            auto W2 = W2full(Slice(), {0, ib});

            (void)gemm<B>(ctx, Csub, Ublk, W1, GemmOptions<Scalar>{});
            (void)gemm<B>(ctx, W1, Tblk, W2, {.transB = t_eff});
            (void)gemm<B>(ctx,
                    W2,
                    Ublk,
                    Csub,
                    {.alpha = Scalar(-1), .beta = Scalar(1), .transB = Transpose::ConjTrans});
        }
    };

    if (!transpose_apply) {
        for (int i0 = 0; i0 < k; i0 += nb) {
            apply_block(i0);
        }
    } else {
        for (int i0 = ((k - 1) / nb) * nb; i0 >= 0; i0 -= nb) {
            apply_block(i0);
        }
    }

    return ctx.get_event();
}

template <typename Scalar>
inline void validate_ormbr_dims(const MatrixView<Scalar, MatrixFormat::Dense>& a,
                                const VectorView<Scalar>& tau,
                                const MatrixView<Scalar, MatrixFormat::Dense>& c,
                                char vect,
                                Side side) {
    if (a.batch_size() != c.batch_size() || tau.batch_size() != a.batch_size()) {
        throw batchlas::invalid_argument("ormbr: batch size mismatch");
    }
    if (a.batch_size() < 1) {
        throw batchlas::invalid_argument("ormbr: invalid batch size");
    }

    const char v = upper_ascii(vect);
    if (v != 'Q' && v != 'P') {
        throw batchlas::invalid_argument("ormbr: vect must be 'Q' or 'P'");
    }

    const int32_t k = std::min<int32_t>(a.rows(), a.cols());
    const int32_t order = (v == 'Q') ? static_cast<int32_t>(a.rows()) : static_cast<int32_t>(a.cols());
    const int32_t nq = (side == Side::Left) ? static_cast<int32_t>(c.rows()) : static_cast<int32_t>(c.cols());
    if (nq != order) {
        throw batchlas::invalid_argument("ormbr: C dimension does not match reflector order");
    }

    const int32_t need_tau = (v == 'Q') ? k : std::max<int32_t>(0, k - 1);
    if (tau.inc() != 1) {
        throw batchlas::invalid_argument("ormbr: tau must be unit-stride");
    }
    if (tau.stride() != tau.size()) {
        throw batchlas::invalid_argument("ormbr: tau must be tightly packed by batch");
    }
    if (tau.size() < static_cast<size_t>(need_tau)) {
        throw batchlas::invalid_argument("ormbr: tau span too small");
    }
}

}  // namespace

template <Backend B, typename Scalar>
Event ormbr(Queue& ctx,
            const MatrixView<Scalar, MatrixFormat::Dense>& a_in,
            const VectorView<Scalar>& tau_in,
            const MatrixView<Scalar, MatrixFormat::Dense>& c_in,
            char vect,
            Side side,
            Transpose trans,
            const Span<std::byte>& ws,
            int32_t block_size) {
    // Beside a host BLAS NETLIB's gemm cannot skip an item: refuse before any child runs. A
    // vendor-free host's children (ormqr_blocked, the P apply's gemm) are natives that honour it.
    if constexpr (!select::native_backend<B>) ::batchlas::detail::refuse_batch_mask("ormbr", a_in, tau_in, c_in);
    // Every operand carries the call's mask whichever brought it, so the children see it. 'Q' runs
    // ormqx_cta or ormqr_blocked.
    const auto mask = ::batchlas::detail::call_mask("ormbr", a_in, tau_in, c_in);
    const bool adopt = !mask.empty() && a_in.batch_size() == c_in.batch_size() && tau_in.batch_size() == c_in.batch_size();
    const auto a = adopt ? a_in.with_batch_mask(mask) : a_in;
    const auto tau = adopt ? tau_in.with_batch_mask(mask) : tau_in;
    const auto c = adopt ? c_in.with_batch_mask(mask) : c_in;
    validate_ormbr_dims(a, tau, c, vect, side);

    const char v = upper_ascii(vect);
    if (v == 'Q') {
        const size_t tau_elems = tau.data().size();
        Span<Scalar> tau_span(const_cast<Scalar*>(tau.data_ptr()), tau_elems);
        if (ormbr_q_use_cta<B, Scalar>(a, c)) {
            return ormqx_cta<B, Scalar>(ctx,
                                        a,
                                        tau,
                                        c,
                                        Uplo::Upper,
                                        side,
                                        trans,
                                        static_cast<int32_t>(a.rows()),
                                        ws);
        }
        Span<std::byte> ws_mut(const_cast<std::byte*>(ws.data()), ws.size());
        return ormqr_blocked<B, Scalar>(ctx, a, c, side, trans, tau_span, ws_mut, block_size);
    }

    if constexpr (detail::is_complex<Scalar>::value) {
        if (trans == Transpose::Trans) {
            throw batchlas::unsupported("ormbr: Trans not supported for complex 'P'; use ConjTrans");
        }
    }

    if (ormbr_p_use_cta<B, Scalar>(a, c, side)) {
        return ormbr_apply_p_cta<B, Scalar>(ctx, a, tau, c, side, trans, ws);
    }

    const int nb = std::max<int>(1, block_size);
    switch (nb) {
        case 16:
            return ormbr_apply_p_blocked_impl<B, Scalar, 16>(ctx, a, tau, c, side, trans, ws, block_size);
        case 32:
            return ormbr_apply_p_blocked_impl<B, Scalar, 32>(ctx, a, tau, c, side, trans, ws, block_size);
        case 64:
            return ormbr_apply_p_blocked_impl<B, Scalar, 64>(ctx, a, tau, c, side, trans, ws, block_size);
        case 128:
            return ormbr_apply_p_blocked_impl<B, Scalar, 128>(ctx, a, tau, c, side, trans, ws, block_size);
        default:
            return ormbr_apply_p_blocked_impl<B, Scalar, -1>(ctx, a, tau, c, side, trans, ws, block_size);
    }
}

template <Backend B, typename Scalar>
size_t ormbr_buffer_size(Queue& ctx,
                         const MatrixView<Scalar, MatrixFormat::Dense>& a,
                         const VectorView<Scalar>& tau,
                         const MatrixView<Scalar, MatrixFormat::Dense>& c,
                         char vect,
                         Side side,
                         Transpose trans,
                         int32_t block_size) {
    validate_ormbr_dims(a, tau, c, vect, side);

    const char v = upper_ascii(vect);
    if (v == 'Q') {
        if (ormbr_q_use_cta<B, Scalar>(a, c)) {
            return 0;
        }
        const size_t tau_elems = tau.data().size();
        Span<Scalar> tau_span(const_cast<Scalar*>(tau.data_ptr()), tau_elems);
        return ormqr_blocked_buffer_size<B, Scalar>(ctx, a, c, side, trans, tau_span, block_size);
    }

    if constexpr (detail::is_complex<Scalar>::value) {
        if (trans == Transpose::Trans) {
            throw batchlas::unsupported("ormbr_buffer_size: Trans not supported for complex 'P'; use ConjTrans");
        }
    }

    if (ormbr_p_use_cta<B, Scalar>(a, c, side)) {
        return ormbr_p_cta_buffer_size<Scalar>(ctx, a);
    }

    const int nb = std::max<int>(1, block_size);
    switch (nb) {
        case 16:
            return ormbr_p_blocked_buffer_size_impl<B, Scalar, 16>(ctx, a, c, side, block_size);
        case 32:
            return ormbr_p_blocked_buffer_size_impl<B, Scalar, 32>(ctx, a, c, side, block_size);
        case 64:
            return ormbr_p_blocked_buffer_size_impl<B, Scalar, 64>(ctx, a, c, side, block_size);
        case 128:
            return ormbr_p_blocked_buffer_size_impl<B, Scalar, 128>(ctx, a, c, side, block_size);
        default:
            return ormbr_p_blocked_buffer_size_impl<B, Scalar, -1>(ctx, a, c, side, block_size);
    }
}

#define ORMBR_INSTANTIATE(back, fp) \
    template Event ormbr<back, BATCHLAS_UNPAREN fp>( \
        Queue&, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        const VectorView<BATCHLAS_UNPAREN fp>&, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        char, \
        Side, \
        Transpose, \
        const Span<std::byte>&, \
        int32_t); \
    template size_t ormbr_buffer_size<back, BATCHLAS_UNPAREN fp>( \
        Queue&, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        const VectorView<BATCHLAS_UNPAREN fp>&, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        char, \
        Side, \
        Transpose, \
        int32_t);

BATCHLAS_INSTANTIATE_SCALAR_ALL_BACKENDS(ORMBR_INSTANTIATE)

#undef ORMBR_INSTANTIATE

#undef ORMBR_INSTANTIATE

}  // namespace batchlas
