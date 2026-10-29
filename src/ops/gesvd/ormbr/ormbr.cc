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

template <typename T>
class PackPPanelKernel;

template <Backend B, typename T>
bool ormbr_q_use_cta(const MatrixView<T, MatrixFormat::Dense>& a,
                     const MatrixView<T, MatrixFormat::Dense>& c) {
    if constexpr (!select::native_gpu_backend<B>) {
        static_cast<void>(a);
        static_cast<void>(c);
        return false;
    } else {
        return a.rows() == a.cols() && a.rows() <= 32 && c.rows() == c.cols() && c.rows() == a.rows();
    }
}

template <Backend B, typename T>
bool ormbr_p_use_cta(const MatrixView<T, MatrixFormat::Dense>& a,
                     const MatrixView<T, MatrixFormat::Dense>& c,
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

template <Backend B, typename T>
Event ormbr_apply_p_cta(Queue& ctx,
                        const MatrixView<T, MatrixFormat::Dense>& a,
                        const VectorView<T>& tau,
                        const MatrixView<T, MatrixFormat::Dense>& c,
                        Side side,
                        Transpose trans,
                        const Span<std::byte>& ws) {
    const int32_t n = static_cast<int32_t>(a.cols());
    const int32_t batch = static_cast<int32_t>(a.batch_size());

    Span<std::byte> ws_mut(const_cast<std::byte*>(ws.data()), ws.size());
    BumpAllocator pool(ws_mut);

    auto aq_buf = pool.allocate<T>(ctx, static_cast<size_t>(n) * static_cast<size_t>(n) * static_cast<size_t>(batch));
    auto tauq_buf = pool.allocate<T>(ctx, static_cast<size_t>(n) * static_cast<size_t>(batch));

    // Rebuilt from raw pointers: re-attach the call's mask, which ormqx_cta then honours.
    const auto mask_span = ::batchlas::detail::call_mask("ormbr", a, tau, c);
    const std::uint8_t* const mask = ::batchlas::detail::mask_ptr(mask_span);
    const auto aq_view = MatrixView<T, MatrixFormat::Dense>(aq_buf.data(), n, n, n, static_cast<int64_t>(n) * n, batch)
                             .with_batch_mask(mask_span);
    const auto tauq_view = VectorView<T>(tauq_buf, n, batch, 1, n).with_batch_mask(mask_span);

    ctx->submit([&](sycl::handler& cgh) {
        auto A = a.kernel_view();
        auto AQ = aq_view.kernel_view();
        const int32_t nb = batch;
        const int32_t nn = n;
        const int64_t total = static_cast<int64_t>(nb) * static_cast<int64_t>(nn) * static_cast<int64_t>(nn);

        cgh.parallel_for(sycl::range<1>(static_cast<size_t>(total)), [=](sycl::id<1> tid) {
            const int64_t idx = static_cast<int64_t>(tid[0]);
            const int32_t b = static_cast<int32_t>(idx / (static_cast<int64_t>(nn) * nn));
            const int64_t rem = idx - static_cast<int64_t>(b) * nn * nn;
            const int32_t row = static_cast<int32_t>(rem % nn);
            const int32_t col = static_cast<int32_t>(rem / nn);
            if (mask != nullptr && mask[b] == 0) return;  // I3

            AQ(row, col, b) = T(0);
            if (col >= 1) {
                const int32_t i = col - 1;
                if (row >= col + 1 && row < nn) {
                    AQ(row, col, b) = batchlas::device::detail::conj(A(i, row, b));
                }
            }
        });
    });

    ctx->submit([&](sycl::handler& cgh) {
        auto TAU = tau;
        auto TAUQ = tauq_view;
        const int32_t nb = batch;
        const int32_t nn = n;
        const int64_t total = static_cast<int64_t>(nb) * static_cast<int64_t>(nn);

        cgh.parallel_for(sycl::range<1>(static_cast<size_t>(total)), [=](sycl::id<1> tid) {
            const int64_t idx = static_cast<int64_t>(tid[0]);
            const int32_t b = static_cast<int32_t>(idx / nn);
            const int32_t j = static_cast<int32_t>(idx - static_cast<int64_t>(b) * nn);
            if (mask != nullptr && mask[b] == 0) return;  // I3

            if (j == 0) {
                TAUQ(j, b) = T(0);
            } else {
                TAUQ(j, b) = (j - 1 < (nn - 1)) ? TAU(j - 1, b) : T(0);
            }
        });
    });

    return ormqx_cta<B, T>(ctx,
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

template <typename T>
size_t ormbr_p_cta_buffer_size(Queue& ctx,
                               const MatrixView<T, MatrixFormat::Dense>& a) {
    const size_t n = static_cast<size_t>(a.rows());
    const size_t batch = static_cast<size_t>(a.batch_size());
    size_t size = 0;
    size += BumpAllocator::allocation_size<T>(ctx, n * n * batch);
    size += BumpAllocator::allocation_size<T>(ctx, n * batch);
    return size;
}

template <typename T>
sycl::event pack_p_panel_batched(Queue& q,
                                 T* v_out,
                                 int ld_v_out,
                                 int stride_v_out,
                                 const MatrixView<T, MatrixFormat::Dense>& a,
                                 int i0,
                                 int ib,
                                 int nq,
                                 const std::uint8_t* mask) {
    const int m = std::max(0, nq - i0 - 1);
    const int ld_a = a.ld();
    const int stride_a = a.stride();
    const T* a_ptr = a.data_ptr();
    const int batch = a.batch_size();

    return q->submit([&](sycl::handler& h) {
        h.parallel_for<PackPPanelKernel<T>>(sycl::range<3>(static_cast<size_t>(batch),
                                                           static_cast<size_t>(m),
                                                           static_cast<size_t>(ib)),
                                            [=](sycl::id<3> idx) {
                                                const int b = static_cast<int>(idx[0]);
                                                const int r = static_cast<int>(idx[1]);
                                                const int c = static_cast<int>(idx[2]);
                                                if (mask != nullptr && mask[b] == 0) return;  // I3
                                                T val = T(0);
                                                if (r == c) {
                                                    val = T(1);
                                                } else if (r > c) {
                                                    const int row = i0 + c;
                                                    const int col = i0 + 1 + r;
                                                    val = batchlas::device::detail::conj(a_ptr[static_cast<std::ptrdiff_t>(b) * stride_a + row + col * ld_a]);
                                                }
                                                v_out[static_cast<std::ptrdiff_t>(b) * stride_v_out + r + c * ld_v_out] = val;
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

template <Backend B, typename T, int NB>
size_t ormbr_p_blocked_buffer_size_impl(Queue& ctx,
                                        const MatrixView<T, MatrixFormat::Dense>& a,
                                        const MatrixView<T, MatrixFormat::Dense>& c,
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
    size += BumpAllocator::allocation_size<T>(ctx, static_cast<size_t>(nq) * static_cast<size_t>(nb) * static_cast<size_t>(batch));
    size += BumpAllocator::allocation_size<T>(ctx, static_cast<size_t>(nb) * static_cast<size_t>(nb) * static_cast<size_t>(batch));

    const size_t w_elems = (side == Side::Left)
                               ? static_cast<size_t>(nb) * static_cast<size_t>(n)
                               : static_cast<size_t>(m) * static_cast<size_t>(nb);
    size += 2 * BumpAllocator::allocation_size<T>(ctx, w_elems * static_cast<size_t>(batch));

    return size;
}

template <Backend B, typename T, int NB>
Event ormbr_apply_p_blocked_impl(Queue& ctx,
                                 const MatrixView<T, MatrixFormat::Dense>& a,
                                 const VectorView<T>& tau,
                                 const MatrixView<T, MatrixFormat::Dense>& c,
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
    auto Vbuf = pool.allocate<T>(ctx, static_cast<size_t>(nq - 1) * static_cast<size_t>(nb) * static_cast<size_t>(batch));
    auto Tbuf = pool.allocate<T>(ctx, static_cast<size_t>(nb) * static_cast<size_t>(nb) * static_cast<size_t>(batch));

    const size_t w_elems = (side == Side::Left)
                               ? static_cast<size_t>(nb) * static_cast<size_t>(nC)
                               : static_cast<size_t>(mC) * static_cast<size_t>(nb);
    auto W1buf = pool.allocate<T>(ctx, w_elems * static_cast<size_t>(batch));
    auto W2buf = pool.allocate<T>(ctx, w_elems * static_cast<size_t>(batch));

    // Workspace views are rebuilt from raw pointers: each re-attaches the call's mask for gemm.
    const auto mask_span = ::batchlas::detail::call_mask("ormbr", a, tau, c);
    const std::uint8_t* const mask = ::batchlas::detail::mask_ptr(mask_span);
    const auto Vmat = MatrixView<T, MatrixFormat::Dense>(Vbuf.data(), nq - 1, nb, nq - 1,
                                                         static_cast<int64_t>(nq - 1) * nb, batch).with_batch_mask(mask_span);
    const auto Tmat = MatrixView<T, MatrixFormat::Dense>(Tbuf.data(), nb, nb, nb, static_cast<int64_t>(nb) * nb, batch)
                          .with_batch_mask(mask_span);

    auto apply_block = [&](int i0) {
        const int ib = std::min(nb, k - i0);
        const int m = nq - i0 - 1;
        if (ib <= 0 || m <= 0) {
            return;
        }

        (void)pack_p_panel_batched<T>(ctx,
                                      Vmat.data_ptr(), Vmat.ld(), Vmat.stride(),
                                      a, i0, ib, nq, mask);

        (void)wy::larft_forward_columnwise_batched_t<OrmbrPWyTag, T, false>(ctx,
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
            const auto W1full = MatrixView<T, MatrixFormat::Dense>(W1buf.data(), nb, nC, nb, static_cast<int64_t>(nb) * nC, batch)
                                    .with_batch_mask(mask_span);
            const auto W2full = MatrixView<T, MatrixFormat::Dense>(W2buf.data(), nb, nC, nb, static_cast<int64_t>(nb) * nC, batch)
                                    .with_batch_mask(mask_span);
            auto W1 = W1full({0, ib}, Slice());
            auto W2 = W2full({0, ib}, Slice());

            // (void) on an Event: deliberate. This Queue is in-order, so the next submission
            // is already ordered after this one and the Event carries nothing the caller needs.
            (void)gemm<B>(ctx, Ublk, Csub, W1, {.transA = Transpose::ConjTrans});
            (void)gemm<B>(ctx, Tblk, W1, W2, {.transA = t_eff});
            (void)gemm<B>(ctx, Ublk, W2, Csub, {.alpha = T(-1), .beta = T(1)});
        } else {
            auto Csub = c(Slice(), {i0 + 1, SliceEnd()});
            const auto W1full = MatrixView<T, MatrixFormat::Dense>(W1buf.data(), mC, nb, mC, static_cast<int64_t>(mC) * nb, batch)
                                    .with_batch_mask(mask_span);
            const auto W2full = MatrixView<T, MatrixFormat::Dense>(W2buf.data(), mC, nb, mC, static_cast<int64_t>(mC) * nb, batch)
                                    .with_batch_mask(mask_span);
            auto W1 = W1full(Slice(), {0, ib});
            auto W2 = W2full(Slice(), {0, ib});

            (void)gemm<B>(ctx, Csub, Ublk, W1, GemmOptions<T>{});
            (void)gemm<B>(ctx, W1, Tblk, W2, {.transB = t_eff});
            (void)gemm<B>(ctx,
                    W2,
                    Ublk,
                    Csub,
                    {.alpha = T(-1), .beta = T(1), .transB = Transpose::ConjTrans});
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

template <typename T>
inline void validate_ormbr_dims(const MatrixView<T, MatrixFormat::Dense>& a,
                                const VectorView<T>& tau,
                                const MatrixView<T, MatrixFormat::Dense>& c,
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

} // namespace

template <Backend B, typename T>
Event ormbr(Queue& ctx,
            const MatrixView<T, MatrixFormat::Dense>& a_in,
            const VectorView<T>& tau_in,
            const MatrixView<T, MatrixFormat::Dense>& c_in,
            char vect,
            Side side,
            Transpose trans,
            const Span<std::byte>& ws,
            int32_t block_size) {
    // NETLIB's gemm is its host BLAS, which cannot skip an item: refuse before any child runs.
    if constexpr (!select::native_gpu_backend<B>) ::batchlas::detail::refuse_batch_mask("ormbr", a_in, tau_in, c_in);
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
        Span<T> tau_span(const_cast<T*>(tau.data_ptr()), tau_elems);
        if (ormbr_q_use_cta<B, T>(a, c)) {
            return ormqx_cta<B, T>(ctx,
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
        return ormqr_blocked<B, T>(ctx, a, c, side, trans, tau_span, ws_mut, block_size);
    }

    if constexpr (detail::is_complex<T>::value) {
        if (trans == Transpose::Trans) {
            throw batchlas::unsupported("ormbr: Trans not supported for complex 'P'; use ConjTrans");
        }
    }

    if (ormbr_p_use_cta<B, T>(a, c, side)) {
        return ormbr_apply_p_cta<B, T>(ctx, a, tau, c, side, trans, ws);
    }

    const int nb = std::max<int>(1, block_size);
    switch (nb) {
        case 16:
            return ormbr_apply_p_blocked_impl<B, T, 16>(ctx, a, tau, c, side, trans, ws, block_size);
        case 32:
            return ormbr_apply_p_blocked_impl<B, T, 32>(ctx, a, tau, c, side, trans, ws, block_size);
        case 64:
            return ormbr_apply_p_blocked_impl<B, T, 64>(ctx, a, tau, c, side, trans, ws, block_size);
        case 128:
            return ormbr_apply_p_blocked_impl<B, T, 128>(ctx, a, tau, c, side, trans, ws, block_size);
        default:
            return ormbr_apply_p_blocked_impl<B, T, -1>(ctx, a, tau, c, side, trans, ws, block_size);
    }
}

template <Backend B, typename T>
size_t ormbr_buffer_size(Queue& ctx,
                         const MatrixView<T, MatrixFormat::Dense>& a,
                         const VectorView<T>& tau,
                         const MatrixView<T, MatrixFormat::Dense>& c,
                         char vect,
                         Side side,
                         Transpose trans,
                         int32_t block_size) {
    validate_ormbr_dims(a, tau, c, vect, side);

    const char v = upper_ascii(vect);
    if (v == 'Q') {
        if (ormbr_q_use_cta<B, T>(a, c)) {
            return 0;
        }
        const size_t tau_elems = tau.data().size();
        Span<T> tau_span(const_cast<T*>(tau.data_ptr()), tau_elems);
        return ormqr_blocked_buffer_size<B, T>(ctx, a, c, side, trans, tau_span, block_size);
    }

    if constexpr (detail::is_complex<T>::value) {
        if (trans == Transpose::Trans) {
            throw batchlas::unsupported("ormbr_buffer_size: Trans not supported for complex 'P'; use ConjTrans");
        }
    }

    if (ormbr_p_use_cta<B, T>(a, c, side)) {
        return ormbr_p_cta_buffer_size<T>(ctx, a);
    }

    const int nb = std::max<int>(1, block_size);
    switch (nb) {
        case 16:
            return ormbr_p_blocked_buffer_size_impl<B, T, 16>(ctx, a, c, side, block_size);
        case 32:
            return ormbr_p_blocked_buffer_size_impl<B, T, 32>(ctx, a, c, side, block_size);
        case 64:
            return ormbr_p_blocked_buffer_size_impl<B, T, 64>(ctx, a, c, side, block_size);
        case 128:
            return ormbr_p_blocked_buffer_size_impl<B, T, 128>(ctx, a, c, side, block_size);
        default:
            return ormbr_p_blocked_buffer_size_impl<B, T, -1>(ctx, a, c, side, block_size);
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

} // namespace batchlas
