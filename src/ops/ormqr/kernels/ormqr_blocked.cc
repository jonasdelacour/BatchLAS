// SPDX-License-Identifier: MIT
#include <src/select/grid_chunk.hh>
#include <src/select/vendor.hh>
#include <batchlas/blas/functions.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/internal/ormqr_blocked.hh>
#include <batchlas/util/mempool.hh>
#include <batchlas/backend_config.h>
#include <batchlas/tuning_params.hh>

#include <src/math-helpers.hh>
#include <src/kernels/common/larft_wy.hh>
#include <src/ops/ormqr/wy_spelling.hh>
#include <src/queue.hh>
#include <src/util/template-instantiations.hh>
#include <src/util/batch_mask.hh>

#include <algorithm>
#include <complex>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <batchlas/settings.hh>

namespace batchlas {

// Namespace-scope so the SYCL kernel name is externally nameable and distinct
// from geqrf_blocked.cc's own tag; do not move it into the anonymous namespace.
struct OrmqrWyTag;

namespace {

using ops::ormqr::wy_trmm_applicable;

template <typename Scalar>
inline void validate_ormqr_dims(const MatrixView<Scalar, MatrixFormat::Dense>& a,
                               const MatrixView<Scalar, MatrixFormat::Dense>& c,
                               Side side,
                               Span<Scalar> tau) {
    if (a.batch_size() != c.batch_size()) {
        throw batchlas::invalid_argument("ormqr_blocked: expected A.batch_size() == C.batch_size()");
    }
    if (a.batch_size() < 1) {
        throw batchlas::invalid_argument("ormqr_blocked: invalid batch_size");
    }
    const int k = std::min(a.rows(), a.cols());
    const int nq = (side == Side::Left) ? c.rows() : c.cols();
    if (a.rows() != nq) {
        throw batchlas::invalid_argument("ormqr_blocked: expected A.rows() == nq (order of Q)");
    }
    const size_t need_tau = static_cast<size_t>(k) * static_cast<size_t>(a.batch_size());
    if (tau.size() < need_tau) {
        throw batchlas::invalid_argument("ormqr_blocked: tau too small for batch");
    }
}

}  // namespace

template <int NB>
inline int resolved_nb(int32_t block_size) {
    if constexpr (NB > 0) {
        return NB;
    }
    return std::max<int>(1, block_size);
}

// Items per pass: C, V, W1 and W2 pass between the three gemms of every block through L2,
// and the gemms are routed at this batch. evidence: docs/perf/qr.md#qr-l2-batch-chunking
template <typename Scalar>
int ormqr_blocked_chunk(Queue& ctx, const MatrixView<Scalar, MatrixFormat::Dense>& a,
                        const MatrixView<Scalar, MatrixFormat::Dense>& c, Side side, int nb) {
    const int64_t nq = a.rows(), k = std::min(a.rows(), a.cols());
    const int64_t w = (side == Side::Left) ? int64_t(nb) * c.cols() : int64_t(c.rows()) * nb;
    const int64_t item = (nq * k + int64_t(c.rows()) * c.cols() + nq * nb + 2 * w) *
                         static_cast<int64_t>(sizeof(Scalar));
    const auto dev = ctx.device();
    // pack_v puts the batch in grid z: no part may exceed it.
    return select::grid_capped_chunk(ctx, static_cast<int>(tuning::blocked_chunk_items(
        a.batch_size(), item, static_cast<int64_t>(dev.get_property(DeviceProperty::GLOBAL_MEM_CACHE_SIZE)),
        static_cast<int64_t>(dev.get_property(DeviceProperty::MAX_COMPUTE_UNITS)),
        tuning::ORMQR_BLOCKED_L2_CHUNK_PERCENT)), a.batch_size());
}

// The largest pass: the last part takes the remainder, and under two chunks there is one.
inline int ormqr_blocked_part(int chunk, int batch) {
    return batch < 2 * chunk ? batch : chunk + batch % chunk;
}

template <typename Scalar>
size_t ormqr_blocked_part_bytes(Queue& ctx, int nq, int m, int n, Side side, int nb, int batch) {
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
size_t ormqr_blocked_buffer_size_impl(Queue& ctx,
                                      const MatrixView<Scalar, MatrixFormat::Dense>& a,
                                      const MatrixView<Scalar, MatrixFormat::Dense>& c,
                                      Side side,
                                      Transpose trans,
                                      Span<Scalar> tau,
                                      int32_t block_size) {
    (void)trans;
    (void)tau;
    const int nb = resolved_nb<NB>(block_size);
    const int part = ormqr_blocked_part(ormqr_blocked_chunk<Scalar>(ctx, a, c, side, nb), a.batch_size());
    return ormqr_blocked_part_bytes<Scalar>(ctx, a.rows(), c.rows(), c.cols(), side, nb, part);
}

template <Backend B, typename Scalar, int NB>
Event ormqr_blocked_impl(Queue& ctx,
                         const MatrixView<Scalar, MatrixFormat::Dense>& a_all,
                         const MatrixView<Scalar, MatrixFormat::Dense>& c_all,
                         Side side,
                         Transpose trans,
                         Span<Scalar> tau_all,
                         Span<std::byte> workspace,
                         int32_t block_size) {
    const int nq = a_all.rows();
    const int mC = c_all.rows();
    const int nC = c_all.cols();
    const int k = std::min(a_all.rows(), a_all.cols());
    const int batch_all = a_all.batch_size();

    const int nb = resolved_nb<NB>(block_size);

    Queue& q = ctx;

    // Chunk by chunk (the queue is in order, so parts sharing the scratch serialise). A caller
    // that sized a larger shape (sytrd's sub-views) may hold less than this shape's part needs,
    // so the chunk halves until it fits. evidence: docs/perf/qr.md#qr-l2-batch-chunking
    int chunk = ormqr_blocked_chunk<Scalar>(ctx, a_all, c_all, side, nb);
    while (chunk > 1 && ormqr_blocked_part_bytes<Scalar>(ctx, nq, mC, nC, side, nb,
                                                         ormqr_blocked_part(chunk, batch_all)) > workspace.size())
        chunk = (chunk + 1) / 2;
    const int part = ormqr_blocked_part(chunk, batch_all);

    BumpAllocator pool(workspace);
    auto Vbuf = pool.allocate<Scalar>(q, static_cast<size_t>(nq) * static_cast<size_t>(nb) * static_cast<size_t>(part));
    auto Tbuf = pool.allocate<Scalar>(q, static_cast<size_t>(nb) * static_cast<size_t>(nb) * static_cast<size_t>(part));

    const size_t w_elems = (side == Side::Left)
                               ? static_cast<size_t>(nb) * static_cast<size_t>(nC)
                               : static_cast<size_t>(mC) * static_cast<size_t>(nb);
    auto W1buf = pool.allocate<Scalar>(q, w_elems * static_cast<size_t>(part));
    auto W2buf = pool.allocate<Scalar>(q, w_elems * static_cast<size_t>(part));

    const bool transpose_apply = (trans != Transpose::NoTrans);

    // a and c carry one mask (ormqr_blocked attaches it to both); batch_range slices it per part.
    for (int b0 = 0, batch = 0; b0 < batch_all; b0 += batch) {
        batch = batch_all - b0 < 2 * chunk ? batch_all - b0 : chunk;   // the last part takes the remainder
        const auto a = a_all.batch_range(b0, batch);
        const auto c = c_all.batch_range(b0, batch);
        const Span<Scalar> tau = tau_all.subspan(static_cast<size_t>(b0) * k, static_cast<size_t>(batch) * k);
        const auto mask = ::batchlas::detail::call_mask("ormqr_blocked", a, c);
        const std::uint8_t* const mask_p = ::batchlas::detail::mask_ptr(mask);

        // Workspace views are rebuilt from raw pointers: each re-attaches the part's mask.
        const auto Vmat = MatrixView<Scalar, MatrixFormat::Dense>(Vbuf.data(), nq, nb, nq, nq * nb, batch).with_batch_mask(mask);
        const auto Tmat = MatrixView<Scalar, MatrixFormat::Dense>(Tbuf.data(), nb, nb, nb, nb * nb, batch).with_batch_mask(mask);

        auto apply_block = [&](int i0) {
            const int ib = std::min(nb, k - i0);

            const int m = nq - i0;
            {
                BATCHLAS_KERNEL_TRACE_SCOPE("ormqr_blocked.pack_v_panel");
                (void)wy::pack_v_panel_batched<OrmqrWyTag, Scalar>(
                    q, Vmat.data_ptr(), Vmat.ld(), Vmat.stride(), a, i0, ib, nq, mask_p);
            }

            {
                BATCHLAS_KERNEL_TRACE_SCOPE("ormqr_blocked.larft");
                (void)wy::larft_forward_columnwise_batched<OrmqrWyTag, Scalar>(
                    q, Tmat.data_ptr(), Tmat.ld(), Tmat.stride(),
                    Vmat.data_ptr(), Vmat.ld(), Vmat.stride(), m, ib,
                    tau.data(), /*tau_stride=*/k, /*tau_offset=*/i0, batch, mask_p);
            }

            if (side == Side::Left) {
                auto Csub = c({i0, SliceEnd()}, Slice());
                auto Vblk = Vmat({0, m}, {0, ib});
                auto Tblk = Tmat({0, ib}, {0, ib});

                const auto W1full = MatrixView<Scalar, MatrixFormat::Dense>(W1buf.data(), nb, nC, nb, nb * nC, batch).with_batch_mask(mask);
                const auto W2full = MatrixView<Scalar, MatrixFormat::Dense>(W2buf.data(), nb, nC, nb, nb * nC, batch).with_batch_mask(mask);
                auto W1 = W1full({0, ib}, Slice());
                auto W2 = W2full({0, ib}, Slice());

                // (void) on an Event: deliberate. This Queue is in-order, so the next submission
                // is already ordered after this one and the Event carries nothing the caller needs.
                (void)gemm<B>(q, Vblk, Csub, W1, {.transA = Transpose::ConjTrans});

                const Transpose t_eff = transpose_apply ? Transpose::ConjTrans : Transpose::NoTrans;
                // W2 = op(T) W1. The trmm substitution is valid only because every
                // trmm route ormqr can reach writes C with beta = 0, as this GEMM does.
                if (wy_trmm_applicable<B, Scalar>(ib)) {
                    (void)trmm<B, Scalar>(q, Tblk, W1, W2, Scalar(1),
                               Side::Left, Uplo::Upper, t_eff, Diag::NonUnit);
                } else {
                    (void)gemm<B>(q, Tblk, W1, W2, {.transA = t_eff});
                }

                (void)gemm<B>(q, Vblk, W2, Csub, {.alpha = Scalar(-1), .beta = Scalar(1)});
            } else {
                auto Csub = c(Slice(), {i0, SliceEnd()});
                auto Vblk = Vmat({0, m}, {0, ib});
                auto Tblk = Tmat({0, ib}, {0, ib});

                const auto W1full = MatrixView<Scalar, MatrixFormat::Dense>(W1buf.data(), mC, nb, mC, mC * nb, batch).with_batch_mask(mask);
                const auto W2full = MatrixView<Scalar, MatrixFormat::Dense>(W2buf.data(), mC, nb, mC, mC * nb, batch).with_batch_mask(mask);
                auto W1 = W1full(Slice(), {0, ib});
                auto W2 = W2full(Slice(), {0, ib});

                (void)gemm<B>(q, Csub, Vblk, W1, GemmOptions<Scalar>{});

                const Transpose t_eff = transpose_apply ? Transpose::ConjTrans : Transpose::NoTrans;
                (void)gemm<B>(q, W1, Tblk, W2, {.transB = t_eff});

                (void)gemm<B>(q,
                        W2,
                        Vblk,
                        Csub,
                        {.alpha = Scalar(-1), .beta = Scalar(1), .transB = Transpose::ConjTrans});
            }

        };

        const bool forward = (side == Side::Left) ? transpose_apply : !transpose_apply;
        if (forward) {
            for (int i0 = 0; i0 < k; i0 += nb) {
                apply_block(i0);
            }
        } else {
            for (int i0 = ((k - 1) / nb) * nb; i0 >= 0; i0 -= nb) {
                apply_block(i0);
            }
        }
    }

    return ctx.get_event();
}

template <Backend B, typename Scalar>
size_t ormqr_blocked_buffer_size(Queue& ctx,
                                 const MatrixView<Scalar, MatrixFormat::Dense>& a,
                                 const MatrixView<Scalar, MatrixFormat::Dense>& c,
                                 Side side,
                                 Transpose trans,
                                 Span<Scalar> tau,
                                 int32_t block_size) {
    validate_ormqr_dims(a, c, side, tau);

    if (block_size == 0) block_size = tuning::ORMQR_BLOCK_SIZE_MEDIUM;  // the header default
    const int nb = std::max<int>(1, block_size);
    switch (nb) {
        case 16:
            return ormqr_blocked_buffer_size_impl<B, Scalar, 16>(ctx, a, c, side, trans, tau, block_size);
        case 32:
            return ormqr_blocked_buffer_size_impl<B, Scalar, 32>(ctx, a, c, side, trans, tau, block_size);
        case 64:
            return ormqr_blocked_buffer_size_impl<B, Scalar, 64>(ctx, a, c, side, trans, tau, block_size);
        case 128:
            return ormqr_blocked_buffer_size_impl<B, Scalar, 128>(ctx, a, c, side, trans, tau, block_size);
        default:
            return ormqr_blocked_buffer_size_impl<B, Scalar, -1>(ctx, a, c, side, trans, tau, block_size);
    }
}

template <Backend B, typename Scalar>
Event ormqr_blocked(Queue& ctx,
                    const MatrixView<Scalar, MatrixFormat::Dense>& a_in,
                    const MatrixView<Scalar, MatrixFormat::Dense>& c_in,
                    Side side,
                    Transpose trans,
                    Span<Scalar> tau,
                    Span<std::byte> workspace,
                    int32_t block_size) {
    // NETLIB's gemm is its host BLAS, which cannot skip an item: refuse before pack_v launches,
    // which would otherwise still be reading a and writing the workspace after the throw. A
    // vendor-free host's gemm and trmm are native and skip it (select::native_backend).
    if constexpr (!select::native_backend<B>) ::batchlas::detail::refuse_batch_mask("ormqr_blocked", a_in, c_in);
    // Both operands carry the call's mask whichever brought it.
    const auto mask = ::batchlas::detail::call_mask("ormqr_blocked", a_in, c_in);
    const bool adopt = !mask.empty() && a_in.batch_size() == c_in.batch_size();
    const auto a = adopt ? a_in.with_batch_mask(mask) : a_in;
    const auto c = adopt ? c_in.with_batch_mask(mask) : c_in;
    validate_ormqr_dims(a, c, side, tau);

    if (!ctx.in_order()) {
        throw batchlas::invalid_argument("ormqr_blocked: requires an in-order Queue");
    }

    if constexpr (detail::is_complex<Scalar>::value) {
        if (trans == Transpose::Trans) {
            throw batchlas::unsupported("ormqr_blocked: Trans not supported for complex; use ConjTrans");
        }
    }

    if (block_size == 0) block_size = tuning::ORMQR_BLOCK_SIZE_MEDIUM;  // the header default
    const int nb = std::max<int>(1, block_size);
    switch (nb) {
        case 16:
            return ormqr_blocked_impl<B, Scalar, 16>(ctx, a, c, side, trans, tau, workspace, block_size);
        case 32:
            return ormqr_blocked_impl<B, Scalar, 32>(ctx, a, c, side, trans, tau, workspace, block_size);
        case 64:
            return ormqr_blocked_impl<B, Scalar, 64>(ctx, a, c, side, trans, tau, workspace, block_size);
        case 128:
            return ormqr_blocked_impl<B, Scalar, 128>(ctx, a, c, side, trans, tau, workspace, block_size);
        default:
            return ormqr_blocked_impl<B, Scalar, -1>(ctx, a, c, side, trans, tau, workspace, block_size);
    }
}

#define ORMQR_BLOCKED_INSTANTIATE(back, fp) \
    template Event ormqr_blocked<back, BATCHLAS_UNPAREN fp>( \
        Queue&, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        Side, Transpose, \
        Span<BATCHLAS_UNPAREN fp>, \
        Span<std::byte>, \
        int32_t); \
    template size_t ormqr_blocked_buffer_size<back, BATCHLAS_UNPAREN fp>( \
        Queue&, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        Side, Transpose, \
        Span<BATCHLAS_UNPAREN fp>, \
        int32_t);

BATCHLAS_INSTANTIATE_SCALAR_ALL_BACKENDS(ORMQR_BLOCKED_INSTANTIATE)

#undef ORMQR_BLOCKED_INSTANTIATE

}  // namespace batchlas
