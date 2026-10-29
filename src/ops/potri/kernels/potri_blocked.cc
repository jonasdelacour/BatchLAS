// potri's blocked family, LAPACK's ?potri: ?trtri (every diagonal block inverted by the CTA kernel
// in one launch, then two trmm per block column) and ?lauum (trmm, gemm, a copy back, the CTA
// kernel on the diagonal block and herk/syrk per block row). Only the uplo triangle is read or
// written; W holds one off-diagonal panel per item (trmm is out of place). An item with an exactly
// zero diagonal entry is left unchanged, as ?potri leaves it: the first CTA launch copies it to its
// slot, the last one copies it back (sycl_potri::Stash). The batch runs in L2-sized parts, so W and
// the slots are one part's. evidence: docs/perf/potri.md#potri-the-algorithm

#include <batchlas/backend_config.h>

#include <batchlas/blas/functions/gemm.hh>
#include <batchlas/blas/functions/herk.hh>
#include <batchlas/blas/functions/syrk.hh>
#include <batchlas/blas/functions/trmm.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/error.hh>
#include <batchlas/tuning_params.hh>
#include <batchlas/util/mempool.hh>

#include <src/ops/potri/potri_native.hh>
#include <src/util/template-instantiations.hh>
#include <src/util/batch_mask.hh>

#include <algorithm>
#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace batchlas::ops::potri::blocked {

template <class T>
using MV = MatrixView<T, MatrixFormat::Dense>;

// Three pointer arrays: no call takes more than three views, and a vendor child fills each view's own.
template <class T>
struct Ws {
    Span<T> w, slots;
    std::array<Span<T*>, 3> ptrs;
    Span<int32_t> flags;  // the whole batch's, when the caller passes no info
};

template <class T>
Ws<T> layout(Queue& q, BumpAllocator& pool, int n, int nb, int part, int batch) {
    Ws<T> ws;
    ws.w = pool.allocate<T>(q, std::max<std::size_t>(1, static_cast<std::size_t>(n) * nb * part));
    ws.slots = pool.allocate<T>(q, static_cast<std::size_t>(n) * (n + 1) / 2 * part);
    for (auto& p : ws.ptrs) p = pool.allocate<T*>(q, static_cast<std::size_t>(part));
    ws.flags = pool.allocate<int32_t>(q, static_cast<std::size_t>(batch));
    return ws;
}

// Items per part: an item is its triangle plus its W, through tuning::blocked_chunk_items (potrf's rule).
template <class T>
int chunk(Queue& q, int n, int nb, int batch) {
    const auto dev = q.device();
    const int64_t item = (static_cast<int64_t>(n) * (n + 1) / 2 + static_cast<int64_t>(n) * nb) *
                         static_cast<int64_t>(sizeof(T));
    return static_cast<int>(tuning::blocked_chunk_items(
        batch, item, static_cast<int64_t>(dev.get_property(DeviceProperty::GLOBAL_MEM_CACHE_SIZE)),
        static_cast<int64_t>(dev.get_property(DeviceProperty::MAX_COMPUTE_UNITS))));
}

// The largest part: the last one takes the remainder, and under two chunks there is one.
inline int part_of(int chunk, int batch) { return batch < 2 * chunk ? batch : chunk + batch % chunk; }

// Zero at n <= nb: one block is the cta family's launch, in place.
template <class T>
std::size_t workspace(Queue& q, const MV<T>& A, int nb) {
    const int n = A.rows(), batch = A.batch_size();
    if (n <= nb) return 0;
    return workspace_bytes(
        [&](BumpAllocator& p) { return layout<T>(q, p, n, nb, part_of(chunk<T>(q, n, nb, batch), batch), batch); });
}

template <class T>
struct Part {
    T* a;
    int ld, stride, batch;
    Ws<T>* ws;
    int n, wstride;
    Span<const std::uint8_t> mask;  // the part's slice of the batch mask; every view carries it
    MV<T> sub(int r0, int nr, int c0, int nc, int role) const {
        return MV<T>(a + static_cast<std::ptrdiff_t>(c0) * ld + r0, nr, nc, ld, stride, batch, ws->ptrs[role].data())
            .with_batch_mask(mask);
    }
    MV<T> w(int nr, int nc, int role) const {
        return MV<T>(ws->w.data(), nr, nc, nr, wstride, batch, ws->ptrs[role].data()).with_batch_mask(mask);
    }
};

template <class T>
inline constexpr Transpose kAdj = std::is_same_v<T, typename base_type<T>::type> ? Transpose::Trans : Transpose::ConjTrans;

// C += op(A) op(A)^H on the uplo triangle: herk for complex scalars, syrk for real ones.
template <Backend B, class T>
Event rank_k(Queue& q, const MV<T>& A, const MV<T>& C, Uplo uplo, Transpose trans) {
    if constexpr (std::is_same_v<T, typename base_type<T>::type>)
        return syrk<B, T>(q, A, C, T(1), T(1), uplo, trans);
    else
        return herk<B, T>(q, A, C, float_t<T>(1), float_t<T>(1), uplo, trans);
}

template <Backend B, class T>
void trtri(Queue& q, const MV<T>& Ap, const Part<T>& p, Uplo uplo, int nb, const sycl_potri::Stash<T>& st) {
    const int n = p.n, nblk = (n + nb - 1) / nb;
    (void)sycl_potri::potri_cta<T>(q, Ap, uplo, sycl_potri::CtaMode::Trtri, nb, false, {}, st);
    if (uplo == Uplo::Lower) {
        // X21 = -X22 L21 X11, last block column first: X22 is already inverted.
        for (int blk = nblk - 2; blk >= 0; --blk) {
            const int j = blk * nb, m = n - j - nb;
            (void)trmm<B, T>(q, p.sub(j + nb, m, j + nb, m, 0), p.sub(j + nb, m, j, nb, 1), p.w(m, nb, 2), T(1),
                             Side::Left, Uplo::Lower, Transpose::NoTrans, Diag::NonUnit);
            (void)trmm<B, T>(q, p.sub(j, nb, j, nb, 0), p.w(m, nb, 1), p.sub(j + nb, m, j, nb, 2), T(-1),
                             Side::Right, Uplo::Lower, Transpose::NoTrans, Diag::NonUnit);
        }
        return;
    }
    // X12 = -X11 U12 X22, first block column first: X11 is already inverted.
    for (int blk = 1; blk < nblk; ++blk) {
        const int j = blk * nb, jb = std::min(nb, n - j);
        (void)trmm<B, T>(q, p.sub(0, j, 0, j, 0), p.sub(0, j, j, jb, 1), p.w(j, jb, 2), T(1), Side::Left, Uplo::Upper,
                         Transpose::NoTrans, Diag::NonUnit);
        (void)trmm<B, T>(q, p.sub(j, jb, j, jb, 0), p.w(j, jb, 1), p.sub(0, j, j, jb, 2), T(-1), Side::Right,
                         Uplo::Upper, Transpose::NoTrans, Diag::NonUnit);
    }
}

// The last diagonal block's Lauum is the part's last launch: it restores the flagged items.
template <Backend B, class T>
void lauum(Queue& q, const Part<T>& p, Uplo uplo, int nb, sycl_potri::Stash<T> st) {
    const int n = p.n;
    const bool lower = uplo == Uplo::Lower;
    for (int i = 0; i < n; i += nb) {
        const int ib = std::min(nb, n - i), rest = n - i - ib;
        const MV<T> Xii = p.sub(i, ib, i, ib, 0);
        if (i > 0) {
            // Lower: Y(i, 0:i) = Xii^H X(i, 0:i) + X(i+ib:, i)^H X(i+ib:, 0:i); Upper the transpose.
            const MV<T> panel = lower ? p.sub(i, ib, 0, i, 1) : p.sub(0, i, i, ib, 1);
            const MV<T> W = lower ? p.w(ib, i, 2) : p.w(i, ib, 2);
            (void)trmm<B, T>(q, Xii, panel, W, T(1), lower ? Side::Left : Side::Right, uplo, kAdj<T>, Diag::NonUnit);
            if (rest > 0) {
                if (lower)
                    (void)gemm<B, T>(q, p.sub(i + ib, rest, i, ib, 0), p.sub(i + ib, rest, 0, i, 1), W, T(1), T(1),
                                     kAdj<T>, Transpose::NoTrans);
                else
                    (void)gemm<B, T>(q, p.sub(0, i, i + ib, rest, 0), p.sub(i, ib, i + ib, rest, 1), W, T(1), T(1),
                                     Transpose::NoTrans, kAdj<T>);
            }
            (void)MV<T>::copy(q, panel, W);
        }
        st.origin = i;
        (void)sycl_potri::potri_cta<T>(q, Xii, uplo, sycl_potri::CtaMode::Lauum, ib, false, {},
                                       rest > 0 ? sycl_potri::Stash<T>{} : st);
        if (rest > 0) {
            const MV<T> Cii = p.sub(i, ib, i, ib, 2);
            if (lower) (void)rank_k<B, T>(q, p.sub(i + ib, rest, i, ib, 0), Cii, uplo, kAdj<T>);
            else (void)rank_k<B, T>(q, p.sub(i, ib, i + ib, rest, 0), Cii, uplo, Transpose::NoTrans);
        }
    }
}

// The caller (potri.cc) checked the workspace and holds an in-order queue.
template <Backend B, class T>
Event run(Queue& q, const MV<T>& A, Uplo uplo, int nb, bool full, Span<std::byte> ws_span, Span<int32_t> info) {
    const int n = A.rows(), batch = A.batch_size();
    constexpr bool kCx = !std::is_same_v<T, typename base_type<T>::type>;
    // One block is the cta family's launch: info, the untouched failed item and full in one kernel.
    if (n <= nb) return sycl_potri::potri_cta<T>(q, A, uplo, sycl_potri::CtaMode::Potri, n, full, info);
    // The Trtri launch writes flags[0, batch) unguarded: refuse a short span here, not only at the entry point.
    if (!info.empty() && info.size() < static_cast<std::size_t>(batch))
        throw batchlas::internal_error("potri: info span shorter than the batch reached the blocked driver");
    int c = chunk<T>(q, n, nb, batch);
    auto part_bytes = [&](int cc) {
        return workspace_bytes([&](BumpAllocator& pl) { return layout<T>(q, pl, n, nb, part_of(cc, batch), batch); });
    };
    while (c > 1 && part_bytes(c) > ws_span.size()) c = (c + 1) / 2;
    BumpAllocator pool(ws_span);
    Ws<T> ws = layout<T>(q, pool, n, nb, part_of(c, batch), batch);
    int32_t* const flags = info.empty() ? ws.flags.data() : info.data();
    const Span<const std::uint8_t> mask = batchlas::detail::call_mask("potri", A);
    for (int b0 = 0, cb = 0; b0 < batch; b0 += cb) {
        cb = batch - b0 < 2 * c ? batch - b0 : c;
        const Span<const std::uint8_t> part_mask =
            mask.empty() ? mask : mask.subspan(static_cast<std::size_t>(b0), static_cast<std::size_t>(cb));
        // Never operator()(Slice, Slice): a 0 stride defaults to ld*cols on a sub-view.
        const Part<T> p{A.data_ptr() + static_cast<std::ptrdiff_t>(b0) * A.stride(), A.ld(), A.stride(), cb, &ws, n, n * nb,
                        part_mask};
        const MV<T> Ap = MV<T>(p.a, n, n, p.ld, p.stride, cb, ws.ptrs[0].data()).with_batch_mask(part_mask);
        const sycl_potri::Stash<T> st{ws.slots.data(), flags + b0, n, 0};
        trtri<B, T>(q, Ap, p, uplo, nb, st);
        lauum<B, T>(q, p, uplo, nb, st);
    }
    // herk leaves the diagonal's imaginary part to the kernel it picks: the finish zeroes it.
    if (kCx || full) return sycl_potri::potri_finish<T>(q, A, uplo, full, flags);
    return q.get_event();
}

template std::size_t workspace<float>(Queue&, const MV<float>&, int);
template std::size_t workspace<double>(Queue&, const MV<double>&, int);
template std::size_t workspace<std::complex<float>>(Queue&, const MV<std::complex<float>>&, int);
template std::size_t workspace<std::complex<double>>(Queue&, const MV<std::complex<double>>&, int);

#define POTRI_BLOCKED_INSTANTIATE(back, fp)                                                                      \
    template Event run<back, BATCHLAS_UNPAREN fp>(Queue&, const MV<BATCHLAS_UNPAREN fp>&, Uplo, int, bool, \
                                                  Span<std::byte>, Span<int32_t>);
BATCHLAS_INSTANTIATE_SCALAR_ALL_BACKENDS(POTRI_BLOCKED_INSTANTIATE)
#undef POTRI_BLOCKED_INSTANTIATE

}  // namespace batchlas::ops::potri::blocked
