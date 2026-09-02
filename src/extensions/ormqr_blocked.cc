#include <batchlas/blas/device.hh>
#include <batchlas/blas/dispatch/route_compiled.hh>
#include <batchlas/blas/functions.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/internal/ormqr_blocked.hh>
#include <batchlas/util/mempool.hh>
#include <batchlas/backend_config.h>

#include "../math-helpers.hh"
#include "larft_wy.hh"
#include "../queue.hh"
#include "../util/template-instantiations.hh"

#include <algorithm>
#include <complex>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace batchlas {

// The caller tag that names this TU's copies of the shared WY kernels. At
// NAMESPACE scope, not inside the anonymous namespace: a SYCL kernel name is
// safest as an externally-nameable type, and geqrf_blocked.cc declares its own.
struct OrmqrWyTag;

namespace {

// The WY block factor W2 = op(T) W1 has T upper triangular -- larft builds it
// that way and zeroes the half below, which is why the GEMM spelling below is
// correct in the first place. A trmm expresses it without multiplying through
// those zeros.
//
// BATCHLAS_ORMQR_WY pins the spelling so both stay measurable from one binary:
// `gemm` is the old one, and `trmm` forces the substitution past the routing
// below -- which is how a route the routing currently rejects gets measured at
// all, rather than by editing the predicate and rebuilding.
enum class WyPin { Measured, Gemm, Trmm };

inline WyPin wy_pin() {
    static const WyPin result = []() {
        const char* v = std::getenv("BATCHLAS_ORMQR_WY");
        if (!v) return WyPin::Measured;
        const std::string s(v);
        if (s == "gemm") return WyPin::Gemm;
        if (s == "trmm") return WyPin::Trmm;
        return WyPin::Measured;
    }();
    return result;
}

// Where the tile kernel is ahead of the GEMM it replaces.
//
// This gate was written as `CUDA && float` on the argument that CUDA float was
// the only combination reaching a batched triangular kernel at all. That
// argument has expired -- the tile kernel is plain SYCL and type-generic, and
// the CUDA router now sends double and complex to it too -- so the gate was
// lifted outright and then measured, ormqr_blocked_benchmark, Side::Left,
// ABBA-ordered against BATCHLAS_ORMQR_WY=gemm on a dedicated 4090, n in
// {256,512,1024}, batch 128-256, nb in {16,32,64}. Ratios are gemm/trmm, so
// above 1.00 is trmm ahead:
//
//   float            1.006x - 1.046x   every cell
//   double           1.004x - 1.016x   every cell
//   complex<float>   0.944x - 0.995x   every cell, ~3-5% behind
//   complex<double>  0.958x - 1.010x   behind at ib 16, level above it
//   netlib float     0.336x - 1.199x   0.34x at n 128, ib 16
//   netlib double    0.379x - 1.064x   0.38x at n 128, ib 16
//
// So the dtype half was stale only for double, and the answer is per-type
// rather than per-precision. m here is ib, in the tens, which was the one regime
// the tile kernel could not use its structure in: the 32-row tile was its floor,
// so ib <= 32 ran at R = 1, (R+1)/2R = 1 skipped no arithmetic whatsoever, and
// ib = 16 additionally threw away half of what it computed at the epilogue.
//
// A 16-row tile was then added to see whether giving complex an R that pays
// would close the gap (trmm_row_tile in trmm_triangular_tiles.hh, which is where
// the tile-16-vs-tile-32 numbers live). It moves every type in the direction the
// R argument predicts, but it does not change this gate. Against the GEMM, on
// tile 16:
//
//   double           1.013x - 1.036x   was 1.004x - 1.016x on tile 32
//   complex<double>  0.996x - 1.018x   was 0.958x - 1.010x; median exactly 1.000
//   complex<float>   0.946x - 0.983x   was 0.944x - 0.995x; still behind
//
// complex<double>'s ib = 16 hole does close (0.958x -> 0.996x), but closing to
// parity is not a reason to switch a call site, and complex<float> stays 2-5%
// behind because the deficit there is not the R saving: a complex multiply is
// four real ones, so the kernel is that much further from cuBLAS's per-flop rate
// to begin with, and one 16x16 triangle's worth of saved arithmetic cannot pay
// it back. Confirmed by trace that complex takes
// trmm_cuda_custom.triangular_tiles and not the expansion fallback, so this is
// the kernel's own shape response, not a mis-route. The type stays excluded; the
// tile is kept because double wants it.
//
// netlib is excluded for a different reason. Its trmm and its gemm are both
// per-batch cblas loops, so the batching is not the difference: OpenBLAS's
// ?trmm is simply weak on a 16x16 triangle against 128 right-hand sides, and
// the route also copies B into C first because cblas_?trmm works in place.
// ROCm is excluded because rocblas_?trmm is a per-batch vendor loop against a
// strided-batched GEMM -- wire the tile kernel into rocblas.cc's trmm and this
// predicate is where to re-measure.
//
// ib <= 64 predates all of the above and stays: past it the tile kernel
// measured 0.83x-0.97x in float (see docs/perf/qr.md#ormqr-wy-trmm-gate),
// and every block size syev uses is inside that anyway.
template <Backend B, typename T>
inline bool wy_trmm_applicable(int ib) {
    const WyPin pin = wy_pin();
    if (pin != WyPin::Measured) {
        return pin == WyPin::Trmm;
    }
    // The routes whose trmm reaches the batched triangular tile kernel. The
    // comment this replaces already said the right thing -- "not a statement
    // about CUDA the vendor, it is where the kernel is wired" -- while the
    // expression said `B == Backend::CUDA`. Now it asks what it means. That is
    // not cosmetic: in a vendor-free build the backend is still Backend::CUDA
    // but the TU carrying the tile kernel is not compiled, so the old form
    // claimed a kernel that is not linked.
    constexpr bool route_has_tile_kernel = dispatch::level3_tile_route_available<B, T>;
    constexpr bool type_beats_gemm_at_this_m = !internal::is_complex<T>::value;
    return route_has_tile_kernel && type_beats_gemm_at_this_m && ib <= 64;
}

// Returns true when BATCHLAS_ORMQR_IMPL=device, false otherwise (legacy is the default).
// Evaluated once and cached for the lifetime of the process.
inline bool use_device_ormqr() {
    static const bool result = []() {
        const char* v = std::getenv("BATCHLAS_ORMQR_IMPL");
        return v && std::string(v) == "device";
    }();
    return result;
}

template <typename T>
inline void validate_ormqr_dims(const MatrixView<T, MatrixFormat::Dense>& a,
                               const MatrixView<T, MatrixFormat::Dense>& c,
                               Side side,
                               Span<T> tau) {
    if (a.batch_size() != c.batch_size()) {
        throw std::runtime_error("ormqr_blocked: expected A.batch_size() == C.batch_size()");
    }
    if (a.batch_size() < 1) {
        throw std::runtime_error("ormqr_blocked: invalid batch_size");
    }
    const int k = std::min(a.rows(), a.cols());
    const int nq = (side == Side::Left) ? c.rows() : c.cols();
    if (a.rows() != nq) {
        throw std::runtime_error("ormqr_blocked: expected A.rows() == nq (order of Q)");
    }
    const size_t need_tau = static_cast<size_t>(k) * static_cast<size_t>(a.batch_size());
    if (tau.size() < need_tau) {
        throw std::runtime_error("ormqr_blocked: tau too small for batch");
    }
}

// LARFT and PACK_V ARE NO LONGER PRIVATE TO THIS FILE.
//
// Both bodies moved verbatim to src/extensions/larft_wy.hh so that WP5's blocked
// geqrf could use them instead of adding a sixth private copy to this tree (the
// count was two here plus a third, now-unreachable one in sytrd_sy2sb.cc:233).
// The work-group ladder, the two implementations and their thresholds are
// unchanged; the only differences are that the kernel names are tagged by caller
// -- so two TUs in different device-code clusters cannot emit the same SYCL
// kernel name -- and that `use_device` is a PARAMETER rather than a getenv read,
// which keeps BATCHLAS_ORMQR_IMPL an ormqr decision.

} // namespace

template <int NB>
inline int resolved_nb(int32_t block_size) {
    if constexpr (NB > 0) {
        return NB;
    }
    return std::max<int>(1, block_size);
}

template <Backend B, typename T, int NB>
size_t ormqr_blocked_buffer_size_impl(Queue& ctx,
                                      const MatrixView<T, MatrixFormat::Dense>& a,
                                      const MatrixView<T, MatrixFormat::Dense>& c,
                                      Side side,
                                      Transpose trans,
                                      Span<T> tau,
                                      int32_t block_size) {
    (void)trans;
    (void)tau;

    const int nq = a.rows();
    const int m = c.rows();
    const int n = c.cols();
    const int batch = a.batch_size();

    const int nb = resolved_nb<NB>(block_size);

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
Event ormqr_blocked_impl(Queue& ctx,
                         const MatrixView<T, MatrixFormat::Dense>& a,
                         const MatrixView<T, MatrixFormat::Dense>& c,
                         Side side,
                         Transpose trans,
                         Span<T> tau,
                         Span<std::byte> workspace,
                         int32_t block_size) {
    const int nq = a.rows();
    const int mC = c.rows();
    const int nC = c.cols();
    const int k = std::min(a.rows(), a.cols());
    const int batch = a.batch_size();

    const int nb = resolved_nb<NB>(block_size);

    Queue& q = ctx;

    BumpAllocator pool(workspace);
    auto Vbuf = pool.allocate<T>(q, static_cast<size_t>(nq) * static_cast<size_t>(nb) * static_cast<size_t>(batch));
    auto Tbuf = pool.allocate<T>(q, static_cast<size_t>(nb) * static_cast<size_t>(nb) * static_cast<size_t>(batch));

    const size_t w_elems = (side == Side::Left)
                               ? static_cast<size_t>(nb) * static_cast<size_t>(nC)
                               : static_cast<size_t>(mC) * static_cast<size_t>(nb);
    auto W1buf = pool.allocate<T>(q, w_elems * static_cast<size_t>(batch));
    auto W2buf = pool.allocate<T>(q, w_elems * static_cast<size_t>(batch));

    MatrixView<T, MatrixFormat::Dense> Vmat(Vbuf.data(), nq, nb, nq, nq * nb, batch);
    MatrixView<T, MatrixFormat::Dense> Tmat(Tbuf.data(), nb, nb, nb, nb * nb, batch);

    const bool transpose_apply = (trans != Transpose::NoTrans);

    auto apply_block = [&](int i0) {
        const int ib = std::min(nb, k - i0);

        const int m = nq - i0;
        {
            BATCHLAS_KERNEL_TRACE_SCOPE("ormqr_blocked.pack_v_panel");
            (void)wy::pack_v_panel_batched<OrmqrWyTag, T>(
                q, Vmat.data_ptr(), Vmat.ld(), Vmat.stride(), a, i0, ib, nq);
        }

        {
            BATCHLAS_KERNEL_TRACE_SCOPE("ormqr_blocked.larft");
            (void)wy::larft_forward_columnwise_batched<OrmqrWyTag, T>(
                q, Tmat.data_ptr(), Tmat.ld(), Tmat.stride(),
                Vmat.data_ptr(), Vmat.ld(), Vmat.stride(), m, ib,
                tau.data(), /*tau_stride=*/k, /*tau_offset=*/i0, batch,
                /*use_device=*/use_device_ormqr());
        }

        if (side == Side::Left) {
            auto Csub = c({i0, SliceEnd()}, Slice());
            auto Vblk = Vmat({0, m}, {0, ib});
            auto Tblk = Tmat({0, ib}, {0, ib});

            MatrixView<T, MatrixFormat::Dense> W1full(W1buf.data(), nb, nC, nb, nb * nC, batch);
            MatrixView<T, MatrixFormat::Dense> W2full(W2buf.data(), nb, nC, nb, nb * nC, batch);
            auto W1 = W1full({0, ib}, Slice());
            auto W2 = W2full({0, ib}, Slice());

            gemm<B>(q, Vblk, Csub, W1, {.transA = Transpose::ConjTrans});

            const Transpose t_eff = transpose_apply ? Transpose::ConjTrans : Transpose::NoTrans;
            // W2 = op(T) W1. Every trmm ormqr is instantiated for writes C
            // rather than accumulating into it -- the tile kernel and the
            // expand-plus-GEMM fallback both pass beta = 0, rocBLAS uses the
            // out-of-place 14-argument form, and the netlib route copies B into
            // C and calls the in-place cblas_?trmm on it -- which matches this
            // GEMM's beta = 0, so the two are interchangeable here. (The
            // accumulating trmm in extensions/trmm.cc is MKL-only and ormqr is
            // not instantiated for MKL.)
            if (wy_trmm_applicable<B, T>(ib)) {
                trmm<B, T>(q, Tblk, W1, W2, T(1),
                           Side::Left, Uplo::Upper, t_eff, Diag::NonUnit);
            } else {
                gemm<B>(q, Tblk, W1, W2, {.transA = t_eff});
            }

            gemm<B>(q, Vblk, W2, Csub, {.alpha = T(-1), .beta = T(1)});
        } else {
            auto Csub = c(Slice(), {i0, SliceEnd()});
            auto Vblk = Vmat({0, m}, {0, ib});
            auto Tblk = Tmat({0, ib}, {0, ib});

            MatrixView<T, MatrixFormat::Dense> W1full(W1buf.data(), mC, nb, mC, mC * nb, batch);
            MatrixView<T, MatrixFormat::Dense> W2full(W2buf.data(), mC, nb, mC, mC * nb, batch);
            auto W1 = W1full(Slice(), {0, ib});
            auto W2 = W2full(Slice(), {0, ib});

            gemm<B>(q, Csub, Vblk, W1, GemmOptions<T>{});

            const Transpose t_eff = transpose_apply ? Transpose::ConjTrans : Transpose::NoTrans;
            gemm<B>(q, W1, Tblk, W2, {.transB = t_eff});

            gemm<B>(q,
                    W2,
                    Vblk,
                    Csub,
                    {.alpha = T(-1), .beta = T(1), .transB = Transpose::ConjTrans});
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

    return ctx.get_event();
}

template <Backend B, typename T>
size_t ormqr_blocked_buffer_size(Queue& ctx,
                                 const MatrixView<T, MatrixFormat::Dense>& a,
                                 const MatrixView<T, MatrixFormat::Dense>& c,
                                 Side side,
                                 Transpose trans,
                                 Span<T> tau,
                                 int32_t block_size) {
    validate_ormqr_dims(a, c, side, tau);

    const int nb = std::max<int>(1, block_size);
    switch (nb) {
        case 16:
            return ormqr_blocked_buffer_size_impl<B, T, 16>(ctx, a, c, side, trans, tau, block_size);
        case 32:
            return ormqr_blocked_buffer_size_impl<B, T, 32>(ctx, a, c, side, trans, tau, block_size);
        case 64:
            return ormqr_blocked_buffer_size_impl<B, T, 64>(ctx, a, c, side, trans, tau, block_size);
        case 128:
            return ormqr_blocked_buffer_size_impl<B, T, 128>(ctx, a, c, side, trans, tau, block_size);
        default:
            return ormqr_blocked_buffer_size_impl<B, T, -1>(ctx, a, c, side, trans, tau, block_size);
    }
}

template <Backend B, typename T>
Event ormqr_blocked(Queue& ctx,
                    const MatrixView<T, MatrixFormat::Dense>& a,
                    const MatrixView<T, MatrixFormat::Dense>& c,
                    Side side,
                    Transpose trans,
                    Span<T> tau,
                    Span<std::byte> workspace,
                    int32_t block_size) {
    validate_ormqr_dims(a, c, side, tau);

    if (!ctx.in_order()) {
        throw std::runtime_error("ormqr_blocked: requires an in-order Queue");
    }

    if constexpr (internal::is_complex<T>::value) {
        if (trans == Transpose::Trans) {
            throw std::runtime_error("ormqr_blocked: Trans not supported for complex; use ConjTrans");
        }
    }

    const int nb = std::max<int>(1, block_size);
    switch (nb) {
        case 16:
            return ormqr_blocked_impl<B, T, 16>(ctx, a, c, side, trans, tau, workspace, block_size);
        case 32:
            return ormqr_blocked_impl<B, T, 32>(ctx, a, c, side, trans, tau, workspace, block_size);
        case 64:
            return ormqr_blocked_impl<B, T, 64>(ctx, a, c, side, trans, tau, workspace, block_size);
        case 128:
            return ormqr_blocked_impl<B, T, 128>(ctx, a, c, side, trans, tau, workspace, block_size);
        default:
            return ormqr_blocked_impl<B, T, -1>(ctx, a, c, side, trans, tau, workspace, block_size);
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

#define ORMQR_BLOCKED_INSTANTIATE_FOR_BACKEND(back) \
    BATCHLAS_FOR_EACH_SCALAR_TYPE_1(ORMQR_BLOCKED_INSTANTIATE, back)

#if BATCHLAS_HAS_CUDA_BACKEND
ORMQR_BLOCKED_INSTANTIATE_FOR_BACKEND(Backend::CUDA)
#endif

#if BATCHLAS_HAS_ROCM_BACKEND
ORMQR_BLOCKED_INSTANTIATE_FOR_BACKEND(Backend::ROCM)
#endif

#if BATCHLAS_HAS_HOST_BACKEND
ORMQR_BLOCKED_INSTANTIATE_FOR_BACKEND(Backend::NETLIB)
#endif

#undef ORMQR_BLOCKED_INSTANTIATE_FOR_BACKEND
#undef ORMQR_BLOCKED_INSTANTIATE

} // namespace batchlas
