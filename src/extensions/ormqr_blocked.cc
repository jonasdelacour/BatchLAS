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

// Namespace-scope so the SYCL kernel name is externally nameable and distinct
// from geqrf_blocked.cc's own tag; do not move it into the anonymous namespace.
struct OrmqrWyTag;

namespace {

// larft builds T upper triangular and zeroes the half below, so W2 = op(T) W1
// can be spelled as a trmm rather than a GEMM through those zeros.
// BATCHLAS_ORMQR_WY=gemm|trmm pins the spelling past the routing below.
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

// Substitute the trmm only where its batched triangular tile kernel is both
// linked and ahead of the GEMM it replaces: real types, ib <= 64.
// evidence: docs/perf/qr.md#ormqr-wy-trmm-gate
template <Backend B, typename T>
inline bool wy_trmm_applicable(int ib) {
    const WyPin pin = wy_pin();
    if (pin != WyPin::Measured) {
        return pin == WyPin::Trmm;
    }
    // Ask whether the tile kernel is linked, not what the backend is: a
    // vendor-free build is still Backend::CUDA but need not compile that TU.
    constexpr bool route_has_tile_kernel = dispatch::level3_tile_route_available<B, T>;
    constexpr bool type_beats_gemm_at_this_m = !internal::is_complex<T>::value;
    return route_has_tile_kernel && type_beats_gemm_at_this_m && ib <= 64;
}

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
            // W2 = op(T) W1. The trmm substitution is valid only because every
            // trmm route ormqr can reach writes C with beta = 0, as this GEMM does.
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
