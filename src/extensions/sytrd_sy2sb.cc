#include <batchlas/blas/extensions.hh>
#include <batchlas/blas/functions.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/mempool.hh>

#include <sycl/sycl.hpp>

#include <batchlas/backend_config.h>
#include <batchlas/tuning_params.hh>

#include "../math-helpers.hh"
#include "../queue.hh"
#include "../ops/geqrf/geqrf.hh"  // geqrf_buffer_size_bound: sized once, run on sub-views
#include "../util/template-instantiations.hh"

#include <algorithm>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <type_traits>
#include <batchlas/settings.hh>

namespace batchlas {

namespace {

// ---------------------------------------------------------------------------
// WY block width for the panel back-transform. ormqr keys its block width on the
// panel HEIGHT, but the dimension that matters is k = kd; nb = kd wins only where
// the GEMMs are big enough (n >= 1024, batch >= 32), and returns 0 (table) elsewhere.
// BATCHLAS_SY2SB_ORMQR_NB: unset -> gate, 0/"off" -> never hint, >0 -> force.
// Read fresh per call for in-process A/B, but it must NOT change between a
// sytrd_sy2sb_buffer_size query and its sytrd_sy2sb call (workspace is linear in nb).
// evidence: docs/perf/sytrd.md#sytrd-the-dense-to-band-ormqr-block-width-hint
inline int32_t sy2sb_ormqr_nb_env(bool& has_override) {
    // Three-valued (unset / "off"|0 / positive), so the field is the raw value
    // and the strcmp-plus-strtol contract stays here.
    const char* v = batchlas::settings().geometry.sy2sb_ormqr_nb.get();
    has_override = false;
    if (!v || !*v) return -1;                       // unset -> use shape gate
    if (std::strcmp(v, "off") == 0 || std::strcmp(v, "OFF") == 0) {
        has_override = true;
        return 0;
    }
    char* end = nullptr;
    const long parsed = std::strtol(v, &end, 10);
    if (end == v || parsed < 0 || parsed > 1024) return -1;
    has_override = true;
    return static_cast<int32_t>(parsed);
}

// Returns the block-size hint to pass to ormqr / ormqr_buffer_size. 0 means "no
// hint", i.e. let the dispatch use its tuning table (the old behaviour).
inline int32_t sy2sb_ormqr_block_size_hint(int n, int batch, int kd) {
    bool has_override = false;
    const int32_t forced = sy2sb_ormqr_nb_env(has_override);
    if (has_override) {
        if (forced == 0) return 0;
        return std::min<int32_t>(forced, std::max(1, kd));
    }
    if (kd <= 0) return 0;
    // Tuned constant next: 0 means "no opinion, use the shape gate below".
    // Clamped to kd because a WY block wider than the panel is meaningless.
    const int32_t tuned = tuning::sy2sb_ormqr_nb_for_n(n);
    if (tuned > 0) return std::min<int32_t>(tuned, std::max(1, kd));
    // Shape gate: only where the win was measured.
    if (n >= 1024 && batch >= 32) return kd;
    return 0;
}

template <typename T>
inline void validate_sytrd_sy2sb_dims(const MatrixView<T, MatrixFormat::Dense>& a,
                                     const MatrixView<T, MatrixFormat::Dense>& ab,
                                     const VectorView<T>& tau,
                                     Uplo uplo,
                                     int32_t kd) {
    if (a.rows() != a.cols()) {
        throw batchlas::invalid_argument("sytrd_sy2sb: A must be square");
    }
    if (kd < 0) {
        throw batchlas::invalid_argument("sytrd_sy2sb: kd must be non-negative");
    }
    if (uplo != Uplo::Lower && uplo != Uplo::Upper) {
        throw batchlas::invalid_argument("sytrd_sy2sb: invalid uplo");
    }

    const int n = a.rows();
    const int kd_i = kd;
    const int tau_need = std::max(0, n - kd_i);

    if (ab.rows() != kd_i + 1 || ab.cols() != n) {
        throw batchlas::invalid_argument("sytrd_sy2sb: AB must be (kd+1) x n");
    }
    if (tau.size() != tau_need) {
        throw batchlas::invalid_argument("sytrd_sy2sb: tau must have size (n-kd)");
    }
    if (a.batch_size() != ab.batch_size() || a.batch_size() != tau.batch_size()) {
        throw batchlas::invalid_argument("sytrd_sy2sb: batch size mismatch");
    }
    if (a.batch_size() < 1) {
        throw batchlas::invalid_argument("sytrd_sy2sb: invalid batch size");
    }
}

template <typename T>
class ZeroABKernel;

template <typename T>
Event zero_ab(Queue& q, const MatrixView<T, MatrixFormat::Dense>& ab) {
    const int rows = ab.rows();
    const int cols = ab.cols();
    const int ldab = ab.ld();
    const int stride_ab = ab.stride();
    T* ab_ptr = ab.data_ptr();
    const int batch = ab.batch_size();

    (void)q->submit([&](sycl::handler& h) {
        h.parallel_for<ZeroABKernel<T>>(
            sycl::range<3>(static_cast<size_t>(batch), static_cast<size_t>(cols), static_cast<size_t>(rows)),
            [=](sycl::id<3> idx) {
                const int b = static_cast<int>(idx[0]);
                const int j = static_cast<int>(idx[1]);
                const int r = static_cast<int>(idx[2]);
                T* AB = ab_ptr + b * stride_ab;
                AB[r + j * ldab] = T(0);
            });
    });

    return q.get_event();
}

template <typename T>
class CopyBandLowerKernel;

template <typename T>
Event copy_band_lower(Queue& q,
                      const MatrixView<T, MatrixFormat::Dense>& a,
                      const MatrixView<T, MatrixFormat::Dense>& ab,
                      int i0,
                      int pk,
                      int kd) {
    const int n = a.rows();
    const int lda = a.ld();
    const int stride_a = a.stride();
    const int ldab = ab.ld();
    const int stride_ab = ab.stride();
    const T* a_ptr = a.data_ptr();
    T* ab_ptr = ab.data_ptr();
    const int batch = a.batch_size();

    (void)q->submit([&](sycl::handler& h) {
        h.parallel_for<CopyBandLowerKernel<T>>(
            sycl::range<2>(static_cast<size_t>(batch), static_cast<size_t>(pk)),
            [=](sycl::id<2> idx) {
                const int b = static_cast<int>(idx[0]);
                const int jj = static_cast<int>(idx[1]);
                const int j = i0 + jj;
                if (j < 0 || j >= n) return;

                const T* A = a_ptr + b * stride_a;
                T* AB = ab_ptr + b * stride_ab;

                const int lk = std::min(kd, n - 1 - j) + 1;
                for (int r = 0; r < lk; ++r) {
                    AB[r + j * ldab] = A[(j + r) + j * lda];
                }
            });
    });

    return q.get_event();
}

template <typename T>
class CopyTauKernel;

template <typename T>
Event copy_tau_panel_to_out(Queue& q,
                            const T* tau_panel,
                            int tau_panel_ld,
                            const VectorView<T>& tau_out,
                            int i0,
                            int pk) {
    const int stride_tau_out = tau_out.stride();
    T* tau_out_ptr = tau_out.data_ptr();
    const int batch = tau_out.batch_size();

    (void)q->submit([&](sycl::handler& h) {
        h.parallel_for<CopyTauKernel<T>>(
            sycl::range<2>(static_cast<size_t>(batch), static_cast<size_t>(pk)),
            [=](sycl::id<2> idx) {
                const int b = static_cast<int>(idx[0]);
                const int j = static_cast<int>(idx[1]);
                tau_out_ptr[b * stride_tau_out + (i0 + j)] = tau_panel[b * tau_panel_ld + j];
            });
    });

    return q.get_event();
}

} // namespace

template <Backend B, typename T>
size_t sytrd_sy2sb_buffer_size(Queue& ctx,
                               const MatrixView<T, MatrixFormat::Dense>& a_in,
                               const MatrixView<T, MatrixFormat::Dense>& ab_out,
                               const VectorView<T>& tau_out,
                               Uplo uplo,
                               int32_t kd) {
    validate_sytrd_sy2sb_dims(a_in, ab_out, tau_out, uplo, kd);

    const int n = a_in.rows();
    const int batch = a_in.batch_size();
    const int kd_i = kd;

    size_t size = 0;
    // tau_panel: kd per batch (packed per panel)
    size += BumpAllocator::allocation_size<T>(ctx, static_cast<size_t>(kd_i) * static_cast<size_t>(batch));

    // Add workspace for GEQRF + ORMQR on the largest panel/trailing block.
    // Some backends require a valid device pointer for tau when querying.
    if (kd_i > 0 && n > kd_i) {
        const int pn0 = n - kd_i;
        const int pk0 = std::min(pn0, kd_i);
        auto V0 = a_in({kd_i, SliceEnd()}, {0, pk0});
        // Widest blocks the loop can pass to ormqr (see the main loop): both
        // shrink with i, so the i = 0 panel bounds every later one.
        auto A_left0 = a_in({kd_i, SliceEnd()}, {pk0, SliceEnd()});
        auto A_right0 = a_in({pk0, SliceEnd()}, {kd_i, SliceEnd()});

        const size_t tau_elems = static_cast<size_t>(pk0) * static_cast<size_t>(batch);
        T* tau_tmp = sycl::malloc_shared<T>(tau_elems, ctx->get_device(), ctx->get_context());
        if (!tau_tmp && tau_elems != 0) {
            throw std::bad_alloc();
        }
        Span<T> tau_span(tau_tmp, tau_elems);

        // MUST use exactly the same hint as the ormqr calls in sytrd_sy2sb below:
        // the blocked provider's V (nq*nb), T (nb*nb) and W1/W2 (nb*nC) are all
        // linear in nb, so a mismatch silently overruns the BumpAllocator.
        const int32_t ormqr_nb_hint = sy2sb_ormqr_block_size_hint(n, batch, kd_i);

        const size_t geqrf_ws = geqrf_buffer_size_bound<B, T>(ctx, V0, tau_span);
        const Transpose trans_left = internal::is_complex<T>::value ? Transpose::ConjTrans : Transpose::Trans;
        const size_t ormqr_l_ws = ormqr_buffer_size<B, T>(ctx, V0, A_left0, Side::Left, trans_left, tau_span, ormqr_nb_hint);
        const size_t ormqr_r_ws = ormqr_buffer_size<B, T>(ctx, V0, A_right0, Side::Right, Transpose::NoTrans, tau_span, ormqr_nb_hint);
        const size_t panel_ws = std::max(geqrf_ws, std::max(ormqr_l_ws, ormqr_r_ws));

        sycl::free(tau_tmp, ctx->get_context());
        size += BumpAllocator::allocation_size<std::byte>(ctx, panel_ws);
    }

    return size;
}

template <Backend B, typename T>
Event sytrd_sy2sb(Queue& ctx,
                  const MatrixView<T, MatrixFormat::Dense>& a_in,
                  const MatrixView<T, MatrixFormat::Dense>& ab_out,
                  const VectorView<T>& tau_out,
                  Uplo uplo,
                  int32_t kd,
                  const Span<std::byte>& ws) {
    validate_sytrd_sy2sb_dims(a_in, ab_out, tau_out, uplo, kd);

    if (!ctx.in_order()) {
        throw batchlas::invalid_argument("sytrd_sy2sb: requires an in-order Queue");
    }

    if (uplo != Uplo::Lower) {
        throw batchlas::unsupported("sytrd_sy2sb: only Uplo::Lower is implemented");
    }

    const int n = a_in.rows();
    const int batch = a_in.batch_size();
    const int kd_i = std::max<int>(0, kd);

    // Quick return: just copy the band.
    if (n <= 0) return ctx.get_event();

    (void)zero_ab<T>(ctx, ab_out);

    if (kd_i == 0 || n <= kd_i) {
        // Full band width covers the matrix (or degenerate kd=0): copy diagonal (and up to kd).
        (void)copy_band_lower<T>(ctx, a_in, ab_out, /*i0=*/0, /*pk=*/n, /*kd=*/kd_i);
        return ctx.get_event();
    }

    BumpAllocator pool(ws);

    // tau panel storage packed with per-batch stride = PK (varies by panel); we reuse this buffer.
    auto tau_panel_buf = pool.allocate<T>(ctx, static_cast<size_t>(kd_i) * static_cast<size_t>(batch));

    // Shared workspace for GEQRF + ORMQR on the largest panel/trailing block.
    const int pn0 = n - kd_i;
    const int pk0 = std::min(pn0, kd_i);
    auto V0 = a_in({kd_i, SliceEnd()}, {0, pk0});
    // Must match the shapes queried in sytrd_sy2sb_buffer_size exactly.
    auto A_left0 = a_in({kd_i, SliceEnd()}, {pk0, SliceEnd()});
    auto A_right0 = a_in({pk0, SliceEnd()}, {kd_i, SliceEnd()});
    const size_t geqrf_ws_bytes = geqrf_buffer_size_bound<B, T>(ctx, V0, Span<T>(tau_panel_buf.data(), static_cast<size_t>(pk0) * static_cast<size_t>(batch)));
    const Transpose trans_left = internal::is_complex<T>::value ? Transpose::ConjTrans : Transpose::Trans;
    // Same hint used for the query and for every ormqr call in the loop below.
    // Keep this identical to sytrd_sy2sb_buffer_size or the pool overruns.
    const int32_t ormqr_nb_hint = sy2sb_ormqr_block_size_hint(n, batch, kd_i);
    const size_t ormqr_l_ws_bytes = ormqr_buffer_size<B, T>(ctx, V0, A_left0, Side::Left, trans_left,
                                                           Span<T>(tau_panel_buf.data(), static_cast<size_t>(pk0) * static_cast<size_t>(batch)),
                                                           ormqr_nb_hint);
    const size_t ormqr_r_ws_bytes = ormqr_buffer_size<B, T>(ctx, V0, A_right0, Side::Right, Transpose::NoTrans,
                                                           Span<T>(tau_panel_buf.data(), static_cast<size_t>(pk0) * static_cast<size_t>(batch)),
                                                           ormqr_nb_hint);
    const size_t panel_ws_bytes = std::max(geqrf_ws_bytes, std::max(ormqr_l_ws_bytes, ormqr_r_ws_bytes));
    auto panel_ws = pool.allocate<std::byte>(ctx, panel_ws_bytes);
    
    
    // Main loop: i advances in blocks of kd.
    for (int i = 0; i <= n - kd_i - 1; i += kd_i) {
        const int pn = n - i - kd_i;
        if (pn <= 0) break;
        const int pk = std::min(pn, kd_i);

        auto V = a_in({i + kd_i, SliceEnd()}, {i, i + pk});           // (pn x pk)

        // The similarity is A := H^H A H with H = diag(I_{i+kd}, Q), so Q^H must hit
        // EVERY column left of the trailing block, not just A22. When pk < kd (final
        // panel), columns [i+pk, i+kd) are neither zero nor in the panel; starting the
        // left apply at column i+pk (and the right apply at row i+pk) covers them.
        // Columns < i are already zero below row i+kd and [i, i+pk) become R inside
        // geqrf, so at pk == kd both reduce to the A22-only apply. The leftover columns
        // are the tail [n-kd, n), which the copy after the loop picks up.
        // Skipping them corrupts the band, visible only at n % kd >= 2 (at 1 the
        // leftover Q is 1x1 with tau = 0).
        auto A_left = a_in({i + kd_i, SliceEnd()}, {i + pk, SliceEnd()});
        auto A_right = a_in({i + pk, SliceEnd()}, {i + kd_i, SliceEnd()});

        // tau panel span is packed with per-batch stride = pk.
        Span<T> tau_panel_span(tau_panel_buf.data(), static_cast<size_t>(pk) * static_cast<size_t>(batch));

        // QR factorization of V in-place.
        // (void) on an Event: deliberate. This Queue is in-order, so the next submission
        // is already ordered after this one and the Event carries nothing the caller needs.
        (void)geqrf<B, T>(ctx, V, tau_panel_span, panel_ws);

        // Copy band portion into AB for columns i..i+pk-1.
        (void)copy_band_lower<T>(ctx, a_in, ab_out, i, pk, kd_i);

        // A := Q^H A Q on the rows/columns Q acts on. The two ranges overlap
        // exactly on the trailing block, so it gets both sides and everything
        // else gets one.
        const Transpose trans_left_it = internal::is_complex<T>::value ? Transpose::ConjTrans : Transpose::Trans;
        // The hint is clamped to k = min(rows, cols) inside the dispatch, so on a
        // short final panel (pk < kd) it degrades to min(nb, pk) and the workspace
        // sized from the i = 0 panel still bounds it.
        (void)ormqr<B, T>(ctx, V, A_left, Side::Left, trans_left_it, tau_panel_span, panel_ws, ormqr_nb_hint);
        (void)ormqr<B, T>(ctx, V, A_right, Side::Right, Transpose::NoTrans, tau_panel_span, panel_ws, ormqr_nb_hint);

        // Store tau panel into output tau at offset i.
        (void)copy_tau_panel_to_out<T>(ctx, tau_panel_buf.data(), /*tau_panel_ld=*/pk, tau_out, i, pk);
    }

    // Copy remaining (already banded) trailing columns into AB.
    const int tail = std::max(0, n - kd_i);
    if (tail < n) {
        (void)copy_band_lower<T>(ctx, a_in, ab_out, /*i0=*/tail, /*pk=*/n - tail, /*kd=*/kd_i);
    }

    return ctx.get_event();
}

#define SYTRD_SY2SB_INSTANTIATE(back, fp) \
    template Event sytrd_sy2sb<back, BATCHLAS_UNPAREN fp>( \
        Queue&, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        const VectorView<BATCHLAS_UNPAREN fp>&, \
        Uplo, \
        int32_t, \
        const Span<std::byte>&); \
    template size_t sytrd_sy2sb_buffer_size<back, BATCHLAS_UNPAREN fp>( \
        Queue&, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        const VectorView<BATCHLAS_UNPAREN fp>&, \
        Uplo, \
        int32_t);

BATCHLAS_INSTANTIATE_SCALAR_ALL_BACKENDS(SYTRD_SY2SB_INSTANTIATE)

#undef SYTRD_SY2SB_INSTANTIATE

#undef SYTRD_SY2SB_INSTANTIATE

} // namespace batchlas
