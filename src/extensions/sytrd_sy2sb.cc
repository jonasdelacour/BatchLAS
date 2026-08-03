#include <blas/extensions.hh>
#include <blas/functions.hh>
#include <blas/matrix.hh>
#include <util/mempool.hh>

#include <sycl/sycl.hpp>

#include <batchlas/backend_config.h>

#include "../math-helpers.hh"
#include "../queue.hh"
#include "../util/template-instantiations.hh"

#include <algorithm>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <type_traits>

namespace batchlas {

namespace {

// ---------------------------------------------------------------------------
// WY block width for the panel back-transform.
//
// sy2sb factors a panel of width kd (default 32) and then calls ormqr on it.
// ormqr's dispatch picks its WY block width from tuning::ormqr_block_size_for_n
// keyed on A.rows() -- the panel *height* (hundreds to thousands) -- while the
// dimension that actually matters is k = kd. Every ORMQR_BLOCK_SIZE_* constant is
// 16, so a kd=32 panel is split into two WY blocks: 2x pack_v, 2x larft and 6
// GEMMs per ormqr instead of 1, 1 and 3, with every GEMM at k=16 instead of 32.
//
// Using nb = kd was measured (prior probe, interleaved A/B, median of 15 rounds,
// idle GPU) on the sy2sb panel loop:
//
//     n=1024 kd=32 batch=64  : 1.19-1.20x faster
//     n=2048 kd=32 batch=32  : 1.36x   faster
//     n=512  kd=32 batch=128 : 0.90x   (regression)
//     n=1024 kd=32 batch=8   : 0.67x   (large regression)
//
// The win comes from GEMM k-depth, not from the lower launch count; LARFT work is
// O(m*k*nb) and doubles with nb, which dominates once the GEMMs are too small to
// benefit. So this is gated to the region where the win was measured: n >= 1024
// and batch >= 32. Outside it we return 0, i.e. exactly today's behaviour.
//
// Override with BATCHLAS_SY2SB_ORMQR_NB:
//   unset          -> shape gate below (default)
//   0 / "off"      -> never hint; restores the pre-change tuning-table behaviour
//   <positive int> -> force that block width unconditionally
//
// Read fresh on every call (like policy_from_env) so an A/B harness can flip it
// inside one process. It must NOT be changed between a sytrd_sy2sb_buffer_size
// query and the matching sytrd_sy2sb call -- that would desynchronise the
// workspace size from the block width actually used.
inline int32_t sy2sb_ormqr_nb_env(bool& has_override) {
    const char* v = std::getenv("BATCHLAS_SY2SB_ORMQR_NB");
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
    // Shape gate: only where the win was measured.
    if (n >= 1024 && batch >= 32) return kd;
    return 0;
}

template <typename U>
inline U conj_if_needed(const U& x, bool do_conj) {
    if (!do_conj) return x;
    if constexpr (internal::is_complex<U>::value) {
        return U(x.real(), -x.imag());
    } else {
        return x;
    }
}

template <typename T>
inline void validate_sytrd_sy2sb_dims(const MatrixView<T, MatrixFormat::Dense>& a,
                                     const MatrixView<T, MatrixFormat::Dense>& ab,
                                     const VectorView<T>& tau,
                                     Uplo uplo,
                                     int32_t kd) {
    if (a.rows() != a.cols()) {
        throw std::invalid_argument("sytrd_sy2sb: A must be square");
    }
    if (kd < 0) {
        throw std::invalid_argument("sytrd_sy2sb: kd must be non-negative");
    }
    if (uplo != Uplo::Lower && uplo != Uplo::Upper) {
        throw std::invalid_argument("sytrd_sy2sb: invalid uplo");
    }

    const int n = a.rows();
    const int kd_i = kd;
    const int tau_need = std::max(0, n - kd_i);

    if (ab.rows() != kd_i + 1 || ab.cols() != n) {
        throw std::invalid_argument("sytrd_sy2sb: AB must be (kd+1) x n");
    }
    if (tau.size() != tau_need) {
        throw std::invalid_argument("sytrd_sy2sb: tau must have size (n-kd)");
    }
    if (a.batch_size() != ab.batch_size() || a.batch_size() != tau.batch_size()) {
        throw std::invalid_argument("sytrd_sy2sb: batch size mismatch");
    }
    if (a.batch_size() < 1) {
        throw std::invalid_argument("sytrd_sy2sb: invalid batch size");
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
class SetUnitLowerPanelKernel;

template <typename T>
Event set_unit_lower_panel(Queue& q,
                           const MatrixView<T, MatrixFormat::Dense>& v,
                           int pk) {
    const int ldv = v.ld();
    const int stride_v = v.stride();
    T* v_ptr = v.data_ptr();
    const int batch = v.batch_size();

    (void)q->submit([&](sycl::handler& h) {
        h.parallel_for<SetUnitLowerPanelKernel<T>>(
            sycl::range<3>(static_cast<size_t>(batch), static_cast<size_t>(pk), static_cast<size_t>(pk)),
            [=](sycl::id<3> idx) {
                const int b = static_cast<int>(idx[0]);
                const int r = static_cast<int>(idx[1]);
                const int c = static_cast<int>(idx[2]);
                T* V = v_ptr + b * stride_v;
                if (r <= c) {
                    V[r + c * ldv] = (r == c) ? T(1) : T(0);
                }
            });
    });

    return q.get_event();
}

// Form T for a block of Householder vectors V (Forward, Columnwise), like LAPACK LARFT.
//
// V is (m x ib) unit-lower (diag=1, upper=0). T is (ib x ib) upper triangular.
//
// tau is packed by-panel: tau[b*ib + j].
template <typename T>
class LarftKernel;

template <typename T>
sycl::event larft_forward_columnwise_batched(Queue& q,
                                            T* t_data,
                                            int ld_t,
                                            int stride_t,
                                            const T* v_data,
                                            int ld_v,
                                            int stride_v,
                                            int m,
                                            int ib,
                                            const T* tau_data,
                                            int tau_ld,
                                            int batch) {
    auto reduce_sum = [](const sycl::group<1>& g, T x) {
        if constexpr (internal::is_complex<T>::value) {
            using R = typename T::value_type;
            const R re = sycl::reduce_over_group(g, x.real(), sycl::plus<R>());
            const R im = sycl::reduce_over_group(g, x.imag(), sycl::plus<R>());
            return T(re, im);
        } else {
            return sycl::reduce_over_group(g, x, sycl::plus<T>());
        }
    };

    const size_t wg = 256;
    const size_t groups = static_cast<size_t>(batch) * static_cast<size_t>(ib);

    return q->submit([&](sycl::handler& h) {
        h.parallel_for<LarftKernel<T>>(
            sycl::nd_range<1>(sycl::range<1>(groups * wg), sycl::range<1>(wg)),
            [=](sycl::nd_item<1> it) {
                const size_t gid = it.get_group_linear_id();
                const int b = static_cast<int>(gid / static_cast<size_t>(ib));
                const int j = static_cast<int>(gid - static_cast<size_t>(b) * static_cast<size_t>(ib));
                if (b >= batch || j >= ib) return;

                T* t_b = t_data + b * stride_t;
                const T* v_b = v_data + b * stride_v;
                const T* tau_b = tau_data + b * tau_ld;

                const T tauj = tau_b[j];

                if (it.get_local_linear_id() == 0) {
                    for (int i = 0; i < ib; ++i) {
                        t_b[i + j * ld_t] = T(0);
                    }
                }
                it.barrier(sycl::access::fence_space::local_space);

                if (tauj == T(0)) {
                    if (it.get_local_linear_id() == 0) {
                        t_b[j + j * ld_t] = T(0);
                    }
                    return;
                }

                const sycl::group<1> g = it.get_group();

                for (int col = 0; col < j; ++col) {
                    T partial = T(0);
                    for (int r = j + 1 + static_cast<int>(it.get_local_linear_id()); r < m;
                         r += static_cast<int>(wg)) {
                        const T v_rc = v_b[r + col * ld_v];
                        const T v_rj = v_b[r + j * ld_v];
                        partial += conj_if_needed(v_rc, /*do_conj=*/true) * v_rj;
                    }

                    const T sum_r = reduce_sum(g, partial);
                    if (it.get_local_linear_id() == 0) {
                        T sum = conj_if_needed(v_b[j + col * ld_v], /*do_conj=*/true) + sum_r;
                        t_b[col + j * ld_t] = -tauj * sum;
                    }
                    it.barrier(sycl::access::fence_space::global_space);
                }

                if (it.get_local_linear_id() == 0) {
                    for (int row = 0; row < j; ++row) {
                        T acc = T(0);
                        for (int col = row; col < j; ++col) {
                            acc += t_b[row + col * ld_t] * t_b[col + j * ld_t];
                        }
                        t_b[row + j * ld_t] = acc;
                    }
                    t_b[j + j * ld_t] = tauj;
                }
            });
    });
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

        const size_t geqrf_ws = geqrf_buffer_size<B, T>(ctx, V0, tau_span);
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
        throw std::runtime_error("sytrd_sy2sb: requires an in-order Queue");
    }

    if (uplo != Uplo::Lower) {
        throw std::runtime_error("sytrd_sy2sb: only Uplo::Lower is implemented");
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
    const size_t geqrf_ws_bytes = geqrf_buffer_size<B, T>(ctx, V0, Span<T>(tau_panel_buf.data(), static_cast<size_t>(pk0) * static_cast<size_t>(batch)));
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

        // The similarity is A := H^H A H with H = diag(I_{i+kd}, Q), so Q^H
        // must hit *every* column left of the trailing block as well, not just
        // A22. Columns < i are already zero below row i+kd (they were banded by
        // earlier panels) and columns [i, i+pk) become R inside geqrf, so when
        // pk == kd the trailing block is genuinely all that is left.
        //
        // On the final panel pk = min(pn, kd) can be < kd, and then columns
        // [i+pk, i+kd) are neither zero nor part of the panel -- they used to be
        // skipped entirely, silently corrupting the band. Widening the left
        // apply to start at column i+pk (and the right apply, its transpose, to
        // start at row i+pk) covers them; both reduce to the old A22-only calls
        // when pk == kd. Those leftover columns are exactly the tail columns
        // [n-kd, n), which the copy after the loop picks up afterwards.
        //
        // This is why the failure needed n % kd >= 2: at n % kd == 1 the
        // leftover Q is 1x1 with tau = 0, i.e. the identity.
        auto A_left = a_in({i + kd_i, SliceEnd()}, {i + pk, SliceEnd()});
        auto A_right = a_in({i + pk, SliceEnd()}, {i + kd_i, SliceEnd()});

        // tau panel span is packed with per-batch stride = pk.
        Span<T> tau_panel_span(tau_panel_buf.data(), static_cast<size_t>(pk) * static_cast<size_t>(batch));

        // QR factorization of V in-place.
        geqrf<B, T>(ctx, V, tau_panel_span, panel_ws);

        // Copy band portion into AB for columns i..i+pk-1.
        (void)copy_band_lower<T>(ctx, a_in, ab_out, i, pk, kd_i);

        // A := Q^H A Q on the rows/columns Q acts on. The two ranges overlap
        // exactly on the trailing block, so it gets both sides and everything
        // else gets one.
        const Transpose trans_left_it = internal::is_complex<T>::value ? Transpose::ConjTrans : Transpose::Trans;
        // The hint is clamped to k = min(rows, cols) inside the dispatch, so on a
        // short final panel (pk < kd) it degrades to min(nb, pk) and the workspace
        // sized from the i = 0 panel still bounds it.
        ormqr<B, T>(ctx, V, A_left, Side::Left, trans_left_it, tau_panel_span, panel_ws, ormqr_nb_hint);
        ormqr<B, T>(ctx, V, A_right, Side::Right, Transpose::NoTrans, tau_panel_span, panel_ws, ormqr_nb_hint);

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

#define SYTRD_SY2SB_INSTANTIATE_FOR_BACKEND(back) \
    BATCHLAS_FOR_EACH_SCALAR_TYPE_1(SYTRD_SY2SB_INSTANTIATE, back)

#if BATCHLAS_HAS_CUDA_BACKEND
SYTRD_SY2SB_INSTANTIATE_FOR_BACKEND(Backend::CUDA)
#endif

#if BATCHLAS_HAS_ROCM_BACKEND
SYTRD_SY2SB_INSTANTIATE_FOR_BACKEND(Backend::ROCM)
#endif

#if BATCHLAS_HAS_HOST_BACKEND
SYTRD_SY2SB_INSTANTIATE_FOR_BACKEND(Backend::NETLIB)
#endif

#undef SYTRD_SY2SB_INSTANTIATE_FOR_BACKEND
#undef SYTRD_SY2SB_INSTANTIATE

#undef SYTRD_SY2SB_INSTANTIATE

} // namespace batchlas
