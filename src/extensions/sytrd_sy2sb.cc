#include <blas/extensions.hh>
#include <blas/functions.hh>
#include <blas/matrix.hh>
#include <util/mempool.hh>

#include <sycl/sycl.hpp>

#include <batchlas/backend_config.h>

#include "../math-helpers.hh"
#include "../queue.hh"

#include <algorithm>
#include <complex>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace batchlas {

namespace {

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
        auto A22_0 = a_in({kd_i, SliceEnd()}, {kd_i, SliceEnd()});

        const size_t tau_elems = static_cast<size_t>(pk0) * static_cast<size_t>(batch);
        T* tau_tmp = sycl::malloc_shared<T>(tau_elems, ctx->get_device(), ctx->get_context());
        if (!tau_tmp && tau_elems != 0) {
            throw std::bad_alloc();
        }
        Span<T> tau_span(tau_tmp, tau_elems);

        const size_t geqrf_ws = geqrf_buffer_size<B, T>(ctx, V0, tau_span);
        const Transpose trans_left = internal::is_complex<T>::value ? Transpose::ConjTrans : Transpose::Trans;
        const size_t ormqr_l_ws = ormqr_buffer_size<B, T>(ctx, V0, A22_0, Side::Left, trans_left, tau_span);
        const size_t ormqr_r_ws = ormqr_buffer_size<B, T>(ctx, V0, A22_0, Side::Right, Transpose::NoTrans, tau_span);
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
    auto A22_0 = a_in({kd_i, SliceEnd()}, {kd_i, SliceEnd()});
    const size_t geqrf_ws_bytes = geqrf_buffer_size<B, T>(ctx, V0, Span<T>(tau_panel_buf.data(), static_cast<size_t>(pk0) * static_cast<size_t>(batch)));
    const Transpose trans_left = internal::is_complex<T>::value ? Transpose::ConjTrans : Transpose::Trans;
    const size_t ormqr_l_ws_bytes = ormqr_buffer_size<B, T>(ctx, V0, A22_0, Side::Left, trans_left,
                                                           Span<T>(tau_panel_buf.data(), static_cast<size_t>(pk0) * static_cast<size_t>(batch)));
    const size_t ormqr_r_ws_bytes = ormqr_buffer_size<B, T>(ctx, V0, A22_0, Side::Right, Transpose::NoTrans,
                                                           Span<T>(tau_panel_buf.data(), static_cast<size_t>(pk0) * static_cast<size_t>(batch)));
    const size_t panel_ws_bytes = std::max(geqrf_ws_bytes, std::max(ormqr_l_ws_bytes, ormqr_r_ws_bytes));
    auto panel_ws = pool.allocate<std::byte>(ctx, panel_ws_bytes);
    
    
    // Main loop: i advances in blocks of kd.
    for (int i = 0; i <= n - kd_i - 1; i += kd_i) {
        const int pn = n - i - kd_i;
        if (pn <= 0) break;
        const int pk = std::min(pn, kd_i);

        auto V = a_in({i + kd_i, SliceEnd()}, {i, i + pk});           // (pn x pk)
        auto A22 = a_in({i + kd_i, SliceEnd()}, {i + kd_i, SliceEnd()}); // (pn x pn)

        // tau panel span is packed with per-batch stride = pk.
        Span<T> tau_panel_span(tau_panel_buf.data(), static_cast<size_t>(pk) * static_cast<size_t>(batch));

        // QR factorization of V in-place.
        geqrf<B, T>(ctx, V, tau_panel_span, panel_ws);

        // Copy band portion into AB for columns i..i+pk-1.
        (void)copy_band_lower<T>(ctx, a_in, ab_out, i, pk, kd_i);

        // Apply the orthogonal/unitary transform to the trailing block:
        //   A22 := Q^H * A22 * Q, where Q is defined by GEQRF(V).
        const Transpose trans_left_it = internal::is_complex<T>::value ? Transpose::ConjTrans : Transpose::Trans;
        ormqr<B, T>(ctx, V, A22, Side::Left, trans_left_it, tau_panel_span, panel_ws);
        ormqr<B, T>(ctx, V, A22, Side::Right, Transpose::NoTrans, tau_panel_span, panel_ws);

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
    template Event sytrd_sy2sb<back, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        const VectorView<fp>&, \
        Uplo, \
        int32_t, \
        const Span<std::byte>&); \
    template size_t sytrd_sy2sb_buffer_size<back, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        const VectorView<fp>&, \
        Uplo, \
        int32_t);

#if BATCHLAS_HAS_CUDA_BACKEND
SYTRD_SY2SB_INSTANTIATE(Backend::CUDA, float)
SYTRD_SY2SB_INSTANTIATE(Backend::CUDA, double)
SYTRD_SY2SB_INSTANTIATE(Backend::CUDA, std::complex<float>)
SYTRD_SY2SB_INSTANTIATE(Backend::CUDA, std::complex<double>)
#endif

#if BATCHLAS_HAS_ROCM_BACKEND
SYTRD_SY2SB_INSTANTIATE(Backend::ROCM, float)
SYTRD_SY2SB_INSTANTIATE(Backend::ROCM, double)
SYTRD_SY2SB_INSTANTIATE(Backend::ROCM, std::complex<float>)
SYTRD_SY2SB_INSTANTIATE(Backend::ROCM, std::complex<double>)
#endif

#if BATCHLAS_HAS_HOST_BACKEND
SYTRD_SY2SB_INSTANTIATE(Backend::NETLIB, float)
SYTRD_SY2SB_INSTANTIATE(Backend::NETLIB, double)
SYTRD_SY2SB_INSTANTIATE(Backend::NETLIB, std::complex<float>)
SYTRD_SY2SB_INSTANTIATE(Backend::NETLIB, std::complex<double>)
#endif

#undef SYTRD_SY2SB_INSTANTIATE

} // namespace batchlas
