#include <blas/device.hh>
#include <blas/functions.hh>
#include <blas/matrix.hh>
#include <internal/ormqr_blocked.hh>
#include <util/mempool.hh>
#include <batchlas/backend_config.h>

#include "../math-helpers.hh"
#include "../queue.hh"
#include "../util/template-instantiations.hh"

#include <algorithm>
#include <complex>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace batchlas {

namespace {

// The WY block factor W2 = op(T) W1 has T upper triangular -- larft builds it
// that way and zeroes the half below, which is why the GEMM spelling below is
// correct in the first place. A trmm expresses it without multiplying through
// those zeros.
//
// BATCHLAS_ORMQR_WY=gemm pins the old spelling so the substitution stays
// measurable from one binary.
inline bool use_trmm_for_wy() {
    static const bool result = []() {
        const char* v = std::getenv("BATCHLAS_ORMQR_WY");
        return !(v && std::string(v) == "gemm");
    }();
    return result;
}

// Where the tile kernel is ahead of the GEMM it replaces. Float and CUDA
// because that is the only combination with a batched triangular kernel --
// everything else would take trmm_vendor_impl, which expands the triangle into
// a scratch buffer and then runs the very GEMM being replaced, strictly worse.
// ib <= 64 because above it the tile kernel measured 0.83x-0.97x against the
// GEMM (see experiments/TRMM_SYRK_BATCHED_KERNELS.md); every block size syev
// uses is inside that anyway.
template <Backend B, typename T>
inline bool wy_trmm_applicable(int ib) {
    return B == Backend::CUDA && std::is_same_v<T, float> && ib <= 64 && use_trmm_for_wy();
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

// conj_if_needed is only used by the legacy LARFT kernel.
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

// ---------------------------------------------------------------------------
// LARFT — Form T for a block of Householder vectors V (Forward, Columnwise).
// V is (m x ib) unit-lower (diag=1, upper=0).  T is (ib x ib) upper triangular.
// ---------------------------------------------------------------------------

// Kernel name tags — legacy path uses the "Legacy" variants so both sets can
// coexist in the same translation unit without SYCL kernel-name collisions.
template <typename T> class LarftKernelLegacy32;
template <typename T> class LarftKernelLegacy64;
template <typename T> class LarftKernelLegacy128;
template <typename T> class LarftKernelLegacy256;

template <typename T> class LarftKernel32;
template <typename T> class LarftKernel64;
template <typename T> class LarftKernel128;
template <typename T> class LarftKernel256;

// Legacy implementation: manual group-reduction inner loops.
template <typename T, int WG, typename KernelName>
sycl::event larft_forward_columnwise_batched_wg_legacy(Queue& q,
                                                       T* t_data,
                                                       int ld_t,
                                                       int stride_t,
                                                       const T* v_data,
                                                       int ld_v,
                                                       int stride_v,
                                                       int m,
                                                       int ib,
                                                       const T* tau_data,
                                                       int tau_stride,
                                                       int tau_offset,
                                                       int batch) {
    static_assert(WG > 0, "WG must be positive");

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

    return q->submit([&](sycl::handler& h) {
        h.parallel_for<KernelName>(
            sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(batch) * static_cast<size_t>(WG)),
                              sycl::range<1>(static_cast<size_t>(WG))),
            [=](sycl::nd_item<1> it) {
                const int b = static_cast<int>(it.get_group_linear_id());
                if (b >= batch) return;

                T* t_b = t_data + b * stride_t;
                const T* v_b = v_data + b * stride_v;
                const T* tau_b = tau_data + b * tau_stride + tau_offset;

                const sycl::group<1> g = it.get_group();
                const int lid = static_cast<int>(it.get_local_linear_id());

                if (lid == 0) {
                    for (int j = 0; j < ib; ++j) {
                        for (int i = 0; i < ib; ++i) {
                            t_b[i + j * ld_t] = T(0);
                        }
                    }
                }
                it.barrier(sycl::access::fence_space::global_space);

                for (int j = 0; j < ib; ++j) {
                    const T tauj = tau_b[j];
                    if (tauj == T(0)) {
                        if (lid == 0) {
                            t_b[j + j * ld_t] = T(0);
                        }
                        it.barrier(sycl::access::fence_space::global_space);
                        continue;
                    }

                    for (int col = 0; col < j; ++col) {
                        T partial = T(0);
                        for (int r = j + 1 + lid; r < m; r += WG) {
                            const T v_rc = v_b[r + col * ld_v];
                            const T v_rj = v_b[r + j * ld_v];
                            partial += conj_if_needed(v_rc, /*do_conj=*/true) * v_rj;
                        }
                        const T sum_r = reduce_sum(g, partial);
                        if (lid == 0) {
                            const T sum = conj_if_needed(v_b[j + col * ld_v], /*do_conj=*/true) + sum_r;
                            t_b[col + j * ld_t] = -tauj * sum;
                        }
                        it.barrier(sycl::access::fence_space::global_space);
                    }

                    if (lid == 0) {
                        for (int row = 0; row < j; ++row) {
                            T acc = T(0);
                            for (int col = row; col < j; ++col) {
                                acc += t_b[row + col * ld_t] * t_b[col + j * ld_t];
                            }
                            t_b[row + j * ld_t] = acc;
                        }
                        t_b[j + j * ld_t] = tauj;
                    }
                    it.barrier(sycl::access::fence_space::global_space);
                }
            });
    });
}

// Device-BLAS implementation: uses device::fill / gemv / scal / trmv.
template <typename T, int WG, typename KernelName>
sycl::event larft_forward_columnwise_batched_wg_device(Queue& q,
                                                       T* t_data,
                                                       int ld_t,
                                                       int stride_t,
                                                       const T* v_data,
                                                       int ld_v,
                                                       int stride_v,
                                                       int m,
                                                       int ib,
                                                       const T* tau_data,
                                                       int tau_stride,
                                                       int tau_offset,
                                                       int batch) {
    static_assert(WG > 0, "WG must be positive");

    return q->submit([&](sycl::handler& h) {
        h.parallel_for<KernelName>(
            sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(batch) * static_cast<size_t>(WG)),
                              sycl::range<1>(static_cast<size_t>(WG))),
            [=](sycl::nd_item<1> it) {
                const int b = static_cast<int>(it.get_group_linear_id());
                if (b >= batch) return;

                T* t_b = t_data + b * stride_t;
                const T* v_b = v_data + b * stride_v;
                const T* tau_b = tau_data + b * tau_stride + tau_offset;

                const sycl::group<1> g = it.get_group();
                const int lid = static_cast<int>(it.get_local_linear_id());
                auto t_mat = KernelMatrixView<T, MatrixFormat::Dense>(t_b, ib, ib, ld_t, ld_t * ib);
                auto v_mat = KernelMatrixView<T, MatrixFormat::Dense>(const_cast<T*>(v_b), m, ib, ld_v, ld_v * ib);

                batchlas::device::fill(g, t_mat, T(0));
                sycl::group_barrier(g);

                for (int j = 0; j < ib; ++j) {
                    const T tauj = tau_b[j];
                    if (tauj == T(0)) {
                        if (lid == 0) {
                            t_b[j + j * ld_t] = T(0);
                        }
                        sycl::group_barrier(g);
                        continue;
                    }

                    if (j > 0) {
                        auto t_col  = t_mat(Slice(0, j), j);
                        auto t_prev = t_mat(Slice(0, j), Slice(0, j));
                        auto v_prev = v_mat(Slice(j, m), Slice(0, j));
                        auto v_col  = v_mat(Slice(j, m), j);

                        batchlas::device::gemv<Transpose::ConjTrans>(g, v_prev, v_col, t_col, T(1), T(0), static_cast<T*>(nullptr));
                        sycl::group_barrier(g);
                        batchlas::device::scal(g, t_col, -tauj);
                        sycl::group_barrier(g);
                        batchlas::device::trmv<Uplo::Upper, Transpose::NoTrans, Diag::NonUnit>(
                            g, t_prev, t_col, t_col, T(1), T(0));
                        sycl::group_barrier(g);
                    }

                    if (lid == 0) {
                        t_b[j + j * ld_t] = tauj;
                    }
                    sycl::group_barrier(g);
                }
            });
    });
}

// Dispatcher: selects legacy or device LARFT based on use_device_ormqr().
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
                                             int tau_stride,
                                             int tau_offset,
                                             int batch) {
    const bool use_device = use_device_ormqr();

    // WG size selection is the same for both paths; kernel name type differs.
    if (ib <= 8 && m <= 64) {
        if (use_device)
            return larft_forward_columnwise_batched_wg_device<T, 32, LarftKernel32<T>>(
                q, t_data, ld_t, stride_t, v_data, ld_v, stride_v, m, ib, tau_data, tau_stride, tau_offset, batch);
        else
            return larft_forward_columnwise_batched_wg_legacy<T, 32, LarftKernelLegacy32<T>>(
                q, t_data, ld_t, stride_t, v_data, ld_v, stride_v, m, ib, tau_data, tau_stride, tau_offset, batch);
    }
    if (ib <= 16 && m <= 128) {
        if (use_device)
            return larft_forward_columnwise_batched_wg_device<T, 64, LarftKernel64<T>>(
                q, t_data, ld_t, stride_t, v_data, ld_v, stride_v, m, ib, tau_data, tau_stride, tau_offset, batch);
        else
            return larft_forward_columnwise_batched_wg_legacy<T, 64, LarftKernelLegacy64<T>>(
                q, t_data, ld_t, stride_t, v_data, ld_v, stride_v, m, ib, tau_data, tau_stride, tau_offset, batch);
    }
    if (ib <= 32 && m <= 256) {
        if (use_device)
            return larft_forward_columnwise_batched_wg_device<T, 128, LarftKernel128<T>>(
                q, t_data, ld_t, stride_t, v_data, ld_v, stride_v, m, ib, tau_data, tau_stride, tau_offset, batch);
        else
            return larft_forward_columnwise_batched_wg_legacy<T, 128, LarftKernelLegacy128<T>>(
                q, t_data, ld_t, stride_t, v_data, ld_v, stride_v, m, ib, tau_data, tau_stride, tau_offset, batch);
    }
    if (use_device)
        return larft_forward_columnwise_batched_wg_device<T, 256, LarftKernel256<T>>(
            q, t_data, ld_t, stride_t, v_data, ld_v, stride_v, m, ib, tau_data, tau_stride, tau_offset, batch);
    else
        return larft_forward_columnwise_batched_wg_legacy<T, 256, LarftKernelLegacy256<T>>(
            q, t_data, ld_t, stride_t, v_data, ld_v, stride_v, m, ib, tau_data, tau_stride, tau_offset, batch);
}

template <typename T>
class PackVKernel;

template <typename T>
sycl::event pack_v_panel_batched(Queue& q,
                                 T* v_out,
                                 int ld_v_out,
                                 int stride_v_out,
                                 const MatrixView<T, MatrixFormat::Dense>& a,
                                 int i0,
                                 int ib,
                                 int nq) {
    const int m = nq - i0;
    const int ld_a = a.ld();
    const int stride_a = a.stride();
    const T* a_ptr = a.data_ptr();
    const int batch = a.batch_size();

    return q->submit([&](sycl::handler& h) {
        h.parallel_for<PackVKernel<T>>(sycl::range<3>(static_cast<size_t>(batch),
                                                      static_cast<size_t>(m),
                                                      static_cast<size_t>(ib)),
                                       [=](sycl::id<3> idx) {
                                           const int b = static_cast<int>(idx[0]);
                                           const int r = static_cast<int>(idx[1]);
                                           const int c = static_cast<int>(idx[2]);
                                           T val = T(0);
                                           if (r == c) {
                                               val = T(1);
                                           } else if (r > c) {
                                               val = a_ptr[b * stride_a + (i0 + r) + (i0 + c) * ld_a];
                                           }
                                           v_out[b * stride_v_out + r + c * ld_v_out] = val;
                                       });
    });
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
            (void)pack_v_panel_batched<T>(q,
                                          Vmat.data_ptr(), Vmat.ld(), Vmat.stride(),
                                          a, i0, ib, nq);
        }

        {
            BATCHLAS_KERNEL_TRACE_SCOPE("ormqr_blocked.larft");
            (void)larft_forward_columnwise_batched<T>(q,
                                                      Tmat.data_ptr(), Tmat.ld(), Tmat.stride(),
                                                      Vmat.data_ptr(), Vmat.ld(), Vmat.stride(),
                                                      m, ib,
                                                      tau.data(), /*tau_stride=*/k, /*tau_offset=*/i0,
                                                      batch);
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
            // W2 = op(T) W1. trmm overwrites C on the routes that have a
            // batched kernel, matching this GEMM's beta = 0, so the two are
            // interchangeable here.
            if constexpr (B == Backend::CUDA && std::is_same_v<T, float>) {
                if (wy_trmm_applicable<B, T>(ib)) {
                    trmm<B, T>(q, Tblk, W1, W2, T(1),
                               Side::Left, Uplo::Upper, t_eff, Diag::NonUnit);
                } else {
                    gemm<B>(q, Tblk, W1, W2, {.transA = t_eff});
                }
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
