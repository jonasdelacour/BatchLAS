#include <blas/functions.hh>
#include <blas/matrix.hh>
#include <internal/ormqr_blocked.hh>
#include <util/mempool.hh>
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

// Form T for a block of Householder vectors V (Forward, Columnwise) like LAPACK LARFT.
//
// V is (m x ib) unit-lower (diag=1, upper=0). T is (ib x ib) upper triangular.
//
// This is a straightforward implementation intended primarily as a building block.
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
                                            int tau_stride,
                                            int tau_offset,
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

    // One work-group per (batch, j) computing the whole column j of T.
    // Parallelize the dot-products over r=j+1..m-1.
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
                const T* tau_b = tau_data + b * tau_stride + tau_offset;

                const T tauj = tau_b[j];

                // Clear column j.
                // This loop is small; let one lane do it.
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

                // Compute t(0:j-1,j) = -tauj * V(j:m-1,0:j-1)^H * v_j(j:m-1)
                // (v_j has implicit 1 at position j; V is already packed with diag=1).
                for (int col = 0; col < j; ++col) {
                    T partial = T(0);
                    // r starts at j+1; r==j handled by leader with conj(V(j,col))*1.
                    for (int r = j + 1 + static_cast<int>(it.get_local_linear_id()); r < m; r += static_cast<int>(wg)) {
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

                // t(0:j-1,j) = T(0:j-1,0:j-1) * t(0:j-1,j)
                // Upper-triangular mat-vec; small, so do it serially.
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

template <Backend B, typename T>
size_t ormqr_blocked_buffer_size(Queue& ctx,
                                 const MatrixView<T, MatrixFormat::Dense>& a,
                                 const MatrixView<T, MatrixFormat::Dense>& c,
                                 Side side,
                                 Transpose trans,
                                 Span<T> tau,
                                 int32_t block_size) {
    (void)trans;
    validate_ormqr_dims(a, c, side, tau);

    const int nq = a.rows();
    const int m = c.rows();
    const int n = c.cols();
    const int batch = a.batch_size();

    const int nb = std::max<int>(1, block_size);

    size_t size = 0;
    size += BumpAllocator::allocation_size<T>(ctx, static_cast<size_t>(nq) * static_cast<size_t>(nb) * static_cast<size_t>(batch)); // packed V
    size += BumpAllocator::allocation_size<T>(ctx, static_cast<size_t>(nb) * static_cast<size_t>(nb) * static_cast<size_t>(batch)); // T

    const size_t w_elems = (side == Side::Left)
                               ? static_cast<size_t>(nb) * static_cast<size_t>(n)
                               : static_cast<size_t>(m) * static_cast<size_t>(nb);
    size += 2 * BumpAllocator::allocation_size<T>(ctx, w_elems * static_cast<size_t>(batch)); // W1 + W2

    return size;
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

    const int nq = a.rows();
    const int mC = c.rows();
    const int nC = c.cols();
    const int k = std::min(a.rows(), a.cols());
    const int batch = a.batch_size();

    const int nb = std::max<int>(1, block_size);

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

        // Pack V panel for this block into Vmat(0:nq-i0, 0:ib).
        const int m = nq - i0;
        {
            BATCHLAS_KERNEL_TRACE_SCOPE("ormqr_blocked.pack_v_panel");
            (void)pack_v_panel_batched<T>(q,
                                         Vmat.data_ptr(), Vmat.ld(), Vmat.stride(),
                                         a, i0, ib, nq);
        }

        // Form T for this block into Tmat(0:ib,0:ib) for all batch items.
        {
            BATCHLAS_KERNEL_TRACE_SCOPE("ormqr_blocked.larft");
            (void)larft_forward_columnwise_batched<T>(q,
                                                     Tmat.data_ptr(), Tmat.ld(), Tmat.stride(),
                                                     Vmat.data_ptr(), Vmat.ld(), Vmat.stride(),
                                                     m, ib,
                                                     tau.data(), /*tau_stride=*/k, /*tau_offset=*/i0,
                                                     batch);
        }

        // Apply the block reflector to C.
        if (side == Side::Left) {
            auto Csub = c({i0, SliceEnd()}, Slice());
            auto Vblk = Vmat({0, m}, {0, ib});
            auto Tblk = Tmat({0, ib}, {0, ib});

            // W1 = V^H * Csub
            MatrixView<T, MatrixFormat::Dense> W1full(W1buf.data(), nb, nC, nb, nb * nC, batch);
            MatrixView<T, MatrixFormat::Dense> W2full(W2buf.data(), nb, nC, nb, nb * nC, batch);
            auto W1 = W1full({0, ib}, Slice());
            auto W2 = W2full({0, ib}, Slice());

            gemm<B>(q, Vblk, Csub, W1, T(1), T(0), Transpose::ConjTrans, Transpose::NoTrans);

            // W2 = T_eff * W1
            const Transpose t_eff = transpose_apply ? Transpose::ConjTrans : Transpose::NoTrans;
            gemm<B>(q, Tblk, W1, W2, T(1), T(0), t_eff, Transpose::NoTrans);

            // Csub -= V * W2
            gemm<B>(q, Vblk, W2, Csub, T(-1), T(1), Transpose::NoTrans, Transpose::NoTrans);
        } else {
            auto Csub = c(Slice(), {i0, SliceEnd()});
            auto Vblk = Vmat({0, m}, {0, ib});
            auto Tblk = Tmat({0, ib}, {0, ib});

            MatrixView<T, MatrixFormat::Dense> W1full(W1buf.data(), mC, nb, mC, mC * nb, batch);
            MatrixView<T, MatrixFormat::Dense> W2full(W2buf.data(), mC, nb, mC, mC * nb, batch);
            auto W1 = W1full(Slice(), {0, ib});
            auto W2 = W2full(Slice(), {0, ib});

            // W1 = Csub * V
            gemm<B>(q, Csub, Vblk, W1, T(1), T(0), Transpose::NoTrans, Transpose::NoTrans);

            // W2 = W1 * T_eff
            const Transpose t_eff = transpose_apply ? Transpose::ConjTrans : Transpose::NoTrans;
            gemm<B>(q, W1, Tblk, W2, T(1), T(0), Transpose::NoTrans, t_eff);

            // Csub -= W2 * V^H
            gemm<B>(q, W2, Vblk, Csub, T(-1), T(1), Transpose::NoTrans, Transpose::ConjTrans);
        }

    };

    // Block order: forward for NoTrans, backward for Trans/ConjTrans.
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

#define ORMQR_BLOCKED_INSTANTIATE(back, fp) \
    template Event ormqr_blocked<back, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        Side, Transpose, \
        Span<fp>, \
        Span<std::byte>, \
        int32_t); \
    template size_t ormqr_blocked_buffer_size<back, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        Side, Transpose, \
        Span<fp>, \
        int32_t);

#if BATCHLAS_HAS_CUDA_BACKEND
ORMQR_BLOCKED_INSTANTIATE(Backend::CUDA, float)
ORMQR_BLOCKED_INSTANTIATE(Backend::CUDA, double)
ORMQR_BLOCKED_INSTANTIATE(Backend::CUDA, std::complex<float>)
ORMQR_BLOCKED_INSTANTIATE(Backend::CUDA, std::complex<double>)
#endif

#if BATCHLAS_HAS_ROCM_BACKEND
ORMQR_BLOCKED_INSTANTIATE(Backend::ROCM, float)
ORMQR_BLOCKED_INSTANTIATE(Backend::ROCM, double)
ORMQR_BLOCKED_INSTANTIATE(Backend::ROCM, std::complex<float>)
ORMQR_BLOCKED_INSTANTIATE(Backend::ROCM, std::complex<double>)
#endif

#if BATCHLAS_HAS_HOST_BACKEND
ORMQR_BLOCKED_INSTANTIATE(Backend::NETLIB, float)
ORMQR_BLOCKED_INSTANTIATE(Backend::NETLIB, double)
ORMQR_BLOCKED_INSTANTIATE(Backend::NETLIB, std::complex<float>)
ORMQR_BLOCKED_INSTANTIATE(Backend::NETLIB, std::complex<double>)
#endif

#undef ORMQR_BLOCKED_INSTANTIATE

} // namespace batchlas
