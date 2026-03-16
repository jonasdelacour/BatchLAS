#include <blas/extensions.hh>
#include <blas/linalg.hh>
#include <internal/ormbr.hh>
#include <internal/ormqr_blocked.hh>
#include <batchlas/backend_config.h>
#include <util/mempool.hh>

#include "../math-helpers.hh"
#include "../queue.hh"
#include "../util/template-instantiations.hh"

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
inline T conj_if_needed(const T& x) {
    if constexpr (internal::is_complex<T>::value) {
        return T(x.real(), -x.imag());
    } else {
        return x;
    }
}

template <typename T>
inline T conj_if_needed(const T& x, bool do_conj) {
    if (!do_conj) return x;
    return conj_if_needed(x);
}

template <typename T>
class OrmbrPKernel;

template <typename T>
class OrmbrPLarftKernel32;

template <typename T>
class OrmbrPLarftKernel64;

template <typename T>
class OrmbrPLarftKernel128;

template <typename T>
class OrmbrPLarftKernel256;

template <typename T>
class PackPPanelKernel;

template <Backend B, typename T>
bool ormbr_q_use_cta(const MatrixView<T, MatrixFormat::Dense>& a,
                     const MatrixView<T, MatrixFormat::Dense>& c) {
    if constexpr (B != Backend::CUDA) {
        static_cast<void>(a);
        static_cast<void>(c);
        return false;
    } else {
        return a.rows() <= 32 && c.rows() == c.cols() && c.rows() == a.rows();
    }
}

template <Backend B, typename T>
bool ormbr_p_use_cta(const MatrixView<T, MatrixFormat::Dense>& a,
                     const MatrixView<T, MatrixFormat::Dense>& c,
                     Side side) {
    if constexpr (B != Backend::CUDA) {
        static_cast<void>(a);
        static_cast<void>(c);
        static_cast<void>(side);
        return false;
    } else {
        return side == Side::Right &&
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
    const int32_t n = static_cast<int32_t>(a.rows());
    const int32_t batch = static_cast<int32_t>(a.batch_size());

    Span<std::byte> ws_mut(const_cast<std::byte*>(ws.data()), ws.size());
    BumpAllocator pool(ws_mut);

    auto aq_buf = pool.allocate<T>(ctx, static_cast<size_t>(n) * static_cast<size_t>(n) * static_cast<size_t>(batch));
    auto tauq_buf = pool.allocate<T>(ctx, static_cast<size_t>(n) * static_cast<size_t>(batch));

    MatrixView<T, MatrixFormat::Dense> aq_view(aq_buf.data(), n, n, n, static_cast<int64_t>(n) * static_cast<int64_t>(n), batch);
    VectorView<T> tauq_view(tauq_buf, n, batch, 1, n);

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

            AQ(row, col, b) = T(0);
            if (col >= 1) {
                const int32_t i = col - 1;
                if (row >= col + 1 && row < nn) {
                    AQ(row, col, b) = conj_if_needed(A(i, row, b));
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

template <typename T, int WG, typename KernelName>
sycl::event ormbr_larft_forward_columnwise_batched_wg(Queue& q,
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

template <typename T>
sycl::event ormbr_larft_forward_columnwise_batched(Queue& q,
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
    if (ib <= 8 && m <= 64) {
        return ormbr_larft_forward_columnwise_batched_wg<T, 32, OrmbrPLarftKernel32<T>>(
            q,
            t_data, ld_t, stride_t,
            v_data, ld_v, stride_v,
            m, ib,
            tau_data, tau_stride, tau_offset,
            batch);
    }
    if (ib <= 16 && m <= 128) {
        return ormbr_larft_forward_columnwise_batched_wg<T, 64, OrmbrPLarftKernel64<T>>(
            q,
            t_data, ld_t, stride_t,
            v_data, ld_v, stride_v,
            m, ib,
            tau_data, tau_stride, tau_offset,
            batch);
    }
    if (ib <= 32 && m <= 256) {
        return ormbr_larft_forward_columnwise_batched_wg<T, 128, OrmbrPLarftKernel128<T>>(
            q,
            t_data, ld_t, stride_t,
            v_data, ld_v, stride_v,
            m, ib,
            tau_data, tau_stride, tau_offset,
            batch);
    }

    return ormbr_larft_forward_columnwise_batched_wg<T, 256, OrmbrPLarftKernel256<T>>(
        q,
        t_data, ld_t, stride_t,
        v_data, ld_v, stride_v,
        m, ib,
        tau_data, tau_stride, tau_offset,
        batch);
}

template <typename T>
sycl::event pack_p_panel_batched(Queue& q,
                                 T* v_out,
                                 int ld_v_out,
                                 int stride_v_out,
                                 const MatrixView<T, MatrixFormat::Dense>& a,
                                 int i0,
                                 int ib,
                                 int nq) {
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
                                                T val = T(0);
                                                if (r == c) {
                                                    val = T(1);
                                                } else if (r > c) {
                                                    const int row = i0 + c;
                                                    const int col = i0 + 1 + r;
                                                    val = conj_if_needed(a_ptr[b * stride_a + row + col * ld_a]);
                                                }
                                                v_out[b * stride_v_out + r + c * ld_v_out] = val;
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

template <typename T>
Event ormbr_apply_p_unblocked(Queue& ctx,
                              const MatrixView<T, MatrixFormat::Dense>& a,
                              const VectorView<T>& tau,
                              const MatrixView<T, MatrixFormat::Dense>& c,
                              Side side,
                              Transpose trans) {
    auto& c_mut = const_cast<MatrixView<T, MatrixFormat::Dense>&>(c);
    const int32_t n = static_cast<int32_t>(a.rows());
    const int32_t k = std::max<int32_t>(0, std::min<int32_t>(a.rows(), a.cols()) - 1);

    const bool apply_conj_trans =
        (trans == Transpose::ConjTrans) ||
        (trans == Transpose::Trans && !internal::is_complex<T>::value);
    const bool forward = !apply_conj_trans;

    ctx->submit([&](sycl::handler& cgh) {
        auto A = a.kernel_view();
        auto C = c_mut.kernel_view();
        auto TAU = tau;

        cgh.parallel_for<OrmbrPKernel<T>>(sycl::range<1>(static_cast<size_t>(a.batch_size())), [=](sycl::id<1> tid) {
            const int32_t b = static_cast<int32_t>(tid[0]);

            auto apply_reflector = [&](int32_t i) {
                if (i < 0 || i >= k) return;
                const int32_t start = i + 1;
                const int32_t len = n - start;
                if (len <= 0) return;

                T tau_i = TAU(i, b);
                if (apply_conj_trans) {
                    tau_i = conj_if_needed(tau_i);
                }
                if (tau_i == T(0)) return;

                if (side == Side::Right) {
                    const int32_t m = static_cast<int32_t>(C.rows());
                    const int32_t ncols = static_cast<int32_t>(C.cols());
                    if (ncols < n) return;

                    for (int32_t r = 0; r < m; ++r) {
                        T dot = T(0);
                        for (int32_t t = 0; t < len; ++t) {
                            const int32_t col = start + t;
                            const T v = (t == 0) ? T(1) : A(i, col, b);
                            dot += C(r, col, b) * conj_if_needed(v);
                        }
                        dot *= tau_i;
                        for (int32_t t = 0; t < len; ++t) {
                            const int32_t col = start + t;
                            const T v = (t == 0) ? T(1) : A(i, col, b);
                            C(r, col, b) -= dot * v;
                        }
                    }
                } else {
                    const int32_t mrows = static_cast<int32_t>(C.rows());
                    const int32_t ncols = static_cast<int32_t>(C.cols());
                    if (mrows < n) return;

                    for (int32_t col = 0; col < ncols; ++col) {
                        T dot = T(0);
                        for (int32_t t = 0; t < len; ++t) {
                            const int32_t row = start + t;
                            const T v = (t == 0) ? T(1) : A(i, row, b);
                            dot += conj_if_needed(v) * C(row, col, b);
                        }
                        dot *= tau_i;
                        for (int32_t t = 0; t < len; ++t) {
                            const int32_t row = start + t;
                            const T v = (t == 0) ? T(1) : A(i, row, b);
                            C(row, col, b) -= v * dot;
                        }
                    }
                }
            };

            if (forward) {
                for (int32_t i = 0; i < k; ++i) {
                    apply_reflector(i);
                }
            } else {
                for (int32_t i = k - 1; i >= 0; --i) {
                    apply_reflector(i);
                }
            }
        });
    });

    return ctx.get_event();
}

template <Backend B, typename T, int NB>
size_t ormbr_p_blocked_buffer_size_impl(Queue& ctx,
                                        const MatrixView<T, MatrixFormat::Dense>& a,
                                        const MatrixView<T, MatrixFormat::Dense>& c,
                                        Side side,
                                        int32_t block_size) {
    const int nq = std::max<int>(0, static_cast<int>(a.rows()) - 1);
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
    const int nq = static_cast<int>(a.rows());
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

    MatrixView<T, MatrixFormat::Dense> Vmat(Vbuf.data(), nq - 1, nb, nq - 1, static_cast<int64_t>(nq - 1) * static_cast<int64_t>(nb), batch);
    MatrixView<T, MatrixFormat::Dense> Tmat(Tbuf.data(), nb, nb, nb, static_cast<int64_t>(nb) * static_cast<int64_t>(nb), batch);

    auto apply_block = [&](int i0) {
        const int ib = std::min(nb, k - i0);
        const int m = nq - i0 - 1;
        if (ib <= 0 || m <= 0) {
            return;
        }

        (void)pack_p_panel_batched<T>(ctx,
                                      Vmat.data_ptr(), Vmat.ld(), Vmat.stride(),
                                      a, i0, ib, nq);

        (void)ormbr_larft_forward_columnwise_batched<T>(ctx,
                                                        Tmat.data_ptr(), Tmat.ld(), Tmat.stride(),
                                                        Vmat.data_ptr(), Vmat.ld(), Vmat.stride(),
                                                        m, ib,
                                                        tau.data_ptr(), tau.stride(), i0,
                                                        batch);

        auto Ublk = Vmat({0, m}, {0, ib});
        auto Tblk = Tmat({0, ib}, {0, ib});
        const Transpose t_eff = transpose_apply ? Transpose::ConjTrans : Transpose::NoTrans;

        if (side == Side::Left) {
            auto Csub = c({i0 + 1, SliceEnd()}, Slice());
            MatrixView<T, MatrixFormat::Dense> W1full(W1buf.data(), nb, nC, nb, static_cast<int64_t>(nb) * static_cast<int64_t>(nC), batch);
            MatrixView<T, MatrixFormat::Dense> W2full(W2buf.data(), nb, nC, nb, static_cast<int64_t>(nb) * static_cast<int64_t>(nC), batch);
            auto W1 = W1full({0, ib}, Slice());
            auto W2 = W2full({0, ib}, Slice());

            gemm<B>(ctx, Ublk, Csub, W1, T(1), T(0), Transpose::ConjTrans, Transpose::NoTrans);
            gemm<B>(ctx, Tblk, W1, W2, T(1), T(0), t_eff, Transpose::NoTrans);
            gemm<B>(ctx, Ublk, W2, Csub, T(-1), T(1), Transpose::NoTrans, Transpose::NoTrans);
        } else {
            auto Csub = c(Slice(), {i0 + 1, SliceEnd()});
            MatrixView<T, MatrixFormat::Dense> W1full(W1buf.data(), mC, nb, mC, static_cast<int64_t>(mC) * static_cast<int64_t>(nb), batch);
            MatrixView<T, MatrixFormat::Dense> W2full(W2buf.data(), mC, nb, mC, static_cast<int64_t>(mC) * static_cast<int64_t>(nb), batch);
            auto W1 = W1full(Slice(), {0, ib});
            auto W2 = W2full(Slice(), {0, ib});

            gemm<B>(ctx, Csub, Ublk, W1, T(1), T(0), Transpose::NoTrans, Transpose::NoTrans);
            gemm<B>(ctx, W1, Tblk, W2, T(1), T(0), Transpose::NoTrans, t_eff);
            gemm<B>(ctx, W2, Ublk, Csub, T(-1), T(1), Transpose::NoTrans, Transpose::ConjTrans);
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
    if (a.rows() != a.cols()) {
        throw std::invalid_argument("ormbr: current implementation supports square A only");
    }
    if (a.batch_size() != c.batch_size() || tau.batch_size() != a.batch_size()) {
        throw std::invalid_argument("ormbr: batch size mismatch");
    }
    if (a.batch_size() < 1) {
        throw std::invalid_argument("ormbr: invalid batch size");
    }

    const int32_t n = static_cast<int32_t>(a.rows());
    const int32_t k = std::min<int32_t>(a.rows(), a.cols());
    const int32_t nq = (side == Side::Left) ? static_cast<int32_t>(c.rows()) : static_cast<int32_t>(c.cols());
    if (nq != n) {
        throw std::invalid_argument("ormbr: expected nq == A.rows() for square-path staging");
    }

    const char v = upper_ascii(vect);
    if (v != 'Q' && v != 'P') {
        throw std::invalid_argument("ormbr: vect must be 'Q' or 'P'");
    }

    const int32_t need_tau = (v == 'Q') ? k : std::max<int32_t>(0, k - 1);
    if (tau.inc() != 1) {
        throw std::invalid_argument("ormbr: tau must be unit-stride");
    }
    if (tau.stride() != tau.size()) {
        throw std::invalid_argument("ormbr: tau must be tightly packed by batch");
    }
    if (tau.size() < static_cast<size_t>(need_tau)) {
        throw std::invalid_argument("ormbr: tau span too small");
    }
}

} // namespace

template <Backend B, typename T>
Event ormbr(Queue& ctx,
            const MatrixView<T, MatrixFormat::Dense>& a,
            const VectorView<T>& tau,
            const MatrixView<T, MatrixFormat::Dense>& c,
            char vect,
            Side side,
            Transpose trans,
            const Span<std::byte>& ws,
            int32_t block_size) {
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

    if constexpr (internal::is_complex<T>::value) {
        if (trans == Transpose::Trans) {
            throw std::runtime_error("ormbr: Trans not supported for complex 'P'; use ConjTrans");
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

    if constexpr (internal::is_complex<T>::value) {
        if (trans == Transpose::Trans) {
            throw std::runtime_error("ormbr_buffer_size: Trans not supported for complex 'P'; use ConjTrans");
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

#define ORMBR_INSTANTIATE_FOR_BACKEND(back) \
    BATCHLAS_FOR_EACH_SCALAR_TYPE_1(ORMBR_INSTANTIATE, back)

#if BATCHLAS_HAS_CUDA_BACKEND
ORMBR_INSTANTIATE_FOR_BACKEND(Backend::CUDA)
#endif

#if BATCHLAS_HAS_ROCM_BACKEND
ORMBR_INSTANTIATE_FOR_BACKEND(Backend::ROCM)
#endif

#if BATCHLAS_HAS_HOST_BACKEND
ORMBR_INSTANTIATE_FOR_BACKEND(Backend::NETLIB)
#endif

#undef ORMBR_INSTANTIATE_FOR_BACKEND
#undef ORMBR_INSTANTIATE

#undef ORMBR_INSTANTIATE

} // namespace batchlas
