#include <blas/extensions.hh>
#include <blas/functions.hh>
#include <batchlas/backend_config.h>
#include <util/mempool.hh>

#include "../math-helpers.hh"
#include "../queue.hh"
#include "../util/template-instantiations.hh"

#include <algorithm>
#include <stdexcept>

namespace batchlas {

namespace {

template <typename T>
inline T conj_if_needed(const T& x) {
    if constexpr (internal::is_complex<T>::value) {
        return T(x.real(), -x.imag());
    } else {
        return x;
    }
}

inline int32_t gebrd_blocked_resolved_nb(int32_t block_size) {
    return std::max<int32_t>(1, block_size);
}

template <typename T>
struct GebrdBlockedWorkspace {
    MatrixView<T, MatrixFormat::Dense> x;
    MatrixView<T, MatrixFormat::Dense> y;
};

// Single description of gebrd_blocked's workspace; see workspace_bytes() in
// util/mempool.hh.
template <typename T>
GebrdBlockedWorkspace<T> gebrd_blocked_layout(Queue& ctx,
                                              BumpAllocator& pool,
                                              int32_t m,
                                              int32_t n,
                                              int32_t nb,
                                              int32_t batch) {
    auto x_buf = pool.allocate<T>(ctx, static_cast<size_t>(m) * static_cast<size_t>(nb) * static_cast<size_t>(batch));
    auto y_buf = pool.allocate<T>(ctx, static_cast<size_t>(n) * static_cast<size_t>(nb) * static_cast<size_t>(batch));
    return {
        MatrixView<T, MatrixFormat::Dense>(x_buf.data(), m, nb, m, static_cast<int64_t>(m) * static_cast<int64_t>(nb), batch),
        MatrixView<T, MatrixFormat::Dense>(y_buf.data(), n, nb, n, static_cast<int64_t>(n) * static_cast<int64_t>(nb), batch),
    };
}

template <typename T>
inline void validate_gebrd_dims(const MatrixView<T, MatrixFormat::Dense>& a,
                                const VectorView<typename base_type<T>::type>& d,
                                const VectorView<typename base_type<T>::type>& e,
                                const VectorView<T>& tauq,
                                const VectorView<T>& taup,
                                const char* where) {
    if (a.rows() < a.cols()) {
        throw std::invalid_argument(std::string(where) + ": A must satisfy rows >= cols");
    }

    const int32_t k = static_cast<int32_t>(a.cols());
    const int32_t need_e = std::max<int32_t>(0, k - 1);
    if (d.size() != k || e.size() != need_e || tauq.size() != k || taup.size() != k) {
        throw std::invalid_argument(std::string(where) + ": invalid d/e/tau sizes");
    }
    if (a.batch_size() != d.batch_size() || a.batch_size() != e.batch_size() ||
        a.batch_size() != tauq.batch_size() || a.batch_size() != taup.batch_size()) {
        throw std::invalid_argument(std::string(where) + ": batch size mismatch");
    }
    if (a.batch_size() < 1) {
        throw std::invalid_argument(std::string(where) + ": invalid batch size");
    }
}

template <typename T>
Event gebrd_restore_bidiag_upper(Queue& ctx,
                                 const MatrixView<T, MatrixFormat::Dense>& a,
                                 const VectorView<typename base_type<T>::type>& d,
                                 const VectorView<typename base_type<T>::type>& e) {
    auto& a_mut = const_cast<MatrixView<T, MatrixFormat::Dense>&>(a);
    const int32_t k = static_cast<int32_t>(d.size());
    const int32_t batch = static_cast<int32_t>(a.batch_size());

    ctx->submit([&](sycl::handler& h) {
        auto A = a_mut.kernel_view();
        auto D = d;
        auto E = e;

        h.parallel_for(
            sycl::range<2>(static_cast<size_t>(batch), static_cast<size_t>(std::max<int32_t>(1, k))),
            [=](sycl::id<2> idx) {
                const int32_t b = static_cast<int32_t>(idx[0]);
                const int32_t i = static_cast<int32_t>(idx[1]);
                if (i >= k) return;

                A(i, i, b) = static_cast<T>(D(i, b));
                if (i < k - 1) {
                    A(i, i + 1, b) = static_cast<T>(E(i, b));
                }
            });
    });

    return ctx.get_event();
}

template <Backend B, typename T>
Event gebrd_blocked_real(Queue& ctx,
                         const MatrixView<T, MatrixFormat::Dense>& a_in,
                         const VectorView<typename base_type<T>::type>& d_out,
                         const VectorView<typename base_type<T>::type>& e_out,
                         const VectorView<T>& tauq_out,
                         const VectorView<T>& taup_out,
                         const Span<std::byte>& ws,
                         int32_t block_size) {
    static_assert(!internal::is_complex<T>::value, "gebrd_blocked_real expects real scalar type");
    using Real = typename base_type<T>::type;

    validate_gebrd_dims(a_in, d_out, e_out, tauq_out, taup_out, "gebrd_blocked");

    if (!ctx.in_order()) {
        throw std::runtime_error("gebrd_blocked: requires an in-order Queue");
    }

    constexpr int32_t MaxPanelNB = 64;
    const int32_t nb = gebrd_blocked_resolved_nb(block_size);
    if (nb > MaxPanelNB) {
        throw std::invalid_argument("gebrd_blocked: block_size > 64 is not supported");
    }

    auto& a = const_cast<MatrixView<T, MatrixFormat::Dense>&>(a_in);
    const int32_t m = static_cast<int32_t>(a.rows());
    const int32_t n = static_cast<int32_t>(a.cols());
    const int32_t k_total = std::min(m, n);
    const int32_t batch = static_cast<int32_t>(a.batch_size());

    Span<std::byte> ws_mut(const_cast<std::byte*>(ws.data()), ws.size());
    BumpAllocator pool(ws_mut);

    auto ws_layout = gebrd_blocked_layout<T>(ctx, pool, m, n, nb, batch);
    MatrixView<T, MatrixFormat::Dense> x_mat = ws_layout.x;
    MatrixView<T, MatrixFormat::Dense> y_mat = ws_layout.y;
    const int32_t panel_wg = std::min<int32_t>(128, static_cast<int32_t>(ctx->get_device().get_info<sycl::info::device::max_work_group_size>()));

    for (int32_t j0 = 0; j0 < k_total; j0 += nb) {
        const int32_t ib = std::min(nb, k_total - j0);
        auto x_panel = x_mat({j0, SliceEnd()}, {0, ib});
        auto y_panel = y_mat({j0, SliceEnd()}, {0, ib});
        x_panel.fill_zeros(ctx);
        y_panel.fill_zeros(ctx);

        ctx->submit([&](sycl::handler& h) {
            auto A = a.kernel_view();
            auto D = d_out;
            auto E = e_out;
            auto TAUQ = tauq_out;
            auto TAUP = taup_out;
            auto X = x_mat.kernel_view();
            auto Y = y_mat.kernel_view();
            auto work_local = sycl::local_accessor<T, 1>(sycl::range<1>(static_cast<size_t>(2 * nb)), h);

            h.parallel_for(sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(batch) * static_cast<size_t>(panel_wg)),
                                             sycl::range<1>(static_cast<size_t>(panel_wg))),
                           [=](sycl::nd_item<1> item) {
                const int32_t b = static_cast<int32_t>(item.get_group_linear_id());
                if (b >= batch) return;

                const int32_t lid = static_cast<int32_t>(item.get_local_linear_id());
                const int32_t local_size = static_cast<int32_t>(item.get_local_range(0));
                const auto g = item.get_group();

                for (int32_t ii = 0; ii < ib; ++ii) {
                    const int32_t gi = j0 + ii;

                    for (int32_t r = gi + lid; r < m; r += local_size) {
                        T val = A(r, gi, b);
                        for (int32_t k = 0; k < ii; ++k) {
                            const int32_t gk = j0 + k;
                            val -= A(r, gk, b) * Y(gi, k, b);
                            val -= X(r, k, b) * A(gk, gi, b);
                        }
                        A(r, gi, b) = val;
                    }
                    item.barrier(sycl::access::fence_space::global_space);

                    Real sigma_partial = Real(0);
                    for (int32_t r = gi + 1 + lid; r < m; r += local_size) {
                        const T ari = A(r, gi, b);
                        sigma_partial += ari * ari;
                    }
                    const Real sigma = sycl::reduce_over_group(g, sigma_partial, sycl::plus<Real>());

                    T alpha = T(0);
                    if (lid == 0) {
                        alpha = A(gi, gi, b);
                    }
                    alpha = sycl::group_broadcast(g, alpha);
                    T tau_q = T(0);
                    T beta_q = alpha;
                    T scale_q = T(0);
                    if (lid == 0) {
                        const auto scalars = internal::larfg(alpha, sycl::sqrt(sigma), m - gi);
                        beta_q = scalars.beta;
                        tau_q = scalars.tau;
                        scale_q = scalars.scale;
                    }
                    tau_q = sycl::group_broadcast(g, tau_q);
                    beta_q = sycl::group_broadcast(g, beta_q);
                    scale_q = sycl::group_broadcast(g, scale_q);

                    if (lid == 0) {
                        D(gi, b) = static_cast<Real>(beta_q);
                        TAUQ(gi, b) = tau_q;
                    }

                    if (lid == 0) {
                        A(gi, gi, b) = T(1);
                    }
                    if (tau_q != T(0)) {
                        for (int32_t r = gi + 1 + lid; r < m; r += local_size) {
                            A(r, gi, b) *= scale_q;
                        }
                    }
                    item.barrier(sycl::access::fence_space::global_space);

                    if (gi >= n - 1) {
                        continue;
                    }

                    for (int32_t k = 0; k < ii; ++k) {
                        const int32_t gk = j0 + k;
                        T sum_a_partial = T(0);
                        T sum_x_partial = T(0);
                        for (int32_t r = gi + lid; r < m; r += local_size) {
                            const T argi = A(r, gi, b);
                            sum_a_partial += A(r, gk, b) * argi;
                            sum_x_partial += X(r, k, b) * argi;
                        }
                        const T sum_a = sycl::reduce_over_group(g, sum_a_partial, sycl::plus<T>());
                        const T sum_x = sycl::reduce_over_group(g, sum_x_partial, sycl::plus<T>());
                        if (lid == 0) {
                            work_local[k] = sum_a;
                            work_local[nb + k] = sum_x;
                        }
                    }
                    item.barrier(sycl::access::fence_space::local_space);

                    for (int32_t c = gi + 1 + lid; c < n; c += local_size) {
                        T y = T(0);
                        for (int32_t r = gi; r < m; ++r) {
                            y += A(r, c, b) * A(r, gi, b);
                        }
                        for (int32_t k = 0; k < ii; ++k) {
                            y -= Y(c, k, b) * work_local[k];
                            y -= A(j0 + k, c, b) * work_local[nb + k];
                        }
                        Y(c, ii, b) = tau_q * y;
                    }
                    item.barrier(sycl::access::fence_space::global_space);

                    for (int32_t c = gi + 1 + lid; c < n; c += local_size) {
                        T val = A(gi, c, b);
                        for (int32_t k = 0; k <= ii; ++k) {
                            val -= Y(c, k, b) * A(gi, j0 + k, b);
                        }
                        for (int32_t k = 0; k < ii; ++k) {
                            val -= A(j0 + k, c, b) * X(gi, k, b);
                        }
                        A(gi, c, b) = val;
                    }
                    item.barrier(sycl::access::fence_space::global_space);

                    Real sigma_r_partial = Real(0);
                    for (int32_t c = gi + 2 + lid; c < n; c += local_size) {
                        const T aic = A(gi, c, b);
                        sigma_r_partial += aic * aic;
                    }
                    const Real sigma_r = sycl::reduce_over_group(g, sigma_r_partial, sycl::plus<Real>());

                    T alpha_r = T(0);
                    if (lid == 0) {
                        alpha_r = A(gi, gi + 1, b);
                    }
                    alpha_r = sycl::group_broadcast(g, alpha_r);
                    T tau_p = T(0);
                    T beta_p = alpha_r;
                    T scale_p = T(0);
                    if (lid == 0) {
                        const auto scalars = internal::larfg(alpha_r, sycl::sqrt(sigma_r), n - (gi + 1));
                        beta_p = scalars.beta;
                        tau_p = scalars.tau;
                        scale_p = scalars.scale;
                    }
                    tau_p = sycl::group_broadcast(g, tau_p);
                    beta_p = sycl::group_broadcast(g, beta_p);
                    scale_p = sycl::group_broadcast(g, scale_p);

                    if (lid == 0) {
                        E(gi, b) = static_cast<Real>(beta_p);
                        TAUP(gi, b) = tau_p;
                        A(gi, gi + 1, b) = T(1);
                    }
                    if (tau_p != T(0)) {
                        for (int32_t c = gi + 2 + lid; c < n; c += local_size) {
                            A(gi, c, b) *= scale_p;
                        }
                    }
                    item.barrier(sycl::access::fence_space::global_space);

                    for (int32_t k = 0; k <= ii; ++k) {
                        T sum_y_partial = T(0);
                        for (int32_t c = gi + 1 + lid; c < n; c += local_size) {
                            sum_y_partial += Y(c, k, b) * A(gi, c, b);
                        }
                        const T sum_y = sycl::reduce_over_group(g, sum_y_partial, sycl::plus<T>());
                        if (lid == 0) {
                            work_local[k] = sum_y;
                        }
                    }
                    for (int32_t k = 0; k < ii; ++k) {
                        T sum_a_partial = T(0);
                        for (int32_t c = gi + 1 + lid; c < n; c += local_size) {
                            sum_a_partial += A(j0 + k, c, b) * A(gi, c, b);
                        }
                        const T sum_a = sycl::reduce_over_group(g, sum_a_partial, sycl::plus<T>());
                        if (lid == 0) {
                            work_local[nb + k] = sum_a;
                        }
                    }
                    item.barrier(sycl::access::fence_space::local_space);

                    for (int32_t r = gi + 1 + lid; r < m; r += local_size) {
                        T x = T(0);
                        for (int32_t c = gi + 1; c < n; ++c) {
                            x += A(r, c, b) * A(gi, c, b);
                        }
                        for (int32_t k = 0; k <= ii; ++k) {
                            x -= A(r, j0 + k, b) * work_local[k];
                        }
                        for (int32_t k = 0; k < ii; ++k) {
                            x -= X(r, k, b) * work_local[nb + k];
                        }
                        X(r, ii, b) = tau_p * x;
                    }
                    item.barrier(sycl::access::fence_space::global_space);
                }

                if (lid == 0 && n > 0) {
                    TAUP(n - 1, b) = T(0);
                }
            });
        });

        const int32_t j2 = j0 + ib;
        if (j2 < n) {
            auto a22 = a({j2, SliceEnd()}, {j2, SliceEnd()});
            auto v2 = a({j2, SliceEnd()}, {j0, j2});
            auto y2 = y_mat({j2, SliceEnd()}, {0, ib});
            auto x2 = x_mat({j2, SliceEnd()}, {0, ib});
            auto u2 = a({j0, j2}, {j2, SliceEnd()});

            gemm<B>(ctx, v2, y2, a22, {.alpha = T(-1), .beta = T(1), .transB = Transpose::Trans});
            gemm<B>(ctx, x2, u2, a22, {.alpha = T(-1), .beta = T(1)});
        }
    }

    gebrd_restore_bidiag_upper(ctx, a, d_out, e_out);
    return ctx.get_event();
}

} // namespace

template <Backend B, typename T>
Event gebrd_blocked(Queue& ctx,
                    const MatrixView<T, MatrixFormat::Dense>& a_in,
                    const VectorView<typename base_type<T>::type>& d_out,
                    const VectorView<typename base_type<T>::type>& e_out,
                    const VectorView<T>& tauq_out,
                    const VectorView<T>& taup_out,
                    const Span<std::byte>& ws,
                    int32_t block_size) {
    if constexpr (internal::is_complex<T>::value) {
        throw std::runtime_error("gebrd_blocked: complex types are not implemented");
    } else {
        return gebrd_blocked_real<B, T>(ctx, a_in, d_out, e_out, tauq_out, taup_out, ws, block_size);
    }
}

template <Backend B, typename T>
size_t gebrd_blocked_buffer_size(Queue& ctx,
                                 const MatrixView<T, MatrixFormat::Dense>& a,
                                 const VectorView<typename base_type<T>::type>& d,
                                 const VectorView<typename base_type<T>::type>& e,
                                 const VectorView<T>& tauq,
                                 const VectorView<T>& taup,
                                 int32_t block_size) {
    validate_gebrd_dims(a, d, e, tauq, taup, "gebrd_blocked_buffer_size");

    const int32_t m = static_cast<int32_t>(a.rows());
    const int32_t n = static_cast<int32_t>(a.cols());
    const int32_t batch = static_cast<int32_t>(a.batch_size());
    const int32_t nb = gebrd_blocked_resolved_nb(block_size);

    return workspace_bytes([&](BumpAllocator& pool) {
        return gebrd_blocked_layout<T>(ctx, pool, m, n, nb, batch);
    });
}

#define GEBRD_BLOCKED_INSTANTIATE(back, fp) \
    template Event gebrd_blocked<back, BATCHLAS_UNPAREN fp>( \
        Queue&, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        const VectorView<typename base_type<BATCHLAS_UNPAREN fp>::type>&, \
        const VectorView<typename base_type<BATCHLAS_UNPAREN fp>::type>&, \
        const VectorView<BATCHLAS_UNPAREN fp>&, \
        const VectorView<BATCHLAS_UNPAREN fp>&, \
        const Span<std::byte>&, \
        int32_t); \
    template size_t gebrd_blocked_buffer_size<back, BATCHLAS_UNPAREN fp>( \
        Queue&, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        const VectorView<typename base_type<BATCHLAS_UNPAREN fp>::type>&, \
        const VectorView<typename base_type<BATCHLAS_UNPAREN fp>::type>&, \
        const VectorView<BATCHLAS_UNPAREN fp>&, \
        const VectorView<BATCHLAS_UNPAREN fp>&, \
        int32_t);

#define GEBRD_BLOCKED_INSTANTIATE_FOR_BACKEND(back) \
    BATCHLAS_FOR_EACH_SCALAR_TYPE_1(GEBRD_BLOCKED_INSTANTIATE, back)

#if BATCHLAS_HAS_CUDA_BACKEND
GEBRD_BLOCKED_INSTANTIATE_FOR_BACKEND(Backend::CUDA)
#endif

#if BATCHLAS_HAS_ROCM_BACKEND
GEBRD_BLOCKED_INSTANTIATE_FOR_BACKEND(Backend::ROCM)
#endif

#if BATCHLAS_HAS_HOST_BACKEND
GEBRD_BLOCKED_INSTANTIATE_FOR_BACKEND(Backend::NETLIB)
#endif

#undef GEBRD_BLOCKED_INSTANTIATE_FOR_BACKEND
#undef GEBRD_BLOCKED_INSTANTIATE

} // namespace batchlas
