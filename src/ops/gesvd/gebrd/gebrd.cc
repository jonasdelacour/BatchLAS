// SPDX-License-Identifier: MIT
#include <batchlas/blas/extensions.hh>
#include <batchlas/backend_config.h>

#include <src/math-helpers.hh>
#include <src/ops/gesvd/gebrd/larfg_real_beta.hh>
#include <src/queue.hh>
#include <src/util/template-instantiations.hh>
#include <src/util/batch_mask.hh>

#include <stdexcept>

namespace batchlas {

namespace {

template <typename T>
Event gebrd_unblocked_real(Queue& ctx,
                           const MatrixView<T, MatrixFormat::Dense>& a,
                           const VectorView<typename base_type<T>::type>& d,
                           const VectorView<typename base_type<T>::type>& e,
                           const VectorView<T>& tauq,
                           const VectorView<T>& taup) {
    static_assert(!detail::is_complex<T>::value, "gebrd_unblocked_real expects real scalar type");
    using Real = typename base_type<T>::type;

    const int32_t m = static_cast<int32_t>(a.rows());
    const int32_t n = static_cast<int32_t>(a.cols());
    const int32_t k = std::min(m, n);
    const int32_t batch = static_cast<int32_t>(a.batch_size());

    ctx->submit([&](sycl::handler& cgh) {
        auto A = a.kernel_view();
        auto D = d;
        auto E = e;
        auto TAUQ = tauq;
        auto TAUP = taup;

        cgh.parallel_for(sycl::range<1>(static_cast<size_t>(batch)), [=](sycl::id<1> tid) {
            const int32_t b = static_cast<int32_t>(tid[0]);

            for (int32_t i = 0; i < k; ++i) {
                // Left Householder: annihilate A(i+1:m-1, i)
                Real sigma = Real(0);
                for (int32_t r = i + 1; r < m; ++r) {
                    sigma += A(r, i, b) * A(r, i, b);
                }

                const Real alpha = A(i, i, b);
                const auto left_scalars = detail::larfg(alpha, sycl::sqrt(sigma), m - i);
                const Real tau_l = left_scalars.tau;
                const Real beta_l = left_scalars.beta;
                if (tau_l != Real(0)) {
                    const Real scale = left_scalars.scale;
                    A(i, i, b) = Real(1);
                    for (int32_t r = i + 1; r < m; ++r) {
                        A(r, i, b) *= scale;
                    }

                    for (int32_t c = i + 1; c < n; ++c) {
                        Real dot = Real(0);
                        for (int32_t r = i; r < m; ++r) {
                            const Real vr = (r == i) ? Real(1) : A(r, i, b);
                            dot += batchlas::device::detail::conj(vr) * A(r, c, b);
                        }
                        dot *= tau_l;
                        for (int32_t r = i; r < m; ++r) {
                            const Real vr = (r == i) ? Real(1) : A(r, i, b);
                            A(r, c, b) -= vr * dot;
                        }
                    }
                }

                A(i, i, b) = beta_l;
                D(i, b) = beta_l;
                TAUQ(i, b) = T(tau_l);

                if (i >= n - 1) {
                    TAUP(i, b) = T(0);
                    continue;
                }

                // Right Householder: annihilate A(i, i+2:n-1)
                Real sigma_r = Real(0);
                for (int32_t c = i + 2; c < n; ++c) {
                    sigma_r += A(i, c, b) * A(i, c, b);
                }

                const Real alpha_r = A(i, i + 1, b);
                const auto right_scalars = detail::larfg(alpha_r, sycl::sqrt(sigma_r), n - (i + 1));
                const Real tau_r = right_scalars.tau;
                const Real beta_r = right_scalars.beta;
                if (tau_r != Real(0)) {
                    const Real scale_r = right_scalars.scale;
                    A(i, i + 1, b) = Real(1);
                    for (int32_t c = i + 2; c < n; ++c) {
                        A(i, c, b) *= scale_r;
                    }

                    for (int32_t r = i + 1; r < m; ++r) {
                        Real dot = Real(0);
                        for (int32_t c = i + 1; c < n; ++c) {
                            const Real vc = (c == i + 1) ? Real(1) : A(i, c, b);
                            dot += A(r, c, b) * batchlas::device::detail::conj(vc);
                        }
                        dot *= tau_r;
                        for (int32_t c = i + 1; c < n; ++c) {
                            const Real vc = (c == i + 1) ? Real(1) : A(i, c, b);
                            A(r, c, b) -= dot * vc;
                        }
                    }
                }

                A(i, i + 1, b) = beta_r;
                E(i, b) = beta_r;
                TAUP(i, b) = T(tau_r);
            }

            for (int32_t i = k; i < tauq.size(); ++i) {
                TAUQ(i, b) = T(0);
            }
            for (int32_t i = std::max<int32_t>(0, k - 1); i < taup.size(); ++i) {
                TAUP(i, b) = T(0);
            }
        });
    });

    return ctx.get_event();
}

// zgebd2, one work-item per matrix: d, e real; row i holds conj(u_i) on exit (unmbr 'P' order).
template <typename T>
Event gebrd_unblocked_complex(Queue& ctx,
                              const MatrixView<T, MatrixFormat::Dense>& a,
                              const VectorView<typename base_type<T>::type>& d,
                              const VectorView<typename base_type<T>::type>& e,
                              const VectorView<T>& tauq,
                              const VectorView<T>& taup) {
    using Real = typename base_type<T>::type;
    using gebrd_detail::conj;
    const int32_t m = static_cast<int32_t>(a.rows());
    const int32_t n = static_cast<int32_t>(a.cols());
    const int32_t batch = static_cast<int32_t>(a.batch_size());

    ctx->submit([&](sycl::handler& cgh) {
        auto A = a.kernel_view();
        auto D = d;
        auto E = e;
        auto TAUQ = tauq;
        auto TAUP = taup;
        cgh.parallel_for(sycl::range<1>(static_cast<size_t>(batch)), [=](sycl::id<1> tid) {
            const int32_t b = static_cast<int32_t>(tid[0]);
            for (int32_t i = 0; i < n; ++i) {
                Real sigma = Real(0);
                for (int32_t r = i + 1; r < m; ++r) sigma += detail::norm_squared(A(r, i, b));
                const auto lq = gebrd_detail::larfg_real_beta(A(i, i, b), sycl::sqrt(sigma), m - i);
                for (int32_t r = i + 1; r < m; ++r) A(r, i, b) *= lq.scale;
                A(i, i, b) = T(1);
                // A := H^H A, H^H = I - conj(tau) v v^H.
                const T ctau = conj(lq.tau);
                for (int32_t c = i + 1; c < n && lq.tau != T(0); ++c) {
                    T dot = T(0);
                    for (int32_t r = i; r < m; ++r) dot += conj(A(r, i, b)) * A(r, c, b);
                    dot *= ctau;
                    for (int32_t r = i; r < m; ++r) A(r, c, b) -= A(r, i, b) * dot;
                }
                D(i, b) = detail::real_part(lq.beta);
                A(i, i, b) = lq.beta;
                TAUQ(i, b) = lq.tau;
                if (i >= n - 1) {
                    TAUP(i, b) = T(0);
                    continue;
                }
                for (int32_t c = i + 1; c < n; ++c) A(i, c, b) = conj(A(i, c, b));
                Real sigma_r = Real(0);
                for (int32_t c = i + 2; c < n; ++c) sigma_r += detail::norm_squared(A(i, c, b));
                const auto rp = gebrd_detail::larfg_real_beta(A(i, i + 1, b), sycl::sqrt(sigma_r), n - (i + 1));
                for (int32_t c = i + 2; c < n; ++c) A(i, c, b) *= rp.scale;
                A(i, i + 1, b) = T(1);
                // A := A G, G = I - tau u u^H.
                for (int32_t r = i + 1; r < m && rp.tau != T(0); ++r) {
                    T dot = T(0);
                    for (int32_t c = i + 1; c < n; ++c) dot += A(r, c, b) * A(i, c, b);
                    dot *= rp.tau;
                    for (int32_t c = i + 1; c < n; ++c) A(r, c, b) -= dot * conj(A(i, c, b));
                }
                for (int32_t c = i + 2; c < n; ++c) A(i, c, b) = conj(A(i, c, b));
                E(i, b) = detail::real_part(rp.beta);
                A(i, i + 1, b) = rp.beta;
                TAUP(i, b) = rp.tau;
            }
        });
    });
    return ctx.get_event();
}

} // namespace

template <Backend B, typename T>
Event gebrd_unblocked(Queue& ctx,
                      const MatrixView<T, MatrixFormat::Dense>& a,
                      const VectorView<typename base_type<T>::type>& d,
                      const VectorView<typename base_type<T>::type>& e,
                      const VectorView<T>& tauq,
                      const VectorView<T>& taup) {
    ::batchlas::detail::refuse_batch_mask("gebrd_unblocked", a, d, e, tauq, taup);
    static_cast<void>(B);

    if (a.rows() < a.cols()) {
        throw batchlas::invalid_argument("gebrd_unblocked: current implementation requires rows >= cols");
    }
    if constexpr (detail::is_complex<T>::value) {
        return gebrd_unblocked_complex<T>(ctx, a, d, e, tauq, taup);
    } else {
        return gebrd_unblocked_real<T>(ctx, a, d, e, tauq, taup);
    }
}

#define GEBRD_INSTANTIATE(back, fp) \
    template Event gebrd_unblocked<back, BATCHLAS_UNPAREN fp>( \
        Queue&, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        const VectorView<typename base_type<BATCHLAS_UNPAREN fp>::type>&, \
        const VectorView<typename base_type<BATCHLAS_UNPAREN fp>::type>&, \
        const VectorView<BATCHLAS_UNPAREN fp>&, \
        const VectorView<BATCHLAS_UNPAREN fp>&);

BATCHLAS_INSTANTIATE_SCALAR_ALL_BACKENDS(GEBRD_INSTANTIATE)

#undef GEBRD_INSTANTIATE

} // namespace batchlas
