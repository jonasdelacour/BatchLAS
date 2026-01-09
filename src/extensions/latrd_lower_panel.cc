#include <blas/extensions.hh>
#include <blas/matrix.hh>

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
inline U conj_if_needed(const U& x) {
    if constexpr (internal::is_complex<U>::value) {
        return U(x.real(), -x.imag());
    } else {
        return x;
    }
}

template <typename T>
inline typename base_type<T>::type abs2_if_complex(const T& x) {
    using Real = typename base_type<T>::type;
    if constexpr (internal::is_complex<T>::value) {
        const Real re = x.real();
        const Real im = x.imag();
        return re * re + im * im;
    } else {
        return x * x;
    }
}

template <typename Real>
inline Real sign_nonzero_real(Real x) {
    return (sycl::signbit(x) ? Real(-1) : Real(1));
}

template <typename T>
inline T sign_nonzero(const T& x) {
    using Real = typename base_type<T>::type;
    if constexpr (internal::is_complex<T>::value) {
        const Real a = sycl::hypot(x.real(), x.imag());
        if (a == Real(0)) return T(1);
        return x / a;
    } else {
        return T(sign_nonzero_real(static_cast<Real>(x)));
    }
}

template <typename T>
inline T reduce_sum_group(const sycl::group<1>& g, T x) {
    if constexpr (internal::is_complex<T>::value) {
        using R = typename T::value_type;
        const R re = sycl::reduce_over_group(g, x.real(), sycl::plus<R>());
        const R im = sycl::reduce_over_group(g, x.imag(), sycl::plus<R>());
        return T(re, im);
    } else {
        return sycl::reduce_over_group(g, x, sycl::plus<T>());
    }
}

template <typename T>
inline typename base_type<T>::type reduce_sum_group_real(const sycl::group<1>& g,
                                                        typename base_type<T>::type x) {
    using R = typename base_type<T>::type;
    return sycl::reduce_over_group(g, x, sycl::plus<R>());
}

template <typename T, int WG>
class LatrdLowerPanelKernel;

template <typename T, int WG>
Event latrd_lower_panel_batched_wg(Queue& q,
                                  const MatrixView<T, MatrixFormat::Dense>& a,
                                  const VectorView<T>& e,
                                  const VectorView<T>& tau,
                                  const MatrixView<T, MatrixFormat::Dense>& w) {
    constexpr int wg = WG;

    (void)q->submit([&](sycl::handler& h) {
        // Create kernel-passable views inside submit (MatrixView is not trivially copyable).
        KernelMatrixView<T, MatrixFormat::Dense> A_view(a.data_ptr(), a.rows(), a.cols(), a.ld(), a.stride(), a.batch_size());
        KernelMatrixView<T, MatrixFormat::Dense> W_view(w.data_ptr(), w.rows(), w.cols(), w.ld(), w.stride(), w.batch_size());

        // VectorView is device-copyable in BatchLAS and provides indexing + batch abstraction.
        VectorView<T> E_view = e;
        VectorView<T> TAU_view = tau;

        const int n = A_view.rows();
        const int batch = A_view.batch_size();
        const int ib = W_view.cols();

        h.parallel_for<LatrdLowerPanelKernel<T, WG>>(
            sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(batch) * wg), sycl::range<1>(wg)),
            [=](sycl::nd_item<1> it) {
                const int b = static_cast<int>(it.get_group_linear_id());
                if (b >= batch) return;

                const int lid = static_cast<int>(it.get_local_linear_id());
                const sycl::group<1> g = it.get_group();

                auto Ab = A_view.batch_item(b);
                auto Wb = W_view.batch_item(b);

                auto Ah = [&](int r, int c) -> T {
                    // Treat A as Hermitian/symmetric using the lower triangle.
                    if (r >= c) return Ab(r, c);
                    return conj_if_needed(Ab(c, r));
                };

                for (int i = 0; i < ib; ++i) {
                    if (i >= n - 1) break;

                    // Update diagonal element A(i,i) using previously computed V/W (j0..i-1).
                    if (lid == 0) {
                        T aii = Ab(i, i);
                        for (int p = 0; p < i; ++p) {
                            const int pc = p;
                            const T vip = (i == p + 1) ? T(1) : Ab(i, p);
                            const T wip = Wb(i, pc);
                            aii -= vip * conj_if_needed(wip) + wip * conj_if_needed(vip);
                        }
                        Ab(i, i) = aii;
                    }

                    // Update column i entries from row i+1 .. n-1.
                    for (int r = i + 1 + lid; r < n; r += wg) {
                        T val = Ab(r, i);
                        for (int p = 0; p < i; ++p) {
                            const int pc = p;
                            const T wip = Wb(i, pc);
                            const T vip = (i == p + 1) ? T(1) : Ab(i, p);

                            // V(r,p)
                            T vrp = T(0);
                            if (r == p + 1) {
                                vrp = T(1);
                            } else if (r > p + 1) {
                                vrp = Ab(r, p);
                            }

                            const T wrp = Wb(r, pc);
                            val -= vrp * conj_if_needed(wip) + wrp * conj_if_needed(vip);
                        }
                        Ab(r, i) = val;
                    }
                    it.barrier(sycl::access::fence_space::global_space);

                    // Generate Householder reflector to annihilate A(i+2:n-1,i).
                    using Real = typename base_type<T>::type;

                    const int x0 = i + 2;
                    Real sumsq = Real(0);
                    for (int r = x0 + lid; r < n; r += wg) {
                        sumsq += abs2_if_complex(Ab(r, i));
                    }
                    sumsq = reduce_sum_group_real<T>(g, sumsq);

                    const T alpha = (i + 1 < n) ? Ab(i + 1, i) : T(0);

                    T tau_i = T(0);
                    T beta = alpha;
                    T scale = T(0);

                    if (lid == 0) {
                        const Real xnorm = sycl::sqrt(sumsq);
                        if constexpr (internal::is_complex<T>::value) {
                            if (xnorm == Real(0) && alpha.imag() == Real(0)) {
                                tau_i = T(0);
                                beta = alpha;
                                scale = T(0);
                            } else {
                                const Real alpha_abs = sycl::hypot(alpha.real(), alpha.imag());
                                const Real beta_abs = sycl::hypot(alpha_abs, xnorm);
                                const T alpha_sign = (alpha_abs == Real(0)) ? T(1) : (alpha / alpha_abs);
                                beta = -alpha_sign * T(beta_abs);
                                tau_i = (beta - alpha) / beta;
                                scale = T(1) / (alpha - beta);
                            }
                        } else {
                            if (xnorm == Real(0)) {
                                tau_i = T(0);
                                beta = alpha;
                                scale = T(0);
                            } else {
                                beta = -sign_nonzero(alpha) * T(sycl::hypot(static_cast<Real>(alpha), xnorm));
                                tau_i = (beta - alpha) / beta;
                                scale = T(1) / (alpha - beta);
                            }
                        }

                        E_view(i, b) = beta;
                        TAU_view(i, b) = tau_i;

                        // Set v(0)=1 at A(i+1,i).
                        Ab(i + 1, i) = T(1);
                    }

                    // Broadcast tau/scale.
                    tau_i = sycl::group_broadcast(g, tau_i);
                    scale = sycl::group_broadcast(g, scale);

                    if (tau_i != T(0)) {
                        for (int r = x0 + lid; r < n; r += wg) {
                            Ab(r, i) *= scale;
                        }
                    }
                    it.barrier(sycl::access::fence_space::global_space);

                    // Compute W(:, i-j0) for rows i+1..n-1.
                    // w = tau * A(i+1:n-1, i+1:n-1) * v
                    const int col = i;

                    // Compute raw w (before scaling by tau) for the *updated* trailing matrix:
                    //   A := A - V*W^H - W*V^H  (within the current panel, columns j0..i-1)
                    // without explicitly forming A.
                    for (int r = i + 1 + lid; r < n; r += wg) {
                        T acc = T(0);
                        for (int c = i + 1; c < n; ++c) {
                            const T vc = (c == i + 1) ? T(1) : Ab(c, i);
                            acc += Ah(r, c) * vc;
                        }
                        Wb(r, col) = acc;
                    }

                    // Apply intra-panel corrections from previously computed reflectors.
                    for (int p = 0; p < i; ++p) {
                        const int pc = p;

                        // gamma = W(:,pc)^H * v, delta = V(:,p)^H * v
                        T gamma_partial = T(0);
                        T delta_partial = T(0);
                        for (int c = i + 1 + lid; c < n; c += wg) {
                            const T vc = (c == i + 1) ? T(1) : Ab(c, i);

                            gamma_partial += conj_if_needed(Wb(c, pc)) * vc;

                            const T vcp = (c == p + 1) ? T(1) : ((c > p + 1) ? Ab(c, p) : T(0));
                            delta_partial += conj_if_needed(vcp) * vc;
                        }
                        const T gamma = reduce_sum_group(g, gamma_partial);
                        const T delta = reduce_sum_group(g, delta_partial);

                        for (int r = i + 1 + lid; r < n; r += wg) {
                            const T vrp = (r == p + 1) ? T(1) : ((r > p + 1) ? Ab(r, p) : T(0));
                            const T wrp = Wb(r, pc);
                            Wb(r, col) -= vrp * gamma + wrp * delta;
                        }
                    }

                    // Scale by tau.
                    for (int r = i + 1 + lid; r < n; r += wg) {
                        Wb(r, col) *= tau_i;
                    }

                    // dot = v^H * w
                    T dot_partial = T(0);
                    for (int r = i + 1 + lid; r < n; r += wg) {
                        const T vr = (r == i + 1) ? T(1) : Ab(r, i);
                        dot_partial += conj_if_needed(vr) * Wb(r, col);
                    }
                    const T dot = reduce_sum_group(g, dot_partial);

                    // w += (-0.5 * tau * dot) * v
                    const T alpha2 = T(-0.5) * tau_i * dot;
                    for (int r = i + 1 + lid; r < n; r += wg) {
                        const T vr = (r == i + 1) ? T(1) : Ab(r, i);
                        Wb(r, col) += alpha2 * vr;
                    }
                    it.barrier(sycl::access::fence_space::global_space);
                }
            });
    });

    return q.get_event();
}

template <typename T>
Event latrd_lower_panel_batched(Queue& q,
                                const MatrixView<T, MatrixFormat::Dense>& a,
                                const VectorView<T>& e,
                                const VectorView<T>& tau,
                                const MatrixView<T, MatrixFormat::Dense>& w) {
    const int n = a.rows();
    // For very small n, a smaller work-group reduces wasted lanes and barrier overhead.
    if (n <= 64) {
        return latrd_lower_panel_batched_wg<T, 64>(q, a, e, tau, w);
    }
    if (n <= 128) {
        return latrd_lower_panel_batched_wg<T, 128>(q, a, e, tau, w);
    }
    return latrd_lower_panel_batched_wg<T, 256>(q, a, e, tau, w);
}

template <typename T>
inline void validate_latrd_lower_panel_panel_dims(const MatrixView<T, MatrixFormat::Dense>& a,
                                                  const VectorView<T>& e,
                                                  const VectorView<T>& tau,
                                                  const MatrixView<T, MatrixFormat::Dense>& w) {
    if (a.rows() != a.cols()) {
        throw std::invalid_argument("latrd_lower_panel(panel): A must be square");
    }
    if (w.rows() != a.rows()) {
        throw std::invalid_argument("latrd_lower_panel(panel): W must have same number of rows as A");
    }
    const int ib = w.cols();
    if (ib < 0) {
        throw std::invalid_argument("latrd_lower_panel(panel): invalid W dimensions");
    }
    if (e.size() != ib || tau.size() != ib) {
        throw std::invalid_argument("latrd_lower_panel(panel): e/tau must have size equal to W.cols()");
    }
    if (a.batch_size() != e.batch_size() || a.batch_size() != tau.batch_size() || a.batch_size() != w.batch_size()) {
        throw std::invalid_argument("latrd_lower_panel(panel): batch size mismatch");
    }
    if (a.batch_size() < 1) {
        throw std::invalid_argument("latrd_lower_panel(panel): invalid batch size");
    }
    // The algorithm only produces reflectors for columns 0..n-2.
    if (ib > std::max(0, a.rows() - 1)) {
        throw std::invalid_argument("latrd_lower_panel(panel): W.cols() must be <= A.rows()-1");
    }
}

template <typename T>
inline void validate_latrd_lower_panel_dims(const MatrixView<T, MatrixFormat::Dense>& a,
                                            const VectorView<T>& e,
                                            const VectorView<T>& tau,
                                            const MatrixView<T, MatrixFormat::Dense>& w,
                                            int32_t j0,
                                            int32_t ib) {
    if (a.rows() != a.cols()) {
        throw std::invalid_argument("latrd_lower_panel: A must be square");
    }
    const int n = a.rows();
    if (e.size() != std::max(0, n - 1) || tau.size() != std::max(0, n - 1)) {
        throw std::invalid_argument("latrd_lower_panel: invalid e/tau sizes");
    }
    if (w.rows() != n) {
        throw std::invalid_argument("latrd_lower_panel: W must have n rows");
    }
    if (w.cols() < ib) {
        throw std::invalid_argument("latrd_lower_panel: W must have at least ib columns");
    }
    if (j0 < 0 || ib < 0) {
        throw std::invalid_argument("latrd_lower_panel: j0/ib must be non-negative");
    }
    if (j0 > n) {
        throw std::invalid_argument("latrd_lower_panel: j0 out of range");
    }
    if (a.batch_size() != e.batch_size() || a.batch_size() != tau.batch_size() || a.batch_size() != w.batch_size()) {
        throw std::invalid_argument("latrd_lower_panel: batch size mismatch");
    }
    if (a.batch_size() < 1) {
        throw std::invalid_argument("latrd_lower_panel: invalid batch size");
    }
}

} // namespace

template <Backend B, typename T>
Event latrd_lower_panel(Queue& ctx,
                        const MatrixView<T, MatrixFormat::Dense>& a_panel_in,
                        const VectorView<T>& e_panel_out,
                        const VectorView<T>& tau_panel_out,
                        const MatrixView<T, MatrixFormat::Dense>& w_panel_in) {
    (void)B;
    validate_latrd_lower_panel_panel_dims(a_panel_in, e_panel_out, tau_panel_out, w_panel_in);

    auto& a = const_cast<MatrixView<T, MatrixFormat::Dense>&>(a_panel_in);
    auto& e = const_cast<VectorView<T>&>(e_panel_out);
    auto& tau = const_cast<VectorView<T>&>(tau_panel_out);
    auto& w = const_cast<MatrixView<T, MatrixFormat::Dense>&>(w_panel_in);

    // Early-exit for empty panels.
    if (w.cols() == 0) {
        return ctx.get_event();
    }

    return latrd_lower_panel_batched<T>(ctx, a, e, tau, w);
}

template <Backend B, typename T>
Event latrd_lower_panel(Queue& ctx,
                        const MatrixView<T, MatrixFormat::Dense>& a_in,
                        const VectorView<T>& e_out,
                        const VectorView<T>& tau_out,
                        const MatrixView<T, MatrixFormat::Dense>& w_in,
                        int32_t j0,
                        int32_t ib) {
    (void)B;
    validate_latrd_lower_panel_dims(a_in, e_out, tau_out, w_in, j0, ib);

    // Make mutable views (panel overwrites A and outputs e/tau/W).
    auto& a = const_cast<MatrixView<T, MatrixFormat::Dense>&>(a_in);
    auto& e = const_cast<VectorView<T>&>(e_out);
    auto& tau = const_cast<VectorView<T>&>(tau_out);
    auto& w = const_cast<MatrixView<T, MatrixFormat::Dense>&>(w_in);

    auto a_panel = a({j0, SliceEnd()}, {j0, SliceEnd()});
    auto e_panel = e(Slice(j0, j0 + ib));
    auto tau_panel = tau(Slice(j0, j0 + ib));
    auto w_panel = w({j0, SliceEnd()}, {0, ib});
    return latrd_lower_panel<B, T>(ctx, a_panel, e_panel, tau_panel, w_panel);
}

#define LATRD_LOWER_PANEL_INSTANTIATE(back, fp) \
    template Event latrd_lower_panel<back, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        const VectorView<fp>&, \
        const VectorView<fp>&, \
        const MatrixView<fp, MatrixFormat::Dense>&); \
    template Event latrd_lower_panel<back, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        const VectorView<fp>&, \
        const VectorView<fp>&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        int32_t, \
        int32_t);

#if BATCHLAS_HAS_CUDA_BACKEND
LATRD_LOWER_PANEL_INSTANTIATE(Backend::CUDA, float)
LATRD_LOWER_PANEL_INSTANTIATE(Backend::CUDA, double)
LATRD_LOWER_PANEL_INSTANTIATE(Backend::CUDA, std::complex<float>)
LATRD_LOWER_PANEL_INSTANTIATE(Backend::CUDA, std::complex<double>)
#endif

#if BATCHLAS_HAS_ROCM_BACKEND
LATRD_LOWER_PANEL_INSTANTIATE(Backend::ROCM, float)
LATRD_LOWER_PANEL_INSTANTIATE(Backend::ROCM, double)
LATRD_LOWER_PANEL_INSTANTIATE(Backend::ROCM, std::complex<float>)
LATRD_LOWER_PANEL_INSTANTIATE(Backend::ROCM, std::complex<double>)
#endif

#if BATCHLAS_HAS_HOST_BACKEND
LATRD_LOWER_PANEL_INSTANTIATE(Backend::NETLIB, float)
LATRD_LOWER_PANEL_INSTANTIATE(Backend::NETLIB, double)
LATRD_LOWER_PANEL_INSTANTIATE(Backend::NETLIB, std::complex<float>)
LATRD_LOWER_PANEL_INSTANTIATE(Backend::NETLIB, std::complex<double>)
#endif

#undef LATRD_LOWER_PANEL_INSTANTIATE

} // namespace batchlas
