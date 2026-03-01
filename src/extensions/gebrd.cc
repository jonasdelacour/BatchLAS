#include <blas/extensions.hh>
#include <batchlas/backend_config.h>

#include "../math-helpers.hh"
#include "../queue.hh"

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

template <typename T>
Event gebrd_unblocked_real(Queue& ctx,
                           const MatrixView<T, MatrixFormat::Dense>& a,
                           const VectorView<typename base_type<T>::type>& d,
                           const VectorView<typename base_type<T>::type>& e,
                           const VectorView<T>& tauq,
                           const VectorView<T>& taup) {
    static_assert(!internal::is_complex<T>::value, "gebrd_unblocked_real expects real scalar type");
    using Real = typename base_type<T>::type;

    const int32_t n = static_cast<int32_t>(a.rows());
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
                // Left Householder: annihilate A(i+1:n-1, i)
                Real sigma = Real(0);
                for (int32_t r = i + 1; r < n; ++r) {
                    sigma += A(r, i, b) * A(r, i, b);
                }

                const Real alpha = A(i, i, b);
                Real tau_l = Real(0);
                Real beta_l = alpha;
                if (sigma != Real(0)) {
                    beta_l = -sycl::copysign(sycl::hypot(alpha, sycl::sqrt(sigma)), alpha);
                    tau_l = (beta_l - alpha) / beta_l;
                    const Real scale = Real(1) / (alpha - beta_l);
                    A(i, i, b) = Real(1);
                    for (int32_t r = i + 1; r < n; ++r) {
                        A(r, i, b) *= scale;
                    }

                    for (int32_t c = i + 1; c < n; ++c) {
                        Real dot = Real(0);
                        for (int32_t r = i; r < n; ++r) {
                            const Real vr = (r == i) ? Real(1) : A(r, i, b);
                            dot += conj_if_needed(vr) * A(r, c, b);
                        }
                        dot *= tau_l;
                        for (int32_t r = i; r < n; ++r) {
                            const Real vr = (r == i) ? Real(1) : A(r, i, b);
                            A(r, c, b) -= vr * dot;
                        }
                    }
                }

                A(i, i, b) = beta_l;
                D(i, b) = beta_l;
                TAUQ(i, b) = T(tau_l);

                if (i >= n - 1) continue;

                // Right Householder: annihilate A(i, i+2:n-1)
                Real sigma_r = Real(0);
                for (int32_t c = i + 2; c < n; ++c) {
                    sigma_r += A(i, c, b) * A(i, c, b);
                }

                const Real alpha_r = A(i, i + 1, b);
                Real tau_r = Real(0);
                Real beta_r = alpha_r;
                if (sigma_r != Real(0)) {
                    beta_r = -sycl::copysign(sycl::hypot(alpha_r, sycl::sqrt(sigma_r)), alpha_r);
                    tau_r = (beta_r - alpha_r) / beta_r;
                    const Real scale_r = Real(1) / (alpha_r - beta_r);
                    A(i, i + 1, b) = Real(1);
                    for (int32_t c = i + 2; c < n; ++c) {
                        A(i, c, b) *= scale_r;
                    }

                    for (int32_t r = i + 1; r < n; ++r) {
                        Real dot = Real(0);
                        for (int32_t c = i + 1; c < n; ++c) {
                            const Real vc = (c == i + 1) ? Real(1) : A(i, c, b);
                            dot += A(r, c, b) * conj_if_needed(vc);
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

            if (n > 0) {
                TAUP(n - 1, b) = T(0);
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
    static_cast<void>(B);

    if (a.rows() != a.cols()) {
        throw std::invalid_argument("gebrd_unblocked: current implementation supports square matrices only");
    }
    if constexpr (internal::is_complex<T>::value) {
        throw std::runtime_error("gebrd_unblocked: complex types are not implemented yet");
    } else {
        return gebrd_unblocked_real<T>(ctx, a, d, e, tauq, taup);
    }
}

#define GEBRD_INSTANTIATE(back, fp) \
    template Event gebrd_unblocked<back, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        const VectorView<typename base_type<fp>::type>&, \
        const VectorView<typename base_type<fp>::type>&, \
        const VectorView<fp>&, \
        const VectorView<fp>&);

#if BATCHLAS_HAS_CUDA_BACKEND
GEBRD_INSTANTIATE(Backend::CUDA, float)
GEBRD_INSTANTIATE(Backend::CUDA, double)
GEBRD_INSTANTIATE(Backend::CUDA, std::complex<float>)
GEBRD_INSTANTIATE(Backend::CUDA, std::complex<double>)
#endif

#if BATCHLAS_HAS_ROCM_BACKEND
GEBRD_INSTANTIATE(Backend::ROCM, float)
GEBRD_INSTANTIATE(Backend::ROCM, double)
GEBRD_INSTANTIATE(Backend::ROCM, std::complex<float>)
GEBRD_INSTANTIATE(Backend::ROCM, std::complex<double>)
#endif

#if BATCHLAS_HAS_HOST_BACKEND
GEBRD_INSTANTIATE(Backend::NETLIB, float)
GEBRD_INSTANTIATE(Backend::NETLIB, double)
GEBRD_INSTANTIATE(Backend::NETLIB, std::complex<float>)
GEBRD_INSTANTIATE(Backend::NETLIB, std::complex<double>)
#endif

#undef GEBRD_INSTANTIATE

} // namespace batchlas
