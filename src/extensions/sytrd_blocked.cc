#include <blas/extensions.hh>
#include <blas/functions.hh>
#include <blas/matrix.hh>
#include <internal/sytrd_blocked.hh>
#include <util/mempool.hh>

#include <batchlas/backend_config.h>

#include "../math-helpers.hh"
#include "../queue.hh"

#include <algorithm>
#include <atomic>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <iostream>
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

template <typename T>
class UpdateVWLowerSmallKernel;

template <typename T>
class SytrdLowerLocalSmallKernel;

template <typename T>
Event update_vw_lower_small(Queue& q,
                            const MatrixView<T, MatrixFormat::Dense>& v2,
                            const MatrixView<T, MatrixFormat::Dense>& w2,
                            const MatrixView<T, MatrixFormat::Dense>& a22) {
    const int n2 = a22.rows();
    const int ib = v2.cols();
    const int lda = a22.ld();
    const int ldv = v2.ld();
    const int ldw = w2.ld();
    const int stride_a = a22.stride();
    const int stride_v = v2.stride();
    const int stride_w = w2.stride();
    T* a_ptr = a22.data_ptr();
    const T* v_ptr = v2.data_ptr();
    const T* w_ptr = w2.data_ptr();
    const int batch = a22.batch_size();

    (void)q->submit([&](sycl::handler& h) {
        h.parallel_for<UpdateVWLowerSmallKernel<T>>(
            sycl::range<2>(static_cast<size_t>(batch), static_cast<size_t>(n2) * static_cast<size_t>(n2)),
            [=](sycl::id<2> idx) {
                const int b = static_cast<int>(idx[0]);
                const int lin = static_cast<int>(idx[1]);
                const int r = lin % n2;
                const int c = lin / n2;
                if (r < c) return; // lower triangle only

                T* A = a_ptr + b * stride_a;
                const T* V = v_ptr + b * stride_v;
                const T* W = w_ptr + b * stride_w;

                // A(r,c) -= sum_k ( V(r,k) * conj(W(c,k)) + W(r,k) * conj(V(c,k)) )
                T acc = T(0);
                for (int k = 0; k < ib; ++k) {
                    const T vrk = V[r + k * ldv];
                    const T vck = V[c + k * ldv];
                    const T wrk = W[r + k * ldw];
                    const T wck = W[c + k * ldw];
                    acc += vrk * conj_if_needed(wck) + wrk * conj_if_needed(vck);
                }

                A[r + c * lda] -= acc;

                // Keep diagonal real for Hermitian matrices.
                if constexpr (internal::is_complex<T>::value) {
                    if (r == c) {
                        const T x = A[r + c * lda];
                        A[r + c * lda] = T(x.real(), typename T::value_type(0));
                    }
                }
            });
    });

    return q.get_event();
}

template <typename T>
Event sytrd_lower_local_small(Queue& q,
                              const MatrixView<T, MatrixFormat::Dense>& a,
                              const VectorView<T>& e,
                              const VectorView<T>& tau,
                              int n) {
    constexpr int WG = 64;
    const int lda = a.ld();
    const int stride_a = a.stride();
    T* a_ptr = a.data_ptr();
    T* e_ptr = e.data_ptr();
    T* tau_ptr = tau.data_ptr();
    const int stride_e = e.stride();
    const int stride_tau = tau.stride();
    const int batch = a.batch_size();

    (void)q->submit([&](sycl::handler& h) {
        // Allocate just enough local memory for the active n x n tile (plus w vector).
        auto A_local = sycl::local_accessor<T, 1>(sycl::range<1>(static_cast<size_t>(n) * static_cast<size_t>(n)), h);
        auto W_local = sycl::local_accessor<T, 1>(sycl::range<1>(static_cast<size_t>(n)), h);

        h.parallel_for<SytrdLowerLocalSmallKernel<T>>(
            sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(batch) * WG), sycl::range<1>(WG)),
            [=](sycl::nd_item<1> it) {
                const int b = static_cast<int>(it.get_group_linear_id());
                if (b >= batch) return;

                const int lane = static_cast<int>(it.get_local_linear_id());
                const sycl::group<1> g = it.get_group();

                T* A = a_ptr + b * stride_a;
                T* Eb = e_ptr + b * stride_e;
                T* Taub = tau_ptr + b * stride_tau;

                const int ld_loc = n;
                auto Al = [&](int r, int c) -> T& { return A_local[r + c * ld_loc]; };

                // Load A into local memory.
                if (lane < n) {
                    for (int c = 0; c < n; ++c) {
                        Al(lane, c) = A[lane + c * lda];
                    }
                }
                it.barrier(sycl::access::fence_space::local_space);

                // Unblocked SYTD2-style reduction (Lower): for k=0..n-2
                for (int k = 0; k < n - 1; ++k) {
                    // Form Householder reflector to annihilate A(k+2:n-1, k)
                    using Real = typename base_type<T>::type;

                    const int alpha_row = k + 1;
                    const int x0 = k + 2;

                    Real sumsq = Real(0);
                    if (lane >= x0 && lane < n) {
                        sumsq = abs2_if_complex(Al(lane, k));
                    }
                    sumsq = reduce_sum_group_real<T>(g, sumsq);

                    T alpha = T(0);
                    if (lane == alpha_row) {
                        alpha = Al(alpha_row, k);
                    }
                    alpha = sycl::group_broadcast(g, alpha, sycl::id<1>(alpha_row));

                    T tau_k = T(0);
                    T beta = alpha;
                    T scale = T(0);

                    if (lane == alpha_row) {
                        const Real xnorm = sycl::sqrt(sumsq);
                        if constexpr (internal::is_complex<T>::value) {
                            if (xnorm == Real(0) && alpha.imag() == Real(0)) {
                                tau_k = T(0);
                                beta = alpha;
                                scale = T(0);
                            } else {
                                const Real alpha_abs = sycl::hypot(alpha.real(), alpha.imag());
                                const Real beta_abs = sycl::hypot(alpha_abs, xnorm);
                                const T alpha_sign = (alpha_abs == Real(0)) ? T(1) : (alpha / alpha_abs);
                                beta = -alpha_sign * T(beta_abs);
                                tau_k = (beta - alpha) / beta;
                                scale = T(1) / (alpha - beta);
                            }
                        } else {
                            if (xnorm == Real(0)) {
                                tau_k = T(0);
                                beta = alpha;
                                scale = T(0);
                            } else {
                                beta = -sign_nonzero(alpha) * T(sycl::hypot(static_cast<Real>(alpha), xnorm));
                                tau_k = (beta - alpha) / beta;
                                scale = T(1) / (alpha - beta);
                            }
                        }

                        Eb[k] = beta;
                        Taub[k] = tau_k;

                        // Store v(0)=1 at A(k+1,k).
                        Al(alpha_row, k) = T(1);
                    }

                    tau_k = sycl::group_broadcast(g, tau_k, sycl::id<1>(alpha_row));
                    scale = sycl::group_broadcast(g, scale, sycl::id<1>(alpha_row));

                    if (tau_k != T(0)) {
                        if (lane >= x0 && lane < n) {
                            Al(lane, k) *= scale;
                        }
                    }
                    it.barrier(sycl::access::fence_space::local_space);

                    // Compute w = tau * A(k+1:n-1, k+1:n-1) * v
                    if (lane < n) {
                        T w = T(0);
                        if (lane >= alpha_row) {
                            for (int c = alpha_row; c < n; ++c) {
                                const T vc = (c == alpha_row) ? T(1) : Al(c, k);
                                w += Al(lane, c) * vc;
                            }
                            w *= tau_k;
                        }
                        W_local[lane] = w;
                    }
                    it.barrier(sycl::access::fence_space::local_space);

                    // dot = v^H * w
                    T dot_partial = T(0);
                    if (lane >= alpha_row && lane < n) {
                        const T vr = (lane == alpha_row) ? T(1) : Al(lane, k);
                        dot_partial = conj_if_needed(vr) * W_local[lane];
                    }
                    const T dot = reduce_sum_group(g, dot_partial);

                    const T alpha2 = T(-0.5) * tau_k * dot;
                    if (lane >= alpha_row && lane < n) {
                        const T vr = (lane == alpha_row) ? T(1) : Al(lane, k);
                        W_local[lane] += alpha2 * vr;
                    }
                    it.barrier(sycl::access::fence_space::local_space);

                    // Apply rank-2 update to trailing block A(alpha_row:n-1, alpha_row:n-1).
                    if (lane >= alpha_row && lane < n) {
                        const T vr = (lane == alpha_row) ? T(1) : Al(lane, k);
                        const T wr = W_local[lane];
                        for (int c = alpha_row; c <= lane; ++c) {
                            const T vc = (c == alpha_row) ? T(1) : Al(c, k);
                            const T wc = W_local[c];
                            T a_rc = Al(lane, c);
                            a_rc -= vr * conj_if_needed(wc) + wr * conj_if_needed(vc);
                            Al(lane, c) = a_rc;
                            if (lane != c) {
                                Al(c, lane) = conj_if_needed(a_rc);
                            } else if constexpr (internal::is_complex<T>::value) {
                                Al(lane, c) = T(a_rc.real(), typename T::value_type(0));
                            }
                        }
                    }
                    it.barrier(sycl::access::fence_space::local_space);
                }

                // Write back A.
                if (lane < n) {
                    for (int c = 0; c < n; ++c) {
                        A[lane + c * lda] = Al(lane, c);
                    }
                }
            });
    });

    return q.get_event();
}

template <typename T>
class RestoreTridiagKernel;

template <typename T>
Event restore_tridiag_lower(Queue& q,
                                 const MatrixView<T, MatrixFormat::Dense>& a,
                                 const VectorView<T>& d,
                                 const VectorView<T>& e,
                                 int n) {
    const int lda = a.ld();
    const int stride_a = a.stride();
    T* a_ptr = a.data_ptr();
    T* d_ptr = d.data_ptr();
    T* e_ptr = e.data_ptr();
    const int stride_d = d.stride();
    const int stride_e = e.stride();
    const int batch = a.batch_size();

    (void)q->submit([&](sycl::handler& h) {
        h.parallel_for<RestoreTridiagKernel<T>>(sycl::range<2>(static_cast<size_t>(batch), static_cast<size_t>(n)),
                                               [=](sycl::id<2> idx) {
                                                   const int b = static_cast<int>(idx[0]);
                                                   const int i = static_cast<int>(idx[1]);
                                                   T* A = a_ptr + b * stride_a;
                                                   T* Db = d_ptr + b * stride_d;
                                                   T* Eb = e_ptr + b * stride_e;

                                                   if (i < n) {
                                                       Db[i] = A[i + i * lda];
                                                   }
                                                   if (i < n - 1) {
                                                       const T ei = Eb[i];
                                                       A[(i + 1) + i * lda] = ei;
                                                       A[i + (i + 1) * lda] = conj_if_needed(ei);
                                                   }
                                               });
    });

    return q.get_event();
}

template <typename T>
inline void validate_sytrd_dims(const MatrixView<T, MatrixFormat::Dense>& a,
                               const VectorView<T>& d,
                               const VectorView<T>& e,
                               const VectorView<T>& tau) {
    if (a.rows() != a.cols()) {
        throw std::invalid_argument("sytrd_blocked: A must be square");
    }
    const int n = a.rows();
    if (d.size() != n || e.size() != std::max(0, n - 1) || tau.size() != std::max(0, n - 1)) {
        throw std::invalid_argument("sytrd_blocked: invalid d/e/tau sizes");
    }
    if (a.batch_size() != d.batch_size() || a.batch_size() != e.batch_size() || a.batch_size() != tau.batch_size()) {
        throw std::invalid_argument("sytrd_blocked: batch size mismatch");
    }
    if (a.batch_size() < 1) {
        throw std::invalid_argument("sytrd_blocked: invalid batch size");
    }
}

} // namespace

template <Backend B, typename T>
size_t sytrd_blocked_buffer_size(Queue& ctx,
                                 const MatrixView<T, MatrixFormat::Dense>& a,
                                 const VectorView<T>& d,
                                 const VectorView<T>& e,
                                 const VectorView<T>& tau,
                                 Uplo uplo,
                                 int32_t block_size) {
    (void)uplo;
    validate_sytrd_dims(a, d, e, tau);

    const int n = a.rows();
    const int batch = a.batch_size();
    const int nb = std::max<int>(1, block_size);

    size_t size = 0;
    // W workspace: n x nb per batch
    size += BumpAllocator::allocation_size<T>(ctx, static_cast<size_t>(n) * static_cast<size_t>(nb) * static_cast<size_t>(batch));
    return size;
}

template <Backend B, typename T>
Event sytrd_blocked(Queue& ctx,
                    const MatrixView<T, MatrixFormat::Dense>& a_in,
                    const VectorView<T>& d_out,
                    const VectorView<T>& e_out,
                    const VectorView<T>& tau_out,
                    Uplo uplo,
                    const Span<std::byte>& ws,
                    int32_t block_size) {
    validate_sytrd_dims(a_in, d_out, e_out, tau_out);

    if (!ctx.in_order()) {
        throw std::runtime_error("sytrd_blocked: requires an in-order Queue");
    }

    const int n = a_in.rows();
    const int batch = a_in.batch_size();
    const int nb = std::max<int>(1, block_size);

    if (uplo != Uplo::Lower) {
        throw std::runtime_error("sytrd_blocked: only Uplo::Lower is implemented");
    }

    if (n <= 32) {
        // For small matrices prefer the CTA (unblocked) path.
        return sytrd_cta<B, T>(ctx, a_in, d_out, e_out, tau_out, uplo, Span<std::byte>(), /*cta_wg_size_multiplier=*/1);
    }

    // Optional: local-memory unblocked kernel.
    // Auto-enabled for float where it improves the n=33..64 crossover; other types require
    // an explicit env override because it can regress performance.
    //
    // NOTE: The DPC++ CUDA plugin has been observed to abort in
    // detail::adjustNDRangePerKernel for this kernel shape. Prefer stability
    // and fall back to the regular blocked path on CUDA.
    if (n <= 64) {
        auto env_truthy = [](const char* v) -> bool {
            if (!v) return false;
            const std::string s(v);
            return (s == "1" || s == "true" || s == "TRUE" || s == "on" || s == "ON");
        };
        const bool force_local = env_truthy(std::getenv("BATCHLAS_SYTRD_FORCE_LOCAL_SMALL"));
        const bool debug_small = env_truthy(std::getenv("BATCHLAS_DEBUG_SYTRD_SMALL"));

        const size_t local_mem_bytes = ctx.device().get_property(DeviceProperty::LOCAL_MEM_SIZE);
        const size_t max_wg_size = ctx.device().get_property(DeviceProperty::MAX_WORK_GROUP_SIZE);

        constexpr size_t WG = 64;
        const size_t elems = static_cast<size_t>(n) * static_cast<size_t>(n) + static_cast<size_t>(n);
        const size_t bytes_needed = elems * sizeof(T);

        const bool props_ok = (max_wg_size >= WG && (local_mem_bytes == 0 || local_mem_bytes >= bytes_needed));
        if (debug_small) {
            static std::atomic<bool> printed{false};
            if (!printed.exchange(true)) {
                std::cerr << "[sytrd_blocked] n=" << n << " batch=" << batch << " nb=" << nb
                          << " max_wg_size=" << max_wg_size << " local_mem_bytes=" << local_mem_bytes
                          << " bytes_needed=" << bytes_needed << " props_ok=" << (props_ok ? 1 : 0)
                          << " force_local=" << (force_local ? 1 : 0) << "\n";
            }
        }

        constexpr bool allow_local_small = (B != Backend::CUDA);
        const bool auto_local = allow_local_small && std::is_same_v<T, float>;
        if ((auto_local && props_ok) || (allow_local_small && force_local && props_ok)) {
            {
                BATCHLAS_KERNEL_TRACE_SCOPE("sytrd_blocked.local_small");
                (void)sytrd_lower_local_small<T>(ctx, a_in, e_out, tau_out, n);
            }
            // Final: write D from diagonal and ensure tridiagonal entries are present.
            {
                BATCHLAS_KERNEL_TRACE_SCOPE("sytrd_blocked.restore_tridiag");
                (void)restore_tridiag_lower<T>(ctx, a_in, d_out, e_out, n);
            }
            return ctx.get_event();
        }
    }

    MatrixView<T, MatrixFormat::Dense> A = a_in;
    VectorView<T> D = d_out;
    VectorView<T> E = e_out;
    VectorView<T> TAU = tau_out;

    BumpAllocator pool(ws);
    auto Wbuf = pool.allocate<T>(ctx, static_cast<size_t>(n) * static_cast<size_t>(nb) * static_cast<size_t>(batch));
    MatrixView<T, MatrixFormat::Dense> Wmat(Wbuf.data(), n, nb, n, n * nb, batch);

    const int k = n - 1;

    for (int j0 = 0; j0 < k; j0 += nb) {
        const int ib = std::min(nb, k - j0);

        {
            BATCHLAS_KERNEL_TRACE_SCOPE("sytrd_blocked.latrd_lower_panel");
            auto A_panel = A({j0, SliceEnd()}, {j0, SliceEnd()});
            auto E_panel = E(Slice(j0, j0 + ib));
            auto TAU_panel = TAU(Slice(j0, j0 + ib));
            auto W_panel = Wmat({j0, SliceEnd()}, {0, ib});
            (void)latrd_lower_panel<B, T>(ctx, A_panel, E_panel, TAU_panel, W_panel);
        }

        const int j2 = j0 + ib;
        const int n2 = n - j2;
        if (n2 > 0) {
            auto A22 = A({j2, SliceEnd()}, {j2, SliceEnd()});
            auto V2 = A({j2, SliceEnd()}, {j0, j0 + ib});
            auto W2 = Wmat({j2, SliceEnd()}, {0, ib});

            // A22 -= V2*W2^H + W2*V2^H
            {
                BATCHLAS_KERNEL_TRACE_SCOPE("sytrd_blocked.update_vw");
                // For very small trailing blocks, two tiny GEMMs can be dominated by library overhead.
                // Use a simple SYCL kernel to update the Hermitian lower triangle instead.
                if (n2 <= 128) {
                    (void)update_vw_lower_small<T>(ctx, V2, W2, A22);
                } else {
                    gemm<B>(ctx, V2, W2, A22, T(-1), T(1), Transpose::NoTrans, Transpose::ConjTrans);
                    gemm<B>(ctx, W2, V2, A22, T(-1), T(1), Transpose::NoTrans, Transpose::ConjTrans);
                }
            }
        }
    }

    // Final: write D from diagonal and ensure tridiagonal entries are present.
    {
        BATCHLAS_KERNEL_TRACE_SCOPE("sytrd_blocked.restore_tridiag");
        (void)restore_tridiag_lower<T>(ctx, A, D, E, n);
    }

    return ctx.get_event();
}

#define SYTRD_BLOCKED_INSTANTIATE(back, fp) \
    template Event sytrd_blocked<back, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        const VectorView<fp>&, \
        const VectorView<fp>&, \
        const VectorView<fp>&, \
        Uplo, \
        const Span<std::byte>&, \
        int32_t); \
    template size_t sytrd_blocked_buffer_size<back, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        const VectorView<fp>&, \
        const VectorView<fp>&, \
        const VectorView<fp>&, \
        Uplo, \
        int32_t);

#if BATCHLAS_HAS_CUDA_BACKEND
SYTRD_BLOCKED_INSTANTIATE(Backend::CUDA, float)
SYTRD_BLOCKED_INSTANTIATE(Backend::CUDA, double)
SYTRD_BLOCKED_INSTANTIATE(Backend::CUDA, std::complex<float>)
SYTRD_BLOCKED_INSTANTIATE(Backend::CUDA, std::complex<double>)
#endif

#if BATCHLAS_HAS_ROCM_BACKEND
SYTRD_BLOCKED_INSTANTIATE(Backend::ROCM, float)
SYTRD_BLOCKED_INSTANTIATE(Backend::ROCM, double)
SYTRD_BLOCKED_INSTANTIATE(Backend::ROCM, std::complex<float>)
SYTRD_BLOCKED_INSTANTIATE(Backend::ROCM, std::complex<double>)
#endif

#if BATCHLAS_HAS_HOST_BACKEND
SYTRD_BLOCKED_INSTANTIATE(Backend::NETLIB, float)
SYTRD_BLOCKED_INSTANTIATE(Backend::NETLIB, double)
SYTRD_BLOCKED_INSTANTIATE(Backend::NETLIB, std::complex<float>)
SYTRD_BLOCKED_INSTANTIATE(Backend::NETLIB, std::complex<double>)
#endif

#undef SYTRD_BLOCKED_INSTANTIATE

} // namespace batchlas
