#include <blas/device.hh>
#include <blas/extensions.hh>
#include <blas/functions.hh>
#include <blas/matrix.hh>
#include <internal/sytrd_blocked.hh>
#include <util/mempool.hh>
#include <util/sycl-local-accessor-helpers.hh>

#include <batchlas/backend_config.h>
#include <batchlas/tuning_params.hh>

#include "../expansion_budget.hh"
#include "../math-helpers.hh"
#include "../queue.hh"
#include "../util/template-instantiations.hh"

#include <algorithm>
#include <atomic>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <stdexcept>
#include <type_traits>
#include <util/env.hh>

namespace batchlas {

namespace {

// ---------------------------------------------------------------------------
// Environment-variable dispatch helpers
// ---------------------------------------------------------------------------

// Returns true when BATCHLAS_SYTRD_IMPL=device, false otherwise (legacy default).
inline bool use_device_sytrd() {
    static const bool result = []() {
        const char* v = std::getenv("BATCHLAS_SYTRD_IMPL");
        return v && std::string(v) == "device";
    }();
    return result;
}

enum class SytrdTrailingUpdateMode {
    Gemm,
    Rank2k,
    Default,
};

// Override for the trailing update; unset means the per-backend default below.
// Kept in both directions so a regression can be bisected against either route
// without a rebuild.
//
// "syr2k" and "her2k" are the real and complex spellings of the same Rank2k
// route -- one name each, because the value a bisect wants to pin is the route,
// not the primitive. Anything unrecognised falls through to Default silently,
// so a run pinned with a typo is a default-against-default A/B; that is why
// both spellings are accepted rather than only the one this file used to call.
inline SytrdTrailingUpdateMode sytrd_trailing_update_mode() {
    const char* v = std::getenv("BATCHLAS_SYTRD_TRAILING_UPDATE");
    if (!v) return SytrdTrailingUpdateMode::Default;

    const std::string s(v);
    if (s == "syr2k" || s == "SYR2K" || s == "her2k" || s == "HER2K" ||
        s == "rank2k" || s == "RANK2K") {
        return SytrdTrailingUpdateMode::Rank2k;
    }
    if (s == "gemm" || s == "GEMM") {
        return SytrdTrailingUpdateMode::Gemm;
    }
    return SytrdTrailingUpdateMode::Default;
}

// Used by the device sytrd_blocked_impl to allow per-run WG hint overrides.
inline int32_t latrd_lower_panel_wg_hint_override(int32_t full_n, int32_t batch) {
    (void)batch;
    const int32_t fallback = tuning::latrd_lower_panel_wg_hint_for_n(full_n);
    const char* v = std::getenv("BATCHLAS_LATRD_LOWER_PANEL_WG_HINT");
    if (!v || *v == '\0') {
        return fallback;
    }
    const int value = std::atoi(v);
    if (value == 64 || value == 128 || value == 256) {
        return value;
    }
    return fallback;
}

// ---------------------------------------------------------------------------
// Shared helpers (used by both legacy and device kernels)
// ---------------------------------------------------------------------------

// conj_if_needed delegates to device::detail::conj so it works in both host
// and device code and produces the same result as the manual implementation.
template <typename U>
inline U conj_if_needed(const U& x) {
    return batchlas::device::detail::conj(x);
}

// ---------------------------------------------------------------------------
// Helpers used exclusively by the legacy kernels
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Helpers used exclusively by the device kernels
// ---------------------------------------------------------------------------

template <typename T>
inline std::size_t sytrd_rank2k_local_size(const Queue& q) {
    const std::size_t max_work_group_size = q.device().get_property(DeviceProperty::MAX_WORK_GROUP_SIZE);
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, std::complex<float>>) {
        return std::min<std::size_t>(256, max_work_group_size);
    }
    return std::min<std::size_t>(128, max_work_group_size);
}

// ---------------------------------------------------------------------------
// Kernel name tags — "Legacy" variants for the manual-loop kernels so both
// implementations can coexist in the same translation unit without SYCL
// kernel-name collisions.
// ---------------------------------------------------------------------------
template <typename T> class UpdateVWLowerSmallKernelLegacy;
template <typename T> class SytrdLowerLocalSmallKernelLegacy;

template <typename T> class UpdateVWLowerSmallKernel;
template <typename T> class SytrdLowerLocalSmallKernel;

// Shared between both paths (implementation is identical).
template <typename T> class RestoreTridiagKernel;

// ---------------------------------------------------------------------------
// update_vw_lower_small — legacy (manual rank-2 update)
// ---------------------------------------------------------------------------
template <typename T>
Event update_vw_lower_small_legacy(Queue& q,
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
        h.parallel_for<UpdateVWLowerSmallKernelLegacy<T>>(
            sycl::range<2>(static_cast<size_t>(batch), static_cast<size_t>(n2) * static_cast<size_t>(n2)),
            [=](sycl::id<2> idx) {
                const int b = static_cast<int>(idx[0]);
                const int lin = static_cast<int>(idx[1]);
                const int r = lin % n2;
                const int c = lin / n2;
                if (r < c) return;

                T* A = a_ptr + b * stride_a;
                const T* V = v_ptr + b * stride_v;
                const T* W = w_ptr + b * stride_w;

                T acc = T(0);
                for (int k = 0; k < ib; ++k) {
                    const T vrk = V[r + k * ldv];
                    const T vck = V[c + k * ldv];
                    const T wrk = W[r + k * ldw];
                    const T wck = W[c + k * ldw];
                    acc += vrk * conj_if_needed(wck) + wrk * conj_if_needed(vck);
                }

                A[r + c * lda] -= acc;

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

// ---------------------------------------------------------------------------
// update_vw_lower_small — device BLAS (her2k)
// ---------------------------------------------------------------------------
template <typename T>
Event update_vw_lower_small_device(Queue& q,
                                   const MatrixView<T, MatrixFormat::Dense>& v2,
                                   const MatrixView<T, MatrixFormat::Dense>& w2,
                                   const MatrixView<T, MatrixFormat::Dense>& a22) {
    const int batch = a22.batch_size();
    const std::size_t local_size = sytrd_rank2k_local_size<T>(q);
    const auto launch = batchlas::device::make_group_launch_info(static_cast<int>(local_size));
    std::size_t workspace_elements = 0;

    if (batch > 0) {
        const int extent = a22.rows();
        const int contract_extent = v2.cols();
        workspace_elements = batchlas::device::her2k_workspace_elements<T,
            batchlas::device::DeviceBlasPolicy::Auto, Uplo::Lower, Transpose::NoTrans>(
            launch, extent, contract_extent);
    }

    (void)q->submit([&](sycl::handler& h) {
        const auto v_view = v2.kernel_view();
        const auto w_view = w2.kernel_view();
        const auto a_view = a22.kernel_view();

        sycl::local_accessor<T, 1> workspace(sycl::range<1>(std::max<std::size_t>(workspace_elements, 1)), h);
        h.parallel_for<UpdateVWLowerSmallKernel<T>>(
            sycl::nd_range<1>(sycl::range<1>(static_cast<std::size_t>(batch) * local_size),
                              sycl::range<1>(local_size)),
            [=](sycl::nd_item<1> item) {
                const int b = static_cast<int>(item.get_group(0));
                if (b >= batch) return;

                const auto vb = v_view.batch_item(b);
                const auto wb = w_view.batch_item(b);
                const auto ab = a_view.batch_item(b);
                T* workspace_ptr = workspace_elements == 0
                    ? static_cast<T*>(nullptr)
                    : batchlas::util::get_raw_ptr(workspace);
                batchlas::device::her2k<batchlas::device::DeviceBlasPolicy::Auto,
                                        Uplo::Lower, Transpose::NoTrans>(
                    item.get_group(), vb, wb, ab, T(-1), T(1), workspace_ptr);
            });
    });

    return q.get_event();
}

// ---------------------------------------------------------------------------
// sytrd_lower_local_small — legacy (manual SYTD2-style kernel)
// ---------------------------------------------------------------------------
template <typename T>
Event sytrd_lower_local_small_legacy(Queue& q,
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
        auto A_local = sycl::local_accessor<T, 1>(sycl::range<1>(static_cast<size_t>(n) * static_cast<size_t>(n)), h);
        auto W_local = sycl::local_accessor<T, 1>(sycl::range<1>(static_cast<size_t>(n)), h);

        h.parallel_for<SytrdLowerLocalSmallKernelLegacy<T>>(
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

                if (lane < n) {
                    for (int c = 0; c < n; ++c) {
                        Al(lane, c) = A[lane + c * lda];
                    }
                }
                it.barrier(sycl::access::fence_space::local_space);

                for (int k = 0; k < n - 1; ++k) {
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

                if (lane < n) {
                    for (int c = 0; c < n; ++c) {
                        A[lane + c * lda] = Al(lane, c);
                    }
                }
            });
    });

    return q.get_event();
}

// ---------------------------------------------------------------------------
// sytrd_lower_local_small — device BLAS (hemv + her2k)
// ---------------------------------------------------------------------------
template <typename T>
Event sytrd_lower_local_small_device(Queue& q,
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
    const auto launch = batchlas::device::make_group_launch_info(WG);
    std::size_t workspace_elements = 0;

    if (batch > 0 && n > 1) {
        const int extent = n - 1;
        workspace_elements = std::max(
            workspace_elements,
            batchlas::device::hemv_workspace_elements<T,
                batchlas::device::DeviceBlasPolicy::Auto, Uplo::Lower>(launch, extent));
        workspace_elements = std::max(
            workspace_elements,
            batchlas::device::her2k_workspace_elements<T,
                batchlas::device::DeviceBlasPolicy::Auto, Uplo::Lower, Transpose::NoTrans>(
                launch, extent, 1));
    }

    (void)q->submit([&](sycl::handler& h) {
        auto A_local = sycl::local_accessor<T, 1>(sycl::range<1>(static_cast<size_t>(n) * static_cast<size_t>(n)), h);
        auto W_local = sycl::local_accessor<T, 1>(sycl::range<1>(static_cast<size_t>(n)), h);
        sycl::local_accessor<T, 1> workspace(sycl::range<1>(std::max<std::size_t>(workspace_elements, 1)), h);

        h.parallel_for<SytrdLowerLocalSmallKernel<T>>(
            sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(batch) * WG), sycl::range<1>(WG)),
            [=](sycl::nd_item<1> it) {
                const int b = static_cast<int>(it.get_group_linear_id());
                if (b >= batch) return;

                const int lane = static_cast<int>(it.get_local_linear_id());
                const sycl::group<1> g = it.get_group();
                T* Al_ptr = A_local.template get_multi_ptr<sycl::access::decorated::no>().get();
                T* W_ptr  = W_local.template get_multi_ptr<sycl::access::decorated::no>().get();
                T* workspace_ptr = workspace_elements == 0
                    ? static_cast<T*>(nullptr)
                    : batchlas::util::get_raw_ptr(workspace);

                T* A    = a_ptr   + b * stride_a;
                T* Eb   = e_ptr   + b * stride_e;
                T* Taub = tau_ptr + b * stride_tau;

                const int ld_loc = n;
                auto Al = [&](int r, int c) -> T& { return A_local[r + c * ld_loc]; };

                if (lane < n) {
                    for (int c = 0; c < n; ++c) {
                        Al(lane, c) = A[lane + c * lda];
                    }
                }
                it.barrier(sycl::access::fence_space::local_space);

                for (int k = 0; k < n - 1; ++k) {
                    const int alpha_row = k + 1;
                    const int x0 = k + 2;
                    const int tail = n - alpha_row;
                    const bool lane_in_tail = (lane >= alpha_row && lane < n);
                    T alpha = Al(alpha_row, k);
                    const T tau_k = internal::larfg(g, alpha,
                        VectorView<T>(Al_ptr + x0 + k * ld_loc, n - x0));
                    if (lane == 0) {
                        Al(alpha_row, k) = T(1);
                    }
                    it.barrier(sycl::access::fence_space::local_space);

                    auto trailing_view = KernelMatrixView<T, MatrixFormat::Dense>(
                        Al_ptr + alpha_row + alpha_row * ld_loc, tail, tail, ld_loc, ld_loc * n);
                    auto v_view = VectorView<T>(Al_ptr + alpha_row + k * ld_loc, tail);
                    auto w_view = VectorView<T>(W_ptr + alpha_row, tail);
                    batchlas::device::hemv<batchlas::device::DeviceBlasPolicy::Auto, Uplo::Lower>(
                        g, trailing_view, v_view, w_view, tau_k, T(0), workspace_ptr);
                    it.barrier(sycl::access::fence_space::local_space);

                    auto v_vec = VectorView<T>(Al_ptr + alpha_row + k * ld_loc, tail);
                    auto w_vec = VectorView<T>(W_ptr + alpha_row, tail);
                    const T dot = batchlas::device::dotc(g, v_vec, w_vec);

                    const T alpha2 = T(-0.5) * tau_k * dot;
                    batchlas::device::axpy(g, w_vec, v_vec, alpha2);
                    it.barrier(sycl::access::fence_space::local_space);

                    auto v_matrix = KernelMatrixView<T, MatrixFormat::Dense>(
                        Al_ptr + alpha_row + k * ld_loc, tail, 1, ld_loc, ld_loc);
                    auto w_matrix = KernelMatrixView<T, MatrixFormat::Dense>(
                        W_ptr + alpha_row, tail, 1, tail, tail);
                    batchlas::device::her2k<batchlas::device::DeviceBlasPolicy::Auto,
                                            Uplo::Lower, Transpose::NoTrans>(
                        g, v_matrix, w_matrix, trailing_view, T(-1), T(1), workspace_ptr);
                    if constexpr (internal::is_complex<T>::value) {
                        it.barrier(sycl::access::fence_space::local_space);
                        if (lane_in_tail) {
                            for (int c = alpha_row; c < lane; ++c) {
                                Al(c, lane) = conj_if_needed(Al(lane, c));
                            }
                        }
                    }
                    it.barrier(sycl::access::fence_space::local_space);

                    if (lane == 0) {
                        Eb[k]   = alpha;
                        Taub[k] = tau_k;
                    }
                }

                if (lane < n) {
                    for (int c = 0; c < n; ++c) {
                        A[lane + c * lda] = Al(lane, c);
                    }
                }
            });
    });

    return q.get_event();
}

// ---------------------------------------------------------------------------
// restore_tridiag_lower — identical in both paths
// ---------------------------------------------------------------------------
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
        h.parallel_for<RestoreTridiagKernel<T>>(
            sycl::range<2>(static_cast<size_t>(batch), static_cast<size_t>(n)),
            [=](sycl::id<2> idx) {
                const int b = static_cast<int>(idx[0]);
                const int i = static_cast<int>(idx[1]);
                T* A  = a_ptr + b * stride_a;
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

inline int resolved_nb_sytrd(int32_t block_size, int compiled_nb) {
    if (compiled_nb > 0) {
        return compiled_nb;
    }
    return std::max<int>(1, block_size);
}

// Single description of sytrd_blocked's workspace; see workspace_bytes() in
// util/mempool.hh.
template <typename T>
MatrixView<T, MatrixFormat::Dense> sytrd_blocked_layout(Queue& ctx,
                                                        BumpAllocator& pool,
                                                        int n,
                                                        int nb,
                                                        int batch) {
    auto Wbuf = pool.allocate<T>(ctx, static_cast<size_t>(n) * static_cast<size_t>(nb) * static_cast<size_t>(batch));
    return MatrixView<T, MatrixFormat::Dense>(Wbuf.data(), n, nb, n, n * nb, batch);
}

template <Backend B, typename T, int NB>
size_t sytrd_blocked_buffer_size_impl(Queue& ctx,
                                      const MatrixView<T, MatrixFormat::Dense>& a,
                                      const VectorView<T>& d,
                                      const VectorView<T>& e,
                                      const VectorView<T>& tau,
                                      Uplo uplo,
                                      int32_t block_size) {
    (void)uplo;
    (void)d;
    (void)e;
    (void)tau;

    const int n = a.rows();
    const int batch = a.batch_size();
    const int nb = resolved_nb_sytrd(block_size, NB);

    return workspace_bytes([&](BumpAllocator& pool) {
        return sytrd_blocked_layout<T>(ctx, pool, n, nb, batch);
    });
}

template <Backend B, typename T, int NB>
Event sytrd_blocked_impl(Queue& ctx,
                         const MatrixView<T, MatrixFormat::Dense>& a_in,
                         const VectorView<T>& d_out,
                         const VectorView<T>& e_out,
                         const VectorView<T>& tau_out,
                         Uplo uplo,
                         const Span<std::byte>& ws,
                         int32_t block_size) {
    const int n = a_in.rows();
    const int batch = a_in.batch_size();
    const int nb = resolved_nb_sytrd(block_size, NB);

    if (uplo != Uplo::Lower) {
        throw std::runtime_error("sytrd_blocked: only Uplo::Lower is implemented");
    }

    const bool is_legacy = !use_device_sytrd();

    if (n <= 32) {
        bool has_sg32 = false;
        try {
            const auto sizes = ctx->get_device().get_info<sycl::info::device::sub_group_sizes>();
            has_sg32 = std::find(sizes.begin(), sizes.end(), 32) != sizes.end();
        } catch (...) {
            has_sg32 = false;
        }
        if (has_sg32) {
            return sytrd_cta<B, T>(ctx, a_in, d_out, e_out, tau_out, uplo, Span<std::byte>(), /*cta_wg_size_multiplier=*/1);
        }
    }

    if (n <= 64) {
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
                if (is_legacy)
                    (void)sytrd_lower_local_small_legacy<T>(ctx, a_in, e_out, tau_out, n);
                else
                    (void)sytrd_lower_local_small_device<T>(ctx, a_in, e_out, tau_out, n);
            }
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
    MatrixView<T, MatrixFormat::Dense> Wmat = sytrd_blocked_layout<T>(ctx, pool, n, nb, batch);

    const int k = n - 1;
    const char* fuse_env = std::getenv("BATCHLAS_SYTRD_FUSE_PANEL_UPDATE");
    const bool fuse_override_on  = env_truthy(fuse_env);
    const bool fuse_override_off = env_falsy(fuse_env);
    const bool fuse_default = is_legacy
        ? ((B == Backend::CUDA) && (n == 256))
        : tuning::sytrd_fuse_panel_update_for_n(n);
    const bool enable_fused_panel_update = fuse_override_on || (!fuse_override_off && fuse_default);
    // A22 -= V W^H + W V^H is one syr2k, and syr2k touches only the triangle
    // the panel loop goes on to read. Measured on RTX 4090 / sm_89, float,
    // against the two full n2 x n2 GEMMs it replaces:
    //
    //   the update alone, over the shapes the panel loop produces, is 3.4-3.6x
    //   faster;
    //   end to end, n=512 batch=1024 went 264/253/248 ms -> 228/227/232 at
    //   nb=16/24/32, and n=256 batch=2048 went 34.3/34.6/37.0 -> 27.0/30.6/34.0.
    //
    // CUDA and float only, and that is not conservatism: syrk/syr2k reach a
    // batched kernel only through the custom float route. Everything else falls
    // to syr2k_vendor_impl, which is a host loop issuing one cublasXsyr2k per
    // batch member -- in double that measured 7.8x *slower* than the GEMM pair
    // at n=256 batch=1024, i.e. the whole win inverts.
    //
    // complex<float> is admitted too, because her2k is a different function
    // with a different backend route, not syr2k with a conjugate. Its fast route
    // (cublas.cc:644-665) is one batched gemm_vendor into scratch followed by
    // accumulate_hermitian<TwoSided=true>, which is *half* the arithmetic of the
    // two GEMMs it replaces rather than twice it: alpha*A*B^H and
    // conj(alpha)*B*A^H are conjugate transposes of one another, so the fold
    // manufactures the second term from the first. Worth chasing because vendor
    // GEMM is 34.6% of the cfloat solve at n=256 and 14.4% at n=512, and the
    // trailing update is roughly half of that.
    //
    // OPEN: the crossover behind her2k_gemm_preferred (cublas.cc:415-428) was
    // swept over square rank-k shapes. The panel loop issues a *narrow* one --
    // k = ib = nb in {16,24,32} against n2 up to 480 -- where the GEMM is near
    // bandwidth-bound and the fold adds an n2^2*batch write plus read the two
    // direct GEMMs never pay. The halved arithmetic may not survive that. Awaits
    // an A/B of her2k against the GEMM pair at n2 in {224,480}, k in {16,24,32},
    // cfloat, before this is trusted beyond the shapes it was measured at.
    //
    // complex<double> is deliberately left out: it would reach the same fast
    // route, but its scratch is 16 bytes per element, halving the headroom in
    // the fit check below, and none of it has been measured. Admit it when it
    // has been -- guessing is how the 7.8x inversion above got written down.
    constexpr bool rank2k_trailing_update_supported =
        (B == Backend::CUDA) &&
        (std::is_same_v<T, float> || std::is_same_v<T, std::complex<float>>);
    const bool use_rank2k_trailing_update =
        rank2k_trailing_update_supported &&
        (sytrd_trailing_update_mode() != SytrdTrailingUpdateMode::Gemm);
    const int32_t latrd_wg_hint = is_legacy
        ? tuning::latrd_lower_panel_wg_hint()
        : latrd_lower_panel_wg_hint_override(n, batch);

    for (int j0 = 0; j0 < k; j0 += nb) {
        const int ib = std::min(nb, k - j0);

        const int j2 = j0 + ib;
        const int n2 = n - j2;
        const bool fuse_this_panel = enable_fused_panel_update && n2 > 0 && n2 <= 128;

        {
            BATCHLAS_KERNEL_TRACE_SCOPE(fuse_this_panel ? "sytrd_blocked.panel_fused"
                                                        : "sytrd_blocked.panel_only");
            auto A_panel  = A({j0, SliceEnd()}, {j0, SliceEnd()});
            auto E_panel  = E(Slice(j0, j0 + ib));
            auto TAU_panel = TAU(Slice(j0, j0 + ib));
            auto W_panel  = Wmat({j0, SliceEnd()}, {0, ib});
            (void)latrd_lower_panel<B, T>(ctx,
                                          A_panel, E_panel, TAU_panel, W_panel,
                                          latrd_wg_hint, fuse_this_panel);
        }

        if (n2 > 0 && !fuse_this_panel) {
            auto A22 = A({j2, SliceEnd()}, {j2, SliceEnd()});
            auto V2  = A({j2, SliceEnd()}, {j0, j0 + ib});
            auto W2  = Wmat({j2, SliceEnd()}, {0, ib});

            if (n2 <= 128) {
                BATCHLAS_KERNEL_TRACE_SCOPE("sytrd_blocked.update_vw_small");
                if (is_legacy)
                    (void)update_vw_lower_small_legacy<T>(ctx, V2, W2, A22);
                else
                    (void)update_vw_lower_small_device<T>(ctx, V2, W2, A22);
            } else {
                bool rank2k_issued = false;
                if constexpr (rank2k_trailing_update_supported) {
                    if (use_rank2k_trailing_update) {
                        if constexpr (internal::is_complex<T>::value) {
                            // her2k's fast route needs an n2 x n2 x batch scratch
                            // expansion; when that does not fit it drops to a host
                            // loop over cublasCher2k (cublas.cc:676-691), which is
                            // structurally the same route measured 7.8x slower
                            // than the GEMM pair above. So ask the backend's own
                            // predicate first and keep the GEMM pair as the answer
                            // when it says no -- a call site that guessed here
                            // would reinstate that inversion silently.
                            //
                            // Per panel, not hoisted: n2 shrinks every iteration,
                            // so an early panel can fail to fit while later ones
                            // fit, and taking the GEMM pair for just those panels
                            // is the correct behaviour.
                            //
                            // It fits with room at every shape syev routes to
                            // blocked. expanded_ld<complex<float>>(n2) rounds n2 up
                            // to a multiple of 2, so the scratch is
                            // ~n2^2*batch*8 bytes against a GLOBAL_MEM_SIZE/4
                            // budget, ~6.0 GiB on a 24 GiB 4090: n=448 batch=585
                            // (the cfloat blocked/vendor crossover, syev.hh:594)
                            // needs 0.75 GiB and n=512 batch=1024 needs 1.76 GiB,
                            // i.e. >=3.4x headroom. The ceiling is crossed around
                            // n2^2*batch > 8.0e8 elements -- forced blocked at
                            // n=1024 batch=1024 (7.51 GiB) or n=2048 batch=256
                            // (7.75 GiB) -- which is outside the routed region but
                            // reachable by pinning the provider, and is exactly
                            // where an unguarded call would invert.
                            //
                            // The lease is taken per panel inside her2k_vendor and
                            // released before the next one, so the peak is one
                            // panel's scratch, not the loop's sum. On an
                            // out-of-order Queue that route also drains the device
                            // between its GEMM and its fold (cublas.cc:661-663),
                            // once per panel; the benchmarks all build in-order
                            // queues and never see it.
                            const std::size_t her2k_scratch_bytes =
                                backend::detail::expanded_workspace_bytes<T>(ctx, n2, batch);
                            if (backend::detail::expansion_fits(ctx, n2, batch, her2k_scratch_bytes)) {
                                BATCHLAS_KERNEL_TRACE_SCOPE("sytrd_blocked.update_vw_her2k");
                                her2k<B>(ctx, V2, W2, A22,
                                         {.alpha = T(-1), .beta = float_t<T>(1)});
                                rank2k_issued = true;
                            }
                        } else {
                            BATCHLAS_KERNEL_TRACE_SCOPE("sytrd_blocked.update_vw_syr2k");
                            syr2k<B>(ctx, V2, W2, A22, {.alpha = T(-1), .beta = T(1)});
                            rank2k_issued = true;
                        }
                        // No symmetrize: nothing downstream reads A's upper
                        // triangle, so leaving it stale is not observable.
                        // Checked across every reader, not assumed --
                        //   latrd_lower_panel, all three variants: the symmetric
                        //     matvec is the only place tempted to cross the
                        //     diagonal, and all three split it at c == r, taking
                        //     Ab(r,c) for c <= r and conj(Ab(c,r)) for c > r. The
                        //     device variant reaches it through
                        //     device::hemv<Uplo::Lower>, which mirrors the same
                        //     way. The fused trailing update guards with
                        //     `if (r < c) continue` (legacy, grid) or
                        //     device::her2k<Uplo::Lower>.
                        //   restore_tridiag_lower: reads the diagonal, and only
                        //     writes the superdiagonal.
                        // The GEMM pair happened to leave a valid upper triangle
                        // as a side effect; that was never a contract anything
                        // depended on.
                        //
                        // her2k additionally forces imag(diag) = 0 on the block it
                        // writes (cublas.cc:492), which the GEMM pair does not --
                        // it leaves whatever roundoff accumulated there. That is
                        // the correct value for a Hermitian operand and it is
                        // unobservable downstream: syev_blocked.cc:217 takes
                        // D(i,b).real(), and the n2 <= 128 device path above
                        // already does the same through device::her2k. It does
                        // mean cfloat results move in the last bits against the
                        // GEMM pair -- latrd's hemv consumes the diagonal -- so
                        // expect drift, not bitwise equality, when A/B-ing the
                        // two routes.
                    }
                }
                if (!rank2k_issued) {
                    {
                        BATCHLAS_KERNEL_TRACE_SCOPE("sytrd_blocked.update_vw_gemm_vw");
                        gemm<B>(ctx,
                                V2,
                                W2,
                                A22,
                                {.alpha = T(-1), .beta = T(1), .transB = Transpose::ConjTrans});
                    }
                    {
                        BATCHLAS_KERNEL_TRACE_SCOPE("sytrd_blocked.update_vw_gemm_wv");
                        gemm<B>(ctx,
                                W2,
                                V2,
                                A22,
                                {.alpha = T(-1), .beta = T(1), .transB = Transpose::ConjTrans});
                    }
                }
            }
        }
    }

    {
        BATCHLAS_KERNEL_TRACE_SCOPE("sytrd_blocked.restore_tridiag");
        (void)restore_tridiag_lower<T>(ctx, A, D, E, n);
    }

    return ctx.get_event();
}

template <Backend B, typename T>
size_t sytrd_blocked_buffer_size(Queue& ctx,
                                 const MatrixView<T, MatrixFormat::Dense>& a,
                                 const VectorView<T>& d,
                                 const VectorView<T>& e,
                                 const VectorView<T>& tau,
                                 Uplo uplo,
                                 int32_t block_size) {
    validate_sytrd_dims(a, d, e, tau);

    const int nb = std::max<int>(1, block_size);
    switch (nb) {
        case 8:
            return sytrd_blocked_buffer_size_impl<B, T, 8>(ctx, a, d, e, tau, uplo, block_size);
        case 16:
            return sytrd_blocked_buffer_size_impl<B, T, 16>(ctx, a, d, e, tau, uplo, block_size);
        case 32:
            return sytrd_blocked_buffer_size_impl<B, T, 32>(ctx, a, d, e, tau, uplo, block_size);
        case 64:
            return sytrd_blocked_buffer_size_impl<B, T, 64>(ctx, a, d, e, tau, uplo, block_size);
        default:
            return sytrd_blocked_buffer_size_impl<B, T, -1>(ctx, a, d, e, tau, uplo, block_size);
    }
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

    const int nb = std::max<int>(1, block_size);
    switch (nb) {
        case 8:
            return sytrd_blocked_impl<B, T, 8>(ctx, a_in, d_out, e_out, tau_out, uplo, ws, block_size);
        case 16:
            return sytrd_blocked_impl<B, T, 16>(ctx, a_in, d_out, e_out, tau_out, uplo, ws, block_size);
        case 32:
            return sytrd_blocked_impl<B, T, 32>(ctx, a_in, d_out, e_out, tau_out, uplo, ws, block_size);
        case 64:
            return sytrd_blocked_impl<B, T, 64>(ctx, a_in, d_out, e_out, tau_out, uplo, ws, block_size);
        default:
            return sytrd_blocked_impl<B, T, -1>(ctx, a_in, d_out, e_out, tau_out, uplo, ws, block_size);
    }
}

#define SYTRD_BLOCKED_INSTANTIATE(back, fp) \
    template Event sytrd_blocked<back, BATCHLAS_UNPAREN fp>( \
        Queue&, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        const VectorView<BATCHLAS_UNPAREN fp>&, \
        const VectorView<BATCHLAS_UNPAREN fp>&, \
        const VectorView<BATCHLAS_UNPAREN fp>&, \
        Uplo, \
        const Span<std::byte>&, \
        int32_t); \
    template size_t sytrd_blocked_buffer_size<back, BATCHLAS_UNPAREN fp>( \
        Queue&, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        const VectorView<BATCHLAS_UNPAREN fp>&, \
        const VectorView<BATCHLAS_UNPAREN fp>&, \
        const VectorView<BATCHLAS_UNPAREN fp>&, \
        Uplo, \
        int32_t);

#define SYTRD_BLOCKED_INSTANTIATE_FOR_BACKEND(back) \
    BATCHLAS_FOR_EACH_SCALAR_TYPE_1(SYTRD_BLOCKED_INSTANTIATE, back)

#if BATCHLAS_HAS_CUDA_BACKEND
SYTRD_BLOCKED_INSTANTIATE_FOR_BACKEND(Backend::CUDA)
#endif

#if BATCHLAS_HAS_ROCM_BACKEND
SYTRD_BLOCKED_INSTANTIATE_FOR_BACKEND(Backend::ROCM)
#endif

#if BATCHLAS_HAS_HOST_BACKEND
SYTRD_BLOCKED_INSTANTIATE_FOR_BACKEND(Backend::NETLIB)
#endif

#undef SYTRD_BLOCKED_INSTANTIATE_FOR_BACKEND
#undef SYTRD_BLOCKED_INSTANTIATE

} // namespace batchlas
