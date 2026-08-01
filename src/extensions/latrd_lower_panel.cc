#include <blas/device.hh>
#include <blas/extensions.hh>
#include <blas/matrix.hh>

#include <batchlas/backend_config.h>

#include "../math-helpers.hh"
#include "../queue.hh"

#include <util/sycl-local-accessor-helpers.hh>
#include <util/group-invoke.hh>

#include <algorithm>
#include <complex>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace batchlas {

namespace {

// Returns true when BATCHLAS_LATRD_IMPL=device, false otherwise (legacy default).
// Evaluated once and cached for the lifetime of the process.
inline bool use_device_latrd() {
    static const bool result = []() {
        const char* v = std::getenv("BATCHLAS_LATRD_IMPL");
        return v && std::string(v) == "device";
    }();
    return result;
}

// ---------------------------------------------------------------------------
// Helpers used by the legacy kernel
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Helper used by the device kernel
// ---------------------------------------------------------------------------

template <typename T>
inline T hermitian_diagonal(const T& value) {
    return value;
}

template <typename Real>
inline std::complex<Real> hermitian_diagonal(const std::complex<Real>& value) {
    return std::complex<Real>(value.real(), Real(0));
}

// ---------------------------------------------------------------------------
// Kernel name tags — "Legacy" variants so both paths can coexist.
// ---------------------------------------------------------------------------
template <typename T, int WG, bool FuseTrailingUpdate> class LatrdLowerPanelKernelLegacy;
template <typename T, int WG, bool FuseTrailingUpdate> class LatrdLowerPanelKernel;

// ---------------------------------------------------------------------------
// Legacy kernel: manual group-reduction inner loops
// ---------------------------------------------------------------------------
template <typename T, int WG, bool FuseTrailingUpdate>
Event latrd_lower_panel_batched_wg_legacy(Queue& q,
                                          const MatrixView<T, MatrixFormat::Dense>& a,
                                          const VectorView<T>& e,
                                          const VectorView<T>& tau,
                                          const MatrixView<T, MatrixFormat::Dense>& w) {
    constexpr int wg = WG;

    (void)q->submit([&](sycl::handler& h) {
        KernelMatrixView<T, MatrixFormat::Dense> A_view(a.data_ptr(), a.rows(), a.cols(), a.ld(), a.stride(), a.batch_size());
        KernelMatrixView<T, MatrixFormat::Dense> W_view(w.data_ptr(), w.rows(), w.cols(), w.ld(), w.stride(), w.batch_size());

        VectorView<T> E_view = e;
        VectorView<T> TAU_view = tau;

        const int n = A_view.rows();
        const int batch = A_view.batch_size();
        const int ib = W_view.cols();

        auto v_local   = sycl::local_accessor<T, 1>(sycl::range<1>(static_cast<size_t>(n)), h);
        auto wcol_local = sycl::local_accessor<T, 1>(sycl::range<1>(static_cast<size_t>(n)), h);
        auto vip_local  = sycl::local_accessor<T, 1>(sycl::range<1>(static_cast<size_t>(ib)), h);
        auto wip_local  = sycl::local_accessor<T, 1>(sycl::range<1>(static_cast<size_t>(ib)), h);

        h.parallel_for<LatrdLowerPanelKernelLegacy<T, WG, FuseTrailingUpdate>>(
            sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(batch) * wg), sycl::range<1>(wg)),
            [=](sycl::nd_item<1> it) {
                const int b = static_cast<int>(it.get_group_linear_id());
                if (b >= batch) return;

                const int lid = static_cast<int>(it.get_local_linear_id());
                const sycl::group<1> g = it.get_group();

                auto Ab = A_view.batch_item(b);
                auto Wb = W_view.batch_item(b);

                for (int i = 0; i < ib; ++i) {
                    if (i >= n - 1) break;

                    if (lid < i) {
                        const int p = lid;
                        vip_local[p] = (i == p + 1) ? T(1) : Ab(i, p);
                        wip_local[p] = Wb(i, p);
                    }
                    it.barrier(sycl::access::fence_space::local_space);

                    if (lid == 0) {
                        T aii = Ab(i, i);
                        for (int p = 0; p < i; ++p) {
                            const T vip = vip_local[p];
                            const T wip = wip_local[p];
                            aii -= vip * conj_if_needed(wip) + wip * conj_if_needed(vip);
                        }
                        Ab(i, i) = aii;
                    }

                    for (int r = i + 1 + lid; r < n; r += wg) {
                        T val = Ab(r, i);
                        for (int p = 0; p < i; ++p) {
                            const T wip = wip_local[p];
                            const T vip = vip_local[p];

                            T vrp = T(0);
                            if (r == p + 1) {
                                vrp = T(1);
                            } else if (r > p + 1) {
                                vrp = Ab(r, p);
                            }

                            const T wrp = Wb(r, p);
                            val -= vrp * conj_if_needed(wip) + wrp * conj_if_needed(vip);
                        }
                        Ab(r, i) = val;
                    }
                    it.barrier(sycl::access::fence_space::global_space);

                    using Real = typename base_type<T>::type;

                    const int x0 = i + 2;
                    Real sumsq = Real(0);
                    for (int r = x0 + lid; r < n; r += wg) {
                        sumsq += abs2_if_complex(Ab(r, i));
                    }
                    sumsq = reduce_sum_group_real<T>(g, sumsq);

                    T alpha = T(0);
                    if (lid == 0 && i + 1 < n) {
                        alpha = Ab(i + 1, i);
                    }
                    alpha = sycl::group_broadcast(g, alpha);

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
                        Ab(i + 1, i) = T(1);
                    }

                    tau_i = sycl::group_broadcast(g, tau_i);
                    scale = sycl::group_broadcast(g, scale);

                    if (tau_i != T(0)) {
                        for (int r = x0 + lid; r < n; r += wg) {
                            Ab(r, i) *= scale;
                        }
                    }
                    it.barrier(sycl::access::fence_space::global_space);

                    for (int r = i + 1 + lid; r < n; r += wg) {
                        v_local[r] = (r == i + 1) ? T(1) : Ab(r, i);
                    }
                    it.barrier(sycl::access::fence_space::local_space);

                    const int col = i;

                    for (int r = i + 1 + lid; r < n; r += wg) {
                        // Symmetric mat-vec: acc = sum_c Ah(r,c) * v(c), where
                        // Ah(r,c) is Ab(r,c) for c <= r and conj(Ab(c,r)) for c > r.
                        //
                        // Ascending c crosses that boundary exactly once, at c == r,
                        // so the loop splits into two contiguous ranges without
                        // changing the accumulation order (results stay bit-identical).
                        // Removing the per-element branch lets the compiler unroll and
                        // keep several independent loads in flight; this kernel is
                        // memory-latency bound, so memory-level parallelism is what
                        // matters here.
                        //
                        // Measured alternatives that were all slower and rejected:
                        //   - unroll 8, and processing two rows per iteration: both
                        //     cost more registers than the added parallelism returns.
                        //   - computing the c > r term with one sub-group per column
                        //     (fully coalesced, 20 -> 9.7 sectors/request): the extra
                        //     barrier destroys reuse between the two passes and short
                        //     columns near r -> n leave most lanes idle, so DRAM
                        //     traffic and runtime both rose.
                        const int c_split = sycl::min(r, n - 1);

                        T acc = T(0);

                        // c in [i+1, r]: walk row r of the lower triangle.
                        // Consecutive threads hold consecutive r, so each step is a
                        // coalesced access across the warp.
                        #pragma unroll 4
                        for (int c = i + 1; c <= c_split; ++c) {
                            acc += Ab(r, c) * v_local[c];
                        }

                        // c in (r, n): walk column r of the lower triangle.
                        // Contiguous per thread, so successive iterations reuse the
                        // same cache lines.
                        #pragma unroll 4
                        for (int c = c_split + 1; c < n; ++c) {
                            acc += conj_if_needed(Ab(c, r)) * v_local[c];
                        }

                        wcol_local[r] = acc;
                    }

                    for (int p = 0; p < i; ++p) {
                        const int pc = p;

                        T gamma_partial = T(0);
                        T delta_partial = T(0);
                        for (int c = i + 1 + lid; c < n; c += wg) {
                            const T vc = v_local[c];

                            gamma_partial += conj_if_needed(Wb(c, pc)) * vc;

                            const T vcp = (c == p + 1) ? T(1) : ((c > p + 1) ? Ab(c, p) : T(0));
                            delta_partial += conj_if_needed(vcp) * vc;
                        }
                        const T gamma = reduce_sum_group(g, gamma_partial);
                        const T delta = reduce_sum_group(g, delta_partial);

                        for (int r = i + 1 + lid; r < n; r += wg) {
                            const T vrp = (r == p + 1) ? T(1) : ((r > p + 1) ? Ab(r, p) : T(0));
                            const T wrp = Wb(r, pc);
                            wcol_local[r] -= vrp * gamma + wrp * delta;
                        }
                    }

                    for (int r = i + 1 + lid; r < n; r += wg) {
                        wcol_local[r] *= tau_i;
                    }

                    T dot_partial = T(0);
                    for (int r = i + 1 + lid; r < n; r += wg) {
                        const T vr = v_local[r];
                        dot_partial += conj_if_needed(vr) * wcol_local[r];
                    }
                    const T dot = reduce_sum_group(g, dot_partial);

                    const T alpha2 = T(-0.5) * tau_i * dot;
                    for (int r = i + 1 + lid; r < n; r += wg) {
                        const T vr = v_local[r];
                        wcol_local[r] += alpha2 * vr;
                    }

                    for (int r = i + 1 + lid; r < n; r += wg) {
                        Wb(r, col) = wcol_local[r];
                    }
                    it.barrier(sycl::access::fence_space::global_space);
                }

                if constexpr (FuseTrailingUpdate) {
                    const int j2 = ib;
                    const int n2 = n - j2;
                    for (int lin = lid; lin < n2 * n2; lin += wg) {
                        const int r = lin % n2;
                        const int c = lin / n2;
                        if (r < c) continue;

                        const int rr = j2 + r;
                        const int cc = j2 + c;
                        T acc = T(0);
                        for (int k = 0; k < ib; ++k) {
                            const T vrk = (rr == k + 1) ? T(1) : ((rr > k + 1) ? Ab(rr, k) : T(0));
                            const T vck = (cc == k + 1) ? T(1) : ((cc > k + 1) ? Ab(cc, k) : T(0));
                            const T wrk = Wb(rr, k);
                            const T wck = Wb(cc, k);
                            acc += vrk * conj_if_needed(wck) + wrk * conj_if_needed(vck);
                        }

                        T a_rc = Ab(rr, cc) - acc;
                        if constexpr (internal::is_complex<T>::value) {
                            if (rr == cc) {
                                a_rc = T(a_rc.real(), typename T::value_type(0));
                            }
                        }
                        Ab(rr, cc) = a_rc;
                    }
                }
            });
    });

    return q.get_event();
}

// ---------------------------------------------------------------------------
// Device-BLAS kernel: uses device::hemv, dotc, axpy, scal, copy, her2k
// ---------------------------------------------------------------------------
template <typename T, int WG, bool FuseTrailingUpdate>
Event latrd_lower_panel_batched_wg_device(Queue& q,
                                          const MatrixView<T, MatrixFormat::Dense>& a,
                                          const VectorView<T>& e,
                                          const VectorView<T>& tau,
                                          const MatrixView<T, MatrixFormat::Dense>& w) {
    constexpr int wg = WG;
    const int n = a.rows();
    const int batch = a.batch_size();
    const int ib = w.cols();
    const auto launch = batchlas::device::make_group_launch_info(wg);
    std::size_t workspace_elements = 0;

    if (batch > 0 && n > 1) {
        const int extent = n - 1;
        workspace_elements = std::max(
            workspace_elements,
            batchlas::device::hemv_workspace_elements<T,
                batchlas::device::DeviceBlasPolicy::Auto, Uplo::Lower>(launch, extent));

        if constexpr (FuseTrailingUpdate) {
            if (n > ib) {
                const int trailing_extent = n - ib;
                workspace_elements = std::max(
                    workspace_elements,
                    batchlas::device::her2k_workspace_elements<T,
                        batchlas::device::DeviceBlasPolicy::Subgroup16,
                        Uplo::Lower, Transpose::NoTrans>(launch, trailing_extent, ib));
            }
        }
    }

    (void)q->submit([&](sycl::handler& h) {
        auto A_view = a.kernel_view();
        auto W_view = w.kernel_view();

        VectorView<T> E_view = e;
        VectorView<T> TAU_view = tau;

        auto v_local    = sycl::local_accessor<T, 1>(sycl::range<1>(static_cast<size_t>(n)), h);
        auto wcol_local = sycl::local_accessor<T, 1>(sycl::range<1>(static_cast<size_t>(n)), h);
        auto vip_local  = sycl::local_accessor<T, 1>(sycl::range<1>(static_cast<size_t>(ib)), h);
        auto wip_local  = sycl::local_accessor<T, 1>(sycl::range<1>(static_cast<size_t>(ib)), h);
        sycl::local_accessor<T, 1> workspace(sycl::range<1>(std::max<std::size_t>(workspace_elements, 1)), h);

        h.parallel_for<LatrdLowerPanelKernel<T, WG, FuseTrailingUpdate>>(
            sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(batch) * wg), sycl::range<1>(wg)),
            [=](sycl::nd_item<1> it) {
                const int b = static_cast<int>(it.get_group_linear_id());
                if (b >= batch) return;

                const int lid = static_cast<int>(it.get_local_linear_id());
                const sycl::group<1> g = it.get_group();
                T* v_ptr        = util::get_raw_ptr(v_local);
                T* wcol_ptr     = util::get_raw_ptr(wcol_local);
                T* vip_ptr      = util::get_raw_ptr(vip_local);
                T* wip_ptr      = util::get_raw_ptr(wip_local);
                T* workspace_ptr = workspace_elements == 0
                    ? static_cast<T*>(nullptr)
                    : util::get_raw_ptr(workspace);

                auto Ab = A_view.batch_item(b);
                auto Wb = W_view.batch_item(b);

                for (int i = 0; i < ib; ++i) {
                    if (i >= n - 1) break;
                    const int tail = n - (i + 1);
                    auto a_col_tail  = Ab(Slice(i + 1, SliceEnd()), i);
                    auto v_tail      = VectorView<T>(v_ptr + i + 1, tail);
                    auto wcol_tail   = VectorView<T>(wcol_ptr + i + 1, tail);

                    if (i > 0) {
                        auto vip_view = VectorView<T>(vip_ptr, i);
                        auto wip_view = VectorView<T>(wip_ptr, i);
                        batchlas::device::copy(g, Ab(i, Slice(0, i)), vip_view);
                        batchlas::device::copy(g, Wb(i, Slice(0, i)), wip_view);
                    }
                    it.barrier(sycl::access::fence_space::local_space);

                    if (i > 0) {
                        const auto vip_view = VectorView<T>(vip_ptr, i);
                        const auto wip_view = VectorView<T>(wip_ptr, i);
                        const T panel_dot = batchlas::device::dotc(g, vip_view, wip_view);
                        if (lid == 0) {
                            Ab(i, i) = hermitian_diagonal(
                                Ab(i, i) - panel_dot - batchlas::device::detail::conj(panel_dot));
                        }
                    }

                    if (i > 0) {
                        it.barrier(sycl::access::fence_space::local_space);
                        for (int p = 0; p < i; ++p) {
                            auto v_prev = Ab(Slice(i + 1, SliceEnd()), p);
                            auto w_prev = Wb(Slice(i + 1, SliceEnd()), p);
                            batchlas::device::hadamard(g, a_col_tail,
                                [&](T x, T v, T wv) {
                                    return x
                                        - batchlas::device::detail::conj(wip_ptr[p]) * v
                                        - batchlas::device::detail::conj(vip_ptr[p]) * wv;
                                },
                                a_col_tail, v_prev, w_prev);
                        }
                    }
                    it.barrier(sycl::access::fence_space::global_space);

                    const int x0 = i + 2;
                    T alpha_i = i + 1 < n ? Ab(i + 1, i) : T(0);
                    const T tau_i = internal::larfg(g, alpha_i, Ab(Slice(x0, SliceEnd()), i));
                    if (lid == 0) {
                        E_view(i, b)   = alpha_i;
                        TAU_view(i, b) = tau_i;
                        Ab(i + 1, i)   = T(1);
                    }
                    it.barrier(sycl::access::fence_space::local_space);

                    if (lid == 0) {
                        v_ptr[i + 1] = T(1);
                    }
                    if (x0 < n) {
                        batchlas::device::copy(g, Ab(Slice(x0, SliceEnd()), i),
                                               VectorView<T>(v_ptr + x0, n - x0));
                    }
                    it.barrier(sycl::access::fence_space::local_space);

                    auto trailing_view = KernelMatrixView<T, MatrixFormat::Dense>(
                        Ab.data() + (i + 1) + (i + 1) * Ab.ld(),
                        tail, tail, Ab.ld());
                    batchlas::device::hemv<batchlas::device::DeviceBlasPolicy::Auto, Uplo::Lower>(
                        g, trailing_view, v_tail, wcol_tail, T(1), T(0), static_cast<T*>(nullptr));
                    it.barrier(sycl::access::fence_space::local_space);

                    for (int p = 0; p < i; ++p) {
                        const auto vp_tail = Ab(Slice(i + 1), p);
                        const auto wp_tail = Wb(Slice(i + 1), p);
                        const auto gamma   = batchlas::device::dotc(g, wp_tail, v_tail);
                        const auto delta   = batchlas::device::dotc(g, vp_tail, v_tail);
                        for (int r = i + 1 + lid; r < n; r += wg) {
                            wcol_local[r] -= vp_tail(r - (i + 1)) * gamma
                                           + wp_tail(r - (i + 1)) * delta;
                        }
                    }

                    it.barrier(sycl::access::fence_space::local_space);
                    batchlas::device::scal(g, wcol_tail, tau_i);
                    it.barrier(sycl::access::fence_space::local_space);

                    const T dot    = batchlas::device::dotc(g, v_tail, wcol_tail);
                    const T alpha2 = T(-0.5) * tau_i * dot;
                    batchlas::device::axpy(g, v_tail, wcol_tail, alpha2);
                    it.barrier(sycl::access::fence_space::local_space);

                    batchlas::device::copy(g, wcol_tail, Wb(Slice(i + 1, SliceEnd()), i));
                    it.barrier(sycl::access::fence_space::global_space);
                }

                if constexpr (FuseTrailingUpdate) {
                    const int j2 = ib;
                    const int n2 = n - j2;
                    if (n2 > 0) {
                        batchlas::device::her2k<batchlas::device::DeviceBlasPolicy::Subgroup16,
                                                Uplo::Lower, Transpose::NoTrans>(
                            g,
                            Ab(Slice(j2), Slice(0, ib)),
                            Wb(Slice(j2), Slice(0, ib)),
                            Ab(Slice(j2), Slice(j2)),
                            T(-1), T(1),
                            workspace_ptr);
                    }
                }
            });
    });

    return q.get_event();
}

// ---------------------------------------------------------------------------
// Dispatcher: selects legacy or device path based on use_device_latrd().
// Legacy supports WG = 64/128/256; device additionally supports WG = 512.
// ---------------------------------------------------------------------------
template <typename T>
Event latrd_lower_panel_batched(Queue& q,
                                const MatrixView<T, MatrixFormat::Dense>& a,
                                const VectorView<T>& e,
                                const VectorView<T>& tau,
                                const MatrixView<T, MatrixFormat::Dense>& w,
                                int32_t wg_hint,
                                bool fuse_trailing_update) {
    const int n = a.rows();
    const bool use_device = use_device_latrd();

    auto call_legacy = [&](auto wg_tag) {
        constexpr int WG = decltype(wg_tag)::value;
        if (fuse_trailing_update)
            return latrd_lower_panel_batched_wg_legacy<T, WG, true>(q, a, e, tau, w);
        return latrd_lower_panel_batched_wg_legacy<T, WG, false>(q, a, e, tau, w);
    };
    auto call_device = [&](auto wg_tag) {
        constexpr int WG = decltype(wg_tag)::value;
        if (fuse_trailing_update)
            return latrd_lower_panel_batched_wg_device<T, WG, true>(q, a, e, tau, w);
        return latrd_lower_panel_batched_wg_device<T, WG, false>(q, a, e, tau, w);
    };

    if (use_device) {
        if (wg_hint == 64)  return call_device(std::integral_constant<int, 64>{});
        if (wg_hint == 128) return call_device(std::integral_constant<int, 128>{});
        if (wg_hint == 256) return call_device(std::integral_constant<int, 256>{});
        if (n <= 64)  return call_device(std::integral_constant<int, 64>{});
        if (n <= 128) return call_device(std::integral_constant<int, 128>{});
        if (n <= 256) return call_device(std::integral_constant<int, 256>{});
        return call_device(std::integral_constant<int, 512>{});
    } else {
        if (wg_hint == 64)  return call_legacy(std::integral_constant<int, 64>{});
        if (wg_hint == 128) return call_legacy(std::integral_constant<int, 128>{});
        if (wg_hint == 256) return call_legacy(std::integral_constant<int, 256>{});
        if (n <= 64)  return call_legacy(std::integral_constant<int, 64>{});
        if (n <= 128) return call_legacy(std::integral_constant<int, 128>{});
        return call_legacy(std::integral_constant<int, 256>{});
    }
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
                        const MatrixView<T, MatrixFormat::Dense>& w_panel_in,
                        int32_t wg_hint,
                        bool fuse_trailing_update) {
    (void)B;
    validate_latrd_lower_panel_panel_dims(a_panel_in, e_panel_out, tau_panel_out, w_panel_in);

    auto& a   = const_cast<MatrixView<T, MatrixFormat::Dense>&>(a_panel_in);
    auto& e   = const_cast<VectorView<T>&>(e_panel_out);
    auto& tau = const_cast<VectorView<T>&>(tau_panel_out);
    auto& w   = const_cast<MatrixView<T, MatrixFormat::Dense>&>(w_panel_in);

    if (w.cols() == 0) {
        return ctx.get_event();
    }

    return latrd_lower_panel_batched<T>(ctx, a, e, tau, w, wg_hint, fuse_trailing_update);
}

template <Backend B, typename T>
Event latrd_lower_panel(Queue& ctx,
                        const MatrixView<T, MatrixFormat::Dense>& a_in,
                        const VectorView<T>& e_out,
                        const VectorView<T>& tau_out,
                        const MatrixView<T, MatrixFormat::Dense>& w_in,
                        int32_t j0,
                        int32_t ib,
                        int32_t wg_hint,
                        bool fuse_trailing_update) {
    (void)B;
    validate_latrd_lower_panel_dims(a_in, e_out, tau_out, w_in, j0, ib);

    auto& a   = const_cast<MatrixView<T, MatrixFormat::Dense>&>(a_in);
    auto& e   = const_cast<VectorView<T>&>(e_out);
    auto& tau = const_cast<VectorView<T>&>(tau_out);
    auto& w   = const_cast<MatrixView<T, MatrixFormat::Dense>&>(w_in);

    auto a_panel   = a({j0, SliceEnd()}, {j0, SliceEnd()});
    auto e_panel   = e(Slice(j0, j0 + ib));
    auto tau_panel = tau(Slice(j0, j0 + ib));
    auto w_panel   = w({j0, SliceEnd()}, {0, ib});
    return latrd_lower_panel<B, T>(ctx, a_panel, e_panel, tau_panel, w_panel, wg_hint, fuse_trailing_update);
}

#define LATRD_LOWER_PANEL_INSTANTIATE(back, fp) \
    template Event latrd_lower_panel<back, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        const VectorView<fp>&, \
        const VectorView<fp>&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        int32_t, \
        bool); \
    template Event latrd_lower_panel<back, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        const VectorView<fp>&, \
        const VectorView<fp>&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        int32_t, \
        int32_t, \
        int32_t, \
        bool);

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
