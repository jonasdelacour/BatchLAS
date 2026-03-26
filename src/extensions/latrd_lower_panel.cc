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
#include <type_traits>

namespace batchlas {

namespace {

template <typename T>
struct HouseholderScalars {
    T tau{};
    T beta{};
    T scale{};
};

template <typename U>
inline U conj_if_needed(const U& x) {
    return batchlas::device::detail::conjugate_if_needed(x);
}

template <typename Real>
inline Real sign_nonzero_real(Real x) {
    return (sycl::signbit(x) ? Real(-1) : Real(1));
}

template <typename Real>
inline Real sign_nonzero(const Real& x) {
    return sign_nonzero_real(x);
}

template <typename Real>
inline std::complex<Real> sign_nonzero(const std::complex<Real>& x) {
    const Real magnitude = sycl::hypot(x.real(), x.imag());
    if (magnitude == Real(0)) {
        return std::complex<Real>(1);
    }
    return x / magnitude;
}

template <typename T>
inline T hermitian_diagonal(const T& value) {
    return value;
}

template <typename Real>
inline std::complex<Real> hermitian_diagonal(const std::complex<Real>& value) {
    return std::complex<Real>(value.real(), Real(0));
}

template <typename Real>
inline Real dotc_norm_sq_real(const Real& value) {
    return value;
}

template <typename Real>
inline Real dotc_norm_sq_real(const std::complex<Real>& value) {
    return value.real();
}

template <typename Real>
inline HouseholderScalars<Real> compute_householder_scalars(const Real& alpha, Real xnorm) {
    HouseholderScalars<Real> result;
    if (xnorm == Real(0)) {
        result.beta = alpha;
        return result;
    }

    result.beta = -sign_nonzero(alpha) * Real(sycl::hypot(alpha, xnorm));
    result.tau = (result.beta - alpha) / result.beta;
    result.scale = Real(1) / (alpha - result.beta);
    return result;
}

template <typename Real>
inline HouseholderScalars<std::complex<Real>> compute_householder_scalars(const std::complex<Real>& alpha, Real xnorm) {
    using Complex = std::complex<Real>;

    HouseholderScalars<Complex> result;
    if (xnorm == Real(0) && alpha.imag() == Real(0)) {
        result.beta = alpha;
        return result;
    }

    const Real alpha_abs = sycl::hypot(alpha.real(), alpha.imag());
    const Real beta_abs = sycl::hypot(alpha_abs, xnorm);
    result.beta = -sign_nonzero(alpha) * Complex(beta_abs);
    result.tau = (result.beta - alpha) / result.beta;
    result.scale = Complex(1) / (alpha - result.beta);
    return result;
}

template <typename T, int WG, bool FuseTrailingUpdate>
class LatrdLowerPanelKernel;

template <typename T, int WG, bool FuseTrailingUpdate>
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

        // Cache the current reflector vector v and the current W column in local memory
        // to reduce repeated global loads/stores.
        auto v_local = sycl::local_accessor<T, 1>(sycl::range<1>(static_cast<size_t>(n)), h);
        auto wcol_local = sycl::local_accessor<T, 1>(sycl::range<1>(static_cast<size_t>(n)), h);

        // Cache vip/wip (values at row i for previous reflector columns p<i).
        auto vip_local = sycl::local_accessor<T, 1>(sycl::range<1>(static_cast<size_t>(ib)), h);
        auto wip_local = sycl::local_accessor<T, 1>(sycl::range<1>(static_cast<size_t>(ib)), h);

        h.parallel_for<LatrdLowerPanelKernel<T, WG, FuseTrailingUpdate>>(
            sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(batch) * wg), sycl::range<1>(wg)),
            [=](sycl::nd_item<1> it) {
                const int b = static_cast<int>(it.get_group_linear_id());
                if (b >= batch) return;

                const int lid = static_cast<int>(it.get_local_linear_id());
                const sycl::group<1> g = it.get_group();
                T* v_ptr = util::get_raw_ptr(v_local);
                T* wcol_ptr = util::get_raw_ptr(wcol_local);
                T* vip_ptr = util::get_raw_ptr(vip_local);
                T* wip_ptr = util::get_raw_ptr(wip_local);

                auto Ab = A_view.batch_item(b);
                auto Wb = W_view.batch_item(b);

                for (int i = 0; i < ib; ++i) {
                    if (i >= n - 1) break;
                    const int tail = n - (i + 1);
                    auto a_col_tail = Ab(Slice(i + 1, SliceEnd()), i);
                    auto v_tail = VectorView<T>(v_ptr + i + 1, tail);
                    auto wcol_tail = VectorView<T>(wcol_ptr + i + 1, tail);

                    // Cache vip/wip for p<i (shared across all r updates below).
                    if (i > 0) {
                        auto vip_view = VectorView<T>(vip_ptr, i);
                        auto wip_view = VectorView<T>(wip_ptr, i);
                        batchlas::device::copy(it, Ab(i, Slice(0, i)), vip_view);
                        batchlas::device::copy(it, Wb(i, Slice(0, i)), wip_view);
                    }
                    it.barrier(sycl::access::fence_space::local_space);

                    // Update diagonal element A(i,i) using previously computed V/W (j0..i-1).
                    if (i > 0) {
                        const auto vip_view = VectorView<T>(vip_ptr, i);
                        const auto wip_view = VectorView<T>(wip_ptr, i);
                        const T panel_dot = batchlas::device::dotc(it, vip_view, wip_view);
                        if (lid == 0) {
                            Ab(i, i) = hermitian_diagonal(Ab(i, i) - panel_dot - conj_if_needed(panel_dot));
                        }
                    }

                    // Update column i entries from row i+1 .. n-1.
                    if (i > 0) {
                        // Conjugate cached vip/wip in-place for gemv (LACGV-style).
                        const int local_size = static_cast<int>(it.get_local_range(0));
                        for (int j = lid; j < i; j += local_size) {
                            vip_ptr[j] = conj_if_needed(vip_ptr[j]);
                            wip_ptr[j] = conj_if_needed(wip_ptr[j]);
                        }
                        it.barrier(sycl::access::fence_space::local_space);

                        auto v_prev = KernelMatrixView<T, MatrixFormat::Dense>(
                            Ab.data() + (i + 1), tail, i, Ab.ld());
                        auto w_prev = KernelMatrixView<T, MatrixFormat::Dense>(
                            Wb.data() + (i + 1), tail, i, Wb.ld());
                        batchlas::device::gemv(it, v_prev, VectorView<T>(wip_ptr, i), a_col_tail, T(-1), T(1));
                        batchlas::device::gemv(it, w_prev, VectorView<T>(vip_ptr, i), a_col_tail, T(-1), T(1));
                    }
                    // Ensure updated Ab(r,i) values are visible before other lanes read them
                    // (later phases use a different lane-to-row mapping).
                    it.barrier(sycl::access::fence_space::global_space);

                    // Generate Householder reflector to annihilate A(i+2:n-1,i).
                    using Real = typename base_type<T>::type;

                    const int x0 = i + 2;
                    const Real sumsq = x0 < n
                        ? dotc_norm_sq_real(batchlas::device::dotc(it, Ab(Slice(x0, SliceEnd()), i), Ab(Slice(x0, SliceEnd()), i)))
                        : Real(0);
                    
                    auto [alpha, tau_i, scale] = invoke_one_broadcast(g, [=] {
                        T alpha = i + 1 < n ? Ab(i + 1, i) : T(0);
                        const auto householder = compute_householder_scalars(alpha, sycl::sqrt(sumsq));
                        E_view(i, b) = householder.beta;
                        TAU_view(i, b) = householder.tau;
                        Ab(i + 1, i) = T(1);
                        return std::array<T,3>{alpha, householder.tau, householder.scale};
                    });

                    if (tau_i != T(0) && x0 < n) {
                        batchlas::device::scal(it, Ab(Slice(x0, SliceEnd()), i), scale);
                    }

                    // Ensure scaling stores are visible before other lanes read Ab(r,i)
                    // (v_local fill uses a different lane-to-row mapping).
                    it.barrier(sycl::access::fence_space::global_space);

                    // Build v in local memory for the upcoming A*v, dot-products, and updates.
                    if (lid == 0) {
                        v_ptr[i + 1] = T(1);
                    }
                    if (x0 < n) {
                        batchlas::device::copy(it, Ab(Slice(x0, SliceEnd()), i), VectorView<T>(v_ptr + x0, n - x0));
                    }
                    it.barrier(sycl::access::fence_space::local_space);

                    // Compute W(:, i-j0) for rows i+1..n-1.
                    // w = tau * A(i+1:n-1, i+1:n-1) * v
                    const int col = i;

                    // Compute raw w (before scaling by tau) for the *updated* trailing matrix:
                    //   A := A - V*W^H - W*V^H  (within the current panel, columns j0..i-1)
                    // without explicitly forming A.
                    auto trailing_view = KernelMatrixView<T, MatrixFormat::Dense>(
                        Ab.data() + (i + 1) + (i + 1) * Ab.ld(),
                        tail,
                        tail,
                        Ab.ld());
                    batchlas::device::hemv(it,
                                           trailing_view,
                                           v_tail,
                                           wcol_tail,
                                           T(1),
                                           T(0),
                                           Uplo::Lower,
                                           batchlas::device::DeviceBlasPolicy::Auto);
                    // Generic hemv may materialize the output from only a subset of lanes,
                    // so all threads must wait before applying the correction updates.
                    it.barrier(sycl::access::fence_space::local_space);

                    // Apply intra-panel corrections from previously computed reflectors.
                    for (int p = 0; p < i; ++p) {
                        const int pc = p;

                        // gamma = W(:,pc)^H * v, delta = V(:,p)^H * v
                        const auto vp_tail = Ab(Slice(i + 1, SliceEnd()), p);
                        const auto wp_tail = Wb(Slice(i + 1, SliceEnd()), pc);
                        const T gamma = batchlas::device::dotc(it, wp_tail, v_tail);
                        const T delta = batchlas::device::dotc(it, vp_tail, v_tail);
                        for (int r = i + 1 + lid; r < n; r += wg) {
                            wcol_local[r] -= vp_tail(r - (i + 1)) * gamma + wp_tail(r - (i + 1)) * delta;
                        }
                    }

                    // The correction loop writes shared local W entries with a lane-to-row
                    // mapping that differs from the subsequent BLAS helpers.
                    it.barrier(sycl::access::fence_space::local_space);

                    // Scale by tau.
                    batchlas::device::scal(it, wcol_tail, tau_i);

                    // dotc/axpy may read W with a different work distribution than scal.
                    it.barrier(sycl::access::fence_space::local_space);

                    // dot = v^H * w
                    const T dot = batchlas::device::dotc(it, v_tail, wcol_tail);

                    // w += (-0.5 * tau * dot) * v
                    const T alpha2 = T(-0.5) * tau_i * dot;
                    batchlas::device::axpy(it, v_tail, wcol_tail, alpha2);

                    // Commit the fully updated local W column after all local-memory writes land.
                    it.barrier(sycl::access::fence_space::local_space);

                    // Commit the computed W column to global memory once.
                    batchlas::device::copy(it, wcol_tail, Wb(Slice(i + 1, SliceEnd()), col));
                    // Ensure all global writes to A/W from this iteration are visible before the next iteration
                    // reads Wb(i,p) / Ab(i,p) computed by other lanes.
                    it.barrier(sycl::access::fence_space::global_space);
                }

                if constexpr (FuseTrailingUpdate) {
                    const int j2 = ib;
                    const int n2 = n - j2;
                    if (n2 > 0) {
                        auto trailing_view = KernelMatrixView<T, MatrixFormat::Dense>(
                            Ab.data() + j2 + j2 * Ab.ld(),
                            n2,
                            n2,
                            Ab.ld());
                        auto v_panel_view = KernelMatrixView<T, MatrixFormat::Dense>(
                            Ab.data() + j2,
                            n2,
                            ib,
                            Ab.ld());
                        auto w_panel_view = KernelMatrixView<T, MatrixFormat::Dense>(
                            Wb.data() + j2,
                            n2,
                            ib,
                            Wb.ld());

                        batchlas::device::her2k(it,
                                                v_panel_view,
                                                w_panel_view,
                                                trailing_view,
                                                T(-1),
                                                T(1),
                                                Uplo::Lower,
                                                Transpose::NoTrans,
                                                batchlas::device::DeviceBlasPolicy::Auto);
                    }
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
                                const MatrixView<T, MatrixFormat::Dense>& w,
                                int32_t wg_hint,
                                bool fuse_trailing_update) {
    const int n = a.rows();
    auto call = [&](auto wg_tag) {
        constexpr int WG = decltype(wg_tag)::value;
        if (fuse_trailing_update) {
            return latrd_lower_panel_batched_wg<T, WG, true>(q, a, e, tau, w);
        }
        return latrd_lower_panel_batched_wg<T, WG, false>(q, a, e, tau, w);
    };

    if (wg_hint == 64) {
        return call(std::integral_constant<int, 64>{});
    }
    if (wg_hint == 128) {
        return call(std::integral_constant<int, 128>{});
    }
    if (wg_hint == 256) {
        return call(std::integral_constant<int, 256>{});
    }
    // For small panels, a smaller work-group reduces wasted lanes, barrier overhead,
    // and register pressure in the reduction-heavy panel kernel.
    if (n <= 128) {
        return call(std::integral_constant<int, 64>{});
    }
    if (n <= 256) {
        return call(std::integral_constant<int, 128>{});
    }
    return call(std::integral_constant<int, 256>{});
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
                        const MatrixView<T, MatrixFormat::Dense>& w_panel_in,
                        int32_t wg_hint,
                        bool fuse_trailing_update) {
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

    // Make mutable views (panel overwrites A and outputs e/tau/W).
    auto& a = const_cast<MatrixView<T, MatrixFormat::Dense>&>(a_in);
    auto& e = const_cast<VectorView<T>&>(e_out);
    auto& tau = const_cast<VectorView<T>&>(tau_out);
    auto& w = const_cast<MatrixView<T, MatrixFormat::Dense>&>(w_in);

    auto a_panel = a({j0, SliceEnd()}, {j0, SliceEnd()});
    auto e_panel = e(Slice(j0, j0 + ib));
    auto tau_panel = tau(Slice(j0, j0 + ib));
    auto w_panel = w({j0, SliceEnd()}, {0, ib});
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
