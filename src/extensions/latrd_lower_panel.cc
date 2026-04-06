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
inline T hermitian_diagonal(const T& value) {
    return value;
}

template <typename Real>
inline std::complex<Real> hermitian_diagonal(const std::complex<Real>& value) {
    return std::complex<Real>(value.real(), Real(0));
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
        auto A_view = a.kernel_view();
        auto W_view = w.kernel_view();

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
                        batchlas::device::copy(g, Ab(i, Slice(0, i)), vip_view);
                        batchlas::device::copy(g, Wb(i, Slice(0, i)), wip_view);
                    }
                    it.barrier(sycl::access::fence_space::local_space);

                    // Update diagonal element A(i,i) using previously computed V/W (j0..i-1).
                    if (i > 0) {
                        const auto vip_view = VectorView<T>(vip_ptr, i);
                        const auto wip_view = VectorView<T>(wip_ptr, i);
                        const T panel_dot = batchlas::device::dotc(g, vip_view, wip_view);
                        if (lid == 0) {
                            Ab(i, i) = hermitian_diagonal(Ab(i, i) - panel_dot - batchlas::device::detail::conj(panel_dot));
                        }
                    }

                    // Update column i entries from row i+1 .. n-1 using axpy-loop (mode1 from benchmark).
                    if (i > 0) {
                        it.barrier(sycl::access::fence_space::local_space);
                        // axpy-loop update: a_col_tail -= sum_p (v_prev * wip + w_prev * vip)
                        for (int p = 0; p < i; ++p) {
                            auto v_prev = Ab(Slice(i + 1, SliceEnd()), p);
                            auto w_prev = Wb(Slice(i + 1, SliceEnd()), p);

                            // $A(i+1:n-1,i) -= conj(wip(p)) * V(:,p) + conj(vip(p)) * W(:,p)$
                            batchlas::device::hadamard(g, a_col_tail, [&](T x, T w, T v) { return x - batchlas::device::detail::conj(wip_ptr[p]) * v - batchlas::device::detail::conj(vip_ptr[p]) * w; }, a_col_tail, w_prev, v_prev);
                        }
                    }
                    it.barrier(sycl::access::fence_space::global_space);

                    // Generate Householder reflector to annihilate A(i+2:n-1,i).
                    const int x0 = i + 2;
                    T alpha_i = i + 1 < n ? Ab(i + 1, i) : T(0);
                    const T tau_i = internal::larfg(g, alpha_i, Ab(Slice(x0, SliceEnd()), i));
                    if (lid == 0) {
                        E_view(i, b) = alpha_i;
                        TAU_view(i, b) = tau_i;
                        Ab(i + 1, i) = T(1);
                    }

                    // Ensure scaling stores are visible before other lanes read Ab(r,i)
                    // (v_local fill uses a different lane-to-row mapping).
                    it.barrier(sycl::access::fence_space::local_space);

                    // Build v in local memory for the upcoming A*v, dot-products, and updates.
                    if (lid == 0) {
                        v_ptr[i + 1] = T(1);
                    }
                    if (x0 < n) {
                        batchlas::device::copy(g, Ab(Slice(x0, SliceEnd()), i), VectorView<T>(v_ptr + x0, n - x0));
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
                    batchlas::device::hemv<batchlas::device::DeviceBlasPolicy::Auto, Uplo::Lower>(
                        g,
                        trailing_view,
                        v_tail,
                        wcol_tail,
                        T(1),
                        T(0));
                    it.barrier(sycl::access::fence_space::local_space);

                    // Apply intra-panel corrections from previously computed reflectors.
                    for (int p = 0; p < i; ++p) {
                        const int pc = p;

                        // gamma = W(:,pc)^H * v, delta = V(:,p)^H * v
                        // W(:,)
                        const auto vp_tail = Ab(Slice(i + 1), p);
                        const auto wp_tail = Wb(Slice(i + 1), pc);
                        const auto gamma = batchlas::device::dotc(g, wp_tail, v_tail);
                        const auto delta = batchlas::device::dotc(g, vp_tail, v_tail);
                        for (int r = i + 1 + lid; r < n; r += wg) {
                            wcol_local[r] -= vp_tail(r - (i + 1)) * gamma + wp_tail(r - (i + 1)) * delta;
                        }
                    }

                    it.barrier(sycl::access::fence_space::local_space);
                    // Scale by tau.
                    batchlas::device::scal(g, wcol_tail, tau_i);
                    it.barrier(sycl::access::fence_space::local_space);

                    // dot = v^H * w
                    const T dot = batchlas::device::dotc(g, v_tail, wcol_tail);

                    // w += (-0.5 * tau * dot) * v
                    const T alpha2 = T(-0.5) * tau_i * dot;
                    batchlas::device::axpy(g, v_tail, wcol_tail, alpha2);
                    it.barrier(sycl::access::fence_space::local_space);

                    // Commit the computed W column to global memory once.
                    batchlas::device::copy(g, wcol_tail, Wb(Slice(i + 1, SliceEnd()), col));
                    // Ensure all global writes to A/W from this iteration are visible before the next iteration
                    // reads Wb(i,p) / Ab(i,p) computed by other lanes.
                    it.barrier(sycl::access::fence_space::global_space);
                }

                if constexpr (FuseTrailingUpdate) {
                    const int j2 = ib;
                    const int n2 = n - j2;
                    if (n2 > 0) {
                        
                        // Trailing update: $A := A - V*W^H - W*V^H$                         
                        batchlas::device::her2k<batchlas::device::DeviceBlasPolicy::Subgroup16,
                                                        Uplo::Lower,
                                                        Transpose::NoTrans>(
                            g,
                            Ab(Slice(j2), Slice(0, ib)),
                            Wb(Slice(j2), Slice(0, ib)),
                            Ab(Slice(j2), Slice(j2)),
                            T(-1),
                            T(1));
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
    if (n <= 64) {
        return call(std::integral_constant<int, 64>{});
    }if (n <= 128) {
        return call(std::integral_constant<int, 128>{});
    }if (n <= 256) {
        return call(std::integral_constant<int, 256>{});
    }
    return call(std::integral_constant<int, 512>{});
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
