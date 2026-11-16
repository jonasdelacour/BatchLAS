// SPDX-License-Identifier: MIT
// device LA parity: symv lower (x staged in local memory), ger (in place), latrd column step.
#include <benchmarks/sass_lab/device_lab.hh>

namespace lab {

template <class Real, int N, int V, int WG> sycl::event w_symv(sycl::queue& q, Data<Real>& d, int n_rt) {
    const Data<Real> dd = d;
    const int n0 = ext<N>(n_rt);
    return launch<KWSymv<Real, N, V, WG>, WG, Real>(q, dd.batch, n0, 0, 0,
                                                    [=](sycl::nd_item<1> it, auto xs, auto, auto) {
        const int b = static_cast<int>(it.get_group_linear_id());
        const int n = ext<N>(n_rt);
        const int ld = ext<N>(dd.ld);
        const int lid = static_cast<int>(it.get_local_linear_id());
        Real* a = dd.A + static_cast<std::ptrdiff_t>(b) * Data<Real>::ms;
        const Real* x = dd.X + static_cast<std::ptrdiff_t>(b) * Data<Real>::vs;
        Real* y = dd.Out + static_cast<std::ptrdiff_t>(b) * Data<Real>::ms;
        if constexpr (V == kHand || V == kLean) {
            for (int r = lid; r < n; r += WG) xs[r] = x[r];
            it.barrier(sycl::access::fence_space::local_space);
            if constexpr (V == kHand) {
                for (int r = lid; r < n; r += WG) {
                    Real acc = Real(0);
#pragma unroll 4
                    for (int c = 0; c <= r; ++c) acc += a[r + c * ld] * xs[c];
#pragma unroll 4
                    for (int c = r + 1; c < n; ++c) acc += a[c + r * ld] * xs[c];
                    y[r] = acc;
                }
            } else {
                lean::symv_lower(WgExec<WG>{it}, LMat<Real>{a, n, n, ld}, LVec<Real>{raw(xs), n}, LVec<Real>{y, n});
            }
        } else {
            const batchlas::device::WorkGroup<WG> ex{it};
            const auto xl = batchlas::device::vec<batchlas::device::Mem::Local>(raw(xs), n);
            batchlas::device::copy(ex, batchlas::device::vec(x, n), xl);
            batchlas::device::symv<Uplo::Lower>(ex, batchlas::device::One{}, batchlas::device::mat(a, n, n, ld), ex.publish(xl), batchlas::device::Zero{}, batchlas::device::vec(y, n));
        }
    });
}

// In place on Out: A_b += alpha x_b y_b^T.
template <class Real, int N, int V, int WG> sycl::event w_ger(sycl::queue& q, Data<Real>& d, int n_rt) {
    const Data<Real> dd = d;
    return launch<KWGer<Real, N, V, WG>, WG, Real>(q, dd.batch, 0, 0, 0, [=](sycl::nd_item<1> it, auto, auto, auto) {
        const int b = static_cast<int>(it.get_group_linear_id());
        const int n = ext<N>(n_rt);
        const int ld = ext<N>(dd.ld);
        Real* a = dd.Out + static_cast<std::ptrdiff_t>(b) * Data<Real>::ms;
        Real* x = dd.X + static_cast<std::ptrdiff_t>(b) * Data<Real>::vs;
        Real* y = dd.Y + static_cast<std::ptrdiff_t>(b) * Data<Real>::vs;
        const Real alpha = dd.alpha;
        if constexpr (V == kHand) {
            const int lid = static_cast<int>(it.get_local_linear_id());
            for (int c = 0; c < n; ++c) {
                const Real yc = alpha * y[c];
                for (int r = lid; r < n; r += WG) a[r + c * ld] += x[r] * yc;
            }
        } else if constexpr (V == kLean) {
            lean::ger(WgExec<WG>{it}, LVec<Real>{x, n}, LVec<Real>{y, n}, LMat<Real>{a, n, n, ld}, alpha);
        } else {
            batchlas::device::ger(batchlas::device::WorkGroup<WG>{it}, alpha, batchlas::device::vec(x, n), batchlas::device::assume_published(batchlas::device::vec(y, n)),
                     batchlas::device::mat(a, n, n, ld));
        }
    });
}

// One latrd column step (real types), as ops_l3.cc latrd_op:
//   v := x (local); w := A v; for p < K: w -= V(:,p) (W(:,p)^T v) + W(:,p) (V(:,p)^T v);
//   w *= tau; w += (-tau/2 v^T w) v; Out col 0 := w.
template <class Real, int N, int V, int WG> sycl::event w_latrd(sycl::queue& q, Data<Real>& d, int n_rt) {
    const Data<Real> dd = d;
    const int n0 = ext<N>(n_rt);
    return launch<KWLatrd<Real, N, V, WG>, WG, Real>(q, dd.batch, n0, n0, 0, [=](sycl::nd_item<1> it, auto v_local, auto w_local, auto) {
        const int b = static_cast<int>(it.get_group_linear_id());
        const int n = ext<N>(n_rt);
        const int ld = ext<N>(dd.ld);
        const int lid = static_cast<int>(it.get_local_linear_id());
        Real* a = dd.A + static_cast<std::ptrdiff_t>(b) * Data<Real>::ms;
        Real* x = dd.X + static_cast<std::ptrdiff_t>(b) * Data<Real>::vs;
        Real* vp = dd.V + static_cast<std::ptrdiff_t>(b) * Data<Real>::ps;
        Real* wp = dd.W + static_cast<std::ptrdiff_t>(b) * Data<Real>::ps;
        Real* y = dd.Out + static_cast<std::ptrdiff_t>(b) * Data<Real>::ms;
        const Real tau = dd.tau;
        if constexpr (V == kHand) {
            const sycl::group<1> g = it.get_group();
            for (int r = lid; r < n; r += WG) v_local[r] = x[r];
            it.barrier(sycl::access::fence_space::local_space);
            for (int r = lid; r < n; r += WG) {
                const int c_split = sycl::min(r, n - 1);
                Real acc = Real(0);
#pragma unroll 4
                for (int c = 0; c <= c_split; ++c) acc += a[r + c * ld] * v_local[c];
#pragma unroll 4
                for (int c = c_split + 1; c < n; ++c) acc += a[c + r * ld] * v_local[c];
                w_local[r] = acc;
            }
            for (int p = 0; p < kPanelK; ++p) {
                Real gamma_partial = Real(0), delta_partial = Real(0);
                for (int c = lid; c < n; c += WG) {
                    const Real vc = v_local[c];
                    gamma_partial += wp[c + p * ld] * vc;
                    delta_partial += vp[c + p * ld] * vc;
                }
                const Real gamma = lab::group_sum(g, gamma_partial);
                const Real delta = lab::group_sum(g, delta_partial);
                for (int r = lid; r < n; r += WG) w_local[r] -= vp[r + p * ld] * gamma + wp[r + p * ld] * delta;
            }
            for (int r = lid; r < n; r += WG) w_local[r] *= tau;
            Real dot_partial = Real(0);
            for (int r = lid; r < n; r += WG) dot_partial += v_local[r] * w_local[r];
            const Real dot = lab::group_sum(g, dot_partial);
            const Real alpha2 = Real(-0.5) * tau * dot;
            for (int r = lid; r < n; r += WG) w_local[r] += alpha2 * v_local[r];
            for (int r = lid; r < n; r += WG) y[r] = w_local[r];
        } else if constexpr (V == kLean) {
            const WgExec<WG> ex{it};
            const LVec<Real> v{raw(v_local), n}, w{raw(w_local), n};
            const LMat<Real> av{a, n, n, ld}, vv{vp, n, kPanelK, ld}, wv{wp, n, kPanelK, ld};
            lean::copy(ex, LVec<Real>{x, n}, v);
            it.barrier(sycl::access::fence_space::local_space);
            lean::symv_lower(ex, av, v, w);
            for (int p = 0; p < kPanelK; ++p) {
                const Real gamma = lean::dot(ex, wv.col(p), v);
                const Real delta = lean::dot(ex, vv.col(p), v);
                lean::axpy(ex, vv.col(p), w, -gamma);
                lean::axpy(ex, wv.col(p), w, -delta);
            }
            lean::scal(ex, w, tau);
            const Real dot = lean::dot(ex, v, w);
            lean::axpy(ex, v, w, Real(-0.5) * tau * dot);
            lean::copy(ex, w, LVec<Real>{y, n});
        } else {
            const batchlas::device::WorkGroup<WG> ex{it};
            const auto v = batchlas::device::vec<batchlas::device::Mem::Local>(raw(v_local), n);
            const auto w = batchlas::device::vec<batchlas::device::Mem::Local>(raw(w_local), n);
            const auto av = batchlas::device::mat(a, n, n, ld), vv = batchlas::device::mat(vp, n, kPanelK, ld), wv = batchlas::device::mat(wp, n, kPanelK, ld);
            batchlas::device::copy(ex, batchlas::device::vec(x, n), v);
            batchlas::device::symv<Uplo::Lower>(ex, batchlas::device::One{}, av, ex.publish(v), batchlas::device::Zero{}, w);  // w in v's layout
            for (int p = 0; p < kPanelK; ++p) {
                const Real gamma = batchlas::device::dot(ex, wv.col(p), v);
                const Real delta = batchlas::device::dot(ex, vv.col(p), v);
                batchlas::device::axpy(ex, -gamma, vv.col(p), w);
                batchlas::device::axpy(ex, -delta, wv.col(p), w);
            }
            batchlas::device::scal(ex, tau, w);
            const Real dot = batchlas::device::dot(ex, v, w);
            batchlas::device::axpy(ex, Real(-0.5) * tau * dot, v, w);
            batchlas::device::copy(ex, w, batchlas::device::vec(y, n));
        }
    });
}

template <class T, int N, int WG> void reg_device_b_wg() {
#define LAB_V(VV)                                                       \
    addw<T, N, VV>("symv", WG, false, w_symv<T, N, VV, WG>);           \
    addw<T, N, VV>("ger", WG, true, w_ger<T, N, VV, WG>);              \
    addw<T, N, VV>("latrd", WG, false, w_latrd<T, N, VV, WG>);
    LAB_V(kHand) LAB_V(kLean) LAB_V(kDevice)
#undef LAB_V
}

template <class Real, int N> void reg_device_b() {
    reg_device_b_wg<Real, N, 64>();
    reg_device_b_wg<Real, N, 256>();
}

void register_device_b() {
    reg_device_b<float, 32>(); reg_device_b<float, 64>();
    reg_device_b<double, 32>(); reg_device_b<double, 64>();
}

}  // namespace lab
