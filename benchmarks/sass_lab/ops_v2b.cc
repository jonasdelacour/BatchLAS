// SPDX-License-Identifier: MIT
// v2 parity: symv lower (x staged in local memory), ger (in place), latrd column step.
#include <benchmarks/sass_lab/v2lab.hh>

namespace lab {

template <class T, int N, int V, int WG> sycl::event w_symv(sycl::queue& q, Data<T>& d, int n_rt) {
    const Data<T> dd = d;
    const int n0 = ext<N>(n_rt);
    return launch<KWSymv<T, N, V, WG>, WG, T>(q, dd.batch, n0, 0, 0, [=](sycl::nd_item<1> it, auto xs, auto, auto) {
        const int b = static_cast<int>(it.get_group_linear_id());
        const int n = ext<N>(n_rt);
        const int ld = ext<N>(dd.ld);
        const int lid = static_cast<int>(it.get_local_linear_id());
        T* a = dd.A + static_cast<std::ptrdiff_t>(b) * Data<T>::ms;
        const T* x = dd.X + static_cast<std::ptrdiff_t>(b) * Data<T>::vs;
        T* y = dd.Out + static_cast<std::ptrdiff_t>(b) * Data<T>::ms;
        if constexpr (V == kHand || V == kLean) {
            for (int r = lid; r < n; r += WG) xs[r] = x[r];
            it.barrier(sycl::access::fence_space::local_space);
            if constexpr (V == kHand) {
                for (int r = lid; r < n; r += WG) {
                    T acc = T(0);
#pragma unroll 4
                    for (int c = 0; c <= r; ++c) acc += a[r + c * ld] * xs[c];
#pragma unroll 4
                    for (int c = r + 1; c < n; ++c) acc += a[c + r * ld] * xs[c];
                    y[r] = acc;
                }
            } else {
                lean::symv_lower(WgExec<WG>{it}, LMat<T>{a, n, n, ld}, LVec<T>{raw(xs), n}, LVec<T>{y, n});
            }
        } else {
            const dv2::WorkGroup<WG> ex{it};
            const auto xl = dv2::vec<dv2::Mem::Local>(raw(xs), n);
            dv2::copy(ex, dv2::vec(x, n), xl);
            dv2::symv<Uplo::Lower>(ex, dv2::One{}, dv2::mat(a, n, n, ld), ex.publish(xl), dv2::Zero{}, dv2::vec(y, n));
        }
    });
}

// In place on Out: A_b += alpha x_b y_b^T.
template <class T, int N, int V, int WG> sycl::event w_ger(sycl::queue& q, Data<T>& d, int n_rt) {
    const Data<T> dd = d;
    return launch<KWGer<T, N, V, WG>, WG, T>(q, dd.batch, 0, 0, 0, [=](sycl::nd_item<1> it, auto, auto, auto) {
        const int b = static_cast<int>(it.get_group_linear_id());
        const int n = ext<N>(n_rt);
        const int ld = ext<N>(dd.ld);
        T* a = dd.Out + static_cast<std::ptrdiff_t>(b) * Data<T>::ms;
        T* x = dd.X + static_cast<std::ptrdiff_t>(b) * Data<T>::vs;
        T* y = dd.Y + static_cast<std::ptrdiff_t>(b) * Data<T>::vs;
        const T alpha = dd.alpha;
        if constexpr (V == kHand) {
            const int lid = static_cast<int>(it.get_local_linear_id());
            for (int c = 0; c < n; ++c) {
                const T yc = alpha * y[c];
                for (int r = lid; r < n; r += WG) a[r + c * ld] += x[r] * yc;
            }
        } else if constexpr (V == kLean) {
            lean::ger(WgExec<WG>{it}, LVec<T>{x, n}, LVec<T>{y, n}, LMat<T>{a, n, n, ld}, alpha);
        } else {
            dv2::ger(dv2::WorkGroup<WG>{it}, alpha, dv2::vec(x, n), dv2::assume_published(dv2::vec(y, n)),
                     dv2::mat(a, n, n, ld));
        }
    });
}

// One latrd column step (real T), as ops_l3.cc latrd_op:
//   v := x (local); w := A v; for p < K: w -= V(:,p) (W(:,p)^T v) + W(:,p) (V(:,p)^T v);
//   w *= tau; w += (-tau/2 v^T w) v; Out col 0 := w.
template <class T, int N, int V, int WG> sycl::event w_latrd(sycl::queue& q, Data<T>& d, int n_rt) {
    const Data<T> dd = d;
    const int n0 = ext<N>(n_rt);
    return launch<KWLatrd<T, N, V, WG>, WG, T>(q, dd.batch, n0, n0, 0, [=](sycl::nd_item<1> it, auto v_local, auto w_local, auto) {
        const int b = static_cast<int>(it.get_group_linear_id());
        const int n = ext<N>(n_rt);
        const int ld = ext<N>(dd.ld);
        const int lid = static_cast<int>(it.get_local_linear_id());
        T* a = dd.A + static_cast<std::ptrdiff_t>(b) * Data<T>::ms;
        T* x = dd.X + static_cast<std::ptrdiff_t>(b) * Data<T>::vs;
        T* vp = dd.V + static_cast<std::ptrdiff_t>(b) * Data<T>::ps;
        T* wp = dd.W + static_cast<std::ptrdiff_t>(b) * Data<T>::ps;
        T* y = dd.Out + static_cast<std::ptrdiff_t>(b) * Data<T>::ms;
        const T tau = dd.tau;
        if constexpr (V == kHand) {
            const sycl::group<1> g = it.get_group();
            for (int r = lid; r < n; r += WG) v_local[r] = x[r];
            it.barrier(sycl::access::fence_space::local_space);
            for (int r = lid; r < n; r += WG) {
                const int c_split = sycl::min(r, n - 1);
                T acc = T(0);
#pragma unroll 4
                for (int c = 0; c <= c_split; ++c) acc += a[r + c * ld] * v_local[c];
#pragma unroll 4
                for (int c = c_split + 1; c < n; ++c) acc += a[c + r * ld] * v_local[c];
                w_local[r] = acc;
            }
            for (int p = 0; p < kPanelK; ++p) {
                T gamma_partial = T(0), delta_partial = T(0);
                for (int c = lid; c < n; c += WG) {
                    const T vc = v_local[c];
                    gamma_partial += wp[c + p * ld] * vc;
                    delta_partial += vp[c + p * ld] * vc;
                }
                const T gamma = lab::group_sum(g, gamma_partial);
                const T delta = lab::group_sum(g, delta_partial);
                for (int r = lid; r < n; r += WG) w_local[r] -= vp[r + p * ld] * gamma + wp[r + p * ld] * delta;
            }
            for (int r = lid; r < n; r += WG) w_local[r] *= tau;
            T dot_partial = T(0);
            for (int r = lid; r < n; r += WG) dot_partial += v_local[r] * w_local[r];
            const T dot = lab::group_sum(g, dot_partial);
            const T alpha2 = T(-0.5) * tau * dot;
            for (int r = lid; r < n; r += WG) w_local[r] += alpha2 * v_local[r];
            for (int r = lid; r < n; r += WG) y[r] = w_local[r];
        } else if constexpr (V == kLean) {
            const WgExec<WG> ex{it};
            const LVec<T> v{raw(v_local), n}, w{raw(w_local), n};
            const LMat<T> av{a, n, n, ld}, vv{vp, n, kPanelK, ld}, wv{wp, n, kPanelK, ld};
            lean::copy(ex, LVec<T>{x, n}, v);
            it.barrier(sycl::access::fence_space::local_space);
            lean::symv_lower(ex, av, v, w);
            for (int p = 0; p < kPanelK; ++p) {
                const T gamma = lean::dot(ex, wv.col(p), v);
                const T delta = lean::dot(ex, vv.col(p), v);
                lean::axpy(ex, vv.col(p), w, -gamma);
                lean::axpy(ex, wv.col(p), w, -delta);
            }
            lean::scal(ex, w, tau);
            const T dot = lean::dot(ex, v, w);
            lean::axpy(ex, v, w, T(-0.5) * tau * dot);
            lean::copy(ex, w, LVec<T>{y, n});
        } else {
            const dv2::WorkGroup<WG> ex{it};
            const auto v = dv2::vec<dv2::Mem::Local>(raw(v_local), n);
            const auto w = dv2::vec<dv2::Mem::Local>(raw(w_local), n);
            const auto av = dv2::mat(a, n, n, ld), vv = dv2::mat(vp, n, kPanelK, ld), wv = dv2::mat(wp, n, kPanelK, ld);
            dv2::copy(ex, dv2::vec(x, n), v);
            dv2::symv<Uplo::Lower>(ex, dv2::One{}, av, ex.publish(v), dv2::Zero{}, w);  // w in v's layout
            for (int p = 0; p < kPanelK; ++p) {
                const T gamma = dv2::dot(ex, wv.col(p), v);
                const T delta = dv2::dot(ex, vv.col(p), v);
                dv2::axpy(ex, -gamma, vv.col(p), w);
                dv2::axpy(ex, -delta, wv.col(p), w);
            }
            dv2::scal(ex, tau, w);
            const T dot = dv2::dot(ex, v, w);
            dv2::axpy(ex, T(-0.5) * tau * dot, v, w);
            dv2::copy(ex, w, dv2::vec(y, n));
        }
    });
}

template <class T, int N, int WG> void reg_v2b_wg() {
#define LAB_V(VV)                                                       \
    addw<T, N, VV>("symv", WG, false, w_symv<T, N, VV, WG>);           \
    addw<T, N, VV>("ger", WG, true, w_ger<T, N, VV, WG>);              \
    addw<T, N, VV>("latrd", WG, false, w_latrd<T, N, VV, WG>);
    LAB_V(kHand) LAB_V(kLean) LAB_V(kV2)
#undef LAB_V
}

template <class T, int N> void reg_v2b() {
    reg_v2b_wg<T, N, 64>();
    reg_v2b_wg<T, N, 256>();
}

void register_v2b() {
    reg_v2b<float, 32>(); reg_v2b<float, 64>();
    reg_v2b<double, 32>(); reg_v2b<double, 64>();
}

}  // namespace lab
