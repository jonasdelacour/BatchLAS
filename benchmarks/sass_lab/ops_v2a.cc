// SPDX-License-Identifier: MIT
// v2 parity, level 1 and gemv: axpy(+scal), dot, nrm2, gemv N, gemv T (A, x in global memory).
#include "v2lab.hh"

namespace lab {

// In place on Out: col1 += alpha col0; col2 *= alpha.
template <class T, int N, int V, int WG> sycl::event w_axpy(sycl::queue& q, Data<T>& d, int n_rt) {
    const Data<T> dd = d;
    return launch<KWAxpy<T, N, V, WG>, WG, T>(q, dd.batch, 0, 0, 0, [=](sycl::nd_item<1> it, auto, auto, auto) {
        const int b = static_cast<int>(it.get_group_linear_id());
        const int n = ext<N>(n_rt);
        const int ld = ext<N>(dd.ld);
        T* a = dd.Out + static_cast<std::ptrdiff_t>(b) * Data<T>::ms;
        const T alpha = dd.alpha;
        if constexpr (V == kHand) {
            const int lid = static_cast<int>(it.get_local_linear_id());
            for (int r = lid; r < n; r += WG) a[r + ld] += alpha * a[r];
            for (int r = lid; r < n; r += WG) a[r + 2 * ld] *= alpha;
        } else if constexpr (V == kLean) {
            const WgExec<WG> ex{it};
            const LMat<T> av{a, n, n, ld};
            lean::axpy(ex, av.col(0), av.col(1), alpha);
            lean::scal(ex, av.col(2), alpha);
        } else {
            const dv2::WorkGroup<WG> ex{it};
            const auto av = dv2::mat(a, n, n, ld);
            dv2::axpy(ex, alpha, av.col(0), av.col(1));
            dv2::scal(ex, alpha, av.col(2));
        }
    });
}

// out[b] = x . y (Nrm2: ||x||). WG == 32: one sub-group per item; else a work-group reduction.
template <bool Nrm, class Name, class T, int N, int V, int WG> sycl::event w_dot(sycl::queue& q, Data<T>& d, int n_rt) {
    const Data<T> dd = d;
    return launch<Name, WG, T>(q, dd.batch, 0, 0, 0, [=](sycl::nd_item<1> it, auto, auto, auto) {
        const int b = static_cast<int>(it.get_group_linear_id());
        const int n = ext<N>(n_rt);
        T* x = dd.X + static_cast<std::ptrdiff_t>(b) * Data<T>::vs;
        T* y = Nrm ? x : dd.Y + static_cast<std::ptrdiff_t>(b) * Data<T>::vs;
        T r{};
        if constexpr (V == kHand) {
            const int lid = static_cast<int>(it.get_local_linear_id());
            T acc = T(0);
            for (int i = lid; i < n; i += WG) acc += x[i] * y[i];
            if constexpr (WG == 32) r = batchlas::device::detail::reduce_sum_group(it.get_sub_group(), acc);
            else r = batchlas::device::detail::reduce_sum_group(it.get_group(), acc);
            if constexpr (Nrm) r = sycl::sqrt(r);
        } else if constexpr (V == kLean) {
            if constexpr (WG == 32) r = lean::dot(SgExec<32>{it.get_sub_group()}, LVec<T>{x, n}, LVec<T>{y, n});
            else r = lean::dot(WgExec<WG>{it}, LVec<T>{x, n}, LVec<T>{y, n});
            if constexpr (Nrm) r = sycl::sqrt(r);
        } else {
            auto body = [&](const auto& ex) {
                if constexpr (!Nrm) r = dv2::dot(ex, dv2::vec(x, n), dv2::vec(y, n));
                else if constexpr (V == kV2) r = dv2::nrm2<dv2::alg::Unscaled>(ex, dv2::vec(x, n));
                else r = dv2::nrm2(ex, dv2::vec(x, n));
            };
            if constexpr (WG == 32) body(dv2::SubGroup<32>{it.get_sub_group()});
            else body(dv2::WorkGroup<WG>{it});
        }
        if (it.get_local_linear_id() == 0) dd.Out[static_cast<std::ptrdiff_t>(b) * Data<T>::ms] = r;
    });
}

// y_b (Out column 0) := op(A_b) x_b, everything in global memory.
template <Transpose TR, class Name, class T, int N, int V, int WG> sycl::event w_gemv(sycl::queue& q, Data<T>& d, int n_rt) {
    const Data<T> dd = d;
    return launch<Name, WG, T>(q, dd.batch, 0, 0, 0, [=](sycl::nd_item<1> it, auto, auto, auto) {
        const int b = static_cast<int>(it.get_group_linear_id());
        const int n = ext<N>(n_rt);
        const int ld = ext<N>(dd.ld);
        const int lid = static_cast<int>(it.get_local_linear_id());
        T* a = dd.A + static_cast<std::ptrdiff_t>(b) * Data<T>::ms;
        T* x = dd.X + static_cast<std::ptrdiff_t>(b) * Data<T>::vs;
        T* y = dd.Out + static_cast<std::ptrdiff_t>(b) * Data<T>::ms;
        if constexpr (V == kHand) {
            if constexpr (TR == Transpose::NoTrans) {
                for (int r = lid; r < n; r += WG) {
                    T acc = T(0);
#pragma unroll 4
                    for (int c = 0; c < n; ++c) acc += a[r + c * ld] * x[c];
                    y[r] = acc;
                }
            } else {  // the latrd gamma/delta idiom: a work-group reduction per output column
                for (int c = 0; c < n; ++c) {
                    T partial = T(0);
                    for (int r = lid; r < n; r += WG) partial += a[r + c * ld] * x[r];
                    const T s = batchlas::device::detail::reduce_sum_group(it.get_group(), partial);
                    if (lid == 0) y[c] = s;
                }
            }
        } else if constexpr (V == kLean) {
            if constexpr (TR == Transpose::NoTrans) lean::gemv_n(WgExec<WG>{it}, LMat<T>{a, n, n, ld}, LVec<T>{x, n}, LVec<T>{y, n});
            else lean::gemv_t(WgExec<WG>{it}, LMat<T>{a, n, n, ld}, LVec<T>{x, n}, LVec<T>{y, n});
        } else {
            dv2::gemv<TR>(dv2::WorkGroup<WG>{it}, dv2::One{}, dv2::mat(a, n, n, ld), dv2::assume_published(dv2::vec(x, n)),
                          dv2::Zero{}, dv2::vec(y, n));
        }
    });
}

template <class T, int N, int WG> void reg_v2a_wg() {
#define LAB_V(VV)                                                                                                   \
    addw<T, N, VV>("axpy", WG, true, w_axpy<T, N, VV, WG>);                                                        \
    addw<T, N, VV>("dot", WG, false, w_dot<false, KWDot<T, N, VV, WG>, T, N, VV, WG>);                              \
    addw<T, N, VV>("nrm2", WG, false, w_dot<true, KWNrm2<T, N, VV, WG>, T, N, VV, WG>);                             \
    if constexpr (WG != 32) {                                                                                       \
        addw<T, N, VV>("gemvN", WG, false, w_gemv<Transpose::NoTrans, KWGemvN<T, N, VV, WG>, T, N, VV, WG>);        \
        addw<T, N, VV>("gemvT", WG, false, w_gemv<Transpose::Trans, KWGemvT<T, N, VV, WG>, T, N, VV, WG>);          \
    }
    LAB_V(kHand) LAB_V(kLean) LAB_V(kV2)
#undef LAB_V
    addw<T, N, kV2Scaled>("nrm2", WG, false, w_dot<true, KWNrm2<T, N, kV2Scaled, WG>, T, N, kV2Scaled, WG>);
}

template <class T, int N> void reg_v2a() {
    reg_v2a_wg<T, N, 32>();
    reg_v2a_wg<T, N, 64>();
    reg_v2a_wg<T, N, 256>();
}

void register_v2a() {
    reg_v2a<float, 32>(); reg_v2a<float, 64>();
    reg_v2a<double, 32>(); reg_v2a<double, 64>();
}

}  // namespace lab
