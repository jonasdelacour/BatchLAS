// SPDX-License-Identifier: MIT
// device LA parity, level 1 and gemv: axpy(+scal), dot, nrm2, gemv N, gemv T (A, x in global memory).
#include <benchmarks/sass_lab/device_lab.hh>

namespace lab {

// In place on Out: col1 += alpha col0; col2 *= alpha.
template <class Real, int N, int V, int WG> sycl::event w_axpy(sycl::queue& q, Data<Real>& d, int n_rt) {
    const Data<Real> dd = d;
    return launch<KWAxpy<Real, N, V, WG>, WG, Real>(q, dd.batch, 0, 0, 0, [=](sycl::nd_item<1> it, auto, auto, auto) {
        const int b = static_cast<int>(it.get_group_linear_id());
        const int n = ext<N>(n_rt);
        const int ld = ext<N>(dd.ld);
        Real* a = dd.Out + static_cast<std::ptrdiff_t>(b) * Data<Real>::ms;
        const Real alpha = dd.alpha;
        if constexpr (V == kHand) {
            const int lid = static_cast<int>(it.get_local_linear_id());
            for (int r = lid; r < n; r += WG) a[r + ld] += alpha * a[r];
            for (int r = lid; r < n; r += WG) a[r + 2 * ld] *= alpha;
        } else if constexpr (V == kLean) {
            const WgExec<WG> ex{it};
            const LMat<Real> av{a, n, n, ld};
            lean::axpy(ex, av.col(0), av.col(1), alpha);
            lean::scal(ex, av.col(2), alpha);
        } else {
            const batchlas::device::WorkGroup<WG> ex{it};
            const auto av = batchlas::device::mat(a, n, n, ld);
            batchlas::device::axpy(ex, alpha, av.col(0), av.col(1));
            batchlas::device::scal(ex, alpha, av.col(2));
        }
    });
}

// out[b] = x . y (Nrm2: ||x||). WG == 32: one sub-group per item; else a work-group reduction.
template <bool Nrm, class Name, class Real, int N, int V, int WG> sycl::event w_dot(sycl::queue& q, Data<Real>& d,
                                                                                    int n_rt) {
    const Data<Real> dd = d;
    return launch<Name, WG, Real>(q, dd.batch, 0, 0, 0, [=](sycl::nd_item<1> it, auto, auto, auto) {
        const int b = static_cast<int>(it.get_group_linear_id());
        const int n = ext<N>(n_rt);
        Real* x = dd.X + static_cast<std::ptrdiff_t>(b) * Data<Real>::vs;
        Real* y = Nrm ? x : dd.Y + static_cast<std::ptrdiff_t>(b) * Data<Real>::vs;
        Real r{};
        if constexpr (V == kHand) {
            const int lid = static_cast<int>(it.get_local_linear_id());
            Real acc = Real(0);
            for (int i = lid; i < n; i += WG) acc += x[i] * y[i];
            if constexpr (WG == 32) r = lab::group_sum(it.get_sub_group(), acc);
            else r = lab::group_sum(it.get_group(), acc);
            if constexpr (Nrm) r = sycl::sqrt(r);
        } else if constexpr (V == kLean) {
            if constexpr (WG == 32) r = lean::dot(SgExec<32>{it.get_sub_group()}, LVec<Real>{x, n}, LVec<Real>{y, n});
            else r = lean::dot(WgExec<WG>{it}, LVec<Real>{x, n}, LVec<Real>{y, n});
            if constexpr (Nrm) r = sycl::sqrt(r);
        } else {
            auto body = [&](const auto& ex) {
                if constexpr (!Nrm) r = batchlas::device::dot(ex, batchlas::device::vec(x, n), batchlas::device::vec(y, n));
                else if constexpr (V == kDevice) r = batchlas::device::nrm2<batchlas::device::alg::Unscaled>(ex, batchlas::device::vec(x, n));
                else if constexpr (V == kDeviceBlue) r = batchlas::device::nrm2<batchlas::device::alg::Blue>(ex, batchlas::device::vec(x, n));
                else r = batchlas::device::nrm2<batchlas::device::alg::Scaled>(ex, batchlas::device::vec(x, n));
            };
            if constexpr (WG == 32) body(batchlas::device::SubGroup<32>{it.get_sub_group()});
            else body(batchlas::device::WorkGroup<WG>{it});
        }
        if (it.get_local_linear_id() == 0) dd.Out[static_cast<std::ptrdiff_t>(b) * Data<Real>::ms] = r;
    });
}

// out[b] = x.y, out[b] + ms/2 = y.y from one loop: the latrd gamma/delta pair. hand and lean are
// the legacy idiom (one loop, two group reductions; lean has no fused pair, so it is the hand
// body); device is dotN<2> without Scratch (one reduction per value, pack-expanded); kDeviceScratch is
// dotN<2> with Scratch (one packed exchange).
template <class Real, int N, int V, int WG> sycl::event w_dotn2(sycl::queue& q, Data<Real>& d, int n_rt) {
    const Data<Real> dd = d;
    return launch<KWDotN2<Real, N, V, WG>, WG, Real>(q, dd.batch, 0, 0, sizeof(Real) * 2 * (WG / 32), [=](sycl::nd_item<1> it, auto, auto, auto scr) {
        const int b = static_cast<int>(it.get_group_linear_id());
        const int n = ext<N>(n_rt);
        Real* x = dd.X + static_cast<std::ptrdiff_t>(b) * Data<Real>::vs;
        Real* y = dd.Y + static_cast<std::ptrdiff_t>(b) * Data<Real>::vs;
        Real r0{}, r1{};
        if constexpr (V == kHand || V == kLean) {
            const int lid = static_cast<int>(it.get_local_linear_id());
            Real a0 = Real(0), a1 = Real(0);
            for (int i = lid; i < n; i += WG) {
                const Real yi = y[i];
                a0 += x[i] * yi;
                a1 += y[i] * yi;
            }
            r0 = lab::group_sum(it.get_group(), a0);
            r1 = lab::group_sum(it.get_group(), a1);
        } else {
            const batchlas::device::WorkGroup<WG> ex{it};
            const auto xs = std::array<decltype(batchlas::device::vec(x, n)), 2>{batchlas::device::vec(x, n), batchlas::device::vec(y, n)};
            std::array<Real, 2> r;
            if constexpr (V == kDevice) r = batchlas::device::dotN<2>(ex, xs, batchlas::device::vec(y, n));
            else r = batchlas::device::dotN<2>(ex, batchlas::device::Scratch<Real, 2 * (WG / 32)>{raw(scr)}, xs, batchlas::device::vec(y, n));
            r0 = r[0], r1 = r[1];
        }
        if (it.get_local_linear_id() == 0) {
            dd.Out[static_cast<std::ptrdiff_t>(b) * Data<Real>::ms] = r0;
            dd.Out[static_cast<std::ptrdiff_t>(b) * Data<Real>::ms + 1] = r1;
        }
    });
}

// y_b (Out column 0) := op(A_b) x_b, everything in global memory.
template <Transpose TR, class Name, class Real, int N, int V, int WG> sycl::event w_gemv(sycl::queue& q, Data<Real>& d, int n_rt) {
    const Data<Real> dd = d;
    return launch<Name, WG, Real>(q, dd.batch, 0, 0, 0, [=](sycl::nd_item<1> it, auto, auto, auto) {
        const int b = static_cast<int>(it.get_group_linear_id());
        const int n = ext<N>(n_rt);
        const int ld = ext<N>(dd.ld);
        const int lid = static_cast<int>(it.get_local_linear_id());
        Real* a = dd.A + static_cast<std::ptrdiff_t>(b) * Data<Real>::ms;
        Real* x = dd.X + static_cast<std::ptrdiff_t>(b) * Data<Real>::vs;
        Real* y = dd.Out + static_cast<std::ptrdiff_t>(b) * Data<Real>::ms;
        if constexpr (V == kHand) {
            if constexpr (TR == Transpose::NoTrans) {
                for (int r = lid; r < n; r += WG) {
                    Real acc = Real(0);
#pragma unroll 4
                    for (int c = 0; c < n; ++c) acc += a[r + c * ld] * x[c];
                    y[r] = acc;
                }
            } else {  // the latrd gamma/delta idiom: a work-group reduction per output column
                for (int c = 0; c < n; ++c) {
                    Real partial = Real(0);
                    for (int r = lid; r < n; r += WG) partial += a[r + c * ld] * x[r];
                    const Real s = lab::group_sum(it.get_group(), partial);
                    if (lid == 0) y[c] = s;
                }
            }
        } else if constexpr (V == kLean) {
            if constexpr (TR == Transpose::NoTrans) lean::gemv_n(WgExec<WG>{it}, LMat<Real>{a, n, n, ld}, LVec<Real>{x, n}, LVec<Real>{y, n});
            else lean::gemv_t(WgExec<WG>{it}, LMat<Real>{a, n, n, ld}, LVec<Real>{x, n}, LVec<Real>{y, n});
        } else {
            batchlas::device::gemv<TR>(batchlas::device::WorkGroup<WG>{it}, batchlas::device::One{}, batchlas::device::mat(a, n, n, ld), batchlas::device::assume_published(batchlas::device::vec(x, n)),
                          batchlas::device::Zero{}, batchlas::device::vec(y, n));
        }
    });
}

template <class T, int N, int WG> void reg_device_a_wg() {
#define LAB_V(VV)                                                                                                   \
    addw<T, N, VV>("axpy", WG, true, w_axpy<T, N, VV, WG>);                                                        \
    addw<T, N, VV>("dot", WG, false, w_dot<false, KWDot<T, N, VV, WG>, T, N, VV, WG>);                              \
    addw<T, N, VV>("nrm2", WG, false, w_dot<true, KWNrm2<T, N, VV, WG>, T, N, VV, WG>);                             \
    if constexpr (WG != 32) {                                                                                       \
        addw<T, N, VV>("gemvN", WG, false, w_gemv<Transpose::NoTrans, KWGemvN<T, N, VV, WG>, T, N, VV, WG>);        \
        addw<T, N, VV>("gemvT", WG, false, w_gemv<Transpose::Trans, KWGemvT<T, N, VV, WG>, T, N, VV, WG>);          \
    }
    LAB_V(kHand) LAB_V(kLean) LAB_V(kDevice)
#undef LAB_V
    addw<T, N, kDeviceScaled>("nrm2", WG, false, w_dot<true, KWNrm2<T, N, kDeviceScaled, WG>, T, N, kDeviceScaled, WG>);
    addw<T, N, kDeviceBlue>("nrm2", WG, false, w_dot<true, KWNrm2<T, N, kDeviceBlue, WG>, T, N, kDeviceBlue, WG>);
    if constexpr (WG != 32) {
        addw<T, N, kHand>("dotN2", WG, false, w_dotn2<T, N, kHand, WG>);
        addw<T, N, kLean>("dotN2", WG, false, w_dotn2<T, N, kLean, WG>);
        addw<T, N, kDevice>("dotN2", WG, false, w_dotn2<T, N, kDevice, WG>);
        addw<T, N, kDeviceScratch>("dotN2", WG, false, w_dotn2<T, N, kDeviceScratch, WG>);
    }
}

template <class Real, int N> void reg_device_a() {
    reg_device_a_wg<Real, N, 32>();
    reg_device_a_wg<Real, N, 64>();
    reg_device_a_wg<Real, N, 256>();
}

void register_device_a() {
    reg_device_a<float, 32>(); reg_device_a<float, 64>();
    reg_device_a<double, 32>(); reg_device_a<double, 64>();
}

}  // namespace lab
