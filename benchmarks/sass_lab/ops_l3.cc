// SPDX-License-Identifier: MIT
// A latrd-style fused column step and a 32x32x32 gemm from local memory.
#include <benchmarks/sass_lab/lab.hh>

namespace lab {

template <class T, int N, int V> class KLatrd;
template <class T, int N, int V> class KGemm;

constexpr int kWG3 = 64;
constexpr int kWGG = 256;

// One latrd column step (real T), as src/ops/syev/sytrd/latrd_lower_panel.cc does it:
//   v := x (to local); w := A v (A symmetric lower); for p < K: w -= V(:,p) (W(:,p)^T v) + W(:,p) (V(:,p)^T v);
//   w *= tau; w += (-tau/2 v^T w) v; Out col 0 := w.
template <class T, int N, int V> sycl::event latrd_op(sycl::queue& q, Data<T>& d, int n_rt) {
    const Data<T> dd = d;
    const int n0 = ext<N>(n_rt);
    std::size_t ws = 0;
    if constexpr (V == kDblasWs)
        ws = batchlas::device::symv_workspace_elements<T, Uplo::Lower>(batchlas::device::make_group_launch_info(kWG3), n0);
    return launch<KLatrd<T, N, V>, kWG3, T, is_reqd(V)>(q, dd.batch, n0, n0, ws, [=](sycl::nd_item<1> it, auto v_local, auto w_local, auto lws) {
        const int b = static_cast<int>(it.get_group_linear_id());
        const int n = ext<N>(n_rt);
        const int ld = ext<N>(dd.ld);
        const int lid = static_cast<int>(it.get_local_linear_id());
        const sycl::group<1> g = it.get_group();
        T* a = dd.A + static_cast<std::ptrdiff_t>(b) * Data<T>::ms;
        T* x = dd.X + static_cast<std::ptrdiff_t>(b) * Data<T>::vs;
        T* vp = dd.V + static_cast<std::ptrdiff_t>(b) * Data<T>::ps;
        T* wp = dd.W + static_cast<std::ptrdiff_t>(b) * Data<T>::ps;
        T* y = dd.Out + static_cast<std::ptrdiff_t>(b) * Data<T>::ms;
        const T tau = dd.tau;
        if constexpr (is_hand(V)) {
            for (int r = lid; r < n; r += kWG3) v_local[r] = x[r];
            it.barrier(sycl::access::fence_space::local_space);
            for (int r = lid; r < n; r += kWG3) {
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
                for (int c = lid; c < n; c += kWG3) {
                    const T vc = v_local[c];
                    gamma_partial += wp[c + p * ld] * vc;
                    delta_partial += vp[c + p * ld] * vc;
                }
                const T gamma = batchlas::device::detail::reduce_sum_group(g, gamma_partial);
                const T delta = batchlas::device::detail::reduce_sum_group(g, delta_partial);
                for (int r = lid; r < n; r += kWG3) w_local[r] -= vp[r + p * ld] * gamma + wp[r + p * ld] * delta;
            }
            for (int r = lid; r < n; r += kWG3) w_local[r] *= tau;
            T dot_partial = T(0);
            for (int r = lid; r < n; r += kWG3) dot_partial += v_local[r] * w_local[r];
            const T dot = batchlas::device::detail::reduce_sum_group(g, dot_partial);
            const T alpha2 = T(-0.5) * tau * dot;
            for (int r = lid; r < n; r += kWG3) w_local[r] += alpha2 * v_local[r];
            for (int r = lid; r < n; r += kWG3) y[r] = w_local[r];
        } else if constexpr (is_dblas(V) || V == kDblasWs) {
            const auto gx = dgroup<V, kWG3>(it);
            // As latrd_lower_panel_batched_wg_device: one device op per step, barriers between.
            T* v_ptr = raw(v_local);
            T* w_ptr = raw(w_local);
            const VectorView<T> v_view(v_ptr, n), w_view(w_ptr, n);
            const KMV<T> av(a, n, n, ld), vv(vp, n, kPanelK, ld), wv(wp, n, kPanelK, ld);
            batchlas::device::copy(gx, VectorView<T>(x, n), v_view);
            it.barrier(sycl::access::fence_space::local_space);
            batchlas::device::symv<Uplo::Lower>(gx, av, v_view, w_view, T(1), T(0), (V == kDblasWs && ws > 0) ? raw(lws) : nullptr);
            it.barrier(sycl::access::fence_space::local_space);
            for (int p = 0; p < kPanelK; ++p) {
                const auto vcol = vv(Slice(), p);
                const auto wcol = wv(Slice(), p);
                const T gamma = batchlas::device::dotc(gx, wcol, v_view);
                const T delta = batchlas::device::dotc(gx, vcol, v_view);
                for (int r = lid; r < n; r += kWG3) w_local[r] -= vcol(r) * gamma + wcol(r) * delta;
            }
            it.barrier(sycl::access::fence_space::local_space);
            batchlas::device::scal(gx, w_view, tau);
            it.barrier(sycl::access::fence_space::local_space);
            const T dot = batchlas::device::dotc(gx, v_view, w_view);
            batchlas::device::axpy(gx, v_view, w_view, T(-0.5) * tau * dot);
            it.barrier(sycl::access::fence_space::local_space);
            batchlas::device::copy(gx, w_view, VectorView<T>(y, n));
        } else {
            // Lean op sequence: every row op uses the same r = lid + k*WG partition, so only the
            // staging of v needs a barrier (the reductions carry their own).
            auto body = [&](const auto& ex) {
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
            };
            if constexpr (V == kLean) body(WgExec<kWG3>{it});
            else body(WgExecRt{it});
        }
    });
}

template <class T> std::size_t gemm_ws(int n) {
    return batchlas::device::gemm_workspace_elements<T>(batchlas::device::make_nd_item_1d_launch_info(kWGG, 32), n, n, n);
}

// Out_b := A_b B_b with A_b, B_b staged in local memory (ld = n); 256 work-items.
template <class T, int N, int V> sycl::event gemm_op(sycl::queue& q, Data<T>& d, int n_rt) {
    const Data<T> dd = d;
    const int n0 = ext<N>(n_rt);
    const std::size_t ws = V == kDblasWs ? gemm_ws<T>(n0) : 0;
    const std::size_t nn = static_cast<std::size_t>(n0) * n0;
    return launch<KGemm<T, N, V>, kWGG, T, is_reqd(V)>(q, dd.batch, nn, nn, ws, [=](sycl::nd_item<1> it, auto as, auto bs, auto lws) {
        const int b = static_cast<int>(it.get_group_linear_id());
        const int n = ext<N>(n_rt);
        const int ld = ext<N>(dd.ld);
        const int lid = static_cast<int>(it.get_local_linear_id());
        const T* a = dd.A + static_cast<std::ptrdiff_t>(b) * Data<T>::ms;
        const T* bm = dd.B + static_cast<std::ptrdiff_t>(b) * Data<T>::ms;
        T* c = dd.Out + static_cast<std::ptrdiff_t>(b) * Data<T>::ms;
        for (int idx = lid; idx < n * n; idx += kWGG) {
            const int r = idx % n, k = idx / n;
            as[idx] = a[r + k * ld];
            bs[idx] = bm[r + k * ld];
        }
        it.barrier(sycl::access::fence_space::local_space);
        constexpr int RPT = 4;  // 32 x 32 / 256 work-items
        if constexpr (is_hand(V)) {
            const int tpc = n / RPT;
            for (int t = lid; t < tpc * n; t += kWGG) {
                const int col = t / tpc, r0 = (t % tpc) * RPT;
                T acc[RPT];
#pragma unroll
                for (int i = 0; i < RPT; ++i) acc[i] = T(0);
                for (int k = 0; k < n; ++k) {
                    const T bv = bs[k + col * n];
#pragma unroll
                    for (int i = 0; i < RPT; ++i) acc[i] += as[r0 + i + k * n] * bv;
                }
#pragma unroll
                for (int i = 0; i < RPT; ++i) c[r0 + i + col * ld] = acc[i];
            }
        } else if constexpr (V == kDblas || V == kDblasReqd || V == kDblasWs) {
            batchlas::device::gemm(it, KMV<T>(raw(as), n, n, n), KMV<T>(raw(bs), n, n, n), KMV<T>(c, n, n, ld),
                                   T(1), T(0), (V == kDblasWs && ws > 0) ? raw(lws) : nullptr);
        } else if constexpr (V == kLean || V == kLeanRtWg) {
            using E = std::conditional_t<V == kLean, WgExec<kWGG>, WgExecRt>;
            lean::gemm_nn<RPT>(E{it}, LMat<T>{raw(as), n, n, n}, LMat<T>{raw(bs), n, n, n}, LMat<T>{c, n, n, ld});
        } else if constexpr (V == kLeanBv) {
            lean::gemm_nn<RPT>(WgExec<kWGG>{it}, KMV<T>(raw(as), n, n, n), KMV<T>(raw(bs), n, n, n), KMV<T>(c, n, n, ld));
        }
    });
}

template <class T, int N> void reg_l3() {
    add<T, N, kHand>("latrd", false, latrd_op<T, N, kHand>);
    add<T, N, kDblas>("latrd", false, latrd_op<T, N, kDblas>);
    add<T, N, kDblasWs>("latrd", false, latrd_op<T, N, kDblasWs>);
    add<T, N, kLean>("latrd", false, latrd_op<T, N, kLean>);
    add<T, N, kLeanRtWg>("latrd", false, latrd_op<T, N, kLeanRtWg>);
    add<T, N, kDblasReqd>("latrd", false, latrd_op<T, N, kDblasReqd>);
    add<T, N, kDblasStatic>("latrd", false, latrd_op<T, N, kDblasStatic>);
    add<T, N, kHandReqd>("latrd", false, latrd_op<T, N, kHandReqd>);
}
template <class T, int N> void reg_gemm() {
    add<T, N, kHand>("gemm", false, gemm_op<T, N, kHand>);
    add<T, N, kDblas>("gemm", false, gemm_op<T, N, kDblas>);
    add<T, N, kDblasWs>("gemm", false, gemm_op<T, N, kDblasWs>);
    add<T, N, kLean>("gemm", false, gemm_op<T, N, kLean>);
    add<T, N, kLeanRtWg>("gemm", false, gemm_op<T, N, kLeanRtWg>);
    add<T, N, kLeanBv>("gemm", false, gemm_op<T, N, kLeanBv>);
    add<T, N, kDblasReqd>("gemm", false, gemm_op<T, N, kDblasReqd>);
    add<T, N, kHandReqd>("gemm", false, gemm_op<T, N, kHandReqd>);
}

void register_l3() {
    reg_l3<float, 32>(); reg_l3<float, 64>(); reg_l3<float, 0>();
    reg_l3<double, 32>(); reg_l3<double, 64>(); reg_l3<double, 0>();
    reg_gemm<float, 32>(); reg_gemm<float, 0>();
    reg_gemm<double, 32>(); reg_gemm<double, 0>();
}

}  // namespace lab
