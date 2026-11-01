// SPDX-License-Identifier: MIT
// gemv pairs: N and T, matrix from global memory or staged in local memory; work-group of 64.
#include <benchmarks/sass_lab/lab.hh>

namespace lab {

template <class T, int N, int V> class KGemvNg;
template <class T, int N, int V> class KGemvTg;
template <class T, int N, int V> class KGemvNs;
template <class T, int N, int V> class KGemvTs;

constexpr int kWG = 64;

template <class T> std::size_t gemv_ws(int n) {
    return batchlas::device::gemv_workspace_elements<T>(batchlas::device::make_group_launch_info(kWG), n, n);
}

// Hand gemv N: one work-item per row (the dominant BatchLAS idiom).
template <class T, class AP, class XP>
LAB_INLINE void hand_gemv_n(int lid, int n, int ld, const AP& a, const XP& x, T* y) {
    for (int r = lid; r < n; r += kWG) {
        T acc = T(0);
#pragma unroll 4
        for (int c = 0; c < n; ++c) acc += a[r + c * ld] * x[c];
        y[r] = acc;
    }
}
// Hand gemv T: latrd gamma/delta idiom, a work-group reduction per output column.
template <class T, class AP, class XP>
LAB_INLINE void hand_gemv_t(const sycl::group<1>& g, int lid, int n, int ld, const AP& a, const XP& x, T* y) {
    for (int c = 0; c < n; ++c) {
        T partial = T(0);
        for (int r = lid; r < n; r += kWG) partial += a[r + c * ld] * x[r];
        const T s = batchlas::device::detail::reduce_sum_group(g, partial);
        if (lid == 0) y[c] = s;
    }
}

template <Transpose TR, class T, int V, class Ex>
LAB_INLINE void variant_gemv(const sycl::nd_item<1>& it, const Ex&, T* a, int n, int ld, T* x, T* y, T* ws) {
    if constexpr (is_dblas(V) || V == kDblasWs) {
        batchlas::device::gemv<TR>(dgroup<V, kWG>(it), KMV<T>(a, n, n, ld), VectorView<T>(x, n), VectorView<T>(y, n),
                                   T(1), T(0), V == kDblasWs ? ws : nullptr);
    } else if constexpr (V == kLean || V == kLeanRtWg) {
        using E = std::conditional_t<V == kLean, WgExec<kWG>, WgExecRt>;
        if constexpr (TR == Transpose::NoTrans) lean::gemv_n(E{it}, LMat<T>{a, n, n, ld}, LVec<T>{x, n}, LVec<T>{y, n});
        else lean::gemv_t(E{it}, LMat<T>{a, n, n, ld}, LVec<T>{x, n}, LVec<T>{y, n});
    } else if constexpr (V == kLeanBv) {
        if constexpr (TR == Transpose::NoTrans)
            lean::gemv_n(WgExec<kWG>{it}, KMV<T>(a, n, n, ld), VectorView<T>(x, n), VectorView<T>(y, n));
        else
            lean::gemv_t(WgExec<kWG>{it}, KMV<T>(a, n, n, ld), VectorView<T>(x, n), VectorView<T>(y, n));
    }
}

// y_b (Out column 0) := op(A_b) x_b, everything in global memory.
template <Transpose TR, class Name, class T, int N, int V> sycl::event gemv_g(sycl::queue& q, Data<T>& d, int n_rt) {
    const Data<T> dd = d;
    const std::size_t ws = V == kDblasWs ? gemv_ws<T>(ext<N>(n_rt)) : 0;
    return launch<Name, kWG, T, is_reqd(V)>(q, dd.batch, ws, 0, 0, [=](sycl::nd_item<1> it, auto lws, auto, auto) {
        const int b = static_cast<int>(it.get_group_linear_id());
        const int n = ext<N>(n_rt);
        const int ld = ext<N>(dd.ld);
        const int lid = static_cast<int>(it.get_local_linear_id());
        T* a = dd.A + static_cast<std::ptrdiff_t>(b) * Data<T>::ms;
        T* x = dd.X + static_cast<std::ptrdiff_t>(b) * Data<T>::vs;
        T* y = dd.Out + static_cast<std::ptrdiff_t>(b) * Data<T>::ms;
        if constexpr (is_hand(V)) {
            if constexpr (TR == Transpose::NoTrans) hand_gemv_n<T>(lid, n, ld, a, x, y);
            else hand_gemv_t<T>(it.get_group(), lid, n, ld, a, x, y);
        } else {
            variant_gemv<TR, T, V>(it, 0, a, n, ld, x, y, ws > 0 ? raw(lws) : nullptr);
        }
    });
}

// As gemv_g with A_b and x_b first copied to local memory (ld = n) by the same hand loop.
template <Transpose TR, class Name, class T, int N, int V> sycl::event gemv_s(sycl::queue& q, Data<T>& d, int n_rt) {
    const Data<T> dd = d;
    const int n0 = ext<N>(n_rt);
    const std::size_t ws = V == kDblasWs ? gemv_ws<T>(n0) : 0;
    return launch<Name, kWG, T, is_reqd(V)>(q, dd.batch, static_cast<std::size_t>(n0) * n0, n0, ws,
                                [=](sycl::nd_item<1> it, auto as, auto xs, auto lws) {
        const int b = static_cast<int>(it.get_group_linear_id());
        const int n = ext<N>(n_rt);
        const int ld = ext<N>(dd.ld);
        const int lid = static_cast<int>(it.get_local_linear_id());
        const T* a = dd.A + static_cast<std::ptrdiff_t>(b) * Data<T>::ms;
        const T* x = dd.X + static_cast<std::ptrdiff_t>(b) * Data<T>::vs;
        T* y = dd.Out + static_cast<std::ptrdiff_t>(b) * Data<T>::ms;
        for (int c = 0; c < n; ++c)
            for (int r = lid; r < n; r += kWG) as[r + c * n] = a[r + c * ld];
        for (int r = lid; r < n; r += kWG) xs[r] = x[r];
        it.barrier(sycl::access::fence_space::local_space);
        if constexpr (is_hand(V)) {
            if constexpr (TR == Transpose::NoTrans) hand_gemv_n<T>(lid, n, n, as, xs, y);
            else hand_gemv_t<T>(it.get_group(), lid, n, n, as, xs, y);
        } else {
            variant_gemv<TR, T, V>(it, 0, raw(as), n, n, raw(xs), y, ws > 0 ? raw(lws) : nullptr);
        }
    });
}

template <class T, int N> void reg_l2a() {
    constexpr auto NT = Transpose::NoTrans;
    constexpr auto TT = Transpose::Trans;
#define LAB_REG(OPNAME, FN, TRV, NAME)                                                             \
    add<T, N, kHand>(OPNAME, false, FN<TRV, NAME<T, N, kHand>, T, N, kHand>);                      \
    add<T, N, kDblas>(OPNAME, false, FN<TRV, NAME<T, N, kDblas>, T, N, kDblas>);                   \
    add<T, N, kLean>(OPNAME, false, FN<TRV, NAME<T, N, kLean>, T, N, kLean>);                      \
    add<T, N, kLeanRtWg>(OPNAME, false, FN<TRV, NAME<T, N, kLeanRtWg>, T, N, kLeanRtWg>);          \
    add<T, N, kLeanBv>(OPNAME, false, FN<TRV, NAME<T, N, kLeanBv>, T, N, kLeanBv>);
#define LAB_REG2(OPNAME, FN, TRV, NAME)                                                            \
    add<T, N, kDblasReqd>(OPNAME, false, FN<TRV, NAME<T, N, kDblasReqd>, T, N, kDblasReqd>);       \
    add<T, N, kDblasStatic>(OPNAME, false, FN<TRV, NAME<T, N, kDblasStatic>, T, N, kDblasStatic>); \
    add<T, N, kHandReqd>(OPNAME, false, FN<TRV, NAME<T, N, kHandReqd>, T, N, kHandReqd>);
    LAB_REG("gemvN_g", gemv_g, NT, KGemvNg)
    LAB_REG("gemvT_g", gemv_g, TT, KGemvTg)
    LAB_REG("gemvN_s", gemv_s, NT, KGemvNs)
    LAB_REG("gemvT_s", gemv_s, TT, KGemvTs)
    LAB_REG2("gemvN_g", gemv_g, NT, KGemvNg) LAB_REG2("gemvT_g", gemv_g, TT, KGemvTg)
    LAB_REG2("gemvN_s", gemv_s, NT, KGemvNs) LAB_REG2("gemvT_s", gemv_s, TT, KGemvTs)
#undef LAB_REG
#undef LAB_REG2
    add<T, N, kDblasWs>("gemvN_g", false, gemv_g<NT, KGemvNg<T, N, kDblasWs>, T, N, kDblasWs>);
    add<T, N, kDblasWs>("gemvN_s", false, gemv_s<NT, KGemvNs<T, N, kDblasWs>, T, N, kDblasWs>);
}

void register_l2a() {
    reg_l2a<float, 32>(); reg_l2a<float, 64>(); reg_l2a<float, 0>();
    reg_l2a<double, 32>(); reg_l2a<double, 64>(); reg_l2a<double, 0>();
}

}  // namespace lab
