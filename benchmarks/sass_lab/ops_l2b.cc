// SPDX-License-Identifier: MIT
// ger, symv (lower) and trmv (lower, non-unit) pairs; work-group of 64.
#include <benchmarks/sass_lab/lab.hh>

namespace lab {

template <class T, int N, int V> class KGer;
template <class T, int N, int V> class KSymv;
template <class T, int N, int V> class KTrmv;

constexpr int kWG2 = 64;

// In place on Out: A_b += alpha x_b y_b^T.
template <class T, int N, int V> sycl::event ger_op(sycl::queue& q, Data<T>& d, int n_rt) {
    const Data<T> dd = d;
    return launch<KGer<T, N, V>, kWG2, T, is_reqd(V)>(q, dd.batch, 0, 0, 0, [=](sycl::nd_item<1> it, auto, auto, auto) {
        const int b = static_cast<int>(it.get_group_linear_id());
        const int n = ext<N>(n_rt);
        const int ld = ext<N>(dd.ld);
        T* a = dd.Out + static_cast<std::ptrdiff_t>(b) * Data<T>::ms;
        T* x = dd.X + static_cast<std::ptrdiff_t>(b) * Data<T>::vs;
        T* y = dd.Y + static_cast<std::ptrdiff_t>(b) * Data<T>::vs;
        const T alpha = dd.alpha;
        if constexpr (is_hand(V)) {
            const int lid = static_cast<int>(it.get_local_linear_id());
            for (int c = 0; c < n; ++c) {
                const T yc = alpha * y[c];
                for (int r = lid; r < n; r += kWG2) a[r + c * ld] += x[r] * yc;
            }
        } else if constexpr (is_dblas(V)) {
            batchlas::device::ger(dgroup<V, kWG2>(it), VectorView<T>(x, n), VectorView<T>(y, n), KMV<T>(a, n, n, ld), alpha);
        } else if constexpr (V == kLean) {
            lean::ger(WgExec<kWG2>{it}, LVec<T>{x, n}, LVec<T>{y, n}, LMat<T>{a, n, n, ld}, alpha);
        } else if constexpr (V == kLeanRtWg) {
            lean::ger(WgExecRt{it}, LVec<T>{x, n}, LVec<T>{y, n}, LMat<T>{a, n, n, ld}, alpha);
        } else if constexpr (V == kLeanBv) {
            lean::ger(WgExec<kWG2>{it}, VectorView<T>(x, n), VectorView<T>(y, n), KMV<T>(a, n, n, ld), alpha);
        }
    });
}

// y_b (Out column 0) := op(A_b) x_b with x_b staged in local memory, IsSymv: symmetric (lower)
// else triangular lower non-unit.
template <bool IsSymv, class Name, class T, int N, int V> sycl::event tri_op(sycl::queue& q, Data<T>& d, int n_rt) {
    const Data<T> dd = d;
    const int n0 = ext<N>(n_rt);
    std::size_t ws = 0;
    if constexpr (IsSymv && V == kDblasWs)
        ws = batchlas::device::symv_workspace_elements<T, Uplo::Lower>(batchlas::device::make_group_launch_info(kWG2), n0);
    return launch<Name, kWG2, T, is_reqd(V)>(q, dd.batch, n0, ws, 0, [=](sycl::nd_item<1> it, auto xs, auto lws, auto) {
        const int b = static_cast<int>(it.get_group_linear_id());
        const int n = ext<N>(n_rt);
        const int ld = ext<N>(dd.ld);
        const int lid = static_cast<int>(it.get_local_linear_id());
        T* a = dd.A + static_cast<std::ptrdiff_t>(b) * Data<T>::ms;
        const T* x = dd.X + static_cast<std::ptrdiff_t>(b) * Data<T>::vs;
        T* y = dd.Out + static_cast<std::ptrdiff_t>(b) * Data<T>::ms;
        for (int r = lid; r < n; r += kWG2) xs[r] = x[r];
        it.barrier(sycl::access::fence_space::local_space);
        if constexpr (is_hand(V)) {
            // latrd split loop: row r of the lower triangle, then column r below the diagonal.
            for (int r = lid; r < n; r += kWG2) {
                T acc = T(0);
#pragma unroll 4
                for (int c = 0; c <= r; ++c) acc += a[r + c * ld] * xs[c];
                if constexpr (IsSymv) {
#pragma unroll 4
                    for (int c = r + 1; c < n; ++c) acc += a[c + r * ld] * xs[c];
                }
                y[r] = acc;
            }
        } else if constexpr (is_dblas(V) || V == kDblasWs) {
            const KMV<T> av(a, n, n, ld);
            const VectorView<T> xv(raw(xs), n), yv(y, n);
            if constexpr (IsSymv)
                batchlas::device::symv<Uplo::Lower>(dgroup<V, kWG2>(it), av, xv, yv, T(1), T(0),
                                                    (V == kDblasWs && ws > 0) ? raw(lws) : nullptr);
            else
                batchlas::device::trmv<Uplo::Lower, Transpose::NoTrans, Diag::NonUnit>(dgroup<V, kWG2>(it), av, xv, yv,
                                                                                      T(1), T(0));
        } else if constexpr (V == kLean || V == kLeanRtWg) {
            using E = std::conditional_t<V == kLean, WgExec<kWG2>, WgExecRt>;
            if constexpr (IsSymv) lean::symv_lower(E{it}, LMat<T>{a, n, n, ld}, LVec<T>{raw(xs), n}, LVec<T>{y, n});
            else lean::trmv_lower(E{it}, LMat<T>{a, n, n, ld}, LVec<T>{raw(xs), n}, LVec<T>{y, n});
        } else if constexpr (V == kLeanBv) {
            if constexpr (IsSymv)
                lean::symv_lower(WgExec<kWG2>{it}, KMV<T>(a, n, n, ld), VectorView<T>(raw(xs), n), VectorView<T>(y, n));
            else
                lean::trmv_lower(WgExec<kWG2>{it}, KMV<T>(a, n, n, ld), VectorView<T>(raw(xs), n), VectorView<T>(y, n));
        }
    });
}

template <class T, int N> void for_exp() {
    add<T, N, kDblasReqd>("ger", true, ger_op<T, N, kDblasReqd>);
    add<T, N, kDblasStatic>("ger", true, ger_op<T, N, kDblasStatic>);
    add<T, N, kHandReqd>("ger", true, ger_op<T, N, kHandReqd>);
    add<T, N, kDblasReqd>("symv", false, tri_op<true, KSymv<T, N, kDblasReqd>, T, N, kDblasReqd>);
    add<T, N, kDblasStatic>("symv", false, tri_op<true, KSymv<T, N, kDblasStatic>, T, N, kDblasStatic>);
    add<T, N, kHandReqd>("symv", false, tri_op<true, KSymv<T, N, kHandReqd>, T, N, kHandReqd>);
    add<T, N, kDblasReqd>("trmv", false, tri_op<false, KTrmv<T, N, kDblasReqd>, T, N, kDblasReqd>);
    add<T, N, kDblasStatic>("trmv", false, tri_op<false, KTrmv<T, N, kDblasStatic>, T, N, kDblasStatic>);
    add<T, N, kHandReqd>("trmv", false, tri_op<false, KTrmv<T, N, kHandReqd>, T, N, kHandReqd>);
}

template <class T, int N> void reg_l2b() {
    add<T, N, kHand>("ger", true, ger_op<T, N, kHand>);
    add<T, N, kDblas>("ger", true, ger_op<T, N, kDblas>);
    add<T, N, kLean>("ger", true, ger_op<T, N, kLean>);
    add<T, N, kLeanRtWg>("ger", true, ger_op<T, N, kLeanRtWg>);
    add<T, N, kLeanBv>("ger", true, ger_op<T, N, kLeanBv>);
#define LAB_REG(OPNAME, SYM, NAME, VV) add<T, N, VV>(OPNAME, false, tri_op<SYM, NAME<T, N, VV>, T, N, VV>);
    LAB_REG("symv", true, KSymv, kHand) LAB_REG("symv", true, KSymv, kDblas) LAB_REG("symv", true, KSymv, kDblasWs)
    LAB_REG("symv", true, KSymv, kLean) LAB_REG("symv", true, KSymv, kLeanRtWg) LAB_REG("symv", true, KSymv, kLeanBv)
    LAB_REG("trmv", false, KTrmv, kHand) LAB_REG("trmv", false, KTrmv, kDblas) LAB_REG("trmv", false, KTrmv, kLean)
    LAB_REG("trmv", false, KTrmv, kLeanRtWg) LAB_REG("trmv", false, KTrmv, kLeanBv)
    for_exp<T, N>();
#undef LAB_REG
}

void register_l2b() {
    reg_l2b<float, 32>(); reg_l2b<float, 64>(); reg_l2b<float, 0>();
    reg_l2b<double, 32>(); reg_l2b<double, 64>(); reg_l2b<double, 0>();
}

}  // namespace lab
