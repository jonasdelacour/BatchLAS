// SPDX-License-Identifier: MIT
// Level-1 pairs: sub-group dot / nrm2, work-group axpy + scal over matrix columns.
#include "lab.hh"

namespace lab {

template <class T, int N, int V> class KDot;
template <class T, int N, int V> class KNrm2;
template <class T, int N, int V> class KAxpyScal;

// out[b] = x_b . y_b, one sub-group (32) per batch item, vectors in global memory.
template <class T, int N, int V> sycl::event dot_op(sycl::queue& q, Data<T>& d, int n_rt) {
    const Data<T> dd = d;
    return launch<KDot<T, N, V>, 32, T, is_reqd(V)>(q, dd.batch, 0, 0, 0, [=](sycl::nd_item<1> it, auto, auto, auto) {
        const int b = static_cast<int>(it.get_group_linear_id());
        const int n = ext<N>(n_rt);
        T* x = dd.X + static_cast<std::ptrdiff_t>(b) * Data<T>::vs;
        T* y = dd.Y + static_cast<std::ptrdiff_t>(b) * Data<T>::vs;
        const sycl::sub_group sg = it.get_sub_group();
        T r{};
        if constexpr (is_hand(V)) {
            const int lid = static_cast<int>(sg.get_local_linear_id());
            T acc = T(0);
            for (int i = lid; i < n; i += 32) acc += x[i] * y[i];
            r = batchlas::device::detail::reduce_sum_group(sg, acc);
        } else if constexpr (is_dblas(V)) {
            r = batchlas::device::dotu(dsub<V>(sg), VectorView<T>(x, n), VectorView<T>(y, n));
        } else if constexpr (V == kLean) {
            r = lean::dot(SgExec<32>{sg}, LVec<T>{x, n}, LVec<T>{y, n});
        } else if constexpr (V == kLeanRtWg) {
            r = lean::dot(SgExecRt{sg}, LVec<T>{x, n}, LVec<T>{y, n});
        } else if constexpr (V == kLeanBv) {
            r = lean::dot(SgExec<32>{sg}, VectorView<T>(x, n), VectorView<T>(y, n));
        }
        if (it.get_local_linear_id() == 0) dd.Out[static_cast<std::ptrdiff_t>(b) * Data<T>::ms] = r;
    });
}

// out[b] = ||x_b||_2.
template <class T, int N, int V> sycl::event nrm2_op(sycl::queue& q, Data<T>& d, int n_rt) {
    const Data<T> dd = d;
    return launch<KNrm2<T, N, V>, 32, T, is_reqd(V)>(q, dd.batch, 0, 0, 0, [=](sycl::nd_item<1> it, auto, auto, auto) {
        const int b = static_cast<int>(it.get_group_linear_id());
        const int n = ext<N>(n_rt);
        T* x = dd.X + static_cast<std::ptrdiff_t>(b) * Data<T>::vs;
        const sycl::sub_group sg = it.get_sub_group();
        T r{};
        if constexpr (is_hand(V)) {
            const int lid = static_cast<int>(sg.get_local_linear_id());
            T acc = T(0);
            for (int i = lid; i < n; i += 32) acc += x[i] * x[i];
            r = sycl::sqrt(batchlas::device::detail::reduce_sum_group(sg, acc));
        } else if constexpr (is_dblas(V)) {
            const VectorView<T> xv(x, n);
            r = sycl::sqrt(batchlas::device::dotc(dsub<V>(sg), xv, xv));
        } else if constexpr (V == kLean) {
            r = sycl::sqrt(lean::dot(SgExec<32>{sg}, LVec<T>{x, n}, LVec<T>{x, n}));
        } else if constexpr (V == kLeanRtWg) {
            r = sycl::sqrt(lean::dot(SgExecRt{sg}, LVec<T>{x, n}, LVec<T>{x, n}));
        } else if constexpr (V == kLeanBv) {
            const VectorView<T> xv(x, n);
            r = sycl::sqrt(lean::dot(SgExec<32>{sg}, xv, xv));
        }
        if (it.get_local_linear_id() == 0) dd.Out[static_cast<std::ptrdiff_t>(b) * Data<T>::ms] = r;
    });
}

// In place on Out: col1 += alpha col0; col2 *= alpha. Work-group of 64, two sequential loops.
template <class T, int N, int V> sycl::event axpy_scal_op(sycl::queue& q, Data<T>& d, int n_rt) {
    const Data<T> dd = d;
    constexpr int WG = 64;
    return launch<KAxpyScal<T, N, V>, WG, T, is_reqd(V)>(q, dd.batch, 0, 0, 0, [=](sycl::nd_item<1> it, auto, auto, auto) {
        const int b = static_cast<int>(it.get_group_linear_id());
        const int n = ext<N>(n_rt);
        const int ld = ext<N>(dd.ld);
        T* a = dd.Out + static_cast<std::ptrdiff_t>(b) * Data<T>::ms;
        const T alpha = dd.alpha;
        if constexpr (is_hand(V)) {
            const int lid = static_cast<int>(it.get_local_linear_id());
            for (int r = lid; r < n; r += WG) a[r + ld] += alpha * a[r];
            for (int r = lid; r < n; r += WG) a[r + 2 * ld] *= alpha;
        } else if constexpr (is_dblas(V)) {
            const auto g = dgroup<V, WG>(it);
            const KMV<T> av(a, n, n, ld);
            batchlas::device::axpy(g, av(Slice(), 0), av(Slice(), 1), alpha);
            batchlas::device::scal(g, av(Slice(), 2), alpha);
        } else if constexpr (V == kLean || V == kLeanRtWg) {
            using Ex = std::conditional_t<V == kLean, WgExec<WG>, WgExecRt>;
            const Ex ex{it};
            const LMat<T> av{a, n, n, ld};
            lean::axpy(ex, av.col(0), av.col(1), alpha);
            lean::scal(ex, av.col(2), alpha);
        } else if constexpr (V == kLeanBv) {
            const WgExec<WG> ex{it};
            const KMV<T> av(a, n, n, ld);
            lean::axpy(ex, av(Slice(), 0), av(Slice(), 1), alpha);
            lean::scal(ex, av(Slice(), 2), alpha);
        }
    });
}

template <class T, int N> void reg_l1() {
    add<T, N, kHand>("dot", false, dot_op<T, N, kHand>);
    add<T, N, kDblas>("dot", false, dot_op<T, N, kDblas>);
    add<T, N, kLean>("dot", false, dot_op<T, N, kLean>);
    add<T, N, kLeanRtWg>("dot", false, dot_op<T, N, kLeanRtWg>);
    add<T, N, kLeanBv>("dot", false, dot_op<T, N, kLeanBv>);
    add<T, N, kHand>("nrm2", false, nrm2_op<T, N, kHand>);
    add<T, N, kDblas>("nrm2", false, nrm2_op<T, N, kDblas>);
    add<T, N, kLean>("nrm2", false, nrm2_op<T, N, kLean>);
    add<T, N, kHand>("axpy_scal", true, axpy_scal_op<T, N, kHand>);
    add<T, N, kDblas>("axpy_scal", true, axpy_scal_op<T, N, kDblas>);
    add<T, N, kLean>("axpy_scal", true, axpy_scal_op<T, N, kLean>);
    add<T, N, kLeanRtWg>("axpy_scal", true, axpy_scal_op<T, N, kLeanRtWg>);
    add<T, N, kLeanBv>("axpy_scal", true, axpy_scal_op<T, N, kLeanBv>);
#define LAB_EXP(OP, FN, INPL)                                              \
    add<T, N, kDblasReqd>(OP, INPL, FN<T, N, kDblasReqd>);                 \
    add<T, N, kDblasStatic>(OP, INPL, FN<T, N, kDblasStatic>);             \
    add<T, N, kHandReqd>(OP, INPL, FN<T, N, kHandReqd>);
    LAB_EXP("dot", dot_op, false) LAB_EXP("nrm2", nrm2_op, false) LAB_EXP("axpy_scal", axpy_scal_op, true)
#undef LAB_EXP
}

void register_l1() {
    reg_l1<float, 32>(); reg_l1<float, 64>(); reg_l1<float, 0>();
    reg_l1<double, 32>(); reg_l1<double, 64>(); reg_l1<double, 0>();
}

}  // namespace lab
