// SPDX-License-Identifier: MIT
// SASS lab: hand-written vs batchlas::device kernels of the same computation and shape.
// Built by run_lab.sh outside CMake with the library's device flags; not part of the build.
#pragma once

#include <sycl/sycl.hpp>

// Experiment "dblas_static": an executor whose size is a type-level constant, plugged into the
// unmodified batchlas::device templates through the four hooks they use. The hooks must be declared
// before the op templates are defined, so this block precedes <batchlas/blas/device.hh>.
namespace lab {
template <int WG> struct StaticGroup { sycl::group<1> g; };
template <int SG> struct StaticSubGroup { sycl::sub_group g; };
}  // namespace lab
namespace sycl {
template <int WG> inline void group_barrier(const lab::StaticGroup<WG>& s) { sycl::group_barrier(s.g); }
}  // namespace sycl
#include <batchlas/blas/device/detail/group_blas_common.hh>
namespace batchlas::device::detail {
template <int WG> inline constexpr int group_local_linear_id(const lab::StaticGroup<WG>& s) {
    return static_cast<int>(s.g.get_local_linear_id());
}
template <int WG> inline constexpr int group_local_linear_range(const lab::StaticGroup<WG>&) { return WG; }
template <int WG, typename T> inline constexpr T reduce_sum_group(const lab::StaticGroup<WG>& s, const T& v) {
    return reduce_sum_group(s.g, v);
}
template <int SG> inline constexpr int group_local_linear_id(const lab::StaticSubGroup<SG>& s) {
    return static_cast<int>(s.g.get_local_linear_id());
}
template <int SG> inline constexpr int group_local_linear_range(const lab::StaticSubGroup<SG>&) { return SG; }
template <int SG, typename T> inline constexpr T reduce_sum_group(const lab::StaticSubGroup<SG>& s, const T& v) {
    return reduce_sum_group(s.g, v);
}
}  // namespace batchlas::device::detail
// Unqualified calls inside group_blas_common.hh (group_is_leader) find these by ADL.
namespace lab {
template <int WG> inline constexpr int group_local_linear_id(const StaticGroup<WG>& s) {
    return static_cast<int>(s.g.get_local_linear_id());
}
template <int WG> inline constexpr int group_local_linear_range(const StaticGroup<WG>&) { return WG; }
template <int SG> inline constexpr int group_local_linear_id(const StaticSubGroup<SG>& s) {
    return static_cast<int>(s.g.get_local_linear_id());
}
template <int SG> inline constexpr int group_local_linear_range(const StaticSubGroup<SG>&) { return SG; }
}  // namespace lab

#include <batchlas/blas/device.hh>
#include <batchlas/util/sycl-local-accessor-helpers.hh>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#define LAB_INLINE __attribute__((always_inline)) inline

namespace lab {

using batchlas::Diag;
using batchlas::MatrixFormat;
using batchlas::Slice;
using batchlas::Transpose;
using batchlas::Uplo;
using batchlas::VectorView;
template <class T> using KMV = batchlas::KernelMatrixView<T, MatrixFormat::Dense>;

// Variants. Keep in sync with VARIANTS in sass_stats.py.
enum Variant : int {
    kHand = 0,      // hand-written, BatchLAS idiom (raw pointers, constexpr WG, accessors)
    kDblas = 1,     // batchlas::device as used in-tree (nullptr workspace)
    kDblasWs = 2,   // batchlas::device with its workspace (staged / tiled path)
    kLean = 3,      // lean API: int32 views, constexpr WG executor, always_inline, ct path, no inner barriers
    kLeanRtWg = 4,  // lean API, executor size read at run time (as batchlas::device does)
    kLeanBv = 5,    // lean algorithms over batchlas VectorView / KernelMatrixView
    kDblasReqd = 6,    // kDblas in a kernel with [[sycl::reqd_work_group_size(WG)]]
    kDblasStatic = 7,  // kDblas with a StaticGroup<WG> executor (compile-time executor size)
    kHandReqd = 8,     // kHand in a kernel with [[sycl::reqd_work_group_size(WG)]]
};
constexpr bool is_hand(int v) { return v == kHand || v == kHandReqd; }
constexpr bool is_dblas(int v) { return v == kDblas || v == kDblasReqd || v == kDblasStatic; }
constexpr bool is_reqd(int v) { return v == kDblasReqd || v == kHandReqd; }
// The executor a batchlas::device call receives under variant V.
template <int V, int WG> inline auto dgroup(const sycl::nd_item<1>& it) {
    if constexpr (V == kDblasStatic) return StaticGroup<WG>{it.get_group()};
    else return it.get_group();
}
template <int V> inline auto dsub(const sycl::sub_group& sg) {
    if constexpr (V == kDblasStatic) return StaticSubGroup<32>{sg};
    else return sg;
}

constexpr int kMaxN = 64;
constexpr int kPanelK = 4;

template <class T> struct Data {
    T* A = nullptr;    // batch x (kMaxN*kMaxN), column-major, ld = n
    T* B = nullptr;    // second matrix (gemm)
    T* X = nullptr;    // batch x kMaxN
    T* Y = nullptr;    // batch x kMaxN
    T* V = nullptr;    // panel batch x (kMaxN*kPanelK), ld = n
    T* W = nullptr;    // panel
    T* Out = nullptr;  // batch x (kMaxN*kMaxN); output of every op
    int ld = 0;
    int batch = 0;
    T alpha = T(0.5);
    T tau = T(1.25);
    static constexpr std::ptrdiff_t ms = kMaxN * kMaxN;
    static constexpr std::ptrdiff_t vs = kMaxN;
    static constexpr std::ptrdiff_t ps = kMaxN * kPanelK;
};

struct Entry {
    std::string op, type;
    int N, V;
    bool inplace;  // op updates Out in place (reset Out = A before the check)
    std::function<sycl::event(sycl::queue&, void*, int)> run;
};
std::vector<Entry>& registry();

template <int N> LAB_INLINE constexpr int ext(int rt) {
    if constexpr (N > 0) return N; else return rt;
}

// ---- executors -------------------------------------------------------------------------
template <int WG> struct WgExec {  // work-group, size known at compile time
    sycl::nd_item<1> it;
    LAB_INLINE int lid() const { return static_cast<int>(it.get_local_linear_id()); }
    LAB_INLINE constexpr int size() const { return WG; }
    template <class T> LAB_INLINE T sum(T v) const {
        return batchlas::device::detail::reduce_sum_group(it.get_group(), v);
    }
    LAB_INLINE void barrier() const { sycl::group_barrier(it.get_group()); }
    LAB_INLINE sycl::sub_group sg() const { return it.get_sub_group(); }
};
struct WgExecRt {  // work-group, size read at run time
    sycl::nd_item<1> it;
    LAB_INLINE int lid() const { return static_cast<int>(it.get_local_linear_id()); }
    LAB_INLINE int size() const { return static_cast<int>(it.get_local_range(0)); }
    template <class T> LAB_INLINE T sum(T v) const {
        return batchlas::device::detail::reduce_sum_group(it.get_group(), v);
    }
    LAB_INLINE void barrier() const { sycl::group_barrier(it.get_group()); }
    LAB_INLINE sycl::sub_group sg() const { return it.get_sub_group(); }
};
template <int SG> struct SgExec {
    sycl::sub_group g;
    LAB_INLINE int lid() const { return static_cast<int>(g.get_local_linear_id()); }
    LAB_INLINE constexpr int size() const { return SG; }
    template <class T> LAB_INLINE T sum(T v) const { return batchlas::device::detail::reduce_sum_group(g, v); }
};
struct SgExecRt {
    sycl::sub_group g;
    LAB_INLINE int lid() const { return static_cast<int>(g.get_local_linear_id()); }
    LAB_INLINE int size() const { return static_cast<int>(g.get_local_range()[0]); }
    template <class T> LAB_INLINE T sum(T v) const { return batchlas::device::detail::reduce_sum_group(g, v); }
};

// ---- lean views: pointer + int32 extents, unit stride vectors ------------------------
template <class T> struct LVec {
    T* p;
    int n;
    LAB_INLINE T& operator()(int i) const { return p[i]; }
    LAB_INLINE int size() const { return n; }
};
template <class T> struct LMat {
    T* p;
    int r, c, ld;
    LAB_INLINE T& operator()(int i, int j) const { return p[i + j * ld]; }
    LAB_INLINE int rows() const { return r; }
    LAB_INLINE int cols() const { return c; }
    LAB_INLINE LVec<T> col(int j) const { return {p + j * ld, r}; }
};
// batchlas views get the same free accessors so lean ops take either.
template <class T> LAB_INLINE LVec<T> col_of(const LMat<T>& a, int j) { return a.col(j); }
template <class T> LAB_INLINE VectorView<T> col_of(const KMV<T>& a, int j) {
    return VectorView<T>(a.data_ + static_cast<std::ptrdiff_t>(j) * a.ld_, a.rows_);
}

// ---- lean ops (caller owns barriers; every row-partitioned op uses r = lid + k*size) --
namespace lean {

template <class Ex, class X, class Y> LAB_INLINE auto dot(const Ex& ex, const X& x, const Y& y) {
    using T = std::remove_cvref_t<decltype(x(0))>;
    T acc = T(0);
    const int n = x.size();
    for (int i = ex.lid(); i < n; i += ex.size()) acc += x(i) * y(i);
    return ex.sum(acc);
}
template <class Ex, class X, class Y, class T> LAB_INLINE void axpy(const Ex& ex, const X& x, const Y& y, T a) {
    const int n = x.size();
    for (int i = ex.lid(); i < n; i += ex.size()) y(i) += a * x(i);
}
template <class Ex, class X, class T> LAB_INLINE void scal(const Ex& ex, const X& x, T a) {
    const int n = x.size();
    for (int i = ex.lid(); i < n; i += ex.size()) x(i) *= a;
}
template <class Ex, class X, class Y> LAB_INLINE void copy(const Ex& ex, const X& x, const Y& y) {
    const int n = x.size();
    for (int i = ex.lid(); i < n; i += ex.size()) y(i) = x(i);
}
// y := A x (beta = 0 is a compile-time fact: y is not read). One work-item per row.
template <class Ex, class A, class X, class Y> LAB_INLINE void gemv_n(const Ex& ex, const A& a, const X& x, const Y& y) {
    using T = std::remove_cvref_t<decltype(x(0))>;
    const int m = a.rows(), n = a.cols();
    for (int r = ex.lid(); r < m; r += ex.size()) {
        T acc = T(0);
#pragma unroll 4
        for (int c = 0; c < n; ++c) acc += a(r, c) * x(c);
        y(r) = acc;
    }
}
// y := A^T x. One sub-group per output column: shuffles only, no local memory, no barrier.
template <class Ex, class A, class X, class Y> LAB_INLINE void gemv_t(const Ex& ex, const A& a, const X& x, const Y& y) {
    using T = std::remove_cvref_t<decltype(x(0))>;
    const auto sg = ex.sg();
    const int lane = static_cast<int>(sg.get_local_linear_id());
    const int sgs = 32;
    const int nsg = ex.size() / sgs;
    const int sgid = ex.lid() / sgs;
    const int m = a.rows(), n = a.cols();
    for (int c = sgid; c < n; c += nsg) {
        T acc = T(0);
        for (int r = lane; r < m; r += sgs) acc += a(r, c) * x(r);
        acc = batchlas::device::detail::reduce_sum_group(sg, acc);
        if (lane == 0) y(c) = acc;
    }
}
// A += alpha x y^T, column loop with coalesced rows.
template <class Ex, class X, class Y, class A, class T>
LAB_INLINE void ger(const Ex& ex, const X& x, const Y& y, const A& a, T alpha) {
    const int m = a.rows(), n = a.cols();
    for (int c = 0; c < n; ++c) {
        const T yc = alpha * y(c);
        for (int r = ex.lid(); r < m; r += ex.size()) a(r, c) += x(r) * yc;
    }
}
// y := A x, A symmetric, lower triangle stored. One work-item per row, latrd split loop.
template <class Ex, class A, class X, class Y> LAB_INLINE void symv_lower(const Ex& ex, const A& a, const X& x, const Y& y) {
    using T = std::remove_cvref_t<decltype(x(0))>;
    const int n = a.rows();
    for (int r = ex.lid(); r < n; r += ex.size()) {
        T acc = T(0);
#pragma unroll 4
        for (int c = 0; c <= r; ++c) acc += a(r, c) * x(c);
#pragma unroll 4
        for (int c = r + 1; c < n; ++c) acc += a(c, r) * x(c);
        y(r) = acc;
    }
}
// y := L x, out of place, non-unit diagonal. One work-item per row.
template <class Ex, class A, class X, class Y> LAB_INLINE void trmv_lower(const Ex& ex, const A& a, const X& x, const Y& y) {
    using T = std::remove_cvref_t<decltype(x(0))>;
    const int n = a.rows();
    for (int r = ex.lid(); r < n; r += ex.size()) {
        T acc = T(0);
#pragma unroll 4
        for (int c = 0; c <= r; ++c) acc += a(r, c) * x(c);
        y(r) = acc;
    }
}
// C := A B with a per-work-item column strip of RPT rows held in registers.
template <int RPT, class Ex, class A, class B, class C>
LAB_INLINE void gemm_nn(const Ex& ex, const A& a, const B& b, const C& c) {
    using T = std::remove_cvref_t<decltype(a(0, 0))>;
    const int m = a.rows(), n = b.cols(), k = a.cols();
    const int tpc = m / RPT;  // work-items per output column
    for (int t = ex.lid(); t < tpc * n; t += ex.size()) {
        const int col = t / tpc;
        const int r0 = (t % tpc) * RPT;
        T acc[RPT];
#pragma unroll
        for (int i = 0; i < RPT; ++i) acc[i] = T(0);
        for (int kk = 0; kk < k; ++kk) {
            const T bv = b(kk, col);
#pragma unroll
            for (int i = 0; i < RPT; ++i) acc[i] += a(r0 + i, kk) * bv;
        }
#pragma unroll
        for (int i = 0; i < RPT; ++i) c(r0 + i, col) = acc[i];
    }
}

}  // namespace lean

// One nd_range launch, WG work-items per batch item, three local buffers of T (>= 1 element).
template <class Name, int WG, class T, bool Reqd = false, class F>
inline sycl::event launch(sycl::queue& q, int batch, std::size_t s0, std::size_t s1, std::size_t s2, F f) {
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<T, 1> l0(sycl::range<1>(s0 > 0 ? s0 : 1), h);
        sycl::local_accessor<T, 1> l1(sycl::range<1>(s1 > 0 ? s1 : 1), h);
        sycl::local_accessor<T, 1> l2(sycl::range<1>(s2 > 0 ? s2 : 1), h);
        const sycl::nd_range<1> r(static_cast<std::size_t>(batch) * WG, WG);
        if constexpr (Reqd)
            h.parallel_for<Name>(r, [=](sycl::nd_item<1> it) [[sycl::reqd_work_group_size(WG)]] { f(it, l0, l1, l2); });
        else
            h.parallel_for<Name>(r, [=](sycl::nd_item<1> it) { f(it, l0, l1, l2); });
    });
}
template <class T, int D> LAB_INLINE T* raw(const sycl::local_accessor<T, D>& a) {
    return batchlas::util::get_raw_ptr(a);
}

// Registration helper: one Entry per (op, T, N, V).
template <class T> inline const char* tname() { return sizeof(T) == 4 ? "float" : "double"; }
template <class T, int N, int V, class F>
inline void add(const char* op, bool inplace, F f) {
    registry().push_back(Entry{op, tname<T>(), N, V, inplace, [f](sycl::queue& q, void* d, int n) {
        return f(q, *static_cast<Data<T>*>(d), n);
    }});
}

}  // namespace lab
