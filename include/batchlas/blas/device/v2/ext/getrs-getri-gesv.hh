// SPDX-License-Identifier: MIT
#pragma once
// Device LA v2, LU-solve family (getrs, getri, gesv, potrs): the ops those kernels are made of.
//   FusedArith<T>   the arithmetic of src/sycl/device_scalar.hh (fma forms, Smith division), so
//                   a port keeps its kernel's rounding bit for bit (core Arith::mac rounds twice)
//   WorkGroupRt<SG> a work-group whose width is a launch-time value (sized from n and a
//                   register gate); core level-1 ops reach it through for_each_owned
//   MatZ            a column-major view with 64-bit index arithmetic and sub-view offsets
//   lacpy_flat, lapmr_copy, laswp, laswp_cols, perm_index, LocalMinSlot
//   trsv_lane, trsm_lane, trsm_lane_slots   lane i owns row i, the recurrence is shuffles only
//   gemm_nr<NoTrans|Trans, NR>              C -= op(A) X with X.cols() <= NR accumulators
//   frag_bcast, frag_select, RecipOrDiv, ger_frag_bcast, trsm_frag_upper_relabel

#include <climits>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <sycl/sycl.hpp>

#include <batchlas/backend_config.h>
#include <batchlas/blas/device/v2/dla.hh>
#include <batchlas/util/group-collectives.hh>

#ifndef BATCHLAS_DLA_UNROLL_FULL  // src/sycl/kernel_attrs.hh's BATCHLAS_UNROLL_FULL, which include/ cannot name
#if BATCHLAS_SYCL_IMPL_ACPP
#define BATCHLAS_DLA_UNROLL_FULL _Pragma("clang loop unroll(full)")
#else
#define BATCHLAS_DLA_UNROLL_FULL _Pragma("unroll")
#endif
#endif

namespace batchlas::device::v2 {

namespace alg {
struct FlatDivMod {};  // flat index e owned by lid == e mod size; (i, c) = (e % m, e / m)
struct Walk {};        // as FlatDivMod, but (row, col) advanced by subtraction: one divide per item
struct OnePerItem {};  // item c owns column c; the caller guarantees cols <= size
}  // namespace alg

// ---- arithmetic ----------------------------------------------------------------------------
namespace detail {
template <class T> concept MemberComplex = requires(T t) {
    t.re;
    t.im;
};
template <class T> inline constexpr bool is_std_complex_v = false;
template <class R> inline constexpr bool is_std_complex_v<std::complex<R>> = true;
template <class T> concept AnyComplex = MemberComplex<T> || is_std_complex_v<T>;
template <class T> struct complex_real {
    using type = typename T::value_type;
};
template <class T> requires MemberComplex<T> struct complex_real<T> {
    using type = std::remove_cvref_t<decltype(std::declval<T>().re)>;
};
}  // namespace detail

template <class T> struct FusedArith {
    using real_t = T;
    static constexpr bool is_complex = false;
    BATCHLAS_DLA_INLINE static T re(T a) { return a; }
    BATCHLAS_DLA_INLINE static T zero() { return T(0); }
    BATCHLAS_DLA_INLINE static T one() { return T(1); }
    BATCHLAS_DLA_INLINE static T conj(T a) { return a; }
    BATCHLAS_DLA_INLINE static T mul(T a, T b) { return a * b; }
    BATCHLAS_DLA_INLINE static T sub(T a, T b) { return a - b; }
    BATCHLAS_DLA_INLINE static T div(T a, T b) { return a / b; }
    BATCHLAS_DLA_INLINE static T recip(T d) { return T(1) / d; }
    BATCHLAS_DLA_INLINE static void fma_acc(T& acc, T a, T b) { acc = sycl::fma(a, b, acc); }
    BATCHLAS_DLA_INLINE static bool is_zero(T x) { return x == T(0); }
    BATCHLAS_DLA_INLINE static bool isfinite(T x) { return sycl::isfinite(x); }
};

template <class T> requires detail::AnyComplex<T> struct FusedArith<T> {
    static constexpr bool member = detail::MemberComplex<T>;
    using real_t = typename detail::complex_real<T>::type;
    using R = real_t;
    static constexpr bool is_complex = true;
    BATCHLAS_DLA_INLINE static R re(T a) {
        if constexpr (member) return a.re;
        else return a.real();
    }
    BATCHLAS_DLA_INLINE static R im(T a) {
        if constexpr (member) return a.im;
        else return a.imag();
    }
    BATCHLAS_DLA_INLINE static T make(R r, R i) {
        if constexpr (member) return T{r, i};
        else return T(r, i);
    }
    BATCHLAS_DLA_INLINE static T zero() { return make(R(0), R(0)); }
    BATCHLAS_DLA_INLINE static T one() { return make(R(1), R(0)); }
    BATCHLAS_DLA_INLINE static T conj(T a) { return make(re(a), -im(a)); }
    BATCHLAS_DLA_INLINE static T mul(T a, T b) {
        return make(sycl::fma(re(a), re(b), -im(a) * im(b)), sycl::fma(re(a), im(b), im(a) * re(b)));
    }
    BATCHLAS_DLA_INLINE static T sub(T a, T b) { return make(re(a) - re(b), im(a) - im(b)); }
    BATCHLAS_DLA_INLINE static void fma_acc(T& acc, T a, T b) {
        R r = re(acc), i = im(acc);
        r = sycl::fma(re(a), re(b), r);
        r = sycl::fma(-im(a), im(b), r);
        i = sycl::fma(re(a), im(b), i);
        i = sycl::fma(im(a), re(b), i);
        acc = make(r, i);
    }
    // Smith's algorithm, not (c - di) / (c^2 + d^2), which overflows.
    BATCHLAS_DLA_INLINE static T div(T a, T b) {
        if (sycl::fabs(re(b)) >= sycl::fabs(im(b))) {
            const R r = im(b) / re(b);
            const R den = sycl::fma(im(b), r, re(b));
            return make(sycl::fma(im(a), r, re(a)) / den, sycl::fma(-re(a), r, im(a)) / den);
        }
        const R r = re(b) / im(b);
        const R den = sycl::fma(re(b), r, im(b));
        return make(sycl::fma(re(a), r, im(a)) / den, sycl::fma(im(a), r, -re(a)) / den);
    }
    BATCHLAS_DLA_INLINE static T recip(T d) {
        if (sycl::fabs(re(d)) >= sycl::fabs(im(d))) {
            const R r = im(d) / re(d);
            const R den = sycl::fma(im(d), r, re(d));
            return make(R(1) / den, -r / den);
        }
        const R r = re(d) / im(d);
        const R den = sycl::fma(re(d), r, im(d));
        return make(r / den, R(-1) / den);
    }
    BATCHLAS_DLA_INLINE static bool is_zero(T x) { return re(x) == R(0) && im(x) == R(0); }
    BATCHLAS_DLA_INLINE static bool isfinite(T x) { return sycl::isfinite(re(x)) && sycl::isfinite(im(x)); }
};

// ---- executor with a launch-time width -----------------------------------------------------
// The kernel computes every member exactly as its body did (lid from get_local_id, the width it
// captured, nteams from the host or the sub-group range), so a port keeps its index arithmetic.
template <int SG = 32> struct WorkGroupRt {
    static constexpr int team_size = SG, sg_size = SG;
    static constexpr bool sub_whole = false;
    static constexpr Scope natural_scope = Scope::Group;
    sycl::nd_item<1> it;
    int lid_;
    int size_;
    int nteams_;

    BATCHLAS_DLA_INLINE int lid() const { return lid_; }
    BATCHLAS_DLA_INLINE int size() const { return size_; }
    BATCHLAS_DLA_INLINE int nteams() const { return nteams_; }
    BATCHLAS_DLA_INLINE sycl::sub_group sub() const { return it.get_sub_group(); }
    BATCHLAS_DLA_INLINE int lane() const { return static_cast<int>(it.get_sub_group().get_local_linear_id()); }
    BATCHLAS_DLA_INLINE int team() const { return static_cast<int>(it.get_sub_group().get_group_linear_id()); }
    BATCHLAS_DLA_INLINE bool leader() const { return lid_ == 0; }
    template <class T, class Op = sycl::plus<T>> BATCHLAS_DLA_INLINE T team_reduce(T v, Op op = {}) const {
        return detail::sycl_reduce(it.get_sub_group(), v, op);
    }
    template <class T, class Op = sycl::plus<T>> BATCHLAS_DLA_INLINE T group_reduce(T v, Op op = {}) const {
        return detail::sycl_reduce(it.get_group(), v, op);
    }
    // nd_item::barrier, the fused-solve kernels' spelling.
    BATCHLAS_DLA_INLINE void barrier(Mem m = Mem::Local) const {
        if (m == Mem::Local) it.barrier(sycl::access::fence_space::local_space);
        else it.barrier(sycl::access::fence_space::global_and_local);
    }
    // sycl::group_barrier, the permutation kernels' spelling.
    BATCHLAS_DLA_INLINE void group_barrier() const { sycl::group_barrier(it.get_group()); }
};

template <class Ex> inline constexpr bool is_rt_group_v = false;
template <int SG> inline constexpr bool is_rt_group_v<WorkGroupRt<SG>> = true;

namespace detail {
template <class Ex> BATCHLAS_DLA_INLINE int ex_size(const Ex& ex) {
    if constexpr (is_rt_group_v<Ex>) return ex.size();
    else return Ex::size;
}
template <class Ex> BATCHLAS_DLA_INLINE int ex_nteams(const Ex& ex) {
    if constexpr (is_rt_group_v<Ex>) return ex.nteams();
    else return Ex::nteams;
}
}  // namespace detail

// Found by ADL from every core level-1 op (map, copy, axpy, scal, swap).
template <int SG, class F> BATCHLAS_DLA_INLINE void for_each_owned(const WorkGroupRt<SG>& ex, int n, F&& f) {
    for (int i = ex.lid(); i < n; i += ex.size()) f(i);
}

// ---- operands with 64-bit index arithmetic -------------------------------------------------
// MatZ(i, j) = p[size_t(i0 + i) + size_t(j0 + j) * size_t(ld)]: a sub-view keeps the base pointer
// and adds offsets, the legacy kernels' spelling. Packed: ld == m and no offset, element e = p[e].
template <class T, bool Packed = false> struct MatZ {
    using value_type = std::remove_cv_t<T>;
    T* p;
    int m;
    int n;
    int ld;
    int i0 = 0;
    int j0 = 0;

    BATCHLAS_DLA_INLINE T& operator()(int i, int j) const {
        return p[static_cast<std::size_t>(i0 + i) + static_cast<std::size_t>(j0 + j) * static_cast<std::size_t>(ld)];
    }
    BATCHLAS_DLA_INLINE T& flat(int e) const {
        static_assert(Packed, "v2::MatZ::flat: only a Packed view is flat");
        return p[e];
    }
    BATCHLAS_DLA_INLINE int rows() const { return m; }
    BATCHLAS_DLA_INLINE int cols() const { return n; }
    BATCHLAS_DLA_INLINE T* col_ptr(int j) const {
        return p + static_cast<std::size_t>(i0) + static_cast<std::size_t>(j0 + j) * static_cast<std::size_t>(ld);
    }
    BATCHLAS_DLA_INLINE MatZ<T, false> sub(int i, int j, int r, int c) const { return {p, r, c, ld, i0 + i, j0 + j}; }
};
template <class V> inline constexpr bool is_packed_v = false;
template <class T> inline constexpr bool is_packed_v<MatZ<T, true>> = true;

// op views; Conj and ConjIf read then conjugate, ConjIf's flag is a run-time value.
template <class V> struct TransView {
    using value_type = typename V::value_type;
    V v;
    BATCHLAS_DLA_INLINE decltype(auto) operator()(int i, int j) const { return v(j, i); }
    BATCHLAS_DLA_INLINE int rows() const { return v.cols(); }
    BATCHLAS_DLA_INLINE int cols() const { return v.rows(); }
};
template <class V> struct ConjView {
    using value_type = typename V::value_type;
    V v;
    BATCHLAS_DLA_INLINE value_type operator()(int i, int j) const { return FusedArith<value_type>::conj(v(i, j)); }
    BATCHLAS_DLA_INLINE int rows() const { return v.rows(); }
    BATCHLAS_DLA_INLINE int cols() const { return v.cols(); }
};
template <class V> struct ConjIfView {
    using value_type = typename V::value_type;
    V v;
    bool c;
    BATCHLAS_DLA_INLINE value_type operator()(int i, int j) const {
        const value_type a = v(i, j);
        return c ? FusedArith<value_type>::conj(a) : a;
    }
    BATCHLAS_DLA_INLINE int rows() const { return v.rows(); }
    BATCHLAS_DLA_INLINE int cols() const { return v.cols(); }
};
template <class V> BATCHLAS_DLA_INLINE TransView<V> trans(V v) { return {v}; }
template <class V> BATCHLAS_DLA_INLINE ConjView<V> conj(V v) { return {v}; }
template <class V> BATCHLAS_DLA_INLINE ConjIfView<V> conj_if(V v, bool c) { return {v, c}; }

// LAPACK pivots: Ipiv1 reads the 1-based wire format, Ipiv0 a 0-based local copy.
struct Ipiv1 {
    const int* p;
    BATCHLAS_DLA_INLINE int operator()(int k) const { return p[k] - 1; }
};
struct Ipiv0 {
    const int* p;
    BATCHLAS_DLA_INLINE int operator()(int k) const { return p[k]; }
};

// ---- elementwise ---------------------------------------------------------------------------
namespace detail {
template <class V> BATCHLAS_DLA_INLINE decltype(auto) at_flat(const V& v, int i, int c, int e) {
    if constexpr (is_packed_v<V>) return v.flat(e);
    else return v(i, c);
}
}  // namespace detail

// f(row, col, e) for every element of an m x n block this item owns, e the column-major flat index
// (Walk passes e = 0: only FlatDivMod may address a Packed operand). Named _flat, not _2d: getrf.hh's
// for_each_owned_2d / lacpy take the same arguments and an unqualified call would be ambiguous.
template <class Alg = alg::FlatDivMod, class Ex, class F>
BATCHLAS_DLA_INLINE void for_each_owned_flat(const Ex& ex, int m, int n, F&& f) {
    const int size = detail::ex_size(ex);
    if constexpr (std::is_same_v<Alg, alg::FlatDivMod>) {
        for (int e = ex.lid(); e < m * n; e += size) {
            const int i = e % m, c = e / m;
            f(i, c, e);
        }
    } else {
        static_assert(std::is_same_v<Alg, alg::Walk>, "v2::for_each_owned_flat: alg::FlatDivMod or alg::Walk");
        int col = ex.lid() / m;
        int row = ex.lid() - col * m;
        while (col < n) {
            f(row, col, 0);
            row += size;
            while (row >= m) {
                row -= m;
                ++col;
            }
        }
    }
}

// B := A over B's extent (LAPACK lacpy 'A') over the flat traversal: a Packed operand is addressed by its flat index.
template <class Alg = alg::FlatDivMod, class Ex, class MA, class MB>
BATCHLAS_DLA_INLINE void lacpy_flat(const Ex& ex, const MA& a, const MB& b) {
    static_assert(std::is_same_v<Alg, alg::FlatDivMod> || !(is_packed_v<MA> || is_packed_v<MB>),
                  "v2::lacpy_flat: alg::Walk has no flat index; a Packed operand needs alg::FlatDivMod");
    for_each_owned_flat<Alg>(ex, b.rows(), b.cols(),
                             [&](int i, int c, int e) { detail::at_flat(b, i, c, e) = detail::at_flat(a, i, c, e); });
}

// B(i, :) := A(perm(i), :) (LAPACK lapmr, forward, out of place).
template <class Alg = alg::Walk, class Ex, class MA, class P, class MB>
BATCHLAS_DLA_INLINE void lapmr_copy(const Ex& ex, const MA& a, const P& perm, const MB& b) {
    for_each_owned_flat<Alg>(ex, b.rows(), b.cols(), [&](int i, int c, int) { b(i, c) = a(perm(i), c); });
}

// ---- interchanges --------------------------------------------------------------------------
// x := P x for the interchange list ipiv(k0 .. k1-1) (LAPACK laswp on one column). Forward
// applies k0 first (getrs NoTrans); Backward applies k1-1 first (the inverse, getrs Trans).
template <ApplyOrder O, class X, class P> BATCHLAS_DLA_INLINE void laswp(const Thread&, const X& x, const P& ipiv, int k0, int k1) {
    auto step = [&](int k) {
        const int p = ipiv(k);
        if (p != k) {
            const auto t = x(k);
            x(k) = x(p);
            x(p) = t;
        }
    };
    if constexpr (O == ApplyOrder::Forward) {
        for (int k = k0; k < k1; ++k) step(k);
    } else {
        for (int k = k1 - 1; k >= k0; --k) step(k);
    }
}

// laswp on every column of Y over all its rows; item c walks column c, serially.
template <ApplyOrder O, class Alg = alg::OnePerItem, class Ex, class MY, class P>
BATCHLAS_DLA_INLINE void laswp_cols(const Ex& ex, const MY& y, const P& ipiv) {
    static_assert(std::is_same_v<Alg, alg::OnePerItem>, "v2::laswp_cols: alg::OnePerItem is the only algorithm");
    const int c = ex.lid();
    if (c < y.cols()) laswp<O>(Thread{}, vec<Mem::Local>(y.col_ptr(c), y.rows()), ipiv, 0, y.rows());
}

// Where index i lands under the interchange list ipiv(k0 .. k1-1): one entry of laswp applied
// to the identity, with no index array.
template <ApplyOrder O, class P> BATCHLAS_DLA_INLINE int perm_index(const P& ipiv, int k0, int k1, int i) {
    int r = i;
    auto step = [&](int k) {
        const int p = ipiv(k);
        if (r == k) r = p;
        else if (r == p) r = k;
    };
    if constexpr (O == ApplyOrder::Forward) {
        for (int k = k0; k < k1; ++k) step(k);
    } else {
        for (int k = k1 - 1; k >= k0; --k) step(k);
    }
    return r;
}

// A work-group minimum in one local int through relaxed work-group atomics (no tree): LAPACK
// info, the first index that fails. reset by the leader, a barrier, offer, a barrier, info().
struct LocalMinSlot {
    int* p;
    template <class Ex> BATCHLAS_DLA_INLINE void reset(const Ex& ex) const {
        if (ex.lid() == 0) *p = INT_MAX;
    }
    BATCHLAS_DLA_INLINE void offer(int v) const {
        sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::work_group, sycl::access::address_space::local_space>(*p)
            .fetch_min(v);
    }
    BATCHLAS_DLA_INLINE int info() const {
        const int v = *p;
        return v == INT_MAX ? 0 : v;
    }
};

// ---- sub-group triangular solves: lane i owns row i ----------------------------------------
// Sum over SG lanes by shift_group_left; only lane 0 holds the total. No static local memory
// (sycl::reduce_over_group adds some and reopens the 48 KB launch hole).
template <int SG, class G, class T> BATCHLAS_DLA_INLINE T sum_to_lane0(const G& g, T v) {
    using Ar = FusedArith<T>;
    if constexpr (Ar::is_complex) {
        auto re = Ar::re(v);
        auto im = Ar::im(v);
        for (int off = SG / 2; off > 0; off >>= 1) {
            re += sycl::shift_group_left(g, re, off);
            im += sycl::shift_group_left(g, im, off);
        }
        return Ar::make(re, im);
    } else {
        for (int off = SG / 2; off > 0; off >>= 1) v += sycl::shift_group_left(g, v, off);
        return v;
    }
}

// x := op(A)^{-1} x for one column held one row per lane (rows >= A.rows() idle). Lower runs
// forward, Upper backward; Unit skips the last pivot, which updates nothing. Collective: every
// lane of the sub-group calls it, and the lane guards are inside.
template <Uplo UL, Diag DG, int SG, class MA, class T> BATCHLAS_DLA_INLINE T trsv_lane(const SubGroup<SG>& ex, const MA& a, T v) {
    using Ar = FusedArith<T>;
    const int lane = ex.lid();
    const int jb = a.rows();
    auto pivot = [&](int s) {
        if constexpr (DG == Diag::NonUnit) {
            if (lane == s) v = Ar::div(v, a(s, s));
        }
        return batchlas::portable::group_broadcast(ex.sg, v, s);
    };
    if constexpr (UL == Uplo::Lower) {
        const int end = DG == Diag::Unit ? jb - 1 : jb;
        for (int s = 0; s < end; ++s) {
            const T vs = pivot(s);
            if (lane > s && lane < jb) v = Ar::sub(v, Ar::mul(a(lane, s), vs));
        }
    } else {
        const int stop = DG == Diag::Unit ? 1 : 0;
        for (int s = jb - 1; s >= stop; --s) {
            const T vs = pivot(s);
            if (lane < s) v = Ar::sub(v, Ar::mul(a(lane, s), vs));
        }
    }
    return v;
}

// Y(:, c) := op(A)^{-1} Y(:, c) for c = c0, c0 + cs, ... < Y.cols(): one sub-group, one column at
// a time (teams take the columns round-robin).
template <Uplo UL, Diag DG, int SG, class MA, class MY>
BATCHLAS_DLA_INLINE void trsm_lane(const SubGroup<SG>& ex, const MA& a, const MY& y, int c0, int cs) {
    using T = typename MY::value_type;
    const int lane = ex.lid();
    const int jb = a.rows();
    for (int c = c0; c < y.cols(); c += cs) {
        T v = (lane < jb) ? y(lane, c) : FusedArith<T>::zero();
        v = trsv_lane<UL, DG>(ex, a, v);
        if (lane < jb) y(lane, c) = v;
    }
}

// Several right-hand sides per sub-group: slot h (lanes h*w .. h*w+w-1) of P slots solves column
// g*P + h, lane l = lane mod w owns row l. Non-unit only (Cholesky factors). An idle lane never
// divides: 0 / d takes the slow division path.
struct LaneSlots {
    int w;
    int P;
    int l;
    int h;
};
template <Uplo UL, int SG, class MA, class MY>
BATCHLAS_DLA_INLINE void trsm_lane_slots(const SubGroup<SG>& ex, const MA& a, const MY& y, int c0, int cs, LaneSlots sl) {
    using T = typename MY::value_type;
    using Ar = FusedArith<T>;
    const int jb = a.rows();
    const int nrhs = y.cols();
    for (int g = c0; g * sl.P < nrhs; g += cs) {
        const int c = g * sl.P + sl.h;
        const bool on = sl.h < sl.P && c < nrhs && sl.l < jb;
        const int cc = c < nrhs ? c : 0;
        T v = on ? y(sl.l, cc) : Ar::zero();
        auto pivot = [&](int s) {
            if (on && sl.l == s) v = Ar::div(v, a(s, s));
            return batchlas::portable::select_from_group(ex.sg, v, sl.h * sl.w + s);
        };
        if constexpr (UL == Uplo::Lower) {
            for (int s = 0; s < jb; ++s) {
                const T vs = pivot(s);
                if (on && sl.l > s) v = Ar::sub(v, Ar::mul(a(sl.l, s), vs));
            }
        } else {
            for (int s = jb - 1; s >= 0; --s) {
                const T vs = pivot(s);
                if (on && sl.l < s) v = Ar::sub(v, Ar::mul(a(sl.l, s), vs));
            }
        }
        if (on) y(sl.l, cc) = v;
    }
}

// ---- block update --------------------------------------------------------------------------
// C := C - op(A) X, X.cols() <= NR read at run time (accumulators past it are never touched).
// NoTrans, alg::RowOwner: item per row of C, k serial (coalesced down A's columns).
// Trans, alg::SgPerCol: sub-group per row of C, lanes stride k, sum_to_lane0, lane 0 writes.
// ConjTrans is Trans of conj(A) or conj_if(A, flag).
template <Transpose TA, int NR, class Ex, class MA, class MX, class MC>
BATCHLAS_DLA_INLINE void gemm_nr(const Ex& ex, NegOne, const MA& a, const MX& x, One, const MC& c) {
    using T = typename MC::value_type;
    using Ar = FusedArith<T>;
    const int nrhs = x.cols();
    if constexpr (TA == Transpose::NoTrans) {
        const int m = c.rows(), k = a.cols();
        for (int i = ex.lid(); i < m; i += detail::ex_size(ex)) {
            T acc[NR];
            BATCHLAS_DLA_UNROLL_FULL
            for (int r = 0; r < NR; ++r) acc[r] = Ar::zero();
            for (int kk = 0; kk < k; ++kk) {
                const T av = a(i, kk);
                BATCHLAS_DLA_UNROLL_FULL
                for (int r = 0; r < NR; ++r)
                    if (r < nrhs) Ar::fma_acc(acc[r], av, x(kk, r));
            }
            BATCHLAS_DLA_UNROLL_FULL
            for (int r = 0; r < NR; ++r)
                if (r < nrhs) c(i, r) = Ar::sub(c(i, r), acc[r]);
        }
    } else {
        static_assert(TA == Transpose::Trans, "v2::gemm_nr: ConjTrans is Trans of conj(A)");
        constexpr int SG = Ex::team_size;
        const int m = c.rows(), k = a.rows();
        const int lane = ex.lane();
        for (int t = ex.team(); t < m; t += detail::ex_nteams(ex)) {
            T acc[NR];
            BATCHLAS_DLA_UNROLL_FULL
            for (int r = 0; r < NR; ++r) acc[r] = Ar::zero();
            for (int i = lane; i < k; i += SG) {
                const T av = a(i, t);
                BATCHLAS_DLA_UNROLL_FULL
                for (int r = 0; r < NR; ++r)
                    if (r < nrhs) Ar::fma_acc(acc[r], av, x(i, r));
            }
            BATCHLAS_DLA_UNROLL_FULL
            for (int r = 0; r < NR; ++r)
                if (r < nrhs) {
                    const T s = sum_to_lane0<SG>(ex.sub(), acc[r]);
                    if (lane == 0) c(t, r) = Ar::sub(c(t, r), s);
                }
        }
    }
}

// ---- register fragments over a sub-group partition (the tiny tier) --------------------------
// Part is a partition handle whose select_from_group is found by ADL (SubGroupPartition).
// Collective: src must be partition-uniform and every lane must reach the call. Complex
// shuffles as two reals, in that order.
template <class Part, class T> BATCHLAS_DLA_INLINE T frag_bcast(const Part& part, T v, std::uint32_t src) {
    using Ar = FusedArith<T>;
    if constexpr (Ar::is_complex) {
        const auto r = select_from_group(part, Ar::re(v), src);
        const auto i = select_from_group(part, Ar::im(v), src);
        return Ar::make(r, i);
    } else {
        return select_from_group(part, v, src);
    }
}

// cond ? a : b component by component (a 16-byte aggregate select is not built and the
// fragment would leave the register file).
template <class T> BATCHLAS_DLA_INLINE T frag_select(bool cond, T a, T b) {
    using Ar = FusedArith<T>;
    if constexpr (Ar::is_complex) return Ar::make(cond ? Ar::re(a) : Ar::re(b), cond ? Ar::im(a) : Ar::im(b));
    else return cond ? a : b;
}

// LAPACK getf2's pivot scaling: multiply by the reciprocal when it is finite and nonzero, else
// divide (the sfmin arm), else (an exact zero pivot) the caller's singular value. Branches on the
// uniform flags, never a select: a select pays the divide every time.
template <class T> struct RecipOrDiv {
    T d;
    T rc;
    bool zero;
    bool use_mul;
    BATCHLAS_DLA_INLINE static RecipOrDiv make(T d) {
        using Ar = FusedArith<T>;
        const bool z = Ar::is_zero(d);
        const T rc = Ar::recip(d);
        return {d, rc, z, !z && Ar::isfinite(rc) && !Ar::is_zero(rc)};
    }
    BATCHLAS_DLA_INLINE T solve(T y, T if_singular) const {
        using Ar = FusedArith<T>;
        T x = if_singular;
        if (use_mul) x = Ar::mul(y, rc);
        else if (!zero) x = Ar::div(y, d);
        return x;
    }
};

// x[k] -= m * x[k]@src for k0 <= k < min(N, kend), applied where act (a rank-1 row update by the
// broadcast pivot row; the source lane is excluded by its own act flag).
template <int N, class Part, class T>
BATCHLAS_DLA_INLINE void ger_frag_bcast(const Part& part, T m, T (&x)[N], std::uint32_t src, int k0, int kend, bool act) {
    using Ar = FusedArith<T>;
    BATCHLAS_DLA_UNROLL_FULL
    for (int k = k0; k < N; ++k) {
        if (k >= kend) continue;
        const T u = frag_bcast(part, x[k], src);
        if (act) x[k] = Ar::sub(x[k], Ar::mul(m, u));
    }
}

// U X = Y on a row fragment whose rows carry lazy labels (rowid: the matrix row this lane holds;
// row_lane at lane i: the lane holding row i). a: the lane's row of U, b: its row of Y (consumed),
// x: its row of X, written where rowid == i. A singular U leaves X zero in that row.
template <int NC, int NR, class Part, class T>
BATCHLAS_DLA_INLINE void trsm_frag_upper_relabel(const Part& part, const T (&a)[NC], T (&b)[NR], T (&x)[NR], int rowid,
                                                 int row_lane, int n, int nrhs) {
    using Ar = FusedArith<T>;
    BATCHLAS_DLA_UNROLL_FULL
    for (int i = NC - 1; i >= 0; --i) {
        if (i >= n) continue;
        const std::uint32_t il = static_cast<std::uint32_t>(select_from_group(part, row_lane, static_cast<std::uint32_t>(i)));
        const auto sr = RecipOrDiv<T>::make(frag_bcast(part, a[i], il));
        BATCHLAS_DLA_UNROLL_FULL
        for (int k = 0; k < NR; ++k) {
            if (k >= nrhs) continue;
            const T xi = sr.solve(frag_bcast(part, b[k], il), Ar::zero());
            // Lane il writes x, never b: it is this step's shuffle source.
            x[k] = frag_select(rowid == i, xi, x[k]);
            const T upd = Ar::sub(b[k], Ar::mul(a[i], xi));
            b[k] = frag_select(rowid < i, upd, b[k]);
        }
    }
}

}  // namespace batchlas::device::v2
