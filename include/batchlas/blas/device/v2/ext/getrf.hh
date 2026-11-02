// SPDX-License-Identifier: MIT
#pragma once
// Device LA v2, LU family: ?GETF2 (explicit row swaps on a Mat, and the lazy-relabel form on a
// RowFrag), the (value, index) pivot search, ?RSCL, the flat rank-1 update, ?LASWP, 2-D owned
// traversals, ?LACPY, and DynGroup, the executor for kernels whose launch picks the width at
// run time. Additive: the core headers are unchanged. Scalar arithmetic is fp::, whose complex
// forms are the fma-written ones of src/sycl/device_scalar.hh, so ports stay bit-identical.

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <batchlas/blas/device/v2/level1.hh>

#if BATCHLAS_SYCL_IMPL_ACPP
#define BATCHLAS_DLA_UNROLL_FULL _Pragma("clang loop unroll(full)")
#else
#define BATCHLAS_DLA_UNROLL_FULL _Pragma("unroll")
#endif

namespace batchlas::device::v2 {

// ---- scalar arithmetic (complex: explicit fma forms, Smith's division) ----------------------
// Plain inline, as src/sycl/device_scalar.hh's dev_* are, and a POD complex with .re/.im
// (sycl_device::Cx) is spelled through its fields: Arith's by-value re()/im()/make() pass the
// pair through an i64, instcombine then folds the zero tests into an integer mask, and the
// CTA panel's SASS drifts (SHFL class +-1, sm_89).
namespace fp {
template <class T> concept pod_complex = requires(T t) {
    t.re;
    t.im;
};
template <class T> inline bool is_zero(T x) {
    using R = real_t<T>;
    if constexpr (pod_complex<T>) return x.re == R(0) && x.im == R(0);
    else if constexpr (Arith<T>::is_complex) return Arith<T>::re(x) == R(0) && Arith<T>::im(x) == R(0);
    else return x == T(0);
}
template <class T> inline bool isfinite(T x) {
    if constexpr (pod_complex<T>) return sycl::isfinite(x.re) && sycl::isfinite(x.im);
    else if constexpr (Arith<T>::is_complex) return sycl::isfinite(Arith<T>::re(x)) && sycl::isfinite(Arith<T>::im(x));
    else return sycl::isfinite(x);
}
template <class T> inline real_t<T> abs1(T a) {
    if constexpr (pod_complex<T>) return sycl::fabs(a.re) + sycl::fabs(a.im);
    else return v2::abs1(a);
}
template <class T> inline T mul(T a, T b) {
    using A = Arith<T>;
    if constexpr (pod_complex<T>) {
        return T{sycl::fma(a.re, b.re, -a.im * b.im), sycl::fma(a.re, b.im, a.im * b.re)};
    } else if constexpr (A::is_complex) {
        return A::make(sycl::fma(A::re(a), A::re(b), -A::im(a) * A::im(b)),
                       sycl::fma(A::re(a), A::im(b), A::im(a) * A::re(b)));
    } else {
        return a * b;
    }
}
template <class T> inline T add(T a, T b) {
    if constexpr (pod_complex<T>) return T{a.re + b.re, a.im + b.im};
    else return Arith<T>::add(a, b);
}
template <class T> inline T sub(T a, T b) {
    if constexpr (pod_complex<T>) return T{a.re - b.re, a.im - b.im};
    else return Arith<T>::sub(a, b);
}
template <class T> inline T recip(T d) {
    using A = Arith<T>;
    using R = real_t<T>;
    if constexpr (pod_complex<T>) {
        if (sycl::fabs(d.re) >= sycl::fabs(d.im)) {
            const R r = d.im / d.re;
            const R den = sycl::fma(d.im, r, d.re);
            return T{R(1) / den, -r / den};
        }
        const R r = d.re / d.im;
        const R den = sycl::fma(d.re, r, d.im);
        return T{r / den, R(-1) / den};
    } else if constexpr (A::is_complex) {
        const R dr = A::re(d), di = A::im(d);
        if (sycl::fabs(dr) >= sycl::fabs(di)) {
            const R r = di / dr;
            const R den = sycl::fma(di, r, dr);
            return A::make(R(1) / den, -r / den);
        }
        const R r = dr / di;
        const R den = sycl::fma(dr, r, di);
        return A::make(r / den, R(-1) / den);
    } else {
        return T(1) / d;
    }
}
template <class T> inline T div(T a, T b) {
    using A = Arith<T>;
    using R = real_t<T>;
    if constexpr (pod_complex<T>) {
        if (sycl::fabs(b.re) >= sycl::fabs(b.im)) {
            const R r = b.im / b.re;
            const R den = sycl::fma(b.im, r, b.re);
            return T{sycl::fma(a.im, r, a.re) / den, sycl::fma(-a.re, r, a.im) / den};
        }
        const R r = b.re / b.im;
        const R den = sycl::fma(b.re, r, b.im);
        return T{sycl::fma(a.re, r, a.im) / den, sycl::fma(a.im, r, -a.re) / den};
    } else if constexpr (A::is_complex) {
        const R ar = A::re(a), ai = A::im(a), br = A::re(b), bi = A::im(b);
        if (sycl::fabs(br) >= sycl::fabs(bi)) {
            const R r = bi / br;
            const R den = sycl::fma(bi, r, br);
            return A::make(sycl::fma(ai, r, ar) / den, sycl::fma(-ar, r, ai) / den);
        }
        const R r = br / bi;
        const R den = sycl::fma(br, r, bi);
        return A::make(sycl::fma(ar, r, ai) / den, sycl::fma(ai, r, -ar) / den);
    } else {
        return a / b;
    }
}
}  // namespace fp

// ---- DynGroup: a run-time-width executor ------------------------------------------------------
// For kernels whose launch picks the width per shape (getrf's leaves: 64..512). S = Sub: one
// sub-group is the executor (several per work-group, each its own problem); S = Group: the work-
// group. No static `size`, so core ops that need one (gemv, ger) do not compile on it; Cyclic
// level-1 ops and reductions work through the for_each_owned overload below. Its barrier is
// sycl::group_barrier at its scope (all memory), whatever Mem is asked for. Ids and ranges are
// read once, at construction, as the hand-written kernels read them at the top: read at each use,
// the sub-group range is rematerialised per op (+4 REG on sm_89).
template <Scope S> struct DynGroup {
    static_assert(S == Scope::Sub || S == Scope::Group, "v2::DynGroup: Scope::Sub or Scope::Group");
    static constexpr bool sub_whole = (S == Scope::Sub);
    static constexpr Scope natural_scope = S;
    sycl::nd_item<1> it;
    int lane_;
    int lanes_;
    int team_;
    int nteams_;
    int width_;
    int lid_;

    BATCHLAS_DLA_INLINE DynGroup(sycl::nd_item<1> i)
        : it(i),
          lane_(static_cast<int>(i.get_sub_group().get_local_linear_id())),
          lanes_(static_cast<int>(i.get_sub_group().get_local_linear_range())),
          team_(static_cast<int>(i.get_sub_group().get_group_linear_id())),
          nteams_(static_cast<int>(i.get_sub_group().get_group_linear_range())),
          width_(S == Scope::Sub ? lanes_ : static_cast<int>(i.get_local_range(0))),
          lid_(S == Scope::Sub ? lane_ : static_cast<int>(i.get_local_linear_id())) {}
    // The width as the host launched it (a kernel argument, equal to the local range): a stride
    // the compiler cannot range-bound, as in kernels written against a captured `wg`.
    BATCHLAS_DLA_INLINE DynGroup(sycl::nd_item<1> i, int width) : DynGroup(i) { width_ = width; }

    BATCHLAS_DLA_INLINE sycl::sub_group sub() const { return it.get_sub_group(); }
    BATCHLAS_DLA_INLINE int lane() const { return lane_; }
    BATCHLAS_DLA_INLINE int team_width() const { return lanes_; }
    BATCHLAS_DLA_INLINE int team() const { return team_; }
    BATCHLAS_DLA_INLINE int nteams() const { return nteams_; }
    BATCHLAS_DLA_INLINE int lid() const { return lid_; }
    BATCHLAS_DLA_INLINE int width() const { return width_; }
    BATCHLAS_DLA_INLINE bool leader() const { return lid_ == 0; }
    BATCHLAS_DLA_INLINE void barrier(Mem = Mem::Local) const {
        if constexpr (S == Scope::Sub) sycl::group_barrier(it.get_sub_group());
        else sycl::group_barrier(it.get_group());
    }
    template <class T, class Op = sycl::plus<T>> BATCHLAS_DLA_INLINE T team_reduce(T v, Op op = {}) const {
        return detail::sycl_reduce(it.get_sub_group(), v, op);
    }
    template <class T, class Op = sycl::plus<T>> BATCHLAS_DLA_INLINE T group_reduce(T v, Op op = {}) const {
        if constexpr (S == Scope::Sub) return team_reduce(v, op);
        else return detail::sycl_reduce(it.get_group(), v, op);
    }
};

template <class Ex> inline constexpr bool is_dyn_group_v = false;
template <Scope S> inline constexpr bool is_dyn_group_v<DynGroup<S>> = true;

// Found by ADL from every core level-1 op, and more specialised than the core template.
template <Scope S, class F> BATCHLAS_DLA_INLINE void for_each_owned(const DynGroup<S>& ex, int n, F&& f) {
    const int w = ex.width();
    for (int i = ex.lid(); i < n; i += w) f(i);
}

namespace detail {
template <class Ex> BATCHLAS_DLA_INLINE int width_of(const Ex& ex) {
    if constexpr (requires { Ex::size; }) return Ex::size;
    else return ex.width();
}
template <class Ex> BATCHLAS_DLA_INLINE int nteams_of(const Ex& ex) {
    if constexpr (requires { Ex::nteams; }) return Ex::nteams;
    else return ex.nteams();
}
// The object a team's shuffles go through.
template <class Ex> BATCHLAS_DLA_INLINE auto team_handle(const Ex& ex) {
    if constexpr (is_dyn_group_v<Ex>) return ex.sub();
    else if constexpr (requires { ex.part; }) return ex.part;
    else if constexpr (requires { ex.sg; }) return ex.sg;
    else return ex.it.get_sub_group();
}
template <class H, class R> BATCHLAS_DLA_INLINE R select_lane(const H& h, R v, std::uint32_t src) {
    if constexpr (sycl::is_group_v<H>) return batchlas::portable::select_from_group(h, v, src);
    else return select_from_group(h, v, src);  // ADL: SubGroupPartition (src/kernels/common)
}
// Read across owners: Published, or one item is the executor.
template <class Ex, class V> constexpr void require_pub() {
    static_assert(is_published_v<V> || is_thread_v<Ex>, "v2: this operand is read across owners; publish it");
}
}  // namespace detail

// ---- views with absolute offsets and 64-bit addressing ---------------------------------------
// A view keeps the base pointer and absolute int row/column offsets; an element is
// p[sext(r) + sext(c) * ld], the legacy addressing (int32 Mat indexing moves ~8% of the CTA
// panel's instructions on sm_89). On a DynGroup, iamax_loc and rscl walk a column's absolute
// rows (r = r0 + lid; r < r0 + n), the legacy loop form.
template <class T, Mem M> BATCHLAS_DLA_INLINE T& abs_at(T* p, int ld, int r, int c) {
    return p[static_cast<std::ptrdiff_t>(r) + static_cast<std::ptrdiff_t>(c) * ld];
}
template <class T, Mem M = Mem::Global> struct AbsCol {
    using value_type = std::remove_cv_t<T>;
    using layout = Cyclic;
    static constexpr Mem mem = M;
    T* p;
    int r0;
    int c;
    int ld;
    int n;
    BATCHLAS_DLA_INLINE T& at(int r) const { return abs_at<T, M>(p, ld, r, c); }
    BATCHLAS_DLA_INLINE T& operator()(int i) const { return at(r0 + i); }
    BATCHLAS_DLA_INLINE int size() const { return n; }
    BATCHLAS_DLA_INLINE AbsCol tail(int k) const { return {p, r0 + k, c, ld, n - k}; }
};
template <class T, Mem M = Mem::Global> struct AbsRow {
    using value_type = std::remove_cv_t<T>;
    using layout = Cyclic;
    static constexpr Mem mem = M;
    T* p;
    int r;
    int c0;
    int ld;
    int n;
    BATCHLAS_DLA_INLINE T& operator()(int j) const { return abs_at<T, M>(p, ld, r, c0 + j); }
    BATCHLAS_DLA_INLINE int size() const { return n; }
    BATCHLAS_DLA_INLINE AbsRow tail(int k) const { return {p, r, c0 + k, ld, n - k}; }
};
template <class T, Mem M = Mem::Global> struct AbsMat {
    using value_type = std::remove_cv_t<T>;
    static constexpr Mem mem = M;
    T* p;
    int m;
    int n;
    int ld;
    int i0 = 0;
    int j0 = 0;
    BATCHLAS_DLA_INLINE T& operator()(int i, int j) const { return abs_at<T, M>(p, ld, i0 + i, j0 + j); }
    BATCHLAS_DLA_INLINE int rows() const { return m; }
    BATCHLAS_DLA_INLINE int cols() const { return n; }
    BATCHLAS_DLA_INLINE AbsCol<T, M> col(int j) const { return {p, i0, j0 + j, ld, m}; }
    BATCHLAS_DLA_INLINE AbsRow<T, M> row(int i) const { return {p, i0 + i, j0, ld, n}; }
    BATCHLAS_DLA_INLINE AbsMat sub(int i, int j, int r, int c) const { return {p, r, c, ld, i0 + i, j0 + j}; }
};
template <Mem M = Mem::Global, class T> BATCHLAS_DLA_INLINE AbsMat<T, M> abs_mat(T* p, int m, int n, int ld) {
    return {p, m, n, ld};
}
template <class X> inline constexpr bool is_abs_col_v = false;
template <class T, Mem M> inline constexpr bool is_abs_col_v<AbsCol<T, M>> = true;

// ---- 2-D owned traversal, ?LACPY --------------------------------------------------------------
namespace alg {
struct Flat {};     // flat e over m*n, row fastest: (e mod m, e / m); fills the executor on any shape
struct Stepped {};  // same order, one division up front, then row += width with carry into col
}  // namespace alg

// f(r, c) for every element of an m x n index space this item owns. Idx is the flat index type
// (alg::Flat only).
template <class Alg = alg::Flat, class Idx = int, class Ex, class F>
BATCHLAS_DLA_INLINE void for_each_owned_2d(const Ex& ex, int m, int n, F&& f) {
    if constexpr (std::is_same_v<Alg, alg::Flat>) {
        const Idx total = static_cast<Idx>(m) * static_cast<Idx>(n);
        const Idx w = static_cast<Idx>(detail::width_of(ex));
        for (Idx e = static_cast<Idx>(ex.lid()); e < total; e += w) {
            const Idx q = e / static_cast<Idx>(m);  // the remainder by subtraction keeps nsw
            f(static_cast<int>(e - q * static_cast<Idx>(m)), static_cast<int>(q));
        }
    } else {
        static_assert(std::is_same_v<Alg, alg::Stepped>, "v2::for_each_owned_2d: alg::Flat or alg::Stepped");
        const int w = detail::width_of(ex);
        int c = ex.lid() / m;
        int r = ex.lid() - c * m;
        while (c < n) {
            f(r, c);
            r += w;
            while (r >= m) {
                r -= m;
                ++c;
            }
        }
    }
}

// B := A over B's m x n extent (LAPACK ?LACPY, uplo = 'A'). alg::Flat is written out, not a
// lambda through for_each_owned_2d: the lambda's later inlining moves the 64-bit staging address
// math of getrf's resident panel (-5% instructions, sm_89).
template <class Alg = alg::Flat, class Idx = int, class Ex, class MA, class MB>
BATCHLAS_DLA_INLINE void lacpy(const Ex& ex, const MA& a, const MB& b) {
    if constexpr (std::is_same_v<Alg, alg::Flat>) {
        const int m = b.rows();
        const Idx total = static_cast<Idx>(m) * static_cast<Idx>(b.cols());
        for (Idx e = static_cast<Idx>(ex.lid()); e < total; e += static_cast<Idx>(detail::width_of(ex))) {
            const int r = static_cast<int>(e % static_cast<Idx>(m));
            const int c = static_cast<int>(e / static_cast<Idx>(m));
            b(r, c) = a(r, c);
        }
    } else {
        for_each_owned_2d<Alg, Idx>(ex, b.rows(), b.cols(), [&](int r, int c) { b(r, c) = a(r, c); });
    }
}

// ---- level 1/2 pieces of ?GETF2 -----------------------------------------------------------------
// x := x / d with ?GETF2's arms: one reciprocal and a multiply when 1/d is finite and nonzero,
// else a division per element (the sfmin arm). d != 0 is the caller's test.
template <class Ex, class T, class X> BATCHLAS_DLA_INLINE void rscl(const Ex& ex, T d, const X& x) {
    const T r = fp::recip(d);
    const bool use_mul = fp::isfinite(r) && !fp::is_zero(r);
    if constexpr (is_abs_col_v<X> && is_dyn_group_v<Ex>) {
        const int w = ex.width(), hi = x.r0 + x.n;
        if (use_mul) {
            for (int i = x.r0 + ex.lid(); i < hi; i += w) x.at(i) = fp::mul(static_cast<T>(x.at(i)), r);
        } else {
            for (int i = x.r0 + ex.lid(); i < hi; i += w) x.at(i) = fp::div(static_cast<T>(x.at(i)), d);
        }
    } else {
        if (use_mul) for_each_owned(ex, x.size(), [&](int i) { x(i) = fp::mul(static_cast<T>(x(i)), r); });
        else for_each_owned(ex, x.size(), [&](int i) { x(i) = fp::div(static_cast<T>(x(i)), d); });
    }
}

// A := A - x y^T (NegOne) or A + x y^T (One), flattened over A's m*n elements (alg::Flat) so a
// short extent still fills the executor. x and y are read across owners.
template <class AS, class Ex, class X, class Y, class MA>
BATCHLAS_DLA_INLINE void ger_flat(const Ex& ex, AS, const X& xp, const Y& yp, const MA& a) {
    static_assert(std::is_same_v<AS, NegOne> || std::is_same_v<AS, One>, "v2::ger_flat: alpha is NegOne or One");
    detail::require_pub<Ex, X>();
    detail::require_pub<Ex, Y>();
    using T = typename MA::value_type;
    const auto& x = detail::unpub(xp);
    const auto& y = detail::unpub(yp);
    const int m = a.rows(), n = a.cols();
    if (m > 0 && n > 0) {
        // Written out (e mod m, e / m), as ?GETF2's hand loop was; through for_each_owned_2d's
        // lambda the panel kernels drift by a VOTE and +4 REG (sm_89).
        const int total = m * n;
        const int w = detail::width_of(ex);
        for (int e = ex.lid(); e < total; e += w) {
            const int r = e % m;
            const int c = e / m;
            if constexpr (std::is_same_v<AS, NegOne>) {
                a(r, c) = fp::sub(static_cast<T>(a(r, c)), fp::mul(static_cast<T>(x(r)), static_cast<T>(y(c))));
            } else {
                a(r, c) = fp::add(static_cast<T>(a(r, c)), fp::mul(static_cast<T>(x(r)), static_cast<T>(y(c))));
            }
        }
    }
}

// ---- pivot search ---------------------------------------------------------------------------
template <class R> struct MaxLoc {
    R v;
    int i;
};
// Take (ov, oi) when larger, or equal with a lower index. A NaN never wins, so a NaN-free seed
// stays NaN-free and every lane agrees on the winner.
template <class R> BATCHLAS_DLA_INLINE void maxloc_merge(MaxLoc<R>& a, R ov, int oi) {
    if (ov > a.v || (ov == a.v && oi < a.i)) {
        a.v = ov;
        a.i = oi;
    }
}

struct XorDown {};  // masks width/2 .. 1
struct XorUp {};    // masks 1 .. width/2

// xor butterfly over the team (sub-group or partition): every lane ends with the team's best.
template <class Order = XorDown, class Ex, class R> BATCHLAS_DLA_INLINE void maxloc_team(const Ex& ex, MaxLoc<R>& a) {
    if constexpr (is_thread_v<Ex>) {
        return;
    } else if constexpr (is_dyn_group_v<Ex>) {
        static_assert(std::is_same_v<Order, XorDown>, "v2::maxloc_team: a run-time width is walked XorDown");
        const auto sg = ex.sub();
        for (std::uint32_t off = static_cast<std::uint32_t>(ex.team_width()) / 2; off > 0; off >>= 1) {
            const R ov = batchlas::portable::permute_group_by_xor(sg, a.v, off);
            const int oi = batchlas::portable::permute_group_by_xor(sg, a.i, off);
            maxloc_merge(a, ov, oi);
        }
    } else {
        constexpr int P = Ex::team_size;
        const auto h = detail::team_handle(ex);
        if constexpr (std::is_same_v<Order, XorUp>) {
            BATCHLAS_DLA_UNROLL_FULL
            for (std::uint32_t m = 1u; m < static_cast<std::uint32_t>(P); m <<= 1) {
                const R ov = detail::xor_shuffle(h, a.v, m);
                const int oi = detail::xor_shuffle(h, a.i, m);
                maxloc_merge(a, ov, oi);
            }
        } else {
            BATCHLAS_DLA_UNROLL_FULL
            for (std::uint32_t m = static_cast<std::uint32_t>(P) / 2; m > 0; m >>= 1) {
                const R ov = detail::xor_shuffle(h, a.v, m);
                const int oi = detail::xor_shuffle(h, a.i, m);
                maxloc_merge(a, ov, oi);
            }
        }
    }
}

// Team results joined across the executor's teams: lane 0 of team t writes slots[t], ONE
// barrier, every item scans the slots in team order from {-1, none}. No trailing barrier: the
// caller barriers before the slots are written again. A Scope::Sub executor returns `a`.
template <class Ex, class R, class VS, class IS>
BATCHLAS_DLA_INLINE MaxLoc<R> maxloc_group(const Ex& ex, MaxLoc<R> a, const VS& vs, const IS& is, int none) {
    if constexpr (Ex::natural_scope == Scope::Sub) {
        return a;
    } else {
        if (ex.lane() == 0) {
            vs[static_cast<std::size_t>(ex.team())] = a.v;
            is[static_cast<std::size_t>(ex.team())] = a.i;
        }
        ex.barrier(Mem::Local);
        MaxLoc<R> f{R(-1), none};
        const int nt = detail::nteams_of(ex);
        for (int t = 0; t < nt; ++t) maxloc_merge(f, static_cast<R>(vs[static_cast<std::size_t>(t)]), static_cast<int>(is[static_cast<std::size_t>(t)]));
        return f;
    }
}

// ?IAMAX with cabs1 (|re| + |im|): 0-based index of the first maximum; NaN elements are skipped;
// x.size() when none qualifies. vs/is: nteams slots of local memory (unused at Scope::Sub).
template <class Ex, class X, class VS, class IS>
BATCHLAS_DLA_INLINE int iamax_loc(const Ex& ex, const X& x, const VS& vs, const IS& is) {
    using R = real_t<elem_t<X>>;
    if constexpr (is_abs_col_v<X> && is_dyn_group_v<Ex>) {
        // Absolute rows, absolute candidates, sentinel r0 + n; relative on return.
        const int w = ex.width(), hi = x.r0 + x.n;
        MaxLoc<R> a{R(-1), hi};
        for (int i = x.r0 + ex.lid(); i < hi; i += w) {
            const R v = fp::abs1(x.at(i));
            if (v > a.v) {
                a.v = v;
                a.i = i;
            }
        }
        maxloc_team(ex, a);
        return maxloc_group(ex, a, vs, is, hi).i - x.r0;
    } else {
        const int n = x.size();
        MaxLoc<R> a{R(-1), n};
        for_each_owned(ex, n, [&](int i) {
            const R v = fp::abs1(x(i));
            if (v > a.v) {
                a.v = v;
                a.i = i;
            }
        });
        maxloc_team(ex, a);
        return maxloc_group(ex, a, vs, is, n).i;
    }
}

// LAPACK ipiv sink: the row chosen at step k (0-based in the panel) is stored 1-based plus
// `base`, so a panel at global row `base` writes global pivots with no fix-up pass.
struct Ipiv {
    int* p;
    int base;
    BATCHLAS_DLA_INLINE void set(int k, int row) const { p[k] = base + row + 1; }
};

// ---- ?GETF2, explicit row interchanges ------------------------------------------------------
// LU with partial pivoting of the m x n operand A, in place, for kmax steps (kmax <= min(m, n)):
// per column, iamax, ipiv, swap of the WHOLE rows (L's finished columns travel too), rscl,
// then the trailing rank-1 update, with three barriers (four at Scope::Group: iamax's). info
// enters as this item's running value (leader: first failure so far, global 1-based; others 0)
// and leaves updated: the first EXACT zero pivot, at column ipiv.base + k + 1 (the panel sits on
// the diagonal), leaves its column unscaled.
template <class Ex, class MA, class VS, class IS>
BATCHLAS_DLA_INLINE std::int32_t getf2(const Ex& ex, const MA& a, int kmax, Ipiv ipiv, std::int32_t info, const VS& vs,
                                       const IS& is) {
    using T = typename MA::value_type;
    constexpr Mem fence = MA::mem == Mem::Local ? Mem::Local : Mem::Global;
    const int m = a.rows(), n = a.cols();
    for (int k = 0; k < kmax; ++k) {
        int p = k + iamax_loc(ex, a.col(k).tail(k), vs, is);
        if (p >= m) p = k;  // every entry NaN
        if (ex.leader()) ipiv.set(k, p);
        if (p != k) v2::swap(ex, a.row(k), a.row(p));
        ex.barrier(fence);
        const T d = a(k, k);
        if (fp::is_zero(d)) {
            if (ex.leader() && info == 0) info = static_cast<std::int32_t>(ipiv.base + k + 1);
        } else {
            rscl(ex, d, a.col(k).tail(k + 1));
        }
        ex.barrier(fence);
        ger_flat(ex, NegOne{}, assume_published(a.col(k).tail(k + 1)), assume_published(a.row(k).tail(k + 1)),
                 a.sub(k + 1, k + 1, m - k - 1, n - k - 1));
        ex.barrier(fence);
    }
    return info;
}

// ---- ?GETF2 on a RowFrag, pivoting by lazy relabel ----------------------------------------------
// Each item holds one matrix row in rA and its current logical row index in `rowid`; an
// interchange swaps two labels and moves no data. The winner of column j's search is, after the
// relabel, the item with rowid == j, so `act = rowid > j` keeps it from writing rA while others
// read its row. Two policies plug in:
//   Piv: argmax(j, mag, rowid) -> PivotWinner (the search, NaN mapped by the policy),
//        record(j, w) (ipiv), on_zero(zero, j) (info)
//   Bc:  publish(j, rA, rowid, w) (make the pivot row readable), pivot(j, rA, w) -> T,
//        for_right(j, ncols, rA, w, f) calling f(k, u_k) for the pivot row's k > j, k < ncols
// The j loop has a compile-time trip count and skips with `continue` (never `break`), so rA is
// indexed by constants only; kmax must be uniform over the executor (collectives are skipped).
struct PivotWinner {
    int row;  // winning logical row
    int src;  // the lane that holds it (policies that broadcast by lane)
};

template <int NC, class T, class Piv, class Bc>
BATCHLAS_DLA_INLINE void getf2_relabel(Frag<T, NC>& rA, int& rowid, int kmax, int ncols, Piv& piv, Bc& bc) {
    BATCHLAS_DLA_UNROLL_FULL
    for (int j = 0; j < NC; ++j) {
        if (j >= kmax) continue;
        const PivotWinner w = piv.argmax(j, fp::abs1(rA[j]), rowid);  // ?IAMAX
        piv.record(j, w);
        if (rowid == w.row) rowid = j;  // ?SWAP as two labels
        else if (rowid == j) rowid = w.row;
        bc.publish(j, rA, rowid, w);
        const T d = bc.pivot(j, rA, w);
        const bool zero = fp::is_zero(d);  // EXACT zero, no epsilon
        piv.on_zero(zero, j);
        const T rc = fp::recip(d);
        const bool use_mul = !zero && fp::isfinite(rc) && !fp::is_zero(rc);
        const bool act = (rowid > j);
        if (act) {  // ?RSCL of this row's multiplier; if/else, never ?: on a complex scalar
            if (use_mul) rA[j] = fp::mul(rA[j], rc);
            else if (!zero) rA[j] = fp::div(rA[j], d);
        }
        bc.for_right(j, ncols, rA, w, [&](int k, T u) {  // ?GER, this row; a zero pivot's multipliers are 0
            if (act) rA[k] = fp::sub(rA[k], fp::mul(rA[j], u));
        });
    }
}

// Piv for a RowFrag over a Scope::Group executor: candidates are rows >= j (a pad item carries
// rowid -1), a NaN magnitude maps to -1, a non-candidate carries `m`; maxloc_team then
// maxloc_group (one barrier). The leader writes ipiv and keeps info (global, 1-based, sticky).
template <class Ex, class VS, class IS> struct PivotRowsGroup {
    const Ex& ex;
    VS vs;
    IS is;
    int m;
    Ipiv ipiv;
    std::int32_t& info;
    template <class R> BATCHLAS_DLA_INLINE PivotWinner argmax(int j, R mag, int rowid) const {
        const bool cand = (rowid >= j);
        MaxLoc<R> a{(cand && (mag == mag)) ? mag : R(-1), cand ? rowid : m};
        maxloc_team(ex, a);
        a = maxloc_group(ex, a, vs, is, m);
        return {a.i >= m ? j : a.i, 0};  // >= m: unreachable, row j exists
    }
    BATCHLAS_DLA_INLINE void record(int j, PivotWinner w) const {
        if (ex.leader()) ipiv.set(j, w.row);
    }
    BATCHLAS_DLA_INLINE void on_zero(bool zero, int j) const {
        if (zero && ex.leader() && info == 0) info = static_cast<std::int32_t>(ipiv.base + j + 1);
    }
};
template <class Ex, class VS, class IS>
BATCHLAS_DLA_INLINE PivotRowsGroup<Ex, VS, IS> pivot_rows_group(const Ex& ex, VS vs, IS is, int m, Ipiv ipiv,
                                                                std::int32_t& info) {
    return {ex, vs, is, m, ipiv, info};
}

// Bc through a local-memory row sx (>= NC elements): the item whose rowid is j writes the row,
// one barrier; everyone reads it.
template <class Ex, class SX> struct BcastRowLocal {
    const Ex& ex;
    SX sx;
    template <int NC, class T> BATCHLAS_DLA_INLINE void publish(int j, const Frag<T, NC>& rA, int rowid, PivotWinner) const {
        if (rowid == j) {
            BATCHLAS_DLA_UNROLL_FULL
            for (int k = j; k < NC; ++k) sx[static_cast<std::size_t>(k)] = rA[k];
        }
        ex.barrier(Mem::Local);
    }
    template <int NC, class T> BATCHLAS_DLA_INLINE T pivot(int j, const Frag<T, NC>&, PivotWinner) const {
        return sx[static_cast<std::size_t>(j)];
    }
    template <int NC, class T, class F>
    BATCHLAS_DLA_INLINE void for_right(int j, int ncols, const Frag<T, NC>&, PivotWinner, F&& f) const {
        BATCHLAS_DLA_UNROLL_FULL
        for (int k = j + 1; k < NC; ++k) {
            if (k >= ncols) continue;
            f(k, static_cast<T>(sx[static_cast<std::size_t>(k)]));
        }
    }
};
template <class Ex, class SX> BATCHLAS_DLA_INLINE BcastRowLocal<Ex, SX> bcast_row_local(const Ex& ex, SX sx) {
    return {ex, sx};
}

// v from lane src of the team, to every lane (complex: two real selects).
template <class Ex, class T> BATCHLAS_DLA_INLINE T bcast(const Ex& ex, T v, std::uint32_t src) {
    using A = Arith<T>;
    const auto h = detail::team_handle(ex);
    if constexpr (A::is_complex) {
        const auto re = detail::select_lane(h, A::re(v), src);
        const auto im = detail::select_lane(h, A::im(v), src);
        return A::make(re, im);
    } else {
        return detail::select_lane(h, v, src);
    }
}

// Bc by shuffles from the winner's lane (w.src): no memory, no barrier, one select per element.
template <class Ex> struct BcastShuffle {
    const Ex& ex;
    template <int NC, class T> BATCHLAS_DLA_INLINE void publish(int, const Frag<T, NC>&, int, PivotWinner) const {}
    template <int NC, class T> BATCHLAS_DLA_INLINE T pivot(int j, const Frag<T, NC>& rA, PivotWinner w) const {
        return bcast(ex, rA[j], static_cast<std::uint32_t>(w.src));
    }
    template <int NC, class T, class F>
    BATCHLAS_DLA_INLINE void for_right(int j, int ncols, const Frag<T, NC>& rA, PivotWinner w, F&& f) const {
        BATCHLAS_DLA_UNROLL_FULL
        for (int k = j + 1; k < NC; ++k) {
            if (k >= ncols) continue;
            f(k, bcast(ex, rA[k], static_cast<std::uint32_t>(w.src)));
        }
    }
};
template <class Ex> BATCHLAS_DLA_INLINE BcastShuffle<Ex> bcast_shuffle(const Ex& ex) { return {ex}; }

// ---- ?LASWP ------------------------------------------------------------------------------------
enum class LaswpDir { Forward, Backward };

// Interchange x(k) and x(to(k)) for k = k0 .. k1-1 (Forward: P x, LAPACK incx = +1) or
// k1-1 .. k0 (Backward: P^T x, incx = -1). `to` returns a 0-based target; x's extent is not read
// (targets lie at or below k). One item owns x.
template <LaswpDir D, class X, class To> BATCHLAS_DLA_INLINE void laswp(Thread, const X& x, To&& to, int k0, int k1) {
    auto step = [&](int k) {
        const int p = to(k);
        if (p != k) {
            const auto t = x(k);
            x(k) = x(p);
            x(p) = t;
        }
    };
    if constexpr (D == LaswpDir::Forward) {
        for (int k = k0; k < k1; ++k) step(k);
    } else {
        for (int k = k1 - 1; k >= k0; --k) step(k);
    }
}
// Every column of A, distributed over ex (Cyclic over columns).
template <LaswpDir D, class Ex, class MA, class To>
BATCHLAS_DLA_INLINE void laswp(const Ex& ex, const MA& a, To&& to, int k0, int k1) {
    for_each_owned(ex, a.cols(), [&](int c) { laswp<D>(Thread{}, a.col(c), to, k0, k1); });
}

}  // namespace batchlas::device::v2
