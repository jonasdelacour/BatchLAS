// SPDX-License-Identifier: MIT
#pragma once
// Device LA v2 extension for the sytrd CTA / blocked family: ops for tiles that fit the
// executor (one row per item), the RowFrag larfg, rank-2 Hermitian updates, the her2k entry and
// the unblocked sytd2 on a resident tile.
//
//   Fit<Ex>        operands fit the executor: item lid owns index lid - origin, so every owned
//                  loop is one guarded iteration and ownership survives a shrinking trailing block
//   team_barrier   the caller-owned barrier of any executor;  broadcast  one item's value to all
//   larfg_scalars, larfg_rowfrag, abs2, her2_term, ImpliedUnit
//   gemv / dotc on Fit, her2_full, her2_mirror<UL>, lacpy, lacpy_herm<UL>, dot2c, sytd2<UL>
//
// Expression shapes follow the kernels these ops replaced: fp-contract=on fuses a*b+c within one
// expression only, so a float/double port keeps its bits only if each expression keeps its shape.
// Complex inner loops go through Arith (no Annex-G operator*): complex ports are faster, not
// bit-identical. Scalar (once per step) complex work keeps T's operators.

#include <cstdint>
#include <type_traits>
#include <utility>

#include <batchlas/blas/device/v2/dla.hh>
#include <batchlas/util/group-collectives.hh>

namespace batchlas::device::v2 {

// ---- executors ---------------------------------------------------------------------------
template <class Ex> struct Fit;

template <class Ex> inline constexpr bool is_fit_v = false;
template <class Ex> inline constexpr bool is_fit_v<Fit<Ex>> = true;
template <class Ex> inline constexpr bool is_partition_v = false;
template <int P, bool M, class I> inline constexpr bool is_partition_v<Partition<P, M, I>> = true;
template <class Ex> inline constexpr bool is_sub_group_v = false;
template <int SG> inline constexpr bool is_sub_group_v<SubGroup<SG>> = true;
template <class Ex> inline constexpr bool is_teams_v = false;
template <int WG, int SG> inline constexpr bool is_teams_v<Teams<WG, SG>> = true;

// Ex whose operand extents fit it (n <= size + origin): item lid owns element / row
// lid - origin. Ops with a Fit overload (for_each_owned and every core op built on it, gemv,
// dotc, her2_full, her2_mirror, lacpy_herm) honour the origin; core ops with their own loops
// (symv, ger) keep the Cyclic layout. lid() is the underlying item id.
template <class Ex> struct Fit {
    static constexpr int size = Ex::size, team_size = Ex::team_size, nteams = Ex::nteams;
    static constexpr bool sub_whole = Ex::sub_whole;
    static constexpr Scope natural_scope = Ex::natural_scope;
    Ex ex;
    int origin = 0;

    BATCHLAS_DLA_INLINE int lid() const { return ex.lid(); }
    BATCHLAS_DLA_INLINE int owned() const { return ex.lid() - origin; }
    BATCHLAS_DLA_INLINE int lane() const { return ex.lane(); }
    BATCHLAS_DLA_INLINE int team() const { return ex.team(); }
    BATCHLAS_DLA_INLINE bool leader() const { return ex.leader(); }
    // The same items, owning index lid - o (a trailing block that starts at row o).
    BATCHLAS_DLA_INLINE Fit at(int o) const { return Fit{ex, o}; }
    template <class T, class Op = sycl::plus<T>> BATCHLAS_DLA_INLINE T team_reduce(T v, Op op = {}) const {
        return ex.team_reduce(v, op);
    }
    template <class T, class Op = sycl::plus<T>> BATCHLAS_DLA_INLINE T group_reduce(T v, Op op = {}) const {
        return ex.group_reduce(v, op);
    }
    BATCHLAS_DLA_INLINE void barrier(Mem m = Mem::Local) const
        requires is_work_group_v<Ex>
    {
        ex.barrier(m);
    }
};
template <class Ex> inline constexpr bool is_work_group_v<Fit<Ex>> = is_work_group_v<Ex>;

template <class Ex> BATCHLAS_DLA_INLINE Fit<Ex> fit(const Ex& ex) { return Fit<Ex>{ex, 0}; }

// The v2 executor over an existing partition object (SubGroupPartition, src/kernels/common/
// sg_partition), keeping that object's base lane rather than recomputing it.
template <class Impl> BATCHLAS_DLA_INLINE auto adopt_partition(const Impl& p) {
    Partition<static_cast<int>(Impl::kP), Impl::kMasked, Impl> ex(p.sg);
    ex.part = p;
    return ex;
}

// On a Fit executor an index has one owner and one guarded call (no loop, no stride).
template <class Ex, class F> BATCHLAS_DLA_INLINE void for_each_owned(const Fit<Ex>& ex, int n, F&& f) {
    const int i = ex.owned();
    if (i >= 0 && i < n) f(i);
}

// Orders the executor's local-memory accesses: a work-group barrier on WorkGroup, a sub-group or
// partition barrier on SubGroup / Teams / Partition, nothing on Thread. The caller owns it (D4).
template <class Ex> BATCHLAS_DLA_INLINE void team_barrier(const Ex& ex) {
    if constexpr (is_fit_v<Ex>) {
        team_barrier(ex.ex);
    } else if constexpr (is_work_group_v<Ex>) {
        ex.barrier(Mem::Local);
    } else if constexpr (is_partition_v<Ex>) {
        if constexpr (sycl::is_group_v<std::remove_cvref_t<decltype(ex.part)>>) sycl::group_barrier(ex.part);
        else group_barrier(ex.part);  // ADL: SubGroupPartition
    } else if constexpr (is_sub_group_v<Ex>) {
        sycl::group_barrier(ex.sg);
    } else if constexpr (is_teams_v<Ex>) {
        sycl::group_barrier(ex.it.get_sub_group());
    } else {
        static_assert(is_thread_v<Ex>, "v2::team_barrier: unknown executor");
    }
}

// The value item `src` (an id of the underlying executor) holds, returned on every item.
template <class Ex, class T> BATCHLAS_DLA_INLINE T broadcast(const Ex& ex, T v, int src) {
    if constexpr (is_fit_v<Ex>) {
        return broadcast(ex.ex, v, src);
    } else if constexpr (is_work_group_v<Ex>) {
        return batchlas::portable::group_broadcast(ex.it.get_group(), v, sycl::id<1>(src));
    } else if constexpr (is_partition_v<Ex>) {
        if constexpr (sycl::is_group_v<std::remove_cvref_t<decltype(ex.part)>>) {
            const int base = static_cast<int>(ex.part.get_local_linear_id()) & ~(Ex::size - 1);
            return batchlas::portable::select_from_group(ex.part, v, static_cast<std::uint32_t>(base + src));
        } else {
            return select_from_group(ex.part, v, static_cast<std::uint32_t>(src));  // ADL: SubGroupPartition
        }
    } else if constexpr (is_sub_group_v<Ex>) {
        return batchlas::portable::select_from_group(ex.sg, v, static_cast<std::uint32_t>(src));
    } else if constexpr (is_teams_v<Ex>) {
        return batchlas::portable::select_from_group(ex.it.get_sub_group(), v, static_cast<std::uint32_t>(src));
    } else {
        static_assert(is_thread_v<Ex>, "v2::broadcast: unknown executor");
        (void)src;
        return v;
    }
}

// ---- operands ----------------------------------------------------------------------------
// Views at a base B (a pointer, or a sycl::local_accessor) plus an int32 element offset, for
// tiles carved per team out of one local accessor. With the accessor as B every access is the
// hand kernels' A_local[off + i + j * ld]: a uniform base sm_89 folds into [R.X4+UR] addressing,
// and the same IR size, so LLVM makes the same unrolling decisions. A Vec/Mat at &A_local[off]
// costs an IMAD per access and unrolls differently.
template <class B> using base_elem_t = std::remove_cvref_t<decltype(std::declval<const B&>()[0])>;

template <class B, class N = Dyn, class Inc = Unit, Mem M = Mem::Local> struct OffVec {
    using value_type = base_elem_t<B>;
    using layout = Cyclic;
    static constexpr Mem mem = M;
    B p;
    int off;
    N n;
    Inc inc{};

    BATCHLAS_DLA_INLINE auto& operator()(int i) const { return p[off + i * int(inc)]; }
    BATCHLAS_DLA_INLINE int size() const { return int(n); }
    BATCHLAS_DLA_INLINE OffVec<B, Dyn, Inc, M> tail(int k) const { return {p, off + k * int(inc), Dyn{int(n) - k}, inc}; }
    BATCHLAS_DLA_INLINE OffVec<B, Dyn, Inc, M> head(int k) const { return {p, off, Dyn{k}, inc}; }
};

template <class B, class R = Dyn, class C = Dyn, class Ld = Dyn, Mem M = Mem::Local> struct OffMat {
    using value_type = base_elem_t<B>;
    using layout = Cyclic;
    static constexpr Mem mem = M;
    B p;
    int off;
    R m;
    C n;
    Ld ld;

    BATCHLAS_DLA_INLINE auto& operator()(int i, int j) const { return p[off + i + j * int(ld)]; }
    BATCHLAS_DLA_INLINE int rows() const { return int(m); }
    BATCHLAS_DLA_INLINE int cols() const { return int(n); }
    BATCHLAS_DLA_INLINE OffVec<B, R, Unit, M> col(int j) const { return {p, off + j * int(ld), m}; }
    BATCHLAS_DLA_INLINE OffVec<B, C, Ld, M> row(int i) const { return {p, off + i, n, ld}; }
    BATCHLAS_DLA_INLINE OffMat<B, Dyn, Dyn, Ld, M> sub(int i, int j, int r, int c) const {
        return {p, off + i + j * int(ld), Dyn{r}, Dyn{c}, ld};
    }
};

// m x n at base p (a pointer or local accessor), leading dimension ld: A(i, j) is p[i + j * ld].
template <Mem M = Mem::Global, class B> BATCHLAS_DLA_INLINE OffMat<B, Dyn, Dyn, Dyn, M> off_mat(B p, int m, int n, int ld) {
    return {p, 0, Dyn{m}, Dyn{n}, Dyn{ld}};
}

// ---- scalars and element kernels -----------------------------------------------------------
// |a|^2 as one expression (re*re + im*im).
template <class T> BATCHLAS_DLA_INLINE real_t<T> abs2(T a) {
    using A = Arith<T>;
    if constexpr (A::is_complex) {
        const real_t<T> re = A::re(a);
        const real_t<T> im = A::im(a);
        return re * re + im * im;
    } else {
        return a * a;
    }
}

// x1 conj(y1) + x2 conj(y2): one term of a rank-2 Hermitian update (her2, her2k).
template <class T> BATCHLAS_DLA_INLINE T her2_term(T x1, T y1, T x2, T y2) {
    using A = Arith<T>;
    if constexpr (A::is_complex) return A::add(A::mul(x1, A::conj(y1)), A::mul(x2, A::conj(y2)));
    else return x1 * y1 + x2 * y2;  // one expression: fma(x1, y1, x2 * y2)
}

template <class T> struct LarfgScalars {
    T tau;
    T beta;
    T scale;  // x := scale x gives v (implicit unit)
};

// DLARFG's scalars for a reflector of length >= 2 from alpha and ||x||_2: beta = -sign(alpha)
// ||(alpha, x)||, tau = (beta - alpha) / beta, scale = 1 / (alpha - beta). x == 0 with a real
// alpha gives tau = 0, beta = alpha, scale = 0. Bit-identical to src/math-helpers.hh detail::larfg.
template <class T> BATCHLAS_DLA_INLINE LarfgScalars<T> larfg_scalars(const T& alpha, real_t<T> xnorm) {
    using A = Arith<T>;
    using R = real_t<T>;
    LarfgScalars<T> s{T(0), alpha, T(0)};
    if constexpr (A::is_complex) {
        if (xnorm == R(0) && A::im(alpha) == R(0)) return s;
        const R alpha_abs = sycl::hypot(A::re(alpha), A::im(alpha));
        const R beta_abs = sycl::hypot(alpha_abs, xnorm);
        const T alpha_sign = (alpha_abs == R(0)) ? T(1) : (alpha / alpha_abs);
        s.beta = -alpha_sign * T(beta_abs);
    } else {
        if (xnorm == R(0)) return s;
        const R sign = sycl::signbit(alpha) ? R(-1) : R(1);
        s.beta = T(-sign * sycl::hypot(alpha, xnorm));
    }
    s.tau = (s.beta - alpha) / s.beta;
    s.scale = T(1) / (alpha - s.beta);
    return s;
}

// DLARFG on a vector in RowFrag layout (element i in a register of item i, n <= Ex::size): item
// alpha_item holds alpha; items with x_active hold x (the others pass x = 0, x_active = false).
// On return alpha_item holds beta and the x items hold v (implicit unit); tau is returned on every
// item. ||x|| and alpha are replicated, so every item forms the scalars (no serial leader).
// n <= 1 gives tau = 0.
template <class Ex, class T>
BATCHLAS_DLA_INLINE T larfg_rowfrag(const Ex& ex, int n, int alpha_item, T& alpha, T& x, bool x_active) {
    using A = Arith<T>;
    using R = real_t<T>;
    const R xsq = x_active ? abs2(x) : R(0);
    const R xnorm = sycl::sqrt(detail::scope_reduce<Scope::Auto>(ex, xsq));
    const T alpha_b = broadcast(ex, alpha, alpha_item);
    T beta = alpha_b, tau = A::zero(), scale = A::zero();
    if (n > 1) {
        const auto s = larfg_scalars(alpha_b, xnorm);
        beta = s.beta;
        tau = s.tau;
        scale = s.scale;
    }
    if (ex.lid() == alpha_item) alpha = beta;
    else if (x_active && tau != A::zero()) x = A::mul(x, scale);
    return tau;
}

// v with element k read as 1 (LAPACK's temporary unit of a stored reflector); v(k) is not read.
template <class V> struct ImpliedUnit {
    V v;
    int k;
    using value_type = typename V::value_type;
    BATCHLAS_DLA_INLINE value_type operator()(int i) const {
        return i == k ? Arith<value_type>::one() : static_cast<value_type>(v(i));
    }
    BATCHLAS_DLA_INLINE int size() const { return v.size(); }
    BATCHLAS_DLA_INLINE ImpliedUnit<decltype(std::declval<V>().tail(0))> tail(int t) const { return {v.tail(t), k - t}; }
};
template <class V> BATCHLAS_DLA_INLINE ImpliedUnit<V> implied_unit(V v, int k) { return {v, k}; }

// ---- level 1 / 2 on Fit ----------------------------------------------------------------------
// x^H y, one product per item (no accumulator), then the Scope S reduction.
template <Scope S = Scope::Auto, class Ex, class X, class Y> BATCHLAS_DLA_INLINE auto dotc(const Fit<Ex>& ex, const X& x, const Y& y) {
    using T = elem_t<X>;
    using A = Arith<T>;
    const int i = ex.owned();
    const T p = (i >= 0 && i < x.size()) ? A::mul(A::conj(x(i)), y(i)) : A::zero();
    return detail::scope_reduce<S>(ex, p);
}

// y := alpha A x + beta y with item r computing output row r (alg::RowOwner without the loop).
template <Transpose TR, class Alg = void, class Ex, class AS, class MA, class X, class BS, class Y>
BATCHLAS_DLA_INLINE void gemv(const Fit<Ex>& ex, AS alpha_, const MA& a, const X& xp, BS beta_, const Y& y) {
    static_assert(TR == Transpose::NoTrans, "v2::gemv on Fit: NoTrans only (one output row per item)");
    static_assert(std::is_void_v<Alg>, "v2::gemv on Fit: the algorithm is fixed");
    using T = typename MA::value_type;
    using Ar = Arith<T>;
    detail::require_published<Ex, X>();
    const auto& x = detail::unpub(xp);
    const auto alpha = as_tag(alpha_);
    const auto beta = as_tag(beta_);
    const int m = a.rows(), n = a.cols(), r = ex.owned();
    if (r >= 0 && r < m) {
        T acc = Ar::zero();
        for (int c = 0; c < n; ++c) Ar::mac(acc, a(r, c), x(c));
        store(alpha, acc, beta, y(r));
    }
}

namespace detail {
// a + alpha x conj(y) + conj(alpha) y conj(x) for one element; NegOne is a - (x conj(y) + y conj(x)).
template <class S, class T> BATCHLAS_DLA_INLINE T her2_apply(S alpha, T a, T xr, T yc, T yr, T xc) {
    using Ar = Arith<T>;
    if constexpr (std::is_same_v<S, NegOne>) return Ar::sub(a, her2_term(xr, yc, yr, xc));
    else if constexpr (std::is_same_v<S, One>) return Ar::add(a, her2_term(xr, yc, yr, xc));
    else return Ar::add(a, her2_term(tag_mul(alpha, xr), yc, Ar::mul(Ar::conj(static_cast<T>(alpha.v)), yr), xc));
}
}  // namespace detail

// A := alpha x y^H + conj(alpha) y x^H + A on the FULL square A (both triangles), one row per item.
template <class Ex, class AS, class X, class Y, class MA>
BATCHLAS_DLA_INLINE void her2_full(const Fit<Ex>& ex, AS alpha_, const X& xp, const Y& yp, const MA& a) {
    using T = typename MA::value_type;
    detail::require_published<Ex, X>();
    detail::require_published<Ex, Y>();
    const auto& x = detail::unpub(xp);
    const auto& y = detail::unpub(yp);
    const auto alpha = as_tag(alpha_);
    const int m = a.rows(), n = a.cols(), r = ex.owned();
    if (r >= 0 && r < m) {
        const T xr = x(r), yr = y(r);
        for (int c = 0; c < n; ++c) {
            const T xc = x(c), yc = y(c);
            a(r, c) = detail::her2_apply(alpha, static_cast<T>(a(r, c)), xr, yc, yr, xc);
        }
    }
}

// her2 on the UL triangle of a Hermitian A held in full storage: each updated element also writes
// its conjugate mirror, and Im(diag) is zeroed. One row per item; row r touches columns on its side
// of the diagonal only.
template <Uplo UL, class Ex, class AS, class X, class Y, class MA>
BATCHLAS_DLA_INLINE void her2_mirror(const Fit<Ex>& ex, AS alpha_, const X& xp, const Y& yp, const MA& a) {
    using T = typename MA::value_type;
    using Ar = Arith<T>;
    detail::require_published<Ex, X>();
    detail::require_published<Ex, Y>();
    const auto& x = detail::unpub(xp);
    const auto& y = detail::unpub(yp);
    const auto alpha = as_tag(alpha_);
    const int n = a.rows(), r = ex.owned();
    if (r >= 0 && r < n) {
        const T xr = x(r), yr = y(r);
        const int c0 = UL == Uplo::Lower ? 0 : r, c1 = UL == Uplo::Lower ? r : n - 1;
        for (int c = c0; c <= c1; ++c) {
            const T xc = x(c), yc = y(c);
            const T v = detail::her2_apply(alpha, static_cast<T>(a(r, c)), xr, yc, yr, xc);
            a(r, c) = v;
            if (r != c) a(c, r) = Ar::conj(v);
            else if constexpr (Ar::is_complex) a(r, c) = Ar::make(Ar::re(v), real_t<T>(0));
        }
    }
}

// B := A (m x n), rows over the executor (one row per item on Fit).
template <class Ex, class MA, class MB> BATCHLAS_DLA_INLINE void lacpy(const Ex& ex, const MA& a, const MB& b) {
    const int n = a.cols();
    for_each_owned(ex, a.rows(), [&](int r) {
        for (int c = 0; c < n; ++c) b(r, c) = a(r, c);
    });
}

// tile := the Hermitian matrix whose UL triangle is a (n x n, n = a.rows()), Im(diag) dropped,
// zero past n up to the tile's extent. Item r reads row r of the triangle (coalesced) and writes
// it and its conjugate mirror, so a barrier must precede reads of other rows. active == false
// (a masked-off batch item) zero-fills the whole tile and reads nothing.
template <Uplo UL, class Ex, class MA, class MT>
BATCHLAS_DLA_INLINE void lacpy_herm(const Fit<Ex>& ex, const MA& a, const MT& tile, bool active = true) {
    using T = typename MT::value_type;
    using Ar = Arith<T>;
    const int n = a.rows(), r = ex.owned();
    if (r < 0 || r >= tile.rows()) return;
    if (active && r < n) {
        for (int c = 0; c < n; ++c) {
            if (UL == Uplo::Upper ? r > c : r < c) continue;
            T v = a(r, c);
            if (r == c) {
                if constexpr (Ar::is_complex) v = Ar::make(Ar::re(v), real_t<T>(0));
            } else {
                tile(c, r) = Ar::conj(v);
            }
            tile(r, c) = v;
        }
        for (int c = n; c < tile.cols(); ++c) tile(r, c) = Ar::zero();
    } else {
        for (int c = 0; c < tile.cols(); ++c) tile(r, c) = Ar::zero();
    }
}

// sum_k x1(k) conj(y1(k)) + x2(k) conj(y2(k)), both terms per k: one her2k / syr2k entry
// (A(r, c) -= dot2c(ex, V.row(r), W.row(c), W.row(r), V.row(c))).
template <Scope S = Scope::Auto, class Ex, class X1, class Y1, class X2, class Y2>
BATCHLAS_DLA_INLINE auto dot2c(const Ex& ex, const X1& x1, const Y1& y1, const X2& x2, const Y2& y2) {
    using T = elem_t<X1>;
    T acc = Arith<T>::zero();
    for_each_owned(ex, x1.size(), [&](int k) { acc = Arith<T>::add(acc, her2_term<T>(x1(k), y1(k), x2(k), y2(k))); });
    return detail::scope_reduce<S>(ex, acc);
}

// ---- LAPACK ---------------------------------------------------------------------------------
struct NoStep {
    template <class... A> BATCHLAS_DLA_INLINE void operator()(const A&...) const {}
};

namespace detail {
// One sytd2 step on the active block blk: [alpha; x] = col in RowFrag (alpha on item ar), row is
// col's conjugate mirror. Reflector; mirrored write-back; then, unless tau == 0, v := reflector,
// w := tau A v, w += (-tau/2)(v^H w) v, A -= v w^H + w v^H.
template <Uplo UL, class Ex, class Blk, class Col, class Row, class V, class W, class T, class F>
BATCHLAS_DLA_INLINE void sytd2_step(const Ex& ex, int i, int m, int ar, const Blk& blk, const Col& col, const Row& row,
                                    const V& v, const W& w, bool x_active, T& tau_lane, F& on_step) {
    using Ar = Arith<T>;
    const Fit<Ex> fx{ex, 0};
    const int lane = ex.lid();
    const auto vm = v.head(m);
    const auto wm = w.head(m);
    const bool is_alpha = lane == ar;
    T alpha = Ar::zero(), x = Ar::zero();
    if (lane < m) {
        const T e = col(lane);
        if (is_alpha) alpha = e;
        else x = e;
    }
    const T taui = larfg_rowfrag(ex, m, ar, alpha, x, x_active);
    if (x_active) {  // both triangles: the block update reads the full block
        col(lane) = x;
        row(lane) = Ar::conj(x);
    }
    if (is_alpha) {
        col(ar) = alpha;
        row(ar) = Ar::conj(alpha);
        if constexpr (UL == Uplo::Upper) tau_lane = taui;
        on_step(i, alpha, taui);
    }
    if (taui != Ar::zero()) {
        copy(fx, implied_unit(col, ar), vm);
        team_barrier(ex);
        if (is_alpha) {  // LAPACK's temporary unit; the block ops read v, not the column
            col(ar) = Ar::one();
            row(ar) = Ar::one();
        }
        team_barrier(ex);
        gemv<Transpose::NoTrans>(fx, Scalar<T>{taui}, blk, assume_published(vm), Zero{}, wm);
        team_barrier(ex);
        const T alpha2 = T(-0.5) * taui * dotc(fx, vm, wm);
        axpy(fx, Scalar<T>{alpha2}, vm, wm);
        team_barrier(ex);
        her2_full(fx, NegOne{}, assume_published(vm), assume_published(wm), blk);
        team_barrier(ex);
        if (is_alpha) {
            col(ar) = alpha;
            row(ar) = Ar::conj(alpha);
        }
        team_barrier(ex);
    } else if constexpr (UL == Uplo::Upper) {
        if (is_alpha) row(ar) = Ar::conj(alpha);
    }
}
}  // namespace detail

// xSYTD2 / xHETD2 on the n x n leading block of a resident tile held in FULL storage (both
// triangles, real diagonal), one row per item (n <= Ex::size); v and w are length >= n scratch.
// Leaves LAPACK's packing, every element's conjugate mirror kept:
//   d(i) = A(i,i); Upper: e(i) = A(i,i+1), reflector i in A(0:i-1, i+1) (unit at row i);
//                  Lower: e(i) = A(i+1,i), reflector i in A(i+2:n-1, i) (unit at row i+1).
// on_step(i, beta, tau) runs on the item holding reflector i's alpha right after it is formed.
// Upper returns tau(lid) on item lid (0 past n-2); Lower returns 0 (tau goes through on_step).
// Step k touches only its active block, so column k is final once step k has run.
template <Uplo UL, class Ex, class MA, class V, class W, class F = NoStep>
BATCHLAS_DLA_INLINE typename MA::value_type sytd2(const Ex& ex, const MA& a, const V& v, const W& w, int n,
                                                  F&& on_step = {}) {
    using T = typename MA::value_type;
    const int lane = ex.lid();
    T tau_lane = Arith<T>::zero();
    if constexpr (UL == Uplo::Upper) {
        for (int k = n - 1; k >= 1; --k) {  // annihilate A(0:k-2, k); alpha = A(k-1, k)
            detail::sytd2_step<UL>(ex, k - 1, k, k - 1, a.sub(0, 0, k, k), a.col(k).head(k), a.row(k).head(k), v, w,
                                   lane < k - 1, tau_lane, on_step);
            team_barrier(ex);
        }
    } else {
        for (int i = 0; i <= n - 2; ++i) {  // annihilate A(i+2:n-1, i); alpha = A(i+1, i) on item 0
            const int s = i + 1, m = n - s;
            detail::sytd2_step<UL>(ex, i, m, 0, a.sub(s, s, m, m), a.col(i).tail(s).head(m), a.row(i).tail(s).head(m), v,
                                   w, lane >= 1 && lane < m, tau_lane, on_step);
        }
    }
    return tau_lane;
}

}  // namespace batchlas::device::v2
