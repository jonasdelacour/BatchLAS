// SPDX-License-Identifier: MIT
#pragma once
// Band reduction (sy2sb / sb2st) and its Q2 back-transform: the device LA v2 ops only this
// family uses. Shared ops come from the v2.1 core (Teams, sync, Vec/Mat, dotc, sumsq,
// larfg_scalars, lacpy, map); these have no core form:
//   StridedTeams<C>         C strided teams per sub-group (xor masks C, 2C, ...)
//   StridedMat, win_of      a row-major tile and index-form windows of any 2-D operand
//   fn_vec                  a vector over an index map
//   HermBandLower           a Hermitian matrix in LAPACK lower band storage
//   larf_native, hemv_band, her2   lockstep team loops in T's own operators
//   largv, lartv, lar2v, rot_by    plane rotations through the caller's Rot kernels
// Each element is the C++ expression of the kernel it was lifted from (std::complex operators,
// one expression per FMA chain), so a strict port stays bit-identical.

#include <complex>
#include <cstdint>
#include <type_traits>

#include <sycl/sycl.hpp>

#include <batchlas/blas/device/v2/dla.hh>
#include <batchlas/util/group-collectives.hh>

namespace batchlas::sb2st_v2 {

namespace v2 = batchlas::device::v2;

template <class T> using native_t = v2::arith_t<v2::arith::Native, T>;

namespace detail {
template <class T> BATCHLAS_DLA_INLINE T real_as(const T& a) {
    if constexpr (v2::Arith<T>::is_complex) return T(a.real(), typename T::value_type(0));
    else return a;
}
template <class G, class T> BATCHLAS_DLA_INLINE T shfl_xor(const G& g, const T& v, std::uint32_t m) {
    if constexpr (v2::Arith<T>::is_complex) {
        using R = v2::real_t<T>;
        const R re = batchlas::portable::permute_group_by_xor(g, static_cast<R>(v.real()), m);
        const R im = batchlas::portable::permute_group_by_xor(g, static_cast<R>(v.imag()), m);
        return T(re, im);
    } else {
        return batchlas::portable::permute_group_by_xor(g, v, m);
    }
}
// Ascending xor butterfly over the distances Lo, 2 Lo, ..., Hi / 2 (core's has Lo = 1 only).
template <int Lo, int Hi, class G, class T, class Op> BATCHLAS_DLA_INLINE T bfly_from(const G& g, T v, Op op) {
    constexpr bool plus = std::is_same_v<Op, sycl::plus<T>>;
    static_assert(plus || !v2::Arith<T>::is_complex, "sb2st_v2: a complex team reduction is a sum");
    auto step = [&](std::uint32_t m) BATCHLAS_DLA_BODY {
        if constexpr (plus) v += shfl_xor(g, v, m);
        else v = op(v, shfl_xor(g, v, m));
    };
    for (std::uint32_t m = Lo; m < static_cast<std::uint32_t>(Hi); m <<= 1) step(m);
    return v;
}
template <class V> using vt = typename V::value_type;

// The lockstep team loop: output o = ob * nteams + team for ob < ceil(nout / nteams); f(o, o < nout)
// runs on every lane so the team_reduce inside f is reached uniformly (core larf's j += nteams
// loop diverges inside a sub-group when P < SG).
template <class Ex, class F> BATCHLAS_DLA_INLINE void for_each_team_out(const Ex& ex, int nout, F&& f) {
    const int nb = (nout + Ex::nteams - 1) / Ex::nteams;
    for (int ob = 0; ob < nb; ++ob) {
        const int o = (Ex::nteams == 1) ? ob : ob * Ex::nteams + ex.team();
        f(o, o < nout);
    }
}
}  // namespace detail

// ---- executor -----------------------------------------------------------------------------
// C strided teams inside each sub-group: lane l is member l / C of team l mod C (G = SG / C
// members, xor distances C, 2C, ...): the column-tile layout of a resident Z tile. WG == SG
// lets lane() skip the modulo; barrier() is the sub-group's then, else the work-group's.
template <int C, int WG = 32, int SG = 32> struct StridedTeams {
    static_assert(v2::detail::is_pow2(C) && C <= SG && WG % SG == 0, "sb2st_v2::StridedTeams: C | SG | WG");
    static constexpr int size = SG / C, team_size = SG / C, nteams = C, sg_size = SG;
    static constexpr bool sub_whole = true;
    static constexpr v2::Scope natural_scope = v2::Scope::Sub;
    sycl::nd_item<1> it;
    int lane_ = (WG == SG) ? static_cast<int>(it.get_local_linear_id()) : static_cast<int>(it.get_local_linear_id()) % SG;
    int grp_ = lane_ / C;  // member index (the hand kernels' grp)
    int col_ = lane_ % C;  // team index (the hand kernels' col)

    BATCHLAS_DLA_INLINE int lane() const { return lane_; }
    BATCHLAS_DLA_INLINE int lid() const { return grp_; }
    BATCHLAS_DLA_INLINE int team() const { return col_; }
    BATCHLAS_DLA_INLINE bool leader() const { return lid() == 0; }
    BATCHLAS_DLA_INLINE sycl::sub_group sub() const { return it.get_sub_group(); }
    BATCHLAS_DLA_INLINE void barrier(v2::Mem = v2::Mem::Local) const {
        if constexpr (WG == SG) sycl::group_barrier(it.get_sub_group());
        else sycl::group_barrier(it.get_group());
    }
    template <class T, class Op = sycl::plus<T>> BATCHLAS_DLA_INLINE T team_reduce(T v, Op op = {}) const {
        return detail::bfly_from<C, SG>(it.get_sub_group(), v, op);
    }
    template <class T, class Op = sycl::plus<T>> BATCHLAS_DLA_INLINE T group_reduce(T v, Op op = {}) const {
        return team_reduce(v, op);
    }
};

// ---- operands -------------------------------------------------------------------------------
// A matrix with both strides explicit: A(i, j) = p[i * rs + j * cs]. Row-major tiles are
// StridedMat<T, Ext<C>, Unit>.
template <class T, class RS = v2::Dyn, class CS = v2::Dyn> struct StridedMat {
    using value_type = std::remove_cv_t<T>;
    T* p;
    int m;
    int n;
    RS rs{};
    CS cs{};
    BATCHLAS_DLA_INLINE T& operator()(int i, int j) const { return p[i * int(rs) + j * int(cs)]; }
    BATCHLAS_DLA_INLINE int rows() const { return m; }
    BATCHLAS_DLA_INLINE int cols() const { return n; }
};
// The row-major m x C tile at p (row stride C): a column tile resident in local memory.
template <int C, class T> BATCHLAS_DLA_INLINE StridedMat<T, v2::Ext<C>, v2::Unit> row_major_tile(T* p, int m) {
    return {p, m, C};
}

// Index-form window of any 2-D operand: element (i, j) is a(i0 + i, j0 + j), the pointer never
// moves (core Mat::sub covers column-major Mats; this is for StridedMat and HermBandLower).
template <class M> struct WinOf {
    using value_type = std::remove_cvref_t<decltype(std::declval<const M&>()(0, 0))>;
    M a;
    int i0, j0, m, n;
    BATCHLAS_DLA_INLINE decltype(auto) operator()(int i, int j) const { return a(i0 + i, j0 + j); }
    BATCHLAS_DLA_INLINE int rows() const { return m; }
    BATCHLAS_DLA_INLINE int cols() const { return n; }
};
template <class M> BATCHLAS_DLA_INLINE WinOf<M> win_of(const M& a, int i0, int j0, int m, int n) { return {a, i0, j0, m, n}; }

// x(t) = f(t), t < n: a vector over an index map, f returning the element (a reference to write
// through). Each element keeps the caller's address expression, e.g. a band chase's rotation
// sequence AB(r, j1 + t kd1 + dc), so a port keeps the hand kernel's index arithmetic.
template <class F> struct FnVec {
    using value_type = std::remove_cvref_t<decltype(std::declval<const F&>()(0))>;
    F f;
    int n;
    BATCHLAS_DLA_INLINE decltype(auto) operator()(int i) const { return f(i); }
    BATCHLAS_DLA_INLINE int size() const { return n; }
};
template <class F> BATCHLAS_DLA_INLINE FnVec<F> fn_vec(int n, F f) { return {f, n}; }

// A Hermitian (real: symmetric) matrix in LAPACK lower band storage, half-bandwidth kd:
// A(i, j), i >= j, lives at p[(i - j) + j * ld]. A(i, j) above the diagonal reads conj(A(j, i));
// outside the band it reads 0 and a write is dropped. Elements are BandRef proxies; get(i, j)
// reads a value (core reductions take value-returning operands).
template <class T> struct HermBandLower;
template <class T> struct BandRef {
    HermBandLower<T> b;
    int i, j;
    BATCHLAS_DLA_INLINE operator T() const { return b.get(i, j); }
    BATCHLAS_DLA_INLINE const BandRef& operator=(const T& v) const {
        b.set(i, j, v);
        return *this;
    }
    BATCHLAS_DLA_INLINE const BandRef& operator=(const BandRef& o) const {
        b.set(i, j, static_cast<T>(o));
        return *this;
    }
};
// Rows [i0, i0 + m) x columns [j0, j0 + n) of a band matrix.
template <class T> struct BandWin {
    using value_type = T;
    HermBandLower<T> b;
    int i0, j0, m, n;
    BATCHLAS_DLA_INLINE BandRef<T> operator()(int r, int c) const { return {b, i0 + r, j0 + c}; }
    BATCHLAS_DLA_INLINE int rows() const { return m; }
    BATCHLAS_DLA_INLINE int cols() const { return n; }
};
// A(i0 : i0 + n, j): a column segment of a band matrix.
template <class T> struct BandVec {
    using value_type = T;
    HermBandLower<T> b;
    int i0, j, n;
    BATCHLAS_DLA_INLINE BandRef<T> operator()(int k) const { return {b, i0 + k, j}; }
    BATCHLAS_DLA_INLINE int size() const { return n; }
    BATCHLAS_DLA_INLINE BandVec tail(int k) const { return {b, i0 + k, j, n - k}; }
};
template <class T> struct HermBandLower {
    using value_type = T;
    T* p;
    int ld;
    int kd;
    BATCHLAS_DLA_INLINE T get(int i, int j) const {
        if (i >= j) {
            const int r = i - j;
            return (r <= kd) ? p[r + j * ld] : T(0);
        }
        const int r = j - i;
        return (r <= kd) ? native_t<T>::conj(p[r + i * ld]) : T(0);
    }
    BATCHLAS_DLA_INLINE void set(int i, int j, T v) const {
        if (i >= j) {
            const int r = i - j;
            if (r <= kd) p[r + j * ld] = v;
        } else {
            const int r = j - i;
            if (r <= kd) p[r + i * ld] = native_t<T>::conj(v);
        }
    }
    BATCHLAS_DLA_INLINE BandRef<T> operator()(int i, int j) const { return {*this, i, j}; }
    BATCHLAS_DLA_INLINE BandWin<T> sub(int i0, int j0, int m, int n) const { return {*this, i0, j0, m, n}; }
    BATCHLAS_DLA_INLINE BandVec<T> col(int j, int i0, int m) const { return {*this, i0, j, m}; }
};
template <class V> inline constexpr bool is_band_win_v = false;
template <class T> inline constexpr bool is_band_win_v<BandWin<T>> = true;

// ---- Householder --------------------------------------------------------------------------
namespace larf_form {
struct ScaleW {};  // C -= v (tau v^H C): tau folds into the reduced vector (LAPACK larf)
struct ScaleV {};  // C -= (tau v) (v^H C): tau folds into v
}  // namespace larf_form

// C := (I - tau v v^H) C (Side::Left) or C (I - tau v v^H) (Side::Right) in T's own operators.
// Each team owns whole columns (Left) or rows (Right) of C, reduces v^H C (C v) with team_reduce
// in a lockstep loop and updates its own vector: no barrier inside; the caller publishes C.
// Core larf differs: Arith (explicit) products, ScaleW only, a non-lockstep team loop.
template <Side SD, class Form = larf_form::ScaleW, class Ex, class Tau, class VP, class MC>
BATCHLAS_DLA_INLINE void larf_native(const Ex& ex, Tau tau, const VP& vp, const MC& c) {
    using T = detail::vt<MC>;
    using N = native_t<T>;
    constexpr bool left = SD == Side::Left;
    constexpr bool scale_w = std::is_same_v<Form, larf_form::ScaleW>;
    static_assert(scale_w || (std::is_same_v<Form, larf_form::ScaleV> && left), "sb2st_v2::larf_native: ScaleV is Side::Left only");
    v2::detail::require_published<Ex, VP>();
    const auto& v = v2::detail::unpub(vp);
    const int nout = left ? c.cols() : c.rows();
    const int len = left ? c.rows() : c.cols();
    detail::for_each_team_out(ex, nout, [&](int o, bool live) BATCHLAS_DLA_BODY {
        T w = T(0);
        if (live) {
            for (int k = ex.lid(); k < len; k += Ex::size) {
                if constexpr (left) w += N::conj(static_cast<T>(v(k))) * static_cast<T>(c(k, o));
                else w += static_cast<T>(c(o, k)) * static_cast<T>(v(k));
            }
        }
        if constexpr (scale_w) w = tau * ex.team_reduce(w);
        else w = ex.team_reduce(w);
        if (live) {
            for (int k = ex.lid(); k < len; k += Ex::size) {
                if constexpr (!left) c(o, k) = static_cast<T>(c(o, k)) - w * N::conj(static_cast<T>(v(k)));
                else if constexpr (scale_w) c(k, o) = static_cast<T>(c(k, o)) - static_cast<T>(v(k)) * w;
                else c(k, o) = static_cast<T>(c(k, o)) - tau * static_cast<T>(v(k)) * w;
            }
        }
    });
}

// y := alpha A x + beta y for a window of a Hermitian band matrix (every entry readable through
// the mirror): one team per row, lanes stride the columns, lockstep.
template <Uplo UL, class Ex, class AS, class MA, class X, class BS, class Y>
    requires is_band_win_v<MA>
BATCHLAS_DLA_INLINE void hemv_band(const Ex& ex, AS alpha_, const MA& a, const X& xp, BS beta_, const Y& y) {
    using T = detail::vt<MA>;
    static_assert(UL == Uplo::Lower, "sb2st_v2::hemv_band: lower band storage only");
    v2::detail::require_published<Ex, X>();
    const auto& x = v2::detail::unpub(xp);
    const auto alpha = v2::as_tag(alpha_);
    const auto beta = v2::as_tag(beta_);
    const int m = a.rows(), n = a.cols();
    detail::for_each_team_out(ex, m, [&](int r, bool live) BATCHLAS_DLA_BODY {
        T acc = T(0);
        if (live) {
            for (int j = ex.lid(); j < n; j += Ex::size) acc += static_cast<T>(a(r, j)) * static_cast<T>(x(j));
        }
        acc = ex.team_reduce(acc);
        if (live && ex.leader()) v2::store(alpha, acc, beta, y(r));
    });
}

// A := alpha x y^H + conj(alpha) y x^H + A on the lower triangle; for complex T the diagonal is
// rounded back to real. Team per row, lanes stride the columns; no reduction.
template <Uplo UL, class Ex, class AS, class XP, class YP, class MA>
BATCHLAS_DLA_INLINE void her2(const Ex& ex, AS alpha_, const XP& xp, const YP& yp, const MA& a) {
    using T = detail::vt<MA>;
    using N = native_t<T>;
    using S = v2::tag_t<AS>;
    static_assert(UL == Uplo::Lower, "sb2st_v2::her2: Uplo::Lower only");
    v2::detail::require_published<Ex, XP>();
    v2::detail::require_published<Ex, YP>();
    const auto& x = v2::detail::unpub(xp);
    const auto& y = v2::detail::unpub(yp);
    const S alpha = v2::as_tag(alpha_);
    (void)alpha;  // read only by the Scalar form
    const int m = a.rows();
    for (int r = (Ex::nteams == 1 ? 0 : ex.team()); r < m; r += Ex::nteams) {
        for (int j = ex.lid(); j <= r; j += Ex::size) {
            T val;
            if constexpr (std::is_same_v<S, v2::NegOne>)
                val = static_cast<T>(a(r, j)) - static_cast<T>(x(r)) * N::conj(static_cast<T>(y(j))) -
                      static_cast<T>(y(r)) * N::conj(static_cast<T>(x(j)));
            else if constexpr (std::is_same_v<S, v2::One>)
                val = static_cast<T>(a(r, j)) + static_cast<T>(x(r)) * N::conj(static_cast<T>(y(j))) +
                      static_cast<T>(y(r)) * N::conj(static_cast<T>(x(j)));
            else
                val = static_cast<T>(a(r, j)) + static_cast<T>(alpha.v) * static_cast<T>(x(r)) * N::conj(static_cast<T>(y(j))) +
                      N::conj(static_cast<T>(alpha.v)) * static_cast<T>(y(r)) * N::conj(static_cast<T>(x(j)));
            if constexpr (v2::Arith<T>::is_complex) {
                if (r == j) val = detail::real_as(val);
            }
            a(r, j) = val;
        }
    }
}

// ---- plane rotations ----------------------------------------------------------------------------
// Rot supplies the scalar kernels (LAPACK xLARTG, one xROT pair, one xLAR2V triple), c real:
//   Rot::lartg(f, g, c&, s&, r&)   Rot::rot_pair(x&, y&, c, s)   Rot::lar2v_pair(x&, y&, z&, c, s)
// The ops distribute the rotation index over the executor's Cyclic layout; LAPACK's vector strides
// are the operands' own. Core rot / lasr fix the arithmetic (Arith products), these keep Rot's.

// (c(t), s(t)) := lartg(x(t), y(t)); x(t) := r, y(t) := s  (xLARGV, y holds g in and s out)
template <class Rot, class Ex, class X, class Y, class C>
BATCHLAS_DLA_INLINE void largv(const Ex& ex, const X& x, const Y& y, const C& c) {
    using T = detail::vt<X>;
    using R = v2::real_t<T>;
    v2::for_each_owned(ex, x.size(), [&](int t) BATCHLAS_DLA_BODY {
        const T f = x(t);
        const T g = y(t);
        R ct = R(0);
        T st = T(0);
        T rt = T(0);
        Rot::lartg(f, g, ct, st, rt);
        c(t) = ct;
        y(t) = st;
        x(t) = rt;
    });
}

// (x(t), y(t)) := rotation t applied to the pair  (xLARTV)
template <class Rot, class Ex, class X, class Y, class C, class S>
BATCHLAS_DLA_INLINE void lartv(const Ex& ex, const X& x, const Y& y, const C& c, const S& s) {
    v2::for_each_owned(ex, x.size(), [&](int t) BATCHLAS_DLA_BODY { Rot::rot_pair(x(t), y(t), c(t), s(t)); });
}

// two-sided rotation t of the 2x2 Hermitian block [x z; conj(z) y]  (xLAR2V)
template <class Rot, class Ex, class X, class Y, class Z, class C, class S>
BATCHLAS_DLA_INLINE void lar2v(const Ex& ex, const X& x, const Y& y, const Z& z, const C& c, const S& s) {
    v2::for_each_owned(ex, x.size(), [&](int t) BATCHLAS_DLA_BODY { Rot::lar2v_pair(x(t), y(t), z(t), c(t), s(t)); });
}

// one rotation (c, s) applied to the vector pair (x, y)  (xROT through Rot::rot_pair)
template <class Rot, class Ex, class X, class Y, class Rc, class Sc>
BATCHLAS_DLA_INLINE void rot_by(const Ex& ex, const X& x, const Y& y, Rc c, const Sc& s) {
    v2::for_each_owned(ex, x.size(), [&](int i) BATCHLAS_DLA_BODY { Rot::rot_pair(x(i), y(i), c, s); });
}

}  // namespace batchlas::sb2st_v2
