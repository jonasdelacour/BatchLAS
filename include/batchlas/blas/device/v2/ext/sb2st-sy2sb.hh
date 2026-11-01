// SPDX-License-Identifier: MIT
#pragma once
// Device LA v2 extension for band reduction (sy2sb / sb2st) and its back-transform: team
// executors narrower than a sub-group, a Hermitian band operand, Householder (larfg, larf,
// hemv / her2 on a band window) and plane-rotation (largv, lartv, lar2v, rot) ops.
//
// Arithmetic form: these ops evaluate each element with the same C++ expression as the kernels
// they were lifted from (std::complex operators, one expression per FMA chain), so a 1:1 port is
// bit-identical and keeps its SASS (-ffp-contract=on fuses within one expression only). The
// Arith<T> explicit-complex form is a separate, measured step (docs/perf/sytrd.md).

#include <complex>
#include <cstdint>
#include <type_traits>

#include <sycl/sycl.hpp>

#include <batchlas/blas/device/v2/dla.hh>
#include <batchlas/util/group-collectives.hh>

namespace batchlas::device::v2 {

namespace arith {
struct Native {};  // the operators as written (Annex G complex product); legacy bit-identical
}  // namespace arith

namespace larf_form {
struct ScaleW {};  // C -= v (tau v^H C): tau folds into the reduced vector (LAPACK larf)
struct ScaleV {};  // C -= (tau v) (v^H C): tau folds into v
}  // namespace larf_form

enum class Bfly { Down, Up };  // xor-butterfly mask order: P/2 .. 1, or 1 .. P/2

namespace detail {
template <class T> BATCHLAS_DLA_INLINE T nconj(const T& a) {
    if constexpr (Arith<T>::is_complex) return std::conj(a);
    else return a;
}
template <class T> BATCHLAS_DLA_INLINE T real_as(const T& a) {
    if constexpr (Arith<T>::is_complex) return T(a.real(), typename T::value_type(0));
    else return a;
}
// |a|^2 as its own expression: the caller's `acc += abs2n(a)` must not fuse into an FMA.
template <class T> BATCHLAS_DLA_INLINE real_t<T> abs2n(const T& a) {
    using R = real_t<T>;
    if constexpr (Arith<T>::is_complex)
        return static_cast<R>(a.real()) * static_cast<R>(a.real()) + static_cast<R>(a.imag()) * static_cast<R>(a.imag());
    else return static_cast<R>(a) * static_cast<R>(a);
}
template <class G, class T> BATCHLAS_DLA_INLINE T shfl_xor(const G& g, const T& v, std::uint32_t m) {
    if constexpr (Arith<T>::is_complex) {
        using R = real_t<T>;
        const R re = batchlas::portable::permute_group_by_xor(g, static_cast<R>(v.real()), m);
        const R im = batchlas::portable::permute_group_by_xor(g, static_cast<R>(v.imag()), m);
        return T(re, im);
    } else {
        return batchlas::portable::permute_group_by_xor(g, v, m);
    }
}
// Butterfly over lanes whose xor-distance is a multiple of Lo below Hi (Lo, 2 Lo, ..., Hi / 2).
template <int Lo, int Hi, Bfly O, class G, class T, class Op> BATCHLAS_DLA_INLINE T bfly(const G& g, T v, Op op) {
    constexpr bool plus = std::is_same_v<Op, sycl::plus<T>>;
    static_assert(plus || !Arith<T>::is_complex, "v2: a complex team reduction is a sum");
    auto step = [&](std::uint32_t m) {
        if constexpr (plus) v += shfl_xor(g, v, m);
        else v = op(v, shfl_xor(g, v, m));
    };
    if constexpr (O == Bfly::Up) {
        for (std::uint32_t m = Lo; m < static_cast<std::uint32_t>(Hi); m <<= 1) step(m);
    } else {
#pragma unroll
        for (std::uint32_t m = Hi / 2; m >= static_cast<std::uint32_t>(Lo); m >>= 1) step(m);
    }
    return v;
}
}  // namespace detail

// ---- executors ------------------------------------------------------------------------------
// WG/P teams of P consecutive lanes (P <= SG): team t = lid / P walks rows t, t + WG/P, ...; its
// lanes stride the reduced dimension. Lockstep: every lane of a sub-group must reach each
// team_reduce, so the team ops below run a uniform trip count and mask the work instead.
template <int WG, int P, Bfly O = Bfly::Down, int SG = 32> struct TeamsOf {
    static_assert(detail::is_pow2(P) && P <= SG && WG % SG == 0, "v2::TeamsOf: P | SG | WG");
    static constexpr int size = P, team_size = P, nteams = WG / P, sg_size = SG;
    static constexpr bool sub_whole = true;
    static constexpr Scope natural_scope = Scope::Sub;
    sycl::nd_item<1> it;

    BATCHLAS_DLA_INLINE int lid() const { return static_cast<int>(it.get_local_linear_id()) % P; }
    BATCHLAS_DLA_INLINE int lane() const { return lid(); }
    BATCHLAS_DLA_INLINE int team() const { return static_cast<int>(it.get_local_linear_id()) / P; }
    BATCHLAS_DLA_INLINE bool leader() const { return lid() == 0; }
    template <class T, class Op = sycl::plus<T>> BATCHLAS_DLA_INLINE T team_reduce(T v, Op op = {}) const {
        return detail::bfly<1, P, O>(it.get_sub_group(), v, op);
    }
    template <class T, class Op = sycl::plus<T>> BATCHLAS_DLA_INLINE T group_reduce(T v, Op op = {}) const {
        return team_reduce(v, op);
    }
};

// C strided teams inside each sub-group: lane l is member l / C of team l mod C (G = SG / C
// members, xor distances C, 2C, ...). With one column per team this is the column-tile layout
// of a resident Z tile; WG == SG lets lane() skip the modulo.
template <int C, int WG = 32, int SG = 32> struct StridedTeams {
    static_assert(detail::is_pow2(C) && C <= SG && WG % SG == 0, "v2::StridedTeams: C | SG | WG");
    static constexpr int size = SG / C, team_size = SG / C, nteams = C, sg_size = SG;
    static constexpr bool sub_whole = true;
    static constexpr Scope natural_scope = Scope::Sub;
    sycl::nd_item<1> it;

    BATCHLAS_DLA_INLINE int lane() const {
        const int l = static_cast<int>(it.get_local_linear_id());
        if constexpr (WG == SG) return l;
        else return l % SG;
    }
    BATCHLAS_DLA_INLINE int lid() const { return lane() / C; }
    BATCHLAS_DLA_INLINE int team() const { return lane() % C; }
    BATCHLAS_DLA_INLINE bool leader() const { return lid() == 0; }
    template <class T, class Op = sycl::plus<T>> BATCHLAS_DLA_INLINE T team_reduce(T v, Op op = {}) const {
        return detail::bfly<C, SG, Bfly::Up>(it.get_sub_group(), v, op);
    }
    template <class T, class Op = sycl::plus<T>> BATCHLAS_DLA_INLINE T group_reduce(T v, Op op = {}) const {
        return team_reduce(v, op);
    }
};

// sycl::group_barrier on the executor's group (fence scope = the group, all address spaces):
// the form the band kernels use, kept so ports keep their fences.
template <int WG, int SG> BATCHLAS_DLA_INLINE void group_barrier(const WorkGroup<WG, SG>& ex) {
    sycl::group_barrier(ex.it.get_group());
}
template <int C, int WG, int SG> BATCHLAS_DLA_INLINE void group_barrier(const StridedTeams<C, WG, SG>& ex) {
    if constexpr (WG == SG) sycl::group_barrier(ex.it.get_sub_group());
    else sycl::group_barrier(ex.it.get_group());
}

// ---- operands -------------------------------------------------------------------------------
// A vector over any indexable base (a local_accessor keeps its address space): x(i) = base[off + i].
template <class Base> struct IdxVec {
    using value_type = std::remove_cvref_t<decltype(std::declval<const Base&>()[0])>;
    static constexpr Mem mem = Mem::Local;
    Base base;
    int off;
    int n;
    BATCHLAS_DLA_INLINE decltype(auto) operator()(int i) const { return base[off + i]; }
    BATCHLAS_DLA_INLINE int size() const { return n; }
    BATCHLAS_DLA_INLINE IdxVec tail(int k) const { return {base, off + k, n - k}; }
    BATCHLAS_DLA_INLINE IdxVec head(int k) const { return {base, off, k}; }
};
template <class Base> BATCHLAS_DLA_INLINE IdxVec<Base> vec_of(const Base& b, int n, int off = 0) { return {b, off, n}; }

// A matrix with both strides explicit: A(i, j) = p[i * rs + j * cs]. Row-major tiles are
// StridedMat<T, Ext<C>, Unit>.
template <class T, class RS = Dyn, class CS = Dyn> struct StridedMat {
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
template <int C, class T> BATCHLAS_DLA_INLINE StridedMat<T, Ext<C>, Unit> row_major_tile(T* p, int m) {
    return {p, m, C};
}

// A Hermitian (real: symmetric) matrix in LAPACK lower band storage, half-bandwidth kd:
// A(i, j), i >= j, lives at p[(i - j) + j * ld]. A(i, j) above the diagonal reads conj(A(j, i));
// outside the band it reads 0 and a write is dropped.
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
        return (r <= kd) ? detail::nconj(p[r + i * ld]) : T(0);
    }
    BATCHLAS_DLA_INLINE void set(int i, int j, T v) const {
        if (i >= j) {
            const int r = i - j;
            if (r <= kd) p[r + j * ld] = v;
        } else {
            const int r = j - i;
            if (r <= kd) p[r + i * ld] = detail::nconj(v);
        }
    }
    BATCHLAS_DLA_INLINE BandRef<T> operator()(int i, int j) const { return {*this, i, j}; }
    BATCHLAS_DLA_INLINE BandWin<T> sub(int i0, int j0, int m, int n) const { return {*this, i0, j0, m, n}; }
    BATCHLAS_DLA_INLINE BandVec<T> col(int j, int i0, int m) const { return {*this, i0, j, m}; }
};
template <class V> inline constexpr bool is_band_win_v = false;
template <class T> inline constexpr bool is_band_win_v<BandWin<T>> = true;

namespace detail {
template <class V> using vt = typename V::value_type;

// The lockstep team loop: output o = ob * nteams + team for ob < ceil(nout / nteams); f(o, o < nout)
// runs on every lane so the team_reduce inside f is reached uniformly.
template <class Ex, class F> BATCHLAS_DLA_INLINE void for_each_team_out(const Ex& ex, int nout, F&& f) {
    const int nb = (nout + Ex::nteams - 1) / Ex::nteams;
    for (int ob = 0; ob < nb; ++ob) {
        const int o = (Ex::nteams == 1) ? ob : ob * Ex::nteams + ex.team();
        f(o, o < nout);
    }
}
}  // namespace detail

// ---- level 1 -----------------------------------------------------------------------------
// x^H y in the operator form (no explicit-complex rewrite): v2::dotc<arith::Native>(ex, x, y).
template <class AP, Scope S = Scope::Auto, class Ex, class X, class Y>
    requires std::is_same_v<AP, arith::Native>
BATCHLAS_DLA_INLINE auto dotc(const Ex& ex, const X& x, const Y& y) {
    using T = detail::vt<X>;
    T acc = T(0);
    for_each_owned(ex, x.size(), [&](int i) { acc += detail::nconj(static_cast<T>(x(i))) * static_cast<T>(y(i)); });
    return detail::scope_reduce<S>(ex, acc);
}

// ||x||_2^2 = sum |x_i|^2: one real reduction, unscaled (overflows past sqrt(max)).
template <Scope S = Scope::Auto, class Ex, class X> BATCHLAS_DLA_INLINE auto nrm2sq(const Ex& ex, const X& x) {
    using T = detail::vt<X>;
    using R = real_t<T>;
    R acc = R(0);
    for_each_owned(ex, x.size(), [&](int i) { acc += detail::abs2n(static_cast<T>(x(i))); });
    return detail::scope_reduce<S>(ex, acc);
}

// y(i) = i < x.size() ? x(i) : 0 for i < y.size()  (lacpy + laset of the tail)
template <class Ex, class X, class Y> BATCHLAS_DLA_INLINE void copy_pad(const Ex& ex, const X& x, const Y& y) {
    using T = detail::vt<Y>;
    const int m = x.size();
    for_each_owned(ex, y.size(), [&](int i) { y(i) = (i < m) ? static_cast<T>(x(i)) : T(0); });
}

// x := conj(x) (LAPACK xLACGV); a no-op for real T.
template <class Ex, class X> BATCHLAS_DLA_INLINE void conj_inplace(const Ex& ex, const X& x) {
    using T = detail::vt<X>;
    if constexpr (Arith<T>::is_complex)
        for_each_owned(ex, x.size(), [&](int i) { x(i) = detail::nconj(static_cast<T>(x(i))); });
}

// ---- Householder ------------------------------------------------------------------------------
template <class T> struct ReflScalars {
    T tau;
    T beta;
    T scale;  // v(1:) = scale * x(1:); unused when tau == 0
};

// LAPACK xLARFG scalars from alpha and ||x||: H = I - tau v v^H, H^H (alpha; x) = (beta; 0).
template <class T> BATCHLAS_DLA_INLINE ReflScalars<T> larfg_from_norm(const T& alpha, real_t<T> xnorm, int len) {
    using R = real_t<T>;
    ReflScalars<T> res{T(0), alpha, T(0)};
    if (len <= 1) return res;
    if constexpr (Arith<T>::is_complex) {
        if (xnorm == R(0) && alpha.imag() == R(0)) return res;
        const R alpha_abs = sycl::hypot(alpha.real(), alpha.imag());
        const R beta_abs = sycl::hypot(alpha_abs, xnorm);
        const T alpha_sign = (alpha_abs == R(0)) ? T(1) : (alpha / alpha_abs);
        res.beta = -alpha_sign * T(beta_abs);
        res.tau = (res.beta - alpha) / res.beta;
        res.scale = T(1) / (alpha - res.beta);
    } else {
        if (xnorm == R(0)) return res;
        const R alpha_r = static_cast<R>(alpha);
        const R sign = sycl::signbit(alpha_r) ? R(-1) : R(1);
        const R beta = -sign * sycl::hypot(alpha_r, xnorm);
        res.beta = T(beta);
        res.tau = (res.beta - alpha) / res.beta;
        res.scale = T(1) / (alpha - res.beta);
    }
    return res;
}

// The reflector of y = (alpha; x): unscaled ||x|| (one reduction of scope S), then the larfg
// scalars. y is not written: where beta and v go is the caller's choice.
template <Scope S = Scope::Auto, class Ex, class Y> BATCHLAS_DLA_INLINE auto larfg(const Ex& ex, const Y& y) {
    using T = detail::vt<Y>;
    const auto ss = nrm2sq<S>(ex, y.tail(1));
    const auto xnorm = sycl::sqrt(ss);
    const T alpha = y(0);
    return larfg_from_norm<T>(alpha, xnorm, y.size());
}

// C := (I - tau v v^H) C (Side::Left) or C (I - tau v v^H) (Side::Right). Each team owns whole
// columns (Left) or rows (Right) of C, reduces v^H C (C v) with team_reduce and updates its own
// vector, so no barrier is needed inside; the caller publishes C before the next reader.
template <Side SD, class Form = larf_form::ScaleW, class Ex, class VP, class Tau, class MC>
BATCHLAS_DLA_INLINE void larf(const Ex& ex, const VP& vp, Tau tau, const MC& c) {
    using T = detail::vt<MC>;
    constexpr bool left = SD == Side::Left;
    constexpr bool scale_w = std::is_same_v<Form, larf_form::ScaleW>;
    static_assert(scale_w || (std::is_same_v<Form, larf_form::ScaleV> && left), "v2::larf: ScaleV is Side::Left only");
    detail::require_published<Ex, VP>();
    const auto& v = detail::unpub(vp);
    const int nout = left ? c.cols() : c.rows();
    const int len = left ? c.rows() : c.cols();
    detail::for_each_team_out(ex, nout, [&](int o, bool live) {
        T w = T(0);
        if (live) {
            for (int k = ex.lid(); k < len; k += Ex::size) {
                if constexpr (left) w += detail::nconj(static_cast<T>(v(k))) * static_cast<T>(c(k, o));
                else w += static_cast<T>(c(o, k)) * static_cast<T>(v(k));
            }
        }
        if constexpr (scale_w) w = tau * ex.team_reduce(w);
        else w = ex.team_reduce(w);
        if (live) {
            for (int k = ex.lid(); k < len; k += Ex::size) {
                if constexpr (!left) c(o, k) = static_cast<T>(c(o, k)) - w * detail::nconj(static_cast<T>(v(k)));
                else if constexpr (scale_w) c(k, o) = static_cast<T>(c(k, o)) - static_cast<T>(v(k)) * w;
                else c(k, o) = static_cast<T>(c(k, o)) - tau * static_cast<T>(v(k)) * w;
            }
        }
    });
}

// y := alpha A x + beta y for a window of a Hermitian band matrix (every entry readable through
// the mirror): one team per row, lanes stride the columns, lockstep (alg::TeamPerRow).
template <Uplo UL, class Alg = void, class Ex, class AS, class MA, class X, class BS, class Y>
    requires is_band_win_v<MA>
BATCHLAS_DLA_INLINE void hemv(const Ex& ex, AS alpha_, const MA& a, const X& xp, BS beta_, const Y& y) {
    using T = detail::vt<MA>;
    static_assert(UL == Uplo::Lower, "v2::hemv (band): lower band storage only");
    detail::require_published<Ex, X>();
    const auto& x = detail::unpub(xp);
    const auto alpha = as_tag(alpha_);
    const auto beta = as_tag(beta_);
    const int m = a.rows(), n = a.cols();
    detail::for_each_team_out(ex, m, [&](int r, bool live) {
        T acc = T(0);
        if (live) {
            for (int j = ex.lid(); j < n; j += Ex::size) acc += static_cast<T>(a(r, j)) * static_cast<T>(x(j));
        }
        acc = ex.team_reduce(acc);
        if (live && ex.leader()) store(alpha, acc, beta, y(r));
    });
}

// A := alpha x y^H + conj(alpha) y x^H + A on the lower triangle; for complex T the diagonal is
// rounded back to real. Team per row, lanes stride the columns; no reduction.
template <Uplo UL, class Ex, class AS, class XP, class YP, class MA>
BATCHLAS_DLA_INLINE void her2(const Ex& ex, AS alpha_, const XP& xp, const YP& yp, const MA& a) {
    using T = detail::vt<MA>;
    using S = tag_t<AS>;
    static_assert(UL == Uplo::Lower, "v2::her2: Uplo::Lower only");
    detail::require_published<Ex, XP>();
    detail::require_published<Ex, YP>();
    const auto& x = detail::unpub(xp);
    const auto& y = detail::unpub(yp);
    const S alpha = as_tag(alpha_);
    (void)alpha;  // read only by the Scalar form
    const int m = a.rows();
    for (int r = (Ex::nteams == 1 ? 0 : ex.team()); r < m; r += Ex::nteams) {
        for (int j = ex.lid(); j <= r; j += Ex::size) {
            T val;
            if constexpr (std::is_same_v<S, NegOne>)
                val = static_cast<T>(a(r, j)) - static_cast<T>(x(r)) * detail::nconj(static_cast<T>(y(j))) -
                      static_cast<T>(y(r)) * detail::nconj(static_cast<T>(x(j)));
            else if constexpr (std::is_same_v<S, One>)
                val = static_cast<T>(a(r, j)) + static_cast<T>(x(r)) * detail::nconj(static_cast<T>(y(j))) +
                      static_cast<T>(y(r)) * detail::nconj(static_cast<T>(x(j)));
            else
                val = static_cast<T>(a(r, j)) + static_cast<T>(alpha.v) * static_cast<T>(x(r)) * detail::nconj(static_cast<T>(y(j))) +
                      detail::nconj(static_cast<T>(alpha.v)) * static_cast<T>(y(r)) * detail::nconj(static_cast<T>(x(j)));
            if constexpr (Arith<T>::is_complex) {
                if (r == j) val = detail::real_as(val);
            }
            a(r, j) = val;
        }
    }
}

// ---- plane rotations ----------------------------------------------------------------------------
// Rot supplies the scalar kernels (LAPACK xLARTG, one xROT pair, one xLAR2V triple):
//   Rot::lartg(f, g, c&, s&, r&)   Rot::rot_pair(x&, y&, c, s)   Rot::lar2v_pair(x&, y&, z&, c, s)
// with c real. The ops distribute the rotation index over the executor's Cyclic layout; the
// strides of LAPACK's vector forms are the operands' own strides.

// (c(t), s(t)) := lartg(x(t), y(t)); x(t) := r, y(t) := s  (xLARGV, y holds g in and s out)
template <class Rot, class Ex, class X, class Y, class C>
BATCHLAS_DLA_INLINE void largv(const Ex& ex, const X& x, const Y& y, const C& c) {
    using T = detail::vt<X>;
    using R = real_t<T>;
    for_each_owned(ex, x.size(), [&](int t) {
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
    for_each_owned(ex, x.size(), [&](int t) { Rot::rot_pair(x(t), y(t), c(t), s(t)); });
}

// two-sided rotation t of the 2x2 Hermitian block [x z; conj(z) y]  (xLAR2V)
template <class Rot, class Ex, class X, class Y, class Z, class C, class S>
BATCHLAS_DLA_INLINE void lar2v(const Ex& ex, const X& x, const Y& y, const Z& z, const C& c, const S& s) {
    for_each_owned(ex, x.size(), [&](int t) { Rot::lar2v_pair(x(t), y(t), z(t), c(t), s(t)); });
}

// one rotation (c, s) applied to the vector pair (x, y)  (xROT)
template <class Rot, class Ex, class X, class Y, class Rc, class Sc>
BATCHLAS_DLA_INLINE void rot(const Ex& ex, const X& x, const Y& y, Rc c, const Sc& s) {
    for_each_owned(ex, x.size(), [&](int i) { Rot::rot_pair(x(i), y(i), c, s); });
}

}  // namespace batchlas::device::v2
