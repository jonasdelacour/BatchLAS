// SPDX-License-Identifier: MIT
#pragma once
// Device LA v2, scalar LAPACK: the 2x2 and reflector kernels every eigen / SVD / QR family
// recomputes per item. Pure functions of their arguments, no executor; host-callable.
//   safmin / safmax          dlamch('S') and its reciprocal
//   lartg                    plane rotation [c s; -conj(s) c] [f; g] = [r; 0] (LAPACK 3.10)
//   lae2 / laev2             eigenvalues (and eigenvector) of [[a, b], [b, c]]
//   las2                     singular values of [[f, g], [0, h]]
//   wilkinson_shift          dsteqr's shift from the leading 2x2 of a tridiagonal window
//   larfg_scalars<Conv>      the reflector scalars of xLARFG under three conventions
//   Givens<R, S>             a rotation (c real, s real or complex)
//   inv_sqrt<Edge>           1 / sqrt(a): rsqrt.approx + one Newton step on NVPTX float

#include <array>
#include <cmath>
#include <complex>
#include <limits>
#include <type_traits>

#include <batchlas/device/core.hh>

namespace batchlas::device {

template <class R> constexpr R safmin() {
    constexpr R eps = std::numeric_limits<R>::epsilon();
    constexpr R s1 = std::numeric_limits<R>::min();
    constexpr R s2 = R(1) / std::numeric_limits<R>::max();
    return s2 > s1 ? s2 * (R(1) + eps) : s1;
}
template <class R> constexpr R safmax() { return R(1) / safmin<R>(); }

// lassq's pair: scale = max |component|, scale^2 * ssq = sum |x_i|^2.
template <class R> struct ScaledSsq {
    R scale;
    R ssq;
};

// A plane rotation; S = R (real) or T (complex s, c stays real).
template <class R, class S = R> struct Givens {
    R c;
    S s;
};

// |a| as LAPACK's abs (hypot for complex), not BLAS abs1.
template <class T> BATCHLAS_DLA_INLINE real_t<T> modulus(const T& a) {
    using A = Arith<T>;
    if constexpr (A::is_complex) return sycl::hypot(A::re(a), A::im(a));
    else return sycl::fabs(a);
}

// ---- lartg -------------------------------------------------------------------------------------
template <class T> struct LartgResult {
    real_t<T> c;
    T s;
    T r;
};

namespace detail {
// CLARTG/ZLARTG with f, g != 0, the scalar math in W (scaled against overflow / underflow).
template <class W, class T> BATCHLAS_DLA_INLINE LartgResult<T> lartg_complex_scaled(const T& f, const T& g) {
    using R = real_t<T>;
    const W f_abs_w = sycl::hypot(static_cast<W>(f.real()), static_cast<W>(f.imag()));
    const W g_abs_w = sycl::hypot(static_cast<W>(g.real()), static_cast<W>(g.imag()));
    const W u = sycl::fmin(safmax<W>(), sycl::fmax(safmin<W>(), sycl::fmax(f_abs_w, g_abs_w)));
    const W fs = f_abs_w / u;
    const W gs = g_abs_w / u;
    const W norm = u * sycl::sqrt(fs * fs + gs * gs);
    const std::complex<W> f_w{static_cast<W>(f.real()), static_cast<W>(f.imag())};
    const std::complex<W> g_w{static_cast<W>(g.real()), static_cast<W>(g.imag())};
    const std::complex<W> alpha = f_w / f_abs_w;  // sgn(f)
    const W c_w = f_abs_w / norm;
    const std::complex<W> conjg_w{g_w.real(), -g_w.imag()};
    const std::complex<W> s_w = alpha * conjg_w / norm;
    const std::complex<W> r_w = alpha * norm;
    return {static_cast<R>(c_w), T{static_cast<R>(s_w.real()), static_cast<R>(s_w.imag())},
            T{static_cast<R>(r_w.real()), static_cast<R>(r_w.imag())}};
}
}  // namespace detail

// W: lartg_alg::Exact (= void) or a working precision type of the complex branch (void:
// real_t<T>); the real branch is SLARTG/DLARTG with its unscaled fast path, bit-identical to
// src/math-helpers.hh lartg. lartg_alg::Fast, float only (other T: Exact): inside the unscaled
// range r = sqrt(f^2 + g^2) with 1/r from rsqrt + one fused Newton step (< 0.5 ulp), outside
// it the Exact path; g == 0 gives (1, 0, f) by a final select, not an early return (an
// identity rotation does not diverge from its neighbours): steqr_cta's rotation.
namespace lartg_alg {
struct Exact {};
struct Fast {};
}  // namespace lartg_alg
template <class W = void, class T> inline LartgResult<T> lartg(const T& f, const T& g) {
    using R = real_t<T>;
    if constexpr (std::is_same_v<W, lartg_alg::Exact>) {
        return lartg<void>(f, g);
    } else if constexpr (std::is_same_v<W, lartg_alg::Fast>) {
        if constexpr (std::is_same_v<T, float>) {
            const bool gz = (g == T(0));
            const T f_abs = sycl::fabs(f), g_abs = sycl::fabs(g);
            const T rtmin = sycl::sqrt(safmin<T>());
            const T rtmax = sycl::sqrt(safmax<T>() / T(2));
            T c = T(1), s = T(0), r = f;
            if (f_abs > rtmin && f_abs < rtmax && (gz || (g_abs > rtmin && g_abs < rtmax))) {
                const T t = f * f + g * g;
                T inv = sycl::rsqrt(t);
                inv = sycl::fma(inv * T(0.5), sycl::fma(-(t * inv), inv, T(1)), inv);
                const T d = sycl::sqrt(t);
                c = f_abs * inv;
                s = g * sycl::copysign(inv, f);
                r = sycl::copysign(d, f);
            } else if (!gz) {
                const auto e = lartg<void>(f, g);
                c = e.c;
                s = e.s;
                r = e.r;
            }
            return {gz ? T(1) : c, gz ? T(0) : s, gz ? f : r};
        } else {
            return lartg<void>(f, g);
        }
    } else {
    const R f_abs = modulus(f);
    const R g_abs = modulus(g);
    if (g_abs == R(0)) return {R(1), T(0), f};
    if (f_abs == R(0)) {
        if constexpr (Arith<T>::is_complex) {
            const T conjg = T(g.real(), -g.imag());
            return {R(0), conjg / T(g_abs), T(g_abs, R(0))};
        } else {
            return {R(0), T(sycl::copysign(R(1), static_cast<R>(g))), T(g_abs)};
        }
    }
    if constexpr (Arith<T>::is_complex) {
        return detail::lartg_complex_scaled<std::conditional_t<std::is_void_v<W>, R, W>>(f, g);
    } else {
        const R rtmin = sycl::sqrt(safmin<R>());
        const R rtmax = sycl::sqrt(safmax<R>() / R(2));
        const R f_r = static_cast<R>(f), g_r = static_cast<R>(g);
        if (f_abs > rtmin && f_abs < rtmax && g_abs > rtmin && g_abs < rtmax) {
            const R d = sycl::sqrt(f_r * f_r + g_r * g_r);
            const R c = f_abs / d;
            const R r = sycl::copysign(d, f_r);
            return {c, T(g_r / r), T(r)};
        }
        const R u = sycl::fmin(safmax<R>(), sycl::fmax(safmin<R>(), sycl::fmax(f_abs, g_abs)));
        const R fs = f_r / u, gs = g_r / u;
        const R d = sycl::sqrt(fs * fs + gs * gs);
        const R c = sycl::fabs(fs) / d;
        const R r_unscaled = sycl::copysign(d, f_r);
        return {c, T(gs / r_unscaled), T(r_unscaled * u)};
    }
    }
}

// ---- symmetric / triangular 2x2 ---------------------------------------------------------------
// sqrt(adf^2 + (2b)^2) without overflow / underflow (shared by lae2 and laev2).
namespace detail {
template <class R> BATCHLAS_DLA_INLINE R lae2_rt(R adf, R ab) {
    if (adf > ab) return adf * sycl::sqrt(R(1) + (ab / adf) * (ab / adf));
    if (adf < ab) return ab * sycl::sqrt(R(1) + (adf / ab) * (adf / ab));
    return ab * sycl::sqrt(R(2));  // includes ab = adf = 0
}
}  // namespace detail

// SLAE2/DLAE2: {rt1, rt2} of [[a, b], [b, c]], |rt1| >= |rt2|.
template <class R> BATCHLAS_DLA_INLINE std::array<R, 2> lae2(R a, R b, R c) {
    const R sm = a + c, df = a - c, adf = sycl::fabs(df), tb = b + b, ab = sycl::fabs(tb);
    const bool amax = sycl::fabs(a) > sycl::fabs(c);
    const R acmx = amax ? a : c, acmn = amax ? c : a;
    const R rt = detail::lae2_rt(adf, ab);
    if (sm < R(0)) {
        const R rt1 = R(0.5) * (sm - rt);
        return {rt1, (acmx / rt1) * acmn - (b / rt1) * b};  // this order keeps rt2 accurate
    }
    if (sm > R(0)) {
        const R rt1 = R(0.5) * (sm + rt);
        return {rt1, (acmx / rt1) * acmn - (b / rt1) * b};
    }
    return {R(0.5) * rt, R(-0.5) * rt};
}

// SLAEV2/DLAEV2: {rt1, rt2, cs1, sn1}; (cs1, sn1) is the unit eigenvector of rt1.
template <class R> BATCHLAS_DLA_INLINE std::array<R, 4> laev2(R a, R b, R c) {
    const R sm = a + c, df = a - c, adf = sycl::fabs(df), tb = b + b, ab = sycl::fabs(tb);
    const bool amax = sycl::fabs(a) > sycl::fabs(c);
    const R acmx = amax ? a : c, acmn = amax ? c : a;
    const R rt = detail::lae2_rt(adf, ab);
    R rt1, rt2;
    int sgn1;
    if (sm < R(0)) {
        rt1 = R(0.5) * (sm - rt);
        sgn1 = -1;
        rt2 = (acmx / rt1) * acmn - (b / rt1) * b;
    } else if (sm > R(0)) {
        rt1 = R(0.5) * (sm + rt);
        sgn1 = 1;
        rt2 = (acmx / rt1) * acmn - (b / rt1) * b;
    } else {
        rt1 = R(0.5) * rt;
        rt2 = R(-0.5) * rt;
        sgn1 = 1;
    }
    const int sgn2 = df >= R(0) ? 1 : -1;
    const R cs = df >= R(0) ? df + rt : df - rt;
    R cs1, sn1;
    if (sycl::fabs(cs) > ab) {
        const R ct = -tb / cs;
        sn1 = R(1) / sycl::sqrt(R(1) + ct * ct);
        cs1 = ct * sn1;
    } else if (ab == R(0)) {
        cs1 = R(1);
        sn1 = R(0);
    } else {
        const R tn = -cs / tb;
        cs1 = R(1) / sycl::sqrt(R(1) + tn * tn);
        sn1 = tn * cs1;
    }
    if (sgn1 == sgn2) {
        const R tn = cs1;
        cs1 = -sn1;
        sn1 = tn;
    }
    return {rt1, rt2, cs1, sn1};
}

// SLAS2/DLAS2: {ssmin, ssmax} of the upper triangular [[f, g], [0, h]], nothing squared.
template <class R> BATCHLAS_DLA_INLINE std::array<R, 2> las2(R f, R g, R h) {
    const R fa = sycl::fabs(f), ga = sycl::fabs(g), ha = sycl::fabs(h);
    const R fhmn = fa < ha ? fa : ha, fhmx = fa < ha ? ha : fa;
    if (fhmn == R(0)) {
        if (fhmx == R(0)) return {R(0), ga};
        const R mx = fhmx > ga ? fhmx : ga, mn = fhmx > ga ? ga : fhmx;
        return {R(0), mx * sycl::sqrt(R(1) + (mn / mx) * (mn / mx))};
    }
    if (ga < fhmx) {
        const R as = R(1) + fhmn / fhmx, at = (fhmx - fhmn) / fhmx, au = (ga / fhmx) * (ga / fhmx);
        const R c = R(2) / (sycl::sqrt(as * as + au) + sycl::sqrt(at * at + au));
        return {fhmn * c, fhmx / c};
    }
    const R au = fhmx / ga;
    if (au == R(0)) return {(fhmn * fhmx) / ga, ga};
    const R as = R(1) + fhmn / fhmx, at = (fhmx - fhmn) / fhmx;
    const R c = R(1) / (sycl::sqrt(R(1) + (as * au) * (as * au)) + sycl::sqrt(R(1) + (at * au) * (at * au)));
    return {R(2) * ((fhmn * c) * au), ga / (c + c)};
}

// dsteqr's Wilkinson shift for the window starting at l: the eigenvalue of
// [[d(l), e(l)], [e(l), d(l+1)]] nearest d(l): g = (d(l+1) - d(l)) / (2 e(l)),
// d(l) - e(l) / (g + sign(hypot(g, 1), g)). e(l) != 0.
template <class R> BATCHLAS_DLA_INLINE R wilkinson_shift(R dl, R el, R dl1) {
    const R g = (dl1 - dl) / (R(2) * el);
    const R r = sycl::hypot(g, R(1));
    return dl - el / (g + sycl::copysign(r, g));
}

// ---- xLARFG scalars -----------------------------------------------------------------------------
// H^H [alpha; x] = [beta; 0], H = I - tau v v^H, v = [1; x * scale] (use_mul) or [1; x / scale]
// (the reciprocal overflowed or vanished). identity: tau = 0, H = I, x unchanged. Conventions:
//   conv::Lapack    xLARFG from (alpha, ScaledSsq{s, ssq}) with s = max |component| over alpha
//                   and x, ssq = sum |x_i / s|^2 (geqrf's two-pass norm): beta = -sign(alphr) *
//                   ||[alpha; x]|| real, tau complex; scale is Smith's 1 / (alpha - beta)
//                   (FmaArith::recip), use_mul unless it is not finite or zero.
//   conv::Phase     from (alpha, ||x||, len): beta = -sgn(alpha) ||[alpha; x]|| keeps alpha's
//                   phase (complex beta, real tau); len <= 1, or a real alpha over x == 0 ->
//                   identity (src detail::larfg).
//   conv::RealBeta  from (alpha, ||x||, len): LAPACK ZLARFG's beta real / tau complex; a complex
//                   alpha reflects even at len == 1. Real T: identical to Phase.
namespace conv {
struct Lapack {};
struct Phase {};
struct RealBeta {};
}  // namespace conv

template <class T> struct LarfgScalars {
    T tau;
    T beta;
    T scale;
    bool use_mul;
    bool identity;
};

template <class Conv = conv::Lapack, class T> BATCHLAS_DLA_INLINE LarfgScalars<T> larfg_scalars(const T& alpha, ScaledSsq<real_t<T>> q) {
    static_assert(std::is_same_v<Conv, conv::Lapack>, "v2::larfg_scalars(alpha, ScaledSsq): conv::Lapack");
    using A = Arith<T>;
    using R = real_t<T>;
    LarfgScalars<T> out{A::zero(), alpha, A::one(), true, true};
    const R alphi = A::im(alpha);
    if (q.ssq == R(0) && alphi == R(0)) return out;  // a complex alpha still needs a reflector
    if (!(q.scale > R(0))) return out;
    const R alphr = A::re(alpha);
    const R ar = alphr / q.scale, ai = alphi / q.scale;
    const R nrm = sycl::sqrt(sycl::fma(ar, ar, sycl::fma(ai, ai, q.ssq)));
    const R beta_s = (alphr >= R(0)) ? -nrm : nrm;  // -SIGN(nrm, alphr): +0 counts as positive
    out.tau = A::make((beta_s - ar) / beta_s, -ai / beta_s);
    out.beta = A::make(beta_s * q.scale, R(0));
    const T d = A::sub(alpha, out.beta);
    const T r = FmaArith<T>::recip(d);
    out.use_mul = A::isfinite(r) && !A::is_zero(r);
    out.scale = out.use_mul ? r : d;
    out.identity = false;
    return out;
}

template <class Conv, class T> inline LarfgScalars<T> larfg_scalars(const T& alpha, real_t<T> xnorm, int len) {
    static_assert(std::is_same_v<Conv, conv::Phase> || std::is_same_v<Conv, conv::RealBeta>,
                  "v2::larfg_scalars(alpha, xnorm, len): conv::Phase or conv::RealBeta");
    using R = real_t<T>;
    LarfgScalars<T> out{T(0), alpha, T(0), true, true};
    if constexpr (std::is_same_v<Conv, conv::RealBeta> && Arith<T>::is_complex) {
        if (len < 1 || (xnorm == R(0) && alpha.imag() == R(0))) return out;
        const R norm = sycl::hypot(sycl::hypot(alpha.real(), alpha.imag()), xnorm);
        if (norm == R(0)) return out;
        const R beta = sycl::signbit(alpha.real()) ? norm : -norm;
        out.beta = T(beta);
        out.tau = T((beta - alpha.real()) / beta, -alpha.imag() / beta);
    } else if constexpr (Arith<T>::is_complex) {
        if (len <= 1) return out;
        if (xnorm == R(0) && alpha.imag() == R(0)) return out;
        const R alpha_abs = sycl::hypot(alpha.real(), alpha.imag());
        const R beta_abs = sycl::hypot(alpha_abs, xnorm);
        const T alpha_sign = (alpha_abs == R(0)) ? T(1) : (alpha / alpha_abs);
        out.beta = -alpha_sign * T(beta_abs);
        out.tau = (out.beta - alpha) / out.beta;
    } else {
        if (len <= 1 || xnorm == R(0)) return out;
        const R sign = sycl::signbit(alpha) ? R(-1) : R(1);
        out.beta = T(-sign * sycl::hypot(alpha, xnorm));
        out.tau = (out.beta - alpha) / out.beta;
    }
    out.scale = T(1) / (alpha - out.beta);
    out.identity = false;
    return out;
}

// ---- v2.2 scalar LAPACK (design A.2 scalar.hh) ----------------------------------------------
// dlamch: lamch<'E'>() eps (relative machine precision, eps/2 of numeric_limits as LAPACK 3.x
// with rounding), 'S' safe minimum, 'P' eps * base, 'B' base, 'N' digits, 'U' underflow
// threshold (min normal), 'O' overflow threshold (max), 'R' 1 (rounding mode).
template <char C, class R> constexpr R lamch() {
    constexpr R eps = std::numeric_limits<R>::epsilon() * R(0.5);
    if constexpr (C == 'E') return eps;
    else if constexpr (C == 'S') return safmin<R>();
    else if constexpr (C == 'P') return eps * R(std::numeric_limits<R>::radix);
    else if constexpr (C == 'B') return R(std::numeric_limits<R>::radix);
    else if constexpr (C == 'N') return R(std::numeric_limits<R>::digits);
    else if constexpr (C == 'U') return std::numeric_limits<R>::min();
    else if constexpr (C == 'O') return std::numeric_limits<R>::max();
    else {
        static_assert(C == 'R', "lamch: one of E S P B N U O R");
        return R(1);
    }
}

// A quiet NaN of T's real type (both parts NaN for complex T).
template <class T> BATCHLAS_DLA_INLINE T qnan() {
    using A = Arith<T>;
    using R = typename A::real_t;
    return A::make(std::numeric_limits<R>::quiet_NaN(), std::numeric_limits<R>::quiet_NaN());
}

// x / |x|, and 1 at x == 0 (a unit-modulus sign: the phase a Hermitian tridiagonal's
// off-diagonal is rotated by). Real T: copysign(1, x) with +1 at +-0.
template <class T> BATCHLAS_DLA_INLINE T phase(T x) {
    using A = Arith<T>;
    using R = typename A::real_t;
    if constexpr (A::is_complex) {
        const R m = sycl::hypot(A::re(x), A::im(x));
        if (m == R(0)) return A::one();
        return A::make(A::re(x) / m, A::im(x) / m);
    } else {
        return x == R(0) ? R(1) : sycl::copysign(R(1), x);
    }
}
// sign(re x) as +-1 with +1 at 0 (LAPACK's SIGN(ONE, x) on a real part).
template <class T> BATCHLAS_DLA_INLINE real_t<T> sign_nonzero_real(T x) {
    using R = real_t<T>;
    const R r = Arith<T>::re(x);
    return r < R(0) ? R(-1) : R(1);
}

// The eigenvalue of [[a, b], [b, c]] nearer to c (dsteqr's shift target): rt1 or rt2 of lae2,
// whichever is closer to c (ties: rt1). The open-coded steqr form.
template <class R> BATCHLAS_DLA_INLINE R eig2x2_nearest(R a, R b, R c) {
    const auto [rt1, rt2] = lae2(a, b, c);
    return sycl::fabs(rt1 - c) <= sycl::fabs(rt2 - c) ? rt1 : rt2;
}

// ---- 1 / sqrt -----------------------------------------------------------------------------------
// inv_sqrt<E>(a, sqrt_a) = 1 / sqrt(a), sqrt_a = sqrt(a) the divisor for callers that hold it.
// NVPTX float (not AdaptiveCpp): rsqrt.approx + one FMA Newton step, y + (y/2)(1 - (a y) y), near
// correctly rounded and the same whatever is inlined (decision 10; a plain 1 / sqrt(a) folds into
// rsqrt.approx by code shape). Edge: Lapack 1/inf = 0, not NaN; FiniteGeOne a finite a >= 1 (no
// guard, the Jacobi cosine 1 + t^2); Exact 1 / sqrt_a on every target. Elsewhere 1 / sqrt_a.
enum class InvSqrtEdge { Lapack, FiniteGeOne, Exact };

template <InvSqrtEdge E = InvSqrtEdge::Lapack, class R> inline R inv_sqrt(R a, R sqrt_a) {
#if !BATCHLAS_SYCL_IMPL_ACPP && defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
    if constexpr (std::is_same_v<R, float> && E != InvSqrtEdge::Exact) {
        float y;
        asm("rsqrt.approx.f32 %0, %1;" : "=f"(y) : "f"(a));
        const float e = sycl::fma(-(a * y), y, 1.0f);
        const float r = sycl::fma(0.5f * y, e, y);
        if constexpr (E == InvSqrtEdge::Lapack) return sycl::isinf(a) ? 0.0f : r;
        else return r;
    }
#endif
    (void)a;
    return R(1) / sqrt_a;
}
template <InvSqrtEdge E = InvSqrtEdge::Lapack, class R> inline R inv_sqrt(R a) { return inv_sqrt<E>(a, sycl::sqrt(a)); }

// Multiply by the reciprocal when it is finite and nonzero, else divide (the sfmin arm), else
// (an exact zero d) the caller's value. make() decides once per pivot; apply() branches on the
// (uniform) flags, never a select (a select pays the divide every time).
// evidence: docs/perf/lu.md#the-rhs-pad-column-cost
template <class T, class Ops = arith::Fma> struct SafeRecip {
    T d;
    T r;
    bool zero;
    bool use_mul;
    BATCHLAS_DLA_INLINE static SafeRecip make(T d) {
        using Ar = arith_t<Ops, T>;
        const bool z = Arith<T>::is_zero(d);  // EXACT zero, no epsilon
        const T rc = Ar::recip(d);
        return {d, rc, z, !z && Arith<T>::isfinite(rc) && !Arith<T>::is_zero(rc)};
    }
    BATCHLAS_DLA_INLINE T apply(T y, T if_zero) const {
        using Ar = arith_t<Ops, T>;
        T x = if_zero;
        if (use_mul) x = Ar::mul(y, r);
        else if (!zero) x = Ar::div(y, d);
        return x;
    }
};

}  // namespace batchlas::device
