// SPDX-License-Identifier: MIT
#pragma once
/** @file
 * @brief Scalar LAPACK: the 2x2 and reflector kernels every eigen, SVD and QR family recomputes
 * per item.
 * @ingroup api_device_arith
 *
 * They are pure functions of their arguments with no executor, and callable on the host:
 * safmin and safmax (dlamch('S') and its reciprocal), lamch, lartg (a plane rotation, LAPACK
 * 3.10), lae2 and laev2 (the eigenvalues and eigenvector of a symmetric 2x2), las2 (the singular
 * values of an upper triangular 2x2), lapack_shift and eig2x2_nearest (dsteqr's shifts),
 * larfg_scalars (the reflector scalars of xLARFG under four conventions), house2, Givens, qnan,
 * phase, inv_sqrt and SafeRecip.
 */

#include <array>
#include <cmath>
#include <complex>
#include <limits>
#include <type_traits>

#include <batchlas/device/core.hh>
#include <batchlas/device/detail/sscp_target.hh>

namespace batchlas::device {

/// @brief LAPACK's safe minimum, dlamch('S'): the smallest x whose reciprocal does not overflow.
/// @ingroup api_device_arith
/// @tparam Real float or double
/// @return the safe minimum
template <class Real> constexpr Real safmin() {
    constexpr Real eps = std::numeric_limits<Real>::epsilon();
    constexpr Real s1 = std::numeric_limits<Real>::min();
    constexpr Real s2 = Real(1) / std::numeric_limits<Real>::max();
    return s2 > s1 ? s2 * (Real(1) + eps) : s1;
}
/// @brief The reciprocal of safmin.
/// @ingroup api_device_arith
/// @tparam Real float or double
/// @return 1 / safmin
template <class Real> constexpr Real safmax() { return Real(1) / safmin<Real>(); }

/// @brief lassq's pair: scale = max |component|, and scale^2 * ssq = sum |x_i|^2.
/// @ingroup api_device_arith
/// @tparam Real the real type
template <class Real> struct ScaledSsq {
    Real scale;  ///< The scale, max |component|.
    Real ssq;  ///< The scaled sum of squares.
};

/// @brief A plane rotation with a real c and a real or complex s.
/// @ingroup api_device_arith
/// @tparam Real the real type of c
/// @tparam S Real (a real rotation) or the complex type (complex s)
template <class Real, class S = Real> struct Givens {
    Real c;  ///< The cosine.
    S s;  ///< The sine.
};

/// @brief |a| as LAPACK's abs: fabs for a real a, hypot(re, im) for a complex a (not BLAS's abs1).
/// @ingroup api_device_arith
/// @param a the value
/// @return the modulus
template <class Scalar> BATCHLAS_DEVICE_INLINE real_t<Scalar> modulus(const Scalar& a) {
    using A = Arith<Scalar>;
    if constexpr (A::is_complex) return sycl::hypot(A::re(a), A::im(a));
    else return sycl::fabs(a);
}

// ---- lartg -------------------------------------------------------------------------------------
/// @brief The rotation lartg returns: [c s; -conj(s) c] [f; g] = [r; 0].
/// @ingroup api_device_arith
/// @tparam Scalar the scalar type of f and g
template <class Scalar> struct LartgResult {
    real_t<Scalar> c;  ///< The cosine, real.
    Scalar s;  ///< The sine.
    Scalar r;  ///< The rotated value.
};

namespace detail {
// CLARTG/ZLARTG with f, g != 0, the scalar math in W (scaled against overflow / underflow).
template <class W, class Scalar> BATCHLAS_DEVICE_INLINE LartgResult<Scalar> lartg_complex_scaled(const Scalar& f,
                                                                                                 const Scalar& g) {
    using R = real_t<Scalar>;
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
    return {static_cast<R>(c_w), Scalar{static_cast<R>(s_w.real()), static_cast<R>(s_w.imag())},
            Scalar{static_cast<R>(r_w.real()), static_cast<R>(r_w.imag())}};
}
}  // namespace detail

namespace alg {
/// @brief lartg's exact LAPACK algorithm.
/// @ingroup api_device_arith
struct LartgExact {};
/// @brief lartg's fast float path: rsqrt and one Newton step inside the unscaled range.
/// @ingroup api_device_arith
struct LartgFast {};
/// @brief lartg in double for `std::complex<float>`, except on Metal.
/// @ingroup api_device_arith
struct Promote {};
}  // namespace alg
/// @brief The plane rotation of xLARTG (LAPACK 3.10): c, s and r with [c s; -conj(s) c] [f; g] =
/// [r; 0].
/// @ingroup api_device_arith
///
/// The real branch is SLARTG / DLARTG with its unscaled fast path, bit-identical to the host's
/// lartg. W selects the arithmetic:
/// - void or alg::LartgExact: the exact LAPACK algorithm; the complex branch works in real_t.
/// - a floating type: the working precision of the complex branch.
/// - alg::Promote: `std::complex<float>` works in double except on Metal, which has no fp64
///   (chosen at JIT time), for less rounding noise in a downstream bulge chase; every other type
///   is Exact.
/// - alg::LartgFast, float only (other types: Exact): inside the unscaled range r = sqrt(f^2 +
///   g^2) with 1 / r from rsqrt and one fused Newton step (< 0.5 ulp); outside it the Exact path.
///   g == 0 gives (1, 0, f) by a final select, not an early return, so an identity rotation does
///   not diverge from its neighbours.
/// @tparam W the arithmetic, as above
/// @param f the first component
/// @param g the second component
/// @return the LartgResult
// LartgFast is steqr_cta's rotation.
template <class W = void, class Scalar> BATCHLAS_DEVICE_HELPER LartgResult<Scalar> lartg(const Scalar& f,
                                                                                         const Scalar& g) {
    using R = real_t<Scalar>;
    if constexpr (std::is_same_v<W, alg::LartgExact>) {
        return lartg<void>(f, g);
    } else if constexpr (std::is_same_v<W, alg::Promote>) {
        if constexpr (Arith<Scalar>::is_complex && std::is_same_v<R, float>) {
            if (::batchlas::sycl_impl::on_metal()) return lartg<float>(f, g);
            return lartg<double>(f, g);
        } else {
            return lartg<void>(f, g);
        }
    } else if constexpr (std::is_same_v<W, alg::LartgFast>) {
        if constexpr (std::is_same_v<Scalar, float>) {
            const bool gz = (g == Scalar(0));
            const Scalar f_abs = sycl::fabs(f), g_abs = sycl::fabs(g);
            const Scalar rtmin = sycl::sqrt(safmin<Scalar>());
            const Scalar rtmax = sycl::sqrt(safmax<Scalar>() / Scalar(2));
            Scalar c = Scalar(1), s = Scalar(0), r = f;
            if (f_abs > rtmin && f_abs < rtmax && (gz || (g_abs > rtmin && g_abs < rtmax))) {
                const Scalar t = f * f + g * g;
                Scalar inv = sycl::rsqrt(t);
                inv = sycl::fma(inv * Scalar(0.5), sycl::fma(-(t * inv), inv, Scalar(1)), inv);
                const Scalar d = sycl::sqrt(t);
                c = f_abs * inv;
                s = g * sycl::copysign(inv, f);
                r = sycl::copysign(d, f);
            } else if (!gz) {
                const auto e = lartg<void>(f, g);
                c = e.c;
                s = e.s;
                r = e.r;
            }
            return {gz ? Scalar(1) : c, gz ? Scalar(0) : s, gz ? f : r};
        } else {
            return lartg<void>(f, g);
        }
    } else {
    const R f_abs = modulus(f);
    const R g_abs = modulus(g);
    if (g_abs == R(0)) return {R(1), Scalar(0), f};
    if (f_abs == R(0)) {
        if constexpr (Arith<Scalar>::is_complex) {
            const Scalar conjg = Scalar(g.real(), -g.imag());
            return {R(0), conjg / Scalar(g_abs), Scalar(g_abs, R(0))};
        } else {
            return {R(0), Scalar(sycl::copysign(R(1), static_cast<R>(g))), Scalar(g_abs)};
        }
    }
    if constexpr (Arith<Scalar>::is_complex) {
        return detail::lartg_complex_scaled<std::conditional_t<std::is_void_v<W>, R, W>>(f, g);
    } else {
        const R rtmin = sycl::sqrt(safmin<R>());
        const R rtmax = sycl::sqrt(safmax<R>() / R(2));
        const R f_r = static_cast<R>(f), g_r = static_cast<R>(g);
        if (f_abs > rtmin && f_abs < rtmax && g_abs > rtmin && g_abs < rtmax) {
            const R d = sycl::sqrt(f_r * f_r + g_r * g_r);
            const R c = f_abs / d;
            const R r = sycl::copysign(d, f_r);
            return {c, Scalar(g_r / r), Scalar(r)};
        }
        const R u = sycl::fmin(safmax<R>(), sycl::fmax(safmin<R>(), sycl::fmax(f_abs, g_abs)));
        const R fs = f_r / u, gs = g_r / u;
        const R d = sycl::sqrt(fs * fs + gs * gs);
        const R c = sycl::fabs(fs) / d;
        const R r_unscaled = sycl::copysign(d, f_r);
        return {c, Scalar(gs / r_unscaled), Scalar(r_unscaled * u)};
    }
    }
}

// ---- symmetric / triangular 2x2 ---------------------------------------------------------------
// sqrt(adf^2 + (2b)^2) without overflow / underflow (shared by lae2 and laev2).
namespace detail {
template <class Real> BATCHLAS_DEVICE_INLINE Real lae2_rt(Real adf, Real ab) {
    if (adf > ab) return adf * sycl::sqrt(Real(1) + (ab / adf) * (ab / adf));
    if (adf < ab) return ab * sycl::sqrt(Real(1) + (adf / ab) * (adf / ab));
    return ab * sycl::sqrt(Real(2));  // includes ab = adf = 0
}
}  // namespace detail

/// @brief SLAE2 / DLAE2: the eigenvalues {rt1, rt2} of [[a, b], [b, c]], |rt1| >= |rt2|.
/// @ingroup api_device_arith
/// @param a the (1, 1) entry
/// @param b the off-diagonal entry
/// @param c the (2, 2) entry
/// @return {rt1, rt2}
template <class Real> BATCHLAS_DEVICE_INLINE std::array<Real, 2> lae2(const Real& a, const Real& b, const Real& c) {
    const Real sm = a + c, df = a - c, adf = sycl::fabs(df), tb = b + b, ab = sycl::fabs(tb);
    const bool amax = sycl::fabs(a) > sycl::fabs(c);
    const Real acmx = amax ? a : c, acmn = amax ? c : a;
    const Real rt = detail::lae2_rt(adf, ab);
    if (sm < Real(0)) {
        const Real rt1 = Real(0.5) * (sm - rt);
        return {rt1, (acmx / rt1) * acmn - (b / rt1) * b};  // this order keeps rt2 accurate
    }
    if (sm > Real(0)) {
        const Real rt1 = Real(0.5) * (sm + rt);
        return {rt1, (acmx / rt1) * acmn - (b / rt1) * b};
    }
    return {Real(0.5) * rt, Real(-0.5) * rt};
}

/// @brief SLAEV2 / DLAEV2: {rt1, rt2, cs1, sn1}, where (cs1, sn1) is the unit eigenvector of rt1.
/// @ingroup api_device_arith
/// @param a the (1, 1) entry
/// @param b the off-diagonal entry
/// @param c the (2, 2) entry
/// @return {rt1, rt2, cs1, sn1}, |rt1| >= |rt2|
template <class Real> BATCHLAS_DEVICE_INLINE std::array<Real, 4> laev2(const Real& a, const Real& b, const Real& c) {
    const Real sm = a + c, df = a - c, adf = sycl::fabs(df), tb = b + b, ab = sycl::fabs(tb);
    const bool amax = sycl::fabs(a) > sycl::fabs(c);
    const Real acmx = amax ? a : c, acmn = amax ? c : a;
    const Real rt = detail::lae2_rt(adf, ab);
    Real rt1, rt2;
    int sgn1;
    if (sm < Real(0)) {
        rt1 = Real(0.5) * (sm - rt);
        sgn1 = -1;
        rt2 = (acmx / rt1) * acmn - (b / rt1) * b;
    } else if (sm > Real(0)) {
        rt1 = Real(0.5) * (sm + rt);
        sgn1 = 1;
        rt2 = (acmx / rt1) * acmn - (b / rt1) * b;
    } else {
        rt1 = Real(0.5) * rt;
        rt2 = Real(-0.5) * rt;
        sgn1 = 1;
    }
    const int sgn2 = df >= Real(0) ? 1 : -1;
    const Real cs = df >= Real(0) ? df + rt : df - rt;
    Real cs1, sn1;
    if (sycl::fabs(cs) > ab) {
        const Real ct = -tb / cs;
        sn1 = Real(1) / sycl::sqrt(Real(1) + ct * ct);
        cs1 = ct * sn1;
    } else if (ab == Real(0)) {
        cs1 = Real(1);
        sn1 = Real(0);
    } else {
        const Real tn = -cs / tb;
        cs1 = Real(1) / sycl::sqrt(Real(1) + tn * tn);
        sn1 = tn * cs1;
    }
    if (sgn1 == sgn2) {
        const Real tn = cs1;
        cs1 = -sn1;
        sn1 = tn;
    }
    return {rt1, rt2, cs1, sn1};
}

/// @brief SLAS2 / DLAS2: the singular values {ssmin, ssmax} of [[f, g], [0, h]], nothing squared.
/// @ingroup api_device_arith
/// @param f the (1, 1) entry
/// @param g the (1, 2) entry
/// @param h the (2, 2) entry
/// @return {ssmin, ssmax}
template <class Real> BATCHLAS_DEVICE_INLINE std::array<Real, 2> las2(const Real& f, const Real& g, const Real& h) {
    const Real fa = sycl::fabs(f), ga = sycl::fabs(g), ha = sycl::fabs(h);
    const Real fhmn = fa < ha ? fa : ha, fhmx = fa < ha ? ha : fa;
    if (fhmn == Real(0)) {
        if (fhmx == Real(0)) return {Real(0), ga};
        const Real mx = fhmx > ga ? fhmx : ga, mn = fhmx > ga ? ga : fhmx;
        return {Real(0), mx * sycl::sqrt(Real(1) + (mn / mx) * (mn / mx))};
    }
    if (ga < fhmx) {
        const Real as = Real(1) + fhmn / fhmx, at = (fhmx - fhmn) / fhmx, au = (ga / fhmx) * (ga / fhmx);
        const Real c = Real(2) / (sycl::sqrt(as * as + au) + sycl::sqrt(at * at + au));
        return {fhmn * c, fhmx / c};
    }
    const Real au = fhmx / ga;
    if (au == Real(0)) return {(fhmn * fhmx) / ga, ga};
    const Real as = Real(1) + fhmn / fhmx, at = (fhmx - fhmn) / fhmx;
    const Real c = Real(1) / (sycl::sqrt(Real(1) + (as * au) * (as * au)) + sycl::sqrt(Real(1) + (at * au) * (at * au)));
    return {Real(2) * ((fhmn * c) * au), ga / (c + c)};
}

/// @brief dsteqr's Wilkinson shift for the window starting at l: the eigenvalue of
/// [[d(l), e(l)], [e(l), d(l+1)]] nearest d(l).
/// @ingroup api_device_arith
///
/// It is d(l) - e(l) / (g + sign(hypot(g, 1), g)) with g = (d(l+1) - d(l)) / (2 e(l)).
/// @param dl d(l)
/// @param el e(l), nonzero
/// @param dl1 d(l+1)
/// @return the shift
template <class Real> BATCHLAS_DEVICE_INLINE Real lapack_shift(Real dl, Real el, Real dl1) {
    const Real g = (dl1 - dl) / (Real(2) * el);
    const Real r = sycl::hypot(g, Real(1));
    return dl - el / (g + sycl::copysign(r, g));
}

namespace alg {
/// @brief xLARFG under the LAPACK convention: larfg_scalars from (alpha, ScaledSsq).
/// @ingroup api_device_arith
struct LarfgLapack {};
/// @brief xLARFG with beta keeping alpha's phase (complex beta, real tau).
/// @ingroup api_device_arith
struct Phase {};
/// @brief alg::Phase without the len <= 1 identity, as ZLARFG N = 1.
/// @ingroup api_device_arith
struct PhaseN1 {};
/// @brief LAPACK ZLARFG's real beta and complex tau.
/// @ingroup api_device_arith
struct RealBeta {};
}  // namespace alg

/// @brief The scalars of an elementary reflector, as larfg_scalars returns them.
/// @ingroup api_device_arith
///
/// `H^H [alpha; x] = [beta; 0]` with `H = I - tau v v^H` and `v = [1; x * scale]` (use_mul) or
/// `[1; x / scale]` (the reciprocal overflowed or vanished). identity: tau = 0, H = I, x unchanged.
/// @tparam Scalar the scalar type
template <class Scalar> struct LarfgScalars {
    Scalar tau;  ///< The reflector's tau (0 for the identity).
    Scalar beta;  ///< The new alpha, beta.
    Scalar scale;  ///< The multiplier of x (use_mul) or its divisor.
    bool use_mul;  ///< True: v = [1; x * scale]; false: v = [1; x / scale].
    bool identity;  ///< True when H = I: tau is 0 and x is unchanged.
};

/// @brief The reflector scalars of xLARFG from alpha and the scaled sum of squares of [alpha; x]
/// (alg::LarfgLapack).
/// @ingroup api_device_arith
///
/// `q.scale` = max |component| over alpha and x, `q.ssq` = sum |x_i / s|^2 (geqrf's two-pass
/// norm). beta = -sign(re alpha) ||[alpha; x]|| is real and tau complex; scale is Smith's
/// 1 / (alpha - beta), and use_mul holds unless that is not finite or zero.
/// @tparam Conv alg::LarfgLapack
/// @param alpha the first element
/// @param q the ScaledSsq of [alpha; x]
/// @return the LarfgScalars
template <class Conv = alg::LarfgLapack, class Scalar> BATCHLAS_DEVICE_INLINE LarfgScalars<Scalar> larfg_scalars(const Scalar& alpha, ScaledSsq<real_t<Scalar>> q) {
    static_assert(std::is_same_v<Conv, alg::LarfgLapack>, "larfg_scalars(alpha, ScaledSsq): alg::LarfgLapack");
    using A = Arith<Scalar>;
    using R = real_t<Scalar>;
    LarfgScalars<Scalar> out{A::zero(), alpha, A::one(), true, true};
    const R alphi = A::im(alpha);
    if (q.ssq == R(0) && alphi == R(0)) return out;  // a complex alpha still needs a reflector
    if (!(q.scale > R(0))) return out;
    const R alphr = A::re(alpha);
    const R ar = alphr / q.scale, ai = alphi / q.scale;
    const R nrm = sycl::sqrt(sycl::fma(ar, ar, sycl::fma(ai, ai, q.ssq)));
    const R beta_s = (alphr >= R(0)) ? -nrm : nrm;  // -SIGN(nrm, alphr): +0 counts as positive
    out.tau = A::make((beta_s - ar) / beta_s, -ai / beta_s);
    out.beta = A::make(beta_s * q.scale, R(0));
    const Scalar d = A::sub(alpha, out.beta);
    const Scalar r = FmaArith<Scalar>::recip(d);
    out.use_mul = A::isfinite(r) && !A::is_zero(r);
    out.scale = out.use_mul ? r : d;
    out.identity = false;
    return out;
}

/// @brief The reflector scalars of xLARFG from alpha, ||x|| and the reflector length.
/// @ingroup api_device_arith
///
/// - alg::Phase: beta = -sgn(alpha) ||[alpha; x]|| keeps alpha's phase (complex beta, real tau);
///   len <= 1, or a real alpha over x == 0, gives the identity.
/// - alg::PhaseN1: alg::Phase without the len <= 1 identity: a complex alpha reflects at len 1
///   too, as ZLARFG N = 1 (the xLATRD panels). For a real Scalar it is identical to Phase.
/// - alg::RealBeta: LAPACK ZLARFG's real beta and complex tau; a complex alpha reflects even at
///   len == 1. For a real Scalar it is identical to Phase.
///
/// An |alpha - beta| below 1 / realmax (every entry subnormal) gives the identity, as an unscaled
/// ||x|| or a flush-to-zero device does.
/// @tparam Conv alg::Phase, alg::PhaseN1 or alg::RealBeta
/// @param alpha the first element
/// @param xnorm ||x||
/// @param len the reflector's length, including alpha
/// @return the LarfgScalars
template <class Conv, class Scalar> BATCHLAS_DEVICE_HELPER LarfgScalars<Scalar> larfg_scalars(const Scalar& alpha, real_t<Scalar> xnorm, int len) {
    static_assert(std::is_same_v<Conv, alg::Phase> || std::is_same_v<Conv, alg::RealBeta>,
                  "larfg_scalars(alpha, xnorm, len): alg::Phase or alg::RealBeta");
    using R = real_t<Scalar>;
    LarfgScalars<Scalar> out{Scalar(0), alpha, Scalar(0), true, true};
    if constexpr (std::is_same_v<Conv, alg::RealBeta> && Arith<Scalar>::is_complex) {
        if (len < 1 || (xnorm == R(0) && alpha.imag() == R(0))) return out;
        const R norm = sycl::hypot(sycl::hypot(alpha.real(), alpha.imag()), xnorm);
        if (norm == R(0)) return out;
        const R beta = sycl::signbit(alpha.real()) ? norm : -norm;
        out.beta = Scalar(beta);
        out.tau = Scalar((beta - alpha.real()) / beta, -alpha.imag() / beta);
    } else if constexpr (Arith<Scalar>::is_complex) {
        if (len <= 1) return out;
        if (xnorm == R(0) && alpha.imag() == R(0)) return out;
        const R alpha_abs = sycl::hypot(alpha.real(), alpha.imag());
        const R beta_abs = sycl::hypot(alpha_abs, xnorm);
        const Scalar alpha_sign = (alpha_abs == R(0)) ? Scalar(1) : (alpha / alpha_abs);
        out.beta = -alpha_sign * Scalar(beta_abs);
        out.tau = (out.beta - alpha) / out.beta;
    } else {
        if (len <= 1 || xnorm == R(0)) return out;
        const R sign = sycl::signbit(alpha) ? R(-1) : R(1);
        out.beta = Scalar(-sign * sycl::hypot(alpha, xnorm));
        out.tau = (out.beta - alpha) / out.beta;
    }
    out.scale = Scalar(1) / (alpha - out.beta);
    // |alpha - beta| below 1 / realmax (every entry subnormal) has no finite reciprocal, and
    // x * scale would be inf or NaN. Identity, as an unscaled ||x|| (x^2 underflows to 0) or a
    // flush-to-zero device gives: a scaled ||x|| keeps such a column's norm nonzero.
    if (!Arith<Scalar>::isfinite(out.scale)) return {Scalar(0), alpha, Scalar(0), true, true};
    out.identity = false;
    return out;
}
/// @cond
template <class Conv, class Scalar>
    requires std::is_same_v<Conv, alg::PhaseN1>
BATCHLAS_DEVICE_HELPER LarfgScalars<Scalar> larfg_scalars(const Scalar& alpha, real_t<Scalar> xnorm, int) {
    return larfg_scalars<alg::Phase>(alpha, xnorm, 2);  // len >= 2: no len <= 1 identity
}
/// @endcond

/// @brief dlamch: the machine constant C of Real.
/// @ingroup api_device_arith
///
/// 'E' eps (relative machine precision: half of numeric_limits' epsilon, as LAPACK 3.x with
/// rounding), 'S' safe minimum, 'P' eps * base, 'B' base, 'N' digits, 'U' underflow threshold (the
/// smallest normal), 'O' overflow threshold (the largest), 'R' 1 (rounding mode).
/// @tparam C one of E S P B N U O R
/// @tparam Real float or double
/// @return the constant
template <char C, class Real> constexpr Real lamch() {
    constexpr Real eps = std::numeric_limits<Real>::epsilon() * Real(0.5);
    if constexpr (C == 'E') return eps;
    else if constexpr (C == 'S') return safmin<Real>();
    else if constexpr (C == 'P') return eps * Real(std::numeric_limits<Real>::radix);
    else if constexpr (C == 'B') return Real(std::numeric_limits<Real>::radix);
    else if constexpr (C == 'N') return Real(std::numeric_limits<Real>::digits);
    else if constexpr (C == 'U') return std::numeric_limits<Real>::min();
    else if constexpr (C == 'O') return std::numeric_limits<Real>::max();
    else {
        static_assert(C == 'R', "lamch: one of E S P B N U O R");
        return Real(1);
    }
}

namespace detail {
// dsteqr's SSFMIN / SSFMAX (the range dlascl scales a block into) and EPS2 = eps^2.
template <class Real> inline constexpr Real ssfmin() {
    return std::sqrt(safmin<Real>()) / (lamch<'P', Real>() * lamch<'P', Real>());
}
template <class Real> inline constexpr Real ssfmax() { return std::sqrt(safmax<Real>()) / Real(3.0); }
template <class Real> inline constexpr Real eps2() { return lamch<'E', Real>() * lamch<'E', Real>(); }
}  // namespace detail

namespace alg {
/// @brief inv = 1 / s (the CTA solvers).
/// @ingroup api_device_arith
struct UnscaleRecip {};
/// @brief inv = anorm / bound (dsteqr's own dlascl(bound, anorm)).
/// @ingroup api_device_arith
struct UnscaleRatio {};
}  // namespace alg
/// @brief dsteqr's dlascl decision for a block of max-norm anorm.
/// @ingroup api_device_arith
///
/// s brings the block into [ssfmin, ssfmax] (ssfmax / anorm above, ssfmin / anorm below, 1 inside
/// or at anorm == 0) and inv undoes it, as alg::UnscaleRecip or alg::UnscaleRatio says; inside the
/// range both are exactly 1.
/// @tparam Real the real type
template <class Real> struct BlockScale {
    Real s;  ///< The scale factor.
    Real inv;  ///< The factor that undoes it.
};
namespace detail {
template <class Unscale = alg::UnscaleRecip, class Real> BATCHLAS_DEVICE_INLINE BlockScale<Real> block_scale(Real anorm) {
    static_assert(std::is_same_v<Unscale, alg::UnscaleRecip> || std::is_same_v<Unscale, alg::UnscaleRatio>,
                  "block_scale: alg::UnscaleRecip or alg::UnscaleRatio");
    Real s = Real(1), inv = Real(1);
    if (anorm > ssfmax<Real>()) {
        s = ssfmax<Real>() / anorm;
        if constexpr (std::is_same_v<Unscale, alg::UnscaleRatio>) inv = anorm / ssfmax<Real>();
    } else if (anorm < ssfmin<Real>() && anorm != Real(0)) {
        // Below ssfmin * min (float 9e-44, a subnormal block: stedc's graded float leaves) ssfmin /
        // anorm overflows and the sweep made NaN. s stops at 1 / min, exact both ways, which leaves
        // the block at >= 2^-23 (float): normal, and far above sqrt(safmin).
        constexpr Real cap = Real(1) / std::numeric_limits<Real>::min();
        s = ssfmin<Real>() / anorm;
        if (!(s <= cap)) {
            s = cap;
            inv = std::numeric_limits<Real>::min();
        } else if constexpr (std::is_same_v<Unscale, alg::UnscaleRatio>) {
            inv = anorm / ssfmin<Real>();
        }
    }
    if constexpr (std::is_same_v<Unscale, alg::UnscaleRecip>) inv = Real(1) / s;
    return {s, inv};
}
}  // namespace detail

/// @brief A quiet NaN of Scalar; both parts are NaN for a complex Scalar.
/// @ingroup api_device_arith
/// @return the NaN
template <class Scalar> BATCHLAS_DEVICE_INLINE Scalar qnan() {
    using A = Arith<Scalar>;
    using R = typename A::real_t;
    return A::make(std::numeric_limits<R>::quiet_NaN(), std::numeric_limits<R>::quiet_NaN());
}

/// @brief x / |x|, and 1 at x == 0: a unit-modulus sign.
/// @ingroup api_device_arith
///
/// It is the phase a Hermitian tridiagonal's off-diagonal is rotated by. For a real Scalar it is
/// copysign(1, x), with +1 at +-0.
/// @param x the value
/// @return the phase
template <class Scalar> BATCHLAS_DEVICE_INLINE Scalar phase(Scalar x) {
    using A = Arith<Scalar>;
    using R = typename A::real_t;
    if constexpr (A::is_complex) {
        const R m = sycl::hypot(A::re(x), A::im(x));
        if (m == R(0)) return A::one();
        return A::make(A::re(x) / m, A::im(x) / m);
    } else {
        return x == R(0) ? R(1) : sycl::copysign(R(1), x);
    }
}
/// @brief sign(re x) as +-1 with +1 at 0: LAPACK's SIGN(ONE, x) on a real part.
/// @ingroup api_device_arith
/// @param x the value
/// @return +1 or -1
template <class Scalar> BATCHLAS_DEVICE_INLINE real_t<Scalar> sign_nonzero_real(Scalar x) {
    using R = real_t<Scalar>;
    const R r = Arith<Scalar>::re(x);
    return r < R(0) ? R(-1) : R(1);
}

/// @brief The eigenvalue of [[a, b], [b, c]] nearer to c: dsteqr's shift target.
/// @ingroup api_device_arith
///
/// It is rt1 or rt2 of lae2, whichever is closer to c.
/// @tparam Tie the tie rule: alg::TieLowest (rt1, the open-coded steqr form) or alg::TieHighest
///         (rt2, the CTA Wilkinson strategy)
/// @param a the (1, 1) entry
/// @param b the off-diagonal entry
/// @param c the (2, 2) entry
/// @return the nearer eigenvalue
// Plain inline, operands by reference: the forms it replaces.
template <class Tie = alg::TieLowest, class Real> BATCHLAS_DEVICE_HELPER Real eig2x2_nearest(const Real& a, const Real& b, const Real& c) {
    static_assert(std::is_same_v<Tie, alg::TieLowest> || std::is_same_v<Tie, alg::TieHighest>,
                  "eig2x2_nearest: alg::TieLowest or alg::TieHighest");
    const auto [rt1, rt2] = lae2(a, b, c);
    if constexpr (std::is_same_v<Tie, alg::TieLowest>) return sycl::fabs(rt1 - c) <= sycl::fabs(rt2 - c) ? rt1 : rt2;
    else return sycl::fabs(rt1 - c) < sycl::fabs(rt2 - c) ? rt1 : rt2;
}

/// @brief The tridiagonal solver's 2-vector Householder: v with (I - 2 v v^T) (a0, a1)^T =
/// (-sign(a0) ||a||, 0)^T.
/// @ingroup api_device_arith
///
/// Apply it with `rot<Reflector2>`. The norms are unscaled: the inputs are O(||T||).
/// @param a0 the first component
/// @param a1 the second component
/// @return {v0, v1}
template <class Real> BATCHLAS_DEVICE_INLINE std::array<Real, 2> house2(Real a0, Real a1) {
    const Real anorm = sycl::sqrt(a0 * a0 + a1 * a1);
    const Real alpha = -sycl::copysign(anorm, a0);
    const Real v0 = a0 - alpha, v1 = a1;
    const Real vnorm = sycl::sqrt(v0 * v0 + v1 * v1);
    const Real inv = Real(1) / vnorm;
    return {v0 * inv, v1 * inv};
}

/// @brief The edge behaviour of inv_sqrt.
/// @ingroup api_device_arith
///
/// - Lapack: 1 / inf = 0, not NaN.
/// - FiniteGeOne: a finite a >= 1, no guard (the Jacobi cosine 1 + t^2).
/// - Exact: 1 / sqrt_a on every target.
enum class InvSqrtEdge { Lapack, FiniteGeOne, Exact };

/// @brief 1 / sqrt(a), where sqrt_a = sqrt(a) is the divisor for callers that hold it.
/// @ingroup api_device_arith
///
/// On NVPTX float (not AdaptiveCpp) it is rsqrt.approx and one FMA Newton step, y + (y / 2)(1 -
/// (a y) y): near correctly rounded, and the same whatever is inlined. On AdaptiveCpp float with
/// FiniteGeOne it is the same Newton step from 1 / sqrt_a. Everywhere else, and with Exact, it is
/// 1 / sqrt_a.
/// @tparam E the edge behaviour
/// @param a the argument
/// @param sqrt_a sqrt(a)
/// @return 1 / sqrt(a)
// A plain 1 / sqrt(a) folds into rsqrt.approx by code shape (decision 10).
template <InvSqrtEdge E = InvSqrtEdge::Lapack, class Real> BATCHLAS_DEVICE_HELPER Real inv_sqrt(Real a, Real sqrt_a) {
#if !BATCHLAS_SYCL_IMPL_ACPP && defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
    if constexpr (std::is_same_v<Real, float> && E != InvSqrtEdge::Exact) {
        float y;
        asm("rsqrt.approx.f32 %0, %1;" : "=f"(y) : "f"(a));
        const float e = sycl::fma(-(a * y), y, 1.0f);
        const float r = sycl::fma(0.5f * y, e, y);
        if constexpr (E == InvSqrtEdge::Lapack) return sycl::isinf(a) ? 0.0f : r;
        else return r;
    }
#elif BATCHLAS_SYCL_IMPL_ACPP
    // AdaptiveCpp float FiniteGeOne: the same FMA Newton step, from 1 / sqrt_a (two roundings, and
    // Metal's divide and sqrt are not correctly rounded), so the Jacobi cosine is near correctly
    // rounded as on NVPTX (n = 32 graded ||Z^H Z - I||: 252 eps without it on apple_m5_max, bound 180).
    if constexpr (std::is_same_v<Real, float> && E == InvSqrtEdge::FiniteGeOne) {
        const float y = 1.0f / sqrt_a;
        const float e = sycl::fma(-(a * y), y, 1.0f);
        return sycl::fma(0.5f * y, e, y);
    }
#endif
    (void)a;
    return Real(1) / sqrt_a;
}
/// @brief 1 / sqrt(a), as `inv_sqrt<E>(a, sqrt(a))`.
/// @ingroup api_device_arith
/// @tparam E the edge behaviour
/// @param a the argument
/// @return 1 / sqrt(a)
template <InvSqrtEdge E = InvSqrtEdge::Lapack, class Real> BATCHLAS_DEVICE_HELPER Real inv_sqrt(Real a) { return inv_sqrt<E>(a, sycl::sqrt(a)); }

/// @brief A safe reciprocal: multiply by it when it is finite and nonzero, else divide, else (an
/// exact zero d) take the caller's value.
/// @ingroup api_device_arith
///
/// make() decides once per pivot; apply() branches on the uniform flags, never a select (a select
/// pays the divide every time).
/// @tparam Scalar the scalar type
/// @tparam Ops the arithmetic of the reciprocal, the product and the quotient
// evidence: docs/perf/lu.md#the-rhs-pad-column-cost
template <class Scalar, class Ops = arith::Fma> struct SafeRecip {
    Scalar d;  ///< The divisor.
    Scalar r;  ///< Its reciprocal.
    bool zero;  ///< True when d is exactly zero.
    bool use_mul;  ///< True when the reciprocal is finite and nonzero.
    /// @brief The decision for divisor d.
    /// @param d the divisor
    /// @return the SafeRecip
    BATCHLAS_DEVICE_INLINE static SafeRecip make(Scalar d) {
        using Ar = arith_t<Ops, Scalar>;
        const bool z = Arith<Scalar>::is_zero(d);  // EXACT zero, no epsilon
        const Scalar rc = Ar::recip(d);
        return {d, rc, z, !z && Arith<Scalar>::isfinite(rc) && !Arith<Scalar>::is_zero(rc)};
    }
    /// @brief y / d: y * r, or y / d, or if_zero when d is exactly zero.
    /// @param y the dividend
    /// @param if_zero the result for an exactly zero d
    /// @return the quotient
    BATCHLAS_DEVICE_INLINE Scalar apply(Scalar y, Scalar if_zero) const {
        using Ar = arith_t<Ops, Scalar>;
        Scalar x = if_zero;
        if (use_mul) x = Ar::mul(y, r);
        else if (!zero) x = Ar::div(y, d);
        return x;
    }
};

}  // namespace batchlas::device
