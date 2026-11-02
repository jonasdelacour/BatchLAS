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

#include <array>
#include <cmath>
#include <complex>
#include <limits>
#include <type_traits>

#include <batchlas/blas/device/v2/core.hh>

namespace batchlas::device::v2 {

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

// W: the working precision of the complex branch (void: real_t<T>); the real branch is
// SLARTG/DLARTG with its unscaled fast path. Bit-identical to src/math-helpers.hh lartg.
template <class W = void, class T> inline LartgResult<T> lartg(const T& f, const T& g) {
    using R = real_t<T>;
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

}  // namespace batchlas::device::v2
