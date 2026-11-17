// SPDX-License-Identifier: MIT
#pragma once
// Force-included (-include) into every acpp TU: device definitions of the C99 Annex G libcalls
// clang emits for std::complex * and /, absent from the SSCP device library. Each TU needs its own
// weak hidden copy: SSCP never imports a cross-TU symbol whose name starts with "__".
// Ported from DPC++'s __sycl_complex_impl.hpp = LLVM compiler-rt (Apache-2.0 WITH LLVM-exception).
#include <sycl/sycl.hpp>

namespace batchlas_annexg {
template <typename Real> inline Real cpsgn(Real m, Real s) { return __builtin_copysign(m, s); }
inline float cpsgn(float m, float s) { return __builtin_copysignf(m, s); }
template <typename Real> inline bool isnan_(Real x) { return __builtin_isnan(x); }
template <typename Real> inline bool isinf_(Real x) { return __builtin_isinf(x); }
template <typename Real> inline bool isfinite_(Real x) { return __builtin_isfinite(x); }

template <typename Real>
inline void mul(Real a, Real b, Real c, Real d, Real& zr, Real& zi) {
    const Real ac = a * c, bd = b * d, ad = a * d, bc = b * c;
    zr = ac - bd;
    zi = ad + bc;
    if (isnan_(zr) && isnan_(zi)) {
        bool recalc = false;
        if (isinf_(a) || isinf_(b)) {
            a = cpsgn(isinf_(a) ? Real(1) : Real(0), a);
            b = cpsgn(isinf_(b) ? Real(1) : Real(0), b);
            if (isnan_(c)) c = cpsgn(Real(0), c);
            if (isnan_(d)) d = cpsgn(Real(0), d);
            recalc = true;
        }
        if (isinf_(c) || isinf_(d)) {
            c = cpsgn(isinf_(c) ? Real(1) : Real(0), c);
            d = cpsgn(isinf_(d) ? Real(1) : Real(0), d);
            if (isnan_(a)) a = cpsgn(Real(0), a);
            if (isnan_(b)) b = cpsgn(Real(0), b);
            recalc = true;
        }
        if (!recalc && (isinf_(ac) || isinf_(bd) || isinf_(ad) || isinf_(bc))) {
            if (isnan_(a)) a = cpsgn(Real(0), a);
            if (isnan_(b)) b = cpsgn(Real(0), b);
            if (isnan_(c)) c = cpsgn(Real(0), c);
            if (isnan_(d)) d = cpsgn(Real(0), d);
            recalc = true;
        }
        if (recalc) {
            const Real inf = __builtin_inff();
            zr = inf * (a * c - b * d);
            zi = inf * (a * d + b * c);
        }
    }
}

template <typename Real>
inline void div(Real a, Real b, Real c, Real d, Real& zr, Real& zi) {
    int ilogbw = 0;
    const Real logbw = sycl::logb(sycl::fmax(sycl::fabs(c), sycl::fabs(d)));
    if (isfinite_(logbw)) {
        ilogbw = static_cast<int>(logbw);
        c = sycl::ldexp(c, -ilogbw);
        d = sycl::ldexp(d, -ilogbw);
    }
    const Real denom = c * c + d * d;
    zr = sycl::ldexp((a * c + b * d) / denom, -ilogbw);
    zi = sycl::ldexp((b * c - a * d) / denom, -ilogbw);
    if (isnan_(zr) && isnan_(zi)) {
        const Real inf = __builtin_inff();
        if (denom == Real(0) && (!isnan_(a) || !isnan_(b))) {
            zr = cpsgn(inf, c) * a;
            zi = cpsgn(inf, c) * b;
        } else if ((isinf_(a) || isinf_(b)) && isfinite_(c) && isfinite_(d)) {
            a = cpsgn(isinf_(a) ? Real(1) : Real(0), a);
            b = cpsgn(isinf_(b) ? Real(1) : Real(0), b);
            zr = inf * (a * c + b * d);
            zi = inf * (b * c - a * d);
        } else if (isinf_(logbw) && logbw > Real(0) && isfinite_(a) && isfinite_(b)) {
            c = cpsgn(isinf_(c) ? Real(1) : Real(0), c);
            d = cpsgn(isinf_(d) ? Real(1) : Real(0), d);
            zr = Real(0) * (a * c + b * d);
            zi = Real(0) * (b * c - a * d);
        }
    }
}

template <typename Real, typename C, void (*Op)(Real, Real, Real, Real, Real&, Real&)>
inline C apply(Real a, Real b, Real c, Real d) {
    C z;
    Real r, i;
    Op(a, b, c, d, r, i);
    __real__ z = r;
    __imag__ z = i;
    return z;
}
}  // namespace batchlas_annexg

#define BATCHLAS_ANNEXG_DEF(name, T, op) \
    extern "C" __attribute__((weak, visibility("hidden"))) T _Complex name(T a, T b, T c, T d) { \
        return batchlas_annexg::apply<T, T _Complex, batchlas_annexg::op<T>>(a, b, c, d); \
    }
BATCHLAS_ANNEXG_DEF(__mulsc3, float, mul)
BATCHLAS_ANNEXG_DEF(__muldc3, double, mul)
BATCHLAS_ANNEXG_DEF(__divsc3, float, div)
BATCHLAS_ANNEXG_DEF(__divdc3, double, div)
#undef BATCHLAS_ANNEXG_DEF
