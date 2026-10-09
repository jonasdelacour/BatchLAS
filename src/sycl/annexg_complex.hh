#pragma once
// Force-included (-include) into every acpp TU: device definitions of the C99 Annex G libcalls
// clang emits for std::complex * and /, absent from the SSCP device library. Each TU needs its own
// weak hidden copy: SSCP never imports a cross-TU symbol whose name starts with "__".
// Ported from DPC++'s __sycl_complex_impl.hpp = LLVM compiler-rt (Apache-2.0 WITH LLVM-exception).
#include <sycl/sycl.hpp>

namespace batchlas_annexg {
template <typename T> inline T cpsgn(T m, T s) { return __builtin_copysign(m, s); }
inline float cpsgn(float m, float s) { return __builtin_copysignf(m, s); }
template <typename T> inline bool isnan_(T x) { return __builtin_isnan(x); }
template <typename T> inline bool isinf_(T x) { return __builtin_isinf(x); }
template <typename T> inline bool isfinite_(T x) { return __builtin_isfinite(x); }

template <typename T>
inline void mul(T a, T b, T c, T d, T& zr, T& zi) {
    const T ac = a * c, bd = b * d, ad = a * d, bc = b * c;
    zr = ac - bd;
    zi = ad + bc;
    if (isnan_(zr) && isnan_(zi)) {
        bool recalc = false;
        if (isinf_(a) || isinf_(b)) {
            a = cpsgn(isinf_(a) ? T(1) : T(0), a);
            b = cpsgn(isinf_(b) ? T(1) : T(0), b);
            if (isnan_(c)) c = cpsgn(T(0), c);
            if (isnan_(d)) d = cpsgn(T(0), d);
            recalc = true;
        }
        if (isinf_(c) || isinf_(d)) {
            c = cpsgn(isinf_(c) ? T(1) : T(0), c);
            d = cpsgn(isinf_(d) ? T(1) : T(0), d);
            if (isnan_(a)) a = cpsgn(T(0), a);
            if (isnan_(b)) b = cpsgn(T(0), b);
            recalc = true;
        }
        if (!recalc && (isinf_(ac) || isinf_(bd) || isinf_(ad) || isinf_(bc))) {
            if (isnan_(a)) a = cpsgn(T(0), a);
            if (isnan_(b)) b = cpsgn(T(0), b);
            if (isnan_(c)) c = cpsgn(T(0), c);
            if (isnan_(d)) d = cpsgn(T(0), d);
            recalc = true;
        }
        if (recalc) {
            const T inf = __builtin_inff();
            zr = inf * (a * c - b * d);
            zi = inf * (a * d + b * c);
        }
    }
}

template <typename T>
inline void div(T a, T b, T c, T d, T& zr, T& zi) {
    int ilogbw = 0;
    const T logbw = sycl::logb(sycl::fmax(sycl::fabs(c), sycl::fabs(d)));
    if (isfinite_(logbw)) {
        ilogbw = static_cast<int>(logbw);
        c = sycl::ldexp(c, -ilogbw);
        d = sycl::ldexp(d, -ilogbw);
    }
    const T denom = c * c + d * d;
    zr = sycl::ldexp((a * c + b * d) / denom, -ilogbw);
    zi = sycl::ldexp((b * c - a * d) / denom, -ilogbw);
    if (isnan_(zr) && isnan_(zi)) {
        const T inf = __builtin_inff();
        if (denom == T(0) && (!isnan_(a) || !isnan_(b))) {
            zr = cpsgn(inf, c) * a;
            zi = cpsgn(inf, c) * b;
        } else if ((isinf_(a) || isinf_(b)) && isfinite_(c) && isfinite_(d)) {
            a = cpsgn(isinf_(a) ? T(1) : T(0), a);
            b = cpsgn(isinf_(b) ? T(1) : T(0), b);
            zr = inf * (a * c + b * d);
            zi = inf * (b * c - a * d);
        } else if (isinf_(logbw) && logbw > T(0) && isfinite_(a) && isfinite_(b)) {
            c = cpsgn(isinf_(c) ? T(1) : T(0), c);
            d = cpsgn(isinf_(d) ? T(1) : T(0), d);
            zr = T(0) * (a * c + b * d);
            zi = T(0) * (b * c - a * d);
        }
    }
}

template <typename T, typename C, void (*Op)(T, T, T, T, T&, T&)>
inline C apply(T a, T b, T c, T d) {
    C z;
    T r, i;
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
