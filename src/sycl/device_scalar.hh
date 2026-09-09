#pragma once

// The POD device scalar, shared by every SYCL kernel in this directory.
//
// WHY IT EXISTS. std::complex must never reach device code: its operator* is
// Annex-G conformant, which means an isnan branch and a call to __mulsc3 /
// __muldc3 in the inner loop. The fix is to re-type to a plain aggregate at the
// POINTER BOUNDARY in the launcher -- operands and scalars alike -- so no
// std::complex crosses into the kernel body. Verified in the PTX of the GEMM
// instantiations that use this: zero __mulsc3, zero __muldc3, zero call.uni.
//
// These types started life inside src/sycl/gemm/register_64x64_k16_wide.hh.
// They were lifted here when TRSM needed them, rather than having a TRSM
// translation unit include a GEMM *kernel* header to get 25 lines of type
// plumbing. The GEMM header now includes this one and aliases the names into
// its own namespace, so no GEMM code changed; the move was verified by
// re-running scripts/register_probe.sh and confirming the wide-scalar kernels
// still report 56 / 76 / 80 / 132 registers with zero spill.
//
// TRSM ADDED THE ARITHMETIC GEMM DID NOT NEED. GEMM multiplies and accumulates;
// a triangular solve must also DIVIDE, conjugate, and test for finiteness.
// Those are at the bottom of this file, and the division in particular is not
// the textbook formula -- see the note there.

#include <sycl/sycl.hpp>

#include <complex>
#include <type_traits>

namespace batchlas::sycl_device {

// A plain aggregate complex. Layout-compatible with std::complex, which is what
// lets the launcher reinterpret_cast at the boundary.
template <typename R>
struct Cx {
    R re;
    R im;
};

template <typename T>
struct DevMap {
    using type = T;
    using real = T;
    static constexpr bool is_complex = false;
};

template <typename R>
struct DevMap<std::complex<R>> {
    using type = Cx<R>;
    using real = R;
    static constexpr bool is_complex = true;
};

// The same question asked of the DEVICE type rather than the source type.
// DevMap<T>::is_complex keys on T, which a kernel body templated on D cannot
// see; and sizeof is no substitute, since Cx<float> and double are both 8
// bytes. Callers that must branch on "is this scalar two components" need this.
template <typename D> struct IsDevComplex           : std::false_type {};
template <typename R> struct IsDevComplex<Cx<R>>    : std::true_type  {};
template <typename D> inline constexpr bool dev_is_complex_v = IsDevComplex<D>::value;

static_assert(sizeof(Cx<float>) == sizeof(std::complex<float>), "layout");
static_assert(sizeof(Cx<double>) == sizeof(std::complex<double>), "layout");
static_assert(alignof(Cx<float>) == alignof(std::complex<float>), "layout");
static_assert(alignof(Cx<double>) == alignof(std::complex<double>), "layout");

// --- zero test -------------------------------------------------------------

template <typename R>
inline bool dev_is_zero(R x) {
    return x == R(0);
}
template <typename R>
inline bool dev_is_zero(Cx<R> x) {
    return x.re == R(0) && x.im == R(0);
}

// --- multiply-accumulate, written out --------------------------------------
// Real is one FMA; complex is four, with no branches and no libcall.

inline void fma_acc(float& acc, float a, float b) { acc = sycl::fma(a, b, acc); }
inline void fma_acc(double& acc, double a, double b) { acc = sycl::fma(a, b, acc); }

template <typename R>
inline void fma_acc(Cx<R>& acc, Cx<R> a, Cx<R> b) {
    acc.re = sycl::fma(a.re, b.re, acc.re);
    acc.re = sycl::fma(-a.im, b.im, acc.re);
    acc.im = sycl::fma(a.re, b.im, acc.im);
    acc.im = sycl::fma(a.im, b.re, acc.im);
}

// ===========================================================================
// The arithmetic a triangular solve needs and a GEMM does not.
// ===========================================================================

// --- construction and conjugation ------------------------------------------

template <typename D>
inline D dev_one() {
    if constexpr (std::is_same_v<D, float> || std::is_same_v<D, double>) {
        return D(1);
    } else {
        return D{typename std::remove_reference_t<decltype(D{}.re)>(1),
                 typename std::remove_reference_t<decltype(D{}.re)>(0)};
    }
}

inline float dev_conj(float x) { return x; }
inline double dev_conj(double x) { return x; }

template <typename R>
inline Cx<R> dev_conj(Cx<R> x) {
    return Cx<R>{x.re, -x.im};
}

// --- plain multiply and subtract -------------------------------------------

inline float dev_mul(float a, float b) { return a * b; }
inline double dev_mul(double a, double b) { return a * b; }

template <typename R>
inline Cx<R> dev_mul(Cx<R> a, Cx<R> b) {
    // Written out for the same reason fma_acc is: keep Annex-G out of it.
    return Cx<R>{sycl::fma(a.re, b.re, -a.im * b.im),
                 sycl::fma(a.re, b.im, a.im * b.re)};
}

inline float dev_sub(float a, float b) { return a - b; }
inline double dev_sub(double a, double b) { return a - b; }

template <typename R>
inline Cx<R> dev_sub(Cx<R> a, Cx<R> b) {
    return Cx<R>{a.re - b.re, a.im - b.im};
}

// --- finiteness ------------------------------------------------------------
// Both components must be finite. They can go non-finite independently, so
// testing one is not testing the value.

inline bool dev_isfinite(float x) { return sycl::isfinite(x); }
inline bool dev_isfinite(double x) { return sycl::isfinite(x); }

template <typename R>
inline bool dev_isfinite(Cx<R> x) {
    return sycl::isfinite(x.re) && sycl::isfinite(x.im);
}

// --- division and reciprocal -----------------------------------------------
//
// NOT the textbook 1/(c+di) = (c - di)/(c^2 + d^2). That squares the operands,
// so it overflows to infinity for any |c| or |d| above about 1e19 in float or
// 1e154 in double -- and the result is then 0, silently, for an input that is
// perfectly representable and whose true reciprocal is also representable.
// Underflow at the small end loses the value the same way.
//
// This is SMITH'S ALGORITHM: divide through by the larger component first, so
// nothing larger than max(|c|,|d|) is ever squared. Verified against exact
// arithmetic including at 1e200, where the textbook form returns 0 and this
// returns the correct 5e-201.

template <typename R>
inline Cx<R> dev_recip(Cx<R> d) {
    if (sycl::fabs(d.re) >= sycl::fabs(d.im)) {
        const R r = d.im / d.re;
        const R den = sycl::fma(d.im, r, d.re);
        return Cx<R>{R(1) / den, -r / den};
    }
    const R r = d.re / d.im;
    const R den = sycl::fma(d.re, r, d.im);
    return Cx<R>{r / den, R(-1) / den};
}

inline float dev_recip(float d) { return 1.0f / d; }
inline double dev_recip(double d) { return 1.0 / d; }

template <typename R>
inline Cx<R> dev_div(Cx<R> a, Cx<R> b) {
    if (sycl::fabs(b.re) >= sycl::fabs(b.im)) {
        const R r = b.im / b.re;
        const R den = sycl::fma(b.im, r, b.re);
        return Cx<R>{sycl::fma(a.im, r, a.re) / den, sycl::fma(-a.re, r, a.im) / den};
    }
    const R r = b.re / b.im;
    const R den = sycl::fma(b.re, r, b.im);
    return Cx<R>{sycl::fma(a.re, r, a.im) / den, sycl::fma(a.im, r, -a.re) / den};
}

inline float dev_div(float a, float b) { return a / b; }
inline double dev_div(double a, double b) { return a / b; }

// --- real component, real construction, real scaling, real division --------
//
// POTRF needs these and a GEMM does not: a Cholesky diagonal is REAL by
// construction (it is a sqrt of a real), so scaling and dividing by it must not
// go through the complex paths. dev_div(a, Cx{d,0}) would run Smith's algorithm
// -- three divisions and two fmas to compute what is two divisions -- and
// dev_mul(a, Cx{s,0}) is four fmas for two multiplies. They are also the
// shared spelling of a `real_part` that exists PRIVATELY in at least eight
// translation units in this tree (ritz_values.cc:67, syev_jacobi_cta.cc:85,
// syev_cta_fused.cc:80, ortho.cc:191, sytrd_sb2st.cc:97, lanczos.cc:46,
// band_reduction.cc:41, sytrd_sb2st_cta.cc:98); potrf is not adding a ninth.
//
// dev_div_real is a DIVISION and dev_mul_real is a RECIPROCAL-MULTIPLY, and
// that asymmetry is deliberate, not an oversight: reference ?trsm divides
// (B(i,j)/A(j,j)) while reference ?potf2 scales by a precomputed reciprocal
// (sscal(1/ajj, ...)). potrf's (P2) panel solve is the trsm and (P1)'s column
// scale is the potf2. Unifying them would change the rounding of one of the
// two away from its LAPACK reference.

inline float  dev_real(float x)  { return x; }
inline double dev_real(double x) { return x; }
template <typename R>
inline R dev_real(Cx<R> x) { return x.re; }

template <typename D, typename R>
inline D dev_from_real(R x) {
    if constexpr (std::is_same_v<D, R>) {
        return x;
    } else {
        return D{x, R(0)};
    }
}

inline float  dev_mul_real(float a, float s)   { return a * s; }
inline double dev_mul_real(double a, double s) { return a * s; }
template <typename R>
inline Cx<R> dev_mul_real(Cx<R> a, R s) { return Cx<R>{a.re * s, a.im * s}; }

inline float  dev_div_real(float a, float d)   { return a / d; }
inline double dev_div_real(double a, double d) { return a / d; }
template <typename R>
inline Cx<R> dev_div_real(Cx<R> a, R d) { return Cx<R>{a.re / d, a.im / d}; }

}  // namespace batchlas::sycl_device
