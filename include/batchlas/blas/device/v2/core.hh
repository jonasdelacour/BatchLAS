// SPDX-License-Identifier: MIT
#pragma once
// Device LA v2, core vocabulary: extents, memory tags, scalar tags, the arithmetic trait and
// the algorithm tags. Every choice a kernel's SASS depends on is a type here, never a runtime
// value.

#include <complex>
#include <type_traits>

#include <sycl/sycl.hpp>

#include <batchlas/blas/enums.hh>

// Inline policy (synthesis 2.9): an op takes the attribute of the code it replaces.
//   BATCHLAS_DLA_INLINE  always_inline: leaf arithmetic, accessors, ops replacing expression text
//   BATCHLAS_DLA_HELPER  plain inline: ops replacing a plain-inline hand helper (phases,
//                        reductions, bcast, epilogues); LLVM unrolls the caller's loop while
//                        the call is still there, as it did for the hand helper
//   BATCHLAS_DLA_BODY    on a lambda passed to a loop op, `[&](int i) BATCHLAS_DLA_BODY {...}`:
//                        a lambda body is otherwise optimised alone (LFTR, IV canonicalisation)
// Each is overridable (-D) for a backend whose inliner needs another spelling.
#ifndef BATCHLAS_DLA_INLINE
#define BATCHLAS_DLA_INLINE __attribute__((always_inline)) inline
#endif
#ifndef BATCHLAS_DLA_HELPER
#define BATCHLAS_DLA_HELPER inline
#endif
#ifndef BATCHLAS_DLA_BODY
#define BATCHLAS_DLA_BODY __attribute__((always_inline))
#endif

namespace batchlas::device::v2 {

// ---- loop policy --------------------------------------------------------------------------
// The unroll policy of a v2-owned loop, emitted as loop metadata: Default adds none (LLVM and
// ptxas decide, the hand loop's behaviour); None is llvm.loop.unroll.disable (PTX .pragma
// "nounroll"); Count<N> is unroll_count(N); Full is unroll(full) (a compile-time trip count).
namespace unroll {
struct Default {};
struct None {};
template <int N> struct Count {
    static constexpr int count = N;
};
struct Full {};
}  // namespace unroll

// Elements [lo, hi) in the parent's coordinates (an op on a Range owns i - lo mod size).
struct Range {
    int lo;
    int hi;
};

// ---- extents ---------------------------------------------------------------------------
template <int N> struct Ext {
    static constexpr int value = N;
    BATCHLAS_DLA_INLINE constexpr operator int() const { return N; }
};
struct Dyn {
    int value;
    BATCHLAS_DLA_INLINE constexpr operator int() const { return value; }
};
using Unit = Ext<1>;

enum class Mem { Global, Local, Reg };
enum class Scope { Auto, Sub, Group };

// ---- layouts: who owns which element -------------------------------------------------
struct Cyclic {};      // element / row i owned by lid == i mod size (r = lid + k*size)
struct RowFrag {};     // lane l holds matrix row l in registers
struct ColPerLane {};  // lane l owns column l
struct SgPerCol {};    // team t owns columns c = t mod nteams; its lanes stride the rows
struct Uniform {};     // replicated value (every reduction result)

// ---- scalar tags: alpha/beta values the compiler can see --------------------------------
struct One {};
struct Zero {};
struct NegOne {};
template <class T> struct Scalar {
    T v;
};
template <class T> Scalar(T) -> Scalar<T>;

template <class S> inline constexpr bool is_scalar_tag_v = false;
template <> inline constexpr bool is_scalar_tag_v<One> = true;
template <> inline constexpr bool is_scalar_tag_v<Zero> = true;
template <> inline constexpr bool is_scalar_tag_v<NegOne> = true;
template <class T> inline constexpr bool is_scalar_tag_v<Scalar<T>> = true;

// A plain value becomes Scalar<T>; a tag passes through.
template <class S> BATCHLAS_DLA_INLINE constexpr auto as_tag(S s) {
    if constexpr (is_scalar_tag_v<S>) return s;
    else return Scalar<S>{s};
}
template <class S> using tag_t = decltype(as_tag(std::declval<S>()));

// ---- arithmetic trait ---------------------------------------------------------------------
// Inner loops multiply through Arith only: for complex it is explicit real arithmetic, never
// the Annex-G operator* (isnan branches + __mulsc3). Specialise it for another scalar type
// (src/kernels/common/dla_v2_scalar.hh does so for sycl_device::Cx).
template <class T> struct Arith {
    using real_t = T;
    static constexpr bool is_complex = false;
    BATCHLAS_DLA_INLINE static T mul(T a, T b) { return a * b; }
    BATCHLAS_DLA_INLINE static void mac(T& acc, T a, T b) { acc += a * b; }
    BATCHLAS_DLA_INLINE static void mac_conj(T& acc, T a, T b) { acc += a * b; }
    BATCHLAS_DLA_INLINE static T conj(T a) { return a; }
    BATCHLAS_DLA_INLINE static T re(T a) { return a; }
    BATCHLAS_DLA_INLINE static T im(T) { return T(0); }
    BATCHLAS_DLA_INLINE static T make(T r, T) { return r; }
    BATCHLAS_DLA_INLINE static T add(T a, T b) { return a + b; }
    BATCHLAS_DLA_INLINE static T sub(T a, T b) { return a - b; }
    BATCHLAS_DLA_INLINE static T neg(T a) { return -a; }
    BATCHLAS_DLA_INLINE static T zero() { return T(0); }
    BATCHLAS_DLA_INLINE static T one() { return T(1); }
    // c -= a * b as ONE expression (contracts to one FMA; two statements would round twice).
    BATCHLAS_DLA_INLINE static void msub(T& c, const T& a, const T& b) { c = c - a * b; }
    BATCHLAS_DLA_INLINE static T div(const T& a, const T& b) { return a / b; }
    BATCHLAS_DLA_INLINE static T recip(const T& a) { return T(1) / a; }
    BATCHLAS_DLA_INLINE static T div_real(const T& a, const T& s) { return a / s; }
    BATCHLAS_DLA_INLINE static T scale_real(const T& s, const T& a) { return s * a; }
    BATCHLAS_DLA_INLINE static bool is_zero(const T& a) { return a == T(0); }
    BATCHLAS_DLA_INLINE static bool isfinite(const T& a) { return sycl::isfinite(a); }
};

template <class R> struct Arith<std::complex<R>> {
    using T = std::complex<R>;
    using real_t = R;
    static constexpr bool is_complex = true;
    BATCHLAS_DLA_INLINE static T mul(T a, T b) {
        return T(a.real() * b.real() - a.imag() * b.imag(), a.real() * b.imag() + a.imag() * b.real());
    }
    // The latrd split-loop form: bit-identical to the legacy kernel's mac().
    BATCHLAS_DLA_INLINE static void mac(T& acc, T a, T b) {
        const R pr = a.real() * b.real() - a.imag() * b.imag();
        const R pi = a.real() * b.imag() + a.imag() * b.real();
        acc = T(acc.real() + pr, acc.imag() + pi);
    }
    BATCHLAS_DLA_INLINE static void mac_conj(T& acc, T a, T b) {
        const R pr = a.real() * b.real() + a.imag() * b.imag();
        const R pi = a.real() * b.imag() - a.imag() * b.real();
        acc = T(acc.real() + pr, acc.imag() + pi);
    }
    BATCHLAS_DLA_INLINE static T conj(T a) { return T(a.real(), -a.imag()); }
    BATCHLAS_DLA_INLINE static R re(T a) { return a.real(); }
    BATCHLAS_DLA_INLINE static R im(T a) { return a.imag(); }
    BATCHLAS_DLA_INLINE static T make(R r, R i) { return T(r, i); }
    BATCHLAS_DLA_INLINE static T add(T a, T b) { return T(a.real() + b.real(), a.imag() + b.imag()); }
    BATCHLAS_DLA_INLINE static T sub(T a, T b) { return T(a.real() - b.real(), a.imag() - b.imag()); }
    BATCHLAS_DLA_INLINE static T neg(T a) { return T(-a.real(), -a.imag()); }
    BATCHLAS_DLA_INLINE static T zero() { return T(R(0), R(0)); }
    BATCHLAS_DLA_INLINE static T one() { return T(R(1), R(0)); }
    BATCHLAS_DLA_INLINE static void msub(T& c, const T& a, const T& b) {
        const R ar = a.real(), ai = a.imag(), br = b.real(), bi = b.imag();
        c = T(c.real() - (ar * br - ai * bi), c.imag() - (ar * bi + ai * br));
    }
    // Smith's algorithm (no Annex-G, no (c - di) / (c^2 + d^2) overflow past ~1e19 / ~1e154).
    BATCHLAS_DLA_INLINE static T div(const T& a, const T& b) {
        const R ar = a.real(), ai = a.imag(), br = b.real(), bi = b.imag();
        if (sycl::fabs(br) >= sycl::fabs(bi)) {
            const R r = bi / br, den = br + bi * r;
            return T((ar + ai * r) / den, (ai - ar * r) / den);
        }
        const R r = br / bi, den = bi + br * r;
        return T((ar * r + ai) / den, (ai * r - ar) / den);
    }
    BATCHLAS_DLA_INLINE static T recip(const T& d) {
        const R dr = d.real(), di = d.imag();
        if (sycl::fabs(dr) >= sycl::fabs(di)) {
            const R r = di / dr, den = dr + di * r;
            return T(R(1) / den, -r / den);
        }
        const R r = dr / di, den = di + dr * r;
        return T(r / den, R(-1) / den);
    }
    BATCHLAS_DLA_INLINE static T div_real(const T& a, const R& s) { return T(a.real() / s, a.imag() / s); }
    BATCHLAS_DLA_INLINE static T scale_real(const R& s, const T& a) { return T(s * a.real(), s * a.imag()); }
    BATCHLAS_DLA_INLINE static bool is_zero(const T& a) { return a.real() == R(0) && a.imag() == R(0); }
    BATCHLAS_DLA_INLINE static bool isfinite(const T& a) { return sycl::isfinite(a.real()) && sycl::isfinite(a.imag()); }
};

template <class T> using real_t = typename Arith<std::remove_cv_t<T>>::real_t;

// The arithmetic an op's loop uses. arith::Explicit (default) is Arith<T>. arith::Native is
// T's own operators (std::complex keeps its Annex-G multiply): setup code outside inner loops,
// and strict ports whose SASS and rounding must not move (D6: explicit arithmetic everywhere
// costs occupancy). arith::ExplicitInline is Arith<T> with mac/mac_conj as plain `inline`
// calls, the hand kernels' helper form: LLVM applies `#pragma unroll 4` while the call is still
// in the body, which leaves the loop open to runtime unrolling after inlining. The always_inline
// form unrolls only by 4 (latrd symv: 47 vs 78 FFMA at f32 WG 128, hand = 78).
// arith::Fma is explicit fma chains, bit-equal to sycl_device::dev_mul / fma_acc / dev_recip /
// dev_div (src/sycl/device_scalar.hh): the arithmetic of gemm, gemv, potri, trsm, getrs, geqrf.
// Complex rule (D6): a strict port of a kernel that uses std::complex operators takes Native
// (bit identity, Annex-G kept); explicit real arithmetic (Explicit, ExplicitInline, Fma) is for
// algo ports, under tolerance tests with an Inf/NaN case.
namespace arith {
struct Explicit {};
struct Native {};
struct ExplicitInline {};
struct Fma {};
}  // namespace arith

// Operands by reference and complex parts read one at a time, as in the hand helper: the loads
// sit inside the call, and a by-value std::complex is one 8/16-byte load (other unrolling).
template <class T> struct ArithInline : Arith<T> {
    static inline void mac(T& acc, const T& a, const T& b) {
        if constexpr (Arith<T>::is_complex) {
            using R = typename Arith<T>::real_t;
            const R ar = a.real(), ai = a.imag(), br = b.real(), bi = b.imag();
            const R pr = ar * br - ai * bi;
            const R pi = ar * bi + ai * br;
            acc = T(acc.real() + pr, acc.imag() + pi);
        } else {
            acc += a * b;
        }
    }
    static inline void mac_conj(T& acc, const T& a, const T& b) {
        if constexpr (Arith<T>::is_complex) {
            using R = typename Arith<T>::real_t;
            const R ar = a.real(), ai = a.imag(), br = b.real(), bi = b.imag();
            const R pr = ar * br + ai * bi;
            const R pi = ar * bi - ai * br;
            acc = T(acc.real() + pr, acc.imag() + pi);
        } else {
            acc += a * b;
        }
    }
};

template <class T> struct NativeArith {
    using real_t = typename Arith<T>::real_t;
    static constexpr bool is_complex = Arith<T>::is_complex;
    BATCHLAS_DLA_INLINE static T cj(T a) {
        if constexpr (is_complex) return std::conj(a);
        else return a;
    }
    BATCHLAS_DLA_INLINE static T mul(T a, T b) { return a * b; }
    BATCHLAS_DLA_INLINE static void mac(T& acc, T a, T b) { acc += a * b; }
    BATCHLAS_DLA_INLINE static void mac_conj(T& acc, T a, T b) { acc += cj(a) * b; }
    BATCHLAS_DLA_INLINE static T conj(T a) { return cj(a); }
    BATCHLAS_DLA_INLINE static T add(T a, T b) { return a + b; }
    BATCHLAS_DLA_INLINE static T sub(T a, T b) { return a - b; }
    BATCHLAS_DLA_INLINE static T neg(T a) { return -a; }
    BATCHLAS_DLA_INLINE static T zero() { return T(0); }
    BATCHLAS_DLA_INLINE static T one() { return T(1); }
    BATCHLAS_DLA_INLINE static real_t re(const T& a) { return Arith<T>::re(a); }
    BATCHLAS_DLA_INLINE static real_t im(const T& a) { return Arith<T>::im(a); }
    BATCHLAS_DLA_INLINE static T make(real_t r, real_t i) { return Arith<T>::make(r, i); }
    BATCHLAS_DLA_INLINE static void msub(T& c, const T& a, const T& b) { c -= a * b; }
    BATCHLAS_DLA_INLINE static T div(const T& a, const T& b) { return a / b; }
    BATCHLAS_DLA_INLINE static T recip(const T& a) { return T(1) / a; }
    BATCHLAS_DLA_INLINE static T div_real(const T& a, const real_t& s) { return a / s; }
    BATCHLAS_DLA_INLINE static T scale_real(const real_t& s, const T& a) { return s * a; }
    BATCHLAS_DLA_INLINE static bool is_zero(const T& a) { return a == T(0); }
    BATCHLAS_DLA_INLINE static bool isfinite(const T& a) { return Arith<T>::isfinite(a); }
};

// Explicit fma chains (src/sycl/device_scalar.hh, bit for bit): real mac is one fma, complex
// four; mul is dev_mul's two; div / recip are Smith's with fma denominators.
template <class T> struct FmaArith : Arith<T> {
    using A = Arith<T>;
    using R = typename A::real_t;
    BATCHLAS_DLA_INLINE static T mul(const T& a, const T& b) {
        if constexpr (A::is_complex)
            return A::make(sycl::fma(A::re(a), A::re(b), -A::im(a) * A::im(b)), sycl::fma(A::re(a), A::im(b), A::im(a) * A::re(b)));
        else return a * b;
    }
    BATCHLAS_DLA_INLINE static void mac(T& acc, const T& a, const T& b) {
        if constexpr (A::is_complex) {
            R re = sycl::fma(A::re(a), A::re(b), A::re(acc));
            re = sycl::fma(-A::im(a), A::im(b), re);
            R im = sycl::fma(A::re(a), A::im(b), A::im(acc));
            im = sycl::fma(A::im(a), A::re(b), im);
            acc = A::make(re, im);
        } else {
            acc = sycl::fma(a, b, acc);
        }
    }
    BATCHLAS_DLA_INLINE static void mac_conj(T& acc, const T& a, const T& b) {
        if constexpr (A::is_complex) {
            R re = sycl::fma(A::re(a), A::re(b), A::re(acc));
            re = sycl::fma(A::im(a), A::im(b), re);
            R im = sycl::fma(A::re(a), A::im(b), A::im(acc));
            im = sycl::fma(-A::im(a), A::re(b), im);
            acc = A::make(re, im);
        } else {
            acc = sycl::fma(a, b, acc);
        }
    }
    BATCHLAS_DLA_INLINE static void msub(T& c, const T& a, const T& b) {
        if constexpr (A::is_complex) {
            R re = sycl::fma(-A::re(a), A::re(b), A::re(c));
            re = sycl::fma(A::im(a), A::im(b), re);
            R im = sycl::fma(-A::re(a), A::im(b), A::im(c));
            im = sycl::fma(-A::im(a), A::re(b), im);
            c = A::make(re, im);
        } else {
            c = sycl::fma(-a, b, c);
        }
    }
    BATCHLAS_DLA_INLINE static T recip(const T& d) {
        if constexpr (A::is_complex) {
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
    BATCHLAS_DLA_INLINE static T div(const T& a, const T& b) {
        if constexpr (A::is_complex) {
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
};

template <class Ops, class T>
using arith_t = std::conditional_t<
    std::is_same_v<Ops, arith::Native>, NativeArith<std::remove_cv_t<T>>,
    std::conditional_t<std::is_same_v<Ops, arith::ExplicitInline>, ArithInline<std::remove_cv_t<T>>,
                       std::conditional_t<std::is_same_v<Ops, arith::Fma>, FmaArith<std::remove_cv_t<T>>, Arith<std::remove_cv_t<T>>>>>;

// |a|^2 = re^2 + im^2, by value. Its result is a separate rounding: it cannot contract into the
// caller's accumulation (syevx), so accumulate with sumsq / acc_abs2 (level1.hh) instead.
template <class T> BATCHLAS_DLA_INLINE real_t<T> abs2(T a) {
    using A = Arith<std::remove_cv_t<T>>;
    if constexpr (A::is_complex) return A::re(a) * A::re(a) + A::im(a) * A::im(a);
    else return a * a;
}

// BLAS |re| + |im| (dcabs1): the key of asum, amax and iamax.
template <class T> BATCHLAS_DLA_INLINE real_t<T> abs1(T a) {
    using A = Arith<std::remove_cv_t<T>>;
    if constexpr (A::is_complex) return sycl::fabs(A::re(a)) + sycl::fabs(A::im(a));
    else return sycl::fabs(a);
}

// cond ? a : b component by component: LLVM builds no `select` of a complex aggregate, and
// a register array read through one stops being promotable. The components are read before
// the ternaries: a call inside a `?:` arm leaves a branchier CFG (complex<float> SASS moves).
template <class T> BATCHLAS_DLA_INLINE T select(bool cond, T a, T b) {
    using A = Arith<T>;
    if constexpr (A::is_complex) {
        const auto ar = A::re(a), ai = A::im(a), br = A::re(b), bi = A::im(b);
        return A::make(cond ? ar : br, cond ? ai : bi);
    } else {
        return cond ? a : b;
    }
}

// s * x for a tag s; Zero never reads x.
template <class T, class S> BATCHLAS_DLA_INLINE T tag_mul(S s, T x) {
    if constexpr (std::is_same_v<S, One>) return x;
    else if constexpr (std::is_same_v<S, Zero>) return Arith<T>::zero();
    else if constexpr (std::is_same_v<S, NegOne>) return Arith<T>::neg(x);
    else return Arith<T>::mul(static_cast<T>(s.v), x);
}

// y := alpha * acc + beta * y; beta == Zero is a store that never loads y.
template <class A, class B, class T, class Ref> BATCHLAS_DLA_INLINE void store(A alpha, T acc, B beta, Ref&& y) {
    if constexpr (std::is_same_v<B, Zero>) {
        y = tag_mul(alpha, acc);
    } else if constexpr (std::is_same_v<B, One>) {
        y = Arith<T>::add(tag_mul(alpha, acc), static_cast<T>(y));
    } else {
        y = Arith<T>::add(tag_mul(alpha, acc), tag_mul(beta, static_cast<T>(y)));
    }
}

// ---- algorithm tags (every default is a type, chosen from executor and operation) -------
namespace alg {
struct RowOwner {};             // gemv N: one work-item per output row, serial dot, no reduction
struct SgPerCol {};             // gemv T/C: one team (sub-group) per output column, shuffles only
struct Serial {};               // Thread executor
struct GroupReducePerOutput {}; // v1's generic algorithm: one group reduction per output; opt-in only
struct RowOwnerSplit {};        // symv/hemv lower: the latrd split loop (row part, then column part)
struct RowOwnerSplitStoredDiag {};  // hemv RowOwnerSplit reading A(r,r) as stored (caller: Im == 0)
struct Cyclic {};               // ger: column loop outside, owned rows inside; no % or /
struct Scaled {};               // nrm2: two-pass lassq, overflow-safe
struct Unscaled {};             // nrm2: sqrt(dotc(x, x)); overflows past sqrt(max)
struct Blue {};                 // nrm2: dnrm2.f90's three accumulators, one exchange
struct Fused {};                // sumsq: re^2 and im^2 contract into the running sum
struct Unfused {};              // sumsq: the running sum adds abs2(x), its own rounding
// for_each_owned_2d: how an item walks the column-major m x n grid e = lid + s * size.
struct Flat {};                 // (e - (e/m)*m, e/m) per step, e in Idx (64-bit: v1 index math)
struct DivMod {};               // (e % m, e / m) per step, the literal hand spelling
struct Stepped {};              // one division per item, then r += size with a carry loop
struct Walk {};                 // one division per item, (dr, dc) = (size % m, size / m) per step
}  // namespace alg

// Butterfly order of a team reduction: xor masks P/2 .. 1 (Down) or 1 .. P/2 (Asc). The two
// sum in different orders, so a strict port names the hand kernel's.
namespace bfly {
struct Down {};
struct Asc {};
}  // namespace bfly

// Op tags for scratch_elements.
namespace op {
template <int K> struct ReduceN {};
using Reduce = ReduceN<1>;
}  // namespace op

}  // namespace batchlas::device::v2
