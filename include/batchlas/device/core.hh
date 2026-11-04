// SPDX-License-Identifier: MIT
#pragma once
// Device LA vocabulary: extents, memory, layout, scalar and algorithm tags (Arith: arith.hh).
// Every choice a kernel's SASS depends on is a type here, never a runtime value.

#include <complex>
#include <type_traits>
#include <utility>

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

namespace batchlas::device::inline v2 {

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

struct Cyclic {};      // element / row i owned by lid == i mod size (r = lid + k*size)
struct RowFrag {};     // lane l holds matrix row l in registers
struct ColPerLane {};  // lane l owns column l
struct SgPerCol {};    // team t owns columns c = t mod nteams; its lanes stride the rows
struct Uniform {};     // replicated value (every reduction result)


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
struct Walk {};
struct Default {};             // a tier's own default algorithm (tier::Cta / Blocked)                 // one division per item, (dr, dc) = (size % m, size / m) per step
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

// ---- v2.2 vocabulary (design A.2 core.hh) ---------------------------------------------------
// A scalar whose zero-ness was decided on the host (alpha == 0, beta == 0 from the BLAS value):
// an epilogue never reads y when beta's flag is set, without comparing a value in the kernel.
template <class T> struct MaybeZero {
    T v;
    bool zero;
};
template <class T> MaybeZero(T, bool) -> MaybeZero<T>;
template <class T> inline constexpr bool is_scalar_tag_v<MaybeZero<T>> = true;

// The absent operand / callback: an op taking an optional tau, phase or per-step hook takes
// None and compiles the step out (if constexpr (is_none_v<X>)).
struct None {};
struct NoOp {
    template <class... A> BATCHLAS_DLA_INLINE constexpr void operator()(A&&...) const {}
};
template <class X> inline constexpr bool is_none_v = std::is_same_v<std::remove_cvref_t<X>, None>;

// Element (row) i owned by work-item i: no stride, one element per item (i < size).
struct OnePerItem {};

// f(std::integral_constant<int, I>{}) for I = 0 .. N-1, unrolled by construction (every I is a
// constant expression inside f, so register arrays indexed by it stay in registers).
template <int N, class F> BATCHLAS_DLA_INLINE constexpr void static_for(F&& f) {
    [&]<int... I>(std::integer_sequence<int, I...>) { (f(std::integral_constant<int, I>{}), ...); }(std::make_integer_sequence<int, N>{});
}

// The compile-time value of an extent (Ext<N>: N), or -1 for a run-time one.
template <class E> inline constexpr int static_extent_v = -1;
template <int N> inline constexpr int static_extent_v<Ext<N>> = N;

// NaN-losing max / min (sycl::fmax / fmin: a NaN operand yields the other), as reduction ops.
namespace op {
struct FMax {
    template <class R> BATCHLAS_DLA_INLINE R operator()(R a, R b) const { return sycl::fmax(a, b); }
};
struct FMin {
    template <class R> BATCHLAS_DLA_INLINE R operator()(R a, R b) const { return sycl::fmin(a, b); }
};
}  // namespace op

}  // namespace batchlas::device

#include <batchlas/device/arith.hh>   // every core.hh user also reads Arith (arith.hh includes core.hh)
