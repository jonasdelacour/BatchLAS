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
//   BATCHLAS_DEVICE_INLINE  always_inline: leaf arithmetic, accessors, ops replacing expression text
//   BATCHLAS_DEVICE_HELPER  plain inline: ops replacing a plain-inline hand helper (phases,
//                        reductions, bcast, epilogues); LLVM unrolls the caller's loop while
//                        the call is still there, as it did for the hand helper
//   BATCHLAS_DEVICE_BODY    on a lambda passed to a loop op, `[&](int i) BATCHLAS_DEVICE_BODY {...}`:
//                        a lambda body is otherwise optimised alone (LFTR, IV canonicalisation)
// Each is overridable (-D) for a backend whose inliner needs another spelling.
#ifndef BATCHLAS_DEVICE_INLINE
#define BATCHLAS_DEVICE_INLINE __attribute__((always_inline)) inline
#endif
#ifndef BATCHLAS_DEVICE_HELPER
#define BATCHLAS_DEVICE_HELPER inline
#endif
#ifndef BATCHLAS_DEVICE_BODY
#define BATCHLAS_DEVICE_BODY __attribute__((always_inline))
#endif

namespace batchlas::device {

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
    BATCHLAS_DEVICE_INLINE constexpr operator int() const { return N; }
};
struct Dyn {
    int value;
    BATCHLAS_DEVICE_INLINE constexpr operator int() const { return value; }
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
template <class S> BATCHLAS_DEVICE_INLINE constexpr auto as_tag(S s) {
    if constexpr (is_scalar_tag_v<S>) return s;
    else return Scalar<S>{s};
}
template <class S> using tag_t = decltype(as_tag(std::declval<S>()));

namespace alg {
struct RowOwner {};             // gemv N: one work-item per output row, serial dot, no reduction
struct SgPerCol {};             // gemv T/C: one team (sub-group) per output column, shuffles only
struct Serial {};               // Thread executor
struct GroupReducePerOutput {}; // the legacy generic algorithm: one group reduction per output; opt-in only
struct RowOwnerSplit {};        // symv/hemv lower: the latrd split loop (row part, then column part)
struct RowOwnerSplitStoredDiag {};  // hemv RowOwnerSplit reading A(r,r) as stored (caller: Im == 0)
struct Cyclic {};               // ger: column loop outside, owned rows inside; no % or /
struct Scaled {};               // nrm2: two-pass lassq, overflow-safe
struct Unscaled {};             // nrm2: sqrt(dotc(x, x)); overflows past sqrt(max)
struct Blue {};                 // nrm2: dnrm2.f90's three accumulators, one exchange
struct Fused {};                // sumsq: re^2 and im^2 contract into the running sum
struct Unfused {};              // sumsq: the running sum adds abs2(x), its own rounding
// for_each_owned_2d: how an item walks the column-major m x n grid e = lid + s * size.
struct Flat {};                 // (e - (e/m)*m, e/m) per step, e in Idx (64-bit: the legacy index math)
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

// A scalar with a host-decided zero flag: an epilogue never reads y when beta's flag is set.
template <class T> struct MaybeZero {
    T v;
    bool zero;
};
template <class T> MaybeZero(T, bool) -> MaybeZero<T>;
template <class T> inline constexpr bool is_scalar_tag_v<MaybeZero<T>> = true;

// The absent operand / callback: an op compiles the step out (if constexpr (is_none_v<X>)).
struct None {};
struct NoOp {
    template <class... A> BATCHLAS_DEVICE_INLINE constexpr void operator()(A&&...) const {}
};
template <class X> inline constexpr bool is_none_v = std::is_same_v<std::remove_cvref_t<X>, None>;

// Element (row) i owned by work-item i: no stride, one element per item (i < size).
struct OnePerItem {};

// f(std::integral_constant<int, I>{}) for I = 0 .. N-1: every I a constant expression in f.
template <int N, class F> BATCHLAS_DEVICE_INLINE constexpr void static_for(F&& f) {
    [&]<int... I>(std::integer_sequence<int, I...>) { (f(std::integral_constant<int, I>{}), ...); }(std::make_integer_sequence<int, N>{});
}

// The compile-time value of an extent (Ext<N>: N), or -1 for a run-time one.
template <class E> inline constexpr int static_extent_v = -1;
template <int N> inline constexpr int static_extent_v<Ext<N>> = N;

// NaN-losing max / min (sycl::fmax / fmin: a NaN operand yields the other), as reduction ops.
namespace op {
struct FMax {
    template <class R> BATCHLAS_DEVICE_INLINE R operator()(R a, R b) const { return sycl::fmax(a, b); }
};
struct FMin {
    template <class R> BATCHLAS_DEVICE_INLINE R operator()(R a, R b) const { return sycl::fmin(a, b); }
};
}  // namespace op

// The part of a matrix an op touches: Full, a triangle with its diagonal (Lower, Upper) or
// without it (StrictLower, StrictUpper).
enum class Part { Full, Lower, Upper, StrictLower, StrictUpper };
namespace detail {
template <Part P> BATCHLAS_DEVICE_INLINE bool in_part(int i, int j) {
    if constexpr (P == Part::Full) return true;
    else if constexpr (P == Part::Lower) return i >= j;
    else if constexpr (P == Part::Upper) return i <= j;
    else if constexpr (P == Part::StrictLower) return i > j;
    else return i < j;
}
}  // namespace detail

// sync<S>(ex) barrier scopes: Part = the executor's own unit (a masked partition's own barrier;
// nothing on Teams, whose team is a lockstep slice of a sub-group), Sub = the sub-group,
// Group = the whole executor (sync(ex): work-group on WorkGroup, sub-group on a team executor).
namespace scope {
struct Part {};
struct Sub {};
struct Group {};
}  // namespace scope

// Team loops: Lockstep runs ceil(n / nteams) trips on every team; Diverge stops at its own bound.
namespace loop {
struct Lockstep {};
struct Diverge {};
}  // namespace loop

namespace alg {
struct TeamPerRow {};                   // gemv N / her2: one team per output row, lanes stride columns
template <int W> struct ColDot {        // gemv T/C: W consecutive lanes per output column
    static_assert(W > 0 && (W & (W - 1)) == 0, "alg::ColDot: W is a power of two");
    static constexpr int width = W;
};
struct LanePerRow {};                   // trsv: lane i holds x(i); column j reaches the lanes by bcast
}  // namespace alg

// Which side of the accumulated product alpha multiplies (gemv): Left alpha * acc, Right acc * alpha.
namespace alpha_side {
struct Left {};
struct Right {};
}  // namespace alpha_side

// Triangular-solve diagonal: Unit (not read), Div, Recip (x * (1 / a_ii)), Real (x / re(a_ii)).
namespace tri_diag {
struct Unit {};
struct Div {};
struct Recip {};
struct Real {};
}  // namespace tri_diag

}  // namespace batchlas::device

#include <batchlas/device/arith.hh>   // every core.hh user also reads Arith (arith.hh includes core.hh)
