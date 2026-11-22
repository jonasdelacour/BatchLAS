// SPDX-License-Identifier: MIT
#pragma once
/// @file
/// @brief The device vocabulary: extents, memory spaces, layouts, scalar tags and algorithm tags.
/// @ingroup api_device_operands
///
/// Every choice a kernel's machine code depends on is a type here, never a run-time value. The
/// Arith trait is in arith.hh, which this header includes.
///
/// Inline policy: an op takes the attribute of the code it replaces.
/// - `BATCHLAS_DEVICE_INLINE` (always_inline): leaf arithmetic, accessors, ops that replace
///   expression text.
/// - `BATCHLAS_DEVICE_HELPER` (plain inline): ops that replace a plain-inline hand helper (phases,
///   reductions, bcast, epilogues); LLVM unrolls the caller's loop while the call is still there,
///   as it did for the hand helper.
/// - `BATCHLAS_DEVICE_BODY`: on a lambda passed to a loop op,
///   `[&](int i) BATCHLAS_DEVICE_BODY {...}`; a lambda body is otherwise optimised alone.
///
/// Each is overridable with `-D` for a backend whose inliner needs another spelling.

#include <complex>
#include <type_traits>
#include <utility>

#include <sycl/sycl.hpp>

#include <batchlas/blas/enums.hh>

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

/// @brief Algorithm tags: the template arguments that choose how a routine runs, so a kernel
/// compiles only the path it takes.
namespace alg {
/// @brief Loop unrolling: no metadata; LLVM and ptxas decide.
/// @ingroup api_device_exec
struct UnrollDefault {};
/// @brief Loop unrolling: `llvm.loop.unroll.disable`.
/// @ingroup api_device_exec
struct UnrollNone {};
/// @brief Loop unrolling: `unroll_count(N)`.
/// @ingroup api_device_exec
/// @tparam N the unroll count, >= 1
template <int N> struct UnrollCount {
    static constexpr int count = N;  ///< The unroll count, N.
};
/// @brief Loop unrolling: `unroll(full)`.
/// @ingroup api_device_exec
struct UnrollFull {};
}  // namespace alg

/// @brief Elements [lo, hi) in the parent's coordinates: an op on a Range owns i - lo mod size.
/// @ingroup api_device_operands
struct Range {
    int lo;  ///< The first element.
    int hi;  ///< One past the last element.
};

/// @brief A compile-time extent N: free, it folds into every index.
/// @ingroup api_device_operands
/// @tparam N the extent
template <int N> struct Ext {
    static constexpr int value = N;  ///< The extent, N.
    /// @brief The extent as an int.
    BATCHLAS_DEVICE_INLINE constexpr operator int() const { return N; }
};
/// @brief A run-time extent: one int.
/// @ingroup api_device_operands
struct Dyn {
    int value;  ///< The extent.
    /// @brief The extent as an int.
    BATCHLAS_DEVICE_INLINE constexpr operator int() const { return value; }
};
/// @brief The extent (or stride) 1, `Ext<1>`.
/// @ingroup api_device_operands
using Unit = Ext<1>;

/// @brief The memory space of an operand: Global, Local or Reg (registers).
/// @ingroup api_device_operands
enum class Mem { Global, Local, Reg };
/// @brief The scope of a collective.
/// @ingroup api_device_exec
///
/// Own is the executor's own unit (a reduction's natural_scope; sync's own barrier), Sub the
/// sub-group (one team), Group the whole executor.
enum class Scope { Own, Sub, Group };

/// @brief Layout: element or row i is owned by the item with lid == i mod size (r = lid + k size).
/// @ingroup api_device_operands
struct Cyclic {};
/// @brief Layout: lane l holds matrix row l in registers.
/// @ingroup api_device_operands
struct RowFrag {};
/// @brief Layout: lane l owns column l.
/// @ingroup api_device_operands
struct ColPerLane {};
/// @brief Layout: team t owns the columns c with c mod nteams == t; its lanes stride the rows.
/// @ingroup api_device_operands
struct SgPerCol {};
/// @brief Layout: a value replicated on every item (every reduction result).
/// @ingroup api_device_operands
struct Uniform {};

/// @brief The scalar tag 1: a multiply by it compiles away.
/// @ingroup api_device_arith
struct One {};
/// @brief The scalar tag 0: a term multiplied by it is never read.
/// @ingroup api_device_arith
struct Zero {};
/// @brief The scalar tag -1: a multiply by it is a negation.
/// @ingroup api_device_arith
struct NegOne {};
/// @brief A run-time scalar v as a tag.
/// @ingroup api_device_arith
/// @tparam T the scalar type
template <class T> struct Scalar {
    T v;  ///< The value.
};
/// @cond
template <class T> Scalar(T) -> Scalar<T>;
/// @endcond

/// @brief True when S is a scalar tag: One, Zero, NegOne, `Scalar<T>` or `MaybeZero<T>`.
/// @ingroup api_device_arith
/// @tparam S the type to test
template <class S> inline constexpr bool is_scalar_tag_v = false;
/// @cond
template <> inline constexpr bool is_scalar_tag_v<One> = true;
template <> inline constexpr bool is_scalar_tag_v<Zero> = true;
template <> inline constexpr bool is_scalar_tag_v<NegOne> = true;
template <class T> inline constexpr bool is_scalar_tag_v<Scalar<T>> = true;
/// @endcond

/// @brief A plain value as `Scalar<S>`; a tag passes through.
/// @ingroup api_device_arith
/// @param s the value or tag
/// @return the tag
template <class S> BATCHLAS_DEVICE_INLINE constexpr auto as_tag(S s) {
    if constexpr (is_scalar_tag_v<S>) return s;
    else return Scalar<S>{s};
}
/// @brief The tag type of S, `decltype(as_tag(S))`.
/// @ingroup api_device_arith
/// @tparam S a value or tag type
template <class S> using tag_t = decltype(as_tag(std::declval<S>()));

namespace alg {
/// @brief gemv NoTrans: one work-item per output row, a serial dot, no reduction.
/// @ingroup api_device_blas2
struct RowOwner {};
/// @brief gemv Trans / ConjTrans: one team (sub-group) per output column, shuffles only.
/// @ingroup api_device_blas2
struct SgPerCol {};
/// @brief The serial algorithm of the Thread executor.
/// @ingroup api_device_blas2
struct Serial {};
/// @brief The legacy generic algorithm: one group reduction per output; opt-in only.
/// @ingroup api_device_blas2
struct GroupReducePerOutput {};
/// @brief symv / hemv Lower: the latrd split loop, the row part then the column part.
/// @ingroup api_device_blas2
struct RowOwnerSplit {};
/// @brief hemv RowOwnerSplit reading A(r, r) as stored; the caller guarantees Im A(r, r) == 0.
/// @ingroup api_device_blas2
struct RowOwnerSplitStoredDiag {};
/// @brief ger: the column loop outside, the owned rows inside; no `%` or `/`.
/// @ingroup api_device_blas2
struct Cyclic {};
/// @brief nrm2: the two-pass lassq, overflow-safe.
/// @ingroup api_device_blas1
struct Scaled {};
/// @brief nrm2: sqrt(dotc(x, x)); overflows past sqrt(max).
/// @ingroup api_device_blas1
struct Unscaled {};
/// @brief nrm2: dnrm2.f90's three accumulators, one exchange.
/// @ingroup api_device_blas1
struct Blue {};
/// @brief sumsq: re^2 and im^2 contract into the running sum; trsv: the update is one Ops::msub.
/// @ingroup api_device_blas1
struct Fused {};
/// @brief sumsq: the running sum adds abs2(x), its own rounding; trsv: sub of mul, a * x_j, two
/// roundings (getrs, sytrs).
/// @ingroup api_device_blas1
struct Unfused {};
// for_each_owned_2d: how an item walks the column-major m x n grid e = lid + s * size.
/// @brief for_each_owned_2d: (e - (e / m) m, e / m) per step, e in Idx (64-bit: the legacy index
/// math).
/// @ingroup api_device_blas1
struct Flat {};
/// @brief for_each_owned_2d: (e % m, e / m) per step, the literal hand spelling.
/// @ingroup api_device_blas1
struct DivMod {};
/// @brief for_each_owned_2d: one division per item, then r += size with a carry loop.
/// @ingroup api_device_blas1
struct Stepped {};
/// @brief for_each_owned_2d: one division per item, then (dr, dc) = (size % m, size / m) per step.
/// @ingroup api_device_blas1
struct Walk {};
/// @brief A tier's own default algorithm (tier::Cta, tier::Blocked).
/// @ingroup api_device_launch
struct Default {};
/// @brief potrf tier::Blocked: each panel updated from the columns to its left.
/// @ingroup api_device_launch
struct LeftLooking {};
}  // namespace alg

// Butterfly orders and pivot-search tie rules.
namespace alg {
/// @brief Butterfly order: xor masks P / 2 down to 1.
/// @ingroup api_device_blas1
struct BflyDown {};
/// @brief Butterfly order: xor masks 1 up to P / 2, another summation order.
/// @ingroup api_device_blas1
struct BflyAsc {};
/// @brief Pivot-search tie rule: among equal keys, the lower index.
/// @ingroup api_device_blas1
struct TieLowest {};
/// @brief Pivot-search tie rule: among equal keys, the higher index.
/// @ingroup api_device_blas1
struct TieHighest {};
/// @brief Pivot-search tie rule chosen at run time.
/// @ingroup api_device_blas1
struct TieRt {
    bool low;  ///< True: the lower index; false: the higher (sytrf Upper takes the higher logical index).
};
}  // namespace alg

/// @brief Reduction tags: the Op of scratch_elements, and the NaN-losing max and min.
namespace reduce_op {
/// @brief A reduction of K values at once, for scratch_elements.
/// @ingroup api_device_blas1
/// @tparam K the number of values
template <int K> struct ReduceN {};
/// @brief A reduction of one value, `ReduceN<1>`.
/// @ingroup api_device_blas1
using Reduce = ReduceN<1>;
}  // namespace reduce_op

/// @brief A scalar with a host-decided zero flag: an epilogue never reads y when beta's flag is set.
/// @ingroup api_device_arith
/// @tparam Scalar the scalar type
template <class Scalar> struct MaybeZero {
    Scalar v;  ///< The value.
    bool zero;  ///< The host's zero flag.
};
/// @cond
template <class Scalar> MaybeZero(Scalar, bool) -> MaybeZero<Scalar>;
template <class Scalar> inline constexpr bool is_scalar_tag_v<MaybeZero<Scalar>> = true;
/// @endcond

/// @brief The absent operand: an op compiles the step out, `if constexpr (is_none_v<X>)`.
/// @ingroup api_device_operands
struct None {};
/// @brief The absent callback: a call that does nothing.
/// @ingroup api_device_operands
struct NoOp {
    /// @brief Does nothing.
    template <class... A> BATCHLAS_DEVICE_INLINE constexpr void operator()(A&&...) const {}
};
/// @brief True when X is None.
/// @ingroup api_device_operands
/// @tparam X the type to test
template <class X> inline constexpr bool is_none_v = std::is_same_v<std::remove_cvref_t<X>, None>;

/// @brief Layout: element or row i owned by work-item i, one element per item (i < size), no stride.
/// @ingroup api_device_operands
struct OnePerItem {};

/// @brief Calls `f(std::integral_constant<int, I>{})` for I = 0 to N - 1, each I a constant
/// expression in f.
/// @ingroup api_device_lowlevel
/// @tparam N the trip count
/// @param f the callable
template <int N, class F> BATCHLAS_DEVICE_INLINE constexpr void static_for(F&& f) {
    [&]<int... I>(std::integer_sequence<int, I...>) { (f(std::integral_constant<int, I>{}), ...); }(std::make_integer_sequence<int, N>{});
}

/// @brief The compile-time value of an extent (`Ext<N>`: N), or -1 for a run-time one.
/// @ingroup api_device_operands
/// @tparam E the extent type
template <class E> inline constexpr int static_extent_v = -1;
/// @cond
template <int N> inline constexpr int static_extent_v<Ext<N>> = N;
/// @endcond

// NaN-losing max and min (sycl::fmax, sycl::fmin: a NaN operand yields the other).
namespace reduce_op {
/// @brief The NaN-losing max as a reduction op: a NaN operand yields the other.
/// @ingroup api_device_blas1
struct FMax {
    /// @brief sycl::fmax(a, b).
    template <class Real> BATCHLAS_DEVICE_INLINE Real operator()(Real a, Real b) const { return sycl::fmax(a, b); }
};
/// @brief The NaN-losing min as a reduction op: a NaN operand yields the other.
/// @ingroup api_device_blas1
struct FMin {
    /// @brief sycl::fmin(a, b).
    template <class Real> BATCHLAS_DEVICE_INLINE Real operator()(Real a, Real b) const { return sycl::fmin(a, b); }
};
}  // namespace reduce_op

/// @brief The part of a matrix an op touches: Full, a triangle with its diagonal (Lower, Upper) or
/// without it (StrictLower, StrictUpper).
/// @ingroup api_device_operands
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

namespace alg {
/// @brief Team loops: Lockstep runs ceil(n / nteams) trips on every team.
/// @ingroup api_device_exec
struct Lockstep {};
/// @brief Team loops: Diverge stops each team at its own bound.
/// @ingroup api_device_exec
struct Diverge {};

/// @brief gemv NoTrans / her2: one team per output row, its lanes striding the columns.
/// @ingroup api_device_blas2
struct TeamPerRow {};
/// @brief gemv Trans / ConjTrans: W consecutive lanes per output column.
/// @ingroup api_device_blas2
/// @tparam W the lanes per column, a power of two
template <int W> struct ColDot {
    static_assert(W > 0 && (W & (W - 1)) == 0, "alg::ColDot: W is a power of two");
    static constexpr int width = W;  ///< The lanes per column, W.
};
/// @brief trsv: lane i holds x(i); column j reaches the lanes by bcast.
/// @ingroup api_device_blas2
struct LanePerRow {};
/// @brief trsv LanePerRow on a RowFrag block, its step loop rolled.
/// @ingroup api_device_blas2
struct LanePerRowRolled {};
/// @brief trsv Serial on R rows from r0, the right-hand side read after the dot.
/// @ingroup api_device_blas2
/// @tparam R the rows
template <int R> struct SerialRows {
    static constexpr int rows = R;  ///< The rows, R.
};
/// @brief trsv / trmv Trans or ConjTrans: W accumulators per column dot.
/// @ingroup api_device_blas2
/// @tparam W the accumulators, a power of two
template <int W> struct SplitDot {
    static_assert(W > 0 && (W & (W - 1)) == 0, "alg::SplitDot: W is a power of two");
    static constexpr int width = W;  ///< The accumulators, W.
};
/// @brief trsv Diag::NonUnit: x / a_ii.
/// @ingroup api_device_blas2
struct Div {};
/// @brief trsv Diag::NonUnit: x * (1 / a_ii).
/// @ingroup api_device_blas2
struct Recip {};
/// @brief trsv Diag::NonUnit: x / re(a_ii).
/// @ingroup api_device_blas2
struct RealDiv {};
/// @brief trsv Diag::NonUnit: a diagonal step object's apply.
/// @ingroup api_device_blas2
struct DiagApply {};
/// @brief trsv update: sub of mul, x_j * a, the X op(A) = B operand order (potrf; a complex fma
/// product is not bit-commutative).
/// @ingroup api_device_blas2
struct UnfusedRight {};
/// @brief trsv update: the row's dot from zero, then x_i := b(i) - dot (with alg::SerialRows).
/// @ingroup api_device_blas2
struct DotThenSub {};
/// @brief trsv update: x += a x_j on every row, with a the caller's negated multipliers, zero on
/// and above the diagonal (with alg::LanePerRow on a RowFrag).
/// @ingroup api_device_blas2
struct Axpy {};
}  // namespace alg

/// @brief The diagonal a copy or unit-head view gives: Keep (stored), Unit (1), Real ((re, 0)),
/// Zero (0).
/// @ingroup api_device_operands
enum class DiagFill { Keep, Unit, Real, Zero };

/// @brief xORMBR's VECT: apply Q or P.
/// @ingroup api_device_lapack_qr
enum class Vect { Q, P };
/// @brief xSYGST's ITYPE: AinvB (1) is inv(F) A inv(F)^H; AB (2) and BA (3) are F^H A F.
/// @ingroup api_device_lapack_eig
enum class Itype { AinvB = 1, AB, BA };

}  // namespace batchlas::device

#include <batchlas/device/arith.hh>   // every core.hh user also reads Arith (arith.hh includes core.hh)
