// SPDX-License-Identifier: MIT
#pragma once
/// @file
/// @brief Reductions and lane exchanges: the complex-aware xor shuffle, the butterfly, the
/// leader-only shift fold, the reduce-scatter, and the ArgMax pair with its merge rule.
/// @ingroup api_device_blas1
///
/// A `part` is a sycl::sub_group or a SubGroupPartition. allreduce, with or without Scratch, is in
/// level1.hh.

#include <climits>
#include <cstdint>
#include <limits>
#include <type_traits>

#include <sycl/sycl.hpp>

#include <batchlas/device/core.hh>
#include <batchlas/device/exec.hh>

namespace batchlas::device {

namespace alg {
/// @brief iamax: a NaN key ranks as +inf, so it is found and no compare sees a NaN.
/// @ingroup api_device_blas1
struct NanAsInf {};
/// @brief A NaN key never beats a number.
/// @ingroup api_device_blas1
struct NanSkip {};
/// @brief amax_by: any NaN key makes the result NaN, as amax does.
/// @ingroup api_device_blas1
struct NanPropagate {};
}  // namespace alg

/// @brief The {index, value} of a maximum; index < 0 is "no candidate" (an item that owned
/// nothing).
/// @ingroup api_device_blas1
/// @tparam Real the key type
template <class Real> struct ArgMax {
    int index;  ///< The index of the maximum, or < 0 for no candidate.
    Real value;  ///< The key of the maximum.
};

namespace detail {
template <class Part, class R> BATCHLAS_DEVICE_INLINE R shfl_xor_word(const Part& part, R v, std::uint32_t m) {
    return xor_shuffle(part, v, m);  // exec.hh: portable::permute_group_by_xor, or ADL into SubGroupPartition
}
template <class Part, class R> BATCHLAS_DEVICE_INLINE R shfl_down_word(const Part& part, R v, std::uint32_t d) {
    if constexpr (std::is_same_v<Part, sycl::sub_group>) return sycl::shift_group_left(part, v, d);
    else return shift_group_left(part, v, d);  // ADL
}
template <class T> inline constexpr bool is_argmax_v = false;
template <class Real> inline constexpr bool is_argmax_v<ArgMax<Real>> = true;
}  // namespace detail

/// @brief v of lane (lane ^ m), member by member for a complex value or an ArgMax pair.
/// @ingroup api_device_blas1
///
/// A shuffle moves one 32- or 64-bit word per call. part is a sycl::sub_group, a
/// SubGroupPartition, or an executor holding one (Partition: its `.part`).
/// @param part the sub-group, partition, or partition executor
/// @param v this lane's value
/// @param m the xor mask
/// @return lane (lane ^ m)'s v
template <class Part, class T> BATCHLAS_DEVICE_INLINE T shuffle_xor(const Part& part, T v, std::uint32_t m) {
    if constexpr (requires { part.part; }) {
        return shuffle_xor(part.part, v, m);
    } else if constexpr (detail::is_argmax_v<T>) {
        return T{detail::shfl_xor_word(part, v.index, m), detail::shfl_xor_word(part, v.value, m)};
    } else if constexpr (Arith<T>::is_complex) {
        using A = Arith<T>;
        return A::make(detail::shfl_xor_word(part, A::re(v), m), detail::shfl_xor_word(part, A::im(v), m));
    } else {
        return detail::shfl_xor_word(part, v, m);
    }
}

/// @brief v of lane (lane + d); past the end it is unspecified.
/// @ingroup api_device_blas1
/// @param part the sub-group or partition
/// @param v this lane's value
/// @param d the lane offset
/// @return lane (lane + d)'s v
template <class Part, class T> BATCHLAS_DEVICE_INLINE T shuffle_down(const Part& part, T v, std::uint32_t d) {
    if constexpr (Arith<T>::is_complex) {
        using A = Arith<T>;
        return A::make(detail::shfl_down_word(part, A::re(v), d), detail::shfl_down_word(part, A::im(v), d));
    } else {
        return detail::shfl_down_word(part, v, d);
    }
}

// detail::butterfly_inline is the same body always_inline, for an op that replaces an
// always_inline hand butterfly (inline policy, core.hh); butterfly forwards to it.
namespace detail {
template <int Lo, int Hi, class Order = alg::BflyDown, class Part, class T, class Op>
BATCHLAS_DEVICE_INLINE T butterfly_inline(const Part& part, T v, Op op) {
    static_assert(detail::is_pow2(Lo) && detail::is_pow2(Hi) && Lo <= Hi, "butterfly: Lo <= Hi, powers of two");
    static_assert(std::is_same_v<Order, alg::BflyDown> || std::is_same_v<Order, alg::BflyAsc>, "butterfly: alg::BflyDown or alg::BflyAsc");
    if constexpr (std::is_same_v<Order, alg::BflyDown>) {
#pragma unroll
        for (int m = Hi / 2; m >= Lo; m >>= 1) v = op(v, shuffle_xor(part, v, static_cast<std::uint32_t>(m)));
    } else {
#pragma unroll
        for (int m = Lo; m < Hi; m <<= 1) v = op(v, shuffle_xor(part, v, static_cast<std::uint32_t>(m)));
    }
    return v;
}
}  // namespace detail
/// @brief COLLECTIVE: the xor butterfly over the masks Lo, 2 Lo, ..., Hi / 2 (powers of two).
/// @ingroup api_device_blas1
///
/// Every lane of each aligned Hi-lane block ends with op over the Hi / Lo values whose lane agrees
/// with it outside [Lo, Hi). Op is called `op(mine, theirs)` and must be associative; for a
/// commutative op every lane gets the identical value.
/// @tparam Lo the smallest mask, a power of two
/// @tparam Hi the block width, a power of two >= Lo
/// @tparam Order alg::BflyDown (masks Hi / 2 down to Lo) or alg::BflyAsc (Lo up to Hi / 2):
///         two evaluation orders, so a strict port names its hand kernel's
/// @param part the sub-group or partition
/// @param v this lane's value
/// @param op the associative operation
/// @return the reduction over the lane's block
template <int Lo, int Hi, class Order = alg::BflyDown, class Part, class T, class Op>
BATCHLAS_DEVICE_HELPER T butterfly(const Part& part, T v, Op op) {
    return detail::butterfly_inline<Lo, Hi, Order>(part, v, op);
}
/// @brief COLLECTIVE: the butterfly over a run-time, uniform power-of-two width hi: masks hi / 2
/// down to lo, or lo up to hi / 2.
/// @ingroup api_device_blas1
/// @tparam Order alg::BflyDown or alg::BflyAsc
/// @param part the sub-group or partition
/// @param v this lane's value
/// @param op the associative operation
/// @param lo the smallest mask, >= 1
/// @param hi the block width, a power of two
/// @return the reduction over the lane's block
// always_inline with unsigned masks: the hand form it replaced (plain inline, cdouble geqr2
// resident: +4 REG).
template <class Order = alg::BflyDown, class Part, class T, class Op>
BATCHLAS_DEVICE_INLINE T butterfly(const Part& part, T v, Op op, int lo, int hi) {
    const auto ulo = static_cast<std::uint32_t>(lo), uhi = static_cast<std::uint32_t>(hi);
    if constexpr (std::is_same_v<Order, alg::BflyDown>) {
        for (std::uint32_t m = uhi / 2; m >= ulo; m >>= 1) v = op(v, shuffle_xor(part, v, m));
    } else {
        for (std::uint32_t m = ulo; m < uhi; m <<= 1) v = op(v, shuffle_xor(part, v, m));
    }
    return v;
}

/// @brief COLLECTIVE, leader-only result: the shift fold of lanes {l + s t : t < Count} onto head
/// lane l.
/// @ingroup api_device_blas1
///
/// It computes op(v0, op(v1, ...)) in the tree order Count / 2 down to 1; the other lanes hold
/// partials, so broadcast the result if every lane needs it. Count * s must not exceed the
/// partition width.
/// @tparam Count the number of lanes folded, a power of two
/// @tparam Stride the lane stride type: Unit, Ext or a run-time Dyn
/// @param part the sub-group or partition
/// @param v this lane's value
/// @param op the operation (default sycl::plus)
/// @param stride the lane stride s
/// @return the fold on the head lane; partials elsewhere
template <int Count, class Part, class T, class Op = sycl::plus<T>, class Stride = Unit>
BATCHLAS_DEVICE_HELPER T shift_reduce(const Part& part, T v, Op op = {}, Stride stride = {}) {
    static_assert(detail::is_pow2(Count), "shift_reduce: Count is a power of two");
#pragma unroll
    for (int d = Count / 2; d >= 1; d >>= 1) v = op(v, shuffle_down(part, v, static_cast<std::uint32_t>(int(stride) * d)));
    return v;
}

namespace detail {
template <int N> constexpr int ilog2() {
    int r = 0;
    for (int v = N; v > 1; v >>= 1) ++r;
    return r;
}
}  // namespace detail

/// @brief COLLECTIVE: the reduce-scatter of K values over the P = Ex::size lanes of a partition
/// executor.
/// @ingroup api_device_blas1
///
/// Lane l returns the sum over all lanes of `x[l / (P / K)]`: log2(K) halving steps over the xor
/// masks P / 2 down to P / K, then an all-reduce over the low log2(P / K) lane bits. Consumes x.
/// @tparam K the number of values, a power of two dividing P
/// @tparam Ex a partition executor (sub_whole)
/// @tparam X `Frag<T, K>` or `T[K]`
/// @param ex the executor
/// @param x the K values; overwritten
/// @return this lane's sum
// The one-sided Jacobi Gram and column-norm passes use it.
template <int K, class Ex, class X> BATCHLAS_DEVICE_INLINE auto reduce_scatter(const Ex& ex, X& x) {
    using T = std::remove_cvref_t<decltype(x[0])>;
    constexpr int P = Ex::size;
    static_assert(Ex::sub_whole && detail::is_pow2(K) && K <= P, "reduce_scatter: K | P on a partition executor");
    constexpr int kSteps = detail::ilog2<K>();
    constexpr int kAll = detail::ilog2<P>();
    const std::uint32_t lane = static_cast<std::uint32_t>(ex.lid());
#pragma unroll
    for (int st = 0; st < kSteps; ++st) {
        const int mask = P >> (st + 1);
        const int half = K >> (st + 1);
        const bool hi = (lane & static_cast<std::uint32_t>(mask)) != 0u;
#pragma unroll
        for (int j = 0; j < half; ++j) {
            // Copy both slots first: `hi ? x[a] : x[b]` is an lvalue, a lane-dependent slot index
            // that turns x into a stack array or compare/select chains (float 8x8 +73% instructions).
            const T lo = x[j];
            const T up = x[j + half];
            const T own = hi ? up : lo;
            const T send = hi ? lo : up;
            x[j] = Arith<T>::add(own, shuffle_xor(ex, send, static_cast<std::uint32_t>(mask)));
        }
    }
#pragma unroll
    for (int st = kSteps; st < kAll; ++st) {
        const std::uint32_t mask = static_cast<std::uint32_t>(P >> (st + 1));
        x[0] = Arith<T>::add(x[0], shuffle_xor(ex, x[0], mask));
    }
    return T(x[0]);
}

namespace detail {
template <class Tie> BATCHLAS_DEVICE_INLINE bool tie_prefers(const Tie& tie, int oi, int bi) {
    if constexpr (std::is_same_v<Tie, alg::TieLowest>) return oi < bi;
    else if constexpr (std::is_same_v<Tie, alg::TieHighest>) return oi > bi;
    else {
        static_assert(std::is_same_v<Tie, alg::TieRt>, "iamax: alg::TieLowest, alg::TieHighest or alg::TieRt");
        return tie.low ? oi < bi : oi > bi;
    }
}
// A 32-bit key compares bitwise: a short-circuit `||` became a branch per butterfly step (getrf
// tiny float: BRA 319 -> 383, +7%). A 64-bit key short-circuits: the equality compare runs on the
// 1:64 FP64 pipe only after a tie (sytrf double, bitwise: 1.016-1.021x, short-circuit <= 1.007x).
template <class Real, class Tie> BATCHLAS_DEVICE_INLINE bool argmax_beats(Real ov, int oi, Real bv, int bi,
                                                                          const Tie& tie) {
    if constexpr (sizeof(Real) > 4) return ov > bv || (ov == bv && tie_prefers(tie, oi, bi));
    else return (ov > bv) | ((ov == bv) & tie_prefers(tie, oi, bi));
}
// a := (ov, oi) when it beats a (members assigned one by one: AdaptiveCpp's Metal backend
// miscompiles a select of the whole struct).
template <class Real, class Tie> BATCHLAS_DEVICE_INLINE void argmax_take(ArgMax<Real>& a, Real ov, int oi,
                                                                         const Tie& tie) {
    if (argmax_beats(ov, oi, a.value, a.index, tie)) {
        a.value = ov;
        a.index = oi;
    }
}
}  // namespace detail
/// @brief The better of two ArgMax candidates: the larger key, or on a tie the index Tie prefers.
/// @ingroup api_device_blas1
///
/// Keys must be NaN-free (iamax maps them by its NaN policy); then the rule is commutative and
/// associative, so every butterfly order gives the same pair. An empty candidate carries the
/// seed key and loses.
/// @tparam Tie alg::TieLowest, alg::TieHighest, or alg::TieRt (chosen at run time)
/// @param a one candidate
/// @param b the other candidate
/// @param tie the tie rule
/// @return the winner
template <class Real, class Tie = alg::TieLowest>
BATCHLAS_DEVICE_INLINE ArgMax<Real> argmax_merge(const ArgMax<Real>& a, const ArgMax<Real>& b, const Tie& tie = {}) {
    ArgMax<Real> r = a;
    detail::argmax_take(r, b.value, b.index, tie);
    return r;
}
/// @brief argmax_merge as a binary operation, for a butterfly or a group reduction.
/// @ingroup api_device_blas1
/// @tparam Tie the tie rule type
template <class Tie = alg::TieLowest> struct ArgMaxMerge {
    Tie tie{};  ///< The tie rule.
    /// @brief The winner of a and b.
    /// @param a one candidate
    /// @param b the other candidate
    /// @return argmax_merge(a, b, tie)
    template <class Real> BATCHLAS_DEVICE_INLINE ArgMax<Real> operator()(const ArgMax<Real>& a,
                                                                         const ArgMax<Real>& b) const {
        return argmax_merge(a, b, tie);
    }
};

/// @brief nteams value and index slots of local memory for a Scope::Group pivot exchange.
/// @ingroup api_device_blas1
///
/// Standalone = false: ONE barrier, and the caller's next barrier precedes the next write to the
/// slots. Standalone = true: self-contained; a second barrier frees the slots on return, and an
/// executor of one team takes neither.
/// @tparam VS the value slots' type (a pointer or local view)
/// @tparam IS the index slots' type
/// @tparam Fence the barriers' fence
/// @tparam Standalone true for the self-contained form
// getrf's B1 uses Standalone = false; sytrf, whose matrix the Global fence orders, uses true.
template <class VS, class IS, Mem Fence = Mem::Local, bool Standalone = false> struct ArgMaxSlots {
    static constexpr Mem fence = Fence;  ///< The barriers' fence.
    static constexpr bool standalone = Standalone;  ///< The Standalone argument.
    VS v;  ///< The value slots, nteams of them.
    IS i;  ///< The index slots, nteams of them.
};
/// @brief The ArgMaxSlots over value slots v and index slots i.
/// @ingroup api_device_blas1
/// @tparam Fence the barriers' fence
/// @tparam Standalone true for the self-contained form
/// @param v the value slots
/// @param i the index slots
/// @return the ArgMaxSlots
template <Mem Fence = Mem::Local, bool Standalone = false, class VS, class IS>
BATCHLAS_DEVICE_INLINE ArgMaxSlots<VS, IS, Fence, Standalone> argmax_slots(VS v, IS i) {
    return {v, i};
}
/// @brief True when S is an ArgMaxSlots.
/// @ingroup api_device_blas1
/// @tparam S the type to test
template <class S> inline constexpr bool is_argmax_slots_v = false;
/// @cond
template <class VS, class IS, Mem F, bool St> inline constexpr bool is_argmax_slots_v<ArgMaxSlots<VS, IS, F, St>> = true;
/// @endcond

/// @brief COLLECTIVE: every item's candidate a to the executor's winner, uniform.
/// @ingroup api_device_blas1
///
/// The team folds by an xor butterfly: a Partition over its P lanes (masks 1 up to P / 2, the tiny
/// tiers' order), a sub-group over its `lanes` (masks lanes / 2 down to 1). At Scope::Group lane 0
/// of team t then writes slots[t], ONE barrier, and every item scans the slots in team order from
/// {none, seed}; with no slots (None) two group collectives fold it, the key and then the index.
/// @tparam Tie the tie rule type
/// @tparam S an ArgMaxSlots, or None
/// @tparam W None (the team size), or int (a run-time executor's width, read once by the caller)
/// @param ex the executor
/// @param a this item's candidate
/// @param tie the tie rule
/// @param slots the local slots of a Scope::Group exchange, or None
/// @param none the index of no candidate
/// @param seed the key of no candidate
/// @param lanes the sub-group width to fold over (W = int)
/// @return the winner, on every item
template <class Ex, class Real, class Tie, class S, class W = None>
BATCHLAS_DEVICE_INLINE ArgMax<Real> argmax_allreduce(const Ex& ex, ArgMax<Real> a, const Tie& tie, S slots, int none, Real seed, W lanes = {}) {
    if constexpr (requires { ex.part; }) {
        constexpr int P = Ex::team_size;
#pragma unroll
        for (std::uint32_t mk = 1u; mk < static_cast<std::uint32_t>(P); mk <<= 1) {
            const Real ov = detail::xor_shuffle(ex.part, a.value, mk);
            const int oi = detail::xor_shuffle(ex.part, a.index, mk);
            detail::argmax_take(a, ov, oi, tie);
        }
        return a;
    } else if constexpr (is_thread_v<Ex>) {
        return a;
    } else {
        const auto sg = ex.sub();
        int w;
        if constexpr (std::is_same_v<W, None>) w = detail::team_size_of(ex);
        else w = lanes;
        for (std::uint32_t off = static_cast<std::uint32_t>(w) / 2u; off > 0u; off >>= 1) {
            const Real ov = detail::xor_shuffle(sg, a.value, off);
            const int oi = detail::xor_shuffle(sg, a.index, off);
            detail::argmax_take(a, ov, oi, tie);
        }
        if constexpr (Ex::natural_scope == Scope::Sub) {
            return a;
        } else if constexpr (is_argmax_slots_v<S>) {
            if constexpr (S::standalone) {
                if (detail::nteams_of(ex) == 1) return a;
            }
            if (ex.lane() == 0) {
                slots.v[static_cast<std::size_t>(ex.team())] = a.value;
                slots.i[static_cast<std::size_t>(ex.team())] = a.index;
            }
            ex.barrier(S::fence);
            ArgMax<Real> r{none, seed};
            const int nt = detail::nteams_of(ex);
            for (int t = 0; t < nt; ++t) {  // the legacy statements: ptxas unrolls this scan by them
                const Real v = slots.v[static_cast<std::size_t>(t)];
                const int ii = slots.i[static_cast<std::size_t>(t)];
                detail::argmax_take(r, v, ii, tie);
            }
            if constexpr (S::standalone) ex.barrier(S::fence);
            return r;
        } else {
            const Real m = ex.group_reduce(a.value, sycl::maximum<Real>());
            bool low;
            if constexpr (std::is_same_v<Tie, alg::TieRt>) low = tie.low;
            else low = std::is_same_v<Tie, alg::TieLowest>;
            const int key = a.value == m ? (low ? a.index : -a.index) : INT_MAX;
            const int k = ex.group_reduce(key, sycl::minimum<int>());
            return {k == INT_MAX ? none : (low ? k : -k), m};
        }
    }
}

/// @brief The minimum offered by any item of a work-group: one local int, a relaxed atomic.
/// @ingroup api_device_blas1
///
/// The leader calls reset(), then a barrier, offer() from any item, a barrier, then value()
/// (INT_MAX if nothing was offered; or_zero(): 0).
struct AtomicMinSlot {
    int* p;  ///< The local int.
    /// @brief The leader sets the slot to INT_MAX.
    /// @param ex the executor
    template <class Ex> BATCHLAS_DEVICE_INLINE void reset(const Ex& ex) const {
        if (ex.leader()) *p = INT_MAX;
    }
    /// @brief Offers v: the slot keeps the minimum.
    /// @param v the value
    BATCHLAS_DEVICE_INLINE void offer(int v) const {
        sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::work_group, sycl::access::address_space::local_space>(*p)
            .fetch_min(v);
    }
    /// @brief The minimum offered, INT_MAX if none.
    BATCHLAS_DEVICE_INLINE int value() const { return *p; }
    /// @brief The minimum offered, 0 if none.
    BATCHLAS_DEVICE_INLINE int or_zero() const {
        const int v = *p;
        return v == INT_MAX ? 0 : v;
    }
};
/// @brief The AtomicMinSlot over the local int at p.
/// @ingroup api_device_blas1
/// @param p a local int
/// @return the slot
BATCHLAS_DEVICE_INLINE AtomicMinSlot atomic_min_slot(int* p) { return {p}; }

}  // namespace batchlas::device
