// SPDX-License-Identifier: MIT
#pragma once
// Device LA reductions and lane exchanges (v22 design A.2 reduce.hh): one complex-aware xor
// shuffle, ONE butterfly (any mask range, either order, any associative op), the leader-only
// shift fold, the reduce-scatter, the ArgMax pair with its merge rule, and the Group-scope
// allreduce of any op through Scratch. A `part` is a sycl::sub_group or a SubGroupPartition (detail/sg_partition/).

#include <climits>
#include <cstdint>
#include <limits>
#include <type_traits>

#include <sycl/sycl.hpp>

#include <batchlas/device/core.hh>
#include <batchlas/device/exec.hh>

namespace batchlas::device {

namespace alg {
struct NanAsInf {};  // iamax: a NaN key ranks as +inf (found, and no compare sees a NaN)
struct NanSkip {};
struct NanPropagate {};
}  // namespace alg

// {index, value} of a maximum; index < 0 is "no candidate" (an item that owned nothing).
template <class R> struct ArgMax {
    int index;
    R value;
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
template <class R> inline constexpr bool is_argmax_v<ArgMax<R>> = true;
}  // namespace detail

// v of lane (lane ^ m). Complex values and ArgMax pairs move member by member (a shuffle is
// one 32/64-bit word per call). `part` is a sycl::sub_group, a SubGroupPartition, or an executor
// holding one (Partition: its .part), as the Jacobi kernels call it.
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

// v of lane (lane + d); past the end: unspecified.
template <class Part, class T> BATCHLAS_DEVICE_INLINE T shuffle_down(const Part& part, T v, std::uint32_t d) {
    if constexpr (Arith<T>::is_complex) {
        using A = Arith<T>;
        return A::make(detail::shfl_down_word(part, A::re(v), d), detail::shfl_down_word(part, A::im(v), d));
    } else {
        return detail::shfl_down_word(part, v, d);
    }
}

// xor butterfly over the masks Lo, 2 Lo, .., Hi / 2 (powers of two): every lane of each aligned
// Hi-lane block ends with op over the Hi / Lo values whose lane agrees with it outside [Lo, Hi).
// Order alg::BflyDown applies masks Hi/2 .. Lo, alg::BflyAsc Lo .. Hi/2 (another evaluation order,
// so a strict port names its hand kernel's). Op is called op(mine, theirs) and must be
// associative; for a commutative op every lane gets the identical value.
// detail::butterfly_inline is the same body always_inline, for an op that replaces an always_inline
// hand butterfly (inline policy, core.hh); butterfly forwards to it.
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
template <int Lo, int Hi, class Order = alg::BflyDown, class Part, class T, class Op>
BATCHLAS_DEVICE_HELPER T butterfly(const Part& part, T v, Op op) {
    return detail::butterfly_inline<Lo, Hi, Order>(part, v, op);
}
// The same over a run-time, uniform power-of-two width hi (masks hi/2 .. lo, or lo .. hi/2),
// lo >= 1. always_inline, unsigned masks: the hand form it replaced (plain inline, cdouble
// geqr2 resident: +4 REG).
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

// Shift fold of lanes {l + s t : t < Count} onto head lane l (stride s: Unit, Ext or a run-time
// Dyn), op(v0, op(v1, ..)) in the tree order Count/2 .. 1; other lanes hold partials (LEADER-ONLY
// result: broadcast it if every lane needs it). Count * s <= the partition width.
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

// Reduce-scatter of K values over the P = Ex::size lanes of a partition executor (K | P, both
// powers of two; x a Frag<T, K> or T[K]): lane l returns the sum over all lanes of x[l / (P / K)].
// log2(K) halving steps over the xor masks P/2 .. P/K, then an all-reduce over the low
// log2(P / K) lane bits. Consumes x. (The one-sided Jacobi Gram and column-norm passes.)
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

// ---- the pivot search's merge and fold (iamax, level1.hh) ------------------------------------
// One rule: (ov, oi) beats (bv, bi) when larger, or equal with the index the Tie prefers
// (alg::TieLowest, alg::TieHighest, or alg::TieRt{low} at run time). Keys reach it NaN-free
// (iamax maps them by its NaN policy), so the rule is commutative and associative and every
// butterfly order gives the same pair. An empty candidate carries the seed key and loses.
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
template <class R, class Tie> BATCHLAS_DEVICE_INLINE bool argmax_beats(R ov, int oi, R bv, int bi, const Tie& tie) {
    if constexpr (sizeof(R) > 4) return ov > bv || (ov == bv && tie_prefers(tie, oi, bi));
    else return (ov > bv) | ((ov == bv) & tie_prefers(tie, oi, bi));
}
// a := (ov, oi) when it beats a (members assigned one by one: AdaptiveCpp's Metal backend
// miscompiles a select of the whole struct).
template <class R, class Tie> BATCHLAS_DEVICE_INLINE void argmax_take(ArgMax<R>& a, R ov, int oi, const Tie& tie) {
    if (argmax_beats(ov, oi, a.value, a.index, tie)) {
        a.value = ov;
        a.index = oi;
    }
}
}  // namespace detail
template <class R, class Tie = alg::TieLowest>
BATCHLAS_DEVICE_INLINE ArgMax<R> argmax_merge(const ArgMax<R>& a, const ArgMax<R>& b, const Tie& tie = {}) {
    ArgMax<R> r = a;
    detail::argmax_take(r, b.value, b.index, tie);
    return r;
}
template <class Tie = alg::TieLowest> struct ArgMaxMerge {
    Tie tie{};
    template <class R> BATCHLAS_DEVICE_INLINE ArgMax<R> operator()(const ArgMax<R>& a, const ArgMax<R>& b) const {
        return argmax_merge(a, b, tie);
    }
};

// nteams value and index slots of local memory for a Group-scope pivot exchange. Fence is the
// barriers'. Standalone = false: ONE barrier, and the caller's next barrier precedes the next
// write to the slots (getrf's B1); true: self-contained, a second barrier frees the slots on
// return and an executor of one team takes neither (sytrf, whose matrix the Global fence orders).
template <class VS, class IS, Mem Fence = Mem::Local, bool Standalone = false> struct ArgMaxSlots {
    static constexpr Mem fence = Fence;
    static constexpr bool standalone = Standalone;
    VS v;
    IS i;
};
template <Mem Fence = Mem::Local, bool Standalone = false, class VS, class IS>
BATCHLAS_DEVICE_INLINE ArgMaxSlots<VS, IS, Fence, Standalone> argmax_slots(VS v, IS i) {
    return {v, i};
}
template <class S> inline constexpr bool is_argmax_slots_v = false;
template <class VS, class IS, Mem F, bool St> inline constexpr bool is_argmax_slots_v<ArgMaxSlots<VS, IS, F, St>> = true;

// Every item's candidate a to the executor's winner, uniform. The team folds by an xor butterfly
// (a Partition over its P lanes, masks 1 .. P/2, the tiny tiers' order; a sub-group over its
// `lanes` (None: team_size_of; a runtime executor's caller passes the width it read once), masks
// lanes/2 .. 1). At Group scope lane 0 of team t then writes slots[t], ONE barrier, and every
// item scans the slots in team order from {none, seed}; with no slots (None) two group
// collectives fold it (the key, then the index).
template <class Ex, class R, class Tie, class S, class W = None>
BATCHLAS_DEVICE_INLINE ArgMax<R> argmax_allreduce(const Ex& ex, ArgMax<R> a, const Tie& tie, S slots, int none, R seed, W lanes = {}) {
    if constexpr (requires { ex.part; }) {
        constexpr int P = Ex::team_size;
#pragma unroll
        for (std::uint32_t mk = 1u; mk < static_cast<std::uint32_t>(P); mk <<= 1) {
            const R ov = detail::xor_shuffle(ex.part, a.value, mk);
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
            const R ov = detail::xor_shuffle(sg, a.value, off);
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
            ArgMax<R> r{none, seed};
            const int nt = detail::nteams_of(ex);
            for (int t = 0; t < nt; ++t) {  // the legacy statements: ptxas unrolls this scan by them
                const R v = slots.v[static_cast<std::size_t>(t)];
                const int ii = slots.i[static_cast<std::size_t>(t)];
                detail::argmax_take(r, v, ii, tie);
            }
            if constexpr (S::standalone) ex.barrier(S::fence);
            return r;
        } else {
            const R m = ex.group_reduce(a.value, sycl::maximum<R>());
            bool low;
            if constexpr (std::is_same_v<Tie, alg::TieRt>) low = tie.low;
            else low = std::is_same_v<Tie, alg::TieLowest>;
            const int key = a.value == m ? (low ? a.index : -a.index) : INT_MAX;
            const int k = ex.group_reduce(key, sycl::minimum<int>());
            return {k == INT_MAX ? none : (low ? k : -k), m};
        }
    }
}

// op over every item of a WorkGroup<WG, SG> (any op, any T), uniform result: a butterfly per
// sub-group, team leaders write Scratch (scratch_elements<reduce_op::Reduce, Ex, T>() slots), one
// barrier, then every item folds the nteams entries in team order. trailing_barrier = false
// skips the barrier that lets the caller reuse the scratch at once (pass it when the next
// write to the scratch is behind another barrier anyway). Sub-scope executors (SubGroup,
// Partition, Teams) take the butterfly alone and ignore the scratch.
template <class Ex, class T, int Slots, class Op>
BATCHLAS_DEVICE_HELPER T allreduce(const Ex& ex, Scratch<T, Slots> scr, T v, Op op, bool trailing_barrier = true) {
    if constexpr (is_work_group_v<Ex>) {
        static_assert(Slots >= Ex::nteams, "allreduce: the scratch holds one slot per sub-group");
        v = butterfly<1, Ex::sg_size>(ex.sub(), v, op);
        if constexpr (Ex::nteams == 1) {
            return v;
        } else {
            if (ex.lane() == 0) scr.p[ex.team()] = v;
            ex.barrier();
            T r = scr.p[0];
#pragma unroll
            for (int t = 1; t < Ex::nteams; ++t) r = op(r, scr.p[t]);
            if (trailing_barrier) ex.barrier();
            return r;
        }
    } else if constexpr (requires { ex.part; }) {
        return butterfly<1, Ex::size>(ex.part, v, op);
    } else if constexpr (is_thread_v<Ex>) {
        return v;
    } else {
        static_assert(Ex::nteams == 1 || Ex::size < Ex::sg_size || Ex::size == Ex::sg_size, "allreduce: unsupported executor");
        return butterfly<1, Ex::size>(ex.sub(), v, op);
    }
}

// The minimum offered by any item of a work-group (one local int, a relaxed atomic): reset() by
// the leader, barrier, offer() from any item, barrier, value() (INT_MAX if none; or_zero(): 0).
struct AtomicMinSlot {
    int* p;
    template <class Ex> BATCHLAS_DEVICE_INLINE void reset(const Ex& ex) const {
        if (ex.leader()) *p = INT_MAX;
    }
    BATCHLAS_DEVICE_INLINE void offer(int v) const {
        sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::work_group, sycl::access::address_space::local_space>(*p)
            .fetch_min(v);
    }
    BATCHLAS_DEVICE_INLINE int value() const { return *p; }
    BATCHLAS_DEVICE_INLINE int or_zero() const {
        const int v = *p;
        return v == INT_MAX ? 0 : v;
    }
};
BATCHLAS_DEVICE_INLINE AtomicMinSlot atomic_min_slot(int* p) { return {p}; }

}  // namespace batchlas::device
