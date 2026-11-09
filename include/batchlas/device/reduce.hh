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

namespace batchlas::device::inline v2 {

namespace tie {
struct Lowest {};
struct Highest {};
}  // namespace tie
namespace nan {
struct First {};
struct Skip {};
struct Propagate {};
}  // namespace nan

// {index, value} of a maximum; index < 0 is "no candidate" (an item that owned nothing).
template <class R> struct ArgMax {
    int index;
    R value;
};

namespace detail {
template <class Part, class R> BATCHLAS_DLA_INLINE R shfl_xor_word(const Part& part, R v, std::uint32_t m) {
    return xor_shuffle(part, v, m);  // exec.hh: portable::permute_group_by_xor, or ADL into SubGroupPartition
}
template <class Part, class R> BATCHLAS_DLA_INLINE R shfl_down_word(const Part& part, R v, std::uint32_t d) {
    if constexpr (std::is_same_v<Part, sycl::sub_group>) return sycl::shift_group_left(part, v, d);
    else return shift_group_left(part, v, d);  // ADL
}
template <class T> inline constexpr bool is_argmax_v = false;
template <class R> inline constexpr bool is_argmax_v<ArgMax<R>> = true;
}  // namespace detail

// v of lane (lane ^ m). Complex values and ArgMax pairs move member by member (a shuffle is
// one 32/64-bit word per call). `part` is a sycl::sub_group, a SubGroupPartition, or an executor
// holding one (Partition: its .part), as the Jacobi kernels call it.
template <class Part, class T> BATCHLAS_DLA_INLINE T shuffle_xor(const Part& part, T v, std::uint32_t m) {
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
template <class Part, class T> BATCHLAS_DLA_INLINE T shuffle_down(const Part& part, T v, std::uint32_t d) {
    if constexpr (Arith<T>::is_complex) {
        using A = Arith<T>;
        return A::make(detail::shfl_down_word(part, A::re(v), d), detail::shfl_down_word(part, A::im(v), d));
    } else {
        return detail::shfl_down_word(part, v, d);
    }
}

// xor butterfly over the masks Lo, 2 Lo, .., Hi / 2 (powers of two): every lane of each aligned
// Hi-lane block ends with op over the Hi / Lo values whose lane agrees with it outside [Lo, Hi).
// Order bfly::Down applies masks Hi/2 .. Lo, bfly::Asc Lo .. Hi/2 (another evaluation order,
// so a strict port names its hand kernel's). Op is called op(mine, theirs) and must be
// associative; for a commutative op every lane gets the identical value.
// bfly::inline_body is the same body always_inline, for an op that replaces an always_inline hand
// butterfly (inline policy, core.hh); butterfly forwards to it.
namespace bfly {
template <int Lo, int Hi, class Order = Down, class Part, class T, class Op>
BATCHLAS_DLA_INLINE T inline_body(const Part& part, T v, Op op) {
    static_assert(detail::is_pow2(Lo) && detail::is_pow2(Hi) && Lo <= Hi, "butterfly: Lo <= Hi, powers of two");
    static_assert(std::is_same_v<Order, bfly::Down> || std::is_same_v<Order, bfly::Asc>, "butterfly: bfly::Down or bfly::Asc");
    if constexpr (std::is_same_v<Order, bfly::Down>) {
#pragma unroll
        for (int m = Hi / 2; m >= Lo; m >>= 1) v = op(v, shuffle_xor(part, v, static_cast<std::uint32_t>(m)));
    } else {
#pragma unroll
        for (int m = Lo; m < Hi; m <<= 1) v = op(v, shuffle_xor(part, v, static_cast<std::uint32_t>(m)));
    }
    return v;
}
}  // namespace bfly
template <int Lo, int Hi, class Order = bfly::Down, class Part, class T, class Op>
BATCHLAS_DLA_HELPER T butterfly(const Part& part, T v, Op op) {
    return bfly::inline_body<Lo, Hi, Order>(part, v, op);
}
// The same over a run-time, uniform power-of-two width hi (masks hi/2 .. lo, or lo .. hi/2),
// lo >= 1. always_inline, unsigned masks: the hand form it replaced (plain inline, cdouble
// geqr2 resident: +4 REG).
template <class Order = bfly::Down, class Part, class T, class Op>
BATCHLAS_DLA_INLINE T butterfly(const Part& part, T v, Op op, int lo, int hi) {
    const auto ulo = static_cast<std::uint32_t>(lo), uhi = static_cast<std::uint32_t>(hi);
    if constexpr (std::is_same_v<Order, bfly::Down>) {
        for (std::uint32_t m = uhi / 2; m >= ulo; m >>= 1) v = op(v, shuffle_xor(part, v, m));
    } else {
        for (std::uint32_t m = ulo; m < uhi; m <<= 1) v = op(v, shuffle_xor(part, v, m));
    }
    return v;
}

// Shift fold of Count lanes: lane 0 of each aligned Count-lane block ends with
// op(v0, op(v1, ..)) in the tree order Count/2 .. 1; other lanes hold partial values (the
// result is LEADER-ONLY: broadcast it if every lane needs it). Count <= the partition width.
template <int Count, class Part, class T, class Op = sycl::plus<T>>
BATCHLAS_DLA_HELPER T shift_reduce(const Part& part, T v, Op op = {}) {
    static_assert(detail::is_pow2(Count), "shift_reduce: Count is a power of two");
#pragma unroll
    for (int d = Count / 2; d >= 1; d >>= 1) v = op(v, shuffle_down(part, v, static_cast<std::uint32_t>(d)));
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
template <int K, class Ex, class X> BATCHLAS_DLA_INLINE auto reduce_scatter(const Ex& ex, X& x) {
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

// The better of two ArgMax candidates under one tie and NaN rule (the merge of every argmax
// reduction): an empty candidate (index < 0) loses; nan::First: a NaN value wins and among
// NaNs the lower index; nan::Skip: a NaN value never beats a number; otherwise the larger value,
// equal values by Tie (tie::Lowest: lower index, tie::Highest: higher index). Commutative and
// associative, so every butterfly order gives the same pair.
template <class Tie = tie::Lowest, class NaN = nan::First, class R>
BATCHLAS_DLA_INLINE ArgMax<R> argmax_merge(const ArgMax<R>& a, const ArgMax<R>& b) {
    static_assert(std::is_same_v<Tie, tie::Lowest> || std::is_same_v<Tie, tie::Highest>, "argmax_merge: tie::Lowest or tie::Highest");
    static_assert(std::is_same_v<NaN, nan::First> || std::is_same_v<NaN, nan::Skip>, "argmax_merge: nan::First or nan::Skip");
    constexpr bool low = std::is_same_v<Tie, tie::Lowest>;
    if (b.index < 0) return a;
    if (a.index < 0) return b;
    const bool an = sycl::isnan(a.value), bn = sycl::isnan(b.value);
    if (an || bn) {
        if constexpr (std::is_same_v<NaN, nan::First>) {
            if (an && bn) return a.index <= b.index ? a : b;
            return an ? a : b;
        } else {
            if (an && bn) return a.index <= b.index ? a : b;
            return an ? b : a;
        }
    }
    if (a.value != b.value) return a.value > b.value ? a : b;
    if constexpr (low) return a.index <= b.index ? a : b;
    else return a.index >= b.index ? a : b;
}

template <class Tie = tie::Lowest, class NaN = nan::First> struct ArgMaxMerge {
    template <class R> BATCHLAS_DLA_INLINE ArgMax<R> operator()(const ArgMax<R>& a, const ArgMax<R>& b) const {
        return argmax_merge<Tie, NaN>(a, b);
    }
};

// op over every item of a WorkGroup<WG, SG> (any op, any T), uniform result: a butterfly per
// sub-group, team leaders write Scratch (scratch_elements<op::Reduce, Ex, T>() slots), one
// barrier, then every item folds the nteams entries in team order. trailing_barrier = false
// skips the barrier that lets the caller reuse the scratch at once (pass it when the next
// write to the scratch is behind another barrier anyway). Sub-scope executors (SubGroup,
// Partition, Teams) take the butterfly alone and ignore the scratch.
template <class Ex, class T, int Slots, class Op>
BATCHLAS_DLA_HELPER T allreduce(const Ex& ex, Scratch<T, Slots> scr, T v, Op op, bool trailing_barrier = true) {
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
    template <class Ex> BATCHLAS_DLA_INLINE void reset(const Ex& ex) const {
        if (ex.leader()) *p = INT_MAX;
    }
    BATCHLAS_DLA_INLINE void offer(int v) const {
        sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::work_group, sycl::access::address_space::local_space>(*p)
            .fetch_min(v);
    }
    BATCHLAS_DLA_INLINE int value() const { return *p; }
    BATCHLAS_DLA_INLINE int or_zero() const {
        const int v = *p;
        return v == INT_MAX ? 0 : v;
    }
};
BATCHLAS_DLA_INLINE AtomicMinSlot atomic_min_slot(int* p) { return {p}; }

}  // namespace batchlas::device
