// SPDX-License-Identifier: MIT
#pragma once
/** @file
 * @brief Device executors: the work-item teams that own one matrix, and the barriers over them.
 * @ingroup api_device_exec
 *
 * The size of every executor is a type constant (RtGroup's is a launch value), so loop strides,
 * trip counts and team counts fold at compile time. Only WorkGroup, RtGroup and Grid2D have a work-group
 * barrier. Common interface: the constants `size` (the work-items sharing a Cyclic operand; item
 * lid owns index i when i mod size == lid), `team_size`, `nteams` and `sub_whole` (true when one
 * team is the whole executor, so a Scope::Sub result is uniform), and the members `lid()`,
 * `lane()`, `team()`, `leader()`, `team_reduce(v, op)` and `group_reduce(v, op)`. A team is the
 * unit a Scope::Sub reduction spans: a sub-group, a partition or one item.
 *
 * Lane movement on every launch executor: `bcast(v, src)` (collective, src uniform: item src's v
 * to every item), `uniform(v)` (`bcast(v, 0)`), `shuffle(v, src)` (collective, src per item: v of
 * lane src of the item's team) and `shift(v, d)` (collective: v of lane `lane + d` of the item's
 * team, unspecified past its end). On WorkGroup and `RtGroup<Scope::Group>` shuffle and shift act
 * inside the item's sub-group; on Thread all four return v. The index-only executors (Slot,
 * ElemAt, Grid1D) have none.
 *
 * Each launch executor (WorkGroup, Teams, SubGroup, Partition, RtGroup, Thread) is built from the
 * kernel's item as `Ex ex{it}` (`launch<R, Tier, T>::exec`). Fit and the index-only executors
 * wrap an executor or a caller-mapped index instead.
 */

#include <cstdint>
#include <type_traits>

#include <sycl/sycl.hpp>

#include <batchlas/device/core.hh>
#include <batchlas/util/group-collectives.hh>

namespace batchlas::device {

template <class V> struct Published;

namespace detail {

template <class T, class G, class Op> BATCHLAS_DEVICE_INLINE T sycl_reduce(const G& g, T v, Op op) {
    using A = Arith<T>;
    if constexpr (A::is_complex) {
        using R = typename A::real_t;
        static_assert(std::is_same_v<Op, sycl::plus<T>>, "reduce: a complex reduction is a sum");
        return A::make(batchlas::portable::reduce_over_group(g, A::re(v), sycl::plus<R>()),
                       batchlas::portable::reduce_over_group(g, A::im(v), sycl::plus<R>()));
    } else {
        return batchlas::portable::reduce_over_group(g, v, op);
    }
}

template <class Impl, class R> BATCHLAS_DEVICE_INLINE R xor_shuffle(const Impl& part, R v, std::uint32_t m) {
    if constexpr (sycl::is_group_v<Impl>) return batchlas::portable::permute_group_by_xor(part, v, m);
    else return permute_group_by_xor(part, v, m);  // ADL: SubGroupPartition (detail/sg_partition/)
}

// xor butterfly over P lanes; every lane ends with the same value. O = alg::BflyDown: masks
// P/2 .. 1; alg::BflyAsc: 1 .. P/2 (another summation order, geqrf's tiny reductions).
template <int P, class O = alg::BflyDown, class Impl, class T, class Op> BATCHLAS_DEVICE_INLINE T xor_butterfly(const Impl& part, T v, Op op) {
    using A = Arith<T>;
    static_assert(std::is_same_v<O, alg::BflyDown> || std::is_same_v<O, alg::BflyAsc>, "xor_butterfly: the order is alg::BflyDown or alg::BflyAsc");
    constexpr bool down = std::is_same_v<O, alg::BflyDown>;
    if constexpr (A::is_complex) {
        using R = typename A::real_t;
        R re = A::re(v), im = A::im(v);
        if constexpr (down) {
#pragma unroll
            for (int m = P / 2; m >= 1; m >>= 1) {
                re += xor_shuffle(part, re, static_cast<std::uint32_t>(m));
                im += xor_shuffle(part, im, static_cast<std::uint32_t>(m));
            }
        } else {
#pragma unroll
            for (int m = 1; m < P; m <<= 1) {
                re += xor_shuffle(part, re, static_cast<std::uint32_t>(m));
                im += xor_shuffle(part, im, static_cast<std::uint32_t>(m));
            }
        }
        return A::make(re, im);
    } else {
        if constexpr (down) {
#pragma unroll
            for (int m = P / 2; m >= 1; m >>= 1) v = op(v, xor_shuffle(part, v, static_cast<std::uint32_t>(m)));
        } else {
#pragma unroll
            for (int m = 1; m < P; m <<= 1) v = op(v, xor_shuffle(part, v, static_cast<std::uint32_t>(m)));
        }
        return v;
    }
}

inline constexpr bool is_pow2(int p) { return p > 0 && (p & (p - 1)) == 0; }

// Work-group broadcast of a real word. NVPTX: portable::group_broadcast. Elsewhere an
// OR-reduction of the source's bits (exact for every value, NaN payloads and -0 included):
// DPC++'s CPU device returned another item's value for half the sources of
// sycl::group_broadcast at WG 64..160 under reqd_sub_group_size(32) (2026-11-02).
template <class G, class R> BATCHLAS_DEVICE_INLINE R group_bcast(const G& g, R x, int src, [[maybe_unused]] int lid) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
    return batchlas::portable::group_broadcast(g, x, src);
#else
    using U = std::conditional_t<sizeof(R) == 8, std::uint64_t, std::uint32_t>;
    static_assert(sizeof(R) == sizeof(U), "group_bcast: a 4- or 8-byte word");
    const U u = lid == src ? sycl::bit_cast<U>(x) : U(0);
    return sycl::bit_cast<R>(sycl::reduce_over_group(g, u, sycl::bit_or<U>()));
#endif
}

// f applied to each real word of v (complex: re then im), reassembled.
template <class T, class F> BATCHLAS_DEVICE_INLINE T per_word(T v, F&& f) {
    using A = Arith<T>;
    if constexpr (A::is_complex) return A::make(f(A::re(v)), f(A::im(v)));
    else return f(v);
}
// Lane `src` (any, per lane) and lane + d of a sycl::sub_group or a SubGroupPartition (ADL).
template <class Impl, class R> BATCHLAS_DEVICE_INLINE R select_word(const Impl& g, R v, std::uint32_t src) {
    if constexpr (sycl::is_group_v<Impl>) return sycl::select_from_group(g, v, src);
    else return select_from_group(g, v, src);
}
template <class Impl, class R> BATCHLAS_DEVICE_INLINE R shift_word(const Impl& g, R v, std::uint32_t d) {
    if constexpr (sycl::is_group_v<Impl>) return sycl::shift_group_left(g, v, d);
    else return shift_group_left(g, v, d);
}
// Lane collectives of a team that is the aligned P-lane block of a sub-group (P == SG: the
// sub-group): bcast/shuffle address lanes of the block, shift(v, d) reads lane + d of the
// sub-group (past the block: the next block's value, unspecified for the caller).
template <int P, class T> BATCHLAS_DEVICE_INLINE T block_select(const sycl::sub_group& sg, T v, int src) {
    const auto base = static_cast<std::uint32_t>(sg.get_local_linear_id()) & ~std::uint32_t(P - 1);
    return per_word(v, [&](auto w) { return select_word(sg, w, base + static_cast<std::uint32_t>(src)); });
}
template <class G, class T> BATCHLAS_DEVICE_INLINE T team_shift(const G& g, T v, int d) {
    return per_word(v, [&](auto w) { return shift_word(g, w, static_cast<std::uint32_t>(d)); });
}

}  // namespace detail


/// @brief The whole work-group of WG items, in sub-groups of SG: one matrix per work-group.
/// @ingroup api_device_exec
///
/// The kernel must be launched with local range WG and sub-group size SG; a debug build asserts
/// the local range. A team is one sub-group. This, RtGroup and Grid2D are the only executors with a
/// work-group barrier.
/// @tparam WG the work-group size; SG divides it
/// @tparam SG the sub-group size the kernel is launched with
template <int WG, int SG = 32> struct WorkGroup {
    static_assert(WG % SG == 0, "WorkGroup: SG must divide WG");
    ///@{
    /// @brief `size` = WG items share a Cyclic operand, in `nteams` = WG / SG teams of `team_size`
    /// = `sg_size` = SG lanes.
    static constexpr int size = WG, team_size = SG, nteams = WG / SG, sg_size = SG;
    ///@}
    /// @brief True when the work-group is one sub-group (WG == SG).
    static constexpr bool sub_whole = (WG == SG);
    static constexpr Scope natural_scope = Scope::Group;  ///< Scope::Group: a reduction spans the work-group.
    sycl::nd_item<1> it;  ///< The kernel's item.

    /// @brief This item's index in the work-group, 0 to WG - 1.
    BATCHLAS_DEVICE_INLINE int lid() const { return static_cast<int>(it.get_local_linear_id()); }
    /// @brief This item's lane in its sub-group, 0 to SG - 1.
    BATCHLAS_DEVICE_INLINE int lane() const { return static_cast<int>(it.get_sub_group().get_local_linear_id()); }
    /// @brief This item's sub-group (team) index, 0 to WG / SG - 1.
    BATCHLAS_DEVICE_INLINE int team() const { return lid() / SG; }
    /// @brief True on item 0.
    BATCHLAS_DEVICE_INLINE bool leader() const { return lid() == 0; }
    /// @brief This item's sub-group.
    BATCHLAS_DEVICE_INLINE sycl::sub_group sub() const { return it.get_sub_group(); }

    /// @brief COLLECTIVE: the reduction of v over the item's sub-group, on every lane.
    /// @tparam T the value type; a complex reduction is a sum
    /// @tparam Op the binary operation (default sycl::plus)
    /// @param v this item's value
    /// @param op the operation
    /// @return the reduction, uniform over the sub-group
    template <class T, class Op = sycl::plus<T>> BATCHLAS_DEVICE_INLINE T team_reduce(T v, Op op = {}) const {
        return detail::sycl_reduce(it.get_sub_group(), v, op);
    }
    /// @brief COLLECTIVE: the reduction of v over the work-group, on every item.
    /// @tparam T the value type; a complex reduction is a sum
    /// @tparam Op the binary operation (default sycl::plus)
    /// @param v this item's value
    /// @param op the operation
    /// @return the reduction, uniform over the work-group
    template <class T, class Op = sycl::plus<T>> BATCHLAS_DEVICE_INLINE T group_reduce(T v, Op op = {}) const {
        return detail::sycl_reduce(it.get_group(), v, op);
    }
    /// @brief The work-group barrier.
    /// @param m Mem::Local fences local memory; any other value fences global and local memory
    BATCHLAS_DEVICE_INLINE void barrier(Mem m = Mem::Local) const {
        if (m == Mem::Local) it.barrier(sycl::access::fence_space::local_space);
        else it.barrier(sycl::access::fence_space::global_and_local);
    }
    /// @brief COLLECTIVE: a barrier, then the operand re-typed as readable by every item.
    /// @tparam V the operand type
    /// @param v the operand every item has finished writing
    /// @return v as a Published operand
    template <class V> BATCHLAS_DEVICE_INLINE Published<V> publish(V v) const;
    /// @brief COLLECTIVE, src uniform: item src's v on every item of the work-group.
    /// @param v this item's value (read on item src)
    /// @param src the source item, 0 to WG - 1
    /// @return item src's v
    template <class T> BATCHLAS_DEVICE_HELPER T bcast(T v, int src) const {
        return detail::per_word(v, [&](auto w) { return detail::group_bcast(it.get_group(), w, src, lid()); });
    }
    /// @brief COLLECTIVE: the leader's v on every item, `bcast(v, 0)`.
    /// @param v this item's value (read on item 0)
    /// @return item 0's v
    template <class T> BATCHLAS_DEVICE_HELPER T uniform(T v) const { return bcast(v, 0); }
    /// @brief COLLECTIVE, src per item: v of lane src of the item's sub-group.
    /// @param v this item's value
    /// @param src the source lane, 0 to SG - 1
    /// @return lane src's v
    template <class T> BATCHLAS_DEVICE_INLINE T shuffle(T v, int src) const { return detail::block_select<SG>(it.get_sub_group(), v, src); }
    /// @brief COLLECTIVE: v of lane `lane() + d` of the item's sub-group; unspecified past its end.
    /// @param v this item's value
    /// @param d the lane offset
    /// @return lane `lane() + d`'s v
    template <class T> BATCHLAS_DEVICE_INLINE T shift(T v, int d) const { return detail::team_shift(it.get_sub_group(), v, d); }
};

/// @brief Teams of P lanes working independently inside a work-group of WG items: each team is an
/// executor.
/// @ingroup api_device_exec
///
/// P == SG gives whole sub-groups; P < SG gives aligned P-lane chunks of one sub-group, reduced
/// with xor butterflies of masks below P. Cyclic ops stride by P. There is no work-group barrier
/// by construction: `sync(ex)` is a sub-group barrier that every lane must reach.
/// @tparam WG the work-group size; SG divides it
/// @tparam P the team size: a power of two that divides SG
/// @tparam O the butterfly order, alg::BflyDown (on whole sub-groups:
///         portable::reduce_over_group) or alg::BflyAsc
/// @tparam SG the sub-group size the kernel is launched with
template <int WG, int P = 32, class O = alg::BflyDown, int SG = 32> struct Teams {
    static_assert(detail::is_pow2(P) && SG % P == 0 && WG % SG == 0, "Teams: P a power of two dividing SG, SG dividing WG");
    ///@{
    /// @brief `size` = `team_size` = P lanes per team, `nteams` = WG / P teams per work-group,
    /// `sg_size` = SG.
    static constexpr int size = P, team_size = P, nteams = WG / P, sg_size = SG;
    ///@}
    /// @brief Always true: a team is the executor.
    static constexpr bool sub_whole = true;
    static constexpr Scope natural_scope = Scope::Sub;  ///< Scope::Sub: a reduction spans the team.
    sycl::nd_item<1> it;  ///< The kernel's item.

    /// @brief This item's index in its team, the same as lane().
    BATCHLAS_DEVICE_INLINE int lid() const { return lane(); }
    /// @brief This item's lane in its team, 0 to P - 1.
    BATCHLAS_DEVICE_INLINE int lane() const {
        if constexpr (P == SG) return static_cast<int>(it.get_sub_group().get_local_linear_id());
        else return static_cast<int>(it.get_sub_group().get_local_linear_id()) & (P - 1);
    }
    /// @brief This item's team index in the work-group, 0 to WG / P - 1.
    BATCHLAS_DEVICE_INLINE int team() const { return static_cast<int>(it.get_local_linear_id()) / P; }
    /// @brief True on lane 0 of the team.
    BATCHLAS_DEVICE_INLINE bool leader() const { return lane() == 0; }
    /// @brief This item's sub-group.
    BATCHLAS_DEVICE_INLINE sycl::sub_group sub() const { return it.get_sub_group(); }
    /// @brief COLLECTIVE: the reduction of v over the team, on every lane.
    /// @tparam T the value type; a complex reduction is a sum
    /// @tparam Op the binary operation (default sycl::plus)
    /// @param v this lane's value
    /// @param op the operation
    /// @return the reduction, uniform over the team
    template <class T, class Op = sycl::plus<T>> BATCHLAS_DEVICE_INLINE T team_reduce(T v, Op op = {}) const {
        if constexpr (P == SG && std::is_same_v<O, alg::BflyDown>) return detail::sycl_reduce(it.get_sub_group(), v, op);
        else return detail::xor_butterfly<P, O>(it.get_sub_group(), v, op);
    }
    /// @brief COLLECTIVE: the same as team_reduce (the team is the executor).
    /// @tparam T the value type; a complex reduction is a sum
    /// @tparam Op the binary operation (default sycl::plus)
    /// @param v this lane's value
    /// @param op the operation
    /// @return the reduction, uniform over the team
    template <class T, class Op = sycl::plus<T>> BATCHLAS_DEVICE_INLINE T group_reduce(T v, Op op = {}) const {
        return team_reduce(v, op);
    }
    /// @brief COLLECTIVE, src uniform: lane src's v on every lane of the team.
    /// @param v this lane's value (read on lane src)
    /// @param src the source lane, 0 to P - 1
    /// @return lane src's v
    template <class T> BATCHLAS_DEVICE_INLINE T bcast(T v, int src) const { return detail::block_select<P>(it.get_sub_group(), v, src); }
    /// @brief COLLECTIVE: lane 0's v on every lane, `bcast(v, 0)`.
    /// @param v this lane's value (read on lane 0)
    /// @return lane 0's v
    template <class T> BATCHLAS_DEVICE_INLINE T uniform(T v) const { return bcast(v, 0); }
    /// @brief COLLECTIVE, src per lane: v of lane src of the team.
    /// @param v this lane's value
    /// @param src the source lane, 0 to P - 1
    /// @return lane src's v
    template <class T> BATCHLAS_DEVICE_INLINE T shuffle(T v, int src) const { return detail::block_select<P>(it.get_sub_group(), v, src); }
    /// @brief COLLECTIVE: v of lane `lane() + d` of the sub-group; past the team's end it is the
    /// next team's value, unspecified for the caller.
    /// @param v this lane's value
    /// @param d the lane offset
    /// @return lane `lane() + d`'s v
    template <class T> BATCHLAS_DEVICE_INLINE T shift(T v, int d) const { return detail::team_shift(it.get_sub_group(), v, d); }
};

/// @brief One sub-group of SG lanes as the executor.
/// @ingroup api_device_exec
///
/// A sub-group barrier is the strongest synchronisation it has.
/// @tparam SG the sub-group size the kernel is launched with
template <int SG = 32> struct SubGroup {
    ///@{
    /// @brief `size` = `team_size` = `sg_size` = SG lanes, in `nteams` = 1 team.
    static constexpr int size = SG, team_size = SG, nteams = 1, sg_size = SG;
    ///@}
    /// @brief Always true: the sub-group is one team.
    static constexpr bool sub_whole = true;
    static constexpr Scope natural_scope = Scope::Sub;  ///< Scope::Sub: a reduction spans the sub-group.
    sycl::sub_group sg;  ///< The sub-group.

    /// @brief The executor of sub-group s.
    /// @param s the sub-group
    BATCHLAS_DEVICE_INLINE SubGroup(sycl::sub_group s) : sg(s) {}
    /// @brief The executor of the item's sub-group.
    /// @param it the kernel's item
    BATCHLAS_DEVICE_INLINE explicit SubGroup(sycl::nd_item<1> it) : sg(it.get_sub_group()) {}
    /// @brief The sub-group.
    BATCHLAS_DEVICE_INLINE sycl::sub_group sub() const { return sg; }
    /// @brief This lane's index in the sub-group, 0 to SG - 1.
    BATCHLAS_DEVICE_INLINE int lid() const { return static_cast<int>(sg.get_local_linear_id()); }
    /// @brief The same as lid().
    BATCHLAS_DEVICE_INLINE int lane() const { return lid(); }
    /// @brief Always 0: one team.
    BATCHLAS_DEVICE_INLINE int team() const { return 0; }
    /// @brief True on lane 0.
    BATCHLAS_DEVICE_INLINE bool leader() const { return lid() == 0; }
    /// @brief COLLECTIVE: the reduction of v over the sub-group, on every lane.
    /// @tparam T the value type; a complex reduction is a sum
    /// @tparam Op the binary operation (default sycl::plus)
    /// @param v this lane's value
    /// @param op the operation
    /// @return the reduction, uniform over the sub-group
    template <class T, class Op = sycl::plus<T>> BATCHLAS_DEVICE_INLINE T team_reduce(T v, Op op = {}) const {
        return detail::sycl_reduce(sg, v, op);
    }
    /// @brief COLLECTIVE: the same as team_reduce.
    /// @tparam T the value type; a complex reduction is a sum
    /// @tparam Op the binary operation (default sycl::plus)
    /// @param v this lane's value
    /// @param op the operation
    /// @return the reduction, uniform over the sub-group
    template <class T, class Op = sycl::plus<T>> BATCHLAS_DEVICE_INLINE T group_reduce(T v, Op op = {}) const {
        return team_reduce(v, op);
    }
    /// @brief COLLECTIVE, src uniform: lane src's v on every lane.
    /// @param v this lane's value (read on lane src)
    /// @param src the source lane, 0 to SG - 1
    /// @return lane src's v
    template <class T> BATCHLAS_DEVICE_INLINE T bcast(T v, int src) const { return detail::block_select<SG>(sg, v, src); }
    /// @brief COLLECTIVE: lane 0's v on every lane, `bcast(v, 0)`.
    /// @param v this lane's value (read on lane 0)
    /// @return lane 0's v
    template <class T> BATCHLAS_DEVICE_INLINE T uniform(T v) const { return bcast(v, 0); }
    /// @brief COLLECTIVE, src per lane: v of lane src.
    /// @param v this lane's value
    /// @param src the source lane, 0 to SG - 1
    /// @return lane src's v
    template <class T> BATCHLAS_DEVICE_INLINE T shuffle(T v, int src) const { return detail::block_select<SG>(sg, v, src); }
    /// @brief COLLECTIVE: v of lane `lane() + d`; unspecified past the sub-group's end.
    /// @param v this lane's value
    /// @param d the lane offset
    /// @return lane `lane() + d`'s v
    template <class T> BATCHLAS_DEVICE_INLINE T shift(T v, int d) const { return detail::team_shift(sg, v, d); }
};

/// @brief P consecutive lanes of a sub-group as the executor: one matrix per partition.
/// @ingroup api_device_exec
///
/// With `Impl = sycl::sub_group` (the default) it is the built-in lockstep form: every lane of the
/// sub-group reaches every collective. For masked chunks pass `SubGroupPartition<P, true>`
/// (SgpPartition in sg_partition.hh), whose lanes may diverge from the other partitions.
/// @tparam P the partition size, a power of two up to the sub-group size
/// @tparam Masked true for a masked partition; then Impl must be `SubGroupPartition<P, true>`
/// @tparam Impl the partition object: sycl::sub_group, or a SubGroupPartition
template <int P, bool Masked = false, class Impl = sycl::sub_group> struct Partition {
    static_assert(detail::is_pow2(P), "Partition: P must be a power of two");
    static_assert(!(Masked && std::is_same_v<Impl, sycl::sub_group>),
                  "Partition: a Masked partition needs Impl = SubGroupPartition<P, true>");
    ///@{
    /// @brief `size` = `team_size` = P lanes, in `nteams` = 1 team.
    static constexpr int size = P, team_size = P, nteams = 1;
    ///@}
    /// @brief Always true: the partition is one team.
    static constexpr bool sub_whole = true;
    static constexpr Scope natural_scope = Scope::Sub;  ///< Scope::Sub: a reduction spans the partition.
    Impl part;  ///< The partition object.

    /// @brief The partition of sub-group sg this lane belongs to.
    /// @param sg the sub-group
    BATCHLAS_DEVICE_INLINE explicit Partition(sycl::sub_group sg) : part(sg) {}
    /// @brief The partition of the item's sub-group this lane belongs to.
    /// @param it the kernel's item
    BATCHLAS_DEVICE_INLINE explicit Partition(sycl::nd_item<1> it) : part(it.get_sub_group()) {}
    /// @brief The tag of the adopting constructor.
    struct Wrap {};
    /// @brief Adopts an existing partition object.
    /// @param p the partition object
    BATCHLAS_DEVICE_INLINE Partition(Wrap, Impl p) : part(p) {}
    /// @brief This lane's index in the partition, 0 to P - 1.
    BATCHLAS_DEVICE_INLINE int lid() const {
        if constexpr (std::is_same_v<Impl, sycl::sub_group>) return static_cast<int>(part.get_local_linear_id()) & (P - 1);
        else return static_cast<int>(part.get_local_linear_id());
    }
    /// @brief The same as lid().
    BATCHLAS_DEVICE_INLINE int lane() const { return lid(); }
    /// @brief Always 0: one team.
    BATCHLAS_DEVICE_INLINE int team() const { return 0; }
    /// @brief True on lane 0 of the partition.
    BATCHLAS_DEVICE_INLINE bool leader() const { return lid() == 0; }
    /// @brief COLLECTIVE: the reduction of v over the partition (an xor butterfly), on every lane.
    /// @tparam T the value type; a complex reduction is a sum
    /// @tparam Op the binary operation (default sycl::plus)
    /// @param v this lane's value
    /// @param op the operation
    /// @return the reduction, uniform over the partition
    template <class T, class Op = sycl::plus<T>> BATCHLAS_DEVICE_INLINE T team_reduce(T v, Op op = {}) const {
        return detail::xor_butterfly<P>(part, v, op);
    }
    /// @brief COLLECTIVE: the same as team_reduce.
    /// @tparam T the value type; a complex reduction is a sum
    /// @tparam Op the binary operation (default sycl::plus)
    /// @param v this lane's value
    /// @param op the operation
    /// @return the reduction, uniform over the partition
    template <class T, class Op = sycl::plus<T>> BATCHLAS_DEVICE_INLINE T group_reduce(T v, Op op = {}) const {
        return team_reduce(v, op);
    }
    /// @brief The whole sub-group this partition lives in.
    BATCHLAS_DEVICE_INLINE sycl::sub_group sub() const {
        if constexpr (std::is_same_v<Impl, sycl::sub_group>) return part;
        else return part.sg;
    }
    /// @brief A sub-group barrier (warp sync): every lane of the sub-group must reach it.
    BATCHLAS_DEVICE_INLINE void barrier(Mem = Mem::Local) const { sycl::group_barrier(sub()); }
    /// @brief COLLECTIVE: a sub-group barrier, then the operand re-typed as readable by every lane.
    /// @tparam V the operand type
    /// @param v the operand every lane has finished writing
    /// @return v as a Published operand
    template <class V> BATCHLAS_DEVICE_INLINE Published<V> publish(V v) const;
    /// @brief COLLECTIVE: lane src's v on every lane of the partition.
    ///
    /// src must be uniform over the partition and every lane must reach the call. A complex value
    /// moves as two reals.
    /// @param v this lane's value (read on lane src)
    /// @param src the source lane, 0 to P - 1
    /// @return lane src's v
    // Plain `inline`, not BATCHLAS_DEVICE_INLINE: inlined early, the source's real view lands
    // ahead of the shuffles and complex posv_tiny reschedules; this inlines when the hand
    // helper (tiny_bcast) did.
    template <class T> inline T bcast(T v, std::uint32_t src) const {
        using A = Arith<T>;
        if constexpr (A::is_complex) return A::make(bcast_real(A::re(v), src), bcast_real(A::im(v), src));
        else return bcast_real(v, src);
    }
    /// @brief COLLECTIVE: lane 0's v on every lane, `bcast(v, 0u)`.
    /// @param v this lane's value (read on lane 0)
    /// @return lane 0's v
    template <class T> BATCHLAS_DEVICE_INLINE T uniform(T v) const { return bcast(v, 0u); }
    /// @brief COLLECTIVE, src per lane: v of lane src of the partition.
    /// @param v this lane's value
    /// @param src the source lane, 0 to P - 1
    /// @return lane src's v
    template <class T> BATCHLAS_DEVICE_INLINE T shuffle(T v, int src) const {
        if constexpr (std::is_same_v<Impl, sycl::sub_group>) return detail::block_select<P>(part, v, src);
        else return detail::per_word(v, [&](auto w) { return detail::select_word(part, w, static_cast<std::uint32_t>(src)); });
    }
    /// @brief COLLECTIVE: v of lane `lane() + d` of the partition; unspecified past its end.
    /// @param v this lane's value
    /// @param d the lane offset
    /// @return lane `lane() + d`'s v
    template <class T> BATCHLAS_DEVICE_INLINE T shift(T v, int d) const { return detail::team_shift(part, v, d); }

private:
    template <class R> inline R bcast_real(R v, std::uint32_t src) const {
        if constexpr (std::is_same_v<Impl, sycl::sub_group>) {
            const auto base = static_cast<std::uint32_t>(part.get_local_linear_id()) & ~std::uint32_t(P - 1);
            return sycl::select_from_group(part, v, base + src);
        } else {
            return select_from_group(part, v, src);  // ADL: SubGroupPartition (detail/sg_partition/)
        }
    }
};

/// @brief One work-item that owns whole operands: serial loops, no collectives.
/// @ingroup api_device_exec
///
/// Every reduction and lane exchange returns its argument, and sync is a no-op.
struct Thread {
    ///@{
    /// @brief `size` = `team_size` = `nteams` = 1.
    static constexpr int size = 1, team_size = 1, nteams = 1;
    ///@}
    /// @brief Always true.
    static constexpr bool sub_whole = true;
    static constexpr Scope natural_scope = Scope::Sub;  ///< Scope::Sub, the item itself.
    /// @brief The executor of the calling item.
    constexpr Thread() = default;
    /// @brief The executor of the calling item; the item is not read.
    BATCHLAS_DEVICE_INLINE explicit Thread(sycl::nd_item<1>) {}
    /// @brief Always 0.
    BATCHLAS_DEVICE_INLINE int lid() const { return 0; }
    /// @brief Always 0.
    BATCHLAS_DEVICE_INLINE int lane() const { return 0; }
    /// @brief Always 0.
    BATCHLAS_DEVICE_INLINE int team() const { return 0; }
    /// @brief Always true.
    BATCHLAS_DEVICE_INLINE bool leader() const { return true; }
    /// @brief Returns v: the reduction over one item.
    /// @param v the value
    /// @return v
    template <class T, class Op = sycl::plus<T>> BATCHLAS_DEVICE_INLINE T team_reduce(T v, Op = {}) const { return v; }
    /// @brief Returns v: the reduction over one item.
    /// @param v the value
    /// @return v
    template <class T, class Op = sycl::plus<T>> BATCHLAS_DEVICE_INLINE T group_reduce(T v, Op = {}) const { return v; }
    /// @brief Returns v.
    /// @param v the value
    /// @return v
    template <class T> BATCHLAS_DEVICE_INLINE T bcast(T v, int) const { return v; }
    /// @brief Returns v.
    /// @param v the value
    /// @return v
    template <class T> BATCHLAS_DEVICE_INLINE T uniform(T v) const { return v; }
    /// @brief Returns v.
    /// @param v the value
    /// @return v
    template <class T> BATCHLAS_DEVICE_INLINE T shuffle(T v, int) const { return v; }
    /// @brief Returns v.
    /// @param v the value
    /// @return v
    template <class T> BATCHLAS_DEVICE_INLINE T shift(T v, int) const { return v; }
};

/// @brief A work-group whose size is a launch value (one binary for every work-group size), or one
/// sub-group of it (S = Scope::Sub).
/// @ingroup api_device_exec
///
/// Ids and ranges are read once, at construction. Loops stride by the run-time width(); a team is
/// one sub-group.
/// @tparam S Scope::Group (the work-group) or Scope::Sub (the item's sub-group)
/// @tparam SG the sub-group size the kernel is launched with
// Read per use, the sub-group range is rematerialised per op (+3..4 REG, geqrf); hand kernels
// read them once at the top.
template <Scope S = Scope::Group, int SG = 32> struct RtGroup {
    static_assert(S == Scope::Sub || S == Scope::Group, "RtGroup: Scope::Sub or Scope::Group");
    static constexpr bool runtime = true;  ///< True: the sizes are launch values (is_runtime_v).
    static constexpr bool sub_whole = (S == Scope::Sub);  ///< True for Scope::Sub.
    static constexpr Scope natural_scope = S;  ///< S: a reduction spans the executor's scope.
    static constexpr int sg_size = SG;  ///< The sub-group size the kernel is launched with.
    sycl::nd_item<1> it;  ///< The kernel's item.
    ///@{
    /// @brief The ids and ranges, read at construction; use the accessors.
    int lane_, lid_, width_, team_, nteams_;
    ///@}

    /// @brief The executor of the calling item.
    /// @param i the kernel's item
    BATCHLAS_DEVICE_INLINE explicit RtGroup(sycl::nd_item<1> i)
        : it(i), lane_(static_cast<int>(i.get_sub_group().get_local_linear_id())),
          lid_(S == Scope::Sub ? lane_ : static_cast<int>(i.get_local_linear_id())),
          width_(S == Scope::Sub ? static_cast<int>(i.get_sub_group().get_local_linear_range())
                                 : static_cast<int>(i.get_local_range(0))),
          team_(S == Scope::Sub ? 0 : static_cast<int>(i.get_sub_group().get_group_linear_id())),
          nteams_(S == Scope::Sub ? 1 : static_cast<int>(i.get_sub_group().get_group_linear_range())) {}

    /// @brief This item's index in the executor, 0 to width() - 1.
    BATCHLAS_DEVICE_INLINE int lid() const { return lid_; }
    /// @brief The executor's size: the local range (Scope::Group) or the sub-group's range (Scope::Sub).
    BATCHLAS_DEVICE_INLINE int width() const { return width_; }
    /// @brief This item's lane in its sub-group.
    BATCHLAS_DEVICE_INLINE int lane() const { return lane_; }
    /// @brief This item's sub-group index (0 for Scope::Sub).
    BATCHLAS_DEVICE_INLINE int team() const { return team_; }
    /// @brief The number of sub-groups (1 for Scope::Sub).
    BATCHLAS_DEVICE_INLINE int nteams() const { return nteams_; }
    /// @brief The size of this item's sub-group.
    BATCHLAS_DEVICE_INLINE int team_size() const { return static_cast<int>(it.get_sub_group().get_local_linear_range()); }
    /// @brief True on item 0.
    BATCHLAS_DEVICE_INLINE bool leader() const { return lid_ == 0; }
    /// @brief This item's sub-group.
    BATCHLAS_DEVICE_INLINE sycl::sub_group sub() const { return it.get_sub_group(); }
    /// @brief The sub-group this item belongs to, as its own executor.
    BATCHLAS_DEVICE_INLINE RtGroup<Scope::Sub, SG> team_exec() const { return RtGroup<Scope::Sub, SG>(it); }
    /// @brief COLLECTIVE: the reduction of v over the item's sub-group, on every lane.
    /// @tparam T the value type; a complex reduction is a sum
    /// @tparam Op the binary operation (default sycl::plus)
    /// @param v this item's value
    /// @param op the operation
    /// @return the reduction, uniform over the sub-group
    template <class T, class Op = sycl::plus<T>> BATCHLAS_DEVICE_INLINE T team_reduce(T v, Op op = {}) const {
        return detail::sycl_reduce(it.get_sub_group(), v, op);
    }
    /// @brief COLLECTIVE: the reduction of v over the executor, on every item.
    /// @tparam T the value type; a complex reduction is a sum
    /// @tparam Op the binary operation (default sycl::plus)
    /// @param v this item's value
    /// @param op the operation
    /// @return the reduction, uniform over the executor
    template <class T, class Op = sycl::plus<T>> BATCHLAS_DEVICE_INLINE T group_reduce(T v, Op op = {}) const {
        if constexpr (S == Scope::Sub) return team_reduce(v, op);
        else return detail::sycl_reduce(it.get_group(), v, op);
    }
    /// @brief The executor's barrier: the work-group's (Scope::Group) or the sub-group's (Scope::Sub).
    /// @param m Mem::Local fences local memory; any other value fences global and local memory
    ///        (Scope::Group only)
    BATCHLAS_DEVICE_INLINE void barrier(Mem m = Mem::Local) const {
        if constexpr (S == Scope::Sub) sycl::group_barrier(it.get_sub_group());
        else if (m == Mem::Local) it.barrier(sycl::access::fence_space::local_space);
        else it.barrier(sycl::access::fence_space::global_and_local);
    }
    /// @brief COLLECTIVE: a barrier, then the operand re-typed as readable by every item.
    /// @tparam V the operand type
    /// @param v the operand every item has finished writing
    /// @return v as a Published operand
    template <class V> BATCHLAS_DEVICE_INLINE Published<V> publish(V v) const;
    /// @brief COLLECTIVE, src uniform: item src's v on every item of the executor.
    /// @param v this item's value (read on item src)
    /// @param src the source item, 0 to width() - 1
    /// @return item src's v
    template <class T> BATCHLAS_DEVICE_HELPER T bcast(T v, int src) const {
        using A = Arith<T>;
        auto one = [&](auto x) {
            if constexpr (S == Scope::Sub) return sycl::select_from_group(it.get_sub_group(), x, src);
            else return detail::group_bcast(it.get_group(), x, src, lid_);
        };
        if constexpr (A::is_complex) return A::make(one(A::re(v)), one(A::im(v)));
        else return one(v);
    }
    /// @brief COLLECTIVE: item 0's v on every item, `bcast(v, 0)`.
    /// @param v this item's value (read on item 0)
    /// @return item 0's v
    template <class T> BATCHLAS_DEVICE_HELPER T uniform(T v) const { return bcast(v, 0); }
    /// @brief COLLECTIVE, src per item: v of lane src of the item's sub-group.
    /// @param v this item's value
    /// @param src the source lane
    /// @return lane src's v
    template <class T> BATCHLAS_DEVICE_INLINE T shuffle(T v, int src) const {
        return detail::per_word(v, [&](auto w) { return detail::select_word(it.get_sub_group(), w, static_cast<std::uint32_t>(src)); });
    }
    /// @brief COLLECTIVE: v of lane `lane() + d` of the item's sub-group; unspecified past its end.
    /// @param v this item's value
    /// @param d the lane offset
    /// @return lane `lane() + d`'s v
    template <class T> BATCHLAS_DEVICE_INLINE T shift(T v, int d) const { return detail::team_shift(it.get_sub_group(), v, d); }
};

/// @brief True when Ex's sizes are launch values (RtGroup, `Slot<0>`): read them through the
/// members, not the type constants.
/// @ingroup api_device_exec
/// @tparam Ex the executor type
template <class Ex> inline constexpr bool is_runtime_v = requires { Ex::runtime; };

namespace detail {
// The executor's Cyclic stride, team width and team count: type constants, or launch values
// for RtGroup. Ops read sizes only through these.
template <class Ex> BATCHLAS_DEVICE_INLINE int size_of(const Ex& ex) {
    if constexpr (is_runtime_v<Ex>) return ex.width();
    else return Ex::size;
}
template <class Ex> BATCHLAS_DEVICE_INLINE int team_size_of(const Ex& ex) {
    if constexpr (is_runtime_v<Ex>) return ex.team_size();
    else return Ex::team_size;
}
template <class Ex> BATCHLAS_DEVICE_INLINE int nteams_of(const Ex& ex) {
    if constexpr (is_runtime_v<Ex>) return ex.nteams();
    else return Ex::nteams;
}
}  // namespace detail

/// @brief True when Ex is a WorkGroup.
/// @ingroup api_device_exec
/// @tparam Ex the executor type
template <class Ex> inline constexpr bool is_work_group_v = false;
/// @cond
template <int WG, int SG> inline constexpr bool is_work_group_v<WorkGroup<WG, SG>> = true;
/// @endcond
/// @brief True when Ex is Thread.
/// @ingroup api_device_exec
/// @tparam Ex the executor type
template <class Ex> inline constexpr bool is_thread_v = std::is_same_v<Ex, Thread>;
/// @brief True when Ex is a Teams.
/// @ingroup api_device_exec
/// @tparam Ex the executor type
template <class Ex> inline constexpr bool is_teams_v = false;
/// @cond
template <int WG, int P, class O, int SG> inline constexpr bool is_teams_v<Teams<WG, P, O, SG>> = true;
/// @endcond

/// @brief A barrier over the executor's scope.
/// @ingroup api_device_exec
///
/// A work-group barrier on WorkGroup and `RtGroup<Scope::Group>`, a sub-group barrier on Teams,
/// SubGroup, Partition and `RtGroup<Scope::Sub>`, nothing on Thread.
/// @tparam Ex the executor type
/// @param ex the executor; every item of its scope must reach the call
/// @param m Mem::Local fences local memory; any other value fences global and local memory
template <class Ex> BATCHLAS_DEVICE_INLINE void sync(const Ex& ex, Mem m = Mem::Local) {
    if constexpr (is_thread_v<Ex>) return;
    else if constexpr (requires { ex.barrier(m); }) ex.barrier(m);
    else sycl::group_barrier(ex.sub());
}

/// @brief Scratch for a Scope::Group packed exchange, in local memory the caller allocates.
/// @ingroup api_device_blas1
///
/// Its size is the constexpr `scratch_elements<Op, Ex, T>()`.
/// @tparam T the element type
/// @tparam Slots the number of independent exchanges the scratch holds
template <class T, int Slots> struct Scratch {
    T* p;  ///< The caller's local memory, at least scratch_elements elements.
    static constexpr int slots = Slots;  ///< The Slots template argument.
};

/// @brief The elements of T a Scope::Group packed exchange of Op needs on Ex.
/// @ingroup api_device_blas1
///
/// `nteams` per reduced value on a WorkGroup (K values for `reduce_op::ReduceN<K>`), 0 on every
/// other executor.
/// @tparam Op the reduction tag, `reduce_op::ReduceN<K>`
/// @tparam Ex the executor type
/// @tparam T the element type
/// @return the number of elements
template <class Op, class Ex, class T> constexpr int scratch_elements() {
    if constexpr (is_work_group_v<Ex>) {
        if constexpr (std::is_same_v<Op, reduce_op::ReduceN<1>>) return Ex::nteams;
        else return []<int K>(reduce_op::ReduceN<K>*) { return Ex::nteams * K; }(static_cast<Op*>(nullptr));
    } else {
        return 0;
    }
}

/// @brief A barrier at scope S of the executor (Scope in core.hh).
/// @ingroup api_device_exec
///
/// Scope::Group is `sync(ex, m)`. Scope::Sub is the sub-group barrier (nothing on Thread).
/// Scope::Own is a Partition's own barrier (a SubGroupPartition's masked one, else the
/// sub-group's), the work-group barrier on WorkGroup and `RtGroup<Scope::Group>`, the sub-group's
/// on SubGroup and `RtGroup<Scope::Sub>`, and nothing on Teams (lockstep slices of one sub-group)
/// and Thread. An adaptor (Fit) syncs the executor it wraps.
/// @tparam S the scope
/// @tparam Ex the executor type
/// @param ex the executor; every item of the scope must reach the call
/// @param m Mem::Local fences local memory; any other value fences global and local memory
template <Scope S, class Ex> BATCHLAS_DEVICE_INLINE void sync(const Ex& ex, Mem m = Mem::Local) {
    if constexpr (requires { ex.ex; }) {
        sync<S>(ex.ex, m);  // an adaptor (Fit) syncs its executor
    } else if constexpr (is_thread_v<Ex>) {
        return;
    } else if constexpr (S == Scope::Group) {
        sync(ex, m);
    } else if constexpr (S == Scope::Sub) {
        sycl::group_barrier(ex.sub());
    } else if constexpr (requires { ex.part; }) {
        if constexpr (sycl::is_group_v<std::remove_cvref_t<decltype(ex.part)>>) sycl::group_barrier(ex.part);
        else group_barrier(ex.part);  // ADL: SubGroupPartition's own barrier
    } else if constexpr (is_teams_v<Ex>) {
        return;  // lockstep slices of one sub-group
    } else if constexpr (is_work_group_v<Ex> || is_runtime_v<Ex>) {
        ex.barrier(m);  // RtGroup<Sub>::barrier is the sub-group's
    } else {
        sycl::group_barrier(ex.sub());
    }
}

/// @brief Item id of a caller-mapped slot of N items: an index-only executor, no collectives.
/// @ingroup api_device_exec
///
/// Cyclic ops stride by N and there is one team; no member exchanges values. `Slot<0>` is the
/// run-time width form.
/// @tparam N the slot width, at least 1 (0: the run-time width form)
template <int N = 0> struct Slot {
    static_assert(N >= 1, "Slot<N>: N >= 1 (Slot<0> is the run-time width form)");
    ///@{
    /// @brief `size` = `team_size` = N, `nteams` = 1.
    static constexpr int size = N, team_size = N, nteams = 1;
    ///@}
    /// @brief Always true.
    static constexpr bool sub_whole = true;
    static constexpr Scope natural_scope = Scope::Sub;  ///< Scope::Sub.
    int id;  ///< This item's index in the slot, 0 to N - 1.
    /// @brief This item's index, id.
    BATCHLAS_DEVICE_INLINE int lid() const { return id; }
    /// @brief The same as lid().
    BATCHLAS_DEVICE_INLINE int lane() const { return id; }
    /// @brief Always 0.
    BATCHLAS_DEVICE_INLINE int team() const { return 0; }
    /// @brief True for id 0.
    BATCHLAS_DEVICE_INLINE bool leader() const { return id == 0; }
};
/// @brief Item id of a caller-mapped slot of a run-time width n: an index-only executor.
/// @ingroup api_device_exec
template <> struct Slot<0> {
    static constexpr bool runtime = true;  ///< True: the width is a launch value.
    static constexpr bool sub_whole = true;  ///< Always true.
    static constexpr Scope natural_scope = Scope::Sub;  ///< Scope::Sub.
    int id;  ///< This item's index in the slot, 0 to n - 1.
    int n;  ///< The slot width.
    /// @brief This item's index, id.
    BATCHLAS_DEVICE_INLINE int lid() const { return id; }
    /// @brief The same as lid().
    BATCHLAS_DEVICE_INLINE int lane() const { return id; }
    /// @brief The slot width, n.
    BATCHLAS_DEVICE_INLINE int width() const { return n; }
    /// @brief The slot width, n.
    BATCHLAS_DEVICE_INLINE int team_size() const { return n; }
    /// @brief Always 1.
    BATCHLAS_DEVICE_INLINE int nteams() const { return 1; }
    /// @brief Always 0.
    BATCHLAS_DEVICE_INLINE int team() const { return 0; }
    /// @brief True for id 0.
    BATCHLAS_DEVICE_INLINE bool leader() const { return id == 0; }
};
/// @brief One element of a 2-D launch addressed in a block at (r0, c0), as `ElemAt::sub` returns
/// it.
/// @ingroup api_device_exec
struct ElemIn {
    ///@{
    /// @brief The launch's element (r, c).
    int r, c;
    ///@}
    ///@{
    /// @brief The origin (r0, c0) of the block the op addresses.
    int r0, c0;
    ///@}
};
/// @brief One element (r, c) of a 2-D launch: for_each_owned_2d visits it alone.
/// @ingroup api_device_exec
///
/// The visit is guarded by the op's extent unless Exact. `sub(r0, c0)` is the same item addressing
/// the block at (r0, c0); its columns are tested first.
/// @tparam Exact true when the launch covers the extent exactly, so there is no bounds test
template <bool Exact = false> struct ElemAt {
    ///@{
    /// @brief The launch's element (r, c).
    int r, c;
    ///@}
    /// @brief The element's row, r.
    BATCHLAS_DEVICE_INLINE int row() const { return r; }
    /// @brief The element's column, c.
    BATCHLAS_DEVICE_INLINE int col() const { return c; }
    /// @brief The same element, addressing the block at (r0, c0).
    /// @param r0 the block's first row
    /// @param c0 the block's first column
    /// @return the ElemIn
    BATCHLAS_DEVICE_INLINE ElemIn sub(int r0, int c0) const { return {r, c, r0, c0}; }
};
/// @cond
ElemAt(int, int) -> ElemAt<false>;
/// @endcond
/// @brief Element i of a 1-D launch: for_each_owned calls f(i), guarded by `i < n` unless Exact.
/// @ingroup api_device_exec
/// @tparam Exact true when the launch covers the extent exactly, so there is no bounds test
template <bool Exact = false> struct Grid1D {
    int i;  ///< The launch's element.
};
/// @cond
Grid1D(int) -> Grid1D<false>;
/// @endcond

/// @brief Ex restricted to one index per item: item lid owns index lid, for n <= size.
/// @ingroup api_device_exec
///
/// for_each_owned on a Fit is one guarded call, no loop. Collectives and barriers forward to the
/// wrapped executor.
/// @tparam Ex the wrapped executor type
template <class Ex> struct Fit {
    ///@{
    /// @brief The wrapped executor's `size`, `team_size` and `nteams`.
    static constexpr int size = Ex::size, team_size = Ex::team_size, nteams = Ex::nteams;
    ///@}
    /// @brief The wrapped executor's sub_whole.
    static constexpr bool sub_whole = Ex::sub_whole;
    static constexpr Scope natural_scope = Ex::natural_scope;  ///< The wrapped executor's natural_scope.
    Ex ex;  ///< The wrapped executor.
    /// @brief The wrapped executor's lid().
    BATCHLAS_DEVICE_INLINE int lid() const { return ex.lid(); }
    /// @brief The wrapped executor's lane().
    BATCHLAS_DEVICE_INLINE int lane() const { return ex.lane(); }
    /// @brief The wrapped executor's team().
    BATCHLAS_DEVICE_INLINE int team() const { return ex.team(); }
    /// @brief The wrapped executor's leader().
    BATCHLAS_DEVICE_INLINE bool leader() const { return ex.leader(); }
    /// @brief The wrapped executor's sub().
    BATCHLAS_DEVICE_INLINE auto sub() const { return ex.sub(); }
    /// @brief COLLECTIVE: the wrapped executor's team_reduce.
    /// @param v this item's value
    /// @param op the operation
    /// @return the reduction over the item's team
    template <class T, class Op = sycl::plus<T>> BATCHLAS_DEVICE_INLINE T team_reduce(T v, Op op = {}) const { return ex.team_reduce(v, op); }
    /// @brief COLLECTIVE: the wrapped executor's group_reduce.
    /// @param v this item's value
    /// @param op the operation
    /// @return the reduction over the executor
    template <class T, class Op = sycl::plus<T>> BATCHLAS_DEVICE_INLINE T group_reduce(T v, Op op = {}) const { return ex.group_reduce(v, op); }
    /// @brief COLLECTIVE: the wrapped executor's bcast.
    /// @param v this item's value
    /// @param src the source item
    /// @return item src's v
    template <class T, class S> BATCHLAS_DEVICE_INLINE T bcast(T v, S src) const { return ex.bcast(v, src); }
    /// @brief COLLECTIVE: the wrapped executor's uniform.
    /// @param v this item's value
    /// @return the leader's v
    template <class T> BATCHLAS_DEVICE_INLINE T uniform(T v) const { return ex.uniform(v); }
    /// @brief COLLECTIVE: the wrapped executor's shuffle.
    /// @param v this item's value
    /// @param src the source lane
    /// @return lane src's v
    template <class T> BATCHLAS_DEVICE_INLINE T shuffle(T v, int src) const { return ex.shuffle(v, src); }
    /// @brief COLLECTIVE: the wrapped executor's shift.
    /// @param v this item's value
    /// @param d the lane offset
    /// @return lane `lane() + d`'s v
    template <class T> BATCHLAS_DEVICE_INLINE T shift(T v, int d) const { return ex.shift(v, d); }
};
/// @cond
template <class Ex> Fit(Ex) -> Fit<Ex>;
/// @endcond
/// @brief True when Ex is a Fit.
/// @ingroup api_device_exec
/// @tparam Ex the executor type
template <class Ex> inline constexpr bool is_fit_v = false;
/// @cond
template <class Ex> inline constexpr bool is_fit_v<Fit<Ex>> = true;
/// @endcond

// The broadcast policies of the unblocked factorizations (potf2, getf2): how a pivot column or row
// reaches the lanes that update with it.
namespace alg {
/// @brief One indexed shuffle per element: no memory, no barrier.
/// @ingroup api_device_lowlevel
struct BcastShuffle {};
/// @brief Every lane stores its element in local memory, one sub-group barrier, then vector loads.
/// @ingroup api_device_lowlevel
///
/// Double-buffered by column parity.
/// @tparam Mpw the partitions per sub-group
/// @tparam Acc the local memory accessor or pointer type
template <int Mpw, class Acc> struct BcastSlm {
    static constexpr int mpw = Mpw;  ///< The Mpw template argument.
    Acc slm;  ///< The local memory.
    int pidx;  ///< This partition's index in its sub-group, 0 to Mpw - 1.
    /// @brief The first unit of this partition's buffer for column j.
    /// @param j the column
    /// @param nv the units per column
    /// @return the offset, in units, into slm
    BATCHLAS_DEVICE_INLINE int buffer(int j, int nv) const { return ((j & 1) * Mpw + pidx) * nv; }
};
}  // namespace alg

}  // namespace batchlas::device
