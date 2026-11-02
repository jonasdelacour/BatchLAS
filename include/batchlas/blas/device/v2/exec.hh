// SPDX-License-Identifier: MIT
#pragma once
// Device LA v2 executors. The size of every executor is a type constant, so loop strides,
// trip counts and team counts fold at compile time. Only WorkGroup has a work-group barrier.
//
// Common interface (static): size, team_size, nteams, sub_whole; (members) lid(), lane(),
// team(), leader(), team_reduce(v, op), group_reduce(v, op).
//   size       work-items sharing a Cyclic operand (i owned by lid == i mod size)
//   team       the unit a Scope::Sub reduction spans (a sub-group, a partition, one item)
//   sub_whole  true when one team is the whole executor (Scope::Sub is then uniform)

#include <cstdint>
#include <type_traits>

#include <sycl/sycl.hpp>

#include <batchlas/blas/device/v2/core.hh>
#include <batchlas/util/group-collectives.hh>

namespace batchlas::device::v2 {

template <class V> struct Published;

namespace detail {

template <class T, class G, class Op> BATCHLAS_DLA_INLINE T sycl_reduce(const G& g, T v, Op op) {
    using A = Arith<T>;
    if constexpr (A::is_complex) {
        using R = typename A::real_t;
        static_assert(std::is_same_v<Op, sycl::plus<T>>, "v2: a complex reduction is a sum");
        return A::make(batchlas::portable::reduce_over_group(g, A::re(v), sycl::plus<R>()),
                       batchlas::portable::reduce_over_group(g, A::im(v), sycl::plus<R>()));
    } else {
        return batchlas::portable::reduce_over_group(g, v, op);
    }
}

template <class Impl, class R> BATCHLAS_DLA_INLINE R xor_shuffle(const Impl& part, R v, std::uint32_t m) {
    if constexpr (sycl::is_group_v<Impl>) return batchlas::portable::permute_group_by_xor(part, v, m);
    else return permute_group_by_xor(part, v, m);  // ADL: SubGroupPartition (src/kernels/common)
}

// xor butterfly over P lanes; every lane ends with the same value. O = bfly::Down: masks
// P/2 .. 1; bfly::Asc: 1 .. P/2 (another summation order, geqrf's tiny reductions).
template <int P, class O = bfly::Down, class Impl, class T, class Op> BATCHLAS_DLA_INLINE T butterfly(const Impl& part, T v, Op op) {
    using A = Arith<T>;
    static_assert(std::is_same_v<O, bfly::Down> || std::is_same_v<O, bfly::Asc>, "v2: butterfly order is bfly::Down or bfly::Asc");
    constexpr bool down = std::is_same_v<O, bfly::Down>;
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
template <class G, class R> BATCHLAS_DLA_INLINE R group_bcast(const G& g, R x, int src, [[maybe_unused]] int lid) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
    return batchlas::portable::group_broadcast(g, x, src);
#else
    using U = std::conditional_t<sizeof(R) == 8, std::uint64_t, std::uint32_t>;
    static_assert(sizeof(R) == sizeof(U), "v2::group_bcast: a 4- or 8-byte word");
    const U u = lid == src ? sycl::bit_cast<U>(x) : U(0);
    return sycl::bit_cast<R>(sycl::reduce_over_group(g, u, sycl::bit_or<U>()));
#endif
}

}  // namespace detail

// Whole work-group of WG items in sub-groups of SG. The kernel must be launched with local
// range WG (and sub-group size SG); a debug build asserts the local range.
template <int WG, int SG = 32> struct WorkGroup {
    static_assert(WG % SG == 0, "v2::WorkGroup: SG must divide WG");
    static constexpr int size = WG, team_size = SG, nteams = WG / SG, sg_size = SG;
    static constexpr bool sub_whole = (WG == SG);
    static constexpr Scope natural_scope = Scope::Group;
    sycl::nd_item<1> it;

    BATCHLAS_DLA_INLINE int lid() const { return static_cast<int>(it.get_local_linear_id()); }
    BATCHLAS_DLA_INLINE int lane() const { return static_cast<int>(it.get_sub_group().get_local_linear_id()); }
    BATCHLAS_DLA_INLINE int team() const { return lid() / SG; }
    BATCHLAS_DLA_INLINE bool leader() const { return lid() == 0; }
    BATCHLAS_DLA_INLINE sycl::sub_group sub() const { return it.get_sub_group(); }

    template <class T, class Op = sycl::plus<T>> BATCHLAS_DLA_INLINE T team_reduce(T v, Op op = {}) const {
        return detail::sycl_reduce(it.get_sub_group(), v, op);
    }
    template <class T, class Op = sycl::plus<T>> BATCHLAS_DLA_INLINE T group_reduce(T v, Op op = {}) const {
        return detail::sycl_reduce(it.get_group(), v, op);
    }
    // The only work-group barrier in v2.
    BATCHLAS_DLA_INLINE void barrier(Mem m = Mem::Local) const {
        if (m == Mem::Local) it.barrier(sycl::access::fence_space::local_space);
        else it.barrier(sycl::access::fence_space::global_and_local);
    }
    // Barrier + type change: the operand is now readable by every item (D4).
    template <class V> BATCHLAS_DLA_INLINE Published<V> publish(V v) const;
};

// Teams of P lanes working independently inside a WG launch (P == SG: whole sub-groups; P < SG:
// aligned P-lane chunks of one sub-group, xor butterflies with masks < P). A team is the
// executor: Cyclic ops stride by P. O names the butterfly order (bfly::Down on whole
// sub-groups is portable::reduce_over_group, the legacy tree). No work-group barrier exists
// here by construction; sync(ex) is a sub-group barrier that every lane must reach.
template <int WG, int P = 32, class O = bfly::Down, int SG = 32> struct Teams {
    static_assert(detail::is_pow2(P) && SG % P == 0 && WG % SG == 0, "v2::Teams: P a power of two dividing SG, SG dividing WG");
    static constexpr int size = P, team_size = P, nteams = WG / P, sg_size = SG;
    static constexpr bool sub_whole = true;
    static constexpr Scope natural_scope = Scope::Sub;
    sycl::nd_item<1> it;

    BATCHLAS_DLA_INLINE int lid() const { return lane(); }
    BATCHLAS_DLA_INLINE int lane() const {
        if constexpr (P == SG) return static_cast<int>(it.get_sub_group().get_local_linear_id());
        else return static_cast<int>(it.get_sub_group().get_local_linear_id()) & (P - 1);
    }
    BATCHLAS_DLA_INLINE int team() const { return static_cast<int>(it.get_local_linear_id()) / P; }
    BATCHLAS_DLA_INLINE bool leader() const { return lane() == 0; }
    BATCHLAS_DLA_INLINE sycl::sub_group sub() const { return it.get_sub_group(); }
    template <class T, class Op = sycl::plus<T>> BATCHLAS_DLA_INLINE T team_reduce(T v, Op op = {}) const {
        if constexpr (P == SG && std::is_same_v<O, bfly::Down>) return detail::sycl_reduce(it.get_sub_group(), v, op);
        else return detail::butterfly<P, O>(it.get_sub_group(), v, op);
    }
    template <class T, class Op = sycl::plus<T>> BATCHLAS_DLA_INLINE T group_reduce(T v, Op op = {}) const {
        return team_reduce(v, op);
    }
};

template <int SG = 32> struct SubGroup {
    static constexpr int size = SG, team_size = SG, nteams = 1, sg_size = SG;
    static constexpr bool sub_whole = true;
    static constexpr Scope natural_scope = Scope::Sub;
    sycl::sub_group sg;

    BATCHLAS_DLA_INLINE sycl::sub_group sub() const { return sg; }
    BATCHLAS_DLA_INLINE int lid() const { return static_cast<int>(sg.get_local_linear_id()); }
    BATCHLAS_DLA_INLINE int lane() const { return lid(); }
    BATCHLAS_DLA_INLINE int team() const { return 0; }
    BATCHLAS_DLA_INLINE bool leader() const { return lid() == 0; }
    template <class T, class Op = sycl::plus<T>> BATCHLAS_DLA_INLINE T team_reduce(T v, Op op = {}) const {
        return detail::sycl_reduce(sg, v, op);
    }
    template <class T, class Op = sycl::plus<T>> BATCHLAS_DLA_INLINE T group_reduce(T v, Op op = {}) const {
        return team_reduce(v, op);
    }
};

// P consecutive lanes of a sub-group. Impl = sycl::sub_group is the built-in lockstep form
// (all lanes of the sub-group reach every collective); for Masked chunks pass the src-side
// SubGroupPartition<P, true> (src/kernels/common/dla_v2_partition.hh), which include/ cannot
// name: sg_partition lives under src/ and pulls src/sycl/kernel_attrs.hh.
template <int P, bool Masked = false, class Impl = sycl::sub_group> struct Partition {
    static_assert(detail::is_pow2(P), "v2::Partition: P must be a power of two");
    static_assert(!(Masked && std::is_same_v<Impl, sycl::sub_group>),
                  "v2::Partition: a Masked partition needs Impl = SubGroupPartition<P, true>");
    static constexpr int size = P, team_size = P, nteams = 1;
    static constexpr bool sub_whole = true;
    static constexpr Scope natural_scope = Scope::Sub;
    Impl part;

    BATCHLAS_DLA_INLINE explicit Partition(sycl::sub_group sg) : part(sg) {}
    struct Wrap {};
    BATCHLAS_DLA_INLINE Partition(Wrap, Impl p) : part(p) {}  // adopt an existing partition object
    BATCHLAS_DLA_INLINE int lid() const {
        if constexpr (std::is_same_v<Impl, sycl::sub_group>) return static_cast<int>(part.get_local_linear_id()) & (P - 1);
        else return static_cast<int>(part.get_local_linear_id());
    }
    BATCHLAS_DLA_INLINE int lane() const { return lid(); }
    BATCHLAS_DLA_INLINE int team() const { return 0; }
    BATCHLAS_DLA_INLINE bool leader() const { return lid() == 0; }
    template <class T, class Op = sycl::plus<T>> BATCHLAS_DLA_INLINE T team_reduce(T v, Op op = {}) const {
        return detail::butterfly<P>(part, v, op);
    }
    template <class T, class Op = sycl::plus<T>> BATCHLAS_DLA_INLINE T group_reduce(T v, Op op = {}) const {
        return team_reduce(v, op);
    }
    // The whole sub-group this partition lives in.
    BATCHLAS_DLA_INLINE sycl::sub_group sub() const {
        if constexpr (std::is_same_v<Impl, sycl::sub_group>) return part;
        else return part.sg;
    }
    // A sub-group barrier (warp sync): every lane of the sub-group must reach it.
    BATCHLAS_DLA_INLINE void barrier(Mem = Mem::Local) const { sycl::group_barrier(sub()); }
    template <class V> BATCHLAS_DLA_INLINE Published<V> publish(V v) const;
    // COLLECTIVE: lane `src`'s v to every lane of the partition; `src` must be
    // partition-uniform and every lane must reach the call. Complex moves as two reals.
    // Plain `inline`, not BATCHLAS_DLA_INLINE: inlined early, the source's real view lands
    // ahead of the shuffles and complex posv_tiny reschedules; this inlines when the hand
    // helper (tiny_bcast) did.
    template <class T> inline T bcast(T v, std::uint32_t src) const {
        using A = Arith<T>;
        if constexpr (A::is_complex) return A::make(bcast_real(A::re(v), src), bcast_real(A::im(v), src));
        else return bcast_real(v, src);
    }

  private:
    template <class R> inline R bcast_real(R v, std::uint32_t src) const {
        if constexpr (std::is_same_v<Impl, sycl::sub_group>) {
            const auto base = static_cast<std::uint32_t>(part.get_local_linear_id()) & ~std::uint32_t(P - 1);
            return sycl::select_from_group(part, v, base + src);
        } else {
            return select_from_group(part, v, src);  // ADL: SubGroupPartition (src/kernels/common)
        }
    }
};

// One work-item that owns whole operands (serial loops, no collectives).
struct Thread {
    static constexpr int size = 1, team_size = 1, nteams = 1;
    static constexpr bool sub_whole = true;
    static constexpr Scope natural_scope = Scope::Sub;
    BATCHLAS_DLA_INLINE int lid() const { return 0; }
    BATCHLAS_DLA_INLINE int lane() const { return 0; }
    BATCHLAS_DLA_INLINE int team() const { return 0; }
    BATCHLAS_DLA_INLINE bool leader() const { return true; }
    template <class T, class Op = sycl::plus<T>> BATCHLAS_DLA_INLINE T team_reduce(T v, Op = {}) const { return v; }
    template <class T, class Op = sycl::plus<T>> BATCHLAS_DLA_INLINE T group_reduce(T v, Op = {}) const { return v; }
};

// A work-group whose size is a launch value (one binary for every WG), or one sub-group of it
// (S = Scope::Sub). Ids and ranges are read once, at construction, as hand kernels read them
// at the top (read per use, the sub-group range is rematerialised per op: +3..4 REG, geqrf).
// Loops stride by the runtime width(); SG is the sub-group size the kernel is launched with.
template <Scope S = Scope::Group, int SG = 32> struct RtGroup {
    static_assert(S == Scope::Sub || S == Scope::Group, "v2::RtGroup: Scope::Sub or Scope::Group");
    static constexpr bool runtime = true;
    static constexpr bool sub_whole = (S == Scope::Sub);
    static constexpr Scope natural_scope = S;
    static constexpr int sg_size = SG;
    sycl::nd_item<1> it;
    int lane_, lid_, width_, team_, nteams_;

    BATCHLAS_DLA_INLINE explicit RtGroup(sycl::nd_item<1> i)
        : it(i), lane_(static_cast<int>(i.get_sub_group().get_local_linear_id())),
          lid_(S == Scope::Sub ? lane_ : static_cast<int>(i.get_local_linear_id())),
          width_(S == Scope::Sub ? static_cast<int>(i.get_sub_group().get_local_linear_range())
                                 : static_cast<int>(i.get_local_range(0))),
          team_(S == Scope::Sub ? 0 : static_cast<int>(i.get_sub_group().get_group_linear_id())),
          nteams_(S == Scope::Sub ? 1 : static_cast<int>(i.get_sub_group().get_group_linear_range())) {}

    BATCHLAS_DLA_INLINE int lid() const { return lid_; }
    BATCHLAS_DLA_INLINE int width() const { return width_; }
    BATCHLAS_DLA_INLINE int lane() const { return lane_; }
    BATCHLAS_DLA_INLINE int team() const { return team_; }
    BATCHLAS_DLA_INLINE int nteams() const { return nteams_; }
    BATCHLAS_DLA_INLINE int team_size() const { return static_cast<int>(it.get_sub_group().get_local_linear_range()); }
    BATCHLAS_DLA_INLINE bool leader() const { return lid_ == 0; }
    BATCHLAS_DLA_INLINE sycl::sub_group sub() const { return it.get_sub_group(); }
    // The sub-group this item belongs to, as its own executor.
    BATCHLAS_DLA_INLINE RtGroup<Scope::Sub, SG> team_exec() const { return RtGroup<Scope::Sub, SG>(it); }
    template <class T, class Op = sycl::plus<T>> BATCHLAS_DLA_INLINE T team_reduce(T v, Op op = {}) const {
        return detail::sycl_reduce(it.get_sub_group(), v, op);
    }
    template <class T, class Op = sycl::plus<T>> BATCHLAS_DLA_INLINE T group_reduce(T v, Op op = {}) const {
        if constexpr (S == Scope::Sub) return team_reduce(v, op);
        else return detail::sycl_reduce(it.get_group(), v, op);
    }
    BATCHLAS_DLA_INLINE void barrier(Mem m = Mem::Local) const {
        if constexpr (S == Scope::Sub) sycl::group_barrier(it.get_sub_group());
        else if (m == Mem::Local) it.barrier(sycl::access::fence_space::local_space);
        else it.barrier(sycl::access::fence_space::global_and_local);
    }
    template <class V> BATCHLAS_DLA_INLINE Published<V> publish(V v) const;
    // COLLECTIVE: item `src`'s v to every item of the executor (`src` uniform).
    template <class T> BATCHLAS_DLA_HELPER T bcast(T v, int src) const {
        using A = Arith<T>;
        auto one = [&](auto x) {
            if constexpr (S == Scope::Sub) return sycl::select_from_group(it.get_sub_group(), x, src);
            else return detail::group_bcast(it.get_group(), x, src, lid_);
        };
        if constexpr (A::is_complex) return A::make(one(A::re(v)), one(A::im(v)));
        else return one(v);
    }
};

template <class Ex> inline constexpr bool is_runtime_v = requires { Ex::runtime; };

namespace detail {
// The executor's Cyclic stride, team width and team count: type constants, or launch values
// for RtGroup. Ops read sizes only through these.
template <class Ex> BATCHLAS_DLA_INLINE int size_of(const Ex& ex) {
    if constexpr (is_runtime_v<Ex>) return ex.width();
    else return Ex::size;
}
template <class Ex> BATCHLAS_DLA_INLINE int team_size_of(const Ex& ex) {
    if constexpr (is_runtime_v<Ex>) return ex.team_size();
    else return Ex::team_size;
}
template <class Ex> BATCHLAS_DLA_INLINE int nteams_of(const Ex& ex) {
    if constexpr (is_runtime_v<Ex>) return ex.nteams();
    else return Ex::nteams;
}
}  // namespace detail

template <class Ex> inline constexpr bool is_work_group_v = false;
template <int WG, int SG> inline constexpr bool is_work_group_v<WorkGroup<WG, SG>> = true;
template <class Ex> inline constexpr bool is_thread_v = std::is_same_v<Ex, Thread>;

// A barrier over the executor's scope: work-group barrier on WorkGroup / RtGroup<Group>, a
// sub-group barrier on Teams, SubGroup, Partition and RtGroup<Sub>, nothing on Thread.
template <class Ex> BATCHLAS_DLA_INLINE void sync(const Ex& ex, Mem m = Mem::Local) {
    if constexpr (is_thread_v<Ex>) return;
    else if constexpr (requires { ex.barrier(m); }) ex.barrier(m);
    else sycl::group_barrier(ex.sub());
}

// Scratch for a Group-scope packed exchange. The caller allocates the local memory; its size
// is the constexpr scratch_elements<Op, Ex, T>().
template <class T, int Slots> struct Scratch {
    T* p;
    static constexpr int slots = Slots;
};

template <class Op, class Ex, class T> constexpr int scratch_elements() {
    if constexpr (is_work_group_v<Ex>) {
        if constexpr (std::is_same_v<Op, op::ReduceN<1>>) return Ex::nteams;
        else return []<int K>(op::ReduceN<K>*) { return Ex::nteams * K; }(static_cast<Op*>(nullptr));
    } else {
        return 0;
    }
}

}  // namespace batchlas::device::v2
