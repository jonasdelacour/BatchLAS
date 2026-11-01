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

// xor butterfly over P lanes; every lane ends with the same value.
template <int P, class Impl, class T, class Op> BATCHLAS_DLA_INLINE T butterfly(const Impl& part, T v, Op op) {
    using A = Arith<T>;
    if constexpr (A::is_complex) {
        using R = typename A::real_t;
        R re = A::re(v), im = A::im(v);
#pragma unroll
        for (int m = P / 2; m >= 1; m >>= 1) {
            re += xor_shuffle(part, re, static_cast<std::uint32_t>(m));
            im += xor_shuffle(part, im, static_cast<std::uint32_t>(m));
        }
        return A::make(re, im);
    } else {
#pragma unroll
        for (int m = P / 2; m >= 1; m >>= 1) v = op(v, xor_shuffle(part, v, static_cast<std::uint32_t>(m)));
        return v;
    }
}

inline constexpr bool is_pow2(int p) { return p > 0 && (p & (p - 1)) == 0; }

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

// Teams of SG items (sub-groups) working independently inside a WG launch. A team is the
// executor: Cyclic ops stride by SG. No work-group barrier exists here by construction.
template <int WG, int SG = 32> struct Teams {
    static_assert(WG % SG == 0, "v2::Teams: SG must divide WG");
    static constexpr int size = SG, team_size = SG, nteams = WG / SG, sg_size = SG;
    static constexpr bool sub_whole = true;
    static constexpr Scope natural_scope = Scope::Sub;
    sycl::nd_item<1> it;

    BATCHLAS_DLA_INLINE int lid() const { return lane(); }
    BATCHLAS_DLA_INLINE int lane() const { return static_cast<int>(it.get_sub_group().get_local_linear_id()); }
    BATCHLAS_DLA_INLINE int team() const { return static_cast<int>(it.get_local_linear_id()) / SG; }
    BATCHLAS_DLA_INLINE bool leader() const { return lane() == 0; }
    template <class T, class Op = sycl::plus<T>> BATCHLAS_DLA_INLINE T team_reduce(T v, Op op = {}) const {
        return detail::sycl_reduce(it.get_sub_group(), v, op);
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

template <class Ex> inline constexpr bool is_work_group_v = false;
template <int WG, int SG> inline constexpr bool is_work_group_v<WorkGroup<WG, SG>> = true;
template <class Ex> inline constexpr bool is_thread_v = std::is_same_v<Ex, Thread>;

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
