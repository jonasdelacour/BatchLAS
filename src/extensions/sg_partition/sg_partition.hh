#pragma once
// Chunked sub-group partitions: BatchLAS's own replacement for DPC++'s
// sycl::ext::oneapi::experimental::chunked_partition.
//
// SubGroupPartition<P, Masked> splits a sub-group into chunks of P consecutive
// lanes; chunk i owns lanes [i*P, (i+1)*P). Every collective below acts on one
// chunk only.
//
//   Masked == true   the chunks of a sub-group may take different control flow;
//                    a collective only needs every lane of its own chunk.
//   Masked == false  lockstep: every lane of the sub-group reaches every
//                    collective (the chunks run one instruction stream). On
//                    NVIDIA this lets the shuffles use a constant full mask.
//
// The typed front end in this file reduces everything to 32-bit word
// primitives supplied by one backend per target (backend_*.hh). Kernels call
// the collectives unqualified, so ADL picks these overloads for a partition
// and sycl::'s for a plain sub_group or work-group.

#include <sycl/sycl.hpp>

#include <cstdint>
#include <cstring>
#include <type_traits>

#include "backend_generic.hh"

#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
#include "backend_nvptx.hh"
#elif defined(__SYCL_DEVICE_ONLY__) && defined(__AMDGCN__)
#include "backend_amdgcn.hh"
#elif defined(__SYCL_DEVICE_ONLY__) && (defined(__SPIR__) || defined(__SPIRV__))
#include "backend_spirv.hh"
#endif

namespace batchlas {

namespace sgp {

#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
template <uint32_t P, bool Masked> using Backend = NvptxBackend<P, Masked>;
inline constexpr bool kMaskedByDefault = true;
#elif defined(__SYCL_DEVICE_ONLY__) && defined(__AMDGCN__)
template <uint32_t P, bool Masked> using Backend = AmdgcnBackend<P, Masked>;
inline constexpr bool kMaskedByDefault = AmdgcnBackend<1, true>::masked_by_default;
#elif defined(__SYCL_DEVICE_ONLY__) && (defined(__SPIR__) || defined(__SPIRV__))
template <uint32_t P, bool Masked> using Backend = SpirvBackend<P, Masked>;
inline constexpr bool kMaskedByDefault = SpirvBackend<1, true>::masked_by_default;
#else
template <uint32_t P, bool Masked> using Backend = GenericBackend<P, Masked>;
inline constexpr bool kMaskedByDefault = false;
#endif

// Any trivially copyable T moves as ceil(sizeof(T)/4) 32-bit words.
template <typename T>
inline constexpr uint32_t kWords = static_cast<uint32_t>((sizeof(T) + 3) / 4);

template <typename T>
struct Words {
    uint32_t w[kWords<T>];
};

template <typename T>
inline Words<T> to_words(const T& v) {
    Words<T> out{};
    std::memcpy(out.w, &v, sizeof(T));
    return out;
}

template <typename T>
inline T from_words(const Words<T>& in) {
    T v;
    std::memcpy(&v, in.w, sizeof(T));
    return v;
}

template <typename T, typename F>
inline T map_words(const T& v, F&& f) {
    static_assert(std::is_trivially_copyable_v<T>, "partition collectives need a trivially copyable T");
    if constexpr (sizeof(T) == 4) {
        return sycl::bit_cast<T>(f(sycl::bit_cast<uint32_t>(v)));
    } else {
        Words<T> in = to_words(v);
#pragma unroll
        for (uint32_t i = 0; i < kWords<T>; ++i) in.w[i] = f(in.w[i]);
        return from_words<T>(in);
    }
}

template <uint32_t P>
inline constexpr bool is_pow2_v = P != 0 && (P & (P - 1)) == 0;

// Runs f(ops), where ops is a backend-like type whose primitives f calls. A
// backend with a region() hook chooses ops once for the region (NVPTX tests
// warp convergence once), so a multi-word value or a whole scan pays once.
template <typename B, typename F>
inline decltype(auto) region(F&& f) {
    if constexpr (requires { B::region(f); }) {
        return B::region(f);
    } else {
        return f(B{});
    }
}

} // namespace sgp

// Kept for the call sites that ask whether masked partitions are the native
// (divergence-safe, zero-padding) choice on this target.
inline constexpr bool kUseNativeChunkedPartition = sgp::kMaskedByDefault;

template <size_t P, bool Masked = sgp::kMaskedByDefault>
struct SubGroupPartition {
    static_assert(sgp::is_pow2_v<static_cast<uint32_t>(P)>, "partition size must be a power of two");
    static constexpr uint32_t kP = static_cast<uint32_t>(P);
    static constexpr bool kMasked = Masked;
    using backend = sgp::Backend<kP, Masked>;

    sycl::sub_group sg;
    uint32_t base;  ///< sub-group lane of this chunk's first lane

    explicit SubGroupPartition(sycl::sub_group sg_)
        : sg(sg_), base(static_cast<uint32_t>(sg_.get_local_linear_id()) & ~(kP - 1u)) {}

    uint32_t get_local_linear_id() const {
        return static_cast<uint32_t>(sg.get_local_linear_id()) & (kP - 1u);
    }
    uint32_t get_local_linear_range() const { return kP; }
    sycl::range<1> get_local_range() const { return sycl::range<1>{P}; }
    sycl::id<1> get_local_id() const { return sycl::id<1>{get_local_linear_id()}; }
    bool leader() const { return get_local_linear_id() == 0u; }

    uint32_t get_group_linear_id() const {
        return static_cast<uint32_t>(sg.get_local_linear_id()) / kP;
    }
    uint32_t get_group_linear_range() const {
        return static_cast<uint32_t>(sg.get_local_linear_range()) / kP;
    }
};

template <size_t P, bool Masked = sgp::kMaskedByDefault>
inline SubGroupPartition<P, Masked> make_partition(sycl::sub_group sg) {
    return SubGroupPartition<P, Masked>(sg);
}

template <typename G>
inline constexpr bool is_sub_group_partition_v = false;
template <size_t P, bool M>
inline constexpr bool is_sub_group_partition_v<SubGroupPartition<P, M>> = true;

// ---------------------------------------------------------------------------
// Data movement
// ---------------------------------------------------------------------------

// Value of local lane `local_id` of this chunk.
template <size_t P, bool M, typename T>
inline T select_from_group(SubGroupPartition<P, M> part, T v, uint32_t local_id) {
    using B = typename SubGroupPartition<P, M>::backend;
    return sgp::region<B>([&](auto ops) {
        return sgp::map_words(v, [&](uint32_t w) {
            return decltype(ops)::shfl_idx(part.sg, part.base, w, local_id);
        });
    });
}

// Value of local lane (id ^ mask); mask < P.
template <size_t P, bool M, typename T>
inline T permute_group_by_xor(SubGroupPartition<P, M> part, T v, uint32_t mask) {
    using B = typename SubGroupPartition<P, M>::backend;
    return sgp::region<B>([&](auto ops) {
        return sgp::map_words(v, [&](uint32_t w) {
            return decltype(ops)::shfl_xor(part.sg, part.base, w, mask);
        });
    });
}

// Value of local lane id + delta; unspecified when that leaves the chunk.
template <size_t P, bool M, typename T>
inline T shift_group_left(SubGroupPartition<P, M> part, T v, uint32_t delta = 1) {
    using B = typename SubGroupPartition<P, M>::backend;
    return sgp::region<B>([&](auto ops) {
        return sgp::map_words(v, [&](uint32_t w) {
            return decltype(ops)::shfl_down(part.sg, part.base, w, delta);
        });
    });
}

// Value of local lane id - delta; unspecified when that leaves the chunk.
template <size_t P, bool M, typename T>
inline T shift_group_right(SubGroupPartition<P, M> part, T v, uint32_t delta = 1) {
    using B = typename SubGroupPartition<P, M>::backend;
    return sgp::region<B>([&](auto ops) {
        return sgp::map_words(v, [&](uint32_t w) {
            return decltype(ops)::shfl_up(part.sg, part.base, w, delta);
        });
    });
}

template <size_t P, bool M, typename T>
inline T group_broadcast(SubGroupPartition<P, M> part, T v, uint32_t local_id = 0) {
    return select_from_group(part, v, local_id);
}

template <size_t P, bool M, typename T>
inline T sg_leader_broadcast(SubGroupPartition<P, M> part, T v) {
    return select_from_group(part, v, 0u);
}

// Orders this chunk's local-memory accesses before and after the call. A
// lockstep partition barriers the whole sub-group.
template <size_t P, bool M>
inline void group_barrier(SubGroupPartition<P, M> part) noexcept {
    SubGroupPartition<P, M>::backend::barrier(part.sg, part.base);
}

// ---------------------------------------------------------------------------
// Votes
// ---------------------------------------------------------------------------

// Bit i set iff local lane i of this chunk passed pred. 32 bits, so P <= 32.
template <size_t P, bool M>
inline uint32_t ballot(SubGroupPartition<P, M> part, bool pred) {
    static_assert(P <= 32, "ballot returns 32 bits; use any_of_group/all_of_group for P = 64");
    return SubGroupPartition<P, M>::backend::ballot(part.sg, part.base, pred);
}

// P = 64 does not fit a ballot word, so it votes by reduction.
template <size_t P, bool M>
inline bool any_of_group(SubGroupPartition<P, M> part, bool pred) {
    if constexpr (P > 32) {
        return reduce_over_group(part, static_cast<uint32_t>(pred), sycl::bit_or<uint32_t>()) != 0u;
    } else {
        return ballot(part, pred) != 0u;
    }
}

template <size_t P, bool M>
inline bool all_of_group(SubGroupPartition<P, M> part, bool pred) {
    if constexpr (P > 32) {
        return reduce_over_group(part, static_cast<uint32_t>(pred), sycl::bit_and<uint32_t>()) != 0u;
    } else {
        constexpr uint32_t full = P >= 32 ? ~0u : ((1u << P) - 1u);
        return ballot(part, pred) == full;
    }
}

template <size_t P, bool M>
inline bool none_of_group(SubGroupPartition<P, M> part, bool pred) {
    return !any_of_group(part, pred);
}

// ---------------------------------------------------------------------------
// Reductions and scans. Op is any binary functor (sycl::plus<>, sycl::maximum<>,
// ...); the result of a reduction is the same on every lane of the chunk.
// ---------------------------------------------------------------------------

template <size_t P, bool M, typename T, typename Op>
inline T reduce_over_group(SubGroupPartition<P, M> part, T v, Op op) {
    using B = typename SubGroupPartition<P, M>::backend;
    if constexpr (B::template has_native_reduce<T, Op>) {
        return B::reduce(part.sg, part.base, v, op);
    } else {
#pragma unroll
        for (uint32_t m = 1; m < static_cast<uint32_t>(P); m <<= 1)
            v = op(v, permute_group_by_xor(part, v, m));
        return v;
    }
}

template <size_t P, bool M, typename T, typename Op>
inline T inclusive_scan_over_group(SubGroupPartition<P, M> part, T v, Op op) {
    using B = typename SubGroupPartition<P, M>::backend;
    const uint32_t lid = part.get_local_linear_id();
    return sgp::region<B>([&](auto ops) {
        T x = v;
#pragma unroll
        for (uint32_t d = 1; d < static_cast<uint32_t>(P); d <<= 1) {
            const T other = sgp::map_words(x, [&](uint32_t w) {
                return decltype(ops)::shfl_up(part.sg, part.base, w, d);
            });
            if (lid >= d) x = op(other, x);
        }
        return x;
    });
}

template <size_t P, bool M, typename T, typename Op>
inline T exclusive_scan_over_group(SubGroupPartition<P, M> part, T v, T init, Op op) {
    const T inc = inclusive_scan_over_group(part, v, op);
    const T prev = shift_group_right(part, inc, 1u);
    return part.get_local_linear_id() == 0u ? init : op(init, prev);
}

// ---------------------------------------------------------------------------
// Lockstep hooks: the domain whose chunks must run one instruction stream.
//
// A masked partition is its own domain, so the chunks of a sub-group may take
// different trip counts. A lockstep partition shuffles over the whole
// sub-group, so its chunks must agree on every branch that guards a
// collective: the votes span the sub-group and the caller pads short chunks
// with no-op work. `v` must be uniform across the chunk.
// ---------------------------------------------------------------------------
template <typename Group>
inline constexpr bool lockstep_spans_subgroup_v = false;

template <size_t P, bool M>
inline constexpr bool lockstep_spans_subgroup_v<SubGroupPartition<P, M>> = !M;

template <size_t P, bool M>
inline bool lockstep_any(SubGroupPartition<P, M> part, bool v) {
    if constexpr (M) {
        return v;
    } else {
        return sycl::any_of_group(part.sg, v);
    }
}

template <size_t P, bool M>
inline int32_t lockstep_max(SubGroupPartition<P, M> part, int32_t v) {
    if constexpr (M) {
        return v;
    } else {
        return sycl::reduce_over_group(part.sg, v, sycl::maximum<int32_t>());
    }
}

} // namespace batchlas
