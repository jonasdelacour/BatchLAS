#pragma once
// Sub-group partition compatibility layer.
//
// Provides a SubGroupPartition<P> type that emulates the SYCL
// chunked_partition<P> / fixed_size_group<P> API using only plain sub_group
// shuffle operations (SubgroupShuffleINTEL / SubgroupShuffleXorINTEL /
// SubgroupShuffleDownINTEL) which are available on all GPU backends including
// AMD HIP.
//
// Background: SYCL's fixed_size_group shuffle operations route through
// __spirv_GroupNonUniformShuffle, which is absent from AMD's AMDGCN SPIRV
// device library.  The intel/llvm test suite marks fixed_size_group/chunk
// algorithms as "UNSUPPORTED: target-amd" for this reason.  This wrapper
// bypasses GroupNonUniformShuffle entirely.
//
// Usage in kernels:
//   Replace:  sycl::ext::oneapi::experimental::chunked_partition<P>(sg)
//   With:     batchlas::make_partition<P>(sg)
//
//   Then drop the "sycl::" qualifier from permute_group_by_xor,
//   select_from_group, group_barrier, and shift_group_left calls so that
//   ADL dispatches to the overloads below for SubGroupPartition arguments and
//   falls back to the sycl:: versions for plain sub_group / sycl::group<N>.

#include <sycl/sycl.hpp>
#include <cstdint>
#include <utility>

namespace batchlas {

template <size_t P>
using NativeSubGroupPartition = decltype(
    sycl::ext::oneapi::experimental::chunked_partition<P>(std::declval<sycl::sub_group>()));

#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
inline constexpr bool kUseNativeChunkedPartition = true;
#else
inline constexpr bool kUseNativeChunkedPartition = false;
#endif

// ---------------------------------------------------------------------------
// SubGroupPartition<P>
//
// Wraps a sycl::sub_group and the absolute base lane of this chunk.
// Chunk i occupies lanes [i*P, (i+1)*P) of the sub_group.
// ---------------------------------------------------------------------------
template <size_t P>
struct SubGroupPartition {
    sycl::sub_group sg;
        NativeSubGroupPartition<P> native;
    uint32_t base;  ///< absolute sub_group lane of this chunk's first lane

    explicit SubGroupPartition(sycl::sub_group sg_)
                : sg(sg_),
                    native(sycl::ext::oneapi::experimental::chunked_partition<P>(sg_)),
          base((static_cast<uint32_t>(sg_.get_local_linear_id()) / static_cast<uint32_t>(P))
               * static_cast<uint32_t>(P))
    {}

    // --- Local (within-chunk) identity ---
    uint32_t get_local_linear_id() const {
        return static_cast<uint32_t>(sg.get_local_linear_id()) - base;
    }
    uint32_t get_local_linear_range() const { return static_cast<uint32_t>(P); }

    // get_local_range().size() is used by group_local_linear_range() in
    // group_blas_common.hh — return a range<1> so .size() gives P.
    sycl::range<1> get_local_range() const { return sycl::range<1>{P}; }

    sycl::id<1> get_local_id() const { return sycl::id<1>{get_local_linear_id()}; }

    bool leader() const { return get_local_linear_id() == 0u; }

    // --- Group (which chunk am I?) ---
    uint32_t get_group_linear_id() const {
        return static_cast<uint32_t>(sg.get_local_linear_id()) / static_cast<uint32_t>(P);
    }
    uint32_t get_group_linear_range() const {
        return static_cast<uint32_t>(sg.get_local_linear_range()) / static_cast<uint32_t>(P);
    }
};

// ---------------------------------------------------------------------------
// Factory — replaces sycl::ext::oneapi::experimental::chunked_partition<P>(sg)
// ---------------------------------------------------------------------------
template <size_t P>
inline SubGroupPartition<P> make_partition(sycl::sub_group sg) {
    return SubGroupPartition<P>(sg);
}

// ---------------------------------------------------------------------------
// AMD-safe collective overloads for SubGroupPartition<P>
//
// All overloads are in namespace batchlas so ADL finds them when the first
// argument is SubGroupPartition<P>.  Callers should use unqualified names:
//   permute_group_by_xor(part, v, mask)
//   select_from_group(part, v, local_id)
//   group_barrier(part)
//   shift_group_left(part, v, delta)
// ---------------------------------------------------------------------------

// permute_group_by_xor:
//   mask < P, so the XOR address stays within the same chunk.
//   Route through the underlying sub_group (SubgroupShuffleXorINTEL).
template <size_t P, typename T>
inline T permute_group_by_xor(SubGroupPartition<P> part, T v, uint32_t mask) {
    if constexpr (kUseNativeChunkedPartition) {
        return sycl::permute_group_by_xor(part.native, v, mask);
    } else {
        return sycl::permute_group_by_xor(part.sg, v, mask);
    }
}

// select_from_group:
//   local_id is 0-based within the chunk; convert to absolute sub_group lane.
//   Each lane supplies its own base, so every chunk reads its own member.
template <size_t P, typename T>
inline T select_from_group(SubGroupPartition<P> part, T v, uint32_t local_id) {
    if constexpr (kUseNativeChunkedPartition) {
        return sycl::select_from_group(part.native, v,
                                       typename NativeSubGroupPartition<P>::id_type{local_id});
    } else {
        return sycl::select_from_group(part.sg, v, part.base + local_id);
    }
}

// group_barrier:
//   On NVIDIA Volta+ (sm_70+) Independent Thread Scheduling (ITS) means that
//   threads within a warp can diverge and re-converge independently.  Without
//   an explicit __syncwarp() (or its SYCL equivalent sycl::group_barrier on
//   the sub_group), shared-memory writes made inside a divergent section may
//   not be visible to other lanes after reconvergence.  Routing through
//   sycl::group_barrier(sub_group) emits __syncwarp() on CUDA and is a
//   cheap no-op on AMD (where wave lanes are always coherent after merge).
template <size_t P>
inline void group_barrier(SubGroupPartition<P> part) noexcept {
    if constexpr (kUseNativeChunkedPartition) {
        sycl::group_barrier(part.native);
    }
}

// shift_group_left:
//   Shift-down within the chunk.  Route through sub_group's shift
//   (SubgroupShuffleDownINTEL).  The boundary value (last lane of the chunk
//   reading across to the next chunk) is undefined by the SYCL spec, matching
//   the fixed_size_group semantics; callers guard that case.
template <size_t P, typename T>
inline T shift_group_left(SubGroupPartition<P> part, T v, uint32_t delta) {
    if constexpr (kUseNativeChunkedPartition) {
        return sycl::shift_group_left(part.native, v, delta);
    } else {
        return sycl::shift_group_left(part.sg, v, delta);
    }
}

// ---------------------------------------------------------------------------
// sg_leader_broadcast — used by group-invoke.hh's broadcast_from_leader_impl
//   via ADL to broadcast the leader's value to every lane in the chunk.
//
//   sycl::select_from_group(sg, v, part.base) reads lane `base` for all
//   callers.  Because each chunk has a different base, each chunk gets its
//   own leader's value.
// ---------------------------------------------------------------------------
template <size_t P, typename T>
inline T sg_leader_broadcast(SubGroupPartition<P> part, T value) {
    if constexpr (kUseNativeChunkedPartition) {
        return sycl::select_from_group(part.native, value,
                                       typename NativeSubGroupPartition<P>::id_type{});
    } else {
        return sycl::select_from_group(part.sg, value, part.base);
    }
}

} // namespace batchlas
