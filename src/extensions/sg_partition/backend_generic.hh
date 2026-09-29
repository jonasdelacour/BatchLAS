#pragma once
// Generic backend: every primitive is a plain sycl::sub_group collective.
//
// Used on the host pass, on SYCL Native CPU, and as the reference the device
// backends are tested against. The sub-group collectives it calls are only
// defined for converged sub-groups, so a Masked partition that diverges per
// chunk relies on SIMD hardware tolerating that (true on the OpenCL CPU device).
// The device backends exist to remove that reliance.

#include <sycl/sycl.hpp>

#include <cstdint>

namespace batchlas::sgp {

template <uint32_t P, bool Masked>
struct GenericBackend {
    static constexpr const char* name = "generic";

    // src is a lane of this chunk, 0 <= src < P.
    static uint32_t shfl_idx(const sycl::sub_group& sg, uint32_t base, uint32_t v, uint32_t src) {
        return sycl::select_from_group(sg, v, base + src);
    }

    // mask < P, so lane ^ mask stays in the chunk.
    static uint32_t shfl_xor(const sycl::sub_group& sg, uint32_t, uint32_t v, uint32_t mask) {
        return sycl::permute_group_by_xor(sg, v, mask);
    }

    // Lanes whose source falls outside the chunk get an unspecified value.
    static uint32_t shfl_down(const sycl::sub_group& sg, uint32_t, uint32_t v, uint32_t delta) {
        return sycl::shift_group_left(sg, v, delta);
    }

    static uint32_t shfl_up(const sycl::sub_group& sg, uint32_t, uint32_t v, uint32_t delta) {
        return sycl::shift_group_right(sg, v, delta);
    }

    // Bit i set iff local lane i of this chunk has pred.
    static uint32_t ballot(const sycl::sub_group& sg, uint32_t base, bool pred) {
        const uint32_t lane = static_cast<uint32_t>(sg.get_local_linear_id()) - base;
        uint32_t bits = pred ? (1u << lane) : 0u;
        for (uint32_t m = 1; m < P; m <<= 1) bits |= sycl::permute_group_by_xor(sg, bits, m);
        return bits;
    }

    static void barrier(const sycl::sub_group& sg, uint32_t) { sycl::group_barrier(sg); }

    // Optional fast path for reduce_over_group; the front end falls back to a
    // shfl_xor butterfly when this is false.
    template <typename T, typename Op>
    static constexpr bool has_native_reduce = false;

    template <typename T, typename Op>
    static T reduce(const sycl::sub_group&, uint32_t, T v, Op) { return v; }
};

} // namespace batchlas::sgp
