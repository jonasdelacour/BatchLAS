#pragma once
// Generic backend: plain sycl::sub_group collectives (host pass, Native CPU, non-PTX acpp JIT).
// Those are defined only for a converged sub-group, so a diverging Masked
// partition here relies on SIMD hardware tolerating it.

#include <sycl/sycl.hpp>

#include <cstdint>

#include <batchlas/backend_config.h>

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
#if BATCHLAS_SYCL_IMPL_ACPP
        if (mask == 0u) return v;  // acpp 25.10 CPU gives 0 for xor 0 (sscp/host/shuffle.cpp:35)
#endif
        return sycl::permute_group_by_xor(sg, v, mask);
    }

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

    // Optional; when false the front end reduces with a shfl_xor butterfly.
    template <typename T, typename Op>
    static constexpr bool has_native_reduce = false;

    template <typename T, typename Op>
    static T reduce(const sycl::sub_group&, uint32_t, T v, Op) { return v; }
};

} // namespace batchlas::sgp
