#pragma once
// AMDGCN backend: SYCL adaptor over backend_amdgcn_core.hh, included by
// sg_partition.hh on the AMDGCN device pass only. Masked is free and the
// default: a diverged chunk is just EXEC-masked and every primitive reads only
// its own chunk. DPC++ on AMD has sub-group size == wave size and lowers the
// sub-group local id to mbcnt, so `base` is a hardware lane.

#include <sycl/sycl.hpp>

#include <cstdint>
#include <type_traits>

#include "backend_amdgcn_core.hh"

namespace batchlas::sgp {

namespace amdgcn_detail {

// core::reduce's mirror steps need an order-independent op.
template <typename T, typename Op>
inline constexpr bool commutative_op_v =
    std::is_same_v<Op, sycl::plus<T>> || std::is_same_v<Op, sycl::plus<>> ||
    std::is_same_v<Op, sycl::multiplies<T>> || std::is_same_v<Op, sycl::multiplies<>> ||
    std::is_same_v<Op, sycl::minimum<T>> || std::is_same_v<Op, sycl::minimum<>> ||
    std::is_same_v<Op, sycl::maximum<T>> || std::is_same_v<Op, sycl::maximum<>> ||
    std::is_same_v<Op, sycl::bit_and<T>> || std::is_same_v<Op, sycl::bit_and<>> ||
    std::is_same_v<Op, sycl::bit_or<T>> || std::is_same_v<Op, sycl::bit_or<>> ||
    std::is_same_v<Op, sycl::bit_xor<T>> || std::is_same_v<Op, sycl::bit_xor<>>;

template <typename T>
inline constexpr bool word_arith_v =
    std::is_arithmetic_v<T> && !std::is_same_v<T, bool> && (sizeof(T) == 4 || sizeof(T) == 8);

} // namespace amdgcn_detail

template <uint32_t P, bool Masked>
struct AmdgcnBackend {
    static_assert(P <= amdgcn::kWave,
                  "partition wider than the wavefront (RDNA wave64: define BATCHLAS_AMDGCN_WAVE_SIZE=64)");

    static constexpr const char* name = "amdgcn";
    static constexpr bool masked_by_default = true;

    static uint32_t shfl_idx(const sycl::sub_group&, uint32_t base, uint32_t v, uint32_t src) {
        return amdgcn::shfl_idx<P>(amdgcn::lane_id(), base, v, src);
    }

    static uint32_t shfl_xor(const sycl::sub_group&, uint32_t, uint32_t v, uint32_t mask) {
        return amdgcn::shfl_xor<P>(amdgcn::lane_id(), v, mask);
    }

    // Out-of-chunk sources: 0 from a row-local DPP shift, else a neighbour's value.
    static uint32_t shfl_down(const sycl::sub_group&, uint32_t, uint32_t v, uint32_t delta) {
        return amdgcn::shfl_down<P>(amdgcn::lane_id(), v, delta);
    }

    static uint32_t shfl_up(const sycl::sub_group&, uint32_t, uint32_t v, uint32_t delta) {
        return amdgcn::shfl_up<P>(amdgcn::lane_id(), v, delta);
    }

    // The front end's vote word is 32 bits; refuse P == 64 rather than truncate.
    static uint32_t ballot(const sycl::sub_group&, uint32_t base, bool pred) {
        static_assert(P <= 32, "sg_partition ballot is 32-bit; a 64-lane chunk needs a 64-bit vote word");
        return static_cast<uint32_t>(amdgcn::ballot<P>(base, pred));
    }

    static void barrier(const sycl::sub_group&, uint32_t) { amdgcn::barrier(); }

    template <typename T, typename Op>
    static constexpr bool has_native_reduce =
        P > 1u && amdgcn_detail::word_arith_v<T> && amdgcn_detail::commutative_op_v<T, Op>;

    template <typename T, typename Op>
    static T reduce(const sycl::sub_group&, uint32_t, T v, Op op) {
        return amdgcn::reduce<P>(v, op);
    }
};

} // namespace batchlas::sgp
