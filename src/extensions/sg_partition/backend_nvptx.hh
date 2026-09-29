#pragma once
// NVPTX backend (device pass only): one warp-synchronous PTX instruction per
// primitive.
//
// For any member mask that is not an immediate, ptxas guards the collectives
// of each basic block with MATCH.ANY + REDUX.OR + VOTEU.ANY + BRA.DIV, whatever
// the mask's source (DPC++ chunked_partition and CUDA tiled_partition too).
// MATCH.ANY slows with the number of distinct masks in the warp, 32 / P.
// So a masked collective narrower than the warp first tests activemask: if all
// 32 lanes are here, all took the same branch and will execute the same
// shfl.sync, the immediate full mask is legal, and ptxas adds no check. That is
// the premise ptxas's own fast path rests on, without MATCH.ANY. A diverged
// warp, or one with an exited or unlaunched lane, uses the chunk mask.
//
// The `c` operand's segment mask ((32 - P) << 8) makes idx take the chunk-local
// lane and makes down/up clamp at the chunk edge.

#include <sycl/sycl.hpp>

#include <cstdint>
#include <cstring>
#include <type_traits>

namespace batchlas::sgp {

namespace nvptx_detail {

// One fixed member mask: the whole warp (Full) or this lane's chunk.
// sub-group lane == %laneid under DPC++ CUDA for every work-group shape, so
// `base` is a warp lane.
// P > 32 cannot occur on a 32-lane warp; it only has to compile, because a
// kernel written for 64-wide sub-groups is still built for every target.
template <uint32_t P, bool Full>
struct Mask {
    static constexpr uint32_t kLow = P >= 32 ? ~0u : ((1u << (P & 31u)) - 1u);
    static constexpr int32_t kSeg = P >= 32 ? 0 : static_cast<int32_t>((32u - P) << 8);

    static uint32_t mask(uint32_t base) {
        if constexpr (Full || P >= 32) return ~0u;
        else return kLow << base;
    }

    static uint32_t shfl_idx(uint32_t base, uint32_t v, uint32_t src) {
        return static_cast<uint32_t>(
            __nvvm_shfl_sync_idx_i32(mask(base), static_cast<int32_t>(v), src, kSeg | 0x1f));
    }
    static uint32_t shfl_xor(uint32_t base, uint32_t v, uint32_t m) {
        return static_cast<uint32_t>(
            __nvvm_shfl_sync_bfly_i32(mask(base), static_cast<int32_t>(v), m, kSeg | 0x1f));
    }
    // Out-of-chunk sources clamp to the caller's own value.
    static uint32_t shfl_down(uint32_t base, uint32_t v, uint32_t delta) {
        return static_cast<uint32_t>(
            __nvvm_shfl_sync_down_i32(mask(base), static_cast<int32_t>(v), delta, kSeg | 0x1f));
    }
    static uint32_t shfl_up(uint32_t base, uint32_t v, uint32_t delta) {
        return static_cast<uint32_t>(
            __nvvm_shfl_sync_up_i32(mask(base), static_cast<int32_t>(v), delta, kSeg));
    }
    // Lanes outside the chunk can be set (the full mask, or converged
    // neighbours), so extract the chunk's P bits.
    static uint32_t ballot(uint32_t base, bool pred) {
        const uint32_t bits = static_cast<uint32_t>(__nvvm_vote_ballot_sync(mask(base), pred));
        if constexpr (P >= 32) return bits;
        else return (bits >> base) & kLow;
    }
    static void barrier(uint32_t base) { __nvvm_bar_warp_sync(mask(base)); }

    template <typename T, typename Op>
    static T butterfly(uint32_t base, T v, Op op) {
        constexpr uint32_t kW = static_cast<uint32_t>((sizeof(T) + 3) / 4);
#pragma unroll
        for (uint32_t m = 1; m < P; m <<= 1) {
            uint32_t w[kW] = {};
            std::memcpy(w, &v, sizeof(T));
#pragma unroll
            for (uint32_t i = 0; i < kW; ++i) w[i] = shfl_xor(base, w[i], m);
            T other;
            std::memcpy(&other, w, sizeof(T));
            v = op(v, other);
        }
        return v;
    }
};

// Not CSE-able (the intrinsic is convergent and reads the active set), so
// each call tests the warp at that point.
inline bool warp_converged() { return __nvvm_activemask() == ~0u; }

template <typename Op, typename T>
inline constexpr bool is_op_v = std::is_same_v<Op, T>;

// redux.sync (sm_80+) only with the full mask: a per-chunk mask makes ptxas
// serialise the warp into WARPSYNC.EXCLUSIVE groups, slower than a butterfly.
template <typename T, typename Op>
inline constexpr bool redux_ok_v =
    (std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t>) &&
    (is_op_v<Op, sycl::plus<T>> || is_op_v<Op, sycl::plus<>> || is_op_v<Op, sycl::minimum<T>> ||
     is_op_v<Op, sycl::minimum<>> || is_op_v<Op, sycl::maximum<T>> || is_op_v<Op, sycl::maximum<>> ||
     is_op_v<Op, sycl::bit_and<T>> || is_op_v<Op, sycl::bit_and<>> || is_op_v<Op, sycl::bit_or<T>> ||
     is_op_v<Op, sycl::bit_or<>> || is_op_v<Op, sycl::bit_xor<T>> || is_op_v<Op, sycl::bit_xor<>>);

template <typename T, typename Op>
inline T redux_full(T v, Op) {
#if defined(__SYCL_CUDA_ARCH__) && __SYCL_CUDA_ARCH__ >= 800
    constexpr uint32_t kAll = ~0u;
    constexpr bool kSigned = std::is_same_v<T, int32_t>;
    if constexpr (is_op_v<Op, sycl::plus<T>> || is_op_v<Op, sycl::plus<>>) {
        return static_cast<T>(__nvvm_redux_sync_add(static_cast<int32_t>(v), kAll));
    } else if constexpr (is_op_v<Op, sycl::minimum<T>> || is_op_v<Op, sycl::minimum<>>) {
        if constexpr (kSigned) return __nvvm_redux_sync_min(v, kAll);
        else return __nvvm_redux_sync_umin(v, kAll);
    } else if constexpr (is_op_v<Op, sycl::maximum<T>> || is_op_v<Op, sycl::maximum<>>) {
        if constexpr (kSigned) return __nvvm_redux_sync_max(v, kAll);
        else return __nvvm_redux_sync_umax(v, kAll);
    } else if constexpr (is_op_v<Op, sycl::bit_and<T>> || is_op_v<Op, sycl::bit_and<>>) {
        return static_cast<T>(__nvvm_redux_sync_and(static_cast<int32_t>(v), kAll));
    } else if constexpr (is_op_v<Op, sycl::bit_or<T>> || is_op_v<Op, sycl::bit_or<>>) {
        return static_cast<T>(__nvvm_redux_sync_or(static_cast<int32_t>(v), kAll));
    } else {
        return static_cast<T>(__nvvm_redux_sync_xor(static_cast<int32_t>(v), kAll));
    }
#else
    return v;
#endif
}

#if defined(__SYCL_CUDA_ARCH__) && __SYCL_CUDA_ARCH__ >= 800
inline constexpr bool kHasRedux = true;
#else
inline constexpr bool kHasRedux = false;
#endif

} // namespace nvptx_detail

// The primitives of a diverged warp: always the chunk mask. region() hands this
// out once instead of testing convergence per word.
template <uint32_t P>
struct NvptxChunkOps {
    using C = nvptx_detail::Mask<P, false>;
    static uint32_t shfl_idx(const sycl::sub_group&, uint32_t b, uint32_t v, uint32_t s) { return C::shfl_idx(b, v, s); }
    static uint32_t shfl_xor(const sycl::sub_group&, uint32_t b, uint32_t v, uint32_t m) { return C::shfl_xor(b, v, m); }
    static uint32_t shfl_down(const sycl::sub_group&, uint32_t b, uint32_t v, uint32_t d) { return C::shfl_down(b, v, d); }
    static uint32_t shfl_up(const sycl::sub_group&, uint32_t b, uint32_t v, uint32_t d) { return C::shfl_up(b, v, d); }
    static uint32_t ballot(const sycl::sub_group&, uint32_t b, bool pred) { return C::ballot(b, pred); }
    static void barrier(const sycl::sub_group&, uint32_t b) { C::barrier(b); }
};

template <uint32_t P, bool Masked>
struct NvptxBackend {
    static constexpr const char* name = "nvptx";
    static constexpr bool masked_by_default = true;

    // Masked chunks narrower than the warp are the only case that needs the
    // run-time test; everything else is one immediate-mask instruction.
    static constexpr bool kDynamic = Masked && P < 32;
    using Full = nvptx_detail::Mask<P, true>;
    using Chunk = nvptx_detail::Mask<P, false>;

    static uint32_t shfl_idx(const sycl::sub_group&, uint32_t base, uint32_t v, uint32_t src) {
        if constexpr (kDynamic) {
            if (!nvptx_detail::warp_converged()) return Chunk::shfl_idx(base, v, src);
        }
        return Full::shfl_idx(base, v, src);
    }

    static uint32_t shfl_xor(const sycl::sub_group&, uint32_t base, uint32_t v, uint32_t mask) {
        if constexpr (kDynamic) {
            if (!nvptx_detail::warp_converged()) return Chunk::shfl_xor(base, v, mask);
        }
        return Full::shfl_xor(base, v, mask);
    }

    static uint32_t shfl_down(const sycl::sub_group&, uint32_t base, uint32_t v, uint32_t delta) {
        if constexpr (kDynamic) {
            if (!nvptx_detail::warp_converged()) return Chunk::shfl_down(base, v, delta);
        }
        return Full::shfl_down(base, v, delta);
    }

    static uint32_t shfl_up(const sycl::sub_group&, uint32_t base, uint32_t v, uint32_t delta) {
        if constexpr (kDynamic) {
            if (!nvptx_detail::warp_converged()) return Chunk::shfl_up(base, v, delta);
        }
        return Full::shfl_up(base, v, delta);
    }

    static uint32_t ballot(const sycl::sub_group&, uint32_t base, bool pred) {
        if constexpr (kDynamic) {
            if (!nvptx_detail::warp_converged()) return Chunk::ballot(base, pred);
        }
        return Full::ballot(base, pred);
    }

    static void barrier(const sycl::sub_group&, uint32_t base) {
        if constexpr (kDynamic) {
            if (!nvptx_detail::warp_converged()) return Chunk::barrier(base);
        }
        Full::barrier(base);
    }

    // One convergence test for everything f does: f gets the full-mask (lockstep)
    // backend when the warp is converged, the chunk-mask ops otherwise.
    template <typename F>
    static decltype(auto) region(F&& f) {
        if constexpr (kDynamic) {
            if (!nvptx_detail::warp_converged()) return f(NvptxChunkOps<P>{});
        }
        return f(NvptxBackend<P, false>{});
    }

    // P = 32 integer reductions are one redux.sync. A masked P < 32 partition
    // takes every reduction so the whole butterfly shares one convergence test.
    template <typename T, typename Op>
    static constexpr bool has_native_reduce =
        (P >= 32 && nvptx_detail::kHasRedux && nvptx_detail::redux_ok_v<T, Op>) ||
        (kDynamic && std::is_trivially_copyable_v<T>);

    template <typename T, typename Op>
    static T reduce(const sycl::sub_group&, uint32_t base, T v, Op op) {
        if constexpr (P >= 32 && nvptx_detail::kHasRedux && nvptx_detail::redux_ok_v<T, Op>) {
            return nvptx_detail::redux_full(v, op);
        } else {
            if (nvptx_detail::warp_converged()) return Full::butterfly(base, v, op);
            return Chunk::butterfly(base, v, op);
        }
    }
};

} // namespace batchlas::sgp
