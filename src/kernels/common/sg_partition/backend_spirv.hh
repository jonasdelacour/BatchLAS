#pragma once
// SPIR-V backend (Intel GPUs, the OpenCL CPU device); device pass only.
//
// Every primitive is a Subgroup-scope *non-uniform* group instruction. Unlike
// the ones behind sycl::sub_group, these are defined when only some lanes are
// active, so a chunk whose P lanes all arrive is exact whatever its
// neighbours do. Masked and lockstep share the code, except the barrier.
// Reductions are one ClusteredReduce with ClusterSize = P.
//
// masked_by_default: on SIMD hardware divergence is an execution mask, so a
// non-uniform op is the uniform instruction under that mask. Masked is free.

#include <sycl/sycl.hpp>

#include <complex>
#include <cstdint>
#include <type_traits>

namespace batchlas::sgp {

namespace spirv_detail {

inline constexpr auto kScope = __spv::Scope::Subgroup;
inline constexpr uint32_t kClustered = static_cast<uint32_t>(__spv::GroupOperation::ClusteredReduce);

template <typename Op, template <typename> class F, typename T>
inline constexpr bool is_op_v = std::is_same_v<Op, F<T>> || std::is_same_v<Op, F<void>>;

// T with a native clustered instruction: 32/64-bit integers and float/double.
template <typename T>
inline constexpr bool is_int_v = std::is_integral_v<T> && !std::is_same_v<T, bool> &&
                                 (sizeof(T) == 4 || sizeof(T) == 8);
template <typename T>
inline constexpr bool is_fp_v = std::is_same_v<T, float> || std::is_same_v<T, double>;

template <typename T>
struct is_complex : std::false_type {};
template <typename R>
struct is_complex<std::complex<R>> : std::bool_constant<is_fp_v<R>> {};

// Canonical fixed-width type, so long / long long pick one overload.
template <typename T>
using int_t = std::conditional_t<std::is_signed_v<T>,
                                 std::conditional_t<sizeof(T) == 4, int32_t, int64_t>,
                                 std::conditional_t<sizeof(T) == 4, uint32_t, uint64_t>>;

template <typename T, typename Op>
inline constexpr bool scalar_supported_v =
    (is_fp_v<T> && (is_op_v<Op, sycl::plus, T> || is_op_v<Op, sycl::multiplies, T> ||
                    is_op_v<Op, sycl::minimum, T> || is_op_v<Op, sycl::maximum, T>)) ||
    (is_int_v<T> && (is_op_v<Op, sycl::plus, T> || is_op_v<Op, sycl::multiplies, T> ||
                     is_op_v<Op, sycl::minimum, T> || is_op_v<Op, sycl::maximum, T> ||
                     is_op_v<Op, sycl::bit_and, T> || is_op_v<Op, sycl::bit_or, T> ||
                     is_op_v<Op, sycl::bit_xor, T>));

template <typename T, typename Op>
inline constexpr bool supported_v =
    scalar_supported_v<T, Op> || (is_complex<T>::value && is_op_v<Op, sycl::plus, T>);

template <uint32_t P, typename T, typename Op>
inline T clustered_scalar(T v) {
    if constexpr (is_fp_v<T>) {
        if constexpr (is_op_v<Op, sycl::plus, T>) return __spirv_GroupNonUniformFAdd(kScope, kClustered, v, P);
        else if constexpr (is_op_v<Op, sycl::multiplies, T>) return __spirv_GroupNonUniformFMul(kScope, kClustered, v, P);
        else if constexpr (is_op_v<Op, sycl::minimum, T>) return __spirv_GroupNonUniformFMin(kScope, kClustered, v, P);
        else return __spirv_GroupNonUniformFMax(kScope, kClustered, v, P);
    } else {
        using I = int_t<T>;
        const I x = static_cast<I>(v);
        if constexpr (is_op_v<Op, sycl::plus, T>) return static_cast<T>(__spirv_GroupNonUniformIAdd(kScope, kClustered, x, P));
        else if constexpr (is_op_v<Op, sycl::multiplies, T>) return static_cast<T>(__spirv_GroupNonUniformIMul(kScope, kClustered, x, P));
        else if constexpr (is_op_v<Op, sycl::bit_and, T>) return static_cast<T>(__spirv_GroupNonUniformBitwiseAnd(kScope, kClustered, x, P));
        else if constexpr (is_op_v<Op, sycl::bit_or, T>) return static_cast<T>(__spirv_GroupNonUniformBitwiseOr(kScope, kClustered, x, P));
        else if constexpr (is_op_v<Op, sycl::bit_xor, T>) return static_cast<T>(__spirv_GroupNonUniformBitwiseXor(kScope, kClustered, x, P));
        else if constexpr (std::is_signed_v<T>) {
            if constexpr (is_op_v<Op, sycl::minimum, T>) return static_cast<T>(__spirv_GroupNonUniformSMin(kScope, kClustered, x, P));
            else return static_cast<T>(__spirv_GroupNonUniformSMax(kScope, kClustered, x, P));
        } else {
            if constexpr (is_op_v<Op, sycl::minimum, T>) return static_cast<T>(__spirv_GroupNonUniformUMin(kScope, kClustered, x, P));
            else return static_cast<T>(__spirv_GroupNonUniformUMax(kScope, kClustered, x, P));
        }
    }
}

} // namespace spirv_detail

template <uint32_t P, bool Masked>
struct SpirvBackend {
    static constexpr const char* name = "spirv";
    static constexpr bool masked_by_default = true;

    // src is a lane of this chunk, 0 <= src < P.
    static uint32_t shfl_idx(const sycl::sub_group&, uint32_t base, uint32_t v, uint32_t src) {
        return __spirv_GroupNonUniformShuffle(spirv_detail::kScope, v, base + src);
    }

    // mask < P, so lane ^ mask stays in the chunk.
    static uint32_t shfl_xor(const sycl::sub_group&, uint32_t, uint32_t v, uint32_t mask) {
        return __spirv_GroupNonUniformShuffleXor(spirv_detail::kScope, v, mask);
    }

    // A source outside the chunk is another chunk's lane (possibly inactive):
    // undefined, as the front end allows.
    static uint32_t shfl_down(const sycl::sub_group&, uint32_t, uint32_t v, uint32_t delta) {
        return __spirv_GroupNonUniformShuffleDown(spirv_detail::kScope, v, delta);
    }

    static uint32_t shfl_up(const sycl::sub_group&, uint32_t, uint32_t v, uint32_t delta) {
        return __spirv_GroupNonUniformShuffleUp(spirv_detail::kScope, v, delta);
    }

    // Inactive invocations contribute 0 bits; only this chunk's P bits are kept.
    static uint32_t ballot(const sycl::sub_group&, uint32_t base, bool pred) {
        const auto b = __spirv_GroupNonUniformBallot(spirv_detail::kScope, pred);
        const uint64_t bits = static_cast<uint64_t>(b[0]) | (static_cast<uint64_t>(b[1]) << 32);
        constexpr uint64_t chunk = P >= 32 ? 0xffffffffull : ((1ull << P) - 1ull);
        return static_cast<uint32_t>((bits >> base) & chunk);
    }

    // SPIR-V has no barrier over part of a sub-group, and OpControlBarrier at
    // Subgroup scope needs every invocation. A chunk's lanes share one
    // instruction stream on SIMD hardware, so a masked partition only needs the
    // memory ordering; a lockstep one takes the full sub-group barrier.
    static void barrier(const sycl::sub_group&, uint32_t) {
        constexpr uint32_t sem = static_cast<uint32_t>(__spv::MemorySemanticsMask::SequentiallyConsistent) |
                                 static_cast<uint32_t>(__spv::MemorySemanticsMask::SubgroupMemory) |
                                 static_cast<uint32_t>(__spv::MemorySemanticsMask::WorkgroupMemory) |
                                 static_cast<uint32_t>(__spv::MemorySemanticsMask::CrossWorkgroupMemory);
        if constexpr (Masked) {
            __spirv_MemoryBarrier(spirv_detail::kScope, sem);
        } else {
            __spirv_ControlBarrier(spirv_detail::kScope, spirv_detail::kScope, sem);
        }
    }

    template <typename T, typename Op>
    static constexpr bool has_native_reduce = P > 1u && spirv_detail::supported_v<T, Op>;

    template <typename T, typename Op>
    static T reduce(const sycl::sub_group&, uint32_t, T v, Op) {
        if constexpr (spirv_detail::is_complex<T>::value) {
            using R = typename T::value_type;
            return T(spirv_detail::clustered_scalar<P, R, sycl::plus<R>>(v.real()),
                     spirv_detail::clustered_scalar<P, R, sycl::plus<R>>(v.imag()));
        } else {
            return spirv_detail::clustered_scalar<P, T, Op>(v);
        }
    }
};

} // namespace batchlas::sgp
