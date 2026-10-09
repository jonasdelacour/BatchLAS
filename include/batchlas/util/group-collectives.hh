#pragma once

/// @file
/// @brief `batchlas::portable::` group broadcast, shuffles and sums for any trivially copyable or complex `T`.
///
/// Under DPC++ these names are the `sycl::` functions themselves (using-declarations),
/// so a call compiles to exactly the direct `sycl::` call. Under AdaptiveCpp they are
/// wrappers: a value not 1, 2, 4 or 8 bytes wide moves word by word and a `std::complex` sum
/// reduces its real and imaginary parts separately. A separate namespace keeps
/// unqualified calls on sub-group partitions (sg_partition.hh) resolving as before.
///
/// Installed with the rest of `include/batchlas` (the install rule copies the
/// tree wholesale); the device BLAS headers include it. Not a stable interface.
/// @ingroup api_internal_helpers

#include <batchlas/backend_config.h>

#include <sycl/sycl.hpp>

#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace batchlas::portable {

#if BATCHLAS_SYCL_IMPL_ACPP || defined(BATCHLAS_DOXYGEN)

namespace detail {

// AdaptiveCpp's SSCP collectives move a T of exactly 1, 2, 4 or 8 bytes (static_assert otherwise);
// its reductions take arithmetic T only.
template <typename T>
inline constexpr bool kSplitWords = sizeof(T) > 8 || (sizeof(T) & (sizeof(T) - 1)) != 0;

template <typename T>
struct is_std_complex : std::false_type {};
template <typename R>
struct is_std_complex<std::complex<R>> : std::true_type {};

template <typename T, typename Op>
inline constexpr bool kSplitComplexSum =
    is_std_complex<T>::value && (std::is_same_v<Op, sycl::plus<T>> || std::is_same_v<Op, sycl::plus<>>);

template <typename T, typename F>
inline T map_words(const T& v, F&& f) {
    static_assert(std::is_trivially_copyable_v<T>, "group collectives need a trivially copyable T");
    using W = std::conditional_t<sizeof(T) % 8 == 0, std::uint64_t, std::uint32_t>;
    constexpr std::size_t kN = (sizeof(T) + sizeof(W) - 1) / sizeof(W);
    W w[kN] = {};
    std::memcpy(w, &v, sizeof(T));
#pragma clang loop unroll(full)
    for (std::size_t i = 0; i < kN; ++i) w[i] = f(w[i]);
    T out;
    std::memcpy(&out, w, sizeof(T));
    return out;
}

} // namespace detail

/// @addtogroup api_internal_helpers
/// @{

/// @brief `sycl::group_broadcast` for any trivially copyable `T`. Collective.
template <typename Group, typename T, typename... Id>
    requires sycl::is_group_v<Group>
inline T group_broadcast(Group g, T x, Id... id) {
    if constexpr (detail::kSplitWords<T>) {
        return detail::map_words(x, [&](auto w) { return sycl::group_broadcast(g, w, id...); });
    } else {
        return sycl::group_broadcast(g, x, id...);
    }
}

/// @brief `sycl::select_from_group` for any trivially copyable `T`. Collective.
template <typename Group, typename T, typename Id>
    requires sycl::is_group_v<Group>
inline T select_from_group(Group g, T x, Id id) {
    if constexpr (detail::kSplitWords<T>) {
        return detail::map_words(x, [&](auto w) { return sycl::select_from_group(g, w, id); });
    } else {
        return sycl::select_from_group(g, x, id);
    }
}

/// @brief `sycl::shift_group_left` for any trivially copyable `T`. Collective.
template <typename Group, typename T, typename... Delta>
    requires sycl::is_group_v<Group>
inline T shift_group_left(Group g, T x, Delta... delta) {
    if constexpr (detail::kSplitWords<T>) {
        return detail::map_words(x, [&](auto w) { return sycl::shift_group_left(g, w, delta...); });
    } else {
        return sycl::shift_group_left(g, x, delta...);
    }
}

/// @brief `sycl::shift_group_right` for any trivially copyable `T`. Collective.
template <typename Group, typename T, typename... Delta>
    requires sycl::is_group_v<Group>
inline T shift_group_right(Group g, T x, Delta... delta) {
    if constexpr (detail::kSplitWords<T>) {
        return detail::map_words(x, [&](auto w) { return sycl::shift_group_right(g, w, delta...); });
    } else {
        return sycl::shift_group_right(g, x, delta...);
    }
}

/// @brief `sycl::permute_group_by_xor` for any trivially copyable `T`. Collective.
template <typename Group, typename T, typename Mask>
    requires sycl::is_group_v<Group>
inline T permute_group_by_xor(Group g, T x, Mask mask) {
    if constexpr (detail::kSplitWords<T>) {
        return detail::map_words(x, [&](auto w) { return sycl::permute_group_by_xor(g, w, mask); });
    } else {
        return sycl::permute_group_by_xor(g, x, mask);
    }
}

/// @brief `sycl::reduce_over_group`, also for a `std::complex` sum. Collective.
template <typename Group, typename T, typename Op>
    requires sycl::is_group_v<Group>
inline T reduce_over_group(Group g, T x, Op op) {
    if constexpr (detail::kSplitComplexSum<T, Op>) {
        using R = typename T::value_type;
        return T(sycl::reduce_over_group(g, x.real(), sycl::plus<R>()),
                 sycl::reduce_over_group(g, x.imag(), sycl::plus<R>()));
    } else {
        return sycl::reduce_over_group(g, x, op);
    }
}

/// @brief `sycl::joint_reduce` with an initial value, also for a `std::complex` sum. Collective.
/// @pre `first` and `last` are raw pointers: AdaptiveCpp has no accessor-iterator overload.
template <typename Group, typename Ptr, typename T, typename Op>
    requires sycl::is_group_v<Group>
inline T joint_reduce(Group g, Ptr first, Ptr last, T init, Op op) {
    if constexpr (detail::kSplitComplexSum<T, Op>) {
        T part = T(0);
        for (Ptr p = first + g.get_local_linear_id(); p < last; p += g.get_local_linear_range()) part += *p;
        return op(init, portable::reduce_over_group(g, part, op));
    } else {
        return sycl::joint_reduce(g, first, last, init, op);
    }
}

/// @}

#else

using sycl::group_broadcast;
using sycl::joint_reduce;
using sycl::permute_group_by_xor;
using sycl::reduce_over_group;
using sycl::select_from_group;
using sycl::shift_group_left;
using sycl::shift_group_right;

#endif

} // namespace batchlas::portable
