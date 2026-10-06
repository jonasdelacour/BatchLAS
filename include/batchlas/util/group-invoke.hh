#pragma once

#include <sycl/sycl.hpp>

#include <functional>
#include <type_traits>
#include <utility>

namespace batchlas {

namespace detail {

template <typename Group>
using group_type_t = std::remove_cv_t<std::remove_reference_t<Group>>;

template <typename Group>
inline constexpr bool is_sub_group_v = std::is_same_v<group_type_t<Group>, sycl::sub_group>;

template <typename Group, typename T>
inline constexpr T broadcast_from_leader_impl(const Group& group, T value) {
    if constexpr (requires { sg_leader_broadcast(group, value); }) {
        // SubGroupPartition<P> (sg_partition.hh) provides sg_leader_broadcast() via ADL.
        return sg_leader_broadcast(group, value);
    } else {
        return sycl::group_broadcast(group, value);
    }
}

} // namespace detail

/// @addtogroup internal_helpers
/// @{

/// @brief Calls `fn(args...)` on the group's leader work-item only.
///
/// No barrier and no broadcast: other work-items do not see side effects until
/// the caller synchronises.
/// @tparam Group  any type with `leader()`: a SYCL group, sub-group, or sub-group partition
template <typename Group, typename Fn, typename... Args>
inline constexpr void invoke_one(const Group& group, Fn&& fn, Args&&... args) {
    if (group.leader()) {
        std::invoke(std::forward<Fn>(fn), std::forward<Args>(args)...);
    }
}

/// @brief Returns the leader's `value` on every work-item of `group`. Collective.
///
/// Uses the group's `sg_leader_broadcast` when one is found by ADL (sub-group
/// partitions), else `sycl::group_broadcast`.
/// @pre `T` is trivially copyable.
template <typename Group, typename T>
inline constexpr T broadcast_from_leader(const Group& group, T value) {
    static_assert(std::is_trivially_copyable_v<T>,
                  "broadcast_from_leader requires T to be trivially copyable");
    return detail::broadcast_from_leader_impl(group, value);
}

/// @brief Calls `fn(args...)` on the leader and returns its result on every work-item. Collective.
/// @return the leader's result; non-leaders contribute a value-initialised placeholder that is discarded
/// @pre the result type is trivially copyable.
template <typename Group, typename Fn, typename... Args>
inline constexpr auto invoke_one_broadcast(const Group& group, Fn&& fn, Args&&... args)
    -> std::invoke_result_t<Fn, Args...> {
    using R = std::invoke_result_t<Fn, Args...>;
    static_assert(std::is_trivially_copyable_v<R>,
                  "invoke_one_broadcast requires return type to be trivially copyable");

    R value{};
    if (group.leader()) {
        value = std::invoke(std::forward<Fn>(fn), std::forward<Args>(args)...);
    }

    return detail::broadcast_from_leader_impl(group, value);
}

/// @}

} // namespace batchlas
