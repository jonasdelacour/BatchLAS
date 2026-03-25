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

template <typename Group>
inline constexpr bool is_user_constructed_group_v =
    sycl::ext::oneapi::experimental::is_user_constructed_group_v<group_type_t<Group>>;

template <typename Group, typename T>
inline constexpr T broadcast_from_leader_impl(const Group& group, T value) {
    if constexpr (is_user_constructed_group_v<Group>) {
        return sycl::select_from_group(group, value, typename group_type_t<Group>::id_type{});
    } else {
        return sycl::group_broadcast(group, value);
    }
}

} // namespace detail

template <typename Group, typename Fn, typename... Args>
inline constexpr void invoke_one(const Group& group, Fn&& fn, Args&&... args) {
    if (group.leader()) {
        std::invoke(std::forward<Fn>(fn), std::forward<Args>(args)...);
    }
}

template <typename Group, typename T>
inline constexpr T broadcast_from_leader(const Group& group, T value) {
    static_assert(std::is_trivially_copyable_v<T>,
                  "broadcast_from_leader requires T to be trivially copyable");
    return detail::broadcast_from_leader_impl(group, value);
}

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

} // namespace batchlas
