#pragma once

#include <sycl/sycl.hpp>

#include <functional>
#include <type_traits>
#include <utility>

namespace batchlas {

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
    return sycl::select_from_group(group, value, 0);
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

    return sycl::select_from_group(group, value, 0);
}

} // namespace batchlas
