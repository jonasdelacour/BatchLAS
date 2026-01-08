#pragma once

#include <util/minibench.hh>
#include <util/bench_structured.hh>

#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>

// This header provides the SYCL/BatchLAS-specific structured-benchmark sugar:
//   state.SetKernel(q, arg1, arg2, ..., kernel_fn);
//
// It is an extension on top of the generic `minibench` framework.

namespace minibench {
namespace detail {

template <typename...>
struct always_false : std::false_type {};

template <typename StateT, typename Q, typename Tuple, typename KernelFn, size_t... Is>
inline void set_kernel_from_tuple(StateT& state,
                                 std::shared_ptr<Q> q,
                                 Tuple&& tup,
                                 KernelFn&& kernel_fn,
                                 std::index_sequence<Is...>) {
    auto k = std::forward<KernelFn>(kernel_fn);

    // Adapter: supports either
    //  (a) k(x0, x1, ...)
    //  (b) k(*q, kernel_arg(x0), kernel_arg(x1), ...)
    // This enables callsites to pass a function pointer like `sytrd_cta<B>` as the kernel.
    auto adapter = [q, k = std::move(k)](auto&... xs) mutable {
        if constexpr (std::is_invocable_v<decltype(k)&, decltype(xs)&...>) {
            k(xs...);
        } else if constexpr (std::is_invocable_v<decltype(k)&,
                                                ::Queue&,
                                                decltype(::bench::detail::kernel_arg(xs))...>) {
            k(*q, ::bench::detail::kernel_arg(xs)...);
        } else {
            static_assert(always_false<decltype(k)>::value,
                          "Kernel callable is not invocable as k(args...) nor as k(*q, kernel_arg(args)...) ");
        }
    };

    // Take ownership of all managed args automatically.
    state.SetKernel(::bench::Kernel(q,
                                    std::move(std::get<Is>(tup))...)
                      (std::move(adapter)));
}

} // namespace detail

template <typename Q, typename... Args>
inline void State::SetKernel(std::shared_ptr<Q> q, Args&&... args) {
    static_assert(sizeof...(Args) >= 1,
                  "State::SetKernel(q, ..., kernel_fn) requires at least one argument (the kernel callable)");

    auto tup = std::forward_as_tuple(std::forward<Args>(args)...);
    constexpr size_t N = sizeof...(Args);
    auto&& kernel_fn = std::get<N - 1>(tup);

    detail::set_kernel_from_tuple(*this,
                                 std::move(q),
                                 tup,
                                 std::forward<decltype(kernel_fn)>(kernel_fn),
                                 std::make_index_sequence<N - 1>{});
}

} // namespace minibench
