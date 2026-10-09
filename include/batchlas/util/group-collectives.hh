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

template <typename G>
inline constexpr bool kWorkGroup = false;
template <int D>
inline constexpr bool kWorkGroup<sycl::group<D>> = true;

template <typename T>
inline constexpr bool kFloating = std::is_floating_point_v<T> || std::is_same_v<T, sycl::half>;

template <typename T>
struct sycl_vec : std::false_type {};
template <typename R, int N>
struct sycl_vec<sycl::vec<R, N>> : std::true_type {
    using elem = R;
    static constexpr int size = N;
};

// sycl::plus<vec<R, N>> -> sycl::plus<R>, for a component-wise reduce.
template <typename Op, typename R>
struct rebind_op;
template <template <typename> class OpT, typename V, typename R>
struct rebind_op<OpT<V>, R> {
    using type = OpT<R>;
};

template <typename T>
using local_ptr = __attribute__((address_space(3))) T*;

// One scratch per T per kernel, shared by every op and call site, as AdaptiveCpp's own is.
template <typename T>
__attribute__((always_inline)) inline local_ptr<T> wg_scratch() {
    local_ptr<T> p = nullptr;
    __acpp_if_target_sscp(static __attribute__((loader_uninitialized))
                          __attribute__((address_space(3))) T buf[32];
                          p = &buf[0];)
    return p;
}

// TRAP (AdaptiveCpp 25.10): its work-group reduce of a floating value is rounded to ~16 (float)
// or ~42 (double) mantissa bits. wg_reduce bit-casts the result to an integer and broadcasts
// it through its FLOAT scratch, a value conversion (sscp/builtins/detail/reduction.hpp:101,
// detail/broadcast.hpp:29). Same algorithm with a typed final broadcast; sub-group reduces are exact.
template <typename Group, typename T, typename Op>
inline T wg_reduce_floating(Group g, T x, Op op) {
    // Host-compiled kernels (omp.library-only) have no wg_scratch.
    __acpp_if_target_host(return sycl::reduce_over_group(g, x, op);)
    const sycl::sub_group sg{};
    const T s = sycl::reduce_over_group(sg, x, op);
    const std::uint32_t width = sg.get_max_local_range()[0];
    const std::uint32_t nsg = (static_cast<std::uint32_t>(g.get_local_linear_range()) + width - 1) / width;
    if (nsg == 1) return s;
    const std::uint32_t lid = static_cast<std::uint32_t>(g.get_local_linear_id());
    const std::uint32_t sgid = lid / width;
    const bool leader = sg.get_local_linear_id() == 0;
    local_ptr<T> buf = wg_scratch<T>();
    if (leader && sgid < 32) buf[sgid] = s;
    sycl::group_barrier(g);
    for (std::uint32_t i = 32; i < nsg; i += 32) {
        if (leader && sgid >= i && sgid < i + 32) buf[sgid % 32] = op(T(buf[sgid % 32]), s);
        sycl::group_barrier(g);
    }
    if (lid == 0) {
        T acc = T(buf[0]);
        for (std::uint32_t k = 1; k < (nsg < 32 ? nsg : 32); ++k) acc = op(acc, T(buf[k]));
        buf[0] = acc;
    }
    sycl::group_barrier(g);
    const T r = T(buf[0]);
    sycl::group_barrier(g);
    return r;
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

/// @brief `sycl::reduce_over_group`, also for a `std::complex` sum; a work-group floating
/// result is exact (see detail::wg_reduce_floating). Collective.
template <typename Group, typename T, typename Op>
    requires sycl::is_group_v<Group>
inline T reduce_over_group(Group g, T x, Op op) {
    if constexpr (detail::kSplitComplexSum<T, Op>) {
        using R = typename T::value_type;
        return T(portable::reduce_over_group(g, x.real(), sycl::plus<R>()),
                 portable::reduce_over_group(g, x.imag(), sycl::plus<R>()));
    } else if constexpr (ACPP_LIBKERNEL_IS_DEVICE_PASS_SSCP && detail::kWorkGroup<Group> &&
                         detail::kFloating<T>) {
        return detail::wg_reduce_floating(g, x, op);
    } else if constexpr (ACPP_LIBKERNEL_IS_DEVICE_PASS_SSCP && detail::kWorkGroup<Group> &&
                         detail::sycl_vec<T>::value) {
        using R = typename detail::sycl_vec<T>::elem;
        using ElemOp = typename detail::rebind_op<Op, R>::type;
        T out;
        for (int i = 0; i < detail::sycl_vec<T>::size; ++i)
            out[i] = portable::reduce_over_group(g, static_cast<R>(x[i]), ElemOp());
        return out;
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
    } else if constexpr (detail::kFloating<T> && sycl::has_known_identity_v<Op, T>) {
        // Not sycl::joint_reduce: it pads idle items with numeric_limits<T>::min() as the maximum
        // identity, the smallest POSITIVE value (sscp/group_functions.hpp:347).
        T part = sycl::known_identity_v<Op, T>;
        for (Ptr p = first + g.get_local_linear_id(); p < last; p += g.get_local_linear_range())
            part = op(part, static_cast<T>(*p));
        return op(init, portable::reduce_over_group(g, part, op));
    } else {
        return sycl::joint_reduce(g, first, last, init, op);
    }
}

/// @brief `sycl::joint_reduce` without an initial value (an empty range gives `T{}`). Collective.
/// @pre `first` and `last` are raw pointers.
template <typename Group, typename Ptr, typename Op>
    requires sycl::is_group_v<Group>
inline auto joint_reduce(Group g, Ptr first, Ptr last, Op op) {
    using T = std::remove_cv_t<std::remove_reference_t<decltype(*first)>>;
    if constexpr ((detail::kFloating<T> && sycl::has_known_identity_v<Op, T>) ||
                  detail::kSplitComplexSum<T, Op>) {
        if (first == last) return T{};
        // sycl::joint_reduce's mapping (item i reads first[i], first[i + range], ...): stedc fills
        // its scratch per item and calls this with no barrier, as that mapping allows.
        if constexpr (detail::kSplitComplexSum<T, Op>) return portable::joint_reduce(g, first, last, T{}, op);
        else return portable::joint_reduce(g, first, last, sycl::known_identity_v<Op, T>, op);
    } else {
        return sycl::joint_reduce(g, first, last, op);
    }
}

/// @brief `sycl::joint_exclusive_scan` with an initial value; `result` may equal `first`. Collective.
/// @pre `first`, `last` and `result` are raw pointers.
template <typename Group, typename InPtr, typename OutPtr, typename T, typename Op>
    requires sycl::is_group_v<Group>
inline OutPtr joint_exclusive_scan(Group g, InPtr first, InPtr last, OutPtr result, T init, Op op) {
    // AdaptiveCpp's own version writes result[i + 1] while scanning first[i], which corrupts the
    // next chunk's input when the scan is in place. This one reads and writes index i in one chunk.
    const std::ptrdiff_t n = last - first;
    const std::ptrdiff_t lid = static_cast<std::ptrdiff_t>(g.get_local_linear_id());
    const std::ptrdiff_t width = static_cast<std::ptrdiff_t>(g.get_local_linear_range());
    T carry = init;
    for (std::ptrdiff_t chunk = 0; chunk < n; chunk += width) {
        const std::ptrdiff_t i = chunk + lid;
        const T x = i < n ? static_cast<T>(first[i]) : carry;
        const T out = sycl::exclusive_scan_over_group(g, x, carry, op);
        if (i < n) result[i] = out;
        carry = sycl::group_broadcast(g, op(out, x), width - 1);
    }
    sycl::group_barrier(g);
    return result + n;
}

/// @}

#else

using sycl::group_broadcast;
using sycl::joint_exclusive_scan;
using sycl::joint_reduce;
using sycl::permute_group_by_xor;
using sycl::reduce_over_group;
using sycl::select_from_group;
using sycl::shift_group_left;
using sycl::shift_group_right;

#endif

} // namespace batchlas::portable
