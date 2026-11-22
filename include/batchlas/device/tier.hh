// SPDX-License-Identifier: MIT
#pragma once
/** @file
 * @brief Device tiers: how a LAPACK-level routine runs inside a kernel, chosen at compile time,
 * and with_tier, which picks one on the host for a run-time n.
 * @ingroup api_device_launch
 *
 * A tier is a type, so one kernel never pays the registers of two paths. Per routine (a tag type
 * in device::routine) and scalar T, `tier_info<Routine, Tier, T>` states the executor type,
 * `wg_size`, `sg_size`, `per_group`, `admits(n)` and `buffer_size(n)`; `tiers_t<Routine, T>` lists
 * the routine's tiers smallest first. launch.hh turns a tier into a launch (nd_range, executor,
 * matrix slot, workspace).
 */
// A run-time `if (n <= 32)` between two tiers compiles both, and the kernel carries the larger
// register count of the two: the old design's failure.

#include <algorithm>
#include <cstddef>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <sycl/sycl.hpp>

#include <batchlas/config.hh>
#include <batchlas/device/core.hh>
#include <batchlas/device/workspace.hh>
#include <batchlas/error.hh>

namespace batchlas::device {

/// @brief The tiers a LAPACK-level routine runs in: tier::Reg, tier::Cta, tier::Blocked, and the
/// tier::upto request.
namespace tier {
/// @brief The register tier: the matrix in the registers of a Partition or SubGroup executor.
/// @ingroup api_device_launch
///
/// Admits n <= N <= 32 (a routine may cap N lower for a wide scalar).
/// @tparam N the largest order the tier admits, 1 to 32
/// @tparam Exact true admits only n == N where the routine honours it (getrf); false admits any
///         n <= N
template <int N, bool Exact = false> struct Reg {
    static_assert(N >= 1 && N <= 32, "tier::Reg: 1 <= N <= 32");
    static constexpr int n_max = N;  ///< The largest order the tier admits.
    static constexpr bool exact = Exact;  ///< The Exact template argument.
};
/// @brief The local-memory tier: the matrix in the local memory of one work-group.
/// @ingroup api_device_launch
///
/// The work-group is a WorkGroup executor's own, or the one a partition executor shares
/// (tier_info::executor says which). Admits n <= N.
/// @tparam N the largest order the tier admits
/// @tparam Alg the routine's algorithm for this tier; alg::Default is the routine's own default
template <int N, class Alg = alg::Default> struct Cta {
    static_assert(N >= 1, "tier::Cta: N >= 1");
    static constexpr int n_max = N;  ///< The largest order the tier admits.
    using alg_type = Alg;  ///< The algorithm tag.
};
/// @brief The blocked tier: any n, with the matrix in global memory.
/// @ingroup api_device_launch
///
/// A diagonal block or panel of order NB is staged in local memory and the caller's global
/// workspace holds the rest.
/// @tparam NB the block (panel) order
/// @tparam Alg the routine's algorithm for this tier: alg::Default, or e.g. alg::LeftLooking for
///         potrf
template <int NB, class Alg = alg::Default> struct Blocked {
    static_assert(NB >= 1, "tier::Blocked: NB >= 1");
    static constexpr int nb = NB;  ///< The block order.
    static constexpr int n_max = -1;  ///< -1: the tier admits any n.
    using alg_type = Alg;  ///< The algorithm tag.
};
/// @brief A compile-time request for the first tier of the routine's list that admits NMax.
/// @ingroup api_device_launch
///
/// resolve_tier_t turns it into that tier. Naming an NMax that no tier of the routine admits is a
/// compile error.
/// @tparam NMax the order the chosen tier must admit
template <int NMax> struct upto {
    static constexpr int n_max = NMax;  ///< The order the chosen tier must admit.
};
/// @brief A list of tiers, smallest first, as tiers_t gives it and with_tier visits it.
/// @ingroup api_device_launch
/// @tparam Tiers the tiers
template <class... Tiers> struct list {};

/// @brief True when T is a tier::upto.
/// @ingroup api_device_launch
/// @tparam T the type to test
template <class T> inline constexpr bool is_upto_v = false;
/// @cond
template <int N> inline constexpr bool is_upto_v<upto<N>> = true;
/// @endcond
/// @brief True when T is a tier::Reg.
/// @ingroup api_device_launch
/// @tparam T the type to test
template <class T> inline constexpr bool is_reg_v = false;
/// @cond
template <int N, bool E> inline constexpr bool is_reg_v<Reg<N, E>> = true;
/// @endcond
/// @brief True when T is a tier::Cta.
/// @ingroup api_device_launch
/// @tparam T the type to test
template <class T> inline constexpr bool is_cta_v = false;
/// @cond
template <int N, class A> inline constexpr bool is_cta_v<Cta<N, A>> = true;
/// @endcond
/// @brief True when T is a tier::Blocked.
/// @ingroup api_device_launch
/// @tparam T the type to test
template <class T> inline constexpr bool is_blocked_v = false;
/// @cond
template <int NB, class A> inline constexpr bool is_blocked_v<Blocked<NB, A>> = true;
/// @endcond
}  // namespace tier

/// @brief The routine tags, `device::routine::<op><modes...>`, each declared next to its op.
namespace routine {}

/// @brief What tier Tier of Routine is for scalar T; specialised by each routine for each tier it
/// implements.
/// @ingroup api_device_launch
///
/// A specialisation states `executor` (the executor type), `wg_size`, `sg_size`, `per_group`
/// (matrices per work-group, one per executor), `admits(n)` and `buffer_size(extents...)` (a
/// WorkspaceSize). The primary template is incomplete: naming a tier a routine does not have is a
/// compile error.
/// @tparam Routine a routine tag from device::routine
/// @tparam Tier a concrete tier: tier::Reg, tier::Cta or tier::Blocked
/// @tparam T the scalar type
template <class Routine, class Tier, class T> struct tier_info;

/// @brief The tiers of Routine for Scalar, smallest first: `tiers<Routine, Scalar>::type` is a
/// tier::list.
/// @ingroup api_device_launch
///
/// Specialised by each routine next to its tier_info specialisations.
/// @tparam Routine a routine tag from device::routine
/// @tparam Scalar the scalar type
template <class Routine, class Scalar> struct tiers;
/// @brief The routine's tiers for Scalar, smallest first: `tiers<Routine, Scalar>::type`.
/// @ingroup api_device_launch
/// @tparam Routine a routine tag from device::routine
/// @tparam Scalar the scalar type
template <class Routine, class Scalar> using tiers_t = typename tiers<Routine, Scalar>::type;

/// @brief Satisfied when Routine implements Tier for Scalar: tier_info has a specialisation with
/// a `wg_size`.
/// @ingroup api_device_launch
/// @tparam Routine a routine tag from device::routine
/// @tparam Tier a concrete tier
/// @tparam Scalar the scalar type
template <class Routine, class Tier, class Scalar>
concept HasTier = requires { tier_info<Routine, Tier, Scalar>::wg_size; };

namespace detail {
template <class Tier> constexpr bool bound_admits(int n) { return Tier::n_max < 0 || n <= Tier::n_max; }

template <int NMax, class List> struct resolve_upto;
template <int NMax, class First, class... Rest> struct resolve_upto<NMax, tier::list<First, Rest...>> {
    using type = std::conditional_t<bound_admits<First>(NMax), First, typename resolve_upto<NMax, tier::list<Rest...>>::type>;
};
template <int NMax> struct resolve_upto<NMax, tier::list<>> {
    using type = void;
};
}  // namespace detail

/// @brief Resolves `tier::upto<NMax>` to the first tier of the routine's list that admits NMax; a
/// concrete tier passes through.
/// @ingroup api_device_launch
/// @tparam Routine a routine tag from device::routine
/// @tparam T the scalar type
/// @tparam Tier a concrete tier or a tier::upto
template <class Routine, class T, class Tier> struct resolve_tier {
    using type = Tier;  ///< The tier, unchanged.
};
/// @brief resolve_tier of a `tier::upto<NMax>`: the first tier of the routine's list that admits
/// NMax.
/// @ingroup api_device_launch
/// @tparam Routine a routine tag from device::routine
/// @tparam T the scalar type
/// @tparam NMax the order the tier must admit
template <class Routine, class T, int NMax> struct resolve_tier<Routine, T, tier::upto<NMax>> {
    /// @brief The first tier of the routine's list that admits NMax.
    using type = typename detail::resolve_upto<NMax, tiers_t<Routine, T>>::type;
    static_assert(!std::is_void_v<type>, "tier::upto<NMax>: no tier of this routine admits NMax");
};
/// @brief The resolved tier, `resolve_tier<Routine, T, Tier>::type`.
/// @ingroup api_device_launch
/// @tparam Routine a routine tag from device::routine
/// @tparam T the scalar type
/// @tparam Tier a concrete tier or a tier::upto
template <class Routine, class T, class Tier> using resolve_tier_t = typename resolve_tier<Routine, T, Tier>::type;

#if BATCHLAS_SYCL_IMPL_ACPP && BATCHLAS_ACPP_SLM_OPTIN
extern "C" int batchlas_acpp_slm_optin_bytes(int cuda_ordinal);
#endif

/// @brief The bytes of local memory each work-item of a launch costs before the kernel's own: 8 on
/// AdaptiveCpp's Metal backend, 0 elsewhere.
/// @ingroup api_device_lowlevel
/// @param d the device
/// @return the per-item launch scratch, in bytes
// evidence: docs/design/sycl-implementations.md#sycl-impl-slm-budget
inline std::size_t local_mem_scratch_per_item([[maybe_unused]] const sycl::device& d) {
#if BATCHLAS_SYCL_IMPL_ACPP && BATCHLAS_HAS_SYCL_BACKEND
    return d.get_backend() == sycl::backend::metal ? std::size_t{8} : std::size_t{0};
#else
    return 0;
#endif
}
/// @brief The local memory the device reports, raised to the AdaptiveCpp CUDA opt-in maximum when
/// that build option is on.
/// @ingroup api_device_lowlevel
/// @param d the device
/// @return the local memory size, in bytes, before any per-item launch scratch
inline std::size_t local_mem_reported(const sycl::device& d) {
    std::size_t reported = d.get_info<sycl::info::device::local_mem_size>();
#if BATCHLAS_SYCL_IMPL_ACPP && BATCHLAS_ACPP_SLM_OPTIN
    if (d.get_backend() == sycl::backend::cuda) {
        const int optin = batchlas_acpp_slm_optin_bytes(sycl::get_native<sycl::backend::cuda>(d));
        if (optin > 0) reported = std::max(reported, static_cast<std::size_t>(optin));
    }
#endif
    return reported;
}
/// @brief The local memory a work-group of wg items may launch with: the reported size less the
/// per-item launch scratch of wg items.
/// @ingroup api_device_launch
///
/// Every local-memory capacity in the library is computed from this function.
/// @param d the device
/// @param wg the work-group size, in work-items
/// @return the usable local memory, in bytes (0 when the scratch exceeds the reported size)
inline std::size_t local_mem_bytes(const sycl::device& d, std::size_t wg) {
    const std::size_t reported = local_mem_reported(d), scratch = local_mem_scratch_per_item(d) * wg;
    return reported > scratch ? reported - scratch : 0;
}
/// @brief The local memory ANY work-group may request on d: the capacity at the device's maximum
/// work-group size.
/// @ingroup api_device_launch
///
/// A caller that knows its work-group size may ask local_mem_bytes(d, wg) for more.
/// @param d the device
/// @return the usable local memory, in bytes
inline std::size_t local_mem_bytes(const sycl::device& d) {
    if (local_mem_scratch_per_item(d) == 0) return local_mem_bytes(d, 0);
    return local_mem_bytes(d, d.get_info<sycl::info::device::max_work_group_size>());
}

/// @brief The device facts with_tier filters tiers on.
/// @ingroup api_device_launch
///
/// A tier of wg items may use local_mem_for(wg) bytes of local memory.
struct DeviceLimits {
    std::size_t local_mem_bytes = 0;  ///< The local memory the device reports (local_mem_reported).
    std::vector<std::size_t> sub_group_sizes;  ///< The sub-group sizes the device supports.
    std::size_t max_work_group_size = 0;  ///< The device's maximum work-group size.
    bool fp64 = false;  ///< Whether the device has the fp64 aspect.
    std::size_t local_scratch_per_item = 0;  ///< The per-item launch scratch (local_mem_scratch_per_item).
    /// @brief The local memory a work-group of wg items may use.
    /// @param wg the work-group size, in work-items
    /// @return local_mem_bytes less wg times local_scratch_per_item, or 0
    std::size_t local_mem_for(std::size_t wg) const {
        const std::size_t scratch = local_scratch_per_item * wg;
        return local_mem_bytes > scratch ? local_mem_bytes - scratch : 0;
    }
    /// @brief The limits of device d.
    /// @param d the device
    /// @return its DeviceLimits
    static DeviceLimits of(const sycl::device& d) {
        return {local_mem_reported(d), d.get_info<sycl::info::device::sub_group_sizes>(),
                d.get_info<sycl::info::device::max_work_group_size>(), d.has(sycl::aspect::fp64),
                local_mem_scratch_per_item(d)};
    }
};

namespace detail {
template <class Routine, class Tier, class Scalar> bool tier_fits(const DeviceLimits& lim, int n) {
    using I = tier_info<Routine, Tier, Scalar>;
    if (!bound_admits<Tier>(n) || !I::admits(n)) return false;
    if (std::is_same_v<typename Arith<Scalar>::real_t, double> && !lim.fp64) return false;
    if (static_cast<std::size_t>(I::wg_size) > lim.max_work_group_size) return false;
    bool sg_ok = false;
    for (std::size_t s : lim.sub_group_sizes) sg_ok = sg_ok || s == static_cast<std::size_t>(I::sg_size);
    if (!sg_ok) return false;
    // A launch holds per_group matrices' local slices (launch.hh's local_workspace).
    return I::buffer_size(n).local_bytes * static_cast<std::size_t>(I::per_group) <= lim.local_mem_for(I::wg_size);
}
template <class Routine, class Scalar, class F, class... Tiers> bool visit_first(const DeviceLimits& lim, int n, F& f, tier::list<Tiers...>) {
    bool done = false;
    ((!done && tier_fits<Routine, Tiers, Scalar>(lim, n) ? (f(Tiers{}), done = true) : false), ...);
    return done;
}
}  // namespace detail

/// @brief Host side: calls `f(Tier{})` for the first tier of List that admits n on the device.
/// @ingroup api_device_launch
///
/// A tier admits n when its bound and tier_info's `admits(n)` hold, its `per_group` local slices
/// fit the device's local memory less the per-item launch scratch, its `wg_size` fits the maximum
/// work-group size, its `sg_size` is one of the device's sub-group sizes, and, when the real type
/// of %Scalar (`Arith<Scalar>::real_t`) is double, so for double and `std::complex<double>`, the
/// device has fp64. Each listed tier is a separate instantiation of whatever kernel f launches;
/// exactly one runs.
/// @tparam Routine a routine tag from device::routine
/// @tparam Scalar the scalar type
/// @tparam List the tiers to try, smallest first (default: the routine's tiers_t)
/// @tparam F a callable taking a tier by value
/// @param lim the device limits
/// @param n the order of the problem
/// @param f the callable that launches the kernel for the chosen tier
/// @param name the routine name for the error message (default: `Routine::name`)
/// @throws batchlas::invalid_argument `"<name>: no device tier admits n=<n>"` when no tier does
template <class Routine, class Scalar, class List = tiers_t<Routine, Scalar>, class F>
void with_tier(const DeviceLimits& lim, int n, F&& f, const char* name = Routine::name) {
    if (!detail::visit_first<Routine, Scalar>(lim, n, f, List{}))
        throw batchlas::invalid_argument(std::string(name) + ": no device tier admits n=" + std::to_string(n));
}
/// @brief Host side: with_tier on the limits of device d (DeviceLimits::of(d)).
/// @ingroup api_device_launch
/// @tparam Routine a routine tag from device::routine
/// @tparam Scalar the scalar type
/// @tparam List the tiers to try, smallest first (default: the routine's tiers_t)
/// @tparam F a callable taking a tier by value
/// @param d the device
/// @param n the order of the problem
/// @param f the callable that launches the kernel for the chosen tier
/// @param name the routine name for the error message (default: `Routine::name`)
/// @throws batchlas::invalid_argument `"<name>: no device tier admits n=<n>"` when no tier does
template <class Routine, class Scalar, class List = tiers_t<Routine, Scalar>, class F>
void with_tier(const sycl::device& d, int n, F&& f, const char* name = Routine::name) {
    with_tier<Routine, Scalar, List>(DeviceLimits::of(d), n, std::forward<F>(f), name);
}

}  // namespace batchlas::device
