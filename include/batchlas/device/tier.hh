// SPDX-License-Identifier: MIT
#pragma once
// Device LA tiers: how a LAPACK-level op runs inside a kernel, chosen at COMPILE
// time so one kernel never pays the registers of two paths (a run-time `if (n <= 32)` between
// two tiers compiles both and the kernel carries the max REG: the old design's failure).
//   tier::Reg<N, Exact>      the matrix in registers of a Partition / SubGroup; n <= N <= 32
//   tier::Cta<N, Alg>        the matrix in local memory (of a WorkGroup, or of the work-group a
//                            partition executor shares: tier_info::executor); n <= N
//   tier::Blocked<NB, Alg>   any n: A in global, a diagonal block / panel in local memory, the
//                            caller's global workspace for the rest
//   tier::upto<NMax>         the first tier of the routine's list admitting NMax (compile time)
// Per routine (a tag type in device::routine) and scalar T, tier_info<Routine, Tier, T> states
// the executor type, wg_size, sg_size, per_group (matrices per work-group: one per executor),
// admits(n) and buffer_size(n); tiers_t<Routine, T> lists the routine's tiers smallest first.
// with_tier (host) visits the first admitting tier for a run-time n; launch.hh turns a tier into
// a launch (nd_range, executor, matrix slot, workspace).

#include <algorithm>
#include <cstddef>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <sycl/sycl.hpp>

#include <batchlas/backend_config.h>
#include <batchlas/device/core.hh>
#include <batchlas/device/workspace.hh>
#include <batchlas/error.hh>

namespace batchlas::device {

namespace tier {
template <int N, bool Exact = false> struct Reg {
    static_assert(N >= 1 && N <= 32, "tier::Reg: 1 <= N <= 32");
    static constexpr int n_max = N;
    static constexpr bool exact = Exact;
};
template <int N, class Alg = alg::Default> struct Cta {
    static_assert(N >= 1, "tier::Cta: N >= 1");
    static constexpr int n_max = N;
    using alg_type = Alg;
};
template <int NB, class Alg = alg::Default> struct Blocked {
    static_assert(NB >= 1, "tier::Blocked: NB >= 1");
    static constexpr int nb = NB;
    static constexpr int n_max = -1;  // any n
    using alg_type = Alg;
};
template <int NMax> struct upto {
    static constexpr int n_max = NMax;
};
template <class... Tiers> struct list {};

template <class T> inline constexpr bool is_upto_v = false;
template <int N> inline constexpr bool is_upto_v<upto<N>> = true;
template <class T> inline constexpr bool is_reg_v = false;
template <int N, bool E> inline constexpr bool is_reg_v<Reg<N, E>> = true;
template <class T> inline constexpr bool is_cta_v = false;
template <int N, class A> inline constexpr bool is_cta_v<Cta<N, A>> = true;
template <class T> inline constexpr bool is_blocked_v = false;
template <int NB, class A> inline constexpr bool is_blocked_v<Blocked<NB, A>> = true;
}  // namespace tier

namespace routine {}  // routine tags: device::routine::<op><modes...>, declared next to each op

// Per (Routine, Tier, T): specialised by each routine for each tier it implements. The primary
// template is incomplete: naming a tier a routine does not have is a compile error.
template <class Routine, class Tier, class T> struct tier_info;

// tiers<Routine, T>::type = tier::list<...>: the routine's tiers for T, smallest first.
template <class Routine, class T> struct tiers;
template <class Routine, class T> using tiers_t = typename tiers<Routine, T>::type;

template <class Routine, class Tier, class T>
concept HasTier = requires { tier_info<Routine, Tier, T>::wg_size; };

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

// tier::upto<NMax> -> the first admitting tier (a concrete tier passes through).
template <class Routine, class T, class Tier> struct resolve_tier {
    using type = Tier;
};
template <class Routine, class T, int NMax> struct resolve_tier<Routine, T, tier::upto<NMax>> {
    using type = typename detail::resolve_upto<NMax, tiers_t<Routine, T>>::type;
    static_assert(!std::is_void_v<type>, "tier::upto<NMax>: no tier of this routine admits NMax");
};
template <class Routine, class T, class Tier> using resolve_tier_t = typename resolve_tier<Routine, T, Tier>::type;

#if BATCHLAS_SYCL_IMPL_ACPP && BATCHLAS_ACPP_SLM_OPTIN
extern "C" int batchlas_acpp_slm_optin_bytes(int cuda_ordinal);
#endif

// The local memory a work-group of wg items may launch with (every SLM capacity's source): the
// reported size, the acpp CUDA opt-in maximum, less acpp Metal's 8 B per item of launch scratch.
// evidence: docs/design/sycl-implementations.md#sycl-impl-slm-budget
inline std::size_t local_mem_scratch_per_item([[maybe_unused]] const sycl::device& d) {
#if BATCHLAS_SYCL_IMPL_ACPP && BATCHLAS_HAS_SYCL_BACKEND
    return d.get_backend() == sycl::backend::metal ? std::size_t{8} : std::size_t{0};
#else
    return 0;
#endif
}
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
inline std::size_t local_mem_bytes(const sycl::device& d, std::size_t wg) {
    const std::size_t reported = local_mem_reported(d), scratch = local_mem_scratch_per_item(d) * wg;
    return reported > scratch ? reported - scratch : 0;
}
// What ANY work-group may request; a caller that knows its width may ask (d, wg) for more.
inline std::size_t local_mem_bytes(const sycl::device& d) {
    if (local_mem_scratch_per_item(d) == 0) return local_mem_bytes(d, 0);
    return local_mem_bytes(d, d.get_info<sycl::info::device::max_work_group_size>());
}

// The device facts with_tier filters on; a tier of wg items may use local_mem_for(wg).
struct DeviceLimits {
    std::size_t local_mem_bytes = 0;
    std::vector<std::size_t> sub_group_sizes;
    std::size_t max_work_group_size = 0;
    bool fp64 = false;
    std::size_t local_scratch_per_item = 0;
    std::size_t local_mem_for(std::size_t wg) const {
        const std::size_t scratch = local_scratch_per_item * wg;
        return local_mem_bytes > scratch ? local_mem_bytes - scratch : 0;
    }
    static DeviceLimits of(const sycl::device& d) {
        return {local_mem_reported(d), d.get_info<sycl::info::device::sub_group_sizes>(),
                d.get_info<sycl::info::device::max_work_group_size>(), d.has(sycl::aspect::fp64),
                local_mem_scratch_per_item(d)};
    }
};

namespace detail {
template <class Routine, class Tier, class T> bool tier_fits(const DeviceLimits& lim, int n) {
    using I = tier_info<Routine, Tier, T>;
    if (!bound_admits<Tier>(n) || !I::admits(n)) return false;
    if (std::is_same_v<typename Arith<T>::real_t, double> && !lim.fp64) return false;
    if (static_cast<std::size_t>(I::wg_size) > lim.max_work_group_size) return false;
    bool sg_ok = false;
    for (std::size_t s : lim.sub_group_sizes) sg_ok = sg_ok || s == static_cast<std::size_t>(I::sg_size);
    if (!sg_ok) return false;
    // A launch holds per_group matrices' local slices (launch.hh's local_workspace).
    return I::buffer_size(n).local_bytes * static_cast<std::size_t>(I::per_group) <= lim.local_mem_for(I::wg_size);
}
template <class Routine, class T, class F, class... Tiers> bool visit_first(const DeviceLimits& lim, int n, F& f, tier::list<Tiers...>) {
    bool done = false;
    ((!done && tier_fits<Routine, Tiers, T>(lim, n) ? (f(Tiers{}), done = true) : false), ...);
    return done;
}
}  // namespace detail

// Host side: call f(Tier{}) for the first tier of List (default: the routine's list) that admits
// n, fits the device's local memory (per_group slices, less the per-item launch scratch) and
// work-group size, runs at a sub-group size the device
// has, and (for a double T) a device with fp64. Each listed tier is a separate instantiation of
// whatever kernel f launches; exactly one runs. Throws batchlas::invalid_argument("<name>: no
// device tier admits n=<n>") when none does.
template <class Routine, class T, class List = tiers_t<Routine, T>, class F>
void with_tier(const DeviceLimits& lim, int n, F&& f, const char* name = Routine::name) {
    if (!detail::visit_first<Routine, T>(lim, n, f, List{}))
        throw batchlas::invalid_argument(std::string(name) + ": no device tier admits n=" + std::to_string(n));
}
template <class Routine, class T, class List = tiers_t<Routine, T>, class F>
void with_tier(const sycl::device& d, int n, F&& f, const char* name = Routine::name) {
    with_tier<Routine, T, List>(DeviceLimits::of(d), n, std::forward<F>(f), name);
}

}  // namespace batchlas::device
