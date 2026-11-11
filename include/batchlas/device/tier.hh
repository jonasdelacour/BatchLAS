// SPDX-License-Identifier: MIT
#pragma once
// Device LA tiers (v22 design C.4): how a LAPACK-level op runs inside a kernel, chosen at COMPILE
// time so one kernel never pays the registers of two paths (a run-time `if (n <= 32)` between
// two tiers compiles both and the kernel carries the max REG: v1's failure).
//   tier::Reg<N, Exact>      the matrix in registers of a Partition / SubGroup; n <= N <= 32
//   tier::Cta<N, Alg>        the matrix in local memory of one WorkGroup; n <= N
//   tier::Blocked<NB, Alg>   any n: A in global, a diagonal block / panel in local memory, the
//                            caller's global workspace for the rest
//   tier::upto<NMax>         the first tier of the routine's list admitting NMax (compile time)
// Per routine (a tag type in device::routine) and scalar T, tier_info<Routine, Tier, T> states
// wg_size, sg_size, admits(n) and buffer_size(n); tiers_t<Routine, T> lists the routine's tiers
// smallest first. with_tier (host) visits the first admitting tier for a run-time n.

#include <cstddef>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <sycl/sycl.hpp>

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

// The device facts with_tier filters on.
struct DeviceLimits {
    std::size_t local_mem_bytes = 0;
    std::vector<std::size_t> sub_group_sizes;
    std::size_t max_work_group_size = 0;
    bool fp64 = false;
    static DeviceLimits of(const sycl::device& d) {
        return {d.get_info<sycl::info::device::local_mem_size>(), d.get_info<sycl::info::device::sub_group_sizes>(),
                d.get_info<sycl::info::device::max_work_group_size>(), d.has(sycl::aspect::fp64)};
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
    return I::buffer_size(n).local_bytes <= lim.local_mem_bytes;
}
template <class Routine, class T, class F, class... Tiers> bool visit_first(const DeviceLimits& lim, int n, F& f, tier::list<Tiers...>) {
    bool done = false;
    ((!done && tier_fits<Routine, Tiers, T>(lim, n) ? (f(Tiers{}), done = true) : false), ...);
    return done;
}
}  // namespace detail

// Host side: call f(Tier{}) for the first tier of List (default: the routine's list) that admits
// n, fits the device's local memory and work-group size, runs at a sub-group size the device
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
