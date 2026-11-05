// SPDX-License-Identifier: MIT
#pragma once
// LAPACK ?sysv on device (v22 design C): A = P L D L^T P^T by sytrf, then B := A^-1 B by sytrs
// when the factor is nonsingular (info == 0; otherwise B is left as given, as LAPACK's).
// Tiers (one per kernel; with_tier picks one on the host for a run-time n):
//   tier::Cta<N>, N <= 32   fused in one local-memory tile: sytrf `cta`'s factor on sub-group 0,
//                           then sytrs `cta`'s converted solve, one sub-group per right-hand side
//                           (the host route sysv `cta` launches these two bodies)
//   tier::Blocked<1>        any n: sytrf Blocked<1>, then sytrs Blocked<1> (sysv `compose` over
//                           sytrf `wg` and sytrs `wg`); ex at most 256 items
// Tests: tests/device_api_sytrf_tests.cc.

#include <cstddef>
#include <type_traits>

#include <batchlas/device.hh>
#include <batchlas/device/lapack/sytrf.hh>
#include <batchlas/device/lapack/sytrs.hh>

namespace batchlas::device::inline v2 {

namespace routine {
template <Uplo UL> struct sysv {
    static_assert(UL == Uplo::Lower || UL == Uplo::Upper, "sysv: Uplo::Lower or Uplo::Upper");
    static constexpr const char* name = "sysv";
};
}  // namespace routine

namespace sysv_detail {

// The fused tile of sysv `cta`: the (N + 1) x N factor then e (N); the int part holds ipiv's
// copy [0, N), ps, perm, kind (N each), the validity flag at 4N and the factor's info at 4N + 1.
template <int N, class T> struct SysvCtaPlan {
    ws::Slot<T> tile;
    ws::Slot<int> idx;
    WorkspaceSize size;
};
template <int N, class T> constexpr SysvCtaPlan<N, T> sysv_cta_plan() {
    Plan p;
    const auto tile = p.take<T>(ws::Space::Local, static_cast<std::size_t>(N + 1) * N + N);
    const auto idx = p.take<int>(ws::Space::Local, 4 * static_cast<std::size_t>(N) + 2);
    return {tile, idx, p.size()};
}

template <Uplo UL, int N, class Ex, class MA, class MB>
BATCHLAS_DLA_INLINE int sysv_run(tier::Cta<N>, const Ex& ex, const MA& A, const MB& B, int* ipiv, const Workspace& ws) {
    using D = std::remove_cv_t<typename MA::value_type>;
    using R = real_t<D>;
    constexpr bool rev = UL == Uplo::Upper;
    constexpr int ldt = N + 1;
    constexpr auto pl = sysv_cta_plan<N, D>();
    const int n = A.rows();
    D* const tp = carve_local(ws, pl.tile).get_raw();
    int* const pp = carve_local(ws, pl.idx).get_raw();
    const SubGroup<32> g{ex.sub()};
    sytrf_detail::sytrf_cta_factor(
        ex, g, sytrf_detail::sytrf_team(ex) == 0, sytrf_detail::sytrf_view(A, rev), sytrf_detail::rev_mat(tp, ldt, n, false), n, rev,
        sytrf_detail::ArgMaxScratch<R>{nullptr, nullptr},
        [&](int k, int v) {
            ipiv[k] = v;
            pp[k] = v;
        },
        [&](int st) { pp[4 * N + 1] = st; });
    sync<scope::Group>(ex, Mem::Global);
    const int info = pp[4 * N + 1];
    if (info == 0) {
        const sytrs_detail::Staged<D> L{tp, ldt, n, tp + ldt * N, pp + N, pp + 2 * N, pp + 3 * N};
        sytrs_detail::sytrs_cta_solve<1>(ex, ex.sub(), ex.lane(), sytrf_detail::sytrf_team(ex), sytrf_detail::sytrf_nteams(ex), L, pp,
                                         rev, pp + 4 * N, &B(0, 0), sytrf_detail::sytrf_ld(B), B.cols());
    }
    sync<scope::Group>(ex, Mem::Global);
    return info;
}

template <Uplo UL, class Ex, class MA, class MB>
BATCHLAS_DLA_INLINE int sysv_run(tier::Blocked<1>, const Ex& ex, const MA& A, const MB& B, int* ipiv, const Workspace& ws) {
    // Both halves carve the same workspace from offset 0: sytrf's ends on a barrier.
    const int info = sytrf_detail::sytrf_run<UL>(tier::Blocked<1>{}, ex, A, ipiv, ws);
    if (info != 0) return info;
    sytrs_detail::sytrs_run<UL>(tier::Blocked<1>{}, ex, A, B, ipiv, ws);
    return 0;
}

constexpr WorkspaceSize sysv_max(const WorkspaceSize& a, const WorkspaceSize& b) {
    return {a.local_bytes > b.local_bytes ? a.local_bytes : b.local_bytes,
            a.global_bytes > b.global_bytes ? a.global_bytes : b.global_bytes, a.align > b.align ? a.align : b.align};
}

}  // namespace sysv_detail

template <Uplo UL, class T> struct tiers<routine::sysv<UL>, T> {
    using type = tier::list<tier::Cta<32>, tier::Blocked<1>>;
};
template <Uplo UL, int N, class T> struct tier_info<routine::sysv<UL>, tier::Cta<N>, T> {
    static_assert(N <= 32, "sysv tier::Cta<N>: N <= 32 (one sub-group factors)");
    static constexpr int wg_size = 32, sg_size = 32;
    static constexpr bool admits(int n) { return n >= 0 && n <= N; }
    static constexpr WorkspaceSize buffer_size(int) { return sysv_detail::sysv_cta_plan<N, T>().size; }
};
template <Uplo UL, class T> struct tier_info<routine::sysv<UL>, tier::Blocked<1>, T> {
    static constexpr int wg_size = sytrf_detail::sytrf_wg_max, sg_size = 32;
    static constexpr bool admits(int n) { return n >= 0; }
    static constexpr WorkspaceSize buffer_size(int) {
        return sysv_detail::sysv_max(sytrf_detail::sytrf_wg_plan<real_t<T>>().size, sytrs_detail::sytrs_wg_plan<real_t<T>>().size);
    }
};

template <Uplo UL, class Tier, class T> constexpr WorkspaceSize sysv_buffer_size(int n) {
    using Rt = routine::sysv<UL>;
    return tier_info<Rt, resolve_tier_t<Rt, T, Tier>, T>::buffer_size(n);
}

// ?sysv of one matrix inside a kernel: A := sytrf's factor, ipiv[0, n) := its pivots, and B :=
// A^-1 B when info == 0. Collective as sytrf; A, ipiv and B visible to every item on return.
// Returns sytrf's info (k > 0: B untouched) or a negative LAPACK argument number (sytrs_check's,
// ?sysv numbers them as ?sytrs), the same on every item.
template <Uplo UL, class Tier, class Ex, class MA, class MB>
inline int sysv(const Ex& ex, const MA& A, const MB& B, int* ipiv, const Workspace& ws) {
    using D = std::remove_cv_t<typename MA::value_type>;
    static_assert(std::is_same_v<D, std::remove_cv_t<typename MB::value_type>>, "sysv: A and B of one scalar type");
    using Tr = resolve_tier_t<routine::sysv<UL>, D, Tier>;
    static_assert(sytrf_detail::sytrf_exec_ok<Ex>(), "sysv: an executor of whole 32-wide sub-groups");
    const int chk = sytrs_detail::sytrs_check<Tr>(ex, A, B);
    if (chk != 0 || A.rows() == 0) return chk;
    return sysv_detail::sysv_run<UL>(Tr{}, ex, A, B, ipiv, ws);
}

}  // namespace batchlas::device
