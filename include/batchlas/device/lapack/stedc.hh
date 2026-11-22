// SPDX-License-Identifier: MIT
#pragma once
/// @file
/// @brief The divide-and-conquer tridiagonal eigensolver `stedc` (xSTEDC) inside a kernel: the
/// eigenvalues, ascending, and optionally the eigenvectors of a real symmetric tridiagonal (d, e).
///
/// One matrix per work-group, any n, in one tier, `tier::Blocked<1>`, on a `WorkGroup<256, 32>`:
/// every off-diagonal is a tear, so the leaves are 1 x 1 and the merges run bottom-up, widths 1,
/// 2, 4 and so on. Each merge deflates (`laed2`), solves the secular equation one root per
/// sub-group (`laed3`, `laed4`) and back-transforms with the in-group gemm. The tier needs
/// 32-wide sub-groups. Real types only. The merge kernels are in @ref api_device_lowlevel.
/// @ingroup api_device_lapack_eig
// Design C.7, syev Blocked's "stedc (risky)". detail::dc_layout is the workspace plan: the gemm's
// staging and one slot in local memory; per-matrix vectors and two n x n blocks in global memory.

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <sycl/sycl.hpp>

#include <batchlas/blas/enums.hh>
#include <batchlas/device.hh>
#include <batchlas/device/lapack/laed2.hh>
#include <batchlas/device/lapack/laed3.hh>
#include <batchlas/device/lapack/laed4.hh>
#include <batchlas/device/level3/group.hh>
#include <batchlas/device/sg_partition.hh>
#include <batchlas/device/tier.hh>
#include <batchlas/device/workspace.hh>

namespace batchlas::device {

/// @brief Run-time options of `stedc`.
/// @tparam Real  the tridiagonal's real type
/// @ingroup api_device_lapack_eig
template <class Real> struct StedcOptions {
    /// The secular iteration cap per root, >= 1 (the host StedcParams::max_sec_iter); a root that
    /// reaches it makes `info` 1.
    std::int32_t max_sec_iter = 50;
};

namespace detail {

// The executor the tier is written for (the gemm's 16 x 16 items) and one root per sub-group.
inline constexpr int wg_size = 256, sg_size = 32, part_size = 32;
template <class Real> using gemm_cfg = l3::cfg::RegTiled<Real, 64, 64, 16, 4, 4>;

// The workspace plan (design C.5). Local: the back-transform's staging and the merge's kept
// count. Global, per matrix: the poles and weights before (A) and after (B) the sort, the roots,
// |.| scratch, the column maps and deflation scans (n each), the secular vectors Q' and the
// gathered columns (n x n each), and Z itself when no vectors are wanted.
template <class Real> struct StedcLayout {
    WsSlot<Real> smem;
    WsSlot<std::int32_t> reduced;
    WsSlot<Real> d_a, z_a, d_b, z_b, lam, absx;
    WsSlot<std::int32_t> map_a, map_b, keep, perm, plocal, scan_inc, scan_exc, rank;
    WsSlot<Real> qp, tmp, z;
    WorkspaceSize size;
};
template <class Real> constexpr StedcLayout<Real> dc_layout(int n, bool own_z) {
    Plan p;
    const std::size_t un = std::size_t(n), nn = un * un;
    const auto smem = p.take<Real>(WsSpace::Local, std::size_t(l3::smem_elems<gemm_cfg<Real>>()));
    const auto reduced = p.take<std::int32_t>(WsSpace::Local, 1);
    const auto d_a = p.take<Real>(WsSpace::Global, un);
    const auto z_a = p.take<Real>(WsSpace::Global, un);
    const auto d_b = p.take<Real>(WsSpace::Global, un);
    const auto z_b = p.take<Real>(WsSpace::Global, un);
    const auto lam = p.take<Real>(WsSpace::Global, un);
    const auto absx = p.take<Real>(WsSpace::Global, un);
    const auto map_a = p.take<std::int32_t>(WsSpace::Global, un);
    const auto map_b = p.take<std::int32_t>(WsSpace::Global, un);
    const auto keep = p.take<std::int32_t>(WsSpace::Global, un);
    const auto perm = p.take<std::int32_t>(WsSpace::Global, un);
    const auto plocal = p.take<std::int32_t>(WsSpace::Global, un);
    const auto scan_inc = p.take<std::int32_t>(WsSpace::Global, un);
    const auto scan_exc = p.take<std::int32_t>(WsSpace::Global, un);
    const auto rank = p.take<std::int32_t>(WsSpace::Global, un);
    const auto qp = p.take<Real>(WsSpace::Global, nn);
    const auto tmp = p.take<Real>(WsSpace::Global, nn);
    const auto z = p.take<Real>(WsSpace::Global, own_z ? nn : 0);
    return {smem, reduced, d_a, z_a, d_b, z_b, lam, absx, map_a, map_b, keep, perm, plocal, scan_inc, scan_exc, rank, qp, tmp, z, p.size()};
}

// rank(i) among x(0..s) by (value, index): a stable ascending sort as a gather-free scatter.
template <class Real> BATCHLAS_DEVICE_INLINE int sort_rank(const Real* x, int s, int i) {
    const Real xi = x[i];
    int r = 0;
    for (int j = 0; j < s; ++j) {
        const Real xj = x[j];
        r += (xj < xi || (xj == xi && j < i)) ? 1 : 0;
    }
    return r;
}

// The workspace's pointers for one matrix.
template <class Real> struct Ptrs {
    Real *d_a, *z_a, *d_b, *z_b, *lam, *absx, *qp, *tmp;
    std::int32_t *map_a, *map_b, *keep, *perm, *plocal, *scan_inc, *scan_exc, *rank;
    local_mptr<std::int32_t> reduced;
    Workspace smem;
};
template <class Real> BATCHLAS_DEVICE_INLINE Ptrs<Real> ptrs(const Workspace& ws, const StedcLayout<Real>& L) {
    return {carve_global(ws, L.d_a),     carve_global(ws, L.z_a),      carve_global(ws, L.d_b),      carve_global(ws, L.z_b),
            carve_global(ws, L.lam),     carve_global(ws, L.absx),     carve_global(ws, L.qp),       carve_global(ws, L.tmp),
            carve_global(ws, L.map_a),   carve_global(ws, L.map_b),    carve_global(ws, L.keep),     carve_global(ws, L.perm),
            carve_global(ws, L.plocal),  carve_global(ws, L.scan_inc), carve_global(ws, L.scan_exc), carve_global(ws, L.rank),
            carve_local(ws, L.reduced),  sub_workspace(ws, L.smem)};
}

// One merge: rows / columns [o, o + s) of Z hold diag(Q1, Q2) (Q1 m x m), w(o..o + s) the two
// halves' ascending eigenvalues, rho the tear e(o + m - 1). On return they hold the merged
// problem's, ascending. Returns this item's count of unconverged roots.
template <class Ex, class PEx, class W, class Real>
BATCHLAS_DEVICE_HELPER int merge(const Ex& ex, const PEx& pex, int part_id, int parts, int n, int o, int s, int m, Real rho, const W& w,
                              const l3::Global<Real>& Z, const Ptrs<Real>& P, std::int32_t max_iter) {
    const int ld = Z.ld;
    Real* const q0 = Z.p + o + static_cast<std::ptrdiff_t>(o) * ld;
    const auto Qn = mat(q0, s, s, ld);
    const Real sign = rho >= Real(0) ? Real(1) : Real(-1);

    for_each_owned(ex, s, [&](int i) { P.d_a[i] = w(o + i); });
    laed2_merge_vector(ex, vec(P.z_a, s), m, Qn.sub(0, 0, m, m).row(m - 1), Qn.sub(m, m, s - m, s - m).row(0), sign);
    sync(ex, Mem::Global);

    // The poles ascending (the halves are each sorted; the rank sort does not rely on it).
    for_each_owned(ex, s, [&](int i) {
        const int r = sort_rank(P.d_a, s, i);
        P.d_b[r] = P.d_a[i];
        P.z_b[r] = P.z_a[i];
        P.map_b[r] = i;
    });
    sync(ex, Mem::Global);

    const auto red = P.reduced;
    laed2(ex, std::int64_t(s), vec(P.d_b, s), vec(P.z_b, s), Qn, vec(P.absx, s), vec(P.d_b, s), [&]() { return rho; }, vec(P.keep, s),
          [&]() { return P.keep; }, vec(P.perm, s), vec(P.map_b, s), P.plocal, P.scan_inc, [&]() { return P.scan_inc; }, P.scan_exc,
          [&]() { return P.scan_exc; }, [&](std::int32_t c) { red[0] = c; });
    sync(ex, Mem::Global);
    const int dd = red[0];

    // Kept poles first: the secular problem is d_a(0..dd), z_a(0..dd).
    for_each_owned(ex, s, [&](int i) {
        const int src = P.perm[i];
        P.d_a[i] = P.d_b[src];
        P.z_a[i] = P.z_b[src];
        P.map_a[i] = P.map_b[src];
    });
    sync(ex, Mem::Global);

    int bad = 0;
    if (dd > 0) {
        const auto Qp = mat(P.qp, dd, dd, n);
        auto Qp_rows = Qp;
        const auto dv = vec(P.d_a, dd);
        const auto zv = vec(P.z_a, dd);
        const VectorView<Real> v(P.z_a, dd);
        laed3<Real>(ex, pex, part_id, parts, dd, dv, zv, sycl::fabs(Real(2) * rho), max_iter, true, Qp, Qp_rows, v, 0,
                    [&](std::int32_t k, Real origin, Real tau, bool converged) {
                        P.lam[k] = origin + tau;
                        bad += converged ? 0 : 1;
                    });
    }

    // Back-transform: Z(:, k) = Q[:, map_a(k)] for the deflated k >= dd, Q[:, map_a(0..dd)] Q'
    // for the rest, then every column to its eigenvalue's rank.
    for_each_owned(ex, s * s, [&](int t) {
        const int i = t % s, k = t / s;
        P.tmp[i + static_cast<std::ptrdiff_t>(k) * n] = Qn(i, P.map_a[k]);
    });
    for_each_owned(ex, s, [&](int k) { P.d_b[k] = k < dd ? P.lam[k] : P.d_a[k]; });
    sync(ex, Mem::Global);
    if (dd > 0) {
        gemm<Transpose::NoTrans, Transpose::NoTrans, gemm_cfg<Real>>(ex, One{}, l3::Global<Real>{P.tmp, n, s, dd},
                                                                     l3::Global<Real>{P.qp, n, dd, dd}, Zero{}, l3::Global<Real>{q0, ld, s, dd}, P.smem);
    }
    for_each_owned(ex, (s - dd) * s, [&](int t) {
        const int i = t % s, k = dd + t / s;
        Qn(i, k) = P.tmp[i + static_cast<std::ptrdiff_t>(k) * n];
    });
    for_each_owned(ex, s, [&](int k) {
        const int r = sort_rank(P.d_b, s, k);
        P.rank[k] = r;
        w(o + r) = P.d_b[k];
    });
    sync(ex, Mem::Global);
    for_each_owned(ex, s * s, [&](int t) {
        const int i = t % s, k = t / s;
        P.tmp[i + static_cast<std::ptrdiff_t>(P.rank[k]) * n] = Qn(i, k);
    });
    sync(ex, Mem::Global);
    for_each_owned(ex, s * s, [&](int t) {
        const int i = t % s, k = t / s;
        Qn(i, k) = P.tmp[i + static_cast<std::ptrdiff_t>(k) * n];
    });
    sync(ex, Mem::Global);
    return bad;
}

// The op on Z (n x n, column-major, ld): leaves, then the merges bottom-up.
template <class Ex, class D, class E, class W, class Real>
BATCHLAS_DEVICE_HELPER int bottom_up(const Ex& ex, const D& d, const E& e, const W& w, const l3::Global<Real>& Z, const Ptrs<Real>& P,
                            std::int32_t max_iter) {
    const int n = d.size();
    // Every off-diagonal tears: w(i) = d(i) - |e(i-1)| - |e(i)|, the 1 x 1 leaves; Z = I.
    for_each_owned(ex, n, [&](int i) {
        Real di = d(i);
        if (i > 0) di -= sycl::fabs(e(i - 1));
        if (i + 1 < n) di -= sycl::fabs(e(i));
        w(i) = di;
    });
    for (int j = 0; j < n; ++j) for_each_owned(ex, n, [&](int i) { Z(i, j) = i == j ? Real(1) : Real(0); });
    sync(ex, Mem::Global);

    const SgpPartition<part_size> pex{ex.it.get_sub_group()};
    const int sg_id = static_cast<int>(ex.it.get_sub_group().get_group_linear_id());
    const int parts_per_sg = static_cast<int>(pex.part.get_group_linear_range());
    const int part_id = sg_id * parts_per_sg + static_cast<int>(pex.part.get_group_linear_id());
    const int parts = Ex::size / part_size;

    int bad = 0;
    for (int width = 1; width < n; width *= 2) {
        for (int o = 0; o + width < n; o += 2 * width) {
            const int s = sycl::min(2 * width, n - o);
            bad += merge(ex, pex, part_id, parts, n, o, s, width, Real(e(o + width - 1)), w, Z, P, max_iter);
        }
    }
    return allreduce(ex, bad, sycl::maximum<int>()) > 0 ? 1 : 0;
}

template <class Ex, class D, class Real> constexpr void assert_types() {
    static_assert(std::is_floating_point_v<Real>, "stedc: a real T");
    static_assert(std::is_same_v<elem_t<D>, Real>, "stedc: params of the matrix's real type");
    static_assert(is_work_group_v<Ex> && Ex::size == wg_size && Ex::sg_size == sg_size, "stedc: executor WorkGroup<256, 32>");
}

}  // namespace detail

namespace routine {
/// @brief The routine tag of `stedc` (xSTEDC) for `with_tier`, `launch` and `buffer_size`.
/// @tparam J  `JobType::EigenVectors` (Z is an argument) or `JobType::NoEigenVectors` (Z lives
///            in the workspace)
/// @ingroup api_device_lapack_eig
template <JobType J> struct stedc {
    /// The name `with_tier` puts in its error message.
    static constexpr const char* name = "stedc";
};
}  // namespace routine

/// @brief The tiers `with_tier` tries for `stedc`: `tier::Blocked<1>` only.
/// @ingroup api_device_lapack_eig
template <JobType J, class Real> struct tiers<routine::stedc<J>, Real> {
    using type = tier::list<tier::Blocked<1>>;
};
/// @brief The one tier of `stedc`, `tier::Blocked<1>`: real data, any n >= 1, one matrix per
/// `WorkGroup<256, 32>` (`wg_size` 256, the gemm's item count; `sg_size` 32, one secular root per
/// sub-group), global and local workspace.
///
/// The merges never hold the matrix in local memory, so one tier serves every n.
/// @ingroup api_device_lapack_eig
template <JobType J, class Real> struct tier_info<routine::stedc<J>, tier::Blocked<1>, Real> {
    static_assert(std::is_floating_point_v<Real>, "stedc: a real T");
    static constexpr int wg_size = detail::wg_size, sg_size = detail::sg_size;
    using executor = WorkGroup<wg_size, sg_size>;
    static constexpr int per_group = 1;
    static constexpr bool admits(int n) { return n >= 1; }
    static constexpr WorkspaceSize buffer_size(int n) { return detail::dc_layout<Real>(n, J == JobType::NoEigenVectors).size; }
};

/// @brief xSTEDC with eigenvectors on one real symmetric tridiagonal (d, e) inside a kernel: the
/// eigenvalues in w, ascending, and Z(:, j) the unit eigenvector of w(j).
///
/// n = `d.size()`. COLLECTIVE over `ex`, a `WorkGroup<256, 32>` launched at sub-group size 32
/// (`BATCHLAS_REQD_SG_SIZE(L::sg_size)`). d and e must be visible to every item on entry and are
/// not written; w and Z are visible on return. n <= 0 returns 0 at once.
/// @tparam J     `JobType::EigenVectors` (the overload without Z takes `NoEigenVectors`)
/// @tparam Tier  `tier::Blocked<1>`
/// @param[in]  ex    the work-group executor
/// @param[in]  d     n reals, the diagonal
/// @param[in]  e     n - 1 reals, the off-diagonal
/// @param[out] w     n reals, the eigenvalues ascending
/// @param[out] z     n x n or larger (its leading n x n block), the eigenvectors
/// @param      ws    this matrix's slices of `buffer_size<routine::stedc<J>, Tier, T>(n)`
/// @param[in]  opts  the secular iteration cap
/// @retval 0  success
/// @retval 1  a secular root ran out of `opts.max_sec_iter` (xSTEDC's info)
/// @return    The value is uniform over the executor.
/// @ingroup api_device_lapack_eig
template <JobType J, class Tier, class Ex, class D, class E, class W, class Z, class Real>
BATCHLAS_DEVICE_HELPER int stedc(const Ex& ex, const D& d, const E& e, const W& w, const Z& z, const Workspace& ws,
                              const StedcOptions<Real>& opts) {
    static_assert(J == JobType::EigenVectors, "stedc: J = NoEigenVectors takes no Z (its Z lives in the workspace)");
    static_assert(std::is_same_v<resolve_tier_t<routine::stedc<J>, Real, Tier>, tier::Blocked<1>>, "stedc: tier::Blocked<1>");
    detail::assert_types<Ex, D, Real>();
    const int n = d.size();
    if (n <= 0) return 0;
    const auto g = l3::global_of(z);
    const auto L = detail::dc_layout<Real>(n, false);
    return detail::bottom_up(ex, d, e, w, l3::Global<Real>{g.p, g.ld, n, n}, detail::ptrs(ws, L), opts.max_sec_iter);
}

/// @brief xSTEDC without eigenvectors on one real symmetric tridiagonal (d, e) inside a kernel:
/// the eigenvalues in w, ascending.
///
/// The merges still form the vectors, in the workspace, so the workspace is larger than with
/// `JobType::EigenVectors`. Otherwise as the overload with Z.
/// @tparam J     `JobType::NoEigenVectors`
/// @tparam Tier  `tier::Blocked<1>`
/// @param[in]  ex    the work-group executor
/// @param[in]  d     n reals, the diagonal
/// @param[in]  e     n - 1 reals, the off-diagonal
/// @param[out] w     n reals, the eigenvalues ascending
/// @param      ws    this matrix's slices of `buffer_size<routine::stedc<J>, Tier, T>(n)`
/// @param[in]  opts  the secular iteration cap
/// @retval 0  success
/// @retval 1  a secular root ran out of `opts.max_sec_iter`
/// @return    The value is uniform over the executor.
/// @ingroup api_device_lapack_eig
template <JobType J, class Tier, class Ex, class D, class E, class W, class Real>
BATCHLAS_DEVICE_HELPER int stedc(const Ex& ex, const D& d, const E& e, const W& w, const Workspace& ws, const StedcOptions<Real>& opts) {
    static_assert(J == JobType::NoEigenVectors, "stedc: J = EigenVectors takes Z");
    static_assert(std::is_same_v<resolve_tier_t<routine::stedc<J>, Real, Tier>, tier::Blocked<1>>, "stedc: tier::Blocked<1>");
    detail::assert_types<Ex, D, Real>();
    const int n = d.size();
    if (n <= 0) return 0;
    const auto L = detail::dc_layout<Real>(n, true);
    return detail::bottom_up(ex, d, e, w, l3::Global<Real>{carve_global(ws, L.z), n, n, n}, detail::ptrs(ws, L), opts.max_sec_iter);
}

}  // namespace batchlas::device
