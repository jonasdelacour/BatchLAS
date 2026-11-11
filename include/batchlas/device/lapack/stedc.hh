// SPDX-License-Identifier: MIT
#pragma once
// xSTEDC: eigenvalues and eigenvectors of a symmetric tridiagonal (d, e) by divide and conquer,
// one matrix inside one work-group, any n (design C.7, syev Blocked's "stedc (risky)").
//   stedc                    the public op, tier::Blocked<1>: every off-diagonal is a tear, so
//                            the leaves are 1 x 1 (no leaf solver) and the merges run bottom-up,
//                            widths 1, 2, 4, ...; each merge is the host merge's bodies:
//                            laed2_merge_vector, a rank sort, laed2 (deflation), laed3 (roots on
//                            one sub-group each, rescale, vectors), then the back-transform
//                            Q[:, map] * Q' on the in-group level 3 gemm
//   stedc_detail::dc_layout  the workspace plan: the gemm's staging and one slot in local
//                            memory; per-matrix vectors and two n x n blocks in global memory
// Eigenvalues are returned ascending, Z(:, j) the vector of w(j). Real T only.

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

namespace stedc_detail {

// max_sec_iter: the secular iteration cap per root (StedcParams::max_sec_iter, >= 1).
template <class R> struct Params {
    std::int32_t max_sec_iter = 50;
};

// The executor the tier is written for (the gemm's 16 x 16 items) and one root per sub-group.
inline constexpr int wg_size = 256, sg_size = 32, part_size = 32;
template <class R> using gemm_cfg = l3::cfg::RegTiled<R, 64, 64, 16, 4, 4>;

// The workspace plan (design C.5). Local: the back-transform's staging and the merge's kept
// count. Global, per matrix: the poles and weights before (A) and after (B) the sort, the roots,
// |.| scratch, the column maps and deflation scans (n each), the secular vectors Q' and the
// gathered columns (n x n each), and Z itself when no vectors are wanted.
template <class R> struct Layout {
    ws::Slot<R> smem;
    ws::Slot<std::int32_t> reduced;
    ws::Slot<R> d_a, z_a, d_b, z_b, lam, absx;
    ws::Slot<std::int32_t> map_a, map_b, keep, perm, plocal, scan_inc, scan_exc, rank;
    ws::Slot<R> qp, tmp, z;
    WorkspaceSize size;
};
template <class R> constexpr Layout<R> dc_layout(int n, bool own_z) {
    Plan p;
    const std::size_t un = std::size_t(n), nn = un * un;
    const auto smem = p.take<R>(ws::Space::Local, std::size_t(l3::smem_elems<gemm_cfg<R>>()));
    const auto reduced = p.take<std::int32_t>(ws::Space::Local, 1);
    const auto d_a = p.take<R>(ws::Space::Global, un);
    const auto z_a = p.take<R>(ws::Space::Global, un);
    const auto d_b = p.take<R>(ws::Space::Global, un);
    const auto z_b = p.take<R>(ws::Space::Global, un);
    const auto lam = p.take<R>(ws::Space::Global, un);
    const auto absx = p.take<R>(ws::Space::Global, un);
    const auto map_a = p.take<std::int32_t>(ws::Space::Global, un);
    const auto map_b = p.take<std::int32_t>(ws::Space::Global, un);
    const auto keep = p.take<std::int32_t>(ws::Space::Global, un);
    const auto perm = p.take<std::int32_t>(ws::Space::Global, un);
    const auto plocal = p.take<std::int32_t>(ws::Space::Global, un);
    const auto scan_inc = p.take<std::int32_t>(ws::Space::Global, un);
    const auto scan_exc = p.take<std::int32_t>(ws::Space::Global, un);
    const auto rank = p.take<std::int32_t>(ws::Space::Global, un);
    const auto qp = p.take<R>(ws::Space::Global, nn);
    const auto tmp = p.take<R>(ws::Space::Global, nn);
    const auto z = p.take<R>(ws::Space::Global, own_z ? nn : 0);
    return {smem, reduced, d_a, z_a, d_b, z_b, lam, absx, map_a, map_b, keep, perm, plocal, scan_inc, scan_exc, rank, qp, tmp, z, p.size()};
}

// rank(i) among x(0..s) by (value, index): a stable ascending sort as a gather-free scatter.
template <class R> BATCHLAS_DLA_INLINE int sort_rank(const R* x, int s, int i) {
    const R xi = x[i];
    int r = 0;
    for (int j = 0; j < s; ++j) {
        const R xj = x[j];
        r += (xj < xi || (xj == xi && j < i)) ? 1 : 0;
    }
    return r;
}

// The workspace's pointers for one matrix.
template <class R> struct Ptrs {
    R *d_a, *z_a, *d_b, *z_b, *lam, *absx, *qp, *tmp;
    std::int32_t *map_a, *map_b, *keep, *perm, *plocal, *scan_inc, *scan_exc, *rank;
    local_mptr<std::int32_t> reduced;
    l3::Smem<gemm_cfg<R>> smem;
};
template <class R> BATCHLAS_DLA_INLINE Ptrs<R> ptrs(const Workspace& ws, const Layout<R>& L) {
    return {carve_global(ws, L.d_a),     carve_global(ws, L.z_a),      carve_global(ws, L.d_b),      carve_global(ws, L.z_b),
            carve_global(ws, L.lam),     carve_global(ws, L.absx),     carve_global(ws, L.qp),       carve_global(ws, L.tmp),
            carve_global(ws, L.map_a),   carve_global(ws, L.map_b),    carve_global(ws, L.keep),     carve_global(ws, L.perm),
            carve_global(ws, L.plocal),  carve_global(ws, L.scan_inc), carve_global(ws, L.scan_exc), carve_global(ws, L.rank),
            carve_local(ws, L.reduced),  l3::Smem<gemm_cfg<R>>{carve_local(ws, L.smem).get_raw()}};
}

// One merge: rows / columns [o, o + s) of Z hold diag(Q1, Q2) (Q1 m x m), w(o..o + s) the two
// halves' ascending eigenvalues, rho the tear e(o + m - 1). On return they hold the merged
// problem's, ascending. Returns this item's count of unconverged roots.
template <class Ex, class PEx, class W, class R>
BATCHLAS_DLA_HELPER int merge(const Ex& ex, const PEx& pex, int part_id, int parts, int n, int o, int s, int m, R rho, const W& w,
                              const l3::Global<R>& Z, const Ptrs<R>& P, std::int32_t max_iter) {
    const int ld = Z.ld;
    R* const q0 = Z.p + o + static_cast<std::ptrdiff_t>(o) * ld;
    const auto Qn = mat(q0, s, s, ld);
    const R sign = rho >= R(0) ? R(1) : R(-1);

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
        const VectorView<R> v(P.z_a, dd);
        laed3<R>(ex, pex, part_id, parts, dd, dv, zv, sycl::fabs(R(2) * rho), max_iter, true, Qp, Qp_rows, v, 0,
                 [&](std::int32_t k, const laed4_root<R>& root) {
                     P.lam[k] = root.origin + root.tau;
                     bad += root.converged ? 0 : 1;
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
        gemm<Transpose::NoTrans, Transpose::NoTrans, gemm_cfg<R>>(ex, l3::Global<R>{P.tmp, n, s, dd}, l3::Global<R>{P.qp, n, dd, dd},
                                                                  l3::Global<R>{q0, ld, s, dd}, One{}, Zero{}, P.smem);
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
template <class Ex, class D, class E, class W, class R>
BATCHLAS_DLA_HELPER int bottom_up(const Ex& ex, const D& d, const E& e, const W& w, const l3::Global<R>& Z, const Ptrs<R>& P,
                            std::int32_t max_iter) {
    const int n = d.size();
    // Every off-diagonal tears: w(i) = d(i) - |e(i-1)| - |e(i)|, the 1 x 1 leaves; Z = I.
    for_each_owned(ex, n, [&](int i) {
        R di = d(i);
        if (i > 0) di -= sycl::fabs(e(i - 1));
        if (i + 1 < n) di -= sycl::fabs(e(i));
        w(i) = di;
    });
    for (int j = 0; j < n; ++j) for_each_owned(ex, n, [&](int i) { Z(i, j) = i == j ? R(1) : R(0); });
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
            bad += merge(ex, pex, part_id, parts, n, o, s, width, R(e(o + width - 1)), w, Z, P, max_iter);
        }
    }
    return allreduce(ex, bad, sycl::maximum<int>()) > 0 ? 1 : 0;
}

template <class Ex, class D, class R> constexpr void assert_types() {
    static_assert(std::is_floating_point_v<R>, "stedc: a real T");
    static_assert(std::is_same_v<elem_t<D>, R>, "stedc: params of the matrix's real type");
    static_assert(is_work_group_v<Ex> && Ex::size == wg_size && Ex::sg_size == sg_size, "stedc: executor WorkGroup<256, 32>");
}

}  // namespace stedc_detail

namespace routine {
template <JobType J = JobType::EigenVectors> struct stedc {
    static constexpr const char* name = "stedc";
};
}  // namespace routine

template <JobType J, class T> struct tiers<routine::stedc<J>, T> {
    using type = tier::list<tier::Blocked<1>>;
};
// One tier for every n: the merges never hold the matrix in local memory. The executor is
// WorkGroup<256, 32> (the gemm's item count) at sub-group size 32 (one root per sub-group).
template <JobType J, class T> struct tier_info<routine::stedc<J>, tier::Blocked<1>, T> {
    static_assert(std::is_floating_point_v<T>, "stedc: a real T");
    static constexpr int wg_size = stedc_detail::wg_size, sg_size = stedc_detail::sg_size;
    static constexpr bool admits(int n) { return n >= 1; }
    static constexpr WorkspaceSize buffer_size(int n) { return stedc_detail::dc_layout<T>(n, J == JobType::NoEigenVectors).size; }
};

template <JobType J, class Tier, class T> constexpr WorkspaceSize stedc_buffer_size(int n) {
    return tier_info<routine::stedc<J>, resolve_tier_t<routine::stedc<J>, T, Tier>, T>::buffer_size(n);
}

/// Eigenvalues w (ascending) and eigenvectors Z of one symmetric tridiagonal (d, e), host order
/// (d, e, w, ws, params, Z) with jobz the template J = EigenVectors. COLLECTIVE over ex, a
/// WorkGroup<256, 32> launched at sub-group size 32; d and e must be visible to every item on
/// entry and are not written; w and Z (n x n or larger: its leading n x n block) are visible on
/// return. Returns xSTEDC's info, uniform: 0, or 1 when a secular root ran out of
/// params.max_sec_iter. Workspace: stedc_buffer_size<J, Tier, T>(n), ws.global this matrix's slice.
template <JobType J, class Tier, class Ex, class D, class E, class W, class Z, class R>
BATCHLAS_DLA_HELPER int stedc(const Ex& ex, const D& d, const E& e, const W& w, const Workspace& ws, const stedc_detail::Params<R>& p,
                              const Z& z) {
    static_assert(J == JobType::EigenVectors, "stedc: J = NoEigenVectors takes no Z (its Z lives in the workspace)");
    static_assert(std::is_same_v<resolve_tier_t<routine::stedc<J>, R, Tier>, tier::Blocked<1>>, "stedc: tier::Blocked<1>");
    stedc_detail::assert_types<Ex, D, R>();
    const int n = d.size();
    if (n <= 0) return 0;
    const auto g = l3::global_of(z);
    const auto L = stedc_detail::dc_layout<R>(n, false);
    return stedc_detail::bottom_up(ex, d, e, w, l3::Global<R>{g.p, g.ld, n, n}, stedc_detail::ptrs(ws, L), p.max_sec_iter);
}

/// Eigenvalues only (J = NoEigenVectors): the merges still form the vectors, in the workspace.
template <JobType J, class Tier, class Ex, class D, class E, class W, class R>
BATCHLAS_DLA_HELPER int stedc(const Ex& ex, const D& d, const E& e, const W& w, const Workspace& ws, const stedc_detail::Params<R>& p) {
    static_assert(J == JobType::NoEigenVectors, "stedc: J = EigenVectors takes Z");
    static_assert(std::is_same_v<resolve_tier_t<routine::stedc<J>, R, Tier>, tier::Blocked<1>>, "stedc: tier::Blocked<1>");
    stedc_detail::assert_types<Ex, D, R>();
    const int n = d.size();
    if (n <= 0) return 0;
    const auto L = stedc_detail::dc_layout<R>(n, true);
    return stedc_detail::bottom_up(ex, d, e, w, l3::Global<R>{carve_global(ws, L.z), n, n, n}, stedc_detail::ptrs(ws, L), p.max_sec_iter);
}

}  // namespace batchlas::device
