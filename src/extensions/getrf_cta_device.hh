#pragma once

// All of the native batched GETRF panel's device code. FOUR spellings here are wrong
// answers or launch failures if changed -- the butterfly pivot search, the cabs1 pivot
// metric, info's exact-zero/1-based/first-failure rule, and the top-of-loop barriers.
// evidence: docs/perf/lu.md#the-shape-of-the-getrf-device-code

#include "../sycl/device_scalar.hh"

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>

namespace batchlas::getrf_native {

using batchlas::sycl_device::Cx;

template <typename D> struct RealOf        { using type = D; };
template <typename R> struct RealOf<Cx<R>> { using type = R; };
template <typename D> using real_of = typename RealOf<D>::type;

template <typename D>
inline D lu_zero() {
    if constexpr (batchlas::sycl_device::dev_is_complex_v<D>) {
        return D{real_of<D>(0), real_of<D>(0)};
    } else {
        return D(0);
    }
}

// LAPACK's cabs1: |Re| + |Im|; deliberately not the modulus (see header).
template <typename D>
inline real_of<D> lu_cabs1(D a) {
    if constexpr (batchlas::sycl_device::dev_is_complex_v<D>) {
        return sycl::fabs(a.re) + sycl::fabs(a.im);
    } else {
        return sycl::fabs(a);
    }
}

template <typename D>
struct LuGlobalTile {
    D* p;
    int ld;
    D& at(int r, int c) const {
        return p[static_cast<std::ptrdiff_t>(r) + static_cast<std::ptrdiff_t>(c) * ld];
    }
};

// Own base per matrix (G share one accessor), and the launcher pads ld ODD against a
// bank conflict. evidence: docs/perf/lu.md#the-shape-of-the-getrf-device-code
template <typename D> using LuRawTile = LuGlobalTile<D>;

// CONSTANT, not a function of wg width: the capacity query, the fit predicate and the
// launcher must agree on the SLM footprint.
inline constexpr int kLuRedSlots = 32;

// Which group the panel's phase barriers synchronise. SubGroup is what makes G-packing
// legal: the G sub-groups sharing a work-group then never synchronise with each other.
// Choosing WorkGroup for a packed launch is a RACE by construction -- wrong answers, not
// a launch failure -- so the launcher derives the two together and asserts the pairing.
enum class LuScope { SubGroup, WorkGroup };

// LAPACK ?GETF2 on an m x n tile, in place, for `kmax` (not min(m, n)) steps.
// `piv_base` is the panel's first GLOBAL row index, so ipiv and info come out global
// and 1-based with no fix-up pass. `info_item` is READ as well as written
// (first-failure-wins across panels), so both launchers zero it beforehand.
template <typename D, LuScope SC, typename Tile, typename ValAcc, typename IdxAcc>
inline void getf2_panel_device(sycl::nd_item<1> it, Tile A, int m, int n, int kmax,
                               int* piv_item, int piv_base, int32_t* info_item,
                               ValAcc rval, IdxAcc ridx) {
    using R = real_of<D>;

    const auto g = it.get_group();
    const auto sg = it.get_sub_group();
    const int lane = static_cast<int>(sg.get_local_linear_id());
    const int nlanes = static_cast<int>(sg.get_local_linear_range());
    const int team = static_cast<int>(sg.get_group_linear_id());
    const int nteams = static_cast<int>(sg.get_group_linear_range());

    // Under SubGroup scope the work-group holds several independent matrices, so the
    // item's own lane index and the sub-group width -- not the work-group's -- are what
    // stride this matrix's rows.
    const int wg = (SC == LuScope::SubGroup) ? nlanes
                                             : static_cast<int>(it.get_local_range(0));
    const int tid = (SC == LuScope::SubGroup) ? lane
                                              : static_cast<int>(it.get_local_linear_id());

    // The phase barriers, at the scope this instantiation owns.
    const auto phase_barrier = [&]() {
        if constexpr (SC == LuScope::SubGroup) {
            sycl::group_barrier(sg);
        } else {
            sycl::group_barrier(g);
        }
    };

    int32_t info_local = (tid == 0) ? *info_item : 0;

    for (int k = 0; k < kmax; ++k) {
        R bv = R(-1);
        int bi = m;                       // "no candidate" -- never wins a tie
        for (int i = k + tid; i < m; i += wg) {
            const R v = lu_cabs1<D>(A.at(i, k));
            if (v > bv) { bv = v; bi = i; }
        }

        for (uint32_t off = static_cast<uint32_t>(nlanes) / 2; off > 0; off >>= 1) {
            const R ov = sycl::permute_group_by_xor(sg, bv, off);
            const int oi = sycl::permute_group_by_xor(sg, bi, off);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        R fv;
        int p;
        if constexpr (SC == LuScope::SubGroup) {
            // The butterfly above already spans every lane that scanned this column, so
            // there is nothing to combine and no B1: the SLM slots and the cross-team
            // scan exist only to join several sub-groups working on ONE matrix.
            fv = bv;
            p = bi;
            static_cast<void>(rval);
            static_cast<void>(ridx);
        } else {
            if (lane == 0) {
                rval[static_cast<std::size_t>(team)] = bv;
                ridx[static_cast<std::size_t>(team)] = bi;
            }
            sycl::group_barrier(g);                                    // B1

            fv = R(-1);
            p = m;
            for (int t = 0; t < nteams; ++t) {
                const R v = rval[static_cast<std::size_t>(t)];
                const int ii = ridx[static_cast<std::size_t>(t)];
                if (v > fv || (v == fv && ii < p)) { fv = v; p = ii; }
            }
        }
        static_cast<void>(fv);
        if (p >= m) p = k;   // unreachable: k < m, so column k always has a row

        if (tid == 0) {
            piv_item[k] = piv_base + p + 1;      // GLOBAL, 1-BASED, LAPACK ipiv
        }

        // Across ALL n columns, including the finished L to the left of k:
        // exchanging only columns >= k is the classic silently-wrong LU.
        if (p != k) {
            for (int c = tid; c < n; c += wg) {
                D& x = A.at(k, c);
                D& y = A.at(p, c);
                const D t = x;
                x = y;
                y = t;
            }
        }
        phase_barrier();                                               // B2

        const D d = A.at(k, k);
        if (batchlas::sycl_device::dev_is_zero(d)) {
            // EXACT zero, no epsilon: the column is left alone, keeping the item finite.
            if (tid == 0 && info_local == 0) {
                info_local = static_cast<int32_t>(piv_base + k + 1);
            }
        } else {
            const D r = batchlas::sycl_device::dev_recip(d);
            const bool use_mul = batchlas::sycl_device::dev_isfinite(r) &&
                                 !batchlas::sycl_device::dev_is_zero(r);
            if (use_mul) {
                for (int i = k + 1 + tid; i < m; i += wg) {
                    A.at(i, k) = batchlas::sycl_device::dev_mul(A.at(i, k), r);
                }
            } else {
                // ?GETF2's small-pivot arm: the reciprocal overflowed, so divide.
                for (int i = k + 1 + tid; i < m; i += wg) {
                    A.at(i, k) = batchlas::sycl_device::dev_div(A.at(i, k), d);
                }
            }
        }
        phase_barrier();                                               // B3

        // Flattened so the work-group saturates when one extent is short; the two
        // runtime divisions beat a power-of-two split (docs/perf/lu.md#negative-results).
        const int mm = m - k - 1;
        const int nn = n - k - 1;
        if (mm > 0 && nn > 0) {
            const int total = mm * nn;
            for (int e = tid; e < total; e += wg) {
                const int i = k + 1 + (e % mm);
                const int j = k + 1 + (e / mm);
                A.at(i, j) = batchlas::sycl_device::dev_sub(
                    A.at(i, j),
                    batchlas::sycl_device::dev_mul(A.at(i, k), A.at(k, j)));
            }
        }
        phase_barrier();                                               // B4
    }

    if (tid == 0) {
        *info_item = info_local;
    }
}

}  // namespace batchlas::getrf_native
