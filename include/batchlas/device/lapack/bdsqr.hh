// SPDX-License-Identifier: MIT
#pragma once
// LAPACK xBDSQR on a real upper bidiagonal B = (d, e): implicit shifted QR sweeps with DBDSQR's
// zero-shift chase for a negligible diagonal, the singular values in s, the rotations accumulated
// into U (U := U Q over its n columns) and Vh (Vh := P^T Vh over its n rows), optionally sorted
// descending. ONE body for every executor:
//   Thread                one item runs the sweep and applies each rotation as it forms it
//   tier::Blocked<1>      any n on a work-group: the leader runs the scalar recurrence and stores
//                         one block's rotations in the global workspace; every item then applies
//                         them, in the same order, to the rows of U and the columns of Vh it owns
// Each element of U and Vh sees the same rotations in the same order on both executors, and the
// same as the host route's kernel (gesvd/bdsqr/bdsqr.cc, one item per matrix). That kernel stays
// its own body: as a launcher over the Thread path its outputs were bit-identical but it ran
// 1.03-1.08x (n = 32) and 1.38x / 1.77x (float / double, n = 64, values only) slower, batch 1024.

#include <algorithm>
#include <cstdint>
#include <limits>

#include <sycl/sycl.hpp>

#include <batchlas/device/core.hh>
#include <batchlas/device/exec.hh>
#include <batchlas/device/level1.hh>
#include <batchlas/device/operand.hh>
#include <batchlas/device/scalar.hh>
#include <batchlas/device/tier.hh>
#include <batchlas/device/workspace.hh>

namespace batchlas::device::inline v2 {

namespace routine {
struct bdsqr {
    static constexpr const char* name = "bdsqr";
};
}  // namespace routine

namespace bdsqr_detail {
// DBDSQR's tolerance (tolmul * eps, tolmul = clamp(eps^(-1/8), 10, 100)) and its iteration cap
// (maxitr * n^2, maxitr = 6, at least 32).
template <class R> struct Params {
    R tol;
    int maxit;
};
template <class R> inline Params<R> params(int n) {
    const R eps = std::numeric_limits<R>::epsilon();
    const R tolmul = sycl::fmax(R(10), sycl::fmin(R(100), sycl::pow(eps, R(-0.125))));
    return {tolmul * eps, std::max<int>(32, 6 * n * n)};
}

// One step of the leader's scan, published to the work-group.
enum Step : int { kNone = 0, kStop = 1, kChaseU = 2, kChaseV = 3, kSweep = 4 };
inline constexpr int kCtrl = 8;  // [0] converged, [1] step, [2] lo, [3] hi, [4] l, [5] iters, [6] info

template <class R> struct Layout {
    ws::Slot<int> ctrl;  // local: the published scan state
    ws::Slot<R> rot;     // global: 2 (n - 1) (c, s) of U, then 2 (n - 1) of Vh, one block's
    ws::Slot<int> perm;  // global: n sign flags, then the sort's swap targets
    WorkspaceSize size;
};
template <class R> constexpr Layout<R> sweep_plan(int n) {
    Plan p;
    const ws::Slot<int> ctrl = p.take<int>(ws::Space::Local, kCtrl);
    const ws::Slot<R> rot = p.take<R>(ws::Space::Global, std::size_t(4) * std::size_t(n > 1 ? n - 1 : 1));
    const ws::Slot<int> perm = p.take<int>(ws::Space::Global, std::size_t(n > 0 ? n : 1));
    return {ctrl, rot, perm, p.size()};
}

// (x0, x1) := (c x0 + s x1, -s x0 + c x1): DBDSQR's rotation in its expression order.
template <class R> BATCHLAS_DLA_INLINE void rot_pair(R& x0, R& x1, R c, R s) {
    const R a = x0, b = x1;
    x0 = c * a + s * b;
    x1 = -s * a + c * b;
}
// Columns c0, c1 of a over the rows the item owns; rows r0, r1 over the columns it owns.
template <class Ex, class M, class R> BATCHLAS_DLA_INLINE void rot_cols(const Ex& ex, const M& a, int c0, int c1, R c, R s) {
    for_each_owned(ex, a.rows(), [&](int r) { rot_pair<R>(a(r, c0), a(r, c1), c, s); });
}
template <class Ex, class M, class R> BATCHLAS_DLA_INLINE void rot_rows(const Ex& ex, const M& a, int r0, int r1, R c, R s) {
    for_each_owned(ex, a.cols(), [&](int col) { rot_pair<R>(a(r0, col), a(r1, col), c, s); });
}
// The sort's interchange of elements (i0, j0) and (i1, j1) of a (any operand: ItemMat has no
// col / row views for core swap).
template <class M> BATCHLAS_DLA_INLINE void swap_at(const M& a, int i0, int j0, int i1, int j1) {
    const auto tmp = a(i0, j0);
    a(i0, j0) = a(i1, j1);
    a(i1, j1) = tmp;
}
}  // namespace bdsqr_detail

template <class T> struct tiers<routine::bdsqr, T> {
    using type = tier::list<tier::Blocked<1>>;
};
// wg_size is what with_tier checks; the op runs on any work-group width (items past the owned
// count idle). Real T only (B is real; gebrd's complex reduction hands bdsqr a real B too).
template <class T> struct tier_info<routine::bdsqr, tier::Blocked<1>, T> {
    static_assert(std::is_floating_point_v<T>, "bdsqr: a real T");
    static constexpr int wg_size = 256, sg_size = 32;
    static constexpr bool admits(int n) { return n >= 1; }
    static constexpr WorkspaceSize buffer_size(int n) { return bdsqr_detail::sweep_plan<T>(n).size; }
};
template <class Tier, class T> constexpr WorkspaceSize bdsqr_buffer_size(int n) {
    return tier_info<routine::bdsqr, resolve_tier_t<routine::bdsqr, T, Tier>, T>::buffer_size(n);
}

// xBDSQR of one bidiagonal, host order (d, e, s, ws, U, Vh, sort_desc[, params]). d (n) and e
// (n - 1) are working storage (undefined on return); s receives the n singular values (|d| after
// convergence, descending when sort_desc). U is accumulated when it has n columns and at least one
// row, Vh when it has n rows and at least one column (pass a 0 x 0 operand to skip either).
// COLLECTIVE over ex with the same arguments; d, e, U, Vh visible on entry; outputs visible on
// return. Workspace: bdsqr_buffer_size<Tier, T>(n) on a work-group, none on Thread. Returns xBDSQR's
// status as the host's flag: 0, or 1 when the sweep hit maxit or a singular value is NaN.
template <class Tier = tier::Blocked<1>, class Ex, class D, class E, class S, class MU, class MV, class R>
BATCHLAS_DLA_INLINE int bdsqr(const Ex& ex, const D& d, const E& e, const S& s, const Workspace& ws, const MU& U, const MV& Vh,
                              bool sort_desc, const bdsqr_detail::Params<R>& p) {
    using namespace bdsqr_detail;
    static_assert(std::is_same_v<elem_t<D>, R>, "bdsqr: params of the bidiagonal's real type");
    static_assert(std::is_same_v<resolve_tier_t<routine::bdsqr, R, Tier>, tier::Blocked<1>>, "bdsqr: tier::Blocked<1>");
    constexpr bool solo = is_thread_v<Ex>;
    const R eps = std::numeric_limits<R>::epsilon();
    const int n = d.size();
    const bool acc_u = U.rows() > 0 && U.cols() == n;
    const bool acc_vh = Vh.rows() == n && Vh.cols() > 0;
    const auto L = sweep_plan<R>(n);
    int* ctrl = nullptr;
    R* ru = nullptr;
    R* rv = nullptr;
    int* perm = nullptr;
    if constexpr (!solo) {
        ctrl = carve_local(ws, L.ctrl).get_raw();
        ru = carve_global(ws, L.rot);
        rv = ru + 2 * (n > 1 ? n - 1 : 1);
        perm = carve_global(ws, L.perm);
    }
    // Rotation i of the current block: applied now (Thread) or stored for the work-group.
    auto emit_u = [&]([[maybe_unused]] int i, [[maybe_unused]] int c0, [[maybe_unused]] int c1, R c, R sn) {
        if constexpr (solo) {
            if (acc_u) rot_cols(ex, U, c0, c1, c, sn);
        } else {
            ru[2 * i] = c;
            ru[2 * i + 1] = sn;
        }
    };
    auto emit_vh = [&]([[maybe_unused]] int i, [[maybe_unused]] int r0, [[maybe_unused]] int r1, R c, R sn) {
        if constexpr (solo) {
            if (acc_vh) rot_rows(ex, Vh, r0, r1, c, sn);
        } else {
            rv[2 * i] = c;
            rv[2 * i + 1] = sn;
        }
    };

    int iters = 0;
    bool converged = (n <= 1);
    R abs_thresh = R(0);  // the leader's
    while (!converged && iters < p.maxit) {
        int conv = 1;
        if (ex.leader()) {
            R smax = R(0);
            for (int i = 0; i < n; ++i) smax = sycl::fmax(smax, sycl::fabs(d(i)));
            for (int i = 0; i < n - 1; ++i) smax = sycl::fmax(smax, sycl::fabs(e(i)));
            abs_thresh = p.tol * smax;
            for (int i = 0; i < n - 1; ++i) {
                const R thresh = sycl::fmax(eps * (sycl::fabs(d(i)) + sycl::fabs(d(i + 1))), abs_thresh);
                if (sycl::fabs(e(i)) <= thresh) e(i) = R(0);
                else conv = 0;
            }
        }
        if constexpr (!solo) {
            if (ex.leader()) ctrl[0] = conv;
            sync(ex, Mem::Local);
            conv = ctrl[0];
        }
        converged = conv != 0;
        if (converged) break;

        int l = 0;
        while (l < n - 1) {
            int step = kNone, lo = 0, hi = 0;
            if (ex.leader()) {
                while (l < n - 1 && e(l) == R(0)) ++l;
                if (l >= n - 1) {
                    step = kStop;
                } else {
                    int m = l;
                    while (m < n - 1 && e(m) != R(0)) ++m;
                    int zrow = -1;
                    for (int i = l; i <= m; ++i) {
                        if (sycl::fabs(d(i)) <= sycl::fmax(abs_thresh, std::numeric_limits<R>::min())) {
                            zrow = i;
                            break;
                        }
                    }
                    if (zrow >= 0) {
                        // DBDSQR's zero-shift chase: a zero inside the block empties its row with
                        // left rotations (rightwards); one at the bottom empties column m with
                        // right rotations (leftwards).
                        // evidence: docs/perf/gesvd.md#gesvd-bdsqr-the-missing-zero-shift-chase
                        d(zrow) = R(0);
                        if (zrow < m) {
                            R f2 = e(zrow);
                            e(zrow) = R(0);
                            for (int j = zrow + 1; j <= m; ++j) {
                                const auto rz = lartg(static_cast<R>(d(j)), f2);
                                d(j) = rz.r;
                                emit_u(j - zrow - 1, j, zrow, rz.c, rz.s);
                                if (j < m) {
                                    f2 = -rz.s * e(j);
                                    e(j) = rz.c * e(j);
                                }
                            }
                            step = kChaseU;
                            lo = zrow;
                        } else if (m > l) {
                            R f2 = e(m - 1);
                            e(m - 1) = R(0);
                            for (int j = m - 1; j >= l; --j) {
                                const auto rz = lartg(static_cast<R>(d(j)), f2);
                                d(j) = rz.r;
                                emit_vh(m - 1 - j, j, m, rz.c, rz.s);
                                if (j > l) {
                                    f2 = -rz.s * e(j - 1);
                                    e(j - 1) = rz.c * e(j - 1);
                                }
                            }
                            step = kChaseV;
                            lo = l;
                        }
                    } else {
                        // DBDSQR's shift: the smaller singular value of the trailing 2 x 2 (xLAS2),
                        // dropped when negligible against d(l).
                        R shift = las2(static_cast<R>(d(m - 1)), static_cast<R>(e(m - 1)), static_cast<R>(d(m)))[0];
                        const R sll = sycl::fabs(d(l));
                        if (sll > R(0) && (shift / sll) * (shift / sll) < eps) shift = R(0);
                        R f = (sll - shift) * (sycl::copysign(R(1), d(l)) + shift / d(l));
                        R g = e(l);
                        for (int k = l; k <= m - 1; ++k) {
                            const auto r1 = lartg(f, g);
                            emit_vh(k - l, k, k + 1, r1.c, r1.s);
                            if (k > l) e(k - 1) = r1.r;
                            const R dk = d(k), ek = e(k), dk1 = d(k + 1);
                            f = r1.c * dk + r1.s * ek;
                            e(k) = r1.c * ek - r1.s * dk;
                            g = r1.s * dk1;
                            d(k + 1) = r1.c * dk1;
                            const auto r2 = lartg(f, g);
                            emit_u(k - l, k, k + 1, r2.c, r2.s);
                            d(k) = r2.r;
                            const R dk1b = d(k + 1), ekb = e(k);
                            f = r2.c * ekb + r2.s * dk1b;
                            d(k + 1) = r2.c * dk1b - r2.s * ekb;
                            if (k < m - 1) {
                                g = r2.s * e(k + 1);
                                e(k + 1) = r2.c * e(k + 1);
                            } else {
                                g = R(0);
                            }
                            e(k) = f;
                        }
                        step = kSweep;
                        lo = l;
                    }
                    hi = m;
                    iters += m - l + 1;
                    l = m + 1;
                }
            }
            if constexpr (!solo) {
                if (ex.leader()) {
                    ctrl[1] = step;
                    ctrl[2] = lo;
                    ctrl[3] = hi;
                    ctrl[4] = l;
                    ctrl[5] = iters;
                }
                sync(ex, Mem::Global);
                step = ctrl[1];
                lo = ctrl[2];
                hi = ctrl[3];
                l = ctrl[4];
                iters = ctrl[5];
                if (step == kStop) break;
                // The block's rotations, in the leader's order, on every owned row / column.
                if (step == kChaseU && acc_u) {
                    for_each_owned(ex, U.rows(), [&](int r) {
                        for (int j = lo + 1; j <= hi; ++j) rot_pair<R>(U(r, j), U(r, lo), ru[2 * (j - lo - 1)], ru[2 * (j - lo - 1) + 1]);
                    });
                } else if (step == kChaseV && acc_vh) {
                    for_each_owned(ex, Vh.cols(), [&](int col) {
                        for (int j = hi - 1; j >= lo; --j)
                            rot_pair<R>(Vh(j, col), Vh(hi, col), rv[2 * (hi - 1 - j)], rv[2 * (hi - 1 - j) + 1]);
                    });
                } else if (step == kSweep) {
                    if (acc_vh)
                        for_each_owned(ex, Vh.cols(), [&](int col) {
                            for (int k = lo; k < hi; ++k) rot_pair<R>(Vh(k, col), Vh(k + 1, col), rv[2 * (k - lo)], rv[2 * (k - lo) + 1]);
                        });
                    if (acc_u)
                        for_each_owned(ex, U.rows(), [&](int r) {
                            for (int k = lo; k < hi; ++k) rot_pair<R>(U(r, k), U(r, k + 1), ru[2 * (k - lo)], ru[2 * (k - lo) + 1]);
                        });
                }
                sync(ex, Mem::Global);
            } else if (step == kStop) {
                break;
            }
        }
    }

    // s := |d|, a negative d(i) flipping row i of Vh; then the descending selection sort, its
    // swaps applied to U's columns and Vh's rows; info from the convergence and a NaN scan.
    if (ex.leader()) {
        for (int i = 0; i < n; ++i) {
            const bool neg = d(i) < R(0);
            if (neg) {
                d(i) = -d(i);
                if constexpr (solo) {
                    if (acc_vh) for_each_owned(ex, Vh.cols(), [&](int col) { Vh(i, col) *= R(-1); });
                }
            }
            if constexpr (!solo) perm[i] = neg ? 1 : 0;
            s(i) = d(i);
        }
    }
    if constexpr (!solo) {
        sync(ex, Mem::Global);
        if (acc_vh)
            for_each_owned(ex, Vh.cols(), [&](int col) {
                for (int i = 0; i < n; ++i)
                    if (perm[i] != 0) Vh(i, col) *= R(-1);
            });
        sync(ex, Mem::Global);
    }
    int info = 0;
    if (ex.leader()) {
        for (int i = 0; i < n; ++i) {
            int best = i;
            if (sort_desc) {
                for (int j = i + 1; j < n; ++j)
                    if (s(j) > s(best)) best = j;
                if (best != i) {
                    const R tmp = s(i);
                    s(i) = s(best);
                    s(best) = tmp;
                    if constexpr (solo) {
                        if (acc_u) for_each_owned(ex, U.rows(), [&](int r) { swap_at(U, r, i, r, best); });
                        if (acc_vh) for_each_owned(ex, Vh.cols(), [&](int col) { swap_at(Vh, i, col, best, col); });
                    }
                }
            }
            if constexpr (!solo) perm[i] = best;
        }
        bool bad = false;
        for (int i = 0; i < n; ++i) bad = bad || !(s(i) == s(i));
        info = (converged && !bad) ? 0 : 1;
        if constexpr (!solo) ctrl[6] = info;
    }
    if constexpr (!solo) {
        sync(ex, Mem::Global);
        info = ctrl[6];
        if (sort_desc) {
            if (acc_u)
                for_each_owned(ex, U.rows(), [&](int r) {
                    for (int i = 0; i < n; ++i)
                        if (perm[i] != i) swap_at(U, r, i, r, perm[i]);
                });
            if (acc_vh)
                for_each_owned(ex, Vh.cols(), [&](int col) {
                    for (int i = 0; i < n; ++i)
                        if (perm[i] != i) swap_at(Vh, i, col, perm[i], col);
                });
        }
        sync(ex, Mem::Global);
    }
    return info;
}

// The defaults: DBDSQR's tolerance and iteration cap for n = d.size().
template <class Tier = tier::Blocked<1>, class Ex, class D, class E, class S, class MU, class MV>
BATCHLAS_DLA_INLINE int bdsqr(const Ex& ex, const D& d, const E& e, const S& s, const Workspace& ws, const MU& U, const MV& Vh,
                              bool sort_desc = true) {
    return bdsqr<Tier>(ex, d, e, s, ws, U, Vh, sort_desc, bdsqr_detail::params<elem_t<D>>(d.size()));
}
// Singular values only (the host's values-only overload).
template <class Tier = tier::Blocked<1>, class Ex, class D, class E, class S>
BATCHLAS_DLA_INLINE int bdsqr(const Ex& ex, const D& d, const E& e, const S& s, const Workspace& ws, bool sort_desc = true) {
    using R = elem_t<D>;
    const auto none = device::mat(static_cast<R*>(nullptr), 0, 0, 1);
    return bdsqr<Tier>(ex, d, e, s, ws, none, none, sort_desc, bdsqr_detail::params<R>(d.size()));
}

}  // namespace batchlas::device
