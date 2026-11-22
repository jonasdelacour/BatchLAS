// SPDX-License-Identifier: MIT
#pragma once
/// @file
/// @brief Bidiagonal SVD by QR iteration inside a kernel: `bdsqr`, LAPACK's xBDSQR on a real
/// upper bidiagonal B = (d, e).
///
/// Implicit shifted QR sweeps with DBDSQR's zero-shift chase for a negligible diagonal entry give
/// the singular values; the rotations are accumulated into U (\f$U \leftarrow U Q\f$ over its n
/// columns) and Vh (\f$V^H \leftarrow P^T V^H\f$ over its n rows), optionally sorted descending.
/// One body runs on two executors:
/// - a `Thread`: one work-item runs the sweep and applies each rotation as it forms it;
/// - `tier::Blocked<1>` on a work-group, any n: the leader runs the scalar recurrence and stores
///   one block's rotations in the global workspace, then every item applies them, in the same
///   order, to the rows of U and the columns of Vh it owns.
///
/// Each element of U and Vh sees the same rotations in the same order on both executors.
/// Reference: @ref api_device_lapack_svd.
/// @ingroup api_device_lapack_svd
// The host route (gesvd/bdsqr/bdsqr.cc) is a launcher over the Thread path, one item per matrix.

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

namespace batchlas::device {

namespace routine {
/// @brief The routine tag of `bdsqr`, for `with_tier`, `launch` and `buffer_size`.
/// @ingroup api_device_lapack_svd
struct bdsqr {
    static constexpr const char* name = "bdsqr";
};
}  // namespace routine

/// @brief `bdsqr`'s options; the zero-initialised defaults are DBDSQR's.
/// @tparam Real  the bidiagonal's real type
/// @ingroup api_device_lapack_svd
template <class Real> struct BdsqrOptions {
    /// Sort the singular values descending, permuting U's columns and Vh's rows with them.
    bool sort_desc = true;
    /// The relative tolerance; <= 0 is DBDSQR's tolmul * eps, tolmul = clamp(eps^(-1/8), 10, 100).
    Real tol = Real(0);
    /// The iteration cap; <= 0 is DBDSQR's max(32, 6 n^2).
    int maxit = 0;
};

namespace detail {
// DBDSQR's tolerance (tolmul * eps, tolmul = clamp(eps^(-1/8), 10, 100)) and its iteration cap
// (maxitr * n^2, maxitr = 6, at least 32).
template <class Real> struct bdsqr_params {
    Real tol;
    int maxit;
};
template <class Real> inline bdsqr_params<Real> params(int n) {
    const Real eps = std::numeric_limits<Real>::epsilon();
    const Real tolmul = sycl::fmax(Real(10), sycl::fmin(Real(100), sycl::pow(eps, Real(-0.125))));
    return {tolmul * eps, std::max<int>(32, 6 * n * n)};
}

// One step of the leader's scan, published to the work-group.
enum Step : int { kNone = 0, kStop = 1, kChaseU = 2, kChaseV = 3, kSweep = 4 };
inline constexpr int kCtrl = 8;  // [0] converged, [1] step, [2] lo, [3] hi, [4] l, [5] iters, [6] info

template <class Real> struct BdsqrLayout {
    WsSlot<int> ctrl;  // local: the published scan state
    WsSlot<Real> rot;     // global: 2 (n - 1) (c, s) of U, then 2 (n - 1) of Vh, one block's
    WsSlot<int> perm;  // global: n sign flags, then the sort's swap targets
    WorkspaceSize size;
};
template <class Real> constexpr BdsqrLayout<Real> sweep_plan(int n) {
    Plan p;
    const WsSlot<int> ctrl = p.take<int>(WsSpace::Local, kCtrl);
    const WsSlot<Real> rot = p.take<Real>(WsSpace::Global, std::size_t(4) * std::size_t(n > 1 ? n - 1 : 1));
    const WsSlot<int> perm = p.take<int>(WsSpace::Global, std::size_t(n > 0 ? n : 1));
    return {ctrl, rot, perm, p.size()};
}

// (x0, x1) := (c x0 + s x1, -s x0 + c x1): DBDSQR's rotation in its expression order.
template <class Real> BATCHLAS_DEVICE_INLINE void rot_pair(Real& x0, Real& x1, Real c, Real s) {
    const Real a = x0, b = x1;
    x0 = c * a + s * b;
    x1 = -s * a + c * b;
}
// Columns c0, c1 of a over the rows the item owns; rows r0, r1 over the columns it owns.
template <class Ex, class M, class Real> BATCHLAS_DEVICE_INLINE void rot_cols(const Ex& ex, const M& a, int c0, int c1, Real c, Real s) {
    for_each_owned(ex, a.rows(), [&](int r) { rot_pair<Real>(a(r, c0), a(r, c1), c, s); });
}
template <class Ex, class M, class Real> BATCHLAS_DEVICE_INLINE void rot_rows(const Ex& ex, const M& a, int r0, int r1, Real c, Real s) {
    for_each_owned(ex, a.cols(), [&](int col) { rot_pair<Real>(a(r0, col), a(r1, col), c, s); });
}
// The sort's interchange of elements (i0, j0) and (i1, j1) of a (any operand: ItemMat has no
// col / row views for core swap).
template <class M> BATCHLAS_DEVICE_INLINE void swap_at(const M& a, int i0, int j0, int i1, int j1) {
    const auto tmp = a(i0, j0);
    a(i0, j0) = a(i1, j1);
    a(i1, j1) = tmp;
}
}  // namespace detail

/// @brief The tiers `with_tier` tries for `bdsqr`: `tier::Blocked<1>`.
/// @ingroup api_device_lapack_svd
template <class Real> struct tiers<routine::bdsqr, Real> {
    using type = tier::list<tier::Blocked<1>>;
};
/// @brief `bdsqr` in `tier::Blocked<1>`: one bidiagonal of any order n per work-group, real types
/// only.
///
/// `wg_size` (256) is what `with_tier` checks; the routine runs on any work-group width (items
/// past the owned count idle). B is real for every `gebrd` scalar type, so `Real` is a real
/// type. `buffer_size(n)` is one bidiagonal's local and global bytes.
/// @ingroup api_device_lapack_svd
template <class Real> struct tier_info<routine::bdsqr, tier::Blocked<1>, Real> {
    static_assert(std::is_floating_point_v<Real>, "bdsqr: a real T");
    static constexpr int wg_size = 256, sg_size = 32;
    using executor = WorkGroup<wg_size, sg_size>;
    static constexpr int per_group = 1;
    static constexpr bool admits(int n) { return n >= 1; }
    static constexpr WorkspaceSize buffer_size(int n) { return detail::sweep_plan<Real>(n).size; }
};

namespace detail {
// xBDSQR's body with the tolerance and iteration cap resolved: the public bdsqr below (DBDSQR's
// values for BdsqrOptions' zeros) and the host route (p formed on the host, a kernel argument).
template <class Tier, class Ex, class D, class E, class S, class MU, class MV, class Real>
BATCHLAS_DEVICE_INLINE int bdsqr_sweep(const Ex& ex, const D& d, const E& e, const S& s, const MU& U, const MV& Vh, const Workspace& ws,
                                       bool sort_desc, const bdsqr_params<Real>& p) {
    static_assert(std::is_same_v<elem_t<D>, Real>, "bdsqr: params of the bidiagonal's real type");
    static_assert(std::is_same_v<resolve_tier_t<routine::bdsqr, Real, Tier>, tier::Blocked<1>>,
                  "bdsqr: tier::Blocked<1>");
    constexpr bool solo = is_thread_v<Ex>;
    const Real eps = std::numeric_limits<Real>::epsilon();
    const int n = d.size();
    const bool acc_u = U.rows() > 0 && U.cols() == n;
    const bool acc_vh = Vh.rows() == n && Vh.cols() > 0;
    const auto L = detail::sweep_plan<Real>(n);
    int* ctrl = nullptr;
    Real* ru = nullptr;
    Real* rv = nullptr;
    int* perm = nullptr;
    if constexpr (!solo) {
        ctrl = carve_local(ws, L.ctrl).get_raw();
        ru = carve_global(ws, L.rot);
        rv = ru + 2 * (n > 1 ? n - 1 : 1);
        perm = carve_global(ws, L.perm);
    }
    // Rotation i of the current block: applied now (Thread) or stored for the work-group.
    auto emit_u = [&]([[maybe_unused]] int i, [[maybe_unused]] int c0, [[maybe_unused]] int c1, Real c, Real sn) {
        if constexpr (solo) {
            if (acc_u) detail::rot_cols(ex, U, c0, c1, c, sn);
        } else {
            ru[2 * i] = c;
            ru[2 * i + 1] = sn;
        }
    };
    auto emit_vh = [&]([[maybe_unused]] int i, [[maybe_unused]] int r0, [[maybe_unused]] int r1, Real c, Real sn) {
        if constexpr (solo) {
            if (acc_vh) detail::rot_rows(ex, Vh, r0, r1, c, sn);
        } else {
            rv[2 * i] = c;
            rv[2 * i + 1] = sn;
        }
    };

    int iters = 0;
    bool converged = (n <= 1);
    Real abs_thresh = Real(0);  // the leader's
    while (!converged && iters < p.maxit) {
        int conv = 1;
        if (ex.leader()) {
            Real smax = Real(0);
            for (int i = 0; i < n; ++i) smax = sycl::fmax(smax, sycl::fabs(d(i)));
            for (int i = 0; i < n - 1; ++i) smax = sycl::fmax(smax, sycl::fabs(e(i)));
            abs_thresh = p.tol * smax;
            for (int i = 0; i < n - 1; ++i) {
                const Real thresh = sycl::fmax(eps * (sycl::fabs(d(i)) + sycl::fabs(d(i + 1))), abs_thresh);
                if (sycl::fabs(e(i)) <= thresh) e(i) = Real(0);
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
            int step = detail::kNone, lo = 0, hi = 0;
            if (ex.leader()) {
                // One exit (no short circuit; n >= 2 here, so the clamped index is in range). The
                // `&&` form's l >= n - 1 exit jumps straight to the latch, ptxas then sets no
                // reconvergence point after the scan, and on Thread (one item per matrix) a warp's
                // items enter the sweep 2-3 at a time instead of ~15: 3.4x warp instructions,
                // 1.33x time at n = 64 (float, values only, batch 1024).
                while ((l < n - 1) & (e(sycl::min(l, n - 2)) == Real(0))) ++l;
                if (l >= n - 1) {
                    step = detail::kStop;
                } else {
                    int m = l;
                    while (m < n - 1 && e(m) != Real(0)) ++m;
                    int zrow = -1;
                    for (int i = l; i <= m; ++i) {
                        if (sycl::fabs(d(i)) <= sycl::fmax(abs_thresh, std::numeric_limits<Real>::min())) {
                            zrow = i;
                            break;
                        }
                    }
                    if (zrow >= 0) {
                        // DBDSQR's zero-shift chase: a zero inside the block empties its row with
                        // left rotations (rightwards); one at the bottom empties column m with
                        // right rotations (leftwards).
                        // evidence: docs/perf/gesvd.md#gesvd-bdsqr-the-missing-zero-shift-chase
                        d(zrow) = Real(0);
                        if (zrow < m) {
                            Real f2 = e(zrow);
                            e(zrow) = Real(0);
                            for (int j = zrow + 1; j <= m; ++j) {
                                const auto rz = lartg(static_cast<Real>(d(j)), f2);
                                d(j) = rz.r;
                                emit_u(j - zrow - 1, j, zrow, rz.c, rz.s);
                                if (j < m) {
                                    f2 = -rz.s * e(j);
                                    e(j) = rz.c * e(j);
                                }
                            }
                            step = detail::kChaseU;
                            lo = zrow;
                        } else if (m > l) {
                            Real f2 = e(m - 1);
                            e(m - 1) = Real(0);
                            for (int j = m - 1; j >= l; --j) {
                                const auto rz = lartg(static_cast<Real>(d(j)), f2);
                                d(j) = rz.r;
                                emit_vh(m - 1 - j, j, m, rz.c, rz.s);
                                if (j > l) {
                                    f2 = -rz.s * e(j - 1);
                                    e(j - 1) = rz.c * e(j - 1);
                                }
                            }
                            step = detail::kChaseV;
                            lo = l;
                        }
                    } else {
                        // DBDSQR's shift: the smaller singular value of the trailing 2 x 2 (xLAS2),
                        // dropped when negligible against d(l).
                        Real shift = las2(static_cast<Real>(d(m - 1)), static_cast<Real>(e(m - 1)),
                                          static_cast<Real>(d(m)))[0];
                        const Real sll = sycl::fabs(d(l));
                        if (sll > Real(0) && (shift / sll) * (shift / sll) < eps) shift = Real(0);
                        Real f = (sll - shift) * (sycl::copysign(Real(1), d(l)) + shift / d(l));
                        Real g = e(l);
                        for (int k = l; k <= m - 1; ++k) {
                            const auto r1 = lartg(f, g);
                            emit_vh(k - l, k, k + 1, r1.c, r1.s);
                            if (k > l) e(k - 1) = r1.r;
                            const Real dk = d(k), ek = e(k), dk1 = d(k + 1);
                            f = r1.c * dk + r1.s * ek;
                            e(k) = r1.c * ek - r1.s * dk;
                            g = r1.s * dk1;
                            d(k + 1) = r1.c * dk1;
                            const auto r2 = lartg(f, g);
                            emit_u(k - l, k, k + 1, r2.c, r2.s);
                            d(k) = r2.r;
                            const Real dk1b = d(k + 1), ekb = e(k);
                            f = r2.c * ekb + r2.s * dk1b;
                            d(k + 1) = r2.c * dk1b - r2.s * ekb;
                            if (k < m - 1) {
                                g = r2.s * e(k + 1);
                                e(k + 1) = r2.c * e(k + 1);
                            } else {
                                g = Real(0);
                            }
                            e(k) = f;
                        }
                        step = detail::kSweep;
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
                if (step == detail::kStop) break;
                // The block's rotations, in the leader's order, on every owned row / column.
                if (step == detail::kChaseU && acc_u) {
                    for_each_owned(ex, U.rows(), [&](int r) {
                        for (int j = lo + 1; j <= hi; ++j) detail::rot_pair<Real>(U(r, j), U(r, lo), ru[2 * (j - lo - 1)], ru[2 * (j - lo - 1) + 1]);
                    });
                } else if (step == detail::kChaseV && acc_vh) {
                    for_each_owned(ex, Vh.cols(), [&](int col) {
                        for (int j = hi - 1; j >= lo; --j)
                            detail::rot_pair<Real>(Vh(j, col), Vh(hi, col), rv[2 * (hi - 1 - j)], rv[2 * (hi - 1 - j) + 1]);
                    });
                } else if (step == detail::kSweep) {
                    if (acc_vh)
                        for_each_owned(ex, Vh.cols(), [&](int col) {
                            for (int k = lo; k < hi; ++k) detail::rot_pair<Real>(Vh(k, col), Vh(k + 1, col), rv[2 * (k - lo)], rv[2 * (k - lo) + 1]);
                        });
                    if (acc_u)
                        for_each_owned(ex, U.rows(), [&](int r) {
                            for (int k = lo; k < hi; ++k) detail::rot_pair<Real>(U(r, k), U(r, k + 1), ru[2 * (k - lo)], ru[2 * (k - lo) + 1]);
                        });
                }
                sync(ex, Mem::Global);
            } else if (step == detail::kStop) {
                break;
            }
        }
    }

    // s := |d|, a negative d(i) flipping row i of Vh; then the descending selection sort, its
    // swaps applied to U's columns and Vh's rows; info from the convergence and a NaN scan.
    if (ex.leader()) {
        for (int i = 0; i < n; ++i) {
            const bool neg = d(i) < Real(0);
            if (neg) {
                d(i) = -d(i);
                if constexpr (solo) {
                    if (acc_vh) for_each_owned(ex, Vh.cols(), [&](int col) { Vh(i, col) *= Real(-1); });
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
                    if (perm[i] != 0) Vh(i, col) *= Real(-1);
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
                    const Real tmp = s(i);
                    s(i) = s(best);
                    s(best) = tmp;
                    if constexpr (solo) {
                        if (acc_u) for_each_owned(ex, U.rows(), [&](int r) { detail::swap_at(U, r, i, r, best); });
                        if (acc_vh) for_each_owned(ex, Vh.cols(), [&](int col) { detail::swap_at(Vh, i, col, best, col); });
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
                        if (perm[i] != i) detail::swap_at(U, r, i, r, perm[i]);
                });
            if (acc_vh)
                for_each_owned(ex, Vh.cols(), [&](int col) {
                    for (int i = 0; i < n; ++i)
                        if (perm[i] != i) detail::swap_at(Vh, i, col, perm[i], col);
                });
        }
        sync(ex, Mem::Global);
    }
    return info;
}
}  // namespace detail

/// @brief xBDSQR inside a kernel: the singular values of one real upper bidiagonal B = (d, e),
/// accumulating the rotations into U and Vh.
///
/// `d` (n) and `e` (n - 1) are working storage, undefined on return; `s` receives the n singular
/// values (the absolute values of d after convergence, descending when `opts.sort_desc`). `U` is
/// accumulated (\f$U \leftarrow U Q\f$) when it has n columns and at least one row, `Vh`
/// (\f$V^H \leftarrow P^T V^H\f$) when it has n rows and at least one column: pass a 0 x 0
/// operand to skip either, or call the overload without them. To get B's own singular vectors,
/// seed U and Vh with the identity. Collective over `ex`, every item with the same arguments;
/// `d`, `e`, `U` and `Vh` visible to every item on entry, outputs visible on return.
/// @tparam Tier  `tier::Blocked<1>`, the default and only tier
/// @tparam Ex    a `Thread` (no workspace) or a work-group executor
/// @param[in]     ex    the executor
/// @param[in,out] d     the n diagonal entries of B; destroyed
/// @param[in,out] e     the n - 1 superdiagonal entries of B; destroyed
/// @param[out]    s     the n singular values
/// @param[in,out] U     the left accumulator, rows by n columns
/// @param[in,out] Vh    the right accumulator, n rows by columns
/// @param[in]     ws    on a work-group, `ws.local` and this bidiagonal's `ws.global` slice of
///                      `device::buffer_size<routine::bdsqr, Tier, T>(n)`; unused on a `Thread`
/// @param[in]     opts  sorting, tolerance and iteration cap
/// @return `info`, xBDSQR's status as the host's flag, the same on every item
/// @retval 0  converged
/// @retval 1  the sweep reached `maxit`, or a singular value is NaN
/// @ingroup api_device_lapack_svd
template <class Tier = tier::Blocked<1>, class Ex, class D, class E, class S, class MU, class MV>
BATCHLAS_DEVICE_INLINE int bdsqr(const Ex& ex, const D& d, const E& e, const S& s, const MU& U, const MV& Vh, const Workspace& ws,
                              const BdsqrOptions<elem_t<D>>& opts = {}) {
    using R = elem_t<D>;
    const auto dflt = detail::params<R>(d.size());
    const detail::bdsqr_params<R> p{opts.tol > R(0) ? opts.tol : dflt.tol, opts.maxit > 0 ? opts.maxit : dflt.maxit};
    return detail::bdsqr_sweep<Tier>(ex, d, e, s, U, Vh, ws, opts.sort_desc, p);
}
/// @brief xBDSQR inside a kernel, singular values only: `bdsqr` with no U and Vh.
///
/// The same contract as the overload with U and Vh.
/// @param[in]     ex    the executor
/// @param[in,out] d     the n diagonal entries of B; destroyed
/// @param[in,out] e     the n - 1 superdiagonal entries of B; destroyed
/// @param[out]    s     the n singular values
/// @param[in]     ws    as for the overload with U and Vh
/// @param[in]     opts  sorting, tolerance and iteration cap
/// @return `info`: 0 converged, 1 the sweep reached `maxit` or a singular value is NaN
/// @ingroup api_device_lapack_svd
template <class Tier = tier::Blocked<1>, class Ex, class D, class E, class S>
BATCHLAS_DEVICE_INLINE int bdsqr(const Ex& ex, const D& d, const E& e, const S& s, const Workspace& ws,
                              const BdsqrOptions<elem_t<D>>& opts = {}) {
    using R = elem_t<D>;
    const auto none = device::mat(static_cast<R*>(nullptr), 0, 0, 1);
    return bdsqr<Tier>(ex, d, e, s, none, none, ws, opts);
}

}  // namespace batchlas::device
