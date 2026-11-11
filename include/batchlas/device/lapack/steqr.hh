// SPDX-License-Identifier: MIT
#pragma once
// xSTEQR on a tridiagonal (d, e), e(i) coupling d(i) and d(i+1): steqr<J, tier::Cta<N>>, one
// problem per SgpPartition<N> (lane i holds d(i), e(i) and row i of Q; Q in local memory).
// detail: the split test and bulge-chase similarity, the CTA solver (nested / flat /
// lockstep over one implicit QL step) the host steqr_cta and syev_cta_fused kernels share, and
// the forms core does not reproduce bit for bit yet (serial dlanst 'M', LaneVec scal, the
// two-window reversal, the strict-tie 2x2 shift).
// evidence: docs/perf/steqr.md#cta-steqr-chase-micro-structure-decisions

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <batchlas/blas/extensions.hh>  // SteqrShiftStrategy, SteqrUpdateScheme
#include <batchlas/device.hh>
#include <batchlas/device/sg_partition.hh>
#include <batchlas/device/detail/sg_partition/sg_partition.hh>

namespace batchlas::device {

namespace detail {

// dsteqr's split tolerances: e(i) is negligible when |e(i)|^2 <= eps2 |d(i)| |d(i+1)| + safmin.
template <class R> struct SplitTol {
    R eps2;
    R safmin;
};

// max(|d(i)|, |e(i)|), dlanst 'M', serial on one item. fmax ignores a NaN entry (steqr's rule).
template <class D, class E> BATCHLAS_DEVICE_INLINE auto lanst_max(const Thread&, const D& d, const E& e) {
    using R = elem_t<D>;
    static_assert(!Arith<R>::is_complex, "steqr_detail::lanst_max: d and e are real");
    const int n = d.size();
    R acc = sycl::fabs(d(n - 1));
    for (int i = 0; i < n - 1; ++i) acc = sycl::fmax(acc, sycl::fmax(sycl::fabs(d(i)), sycl::fabs(e(i))));
    return acc;
}

// x := a x on the window [x.lo, x.hi) only.
template <class Ex, class T> BATCHLAS_DEVICE_INLINE void scal_lane(const Ex& ex, const T& a, const LaneVec<T>& x) {
    if (x.owns(ex)) x.v *= a;
}

// e(j) := 0 for each negligible j < e.size(), serial; true when one was (or already was) zero.
template <class D, class E, class R> BATCHLAS_DEVICE_INLINE bool tridiag_split(const Thread& th, const D& d, const E& e, SplitTol<R> tol) {
    bool any = false;
    for_each_owned(th, e.size(), [&](int j) {
        if (sycl::fabs(e(j)) * sycl::fabs(e(j)) <= tol.eps2 * sycl::fabs(d(j)) * sycl::fabs(d(j + 1)) + tol.safmin) {
            e(j) = R(0);
            any = true;
        }
    });
    return any;
}

// Lane-register form: d is this item's d(lid); e's window must end before the last valid d
// (d(lid + 1) is read from the next item). An e already zero is left untouched.
template <class Ex, class T> BATCHLAS_DEVICE_INLINE void tridiag_split(const Ex& ex, const T& d, const LaneVec<T>& e, SplitTol<T> tol) {
    const T d_next = ex.shift(d, 1);  // every item shuffles; the last one's value is unused
    if (e.owns(ex)) {
        if (e.v != T(0)) {
            const T rhs = tol.eps2 * sycl::fabs(d) * sycl::fabs(d_next) + tol.safmin;
            if (sycl::fabs(e.v) * sycl::fabs(e.v) <= rhs) e.v = T(0);
        }
    }
}

// Both windows reversed when `on`: the two sources first, then the two shuffles, so the
// shuffles share one convergence check (the T := J T J of a tridiagonal block, d and e).
template <class Ex, class T, class U> BATCHLAS_DEVICE_INLINE void reverse(const Ex& ex, const LaneVec<T>& x, const LaneVec<U>& y, bool on) {
    const int l = ex.lid();
    const int src_x = (on && l >= x.lo && l < x.hi) ? (x.lo + x.hi - 1 - l) : l;
    const int src_y = (on && l >= y.lo && l < y.hi) ? (y.lo + y.hi - 1 - l) : l;
    x.v = ex.shuffle(static_cast<std::remove_cv_t<T>>(x.v), src_x);
    y.v = ex.shuffle(static_cast<std::remove_cv_t<U>>(y.v), src_y);
}

// Memory form, serial: x(lo..hi-1) := x(hi-1..lo).
template <class X> BATCHLAS_DEVICE_INLINE void reverse(const Thread&, int lo, int hi, const X& x) {
    for (int a = lo, b = hi - 1; a < b; ++a, --b) {
        const auto t = x(a);
        x(a) = x(b);
        x(b) = t;
    }
}

// v(idx) in a sweep's virtual indexing: v(n - Off - idx) when !forward (QL). Plain inline: an
// always_inline form moved the Steqr*Francis SASS; kernels call this, not a local lambda.
template <class T, int Off, class V, class N> T virt_get(const V& v, bool forward, N n, std::size_t idx) {
    return forward ? v(idx) : v(n - Off - idx);
}

// One bulge-chase step on (d, e): T := G^T T G on rows/cols (i, i+1), prev_bulge the fill at
// (i-1, i+1); returns the fill at (i, i+2). forward == false indexes bottom-up (QL).
template <class D, class E, class T, class I>
BATCHLAS_DEVICE_INLINE T tridiag_rot(const Thread&, const D& d, const E& e, const T& prev_bulge, I i, I j, Givens<T> g, bool forward) {
    const T c = g.c;
    const T s = g.s;
    const I n = d.size();
    const I ne = e.size();
    auto d_get = [&](I k) -> T { return forward ? d(k) : d(n - 1 - k); };
    auto d_set = [&](I k, T val) {
        if (forward) d(k) = val;
        else d(n - 1 - k) = val;
    };
    auto e_get = [&](I k) -> T { return forward ? e(k) : e(ne - 1 - k); };
    auto e_set = [&](I k, T val) {
        if (forward) e(k) = val;
        else e(ne - 1 - k) = val;
    };

    const T di = d_get(i);
    const T dj = d_get(j);
    const T ei = e_get(i);
    const T ej = (j < ne) ? e_get(j) : T(0);

    // Explicit fma in the shape -ffp-contract=on gives (left product fused), so the x86 SIMD
    // paths cannot contract differently from NVPTX: c(c di - ei s) - s(ei c - s dj), etc.
    d_set(i, sycl::fma(c, sycl::fma(c, di, -(ei * s)), -(s * sycl::fma(ei, c, -(s * dj)))));
    d_set(j, sycl::fma(c, sycl::fma(c, dj, ei * s), s * sycl::fma(ei, c, s * di)));
    if (i > 0) e_set(i - 1, sycl::fma(e_get(i - 1), c, -(prev_bulge * s)));
    e_set(i, sycl::fma(c, sycl::fma(c, ei, s * di), -(s * sycl::fma(c, dj, s * ei))));
    if (j < ne) e_set(j, c * ej);
    return -ej * s;
}

// ssfmin / ssfmax / eps2 and block_scale: scalar.hh.

// The eigenvalue of [[a, b], [b, c]] closest to c, a tie going to the second root (the CTA
// Wilkinson strategy; core eig2x2_nearest ties to the first). Plain inline, as the form it replaced.
template <class R> inline R eig2x2_closest(const R& a, const R& b, const R& c) {
    const auto [l1, l2] = lae2(a, b, c);
    return std::abs(l1 - c) < std::abs(l2 - c) ? l1 : l2;
}

// ---- the work-group solver's sweep (host steqr_wg and steqr<J, tier::Blocked<1>>) ---------------
// Core lae2 / laev2 with the operands bound by reference: by value costs the wg Francis kernel 3
// LDG and 4 REG on sm_89 (src/math-helpers.hh keeps the same binding).
template <class R> BATCHLAS_DEVICE_INLINE auto lae2_cref(const R& a, const R& b, const R& c) { return lae2(a, b, c); }
template <class R> BATCHLAS_DEVICE_INLINE auto laev2_cref(const R& a, const R& b, const R& c) { return laev2(a, b, c); }

// {c, -s} of lartg(a, b): the work-group chase's rotation convention.
template <class R> inline std::array<R, 2> wg_givens(const R& a, const R& b) {
    const auto [c_l, s_l, r] = lartg(a, b);
    (void)r;
    return std::array<R, 2>{c_l, -s_l};
}

// An unreduced 2 x 2 block (n_local == 2) in closed form, e(0) := 0. With vectors, rot(g) takes
// the rotation {c, -s} of laev2's eigenvector (applied as rotation 0, forward order).
template <class R, class D, class E, class Rot> BATCHLAS_DEVICE_INLINE void wg_solve_2x2(const D& d_, const E& e_, bool vecs, Rot&& rot) {
    if (vecs) {
        auto [rt1, rt2, c, s] = laev2_cref(d_(0), e_(0), d_(1));
        d_(0) = rt1;
        d_(1) = rt2;
        e_(0) = R(0);
        rot(Givens<R>{c, -s});
    } else {
        auto [rt1, rt2] = lae2_cref(d_(0), e_(0), d_(1));
        d_(0) = rt1;
        d_(1) = rt2;
        e_(0) = R(0);
    }
}

// The sweep converges the end the shift is taken from (virtual index n_local - 1), so orient
// that end at the SMALLER |d|, as dsteqr does: true = QR (forward), false = QL.
template <class D, class N> BATCHLAS_DEVICE_INLINE bool wg_forward(const D& d_, N n_local) { return std::abs(d_(n_local - 1)) < std::abs(d_(0)); }

// One implicit QR (QL when !QR) sweep of an unreduced block (d_, e_), n_local >= 3: scale into
// [ssfmin, ssfmax], Wilkinson shift from the far end, chase the bulge with lartg rotations
// (rot(j, g) receives rotation j), the split test, unscale. True when an e(j) became negligible.
template <class R, class D, class E, class N, class Rot>
BATCHLAS_DEVICE_INLINE bool wg_sweep(const D& d_, const E& e_, N n_local, bool QR, Rot&& rot) {
    const Thread th{};
    const SplitTol<R> tol{eps2<R>(), safmin<R>()};
    const R anorm = lanst_max(th, d_, e_);
    if (anorm > ssfmax<R>()) {
        const R s = ssfmax<R>() / anorm;
        scal(th, s, d_);
        scal(th, s, e_);
    } else if (anorm < ssfmin<R>() && anorm != R(0)) {
        const R s = ssfmin<R>() / anorm;
        scal(th, s, d_);
        scal(th, s, e_);
    }

    const auto shift = eig2x2_closest(virt_get<R, 1>(d_, QR, n_local, n_local - 2), virt_get<R, 2>(e_, QR, n_local, n_local - 2),
                                      virt_get<R, 1>(d_, QR, n_local, n_local - 1));

    auto [c0, s0] = wg_givens(virt_get<R, 1>(d_, QR, n_local, 0) - shift, virt_get<R, 2>(e_, QR, n_local, 0));
    rot(std::size_t{0}, Givens<R>{c0, s0});
    auto bulge = tridiag_rot(th, d_, e_, R(0), std::size_t{0}, std::size_t{1}, Givens<R>{c0, s0}, QR);
    for (std::size_t j = 1; j < n_local - 1; ++j) {
        auto [cj, sj] = wg_givens(virt_get<R, 2>(e_, QR, n_local, j - 1), bulge);
        rot(j, Givens<R>{cj, sj});
        bulge = tridiag_rot(th, d_, e_, bulge, j, j + 1, Givens<R>{cj, sj}, QR);
    }

    const bool deflatable = tridiag_split(th, d_, e_, tol);

    if (anorm > ssfmax<R>()) {
        const R s = anorm / ssfmax<R>();
        scal(th, s, d_);
        scal(th, s, e_);
    } else if (anorm < ssfmin<R>() && anorm != R(0)) {
        const R s = anorm / ssfmin<R>();
        scal(th, s, d_);
        scal(th, s, e_);
    }
    return deflatable;
}

// ---- the CTA solver --------------------------------------------------------------------------
// Compile-time selectable shared-memory cache for Q, column-major:
// Q_local[base_q + row + col*LDQ] = Q(row, col). LDQ == P suits row-indexed use;
// a consumer that reads the tile by column (fused SYEV) needs LDQ == P+1 (banks).
template <typename T, size_t P, size_t LDQ, bool ComputeVecs, typename LocalAcc>
struct QSharedCache;

template <typename T, size_t P, size_t LDQ, typename LocalAcc>
struct QSharedCache<T, P, LDQ, true, LocalAcc> {
    LocalAcc Q_local;
    int32_t base_q;
    int32_t lane;
    int32_t n;
    int32_t idx{};
    T carry{};

    QSharedCache(LocalAcc q, int32_t bq, int32_t ln, int32_t n_)
        : Q_local(q), base_q(bq), lane(ln), n(n_) {}

    template <typename QProb>
    inline void load(const QProb& Q_prob) {
        const int32_t pN = static_cast<int32_t>(LDQ);
        // Zero, not skip, the padding rows (lane >= n): the chase then runs unguarded
        // on all P lanes, and `store` reads back only the first n rows.
        if (lane < n) {
            for (int32_t c = 0; c < n; ++c) {
                Q_local[base_q + lane + c * pN] = Q_prob(lane, c);
            }
        } else {
            for (int32_t c = 0; c < n; ++c) {
                Q_local[base_q + lane + c * pN] = T(0);
            }
        }
    }

    template <typename QProb>
    inline void store(QProb& Q_prob) const {
        if (lane >= n) return;
        const int32_t pN = static_cast<int32_t>(LDQ);
        for (int32_t c = 0; c < n; ++c) {
            Q_prob(lane, c) = Q_local[base_q + lane + c * pN];
        }
    }

    inline void apply(int32_t col0, int32_t col1, T c, T s) {
        const int32_t pN = static_cast<int32_t>(LDQ);
        device::rot<device::alg::RotTransposed, device::arith::Explicit, T, T, T>(device::Givens<T>{c, s}, Q_local[base_q + lane + col0 * pN],
                        Q_local[base_q + lane + col1 * pN]);
    }

    // Streaming form of `apply` for a bulge chase: successive rotations share a
    // column, so it stays in a register, and the element index moves by +/-P, so it
    // advances by a compile-time constant (partner via the immediate offset).
    // evidence: docs/perf/steqr.md#cta-steqr-chase-micro-structure-decisions
    inline void chase_begin(int32_t col) {
        idx = base_q + lane + col * static_cast<int32_t>(LDQ);
        carry = Q_local[idx];
    }

    // Dir is +1 when the chase walks toward higher column indices and -1 when it
    // walks toward lower ones; the written column is the current one and the read
    // column is the neighbour in that direction.
    template <int32_t Dir>
    inline void chase_step(T c, T s) {
        const int32_t next = idx + Dir * static_cast<int32_t>(LDQ);
        T q0 = carry;
        T q1 = Q_local[next];
        device::rot<device::alg::RotTransposed>(device::Givens<T>{c, s}, q0, q1);
        Q_local[idx] = q0;
        carry = q1;
        idx = next;
    }

    // Padded form for a chunk that idles through a lockstep chase: an idle
    // step reads its own current word and writes nothing, so the index never
    // leaves the lane's own row, even in a tile shared with other chunks.
    template <int32_t Dir>
    inline void chase_step_masked(T c, T s, bool act) {
        const int32_t next = act ? idx + Dir * static_cast<int32_t>(LDQ) : idx;
        T q0 = carry;
        T q1 = Q_local[next];
        device::rot<device::alg::RotTransposed>(device::Givens<T>{c, s}, q0, q1);
        if (act) Q_local[idx] = q0;
        carry = act ? q1 : carry;
        idx = next;
    }

    inline void chase_end(int32_t) {
        Q_local[idx] = carry;
    }

    inline void apply_if(bool pred, int32_t col0, int32_t col1, T c, T s) {
        if (pred) apply(col0, col1, c, s);
    }

    // Q := Q*J on columns [bb, be]; lane-private (own row), so no barrier.
    inline void reverse_columns(int32_t bb, int32_t be, bool rev) {
        if (!rev) return;
        const int32_t pN = static_cast<int32_t>(LDQ);
        device::detail::reverse(device::Thread{}, bb, be + 1,
                          [&](int32_t col) -> T& { return Q_local[base_q + lane + col * pN]; });
    }
};

template <typename T, size_t P, size_t LDQ, typename LocalAcc>
struct QSharedCache<T, P, LDQ, false, LocalAcc> {
    QSharedCache(LocalAcc, int32_t, int32_t, int32_t) {}

    template <typename QProb>
    inline void load(const QProb&) {}

    template <typename QProb>
    inline void store(QProb&) const {}

    inline void apply(int32_t, int32_t, T, T) {}
    inline void chase_begin(int32_t) {}
    template <int32_t Dir>
    inline void chase_step(T, T) {}
    template <int32_t Dir>
    inline void chase_step_masked(T, T, bool) {}
    inline void chase_end(int32_t) {}
    inline void apply_if(bool, int32_t, int32_t, T, T) {}
    inline void reverse_columns(int32_t, int32_t, bool) {}
};

// dsteqr's relative split test (no absolute threshold).
// evidence: docs/algorithms/steqr.md#steqr-the-relative-deflation-criterion
template <typename T>
inline SplitTol<T> steqr_split_tol() {
    return {eps2<T>(), safmin<T>()};
}

// T := J*T*J and Q := Q*J on [bb, be] when `rev` (J the reversal): a QR sweep is a QL
// sweep on J*T*J. d mirrors about bb+be, e about bb+be-1; e(bb-1) and e(be) stay.
// evidence: docs/algorithms/steqr.md#steqr-choosing-qr-versus-ql
template <size_t P, typename T, typename Partition, typename QCache>
inline void reverse_block(const Partition& partition,
                          T& diag,
                          T& offdiag,
                          QCache& qcache,
                          int32_t bb,
                          int32_t be,
                          bool rev) {
    const auto ex = device::wrap(partition);
    detail::reverse(ex, device::lane_vec(diag, bb, be + 1), device::lane_vec(offdiag, bb, be), rev);
    qcache.reverse_columns(bb, be, rev);
}

template <typename T, size_t P, typename Partition, typename QCache>
inline void solve_2x2_and_update(Partition partition,
                                 T& diag,
                                 T& offdiag,
                                 int32_t l0,
                                 QCache& qcache,
                                 bool pred = true) {
    // `pred` false: shuffles still run (l0 must be in [0, P-2]), nothing is written.
    const T a = select_from_group(partition, diag, l0);
    const T b = select_from_group(partition, offdiag, l0);
    const T c2 = select_from_group(partition, diag, l0 + 1);

    // Every input is already partition-uniform, so evaluate laev2 redundantly on
    // all lanes instead of computing on the leader and broadcasting four values.
    const auto [rt1, rt2, cs, sn] = device::laev2(a, b, c2);
    const int32_t lane = static_cast<int32_t>(partition.get_local_linear_id());
    if (pred && lane == l0) {
        diag = rt1;
        offdiag = T(0);
    }
    if (pred && lane == (l0 + 1)) {
        diag = rt2;
    }

    // QL eigenvector update: apply (cs, sn) on columns (l0+1, l0).
    qcache.apply_if(pred, l0 + 1, l0, cs, sn);
}

// Pad: the chunks of the lockstep domain share one trip count, and a chunk
// with `active` false or a shorter chase idles through identity rotations.
// Idle state is never multiplied by a mask, only discarded by selects, so a
// failed chunk's NaNs cannot reach d, e or Q. EXP only.
template <typename T, size_t P, bool Pad = false, typename Partition, typename QCache>
inline void implicit_ql_step(const Partition& partition,
                             T& diag,
                             T& offdiag,
                             QCache& qcache,
                             int32_t n,
                             int32_t l,
                             int32_t m,
                             SteqrShiftStrategy shift_strategy,
                             SteqrUpdateScheme update_scheme,
                             bool active = true) {
    const int32_t lane = static_cast<int32_t>(partition.get_local_linear_id());
    // An idle chunk's l and m are stale: clamp every shuffle source into the chunk.
    const int32_t ls = Pad ? sycl::clamp(l, int32_t(0), static_cast<int32_t>(P) - 2) : l;
    const int32_t ms = Pad ? sycl::clamp(m, int32_t(0), static_cast<int32_t>(P) - 1) : m;

    // EXP scheme: QL as a QR-style bulge chase in *virtual reversed* indexing inside
    // [l..m]. The only chase: QR sweeps run it on a mirrored block (reverse_block).
    const auto explicit_ql_step_exp = [&]() {
        // Preload shift inputs.
        T p0  = select_from_group(partition, diag, ls);
        T e0  = select_from_group(partition, offdiag, ls);
        T dlp1 = select_from_group(partition, diag, ls + 1);
        if constexpr (Pad) {
            // A benign 2x2 keeps an idle chunk's shift finite (e0 == 0 divides by 0).
            p0 = active ? p0 : T(0);
            e0 = active ? e0 : T(1);
            dlp1 = active ? dlp1 : T(0);
        }

        // Partition-uniform scalar state for the virtual QR bulge chase.
        // Every lane evaluates it redundantly: the inputs are broadcast values, so
        // the results agree bit-for-bit and no result broadcast is needed.
        T mu = T(0);

        if (shift_strategy == SteqrShiftStrategy::Wilkinson) {
            // For QL, the shift is formed from the leading 2x2 of the physical block (l,l+1).
            // wilkinson_shift picks the eigenvalue closest to its 3rd argument; we want closest to D(l).
            mu = eig2x2_closest(dlp1, e0, p0);
        } else {
            mu = device::wilkinson_shift(p0, e0, dlp1);
        }

        const int32_t nrot = Pad ? lockstep_max(partition, active ? m - l : 0) : m - l;
        // Virtual index v in [0..m-l-1] maps to physical indices:
        //   d_v(v)   = d( m - v )
        //   d_v(v+1) = d( m - v - 1 )
        //   e_v(v)   = e( m - v - 1 )  (couples the two diags above)
        //   e_v(v+1) = e( m - v - 2 )
        // Iteration v+1's (di, ei) is the (dj_new, ej_new) this one produced, so it
        // is carried in registers. Shuffles run UNCONDITIONALLY with a clamped source:
        // a non-uniform condition costs a MATCH/VOTE/BRA.DIV wrapper and lockstep.
        T di = select_from_group(partition, diag, ms);
        T ei = select_from_group(partition, offdiag, std::max(ms - 1, 0));
        ei = (ms >= 1) ? ei : T(0);
        T e_own = T(0);

        // Snapshot the tridiagonal before the chase. The chase writes lane `hi` but only
        // reads lanes below it, so broadcasts see pre-chase values; immutable
        // snapshots let the compiler hoist the shuffles off the lartg dependency path.
        // evidence: docs/perf/steqr.md#cta-steqr-chase-micro-structure-decisions
        const T diag_snap = diag;
        const T offdiag_snap = offdiag;

        // Seed the running (eprev, bulge) pair with (d(m) - mu, e(m-1)) so the first
        // rotation needs no special case in the hottest loop.
        T eprev = di - mu;
        T bulge = ei;

        // e(hi) is written only for v > 0 (the bulge has to have moved past it) and
        // only when hi indexes a real offdiagonal.  For v > 0 <=> hi < m, so both
        // conditions collapse into a single compare against this limit.
        const int32_t e_hi_limit = std::min(m, n - 1);

        qcache.chase_begin(ms);

        for (int32_t v = 0; v < nrot; ++v) {
            const int32_t hi = m - v;
            const int32_t lo = m - v - 1;
            const bool act = !Pad || (active && v < m - l);

            const T dj = select_from_group(partition, diag_snap, Pad ? std::max(lo, 0) : lo);

            // Next virtual offdiag (toward physical l). It is safe to read outside the block because
            // deflation boundaries force those couplings to zero.
            const T ej_raw = select_from_group(partition, offdiag_snap, std::max(lo - 1, 0));
            const T ej = (lo >= 1) ? ej_raw : T(0);

            const auto upd = [&]() {
                // Idle: (1, 0) is the exact identity rotation c = 1, s = 0.
                const T x = act ? eprev : T(1);
                const T y = act ? bulge : T(0);

                const auto [c1, s1, r1] = device::lartg<device::alg::LartgFast>(x, y);
                const T sigma = -s1; // match steqr.cc / saved-rotation convention

                // Update the offdiagonal to the *higher* physical index (virtual e(v-1) -> physical e(hi)).
                // This corresponds to LAPACK's QL inner-loop assignment E(i+1)=r.
                const T e_hi_new = x * c1 - y * sigma; // only meaningful when !first

                // Explicit similarity update (detail::tridiag_rot), virtual order.
                const T di_new = c1 * (c1 * di - ei * sigma) - sigma * (ei * c1 - sigma * dj);
                const T dj_new = c1 * (c1 * dj + ei * sigma) + sigma * (ei * c1 + sigma * di);
                const T ei_new = c1 * (c1 * ei + sigma * di) - sigma * (c1 * dj + sigma * ei);

                const T ej_new = c1 * ej;
                const T bulge_new = -ej * sigma;

                // Advance the (uniform) chase state.
                eprev = act ? ei_new : eprev;
                bulge = act ? bulge_new : bulge;

                // Return {c, sigma, di_new, dj_new, ei_new, ej_new, e_hi_new}
                return std::array<T, 7>{c1, sigma, di_new, dj_new, ei_new, ej_new, e_hi_new};
            }();

            const T c1 = upd[0];
            const T sigma = upd[1];

            // d(hi), e(hi) are final once the bulge passes; d(lo), e(lo), e(lo-1) are
            // recomputed next rotation, so they are carried and written once after
            // the chase. Selects, not `if`s: one target lane means an `if` diverges
            // on every rotation.
            const bool owns_hi = act && (lane == hi);
            diag = owns_hi ? upd[2] : diag;
            offdiag = (owns_hi && hi < e_hi_limit) ? upd[6] : offdiag;

            // QL eigenvector update: columns are reversed in physical ordering.
            // In your existing PG path you do apply(i+1, i, c, -s); here the physical pair is (hi, lo).
            if constexpr (Pad) {
                qcache.template chase_step_masked<-1>(c1, sigma, act);
            } else {
                qcache.template chase_step<-1>(c1, sigma);
            }

            // Carry the values the next iteration would otherwise re-shuffle:
            // d(lo) and e(lo-1) are exactly what was just written.
            di = act ? upd[3] : di;
            ei = act ? upd[5] : ei;
            e_own = act ? upd[4] : e_own;
        }

        // Flush the carried tail values (lo == l on the final iteration).
        if (active && lane == l) {
            diag = di;
            offdiag = e_own;
        }
        if (active && l >= 1 && lane == (l - 1)) {
            offdiag = ei;
        }

        if (active) qcache.chase_end(l);
    };

    if (Pad || update_scheme == SteqrUpdateScheme::EXP) {
        explicit_ql_step_exp();
        return;
    }

    // Broadcast values needed for the shift (all lanes participate).
    const T p0 = select_from_group(partition, diag, l);
    const T e0 = select_from_group(partition, offdiag, l);
    const T dlp1 = select_from_group(partition, diag, l + 1);
    const T dm = select_from_group(partition, diag, m);

    // Partition-uniform scalar state (evaluated redundantly on every lane).
    T g = T(0);
    T c = T(1);
    T s = T(1);
    T p = T(0);

    {
        T mu = T(0);
        if (shift_strategy == SteqrShiftStrategy::Wilkinson) {
            // Want eigenvalue closest to D(l); wilkinson_shift picks closest to its third arg.
            mu = eig2x2_closest(dlp1, e0, p0);
        } else {
            // LAPACK-style stable implicit shift.
            mu = device::wilkinson_shift(p0, e0, dlp1);
        }
        g = dm - mu;
    }

    qcache.chase_begin(m);

    for (int32_t i = m; i-- > l;) {
        // Broadcast the tridiagonal entries needed for this step.
        const T ei = select_from_group(partition, offdiag, i);
        const T di = select_from_group(partition, diag, i);
        const T dip1 = select_from_group(partition, diag, i + 1);

        // Whether E(i+1) should be updated is a pure function of (i, m, n).
        const bool do_e_upd = (i != (m - 1)) && ((i + 1) < (n - 1));

        const auto [c1b, s1b, d_ip1_new_b, r1_out_b] = [&]() {
            // {c1, s1, d_ip1_new, r1_out}
            const T f = s * ei;
            T rout = T(0);

            const auto [c1, s1, r1] = device::lartg<device::alg::LartgFast>(g, f);

            // In the original local-memory version: E(i+1) = r1 for i != m-1, when i+1 < N-1.
            if (do_e_upd) {
                rout = r1;
            }

            const T g2 = dip1 - p;
            const T r2 = (di - g2) * s1 + T(2) * c1 * (c * ei);
            p = s1 * r2;

            const T d_ip1_new = g2 + p;
            g = c1 * r2 - (c * ei);
            c = c1;
            s = s1;

            return std::array{c1, s1, d_ip1_new, rout};
        }();

        // Apply D/E updates directly to registers (predicated, not branched).
        const bool owns_ip1 = (lane == (i + 1));
        diag = owns_ip1 ? d_ip1_new_b : diag;
        offdiag = (owns_ip1 && do_e_upd) ? r1_out_b : offdiag;

        // QL uses reversed-column convention; keep the same sign as before: apply(i+1,i,c,-s).
        qcache.template chase_step<-1>(c1b, -s1b);
    }

    qcache.chase_end(l);

    // Final updates: D(l) = D(l) - p, and E(l) = g.
    const T d_l_new_b = p0 - p;
    const T e_l_new_b = g;
    if (lane == l) {
        diag = d_l_new_b;
        if (l < (n - 1)) {
            offdiag = e_l_new_b;
        }
    }
}

// One problem's STEQR outer loop: split at zero offdiagonals, then shifted QL sweeps
// per block (QR as QL on the mirrored block). Lane i owns d(i), e(i); `qcache`
// accumulates rotations (no-op without vectors). True if any block failed.
template <typename T, size_t P, typename Partition, typename QCache>
inline bool steqr_cta_solve_nested(const Partition& partition,
                                   T& diag,
                                   T& offdiag,
                                   QCache& qcache,
                                   int32_t n,
                                   int32_t max_sweeps,
                                   SteqrShiftStrategy cta_shift_strategy,
                                   SteqrUpdateScheme cta_update_scheme) {
    const auto ex = device::wrap(partition);
    const auto tol = steqr_split_tol<T>();
    const auto is_zero = [](T x) { return x == T(0); };
    const int32_t lane = static_cast<int32_t>(partition.get_local_linear_id());

    // Defensive budget against unbounded looping: one unit per implicit step.
    int32_t sweep_budget = max_sweeps * n;
    bool failed = false;

    for (int32_t next_block_begin = 0; next_block_begin < n;) {  // blocks split at E == 0
        const int32_t block_begin = next_block_begin;

        // Mark the split explicitly as LAPACK does: E(block_begin-1)=0.
        if (block_begin > 0 && lane == (block_begin - 1) && lane < (n - 1)) {
            offdiag = T(0);
        }

        // Split the tail, then the block ends at its first zero E (n-1 if none).
        const auto tail_e = device::lane_vec(offdiag, block_begin, n - 1);
        detail::tridiag_split(ex, diag, tail_e, tol);
        const int32_t block_end = device::find_first(ex, tail_e, n - 1, is_zero);
        next_block_begin = block_end + 1;
        if (block_end <= block_begin) {
            continue;  // size-0/1 block
        }

        // dlascl: bring the block norm into the safe range (overflow/underflow).
        const auto blk_d = device::lane_vec(diag, block_begin, block_end + 1);
        const auto blk_e = device::lane_vec(offdiag, block_begin, block_end);
        const T anorm = device::lanst<NormType::Max>(ex, blk_d, blk_e);

        T scale = T(1);
        if (anorm > ssfmax<T>()) {
            scale = ssfmax<T>() / anorm;
        } else if (anorm < ssfmin<T>() && anorm != T(0)) {
            scale = ssfmin<T>() / anorm;
        }
        const T inv_scale = T(1) / scale;

        if (scale != T(1)) {
            detail::scal_lane(ex, scale, blk_d);
            detail::scal_lane(ex, scale, blk_e);
        }

        // QL if |D(l)| <= |D(lend)|, QR otherwise (as dsteqr: the graded small end
        // converges first; the inverted rule loses relative accuracy). QR runs as QL
        // on the mirrored block so chunks share one loop nest; mirrored back even on
        // failure. evidence: docs/perf/steqr.md#cta-steqr-chase-micro-structure-decisions
        const T d_first = sycl::fabs(select_from_group(partition, diag, block_begin));
        const T d_last = sycl::fabs(select_from_group(partition, diag, block_end));
        const bool rev = d_last < d_first;
        reverse_block<P>(partition, diag, offdiag, qcache, block_begin, block_end, rev);

        // QL iteration: converge from the top (l grows).
        for (int32_t l = block_begin; l <= block_end && !failed;) {
            if (l == block_end) {
                l += 1;
                continue;
            }

            bool advanced = false;
            for (int32_t sweep = 0; sweep < max_sweeps; ++sweep) {
                // Split the active subproblem [l..lend]; m = its first zero E, else lend.
                const auto act_e = device::lane_vec(offdiag, l, block_end);
                detail::tridiag_split(ex, diag, act_e, tol);
                const int32_t m = device::find_first(ex, act_e, block_end, is_zero);

                if (m == l) {
                    l += 1;
                    advanced = true;
                    break;
                }

                if (m == l + 1) {
                    solve_2x2_and_update<T, P>(partition, diag, offdiag, l, qcache);
                    l += 2;
                    advanced = true;
                    break;
                }

                if (sweep_budget <= 0) {
                    failed = true;
                    break;
                }
                sweep_budget -= 1;

                implicit_ql_step<T, P>(partition, diag, offdiag, qcache, n, l, m, cta_shift_strategy, cta_update_scheme);
            }

            if (!advanced) {
                failed = true;
            }
        }

        reverse_block<P>(partition, diag, offdiag, qcache, block_begin, block_end, rev);

        // Rescale converged block back to the original magnitude.
        if (scale != T(1)) {
            detail::scal_lane(ex, inv_scale, blk_d);
            detail::scal_lane(ex, inv_scale, blk_e);
        }

        if (failed) break;
    }

    return failed;
}

// steqr_cta_solve_nested with the sweep hoisted out of the loop nest, for a partition
// whose collectives are chunk-local (a masked SubGroupPartition): each chunk settles
// until it has a sweep or is done, then every chunk with a sweep chases in the same
// pass. Same per-chunk operation sequence as the nested solver: bitwise equal.
// evidence: docs/perf/steqr.md#lockstep-flat-solver
template <typename T, size_t P, typename Partition, typename QCache>
inline bool steqr_cta_solve_flat(const Partition& partition,
                                 T& diag,
                                 T& offdiag,
                                 QCache& qcache,
                                 int32_t n,
                                 int32_t max_sweeps,
                                 SteqrShiftStrategy cta_shift_strategy,
                                 SteqrUpdateScheme cta_update_scheme) {
    const auto ex = device::wrap(partition);
    const auto tol = steqr_split_tol<T>();
    const auto is_zero = [](T x) { return x == T(0); };
    const int32_t lane = static_cast<int32_t>(partition.get_local_linear_id());
    int32_t sweep_budget = max_sweeps * n;
    int32_t next_block_begin = 0;
    int32_t block_begin = 0;
    int32_t block_end = 0;
    int32_t l = 0;
    int32_t m = 0;
    int32_t sweeps_at_l = 0;
    bool rev = false;
    T scale = T(1);
    T inv_scale = T(1);
    bool need_block = true;
    bool done = false;
    bool failed = false;

    while (!done) {
        // Settle: every exit of this loop is either "sweep [l, m] next" or done,
        // so the chunks of a warp reconverge right before the chase.
        for (;;) {
            if (need_block) {
                // ---- Open the next block (nested: top of the split loop). ----
                const int32_t bb = next_block_begin;
                if (bb > 0 && lane == (bb - 1) && lane < (n - 1)) {
                    offdiag = T(0);
                }
                const auto tail_e = device::lane_vec(offdiag, bb, n - 1);
                detail::tridiag_split(ex, diag, tail_e, tol);
                const int32_t be = device::find_first(ex, tail_e, n - 1, is_zero);
                next_block_begin = be + 1;
                if (be <= bb) {
                    if (next_block_begin >= n) {
                        done = true;
                        break;
                    }
                    continue;
                }

                const auto blk_d = device::lane_vec(diag, bb, be + 1);
                const auto blk_e = device::lane_vec(offdiag, bb, be);
                const T anorm = device::lanst<NormType::Max>(ex, blk_d, blk_e);
                scale = T(1);
                if (anorm > ssfmax<T>()) {
                    scale = ssfmax<T>() / anorm;
                } else if (anorm < ssfmin<T>() && anorm != T(0)) {
                    scale = ssfmin<T>() / anorm;
                }
                inv_scale = T(1) / scale;
                if (scale != T(1)) {
                    detail::scal_lane(ex, scale, blk_d);
                    detail::scal_lane(ex, scale, blk_e);
                }

                const T d_first = sycl::fabs(select_from_group(partition, diag, bb));
                const T d_last = sycl::fabs(select_from_group(partition, diag, be));
                rev = d_last < d_first;
                reverse_block<P>(partition, diag, offdiag, qcache, bb, be, rev);
                block_begin = bb;
                block_end = be;
                l = bb;
                sweeps_at_l = 0;
                need_block = false;
            }

            // ---- Test [l, block_end] (nested: sweep loop top). ----
            // The sweep cap comes first: the nested loop gives up after its last
            // sweep without deflating or searching for m again.
            bool closing = false;
            if (sweeps_at_l >= max_sweeps) {
                failed = true;
                closing = true;
            } else {
                const auto act_e = device::lane_vec(offdiag, l, block_end);
                detail::tridiag_split(ex, diag, act_e, tol);
                m = device::find_first(ex, act_e, block_end, is_zero);
                if (m == l) {
                    l += 1;
                } else if (m == l + 1) {
                    solve_2x2_and_update<T, P>(partition, diag, offdiag, l, qcache);
                    l += 2;
                } else if (sweep_budget <= 0) {
                    failed = true;
                } else {
                    break;
                }
                sweeps_at_l = 0;
                closing = failed || l >= block_end;
            }

            // ---- Close the block: mirror back and rescale, even on failure. ----
            if (closing) {
                reverse_block<P>(partition, diag, offdiag, qcache, block_begin, block_end, rev);
                if (scale != T(1)) {
                    detail::scal_lane(ex, inv_scale, device::lane_vec(diag, block_begin, block_end + 1));
                    detail::scal_lane(ex, inv_scale, device::lane_vec(offdiag, block_begin, block_end));
                }
                need_block = true;
                if (failed || next_block_begin >= n) {
                    done = true;
                    break;
                }
            }
        }
        if (done) break;

        implicit_ql_step<T, P>(partition, diag, offdiag, qcache, n, l, m,
                               cta_shift_strategy, cta_update_scheme);
        sweep_budget -= 1;
        sweeps_at_l += 1;
    }
    return failed;
}

// The same state machine when the lockstep domain is the whole sub-group: every branch
// guarding a collective runs under a sub-group vote with per-chunk gated updates, and
// a chunk outside a phase feeds in-range sources and discards the results.
template <typename T, size_t P, typename Partition, typename QCache>
inline bool steqr_cta_solve_lockstep(const Partition& partition,
                                     T& diag,
                                     T& offdiag,
                                     QCache& qcache,
                                     int32_t n,
                                     int32_t max_sweeps,
                                     SteqrShiftStrategy cta_shift_strategy,
                                     SteqrUpdateScheme cta_update_scheme) {
    const auto ex = device::wrap(partition);
    const auto tol = steqr_split_tol<T>();
    const int32_t lane = static_cast<int32_t>(partition.get_local_linear_id());

    int32_t sweep_budget = max_sweeps * n;
    int32_t next_block_begin = 0;
    int32_t block_begin = 0;
    int32_t block_end = 0;
    int32_t l = 0;
    int32_t m = 0;
    int32_t sweeps_at_l = 0;
    bool rev = false;
    T scale = T(1);
    T inv_scale = T(1);
    bool need_block = true;
    bool ready = false;  // a sweep on [l, m] is pending
    bool done = false;
    bool failed = false;

    while (lockstep_any(partition, !done)) {
        while (lockstep_any(partition, !done && !ready)) {
            // ---- Open the next block (nested: top of the split loop). ----
            const bool opening = !done && !ready && need_block;
            if (lockstep_any(partition, opening)) {
                const int32_t bb = opening ? next_block_begin : 0;
                if (opening && bb > 0 && lane == (bb - 1) && lane < (n - 1)) {
                    offdiag = T(0);
                }
                detail::tridiag_split(ex, diag, device::lane_vec(offdiag, opening ? bb : n, n - 1), tol);
                const int32_t be = device::find_first(ex, device::lane_vec(offdiag, bb, n - 1), n - 1,
                                                  [&](T x) { return opening && x == T(0); });
                const bool opened = opening && be > bb;

                const auto blk_d = device::lane_vec(diag, bb, opened ? be + 1 : bb);  // empty unless opened
                const auto blk_e = device::lane_vec(offdiag, bb, opened ? be : bb);
                const T anorm = device::lanst<NormType::Max>(ex, blk_d, blk_e);
                T sc = T(1);
                if (anorm > ssfmax<T>()) {
                    sc = ssfmax<T>() / anorm;
                } else if (anorm < ssfmin<T>() && anorm != T(0)) {
                    sc = ssfmin<T>() / anorm;
                }
                if (opened && sc != T(1)) {
                    detail::scal_lane(ex, sc, blk_d);
                    detail::scal_lane(ex, sc, blk_e);
                }

                const T d_first = sycl::fabs(select_from_group(partition, diag, bb));
                const T d_last = sycl::fabs(select_from_group(partition, diag, be));
                const bool rv = opened && (d_last < d_first);
                reverse_block<P>(partition, diag, offdiag, qcache, bb, be, rv);

                // A size-0/1 block is skipped: stay in `need_block` for the next one.
                next_block_begin = opening ? be + 1 : next_block_begin;
                if (opened) {
                    block_begin = bb;
                    block_end = be;
                    l = bb;
                    sweeps_at_l = 0;
                    rev = rv;
                    scale = sc;
                    inv_scale = T(1) / sc;
                    need_block = false;
                }
                done = done || (opening && !opened && be + 1 >= n);
            }

            // ---- Test the active subproblem [l, block_end] (nested: sweep loop top). ----
            const bool testing = !done && !ready && !need_block;
            // The sweep cap comes first: the nested loop gives up after its last
            // sweep without deflating or searching for m again.
            const bool cap = testing && sweeps_at_l >= max_sweeps;
            const bool active = testing && !cap;
            if (lockstep_any(partition, active)) {
                detail::tridiag_split(ex, diag, device::lane_vec(offdiag, active ? l : n, block_end), tol);
                const int32_t mm = device::find_first(ex, device::lane_vec(offdiag, l, block_end), block_end,
                                                  [&](T x) { return active && x == T(0); });
                const bool adv1 = active && mm == l;
                const bool adv2 = active && mm == l + 1;
                const bool want = active && !adv1 && !adv2;
                failed = failed || (want && sweep_budget <= 0);
                m = (want && sweep_budget > 0) ? mm : m;
                ready = ready || (want && sweep_budget > 0);

                if (lockstep_any(partition, adv2)) {
                    const int32_t l0 = sycl::clamp(l, int32_t(0), static_cast<int32_t>(P) - 2);
                    solve_2x2_and_update<T, P>(partition, diag, offdiag, l0, qcache, adv2);
                }
                l += adv1 ? 1 : (adv2 ? 2 : 0);
                sweeps_at_l = (adv1 || adv2) ? 0 : sweeps_at_l;
            }
            failed = failed || cap;

            // ---- Close the block: mirror back and rescale, even on failure. ----
            const bool closing = testing && (l >= block_end || failed);
            if (lockstep_any(partition, closing)) {
                reverse_block<P>(partition, diag, offdiag, qcache, block_begin, block_end, closing && rev);
                if (closing && scale != T(1)) {
                    detail::scal_lane(ex, inv_scale, device::lane_vec(diag, block_begin, block_end + 1));
                    detail::scal_lane(ex, inv_scale, device::lane_vec(offdiag, block_begin, block_end));
                }
                need_block = need_block || closing;
                done = done || (closing && (failed || next_block_begin >= n));
            }
        }

        // ---- Every chunk with a pending sweep chases in the same pass. ----
        if (lockstep_any(partition, ready)) {
            implicit_ql_step<T, P, true>(partition, diag, offdiag, qcache, n, l, m,
                                         cta_shift_strategy, cta_update_scheme, ready);
        }
        sweep_budget -= ready ? 1 : 0;
        sweeps_at_l += ready ? 1 : 0;
        ready = false;
    }

    return failed;
}

// Where the hoisted sweep pays for its bookkeeping. It does not at P == 32
// (one chunk per warp) or P == 4, whose 1-3 rotation sweeps are too short for
// realignment to beat the extra per-pass work; float P == 8 breaks even.
// evidence: docs/perf/steqr.md#lockstep-flat-solver
template <typename T, size_t P>
inline constexpr bool kSteqrCtaFlatPays = (P == 16) || (P == 8 && std::is_same_v<T, double>);

// Returns true if any block failed to converge within budget.
//
// A full-sub-group (emulated) partition promises every lane of the sub-group runs
// this. P == 32 runs the nested loops maskless; smaller chunks on NVPTX take the
// chunk-masked partition (every maskless form measured slower there); elsewhere EXP
// takes the lockstep solver, while PG's nested loops are legal only while the chunks
// happen not to diverge. Judge changes here by real-kernel A/B, not microbenchmarks.
// evidence: docs/perf/steqr.md#full-warp-partition
template <typename T, size_t P, typename Partition, typename QCache>
inline bool steqr_cta_solve(const Partition& partition,
                            T& diag,
                            T& offdiag,
                            QCache& qcache,
                            int32_t n,
                            int32_t max_sweeps,
                            SteqrShiftStrategy cta_shift_strategy,
                            SteqrUpdateScheme cta_update_scheme) {
    if constexpr (lockstep_spans_subgroup_v<Partition> && P < 32) {
        if constexpr (batchlas::sgp::kMaskedByDefault) {
            const auto chunk = make_partition<P, true>(partition.sg);
            return steqr_cta_solve<T, P>(chunk, diag, offdiag, qcache, n, max_sweeps,
                                         cta_shift_strategy, cta_update_scheme);
        } else if (cta_update_scheme == SteqrUpdateScheme::EXP) {
            return steqr_cta_solve_lockstep<T, P>(partition, diag, offdiag, qcache, n, max_sweeps,
                                                  cta_shift_strategy, cta_update_scheme);
        }
    } else if constexpr (kSteqrCtaFlatPays<T, P>) {
        return steqr_cta_solve_flat<T, P>(partition, diag, offdiag, qcache, n, max_sweeps,
                                          cta_shift_strategy, cta_update_scheme);
    }
    return steqr_cta_solve_nested<T, P>(partition, diag, offdiag, qcache, n, max_sweeps,
                                        cta_shift_strategy, cta_update_scheme);
}

// No report between the solve and the stores (the device op returns the flag instead).
struct NoReport {
    BATCHLAS_DEVICE_INLINE void operator()(bool) const {}
};

// The CTA tier's one body (the host steqr_cta kernel's too): load Q, d, e; solve; report(failed);
// store. live == false (a dead chunk, a masked-off item) solves zeros, so every lane reaches every
// collective, and writes nothing. True when a block did not converge within max_sweeps.
template <typename T, size_t P, bool ComputeVecs, typename Partition, typename D, typename E, typename Z, typename QCache,
          typename Report = NoReport>
BATCHLAS_DEVICE_INLINE bool steqr_cta_run(const Partition& partition, D d, E e, Z z, QCache& qcache, int32_t n,
                                       int32_t max_sweeps, SteqrShiftStrategy shift, SteqrUpdateScheme update, bool live,
                                       Report report = {}) {
    const int32_t lane = static_cast<int32_t>(partition.get_local_linear_id());
    if constexpr (ComputeVecs) {
        if (live) qcache.load(z);  // lane-private; a dead tile is never read
    }
    T diag = (live && lane < n) ? d(lane) : T(0);
    T offdiag = (live && lane < (n - 1)) ? e(lane) : T(0);
    const bool failed = steqr_cta_solve<T, P>(partition, diag, offdiag, qcache, n, max_sweeps, shift, update);
    report(failed);
    if (live && lane < n) d(lane) = diag;
    if (live && lane < (n - 1)) e(lane) = offdiag;
    if constexpr (ComputeVecs) {
        if (live) qcache.store(z);
    }
    return failed;
}

// The CTA tier's workspace: Q (N x N, ld N) in local memory with vectors, nothing otherwise.
template <bool Vecs, int N, class T> struct steqr_cta_plan {
    Plan plan{};
    WsSlot<T> q{};
    constexpr steqr_cta_plan() : q(plan.take<T>(WsSpace::Local, Vecs ? static_cast<std::size_t>(N) * N : 0)) {}
};

// ---- the Blocked tier: one problem per work-group, any n ------------------------------------------
// One unreduced block [lo, hi) of a pass: its orientation (qr: 1 forward), run (still sweeping)
// and rot (rotations produced in the current sweep, rotation j of it at rot[lo + j]).
struct WgBlock {
    std::int32_t lo, hi, qr, run, rot;
};


// The Blocked tier's workspace, all global: one sweep's rotations (n - 1, with vectors) and the
// block table (at most n / 2 + 1 blocks of length >= 2).
template <bool Vecs, class T> struct wg_plan {
    Plan plan{};
    WsSlot<Givens<T>> rot{};
    WsSlot<WgBlock> blk{};
    constexpr explicit wg_plan(int n)
        : rot(plan.take<Givens<T>>(WsSpace::Global, Vecs && n > 1 ? static_cast<std::size_t>(n - 1) : 0)),
          blk(plan.take<WgBlock>(WsSpace::Global, static_cast<std::size_t>(n / 2 + 1))) {}
};

// The host steqr_wg algorithm inside one work-group: n - 1 passes (none after one that finds no
// unreduced block: later passes would find none either); each pass lists the blocks split at
// e(j) == 0 (the leader, serially), then every block sweeps (wg_sweep / wg_solve_2x2, one item
// per block) until its first deflation or max_sweeps. With vectors the blocks sweep in lockstep:
// after each sweep every item applies that sweep's rotations to its rows of z (core lasr, the
// host rotation kernel's), so z sees each block's rotations in the host's order. Returns the
// number of e(j) != 0 left (LAPACK's info), uniform.
template <bool Vecs, int WG, class T, class D, class E, class Z>
BATCHLAS_DEVICE_INLINE int steqr_wg_run(const WorkGroup<WG>& ex, const D& d, const E& e, const Z& z, int n, int max_sweeps, const Workspace& ws) {
    const wg_plan<Vecs, T> plan(n);
    Givens<T>* const rot = carve_global(ws, plan.rot);
    WgBlock* const blk = carve_global(ws, plan.blk);
    const int lid = ex.lid();
    for (int pass = 0; pass < n - 1; ++pass) {
        int nblk = 0;
        if (ex.leader()) {
            int ix = 0;
            while (ix < n - 1) {
                const int start = ix;
                if (e(ix) != T(0)) {
                    for (ix = ix + 1; ix < n - 1; ++ix)
                        if (e(ix) == T(0)) break;
                } else {
                    ++ix;
                    continue;
                }
                blk[nblk++] = WgBlock{start, ix + 1, 1, 1, 0};
            }
        }
        nblk = ex.bcast(nblk, 0);
        ex.barrier(Mem::Global);
        if (nblk == 0) break;
        for (int s = lid; s < nblk; s += WG) {
            WgBlock& b = blk[s];
            const int len = b.hi - b.lo;
            if (len > 2) b.qr = wg_forward(d.tail(b.lo).head(len), len) ? 1 : 0;
        }
        if constexpr (!Vecs) {
            const auto noop = [](auto&&...) BATCHLAS_DEVICE_BODY {};
            for (int s = lid; s < nblk; s += WG) {
                const WgBlock b = blk[s];
                const int len = b.hi - b.lo;
                const auto d_ = d.tail(b.lo).head(len);
                const auto e_ = e.tail(b.lo).head(len - 1);
                if (len == 2) {
                    wg_solve_2x2<T>(d_, e_, false, noop);
                    continue;
                }
                for (int k = 0; k < max_sweeps; ++k)
                    if (wg_sweep<T>(d_, e_, len, b.qr != 0, noop)) break;
            }
            ex.barrier(Mem::Global);
        } else {
            for (int k = 0; k < max_sweeps; ++k) {
                int live = 0;
                for (int s = lid; s < nblk; s += WG) {
                    WgBlock& b = blk[s];
                    b.rot = 0;
                    if (!b.run) continue;
                    const int len = b.hi - b.lo;
                    const auto d_ = d.tail(b.lo).head(len);
                    const auto e_ = e.tail(b.lo).head(len - 1);
                    Givens<T>* const r = rot + b.lo;
                    if (len == 2) {
                        wg_solve_2x2<T>(d_, e_, true, [&](const Givens<T>& g) BATCHLAS_DEVICE_BODY { r[0] = g; });
                        b.run = 0;
                    } else if (wg_sweep<T>(d_, e_, len, b.qr != 0, [&](std::size_t j, const Givens<T>& g) BATCHLAS_DEVICE_BODY { r[j] = g; })) {
                        b.run = 0;
                    }
                    b.rot = 1;
                    live += b.run;
                }
                ex.barrier(Mem::Global);
                for (int s = 0; s < nblk; ++s) {
                    const WgBlock b = blk[s];
                    if (!b.rot) continue;
                    const int ncols = b.hi - b.lo;
                    const bool fwd = b.qr != 0;
                    for (int row = lid; row < z.rows(); row += WG) {
                        // Row `row` of z in the sweep's virtual column order (dlasr SIDE = R).
                        const auto q_row = [&](int v) -> decltype(auto) { return z(row, b.lo + (fwd ? v : ncols - 1 - v)); };
                        const auto rots = [&](int j) { return rot[b.lo + j]; };
                        lasr<alg::RotTransposed>(Thread{}, ncols, rots, q_row);
                    }
                }
                live = ex.group_reduce(live);
                ex.barrier(Mem::Global);
                if (live == 0) break;
            }
        }
    }
    int left = 0;
    for (int j = lid; j < n - 1; j += WG) left += e(j) != T(0) ? 1 : 0;
    return ex.group_reduce(left);
}

}  // namespace detail

namespace routine {
// xSTEQR, jobz J (EigenVectors: Q := Q * the rotations, LAPACK compz = 'V').
template <JobType J> struct steqr {
    static constexpr const char* name = "steqr";
};
}  // namespace routine

// The run-time knobs of the solve (the host SteqrParams fields of the same meaning). The Blocked
// tier reads max_sweeps only (steqr_wg's shift and update, as the host route).
struct steqr_opts {
    int max_sweeps = 50;
    SteqrShiftStrategy shift = SteqrShiftStrategy::Lapack;
    SteqrUpdateScheme update = SteqrUpdateScheme::EXP;
};

template <JobType J, int N, class Alg, class T> struct tier_info<routine::steqr<J>, tier::Cta<N, Alg>, T> {
    static_assert(N == 4 || N == 8 || N == 16 || N == 32, "steqr: tier::Cta<N> has N in {4, 8, 16, 32}");
    static_assert(!Arith<T>::is_complex, "steqr: d and e are real");
    static constexpr int wg_size = 32;  // the smallest launch: one sub-group, 32 / N problems
    static constexpr int sg_size = 32;
    static constexpr bool admits(int n) { return n >= 1 && n <= N; }
    static constexpr WorkspaceSize buffer_size(int) {
        return detail::steqr_cta_plan<J == JobType::EigenVectors, N, T>{}.plan.size();
    }
};
// tier::Blocked<1>: no panel (the chase is serial per block, z's rows parallel), any n.
template <JobType J, int NB, class Alg, class T> struct tier_info<routine::steqr<J>, tier::Blocked<NB, Alg>, T> {
    static_assert(NB == 1, "steqr: tier::Blocked<1> (the work-group solver has no panel width)");
    static_assert(!Arith<T>::is_complex, "steqr: d and e are real");
    static constexpr int wg_size = 256;  // one problem per work-group, the size sytrd's Blocked tier runs at
    static constexpr int sg_size = 32;
    static constexpr bool admits(int n) { return n >= 1; }
    static constexpr WorkspaceSize buffer_size(int n) { return detail::wg_plan<J == JobType::EigenVectors, T>(n).plan.size(); }
};
template <JobType J, class T> struct tiers<routine::steqr<J>, T> {
    using type = tier::list<tier::Cta<4>, tier::Cta<8>, tier::Cta<16>, tier::Cta<32>, tier::Blocked<1>>;
};

// Bytes of ONE problem's workspace (Cta: local Q with vectors; Blocked: global rotations and the
// block table).
template <JobType J, class Tier, class T> constexpr WorkspaceSize steqr_buffer_size(int n) {
    return tier_info<routine::steqr<J>, resolve_tier_t<routine::steqr<J>, T, Tier>, T>::buffer_size(n);
}

// xSTEQR on one problem: d (n = d.size()), e (n - 1) and, for EigenVectors, z (n x n, the matrix
// the rotations accumulate into: the identity, or sytrd's Q) are overwritten; eigenvalues
// UNSORTED in d (the host route sorts in a separate pass).
//   tier::Cta<N>      n <= N, one problem per SgpPartition<N>, COLLECTIVE over the whole sub-group
//                     (every lane calls with its partition's problem); ws.local. Returns 1 when a
//                     block did not converge within opts.max_sweeps.
//   tier::Blocked<1>  any n, one problem per WorkGroup<256>, COLLECTIVE; the host steqr_wg
//                     algorithm (same sweep body, same pass order); ws.global. Returns the number
//                     of e(j) != 0 left (LAPACK's info). z may be complex (real rotations).
// Returns 0 on success (uniform); -2 when n is outside [1, N].
template <JobType J, class Tier, class Ex, class D, class E, class Z>
BATCHLAS_DEVICE_INLINE int steqr(const Ex& ex, const D& d, const E& e, const Z& z, const Workspace& ws,
                              const steqr_opts& opts = {}) {
    using T = std::remove_cv_t<elem_t<D>>;
    using R = routine::steqr<J>;
    using Tr = resolve_tier_t<R, T, Tier>;
    static_assert(HasTier<R, Tr, T>, "steqr: the routine has no such tier (tier::Cta<N>, tier::Blocked<1>)");
    constexpr bool kVecs = J == JobType::EigenVectors;
    if constexpr (tier::is_blocked_v<Tr>) {
        constexpr int WG = tier_info<R, Tr, T>::wg_size;
        static_assert(std::is_same_v<Ex, WorkGroup<WG>>, "steqr: tier::Blocked<1> runs on device::WorkGroup<256> (one problem per work-group)");
        const int n = d.size();
        if (n < 1) return -2;
        return detail::steqr_wg_run<kVecs, WG, T>(ex, d, e, z, n, opts.max_sweeps, ws);
    } else {
        constexpr int N = Tr::n_max;
        constexpr std::size_t P = N;
        static_assert(std::is_same_v<Ex, SgpPartition<N, false>> || std::is_same_v<Ex, SgpPartition<N, true>>,
                      "steqr: tier::Cta<N> runs on device::SgpPartition<N> (device::wrap(make_partition<N>(sg)))");
        const int n = d.size();
        if (n < 1 || n > N) return -2;
        using LP = typename local_mptr<T>::pointer;  // decorated: a generic pointer costs generic LD/ST
        const detail::steqr_cta_plan<kVecs, N, T> plan{};
        detail::QSharedCache<T, P, P, kVecs, LP> qcache(carve_local(ws, plan.q).get(), 0, ex.lid(), n);
        const bool failed = detail::steqr_cta_run<T, P, kVecs>(ex.part, d, e, z, qcache, n, opts.max_sweeps,
                                                                     opts.shift, opts.update, true);
        return failed ? 1 : 0;
    }
}

}  // namespace batchlas::device
