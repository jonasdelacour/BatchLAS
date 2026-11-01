// SPDX-License-Identifier: MIT
#pragma once

// The execution layer of CTA steqr: lane i of a P-lane chunk owns d_i and e_i in registers,
// every phase is a function of that view, and the drivers in steqr_cta_device.hh only
// sequence the phases. Collectives run on every lane with clamped sources; a one-lane write
// is a select; a phase that a chunk sits out takes `live` false and discards its results.
// evidence: docs/perf/steqr.md#cta-steqr-chase-micro-structure-decisions

#include <batchlas/blas/extensions.hh>
#include <src/kernels/common/sg_partition/sg_partition.hh>
#include <src/ops/syev/steqr/ql_chase.hh>
#include <src/ops/syev/tridiag/plane_rotation.hh>
#include <src/ops/syev/tridiag/tridiag_math.hh>
#include <src/sycl/kernel_attrs.hh>

#include <algorithm>
#include <cstdint>

namespace batchlas::tridiag {

// The closed index range [first, last] of a block or a window.
struct Range {
    int32_t first, last;
};

// Butterfly (XOR-shuffle) all-reduce within the partition: log2(P) shuffles, every
// lane active, no local memory or barriers (boundary searches run once per sweep).
template <size_t P, typename Partition>
__attribute__((always_inline)) inline int32_t partition_reduce_min(const Partition& partition, int32_t value) {
BATCHLAS_UNROLL_FULL
    for (uint32_t mask = 1u; mask < static_cast<uint32_t>(P); mask <<= 1) {
        const int32_t other = permute_group_by_xor(partition, value, mask);
        value = (other < value) ? other : value;
    }
    return value;
}

template <size_t P, typename T, typename Partition>
__attribute__((always_inline)) inline T partition_reduce_fmax(const Partition& partition, T value) {
BATCHLAS_UNROLL_FULL
    for (uint32_t mask = 1u; mask < static_cast<uint32_t>(P); mask <<= 1) {
        value = sycl::fmax(value, permute_group_by_xor(partition, value, mask));
    }
    return value;
}

// T = tridiag(e, d, e) spread over a chunk: this lane holds d_lane and e_lane by value.
// d_at/e_at are broadcasts, so every lane calls them with an in-range source.
template <typename T, size_t P, typename Part>
struct TridiagLanes {
    Part part;
    int32_t lane, n;
    T d, e;

    __attribute__((always_inline)) TridiagLanes(const Part& p, int32_t n_, T d_, T e_)
        : part(p), lane(static_cast<int32_t>(p.get_local_linear_id())), n(n_), d(d_), e(e_) {}

    __attribute__((always_inline)) T d_at(int32_t i) const { return select_from_group(part, d, i); }
    __attribute__((always_inline)) T e_at(int32_t i) const { return select_from_group(part, e, i); }
    // d_{lane+1}; unspecified on the last lane.
    __attribute__((always_inline)) T d_next() const { return shift_group_left(part, d, 1); }
    __attribute__((always_inline)) bool owns_d(Range r) const { return lane >= r.first && lane <= r.last; }
    // e_i couples d_i and d_{i+1}: inside r when both are.
    __attribute__((always_inline)) bool owns_e(Range r) const { return lane >= r.first && lane < r.last; }
};

// e_i := 0 for every negligible e_i of r (LAPACK's relative test; no absolute threshold).
// evidence: docs/algorithms/steqr.md#steqr-the-relative-deflation-criterion
template <typename T, size_t P, typename Part>
__attribute__((always_inline)) inline void mark_negligible(TridiagLanes<T, P, Part>& t, Range r) {
    const bool tested = (t.lane + 1 < t.n) && t.owns_e(r);
    const T d_next = t.d_next();
    if (tested) {
        if (t.e != T(0)) {
            if (negligible(t.e, t.d, d_next)) {
                t.e = T(0);
            }
        }
    }
}

// The first m in [r.first, r.last) with e_m == 0, else r.last: T(r) splits after row m.
template <typename T, size_t P, typename Part>
__attribute__((always_inline)) inline int32_t first_split(const TridiagLanes<T, P, Part>& t, Range r) {
    const int32_t candidate = (t.lane >= r.first && t.lane < r.last && t.e == T(0)) ? t.lane : r.last;
    return partition_reduce_min<P>(t.part, candidate);
}

// max |T(r)| over d and e; a dead phase (`live` false) contributes 0.
template <typename T, size_t P, typename Part>
__attribute__((always_inline)) inline T max_abs(const TridiagLanes<T, P, Part>& t, Range r, bool live = true) {
    T candidate = T(0);
    if (live && t.owns_d(r)) {
        candidate = sycl::fabs(t.d);
    }
    if (live && t.owns_e(r)) {
        candidate = sycl::fmax(candidate, sycl::fabs(t.e));
    }
    return partition_reduce_fmax<P>(t.part, candidate);
}

// T(r) <- J T(r) J and Z <- Z J when `flip` (J the reversal): d mirrors about first+last,
// e about first+last-1; the couplings outside r stay. A QR sweep is a QL sweep on J T J.
// evidence: docs/algorithms/steqr.md#steqr-choosing-qr-versus-ql
template <typename T, size_t P, typename Part, typename QCache>
__attribute__((always_inline)) inline void mirror(TridiagLanes<T, P, Part>& t, QCache& q, Range r, bool flip) {
    const int32_t src_d = (flip && t.owns_d(r)) ? (r.first + r.last - t.lane) : t.lane;
    const int32_t src_e = (flip && t.owns_e(r)) ? (r.first + r.last - 1 - t.lane) : t.lane;
    t.d = select_from_group(t.part, t.d, static_cast<uint32_t>(src_d));
    t.e = select_from_group(t.part, t.e, static_cast<uint32_t>(src_e));
    q.reverse_columns(r.first, r.last, flip);
}

// The next unreduced block from `begin`: e_{begin-1} := 0 (the split), the negligible e of the
// tail are zeroed, and the block ends at the first zero e. A dead phase searches nothing.
template <typename T, size_t P, typename Part>
__attribute__((always_inline)) inline Range find_block(TridiagLanes<T, P, Part>& t, int32_t begin, bool live = true) {
    if (live && begin > 0 && t.lane == (begin - 1) && t.lane < (t.n - 1)) {
        t.e = T(0);
    }
    const Range tail{live ? begin : t.n, t.n - 1};
    mark_negligible(t, tail);
    return {begin, first_split(t, tail)};
}

// An unreduced block in QL orientation: scaled into the safe range, and mirrored when its
// last |d| is the smaller end (the small end converges first; the inverted rule loses
// relative accuracy). The orientation holds for the whole block.
template <typename T>
struct OpenBlock {
    Range block{0, 0};
    BlockScale<T> scale{};
    bool reversed = false;
};

template <typename T, size_t P, typename Part, typename QCache>
__attribute__((always_inline)) inline OpenBlock<T> open_block(TridiagLanes<T, P, Part>& t, QCache& q, Range block,
                                                              bool live = true) {
    const BlockScale<T> scale = BlockScale<T>::for_norm(max_abs(t, block, live));
    if (live && scale.active()) {
        if (t.owns_d(block)) {
            t.d = scale.to_safe(t.d);
        }
        if (t.owns_e(block)) {
            t.e = scale.to_safe(t.e);
        }
    }
    // evidence: docs/perf/steqr.md#cta-steqr-chase-micro-structure-decisions
    const T d_first = sycl::fabs(t.d_at(block.first));
    const T d_last = sycl::fabs(t.d_at(block.last));
    const bool reversed = live && (d_last < d_first);
    mirror(t, q, block, reversed);
    return {block, scale, reversed};
}

// Mirror back and undo the scale; also after a failure.
template <typename T, size_t P, typename Part, typename QCache>
__attribute__((always_inline)) inline void close_block(TridiagLanes<T, P, Part>& t, QCache& q, const OpenBlock<T>& b,
                                                       bool live = true) {
    mirror(t, q, b.block, live && b.reversed);
    if (live && b.scale.active()) {
        if (t.owns_d(b.block)) {
            t.d = b.scale.back(t.d);
        }
        if (t.owns_e(b.block)) {
            t.e = b.scale.back(t.e);
        }
    }
}

// What the window [l, m] (m the first split at or after l) needs next.
enum class WindowAction { Converged1, Exact2x2, Sweep, OutOfBudget };

__attribute__((always_inline)) inline WindowAction classify(int32_t l, int32_t m, int32_t budget) {
    if (m == l) return WindowAction::Converged1;
    if (m == l + 1) return WindowAction::Exact2x2;
    if (budget <= 0) return WindowAction::OutOfBudget;
    return WindowAction::Sweep;
}

// (lambda1, lambda2, R) = eig [[d_l, e_l], [e_l, d_l+1]], e_l := 0, Z <- Z R^T on (l, l+1).
// The inputs are broadcasts, so every lane evaluates the 2x2 redundantly. `live` false still
// runs the broadcasts (l in [0, P-2]) and writes nothing.
template <typename T, size_t P, typename Part, typename QCache>
__attribute__((always_inline)) inline void solve_exact_2x2(TridiagLanes<T, P, Part>& t, QCache& q, int32_t l,
                                                           bool live = true) {
    const Sym2x2<T> A{t.d_at(l), t.e_at(l), t.d_at(l + 1)};
    const auto [rt1, rt2, R] = A.eig();
    if (live && t.lane == l) {
        t.d = rt1;
        t.e = T(0);
    }
    if (live && t.lane == (l + 1)) {
        t.d = rt2;
    }
    if (live) q.rotate(l, l + 1, R);
}

// One implicit QL sweep on the unreduced window [l, m], EXP update: the shift sigma comes from
// the leading 2x2, the seed pair is (d_m - sigma, e_{m-1}), and for each plane (lo, hi) =
// (k-1, k), k = m down to l+1, R annihilates the pair, [[d_hi, e_lo], [e_lo, d_lo]] <- R A R^T,
// the bulge moves to e_{lo-1}, and Z <- Z R^T on (hi, lo).
// Pad: the chunks of a lockstep domain share one trip count, and a chunk with `active` false
// or a shorter chase idles through identity rotations. Idle state is only discarded by
// selects, never multiplied by a mask, so a failed chunk's NaNs reach neither T nor Z.
template <bool Pad, typename T, size_t P, typename Part, typename QCache>
__attribute__((always_inline)) inline void exp_sweep(TridiagLanes<T, P, Part>& t, QCache& q, Range window,
                                                     SteqrShiftStrategy strategy, bool active) {
    const int32_t l = window.first, m = window.last;
    // An idle chunk's l and m are stale: clamp every shuffle source into the chunk.
    const int32_t ls = Pad ? sycl::clamp(l, int32_t(0), static_cast<int32_t>(P) - 2) : l;
    const int32_t ms = Pad ? sycl::clamp(m, int32_t(0), static_cast<int32_t>(P) - 1) : m;

    T d_l = t.d_at(ls);
    T e_l = t.e_at(ls);
    T d_l1 = t.d_at(ls + 1);
    if constexpr (Pad) {
        // A benign 2x2 keeps an idle chunk's shift finite (e_l == 0 divides by 0).
        d_l = active ? d_l : T(0);
        e_l = active ? e_l : T(1);
        d_l1 = active ? d_l1 : T(0);
    }
    const T sigma = wilkinson_shift(strategy, d_l, e_l, d_l1);

    const int32_t nrot = Pad ? lockstep_max(t.part, active ? m - l : 0) : m - l;
    // The rows the next plane rotates: d_hi and e_lo (= e_{hi-1}) come out of the previous
    // rotation, so they are carried in registers; shuffles run unconditionally, clamped.
    T d_hi = t.d_at(ms);
    T e_mid = t.e_at(std::max(ms - 1, 0));
    e_mid = (ms >= 1) ? e_mid : T(0);
    T e_l_new = T(0);

    // The chase writes row hi but reads only rows below it, so it may read T as it was before
    // the sweep; an immutable snapshot lets the compiler hoist those shuffles off the
    // rotation chain. evidence: docs/perf/steqr.md#cta-steqr-chase-micro-structure-decisions
    const TridiagLanes<T, P, Part> before = t;

    ChaseState<T> pair{d_hi - sigma, e_mid};
    // e_hi = r is final once the bulge has passed (hi < m), and only for a real e (hi < n-1).
    const int32_t e_hi_limit = std::min(m, t.n - 1);

    q.chase_begin(ms);
    for (int32_t v = 0; v < nrot; ++v) {
        const int32_t hi = m - v;
        const int32_t lo = m - v - 1;
        const bool act = !Pad || (active && v < m - l);

        const T d_lo = before.d_at(Pad ? std::max(lo, 0) : lo);
        // e_{lo-1} outside the block is a deflated 0, so the read is safe.
        const T e_below_raw = before.e_at(std::max(lo - 1, 0));
        const T e_below = (lo >= 1) ? e_below_raw : T(0);

        // Idle: (1, 0) gives the exact identity rotation.
        const ChaseState<T> in{act ? pair.pivot : T(1), act ? pair.bulge : T(0)};
        const ExpRotation<T> r = exp_rotation(in, d_hi, e_mid, d_lo, e_below);
        pair.pivot = act ? r.e_mid : pair.pivot;
        pair.bulge = act ? r.bulge : pair.bulge;

        // d_hi and e_hi are final; a select, since an `if` on one target lane diverges.
        const bool owns_hi = act && (t.lane == hi);
        t.d = owns_hi ? r.d_hi : t.d;
        t.e = (owns_hi && hi < e_hi_limit) ? r.e_hi : t.e;

        // Z <- Z R^T on (hi, lo).
        if constexpr (Pad) {
            q.template chase_step_masked<-1>(r.R.c, -r.R.s, act);
        } else {
            q.template chase_step<-1>(r.R.c, -r.R.s);
        }

        d_hi = act ? r.d_lo : d_hi;
        e_mid = act ? r.e_below : e_mid;
        e_l_new = act ? r.e_mid : e_l_new;
    }

    // The last plane was (l, l+1): flush d_l, e_l and e_{l-1}.
    if (active && t.lane == l) {
        t.d = d_hi;
        t.e = e_l_new;
    }
    if (active && l >= 1 && t.lane == (l - 1)) {
        t.e = e_mid;
    }

    if (active) q.chase_end(l);
}

// The same sweep with the PG (Pal-Walker-Kahan) update, DSTEQR's QL inner loop: the carry
// (R_prev, g, p) seeds at ((1, 1), d_m - sigma, 0), and d_l -= p, e_l = g at the end.
template <typename T, size_t P, typename Part, typename QCache>
__attribute__((always_inline)) inline void pg_sweep(TridiagLanes<T, P, Part>& t, QCache& q, Range window,
                                                    SteqrShiftStrategy strategy) {
    const int32_t l = window.first, m = window.last;
    const T d_l = t.d_at(l);
    const T e_l = t.e_at(l);
    const T d_l1 = t.d_at(l + 1);
    const T d_m = t.d_at(m);

    const T sigma = wilkinson_shift(strategy, d_l, e_l, d_l1);
    PgCarry<T> carry{{T(1), T(1)}, T(0), T(0)};
    carry.g = d_m - sigma;

    const int32_t e_hi_limit = std::min(m, t.n - 1);
    q.chase_begin(m);
    for (int32_t hi = m; hi > l; --hi) {
        const int32_t lo = hi - 1;
        const T e_lo = t.e_at(lo);
        const T d_lo = t.d_at(lo);
        const T d_hi = t.d_at(hi);

        const PgRotation<T> r = pg_rotation(carry, d_lo, e_lo, d_hi);
        carry = r.next;

        const bool owns_hi = (t.lane == hi);
        t.d = owns_hi ? r.d_hi : t.d;
        t.e = (owns_hi && hi < e_hi_limit) ? r.e_hi : t.e;

        // Z <- Z R^T on (hi, lo).
        q.template chase_step<-1>(r.R.c, -r.R.s);
    }
    q.chase_end(l);

    const T d_l_new = d_l - carry.p;
    if (t.lane == l) {
        t.d = d_l_new;
        if (l < (t.n - 1)) {
            t.e = carry.g;
        }
    }
}

// One implicit QL sweep on [l, m]: QR sweeps run it on a mirrored block. Pad is EXP only.
template <bool Pad, typename T, size_t P, typename Part, typename QCache>
__attribute__((always_inline)) inline void implicit_ql_sweep(TridiagLanes<T, P, Part>& t, QCache& q, Range window,
                                                             SteqrShiftStrategy strategy, SteqrUpdateScheme scheme,
                                                             bool active = true) {
    if (Pad || scheme == SteqrUpdateScheme::EXP) {
        exp_sweep<Pad>(t, q, window, strategy, active);
    } else {
        pg_sweep(t, q, window, strategy);
    }
}

} // namespace batchlas::tridiag
