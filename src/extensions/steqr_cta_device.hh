#pragma once

// Device-side building blocks of the CTA (sub-group-partition) STEQR solver.
//
// These live in a header because two translation units need the *same* code:
//   - steqr_cta.cc      : the standalone tridiagonal eigensolver kernel
//   - syev_cta_fused.cc : the monolithic SYEV kernel, which runs this solve
//                         in-place between tridiagonalization and back-transform
//
// Keeping a single definition is what makes the fused-vs-partitioned comparison
// a measurement of *fusion* rather than of two independently drifting solvers.

#include <batchlas/blas/matrix.hh>
#include <batchlas/blas/extensions.hh>
#include <batchlas/util/group-invoke.hh>
#include "sg_compat.hh"
#include "../math-helpers.hh"

#include <array>
#include <cstdint>
#include <type_traits>

namespace batchlas {

    // Givens rotation specialised for the real bulge chase.
    //
    // On the in-range fast path this is algebraically identical to internal::lartg(),
    // but it forms the single quantity 1/sqrt(f^2 + g^2) with a hardware reciprocal
    // square root plus one Newton refinement instead of one IEEE square root and two
    // IEEE divisions.  The bulge chase calls this once per rotation (O(n^2) times per
    // problem), and div/sqrt expansion dominated the instruction mix there.
    // Out-of-range inputs fall back to the fully scaled reference implementation.
    template <typename T>
    struct cta_rotation {
        T c;
        T s;
        T r;
    };

    template <typename T>
    inline cta_rotation<T> cta_lartg(const T f, const T g) {
        if constexpr (std::is_same_v<T, float>) {
            // Range guard: |f| and |g| inside (sqrt(safmin), sqrt(safmax/2)) keeps
            // f^2 + g^2 finite and normal; anything else (NaN included, since every
            // compare is false) takes the scaled reference implementation.
            //
            // g == 0 must return exactly (1, 0, f), which rsqrt + Newton does not
            // guarantee. It is forced by the final select, not an early return, so
            // an identity rotation takes the same path as its neighbours instead of
            // diverging; the result is bitwise what the early return gave.
            const bool gz = (g == T(0));
            const T f_abs = sycl::fabs(f);
            const T g_abs = sycl::fabs(g);

            const T rtmin = sycl::sqrt(internal::safmin<T>());
            const T rtmax = sycl::sqrt(internal::safmax<T>() / T(2));

            T c = T(1), s = T(0), r = f;
            if (f_abs > rtmin && f_abs < rtmax && (gz || (g_abs > rtmin && g_abs < rtmax))) {
                const T t = f * f + g * g;
                T inv = sycl::rsqrt(t);
                // One fused Newton step brings the hardware estimate below 0.5 ulp.
                inv = sycl::fma(inv * T(0.5), sycl::fma(-(t * inv), inv, T(1)), inv);
                const T d = sycl::sqrt(t);
                const T signed_inv = sycl::copysign(inv, f);
                c = f_abs * inv;
                s = g * signed_inv;
                r = sycl::copysign(d, f);
            } else if (!gz) {
                const auto res = internal::lartg(f, g);
                c = res.c;
                s = res.s;
                r = res.r;
            }
            return {gz ? T(1) : c, gz ? T(0) : s, gz ? f : r};
        } else {
            const auto res = internal::lartg(f, g);
            return {res.c, res.s, res.r};
        }
    }

    template <typename T>
    inline T wilkinson_shift(const T& a, const T& b, const T& c) {
        // a,b,c represent the 2x2 block:
        //   [ a  b ]
        //   [ b  c ]
        // Return the eigenvalue closest to c.
        const auto [lambda1, lambda2] = internal::eigenvalues_2x2(a, b, c);
        return std::abs(lambda1 - c) < std::abs(lambda2 - c) ? lambda1 : lambda2;
    }

    // Compile-time selectable shared-memory cache for Q.
    //
    // Storage is column-major in local memory with a caller-chosen leading
    // dimension: Q_local[base_q + row + col*LDQ] = Q(row, col).  LDQ == P is
    // right for the standalone solver, which only ever indexes the tile by
    // lane == row (already conflict-free).  A consumer that later reads the tile
    // *by column* (lane == column, as the fused SYEV back-transform does) wants
    // LDQ == P+1 so that consecutive lanes land in different banks.
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
            // Zero rather than skip the padding rows (lane >= n).  Once every row of
            // the tile holds a defined value, the chase can run unguarded on all P
            // lanes: its column indices are always inside the tile, and `store` only
            // reads back the first n rows.  That removes a divergent branch from the
            // innermost loop of the solver.
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
            const int32_t i0 = base_q + lane + col0 * pN;
            const int32_t i1 = base_q + lane + col1 * pN;
            const T q0 = Q_local[i0];
            const T q1 = Q_local[i1];
            Q_local[i0] = c * q0 - s * q1;
            Q_local[i1] = s * q0 + c * q1;
        }

        // Streaming form of `apply` for a bulge chase.
        //
        // Successive rotations in a chase always share a column (rotation k writes
        // columns (a, b) and rotation k+1 reads column b again), so the shared column
        // can stay in a register.  That halves both the shared-memory traffic and the
        // address arithmetic of the eigenvector update.
        //
        // A chase also visits columns strictly consecutively, so the element index
        // only ever moves by +/-P between steps.  Keeping it in a register and
        // advancing it by a compile-time constant turns the per-step address
        // computation into a single integer add, and lets the partner column be
        // reached through the load/store instruction's immediate offset.
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
            const T q1 = Q_local[next];
            Q_local[idx] = c * carry - s * q1;
            carry = s * carry + c * q1;
            idx = next;
        }

        // Padded form for a chunk that idles through a lockstep chase: an idle
        // step reads its own current word and writes nothing, so the index never
        // leaves the lane's own row, even in a tile shared with other chunks.
        template <int32_t Dir>
        inline void chase_step_masked(T c, T s, bool act) {
            const int32_t next = act ? idx + Dir * static_cast<int32_t>(LDQ) : idx;
            const T q1 = Q_local[next];
            if (act) Q_local[idx] = c * carry - s * q1;
            carry = act ? s * carry + c * q1 : carry;
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
            for (int32_t a = bb, b = be; a < b; ++a, --b) {
                const int32_t ia = base_q + lane + a * pN;
                const int32_t ib = base_q + lane + b * pN;
                const T qa = Q_local[ia];
                Q_local[ia] = Q_local[ib];
                Q_local[ib] = qa;
            }
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

    template <typename T, typename Partition>
    inline void deflate(Partition partition,
                        T& e,
                        T& d,
                        int32_t n,
                        int32_t start_ix,
                        int32_t end_ix,
                        T zero_threshold) {
        // `zero_threshold` is currently unused: deflation follows LAPACK's relative test.
        (void)zero_threshold;
        const int32_t lane = static_cast<int32_t>(partition.get_local_linear_id());
        const bool lane_in_active_range = (lane + 1 < n) && (lane >= start_ix) && (lane + 1 < end_ix);

        // We need d_{i+1} (neighbor lane's diagonal). A 1-lane shift is the most direct.
        // Note: for lanes without i+1 (last lane), the result is unspecified, but those
        // lanes never use d_ip1 due to lane_in_active_range.
        const T d_ip1 = shift_group_left(partition, d, 1);

        if (lane_in_active_range) {
            if (e != T(0)) {
                // LAPACK-style relative deflation test:
                // |e|^2 <= eps2 * |d_i| * |d_{i+1}| + safmin
                const T rhs = internal::eps2<T>() * sycl::fabs(d) * sycl::fabs(d_ip1) + internal::safmin<T>();
                if (sycl::fabs(e) * sycl::fabs(e) <= rhs) {
                    e = T(0);
                }
            }
        }
    }

    // T := J*T*J and Q := Q*J on the block [bb, be] when `rev`, J the reversal.
    //
    // A QR sweep on T is exactly a QL sweep on J*T*J (shift, rotation and
    // deflation formulas are mirror images), so a QR block is mirrored, swept
    // by the QL chase and mirrored back. d mirrors about bb+be, e about
    // bb+be-1; e(bb-1) (the split mark) and e(be) (zero) do not move.
    template <size_t P, typename T, typename Partition, typename QCache>
    inline void reverse_block(const Partition& partition,
                              T& diag,
                              T& offdiag,
                              QCache& qcache,
                              int32_t bb,
                              int32_t be,
                              bool rev) {
        const int32_t lane = static_cast<int32_t>(partition.get_local_linear_id());
        const int32_t src_d = (rev && lane >= bb && lane <= be) ? (bb + be - lane) : lane;
        const int32_t src_e = (rev && lane >= bb && lane < be) ? (bb + be - 1 - lane) : lane;
        diag = select_from_group(partition, diag, static_cast<uint32_t>(src_d));
        offdiag = select_from_group(partition, offdiag, static_cast<uint32_t>(src_e));
        qcache.reverse_columns(bb, be, rev);
    }

    // Butterfly (XOR-shuffle) all-reduce within the partition.
    //
    // These replace the previous shared-memory + leader-lane serial loops.  A
    // butterfly reduction needs log2(P) shuffles, keeps every lane active, and
    // touches neither local memory nor barriers, which matters because the
    // block/subproblem boundary searches run once per QL/QR sweep.
    template <size_t P, typename Partition>
    inline int32_t partition_reduce_min(const Partition& partition, int32_t value) {
#pragma unroll
        for (uint32_t mask = 1u; mask < static_cast<uint32_t>(P); mask <<= 1) {
            const int32_t other = permute_group_by_xor(partition, value, mask);
            value = (other < value) ? other : value;
        }
        return value;
    }

    template <size_t P, typename Partition>
    inline int32_t partition_reduce_max(const Partition& partition, int32_t value) {
#pragma unroll
        for (uint32_t mask = 1u; mask < static_cast<uint32_t>(P); mask <<= 1) {
            const int32_t other = permute_group_by_xor(partition, value, mask);
            value = (other > value) ? other : value;
        }
        return value;
    }

    template <size_t P, typename T, typename Partition>
    inline T partition_reduce_fmax(const Partition& partition, T value) {
#pragma unroll
        for (uint32_t mask = 1u; mask < static_cast<uint32_t>(P); mask <<= 1) {
            value = sycl::fmax(value, permute_group_by_xor(partition, value, mask));
        }
        return value;
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
        const auto [rt1, rt2, cs, sn] = internal::laev2(a, b, c2);
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

        // EXP update scheme = explicit similarity update (bulge-chase), matching the logic in steqr.cc.
        // We implement QL by operating on a *virtual reversed* indexing inside [l..m] and running a QR-style
        // bulge chase in that virtual space. This is the only chase: QR sweeps run it on a mirrored block
        // (see reverse_block).
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
                mu = wilkinson_shift(dlp1, e0, p0);
            } else {
                const T gg = (dlp1 - p0) / (T(2) * e0);
                const T rr = sycl::hypot(gg, T(1));
                mu = p0 - e0 / (gg + sycl::copysign(rr, gg));
            }

            const int32_t nrot = Pad ? lockstep_max(partition, active ? m - l : 0) : m - l;
            // Virtual index v in [0..m-l-1] maps to physical indices:
            //   d_v(v)   = d( m - v )
            //   d_v(v+1) = d( m - v - 1 )
            //   e_v(v)   = e( m - v - 1 )  (couples the two diags above)
            //   e_v(v+1) = e( m - v - 2 )
            //
            // The chase walks physical indices downward one step at a time, so the
            // (di, ei) pair of iteration v+1 is exactly the (dj_new, ej_new) pair this
            // iteration just produced.  Carrying them in registers halves the number of
            // cross-lane shuffles in the hottest loop of the solver.
            // Shuffles run unconditionally with a clamped source and the value is
            // selected afterwards: a shuffle under a condition not provably warp-uniform
            // costs a MATCH/VOTE/BRA.DIV wrapper per call and breaks full-warp lockstep.
            T di = select_from_group(partition, diag, ms);
            T ei = select_from_group(partition, offdiag, std::max(ms - 1, 0));
            ei = (ms >= 1) ? ei : T(0);
            T e_own = T(0);

            // Snapshot the tridiagonal before the chase.
            //
            // The chase writes lane `hi` at iteration v but only ever reads lanes
            // strictly below it, so every broadcast below observes the pre-chase value.
            // Shuffling from immutable snapshots is what makes that visible to the
            // compiler: reading `diag`/`offdiag` directly forces each SHFL to be
            // ordered after the previous iteration's conditional write, which chains
            // them onto the lartg dependency path.  From snapshots the shuffles are
            // loop-invariant-free and can be hoisted and overlapped with the rotation
            // arithmetic instead.
            const T diag_snap = diag;
            const T offdiag_snap = offdiag;

            // The first rotation uses (d(m) - mu, e(m-1)); every later one uses the
            // running (eprev, bulge) pair.  Seeding the running pair with the initial
            // values makes the two cases identical, which removes a loop-carried bool,
            // its two selects and a branch from every iteration of the hottest loop.
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

                    const auto [c1, s1, r1] = cta_lartg(x, y);
                    const T sigma = -s1; // match steqr.cc / saved-rotation convention

                    // Update the offdiagonal to the *higher* physical index (virtual e(v-1) -> physical e(hi)).
                    // This corresponds to LAPACK's QL inner-loop assignment E(i+1)=r.
                    const T e_hi_new = x * c1 - y * sigma; // only meaningful when !first

                    // Explicit similarity update for the local (di, ei, dj) pair plus propagation into ej.
                    // This matches the formulas used in steqr.cc's apply_givens_rotation for QR sweeps,
                    // applied in the virtual ordering.
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

                // Only two of the five candidate register updates survive to the next
                // iteration: d(hi) and e(hi) are final once the bulge has moved past
                // them, while d(lo), e(lo) and e(lo-1) are recomputed by the following
                // rotation.  Those three are therefore carried in registers and written
                // once after the chase instead of every iteration.
                //
                // Written as selects, not `if`s: exactly one lane of the partition is
                // ever the target, so a branch here is a guaranteed divergence (and a
                // BSSY/BSYNC pair) on every single rotation.
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
                mu = wilkinson_shift(dlp1, e0, p0);
            } else {
                // LAPACK-style stable implicit shift.
                const T gg = (dlp1 - p0) / (T(2) * e0);
                const T rr = sycl::hypot(gg, T(1));
                mu = p0 - e0 / (gg + sycl::copysign(rr, gg));
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

                const auto [c1, s1, r1] = cta_lartg(g, f);

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

    // One problem's worth of the STEQR outer loop: split into blocks separated by
    // zero offdiagonals, then run shifted QL sweeps on each block (QR as QL on the
    // mirrored block).
    //
    // `diag`/`offdiag` are register-resident with lane i owning d(i) and e(i);
    // `qcache` accumulates the rotations (a no-op when eigenvectors are not
    // wanted). Returns true if any block failed to converge within budget.
    template <typename T, size_t P, typename Partition, typename QCache>
    inline bool steqr_cta_solve_nested(const Partition& partition,
                                       T& diag,
                                       T& offdiag,
                                       QCache& qcache,
                                       int32_t n,
                                       int32_t max_sweeps,
                                       T zero_threshold,
                                       SteqrShiftStrategy cta_shift_strategy,
                                       SteqrUpdateScheme cta_update_scheme) {
        const int32_t lane = static_cast<int32_t>(partition.get_local_linear_id());

        // Defensive convergence budget to avoid unbounded looping on hard inputs.
        // Each implicit step consumes one unit. Budget scales with problem size.
        int32_t sweep_budget = max_sweeps * n;
        bool failed = false;

        // ---- Outer split loop over blocks separated by E==0 ----
        for (int32_t next_block_begin = 0; next_block_begin < n;) {
            const int32_t block_begin = next_block_begin;

            // Mark the split explicitly as LAPACK does: E(block_begin-1)=0.
            if (block_begin > 0 && lane == (block_begin - 1) && lane < (n - 1)) {
                offdiag = T(0);
            }

            // Deflation pass over the remaining tail to create more zeros in E.
            deflate(partition, offdiag, diag, n, block_begin, n, zero_threshold);

            // Find end of current block: first i>=block_begin where E(i)==0; if none, block ends at n-1.
            const int32_t block_end_candidate =
                (lane >= block_begin && lane < (n - 1) && offdiag == T(0)) ? lane : (n - 1);
            const int32_t block_end = partition_reduce_min<P>(partition, block_end_candidate);

            // Next block starts after block_end.
            next_block_begin = block_end + 1;

            // Size-0/1 block.
            if (block_end <= block_begin) {
                continue;
            }

            // Numerical scaling (LAPACK-style): bring the active block norm into
            // a safe range to avoid overflow/underflow on tough inputs.
            T anorm_cand = T(0);
            if (lane >= block_begin && lane <= block_end) {
                anorm_cand = sycl::fabs(diag);
            }
            if (lane >= block_begin && lane < block_end) {
                anorm_cand = sycl::fmax(anorm_cand, sycl::fabs(offdiag));
            }
            const T anorm = partition_reduce_fmax<P>(partition, anorm_cand);

            T scale = T(1);
            if (anorm > internal::ssfmax<T>()) {
                scale = internal::ssfmax<T>() / anorm;
            } else if (anorm < internal::ssfmin<T>() && anorm != T(0)) {
                scale = internal::ssfmin<T>() / anorm;
            }
            const T inv_scale = T(1) / scale;

            if (scale != T(1)) {
                if (lane >= block_begin && lane <= block_end) {
                    diag *= scale;
                }
                if (lane >= block_begin && lane < block_end) {
                    offdiag *= scale;
                }
            }

            // Choose between QL and QR as LAPACK dsteqr does: QL if |D(l)| <= |D(lend)|,
            // QR otherwise, so a graded block converges its small end first. The
            // inverted rule took ~2x the steps and lost relative accuracy on graded input.
            // QR runs as QL on the mirrored block (reverse_block), so chunks of one
            // warp that pick different directions share one loop nest instead of
            // running two back to back. It is mirrored back even on failure.
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
                    // Deflate within current active subproblem [l..lend].
                    deflate(partition, offdiag, diag, n, l, block_end + 1, zero_threshold);

                    // Find first m in [l..lend-1] such that E(m)==0; if none, m=lend.
                    const int32_t m_candidate = (lane >= l && lane < block_end && offdiag == T(0)) ? lane : block_end;
                    const int32_t m = partition_reduce_min<P>(partition, m_candidate);

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
                if (lane >= block_begin && lane <= block_end) {
                    diag *= inv_scale;
                }
                if (lane >= block_begin && lane < block_end) {
                    offdiag *= inv_scale;
                }
            }

            if (failed) break;
        }

        return failed;
    }

    // steqr_cta_solve_nested with the sweep hoisted out of the loop nest, for a
    // partition whose collectives are chunk-local (native chunked_partition).
    //
    // The nested loops realign the chunks of a warp only at their exits, so a
    // chunk that advances past an eigenvalue waits while its neighbours finish
    // sweeping theirs, and the warp pays the sum over eigenvalues of the slowest
    // chunk. Here a chunk settles (opens, tests, 2x2-solves, closes) until it has
    // a sweep to run or is done, and every chunk with a sweep chases in the same
    // pass, so the warp pays closer to the slowest chunk's total. The per-chunk
    // operation sequence is the nested solver's, so the results are bitwise equal.
    template <typename T, size_t P, typename Partition, typename QCache>
    inline bool steqr_cta_solve_flat(const Partition& partition,
                                     T& diag,
                                     T& offdiag,
                                     QCache& qcache,
                                     int32_t n,
                                     int32_t max_sweeps,
                                     T zero_threshold,
                                     SteqrShiftStrategy cta_shift_strategy,
                                     SteqrUpdateScheme cta_update_scheme) {
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
                    deflate(partition, offdiag, diag, n, bb, n, zero_threshold);
                    const int32_t be = partition_reduce_min<P>(
                        partition, (lane >= bb && lane < (n - 1) && offdiag == T(0)) ? lane : (n - 1));
                    next_block_begin = be + 1;
                    if (be <= bb) {
                        if (next_block_begin >= n) {
                            done = true;
                            break;
                        }
                        continue;
                    }

                    T anorm_cand = T(0);
                    if (lane >= bb && lane <= be) {
                        anorm_cand = sycl::fabs(diag);
                    }
                    if (lane >= bb && lane < be) {
                        anorm_cand = sycl::fmax(anorm_cand, sycl::fabs(offdiag));
                    }
                    const T anorm = partition_reduce_fmax<P>(partition, anorm_cand);
                    scale = T(1);
                    if (anorm > internal::ssfmax<T>()) {
                        scale = internal::ssfmax<T>() / anorm;
                    } else if (anorm < internal::ssfmin<T>() && anorm != T(0)) {
                        scale = internal::ssfmin<T>() / anorm;
                    }
                    inv_scale = T(1) / scale;
                    if (scale != T(1)) {
                        if (lane >= bb && lane <= be) {
                            diag *= scale;
                        }
                        if (lane >= bb && lane < be) {
                            offdiag *= scale;
                        }
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
                    deflate(partition, offdiag, diag, n, l, block_end + 1, zero_threshold);
                    m = partition_reduce_min<P>(
                        partition, (lane >= l && lane < block_end && offdiag == T(0)) ? lane : block_end);
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
                        if (lane >= block_begin && lane <= block_end) {
                            diag *= inv_scale;
                        }
                        if (lane >= block_begin && lane < block_end) {
                            offdiag *= inv_scale;
                        }
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

    // The same state machine for a partition whose lockstep domain is the whole
    // sub-group (emulated partition): every branch that guards a collective must
    // be taken by all chunks, so each phase runs under a sub-group vote with its
    // updates gated per chunk, and the sweep is the padded chase. A chunk outside
    // a phase feeds its collectives in-range sources and discards the results.
    template <typename T, size_t P, typename Partition, typename QCache>
    inline bool steqr_cta_solve_lockstep(const Partition& partition,
                                         T& diag,
                                         T& offdiag,
                                         QCache& qcache,
                                         int32_t n,
                                         int32_t max_sweeps,
                                         T zero_threshold,
                                         SteqrShiftStrategy cta_shift_strategy,
                                         SteqrUpdateScheme cta_update_scheme) {
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
                    deflate(partition, offdiag, diag, n, opening ? bb : n, n, zero_threshold);
                    const int32_t be = partition_reduce_min<P>(
                        partition, (opening && lane >= bb && lane < (n - 1) && offdiag == T(0)) ? lane : (n - 1));
                    const bool opened = opening && be > bb;

                    T anorm_cand = T(0);
                    if (opened && lane >= bb && lane <= be) {
                        anorm_cand = sycl::fabs(diag);
                    }
                    if (opened && lane >= bb && lane < be) {
                        anorm_cand = sycl::fmax(anorm_cand, sycl::fabs(offdiag));
                    }
                    const T anorm = partition_reduce_fmax<P>(partition, anorm_cand);
                    T sc = T(1);
                    if (anorm > internal::ssfmax<T>()) {
                        sc = internal::ssfmax<T>() / anorm;
                    } else if (anorm < internal::ssfmin<T>() && anorm != T(0)) {
                        sc = internal::ssfmin<T>() / anorm;
                    }
                    if (opened && sc != T(1)) {
                        if (lane >= bb && lane <= be) {
                            diag *= sc;
                        }
                        if (lane >= bb && lane < be) {
                            offdiag *= sc;
                        }
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
                    deflate(partition, offdiag, diag, n, active ? l : n, block_end + 1, zero_threshold);
                    const int32_t mm = partition_reduce_min<P>(
                        partition, (active && lane >= l && lane < block_end && offdiag == T(0)) ? lane : block_end);
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
                        if (lane >= block_begin && lane <= block_end) {
                            diag *= inv_scale;
                        }
                        if (lane >= block_begin && lane < block_end) {
                            offdiag *= inv_scale;
                        }
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
    // A sub-group-wide partition cannot run the nested loops legally when its
    // chunks diverge, so it takes the lockstep solver (EXP; the padded chase has
    // no PG form) whenever a warp holds more than one chunk.
    template <typename T, size_t P, typename Partition, typename QCache>
    inline bool steqr_cta_solve(const Partition& partition,
                                T& diag,
                                T& offdiag,
                                QCache& qcache,
                                int32_t n,
                                int32_t max_sweeps,
                                T zero_threshold,
                                SteqrShiftStrategy cta_shift_strategy,
                                SteqrUpdateScheme cta_update_scheme) {
        if constexpr (lockstep_spans_subgroup_v<Partition> && P < 32) {
            if (cta_update_scheme == SteqrUpdateScheme::EXP) {
                return steqr_cta_solve_lockstep<T, P>(partition, diag, offdiag, qcache, n, max_sweeps,
                                                      zero_threshold, cta_shift_strategy, cta_update_scheme);
            }
        } else if constexpr (kSteqrCtaFlatPays<T, P>) {
            return steqr_cta_solve_flat<T, P>(partition, diag, offdiag, qcache, n, max_sweeps,
                                              zero_threshold, cta_shift_strategy, cta_update_scheme);
        }
        return steqr_cta_solve_nested<T, P>(partition, diag, offdiag, qcache, n, max_sweeps,
                                            zero_threshold, cta_shift_strategy, cta_update_scheme);
    }

} // namespace batchlas
