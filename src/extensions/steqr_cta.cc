#include <blas/matrix.hh>
#include <blas/functions.hh>
#include <blas/extensions.hh>
#include <blas/extra.hh>
#include <util/kernel-heuristics.hh>
#include <util/group-invoke.hh>
#include "sg_compat.hh"
#include <util/mempool.hh>
#include <batchlas/backend_config.h>
#include "steqr_internal.hh"
#include "../math-helpers.hh"
#include "../queue.hh"
#include "../util/kernel-trace.hh"
#include "../util/template-instantiations.hh"
#include <internal/sort.hh>
#include <array>
#include <numeric>

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
            // Range guard.
            //
            // The previous formulation tested |f| and |g| separately against
            // sqrt(safmin) and sqrt(safmax/2): four FSETPs plus three predicate
            // merges plus a separate `g == 0` early-out, i.e. ~9 instructions on
            // the hottest path in the solver.  Everything those tests protect is
            // a property of t = f^2 + g^2 alone, so one interval test on `t`
            // (which we have to form anyway) is both cheaper and stricter:
            //   * overflow of f*f or g*g yields t == inf  -> rejected by t < tmax
            //   * total underflow yields t == 0           -> rejected by t > tmin
            //   * NaN inputs yield t == NaN               -> both compares false
            // The `g == 0` case needs no special handling either: it gives
            // inv = 1/|f|, hence c = 1, s = 0 and r = f exactly.
            if (g == T(0)) return {T(1), T(0), f};

            const T f_abs = sycl::fabs(f);
            const T g_abs = sycl::fabs(g);

            const T rtmin = sycl::sqrt(internal::safmin<T>());
            const T rtmax = sycl::sqrt(internal::safmax<T>() / T(2));

            if (f_abs > rtmin && f_abs < rtmax && g_abs > rtmin && g_abs < rtmax) {
                const T t = f * f + g * g;
                T inv = sycl::rsqrt(t);
                // One fused Newton step brings the hardware estimate below 0.5 ulp.
                inv = sycl::fma(inv * T(0.5), sycl::fma(-(t * inv), inv, T(1)), inv);
                const T d = sycl::sqrt(t);
                const T signed_inv = sycl::copysign(inv, f);
                return {f_abs * inv, g * signed_inv, sycl::copysign(d, f)};
            }
        }

        const auto res = internal::lartg(f, g);
        return {res.c, res.s, res.r};
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

    template <typename T, size_t P, bool ComputeVecs>
    class SteqrCTAKernel;

    // Compile-time selectable shared-memory cache for Q.
    // Storage is column-major in local memory: Q_local[base_q + row + col*P] = Q(row, col)
    template <typename T, size_t P, bool ComputeVecs, typename LocalAcc>
    struct QSharedCache;

    template <typename T, size_t P, typename LocalAcc>
    struct QSharedCache<T, P, true, LocalAcc> {
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
            const int32_t pN = static_cast<int32_t>(P);
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
            const int32_t pN = static_cast<int32_t>(P);
            for (int32_t c = 0; c < n; ++c) {
                Q_prob(lane, c) = Q_local[base_q + lane + c * pN];
            }
        }

        inline void apply(int32_t col0, int32_t col1, T c, T s) {
            const int32_t pN = static_cast<int32_t>(P);
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
            idx = base_q + lane + col * static_cast<int32_t>(P);
            carry = Q_local[idx];
        }

        // Dir is +1 when the chase walks toward higher column indices and -1 when it
        // walks toward lower ones; the written column is the current one and the read
        // column is the neighbour in that direction.
        template <int32_t Dir>
        inline void chase_step(T c, T s) {
            const int32_t next = idx + Dir * static_cast<int32_t>(P);
            const T q1 = Q_local[next];
            Q_local[idx] = c * carry - s * q1;
            carry = s * carry + c * q1;
            idx = next;
        }

        inline void chase_end(int32_t) {
            Q_local[idx] = carry;
        }
    };

    template <typename T, size_t P, typename LocalAcc>
    struct QSharedCache<T, P, false, LocalAcc> {
        QSharedCache(LocalAcc, int32_t, int32_t, int32_t) {}

        template <typename QProb>
        inline void load(const QProb&) {}

        template <typename QProb>
        inline void store(QProb&) const {}

        inline void apply(int32_t, int32_t, T, T) {}
        inline void chase_begin(int32_t) {}
        template <int32_t Dir>
        inline void chase_step(T, T) {}
        inline void chase_end(int32_t) {}
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
                                     bool ql,
                                     QCache& qcache) {
        const T a = select_from_group(partition, diag, l0);
        const T b = select_from_group(partition, offdiag, l0);
        const T c2 = select_from_group(partition, diag, l0 + 1);

        // Every input is already partition-uniform, so evaluate laev2 redundantly on
        // all lanes instead of computing on the leader and broadcasting four values.
        const auto [rt1, rt2, cs, sn] = internal::laev2(a, b, c2);
        const int32_t lane = static_cast<int32_t>(partition.get_local_linear_id());
        if (lane == l0) {
            diag = rt1;
            offdiag = T(0);
        }
        if (lane == (l0 + 1)) {
            diag = rt2;
        }

        // Inline QR/QL eigenvector update:
        // - QR: apply (cs, -sn) on columns (l0, l0+1)
        // - QL: apply (cs,  sn) on columns (l0+1, l0)
        const int32_t col0 = ql ? (l0 + 1) : l0;
        const int32_t col1 = ql ? l0 : (l0 + 1);
        const T s_eff = ql ? sn : -sn;
        qcache.apply(col0, col1, cs, s_eff);
    }

    template <typename T, size_t P, typename Partition, typename QCache>
    inline void implicit_ql_step(const Partition& partition,
                                 T& diag,
                                 T& offdiag,
                                 QCache& qcache,
                                 int32_t n,
                                 int32_t l,
                                 int32_t m,
                                 SteqrShiftStrategy shift_strategy,
                                 SteqrUpdateScheme update_scheme) {
        const int32_t lane = static_cast<int32_t>(partition.get_local_linear_id());

        // EXP update scheme = explicit similarity update (bulge-chase), matching the logic in steqr.cc.
        // We implement QL by operating on a *virtual reversed* indexing inside [l..m] and running a QR-style
        // bulge chase in that virtual space.
        const auto explicit_ql_step_exp = [&]() {
            // Preload shift inputs.
            const T p0  = select_from_group(partition, diag, l);
            const T e0  = select_from_group(partition, offdiag, l);
            const T dlp1 = select_from_group(partition, diag, l + 1);

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

            const int32_t nb = m - l + 1; // block length
            // Virtual index v in [0..nb-2] maps to physical indices:
            //   d_v(v)   = d( m - v )
            //   d_v(v+1) = d( m - v - 1 )
            //   e_v(v)   = e( m - v - 1 )  (couples the two diags above)
            //   e_v(v+1) = e( m - v - 2 )
            //
            // The chase walks physical indices downward one step at a time, so the
            // (di, ei) pair of iteration v+1 is exactly the (dj_new, ej_new) pair this
            // iteration just produced.  Carrying them in registers halves the number of
            // cross-lane shuffles in the hottest loop of the solver.
            T di = select_from_group(partition, diag, m);
            T ei = (m - 1) >= 0 ? select_from_group(partition, offdiag, m - 1) : T(0);
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

            qcache.chase_begin(m);

            for (int32_t v = 0; v < nb - 1; ++v) {
                const int32_t hi = m - v;
                const int32_t lo = m - v - 1;

                const T dj = select_from_group(partition, diag_snap, lo);

                // Next virtual offdiag (toward physical l). It is safe to read outside the block because
                // deflation boundaries force those couplings to zero.
                const bool have_ej = (lo - 1) >= 0;
                const T ej = have_ej ? select_from_group(partition, offdiag_snap, lo - 1) : T(0);

                const auto upd = [&]() {
                    const T x = eprev;
                    const T y = bulge;

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
                    eprev = ei_new;
                    bulge = bulge_new;

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
                const bool owns_hi = (lane == hi);
                diag = owns_hi ? upd[2] : diag;
                offdiag = (owns_hi && hi < e_hi_limit) ? upd[6] : offdiag;

                // QL eigenvector update: columns are reversed in physical ordering.
                // In your existing PG path you do apply(i+1, i, c, -s); here the physical pair is (hi, lo).
                qcache.template chase_step<-1>(c1, sigma);

                // Carry the values the next iteration would otherwise re-shuffle:
                // d(lo) and e(lo-1) are exactly what was just written.
                di = upd[3];
                ei = upd[5];
                e_own = upd[4];
            }

            // Flush the carried tail values (lo == l on the final iteration).
            if (lane == l) {
                diag = di;
                offdiag = e_own;
            }
            if (l >= 1 && lane == (l - 1)) {
                offdiag = ei;
            }

            qcache.chase_end(l);
        };

        if (update_scheme == SteqrUpdateScheme::EXP) {
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

    template <typename T, size_t P, typename Partition, typename QCache>
    inline void implicit_qr_step(const Partition& partition,
                                 T& diag,
                                 T& offdiag,
                                 QCache& qcache,
                                 int32_t n,
                                 int32_t m,
                                 int32_t l,
                                 SteqrShiftStrategy shift_strategy,
                                 SteqrUpdateScheme update_scheme) {
        const int32_t lane = static_cast<int32_t>(partition.get_local_linear_id());

        // EXP update scheme = explicit similarity update (bulge-chase), matching steqr.cc.
        const auto explicit_qr_step_exp = [&]() {
            // Shift from trailing 2x2 of the physical block (l-1,l).
            const T p0  = select_from_group(partition, diag, l);
            const T e0  = select_from_group(partition, offdiag, l - 1);
            const T dlm1 = select_from_group(partition, diag, l - 1);

            T mu = T(0);

            // Partition-uniform: computed redundantly on every lane.
            if (shift_strategy == SteqrShiftStrategy::Wilkinson) {
                mu = wilkinson_shift(dlm1, e0, p0);
            } else {
                const T gg = (dlm1 - p0) / (T(2) * e0);
                const T rr = sycl::hypot(gg, T(1));
                mu = p0 - e0 / (gg + sycl::copysign(rr, gg));
            }

            // The chase walks physical indices upward one step at a time, so the (di, ei)
            // pair of iteration i+1 is exactly the (dj_new, ej_new) pair produced here.
            // Carrying them in registers halves the cross-lane shuffles in this loop.
            T di = select_from_group(partition, diag, m);
            T ei = select_from_group(partition, offdiag, m);
            T e_own = T(0);

            // Snapshot the tridiagonal before the chase; see the QL path for why.
            // The chase writes lanes i and i-1 at iteration i but reads only lanes
            // i+1 and above, so the broadcasts always want the pre-chase values.
            // Sourcing them from immutable copies frees the compiler to hoist the
            // SHFLs off the rotation's dependency chain.
            const T diag_snap = diag;
            const T offdiag_snap = offdiag;

            // Seed the running (eprev, bulge) pair with the first rotation's operands so
            // that every iteration takes the same path; see the QL chase for details.
            T eprev = di - mu;
            T bulge = ei;

            qcache.chase_begin(m);

            for (int32_t i = m; i < l; ++i) {
                const T dj = select_from_group(partition, diag_snap, i + 1);

                const bool have_ej = (i + 1) < (n - 1);
                const T ej = have_ej ? select_from_group(partition, offdiag_snap, i + 1) : T(0);

                const auto upd = [&]() {
                    const T x = eprev;
                    const T y = bulge;

                    const auto [c1, s1, r1] = cta_lartg(x, y);
                    const T sigma = -s1;

                    // Update physical e(i-1) for i>m.
                    const T e_im1_new = x * c1 - y * sigma;

                    // Explicit similarity update for local pair (di, ei, dj) and propagation into ej.
                    const T di_new = c1 * (c1 * di - ei * sigma) - sigma * (ei * c1 - sigma * dj);
                    const T dj_new = c1 * (c1 * dj + ei * sigma) + sigma * (ei * c1 + sigma * di);
                    const T ei_new = c1 * (c1 * ei + sigma * di) - sigma * (c1 * dj + sigma * ei);

                    const T ej_new = c1 * ej;
                    const T bulge_new = -ej * sigma;

                    eprev = ei_new;
                    bulge = bulge_new;

                    // Return {c, sigma, di_new, dj_new, ei_new, ej_new, e_im1_new}
                    return std::array<T, 7>{c1, sigma, di_new, dj_new, ei_new, ej_new, e_im1_new};
                }();

                const T c1 = upd[0];
                const T sigma = upd[1];

                // Only d(i) and e(i-1) survive to the next rotation; d(i+1), e(i+1)
                // and e(i) are all recomputed by it, so they are carried in registers
                // and flushed once after the chase.  Selects rather than branches: see
                // the QL chase for why.
                diag = (lane == i) ? upd[2] : diag;
                offdiag = (i > m && lane == (i - 1)) ? upd[6] : offdiag;

                // QR eigenvector update: columns (i, i+1).
                qcache.template chase_step<1>(c1, sigma);

                // Carry the values the next iteration would otherwise re-shuffle:
                // d(i+1) and e(i+1) are exactly what was just written.
                di = upd[3];
                ei = upd[5];
                e_own = upd[4];
            }

            // Flush the carried tail values (i == l-1 on the final iteration).
            if (lane == l) {
                diag = di;
                if (l < (n - 1)) {
                    offdiag = ei;
                }
            }
            if (lane == (l - 1)) {
                offdiag = e_own;
            }

            qcache.chase_end(l);
        };

        if (update_scheme == SteqrUpdateScheme::EXP) {
            explicit_qr_step_exp();
            return;
        }

        // Broadcast values needed for the shift (all lanes participate).
        const T p0 = select_from_group(partition, diag, l);
        const T e0 = select_from_group(partition, offdiag, l - 1);
        const T dlm1 = select_from_group(partition, diag, l - 1);
        const T dm = select_from_group(partition, diag, m);

        // Partition-uniform scalar state (evaluated redundantly on every lane).
        T g = T(0);
        T c = T(1);
        T s = T(1);
        T p = T(0);

        {
            T mu = T(0);
            if (shift_strategy == SteqrShiftStrategy::Wilkinson) {
                mu = wilkinson_shift(dlm1, e0, p0);
            } else {
                const T gg = (dlm1 - p0) / (T(2) * e0);
                const T rr = sycl::hypot(gg, T(1));
                mu = p0 - e0 / (gg + sycl::copysign(rr, gg));
            }
            g = dm - mu;
        }

        qcache.chase_begin(m);

        for (int32_t i = m; i < l; ++i) {
            // Broadcast the tridiagonal entries needed for this step.
            const T ei = select_from_group(partition, offdiag, i);
            const T di = select_from_group(partition, diag, i);
            const T dip1 = select_from_group(partition, diag, i + 1);

            // Whether E(i-1) should be updated is a pure function of (i, m).
            const bool do_e_upd = (i != m);

            const auto [c1b, s1b, d_i_new_b, r1_out_b] = [&]() {
                // {c1, s1, d_i_new, r1_out}
                const T f = s * ei;
                T rout = T(0);

                const auto [c1, s1, r1] = cta_lartg(g, f);

                // In the original local-memory version: E(i-1) = r1 for i != m.
                if (do_e_upd) {
                    rout = r1;
                }

                const T g2 = di - p;
                const T r2 = (dip1 - g2) * s1 + T(2) * c1 * (c * ei);
                p = s1 * r2;

                const T d_i_new = g2 + p;
                g = c1 * r2 - (c * ei);
                c = c1;
                s = s1;

                return std::array{c1, s1, d_i_new, rout};
            }();

            // Apply D/E updates directly to registers (predicated, not branched).
            diag = (lane == i) ? d_i_new_b : diag;
            offdiag = (do_e_upd && lane == (i - 1)) ? r1_out_b : offdiag;

            // Match previous sign convention: apply(i,i+1,c,-s).
            qcache.template chase_step<1>(c1b, -s1b);
        }

        qcache.chase_end(l);

        // Final updates: D(l) = D(l) - p, and E(l-1) = g.
        const T d_l_new_b = p0 - p;
        const T e_lm1_new_b = g;
        if (lane == l) {
            diag = d_l_new_b;
        }
        if (lane == (l - 1)) {
            offdiag = e_lm1_new_b;
        }
    }


    template <typename T, size_t P, bool ComputeVecs>
    inline void steqr_cta_impl(Queue& ctx,
                              VectorView<T>& d,
                              VectorView<T>& e,
                              MatrixView<T, MatrixFormat::Dense>& eigvects,
                              int32_t n,
                              size_t max_sweeps,
                              T zero_threshold,
                              SteqrShiftStrategy cta_shift_strategy,
                              SteqrUpdateScheme cta_update_scheme,
                              size_t cta_wg_size_multiplier,
                              int32_t* status,
                              BumpAllocator& pool) {
        (void)pool;
        const auto batch_size = d.batch_size();
        if (n < 1 || n > static_cast<int32_t>(P) || d.size() != n || e.size() != (n - 1)) {
            throw std::runtime_error("steqr_cta_impl: invalid n or vector sizes for CTA partition.");
        }

        ctx->submit([&](sycl::handler& cgh) {
            auto Q_view = eigvects.kernel_view();
            const auto dev = ctx->get_device();
            const auto sg_sizes = dev.get_info<sycl::info::device::sub_group_sizes>();

            // CTA path assumes warp-sized sub-groups on NVIDIA.
            const int32_t sg_size = 32;

            // Baseline work-group size is LCM(P, sg_size), so we can form fixed-size partitions of size P.
            // Allow scaling it at runtime to tune the number of sub-groups per work-group.
            const int32_t base_wg_size = std::lcm<int32_t>(static_cast<int32_t>(P), static_cast<int32_t>(sg_size));
            int32_t wg_size_multiplier = std::max<int32_t>(int32_t(1), cta_wg_size_multiplier);
            int32_t wg_size = base_wg_size * wg_size_multiplier;

            const int32_t max_wg_size = static_cast<int32_t>(dev.get_info<sycl::info::device::max_work_group_size>());
            if (wg_size > max_wg_size) {
                const int32_t max_mul = std::max<int32_t>(int32_t(1), max_wg_size / base_wg_size);
                wg_size_multiplier = std::min(wg_size_multiplier, max_mul);
                wg_size = base_wg_size * wg_size_multiplier;
            }

            const int32_t probs_per_wg = wg_size / static_cast<int32_t>(P);
            const int32_t num_wg = (batch_size + probs_per_wg - 1) / probs_per_wg;
            const int32_t global_size = num_wg * wg_size;

            auto Q_local = sycl::local_accessor<T, 1>(
                sycl::range<1>(ComputeVecs ? (probs_per_wg * P * P) : 1), cgh);
            cgh.parallel_for<SteqrCTAKernel<T, P, ComputeVecs>>(
                sycl::nd_range<1>(global_size, wg_size),
                [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(sg_size)]] {
                    const auto wg = it.get_group();
                    const int32_t wg_id = static_cast<int32_t>(wg.get_group_linear_id());

                    const auto sg = it.get_sub_group();
                    const auto partition = make_partition<P>(sg);
                    // NOTE: chunked_partition<P>(sg) partitions *within a sub-group*.
                    // If the work-group contains multiple sub-groups, partition.get_group_linear_id()
                    // repeats for each sub-group. Make part_id unique within the whole work-group.
                    const int32_t sg_id = static_cast<int32_t>(sg.get_group_linear_id());
                    const int32_t parts_per_sg = static_cast<int32_t>(partition.get_group_linear_range());
                    const int32_t part_id = sg_id * parts_per_sg + static_cast<int32_t>(partition.get_group_linear_id());
                    const int32_t lane = static_cast<int32_t>(partition.get_local_linear_id());
                    const int32_t prob_id = static_cast<int32_t>(wg_id) * static_cast<int32_t>(probs_per_wg) + part_id;
                    if (prob_id >= static_cast<int32_t>(batch_size)) return;
                    auto d_prob = d.batch_item(prob_id);
                    auto e_prob = e.batch_item(prob_id);

                    // Compile-time selectable eigenvector accumulation (shared-memory Q).
                    const int32_t base_q = part_id * static_cast<int32_t>(P) * static_cast<int32_t>(P);
                    using QLocalAccT = decltype(Q_local);
                    QSharedCache<T, P, ComputeVecs, QLocalAccT> qcache(Q_local, base_q, lane, n);

                    if constexpr (ComputeVecs) {
                        auto Q_prob = Q_view.batch_item(prob_id);
                        qcache.load(Q_prob);
                    }

                    // Load D/E into registers (one element per lane).
                    T diag = (lane < n) ? d_prob(lane) : T(0);
                    T offdiag = (lane < (n - 1)) ? e_prob(lane) : T(0);

                    // Defensive convergence budget to avoid unbounded looping on hard inputs.
                    // Each implicit step consumes one unit. Budget scales with problem size.
                    int32_t sweep_budget = static_cast<int32_t>(max_sweeps) * n;
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
                        // We scale the tridiagonal entries by `scale` during iteration and
                        // rescale back by `inv_scale` once the block converges.
                        //
                        // NOTE: reduce_over_group() for floating point on chunked partitions
                        // is not available on some backends (e.g. CUDA), so the max norm is
                        // computed with an XOR-shuffle butterfly over the register-resident
                        // tridiagonal entries.
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
                            // Scale down to avoid overflow.
                            scale = internal::ssfmax<T>() / anorm;
                        } else if (anorm < internal::ssfmin<T>() && anorm != T(0)) {
                            // Scale up to avoid underflow.
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

                        // Choose between QL and QR (matches steqr.cc):
                        // - QR if |D(l)| <= |D(lend)|
                        // - QL otherwise
                        const T d_first = std::abs(select_from_group(partition, diag, block_begin));
                        const T d_last = std::abs(select_from_group(partition, diag, block_end));
                        const bool use_ql = (d_last < d_first);
                        if (use_ql) {
                            // ---------------- QL iteration: converge from the top (l grows) ----------------
                            for (int32_t l = block_begin; l <= block_end && !failed;) {
                                if (l == block_end) {
                                    l += 1;
                                    continue;
                                }

                                // Iterate up to max_sweeps times to converge eigenvalue at position l.
                                bool advanced = false;
                                for (int32_t sweep = 0; sweep < static_cast<int32_t>(max_sweeps); ++sweep) {
                                    // Deflate within current active subproblem [l..lend].
                                    deflate(partition, offdiag, diag, n, l, block_end + 1, zero_threshold);

                                    // Find first m in [l..lend-1] such that E(m)==0; if none, m=lend.
                                    const int32_t m_candidate = (lane >= l && lane < block_end && offdiag == T(0)) ? lane : block_end;
                                    const int32_t m = partition_reduce_min<P>(partition, m_candidate);

                                    if (m == l) {
                                        // Converged! Move to next eigenvalue.
                                        l += 1;
                                        advanced = true;
                                        break;
                                    }

                                    if (m == l + 1) {
                                        // 2x2 block at (l,l+1).
                                        solve_2x2_and_update<T, P>(partition, diag, offdiag, l, /*ql=*/true, qcache);

                                        l += 2;
                                        advanced = true;
                                        break;
                                    }

                                    if (sweep_budget <= 0) {
                                        failed = true;
                                        break;
                                    }
                                    sweep_budget -= 1;

                                    // ---- Implicit QL step on subblock [l..m] (m>=l+2) ----
                                    implicit_ql_step<T, P>(partition, diag, offdiag, qcache, n, l, m, cta_shift_strategy, cta_update_scheme);
                                }  // end sweep loop for QL

                                if (!advanced) {
                                    // Did not converge within max_sweeps (or hit the sweep budget).
                                    failed = true;
                                }
                            }  // end QL l loop
                        } else {
                            // ---------------- QR iteration: converge from the bottom (l shrinks) ----------------
                            // Use signed indices for the descending loop to avoid unsigned underflow.
                            for (int32_t l = static_cast<int32_t>(block_end);
                                 l >= static_cast<int32_t>(block_begin);
                                 /* manual step */) {
                                if (failed) break;
                                if (l == static_cast<int32_t>(block_begin)) {
                                    break;
                                }

                                // Iterate up to max_sweeps times to converge eigenvalue at position l.
                                bool advanced = false;
                                for (int32_t sweep = 0; sweep < static_cast<int32_t>(max_sweeps); ++sweep) {
                                    deflate(partition, offdiag, diag, n, block_begin, static_cast<int32_t>(l) + 1, zero_threshold);

                                    // Find m scanning downward: look for E(i)==0 and take the largest i+1.
                                    const int32_t l_u = static_cast<int32_t>(l);
                                    const int32_t m_candidate =
                                        (lane >= block_begin && lane < l_u && offdiag == T(0)) ? (lane + 1) : block_begin;
                                    const int32_t m = partition_reduce_max<P>(partition, m_candidate);

                                    if (m == l) {
                                        // Converged! Move to next eigenvalue.
                                        l -= 1;
                                        advanced = true;
                                        break;
                                    }

                                    if (m + 1 == l) {
                                        // 2x2 block at (l-1,l).
                                        const size_t l0 = l_u - 1;
                                        solve_2x2_and_update<T, P>(partition, diag, offdiag, l0, /*ql=*/false, qcache);

                                        if (l <= 1) {
                                            l = static_cast<int32_t>(block_begin);
                                        } else {
                                            l -= 2;
                                        }
                                        advanced = true;
                                        break;
                                    }


                                    if (sweep_budget <= 0) {
                                        failed = true;
                                        break;
                                    }
                                    sweep_budget -= 1;

                                    // ---- Implicit QR step on subblock [m..l] ----
                                    implicit_qr_step<T, P>(partition, diag, offdiag, qcache, n, m, l_u, cta_shift_strategy, cta_update_scheme);
                                }  // end sweep loop for QR

                                if (!advanced) {
                                    failed = true;
                                    break;
                                }
                            }  // end QR l loop
                        }  // end if (do_ql) else

                        // Rescale converged block back to the original magnitude.
                        if (scale != T(1)) {
                            if (lane >= block_begin && lane <= block_end) {
                                diag *= inv_scale;
                            }
                            if (lane >= block_begin && lane < block_end) {
                                offdiag *= inv_scale;
                            }
                        }

                        if (failed) {
                            // Mark the problem as failed to converge.
                            // We cannot throw from device code; host can decide how to handle it.
                            if (lane == 0 && status) {
                                status[prob_id] = 1;
                            }
                            break;
                        }
                    }  // end outer block split loop

                    // Store back D/E (one element per lane).
                    if (lane < n) {
                        d_prob(lane) = diag;
                    }
                    if (lane < (n - 1)) {
                        e_prob(lane) = offdiag;
                    }

                    if constexpr (ComputeVecs) {
                        auto Q_prob = Q_view.batch_item(prob_id);
                        qcache.store(Q_prob);
                    }
                });
        });

    }

    template <Backend B, typename T>
    Event steqr_cta(Queue& ctx, const VectorView<T>& d_in, const VectorView<T>& e_in,
                    const VectorView<T>& eigenvalues, const Span<std::byte>& ws,
                    JobType jobz, SteqrParams<T> params,
                    const MatrixView<T, MatrixFormat::Dense>& eigvects) {
        BATCHLAS_KERNEL_TRACE_SCOPE("steqr_cta");
        if (eigvects.rows() != eigvects.cols()) {
            throw std::invalid_argument("Matrix must be square for eigenvalue computation.");
        }
        if (jobz == JobType::EigenVectors && !params.back_transform) {
            eigvects.fill_identity(ctx);
        }

        const int64_t n = d_in.size();
        const int64_t batch_size = d_in.batch_size();
        auto pool = BumpAllocator(ws);

        const auto increment = params.transpose_working_vectors ? batch_size : 1;
        const auto d_stride = params.transpose_working_vectors ? 1 : n;
        const auto e_stride = params.transpose_working_vectors ? 1 : n - 1;

        auto d = VectorView<T>(pool.allocate<T>(ctx, VectorView<T>::required_span_length(n, increment, d_stride, batch_size)),
                               n, batch_size, increment, d_stride);
        auto e = VectorView<T>(pool.allocate<T>(ctx, VectorView<T>::required_span_length(n - 1, increment, e_stride, batch_size)),
                               n - 1, batch_size, increment, e_stride);

        auto status = pool.allocate<int32_t>(ctx, std::max<int64_t>(int64_t(1), batch_size)).data();
        ctx->memset(status, 0, sizeof(int32_t) * static_cast<size_t>(std::max<int64_t>(int64_t(1), batch_size)));

        VectorView<T>::copy(ctx, d, d_in);
        VectorView<T>::copy(ctx, e, e_in);

        auto& eigvects_mut = const_cast<MatrixView<T, MatrixFormat::Dense>&>(eigvects);

        // CTA backend: choose an optimal compile-time partition size P in {4,8,16,32}.
        // Requires warp-sized sub-groups (32) on NVIDIA.
        if (n < 1 || n > 32) {
            throw std::invalid_argument("steqr_cta currently supports 1 <= n <= 32.");
        }

        const auto dev = ctx->get_device();
        bool has32 = false;
        {
            const auto sg_sizes = dev.get_info<sycl::info::device::sub_group_sizes>();
            for (auto sgs : sg_sizes) {
                if (static_cast<int32_t>(sgs) == 32) {
                    has32 = true;
                    break;
                }
            }
        }

        if (!has32) {
            return steqr_wg<B, T>(ctx, d_in, e_in, eigenvalues, ws, jobz, params, eigvects);
        }

        const int32_t n_i32 = static_cast<int32_t>(n);

        auto launch = [&](auto P_tag) {
            constexpr int32_t P = decltype(P_tag)::value;
            if (jobz == JobType::EigenVectors) {
                steqr_cta_impl<T, P, true>(ctx, d, e, eigvects_mut, n_i32,
                                           params.max_sweeps, params.zero_threshold,
                                           params.cta_shift_strategy, params.cta_update_scheme, params.cta_wg_size_multiplier,
                                           status,
                                           pool);
            } else {
                steqr_cta_impl<T, P, false>(ctx, d, e, eigvects_mut, n_i32,
                                            params.max_sweeps, params.zero_threshold,
                                            params.cta_shift_strategy, params.cta_update_scheme, params.cta_wg_size_multiplier,
                                            status,
                                            pool);
            }
        };

        if (n_i32 <= 4) {
            launch(std::integral_constant<int32_t, 4>{});
        } else if (n_i32 <= 8) {
            launch(std::integral_constant<int32_t, 8>{});
        } else if (n_i32 <= 16) {
            launch(std::integral_constant<int32_t, 16>{});
        } else {
            launch(std::integral_constant<int32_t, 32>{});
        }

        // Copy back eigenvalues.
        VectorView<T>::copy(ctx, eigenvalues, d);

        // Optional fail-fast diagnostics: avoids silent non-convergence.
        // Note: checking requires synchronization, so keep it opt-in.
        if (const char* v = std::getenv("BATCHLAS_STEQR_CTA_CHECK")) {
            const bool enabled = (v[0] == '1') || (v[0] == 't') || (v[0] == 'T') || (v[0] == 'y') || (v[0] == 'Y');
            if (enabled) {
                ctx.wait();
                for (int64_t i = 0; i < batch_size; ++i) {
                    if (status[i] != 0) {
                        throw std::runtime_error("steqr_cta: failed to converge within sweep budget.");
                    }
                }
            }
        }

        if (params.sort) {
            auto ws_sort = pool.allocate<std::byte>(ctx, sort_buffer_size<T>(ctx, eigenvalues.data(), eigvects, jobz));
            sort(ctx, eigenvalues, eigvects, jobz, params.sort_order, ws_sort);
        }

        return ctx.get_event();
    }

    template <typename T>
    size_t steqr_cta_buffer_size(Queue& ctx, const VectorView<T>& d, const VectorView<T>& e,
                                 const VectorView<T>& eigenvalues, JobType jobz, SteqrParams<T> params) {
        const auto n = d.size();
        const auto batch_size = d.batch_size();
        const auto d_stride = d.stride() > 0 ? d.stride() : n * d.inc();
        const auto e_stride = e.stride() > 0 ? e.stride() : (n - 1) * e.inc();
        const auto d_size = VectorView<T>::required_span_length(n, d.inc(), d_stride, batch_size);
        const auto e_size = VectorView<T>::required_span_length(n - 1, e.inc(), e_stride, batch_size);

        size_t size = BumpAllocator::allocation_size<T>(ctx, d_size)
                + BumpAllocator::allocation_size<T>(ctx, e_size);

        // steqr_cta allocates a per-problem status array (int32_t) for non-convergence tracking.
        size += BumpAllocator::allocation_size<int32_t>(ctx, std::max<int64_t>(int64_t(1), batch_size));

        size += sort_buffer_size<T>(ctx, eigenvalues.data(),
                                    MatrixView<T, MatrixFormat::Dense>(nullptr, n, n, n, n * n, batch_size), jobz);

        const auto dev = ctx->get_device();
        bool has32 = false;
        {
            const auto sg_sizes = dev.get_info<sycl::info::device::sub_group_sizes>();
            for (auto sgs : sg_sizes) {
                if (static_cast<int32_t>(sgs) == 32) {
                    has32 = true;
                    break;
                }
            }
        }

        if (!has32) {
            const auto steqr_size = steqr_wg_buffer_size<T>(ctx, d, e, eigenvalues, jobz, params);
            size = std::max<size_t>(size, steqr_size);
        }
        return size;
    }

#define STEQR_CTA_INSTANTIATE(back, fp) \
    template Event steqr_cta<back, BATCHLAS_UNPAREN fp>(Queue&, const VectorView<BATCHLAS_UNPAREN fp>&, const VectorView<BATCHLAS_UNPAREN fp>&, const VectorView<BATCHLAS_UNPAREN fp>&, const Span<std::byte>&, JobType, SteqrParams<BATCHLAS_UNPAREN fp>, const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&);

#define STEQR_CTA_INSTANTIATE_FOR_BACKEND(back) \
    BATCHLAS_FOR_EACH_REAL_TYPE_1(STEQR_CTA_INSTANTIATE, back)

#if BATCHLAS_HAS_CUDA_BACKEND
    STEQR_CTA_INSTANTIATE_FOR_BACKEND(Backend::CUDA)
#endif

#if BATCHLAS_HAS_ROCM_BACKEND
    STEQR_CTA_INSTANTIATE_FOR_BACKEND(Backend::ROCM)
#endif

#if BATCHLAS_HAS_HOST_BACKEND
    STEQR_CTA_INSTANTIATE_FOR_BACKEND(Backend::NETLIB)
#endif

    template size_t steqr_cta_buffer_size<float>(Queue&, const VectorView<float>&, const VectorView<float>&, const VectorView<float>&, JobType, SteqrParams<float>);
    template size_t steqr_cta_buffer_size<double>(Queue&, const VectorView<double>&, const VectorView<double>&, const VectorView<double>&, JobType, SteqrParams<double>);

    #undef STEQR_CTA_INSTANTIATE_FOR_BACKEND
    #undef STEQR_CTA_INSTANTIATE
}
