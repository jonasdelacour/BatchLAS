// SPDX-License-Identifier: MIT
#pragma once

// Device-side CTA (sub-group-partition) STEQR solver, shared by steqr_cta.cc and
// syev_cta_fused.cc. Keep ONE definition: it is what makes a fused-vs-partitioned
// benchmark measure fusion and nothing else. The phases live in steqr_cta_lanes.hh; the
// three drivers here only differ in loop shape.
// evidence: docs/perf/steqr.md#cta-steqr-chase-micro-structure-decisions

#include <batchlas/blas/matrix.hh>
#include <batchlas/blas/extensions.hh>
#include <batchlas/util/group-invoke.hh>
#include <src/kernels/common/sg_partition/sg_partition.hh>
#include <src/math-helpers.hh>
#include <src/ops/syev/steqr/steqr_cta_lanes.hh>
#include <src/ops/syev/tridiag/plane_rotation.hh>
#include <src/sycl/kernel_attrs.hh>

#include <cstdint>
#include <type_traits>

namespace batchlas {

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
            const int32_t i0 = base_q + lane + col0 * pN;
            const int32_t i1 = base_q + lane + col1 * pN;
            const T q0 = Q_local[i0];
            const T q1 = Q_local[i1];
            Q_local[i0] = c * q0 - s * q1;
            Q_local[i1] = s * q0 + c * q1;
        }

        // [z_j, z_k] <- [z_j, z_k] R^T on this lane's row of Z.
        inline void rotate(int32_t j, int32_t k, tridiag::Givens<T> R) { apply(k, j, R.c, R.s); }

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
        inline void rotate(int32_t, int32_t, tridiag::Givens<T>) {}
        inline void chase_begin(int32_t) {}
        template <int32_t Dir>
        inline void chase_step(T, T) {}
        template <int32_t Dir>
        inline void chase_step_masked(T, T, bool) {}
        inline void chase_end(int32_t) {}
        inline void reverse_columns(int32_t, int32_t, bool) {}
    };

    // STEQR on one chunk: split T into unreduced blocks; per block, scale and orient, then
    // for each window [l, m] converge d_l, solve an exact 2x2, or sweep; mirror back and
    // unscale. Z accumulates every rotation (a no-op without vectors). True if a block failed.
    template <typename T, size_t P, typename Part, typename QCache>
    __attribute__((always_inline)) inline bool steqr_cta_solve_nested(tridiag::TridiagLanes<T, P, Part>& t,
                                                                      QCache& qcache,
                                                                      int32_t max_sweeps,
                                                                      SteqrShiftStrategy shift_strategy,
                                                                      SteqrUpdateScheme update_scheme) {
        using namespace tridiag;
        // Each sweep consumes one unit of a budget that scales with n.
        int32_t sweep_budget = max_sweeps * t.n;
        bool failed = false;

        for (int32_t next_begin = 0; next_begin < t.n;) {
            const Range block = find_block(t, next_begin);
            next_begin = block.last + 1;
            if (block.last <= block.first) continue;  // a 1x1 block is an eigenvalue

            const OpenBlock<T> open = open_block(t, qcache, block);
            for (int32_t l = block.first; l <= block.last && !failed;) {
                if (l == block.last) {
                    l += 1;
                    continue;
                }

                bool advanced = false;
                for (int32_t sweep = 0; sweep < max_sweeps; ++sweep) {
                    const Range window{l, block.last};
                    mark_negligible(t, window);
                    const int32_t m = first_split(t, window);
                    const WindowAction action = classify(l, m, sweep_budget);

                    if (action == WindowAction::Converged1) {
                        l += 1;
                        advanced = true;
                        break;
                    }
                    if (action == WindowAction::Exact2x2) {
                        solve_exact_2x2(t, qcache, l);
                        l += 2;
                        advanced = true;
                        break;
                    }
                    if (action == WindowAction::OutOfBudget) {
                        failed = true;
                        break;
                    }
                    sweep_budget -= 1;
                    implicit_ql_sweep<false>(t, qcache, Range{l, m}, shift_strategy, update_scheme);
                }

                if (!advanced) {
                    failed = true;
                }
            }
            close_block(t, qcache, open);

            if (failed) break;
        }
        return failed;
    }

    // steqr_cta_solve_nested with the sweep hoisted out of the loop nest, for a partition
    // whose collectives are chunk-local (a masked SubGroupPartition): each chunk settles
    // until it has a sweep or is done, then every chunk with a sweep chases in the same
    // pass. Same per-chunk operation sequence as the nested solver: bitwise equal.
    // evidence: docs/perf/steqr.md#lockstep-flat-solver
    template <typename T, size_t P, typename Part, typename QCache>
    __attribute__((always_inline)) inline bool steqr_cta_solve_flat(tridiag::TridiagLanes<T, P, Part>& t,
                                                                    QCache& qcache,
                                                                    int32_t max_sweeps,
                                                                    SteqrShiftStrategy shift_strategy,
                                                                    SteqrUpdateScheme update_scheme) {
        using namespace tridiag;
        int32_t sweep_budget = max_sweeps * t.n;
        int32_t next_begin = 0;
        int32_t l = 0;
        int32_t m = 0;
        int32_t sweeps_at_l = 0;
        OpenBlock<T> open{};
        bool need_block = true;
        bool done = false;
        bool failed = false;

        while (!done) {
            // Settle: every exit of this loop is either "sweep [l, m] next" or done,
            // so the chunks of a warp reconverge right before the chase.
            for (;;) {
                if (need_block) {
                    const Range block = find_block(t, next_begin);
                    next_begin = block.last + 1;
                    if (block.last <= block.first) {
                        if (next_begin >= t.n) {
                            done = true;
                            break;
                        }
                        continue;
                    }
                    open = open_block(t, qcache, block);
                    l = block.first;
                    sweeps_at_l = 0;
                    need_block = false;
                }

                // The sweep cap comes first: the nested loop gives up after its last
                // sweep without deflating or searching for m again.
                bool closing = false;
                if (sweeps_at_l >= max_sweeps) {
                    failed = true;
                    closing = true;
                } else {
                    const Range window{l, open.block.last};
                    mark_negligible(t, window);
                    m = first_split(t, window);
                    const WindowAction action = classify(l, m, sweep_budget);
                    if (action == WindowAction::Converged1) {
                        l += 1;
                    } else if (action == WindowAction::Exact2x2) {
                        solve_exact_2x2(t, qcache, l);
                        l += 2;
                    } else if (action == WindowAction::OutOfBudget) {
                        failed = true;
                    } else {
                        break;
                    }
                    sweeps_at_l = 0;
                    closing = failed || l >= open.block.last;
                }

                if (closing) {
                    close_block(t, qcache, open);
                    need_block = true;
                    if (failed || next_begin >= t.n) {
                        done = true;
                        break;
                    }
                }
            }
            if (done) break;

            implicit_ql_sweep<false>(t, qcache, Range{l, m}, shift_strategy, update_scheme);
            sweep_budget -= 1;
            sweeps_at_l += 1;
        }
        return failed;
    }

    // The same state machine when the lockstep domain is the whole sub-group: every phase
    // runs under a sub-group vote, and a chunk outside the phase passes `live` false: it
    // feeds in-range sources and discards the results.
    template <typename T, size_t P, typename Part, typename QCache>
    __attribute__((always_inline)) inline bool steqr_cta_solve_lockstep(tridiag::TridiagLanes<T, P, Part>& t,
                                                                        QCache& qcache,
                                                                        int32_t max_sweeps,
                                                                        SteqrShiftStrategy shift_strategy,
                                                                        SteqrUpdateScheme update_scheme) {
        using namespace tridiag;
        const Part& part = t.part;
        int32_t sweep_budget = max_sweeps * t.n;
        int32_t next_begin = 0;
        int32_t l = 0;
        int32_t m = 0;
        int32_t sweeps_at_l = 0;
        OpenBlock<T> open{};
        bool need_block = true;
        bool ready = false;  // a sweep on [l, m] is pending
        bool done = false;
        bool failed = false;

        while (lockstep_any(part, !done)) {
            while (lockstep_any(part, !done && !ready)) {
                const bool opening = !done && !ready && need_block;
                if (lockstep_any(part, opening)) {
                    const Range block = find_block(t, opening ? next_begin : 0, opening);
                    // A 1x1 block is skipped: stay in `need_block` for the next one.
                    const bool opened = opening && block.last > block.first;
                    const OpenBlock<T> fresh = open_block(t, qcache, block, opened);
                    next_begin = opening ? block.last + 1 : next_begin;
                    if (opened) {
                        open = fresh;
                        l = block.first;
                        sweeps_at_l = 0;
                        need_block = false;
                    }
                    done = done || (opening && !opened && block.last + 1 >= t.n);
                }

                // The sweep cap comes first: the nested loop gives up after its last
                // sweep without deflating or searching for m again.
                const bool testing = !done && !ready && !need_block;
                const bool cap = testing && sweeps_at_l >= max_sweeps;
                const bool active = testing && !cap;
                if (lockstep_any(part, active)) {
                    const Range window{active ? l : t.n, open.block.last};
                    mark_negligible(t, window);
                    const int32_t mm = first_split(t, window);
                    const WindowAction action = classify(l, mm, sweep_budget);
                    const bool adv1 = active && action == WindowAction::Converged1;
                    const bool adv2 = active && action == WindowAction::Exact2x2;
                    const bool sweep = active && action == WindowAction::Sweep;
                    failed = failed || (active && action == WindowAction::OutOfBudget);
                    m = sweep ? mm : m;
                    ready = ready || sweep;

                    if (lockstep_any(part, adv2)) {
                        const int32_t l0 = sycl::clamp(l, int32_t(0), static_cast<int32_t>(P) - 2);
                        solve_exact_2x2(t, qcache, l0, adv2);
                    }
                    l += adv1 ? 1 : (adv2 ? 2 : 0);
                    sweeps_at_l = (adv1 || adv2) ? 0 : sweeps_at_l;
                }
                failed = failed || cap;

                const bool closing = testing && (l >= open.block.last || failed);
                if (lockstep_any(part, closing)) {
                    close_block(t, qcache, open, closing);
                    need_block = need_block || closing;
                    done = done || (closing && (failed || next_begin >= t.n));
                }
            }

            // Every chunk with a pending sweep chases in the same pass.
            if (lockstep_any(part, ready)) {
                implicit_ql_sweep<true>(t, qcache, Range{l, m}, shift_strategy, update_scheme, ready);
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

    // Returns true if any block failed to converge within budget. Lane i holds d_i in `diag`
    // and e_i in `offdiag`, on entry and on exit.
    //
    // A full-sub-group (emulated) partition promises every lane of the sub-group runs
    // this. P == 32 runs the nested loops maskless; smaller chunks on NVPTX take the
    // chunk-masked partition (every maskless form measured slower there); elsewhere EXP
    // takes the lockstep solver, while PG's nested loops are legal only while the chunks
    // happen not to diverge. Judge changes here by real-kernel A/B, not microbenchmarks.
    // evidence: docs/perf/steqr.md#full-warp-partition
    template <typename T, size_t P, typename Partition, typename QCache>
    __attribute__((always_inline)) inline bool steqr_cta_solve(const Partition& partition,
                                                               T& diag,
                                                               T& offdiag,
                                                               QCache& qcache,
                                                               int32_t n,
                                                               int32_t max_sweeps,
                                                               SteqrShiftStrategy cta_shift_strategy,
                                                               SteqrUpdateScheme cta_update_scheme) {
        if constexpr (lockstep_spans_subgroup_v<Partition> && P < 32 && batchlas::sgp::kMaskedByDefault) {
            const auto chunk = make_partition<P, true>(partition.sg);
            return steqr_cta_solve<T, P>(chunk, diag, offdiag, qcache, n, max_sweeps,
                                         cta_shift_strategy, cta_update_scheme);
        } else {
            tridiag::TridiagLanes<T, P, Partition> t(partition, n, diag, offdiag);
            bool failed = false;
            if constexpr (lockstep_spans_subgroup_v<Partition> && P < 32) {
                if (cta_update_scheme == SteqrUpdateScheme::EXP) {
                    failed = steqr_cta_solve_lockstep(t, qcache, max_sweeps, cta_shift_strategy, cta_update_scheme);
                } else {
                    failed = steqr_cta_solve_nested(t, qcache, max_sweeps, cta_shift_strategy, cta_update_scheme);
                }
            } else if constexpr (kSteqrCtaFlatPays<T, P>) {
                failed = steqr_cta_solve_flat(t, qcache, max_sweeps, cta_shift_strategy, cta_update_scheme);
            } else {
                failed = steqr_cta_solve_nested(t, qcache, max_sweeps, cta_shift_strategy, cta_update_scheme);
            }
            diag = t.d;
            offdiag = t.e;
            return failed;
        }
    }

} // namespace batchlas
