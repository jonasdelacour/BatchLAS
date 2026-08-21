#pragma once

// The native batched POTRF CTA kernel -- ALL of its device code.
//
// One matrix is staged whole into local memory and factorised there by a
// right-looking blocked recurrence; global memory is touched exactly twice, once
// to load the triangle and once to store the factor.
//
// READ WP4_POTRF_SPEC_CORRECTIONS.md BEFORE WP4_POTRF_SPEC.md. The spec predates
// WP0-WP3 and where they disagree the corrections win. The places that matter to
// this file are called out where they bite.
//
// ---------------------------------------------------------------------------
// THE TWO DEFECTS THIS SHAPE EXISTS TO AVOID
// ---------------------------------------------------------------------------
//
// 1. THE STALE PIVOT. The obvious (P1) reads the pivot from the local tile,
//    S(j+k, j+k). That word is written by the tile load and then only at
//    iteration k, AFTER the read -- so it is the ORIGINAL diagonal, not the
//    updated Schur diagonal a_kk - sum_{p<k} |l_kp|^2. Every column from 1 on
//    comes out wrong and `info` reports 0 for any matrix whose first failing
//    leading minor is > 1, which is every realistic Gram-matrix failure. The
//    updated value is not missing, it is in the WRONG LANE: lane k's register
//    d[k] holds it. potrf_diag_block_subgroup broadcasts it with one
//    select_from_group per column. tests/potrf_tests.cc's InfoIndexIsExact is
//    the test that fails if anyone reverts this, and it was PROVEN to fail by
//    reintroducing exactly this line.
//
// 2. THE SCOPE MISMATCH. `Scope` is a template parameter and the phase barriers
//    follow it: sub-group barriers when one 32-wide sub-group owns a matrix,
//    work-group barriers when 2 or 4 sub-groups do. Writing group_barrier(sg)
//    in the second case makes (P1)->(P2)->(P3) straight races on the tile, with
//    no crash and a plausible wrong factor. Scope is DERIVED by
//    potrf_cta_launch_params (potrf_cta.cc) and asserted there, never chosen by
//    hand -- WP4_POTRF_SPEC.md:225 asserts a scope for the blocked leaf that
//    contradicts its own L ladder at :189-195 for float, which is W10.
//
//    The one place the two barriers are NOT interchangeable and the difference
//    is invisible in the source: (P1)'s internal per-k barrier is ALWAYS
//    group_barrier(sub_group). Under Scope::WorkGroup (P1) runs inside
//    `if (sg_id == 0)`, so a work-group barrier there is a barrier in divergent
//    control flow -- undefined behaviour, and it deadlocks or passes depending
//    on the day.
//
// ---------------------------------------------------------------------------
// BARRIER AUDIT -- every site, both scopes
// ---------------------------------------------------------------------------
//   E   early return for matrix_id >= batch, BEFORE B0.  Sub-group-uniform under
//       SubGroup (matrix_id = wg*G + sg_id); cannot fire under WorkGroup
//       (G == 1 => num_wg == batch).
//   B0  after the fused load + off[] writer + diag[] zero + *fail = 0.  Top
//       level of the body, reached by every surviving work-item.
//   B1' inside (P1), once per k.  ALWAYS group_barrier(sub_group).  See above.
//   B1  after (P1), OUTSIDE the `if (p1_active)`.  Top level of the panel loop.
//   B2  after (P2), OUTSIDE the `if (ok && m2 > 0)`.  Top level.
//   B3  after (P3), OUTSIDE the same predicate, before the next panel.
//
// Three structural rules that make that audit true:
//   * the panel loop trip count is uniform (n and NB are kernel-uniform) and the
//     failure path is a PREDICATED SKIP, never a `break` or `return`, so B2/B3
//     are reached the same number of times on the failure and success paths;
//   * B2 and B3 fire unconditionally including on the last panel where m2 == 0,
//     where they are no-ops -- and B3 of the last panel is also what separates
//     the final (P1) write from the store-back loop.  There is no B4;
//   * (P2) and (P3) contain no internal barriers: (P2) reads L11 (published
//     before B1) and writes the A21 panel; (P3) reads L21 (published before B2)
//     and writes A22.  Disjoint ranges in both cases.
//
// ---------------------------------------------------------------------------
// W9 RESOLVED -- off[], the (P3) tile-index prefix table
// ---------------------------------------------------------------------------
// W9 records that off[] had no term in the SLM formula, no specified writer and
// no barrier.  All three, decided here:
//
//   SIZED    one copy PER MATRIX, Rt0 + 1 int32 entries, where
//            Rt0 = ceil_div(n - NB, TS) is the tile-row count of the FIRST
//            trailing update -- the largest over all panels.  It is a term in
//            slm_per_matrix in potrf_cta.cc, so the fit ceiling accounts for it.
//            Per-matrix rather than per-work-group (the alternative W9 offers)
//            because at G > 1 a shared table would be written concurrently by
//            every sub-group and read across sub-group barriers that do not
//            order those writes; per-matrix removes the question instead of
//            answering it, and at G == 1 -- which is where potrf_cta_max_n is
//            evaluated -- the two are numerically identical.
//            The +1 entry is the sentinel off[Rt] = Rt(Rt+1)/2; without it the
//            search's upper bound is an off-by-one that reads past the table.
//   WRITTEN  by this matrix's own tid loop, before B0, in the same pre-panel
//            region as the tile load.
//   SYNCED   by B0.  No other barrier, and in particular NO PER-PANEL REWRITE:
//            W9's "must be rewritten at the start of every panel" is the half of
//            it that is wrong, because a per-panel rewrite needs a barrier
//            between the write and the search and there is no slot for one.
//            Instead the table is built once for Rt0 and the search RESCALES:
//            off_j[c] = c*Rt_j - c(c-1)/2 = off[c] - c*(Rt0 - Rt_j), so the
//            per-panel dR = Rt0 - Rt_j enters as one integer multiply-subtract
//            per probe (at most 5 probes, amortised over ib*TS^2 >= 64 FMAs).
//
// ---------------------------------------------------------------------------
// REGISTER RESIDENCY
// ---------------------------------------------------------------------------
// d[NB], x[NB] and acc[TS][TS] are declared INSIDE the function that fills them
// and are never parameters, by reference or otherwise: taking the address of an
// accumulator array spills it, measured at 43% once in this tree. Every index
// into them is a fully-unrolled loop counter, which is what keeps them out of
// .local -- a dynamic index into one of these arrays puts the whole array in
// local memory with ZERO reported spill, so the register gate for this kernel is
// the three-condition one (stack frame == 0 AND 0 spill AND regs*WG <= 65536).

#include "../sycl/device_scalar.hh"

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>

namespace batchlas::potrf_native {

enum class PotrfScope { SubGroup, WorkGroup };

// The phase barrier, scope-parametric. See the audit above for why (P1)'s
// internal barrier does NOT go through this.
template <PotrfScope SC>
inline void potrf_phase_barrier(const sycl::nd_item<1>& it) {
    if constexpr (SC == PotrfScope::SubGroup) {
        sycl::group_barrier(it.get_sub_group());
    } else {
        sycl::group_barrier(it.get_group());
    }
}

// ---------------------------------------------------------------------------
// (P1) -- the ib x ib diagonal block, one lane per row, on ONE sub-group.
//
// Lane r owns row j+r of the block in registers d[0..NB). diag[] and *fail live
// in local memory. There is no rinv[] array: (P1) scales a COLUMN by a
// reciprocal (which is what reference ?potf2's sscal(1/ajj) does) and (P2)
// DIVIDES (which is what reference ?trsm does); a shared precomputed reciprocal
// would change (P2)'s rounding away from its reference.
// ---------------------------------------------------------------------------
template <typename D, typename R, int NB>
inline void potrf_diag_block_subgroup(const sycl::sub_group& sg,
                                      D* __restrict S, int lda,
                                      int j, int ib,
                                      R* __restrict diag,
                                      int* __restrict fail) {
    const int lane = static_cast<int>(sg.get_local_linear_id());

    // Row `lane` of the block, lower triangle only. Out-of-range entries are
    // ZEROED rather than left indeterminate: lanes ib..31 still take part in
    // every select_from_group below, and reading an uninitialised register is
    // UB even when the value is discarded.
    D d[NB];
#pragma unroll
    for (int c = 0; c < NB; ++c) {
        d[c] = (lane < ib && c < ib && c <= lane)
                   ? S[(j + lane) + static_cast<std::ptrdiff_t>(j + c) * lda]
                   : D{};
    }

    // NO `break` IN THIS LOOP, AND THAT IS A REGISTER-RESIDENCY DECISION.
    //
    // The natural spelling is `if (k >= ib) break;` plus a second `break` on the
    // failure test. Both are uniform and both are correct -- and both make the
    // trip count data-dependent, so `#pragma unroll` FAILS
    // (-Wpass-failed=transform-warning: "loop not unrolled") and `d[k]` / `d[c]`
    // acquire a dynamic index. Measured consequence: at NB = 16 with an 8-byte
    // scalar, double and complex<float> reported a 128-byte stack frame -- byte
    // for byte d[16] -- with zero spill, which is an array that was never in
    // registers to be spilled out of.
    //
    // Predicating instead of breaking costs NB - ib extra sub-group barriers on
    // the ragged last panel only, and they are uniform across the sub-group, so
    // no work-item is stranded. `alive` and `active` are sub-group-uniform by
    // construction: ib is a uniform argument and akk is a shuffled value.
    bool alive = true;
#pragma unroll
    for (int k = 0; k < NB; ++k) {
        // THE FIX FOR THE STALE PIVOT. The updated Schur diagonal lives in lane
        // k's register d[k]; the tile still holds the ORIGINAL value. Shuffle
        // the REAL part, not the scalar: only real_part(d[k]) is ever consumed,
        // so for complex D this is one 32-bit shuffle instead of two and no
        // aggregate crosses a group algorithm.
        //
        // Executed for k >= ib too, where d[k] is the zero this function
        // initialised it to -- never garbage.
        const R akk = sycl::select_from_group(sg, sycl_device::dev_real(d[k]),
                                              static_cast<uint32_t>(k));

        bool active = alive && (k < ib);

        // `!(akk > 0)` and not `akk <= 0`: this spelling also catches NaN, which
        // is LAPACK's AJJ.LE.ZERO .OR. SISNAN(AJJ). It precedes the sqrt and the
        // reciprocal, so a non-PD item executes neither -- that is what makes a
        // failed item's tile finite and bounded by the input rather than NaN.
        if (active && !(akk > R(0))) {
            if (lane == 0) *fail = j + k + 1;  // 1-based GLOBAL column
            active = false;
            alive = false;                     // sticky: FIRST FAILURE WINS
        }

        // The `active ? ... : 1` is what keeps sqrt off a negative argument on
        // the inactive path -- the guard above is a predicate now, not a branch
        // out of the loop, so the arithmetic below is still emitted.
        const R dkk = active ? sycl::sqrt(akk) : R(1);
        const R r = R(1) / dkk;  // NOT rsqrt: rsqrt.approx is not the reference

        if (active) {
            if (lane == k) {
                d[k] = sycl_device::dev_from_real<D>(dkk);
            } else if (lane > k && lane < ib) {
                // `lane < ib` is not decoration. Without it, lanes ib..31 read
                // and write registers that were never initialised for a value
                // that is never published -- and the same missing guard on the
                // publish below is a live wrong-answer bug, so the two are kept
                // symmetric.
                d[k] = sycl_device::dev_mul_real(d[k], r);
            }

            // PUBLISH COLUMN k. The `lane < ib` here is [FIX-A1.4]: unguarded,
            // lanes ib..31 write S(j+ib .. j+31, j+k) -- into the A21 panel (P2)
            // is about to read, on EVERY panel where ib < 32, and past the tile
            // entirely on the ragged last panel, landing in a NEIGHBOURING
            // MATRIX when G > 1. tests/potrf_tests.cc's PackedBatchMatchesSolo
            // is the test that catches that second half, and removing this guard
            // was executed and turned it red.
            if (lane < ib && lane >= k) {
                S[(j + lane) + static_cast<std::ptrdiff_t>(j + k) * lda] = d[k];
            }
            if (lane == 0) diag[k] = dkk;
        }

        // B1'. ALWAYS the sub-group, and UNCONDITIONAL -- it is outside the
        // `if (active)` so the barrier count does not depend on the failure
        // path. The update below reads column k from the tile, published one
        // line above by a DIFFERENT lane.
        sycl::group_barrier(sg);

        // No WAR hazard and therefore no second barrier: step k+1 publishes
        // column k+1 while this reads column k.
        if (active) {
#pragma unroll
            for (int c = k + 1; c < NB; ++c) {
                if (c < ib && lane < ib && lane >= c) {
                    d[c] = sycl_device::dev_sub(
                        d[c],
                        sycl_device::dev_mul(
                            d[k],
                            sycl_device::dev_conj(
                                S[(j + c) + static_cast<std::ptrdiff_t>(j + k) * lda])));
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// (P2) -- the panel solve L21 = A21 * L11^-H.
//
// Side::Right, Uplo::Lower, ConjTrans, Diag::NonUnit, alpha = 1. Every ROW of
// A21 is independent, so this is embarrassingly parallel over rows and contains
// no barrier. x[NB] is the row, resident in registers.
//
// NB is a template parameter and both loops are fully unrolled ON PURPOSE: with
// a runtime bound, x[] acquires a dynamic index and the whole array lands in
// local memory. That costs performance and not correctness, so no test catches
// it -- the register probe is the only instrument that sees it.
// ---------------------------------------------------------------------------
template <typename D, typename R, int NB>
inline void potrf_panel_solve_rows(int tid, int L,
                                   D* __restrict S, int lda,
                                   int j, int ib, int m2,
                                   const R* __restrict diag) {
    for (int row = tid; row < m2; row += L) {
        const int i = j + ib + row;

        D x[NB];
#pragma unroll
        for (int c = 0; c < NB; ++c) {
            x[c] = (c < ib) ? S[i + static_cast<std::ptrdiff_t>(j + c) * lda] : D{};
        }

#pragma unroll
        for (int c = 0; c < NB; ++c) {
            // A PREDICATE, never a `break`: a break would make the unrolled
            // loop's trip count data-dependent for no gain.
            if (c < ib) {
                D s = x[c];
#pragma unroll
                for (int p = 0; p < NB; ++p) {
                    if (p < c) {
                        s = sycl_device::dev_sub(
                            s,
                            sycl_device::dev_mul(
                                x[p],
                                sycl_device::dev_conj(
                                    S[(j + c) + static_cast<std::ptrdiff_t>(j + p) * lda])));
                    }
                }
                // A DIVIDE by a REAL, not a multiply by a reciprocal. Reference
                // ?trsm divides; diag[c] is real by construction (it is a sqrt
                // of a real), so this is two divides for complex D and not
                // Smith's algorithm.
                x[c] = sycl_device::dev_div_real(s, diag[c]);
            }
        }

#pragma unroll
        for (int c = 0; c < NB; ++c) {
            if (c < ib) S[i + static_cast<std::ptrdiff_t>(j + c) * lda] = x[c];
        }
    }
}

// ---------------------------------------------------------------------------
// (P3) -- the trailing update A22 -= L21 L21^H, TS x TS register tiles.
//
// The triangle is handled at TILE granularity: only the Rt(Rt+1)/2 tiles that
// intersect the lower triangle are visited at all, and the per-element
// `ra >= cb` guard then trims the diagonal tiles. Splitting a tile by rows
// instead (a "band split") is the trap recorded in the triangular-kernel design
// rules; this is the tile-granular form.
//
// `off` is the Rt0-based prefix table and `dR = Rt0 - Rt` rescales it to this
// panel -- see the W9 note at the top. No barrier.
// ---------------------------------------------------------------------------
template <typename D, typename R, int TS>
inline void potrf_trailing_tiles(int tid, int L,
                                 D* __restrict S, int lda,
                                 int j, int ib, int m2,
                                 int Rt, int dR,
                                 const int* __restrict off) {
    const int Ntiles = Rt * (Rt + 1) / 2;
    const int base = j + ib;

    for (int t = tid; t < Ntiles; t += L) {
        // Largest ct in [0, Rt) with off_j[ct] <= t. A binary search over an
        // integer prefix table, NOT sqrt(double(...)) plus a while-fixup: a
        // floating-point inversion of an integer map with a hand-written
        // correction loop is a latent wrong answer for no gain.
        int lo = 0, hi = Rt - 1;
        while (lo < hi) {
            const int mid = (lo + hi + 1) >> 1;
            if (off[mid] - mid * dR <= t) {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        const int ct = lo;
        const int rt = ct + (t - (off[ct] - ct * dR));
        const int r0 = rt * TS;
        const int c0 = ct * TS;

        D acc[TS][TS];
#pragma unroll
        for (int a = 0; a < TS; ++a) {
#pragma unroll
            for (int b = 0; b < TS; ++b) acc[a][b] = D{};
        }

        for (int k = 0; k < ib; ++k) {
            D va[TS], vb[TS];
#pragma unroll
            for (int a = 0; a < TS; ++a) {
                va[a] = (r0 + a < m2)
                            ? S[(base + r0 + a) + static_cast<std::ptrdiff_t>(j + k) * lda]
                            : D{};
            }
#pragma unroll
            for (int b = 0; b < TS; ++b) {
                vb[b] = (c0 + b < m2)
                            ? S[(base + c0 + b) + static_cast<std::ptrdiff_t>(j + k) * lda]
                            : D{};
            }
#pragma unroll
            for (int a = 0; a < TS; ++a) {
#pragma unroll
                for (int b = 0; b < TS; ++b) {
                    // acc += va * conj(vb). fma_acc is 1 FMA for real and 4 for
                    // complex, with no isnan branch and no __mulsc3 call --
                    // std::complex operator* would emit both.
                    //
                    // MEASURED AND REFUTED, recorded so nobody re-derives it:
                    // fma_acc takes its accumulator by REFERENCE, which is the
                    // shape of this tree's recorded 43% out-parameter spill. It
                    // is NOT what puts double's and complex<float>'s acc[4][4]
                    // in local memory here. Routing the update through a scalar
                    // temporary (`D t = acc[a][b]; fma_acc(t, ...); acc[a][b] = t;`)
                    // so the array's address is never taken produced a
                    // BYTE-IDENTICAL ptxas report for all eight instantiations:
                    // 110/106/144/128/201/188/128/112 registers, the same 128 B
                    // frames on the same two types. See the step 1.7 note in
                    // potrf_cta.cc.
                    sycl_device::fma_acc(acc[a][b], va[a], sycl_device::dev_conj(vb[b]));
                }
            }
        }

#pragma unroll
        for (int a = 0; a < TS; ++a) {
#pragma unroll
            for (int b = 0; b < TS; ++b) {
                const int ra = r0 + a;
                const int cb = c0 + b;
                if (ra < m2 && cb < m2 && ra >= cb) {
                    D v = sycl_device::dev_sub(
                        S[(base + ra) + static_cast<std::ptrdiff_t>(base + cb) * lda],
                        acc[a][b]);
                    // FORCE THE HERMITIAN DIAGONAL REAL. This is the line most
                    // easily omitted; the residual test barely sees it (the
                    // imaginary residue is O(eps) and the next sqrt takes the
                    // real part anyway), which is why there is a dedicated test
                    // asserting imag(diag(L)) == 0 exactly.
                    // FORCE THE HERMITIAN DIAGONAL REAL.
                    //
                    // DEFENCE IN DEPTH, AND MEASURED TO BE EXACTLY THAT.
                    // Removing this line turned NO test red, and the reason is
                    // structural rather than a gap in the suite: every diagonal
                    // entry of the OUTPUT is written by (P1)'s publish, which
                    // stores dev_from_real(sqrt(akk)) and therefore has an
                    // exactly zero imaginary part regardless; and the only
                    // consumer of a tile diagonal is (P1)'s pivot, which takes
                    // dev_real() of it. So the rounding residue this line
                    // discards is never read. It stays because the moment any
                    // future variant consumes a trailing-update diagonal
                    // directly -- the blocked driver's leaf boundary is the
                    // obvious candidate -- the residue becomes live, and because
                    // it costs one predicated move on 1/TS of the tiles.
                    if constexpr (sycl_device::dev_is_complex_v<D>) {
                        if (ra == cb) {
                            v = sycl_device::dev_from_real<D>(sycl_device::dev_real(v));
                        }
                    }
                    S[(base + ra) + static_cast<std::ptrdiff_t>(base + cb) * lda] = v;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The whole body for ONE matrix.
//
// S, diag, fail and off are already offset to THIS matrix; Ag is already offset
// to this matrix's global base. `upper` is a LOAD/STORE TRANSFORM and not a
// second algorithm: A = U^H U with U upper is the same recurrence on
// S(i,c) = conj(A(c,i)), so the factorisation compiles once per
// (D, NB, TS, Scope) rather than once per (..., Uplo).
// ---------------------------------------------------------------------------
template <typename D, typename R, int NB, int TS, PotrfScope SC>
inline void potrf_cta_body(const sycl::nd_item<1>& it,
                           const sycl::sub_group& sg,
                           int tid, int L, bool p1_active,
                           D* __restrict S, int lda,
                           R* __restrict diag,
                           int* __restrict fail,
                           int* __restrict off,
                           D* __restrict Ag, int ldg,
                           int n, bool upper) {
    const int m2_0 = (n > NB) ? (n - NB) : 0;
    const int Rt0 = (m2_0 + TS - 1) / TS;

    // ONE FUSED LOAD. Not a device::fill followed by a triangular load: the two
    // use different index maps, so a fill store can land after a load store and
    // silently zero a live element. Every element of the lda x n tile is written
    // exactly once here, with a triangle value or a zero -- including the
    // lda - n pad row, which the (P3) tile loop can read through its `< m2`
    // guards only because it is zero.
    for (int c = 0; c < n; ++c) {
        for (int i = tid; i < lda; i += L) {
            D v{};
            if (i < n && i >= c) {
                v = upper ? sycl_device::dev_conj(Ag[c + static_cast<std::ptrdiff_t>(i) * ldg])
                          : Ag[i + static_cast<std::ptrdiff_t>(c) * ldg];
                // The diagonal is loaded as (real(A(c,c)), 0), matching
                // LAPACK/cuSOLVER: "imaginary parts of the diagonal need not be
                // set and are assumed zero". Discarding the residue HERE is what
                // keeps it out of the sqrt.
                if (i == c) {
                    v = sycl_device::dev_from_real<D>(sycl_device::dev_real(v));
                }
            }
            S[i + static_cast<std::ptrdiff_t>(c) * lda] = v;
        }
    }
    for (int c = tid; c < NB; c += L) diag[c] = R(0);
    for (int c = tid; c <= Rt0; c += L) off[c] = c * Rt0 - ((c * (c - 1)) >> 1);
    if (tid == 0) *fail = 0;

    potrf_phase_barrier<SC>(it);  // B0

    // `ok` is uniform because it is an SLM read taken after a phase barrier.
    // It is a PREDICATE and never a `break` out of the panel loop: the loop's
    // trip count must stay uniform or B2/B3 are orphaned.
    bool ok = true;
    for (int j = 0; j < n; j += NB) {
        const int ib = (n - j < NB) ? (n - j) : NB;
        const int m2 = n - j - ib;

        if (ok && p1_active) {
            potrf_diag_block_subgroup<D, R, NB>(sg, S, lda, j, ib, diag, fail);
        }
        potrf_phase_barrier<SC>(it);  // B1
        ok = (*fail == 0);

        if (ok && m2 > 0) {
            potrf_panel_solve_rows<D, R, NB>(tid, L, S, lda, j, ib, m2, diag);
        }
        potrf_phase_barrier<SC>(it);  // B2

        if (ok && m2 > 0) {
            const int Rt = (m2 + TS - 1) / TS;
            potrf_trailing_tiles<D, R, TS>(tid, L, S, lda, j, ib, m2, Rt, Rt0 - Rt, off);
        }
        potrf_phase_barrier<SC>(it);  // B3
    }

    // Store back. `i >= c` is the guard that makes "the other triangle is not
    // written" true; the load's `i >= c` is what makes the stronger "not read"
    // true, and ortho.cc:156-161 depends on the stronger one because it leaves
    // the other half of its Gram matrix uninitialised.
    for (int c = 0; c < n; ++c) {
        for (int i = tid; i < n; i += L) {
            if (i >= c) {
                const D v = S[i + static_cast<std::ptrdiff_t>(c) * lda];
                if (upper) {
                    Ag[c + static_cast<std::ptrdiff_t>(i) * ldg] = sycl_device::dev_conj(v);
                } else {
                    Ag[i + static_cast<std::ptrdiff_t>(c) * ldg] = v;
                }
            }
        }
    }
}

}  // namespace batchlas::potrf_native
