#pragma once

// The native batched POTRF CTA kernel -- all of its device code. One matrix is
// staged whole into local memory and factorised there by a right-looking blocked
// recurrence; global memory is touched exactly twice. evidence: docs/perf/potrf.md
//
// Three invariants a maintainer can silently break:
// * Phase barriers follow the `Scope` template parameter -- sub-group when one
//   32-wide sub-group owns a matrix, work-group when 2 or 4 do. Getting it wrong
//   races (P1)->(P2)->(P3) on the tile: no crash, a plausible wrong factor. Scope
//   is DERIVED by potrf_cta_launch_params (potrf_cta.cc), never chosen by hand.
// * Every phase barrier sits at the top level, outside the `if` guarding its
//   phase, and failure is a predicated skip and never a `break`, so the barrier
//   count is identical on the failure and success paths.
// * d[NB], x[NB] and acc[TS][TS] must stay in registers: never a parameter (an
//   accumulator array whose address is taken spills), never a dynamic index (that
//   moves the array to local memory with ZERO reported spill). Hence the
//   unrolled, predicated loops. evidence: docs/perf/potrf.md#register-gate

#include "../sycl/device_scalar.hh"

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>

namespace batchlas::potrf_native {

enum class PotrfScope { SubGroup, WorkGroup };

template <PotrfScope SC>
inline void potrf_phase_barrier(const sycl::nd_item<1>& it) {
    if constexpr (SC == PotrfScope::SubGroup) {
        sycl::group_barrier(it.get_sub_group());
    } else {
        sycl::group_barrier(it.get_group());
    }
}

// (P1) -- the ib x ib diagonal block, one lane per row, on ONE sub-group. Lane r
// owns row j+r in registers d[0..NB).
template <typename D, typename R, int NB>
inline void potrf_diag_block_subgroup(const sycl::sub_group& sg,
                                      D* __restrict S, int lda,
                                      int j, int ib,
                                      R* __restrict diag,
                                      int* __restrict fail) {
    const int lane = static_cast<int>(sg.get_local_linear_id());

    // Out-of-range entries are ZEROED, not left indeterminate: lanes ib..31 still
    // take part in every select_from_group below.
    D d[NB];
#pragma unroll
    for (int c = 0; c < NB; ++c) {
        d[c] = (lane < ib && c < ib && c <= lane)
                   ? S[(j + lane) + static_cast<std::ptrdiff_t>(j + c) * lda]
                   : D{};
    }

    bool alive = true;
#pragma unroll
    for (int k = 0; k < NB; ++k) {
        // THE PIVOT IS A REGISTER, NOT THE TILE: the updated Schur diagonal is
        // lane k's d[k]; S(j+k, j+k) still holds the ORIGINAL value, so reading
        // it there gives wrong columns and `info == 0` for any matrix failing at
        // a leading minor > 1 (tests/potrf_tests.cc InfoIndexIsExact).
        const R akk = sycl::select_from_group(sg, sycl_device::dev_real(d[k]),
                                              static_cast<uint32_t>(k));

        bool active = alive && (k < ib);

        // `!(akk > 0)`, not `akk <= 0`, so NaN is caught too -- LAPACK's
        // AJJ.LE.ZERO .OR. SISNAN(AJJ). It precedes the sqrt and the reciprocal.
        if (active && !(akk > R(0))) {
            if (lane == 0) *fail = j + k + 1;  // 1-based GLOBAL column
            active = false;
            alive = false;                     // sticky: FIRST FAILURE WINS
        }

        const R dkk = active ? sycl::sqrt(akk) : R(1);
        const R r = R(1) / dkk;  // NOT rsqrt: rsqrt.approx is not the reference

        if (active) {
            if (lane == k) {
                d[k] = sycl_device::dev_from_real<D>(dkk);
            } else if (lane > k && lane < ib) {
                d[k] = sycl_device::dev_mul_real(d[k], r);
            }

            // `lane < ib` is load-bearing: unguarded, lanes ib..31 write into
            // the A21 panel (P2) is about to read, and past the tile entirely on
            // the ragged last panel -- into a NEIGHBOURING MATRIX when G > 1.
            if (lane < ib && lane >= k) {
                S[(j + lane) + static_cast<std::ptrdiff_t>(j + k) * lda] = d[k];
            }
            if (lane == 0) diag[k] = dkk;
        }

        // B1'. ALWAYS the sub-group, and unconditional; (P1) may run inside
        // `if (sg_id == 0)`, where a work-group barrier would diverge.
        sycl::group_barrier(sg);

        // No WAR hazard, hence no second barrier: this reads column k, and step
        // k+1 publishes column k+1.
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

// (P2) -- the panel solve L21 = A21 * L11^-H (Side::Right, Lower, ConjTrans,
// NonUnit, alpha = 1). Rows of A21 are independent, so there is no barrier here.
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
                // A DIVIDE, as reference ?trsm divides -- (P1) scales by a
                // reciprocal because ?potf2 does. diag[c] is real, so complex D
                // takes two divides, not Smith's algorithm.
                x[c] = sycl_device::dev_div_real(s, diag[c]);
            }
        }

#pragma unroll
        for (int c = 0; c < NB; ++c) {
            if (c < ib) S[i + static_cast<std::ptrdiff_t>(j + c) * lda] = x[c];
        }
    }
}

// (P3) -- the trailing update A22 -= L21 L21^H, TS x TS register tiles, triangle
// at TILE granularity with `ra >= cb` trimming the diagonal tiles. `off` is the
// Rt0-based prefix table; `dR = Rt0 - Rt` rescales it here. No barrier.
template <typename D, typename R, int TS>
inline void potrf_trailing_tiles(int tid, int L,
                                 D* __restrict S, int lda,
                                 int j, int ib, int m2,
                                 int Rt, int dR,
                                 const int* __restrict off) {
    const int Ntiles = Rt * (Rt + 1) / 2;
    const int base = j + ib;

    for (int t = tid; t < Ntiles; t += L) {
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
                    // Force the Hermitian diagonal real. Nothing fails today
                    // without it -- (P1) rewrites every output diagonal from a
                    // real sqrt -- but the residue goes live the moment anything
                    // reads a trailing-update diagonal directly.
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

// The whole body for ONE matrix; S, diag, fail, off and Ag are already offset to
// it. `upper` is a LOAD/STORE TRANSFORM, not a second algorithm: A = U^H U is the
// same recurrence on S(i,c) = conj(A(c,i)), so this compiles once per
// (D, NB, TS, Scope) rather than once per (..., Uplo).
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

    // ONE FUSED LOAD, not a fill then a triangular load whose differing index
    // maps let a fill store land after a load store and zero a live element.
    // Every element of the lda x n tile is written exactly once here, pad rows
    // included -- which is why (P3) may read through its `< m2` guards.
    for (int c = 0; c < n; ++c) {
        for (int i = tid; i < lda; i += L) {
            D v{};
            if (i < n && i >= c) {
                v = upper ? sycl_device::dev_conj(Ag[c + static_cast<std::ptrdiff_t>(i) * ldg])
                          : Ag[i + static_cast<std::ptrdiff_t>(c) * ldg];
                // Diagonal loads as (real(A(c,c)), 0), per LAPACK/cuSOLVER.
                if (i == c) {
                    v = sycl_device::dev_from_real<D>(sycl_device::dev_real(v));
                }
            }
            S[i + static_cast<std::ptrdiff_t>(c) * lda] = v;
        }
    }
    for (int c = tid; c < NB; c += L) diag[c] = R(0);
    // off[] is (P3)'s tile-index prefix table: Rt0 + 1 entries per matrix,
    // published by B0 and NEVER rewritten per panel -- no barrier slot exists for
    // that, so (P3) rescales by dR. The sentinel off[Rt0] bounds its search.
    for (int c = tid; c <= Rt0; c += L) off[c] = c * Rt0 - ((c * (c - 1)) >> 1);
    if (tid == 0) *fail = 0;

    potrf_phase_barrier<SC>(it);  // B0

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

    // `i >= c` here means the other triangle is not WRITTEN; the same guard on
    // the LOAD means it is not READ, which ortho.cc depends on.
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
