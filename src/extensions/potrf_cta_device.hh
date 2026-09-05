#pragma once

// The native batched POTRF CTA kernel: one matrix is staged whole into local
// memory and factorised by a right-looking blocked recurrence, touching global
// memory exactly twice. evidence: docs/perf/potrf.md
//
// Three invariants a maintainer can silently break:
// * Phase barriers follow `Scope`, which potrf_cta_launch_params (potrf_cta.cc)
//   derives; the wrong scope races (P1)->(P2)->(P3) into a plausible wrong
//   factor rather than a crash.
// * Every phase barrier sits at the top level, outside the `if` guarding its
//   phase, and failure is a predicated skip, never a `break`, so both paths
//   execute the same barrier count.
// * d[NB], x[NB] and acc[TS][TS] must stay in registers: never a parameter,
//   never a dynamic index (which silently relocates them to local memory).
//   evidence: docs/perf/potrf.md#register-gate

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

// (P1) -- the ib x ib diagonal block on ONE sub-group; lane r owns row j+r.
template <typename D, typename R, int NB>
inline void potrf_diag_block_subgroup(const sycl::sub_group& sg,
                                      D* __restrict S, int lda,
                                      int j, int ib,
                                      R* __restrict diag,
                                      int* __restrict fail) {
    const int lane = static_cast<int>(sg.get_local_linear_id());

    // Out-of-range entries are zeroed, not left indeterminate: lanes ib..31 still
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
        // The pivot is lane k's register d[k], not S(j+k, j+k), which still holds
        // the ORIGINAL value: reading the tile gives wrong columns and `info == 0`
        // for any matrix failing at a leading minor > 1 (InfoIndexIsExact).
        const R akk = sycl::select_from_group(sg, sycl_device::dev_real(d[k]),
                                              static_cast<uint32_t>(k));

        bool active = alive && (k < ib);

        // `!(akk > 0)`, not `akk <= 0`, so NaN is rejected too, as LAPACK does.
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

            // `lane < ib` is load-bearing: unguarded, lanes ib..31 clobber the A21
            // panel (P2) reads, and on the ragged last panel a neighbouring matrix.
            if (lane < ib && lane >= k) {
                S[(j + lane) + static_cast<std::ptrdiff_t>(j + k) * lda] = d[k];
            }
            if (lane == 0) diag[k] = dkk;
        }

        // B1'. ALWAYS the sub-group and unconditional: (P1) may run inside
        // `if (sg_id == 0)`, where a work-group barrier would diverge.
        sycl::group_barrier(sg);

        // No WAR hazard, hence no second barrier: step k+1 publishes a later column.
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

// (P2) -- the panel solve L21 = A21 * L11^-H; rows are independent, no barrier.
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
                // A DIVIDE, as reference ?trsm divides; (P1) scales by a
                // reciprocal because ?potf2 does.
                x[c] = sycl_device::dev_div_real(s, diag[c]);
            }
        }

#pragma unroll
        for (int c = 0; c < NB; ++c) {
            if (c < ib) S[i + static_cast<std::ptrdiff_t>(j + c) * lda] = x[c];
        }
    }
}

// (P3) -- the trailing update A22 -= L21 L21^H in TS x TS register tiles, the
// triangle trimmed at TILE granularity by `ra >= cb`. `off` is the Rt0-based
// prefix table, rescaled here by `dR = Rt0 - Rt`. No barrier.
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
                    // Force the Hermitian diagonal real: dead today ((P1) rewrites
                    // every output diagonal), live if anything ever reads one here.
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
// same recurrence on S(i,c) = conj(A(c,i)), so Uplo costs no extra instantiation.
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

    // One fused load writes every element of the lda x n tile exactly once, pad
    // rows included -- which is why (P3) may read through its `< m2` guards.
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
    // off[] is (P3)'s tile-index prefix table, published once at B0 and NEVER
    // rewritten per panel -- hence (P3)'s dR rescale. off[Rt0] is its sentinel.
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
