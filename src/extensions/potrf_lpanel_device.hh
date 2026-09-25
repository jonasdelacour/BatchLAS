#pragma once

// The native batched POTRF LPANEL kernel: LEFT-LOOKING, thread-per-row. Local memory holds
// one `n x NB` panel, not the matrix; the columns to its left are re-read from global memory.
// Four invariants a maintainer can silently break. (1) EVERY barrier is a WORK-GROUP barrier
// at the top level, outside the `if` guarding its phase, and failure is a predicated skip,
// never a `break` -- the schedule is then a function of (n, NB) alone, which is what makes
// packing G matrices into one work-group legal, the opposite of potrf_cta_device.hh. (2) B6 is
// not optional: the next panel's left-looking read is a GLOBAL read of the columns the store
// just wrote, by different work-items. (3) Every global read and every store is guarded by
// `row >= column`, so the other triangle is neither read nor written; ortho.cc depends on it.
// (4) rp[], rS[] and rA[] must stay in registers -- never a parameter, never a dynamic index.
// evidence: docs/perf/potrf.md#the-lpanel-tier

#include "../sycl/device_scalar.hh"

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>

namespace batchlas::potrf_native {

// The whole body for ONE matrix, Uplo::Lower; sA, sB, fail and Ag are already offset to it.
// `L >= n` lanes: lane `r` owns row `r` for the whole call, because rp/rS/rA must survive the
// k loop. `store` is false for a packed group's dead slots, which still run the WHOLE body --
// the barrier count is work-group uniform -- with only the global stores suppressed.
template <typename D, typename R, int NB>
inline void potrf_lpanel_body(const sycl::nd_item<1>& it,
                              int tid, int L, bool store,
                              D* __restrict sA, int slda,
                              D* __restrict sB,
                              int* __restrict fail,
                              D* __restrict Ag, int ldg,
                              int n) {
    if (tid == 0) *fail = 0;
    sycl::group_barrier(it.get_group());  // B0

    const int row = tid;
    bool ok = true;  // matrix-uniform: every lane derives it from the same local pivot

    for (int j = 0; j < n; j += NB) {
        const int ib = (n - j < NB) ? (n - j) : NB;
        const bool panel_ok = ok;
        const bool live = panel_ok && (row >= j) && (row < n);

        // (1) prefetch; `row >= j + i` keeps the strictly upper diagonal block unread.
        D rp[NB];
        D rS[NB];
#pragma unroll
        for (int i = 0; i < NB; ++i) {
            D v{};
            if (live && i < ib && row >= j + i) {
                v = Ag[row + static_cast<std::ptrdiff_t>(j + i) * ldg];
                // A Hermitian diagonal's imaginary part is contractually ignored.
                if (row == j + i) v = sycl_device::dev_from_real<D>(sycl_device::dev_real(v));
            }
            rp[i] = v;
            rS[i] = D{};
        }

        // (2) rS(row,i) = sum_{k<j} L(row,k)*conj(L(j+i,k)); (4) factors the result in local.
        for (int k = 0; k < j; k += NB) {
            const int kb = ((j - k) < NB) ? (j - k) : NB;

            // Out-of-range entries are ZEROED, not skipped: the FMA loop below is then a
            // fully unrolled NB x NB with no inner bound test.
            for (int e = tid; e < NB * NB; e += L) {
                const int i = e % NB;
                const int kk = e / NB;
                D v{};
                if (i < ib && kk < kb) {
                    v = sycl_device::dev_conj(
                        Ag[(j + i) + static_cast<std::ptrdiff_t>(k + kk) * ldg]);
                }
                sB[i + kk * NB] = v;
            }
            sycl::group_barrier(it.get_group());  // B1

            if (live) {
                D rA[NB];
#pragma unroll
                for (int kk = 0; kk < NB; ++kk) {
                    rA[kk] = (kk < kb)
                                 ? Ag[row + static_cast<std::ptrdiff_t>(k + kk) * ldg]
                                 : D{};
                }
#pragma unroll
                for (int i = 0; i < NB; ++i) {
#pragma unroll
                    for (int kk = 0; kk < NB; ++kk) {
                        sycl_device::fma_acc(rS[i], rA[kk], sB[i + kk * NB]);
                    }
                }
            }
            sycl::group_barrier(it.get_group());  // B2 -- WAR on sB
        }

        // (3) publish; the strictly upper diagonal block is ZEROED, not left indeterminate.
        if (live) {
#pragma unroll
            for (int i = 0; i < NB; ++i) {
                if (i < ib) {
                    D v{};
                    if (row >= j + i) {
                        v = sycl_device::dev_sub(rp[i], rS[i]);
                        // fma_acc(x, conj(x)) leaves a residue in the imaginary part.
                        if (row == j + i) {
                            v = sycl_device::dev_from_real<D>(sycl_device::dev_real(v));
                        }
                    }
                    sA[row + static_cast<std::ptrdiff_t>(i) * slda] = v;
                }
            }
        }
        sycl::group_barrier(it.get_group());  // B3

        for (int i = 0; i < ib; ++i) {
            const R d = sycl_device::dev_real(sA[(j + i) + static_cast<std::ptrdiff_t>(i) * slda]);

            // `!(d > 0)`, not `d <= 0`, so NaN is rejected too, as LAPACK does; every lane
            // reads the same local cell, so `bad` needs no broadcast.
            const bool bad = ok && !(d > R(0));
            if (bad) {
                ok = false;                             // sticky: FIRST FAILURE WINS
                if (tid == 0) *fail = j + i + 1;        // 1-based GLOBAL column
            }

            // WAR: every lane READ this cell just above, lane j+i overwrites it below.
            sycl::group_barrier(it.get_group());  // B3b, evidence: docs/perf/potrf.md#the-pivot-cell-war-race

            if (ok) {
                const R dkk = sycl::sqrt(d);
                const R rs = R(1) / dkk;  // NOT rsqrt: rsqrt.approx is not the reference
                if (row == j + i) {
                    sA[row + static_cast<std::ptrdiff_t>(i) * slda] =
                        sycl_device::dev_from_real<D>(dkk);
                } else if (row > j + i && row < n) {
                    sA[row + static_cast<std::ptrdiff_t>(i) * slda] = sycl_device::dev_mul_real(
                        sA[row + static_cast<std::ptrdiff_t>(i) * slda], rs);
                }
            }
            sycl::group_barrier(it.get_group());  // B4

            if (ok && row > j + i && row < n) {
                const D lri = sA[row + static_cast<std::ptrdiff_t>(i) * slda];
                for (int c = i + 1; c < ib; ++c) {
                    const D ljc = sA[(j + c) + static_cast<std::ptrdiff_t>(i) * slda];
                    sA[row + static_cast<std::ptrdiff_t>(c) * slda] = sycl_device::dev_sub(
                        sA[row + static_cast<std::ptrdiff_t>(c) * slda],
                        sycl_device::dev_mul(lri, sycl_device::dev_conj(ljc)));
                }
            }
            sycl::group_barrier(it.get_group());  // B5
        }

        // (5) store. Guarded by the panel's ENTRY state, not by `ok`: a panel that failed
        // part-way still holds finite numbers.
        if (store && panel_ok && row < n) {
            for (int i = 0; i < ib; ++i) {
                if (row >= j + i) {
                    Ag[row + static_cast<std::ptrdiff_t>(j + i) * ldg] =
                        sA[row + static_cast<std::ptrdiff_t>(i) * slda];
                }
            }
        }
        sycl::group_barrier(it.get_group());  // B6 -- the next panel READS these columns
    }
}

}  // namespace batchlas::potrf_native
