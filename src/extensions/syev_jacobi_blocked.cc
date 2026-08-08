#include <blas/matrix.hh>
#include <blas/functions.hh>
#include <blas/extensions.hh>
#include <blas/extra.hh>
#include <util/mempool.hh>
#include <util/env.hh>
#include <batchlas/backend_config.h>
#include "../math-helpers.hh"
#include "../queue.hh"
#include "../util/template-instantiations.hh"
#include <algorithm>
#include <complex>
#include <limits>
#include <stdexcept>

namespace batchlas {

// Kernel name tag. Must live outside the anonymous namespace below so it does
// not depend on internal-linkage entities.
template <typename T, bool ComputeVectors>
class SyevJacobiBlockedKernel;

// ---------------------------------------------------------------------------
// Blocked two-sided Jacobi eigensolver (JACOBI_EIGENSOLVER_PLAN.md Tier B/C,
// SYEV_PERF_IMPLEMENTATION_PLAN.md WP8).
//
// syev_jacobi_cta solves one problem inside one sub-group partition, which caps
// it at n <= 32: A and Z have to fit in local memory *per partition*. This
// kernel lifts the same algorithm to n in the hundreds by giving each matrix a
// whole work-group and keeping only the current 2nb x 2nb pivot block resident:
//
//   for each sweep
//     for each block pair (p, q), cyclically
//       S  <- A[idx, idx]                    idx = cols(p) ++ cols(q)
//       U  <- inner two-sided Jacobi on S    (local memory, rotation-based)
//       A[:, idx] <- A[:, idx] * U           (panel update, coalesced)
//       A[idx, idx] <- S                     (already U^H A U)
//       A[idx, r]   <- conj(A[r, idx])       (mirror, r outside idx)
//       Z[:, idx] <- Z[:, idx] * U
//
// Two structural points that are not obvious from the plan sketch:
//
// * **The row update is a transpose, not a GEMM.** The sketch has
//   `A[(p,q), :] <- U^T A[(p,q), :]` as a second m x n product. It does not
//   have to be computed. Writing A' = V^H A V with V block-diagonal and
//   B = A V, the rows outside the pivot block satisfy
//   A'[idx, r] = conj(A'[r, idx]) because A' is Hermitian and A'[r, idx] is
//   exactly what the *column* update already produced. So the second product
//   collapses to the m x m pivot block -- which the inner solve has computed
//   anyway, as S -- plus a transposing copy. That removes n*m^2 flops per pair
//   and, more importantly, removes the strided-read panel the row update would
//   otherwise need.
//
// * **The pivot block covers everything when l == 2.** With two block columns
//   the pivot block is the whole matrix, the panel update and the mirror are
//   empty, and the kernel degenerates into a fully local-memory-resident
//   work-group Jacobi. That is the Tier B regime of the plan (32 < n <~ 128)
//   and it is where this kernel is fastest, because it never touches global
//   memory between the initial load and the final store.
//
// Accuracy is the reason the solver exists: with the *relative* off-diagonal
// threshold below, the eigenvalue error is governed by the condition number of
// the column-equilibrated matrix rather than of A itself (Demmel & Veselic,
// SIMAX 13(4), 1992), so graded input keeps its small eigenvalues where a
// tridiagonalizing path loses them entirely. The blocking does not change the
// criterion: a rotation is applied for exactly the same pivot pairs a scalar
// Jacobi would rotate, they are just applied via the accumulated block
// transform. See JACOBI_EIGENSOLVER_PLAN.md sections 4 and 5 for what the
// theory does and does not cover for the blocked variant.
// ---------------------------------------------------------------------------

namespace {

template <typename U>
inline U conj_jb(const U& x) {
    if constexpr (internal::is_complex<U>::value) {
        return U(x.real(), -x.imag());
    } else {
        return x;
    }
}

template <typename U>
inline typename base_type<U>::type abs_jb(const U& x) {
    using Real = typename base_type<U>::type;
    if constexpr (internal::is_complex<U>::value) {
        return sycl::hypot(x.real(), x.imag());
    } else {
        return sycl::fabs(x);
    }
}

template <typename U>
inline U force_real_jb(const U& x) {
    if constexpr (internal::is_complex<U>::value) {
        return U(x.real(), typename base_type<U>::type(0));
    } else {
        return x;
    }
}

template <typename U>
inline typename base_type<U>::type real_jb(const U& x) {
    if constexpr (internal::is_complex<U>::value) {
        return x.real();
    } else {
        return x;
    }
}

// Textbook complex multiply. std::complex<T>::operator* implements C99 Annex G,
// which clang lowers to a per-operand isnan branch plus a __mulsc3 / __muldc3
// call -- see the same fix in latrd_lower_panel.cc:121 and syev.hh:650. The
// panel update below is one long chain of these, so the libcall form is not
// affordable there.
template <typename U>
inline U mul_jb(const U& a, const U& b) {
    if constexpr (internal::is_complex<U>::value) {
        using Real = typename base_type<U>::type;
        const Real ar = a.real();
        const Real ai = a.imag();
        const Real br = b.real();
        const Real bi = b.imag();
        return U(ar * br - ai * bi, ar * bi + ai * br);
    } else {
        return a * b;
    }
}

// Longest pivot-block order the panel update can hold in registers.
//
// The update keeps one full row of the m-column panel live (`T arow[MaxM]`) so
// that the whole row can be read once, coalesced, and rewritten in place. That
// is the single biggest lever on the update's cost and it is worth spending
// registers on -- but only 64 of them: see register-residency notes in
// gemm_128x128 work, where an over-wide thread tile spilled the accumulator and
// cost 43%. Each entry is sized so MaxM * sizeof(T) == 256 bytes.
template <typename T>
constexpr int jacobi_blocked_max_m() {
    if constexpr (sizeof(T) <= 4) {
        return 64;   // float
    } else if constexpr (sizeof(T) <= 8) {
        return 32;   // double, complex<float>
    } else {
        return 16;   // complex<double>
    }
}

// Round-robin ("chess tournament") pairing, identical to syev_jacobi_cta's.
// Round t of an even order m produces m/2 disjoint pairs and the m-1 rounds
// together cover every index pair exactly once.
inline void round_robin_pair_jb(int32_t m, int32_t t, int32_t k, int32_t& p, int32_t& q) {
    const int32_t ring = m - 1;
    if (k == 0) {
        p = 0;
        q = (t % ring) + 1;
    } else {
        p = ((t + k) % ring) + 1;
        q = (((t - k) % ring + ring) % ring) + 1;
    }
    if (p > q) {
        const int32_t tmp = p;
        p = q;
        q = tmp;
    }
}

// Block partition: l block columns of width nb, the last possibly short.
struct BlockPlan {
    int32_t nb = 0;      // block width
    int32_t l = 0;       // number of block columns
    int32_t m_max = 0;   // largest pivot-block order that can occur
};

// Pick the widest pivot block that fits both local memory and the register
// budget of the panel update. Wider is strictly better for the outer loop:
// global traffic per sweep is ~6 n^3 / nb, so nb is the only knob that moves
// it. Local memory is the binding constraint at every scalar type.
template <typename T>
BlockPlan plan_blocks(int32_t n, std::size_t local_mem_bytes, int32_t forced_nb) {
    using Real = typename base_type<T>::type;
    constexpr int32_t kMaxM = jacobi_blocked_max_m<T>();
    constexpr bool kNeedPhase = internal::is_complex<T>::value;

    // Fixed overhead that does not scale with m: the rank scratch used by the
    // final sort, plus the two rotation counters.
    const std::size_t fixed = static_cast<std::size_t>(n) * sizeof(int32_t) + 4 * sizeof(int32_t);

    int32_t m_max = 0;
    for (int32_t m = kMaxM; m >= 4; m -= 2) {
        const std::size_t ld = static_cast<std::size_t>(m) + 1;
        std::size_t bytes = 2 * static_cast<std::size_t>(m) * ld * sizeof(T);          // S and U
        bytes += static_cast<std::size_t>(m) * static_cast<std::size_t>(m / 2) * sizeof(int16_t);  // pair table
        bytes += static_cast<std::size_t>(m / 2) * 2 * sizeof(Real);                   // (c, s)
        if constexpr (kNeedPhase) {
            bytes += static_cast<std::size_t>(m / 2) * sizeof(T);                      // diagonal phase
        }
        bytes += fixed;
        if (bytes <= local_mem_bytes) {
            m_max = m;
            break;
        }
    }
    if (m_max == 0) {
        throw std::runtime_error("syev_jacobi_blocked: device local memory is too small for any pivot block.");
    }

    BlockPlan plan;
    plan.m_max = m_max;

    // l >= 2 always: with a single block column there are no pivot pairs. The
    // l == 2 case is the fully resident one and is deliberately reachable.
    int32_t nb = std::min<int32_t>(m_max / 2, (n + 1) / 2);
    if (forced_nb > 0) {
        nb = std::min<int32_t>(std::min<int32_t>(forced_nb, m_max / 2), (n + 1) / 2);
    }
    nb = std::max<int32_t>(nb, 1);

    int32_t l = (n + nb - 1) / nb;
    l = std::max<int32_t>(l, 2);
    // Rebalance so the blocks are as even as the partition allows, then make
    // sure the rebalanced width still fits.
    nb = (n + l - 1) / l;
    while (2 * nb > m_max) {
        ++l;
        nb = (n + l - 1) / l;
    }
    l = (n + nb - 1) / nb;

    plan.nb = nb;
    plan.l = l;
    return plan;
}

template <typename T, bool ComputeVectors>
void syev_jacobi_blocked_impl(Queue& ctx,
                              MatrixView<T, MatrixFormat::Dense>& a,
                              typename base_type<T>::type* w_ptr,
                              T* z_ptr,
                              int32_t n,
                              bool upper,
                              const BlockPlan& plan,
                              JacobiParams<T> params,
                              int32_t wg_size) {
    using Real = typename base_type<T>::type;
    constexpr int32_t kMaxM = jacobi_blocked_max_m<T>();
    constexpr bool kNeedPhase = internal::is_complex<T>::value;

    const int32_t batch_size = static_cast<int32_t>(a.batch_size());

    ctx->submit([&](sycl::handler& cgh) {
        auto A_view = a.kernel_view();

        const int32_t m_max = plan.m_max;
        const int32_t nb = plan.nb;
        const int32_t l = plan.l;
        const int32_t LDS = m_max + 1;
        const int32_t W = wg_size;

        // Rows of the panel staged through local memory when the mirror runs.
        // The staging tile reuses the S buffer, which is dead by then, so it
        // costs nothing extra.
        const int32_t RB = std::min<int32_t>(m_max, n);

        auto S_local = sycl::local_accessor<T, 1>(sycl::range<1>(static_cast<std::size_t>(m_max) * LDS), cgh);
        auto U_local = sycl::local_accessor<T, 1>(sycl::range<1>(static_cast<std::size_t>(m_max) * LDS), cgh);
        auto Pair_local = sycl::local_accessor<int16_t, 1>(
            sycl::range<1>(static_cast<std::size_t>(m_max) * static_cast<std::size_t>(m_max / 2)), cgh);
        auto Rcs_local = sycl::local_accessor<sycl::vec<Real, 2>, 1>(
            sycl::range<1>(static_cast<std::size_t>(m_max / 2)), cgh);
        auto Rd_local = sycl::local_accessor<T, 1>(
            sycl::range<1>(kNeedPhase ? static_cast<std::size_t>(m_max / 2) : 1), cgh);
        auto Rank_local = sycl::local_accessor<int32_t, 1>(sycl::range<1>(static_cast<std::size_t>(n)), cgh);
        // [0] rotations applied to the current pivot block, [1] rotations in the
        // current inner sweep, [2] "this block is above threshold at all".
        auto Cnt_local = sycl::local_accessor<int32_t, 1>(sycl::range<1>(4), cgh);

        const int32_t nn = n;
        const int32_t max_sweeps = std::max<int32_t>(int32_t(1), static_cast<int32_t>(params.max_sweeps));
        // 0 selects by the measured rule below; anything else is a literal cap,
        // itself clamped by max_sweeps, so a large value means "to convergence".
        //
        // Measured, RTX 4090 / float / saturating batch, us per matrix:
        //
        //             l == 2 (resident)      l > 2 (blocked)
        //   exact          3.64 (n=64)        76.2 (n=128)  383 (n=256, vals)
        //   1 sweep        4.81 (n=64)        55.1 (n=128)  238 (n=256, vals)
        //
        // The crossover has a cause, not just a number. When l == 2 the pivot
        // block *is* the matrix, so an exact inner solve finishes the whole
        // problem in one outer sweep and there is no outer cost left to
        // amortize. When l > 2 the inner solve costs O(m^3) per pivot block
        // against the outer update's O(n m^2), so extra inner sweeps buy
        // convergence at a worse rate than another outer sweep does -- which is
        // exactly why MAGMA's batched SVD and Novakovic's block-oriented variant
        // both use the single inexact sweep.
        const bool resident_solve = (l == 2);
        const int32_t inner_cap =
            (params.inner_sweeps == 0)
                ? (resident_solve ? max_sweeps : 1)
                : std::min<int32_t>(max_sweeps, static_cast<int32_t>(params.inner_sweeps));
        const bool do_sort = params.sort;
        const bool ascending = (params.sort_order == SortOrder::Ascending);
        const bool is_upper = upper;

        // Relative off-diagonal threshold, |a_pq| > tol * sqrt(|a_pp| * |a_qq|).
        // The classical absolute test would forfeit the relative-accuracy result
        // this solver exists for.
        const Real tol = params.tol_multiplier * static_cast<Real>(nn) * std::numeric_limits<Real>::epsilon();
        const Real tiny = std::numeric_limits<Real>::min();
        const Real tau_big = Real(1) / sycl::sqrt(std::numeric_limits<Real>::epsilon());

        Real* Wout = w_ptr;
        T* Zbase = z_ptr;

        cgh.parallel_for<SyevJacobiBlockedKernel<T, ComputeVectors>>(
            sycl::nd_range<1>(static_cast<std::size_t>(batch_size) * W, W),
            [=](sycl::nd_item<1> it) {
                const auto wg = it.get_group();
                const int32_t prob = static_cast<int32_t>(wg.get_group_linear_id());
                const int32_t tid = static_cast<int32_t>(it.get_local_linear_id());

                auto A_prob = A_view.batch_item(prob);
                T* Ad = A_prob.data();
                const int32_t lda = static_cast<int32_t>(A_prob.ld());
                T* Zd = ComputeVectors ? (Zbase + static_cast<int64_t>(prob) * nn * nn) : nullptr;

                // ---- Symmetrize into full storage, and seed Z = I. ----
                // Only one triangle of A is meaningful on input; the blocked
                // update reads whole columns, so the mirror has to exist.
                for (int32_t idx = tid; idx < nn * nn; idx += W) {
                    const int32_t j = idx / nn;
                    const int32_t i = idx - j * nn;
                    if (i == j) {
                        Ad[i + j * lda] = force_real_jb(Ad[i + j * lda]);
                    } else if (is_upper ? (i > j) : (i < j)) {
                        Ad[i + j * lda] = conj_jb(Ad[j + i * lda]);
                    }
                    if constexpr (ComputeVectors) {
                        Zd[i + j * nn] = (i == j) ? T(1) : T(0);
                    }
                }
                sycl::group_barrier(wg);

                // ---- Sweeps over block pivot pairs, cyclic ordering. ----
                for (int32_t sweep = 0; sweep < max_sweeps; ++sweep) {
                    int32_t sweep_rot = 0;

                    for (int32_t bp = 0; bp < l - 1; ++bp) {
                        for (int32_t bq = bp + 1; bq < l; ++bq) {
                            const int32_t p0 = bp * nb;
                            const int32_t q0 = bq * nb;
                            const int32_t mp = std::min<int32_t>(nb, nn - p0);
                            const int32_t mq = std::min<int32_t>(nb, nn - q0);
                            const int32_t m = mp + mq;
                            if (mp <= 0 || mq <= 0) continue;

                            // Global column (and row) index of pivot slot k.
                            const auto gidx = [=](int32_t k) {
                                return (k < mp) ? (p0 + k) : (q0 + k - mp);
                            };

                            const int32_t me = m + (m & 1);
                            const int32_t rounds = me - 1;
                            const int32_t ppr = me / 2;

                            // Thread decomposition used by every m-wide phase:
                            // consecutive threads take consecutive row/column
                            // indices so both local and global accesses stay
                            // stride-1, and each group of m threads owns one
                            // pivot pair.
                            const int32_t slot = tid / m;
                            const int32_t lane = tid - slot * m;
                            const int32_t slots = W / m;
                            const bool slot_active = (slot < slots);

                            // Closes the previous pivot pair: everything below
                            // reuses S, U, the pair table and the counters.
                            sycl::group_barrier(wg);

                            // ---- Gather the pivot block; seed U = I. ----
                            if (slot_active) {
                                for (int32_t j = slot; j < m; j += slots) {
                                    const int32_t gj = gidx(j);
                                    S_local[lane + j * LDS] = Ad[gidx(lane) + gj * lda];
                                    U_local[lane + j * LDS] = (lane == j) ? T(1) : T(0);
                                }
                            }

                            // ---- Pivot pair table for this block order. ----
                            for (int32_t idx = tid; idx < rounds * ppr; idx += W) {
                                const int32_t t = idx / ppr;
                                const int32_t k = idx - t * ppr;
                                int32_t p = 0;
                                int32_t q = 0;
                                round_robin_pair_jb(me, t, k, p, q);
                                Pair_local[idx] = static_cast<int16_t>(p | (q << 8));
                            }
                            if (tid == 0) {
                                Cnt_local[0] = 0;
                                Cnt_local[2] = 0;
                            }
                            sycl::group_barrier(wg);

                            // ---- Is this block already diagonal to threshold? ----
                            // Without this, a converged block still costs a full
                            // inner sweep -- m-1 rounds and three work-group
                            // barriers each -- just to discover that it rotates
                            // nothing. The final verification sweep is the whole
                            // matrix in that state, so the test pays for itself
                            // there alone. It is exactly the criterion the inner
                            // solve applies, so it can never skip a pair the
                            // inner solve would have rotated.
                            if (slot_active) {
                                int32_t dirty = 0;
                                for (int32_t j = slot; j < m; j += slots) {
                                    if (lane == j) continue;
                                    const Real ajj = real_jb(S_local[j + j * LDS]);
                                    const Real aii = real_jb(S_local[lane + lane * LDS]);
                                    const Real g_abs = abs_jb(S_local[lane + j * LDS]);
                                    const Real thresh = tol * sycl::sqrt(sycl::fabs(aii) * sycl::fabs(ajj));
                                    if (g_abs > thresh && g_abs > tiny) dirty = 1;
                                }
                                if (dirty) {
                                    sycl::atomic_ref<int32_t, sycl::memory_order::relaxed,
                                                     sycl::memory_scope::work_group,
                                                     sycl::access::address_space::local_space>
                                        cnt(Cnt_local[2]);
                                    cnt.fetch_add(1);
                                }
                            }
                            sycl::group_barrier(wg);
                            if (Cnt_local[2] == 0) continue;

                            // ---- Inner two-sided Jacobi on S, accumulating U. ----
                            for (int32_t isweep = 0; isweep < inner_cap; ++isweep) {
                                if (tid == 0) Cnt_local[1] = 0;
                                sycl::group_barrier(wg);

                                for (int32_t t = 0; t < rounds; ++t) {
                                    const int32_t tab = t * ppr;

                                    if (tid < ppr) {
                                        const int32_t pq = static_cast<int32_t>(Pair_local[tab + tid]);
                                        const int32_t p = pq & 0xFF;
                                        const int32_t q = (pq >> 8) & 0xFF;

                                        Real c_rot = Real(1);
                                        Real s_rot = Real(0);
                                        T d_rot = T(1);
                                        bool active = (q < m);

                                        if (active) {
                                            const T apq = S_local[p + q * LDS];
                                            const Real app = real_jb(S_local[p + p * LDS]);
                                            const Real aqq = real_jb(S_local[q + q * LDS]);
                                            const Real g_abs = abs_jb(apq);
                                            const Real thresh = tol * sycl::sqrt(sycl::fabs(app) * sycl::fabs(aqq));

                                            if (g_abs > thresh && g_abs > tiny) {
                                                Real g;
                                                if constexpr (internal::is_complex<T>::value) {
                                                    g = g_abs;
                                                    d_rot = T(apq.real() / g_abs, -apq.imag() / g_abs);
                                                } else {
                                                    g = apq;
                                                    d_rot = T(1);
                                                }
                                                const Real tau = (aqq - app) / (Real(2) * g);
                                                Real tt;
                                                if (sycl::fabs(tau) > tau_big) {
                                                    tt = Real(1) / (Real(2) * tau);
                                                } else {
                                                    tt = sycl::copysign(Real(1), tau)
                                                       / (sycl::fabs(tau) + sycl::sqrt(Real(1) + tau * tau));
                                                }
                                                c_rot = Real(1) / sycl::sqrt(Real(1) + tt * tt);
                                                s_rot = tt * c_rot;
                                                // A rotation that rounds to the
                                                // identity never annihilates
                                                // a_pq, so counting it would keep
                                                // the sweep loop alive forever.
                                                if (s_rot == Real(0)) active = false;
                                            } else {
                                                active = false;
                                            }
                                        }

                                        if (!active) {
                                            c_rot = Real(1);
                                            s_rot = Real(0);
                                            d_rot = T(1);
                                        }
                                        Rcs_local[tid] = sycl::vec<Real, 2>(c_rot, s_rot);
                                        if constexpr (kNeedPhase) {
                                            Rd_local[tid] = d_rot;
                                        }
                                        if (active) {
                                            sycl::atomic_ref<int32_t, sycl::memory_order::relaxed,
                                                             sycl::memory_scope::work_group,
                                                             sycl::access::address_space::local_space>
                                                cnt(Cnt_local[1]);
                                            cnt.fetch_add(1);
                                        }
                                    }
                                    sycl::group_barrier(wg);

                                    // Phase 1: S <- S * J and U <- U * J.
                                    // lane == row, so both arrays are touched
                                    // with stride 1.
                                    if (slot_active) {
                                        for (int32_t k = slot; k < ppr; k += slots) {
                                            const sycl::vec<Real, 2> cs = Rcs_local[k];
                                            const Real ck = cs[0];
                                            const Real sk = cs[1];
                                            if (sk == Real(0)) continue;

                                            const int32_t pq = static_cast<int32_t>(Pair_local[tab + k]);
                                            const int32_t pk = pq & 0xFF;
                                            const int32_t qk = (pq >> 8) & 0xFF;

                                            T u11 = T(ck);
                                            T u12 = T(sk);
                                            T u21 = T(-sk);
                                            T u22 = T(ck);
                                            if constexpr (kNeedPhase) {
                                                const T dk = Rd_local[k];
                                                u21 = -mul_jb(dk, T(sk));
                                                u22 = mul_jb(dk, T(ck));
                                            }

                                            const int32_t ip = lane + pk * LDS;
                                            const int32_t iq = lane + qk * LDS;
                                            const T sp = S_local[ip];
                                            const T sq = S_local[iq];
                                            S_local[ip] = mul_jb(sp, u11) + mul_jb(sq, u21);
                                            S_local[iq] = mul_jb(sp, u12) + mul_jb(sq, u22);

                                            const T up = U_local[ip];
                                            const T uq = U_local[iq];
                                            U_local[ip] = mul_jb(up, u11) + mul_jb(uq, u21);
                                            U_local[iq] = mul_jb(up, u12) + mul_jb(uq, u22);
                                        }
                                    }
                                    sycl::group_barrier(wg);

                                    // Phase 2: S <- J^H * S. lane == column, so
                                    // the stride is LDS == m_max + 1, which is
                                    // odd and therefore conflict-free. The
                                    // annihilated entries are stored as exact
                                    // zeros here rather than in a separate pass.
                                    if (slot_active) {
                                        for (int32_t k = slot; k < ppr; k += slots) {
                                            const sycl::vec<Real, 2> cs = Rcs_local[k];
                                            const Real ck = cs[0];
                                            const Real sk = cs[1];
                                            if (sk == Real(0)) continue;

                                            const int32_t pq = static_cast<int32_t>(Pair_local[tab + k]);
                                            const int32_t pk = pq & 0xFF;
                                            const int32_t qk = (pq >> 8) & 0xFF;

                                            const T cu11 = T(ck);
                                            const T cu12 = T(sk);
                                            T cu21 = T(-sk);
                                            T cu22 = T(ck);
                                            if constexpr (kNeedPhase) {
                                                const T dk = Rd_local[k];
                                                cu21 = -mul_jb(conj_jb(dk), T(sk));
                                                cu22 = mul_jb(conj_jb(dk), T(ck));
                                            }

                                            const int32_t ip = pk + lane * LDS;
                                            const int32_t iq = qk + lane * LDS;
                                            const T sp = S_local[ip];
                                            const T sq = S_local[iq];
                                            T new_p = mul_jb(cu11, sp) + mul_jb(cu21, sq);
                                            T new_q = mul_jb(cu12, sp) + mul_jb(cu22, sq);
                                            if (lane == qk) {
                                                new_p = T(0);
                                                new_q = force_real_jb(new_q);
                                            } else if (lane == pk) {
                                                new_q = T(0);
                                                new_p = force_real_jb(new_p);
                                            }
                                            S_local[ip] = new_p;
                                            S_local[iq] = new_q;
                                        }
                                    }
                                    sycl::group_barrier(wg);
                                }

                                const int32_t inner_rot = Cnt_local[1];
                                if (tid == 0) Cnt_local[0] += inner_rot;
                                sycl::group_barrier(wg);
                                if (inner_rot == 0) break;
                            }

                            const int32_t pair_rot = Cnt_local[0];
                            sweep_rot += pair_rot;
                            if (pair_rot == 0) continue;

                            const bool full_block = (m == nn);

                            // ---- Apply U. ----
                            // When the pivot block is the whole matrix there is
                            // no panel outside it and no rows to mirror: the
                            // inner solve has already produced the answer.
                            if (!full_block) {
                                // A[:, idx] <- A[:, idx] * U over all n rows.
                                // The whole panel row is held in registers, so
                                // each global element is read once and written
                                // once, both coalesced, and U is read from local
                                // memory as a work-group-wide broadcast.
                                for (int32_t r = tid; r < nn; r += W) {
                                    T arow[kMaxM];
                                    for (int32_t k = 0; k < mp; ++k) {
                                        arow[k] = Ad[r + (p0 + k) * lda];
                                    }
                                    for (int32_t k = 0; k < mq; ++k) {
                                        arow[mp + k] = Ad[r + (q0 + k) * lda];
                                    }
                                    for (int32_t j = 0; j < m; ++j) {
                                        T acc = T(0);
                                        for (int32_t k = 0; k < m; ++k) {
                                            acc += mul_jb(arow[k], U_local[k + j * LDS]);
                                        }
                                        Ad[r + gidx(j) * lda] = acc;
                                    }
                                }
                                sycl::group_barrier(wg);
                            }

                            // A[idx, idx] <- S, which the inner solve left equal
                            // to U^H A[idx, idx] U.
                            if (slot_active) {
                                for (int32_t j = slot; j < m; j += slots) {
                                    Ad[gidx(lane) + gidx(j) * lda] = S_local[lane + j * LDS];
                                }
                            }
                            sycl::group_barrier(wg);

                            if (!full_block) {
                                // Mirror: A[idx, r] <- conj(A[r, idx]) for rows
                                // r outside the pivot block. Staged through the
                                // (now dead) S buffer because the destination is
                                // transposed -- writing it directly would turn
                                // one coalesced store into m scattered ones.
                                const int32_t LDT = m + 1;
                                for (int32_t r0 = 0; r0 < nn; r0 += RB) {
                                    const int32_t rows = std::min<int32_t>(RB, nn - r0);
                                    sycl::group_barrier(wg);
                                    // Load: lane == row, so the global read is
                                    // stride 1 and the local write has stride
                                    // LDT (odd).
                                    for (int32_t idx = tid; idx < rows * m; idx += W) {
                                        const int32_t k = idx / rows;
                                        const int32_t i = idx - k * rows;
                                        S_local[k + i * LDT] = Ad[(r0 + i) + gidx(k) * lda];
                                    }
                                    sycl::group_barrier(wg);
                                    // Store: lane == pivot slot, so the global
                                    // write is stride 1 within each of the two
                                    // block ranges and the local read has stride 1.
                                    for (int32_t idx = tid; idx < rows * m; idx += W) {
                                        const int32_t i = idx / m;
                                        const int32_t k = idx - i * m;
                                        const int32_t r = r0 + i;
                                        const bool inside = (r >= p0 && r < p0 + mp) || (r >= q0 && r < q0 + mq);
                                        if (inside) continue;
                                        Ad[gidx(k) + r * lda] = conj_jb(S_local[k + i * LDT]);
                                    }
                                }
                                sycl::group_barrier(wg);
                            }

                            if constexpr (ComputeVectors) {
                                for (int32_t r = tid; r < nn; r += W) {
                                    T zrow[kMaxM];
                                    for (int32_t k = 0; k < mp; ++k) {
                                        zrow[k] = Zd[r + (p0 + k) * nn];
                                    }
                                    for (int32_t k = 0; k < mq; ++k) {
                                        zrow[mp + k] = Zd[r + (q0 + k) * nn];
                                    }
                                    for (int32_t j = 0; j < m; ++j) {
                                        T acc = T(0);
                                        for (int32_t k = 0; k < m; ++k) {
                                            acc += mul_jb(zrow[k], U_local[k + j * LDS]);
                                        }
                                        Zd[r + gidx(j) * nn] = acc;
                                    }
                                }
                            }
                            sycl::group_barrier(wg);
                        }
                    }

                    // Converged when a whole sweep found no pivot pair above the
                    // relative threshold.
                    if (sweep_rot == 0) break;
                }

                // ---- Rank-based sort, then write back. ----
                // Ranks are computed in parallel and ties broken on index, so
                // the permutation is a bijection and needs no scratch sort.
                for (int32_t i = tid; i < nn; i += W) {
                    const Real wi = real_jb(Ad[i + i * lda]);
                    int32_t rank = i;
                    if (do_sort) {
                        rank = 0;
                        for (int32_t k = 0; k < nn; ++k) {
                            const Real wk = real_jb(Ad[k + k * lda]);
                            const bool before = ascending ? (wk < wi || (wk == wi && k < i))
                                                          : (wk > wi || (wk == wi && k < i));
                            if (before) ++rank;
                        }
                    }
                    Rank_local[i] = rank;
                    Wout[static_cast<int64_t>(prob) * nn + rank] = wi;
                }
                sycl::group_barrier(wg);

                if constexpr (ComputeVectors) {
                    for (int32_t c = 0; c < nn; ++c) {
                        const int32_t dst = Rank_local[c];
                        for (int32_t r = tid; r < nn; r += W) {
                            Ad[r + dst * lda] = Zd[r + c * nn];
                        }
                    }
                }
            });
    });
}

template <typename T>
void validate_jacobi_blocked(const MatrixView<T, MatrixFormat::Dense>& a, JobType jobz, const char* who) {
    if (a.rows() != a.cols()) {
        throw std::invalid_argument(std::string(who) + ": A must be square.");
    }
    if (jobz != JobType::NoEigenVectors && jobz != JobType::EigenVectors) {
        throw std::invalid_argument(std::string(who) + ": invalid JobType.");
    }
    if (a.rows() < 2 || a.rows() > 1024) {
        throw std::invalid_argument(std::string(who) + " supports 2 <= n <= 1024.");
    }
}

// Default work-group size, measured on RTX 4090 / float / saturating batch
// (us per matrix, eigenvectors):
//
//            wg=256   wg=512   wg=768   wg=1024
//   n=64      3.36     3.00     2.94      3.96
//   n=128    55.4     51.5     52.0      44.8
//   n=256   398      398      421       372
//
// Two regimes, and the split is the same l == 2 boundary as the inner-sweep
// rule. A resident solve (l == 2) never leaves local memory, so it is bound by
// local-memory traffic and barriers and wants *two* work-groups per SM to hide
// them -- and with Ada's 1536-thread-per-SM ceiling that caps the work-group at
// 768. Going to 1024 halves residency and costs 35%, which is the cliff in the
// n=64 row. A blocked solve (l > 2) is bound by global-memory latency on the
// panel update instead, where one wide work-group issuing more concurrent loads
// beats two narrow ones.
int32_t jacobi_blocked_wg_size(Queue& ctx, const BlockPlan& plan) {
    const auto dev = ctx->get_device();
    const int32_t max_wg = static_cast<int32_t>(dev.get_info<sycl::info::device::max_work_group_size>());
    const int32_t fallback = (plan.l == 2) ? 768 : 1024;
    int32_t wg = env_positive_int_or("BATCHLAS_JACOBI_BLOCKED_WG", fallback);
    wg = std::min(wg, max_wg);
    // Every m-wide phase decomposes the work-group into groups of m threads, so
    // the work-group has to be at least one pivot-block order wide.
    wg = std::max(wg, plan.m_max);
    wg = (wg / 32) * 32;
    return std::max<int32_t>(wg, 32);
}

} // namespace

template <Backend B, typename T>
Event syev_jacobi_blocked(Queue& ctx,
                          const MatrixView<T, MatrixFormat::Dense>& a_in,
                          Span<typename base_type<T>::type> eigenvalues,
                          JobType jobz,
                          Uplo uplo,
                          const Span<std::byte>& ws,
                          JacobiParams<T> params) {
    validate_jacobi_blocked(a_in, jobz, "syev_jacobi_blocked");

    const int32_t n = static_cast<int32_t>(a_in.rows());
    const int64_t batch = a_in.batch_size();

    if (eigenvalues.size() < static_cast<std::size_t>(n) * static_cast<std::size_t>(batch)) {
        throw std::invalid_argument("syev_jacobi_blocked: eigenvalues span too small for n*batch.");
    }

    const std::size_t local_mem = ctx->get_device().get_info<sycl::info::device::local_mem_size>();
    const int32_t forced_nb = env_positive_int_or("BATCHLAS_JACOBI_BLOCKED_NB",
                                                  static_cast<int32_t>(params.block_size));
    const BlockPlan plan = plan_blocks<T>(n, local_mem, forced_nb);
    const int32_t wg = jacobi_blocked_wg_size(ctx, plan);

    auto& a = const_cast<MatrixView<T, MatrixFormat::Dense>&>(a_in);
    const bool upper = (uplo == Uplo::Upper);
    const bool vectors = (jobz == JobType::EigenVectors);

    if (vectors) {
        auto ws_mut = const_cast<Span<std::byte>&>(ws);
        BumpAllocator pool(ws_mut);
        auto z = pool.allocate<T>(ctx, static_cast<std::size_t>(n) * n * static_cast<std::size_t>(batch));
        syev_jacobi_blocked_impl<T, true>(ctx, a, eigenvalues.data(), z.data(), n, upper, plan, params, wg);
    } else {
        syev_jacobi_blocked_impl<T, false>(ctx, a, eigenvalues.data(), nullptr, n, upper, plan, params, wg);
    }

    return ctx.get_event();
}

template <Backend B, typename T>
size_t syev_jacobi_blocked_buffer_size(Queue& ctx,
                                       const MatrixView<T, MatrixFormat::Dense>& a,
                                       JobType jobz,
                                       JacobiParams<T> params) {
    (void)params;
    validate_jacobi_blocked(a, jobz, "syev_jacobi_blocked_buffer_size");
    if (jobz != JobType::EigenVectors) return 0;

    const std::size_t n = static_cast<std::size_t>(a.rows());
    const std::size_t batch = static_cast<std::size_t>(a.batch_size());
    return BumpAllocator::allocation_size<T>(ctx, n * n * batch);
}

#define SYEV_JACOBI_BLOCKED_INSTANTIATE(back, fp) \
    template Event syev_jacobi_blocked<back, BATCHLAS_UNPAREN fp>(Queue&, const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
                                                                  Span<typename base_type<BATCHLAS_UNPAREN fp>::type>, JobType, Uplo, \
                                                                  const Span<std::byte>&, JacobiParams<BATCHLAS_UNPAREN fp>); \
    template size_t syev_jacobi_blocked_buffer_size<back, BATCHLAS_UNPAREN fp>(Queue&, const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
                                                                               JobType, JacobiParams<BATCHLAS_UNPAREN fp>);

#define SYEV_JACOBI_BLOCKED_INSTANTIATE_FOR_BACKEND(back) \
    BATCHLAS_FOR_EACH_SCALAR_TYPE_1(SYEV_JACOBI_BLOCKED_INSTANTIATE, back)

#if BATCHLAS_HAS_CUDA_BACKEND
    SYEV_JACOBI_BLOCKED_INSTANTIATE_FOR_BACKEND(Backend::CUDA)
#endif

#if BATCHLAS_HAS_ROCM_BACKEND
    SYEV_JACOBI_BLOCKED_INSTANTIATE_FOR_BACKEND(Backend::ROCM)
#endif

#if BATCHLAS_HAS_HOST_BACKEND
    SYEV_JACOBI_BLOCKED_INSTANTIATE_FOR_BACKEND(Backend::NETLIB)
#endif

#undef SYEV_JACOBI_BLOCKED_INSTANTIATE_FOR_BACKEND
#undef SYEV_JACOBI_BLOCKED_INSTANTIATE

} // namespace batchlas
