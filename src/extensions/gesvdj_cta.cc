#include <blas/matrix.hh>
#include <blas/functions.hh>
#include <blas/extensions.hh>
#include <blas/extra.hh>
#include <util/kernel-heuristics.hh>
#include <util/mempool.hh>
#include <util/group-invoke.hh>
#include "sg_compat.hh"
#include <batchlas/backend_config.h>
#include "../math-helpers.hh"
#include "../queue.hh"
#include "../util/template-instantiations.hh"
#include <algorithm>
#include <complex>
#include <limits>
#include <numeric>
using namespace sycl::ext::oneapi;

namespace batchlas {

// Kernel name tag. Must live outside the anonymous namespace so it does not
// depend on internal-linkage entities.
template <typename T, size_t P, size_t C, bool ComputeV>
class GesvdjCTAKernel;

// ---------------------------------------------------------------------------
// One-sided (Hestenes) Jacobi SVD, partition-resident.
//
// One SubGroupPartition<P> owns one problem; the working matrix and (optionally)
// the accumulated right factor live in local memory for the whole solve, so a
// full SVD is a single kernel launch.
//
// WHY THIS EXISTS. The pre-existing gesvd_cta / gesvd_blocked paths form the
// tridiagonal of B^T B explicitly (gesvd_blocked.cc:220) and recover
// sigma = sqrt(lambda). That squares the condition number: measured relative
// error in the singular values is 7.8e-3 at kappa=10 and 2.13 at kappa=1e6
// (float, n=32), against cuSOLVER gesvdjBatched's 4.1e-6 and 9.3e-3, and the
// computed U/V stop being orthogonal at all by kappa=1e4. See GESVD_PLAN.md
// section 2.1 and benchmarks/gesvd_relacc.cc.
//
// One-sided Jacobi avoids that because sigma_i is a COLUMN NORM of the rotated
// A, never the square root of a difference of large numbers, and the rotations
// applied to A are orthogonal. The 2x2 Gram of a pivot pair is recomputed fresh
// from the current columns every time, so its rounding error is columnwise
// relative -- which is what makes the threshold
//     |a_pq| > tol * sqrt(|a_pp| * |a_qq|)
// a genuine relative test and delivers the Demmel-Veselic bound
//     |d sigma_i| / sigma_i <= O(eps) * kappa(A_c).
//
// Structurally this is syev_jacobi_cta with Phase 2 (the A <- U^H A row update)
// deleted -- that deletion is exactly the difference between two-sided and
// one-sided Jacobi -- plus a Gram phase, column norms, and an SVD epilogue.
//
// References:
// - Hestenes, "Inversion of matrices by biorthogonalization", 1958.
// - Demmel & Veselic, SIAM J. Matrix Anal. Appl. 13(4), 1992.
// - Drmac & Veselic, LAPACK Working Notes 169/170 (threshold form, SVA
//   recurrence, convergence test).
// - Golub & Van Loan, Matrix Computations, Alg. 8.5.1 (2x2 rotation).
// - GESVD_IMPL_SPEC.md Part C for the design decision and its review.
// ---------------------------------------------------------------------------

namespace {

template <typename U>
inline U conj_if_complex_g(const U& x) {
    if constexpr (internal::is_complex<U>::value) {
        return U(x.real(), -x.imag());
    } else {
        return x;
    }
}

template <typename U>
inline typename base_type<U>::type abs_if_complex_g(const U& x) {
    if constexpr (internal::is_complex<U>::value) {
        return sycl::hypot(x.real(), x.imag());
    } else {
        return sycl::fabs(x);
    }
}

template <typename U>
inline typename base_type<U>::type norm2_g(const U& x) {
    using Real = typename base_type<U>::type;
    if constexpr (internal::is_complex<U>::value) {
        return x.real() * x.real() + x.imag() * x.imag();
    } else {
        return static_cast<Real>(x * x);
    }
}

// permute_group_by_xor does not accept std::complex, so complex is shuffled as
// two real halves.
template <typename Group, typename U>
inline U xor_shuffle_g(const Group& g, const U& v, uint32_t mask) {
    if constexpr (internal::is_complex<U>::value) {
        return U(permute_group_by_xor(g, v.real(), mask),
                 permute_group_by_xor(g, v.imag(), mask));
    } else {
        return permute_group_by_xor(g, v, mask);
    }
}

// Butterfly all-reduce, result replicated across the partition. Must be called
// by every lane -- a non-participating lane poisons the result.
template <typename Group, typename U>
inline U part_sum_g(const Group& g, U v) {
    const uint32_t lanes = static_cast<uint32_t>(g.get_local_linear_range());
    for (uint32_t offset = lanes / 2; offset > 0; offset >>= 1) {
        v = v + xor_shuffle_g(g, v, offset);
    }
    return v;
}

template <typename Group, typename Real>
inline Real part_max_g(const Group& g, Real v) {
    const uint32_t lanes = static_cast<uint32_t>(g.get_local_linear_range());
    for (uint32_t offset = lanes / 2; offset > 0; offset >>= 1) {
        v = sycl::fmax(v, permute_group_by_xor(g, v, offset));
    }
    return v;
}

template <typename Group, typename Real>
inline Real part_min_g(const Group& g, Real v) {
    const uint32_t lanes = static_cast<uint32_t>(g.get_local_linear_range());
    for (uint32_t offset = lanes / 2; offset > 0; offset >>= 1) {
        v = sycl::fmin(v, permute_group_by_xor(g, v, offset));
    }
    return v;
}

// Round-robin ("chess tournament") pairing; identical to syev_jacobi_cta.cc:115.
// For even mp, round t in [0, mp-2] gives mp/2 disjoint pairs and the mp-1
// rounds cover every index pair exactly once.
inline void round_robin_pair_g(int32_t mp, int32_t t, int32_t k, int32_t& p, int32_t& q) {
    const int32_t ring = mp - 1;
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

template <typename T, size_t P, size_t C, bool ComputeV>
inline void gesvdj_cta_impl(Queue& ctx,
                            const MatrixView<T, MatrixFormat::Dense>& a_in,
                            typename base_type<T>::type* s_ptr,
                            const MatrixView<T, MatrixFormat::Dense>& u_in,
                            const MatrixView<T, MatrixFormat::Dense>& vh_in,
                            bool want_left,
                            bool transposed,
                            int32_t rows,     // R = max(m,n), rows of the solved matrix
                            int32_t cols,     // C = min(m,n), cols of the solved matrix
                            int32_t left_cols,// columns of the left factor to emit: R (All) or C (Thin)
                            GesvdjParams<T> params) {
    using Real = typename base_type<T>::type;

    const auto batch_size = a_in.batch_size();

    ctx->submit([&](sycl::handler& cgh) {
        auto A_view = a_in.kernel_view();
        auto U_view = u_in.kernel_view();
        auto Vh_view = vh_in.kernel_view();

        const auto dev = ctx->get_device();
        const int32_t sg_size = 32;

        // LD is odd, which is what makes the conj-transposed writeback
        // conflict-free: lane i reads [r + c_i*LD] with c_i a permutation, and
        // gcd(LD,32)=1 turns c -> c*LD mod 32 into a bijection. With LD == P the
        // same access serialises 32 ways.
        // P is the PARTITION WIDTH (lanes). C is the TILE CAPACITY (rows and
        // columns of the resident matrix). They are equal on every rung that
        // existed before n > 32 support, and C > P means each lane owns
        // kRPL = C/P rows rather than one.
        //
        // Splitting them is what keeps the n <= 32 path byte-identical: at
        // C == P every generalised expression below reduces syntactically to
        // what it was.
        static_assert(C % P == 0, "tile capacity must be a whole number of partition widths");
        static_assert(C <= 64, "int16 pair packing p|(q<<8) overflows above C=127; 64 is the tested cap");
        constexpr size_t kRPL = C / P;                       // rows per lane
        static_assert(kRPL == 1 || P == 32,
                      "multi-row lanes are only defined on a full 32-wide partition");
        constexpr int32_t LD = static_cast<int32_t>(C) + 1;
        constexpr size_t kTileElems = static_cast<size_t>(LD) * C;
        constexpr size_t kRotSlots = (C / 2 > 0) ? (C / 2) : 1;
        constexpr size_t kPairSlots = (C - 1) * kRotSlots;
        // Pairs processed per Gram reduce-scatter. Fixed at P/2 (capped by the
        // slot count for the small rungs) because the scatter lands pair k in
        // lanes 2k, 2k+1 -- more than P/2 pairs in flight has nowhere to land.
        constexpr size_t kGramChunk = (kRotSlots < P / 2) ? kRotSlots : (P / 2 > 0 ? P / 2 : 1);
        constexpr size_t kChunks = kRotSlots / kGramChunk;
        static_assert(kChunks * kGramChunk == kRotSlots, "round must split evenly into Gram chunks");
        constexpr bool kNeedPhase = internal::is_complex<T>::value;

        // Local-memory budget. Note this clamps probs_per_wg DIRECTLY rather
        // than the multiplier: syev_jacobi_cta.cc:193-201 clamps the multiplier
        // but usage is probs_per_wg * bytes_per_prob, and since
        // base_wg_size = lcm(P,32) = 32 for every supported P the two differ by
        // 32/P -- an under-count of up to 8x at P=4. Harmless there with one
        // small tile; not harmless with two resident tiles.
        const int32_t probs_per_warp = sg_size / static_cast<int32_t>(P);
        constexpr size_t kPairTabBytes = kPairSlots * sizeof(int16_t);
        const size_t bytes_per_prob =
            (1 + (ComputeV ? 1 : 0)) * kTileElems * sizeof(T)
            + C * sizeof(Real)
            + 2 * kRotSlots * sizeof(Real)
            + (kNeedPhase ? kRotSlots * sizeof(T) : 0)
            + C * sizeof(int16_t);

        const size_t local_mem_bytes = dev.get_info<sycl::info::device::local_mem_size>();
        const size_t avail = (local_mem_bytes > kPairTabBytes) ? (local_mem_bytes - kPairTabBytes) : 1;
        const int32_t max_probs = std::max<int32_t>(
            int32_t(1), static_cast<int32_t>(avail / std::max<size_t>(size_t(1), bytes_per_prob)));

        int32_t pw = std::min<int32_t>(
            probs_per_warp * std::max<int32_t>(int32_t(1), static_cast<int32_t>(params.cta_wg_size_multiplier)),
            max_probs);
        pw = std::max<int32_t>(1, (pw / probs_per_warp) * probs_per_warp);

        const int32_t max_wg_size = static_cast<int32_t>(dev.get_info<sycl::info::device::max_work_group_size>());
        while (pw > probs_per_warp && pw * static_cast<int32_t>(P) > max_wg_size) {
            pw -= probs_per_warp;
        }

        const int32_t probs_per_wg = pw;
        const int32_t wg_size = pw * static_cast<int32_t>(P);
        const int32_t nb = static_cast<int32_t>(batch_size);
        const int32_t num_wg = (nb + probs_per_wg - 1) / probs_per_wg;
        const int32_t global_size = num_wg * wg_size;
        const int32_t wg_sz = wg_size;

        // Conditionally unused accessors are sized 1, never 0, and their base is
        // forced to 0 (the trap at syev_jacobi_cta.cc:207-208).
        auto A_local = sycl::local_accessor<T, 1>(
            sycl::range<1>(static_cast<size_t>(probs_per_wg) * kTileElems), cgh);
        auto V_local = sycl::local_accessor<T, 1>(
            sycl::range<1>(ComputeV ? static_cast<size_t>(probs_per_wg) * kTileElems : 1), cgh);
        auto Nrm_local = sycl::local_accessor<Real, 1>(
            sycl::range<1>(static_cast<size_t>(probs_per_wg) * C), cgh);
        auto Rcs_local = sycl::local_accessor<sycl::vec<Real, 2>, 1>(
            sycl::range<1>(static_cast<size_t>(probs_per_wg) * kRotSlots), cgh);
        auto Rd_local = sycl::local_accessor<T, 1>(
            sycl::range<1>(kNeedPhase ? static_cast<size_t>(probs_per_wg) * kRotSlots : 1), cgh);
        auto Inv_local = sycl::local_accessor<int16_t, 1>(
            sycl::range<1>(static_cast<size_t>(probs_per_wg) * C), cgh);
        auto Pair_local = sycl::local_accessor<int16_t, 1>(sycl::range<1>(kPairSlots), cgh);

        const int32_t RR = rows;
        const int32_t CC = cols;
        // Columns of the left factor actually emitted: RR for All, CC for Thin.
        // Kernel-uniform, so every partition-wide reduction below stays uniform
        // and no barrier structure depends on it.
        const int32_t LC = left_cols;
        // Pivot index space padded to even so the round-robin schedule is well
        // defined; a padded index is never paired with a real one because pairs
        // touching index >= CC are skipped.
        const int32_t mp = (CC % 2 == 0) ? CC : (CC + 1);
        const int32_t max_sweeps = std::max<int32_t>(int32_t(1), static_cast<int32_t>(params.max_sweeps));
        const bool want_left_f = want_left;
        const bool transposed_f = transposed;

        // Relative off-diagonal threshold (Demmel & Veselic; LAWN 169 Remark
        // 2.2). The classical absolute test |a_pq| <= tol*max|a_kl| would
        // forfeit the entire relative-accuracy advantage that motivates this
        // kernel.
        const Real tol = params.tol_multiplier * static_cast<Real>(CC) * std::numeric_limits<Real>::epsilon();
        const Real tiny = std::numeric_limits<Real>::min();
        const Real tau_big = Real(1) / sycl::sqrt(std::numeric_limits<Real>::epsilon());
        const Real zero_mult = params.zero_sigma_multiplier;

        Real* S = s_ptr;
        int32_t* SW = (params.sweep_counts.size() >= static_cast<size_t>(batch_size))
                          ? params.sweep_counts.data()
                          : nullptr;

        // The 32 here is load-bearing and was previously only checked on the
        // host, which cannot constrain what the compiler picks. The exact-norm
        // reduction and the Gram reduce-scatter below are hardcoded 5- and
        // (4+1)-step butterflies, probs_per_warp is sg_size/P with sg_size
        // fixed at 32, and part_id assumes the same -- every one of those is
        // silently wrong at any other sub-group width. Sibling CTA kernels
        // (steqr_cta, syev_cta_fused, sytrd_sb2st_cta) all carry the attribute;
        // this one and syev_jacobi_cta did not.
        cgh.parallel_for<GesvdjCTAKernel<T, P, C, ComputeV>>(
            sycl::nd_range<1>(global_size, wg_size),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                const auto wg = it.get_group();
                const int32_t wg_id = static_cast<int32_t>(wg.get_group_linear_id());
                const int32_t local_id = static_cast<int32_t>(it.get_local_linear_id());

                const int32_t pairs_per_round = mp / 2;
                const int32_t rounds = mp - 1;

                // Pair table, shared by the whole work-group. Filled and read
                // with the COMPILE-TIME stride kRotSlots (not pairs_per_round):
                // those differ whenever CC < P, and a mismatch makes the
                // unrolled k loop dereference garbage column indices. Unused
                // slots get a sentinel that fails the `< CC` test.
                //
                // This must precede the `prob_id >= nb` early return, since it
                // ends in a work-group barrier.
                for (int32_t idx = local_id; idx < rounds * static_cast<int32_t>(kRotSlots); idx += wg_sz) {
                    const int32_t t = idx / static_cast<int32_t>(kRotSlots);
                    const int32_t k = idx - t * static_cast<int32_t>(kRotSlots);
                    int32_t p = 0;
                    int32_t q = 0;
                    if (k < pairs_per_round) {
                        round_robin_pair_g(mp, t, k, p, q);
                    } else {
                        p = static_cast<int32_t>(C);
                        q = static_cast<int32_t>(C);
                    }
                    Pair_local[idx] = static_cast<int16_t>(p | (q << 8));
                }
                sycl::group_barrier(wg);

                const auto sg = it.get_sub_group();
                const auto part = make_partition<P>(sg);

                const int32_t sg_id = static_cast<int32_t>(sg.get_group_linear_id());
                const int32_t parts_per_sg = static_cast<int32_t>(part.get_group_linear_range());
                const int32_t part_id = sg_id * parts_per_sg + static_cast<int32_t>(part.get_group_linear_id());

                const int32_t lane = static_cast<int32_t>(part.get_local_linear_id());
                const int32_t prob_id = wg_id * probs_per_wg + part_id;
                if (prob_id >= nb) return;

                auto A_prob = A_view.batch_item(prob_id);
                auto U_prob = U_view.batch_item(prob_id);
                auto Vh_prob = Vh_view.batch_item(prob_id);

                const int32_t base_a = part_id * static_cast<int32_t>(kTileElems);
                const int32_t base_v = ComputeV ? (part_id * static_cast<int32_t>(kTileElems)) : 0;
                const int32_t base_n = part_id * static_cast<int32_t>(C);
                const int32_t base_r = part_id * static_cast<int32_t>(kRotSlots);
                const int32_t base_p = part_id * static_cast<int32_t>(C);

                // ---- Load. lane = ROW. ----
                // Lane-as-row makes the global read A_prob(lane,c) (address
                // c*ld + lane) coalesced; lane-as-column would stride by ld.
                // For m < n we read A_prob(c,lane) instead, i.e. transpose at
                // load time -- free, since the tile is a few KB, and it avoids
                // gesvd_blocked's out-of-place transpose + recursion.
                // The pad region is written as exact zero so a padded pair's
                // Gram is identically 0 and falls below any threshold.
                // Lane owns rows lane, lane+P, lane+2P, ... At kRPL == 1 this
                // is textually the single-row loop it replaces. The global read
                // stays coalesced for every rr: A_prob(row, c) is at
                // c*ld + lane + rr*P, contiguous across lanes.
                for (int32_t rr = 0; rr < static_cast<int32_t>(kRPL); ++rr) {
                    const int32_t row = lane + rr * static_cast<int32_t>(P);
                    for (int32_t c = 0; c < static_cast<int32_t>(C); ++c) {
                        T v = T(0);
                        if (row < RR && c < CC) {
                            // For m < n we solve A^H, not A^T. A^H = U' S V'^H gives
                            // A = V' S U'^H, so U = V' and Vh = U'^H -- the SAME role
                            // mapping as the m >= n case, just swapped between the
                            // two outputs. Solving A^T instead would give
                            // A = conj(V') S U'^T, whose conjugations differ, and is
                            // wrong for complex (it is invisible in real arithmetic).
                            v = transposed_f ? conj_if_complex_g(A_prob(c, row)) : A_prob(row, c);
                        }
                        A_local[base_a + row + c * LD] = v;
                        if constexpr (ComputeV) {
                            V_local[base_v + row + c * LD] = (row == c && row < CC) ? T(1) : T(0);
                        }
                    }
                }
                group_barrier(part);

                // ---- Exact column norms (S0). ----
                // Reduce-scatter of P values over P lanes: 5 steps, no trailing
                // all-reduce needed because V == L here. Leaves lane c holding
                // ||A_c||^2.
                // x stays Real[P], NOT Real[C]. Widening it to 64 would cost 64
                // Real registers per lane and force a sixth reduction step; the
                // hardcoded 5 steps are correct because the LANE count is still
                // 32. Instead the C columns are covered in C/P passes of the
                // unchanged 32-wide reduce-scatter, and each lane's kRPL rows are
                // summed into x[c] before the butterfly runs.
                auto exact_norms = [&]() {
                    for (int32_t h = 0; h < static_cast<int32_t>(kRPL); ++h) {
                        const int32_t col0 = h * static_cast<int32_t>(P);
                        Real x[P];
#pragma unroll
                        for (int32_t c = 0; c < static_cast<int32_t>(P); ++c) {
                            Real acc = Real(0);
#pragma unroll
                            for (int32_t rr = 0; rr < static_cast<int32_t>(kRPL); ++rr) {
                                acc += norm2_g(A_local[base_a + lane + rr * static_cast<int32_t>(P)
                                                       + (col0 + c) * LD]);
                            }
                            x[c] = acc;
                        }
#pragma unroll
                        for (int32_t step = 0; step < 5; ++step) {
                            const uint32_t mask = static_cast<uint32_t>(P) >> (step + 1);
                            if (mask == 0u) break;
                            const bool hi = (static_cast<uint32_t>(lane) & mask) != 0u;
                            const int32_t half = static_cast<int32_t>(mask);
#pragma unroll
                            for (int32_t j = 0; j < half; ++j) {
                                const Real own = hi ? x[j + half] : x[j];
                                const Real send = hi ? x[j] : x[j + half];
                                x[j] = own + permute_group_by_xor(part, send, mask);
                            }
                        }
                        Nrm_local[base_n + col0 + lane] = x[0];
                    }
                };

                exact_norms();
                group_barrier(part);

                // ---- Global rescale (C.6). ----
                // A SINGLE global power-of-two factor. Per-column equilibration
                // would be a different matrix: scaling A by D gives
                // A*D = U S W^H, so A = U S (D^-1 W)^H and D^-1 W is not
                // orthogonal. kappa(A_c) is an ANALYSIS quantity, delivered by
                // the rotation/threshold formulas already being per-pair
                // scale-invariant, not by performing a scaling. LAPACK ?GESVJ
                // likewise scales only by a scalar.
                //
                // Centre on the geometric mean rather than the max: in float
                // that tolerates a column-norm ratio of ~1.7e38 instead of
                // ~9.2e18, and graded matrices -- the class the whole accuracy
                // argument is about -- are exactly where the ratio is large.
                Real my_n2 = (lane < CC) ? Nrm_local[base_n + lane] : Real(0);
                const Real nmax = part_max_g(part, my_n2);
                const Real nmin_in = (lane < CC && my_n2 > Real(0))
                                         ? my_n2
                                         : std::numeric_limits<Real>::max();
                const Real nmin = part_min_g(part, nmin_in);

                if (nmax == Real(0)) {
                    // Identically zero input: sigma = 0 and any orthonormal
                    // U/V is admissible. Fall through with beta = 1; the
                    // completion path fills U and V is already the identity.
                }
                Real beta = Real(1);
                if (nmax > Real(0) && nmin <= nmax) {
                    const Real e = sycl::round(Real(0.25) * (sycl::log2(nmax) + sycl::log2(nmin)));
                    beta = sycl::exp2(-e);
                }
                const Real inv_beta = Real(1) / beta;

                // ---- Global rescale ----
                // de Rijk pre-ordering (sorting columns by decreasing norm before
                // the first sweep) was implemented here and REMOVED. Measured mean
                // sweeps at n=32 float, kappa = 1e1 / 1e4 / 1e6:
                //     with    : 8.91 / 13.52 / 15.53
                //     without : 8.95 / 13.25 / 15.22
                // i.e. no reduction, and slightly worse at high conditioning.
                // Merely having the untaken branch in the kernel cost 13% of wall
                // clock (7.80 -> 8.88 ms at n=32/batch=16384) through register
                // pressure, so it is not worth keeping behind a runtime flag
                // either. See GESVD_PLAN.md Tier 2.
                if (beta != Real(1)) {
                    for (int32_t rr = 0; rr < static_cast<int32_t>(kRPL); ++rr) {
                        const int32_t row = lane + rr * static_cast<int32_t>(P);
                        for (int32_t c = 0; c < static_cast<int32_t>(C); ++c) {
                            A_local[base_a + row + c * LD] = A_local[base_a + row + c * LD] * T(beta);
                        }
                    }
                    group_barrier(part);
                    exact_norms();
                    group_barrier(part);
                }

                // ---- Sweeps ----
                // Termination requires TWO consecutive zero-rotation sweeps.
                // The second is the verification sweep: norms are recomputed
                // exactly at every sweep start, but within a sweep they are
                // maintained by the analytic recurrence and can drift, and a
                // drifted (inflated) a_pp raises the threshold so a genuinely
                // non-negligible a_pq is skipped. Without the re-check the loop
                // exits and sigma is a column norm of a NON-CONVERGED A -- a
                // silent wrong answer that sigma-comes-from-A does not prevent.
                // The verification sweep applies no rotations, so it costs only
                // the Gram + threshold pass.
                int32_t zero_sweeps = 0;
                int32_t sweeps_used = 0;
                for (int32_t sweep = 0; sweep < max_sweeps; ++sweep) {
                    sweeps_used = sweep + 1;
                    if (sweep > 0) {
                        exact_norms();
                        group_barrier(part);
                    }

                    int32_t rot_count = 0;

                    for (int32_t t = 0; t < rounds; ++t) {
                        const int32_t tab_base = t * static_cast<int32_t>(kRotSlots);

                        // The round is processed in CHUNKS of kGramChunk = P/2
                        // pairs. That constant is what keeps the Gram
                        // reduce-scatter, the k_of_lane = lane>>1 mapping and
                        // the `lane % 2 == 0` guards below EXACTLY as they were:
                        // the scatter lands chunk-local pair k in lanes 2k and
                        // 2k+1, which needs at most P/2 pairs in flight. A round
                        // at C=64 has 32 pairs, so it takes two chunks; at C=32
                        // there is one chunk and this loop disappears.
                        //
                        // Chunking is safe because a round's pairs are a perfect
                        // matching: chunks touch disjoint columns, so the
                        // Gram/apply of one cannot disturb another, and the Nrm
                        // writes never collide.
                        //
                        // It is also what stops the register arrays growing with
                        // C. Holding a whole C=64 round would need
                        // ap[32][2] + aq[32][2] + g[32] = 160 live T; chunked it
                        // is 16*2 + 16*2 + 16 = 80, against 48 at C=32.
                        for (int32_t ch = 0; ch < static_cast<int32_t>(kChunks); ++ch) {
                        const int32_t tab_base = t * static_cast<int32_t>(kRotSlots)
                                               + ch * static_cast<int32_t>(kGramChunk);

                        // Pair indices for the chunk, held in registers.
                        // Every index into pk/qk/ap/aq/g must be compile-time or
                        // the arrays spill to local memory.
                        int32_t pk[kGramChunk];
                        int32_t qk[kGramChunk];
                        T ap[kGramChunk][kRPL];
                        T aq[kGramChunk][kRPL];
                        T g[kGramChunk];

#pragma unroll
                        for (int32_t k = 0; k < static_cast<int32_t>(kGramChunk); ++k) {
                            const int32_t pq = static_cast<int32_t>(Pair_local[tab_base + k]);
                            pk[k] = pq & 0xFF;
                            qk[k] = (pq >> 8) & 0xFF;
                            const bool ok = (pk[k] < CC) && (qk[k] < CC);
                            const int32_t ip = ok ? pk[k] : 0;
                            const int32_t iq = ok ? qk[k] : 0;
                            // conj(A_p) * A_q, accumulated over this lane's rows
                            // BEFORE the butterfly, which then sums across lanes.
                            T acc = T(0);
#pragma unroll
                            for (int32_t rr = 0; rr < static_cast<int32_t>(kRPL); ++rr) {
                                const int32_t row = lane + rr * static_cast<int32_t>(P);
                                ap[k][rr] = A_local[base_a + row + ip * LD];
                                aq[k][rr] = A_local[base_a + row + iq * LD];
                                acc = acc + conj_if_complex_g(ap[k][rr]) * aq[k][rr];
                            }
                            g[k] = ok ? acc : T(0);
                        }

                        // Reduce-scatter: kGramChunk dot products in
                        // log2(kGramChunk) scatter steps PLUS log2(P/kGramChunk)
                        // all-reduce steps. At P=32, kGramChunk=16, that is 4 + 1.
                        //
                        // Writing it as five halving steps is WRONG and silently
                        // sums each dot product over only half the rows: the
                        // fifth `half` is 0 so its inner loop never runs. This
                        // was found by adversarial review before the kernel was
                        // written; see GESVD_IMPL_SPEC.md C.5 G3.
#pragma unroll
                        for (int32_t step = 0; step < 4; ++step) {
                            const uint32_t mask = static_cast<uint32_t>(kGramChunk) >> step;
                            if (mask == 0u) break;
                            const bool hi = (static_cast<uint32_t>(lane) & mask) != 0u;
                            const int32_t half = static_cast<int32_t>(mask) / 2;
                            if (half == 0) break;
#pragma unroll
                            for (int32_t j = 0; j < half; ++j) {
                                const T own = hi ? g[j + half] : g[j];
                                const T send = hi ? g[j] : g[j + half];
                                g[j] = own + xor_shuffle_g(part, send, mask);
                            }
                        }
                        // Final stage is an ALL-REDUCE over the surviving lane
                        // pair, not another halving.
                        g[0] = g[0] + xor_shuffle_g(part, g[0], 1u);
                        // g[0] now holds a_{p_k q_k} for k = lane>>1, replicated
                        // in lanes 2k and 2k+1.

                        const int32_t k_of_lane = lane >> 1;
                        // Slot index within the ROUND, which is what
                        // pairs_per_round counts.
                        const int32_t slot = ch * static_cast<int32_t>(kGramChunk) + k_of_lane;
                        // Re-read the pair from LDS rather than indexing pk[]
                        // with the runtime index k_of_lane: a runtime index into
                        // a register array spills the whole array.
                        const int32_t pq_l = static_cast<int32_t>(Pair_local[tab_base + k_of_lane]);
                        const int32_t kp = pq_l & 0xFF;
                        const int32_t kq = (pq_l >> 8) & 0xFF;

                        bool active = (slot < pairs_per_round) && (kp < CC) && (kq < CC);

                        Real c_rot = Real(1);
                        Real s_rot = Real(0);
                        T d_rot = T(1);
                        Real tt = Real(0);
                        Real gr = Real(0);
                        Real app = Real(0);
                        Real aqq = Real(0);

                        if (active) {
                            app = Nrm_local[base_n + kp];
                            aqq = Nrm_local[base_n + kq];
                            const T apq = g[0];
                            const Real g_abs = abs_if_complex_g(apq);
                            const Real thresh = tol * sycl::sqrt(sycl::fabs(app) * sycl::fabs(aqq));

                            if (g_abs > thresh && g_abs > tiny) {
                                Real gv;
                                if constexpr (internal::is_complex<T>::value) {
                                    gv = g_abs;
                                    d_rot = T(apq.real() / g_abs, -apq.imag() / g_abs);
                                } else {
                                    gv = apq;
                                    d_rot = T(1);
                                }
                                gr = gv;
                                const Real tau = (aqq - app) / (Real(2) * gv);
                                if (sycl::fabs(tau) > tau_big) {
                                    tt = Real(1) / (Real(2) * tau);
                                } else {
                                    tt = sycl::copysign(Real(1), tau)
                                       / (sycl::fabs(tau) + sycl::sqrt(Real(1) + tau * tau));
                                }
                                c_rot = Real(1) / sycl::sqrt(Real(1) + tt * tt);
                                s_rot = tt * c_rot;
                                // A rotation that rounds to the identity never
                                // annihilates a_pq, so counting it would keep
                                // the sweep loop alive for all max_sweeps.
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

                        // Analytic norm recurrence (LAPACK ?GESVJ maintains SVA
                        // the same way). Exact in exact arithmetic: with
                        // t^2 + 2*tau*t - 1 = 0 the updated norms are
                        // a_pp - t*a_pq and a_qq + t*a_pq.
                        // Even lanes only, so each column is written once.
                        if (active && (lane % 2 == 0)) {
                            // BOTH updates are clamped at zero. Clamping only the
                            // p side (the obvious cancelling one) lets the q side
                            // go negative on a badly conditioned problem; sqrt of
                            // that is NaN, every rank comparison against NaN is
                            // false, two columns then receive the same rank, and
                            // the Inv_local entry nobody wrote is read as a
                            // garbage column index -- an out-of-bounds LDS access,
                            // not a wrong number. Observed as
                            // CUDA_ERROR_ILLEGAL_ADDRESS at kappa >= 1e4.
                            Nrm_local[base_n + kp] = sycl::fmax(app - tt * gr, Real(0));
                            Nrm_local[base_n + kq] = sycl::fmax(aqq + tt * gr, Real(0));
                        }

                        if (lane % 2 == 0 && slot < static_cast<int32_t>(kRotSlots)) {
                            Rcs_local[base_r + slot] = sycl::vec<Real, 2>(c_rot, s_rot);
                            if constexpr (kNeedPhase) {
                                Rd_local[base_r + slot] = d_rot;
                            }
                        }
                        group_barrier(part);

                        // Counted on even lanes only. Must be executed by ALL
                        // lanes -- it is a butterfly XOR reduction and a
                        // non-participating lane poisons the result.
                        const int32_t round_active =
                            part_sum_g(part, (active && (lane % 2 == 0)) ? int32_t(1) : int32_t(0));
                        rot_count += round_active;
                        if (round_active == 0) continue;

                        // ---- A <- A*U, V <- V*U. lane owns rows lane+rr*P. ----
                        // ap[k][rr]/aq[k][rr] are still live from the Gram phase;
                        // that is kRPL*32 LDS loads per chunk this mapping saves
                        // and no other does. There is deliberately NO second
                        // phase: syev_jacobi_cta's A <- U^H A row update is what
                        // makes it two-sided, and deleting it is what makes this
                        // one-sided.
#pragma unroll
                        for (int32_t k = 0; k < static_cast<int32_t>(kGramChunk); ++k) {
                            const sycl::vec<Real, 2> cs =
                                Rcs_local[base_r + ch * static_cast<int32_t>(kGramChunk) + k];
                            const Real ck = cs[0];
                            const Real sk = cs[1];
                            // Warp-uniform skip: every lane reads the same slot,
                            // so a converged pair is genuinely free rather than
                            // predicated-off. Jacobi's last sweeps are almost
                            // entirely such rounds.
                            if (sk == Real(0)) continue;

                            T u11 = T(ck);
                            T u12 = T(sk);
                            T u21 = T(-sk);
                            T u22 = T(ck);
                            if constexpr (kNeedPhase) {
                                const T dk = Rd_local[base_r + ch * static_cast<int32_t>(kGramChunk) + k];
                                u21 = -(dk * T(sk));
                                u22 = dk * T(ck);
                            }

#pragma unroll
                            for (int32_t rr = 0; rr < static_cast<int32_t>(kRPL); ++rr) {
                                const int32_t row = lane + rr * static_cast<int32_t>(P);
                                A_local[base_a + row + pk[k] * LD] = ap[k][rr] * u11 + aq[k][rr] * u21;
                                A_local[base_a + row + qk[k] * LD] = ap[k][rr] * u12 + aq[k][rr] * u22;

                                if constexpr (ComputeV) {
                                    const int32_t vp = base_v + row + pk[k] * LD;
                                    const int32_t vq = base_v + row + qk[k] * LD;
                                    const T vpv = V_local[vp];
                                    const T vqv = V_local[vq];
                                    V_local[vp] = vpv * u11 + vqv * u21;
                                    V_local[vq] = vpv * u12 + vqv * u22;
                                }
                            }
                        }
                        group_barrier(part);
                        }
                    }

                    if (rot_count == 0) {
                        if (++zero_sweeps >= 2) break;
                    } else {
                        zero_sweeps = 0;
                    }
                }

                // ---- Epilogue ----
                // sigma comes from A, ALWAYS. Nrm_local's incrementally
                // maintained values exist only to choose rotations, where an
                // error perturbs the schedule; reading sigma from them instead
                // is a one-line shortcut that passes every existing test and
                // reintroduces the normal-equations defect through the side
                // door.
                exact_norms();
                group_barrier(part);


                // Rank sort, descending, ties broken on index so the
                // permutation is a bijection. Descending is the gesvd contract:
                // finalize_values_only produces it by index reversal
                // (gesvd_blocked.cc:305), and both has_tiny_singular_values and
                // patch_zero_left_vectors read sb[0] as sigma_max.
                if (SW != nullptr && lane == 0) {
                    SW[prob_id] = sweeps_used;
                }

                // Seed the permutation with the identity BEFORE the sort. The
                // sort writes Inv_local[rank] for each column, which covers every
                // slot only if the ranks are a bijection. That holds for any
                // finite input, but a defensive identity means a hypothetical
                // rank collision degrades to a wrong permutation rather than to
                // a garbage column index used to address local memory.
                // This is the ONE place where lane is a COLUMN index rather
                // than a row index, so it is the only place that needs a
                // columns-per-lane loop: lane owns columns lane, lane+P, ...
                for (int32_t cc = 0; cc < static_cast<int32_t>(kRPL); ++cc) {
                    const int32_t col = lane + cc * static_cast<int32_t>(P);
                    Inv_local[base_p + col] = static_cast<int16_t>(col);
                }
                group_barrier(part);

                for (int32_t cc = 0; cc < static_cast<int32_t>(kRPL); ++cc) {
                    const int32_t col = lane + cc * static_cast<int32_t>(P);
                    const Real sigma_col =
                        (col < CC) ? (inv_beta * sycl::sqrt(Nrm_local[base_n + col])) : Real(0);
                    if (col < CC) {
                        int32_t rank = 0;
                        for (int32_t j = 0; j < CC; ++j) {
                            const Real sj = inv_beta * sycl::sqrt(Nrm_local[base_n + j]);
                            const bool before = (sj > sigma_col) || (sj == sigma_col && j < col);
                            if (before) ++rank;
                        }
                        Inv_local[base_p + rank] = static_cast<int16_t>(col);
                    }
                    // Output columns CC..RR-1 of the left factor have no source
                    // column; park them on the free tile columns CC..RR-1, which
                    // the pad guarantees are zero and which no real column
                    // occupies. Disjoint from the rank slots above (0..CC-1).
                    if (col >= CC && col < RR) {
                        Inv_local[base_p + col] = static_cast<int16_t>(col);
                    }
                }
                group_barrier(part);

                const Real sigma_max = (CC > 0) ? (inv_beta * sycl::sqrt(Nrm_local[base_n + static_cast<int32_t>(Inv_local[base_p])])) : Real(0);
                // Relative to sigma_max only. gesvd_blocked.cc:317 uses
                // eps*fmax(1, sigma_max), which declares EVERY sigma zero on a
                // uniformly small input (sigma_max = 1e-10) and fabricates every
                // U column.
                const Real tol_zero = zero_mult * std::numeric_limits<Real>::epsilon() * sigma_max;

                for (int32_t cc = 0; cc < static_cast<int32_t>(kRPL); ++cc) {
                    const int32_t col = lane + cc * static_cast<int32_t>(P);
                    if (col < CC) {
                        const int32_t src = static_cast<int32_t>(Inv_local[base_p + col]);
                        S[static_cast<int64_t>(prob_id) * CC + col] =
                            inv_beta * sycl::sqrt(Nrm_local[base_n + src]);
                    }
                }

                // ---- Left factor: U_c = A_c / sigma_c ----
                if (want_left_f) {
                    // Normalise the accepted columns in place. A has been fully
                    // consumed into sigma by now, so overwriting it is safe.
                    for (int32_t cc = 0; cc < static_cast<int32_t>(kRPL); ++cc) {
                        const int32_t col = lane + cc * static_cast<int32_t>(P);
                        if (col >= CC) continue;
                        const Real n2_c = Nrm_local[base_n + col];
                        const Real s_c = inv_beta * sycl::sqrt(n2_c);
                        if (s_c > tol_zero && n2_c > Real(0)) {
                            // Divide by the norm of the SCALED column, which is
                            // sqrt(n2_c) -- not by sigma, which is that times
                            // 1/beta.
                            const Real inv_s = Real(1) / sycl::sqrt(n2_c);
                            for (int32_t r = 0; r < static_cast<int32_t>(C); ++r) {
                                A_local[base_a + r + col * LD] = A_local[base_a + r + col * LD] * T(inv_s);
                            }
                        }
                    }
                    group_barrier(part);

                    // Completion. Columns CC..RR-1, and any column whose sigma
                    // is below tol_zero, are not determined by A. Fill them from
                    // the orthogonal complement of the columns already accepted.
                    // patch_zero_left_vectors solves the same problem by copying
                    // from a second tridiagonal eigensolve, which is not
                    // available inside a fused kernel, so this is new code.
                    //
                    // Gated on a warp-uniform predicate: a well-conditioned
                    // square input pays one compare and skips the branch.
                    int32_t deficient_here = 0;
                    for (int32_t cc = 0; cc < static_cast<int32_t>(kRPL); ++cc) {
                        const int32_t col = lane + cc * static_cast<int32_t>(P);
                        if (col < CC && inv_beta * sycl::sqrt(Nrm_local[base_n + col]) <= tol_zero) {
                            ++deficient_here;
                        }
                    }
                    const int32_t any_def = part_sum_g(part, deficient_here);

                    // LC, not RR: a Thin request wants only the CC columns the
                    // solve already produced, so the completion block is skipped
                    // outright unless a column came out numerically deficient
                    // (any_def > 0), which still has to be repaired.
                    if (any_def > 0 || LC > CC) {
                        // The trial index cursor RUNS ACROSS dst; it is not reset
                        // per column. That is what makes this terminate.
                        //
                        // With a per-column restart and an "accept if residual
                        // norm > 1/2" rule (the original design), the last
                        // columns are never filled: once d dimensions remain out
                        // of RR, a canonical basis vector's residual norm^2 is
                        // only about d/RR, so at d = 1, RR = 32 every trial
                        // measures ~0.03 and is rejected. The column is then left
                        // zero and U is not orthogonal -- which is exactly how
                        // this showed up (defect exactly 1.0 on a 32x8 input).
                        //
                        // Running the cursor and accepting above 1/(2*RR) is
                        // provably sufficient: when d vectors are still needed
                        // and the cursor has consumed j0 canonical vectors, the
                        // remaining ones must contain one with residual norm^2 >=
                        // d/(RR - j0) >= 1/RR, so some trial always passes.
                        const Real accept_tol = Real(1) / (Real(2) * static_cast<Real>(RR));
                        int32_t jcur = 0;

                        for (int32_t dst = 0; dst < LC; ++dst) {
                            const int32_t cdst = static_cast<int32_t>(Inv_local[base_p + dst]);
                            // With LC == CC this never fires, so only genuinely
                            // deficient columns are rebuilt.
                            bool needs = (dst >= CC);
                            if (!needs) {
                                needs = (inv_beta * sycl::sqrt(Nrm_local[base_n + cdst])) <= tol_zero;
                            }
                            if (!needs) continue;

                            bool filled = false;
                            while (jcur < RR && !filled) {
                                const int32_t j = jcur++;
                                // The trial vector is distributed over the
                                // lane's kRPL rows. Each lane sums its own rows
                                // FIRST and the butterfly then sums across
                                // lanes -- the butterfly itself stays 32-wide.
                                T v[kRPL];
#pragma unroll
                                for (int32_t rr = 0; rr < static_cast<int32_t>(kRPL); ++rr) {
                                    const int32_t row = lane + rr * static_cast<int32_t>(P);
                                    v[rr] = (row == j) ? T(1) : T(0);
                                }
                                // TWO passes of classical Gram-Schmidt. One pass
                                // against an ill-conditioned accepted set loses
                                // exactly the orthogonality this patch exists to
                                // provide, in the near-deficient case that
                                // triggers it.
                                for (int32_t pass = 0; pass < 2; ++pass) {
                                    for (int32_t d2 = 0; d2 < dst; ++d2) {
                                        const int32_t c2 = static_cast<int32_t>(Inv_local[base_p + d2]);
                                        T part_dot = T(0);
                                        T qv[kRPL];
#pragma unroll
                                        for (int32_t rr = 0; rr < static_cast<int32_t>(kRPL); ++rr) {
                                            const int32_t row = lane + rr * static_cast<int32_t>(P);
                                            qv[rr] = (row < RR) ? A_local[base_a + row + c2 * LD] : T(0);
                                            part_dot = part_dot + conj_if_complex_g(qv[rr]) * v[rr];
                                        }
                                        const T dot = part_sum_g(part, part_dot);
#pragma unroll
                                        for (int32_t rr = 0; rr < static_cast<int32_t>(kRPL); ++rr) {
                                            v[rr] = v[rr] - qv[rr] * dot;
                                        }
                                    }
                                }
                                Real part_n2 = Real(0);
#pragma unroll
                                for (int32_t rr = 0; rr < static_cast<int32_t>(kRPL); ++rr) {
                                    const int32_t row = lane + rr * static_cast<int32_t>(P);
                                    if (row < RR) part_n2 += norm2_g(v[rr]);
                                }
                                const Real nrm2 = part_sum_g(part, part_n2);
                                if (nrm2 > accept_tol) {
                                    const Real inv_nr = Real(1) / sycl::sqrt(nrm2);
#pragma unroll
                                    for (int32_t rr = 0; rr < static_cast<int32_t>(kRPL); ++rr) {
                                        const int32_t row = lane + rr * static_cast<int32_t>(P);
                                        A_local[base_a + row + cdst * LD] = v[rr] * T(inv_nr);
                                    }
                                    filled = true;
                                }
                            }
                            group_barrier(part);
                        }
                    }
                    group_barrier(part);

                    // Writeback. Two orientations, each chosen so its own output
                    // coalesces.
                    if (!transposed_f) {
                        // U(lane, dst): global address dst*ld + lane, so
                        // consecutive lanes are contiguous.
                        // dst is the output COLUMN, so Thin truncates the loop.
                        for (int32_t rr = 0; rr < static_cast<int32_t>(kRPL); ++rr) {
                            const int32_t row = lane + rr * static_cast<int32_t>(P);
                            if (row >= RR) continue;
                            for (int32_t dst = 0; dst < LC; ++dst) {
                                const int32_t c = static_cast<int32_t>(Inv_local[base_p + dst]);
                                U_prob(row, dst) = A_local[base_a + row + c * LD];
                            }
                        }
                    } else {
                        // Vh(lane, r) = conj(L(r, c_lane)): lane is the OUTPUT
                        // ROW. The LDS read [r + c*LD] is conflict-free because
                        // LD is odd and c is a permutation.
                        //
                        // In THIS orientation lane is the rank index (what dst
                        // is above) while r runs over the output's columns, so
                        // the thin restriction lands on `lane`, not on `r`.
                        // Truncating r instead would emit a factor of the wrong
                        // shape while still writing something plausible.
                        for (int32_t cc = 0; cc < static_cast<int32_t>(kRPL); ++cc) {
                            const int32_t dst = lane + cc * static_cast<int32_t>(P);
                            if (dst >= LC) continue;
                            const int32_t c = static_cast<int32_t>(Inv_local[base_p + dst]);
                            for (int32_t r = 0; r < RR; ++r) {
                                Vh_prob(dst, r) = conj_if_complex_g(A_local[base_a + r + c * LD]);
                            }
                        }
                    }
                }

                // ---- Right factor: the accumulated rotation matrix ----
                if constexpr (ComputeV) {
                    if (!transposed_f) {
                        // Vh(lane, r) = conj(V(r, c_lane))
                        for (int32_t cc = 0; cc < static_cast<int32_t>(kRPL); ++cc) {
                            const int32_t dst = lane + cc * static_cast<int32_t>(P);
                            if (dst >= CC) continue;
                            const int32_t c = static_cast<int32_t>(Inv_local[base_p + dst]);
                            for (int32_t r = 0; r < CC; ++r) {
                                Vh_prob(dst, r) = conj_if_complex_g(V_local[base_v + r + c * LD]);
                            }
                        }
                    } else {
                        // U = V' directly. Here lane is the output ROW.
                        for (int32_t rr = 0; rr < static_cast<int32_t>(kRPL); ++rr) {
                            const int32_t row = lane + rr * static_cast<int32_t>(P);
                            if (row >= CC) continue;
                            for (int32_t dst = 0; dst < CC; ++dst) {
                                const int32_t c = static_cast<int32_t>(Inv_local[base_p + dst]);
                                U_prob(row, dst) = V_local[base_v + row + c * LD];
                            }
                        }
                    }
                }
            });
    });
}

// Largest max(m, n) this kernel accepts, per scalar type.
//
// The limit is local memory, and the binding constraint is OCCUPANCY rather
// than the hard cap. Per-problem LDS at C=64 with the V tile resident is
// 37,952 B for float, 71,744 B for double and complex<float>, and 138,816 B
// for complex<double>; this device reports 101,376 B, so complex<double> with
// vectors does not launch at all and the others fall to 2 or 1 work-groups per
// SM against 10 at C=32.
//
// Values-only halves it (no V tile), which is why the cap is job-dependent.
// The specific numbers here are set by measurement, not by the limit -- see
// the table in gesvd_supports_jacobi.
template <typename T>
constexpr int32_t gesvdj_cta_max_dim(bool want_vectors) {
    if constexpr (std::is_same_v<T, std::complex<double>>) {
        return want_vectors ? 32 : 64;
    } else {
        return 64;
    }
}

inline bool want_vectors_for_cap(SvdVectors jobu, SvdVectors jobvh) {
    return jobu != SvdVectors::None || jobvh != SvdVectors::None;
}

template <typename T>
void validate_gesvdj_dims(const MatrixView<T, MatrixFormat::Dense>& a,
                          Span<typename base_type<T>::type> singular_values,
                          const MatrixView<T, MatrixFormat::Dense>& u,
                          const MatrixView<T, MatrixFormat::Dense>& vh,
                          SvdVectors jobu,
                          SvdVectors jobvh,
                          const char* where) {
    if (a.batch_size() < 1 || a.rows() < 1 || a.cols() < 1) {
        throw std::invalid_argument(std::string(where) + ": invalid matrix dimensions or batch size");
    }
    const int64_t m = a.rows();
    const int64_t n = a.cols();
    const int64_t k = std::min(m, n);
    const int64_t batch = a.batch_size();
    if (singular_values.size() < static_cast<std::size_t>(k) * static_cast<std::size_t>(batch)) {
        throw std::invalid_argument(std::string(where) + ": singular_values span too small");
    }
    // Guard on "is computed at all" rather than "== All", and take the expected
    // column/row count from the job, so Thin is checked against m x k / k x n.
    // Testing `== All` here would let a Thin request through with an
    // unvalidated, probably wrongly-sized, output view.
    jobu = canonical_jobu(jobu, m, k);
    jobvh = canonical_jobvh(jobvh, n, k);
    if (jobu != SvdVectors::None) {
        const int64_t want_cols = svd_u_cols(jobu, m, k);
        if (u.rows() != m || u.cols() != want_cols || u.batch_size() != batch) {
            throw std::invalid_argument(std::string(where) + ": U must be (" +
                                        std::to_string(m) + " x " + std::to_string(want_cols) +
                                        ") with matching batch size");
        }
    }
    if (jobvh != SvdVectors::None) {
        const int64_t want_rows = svd_vh_rows(jobvh, n, k);
        if (vh.rows() != want_rows || vh.cols() != n || vh.batch_size() != batch) {
            throw std::invalid_argument(std::string(where) + ": Vh must be (" +
                                        std::to_string(want_rows) + " x " + std::to_string(n) +
                                        ") with matching batch size");
        }
    }
}

} // namespace

template <Backend B, typename T>
Event gesvdj_cta(Queue& ctx,
                 const MatrixView<T, MatrixFormat::Dense>& a_in,
                 Span<typename base_type<T>::type> singular_values,
                 const MatrixView<T, MatrixFormat::Dense>& u_out,
                 const MatrixView<T, MatrixFormat::Dense>& vh_out,
                 SvdVectors jobu,
                 SvdVectors jobvh,
                 const Span<std::byte>& ws,
                 GesvdjParams<T> params) {
    (void)ws;

    validate_gesvdj_dims(a_in, singular_values, u_out, vh_out, jobu, jobvh, "gesvdj_cta");

    const int32_t m = static_cast<int32_t>(a_in.rows());
    const int32_t n = static_cast<int32_t>(a_in.cols());
    {
        const int64_t k = std::min<int64_t>(m, n);
        jobu = canonical_jobu(jobu, m, k);
        jobvh = canonical_jobvh(jobvh, n, k);
    }
    if (std::max(m, n) > gesvdj_cta_max_dim<T>(want_vectors_for_cap(jobu, jobvh))) {
        throw std::invalid_argument(
            "gesvdj_cta: max(m, n) exceeds the supported cap for this scalar type "
            "(see gesvdj_cta_max_dim)");
    }

    {
        const auto dev = ctx->get_device();
        const auto sg_sizes = dev.get_info<sycl::info::device::sub_group_sizes>();
        bool has32 = false;
        for (auto s : sg_sizes) {
            if (static_cast<int32_t>(s) == 32) { has32 = true; break; }
        }
        if (!has32) {
            throw std::runtime_error("gesvdj_cta: device does not support subgroup size 32 required for CTA kernels.");
        }
    }

    const bool transposed = (m < n);
    const int32_t RR = transposed ? n : m;
    const int32_t CC = transposed ? m : n;

    const bool want_u = (jobu != SvdVectors::None);
    const bool want_vh = (jobvh != SvdVectors::None);
    // The R-sized factor lands in U when m >= n and in Vh when m < n, because
    // A^T = U' S V'^H gives A = V' S U'^H.
    const bool want_left = transposed ? want_vh : want_u;
    const bool want_right = transposed ? want_u : want_vh;

    // How many columns of the RR-sized left factor to produce. The solve yields
    // CC of them for free -- they are the rotated, normalised columns of A --
    // and any beyond that have to be manufactured by an in-kernel Gram-Schmidt
    // against canonical basis vectors. A Thin request wants exactly those CC, so
    // left_cols == CC skips that block entirely.
    //
    // Only the left factor can be thin: the right factor is CC x CC, and Thin
    // never shrinks it (the same m<=n / m>=n coincidence that canonicalisation
    // exploits).
    const SvdVectors job_left = transposed ? jobvh : jobu;
    const int32_t left_cols = (job_left == SvdVectors::All) ? RR : CC;

    auto* s_ptr = singular_values.data();

    // (P, C): P lanes per partition, C the tile capacity. Every rung here has
    // C == P, i.e. one row per lane -- the shape this kernel has always had.
    auto launch = [&](auto P_tag, auto C_tag) {
        constexpr size_t Pv = decltype(P_tag)::value;
        constexpr size_t Cv = decltype(C_tag)::value;
        if (want_right) {
            gesvdj_cta_impl<T, Pv, Cv, true>(ctx, a_in, s_ptr, u_out, vh_out, want_left, transposed, RR, CC, left_cols, params);
        } else {
            gesvdj_cta_impl<T, Pv, Cv, false>(ctx, a_in, s_ptr, u_out, vh_out, want_left, transposed, RR, CC, left_cols, params);
        }
    };

    constexpr auto k4 = std::integral_constant<size_t, 4>{};
    constexpr auto k8 = std::integral_constant<size_t, 8>{};
    constexpr auto k16 = std::integral_constant<size_t, 16>{};
    constexpr auto k32 = std::integral_constant<size_t, 32>{};
    constexpr auto k64 = std::integral_constant<size_t, 64>{};

    // P is capped at 32 by the sub-group width; above that the tile capacity C
    // grows instead and each lane takes kRPL = C/P rows.
    const int32_t md = std::max(m, n);
    if (md <= 4) {
        launch(k4, k4);
    } else if (md <= 8) {
        launch(k8, k8);
    } else if (md <= 16) {
        launch(k16, k16);
    } else if (md <= 32) {
        launch(k32, k32);
    } else {
        launch(k32, k64);
    }

    return ctx.get_event();
}

template <Backend B, typename T>
size_t gesvdj_cta_buffer_size(Queue& ctx,
                              const MatrixView<T, MatrixFormat::Dense>& a,
                              Span<typename base_type<T>::type> singular_values,
                              const MatrixView<T, MatrixFormat::Dense>& u_out,
                              const MatrixView<T, MatrixFormat::Dense>& vh_out,
                              SvdVectors jobu,
                              SvdVectors jobvh,
                              GesvdjParams<T> params) {
    (void)ctx;
    (void)params;
    validate_gesvdj_dims(a, singular_values, u_out, vh_out, jobu, jobvh, "gesvdj_cta_buffer_size");
    if (std::max(a.rows(), a.cols()) > gesvdj_cta_max_dim<T>(want_vectors_for_cap(jobu, jobvh))) {
        throw std::invalid_argument(
            "gesvdj_cta_buffer_size: max(m, n) exceeds the supported cap for this scalar type");
    }
    // Everything is LDS-resident for the lifetime of the kernel.
    return 0;
}

#define GESVDJ_CTA_INSTANTIATE(back, fp) \
    template Event gesvdj_cta<back, BATCHLAS_UNPAREN fp>( \
        Queue&, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        Span<typename base_type<BATCHLAS_UNPAREN fp>::type>, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        SvdVectors, SvdVectors, \
        const Span<std::byte>&, \
        GesvdjParams<BATCHLAS_UNPAREN fp>); \
    template size_t gesvdj_cta_buffer_size<back, BATCHLAS_UNPAREN fp>( \
        Queue&, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        Span<typename base_type<BATCHLAS_UNPAREN fp>::type>, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        SvdVectors, SvdVectors, \
        GesvdjParams<BATCHLAS_UNPAREN fp>);

#define GESVDJ_CTA_INSTANTIATE_FOR_BACKEND(back) \
    BATCHLAS_FOR_EACH_SCALAR_TYPE_1(GESVDJ_CTA_INSTANTIATE, back)

#if BATCHLAS_HAS_CUDA_BACKEND
    GESVDJ_CTA_INSTANTIATE_FOR_BACKEND(Backend::CUDA)
#endif

#if BATCHLAS_HAS_ROCM_BACKEND
    GESVDJ_CTA_INSTANTIATE_FOR_BACKEND(Backend::ROCM)
#endif

#if BATCHLAS_HAS_HOST_BACKEND
    GESVDJ_CTA_INSTANTIATE_FOR_BACKEND(Backend::NETLIB)
#endif

#undef GESVDJ_CTA_INSTANTIATE_FOR_BACKEND
#undef GESVDJ_CTA_INSTANTIATE

} // namespace batchlas
