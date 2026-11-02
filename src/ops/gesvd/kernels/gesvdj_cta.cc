// SPDX-License-Identifier: MIT
#include <batchlas/blas/device/detail/group_blas_common.hh>
#include <batchlas/blas/device/v2/dla.hh>
#include <batchlas/blas/device/v2/ext/jacobi.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/blas/functions.hh>
#include <batchlas/blas/extensions.hh>
#include <batchlas/blas/extra.hh>
#include <batchlas/util/kernel-heuristics.hh>
#include <batchlas/util/mempool.hh>
#include <batchlas/util/group-invoke.hh>
#include <batchlas/util/sycl-local-accessor-helpers.hh>
#include <src/kernels/common/dla_v2_partition.hh>
#include <src/ops/gesvd/gesvd_native.hh>
#include <src/kernels/common/sg_partition/sg_partition.hh>
#include <batchlas/backend_config.h>
#include <src/math-helpers.hh>
#include <src/queue.hh>
#include <src/util/template-instantiations.hh>
#include <src/kernels/common/info_span.hh>
#include <src/sycl/kernel_attrs.hh>
#include <src/sycl/local_mem.hh>
#include <src/util/batch_mask.hh>
#include <algorithm>
#include <complex>
#include <limits>
#include <numeric>

namespace batchlas {

// Kernel name tag: outside the anonymous namespace so it does not depend on
// internal-linkage entities.
template <typename T, size_t P, size_t C, bool ComputeV>
class GesvdjCTAKernel;

// One-sided (Hestenes) Jacobi SVD. One SubGroupPartition<P> owns one problem;
// A and (optionally) V stay in local memory for the whole solve, one launch.
// sigma is a column norm of the rotated A, never sqrt(lambda) of B^T B, which
// is what keeps relative accuracy (Demmel-Veselic; LAWN 169/170).
// evidence: docs/design/gesvd.md#gesvd-design-why-one-sided-jacobi
// evidence: docs/design/gesvd.md#gesvdj_cta-the-lane-equals-row-mapping-decision

namespace {

namespace v2 = ::batchlas::device::v2;

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
                            GesvdjParams<T> params,
                            int32_t* info,
                            const std::uint8_t* mask) {
    using Real = typename base_type<T>::type;

    const auto batch_size = a_in.batch_size();

    ctx->submit([&](sycl::handler& cgh) {
        auto A_view = a_in.kernel_view();
        auto U_view = u_in.kernel_view();
        auto Vh_view = vh_in.kernel_view();

        const auto dev = ctx->get_device();
        const int32_t sg_size = 32;

        // P = partition width (lanes), C = tile capacity; C > P means each lane
        // owns kRPL = C/P rows. LD = C+1 is odd so the V^H writeback is
        // bank-conflict-free.
        // evidence: docs/design/gesvd.md#gesvdj_cta-local-memory-budget-formula
        static_assert(C % P == 0, "tile capacity must be a whole number of partition widths");
        static_assert(C <= 64, "int16 pair packing p|(q<<8) overflows above C=127; 64 is the tested cap");
        constexpr size_t kRPL = C / P;                       // rows per lane
        static_assert(kRPL == 1 || P == 32,
                      "multi-row lanes are only defined on a full 32-wide partition");
        constexpr int32_t LD = static_cast<int32_t>(C) + 1;
        constexpr size_t kTileElems = static_cast<size_t>(LD) * C;
        constexpr size_t kRotSlots = (C / 2 > 0) ? (C / 2) : 1;
        constexpr size_t kPairSlots = (C - 1) * kRotSlots;
        // At most P/2 pairs per reduce-scatter: pair k lands in lanes 2k, 2k+1.
        constexpr size_t kGramChunk = (kRotSlots < P / 2) ? kRotSlots : (P / 2 > 0 ? P / 2 : 1);
        constexpr size_t kChunks = kRotSlots / kGramChunk;
        static_assert(kChunks * kGramChunk == kRotSlots, "round must split evenly into Gram chunks");
        constexpr bool kNeedPhase = detail::is_complex<T>::value;

        // Clamp probs_per_wg DIRECTLY, not a multiplier: a clamp in problems
        // under-counts it by 32/P, a launch failure with two tiles.
        // evidence: docs/design/gesvd.md#gesvdj_cta-local-memory-budget-formula
        const int32_t probs_per_warp = sg_size / static_cast<int32_t>(P);
        constexpr size_t kPairTabBytes = kPairSlots * sizeof(int16_t);
        const size_t bytes_per_prob =
            (1 + (ComputeV ? 1 : 0)) * kTileElems * sizeof(T)
            + C * sizeof(Real)
            + 2 * kRotSlots * sizeof(Real)
            + (kNeedPhase ? kRotSlots * sizeof(T) : 0)
            + C * sizeof(int16_t);

        const size_t local_mem_bytes = impl::local_mem_bytes(dev);
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
        if (static_cast<size_t>(pw) * bytes_per_prob + kPairTabBytes > local_mem_bytes) {
            throw batchlas::unsupported("gesvdj_cta: one work-group needs " +
                                        std::to_string(static_cast<size_t>(pw) * bytes_per_prob + kPairTabBytes) +
                                        " B of local memory; the device launches " + std::to_string(local_mem_bytes));
        }
        const int32_t nb = static_cast<int32_t>(batch_size);
        const int32_t num_wg = (nb + probs_per_wg - 1) / probs_per_wg;
        const int32_t global_size = num_wg * wg_size;
        const int32_t wg_sz = wg_size;

        // Conditionally unused accessors are sized 1, never 0, base forced to 0.
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
        // Left-factor columns emitted (RR All, CC Thin); kernel-uniform, so no
        // reduction or barrier depends on it.
        const int32_t LC = left_cols;
        // Padded to even for the round-robin; pairs touching >= CC are skipped.
        const int32_t mp = (CC % 2 == 0) ? CC : (CC + 1);
        const int32_t max_sweeps = std::max<int32_t>(int32_t(1), static_cast<int32_t>(params.max_sweeps));
        const bool want_left_f = want_left;
        const bool transposed_f = transposed;

        // Relative threshold (LAWN 169 Remark 2.2); an absolute test would
        // forfeit the relative accuracy this kernel exists for.
        const auto thr = v2::JacobiThreshold<Real>::relative(params.tol_multiplier, CC);
        const Real zero_mult = params.zero_sigma_multiplier;

        Real* S = s_ptr;
        int32_t* SW = (params.sweep_counts.size() >= static_cast<size_t>(batch_size))
                          ? params.sweep_counts.data()
                          : nullptr;
        // nullptr when status was not requested, making info_store a no-op.
        int32_t* const info_dev = info;

        // reqd_sub_group_size(32) is load-bearing: the 5- and (4+1)-step
        // butterflies, probs_per_warp and part_id are silently wrong otherwise.
        batchlas::impl::parallel_for<GesvdjCTAKernel<T, P, C, ComputeV>>(cgh,
            sycl::nd_range<1>(global_size, wg_size),
            [=](sycl::nd_item<1> it) BATCHLAS_REQD_SG_SIZE(32) {
                constexpr int kP = static_cast<int>(P);
                constexpr int kC = static_cast<int>(C);
                constexpr int K = static_cast<int>(kGramChunk);
                using Rows = v2::Frag<T, static_cast<int>(kRPL)>;  // one column on this lane's rows
                const auto wg = it.get_group();
                const int32_t wg_id = static_cast<int32_t>(wg.get_group_linear_id());
                const int32_t local_id = static_cast<int32_t>(it.get_local_linear_id());

                const int32_t pairs_per_round = mp / 2;
                const int32_t rounds = mp - 1;

                // Pair table: stride kRotSlots (not pairs_per_round, they differ
                // when CC < P); unused slots hold a sentinel failing `< CC`.
                for (int32_t idx = local_id; idx < rounds * static_cast<int32_t>(kRotSlots); idx += wg_sz) {
                    const int32_t t = idx / static_cast<int32_t>(kRotSlots);
                    const int32_t k = idx - t * static_cast<int32_t>(kRotSlots);
                    Pair_local[idx] = v2::pack_pair(k < pairs_per_round ? v2::round_robin_pair(mp, t, k) : v2::PivotPair{kC, kC});
                }
                sycl::group_barrier(wg);

                const auto sg = it.get_sub_group();
                const v2::SgpPartition<kP> ex{sg};
                const int32_t sg_id = static_cast<int32_t>(sg.get_group_linear_id());
                const int32_t parts_per_sg = static_cast<int32_t>(ex.part.get_group_linear_range());
                const int32_t part_id = sg_id * parts_per_sg + static_cast<int32_t>(ex.part.get_group_linear_id());

                const int32_t lane = ex.lid();
                const int32_t prob_id = wg_id * probs_per_wg + part_id;
                // NO EARLY RETURN (I2): partitions share a sub-group's barriers, so a dead
                // partition (a batch tail or an inactive item) solves a zero tile and stores nothing.
                const bool live = prob_id < nb && (mask == nullptr || mask[prob_id] != 0);
                const int32_t pid = live ? prob_id : 0;

                auto A_prob = A_view.batch_item(pid);
                auto U_prob = U_view.batch_item(pid);
                auto Vh_prob = Vh_view.batch_item(pid);

                // Tiles and slot arrays index the kernel's accessors (acc[base + i + j*LD]).
                const auto A = v2::acc_tile<kC, kC, LD>(A_local, part_id * static_cast<int32_t>(kTileElems));
                // Without V, V_local is never named, so the kernel does not capture it (an extra
                // local accessor argument moves the shared-memory layout).
                std::conditional_t<ComputeV, decltype(v2::acc_tile<kC, kC, LD>(V_local, 0)), v2::NoPhase> V{};
                if constexpr (ComputeV) V = v2::acc_tile<kC, kC, LD>(V_local, part_id * static_cast<int32_t>(kTileElems));
                const auto nrm = v2::acc_vec(Nrm_local, part_id * kC);
                const auto inv = v2::acc_vec(Inv_local, part_id * kC);
                const int32_t base_r = part_id * static_cast<int32_t>(kRotSlots);

                // ---- Load. lane = ROW (rows lane, lane+P, ...), coalesced. ----
                // m < n is transposed at load time. The pad is exact zero so a
                // padded pair's Gram is 0 and falls below any threshold.
                for (int32_t rr = 0; rr < static_cast<int32_t>(kRPL); ++rr) {
                    const int32_t row = lane + rr * kP;
                    for (int32_t c = 0; c < kC; ++c) {
                        T v = T(0);
                        if (live && row < RR && c < CC) {
                            // Solve A^H, not A^T: A^T is wrong for complex only.
                            // evidence: docs/design/gesvd.md#gesvdj_cta-rank-deficiency-thin-and-m--n
                            v = transposed_f ? v2::detail::conj_ref<T>(A_prob(c, row)) : A_prob(row, c);
                        }
                        A(row, c) = v;
                        if constexpr (ComputeV) V(row, c) = (row == c && row < CC) ? T(1) : T(0);
                    }
                }
                v2::sub_barrier(ex);

                // ---- Exact column norms: lane c ends holding ||A_c||^2. ----
                // P columns per pass reduce-scattered over P lanes (no all-reduce); C/P passes.
                auto exact_norms = [&]() {
                    for (int32_t h = 0; h < static_cast<int32_t>(kRPL); ++h) {
                        nrm[h * kP + lane] = v2::col_sumsq(ex, A.sub(0, h * kP, v2::Ext<kC>{}, v2::Ext<kP>{}));
                    }
                };

                exact_norms();
                v2::sub_barrier(ex);

                // ---- Global rescale: ONE power-of-two factor, centred on the
                // geometric mean. Per-column equilibration is not a factorisation
                // of A. Only columns < P feed nmax/nmin (the C=64 rung's upper
                // half does not); beta stays exact either way. Zero input: beta = 1;
                // completion fills U, V is identity.
                // evidence: docs/design/gesvd.md#gesvdj_cta-global-power-of-two-scaling
                const Real my_n2 = (lane < CC) ? nrm[lane] : Real(0);
                const Real nmax = ex.team_reduce(my_n2, v2::FMax{});
                const Real nmin = ex.team_reduce((lane < CC && my_n2 > Real(0)) ? my_n2 : std::numeric_limits<Real>::max(),
                                                 v2::FMin{});
                Real beta = Real(1);
                if (nmax > Real(0) && nmin <= nmax) {
                    const Real e = sycl::round(Real(0.25) * (sycl::log2(nmax) + sycl::log2(nmin)));
                    beta = sycl::exp2(-e);
                }
                const Real inv_beta = Real(1) / beta;

                // Do not add de Rijk pre-ordering here, even behind a flag: it
                // saved no sweeps and an untaken branch cost 13%.
                // evidence: docs/perf/gesvd.md#gesvdj_cta-tier-2-preconditioning-tested-and-rejected
                if (beta != Real(1)) {
                    for (int32_t rr = 0; rr < static_cast<int32_t>(kRPL); ++rr) {
                        const int32_t row = lane + rr * kP;
                        for (int32_t c = 0; c < kC; ++c) A(row, c) = A(row, c) * T(beta);
                    }
                    v2::sub_barrier(ex);
                    exact_norms();
                    v2::sub_barrier(ex);
                }

                // ---- Sweeps. Terminate only after TWO consecutive zero-rotation
                // sweeps: in-sweep norms drift, and one clean sweep can be a
                // drifted threshold skipping a live a_pq (silent wrong sigma).
                // evidence: docs/design/gesvd.md#gesvdj_cta-sweep-convergence-and-extraction-rules
                int32_t zero_sweeps = 0;
                int32_t sweeps_used = 0;
                for (int32_t sweep = 0; sweep < max_sweeps; ++sweep) {
                    sweeps_used = sweep + 1;
                    if (sweep > 0) {
                        exact_norms();
                        v2::sub_barrier(ex);
                    }

                    int32_t rot_count = 0;

                    for (int32_t t = 0; t < rounds; ++t) {
                        // Chunks of K pairs (two at C=64). Safe because a round is a
                        // perfect matching: chunks touch disjoint columns. Also caps
                        // live registers (80 T, not 160).
                        for (int32_t ch = 0; ch < static_cast<int32_t>(kChunks); ++ch) {
                        // Real T never names Rd_local, so the kernel does not capture it.
                        std::conditional_t<kNeedPhase, v2::AccVec<decltype(Rd_local)>, v2::NoPhase> rd{};
                        if constexpr (kNeedPhase) rd = v2::acc_vec(Rd_local, base_r + ch * K);
                        const auto rnd = v2::rot_round<T>(v2::acc_vec(Rcs_local, base_r + ch * K), rd,
                                                          v2::acc_vec(Pair_local, t * static_cast<int32_t>(kRotSlots) + ch * K));

                        // Gram entries a_{p q} = A_p^H A_q of the chunk's pairs over this lane's
                        // rows. Every index into pk/qk/ap/aq/g must be compile-time or they spill.
                        // Plain arrays and one fused load+dot loop: a PivotPair array with split
                        // load_owned / dotc_owned loops gave complex T +8..17 BRA.
                        int32_t pk[K];
                        int32_t qk[K];
                        T ap[K][kRPL];
                        T aq[K][kRPL];
                        v2::Frag<T, K> g;
BATCHLAS_UNROLL_FULL
                        for (int32_t k = 0; k < K; ++k) {
                            const v2::PivotPair pqk = rnd.pair(k);
                            pk[k] = pqk.p;
                            qk[k] = pqk.q;
                            const bool ok = (pk[k] < CC) && (qk[k] < CC);
                            const T acc = v2::load_dotc_owned<static_cast<int>(kRPL)>(ex, A.col(ok ? pk[k] : 0),
                                                                                     A.col(ok ? qk[k] : 0), ap[k], aq[k]);
                            g[k] = ok ? acc : T(0);
                        }

                        // Reduce-scatter is 4 scatter + 1 all-reduce step at P=32: lane l ends
                        // with a_{p q} of pair l >> 1. Five halving steps silently sum over half the rows.
                        // evidence: docs/design/gesvd.md#gesvdj_cta-the-reduce-scatter-g3-trap
                        const T apq = v2::reduce_scatter<K>(ex, g);

                        const int32_t k_of_lane = lane >> 1;
                        const int32_t slot = ch * K + k_of_lane;
                        // Re-read from LDS: a runtime index into pq[] would spill it.
                        const v2::PivotPair kpq = rnd.pair(k_of_lane);

                        bool active = (slot < pairs_per_round) && (kpq.p < CC) && (kpq.q < CC);
                        Real app = Real(0);
                        Real aqq = Real(0);
                        v2::JacobiRotation<T> jr = v2::identity_rotation<T>();
                        if (active) {
                            app = nrm[kpq.p];
                            aqq = nrm[kpq.q];
                            // An identity rotation must not count, or every problem burns all max_sweeps.
                            jr = v2::sym_schur2<T>(app, aqq, apq, thr);
                        }
                        active = jr.active;

                        // Analytic norm recurrence (as ?GESVJ's SVA), even lanes
                        // only. BOTH sides clamp at zero: a negative q side NaNs
                        // the rank sort into an out-of-bounds LDS index.
                        if (active && (lane % 2 == 0)) {
                            const auto n2 = v2::rotated_norms2(jr, app, aqq);
                            nrm[kpq.p] = n2.p;
                            nrm[kpq.q] = n2.q;
                        }
                        if (lane % 2 == 0 && slot < static_cast<int32_t>(kRotSlots)) rnd.store(k_of_lane, jr);
                        v2::sub_barrier(ex);

                        // Every lane must take part in the count.
                        const int32_t round_active = v2::count(ex, active && (lane % 2 == 0));
                        rot_count += round_active;
                        if (round_active == 0) continue;

                        // ---- A <- A*U, V <- V*U, reusing ap/aq from the Gram.
                        // Deliberately no A <- U^H A phase: that is two-sided.
BATCHLAS_UNROLL_FULL
                        for (int32_t k = 0; k < K; ++k) {
                            const sycl::vec<Real, 2> cs = rnd.cs[k];
                            // Warp-uniform skip: converged pairs are free.
                            if (cs[1] == Real(0)) continue;
                            const v2::Rot2<T> u = rnd.template rot<false>(k, cs[0], cs[1]);
                            v2::for_each_owned_static<kC, true>(ex, [&](int rr, int row) {
                                const v2::Pair2<T> r = v2::rot_right(u, ap[k][rr], aq[k][rr]);
                                A(row, pk[k]) = r.x;
                                A(row, qk[k]) = r.y;
                                if constexpr (ComputeV) v2::rot_right_inplace(u, V(row, pk[k]), V(row, qk[k]));
                            });
                        }
                        v2::sub_barrier(ex);
                        }
                    }

                    if (rot_count == 0) {
                        if (++zero_sweeps >= 2) break;
                    } else {
                        zero_sweeps = 0;
                    }
                }

                // ---- Epilogue. sigma comes from A, ALWAYS: the incremental
                // norms only choose rotations; reading sigma from them passes
                // every test and reintroduces the normal-equations defect.
                exact_norms();
                v2::sub_barrier(ex);

                if (live && SW != nullptr && lane == 0) {
                    SW[prob_id] = sweeps_used;
                }
                // Converged means `zero_sweeps >= 2`, not `sweeps_used <
                // max_sweeps`. zero_sweeps is partition-uniform, so lane 0 reports.
                // A STORE, not a raise: this kernel is the item's single writer.
                if (live && lane == 0) {
                    detail::info_store(info_dev, prob_id, (zero_sweeps < 2) ? 1 : 0);
                }

                // Rank sort, DESCENDING (the gesvd contract), ties on index. The
                // identity seed makes a rank collision a wrong permutation, not
                // a garbage LDS index. Here lane is a COLUMN index.
                // Plain owned-column loops here: a for_each_owned_static lambda is simplified and
                // unrolled as its own function before inlining, which moved LDS/STS and REG.
                for (int32_t cc = 0; cc < static_cast<int32_t>(kRPL); ++cc) {
                    const int32_t col = lane + cc * kP;
                    inv[col] = static_cast<int16_t>(col);
                }
                v2::sub_barrier(ex);

                for (int32_t cc = 0; cc < static_cast<int32_t>(kRPL); ++cc) {
                    const int32_t col = lane + cc * kP;
                    const Real sigma_col = (col < CC) ? (inv_beta * sycl::sqrt(nrm[col])) : Real(0);
                    if (col < CC) {
                        int32_t rank = 0;
                        for (int32_t j = 0; j < CC; ++j) {
                            const Real sj = inv_beta * sycl::sqrt(nrm[j]);
                            const bool before = (sj > sigma_col) || (sj == sigma_col && j < col);
                            if (before) ++rank;
                        }
                        inv[rank] = static_cast<int16_t>(col);
                    }
                    // Left columns CC..RR-1 park on the zero pad columns,
                    // disjoint from the rank slots 0..CC-1.
                    if (col >= CC && col < RR) {
                        inv[col] = static_cast<int16_t>(col);
                    }
                }
                v2::sub_barrier(ex);

                const Real sigma_max = (CC > 0) ? (inv_beta * sycl::sqrt(nrm[static_cast<int32_t>(inv[0])])) : Real(0);
                // Relative to sigma_max only; eps*fmax(1, sigma_max) would zero
                // every sigma of a uniformly small input.
                const Real tol_zero = zero_mult * std::numeric_limits<Real>::epsilon() * sigma_max;

                for (int32_t cc = 0; cc < static_cast<int32_t>(kRPL); ++cc) {
                    const int32_t col = lane + cc * kP;
                    if (live && col < CC) {
                        const int32_t src = static_cast<int32_t>(inv[col]);
                        S[static_cast<int64_t>(prob_id) * CC + col] = inv_beta * sycl::sqrt(nrm[src]);
                    }
                }

                // ---- Left factor: U_c = A_c / sigma_c ----
                if (want_left_f) {
                    // Normalise in place by the SCALED norm sqrt(n2_c), not sigma; A
                    // is fully consumed into sigma.
                    for (int32_t cc = 0; cc < static_cast<int32_t>(kRPL); ++cc) {
                        const int32_t col = lane + cc * kP;
                        if (col >= CC) continue;
                        const Real n2_c = nrm[col];
                        const Real s_c = inv_beta * sycl::sqrt(n2_c);
                        if (s_c > tol_zero && n2_c > Real(0)) {
                            const Real inv_s = Real(1) / sycl::sqrt(n2_c);
                            for (int32_t r = 0; r < kC; ++r) A(r, col) = A(r, col) * T(inv_s);
                        }
                    }
                    v2::sub_barrier(ex);

                    // Completion of columns CC..RR-1 and any sigma <= tol_zero,
                    // behind a warp-uniform gate.
                    int32_t deficient_here = 0;
                    for (int32_t cc = 0; cc < static_cast<int32_t>(kRPL); ++cc) {
                        const int32_t col = lane + cc * kP;
                        if (col < CC && inv_beta * sycl::sqrt(nrm[col]) <= tol_zero) {
                            ++deficient_here;
                        }
                    }
                    const int32_t any_def = ex.team_reduce(deficient_here);

                    // LC, not RR: Thin skips completion unless a column is deficient.
                    if (any_def > 0 || LC > CC) {
                        // The trial cursor RUNS ACROSS dst (never reset), with
                        // acceptance above 1/(2*RR): that is what terminates.
                        // evidence: docs/design/gesvd.md#gesvdj_cta-rank-deficiency-thin-and-m--n
                        const Real accept_tol = Real(1) / (Real(2) * static_cast<Real>(RR));
                        int32_t jcur = 0;

                        for (int32_t dst = 0; dst < LC; ++dst) {
                            const int32_t cdst = static_cast<int32_t>(inv[dst]);
                            bool needs = (dst >= CC);
                            if (!needs) {
                                needs = (inv_beta * sycl::sqrt(nrm[cdst])) <= tol_zero;
                            }
                            if (!needs) continue;

                            bool filled = false;
                            while (jcur < RR && !filled) {
                                const int32_t j = jcur++;
                                Rows v;
                                v2::set_unit_owned(ex, j, v);
                                // TWO CGS passes: one loses orthogonality exactly
                                // in the near-deficient case that triggers this.
                                for (int32_t pass = 0; pass < 2; ++pass) {
                                    for (int32_t d2 = 0; d2 < dst; ++d2) {
                                        Rows qv;
                                        v2::load_owned_masked(ex, A.col(static_cast<int32_t>(inv[d2])), RR, qv);
                                        const T dot = ex.team_reduce(v2::dotc_owned(qv, v));
                                        v2::sub_mul(v, qv, dot);
                                    }
                                }
                                const Real nrm2 = ex.team_reduce(v2::sumsq_owned(ex, v, RR));
                                if (nrm2 > accept_tol) {
                                    const Real inv_nr = Real(1) / sycl::sqrt(nrm2);
                                    v2::for_each_owned_static<kC, true>(ex, [&](int rr, int row) {
                                        A(row, cdst) = v[rr] * T(inv_nr);
                                    });
                                    filled = true;
                                }
                            }
                            v2::sub_barrier(ex);
                        }
                    }
                    v2::sub_barrier(ex);

                    // Writeback, each orientation coalescing its own output. Hand loops through
                    // the v1 views: v2::copy/map over v2::mat(..).row() sub-views changed the
                    // unrolling (STG +17..+36).
                    if (!transposed_f) {
                        // U(lane, dst) = A(lane, inv[dst]): dst is the output COLUMN, Thin truncates it.
                        for (int32_t rr = 0; rr < static_cast<int32_t>(kRPL); ++rr) {
                            const int32_t row = lane + rr * kP;
                            if (!live || row >= RR) continue;
                            for (int32_t dst = 0; dst < LC; ++dst) U_prob(row, dst) = A(row, static_cast<int32_t>(inv[dst]));
                        }
                    } else {
                        // Vh(lane, r) = conj(L(r, c_lane)): lane is the OUTPUT ROW,
                        // so Thin bounds `lane`, never `r` (wrong shape, plausible
                        // numbers).
                        for (int32_t cc = 0; cc < static_cast<int32_t>(kRPL); ++cc) {
                            const int32_t dst = lane + cc * kP;
                            if (!live || dst >= LC) continue;
                            const int32_t c = static_cast<int32_t>(inv[dst]);
                            for (int32_t r = 0; r < RR; ++r) Vh_prob(dst, r) = v2::detail::conj_ref<T>(A(r, c));
                        }
                    }
                }

                // ---- Right factor: the accumulated rotation matrix ----
                if constexpr (ComputeV) {
                    if (!transposed_f) {
                        // Vh(lane, r) = conj(V(r, c_lane))
                        for (int32_t cc = 0; cc < static_cast<int32_t>(kRPL); ++cc) {
                            const int32_t dst = lane + cc * kP;
                            if (!live || dst >= CC) continue;
                            const int32_t c = static_cast<int32_t>(inv[dst]);
                            for (int32_t r = 0; r < CC; ++r) Vh_prob(dst, r) = v2::detail::conj_ref<T>(V(r, c));
                        }
                    } else {
                        // U = V' directly. Here lane is the output ROW.
                        for (int32_t rr = 0; rr < static_cast<int32_t>(kRPL); ++rr) {
                            const int32_t row = lane + rr * kP;
                            if (!live || row >= CC) continue;
                            for (int32_t dst = 0; dst < CC; ++dst) U_prob(row, dst) = V(row, static_cast<int32_t>(inv[dst]));
                        }
                    }
                }
            });
    });
}

// Largest max(m, n) this kernel accepts on ctx's device: the device's local memory budget
// sets it, and it is job-dependent (values-only drops the V tile). gesvd_native.hh holds the
// formula so gesvd's can_run states the same ceiling.
// evidence: docs/design/gesvd.md#gesvdj_cta-local-memory-budget-formula
template <typename T>
int32_t gesvdj_cta_max_dim(Queue& ctx, bool want_vectors) {
    const std::size_t budget = resident::device_slm_budget(impl::local_mem_bytes(ctx->get_device()));
    return static_cast<int32_t>(sycl_gesvd::gesvd_jacobi_max_dim<T>(want_vectors, budget));
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
        throw batchlas::invalid_argument(std::string(where) + ": invalid matrix dimensions or batch size");
    }
    const int64_t m = a.rows();
    const int64_t n = a.cols();
    const int64_t k = std::min(m, n);
    const int64_t batch = a.batch_size();
    if (singular_values.size() < static_cast<std::size_t>(k) * static_cast<std::size_t>(batch)) {
        throw batchlas::invalid_argument(std::string(where) + ": singular_values span too small");
    }
    // Guard on `!= None`, not `== All`, or Thin skips validation.
    jobu = canonical_jobu(jobu, m, k);
    jobvh = canonical_jobvh(jobvh, n, k);
    if (jobu != SvdVectors::None) {
        const int64_t want_cols = svd_u_cols(jobu, m, k);
        if (u.rows() != m || u.cols() != want_cols || u.batch_size() != batch) {
            throw batchlas::invalid_argument(std::string(where) + ": U must be (" +
                                        std::to_string(m) + " x " + std::to_string(want_cols) +
                                        ") with matching batch size");
        }
    }
    if (jobvh != SvdVectors::None) {
        const int64_t want_rows = svd_vh_rows(jobvh, n, k);
        if (vh.rows() != want_rows || vh.cols() != n || vh.batch_size() != batch) {
            throw batchlas::invalid_argument(std::string(where) + ": Vh must be (" +
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
                 GesvdjParams<T> params,
                 Span<int32_t> info) {
    // The packed kernel folds the mask into `live` (I2): an inactive item's outputs are untouched.
    const std::uint8_t* const mask =
        ::batchlas::detail::mask_ptr(::batchlas::detail::call_mask("gesvdj_cta", a_in, u_out, vh_out));
    (void)ws;

    validate_gesvdj_dims(a_in, singular_values, u_out, vh_out, jobu, jobvh, "gesvdj_cta");

    const int32_t m = static_cast<int32_t>(a_in.rows());
    const int32_t n = static_cast<int32_t>(a_in.cols());
    {
        const int64_t k = std::min<int64_t>(m, n);
        jobu = canonical_jobu(jobu, m, k);
        jobvh = canonical_jobvh(jobvh, n, k);
    }
    if (std::max(m, n) > gesvdj_cta_max_dim<T>(ctx, want_vectors_for_cap(jobu, jobvh))) {
        throw batchlas::invalid_argument(
            "gesvdj_cta: max(m, n) exceeds the supported cap for this scalar type "
            "(see gesvdj_cta_max_dim)");
    }

    if (!ctx.device().supports_sub_group_size(32)) {
        throw batchlas::unsupported("gesvdj_cta: device does not support subgroup size 32 required for CTA kernels.");
    }

    const bool transposed = (m < n);
    const int32_t RR = transposed ? n : m;
    const int32_t CC = transposed ? m : n;

    const bool want_u = (jobu != SvdVectors::None);
    const bool want_vh = (jobvh != SvdVectors::None);
    // The R-sized factor lands in U when m >= n and in Vh when m < n, because
    // A^H = U' S V'^H gives A = V' S U'^H.
    const bool want_left = transposed ? want_vh : want_u;
    const bool want_right = transposed ? want_u : want_vh;

    // Left-factor columns to emit: the solve yields CC for free, the rest come
    // from in-kernel completion. Only the left factor can be thin.
    const SvdVectors job_left = transposed ? jobvh : jobu;
    const int32_t left_cols = (job_left == SvdVectors::All) ? RR : CC;

    auto* s_ptr = singular_values.data();

    // No clear: the kernel STORES every item's status (caller USM, no workspace).
    int32_t* info_ptr = detail::info_ptr(info, a_in.batch_size());

    // (P, C): P lanes per partition, C the tile capacity.
    auto launch = [&](auto P_tag, auto C_tag) {
        constexpr size_t Pv = decltype(P_tag)::value;
        constexpr size_t Cv = decltype(C_tag)::value;
        if (want_right) {
            gesvdj_cta_impl<T, Pv, Cv, true>(ctx, a_in, s_ptr, u_out, vh_out, want_left, transposed, RR, CC, left_cols, params, info_ptr, mask);
        } else {
            gesvdj_cta_impl<T, Pv, Cv, false>(ctx, a_in, s_ptr, u_out, vh_out, want_left, transposed, RR, CC, left_cols, params, info_ptr, mask);
        }
    };

    constexpr auto k4 = std::integral_constant<size_t, 4>{};
    constexpr auto k8 = std::integral_constant<size_t, 8>{};
    constexpr auto k16 = std::integral_constant<size_t, 16>{};
    constexpr auto k32 = std::integral_constant<size_t, 32>{};
    constexpr auto k64 = std::integral_constant<size_t, 64>{};

    // P caps at 32 (sub-group width); above that C grows, kRPL = C/P rows/lane.
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
    (void)params;
    validate_gesvdj_dims(a, singular_values, u_out, vh_out, jobu, jobvh, "gesvdj_cta_buffer_size");
    if (std::max(a.rows(), a.cols()) > gesvdj_cta_max_dim<T>(ctx, want_vectors_for_cap(jobu, jobvh))) {
        throw batchlas::invalid_argument(
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
        GesvdjParams<BATCHLAS_UNPAREN fp>, \
        Span<int32_t>); \
    template size_t gesvdj_cta_buffer_size<back, BATCHLAS_UNPAREN fp>( \
        Queue&, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        Span<typename base_type<BATCHLAS_UNPAREN fp>::type>, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        SvdVectors, SvdVectors, \
        GesvdjParams<BATCHLAS_UNPAREN fp>);

    BATCHLAS_INSTANTIATE_SCALAR_ALL_BACKENDS(GESVDJ_CTA_INSTANTIATE)

#undef GESVDJ_CTA_INSTANTIATE

} // namespace batchlas
