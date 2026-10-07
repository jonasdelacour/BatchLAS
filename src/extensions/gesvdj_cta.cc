#include <batchlas/blas/matrix.hh>
#include <batchlas/blas/functions.hh>
#include <batchlas/blas/extensions.hh>
#include <batchlas/blas/extra.hh>
#include <batchlas/util/kernel-heuristics.hh>
#include <batchlas/util/mempool.hh>
#include <batchlas/util/group-invoke.hh>
#include "gesvd_native.hh"
#include "sg_compat.hh"
#include <batchlas/backend_config.h>
#include "../math-helpers.hh"
#include "../queue.hh"
#include "../util/template-instantiations.hh"
#include "info_span.hh"
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

// Round-robin pairing, identical to syev_jacobi_cta's: for even mp the mp-1
// rounds of mp/2 disjoint pairs cover every index pair exactly once.
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
                            GesvdjParams<T> params,
                            int32_t* info) {
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
        constexpr bool kNeedPhase = internal::is_complex<T>::value;

        // Clamp probs_per_wg DIRECTLY, not the multiplier as syev_jacobi_cta
        // does: that under-counts by 32/P, a launch failure with two tiles.
        // evidence: docs/design/gesvd.md#gesvdj_cta-local-memory-budget-formula
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
        const Real tol = params.tol_multiplier * static_cast<Real>(CC) * std::numeric_limits<Real>::epsilon();
        const Real tiny = std::numeric_limits<Real>::min();
        const Real tau_big = Real(1) / sycl::sqrt(std::numeric_limits<Real>::epsilon());
        const Real zero_mult = params.zero_sigma_multiplier;

        Real* S = s_ptr;
        int32_t* SW = (params.sweep_counts.size() >= static_cast<size_t>(batch_size))
                          ? params.sweep_counts.data()
                          : nullptr;
        // nullptr when status was not requested, making info_store a no-op.
        int32_t* const info_dev = info;

        // reqd_sub_group_size(32) is load-bearing: the 5- and (4+1)-step
        // butterflies, probs_per_warp and part_id are silently wrong otherwise.
        cgh.parallel_for<GesvdjCTAKernel<T, P, C, ComputeV>>(
            sycl::nd_range<1>(global_size, wg_size),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                const auto wg = it.get_group();
                const int32_t wg_id = static_cast<int32_t>(wg.get_group_linear_id());
                const int32_t local_id = static_cast<int32_t>(it.get_local_linear_id());

                const int32_t pairs_per_round = mp / 2;
                const int32_t rounds = mp - 1;

                // Pair table: stride kRotSlots (not pairs_per_round, they differ
                // when CC < P); unused slots hold a sentinel failing `< CC`. Must
                // precede the `prob_id >= nb` return: it ends in a wg barrier.
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

                // ---- Load. lane = ROW (rows lane, lane+P, ...), coalesced. ----
                // m < n is transposed at load time. The pad is exact zero so a
                // padded pair's Gram is 0 and falls below any threshold.
                for (int32_t rr = 0; rr < static_cast<int32_t>(kRPL); ++rr) {
                    const int32_t row = lane + rr * static_cast<int32_t>(P);
                    for (int32_t c = 0; c < static_cast<int32_t>(C); ++c) {
                        T v = T(0);
                        if (row < RR && c < CC) {
                            // Solve A^H, not A^T: A^T is wrong for complex only.
                            // evidence: docs/design/gesvd.md#gesvdj_cta-rank-deficiency-thin-and-m--n
                            v = transposed_f ? conj_if_complex_g(A_prob(c, row)) : A_prob(row, c);
                        }
                        A_local[base_a + row + c * LD] = v;
                        if constexpr (ComputeV) {
                            V_local[base_v + row + c * LD] = (row == c && row < CC) ? T(1) : T(0);
                        }
                    }
                }
                group_barrier(part);

                // ---- Exact column norms: lane c ends holding ||A_c||^2. ----
                // P values over P lanes: 5 scatter steps, no all-reduce. x stays
                // Real[P] (not Real[C]); C columns take C/P passes.
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

                // ---- Global rescale: ONE power-of-two factor, centred on the
                // geometric mean. Per-column equilibration is not a factorisation
                // of A. Only columns < P feed nmax/nmin (the C=64 rung's upper
                // half does not); beta stays exact either way.
                // evidence: docs/design/gesvd.md#gesvdj_cta-global-power-of-two-scaling
                Real my_n2 = (lane < CC) ? Nrm_local[base_n + lane] : Real(0);
                const Real nmax = part_max_g(part, my_n2);
                const Real nmin_in = (lane < CC && my_n2 > Real(0))
                                         ? my_n2
                                         : std::numeric_limits<Real>::max();
                const Real nmin = part_min_g(part, nmin_in);

                if (nmax == Real(0)) {
                    // Zero input: beta = 1; completion fills U, V is identity.
                }
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
                        const int32_t row = lane + rr * static_cast<int32_t>(P);
                        for (int32_t c = 0; c < static_cast<int32_t>(C); ++c) {
                            A_local[base_a + row + c * LD] = A_local[base_a + row + c * LD] * T(beta);
                        }
                    }
                    group_barrier(part);
                    exact_norms();
                    group_barrier(part);
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
                        group_barrier(part);
                    }

                    int32_t rot_count = 0;

                    for (int32_t t = 0; t < rounds; ++t) {
                        const int32_t tab_base = t * static_cast<int32_t>(kRotSlots);

                        // Chunks of kGramChunk pairs (two at C=64). Safe because
                        // a round is a perfect matching: chunks touch disjoint
                        // columns. Also caps live registers (80 T, not 160).
                        for (int32_t ch = 0; ch < static_cast<int32_t>(kChunks); ++ch) {
                        const int32_t tab_base = t * static_cast<int32_t>(kRotSlots)
                                               + ch * static_cast<int32_t>(kGramChunk);

                        // Every index into pk/qk/ap/aq/g must be compile-time or
                        // the arrays spill.
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
                            // conj(A_p)*A_q over this lane's rows, then across lanes.
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

                        // Reduce-scatter is 4 scatter + 1 all-reduce step at P=32.
                        // Five halving steps silently sum over half the rows.
                        // evidence: docs/design/gesvd.md#gesvdj_cta-the-reduce-scatter-g3-trap
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
                        // Final ALL-REDUCE: g[0] = a_{p_k q_k}, k = lane>>1.
                        g[0] = g[0] + xor_shuffle_g(part, g[0], 1u);

                        const int32_t k_of_lane = lane >> 1;
                        const int32_t slot = ch * static_cast<int32_t>(kGramChunk) + k_of_lane;
                        // Re-read from LDS: a runtime index into pk[] would spill it.
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
                                // An identity rotation must not count, or every
                                // problem burns all max_sweeps.
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

                        // Analytic norm recurrence (as ?GESVJ's SVA), even lanes
                        // only. BOTH sides clamp at zero: a negative q side NaNs
                        // the rank sort into an out-of-bounds LDS index.
                        if (active && (lane % 2 == 0)) {
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

                        // Every lane must execute this butterfly.
                        const int32_t round_active =
                            part_sum_g(part, (active && (lane % 2 == 0)) ? int32_t(1) : int32_t(0));
                        rot_count += round_active;
                        if (round_active == 0) continue;

                        // ---- A <- A*U, V <- V*U, reusing ap/aq from the Gram.
                        // Deliberately no A <- U^H A phase: that is two-sided.
#pragma unroll
                        for (int32_t k = 0; k < static_cast<int32_t>(kGramChunk); ++k) {
                            const sycl::vec<Real, 2> cs =
                                Rcs_local[base_r + ch * static_cast<int32_t>(kGramChunk) + k];
                            const Real ck = cs[0];
                            const Real sk = cs[1];
                            // Warp-uniform skip: converged pairs are free.
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

                // ---- Epilogue. sigma comes from A, ALWAYS: the incremental
                // Nrm_local only chooses rotations; reading sigma from it passes
                // every test and reintroduces the normal-equations defect.
                exact_norms();
                group_barrier(part);


                if (SW != nullptr && lane == 0) {
                    SW[prob_id] = sweeps_used;
                }
                // Converged means `zero_sweeps >= 2`, not `sweeps_used <
                // max_sweeps`. zero_sweeps is partition-uniform, so lane 0 reports.
                // A STORE, not a raise: this kernel is the item's single writer.
                if (lane == 0) {
                    detail::info_store(info_dev, prob_id, (zero_sweeps < 2) ? 1 : 0);
                }

                // Rank sort, DESCENDING (the gesvd contract), ties on index. The
                // identity seed makes a rank collision a wrong permutation, not
                // a garbage LDS index. Here lane is a COLUMN index.
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
                    // Left columns CC..RR-1 park on the zero pad columns,
                    // disjoint from the rank slots 0..CC-1.
                    if (col >= CC && col < RR) {
                        Inv_local[base_p + col] = static_cast<int16_t>(col);
                    }
                }
                group_barrier(part);

                const Real sigma_max = (CC > 0) ? (inv_beta * sycl::sqrt(Nrm_local[base_n + static_cast<int32_t>(Inv_local[base_p])])) : Real(0);
                // Relative to sigma_max only; eps*fmax(1, sigma_max) would zero
                // every sigma of a uniformly small input.
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
                    // Normalise in place; A is fully consumed into sigma.
                    for (int32_t cc = 0; cc < static_cast<int32_t>(kRPL); ++cc) {
                        const int32_t col = lane + cc * static_cast<int32_t>(P);
                        if (col >= CC) continue;
                        const Real n2_c = Nrm_local[base_n + col];
                        const Real s_c = inv_beta * sycl::sqrt(n2_c);
                        if (s_c > tol_zero && n2_c > Real(0)) {
                            // Divide by the SCALED norm sqrt(n2_c), not sigma.
                            const Real inv_s = Real(1) / sycl::sqrt(n2_c);
                            for (int32_t r = 0; r < static_cast<int32_t>(C); ++r) {
                                A_local[base_a + r + col * LD] = A_local[base_a + r + col * LD] * T(inv_s);
                            }
                        }
                    }
                    group_barrier(part);

                    // Completion of columns CC..RR-1 and any sigma <= tol_zero,
                    // behind a warp-uniform gate.
                    int32_t deficient_here = 0;
                    for (int32_t cc = 0; cc < static_cast<int32_t>(kRPL); ++cc) {
                        const int32_t col = lane + cc * static_cast<int32_t>(P);
                        if (col < CC && inv_beta * sycl::sqrt(Nrm_local[base_n + col]) <= tol_zero) {
                            ++deficient_here;
                        }
                    }
                    const int32_t any_def = part_sum_g(part, deficient_here);

                    // LC, not RR: Thin skips completion unless a column is deficient.
                    if (any_def > 0 || LC > CC) {
                        // The trial cursor RUNS ACROSS dst (never reset), with
                        // acceptance above 1/(2*RR): that is what terminates.
                        // evidence: docs/design/gesvd.md#gesvdj_cta-rank-deficiency-thin-and-m--n
                        const Real accept_tol = Real(1) / (Real(2) * static_cast<Real>(RR));
                        int32_t jcur = 0;

                        for (int32_t dst = 0; dst < LC; ++dst) {
                            const int32_t cdst = static_cast<int32_t>(Inv_local[base_p + dst]);
                            bool needs = (dst >= CC);
                            if (!needs) {
                                needs = (inv_beta * sycl::sqrt(Nrm_local[base_n + cdst])) <= tol_zero;
                            }
                            if (!needs) continue;

                            bool filled = false;
                            while (jcur < RR && !filled) {
                                const int32_t j = jcur++;
                                T v[kRPL];
#pragma unroll
                                for (int32_t rr = 0; rr < static_cast<int32_t>(kRPL); ++rr) {
                                    const int32_t row = lane + rr * static_cast<int32_t>(P);
                                    v[rr] = (row == j) ? T(1) : T(0);
                                }
                                // TWO CGS passes: one loses orthogonality exactly
                                // in the near-deficient case that triggers this.
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

                    // Writeback, each orientation coalescing its own output.
                    if (!transposed_f) {
                        // U(lane, dst): dst is the output COLUMN, Thin truncates it.
                        for (int32_t rr = 0; rr < static_cast<int32_t>(kRPL); ++rr) {
                            const int32_t row = lane + rr * static_cast<int32_t>(P);
                            if (row >= RR) continue;
                            for (int32_t dst = 0; dst < LC; ++dst) {
                                const int32_t c = static_cast<int32_t>(Inv_local[base_p + dst]);
                                U_prob(row, dst) = A_local[base_a + row + c * LD];
                            }
                        }
                    } else {
                        // Vh(lane, r) = conj(L(r, c_lane)): lane is the OUTPUT ROW,
                        // so Thin bounds `lane`, never `r` (wrong shape, plausible
                        // numbers).
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

// Largest max(m, n) this kernel accepts, per scalar type. Local memory sets it,
// and occupancy rather than the hard cap binds: complex<double> with a V tile
// does not launch at C=64, and values-only (no V tile) halves the budget, so the
// cap is job-dependent. The values live in gesvd_native.hh so gesvd's can_run
// states the same ceiling.
// evidence: docs/design/gesvd.md#gesvdj_cta-local-memory-budget-formula
template <typename T>
constexpr int32_t gesvdj_cta_max_dim(bool want_vectors) {
    return static_cast<int32_t>(sycl_gesvd::gesvd_jacobi_max_dim<T>(want_vectors));
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
        throw batchlas::invalid_argument(
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
            throw batchlas::unsupported("gesvdj_cta: device does not support subgroup size 32 required for CTA kernels.");
        }
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
            gesvdj_cta_impl<T, Pv, Cv, true>(ctx, a_in, s_ptr, u_out, vh_out, want_left, transposed, RR, CC, left_cols, params, info_ptr);
        } else {
            gesvdj_cta_impl<T, Pv, Cv, false>(ctx, a_in, s_ptr, u_out, vh_out, want_left, transposed, RR, CC, left_cols, params, info_ptr);
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
    (void)ctx;
    (void)params;
    validate_gesvdj_dims(a, singular_values, u_out, vh_out, jobu, jobvh, "gesvdj_cta_buffer_size");
    if (std::max(a.rows(), a.cols()) > gesvdj_cta_max_dim<T>(want_vectors_for_cap(jobu, jobvh))) {
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
