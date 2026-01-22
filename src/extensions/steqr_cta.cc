#include <blas/matrix.hh>
#include <blas/functions.hh>
#include <blas/extensions.hh>
#include <blas/extra.hh>
#include <util/kernel-heuristics.hh>
#include <util/group-invoke.hh>
#include <util/mempool.hh>
#include <batchlas/backend_config.h>
#include "../math-helpers.hh"
#include "../queue.hh"
#include <internal/sort.hh>
#include <array>
#include <numeric>

namespace batchlas {

    template <typename T>
    inline T wilkinson_shift(const T& a, const T& b, const T& c) {
        // a,b,c represent the 2x2 block:
        //   [ a  b ]
        //   [ b  c ]
        // Return the eigenvalue closest to c.
        const auto [lambda1, lambda2] = internal::eigenvalues_2x2(a, b, c);
        return sycl::fabs(lambda1 - c) < sycl::fabs(lambda2 - c) ? lambda1 : lambda2;
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

        QSharedCache(LocalAcc q, int32_t bq, int32_t ln, int32_t n_)
            : Q_local(q), base_q(bq), lane(ln), n(n_) {}

        template <typename QProb>
        inline void load(const QProb& Q_prob) {
            if (lane >= n) return;
            const int32_t pN = static_cast<int32_t>(P);
            for (int32_t c = 0; c < n; ++c) {
                Q_local[base_q + lane + c * pN] = Q_prob(lane, c);
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
            if (lane >= n) return;
            const int32_t pN = static_cast<int32_t>(P);
            const int32_t i0 = base_q + lane + col0 * pN;
            const int32_t i1 = base_q + lane + col1 * pN;
            const T q0 = Q_local[i0];
            const T q1 = Q_local[i1];
            Q_local[i0] = c * q0 - s * q1;
            Q_local[i1] = s * q0 + c * q1;
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
        const T d_ip1 = sycl::shift_group_left(partition, d, 1);

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

    template <typename T, size_t P, typename Partition, typename LocalAcc>
    inline void stage_tridiag_to_local(const Partition& partition,
                                       int32_t n,
                                       LocalAcc& D_local,
                                       LocalAcc& E_local,
                                       int32_t base_d,
                                       int32_t base_e,
                                       int32_t lane,
                                       T diag,
                                       T offdiag) {
        // Ensure the whole local tile is initialized.
        if (lane < n) {
            D_local[base_d + lane] = diag;
        } else {
            D_local[base_d + lane] = T(0);
        }
        if (lane < (n - 1)) {
            E_local[base_e + lane] = offdiag;
        } else if (lane < static_cast<int32_t>(P - 1)) {
            E_local[base_e + lane] = T(0);
        }
        sycl::group_barrier(partition);
    }

    template <typename T, size_t P, typename Partition, typename LocalAcc>
    inline std::pair<T, T> reload_tridiag_from_local(const Partition& partition,
                                                     int32_t n,
                                                     const LocalAcc& D_local,
                                                     const LocalAcc& E_local,
                                                     int32_t base_d,
                                                     int32_t base_e,
                                                     int32_t lane) {
        sycl::group_barrier(partition);
        const T diag = (lane < n) ? D_local[base_d + lane] : T(0);
        const T offdiag = (lane < (n - 1)) ? E_local[base_e + lane] : T(0);
        return {diag, offdiag};
    }

    template <typename T, size_t P, typename Partition, typename QCache>
    inline void solve_2x2_and_update(Partition partition,
                                     T& diag,
                                     T& offdiag,
                                     int32_t l0,
                                     bool ql,
                                     QCache& qcache) {
        const T a = sycl::select_from_group(partition, diag, l0);
        const T b = sycl::select_from_group(partition, offdiag, l0);
        const T c2 = sycl::select_from_group(partition, diag, l0 + 1);

        const auto [rt1, rt2, cs, sn] = invoke_one_broadcast(partition, [&]() {
            return internal::laev2(a, b, c2);
        });

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
                                 SteqrShiftStrategy shift_strategy) {
        const int32_t lane = static_cast<int32_t>(partition.get_local_linear_id());

        // Broadcast values needed for the shift (all lanes participate).
        const T p0 = sycl::select_from_group(partition, diag, l);
        const T e0 = sycl::select_from_group(partition, offdiag, l);
        const T dlp1 = sycl::select_from_group(partition, diag, l + 1);
        const T dm = sycl::select_from_group(partition, diag, m);

        // Leader-lane scalar state.
        T g = T(0);
        T c = T(1);
        T s = T(1);
        T p = T(0);

        invoke_one(partition, [&]() {
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
        });

        for (int32_t i = m; i-- > l;) {
            // Broadcast the tridiagonal entries needed for this step.
            const T ei = sycl::select_from_group(partition, offdiag, i);
            const T di = sycl::select_from_group(partition, diag, i);
            const T dip1 = sycl::select_from_group(partition, diag, i + 1);

            // Whether E(i+1) should be updated is a pure function of (i, m, n).
            const bool do_e_upd = (i != (m - 1)) && ((i + 1) < (n - 1));

            const auto [c1b, s1b, d_ip1_new_b, r1_out_b] = invoke_one_broadcast(partition, [&]() {
                // {c1, s1, d_ip1_new, r1_out}
                const T f = s * ei;
                T rout = T(0);

                const auto [c1, s1, r1] = internal::lartg(g, f);

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
            });

            // Apply D/E updates directly to registers.
            if (lane == (i + 1)) {
                diag = d_ip1_new_b;
                if (do_e_upd) {
                    offdiag = r1_out_b;  // this lane owns E(i+1)
                }
            }

            // QL uses reversed-column convention; keep the same sign as before: apply(i+1,i,c,-s).
            qcache.apply(i + 1, i, c1b, -s1b);
        }

        // Final updates: D(l) = D(l) - p, and E(l) = g.
        const auto [d_l_new_b, e_l_new_b] = invoke_one_broadcast(partition, [&]() {
            return std::array{p0 - p, g};
        });
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
                                 SteqrShiftStrategy shift_strategy) {
        const int32_t lane = static_cast<int32_t>(partition.get_local_linear_id());

        // Broadcast values needed for the shift (all lanes participate).
        const T p0 = sycl::select_from_group(partition, diag, l);
        const T e0 = sycl::select_from_group(partition, offdiag, l - 1);
        const T dlm1 = sycl::select_from_group(partition, diag, l - 1);
        const T dm = sycl::select_from_group(partition, diag, m);

        // Leader-lane scalar state.
        T g = T(0);
        T c = T(1);
        T s = T(1);
        T p = T(0);

        invoke_one(partition, [&]() {
            T mu = T(0);
            if (shift_strategy == SteqrShiftStrategy::Wilkinson) {
                mu = wilkinson_shift(dlm1, e0, p0);
            } else {
                const T gg = (dlm1 - p0) / (T(2) * e0);
                const T rr = sycl::hypot(gg, T(1));
                mu = p0 - e0 / (gg + sycl::copysign(rr, gg));
            }
            g = dm - mu;
        });

        for (int32_t i = m; i < l; ++i) {
            // Broadcast the tridiagonal entries needed for this step.
            const T ei = sycl::select_from_group(partition, offdiag, i);
            const T di = sycl::select_from_group(partition, diag, i);
            const T dip1 = sycl::select_from_group(partition, diag, i + 1);

            // Whether E(i-1) should be updated is a pure function of (i, m).
            const bool do_e_upd = (i != m);

            const auto [c1b, s1b, d_i_new_b, r1_out_b] = invoke_one_broadcast(partition, [&]() {
                // {c1, s1, d_i_new, r1_out}
                const T f = s * ei;
                T rout = T(0);

                const auto [c1, s1, r1] = internal::lartg(g, f);

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
            });

            // Apply D/E updates directly to registers.
            if (lane == i) {
                diag = d_i_new_b;
            }
            if (do_e_upd && lane == (i - 1)) {
                offdiag = r1_out_b;  // this lane owns E(i-1)
            }

            // Match previous sign convention: apply(i,i+1,c,-s).
            qcache.apply(i, i + 1, c1b, -s1b);
        }

        // Final updates: D(l) = D(l) - p, and E(l-1) = g.
        const auto [d_l_new_b, e_lm1_new_b] = invoke_one_broadcast(partition, [&]() {
            return std::array{p0 - p, g};
        });
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
            auto D_local = sycl::local_accessor<T, 1>(sycl::range<1>(probs_per_wg * P), cgh);
            auto E_local = sycl::local_accessor<T, 1>(sycl::range<1>(probs_per_wg * (P - 1)), cgh);

            cgh.parallel_for<SteqrCTAKernel<T, P, ComputeVecs>>(
                sycl::nd_range<1>(global_size, wg_size),
                [=](sycl::nd_item<1> it) {
                    const auto wg = it.get_group();
                    const int32_t wg_id = static_cast<int32_t>(wg.get_group_linear_id());

                    const auto sg = it.get_sub_group();
                    const auto partition = sycl::ext::oneapi::experimental::chunked_partition<P>(sg);
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
                    const int32_t base_d = part_id * static_cast<int32_t>(P);
                    const int32_t base_e = part_id * static_cast<int32_t>(P - 1);

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
                        const int32_t block_end = sycl::reduce_over_group(partition, block_end_candidate, sycl::minimum<int32_t>());

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
                        // is not available on some backends (e.g. CUDA). We compute the max
                        // norm via shared local memory and a leader-lane loop instead.
                        stage_tridiag_to_local<T, P>(partition, n, D_local, E_local, base_d, base_e, lane, diag, offdiag);

                        const T scale = invoke_one_broadcast(partition, [&]() {
                            T anorm = sycl::fabs(D_local[base_d + block_end]);
                            for (int32_t idx = block_begin; idx < block_end; ++idx) {
                                anorm = sycl::fmax(anorm, sycl::fabs(D_local[base_d + idx]));
                                anorm = sycl::fmax(anorm, sycl::fabs(E_local[base_e + idx]));
                            }

                            if (anorm > internal::ssfmax<T>()) {
                                // Scale down to avoid overflow.
                                return internal::ssfmax<T>() / anorm;
                            }
                            if (anorm < internal::ssfmin<T>() && anorm != T(0)) {
                                // Scale up to avoid underflow.
                                return internal::ssfmin<T>() / anorm;
                            }
                            return T(1);
                        });
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
                        const T d_first = sycl::fabs(sycl::select_from_group(partition, diag, block_begin));
                        const T d_last = sycl::fabs(sycl::select_from_group(partition, diag, block_end));
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
                                    const int32_t m = sycl::reduce_over_group(partition, m_candidate, sycl::minimum<int32_t>());

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
                                    implicit_ql_step<T, P>(partition, diag, offdiag, qcache, n, l, m, cta_shift_strategy);
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
                                    const int32_t m = sycl::reduce_over_group(partition, m_candidate, sycl::maximum<int32_t>());

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
                                    implicit_qr_step<T, P>(partition, diag, offdiag, qcache, n, m, l_u, cta_shift_strategy);
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

        {
            const auto dev = ctx->get_device();
            const auto sg_sizes = dev.get_info<sycl::info::device::sub_group_sizes>();
            bool has32 = false;
            for (auto sgs : sg_sizes) {
                if (static_cast<int32_t>(sgs) == 32) {
                    has32 = true;
                    break;
                }
            }
            if (!has32) {
                throw std::runtime_error("steqr_cta: device does not support subgroup size 32 required for CTA kernels.");
            }
        }

        const int32_t n_i32 = static_cast<int32_t>(n);

        auto launch = [&](auto P_tag) {
            constexpr int32_t P = decltype(P_tag)::value;
            if (jobz == JobType::EigenVectors) {
                steqr_cta_impl<T, P, true>(ctx, d, e, eigvects_mut, n_i32,
                                           params.max_sweeps, params.zero_threshold,
                                           params.cta_shift_strategy, params.cta_wg_size_multiplier,
                                           status,
                                           pool);
            } else {
                steqr_cta_impl<T, P, false>(ctx, d, e, eigvects_mut, n_i32,
                                            params.max_sweeps, params.zero_threshold,
                                            params.cta_shift_strategy, params.cta_wg_size_multiplier,
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
        return size;
    }

#if BATCHLAS_HAS_CUDA_BACKEND
    template Event steqr_cta<Backend::CUDA, float>(Queue&, const VectorView<float>&, const VectorView<float>&, const VectorView<float>&, const Span<std::byte>&, JobType, SteqrParams<float>, const MatrixView<float, MatrixFormat::Dense>&);
    template Event steqr_cta<Backend::CUDA, double>(Queue&, const VectorView<double>&, const VectorView<double>&, const VectorView<double>&, const Span<std::byte>&, JobType, SteqrParams<double>, const MatrixView<double, MatrixFormat::Dense>&);
#endif

#if BATCHLAS_HAS_HOST_BACKEND
    template Event steqr_cta<Backend::NETLIB, float>(Queue&, const VectorView<float>&, const VectorView<float>&, const VectorView<float>&, const Span<std::byte>&, JobType, SteqrParams<float>, const MatrixView<float, MatrixFormat::Dense>&);
    template Event steqr_cta<Backend::NETLIB, double>(Queue&, const VectorView<double>&, const VectorView<double>&, const VectorView<double>&, const Span<std::byte>&, JobType, SteqrParams<double>, const MatrixView<double, MatrixFormat::Dense>&);
#endif

    template size_t steqr_cta_buffer_size<float>(Queue&, const VectorView<float>&, const VectorView<float>&, const VectorView<float>&, JobType, SteqrParams<float>);
    template size_t steqr_cta_buffer_size<double>(Queue&, const VectorView<double>&, const VectorView<double>&, const VectorView<double>&, JobType, SteqrParams<double>);
}