#include <blas/matrix.hh>
#include <blas/functions.hh>
#include <blas/extensions.hh>
#include <blas/extra.hh>
#include <util/kernel-heuristics.hh>
#include <util/mempool.hh>
#include <batchlas/backend_config.h>
#include "../math-helpers.hh"
#include "../queue.hh"
#include <internal/sort.hh>
#include <numeric>
using namespace sycl::ext::oneapi;

namespace batchlas {

    template <size_t N>
    inline bool steqr_cta_debug_enabled(size_t prob_id, size_t lane) {
        // Keep device-side printf noise low: only print for the smallest debug case.
        return (N == 8) && (prob_id == 0) && (lane == 0);
    }

    template <typename T, size_t N>
    class SteqrCTAKernel;

    template <typename T, size_t N, typename Partition, typename LocalAcc>
    inline void apply_col_rotation(const Partition& partition,
                                   LocalAcc& Q_local,
                                   int32_t base,
                                   int32_t col0,
                                   int32_t col1,
                                   T c,
                                   T s) {
        // Right-multiply Q by the Givens rotation acting on columns (col0,col1).
        const int32_t row = static_cast<int32_t>(partition.get_local_linear_id());
        const int32_t i0 = base + row + col0 * N;
        const int32_t i1 = base + row + col1 * N;
        const T q0 = Q_local[i0];
        const T q1 = Q_local[i1];
        // Apply Q <- Q * G where G = [[c, s], [-s, c]].
        Q_local[i0] = c * q0 - s * q1;
        Q_local[i1] = s * q0 + c * q1;
    }

    template <typename T, typename Partition>
    inline void deflate(Partition partition,
                        T& e,
                        T& d,
                        int32_t start_ix,
                        int32_t end_ix,
                        T zero_threshold) {
        (void)zero_threshold;
        const int32_t lane = static_cast<int32_t>(partition.get_local_linear_id());
        const int32_t partition_size = static_cast<int32_t>(partition.get_local_range().size());
        const bool lane_in_active_range = (lane + 1 < partition_size) && (lane >= start_ix) && (lane + 1 < end_ix);

        // We need d_{i+1} (neighbor lane's diagonal). A 1-lane shift is the most direct.
        // Note: for lanes without i+1 (last lane), the result is unspecified, but those
        // lanes never use d_ip1 due to lane_in_active_range.
        const T d_ip1 = sycl::shift_group_left(partition, d, 1);

        if (lane_in_active_range) {
            // Absolute cutoff (user-controlled) first.
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

    template <typename T, size_t N, typename Partition, typename LocalAcc>
    inline void stage_tridiag_to_local(const Partition& partition,
                                       LocalAcc& D_local,
                                       LocalAcc& E_local,
                                       int32_t base_d,
                                       int32_t base_e,
                                       int32_t lane,
                                       T diag,
                                       T offdiag) {
        D_local[base_d + lane] = diag;
        if (lane < (N - 1)) {
            E_local[base_e + lane] = offdiag;
        }
        sycl::group_barrier(partition);
    }

    template <typename T, size_t N, typename Partition, typename LocalAcc>
    inline std::pair<T, T> reload_tridiag_from_local(const Partition& partition,
                                                     const LocalAcc& D_local,
                                                     const LocalAcc& E_local,
                                                     int32_t base_d,
                                                     int32_t base_e,
                                                     int32_t lane) {
        sycl::group_barrier(partition);
        const T diag = D_local[base_d + lane];
        const T offdiag = (lane < (N - 1)) ? E_local[base_e + lane] : T(0);
        return {diag, offdiag};
    }

    template <typename T, size_t N, typename Partition, typename LocalAcc>
    inline void solve_2x2_and_update(Partition partition,
                                    T& diag,
                                    T& offdiag,
                                    int32_t l0,
                                    bool ql,
                                    LocalAcc& Q_local,
                                    int32_t base) {
        const T a = sycl::select_from_group(partition, diag, l0);
        const T b = sycl::select_from_group(partition, offdiag, l0);
        const T c2 = sycl::select_from_group(partition, diag, l0 + 1);
        auto [rt1, rt2, cs, sn] = internal::laev2(a, b, c2);

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
        apply_col_rotation<T, N>(partition, Q_local, base, col0, col1, cs, s_eff);
    }

    template <typename T, size_t N, typename Partition, typename LocalAccD, typename LocalAccE, typename LocalAccQ>
    inline void implicit_ql_step(const Partition& partition,
                                 LocalAccD& D_local,
                                 LocalAccE& E_local,
                                 LocalAccQ& Q_local,
                                 int32_t base_d,
                                 int32_t base_e,
                                 int32_t base_q,
                                 int32_t l,
                                 int32_t m) {
        const int32_t lane = static_cast<int32_t>(partition.get_local_linear_id());

        // Leader-lane scalar state.
        T g = T(0);
        T c = T(1);
        T s = T(1);
        T p = T(0);

        if (lane == 0) {
            const T p0 = D_local[base_d + l];
            const T e0 = E_local[base_e + l];
            const T dlp1 = D_local[base_d + (l + 1)];

            T gg = (dlp1 - p0) / (T(2) * e0);
            const T rr = sycl::hypot(gg, T(1));
            g = D_local[base_d + m] - p0 + e0 / (gg + sycl::copysign(rr, gg));
        }

        for (int32_t i = m; i-- > l;) {
            T c1 = T(0);
            T s1 = T(0);

            if (lane == 0) {
                const T ei = E_local[base_e + i];
                const T di = D_local[base_d + i];
                const T dip1 = D_local[base_d + (i + 1)];

                const T f = s * ei;
                const T b = c * ei;

                T r1;
                std::tie(c1, s1, r1) = internal::lartg(g, f);

                if (i != (m - 1) && (i + 1) < (N - 1)) {
                    E_local[base_e + (i + 1)] = r1;
                }

                const T g2 = dip1 - p;
                const T r2 = (di - g2) * s1 + T(2) * c1 * b;
                p = s1 * r2;

                D_local[base_d + (i + 1)] = g2 + p;
                g = c1 * r2 - b;
                c = c1;
                s = s1;
            }

            const T c1b = sycl::select_from_group(partition, c1, 0);
            const T s1b = sycl::select_from_group(partition, s1, 0);
            // QL uses the reversed-column convention; this matches the previous inline code.
            apply_col_rotation<T, N>(partition, Q_local, base_q, i + 1, i, c1b, -s1b);
        }

        if (lane == 0) {
            D_local[base_d + l] = D_local[base_d + l] - p;
            if (l < (N - 1)) {
                E_local[base_e + l] = g;
            }
        }
    }

    template <typename T, size_t N, typename Partition, typename LocalAccD, typename LocalAccE, typename LocalAccQ>
    inline void implicit_qr_step(const Partition& partition,
                                 LocalAccD& D_local,
                                 LocalAccE& E_local,
                                 LocalAccQ& Q_local,
                                 int32_t base_d,
                                 int32_t base_e,
                                 int32_t base_q,
                                 int32_t m,
                                 int32_t l) {
        const int32_t lane = static_cast<int32_t>(partition.get_local_linear_id());

        // Leader-lane scalar state.
        T g = T(0);
        T c = T(1);
        T s = T(1);
        T p = T(0);

        if (lane == 0) {
            const T p0 = D_local[base_d + l];
            const T e0 = E_local[base_e + (l - 1)];
            const T dlm1 = D_local[base_d + (l - 1)];

            T gg = (dlm1 - p0) / (T(2) * e0);
            const T rr = sycl::hypot(gg, T(1));
            g = D_local[base_d + m] - p0 + e0 / (gg + sycl::copysign(rr, gg));
        }

        for (int32_t i = m; i < l; ++i) {
            T c1 = T(0);
            T s1 = T(0);

            if (lane == 0) {
                const T ei = E_local[base_e + i];
                const T di = D_local[base_d + i];
                const T dip1 = D_local[base_d + (i + 1)];

                const T f = s * ei;
                const T b = c * ei;

                T r1;
                std::tie(c1, s1, r1) = internal::lartg(g, f);
                if (i != m) {
                    E_local[base_e + (i - 1)] = r1;
                }

                const T g2 = di - p;
                const T r2 = (dip1 - g2) * s1 + T(2) * c1 * b;
                p = s1 * r2;

                D_local[base_d + i] = g2 + p;
                g = c1 * r2 - b;
                c = c1;
                s = s1;
            }

            const T c1b = sycl::select_from_group(partition, c1, 0);
            const T s1b = sycl::select_from_group(partition, s1, 0);
            apply_col_rotation<T, N>(partition, Q_local, base_q, i, i + 1, c1b, -s1b);
        }

        if (lane == 0) {
            D_local[base_d + l] = D_local[base_d + l] - p;
            E_local[base_e + (l - 1)] = g;
        }
    }

    template <typename T, size_t N>
    inline void steqr_cta_impl(Queue& ctx,
                               VectorView<T>& d,
                               VectorView<T>& e,
                               MatrixView<T, MatrixFormat::Dense>& eigvects,
                               size_t max_sweeps,
                               T zero_threshold,
                               BumpAllocator& pool) {
        auto batch_size = d.batch_size();
        if (d.size() > N || e.size() > N - 1) {
            throw std::runtime_error("steqr_cta_impl: Vector sizes exceed template parameter N.");
        }
        // Implementation of the CTA-based STEQR algorithm goes here.
        ctx -> submit([&](sycl::handler& cgh) {
            auto Q_view = eigvects.kernel_view();
            const auto dev = ctx->get_device();
            const auto sg_sizes = dev.get_info<sycl::info::device::sub_group_sizes>();

            // Heuristic: prefer 16 for very small N when supported, otherwise 32 when supported.
            int32_t sg_size = sg_sizes.empty() ? int32_t(1) : static_cast<int32_t>(sg_sizes[0]);
            for (auto sgs : sg_sizes) {
                if (static_cast<int32_t>(sgs) == 32) sg_size = 32;
            }
            if constexpr (N <= 16) {
                for (auto sgs : sg_sizes) {
                    if (static_cast<int32_t>(sgs) == 16) sg_size = 16;
                }
            }

            // Minimal, safe mapping: work-group size is LCM(N, sg_size), so we can form fixed-size partitions of size N.
            const int32_t wg_size = std::lcm<int32_t>(static_cast<int32_t>(N), sg_size);
            const int32_t probs_per_wg = wg_size / static_cast<int32_t>(N);

            const int32_t num_wg = (batch_size + probs_per_wg - 1) / probs_per_wg;
            const int32_t global_size = num_wg * wg_size;

            auto Q_local = sycl::local_accessor<T, 1>(sycl::range<1>(probs_per_wg * N * N), cgh);
            auto D_local = sycl::local_accessor<T, 1>(sycl::range<1>(probs_per_wg * N), cgh);
            auto E_local = sycl::local_accessor<T, 1>(sycl::range<1>(probs_per_wg * (N - 1)), cgh);

            cgh.parallel_for<SteqrCTAKernel<T, N>>(
                sycl::nd_range<1>(global_size, wg_size),
                [=](sycl::nd_item<1> it) {
                    const auto wg = it.get_group();
                    const size_t wg_id = static_cast<size_t>(wg.get_group_linear_id());

                    const auto sg = it.get_sub_group();
                    const auto partition = sycl::ext::oneapi::experimental::chunked_partition<N>(sg);
                    // NOTE: chunked_partition<N>(sg) partitions *within a sub-group*.
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
                    auto Q_prob = Q_view.batch_item(prob_id);                    

                    const int32_t base = part_id * N * N;
                    const int32_t base_d = part_id * N;
                    const int32_t base_e = part_id * (N - 1);

                    // Initialize Q_local for this problem as identity (column-major storage).
                    // Each lane handles its row: Q_local[base + row + col*N] = Q(row, col)
                    for (int32_t i = 0; i < N; ++i) {
                        Q_local[base + lane + i * N] = Q_prob(lane, i);
                    }

                    // Load D/E into registers (one element per lane).
                    T diag = d_prob(lane);
                    T offdiag = (lane < N - 1) ? e_prob(lane) : T(0);

                    // ---- Outer split loop over blocks separated by E==0 ----
                    for (int32_t next_block_begin = 0; next_block_begin < N;) {
                        const int32_t block_begin = next_block_begin;

                        // Mark the split explicitly as LAPACK does: E(block_begin-1)=0.
                        if (block_begin > 0 && lane == (block_begin - 1) && lane < (N - 1)) {
                            offdiag = T(0);
                        }

                        // Deflation pass over the remaining tail to create more zeros in E.
                        deflate(partition, offdiag, diag, block_begin, N, zero_threshold);

                        // Find end of current block: first i>=block_begin where E(i)==0; if none, block ends at N-1.
                        const int32_t block_end_candidate =
                            (lane >= block_begin && lane < (N - 1) && offdiag == T(0)) ? lane : (N - 1);
                        const int32_t block_end = sycl::reduce_over_group(partition, block_end_candidate, sycl::minimum<int32_t>());

                        // Next block starts after block_end.
                        next_block_begin = block_end + 1;

                        // Size-0/1 block.
                        if (block_end <= block_begin) {
                            continue;
                        }

                        // Choose between QL and QR (matches steqr.cc):
                        // - QR if |D(l)| <= |D(lend)|
                        // - QL otherwise
                        const T d_first = sycl::fabs(sycl::select_from_group(partition, diag, block_begin));
                        const T d_last = sycl::fabs(sycl::select_from_group(partition, diag, block_end));
                        const bool use_ql = (d_last < d_first);
                        if (use_ql) {
                            // ---------------- QL iteration: converge from the top (l grows) ----------------
                            for (int32_t l = block_begin; l <= block_end;) {
                                if (l == block_end) {
                                    l += 1;
                                    continue;
                                }

                                // Iterate up to max_sweeps times to converge eigenvalue at position l.
                                for (int32_t sweep = 0; sweep < max_sweeps; ++sweep) {
                                    // Deflate within current active subproblem [l..lend].
                                    deflate(partition, offdiag, diag, l, block_end + 1, zero_threshold);

                                    // Find first m in [l..lend-1] such that E(m)==0; if none, m=lend.
                                    const int32_t m_candidate = (lane >= l && lane < block_end && offdiag == T(0)) ? lane : block_end;
                                    const int32_t m = sycl::reduce_over_group(partition, m_candidate, sycl::minimum<int32_t>());

                                    if (m == l) {
                                        // Converged! Move to next eigenvalue.
                                        l += 1;
                                        break;
                                    }

                                    if (m == l + 1) {
                                        // 2x2 block at (l,l+1).
                                        solve_2x2_and_update<T, N>(partition, diag, offdiag, l, /*ql=*/true, Q_local, base);

                                        l += 2;
                                        break;
                                    }

                                // ---- Implicit QL step on subblock [l..m] (m>=l+2) ----
                                // Stage the current D/E into local memory so the bulge chase can be done scalarly by a leader lane.
                                    stage_tridiag_to_local<T, N>(partition, D_local, E_local, base_d, base_e, lane, diag, offdiag);

                                    implicit_ql_step<T, N>(partition, D_local, E_local, Q_local, base_d, base_e, base, l, m);

                                // Reload D/E registers from local memory for subsequent deflation scans.
                                    std::tie(diag, offdiag) = reload_tridiag_from_local<T, N>(partition, D_local, E_local, base_d, base_e, lane);
                                }  // end sweep loop for QL
                            }  // end QL l loop
                        } else {
                            // ---------------- QR iteration: converge from the bottom (l shrinks) ----------------
                            // Use signed indices for the descending loop to avoid unsigned underflow.
                            for (int32_t l = static_cast<int32_t>(block_end);
                                 l >= static_cast<int32_t>(block_begin);
                                 /* manual step */) {
                                if (l == static_cast<int32_t>(block_begin)) {
                                    break;
                                }

                                // Iterate up to max_sweeps times to converge eigenvalue at position l.
                                for (int32_t sweep = 0; sweep < max_sweeps; ++sweep) {
                                    deflate(partition, offdiag, diag, block_begin, static_cast<int32_t>(l) + 1, zero_threshold);

                                    // Find m scanning downward: look for E(i)==0 and take the largest i+1.
                                    const int32_t l_u = static_cast<int32_t>(l);
                                    const int32_t m_candidate =
                                        (lane >= block_begin && lane < l_u && offdiag == T(0)) ? (lane + 1) : block_begin;
                                    const int32_t m = sycl::reduce_over_group(partition, m_candidate, sycl::maximum<int32_t>());

                                    if (m == l) {
                                        // Converged! Move to next eigenvalue.
                                        l -= 1;
                                        break;
                                    }

                                    if (m + 1 == l) {
                                        // 2x2 block at (l-1,l).
                                        const size_t l0 = l_u - 1;
                                        solve_2x2_and_update<T, N>(partition, diag, offdiag, l0, /*ql=*/false, Q_local, base);

                                        if (l <= 1) {
                                            l = static_cast<int32_t>(block_begin);
                                        } else {
                                            l -= 2;
                                        }
                                        break;
                                    }


                                // ---- Implicit QR step on subblock [m..l] ----
                                // Stage the current D/E into local memory so the bulge chase can be done scalarly by a leader lane.
                                    stage_tridiag_to_local<T, N>(partition, D_local, E_local, base_d, base_e, lane, diag, offdiag);

                                    implicit_qr_step<T, N>(partition, D_local, E_local, Q_local, base_d, base_e, base, m, l_u);

                                // Reload D/E registers from local memory for subsequent deflation scans.
                                    std::tie(diag, offdiag) = reload_tridiag_from_local<T, N>(partition, D_local, E_local, base_d, base_e, lane);
                                }  // end sweep loop for QR
                            }  // end QR l loop
                        }  // end if (do_ql) else
                    }  // end outer block split loop

                    // Make sure all lanes are done updating Q_local before copying out.
                    sycl::group_barrier(partition);

                    // Store back D/E (one element per lane).
                    d_prob(lane) = diag;
                    if (lane < (N - 1)) {
                        e_prob(lane) = offdiag;
                    }

                    // Barrier to ensure D/E writes complete before Q writes
                    sycl::group_barrier(partition);

                    // Store back Q (each lane copies its row).
                    // Q_local is stored in column-major order: Q_local[base + row + col*N] = Q(row, col)
                    // Each lane (representing a row) copies all columns of that row.
                    for (int32_t c = 0; c < N; ++c) {
                        Q_prob(lane, c) = Q_local[base + lane + c * N];
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
        if (!params.back_transform) {
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

        VectorView<T>::copy(ctx, d, d_in);
        VectorView<T>::copy(ctx, e, e_in);

        // Dispatch to a compile-time-specialized kernel.
        switch (n) {
            case 8:
                steqr_cta_impl<T, 8>(ctx, d, e, const_cast<MatrixView<T, MatrixFormat::Dense>&>(eigvects), params.max_sweeps, params.zero_threshold, pool);
                break;
            case 16:
                steqr_cta_impl<T, 16>(ctx, d, e, const_cast<MatrixView<T, MatrixFormat::Dense>&>(eigvects), params.max_sweeps, params.zero_threshold, pool);
                break;
            case 32:
                steqr_cta_impl<T, 32>(ctx, d, e, const_cast<MatrixView<T, MatrixFormat::Dense>&>(eigvects), params.max_sweeps, params.zero_threshold, pool);
                break;
            default:
                throw std::invalid_argument("steqr_cta currently supports n in {8,16,32}.");
        }

        // Copy back eigenvalues.
        VectorView<T>::copy(ctx, eigenvalues, d);

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