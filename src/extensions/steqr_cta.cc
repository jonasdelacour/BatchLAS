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

    template <typename T, size_t N>
    class SteqrCTAKernel;

    template <typename T>
    inline void lartg_device(T f, T g, T& c, T& s, T& r) {
        // Stable Givens rotation similar to LAPACK xLARTG.
        if (g == T(0)) {
            c = T(1);
            s = T(0);
            r = f;
            return;
        }
        if (f == T(0)) {
            c = T(0);
            s = T(1);
            r = g;
            return;
        }
        const T scale = sycl::fabs(f) + sycl::fabs(g);
        const T fs = f / scale;
        const T gs = g / scale;
        r = scale * sycl::sqrt(fs * fs + gs * gs);
        r = sycl::copysign(r, f);
        c = f / r;
        s = g / r;
    }

    template <typename T>
    inline void laev2_device(T a, T b, T c, T& rt1, T& rt2, T& cs1, T& sn1) {
        // Symmetric 2x2 eigendecomposition: [a b; b c].
        if (b == T(0)) {
            rt1 = a;
            rt2 = c;
            cs1 = T(1);
            sn1 = T(0);
            return;
        }

        const T sm = a + c;
        const T df = a - c;
        const T rt = sycl::hypot(df, T(2) * b);
        const T rt_sign = sycl::copysign(rt, sm);
        rt1 = (sm + rt_sign) / T(2);
        rt2 = (sm - rt_sign) / T(2);

        // Jacobi rotation that diagonalizes.
        const T tau = (c - a) / (T(2) * b);
        const T t = sycl::copysign(T(1), tau) / (sycl::fabs(tau) + sycl::sqrt(T(1) + tau * tau));
        cs1 = T(1) / sycl::sqrt(T(1) + t * t);
        sn1 = t * cs1;
    }

    template <typename T, size_t N, typename Partition, typename LocalAcc>
    inline void apply_col_rotation(const Partition& partition,
                                   LocalAcc& Q_local,
                                   size_t base,
                                   size_t col0,
                                   size_t col1,
                                   T c,
                                   T s) {
        // Right-multiply Q by the Givens rotation acting on columns (col0,col1).
        const size_t row = static_cast<size_t>(partition.get_local_linear_id());
        const size_t i0 = base + row * N + col0;
        const size_t i1 = base + row * N + col1;
        const T q0 = Q_local[i0];
        const T q1 = Q_local[i1];
        Q_local[i0] = c * q0 + s * q1;
        Q_local[i1] = -s * q0 + c * q1;
    }

    template <typename T, size_t N, typename Partition, typename LocalAcc>
    inline void bitonic_sort_with_cols(const Partition& partition,
                                       T& d_lane,
                                       LocalAcc& Q_local,
                                       size_t base) {
        // Bitonic sort of d across lanes (ascending). N is power-of-two.
        // When swapping eigenvalues between lanes, also swap the corresponding columns in Q_local.
        const size_t lane = static_cast<size_t>(partition.get_local_linear_id());

        for (size_t k = 2; k <= N; k <<= 1) {
            for (size_t j = (k >> 1); j > 0; j >>= 1) {
                const size_t partner = lane ^ j;
                const T other = sycl::permute_group_by_xor(partition, d_lane, j);
                const bool up = ((lane & k) == 0);
                const bool do_swap = up ? (d_lane > other) : (d_lane < other);

                if (do_swap) {
                    d_lane = other;
                }

                // Swap columns once per pair to avoid races.
                if (do_swap && lane < partner) {
                    for (size_t r = 0; r < N; ++r) {
                        const size_t a = base + r * N + lane;
                        const size_t b = base + r * N + partner;
                        const T tmp = Q_local[a];
                        Q_local[a] = Q_local[b];
                        Q_local[b] = tmp;
                    }
                }
            }
        }
    }

    template <typename T, typename Partition>
    inline bool deflate(Partition partition,
                        T& e,
                        T& d,
                        size_t start_ix,
                        size_t end_ix,
                        T zero_threshold) {
        const size_t lane = static_cast<size_t>(partition.get_local_linear_id());
        const size_t partition_size = static_cast<size_t>(partition.get_local_range().size());
        const bool lane_in_active_range = (lane + 1 < partition_size) && (lane >= start_ix) && (lane + 1 < end_ix);

        // We need d_{i+1} (neighbor lane's diagonal). A 1-lane shift is the most direct.
        // Note: for lanes without i+1 (last lane), the result is unspecified, but those
        // lanes never use d_ip1 due to lane_in_active_range.
        const T d_ip1 = sycl::shift_group_left(partition, d, 1);

        bool deflated_local = false;
        if (lane_in_active_range) {
            // Absolute cutoff (user-controlled) first.
            if (sycl::fabs(e) <= zero_threshold) {
                e = T(0);
                deflated_local = true;
            } else if (e != T(0)) {
                // LAPACK-style relative deflation test:
                // |e|^2 <= eps2 * |d_i| * |d_{i+1}| + safmin
                const T rhs = internal::eps2<T>() * sycl::fabs(d) * sycl::fabs(d_ip1) + internal::safmin<T>();
                if (sycl::fabs(e) * sycl::fabs(e) <= rhs) {
                    e = T(0);
                    deflated_local = true;
                }
            }
        }

        // Return whether any lane deflated in this partition.
        return sycl::any_of_group(partition, deflated_local);
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
            size_t sg_size = sg_sizes.empty() ? size_t(1) : static_cast<size_t>(sg_sizes[0]);
            for (auto sgs : sg_sizes) {
                if (static_cast<size_t>(sgs) == 32) sg_size = 32;
            }
            if constexpr (N <= 16) {
                for (auto sgs : sg_sizes) {
                    if (static_cast<size_t>(sgs) == 16) sg_size = 16;
                }
            }

            // Minimal, safe mapping: work-group size is LCM(N, sg_size), so we can form fixed-size partitions of size N.
            const size_t wg_size = std::lcm<size_t>(static_cast<size_t>(N), sg_size);
            const size_t probs_per_wg = wg_size / static_cast<size_t>(N);

            const size_t num_wg = (batch_size + probs_per_wg - 1) / probs_per_wg;
            const size_t global_size = num_wg * wg_size;

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
                    const size_t part_id = static_cast<size_t>(partition.get_group_linear_id());
                    const size_t lane = static_cast<size_t>(partition.get_local_linear_id());
                    const size_t prob_id = wg_id * probs_per_wg + part_id;
                    if (prob_id >= batch_size) return;
                    auto d_prob = d.batch_item(prob_id);
                    auto e_prob = e.batch_item(prob_id);
                    auto Q_prob = Q_view.batch_item(prob_id);                    

                    const size_t base = part_id * N * N;
                    const size_t base_d = part_id * N;
                    const size_t base_e = part_id * (N - 1);

                    // Initialize Q_local for this problem as identity.
                    for (size_t i = lane; i < N * N; i += static_cast<size_t>(partition.get_local_range().size())) {
                        Q_local[base + i] = (i / N == i % N) ? T(1) : T(0);
                    }

                    // Load D/E into registers (one element per lane).
                    T d_ = d_prob(lane);
                    T e_ = (lane < N - 1) ? e_prob(lane) : T(0);

                    // ---- Outer LAPACK-like split loop over blocks separated by E==0 ----
                    size_t l1 = 0;
                    while (l1 < N) {
                        // Mark the split explicitly as LAPACK does: E(l1-1)=0.
                        if (l1 > 0 && lane == (l1 - 1) && lane < (N - 1)) {
                            e_ = T(0);
                        }

                        // Deflation pass over the remaining tail to create more zeros in E.
                        (void)deflate(partition, e_, d_, l1, N, zero_threshold);

                        // Find end of current block: first index i>=l1 where E(i)==0; if none, block ends at N-1.
                        const size_t cand_m = (lane >= l1 && lane < (N - 1) && e_ == T(0)) ? lane : (N - 1);
                        const size_t lendsv = sycl::reduce_over_group(partition, cand_m, sycl::minimum<size_t>());
                        const size_t lsv = l1;

                        // Next block starts after lendsv.
                        l1 = lendsv + 1;

                        // Size-1 block.
                        if (lendsv <= lsv) {
                            continue;
                        }

                        // Choose between QL and QR (LAPACK swaps ends when |D(lend)| < |D(l)|).
                        const T dl = sycl::fabs(sycl::select_from_group(partition, d_, lsv));
                        const T dlen = sycl::fabs(sycl::select_from_group(partition, d_, lendsv));
                        const bool do_ql = !(dlen < dl);

                        if (do_ql) {
                            // ---------------- QL iteration: converge from the top (l grows) ----------------
                            size_t l = lsv;
                            const size_t lend = lendsv;

                            while (l <= lend) {
                                if (l == lend) {
                                    l += 1;
                                    continue;
                                }

                                // Inner loop: iterate up to max_sweeps times to converge eigenvalue at position l
                                size_t jiter = 0;
                                while (jiter < max_sweeps) {
                                    // Deflate within current active subproblem [l..lend].
                                    (void)deflate(partition, e_, d_, l, lend + 1, zero_threshold);

                                    // Find first m in [l..lend-1] such that E(m)==0; if none, m=lend.
                                    const size_t cand = (lane >= l && lane < lend && e_ == T(0)) ? lane : lend;
                                    const size_t m = sycl::reduce_over_group(partition, cand, sycl::minimum<size_t>());

                                    if (m == l) {
                                        // Converged! Move to next eigenvalue.
                                        l += 1;
                                        break;
                                    }

                                    if (m == l + 1) {
                                        // 2x2 block at (l,l+1).
                                        const T a = sycl::select_from_group(partition, d_, l);
                                        const T b = sycl::select_from_group(partition, e_, l);
                                        const T c = sycl::select_from_group(partition, d_, l + 1);
                                        T rt1, rt2, cs, sn;
                                        laev2_device(a, b, c, rt1, rt2, cs, sn);

                                        if (lane == l) {
                                            d_ = rt1;
                                            e_ = T(0);
                                        }
                                        if (lane == (l + 1)) {
                                            d_ = rt2;
                                        }

                                        apply_col_rotation<T, N>(partition, Q_local, base, l, l + 1, cs, sn);

                                        l += 2;
                                        break;
                                    }

                                    // Perform one QL sweep iteration
                                    jiter++;
                                // ---- Implicit QL step on subblock [l..m] (m>=l+2) ----
                                // Stage the current D/E into local memory so the bulge chase can be done scalarly by a leader lane.
                                D_local[base_d + lane] = d_;
                                if (lane < (N - 1)) {
                                    E_local[base_e + lane] = e_;
                                }
                                sycl::group_barrier(partition);

                                // Leader-lane scalar state.
                                T g = T(0);
                                T c = T(1);
                                T s = T(1);  // LAPACK initializes S=ONE before bulge chase
                                T p = T(0);

                                if (lane == 0) {
                                    const T p0 = D_local[base_d + l];
                                    const T e0 = E_local[base_e + l];
                                    const T dlp1 = D_local[base_d + (l + 1)];

                                    // Wilkinson-like shift initialization (STEQR style).
                                    T gg = (dlp1 - p0) / (T(2) * e0);
                                    const T rr = sycl::hypot(gg, T(1));
                                    g = D_local[base_d + m] - p0 + e0 / (gg + sycl::copysign(rr, gg));
                                    
                                    // p starts at zero in LAPACK bulge chase (NOT d[m])
                                    // p = D_local[base_d + m];  // WRONG!
                                }

                                // Bulge chase upward: i = m-1 down to l.
                                // The first step uses g and E[m-1] to form the first Givens rotation.
                                // Subsequent steps use the previous rotation's sine/cosine.
                                for (size_t i = m; i-- > l;) {
                                    T c1 = T(0);
                                    T s1 = T(0);

                                    if (lane == 0) {
                                        const T ei = E_local[base_e + i];
                                        const T di = D_local[base_d + i];
                                        const T dip1 = D_local[base_d + (i + 1)];

                                        // LAPACK computes F = S*E(I) and B = C*E(I) for ALL iterations
                                        // including the first one (where C=1, S=1 initially)
                                        T f = s * ei;
                                        T b = c * ei;


                                        T r1;
                                        lartg_device(g, f, c1, s1, r1);


                                        // Propagate the updated off-diagonal one position up (matches STEQR's in-loop updates).
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

                                    // Broadcast the rotation so all lanes can update eigenvectors in parallel.
                                    const T c1b = sycl::group_broadcast(partition, c1, 0);
                                    const T s1b = sycl::group_broadcast(partition, s1, 0);
                                    apply_col_rotation<T, N>(partition, Q_local, base, i, i + 1, c1b, s1b);
                                }

                                if (lane == 0) {
                                    D_local[base_d + l] = D_local[base_d + l] - p;
                                    if (l < (N - 1)) {
                                        E_local[base_e + l] = g;
                                    }

                                }

                                sycl::group_barrier(partition);

                                // Reload D/E registers from local memory for subsequent deflation scans.
                                d_ = D_local[base_d + lane];
                                e_ = (lane < (N - 1)) ? E_local[base_e + lane] : T(0);
                                }  // end inner jiter loop for QL
                            }  // end while (l <= lend) for QL
                        } else {
                            // ---------------- QR iteration: converge from the bottom (l shrinks) ----------------
                            size_t l = lendsv;
                            const size_t lend = lsv;

                            while (l >= lend) {
                                if (l == lend) {
                                    if (l == 0) break;
                                    l -= 1;
                                    continue;
                                }

                                // Inner loop: iterate up to max_sweeps times to converge eigenvalue at position l
                                size_t jiter = 0;
                                while (jiter < max_sweeps) {
                                    (void)deflate(partition, e_, d_, lend, l + 1, zero_threshold);

                                    // Find m scanning downward: look for E(i)==0 and take the largest i+1.
                                    const size_t cand_m2 = (lane >= lend && lane < l && e_ == T(0)) ? (lane + 1) : lend;
                                    const size_t m = sycl::reduce_over_group(partition, cand_m2, sycl::maximum<size_t>());

                                    if (m == l) {
                                        // Converged! Move to next eigenvalue.
                                        if (l == 0) break;
                                        l -= 1;
                                        break;
                                    }

                                    if (m + 1 == l) {
                                        // 2x2 block at (l-1,l).
                                        const size_t l0 = l - 1;
                                        const T a = sycl::select_from_group(partition, d_, l0);
                                        const T b = sycl::select_from_group(partition, e_, l0);
                                        const T c2 = sycl::select_from_group(partition, d_, l);
                                        T rt1, rt2, cs, sn;
                                        laev2_device(a, b, c2, rt1, rt2, cs, sn);

                                        if (lane == l0) {
                                            d_ = rt1;
                                            e_ = T(0);
                                        }
                                        if (lane == l) {
                                            d_ = rt2;
                                        }

                                        apply_col_rotation<T, N>(partition, Q_local, base, l0, l, cs, sn);

                                        if (l <= 1) break;
                                        l -= 2;
                                        break;
                                    }

                                    // Perform one QR sweep iteration
                                    jiter++;

                                // ---- Implicit QR step on subblock [m..l] ----
                                // Stage the current D/E into local memory so the bulge chase can be done scalarly by a leader lane.
                                D_local[base_d + lane] = d_;
                                if (lane < (N - 1)) {
                                    E_local[base_e + lane] = e_;
                                }
                                sycl::group_barrier(partition);

                                // Leader-lane scalar state.
                                T g = T(0);
                                T c = T(1);
                                T s = T(1);  // LAPACK initializes S=ONE before bulge chase
                                T p = T(0);

                                if (lane == 0) {
                                    const T p0 = D_local[base_d + l];
                                    const T e0 = E_local[base_e + (l - 1)];
                                    const T dlm1 = D_local[base_d + (l - 1)];

                                    T gg = (dlm1 - p0) / (T(2) * e0);
                                    const T rr = sycl::hypot(gg, T(1));
                                    g = D_local[base_d + m] - p0 + e0 / (gg + sycl::copysign(rr, gg));
                                    
                                    // p starts at zero in LAPACK bulge chase (NOT d[m])
                                    // p = D_local[base_d + m];  // WRONG!
                                }

                                // Bulge chase downward: i = m .. l-1.
                                // The first step uses g and E[m] to form the first Givens rotation.
                                // Subsequent steps use the previous rotation's sine/cosine.
                                for (size_t i = m; i < l; ++i) {
                                    T c1 = T(0);
                                    T s1 = T(0);

                                    if (lane == 0) {
                                        const T ei = E_local[base_e + i];
                                        const T di = D_local[base_d + i];
                                        const T dip1 = D_local[base_d + (i + 1)];

                                        // LAPACK computes F = S*E(I) and B = C*E(I) for ALL iterations
                                        // including the first one (where C=1, S=1 initially)
                                        T f = s * ei;
                                        T b = c * ei;

                                        T r1;
                                        lartg_device(g, f, c1, s1, r1);

                                        // Propagate the updated off-diagonal one position down (matches the QR in-loop updates).
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

                                    // Broadcast the rotation so all lanes can update eigenvectors in parallel.
                                    const T c1b = sycl::group_broadcast(partition, c1, 0);
                                    const T s1b = sycl::group_broadcast(partition, s1, 0);
                                    apply_col_rotation<T, N>(partition, Q_local, base, i, i + 1, c1b, s1b);
                                }

                                if (lane == 0) {
                                    D_local[base_d + l] = D_local[base_d + l] - p;
                                    E_local[base_e + (l - 1)] = g;
                                }

                                sycl::group_barrier(partition);

                                // Reload D/E registers from local memory for subsequent deflation scans.
                                d_ = D_local[base_d + lane];
                                e_ = (lane < (N - 1)) ? E_local[base_e + lane] : T(0);
                                }  // end inner jiter loop for QR
                            }  // end while (l >= lend) for QR
                        }  // end if (do_ql) else
                    }  // end outer block split loop

                    // Make sure all lanes are done updating Q_local before copying out.
                    sycl::group_barrier(partition);

                    // Store back D/E (one element per lane).
                    d_prob(lane) = d_;
                    if (lane < (N - 1)) {
                        e_prob(lane) = e_;
                    }

                    // Store back Q (strided copy over N*N elements).
                    for (size_t idx = lane; idx < N * N; idx += static_cast<size_t>(partition.get_local_range().size())) {
                        const size_t r = idx / N;
                        const size_t c = idx - r * N;
                        Q_prob(r, c) = Q_local[base + idx];
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