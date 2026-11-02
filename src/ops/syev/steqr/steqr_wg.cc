// SPDX-License-Identifier: MIT
#include <batchlas/blas/matrix.hh>
#include <batchlas/blas/functions.hh>
#include <batchlas/blas/extensions.hh>
#include <batchlas/blas/extra.hh>
#include <batchlas/util/kernel-heuristics.hh>
#include <batchlas/util/mempool.hh>
#include <array>
#include <tuple>
#include <utility>
#include <batchlas/backend_config.h>
#include <src/math-helpers.hh>
#include <src/queue.hh>
#include <src/util/template-instantiations.hh>
#include <src/sort.hh>
#include <src/kernels/common/info_span.hh>
#include <src/util/batch_mask.hh>
#include <src/ops/syev/steqr/steqr_internal.hh>
#include <src/ops/syev/tridiag/plane_rotation.hh>

namespace batchlas {
namespace detail {

// {c, -s} of LAPACK lartg(a, b): the rotation convention of the work-group bulge chase.
template <typename T>
inline std::array<T, 2> givens_rotation(const T& a, const T& b) {
    auto [c_l, s_l, r] = detail::lartg(a, b);
    (void)r;
    return std::array<T, 2>{c_l, -s_l};
}

// givens = {c, -s} of R = [[c, s], [-s, c]]: rows (i, j) <- R A R^T, e_{i-1} <- c e_{i-1} + s bulge,
// e_j <- c e_j; returns the new bulge s e_j. QR == true indexes top-down (virtual == physical);
// false bottom-up, d: k -> n-1-k and e: k -> ne-1-k = n-2-k, so one chase does QR and QL.
template <typename T>
inline T apply_givens_rotation(const VectorView<T>& d,
                               const VectorView<T>& e,
                               const T& prev_bulge,
                               size_t i,
                               size_t j,
                               const std::array<T, 2>& givens,
                               bool QR) {
    const T c = givens[0];
    const T s = givens[1];

    const size_t n = d.size();
    const size_t ne = e.size();

    auto d_get = [&](size_t k) -> T {
        return QR ? d(k) : d(n - 1 - k);
    };
    auto d_set = [&](size_t k, T val) {
        if (QR) {
            d(k) = val;
        } else {
            d(n - 1 - k) = val;
        }
    };
    auto e_get = [&](size_t k) -> T {
        return QR ? e(k) : e(ne - 1 - k);
    };
    auto e_set = [&](size_t k, T val) {
        if (QR) {
            e(k) = val;
        } else {
            e(ne - 1 - k) = val;
        }
    };

    T di = d_get(i);
    T dj = d_get(j);
    T ei = e_get(i);
    T ej = (j < ne) ? e_get(j) : T(0);

    T di_new = c * (c * di - ei * s) - s * (ei * c - s * dj);
    T dj_new = c * (c * dj + ei * s) + s * (ei * c + s * di);
    d_set(i, di_new);
    d_set(j, dj_new);

    if (i > 0) {
        T e_im1 = e_get(i - 1);
        e_set(i - 1, e_im1 * c - prev_bulge * s);
    }

    T ei_new = c * (c * ei + s * di) - s * (c * dj + s * ei);
    e_set(i, ei_new);

    if (j < ne) {
        e_set(j, c * ej);
    }

    return -ej * s;
}

} // namespace detail


namespace {

template <typename T> struct SteqrWgFrancisKernel {};
template <typename T> struct SteqrWgRotationKernel {};
template <typename T> struct SteqrLegacyFrancisKernel {};
template <typename T> struct SteqrLegacyRotationKernel {};

// The per-pass scratch steqr_wg_impl carves off the pool it is handed; steqr_wg_layout sizes it
// through workspace_bytes.
inline std::pair<Span<int32_t>, VectorView<std::array<int32_t, 2>>> steqr_wg_pass_layout(Queue& ctx, BumpAllocator& pool,
                                                                                          int64_t n, int64_t batch_size) {
    auto scan = pool.allocate<int32_t>(ctx, batch_size);
    auto tdi = pool.allocate<std::array<int32_t, 2>>(
        ctx, VectorView<std::array<int32_t, 2>>::required_span_length(n / 2, 1, n / 2, batch_size));
    return {scan, VectorView<std::array<int32_t, 2>>(tdi, n / 2, batch_size, 1, n / 2)};
}

// One pass: split every item into unreduced blocks, run up to max_sweeps sweeps on each block
// until it splits, then replay each block's stored rotations on its columns of Q. steqr_wg replays
// every stored slot; steqr_legacy (ReplayPerformedOnly) replays the sweep_counts[gid] performed.
template <typename T, bool ReplayPerformedOnly = false>
Event steqr_wg_impl(Queue& ctx,
                    const VectorView<T>& d,
                    const VectorView<T>& e,
                    JobType jobz,
                    const MatrixView<T, MatrixFormat::Dense>& Q,
                    const MatrixView<std::array<T, 2>, MatrixFormat::Dense>& givens_rotations,
                    const Span<std::array<int32_t, 3>>& deflation_indices,
                    const Span<ApplyOrder>& order_view,
                    const Span<int32_t>& sweep_counts,
                    BumpAllocator allocator,
                    size_t max_sweeps) {
    BATCHLAS_KERNEL_TRACE_SCOPE("steqr_wg");
    auto n = d.size();
    auto batch_size = d.batch_size();
    bool store_givens = jobz == JobType::EigenVectors;
    auto ncus = ctx.device().get_property(DeviceProperty::MAX_COMPUTE_UNITS);
    const auto [scan_view, temp_deflation_indices] = steqr_wg_pass_layout(ctx, allocator, n, batch_size);

    {
        BATCHLAS_KERNEL_TRACE_SCOPE("steqr_wg:deflation_ranges");
        // Per item: the maximal runs of nonzero e, as blocks [start, end] (1x1 runs skipped).
        ctx->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::range(batch_size), [=](sycl::id<1> id) {
                auto i = id[0];
                auto ebid = e.batch_item(i);
                auto ix = 0;
                auto sub_problem_ix = 0;
                // An inactive item lists no sub-problem, so no sweep or rotation touches it.
                while (e.active(i) && ix < n - 1) {
                    auto start_ix = ix;
                    if (ebid(ix) != T(0)) {
                        for (ix = ix + 1; ix < n - 1; ++ix) {
                            if (ebid(ix) == T(0)) break;
                        }
                    } else {
                        ix++;
                        continue;
                    }
                    auto end_ix = ix + 1;
                    temp_deflation_indices(sub_problem_ix, i) = {start_ix, end_ix};
                    sub_problem_ix++;
                }
                scan_view[i] = sub_problem_ix;
            });
        });
    }

    {
        BATCHLAS_KERNEL_TRACE_SCOPE("steqr_wg:deflation_scan");
        // (void) on an Event: deliberate. This Queue is in-order, so the next submission
        // is already ordered after this one and the Event carries nothing the caller needs.
        (void)detail::scan_inclusive_inplace<int32_t>(ctx, scan_view);
    }

    {
        BATCHLAS_KERNEL_TRACE_SCOPE("steqr_wg:deflation_writeout");
        ctx->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::range(batch_size), [=](sycl::id<1> id) {
                auto i = id[0];
                auto num_sub_problems = scan_view[i] - (i == 0 ? 0 : scan_view[i - 1]);
                auto offset = (i == 0 ? 0 : scan_view[i - 1]);
                for (size_t j = 0; j < num_sub_problems; ++j) {
                    deflation_indices[offset + j] = {static_cast<int32_t>(i), temp_deflation_indices(j, i)[0], temp_deflation_indices(j, i)[1]};
                }
            });
        });
    }

    {
        BATCHLAS_KERNEL_TRACE_SCOPE("steqr_wg:francis_sweep");
        ctx->submit([&](sycl::handler& cgh) {
            auto rotations_view = givens_rotations.kernel_view();
            using FrancisKernel = std::conditional_t<ReplayPerformedOnly, SteqrLegacyFrancisKernel<T>, SteqrWgFrancisKernel<T>>;
            batchlas::impl::parallel_for<FrancisKernel>(cgh, sycl::nd_range(sycl::range(ncus * 128), sycl::range(64)), [=](sycl::nd_item<1> item) {
                auto i = item.get_global_id(0);
                auto n_problems = scan_view[batch_size - 1];

                for (int gid = i; gid < n_problems; gid += item.get_global_range(0)) {
                    const auto [batch_ix, start_ix, end_ix] = deflation_indices[gid];
                    const int32_t len = end_ix - start_ix;
                    const VectorView<T> D = d.batch_item(batch_ix)(Slice(start_ix, end_ix));
                    const VectorView<T> E = e.batch_item(batch_ix)(Slice(start_ix, end_ix - 1));

                    if constexpr (ReplayPerformedOnly) {
                        if (store_givens) sweep_counts[gid] = 0;
                    }
                    // Converged1: a 1x1 block has no e to read and nothing to do.
                    if (len == 1) continue;
                    // Exact2x2: R A R^T = diag(rt1, rt2), Z <- Z R^T on (0, 1). Stored pairs are R^T = {c, -s}.
                    if (len == 2) {
                        if (store_givens) {
                            if constexpr (ReplayPerformedOnly) {
                                auto [rt1, rt2, c, s] = detail::laev2(D(0), E(0), D(1));
                                D(0) = rt1;
                                D(1) = rt2;
                                E(0) = T(0);
                                rotations_view(0, 0, gid) = {c, -s};
                                order_view[gid] = ApplyOrder::Forward;
                                sweep_counts[gid] = 1;
                            } else {
                                const auto [rt1, rt2, R] = tridiag::Sym2x2<T>{D(0), E(0), D(1)}.eig();
                                D(0) = rt1;
                                D(1) = rt2;
                                E(0) = T(0);
                                const tridiag::Givens<T> Rt = R.transpose();
                                rotations_view(0, 0, gid) = {Rt.c, Rt.s};
                                order_view[gid] = ApplyOrder::Forward;
                            }
                        } else {
                            auto [rt1, rt2] = detail::eigenvalues_2x2(D(0), E(0), D(1));
                            D(0) = rt1;
                            D(1) = rt2;
                            E(0) = T(0);
                        }
                        continue;
                    }

                    // Sweep: base's statements verbatim. The OrientedBlock/Sym2x2 form of the same
                    // arithmetic made CPU-queue (Backend::NETLIB) bits depend on batch position and
                    // cost 4-6 registers on sm_89 (steqr_mask_tests CtaEveryPattern).
                    const VectorView<T> d_ = D, e_ = E;
                    const int32_t n_local = len;
                    // The sweep converges the end the shift is taken from (virtual index
                    // n_local - 1), so orient that end at the SMALLER |d|, as dsteqr does.
                    bool QR = std::abs(d_(n_local - 1)) < std::abs(d_(0));
                    order_view[gid] = QR ? ApplyOrder::Forward : ApplyOrder::Backward;
                    // dsteqr per sweep: scale into [ssfmin, ssfmax] (per sweep here, per block in dsteqr),
                    // Wilkinson shift from the converging end, chase, split negligible e, unscale.
                    for (size_t k = 0; k < max_sweeps; ++k) {
                        auto anorm = std::abs(d_(n_local - 1));
                        for (size_t idx = 0; idx < n_local - 1; ++idx) {
                            anorm = std::fmax(anorm, std::fmax(std::abs(d_(idx)), std::abs(e_(idx))));
                        }

                        if (anorm > detail::ssfmax<T>()) {
                            auto alpha = detail::ssfmax<T>() / anorm;
                            for (size_t idx = 0; idx < n_local; ++idx) d_(idx) *= alpha;
                            for (size_t idx = 0; idx < n_local - 1; ++idx) e_(idx) *= alpha;
                        } else if (anorm < detail::ssfmin<T>() && anorm != T(0)) {
                            auto alpha = detail::ssfmin<T>() / anorm;
                            for (size_t idx = 0; idx < n_local; ++idx) d_(idx) *= alpha;
                            for (size_t idx = 0; idx < n_local - 1; ++idx) e_(idx) *= alpha;
                        }

                        auto d_get = [&](size_t idx) -> T {
                            return QR ? d_(idx) : d_(n_local - 1 - idx);
                        };
                        auto e_get = [&](size_t idx) -> T {
                            return QR ? e_(idx) : e_(n_local - 2 - idx);
                        };

                        const auto shift = detail::wilkinson_shift(d_get(n_local - 2),
                                                           e_get(n_local - 2),
                                                           d_get(n_local - 1));

                        auto [c0, s0] = detail::givens_rotation(d_get(0) - shift, e_get(0));
                        if (store_givens) {
                            rotations_view(0, k, gid) = {c0, s0};
                        }
                        auto bulge = detail::apply_givens_rotation(d_, e_, T(0), 0, 1, {c0, s0}, QR);

                        for (size_t j = 1; j < n_local - 1; ++j) {
                            auto [cj, sj] = detail::givens_rotation(e_get(j - 1), bulge);
                            if (store_givens) {
                                rotations_view(j, k, gid) = {cj, sj};
                            }
                            bulge = detail::apply_givens_rotation(d_, e_, bulge, j, j + 1, {cj, sj}, QR);
                        }

                        bool deflatable = false;
                        for (size_t j = 0; j < n_local - 1; ++j) {
                            if (std::abs(e_(j)) * std::abs(e_(j)) <= detail::eps2<T>() * std::abs(d_(j)) * std::abs(d_(j + 1)) +
                                                                   detail::safmin<T>()) {
                                e_(j) = T(0);
                                deflatable = true;
                            }
                        }

                        if (anorm > detail::ssfmax<T>()) {
                            auto alpha = anorm / detail::ssfmax<T>();
                            for (size_t idx = 0; idx < n_local; ++idx) d_(idx) *= alpha;
                            for (size_t idx = 0; idx < n_local - 1; ++idx) e_(idx) *= alpha;
                        } else if (anorm < detail::ssfmin<T>() && anorm != T(0)) {
                            auto alpha = anorm / detail::ssfmin<T>();
                            for (size_t idx = 0; idx < n_local; ++idx) d_(idx) *= alpha;
                            for (size_t idx = 0; idx < n_local - 1; ++idx) e_(idx) *= alpha;
                        }

                        if constexpr (ReplayPerformedOnly) {
                            if (store_givens) sweep_counts[gid] = static_cast<int32_t>(k + 1);
                        }
                        if (deflatable) break;
                    }
                }
            });
        });

        if constexpr (ReplayPerformedOnly) {
            ctx -> submit([&](sycl::handler& cgh) {
                auto Qview = Q.kernel_view();
                auto rotations_view = givens_rotations.kernel_view();
                batchlas::impl::parallel_for<SteqrLegacyRotationKernel<T>>(cgh, sycl::nd_range(sycl::range(Q.rows()*Q.batch_size()*2), sycl::range(Q.rows())), [=](sycl::nd_item<1> item) {
                    auto bid = item.get_group(0);
                    auto k = item.get_local_linear_id();
                    auto n_problems = scan_view[batch_size - 1];
                    for (int gid = bid; gid < n_problems; gid += item.get_group_range(0)) {
                    auto [batch_ix, start_ix, end_ix] = deflation_indices[gid];
                    auto Q_ = Qview.batch_item(batch_ix)(Slice(), Slice(start_ix, end_ix));
                    const int ncols = static_cast<int>(Q_.cols());
                    bool forward = order_view[gid] == ApplyOrder::Forward;
                    auto col_index = [&](int v) -> int {
                        return (forward)
                                ? v
                                : (ncols - 1 - v);
                    };

                    // Only replay the sweeps that were actually performed for this sub-problem.
                    // Slots beyond that are stale/uninitialised and must not be applied.
                    // sweep_counts is empty when eigenvectors were not requested; nothing to replay then.
                    const int n_sweeps = sweep_counts.size() == 0 ? 0 : static_cast<int>(sweep_counts[gid]);
                    for (int i = 0; i < n_sweeps; ++i) {
                        for (int j = 0; j < ncols - 1; ++j) {
                            auto [c, s] = rotations_view(j, i, gid);
                            if (c == T(1) && s == T(0)) continue; // Skip identity rotations

                            // Map virtual indices (j, j+1) to physical column indices.
                            int ix1 = col_index(j);
                            int ix2 = col_index(j + 1);

                            const T x = Q_(k, ix1);
                            const T y = Q_(k, ix2);
                            Q_(k, ix1) = c * x - s * y;
                            Q_(k, ix2) = s * x + c * y;
                        }
                    }
                    }
                });
            });
        } else {
            // Without vectors no rotation was stored and there is nothing to replay.
            if (store_givens) {
                ctx->submit([&](sycl::handler& cgh) {
                    auto Qview = Q.kernel_view();
                    auto rotations_view = givens_rotations.kernel_view();
                    batchlas::impl::parallel_for<SteqrWgRotationKernel<T>>(cgh, sycl::nd_range(sycl::range(Q.rows() * Q.batch_size() * 2), sycl::range(Q.rows())), [=](sycl::nd_item<1> item) {
                        auto bid = item.get_group(0);
                        auto row = item.get_local_linear_id();
                        auto n_problems = scan_view[batch_size - 1];
                        for (int gid = bid; gid < n_problems; gid += item.get_group_range(0)) {
                            const auto [batch_ix, start_ix, end_ix] = deflation_indices[gid];
                            auto Z = Qview.batch_item(batch_ix)(Slice(), Slice(start_ix, end_ix));
                            const int len = static_cast<int>(Z.cols());
                            const bool reversed = order_view[gid] != ApplyOrder::Forward;
                            auto column = [&](int v) -> int { return reversed ? len - 1 - v : v; };

                            // Every stored sweep in order (an unperformed one holds the pass's identity
                            // fill), each Z <- Z R^T on the oriented columns (j, j+1).
                            for (int sweep = 0; sweep < rotations_view.cols(); ++sweep) {
                                for (int j = 0; j < len - 1; ++j) {
                                    auto [c, s] = rotations_view(j, sweep, gid);
                                    const int lo = column(j);
                                    const int hi = column(j + 1);

                                    const T x = Z(row, lo);
                                    const T y = Z(row, hi);
                                    Z(row, lo) = c * x - s * y;
                                    Z(row, hi) = s * x + c * y;
                                }
                            }
                        }
                    });
                });
            }
        }
    }

    return ctx.get_event();
}

} // namespace

// Single description of steqr_wg's workspace (as steqr_legacy_layout); see workspace_bytes() in
// util/mempool.hh. Returns d, e, givens rotations, apply order, deflation indices and the scratch
// that each pass, then sort(), sub-allocates from. Sized from (n, batch, jobz) alone: sort()
// takes n * batch indices and an n x n x batch copy, whatever the caller's padding.
template <typename T>
auto steqr_wg_layout(Queue& ctx, BumpAllocator& pool, int64_t n, int64_t batch_size, JobType jobz,
                     const SteqrParams<T>& params) {
    const bool want_vectors = jobz == JobType::EigenVectors;
    const auto increment = params.transpose_working_vectors ? batch_size : 1;
    const auto d_stride = params.transpose_working_vectors ? 1 : n;
    const auto e_stride = params.transpose_working_vectors ? 1 : n - 1;
    auto d = VectorView<T>(pool.allocate<T>(ctx, VectorView<T>::required_span_length(n, increment, d_stride, batch_size)),
                           n, batch_size, increment, d_stride);
    auto e = VectorView<T>(pool.allocate<T>(ctx, VectorView<T>::required_span_length(n - 1, increment, e_stride, batch_size)),
                           n - 1, batch_size, increment, e_stride);

    const auto n_sweeps_to_store = (want_vectors && params.block_rotations)
                                       ? std::max<int64_t>(params.block_size * 2, params.max_sweeps)
                                       : int64_t(params.max_sweeps);
    const auto stride = (n - 1) * n_sweeps_to_store;
    const auto max_subproblems = n / 2 + 1;
    auto givens_rotations =
        want_vectors ? MatrixView<std::array<T, 2>>(pool.allocate<std::array<T, 2>>(ctx, stride * max_subproblems * batch_size).data(),
                                                    n - 1, n_sweeps_to_store, n - 1, stride, max_subproblems * batch_size)
                     : MatrixView<std::array<T, 2>>();
    auto apply_order = pool.allocate<ApplyOrder>(ctx, batch_size * max_subproblems);
    auto deflation_indices = pool.allocate<std::array<int32_t, 3>>(ctx, batch_size * max_subproblems);

    const size_t pass_scratch = workspace_bytes([&](BumpAllocator& p) { return steqr_wg_pass_layout(ctx, p, n, batch_size); });
    const MatrixView<T, MatrixFormat::Dense> sort_shape(nullptr, n, n, n, n * n, batch_size);
    const size_t sort_scratch = params.sort ? sort_buffer_size<T>(ctx, Span<T>(), sort_shape, jobz) : 0;
    auto scratch = pool.allocate<std::byte>(ctx, std::max(pass_scratch, sort_scratch));
    return std::make_tuple(d, e, givens_rotations, apply_order, deflation_indices, scratch);
}

template <Backend B, typename T>
Event steqr_wg(Queue& ctx,
               const VectorView<T>& d_in,
               const VectorView<T>& e_in,
               const VectorView<T>& eigenvalues,
               const Span<std::byte>& ws,
               JobType jobz,
               SteqrParams<T> params,
               const MatrixView<T, MatrixFormat::Dense>& eigvects,
               Span<int32_t> info) {
    const int64_t n = d_in.size();
    const int64_t batch_size = d_in.batch_size();
    // The working copies are rebuilt from the pool, so they take the call's mask back on.
    const auto mask = batchlas::detail::call_mask("steqr_wg", d_in, e_in, eigenvalues, eigvects);
    const VectorView<T> evals = eigenvalues.with_batch_mask(mask);
    if (params.sort) require_sortable(evals);

    if (jobz == JobType::EigenVectors) {
        if (eigvects.rows() != eigvects.cols()) {
            throw batchlas::invalid_argument("Matrix must be square for eigenvector computation.");
        }
        if (eigvects.rows() != n || eigvects.batch_size() != batch_size) {
            throw batchlas::invalid_argument("Eigenvector matrix has incompatible dimensions.");
        }
        if (!params.back_transform) {
            (void)eigvects.with_batch_mask(mask).fill_identity(ctx);
        }
    }
    const MatrixView<T, MatrixFormat::Dense> Z =
        jobz == JobType::EigenVectors ? eigvects.with_batch_mask(mask) : eigvects;

    auto pool = BumpAllocator(ws);
    auto [d_ws, e_ws, givens_rotations, apply_order, deflation_indices, scratch] =
        steqr_wg_layout<T>(ctx, pool, n, batch_size, jobz, params);
    const VectorView<T> d = d_ws.with_batch_mask(mask);
    const VectorView<T> e = e_ws.with_batch_mask(mask);
    (void)VectorView<T>::copy(ctx, d, d_in);
    (void)VectorView<T>::copy(ctx, e, e_in);

    // A pass splits off at least one e_j from every block that deflates within max_sweeps sweeps,
    // so n - 1 passes suffice unless a block stalls; any e_j left over is reported in info below.
    for (int64_t i = 0; i < n - 1; ++i) {
        if (jobz == JobType::EigenVectors) {
            (void)givens_rotations.fill(ctx, std::array<T, 2>{1, 0});
        }
        (void)steqr_wg_impl(ctx,
                      d,
                      e,
                      jobz,
                      Z,
                      givens_rotations,
                      deflation_indices,
                      apply_order,
                      Span<int32_t>(),
                      BumpAllocator(scratch),
                      params.max_sweeps);
    }

    // Per-item convergence status. The driver runs a FIXED n-1 passes with no test, so
    // the honest predicate is post hoc (as steqr_legacy): count, per item, the
    // off-diagonals that survived every pass -- LAPACK ?steqr's `info`. No workspace
    // (caller USM; an empty span skips the launch) and no zeroing: `steqr`/`stedc`
    // already cleared the span, and info_report only raises. See info_span.hh.
    if (int32_t* info_out = detail::info_ptr(info, batch_size)) {
        ctx->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::range<1>(static_cast<size_t>(batch_size)), [=](sycl::id<1> id) {
                if (!e.active(id[0])) return;  // I3
                auto e_item = e.batch_item(id[0]);
                int32_t unconverged = 0;
                for (int64_t j = 0; j < n - 1; ++j) {
                    if (e_item(j) != T(0)) ++unconverged;
                }
                detail::info_report(info_out, static_cast<int64_t>(id[0]), unconverged);
            });
        });
    }

    ctx->submit([&](sycl::handler& cgh) {
        batchlas::impl::parallel_for(cgh, sycl::nd_range(sycl::range(batch_size * n), sycl::range(n)), [=](sycl::nd_item<1> item) {
            auto bid = item.get_group_linear_id();
            auto intra = item.get_local_id(0);
            if (!evals.active(bid)) return;  // I1, no barrier
            evals(intra, bid) = d(intra, bid);
        });
    });

    if (params.sort) {
        // The passes are finished, so their scratch is free for sort to take over.
        (void)sort(ctx, evals, Z, jobz, params.sort_order, scratch);
    }
    return ctx.get_event();
}

template <typename T>
size_t steqr_wg_buffer_size(Queue& ctx,
                            const VectorView<T>& d,
                            const VectorView<T>& e,
                            const VectorView<T>& eigenvalues,
                            JobType jobz,
                            SteqrParams<T> params) {
    static_cast<void>(e);
    static_cast<void>(eigenvalues);
    const int64_t n = d.size();
    const int64_t batch_size = d.batch_size();
    return workspace_bytes([&](BumpAllocator& pool) {
        return steqr_wg_layout<T>(ctx, pool, n, batch_size, jobz, params);
    });
}

#define STEQR_WG_INSTANTIATE(back, fp) \
template Event steqr_wg<back, BATCHLAS_UNPAREN fp>(Queue&, const VectorView<BATCHLAS_UNPAREN fp>&, const VectorView<BATCHLAS_UNPAREN fp>&, const VectorView<BATCHLAS_UNPAREN fp>&, const Span<std::byte>&, JobType, SteqrParams<BATCHLAS_UNPAREN fp>, const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, Span<int32_t>);

BATCHLAS_INSTANTIATE_REAL_ALL_BACKENDS(STEQR_WG_INSTANTIATE)

#define STEQR_WG_BUFFER_SIZE_INSTANTIATE(fp) \
template size_t steqr_wg_buffer_size<BATCHLAS_UNPAREN fp>(Queue&, const VectorView<BATCHLAS_UNPAREN fp>&, const VectorView<BATCHLAS_UNPAREN fp>&, const VectorView<BATCHLAS_UNPAREN fp>&, JobType, SteqrParams<BATCHLAS_UNPAREN fp>);

BATCHLAS_FOR_EACH_REAL_TYPE(STEQR_WG_BUFFER_SIZE_INSTANTIATE)

#undef STEQR_WG_BUFFER_SIZE_INSTANTIATE
#undef STEQR_WG_INSTANTIATE

// steqr_legacy: NETLIB stedc's leaf solver. The same pass as steqr_wg (ReplayPerformedOnly), an
// early exit once T is split, and its own workspace layout (unchanged sizes).
template <typename T>
struct SteqrLegacyWorkspace {
    VectorView<T> d;
    VectorView<T> e;
    Span<T> apply_Q_ws;
    MatrixView<std::array<T, 2>, MatrixFormat::Dense> givens_rotations;
    Span<ApplyOrder> apply_order;
    Span<std::array<int32_t, 3>> deflation_indices;
    Span<int32_t> sweep_counts;
    // Reused, not shared: each steqr_wg_impl pass sub-allocates from a copy of this,
    // and once the passes are done sort() takes it over. Sized for whichever
    // needs more.
    Span<std::byte> scratch;
};

// Single description of steqr_legacy's workspace; see workspace_bytes() in
// util/mempool.hh.
//
// `eigvects` is only inspected for its shape (to size the sort scratch), so the
// sizing entry point may pass a shape-only view over no memory.
template <typename T>
SteqrLegacyWorkspace<T> steqr_legacy_layout(Queue& ctx,
                                            BumpAllocator& pool,
                                            int64_t n,
                                            int64_t batch_size,
                                            JobType jobz,
                                            const SteqrParams<T>& params,
                                            const VectorView<T>& eigenvalues,
                                            const MatrixView<T, MatrixFormat::Dense>& eigvects) {
    const bool want_vectors = jobz == JobType::EigenVectors;
    const auto increment = params.transpose_working_vectors ? batch_size : 1;
    const auto d_stride = params.transpose_working_vectors ? 1 : n;
    const auto e_stride = params.transpose_working_vectors ? 1 : n - 1;

    auto d = VectorView<T>(pool.allocate<T>(ctx, VectorView<T>::required_span_length(n, increment, d_stride, batch_size)),
                           n, batch_size, increment, d_stride);
    auto e = VectorView<T>(pool.allocate<T>(ctx, VectorView<T>::required_span_length(n - 1, increment, e_stride, batch_size)),
                           n - 1, batch_size, increment, e_stride);

    // Nothing reads this any more -- it is kept only so that converting this
    // function does not change how much workspace callers must supply. Removing
    // it is a separate, deliberate change.
    auto apply_Q_ws = pool.allocate<T>(
        ctx, want_vectors ? (batch_size * params.block_size * 2 * params.block_size * 2 + batch_size * n * params.block_size * 4) : 0);

    const auto n_sweeps_to_store = (want_vectors && params.block_rotations)
                                       ? std::max(params.block_size * 2, params.max_sweeps)
                                       : params.max_sweeps;
    const auto stride = (n - 1) * n_sweeps_to_store;
    const auto max_subproblems = n / 2 + 1;

    auto givens_rotations =
        want_vectors ? MatrixView<std::array<T, 2>>(pool.allocate<std::array<T, 2>>(ctx, stride * max_subproblems * batch_size).data(),
                                                    n - 1, n_sweeps_to_store, n - 1, stride, max_subproblems * batch_size)
                     : MatrixView<std::array<T, 2>>();
    auto apply_order = pool.allocate<ApplyOrder>(ctx, batch_size * max_subproblems);
    auto deflation_indices = pool.allocate<std::array<int32_t, 3>>(ctx, batch_size * max_subproblems);
    auto sweep_counts = want_vectors ? pool.allocate<int32_t>(ctx, batch_size * max_subproblems) : Span<int32_t>();

    const size_t pass_scratch = workspace_bytes([&](BumpAllocator& p) { return steqr_wg_pass_layout(ctx, p, n, batch_size); });
    const size_t sort_scratch = params.sort ? sort_buffer_size<T>(ctx, eigenvalues.data(), eigvects, jobz) : 0;
    auto scratch = pool.allocate<std::byte>(ctx, std::max(pass_scratch, sort_scratch));

    return {d, e, apply_Q_ws, givens_rotations, apply_order, deflation_indices, sweep_counts, scratch};
}

template <Backend B, typename T>
Event steqr_legacy(Queue& ctx, const VectorView<T>& d_in, const VectorView<T>& e_in, const VectorView<T>& eigenvalues_in, const Span<std::byte>& ws,
                   JobType jobz, SteqrParams<T> params, const MatrixView<T, MatrixFormat::Dense>& eigvects_in,
                   Span<int32_t> info) {
    const int64_t n = d_in.size();
    const int64_t batch_size = d_in.batch_size();
    // The working copies are rebuilt from the pool, so they take the call's mask back on.
    const auto mask = batchlas::detail::call_mask("steqr_legacy", d_in, e_in, eigenvalues_in, eigvects_in);
    const VectorView<T> eigenvalues = eigenvalues_in.with_batch_mask(mask);
    const MatrixView<T, MatrixFormat::Dense> eigvects =
        jobz == JobType::EigenVectors ? eigvects_in.with_batch_mask(mask) : eigvects_in;

    if (jobz == JobType::EigenVectors) {
        // Ensure the eigenvector matrix is square and matches the problem size.
        if (eigvects.rows() != eigvects.cols()) {
            throw batchlas::invalid_argument("Matrix must be square for eigenvector computation.");
        }
        if (eigvects.rows() != n || eigvects.batch_size() != batch_size) {
            throw batchlas::invalid_argument("Eigenvector matrix has incompatible dimensions.");
        }
        if (!params.back_transform) {
            (void)eigvects.fill_identity(ctx);
        }
    }

    auto pool = BumpAllocator(ws);
    auto wsl = steqr_legacy_layout<T>(ctx, pool, n, batch_size, jobz, params, eigenvalues, eigvects);
    const VectorView<T> d = wsl.d.with_batch_mask(mask);
    const VectorView<T> e = wsl.e.with_batch_mask(mask);
    //Copy inputs to working buffers
    (void)VectorView<T>::copy(ctx, d, d_in);
    (void)VectorView<T>::copy(ctx, e, e_in);

    auto givens_rotations = wsl.givens_rotations;
    auto apply_order = wsl.apply_order;
    auto deflation_indices = wsl.deflation_indices;
    auto sweep_counts = wsl.sweep_counts;

    // n - 1 passes suffice unless a block stalls (see steqr_wg). Random inputs converge long
    // before that, and each pass is O(n^2 * batch) work, so bail out once T is fully split.
    UnifiedVector<int32_t> not_converged(1, 0);
    auto converged_flag = not_converged.to_span();
    ctx.wait();
    for (int64_t i = 0; i < n - 1; ++i) {
        not_converged[0] = 0; // Safe: the queue is idle at this point.
        (void)steqr_wg_impl<T, true>(ctx, d, e, jobz, eigvects, givens_rotations, deflation_indices, apply_order, sweep_counts,
                   BumpAllocator(wsl.scratch), params.max_sweeps);

        ctx -> submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::range(batch_size), [=](sycl::id<1> id) {
                if (!e.active(id[0])) return;  // I3: neutral for the host's convergence read
                auto ebid = e.batch_item(id[0]);
                for (int64_t j = 0; j < n - 1; ++j) {
                    if (ebid(j) != T(0)) {
                        sycl::atomic_ref<int32_t, sycl::memory_order::relaxed,
                                         sycl::memory_scope::device> flag(converged_flag[0]);
                        flag.store(1);
                        return;
                    }
                }
            });
        });
        ctx.wait();
        if (not_converged[0] == 0) break;
    }

    // Per-item status, from the same predicate the loop above already tests.
    //
    // `converged_flag` is one int32 for the WHOLE batch and is a loop-termination
    // signal, not diagnostics: it says some item still has a live off-diagonal, and
    // it is reset at the top of every pass. This re-runs the scan once, per item and
    // per surviving off-diagonal, which is what LAPACK's ?steqr info counts. No
    // workspace: `info` is the caller's USM and an empty span skips the launch.
    // No zeroing here: `steqr` (and `stedc`, for a leaf solve) has already cleared
    // the span, and info_report only ever raises. See info_span.hh.
    if (int32_t* info_out = detail::info_ptr(info, batch_size)) {
        ctx->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::range<1>(static_cast<size_t>(batch_size)), [=](sycl::id<1> id) {
                if (!e.active(id[0])) return;  // I3
                auto ebid = e.batch_item(id[0]);
                int32_t unconverged = 0;
                for (int64_t j = 0; j < n - 1; ++j) {
                    if (ebid(j) != T(0)) ++unconverged;
                }
                detail::info_report(info_out, static_cast<int64_t>(id[0]), unconverged);
            });
        });
    }

    ctx -> submit([&](sycl::handler& cgh) {
        batchlas::impl::parallel_for(cgh, sycl::nd_range(sycl::range(batch_size* n), sycl::range(n)), [=](sycl::nd_item<1> item) {
            auto bid = item.get_group_linear_id();
            auto intra = item.get_local_id(0);
            if (!eigenvalues.active(bid)) return;  // I1, no barrier
            eigenvalues(intra, bid) = d(intra, bid);
        });
    });

    if (params.sort){
        // The passes are finished, so their scratch is free for sort to take over.
        (void)sort(ctx, eigenvalues, eigvects, jobz, params.sort_order, wsl.scratch);
    }
    return ctx.get_event();
}

template <typename T>
size_t steqr_legacy_buffer_size(Queue& ctx, const VectorView<T>& d, const VectorView<T>& e,
                                const VectorView<T>& eigenvalues, JobType jobz, SteqrParams<T> params) {
    static_cast<void>(e);
    const auto n = d.size();
    const auto batch_size = d.batch_size();
    // Shape-only stand-in for the eigenvector matrix; the layout reads its
    // dimensions to size the sort scratch and nothing else.
    const MatrixView<T, MatrixFormat::Dense> eigvects_shape(nullptr, n, n, n, n * n, batch_size);
    return workspace_bytes([&](BumpAllocator& pool) {
        return steqr_legacy_layout<T>(ctx, pool, n, batch_size, jobz, params, eigenvalues, eigvects_shape);
    });
}



#define STEQR_LEGACY_INSTANTIATE(back, fp) \
template Event steqr_legacy<back, BATCHLAS_UNPAREN fp>(Queue&, const VectorView<BATCHLAS_UNPAREN fp>&, const VectorView<BATCHLAS_UNPAREN fp>&, const VectorView<BATCHLAS_UNPAREN fp>&, const Span<std::byte>&, JobType, SteqrParams<BATCHLAS_UNPAREN fp>, const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, Span<int32_t>);

#define STEQR_LEGACY_INSTANTIATE_FOR_BACKEND(back) \
    BATCHLAS_FOR_EACH_REAL_TYPE_1(STEQR_LEGACY_INSTANTIATE, back)

#if BATCHLAS_HAS_CUDA_BACKEND
STEQR_LEGACY_INSTANTIATE_FOR_BACKEND(Backend::CUDA)
#endif

#if BATCHLAS_HAS_SYCL_BACKEND
BATCHLAS_FOR_EACH_DEVICE_REAL_TYPE_1(STEQR_LEGACY_INSTANTIATE, Backend::SYCL)
#endif

#if BATCHLAS_HAS_HOST_BACKEND
STEQR_LEGACY_INSTANTIATE_FOR_BACKEND(Backend::NETLIB)
#endif

#define STEQR_LEGACY_BUFFER_SIZE_INSTANTIATE(fp) \
template size_t steqr_legacy_buffer_size<BATCHLAS_UNPAREN fp>(Queue&, const VectorView<BATCHLAS_UNPAREN fp>&, const VectorView<BATCHLAS_UNPAREN fp>&, const VectorView<BATCHLAS_UNPAREN fp>&, JobType, SteqrParams<BATCHLAS_UNPAREN fp>);

BATCHLAS_FOR_EACH_REAL_TYPE(STEQR_LEGACY_BUFFER_SIZE_INSTANTIATE)

#undef STEQR_LEGACY_BUFFER_SIZE_INSTANTIATE
#undef STEQR_LEGACY_INSTANTIATE_FOR_BACKEND
#undef STEQR_LEGACY_INSTANTIATE

} // namespace batchlas
