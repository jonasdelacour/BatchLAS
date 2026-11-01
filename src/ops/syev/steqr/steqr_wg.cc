// SPDX-License-Identifier: MIT
#include <batchlas/blas/matrix.hh>
#include <batchlas/blas/functions.hh>
#include <batchlas/blas/extensions.hh>
#include <batchlas/blas/extra.hh>
#include <batchlas/util/kernel-heuristics.hh>
#include <batchlas/util/mempool.hh>
#include <tuple>
#include <utility>
#include <batchlas/backend_config.h>
#include <src/math-helpers.hh>
#include <src/queue.hh>
#include <src/util/template-instantiations.hh>
#include <src/sort.hh>
#include <src/kernels/common/info_span.hh>
#include <src/util/batch_mask.hh>
#include <src/ops/syev/steqr/ql_chase.hh>
#include <src/ops/syev/steqr/steqr_internal.hh>
#include <src/ops/syev/tridiag/plane_rotation.hh>
#include <src/ops/syev/tridiag/tridiag_math.hh>

namespace batchlas {

namespace {

template <typename T> struct SteqrWgFrancisKernel {};
template <typename T> struct SteqrWgRotationKernel {};

// The per-pass scratch steqr_wg_impl carves off the pool it is handed; steqr_wg_layout sizes it
// through workspace_bytes. `scan` counts each item's unreduced blocks, then holds their offsets;
// `blocks` lists them. The list is dead once written out, so its storage then holds each block's
// sweep count: blocks have len >= 2, so an item has at most n/2 of them.
struct WgPassScratch {
    Span<int32_t> scan;
    VectorView<std::array<int32_t, 2>> blocks;
    Span<int32_t> sweeps;
};

inline WgPassScratch steqr_wg_pass_layout(Queue& ctx, BumpAllocator& pool, int64_t n, int64_t batch_size) {
    auto scan = pool.allocate<int32_t>(ctx, batch_size);
    auto tdi = pool.allocate<std::array<int32_t, 2>>(
        ctx, VectorView<std::array<int32_t, 2>>::required_span_length(n / 2, 1, n / 2, batch_size));
    return {scan, VectorView<std::array<int32_t, 2>>(tdi, n / 2, batch_size, 1, n / 2),
            Span<int32_t>(reinterpret_cast<int32_t*>(tdi.data()), 2 * tdi.size())};
}

// max(|d|, |e|) over the block, in physical order.
template <typename T>
T block_norm(const OrientedBlock<T>& B) {
    T anorm = std::abs(B.D(B.len - 1));
    for (int32_t p = 0; p + 1 < B.len; ++p) {
        anorm = std::fmax(anorm, std::fmax(std::abs(B.D(p)), std::abs(B.E(p))));
    }
    return anorm;
}

// The block needs a scale iff its norm leaves [ssfmin, ssfmax].
template <typename T>
bool outside_safe_range(T anorm) {
    return anorm > detail::ssfmax<T>() || (anorm < detail::ssfmin<T>() && anorm != T(0));
}

// The undo of BlockScale::for_norm(anorm), as anorm / ssfmax (or / ssfmin), not 1 / factor.
template <typename T>
T unscale_factor(T anorm) {
    return anorm > detail::ssfmax<T>() ? anorm / detail::ssfmax<T>() : anorm / detail::ssfmin<T>();
}

template <typename T>
void scale_block(const OrientedBlock<T>& B, T alpha) {
    for (int32_t p = 0; p < B.len; ++p) B.D(p) *= alpha;
    for (int32_t p = 0; p + 1 < B.len; ++p) B.E(p) *= alpha;
}

// e_j := 0 wherever it is negligible against its neighbours; true if the block split.
template <typename T>
bool split_negligible(const OrientedBlock<T>& B) {
    bool split = false;
    for (int32_t j = 0; j + 1 < B.len; ++j) {
        if (tridiag::negligible(B.E(j), B.D(j), B.D(j + 1))) {
            B.E(j) = T(0);
            split = true;
        }
    }
    return split;
}

// Plane (k, k+1) of the chase. R annihilates (pivot, bulge); with R^T = [[c, s], [-s, c]]:
// [[d_k, e_k], [e_k, d_k+1]] <- R A R^T, e_k-1 <- c pivot - s bulge = r, e_k+1 <- c e_k+1, and
// the new bulge -s e_k+1 sits one plane down. Returns the next (pivot, bulge) = (e_k, bulge).
template <typename T>
tridiag::ChaseState<T> rotate_plane(const OrientedBlock<T>& B, int32_t k, tridiag::Givens<T> R,
                                    tridiag::ChaseState<T> st) {
    const tridiag::Givens<T> Rt = R.transpose();
    const bool has_below = k + 2 < B.len;
    const T e_below = has_below ? B.e(k + 1) : T(0);
    const tridiag::Sym2x2<T> A = tridiag::Sym2x2<T>{B.d(k), B.e(k), B.d(k + 1)}.congruence(R);
    B.d(k) = A.a;
    B.d(k + 1) = A.c;
    if (k > 0) B.e(k - 1) = st.pivot * Rt.c - st.bulge * Rt.s;
    B.e(k) = A.b;
    if (has_below) B.e(k + 1) = Rt.c * e_below;
    return {A.b, -e_below * Rt.s};
}

// One implicit QR sweep with the given shift: the first rotation brings the shift in through
// (pivot, bulge) = (d_0 - shift, e_0), and each plane down to (len-2, len-1) annihilates the
// bulge the plane above it left. `store(k, R^T)` records rotation k for Z <- Z R^T.
template <typename T, typename Store>
void implicit_qr_sweep(const OrientedBlock<T>& B, T shift, const Store& store) {
    tridiag::ChaseState<T> st{B.d(0) - shift, B.e(0)};
    for (int32_t k = 0; k + 1 < B.len; ++k) {
        const tridiag::Givens<T> R = tridiag::annihilate(st.pivot, st.bulge).R;
        store(k, R.transpose());
        st = rotate_plane(B, k, R, st);
    }
}

// One pass: split every item into unreduced blocks, run up to max_sweeps sweeps on each block
// until it splits, then replay each block's rotations on its columns of Q.
template <typename T>
Event steqr_wg_impl(Queue& ctx,
                    const VectorView<T>& d,
                    const VectorView<T>& e,
                    JobType jobz,
                    const MatrixView<T, MatrixFormat::Dense>& Q,
                    const MatrixView<std::array<T, 2>, MatrixFormat::Dense>& givens_rotations,
                    const Span<std::array<int32_t, 3>>& deflation_indices,
                    const Span<ApplyOrder>& order_view,
                    BumpAllocator allocator,
                    size_t max_sweeps) {
    BATCHLAS_KERNEL_TRACE_SCOPE("steqr_wg");
    auto n = d.size();
    auto batch_size = d.batch_size();
    bool store_givens = jobz == JobType::EigenVectors;
    auto ncus = ctx.device().get_property(DeviceProperty::MAX_COMPUTE_UNITS);
    const auto [scan_view, temp_deflation_indices, sweeps] = steqr_wg_pass_layout(ctx, allocator, n, batch_size);

    {
        BATCHLAS_KERNEL_TRACE_SCOPE("steqr_wg:deflation_ranges");
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
            batchlas::impl::parallel_for<SteqrWgFrancisKernel<T>>(cgh, sycl::nd_range(sycl::range(ncus * 128), sycl::range(64)), [=](sycl::nd_item<1> item) {
                auto i = item.get_global_id(0);
                auto n_problems = scan_view[batch_size - 1];

                for (int gid = i; gid < n_problems; gid += item.get_global_range(0)) {
                    const auto [batch_ix, start_ix, end_ix] = deflation_indices[gid];
                    const int32_t len = end_ix - start_ix;
                    const VectorView<T> D = d.batch_item(batch_ix)(Slice(start_ix, end_ix));
                    const VectorView<T> E = e.batch_item(batch_ix)(Slice(start_ix, end_ix - 1));

                    // Exact2x2: R A R^T = diag(rt1, rt2), Z <- Z R^T on (0, 1).
                    if (len == 2) {
                        if (store_givens) {
                            const auto [rt1, rt2, R] = tridiag::Sym2x2<T>{D(0), E(0), D(1)}.eig();
                            D(0) = rt1;
                            D(1) = rt2;
                            E(0) = T(0);
                            const tridiag::Givens<T> Rt = R.transpose();
                            rotations_view(0, 0, gid) = {Rt.c, Rt.s};
                            order_view[gid] = ApplyOrder::Forward;
                            sweeps[gid] = 1;
                        } else {
                            auto [rt1, rt2] = detail::eigenvalues_2x2(D(0), E(0), D(1));
                            D(0) = rt1;
                            D(1) = rt2;
                            E(0) = T(0);
                        }
                        continue;
                    }

                    // dsteqr's rule: converge the smaller |d| end first. The sweep converges the
                    // virtual end len-1, so a block whose small end is physically first is reversed.
                    const bool qr = std::abs(D(len - 1)) < std::abs(D(0));
                    const OrientedBlock<T> B{D, E, len, !qr};
                    order_view[gid] = qr ? ApplyOrder::Forward : ApplyOrder::Backward;
                    for (size_t k = 0; k < max_sweeps; ++k) {
                        const T anorm = block_norm(B);
                        const bool scaled = outside_safe_range(anorm);
                        if (scaled) scale_block(B, tridiag::BlockScale<T>::for_norm(anorm).factor);

                        const T shift = tridiag::wilkinson_shift(SteqrShiftStrategy::Wilkinson, B.d(len - 1), B.e(len - 2), B.d(len - 2));
                        implicit_qr_sweep(B, shift, [&](int32_t j, tridiag::Givens<T> Rt) {
                            if (store_givens) rotations_view(j, k, gid) = {Rt.c, Rt.s};
                        });
                        const bool split = split_negligible(B);

                        if (scaled) scale_block(B, unscale_factor(anorm));
                        if (store_givens) sweeps[gid] = static_cast<int32_t>(k + 1);
                        if (split) break;
                    }
                }
            });
        });

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

                        // The pass's sweeps in order, each Z <- Z R^T on the oriented columns (j, j+1).
                        for (int sweep = 0; sweep < sweeps[gid]; ++sweep) {
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

    // Each pass splits off at least one e_j per block, so n - 1 passes always suffice.
    for (int64_t i = 0; i < n - 1; ++i) {
        (void)steqr_wg_impl(ctx,
                      d,
                      e,
                      jobz,
                      Z,
                      givens_rotations,
                      deflation_indices,
                      apply_order,
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

template <typename T>
size_t steqr_legacy_buffer_size(Queue& ctx,
                                const VectorView<T>& d,
                                const VectorView<T>& e,
                                const VectorView<T>& eigenvalues,
                                JobType jobz,
                                SteqrParams<T> params) {
    const int64_t n = d.size();
    const int64_t batch_size = d.batch_size();
    const size_t legacy_only = jobz != JobType::EigenVectors ? 0 : workspace_bytes([&](BumpAllocator& pool) {
        const auto bs = static_cast<int64_t>(params.block_size);
        (void)pool.allocate<T>(ctx, batch_size * bs * 2 * bs * 2 + batch_size * n * bs * 4);  // apply_Q_ws
        return pool.allocate<int32_t>(ctx, batch_size * (n / 2 + 1));                       // sweep_counts
    });
    return steqr_wg_buffer_size<T>(ctx, d, e, eigenvalues, jobz, params) + legacy_only;
}

#define STEQR_WG_INSTANTIATE(back, fp) \
template Event steqr_wg<back, BATCHLAS_UNPAREN fp>(Queue&, const VectorView<BATCHLAS_UNPAREN fp>&, const VectorView<BATCHLAS_UNPAREN fp>&, const VectorView<BATCHLAS_UNPAREN fp>&, const Span<std::byte>&, JobType, SteqrParams<BATCHLAS_UNPAREN fp>, const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, Span<int32_t>);

BATCHLAS_INSTANTIATE_REAL_ALL_BACKENDS(STEQR_WG_INSTANTIATE)

#define STEQR_WG_BUFFER_SIZE_INSTANTIATE(fp) \
template size_t steqr_wg_buffer_size<BATCHLAS_UNPAREN fp>(Queue&, const VectorView<BATCHLAS_UNPAREN fp>&, const VectorView<BATCHLAS_UNPAREN fp>&, const VectorView<BATCHLAS_UNPAREN fp>&, JobType, SteqrParams<BATCHLAS_UNPAREN fp>);

BATCHLAS_FOR_EACH_REAL_TYPE(STEQR_WG_BUFFER_SIZE_INSTANTIATE)

#define STEQR_LEGACY_BUFFER_SIZE_INSTANTIATE(fp) \
template size_t steqr_legacy_buffer_size<BATCHLAS_UNPAREN fp>(Queue&, const VectorView<BATCHLAS_UNPAREN fp>&, const VectorView<BATCHLAS_UNPAREN fp>&, const VectorView<BATCHLAS_UNPAREN fp>&, JobType, SteqrParams<BATCHLAS_UNPAREN fp>);

BATCHLAS_FOR_EACH_REAL_TYPE(STEQR_LEGACY_BUFFER_SIZE_INSTANTIATE)

#undef STEQR_LEGACY_BUFFER_SIZE_INSTANTIATE

#undef STEQR_WG_BUFFER_SIZE_INSTANTIATE
#undef STEQR_WG_INSTANTIATE

} // namespace batchlas
