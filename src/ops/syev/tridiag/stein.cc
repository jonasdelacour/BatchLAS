// SPDX-License-Identifier: MIT
// stein: eigenvectors of a batch of symmetric tridiagonal matrices by inverse
// iteration, given eigenvalues (from `stebz`).
//
// Phase 1: one work-item per vector (a serial tridiagonal LU per shift). Phase 2:
// modified Gram-Schmidt within clusters of close eigenvalues.
// evidence: docs/perf/syevx.md#syevx-tier-1-stebz-and-stein
// evidence: docs/design/syevx.md#syevx-for-batches-of-medium-matrices

#include <src/linalg-impl.hh>
#include <batchlas/util/group-collectives.hh>
#include <batchlas/util/sycl-vector.hh>
#include <batchlas/util/sycl-span.hh>
#include <src/queue.hh>
#include <batchlas/util/mempool.hh>
#include <sycl/sycl.hpp>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <batchlas/blas/linalg.hh>
#include <batchlas/backend_config.h>
#include <src/util/template-instantiations.hh>
#include <src/util/batch_mask.hh>
#include <batchlas/device.hh>
#include <batchlas/device/lapack/stein.hh>

namespace batchlas {

template <Backend B, typename T>
struct SteinIterateKernel;

template <Backend B, typename T>
struct SteinOrthoKernel;

namespace v2 = device;

template <Backend B, typename T>
Event stein(Queue& ctx,
            const VectorView<T>& d,
            const VectorView<T>& e,
            const VectorView<T>& w,
            size_t k_in,
            Span<const int32_t> counts,
            const MatrixView<T, MatrixFormat::Dense>& Z,
            const Span<std::byte>& ws,
            SteinParams<T> params) {
    const std::uint8_t* mask = ::batchlas::detail::mask_ptr(::batchlas::detail::call_mask("stein", d, e, w, Z));
    const int64_t n = d.size();
    const int64_t k = static_cast<int64_t>(k_in);
    const int64_t batch_size = d.batch_size();

    if (n < 1) throw batchlas::invalid_argument("stein: n must be positive");
    if (k < 1) throw batchlas::invalid_argument("stein: k must be positive");
    if (e.size() < n - 1) {
        throw batchlas::invalid_argument("stein: e must have at least n-1 entries per batch item");
    }
    if (Z.rows() < n || Z.cols() < k) {
        throw batchlas::invalid_argument("stein: Z must be at least n x k");
    }
    if (w.size() < k) throw batchlas::invalid_argument("stein: w must hold k eigenvalues");
    if (!counts.empty() && counts.size() < static_cast<size_t>(batch_size)) {
        throw batchlas::invalid_argument("stein: counts must cover every batch item");
    }

    // Null means "no per-item count": every item uses the full capacity k, which
    // is exactly the counts-less overload's behaviour.
    const int32_t* counts_ptr = counts.empty() ? nullptr : counts.data();

    auto pool = BumpAllocator(ws);
    // Five length-n scratch arrays plus the pivot flags, per (batch, vector).
    const size_t slots = static_cast<size_t>(n * k * batch_size);
    auto dl = pool.allocate<T>(ctx, slots);
    auto dd = pool.allocate<T>(ctx, slots);
    auto du = pool.allocate<T>(ctx, slots);
    auto du2 = pool.allocate<T>(ctx, slots);
    auto xb = pool.allocate<T>(ctx, slots);
    auto piv = pool.allocate<unsigned char>(ctx, slots);

    const int32_t max_iter = std::max<int32_t>(1, params.max_iterations);
    const T ortho_threshold = params.ortho_threshold;
    const uint32_t seed = params.seed;

    const size_t wg_iter = std::min<size_t>(
        std::min<size_t>(256, ctx.device().get_property(DeviceProperty::MAX_WORK_GROUP_SIZE)),
        static_cast<size_t>(std::max<int64_t>(k, 1)));

    auto* dl_ptr = dl.data();
    auto* dd_ptr = dd.data();
    auto* du_ptr = du.data();
    auto* du2_ptr = du2.data();
    auto* xb_ptr = xb.data();
    auto* piv_ptr = piv.data();
    auto Zv = Z.kernel_view();

    // Phase 1: independent inverse iteration per eigenvector.
    auto iterate_evt = ctx->submit([&](sycl::handler& h) {
        auto norm_local = sycl::local_accessor<T, 1>(sycl::range<1>(1), h);

        batchlas::impl::parallel_for<SteinIterateKernel<B, T>>(h,
            sycl::nd_range<1>(sycl::range{static_cast<size_t>(batch_size) * wg_iter},
                              sycl::range{wg_iter}),
            [=](sycl::nd_item<1> item) {
                const int64_t tid = static_cast<int64_t>(item.get_local_linear_id());
                const int64_t bid = static_cast<int64_t>(item.get_group_linear_id());
                if (mask != nullptr && mask[bid] == 0) return;  // I1: before the first collective
                const int64_t local_size = static_cast<int64_t>(item.get_local_range(0));

                // ||T||_inf, used for the pivot floor and the perturbation scale. The broadcast
                // through local memory is the legacy one (the reduction is already uniform).
                const v2::RtGroup<> ex{item};
                const T tnorm_g = v2::lanst<NormType::Inf>(ex, v2::vec(d.data_ptr() + bid * d.stride(), int(n), d.inc()),
                                                  v2::vec(e.data_ptr() + bid * e.stride(), int(n) - 1, e.inc()));
                if (ex.leader()) norm_local[0] = tnorm_g;
                ex.barrier(v2::Mem::Global);
                const T tnorm = sycl::max(norm_local[0], std::numeric_limits<T>::min());

                const T eps = std::numeric_limits<T>::epsilon();
                const T pivot_floor = eps * tnorm;

                // Valid prefix for this item. Slots [kb, k) of w are undefined --
                // for a stebz value range the count is data-dependent and per item.
                const int64_t raw = counts_ptr ? static_cast<int64_t>(counts_ptr[bid]) : k;
                const int64_t kb = (raw < 0) ? int64_t(0) : ((raw > k) ? k : raw);

                for (int64_t j = tid; j < k; j += local_size) {
                    // Zero the unused columns rather than skipping them: callers run
                    // uniform-width orthogonal back-transforms over all k columns,
                    // and zero is inert under those while uninitialized workspace is
                    // not. No group barrier in this loop, so the divergence is safe.
                    if (j >= kb) {
                        for (int64_t i = 0; i < n; ++i) Zv(i, j, bid) = T(0);
                        continue;
                    }

                    const int64_t base = (bid * k + j) * n;
                    T* my_dl = dl_ptr + base;
                    T* my_dd = dd_ptr + base;
                    T* my_du = du_ptr + base;
                    T* my_du2 = du2_ptr + base;
                    T* x = xb_ptr + base;
                    unsigned char* my_piv = piv_ptr + base;

                    // Separate exactly-degenerate eigenvalues so their shifted
                    // systems differ; reorthogonalization in phase 2 then has
                    // independent vectors to work with.
                    T lambda = w(j, bid);
                    if (j > 0) {
                        const T prev = w(j - 1, bid);
                        if (sycl::fabs(lambda - prev) < eps * tnorm) {
                            lambda += eps * tnorm * static_cast<T>(j);
                        }
                    }

                    uint32_t rng = seed + static_cast<uint32_t>(bid * 7919 + j * 104729);
                    for (int64_t i = 0; i < n; ++i) {
                        const uint32_t r = v2::stein_detail::lcg(rng);
                        x[i] = T(2) * (static_cast<T>(r >> 8) / static_cast<T>(1u << 24)) - T(1);
                    }

                    for (int32_t iter = 0; iter < max_iter; ++iter) {
                        T nrm = T(0);
                        for (int64_t i = 0; i < n; ++i) nrm += x[i] * x[i];
                        nrm = sycl::sqrt(nrm);
                        if (nrm <= T(0)) { x[0] = T(1); nrm = T(1); }
                        const T inv = T(1) / nrm;
                        for (int64_t i = 0; i < n; ++i) x[i] *= inv;

                        // Rebuild the shifted tridiagonal: the factorization
                        // overwrites it every iteration.
                        for (int64_t i = 0; i < n; ++i) my_dd[i] = d(i, bid) - lambda;
                        for (int64_t i = 0; i < n - 1; ++i) {
                            my_dl[i] = e(i, bid);
                            my_du[i] = e(i, bid);
                        }
                        for (int64_t i = 0; i < n; ++i) my_du2[i] = T(0);

                        v2::lagtf<T>(v2::Thread{}, my_dl, my_dd, my_du, my_du2, my_piv, n, pivot_floor);
                        v2::lagts<T>(v2::Thread{}, my_dl, my_dd, my_du, my_du2, my_piv, n, x);

                        // Rescale eagerly: the solve amplifies by roughly
                        // 1/|lambda - lambda_exact|, which overflows otherwise.
                        T amax = T(0);
                        for (int64_t i = 0; i < n; ++i) amax = sycl::max(amax, sycl::fabs(x[i]));
                        if (amax > T(0)) {
                            const T s = T(1) / amax;
                            for (int64_t i = 0; i < n; ++i) x[i] *= s;
                        }
                    }

                    T nrm = T(0);
                    for (int64_t i = 0; i < n; ++i) nrm += x[i] * x[i];
                    nrm = sycl::sqrt(nrm);
                    const T inv = (nrm > T(0)) ? (T(1) / nrm) : T(1);
                    for (int64_t i = 0; i < n; ++i) Zv(i, j, bid) = x[i] * inv;
                }
            });
    });

    // Phase 2: modified Gram-Schmidt within each cluster of close eigenvalues.
    const size_t wg_ortho = std::min<size_t>(
        std::min<size_t>(256, ctx.device().get_property(DeviceProperty::MAX_WORK_GROUP_SIZE)),
        static_cast<size_t>(std::max<int64_t>(n, 1)));

    ctx->submit([&](sycl::handler& h) {
        // Phase 2 reads what phase 1 wrote; do not rely on queue ordering.
        h.depends_on(iterate_evt);
        auto scratch = sycl::local_accessor<T, 1>(sycl::range<1>(1), h);

        batchlas::impl::parallel_for<SteinOrthoKernel<B, T>>(h,
            sycl::nd_range<1>(sycl::range{static_cast<size_t>(batch_size) * wg_ortho},
                              sycl::range{wg_ortho}),
            [=](sycl::nd_item<1> item) {
                const int64_t bid = static_cast<int64_t>(item.get_group_linear_id());
                if (mask != nullptr && mask[bid] == 0) return;  // I1: before the first collective
                const v2::RtGroup<> ex{item};
                const auto dv = v2::vec(d.data_ptr() + bid * d.stride(), int(n), d.inc());
                const auto ev = v2::vec(e.data_ptr() + bid * e.stride(), int(n) - 1, e.inc());
                const auto Zb = v2::mat(Zv.data() + bid * Zv.stride(), int(n), int(k), int(Zv.ld()));

                const T tnorm = sycl::max(v2::lanst<NormType::Inf>(ex, dv, ev), std::numeric_limits<T>::min());
                const T gap_tol = ortho_threshold * tnorm;

                const int64_t raw = counts_ptr ? static_cast<int64_t>(counts_ptr[bid]) : k;
                const int kb = static_cast<int>((raw < 0) ? int64_t(0) : ((raw > k) ? k : raw));

                // Walk the (ascending) eigenvalues; a gap wider than gap_tol
                // starts a new cluster. Stopping at kb, not k, is a COST bound (a
                // stale tail forms one bogus cluster); phase 1's kb bound is the
                // one that protects correctness.
                // evidence: docs/design/syevx.md#syevx-range-steins-per-item-count-bound
                int cluster_start = 0;
                for (int j = 0; j < kb; ++j) {
                    if (j > 0 && (w(j, bid) - w(j - 1, bid)) > gap_tol) cluster_start = j;
                    const auto zj = Zb.col(j);
                    // Modified Gram-Schmidt against the cluster's earlier vectors.
                    for (int i = cluster_start; i < j; ++i) {
                        const T dot = v2::dot<v2::Scope::Auto, v2::arith::Native>(ex, Zb.col(i), zj);
                        v2::axpy<v2::arith::Native>(ex, -dot, Zb.col(i), zj);
                        ex.barrier(v2::Mem::Global);
                    }
                    if (j > cluster_start) {
                        const T ss = v2::sumsq<v2::alg::Fused>(ex, zj);
                        if (ex.leader()) scratch[0] = sycl::sqrt(ss);
                        ex.barrier(v2::Mem::Global);
                        const T nrm = scratch[0];
                        if (nrm > T(0)) v2::scal<v2::arith::Native>(ex, T(1) / nrm, zj);
                        ex.barrier(v2::Mem::Global);
                    }
                }
            });
    });

    return ctx.get_event();
}

// Uniform-capacity form: every item uses all k slots. Forwards rather than
// duplicating, so there is exactly one kernel definition per (B, T).
template <Backend B, typename T>
Event stein(Queue& ctx,
            const VectorView<T>& d,
            const VectorView<T>& e,
            const VectorView<T>& w,
            size_t k_in,
            const MatrixView<T, MatrixFormat::Dense>& Z,
            const Span<std::byte>& ws,
            SteinParams<T> params) {
    return stein<B, T>(ctx, d, e, w, k_in, Span<const int32_t>(), Z, ws, params);
}

template <Backend B, typename T>
size_t stein_buffer_size(Queue& ctx, size_t n, size_t k, size_t batch_size, SteinParams<T> params) {
    (void)params;
    const size_t slots = n * k * batch_size;
    size_t bytes = 0;
    bytes += BumpAllocator::allocation_size<T>(ctx, slots) * 5;
    bytes += BumpAllocator::allocation_size<unsigned char>(ctx, slots);
    return bytes;
}

#define STEIN_INSTANTIATE(back, fp) \
    template Event stein<back, BATCHLAS_UNPAREN fp>(Queue&, \
        const VectorView<BATCHLAS_UNPAREN fp>&, \
        const VectorView<BATCHLAS_UNPAREN fp>&, \
        const VectorView<BATCHLAS_UNPAREN fp>&, \
        size_t, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        const Span<std::byte>&, \
        SteinParams<BATCHLAS_UNPAREN fp>); \
    template Event stein<back, BATCHLAS_UNPAREN fp>(Queue&, \
        const VectorView<BATCHLAS_UNPAREN fp>&, \
        const VectorView<BATCHLAS_UNPAREN fp>&, \
        const VectorView<BATCHLAS_UNPAREN fp>&, \
        size_t, \
        Span<const int32_t>, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        const Span<std::byte>&, \
        SteinParams<BATCHLAS_UNPAREN fp>); \
    template size_t stein_buffer_size<back, BATCHLAS_UNPAREN fp>(Queue&, size_t, size_t, size_t, SteinParams<BATCHLAS_UNPAREN fp>);

    BATCHLAS_INSTANTIATE_REAL_ALL_BACKENDS(STEIN_INSTANTIATE)

#undef STEIN_INSTANTIATE

} // namespace batchlas
