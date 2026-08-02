// stein: eigenvectors of a batch of symmetric tridiagonal matrices by inverse
// iteration, given eigenvalues (from `stebz`).
//
// Companion to stebz for SYEVX Tier 1 (SYEVX_PLAN.md §8). Two phases:
//
//   1. One work-item per wanted eigenvector solves (T - lambda*I) x = b a few
//      times from a pseudo-random start, using a tridiagonal LU factorization
//      with partial pivoting (LAPACK dgttrf/dgttrs). The factorization depends on
//      lambda, so it is per-vector and inherently serial in n -- hence one
//      work-item rather than one work-group per vector. This is affordable
//      because for medium n the tridiagonal stage is far off the critical path
//      (SYEVX_PLAN.md §4).
//
//   2. Vectors whose eigenvalues form a cluster are reorthogonalized against each
//      other by modified Gram-Schmidt. Inverse iteration alone does not deliver
//      orthogonality on clusters; this is the price bisection pays relative to
//      MRRR, and it costs O(n*k^2) only within clusters.

#include "../linalg-impl.hh"
#include <util/sycl-vector.hh>
#include <util/sycl-span.hh>
#include "../queue.hh"
#include <util/mempool.hh>
#include <sycl/sycl.hpp>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <blas/linalg.hh>
#include <batchlas/backend_config.h>
#include "../util/template-instantiations.hh"

namespace batchlas {

template <Backend B, typename T>
struct SteinIterateKernel;

template <Backend B, typename T>
struct SteinOrthoKernel;

namespace stein_detail {

// Tridiagonal LU with partial pivoting, following LAPACK dgttrf.
//
// On entry dd holds the shifted diagonal and dl/du the off-diagonals; on exit dl
// holds the elimination multipliers, dd/du/du2 the three diagonals of U, and
// swapped[i] records whether rows i and i+1 were interchanged. Pivots smaller
// than `tol` are replaced by `tol`: the system is deliberately near-singular
// (that is the point of inverse iteration), so an exactly zero pivot is expected
// rather than exceptional.
template <typename T>
inline void tridiag_lu(T* dl, T* dd, T* du, T* du2, unsigned char* swapped,
                       int64_t n, T tol) {
    for (int64_t i = 0; i < n - 1; ++i) {
        if (sycl::fabs(dd[i]) >= sycl::fabs(dl[i])) {
            swapped[i] = 0;
            if (sycl::fabs(dd[i]) < tol) dd[i] = (dd[i] < T(0)) ? -tol : tol;
            const T fact = dl[i] / dd[i];
            dl[i] = fact;
            dd[i + 1] -= fact * du[i];
            if (i < n - 2) du2[i] = T(0);
        } else {
            swapped[i] = 1;
            const T fact = dd[i] / dl[i];
            dd[i] = dl[i];
            dl[i] = fact;
            const T tmp = du[i];
            du[i] = dd[i + 1];
            dd[i + 1] = tmp - fact * dd[i + 1];
            if (i < n - 2) {
                du2[i] = du[i + 1];
                du[i + 1] = -fact * du[i + 1];
            }
        }
    }
    if (sycl::fabs(dd[n - 1]) < tol) dd[n - 1] = (dd[n - 1] < T(0)) ? -tol : tol;
}

// Solves L*U*x = b in place, following LAPACK dgttrs (no transpose).
template <typename T>
inline void tridiag_solve(const T* dl, const T* dd, const T* du, const T* du2,
                          const unsigned char* swapped, int64_t n, T* b) {
    for (int64_t i = 0; i < n - 1; ++i) {
        if (!swapped[i]) {
            b[i + 1] -= dl[i] * b[i];
        } else {
            const T tmp = b[i];
            b[i] = b[i + 1];
            b[i + 1] = tmp - dl[i] * b[i];
        }
    }
    b[n - 1] = b[n - 1] / dd[n - 1];
    if (n > 1) b[n - 2] = (b[n - 2] - du[n - 2] * b[n - 1]) / dd[n - 2];
    for (int64_t i = n - 3; i >= 0; --i) {
        b[i] = (b[i] - du[i] * b[i + 1] - du2[i] * b[i + 2]) / dd[i];
    }
}

// Deterministic per-(batch, vector) start values in [-1, 1]. A fixed sequence
// keeps results reproducible across runs, which matters for testing.
inline uint32_t lcg(uint32_t& state) {
    state = state * 1664525u + 1013904223u;
    return state;
}

} // namespace stein_detail

template <Backend B, typename T>
Event stein(Queue& ctx,
            const VectorView<T>& d,
            const VectorView<T>& e,
            const VectorView<T>& w,
            size_t k_in,
            const MatrixView<T, MatrixFormat::Dense>& Z,
            const Span<std::byte>& ws,
            SteinParams<T> params) {
    const int64_t n = d.size();
    const int64_t k = static_cast<int64_t>(k_in);
    const int64_t batch_size = d.batch_size();

    if (n < 1) throw std::runtime_error("stein: n must be positive");
    if (k < 1) throw std::runtime_error("stein: k must be positive");
    if (e.size() < n - 1) {
        throw std::runtime_error("stein: e must have at least n-1 entries per batch item");
    }
    if (Z.rows() < n || Z.cols() < k) {
        throw std::runtime_error("stein: Z must be at least n x k");
    }
    if (w.size() < k) throw std::runtime_error("stein: w must hold k eigenvalues");

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

        h.parallel_for<SteinIterateKernel<B, T>>(
            sycl::nd_range<1>(sycl::range{static_cast<size_t>(batch_size) * wg_iter},
                              sycl::range{wg_iter}),
            [=](sycl::nd_item<1> item) {
                const int64_t tid = static_cast<int64_t>(item.get_local_linear_id());
                const int64_t bid = static_cast<int64_t>(item.get_group_linear_id());
                const int64_t local_size = static_cast<int64_t>(item.get_local_range(0));
                auto cta = item.get_group();

                // ||T||_inf, used for the pivot floor and the perturbation scale.
                T norm_partial = T(0);
                for (int64_t i = tid; i < n; i += local_size) {
                    const T left = (i > 0) ? sycl::fabs(e(i - 1, bid)) : T(0);
                    const T right = (i < n - 1) ? sycl::fabs(e(i, bid)) : T(0);
                    norm_partial = sycl::max(norm_partial, sycl::fabs(d(i, bid)) + left + right);
                }
                const T tnorm_g = sycl::reduce_over_group(cta, norm_partial, sycl::maximum<T>());
                if (tid == 0) norm_local[0] = tnorm_g;
                sycl::group_barrier(cta);
                const T tnorm = sycl::max(norm_local[0], std::numeric_limits<T>::min());

                const T eps = std::numeric_limits<T>::epsilon();
                const T pivot_floor = eps * tnorm;

                for (int64_t j = tid; j < k; j += local_size) {
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
                        const uint32_t r = stein_detail::lcg(rng);
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

                        stein_detail::tridiag_lu<T>(my_dl, my_dd, my_du, my_du2, my_piv, n, pivot_floor);
                        stein_detail::tridiag_solve<T>(my_dl, my_dd, my_du, my_du2, my_piv, n, x);

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

        h.parallel_for<SteinOrthoKernel<B, T>>(
            sycl::nd_range<1>(sycl::range{static_cast<size_t>(batch_size) * wg_ortho},
                              sycl::range{wg_ortho}),
            [=](sycl::nd_item<1> item) {
                const int64_t tid = static_cast<int64_t>(item.get_local_linear_id());
                const int64_t bid = static_cast<int64_t>(item.get_group_linear_id());
                const int64_t local_size = static_cast<int64_t>(item.get_local_range(0));
                auto cta = item.get_group();

                T norm_partial = T(0);
                for (int64_t i = tid; i < n; i += local_size) {
                    const T left = (i > 0) ? sycl::fabs(e(i - 1, bid)) : T(0);
                    const T right = (i < n - 1) ? sycl::fabs(e(i, bid)) : T(0);
                    norm_partial = sycl::max(norm_partial, sycl::fabs(d(i, bid)) + left + right);
                }
                const T tnorm = sycl::max(
                    sycl::reduce_over_group(cta, norm_partial, sycl::maximum<T>()),
                    std::numeric_limits<T>::min());
                const T gap_tol = ortho_threshold * tnorm;

                // Walk the (ascending) eigenvalues; a gap wider than gap_tol
                // starts a new cluster. Only within-cluster pairs need work.
                int64_t cluster_start = 0;
                for (int64_t j = 0; j < k; ++j) {
                    if (j > 0 && (w(j, bid) - w(j - 1, bid)) > gap_tol) {
                        cluster_start = j;
                    }
                    for (int64_t i = cluster_start; i < j; ++i) {
                        T dot_partial = T(0);
                        for (int64_t r = tid; r < n; r += local_size) {
                            dot_partial += Zv(r, i, bid) * Zv(r, j, bid);
                        }
                        const T dot = sycl::reduce_over_group(cta, dot_partial, sycl::plus<T>());
                        for (int64_t r = tid; r < n; r += local_size) {
                            Zv(r, j, bid) -= dot * Zv(r, i, bid);
                        }
                        sycl::group_barrier(cta);
                    }

                    if (j > cluster_start) {
                        T nrm_partial = T(0);
                        for (int64_t r = tid; r < n; r += local_size) {
                            const T v = Zv(r, j, bid);
                            nrm_partial += v * v;
                        }
                        const T nrm2 = sycl::reduce_over_group(cta, nrm_partial, sycl::plus<T>());
                        if (tid == 0) scratch[0] = sycl::sqrt(nrm2);
                        sycl::group_barrier(cta);
                        const T nrm = scratch[0];
                        if (nrm > T(0)) {
                            const T inv = T(1) / nrm;
                            for (int64_t r = tid; r < n; r += local_size) Zv(r, j, bid) *= inv;
                        }
                        sycl::group_barrier(cta);
                    }
                }
            });
    });

    return ctx.get_event();
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
    template size_t stein_buffer_size<back, BATCHLAS_UNPAREN fp>(Queue&, size_t, size_t, size_t, SteinParams<BATCHLAS_UNPAREN fp>);

#define STEIN_INSTANTIATE_FOR_BACKEND(back) \
    BATCHLAS_FOR_EACH_REAL_TYPE_1(STEIN_INSTANTIATE, back)

#if BATCHLAS_HAS_HOST_BACKEND
    STEIN_INSTANTIATE_FOR_BACKEND(Backend::NETLIB)
#endif
#if BATCHLAS_HAS_CUDA_BACKEND
    STEIN_INSTANTIATE_FOR_BACKEND(Backend::CUDA)
#endif
#if BATCHLAS_HAS_ROCM_BACKEND
    STEIN_INSTANTIATE_FOR_BACKEND(Backend::ROCM)
#endif

#undef STEIN_INSTANTIATE_FOR_BACKEND
#undef STEIN_INSTANTIATE

} // namespace batchlas
