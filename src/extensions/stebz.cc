// stebz: selected eigenvalues of a batch of symmetric tridiagonal matrices, by
// bisection on Sturm sequence sign counts.
//
// This is the tridiagonal kernel for SYEVX (SYEVX_PLAN.md Tier 1). Bisection is
// used rather than MRRR because in the subset regime its only real weakness --
// O(nk^2) reorthogonalization of clustered eigenvectors, handled in `stein` --
// is negligible against the O(n^3) tridiagonalization that precedes it, while
// MRRR's implementation cost is very large. See SYEVX_PLAN.md §6.1-6.2.
//
// Parallelization: one work-group per batch item, one work-item per wanted
// eigenvalue. Eigenvalues are mutually independent, so no communication is needed
// beyond the initial reduction for the Gershgorin bounds.

#include "../linalg-impl.hh"
#include <batchlas/util/sycl-vector.hh>
#include <batchlas/util/sycl-span.hh>
#include "../queue.hh"
#include <batchlas/util/mempool.hh>
#include <sycl/sycl.hpp>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <batchlas/blas/linalg.hh>
#include <batchlas/backend_config.h>
#include "../util/template-instantiations.hh"

namespace batchlas {

template <Backend B, typename T>
struct StebzKernel;

namespace stebz_detail {

// Number of eigenvalues <= x, from the sign changes of the Sturm sequence.
//
// Follows LAPACK dlaebz: the recurrence q_i = d_i - x - e_{i-1}^2 / q_{i-1} is
// guarded by clamping |q| below pivmin, which keeps the count monotone in x even
// when a pivot underflows. Without the guard a zero pivot produces an infinity and
// the count becomes non-monotone, which breaks the bisection invariant.
template <typename T, typename DView, typename EView>
inline int64_t sturm_count(const DView& d,
                           const EView& e,
                           int64_t n,
                           int64_t b,
                           T x,
                           T pivmin) {
    int64_t count = 0;
    T q = d(0, b) - x;
    if (sycl::fabs(q) < pivmin) q = -pivmin;
    if (q <= T(0)) ++count;

    for (int64_t i = 1; i < n; ++i) {
        const T ei = e(i - 1, b);
        q = d(i, b) - x - (ei * ei) / q;
        if (sycl::fabs(q) < pivmin) q = -pivmin;
        if (q <= T(0)) ++count;
    }
    return count;
}

} // namespace stebz_detail

template <Backend B, typename T>
Event stebz(Queue& ctx,
            const VectorView<T>& d,
            const VectorView<T>& e,
            const VectorView<T>& w,
            Span<int32_t> m,
            const Span<std::byte>& ws,
            StebzParams<T> params) {
    const int64_t n = d.size();
    const int64_t batch_size = d.batch_size();

    if (n < 1) throw std::runtime_error("stebz: n must be positive");
    if (e.size() < n - 1) {
        throw std::runtime_error("stebz: e must have at least n-1 entries per batch item");
    }
    if (static_cast<int64_t>(m.size()) < batch_size) {
        throw std::runtime_error("stebz: m must cover every batch item");
    }

    // Resolve the requested index range. For Value ranges the count is
    // data-dependent and is determined on the device.
    int64_t il = 0;
    int64_t iu = n - 1;
    if (params.range == EigenRangeType::Index) {
        il = params.il;
        iu = (params.iu < 0) ? (n - 1) : params.iu;
        if (il < 0 || iu >= n || il > iu) {
            throw std::runtime_error("stebz: invalid index range");
        }
    }
    const bool value_range = (params.range == EigenRangeType::Value);
    const int64_t max_wanted = value_range ? n : (iu - il + 1);

    if (w.size() < max_wanted) {
        throw std::runtime_error("stebz: w is too small for the requested range");
    }

    (void)ws; // Bisection needs no scratch: each work-item's state is in registers.

    const T vl = params.vl;
    const T vu = params.vu;
    const T abstol = params.abstol;
    const int32_t max_iter = std::max<int32_t>(1, params.max_iterations);
    const bool descending = (params.order == SortOrder::Descending);

    const size_t wg = std::min<size_t>(
        std::min<size_t>(256, ctx.device().get_property(DeviceProperty::MAX_WORK_GROUP_SIZE)),
        static_cast<size_t>(std::max<int64_t>(max_wanted, 1)));

    auto* m_ptr = m.data();

    ctx->submit([&](sycl::handler& h) {
        auto bounds = sycl::local_accessor<T, 1>(sycl::range<1>(3), h);   // lo, hi, pivmin
        auto found = sycl::local_accessor<int32_t, 1>(sycl::range<1>(2), h); // count, first index

        h.parallel_for<StebzKernel<B, T>>(
            sycl::nd_range<1>(sycl::range{static_cast<size_t>(batch_size) * wg}, sycl::range{wg}),
            [=](sycl::nd_item<1> item) {
                const int64_t tid = static_cast<int64_t>(item.get_local_linear_id());
                const int64_t bid = static_cast<int64_t>(item.get_group_linear_id());
                const int64_t local_size = static_cast<int64_t>(item.get_local_range(0));
                auto cta = item.get_group();

                // Gershgorin bounds: every eigenvalue lies in
                // [min_i (d_i - r_i), max_i (d_i + r_i)] with r_i = |e_{i-1}| + |e_i|.
                T lo_partial = std::numeric_limits<T>::max();
                T hi_partial = -std::numeric_limits<T>::max();
                T e2_max_partial = T(0);
                for (int64_t i = tid; i < n; i += local_size) {
                    const T left = (i > 0) ? sycl::fabs(e(i - 1, bid)) : T(0);
                    const T right = (i < n - 1) ? sycl::fabs(e(i, bid)) : T(0);
                    const T radius = left + right;
                    lo_partial = sycl::min(lo_partial, d(i, bid) - radius);
                    hi_partial = sycl::max(hi_partial, d(i, bid) + radius);
                    e2_max_partial = sycl::max(e2_max_partial, right * right);
                }

                const T lo_all = sycl::reduce_over_group(cta, lo_partial, sycl::minimum<T>());
                const T hi_all = sycl::reduce_over_group(cta, hi_partial, sycl::maximum<T>());
                const T e2_max = sycl::reduce_over_group(cta, e2_max_partial, sycl::maximum<T>());

                if (tid == 0) {
                    const T eps = std::numeric_limits<T>::epsilon();
                    const T norm = sycl::max(sycl::fabs(lo_all), sycl::fabs(hi_all));
                    // Widen slightly so that count(lo) == 0 and count(hi) == n
                    // strictly, which the bisection invariant relies on.
                    const T pad = eps * norm * T(2) + std::numeric_limits<T>::min();
                    bounds[0] = lo_all - pad;
                    bounds[1] = hi_all + pad;
                    bounds[2] = std::numeric_limits<T>::min() * sycl::max(T(1), e2_max);
                    found[0] = 0;
                    found[1] = 0;
                }
                sycl::group_barrier(cta);

                const T glo = bounds[0];
                const T ghi = bounds[1];
                const T pivmin = bounds[2];
                const T eps = std::numeric_limits<T>::epsilon();
                const T norm = sycl::max(sycl::fabs(glo), sycl::fabs(ghi));
                const T tol = (abstol > T(0)) ? abstol : (eps * sycl::max(norm, T(1)));

                // For a value range, first convert (vl, vu] into an index range:
                // the eigenvalues in the interval are exactly those with index in
                // [count(vl), count(vu)-1].
                int64_t first = 0;
                int64_t last = -1;
                if (value_range) {
                    if (tid == 0) {
                        const int64_t c_lo = stebz_detail::sturm_count<T>(d, e, n, bid, vl, pivmin);
                        const int64_t c_hi = stebz_detail::sturm_count<T>(d, e, n, bid, vu, pivmin);
                        found[0] = static_cast<int32_t>(c_hi - c_lo);
                        found[1] = static_cast<int32_t>(c_lo);
                    }
                    sycl::group_barrier(cta);
                    first = static_cast<int64_t>(found[1]);
                    last = first + static_cast<int64_t>(found[0]) - 1;
                } else {
                    first = il;
                    last = iu;
                    if (tid == 0) found[0] = static_cast<int32_t>(last - first + 1);
                }

                const int64_t count_wanted = last - first + 1;

                // One work-item per wanted eigenvalue; no interaction between them.
                for (int64_t slot = tid; slot < count_wanted; slot += local_size) {
                    const int64_t j = first + slot;

                    // Invariant: count(left) <= j < count(right), so the j-th
                    // eigenvalue (0-based, ascending) lies in (left, right].
                    T left = glo;
                    T right = ghi;
                    for (int32_t iter = 0; iter < max_iter; ++iter) {
                        if ((right - left) <= tol) break;
                        const T mid = left + (right - left) * T(0.5);
                        // Guard against the midpoint not advancing at the limit of
                        // floating-point resolution.
                        if (mid <= left || mid >= right) break;
                        if (stebz_detail::sturm_count<T>(d, e, n, bid, mid, pivmin) >= j + 1) {
                            right = mid;
                        } else {
                            left = mid;
                        }
                    }

                    const T lambda = left + (right - left) * T(0.5);
                    const int64_t out = descending ? (count_wanted - 1 - slot) : slot;
                    w(out, bid) = lambda;
                }

                sycl::group_barrier(cta);
                if (tid == 0) {
                    m_ptr[bid] = found[0];
                }
            });
    });

    return ctx.get_event();
}

template <Backend B, typename T>
size_t stebz_buffer_size(Queue& ctx, size_t n, size_t batch_size, StebzParams<T> params) {
    // Bisection carries its per-eigenvalue state in registers and needs only a
    // handful of work-group locals, so no global scratch is required.
    (void)ctx; (void)n; (void)batch_size; (void)params;
    return 0;
}

#define STEBZ_INSTANTIATE(back, fp) \
    template Event stebz<back, BATCHLAS_UNPAREN fp>(Queue&, \
        const VectorView<BATCHLAS_UNPAREN fp>&, \
        const VectorView<BATCHLAS_UNPAREN fp>&, \
        const VectorView<BATCHLAS_UNPAREN fp>&, \
        Span<int32_t>, \
        const Span<std::byte>&, \
        StebzParams<BATCHLAS_UNPAREN fp>); \
    template size_t stebz_buffer_size<back, BATCHLAS_UNPAREN fp>(Queue&, size_t, size_t, StebzParams<BATCHLAS_UNPAREN fp>);

#define STEBZ_INSTANTIATE_FOR_BACKEND(back) \
    BATCHLAS_FOR_EACH_REAL_TYPE_1(STEBZ_INSTANTIATE, back)

#if BATCHLAS_HAS_HOST_BACKEND
    STEBZ_INSTANTIATE_FOR_BACKEND(Backend::NETLIB)
#endif
#if BATCHLAS_HAS_CUDA_BACKEND
    STEBZ_INSTANTIATE_FOR_BACKEND(Backend::CUDA)
#endif
#if BATCHLAS_HAS_ROCM_BACKEND
    STEBZ_INSTANTIATE_FOR_BACKEND(Backend::ROCM)
#endif

#undef STEBZ_INSTANTIATE_FOR_BACKEND
#undef STEBZ_INSTANTIATE

} // namespace batchlas
