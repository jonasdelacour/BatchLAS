// SPDX-License-Identifier: MIT
// stebz: selected eigenvalues of a batch of symmetric tridiagonal matrices, by
// bisection on Sturm sequence sign counts.
//
// The tridiagonal kernel of syevx's subset paths; bisection rather than MRRR.
// evidence: docs/design/syevx.md#bisection-and-inverse-iteration-stebz-and-stein-build
//
// Parallelization: one work-group per batch item, one work-item per wanted
// eigenvalue. Eigenvalues are mutually independent, so no communication is needed
// beyond the initial reduction for the Gershgorin bounds.

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
#include <batchlas/device/lapack/stebz.hh>

namespace batchlas {

template <Backend B, typename T>
struct StebzKernel;


template <Backend B, typename T>
Event stebz(Queue& ctx,
            const VectorView<T>& d,
            const VectorView<T>& e,
            const VectorView<T>& w,
            Span<int32_t> m,
            const Span<std::byte>& ws,
            StebzParams<T> params) {
    const std::uint8_t* mask = ::batchlas::detail::mask_ptr(::batchlas::detail::call_mask("stebz", d, e, w));
    const int64_t n = d.size();
    const int64_t batch_size = d.batch_size();

    if (n < 1) throw batchlas::invalid_argument("stebz: n must be positive");
    if (e.size() < n - 1) {
        throw batchlas::invalid_argument("stebz: e must have at least n-1 entries per batch item");
    }
    if (static_cast<int64_t>(m.size()) < batch_size) {
        throw batchlas::invalid_argument("stebz: m must cover every batch item");
    }

    // Resolve the requested index range. For Value ranges the count is
    // data-dependent and is determined on the device.
    int64_t il = 0;
    int64_t iu = n - 1;
    if (params.range == EigenRangeType::Index) {
        il = params.il;
        iu = (params.iu < 0) ? (n - 1) : params.iu;
        if (il < 0 || iu >= n || il > iu) {
            throw batchlas::invalid_argument("stebz: invalid index range");
        }
    }
    const bool value_range = (params.range == EigenRangeType::Value);
    const int64_t max_wanted = value_range ? n : (iu - il + 1);

    if (w.size() < max_wanted) {
        throw batchlas::invalid_argument("stebz: w is too small for the requested range");
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
    using Tier = device::tier::Blocked<1>;
    const device::WorkspaceSize wsz = device::stebz_buffer_size<Tier, T>(static_cast<int>(n));

    // Thin launcher (design C.8): the body is the public device::stebz.
    ctx->submit([&](sycl::handler& h) {
        auto slm = sycl::local_accessor<std::byte, 1>(sycl::range<1>(wsz.local_bytes), h);

        batchlas::impl::parallel_for<StebzKernel<B, T>>(h,
            sycl::nd_range<1>(sycl::range{static_cast<size_t>(batch_size) * wg}, sycl::range{wg}),
            [=](sycl::nd_item<1> item) {
                const int64_t bid = static_cast<int64_t>(item.get_group_linear_id());
                if (mask != nullptr && mask[bid] == 0) return;  // I1: before the first collective
                const device::RtGroup<> ex{item};
                const auto dv = device::vec(d.data_ptr() + bid * d.stride(), int(n), d.inc());
                const auto ev = device::vec(e.data_ptr() + bid * e.stride(), int(n) - 1, e.inc());
                const device::stebz_detail::Params<T> p{value_range, il, iu, vl, vu, abstol, descending, max_iter};
                int found = 0;
                device::stebz<Tier>(ex, dv, ev, device::item(w, static_cast<int>(bid)), found,
                                device::Workspace{slm.template get_multi_ptr<sycl::access::decorated::yes>(), nullptr}, p);
                if (ex.leader()) m_ptr[bid] = found;
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

    BATCHLAS_INSTANTIATE_REAL_ALL_BACKENDS(STEBZ_INSTANTIATE)

#undef STEBZ_INSTANTIATE

} // namespace batchlas
