// SPDX-License-Identifier: MIT
// syevx_direct: partial symmetric eigensolve by full decomposition + selection;
// the universal fallback (every scalar type, range and jobz).
// evidence: docs/design/syevx.md#syevx-small-n-why-direct-is-the-answer
// evidence: docs/perf/syevx.md#syevx-routing-thresholds-as-they-stand

#include <src/linalg-impl.hh>
#include <batchlas/util/sycl-vector.hh>
#include <batchlas/util/sycl-span.hh>
#include <src/queue.hh>
#include <batchlas/util/mempool.hh>
#include <sycl/sycl.hpp>
#include <algorithm>
#include <complex>
#include <cstdint>
#include <stdexcept>
#include <batchlas/blas/linalg.hh>
#include <batchlas/backend_config.h>
#include <batchlas/blas/functions/syev.hh>
#include <src/util/template-instantiations.hh>
#include <src/util/batch_mask.hh>
#include <batchlas/device.hh>
#include <batchlas/device/lapack/syevx.hh>

namespace batchlas {

template <Backend B, typename T, MatrixFormat MFormat>
struct SyevxDirectSelectKernel;

namespace {

// Shape of the private copy of A that syev consumes. Packed (ld == n,
// stride == n*n) regardless of the input view's own layout.
template <typename T>
inline MatrixView<T, MatrixFormat::Dense> packed_copy_view(T* data,
                                                           int64_t n,
                                                           int64_t batch_size,
                                                           T** ptr_array) {
    return MatrixView<T, MatrixFormat::Dense>(data,
                                              static_cast<int>(n),
                                              static_cast<int>(n),
                                              static_cast<int>(n),
                                              n * n,
                                              static_cast<int>(batch_size),
                                              ptr_array);
}

} // namespace

template <Backend B, typename T, MatrixFormat MFormat>
Event syevx_direct(Queue& ctx,
                   const MatrixView<T, MFormat>& A,
                   Span<typename base_type<T>::type> W,
                   Span<int32_t> m,
                   size_t neigs,
                   Span<std::byte> workspace,
                   JobType jobz,
                   const MatrixView<T, MatrixFormat::Dense>& V,
                   const SyevxParams<T>& params,
                   Span<int32_t> info) {
    // Honours the mask: the copy, the public syev and the select kernel all skip an inactive item.
    const Span<const std::uint8_t> mask = ::batchlas::detail::call_mask("syevx_direct", A, V);
    using float_type = typename base_type<T>::type;

    if constexpr (MFormat != MatrixFormat::Dense) {
        (void)ctx; (void)A; (void)W; (void)m; (void)neigs; (void)workspace;
        (void)jobz; (void)V; (void)params; (void)info;
        throw batchlas::unsupported("syevx_direct: only dense matrices are supported");
    } else {
        const int64_t n = A.rows();
        const int64_t batch_size = A.batch_size();
        const bool want_eigenvectors = (jobz == JobType::EigenVectors);

        if (A.rows() != A.cols()) {
            throw batchlas::invalid_argument("syevx_direct: A must be square");
        }
        // `neigs` > n is deliberately accepted (a capacity, clamped in the resolver).
        if (!m.empty() && static_cast<int64_t>(m.size()) < batch_size) {
            throw batchlas::invalid_argument("syevx_direct: m must cover every batch item");
        }
        // Range re-checked here: this is a public entry point too. Keep wording and
        // exception type in step with validate_syevx_range_params.
        // evidence: docs/design/syevx.md#syevx-direct-the-range-is-checked-at-its-own-entry-point
        if (params.select == SyevxSelect::Index) {
            const int64_t iu = (params.iu < 0) ? (n - 1) : params.iu;
            if (params.il < 0 || iu >= n || params.il > iu) {
                throw batchlas::invalid_argument(
                    "syevx_direct: SyevxSelect::Index requires 0 <= il <= iu < n (iu < 0 means "
                    "n-1); an empty block is expressed with neigs == 0, not with il > iu");
            }
        }
        if (params.select == SyevxSelect::Value && !(params.vl < params.vu)) {
            throw batchlas::invalid_argument(
                "syevx_direct: SyevxSelect::Value requires vl < vu for the half-open interval "
                "(vl, vu]; an empty or inverted interval is almost always swapped arguments");
        }

        auto pool = BumpAllocator(workspace);
        auto a_copy_data = pool.allocate<T>(ctx, static_cast<size_t>(n * n * batch_size));
        auto a_copy_ptrs = pool.allocate<T*>(ctx, static_cast<size_t>(batch_size));
        auto lambdas = pool.allocate<float_type>(ctx, static_cast<size_t>(n * batch_size));

        auto A_copy = packed_copy_view<T>(a_copy_data.data(), n, batch_size, a_copy_ptrs.data())
                          .with_batch_mask(mask);

        // syev overwrites its input; syevx must leave A intact.
        (void)MatrixView<T, MatrixFormat::Dense>::copy(ctx, A_copy, A);

        auto syev_ws = pool.allocate<std::byte>(
            ctx, syev_buffer_size<B>(ctx, A_copy, lambdas, jobz, Uplo::Lower));
        // (void) on an Event: deliberate (in-order Queue). `info` is syev's verbatim.
        // Positional because the option overload has no `info`; Uplo::Lower must
        // match the syev_buffer_size call above.
        (void)syev<B, T>(ctx, A_copy, lambdas, jobz, Uplo::Lower, syev_ws, info);

        // syev is ascending, so every range is a contiguous block of `lambdas`.
        const auto rr = syevx_resolve_range(n, neigs, params);

        // Capacity, block start and block size are never mixed: conflating them
        // writes item b's answers into item b+1's slots.
        const int64_t capacity = static_cast<int64_t>(neigs);  // stride of W, columns of V
        const bool value_range = rr.value_range;
        const bool reverse = rr.reverse;
        const int64_t range_il = rr.il;                        // index ranges only
        const int64_t range_count = rr.max_count;              // index ranges only
        const float_type vl = params.vl;
        const float_type vu = params.vu;

        const size_t wg = std::min<size_t>(256, static_cast<size_t>(std::max<int64_t>(n, 1)));
        const auto* lam_ptr = lambdas.data();
        auto* w_ptr = W.data();
        const T* src_ptr = A_copy.data_ptr();
        T* dst_ptr = want_eigenvectors ? V.data_ptr() : nullptr;
        const int64_t dst_ld = want_eigenvectors ? V.ld() : 0;
        const int64_t dst_stride = want_eigenvectors ? V.stride() : 0;
        // V may be narrower than a capacity above n; the zeroing pass must not overrun.
        const int64_t dst_cols = want_eigenvectors
                                     ? std::min<int64_t>(static_cast<int64_t>(neigs), V.cols())
                                     : 0;
        int32_t* m_ptr = m.empty() ? nullptr : m.data();
        const std::uint8_t* mk = ::batchlas::detail::mask_ptr(mask);
        const bool masked = mk != nullptr;

        ctx->submit([&](sycl::handler& h) {
            // first index of the block, and its true (untruncated) size.
            auto block = sycl::local_accessor<int64_t, 1>(sycl::range<1>(2), h);

            batchlas::impl::parallel_for<SyevxDirectSelectKernel<B, T, MFormat>>(h,
                sycl::nd_range<1>(sycl::range{static_cast<size_t>(batch_size) * wg}, sycl::range{wg}),
                [=](sycl::nd_item<1> item) {
                    const device::RtGroup<> ex{item};
                    const int64_t bid = static_cast<int64_t>(item.get_group_linear_id());
                    if (masked && !mk[bid]) return;  // I1: uniform per work-group, before the barrier
                    const int nn = static_cast<int>(n);
                    // Truncation keeps the LOWEST `capacity`; `reverse` flips only what was kept;
                    // V's columns [write, dst_cols) are exactly zero, as on the DirectSubset path.
                    const int64_t count = device::syevx_select(
                        ex, device::vec(lam_ptr + bid * n, nn), value_range, vl, vu, range_il, range_count, capacity,
                        reverse, block, device::vec(w_ptr + bid * capacity, static_cast<int>(capacity)), dst_ptr != nullptr,
                        device::mat(src_ptr + bid * n * n, nn, nn, nn),
                        device::mat(dst_ptr + bid * dst_stride, nn, static_cast<int>(dst_cols), static_cast<int>(dst_ld)));

                    if (m_ptr != nullptr && ex.leader()) {
                        // The TRUE count: m[b] > capacity is the truncation signal.
                        m_ptr[bid] = static_cast<int32_t>(count);
                    }
                });
        });

        return ctx.get_event();
    }
}

template <Backend B, typename T, MatrixFormat MFormat>
size_t syevx_direct_buffer_size(Queue& ctx,
                                const MatrixView<T, MFormat>& A,
                                Span<typename base_type<T>::type> W,
                                size_t neigs,
                                JobType jobz,
                                const MatrixView<T, MatrixFormat::Dense>& V,
                                const SyevxParams<T>& params) {
    using float_type = typename base_type<T>::type;

    if constexpr (MFormat != MatrixFormat::Dense) {
        (void)ctx; (void)A; (void)W; (void)neigs; (void)jobz; (void)V; (void)params;
        return 0;
    } else {
        // Range-independent by construction: sized on n and batch alone.
        (void)W; (void)params; (void)neigs;
        const int64_t n = A.rows();
        const int64_t batch_size = A.batch_size();

        size_t work_size = 0;
        work_size += BumpAllocator::allocation_size<T>(ctx, static_cast<size_t>(n * n * batch_size));
        work_size += BumpAllocator::allocation_size<T*>(ctx, static_cast<size_t>(batch_size));
        work_size += BumpAllocator::allocation_size<float_type>(ctx, static_cast<size_t>(n * batch_size));

        // The masked view picks the masked syev family; only the mask's presence matters.
        const auto mask = ::batchlas::detail::call_mask("syevx_direct_buffer_size", A, V);
        auto A_copy = packed_copy_view<T>(nullptr, n, batch_size, nullptr).with_batch_mask(mask);
        work_size += BumpAllocator::allocation_size<std::byte>(
            ctx, syev_buffer_size<B>(ctx, A_copy, Span<float_type>(), jobz, Uplo::Lower));

        return work_size;
    }
}

#define SYEVX_DIRECT_INSTANTIATE(back, fp, fmt) \
    template Event syevx_direct<back, BATCHLAS_UNPAREN fp, fmt>(\
        Queue&,\
        const MatrixView<BATCHLAS_UNPAREN fp, fmt>&,\
        Span<typename base_type<BATCHLAS_UNPAREN fp>::type>,\
        Span<int32_t>,\
        size_t,\
        Span<std::byte>,\
        JobType,\
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&,\
        const SyevxParams<BATCHLAS_UNPAREN fp>&,\
        Span<int32_t>);\
    /* The m-less forwarder is inline and would not need an instantiation to be \
       CALLED, but instantiating it keeps the symbol this library exported before \
       `m` was added, so an object file built against the old header still links. */\
    template Event syevx_direct<back, BATCHLAS_UNPAREN fp, fmt>(\
        Queue&,\
        const MatrixView<BATCHLAS_UNPAREN fp, fmt>&,\
        Span<typename base_type<BATCHLAS_UNPAREN fp>::type>,\
        size_t,\
        Span<std::byte>,\
        JobType,\
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&,\
        const SyevxParams<BATCHLAS_UNPAREN fp>&,\
        Span<int32_t>);\
    template size_t syevx_direct_buffer_size<back, BATCHLAS_UNPAREN fp, fmt>(\
        Queue&,\
        const MatrixView<BATCHLAS_UNPAREN fp, fmt>&,\
        Span<typename base_type<BATCHLAS_UNPAREN fp>::type>,\
        size_t,\
        JobType,\
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&,\
        const SyevxParams<BATCHLAS_UNPAREN fp>&);

#define SYEVX_DIRECT_INSTANTIATE_FOR_BACKEND_TYPE(back, fp) \
    BATCHLAS_FOR_EACH_MATRIX_FORMAT_2(SYEVX_DIRECT_INSTANTIATE, back, fp)

    BATCHLAS_INSTANTIATE_SCALAR_ALL_BACKENDS(SYEVX_DIRECT_INSTANTIATE_FOR_BACKEND_TYPE)

#undef SYEVX_DIRECT_INSTANTIATE_FOR_BACKEND_TYPE
#undef SYEVX_DIRECT_INSTANTIATE

} // namespace batchlas
