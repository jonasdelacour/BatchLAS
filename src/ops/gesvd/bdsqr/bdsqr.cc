// SPDX-License-Identifier: MIT
#include <batchlas/blas/extensions.hh>
#include <batchlas/backend_config.h>

#include <batchlas/util/mempool.hh>

#include <batchlas/device/interop.hh>
#include <batchlas/device/lapack/bdsqr.hh>
#include <src/math-helpers.hh>
#include <src/queue.hh>
#include <src/util/template-instantiations.hh>
#include <src/kernels/common/info_span.hh>
#include <src/util/batch_mask.hh>

#include <algorithm>
#include <stdexcept>

namespace batchlas {

namespace {


template <typename Real>
bool bdsqr_implicit_qr_attempt(Queue& ctx,
                               const VectorView<Real>& d,
                               const VectorView<Real>& e,
                               Span<Real> singular_values_out,
                               const MatrixView<Real, MatrixFormat::Dense>& u,
                               const MatrixView<Real, MatrixFormat::Dense>& vh,
                               int32_t n,
                               int32_t batch,
                               bool sort_desc,
                               Span<int32_t> fail_flags,
                               const std::uint8_t* mask) {
    // DBDSQR's tolerance and iteration cap, formed on the host.
    const auto params = device::bdsqr_detail::params<Real>(n);

    ctx->submit([&](sycl::handler& cgh) {
        auto D = d;
        auto E = e;
        Real* out = singular_values_out.data();
        auto U = u.kernel_view();
        auto Vh = vh.kernel_view();
        int32_t* fail = fail_flags.data();
        const int32_t nn = n;
        const int32_t nb = batch;
        const bool descending = sort_desc;
        const int32_t urows = static_cast<int32_t>(u.rows()), ucols = static_cast<int32_t>(u.cols());
        const int32_t vrows = static_cast<int32_t>(vh.rows()), vcols = static_cast<int32_t>(vh.cols());

        cgh.parallel_for(sycl::range<1>(static_cast<size_t>(nb)), [=](sycl::id<1> tid) {
            const int32_t b = static_cast<int32_t>(tid[0]);
            // I3. The host fold below reads every flag, so an inactive item writes the
            // neutral "converged" and nothing else.
            if (mask != nullptr && mask[b] == 0) {
                fail[b] = 0;
                return;
            }

            Real* db = D.data_ptr() + static_cast<size_t>(b) * static_cast<size_t>(D.stride());
            Real* eb = (nn > 1) ? (E.data_ptr() + static_cast<size_t>(b) * static_cast<size_t>(E.stride())) : nullptr;
            Real* sb = out + static_cast<size_t>(b) * static_cast<size_t>(nn);
            // device::bdsqr on one item (lapack/bdsqr.hh): U and Vh addressed as the host view's (i, j, b).
            fail[b] = device::bdsqr<device::tier::Blocked<1>>(device::Thread{}, device::vec(db, nn), device::vec(eb, nn > 1 ? nn - 1 : 0), device::vec(sb, nn),
                                                      device::Workspace{}, device::item(U, b, urows, ucols), device::item(Vh, b, vrows, vcols),
                                                      descending, params);
        });
    });

    ctx.wait();
    bool ok = true;
    for (int32_t b = 0; b < batch; ++b) {
        if (fail_flags[static_cast<size_t>(b)] != 0) {
            ok = false;
            break;
        }
    }
    return ok;
}

} // namespace

// Single description of bdsqr's workspace; see workspace_bytes() in
// util/mempool.hh.
template <typename T>
Span<int32_t> bdsqr_layout(Queue& ctx, BumpAllocator& pool, int32_t batch) {
    return pool.allocate<int32_t>(ctx, static_cast<size_t>(batch));
}

template <Backend B, typename T>
Event bdsqr(Queue& ctx,
            const VectorView<T>& d,
            const VectorView<T>& e,
            Span<T> singular_values_out,
            const Span<std::byte>& ws,
            bool sort_desc,
            Span<int32_t> info) {
    const auto empty = MatrixView<T, MatrixFormat::Dense>(nullptr, 0, 0, 1, 1, d.batch_size());
    return bdsqr<B, T>(ctx,
                       d,
                       e,
                       singular_values_out,
                       ws,
                       empty,
                       empty,
                       sort_desc,
                       info);
}

template <Backend B, typename T>
Event bdsqr(Queue& ctx,
            const VectorView<T>& d,
            const VectorView<T>& e,
            Span<T> singular_values_out,
            const Span<std::byte>& ws,
            const MatrixView<T, MatrixFormat::Dense>& u,
            const MatrixView<T, MatrixFormat::Dense>& vh,
            bool sort_desc,
            Span<int32_t> info) {
    const auto mask = ::batchlas::detail::call_mask("bdsqr", d, e, u, vh);
    static_cast<void>(B);

    const int32_t n = static_cast<int32_t>(d.size());
    const int32_t batch = static_cast<int32_t>(d.batch_size());

    if (batch < 1 || n < 1) {
        throw batchlas::invalid_argument("bdsqr: invalid dimensions");
    }
    if (e.size() != std::max<int32_t>(0, n - 1) || e.batch_size() != batch) {
        throw batchlas::invalid_argument("bdsqr: e must have length n-1 and matching batch size");
    }
    const size_t need_s = static_cast<size_t>(n) * static_cast<size_t>(batch);
    if (singular_values_out.size() < need_s) {
        throw batchlas::invalid_argument("bdsqr: singular_values_out span too small");
    }
    if (u.rows() > 0 || u.cols() > 0) {
        if (u.cols() != n || u.batch_size() != batch) {
            throw batchlas::invalid_argument("bdsqr: U must have n columns and matching batch size");
        }
    }
    if (vh.rows() > 0 || vh.cols() > 0) {
        if (vh.rows() != n || vh.batch_size() != batch) {
            throw batchlas::invalid_argument("bdsqr: Vh must have n rows and matching batch size");
        }
    }

    if constexpr (detail::is_complex<T>::value) {
        throw batchlas::unsupported("bdsqr: complex types are not implemented yet");
    } else {
        Span<std::byte> ws_mut(const_cast<std::byte*>(ws.data()), ws.size());
        BumpAllocator pool(ws_mut);
        // fail_flags stays a pool draw and stays UNCONDITIONAL: it is the whole of
        // bdsqr's workspace (see bdsqr_layout / bdsqr_buffer_size), the kernel
        // STORES into it rather than raising, and `info` is an accumulator that
        // several producers may share. Folding afterwards keeps both contracts and
        // leaves bdsqr_buffer_size untouched.
        auto fail_flags = bdsqr_layout<T>(ctx, pool, batch);
        const bool ok = bdsqr_implicit_qr_attempt<T>(ctx,
                                                     d,
                                                     e,
                                                     singular_values_out,
                                                     u,
                                                     vh,
                                                     n,
                                                     batch,
                                                     sort_desc,
                                                     fail_flags,
                                                     ::batchlas::detail::mask_ptr(mask));
        // 0 already means converged here, so no polarity flip -- unlike
        // syevx_lobpcg / syevx_filtered, whose flags mean the opposite.
        detail::info_from_flags(ctx, info, fail_flags.data(), batch, /*one_means_converged=*/false, mask);
        // The batch-wide throw is what a caller who asked for NO per-item status
        // still needs: without it a non-converged item is silent again. A caller who
        // supplied `info` gets the per-item answer instead of an exception that says
        // nothing about which item failed.
        //
        // The condition is info_ptr, not info.empty(): a SHORT non-empty span is
        // "not requested" everywhere else in this mechanism, and testing empty()
        // here would drop both the span AND the throw for one.
        if (!ok && detail::info_ptr(info, batch) == nullptr) {
            throw batchlas::convergence_error("bdsqr: native implicit bidiagonal QR did not converge");
        }
        return ctx.get_event();
    }
}

template <typename T>
size_t bdsqr_buffer_size(Queue& ctx,
                         const VectorView<T>& d,
                         const VectorView<T>& e,
                         Span<T> singular_values_out) {
    static_cast<void>(e);
    static_cast<void>(singular_values_out);
    return workspace_bytes([&](BumpAllocator& pool) {
        return bdsqr_layout<T>(ctx, pool, static_cast<int32_t>(d.batch_size()));
    });
}

#define BDSQR_INSTANTIATE(back, fp) \
    template Event bdsqr<back, BATCHLAS_UNPAREN fp>( \
        Queue&, \
        const VectorView<BATCHLAS_UNPAREN fp>&, \
        const VectorView<BATCHLAS_UNPAREN fp>&, \
        Span<BATCHLAS_UNPAREN fp>, \
        const Span<std::byte>&, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        bool, Span<int32_t>); \
    template Event bdsqr<back, BATCHLAS_UNPAREN fp>( \
        Queue&, \
        const VectorView<BATCHLAS_UNPAREN fp>&, \
        const VectorView<BATCHLAS_UNPAREN fp>&, \
        Span<BATCHLAS_UNPAREN fp>, \
        const Span<std::byte>&, \
        bool, Span<int32_t>);

#define BDSQR_BUFFER_INSTANTIATE(fp) \
    template size_t bdsqr_buffer_size<BATCHLAS_UNPAREN fp>( \
        Queue&, \
        const VectorView<BATCHLAS_UNPAREN fp>&, \
        const VectorView<BATCHLAS_UNPAREN fp>&, \
        Span<BATCHLAS_UNPAREN fp>);

BATCHLAS_INSTANTIATE_REAL_ALL_BACKENDS(BDSQR_INSTANTIATE)

BATCHLAS_FOR_EACH_REAL_TYPE(BDSQR_BUFFER_INSTANTIATE)

#undef BDSQR_BUFFER_INSTANTIATE
#undef BDSQR_INSTANTIATE

#undef BDSQR_INSTANTIATE
#undef BDSQR_BUFFER_INSTANTIATE

} // namespace batchlas
