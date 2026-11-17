// SPDX-License-Identifier: MIT
// Bidiagonal SVD via stedc, NOT a port of LAPACK's dlasd*: the perfect-shuffled
// Golub-Kahan matrix [0 B^T; B 0] is tridiagonal with zero diagonal and
// off-diagonal (d_0, e_0, d_1, ..., d_{n-1}), eigenvalues +/- sigma_i. Nothing is
// squared; splitting an eigenvector into even (-> v) / odd (-> u) rows is exact.
// evidence: docs/design/gesvd.md#gesvd-design-why-bdsdc-goes-through-golub-kahan

#include <batchlas/blas/extensions.hh>
#include <batchlas/backend_config.h>

#include <batchlas/util/mempool.hh>
#include <batchlas/util/group-collectives.hh>
#include <batchlas/device.hh>
#include <batchlas/device/lapack/gesvd.hh>
#include <batchlas/device/lapack/bdsdc.hh>

#include <src/math-helpers.hh>
#include <src/queue.hh>
#include <src/util/template-instantiations.hh>
#include <src/util/batch_mask.hh>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace batchlas {

namespace {

template <Backend B, typename Real> class BdsdcBuildGk;
template <Backend B, typename Real> class BdsdcValuesOnly;
template <Backend B, typename Real> class BdsdcExtract;
template <Backend B, typename Real> class BdsdcRepair;

// stedc wants a non-negative off-diagonal. A tridiagonal with signed
// off-diagonals is similar to the |.| one through S = diag(s), s_0 = 1,
// s_{i+1} = s_i * sign(f_i), and that sign vector is folded into the extraction
// so it costs no extra pass over the eigenvector matrix.
template <typename Real>
StedcParams<Real> bdsdc_stedc_params() {
    StedcParams<Real> params{};
    params.leaf_steqr_params.sort = true;
    params.leaf_steqr_params.sort_order = SortOrder::Ascending;
    params.leaf_steqr_params.back_transform = false;
    return params;
}

template <typename Real>
struct BdsdcWorkspace {
    VectorView<Real> gk_d;      // 2n zero diagonal
    VectorView<Real> gk_e;      // 2n-1 interleaved off-diagonal, made non-negative
    VectorView<Real> gk_sign;   // 2n row signs undoing that normalisation
    VectorView<Real> lambda;    // 2n eigenvalues, ascending
    MatrixView<Real, MatrixFormat::Dense> Z;  // 2n x 2n eigenvectors
    Span<std::byte> stedc_ws;
};

template <Backend B, typename Real>
BdsdcWorkspace<Real> bdsdc_layout(Queue& ctx,
                                  BumpAllocator& pool,
                                  int32_t n,
                                  int32_t batch,
                                  bool want_vectors) {
    const int32_t N = 2 * n;
    const size_t nb = static_cast<size_t>(batch);

    BdsdcWorkspace<Real> ws{};
    auto d_span = pool.allocate<Real>(ctx, static_cast<size_t>(N) * nb);
    auto e_span = pool.allocate<Real>(ctx, static_cast<size_t>(N) * nb);
    auto s_span = pool.allocate<Real>(ctx, static_cast<size_t>(N) * nb);
    auto l_span = pool.allocate<Real>(ctx, static_cast<size_t>(N) * nb);

    ws.gk_d = VectorView<Real>(d_span, N, batch, 1, N);
    ws.gk_e = VectorView<Real>(e_span, N - 1, batch, 1, N);
    ws.gk_sign = VectorView<Real>(s_span, N, batch, 1, N);
    ws.lambda = VectorView<Real>(l_span, N, batch, 1, N);

    // Z is allocated even when only values are wanted: stedc unconditionally clears
    // the eigenvector view it is handed, so a null view would fault.
    static_cast<void>(want_vectors);
    auto z_span = pool.allocate<Real>(ctx, static_cast<size_t>(N) * static_cast<size_t>(N) * nb);
    ws.Z = MatrixView<Real, MatrixFormat::Dense>(z_span.data(), N, N, N,
                                                 static_cast<int64_t>(N) * static_cast<int64_t>(N), batch);

    ws.stedc_ws = pool.allocate<std::byte>(
        ctx, stedc_buffer_size<B, Real>(ctx, static_cast<size_t>(N), nb, JobType::EigenVectors,
                                        bdsdc_stedc_params<Real>()));
    return ws;
}

// Build the interleaved Golub-Kahan tridiagonal, with the off-diagonal made
// non-negative and the compensating row signs recorded.
template <Backend B, typename Real>
void bdsdc_build_gk(Queue& ctx,
                    const VectorView<Real>& d,
                    const VectorView<Real>& e,
                    const BdsdcWorkspace<Real>& ws,
                    int32_t n,
                    int32_t batch,
                    const std::uint8_t* mask) {
    ctx->submit([&](sycl::handler& h) {
        auto D = d;
        auto E = e;
        auto GD = ws.gk_d;
        auto GE = ws.gk_e;
        auto GS = ws.gk_sign;
        const int32_t nn = n;
        const int32_t N = 2 * n;
        batchlas::impl::parallel_for<BdsdcBuildGk<B, Real>>(h,
            sycl::range<1>(static_cast<size_t>(batch)), [mask, N, GD, nn, GE, D, E, GS](sycl::id<1> tid) {
            const int32_t b = static_cast<int32_t>(tid[0]);
            if (mask != nullptr && mask[b] == 0) return;  // I3
            device::detail::build_gk(device::item(D, b, nn), device::item(E, b, nn - 1), device::item(GD, b, N),
                                       device::item(GE, b, N - 1), device::item(GS, b, N));
        });
    });
}

template <Backend B, typename Real>
void bdsdc_extract_values(Queue& ctx,
                          const BdsdcWorkspace<Real>& ws,
                          Span<Real> sigma,
                          int32_t n,
                          int32_t batch,
                          bool sort_desc,
                          const std::uint8_t* mask) {
    ctx->submit([&](sycl::handler& h) {
        auto L = ws.lambda;
        Real* out = sigma.data();
        const int32_t nn = n;
        const bool desc = sort_desc;
        batchlas::impl::parallel_for<BdsdcValuesOnly<B, Real>>(h,
            sycl::range<1>(static_cast<size_t>(n) * static_cast<size_t>(batch)), [=](sycl::id<1> tid) {
            const int32_t lin = static_cast<int32_t>(tid[0]);
            const int32_t b = lin / nn;
            const int32_t i = lin - b * nn;
            if (mask != nullptr && mask[b] == 0) return;  // I3
            const int32_t src = desc ? (2 * nn - 1 - i) : (nn + i);
            const Real lam = L(src, b);
            out[static_cast<size_t>(b) * static_cast<size_t>(nn) + i] = sycl::fmax(lam, Real(0));
        });
    });
}

// One work-group per (batch item, output column): split the eigenvector into its
// even rows (-> v) and odd rows (-> u) and normalise each half.
template <Backend B, typename Real>
void bdsdc_extract_vectors(Queue& ctx,
                           const BdsdcWorkspace<Real>& ws,
                           Span<Real> sigma,
                           const MatrixView<Real, MatrixFormat::Dense>& u,
                           const MatrixView<Real, MatrixFormat::Dense>& vh,
                           int32_t n,
                           int32_t batch,
                           bool want_u,
                           bool want_vh,
                           bool sort_desc,
                           const std::uint8_t* mask) {
    constexpr int32_t kGroup = 64;
    auto& u_mut = const_cast<MatrixView<Real, MatrixFormat::Dense>&>(u);
    auto& vh_mut = const_cast<MatrixView<Real, MatrixFormat::Dense>&>(vh);

    ctx->submit([&](sycl::handler& h) {
        auto L = ws.lambda;
        auto S = ws.gk_sign;
        auto Zk = ws.Z.kernel_view();
        auto U = u_mut.kernel_view();
        auto Vh = vh_mut.kernel_view();
        Real* sig = sigma.data();
        const int32_t nn = n;
        const bool desc = sort_desc;
        const bool wu = want_u;
        const bool wv = want_vh;

        batchlas::impl::parallel_for<BdsdcExtract<B, Real>>(h,
            sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(batch) * static_cast<size_t>(n) * kGroup),
                              sycl::range<1>(kGroup)),
            [=](sycl::nd_item<1> it) {
            const device::WorkGroup<kGroup> wg{it};
            const int32_t grp = static_cast<int32_t>(it.get_group(0));
            const int32_t b = grp / nn;
            if (mask != nullptr && mask[b] == 0) return;  // I1: one work-group per (item, column)
            const int32_t i = grp - b * nn;
            const int32_t src = desc ? (2 * nn - 1 - i) : (nn + i);
            // Eigenvector src of the Golub-Kahan matrix; Vh is V^T, so row i holds v_i.
            device::detail::extract_column(wg, device::col(device::item(Zk, b, 2 * nn, 2 * nn), src), device::item(S, b), nn,
                                             device::item(L, b), src, device::vec(sig + static_cast<size_t>(b) * static_cast<size_t>(nn), nn), i,
                                             device::row(device::item(Vh, b, nn, nn), i), device::col(device::item(U, b, nn, nn), i), wu, wv);
        });
    });
}

// Rebuild the singular vectors of the numerically-zero singular values. The
// criterion is sigma, not the column's shape (a degenerate block looks locally
// healthy). Axes are ranked in one pass: residual of e_c against orthonormal Q
// is 1 - sum_t Q(c,t)^2. Full-rank input reads sigma and exits.
// evidence: docs/perf/gesvd.md#gesvd-bdsdc-null-space-repair-criteria-that-failed
template <Backend B, typename Real>
void bdsdc_repair_degenerate(Queue& ctx,
                             const MatrixView<Real, MatrixFormat::Dense>& u,
                             const MatrixView<Real, MatrixFormat::Dense>& vh,
                             Span<Real> sigma,
                             int32_t n,
                             int32_t batch,
                             bool want_u,
                             bool want_vh,
                             const std::uint8_t* mask) {
    constexpr int32_t kGroup = 64;
    auto& u_mut = const_cast<MatrixView<Real, MatrixFormat::Dense>&>(u);
    auto& vh_mut = const_cast<MatrixView<Real, MatrixFormat::Dense>&>(vh);

    ctx->submit([&](sycl::handler& h) {
        auto U = u_mut.kernel_view();
        auto Vh = vh_mut.kernel_view();
        const Real* sig = sigma.data();
        const int32_t nn = n;
        const bool wu = want_u;
        const bool wv = want_vh;

        batchlas::impl::parallel_for<BdsdcRepair<B, Real>>(h,
            sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(batch) * kGroup),
                              sycl::range<1>(kGroup)),
            [=](sycl::nd_item<1> it) {
            const int32_t b = static_cast<int32_t>(it.get_group(0));
            if (mask != nullptr && mask[b] == 0) return;  // I1
            device::detail::repair(device::WorkGroup<kGroup>{it}, device::item(U, b, nn, nn), device::item(Vh, b, nn, nn),
                                     device::vec(sig + static_cast<size_t>(b) * static_cast<size_t>(nn), nn), nn, wu, wv);
        });
    });
}

}  // namespace

template <Backend B, typename Real>
Event bdsdc(Queue& ctx,
            const VectorView<Real>& d,
            const VectorView<Real>& e,
            Span<Real> singular_values_out,
            const Span<std::byte>& ws,
            const MatrixView<Real, MatrixFormat::Dense>& u,
            const MatrixView<Real, MatrixFormat::Dense>& vh,
            bool sort_desc,
            Span<int32_t> info) {
    const auto mask_span = ::batchlas::detail::call_mask("bdsdc", d, e, u, vh);
    const std::uint8_t* const mask = ::batchlas::detail::mask_ptr(mask_span);
    if constexpr (detail::is_complex<Real>::value) {
        throw batchlas::unsupported("bdsdc: complex types are not implemented");
    } else {
        const int32_t n = static_cast<int32_t>(d.size());
        const int32_t batch = static_cast<int32_t>(d.batch_size());
        if (n <= 0) {
            return ctx.get_event();
        }
        if (!ctx.in_order()) {
            throw batchlas::invalid_argument("bdsdc: requires an in-order Queue");
        }
        if (static_cast<int32_t>(singular_values_out.size()) < n * batch) {
            throw batchlas::invalid_argument("bdsdc: singular_values span too small");
        }

        const bool want_u = u.data_ptr() != nullptr && u.rows() > 0 && u.cols() >= n;
        const bool want_vh = vh.data_ptr() != nullptr && vh.cols() > 0 && vh.rows() >= n;
        const bool want_vectors = want_u || want_vh;

        Span<std::byte> ws_mut(const_cast<std::byte*>(ws.data()), ws.size());
        BumpAllocator pool(ws_mut);
        auto layout = bdsdc_layout<B, Real>(ctx, pool, n, batch, want_vectors);
        // Rebuilt from raw pointers: stedc gets the call's mask on every view it is handed.
        layout.gk_d = layout.gk_d.with_batch_mask(mask_span);
        layout.gk_e = layout.gk_e.with_batch_mask(mask_span);
        layout.lambda = layout.lambda.with_batch_mask(mask_span);
        layout.Z = layout.Z.with_batch_mask(mask_span);

        bdsdc_build_gk<B, Real>(ctx, d, e, layout, n, batch, mask);

        // (void) on an Event: deliberate. This Queue is in-order, so the next submission
        // is already ordered after this one and the Event carries nothing the caller needs.
        // bdsdc owns no iteration: the Golub-Kahan build and both extraction
        // passes are direct, so stedc is the only thing here that can fail to
        // converge. Its batch axis is the caller's (the tridiagonal is 2n long,
        // not the batch), so `info` forwards unchanged and stedc does the clear.
        (void)stedc<B, Real>(ctx,
                    layout.gk_d,
                    layout.gk_e,
                    layout.lambda,
                    layout.stedc_ws,
                    JobType::EigenVectors,
                    bdsdc_stedc_params<Real>(),
                    layout.Z,
                    info);

        if (!want_vectors) {
            bdsdc_extract_values<B, Real>(ctx, layout, singular_values_out, n, batch, sort_desc, mask);
            return ctx.get_event();
        }

        bdsdc_extract_vectors<B, Real>(ctx, layout, singular_values_out, u, vh,
                                       n, batch, want_u, want_vh, sort_desc, mask);
        bdsdc_repair_degenerate<B, Real>(ctx, u, vh, singular_values_out, n, batch, want_u, want_vh, mask);
        return ctx.get_event();
    }
}

template <Backend B, typename Real>
Event bdsdc(Queue& ctx,
            const VectorView<Real>& d,
            const VectorView<Real>& e,
            Span<Real> singular_values_out,
            const Span<std::byte>& ws,
            bool sort_desc,
            Span<int32_t> info) {
    return bdsdc<B, Real>(ctx, d, e, singular_values_out, ws,
                          MatrixView<Real, MatrixFormat::Dense>(nullptr, 0, 0, 1, 1, d.batch_size()),
                          MatrixView<Real, MatrixFormat::Dense>(nullptr, 0, 0, 1, 1, d.batch_size()),
                          sort_desc,
                          info);
}

template <Backend B, typename Real>
size_t bdsdc_buffer_size(Queue& ctx,
                         const VectorView<Real>& d,
                         const VectorView<Real>& e,
                         Span<Real> singular_values_out,
                         bool want_vectors) {
    static_cast<void>(e);
    static_cast<void>(singular_values_out);
    const int32_t n = static_cast<int32_t>(d.size());
    const int32_t batch = static_cast<int32_t>(d.batch_size());
    if (n <= 0) return 0;
    return workspace_bytes([&](BumpAllocator& pool) {
        auto layout = bdsdc_layout<B, Real>(ctx, pool, n, batch, want_vectors);
        static_cast<void>(layout);
        return 0;
    });
}

#define BDSDC_INSTANTIATE(back, fp) \
    template Event bdsdc<back, BATCHLAS_UNPAREN fp>( \
        Queue&, \
        const VectorView<BATCHLAS_UNPAREN fp>&, \
        const VectorView<BATCHLAS_UNPAREN fp>&, \
        Span<BATCHLAS_UNPAREN fp>, \
        const Span<std::byte>&, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        bool, Span<int32_t>); \
    template Event bdsdc<back, BATCHLAS_UNPAREN fp>( \
        Queue&, \
        const VectorView<BATCHLAS_UNPAREN fp>&, \
        const VectorView<BATCHLAS_UNPAREN fp>&, \
        Span<BATCHLAS_UNPAREN fp>, \
        const Span<std::byte>&, \
        bool, Span<int32_t>); \
    template size_t bdsdc_buffer_size<back, BATCHLAS_UNPAREN fp>( \
        Queue&, \
        const VectorView<BATCHLAS_UNPAREN fp>&, \
        const VectorView<BATCHLAS_UNPAREN fp>&, \
        Span<BATCHLAS_UNPAREN fp>, \
        bool);

BATCHLAS_INSTANTIATE_REAL_ALL_BACKENDS(BDSDC_INSTANTIATE)

#undef BDSDC_INSTANTIATE

}  // namespace batchlas
