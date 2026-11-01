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
#include <batchlas/blas/device/v2/dla.hh>
#include <batchlas/blas/device/v2/ext/bdsqr-bdsdc.hh>

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

namespace v2 = ::batchlas::device::v2;

template <Backend B, typename T> class BdsdcBuildGk;
template <Backend B, typename T> class BdsdcValuesOnly;
template <Backend B, typename T> class BdsdcExtract;
template <Backend B, typename T> class BdsdcRepair;

// stedc wants a non-negative off-diagonal. A tridiagonal with signed
// off-diagonals is similar to the |.| one through S = diag(s), s_0 = 1,
// s_{i+1} = s_i * sign(f_i), and that sign vector is folded into the extraction
// so it costs no extra pass over the eigenvector matrix.
template <typename T>
StedcParams<T> bdsdc_stedc_params() {
    StedcParams<T> params{};
    params.leaf_steqr_params.sort = true;
    params.leaf_steqr_params.sort_order = SortOrder::Ascending;
    params.leaf_steqr_params.back_transform = false;
    return params;
}

template <typename T>
struct BdsdcWorkspace {
    VectorView<T> gk_d;      // 2n zero diagonal
    VectorView<T> gk_e;      // 2n-1 interleaved off-diagonal, made non-negative
    VectorView<T> gk_sign;   // 2n row signs undoing that normalisation
    VectorView<T> lambda;    // 2n eigenvalues, ascending
    MatrixView<T, MatrixFormat::Dense> Z;  // 2n x 2n eigenvectors
    Span<std::byte> stedc_ws;
};

template <Backend B, typename T>
BdsdcWorkspace<T> bdsdc_layout(Queue& ctx,
                               BumpAllocator& pool,
                               int32_t n,
                               int32_t batch,
                               bool want_vectors) {
    const int32_t N = 2 * n;
    const size_t nb = static_cast<size_t>(batch);

    BdsdcWorkspace<T> ws{};
    auto d_span = pool.allocate<T>(ctx, static_cast<size_t>(N) * nb);
    auto e_span = pool.allocate<T>(ctx, static_cast<size_t>(N) * nb);
    auto s_span = pool.allocate<T>(ctx, static_cast<size_t>(N) * nb);
    auto l_span = pool.allocate<T>(ctx, static_cast<size_t>(N) * nb);

    ws.gk_d = VectorView<T>(d_span, N, batch, 1, N);
    ws.gk_e = VectorView<T>(e_span, N - 1, batch, 1, N);
    ws.gk_sign = VectorView<T>(s_span, N, batch, 1, N);
    ws.lambda = VectorView<T>(l_span, N, batch, 1, N);

    // Z is allocated even when only values are wanted: stedc unconditionally clears
    // the eigenvector view it is handed, so a null view would fault.
    static_cast<void>(want_vectors);
    auto z_span = pool.allocate<T>(ctx, static_cast<size_t>(N) * static_cast<size_t>(N) * nb);
    ws.Z = MatrixView<T, MatrixFormat::Dense>(z_span.data(), N, N, N,
                                              static_cast<int64_t>(N) * static_cast<int64_t>(N), batch);

    ws.stedc_ws = pool.allocate<std::byte>(
        ctx, stedc_buffer_size<B, T>(ctx, static_cast<size_t>(N), nb, JobType::EigenVectors,
                                        bdsdc_stedc_params<T>()));
    return ws;
}

// Build the interleaved Golub-Kahan tridiagonal, with the off-diagonal made
// non-negative and the compensating row signs recorded.
template <Backend B, typename T>
void bdsdc_build_gk(Queue& ctx,
                    const VectorView<T>& d,
                    const VectorView<T>& e,
                    const BdsdcWorkspace<T>& ws,
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
        h.parallel_for<BdsdcBuildGk<B, T>>(
            sycl::range<1>(static_cast<size_t>(batch)), [=](sycl::id<1> tid) {
            const int32_t b = static_cast<int32_t>(tid[0]);
            if (mask != nullptr && mask[b] == 0) return;  // I3
            const v2::Thread th{};
            const auto gd = v2::vec_item(GD, b);
            const auto ge = v2::vec_item(GE, b);
            const auto gs = v2::vec_item(GS, b);
            v2::fill(th, v2::Zero{}, gd);
            // off-diagonal f[2k] = d_k (couples v_k,u_k), f[2k+1] = e_k (couples u_k,v_{k+1})
            v2::interleave(th, v2::vec_item(D, b), v2::vec_item(E, b).head(nn - 1), ge);
            gs(0) = T(1);
            for (int32_t i = 0; i < N - 1; ++i) {
                const T fi = ge(i);
                const T sgn = (fi >= T(0)) ? T(1) : T(-1);
                gs(i + 1) = gs(i) * sgn;
                ge(i) = sycl::fabs(fi);
            }
        });
    });
}

template <Backend B, typename T>
void bdsdc_extract_values(Queue& ctx,
                          const BdsdcWorkspace<T>& ws,
                          Span<T> sigma,
                          int32_t n,
                          int32_t batch,
                          bool sort_desc,
                          const std::uint8_t* mask) {
    ctx->submit([&](sycl::handler& h) {
        auto L = ws.lambda;
        T* out = sigma.data();
        const int32_t nn = n;
        const bool desc = sort_desc;
        h.parallel_for<BdsdcValuesOnly<B, T>>(
            sycl::range<1>(static_cast<size_t>(n) * static_cast<size_t>(batch)), [=](sycl::id<1> tid) {
            const int32_t lin = static_cast<int32_t>(tid[0]);
            const int32_t b = lin / nn;
            const int32_t i = lin - b * nn;
            if (mask != nullptr && mask[b] == 0) return;  // I3
            const int32_t src = desc ? (2 * nn - 1 - i) : (nn + i);
            const T lam = L(src, b);
            out[static_cast<size_t>(b) * static_cast<size_t>(nn) + i] = sycl::fmax(lam, T(0));
        });
    });
}

// One work-group per (batch item, output column): split the eigenvector into its
// even rows (-> v) and odd rows (-> u) and normalise each half.
template <Backend B, typename T>
void bdsdc_extract_vectors(Queue& ctx,
                           const BdsdcWorkspace<T>& ws,
                           Span<T> sigma,
                           const MatrixView<T, MatrixFormat::Dense>& u,
                           const MatrixView<T, MatrixFormat::Dense>& vh,
                           int32_t n,
                           int32_t batch,
                           bool want_u,
                           bool want_vh,
                           bool sort_desc,
                           const std::uint8_t* mask) {
    constexpr int32_t kGroup = 64;
    auto& u_mut = const_cast<MatrixView<T, MatrixFormat::Dense>&>(u);
    auto& vh_mut = const_cast<MatrixView<T, MatrixFormat::Dense>&>(vh);

    ctx->submit([&](sycl::handler& h) {
        auto L = ws.lambda;
        auto S = ws.gk_sign;
        auto Zk = ws.Z.kernel_view();
        auto U = u_mut.kernel_view();
        auto Vh = vh_mut.kernel_view();
        T* sig = sigma.data();
        const int32_t nn = n;
        const bool desc = sort_desc;
        const bool wu = want_u;
        const bool wv = want_vh;

        batchlas::impl::parallel_for<BdsdcExtract<B, T>>(h,
            sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(batch) * static_cast<size_t>(n) * kGroup),
                              sycl::range<1>(kGroup)),
            [=](sycl::nd_item<1> it) {
            const v2::WorkGroup<kGroup> wg{it};
            const int32_t grp = static_cast<int32_t>(it.get_group(0));
            const int32_t b = grp / nn;
            if (mask != nullptr && mask[b] == 0) return;  // I1: one work-group per (item, column)
            const int32_t i = grp - b * nn;
            const int32_t src = desc ? (2 * nn - 1 - i) : (nn + i);

            // Eigenvector src of the Golub-Kahan matrix: even rows -> v, odd rows -> u.
            const auto z = v2::mat_item(Zk, b).col(src);
            const auto s = v2::vec_item(S, b);
            const auto zv = v2::slice(z, 0, nn, 2), zu = v2::slice(z, 1, nn, 2);
            const auto sv = v2::slice(s, 0, nn, 2), su = v2::slice(s, 1, nn, 2);
            const auto [p2, q2] = v2::sumsq_many<v2::Scope::Group>(wg, zv, zu);
            const T pn = sycl::sqrt(p2);
            const T qn = sycl::sqrt(q2);

            if (wg.leader()) {
                sig[static_cast<size_t>(b) * static_cast<size_t>(nn) + i] = sycl::fmax(L(src, b), T(0));
            }

            // Guard the amplification, not just division by zero: the exact half-norm is
            // 1/sqrt(2), but a half rotated away comes back at ~1e-20, and 1/1e-20 writes
            // +-inf into the column, NaNing every later dot product so the repair below
            // rejects all n candidate axes. Below this bound, zero it and let repair rebuild.
            const T half_present = T(0.0625);
            const T pinv = (pn > half_present) ? (T(1) / pn) : T(0);
            const T qinv = (qn > half_present) ? (T(1) / qn) : T(0);
            // Vh is V^T: row i holds v_i. Signs undo the off-diagonal normalisation.
            if (wv) v2::dgmv(wg, pinv, sv, zv, v2::mat_item(Vh, b).row(i).head(nn));
            if (wu) v2::dgmv(wg, qinv, su, zu, v2::mat_item(U, b).col(i).head(nn));
        });
    });
}

// Rebuild the singular vectors of the numerically-zero singular values. The
// criterion is sigma, not the column's shape (a degenerate block looks locally
// healthy). Axes are ranked in one pass: residual of e_c against orthonormal Q
// is 1 - sum_t Q(c,t)^2. Full-rank input reads sigma and exits.
// evidence: docs/perf/gesvd.md#gesvd-bdsdc-null-space-repair-criteria-that-failed
template <Backend B, typename T>
void bdsdc_repair_degenerate(Queue& ctx,
                             const MatrixView<T, MatrixFormat::Dense>& u,
                             const MatrixView<T, MatrixFormat::Dense>& vh,
                             Span<T> sigma,
                             int32_t n,
                             int32_t batch,
                             bool want_u,
                             bool want_vh,
                             const std::uint8_t* mask) {
    constexpr int32_t kGroup = 64;
    auto& u_mut = const_cast<MatrixView<T, MatrixFormat::Dense>&>(u);
    auto& vh_mut = const_cast<MatrixView<T, MatrixFormat::Dense>&>(vh);

    ctx->submit([&](sycl::handler& h) {
        auto U = u_mut.kernel_view();
        auto Vh = vh_mut.kernel_view();
        const T* sig = sigma.data();
        const int32_t nn = n;
        const bool wu = want_u;
        const bool wv = want_vh;
        // Repair sigma_i < eps*sigma_max/(2*tau): below that the +/- pair is
        // unresolved and U gets parallel columns. No factor of n (it would
        // discard informative sigma). Values keep gebrd's eps*||A|| floor.
        // evidence: docs/perf/gesvd.md#gesvd-bdsdc-the-repair-threshold-from-the-resolution-limit
        constexpr T kOrthTarget = T(1e-3);
        const T tol_factor =
            (T(1) / (T(2) * kOrthTarget)) * std::numeric_limits<T>::epsilon();

        batchlas::impl::parallel_for<BdsdcRepair<B, T>>(h,
            sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(batch) * kGroup),
                              sycl::range<1>(kGroup)),
            [=](sycl::nd_item<1> it) {
            const v2::WorkGroup<kGroup> wg{it};
            const v2::Thread th{};  // the uniform scalar prologue, replicated on every item
            const int32_t b = static_cast<int32_t>(it.get_group(0));
            if (mask != nullptr && mask[b] == 0) return;  // I1
            const auto sb = v2::vec(sig + static_cast<size_t>(b) * static_cast<size_t>(nn), nn);

            const T tol = v2::fmax_reduce(th, nn, [&](int i) { return sb(i); }, T(0)) * tol_factor;
            const int32_t ndeg = v2::reduce(th, nn, [&](int i) { return (sb(i) <= tol) ? 1 : 0; });
            if (ndeg == 0) return;

            // Basis vector t is column t of Q: Q = V (= Vh^T) for which == 0, Q = U for which == 1.
            const auto vq = v2::trans(v2::mat_item(Vh, b).sub(0, 0, nn, nn));
            const auto uq = v2::as_smat(v2::mat_item(U, b).sub(0, 0, nn, nn));
            for (int32_t which = 0; which < 2; ++which) {
                if (which == 0 && !wv) continue;
                if (which == 1 && !wu) continue;
                const v2::SMat<T> Q = (which == 0) ? vq : uq;

                for (int32_t j = 0; j < nn; ++j) {
                    if (sb(j) > tol) continue;
                    // The final columns: the earlier rebuilt ones (t < j) and ALL
                    // non-degenerate ones, including those after j. Skipping the latter
                    // loses orthogonality against them under ascending order.
                    auto final_col = [&](int t) { return t != j && !(t > j && sb(t) <= tol); };

                    // Seed axis: e_c with the smallest residual 1 - ||Q(c, final)||^2.
                    const int32_t c_star = v2::iminloc<v2::Scope::Group>(wg, nn, [&](int c) {
                        T acc = T(0);
                        for (int32_t t = 0; t < nn; ++t) {
                            if (!final_col(t)) continue;
                            const T q = Q(c, t);
                            acc += q * q;
                        }
                        return acc;
                    });

                    // w := e_c*, project out the final columns twice (modified Gram-Schmidt;
                    // one pass is not enough when e_c* is nearly in their span), normalise.
                    const auto w = Q.col(j);
                    v2::set_unit(wg, w, c_star);
                    wg.barrier(v2::Mem::Global);
                    for (int32_t pass = 0; pass < 2; ++pass) {
                        for (int32_t t = 0; t < nn; ++t) {
                            if (!final_col(t)) continue;
                            const T dot = v2::dot<v2::Scope::Group>(wg, w, Q.col(t));
                            v2::axpy(wg, -dot, Q.col(t), w);
                            wg.barrier(v2::Mem::Global);
                        }
                    }
                    const T nrm2 = v2::dot<v2::Scope::Group>(wg, w, w);
                    // c_star is the best axis available, so a residual too small to normalise
                    // means the final columns already span the space; leave the column as
                    // extracted rather than amplifying noise.
                    if (nrm2 > std::numeric_limits<T>::min()) {
                        v2::scal(wg, T(1) / sycl::sqrt(nrm2), w);
                    }
                    wg.barrier(v2::Mem::Global);
                }
            }
        });
    });
}

} // namespace

template <Backend B, typename T>
Event bdsdc(Queue& ctx,
            const VectorView<T>& d,
            const VectorView<T>& e,
            Span<T> singular_values_out,
            const Span<std::byte>& ws,
            const MatrixView<T, MatrixFormat::Dense>& u,
            const MatrixView<T, MatrixFormat::Dense>& vh,
            bool sort_desc,
            Span<int32_t> info) {
    const auto mask_span = ::batchlas::detail::call_mask("bdsdc", d, e, u, vh);
    const std::uint8_t* const mask = ::batchlas::detail::mask_ptr(mask_span);
    if constexpr (detail::is_complex<T>::value) {
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
        auto layout = bdsdc_layout<B, T>(ctx, pool, n, batch, want_vectors);
        // Rebuilt from raw pointers: stedc gets the call's mask on every view it is handed.
        layout.gk_d = layout.gk_d.with_batch_mask(mask_span);
        layout.gk_e = layout.gk_e.with_batch_mask(mask_span);
        layout.lambda = layout.lambda.with_batch_mask(mask_span);
        layout.Z = layout.Z.with_batch_mask(mask_span);

        bdsdc_build_gk<B, T>(ctx, d, e, layout, n, batch, mask);

        // (void) on an Event: deliberate. This Queue is in-order, so the next submission
        // is already ordered after this one and the Event carries nothing the caller needs.
        // bdsdc owns no iteration: the Golub-Kahan build and both extraction
        // passes are direct, so stedc is the only thing here that can fail to
        // converge. Its batch axis is the caller's (the tridiagonal is 2n long,
        // not the batch), so `info` forwards unchanged and stedc does the clear.
        (void)stedc<B, T>(ctx,
                    layout.gk_d,
                    layout.gk_e,
                    layout.lambda,
                    layout.stedc_ws,
                    JobType::EigenVectors,
                    bdsdc_stedc_params<T>(),
                    layout.Z,
                    info);

        if (!want_vectors) {
            bdsdc_extract_values<B, T>(ctx, layout, singular_values_out, n, batch, sort_desc, mask);
            return ctx.get_event();
        }

        bdsdc_extract_vectors<B, T>(ctx, layout, singular_values_out, u, vh,
                                    n, batch, want_u, want_vh, sort_desc, mask);
        bdsdc_repair_degenerate<B, T>(ctx, u, vh, singular_values_out, n, batch, want_u, want_vh, mask);
        return ctx.get_event();
    }
}

template <Backend B, typename T>
Event bdsdc(Queue& ctx,
            const VectorView<T>& d,
            const VectorView<T>& e,
            Span<T> singular_values_out,
            const Span<std::byte>& ws,
            bool sort_desc,
            Span<int32_t> info) {
    return bdsdc<B, T>(ctx, d, e, singular_values_out, ws,
                       MatrixView<T, MatrixFormat::Dense>(nullptr, 0, 0, 1, 1, d.batch_size()),
                       MatrixView<T, MatrixFormat::Dense>(nullptr, 0, 0, 1, 1, d.batch_size()),
                       sort_desc,
                       info);
}

template <Backend B, typename T>
size_t bdsdc_buffer_size(Queue& ctx,
                         const VectorView<T>& d,
                         const VectorView<T>& e,
                         Span<T> singular_values_out,
                         bool want_vectors) {
    static_cast<void>(e);
    static_cast<void>(singular_values_out);
    const int32_t n = static_cast<int32_t>(d.size());
    const int32_t batch = static_cast<int32_t>(d.batch_size());
    if (n <= 0) return 0;
    return workspace_bytes([&](BumpAllocator& pool) {
        auto layout = bdsdc_layout<B, T>(ctx, pool, n, batch, want_vectors);
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

} // namespace batchlas
