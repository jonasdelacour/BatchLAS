// SPDX-License-Identifier: MIT
// syev_align: eigenvector continuity (fullerenes plan P2.5). One gemm forms M = V_old^H V; one
// work-item per cluster runs a one-sided Jacobi SVD of M's diagonal block and stores R^H over it;
// a copy of V and one element-wise kernel apply V_c := V_c R. No kernel selection: there is one
// algorithm, no vendor counterpart, and the only routed work is the gemm, which selects for itself.

#include <batchlas/backend_config.h>

#include <batchlas/blas/functions/gemm.hh>
#include <batchlas/blas/functions/syev_align.hh>
#include <batchlas/util/mempool.hh>
#include <batchlas/device/arith.hh>

#include <src/util/batch_mask.hh>
#include <src/queue.hh>
#include <src/util/template-instantiations.hh>

#include <sycl/sycl.hpp>

#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace batchlas {
namespace ops::syev_align {

template <typename Scalar>
using MV = MatrixView<Scalar, MatrixFormat::Dense>;

template <Backend B, typename Scalar>
struct ClusterRotationKernel {};
template <Backend B, typename Scalar>
struct ClusterApplyKernel {};

template <typename Scalar>
inline constexpr bool complex_v = is_std_complex_v<Scalar>;

template <typename Scalar>
struct Workspace {
    Span<Scalar> m;            // m x m per item: M = V_old^H V, then R^H on each cluster's diagonal block
    Span<Scalar> vcopy;        // n x m per item: W of the Jacobi SVD (block (s, s), ld n), then a copy of V
    Span<Scalar> scratch;      // m per item: column flags, then one row of U
    Span<int32_t> start;  // m per item: first column of the column's cluster
    Span<int32_t> end;    // m per item: one past its last column
};

// One description for the call and the size query (a mismatch overruns the workspace).
template <typename Scalar>
Workspace<Scalar> carve(const Device& dev, BumpAllocator& pool, int64_t n, int64_t m, int64_t batch) {
    Workspace<Scalar> w;
    w.m = pool.allocate<Scalar>(dev, size_t(m * m * batch));
    w.vcopy = pool.allocate<Scalar>(dev, size_t(n * m * batch));
    w.scratch = pool.allocate<Scalar>(dev, size_t(m * batch));
    w.start = pool.allocate<int32_t>(dev, size_t(m * batch));
    w.end = pool.allocate<int32_t>(dev, size_t(m * batch));
    return w;
}

// G is the cluster's p x p block of M (ld m), W the accumulated right rotations (ld n). On return
// G holds R^H = U W^H, where B = U Sigma W^H. Columns of U whose singular value is below
// p eps sigma_max are completed by Gram-Schmidt on unit vectors, so R stays unitary.
template <typename Scalar>
inline void procrustes_block(Scalar* G, int64_t ldg, Scalar* W, int64_t ldw, Scalar* flag, int p) {
    using Real = float_t<Scalar>;
    const Real eps = std::numeric_limits<Real>::epsilon();
    auto g = [&](int r, int c) -> Scalar& { return G[c * ldg + r]; };
    auto w = [&](int r, int c) -> Scalar& { return W[c * ldw + r]; };
    for (int c = 0; c < p; ++c)
        for (int r = 0; r < p; ++r) w(r, c) = Scalar(r == c ? 1 : 0);
    for (int sweep = 0; sweep < 64; ++sweep) {
        bool rotated = false;
        for (int i = 0; i + 1 < p; ++i) {
            for (int k = i + 1; k < p; ++k) {
                Real a = 0, b = 0;
                Scalar gam = Scalar(0);
                for (int r = 0; r < p; ++r) {
                    a += device::abs2(g(r, i));
                    b += device::abs2(g(r, k));
                    gam += device::conj_if<true>(g(r, i)) * g(r, k);
                }
                const Real ag = sycl::sqrt(device::abs2(gam));
                if (ag == Real(0) || ag <= eps * sycl::sqrt(a) * sycl::sqrt(b)) continue;
                rotated = true;
                // Column k is first scaled by conj(gam / |gam|) so the pair's inner product is real.
                const Scalar ph = device::conj_if<true>(gam) / ag;
                const Real zeta = (b - a) / (Real(2) * ag);
                const Real t = (zeta >= Real(0) ? Real(1) : Real(-1)) /
                               (sycl::fabs(zeta) + sycl::sqrt(Real(1) + zeta * zeta));
                const Real cs = Real(1) / sycl::sqrt(Real(1) + t * t), sn = cs * t;
                for (int r = 0; r < p; ++r) {
                    const Scalar x = g(r, i), y = g(r, k) * ph;
                    g(r, i) = cs * x - sn * y;
                    g(r, k) = sn * x + cs * y;
                    const Scalar xw = w(r, i), yw = w(r, k) * ph;
                    w(r, i) = cs * xw - sn * yw;
                    w(r, k) = sn * xw + cs * yw;
                }
            }
        }
        if (!rotated) break;
    }
    Real smax = 0;
    for (int c = 0; c < p; ++c) {
        Real s2 = 0;
        for (int r = 0; r < p; ++r) s2 += device::abs2(g(r, c));
        flag[c] = Scalar(sycl::sqrt(s2));
        smax = sycl::fmax(smax, sycl::sqrt(s2));
    }
    const Real thresh = smax * Real(p) * eps;
    for (int c = 0; c < p; ++c) {
        const Real s = float_t<Scalar>(sycl::sqrt(device::abs2(flag[c])));
        if (s > thresh) {
            for (int r = 0; r < p; ++r) g(r, c) /= s;
            flag[c] = Scalar(1);
        } else {
            flag[c] = Scalar(0);
        }
    }
    // Deficient columns: the first unit vector whose residual against the kept columns is at least
    // sqrt(1/(2p)); one always exists because those residuals' squares sum to p - kept >= 1.
    for (int c = 0; c < p; ++c) {
        if (flag[c] != Scalar(0)) continue;
        for (int e = 0; e < p; ++e) {
            for (int r = 0; r < p; ++r) g(r, c) = Scalar(r == e ? 1 : 0);
            for (int pass = 0; pass < 2; ++pass) {
                for (int q = 0; q < p; ++q) {
                    if (q == c || flag[q] == Scalar(0)) continue;
                    Scalar dot = Scalar(0);
                    for (int r = 0; r < p; ++r) dot += device::conj_if<true>(g(r, q)) * g(r, c);
                    for (int r = 0; r < p; ++r) g(r, c) -= dot * g(r, q);
                }
            }
            Real s2 = 0;
            for (int r = 0; r < p; ++r) s2 += device::abs2(g(r, c));
            if (s2 * Real(2 * p) >= Real(1)) {
                const Real s = sycl::sqrt(s2);
                for (int r = 0; r < p; ++r) g(r, c) /= s;
                flag[c] = Scalar(1);
                break;
            }
        }
    }
    // R^H = U W^H, row by row in place over U: row i of the product reads only row i of U.
    for (int i = 0; i < p; ++i) {
        for (int k = 0; k < p; ++k) flag[k] = g(i, k);
        for (int j = 0; j < p; ++j) {
            Scalar acc = Scalar(0);
            for (int k = 0; k < p; ++k) acc += flag[k] * device::conj_if<true>(w(j, k));
            g(i, j) = acc;
        }
    }
}

template <Backend B, typename Scalar>
Event launch(Queue& ctx, const MV<Scalar>& V_old, Span<float_t<Scalar>> eigenvalues, const MV<Scalar>& V,
             float_t<Scalar> tol,
             Span<std::byte> workspace) {
    using Real = float_t<Scalar>;
    const int64_t n = V.rows(), m = V.cols(), batch = V.batch_size();
    BumpAllocator pool(workspace);
    const Workspace<Scalar> ws = carve<Scalar>(ctx.device(), pool, n, m, batch);
    // The workspace views are rebuilt from raw pointers, so they take the call's mask back on.
    const Span<const std::uint8_t> mask_span = detail::call_mask("syev_align", V_old, V);
    const std::uint8_t* mask = detail::mask_ptr(mask_span);
    const MV<Scalar> Mv = MV<Scalar>(ws.m.data(), int(m), int(m), int(m), int(m * m),
                                     int(batch)).with_batch_mask(mask_span);
    constexpr Transpose herm = complex_v<Scalar> ? Transpose::ConjTrans : Transpose::Trans;
    gemm<B, Scalar>(ctx, V_old, V, Mv, Scalar(1), Scalar(0), herm, Transpose::NoTrans);

    Scalar* Mp = ws.m.data();
    Scalar* Wp = ws.vcopy.data();
    Scalar* Fp = ws.scratch.data();
    int32_t* Sp = ws.start.data();
    int32_t* Ep = ws.end.data();
    const Real* lam = eigenvalues.data();
    ctx->parallel_for<ClusterRotationKernel<B, Scalar>>(sycl::range<1>(size_t(m * batch)), [=](sycl::id<1> id) {
        const int64_t b = int64_t(id[0]) / m, j = int64_t(id[0]) % m;
        if (mask != nullptr && mask[b] == 0) return;  // I3
        const Real* l = lam + b * m;
        const Real thr = tol * sycl::fmax(sycl::fabs(l[0]), sycl::fabs(l[m - 1]));
        if (j > 0 && l[j] - l[j - 1] <= thr) return;
        int64_t e = j + 1;
        while (e < m && l[e] - l[e - 1] <= thr) ++e;
        for (int64_t c = j; c < e; ++c) {
            Sp[b * m + c] = int32_t(j);
            Ep[b * m + c] = int32_t(e);
        }
        procrustes_block<Scalar>(Mp + b * m * m + j * m + j, m, Wp + b * n * m + j * n + j, n, Fp + b * m + j,
                                 int(e - j));
    });

    const MV<Scalar> Vc = MV<Scalar>(ws.vcopy.data(), int(n), int(m), int(n), int(n * m),
                                     int(batch)).with_batch_mask(mask_span);
    MV<Scalar>::copy(ctx, Vc, V);
    Scalar* Vp = V.data_ptr();
    const int64_t ldv = V.ld(), sv = V.stride();
    ctx->parallel_for<ClusterApplyKernel<B, Scalar>>(sycl::range<1>(size_t(n * m * batch)), [=](sycl::id<1> id) {
        const int64_t lin = int64_t(id[0]);
        const int64_t b = lin / (n * m), e = lin % (n * m);
        if (mask != nullptr && mask[b] == 0) return;  // I3
        const int64_t i = e % n, j = e / n;
        const int64_t s = Sp[b * m + j], t = Ep[b * m + j];
        const Scalar* vc = Wp + b * n * m;
        const Scalar* rh = Mp + b * m * m;
        Scalar acc = Scalar(0);
        for (int64_t q = s; q < t; ++q) acc += vc[q * n + i] * device::conj_if<true>(rh[q * m + j]);
        Vp[b * sv + j * ldv + i] = acc;
    });
    return ctx.get_event();
}

}  // namespace ops::syev_align

template <Backend B, typename Scalar>
Event syev_align(Queue& ctx, const MatrixView<Scalar, MatrixFormat::Dense>& V_old,
                 Span<typename base_type<Scalar>::type> eigenvalues, const MatrixView<Scalar, MatrixFormat::Dense>& V,
                 typename base_type<Scalar>::type tol, Span<std::byte> workspace) {
    (void)detail::call_mask("syev_align", V_old, V);
    syev_align_validate_params<Scalar>(V_old, eigenvalues, V);
    if (!(tol >= 0)) throw batchlas::invalid_argument("syev_align: tol must be >= 0, got " + std::to_string(tol));
    if (V.batch_size() == 0 || V.cols() == 0) return ctx.create_event_after_external_work();
    // gemm, rotation, copy and apply chain through ws.m / ws.vcopy, so they need an in-order queue.
    return detail::on_in_order_queue(
        ctx, [&](Queue& q) { return ops::syev_align::launch<B, Scalar>(q, V_old, eigenvalues, V, tol, workspace); });
}

template <Backend B, typename Scalar>
size_t syev_align_buffer_size(Queue& ctx, const MatrixView<Scalar, MatrixFormat::Dense>& V_old,
                              Span<typename base_type<Scalar>::type> eigenvalues,
                              const MatrixView<Scalar, MatrixFormat::Dense>& V) {
    (void)detail::call_mask("syev_align", V_old, V);
    syev_align_validate_params<Scalar>(V_old, eigenvalues, V);
    if (V.batch_size() == 0 || V.cols() == 0) return 0;
    BumpAllocator pool = BumpAllocator::measuring();
    (void)ops::syev_align::carve<Scalar>(ctx.device(), pool, V.rows(), V.cols(), V.batch_size());
    return pool.required_bytes();
}

#define SYEV_ALIGN_INSTANTIATE(B_, fp) \
    BATCHLAS_INSTANTIATE_OP(B_, fp, syev_align) BATCHLAS_INSTANTIATE_OP(B_, fp, syev_align_buffer_size)
BATCHLAS_INSTANTIATE_SCALAR_ALL_BACKENDS(SYEV_ALIGN_INSTANTIATE)
#undef SYEV_ALIGN_INSTANTIATE

}  // namespace batchlas
