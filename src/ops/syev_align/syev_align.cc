// syev_align: eigenvector continuity (fullerenes plan P2.5). One gemm forms M = V_old^H V; one
// work-item per cluster runs a one-sided Jacobi SVD of M's diagonal block and stores R^H over it;
// a copy of V and one element-wise kernel apply V_c := V_c R. No kernel selection: there is one
// algorithm, no vendor counterpart, and the only routed work is the gemm, which selects for itself.

#include <batchlas/backend_config.h>

#include <batchlas/blas/functions/gemm.hh>
#include <batchlas/blas/functions/syev_align.hh>
#include <batchlas/blas/device/v2/ext/tridiag-stebz-stein.hh>
#include <batchlas/util/mempool.hh>

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

namespace v2 = ::batchlas::device::v2;

template <class T>
using MV = MatrixView<T, MatrixFormat::Dense>;

template <Backend B, class T>
struct ClusterRotationKernel {};
template <Backend B, class T>
struct ClusterApplyKernel {};

template <class T>
inline constexpr bool complex_v = is_std_complex_v<T>;

template <class T>
inline T conj_of(const T& x) {
    if constexpr (complex_v<T>) return T(x.real(), -x.imag());
    else return x;
}

template <class T>
inline float_t<T> abs2_of(const T& x) {
    if constexpr (complex_v<T>) return x.real() * x.real() + x.imag() * x.imag();
    else return x * x;
}

template <class T>
struct Workspace {
    Span<T> m;            // m x m per item: M = V_old^H V, then R^H on each cluster's diagonal block
    Span<T> vcopy;        // n x m per item: W of the Jacobi SVD (block (s, s), ld n), then a copy of V
    Span<T> scratch;      // m per item: column flags, then one row of U
    Span<int32_t> start;  // m per item: first column of the column's cluster
    Span<int32_t> end;    // m per item: one past its last column
};

// One description for the call and the size query (a mismatch overruns the workspace).
template <class T>
Workspace<T> carve(const Device& dev, BumpAllocator& pool, int64_t n, int64_t m, int64_t batch) {
    Workspace<T> w;
    w.m = pool.allocate<T>(dev, size_t(m * m * batch));
    w.vcopy = pool.allocate<T>(dev, size_t(n * m * batch));
    w.scratch = pool.allocate<T>(dev, size_t(m * batch));
    w.start = pool.allocate<int32_t>(dev, size_t(m * batch));
    w.end = pool.allocate<int32_t>(dev, size_t(m * batch));
    return w;
}

// G is the cluster's p x p block of M (ld m), W the accumulated right rotations (ld n). On return
// G holds R^H = U W^H, where B = U Sigma W^H. Columns of U whose singular value is below
// p eps sigma_max are completed by Gram-Schmidt on unit vectors, so R stays unitary.
template <class T>
inline void procrustes_block(T* G, int64_t ldg, T* W, int64_t ldw, T* flag, int p) {
    using Real = float_t<T>;
    const Real eps = std::numeric_limits<Real>::epsilon();
    const v2::Thread th{};
    const auto g = v2::mat(G, p, p, int(ldg));
    const auto w = v2::mat(W, p, p, int(ldw));
    const auto f = v2::vec(flag, p);
    v2::laset(th, v2::Zero{}, v2::One{}, w);
    // One-sided Jacobi: rotate column pairs of B (and of W) until they are orthogonal.
    for (int sweep = 0; sweep < 64; ++sweep) {
        bool rotated = false;
        for (int i = 0; i + 1 < p; ++i) {
            for (int k = i + 1; k < p; ++k) {
                const auto gr = v2::gram2(th, g.col(i), g.col(k));
                const Real a = gr.xx, b = gr.yy;
                const T gam = gr.xy;
                const Real ag = sycl::sqrt(abs2_of(gam));
                if (ag == Real(0) || ag <= eps * sycl::sqrt(a) * sycl::sqrt(b)) continue;
                rotated = true;
                // Column k is first scaled by conj(gam / |gam|) so the pair's inner product is real.
                const T ph = conj_of(gam) / ag;
                const Real zeta = (b - a) / (Real(2) * ag);
                const Real t = (zeta >= Real(0) ? Real(1) : Real(-1)) /
                               (sycl::fabs(zeta) + sycl::sqrt(Real(1) + zeta * zeta));
                const Real cs = Real(1) / sycl::sqrt(Real(1) + t * t), sn = cs * t;
                v2::rot_phase(th, cs, sn, ph, g.col(i), g.col(k));
                v2::rot_phase(th, cs, sn, ph, w.col(i), w.col(k));
            }
        }
        if (!rotated) break;
    }
    Real smax = 0;
    for (int c = 0; c < p; ++c) {
        const Real s2 = v2::sumsq(th, g.col(c));
        flag[c] = T(sycl::sqrt(s2));
        smax = sycl::fmax(smax, sycl::sqrt(s2));
    }
    const Real thresh = smax * Real(p) * eps;
    for (int c = 0; c < p; ++c) {
        const Real s = float_t<T>(sycl::sqrt(abs2_of(flag[c])));
        if (s > thresh) {
            v2::rscal_div(th, s, g.col(c));
            flag[c] = T(1);
        } else {
            flag[c] = T(0);
        }
    }
    // Deficient columns: the first unit vector whose residual against the kept columns is at least
    // sqrt(1/(2p)); one always exists because those residuals' squares sum to p - kept >= 1.
    for (int c = 0; c < p; ++c) {
        if (flag[c] != T(0)) continue;
        for (int e = 0; e < p; ++e) {
            v2::fill(th, v2::Zero{}, g.col(c));
            g(e, c) = T(1);
            for (int pass = 0; pass < 2; ++pass) {
                for (int q = 0; q < p; ++q) {
                    if (q == c || flag[q] == T(0)) continue;
                    const T dot = v2::dotc(th, g.col(q), g.col(c));
                    v2::axpy(th, -dot, g.col(q), g.col(c));
                }
            }
            const Real s2 = v2::sumsq(th, g.col(c));
            if (s2 * Real(2 * p) >= Real(1)) {
                v2::rscal_div(th, sycl::sqrt(s2), g.col(c));
                flag[c] = T(1);
                break;
            }
        }
    }
    // R^H = U W^H, row by row in place over U: row i of the product reads only row i of U.
    for (int i = 0; i < p; ++i) {
        v2::copy(th, g.row(i), f);
        for (int j = 0; j < p; ++j) g(i, j) = v2::dotc(th, w.row(j), f);
    }
}

template <Backend B, class T>
Event launch(Queue& ctx, const MV<T>& V_old, Span<float_t<T>> eigenvalues, const MV<T>& V, float_t<T> tol,
             Span<std::byte> workspace) {
    using Real = float_t<T>;
    const int64_t n = V.rows(), m = V.cols(), batch = V.batch_size();
    BumpAllocator pool(workspace);
    const Workspace<T> ws = carve<T>(ctx.device(), pool, n, m, batch);
    // The workspace views are rebuilt from raw pointers, so they take the call's mask back on.
    const Span<const std::uint8_t> mask_span = detail::call_mask("syev_align", V_old, V);
    const std::uint8_t* mask = detail::mask_ptr(mask_span);
    const MV<T> Mv = MV<T>(ws.m.data(), int(m), int(m), int(m), int(m * m), int(batch)).with_batch_mask(mask_span);
    constexpr Transpose herm = complex_v<T> ? Transpose::ConjTrans : Transpose::Trans;
    gemm<B, T>(ctx, V_old, V, Mv, T(1), T(0), herm, Transpose::NoTrans);

    T* Mp = ws.m.data();
    T* Wp = ws.vcopy.data();
    T* Fp = ws.scratch.data();
    int32_t* Sp = ws.start.data();
    int32_t* Ep = ws.end.data();
    const Real* lam = eigenvalues.data();
    ctx->parallel_for<ClusterRotationKernel<B, T>>(sycl::range<1>(size_t(m * batch)), [=](sycl::id<1> id) {
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
        procrustes_block<T>(Mp + b * m * m + j * m + j, m, Wp + b * n * m + j * n + j, n, Fp + b * m + j,
                            int(e - j));
    });

    const MV<T> Vc = MV<T>(ws.vcopy.data(), int(n), int(m), int(n), int(n * m), int(batch)).with_batch_mask(mask_span);
    MV<T>::copy(ctx, Vc, V);
    T* Vp = V.data_ptr();
    const int64_t ldv = V.ld(), sv = V.stride();
    ctx->parallel_for<ClusterApplyKernel<B, T>>(sycl::range<1>(size_t(n * m * batch)), [=](sycl::id<1> id) {
        const int64_t lin = int64_t(id[0]);
        const int64_t b = lin / (n * m), e = lin % (n * m);
        if (mask != nullptr && mask[b] == 0) return;  // I3
        const int64_t i = e % n, j = e / n;
        const int64_t s = Sp[b * m + j], t = Ep[b * m + j];
        // V(i, j) = Vc(i, s:t) R^H(j, s:t)^H: row i of the copy against row j of R^H.
        const auto vc = v2::vec(Wp + b * n * m + s * n + i, int(t - s), int(n));
        const auto rh = v2::vec(Mp + b * m * m + s * m + j, int(t - s), int(m));
        Vp[b * sv + j * ldv + i] = v2::dotc(v2::Thread{}, rh, vc);
    });
    return ctx.get_event();
}

}  // namespace ops::syev_align

template <Backend B, typename T>
Event syev_align(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& V_old,
                 Span<typename base_type<T>::type> eigenvalues, const MatrixView<T, MatrixFormat::Dense>& V,
                 typename base_type<T>::type tol, Span<std::byte> workspace) {
    (void)detail::call_mask("syev_align", V_old, V);
    syev_align_validate_params<T>(V_old, eigenvalues, V);
    if (!(tol >= 0)) throw batchlas::invalid_argument("syev_align: tol must be >= 0, got " + std::to_string(tol));
    if (V.batch_size() == 0 || V.cols() == 0) return ctx.create_event_after_external_work();
    // gemm, rotation, copy and apply chain through ws.m / ws.vcopy, so they need an in-order queue.
    return detail::on_in_order_queue(
        ctx, [&](Queue& q) { return ops::syev_align::launch<B, T>(q, V_old, eigenvalues, V, tol, workspace); });
}

template <Backend B, typename T>
size_t syev_align_buffer_size(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& V_old,
                              Span<typename base_type<T>::type> eigenvalues,
                              const MatrixView<T, MatrixFormat::Dense>& V) {
    (void)detail::call_mask("syev_align", V_old, V);
    syev_align_validate_params<T>(V_old, eigenvalues, V);
    if (V.batch_size() == 0 || V.cols() == 0) return 0;
    BumpAllocator pool = BumpAllocator::measuring();
    (void)ops::syev_align::carve<T>(ctx.device(), pool, V.rows(), V.cols(), V.batch_size());
    return pool.required_bytes();
}

#define SYEV_ALIGN_INSTANTIATE(B_, fp) \
    BATCHLAS_INSTANTIATE_OP(B_, fp, syev_align) BATCHLAS_INSTANTIATE_OP(B_, fp, syev_align_buffer_size)
BATCHLAS_INSTANTIATE_SCALAR_ALL_BACKENDS(SYEV_ALIGN_INSTANTIATE)
#undef SYEV_ALIGN_INSTANTIATE

}  // namespace batchlas
