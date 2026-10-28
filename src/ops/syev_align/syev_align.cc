// syev_align: eigenvector continuity (fullerenes plan P2.5). One gemm forms M = V_old^H V; one
// work-item per cluster runs a one-sided Jacobi SVD of M's diagonal block and stores R^H over it;
// a copy of V and one element-wise kernel apply V_c := V_c R. No kernel selection: there is one
// algorithm, no vendor counterpart, and the only routed work is the gemm, which selects for itself.

#include <batchlas/backend_config.h>

#include <batchlas/blas/functions/gemm.hh>
#include <batchlas/blas/functions/syev_align.hh>
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
    auto g = [&](int r, int c) -> T& { return G[c * ldg + r]; };
    auto w = [&](int r, int c) -> T& { return W[c * ldw + r]; };
    for (int c = 0; c < p; ++c)
        for (int r = 0; r < p; ++r) w(r, c) = T(r == c ? 1 : 0);
    for (int sweep = 0; sweep < 64; ++sweep) {
        bool rotated = false;
        for (int i = 0; i + 1 < p; ++i) {
            for (int k = i + 1; k < p; ++k) {
                Real a = 0, b = 0;
                T gam = T(0);
                for (int r = 0; r < p; ++r) {
                    a += abs2_of(g(r, i));
                    b += abs2_of(g(r, k));
                    gam += conj_of(g(r, i)) * g(r, k);
                }
                const Real ag = sycl::sqrt(abs2_of(gam));
                if (ag == Real(0) || ag <= eps * sycl::sqrt(a) * sycl::sqrt(b)) continue;
                rotated = true;
                // Column k is first scaled by conj(gam / |gam|) so the pair's inner product is real.
                const T ph = conj_of(gam) / ag;
                const Real zeta = (b - a) / (Real(2) * ag);
                const Real t = (zeta >= Real(0) ? Real(1) : Real(-1)) /
                               (sycl::fabs(zeta) + sycl::sqrt(Real(1) + zeta * zeta));
                const Real cs = Real(1) / sycl::sqrt(Real(1) + t * t), sn = cs * t;
                for (int r = 0; r < p; ++r) {
                    const T x = g(r, i), y = g(r, k) * ph;
                    g(r, i) = cs * x - sn * y;
                    g(r, k) = sn * x + cs * y;
                    const T xw = w(r, i), yw = w(r, k) * ph;
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
        for (int r = 0; r < p; ++r) s2 += abs2_of(g(r, c));
        flag[c] = T(sycl::sqrt(s2));
        smax = sycl::fmax(smax, sycl::sqrt(s2));
    }
    const Real thresh = smax * Real(p) * eps;
    for (int c = 0; c < p; ++c) {
        const Real s = float_t<T>(sycl::sqrt(abs2_of(flag[c])));
        if (s > thresh) {
            for (int r = 0; r < p; ++r) g(r, c) /= s;
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
            for (int r = 0; r < p; ++r) g(r, c) = T(r == e ? 1 : 0);
            for (int pass = 0; pass < 2; ++pass) {
                for (int q = 0; q < p; ++q) {
                    if (q == c || flag[q] == T(0)) continue;
                    T dot = T(0);
                    for (int r = 0; r < p; ++r) dot += conj_of(g(r, q)) * g(r, c);
                    for (int r = 0; r < p; ++r) g(r, c) -= dot * g(r, q);
                }
            }
            Real s2 = 0;
            for (int r = 0; r < p; ++r) s2 += abs2_of(g(r, c));
            if (s2 * Real(2 * p) >= Real(1)) {
                const Real s = sycl::sqrt(s2);
                for (int r = 0; r < p; ++r) g(r, c) /= s;
                flag[c] = T(1);
                break;
            }
        }
    }
    // R^H = U W^H, row by row in place over U: row i of the product reads only row i of U.
    for (int i = 0; i < p; ++i) {
        for (int k = 0; k < p; ++k) flag[k] = g(i, k);
        for (int j = 0; j < p; ++j) {
            T acc = T(0);
            for (int k = 0; k < p; ++k) acc += flag[k] * conj_of(w(j, k));
            g(i, j) = acc;
        }
    }
}

template <Backend B, class T>
Event launch(Queue& ctx, const MV<T>& V_old, Span<float_t<T>> eigenvalues, const MV<T>& V, float_t<T> tol,
             Span<std::byte> workspace) {
    using Real = float_t<T>;
    const int64_t n = V.rows(), m = V.cols(), batch = V.batch_size();
    BumpAllocator pool(workspace);
    const Workspace<T> ws = carve<T>(ctx.device(), pool, n, m, batch);
    const MV<T> Mv(ws.m.data(), int(m), int(m), int(m), int(m * m), int(batch));
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

    const MV<T> Vc(ws.vcopy.data(), int(n), int(m), int(n), int(n * m), int(batch));
    MV<T>::copy(ctx, Vc, V);
    T* Vp = V.data_ptr();
    const int64_t ldv = V.ld(), sv = V.stride();
    ctx->parallel_for<ClusterApplyKernel<B, T>>(sycl::range<1>(size_t(n * m * batch)), [=](sycl::id<1> id) {
        const int64_t lin = int64_t(id[0]);
        const int64_t b = lin / (n * m), e = lin % (n * m);
        const int64_t i = e % n, j = e / n;
        const int64_t s = Sp[b * m + j], t = Ep[b * m + j];
        const T* vc = Wp + b * n * m;
        const T* rh = Mp + b * m * m;
        T acc = T(0);
        for (int64_t q = s; q < t; ++q) acc += vc[q * n + i] * conj_of(rh[q * m + j]);
        Vp[b * sv + j * ldv + i] = acc;
    });
    return ctx.get_event();
}

}  // namespace ops::syev_align

template <Backend B, typename T>
Event syev_align(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& V_old,
                 Span<typename base_type<T>::type> eigenvalues, const MatrixView<T, MatrixFormat::Dense>& V,
                 typename base_type<T>::type tol, Span<std::byte> workspace) {
    detail::refuse_batch_mask("syev_align", V_old, V);
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
    detail::refuse_batch_mask("syev_align", V_old, V);
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
