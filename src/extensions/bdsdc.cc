// Bidiagonal divide-and-conquer SVD. This does NOT port LAPACK's dlasd0/2/3/4/8:
// it reduces the bidiagonal SVD to a symmetric tridiagonal eigenproblem that
// `stedc` already solves. For B upper bidiagonal with diagonal d and
// superdiagonal e, the Golub-Kahan-Jordan-Wielandt matrix [0 B^T; B 0] has
// eigenvalues +/- sigma_i, and under the perfect shuffle
// y = (v_0, u_0, v_1, u_1, ...) it is TRIDIAGONAL with a zero diagonal and
// off-diagonal (d_0, e_0, d_1, e_1, ..., e_{n-2}, d_{n-1}), 2n-1 entries.
// Nothing is squared, so the condition number is not either; the price is that
// the eigenvector matrix is 2n x 2n.
//
// The extraction is exact, not heuristic: for sigma != 0 the eigenvector of T is
// forced to the interleaved form (alpha v; beta u), so splitting it into even and
// odd rows and normalising each half recovers v and u even when the +/- pair is
// numerically mixed. The exception is the null space, where the whole degenerate
// subspace mixes and the columns come back full-norm and individually plausible
// while being parallel to each other; those are rebuilt by Gram-Schmidt, safe
// because any orthonormal completion satisfies B = U S V^T to within 2*sigma.

#include <batchlas/blas/extensions.hh>
#include <batchlas/backend_config.h>

#include <batchlas/util/mempool.hh>

#include "../math-helpers.hh"
#include "../queue.hh"
#include "../util/template-instantiations.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace batchlas {

namespace {

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
    // the eigenvector view it is handed, so a null view would fault, and D&C needs
    // the sub-problem eigenvectors to form the secular vector regardless.
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
                    int32_t batch) {
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
            for (int32_t i = 0; i < N; ++i) {
                GD(i, b) = T(0);
            }
            // off-diagonal f[2k] = d_k (couples v_k,u_k), f[2k+1] = e_k (couples u_k,v_{k+1})
            for (int32_t k = 0; k < nn; ++k) {
                GE(2 * k, b) = D(k, b);
                if (k < nn - 1) {
                    GE(2 * k + 1, b) = E(k, b);
                }
            }
            GS(0, b) = T(1);
            for (int32_t i = 0; i < N - 1; ++i) {
                const T fi = GE(i, b);
                const T sgn = (fi >= T(0)) ? T(1) : T(-1);
                GS(i + 1, b) = GS(i, b) * sgn;
                GE(i, b) = sycl::fabs(fi);
            }
        });
    });
}

// sigma_i = lambda[N-1-i] (descending) or lambda[N-n+i] (ascending).
template <typename T>
inline int32_t bdsdc_src_column(int32_t i, int32_t n, bool sort_desc) {
    return sort_desc ? (2 * n - 1 - i) : (n + i);
}

template <Backend B, typename T>
void bdsdc_extract_values(Queue& ctx,
                          const BdsdcWorkspace<T>& ws,
                          Span<T> sigma,
                          int32_t n,
                          int32_t batch,
                          bool sort_desc) {
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
            const int32_t src = desc ? (2 * nn - 1 - i) : (nn + i);
            const T lam = L(src, b);
            out[static_cast<size_t>(b) * static_cast<size_t>(nn) + i] = sycl::fmax(lam, T(0));
        });
    });
}

// One work-group per (batch item, output column): split the eigenvector into its
// even rows (-> v) and odd rows (-> u) and normalise each half. The trailing
// sigma ~ 0 columns are rebuilt by the repair pass that follows.
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
                           bool sort_desc) {
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

        h.parallel_for<BdsdcExtract<B, T>>(
            sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(batch) * static_cast<size_t>(n) * kGroup),
                              sycl::range<1>(kGroup)),
            [=](sycl::nd_item<1> it) {
            const int32_t grp = static_cast<int32_t>(it.get_group(0));
            const int32_t lid = static_cast<int32_t>(it.get_local_id(0));
            const int32_t b = grp / nn;
            const int32_t i = grp - b * nn;
            const int32_t src = desc ? (2 * nn - 1 - i) : (nn + i);

            T p2 = T(0);
            T q2 = T(0);
            for (int32_t r = lid; r < nn; r += kGroup) {
                const T pv = Zk(2 * r, src, b);
                const T qv = Zk(2 * r + 1, src, b);
                p2 += pv * pv;
                q2 += qv * qv;
            }
            p2 = sycl::reduce_over_group(it.get_group(), p2, sycl::plus<T>());
            q2 = sycl::reduce_over_group(it.get_group(), q2, sycl::plus<T>());
            const T pn = sycl::sqrt(p2);
            const T qn = sycl::sqrt(q2);

            if (lid == 0) {
                sig[static_cast<size_t>(b) * static_cast<size_t>(nn) + i] = sycl::fmax(L(src, b), T(0));
            }

            // Guard the amplification, not just division by zero. The exact half-norm is
            // 1/sqrt(2); a half that has been rotated away can come back at 1e-20, and
            // 1/1e-20 writes +-inf into the column, which turns every later dot product
            // into NaN and makes the repair below reject all n candidate axes. Below this
            // bound the half has vanished, so zero it and let the repair rebuild.
            const T half_present = T(0.0625);
            const T pinv = (pn > half_present) ? (T(1) / pn) : T(0);
            const T qinv = (qn > half_present) ? (T(1) / qn) : T(0);
            for (int32_t r = lid; r < nn; r += kGroup) {
                if (wv) {
                    // Vh is V^T: row i, column r.
                    Vh(i, r, b) = S(2 * r, b) * Zk(2 * r, src, b) * pinv;
                }
                if (wu) {
                    U(r, i, b) = S(2 * r + 1, b) * Zk(2 * r + 1, src, b) * qinv;
                }
            }
        });
    });
}

// Rebuild the singular vectors of the numerically-zero singular values.
//
// The criterion is sigma, not the shape of the extracted column: a degenerate
// block comes back full-norm and individually plausible while its columns are
// parallel to each other, so nothing local sees it. The candidate axis is ranked,
// not searched -- the residual of e_c against an orthonormal set Q is exactly
// 1 - sum_t Q(c,t)^2, so one pass ranks every axis, where a cursor walk rejects
// most of them one full n-projection at a time. The serial j loop runs only for
// degenerate columns: a bidiagonal of full numerical rank reads sigma and exits.
template <Backend B, typename T>
void bdsdc_repair_degenerate(Queue& ctx,
                             const MatrixView<T, MatrixFormat::Dense>& u,
                             const MatrixView<T, MatrixFormat::Dense>& vh,
                             Span<T> sigma,
                             int32_t n,
                             int32_t batch,
                             bool want_u,
                             bool want_vh) {
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
        // +sigma and -sigma stop being resolvable once 2*sigma reaches the eigenvalue
        // noise floor of the 2n problem, and both halves then normalise to the same
        // (v, u), leaving U with two parallel columns. Column i contributes
        // orthogonality error ~eps*sigma_max/(2*sigma_i), so holding that below a
        // target tau requires sigma_i >= eps*sigma_max/(2*tau): the multiplier is
        // derived from tau, not tuned, and must NOT carry a factor of n -- an
        // n-scaled tolerance discards singular values that still carry information
        // and fires on ordinary random matrices. It does not fix the singular
        // VALUES, which keep gebrd's own eps*||A|| floor; use one-sided Jacobi.
        constexpr T kOrthTarget = T(1e-3);
        const T tol_factor =
            (T(1) / (T(2) * kOrthTarget)) * std::numeric_limits<T>::epsilon();

        h.parallel_for<BdsdcRepair<B, T>>(
            sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(batch) * kGroup),
                              sycl::range<1>(kGroup)),
            [=](sycl::nd_item<1> it) {
            const int32_t b = static_cast<int32_t>(it.get_group(0));
            const int32_t lid = static_cast<int32_t>(it.get_local_id(0));
            auto grp = it.get_group();
            const T* sb = sig + static_cast<size_t>(b) * static_cast<size_t>(nn);

            T smax = T(0);
            for (int32_t i = 0; i < nn; ++i) smax = sycl::fmax(smax, sb[i]);
            const T tol = smax * tol_factor;

            int32_t ndeg = 0;
            for (int32_t i = 0; i < nn; ++i) ndeg += (sb[i] <= tol) ? 1 : 0;
            if (ndeg == 0) return;

            // which = 0 -> rebuild V (rows of Vh), which = 1 -> rebuild U (cols of U)
            for (int32_t which = 0; which < 2; ++which) {
                if (which == 0 && !wv) continue;
                if (which == 1 && !wu) continue;

                for (int32_t j = 0; j < nn; ++j) {
                    if (sb[j] > tol) continue;

                    // Rank every candidate axis by the residual it would leave against the
                    // columns already final: the earlier rebuilt ones (t < j) and ALL
                    // non-degenerate ones, including those after j. Skipping the latter leaves
                    // the rebuilt vectors orthogonal to each other but not to the good columns
                    // whenever the degenerate block is not at the end, i.e. ascending order.
                    T best = std::numeric_limits<T>::max();
                    int32_t best_c = lid;
                    for (int32_t c = lid; c < nn; c += kGroup) {
                        T acc = T(0);
                        for (int32_t t = 0; t < nn; ++t) {
                            if (t == j) continue;
                            if (t > j && sb[t] <= tol) continue;
                            const T q = (which == 0) ? Vh(t, c, b) : U(c, t, b);
                            acc += q * q;
                        }
                        if (acc < best) { best = acc; best_c = c; }
                    }
                    const T best_all = sycl::reduce_over_group(grp, best, sycl::minimum<T>());
                    const int32_t c_star = sycl::reduce_over_group(
                        grp, (best == best_all) ? best_c : nn, sycl::minimum<int32_t>());

                    // Seed w := e_c*, project out the final columns, twice -- one
                    // pass is not enough when e_c* is nearly in their span.
                    for (int32_t r = lid; r < nn; r += kGroup) {
                        const T seed = (r == c_star) ? T(1) : T(0);
                        if (which == 0) { Vh(j, r, b) = seed; } else { U(r, j, b) = seed; }
                    }
                    it.barrier(sycl::access::fence_space::global_and_local);

                    for (int32_t pass = 0; pass < 2; ++pass) {
                        for (int32_t t = 0; t < nn; ++t) {
                            if (t == j) continue;
                            if (t > j && sb[t] <= tol) continue;
                            T dot = T(0);
                            for (int32_t r = lid; r < nn; r += kGroup) {
                                const T wr = (which == 0) ? Vh(j, r, b) : U(r, j, b);
                                const T tr = (which == 0) ? Vh(t, r, b) : U(r, t, b);
                                dot += wr * tr;
                            }
                            dot = sycl::reduce_over_group(grp, dot, sycl::plus<T>());
                            for (int32_t r = lid; r < nn; r += kGroup) {
                                const T tr = (which == 0) ? Vh(t, r, b) : U(r, t, b);
                                if (which == 0) { Vh(j, r, b) -= dot * tr; }
                                else            { U(r, j, b) -= dot * tr; }
                            }
                            it.barrier(sycl::access::fence_space::global_and_local);
                        }
                    }

                    T acc = T(0);
                    for (int32_t r = lid; r < nn; r += kGroup) {
                        const T wr = (which == 0) ? Vh(j, r, b) : U(r, j, b);
                        acc += wr * wr;
                    }
                    const T nrm2 = sycl::reduce_over_group(grp, acc, sycl::plus<T>());
                    // c_star is the best axis available, so a residual too small to
                    // normalise means the final columns already span the space. Leave the
                    // column as extracted rather than amplifying noise.
                    if (nrm2 > std::numeric_limits<T>::min()) {
                        const T inv = T(1) / sycl::sqrt(nrm2);
                        for (int32_t r = lid; r < nn; r += kGroup) {
                            if (which == 0) { Vh(j, r, b) *= inv; } else { U(r, j, b) *= inv; }
                        }
                    }
                    it.barrier(sycl::access::fence_space::global_and_local);
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
            bool sort_desc) {
    if constexpr (internal::is_complex<T>::value) {
        throw std::runtime_error("bdsdc: complex types are not implemented");
    } else {
        const int32_t n = static_cast<int32_t>(d.size());
        const int32_t batch = static_cast<int32_t>(d.batch_size());
        if (n <= 0) {
            return ctx.get_event();
        }
        if (!ctx.in_order()) {
            throw std::runtime_error("bdsdc: requires an in-order Queue");
        }
        if (static_cast<int32_t>(singular_values_out.size()) < n * batch) {
            throw std::invalid_argument("bdsdc: singular_values span too small");
        }

        const bool want_u = u.data_ptr() != nullptr && u.rows() > 0 && u.cols() >= n;
        const bool want_vh = vh.data_ptr() != nullptr && vh.cols() > 0 && vh.rows() >= n;
        const bool want_vectors = want_u || want_vh;

        Span<std::byte> ws_mut(const_cast<std::byte*>(ws.data()), ws.size());
        BumpAllocator pool(ws_mut);
        auto layout = bdsdc_layout<B, T>(ctx, pool, n, batch, want_vectors);

        bdsdc_build_gk<B, T>(ctx, d, e, layout, n, batch);

        stedc<B, T>(ctx,
                    layout.gk_d,
                    layout.gk_e,
                    layout.lambda,
                    layout.stedc_ws,
                    JobType::EigenVectors,
                    bdsdc_stedc_params<T>(),
                    layout.Z);

        if (!want_vectors) {
            bdsdc_extract_values<B, T>(ctx, layout, singular_values_out, n, batch, sort_desc);
            return ctx.get_event();
        }

        bdsdc_extract_vectors<B, T>(ctx, layout, singular_values_out, u, vh,
                                    n, batch, want_u, want_vh, sort_desc);
        bdsdc_repair_degenerate<B, T>(ctx, u, vh, singular_values_out, n, batch, want_u, want_vh);
        return ctx.get_event();
    }
}

template <Backend B, typename T>
Event bdsdc(Queue& ctx,
            const VectorView<T>& d,
            const VectorView<T>& e,
            Span<T> singular_values_out,
            const Span<std::byte>& ws,
            bool sort_desc) {
    return bdsdc<B, T>(ctx, d, e, singular_values_out, ws,
                       MatrixView<T, MatrixFormat::Dense>(nullptr, 0, 0, 1, 1, d.batch_size()),
                       MatrixView<T, MatrixFormat::Dense>(nullptr, 0, 0, 1, 1, d.batch_size()),
                       sort_desc);
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
        bool); \
    template Event bdsdc<back, BATCHLAS_UNPAREN fp>( \
        Queue&, \
        const VectorView<BATCHLAS_UNPAREN fp>&, \
        const VectorView<BATCHLAS_UNPAREN fp>&, \
        Span<BATCHLAS_UNPAREN fp>, \
        const Span<std::byte>&, \
        bool); \
    template size_t bdsdc_buffer_size<back, BATCHLAS_UNPAREN fp>( \
        Queue&, \
        const VectorView<BATCHLAS_UNPAREN fp>&, \
        const VectorView<BATCHLAS_UNPAREN fp>&, \
        Span<BATCHLAS_UNPAREN fp>, \
        bool);

#define BDSDC_INSTANTIATE_FOR_BACKEND(back) \
    BATCHLAS_FOR_EACH_REAL_TYPE_1(BDSDC_INSTANTIATE, back)

#if BATCHLAS_HAS_CUDA_BACKEND
BDSDC_INSTANTIATE_FOR_BACKEND(Backend::CUDA)
#endif

#if BATCHLAS_HAS_ROCM_BACKEND
BDSDC_INSTANTIATE_FOR_BACKEND(Backend::ROCM)
#endif

#if BATCHLAS_HAS_HOST_BACKEND
BDSDC_INSTANTIATE_FOR_BACKEND(Backend::NETLIB)
#endif

#undef BDSDC_INSTANTIATE_FOR_BACKEND
#undef BDSDC_INSTANTIATE

} // namespace batchlas
