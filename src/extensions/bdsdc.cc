// Bidiagonal divide-and-conquer SVD.
//
// This does NOT port LAPACK's dlasd0/2/3/4/8. It reduces the bidiagonal SVD to a
// symmetric tridiagonal eigenproblem that `stedc` already solves, via the
// Golub-Kahan-Jordan-Wielandt form. For B upper bidiagonal with diagonal d and
// superdiagonal e,
//
//     T = [ 0    B^T ]      is symmetric with eigenvalues +/- sigma_i,
//         [ B    0   ]
//
// and under the perfect-shuffle permutation y = (v_0, u_0, v_1, u_1, ...) it is
// TRIDIAGONAL with a zero diagonal and off-diagonal
//
//     (d_0, e_0, d_1, e_1, ..., e_{n-2}, d_{n-1})        (2n-1 entries).
//
// Why this and not a real dlasd4:
//
//   * It never squares anything. The n > 32 accuracy defect this replaces came
//     from forming the tridiagonal of B^T B, which squares the condition number;
//     T's eigenvalues are +/- sigma, not sigma^2.
//   * `stedc` here is the tuned, tested, batched D&C the library already ships.
//     A native dlasd4 would need its own secular equation (the factored
//     (d_j - sigma)(d_j + sigma) form), its own two-kind deflation, its own
//     Loewner vector recomputation -- ~2000 lines of new device code with no
//     reuse, to solve a problem of half the size.
//   * The cost of the 2n problem is the point. Measured, batch=512, float,
//     eigenvectors, RTX 4090: stedc at 2n costs 0.85 ms (n=64), 3.19 ms
//     (n=128), 11.1 ms (n=256) -- against bdsqr's 643 / 3255 / 24388 ms for the
//     same bidiagonal problems. Solving twice the size with the right algorithm
//     beats solving the right size with a sequential one by ~1000x.
//
// The price is memory: the eigenvector matrix is 2n x 2n, so 4x what a native
// bdsdc would allocate.
//
// EXACTNESS OF THE EXTRACTION. For sigma != 0 the eigenvector of T for +sigma is
// exactly (v; u)/sqrt(2) interleaved -- there is no freedom, because
// T(av; bu) = (b sigma v; a sigma u) forces a = b. So splitting the computed
// eigenvector into its even and odd rows and normalising each half is exact, not
// a heuristic. It stays exact even when +sigma and -sigma are numerically mixed:
// a mixed vector is cos(t)(v;u)/sqrt2 + sin(t)(v;-u)/sqrt2 = ((c+s)v; (c-s)u)/sqrt2,
// still of the form (alpha v; beta u), so half-normalisation recovers v and u
// exactly. The only residue is a possible sign flip on u when |t| > 45 deg, which
// costs at most 2*sigma in the residual -- and |t| only approaches 45 deg when
// sigma is at the level of eps*||B||, where 2*sigma is negligible.
//
// The one real failure mode is the NULL SPACE. At sigma ~ 0 the +/- pair is
// degenerate, and with k zero singular values the whole 2k-dimensional space
// span{(v_j;0), (0;u_j)} is degenerate at once: its computed basis is
// [N_v 0; 0 N_u] W for an arbitrary orthogonal W, so the k columns we select
// carry halves N_v*W_top and N_u*W_bot whose columns need NOT be mutually
// orthogonal. Two things follow, both learned the hard way:
//
//   * Detecting this by a vanishing half-norm does not work. Measured, n=16 with
//     two zero singular values: every column came back with norm exactly 1.000
//     and both halves healthy, yet U(14).U(15) = 1.0 -- two identical columns,
//     with V perfectly orthogonal. Nothing vanished; orthogonality was lost
//     BETWEEN the degenerate columns. The criterion has to be sigma itself.
//   * Repairing from the partner (-sigma) column does not work either. It is
//     wrong for well-separated sigma (both partners have equal half-norms, so the
//     tie-break flips u's sign and destroys the reconstruction -- measured 1.18
//     relative on a matrix with no zero singular values at all), and it still
//     fails once the null space has dimension >= 3.
//
// So every column with sigma <= tol is rebuilt by Gram-Schmidt against the
// columns that are already final. Those vectors are arbitrary anyway: any
// orthonormal completion satisfies B = U S V^T to within 2*sigma <= 2*tol.

#include <blas/extensions.hh>
#include <batchlas/backend_config.h>

#include <util/mempool.hh>

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

// stedc is happier with a non-negative off-diagonal, and the Blocked gesvd path
// already normalises that way before calling it. A tridiagonal with signed
// off-diagonals is similar to the |.| one through S = diag(s), s_0 = 1,
// s_{i+1} = s_i * sign(f_i): T' = S T S has the same eigenvalues and z = S z'.
// Here the sign vector is folded straight into the extraction, so it costs no
// extra pass over the eigenvector matrix.
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

    // Z is allocated on both paths. stedc unconditionally clears the eigenvector
    // view it is handed, so a null view would fault; and divide-and-conquer needs
    // the sub-problem eigenvectors to form the secular vector even when only
    // eigenvalues are wanted, so there is nothing to save by omitting it.
    static_cast<void>(want_vectors);
    auto z_span = pool.allocate<T>(ctx, static_cast<size_t>(N) * static_cast<size_t>(N) * nb);
    ws.Z = MatrixView<T, MatrixFormat::Dense>(z_span.data(), N, N, N,
                                              static_cast<int64_t>(N) * static_cast<int64_t>(N), batch);

    ws.stedc_ws = pool.allocate<std::byte>(
        ctx, stedc_workspace_size<B, T>(ctx, static_cast<size_t>(N), nb, JobType::EigenVectors,
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
// even rows (-> v) and odd rows (-> u) and normalise each half. Exact for every
// sigma above the noise floor; the trailing sigma ~ 0 columns are rebuilt by the
// repair pass that follows.
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

            // Guard the amplification, not just division by zero. The exact
            // half-norm is 1/sqrt(2); a half that has been rotated away can come
            // back at 1e-20, and 1/1e-20 writes +-inf into the column. Every
            // subsequent dot product is then NaN, `NaN > accept2` is false, and
            // the repair below never accepts a candidate -- it grinds through all
            // n cursor positions for that column instead. Measured at n=256,
            // batch=512: 93 ms of a 162 ms solve, and asymmetric (U only), which
            // is what gave it away. Anything below this bound is a vanished half
            // whose sigma is at the noise floor, so zero it and let the repair
            // rebuild the column.
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
// Criterion is sigma, not the shape of the extracted column. With k zero
// singular values the entire 2k-dimensional degenerate space mixes, so the
// selected columns come back full-norm and individually plausible while being
// parallel to each other -- measured: two identical U columns with V perfectly
// orthogonal. Only sigma sees that coming.
//
// The replacement vectors are arbitrary: for sigma <= tol, any orthonormal
// completion reconstructs B just as well, to within 2*sigma.
//
// The candidate axis is CHOSEN, not searched. The obvious implementation walks a
// cursor over e_0, e_1, ... and keeps the first whose residual survives
// orthogonalisation, but the missing direction is usually localised -- for an
// upper bidiagonal the left null vector sits at the END of the index range -- so
// the walk rejects ~250 of 256 axes before finding it, each rejection costing a
// full n-projection pass. Measured at n=256, batch=512: V accepted on attempt 1,
// U on attempt 251, and that asymmetry alone was 93 ms of a 162 ms solve. The
// residual of e_c against an orthonormal set Q is exactly 1 - sum_t Q(c,t)^2, so
// one pass over the final columns ranks every axis at once and the best one is
// taken directly.
//
// One work-group per batch item; the j loop is serial because column j must be
// orthogonalised against the columns already final, but it only runs for
// degenerate columns -- for a bidiagonal of full numerical rank this kernel reads
// sigma and exits.
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
        // +sigma and -sigma stop being resolvable once 2*sigma drops to the
        // eigenvalue noise floor of the 2n problem, ~eps*||T||.
        //
        // The constant must NOT carry a factor of n. With 8*n*eps this is 2.4e-4
        // relative at n=256 in float, which sweeps in singular values that carry
        // real information: their vectors get thrown away and rebuilt as
        // arbitrary orthonormal ones. It also made this kernel fire on ordinary
        // random matrices. 16*eps sits an order of magnitude above where an exact
        // zero lands (measured 1.2e-16 relative in double) and orders below any
        // sigma worth keeping.
        const T tol_factor = static_cast<T>(16) * std::numeric_limits<T>::epsilon();

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

                    // Rank every candidate axis by the residual it would leave,
                    // 1 - sum_t Q(c,t)^2 over the columns already final: the
                    // earlier rebuilt ones (t < j) and ALL non-degenerate ones,
                    // including those after j. Skipping the latter would leave the
                    // rebuilt vectors orthogonal to each other but not to the good
                    // columns whenever the degenerate block is not at the end --
                    // which is the case when the caller asks for ascending order.
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
                    // c_star is the best axis available, so a residual too small
                    // to normalise means the final columns already span the space
                    // -- there is no better choice to fall back to. Leave the
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
