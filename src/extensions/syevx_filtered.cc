// syevx_filtered: Chebyshev-filtered subspace iteration.
//
// This is SYEVX_PLAN.md Tier 3. One outer iteration is
//
//     Y  = p_m(A) X          Chebyshev filter, m matvecs
//     Y  = ortho(Y)
//     H  = Y^H A Y           m x m projected problem
//     H  = Z diag(theta) Z^H syev
//     X  = Y Z               Ritz vectors
//
// The filter is the whole point: p_m is the degree-m Chebyshev polynomial of the
// spectrum mapped so that the *unwanted* interval falls in [-1, 1], where |T_m|
// <= 1, while the wanted end falls outside, where T_m grows like
// cosh(m acosh(x)). A handful of iterations therefore does what many
// unpreconditioned Krylov steps could not.
//
// Why this rather than more LOBPCG: the filter needs no preconditioner and no
// factorization, only matvecs. Unpreconditioned LOBPCG is documented to stagnate,
// and the ILU(k) preconditioner syevx_lobpcg can build is only valid when looking
// for the *smallest* eigenpairs (it approximates A^{-1}). Filtering has no such
// restriction and handles either end.
//
// Scaling: the recurrence is the scaled Zhou/Saad form, not the textbook one.
// T_m evaluated directly overflows quickly -- with a spectrum reaching x = 3 on
// the mapped axis, T_25(3) is about 1e38, which is float infinity. Carrying the
// sigma factors keeps every intermediate near unit magnitude, and since the block
// is orthonormalized immediately afterwards the overall scale is irrelevant
// anyway; only the *ratio* between wanted and unwanted components matters.
//
// Known cost, not yet addressed: the convergence test reads a device flag on the
// host once per outer iteration, which serializes the queue. That is the same
// defect as SYEVX_PLAN.md §7.1 in LOBPCG. It costs one sync per outer iteration
// (not per matvec), so it is far less severe here than there.

#include "../linalg-impl.hh"
#include <util/sycl-vector.hh>
#include <util/sycl-span.hh>
#include "../queue.hh"
#include <util/mempool.hh>
#include <sycl/sycl.hpp>
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <cstdio>
#include <type_traits>
#include <blas/linalg.hh>
#include <blas/functions.hh>
#include <batchlas/backend_config.h>
#include "../util/template-instantiations.hh"

namespace batchlas {

template <Backend B, typename T, MatrixFormat MFormat> struct SyevxFilterBoundsKernel;
template <Backend B, typename T, MatrixFormat MFormat> struct SyevxFilterBoundsPartialKernel;
template <Backend B, typename T, MatrixFormat MFormat> struct SyevxFilterBoundsReduceKernel;
template <Backend B, typename T, MatrixFormat MFormat> struct SyevxFilterInitKernel;
template <Backend B, typename T, MatrixFormat MFormat> struct SyevxFilterStepKernel;
template <Backend B, typename T, MatrixFormat MFormat> struct SyevxFilterSigmaKernel;
template <Backend B, typename T, MatrixFormat MFormat> struct SyevxFilterIntervalKernel;
template <Backend B, typename T, MatrixFormat MFormat> struct SyevxFilterResidualKernel;
template <Backend B, typename T, MatrixFormat MFormat> struct SyevxFilterFinalizeKernel;

namespace {

// Default Chebyshev degree. Deliberately mid-range: too low and each outer
// iteration barely separates the spectrum, too high and the extra matvecs are
// wasted because the block is reorthogonalized anyway. Unmeasured on this
// hardware -- see the benchmark note in SYEVX_PLAN.md §2.4.
constexpr size_t kDefaultFilterDegree = 10;

// Bounds for the automatically derived degree (see the note at the interval
// kernel). The lower bound is the historical default, so switching auto on can
// never make a step *less* selective than the old constant did. The upper bound
// is the usual ChASE-style ceiling: past roughly this the extra matvecs stop
// paying for themselves and the precision cap tends to bind anyway.
constexpr int kAutoDegreeMin = 10;
constexpr int kAutoDegreeMax = 40;

// Work-group size of the parallel Gershgorin kernel. One work-item per row, so
// neighbouring items read neighbouring addresses of a column-major matrix and
// every load is coalesced. Row-groups per matrix scale as n / kBoundsWG, i.e.
// the parallelism grows linearly while the work grows as n^2.
constexpr size_t kBoundsWG = 64;

// xorshift-based fill, so the starting block does not depend on host RNG state
// and is reproducible across runs and backends.
inline uint32_t splitmix(uint32_t x) {
    x += 0x9e3779b9u;
    x = (x ^ (x >> 16)) * 0x85ebca6bu;
    x = (x ^ (x >> 13)) * 0xc2b2ae35u;
    return x ^ (x >> 16);
}

} // namespace

template <Backend B, typename T, MatrixFormat MFormat>
Event syevx_filtered(Queue& ctx,
                     const MatrixView<T, MFormat>& A,
                     Span<typename base_type<T>::type> W,
                     size_t neigs,
                     Span<std::byte> workspace,
                     JobType jobz,
                     const MatrixView<T, MatrixFormat::Dense>& V,
                     const SyevxParams<T>& params) {
    using Real = typename base_type<T>::type;

    const int64_t n = A.rows();
    const int64_t batch = A.batch_size();
    const int64_t k = static_cast<int64_t>(neigs);
    const bool want_vectors = (jobz == JobType::EigenVectors);

    if (A.rows() != A.cols()) throw std::runtime_error("syevx_filtered: A must be square");
    if (k < 1 || k > n) throw std::runtime_error("syevx_filtered: invalid neigs");

    // Block size. Extra directions give the filter room to resolve the boundary
    // between wanted and unwanted; without any, the cut sits exactly at the edge
    // of the block and convergence of the last wanted pair is slow.
    int64_t m = k + static_cast<int64_t>(params.extra_directions);
    if (params.extra_directions == 0) m = k + std::max<int64_t>(2, k / 4);
    m = std::min(m, n);

    // Filter degree. Precedence: BATCHLAS_SYEVX_FILTER_DEGREE > params.filter_degree
    // > derived-from-the-problem > kDefaultFilterDegree. Either of the first two
    // pins the degree and turns the derivation off; BATCHLAS_SYEVX_FILTER_DEGREE_AUTO=0
    // also turns it off, restoring the old constant-10 behaviour exactly.
    bool degree_explicit = params.filter_degree > 0;
    size_t degree = degree_explicit ? params.filter_degree : kDefaultFilterDegree;
    if (const char* dv = std::getenv("BATCHLAS_SYEVX_FILTER_DEGREE")) {
        const int parsed = std::atoi(dv);
        if (parsed > 0) { degree = static_cast<size_t>(parsed); degree_explicit = true; }
    }
    // The derivation is OFF by default, opt in with BATCHLAS_SYEVX_FILTER_DEGREE_AUTO=1.
    //
    // It is a large win at small batch and a large LOSS at batch >= 4, so it cannot
    // ship on. MEASURED (RTX 4090, float, n=1024, neigs=8, derived / fixed-10, and
    // the two modes agree to within 0.02x so this is not a jobz effect):
    //
    //   batch          1      2      4      8     16
    //   NoEigenVectors 2.18x  2.25x  0.70x  0.50x  0.48x
    //   EigenVectors   2.20x  2.30x  0.69x  0.49x  0.48x
    //
    // The mechanism is the batch reduction, not the derivation itself: the degree is
    // derived per matrix and then reduced to a MIN across the batch, so the GEMM
    // shapes stay uniform. As the batch grows, the chance that some matrix forces a
    // low degree tends to 1, the whole batch runs at that worst-case degree, and the
    // outer iteration count explodes -- which is exactly the plateau at ~0.48x.
    //
    // Fixing it means changing the batch reduction (a per-matrix degree costs the
    // batched GEMM shape; a quantile instead of a min would keep it), not the
    // per-matrix formula. Until then the constant is the safer default.
    bool auto_degree = false;
    if (const char* av = std::getenv("BATCHLAS_SYEVX_FILTER_DEGREE_AUTO")) {
        if (std::atoi(av) != 0 && !degree_explicit) auto_degree = true;
    }
    bool legacy_bounds = false;
    if (const char* bv = std::getenv("BATCHLAS_SYEVX_BOUNDS_LEGACY")) {
        legacy_bounds = (std::atoi(bv) != 0);
    }
    const bool find_largest = params.find_largest;

    BumpAllocator pool(workspace);

    const size_t block_elems = static_cast<size_t>(n) * m * batch;
    auto x_span     = pool.allocate<T>(ctx, block_elems);
    auto ax_span    = pool.allocate<T>(ctx, block_elems);
    auto y_span     = pool.allocate<T>(ctx, block_elems);
    auto yprev_span = pool.allocate<T>(ctx, block_elems);
    auto tmp_span   = pool.allocate<T>(ctx, block_elems);

    // Batched BLAS needs a per-item pointer array alongside each view.
    auto mk = [&](Span<T> s) {
        auto ptrs = pool.allocate<T*>(ctx, static_cast<size_t>(batch));
        return MatrixView<T, MatrixFormat::Dense>(s.data(), n, m, n,
                                                  static_cast<int64_t>(n) * m, batch,
                                                  ptrs.data());
    };
    auto X = mk(x_span);
    auto AX = mk(ax_span);
    auto Y = mk(y_span);
    auto Yprev = mk(yprev_span);
    auto Tmp = mk(tmp_span);

    auto h_span = pool.allocate<T>(ctx, static_cast<size_t>(m) * m * batch);
    auto h_ptrs = pool.allocate<T*>(ctx, static_cast<size_t>(batch));
    MatrixView<T, MatrixFormat::Dense> H(h_span.data(), m, m, m,
                                         static_cast<int64_t>(m) * m, batch,
                                         h_ptrs.data());
    auto theta_span = pool.allocate<Real>(ctx, static_cast<size_t>(m) * batch);

    // Per-batch filter scalars.
    auto lo_span    = pool.allocate<Real>(ctx, static_cast<size_t>(batch));
    auto hi_span    = pool.allocate<Real>(ctx, static_cast<size_t>(batch));
    auto cc_span    = pool.allocate<Real>(ctx, static_cast<size_t>(batch));
    auto ee_span    = pool.allocate<Real>(ctx, static_cast<size_t>(batch));
    auto sig1_span  = pool.allocate<Real>(ctx, static_cast<size_t>(batch));
    auto sig_span   = pool.allocate<Real>(ctx, static_cast<size_t>(batch));
    auto signew_span= pool.allocate<Real>(ctx, static_cast<size_t>(batch));
    auto ca_span    = pool.allocate<Real>(ctx, static_cast<size_t>(batch));
    auto cb_span    = pool.allocate<Real>(ctx, static_cast<size_t>(batch));

    Span<std::byte> spmm_ws;
    if constexpr (MFormat == MatrixFormat::CSR) {
        spmm_ws = pool.allocate<std::byte>(ctx,
            spmm_buffer_size<B>(ctx, A, X, AX, T(1), T(0), Transpose::NoTrans, Transpose::NoTrans));
    }
    auto ortho_ws = pool.allocate<std::byte>(ctx,
        ortho_buffer_size<B>(ctx, X, Transpose::NoTrans, params.algorithm));
    auto syev_ws = pool.allocate<std::byte>(ctx,
        syev_buffer_size<B>(ctx, H, theta_span, JobType::EigenVectors, Uplo::Lower));

    // Residual norms of the wanted columns, plus a per-batch converged flag.
    auto resid_span = pool.allocate<Real>(ctx, static_cast<size_t>(k) * batch);
    UnifiedVector<int32_t> converged(static_cast<size_t>(batch));
    // Per-batch precision limit on the filter degree; reduced to one value so the
    // recurrence length stays uniform and the GEMM shapes stay batched.
    UnifiedVector<int32_t> degree_cap(static_cast<size_t>(batch));
    // Per-batch degree the current residuals actually ask for (auto mode).
    UnifiedVector<int32_t> degree_need(static_cast<size_t>(batch));

    const Transpose conj_t = is_std_complex_v<T> ? Transpose::ConjTrans : Transpose::Trans;

    auto matvec = [&](const MatrixView<T, MatrixFormat::Dense>& in,
                      const MatrixView<T, MatrixFormat::Dense>& out) {
        if constexpr (MFormat == MatrixFormat::Dense) {
            gemm<B>(ctx, A, in, out, GemmOptions<T>{});
        } else {
            spmm<B>(ctx, A, in, out, T(1), T(0), Transpose::NoTrans, Transpose::NoTrans, spmm_ws);
        }
    };

    // ---- Gershgorin bounds. Cheap, needs no matvec, and only has to be
    // conservative: a loose interval costs filter sharpness, never correctness.
    //
    // Row i contributes the interval [a_ii - sum_{j!=i}|a_ij|, a_ii + sum_{j!=i}|a_ij|];
    // lo/hi are the min/max over rows. The legacy kernel ran that as ONE work-item
    // per matrix walking all n^2 elements with a column-major stride (every load its
    // own sector) -- measured at 80.2% of the whole solve at n=1024, and completely
    // independent of batch. The parallel version below gives each work-item one row
    // and keeps the *inner* j loop serial and in ascending order, so each row's
    // radius is accumulated in exactly the legacy order and the per-row endpoints are
    // bit-identical; only the outer min/max is re-associated, and min/max is exact
    // under any association. The results are therefore bit-identical, not merely
    // equivalent. Neighbouring work-items handle neighbouring rows, so at a fixed j
    // the work-group reads a contiguous run of a column: fully coalesced.
    //
    // BATCHLAS_SYEVX_BOUNDS_LEGACY=1 restores the serial kernel for A/B.
    auto row_bound = [](const auto& Akv, int64_t i, int b, int64_t nn) {
        Real diag = Real(0);
        Real radius = Real(0);
        if constexpr (MFormat == MatrixFormat::Dense) {
            for (int64_t j = 0; j < nn; ++j) {
                const T v = Akv(static_cast<int>(i), static_cast<int>(j), b);
                if (i == j) {
                    if constexpr (is_std_complex_v<T>) diag = v.real();
                    else diag = v;
                } else {
                    if constexpr (is_std_complex_v<T>)
                        radius += sycl::hypot(v.real(), v.imag());
                    else radius += sycl::fabs(v);
                }
            }
        } else {
            const int ro = b * Akv.offset_stride_;
            const int vb = b * Akv.matrix_stride_;
            const int rs = Akv.row_offsets_[ro + i];
            const int re = Akv.row_offsets_[ro + i + 1];
            for (int p = rs; p < re; ++p) {
                const int j = Akv.col_indices_[vb + p];
                const T v = Akv.data_[vb + p];
                if (j == static_cast<int>(i)) {
                    if constexpr (is_std_complex_v<T>) diag = v.real();
                    else diag = v;
                } else {
                    if constexpr (is_std_complex_v<T>)
                        radius += sycl::hypot(v.real(), v.imag());
                    else radius += sycl::fabs(v);
                }
            }
        }
        return sycl::vec<Real, 2>(diag - radius, diag + radius);
    };

    const size_t row_groups =
        std::max<size_t>(1, (static_cast<size_t>(n) + kBoundsWG - 1) / kBoundsWG);
    // Per-(matrix, row-group) partial [min, max]. Small (2 * ceil(n/64) * batch
    // reals) and allocated outside the BumpAllocator, so the workspace query is
    // unaffected -- see the note at syevx_filtered_buffer_size.
    UnifiedVector<Real> bounds_partial(2 * row_groups * static_cast<size_t>(batch));
    if (legacy_bounds) {
        auto lo = lo_span.data();
        auto hi = hi_span.data();
        auto Akv = A.kernel_view();
        const int64_t nn = n;
        ctx->submit([&](sycl::handler& h) {
            h.parallel_for<SyevxFilterBoundsKernel<B, T, MFormat>>(
                sycl::range<1>(static_cast<size_t>(batch)), [=](sycl::id<1> tid) {
                    const int b = static_cast<int>(tid[0]);
                    Real l = std::numeric_limits<Real>::max();
                    Real u = -std::numeric_limits<Real>::max();
                    for (int64_t i = 0; i < nn; ++i) {
                        const sycl::vec<Real, 2> rb = row_bound(Akv, i, b, nn);
                        l = sycl::min(l, rb[0]);
                        u = sycl::max(u, rb[1]);
                    }
                    // Degenerate spectrum: widen so e > 0 below.
                    if (!(u > l)) { u = l + Real(1); }
                    lo[b] = l;
                    hi[b] = u;
                });
        });
    } else {
        auto Akv = A.kernel_view();
        auto part = bounds_partial.data();
        const int64_t nn = n;
        const size_t rg = row_groups;
        ctx->submit([&](sycl::handler& h) {
            h.parallel_for<SyevxFilterBoundsPartialKernel<B, T, MFormat>>(
                sycl::nd_range<1>(sycl::range{static_cast<size_t>(batch) * rg * kBoundsWG},
                                  sycl::range{kBoundsWG}),
                [=](sycl::nd_item<1> item) {
                    const size_t gid = item.get_group_linear_id();
                    const int b = static_cast<int>(gid / rg);
                    const size_t g = gid % rg;
                    const size_t lid = item.get_local_linear_id();
                    const int64_t i = static_cast<int64_t>(g * kBoundsWG + lid);
                    Real l = std::numeric_limits<Real>::max();
                    Real u = -std::numeric_limits<Real>::max();
                    if (i < nn) {
                        const sycl::vec<Real, 2> rb = row_bound(Akv, i, b, nn);
                        l = rb[0];
                        u = rb[1];
                    }
                    l = sycl::reduce_over_group(item.get_group(), l, sycl::minimum<Real>());
                    u = sycl::reduce_over_group(item.get_group(), u, sycl::maximum<Real>());
                    if (lid == 0) {
                        part[2 * gid + 0] = l;
                        part[2 * gid + 1] = u;
                    }
                });
        });
        auto lo = lo_span.data();
        auto hi = hi_span.data();
        const size_t rwg = 128;
        ctx->submit([&](sycl::handler& h) {
            h.parallel_for<SyevxFilterBoundsReduceKernel<B, T, MFormat>>(
                sycl::nd_range<1>(sycl::range{static_cast<size_t>(batch) * rwg},
                                  sycl::range{rwg}),
                [=](sycl::nd_item<1> item) {
                    const int b = static_cast<int>(item.get_group_linear_id());
                    const size_t lid = item.get_local_linear_id();
                    const size_t lsz = item.get_local_range(0);
                    Real l = std::numeric_limits<Real>::max();
                    Real u = -std::numeric_limits<Real>::max();
                    for (size_t g = lid; g < rg; g += lsz) {
                        const size_t idx = static_cast<size_t>(b) * rg + g;
                        l = sycl::min(l, part[2 * idx + 0]);
                        u = sycl::max(u, part[2 * idx + 1]);
                    }
                    l = sycl::reduce_over_group(item.get_group(), l, sycl::minimum<Real>());
                    u = sycl::reduce_over_group(item.get_group(), u, sycl::maximum<Real>());
                    if (lid == 0) {
                        // Degenerate spectrum: widen so e > 0 below.
                        if (!(u > l)) { u = l + Real(1); }
                        lo[b] = l;
                        hi[b] = u;
                    }
                });
        });
    }

    // ---- Random orthonormal start.
    {
        auto Xk = X.kernel_view();
        const int64_t nn = n, mm = m;
        ctx->submit([&](sycl::handler& h) {
            h.parallel_for<SyevxFilterInitKernel<B, T, MFormat>>(
                sycl::range<1>(static_cast<size_t>(batch * nn * mm)), [=](sycl::id<1> tid) {
                    const int64_t idx = static_cast<int64_t>(tid[0]);
                    const int b = static_cast<int>(idx / (nn * mm));
                    const int64_t rem = idx - static_cast<int64_t>(b) * nn * mm;
                    const int r = static_cast<int>(rem % nn);
                    const int c = static_cast<int>(rem / nn);
                    const uint32_t s = splitmix(static_cast<uint32_t>(idx) ^ 0x5bf03635u);
                    const Real v = Real(static_cast<int32_t>(s & 0xffffu) - 32768) / Real(32768);
                    if constexpr (is_std_complex_v<T>) {
                        const uint32_t s2 = splitmix(s);
                        const Real v2 = Real(static_cast<int32_t>(s2 & 0xffffu) - 32768) / Real(32768);
                        Xk(r, c, b) = T(v, v2);
                    } else {
                        Xk(r, c, b) = T(v);
                    }
                });
        });
    }

    ortho<B>(ctx, X, Transpose::NoTrans, ortho_ws, params.algorithm);

    // Rayleigh-Ritz on the current block: X <- X Z, AX <- AX Z, theta <- eigenvalues.
    auto rayleigh_ritz = [&](const MatrixView<T, MatrixFormat::Dense>& blk,
                             const MatrixView<T, MatrixFormat::Dense>& ablk) {
        matvec(blk, ablk);
        gemm<B>(ctx, blk, ablk, H, {.transA = conj_t});
        syev<B>(ctx, H, theta_span, SyevOptions{}, syev_ws);
        gemm<B>(ctx, blk, H, Tmp, GemmOptions<T>{});
        MatrixView<T, MatrixFormat::Dense>::copy(ctx, blk, Tmp);
        gemm<B>(ctx, ablk, H, Tmp, GemmOptions<T>{});
        MatrixView<T, MatrixFormat::Dense>::copy(ctx, ablk, Tmp);
    };

    rayleigh_ritz(X, AX);

    const Real conv_tol = static_cast<Real>(std::abs(params.absolute_tolerance));
    const Real use_tol = conv_tol > Real(0) ? conv_tol
                                            : std::numeric_limits<Real>::epsilon() * Real(100);
    const size_t max_iters = params.iterations > 0 ? params.iterations : 100;

    for (size_t iter = 0; iter < max_iters; ++iter) {
        // ---- Residuals of the wanted columns: r_j = A x_j - theta_j x_j.
        {
            auto Xk = X.kernel_view();
            auto AXk = AX.kernel_view();
            auto th = theta_span.data();
            auto rn = resid_span.data();
            auto flag = converged.data();
            const int64_t nn = n, mm = m, kk = k;
            const bool large = find_largest;
            const Real tolv = use_tol;
            const size_t wg = 128;
            ctx->submit([&](sycl::handler& h) {
                h.parallel_for<SyevxFilterResidualKernel<B, T, MFormat>>(
                    sycl::nd_range<1>(sycl::range{static_cast<size_t>(batch) * wg},
                                      sycl::range{wg}),
                    [=](sycl::nd_item<1> item) {
                        const int b = static_cast<int>(item.get_group_linear_id());
                        const int lid = static_cast<int>(item.get_local_linear_id());
                        const int lsz = static_cast<int>(item.get_local_range(0));
                        int local_ok = 1;
                        for (int64_t j = lid; j < kk; j += lsz) {
                            const int64_t col = large ? (mm - 1 - j) : j;
                            const Real lam = th[b * mm + col];
                            Real acc = Real(0);
                            for (int64_t r = 0; r < nn; ++r) {
                                const T d = AXk(static_cast<int>(r), static_cast<int>(col), b)
                                          - T(lam) * Xk(static_cast<int>(r), static_cast<int>(col), b);
                                if constexpr (is_std_complex_v<T>) {
                                    acc += d.real() * d.real() + d.imag() * d.imag();
                                } else {
                                    acc += d * d;
                                }
                            }
                            const Real nrm = sycl::sqrt(acc);
                            rn[b * kk + j] = nrm;
                            const Real scale = sycl::fmax(sycl::fabs(lam), Real(1));
                            if (!(nrm <= tolv * scale)) local_ok = 0;
                        }
                        const int all_ok = sycl::reduce_over_group(
                            item.get_group(), local_ok, sycl::minimum<int>());
                        if (lid == 0) flag[b] = all_ok;
                    });
            });
        }

        // Host read of the convergence flag: one sync per outer iteration.
        ctx.wait();
        bool all_converged = true;
        for (int64_t b = 0; b < batch; ++b) {
            if (converged[b] == 0) { all_converged = false; break; }
        }
        if (all_converged || iter + 1 == max_iters) break;

        // ---- Filter interval. The cut is the first *unwanted* Ritz value, so
        // the damped interval is exactly the part of the spectrum being
        // rejected; everything wanted lies strictly outside it.
        {
            auto th = theta_span.data();
            auto lo = lo_span.data(); auto hi = hi_span.data();
            auto cc = cc_span.data(); auto ee = ee_span.data();
            auto s1 = sig1_span.data(); auto sg = sig_span.data();
            auto dcap_out = degree_cap.data();
            auto dneed_out = degree_need.data();
            auto rn = resid_span.data();
            const int64_t mm = m, kk = k;
            const bool large = find_largest;
            const Real tolv = use_tol;
            const int auto_min = kAutoDegreeMin;
            const int auto_max = kAutoDegreeMax;
            ctx->submit([&](sycl::handler& h) {
                h.parallel_for<SyevxFilterIntervalKernel<B, T, MFormat>>(
                    sycl::range<1>(static_cast<size_t>(batch)), [=](sycl::id<1> tid) {
                        const int b = static_cast<int>(tid[0]);
                        // `far` is the point at which the scaled polynomial is
                        // normalised to 1. It must be the extreme *Ritz* value,
                        // not the Gershgorin bound: Gershgorin overestimates the
                        // spectral radius badly for a random symmetric matrix
                        // (O(n) versus the true O(sqrt n)), and normalising there
                        // divides by T_m of a point far outside the spectrum. At
                        // degree 40 that underflows the whole block to zero, and
                        // orthogonalising a zero block yields NaN. Normalising at
                        // the wanted end instead keeps p ~ 1 exactly where the
                        // wanted vectors live.
                        Real a, bb, far;
                        if (large) {
                            // Damp [lo, cut]; amplify above it.
                            const Real cut = th[b * mm + (mm - kk - 1 >= 0 ? mm - kk - 1 : 0)];
                            a = lo[b];
                            bb = sycl::fmax(cut, a + (hi[b] - a) * Real(1e-6));
                            far = th[b * mm + (mm - 1)];
                        } else {
                            const Real cut = th[b * mm + (kk < mm ? kk : mm - 1)];
                            bb = hi[b];
                            a = sycl::fmin(cut, bb - (bb - lo[b]) * Real(1e-6));
                            far = th[b * mm];
                        }
                        const Real c = (a + bb) / Real(2);
                        Real e = (bb - a) / Real(2);
                        if (!(e > Real(0))) e = sycl::fmax(sycl::fabs(c), Real(1)) * Real(1e-6);
                        cc[b] = c;
                        ee[b] = e;
                        // sigma1 = e / (far - c); |far - c| > e whenever the far
                        // end really is outside the damped interval.
                        Real den = far - c;
                        if (sycl::fabs(den) < e * Real(1.0001)) {
                            den = (den >= Real(0) ? Real(1) : Real(-1)) * e * Real(1.0001);
                        }
                        s1[b] = e / den;
                        sg[b] = s1[b];

                        // Precision-bounded degree.
                        //
                        // The filter amplifies the most-wanted direction over the
                        // least-wanted one by cosh(d*acosh(y_far)) /
                        // cosh(d*acosh(y_edge)). Once that ratio passes 1/sqrt(eps)
                        // the least-wanted columns are numerically swamped, the
                        // block becomes rank-deficient, and the Cholesky-based
                        // orthogonalization fails -- producing NaN rather than a
                        // slow answer. Bounding d keeps every outer iteration
                        // well-conditioned; the lost sharpness is recovered by
                        // doing another iteration, which is cheap by comparison.
                        const Real y_far = sycl::fabs((far - c) / e);
                        const int64_t edge = large ? (mm - kk >= 0 ? mm - kk : 0)
                                                   : (kk - 1 >= 0 ? kk - 1 : 0);
                        const Real y_edge = sycl::fabs((th[b * mm + edge] - c) / e);
                        //
                        // When y_edge <= 1 the least-wanted direction is inside
                        // the damped band, where |T_d| <= 1. That is the *worst*
                        // case for the ratio, not a case to skip: the growth is
                        // then the full cosh(d*acosh(y_far)), so the edge term
                        // drops to zero rather than the cap being waived.
                        int dcap = 1 << 20;
                        if (y_far > Real(1)) {
                            const Real edge_term =
                                (y_edge > Real(1)) ? sycl::acosh(y_edge) : Real(0);
                            const Real g = sycl::acosh(y_far) - edge_term;
                            if (g > Real(0)) {
                                // Cap the amplification ratio at eps^-1/4 (about
                                // 300 in float). eps^-1/2 was measured to be too
                                // generous: the block still went rank-deficient
                                // and Chol2 returned NaN at degree 40.
                                const Real budget = -Real(0.25) *
                                    sycl::log(std::numeric_limits<Real>::epsilon());
                                const Real d = budget / g;
                                dcap = (d < Real(1)) ? 1
                                     : (d > Real(1 << 20) ? (1 << 20) : static_cast<int>(d));
                            }
                        }
                        dcap_out[b] = dcap;

                        // ---- Degree derived from the problem (ChASE-style).
                        //
                        // Inside the damped band |T_d| <= 1; the least-amplified
                        // *wanted* direction sits at mapped coordinate y_edge > 1
                        // and grows like cosh(d acosh(y_edge)) ~ exp(d rate)/2 with
                        // rate = acosh(y_edge). So one outer iteration shrinks the
                        // unwanted content of the block by roughly exp(-d rate),
                        // and the degree that would finish the job in one more
                        // iteration is
                        //
                        //     d_need = log(worst_residual / target) / rate
                        //
                        // measured in the same scaled units the convergence test
                        // uses. This is exactly the geometry the code already
                        // computes for the precision cap; the cap stays an upper
                        // bound on top of it, because it guards the real measured
                        // failure mode (rank-deficient block -> NaN from Cholesky).
                        Real ratio = Real(1);
                        for (int64_t j = 0; j < kk; ++j) {
                            const int64_t col = large ? (mm - 1 - j) : j;
                            const Real lam = th[b * mm + col];
                            const Real scale = sycl::fmax(sycl::fabs(lam), Real(1));
                            const Real rr = rn[b * kk + j] / (tolv * scale);
                            ratio = sycl::fmax(ratio, rr);
                        }
                        int dneed = auto_max;
                        if (y_edge > Real(1)) {
                            const Real rate = sycl::acosh(y_edge);
                            if (rate > Real(0)) {
                                const Real d = sycl::log(ratio) / rate;
                                dneed = (d < Real(auto_min))
                                            ? auto_min
                                            : (d > Real(auto_max) ? auto_max
                                                                  : static_cast<int>(d) + 1);
                            }
                        }
                        dneed_out[b] = dneed;
                    });
            });
        }

        // ---- Chebyshev recurrence (scaled).
        //
        //   Y_1     = (sigma1/e) (A - cI) X
        //   Y_{j+1} = (2 sigma'/e) (A - cI) Y_j - (sigma sigma') Y_{j-1}
        auto apply_step = [&](const MatrixView<T, MatrixFormat::Dense>& src,
                              const MatrixView<T, MatrixFormat::Dense>& prev,
                              const MatrixView<T, MatrixFormat::Dense>& dst,
                              bool first) {
            matvec(src, Tmp);
            auto Sk = src.kernel_view();
            auto Pk = prev.kernel_view();
            auto Dk = dst.kernel_view();
            auto Tk = Tmp.kernel_view();
            auto ca = ca_span.data();
            auto cb = cb_span.data();
            auto cvals = cc_span.data();
            const int64_t nn = n, mm = m;
            const bool is_first = first;
            ctx->submit([&](sycl::handler& h) {
                h.parallel_for<SyevxFilterStepKernel<B, T, MFormat>>(
                    sycl::range<1>(static_cast<size_t>(batch * nn * mm)), [=](sycl::id<1> tid) {
                        const int64_t idx = static_cast<int64_t>(tid[0]);
                        const int b = static_cast<int>(idx / (nn * mm));
                        const int64_t rem = idx - static_cast<int64_t>(b) * nn * mm;
                        const int r = static_cast<int>(rem % nn);
                        const int c = static_cast<int>(rem / nn);
                        const T av = Tk(r, c, b) - T(cvals[b]) * Sk(r, c, b);
                        T out = T(ca[b]) * av;
                        if (!is_first) out = out - T(cb[b]) * Pk(r, c, b);
                        Dk(r, c, b) = out;
                    });
            });
        };

        // Step 1: coefficients sigma1/e, no previous term.
        {
            auto ca = ca_span.data(); auto cb = cb_span.data();
            auto s1 = sig1_span.data(); auto ee = ee_span.data();
            ctx->submit([&](sycl::handler& h) {
                h.parallel_for(sycl::range<1>(static_cast<size_t>(batch)), [=](sycl::id<1> tid) {
                    const int b = static_cast<int>(tid[0]);
                    ca[b] = s1[b] / ee[b];
                    cb[b] = Real(0);
                });
            });
        }
        apply_step(X, X, Y, /*first=*/true);
        MatrixView<T, MatrixFormat::Dense>::copy(ctx, Yprev, X);

        // The interval kernel just wrote the per-batch precision limit; take the
        // strictest so every item runs the same recurrence length. This reuses
        // the sync already paid for the convergence check above.
        ctx.wait();
        size_t eff_degree = degree;
        if (auto_degree) {
            // Uniform across the batch so the recurrence length -- and hence every
            // GEMM shape -- stays batched: take the largest requirement, then let
            // the strictest precision cap trim it.
            size_t need = static_cast<size_t>(kAutoDegreeMin);
            for (int64_t b = 0; b < batch; ++b) {
                need = std::max(need, static_cast<size_t>(std::max(1, degree_need[b])));
            }
            eff_degree = need;
        }
        for (int64_t b = 0; b < batch; ++b) {
            const size_t cap = static_cast<size_t>(std::max(1, degree_cap[b]));
            eff_degree = std::min(eff_degree, cap);
        }
        if (eff_degree < 1) eff_degree = 1;
        if (std::getenv("BATCHLAS_DEBUG_FILTER_DEGREE")) {
            for (int64_t b = 0; b < batch; ++b) {
                Real rmax = Real(0);
                for (int64_t j = 0; j < k; ++j)
                    rmax = std::max(rmax, resid_span[static_cast<size_t>(b * k + j)]);
                std::fprintf(stderr,
                    "[filt] iter=%zu b=%lld dcap=%d dneed=%d auto=%d rmax=%g tol=%g eff=%zu\n",
                    iter, (long long)b, degree_cap[b], degree_need[b], (int)auto_degree,
                    (double)rmax, (double)use_tol, eff_degree);
            }
        }

        for (size_t j = 2; j <= eff_degree; ++j) {
            {
                auto s1 = sig1_span.data(); auto sg = sig_span.data();
                auto sn = signew_span.data(); auto ee = ee_span.data();
                auto ca = ca_span.data(); auto cb = cb_span.data();
                ctx->submit([&](sycl::handler& h) {
                    h.parallel_for<SyevxFilterSigmaKernel<B, T, MFormat>>(
                        sycl::range<1>(static_cast<size_t>(batch)), [=](sycl::id<1> tid) {
                            const int b = static_cast<int>(tid[0]);
                            const Real snew = Real(1) / (Real(2) / s1[b] - sg[b]);
                            sn[b] = snew;
                            ca[b] = Real(2) * snew / ee[b];
                            cb[b] = sg[b] * snew;
                            sg[b] = snew;
                        });
                });
            }
            // Writes into Yprev, whose old contents are consumed by this same
            // step, then swaps so Y stays the newest iterate.
            apply_step(Y, Yprev, Yprev, /*first=*/false);
            std::swap(Y, Yprev);
        }

        ortho<B>(ctx, Y, Transpose::NoTrans, ortho_ws, params.algorithm);
        MatrixView<T, MatrixFormat::Dense>::copy(ctx, X, Y);
        rayleigh_ritz(X, AX);
    }

    // ---- Emit the wanted end, in the requested order.
    {
        auto th = theta_span.data();
        auto Xk = X.kernel_view();
        auto* w_out = W.data();
        T* v_ptr = want_vectors ? V.data_ptr() : nullptr;
        const int64_t v_ld = want_vectors ? V.ld() : 0;
        const int64_t v_stride = want_vectors ? V.stride() : 0;
        const int64_t nn = n, mm = m, kk = k;
        const bool large = find_largest;
        const size_t wg = 128;
        ctx->submit([&](sycl::handler& h) {
            h.parallel_for<SyevxFilterFinalizeKernel<B, T, MFormat>>(
                sycl::nd_range<1>(sycl::range{static_cast<size_t>(batch) * wg}, sycl::range{wg}),
                [=](sycl::nd_item<1> item) {
                    const int b = static_cast<int>(item.get_group_linear_id());
                    const int lid = static_cast<int>(item.get_local_linear_id());
                    const int lsz = static_cast<int>(item.get_local_range(0));
                    for (int64_t j = lid; j < kk; j += lsz) {
                        const int64_t col = large ? (mm - 1 - j) : j;
                        w_out[b * kk + j] = th[b * mm + col];
                    }
                    if (v_ptr != nullptr) {
                        for (int64_t linear = lid; linear < nn * kk; linear += lsz) {
                            const int64_t r = linear % nn;
                            const int64_t j = linear / nn;
                            const int64_t col = large ? (mm - 1 - j) : j;
                            v_ptr[b * v_stride + j * v_ld + r] =
                                Xk(static_cast<int>(r), static_cast<int>(col), b);
                        }
                    }
                });
        });
    }

    return ctx.get_event();
}

template <Backend B, typename T, MatrixFormat MFormat>
size_t syevx_filtered_buffer_size(Queue& ctx,
                                  const MatrixView<T, MFormat>& A,
                                  Span<typename base_type<T>::type> W,
                                  size_t neigs,
                                  JobType jobz,
                                  const MatrixView<T, MatrixFormat::Dense>& V,
                                  const SyevxParams<T>& params) {
    // NOTE (workspace lockstep): the parallel Gershgorin reduction added to
    // syevx_filtered() needs one extra scratch array (2 * ceil(n/kBoundsWG) * batch
    // reals), but it is a UnifiedVector allocated outside the BumpAllocator -- the
    // same pattern already used for `converged` and `degree_cap`. The sequence of
    // pool.allocate() calls in syevx_filtered() is therefore byte-for-byte unchanged,
    // and this query needs no corresponding change. The degree change touches no
    // allocation at all (the recurrence reuses Y/Yprev regardless of length).
    using Real = typename base_type<T>::type;
    (void)W; (void)V; (void)jobz;

    const int64_t n = A.rows();
    const int64_t batch = A.batch_size();
    const int64_t k = static_cast<int64_t>(neigs);

    int64_t m = k + static_cast<int64_t>(params.extra_directions);
    if (params.extra_directions == 0) m = k + std::max<int64_t>(2, k / 4);
    m = std::min(m, n);

    size_t bytes = 0;
    const size_t block_elems = static_cast<size_t>(n) * m * batch;
    for (int i = 0; i < 5; ++i) {
        bytes += BumpAllocator::allocation_size<T>(ctx, block_elems);
        bytes += BumpAllocator::allocation_size<T*>(ctx, static_cast<size_t>(batch));
    }
    bytes += BumpAllocator::allocation_size<T>(ctx, static_cast<size_t>(m) * m * batch);
    bytes += BumpAllocator::allocation_size<T*>(ctx, static_cast<size_t>(batch));
    bytes += BumpAllocator::allocation_size<Real>(ctx, static_cast<size_t>(m) * batch);
    for (int i = 0; i < 9; ++i) {
        bytes += BumpAllocator::allocation_size<Real>(ctx, static_cast<size_t>(batch));
    }

    // Shapes must match the runtime views so the providers agree.
    MatrixView<T, MatrixFormat::Dense> blk_dummy(nullptr, n, m, n,
                                                 static_cast<int64_t>(n) * m, batch);
    MatrixView<T, MatrixFormat::Dense> h_dummy(nullptr, m, m, m,
                                               static_cast<int64_t>(m) * m, batch);
    if constexpr (MFormat == MatrixFormat::CSR) {
        bytes += BumpAllocator::allocation_size<std::byte>(ctx,
            spmm_buffer_size<B>(ctx, A, blk_dummy, blk_dummy, T(1), T(0),
                                Transpose::NoTrans, Transpose::NoTrans));
    }
    bytes += BumpAllocator::allocation_size<std::byte>(ctx,
        ortho_buffer_size<B>(ctx, blk_dummy, Transpose::NoTrans, params.algorithm));
    bytes += BumpAllocator::allocation_size<std::byte>(ctx,
        syev_buffer_size<B>(ctx, h_dummy, Span<Real>(), JobType::EigenVectors, Uplo::Lower));
    bytes += BumpAllocator::allocation_size<Real>(ctx, static_cast<size_t>(k) * batch);

    return bytes;
}

#define SYEVX_FILTERED_INSTANTIATE(back, fp, fmt) \
    template Event syevx_filtered<back, BATCHLAS_UNPAREN fp, fmt>(\
        Queue&,\
        const MatrixView<BATCHLAS_UNPAREN fp, fmt>&,\
        Span<typename base_type<BATCHLAS_UNPAREN fp>::type>,\
        size_t,\
        Span<std::byte>,\
        JobType,\
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&,\
        const SyevxParams<BATCHLAS_UNPAREN fp>&);\
    template size_t syevx_filtered_buffer_size<back, BATCHLAS_UNPAREN fp, fmt>(\
        Queue&,\
        const MatrixView<BATCHLAS_UNPAREN fp, fmt>&,\
        Span<typename base_type<BATCHLAS_UNPAREN fp>::type>,\
        size_t,\
        JobType,\
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&,\
        const SyevxParams<BATCHLAS_UNPAREN fp>&);

#define SYEVX_FILTERED_INSTANTIATE_FOR_BACKEND_TYPE(back, fp) \
    BATCHLAS_FOR_EACH_MATRIX_FORMAT_2(SYEVX_FILTERED_INSTANTIATE, back, fp)

#define SYEVX_FILTERED_INSTANTIATE_FOR_BACKEND(back)\
    BATCHLAS_FOR_EACH_SCALAR_TYPE_1(SYEVX_FILTERED_INSTANTIATE_FOR_BACKEND_TYPE, back)

#if BATCHLAS_HAS_CUDA_BACKEND
    SYEVX_FILTERED_INSTANTIATE_FOR_BACKEND(Backend::CUDA);
#endif
#if BATCHLAS_HAS_ROCM_BACKEND
    SYEVX_FILTERED_INSTANTIATE_FOR_BACKEND(Backend::ROCM);
#endif
#if BATCHLAS_HAS_HOST_BACKEND
    SYEVX_FILTERED_INSTANTIATE_FOR_BACKEND(Backend::NETLIB);
#endif

#undef SYEVX_FILTERED_INSTANTIATE_FOR_BACKEND
#undef SYEVX_FILTERED_INSTANTIATE_FOR_BACKEND_TYPE
#undef SYEVX_FILTERED_INSTANTIATE

} // namespace batchlas
