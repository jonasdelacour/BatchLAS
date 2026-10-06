#include <batchlas/blas/extensions.hh>
#include <batchlas/blas/extra.hh>
#include <batchlas/blas/functions.hh>
#include <batchlas/blas/linalg.hh>
#include <batchlas/backend_config.h>
#include <batchlas/tuning_params.hh>
#include <batchlas/util/mempool.hh>

#include "../math-helpers.hh"
#include "../util/template-instantiations.hh"
#include "info_span.hh"
#include "stedc_internal.hh"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <batchlas/util/env.hh>
#include <batchlas/settings.hh>

namespace batchlas {

namespace {

enum class GesvdNativeMode {
    Blocked,
    CTA,
};

// Smallest n for which the Blocked path uses the blocked gebrd. 1 is a floor,
// not a tuned constant: blocked wins at every n, square and tall (tall n <= 32
// fails both CTA predicates and lands here).
// evidence: docs/perf/gesvd.md#gesvd-the-unblocked-gebrd-cliff
// Bare atoi is load-bearing: an unparseable BATCHLAS_GESVD_BLOCKED_GEBRD_MIN
// yields 0, which widens the blocked path; env_int_or would change that.
inline bool gesvd_use_blocked_gebrd(int32_t n, GesvdNativeMode mode) {
    const char* v = batchlas::settings().geometry.gesvd_blocked_gebrd_min.get();
    const int32_t threshold = (v != nullptr) ? std::atoi(v) : 1;
    return mode == GesvdNativeMode::Blocked && n >= threshold;
}

// Bidiagonal solver of the Blocked path, BATCHLAS_GESVD_BIDIAG:
//   bdsdc (default)  Golub-Kahan 2n tridiagonal -> stedc   accurate, ~as fast
//   normal           tridiagonal of B^T B                  squares kappa; A/B only
//   bdsqr            one thread per matrix, zero-shift QR  accurate, 300-400x slower
// evidence: docs/perf/gesvd.md#gesvd-tier-3-bdsdc-as-the-bidiagonal-solver
enum class GesvdBidiagSolver { NormalEquations, Bdsdc, Bdsqr };

inline GesvdBidiagSolver gesvd_bidiag_solver() {
    const char* v = batchlas::settings().selection.gesvd_bidiag.get();
    if (v == nullptr) return GesvdBidiagSolver::Bdsdc;
    const std::string s(v);
    if (s == "bdsqr") return GesvdBidiagSolver::Bdsqr;
    if (s == "normal") return GesvdBidiagSolver::NormalEquations;
    return GesvdBidiagSolver::Bdsdc;
}

// True when the Blocked path solves the bidiagonal problem DIRECTLY (bdsdc or
// bdsqr) rather than through the tridiagonal of B^T B. Both direct solvers write
// the bidiagonal singular vectors themselves, so the eigenvector scratch and the
// tridiagonal solver workspace are not allocated on those paths.
inline bool gesvd_direct_bidiag(GesvdNativeMode mode) {
    return mode == GesvdNativeMode::Blocked
        && gesvd_bidiag_solver() != GesvdBidiagSolver::NormalEquations;
}

// A thin tall U forces a direct bidiagonal solve, deliberately overriding
// BATCHLAS_GESVD_BIDIAG=normal: the normal-equations path allocates m x m
// scratch and patch_zero_left_vectors writes m columns of U, overrunning a
// k-column U. The override reads only the arguments, so buffer_size and the
// run path cannot disagree about which path runs.
inline bool gesvd_direct_bidiag(GesvdNativeMode mode, bool thin_tall_u) {
    return gesvd_direct_bidiag(mode) || (mode == GesvdNativeMode::Blocked && thin_tall_u);
}

inline bool gesvd_stage_profile_enabled() {
    return batchlas::settings().diagnostics.gesvd_profile;
}

struct GesvdStageProfiler {
    Queue& ctx;
    const char* where;
    int32_t n;
    int32_t batch;
    bool enabled;

    template <typename Fn>
    void run(const char* stage, Fn&& fn) const {
        if (!enabled) {
            BATCHLAS_KERNEL_TRACE_SCOPE(stage);
            std::forward<Fn>(fn)();
            return;
        }

        const auto t0 = std::chrono::steady_clock::now();
        {
            BATCHLAS_KERNEL_TRACE_SCOPE(stage);
            std::forward<Fn>(fn)();
        }
        ctx.wait_and_throw();
        const auto t1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::fprintf(stderr,
                     "BATCHLAS_GESVD_STAGE,%s,n=%d,batch=%d,stage=%s,ms=%.6f\n",
                     where,
                     static_cast<int>(n),
                     static_cast<int>(batch),
                     stage,
                     ms);
    }
};

template <typename T>
void validate_gesvd_dims(const MatrixView<T, MatrixFormat::Dense>& a,
                         Span<typename base_type<T>::type> singular_values,
                         const MatrixView<T, MatrixFormat::Dense>& u,
                         const MatrixView<T, MatrixFormat::Dense>& vh,
                         SvdVectors jobu,
                         SvdVectors jobvh,
                         const char* where) {
    if (a.batch_size() < 1 || a.rows() < 1 || a.cols() < 1) {
        throw batchlas::invalid_argument(std::string(where) + ": invalid matrix dimensions or batch size");
    }

    const int64_t m = a.rows();
    const int64_t n = a.cols();
    const int64_t k = std::min(m, n);
    const int64_t batch = a.batch_size();
    const std::size_t need_s = static_cast<std::size_t>(k) * static_cast<std::size_t>(batch);
    if (singular_values.size() < need_s) {
        throw batchlas::invalid_argument(std::string(where) + ": singular_values span too small");
    }

    // Guard on "computed at all" and take the expected extent from the job, so
    // Thin is validated against m x k / k x n. Testing `== All` would let a
    // Thin request through unvalidated with a wrongly-sized output view.
    jobu = canonical_jobu(jobu, m, k);
    jobvh = canonical_jobvh(jobvh, n, k);
    if (jobu != SvdVectors::None) {
        const int64_t want_cols = svd_u_cols(jobu, m, k);
        if (u.rows() != m || u.cols() != want_cols || u.batch_size() != batch) {
            throw batchlas::invalid_argument(std::string(where) + ": U must be (" +
                                        std::to_string(m) + " x " + std::to_string(want_cols) +
                                        ") with matching batch size");
        }
    }
    if (jobvh != SvdVectors::None) {
        const int64_t want_rows = svd_vh_rows(jobvh, n, k);
        if (vh.rows() != want_rows || vh.cols() != n || vh.batch_size() != batch) {
            throw batchlas::invalid_argument(std::string(where) + ": Vh must be (" +
                                        std::to_string(want_rows) + " x " + std::to_string(n) +
                                        ") with matching batch size");
        }
    }
}

template <typename T>
SteqrParams<T> gesvd_cta_steqr_params() {
    SteqrParams<T> params{};
    params.max_sweeps = 400;
    params.cta_shift_strategy = SteqrShiftStrategy::Wilkinson;
    params.sort = true;
    params.sort_order = SortOrder::Ascending;
    return params;
}

template <typename T>
SteqrParams<T> gesvd_blocked_steqr_params() {
    SteqrParams<T> params{};
    params.max_sweeps = 400;
    params.sort = true;
    params.sort_order = SortOrder::Ascending;
    params.back_transform = false;
    return params;
}

template <typename T>
StedcParams<T> gesvd_blocked_stedc_params() {
    StedcParams<T> params{};
    params.leaf_steqr_params.sort = true;
    params.leaf_steqr_params.sort_order = SortOrder::Ascending;
    params.leaf_steqr_params.back_transform = false;
    return params;
}

template <Backend B, typename T>
size_t gesvd_solver_workspace_size(Queue& ctx,
                                   int32_t n,
                                   int32_t batch,
                                   JobType jobz,
                                   GesvdNativeMode mode) {
    VectorView<T> diag_dummy(nullptr, n, batch, 1, n);
    VectorView<T> offdiag_dummy(nullptr, std::max<int32_t>(0, n - 1), batch, 1, std::max<int32_t>(1, n - 1));
    VectorView<T> evals_dummy(nullptr, n, batch, 1, n);

    if (mode == GesvdNativeMode::CTA) {
        return steqr_cta_buffer_size<T>(ctx, diag_dummy, offdiag_dummy, evals_dummy, jobz, gesvd_cta_steqr_params<T>());
    }
    return stedc_buffer_size<B, T>(ctx,
                                      static_cast<size_t>(n),
                                      static_cast<size_t>(batch),
                                      JobType::EigenVectors,
                                      gesvd_blocked_stedc_params<T>());
}

template <Backend B, typename T>
Event solve_tridiagonal(Queue& ctx,
                        const VectorView<T>& diag,
                        const VectorView<T>& offdiag,
                        const VectorView<T>& evals,
                        const MatrixView<T, MatrixFormat::Dense>& dense_out,
                        const VectorView<T>& sign_view,
                        JobType jobz,
                        GesvdNativeMode mode,
                        const Span<std::byte>& ws,
                        Span<int32_t> info) {
    if (mode == GesvdNativeMode::CTA) {
        BATCHLAS_KERNEL_TRACE_SCOPE("gesvd.solve_tridiag.steqr_cta");
        return steqr_cta<B, T>(ctx, diag, offdiag, evals, ws, jobz, gesvd_cta_steqr_params<T>(), dense_out, info);
    }

    {
        BATCHLAS_KERNEL_TRACE_SCOPE("gesvd.solve_tridiag.stedc_prepare_signs");
        ctx->submit([&](sycl::handler& h) {
            auto E = offdiag;
            auto S = sign_view;
            const int32_t n = static_cast<int32_t>(diag.size());
            const int32_t batch = static_cast<int32_t>(diag.batch_size());
            h.parallel_for(sycl::range<1>(static_cast<size_t>(batch)), [=](sycl::id<1> tid) {
                const int32_t b = static_cast<int32_t>(tid[0]);
                S(0, b) = T(1);
                for (int32_t i = 0; i < n - 1; ++i) {
                    const T ei = E(i, b);
                    const T sgn = (ei >= T(0)) ? T(1) : T(-1);
                    S(i + 1, b) = S(i, b) * sgn;
                    E(i, b) = sycl::fabs(ei);
                }
            });
        });
    }
    {
        BATCHLAS_KERNEL_TRACE_SCOPE("gesvd.solve_tridiag.stedc");
    }
    // (void) on an Event: deliberate. This Queue is in-order, so the next submission
    // is already ordered after this one and the Event carries nothing the caller needs.
    // stedc_dispatch, NOT stedc: gesvd_native_impl calls this function twice over
    // the SAME batch items (the right tridiagonal, then the left one when U is
    // wanted and the matrix is tall or rank-deficient), and `stedc` clears the span
    // it is handed. The one clear is in gesvd_native_impl. steqr_cta above does not
    // clear either. See src/extensions/stedc_internal.hh.
    (void)stedc_dispatch<B, T>(ctx,
                diag,
                offdiag,
                evals,
                ws,
                JobType::EigenVectors,
                gesvd_blocked_stedc_params<T>(),
                dense_out,
                info);

    if (jobz == JobType::EigenVectors) {
        BATCHLAS_KERNEL_TRACE_SCOPE("gesvd.solve_tridiag.restore_signs");
        ctx->submit([&](sycl::handler& h) {
            auto Z = dense_out.kernel_view();
            auto S = sign_view;
            const int32_t n = static_cast<int32_t>(dense_out.rows());
            const int32_t batch = static_cast<int32_t>(dense_out.batch_size());
            const int64_t total = static_cast<int64_t>(batch) * n * n;
            h.parallel_for(sycl::range<1>(static_cast<size_t>(total)), [=](sycl::id<1> tid) {
                const int64_t idx = static_cast<int64_t>(tid[0]);
                const int32_t b = static_cast<int32_t>(idx / (static_cast<int64_t>(n) * n));
                const int64_t rem = idx - static_cast<int64_t>(b) * n * n;
                const int32_t row = static_cast<int32_t>(rem % n);
                const int32_t col = static_cast<int32_t>(rem / n);
                Z(row, col, b) *= S(row, b);
            });
        });
    }

    return ctx.get_event();
}

template <typename T>
void form_right_tridiagonal(Queue& ctx,
                            const VectorView<T>& bidiag_d,
                            const VectorView<T>& bidiag_e,
                            const VectorView<T>& tri_d,
                            const VectorView<T>& tri_e) {
    const int32_t n = static_cast<int32_t>(bidiag_d.size());
    const int32_t batch = static_cast<int32_t>(bidiag_d.batch_size());

    ctx->submit([&](sycl::handler& h) {
        auto D = bidiag_d;
        auto E = bidiag_e;
        auto TD = tri_d;
        auto TE = tri_e;

        h.parallel_for(sycl::range<1>(static_cast<size_t>(std::max<int32_t>(1, n) * batch)), [=](sycl::id<1> tid) {
            const int32_t linear = static_cast<int32_t>(tid[0]);
            const int32_t b = linear / n;
            const int32_t i = linear - b * n;

            const T di = D(i, b);
            T diag = di * di;
            if (i > 0) {
                const T e_prev = E(i - 1, b);
                diag += e_prev * e_prev;
            }
            TD(i, b) = diag;

            if (i < n - 1) {
                TE(i, b) = di * E(i, b);
            }
        });
    });
}

template <typename T>
void form_left_tridiagonal(Queue& ctx,
                           const VectorView<T>& bidiag_d,
                           const VectorView<T>& bidiag_e,
                           int32_t left_order,
                           const VectorView<T>& tri_d,
                           const VectorView<T>& tri_e) {
    const int32_t k = static_cast<int32_t>(bidiag_d.size());
    const int32_t batch = static_cast<int32_t>(bidiag_d.batch_size());

    ctx->submit([&](sycl::handler& h) {
        auto D = bidiag_d;
        auto E = bidiag_e;
        auto TD = tri_d;
        auto TE = tri_e;

        h.parallel_for(sycl::range<1>(static_cast<size_t>(std::max<int32_t>(1, left_order) * batch)), [=](sycl::id<1> tid) {
            const int32_t linear = static_cast<int32_t>(tid[0]);
            const int32_t b = linear / left_order;
            const int32_t i = linear - b * left_order;

            T diag = T(0);
            if (i < k) {
                const T di = D(i, b);
                diag += di * di;
                if (i < k - 1) {
                    const T ei = E(i, b);
                    diag += ei * ei;
                    TE(i, b) = ei * D(i + 1, b);
                } else if (i < left_order - 1) {
                    TE(i, b) = T(0);
                }
            } else if (i < left_order - 1) {
                TE(i, b) = T(0);
            }
            TD(i, b) = diag;
        });
    });
}

template <typename T>
void finalize_values_only(Queue& ctx,
                          const VectorView<T>& evals,
                          Span<T> singular_values) {
    const int32_t n = static_cast<int32_t>(evals.size());
    const int32_t batch = static_cast<int32_t>(evals.batch_size());

    ctx->submit([&](sycl::handler& h) {
        auto Evals = evals;
        T* s_out = singular_values.data();

        h.parallel_for(sycl::range<1>(static_cast<size_t>(n * batch)), [=](sycl::id<1> tid) {
            const int32_t linear = static_cast<int32_t>(tid[0]);
            const int32_t b = linear / n;
            const int32_t tgt = linear - b * n;
            const int32_t src = (n - 1) - tgt;
            const T lambda = sycl::fmax(Evals(src, b), T(0));
            s_out[static_cast<size_t>(b) * static_cast<size_t>(n) + static_cast<size_t>(tgt)] = sycl::sqrt(lambda);
        });
    });
}

template <typename T>
inline T gesvd_zero_sigma_tol_device(T sigma_max) {
    const T eps = std::numeric_limits<T>::epsilon();
    // Keep the zero-singular fallback conservative. A looser threshold turns
    // mildly small but valid singular values into "zero" columns, which hurts
    // orthogonality and forces unnecessary left-tridiagonal solves.
    return eps * sycl::fmax(T(1), sigma_max);
}

template <typename T>
inline T gesvd_zero_sigma_tol_host(T sigma_max) {
    const T eps = std::numeric_limits<T>::epsilon();
    return eps * std::max<T>(T(1), sigma_max);
}

template <typename T>
inline T gesvd_conj_if_needed(const T& value) {
    if constexpr (internal::is_complex<T>::value) {
        return T(value.real(), -value.imag());
    } else {
        return value;
    }
}

template <typename Real>
void build_hermitian_svd_permutation(Queue& ctx,
                                     const VectorView<Real>& eigenvalues,
                                     const VectorView<int32_t>& permutation) {
    const int32_t n = static_cast<int32_t>(eigenvalues.size());
    const int32_t batch = static_cast<int32_t>(eigenvalues.batch_size());

    ctx->submit([&](sycl::handler& h) {
        auto Evals = eigenvalues;
        auto Perm = permutation;
        h.parallel_for(sycl::range<1>(static_cast<size_t>(batch)), [=](sycl::id<1> tid) {
            const int32_t b = static_cast<int32_t>(tid[0]);
            for (int32_t i = 0; i < n; ++i) {
                Perm(i, b) = i;
            }

            for (int32_t i = 0; i < n; ++i) {
                int32_t best = i;
                int32_t best_idx = Perm(i, b);
                Real best_abs = sycl::fabs(Evals(best_idx, b));
                Real best_val = Evals(best_idx, b);
                for (int32_t j = i + 1; j < n; ++j) {
                    const int32_t cand_idx = Perm(j, b);
                    const Real cand_val = Evals(cand_idx, b);
                    const Real cand_abs = sycl::fabs(cand_val);
                    if (cand_abs > best_abs || (cand_abs == best_abs && cand_val > best_val)) {
                        best = j;
                        best_idx = cand_idx;
                        best_abs = cand_abs;
                        best_val = cand_val;
                    }
                }
                const int32_t tmp = Perm(i, b);
                Perm(i, b) = Perm(best, b);
                Perm(best, b) = tmp;
            }
        });
    });
}

template <typename Real>
void finalize_hermitian_values(Queue& ctx,
                               const VectorView<Real>& eigenvalues,
                               const VectorView<int32_t>& permutation,
                               Span<Real> singular_values) {
    const int32_t n = static_cast<int32_t>(eigenvalues.size());
    const int32_t batch = static_cast<int32_t>(eigenvalues.batch_size());

    ctx->submit([&](sycl::handler& h) {
        auto Evals = eigenvalues;
        auto Perm = permutation;
        Real* s_out = singular_values.data();
        h.parallel_for(sycl::range<1>(static_cast<size_t>(n * batch)), [=](sycl::id<1> tid) {
            const int32_t linear = static_cast<int32_t>(tid[0]);
            const int32_t b = linear / n;
            const int32_t i = linear - b * n;
            const int32_t src = Perm(i, b);
            s_out[static_cast<size_t>(b) * static_cast<size_t>(n) + static_cast<size_t>(i)] = sycl::fabs(Evals(src, b));
        });
    });
}

template <typename T>
void build_hermitian_vectors(Queue& ctx,
                             const MatrixView<T, MatrixFormat::Dense>& eigenvectors,
                             const VectorView<typename base_type<T>::type>& eigenvalues,
                             const VectorView<int32_t>& permutation,
                             const MatrixView<T, MatrixFormat::Dense>& u_out,
                             const MatrixView<T, MatrixFormat::Dense>& vh_out,
                             bool want_u,
                             bool want_vh) {
    using Real = typename base_type<T>::type;

    const int32_t n = static_cast<int32_t>(eigenvectors.rows());
    const int32_t batch = static_cast<int32_t>(eigenvectors.batch_size());
    auto& u_mut = const_cast<MatrixView<T, MatrixFormat::Dense>&>(u_out);
    auto& vh_mut = const_cast<MatrixView<T, MatrixFormat::Dense>&>(vh_out);

    ctx->submit([&](sycl::handler& h) {
        auto Q = eigenvectors.kernel_view();
        auto Evals = eigenvalues;
        auto Perm = permutation;
        auto U = u_mut.kernel_view();
        auto Vh = vh_mut.kernel_view();
        h.parallel_for(sycl::range<1>(static_cast<size_t>(n * batch)), [=](sycl::id<1> tid) {
            const int32_t linear = static_cast<int32_t>(tid[0]);
            const int32_t b = linear / n;
            const int32_t tgt = linear - b * n;
            const int32_t src = Perm(tgt, b);
            const Real lambda = Evals(src, b);
            const T sign = (lambda < Real(0)) ? T(-1) : T(1);

            if (want_u) {
                for (int32_t row = 0; row < n; ++row) {
                    U(row, tgt, b) = Q(row, src, b);
                }
            }

            if (want_vh) {
                for (int32_t col = 0; col < n; ++col) {
                    Vh(tgt, col, b) = sign * gesvd_conj_if_needed(Q(col, src, b));
                }
            }
        });
    });
}

// Seed a square matrix with the identity. bdsqr accumulates its Givens
// rotations into whatever it is handed (u <- u*Q, vh <- P^T*vh), so starting
// from I is what makes it return Q and P^T themselves.
template <typename T>
void set_identity(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& m_out) {
    auto& mut = const_cast<MatrixView<T, MatrixFormat::Dense>&>(m_out);
    const int32_t rows = static_cast<int32_t>(m_out.rows());
    const int32_t cols = static_cast<int32_t>(m_out.cols());
    const int32_t batch = static_cast<int32_t>(m_out.batch_size());
    ctx->submit([&](sycl::handler& h) {
        auto M = mut.kernel_view();
        h.parallel_for(sycl::range<2>(static_cast<size_t>(batch),
                                      static_cast<size_t>(rows) * static_cast<size_t>(cols)),
                       [=](sycl::id<2> id) {
            const int32_t b = static_cast<int32_t>(id[0]);
            const int32_t lin = static_cast<int32_t>(id[1]);
            const int32_t c = lin / rows;
            const int32_t r = lin - c * rows;
            M(r, c, b) = (r == c) ? T(1) : T(0);
        });
    });
}

template <typename T>
void build_bidiag_vectors(Queue& ctx,
                          const VectorView<T>& bidiag_d,
                          const VectorView<T>& bidiag_e,
                          const VectorView<T>& evals,
                          const MatrixView<T, MatrixFormat::Dense>& right_vecs,
                          Span<T> singular_values,
                          const MatrixView<T, MatrixFormat::Dense>& u_out,
                          const MatrixView<T, MatrixFormat::Dense>& vh_out,
                          bool want_u,
                          bool want_vh) {
    const int32_t k = static_cast<int32_t>(bidiag_d.size());
    const int32_t batch = static_cast<int32_t>(bidiag_d.batch_size());
    auto& u_mut = const_cast<MatrixView<T, MatrixFormat::Dense>&>(u_out);
    auto& vh_mut = const_cast<MatrixView<T, MatrixFormat::Dense>&>(vh_out);

    if (want_vh) {
        ctx->submit([&](sycl::handler& h) {
            auto V = right_vecs.kernel_view();
            auto Vh = vh_mut.kernel_view();
            h.parallel_for(sycl::range<1>(static_cast<size_t>(k * batch)), [=](sycl::id<1> tid) {
                const int32_t linear = static_cast<int32_t>(tid[0]);
                const int32_t b = linear / k;
                const int32_t tgt = linear - b * k;
                const int32_t src = (k - 1) - tgt;
                for (int32_t i = 0; i < k; ++i) {
                    Vh(tgt, i, b) = V(i, src, b);
                }
            });
        });
    }

    if (want_u) {
        const int32_t m = static_cast<int32_t>(u_out.rows());
        const int32_t u_cols = static_cast<int32_t>(u_out.cols());
        ctx->submit([&](sycl::handler& h) {
            auto D = bidiag_d;
            auto E = bidiag_e;
            auto Evals = evals;
            auto V = right_vecs.kernel_view();
            auto U = u_mut.kernel_view();
            (void)singular_values;

            h.parallel_for(sycl::range<1>(static_cast<size_t>(u_cols * batch)), [=](sycl::id<1> tid) {
                const int32_t linear = static_cast<int32_t>(tid[0]);
                const int32_t b = linear / u_cols;
                const int32_t tgt = linear - b * u_cols;
                const T sigma_max = sycl::sqrt(sycl::fmax(Evals(k - 1, b), T(0)));
                const T tol_zero = gesvd_zero_sigma_tol_device(sigma_max);

                if (tgt >= k) {
                    for (int32_t i = 0; i < m; ++i) {
                        U(i, tgt, b) = T(0);
                    }
                    return;
                }

                const int32_t src = (k - 1) - tgt;
                const T sigma = sycl::sqrt(sycl::fmax(Evals(src, b), T(0)));
                for (int32_t i = 0; i < m; ++i) {
                    T value = T(0);
                    if (sigma > tol_zero && i < k) {
                        value = D(i, b) * V(i, src, b);
                        if (i < k - 1) {
                            value += E(i, b) * V(i + 1, src, b);
                        }
                        value /= sigma;
                    }
                    U(i, tgt, b) = value;
                }
            });
        });
    }
}

template <typename T>
bool has_tiny_singular_values(Queue& ctx,
                              int32_t k,
                              int32_t batch,
                              Span<T> singular_values) {
    ctx.wait_and_throw();

    for (int32_t b = 0; b < batch; ++b) {
        const T* sb = singular_values.data() + static_cast<size_t>(b) * static_cast<size_t>(k);
        const T sigma_max = sb[0];
        const T tol_zero = gesvd_zero_sigma_tol_host(sigma_max);
        for (int32_t i = 0; i < k; ++i) {
            if (sb[i] <= tol_zero) {
                return true;
            }
        }
    }

    return false;
}

template <typename T>
void patch_zero_left_vectors(Queue& ctx,
                             Span<T> singular_values,
                             int32_t sigma_count,
                             const MatrixView<T, MatrixFormat::Dense>& left_vecs,
                             const MatrixView<T, MatrixFormat::Dense>& u_out) {
    const int32_t m = static_cast<int32_t>(left_vecs.rows());
    const int32_t batch = static_cast<int32_t>(left_vecs.batch_size());
    auto& u_mut = const_cast<MatrixView<T, MatrixFormat::Dense>&>(u_out);

    ctx->submit([&](sycl::handler& h) {
        auto UZero = left_vecs.kernel_view();
        auto U = u_mut.kernel_view();
        const T* s_in = singular_values.data();

        h.parallel_for(sycl::range<1>(static_cast<size_t>(m * batch)), [=](sycl::id<1> tid) {
            const int32_t linear = static_cast<int32_t>(tid[0]);
            const int32_t b = linear / m;
            const int32_t tgt = linear - b * m;
            const int32_t src = (m - 1) - tgt;
            const T* sb = s_in + static_cast<size_t>(b) * static_cast<size_t>(sigma_count);
            const T tol_zero = gesvd_zero_sigma_tol_device(sb[0]);
            if (tgt < sigma_count && sb[tgt] > tol_zero) {
                return;
            }

            for (int32_t i = 0; i < m; ++i) {
                U(i, tgt, b) = UZero(i, src, b);
            }
        });
    });
}

template <Backend B, typename T>
Event apply_left_backtransform(Queue& ctx,
                               const MatrixView<T, MatrixFormat::Dense>& a,
                               const VectorView<T>& tauq,
                               const MatrixView<T, MatrixFormat::Dense>& u_out,
                               GesvdNativeMode mode,
                               const Span<std::byte>& ws) {
    if (mode == GesvdNativeMode::CTA && a.rows() == a.cols() && u_out.rows() == u_out.cols() && u_out.rows() == a.rows()) {
        return ormqx_cta<B, T>(ctx,
                               a,
                               tauq,
                               u_out,
                               Uplo::Upper,
                               Side::Left,
                               Transpose::NoTrans,
                               static_cast<int32_t>(a.rows()),
                               ws);
    }

    return ormbr<B, T>(ctx,
                       a,
                       tauq,
                       u_out,
                       'Q',
                       Side::Left,
                       Transpose::NoTrans,
                       ws,
                       tuning::ormqr_block_size_for_n(static_cast<int32_t>(a.rows())));
}

template <Backend B, typename T>
size_t left_backtransform_workspace_size(Queue& ctx,
                                         const MatrixView<T, MatrixFormat::Dense>& a,
                                         const VectorView<T>& tauq,
                                         const MatrixView<T, MatrixFormat::Dense>& u_out,
                                         GesvdNativeMode mode) {
    if (mode == GesvdNativeMode::CTA && a.rows() == a.cols() && u_out.rows() == u_out.cols() && u_out.rows() == a.rows()) {
        return 0;
    }

    return ormbr_buffer_size<B, T>(ctx,
                                   a,
                                   tauq,
                                   u_out,
                                   'Q',
                                   Side::Left,
                                   Transpose::NoTrans,
                                   tuning::ormqr_block_size_for_n(static_cast<int32_t>(a.rows())));
}

template <Backend B, typename T>
size_t gesvd_native_buffer_size(Queue& ctx,
                                const MatrixView<T, MatrixFormat::Dense>& a,
                                Span<typename base_type<T>::type> singular_values,
                                const MatrixView<T, MatrixFormat::Dense>& u_out,
                                const MatrixView<T, MatrixFormat::Dense>& vh_out,
                                SvdVectors jobu,
                                SvdVectors jobvh,
                                GesvdNativeMode mode,
                                const char* where);

template <Backend B, typename T>
Event gesvd_native_impl(Queue& ctx,
                        const MatrixView<T, MatrixFormat::Dense>& a_in,
                        Span<typename base_type<T>::type> singular_values,
                        const MatrixView<T, MatrixFormat::Dense>& u_out,
                        const MatrixView<T, MatrixFormat::Dense>& vh_out,
                        SvdVectors jobu,
                        SvdVectors jobvh,
                        const Span<std::byte>& ws,
                        GesvdNativeMode mode,
                        const char* where,
                        Span<int32_t> info) {
    validate_gesvd_dims(a_in, singular_values, u_out, vh_out, jobu, jobvh, where);

    if (!ctx.in_order()) {
        throw batchlas::invalid_argument(std::string(where) + ": requires an in-order Queue");
    }

    if constexpr (internal::is_complex<T>::value) {
        throw batchlas::unsupported(std::string(where) + ": complex native path is not implemented");
    } else {
        const int32_t m = static_cast<int32_t>(a_in.rows());
        const int32_t n = static_cast<int32_t>(a_in.cols());
        const int32_t k = std::min(m, n);
        const int32_t batch = static_cast<int32_t>(a_in.batch_size());
        // The one clear for this call. gebrd and both back-transforms are direct,
        // so everything that can fail to converge sits under solve_tridiagonal /
        // bdsdc / bdsqr below, and all of those only raise. See info_span.hh.
        detail::info_clear(ctx, info, batch);
        jobu = canonical_jobu(jobu, m, k);
        jobvh = canonical_jobvh(jobvh, n, k);
        // `!= None`, not `== All`. Keeping the old idiom here is the silent
        // degrade: a Thin request would compute nothing and hand back an
        // untouched U without raising anything.
        const bool want_u = jobu != SvdVectors::None;
        const bool want_vh = jobvh != SvdVectors::None;
        const int32_t u_cols = static_cast<int32_t>(svd_u_cols(jobu, m, k));
        const int32_t vh_rows = static_cast<int32_t>(svd_vh_rows(jobvh, n, k));
        const GesvdStageProfiler profiler{ctx, where, std::max(m, n), batch, gesvd_stage_profile_enabled()};

        if (m < n) {
            Span<std::byte> ws_mut(const_cast<std::byte*>(ws.data()), ws.size());
            BumpAllocator pool(ws_mut);

            auto at_span = pool.allocate<T>(ctx, static_cast<size_t>(n) * static_cast<size_t>(m) * static_cast<size_t>(batch));
            MatrixView<T, MatrixFormat::Dense> at_view(at_span.data(),
                                                       n,
                                                       m,
                                                       n,
                                                       static_cast<int64_t>(n) * static_cast<int64_t>(m),
                                                       batch);

            MatrixView<T, MatrixFormat::Dense> ut_view;
            MatrixView<T, MatrixFormat::Dense> vht_view;

            // With m < n we have k == m, so U is never the thin side here --
            // jobu has already canonicalised to All. V^H is: n x n for All,
            // m x n for Thin. Its transposed counterpart ut_view is therefore
            // n x n or n x m, and sizing it n x n unconditionally would
            // reintroduce, in the workspace, exactly the allocation Thin exists
            // to avoid.
            const SvdVectors trans_jobu = jobvh;   // U of A^T <-> V^H of A
            const SvdVectors trans_jobvh = jobu;   // V^H of A^T <-> U of A
            const int32_t ut_cols = static_cast<int32_t>(svd_u_cols(trans_jobu, n, m));

            if (want_vh) {
                auto ut_span = pool.allocate<T>(ctx, static_cast<size_t>(n) * static_cast<size_t>(ut_cols) * static_cast<size_t>(batch));
                ut_view = MatrixView<T, MatrixFormat::Dense>(ut_span.data(),
                                                             n,
                                                             ut_cols,
                                                             n,
                                                             static_cast<int64_t>(n) * static_cast<int64_t>(ut_cols),
                                                             batch);
            } else {
                auto ut_dummy = pool.allocate<T>(ctx, static_cast<size_t>(batch));
                ut_view = MatrixView<T, MatrixFormat::Dense>(ut_dummy.data(), 1, 1, 1, 1, batch);
            }

            if (want_u) {
                auto vht_span = pool.allocate<T>(ctx, static_cast<size_t>(m) * static_cast<size_t>(m) * static_cast<size_t>(batch));
                vht_view = MatrixView<T, MatrixFormat::Dense>(vht_span.data(),
                                                              m,
                                                              m,
                                                              m,
                                                              static_cast<int64_t>(m) * static_cast<int64_t>(m),
                                                              batch);
            } else {
                auto vht_dummy = pool.allocate<T>(ctx, static_cast<size_t>(batch));
                vht_view = MatrixView<T, MatrixFormat::Dense>(vht_dummy.data(), 1, 1, 1, 1, batch);
            }

            profiler.run("gesvd.transpose_input", [&] {
                (void)transpose(ctx, a_in, at_view);
            });

            // trans_jobu / trans_jobvh are computed above, next to ut_view's
            // extent, so the view and the job it is sized for cannot drift.
            // They propagate the job VALUE rather than collapsing it to
            // All/None, which is what carries Thin into the inner solve.
            const size_t inner_ws_bytes = gesvd_native_buffer_size<B, T>(ctx,
                                                                         at_view,
                                                                         singular_values,
                                                                         ut_view,
                                                                         vht_view,
                                                                         trans_jobu,
                                                                         trans_jobvh,
                                                                         mode,
                                                                         where);
            auto inner_ws = pool.allocate<std::byte>(ctx, inner_ws_bytes);

            profiler.run("gesvd.transpose_solve", [&] {
                (void)gesvd_native_impl<B, T>(ctx,
                                        at_view,
                                        singular_values,
                                        ut_view,
                                        vht_view,
                                        trans_jobu,
                                        trans_jobvh,
                                        inner_ws,
                                        mode,
                                        where,
                                        info);
            });

            if (want_u) {
                profiler.run("gesvd.transpose_u", [&] {
                    (void)transpose(ctx, vht_view, u_out);
                });
            }
            if (want_vh) {
                profiler.run("gesvd.transpose_vh", [&] {
                    (void)transpose(ctx, ut_view, vh_out);
                });
            }

            return ctx.get_event();
        }

        const bool need_vecs = want_u || want_vh;
        const bool tridiag_returns_vectors = (mode == GesvdNativeMode::Blocked) || need_vecs;
        const bool use_blocked_gebrd = gesvd_use_blocked_gebrd(k, mode);
        const int32_t gebrd_block_size = tuning::gebrd_block_size_for_n(k);
        const bool thin_tall_u = want_u && u_cols < m;
        const bool direct_bidiag = gesvd_direct_bidiag(mode, thin_tall_u);
        // Only a FULL U needs order-m tridiagonal buffers; a thin U cannot reach
        // the normal-equations branch, so they shrink to k (else the m x m cost
        // of a 10000 x 32 problem just moves into the workspace).
        const int32_t max_order = (want_u && u_cols == m) ? std::max(m, k) : k;

        auto& a = const_cast<MatrixView<T, MatrixFormat::Dense>&>(a_in);
        Span<std::byte> ws_mut(const_cast<std::byte*>(ws.data()), ws.size());
        BumpAllocator pool(ws_mut);

        auto d_span = pool.allocate<T>(ctx, static_cast<size_t>(k) * static_cast<size_t>(batch));
        auto e_span = pool.allocate<T>(ctx, static_cast<size_t>(std::max(0, k - 1)) * static_cast<size_t>(batch));
        auto tauq_span = pool.allocate<T>(ctx, static_cast<size_t>(k) * static_cast<size_t>(batch));
        auto taup_span = pool.allocate<T>(ctx, static_cast<size_t>(k) * static_cast<size_t>(batch));
        auto tri_d_span = pool.allocate<T>(ctx, static_cast<size_t>(max_order) * static_cast<size_t>(batch));
        auto tri_e_span = pool.allocate<T>(ctx, static_cast<size_t>(std::max(0, max_order - 1)) * static_cast<size_t>(batch));
        auto evals_span = pool.allocate<T>(ctx, static_cast<size_t>(max_order) * static_cast<size_t>(batch));
        auto sign_span = pool.allocate<T>(ctx, static_cast<size_t>(max_order) * static_cast<size_t>(batch));

        VectorView<T> d_view(d_span, k, batch, 1, k);
        VectorView<T> e_view(e_span, std::max(0, k - 1), batch, 1, std::max(1, k - 1));
        VectorView<T> tauq_view(tauq_span, k, batch, 1, k);
        VectorView<T> taup_view(taup_span, k, batch, 1, k);
        VectorView<T> tri_d_right(tri_d_span, k, batch, 1, max_order);
        VectorView<T> tri_e_right(tri_e_span, std::max(0, k - 1), batch, 1, std::max(1, max_order - 1));
        VectorView<T> evals_right(evals_span, k, batch, 1, max_order);
        VectorView<T> sign_right(sign_span, k, batch, 1, max_order);

        MatrixView<T, MatrixFormat::Dense> vecs_view;
        // Not needed on a direct bidiagonal solve (bdsdc/bdsqr): those write
        // straight into u_out / vh_out, with no intermediate k x k matrix.
        if (tridiag_returns_vectors && !direct_bidiag) {
            auto vecs_span = pool.allocate<T>(ctx, static_cast<size_t>(k) * static_cast<size_t>(k) * static_cast<size_t>(batch));
            vecs_view = MatrixView<T, MatrixFormat::Dense>(vecs_span.data(), k, k, k, static_cast<int64_t>(k) * static_cast<int64_t>(k), batch);
        }

        const JobType tridiag_job = tridiag_returns_vectors ? JobType::EigenVectors : JobType::NoEigenVectors;
        // A direct bidiagonal solve touches neither the tridiagonal eigensolver
        // nor its buffers (at n=512 the largest allocation here), so skip them.
        // The sizing function is an upper bound covering both shapes.
        Span<std::byte> solver_ws;
        if (!direct_bidiag) {
            size_t solver_ws_bytes = gesvd_solver_workspace_size<B, T>(ctx, k, batch, tridiag_job, mode);
            if (want_u) {
                solver_ws_bytes = std::max(solver_ws_bytes,
                                           gesvd_solver_workspace_size<B, T>(ctx, m, batch, JobType::EigenVectors, mode));
            }
            solver_ws = pool.allocate<std::byte>(ctx, solver_ws_bytes);
        }
        Span<std::byte> gebrd_ws;
        if (use_blocked_gebrd) {
            const size_t gebrd_ws_bytes = gebrd_blocked_buffer_size<B, T>(ctx, a, d_view, e_view, tauq_view, taup_view, gebrd_block_size);
            gebrd_ws = pool.allocate<std::byte>(ctx, gebrd_ws_bytes);
        }

        profiler.run("gesvd.gebrd", [&] {
            if constexpr (B == Backend::NETLIB) {
                (void)gebrd_unblocked<B, T>(ctx, a, d_view, e_view, tauq_view, taup_view);
            } else if (mode == GesvdNativeMode::CTA && m == n) {
                (void)gebrd_cta<B, T>(ctx, a, d_view, e_view, tauq_view, taup_view);
            } else if (use_blocked_gebrd) {
                (void)gebrd_blocked<B, T>(ctx, a, d_view, e_view, tauq_view, taup_view, gebrd_ws, gebrd_block_size);
            } else {
                (void)gebrd_unblocked<B, T>(ctx, a, d_view, e_view, tauq_view, taup_view);
            }
        });

        // ---- Bidiagonal SVD ----
        // Direct solve on B itself; the else branch (CTA mode, =normal) forms
        // B^T B and squares kappa.
        // evidence: docs/perf/gesvd.md#gesvd-defect-a-the-normal-equations-square-kappa
        // Direct solvers write only the leading k x k of U and V^H; U's columns
        // k..m-1 are seeded to I and completed by the back-transform, since
        // A = (Q_B Q) S (P^T P_B^H) is exactly ormbr 'Q' left and 'P' right.
        if (direct_bidiag) {
            const bool use_bdsdc = gesvd_bidiag_solver() == GesvdBidiagSolver::Bdsdc;
            const size_t bidiag_ws_bytes =
                use_bdsdc ? bdsdc_buffer_size<B, T>(ctx, d_view, e_view, singular_values, need_vecs)
                          : bdsqr_buffer_size<T>(ctx, d_view, e_view, singular_values);
            auto bidiag_ws = pool.allocate<std::byte>(ctx, bidiag_ws_bytes);

            if (!need_vecs) {
                profiler.run("gesvd.bidiag_values", [&] {
                    if (use_bdsdc) {
                        (void)bdsdc<B, T>(ctx, d_view, e_view, singular_values, bidiag_ws, /*sort_desc=*/true, info);
                    } else {
                        (void)bdsqr<B, T>(ctx, d_view, e_view, singular_values, bidiag_ws, /*sort_desc=*/true, info);
                    }
                });
                return ctx.get_event();
            }

            if (want_u) {
                profiler.run("gesvd.bidiag.seed_u", [&] { set_identity<T>(ctx, u_out); });
            }
            if (want_vh) {
                profiler.run("gesvd.bidiag.seed_vh", [&] { set_identity<T>(ctx, vh_out); });
            }

            const MatrixView<T, MatrixFormat::Dense> u_sub =
                want_u ? MatrixView<T, MatrixFormat::Dense>(u_out.data_ptr(), m, k, u_out.ld(), u_out.stride(), batch)
                       : MatrixView<T, MatrixFormat::Dense>(nullptr, 0, 0, 1, 1, batch);
            const MatrixView<T, MatrixFormat::Dense> vh_sub =
                want_vh ? MatrixView<T, MatrixFormat::Dense>(vh_out.data_ptr(), k, n, vh_out.ld(), vh_out.stride(), batch)
                        : MatrixView<T, MatrixFormat::Dense>(nullptr, 0, 0, 1, 1, batch);

            profiler.run("gesvd.bidiag_vectors", [&] {
                if (use_bdsdc) {
                    (void)bdsdc<B, T>(ctx, d_view, e_view, singular_values, bidiag_ws, u_sub, vh_sub, /*sort_desc=*/true, info);
                } else {
                    (void)bdsqr<B, T>(ctx, d_view, e_view, singular_values, bidiag_ws, u_sub, vh_sub, /*sort_desc=*/true, info);
                }
            });
        } else {

        profiler.run("gesvd.form_right_tridiag", [&] {
            form_right_tridiagonal(ctx, d_view, e_view, tri_d_right, tri_e_right);
        });
        profiler.run("gesvd.solve_right_tridiag", [&] {
            (void)solve_tridiagonal<B, T>(ctx,
                                    tri_d_right,
                                    tri_e_right,
                                    evals_right,
                                    vecs_view,
                                    sign_right,
                                    tridiag_job,
                                    mode,
                                    solver_ws,
                                    info);
        });

        if (!need_vecs) {
            profiler.run("gesvd.finalize_values", [&] {
                finalize_values_only(ctx, evals_right, singular_values);
            });
            return ctx.get_event();
        }

        profiler.run("gesvd.finalize_values", [&] {
            finalize_values_only(ctx, evals_right, singular_values);
        });
        profiler.run("gesvd.build_bidiag_vectors", [&] {
            build_bidiag_vectors(ctx, d_view, e_view, evals_right, vecs_view, singular_values, u_out, vh_out, want_u, want_vh);
        });

        if (want_u && (m > k || has_tiny_singular_values(ctx, k, batch, singular_values))) {
            VectorView<T> tri_d_left(tri_d_span, m, batch, 1, max_order);
            VectorView<T> tri_e_left(tri_e_span, std::max(0, m - 1), batch, 1, std::max(1, max_order - 1));
            VectorView<T> evals_left(evals_span, m, batch, 1, max_order);
            VectorView<T> sign_left(sign_span, m, batch, 1, max_order);
            auto left_vecs_span = pool.allocate<T>(ctx, static_cast<size_t>(m) * static_cast<size_t>(m) * static_cast<size_t>(batch));
            MatrixView<T, MatrixFormat::Dense> left_vecs(left_vecs_span.data(), m, m, m, static_cast<int64_t>(m) * static_cast<int64_t>(m), batch);

            profiler.run("gesvd.form_left_tridiag", [&] {
                form_left_tridiagonal(ctx, d_view, e_view, m, tri_d_left, tri_e_left);
            });
            profiler.run("gesvd.solve_left_tridiag", [&] {
                // Same items as the right solve above, hence stedc_dispatch inside
                // rather than stedc: this second call must accumulate, not clear.
                (void)solve_tridiagonal<B, T>(ctx,
                                        tri_d_left,
                                        tri_e_left,
                                        evals_left,
                                        left_vecs,
                                        sign_left,
                                        JobType::EigenVectors,
                                        mode,
                                        solver_ws,
                                        info);
            });
            profiler.run("gesvd.patch_zero_left_vectors", [&] {
                patch_zero_left_vectors(ctx, singular_values, k, left_vecs, u_out);
            });
        }

        }

        if (want_u) {
            const size_t left_ws_bytes = left_backtransform_workspace_size<B, T>(ctx, a, tauq_view, u_out, mode);
            auto left_ws = pool.allocate<std::byte>(ctx, left_ws_bytes);
            profiler.run("gesvd.apply_left_backtransform", [&] {
                (void)apply_left_backtransform<B, T>(ctx, a, tauq_view, u_out, mode, left_ws);
            });
        }

        if (want_vh) {
            const int32_t p_block_size = tuning::ormqr_block_size_for_n(n);
            const size_t p_ws_bytes = ormbr_buffer_size<B, T>(ctx,
                                                              a,
                                                              taup_view,
                                                              vh_out,
                                                              'P',
                                                              Side::Right,
                                                              Transpose::ConjTrans,
                                        p_block_size);
            auto p_ws = pool.allocate<std::byte>(ctx, p_ws_bytes);
            profiler.run("gesvd.apply_right_backtransform", [&] {
                (void)ormbr<B, T>(ctx, a, taup_view, vh_out, 'P', Side::Right, Transpose::ConjTrans, p_ws, p_block_size);
            });
        }

        return ctx.get_event();
    }
}

template <Backend B, typename T>
Event gesvd_native_hermitian_impl(Queue& ctx,
                                  const MatrixView<T, MatrixFormat::Dense>& a_in,
                                  Span<typename base_type<T>::type> singular_values,
                                  const MatrixView<T, MatrixFormat::Dense>& u_out,
                                  const MatrixView<T, MatrixFormat::Dense>& vh_out,
                                  SvdVectors jobu,
                                  SvdVectors jobvh,
                                  const Span<std::byte>& ws,
                                  GesvdNativeMode mode,
                                  Uplo hermitian_uplo,
                                  const char* where,
                                  Span<int32_t> info) {
    using Real = typename base_type<T>::type;

    if (a_in.rows() != a_in.cols()) {
        throw batchlas::invalid_argument(std::string(where) + ": Hermitian path requires square matrices");
    }
    validate_gesvd_dims(a_in, singular_values, u_out, vh_out, jobu, jobvh, where);

    if (!ctx.in_order()) {
        throw batchlas::invalid_argument(std::string(where) + ": requires an in-order Queue");
    }
    if (hermitian_uplo != Uplo::Lower && hermitian_uplo != Uplo::Upper) {
        throw batchlas::invalid_argument(std::string(where) + ": invalid Hermitian triangle selector");
    }

    const int32_t n = static_cast<int32_t>(a_in.rows());
    const int32_t batch = static_cast<int32_t>(a_in.batch_size());
    const bool want_u = jobu == SvdVectors::All;
    const bool want_vh = jobvh == SvdVectors::All;
    const bool need_vecs = want_u || want_vh;
    const GesvdStageProfiler profiler{ctx, where, n, batch, gesvd_stage_profile_enabled()};

    auto& a = const_cast<MatrixView<T, MatrixFormat::Dense>&>(a_in);
    Span<std::byte> ws_mut(const_cast<std::byte*>(ws.data()), ws.size());
    BumpAllocator pool(ws_mut);

    auto eigvals_span = pool.allocate<Real>(ctx, static_cast<size_t>(n) * static_cast<size_t>(batch));
    auto perm_span = pool.allocate<int32_t>(ctx, static_cast<size_t>(n) * static_cast<size_t>(batch));
    VectorView<Real> eigvals_view(eigvals_span, n, batch, 1, n);
    VectorView<int32_t> perm_view(perm_span, n, batch, 1, n);

    const JobType jobz = need_vecs ? JobType::EigenVectors : JobType::NoEigenVectors;
    const size_t syev_ws_bytes = (mode == GesvdNativeMode::CTA)
        ? syev_cta_buffer_size<B, T>(ctx, a, jobz)
        : syev_blocked_buffer_size<B, T>(ctx, a, jobz, hermitian_uplo);
    auto syev_ws = pool.allocate<std::byte>(ctx, syev_ws_bytes);

    profiler.run("gesvd.hermitian.syev", [&] {
        // The Hermitian path is one eigen-decomposition plus a sort and a sign
        // fix-up, so its whole convergence story is this single call; both arms
        // clear `info` themselves (via steqr / stedc).
        if (mode == GesvdNativeMode::CTA) {
            (void)syev_cta<B, T>(ctx, a, eigvals_span, jobz, hermitian_uplo, syev_ws,
                                 SteqrParams<T>(), /*cta_wg_size_multiplier=*/1, info);
        } else {
            (void)syev_blocked<B, T>(ctx, a, eigvals_span, jobz, hermitian_uplo, syev_ws,
                                     StedcParams<Real>(), info);
        }
    });
    profiler.run("gesvd.hermitian.sort", [&] {
        build_hermitian_svd_permutation(ctx, eigvals_view, perm_view);
    });
    profiler.run("gesvd.hermitian.values", [&] {
        finalize_hermitian_values(ctx, eigvals_view, perm_view, singular_values);
    });

    if (need_vecs) {
        profiler.run("gesvd.hermitian.vectors", [&] {
            build_hermitian_vectors(ctx, a, eigvals_view, perm_view, u_out, vh_out, want_u, want_vh);
        });
    }

    return ctx.get_event();
}

template <Backend B, typename T>
size_t gesvd_native_buffer_size(Queue& ctx,
                                const MatrixView<T, MatrixFormat::Dense>& a,
                                Span<typename base_type<T>::type> singular_values,
                                const MatrixView<T, MatrixFormat::Dense>& u_out,
                                const MatrixView<T, MatrixFormat::Dense>& vh_out,
                                SvdVectors jobu,
                                SvdVectors jobvh,
                                GesvdNativeMode mode,
                                const char* where) {
    validate_gesvd_dims(a, singular_values, u_out, vh_out, jobu, jobvh, where);

    if constexpr (internal::is_complex<T>::value) {
        throw batchlas::unsupported(std::string(where) + ": complex native path is not implemented");
    } else {
        const size_t m = static_cast<size_t>(a.rows());
        const size_t n = static_cast<size_t>(a.cols());
        const size_t k = std::min(m, n);
        const size_t batch = static_cast<size_t>(a.batch_size());
        // Every line below must mirror gesvd_native_impl exactly. The two are
        // separate functions over the same decisions, and a divergence is a
        // silent workspace overrun rather than a compile error.
        jobu = canonical_jobu(jobu, static_cast<int64_t>(m), static_cast<int64_t>(k));
        jobvh = canonical_jobvh(jobvh, static_cast<int64_t>(n), static_cast<int64_t>(k));
        const bool want_u = jobu != SvdVectors::None;
        const bool want_vh = jobvh != SvdVectors::None;
        const size_t u_cols = static_cast<size_t>(svd_u_cols(jobu, static_cast<int64_t>(m), static_cast<int64_t>(k)));
        if (m < n) {
            const SvdVectors trans_jobu = jobvh;
            const SvdVectors trans_jobvh = jobu;
            const size_t ut_cols = static_cast<size_t>(
                svd_u_cols(trans_jobu, static_cast<int64_t>(n), static_cast<int64_t>(m)));

            MatrixView<T, MatrixFormat::Dense> ut_view(nullptr,
                                                       static_cast<int64_t>(n),
                                                       want_vh ? static_cast<int64_t>(ut_cols) : 1,
                                                       static_cast<int64_t>(n),
                                                       static_cast<int64_t>(n) * static_cast<int64_t>(std::max<size_t>(1, ut_cols)),
                                                       static_cast<int64_t>(batch));
            // m < n means k == m, so the transposed problem's V^H (which is A's
            // U) is m x m either way -- Thin never shrinks it.
            MatrixView<T, MatrixFormat::Dense> vht_view(nullptr,
                                                        trans_jobvh != SvdVectors::None ? static_cast<int64_t>(m) : 1,
                                                        trans_jobvh != SvdVectors::None ? static_cast<int64_t>(m) : 1,
                                                        trans_jobvh != SvdVectors::None ? static_cast<int64_t>(m) : 1,
                                                        static_cast<int64_t>(std::max<size_t>(1, m)) * static_cast<int64_t>(std::max<size_t>(1, m)),
                                                        static_cast<int64_t>(batch));

            size_t bytes = 0;
            bytes += BumpAllocator::allocation_size<T>(ctx, n * m * batch);
            if (want_vh) {
                bytes += BumpAllocator::allocation_size<T>(ctx, n * ut_cols * batch);
            } else {
                bytes += BumpAllocator::allocation_size<T>(ctx, batch);
            }
            if (want_u) {
                bytes += BumpAllocator::allocation_size<T>(ctx, m * m * batch);
            } else {
                bytes += BumpAllocator::allocation_size<T>(ctx, batch);
            }
            bytes += BumpAllocator::allocation_size<std::byte>(ctx,
                gesvd_native_buffer_size<B, T>(ctx,
                                               MatrixView<T, MatrixFormat::Dense>(nullptr,
                                                                                  static_cast<int64_t>(n),
                                                                                  static_cast<int64_t>(m),
                                                                                  static_cast<int64_t>(n),
                                                                                  static_cast<int64_t>(n) * static_cast<int64_t>(m),
                                                                                  static_cast<int64_t>(batch)),
                                               singular_values,
                                               ut_view,
                                               vht_view,
                                               trans_jobu,
                                               trans_jobvh,
                                               mode,
                                               where));
            return bytes;
        }

        const bool need_vecs = want_u || want_vh;
        const bool tridiag_returns_vectors = (mode == GesvdNativeMode::Blocked) || need_vecs;

        size_t bytes = 0;
        // Mirrors gesvd_native_impl: thin tall U forces the direct bidiagonal
        // solve, which makes the order-m buffers unreachable.
        const bool thin_tall_u = want_u && u_cols < m;
        const bool direct_bidiag = gesvd_direct_bidiag(mode, thin_tall_u);
        const size_t max_order = (want_u && u_cols == m) ? std::max(m, k) : k;
        bytes += BumpAllocator::allocation_size<T>(ctx, k * batch);                                // d
        bytes += BumpAllocator::allocation_size<T>(ctx, (k > 0 ? k - 1 : 0) * batch);             // e
        bytes += BumpAllocator::allocation_size<T>(ctx, k * batch);                                // tauq
        bytes += BumpAllocator::allocation_size<T>(ctx, k * batch);                                // taup
        bytes += BumpAllocator::allocation_size<T>(ctx, max_order * batch);                        // tri_d
        bytes += BumpAllocator::allocation_size<T>(ctx, (max_order > 0 ? max_order - 1 : 0) * batch); // tri_e
        bytes += BumpAllocator::allocation_size<T>(ctx, max_order * batch);                        // evals
        bytes += BumpAllocator::allocation_size<T>(ctx, max_order * batch);                        // sign_view
        const int32_t k_i32 = static_cast<int32_t>(k);
        const int32_t m_i32 = static_cast<int32_t>(m);
        const bool use_blocked_gebrd = gesvd_use_blocked_gebrd(k_i32, mode);
        const int32_t gebrd_block_size = tuning::gebrd_block_size_for_n(k_i32);

        // Mirror the run path's branch exactly. A direct solve needs the
        // bidiagonal solver's own workspace instead; for bdsdc (2k x 2k vectors
        // plus stedc at order 2k) it dominates, and only this branch keeps
        // buffer_size an upper bound.
        if (direct_bidiag) {
            VectorView<T> d_dummy(nullptr, k_i32, static_cast<int32_t>(batch), 1, k_i32);
            const int32_t e_size = static_cast<int32_t>(k > 0 ? k - 1 : 0);
            VectorView<T> e_dummy(nullptr, e_size, static_cast<int32_t>(batch), 1, std::max<int32_t>(1, e_size));
            const size_t bidiag_bytes =
                (gesvd_bidiag_solver() == GesvdBidiagSolver::Bdsdc)
                    ? bdsdc_buffer_size<B, T>(ctx, d_dummy, e_dummy, singular_values, need_vecs)
                    : bdsqr_buffer_size<T>(ctx, d_dummy, e_dummy, singular_values);
            bytes += BumpAllocator::allocation_size<std::byte>(ctx, bidiag_bytes);
        } else {
            if (tridiag_returns_vectors) {
                bytes += BumpAllocator::allocation_size<T>(ctx, k * k * batch);          // right singular vectors
            }
            size_t solver_bytes = gesvd_solver_workspace_size<B, T>(ctx,
                                                                    k_i32,
                                                                    static_cast<int32_t>(batch),
                                                                    tridiag_returns_vectors ? JobType::EigenVectors : JobType::NoEigenVectors,
                                                                    mode);
            if (want_u) {
                solver_bytes = std::max(solver_bytes,
                                        gesvd_solver_workspace_size<B, T>(ctx,
                                                                          m_i32,
                                                                          static_cast<int32_t>(batch),
                                                                          JobType::EigenVectors,
                                                                          mode));
                bytes += BumpAllocator::allocation_size<T>(ctx, m * m * batch);          // left nullspace vectors
            }
            bytes += BumpAllocator::allocation_size<std::byte>(ctx, solver_bytes);
        }
        if (use_blocked_gebrd) {
            VectorView<T> d_dummy(nullptr, static_cast<int32_t>(k), static_cast<int32_t>(batch), 1, static_cast<int32_t>(k));
            const int32_t e_size = static_cast<int32_t>(k > 0 ? k - 1 : 0);
            const int32_t e_stride = std::max<int32_t>(1, e_size);
            VectorView<T> e_dummy(nullptr, e_size, static_cast<int32_t>(batch), 1, e_stride);
            VectorView<T> tauq_dummy(nullptr, static_cast<int32_t>(k), static_cast<int32_t>(batch), 1, static_cast<int32_t>(k));
            VectorView<T> taup_dummy(nullptr, static_cast<int32_t>(k), static_cast<int32_t>(batch), 1, static_cast<int32_t>(k));
            bytes += BumpAllocator::allocation_size<std::byte>(ctx,
                gebrd_blocked_buffer_size<B, T>(ctx, a, d_dummy, e_dummy, tauq_dummy, taup_dummy, gebrd_block_size));
        }

        if (want_u) {
            VectorView<T> tauq_dummy(nullptr, static_cast<int32_t>(k), static_cast<int32_t>(batch), 1, static_cast<int32_t>(k));
            bytes += BumpAllocator::allocation_size<std::byte>(ctx,
                left_backtransform_workspace_size<B, T>(ctx, a, tauq_dummy, u_out, mode));
        }

        if (want_vh) {
            VectorView<T> taup_dummy(nullptr, static_cast<int32_t>(k), static_cast<int32_t>(batch), 1, static_cast<int32_t>(k));
            bytes += BumpAllocator::allocation_size<std::byte>(ctx,
                ormbr_buffer_size<B, T>(ctx,
                                        a,
                                        taup_dummy,
                                        vh_out,
                                        'P',
                                        Side::Right,
                                        Transpose::ConjTrans,
                                        tuning::ormqr_block_size_for_n(static_cast<int32_t>(n))));
        }

        return bytes;
    }
}

template <Backend B, typename T>
size_t gesvd_native_hermitian_buffer_size(Queue& ctx,
                                          const MatrixView<T, MatrixFormat::Dense>& a,
                                          Span<typename base_type<T>::type> singular_values,
                                          const MatrixView<T, MatrixFormat::Dense>& u_out,
                                          const MatrixView<T, MatrixFormat::Dense>& vh_out,
                                          SvdVectors jobu,
                                          SvdVectors jobvh,
                                          GesvdNativeMode mode,
                                          Uplo hermitian_uplo,
                                          const char* where) {
    if (a.rows() != a.cols()) {
        throw batchlas::invalid_argument(std::string(where) + ": Hermitian path requires square matrices");
    }
    validate_gesvd_dims(a, singular_values, u_out, vh_out, jobu, jobvh, where);
    if (hermitian_uplo != Uplo::Lower && hermitian_uplo != Uplo::Upper) {
        throw batchlas::invalid_argument(std::string(where) + ": invalid Hermitian triangle selector");
    }

    const size_t n = static_cast<size_t>(a.rows());
    const size_t batch = static_cast<size_t>(a.batch_size());
    const bool need_vecs = jobu == SvdVectors::All || jobvh == SvdVectors::All;
    const JobType jobz = need_vecs ? JobType::EigenVectors : JobType::NoEigenVectors;

    size_t bytes = 0;
    bytes += BumpAllocator::allocation_size<typename base_type<T>::type>(ctx, n * batch);
    bytes += BumpAllocator::allocation_size<int32_t>(ctx, n * batch);
    bytes += BumpAllocator::allocation_size<std::byte>(ctx,
        mode == GesvdNativeMode::CTA
            ? syev_cta_buffer_size<B, T>(ctx, a, jobz)
            : syev_blocked_buffer_size<B, T>(ctx, a, jobz, hermitian_uplo));
    return bytes;
}

} // namespace

template <Backend B, typename T>
Event gesvd_blocked(Queue& ctx,
                    const MatrixView<T, MatrixFormat::Dense>& a_in,
                    Span<typename base_type<T>::type> singular_values,
                    const MatrixView<T, MatrixFormat::Dense>& u_out,
                    const MatrixView<T, MatrixFormat::Dense>& vh_out,
                    SvdVectors jobu,
                    SvdVectors jobvh,
                    const Span<std::byte>& ws,
                    Span<int32_t> info) {
    return gesvd_native_impl<B, T>(ctx,
                                   a_in,
                                   singular_values,
                                   u_out,
                                   vh_out,
                                   jobu,
                                   jobvh,
                                   ws,
                                   GesvdNativeMode::Blocked,
                                   "gesvd_blocked",
                                   info);
}

template <Backend B, typename T>
Event gesvd_blocked(Queue& ctx,
                    const MatrixView<T, MatrixFormat::Dense>& a_in,
                    Span<typename base_type<T>::type> singular_values,
                    const MatrixView<T, MatrixFormat::Dense>& u_out,
                    const MatrixView<T, MatrixFormat::Dense>& vh_out,
                    SvdVectors jobu,
                    SvdVectors jobvh,
                    Uplo hermitian_uplo,
                    const Span<std::byte>& ws,
                    Span<int32_t> info) {
    return gesvd_native_hermitian_impl<B, T>(ctx,
                                             a_in,
                                             singular_values,
                                             u_out,
                                             vh_out,
                                             jobu,
                                             jobvh,
                                             ws,
                                             GesvdNativeMode::Blocked,
                                             hermitian_uplo,
                                             "gesvd_blocked",
                                             info);
}

template <Backend B, typename T>
size_t gesvd_blocked_buffer_size(Queue& ctx,
                                 const MatrixView<T, MatrixFormat::Dense>& a,
                                 Span<typename base_type<T>::type> singular_values,
                                 const MatrixView<T, MatrixFormat::Dense>& u_out,
                                 const MatrixView<T, MatrixFormat::Dense>& vh_out,
                                 SvdVectors jobu,
                                 SvdVectors jobvh) {
    return gesvd_native_buffer_size<B, T>(ctx,
                                          a,
                                          singular_values,
                                          u_out,
                                          vh_out,
                                          jobu,
                                          jobvh,
                                          GesvdNativeMode::Blocked,
                                          "gesvd_blocked_buffer_size");
}

template <Backend B, typename T>
size_t gesvd_blocked_buffer_size(Queue& ctx,
                                 const MatrixView<T, MatrixFormat::Dense>& a,
                                 Span<typename base_type<T>::type> singular_values,
                                 const MatrixView<T, MatrixFormat::Dense>& u_out,
                                 const MatrixView<T, MatrixFormat::Dense>& vh_out,
                                 SvdVectors jobu,
                                 SvdVectors jobvh,
                                 Uplo hermitian_uplo) {
    return gesvd_native_hermitian_buffer_size<B, T>(ctx,
                                                    a,
                                                    singular_values,
                                                    u_out,
                                                    vh_out,
                                                    jobu,
                                                    jobvh,
                                                    GesvdNativeMode::Blocked,
                                                    hermitian_uplo,
                                                    "gesvd_blocked_buffer_size");
}

template <Backend B, typename T>
Event gesvd_cta(Queue& ctx,
                const MatrixView<T, MatrixFormat::Dense>& a_in,
                Span<typename base_type<T>::type> singular_values,
                const MatrixView<T, MatrixFormat::Dense>& u_out,
                const MatrixView<T, MatrixFormat::Dense>& vh_out,
                SvdVectors jobu,
                SvdVectors jobvh,
                const Span<std::byte>& ws,
                Span<int32_t> info) {
    validate_gesvd_dims(a_in, singular_values, u_out, vh_out, jobu, jobvh, "gesvd_cta");
    if (std::max(a_in.rows(), a_in.cols()) > 32) {
        throw batchlas::invalid_argument("gesvd_cta: currently supports max(m, n) <= 32");
    }
    // Mode CTA always takes the normal-equations branch, whose
    // patch_zero_left_vectors writes m columns of U unconditionally. Refuse a
    // genuinely thin request rather than overrun. Dispatch never gets here --
    // gesvd_supports_cta already declines, and a forced-but-unsupported
    // provider resets to Auto -- so this guards DIRECT callers.
    {
        const int64_t k = std::min<int64_t>(a_in.rows(), a_in.cols());
        if (canonical_jobu(jobu, a_in.rows(), k) == SvdVectors::Thin ||
            canonical_jobvh(jobvh, a_in.cols(), k) == SvdVectors::Thin) {
            throw batchlas::invalid_argument(
                "gesvd_cta: thin singular vectors are not supported (use gesvd_blocked or gesvdj_cta)");
        }
    }
    return gesvd_native_impl<B, T>(ctx,
                                   a_in,
                                   singular_values,
                                   u_out,
                                   vh_out,
                                   jobu,
                                   jobvh,
                                   ws,
                                   GesvdNativeMode::CTA,
                                   "gesvd_cta",
                                   info);
}

template <Backend B, typename T>
Event gesvd_cta(Queue& ctx,
                const MatrixView<T, MatrixFormat::Dense>& a_in,
                Span<typename base_type<T>::type> singular_values,
                const MatrixView<T, MatrixFormat::Dense>& u_out,
                const MatrixView<T, MatrixFormat::Dense>& vh_out,
                SvdVectors jobu,
                SvdVectors jobvh,
                Uplo hermitian_uplo,
                const Span<std::byte>& ws,
                Span<int32_t> info) {
    validate_gesvd_dims(a_in, singular_values, u_out, vh_out, jobu, jobvh, "gesvd_cta");
    if (a_in.rows() != a_in.cols()) {
        throw batchlas::invalid_argument("gesvd_cta: Hermitian path requires square matrices");
    }
    if (std::max(a_in.rows(), a_in.cols()) > 32) {
        throw batchlas::invalid_argument("gesvd_cta: currently supports max(m, n) <= 32");
    }
    // Mode CTA always takes the normal-equations branch, whose
    // patch_zero_left_vectors writes m columns of U unconditionally. Refuse a
    // genuinely thin request rather than overrun. Dispatch never gets here --
    // gesvd_supports_cta already declines, and a forced-but-unsupported
    // provider resets to Auto -- so this guards DIRECT callers.
    {
        const int64_t k = std::min<int64_t>(a_in.rows(), a_in.cols());
        if (canonical_jobu(jobu, a_in.rows(), k) == SvdVectors::Thin ||
            canonical_jobvh(jobvh, a_in.cols(), k) == SvdVectors::Thin) {
            throw batchlas::invalid_argument(
                "gesvd_cta: thin singular vectors are not supported (use gesvd_blocked or gesvdj_cta)");
        }
    }
    return gesvd_native_hermitian_impl<B, T>(ctx,
                                             a_in,
                                             singular_values,
                                             u_out,
                                             vh_out,
                                             jobu,
                                             jobvh,
                                             ws,
                                             GesvdNativeMode::CTA,
                                             hermitian_uplo,
                                             "gesvd_cta",
                                             info);
}

template <Backend B, typename T>
size_t gesvd_cta_buffer_size(Queue& ctx,
                             const MatrixView<T, MatrixFormat::Dense>& a,
                             Span<typename base_type<T>::type> singular_values,
                             const MatrixView<T, MatrixFormat::Dense>& u_out,
                             const MatrixView<T, MatrixFormat::Dense>& vh_out,
                             SvdVectors jobu,
                             SvdVectors jobvh) {
    validate_gesvd_dims(a, singular_values, u_out, vh_out, jobu, jobvh, "gesvd_cta_buffer_size");
    if (std::max(a.rows(), a.cols()) > 32) {
        throw batchlas::invalid_argument("gesvd_cta_buffer_size: currently supports max(m, n) <= 32");
    }
    {
        const int64_t k = std::min<int64_t>(a.rows(), a.cols());
        if (canonical_jobu(jobu, a.rows(), k) == SvdVectors::Thin ||
            canonical_jobvh(jobvh, a.cols(), k) == SvdVectors::Thin) {
            throw batchlas::invalid_argument(
                "gesvd_cta_buffer_size: thin singular vectors are not supported");
        }
    }
    return gesvd_native_buffer_size<B, T>(ctx,
                                          a,
                                          singular_values,
                                          u_out,
                                          vh_out,
                                          jobu,
                                          jobvh,
                                          GesvdNativeMode::CTA,
                                          "gesvd_cta_buffer_size");
}

template <Backend B, typename T>
size_t gesvd_cta_buffer_size(Queue& ctx,
                             const MatrixView<T, MatrixFormat::Dense>& a,
                             Span<typename base_type<T>::type> singular_values,
                             const MatrixView<T, MatrixFormat::Dense>& u_out,
                             const MatrixView<T, MatrixFormat::Dense>& vh_out,
                             SvdVectors jobu,
                             SvdVectors jobvh,
                             Uplo hermitian_uplo) {
    validate_gesvd_dims(a, singular_values, u_out, vh_out, jobu, jobvh, "gesvd_cta_buffer_size");
    if (a.rows() != a.cols()) {
        throw batchlas::invalid_argument("gesvd_cta_buffer_size: Hermitian path requires square matrices");
    }
    if (std::max(a.rows(), a.cols()) > 32) {
        throw batchlas::invalid_argument("gesvd_cta_buffer_size: currently supports max(m, n) <= 32");
    }
    {
        const int64_t k = std::min<int64_t>(a.rows(), a.cols());
        if (canonical_jobu(jobu, a.rows(), k) == SvdVectors::Thin ||
            canonical_jobvh(jobvh, a.cols(), k) == SvdVectors::Thin) {
            throw batchlas::invalid_argument(
                "gesvd_cta_buffer_size: thin singular vectors are not supported");
        }
    }
    return gesvd_native_hermitian_buffer_size<B, T>(ctx,
                                                    a,
                                                    singular_values,
                                                    u_out,
                                                    vh_out,
                                                    jobu,
                                                    jobvh,
                                                    GesvdNativeMode::CTA,
                                                    hermitian_uplo,
                                                    "gesvd_cta_buffer_size");
}

#define GESVD_BLOCKED_INSTANTIATE(back, fp) \
    template Event gesvd_blocked<back, BATCHLAS_UNPAREN fp>( \
        Queue&, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        Span<typename base_type<BATCHLAS_UNPAREN fp>::type>, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        SvdVectors, \
        SvdVectors, \
        const Span<std::byte>&, \
        Span<int32_t>); \
    template Event gesvd_blocked<back, BATCHLAS_UNPAREN fp>( \
        Queue&, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        Span<typename base_type<BATCHLAS_UNPAREN fp>::type>, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        SvdVectors, \
        SvdVectors, \
        Uplo, \
        const Span<std::byte>&, \
        Span<int32_t>); \
    template size_t gesvd_blocked_buffer_size<back, BATCHLAS_UNPAREN fp>( \
        Queue&, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        Span<typename base_type<BATCHLAS_UNPAREN fp>::type>, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        SvdVectors, \
        SvdVectors); \
    template size_t gesvd_blocked_buffer_size<back, BATCHLAS_UNPAREN fp>( \
        Queue&, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        Span<typename base_type<BATCHLAS_UNPAREN fp>::type>, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        SvdVectors, \
        SvdVectors, \
        Uplo); \
    template Event gesvd_cta<back, BATCHLAS_UNPAREN fp>( \
        Queue&, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        Span<typename base_type<BATCHLAS_UNPAREN fp>::type>, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        SvdVectors, \
        SvdVectors, \
        const Span<std::byte>&, \
        Span<int32_t>); \
    template Event gesvd_cta<back, BATCHLAS_UNPAREN fp>( \
        Queue&, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        Span<typename base_type<BATCHLAS_UNPAREN fp>::type>, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        SvdVectors, \
        SvdVectors, \
        Uplo, \
        const Span<std::byte>&, \
        Span<int32_t>); \
    template size_t gesvd_cta_buffer_size<back, BATCHLAS_UNPAREN fp>( \
        Queue&, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        Span<typename base_type<BATCHLAS_UNPAREN fp>::type>, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        SvdVectors, \
        SvdVectors); \
    template size_t gesvd_cta_buffer_size<back, BATCHLAS_UNPAREN fp>( \
        Queue&, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        Span<typename base_type<BATCHLAS_UNPAREN fp>::type>, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        SvdVectors, \
        SvdVectors, \
        Uplo);

BATCHLAS_INSTANTIATE_SCALAR_ALL_BACKENDS(GESVD_BLOCKED_INSTANTIATE)

#undef GESVD_BLOCKED_INSTANTIATE

} // namespace batchlas
