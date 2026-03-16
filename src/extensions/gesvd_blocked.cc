#include <blas/extensions.hh>
#include <blas/functions.hh>
#include <blas/linalg.hh>
#include <batchlas/backend_config.h>
#include <batchlas/tuning_params.hh>
#include <util/mempool.hh>

#include "../math-helpers.hh"
#include "../util/template-instantiations.hh"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace batchlas {

namespace {

enum class GesvdNativeMode {
    Blocked,
    CTA,
};

inline bool gesvd_use_blocked_gebrd(int32_t n, GesvdNativeMode mode) {
    return mode == GesvdNativeMode::Blocked && n >= 128;
}

inline bool gesvd_stage_profile_enabled() {
    auto env_truthy = [](const char* v) {
        if (!v) return false;
        return (std::string(v) == "1" || std::string(v) == "true" || std::string(v) == "TRUE" ||
                std::string(v) == "on" || std::string(v) == "ON");
    };
    return env_truthy(std::getenv("BATCHLAS_GESVD_PROFILE"));
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
    if (a.rows() != a.cols()) {
        throw std::invalid_argument(std::string(where) + ": square matrices only");
    }
    if (a.batch_size() < 1 || a.rows() < 1) {
        throw std::invalid_argument(std::string(where) + ": invalid matrix dimensions or batch size");
    }

    const int64_t n = a.rows();
    const int64_t batch = a.batch_size();
    const std::size_t need_s = static_cast<std::size_t>(n) * static_cast<std::size_t>(batch);
    if (singular_values.size() < need_s) {
        throw std::invalid_argument(std::string(where) + ": singular_values span too small");
    }

    if (jobu == SvdVectors::All && (u.rows() != n || u.cols() != n || u.batch_size() != batch)) {
        throw std::invalid_argument(std::string(where) + ": U must be (n x n) with matching batch size");
    }
    if (jobvh == SvdVectors::All && (vh.rows() != n || vh.cols() != n || vh.batch_size() != batch)) {
        throw std::invalid_argument(std::string(where) + ": Vh must be (n x n) with matching batch size");
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
    return stedc_workspace_size<B, T>(ctx,
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
                        const Span<std::byte>& ws) {
    if (mode == GesvdNativeMode::CTA) {
        BATCHLAS_KERNEL_TRACE_SCOPE("gesvd.solve_tridiag.steqr_cta");
        return steqr_cta<B, T>(ctx, diag, offdiag, evals, ws, jobz, gesvd_cta_steqr_params<T>(), dense_out);
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
    stedc<B, T>(ctx,
                diag,
                offdiag,
                evals,
                ws,
                JobType::EigenVectors,
                gesvd_blocked_stedc_params<T>(),
                dense_out);

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
            if (i < n - 1) {
                const T ei = E(i, b);
                diag += ei * ei;
                TE(i, b) = ei * D(i + 1, b);
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
    const int32_t n = static_cast<int32_t>(bidiag_d.size());
    const int32_t batch = static_cast<int32_t>(bidiag_d.batch_size());
    auto& u_mut = const_cast<MatrixView<T, MatrixFormat::Dense>&>(u_out);
    auto& vh_mut = const_cast<MatrixView<T, MatrixFormat::Dense>&>(vh_out);

    ctx->submit([&](sycl::handler& h) {
        auto D = bidiag_d;
        auto E = bidiag_e;
        auto Evals = evals;
        auto V = right_vecs.kernel_view();
        auto U = u_mut.kernel_view();
        auto Vh = vh_mut.kernel_view();
        (void)singular_values;

        h.parallel_for(sycl::range<1>(static_cast<size_t>(n * batch)), [=](sycl::id<1> tid) {
            const int32_t linear = static_cast<int32_t>(tid[0]);
            const int32_t b = linear / n;
            const int32_t tgt = linear - b * n;
            const int32_t src = (n - 1) - tgt;

            const T sigma_max = sycl::sqrt(sycl::fmax(Evals(n - 1, b), T(0)));
            const T sigma = sycl::sqrt(sycl::fmax(Evals(src, b), T(0)));
            const T tol_zero = gesvd_zero_sigma_tol_device(sigma_max);

            if (want_vh) {
                for (int32_t i = 0; i < n; ++i) {
                    Vh(tgt, i, b) = V(i, src, b);
                }
            }

            if (want_u) {
                for (int32_t i = 0; i < n; ++i) {
                    T value = T(0);
                    if (sigma > tol_zero) {
                        value = D(i, b) * V(i, src, b);
                        if (i < n - 1) {
                            value += E(i, b) * V(i + 1, src, b);
                        }
                        value /= sigma;
                    }
                    U(i, tgt, b) = value;
                }
            }
        });
    });
}

template <typename T>
bool has_tiny_singular_values(Queue& ctx,
                              int32_t n,
                              int32_t batch,
                              Span<T> singular_values) {
    ctx.wait_and_throw();

    for (int32_t b = 0; b < batch; ++b) {
        const T* sb = singular_values.data() + static_cast<size_t>(b) * static_cast<size_t>(n);
        const T sigma_max = sb[0];
        const T tol_zero = gesvd_zero_sigma_tol_host(sigma_max);
        for (int32_t i = 0; i < n; ++i) {
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
                             const MatrixView<T, MatrixFormat::Dense>& left_vecs,
                             const MatrixView<T, MatrixFormat::Dense>& u_out) {
    const int32_t n = static_cast<int32_t>(left_vecs.rows());
    const int32_t batch = static_cast<int32_t>(left_vecs.batch_size());
    auto& u_mut = const_cast<MatrixView<T, MatrixFormat::Dense>&>(u_out);

    ctx->submit([&](sycl::handler& h) {
        auto UZero = left_vecs.kernel_view();
        auto U = u_mut.kernel_view();
        const T* s_in = singular_values.data();

        h.parallel_for(sycl::range<1>(static_cast<size_t>(n * batch)), [=](sycl::id<1> tid) {
            const int32_t linear = static_cast<int32_t>(tid[0]);
            const int32_t b = linear / n;
            const int32_t tgt = linear - b * n;
            const int32_t src = (n - 1) - tgt;
            const T* sb = s_in + static_cast<size_t>(b) * static_cast<size_t>(n);
            const T tol_zero = gesvd_zero_sigma_tol_device(sb[0]);
            if (sb[tgt] > tol_zero) {
                return;
            }

            for (int32_t i = 0; i < n; ++i) {
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
    if (mode == GesvdNativeMode::CTA) {
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
    if (mode == GesvdNativeMode::CTA) {
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
Event gesvd_native_impl(Queue& ctx,
                        const MatrixView<T, MatrixFormat::Dense>& a_in,
                        Span<typename base_type<T>::type> singular_values,
                        const MatrixView<T, MatrixFormat::Dense>& u_out,
                        const MatrixView<T, MatrixFormat::Dense>& vh_out,
                        SvdVectors jobu,
                        SvdVectors jobvh,
                        const Span<std::byte>& ws,
                        GesvdNativeMode mode,
                        const char* where) {
    validate_gesvd_dims(a_in, singular_values, u_out, vh_out, jobu, jobvh, where);

    if (!ctx.in_order()) {
        throw std::runtime_error(std::string(where) + ": requires an in-order Queue");
    }

    if constexpr (internal::is_complex<T>::value) {
        throw std::runtime_error(std::string(where) + ": complex native path is not implemented");
    } else {
        const int32_t n = static_cast<int32_t>(a_in.rows());
        const int32_t batch = static_cast<int32_t>(a_in.batch_size());
        const bool want_u = jobu == SvdVectors::All;
        const bool want_vh = jobvh == SvdVectors::All;
        const bool need_vecs = want_u || want_vh;
        const bool tridiag_returns_vectors = (mode == GesvdNativeMode::Blocked) || need_vecs;
        const bool solve_with_vectors = need_vecs;
        const GesvdStageProfiler profiler{ctx, where, n, batch, gesvd_stage_profile_enabled()};
        const bool use_blocked_gebrd = gesvd_use_blocked_gebrd(n, mode);
        const int32_t gebrd_block_size = tuning::ormqr_block_size_for_n(n);

        auto& a = const_cast<MatrixView<T, MatrixFormat::Dense>&>(a_in);
        Span<std::byte> ws_mut(const_cast<std::byte*>(ws.data()), ws.size());
        BumpAllocator pool(ws_mut);

        auto d_span = pool.allocate<T>(ctx, static_cast<size_t>(n) * static_cast<size_t>(batch));
        auto e_span = pool.allocate<T>(ctx, static_cast<size_t>(std::max(0, n - 1)) * static_cast<size_t>(batch));
        auto tauq_span = pool.allocate<T>(ctx, static_cast<size_t>(n) * static_cast<size_t>(batch));
        auto taup_span = pool.allocate<T>(ctx, static_cast<size_t>(n) * static_cast<size_t>(batch));
        auto tri_d_span = pool.allocate<T>(ctx, static_cast<size_t>(n) * static_cast<size_t>(batch));
        auto tri_e_span = pool.allocate<T>(ctx, static_cast<size_t>(std::max(0, n - 1)) * static_cast<size_t>(batch));
        auto evals_span = pool.allocate<T>(ctx, static_cast<size_t>(n) * static_cast<size_t>(batch));
        auto sign_span = pool.allocate<T>(ctx, static_cast<size_t>(n) * static_cast<size_t>(batch));

        VectorView<T> d_view(d_span, n, batch, 1, n);
        VectorView<T> e_view(e_span, std::max(0, n - 1), batch, 1, std::max(1, n - 1));
        VectorView<T> tauq_view(tauq_span, n, batch, 1, n);
        VectorView<T> taup_view(taup_span, n, batch, 1, n);
        VectorView<T> tri_d_view(tri_d_span, n, batch, 1, n);
        VectorView<T> tri_e_view(tri_e_span, std::max(0, n - 1), batch, 1, std::max(1, n - 1));
        VectorView<T> evals_view(evals_span, n, batch, 1, n);
        VectorView<T> sign_view(sign_span, n, batch, 1, n);

        MatrixView<T, MatrixFormat::Dense> vecs_view;
        if (tridiag_returns_vectors) {
            auto vecs_span = pool.allocate<T>(ctx, static_cast<size_t>(n) * static_cast<size_t>(n) * static_cast<size_t>(batch));
            vecs_view = MatrixView<T, MatrixFormat::Dense>(vecs_span.data(), n, n, n, static_cast<int64_t>(n) * static_cast<int64_t>(n), batch);
        }

        const JobType tridiag_job = tridiag_returns_vectors ? JobType::EigenVectors : JobType::NoEigenVectors;
        const size_t solver_ws_bytes = gesvd_solver_workspace_size<B, T>(ctx, n, batch, tridiag_job, mode);
        auto solver_ws = pool.allocate<std::byte>(ctx, solver_ws_bytes);
        Span<std::byte> gebrd_ws;
        if (use_blocked_gebrd) {
            const size_t gebrd_ws_bytes = gebrd_blocked_buffer_size<B, T>(ctx, a, d_view, e_view, tauq_view, taup_view, gebrd_block_size);
            gebrd_ws = pool.allocate<std::byte>(ctx, gebrd_ws_bytes);
        }

        profiler.run("gesvd.gebrd", [&] {
            if constexpr (B == Backend::NETLIB) {
                gebrd_unblocked<B, T>(ctx, a, d_view, e_view, tauq_view, taup_view);
            } else if (mode == GesvdNativeMode::CTA) {
                gebrd_cta<B, T>(ctx, a, d_view, e_view, tauq_view, taup_view);
            } else if (use_blocked_gebrd) {
                gebrd_blocked<B, T>(ctx, a, d_view, e_view, tauq_view, taup_view, gebrd_ws, gebrd_block_size);
            } else {
                gebrd_unblocked<B, T>(ctx, a, d_view, e_view, tauq_view, taup_view);
            }
        });

        profiler.run("gesvd.form_right_tridiag", [&] {
            form_right_tridiagonal(ctx, d_view, e_view, tri_d_view, tri_e_view);
        });
        profiler.run("gesvd.solve_right_tridiag", [&] {
            solve_tridiagonal<B, T>(ctx,
                                    tri_d_view,
                                    tri_e_view,
                                    evals_view,
                                    vecs_view,
                                    sign_view,
                                    tridiag_job,
                                    mode,
                                    solver_ws);
        });

        if (!need_vecs) {
            profiler.run("gesvd.finalize_values", [&] {
                finalize_values_only(ctx, evals_view, singular_values);
            });
            return ctx.get_event();
        }

        profiler.run("gesvd.finalize_values", [&] {
            finalize_values_only(ctx, evals_view, singular_values);
        });
        profiler.run("gesvd.build_bidiag_vectors", [&] {
            build_bidiag_vectors(ctx, d_view, e_view, evals_view, vecs_view, singular_values, u_out, vh_out, want_u, want_vh);
        });

        if (want_u && has_tiny_singular_values(ctx, n, batch, singular_values)) {
            profiler.run("gesvd.form_left_tridiag", [&] {
                form_left_tridiagonal(ctx, d_view, e_view, tri_d_view, tri_e_view);
            });
            profiler.run("gesvd.solve_left_tridiag", [&] {
                solve_tridiagonal<B, T>(ctx,
                                        tri_d_view,
                                        tri_e_view,
                                        evals_view,
                                        vecs_view,
                                        sign_view,
                                        JobType::EigenVectors,
                                        mode,
                                        solver_ws);
            });
            profiler.run("gesvd.patch_zero_left_vectors", [&] {
                patch_zero_left_vectors(ctx, singular_values, vecs_view, u_out);
            });
        }

        if (want_u) {
            const size_t left_ws_bytes = left_backtransform_workspace_size<B, T>(ctx, a, tauq_view, u_out, mode);
            auto left_ws = pool.allocate<std::byte>(ctx, left_ws_bytes);
            profiler.run("gesvd.apply_left_backtransform", [&] {
                apply_left_backtransform<B, T>(ctx, a, tauq_view, u_out, mode, left_ws);
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
                ormbr<B, T>(ctx, a, taup_view, vh_out, 'P', Side::Right, Transpose::ConjTrans, p_ws, p_block_size);
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
                                  const char* where) {
    using Real = typename base_type<T>::type;

    validate_gesvd_dims(a_in, singular_values, u_out, vh_out, jobu, jobvh, where);

    if (!ctx.in_order()) {
        throw std::runtime_error(std::string(where) + ": requires an in-order Queue");
    }
    if (hermitian_uplo != Uplo::Lower && hermitian_uplo != Uplo::Upper) {
        throw std::invalid_argument(std::string(where) + ": invalid Hermitian triangle selector");
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
        if (mode == GesvdNativeMode::CTA) {
            syev_cta<B, T>(ctx, a, eigvals_span, jobz, hermitian_uplo, syev_ws);
        } else {
            syev_blocked<B, T>(ctx, a, eigvals_span, jobz, hermitian_uplo, syev_ws);
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
        throw std::runtime_error(std::string(where) + ": complex native path is not implemented");
    } else {
        const size_t n = static_cast<size_t>(a.rows());
        const size_t batch = static_cast<size_t>(a.batch_size());
        const bool want_u = jobu == SvdVectors::All;
        const bool want_vh = jobvh == SvdVectors::All;
        const bool need_vecs = want_u || want_vh;
        const bool tridiag_returns_vectors = (mode == GesvdNativeMode::Blocked) || need_vecs;
        const bool solve_with_vectors = need_vecs;

        size_t bytes = 0;
        bytes += BumpAllocator::allocation_size<T>(ctx, n * batch);                      // d
        bytes += BumpAllocator::allocation_size<T>(ctx, (n > 0 ? n - 1 : 0) * batch);   // e
        bytes += BumpAllocator::allocation_size<T>(ctx, n * batch);                      // tauq
        bytes += BumpAllocator::allocation_size<T>(ctx, n * batch);                      // taup
        bytes += BumpAllocator::allocation_size<T>(ctx, n * batch);                      // tri_d
        bytes += BumpAllocator::allocation_size<T>(ctx, (n > 0 ? n - 1 : 0) * batch);   // tri_e
        bytes += BumpAllocator::allocation_size<T>(ctx, n * batch);                      // evals
        bytes += BumpAllocator::allocation_size<T>(ctx, n * batch);                      // sign_view
        const int32_t n_i32 = static_cast<int32_t>(n);
        const bool use_blocked_gebrd = gesvd_use_blocked_gebrd(n_i32, mode);
        const int32_t gebrd_block_size = tuning::ormqr_block_size_for_n(n_i32);

        if (tridiag_returns_vectors) {
            bytes += BumpAllocator::allocation_size<T>(ctx, n * n * batch);              // bidiag or fallback vecs
        }
        bytes += BumpAllocator::allocation_size<std::byte>(ctx,
            gesvd_solver_workspace_size<B, T>(ctx,
                                              static_cast<int32_t>(n),
                                              static_cast<int32_t>(batch),
                                              tridiag_returns_vectors ? JobType::EigenVectors : JobType::NoEigenVectors,
                                              mode));
        if (use_blocked_gebrd) {
            VectorView<T> d_dummy(nullptr, static_cast<int32_t>(n), static_cast<int32_t>(batch), 1, static_cast<int32_t>(n));
            const int32_t e_size = static_cast<int32_t>(n > 0 ? n - 1 : 0);
            const int32_t e_stride = std::max<int32_t>(1, e_size);
            VectorView<T> e_dummy(nullptr, e_size, static_cast<int32_t>(batch), 1, e_stride);
            VectorView<T> tauq_dummy(nullptr, static_cast<int32_t>(n), static_cast<int32_t>(batch), 1, static_cast<int32_t>(n));
            VectorView<T> taup_dummy(nullptr, static_cast<int32_t>(n), static_cast<int32_t>(batch), 1, static_cast<int32_t>(n));
            bytes += BumpAllocator::allocation_size<std::byte>(ctx,
                gebrd_blocked_buffer_size<B, T>(ctx, a, d_dummy, e_dummy, tauq_dummy, taup_dummy, gebrd_block_size));
        }

        if (want_u) {
            VectorView<T> tauq_dummy(nullptr, static_cast<int32_t>(n), static_cast<int32_t>(batch), 1, static_cast<int32_t>(n));
            bytes += BumpAllocator::allocation_size<std::byte>(ctx,
                left_backtransform_workspace_size<B, T>(ctx, a, tauq_dummy, u_out, mode));
        }

        if (want_vh) {
            VectorView<T> taup_dummy(nullptr, static_cast<int32_t>(n), static_cast<int32_t>(batch), 1, static_cast<int32_t>(n));
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
    validate_gesvd_dims(a, singular_values, u_out, vh_out, jobu, jobvh, where);
    if (hermitian_uplo != Uplo::Lower && hermitian_uplo != Uplo::Upper) {
        throw std::invalid_argument(std::string(where) + ": invalid Hermitian triangle selector");
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
                    const Span<std::byte>& ws) {
    return gesvd_native_impl<B, T>(ctx,
                                   a_in,
                                   singular_values,
                                   u_out,
                                   vh_out,
                                   jobu,
                                   jobvh,
                                   ws,
                                   GesvdNativeMode::Blocked,
                                   "gesvd_blocked");
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
                    const Span<std::byte>& ws) {
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
                                             "gesvd_blocked");
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
                const Span<std::byte>& ws) {
    validate_gesvd_dims(a_in, singular_values, u_out, vh_out, jobu, jobvh, "gesvd_cta");
    if (a_in.rows() > 32) {
        throw std::invalid_argument("gesvd_cta: currently supports 1 <= n <= 32");
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
                                   "gesvd_cta");
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
                const Span<std::byte>& ws) {
    validate_gesvd_dims(a_in, singular_values, u_out, vh_out, jobu, jobvh, "gesvd_cta");
    if (a_in.rows() > 32) {
        throw std::invalid_argument("gesvd_cta: currently supports 1 <= n <= 32");
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
                                             "gesvd_cta");
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
    if (a.rows() > 32) {
        throw std::invalid_argument("gesvd_cta_buffer_size: currently supports 1 <= n <= 32");
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
    if (a.rows() > 32) {
        throw std::invalid_argument("gesvd_cta_buffer_size: currently supports 1 <= n <= 32");
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
        const Span<std::byte>&); \
    template Event gesvd_blocked<back, BATCHLAS_UNPAREN fp>( \
        Queue&, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        Span<typename base_type<BATCHLAS_UNPAREN fp>::type>, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        SvdVectors, \
        SvdVectors, \
        Uplo, \
        const Span<std::byte>&); \
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
        const Span<std::byte>&); \
    template Event gesvd_cta<back, BATCHLAS_UNPAREN fp>( \
        Queue&, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        Span<typename base_type<BATCHLAS_UNPAREN fp>::type>, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        SvdVectors, \
        SvdVectors, \
        Uplo, \
        const Span<std::byte>&); \
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

#define GESVD_BLOCKED_INSTANTIATE_FOR_BACKEND(back) \
    BATCHLAS_FOR_EACH_SCALAR_TYPE_1(GESVD_BLOCKED_INSTANTIATE, back)

#if BATCHLAS_HAS_CUDA_BACKEND
GESVD_BLOCKED_INSTANTIATE_FOR_BACKEND(Backend::CUDA)
#endif

#if BATCHLAS_HAS_ROCM_BACKEND
GESVD_BLOCKED_INSTANTIATE_FOR_BACKEND(Backend::ROCM)
#endif

#if BATCHLAS_HAS_HOST_BACKEND
GESVD_BLOCKED_INSTANTIATE_FOR_BACKEND(Backend::NETLIB)
#endif

#undef GESVD_BLOCKED_INSTANTIATE_FOR_BACKEND
#undef GESVD_BLOCKED_INSTANTIATE

} // namespace batchlas
