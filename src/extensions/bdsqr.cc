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
#include <limits>

namespace batchlas {

namespace {

template <typename Real>
inline Real safe_abs(Real x) {
    return sycl::fabs(x);
}

template <typename Real>
inline void apply_rotation_to_rows(const KernelMatrixView<Real, MatrixFormat::Dense>& matrix,
                                   int32_t batch,
                                   int32_t row0,
                                   int32_t row1,
                                   Real c,
                                   Real s) {
    for (int32_t col = 0; col < matrix.cols(); ++col) {
        const Real x0 = matrix(row0, col, batch);
        const Real x1 = matrix(row1, col, batch);
        matrix(row0, col, batch) = c * x0 + s * x1;
        matrix(row1, col, batch) = -s * x0 + c * x1;
    }
}

template <typename Real>
inline void apply_rotation_to_cols(const KernelMatrixView<Real, MatrixFormat::Dense>& matrix,
                                   int32_t batch,
                                   int32_t col0,
                                   int32_t col1,
                                   Real c,
                                   Real s) {
    for (int32_t row = 0; row < matrix.rows(); ++row) {
        const Real x0 = matrix(row, col0, batch);
        const Real x1 = matrix(row, col1, batch);
        matrix(row, col0, batch) = c * x0 + s * x1;
        matrix(row, col1, batch) = -s * x0 + c * x1;
    }
}

template <typename Real>
inline void scale_row(const KernelMatrixView<Real, MatrixFormat::Dense>& matrix,
                      int32_t batch,
                      int32_t row,
                      Real alpha) {
    for (int32_t col = 0; col < matrix.cols(); ++col) {
        matrix(row, col, batch) *= alpha;
    }
}

template <typename Real>
inline void swap_rows(const KernelMatrixView<Real, MatrixFormat::Dense>& matrix,
                      int32_t batch,
                      int32_t row0,
                      int32_t row1) {
    for (int32_t col = 0; col < matrix.cols(); ++col) {
        const Real tmp = matrix(row0, col, batch);
        matrix(row0, col, batch) = matrix(row1, col, batch);
        matrix(row1, col, batch) = tmp;
    }
}

template <typename Real>
inline void swap_cols(const KernelMatrixView<Real, MatrixFormat::Dense>& matrix,
                      int32_t batch,
                      int32_t col0,
                      int32_t col1) {
    for (int32_t row = 0; row < matrix.rows(); ++row) {
        const Real tmp = matrix(row, col0, batch);
        matrix(row, col0, batch) = matrix(row, col1, batch);
        matrix(row, col1, batch) = tmp;
    }
}

template <typename Real>
bool bdsqr_implicit_qr_attempt(Queue& ctx,
                               const VectorView<Real>& d,
                               const VectorView<Real>& e,
                               Span<Real> singular_values_out,
                               const MatrixView<Real, MatrixFormat::Dense>& u,
                               const MatrixView<Real, MatrixFormat::Dense>& vh,
                               int32_t n,
                               int32_t batch,
                               bool sort_desc,
                               Span<int32_t> fail_flags) {
    const Real eps = std::numeric_limits<Real>::epsilon();
    const Real tolmul = sycl::fmax(Real(10), sycl::fmin(Real(100), sycl::pow(eps, Real(-0.125))));
    const Real tol = tolmul * eps;
    const int32_t maxitr = 6; // LAPACK DBDSQR default
    const int32_t maxit = std::max<int32_t>(32, maxitr * n * n);

    ctx->submit([&](sycl::handler& cgh) {
        auto D = d;
        auto E = e;
        Real* out = singular_values_out.data();
        auto U = u.kernel_view();
        auto Vh = vh.kernel_view();
        int32_t* fail = fail_flags.data();
        const int32_t nn = n;
        const int32_t nb = batch;
        const bool descending = sort_desc;
        const bool accumulate_u = u.rows() > 0 && u.cols() == n;
        const bool accumulate_vh = vh.rows() == n && vh.cols() > 0;

        cgh.parallel_for(sycl::range<1>(static_cast<size_t>(nb)), [=](sycl::id<1> tid) {
            const int32_t b = static_cast<int32_t>(tid[0]);

            Real* db = D.data_ptr() + static_cast<size_t>(b) * static_cast<size_t>(D.stride());
            Real* eb = (nn > 1) ? (E.data_ptr() + static_cast<size_t>(b) * static_cast<size_t>(E.stride())) : nullptr;

            int32_t iters = 0;
            bool converged = (nn <= 1);

            while (!converged && iters < maxit) {
                converged = true;

                Real smax = Real(0);
                for (int32_t i = 0; i < nn; ++i) {
                    smax = sycl::fmax(smax, safe_abs(db[i]));
                }
                for (int32_t i = 0; i < nn - 1; ++i) {
                    smax = sycl::fmax(smax, safe_abs(eb[i]));
                }
                const Real abs_thresh = tol * smax;

                for (int32_t i = 0; i < nn - 1; ++i) {
                    const Real rel_thresh = eps * (safe_abs(db[i]) + safe_abs(db[i + 1]));
                    const Real thresh = sycl::fmax(rel_thresh, abs_thresh);
                    if (safe_abs(eb[i]) <= thresh) {
                        eb[i] = Real(0);
                    } else {
                        converged = false;
                    }
                }
                if (converged) break;

                int32_t l = 0;
                while (l < nn - 1) {
                    while (l < nn - 1 && eb[l] == Real(0)) ++l;
                    if (l >= nn - 1) break;

                    int32_t m = l;
                    while (m < nn - 1 && eb[m] != Real(0)) ++m;

                    if (m == l) {
                        eb[l] = Real(0);
                        l = m + 1;
                        continue;
                    }

                    // ---- Zero-diagonal deflation (LAPACK DBDSQR's zero-shift branch).
                    //
                    // Without this the sweep below STAGNATES. If db[l] is zero
                    // then f = -mu and g = db[l]*eb[l] = 0, so lartg returns the
                    // identity rotation, every subsequent rotation in the chase
                    // is also trivial, and nothing is annihilated -- the loop
                    // spins to maxit and bdsqr reports "did not converge".
                    //
                    // A zero on the diagonal means the block has an exact zero
                    // singular value. The fix is a sequence of LEFT rotations
                    // that chases the offending superdiagonal entry along row i
                    // and off the end of the block, leaving a zero row to
                    // deflate. This is a zero-SHIFT step, so it is also the
                    // numerically safe way to handle it.
                    {
                        int32_t zrow = -1;
                        for (int32_t i = l; i <= m; ++i) {
                            const Real dthresh = sycl::fmax(abs_thresh,
                                                            std::numeric_limits<Real>::min());
                            if (safe_abs(db[i]) <= dthresh) { zrow = i; break; }
                        }
                        if (zrow >= 0) {
                            db[zrow] = Real(0);
                            if (zrow < m) {
                                // Zero strictly inside the block: chase e[zrow]
                                // RIGHTWARDS with LEFT rotations, which empties
                                // row zrow.
                                Real f2 = eb[zrow];
                                eb[zrow] = Real(0);
                                for (int32_t j = zrow + 1; j <= m; ++j) {
                                    const auto rz = internal::lartg<Real>(db[j], f2);
                                    db[j] = rz.r;
                                    if (accumulate_u) {
                                        apply_rotation_to_cols(U, b, j, zrow, rz.c, rz.s);
                                    }
                                    if (j < m) {
                                        f2 = -rz.s * eb[j];
                                        eb[j] = rz.c * eb[j];
                                    }
                                }
                            } else if (m > l) {
                                // Zero at the BOTTOM of the block. The rightward
                                // chase has nothing to do here, so handling only
                                // that case leaves eb[l..m-1] untouched while
                                // still advancing past the block -- the sweep
                                // then never converges. Chase LEFTWARDS with
                                // RIGHT rotations instead, emptying column m.
                                Real f2 = eb[m - 1];
                                eb[m - 1] = Real(0);
                                for (int32_t j = m - 1; j >= l; --j) {
                                    const auto rz = internal::lartg<Real>(db[j], f2);
                                    db[j] = rz.r;
                                    if (accumulate_vh) {
                                        apply_rotation_to_rows(Vh, b, j, m, rz.c, rz.s);
                                    }
                                    if (j > l) {
                                        f2 = -rz.s * eb[j - 1];
                                        eb[j - 1] = rz.c * eb[j - 1];
                                    }
                                }
                            }
                            iters += (m - l + 1);
                            l = m + 1;
                            continue;
                        }
                    }

                    const int32_t p = m - 1;
                    const Real a = db[p] * db[p] + eb[p] * eb[p];
                    const Real b12 = db[p] * eb[p];
                    const Real c = db[m] * db[m];
                    const auto eval2 = internal::eigenvalues_2x2(a, b12, c);
                    const Real mu = (safe_abs(eval2[0] - c) < safe_abs(eval2[1] - c)) ? eval2[0] : eval2[1];

                    Real f = db[l] * db[l] - mu;
                    Real g = db[l] * eb[l];

                    for (int32_t k = l; k <= m - 1; ++k) {
                        const auto r1 = internal::lartg<Real>(f, g);
                        const Real cs = r1.c;
                        const Real sn = r1.s;
                        if (accumulate_vh) {
                            apply_rotation_to_rows(Vh, b, k, k + 1, cs, sn);
                        }
                        if (k > l) eb[k - 1] = r1.r;

                        const Real dk = db[k];
                        const Real ek = eb[k];
                        const Real dk1 = db[k + 1];

                        f = cs * dk + sn * ek;
                        eb[k] = cs * ek - sn * dk;
                        g = sn * dk1;
                        db[k + 1] = cs * dk1;

                        const auto r2 = internal::lartg<Real>(f, g);
                        const Real cs2 = r2.c;
                        const Real sn2 = r2.s;
                        if (accumulate_u) {
                            apply_rotation_to_cols(U, b, k, k + 1, cs2, sn2);
                        }
                        db[k] = r2.r;

                        const Real dk1b = db[k + 1];
                        const Real ekb = eb[k];
                        f = cs2 * ekb + sn2 * dk1b;
                        db[k + 1] = cs2 * dk1b - sn2 * ekb;

                        if (k < m - 1) {
                            g = sn2 * eb[k + 1];
                            eb[k + 1] = cs2 * eb[k + 1];
                        } else {
                            g = Real(0);
                        }
                        eb[k] = f;
                    }

                    iters += (m - l + 1);
                    l = m + 1;
                }
            }

            Real* sb = out + static_cast<size_t>(b) * static_cast<size_t>(nn);
            for (int32_t i = 0; i < nn; ++i) {
                if (db[i] < Real(0)) {
                    db[i] = -db[i];
                    if (accumulate_vh) {
                        scale_row(Vh, b, i, Real(-1));
                    }
                }
                sb[i] = db[i];
            }

            if (descending) {
                for (int32_t i = 0; i < nn; ++i) {
                    int32_t best = i;
                    for (int32_t j = i + 1; j < nn; ++j) {
                        if (sb[j] > sb[best]) {
                            best = j;
                        }
                    }
                    if (best != i) {
                        const Real tmp = sb[i];
                        sb[i] = sb[best];
                        sb[best] = tmp;
                        if (accumulate_u) {
                            swap_cols(U, b, i, best);
                        }
                        if (accumulate_vh) {
                            swap_rows(Vh, b, i, best);
                        }
                    }
                }
            }

            bool bad = false;
            for (int32_t i = 0; i < nn; ++i) {
                if (!(sb[i] == sb[i])) {
                    bad = true;
                    break;
                }
            }
            fail[b] = (converged && !bad) ? 0 : 1;
        });
    });

    ctx.wait();
    bool ok = true;
    for (int32_t b = 0; b < batch; ++b) {
        if (fail_flags[static_cast<size_t>(b)] != 0) {
            ok = false;
            break;
        }
    }
    return ok;
}

} // namespace

// Single description of bdsqr's workspace; see workspace_bytes() in
// util/mempool.hh.
template <typename T>
Span<int32_t> bdsqr_layout(Queue& ctx, BumpAllocator& pool, int32_t batch) {
    return pool.allocate<int32_t>(ctx, static_cast<size_t>(batch));
}

template <Backend B, typename T>
Event bdsqr(Queue& ctx,
            const VectorView<T>& d,
            const VectorView<T>& e,
            Span<T> singular_values_out,
            const Span<std::byte>& ws,
            bool sort_desc) {
    const auto empty = MatrixView<T, MatrixFormat::Dense>(nullptr, 0, 0, 1, 1, d.batch_size());
    return bdsqr<B, T>(ctx,
                       d,
                       e,
                       singular_values_out,
                       ws,
                       empty,
                       empty,
                       sort_desc);
}

template <Backend B, typename T>
Event bdsqr(Queue& ctx,
            const VectorView<T>& d,
            const VectorView<T>& e,
            Span<T> singular_values_out,
            const Span<std::byte>& ws,
            const MatrixView<T, MatrixFormat::Dense>& u,
            const MatrixView<T, MatrixFormat::Dense>& vh,
            bool sort_desc) {
    static_cast<void>(B);

    const int32_t n = static_cast<int32_t>(d.size());
    const int32_t batch = static_cast<int32_t>(d.batch_size());

    if (batch < 1 || n < 1) {
        throw std::invalid_argument("bdsqr: invalid dimensions");
    }
    if (e.size() != std::max<int32_t>(0, n - 1) || e.batch_size() != batch) {
        throw std::invalid_argument("bdsqr: e must have length n-1 and matching batch size");
    }
    const size_t need_s = static_cast<size_t>(n) * static_cast<size_t>(batch);
    if (singular_values_out.size() < need_s) {
        throw std::invalid_argument("bdsqr: singular_values_out span too small");
    }
    if (u.rows() > 0 || u.cols() > 0) {
        if (u.cols() != n || u.batch_size() != batch) {
            throw std::invalid_argument("bdsqr: U must have n columns and matching batch size");
        }
    }
    if (vh.rows() > 0 || vh.cols() > 0) {
        if (vh.rows() != n || vh.batch_size() != batch) {
            throw std::invalid_argument("bdsqr: Vh must have n rows and matching batch size");
        }
    }

    if constexpr (internal::is_complex<T>::value) {
        throw std::runtime_error("bdsqr: complex types are not implemented yet");
    } else {
        Span<std::byte> ws_mut(const_cast<std::byte*>(ws.data()), ws.size());
        BumpAllocator pool(ws_mut);
        auto fail_flags = bdsqr_layout<T>(ctx, pool, batch);
        const bool ok = bdsqr_implicit_qr_attempt<T>(ctx,
                                                     d,
                                                     e,
                                                     singular_values_out,
                                                     u,
                                                     vh,
                                                     n,
                                                     batch,
                                                     sort_desc,
                                                     fail_flags);
        if (!ok) {
            throw std::runtime_error("bdsqr: native implicit bidiagonal QR did not converge");
        }
        return ctx.get_event();
    }
}

template <typename T>
size_t bdsqr_buffer_size(Queue& ctx,
                         const VectorView<T>& d,
                         const VectorView<T>& e,
                         Span<T> singular_values_out) {
    static_cast<void>(e);
    static_cast<void>(singular_values_out);
    return workspace_bytes([&](BumpAllocator& pool) {
        return bdsqr_layout<T>(ctx, pool, static_cast<int32_t>(d.batch_size()));
    });
}

#define BDSQR_INSTANTIATE(back, fp) \
    template Event bdsqr<back, BATCHLAS_UNPAREN fp>( \
        Queue&, \
        const VectorView<BATCHLAS_UNPAREN fp>&, \
        const VectorView<BATCHLAS_UNPAREN fp>&, \
        Span<BATCHLAS_UNPAREN fp>, \
        const Span<std::byte>&, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        bool); \
    template Event bdsqr<back, BATCHLAS_UNPAREN fp>( \
        Queue&, \
        const VectorView<BATCHLAS_UNPAREN fp>&, \
        const VectorView<BATCHLAS_UNPAREN fp>&, \
        Span<BATCHLAS_UNPAREN fp>, \
        const Span<std::byte>&, \
        bool);

#define BDSQR_BUFFER_INSTANTIATE(fp) \
    template size_t bdsqr_buffer_size<BATCHLAS_UNPAREN fp>( \
        Queue&, \
        const VectorView<BATCHLAS_UNPAREN fp>&, \
        const VectorView<BATCHLAS_UNPAREN fp>&, \
        Span<BATCHLAS_UNPAREN fp>);

#define BDSQR_INSTANTIATE_FOR_BACKEND(back) \
    BATCHLAS_FOR_EACH_REAL_TYPE_1(BDSQR_INSTANTIATE, back)

#if BATCHLAS_HAS_CUDA_BACKEND
BDSQR_INSTANTIATE_FOR_BACKEND(Backend::CUDA)
#endif

#if BATCHLAS_HAS_ROCM_BACKEND
BDSQR_INSTANTIATE_FOR_BACKEND(Backend::ROCM)
#endif

#if BATCHLAS_HAS_HOST_BACKEND
BDSQR_INSTANTIATE_FOR_BACKEND(Backend::NETLIB)
#endif

BATCHLAS_FOR_EACH_REAL_TYPE(BDSQR_BUFFER_INSTANTIATE)

#undef BDSQR_INSTANTIATE_FOR_BACKEND
#undef BDSQR_BUFFER_INSTANTIATE
#undef BDSQR_INSTANTIATE

#undef BDSQR_INSTANTIATE
#undef BDSQR_BUFFER_INSTANTIATE

} // namespace batchlas
