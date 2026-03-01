#include <blas/extensions.hh>
#include <batchlas/backend_config.h>

#include <util/mempool.hh>

#include "../math-helpers.hh"
#include "../queue.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace batchlas {

namespace {

template <typename Real>
inline Real safe_abs(Real x) {
    return sycl::fabs(x);
}

template <typename Real>
bool bdsqr_implicit_qr_attempt(Queue& ctx,
                               const VectorView<Real>& d,
                               const VectorView<Real>& e,
                               Span<Real> singular_values_out,
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
        int32_t* fail = fail_flags.data();
        const int32_t nn = n;
        const int32_t nb = batch;
        const bool descending = sort_desc;

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
                sb[i] = safe_abs(db[i]);
            }

            // Small n in current tests, so simple O(n^2) ordering is fine in-kernel.
            if (descending) {
                for (int32_t i = 0; i < nn; ++i) {
                    for (int32_t j = i + 1; j < nn; ++j) {
                        if (sb[j] > sb[i]) {
                            const Real tmp = sb[i];
                            sb[i] = sb[j];
                            sb[j] = tmp;
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

template <Backend B, typename T>
Event bdsqr(Queue& ctx,
            const VectorView<T>& d,
            const VectorView<T>& e,
            Span<T> singular_values_out,
            const Span<std::byte>& ws,
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

    if constexpr (internal::is_complex<T>::value) {
        throw std::runtime_error("bdsqr: complex types are not implemented yet");
    } else {
        Span<std::byte> ws_mut(const_cast<std::byte*>(ws.data()), ws.size());
        BumpAllocator pool(ws_mut);
        auto fail_flags = pool.allocate<int32_t>(ctx, static_cast<size_t>(batch));
        const bool ok = bdsqr_implicit_qr_attempt<T>(ctx,
                                                     d,
                                                     e,
                                                     singular_values_out,
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
    return BumpAllocator::allocation_size<int32_t>(ctx, static_cast<size_t>(d.batch_size()));
}

#define BDSQR_INSTANTIATE(back, fp) \
    template Event bdsqr<back, fp>( \
        Queue&, \
        const VectorView<fp>&, \
        const VectorView<fp>&, \
        Span<fp>, \
        const Span<std::byte>&, \
        bool);

#define BDSQR_BUFFER_INSTANTIATE(fp) \
    template size_t bdsqr_buffer_size<fp>( \
        Queue&, \
        const VectorView<fp>&, \
        const VectorView<fp>&, \
        Span<fp>);

#if BATCHLAS_HAS_CUDA_BACKEND
BDSQR_INSTANTIATE(Backend::CUDA, float)
BDSQR_INSTANTIATE(Backend::CUDA, double)
#endif

#if BATCHLAS_HAS_ROCM_BACKEND
BDSQR_INSTANTIATE(Backend::ROCM, float)
BDSQR_INSTANTIATE(Backend::ROCM, double)
#endif

#if BATCHLAS_HAS_HOST_BACKEND
BDSQR_INSTANTIATE(Backend::NETLIB, float)
BDSQR_INSTANTIATE(Backend::NETLIB, double)
#endif

BDSQR_BUFFER_INSTANTIATE(float)
BDSQR_BUFFER_INSTANTIATE(double)

#undef BDSQR_INSTANTIATE
#undef BDSQR_BUFFER_INSTANTIATE

} // namespace batchlas
