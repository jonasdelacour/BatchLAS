#include <blas/extensions.hh>
#include <blas/matrix.hh>
#include <util/mempool.hh>

#include <sycl/sycl.hpp>

#include <batchlas/backend_config.h>

#include "../math-helpers.hh"
#include "../queue.hh"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace batchlas {

namespace {

template <typename U>
inline U conj_if_needed(const U& x) {
    if constexpr (internal::is_complex<U>::value) {
        return U(x.real(), -x.imag());
    } else {
        return x;
    }
}

template <typename T>
inline T real_as_T(const T& x) {
    if constexpr (internal::is_complex<T>::value) {
        return T(x.real(), typename T::value_type(0));
    } else {
        return x;
    }
}

template <typename T>
inline typename base_type<T>::type real_part(const T& x) {
    if constexpr (internal::is_complex<T>::value) {
        return x.real();
    } else {
        return x;
    }
}

template <typename T>
inline void validate_sytrd_sb2st_dims(const MatrixView<T, MatrixFormat::Dense>& ab,
                                     const VectorView<typename base_type<T>::type>& d,
                                     const VectorView<typename base_type<T>::type>& e,
                                     const VectorView<T>& tau,
                                     Uplo uplo,
                                     int32_t kd) {
    if (kd < 0) {
        throw std::invalid_argument("sytrd_sb2st: kd must be non-negative");
    }
    if (uplo != Uplo::Lower && uplo != Uplo::Upper) {
        throw std::invalid_argument("sytrd_sb2st: invalid uplo");
    }

    const int n = ab.cols();
    if (ab.rows() != kd + 1) {
        throw std::invalid_argument("sytrd_sb2st: AB must be (kd+1) x n");
    }

    if (d.size() != n || e.size() != std::max(0, n - 1) || tau.size() != std::max(0, n - 1)) {
        throw std::invalid_argument("sytrd_sb2st: invalid d/e/tau sizes");
    }

    if (ab.batch_size() != d.batch_size() || ab.batch_size() != e.batch_size() || ab.batch_size() != tau.batch_size()) {
        throw std::invalid_argument("sytrd_sb2st: batch size mismatch");
    }
    if (ab.batch_size() < 1) {
        throw std::invalid_argument("sytrd_sb2st: invalid batch size");
    }
}

template <typename T>
class Kd0ExtractKernel;

template <typename T>
Event kd0_extract(Queue& q,
                  const MatrixView<T, MatrixFormat::Dense>& ab,
                  const VectorView<typename base_type<T>::type>& d,
                  const VectorView<typename base_type<T>::type>& e,
                  const VectorView<T>& tau) {
    const int n = ab.cols();
    const int ldab = ab.ld();
    const int stride_ab = ab.stride();
    const int stride_d = d.stride();
    const int stride_e = e.stride();
    const int stride_tau = tau.stride();
    const int batch = ab.batch_size();
    const T* ab_ptr = ab.data_ptr();
    using Real = typename base_type<T>::type;
    Real* d_ptr = d.data_ptr();
    Real* e_ptr = e.data_ptr();
    T* tau_ptr = tau.data_ptr();

    (void)q->submit([&](sycl::handler& h) {
        h.parallel_for<Kd0ExtractKernel<T>>(sycl::range<1>(static_cast<size_t>(batch)),
                                            [=](sycl::id<1> idx) {
            const int b = static_cast<int>(idx[0]);
            const T* AB = ab_ptr + b * stride_ab;
            Real* Db = d_ptr + b * stride_d;
            Real* Eb = e_ptr + b * stride_e;
            T* Taub = tau_ptr + b * stride_tau;

            for (int i = 0; i < n; ++i) {
                Db[i] = real_part(AB[0 + i * ldab]);
                if (i < n - 1) {
                    Eb[i] = Real(0);
                    Taub[i] = T(0);
                }
            }
        });
    });
    return q.get_event();
}

template <typename T>
class Kd1ExtractKernel;

template <Backend B, typename T>
class AbWorkCopyKernel;

template <typename T>
Event kd1_extract_lower(Queue& q,
                        const MatrixView<T, MatrixFormat::Dense>& ab,
                        const VectorView<typename base_type<T>::type>& d,
                        const VectorView<typename base_type<T>::type>& e,
                        const VectorView<T>& tau) {
    const int n = ab.cols();
    const int ldab = ab.ld();
    const int stride_ab = ab.stride();
    const int stride_d = d.stride();
    const int stride_e = e.stride();
    const int stride_tau = tau.stride();
    const int batch = ab.batch_size();
    const T* ab_ptr = ab.data_ptr();
    using Real = typename base_type<T>::type;
    Real* d_ptr = d.data_ptr();
    Real* e_ptr = e.data_ptr();
    T* tau_ptr = tau.data_ptr();

    (void)q->submit([&](sycl::handler& h) {
        h.parallel_for<Kd1ExtractKernel<T>>(sycl::range<1>(static_cast<size_t>(batch)),
                                            [=](sycl::id<1> idx) {
            const int b = static_cast<int>(idx[0]);
            const T* AB = ab_ptr + b * stride_ab;
            Real* Db = d_ptr + b * stride_d;
            Real* Eb = e_ptr + b * stride_e;
            T* Taub = tau_ptr + b * stride_tau;

            for (int i = 0; i < n; ++i) {
                Db[i] = real_part(AB[0 + i * ldab]);
            }

            // Make off-diagonal real via chained phase, as in LAPACK KD=1 special-case.
            T phase = T(1);
            for (int i = 0; i < n - 1; ++i) {
                const T t = AB[1 + i * ldab];
                const Real a = internal::abs(t);
                Eb[i] = a;
                Taub[i] = T(0);
                if constexpr (internal::is_complex<T>::value) {
                    if (a != Real(0)) {
                        const T u = t / T(a);
                        phase *= u;
                    }
                } else {
                    (void)phase;
                }
            }
        });
    });
    return q.get_event();
}

// === Band-only SB2ST path (real symmetric, lower storage) ===

// Complex Hermitian helpers.
template <typename T>
inline typename base_type<T>::type abs_complex(const T& z) {
    using Real = typename base_type<T>::type;
    return sycl::hypot(static_cast<Real>(z.real()), static_cast<Real>(z.imag()));
}

template <typename T, bool IsComplex = internal::is_complex<T>::value>
struct btrd_givens_ops;

// Real/symmetric variant (DSBTRD-style primitives).
template <typename T>
struct btrd_givens_ops<T, false> {
    using Real = typename base_type<T>::type;

    static inline void lartg(const T& f_in, const T& g_in, Real& c, T& s, T& r) {
        auto [ct, st, rt] = internal::lartg(f_in, g_in);
        c = ct;
        s = T(st);
        r = T(rt);
    }

    static inline void rot(int nrot, T* x, int incx, T* y, int incy, Real c, const T& s_in) {
        const Real s = static_cast<Real>(s_in);
        for (int t = 0; t < nrot; ++t) {
            const Real xt = static_cast<Real>(x[t * incx]);
            const Real yt = static_cast<Real>(y[t * incy]);
            x[t * incx] = T(c * xt + s * yt);
            y[t * incy] = T(c * yt - s * xt);
        }
    }

    static inline void lartv(int nrot,
                             T* x,
                             int incx,
                             T* y,
                             int incy,
                             const Real* c,
                             const T* s,
                             int incc) {
        for (int t = 0; t < nrot; ++t) {
            rot(1, &x[t * incx], 1, &y[t * incy], 1, c[t * incc], s[t * incc]);
        }
    }

    static inline void largv(int nrot, T* x, int incx, T* w, int incw, Real* c, int incc) {
        for (int t = 0; t < nrot; ++t) {
            const T f = x[t * incx];
            const T g = w[t * incw];
            Real ct = Real(0);
            T st = T(0);
            T rt = T(0);
            lartg(f, g, ct, st, rt);
            c[t * incc] = ct;
            w[t * incw] = st;
            x[t * incx] = rt;
        }
    }

    static inline void lar2v(int nrot, T* x, T* y, T* z, int incx, const Real* c, const T* s, int incc) {
        for (int t = 0; t < nrot; ++t) {
            const Real xi = static_cast<Real>(x[t * incx]);
            const Real yi = static_cast<Real>(y[t * incx]);
            const Real zi = static_cast<Real>(z[t * incx]);
            const Real ci = c[t * incc];
            const Real si = static_cast<Real>(s[t * incc]);

            const Real t1 = si * zi;
            const Real t2 = ci * zi;
            const Real t3 = t2 - si * xi;
            const Real t4 = t2 + si * yi;
            const Real t5 = ci * xi + t1;
            const Real t6 = ci * yi - t1;

            x[t * incx] = T(ci * t5 + si * t4);
            y[t * incx] = T(ci * t6 - si * t3);
            z[t * incx] = T(ci * t4 - si * t5);
        }
    }

    static inline void prepare_work_for_right(int /*nrot*/, T* /*s*/, int /*stride*/) {}
    static inline void ensure_band_diag_real(T* /*AB*/, int /*ldab*/, int /*n*/) {}

    static inline void finalize_tridiag(const T* AB,
                                        int ldab,
                                        int n,
                                        Real* D,
                                        Real* E,
                                        T* TAU) {
        auto idx = [&](int r1, int c1) -> int { return (r1 - 1) + (c1 - 1) * ldab; };
        for (int j = 0; j < n; ++j) {
            D[j] = static_cast<Real>(AB[idx(1, j + 1)]);
            if (j < n - 1) {
                E[j] = static_cast<Real>(AB[idx(2, j + 1)]);
                TAU[j] = T(0);
            }
        }
    }
};

// Complex/Hermitian variant (ZHBTRD-style primitives).
template <typename T>
struct btrd_givens_ops<T, true> {
    using Real = typename base_type<T>::type;

    static inline void lartg(const T& f, const T& g, Real& c, T& s, T& r) {
        auto [ct, st, rt] = internal::lartg(f, g);
        c = ct;
        s = st;
        r = rt;
    }

    static inline void rot(int nrot, T* x, int incx, T* y, int incy, Real c, const T& s) {
        for (int t = 0; t < nrot; ++t) {
            const T xt = x[t * incx];
            const T yt = y[t * incy];
            x[t * incx] = T(c) * xt + s * yt;
            y[t * incy] = T(c) * yt - conj_if_needed(s) * xt;
        }
    }

    static inline void lartv(int nrot,
                             T* x,
                             int incx,
                             T* y,
                             int incy,
                             const Real* c,
                             const T* s,
                             int incc) {
        for (int t = 0; t < nrot; ++t) {
            rot(1, &x[t * incx], 1, &y[t * incy], 1, c[t * incc], s[t * incc]);
        }
    }

    static inline void largv(int nrot, T* x, int incx, T* w, int incw, Real* c, int incc) {
        for (int t = 0; t < nrot; ++t) {
            const T f = x[t * incx];
            const T g = w[t * incw];
            Real ct = Real(0);
            T st = T(0);
            T rt = T(0);
            lartg(f, g, ct, st, rt);
            c[t * incc] = ct;
            w[t * incw] = st;
            x[t * incx] = rt;
        }
    }

    static inline void lar2v(int nrot, T* x, T* y, T* z, int incx, const Real* c, const T* s, int incc) {
        // Exact port of Netlib LAPACK CLAR2V logic.
        for (int t = 0; t < nrot; ++t) {
            const Real xi = static_cast<Real>(x[t * incx].real());
            const Real yi = static_cast<Real>(y[t * incx].real());
            const T zi = z[t * incx];

            const Real zir = static_cast<Real>(zi.real());
            const Real zii = static_cast<Real>(zi.imag());

            const Real ci = c[t * incc];
            const T si = s[t * incc];
            const Real sir = static_cast<Real>(si.real());
            const Real sii = static_cast<Real>(si.imag());

            const Real t1r = sir * zir - sii * zii;
            const Real t1i = sir * zii + sii * zir;

            const T t2 = T(ci * zir, ci * zii);
            const T t3 = t2 - conj_if_needed(si) * T(xi, Real(0));
            const T t4 = conj_if_needed(t2) + si * T(yi, Real(0));

            const Real t5 = ci * xi + t1r;
            const Real t6 = ci * yi - t1r;

            const Real xnew = ci * t5 + (sir * static_cast<Real>(t4.real()) + sii * static_cast<Real>(t4.imag()));
            const Real ynew = ci * t6 - (sir * static_cast<Real>(t3.real()) - sii * static_cast<Real>(t3.imag()));
            const T znew = T(ci) * t3 + conj_if_needed(si) * T(t6, t1i);

            x[t * incx] = T(xnew, Real(0));
            y[t * incx] = T(ynew, Real(0));
            z[t * incx] = znew;
        }
    }

    static inline void prepare_work_for_right(int nrot, T* s, int stride) {
        // ZHBTRD requirement: conjugate WORK(J1) vector before applying from the right.
        for (int t = 0; t < nrot; ++t) {
            s[t * stride] = conj_if_needed(s[t * stride]);
        }
    }

    static inline void ensure_band_diag_real(T* AB, int ldab, int n) {
        // Ensure diagonal is real (ZHBTRD does this).
        for (int j = 0; j < n; ++j) {
            const int idx = 0 + j * ldab;
            AB[idx] = real_as_T(AB[idx]);
        }
    }

    static inline void finalize_tridiag(T* AB,
                                        int ldab,
                                        int n,
                                        Real* D,
                                        Real* E,
                                        T* TAU) {
        auto idx = [&](int r1, int c1) -> int { return (r1 - 1) + (c1 - 1) * ldab; };

        // Make diagonal real.
        for (int j = 0; j < n; ++j) {
            AB[idx(1, j + 1)] = real_as_T(AB[idx(1, j + 1)]);
        }

        // Make subdiagonal real (phase chaining) and extract D/E.
        for (int j = 0; j < n - 1; ++j) {
            const T t = AB[idx(2, j + 1)];
            const Real a = abs_complex(t);
            E[j] = a;
            T phase = T(1);
            if (a != Real(0)) {
                phase = t / T(a);
            }
            AB[idx(2, j + 1)] = T(a, Real(0));
            if (j < n - 2) {
                AB[idx(2, j + 2)] *= phase;
            }
            TAU[j] = T(0);
        }

        for (int j = 0; j < n; ++j) {
            D[j] = static_cast<Real>(real_part(AB[idx(1, j + 1)]));
        }
    }
};

template <typename T>
class BtrdLowerKernel;

template <typename T>
Event btrd_lower_inplace(Queue& q,
                         T* ab_ptr,
                         int ldab,
                         int stride_ab,
                         int n,
                         int kd,
                         typename base_type<T>::type* c_ptr,
                         int stride_c,
                         T* work_ptr,
                         int stride_work,
                         typename base_type<T>::type* d_ptr,
                         int stride_d,
                         typename base_type<T>::type* e_ptr,
                         int stride_e,
                         T* tau_ptr,
                         int stride_tau,
                         int batch) {
    (void)q->submit([&](sycl::handler& h) {
        constexpr int kWorkGroupSize = 128;
        const size_t groups = static_cast<size_t>(batch);
        const size_t global = groups * static_cast<size_t>(kWorkGroupSize);
        sycl::local_accessor<int, 1> shared_ints(4, h); // {nr, j1, j2, k}

        h.parallel_for<BtrdLowerKernel<T>>(sycl::nd_range<1>(sycl::range<1>(global),
                                                            sycl::range<1>(static_cast<size_t>(kWorkGroupSize))),
                                           [=](sycl::nd_item<1> it) {
            using Real = typename base_type<T>::type;
            using Ops = btrd_givens_ops<T>;

            const int b = static_cast<int>(it.get_group(0));
            const int lid = static_cast<int>(it.get_local_id(0));
            const int lsize = static_cast<int>(it.get_local_range(0));

            T* AB = ab_ptr + b * stride_ab;
            Real* C = c_ptr + b * stride_c;
            T* WORK = work_ptr + b * stride_work;
            Real* D = d_ptr + b * stride_d;
            Real* E = e_ptr + b * stride_e;
            T* TAU = tau_ptr + b * stride_tau;

            for (int i = lid; i < n + kd + 2; i += lsize) {
                C[i] = Real(0);
                WORK[i] = T(0);
            }
            sycl::group_barrier(it.get_group());

            const int kd1 = kd + 1;
            const int kdm1 = kd - 1;
            const int incx = ldab - 1;
            const int inca = kd1 * ldab;
            const int kdn = (n - 1 < kd) ? (n - 1) : kd;

            auto idx = [&](int r1, int c1) -> int {
                return (r1 - 1) + (c1 - 1) * ldab;
            };

            int nr = 0;
            int j1 = kdn + 2;
            int j2 = 1;

            if (lid == 0) {
                shared_ints[0] = nr;
                shared_ints[1] = j1;
                shared_ints[2] = j2;
                shared_ints[3] = 0;
            }
            sycl::group_barrier(it.get_group());

            // Ensure diagonal is real (for Hermitian path). Parallelize across columns.
            if constexpr (internal::is_complex<T>::value) {
                for (int j = lid; j < n; j += lsize) {
                    AB[(0) + j * ldab] = real_as_T(AB[(0) + j * ldab]);
                }
                sycl::group_barrier(it.get_group());
            }

            for (int i = 1; i <= n - 2; ++i) {
                for (int k = kdn + 1; k >= 2; --k) {
                    if (lid == 0) {
                        j1 += kdn;
                        j2 += kdn;
                        shared_ints[0] = nr;
                        shared_ints[1] = j1;
                        shared_ints[2] = j2;
                        shared_ints[3] = k;
                    }
                    sycl::group_barrier(it.get_group());

                    nr = shared_ints[0];
                    j1 = shared_ints[1];
                    j2 = shared_ints[2];
                    // k is only used for scalar work (lid==0) below.

                    if (nr > 0) {
                        // Parallel LARGV over rotations.
                        for (int t = lid; t < nr; t += lsize) {
                            const int j = j1 + t * kd1;
                            const T f = AB[idx(kd1, j - kd1)];
                            const T g = WORK[j];
                            Real ct = Real(0);
                            T st = T(0);
                            T rt = T(0);
                            Ops::lartg(f, g, ct, st, rt);
                            C[j] = ct;
                            WORK[j] = st;
                            AB[idx(kd1, j - kd1)] = rt;
                        }
                        sycl::group_barrier(it.get_group());

                        if (nr > 2 * kd - 1) {
                            // Keep the l-loop sequential, but parallelize across rotations within each l.
                            for (int l = 1; l <= kd - 1; ++l) {
                                for (int t = lid; t < nr; t += lsize) {
                                    const int j = j1 + t * kd1;
                                    Ops::rot(1,
                                             &AB[idx(kd1 - l, j - kd1 + l)],
                                             1,
                                             &AB[idx(kd1 - l + 1, j - kd1 + l)],
                                             1,
                                             C[j],
                                             WORK[j]);
                                }
                                sycl::group_barrier(it.get_group());
                            }
                        } else {
                            if (lid == 0) {
                                const int jend = j1 + (nr - 1) * kd1;
                                for (int jinc = j1; jinc <= jend; jinc += kd1) {
                                    if (kdm1 > 0) {
                                        Ops::rot(kdm1,
                                                 &AB[idx(kd, jinc - kd)],
                                                 incx,
                                                 &AB[idx(kd1, jinc - kd)],
                                                 incx,
                                                 C[jinc],
                                                 WORK[jinc]);
                                    }
                                }
                            }
                            sycl::group_barrier(it.get_group());
                        }
                    }

                    if (lid == 0) {
                        if (k > 2) {
                            if (k <= n - i + 1) {
                                const T f = AB[idx(k - 1, i)];
                                const T g = AB[idx(k, i)];
                                Real ct = Real(0);
                                T st = T(0);
                                T rt = T(0);
                                Ops::lartg(f, g, ct, st, rt);
                                C[i + k - 1] = ct;
                                WORK[i + k - 1] = st;
                                AB[idx(k - 1, i)] = rt;

                                if (k - 3 > 0) {
                                    const Real ct = C[i + k - 1];
                                    const T st = WORK[i + k - 1];
                                    Ops::rot(k - 3,
                                             &AB[idx(k - 2, i + 1)],
                                             ldab - 1,
                                             &AB[idx(k - 1, i + 1)],
                                             ldab - 1,
                                             ct,
                                             st);
                                }
                            }
                            nr += 1;
                            j1 = j1 - kdn - 1;
                        }
                        shared_ints[0] = nr;
                        shared_ints[1] = j1;
                        shared_ints[2] = j2;
                    }
                    sycl::group_barrier(it.get_group());

                    nr = shared_ints[0];
                    j1 = shared_ints[1];
                    j2 = shared_ints[2];

                    if (nr > 0) {
                        // Parallel LAR2V over rotations.
                        for (int t = lid; t < nr; t += lsize) {
                            const int j = j1 + t * kd1;
                            Ops::lar2v(1,
                                       &AB[idx(1, j - 1)],
                                       &AB[idx(1, j)],
                                       &AB[idx(2, j - 1)],
                                       inca,
                                       &C[j],
                                       &WORK[j],
                                       kd1);
                        }
                        sycl::group_barrier(it.get_group());

                        // ZHBTRD requirement: conjugate WORK(J1) before right-application.
                        if constexpr (internal::is_complex<T>::value) {
                            for (int t = lid; t < nr; t += lsize) {
                                const int j = j1 + t * kd1;
                                WORK[j] = conj_if_needed(WORK[j]);
                            }
                            sycl::group_barrier(it.get_group());
                        }

                        if (nr > 2 * kd - 1) {
                            for (int l = 1; l <= kd - 1; ++l) {
                                const int nrt = (j2 + l > n) ? (nr - 1) : nr;
                                if (nrt > 0) {
                                    for (int t = lid; t < nrt; t += lsize) {
                                        const int j = j1 + t * kd1;
                                        Ops::rot(1,
                                                 &AB[idx(l + 2, j - 1)],
                                                 1,
                                                 &AB[idx(l + 1, j)],
                                                 1,
                                                 C[j],
                                                 WORK[j]);
                                    }
                                    sycl::group_barrier(it.get_group());
                                }
                            }
                        } else {
                            if (lid == 0) {
                                const int j1end = j1 + kd1 * (nr - 2);
                                if (j1end >= j1) {
                                    for (int j1inc = j1; j1inc <= j1end; j1inc += kd1) {
                                        if (kdm1 > 0) {
                                            Ops::rot(kdm1,
                                                     &AB[idx(3, j1inc - 1)],
                                                     1,
                                                     &AB[idx(2, j1inc)],
                                                     1,
                                                     C[j1inc],
                                                     WORK[j1inc]);
                                        }
                                    }
                                }
                                const int lend = (n - j2 < kdm1) ? (n - j2) : kdm1;
                                const int last = j1end + kd1;
                                if (lend > 0) {
                                    Ops::rot(lend,
                                             &AB[idx(3, last - 1)],
                                             1,
                                             &AB[idx(2, last)],
                                             1,
                                             C[last],
                                             WORK[last]);
                                }
                            }
                            sycl::group_barrier(it.get_group());
                        }
                    }

                    if (lid == 0) {
                        if (j2 + kdn > n) {
                            nr -= 1;
                            j2 = j2 - kdn - 1;
                        }
                        shared_ints[0] = nr;
                        shared_ints[1] = j1;
                        shared_ints[2] = j2;
                    }
                    sycl::group_barrier(it.get_group());

                    nr = shared_ints[0];
                    j1 = shared_ints[1];
                    j2 = shared_ints[2];

                    if (nr > 0) {
                        for (int t = lid; t < nr; t += lsize) {
                            const int j = j1 + t * kd1;
                            WORK[j + kd] = WORK[j] * AB[idx(kd1, j)];
                            AB[idx(kd1, j)] = T(C[j]) * AB[idx(kd1, j)];
                        }
                        sycl::group_barrier(it.get_group());
                    }
                }
            }

            if (lid == 0) {
                Ops::finalize_tridiag(AB, ldab, n, D, E, TAU);
            }
        });
    });
    return q.get_event();
}

} // namespace

template <Backend B, typename T>
size_t sytrd_sb2st_buffer_size(Queue& ctx,
                               const MatrixView<T, MatrixFormat::Dense>& ab_in,
                               const VectorView<typename base_type<T>::type>& d_out,
                               const VectorView<typename base_type<T>::type>& e_out,
                               const VectorView<T>& tau_out,
                               Uplo uplo,
                               int32_t kd,
                               int32_t block_size) {
    validate_sytrd_sb2st_dims(ab_in, d_out, e_out, tau_out, uplo, kd);

    const int n = ab_in.cols();
    const int batch = ab_in.batch_size();

    (void)block_size;

    size_t size = 0;

    const int kd_i = std::max<int>(0, kd);
    if (n <= 0) return 0;

    // Unified band-only (SBTRD/HBTRD-style) workspace:
    // - AB work copy: (kd+1) x n per batch
    // - C array: length (n+kd+2) per batch (real)
    // - WORK array: length (n+kd+2) per batch (T; real for real types, complex for complex types)
    const int ldab = kd_i + 1;
    const size_t ab_elems = static_cast<size_t>(ldab) * static_cast<size_t>(n) * static_cast<size_t>(batch);
    size += BumpAllocator::allocation_size<T>(ctx, ab_elems);

    const size_t rot_elems = static_cast<size_t>(n + kd_i + 2) * static_cast<size_t>(batch);
    using Real = typename base_type<T>::type;
    size += BumpAllocator::allocation_size<Real>(ctx, rot_elems); // C
    size += BumpAllocator::allocation_size<T>(ctx, rot_elems);    // WORK

    return size;
}

template <Backend B, typename T>
Event sytrd_sb2st(Queue& ctx,
                  const MatrixView<T, MatrixFormat::Dense>& ab_in,
                  const VectorView<typename base_type<T>::type>& d_out,
                  const VectorView<typename base_type<T>::type>& e_out,
                  const VectorView<T>& tau_out,
                  Uplo uplo,
                  int32_t kd,
                  const Span<std::byte>& ws,
                  int32_t block_size) {
    validate_sytrd_sb2st_dims(ab_in, d_out, e_out, tau_out, uplo, kd);

    if (!ctx.in_order()) {
        throw std::runtime_error("sytrd_sb2st: requires an in-order Queue");
    }

    if (uplo != Uplo::Lower) {
        throw std::runtime_error("sytrd_sb2st: only Uplo::Lower is implemented");
    }

    const int n = ab_in.cols();
    const int batch = ab_in.batch_size();
    const int kd_i = std::max<int>(0, kd);

    if (n <= 0) return ctx.get_event();

    // Fast paths.
    if (kd_i == 0) {
        (void)kd0_extract<T>(ctx, ab_in, d_out, e_out, tau_out);
        return ctx.get_event();
    }
    if (kd_i == 1) {
        (void)kd1_extract_lower<T>(ctx, ab_in, d_out, e_out, tau_out);
        return ctx.get_event();
    }

    BumpAllocator pool(ws);

    // Unified band-only SB2ST/HBTRD: copy AB into a mutable workspace and run bulge chasing.
    const int ldab = kd_i + 1;
    const size_t ab_elems = static_cast<size_t>(ldab) * static_cast<size_t>(n) * static_cast<size_t>(batch);
    auto ab_work = pool.allocate<T>(ctx, ab_elems);
    T* ab_ptr = ab_work.data();

    {
        const int stride_in = ab_in.stride();
        const int stride_out = ldab * n;
        const int ldab_in = ab_in.ld();
        const T* src = ab_in.data_ptr();
        (void)ctx->submit([&](sycl::handler& h) {
            h.parallel_for<AbWorkCopyKernel<B, T>>(sycl::range<1>(static_cast<size_t>(batch)),
                                                   [=](sycl::id<1> idx_b) {
                const int b = static_cast<int>(idx_b[0]);
                const T* AB = src + b * stride_in;
                T* W = ab_ptr + b * stride_out;
                for (int j = 0; j < n; ++j) {
                    for (int r = 0; r < ldab; ++r) {
                        W[r + j * ldab] = AB[r + j * ldab_in];
                    }
                }
            });
        });
    }

    using Real = typename base_type<T>::type;
    const size_t rot_elems = static_cast<size_t>(n + kd_i + 2) * static_cast<size_t>(batch);
    auto c_buf = pool.allocate<Real>(ctx, rot_elems);
    auto work_buf = pool.allocate<T>(ctx, rot_elems);

    (void)btrd_lower_inplace<T>(
        ctx,
        ab_ptr,
        ldab,
        ldab * n,
        n,
        kd_i,
        c_buf.data(),
        (n + kd_i + 2),
        work_buf.data(),
        (n + kd_i + 2),
        d_out.data_ptr(),
        d_out.stride(),
        e_out.data_ptr(),
        e_out.stride(),
        tau_out.data_ptr(),
        tau_out.stride(),
        batch);

    return ctx.get_event();
}

#define SYTRD_SB2ST_INSTANTIATE(back, fp) \
    template Event sytrd_sb2st<back, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        const VectorView<typename base_type<fp>::type>&, \
        const VectorView<typename base_type<fp>::type>&, \
        const VectorView<fp>&, \
        Uplo, \
        int32_t, \
        const Span<std::byte>&, \
        int32_t); \
    template size_t sytrd_sb2st_buffer_size<back, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        const VectorView<typename base_type<fp>::type>&, \
        const VectorView<typename base_type<fp>::type>&, \
        const VectorView<fp>&, \
        Uplo, \
        int32_t, \
        int32_t);

#if BATCHLAS_HAS_CUDA_BACKEND
SYTRD_SB2ST_INSTANTIATE(Backend::CUDA, float)
SYTRD_SB2ST_INSTANTIATE(Backend::CUDA, double)
SYTRD_SB2ST_INSTANTIATE(Backend::CUDA, std::complex<float>)
SYTRD_SB2ST_INSTANTIATE(Backend::CUDA, std::complex<double>)
#endif

#if BATCHLAS_HAS_ROCM_BACKEND
SYTRD_SB2ST_INSTANTIATE(Backend::ROCM, float)
SYTRD_SB2ST_INSTANTIATE(Backend::ROCM, double)
SYTRD_SB2ST_INSTANTIATE(Backend::ROCM, std::complex<float>)
SYTRD_SB2ST_INSTANTIATE(Backend::ROCM, std::complex<double>)
#endif

#if BATCHLAS_HAS_HOST_BACKEND
SYTRD_SB2ST_INSTANTIATE(Backend::NETLIB, float)
SYTRD_SB2ST_INSTANTIATE(Backend::NETLIB, double)
SYTRD_SB2ST_INSTANTIATE(Backend::NETLIB, std::complex<float>)
SYTRD_SB2ST_INSTANTIATE(Backend::NETLIB, std::complex<double>)
#endif

#undef SYTRD_SB2ST_INSTANTIATE

} // namespace batchlas
