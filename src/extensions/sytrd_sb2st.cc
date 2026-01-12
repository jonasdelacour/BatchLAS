#include <blas/extensions.hh>
#include <blas/matrix.hh>
#include <util/mempool.hh>

#include <sycl/sycl.hpp>

#include <batchlas/backend_config.h>

#include "../math-helpers.hh"
#include "../queue.hh"

#include "sytrd_sb2st_cta.hh"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace batchlas {

namespace {

template <typename T>
struct StridedView {
    T* ptr = nullptr;
    int inc = 1;
    inline T& operator[](int i) const { return ptr[i * inc]; }
};

template <typename T>
struct ConstStridedView {
    const T* ptr = nullptr;
    int inc = 1;
    inline const T& operator[](int i) const { return ptr[i * inc]; }
};

template <typename T>
inline StridedView<T> make_strided(T* ptr, int inc) {
    return StridedView<T>{ptr, inc};
}

template <typename T>
inline ConstStridedView<T> make_strided(const T* ptr, int inc) {
    return ConstStridedView<T>{ptr, inc};
}

template <typename T>
inline ConstStridedView<T> make_const_strided(T* ptr, int inc) {
    return ConstStridedView<T>{ptr, inc};
}

template <typename Ops, typename T>
inline void lartv_impl(int nrot, StridedView<T> x, StridedView<T> y, ConstStridedView<typename Ops::Real> c, ConstStridedView<T> s) {
    for (int t = 0; t < nrot; ++t) {
        Ops::rot_pair(x[t], y[t], c[t], s[t]);
    }
}

template <typename Ops, typename T>
inline void largv_impl(int nrot, StridedView<T> x, StridedView<T> w, StridedView<typename Ops::Real> c) {
    for (int t = 0; t < nrot; ++t) {
        const T f = x[t];
        const T g = w[t];
        typename Ops::Real ct = typename Ops::Real(0);
        T st = T(0);
        T rt = T(0);
        Ops::lartg(f, g, ct, st, rt);
        c[t] = ct;
        w[t] = st;
        x[t] = rt;
    }
}

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
    const int batch = ab.batch_size();
    using Real = typename base_type<T>::type;

    auto ABv = ab.kernel_view();
    auto dv = d;
    auto ev = e;
    auto tauv = tau;

    (void)q->submit([&](sycl::handler& h) {
        h.parallel_for<Kd0ExtractKernel<T>>(sycl::range<1>(static_cast<size_t>(batch)),
                                            [=](sycl::id<1> idx) {
            const int b = static_cast<int>(idx[0]);
            auto AB = ABv.batch_item(b);
            auto Db = dv.batch_item(b);
            auto Eb = ev.batch_item(b);
            auto Taub = tauv.batch_item(b);

            for (int i = 0; i < n; ++i) {
                Db[i] = real_part(AB(0, i));
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
    const int batch = ab.batch_size();
    using Real = typename base_type<T>::type;

    auto ABv = ab.kernel_view();
    auto dv = d;
    auto ev = e;
    auto tauv = tau;

    (void)q->submit([&](sycl::handler& h) {
        h.parallel_for<Kd1ExtractKernel<T>>(sycl::range<1>(static_cast<size_t>(batch)),
                                            [=](sycl::id<1> idx) {
            const int b = static_cast<int>(idx[0]);
            auto AB = ABv.batch_item(b);
            auto Db = dv.batch_item(b);
            auto Eb = ev.batch_item(b);
            auto Taub = tauv.batch_item(b);

            for (int i = 0; i < n; ++i) {
                Db[i] = real_part(AB(0, i));
            }

            // Make off-diagonal real via chained phase, as in LAPACK KD=1 special-case.
            T phase = T(1);
            for (int i = 0; i < n - 1; ++i) {
                const T t = AB(1, i);
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

    static inline void rot_pair(T& x, T& y, Real c, const T& s_in) {
        const Real s = static_cast<Real>(s_in);
        const Real xt = static_cast<Real>(x);
        const Real yt = static_cast<Real>(y);
        x = T(c * xt + s * yt);
        y = T(c * yt - s * xt);
    }

    static inline void rot(int nrot, StridedView<T> x, StridedView<T> y, Real c, const T& s_in) {
        for (int t = 0; t < nrot; ++t) {
            rot_pair(x[t], y[t], c, s_in);
        }
    }

    static inline void lartv(int nrot,
                             StridedView<T> x,
                             StridedView<T> y,
                             ConstStridedView<Real> c,
                             ConstStridedView<T> s) {
        lartv_impl<btrd_givens_ops<T, false>, T>(nrot, x, y, c, s);
    }

    static inline void largv(int nrot, StridedView<T> x, StridedView<T> w, StridedView<Real> c) {
        largv_impl<btrd_givens_ops<T, false>, T>(nrot, x, w, c);
    }

    static inline void lar2v_pair(T& x, T& y, T& z, Real ci, const T& si_in) {
        const Real xi = static_cast<Real>(x);
        const Real yi = static_cast<Real>(y);
        const Real zi = static_cast<Real>(z);
        const Real si = static_cast<Real>(si_in);

        const Real t1 = si * zi;
        const Real t2 = ci * zi;
        const Real t3 = t2 - si * xi;
        const Real t4 = t2 + si * yi;
        const Real t5 = ci * xi + t1;
        const Real t6 = ci * yi - t1;

        x = T(ci * t5 + si * t4);
        y = T(ci * t6 - si * t3);
        z = T(ci * t4 - si * t5);
    }

    static inline void lar2v(int nrot,
                             StridedView<T> x,
                             StridedView<T> y,
                             StridedView<T> z,
                             ConstStridedView<Real> c,
                             ConstStridedView<T> s) {
        for (int t = 0; t < nrot; ++t) {
            lar2v_pair(x[t], y[t], z[t], c[t], s[t]);
        }
    }

    static inline void prepare_work_for_right(int /*nrot*/, T* /*s*/, int /*stride*/) {}
    static inline void ensure_band_diag_real(T* /*AB*/, int /*ldab*/, int /*n*/) {}

    static inline void finalize_tridiag(const KernelMatrixView<T, MatrixFormat::Dense>& AB,
                                        int n,
                                        VectorView<Real> D,
                                        VectorView<Real> E,
                                        VectorView<T> TAU) {
        auto diag = AB(int32_t(0), Slice{});
        auto sub = AB(int32_t(1), Slice{});
        for (int j = 0; j < n; ++j) {
            D[j] = static_cast<Real>(diag[j]);
            if (j < n - 1) {
                E[j] = static_cast<Real>(sub[j]);
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

    static inline void rot_pair(T& x, T& y, Real c, const T& s) {
        const T xt = x;
        const T yt = y;
        x = T(c) * xt + s * yt;
        y = T(c) * yt - conj_if_needed(s) * xt;
    }

    static inline void rot(int nrot, StridedView<T> x, StridedView<T> y, Real c, const T& s) {
        for (int t = 0; t < nrot; ++t) {
            rot_pair(x[t], y[t], c, s);
        }
    }

    static inline void lartv(int nrot,
                             StridedView<T> x,
                             StridedView<T> y,
                             ConstStridedView<Real> c,
                             ConstStridedView<T> s) {
        lartv_impl<btrd_givens_ops<T, true>, T>(nrot, x, y, c, s);
    }

    static inline void largv(int nrot, StridedView<T> x, StridedView<T> w, StridedView<Real> c) {
        largv_impl<btrd_givens_ops<T, true>, T>(nrot, x, w, c);
    }

    static inline void lar2v_pair(T& x, T& y, T& z, Real ci, const T& si) {
        // Exact port of Netlib LAPACK CLAR2V logic.
        const Real xi = static_cast<Real>(x.real());
        const Real yi = static_cast<Real>(y.real());
        const T zi = z;

        const Real zir = static_cast<Real>(zi.real());
        const Real zii = static_cast<Real>(zi.imag());

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

        x = T(xnew, Real(0));
        y = T(ynew, Real(0));
        z = znew;
    }

    static inline void lar2v(int nrot,
                             StridedView<T> x,
                             StridedView<T> y,
                             StridedView<T> z,
                             ConstStridedView<Real> c,
                             ConstStridedView<T> s) {
        for (int t = 0; t < nrot; ++t) {
            lar2v_pair(x[t], y[t], z[t], c[t], s[t]);
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

    static inline void finalize_tridiag(KernelMatrixView<T, MatrixFormat::Dense> AB,
                                        int n,
                                        VectorView<Real> D,
                                        VectorView<Real> E,
                                        VectorView<T> TAU) {
        auto diag = AB(int32_t(0), Slice{});
        auto sub = AB(int32_t(1), Slice{});

        // Make diagonal real.
        for (int j = 0; j < n; ++j) {
            diag[j] = real_as_T(diag[j]);
        }

        // Make subdiagonal real (phase chaining) and extract D/E.
        for (int j = 0; j < n - 1; ++j) {
            const T t = sub[j];
            const Real a = abs_complex(t);
            E[j] = a;
            T phase = T(1);
            if (a != Real(0)) {
                phase = t / T(a);
            }
            sub[j] = T(a, Real(0));
            if (j < n - 2) {
                sub[j + 1] *= phase;
            }
            TAU[j] = T(0);
        }

        for (int j = 0; j < n; ++j) {
            D[j] = static_cast<Real>(real_part(diag[j]));
        }
    }
};

template <typename T>
class BtrdLowerKernel;

template <typename T>
Event btrd_lower_inplace(Queue& q,
                         KernelMatrixView<T, MatrixFormat::Dense> ab,
                         int n,
                         int kd,
                         VectorView<typename base_type<T>::type> c,
                         VectorView<T> work,
                         VectorView<typename base_type<T>::type> d,
                         VectorView<typename base_type<T>::type> e,
                         VectorView<T> tau) {
    (void)q->submit([&](sycl::handler& h) {
        constexpr int kWorkGroupSize = 128;
        const int batch = ab.batch_size();
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

            auto ABv = ab.batch_item(b);
            auto Cv = c.batch_item(b);
            auto WORKv = work.batch_item(b);
            auto Dv = d.batch_item(b);
            auto Ev = e.batch_item(b);
            auto TAUv = tau.batch_item(b);

            const int ldab = ABv.ld();
            T* AB = ABv.data();
            auto& C = Cv;
            auto& WORK = WORKv;
            auto& D = Dv;
            auto& E = Ev;
            auto& TAU = TAUv;

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

            auto idx = [&](int r, int c) -> int { return r + c * ldab; };

            int nr = 0;
            int j1 = kdn + 1;
            int j2 = 0;

            if (lid == 0) {
                shared_ints[0] = nr;
                shared_ints[1] = j1;
                shared_ints[2] = j2;
                shared_ints[3] = 0;
            }
            sycl::group_barrier(it.get_group());

            // Ensure diagonal is real (for Hermitian path). Parallelize across columns.
            if constexpr (internal::is_complex<T>::value) {
                auto diag = ABv(int32_t(0), Slice{});
                for (int j = lid; j < n; j += lsize) {
                    diag[j] = real_as_T(diag[j]);
                }
                sycl::group_barrier(it.get_group());
            }

            for (int i = 0; i < n - 2; ++i) {
                for (int k = kdn; k >= 1; --k) {
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
                            const T f = ABv(kd, j - kd1);
                            const T g = WORK[j];
                            Real ct = Real(0);
                            T st = T(0);
                            T rt = T(0);
                            Ops::lartg(f, g, ct, st, rt);
                            C[j] = ct;
                            WORK[j] = st;
                            ABv(kd, j - kd1) = rt;
                        }
                        sycl::group_barrier(it.get_group());

                        if (nr > 2 * kd - 1) {
                            // Keep the l-loop sequential, but parallelize across rotations within each l.
                            for (int l = 0; l < kd - 1; ++l) {
                                for (int t = lid; t < nr; t += lsize) {
                                    const int j = j1 + t * kd1;
                                    Ops::rot_pair(ABv(kd - 1 - l, j - kd1 + l + 1),
                                                  ABv(kd - l, j - kd1 + l + 1),
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
                                                 make_strided(AB + idx(kd - 1, jinc - kd), incx),
                                                 make_strided(AB + idx(kd, jinc - kd), incx),
                                                 C[jinc],
                                                 WORK[jinc]);
                                    }
                                }
                            }
                            sycl::group_barrier(it.get_group());
                        }
                    }

                    if (lid == 0) {
                        if (k > 1) {
                            if (k <= n - i - 1) {
                                const T f = ABv(k - 1, i);
                                const T g = ABv(k, i);
                                Real ct = Real(0);
                                T st = T(0);
                                T rt = T(0);
                                Ops::lartg(f, g, ct, st, rt);
                                C[i + k] = ct;
                                WORK[i + k] = st;
                                ABv(k - 1, i) = rt;

                                if (k > 2) {
                                    const Real ct = C[i + k];
                                    const T st = WORK[i + k];
                                    Ops::rot(k - 2,
                                             make_strided(AB + idx(k - 2, i + 1), ldab - 1),
                                             make_strided(AB + idx(k - 1, i + 1), ldab - 1),
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
                                       make_strided(AB + idx(0, j - 1), inca),
                                       make_strided(AB + idx(0, j), inca),
                                       make_strided(AB + idx(1, j - 1), inca),
                                       make_const_strided(C.data_ptr() + j, kd1),
                                       make_const_strided(WORK.data_ptr() + j, kd1));
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
                            for (int l = 0; l < kd - 1; ++l) {
                                const int nrt = (j2 + l + 2 > n) ? (nr - 1) : nr;
                                if (nrt > 0) {
                                    for (int t = lid; t < nrt; t += lsize) {
                                        const int j = j1 + t * kd1;
                                        Ops::rot_pair(ABv(l + 2, j - 1),
                                                      ABv(l + 1, j),
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
                                                     make_strided(AB + idx(2, j1inc - 1), 1),
                                                     make_strided(AB + idx(1, j1inc), 1),
                                                     C[j1inc],
                                                     WORK[j1inc]);
                                        }
                                    }
                                }
                                const int lend = (n - j2 - 1 < kdm1) ? (n - j2 - 1) : kdm1;
                                const int last = j1end + kd1;
                                if (lend > 0) {
                                    Ops::rot(lend,
                                             make_strided(AB + idx(2, last - 1), 1),
                                             make_strided(AB + idx(1, last), 1),
                                             C[last],
                                             WORK[last]);
                                }
                            }
                            sycl::group_barrier(it.get_group());
                        }
                    }

                    if (lid == 0) {
                        if (j2 + kdn >= n) {
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
                            WORK[j + kd] = WORK[j] * ABv(kd, j);
                            ABv(kd, j) = T(C[j]) * ABv(kd, j);
                        }
                        sycl::group_barrier(it.get_group());
                    }
                }
            }

            if (lid == 0) {
                Ops::finalize_tridiag(ABv, n, D, E, TAU);
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
        const int stride_out = ldab * n;
        auto ABsrc = ab_in.kernel_view();
        KernelMatrixView<T, MatrixFormat::Dense> ABdst(ab_ptr, ldab, n, ldab, stride_out, batch);
        (void)ctx->submit([&](sycl::handler& h) {
            h.parallel_for<AbWorkCopyKernel<B, T>>(sycl::range<1>(static_cast<size_t>(batch)),
                                                   [=](sycl::id<1> idx_b) {
                const int b = static_cast<int>(idx_b[0]);
                auto AB = ABsrc.batch_item(b);
                auto W = ABdst.batch_item(b);
                for (int j = 0; j < n; ++j) {
                    for (int r = 0; r < ldab; ++r) {
                        W(r, j) = AB(r, j);
                    }
                }
            });
        });
    }

    using Real = typename base_type<T>::type;
    const size_t rot_elems = static_cast<size_t>(n + kd_i + 2) * static_cast<size_t>(batch);
    auto c_buf = pool.allocate<Real>(ctx, rot_elems);
    auto work_buf = pool.allocate<T>(ctx, rot_elems);

    KernelMatrixView<T, MatrixFormat::Dense> ABwork(ab_ptr, ldab, n, ldab, ldab * n, batch);
    VectorView<Real> Cv(c_buf.data(), n + kd_i + 2, batch, 1, (n + kd_i + 2));
    VectorView<T> WORKv(work_buf.data(), n + kd_i + 2, batch, 1, (n + kd_i + 2));

    enum class Sb2stSubgroupMode {
        Auto,
        ForceOn,
        ForceOff,
    };

    auto parse_mode = []() -> Sb2stSubgroupMode {
        const char* p = std::getenv("BATCHLAS_SB2ST_SUBGROUP");
        if (!p || !*p) return Sb2stSubgroupMode::Auto;
        std::string v(p);
        for (char& ch : v) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        if (v == "1" || v == "true" || v == "on") return Sb2stSubgroupMode::ForceOn;
        if (v == "0" || v == "false" || v == "off") return Sb2stSubgroupMode::ForceOff;
        if (v == "auto") return Sb2stSubgroupMode::Auto;
        // Unknown value: keep behavior unchanged.
        return Sb2stSubgroupMode::Auto;
    };

    const Sb2stSubgroupMode mode = parse_mode();

    bool device_has_sg32 = false;
    {
        const auto dev = ctx->get_device();
        const auto sg_sizes = dev.get_info<sycl::info::device::sub_group_sizes>();
        for (auto sgs : sg_sizes) {
            if (static_cast<int32_t>(sgs) == 32) {
                device_has_sg32 = true;
                break;
            }
        }
    }

    bool can_use_subgroup = false;
    if (mode != Sb2stSubgroupMode::ForceOff) {
        can_use_subgroup = (kd_i <= 32) && device_has_sg32;
    }

    if (mode == Sb2stSubgroupMode::ForceOn && !can_use_subgroup) {
        throw std::runtime_error(
            "sytrd_sb2st: BATCHLAS_SB2ST_SUBGROUP=1 requires kd<=32 and a device supporting sub_group_size=32");
    }

    if (can_use_subgroup) {
        (void)internal::btrd_lower_inplace_subgroup_dispatch<T>(ctx, ABwork, n, kd_i, Cv, WORKv, d_out, e_out, tau_out);
    } else {
        (void)btrd_lower_inplace<T>(ctx, ABwork, n, kd_i, Cv, WORKv, d_out, e_out, tau_out);
    }

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
