#pragma once

#include <batchlas/blas/matrix.hh>

#include "../../../math-helpers.hh"
#include "../../../queue.hh"
#include "sytrd_sb2st_cta_instantiations.hh"

#include <sycl/sycl.hpp>

#include <complex>
#include <cstdint>

namespace batchlas {

namespace internal {

// Band-to-tridiagonal (xSBTRD/xHBTRD) Givens primitives shared by sytrd_sb2st.cc and sytrd_sb2st_cta.cc.
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
inline void lartv_impl(int nrot, StridedView<T> x, StridedView<T> y, ConstStridedView<typename Ops::Real> c,
                       ConstStridedView<T> s) {
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

template <typename T>
inline T real_as_T(const T& x) {
    if constexpr (internal::is_complex<T>::value) {
        return T(x.real(), typename T::value_type(0));
    } else {
        return x;
    }
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
        y = T(c) * yt - batchlas::device::detail::conj(s) * xt;
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
        const T t3 = t2 - batchlas::device::detail::conj(si) * T(xi, Real(0));
        const T t4 = batchlas::device::detail::conj(t2) + si * T(yi, Real(0));

        const Real t5 = ci * xi + t1r;
        const Real t6 = ci * yi - t1r;

        const Real xnew = ci * t5 + (sir * static_cast<Real>(t4.real()) + sii * static_cast<Real>(t4.imag()));
        const Real ynew = ci * t6 - (sir * static_cast<Real>(t3.real()) - sii * static_cast<Real>(t3.imag()));
        const T znew = T(ci) * t3 + batchlas::device::detail::conj(si) * T(t6, t1i);

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
            s[t * stride] = batchlas::device::detail::conj(s[t * stride]);
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
            const Real a = internal::abs(t);
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
Event btrd_lower_inplace_subgroup_dispatch(Queue& q,
                                          KernelMatrixView<T, MatrixFormat::Dense> ab,
                                          int n,
                                          int kd,
                                          VectorView<typename base_type<T>::type> c,
                                          VectorView<T> work,
                                          VectorView<typename base_type<T>::type> d,
                                          VectorView<typename base_type<T>::type> e,
                                          VectorView<T> tau);

// Explicit instantiations live in sytrd_sb2st_cta.cc.
#define BTRD_LOWER_INPLACE_SUBGROUP_DISPATCH_EXTERN(fp, real_fp) \
extern template Event btrd_lower_inplace_subgroup_dispatch<BATCHLAS_UNPAREN fp>(Queue&, \
                                                                                 KernelMatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>, \
                                                                                 int, \
                                                                                 int, \
                                                                                 VectorView<BATCHLAS_UNPAREN real_fp>, \
                                                                                 VectorView<BATCHLAS_UNPAREN fp>, \
                                                                                 VectorView<BATCHLAS_UNPAREN real_fp>, \
                                                                                 VectorView<BATCHLAS_UNPAREN real_fp>, \
                                                                                 VectorView<BATCHLAS_UNPAREN fp>);

BATCHLAS_FOR_EACH_SYTRD_SB2ST_CTA_DISPATCH_TYPE(BTRD_LOWER_INPLACE_SUBGROUP_DISPATCH_EXTERN)

#undef BTRD_LOWER_INPLACE_SUBGROUP_DISPATCH_EXTERN

} // namespace internal

} // namespace batchlas
