#include <blas/extensions.hh>
#include <blas/matrix.hh>
#include <util/mempool.hh>

#include <sycl/sycl.hpp>

#if defined(BATCHLAS_SB2ST_DEBUG_PRINTF)
#include <sycl/ext/oneapi/experimental/builtins.hpp>
#endif

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

#if defined(BATCHLAS_SB2ST_DEBUG_PRINTF)
template <typename T>
inline void dbg_printf_scalar(const char* prefix, int r, int c, const T& x) {
    (void)prefix;
    (void)r;
    (void)c;
    (void)x;
    if constexpr (internal::is_complex<T>::value) {
        sycl::ext::oneapi::experimental::printf("%s(%d,%d) = (%0.6e, %0.6e)\n",
                                                prefix,
                                                r,
                                                c,
                                                static_cast<double>(x.real()),
                                                static_cast<double>(x.imag()));
    } else {
        sycl::ext::oneapi::experimental::printf("%s(%d,%d) = %0.6e\n",
                                                prefix,
                                                r,
                                                c,
                                                static_cast<double>(x));
    }
}

template <typename T>
inline void dbg_printf_scalar1(const char* prefix, int i, const T& x) {
    (void)prefix;
    (void)i;
    (void)x;
    if constexpr (internal::is_complex<T>::value) {
        sycl::ext::oneapi::experimental::printf("%s[%d] = (%0.6e, %0.6e)\n",
                                                prefix,
                                                i,
                                                static_cast<double>(x.real()),
                                                static_cast<double>(x.imag()));
    } else {
        sycl::ext::oneapi::experimental::printf("%s[%d] = %0.6e\n", prefix, i, static_cast<double>(x));
    }
}
#endif

template <typename U>
inline U conj_if_needed(const U& x) {
    if constexpr (internal::is_complex<U>::value) {
        return U(x.real(), -x.imag());
    } else {
        return x;
    }
}

template <typename T>
inline typename base_type<T>::type abs2_if_complex(const T& x) {
    using Real = typename base_type<T>::type;
    if constexpr (internal::is_complex<T>::value) {
        const Real re = x.real();
        const Real im = x.imag();
        return re * re + im * im;
    } else {
        return x * x;
    }
}

template <typename Real>
inline Real sign_nonzero_real(Real x) {
    return (sycl::signbit(x) ? Real(-1) : Real(1));
}

template <typename T>
inline T sign_nonzero(const T& x) {
    using Real = typename base_type<T>::type;
    if constexpr (internal::is_complex<T>::value) {
        const Real a = sycl::hypot(x.real(), x.imag());
        if (a == Real(0)) return T(1);
        return x / a;
    } else {
        return T(sign_nonzero_real(static_cast<Real>(x)));
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

// Sequential (scalar) CLARFG-like Householder generator.
// Overwrites alpha with beta and x with v (implicit v0=1). Returns tau.
template <typename T>
inline T larfg_seq(int m, T& alpha, T* x, int incx) {
    using Real = typename base_type<T>::type;
    if (m <= 1) return T(0);

    Real sumsq = Real(0);
    for (int i = 0; i < m - 1; ++i) {
        sumsq += abs2_if_complex(x[i * incx]);
    }
    const Real xnorm = sycl::sqrt(sumsq);

    if constexpr (internal::is_complex<T>::value) {
        if (xnorm == Real(0) && alpha.imag() == Real(0)) {
            return T(0);
        }

        const Real alpha_abs = sycl::hypot(alpha.real(), alpha.imag());
        const Real beta_abs = sycl::hypot(alpha_abs, xnorm);
        const T alpha_sign = (alpha_abs == Real(0)) ? T(1) : (alpha / alpha_abs);
        const T beta = -alpha_sign * T(beta_abs);
        const T tau = (beta - alpha) / beta;
        const T scale = T(1) / (alpha - beta);
        alpha = beta;
        for (int i = 0; i < m - 1; ++i) {
            x[i * incx] *= scale;
        }
        return tau;
    } else {
        if (xnorm == Real(0)) {
            return T(0);
        }
        const T beta = -sign_nonzero(alpha) * T(sycl::hypot(static_cast<Real>(alpha), xnorm));
        const T tau = (beta - alpha) / beta;
        const T scale = T(1) / (alpha - beta);
        alpha = beta;
        for (int i = 0; i < m - 1; ++i) {
            x[i * incx] *= scale;
        }
        return tau;
    }
}

template <typename T>
inline void larfx_left_seq(int m, int n, const T* v, T tau, T* C, int ldc, T* /*work*/) {
    if (tau == T(0) || m <= 0 || n <= 0) return;
    for (int j = 0; j < n; ++j) {
        T dot = T(0);
        for (int i = 0; i < m; ++i) {
            dot += conj_if_needed(v[i]) * C[i + j * ldc];
        }
        const T gamma = tau * dot;
        for (int i = 0; i < m; ++i) {
            C[i + j * ldc] -= v[i] * gamma;
        }
    }
}

template <typename T>
inline void larfx_right_seq(int m, int n, const T* v, T tau, T* C, int ldc, T* /*work*/) {
    if (tau == T(0) || m <= 0 || n <= 0) return;
    for (int i = 0; i < m; ++i) {
        T dot = T(0);
        for (int j = 0; j < n; ++j) {
            dot += C[i + j * ldc] * v[j];
        }
        const T gamma = tau * dot;
        for (int j = 0; j < n; ++j) {
            C[i + j * ldc] -= gamma * conj_if_needed(v[j]);
        }
    }
}

// CLARFY-style symmetric/Hermitian update: C := H * C * H^H with H = I - tau v v^H.
// Only lower triangle of C is assumed referenced/stored.
template <typename T>
inline void larfy_lower_seq(int n, const T* v, T tau, T* C, int ldc, T* work) {
    if (tau == T(0) || n <= 0) return;

    // w = C * v (Hermitian, lower stored)
    for (int i = 0; i < n; ++i) {
        T acc = T(0);
        // j <= i: use C(i,j)
        for (int j = 0; j <= i; ++j) {
            acc += C[i + j * ldc] * v[j];
        }
        // j > i: use conj(C(j,i))
        for (int j = i + 1; j < n; ++j) {
            acc += conj_if_needed(C[j + i * ldc]) * v[j];
        }
        work[i] = acc;
    }

    // alpha = -0.5 * tau * (v^H * w)
    T vHw = T(0);
    for (int i = 0; i < n; ++i) {
        vHw += conj_if_needed(v[i]) * work[i];
    }
    const T alpha = T(-0.5) * tau * vHw;

    // w = tau*w + alpha*v
    for (int i = 0; i < n; ++i) {
        work[i] = tau * work[i] + alpha * v[i];
    }

    // C := C - v*w^H - w*v^H (lower triangle)
    for (int j = 0; j < n; ++j) {
        const T vj_conj = conj_if_needed(v[j]);
        const T wj_conj = conj_if_needed(work[j]);
        for (int i = j; i < n; ++i) {
            C[i + j * ldc] -= v[i] * wj_conj + work[i] * vj_conj;
        }
    }

    // Keep diagonal real for Hermitian.
    if constexpr (internal::is_complex<T>::value) {
        for (int i = 0; i < n; ++i) {
            const T x = C[i + i * ldc];
            C[i + i * ldc] = T(x.real(), typename T::value_type(0));
        }
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
class CopyBandLowerToWorkKernel;

template <typename T>
Event copy_band_lower_to_work(Queue& q,
                              const MatrixView<T, MatrixFormat::Dense>& ab,
                              T* a_ptr,
                              int lda,
                              int n,
                              int kd) {
    const int ldab = ab.ld();
    const int stride_ab = ab.stride();
    const int stride_a = lda * n;
    const T* ab_ptr = ab.data_ptr();
    const int batch = ab.batch_size();

    (void)q->submit([&](sycl::handler& h) {
        h.single_task<CopyBandLowerToWorkKernel<T>>([=]() {
            for (int b = 0; b < batch; ++b) {
                T* A = a_ptr + b * stride_a;
                const T* AB = ab_ptr + b * stride_ab;
                for (int j = 0; j < n; ++j) {
                    // Start from a clean column.
                    for (int r = 0; r < lda; ++r) {
                        A[r + j * lda] = T(0);
                    }

                    // LAPACK lower-case SB2ST workspace layout:
                    // For Uplo::Lower, the stored lower band occupies the top (kd+1) rows
                    // of the (2*kd+1) x n workspace (diagonal at row 0, first subdiagonal at row 1),
                    // and the bottom kd rows are scratch/workspace.
                    for (int r = 0; r <= kd; ++r) {
                        A[r + j * lda] = AB[r + j * ldab];
                    }
                }
            }
        });
    });

    return q.get_event();
}

template <typename T>
class ZeroExtraBandKernel;

template <typename T>
Event zero_extra_band(Queue& q, T* a_ptr, int lda, int n, int kd, int batch) {
    const int stride_a = lda * n;
    (void)q->submit([&](sycl::handler& h) {
        h.single_task<ZeroExtraBandKernel<T>>([=]() {
            for (int b = 0; b < batch; ++b) {
                T* A = a_ptr + b * stride_a;
                for (int j = 0; j < n; ++j) {
                    // In the extended-band workspace (lda = 2*kd+1), the bottom kd rows
                    // are scratch space for the SB2ST/HB2ST bulge-chasing kernels.
                    // Ensure they are zero.
                    for (int r = kd + 1; r < lda; ++r) {
                        A[r + j * lda] = T(0);
                    }
                }
            }
        });
    });
    return q.get_event();
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
        h.single_task<Kd0ExtractKernel<T>>([=]() {
            for (int b = 0; b < batch; ++b) {
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
            }
        });
    });
    return q.get_event();
}

template <typename T>
class Kd1ExtractKernel;

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

template <typename T>
class BulgeChaseLowerKernel;

template <typename T>
Event bulge_chase_lower(Queue& q,
                        T* a_ptr,
                        int lda,
                        int n,
                        int kd,
                        int ib,
                        T* v_ptr,
                        T* tau_ptr,
                        T* work_ptr,
                        int batch) {
    const int stride_a = lda * n;
    const int stride_v = 2 * n;
    const int stride_tau = 2 * n;
    const int stride_w = std::max(1, kd);

    (void)q->submit([&](sycl::handler& h) {
        // Use a single-task kernel since the current bulge-chasing path is
        // sequential per batch and some SYCL backends have issues with certain
        // parallel_for lowering/NDRange adjustment.
        h.single_task<BulgeChaseLowerKernel<T>>([=]() {
            for (int b = 0; b < batch; ++b) {
                T* A = a_ptr + b * stride_a;
                T* V = v_ptr + b * stride_v;
                T* TAU = tau_ptr + b * stride_tau;
                T* WORK = work_ptr + b * stride_w;

#if defined(BATCHLAS_SB2ST_DEBUG_PRINTF)
                const bool dbg = (b == 0) && (n <= 32) && (kd <= 8);
                using Real = typename base_type<T>::type;
                Real trace0 = Real(0);
                Real norm0 = Real(0);
                if (dbg) {
                    for (int j = 0; j < n; ++j) {
                        trace0 += static_cast<Real>(real_part(A[0 + j * lda]));
                    }
                    for (int j = 0; j < n; ++j) {
                        for (int r = 0; r < lda; ++r) {
                            norm0 += abs2_if_complex(A[r + j * lda]);
                        }
                    }
                    sycl::ext::oneapi::experimental::printf(
                        "SB2ST DBG: begin bulge_chase_lower (n=%d kd=%d lda=%d)\n", n, kd, lda);
                    sycl::ext::oneapi::experimental::printf(
                        "SB2ST DBG: trace0=%0.6e norm0=%0.6e\n",
                        static_cast<double>(trace0),
                        static_cast<double>(sycl::sqrt(norm0)));
                    const int jmax = (n < 8) ? n : 8;
                    const int rmax = (lda < 10) ? lda : 10;
                    for (int j = 0; j < jmax; ++j) {
                        for (int r = 0; r < rmax; ++r) {
                            dbg_printf_scalar<T>("A0", r, j, A[r + j * lda]);
                        }
                    }
                }
#endif

            auto a_at = [&](int row1, int col1) -> T& {
                // 1-based row/col
                return A[(row1 - 1) + (col1 - 1) * lda];
            };

            auto hb2st_kernels_lower = [&](int ttype, int st, int ed, int sweep) {
                const int nb = kd;
                // LAPACK lower-case extended-band layout for HB2ST/SB2ST:
                // - diagonal is stored at row 1 (1-based)
                // - first subdiagonal is at row 2 (1-based)
                const int dpos = 1;
                const int ofdpos = 2;

                // The simplified index arithmetic below can produce out-of-range
                // st/ed values near the matrix boundaries. Clamp and/or skip
                // invalid kernel applications to avoid out-of-bounds accesses.
                if (ed < 1 || st > n) return;
                st = std::max(st, 1);
                ed = std::min(ed, n);
                if (st > ed) return;

                // ttype=1 and ttype=3 access column (st-1).
                if ((ttype == 1 || ttype == 3) && st <= 1) return;

                auto v_index = [&](int pos1) -> T& { return V[pos1 - 1]; };
                auto tau_index = [&](int pos1) -> T& { return TAU[pos1 - 1]; };

                const int vpos = ((sweep - 1) & 1) * n + st;
                const int taupos = ((sweep - 1) & 1) * n + st;

                if (ttype == 1) {
                    const int lm = ed - st + 1;
                    v_index(vpos) = T(1);
                    for (int i = 1; i <= lm - 1; ++i) {
                        v_index(vpos + i) = a_at(ofdpos + i, st - 1);
                        a_at(ofdpos + i, st - 1) = T(0);
                    }

                    T& alpha = a_at(ofdpos, st - 1);
                    T* x = &v_index(vpos + 1);

#if defined(BATCHLAS_SB2ST_DEBUG_PRINTF)
                    if (dbg && sweep <= 2 && st <= 4) {
                        sycl::ext::oneapi::experimental::printf(
                            "SB2ST DBG: ttype=1 sweep=%d st=%d ed=%d lm=%d (col=%d)\n",
                            sweep,
                            st,
                            ed,
                            lm,
                            st - 1);
                        dbg_printf_scalar1<T>("alpha_pre", 0, alpha);
                        if (lm >= 2) {
                            dbg_printf_scalar1<T>("x0_pre", 0, x[0]);
                        }
                    }
#endif
                    const T tau = larfg_seq<T>(lm, alpha, x, 1);
                    tau_index(taupos) = tau;

#if defined(BATCHLAS_SB2ST_DEBUG_PRINTF)
                    if (dbg && sweep <= 2 && st <= 4) {
                        dbg_printf_scalar1<T>("alpha_post", 0, alpha);
                        dbg_printf_scalar1<T>("tau", 0, tau);
                    }
#endif

                    T* C = &a_at(dpos, st);
                    larfy_lower_seq<T>(lm, &v_index(vpos), tau, C, lda - 1, WORK);

#if defined(BATCHLAS_SB2ST_DEBUG_PRINTF)
                    if (dbg && sweep <= 2 && st <= 4) {
                        // Print a few diagonal entries of the updated block.
                        const int imax = (lm < 4) ? lm : 4;
                        for (int ii = 0; ii < imax; ++ii) {
                            dbg_printf_scalar<T>("Cdiag", ii, ii, C[ii + ii * (lda - 1)]);
                        }
                    }
#endif
                } else if (ttype == 3) {
                    const int lm = ed - st + 1;
                    T* C = &a_at(1, st);
#if defined(BATCHLAS_SB2ST_DEBUG_PRINTF)
                    if (dbg && sweep <= 2 && st <= 4) {
                        sycl::ext::oneapi::experimental::printf(
                            "SB2ST DBG: ttype=3 sweep=%d st=%d ed=%d lm=%d (col=%d)\n",
                            sweep,
                            st,
                            ed,
                            lm,
                            st);
                        dbg_printf_scalar1<T>("tau3", 0, tau_index(taupos));
                    }
#endif
                    larfy_lower_seq<T>(lm, &v_index(vpos), tau_index(taupos), C, lda - 1, WORK);
                } else if (ttype == 2) {
                    const int j1 = ed + 1;
                    const int j2 = (ed + nb < n) ? (ed + nb) : n;
                    const int ln = ed - st + 1;
                    const int lm = j2 - j1 + 1;
                    if (lm > 0) {
                        // Apply previous reflector from the right.
                        larfx_right_seq<T>(lm, ln, &v_index(vpos), tau_index(taupos), &a_at(dpos + nb, st), lda - 1, WORK);

                        // Build new reflector.
                        const int vpos2 = ((sweep - 1) & 1) * n + j1;
                        const int taupos2 = ((sweep - 1) & 1) * n + j1;
                        v_index(vpos2) = T(1);
                        for (int i = 1; i <= lm - 1; ++i) {
                            v_index(vpos2 + i) = a_at(dpos + nb + i, st);
                            a_at(dpos + nb + i, st) = T(0);
                        }
                        T& alpha = a_at(dpos + nb, st);
                        T* x = &v_index(vpos2 + 1);
#if defined(BATCHLAS_SB2ST_DEBUG_PRINTF)
                        if (dbg && sweep <= 2 && st <= 4) {
                            sycl::ext::oneapi::experimental::printf(
                                "SB2ST DBG: ttype=2 sweep=%d st=%d ed=%d ln=%d lm=%d (col=%d)\n",
                                sweep,
                                st,
                                ed,
                                ln,
                                lm,
                                st);
                            dbg_printf_scalar1<T>("alpha2_pre", 0, alpha);
                            if (lm >= 2) {
                                dbg_printf_scalar1<T>("x2_0_pre", 0, x[0]);
                            }
                        }
#endif
                        const T tau = larfg_seq<T>(lm, alpha, x, 1);
                        tau_index(taupos2) = tau;
#if defined(BATCHLAS_SB2ST_DEBUG_PRINTF)
                        if (dbg && sweep <= 2 && st <= 4) {
                            dbg_printf_scalar1<T>("alpha2_post", 0, alpha);
                            dbg_printf_scalar1<T>("tau2", 0, tau);
                        }
#endif

                        // Apply from the left.
                        if (ln - 1 > 0) {
                            larfx_left_seq<T>(lm, ln - 1, &v_index(vpos2), tau, &a_at(dpos + nb - 1, st + 1), lda - 1, WORK);
                        }
                    }
                }
            };

            // Parameters from LAPACK (simplified single-thread path).
            const int grsiz = 1;
            const int shift = 3;
            const int thgrsiz = n;
            const int stepercol = (shift + grsiz - 1) / grsiz;
            const int thgrnb = ((n - 1) + thgrsiz - 1) / thgrsiz;

                int stt = 1;
                for (int thgrid = 1; thgrid <= thgrnb; ++thgrid) {
                    stt = (thgrid - 1) * thgrsiz + 1;
                    const int thed = ((stt + thgrsiz - 1) < (n - 1)) ? (stt + thgrsiz - 1) : (n - 1);
                    for (int i = stt; i <= n - 1; ++i) {
                        int ed = (i < thed) ? i : thed;
                        if (stt > ed) break;
                        for (int m = 1; m <= stepercol; ++m) {
                            for (int sweepid = stt; sweepid <= ed; ++sweepid) {
                                for (int k = 1; k <= grsiz; ++k) {
                                    const int myid = (i - sweepid) * (stepercol * grsiz) + (m - 1) * grsiz + k;
                                    const int ttype = (myid == 1) ? 1 : ((myid % 2) + 2);

                                    int colpt = 0;
                                    int stind = 0;
                                    int edind = 0;
                                    int blklastind = 0;

                                    if (ttype == 2) {
                                        colpt = (myid / 2) * kd + sweepid;
                                        stind = colpt - kd + 1;
                                        edind = (colpt < n) ? colpt : n;
                                        blklastind = colpt;
                                    } else {
                                        colpt = ((myid + 1) / 2) * kd + sweepid;
                                        stind = colpt - kd + 1;
                                        edind = (colpt < n) ? colpt : n;
                                        if ((stind >= edind - 1) && (edind == n)) {
                                            blklastind = n;
                                        } else {
                                            blklastind = 0;
                                        }
                                    }

                                    hb2st_kernels_lower(ttype, stind, edind, sweepid);

                                    if (blklastind >= (n - 1)) {
                                        stt = stt + 1;
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }

#if defined(BATCHLAS_SB2ST_DEBUG_PRINTF)
                if (dbg) {
                    Real trace1 = Real(0);
                    Real norm1 = Real(0);
                    Real max_off = Real(0);
                    for (int j = 0; j < n; ++j) {
                        trace1 += static_cast<Real>(real_part(A[0 + j * lda]));
                    }
                    for (int j = 0; j < n; ++j) {
                        for (int r = 0; r < lda; ++r) {
                            norm1 += abs2_if_complex(A[r + j * lda]);
                        }
                        for (int r = 2; r < lda; ++r) {
                            const Real a = internal::abs(A[r + j * lda]);
                            if (a > max_off) max_off = a;
                        }
                    }
                    sycl::ext::oneapi::experimental::printf("SB2ST DBG: end bulge_chase_lower\n");
                    sycl::ext::oneapi::experimental::printf(
                        "SB2ST DBG: trace1=%0.6e norm1=%0.6e max_row>=2=%0.6e\n",
                        static_cast<double>(trace1),
                        static_cast<double>(sycl::sqrt(norm1)),
                        static_cast<double>(max_off));
                    const int jmax = (n < 8) ? n : 8;
                    const int rmax = (lda < 10) ? lda : 10;
                    for (int j = 0; j < jmax; ++j) {
                        for (int r = 0; r < rmax; ++r) {
                            dbg_printf_scalar<T>("A1", r, j, A[r + j * lda]);
                        }
                    }
                }
#endif
            }
        });
    });
    return q.get_event();
}

template <typename T>
class ExtractTridiagLowerKernel;

template <typename T>
class ExpandLowerBandToDenseKernel;

template <typename T>
Event expand_lower_band_to_dense(Queue& q,
                                 const MatrixView<T, MatrixFormat::Dense>& ab,
                                 T* a_ptr,
                                 int lda,
                                 int n,
                                 int kd) {
    const int ldab = ab.ld();
    const int stride_ab = ab.stride();
    const int stride_a = lda * n;
    const T* ab_ptr = ab.data_ptr();
    const int batch = ab.batch_size();

    (void)q->submit([&](sycl::handler& h) {
        h.single_task<ExpandLowerBandToDenseKernel<T>>([=]() {
            for (int b = 0; b < batch; ++b) {
                const T* AB = ab_ptr + b * stride_ab;
                T* A = a_ptr + b * stride_a;

                // Zero-fill.
                for (int j = 0; j < n; ++j) {
                    for (int i = 0; i < n; ++i) {
                        A[i + j * lda] = T(0);
                    }
                }

                // Fill lower band and mirror to upper to form a Hermitian dense matrix.
                for (int j = 0; j < n; ++j) {
                    const int rmax = sycl::min(kd, n - 1 - j);
                    // diagonal
                    {
                        T ajj = AB[0 + j * ldab];
                        if constexpr (internal::is_complex<T>::value) {
                            ajj = T(ajj.real(), typename T::value_type(0));
                        }
                        A[j + j * lda] = ajj;
                    }
                    for (int r = 1; r <= rmax; ++r) {
                        const int i = j + r;
                        const T aij = AB[r + j * ldab];
                        A[i + j * lda] = aij;
                        A[j + i * lda] = conj_if_needed(aij);
                    }
                }
            }
        });
    });
    return q.get_event();
}

// === Band-only SB2ST path (real symmetric, lower storage) ===

template <typename Real>
inline void lartg_real(Real f, Real g, Real& c, Real& s, Real& r) {
    if (g == Real(0)) {
        c = Real(1);
        s = Real(0);
        r = f;
        return;
    }
    if (f == Real(0)) {
        c = Real(0);
        s = Real(1);
        r = g;
        return;
    }
    r = sycl::hypot(f, g);
    c = f / r;
    s = g / r;
}

template <typename Real>
inline void drot_real(int nrot, Real* x, int incx, Real* y, int incy, Real c, Real s) {
    for (int t = 0; t < nrot; ++t) {
        const Real xt = x[t * incx];
        const Real yt = y[t * incy];
        x[t * incx] = c * xt + s * yt;
        y[t * incy] = c * yt - s * xt;
    }
}

template <typename Real>
inline void dlar2v_real(int nrot,
                        Real* x,
                        Real* y,
                        Real* z,
                        int incx,
                        const Real* c,
                        const Real* s,
                        int incc) {
    for (int t = 0; t < nrot; ++t) {
        const Real xi = x[t * incx];
        const Real yi = y[t * incx];
        const Real zi = z[t * incx];
        const Real ci = c[t * incc];
        const Real si = s[t * incc];

        const Real t1 = si * zi;
        const Real t2 = ci * zi;
        const Real t3 = t2 - si * xi;
        const Real t4 = t2 + si * yi;
        const Real t5 = ci * xi + t1;
        const Real t6 = ci * yi - t1;

        x[t * incx] = ci * t5 + si * t4;
        y[t * incx] = ci * t6 - si * t3;
        z[t * incx] = ci * t4 - si * t5;
    }
}

template <typename Real>
inline void dlartv_real(int nrot,
                        Real* x,
                        int incx,
                        Real* y,
                        int incy,
                        const Real* c,
                        const Real* s,
                        int incc) {
    for (int t = 0; t < nrot; ++t) {
        const Real ci = c[t * incc];
        const Real si = s[t * incc];
        const Real xt = x[t * incx];
        const Real yt = y[t * incy];
        x[t * incx] = ci * xt + si * yt;
        y[t * incy] = ci * yt - si * xt;
    }
}

template <typename Real>
inline void dlargv_real(int nrot,
                        Real* x,
                        int incx,
                        Real* w,
                        int incw,
                        Real* c,
                        int incc) {
    for (int t = 0; t < nrot; ++t) {
        const Real f = x[t * incx];
        const Real g = w[t * incw];
        Real ct = Real(0), st = Real(0), rt = Real(0);
        lartg_real<Real>(f, g, ct, st, rt);
        c[t * incc] = ct;
        w[t * incw] = st;
        x[t * incx] = rt;
    }
}

template <typename Real>
class SbtrdLowerKernel;

template <typename Real>
Event sbtrd_lower_inplace(Queue& q,
                          Real* ab_ptr,
                          int ldab,
                          int stride_ab,
                          int n,
                          int kd,
                          Real* c_ptr,
                          int stride_c,
                          Real* work_ptr,
                          int stride_work,
                          Real* d_ptr,
                          int stride_d,
                          Real* e_ptr,
                          int stride_e,
                          Real* tau_ptr,
                          int stride_tau,
                          int batch) {
    (void)q->submit([&](sycl::handler& h) {
        h.single_task<SbtrdLowerKernel<Real>>([=]() {
            // Port of LAPACK DSBTRD lower branch (VECT='N') matching the provided Python.
            for (int b = 0; b < batch; ++b) {
                Real* AB = ab_ptr + b * stride_ab;
                Real* C = c_ptr + b * stride_c;
                Real* WORK = work_ptr + b * stride_work;
                Real* D = d_ptr + b * stride_d;
                Real* E = e_ptr + b * stride_e;
                Real* TAU = tau_ptr + b * stride_tau;

                // Zero work arrays (1-based indexing convenience).
                for (int i = 0; i < n + kd + 2; ++i) {
                    C[i] = Real(0);
                    WORK[i] = Real(0);
                }

                const int kd1 = kd + 1;
                const int kdm1 = kd - 1;
                const int incx = ldab - 1;
                const int inca = kd1 * ldab;
                const int kdn = (n - 1 < kd) ? (n - 1) : kd;

                auto idx = [&](int r1, int c1) -> int {
                    // 1-based (r,c) into 0-based flat index (column-major)
                    return (r1 - 1) + (c1 - 1) * ldab;
                };

                int nr = 0;
                int j1 = kdn + 2;
                int j2 = 1;

                for (int i = 1; i <= n - 2; ++i) {
                    for (int k = kdn + 1; k >= 2; --k) {
                        j1 += kdn;
                        j2 += kdn;

                        if (nr > 0) {
                            // Generate plane rotations.
                            dlargv_real<Real>(nr, &AB[idx(kd1, j1 - kd1)], inca, &WORK[j1], kd1, &C[j1], kd1);

                            // Apply plane rotations from one side.
                            if (nr > 2 * kd - 1) {
                                for (int l = 1; l <= kd - 1; ++l) {
                                    dlartv_real<Real>(nr,
                                                      &AB[idx(kd1 - l, j1 - kd1 + l)],
                                                      inca,
                                                      &AB[idx(kd1 - l + 1, j1 - kd1 + l)],
                                                      inca,
                                                      &C[j1],
                                                      &WORK[j1],
                                                      kd1);
                                }
                            } else {
                                const int jend = j1 + (nr - 1) * kd1;
                                for (int jinc = j1; jinc <= jend; jinc += kd1) {
                                    if (kdm1 > 0) {
                                        drot_real<Real>(kdm1,
                                                       &AB[idx(kd, jinc - kd)],
                                                       incx,
                                                       &AB[idx(kd1, jinc - kd)],
                                                       incx,
                                                       C[jinc],
                                                       WORK[jinc]);
                                    }
                                }
                            }
                        }

                        if (k > 2) {
                            if (k <= n - i + 1) {
                                Real f = AB[idx(k - 1, i)];
                                Real g = AB[idx(k, i)];
                                Real ct = Real(0), st = Real(0), rt = Real(0);
                                lartg_real<Real>(f, g, ct, st, rt);
                                C[i + k - 1] = ct;
                                WORK[i + k - 1] = st;
                                AB[idx(k - 1, i)] = rt;

                                if (k - 3 > 0) {
                                    drot_real<Real>(k - 3,
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

                        if (nr > 0) {
                            // Apply plane rotations to diagonal blocks.
                            dlar2v_real<Real>(nr,
                                              &AB[idx(1, j1 - 1)],
                                              &AB[idx(1, j1)],
                                              &AB[idx(2, j1 - 1)],
                                              inca,
                                              &C[j1],
                                              &WORK[j1],
                                              kd1);

                            // Apply plane rotations from the right.
                            if (nr > 2 * kd - 1) {
                                for (int l = 1; l <= kd - 1; ++l) {
                                    const int nrt = (j2 + l > n) ? (nr - 1) : nr;
                                    if (nrt > 0) {
                                        dlartv_real<Real>(nrt,
                                                          &AB[idx(l + 2, j1 - 1)],
                                                          inca,
                                                          &AB[idx(l + 1, j1)],
                                                          inca,
                                                          &C[j1],
                                                          &WORK[j1],
                                                          kd1);
                                    }
                                }
                            } else {
                                const int j1end = j1 + kd1 * (nr - 2);
                                if (j1end >= j1) {
                                    for (int j1inc = j1; j1inc <= j1end; j1inc += kd1) {
                                        if (kdm1 > 0) {
                                            drot_real<Real>(kdm1,
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
                                    drot_real<Real>(lend,
                                                   &AB[idx(3, last - 1)],
                                                   1,
                                                   &AB[idx(2, last)],
                                                   1,
                                                   C[last],
                                                   WORK[last]);
                                }
                            }
                        }

                        if (j2 + kdn > n) {
                            nr -= 1;
                            j2 = j2 - kdn - 1;
                        }

                        // Create nonzero element outside band and store it in WORK.
                        for (int j = j1; j <= j2; j += kd1) {
                            WORK[j + kd] = WORK[j] * AB[idx(kd1, j)];
                            AB[idx(kd1, j)] = C[j] * AB[idx(kd1, j)];
                        }
                    }
                }

                // Extract D and E.
                for (int j = 0; j < n; ++j) {
                    D[j] = AB[idx(1, j + 1)];
                    if (j < n - 1) {
                        E[j] = AB[idx(2, j + 1)];
                        TAU[j] = Real(0);
                    }
                }
            }
        });
    });
    return q.get_event();
}

// Complex Hermitian band -> real tridiagonal (lower storage), ZHBTRD-style (VECT='N').

template <typename T>
inline typename base_type<T>::type abs_complex(const T& z) {
    using Real = typename base_type<T>::type;
    return sycl::hypot(static_cast<Real>(z.real()), static_cast<Real>(z.imag()));
}

template <typename T>
inline void lartg_complex(const T& f, const T& g, typename base_type<T>::type& c, T& s, T& r) {
    using Real = typename base_type<T>::type;
    const Real g_abs = abs_complex(g);
    if (g_abs == Real(0)) {
        c = Real(1);
        s = T(0);
        r = f;
        return;
    }
    const Real f_abs = abs_complex(f);
    if (f_abs == Real(0)) {
        c = Real(0);
        // Match CLARTG/CLARGV convention: choose s so that r is real.
        s = conj_if_needed(g) / T(g_abs);
        r = T(g_abs, Real(0));
        return;
    }

    const Real norm = sycl::hypot(f_abs, g_abs);
    const T alpha = (f_abs == Real(0)) ? T(1) : (f / T(f_abs));
    c = f_abs / norm;
    s = alpha * conj_if_needed(g) / T(norm);
    r = alpha * T(norm);
}

template <typename T>
inline void crot(int nrot,
                 T* x,
                 int incx,
                 T* y,
                 int incy,
                 typename base_type<T>::type c,
                 const T& s) {
    for (int t = 0; t < nrot; ++t) {
        const T xt = x[t * incx];
        const T yt = y[t * incy];
        x[t * incx] = T(c) * xt + s * yt;
        y[t * incy] = T(c) * yt - conj_if_needed(s) * xt;
    }
}

template <typename T>
inline void clartv(int nrot,
                   T* x,
                   int incx,
                   T* y,
                   int incy,
                   const typename base_type<T>::type* c,
                   const T* s,
                   int incc) {
    for (int t = 0; t < nrot; ++t) {
        crot<T>(1, &x[t * incx], 1, &y[t * incy], 1, c[t * incc], s[t * incc]);
    }
}

template <typename T>
inline void clargv(int nrot,
                   T* x,
                   int incx,
                   T* w,
                   int incw,
                   typename base_type<T>::type* c,
                   int incc) {
    for (int t = 0; t < nrot; ++t) {
        const T f = x[t * incx];
        const T g = w[t * incw];
        typename base_type<T>::type ct = typename base_type<T>::type(0);
        T st = T(0);
        T rt = T(0);
        lartg_complex<T>(f, g, ct, st, rt);
        c[t * incc] = ct;
        w[t * incw] = st;
        x[t * incx] = rt;
    }
}

template <typename T>
inline void clar2v(int nrot,
                   T* x,
                   T* y,
                   T* z,
                   int incx,
                   const typename base_type<T>::type* c,
                   const T* s,
                   int incc) {
    // Exact port of Netlib LAPACK CLAR2V logic.
    using Real = typename base_type<T>::type;
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

template <typename T>
class HbtrdLowerKernel;

template <typename T>
Event hbtrd_lower_inplace(Queue& q,
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
        h.single_task<HbtrdLowerKernel<T>>([=]() {
            using Real = typename base_type<T>::type;
            for (int b = 0; b < batch; ++b) {
                T* AB = ab_ptr + b * stride_ab;
                Real* C = c_ptr + b * stride_c;
                T* WORK = work_ptr + b * stride_work;
                Real* D = d_ptr + b * stride_d;
                Real* E = e_ptr + b * stride_e;
                T* TAU = tau_ptr + b * stride_tau;

                for (int i = 0; i < n + kd + 2; ++i) {
                    C[i] = Real(0);
                    WORK[i] = T(0);
                }

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

                // Ensure diagonal is real (ZHBTRD does this).
                AB[idx(1, 1)] = real_as_T(AB[idx(1, 1)]);

                for (int i = 1; i <= n - 2; ++i) {
                    for (int k = kdn + 1; k >= 2; --k) {
                        j1 += kdn;
                        j2 += kdn;

                        if (nr > 0) {
                            clargv<T>(nr, &AB[idx(kd1, j1 - kd1)], inca, &WORK[j1], kd1, &C[j1], kd1);

                            if (nr > 2 * kd - 1) {
                                for (int l = 1; l <= kd - 1; ++l) {
                                    clartv<T>(nr,
                                              &AB[idx(kd1 - l, j1 - kd1 + l)],
                                              inca,
                                              &AB[idx(kd1 - l + 1, j1 - kd1 + l)],
                                              inca,
                                              &C[j1],
                                              &WORK[j1],
                                              kd1);
                                }
                            } else {
                                const int jend = j1 + (nr - 1) * kd1;
                                for (int jinc = j1; jinc <= jend; jinc += kd1) {
                                    if (kdm1 > 0) {
                                        crot<T>(kdm1,
                                                &AB[idx(kd, jinc - kd)],
                                                incx,
                                                &AB[idx(kd1, jinc - kd)],
                                                incx,
                                                C[jinc],
                                                WORK[jinc]);
                                    }
                                }
                            }
                        }

                        if (k > 2) {
                            if (k <= n - i + 1) {
                                const T f = AB[idx(k - 1, i)];
                                const T g = AB[idx(k, i)];
                                Real ct = Real(0);
                                T st = T(0);
                                T rt = T(0);
                                lartg_complex<T>(f, g, ct, st, rt);
                                C[i + k - 1] = ct;
                                WORK[i + k - 1] = st;
                                AB[idx(k - 1, i)] = rt;

                                if (k - 3 > 0) {
                                    crot<T>(k - 3,
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

                        if (nr > 0) {
                            clar2v<T>(nr,
                                      &AB[idx(1, j1 - 1)],
                                      &AB[idx(1, j1)],
                                      &AB[idx(2, j1 - 1)],
                                      inca,
                                      &C[j1],
                                      &WORK[j1],
                                      kd1);

                            // Conjugate WORK(J1) vector before applying from the right.
                            // This matches the ZHBTRD lower branch and the Python reference.
                            for (int t = 0; t < nr; ++t) {
                                WORK[j1 + t * kd1] = conj_if_needed(WORK[j1 + t * kd1]);
                            }

                            if (nr > 2 * kd - 1) {
                                for (int l = 1; l <= kd - 1; ++l) {
                                    const int nrt = (j2 + l > n) ? (nr - 1) : nr;
                                    if (nrt > 0) {
                                        clartv<T>(nrt,
                                                  &AB[idx(l + 2, j1 - 1)],
                                                  inca,
                                                  &AB[idx(l + 1, j1)],
                                                  inca,
                                                  &C[j1],
                                                  &WORK[j1],
                                                  kd1);
                                    }
                                }
                            } else {
                                const int j1end = j1 + kd1 * (nr - 2);
                                if (j1end >= j1) {
                                    for (int j1inc = j1; j1inc <= j1end; j1inc += kd1) {
                                        if (kdm1 > 0) {
                                            crot<T>(kdm1,
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
                                    crot<T>(lend,
                                            &AB[idx(3, last - 1)],
                                            1,
                                            &AB[idx(2, last)],
                                            1,
                                            C[last],
                                            WORK[last]);
                                }
                            }
                        }

                        if (j2 + kdn > n) {
                            nr -= 1;
                            j2 = j2 - kdn - 1;
                        }

                        for (int j = j1; j <= j2; j += kd1) {
                            WORK[j + kd] = WORK[j] * AB[idx(kd1, j)];
                            AB[idx(kd1, j)] = T(C[j]) * AB[idx(kd1, j)];
                        }
                    }
                }

                // Make subdiagonal real (phase chaining) and extract D/E.
                // This matches _make_offdiag_real_inplace in the Python reference.
                for (int j = 0; j < n; ++j) {
                    AB[idx(1, j + 1)] = real_as_T(AB[idx(1, j + 1)]);
                }

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
        });
    });
    return q.get_event();
}

template <typename T>
class ConvertTridiagToRealKernel;

template <typename T>
Event convert_tridiag_to_real(Queue& q,
                              const VectorView<T>& d_in,
                              const VectorView<T>& e_in,
                              const VectorView<T>& tau_in,
                              const VectorView<typename base_type<T>::type>& d_out,
                              const VectorView<typename base_type<T>::type>& e_out,
                              const VectorView<T>& tau_out) {
    const int n = d_in.size();
    const int batch = d_in.batch_size();
    using Real = typename base_type<T>::type;

    const int stride_din = d_in.stride();
    const int stride_ein = e_in.stride();
    const int stride_tin = tau_in.stride();
    const int stride_dout = d_out.stride();
    const int stride_eout = e_out.stride();
    const int stride_tout = tau_out.stride();

    const T* din_ptr = d_in.data_ptr();
    const T* ein_ptr = e_in.data_ptr();
    const T* tin_ptr = tau_in.data_ptr();
    Real* dout_ptr = d_out.data_ptr();
    Real* eout_ptr = e_out.data_ptr();
    T* tout_ptr = tau_out.data_ptr();

    (void)q->submit([&](sycl::handler& h) {
        h.single_task<ConvertTridiagToRealKernel<T>>([=]() {
            for (int b = 0; b < batch; ++b) {
                const T* Db_in = din_ptr + b * stride_din;
                const T* Eb_in = ein_ptr + b * stride_ein;
                const T* Tb_in = tin_ptr + b * stride_tin;
                Real* Db_out = dout_ptr + b * stride_dout;
                Real* Eb_out = eout_ptr + b * stride_eout;
                T* Tb_out = tout_ptr + b * stride_tout;

                for (int i = 0; i < n; ++i) {
                    Db_out[i] = real_part(Db_in[i]);
                }
                for (int i = 0; i < n - 1; ++i) {
                    const T t = Eb_in[i];
                    if constexpr (internal::is_complex<T>::value) {
                        Eb_out[i] = internal::abs(t);
                    } else {
                        Eb_out[i] = static_cast<Real>(t);
                    }
                    // SB2ST VECT='N' path does not expose reflector info downstream.
                    Tb_out[i] = T(0);
                    (void)Tb_in;
                }
            }
        });
    });
    return q.get_event();
}

template <typename T>
Event extract_tridiag_lower(Queue& q,
                            const T* a_ptr,
                            int lda,
                            int n,
                            int kd,
                            int batch,
                            const VectorView<typename base_type<T>::type>& d,
                            const VectorView<typename base_type<T>::type>& e,
                            const VectorView<T>& tau) {
    (void)kd;
    const int stride_a = lda * n;
    const int stride_d = d.stride();
    const int stride_e = e.stride();
    const int stride_tau = tau.stride();
    using Real = typename base_type<T>::type;
    Real* d_ptr = d.data_ptr();
    Real* e_ptr = e.data_ptr();
    T* tau_ptr = tau.data_ptr();

    (void)q->submit([&](sycl::handler& h) {
        h.single_task<ExtractTridiagLowerKernel<T>>([=]() {
            for (int b = 0; b < batch; ++b) {
                const T* A = a_ptr + b * stride_a;
                Real* Db = d_ptr + b * stride_d;
                Real* Eb = e_ptr + b * stride_e;
                T* Taub = tau_ptr + b * stride_tau;

                for (int i = 0; i < n; ++i) {
                    // LAPACK lower-case SB2ST workspace layout:
                    // diagonal is at row 0 (0-based), first subdiagonal at row 1.
                    Db[i] = real_part(A[0 + i * lda]);
                    if (i < n - 1) {
                        const T t = A[1 + i * lda];
                        if constexpr (internal::is_complex<T>::value) {
                            Eb[i] = internal::abs(t);
                        } else {
                            Eb[i] = static_cast<Real>(t);
                        }
                        Taub[i] = T(0);
                    }
                }

#if defined(BATCHLAS_SB2ST_DEBUG_PRINTF)
                if (b == 0 && n <= 32 && kd <= 8) {
                    sycl::ext::oneapi::experimental::printf("SB2ST DBG: extracted d/e (first 8)\n");
                    const int imax = (n < 8) ? n : 8;
                    for (int i = 0; i < imax; ++i) {
                        sycl::ext::oneapi::experimental::printf("d[%d] = %0.6e\n", i, static_cast<double>(Db[i]));
                        if (i < imax - 1) {
                            sycl::ext::oneapi::experimental::printf("e[%d] = %0.6e\n", i, static_cast<double>(Eb[i]));
                        }
                    }
                }
#endif
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

    if constexpr (!internal::is_complex<T>::value) {
        // Band-only SB2ST (DSBTRD-style) workspace:
        // - AB work copy: (kd+1) x n per batch
        // - c/work arrays: length (n+kd+2) per batch
        const int ldab = kd_i + 1;
        const size_t ab_elems = static_cast<size_t>(ldab) * static_cast<size_t>(n) * static_cast<size_t>(batch);
        size += BumpAllocator::allocation_size<T>(ctx, ab_elems);

        const size_t rot_elems = static_cast<size_t>(n + kd_i + 2) * static_cast<size_t>(batch);
        using Real = typename base_type<T>::type;
        size += BumpAllocator::allocation_size<Real>(ctx, rot_elems); // C
        size += BumpAllocator::allocation_size<Real>(ctx, rot_elems); // WORK
    } else {
        // Band-only HBTRD-style workspace (complex Hermitian, lower storage):
        // - AB work copy: (kd+1) x n per batch
        // - c/work arrays: length (n+kd+2) per batch
        const int ldab = kd_i + 1;
        const size_t ab_elems = static_cast<size_t>(ldab) * static_cast<size_t>(n) * static_cast<size_t>(batch);
        size += BumpAllocator::allocation_size<T>(ctx, ab_elems);

        const size_t rot_elems = static_cast<size_t>(n + kd_i + 2) * static_cast<size_t>(batch);
        using Real = typename base_type<T>::type;
        size += BumpAllocator::allocation_size<Real>(ctx, rot_elems); // C
        size += BumpAllocator::allocation_size<T>(ctx, rot_elems);    // WORK (complex)
    }

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

    if constexpr (!internal::is_complex<T>::value) {
        // Band-only SB2ST: copy AB into a mutable workspace and run DSBTRD-style bulge chasing.
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
                h.single_task([=]() {
                    for (int b = 0; b < batch; ++b) {
                        const T* AB = src + b * stride_in;
                        T* W = ab_ptr + b * stride_out;
                        for (int j = 0; j < n; ++j) {
                            for (int r = 0; r < ldab; ++r) {
                                W[r + j * ldab] = AB[r + j * ldab_in];
                            }
                        }
                    }
                });
            });
        }

        using Real = typename base_type<T>::type;
        const size_t rot_elems = static_cast<size_t>(n + kd_i + 2) * static_cast<size_t>(batch);
        auto c_buf = pool.allocate<Real>(ctx, rot_elems);
        auto work_buf = pool.allocate<Real>(ctx, rot_elems);

        (void)sbtrd_lower_inplace<Real>(
            ctx,
            reinterpret_cast<Real*>(ab_ptr),
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
            reinterpret_cast<Real*>(tau_out.data_ptr()),
            tau_out.stride(),
            batch);
    } else {
        // Band-only HBTRD-style bulge chasing (complex Hermitian, lower storage).
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
                h.single_task([=]() {
                    for (int b = 0; b < batch; ++b) {
                        const T* AB = src + b * stride_in;
                        T* W = ab_ptr + b * stride_out;
                        for (int j = 0; j < n; ++j) {
                            for (int r = 0; r < ldab; ++r) {
                                W[r + j * ldab] = AB[r + j * ldab_in];
                            }
                        }
                    }
                });
            });
        }

        using Real = typename base_type<T>::type;
        const size_t rot_elems = static_cast<size_t>(n + kd_i + 2) * static_cast<size_t>(batch);
        auto c_buf = pool.allocate<Real>(ctx, rot_elems);
        auto work_buf = pool.allocate<T>(ctx, rot_elems);

        (void)hbtrd_lower_inplace<T>(
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
