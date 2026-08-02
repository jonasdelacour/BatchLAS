// Validation for the Householder stage-2 chase (sytrd_sb2st_hh).
//
// The point of this routine, versus the shipped Givens sytrd_sb2st, is that it
// *retains* the reflectors so the eigenvector back-transform is possible. So the
// load-bearing test is not "does the spectrum match" -- it is "do the stored
// reflectors reproduce the similarity transform":
//
//     Q = H_1 H_2 ... H_m  (generation order),  Q^H A Q = T
//
// This mirrors playground/validate_hous2_q.py, which established the same
// property for the Python reference. As there, we compare magnitudes on the
// subdiagonal, because the |.| taken when forming the real tridiagonal is a
// diagonal phase similarity and not part of Q.

#include <gtest/gtest.h>

#include <blas/enums.hh>
#include <blas/extensions.hh>
#include <blas/matrix.hh>
#include <util/sycl-device-queue.hh>
#include <util/sycl-span.hh>

#include <algorithm>
#include <cmath>
#include <complex>
#include <random>
#include <type_traits>
#include <vector>

#include "../src/extensions/sytrd_sb2st_hh.hh"
#include "test_utils.hh"

using namespace batchlas;

namespace {

template <typename U>
inline U conj_if(const U& x) {
    if constexpr (std::is_same_v<U, std::complex<float>> ||
                  std::is_same_v<U, std::complex<double>>) {
        return std::conj(x);
    } else {
        return x;
    }
}

template <typename T>
inline typename base_type<T>::type abs_of(const T& x) {
    if constexpr (std::is_same_v<T, std::complex<float>> ||
                  std::is_same_v<T, std::complex<double>>) {
        return std::abs(x);
    } else {
        return std::abs(x);
    }
}

// Dense Hermitian band matrix, row-major n x n on the host.
template <typename T>
std::vector<T> make_banded_hermitian(int n, int kd, unsigned seed) {
    using Real = typename base_type<T>::type;
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<T> A(static_cast<size_t>(n) * n, T(0));
    for (int j = 0; j < n; ++j) {
        for (int i = j; i < n && i <= j + kd; ++i) {
            T v;
            if constexpr (std::is_same_v<T, std::complex<float>> ||
                          std::is_same_v<T, std::complex<double>>) {
                v = T(static_cast<Real>(dist(gen)), static_cast<Real>(dist(gen)));
                if (i == j) v = T(v.real(), Real(0));
            } else {
                v = static_cast<T>(dist(gen));
            }
            A[static_cast<size_t>(i) * n + j] = v;
            A[static_cast<size_t>(j) * n + i] = conj_if(v);
        }
    }
    return A;
}

template <typename T, Backend Back>
struct Sb2stHhConfig {
    using ScalarType = T;
    static constexpr Backend BackendVal = Back;
};

template <typename Config>
class Sb2stHhTest : public test_utils::BatchLASTest<Config> {};

#if BATCHLAS_HAS_CUDA_BACKEND
using Sb2stHhTypes = ::testing::Types<
    Sb2stHhConfig<float, Backend::CUDA>,
    Sb2stHhConfig<double, Backend::CUDA>,
    Sb2stHhConfig<std::complex<float>, Backend::CUDA>,
    Sb2stHhConfig<std::complex<double>, Backend::CUDA>>;
#elif BATCHLAS_HAS_ROCM_BACKEND
using Sb2stHhTypes = ::testing::Types<
    Sb2stHhConfig<float, Backend::ROCM>,
    Sb2stHhConfig<double, Backend::ROCM>>;
#else
using Sb2stHhTypes = ::testing::Types<Sb2stHhConfig<float, Backend::NETLIB>>;
#endif

TYPED_TEST_SUITE(Sb2stHhTest, Sb2stHhTypes);

TYPED_TEST(Sb2stHhTest, StoredReflectorsReproduceSimilarity) {
    using T = typename TestFixture::ScalarType;
    using Real = typename base_type<T>::type;
    constexpr Backend B = TestFixture::BackendType;

    auto& ctx = *this->ctx;

    const Real tol = std::is_same_v<Real, float> ? Real(2e-3) : Real(1e-10);

    for (int n : {16, 24, 32, 48, 129, 200}) {
        for (int kd : {2, 3, 4, 8, 16, 33, 48, 64}) {
            if (kd >= n) continue;
            const int batch = 3;

            // Host dense reference matrices, one per batch item.
            std::vector<std::vector<T>> Adense;
            for (int b = 0; b < batch; ++b) {
                Adense.push_back(make_banded_hermitian<T>(n, kd, 991u + 7u * n + 13u * kd + b));
            }

            // Pack into lower band storage (kd+1) x n.
            Matrix<T, MatrixFormat::Dense> ab(kd + 1, n, batch);
            for (int b = 0; b < batch; ++b) {
                for (int j = 0; j < n; ++j) {
                    for (int r = 0; r <= kd; ++r) {
                        const int i = j + r;
                        ab(r, j, b) = (i < n) ? Adense[b][static_cast<size_t>(i) * n + j] : T(0);
                    }
                }
            }

            const auto sched = internal::build_sb2st_hh_schedule(n, kd);
            const int nrefl = std::max<int>(1, static_cast<int>(sched.size()));

            Matrix<T, MatrixFormat::Dense> ab_tri(2, n, batch);
            Matrix<T, MatrixFormat::Dense> vmat(std::max(1, kd), nrefl, batch);
            Vector<T> tau(nrefl, batch);
            Vector<Real> d(n, batch);
            Vector<Real> e(std::max(1, n - 1), batch);

            const size_t ws_bytes =
                internal::sytrd_sb2st_hh_buffer_size<B, T>(ctx, n, kd, batch);
            UnifiedVector<std::byte> ws(ws_bytes);

            internal::sytrd_sb2st_hh<B, T>(ctx, ab, ab_tri, d, e, vmat, tau,
                                           Uplo::Lower, kd, ws.to_span());
            ctx.wait();

            for (int b = 0; b < batch; ++b) {
                // Q = H_1 ... H_m in generation order, accumulated as Q <- Q H_k.
                std::vector<T> Q(static_cast<size_t>(n) * n, T(0));
                for (int i = 0; i < n; ++i) Q[static_cast<size_t>(i) * n + i] = T(1);

                for (int k = 0; k < static_cast<int>(sched.size()); ++k) {
                    const T tk = tau(k, b);
                    if (tk == T(0)) continue;
                    const int s = sched[k].start;
                    const int m = sched[k].len;
                    // Q[:, s:s+m] -= (Q[:, s:s+m] v) (tau v^H)
                    for (int i = 0; i < n; ++i) {
                        T acc = T(0);
                        for (int j = 0; j < m; ++j) {
                            acc += Q[static_cast<size_t>(i) * n + (s + j)] * vmat(j, k, b);
                        }
                        for (int j = 0; j < m; ++j) {
                            Q[static_cast<size_t>(i) * n + (s + j)] -=
                                acc * tk * conj_if(vmat(j, k, b));
                        }
                    }
                }

                // B = Q^H A Q
                std::vector<T> AQ(static_cast<size_t>(n) * n, T(0));
                for (int i = 0; i < n; ++i)
                    for (int j = 0; j < n; ++j) {
                        T acc = T(0);
                        for (int l = 0; l < n; ++l)
                            acc += Adense[b][static_cast<size_t>(i) * n + l] *
                                   Q[static_cast<size_t>(l) * n + j];
                        AQ[static_cast<size_t>(i) * n + j] = acc;
                    }
                std::vector<T> Bm(static_cast<size_t>(n) * n, T(0));
                for (int i = 0; i < n; ++i)
                    for (int j = 0; j < n; ++j) {
                        T acc = T(0);
                        for (int l = 0; l < n; ++l)
                            acc += conj_if(Q[static_cast<size_t>(l) * n + i]) *
                                   AQ[static_cast<size_t>(l) * n + j];
                        Bm[static_cast<size_t>(i) * n + j] = acc;
                    }

                Real scale = Real(0);
                for (int i = 0; i < n; ++i)
                    for (int j = 0; j < n; ++j)
                        scale += abs_of(Adense[b][static_cast<size_t>(i) * n + j]) *
                                 abs_of(Adense[b][static_cast<size_t>(i) * n + j]);
                scale = std::max(std::sqrt(scale), Real(1));

                // Q must be orthogonal.
                Real orth = Real(0);
                for (int i = 0; i < n; ++i)
                    for (int j = 0; j < n; ++j) {
                        T acc = T(0);
                        for (int l = 0; l < n; ++l)
                            acc += conj_if(Q[static_cast<size_t>(l) * n + i]) *
                                   Q[static_cast<size_t>(l) * n + j];
                        if (i == j) acc -= T(1);
                        orth += abs_of(acc) * abs_of(acc);
                    }
                EXPECT_LT(std::sqrt(orth), tol) << "n=" << n << " kd=" << kd << " b=" << b;

                // Q^H A Q must be tridiagonal, with diag == d and |subdiag| == e.
                Real offtri = Real(0);
                for (int i = 0; i < n; ++i)
                    for (int j = 0; j < n; ++j)
                        if (std::abs(i - j) > 1)
                            offtri += abs_of(Bm[static_cast<size_t>(i) * n + j]) *
                                      abs_of(Bm[static_cast<size_t>(i) * n + j]);
                EXPECT_LT(std::sqrt(offtri) / scale, tol)
                    << "off-tridiagonal n=" << n << " kd=" << kd << " b=" << b;

                for (int i = 0; i < n; ++i) {
                    const T bii = Bm[static_cast<size_t>(i) * n + i];
                    Real re;
                    if constexpr (std::is_same_v<T, std::complex<float>> ||
                                  std::is_same_v<T, std::complex<double>>) {
                        re = bii.real();
                    } else {
                        re = bii;
                    }
                    EXPECT_NEAR(re, d(i, b), tol * scale)
                        << "d[" << i << "] n=" << n << " kd=" << kd;
                }
                for (int i = 0; i + 1 < n; ++i) {
                    const Real got = abs_of(Bm[static_cast<size_t>(i + 1) * n + i]);
                    EXPECT_NEAR(got, e(i, b), tol * scale)
                        << "e[" << i << "] n=" << n << " kd=" << kd;
                }
            }
        }
    }
}

// The back-transform must implement Z := Q2 Z for the same Q2 the previous test
// validated. Feeding it the identity therefore has to produce Q2 itself, which
// we check against a host accumulation of the stored reflectors.
TYPED_TEST(Sb2stHhTest, BackTransformAppliesQ2) {
    using T = typename TestFixture::ScalarType;
    using Real = typename base_type<T>::type;
    constexpr Backend B = TestFixture::BackendType;

    auto& ctx = *this->ctx;
    const Real tol = std::is_same_v<Real, float> ? Real(2e-3) : Real(1e-10);

    for (int n : {16, 32, 48, 160}) {
        for (int kd : {2, 4, 8, 40, 64}) {
            if (kd >= n) continue;
            const int batch = 2;

            std::vector<std::vector<T>> Adense;
            for (int b = 0; b < batch; ++b) {
                Adense.push_back(make_banded_hermitian<T>(n, kd, 55u + 3u * n + 17u * kd + b));
            }

            Matrix<T, MatrixFormat::Dense> ab(kd + 1, n, batch);
            for (int b = 0; b < batch; ++b)
                for (int j = 0; j < n; ++j)
                    for (int r = 0; r <= kd; ++r) {
                        const int i = j + r;
                        ab(r, j, b) = (i < n) ? Adense[b][static_cast<size_t>(i) * n + j] : T(0);
                    }

            const auto sched = internal::build_sb2st_hh_schedule(n, kd);
            const int nrefl = std::max<int>(1, static_cast<int>(sched.size()));

            Matrix<T, MatrixFormat::Dense> ab_tri(2, n, batch);
            Matrix<T, MatrixFormat::Dense> vmat(std::max(1, kd), nrefl, batch);
            Vector<T> tau(nrefl, batch);
            Vector<Real> d(n, batch);
            Vector<Real> e(std::max(1, n - 1), batch);

            UnifiedVector<std::byte> ws(
                internal::sytrd_sb2st_hh_buffer_size<B, T>(ctx, n, kd, batch));
            internal::sytrd_sb2st_hh<B, T>(ctx, ab, ab_tri, d, e, vmat, tau,
                                           Uplo::Lower, kd, ws.to_span());
            ctx.wait();

            // Z = I, then Z := Q2 Z.
            Matrix<T, MatrixFormat::Dense> Z(n, n, batch);
            for (int b = 0; b < batch; ++b)
                for (int i = 0; i < n; ++i)
                    for (int j = 0; j < n; ++j) Z(i, j, b) = (i == j) ? T(1) : T(0);

            UnifiedVector<int32_t> starts(static_cast<size_t>(sched.size()));
            UnifiedVector<int32_t> lens(static_cast<size_t>(sched.size()));
            for (size_t k = 0; k < sched.size(); ++k) {
                starts[k] = sched[k].start;
                lens[k] = sched[k].len;
            }

            internal::unmqr_hb2st<B, T>(
                ctx, vmat, tau, Z, n, kd,
                Span<const int32_t>(starts.data(), sched.size()),
                Span<const int32_t>(lens.data(), sched.size()));
            ctx.wait();

            for (int b = 0; b < batch; ++b) {
                std::vector<T> Qref(static_cast<size_t>(n) * n, T(0));
                for (int i = 0; i < n; ++i) Qref[static_cast<size_t>(i) * n + i] = T(1);
                for (int k = 0; k < static_cast<int>(sched.size()); ++k) {
                    const T tk = tau(k, b);
                    if (tk == T(0)) continue;
                    const int s = sched[k].start;
                    const int m = sched[k].len;
                    for (int i = 0; i < n; ++i) {
                        T acc = T(0);
                        for (int j = 0; j < m; ++j)
                            acc += Qref[static_cast<size_t>(i) * n + (s + j)] * vmat(j, k, b);
                        for (int j = 0; j < m; ++j)
                            Qref[static_cast<size_t>(i) * n + (s + j)] -=
                                acc * tk * conj_if(vmat(j, k, b));
                    }
                }
                Real err = Real(0);
                for (int i = 0; i < n; ++i)
                    for (int j = 0; j < n; ++j) {
                        const T diff = Z(i, j, b) - Qref[static_cast<size_t>(i) * n + j];
                        err += abs_of(diff) * abs_of(diff);
                    }
                EXPECT_LT(std::sqrt(err), tol)
                    << "Q2 mismatch n=" << n << " kd=" << kd << " b=" << b;
            }
        }
    }
}

} // namespace
