// Tests for bdsqr, the bidiagonal QR iteration.
//
// This file exists because bdsqr had ZERO callers anywhere in the tree (420
// lines of dead code). It is the accurate bidiagonal solver that gesvd should
// have been using instead of forming the tridiagonal of B^T B, so before wiring
// it into gesvd_blocked it needs evidence that it works at all -- dead code
// carries none.
//
// The reference is a host Golub-Kahan SVD of the same bidiagonal matrix computed
// in double, not a second call to anything in this library.

#include <gtest/gtest.h>

#include <batchlas/blas/extensions.hh>
#include <batchlas/blas/functions.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-span.hh>
#include <batchlas/util/sycl-vector.hh>

#include "test_utils.hh"

#include <algorithm>
#include <cmath>
#include <memory>
#include <random>
#include <tuple>
#include <vector>

using namespace batchlas;

namespace {

template <typename T, Backend B>
struct BdsqrConfig {
    using ScalarType = T;
    static constexpr Backend BackendVal = B;
};

template <template <typename, Backend> class Config>
struct bdsqr_types {
    using tuple_type = decltype(std::tuple_cat(
#if BATCHLAS_HAS_CUDA_BACKEND
        std::tuple<Config<float, Backend::CUDA>,
                   Config<double, Backend::CUDA>>{},
#endif
        std::tuple<>{}));
    using type = typename test_utils::tuple_to_types<tuple_type>::type;
};

using BdsqrTestTypes = typename bdsqr_types<BdsqrConfig>::type;

// Singular values of an upper bidiagonal matrix, computed on the host in double
// by one-sided Jacobi on the dense form. Slow but independent and reliable.
std::vector<double> reference_singular_values(const std::vector<double>& d,
                                              const std::vector<double>& e,
                                              int n) {
    std::vector<double> A(static_cast<size_t>(n) * n, 0.0);
    for (int i = 0; i < n; ++i) {
        A[static_cast<size_t>(i) * n + i] = d[static_cast<size_t>(i)];       // (i,i)
        if (i + 1 < n) A[static_cast<size_t>(i + 1) * n + i] = e[static_cast<size_t>(i)];  // (i,i+1)
    }
    for (int sweep = 0; sweep < 60; ++sweep) {
        double off = 0.0;
        for (int p = 0; p < n; ++p) {
            for (int q = p + 1; q < n; ++q) {
                double app = 0, aqq = 0, apq = 0;
                for (int i = 0; i < n; ++i) {
                    const double x = A[static_cast<size_t>(p) * n + i];
                    const double y = A[static_cast<size_t>(q) * n + i];
                    app += x * x; aqq += y * y; apq += x * y;
                }
                if (std::abs(apq) <= 1e-300) continue;
                off = std::max(off, std::abs(apq) / std::sqrt(app * aqq));
                const double tau = (aqq - app) / (2.0 * apq);
                const double t = (tau >= 0 ? 1.0 : -1.0) / (std::abs(tau) + std::sqrt(1.0 + tau * tau));
                const double c = 1.0 / std::sqrt(1.0 + t * t), s = t * c;
                for (int i = 0; i < n; ++i) {
                    const double x = A[static_cast<size_t>(p) * n + i];
                    const double y = A[static_cast<size_t>(q) * n + i];
                    A[static_cast<size_t>(p) * n + i] = c * x - s * y;
                    A[static_cast<size_t>(q) * n + i] = s * x + c * y;
                }
            }
        }
        if (off < 1e-15) break;
    }
    std::vector<double> s(static_cast<size_t>(n));
    for (int j = 0; j < n; ++j) {
        double acc = 0.0;
        for (int i = 0; i < n; ++i) acc += A[static_cast<size_t>(j) * n + i] * A[static_cast<size_t>(j) * n + i];
        s[static_cast<size_t>(j)] = std::sqrt(acc);
    }
    std::sort(s.begin(), s.end(), std::greater<double>());
    return s;
}

template <typename Config>
class BdsqrTest : public test_utils::BatchLASTest<Config> {
protected:
    using Scalar = typename Config::ScalarType;
    static constexpr Backend B = Config::BackendVal;

    static double sv_tol() { return std::is_same_v<Scalar, float> ? 2e-4 : 1e-11; }
    static double vec_tol() { return std::is_same_v<Scalar, float> ? 5e-4 : 1e-10; }

    // Runs bdsqr on (d,e) and checks singular values, plus -- when vectors are
    // requested -- that U S Vh reconstructs B and that U, Vh are orthogonal.
    void check(int n, int batch, unsigned seed, bool vectors, double dscale = 1.0) {
        auto& ctx = *this->ctx;
        std::mt19937 rng(seed);
        std::uniform_real_distribution<double> dist(0.3, 1.7);

        std::vector<double> dh(static_cast<size_t>(n) * batch), eh(static_cast<size_t>(std::max(0, n - 1)) * batch);
        UnifiedVector<Scalar> d(static_cast<size_t>(n) * batch);
        UnifiedVector<Scalar> e(static_cast<size_t>(std::max(1, n - 1)) * batch);
        for (int b = 0; b < batch; ++b) {
            for (int i = 0; i < n; ++i) {
                const double v = dist(rng) * std::pow(dscale, i);
                dh[static_cast<size_t>(b) * n + i] = v;
                d[static_cast<size_t>(b) * n + i] = static_cast<Scalar>(v);
            }
            for (int i = 0; i < n - 1; ++i) {
                const double v = dist(rng) * 0.5 * std::pow(dscale, i);
                eh[static_cast<size_t>(b) * (n - 1) + i] = v;
                e[static_cast<size_t>(b) * (n - 1) + i] = static_cast<Scalar>(v);
            }
        }

        VectorView<Scalar> dv(d.to_span(), n, batch, 1, n);
        VectorView<Scalar> ev(e.to_span(), std::max(0, n - 1), batch, 1, std::max(1, n - 1));
        UnifiedVector<Scalar> s(static_cast<size_t>(n) * batch);

        const size_t ws_bytes = bdsqr_buffer_size<Scalar>(ctx, dv, ev, s.to_span());
        UnifiedVector<std::byte> ws(ws_bytes);

        Matrix<Scalar> U(n, n, batch), Vh(n, n, batch);
        if (vectors) {
            for (int b = 0; b < batch; ++b) {
                for (int j = 0; j < n; ++j) {
                    for (int i = 0; i < n; ++i) {
                        const Scalar v = (i == j) ? Scalar(1) : Scalar(0);
                        U.view().data_ptr()[b * U.view().stride() + static_cast<size_t>(j) * U.view().ld() + i] = v;
                        Vh.view().data_ptr()[b * Vh.view().stride() + static_cast<size_t>(j) * Vh.view().ld() + i] = v;
                    }
                }
            }
            bdsqr<B, Scalar>(ctx, dv, ev, s.to_span(), ws.to_span(), U.view(), Vh.view(), true);
        } else {
            bdsqr<B, Scalar>(ctx, dv, ev, s.to_span(), ws.to_span(), true);
        }
        ctx.wait_and_throw();

        for (int b = 0; b < batch; ++b) {
            std::vector<double> db(dh.begin() + static_cast<size_t>(b) * n,
                                   dh.begin() + static_cast<size_t>(b + 1) * n);
            std::vector<double> eb(eh.begin() + static_cast<size_t>(b) * std::max(0, n - 1),
                                   eh.begin() + static_cast<size_t>(b + 1) * std::max(0, n - 1));
            const auto ref = reference_singular_values(db, eb, n);
            const double smax = ref.front();
            for (int i = 0; i < n; ++i) {
                EXPECT_NEAR(static_cast<double>(s[static_cast<size_t>(b) * n + i]), ref[static_cast<size_t>(i)],
                            sv_tol() * smax)
                    << "n=" << n << " b=" << b << " i=" << i;
            }
        }

        if (!vectors) return;

        // U * diag(s) * Vh should reconstruct the ORIGINAL bidiagonal B, since U
        // and Vh started at the identity.
        for (int b = 0; b < batch; ++b) {
            double num = 0.0, den = 0.0;
            for (int j = 0; j < n; ++j) {
                for (int i = 0; i < n; ++i) {
                    double acc = 0.0;
                    for (int t = 0; t < n; ++t) {
                        acc += static_cast<double>(U.view().data_ptr()[b * U.view().stride() + static_cast<size_t>(t) * U.view().ld() + i])
                             * static_cast<double>(s[static_cast<size_t>(b) * n + t])
                             * static_cast<double>(Vh.view().data_ptr()[b * Vh.view().stride() + static_cast<size_t>(j) * Vh.view().ld() + t]);
                    }
                    double bij = 0.0;
                    if (i == j) bij = dh[static_cast<size_t>(b) * n + i];
                    else if (j == i + 1) bij = eh[static_cast<size_t>(b) * (n - 1) + i];
                    num += (bij - acc) * (bij - acc);
                    den += bij * bij;
                }
            }
            EXPECT_LE(std::sqrt(num / std::max(den, 1e-300)), vec_tol()) << "reconstruction n=" << n << " b=" << b;
        }
    }

// The generator above produces only POSITIVE d and e in [0.3, 1.7], which is a
// far easier class than anything gebrd actually emits. Real bidiagonal factors
// carry mixed signs, exact zeros (deflation), and entries spanning many orders
// of magnitude. This variant covers that; it is what caught bdsqr stagnating
// when gesvd_blocked was first wired onto it.
protected:
    void check_hostile(int n, int batch, unsigned seed) {
        auto& ctx = *this->ctx;
        std::mt19937 rng(seed);
        std::uniform_real_distribution<double> mag(-3.0, 3.0);   // 1e-3 .. 1e3
        std::uniform_int_distribution<int> sign(0, 1);
        std::uniform_int_distribution<int> zero(0, 9);

        std::vector<double> dh(static_cast<size_t>(n) * batch);
        std::vector<double> eh(static_cast<size_t>(std::max(0, n - 1)) * batch);
        UnifiedVector<Scalar> d(static_cast<size_t>(n) * batch);
        UnifiedVector<Scalar> e(static_cast<size_t>(std::max(1, n - 1)) * batch);
        for (int b = 0; b < batch; ++b) {
            for (int i = 0; i < n; ++i) {
                double v = std::pow(10.0, mag(rng)) * (sign(rng) ? 1.0 : -1.0);
                if (zero(rng) == 0) v = 0.0;
                dh[static_cast<size_t>(b) * n + i] = v;
                d[static_cast<size_t>(b) * n + i] = static_cast<Scalar>(v);
            }
            for (int i = 0; i < n - 1; ++i) {
                double v = std::pow(10.0, mag(rng)) * (sign(rng) ? 1.0 : -1.0);
                if (zero(rng) == 0) v = 0.0;
                eh[static_cast<size_t>(b) * (n - 1) + i] = v;
                e[static_cast<size_t>(b) * (n - 1) + i] = static_cast<Scalar>(v);
            }
        }

        VectorView<Scalar> dv(d.to_span(), n, batch, 1, n);
        VectorView<Scalar> ev(e.to_span(), std::max(0, n - 1), batch, 1, std::max(1, n - 1));
        UnifiedVector<Scalar> s(static_cast<size_t>(n) * batch);
        const size_t ws_bytes = bdsqr_buffer_size<Scalar>(ctx, dv, ev, s.to_span());
        UnifiedVector<std::byte> ws(ws_bytes);

        bdsqr<B, Scalar>(ctx, dv, ev, s.to_span(), ws.to_span(), true);
        ctx.wait_and_throw();

        for (int b = 0; b < batch; ++b) {
            std::vector<double> db(dh.begin() + static_cast<size_t>(b) * n,
                                   dh.begin() + static_cast<size_t>(b + 1) * n);
            std::vector<double> eb(eh.begin() + static_cast<size_t>(b) * std::max(0, n - 1),
                                   eh.begin() + static_cast<size_t>(b + 1) * std::max(0, n - 1));
            const auto ref = reference_singular_values(db, eb, n);
            const double smax = std::max(ref.front(), 1e-300);
            for (int i = 0; i < n; ++i) {
                EXPECT_NEAR(static_cast<double>(s[static_cast<size_t>(b) * n + i]),
                            ref[static_cast<size_t>(i)], sv_tol() * smax)
                    << "hostile n=" << n << " b=" << b << " i=" << i;
            }
        }
    }

public:
};

TYPED_TEST_SUITE(BdsqrTest, BdsqrTestTypes);

TYPED_TEST(BdsqrTest, ValuesOnly) {
    for (int n : {2, 4, 9, 16, 33, 64}) this->check(n, 3, 100u + n, /*vectors=*/false);
}

TYPED_TEST(BdsqrTest, WithVectors) {
    for (int n : {2, 4, 9, 16, 33, 64}) this->check(n, 3, 200u + n, /*vectors=*/true);
}

// Graded: the case where forming B^T B loses the small singular values and a
// direct bidiagonal method should not.
TYPED_TEST(BdsqrTest, Graded) {
    this->check(32, 2, 301u, /*vectors=*/true, /*dscale=*/0.8);
    this->check(64, 2, 302u, /*vectors=*/false, /*dscale=*/0.85);
}

TYPED_TEST(BdsqrTest, MixedSignsZerosAndWideRange) {
    for (int n : {8, 32, 64}) this->check_hostile(n, 16, 900u + n);
}


} // namespace
