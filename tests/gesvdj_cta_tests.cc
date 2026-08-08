// Tests for gesvdj_cta, the one-sided Jacobi SVD.
//
// Checks are host-side and self-contained rather than shared with
// gesvd_tests.cc, deliberately: the tolerances there (float: 5e-2 singular
// values, 2e-1 orthogonality, 3e-1 reconstruction) are calibrated for a
// normal-equations solver and are far too loose to detect a regression in this
// kernel. See GESVD_PLAN.md section 2.1.
//
// The rectangular, complex and rank-deficient cases are the ones NOT covered by
// the n=32 square benchmark shape, so they are the reason this file exists.

#include <gtest/gtest.h>

#include <blas/extensions.hh>
#include <blas/functions.hh>
#include <blas/matrix.hh>
#include <util/sycl-span.hh>
#include <util/sycl-vector.hh>

#include "test_utils.hh"

#include <algorithm>
#include <cmath>
#include <complex>
#include <memory>
#include <random>
#include <tuple>
#include <vector>

using namespace batchlas;

namespace {

template <typename T, Backend B>
struct GesvdjConfig {
    using ScalarType = T;
    static constexpr Backend BackendVal = B;
};

template <template <typename, Backend> class Config>
struct gesvdj_types {
    using tuple_type = decltype(std::tuple_cat(
#if BATCHLAS_HAS_CUDA_BACKEND
        std::tuple<Config<float, Backend::CUDA>,
                   Config<double, Backend::CUDA>,
                   Config<std::complex<float>, Backend::CUDA>,
                   Config<std::complex<double>, Backend::CUDA>>{},
#endif
        std::tuple<>{}));
    using type = typename test_utils::tuple_to_types<tuple_type>::type;
};

using GesvdjTestTypes = typename gesvdj_types<GesvdjConfig>::type;

template <typename T> struct is_cplx_h : std::false_type {};
template <typename T> struct is_cplx_h<std::complex<T>> : std::true_type {};
template <typename T> inline constexpr bool is_cplx_v = is_cplx_h<T>::value;

template <typename T>
inline T conj_h(const T& x) {
    if constexpr (is_cplx_v<T>) {
        return std::conj(x);
    } else {
        return x;
    }
}

template <typename T>
inline double abs2_h(const T& x) {
    if constexpr (is_cplx_v<T>) {
        return static_cast<double>(std::norm(x));
    } else {
        return static_cast<double>(x) * static_cast<double>(x);
    }
}

template <typename Config>
class GesvdjCtaTest : public test_utils::BatchLASTest<Config> {
protected:
    using Scalar = typename Config::ScalarType;
    using Real = typename base_type<Scalar>::type;
    static constexpr Backend B = Config::BackendVal;

    static Real recon_tol() { return std::is_same_v<Real, float> ? Real(2e-4f) : Real(1e-11); }
    static Real ortho_tol() { return std::is_same_v<Real, float> ? Real(2e-4f) : Real(1e-11); }

    // ||A - U diag(s) Vh||_F / ||A||_F, computed on the host in double.
    // u_ld / vh_ld are passed explicitly rather than assumed to be m and n:
    // a thin V^H is k x n, so its leading dimension is k, and hardcoding n
    // would silently read the wrong elements.
    static double reconstruction(int m, int n, int batch,
                                 const std::vector<Scalar>& A,
                                 const Scalar* U, int64_t u_stride, int u_ld,
                                 const Scalar* Vh, int64_t vh_stride, int vh_ld,
                                 const Real* s) {
        const int k = std::min(m, n);
        double worst = 0.0;
        for (int b = 0; b < batch; ++b) {
            double num = 0.0, den = 0.0;
            for (int j = 0; j < n; ++j) {
                for (int i = 0; i < m; ++i) {
                    std::complex<double> acc(0.0, 0.0);
                    for (int t = 0; t < k; ++t) {
                        const std::complex<double> u = to_cd(U[b * u_stride + static_cast<size_t>(t) * u_ld + i]);
                        const std::complex<double> v = to_cd(Vh[b * vh_stride + static_cast<size_t>(j) * vh_ld + t]);
                        acc += u * static_cast<double>(s[b * k + t]) * v;
                    }
                    const std::complex<double> a = to_cd(A[static_cast<size_t>(b) * m * n + static_cast<size_t>(j) * m + i]);
                    num += std::norm(a - acc);
                    den += std::norm(a);
                }
            }
            worst = std::max(worst, den > 0.0 ? std::sqrt(num / den) : std::sqrt(num));
        }
        return worst;
    }

    // max |M^H M - I| over the leading `cols` columns of an ld x cols block.
    static double col_orthogonality(int rows, int cols, int batch,
                                    const Scalar* M, int64_t stride, int ld) {
        double worst = 0.0;
        for (int b = 0; b < batch; ++b) {
            for (int p = 0; p < cols; ++p) {
                for (int q = 0; q < cols; ++q) {
                    std::complex<double> acc(0.0, 0.0);
                    for (int t = 0; t < rows; ++t) {
                        acc += std::conj(to_cd(M[b * stride + static_cast<size_t>(p) * ld + t]))
                             * to_cd(M[b * stride + static_cast<size_t>(q) * ld + t]);
                    }
                    const double target = (p == q) ? 1.0 : 0.0;
                    worst = std::max(worst, std::abs(acc - target));
                }
            }
        }
        return worst;
    }

    static std::complex<double> to_cd(const Scalar& x) {
        if constexpr (is_cplx_v<Scalar>) {
            return std::complex<double>(static_cast<double>(x.real()), static_cast<double>(x.imag()));
        } else {
            return std::complex<double>(static_cast<double>(x), 0.0);
        }
    }

    // Runs gesvdj_cta on a caller-provided A and validates the factorisation.
    void check(int m, int n, int batch, std::vector<Scalar> host_A) {
        auto& ctx = *this->ctx;
        const int k = std::min(m, n);

        Matrix<Scalar> A(m, n, batch);
        for (int b = 0; b < batch; ++b) {
            for (int j = 0; j < n; ++j) {
                for (int i = 0; i < m; ++i) {
                    A.view().data_ptr()[b * A.view().stride() + static_cast<size_t>(j) * A.view().ld() + i] =
                        host_A[static_cast<size_t>(b) * m * n + static_cast<size_t>(j) * m + i];
                }
            }
        }

        Matrix<Scalar> U(m, m, batch);
        Matrix<Scalar> Vh(n, n, batch);
        UnifiedVector<Real> s(static_cast<size_t>(k) * batch);

        gesvdj_cta<B, Scalar>(ctx, A.view(), s.to_span(), U.view(), Vh.view(),
                              SvdVectors::All, SvdVectors::All);
        ctx.wait_and_throw();

        // Descending, and non-negative.
        for (int b = 0; b < batch; ++b) {
            for (int i = 0; i < k; ++i) {
                EXPECT_GE(s[b * k + i], Real(0)) << "negative sigma at b=" << b << " i=" << i;
                if (i > 0) {
                    EXPECT_LE(s[b * k + i], s[b * k + i - 1] * (Real(1) + Real(1e-5)))
                        << "not descending at b=" << b << " i=" << i;
                }
            }
        }

        EXPECT_LE(reconstruction(m, n, batch, host_A,
                                 U.view().data_ptr(), U.view().stride(), static_cast<int>(U.view().ld()),
                                 Vh.view().data_ptr(), Vh.view().stride(), static_cast<int>(Vh.view().ld()),
                                 s.data()),
                  static_cast<double>(recon_tol()))
            << "reconstruction m=" << m << " n=" << n;

        EXPECT_LE(col_orthogonality(m, m, batch, U.view().data_ptr(), U.view().stride(),
                                    static_cast<int>(U.view().ld())),
                  static_cast<double>(ortho_tol()))
            << "U orthogonality m=" << m << " n=" << n;

        // Vh's ROWS are the right singular vectors, so check Vh^H's columns by
        // transposing the access: equivalently check that Vh Vh^H = I.
        std::vector<Scalar> vht(static_cast<size_t>(n) * n * batch);
        for (int b = 0; b < batch; ++b) {
            for (int j = 0; j < n; ++j) {
                for (int i = 0; i < n; ++i) {
                    vht[static_cast<size_t>(b) * n * n + static_cast<size_t>(j) * n + i] =
                        conj_h(Vh.view().data_ptr()[b * Vh.view().stride() + static_cast<size_t>(i) * Vh.view().ld() + j]);
                }
            }
        }
        EXPECT_LE(col_orthogonality(n, n, batch, vht.data(), static_cast<int64_t>(n) * n, n),
                  static_cast<double>(ortho_tol()))
            << "V orthogonality m=" << m << " n=" << n;
    }

    // Same validation as check(), but with U as m x k and Vh as k x n.
    //
    // Worth stating what this can and cannot catch on its own: for m >= n a
    // thin V^H IS a full V^H, and for m <= n a thin U IS a full U, so exactly
    // one side is genuinely narrower in each shape. The tall and wide cases
    // below therefore exercise DIFFERENT code -- the tall one skips the
    // in-kernel Gram-Schmidt completion, the wide one takes the transposed
    // writeback where the thin bound lands on `lane` rather than on the inner
    // loop. Both are needed.
    void check_thin(int m, int n, int batch, std::vector<Scalar> host_A) {
        auto& ctx = *this->ctx;
        const int k = std::min(m, n);

        Matrix<Scalar> A(m, n, batch);
        for (int b = 0; b < batch; ++b) {
            for (int j = 0; j < n; ++j) {
                for (int i = 0; i < m; ++i) {
                    A.view().data_ptr()[b * A.view().stride() + static_cast<size_t>(j) * A.view().ld() + i] =
                        host_A[static_cast<size_t>(b) * m * n + static_cast<size_t>(j) * m + i];
                }
            }
        }

        Matrix<Scalar> U(m, k, batch);
        Matrix<Scalar> Vh(k, n, batch);
        UnifiedVector<Real> s(static_cast<size_t>(k) * batch);

        gesvdj_cta<B, Scalar>(ctx, A.view(), s.to_span(), U.view(), Vh.view(),
                              SvdVectors::Thin, SvdVectors::Thin);
        ctx.wait_and_throw();

        for (int b = 0; b < batch; ++b) {
            for (int i = 0; i < k; ++i) {
                EXPECT_GE(s[b * k + i], Real(0)) << "negative sigma at b=" << b << " i=" << i;
                if (i > 0) {
                    EXPECT_LE(s[b * k + i], s[b * k + i - 1] * (Real(1) + Real(1e-5)))
                        << "not descending at b=" << b << " i=" << i;
                }
            }
        }

        // A = U S V^H must still hold exactly: the discarded columns of a full U
        // multiply zero singular values, so thin loses nothing here.
        EXPECT_LE(reconstruction(m, n, batch, host_A,
                                 U.view().data_ptr(), U.view().stride(), static_cast<int>(U.view().ld()),
                                 Vh.view().data_ptr(), Vh.view().stride(), static_cast<int>(Vh.view().ld()),
                                 s.data()),
                  static_cast<double>(recon_tol()))
            << "thin reconstruction m=" << m << " n=" << n;

        EXPECT_LE(col_orthogonality(m, k, batch, U.view().data_ptr(), U.view().stride(),
                                    static_cast<int>(U.view().ld())),
                  static_cast<double>(ortho_tol()))
            << "thin U orthogonality m=" << m << " n=" << n;

        // Vh is k x n; its k ROWS must be orthonormal, so build Vh^H (n x k)
        // and check its columns.
        std::vector<Scalar> vht(static_cast<size_t>(n) * k * batch);
        for (int b = 0; b < batch; ++b) {
            for (int j = 0; j < k; ++j) {
                for (int i = 0; i < n; ++i) {
                    vht[static_cast<size_t>(b) * n * k + static_cast<size_t>(j) * n + i] =
                        conj_h(Vh.view().data_ptr()[b * Vh.view().stride() + static_cast<size_t>(i) * Vh.view().ld() + j]);
                }
            }
        }
        EXPECT_LE(col_orthogonality(n, k, batch, vht.data(), static_cast<int64_t>(n) * k, n),
                  static_cast<double>(ortho_tol()))
            << "thin V orthogonality m=" << m << " n=" << n;
    }

    std::vector<Scalar> random_matrix(int m, int n, int batch, unsigned seed) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<double> d(-1.0, 1.0);
        std::vector<Scalar> v(static_cast<size_t>(m) * n * batch);
        for (auto& x : v) {
            if constexpr (is_cplx_v<Scalar>) {
                x = Scalar(static_cast<Real>(d(rng)), static_cast<Real>(d(rng)));
            } else {
                x = static_cast<Scalar>(d(rng));
            }
        }
        return v;
    }
};

TYPED_TEST_SUITE(GesvdjCtaTest, GesvdjTestTypes);

TYPED_TEST(GesvdjCtaTest, SquareRandom) {
    for (int n : {2, 5, 8, 16, 32}) {
        this->check(n, n, 3, this->random_matrix(n, n, 3, 1234u + n));
    }
}

TYPED_TEST(GesvdjCtaTest, TallRectangular) {
    this->check(32, 8, 3, this->random_matrix(32, 8, 3, 77u));
    this->check(16, 5, 2, this->random_matrix(16, 5, 2, 78u));
}

TYPED_TEST(GesvdjCtaTest, WideRectangular) {
    this->check(8, 32, 3, this->random_matrix(8, 32, 3, 79u));
    this->check(5, 16, 2, this->random_matrix(5, 16, 2, 80u));
}

// Rank deficiency: U's columns beyond the rank are not determined by A and must
// be completed from the orthogonal complement. This is the least-exercised path
// in the kernel -- it is off the square full-rank benchmark shape entirely.
TYPED_TEST(GesvdjCtaTest, RankDeficient) {
    using Scalar = typename TestFixture::Scalar;
    const int n = 16, batch = 2;

    // Rank 1: every column a multiple of the first.
    auto base = this->random_matrix(n, 1, batch, 11u);
    std::vector<Scalar> A(static_cast<size_t>(n) * n * batch, Scalar(0));
    for (int b = 0; b < batch; ++b) {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                A[static_cast<size_t>(b) * n * n + static_cast<size_t>(j) * n + i] =
                    base[static_cast<size_t>(b) * n + i] * static_cast<Scalar>(static_cast<typename TestFixture::Real>(j + 1));
            }
        }
    }
    this->check(n, n, batch, A);

    // Half the columns exactly zero.
    auto Ah = this->random_matrix(n, n, batch, 12u);
    for (int b = 0; b < batch; ++b) {
        for (int j = n / 2; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                Ah[static_cast<size_t>(b) * n * n + static_cast<size_t>(j) * n + i] = Scalar(0);
            }
        }
    }
    this->check(n, n, batch, Ah);
}

TYPED_TEST(GesvdjCtaTest, AllZeros) {
    using Scalar = typename TestFixture::Scalar;
    const int n = 8, batch = 2;
    std::vector<Scalar> A(static_cast<size_t>(n) * n * batch, Scalar(0));
    this->check(n, n, batch, A);
}

// A graded matrix is where the relative-accuracy argument actually bites: the
// tridiagonalizing path loses the small singular values entirely here.
TYPED_TEST(GesvdjCtaTest, GradedColumns) {
    using Scalar = typename TestFixture::Scalar;
    using Real = typename TestFixture::Real;
    const int n = 16, batch = 2;
    auto A = this->random_matrix(n, n, batch, 4242u);
    const double decade = std::is_same_v<Real, float> ? 0.4 : 0.8;
    for (int b = 0; b < batch; ++b) {
        for (int j = 0; j < n; ++j) {
            const auto scale = static_cast<Real>(std::pow(10.0, -decade * j));
            for (int i = 0; i < n; ++i) {
                A[static_cast<size_t>(b) * n * n + static_cast<size_t>(j) * n + i] *= static_cast<Scalar>(scale);
            }
        }
    }
    this->check(n, n, batch, A);
}

TYPED_TEST(GesvdjCtaTest, JobCombinations) {
    using Scalar = typename TestFixture::Scalar;
    using Real = typename TestFixture::Real;
    constexpr Backend B = TestFixture::B;
    auto& ctx = *this->ctx;

    const int n = 12, batch = 3;
    auto host = this->random_matrix(n, n, batch, 900u);

    // Values from the full-vector call are the reference.
    UnifiedVector<Real> s_ref(static_cast<size_t>(n) * batch);
    {
        Matrix<Scalar> A(n, n, batch);
        std::copy(host.begin(), host.end(), A.view().data_ptr());
        Matrix<Scalar> U(n, n, batch), Vh(n, n, batch);
        gesvdj_cta<B, Scalar>(ctx, A.view(), s_ref.to_span(), U.view(), Vh.view(),
                              SvdVectors::All, SvdVectors::All);
        ctx.wait_and_throw();
    }

    for (int ju = 0; ju < 2; ++ju) {
        for (int jv = 0; jv < 2; ++jv) {
            Matrix<Scalar> A(n, n, batch);
            std::copy(host.begin(), host.end(), A.view().data_ptr());
            Matrix<Scalar> U(n, n, batch), Vh(n, n, batch);
            UnifiedVector<Real> s(static_cast<size_t>(n) * batch);
            gesvdj_cta<B, Scalar>(ctx, A.view(), s.to_span(), U.view(), Vh.view(),
                                  ju ? SvdVectors::All : SvdVectors::None,
                                  jv ? SvdVectors::All : SvdVectors::None);
            ctx.wait_and_throw();
            for (size_t i = 0; i < s.size(); ++i) {
                const Real scale = std::max<Real>(Real(1), std::abs(s_ref[i]));
                EXPECT_NEAR(s[i], s_ref[i], TestFixture::recon_tol() * scale)
                    << "jobu=" << ju << " jobvh=" << jv << " i=" << i;
            }
        }
    }
}


// Routing. Two things are checked here, both behavioural:
//  1. BATCHLAS_GESVD_PROVIDER=jacobi actually reaches gesvdj_cta. A forced
//     provider that is unsupported degrades to Auto SILENTLY, so "it ran" is not
//     evidence that the right thing ran -- the accuracy is.
//  2. Complex GENERAL input through the public gesvd() no longer throws. Before
//     this kernel, gesvd_supports_cta and gesvd_supports_blocked both returned
//     false for complex outside the Hermitian branch and dispatch fell through
//     to Vendor, which throws.
TYPED_TEST(GesvdjCtaTest, PublicApiDispatch) {
    using Scalar = typename TestFixture::Scalar;
    using Real = typename TestFixture::Real;
    constexpr Backend B = TestFixture::B;
    auto& ctx = *this->ctx;

    const int n = 12, batch = 2;
    auto host = this->random_matrix(n, n, batch, 31337u);

    Matrix<Scalar> A(n, n, batch);
    std::copy(host.begin(), host.end(), A.view().data_ptr());
    Matrix<Scalar> U(n, n, batch), Vh(n, n, batch);
    UnifiedVector<Real> s(static_cast<size_t>(n) * batch);

    const size_t ws_bytes = gesvd_buffer_size<B, Scalar>(
        ctx, A.view(), s.to_span(), U.view(), Vh.view(), SvdVectors::All, SvdVectors::All);
    UnifiedVector<std::byte> ws(ws_bytes);
    gesvd<B, Scalar>(ctx, A.view(), s.to_span(), U.view(), Vh.view(),
                     SvdVectors::All, SvdVectors::All, ws.to_span());
    ctx.wait_and_throw();

    EXPECT_LE(TestFixture::reconstruction(n, n, batch, host,
                                          U.view().data_ptr(), U.view().stride(), static_cast<int>(U.view().ld()),
                                          Vh.view().data_ptr(), Vh.view().stride(), static_cast<int>(Vh.view().ld()),
                                          s.data()),
              // Loose because the point of this test is that it dispatches and
              // reconstructs at all, not how well. (It used to reach the
              // normal-equations CTA path for real T; Auto now routes n <= 32
              // to gesvdj_cta, so the achieved value is far below this.)
              0.3);
}

// ---------------------------------------------------------------------------
// Thin (economy) vectors.
// ---------------------------------------------------------------------------

TYPED_TEST(GesvdjCtaTest, ThinTallRectangular) {
    // m > n: U is the genuinely thin factor (32 x 8 instead of 32 x 32), and
    // the in-kernel Gram-Schmidt that manufactures the other 24 columns is
    // skipped entirely.
    this->check_thin(32, 8, 3, this->random_matrix(32, 8, 3, 20241u));
}

TYPED_TEST(GesvdjCtaTest, ThinWideRectangular) {
    // m < n: the kernel solves A^H, so the thin factor is written through the
    // TRANSPOSED writeback, where the bound applies to `lane` (the rank index)
    // rather than to the inner loop. Truncating the wrong one still produces
    // plausible-looking output, so this shape is not redundant with the tall one.
    this->check_thin(8, 32, 3, this->random_matrix(8, 32, 3, 20242u));
}

TYPED_TEST(GesvdjCtaTest, ThinSquareEqualsAll) {
    // Square input: Thin and All request the same shapes, so this must behave
    // identically to check() -- it is the canonicalisation path.
    this->check_thin(16, 16, 3, this->random_matrix(16, 16, 3, 20243u));
}

TYPED_TEST(GesvdjCtaTest, ThinRankDeficientStillRepairsColumns) {
    // A numerically deficient column must still be rebuilt even when no
    // completion columns are wanted: the gate is `any_def > 0 || LC > CC`, and
    // dropping the first disjunct would leave a zero column in a thin U that
    // still has to be orthonormal.
    const int m = 32, n = 8, batch = 2;
    auto host = this->random_matrix(m, n, batch, 20244u);
    // Force rank 6 of 8: make columns 6 and 7 copies of column 0.
    for (int b = 0; b < batch; ++b) {
        for (int dup : {6, 7}) {
            for (int i = 0; i < m; ++i) {
                host[static_cast<size_t>(b) * m * n + static_cast<size_t>(dup) * m + i] =
                    host[static_cast<size_t>(b) * m * n + static_cast<size_t>(0) * m + i];
            }
        }
    }
    this->check_thin(m, n, batch, std::move(host));
}

TYPED_TEST(GesvdjCtaTest, ThinMatchesFullLeadingColumns) {
    // The thin U must be the first k columns of the full U, up to a per-column
    // sign (real) or unit phase (complex). This is what makes Thin a genuine
    // economy mode rather than a differently-computed answer.
    using Scalar = typename TestFixture::Scalar;
    using Real = typename TestFixture::Real;
    constexpr Backend B = TestFixture::B;
    auto& ctx = *this->ctx;

    const int m = 32, n = 8, batch = 2;
    const int k = std::min(m, n);
    auto host = this->random_matrix(m, n, batch, 20245u);

    auto load = [&](Matrix<Scalar>& A) {
        for (int b = 0; b < batch; ++b)
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < m; ++i)
                    A.view().data_ptr()[b * A.view().stride() + static_cast<size_t>(j) * A.view().ld() + i] =
                        host[static_cast<size_t>(b) * m * n + static_cast<size_t>(j) * m + i];
    };

    Matrix<Scalar> A_all(m, n, batch);
    load(A_all);
    Matrix<Scalar> U_all(m, m, batch), Vh_all(n, n, batch);
    UnifiedVector<Real> s_all(static_cast<size_t>(k) * batch);
    gesvdj_cta<B, Scalar>(ctx, A_all.view(), s_all.to_span(), U_all.view(), Vh_all.view(),
                          SvdVectors::All, SvdVectors::All);
    ctx.wait_and_throw();

    Matrix<Scalar> A_thin(m, n, batch);
    load(A_thin);
    Matrix<Scalar> U_thin(m, k, batch), Vh_thin(k, n, batch);
    UnifiedVector<Real> s_thin(static_cast<size_t>(k) * batch);
    gesvdj_cta<B, Scalar>(ctx, A_thin.view(), s_thin.to_span(), U_thin.view(), Vh_thin.view(),
                          SvdVectors::Thin, SvdVectors::Thin);
    ctx.wait_and_throw();

    const double tol = static_cast<double>(TestFixture::ortho_tol());
    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < k; ++i) {
            EXPECT_NEAR(static_cast<double>(s_thin[b * k + i]),
                        static_cast<double>(s_all[b * k + i]), tol)
                << "sigma mismatch b=" << b << " i=" << i;
        }
        for (int c = 0; c < k; ++c) {
            // Compare |<u_thin, u_all>| == 1 rather than entrywise: the sign or
            // phase of a singular vector is not determined.
            std::complex<double> acc(0.0, 0.0);
            for (int i = 0; i < m; ++i) {
                acc += std::conj(TestFixture::to_cd(
                           U_thin.view().data_ptr()[b * U_thin.view().stride() + static_cast<size_t>(c) * U_thin.view().ld() + i]))
                     * TestFixture::to_cd(
                           U_all.view().data_ptr()[b * U_all.view().stride() + static_cast<size_t>(c) * U_all.view().ld() + i]);
            }
            EXPECT_NEAR(std::abs(acc), 1.0, 1e-3) << "U column " << c << " differs at b=" << b;
        }
    }
}

} // namespace
