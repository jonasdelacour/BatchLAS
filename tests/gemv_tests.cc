#include <gtest/gtest.h>
#include <batchlas/blas/linalg.hh>
#include <batchlas/blas/matrix.hh>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <type_traits>
#include <limits>
#include <algorithm>
#include "test_utils.hh"

using namespace batchlas;

template <typename T, Backend B>
struct TestConfig {
    using ScalarType = T;
    static constexpr Backend BackendVal = B;
};

using MyTypes = typename test_utils::backend_types<TestConfig>::type;

template <typename Config>
class GemvMatrixViewTest : public test_utils::BatchLASTest<Config> {
protected:
    using ScalarType = typename Config::ScalarType;
    static constexpr Backend BackendType = Config::BackendVal;
    
    const int rows = 10; 
    const int cols = 10;
    const int batch_size = 5;
    UnifiedVector<ScalarType> A_data;
    UnifiedVector<ScalarType> x_data; 
    UnifiedVector<ScalarType> y_data; 
    UnifiedVector<ScalarType> y_expected;

    void SetUp() override {
        test_utils::BatchLASTest<Config>::SetUp();
        
        if (!this->ctx) {
            return;
        }

        // Initialize test matrices and vectors
        this->A_data = UnifiedVector<ScalarType>(this->rows * this->cols * this->batch_size);
        this->x_data = UnifiedVector<ScalarType>(std::max(this->rows, this->cols) * this->batch_size);
        this->y_data = UnifiedVector<ScalarType>(this->rows * this->batch_size, static_cast<ScalarType>(0.0));
        this->y_expected = UnifiedVector<ScalarType>(this->rows * this->batch_size, static_cast<ScalarType>(0.0));

        // Initialize matrix with deterministic values (Column Major for BLAS)
        for (int b = 0; b < this->batch_size; ++b) {
            for (int j = 0; j < this->cols; ++j) {
                for (int i = 0; i < this->rows; ++i) {
                    this->A_data[b * this->rows * this->cols + j * this->rows + i] = 
                        static_cast<ScalarType>(i + j * this->rows + 1 + b * 100);
                }
            }
        }

        // Initialize x vector with sequential values
        for (int b = 0; b < this->batch_size; ++b) {
            int vec_dim = this->cols;
            for (int j = 0; j < vec_dim; ++j) {
                this->x_data[b * std::max(this->rows, this->cols) + j] = static_cast<ScalarType>(j + 1 + b * 10);
            }
        }
        
        // Initialize y vector
        for (int b = 0; b < this->batch_size; ++b) {
            for (int i = 0; i < this->rows; ++i) {
                this->y_data[b * this->rows + i] = static_cast<ScalarType>(0.0);
            }
        }
    }

    // Helper function to compute expected y = alpha*A*x + beta*y
    void computeExpectedGemv(ScalarType alpha, ScalarType beta, Transpose transA) {
        for (int b = 0; b < this->batch_size; ++b) {
            const ScalarType* A_batch = this->A_data.data() + b * this->rows * this->cols;
            const ScalarType* x_batch = this->x_data.data() + b * std::max(this->rows, this->cols);
            ScalarType* y_batch_expected = this->y_expected.data() + b * this->rows;
            const ScalarType* y_batch_initial = this->y_data.data() + b * this->rows; // Initial y for beta calculation

            if (transA == Transpose::NoTrans) {
                for (int i = 0; i < this->rows; ++i) {
                    ScalarType sum = static_cast<ScalarType>(0.0);
                    for (int j = 0; j < this->cols; ++j) {
                        sum += A_batch[i + j * this->rows] * x_batch[j];
                    }
                    y_batch_expected[i] = alpha * sum + beta * y_batch_initial[i];
                }
            } else { 
                if (this->rows != this->cols) {
                    GTEST_SKIP() << "Transpose test skipped for non-square matrix in this fixture setup.";
                    return;
                }

                for (int j = 0; j < this->cols; ++j) { 
                    ScalarType sum = static_cast<ScalarType>(0.0);
                    for (int i = 0; i < this->rows; ++i) { 
                        sum += A_batch[i + j * this->rows] * x_batch[i]; 
                    }
                    y_batch_expected[j] = alpha * sum + beta * y_batch_initial[j];
                }
            }
        }
    }

    typename base_type<ScalarType>::type get_tolerance() {
        return test_utils::tolerance<ScalarType>();
    }
    
    typename base_type<ScalarType>::type get_rel_error_floor() {
        if constexpr (std::is_same_v<ScalarType, float>) {
            return 1e-6f;
        } else {
            return 1e-9;
        }
    }
};

TYPED_TEST_SUITE(GemvMatrixViewTest, MyTypes);

// Test single GEMV operation with no transpose using MatrixView
TYPED_TEST(GemvMatrixViewTest, SingleGemvNoTranspose) {
    using ScalarType = typename TestFixture::ScalarType;
    constexpr Backend BackendType = TestFixture::BackendType;

    MatrixView<ScalarType, MatrixFormat::Dense> A_view(this->A_data.data(), this->rows, this->cols, this->rows, 0); 
    VectorView<ScalarType> x_vec(this->x_data.data(), this->cols, 1); 
    VectorView<ScalarType> y_vec(this->y_data.data(), this->rows, 1); 

    ScalarType alpha = static_cast<ScalarType>(1.0);
    ScalarType beta = static_cast<ScalarType>(0.0);

    this->computeExpectedGemv(alpha, beta, Transpose::NoTrans); 

    gemv(*(this->ctx), A_view, x_vec, y_vec, {.alpha = alpha, .beta = beta});

    this->ctx->wait();
    
    auto tol = this->get_tolerance();
    for (int i = 0; i < this->rows; ++i) {
        EXPECT_NEAR(std::real(this->y_data[i]), std::real(this->y_expected[i]), tol) 
            << "Mismatch at index " << i;
    }
}

// Test single GEMV operation with transpose using MatrixView
TYPED_TEST(GemvMatrixViewTest, SingleGemvWithTranspose) {
    using ScalarType = typename TestFixture::ScalarType;
    constexpr Backend BackendType = TestFixture::BackendType;

    ASSERT_EQ(this->rows, this->cols) << "Transpose test requires square matrix in this fixture setup.";

    MatrixView<ScalarType, MatrixFormat::Dense> A_view(this->A_data.data(), this->rows, this->cols, this->rows, 0); 
    VectorView<ScalarType> x_vec(this->x_data.data(), this->rows, 1);
    VectorView<ScalarType> y_vec(this->y_data.data(), this->cols, 1);

    ScalarType alpha = static_cast<ScalarType>(2.0);
    ScalarType beta = static_cast<ScalarType>(0.0);

    this->computeExpectedGemv(alpha, beta, Transpose::Trans); 

    gemv(*(this->ctx),
                      A_view,
                      x_vec,
                      y_vec,
                      {.alpha = alpha, .beta = beta, .transA = Transpose::Trans});

    this->ctx->wait();

    auto tol = this->get_tolerance();
    for (int i = 0; i < this->cols; ++i) {
        EXPECT_NEAR(std::real(this->y_data[i]), std::real(this->y_expected[i]), tol)
        << "Mismatch with transpose at index " << i;
    }
}


// Test batched GEMV operation using MatrixView
TYPED_TEST(GemvMatrixViewTest, BatchedGemvNoTranspose) {
    using ScalarType = typename TestFixture::ScalarType;
    constexpr Backend BackendType = TestFixture::BackendType;

    MatrixView<ScalarType, MatrixFormat::Dense> A_view(this->A_data.data(), this->rows, this->cols, this->rows, 
                                                this->rows * this->cols, this->batch_size); 
    VectorView<ScalarType> x_vec(this->x_data.data(), this->cols, this->batch_size); 
    VectorView<ScalarType> y_vec(this->y_data.data(), this->rows, this->batch_size); 

    ScalarType alpha = static_cast<ScalarType>(1.0);
    ScalarType beta = static_cast<ScalarType>(0.0);

    this->computeExpectedGemv(alpha, beta, Transpose::NoTrans);

    gemv(*(this->ctx), A_view, x_vec, y_vec, {.alpha = alpha, .beta = beta});

    this->ctx->wait();

    auto tol = this->get_tolerance();
    auto floor_val = this->get_rel_error_floor();
    for (int b = 0; b < this->batch_size; ++b) {
        for (int i = 0; i < this->rows; ++i) {
            auto rel_error = std::abs(this->y_data[b * this->rows + i] - this->y_expected[b * this->rows + i]) / std::max(std::abs(this->y_expected[b * this->rows + i]), floor_val);
            EXPECT_NEAR(rel_error, static_cast<typename base_type<ScalarType>::type>(0.0), tol)
                << "Mismatch at batch " << b << ", index " << i;
        }
    }
}

// Test batched GEMV operation with transpose using MatrixView
TYPED_TEST(GemvMatrixViewTest, BatchedGemvWithTranspose) {
    using ScalarType = typename TestFixture::ScalarType;
    constexpr Backend BackendType = TestFixture::BackendType;

    ASSERT_EQ(this->rows, this->cols) << "Transpose test requires square matrix in this fixture setup.";

    MatrixView<ScalarType, MatrixFormat::Dense> A_view(this->A_data.data(), this->rows, this->cols, this->rows,
                                                this->rows * this->cols, this->batch_size); 
    VectorView<ScalarType> x_vec(this->x_data.data(), this->rows, this->batch_size);
    VectorView<ScalarType> y_vec(this->y_data.data(), this->cols, this->batch_size);

    ScalarType alpha = static_cast<ScalarType>(2.5);
    ScalarType beta = static_cast<ScalarType>(0.0);

    this->computeExpectedGemv(alpha, beta, Transpose::Trans);

    gemv(*(this->ctx),
                      A_view,
                      x_vec,
                      y_vec,
                      {.alpha = alpha, .beta = beta, .transA = Transpose::Trans});

    this->ctx->wait();

    auto tol = this->get_tolerance();
    auto floor_val = this->get_rel_error_floor();
    for (int b = 0; b < this->batch_size; ++b) {
        for (int i = 0; i < this->cols; ++i) { 
            auto rel_error = std::abs(this->y_data[b * this->cols + i] - this->y_expected[b * this->cols + i]) / std::max(std::abs(this->y_expected[b * this->cols + i]), floor_val);
            EXPECT_NEAR(rel_error, static_cast<typename base_type<ScalarType>::type>(0.0), tol)
                << "Mismatch with transpose at batch " << b << ", index " << i;
        }
    }
}

// Test both alpha and beta in batched GEMV
TYPED_TEST(GemvMatrixViewTest, BatchedGemvWithAlphaBeta) {
    using ScalarType = typename TestFixture::ScalarType;
    constexpr Backend BackendType = TestFixture::BackendType;

    MatrixView<ScalarType, MatrixFormat::Dense> A_view(this->A_data.data(), this->rows, this->cols, this->rows, 
                                                this->rows * this->cols, this->batch_size); 
    VectorView<ScalarType> x_vec(this->x_data.data(), this->cols, this->batch_size);
    VectorView<ScalarType> y_vec(this->y_data.data(), this->rows, this->batch_size);

    for (int b = 0; b < this->batch_size; ++b) {
        for (int i = 0; i < this->rows; ++i) {
            this->y_data[b * this->rows + i] = static_cast<ScalarType>(b * 1.0 + i * 0.1);
        }
    }
     this->y_expected = this->y_data; 

    ScalarType alpha = static_cast<ScalarType>(1.5);
    ScalarType beta = static_cast<ScalarType>(0.8);

    this->computeExpectedGemv(alpha, beta, Transpose::NoTrans); 

    gemv(*(this->ctx), A_view, x_vec, y_vec, {.alpha = alpha, .beta = beta});

    this->ctx->wait();

    auto tol = this->get_tolerance();
    auto floor_val = this->get_rel_error_floor();
    for (int b = 0; b < this->batch_size; ++b) {
        for (int i = 0; i < this->rows; ++i) {
            auto rel_error = std::abs(this->y_data[b * this->rows + i] - this->y_expected[b * this->rows + i]) / std::max(std::abs(this->y_expected[b * this->rows + i]), floor_val);
            EXPECT_NEAR(rel_error, static_cast<typename base_type<ScalarType>::type>(0.0), tol)
                << "Mismatch with alpha/beta at batch " << b << ", index " << i;
        }
    }
}



// ===========================================================================
// WP7 -- THE COVERAGE FIXTURE.
//
// WHY A SECOND FIXTURE EXISTS, stated as a list of things the one above cannot
// see. GemvMatrixViewTest is FIXED at 10x10, batch 5, ld == rows, inc == 1,
// SQUARE only, and:
//
//   * Transpose::ConjTrans is NEVER used -- only NoTrans and Trans. ConjTrans
//     is the LIVE PRODUCTION PATH (ortho.cc selects it for all four complex
//     types) and had zero test coverage in this tree.
//   * beta != 0 appears in exactly ONE test, NoTrans only.
//   * THE COMPLEX TESTS USE PURELY REAL DATA: every element is
//     static_cast<ScalarType>(i + j*rows + 1 + b*100), so every imaginary part
//     is identically zero and every imaginary cross-term of every product is
//     zero. A complex kernel that DROPPED the cross-terms entirely, or that got
//     ConjTrans backwards, passes all 40 of those tests.
//   * SingleGemvNoTranspose and SingleGemvWithTranspose compare only
//     std::real(...), so half of every complex answer is unexamined.
//   * ld == rows always, inc == 1 always, so a kernel that ignored either would
//     pass.
//
// Every test below is accompanied by a BREAK -- an actual edit to
// src/sycl/gemv_native.cc, actually applied, rebuilt and run in build-novendor
// (where preferred() being all-false does not matter, because there is no
// vendor to fall back to) -- that turns it RED. A test whose break was not RUN
// does not count; the outputs are in the WP7 report.
// ===========================================================================

namespace {

// Deterministic, reproducible, and -- for complex -- WITH A GENUINELY NON-ZERO
// IMAGINARY PART, which is the single property the fixture above lacks. The
// magnitudes stay in [-1, 1] so a reduction of ~100 terms cannot lose the
// relative tolerance to cancellation.
template <typename T>
T gemv_cov_value(int seed) {
    using R = typename batchlas::base_type<T>::type;
    const R re = static_cast<R>(std::sin(0.7 * seed + 0.3));
    const R im = static_cast<R>(std::cos(1.3 * seed + 1.1));
    if constexpr (test_utils::is_complex<T>::value) {
        return T(re, im);
    } else {
        (void)im;
        return re;
    }
}

}  // namespace

template <typename Config>
class GemvCoverageTest : public test_utils::BatchLASTest<Config> {
protected:
    using ScalarType = typename Config::ScalarType;
    using RealType = typename batchlas::base_type<ScalarType>::type;
    static constexpr Backend BackendType = Config::BackendVal;

    struct Case {
        int m = 0;
        int n = 0;
        int batch = 1;
        int ld = 0;          // >= m; the padded case is what catches an ld-blind kernel
        int xinc = 1;
        int yinc = 1;
        Transpose transA = Transpose::NoTrans;
        ScalarType alpha = ScalarType(1);
        ScalarType beta = ScalarType(0);
        bool y_starts_nan = false;   // only meaningful with beta == 0
        // Poisons the LIVE elements of A with NaN. Only meaningful with
        // alpha == 0, where reference ?GEMV never reads A at all -- it is
        // what makes "alpha == 0 does not touch A" an OBSERVABLE claim
        // rather than an arithmetic identity that no break can move.
        bool a_starts_nan = false;
        // Pushes every batch stride PAST its natural value, so that
        // stride != ld*cols for A and stride != size*inc for x and y.
        //
        // Without this, every case in this suite used the natural stride, and a
        // kernel that DERIVED the batch stride instead of reading it from the
        // view passed all 232 of them. That is not hypothetical: ortho.cc:218-222
        // is the live caller and hands gemv A.stride() == m*A.cols() against a
        // view whose ld*cols is m*i, on every CGS iteration -- so the only guard
        // on stride handling was ortho_tests, a different suite, by accident.
        // The pre-existing `stride` break (b*stride_a -> 0) is strictly weaker:
        // it makes every batch item read matrix 0, which the natural-stride
        // cases already catch.
        int stride_pad = 0;
    };

    // Runs one case through the PUBLIC gemv and checks every element of every
    // batch item against an independent host reference.
    //
    // The reference is written from the BLAS definition, not transcribed from
    // either backend: netlib_lapack.cc and cublas.cc both fold transA the same
    // way, so a reference copied from either proves only that the copy matches.
    void run_case(const Case& c) {
        if (!this->ctx) return;

        const int red = (c.transA == Transpose::NoTrans) ? c.n : c.m;
        const int out = (c.transA == Transpose::NoTrans) ? c.m : c.n;
        const int ld = c.ld > 0 ? c.ld : c.m;
        ASSERT_GE(ld, c.m);

        const int a_stride = ld * c.n + c.stride_pad;
        const int x_stride = red * c.xinc + c.stride_pad;
        const int y_stride = out * c.yinc + c.stride_pad;

        UnifiedVector<ScalarType> A(std::max(1, a_stride * c.batch));
        UnifiedVector<ScalarType> x(std::max(1, x_stride * c.batch));
        UnifiedVector<ScalarType> y(std::max(1, y_stride * c.batch));

        // A's PAD ELEMENTS (rows m..ld-1) are filled with a large poison value.
        // A kernel that walked a column by ld instead of by m -- or that mixed
        // the pad into a reduction -- lands on them and cannot come back with a
        // right answer by luck.
        const ScalarType poison = static_cast<ScalarType>(RealType(1e3));
        for (int b = 0; b < c.batch; ++b) {
            for (int j = 0; j < c.n; ++j) {
                for (int i = 0; i < ld; ++i) {
                    const ScalarType live =
                        c.a_starts_nan
                            ? static_cast<ScalarType>(
                                  std::numeric_limits<RealType>::quiet_NaN())
                            : gemv_cov_value<ScalarType>(b * 7919 + j * 131 + i);
                    A[b * a_stride + j * ld + i] = (i < c.m) ? live : poison;
                }
            }
            // The stride pad itself. The loop above stops at ld*n, so without
            // this the padded tail would be uninitialised rather than poisoned,
            // and a kernel that walked into it could still come back correct.
            for (int t = ld * c.n; t < a_stride; ++t) A[b * a_stride + t] = poison;
        }
        // The x GAPS (the xinc-1 slots between live elements) are poisoned for
        // the same reason: a kernel that ignored inc reads them.
        for (int b = 0; b < c.batch; ++b) {
            for (int t = 0; t < x_stride; ++t) x[b * x_stride + t] = poison;
            for (int r = 0; r < red; ++r) {
                x[b * x_stride + r * c.xinc] =
                    gemv_cov_value<ScalarType>(b * 3571 + r * 17 + 5);
            }
        }

        std::vector<ScalarType> y_initial(static_cast<size_t>(std::max(1, y_stride * c.batch)));
        const ScalarType nan_v =
            c.y_starts_nan ? static_cast<ScalarType>(std::numeric_limits<RealType>::quiet_NaN())
                           : ScalarType(0);
        for (int b = 0; b < c.batch; ++b) {
            for (int t = 0; t < y_stride; ++t) {
                const ScalarType v =
                    c.y_starts_nan ? nan_v : gemv_cov_value<ScalarType>(b * 6151 + t * 29 + 11);
                y[b * y_stride + t] = v;
                y_initial[b * y_stride + t] = v;
            }
        }

        MatrixView<ScalarType, MatrixFormat::Dense> A_view(
            A.data(), c.m, c.n, ld, a_stride, c.batch);
        VectorView<ScalarType> x_vec(x.data(), red, c.batch, c.xinc, x_stride);
        VectorView<ScalarType> y_vec(y.data(), out, c.batch, c.yinc, y_stride);

        // CAMPAIGN TRAP 5, asserted rather than trusted. VectorView takes
        // (data, size, batch_size, inc, stride) while Vector takes
        // (size, batch_size, stride, inc) -- positions 3 and 4 are SWAPPED,
        // both are plain int, and (inc=n, stride=1) fits the same buffer as
        // (inc=1, stride=n), so the span-length assert cannot tell them apart.
        ASSERT_EQ(x_vec.inc(), c.xinc);
        ASSERT_EQ(x_vec.stride(), x_stride);
        ASSERT_EQ(y_vec.inc(), c.yinc);
        ASSERT_EQ(y_vec.stride(), y_stride);
        ASSERT_EQ(A_view.ld(), ld);
        ASSERT_EQ(A_view.stride(), a_stride);

        gemv(*(this->ctx), A_view, x_vec, y_vec,
             {.alpha = c.alpha, .beta = c.beta, .transA = c.transA});
        this->ctx->wait();

        const RealType tol = test_utils::tolerance<ScalarType>();
        for (int b = 0; b < c.batch; ++b) {
            for (int o = 0; o < out; ++o) {
                ScalarType sum = ScalarType(0);
                // ALPHA == 0 NEVER READS A, in the reference as in the kernel.
                // Reference ?GEMV scales y by beta and returns before touching
                // A, so a reference that summed a NaN-filled A here would
                // predict NaN and the test would be checking nothing.
                const bool skip_a = (c.alpha == ScalarType(0));
                // The BACKWARD-ERROR DENOMINATOR, accumulated alongside the
                // sum. Comparing against |expected| alone is not a tolerance,
                // it is a cancellation detector: at m = 97, float, terms of
                // O(1) and a result of O(0.5), the exact sum of magnitudes is
                // ~50 and a correct float reduction in a DIFFERENT ORDER
                // legitimately lands 1.2e-5 away -- which failed a 1e-5 check
                // on |expected| and said nothing about the kernel. The standard
                // BLAS bound is relative to sum|a_r||x_r|, so that is what is
                // used, floored at 1 so a tiny well-conditioned answer is still
                // held to an absolute 1e-5.
                RealType absum = RealType(0);
                for (int r = 0; skip_a ? false : (r < red); ++r) {
                    // op(A)(o, r): the stored element and, for ConjTrans, its
                    // conjugate. Getting this backwards is the classic silent
                    // complex gemv bug and the fixture above cannot see it.
                    ScalarType a;
                    if (c.transA == Transpose::NoTrans) {
                        a = A[b * a_stride + r * ld + o];
                    } else {
                        a = A[b * a_stride + o * ld + r];
                        // std::conj on a real scalar returns std::complex,
                        // so the branch has to be compile-time.
                        if constexpr (test_utils::is_complex<ScalarType>::value) {
                            if (c.transA == Transpose::ConjTrans) a = std::conj(a);
                        }
                    }
                    const ScalarType xr = x[b * x_stride + r * c.xinc];
                    sum += a * xr;
                    absum += std::abs(a) * std::abs(xr);
                }
                ScalarType expected = c.alpha * sum;
                RealType scale = std::abs(c.alpha) * absum;
                if (c.beta != ScalarType(0)) {
                    const ScalarType y0 = y_initial[b * y_stride + o * c.yinc];
                    expected += c.beta * y0;
                    scale += std::abs(c.beta) * std::abs(y0);
                }
                const ScalarType got = y[b * y_stride + o * c.yinc];
                const RealType denom =
                    std::max(std::max(std::abs(expected), scale), RealType(1));
                EXPECT_LE(std::abs(got - expected) / denom, tol)
                    << "batch " << b << " out " << o
                    << " got " << got << " expected " << expected;
            }

            // THE GAPS BETWEEN y's LIVE ELEMENTS MUST BE UNTOUCHED. A kernel
            // that ignored yinc writes into them, and every check above would
            // still pass for yinc == 1.
            for (int t = 0; t < y_stride; ++t) {
                if (c.yinc > 1 && (t % c.yinc) != 0) {
                    const ScalarType before = y_initial[b * y_stride + t];
                    const ScalarType after = y[b * y_stride + t];
                    if (c.y_starts_nan) {
                        EXPECT_TRUE(std::isnan(std::real(after)))
                            << "y gap at " << t << " of batch " << b << " was written";
                    } else {
                        EXPECT_EQ(after, before)
                            << "y gap at " << t << " of batch " << b << " was written";
                    }
                }
            }
        }
    }
};

TYPED_TEST_SUITE(GemvCoverageTest, MyTypes);

// --- the three transA arms, on non-square, ld-padded, inc-strided data ------
//
// m = 70, n = 48 so the two extents differ and cannot be swapped unnoticed;
// ld = 79 so the pad is live; xinc = 2 and yinc = 3 so the two strides differ
// from each other as well as from 1; batch = 6 so a wrong per-item stride shows
// up. beta != 0 on all three, which the fixture above tests only for NoTrans.
//
// m = 70 is deliberately NOT a multiple of 32: the CTA body's lanes stride the
// reduction by 32, so 70 = 2*32 + 6 exercises the partial final round and the
// shuffle ladder folding lanes that contributed nothing.

TYPED_TEST(GemvCoverageTest, NoTransposePaddedStrided) {
    using S = typename TestFixture::ScalarType;
    typename TestFixture::Case c;
    c.m = 70; c.n = 48; c.batch = 6; c.ld = 79; c.xinc = 2; c.yinc = 3;
    c.transA = Transpose::NoTrans;
    c.alpha = static_cast<S>(1.5); c.beta = static_cast<S>(-0.75);
    this->run_case(c);
}

TYPED_TEST(GemvCoverageTest, TransposePaddedStrided) {
    using S = typename TestFixture::ScalarType;
    typename TestFixture::Case c;
    c.m = 70; c.n = 48; c.batch = 6; c.ld = 79; c.xinc = 2; c.yinc = 3;
    c.transA = Transpose::Trans;
    c.alpha = static_cast<S>(1.5); c.beta = static_cast<S>(-0.75);
    this->run_case(c);
}

// THE LIVE PRODUCTION PATH THAT HAD NO TEST AT ALL. ortho.cc selects
// Transpose::ConjTrans for all four complex types; nothing in this tree
// exercised it before WP7. For a real scalar ConjTrans is Trans, and running it
// on the real types too is the cheapest way to catch a kernel that conjugated
// something it should not have.
//
// THE DIRECTION IS TESTED IN BOTH SENSES, because one break can only move one
// of them:
//   * break `conj` (ConjTrans stops conjugating) -> 12 failures, exactly the
//     three ConjTrans cases on the four COMPLEX suites. The float and double
//     suites stay green, which is the evidence that ConjTrans == Trans on a
//     real scalar is what the file asks for and that no test demands the
//     conjugation of a real value.
//   * break `conjalways` (plain Trans conjugates too) -> 20 failures, exactly
//     the five plain-Trans cases on the four complex suites, with the ConjTrans
//     cases and both real types green.
// Both breaks leave ALL 40 of the pre-WP7 GemvMatrixViewTest cases green, in
// either direction, because that fixture's complex data has an identically zero
// imaginary part.
TYPED_TEST(GemvCoverageTest, ConjTransposePaddedStrided) {
    using S = typename TestFixture::ScalarType;
    typename TestFixture::Case c;
    c.m = 70; c.n = 48; c.batch = 6; c.ld = 79; c.xinc = 2; c.yinc = 3;
    c.transA = Transpose::ConjTrans;
    c.alpha = static_cast<S>(1.5); c.beta = static_cast<S>(-0.75);
    this->run_case(c);
}

// A COMPLEX alpha AND beta. Every scalar in the fixture above is real, so the
// alpha/beta multiplies never mix components either -- a kernel that dropped
// the cross-terms of `alpha * sum` alone would pass everything up to here.
TYPED_TEST(GemvCoverageTest, ComplexAlphaBetaConjTranspose) {
    using S = typename TestFixture::ScalarType;
    typename TestFixture::Case c;
    c.m = 33; c.n = 41; c.batch = 4; c.ld = 40; c.xinc = 1; c.yinc = 2;
    c.transA = Transpose::ConjTrans;
    if constexpr (test_utils::is_complex<S>::value) {
        c.alpha = S(0.5, 1.25);
        c.beta = S(-0.75, 0.5);
    } else {
        c.alpha = static_cast<S>(0.5);
        c.beta = static_cast<S>(-0.75);
    }
    this->run_case(c);
}

// A reduction length that is not a multiple of the sub-group, at a batch large
// enough that the flattened launch spans several batch items per work-group.
// m = 97 = 3*32 + 1, so exactly one lane contributes in the final round.
TYPED_TEST(GemvCoverageTest, SubGroupTailTranspose) {
    using S = typename TestFixture::ScalarType;
    typename TestFixture::Case c;
    c.m = 97; c.n = 33; c.batch = 40; c.ld = 97; c.xinc = 1; c.yinc = 1;
    c.transA = Transpose::Trans;
    c.alpha = static_cast<S>(2.0); c.beta = static_cast<S>(0.0);
    this->run_case(c);
}

// Batch 1, which takes a different vendor entry (cublasXgemv, not
// XgemvStridedBatched) and, natively, a launch whose only extent is out_len.
TYPED_TEST(GemvCoverageTest, SingleBatchNonSquareTranspose) {
    using S = typename TestFixture::ScalarType;
    typename TestFixture::Case c;
    c.m = 45; c.n = 19; c.batch = 1; c.ld = 51; c.xinc = 3; c.yinc = 1;
    c.transA = Transpose::Trans;
    c.alpha = static_cast<S>(-1.25); c.beta = static_cast<S>(2.0);
    this->run_case(c);
}

// beta == 0 MEANS y IS NOT READ. Reference ?GEMV writes Y(I) = ZERO rather than
// scaling, so a y full of NaN must come back finite and correct. A kernel that
// evaluated beta*y unconditionally produces 0 * NaN = NaN everywhere.
TYPED_TEST(GemvCoverageTest, BetaZeroDoesNotReadY) {
    using S = typename TestFixture::ScalarType;
    typename TestFixture::Case c;
    c.m = 70; c.n = 48; c.batch = 4; c.ld = 79; c.xinc = 1; c.yinc = 2;
    c.transA = Transpose::Trans;
    c.alpha = static_cast<S>(1.0); c.beta = static_cast<S>(0.0);
    c.y_starts_nan = true;
    this->run_case(c);
}

// alpha == 0 with beta != 1 IS NOT a quick return: reference ?GEMV falls
// through to the beta scaling and returns y = beta*y.
TYPED_TEST(GemvCoverageTest, AlphaZeroScalesY) {
    using S = typename TestFixture::ScalarType;
    typename TestFixture::Case c;
    c.m = 30; c.n = 24; c.batch = 3; c.ld = 30; c.xinc = 1; c.yinc = 1;
    c.transA = Transpose::NoTrans;
    c.alpha = static_cast<S>(0.0); c.beta = static_cast<S>(0.5);
    // A IS ALL NaN. Reference ?GEMV never reads it when alpha == 0, so the
    // answer must be a finite 0.5*y. This is what makes the kernel's
    // `if (!alpha_zero)` guard observable: without it, 0 * NaN = NaN.
    c.a_starts_nan = true;
    this->run_case(c);
}

// --- THE OTHER ORIENTATION: m < n ------------------------------------------
//
// Everything above is m > n (70x48, 97x33, 45x19) apart from one ConjTrans
// case. m < n is a DIFFERENT failure mode for a kernel that confuses the two
// extents, and neither orientation alone catches both halves:
//
//   * A launcher that used n where it meant `out_len` UNDER-launches when
//     m > n. Every output it does write is still correct and the in-kernel
//     `gid >= total` guard keeps it in range -- so with m > n only the
//     unwritten tail is observable, and with m < n it is the OTHER extent that
//     truncates.
//   * A launcher that used the wrong REDUCTION extent truncates the sum when
//     the wrong extent is the smaller one, which is m < n under NoTrans and
//     m > n under Trans -- opposite orientations for the two arms.
//
// Demonstrated rather than argued: break `redextent` (clamp red_len to
// min(m,n)) turns WideNoTranspose RED and leaves NoTransposePaddedStrided
// GREEN, and break `outextent` (clamp out_len and the launch extent to
// min(m,n)) turns WideTranspose RED and leaves TransposePaddedStrided GREEN.

TYPED_TEST(GemvCoverageTest, WideNoTranspose) {
    using S = typename TestFixture::ScalarType;
    typename TestFixture::Case c;
    c.m = 41; c.n = 76; c.batch = 5; c.ld = 44; c.xinc = 2; c.yinc = 2;
    c.transA = Transpose::NoTrans;
    c.alpha = static_cast<S>(1.25); c.beta = static_cast<S>(-0.5);
    this->run_case(c);
}

TYPED_TEST(GemvCoverageTest, WideTranspose) {
    using S = typename TestFixture::ScalarType;
    typename TestFixture::Case c;
    c.m = 41; c.n = 76; c.batch = 5; c.ld = 44; c.xinc = 2; c.yinc = 2;
    c.transA = Transpose::Trans;
    c.alpha = static_cast<S>(1.25); c.beta = static_cast<S>(-0.5);
    this->run_case(c);
}

// --- beta == 0 AND alpha == 0 ON THE **NoTrans** BODY TOO -------------------
//
// BetaZeroDoesNotReadY and AlphaZeroScalesY above are each written on ONE arm,
// and the two arms are DIFFERENT KERNEL BODIES with their own copies of the
// `if (!beta_zero)` and `if (!alpha_zero)` guards -- three copies each, one per
// body. A break applied to all three copies at once would be caught, but a
// break to the copy in the arm that has no test would not: the NoTrans body's
// beta guard and the transposed bodies' alpha guard were both unobserved. These
// two close that, so every copy of both guards is now watched on both arms.

TYPED_TEST(GemvCoverageTest, BetaZeroDoesNotReadYNoTranspose) {
    using S = typename TestFixture::ScalarType;
    typename TestFixture::Case c;
    c.m = 70; c.n = 48; c.batch = 4; c.ld = 79; c.xinc = 2; c.yinc = 3;
    c.transA = Transpose::NoTrans;
    c.alpha = static_cast<S>(1.0); c.beta = static_cast<S>(0.0);
    c.y_starts_nan = true;
    this->run_case(c);
}

TYPED_TEST(GemvCoverageTest, AlphaZeroScalesYConjTranspose) {
    using S = typename TestFixture::ScalarType;
    typename TestFixture::Case c;
    c.m = 37; c.n = 29; c.batch = 3; c.ld = 41; c.xinc = 2; c.yinc = 1;
    c.transA = Transpose::ConjTrans;
    c.alpha = static_cast<S>(0.0); c.beta = static_cast<S>(0.5);
    c.a_starts_nan = true;   // never read when alpha == 0; 0 * NaN = NaN if it is
    this->run_case(c);
}

// --- A REALISTIC BATCH, AND THE LAUNCH GEOMETRY IT ALONE REACHES -----------
//
// Everything above runs at batch <= 40, and the work-group ladder
// (gemv_wg_ladder) picks its geometry from items = out_len * batch, so a small
// batch never leaves the bottom rung. Two configurations that nothing else in
// this file reaches, both computed from the ladder in src/sycl/gemv_native.cc
// against this box (MAX_COMPUTE_UNITS = 128, so the ladder's target is 512
// work-groups):
//
//   LargeBatchTallNoTranspose  m = 1024, batch = 128 -> items = 131072, and
//       wg = 256 is the FIRST rung reaching 512 groups, so body 1 launches
//       256-wide groups instead of the 32-wide groups every other case here
//       produces.
//   LargeBatchConjTranspose    n = 48, batch = 192 -> items = 9216, and for
//       body 3 (one 32-lane SUB-GROUP per output, units_per_wg_shift = 5)
//       wg = 256 gives 8 sub-groups per work-group and 1152 groups >= 512, so
//       it takes the top rung on the first try. Every other transposed case
//       here resolves to 1 or 2 sub-groups per group, which cannot see a
//       mistake in the group -> sub-group -> output mapping.
//
// Demonstrated: break `sgs` (pin sgs_per_wg to 1, i.e. assume one sub-group per
// work-group) is a NO-OP for every case whose ladder already chose 1, and turns
// exactly the multi-sub-group cases RED.
//
// Batch 1 is covered by SingleBatchNonSquareTranspose above, and the two batch
// extremes are separated by break `stride` (every batch item reads matrix 0),
// which is an identity at batch 1: it is red on both cases here and green on
// the batch-1 case.
//
// A PREDICTION THAT WAS REFUTED, recorded because the refutation is the useful
// part. A break called `flatten` was written to mis-pair b and i in the
// flattened index (b = gid % batch, i = gid / batch instead of b = gid /
// out_len, i = gid % out_len), on the expectation that it would turn every
// batch > 1 case red. It was applied, rebuilt and run, and it turned NOTHING
// red -- 176 of 176 still passed. The reason is that the wrong mapping is still
// a BIJECTION onto the same set of (b, i) pairs, and each work-item derives
// every address it touches from its own b and i, so each pair is still computed
// exactly once and correctly; only WHICH work-item does it changes. So no
// correctness test can observe the flattening at all. That is the right scope
// for this file: B5's flattening is a PERFORMANCE property (work-group count
// versus batch), and it is verified by the group-count arithmetic in
// src/sycl/gemv_native.cc, not here.

TYPED_TEST(GemvCoverageTest, LargeBatchTallNoTranspose) {
    using S = typename TestFixture::ScalarType;
    typename TestFixture::Case c;
    c.m = 1024; c.n = 8; c.batch = 128; c.ld = 1029; c.xinc = 2; c.yinc = 1;
    c.transA = Transpose::NoTrans;
    c.alpha = static_cast<S>(1.0); c.beta = static_cast<S>(0.25);
    this->run_case(c);
}

TYPED_TEST(GemvCoverageTest, LargeBatchConjTranspose) {
    using S = typename TestFixture::ScalarType;
    typename TestFixture::Case c;
    c.m = 96; c.n = 48; c.batch = 192; c.ld = 101; c.xinc = 2; c.yinc = 1;
    c.transA = Transpose::ConjTrans;
    c.alpha = static_cast<S>(-1.5); c.beta = static_cast<S>(0.5);
    this->run_case(c);
}

// --- the quick-return contract, checked for EXACT non-modification ----------
//
// Reference ?GEMV:
//
//     IF ((M.EQ.0) .OR. (N.EQ.0) .OR.
//    +    ((ALPHA.EQ.ZERO).AND.(BETA.EQ.ONE))) RETURN
//
// y is left COMPLETELY UNTOUCHED -- it is NOT scaled by beta. Both vendors
// agree, so a native path that scaled it would return an answer that depends on
// which route ran. These two tests compare BIT PATTERNS, not tolerances: the
// claim is "not written", not "written with the right value".

TYPED_TEST(GemvCoverageTest, ZeroReductionLeavesYUntouched) {
    using S = typename TestFixture::ScalarType;
    if (!this->ctx) return;

    const int m = 5, batch = 3, n = 0;
    UnifiedVector<S> A(1);
    UnifiedVector<S> x(1);
    UnifiedVector<S> y(m * batch);
    std::vector<S> before(static_cast<size_t>(m * batch));
    for (int t = 0; t < m * batch; ++t) {
        y[t] = gemv_cov_value<S>(t * 13 + 3);
        before[t] = y[t];
    }
    A[0] = static_cast<S>(1.0);
    x[0] = static_cast<S>(1.0);

    MatrixView<S, MatrixFormat::Dense> A_view(A.data(), m, n, m, 0, batch);
    VectorView<S> x_vec(x.data(), 0, batch, 1, 0);
    VectorView<S> y_vec(y.data(), m, batch, 1, m);

    // alpha != 0 and beta != 1, so ONLY the n == 0 clause can quick-return
    // here. Drop that clause and the reduction is empty, the kernel computes
    // y = beta*y, and every element below moves.
    gemv(*(this->ctx), A_view, x_vec, y_vec,
         {.alpha = static_cast<S>(2.0), .beta = static_cast<S>(0.5),
          .transA = Transpose::NoTrans});
    this->ctx->wait();

    for (int t = 0; t < m * batch; ++t) {
        EXPECT_EQ(y[t], before[t]) << "y[" << t << "] was written for n == 0";
    }
}

// THE OTHER DEGENERATE EXTENT, AND WHY IT HAS TO BE THE TRANSPOSED ARM.
//
// m == 0 under **Trans**: red_len == m == 0 while out_len == n > 0, so the
// launch geometry is NON-empty -- there are n*batch work-items, each with an
// empty reduction -- and a kernel that dropped the `m == 0` clause writes
// y = beta*y over every one of them. Under NoTrans the same m == 0 gives
// out_len == 0, an EMPTY launch, and nothing could move whatever the kernel
// did: the test would be vacuous. So the two halves of
//
//     IF ((M.EQ.0) .OR. (N.EQ.0) .OR. ...) RETURN
//
// are covered on opposite arms, each on the arm where the launch could
// actually write, and break `quickm` (drop the `m == 0` clause) turns this one
// red while leaving the n == 0 test above green.
TYPED_TEST(GemvCoverageTest, ZeroRowsLeavesYUntouched) {
    using S = typename TestFixture::ScalarType;
    if (!this->ctx) return;

    const int m = 0, n = 6, batch = 3;
    UnifiedVector<S> A(n * batch);
    UnifiedVector<S> x(1);
    UnifiedVector<S> y(n * batch);
    std::vector<S> before(static_cast<size_t>(n * batch));
    for (int t = 0; t < n * batch; ++t) {
        A[t] = static_cast<S>(1.0);
        y[t] = gemv_cov_value<S>(t * 19 + 5);
        before[t] = y[t];
    }
    x[0] = static_cast<S>(1.0);

    // ld = 1 because ld == rows == 0 would make the view's resolved stride 0.
    MatrixView<S, MatrixFormat::Dense> A_view(A.data(), m, n, 1, n, batch);
    VectorView<S> x_vec(x.data(), 0, batch, 1, 0);
    VectorView<S> y_vec(y.data(), n, batch, 1, n);
    ASSERT_EQ(A_view.rows(), 0);
    ASSERT_EQ(x_vec.size(), 0);

    // alpha != 0 and beta != 1, so ONLY the m == 0 clause can quick-return.
    gemv(*(this->ctx), A_view, x_vec, y_vec,
         {.alpha = static_cast<S>(2.0), .beta = static_cast<S>(0.5),
          .transA = Transpose::Trans});
    this->ctx->wait();

    for (int t = 0; t < n * batch; ++t) {
        EXPECT_EQ(y[t], before[t]) << "y[" << t << "] was written for m == 0";
    }
}

TYPED_TEST(GemvCoverageTest, AlphaZeroBetaOneLeavesYUntouched) {
    using S = typename TestFixture::ScalarType;
    if (!this->ctx) return;

    const int m = 12, n = 9, batch = 3;
    UnifiedVector<S> A(m * n * batch);
    UnifiedVector<S> x(n * batch);
    UnifiedVector<S> y(m * batch);
    std::vector<S> before(static_cast<size_t>(m * batch));
    // A IS ALL NaN, for the reason recorded on AlphaZeroScalesY: with a
    // finite A the quick return is arithmetically indistinguishable from
    // computing 0*A*x + 1*y, so no break could ever turn this red and the
    // test would be a blind guard. With NaN it is observable.
    for (int t = 0; t < m * n * batch; ++t)
        A[t] = static_cast<S>(std::numeric_limits<typename batchlas::base_type<S>::type>::quiet_NaN());
    for (int t = 0; t < n * batch; ++t) x[t] = gemv_cov_value<S>(t * 3 + 2);
    for (int t = 0; t < m * batch; ++t) {
        y[t] = gemv_cov_value<S>(t * 5 + 7);
        before[t] = y[t];
    }

    MatrixView<S, MatrixFormat::Dense> A_view(A.data(), m, n, m, m * n, batch);
    VectorView<S> x_vec(x.data(), n, batch, 1, n);
    VectorView<S> y_vec(y.data(), m, batch, 1, m);

    gemv(*(this->ctx), A_view, x_vec, y_vec,
         {.alpha = static_cast<S>(0.0), .beta = static_cast<S>(1.0),
          .transA = Transpose::NoTrans});
    this->ctx->wait();

    // EXACT, and against a NaN-filled A. Reference ?GEMV returns before
    // touching either operand; a kernel that fell through would compute
    // 0 * NaN = NaN and every element below would move.
    for (int t = 0; t < m * batch; ++t) {
        EXPECT_EQ(y[t], before[t]) << "y[" << t << "] moved under alpha=0, beta=1";
    }
}

// --- BODY 4, THE SEGMENTED NoTrans BODY (repair phase) ----------------------
//
// WHY THESE EXIST AT ALL. gemv_native_direct picks between body 1 and body 4
// on out_len and on whether the device ENUMERATES a sub-group size of 32:
//
//     out_len <= 16 and sub-group 32 available  ->  body 4 (segmented)
//     everything else                           ->  body 1
//
// That choice is INVISIBLE to the route table -- both are {Native, Direct} --
// so no coverage row, no route column and no route_diff line can tell you which
// body ran. The only instrument that can is a test that is red for one body and
// green for the other, which is what the break table below establishes.
//
// AND THE HOLE THEY CLOSE. When body 4 landed, EVERY NoTrans case in this file
// used m >= 41, i.e. out_len > 16, so every one of them took body 1. The only
// tests reaching body 4 were the forty pre-WP7 GemvMatrixViewTest cases at
// 10x10 -- and those are this file's documented BLIND GUARD: purely real
// complex data, integer values below 2^24, ld == rows, inc == 1, and two of
// them comparing only std::real. A body-4 kernel that dropped every complex
// cross-term, ignored ld, ignored xinc or ignored yinc would have passed all
// forty. Measured, not asserted: see the break table in the repair report.
//
// THE FOUR CASES WALK THE SEGMENT-WIDTH LADDER, because W = gemv_seg_width(m)
// is a template parameter and each value is a SEPARATE INSTANTIATION -- a
// defect in one of them is invisible to the others:
//
//     m = 1   -> W = 32, 32 of 32 lanes live, 5 fold steps
//     m = 4   -> W = 8,  32 of 32 lanes live, 3 fold steps
//     m = 10  -> W = 2,  20 of 32 lanes live  <- the PARTIAL-LANE case: lanes
//                20..31 hold an untouched zero and shuffle from past lane 31
//     m = 16  -> W = 2,  32 of 32 lanes live
//
// and the fifth sits one element ABOVE the gate, where W == 1 and body 1 must
// take the call back.
//
// MEASURED, EACH BREAK APPLIED TO src/sycl/gemv_native.cc, REBUILT IN
// build-novendor, RUN, AND REVERTED (232 cases total; "pre-WP7" = the forty
// GemvMatrixViewTest cases; every red case is Backend 1 = CUDA, because the
// native_cpu queue does not enumerate a sub-group size of 32 and therefore
// never reaches body 4 at all -- which is itself the proof that the gate works):
//
//   break        red   pre-WP7 red   coverage tests turned red
//   ----------------------------------------------------------------------
//   segfold       14        6        the three complex seg cases + Seg.BetaZero
//                                    (NOT SegmentedSingleRow: at m = 1 the fold
//                                     stride seg*w EQUALS w, so the break is an
//                                     identity there -- which is why m = 1 is
//                                     not sufficient on its own)
//   segfold2      14        6        the same four, real types (the real and
//                                    complex folds are separate code)
//   segmap        36       12        all six computing seg cases
//   segld         20        0   <--  ld ignored: pre-WP7 is BLIND (ld == rows)
//   segxinc       16        0   <--  xinc ignored: pre-WP7 is BLIND (inc == 1)
//   segyinc       20        0   <--  yinc ignored: pre-WP7 is BLIND
//   segstride     28        8        per-item stride ignored
//   segbeta        4        0        SegmentedBetaZeroDoesNotReadY alone
//   segalpha       4        0        SegmentedAlphaZeroScalesY alone
//   segwrite      32       12        `lane < seg` narrowed to `lane == 0`
//   segwidth34     4        0        SegmentGateBoundaryNoTranspose ALONE
//   seggate1      20        0        the gate opened to W == 1: the boundary
//                                    case plus every NoTrans case with
//                                    out_len > 32, where 32 lanes cannot cover
//                                    the output at all
//   cross         84        0   <--  complex cross-terms deleted from the
//                                    SHARED fma_acc: all seven cases here red,
//                                    all forty pre-WP7 cases GREEN
//
// ONE PREDICTION WAS REFUTED AND IS RECORDED AS SUCH RATHER THAN DROPPED.
// Break `segactive` (drop the `jsub < wlanes` half of the accumulate guard, so
// the lanes at or above seg*W also walk the reduction) turned NOTHING red:
// 232 passed, 0 failed. That is not a coverage hole, it is the closure property
// body 4's header claims, confirmed from the other side -- those lanes' results
// are never read by a lane below seg, so the guard is a work saving and not a
// correctness condition.

TYPED_TEST(GemvCoverageTest, SegmentedSingleRowNoTranspose) {
    using S = typename TestFixture::ScalarType;
    typename TestFixture::Case c;
    c.m = 1; c.n = 200; c.batch = 9; c.ld = 5; c.xinc = 2; c.yinc = 3;
    c.transA = Transpose::NoTrans;
    c.alpha = static_cast<S>(1.25); c.beta = static_cast<S>(-0.5);
    this->run_case(c);
}

TYPED_TEST(GemvCoverageTest, SegmentedShortOutputNoTranspose) {
    using S = typename TestFixture::ScalarType;
    typename TestFixture::Case c;
    c.m = 4; c.n = 77; c.batch = 7; c.ld = 9; c.xinc = 3; c.yinc = 2;
    c.transA = Transpose::NoTrans;
    c.alpha = static_cast<S>(-1.75); c.beta = static_cast<S>(0.25);
    this->run_case(c);
}

TYPED_TEST(GemvCoverageTest, SegmentedPartialLanesNoTranspose) {
    using S = typename TestFixture::ScalarType;
    typename TestFixture::Case c;
    c.m = 10; c.n = 53; c.batch = 11; c.ld = 13; c.xinc = 2; c.yinc = 2;
    c.transA = Transpose::NoTrans;
    c.alpha = static_cast<S>(0.75); c.beta = static_cast<S>(1.5);
    this->run_case(c);
}

TYPED_TEST(GemvCoverageTest, SegmentedFullLanesNoTranspose) {
    using S = typename TestFixture::ScalarType;
    typename TestFixture::Case c;
    c.m = 16; c.n = 64; c.batch = 5; c.ld = 20; c.xinc = 1; c.yinc = 2;
    c.transA = Transpose::NoTrans;
    c.alpha = static_cast<S>(2.0); c.beta = static_cast<S>(-0.25);
    this->run_case(c);
}

// ONE ELEMENT ABOVE THE GATE. m = 17 gives W = 1, so gemv_native_direct must
// hand this back to body 1; it is also the smallest NoTrans out_len in this
// file, everything else being 41 or more. Break `segwidth34` -- an off-by-one
// in gemv_seg_width's `w * 2 * out_len <= 32`, made 34 -- turns THIS CASE AND
// NOTHING ELSE red (4 red of 232), because out_len == 17 is the only length for
// which 32 < 2*out_len <= 34, so body 4 claims it with W = 2 and then needs 34
// lanes of a 32-lane sub-group.
TYPED_TEST(GemvCoverageTest, SegmentGateBoundaryNoTranspose) {
    using S = typename TestFixture::ScalarType;
    typename TestFixture::Case c;
    c.m = 17; c.n = 45; c.batch = 6; c.ld = 23; c.xinc = 2; c.yinc = 1;
    c.transA = Transpose::NoTrans;
    c.alpha = static_cast<S>(1.5); c.beta = static_cast<S>(-0.75);
    this->run_case(c);
}

// alpha == 0 AND beta == 0 ON BODY 4 TOO. Body 4 carries its own fourth copy of
// the `if (!alpha_zero)` and `if (!beta_zero)` guards -- the file previously
// watched three copies, one per body, and body 4's were unobserved. NaN in the
// operand each guard is supposed to leave unread is what makes the claim
// observable rather than an arithmetic identity.
TYPED_TEST(GemvCoverageTest, SegmentedBetaZeroDoesNotReadY) {
    using S = typename TestFixture::ScalarType;
    typename TestFixture::Case c;
    c.m = 8; c.n = 40; c.batch = 6; c.ld = 11; c.xinc = 2; c.yinc = 3;
    c.transA = Transpose::NoTrans;
    c.alpha = static_cast<S>(1.0); c.beta = static_cast<S>(0.0);
    c.y_starts_nan = true;
    this->run_case(c);
}

TYPED_TEST(GemvCoverageTest, SegmentedAlphaZeroScalesY) {
    using S = typename TestFixture::ScalarType;
    typename TestFixture::Case c;
    c.m = 12; c.n = 33; c.batch = 5; c.ld = 15; c.xinc = 2; c.yinc = 2;
    c.transA = Transpose::NoTrans;
    c.alpha = static_cast<S>(0.0); c.beta = static_cast<S>(0.5);
    c.a_starts_nan = true;
    this->run_case(c);
}

// NON-NATURAL BATCH STRIDES. Every case above this point uses the natural
// stride -- a_stride == ld*n, x_stride == size*inc, y_stride == size*inc -- so a
// kernel that DERIVED each batch stride rather than reading it from the view
// passed the entire suite. The break that proves these four are armed replaces
// A.stride() with ld*cols and X/Y.stride() with size*inc in all four bodies;
// with it applied the suite was fully green before these cases existed.
//
// This is a live property, not a hypothetical one: ortho.cc:218-222 hands the
// native path A.stride() == m*A.cols() against a view whose ld*cols is m*i, on
// every CGS iteration. Until now the only thing guarding it was ortho_tests.
//
// One case per kernel body, so the guard cannot be satisfied by whichever body
// happens to serve a square NoTrans shape.
TYPED_TEST(GemvCoverageTest, PaddedBatchStrideNoTranspose) {
    using S = typename TestFixture::ScalarType;
    typename TestFixture::Case c;
    c.m = 70; c.n = 48; c.batch = 6; c.ld = 79; c.xinc = 2; c.yinc = 3;
    c.transA = Transpose::NoTrans;
    c.alpha = static_cast<S>(1.5); c.beta = static_cast<S>(-0.75);
    c.stride_pad = 37;   // co-prime with ld, xinc and yinc
    this->run_case(c);
}

TYPED_TEST(GemvCoverageTest, PaddedBatchStrideTranspose) {
    using S = typename TestFixture::ScalarType;
    typename TestFixture::Case c;
    c.m = 70; c.n = 48; c.batch = 6; c.ld = 79; c.xinc = 2; c.yinc = 3;
    c.transA = Transpose::Trans;
    c.alpha = static_cast<S>(1.5); c.beta = static_cast<S>(-0.75);
    c.stride_pad = 37;
    this->run_case(c);
}

TYPED_TEST(GemvCoverageTest, PaddedBatchStrideConjTranspose) {
    using S = typename TestFixture::ScalarType;
    typename TestFixture::Case c;
    c.m = 66; c.n = 44; c.batch = 5; c.ld = 71; c.xinc = 3; c.yinc = 2;
    c.transA = Transpose::ConjTrans;
    c.alpha = static_cast<S>(0.75); c.beta = static_cast<S>(1.25);
    c.stride_pad = 23;
    this->run_case(c);
}

// The segmented body (short output length under NoTrans), which the other three
// cases never reach -- their output lengths are 70 and 48, far above its gate.
TYPED_TEST(GemvCoverageTest, PaddedBatchStrideSegmented) {
    using S = typename TestFixture::ScalarType;
    typename TestFixture::Case c;
    c.m = 8; c.n = 40; c.batch = 6; c.ld = 11; c.xinc = 2; c.yinc = 3;
    c.transA = Transpose::NoTrans;
    c.alpha = static_cast<S>(1.25); c.beta = static_cast<S>(-0.5);
    c.stride_pad = 19;
    this->run_case(c);
}

// ... (Keep main function) ...
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// Note: The commented-out tests (SmallMatrixGemv, DoublePrecisionGemv)
// would need similar updates to use MatrixView/VectorView and potentially
// adjustments to the fixture or test logic if they involve non-square matrices
// or different data types.