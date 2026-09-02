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
#include "../src/sycl/gemv_native.hh"
#include "../src/backends/gemv_route.hh"
#include <utility>
#include <cstdlib>

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
        // A GUARD BAND PAST THE END OF y, AND IT EXISTS BECAUSE THE SUITE WAS
        // BLIND WITHOUT IT. Body 5's tail sub-group covers W outputs and can run
        // past the last one; its correctness rests on a mask and a clamp. THREE
        // separate breaks against that pair -- `segTtail` (return instead of
        // mask), `segTtailwrite` (mask dropped at the store) and `segTclampoff2`
        // (mask AND clamp dropped together) -- ALL CAME BACK GREEN over 376
        // cases, because a write past the last batch item lands past the end of
        // this allocation, where nothing was looking. That is the twelfth
        // recorded blind guard in this campaign and it is closed here: the band
        // is poisoned before the call and asserted untouched after it, so an
        // out-of-range y write has somewhere observable to land.
        constexpr int kGuard = 64;
        UnifiedVector<ScalarType> y(std::max(1, y_stride * c.batch) + kGuard);

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
        // The guard band, poisoned with a value no correct kernel produces.
        const ScalarType guard_v = static_cast<ScalarType>(RealType(-98765));
        for (int t = 0; t < kGuard; ++t) y[y_stride * c.batch + t] = guard_v;

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

        // THE GUARD BAND. Nothing may write past the last batch item of y. See
        // the note on its allocation: three separate breaks against body 5's
        // tail masking were green over 376 cases until this existed.
        for (int t = 0; t < kGuard; ++t) {
            EXPECT_EQ(y[y_stride * c.batch + t], guard_v)
                << "y guard band element " << t << " past the end of batch "
                << c.batch << " was written";
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


// ===========================================================================
// BODY 5 -- THE SEGMENTED **TRANSPOSED** CTA KERNEL, GemvSegTKernel<T, W>.
// (src/sycl/gemv_native.cc; W outputs per sub-group, L = 32/W lanes each.)
//
// WHY THE CASES BELOW LOOK NOTHING LIKE BODY 4's. Body 4's gate is on out_len
// and body 5's is on **red_len**, which under Trans/ConjTrans is m -- so where
// body 4 needed short OUTPUT vectors, body 5 needs short REDUCTIONS, i.e. small
// m with n large. A case copied across from the body-4 section reaches body 3
// and proves nothing. That axis confusion is the error route_gemv.hh:273-277
// records being caught twice in WP7, and it is the single easiest way to make
// this whole section vacuous.
//
// AND THE GATE IS PER SCALAR TYPE, which is new in this file. Body 5 runs at
// red_len <= 32 for float, <= 16 for complex<float>, <= 48 for double and
// <= 64 for complex<double>, because body 3 reaches the DRAM roof at a
// different red_len for each (its own GB/s at out_len 2048, batch 512:
// float 833 at red_len 32 but 913 at 48; cfloat 780 at 16 but 911 at 24;
// double 548 at 32, 817 at 48, 929 at 64; cdouble 434 at 32, 456 at 48, 708 at
// 64, 933 at 128). These are TYPED tests over all four scalars, so a single
// case does NOT reach body 5 for all four types unless m <= 16. That is stated
// per case below, and SegTransWidthDecisionSurface is what pins it down: it
// asserts the resolved W through gemv_seg_trans_width_debug -- THE SAME gate
// function the launcher calls -- at every boundary, on both sides, for all four
// types.
//
// REACHABILITY, AND WHY IT NEEDED ITS OWN INSTRUMENT. {Native, CTA} now names
// TWO kernels, so the resolved route column -- the campaign's usual answer to
// "linked is not reachable" -- reads `native:cta` for both of them and cannot
// distinguish a body-5 launch from a body-3 launch. With preferred() all-false
// the CTA route is not taken at all in a vendor-present build, so every break
// here must be run in build-novendor or under an explicit route pin. Both halves
// are checked below rather than assumed.
// ===========================================================================

// MEASURED, EACH BREAK APPLIED TO src/sycl/gemv_native.cc, REBUILT IN
// build-novendor, RUN, AND REVERTED (docs/perf/gemv.md#breaks-that-stayed-green; 376
// cases total; every red case is Backend 1 = CUDA, because the native_cpu queue
// does not enumerate a sub-group size of 32 and therefore never reaches body 5 --
// which is itself the proof that the sub-group gate works). NOT ONE break turned
// a pre-WP7 GemvMatrixViewTest case red, so "the old cases stay green" is a
// measurement about which kernel ran and not an accident of a shared path.
//
//   break           red   coverage tests turned red
//   ------------------------------------------------------------------------
//   segTmap          35   all ten computing seg-T cases (s and o swapped)
//   segTfold         15   nine (complex fold at stride L instead of 1)
//   segTfold2        16   nine (the real fold; separate code from the complex)
//   segTld           31   nine  -- ld ignored
//   segTxinc         27   eight -- xinc ignored
//   segTyinc         35   ten   -- yinc ignored
//   segTstridea       6   PaddedBatchStride + ...WideBand ALONE  <-- the batch
//                         stride derived as ld*cols instead of read from the
//                         view. TWO cases in the whole file see it.
//   segTstridex      31   nine  -- x's per-item stride ignored
//   segTstridey      35   ten   -- y's per-item stride ignored
//   segTconj          6   the four ConjTrans seg-T cases ALONE
//   segTwrite        35   ten (`s == 0` narrowed to `lane == 0`)
//   segTalpha         4   SegTransAlphaZeroScalesY ALONE
//   segTbeta          4   SegTransBetaZeroDoesNotReadY ALONE
//   segTtailwrite    10   the THREE cases with a partial tail sub-group ALONE
//   segTclampoff2    10   the same three (mask AND clamp dropped together)
//   segTgateopen     12   the four decision tests (gate 1 opened to red_len 1e5)
//   segTgateshut     16   the same four (gate 1 shut entirely)
//   segTfloorgone     4   SegTransParallelismGate ALONE (gate 3 removed)
//   segTfloorflat     2   SegTransParallelismGate ALONE (its two rows collapsed)
//   segTw8off         3   SegTransWidthDecisionSurface ALONE (W band off by one)
//   segTemitoff      16   the four decision tests (no type emits body 5)
//
// THREE BREAKS TURNED **NOTHING** RED, AND EACH ONE IS EXPLAINED RATHER THAN
// CELEBRATED. Two of them were also the reason the y guard band above exists.
//
//   segTtail   -- `return` instead of the mask on the partial tail. Green, and
//     it stays green even with the guard band. The reason is the fold's CLOSURE:
//     lane group o reads only lanes o*L .. o*L+L-1, and the groups that would
//     return are the HIGH ones, which no surviving group reads from. So the mask
//     is a SPEC-CONFORMANCE requirement -- a shuffle reached by part of a
//     sub-group is undefined behaviour -- and NOT an observable-value
//     requirement on this hardware. Exactly body 4's `segactive` epitaph, from
//     the other side. No test in any suite can be written to catch it.
//
//   segTclampoff -- the clamp removed, the mask kept. Green, correctly: the
//     clamp only affects the ADDRESSES an out-of-range lane group forms, and
//     `active` still stops it dereferencing them. It is defence in depth, and
//     its partner segTclampoff2 (both removed) IS red, which is what shows the
//     pair is load-bearing together.
//
//   segTlaunch -- the sub-group count handed to the work-group ladder left at
//     out_len*batch instead of ceil(out_len*batch / W), so the launch is W times
//     too big. Green, and correctly: the extra sub-groups all have
//     base >= total and return. It is a PERFORMANCE defect (a W-fold
//     over-launch), not a correctness one, and no value check can see it.
//
// AND THE ONE THAT CHANGED THE SUITE. segTtailwrite and segTclampoff2 were BOTH
// GREEN over 376 cases before the y guard band existed, because an out-of-range
// write lands past the end of y's allocation where nothing was looking. That is
// the twelfth recorded blind guard in this campaign. With the band they turn
// exactly the three partial-tail cases red -- out_len*batch mod W != 0 for
// SegTransPartialLanesAndTail (2385 mod 8 = 1), SegTransOutputsStraddleBatchItems
// (3065 mod 8 = 1) and SegTransPaddedBatchStrideWideBand (8295 mod 4 = 3) -- and
// nothing else, which is the right answer.

// THE DECISION SURFACE. Not a performance claim -- a claim about WHICH KERNEL
// RUNS, asserted at every boundary of the two per-type tables and on both sides
// of each, so that no break can move a boundary without moving this test.
//
// It also pins the two ways the launcher declines AFTER the gate has said yes:
// a device that does not enumerate a sub-group size of 32 (the native_cpu
// queue, where body 5 carries an unsatisfiable reqd_sub_group_size), and
// red_len <= 0.
TYPED_TEST(GemvCoverageTest, SegTransWidthDecisionSurface) {
    using S = typename TestFixture::ScalarType;
    if (!this->ctx) return;
    const bool sg32 = this->ctx->device().supports_sub_group_size(32);

    // (red_len, expected W) transcribed from the two tables in
    // src/sycl/gemv_native.cc, per type, INCLUDING the cell one past each edge.
    // out_len*batch used for the gate-3 input. A million clears both rows of
    // the parallelism floor (16*CU and 64*CU) on any device this suite runs on;
    // the floor itself is asserted separately in SegTransParallelismGate. This
    // test is about the two red_len tables and nothing else.
    const int64_t kItems = 1 << 20;
    std::vector<std::pair<int, int>> want;
    if constexpr (std::is_same_v<S, float>) {
        want = {{1,8},{16,8},{24,8},{25,4},{32,4},{33,1},{48,1},{64,1},{128,1}};
    } else if constexpr (std::is_same_v<S, std::complex<float>>) {
        want = {{1,8},{8,8},{16,8},{17,1},{24,1},{32,1},{48,1},{64,1},{128,1}};
    } else if constexpr (std::is_same_v<S, double>) {
        want = {{1,8},{16,8},{32,8},{33,4},{48,4},{49,1},{64,1},{128,1}};
    } else {
        want = {{1,8},{16,8},{32,8},{33,4},{48,4},{64,4},{65,1},{96,1},{128,1}};
    }
    for (const auto& [rl, w] : want) {
        const int got = batchlas::sycl_gemv::gemv_seg_trans_width_debug<S>(*this->ctx, rl, kItems);
        // On a device with no enumerated sub-group 32 the launcher declines
        // EVERYTHING, and that is the property that keeps the Direct route's
        // no-GPU-gate promise intact -- body 5 must never be the reason a
        // native_cpu queue fails to launch.
        EXPECT_EQ(got, sg32 ? w : 1) << "red_len " << rl << " sg32 " << sg32;
    }
    // Degenerate reduction lengths take body 3, whose quick-return path this
    // file already guards.
    EXPECT_EQ(batchlas::sycl_gemv::gemv_seg_trans_width_debug<S>(*this->ctx, 0, kItems), 1);
    EXPECT_EQ(batchlas::sycl_gemv::gemv_seg_trans_width_debug<S>(*this->ctx, -1, kItems), 1);
}

// AND THE SHAPES BELOW REALLY DO REACH IT. Asserted rather than believed, for
// the exact m values the cases use, so that a gate edit which silently sends
// them all to body 3 turns THIS red instead of leaving the whole section green
// and meaningless. The resolved ROUTE is printed alongside, because a break run
// in a build where the CTA route is not taken is vacuous whatever the gate says.
TYPED_TEST(GemvCoverageTest, SegTransCasesAreReachable) {
    using S = typename TestFixture::ScalarType;
    if (!this->ctx) return;
    if (!this->ctx->device().supports_sub_group_size(32)) return;

    // THE ITEMS THE BODY-5 CASES BELOW ACTUALLY USE, per band, taken from the
    // smallest case in each: 2385 for the W = 8 band (SegTransPartialLanesAndTail,
    // n = 53, batch = 45) and 8288 for the W = 4 band (SegTransWideBandTranspose,
    // n = 37, batch = 224). GATE 3's floor is 16*CU and 64*CU respectively --
    // 2048 and 8192 on this 128-SM box -- so both clear it here.
    //
    // ON A BIGGER DEVICE THEY WOULD NOT, and then gate 3 declines and every
    // body-5 case in this file silently becomes an ordinary body-3 case: the
    // whole section stays green and proves nothing, which is the vacuous-break
    // failure mode this campaign records as trap 4. So it REPORTS rather than
    // passing quietly.
    const int64_t kItems8 = 2385;
    const int64_t kItems4 = 8288;
    const int cu = static_cast<int>(
        this->ctx->device().get_property(DeviceProperty::MAX_COMPUTE_UNITS));
    if (static_cast<int64_t>(16) * cu > kItems8 || static_cast<int64_t>(64) * cu > kItems4) {
        GTEST_SKIP() << "VACUOUS ON THIS DEVICE: MAX_COMPUTE_UNITS = " << cu
                     << " puts gate 3's floors at " << (16 * cu) << " and " << (64 * cu)
                     << ", above the smallest body-5 cases' out_len*batch of "
                     << kItems8 << " and " << kItems4
                     << ". Every body-5 case here would run body 3. Enlarge their "
                        "batch before trusting any break result.";
    }
    const int64_t kItems = kItems8;
    UnifiedVector<S> a(16 * 8), x(8), y(16);
    UnifiedVector<S*> pa(1);
    MatrixView<S, MatrixFormat::Dense> Av(a.data(), 8, 16, 8, 8 * 16, 1, pa.data());
    VectorView<S> Xv(x.data(), 8, 1, Inc{1}, Stride{8});
    VectorView<S> Yv(y.data(), 16, 1, Inc{1}, Stride{16});
    const auto rt = backend::gemv_route<TestFixture::BackendType, S>(
        *this->ctx, Av, Xv, Yv, Transpose::Trans, /*vendor_available=*/false);
    std::cout << "[ROUTE] gemv Trans (vendor_available=false) resolves to "
              << dispatch::to_string(rt.origin) << ":" << dispatch::to_string(rt.algo)
              << std::endl;

    // Every m used by a body-5 case below, with the W it must resolve to.
    // m = 1, 3, 5 and 16 are inside EVERY type's gate; 40 and 48 are inside
    // double's and complex<double>'s only, which is why they are checked with
    // the type-conditional expectation rather than a bare > 1.
    for (int m : {1, 3, 5, 16}) {
        EXPECT_GT(batchlas::sycl_gemv::gemv_seg_trans_width_debug<S>(*this->ctx, m, kItems), 1)
            << "body 5 must serve red_len " << m << " for every scalar type";
    }
    const bool wide = std::is_same_v<S, double> || std::is_same_v<S, std::complex<double>>;
    for (int m : {40, 44}) {
        EXPECT_EQ(batchlas::sycl_gemv::gemv_seg_trans_width_debug<S>(*this->ctx, m, kItems4) > 1, wide)
            << "red_len " << m << " is inside the double gates only";
        if (wide) {
            EXPECT_EQ(batchlas::sycl_gemv::gemv_seg_trans_width_debug<S>(*this->ctx, m, kItems4), 4)
                << "red_len " << m << " is in the W = 4 band";
        }
    }
}

// --- the cases -------------------------------------------------------------
//
// EVERY ONE OF THESE HAS ld > m, xinc > 1, yinc > 1 AND batch > 1, because the
// body-4 break table recorded that the pre-WP7 cases were structurally BLIND to
// ld (ld == rows everywhere), to xinc and to yinc (inc == 1 everywhere) -- three
// breaks that turned zero pre-existing cases red. Body 5 reads all three and a
// separate per-item stride; a case that leaves them at their natural values
// cannot fail when they are ignored.
//
// n IS LARGE AND m IS SMALL ON ALL OF THEM. That is the body-5 shape.

// L = 4 LANES AND ONLY ONE OF THEM HAS ANY WORK. red_len = 1 < L, so lanes
// s = 1, 2, 3 of every group never enter the loop and carry a zero into the
// fold; if the fold's stride or its participation were wrong, their zeros (or
// their neighbour group's partials) would land in y.
TYPED_TEST(GemvCoverageTest, SegTransMinimalReduction) {
    using S = typename TestFixture::ScalarType;
    typename TestFixture::Case c;
    c.m = 1; c.n = 200; c.batch = 16; c.ld = 5; c.xinc = 2; c.yinc = 3;
    c.transA = Transpose::Trans;
    c.alpha = static_cast<S>(1.25); c.beta = static_cast<S>(-0.5);
    this->run_case(c);
}

// PARTIAL LANES, AND A TAIL SUB-GROUP AT THE SAME TIME. red_len = 3 leaves one
// of the four lanes idle, and out_len*batch = 53*11 = 583, which is 7 mod 8 --
// so the LAST sub-group has one lane group in range and seven past the end.
// That is body 5's early-exit trap: the exit is NOT sub-group uniform and must
// be MASKED, never returned, or the fold is reached by part of a sub-group.
TYPED_TEST(GemvCoverageTest, SegTransPartialLanesAndTail) {
    using S = typename TestFixture::ScalarType;
    typename TestFixture::Case c;
    c.m = 3; c.n = 53; c.batch = 45; c.ld = 7; c.xinc = 2; c.yinc = 2;
    c.transA = Transpose::ConjTrans;
    c.alpha = static_cast<S>(0.75); c.beta = static_cast<S>(1.5);
    this->run_case(c);
}

// THE W OUTPUTS OF ONE SUB-GROUP STRADDLE A BATCH BOUNDARY. out_len = 5 and
// W = 8, so no sub-group's eight outputs ever lie inside one matrix: every one
// of them spans two or three batch items, each with its own b, its own column
// j, and its own x and y. A kernel that computed b once per sub-group instead
// of once per lane group -- the obvious simplification -- is wrong here and
// right everywhere else in this file.
TYPED_TEST(GemvCoverageTest, SegTransOutputsStraddleBatchItems) {
    using S = typename TestFixture::ScalarType;
    typename TestFixture::Case c;
    c.m = 5; c.n = 5; c.batch = 613; c.ld = 9; c.xinc = 3; c.yinc = 2;
    c.transA = Transpose::Trans;
    c.alpha = static_cast<S>(-1.75); c.beta = static_cast<S>(0.25);
    this->run_case(c);
}

// THE W = 8 UPPER EDGE FOR EVERY TYPE. red_len = 16 is complex<float>'s whole
// gate and is inside all three others, so this is the one case at the top of
// the W = 8 band that all four instantiations reach. red_len = 16 is also
// exactly 4*L, i.e. four full rounds with no partial one.
TYPED_TEST(GemvCoverageTest, SegTransFullLanesConjTranspose) {
    using S = typename TestFixture::ScalarType;
    typename TestFixture::Case c;
    c.m = 16; c.n = 45; c.batch = 64; c.ld = 19; c.xinc = 2; c.yinc = 3;
    c.transA = Transpose::ConjTrans;
    c.alpha = static_cast<S>(2.0); c.beta = static_cast<S>(-0.25);
    this->run_case(c);
}

// THE W = 4 BAND -- A SEPARATE TEMPLATE INSTANTIATION, AND THEREFORE A SEPARATE
// KERNEL. GemvSegTKernel<T,4> and GemvSegTKernel<T,8> share no code after the
// compiler is done with them (that is the whole point of W being a template
// parameter), so a defect in one is invisible to the other. red_len = 40 is
// inside double's and complex<double>'s gates only; for float and
// complex<float> this case runs on body 3 and is an ordinary Trans case.
TYPED_TEST(GemvCoverageTest, SegTransWideBandTranspose) {
    using S = typename TestFixture::ScalarType;
    typename TestFixture::Case c;
    c.m = 40; c.n = 37; c.batch = 224; c.ld = 47; c.xinc = 2; c.yinc = 2;
    c.transA = Transpose::Trans;
    c.alpha = static_cast<S>(1.5); c.beta = static_cast<S>(-0.75);
    this->run_case(c);
}

// ONE ELEMENT ABOVE complex<float>'s GATE AND INSIDE THE OTHER THREE. m = 17
// is the smallest red_len at which the four types do NOT agree, so a gate that
// collapsed to a single constant -- in either direction -- changes what runs
// here. Paired with SegTransFullLanesConjTranspose at m = 16, it brackets that
// boundary from both sides with a computing case, not only with the decision
// surface above.
TYPED_TEST(GemvCoverageTest, SegTransGateBoundaryConjTranspose) {
    using S = typename TestFixture::ScalarType;
    typename TestFixture::Case c;
    c.m = 17; c.n = 41; c.batch = 64; c.ld = 23; c.xinc = 3; c.yinc = 2;
    c.transA = Transpose::ConjTrans;
    c.alpha = static_cast<S>(1.5); c.beta = static_cast<S>(-0.75);
    this->run_case(c);
}

// alpha == 0 AND beta == 0 ON BODY 5's OWN COPIES OF THE GUARDS. This file now
// watches FIVE copies of `if (!alpha_zero)` and `if (!beta_zero)`, one per body,
// and body 5's would otherwise be the unobserved one. NaN in the operand each
// guard is supposed to leave unread is what makes the claim observable rather
// than an arithmetic identity that no break can move.
TYPED_TEST(GemvCoverageTest, SegTransBetaZeroDoesNotReadY) {
    using S = typename TestFixture::ScalarType;
    typename TestFixture::Case c;
    c.m = 8; c.n = 40; c.batch = 64; c.ld = 11; c.xinc = 2; c.yinc = 3;
    c.transA = Transpose::Trans;
    c.alpha = static_cast<S>(1.0); c.beta = static_cast<S>(0.0);
    c.y_starts_nan = true;
    this->run_case(c);
}

TYPED_TEST(GemvCoverageTest, SegTransAlphaZeroScalesY) {
    using S = typename TestFixture::ScalarType;
    typename TestFixture::Case c;
    c.m = 8; c.n = 40; c.batch = 64; c.ld = 11; c.xinc = 2; c.yinc = 3;
    c.transA = Transpose::ConjTrans;
    c.alpha = static_cast<S>(0.0); c.beta = static_cast<S>(-1.5);
    c.a_starts_nan = true;
    this->run_case(c);
}

// THE BATCH STRIDE, ON BODY 5. The stride_pad field exists because a previous
// rewrite of this suite was blind to batch stride -- a kernel that DERIVED the
// stride as ld*cols instead of reading A.stride() passed all 232 cases -- and
// ortho.cc:218-222 is the live caller that hands a derived-stride-defeating
// view on every CGS iteration. Body 5 reads A.stride(), X.stride() and
// Y.stride() in its own code; none of the four existing PaddedBatchStride cases
// reaches it, because their m values are 70, 70, 66 and 8-under-NoTrans.
TYPED_TEST(GemvCoverageTest, SegTransPaddedBatchStride) {
    using S = typename TestFixture::ScalarType;
    typename TestFixture::Case c;
    c.m = 13; c.n = 39; c.batch = 64; c.ld = 17; c.xinc = 2; c.yinc = 3;
    c.transA = Transpose::Trans;
    c.alpha = static_cast<S>(1.25); c.beta = static_cast<S>(-0.5);
    c.stride_pad = 29;   // co-prime with ld, xinc and yinc
    this->run_case(c);
}

TYPED_TEST(GemvCoverageTest, SegTransPaddedBatchStrideWideBand) {
    using S = typename TestFixture::ScalarType;
    typename TestFixture::Case c;
    c.m = 44; c.n = 35; c.batch = 237; c.ld = 51; c.xinc = 3; c.yinc = 2;
    c.transA = Transpose::ConjTrans;
    c.alpha = static_cast<S>(0.75); c.beta = static_cast<S>(1.25);
    c.stride_pad = 23;
    this->run_case(c);
}


// GATE 3, THE PARALLELISM CONDITION, ON BOTH SIDES. Body 5 launches
// (out_len*batch)/W sub-groups against body 3's out_len*batch, so below
// 8*MAX_COMPUTE_UNITS outputs it gives away parallelism the shape cannot spare.
// Measured: at out_len*batch <= 512 body 5 is 0.891x-0.989x of body 3
// (docs/perf/gemv.md#the-body-5-gates, sixteen losing cells found only
// because the WP7 parity grid reaches out_len = 1 and this pass's own
// (out_len, red_len) plane started at 64).
//
// This asserts the boundary through the launcher's own gate function, at a
// red_len every type admits, so it cannot pass by the gate being declined for
// some other reason.
TYPED_TEST(GemvCoverageTest, SegTransParallelismGate) {
    using S = typename TestFixture::ScalarType;
    if (!this->ctx) return;
    if (!this->ctx->device().supports_sub_group_size(32)) return;
    const int cu = static_cast<int>(
        this->ctx->device().get_property(DeviceProperty::MAX_COMPUTE_UNITS));
    auto w = [&](int red_len, int64_t items) {
        return batchlas::sycl_gemv::gemv_seg_trans_width_debug<S>(*this->ctx, red_len, items);
    };
    // THE W = 8 ROW, floor 16*CU, probed at a red_len every type puts in that band.
    const int64_t f8 = static_cast<int64_t>(16) * cu;
    EXPECT_EQ(w(8, f8 - 1), 1) << "one output short of the W = 8 floor must take body 3";
    EXPECT_EQ(w(8, f8), 8) << "exactly at the W = 8 floor body 5 must serve the call";
    EXPECT_EQ(w(8, 0), 1);
    EXPECT_EQ(w(8, 1), 1);

    // THE W = 4 ROW, floor 64*CU -- FOUR TIMES HIGHER, which is the whole point
    // of the floor being a table. Only double and complex<double> have a W = 4
    // band at red_len 40 (float's ends at 32, complex<float>'s at 16), so the
    // other two must decline at every items.
    const int64_t f4 = static_cast<int64_t>(64) * cu;
    if constexpr (std::is_same_v<S, double> || std::is_same_v<S, std::complex<double>>) {
        EXPECT_EQ(w(40, f4 - 1), 1) << "one output short of the W = 4 floor must take body 3";
        EXPECT_EQ(w(40, f4), 4) << "exactly at the W = 4 floor body 5 must serve the call";
        // AND THE TWO ROWS ARE DIFFERENT. At f8 the W = 8 band is admitted and
        // the W = 4 band is not; a single-number floor cannot produce this.
        EXPECT_EQ(w(8, f8), 8);
        EXPECT_EQ(w(40, f8), 1) << "the W = 4 band's floor is FOUR TIMES the W = 8 band's";
    } else {
        EXPECT_EQ(w(40, f4 * 16), 1) << "red_len 40 is above this type's gate";
    }
    EXPECT_GT(w(8, f8 * 64), 1);
}

// AND THE ENV KNOB BYPASSES ALL THREE GATES, which is what lets a measurement
// ask what body 5 WOULD do above a gate and a test reach body 3 below one.
// BATCHLAS_GEMV_SEGT is re-read per call and never latched, precisely so this
// assertion is not defeated by an earlier gemv call in the same process --
// the campaign's eleventh recorded blind guard.
TYPED_TEST(GemvCoverageTest, SegTransSpellingKnobIsNotLatched) {
    using S = typename TestFixture::ScalarType;
    if (!this->ctx) return;
    if (!this->ctx->device().supports_sub_group_size(32)) return;
    const int64_t kBig = 1 << 20;
    const int kFar = 100000;     // far above every per-type red_len gate
    // Default: gates apply.
    ::unsetenv("BATCHLAS_GEMV_SEGT");
    EXPECT_EQ(batchlas::sycl_gemv::gemv_seg_trans_width_debug<S>(*this->ctx, kFar, kBig), 1);
    EXPECT_GT(batchlas::sycl_gemv::gemv_seg_trans_width_debug<S>(*this->ctx, 8, kBig), 1);
    // Forced: gates bypassed, in both directions, AFTER a default read.
    ::setenv("BATCHLAS_GEMV_SEGT", "4", 1);
    EXPECT_EQ(batchlas::sycl_gemv::gemv_seg_trans_width_debug<S>(*this->ctx, kFar, 1), 4);
    ::setenv("BATCHLAS_GEMV_SEGT", "8", 1);
    EXPECT_EQ(batchlas::sycl_gemv::gemv_seg_trans_width_debug<S>(*this->ctx, kFar, 1), 8);
    ::setenv("BATCHLAS_GEMV_SEGT", "2", 1);
    EXPECT_EQ(batchlas::sycl_gemv::gemv_seg_trans_width_debug<S>(*this->ctx, kFar, 1), 2);
    ::setenv("BATCHLAS_GEMV_SEGT", "off", 1);
    EXPECT_EQ(batchlas::sycl_gemv::gemv_seg_trans_width_debug<S>(*this->ctx, 8, kBig), 1);
    // And unsetting it restores the gates -- which a latched PRESENCE would not.
    ::unsetenv("BATCHLAS_GEMV_SEGT");
    EXPECT_GT(batchlas::sycl_gemv::gemv_seg_trans_width_debug<S>(*this->ctx, 8, kBig), 1);
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