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

        for (int b = 0; b < this->batch_size; ++b) {
            int vec_dim = this->cols;
            for (int j = 0; j < vec_dim; ++j) {
                this->x_data[b * std::max(this->rows, this->cols) + j] = static_cast<ScalarType>(j + 1 + b * 10);
            }
        }
        
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



// Coverage fixture. GemvMatrixViewTest above is fixed at 10x10, ld == rows,
// inc == 1, NoTrans/Trans only, and its complex data has an identically zero
// imaginary part, so a kernel that dropped every complex cross-term or got
// ConjTrans backwards passes all forty of its cases. This fixture covers those
// gaps: non-square, ld-padded, inc-strided, ConjTrans, genuinely complex data.
// evidence: docs/perf/gemv.md#blind-guards-found-and-closed

namespace {

// Deterministic, and for complex WITH A NON-ZERO IMAGINARY PART -- the one
// property the fixture above lacks. Magnitudes stay in [-1, 1] so a reduction
// of ~100 terms cannot lose the relative tolerance to cancellation.
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
        // alpha == 0, where reference ?GEMV never reads A -- it is what makes
        // "alpha == 0 does not touch A" observable rather than an identity.
        bool a_starts_nan = false;
        // Pushes every batch stride past its natural value, so stride != ld*cols
        // for A and != size*inc for x and y. A kernel that DERIVES the stride
        // instead of reading it from the view passes every natural-stride case;
        // ortho.cc hands gemv exactly such a view on every CGS iteration.
        int stride_pad = 0;
    };

    // Runs one case through the public gemv and checks every element of every
    // batch item against a host reference written from the BLAS definition --
    // not transcribed from either backend, which fold transA the same way.
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
        // Guard band past the end of y. Three breaks against body 5's tail mask
        // were green over 376 cases because an out-of-range write landed past
        // this allocation, where nothing was looking. Poisoned before the call
        // and asserted untouched after.
        // evidence: docs/perf/gemv.md#blind-guards-found-and-closed
        constexpr int kGuard = 64;
        UnifiedVector<ScalarType> y(std::max(1, y_stride * c.batch) + kGuard);

        // A's pad rows (m..ld-1) hold a large poison value: a kernel that walked
        // a column by ld instead of m, or that mixed the pad into a reduction,
        // cannot come back with a right answer by luck.
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
            // The stride pad itself; the loop above stops at ld*n, so without
            // this the padded tail would be uninitialised rather than poisoned.
            for (int t = ld * c.n; t < a_stride; ++t) A[b * a_stride + t] = poison;
        }
        // The x gaps (the xinc-1 slots between live elements) are poisoned too.
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
        const ScalarType guard_v = static_cast<ScalarType>(RealType(-98765));
        for (int t = 0; t < kGuard; ++t) y[y_stride * c.batch + t] = guard_v;

        MatrixView<ScalarType, MatrixFormat::Dense> A_view(
            A.data(), c.m, c.n, ld, a_stride, c.batch);
        VectorView<ScalarType> x_vec(x.data(), red, c.batch, c.xinc, x_stride);
        VectorView<ScalarType> y_vec(y.data(), out, c.batch, c.yinc, y_stride);

        // Asserted rather than trusted: VectorView takes (data, size, batch, inc,
        // stride) while Vector takes (size, batch, stride, inc) -- positions 3
        // and 4 swapped, both plain int, and both fit the same buffer length.
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
                // alpha == 0 never reads A, in the reference as in the kernel:
                // a reference that summed a NaN-filled A would predict NaN.
                const bool skip_a = (c.alpha == ScalarType(0));
                // Backward-error denominator. Comparing against |expected|
                // alone is a cancellation detector, not a tolerance; the BLAS
                // bound is relative to sum|a_r||x_r|, floored at 1.
                RealType absum = RealType(0);
                for (int r = 0; skip_a ? false : (r < red); ++r) {
                    // op(A)(o, r), conjugated for ConjTrans. Getting this
                    // backwards is the classic silent complex gemv bug.
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

            // The gaps between y's live elements must be untouched: a kernel
            // that ignored yinc writes into them and every check above passes.
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

        // Nothing may write past the last batch item of y.
        for (int t = 0; t < kGuard; ++t) {
            EXPECT_EQ(y[y_stride * c.batch + t], guard_v)
                << "y guard band element " << t << " past the end of batch "
                << c.batch << " was written";
        }
    }
};

TYPED_TEST_SUITE(GemvCoverageTest, MyTypes);

// --- the three transA arms, on non-square, ld-padded, inc-strided data ------
// m = 70 is deliberately not a multiple of 32: the CTA body strides its
// reduction by 32, so 70 = 2*32 + 6 exercises the partial final round.

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

// ConjTrans is the live production path -- ortho.cc selects it for all four
// complex types -- and had no test in this tree. It runs on the real types too,
// where ConjTrans == Trans, to catch a kernel that conjugates a real value.
// evidence: docs/perf/gemv.md#correctness-findings
TYPED_TEST(GemvCoverageTest, ConjTransposePaddedStrided) {
    using S = typename TestFixture::ScalarType;
    typename TestFixture::Case c;
    c.m = 70; c.n = 48; c.batch = 6; c.ld = 79; c.xinc = 2; c.yinc = 3;
    c.transA = Transpose::ConjTrans;
    c.alpha = static_cast<S>(1.5); c.beta = static_cast<S>(-0.75);
    this->run_case(c);
}

// A complex alpha and beta: every scalar in the fixture above is real, so its
// alpha/beta multiplies never mix components either.
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

// A reduction length that is not a multiple of the sub-group: m = 97 = 3*32 + 1,
// so exactly one lane contributes in the final round.
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

// beta == 0 means y is not read: reference ?GEMV writes Y(I) = ZERO rather than
// scaling, so a y full of NaN must come back finite. 0 * NaN = NaN otherwise.
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
    // A is all NaN: reference ?GEMV never reads it when alpha == 0, which is
    // what makes the kernel's `if (!alpha_zero)` guard observable.
    c.a_starts_nan = true;
    this->run_case(c);
}

// --- THE OTHER ORIENTATION: m < n ------------------------------------------
// Everything above is m > n. m < n is a different failure mode: a launcher that
// used the wrong OUTPUT extent under-launches on one orientation, and one that
// used the wrong REDUCTION extent truncates on the other.

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

// --- beta == 0 and alpha == 0 on the NoTrans body too -----------------------
// The arms are different kernel bodies with their own copies of the
// `if (!beta_zero)` and `if (!alpha_zero)` guards; these watch the other copies.

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

// --- a realistic batch, and the launch geometry it alone reaches ------------
// Everything above runs at batch <= 40, and the work-group ladder picks its
// geometry from out_len * batch, so nothing else in this file reaches a
// work-group holding more than one or two sub-groups.

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
// Reference ?GEMV returns on (M == 0) || (N == 0) || (ALPHA == 0 && BETA == 1)
// with y COMPLETELY UNTOUCHED -- it is not scaled by beta. These tests compare
// bit patterns, not tolerances: the claim is "not written".

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

    // alpha != 0 and beta != 1, so only the n == 0 clause can quick-return here;
    // drop it and the kernel computes y = beta*y and every element moves.
    gemv(*(this->ctx), A_view, x_vec, y_vec,
         {.alpha = static_cast<S>(2.0), .beta = static_cast<S>(0.5),
          .transA = Transpose::NoTrans});
    this->ctx->wait();

    for (int t = 0; t < m * batch; ++t) {
        EXPECT_EQ(y[t], before[t]) << "y[" << t << "] was written for n == 0";
    }
}

// m == 0 has to be tested on the TRANSPOSED arm: red_len == 0 while
// out_len == n > 0, so the launch is non-empty and a kernel that dropped the
// `m == 0` clause writes y = beta*y over all of it. Under NoTrans the launch is
// empty and nothing could move, so the test would be vacuous.
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
    // A is all NaN: with a finite A the quick return is arithmetically
    // indistinguishable from computing 0*A*x + 1*y, so no break could move it.
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

    // Exact, and against a NaN-filled A: a kernel that fell through gets NaN.
    for (int t = 0; t < m * batch; ++t) {
        EXPECT_EQ(y[t], before[t]) << "y[" << t << "] moved under alpha=0, beta=1";
    }
}

// --- body 4, the segmented NoTrans body ------------------------------------
// gemv_native_direct picks body 4 over body 1 when out_len <= 16 and the device
// enumerates a sub-group size of 32. That choice is INVISIBLE to the route table
// -- both are {Native, Direct} -- so only a test that is red for one body and
// green for the other can tell you which ran. The four cases below walk the
// segment-width ladder W = gemv_seg_width(m) at m = 1, 4, 10 and 16, each W
// being a separate template instantiation; m = 10 is the partial-lane case.
// evidence: docs/perf/gemv.md#the-body-4-gate

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

// One element above the gate: m = 17 gives W = 1, so body 1 must take the call
// back. It is also the only out_len for which 32 < 2*out_len <= 34, so an
// off-by-one in gemv_seg_width's `w * 2 * out_len <= 32` shows up here alone.
TYPED_TEST(GemvCoverageTest, SegmentGateBoundaryNoTranspose) {
    using S = typename TestFixture::ScalarType;
    typename TestFixture::Case c;
    c.m = 17; c.n = 45; c.batch = 6; c.ld = 23; c.xinc = 2; c.yinc = 1;
    c.transA = Transpose::NoTrans;
    c.alpha = static_cast<S>(1.5); c.beta = static_cast<S>(-0.75);
    this->run_case(c);
}

// Body 4 carries its own copies of the `if (!alpha_zero)` and `if (!beta_zero)`
// guards; NaN in the operand each leaves unread is what makes them observable.
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

// Non-natural batch strides. Every case above uses the natural stride, so a
// kernel that DERIVED each stride rather than reading it from the view passed
// the whole suite. ortho.cc is the live caller that defeats a derived stride.
// One case per kernel body, so no single body can satisfy the guard alone.
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

// The segmented body, which the three cases above never reach.
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
// Body 5 -- the segmented TRANSPOSED CTA kernel, GemvSegTKernel<T, W>. Its gate
// is on red_len (= m under Trans/ConjTrans), not out_len as body 4's is, so a
// case copied from the body-4 section reaches body 3 and proves nothing. The
// gate is also PER SCALAR TYPE (float <= 32, cfloat <= 16, double <= 48,
// cdouble <= 64), so one case reaches body 5 for all four types only if m <= 16.
// The resolved route reads native:cta for bodies 3 and 5 alike, so a break must
// be run in build-novendor or under an explicit route pin.
// evidence: docs/perf/gemv.md#the-body-5-gates
// ===========================================================================

// evidence: docs/perf/gemv.md#breaks-that-stayed-green

// The decision surface: a claim about WHICH KERNEL RUNS, asserted at every
// boundary of the two per-type tables and on both sides, plus the two ways the
// launcher declines after the gate says yes (no sub-group 32; red_len <= 0).
TYPED_TEST(GemvCoverageTest, SegTransWidthDecisionSurface) {
    using S = typename TestFixture::ScalarType;
    if (!this->ctx) return;
    const bool sg32 = this->ctx->device().supports_sub_group_size(32);

    // (red_len, expected W) transcribed from the two tables in
    // src/sycl/gemv_native.cc, including the cell one past each edge. kItems is
    // large enough to clear both rows of gate 3's parallelism floor.
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
        // A device with no enumerated sub-group 32 declines everything, which
        // is what keeps the Direct route's no-GPU-gate promise intact.
        EXPECT_EQ(got, sg32 ? w : 1) << "red_len " << rl << " sg32 " << sg32;
    }
    // Degenerate reduction lengths take body 3.
    EXPECT_EQ(batchlas::sycl_gemv::gemv_seg_trans_width_debug<S>(*this->ctx, 0, kItems), 1);
    EXPECT_EQ(batchlas::sycl_gemv::gemv_seg_trans_width_debug<S>(*this->ctx, -1, kItems), 1);
}

// And the shapes below really do reach it -- asserted, so a gate edit that sends
// them all to body 3 turns this red instead of leaving the section meaningless.
TYPED_TEST(GemvCoverageTest, SegTransCasesAreReachable) {
    using S = typename TestFixture::ScalarType;
    if (!this->ctx) return;
    if (!this->ctx->device().supports_sub_group_size(32)) return;

    // The out_len*batch the body-5 cases below actually use, per W band, against
    // gate 3's floors of 16*CU and 64*CU. On a bigger device the floors rise
    // above them, every body-5 case silently becomes a body-3 case and the whole
    // section proves nothing -- so it reports rather than passing quietly.
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

    // Every m used by a body-5 case below. m = 1, 3, 5 and 16 are inside every
    // type's gate; 40 and 44 are inside double's and complex<double>'s only.
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
// Every one has ld > m, xinc > 1, yinc > 1 and batch > 1, because body 5 reads
// all four and a case at their natural values cannot fail when they are ignored.

// L = 4 lanes and only one of them has work: red_len = 1 < L, so three lanes of
// every group carry a zero into the fold.
TYPED_TEST(GemvCoverageTest, SegTransMinimalReduction) {
    using S = typename TestFixture::ScalarType;
    typename TestFixture::Case c;
    c.m = 1; c.n = 200; c.batch = 16; c.ld = 5; c.xinc = 2; c.yinc = 3;
    c.transA = Transpose::Trans;
    c.alpha = static_cast<S>(1.25); c.beta = static_cast<S>(-0.5);
    this->run_case(c);
}

// Partial lanes and a tail sub-group at once: out_len*batch = 2385 is 1 mod 8,
// so the last sub-group is mostly past the end. Body 5's early exit is NOT
// sub-group uniform and must be MASKED, never returned, or the fold is torn.
TYPED_TEST(GemvCoverageTest, SegTransPartialLanesAndTail) {
    using S = typename TestFixture::ScalarType;
    typename TestFixture::Case c;
    c.m = 3; c.n = 53; c.batch = 45; c.ld = 7; c.xinc = 2; c.yinc = 2;
    c.transA = Transpose::ConjTrans;
    c.alpha = static_cast<S>(0.75); c.beta = static_cast<S>(1.5);
    this->run_case(c);
}

// The W outputs of one sub-group straddle a batch boundary: out_len = 5 with
// W = 8, so no sub-group's outputs lie inside one matrix. A kernel that computed
// b once per sub-group instead of once per lane group is wrong only here.
TYPED_TEST(GemvCoverageTest, SegTransOutputsStraddleBatchItems) {
    using S = typename TestFixture::ScalarType;
    typename TestFixture::Case c;
    c.m = 5; c.n = 5; c.batch = 613; c.ld = 9; c.xinc = 3; c.yinc = 2;
    c.transA = Transpose::Trans;
    c.alpha = static_cast<S>(-1.75); c.beta = static_cast<S>(0.25);
    this->run_case(c);
}

// The W = 8 upper edge for every type: red_len = 16 is complex<float>'s whole
// gate and inside all three others, and is exactly 4*L -- no partial round.
TYPED_TEST(GemvCoverageTest, SegTransFullLanesConjTranspose) {
    using S = typename TestFixture::ScalarType;
    typename TestFixture::Case c;
    c.m = 16; c.n = 45; c.batch = 64; c.ld = 19; c.xinc = 2; c.yinc = 3;
    c.transA = Transpose::ConjTrans;
    c.alpha = static_cast<S>(2.0); c.beta = static_cast<S>(-0.25);
    this->run_case(c);
}

// The W = 4 band is a separate template instantiation and therefore a separate
// kernel, sharing no code with W = 8. red_len = 40 is inside double's and
// complex<double>'s gates only; for the other two this runs on body 3.
TYPED_TEST(GemvCoverageTest, SegTransWideBandTranspose) {
    using S = typename TestFixture::ScalarType;
    typename TestFixture::Case c;
    c.m = 40; c.n = 37; c.batch = 224; c.ld = 47; c.xinc = 2; c.yinc = 2;
    c.transA = Transpose::Trans;
    c.alpha = static_cast<S>(1.5); c.beta = static_cast<S>(-0.75);
    this->run_case(c);
}

// One element above complex<float>'s gate and inside the other three: m = 17 is
// the smallest red_len at which the four types disagree.
TYPED_TEST(GemvCoverageTest, SegTransGateBoundaryConjTranspose) {
    using S = typename TestFixture::ScalarType;
    typename TestFixture::Case c;
    c.m = 17; c.n = 41; c.batch = 64; c.ld = 23; c.xinc = 3; c.yinc = 2;
    c.transA = Transpose::ConjTrans;
    c.alpha = static_cast<S>(1.5); c.beta = static_cast<S>(-0.75);
    this->run_case(c);
}

// Body 5's own copies of the alpha == 0 and beta == 0 guards -- the fifth pair,
// one per body. NaN in the operand each leaves unread makes them observable.
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

// The batch stride on body 5: it reads A/X/Y.stride() in its own code, and none
// of the four PaddedBatchStride cases above has an m that reaches it.
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


// Gate 3, the parallelism condition, on both sides. Body 5 launches
// (out_len*batch)/W sub-groups against body 3's out_len*batch, so below
// 8*MAX_COMPUTE_UNITS outputs it gives away parallelism the shape cannot spare.
// evidence: docs/perf/gemv.md#the-body-5-gates
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

    // The W = 4 row, floor 64*CU -- four times higher, which is the point of the
    // floor being a table. Only double and complex<double> have a W = 4 band.
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

// The env knob bypasses all three gates. BATCHLAS_GEMV_SEGT is re-read per call
// and never latched, precisely so an earlier gemv call in the same process
// cannot defeat this assertion.
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

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}