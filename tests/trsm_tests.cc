#include <gtest/gtest.h>
#include <batchlas/blas/linalg.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>
#include <batchlas/util/sycl-vector.hh>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <type_traits>
#include "test_utils.hh"
#include "../src/sycl/trsm_native.hh"

using namespace batchlas;

template <typename T, Backend B>
struct TestConfig {
    using ScalarType = T;
    static constexpr Backend BackendVal = B;
};

using TrsmTestTypes = typename test_utils::backend_types<TestConfig>::type;

template<typename Config>
class TrsmOperationsTest : public test_utils::BatchLASTest<Config> {
protected:
    using ScalarType = typename Config::ScalarType;
    static constexpr Backend BackendType = Config::BackendVal;
    
    const int rows = 8;
    const int cols = 8;
    const int ld = 8;
    const int batch_size = 3;
    const ScalarType alpha = static_cast<ScalarType>(1.0);

    void SetUp() override {
        test_utils::BatchLASTest<Config>::SetUp();
    }
    
    bool verifyTrsmResult(const MatrixView<ScalarType, MatrixFormat::Dense>& A,
                          const MatrixView<ScalarType, MatrixFormat::Dense>& B,
                          const MatrixView<ScalarType, MatrixFormat::Dense>& B_original,
                          int batch_idx,
                          Transpose trans = Transpose::NoTrans) {
        const bool trace_enabled = []() {
            const char* v = std::getenv("BATCHLAS_TRSM_TRACE");
            if (!v) return false;
            return (std::string(v) == "1" || std::string(v) == "true" || std::string(v) == "TRUE" ||
                    std::string(v) == "on" || std::string(v) == "ON");
        }();

        bool anyChanges = false;
        for (int i = 0; i < rows && !anyChanges; ++i) {
            for (int j = 0; j < cols && !anyChanges; ++j) {
                if (std::abs(B.at(i, j, batch_idx) - B_original.at(i, j, batch_idx)) > test_utils::tolerance<ScalarType>()) {
                    anyChanges = true;
                }
            }
        }
        
        if (!anyChanges) {
            if (trace_enabled) {
                std::cerr << "TRSM TRACE: output appears unchanged for batch " << batch_idx
                          << " (trans=" << static_cast<int>(trans) << ")" << std::endl;
                for (int k = 0; k < std::min(rows * cols, 4); ++k) {
                    int col = k / rows;
                    int row = k % rows;
                    std::cerr << "  B[" << k << "]=" << B.at(row, col, batch_idx)
                              << " (orig=" << B_original.at(row, col, batch_idx) << ")" << std::endl;
                }
            }
            return false;
        }
        
        bool allMatch = true;
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                ScalarType expected = B_original.at(i, j, batch_idx);
                ScalarType calculated = static_cast<ScalarType>(0.0);
                
                for (int k = 0; k < cols; ++k) {
                    int a_row = (trans == Transpose::NoTrans) ? i : k;
                    int a_col = (trans == Transpose::NoTrans) ? k : i;
                    calculated += A.at(a_row, a_col, batch_idx) * B.at(k, j, batch_idx);
                }
                
                auto tolerance = test_utils::tolerance<ScalarType>();
                if (std::abs(calculated - expected) > tolerance) {
                    if (trace_enabled) {
                        std::cerr << "TRSM TRACE: mismatch at (i=" << i << ", j=" << j << ") batch=" << batch_idx
                                  << " (trans=" << static_cast<int>(trans) << ")\n"
                                  << "  expected=" << expected << "\n"
                                  << "  calculated=" << calculated << "\n"
                                  << "  |diff|=" << std::abs(calculated - expected) << " tol=" << tolerance
                                  << std::endl;
                    }
                    allMatch = false;
                    break;
                }
            }
            if (!allMatch) break;
        }
        return allMatch;
    }
    
    void performTrsmTest(Uplo uplo, Transpose trans, int test_batch_size = 1) {
        // Filled on the host to avoid device-side state and kernel-ordering issues.
        Matrix<ScalarType, MatrixFormat::Dense> A_matrix(rows, rows, test_batch_size);
        Matrix<ScalarType, MatrixFormat::Dense> B_matrix(rows, cols, test_batch_size);

        std::mt19937 rng(42);
        std::uniform_real_distribution<batchlas::float_t<ScalarType>> dist(-1.0, 1.0);

        auto A_view_full = A_matrix.view();
        auto B_view_full = B_matrix.view();

        for (int b = 0; b < test_batch_size; ++b) {
            for (int j = 0; j < rows; ++j) {
                for (int i = 0; i < rows; ++i) {
                    if (i == j) {
                        A_view_full.at(i, j, b) = static_cast<ScalarType>(1.0);
                    } else if ((uplo == Uplo::Lower && i > j) || (uplo == Uplo::Upper && i < j)) {
                        A_view_full.at(i, j, b) = static_cast<ScalarType>(0.5);
                    } else {
                        A_view_full.at(i, j, b) = static_cast<ScalarType>(0.0);
                    }
                }
            }

            for (int j = 0; j < cols; ++j) {
                for (int i = 0; i < rows; ++i) {
                    if constexpr (std::is_same_v<ScalarType, std::complex<float>> ||
                                  std::is_same_v<ScalarType, std::complex<double>>) {
                        B_view_full.at(i, j, b) = ScalarType(dist(rng), dist(rng));
                    } else {
                        B_view_full.at(i, j, b) = static_cast<ScalarType>(dist(rng));
                    }
                }
            }
        }
        
        auto B_original = B_matrix.clone();

        if (test_batch_size == 1) {
            auto A_view = A_matrix.view();
            auto B_view = B_matrix.view();
            
            try {
                trsm(*(this->ctx),
                                  A_view,
                                  B_view,
                                  {.alpha = alpha, .uplo = uplo, .trans = trans});
                this->ctx->wait();
            } catch(const std::exception& e) {
                FAIL() << "TRSM operation failed with exception: " << e.what();
            }
        } else {
            auto A_parent_view = A_matrix.view();
            auto B_parent_view = B_matrix.view();
            
            for (int b = 0; b < test_batch_size; ++b) {
                auto A_view = A_parent_view.batch_item(b);
                auto B_view = B_parent_view.batch_item(b);
                
                try {
                    trsm(*(this->ctx),
                                      A_view,
                                      B_view,
                                      {.alpha = alpha, .uplo = uplo, .trans = trans});
                } catch(const std::exception& e) {
                    FAIL() << "TRSM operation failed for batch " << b << " with exception: " << e.what();
                }
            }
            this->ctx->wait();
        }

        auto A_view = A_matrix.view();
        auto B_view = B_matrix.view();
        auto B_original_view = B_original.view();
        for (int b = 0; b < test_batch_size; ++b) {
            EXPECT_TRUE(verifyTrsmResult(A_view, B_view, B_original_view, b, trans))
                << "TRSM solution verification failed for batch " << b;
        }
    }
};

TYPED_TEST_SUITE(TrsmOperationsTest, TrsmTestTypes);

TYPED_TEST(TrsmOperationsTest, LowerTriangularSolveNoTrans) {
    this->performTrsmTest(Uplo::Lower, Transpose::NoTrans, 1);
}

TYPED_TEST(TrsmOperationsTest, LowerTriangularSolveTrans) {
    this->performTrsmTest(Uplo::Lower, Transpose::Trans, 1);
}

TYPED_TEST(TrsmOperationsTest, UpperTriangularSolveNoTrans) {
    this->performTrsmTest(Uplo::Upper, Transpose::NoTrans, 1);
}

TYPED_TEST(TrsmOperationsTest, UpperTriangularSolveTrans) {
    this->performTrsmTest(Uplo::Upper, Transpose::Trans, 1);
}

TYPED_TEST(TrsmOperationsTest, BatchedLowerTriangularSolveNoTrans) {
    this->performTrsmTest(Uplo::Lower, Transpose::NoTrans, this->batch_size);
}

TYPED_TEST(TrsmOperationsTest, BatchedLowerTriangularSolveTrans) {
    this->performTrsmTest(Uplo::Lower, Transpose::Trans, this->batch_size);
}

TYPED_TEST(TrsmOperationsTest, BatchedUpperTriangularSolveNoTrans) {
    this->performTrsmTest(Uplo::Upper, Transpose::NoTrans, this->batch_size);
}

TYPED_TEST(TrsmOperationsTest, BatchedUpperTriangularSolveTrans) {
    this->performTrsmTest(Uplo::Upper, Transpose::Trans, this->batch_size);
}

// ===========================================================================
// The native CTA kernel (V1), called directly rather than through the facade.
//
// The oracle is an independent multiply-back, NOT a comparison against
// batchlas::trsm: the vendor backends perform the same canonical fold, so a
// kernel reproducing a shared fold error would agree with both of them.
// evidence: docs/perf/trsm.md#design-v1-v2-and-the-canonical-fold
// ===========================================================================
namespace {

// Compiles for real T too; the drivers below are instantiated for float and
// double, where a bare std::conj would not build.
template <typename T>
inline T host_conj(const T& v) {
    if constexpr (batchlas::is_std_complex_v<T>) {
        return std::conj(v);
    } else {
        return v;
    }
}

// Must stay non-real, non-symmetric and non-Hermitian: a real-valued complex
// triangle hides a missing conjugation, and a symmetric or Hermitian one hides
// a Trans/ConjTrans confusion.
template <typename T>
inline T tri_fill(int r, int c, bool diagonal) {
    using R = batchlas::float_t<T>;
    const R re = diagonal ? static_cast<R>(2 + (r % 3))
                          : static_cast<R>(0.02 * (1 + ((r * 7 + c * 3) % 5)));
    if constexpr (batchlas::is_std_complex_v<T>) {
        const R im = diagonal ? static_cast<R>(0.5 + 0.25 * (r % 2))
                              : static_cast<R>(0.013 * (1 + ((r * 3 + c * 11) % 7)));
        return T(re, im);
    } else {
        return T(re);
    }
}

template <typename T>
inline T rhs_fill(int r, int c) {
    using R = batchlas::float_t<T>;
    const R re = static_cast<R>(0.25 * (1 + ((r * 5 + c * 11) % 9)));
    if constexpr (batchlas::is_std_complex_v<T>) {
        return T(re, static_cast<R>(0.17 * (1 + ((r * 2 + c * 7) % 6))));
    } else {
        return T(re);
    }
}

template <typename T>
struct TrsmNativeCase {
    int n;
    int q;
    int batch;
    Side side;
    Uplo uplo;
    Transpose transA;
    Diag diag;
    T alpha;
};

template <typename T>
void RunTrsmNative(const TrsmNativeCase<T>& tc) {
    auto ctx = std::make_shared<Queue>(Device("gpu"), Backend::CUDA);

    const int n = tc.n, q = tc.q, bs = tc.batch;
    Matrix<T, MatrixFormat::Dense> A(n, n, bs);
    // Side::Right solves X op(A) = alpha B with B q x n; Side::Left solves
    // op(A) X = alpha B with B n x q.
    const int brows = (tc.side == Side::Left) ? n : q;
    const int bcols = (tc.side == Side::Left) ? q : n;
    Matrix<T, MatrixFormat::Dense> B(brows, bcols, bs);
    auto Av = A.view();
    auto Bv = B.view();

    // Deliberately not symmetric: a swapped operand order on Side::Right is
    // invisible on a symmetric triangle.
    std::vector<T> a_host(static_cast<size_t>(n) * n * bs);
    std::vector<T> b_in(static_cast<size_t>(brows) * bcols * bs);
    for (int b = 0; b < bs; ++b) {
        for (int c = 0; c < n; ++c) {
            for (int r = 0; r < n; ++r) {
                const bool in_tri = (tc.uplo == Uplo::Lower) ? (r >= c) : (r <= c);
                T v = (r == c) ? tri_fill<T>(r, c, true)
                               : (in_tri ? tri_fill<T>(r, c, false) : T(0));
                Av.at(r, c, b) = v;
                a_host[(static_cast<size_t>(b) * n + c) * n + r] = v;
            }
        }
        for (int c = 0; c < bcols; ++c) {
            for (int r = 0; r < brows; ++r) {
                const T v = rhs_fill<T>(r, c);
                Bv.at(r, c, b) = v;
                b_in[(static_cast<size_t>(b) * bcols + c) * brows + r] = v;
            }
        }
    }

    batchlas::sycl_trsm::trsm_native_v1_dispatch<T>(
        *ctx, A.view(), B.view(), tc.alpha, tc.side, tc.uplo, tc.transA, tc.diag);
    ctx->wait();

    // A real accumulator would drop the imaginary part and pass on wrong answers.
    using Acc = std::conditional_t<batchlas::is_std_complex_v<T>, std::complex<double>, double>;
    // float_t<T>, not T: is_same_v<T,float> is false for std::complex<float>,
    // which would judge a single-precision solve at the double tolerance.
    const double tol = std::is_same_v<batchlas::float_t<T>, float> ? 2e-3 : 1e-10;

    for (int b = 0; b < bs; ++b) {
        std::vector<T> opA(static_cast<size_t>(n) * n, T(0));
        for (int c = 0; c < n; ++c) {
            for (int r = 0; r < n; ++r) {
                const bool in_tri = (tc.uplo == Uplo::Lower) ? (r >= c) : (r <= c);
                if (!in_tri) continue;
                T v = a_host[(static_cast<size_t>(b) * n + c) * n + r];
                if (tc.diag == Diag::Unit && r == c) v = static_cast<T>(1);
                // NoTrans -> A(i,j); Trans -> A(j,i); ConjTrans -> conj(A(j,i)),
                // from the definition, so it cannot share the kernel's fold error.
                if (tc.transA == Transpose::NoTrans) {
                    opA[static_cast<size_t>(r) + static_cast<size_t>(c) * n] = v;
                } else {
                    opA[static_cast<size_t>(c) + static_cast<size_t>(r) * n] =
                        (tc.transA == Transpose::ConjTrans) ? host_conj<T>(v) : v;
                }
            }
        }
        for (int r = 0; r < brows; ++r) {
            for (int c = 0; c < bcols; ++c) {
                Acc got = Acc(0);
                for (int t = 0; t < n; ++t) {
                    // Left : (op(A) X)(r,c) = sum_t opA(r,t) * X(t,c)
                    // Right: (X op(A))(r,c) = sum_t X(r,t) * opA(t,c)
                    got += (tc.side == Side::Left)
                               ? Acc(opA[static_cast<size_t>(r) + static_cast<size_t>(t) * n]) *
                                     Acc(Bv.at(t, c, b))
                               : Acc(Bv.at(r, t, b)) *
                                     Acc(opA[static_cast<size_t>(t) + static_cast<size_t>(c) * n]);
                }
                const Acc want =
                    Acc(tc.alpha) * Acc(b_in[(static_cast<size_t>(b) * bcols + c) * brows + r]);
                ASSERT_LE(std::abs(got - want), tol)
                    << (tc.side == Side::Left ? "op(A)*X != alpha*B at b=" : "X*op(A) != alpha*B at b=") << b << " r=" << r << " c=" << c
                    << "  n=" << n << " q=" << q
                    << "  uplo=" << int(tc.uplo) << " transA=" << int(tc.transA)
                    << " diag=" << int(tc.diag);
            }
        }
    }
}

}  // namespace

// The cross product the kernel folds into one recurrence; folding it wrongly is
// the failure mode the design is most exposed to.
// evidence: docs/perf/trsm.md#design-v1-v2-and-the-canonical-fold
TEST(TrsmNativeCta, CanonicalCrossProductFloat) {
    for (Side sd : {Side::Left, Side::Right})
        for (Uplo up : {Uplo::Lower, Uplo::Upper})
            for (Transpose tr : {Transpose::NoTrans, Transpose::Trans})
                for (Diag dg : {Diag::NonUnit, Diag::Unit})
                    RunTrsmNative<float>({8, 24, 3, sd, up, tr, dg, 1.0f});
}

TEST(TrsmNativeCta, CanonicalCrossProductDouble) {
    for (Side sd : {Side::Left, Side::Right})
        for (Uplo up : {Uplo::Lower, Uplo::Upper})
            for (Transpose tr : {Transpose::NoTrans, Transpose::Trans})
                for (Diag dg : {Diag::NonUnit, Diag::Unit})
                    RunTrsmNative<double>({8, 24, 3, sd, up, tr, dg, 1.0});
}

// alpha scales B once, before the subtraction, not after the divide.
TEST(TrsmNativeCta, AlphaIsAppliedOnce) {
    for (Side sd : {Side::Left, Side::Right}) {
        RunTrsmNative<double>({16, 40, 2, sd, Uplo::Lower, Transpose::NoTrans, Diag::NonUnit, -2.5});
        RunTrsmNative<double>({16, 40, 2, sd, Uplo::Upper, Transpose::Trans, Diag::NonUnit, 0.75});
    }
}

// Rows n..N-1 of a partly filled bucket must contribute nothing; that is what
// makes the fully unrolled loop legal.
TEST(TrsmNativeCta, PartialBucketIsZeroPadded) {
    for (Side sd : {Side::Left, Side::Right})
        for (int n : {5, 9, 13, 17, 30})
            RunTrsmNative<double>({n, 33, 2, sd, Uplo::Lower, Transpose::NoTrans, Diag::NonUnit, 1.0});
}

// With q not a multiple of the work-group size, the tail lanes must not store.
TEST(TrsmNativeCta, RaggedRhsCount) {
    for (Side sd : {Side::Left, Side::Right})
        for (int q : {1, 7, 31, 33, 129, 257})
            RunTrsmNative<double>({8, q, 2, sd, Uplo::Upper, Transpose::NoTrans, Diag::NonUnit, 1.0});
}

// n=32 is the largest order that keeps x[] in registers; there is no N=64
// bucket, and n > 32 is V2's job.
// evidence: docs/perf/trsm.md#the-register-gate-and-the-cta-capacity
TEST(TrsmNativeCta, LargestResidentOrder) {
    for (Side sd : {Side::Left, Side::Right}) {
        RunTrsmNative<float>({32, 64, 2, sd, Uplo::Lower, Transpose::NoTrans, Diag::NonUnit, 1.0f});
        RunTrsmNative<double>({32, 64, 2, sd, Uplo::Upper, Transpose::Trans, Diag::NonUnit, 1.0});
    }
}


// V1's contract, n <= trsm_cta_max_n<T>(), is enforced rather than assumed: an
// over-capacity order once truncated to the leading 32x32 solve in silence.
// evidence: docs/perf/trsm.md#the-bucket-ladder-that-truncated
TEST(TrsmNativeCta, OverCapacityThrowsRatherThanTruncating) {
    auto ctx = std::make_shared<Queue>(Device("gpu"), Backend::CUDA);
    const int n = 33, q = 8, bs = 1;
    Matrix<double, MatrixFormat::Dense> A(n, n, bs);
    Matrix<double, MatrixFormat::Dense> B(q, n, bs);
    auto Av = A.view();
    auto Bv = B.view();
    for (int c = 0; c < n; ++c)
        for (int r = 0; r < n; ++r)
            Av.at(r, c, 0) = (r == c) ? 2.0 : (r > c ? 0.05 : 0.0);
    for (int c = 0; c < n; ++c)
        for (int r = 0; r < q; ++r)
            Bv.at(r, c, 0) = 1.0;

    EXPECT_THROW(
        (batchlas::sycl_trsm::trsm_native_v1_dispatch<double>(
            *ctx, A.view(), B.view(), 1.0, Side::Right, Uplo::Lower,
            Transpose::NoTrans, Diag::NonUnit)),
        std::runtime_error)
        << "n=33 exceeds the CTA capacity; silently solving 32 of 33 rows is the "
           "failure mode this guards";
}


// ===========================================================================
// V2, the blocked driver: the same multiply-back oracle, with n past CTA capacity.
// ===========================================================================
namespace {
template <typename T>
void RunTrsmBlocked(const TrsmNativeCase<T>& tc) {
    auto ctx = std::make_shared<Queue>(Device("gpu"), Backend::CUDA);
    const int n = tc.n, q = tc.q, bs = tc.batch;
    const int brows = (tc.side == Side::Left) ? n : q;
    const int bcols = (tc.side == Side::Left) ? q : n;
    Matrix<T, MatrixFormat::Dense> A(n, n, bs);
    Matrix<T, MatrixFormat::Dense> B(brows, bcols, bs);
    auto Av = A.view();
    auto Bv = B.view();
    std::vector<T> a_host(static_cast<size_t>(n) * n * bs);
    std::vector<T> b_in(static_cast<size_t>(brows) * bcols * bs);
    for (int b = 0; b < bs; ++b) {
        for (int c = 0; c < n; ++c)
            for (int r = 0; r < n; ++r) {
                const bool in_tri = (tc.uplo == Uplo::Lower) ? (r >= c) : (r <= c);
                T v = (r == c) ? tri_fill<T>(r, c, true)
                               : (in_tri ? tri_fill<T>(r, c, false) : T(0));
                Av.at(r, c, b) = v;
                a_host[(static_cast<size_t>(b) * n + c) * n + r] = v;
            }
        for (int c = 0; c < bcols; ++c)
            for (int r = 0; r < brows; ++r) {
                const T v = rhs_fill<T>(r, c);
                Bv.at(r, c, b) = v;
                b_in[(static_cast<size_t>(b) * bcols + c) * brows + r] = v;
            }
    }
    batchlas::sycl_trsm::trsm_native_blocked<T>(
        *ctx, A.view(), B.view(), tc.alpha, tc.side, tc.uplo, tc.transA, tc.diag);
    ctx->wait();

    // A real accumulator would drop the imaginary part and pass on wrong answers.
    using Acc = std::conditional_t<batchlas::is_std_complex_v<T>, std::complex<double>, double>;
    const double tol = std::is_same_v<batchlas::float_t<T>, float> ? 5e-3 : 1e-9;
    for (int b = 0; b < bs; ++b) {
        std::vector<T> opA(static_cast<size_t>(n) * n, T(0));
        for (int c = 0; c < n; ++c)
            for (int r = 0; r < n; ++r) {
                const bool in_tri = (tc.uplo == Uplo::Lower) ? (r >= c) : (r <= c);
                if (!in_tri) continue;
                T v = a_host[(static_cast<size_t>(b) * n + c) * n + r];
                if (tc.diag == Diag::Unit && r == c) v = static_cast<T>(1);
                // NoTrans -> A(i,j); Trans -> A(j,i); ConjTrans -> conj(A(j,i)),
                // from the definition, so it cannot share the kernel's fold error.
                if (tc.transA == Transpose::NoTrans) {
                    opA[static_cast<size_t>(r) + static_cast<size_t>(c) * n] = v;
                } else {
                    opA[static_cast<size_t>(c) + static_cast<size_t>(r) * n] =
                        (tc.transA == Transpose::ConjTrans) ? host_conj<T>(v) : v;
                }
            }
        for (int r = 0; r < brows; ++r)
            for (int c = 0; c < bcols; ++c) {
                Acc got = Acc(0);
                for (int t = 0; t < n; ++t)
                    got += (tc.side == Side::Left)
                               ? Acc(opA[static_cast<size_t>(r) + static_cast<size_t>(t) * n]) *
                                     Acc(Bv.at(t, c, b))
                               : Acc(Bv.at(r, t, b)) *
                                     Acc(opA[static_cast<size_t>(t) + static_cast<size_t>(c) * n]);
                const Acc want =
                    Acc(tc.alpha) * Acc(b_in[(static_cast<size_t>(b) * bcols + c) * brows + r]);
                ASSERT_LE(std::abs(got - want), tol)
                    << "blocked: b=" << b << " r=" << r << " c=" << c << " n=" << n
                    << " side=" << int(tc.side) << " uplo=" << int(tc.uplo)
                    << " transA=" << int(tc.transA) << " diag=" << int(tc.diag)
                    << " |alpha|=" << std::abs(tc.alpha);
            }
    }
}
}  // namespace

// Guards the group barrier between V1's SLM staging loop and the reciprocal
// loop that reads another lane's write. Without it the answers are wrong, but
// only when the work-group ladder picks more than one sub-group: every other
// case in this file lands on wg=32, a single sub-group in lock step, where the
// race cannot express itself. Clearing the ladder is necessary but NOT
// sufficient to reproduce -- the shape below is the one that does.
// evidence: docs/perf/trsm.md#the-missing-group-barrier
namespace {
int trsm_expected_wg(const Queue& ctx, int q, int bs) {
    const auto dev = ctx.device();
    const int max_wg = static_cast<int>(dev.get_property(DeviceProperty::MAX_WORK_GROUP_SIZE));
    const int cu = static_cast<int>(dev.get_property(DeviceProperty::MAX_COMPUTE_UNITS));
    int wg = 32;
    for (int cand : {256, 128, 64, 32}) {
        if (cand > max_wg) continue;
        wg = cand;
        const int64_t groups_c = (q + cand - 1) / cand;
        if (static_cast<int64_t>(bs) * groups_c >= static_cast<int64_t>(4) * cu) break;
    }
    return wg;
}
}  // namespace

TEST(TrsmNativeBlocked, MultiSubGroupWorkGroupStagesItsTriangleCorrectly) {
    auto probe = std::make_shared<Queue>(Device("gpu"), Backend::CUDA);
    const int q = 976, bs = 128;
    ASSERT_GT(trsm_expected_wg(*probe, q, bs), 32)
        << "this device's ladder still picks a single sub-group at q=" << q
        << " batch=" << bs << ", so this test cannot see the defect it exists "
           "for; raise q or batch";

    // Order 48 is one full N=32 block plus a ragged N=16 one; orders that
    // divide evenly into the CTA capacity do not reproduce the race.
    RunTrsmBlocked<float>({48, q, bs, Side::Right, Uplo::Lower,
                           Transpose::Trans, Diag::NonUnit, 1.0f});
    RunTrsmBlocked<double>({48, q, bs, Side::Right, Uplo::Lower,
                            Transpose::Trans, Diag::NonUnit, 1.0});
}

TEST(TrsmNativeBlocked, CrossoverAndBlockStructure) {
    for (Side sd : {Side::Left, Side::Right})
        for (int n : {33, 40, 64, 96, 100})
            RunTrsmBlocked<double>({n, 24, 2, sd, Uplo::Lower, Transpose::NoTrans,
                                    Diag::NonUnit, 1.0});
}

// For blocks i>0 alpha rides the trailing GEMM's beta, not V1. The natural
// beta = 1 is correct at block 0 and wrong at every later one, and no alpha == 1
// test can see it.
TEST(TrsmNativeBlocked, AlphaIsAppliedExactlyOncePerBlock) {
    for (Side sd : {Side::Left, Side::Right})
        for (double a : {-2.5, 0.75, 3.0})
            for (int n : {33, 64, 96})
                RunTrsmBlocked<double>({n, 20, 2, sd, Uplo::Upper, Transpose::NoTrans,
                                        Diag::NonUnit, a});
}

TEST(TrsmNativeBlocked, CanonicalCrossProduct) {
    for (Side sd : {Side::Left, Side::Right})
        for (Uplo up : {Uplo::Lower, Uplo::Upper})
            for (Transpose tr : {Transpose::NoTrans, Transpose::Trans})
                for (Diag dg : {Diag::NonUnit, Diag::Unit})
                    RunTrsmBlocked<double>({70, 16, 2, sd, up, tr, dg, -1.5});
}

TEST(TrsmNativeBlocked, FloatAndRaggedRhs) {
    for (Side sd : {Side::Left, Side::Right})
        for (int q : {1, 33, 129})
            RunTrsmBlocked<float>({48, q, 2, sd, Uplo::Lower, Transpose::Trans,
                                   Diag::NonUnit, 2.0f});
}


// ===========================================================================
// Complex. Two paths reach nothing above, and each is a silent wrong answer:
// ConjTrans, which is identical to Trans for a real scalar, and the complex
// reciprocal, which must be the overflow-safe Smith form -- the textbook
// conj(d)/|d|^2 returns 0 for inputs whose true reciprocal is representable.
// ===========================================================================

TEST(TrsmNativeCta, ComplexCanonicalCrossProductFloat) {
    for (Side sd : {Side::Left, Side::Right})
        for (Uplo up : {Uplo::Lower, Uplo::Upper})
            for (Transpose tr : {Transpose::NoTrans, Transpose::Trans, Transpose::ConjTrans})
                for (Diag dg : {Diag::NonUnit, Diag::Unit})
                    RunTrsmNative<std::complex<float>>(
                        {8, 24, 3, sd, up, tr, dg, std::complex<float>(1.0f, 0.0f)});
}

TEST(TrsmNativeCta, ComplexCanonicalCrossProductDouble) {
    for (Side sd : {Side::Left, Side::Right})
        for (Uplo up : {Uplo::Lower, Uplo::Upper})
            for (Transpose tr : {Transpose::NoTrans, Transpose::Trans, Transpose::ConjTrans})
                for (Diag dg : {Diag::NonUnit, Diag::Unit})
                    RunTrsmNative<std::complex<double>>(
                        {8, 24, 3, sd, up, tr, dg, std::complex<double>(1.0, 0.0)});
}

// A real alpha cannot catch an error that drops the imaginary cross-terms of
// the alpha*B product.
TEST(TrsmNativeCta, ComplexAlphaHasImaginaryPart) {
    for (Side sd : {Side::Left, Side::Right})
        for (Transpose tr : {Transpose::NoTrans, Transpose::ConjTrans}) {
            RunTrsmNative<std::complex<double>>(
                {16, 40, 2, sd, Uplo::Lower, tr, Diag::NonUnit,
                 std::complex<double>(-1.25, 0.75)});
            RunTrsmNative<std::complex<float>>(
                {16, 40, 2, sd, Uplo::Upper, tr, Diag::NonUnit,
                 std::complex<float>(0.5f, -2.0f)});
        }
}

TEST(TrsmNativeCta, ComplexPartialBucketAndRaggedRhs) {
    for (Side sd : {Side::Left, Side::Right}) {
        for (int n : {5, 13, 17, 31, 32})
            RunTrsmNative<std::complex<double>>(
                {n, 33, 2, sd, Uplo::Lower, Transpose::ConjTrans, Diag::NonUnit,
                 std::complex<double>(1.0, 0.0)});
        for (int q : {1, 7, 33, 129})
            RunTrsmNative<std::complex<float>>(
                {8, q, 2, sd, Uplo::Upper, Transpose::ConjTrans, Diag::NonUnit,
                 std::complex<float>(1.0f, 0.0f)});
    }
}

// The blocked driver's trailing GEMM is complex here, and its beta carries a
// complex alpha.
TEST(TrsmNativeBlocked, ComplexCrossoverAndAlpha) {
    for (Side sd : {Side::Left, Side::Right}) {
        for (int n : {33, 64, 70, 96})
            RunTrsmBlocked<std::complex<double>>(
                {n, 20, 2, sd, Uplo::Lower, Transpose::ConjTrans, Diag::NonUnit,
                 std::complex<double>(-1.5, 0.5)});
        RunTrsmBlocked<std::complex<float>>(
            {48, 24, 2, sd, Uplo::Upper, Transpose::Trans, Diag::NonUnit,
             std::complex<float>(2.0f, -0.75f)});
    }
}

TEST(TrsmNativeBlocked, ComplexCanonicalCrossProduct) {
    for (Side sd : {Side::Left, Side::Right})
        for (Uplo up : {Uplo::Lower, Uplo::Upper})
            for (Transpose tr : {Transpose::NoTrans, Transpose::Trans, Transpose::ConjTrans})
                for (Diag dg : {Diag::NonUnit, Diag::Unit})
                    RunTrsmBlocked<std::complex<double>>(
                        {40, 16, 2, sd, up, tr, dg, std::complex<double>(1.0, -0.5)});
}

// ===========================================================================
// Two-level blocking. The trailing update runs at OUTER_NB (default 128), so
// every blocked test above -- all of which stop at order 100 -- is a single
// panel and never runs the outer level. These orders cross OUTER_NB.
// evidence: docs/perf/trsm.md#the-two-level-blocked-driver
// ===========================================================================

TEST(TrsmNativeBlocked, TwoLevelPanelStructure) {
    for (Side sd : {Side::Left, Side::Right})
        for (int n : {129, 256, 300, 384})
            RunTrsmBlocked<double>({n, 24, 2, sd, Uplo::Lower, Transpose::NoTrans,
                                    Diag::NonUnit, 1.0});
}

// A different bug from the inner-level one: a block in panel p > 0 is touched by
// the outer gemm's beta, then an inner gemm's beta, then the solve's alpha --
// three chances to apply alpha and exactly one of them is right.
TEST(TrsmNativeBlocked, AlphaIsAppliedExactlyOnceAcrossPanels) {
    for (Side sd : {Side::Left, Side::Right})
        for (double a : {-2.5, 0.75})
            for (int n : {129, 256, 300})
                RunTrsmBlocked<double>({n, 20, 2, sd, Uplo::Upper, Transpose::NoTrans,
                                        Diag::NonUnit, a});
}

TEST(TrsmNativeBlocked, TwoLevelCanonicalCrossProduct) {
    for (Side sd : {Side::Left, Side::Right})
        for (Uplo up : {Uplo::Lower, Uplo::Upper})
            for (Transpose tr : {Transpose::NoTrans, Transpose::Trans})
                for (Diag dg : {Diag::NonUnit, Diag::Unit})
                    RunTrsmBlocked<double>({160, 16, 2, sd, up, tr, dg, -1.5});
}

TEST(TrsmNativeBlocked, TwoLevelFloatAndComplex) {
    for (Side sd : {Side::Left, Side::Right}) {
        RunTrsmBlocked<float>({192, 24, 2, sd, Uplo::Lower, Transpose::Trans,
                               Diag::NonUnit, 2.0f});
        RunTrsmBlocked<std::complex<float>>(
            {160, 20, 2, sd, Uplo::Lower, Transpose::ConjTrans, Diag::NonUnit,
             std::complex<float>(1.5f, -0.5f)});
    }
}

// OUTER_NB = 32 collapses the driver back to the single-level schedule, which
// must still be correct. evidence: docs/perf/trsm.md#tuning-knobs-and-environment
TEST(TrsmNativeBlocked, OuterBlockKnobIsHonouredAndAlwaysCorrect) {
    struct EnvGuard {
        const char* key;
        std::string saved;
        bool had;
        EnvGuard(const char* k, const char* v) : key(k) {
            const char* old = std::getenv(k);
            had = old != nullptr;
            if (had) saved = old;
            setenv(k, v, 1);
        }
        ~EnvGuard() { had ? setenv(key, saved.c_str(), 1) : unsetenv(key); }
    };
    // trsm_outer_block caches the parse in a function-local static, so the first
    // blocked call in the process fixes the value and setting it here may be a
    // no-op. Hence this asserts correctness under whatever value is live; a
    // schedule assertion would pass or fail on gtest's ordering, not the code.
    for (const char* v : {"64", "32", "256"}) {
        EnvGuard g("BATCHLAS_TRSM_OUTER_NB", v);
        for (Side sd : {Side::Left, Side::Right})
            RunTrsmBlocked<double>({200, 16, 2, sd, Uplo::Lower, Transpose::NoTrans,
                                    Diag::NonUnit, -1.25});
    }
}

// ===========================================================================
// float / Side::Left across the blocked range. Written for V3, a cooperative
// CTA solve that was implemented and then rejected on measurement; kept because
// nothing else covers float / Side::Left at this density of orders.
// evidence: docs/perf/trsm.md#rejected-the-cooperative-cta-solve-v3
// ===========================================================================

TEST(TrsmFloatLeftOrders, SpanningTheBlockedRange) {
    for (int n : {33, 40, 64, 100, 127, 128, 129, 200})
        RunTrsmBlocked<float>({n, 24, 2, Side::Left, Uplo::Lower, Transpose::NoTrans,
                               Diag::NonUnit, 1.0f});
}

TEST(TrsmFloatLeftOrders, CanonicalCrossProduct) {
    for (Uplo up : {Uplo::Lower, Uplo::Upper})
        for (Transpose tr : {Transpose::NoTrans, Transpose::Trans})
            for (Diag dg : {Diag::NonUnit, Diag::Unit})
                for (int n : {64, 129})
                    RunTrsmBlocked<float>({n, 20, 2, Side::Left, up, tr, dg, -1.5f});
}

TEST(TrsmFloatLeftOrders, AlphaAcrossOrders) {
    for (float a : {-2.5f, 0.75f, 3.0f})
        for (int n : {64, 129, 200})
            RunTrsmBlocked<float>({n, 20, 2, Side::Left, Uplo::Upper, Transpose::NoTrans,
                                   Diag::NonUnit, a});
}

// A q that is not a multiple of the work-group's solve count exercises the
// `live` guard on both the load and the store.
TEST(TrsmFloatLeftOrders, RaggedSolveCount) {
    for (int q : {1, 7, 31, 33, 129, 257})
        RunTrsmBlocked<float>({96, q, 2, Side::Left, Uplo::Lower, Transpose::Trans,
                               Diag::NonUnit, 1.25f});
}

// The two sides take different schedules (Side::Left blocks at 128,
// Side::Right at 32), which a later collapse to one constant would break.
TEST(TrsmFloatLeftOrders, RightSideAlso) {
    for (int n : {64, 129, 200})
        for (Transpose tr : {Transpose::NoTrans, Transpose::Trans})
            RunTrsmBlocked<float>({n, 24, 2, Side::Right, Uplo::Lower, tr,
                                   Diag::NonUnit, -0.75f});
}
