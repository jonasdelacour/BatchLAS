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
    
    // Verify that the TRSM solution satisfies A*X = B or A^T*X = B depending on transpose
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

        // First check if B was actually modified from original
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
        
        // Now verify each element of the result by checking AX = B_original or A^T*X = B_original
        bool allMatch = true;
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                ScalarType expected = B_original.at(i, j, batch_idx);
                ScalarType calculated = static_cast<ScalarType>(0.0);
                
                // Calculate the result of A*X or A^T*X for this position
                for (int k = 0; k < cols; ++k) {
                    int a_row = (trans == Transpose::NoTrans) ? i : k;
                    int a_col = (trans == Transpose::NoTrans) ? k : i;
                    calculated += A.at(a_row, a_col, batch_idx) * B.at(k, j, batch_idx);
                }
                
                // Use a reasonable tolerance for floating point comparisons
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
        // Create matrices and fill on host to avoid device-side state or kernel ordering issues
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
        
        // Keep original B for verification
        auto B_original = B_matrix.clone();

        // Create matrix views (matrices are already column-major)
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
            
            // Process each batch using batch_item
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

// Test TRSM operation with a lower triangular matrix (no transpose)
TYPED_TEST(TrsmOperationsTest, LowerTriangularSolveNoTrans) {
    this->performTrsmTest(Uplo::Lower, Transpose::NoTrans, 1);
}

// Test TRSM operation with a lower triangular matrix (transpose)
TYPED_TEST(TrsmOperationsTest, LowerTriangularSolveTrans) {
    this->performTrsmTest(Uplo::Lower, Transpose::Trans, 1);
}

// Test TRSM operation with an upper triangular matrix (no transpose)
TYPED_TEST(TrsmOperationsTest, UpperTriangularSolveNoTrans) {
    this->performTrsmTest(Uplo::Upper, Transpose::NoTrans, 1);
}

// Test TRSM operation with an upper triangular matrix (transpose)
TYPED_TEST(TrsmOperationsTest, UpperTriangularSolveTrans) {
    this->performTrsmTest(Uplo::Upper, Transpose::Trans, 1);
}

// Test batched TRSM operation with lower triangular (no transpose)
TYPED_TEST(TrsmOperationsTest, BatchedLowerTriangularSolveNoTrans) {
    this->performTrsmTest(Uplo::Lower, Transpose::NoTrans, this->batch_size);
}

// Test batched TRSM operation with lower triangular (transpose)
TYPED_TEST(TrsmOperationsTest, BatchedLowerTriangularSolveTrans) {
    this->performTrsmTest(Uplo::Lower, Transpose::Trans, this->batch_size);
}

// Test batched TRSM operation with upper triangular (no transpose)
TYPED_TEST(TrsmOperationsTest, BatchedUpperTriangularSolveNoTrans) {
    this->performTrsmTest(Uplo::Upper, Transpose::NoTrans, this->batch_size);
}

// Test batched TRSM operation with upper triangular (transpose)
TYPED_TEST(TrsmOperationsTest, BatchedUpperTriangularSolveTrans) {
    this->performTrsmTest(Uplo::Upper, Transpose::Trans, this->batch_size);
}

// ===========================================================================
// WP3 step 3 -- the native CTA kernel, Side::Right, called DIRECTLY.
//
// Nothing routes here yet (trsm_cta_max_n<T>() returns 0, so
// RouteTable<Op::trsm,T>::supports() reports both native routes unsupported).
// These call the kernel by hand so its correctness is settled before any
// routing decision depends on it.
//
// THE ORACLE IS AN INDEPENDENT MULTIPLY-BACK, not a comparison against
// batchlas::trsm. That is not fussiness: src/backends/netlib_lapack.cc:445-449
// and src/backends/cublas.cc:1134-1137 perform the SAME canonical fold, so they
// are one implementation with two spellings, and a kernel that reproduced a
// shared fold error would agree with both. Multiplying the answer back through
// op(A) and comparing to alpha*B tests the thing that actually matters.
//
// Side::Right solves X op(A) = alpha B, so the check is X op(A) == alpha B.
// ===========================================================================
namespace {


// Host-side conjugate that compiles for real T as well: the drivers below are
// instantiated for float and double too, so a bare std::conj would not build.
template <typename T>
inline T host_conj(const T& v) {
    if constexpr (batchlas::is_std_complex_v<T>) {
        return std::conj(v);
    } else {
        return v;
    }
}

// Test data. THE IMAGINARY PARTS ARE THE POINT: a missing conjugation is
// invisible on a real-valued complex matrix, and a Hermitian or symmetric
// triangle hides a transposed-vs-conjugate-transposed confusion as well. This
// fill is non-real, non-symmetric and non-Hermitian by construction -- the
// imaginary part is a different function of (r,c) than the real part, and
// neither is symmetric in r,c.
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

    // Well-conditioned triangle, and deliberately NOT symmetric: a swapped Lc
    // operand order (the Side::Right trap, spec section 2.1) is invisible on a
    // symmetric triangle and wrong on this one.
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

    // Accumulate complex in complex; a real accumulator would silently drop the
    // imaginary part of the product and the check would pass on wrong answers.
    using Acc = std::conditional_t<batchlas::is_std_complex_v<T>, std::complex<double>, double>;
    // float_t<T>, not T: std::is_same_v<T,float> is FALSE for
    // std::complex<float>, which would judge a single-precision solve at the
    // double tolerance. base_type is at include/batchlas/blas/enums.hh:19-27.
    // The tolerance is a MAGNITUDE, so it stays real even when Acc is complex.
    const double tol = std::is_same_v<batchlas::float_t<T>, float> ? 2e-3 : 1e-10;

    for (int b = 0; b < bs; ++b) {
        // op(A) built from the DEFINITION, with no reference to the kernel's
        // canonicalisation, so the two cannot share a fold error.
        std::vector<T> opA(static_cast<size_t>(n) * n, T(0));
        for (int c = 0; c < n; ++c) {
            for (int r = 0; r < n; ++r) {
                const bool in_tri = (tc.uplo == Uplo::Lower) ? (r >= c) : (r <= c);
                if (!in_tri) continue;
                T v = a_host[(static_cast<size_t>(b) * n + c) * n + r];
                if (tc.diag == Diag::Unit && r == c) v = static_cast<T>(1);
                // op(A)(i,j): NoTrans -> A(i,j); Trans -> A(j,i);
                // ConjTrans -> conj(A(j,i)). Built from the DEFINITION, so it
                // cannot share a fold error with the kernel's canonicalisation.
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

// The full canonical cross product: BOTH sides x uplo x transA x diag = 16
// cells per scalar type. This is the table WP3_TRSM_SPEC.md section 2.1 folds
// into one recurrence, and folding it wrongly is the failure mode the whole
// design is exposed to.
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

// alpha != 1 is easy to get wrong: it scales B, once, before the subtraction,
// not after the divide.
TEST(TrsmNativeCta, AlphaIsAppliedOnce) {
    for (Side sd : {Side::Left, Side::Right}) {
        RunTrsmNative<double>({16, 40, 2, sd, Uplo::Lower, Transpose::NoTrans, Diag::NonUnit, -2.5});
        RunTrsmNative<double>({16, 40, 2, sd, Uplo::Upper, Transpose::Trans, Diag::NonUnit, 0.75});
    }
}

// n strictly inside its bucket exercises the zero-padded tail: rows n..N-1 must
// contribute nothing, which is what makes the fully unrolled loop legal.
TEST(TrsmNativeCta, PartialBucketIsZeroPadded) {
    for (Side sd : {Side::Left, Side::Right})
        for (int n : {5, 9, 13, 17, 30})
            RunTrsmNative<double>({n, 33, 2, sd, Uplo::Lower, Transpose::NoTrans, Diag::NonUnit, 1.0});
}

// q not a multiple of the work-group size: the tail lanes must be inert, not
// merely harmless -- they must not store.
TEST(TrsmNativeCta, RaggedRhsCount) {
    for (Side sd : {Side::Left, Side::Right})
        for (int q : {1, 7, 31, 33, 129, 257})
            RunTrsmNative<double>({8, q, 2, sd, Uplo::Upper, Transpose::NoTrans, Diag::NonUnit, 1.0});
}

// The largest bucket the register probe cleared. n=32 keeps x[] in registers
// (114 float / 153 double registers, zero stack frame); n=64 does not, which is
// why there is no 64 bucket and why n > 32 is V2's job.
TEST(TrsmNativeCta, LargestResidentOrder) {
    for (Side sd : {Side::Left, Side::Right}) {
        RunTrsmNative<float>({32, 64, 2, sd, Uplo::Lower, Transpose::NoTrans, Diag::NonUnit, 1.0f});
        RunTrsmNative<double>({32, 64, 2, sd, Uplo::Upper, Transpose::Trans, Diag::NonUnit, 1.0});
    }
}


// V1's contract is n <= trsm_cta_max_n<T>() (32). The bucket ladder had a hole:
// smallest_bucket_ge(33) returned 64, which the dispatch switch's `default:`
// label collapsed onto the N=32 instantiation -- so a 33-order solve silently
// solved the leading 32x32 system and left the last row of B untouched. It was
// unreachable through the facade, because supports(CTA) caps the order, but the
// direct entry is what V2 will call on its diagonal blocks, so it was one step
// from being live.
//
// The contract is now enforced rather than assumed: over-capacity throws.
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
// V2, the blocked driver. Same independent multiply-back oracle as V1; the only
// difference is that n exceeds the CTA capacity, so the driver blocks.
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

    // Accumulate complex in complex; a real accumulator would silently drop the
    // imaginary part of the product and the check would pass on wrong answers.
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
                // op(A)(i,j): NoTrans -> A(i,j); Trans -> A(j,i);
                // ConjTrans -> conj(A(j,i)). Built from the DEFINITION, so it
                // cannot share a fold error with the kernel's canonicalisation.
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

// The crossover and the block structure. 33 is one past the capacity (two
// blocks, the second of width 1 -- the short-final-block case); 64 is exactly
// two full blocks; 96 is three; 100 is three plus a ragged 4.
TEST(TrsmNativeBlocked, CrossoverAndBlockStructure) {
    for (Side sd : {Side::Left, Side::Right})
        for (int n : {33, 40, 64, 96, 100})
            RunTrsmBlocked<double>({n, 24, 2, sd, Uplo::Lower, Transpose::NoTrans,
                                    Diag::NonUnit, 1.0});
}

// ALPHA != 1 IS THE TEST THAT MATTERS HERE. alpha is applied exactly once, and
// for blocks i>0 that happens through the trailing GEMM's BETA, not through V1.
// Writing the natural beta = 1 computes B_i - sum(...) where alpha*B_i - sum(...)
// is required: correct at block 0, wrong at every later block, and invisible to
// any alpha == 1 test -- which is every other test in this file.
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
// COMPLEX. Two things here are not exercised anywhere above, and each is a
// silent wrong answer if got wrong:
//
//   * ConjTrans. For a real scalar it is identical to Trans, so every existing
//     cell in this file is blind to it. Canonical::do_conj was written by
//     canonicalise() and read by nothing until complex arrived.
//   * The complex reciprocal. The real path divides by a scalar; complex needs
//     an overflow-safe reciprocal (Smith), and the textbook conj(d)/|d|^2 form
//     silently returns 0 for inputs whose true reciprocal is representable.
//
// The test DATA is what makes these visible. tri_fill gives every element a
// non-zero imaginary part that is a different function of (r,c) than the real
// part, so the triangle is neither real, nor symmetric, nor Hermitian. On a
// real-valued complex matrix a missing conj is invisible; on a Hermitian one a
// Trans/ConjTrans confusion is invisible.
// ===========================================================================

// All 24 canonical cells per type: 2 sides x 2 uplo x 3 transA (NoTrans, Trans,
// ConjTrans) x 2 diag, inside the CTA capacity.
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

// A COMPLEX alpha with a non-zero imaginary part. A real alpha cannot catch an
// error that drops the imaginary cross-terms of the alpha*B product.
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

// V2 with complex: the blocked driver's trailing GEMM is a complex GEMM, and
// its beta carries a complex alpha.
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
// TWO-LEVEL BLOCKING (WP3 step 13).
//
// V2's outer block width was decoupled from the CTA capacity: the trailing
// update now runs at OUTER_NB (default 128) and each panel is solved by the old
// nb = 32 loop against its own prefix.
//
// EVERY BLOCKED TEST ABOVE STOPS AT ORDER 100. With OUTER_NB = 128 that is a
// SINGLE panel, so all of them take LO == 0 and the outer level never runs --
// they passed unchanged against the two-level driver while proving nothing
// about it. These orders are chosen to cross OUTER_NB:
//
//   129  two panels, the second one element wide (short-final-panel)
//   256  exactly two full panels
//   300  two full panels plus a ragged 44
//   384  three full panels
//
// and BATCHLAS_TRSM_OUTER_NB is exercised too, because a tuning knob nobody
// tests is a tuning knob that silently stops working.
// ===========================================================================

TEST(TrsmNativeBlocked, TwoLevelPanelStructure) {
    for (Side sd : {Side::Left, Side::Right})
        for (int n : {129, 256, 300, 384})
            RunTrsmBlocked<double>({n, 24, 2, sd, Uplo::Lower, Transpose::NoTrans,
                                    Diag::NonUnit, 1.0});
}

// THE ALPHA TEST FOR THE OUTER LEVEL, and it is a different bug from the inner
// one. With two levels a block in panel p > 0 is touched by the OUTER gemm
// (beta), then by an inner gemm (beta), then by the solve (alpha) -- three
// chances to apply alpha and exactly one of them is right. The inner-level
// version of this test (AlphaIsAppliedExactlyOncePerBlock) cannot see it: at
// order <= 100 there is only ever one panel.
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

// The knob itself. OUTER_NB = 64 puts four panels into an order that the
// default would cover in two, and OUTER_NB = 32 collapses the driver back to
// the original single-level schedule -- which must still be correct, since it
// is what shipped before this change.
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
    // NOTE: trsm_outer_block caches the parse in a function-local static, so the
    // FIRST blocked call in this process fixes the value. Setting it here can
    // therefore be a no-op depending on test order -- which is precisely why
    // this test asserts CORRECTNESS under whatever value is live rather than
    // asserting a particular schedule. A schedule assertion would pass or fail
    // on gtest's ordering, not on the code.
    for (const char* v : {"64", "32", "256"}) {
        EnvGuard g("BATCHLAS_TRSM_OUTER_NB", v);
        for (Side sd : {Side::Left, Side::Right})
            RunTrsmBlocked<double>({200, 16, 2, sd, Uplo::Lower, Transpose::NoTrans,
                                    Diag::NonUnit, -1.25});
    }
}
