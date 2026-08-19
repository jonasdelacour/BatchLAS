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
                T v;
                const bool in_tri = (tc.uplo == Uplo::Lower) ? (r >= c) : (r <= c);
                if (r == c)        v = static_cast<T>(2 + (r % 3));
                else if (in_tri)   v = static_cast<T>(0.05 * (1 + ((r * 7 + c * 3) % 5)));
                else               v = static_cast<T>(0);
                Av.at(r, c, b) = v;
                a_host[(static_cast<size_t>(b) * n + c) * n + r] = v;
            }
        }
        for (int c = 0; c < bcols; ++c) {
            for (int r = 0; r < brows; ++r) {
                const T v = static_cast<T>(0.25 * (1 + ((r * 5 + c * 11) % 9)));
                Bv.at(r, c, b) = v;
                b_in[(static_cast<size_t>(b) * bcols + c) * brows + r] = v;
            }
        }
    }

    batchlas::sycl_trsm::trsm_native_v1_dispatch<T>(
        *ctx, A.view(), B.view(), tc.alpha, tc.side, tc.uplo, tc.transA, tc.diag);
    ctx->wait();

    using Acc = double;
    const Acc tol = std::is_same_v<T, float> ? Acc(2e-3) : Acc(1e-10);

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
                if (tc.transA == Transpose::NoTrans)
                    opA[static_cast<size_t>(r) + static_cast<size_t>(c) * n] = v;
                else
                    opA[static_cast<size_t>(c) + static_cast<size_t>(r) * n] = v;
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
                ASSERT_NEAR(got, want, tol)
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
