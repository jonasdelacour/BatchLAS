#include <gtest/gtest.h>
#include <batchlas/blas/linalg.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>
#include <batchlas/util/sycl-vector.hh>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <random>
#include <type_traits>
#include "test_utils.hh"
#include <batchlas/verify/residuals.hh>
#include <batchlas/verify/tolerance.hh>
#include <batchlas/util/env.hh>
#include <batchlas/settings.hh>
#include "../src/select/vendor.hh"
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
                (void)trsm(*(this->ctx),
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
                    (void)trsm(*(this->ctx),
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
// The oracle is an independent multiply-back, not a comparison against
// batchlas::trsm: the vendor backends perform the same canonical fold, so a
// kernel reproducing a shared fold error would agree with both of them.
// evidence: docs/perf/trsm.md#design-v1-v2-and-the-canonical-fold
// ===========================================================================
namespace {

// Also compiles for real T: the float and double drivers below would reject a
// bare std::conj.
template <typename T>
inline T host_conj(const T& v) {
    if constexpr (batchlas::is_std_complex_v<T>) {
        return std::conj(v);
    } else {
        return v;
    }
}

// Must stay non-real, non-symmetric and non-Hermitian: a real-valued complex
// triangle hides a missing conjugation, a symmetric or Hermitian one hides a
// Trans/ConjTrans confusion.
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
    // Side::Left solves op(A) X = alpha B (B is n x q); Side::Right, X op(A) = alpha B (B is q x n).
    const int brows = (tc.side == Side::Left) ? n : q;
    const int bcols = (tc.side == Side::Left) ? q : n;
    Matrix<T, MatrixFormat::Dense> B(brows, bcols, bs);
    auto Av = A.view();
    auto Bv = B.view();

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

    (void)batchlas::sycl_trsm::trsm_native_v1_dispatch<T>(
        *ctx, A.view(), B.view(), tc.alpha, tc.side, tc.uplo, tc.transA, tc.diag);
    ctx->wait();

    using MV = MatrixView<T, MatrixFormat::Dense>;
    const MV A0(a_host.data(), n, n, n, n * n, bs);
    const MV B0(b_in.data(), brows, bcols, brows, brows * bcols, bs);
    const double res = batchlas::verify::trsm_residual(A0, tc.side, tc.uplo, tc.transA, tc.diag, Bv, B0,
                                                       batchlas::verify::up(tc.alpha));
    ASSERT_TRUE(batchlas::verify::pass<T>(batchlas::verify::Check::solve, n, res))
        << "residual " << res << " exceeds " << batchlas::verify::bound<T>(batchlas::verify::Check::solve, n)
        << "  n=" << n << " q=" << q << " side=" << int(tc.side) << " uplo=" << int(tc.uplo)
        << " transA=" << int(tc.transA) << " diag=" << int(tc.diag);
}

}  // namespace

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

// n=32 is the largest order that keeps x[] in registers; n > 32 is V2's job.
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
        ((void)batchlas::sycl_trsm::trsm_native_v1_dispatch<double>(
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
    using MV = MatrixView<T, MatrixFormat::Dense>;
    (void)batchlas::sycl_trsm::trsm_native_blocked<T>(
        *ctx, A.view(), B.view(), tc.alpha, tc.side, tc.uplo, tc.transA, tc.diag,
        [](Queue& c, const MV& ga, const MV& gb, const MV& gc, T al, T be, Transpose ta, Transpose tb,
           ComputePrecision p) { return gemm<Backend::CUDA, T>(c, ga, gb, gc, al, be, ta, tb, p); });
    ctx->wait();

    const MV A0(a_host.data(), n, n, n, n * n, bs);
    const MV B0(b_in.data(), brows, bcols, brows, brows * bcols, bs);
    const double res = batchlas::verify::trsm_residual(A0, tc.side, tc.uplo, tc.transA, tc.diag, Bv, B0,
                                                       batchlas::verify::up(tc.alpha));
    ASSERT_TRUE(batchlas::verify::pass<T>(batchlas::verify::Check::solve, n, res))
        << "blocked: residual " << res << " exceeds " << batchlas::verify::bound<T>(batchlas::verify::Check::solve, n)
        << "  n=" << n << " q=" << q << " side=" << int(tc.side) << " uplo=" << int(tc.uplo)
        << " transA=" << int(tc.transA) << " diag=" << int(tc.diag);
}
}  // namespace

// Guards the group barrier between V1's SLM staging loop and the reciprocal loop
// that reads another lane's write. Without it the answers are wrong only when the
// work-group ladder picks more than one sub-group; every other case in this file
// lands on wg=32, one sub-group in lock step, where the race cannot express itself.
// evidence: docs/perf/trsm.md#the-missing-group-barrier
namespace {
int trsm_expected_wg(const Queue& ctx, int q, int bs) {
    const auto dev = ctx.device();
    const int max_wg = static_cast<int>(dev.get_property(DeviceProperty::MAX_WORK_GROUP_SIZE));
    const int cu = static_cast<int>(dev.get_property(DeviceProperty::MAX_COMPUTE_UNITS));
    return batchlas::sycl_trsm::trsm_v1_ladder_wg(max_wg, cu, q, bs);
}
}  // namespace

// The launcher calls this function, so these literals pin the ladder itself.
// evidence: docs/perf/blackwell.md#trsm-v1-ladder-cap
TEST(TrsmNativeCta, LadderRungsAreCappedByRhsCount) {
    using batchlas::sycl_trsm::trsm_v1_ladder_wg;
    // Saturated batch on 188 CUs: the cap alone decides the rung.
    EXPECT_EQ(trsm_v1_ladder_wg(1024, 188, 8, 4096), 32);
    EXPECT_EQ(trsm_v1_ladder_wg(1024, 188, 32, 4096), 32);
    EXPECT_EQ(trsm_v1_ladder_wg(1024, 188, 33, 4096), 64);
    EXPECT_EQ(trsm_v1_ladder_wg(1024, 188, 40, 4096), 64);
    EXPECT_EQ(trsm_v1_ladder_wg(1024, 188, 64, 4096), 64);
    EXPECT_EQ(trsm_v1_ladder_wg(1024, 188, 65, 4096), 128);
    EXPECT_EQ(trsm_v1_ladder_wg(1024, 188, 128, 4096), 128);
    EXPECT_EQ(trsm_v1_ladder_wg(1024, 188, 129, 4096), 256);
    EXPECT_EQ(trsm_v1_ladder_wg(1024, 188, 976, 4096), 256);
    // Unsaturated batch: the ladder keeps descending until 4*cu groups exist.
    EXPECT_EQ(trsm_v1_ladder_wg(1024, 188, 976, 128), 128);
    EXPECT_EQ(trsm_v1_ladder_wg(1024, 128, 976, 128), 256);
    EXPECT_EQ(trsm_v1_ladder_wg(1024, 188, 976, 1), 32);
    // A device work-group limit below a rung skips it.
    EXPECT_EQ(trsm_v1_ladder_wg(128, 188, 976, 4096), 128);
    EXPECT_EQ(trsm_v1_ladder_wg(64, 188, 976, 4096), 64);
}

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

// For blocks i>0 alpha rides the trailing GEMM's beta, not V1: the natural
// beta = 1 is correct at block 0 and wrong at every later one.
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
// Complex. Two paths reach nothing above, each a silent wrong answer: ConjTrans,
// identical to Trans for a real scalar, and the complex reciprocal, which must be
// the overflow-safe Smith form -- conj(d)/|d|^2 returns 0 for inputs whose true
// reciprocal is representable.
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

// A real alpha cannot catch an error that drops the imaginary cross-terms.
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

// One right-hand side makes every trailing update a complex<double> gemm with m or n == 1,
// which segfaulted inside cuBLASLt through cublasGemm*Ex (cublas.cc gemm_vendor_impl).
TEST(TrsmNativeBlocked, ComplexDoubleSingleRhsTrailingGemm) {
    for (Side sd : {Side::Left, Side::Right})
        for (Transpose t : {Transpose::NoTrans, Transpose::ConjTrans})
            RunTrsmBlocked<std::complex<double>>(
                {64, 1, 3, sd, Uplo::Lower, t, Diag::NonUnit, std::complex<double>(1.0, 0.5)});
}

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
// Two-level blocking. The trailing update runs at OUTER_NB (default 128), so the
// blocked tests above all stay inside one panel; these orders cross it.
// evidence: docs/perf/trsm.md#the-two-level-blocked-driver
// ===========================================================================

TEST(TrsmNativeBlocked, TwoLevelPanelStructure) {
    for (Side sd : {Side::Left, Side::Right})
        for (int n : {129, 256, 300, 384})
            RunTrsmBlocked<double>({n, 24, 2, sd, Uplo::Lower, Transpose::NoTrans,
                                    Diag::NonUnit, 1.0});
}

// Distinct from the inner-level bug: a block in panel p > 0 is touched by the
// outer gemm's beta, then an inner gemm's beta, then the solve's alpha.
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
    // The local EnvGuard this test carried was byte-identical to
    // batchlas::ScopedEnvVar (set with overwrite, restore-or-unset on exit) minus
    // the reload: settings() snapshots the environment once, so a raw setenv is
    // invisible to trsm_outer_block and all three arms below would have run the
    // live value three times and passed by construction.
    //
    // trsm_outer_block reads settings().geometry.trsm_outer_nb per call and is
    // deliberately not latched in a function-local static, so each arm really does
    // reach the driver: at nb = 32 these round to outer_nb 64, 32 and 256, i.e.
    // four panels, seven panels, and (256 > n = 200) a single panel that collapses
    // the outer level away. Correctness is the assertion; the schedule is not.
    for (const char* v : {"64", "32", "256"}) {
        batchlas::ScopedEnvVar g("BATCHLAS_TRSM_OUTER_NB", v);
        for (Side sd : {Side::Left, Side::Right})
            RunTrsmBlocked<double>({200, 16, 2, sd, Uplo::Lower, Transpose::NoTrans,
                                    Diag::NonUnit, -1.25});
    }
}

// ===========================================================================
// float / Side::Left across the blocked range -- nothing else covers that pair
// at this density of orders.
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


// ===========================================================================
// The Side::Left sub-group kernel: lane = (matrix, canonical row), 32/N matrices
// per sub-group, QC rhs per lane. Every case reads through a padded ld and batch
// stride, with large finite poison in the padding, in the unused triangle and, for
// Diag::Unit, on the diagonal: a kernel that derives ld or stride, or reads the
// wrong triangle, still returns finite numbers and fails the multiply-back.
// evidence: docs/perf/blackwell.md#trsm-sub-group-left-kernel
// ===========================================================================
namespace {

template <typename T>
struct SgCase {
    int n, q, batch;
    Uplo uplo;
    Transpose transA;
    Diag diag;
    T alpha;
    int pad_a = 3;         // lda = n + pad_a
    int pad_b = 5;         // ldb = n + pad_b
    bool same_items = false;
};

template <typename T>
std::vector<T> RunSgLeft(const SgCase<T>& tc, bool check = true) {
    auto ctx = std::make_shared<Queue>(Device("gpu"), Backend::CUDA);
    const int n = tc.n, q = tc.q, bs = tc.batch;
    const int lda = n + tc.pad_a, ldb = n + tc.pad_b;
    const int sa = lda * n + 7, sb = ldb * q + 11;
    const T poison = T(4096);
    UnifiedVector<T> abuf(static_cast<size_t>(sa) * bs, poison);
    UnifiedVector<T> bbuf(static_cast<size_t>(sb) * bs, poison);
    std::vector<T> b_in(static_cast<size_t>(n) * q * bs);
    for (int b = 0; b < bs; ++b) {
        const int item = tc.same_items ? 0 : b;
        for (int c = 0; c < n; ++c)
            for (int r = 0; r < n; ++r) {
                const bool in_tri = (tc.uplo == Uplo::Lower) ? (r > c) : (r < c);
                T v = poison;
                if (r == c && tc.diag == Diag::NonUnit) v = tri_fill<T>(r + item % 3, c, true);
                if (in_tri) v = tri_fill<T>(r + item % 5, c, false);
                abuf[static_cast<size_t>(b) * sa + c * lda + r] = v;
            }
        for (int c = 0; c < q; ++c)
            for (int r = 0; r < n; ++r) {
                const T v = rhs_fill<T>(r + item % 4, c);
                bbuf[static_cast<size_t>(b) * sb + c * ldb + r] = v;
                b_in[(static_cast<size_t>(b) * q + c) * n + r] = v;
            }
    }
    const std::vector<T> a_host(abuf.begin(), abuf.begin() + abuf.size());
    MatrixView<T, MatrixFormat::Dense> Av(abuf.data(), n, n, lda, sa, bs);
    MatrixView<T, MatrixFormat::Dense> Bv(bbuf.data(), n, q, ldb, sb, bs);
    (void)batchlas::sycl_trsm::trsm_native_sg_left_dispatch<T>(
        *ctx, Av, Bv, tc.alpha, tc.uplo, tc.transA, tc.diag);
    ctx->wait();

    std::vector<T> x(static_cast<size_t>(n) * q * bs);
    for (int b = 0; b < bs; ++b) {
        for (int c = 0; c < q; ++c)
            for (int r = 0; r < n; ++r)
                x[(static_cast<size_t>(b) * q + c) * n + r] =
                    bbuf[static_cast<size_t>(b) * sb + c * ldb + r];
        // The padding rows of B are outside the view: a store there is a defect too.
        for (int c = 0; c < q; ++c)
            for (int r = n; r < ldb; ++r)
                EXPECT_EQ(bbuf[static_cast<size_t>(b) * sb + c * ldb + r], poison)
                    << "store into B's ld padding at b=" << b << " r=" << r << " c=" << c;
    }
    if (!check) return x;

    const MatrixView<T, MatrixFormat::Dense> A0(const_cast<T*>(a_host.data()), n, n, lda, sa, bs);
    const MatrixView<T, MatrixFormat::Dense> B0(b_in.data(), n, q, n, n * q, bs);
    const double res = batchlas::verify::trsm_residual(A0, Side::Left, tc.uplo, tc.transA, tc.diag, Bv, B0,
                                                       batchlas::verify::up(tc.alpha));
    if (!batchlas::verify::pass<T>(batchlas::verify::Check::solve, n, res))
        ADD_FAILURE() << "sg-left n=" << n << " q=" << q << " uplo=" << int(tc.uplo) << " transA=" << int(tc.transA)
                      << " diag=" << int(tc.diag) << " residual=" << res << " exceeds "
                      << batchlas::verify::bound<T>(batchlas::verify::Check::solve, n);
    return x;
}

}  // namespace

TEST(TrsmNativeSgLeft, CanonicalCrossProductStridedComplex) {
    for (Uplo up : {Uplo::Lower, Uplo::Upper})
        for (Transpose tr : {Transpose::NoTrans, Transpose::Trans, Transpose::ConjTrans})
            for (Diag dg : {Diag::NonUnit, Diag::Unit}) {
                RunSgLeft<std::complex<float>>({13, 9, 5, up, tr, dg, {0.5f, -1.25f}});
                RunSgLeft<std::complex<double>>({29, 17, 3, up, tr, dg, {-1.5, 0.75}});
            }
}

TEST(TrsmNativeSgLeft, CanonicalCrossProductStridedReal) {
    for (Uplo up : {Uplo::Lower, Uplo::Upper})
        for (Transpose tr : {Transpose::NoTrans, Transpose::Trans})
            for (Diag dg : {Diag::NonUnit, Diag::Unit}) {
                RunSgLeft<float>({6, 7, 9, up, tr, dg, -2.0f});
                RunSgLeft<double>({32, 33, 3, up, tr, dg, 0.75});
            }
}

// Orders straddle every bucket edge (4|5, 8|9, 16|17, 32) and q straddles every
// QC edge (4|5, 8|9, 16|17), for every type, so each compiled <T, N, QC> kernel
// (rolled and unrolled bodies alike) is launched; batches are not multiples of
// the 32/N matrices a sub-group packs, so the tail sub-group carries absent ones.
TEST(TrsmNativeSgLeft, BucketAndChunkEdges) {
    const int ns[] = {1, 3, 4, 5, 8, 9, 16, 17, 32};
    const int qs[] = {1, 4, 5, 8, 9, 16, 17, 40};
    for (int n : ns)
        for (int q : qs) {
            RunSgLeft<float>({n, q, 7, Uplo::Lower, Transpose::Trans, Diag::Unit, 0.5f});
            RunSgLeft<double>({n, q, 11, Uplo::Upper, Transpose::Trans, Diag::NonUnit, -1.25});
            RunSgLeft<std::complex<float>>(
                {n, q, 13, Uplo::Lower, Transpose::ConjTrans, Diag::NonUnit, {1.0f, 0.5f}});
            RunSgLeft<std::complex<double>>(
                {n, q, 11, Uplo::Upper, Transpose::ConjTrans, Diag::Unit, {-0.5, 1.5}});
        }
}

// The capacity guard: n = 32 launches (above), n = 33 must throw, not truncate.
// ARMED BREAK: bucket n <= 64 into N = 32. OBSERVED: red only on this test.
TEST(TrsmNativeSgLeft, OrderAboveSubGroupWidthThrows) {
    auto ctx = std::make_shared<Queue>(Device("gpu"), Backend::CUDA);
    Matrix<float, MatrixFormat::Dense> A(33, 33, 2);
    Matrix<float, MatrixFormat::Dense> B(33, 4, 2);
    EXPECT_THROW((void)batchlas::sycl_trsm::trsm_native_sg_left_dispatch<float>(
                     *ctx, A.view(), B.view(), 1.0f, Uplo::Lower, Transpose::NoTrans,
                     Diag::NonUnit),
                 std::exception);
}

// No SLM here, but the matrices of one sub-group share every broadcast, so a lane
// index error shows up as a neighbour's data. Items are distinct with period 60
// (the fills use item % 3, % 4, % 5): the per-item multiply-back catches a mixed
// or permuted item, and item b must equal item b % 60 bitwise (determinism).
TEST(TrsmNativeSgLeft, SaturatingBatchIsBitIdentical) {
    auto run = [](auto tag, int n, int q) {
        using T = decltype(tag);
        SgCase<T> tc{n, q, 2048, Uplo::Lower, Transpose::NoTrans, Diag::NonUnit, T(1)};
        const auto x = RunSgLeft<T>(tc, true);
        const size_t item = static_cast<size_t>(n) * q;
        for (int b = 60; b < tc.batch; ++b)
            ASSERT_EQ(0, std::memcmp(x.data() + (b % 60) * item, x.data() + b * item,
                                     item * sizeof(T)))
                << "item " << b << " differs from item " << b % 60 << " at n=" << n
                << " q=" << q;
    };
    run(std::complex<float>{}, 32, 17);
    run(float{}, 8, 9);
    run(double{}, 4, 3);
    run(std::complex<double>{}, 16, 20);
}

// A subnormal diagonal has a reciprocal of inf while the division stays finite.
// The neighbours sharing the sub-group must stay exact. Their diagonal is 2, where
// both paths are exact, so this does not tell a per-matrix fallback from a
// sub-group-wide one.
TEST(TrsmNativeSgLeft, NonFiniteReciprocalFallsBackToDivision) {
    auto ctx = std::make_shared<Queue>(Device("gpu"), Backend::CUDA);
    const int n = 8, q = 3, bs = 4;
    const float tiny = 1e-39f;
    Matrix<float, MatrixFormat::Dense> A(n, n, bs);
    Matrix<float, MatrixFormat::Dense> B(n, q, bs);
    auto Av = A.view();
    auto Bv = B.view();
    for (int b = 0; b < bs; ++b) {
        for (int c = 0; c < n; ++c)
            for (int r = 0; r < n; ++r)
                Av.at(r, c, b) = (r == c) ? ((b == 1 && r == 5) ? tiny : 2.0f) : 0.0f;
        for (int c = 0; c < q; ++c)
            for (int r = 0; r < n; ++r)
                Bv.at(r, c, b) = (b == 1 && r == 5) ? tiny : 2.0f * (1 + c);
    }
    (void)batchlas::sycl_trsm::trsm_native_sg_left_dispatch<float>(
        *ctx, A.view(), B.view(), 1.0f, Uplo::Lower, Transpose::NoTrans,
        Diag::NonUnit);
    ctx->wait();
    for (int b = 0; b < bs; ++b)
        for (int c = 0; c < q; ++c)
            for (int r = 0; r < n; ++r) {
                const float want = (b == 1 && r == 5) ? 1.0f : float(1 + c);
                EXPECT_EQ(Bv.at(r, c, b), want) << "b=" << b << " r=" << r << " c=" << c;
            }
}

// V1's ladder refuses a rung more than half of whose lanes own no column. At a
// saturated batch that picks 64 lanes for q=40 (24 dead) and 32 for q=8, which
// is a multi-sub-group group on the Right side that no other case reaches.
// evidence: docs/perf/blackwell.md#trsm-v1-ladder-cap
TEST(TrsmNativeCta, CappedLadderSaturatedSmallRhs) {
    auto probe = std::make_shared<Queue>(Device("gpu"), Backend::CUDA);
    ASSERT_EQ(trsm_expected_wg(*probe, 40, 4096), 64);
    ASSERT_EQ(trsm_expected_wg(*probe, 8, 4096), 32);
    for (int q : {8, 40}) {
        RunTrsmNative<float>({32, q, 4096, Side::Right, Uplo::Lower, Transpose::Trans,
                              Diag::NonUnit, 1.5f});
        RunTrsmNative<std::complex<float>>({29, q, 4096, Side::Right, Uplo::Upper,
                                            Transpose::ConjTrans, Diag::NonUnit, {1.0f, -0.5f}});
    }
    RunTrsmNative<double>({32, 40, 4096, Side::Left, Uplo::Lower, Transpose::NoTrans,
                           Diag::NonUnit, -1.0});
}

// Complex "vendor" trsm on CUDA is BatchLAS's own substitute kernel (cublas.cc). Its batch
// offset b * strideA was an int product: at order 512 and batch 8193 the last item starts at
// element 2^31 and the kernel faulted (CUDA_ERROR_ILLEGAL_ADDRESS). Needs ~17 GB of device
// memory, so it skips on smaller GPUs; only items 0 and batch-1 hold a system.
TEST(TrsmVendor, ComplexSubstituteIndexesPast2To31Elements) {
    using T = std::complex<float>;
    if constexpr (!batchlas::select::level3_vendor_available<Backend::CUDA>) {
        GTEST_SKIP() << "no vendor BLAS in this build";
    } else {
        auto ctx = std::make_shared<Queue>(Device("gpu"), Backend::CUDA);
        const std::size_t mem = ctx->device().get_property(DeviceProperty::GLOBAL_MEM_SIZE);
        if (mem < (std::size_t(32) << 30)) GTEST_SKIP() << "needs 32 GiB of device memory, has " << (mem >> 30);
        const int n = 512, bs = 8193;
        Matrix<T, MatrixFormat::Dense> A(n, n, bs);
        Matrix<T, MatrixFormat::Dense> B(n, 1, bs);
        const std::size_t sa = static_cast<std::size_t>(A.view().stride());
        ASSERT_GE(sa * (bs - 1), std::size_t(1) << 31) << "the last item must start at or past 2^31 elements";
        std::vector<T> a0(static_cast<std::size_t>(n) * n), b0(n);
        for (int j = 0; j < n; ++j) {
            b0[j] = T(1.0f + 0.01f * j, -0.5f + 0.003f * j);
            for (int i = 0; i < n; ++i)
                a0[i + static_cast<std::size_t>(j) * n] =
                    i == j ? T(float(n + 1), 0.5f) : (i > j ? T(0.3f * std::sin(i + 2.0f * j), 0.2f) : T(9e9f, 0));
        }
        for (int b : {0, bs - 1}) {
            std::copy(a0.begin(), a0.end(), A.view().data_ptr() + sa * b);
            std::copy(b0.begin(), b0.end(), B.view().data_ptr() + static_cast<std::size_t>(n) * b);
        }
        (void)backend::trsm_vendor<Backend::CUDA, T>(*ctx, A.view(), B.view(), Side::Left, Uplo::Lower,
                                                     Transpose::NoTrans, Diag::NonUnit, T(1));
        ctx->wait();
        for (int b : {0, bs - 1}) {
            const T* x = B.view().data_ptr() + static_cast<std::size_t>(n) * b;
            double num = 0, den = 0;
            for (int i = 0; i < n; ++i) {
                std::complex<double> s = 0;
                for (int k = 0; k <= i; ++k)
                    s += std::complex<double>(a0[i + static_cast<std::size_t>(k) * n]) * std::complex<double>(x[k]);
                num += std::norm(s - std::complex<double>(b0[i]));
                den += std::norm(std::complex<double>(b0[i]));
            }
            EXPECT_LT(std::sqrt(num / den), 1e-5) << "item " << b;
        }
    }
}
