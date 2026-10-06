#include <gtest/gtest.h>
#include <batchlas/blas/linalg.hh>
#include <batchlas/blas/matrix.hh>
#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <type_traits>
#include <cstdlib>
#include <string>
#include <cstring>

#include <batchlas/backend_config.h>
#include <batchlas/util/env.hh>
#include "../src/select/vendor.hh"
#include "../src/ops/gemm/choice.hh"
#include "../src/sycl/gemm_kernels.hh"
#include <complex>
#include <utility>
#include "test_utils.hh"

using namespace batchlas;

namespace {

// A gemm pin by name (a family spelling or a class word); an unknown or can_run-false name
// throws.
struct GemmPin {
    explicit GemmPin(const char* word) : pin("gemm", std::string_view(word)) {}
    select::ScopedPin<ops::gemm::GemmChoice> pin;
};

// A CPU queue with a host BLAS keeps it (can_run): native pins throw there, so the
// forced-kernel comparisons below run where the native families do.
bool NativePinsRun(Queue& ctx) { return ctx.device().type == DeviceType::GPU || !batchlas::select::kHasNetlib; }
#define SKIP_UNLESS_NATIVE(ctx) \
    if (!NativePinsRun(ctx)) GTEST_SKIP() << "a CPU queue with a host BLAS runs no native gemm"

// A pin that can_run refuses throws before any kernel runs, and leaves C bit for bit.
template <typename T>
void ExpectPinRefused(Queue& ctx, const char* word, int m, int n, int k, Transpose ta, Transpose tb) {
    SCOPED_TRACE(::testing::Message() << word << " " << m << "x" << n << "x" << k << " ta=" << int(ta)
                                      << " tb=" << int(tb));
    auto A = Matrix<T>::Random(ta == Transpose::NoTrans ? m : k, ta == Transpose::NoTrans ? k : m, false, 2);
    auto B = Matrix<T>::Random(tb == Transpose::NoTrans ? k : n, tb == Transpose::NoTrans ? n : k, false, 2);
    auto C = Matrix<T>::Random(m, n, false, 2);
    auto C0 = C.clone();
    const GemmPin pin(word);
    try {
        (void)gemm(ctx, A.view(), B.view(), C.view(),
                   {.alpha = T(2), .beta = T(-1), .transA = ta, .transB = tb});
        ctx.wait();
        ADD_FAILURE() << "accepted";
    } catch (const std::invalid_argument& e) {
        const std::string w = e.what();  // can_run false, or no such candidate for this scalar (R6)
        EXPECT_TRUE(w.find("cannot run this shape") != std::string::npos ||
                    w.find("is not a compiled gemm") != std::string::npos)
            << w;
    }
    for (int b = 0; b < 2; ++b)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < m; ++i)
                ASSERT_EQ(std::memcmp(&C(i, j, b), &C0(i, j, b), sizeof(T)), 0) << "a refused pin wrote C";
}

template <typename T>
::testing::AssertionResult AssertBatchedBufferNear(const UnifiedVector<T>& actual,
                                                   const UnifiedVector<T>& expected,
                                                   size_t rows,
                                                   size_t cols,
                                                   size_t batch_size,
                                                   typename batchlas::base_type<T>::type tol) {
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t row = 0; row < rows; ++row) {
            for (size_t col = 0; col < cols; ++col) {
                const size_t index = b * rows * cols + row * cols + col;
                const auto actual_value = actual[index];
                const auto expected_value = expected[index];
                if constexpr (test_utils::is_complex<T>::value) {
                    if (std::abs(actual_value.real() - expected_value.real()) > tol ||
                        std::abs(actual_value.imag() - expected_value.imag()) > tol) {
                        return ::testing::AssertionFailure()
                               << "mismatch at batch=" << b << ", row=" << row << ", col=" << col
                               << ": actual=" << actual_value << ", expected=" << expected_value << ", tol=" << tol;
                    }
                } else {
                    if (std::abs(actual_value - expected_value) > tol) {
                        return ::testing::AssertionFailure()
                               << "mismatch at batch=" << b << ", row=" << row << ", col=" << col
                               << ": actual=" << actual_value << ", expected=" << expected_value << ", tol=" << tol;
                    }
                }
            }
        }
    }

    return ::testing::AssertionSuccess();
}

template <typename T>
::testing::AssertionResult AssertBatchedMatrixNear(const Matrix<T>& actual,
                                                   const Matrix<T>& expected,
                                                   int rows,
                                                   int cols,
                                                   int batch_size,
                                                   typename batchlas::base_type<T>::type tol) {
    for (int b = 0; b < batch_size; ++b) {
        for (int col = 0; col < cols; ++col) {
            for (int row = 0; row < rows; ++row) {
                const auto actual_value = actual(row, col, b);
                const auto expected_value = expected(row, col, b);
                if constexpr (test_utils::is_complex<T>::value) {
                    if (std::abs(actual_value.real() - expected_value.real()) > tol ||
                        std::abs(actual_value.imag() - expected_value.imag()) > tol) {
                        return ::testing::AssertionFailure()
                               << "mismatch at batch=" << b << ", row=" << row << ", col=" << col
                               << ": actual=" << actual_value << ", expected=" << expected_value << ", tol=" << tol;
                    }
                } else {
                    if (std::abs(actual_value - expected_value) > tol) {
                        return ::testing::AssertionFailure()
                               << "mismatch at batch=" << b << ", row=" << row << ", col=" << col
                               << ": actual=" << actual_value << ", expected=" << expected_value << ", tol=" << tol;
                    }
                }
            }
        }
    }

    return ::testing::AssertionSuccess();
}

// Pins one named SYCL GEMM kernel and checks it against the vendor result on
// the same random operands. The A/B operand shapes are derived from transA and
// transB so a caller only states the logical m/n/k; alpha and beta default to
// the accumulate-into-C form (1, 1) that almost every kernel test wants, and
// are exposed so the predicated-edge cases can exercise a non-trivial scaling.
// The name is a gemm pin spelling (src/ops/gemm/choice.hh), so a form the kernel
// does not instantiate throws instead of being computed as another form.
template <typename ScalarType, Backend BackendType>
void RunForcedSyclGemmKernelCompare(Queue& ctx,
                                    const char* kernel_name,
                                    int m,
                                    int n,
                                    int k,
                                    int batch_size,
                                    Transpose transA,
                                    Transpose transB,
                                    typename batchlas::base_type<ScalarType>::type tol_scale = 75,
                                    ScalarType alpha = ScalarType(1),
                                    ScalarType beta = ScalarType(1)) {
    SKIP_UNLESS_NATIVE(ctx);
    const int a_rows = transA == Transpose::NoTrans ? m : k;
    const int a_cols = transA == Transpose::NoTrans ? k : m;
    const int b_rows = transB == Transpose::NoTrans ? k : n;
    const int b_cols = transB == Transpose::NoTrans ? n : k;

    auto A = Matrix<ScalarType>::Random(a_rows, a_cols, false, batch_size);
    auto B = Matrix<ScalarType>::Random(b_rows, b_cols, false, batch_size);
    auto C = Matrix<ScalarType>::Random(m, n, false, batch_size);
    auto C_ref = C.clone();

    {
        const GemmPin force_kernel(kernel_name);
        (void)gemm(ctx,
                          A.view(),
                          B.view(),
                          C.view(),
                          {.alpha = alpha, .beta = beta, .transA = transA, .transB = transB});
    }

    {
        const GemmPin vendor_variant("vendor");
        (void)gemm(ctx,
                          A.view(),
                          B.view(),
                          C_ref.view(),
                          {.alpha = alpha, .beta = beta, .transA = transA, .transB = transB});
    }

    ctx.wait();

    auto tol = test_utils::tolerance<ScalarType>() * tol_scale;
    ASSERT_TRUE(AssertBatchedMatrixNear(C, C_ref, m, n, batch_size, tol));
}

} // namespace

template <typename T, Backend B>
struct TestConfig {
    using ScalarType = T;
    static constexpr Backend BackendVal = B;
};

using GemmTestTypes = typename test_utils::backend_types<TestConfig>::type;

template <typename Config>
class GemmTest : public test_utils::BatchLASTest<Config> {
protected:
    using ScalarType = typename Config::ScalarType;
    static constexpr Backend BackendType = Config::BackendVal;

    const int rows = 10;
    const int cols = 10;
    const int ld = 10;
    const int batch_size = 3;
    UnifiedVector<ScalarType> A_data;
    UnifiedVector<ScalarType> B_data;
    UnifiedVector<ScalarType> C_data;

    void SetUp() override {
        test_utils::BatchLASTest<Config>::SetUp();
        
        if (!this->ctx) {
            return;
        }

        // Initialize test matrices
        A_data = UnifiedVector<ScalarType>(rows * cols * batch_size);
        B_data = UnifiedVector<ScalarType>(cols * cols * batch_size);
        C_data = UnifiedVector<ScalarType>(rows * cols * batch_size, static_cast<ScalarType>(0));
        
        for (int b = 0; b < batch_size; ++b) {
            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < cols; ++j) {
                    A_data[b * rows * cols + i * cols + j] = static_cast<ScalarType>(i * cols + j);
                }
            }
        }
        
        for (int b = 0; b < batch_size; ++b) {
            for (int i = 0; i < cols; ++i) {
                for (int j = 0; j < cols; ++j) {
                    B_data[b * cols * cols + i * cols + j] = static_cast<ScalarType>(i == j ? 1.0 : 0.0);
                }
            }
        }
    }

    void printMatrix(UnifiedVector<ScalarType>& matrix_data, int rows, int cols, int ld){
        for (int i = 0; i < cols; ++i) {
            for (int j = 0; j < rows; ++j) {
                std::cout << matrix_data[i * ld + j] << " ";
            }
            std::cout << std::endl;
        }
    }
};

TYPED_TEST_SUITE(GemmTest, GemmTestTypes);

// The old GemmDispatchPolicyTest asserts, against the transcribed sm_89 float table that replaced
// select_kernel_variant (read with Table::nearest, so every device checks them). The native pick
// is the first non-vendor entry: what Auto runs vendor-free and what `native` resolves to.
// A padded ld (272, 258) is layout=strided. The vendor-vs-native order is not asserted here
// (gemm_candidates_tests.cc AutoReadsTheSm89TranscribedTable runs it).
TEST(GemmDispatchPolicyTest, Sm89TableKeepsTheOldNativeSelector) {
    struct Row { const char* ta; const char* layout; int m, n, k; const char* native; };
    const Row rows[] = {
        {"N", "packed", 128, 128, 128, "reg:m=128:n=128:k=8:u=1"},
        {"N", "packed", 256, 256, 256, "reg:m=128:n=128:k=8:u=1"},
        {"N", "packed", 512, 512, 512, "reg:m=128:n=128:k=8:u=1"},
        {"N", "packed", 512, 256, 512, "reg:m=128:n=128:k=8:u=1"},
        {"N", "strided", 256, 256, 256, "reg:m=128:n=128:k=8:u=1"},  // ld 272 and ld 258
        {"T", "packed", 256, 128, 256, "reg:m=128:n=32:k=32:u=1"},
        {"T", "strided", 256, 128, 256, "reg:m=128:n=32:k=32:u=1"},
        // Off-grid: the old selector chose 128x32x16 for packed 512x64x512 NN; the nearest
        // transcribed row is m=512 n=128 k=128, so the table now names 128x128x8.
        {"N", "packed", 512, 64, 512, "reg:m=128:n=128:k=8:u=1"},
        {"N", "strided", 512, 64, 512, "reg:m=128:n=32:k=16:u=1"},
    };
    const auto tables = select::tables_in_borrow_order("gemm", "float", select::device_from_key("sm_89"));
    ASSERT_FALSE(tables.empty());
    const select::Table& t = *tables.front();
    ASSERT_EQ(t.device, "sm_89");
    EXPECT_EQ(t.source.rfind("transcribed:", 0), 0u) << t.file;
    for (const Row& r : rows) {
        const select::Key key{{"ta", r.ta}, {"tb", "N"}, {"layout", r.layout}, {"m", r.m},
                              {"n", r.n},   {"k", r.k},  {"batch", 1}};
        const select::TableRow* row = t.nearest(key);
        ASSERT_NE(row, nullptr);
        const std::string what = t.file + ":" + std::to_string(row->line);
        const auto it = std::find_if(row->ranked.begin(), row->ranked.end(),
                                     [](const auto& e) { return e.spelling != "vendor"; });
        ASSERT_NE(it, row->ranked.end()) << what;
        EXPECT_EQ(it->spelling, r.native) << what << " ta=" << r.ta << " " << r.layout << " " << r.m << "x" << r.n
                                          << "x" << r.k;
    }
}

// Test GEMM operation using identity matrix (C = A * I = A)
TYPED_TEST(GemmTest, GemmWithIdentityMatrix) {
    using ScalarType = typename TestFixture::ScalarType;
    constexpr Backend BackendType = TestFixture::BackendType;
    // Create matrix views with the new matrix handle format - using default template parameters
    MatrixView<ScalarType, MatrixFormat::Dense> A_view(this->A_data.data(), this->rows, this->cols, this->ld);
    MatrixView<ScalarType, MatrixFormat::Dense> B_view(this->B_data.data(), this->cols, this->cols, this->ld);
    MatrixView<ScalarType, MatrixFormat::Dense> C_view(this->C_data.data(), this->rows, this->cols, this->ld);
    
    // Perform C = A * B (which should equal A since B is identity)
    (void)gemm(*(this->ctx),
                      A_view,
                      B_view,
                      C_view,
                      {.alpha = ScalarType(1.0), .beta = ScalarType(0.0)});

    this->ctx->wait();

    auto tol = test_utils::tolerance<ScalarType>();
    ASSERT_TRUE(AssertBatchedBufferNear(this->C_data, this->A_data, this->rows, this->cols, 1, tol));
}

// Test batched GEMM operation
TYPED_TEST(GemmTest, BatchedGemm) {
    using ScalarType = typename TestFixture::ScalarType;
    constexpr Backend BackendType = TestFixture::BackendType;
    // Create batched matrix views - using default template parameters
    MatrixView<ScalarType, MatrixFormat::Dense> A_view(this->A_data.data(), this->rows, this->cols, this->ld, this->rows * this->cols, this->batch_size);
    MatrixView<ScalarType, MatrixFormat::Dense> B_view(this->B_data.data(), this->rows, this->cols, this->ld, this->rows * this->cols, this->batch_size);
    MatrixView<ScalarType, MatrixFormat::Dense> C_view(this->C_data.data(), this->rows, this->cols, this->ld, this->rows * this->cols, this->batch_size);
    
    // Adding the ComputePrecision parameter
    (void)gemm(*(this->ctx),
                      A_view,
                      B_view,
                      C_view,
                      {.alpha = ScalarType(1.0), .beta = ScalarType(0.0)});
    
    this->ctx->wait();

    auto tol = test_utils::tolerance<ScalarType>();
    ASSERT_TRUE(AssertBatchedBufferNear(this->C_data, this->A_data, this->rows, this->cols, this->batch_size, tol));
}

// A heterogeneous batch is split into homogeneous items, each chosen on its own: a
// native pin applies per item (the deleted cuBLASDx heterogeneous kernel's case), and
// an item the pinned family cannot run refuses the whole call.
TYPED_TEST(GemmTest, HeterogeneousBatchedGemmNativePinAppliesPerItem) {
    using ScalarType = typename TestFixture::ScalarType;
    using Real = typename batchlas::base_type<ScalarType>::type;
    SKIP_UNLESS_NATIVE(*this->ctx);
    constexpr int batch = 2, mm = 64, mn = 64, mk = 32;
    auto A = Matrix<ScalarType>::Random(mm, mk, false, batch);
    auto B = Matrix<ScalarType>::Random(mk, mn, false, batch);
    auto C = Matrix<ScalarType>::Random(mm, mn, false, batch);
    auto C0 = C.clone();
    UnifiedVector<int> ar(batch), ac(batch), br(batch), bc(batch), cr(batch), cc(batch);
    ar[0] = 32, ac[0] = 32, br[0] = 32, bc[0] = 32, cr[0] = 32, cc[0] = 32;
    ar[1] = 64, ac[1] = 32, br[1] = 32, bc[1] = 64, cr[1] = 64, cc[1] = 64;
    A.set_active_dims(ar.to_span(), ac.to_span());
    B.set_active_dims(br.to_span(), bc.to_span());
    C.set_active_dims(cr.to_span(), cc.to_span());
    const ScalarType alpha = ScalarType(Real(1.5)), beta = ScalarType(Real(-0.5));
    for (const char* word : {"tiled", "direct", "wide:m=64:n=64:k=16"}) {
        SCOPED_TRACE(word);
        for (int b = 0; b < batch; ++b)
            for (int j = 0; j < mn; ++j)
                for (int i = 0; i < mm; ++i) C(i, j, b) = C0(i, j, b);
        {
            const GemmPin pin(word);
            (void)gemm(*(this->ctx), A.view(), B.view(), C.view(), alpha, beta, Transpose::NoTrans,
                       Transpose::NoTrans, ComputePrecision::Default);
            this->ctx->wait();
        }
        for (int b = 0; b < batch; ++b)
            for (int j = 0; j < cc[b]; ++j)
                for (int i = 0; i < cr[b]; ++i) {
                    ScalarType ref = beta * C0(i, j, b);
                    for (int l = 0; l < ac[b]; ++l) ref += alpha * A(i, l, b) * B(l, j, b);
                    ASSERT_LE(std::abs(C(i, j, b) - ref), test_utils::tolerance<ScalarType>() * 100)
                        << "item " << b << " (" << i << "," << j << ")";
                }
    }
    if constexpr (!test_utils::is_complex<ScalarType>::value) {
        const GemmPin pin("small");  // item 1 is 64 x 64 x 32: small runs it; a 65 would not
        (void)gemm(*(this->ctx), A.view(), B.view(), C.view(), alpha, beta, Transpose::NoTrans, Transpose::NoTrans,
                   ComputePrecision::Default);
        this->ctx->wait();
    } else {
        const GemmPin pin("small");
        EXPECT_THROW(((void)gemm(*(this->ctx), A.view(), B.view(), C.view(), alpha, beta, Transpose::NoTrans,
                                 Transpose::NoTrans, ComputePrecision::Default)),
                     std::invalid_argument);
    }
}

TYPED_TEST(GemmTest, HeterogeneousBatchedGemmUsesPerItemActiveDimensions) {
    using ScalarType = typename TestFixture::ScalarType;
    constexpr Backend BackendType = TestFixture::BackendType;

    constexpr int batch_size = 3;
    constexpr int max_m = 4;
    constexpr int max_n = 5;
    constexpr int max_k = 3;

    Matrix<ScalarType> A(max_m, max_k, batch_size);
    Matrix<ScalarType> B(max_k, max_n, batch_size);
    Matrix<ScalarType> C(max_m, max_n, batch_size);
    Matrix<ScalarType> C_ref(max_m, max_n, batch_size);

    A.fill(ScalarType(0));
    B.fill(ScalarType(0));
    C.fill(ScalarType(0));
    C_ref.fill(ScalarType(0));

    UnifiedVector<int> a_rows(batch_size);
    UnifiedVector<int> a_cols(batch_size);
    UnifiedVector<int> b_rows(batch_size);
    UnifiedVector<int> b_cols(batch_size);
    UnifiedVector<int> c_rows(batch_size);
    UnifiedVector<int> c_cols(batch_size);

    a_rows[0] = 4; a_cols[0] = 3;
    b_rows[0] = 3; b_cols[0] = 5;
    c_rows[0] = 4; c_cols[0] = 5;

    a_rows[1] = 2; a_cols[1] = 3;
    b_rows[1] = 3; b_cols[1] = 2;
    c_rows[1] = 2; c_cols[1] = 2;

    a_rows[2] = 0; a_cols[2] = 3;
    b_rows[2] = 3; b_cols[2] = 4;
    c_rows[2] = 0; c_cols[2] = 4;

    A.set_active_dims(a_rows.to_span(), a_cols.to_span());
    B.set_active_dims(b_rows.to_span(), b_cols.to_span());
    C.set_active_dims(c_rows.to_span(), c_cols.to_span());
    C_ref.set_active_dims(c_rows.to_span(), c_cols.to_span());

    for (int b = 0; b < batch_size; ++b) {
        for (int col = 0; col < A.cols(b); ++col) {
            for (int row = 0; row < A.rows(b); ++row) {
                A(row, col, b) = static_cast<ScalarType>(1 + row + 2 * col + 10 * b);
            }
        }
        for (int col = 0; col < B.cols(b); ++col) {
            for (int row = 0; row < B.rows(b); ++row) {
                B(row, col, b) = static_cast<ScalarType>(1 + row + col + 7 * b);
            }
        }
    }

    for (int b = 0; b < batch_size; ++b) {
        auto Ab = A.view().batch_item(b);
        auto Bb = B.view().batch_item(b);
        auto Cb = C_ref.view().batch_item(b);
        if (Cb.rows() == 0 || Cb.cols() == 0) {
            continue;
        }

        (void)gemm(*(this->ctx), Ab, Bb, Cb, {.alpha = ScalarType(1), .beta = ScalarType(0)});
    }

    (void)gemm(*(this->ctx), A.view(), B.view(), C.view(), ScalarType(1), ScalarType(0),
                                    Transpose::NoTrans, Transpose::NoTrans, ComputePrecision::Default);

    this->ctx->wait();

    for (int b = 0; b < batch_size; ++b) {
        ASSERT_EQ(C.rows(b), C_ref.rows(b));
        ASSERT_EQ(C.cols(b), C_ref.cols(b));
        for (int col = 0; col < C.cols(b); ++col) {
            for (int row = 0; row < C.rows(b); ++row) {
                const auto actual = C(row, col, b);
                const auto expected = C_ref(row, col, b);
                auto tol = test_utils::tolerance<ScalarType>() * 50;
                if constexpr (test_utils::is_complex<ScalarType>::value) {
                    ASSERT_NEAR(actual.real(), expected.real(), tol);
                    ASSERT_NEAR(actual.imag(), expected.imag(), tol);
                } else {
                    ASSERT_NEAR(actual, expected, tol);
                }
            }
        }
    }
}

TYPED_TEST(GemmTest, HeterogeneousBatchedGemmZeroInnerDimensionScalesCByBeta) {
    using ScalarType = typename TestFixture::ScalarType;
    constexpr Backend BackendType = TestFixture::BackendType;

    constexpr int batch_size = 3;
    constexpr int max_m = 64;
    constexpr int max_n = 32;
    constexpr int max_k = 32;

    Matrix<ScalarType> A(max_m, max_k, batch_size);
    Matrix<ScalarType> B(max_k, max_n, batch_size);
    Matrix<ScalarType> C(max_m, max_n, batch_size);
    Matrix<ScalarType> C_ref(max_m, max_n, batch_size);

    A.fill(ScalarType(0));
    B.fill(ScalarType(0));
    C.fill(ScalarType(1));
    C_ref.fill(ScalarType(1));

    UnifiedVector<int> a_rows(batch_size);
    UnifiedVector<int> a_cols(batch_size);
    UnifiedVector<int> b_rows(batch_size);
    UnifiedVector<int> b_cols(batch_size);
    UnifiedVector<int> c_rows(batch_size);
    UnifiedVector<int> c_cols(batch_size);

    a_rows[0] = 32; a_cols[0] = 32;
    b_rows[0] = 32; b_cols[0] = 32;
    c_rows[0] = 32; c_cols[0] = 32;

    a_rows[1] = 32; a_cols[1] = 0;
    b_rows[1] = 0;  b_cols[1] = 32;
    c_rows[1] = 32; c_cols[1] = 32;

    a_rows[2] = 0;  a_cols[2] = 32;
    b_rows[2] = 32; b_cols[2] = 32;
    c_rows[2] = 0;  c_cols[2] = 32;

    A.set_active_dims(a_rows.to_span(), a_cols.to_span());
    B.set_active_dims(b_rows.to_span(), b_cols.to_span());
    C.set_active_dims(c_rows.to_span(), c_cols.to_span());
    C_ref.set_active_dims(c_rows.to_span(), c_cols.to_span());

    for (int col = 0; col < A.cols(0); ++col) {
        for (int row = 0; row < A.rows(0); ++row) {
            A(row, col, 0) = static_cast<ScalarType>(1 + row + col);
        }
    }
    for (int col = 0; col < B.cols(0); ++col) {
        for (int row = 0; row < B.rows(0); ++row) {
            B(row, col, 0) = static_cast<ScalarType>(1 + row + 2 * col);
        }
    }

    const ScalarType alpha = static_cast<ScalarType>(2);
    const ScalarType beta = static_cast<ScalarType>(3);

    for (int batch_index = 0; batch_index < batch_size; ++batch_index) {
        auto Ab = A.view().batch_item(batch_index);
        auto Bb = B.view().batch_item(batch_index);
        auto Cb = C_ref.view().batch_item(batch_index);
        if (Cb.rows() == 0 || Cb.cols() == 0) {
            continue;
        }

        if (Ab.cols() == 0) {
            for (int col = 0; col < Cb.cols(); ++col) {
                for (int row = 0; row < Cb.rows(); ++row) {
                    C_ref(row, col, batch_index) *= beta;
                }
            }
            continue;
        }

        (void)gemm(*(this->ctx), Ab, Bb, Cb, {.alpha = alpha, .beta = beta});
    }

    (void)gemm(*(this->ctx), A.view(), B.view(), C.view(), alpha, beta,
                                    Transpose::NoTrans, Transpose::NoTrans, ComputePrecision::Default);

    this->ctx->wait();

    for (int batch_index = 0; batch_index < batch_size; ++batch_index) {
        for (int col = 0; col < C.cols(batch_index); ++col) {
            for (int row = 0; row < C.rows(batch_index); ++row) {
                const auto actual = C(row, col, batch_index);
                const auto expected = C_ref(row, col, batch_index);
                auto tol = test_utils::tolerance<ScalarType>() * 100;
                if constexpr (test_utils::is_complex<ScalarType>::value) {
                    ASSERT_NEAR(actual.real(), expected.real(), tol);
                    ASSERT_NEAR(actual.imag(), expected.imag(), tol);
                } else {
                    ASSERT_NEAR(actual, expected, tol);
                }
            }
        }
    }
}

TYPED_TEST(GemmTest, BatchedGemmForcedSyclVariant) {
    using ScalarType = typename TestFixture::ScalarType;
    constexpr Backend BackendType = TestFixture::BackendType;

    const GemmPin force_variant("native");

    constexpr int size = 32;
    constexpr int batch_size = 4;
    auto A = Matrix<ScalarType>::Random(size, size, false, batch_size);
    auto B = Matrix<ScalarType>::Random(size, size, false, batch_size);
    auto C = Matrix<ScalarType>::Zeros(size, size, batch_size);
    auto C_ref = Matrix<ScalarType>::Zeros(size, size, batch_size);

    (void)gemm(*(this->ctx),
                      A.view(),
                      B.view(),
                      C.view(),
                      {.alpha = ScalarType(1), .beta = ScalarType(0)});

    const GemmPin vendor_variant("vendor");
    (void)gemm(*(this->ctx),
                      A.view(),
                      B.view(),
                      C_ref.view(),
                      {.alpha = ScalarType(1), .beta = ScalarType(0)});

    this->ctx->wait();

    auto tol = test_utils::tolerance<ScalarType>() * 50;
    ASSERT_TRUE(AssertBatchedMatrixNear(C, C_ref, size, size, batch_size, tol));
}

TYPED_TEST(GemmTest, BatchedGemmForcedSyclVariantLargeSquare) {
    using ScalarType = typename TestFixture::ScalarType;
    constexpr Backend BackendType = TestFixture::BackendType;

    constexpr int size = 128;
    constexpr int batch_size = 2;
    auto A = Matrix<ScalarType>::Random(size, size, false, batch_size);
    auto B = Matrix<ScalarType>::Random(size, size, false, batch_size);
    auto C = Matrix<ScalarType>::Random(size, size, false, batch_size);
    auto C_ref = C.clone();

    {
        const GemmPin force_variant("native");
        (void)gemm(*(this->ctx),
                          A.view(),
                          B.view(),
                          C.view(),
                          {.alpha = ScalarType(1), .beta = ScalarType(1)});
    }

    {
        const GemmPin vendor_variant("vendor");
        (void)gemm(*(this->ctx),
                          A.view(),
                          B.view(),
                          C_ref.view(),
                          {.alpha = ScalarType(1), .beta = ScalarType(1)});
    }

    this->ctx->wait();

    auto tol = test_utils::tolerance<ScalarType>() * 75;
    ASSERT_TRUE(AssertBatchedMatrixNear(C, C_ref, size, size, batch_size, tol));
}

TYPED_TEST(GemmTest, BatchedGemmForcedSyclRegister64Kernel) {
    using ScalarType = typename TestFixture::ScalarType;
    constexpr Backend BackendType = TestFixture::BackendType;
    if constexpr (!std::is_same_v<ScalarType, float>) {
        GTEST_SKIP() << "64x64 SYCL register kernel is only selected for float in this first slice";
    }

    RunForcedSyclGemmKernelCompare<ScalarType, BackendType>(*(this->ctx), "reg:m=64:n=64:k=8:u=1",
                                                            128, 128, 128, 2,
                                                            Transpose::NoTrans, Transpose::NoTrans);
}

TYPED_TEST(GemmTest, BatchedGemmForcedSyclRegister64K16Kernel) {
    using ScalarType = typename TestFixture::ScalarType;
    constexpr Backend BackendType = TestFixture::BackendType;

    if constexpr (!std::is_same_v<ScalarType, float>) {
        GTEST_SKIP() << "64x64x16 SYCL register kernel is only selected for float in this slice";
    }

    RunForcedSyclGemmKernelCompare<ScalarType, BackendType>(*(this->ctx), "reg:m=64:n=64:k=16:u=1",
                                                            128, 128, 128, 2,
                                                            Transpose::NoTrans, Transpose::NoTrans);
}

TYPED_TEST(GemmTest, BatchedGemmForcedSyclRegister128x32K16Kernel) {
    using ScalarType = typename TestFixture::ScalarType;
    constexpr Backend BackendType = TestFixture::BackendType;

    if constexpr (!std::is_same_v<ScalarType, float>) {
        GTEST_SKIP() << "128x32x16 SYCL register kernel is only selected for float in this slice";
    }

    RunForcedSyclGemmKernelCompare<ScalarType, BackendType>(*(this->ctx), "reg:m=128:n=32:k=16:u=1",
                                                            128, 128, 128, 2,
                                                            Transpose::NoTrans, Transpose::NoTrans);
}

TYPED_TEST(GemmTest, BatchedGemmForcedSyclRegister128x32K32Kernel) {
    using ScalarType = typename TestFixture::ScalarType;
    constexpr Backend BackendType = TestFixture::BackendType;

    if constexpr (!std::is_same_v<ScalarType, float>) {
        GTEST_SKIP() << "128x32x32 SYCL register kernel is only selected for float in this slice";
    }

    RunForcedSyclGemmKernelCompare<ScalarType, BackendType>(*(this->ctx), "reg:m=128:n=32:k=32:u=1",
                                                            128, 128, 128, 2,
                                                            Transpose::NoTrans, Transpose::NoTrans);
}

TYPED_TEST(GemmTest, BatchedGemmForcedSyclRegister128x32K32S2U1GenericKernel) {
    using ScalarType = typename TestFixture::ScalarType;
    constexpr Backend BackendType = TestFixture::BackendType;

    if constexpr (!std::is_same_v<ScalarType, float>) {
        GTEST_SKIP() << "128x32x32_s2_u1_generic SYCL register kernel is only selected for float in this slice";
    }

    RunForcedSyclGemmKernelCompare<ScalarType, BackendType>(*(this->ctx), "reg:m=128:n=32:k=32:u=1",
                                                            130, 96, 130, 2,
                                                            Transpose::NoTrans, Transpose::NoTrans,
                                                            100);
}

TYPED_TEST(GemmTest, BatchedGemmForcedSyclRegister32x128K16Kernel) {
    using ScalarType = typename TestFixture::ScalarType;
    constexpr Backend BackendType = TestFixture::BackendType;

    if constexpr (!std::is_same_v<ScalarType, float>) {
        GTEST_SKIP() << "32x128x16 SYCL register kernel is only selected for float in this slice";
    }

    RunForcedSyclGemmKernelCompare<ScalarType, BackendType>(*(this->ctx), "reg:m=32:n=128:k=16:u=1",
                                                            128, 128, 128, 2,
                                                            Transpose::NoTrans, Transpose::NoTrans);
}

TYPED_TEST(GemmTest, BatchedGemmForcedSyclVariantTransposed) {
    using ScalarType = typename TestFixture::ScalarType;
    constexpr Backend BackendType = TestFixture::BackendType;

    constexpr int m = 24;
    constexpr int n = 20;
    constexpr int k = 16;
    constexpr int batch_size = 3;

    auto A = Matrix<ScalarType>::Random(k, m, false, batch_size);
    auto B = Matrix<ScalarType>::Random(n, k, false, batch_size);
    auto C = Matrix<ScalarType>::Random(m, n, false, batch_size);
    auto C_ref = C.clone();

    {
        const GemmPin force_variant("native");
        (void)gemm(*(this->ctx),
                          A.view(),
                          B.view(),
                          C.view(),
                          {.alpha = ScalarType(1), .beta = ScalarType(1), .transA = Transpose::Trans, .transB = Transpose::Trans});
    }

    {
        const GemmPin vendor_variant("vendor");
        (void)gemm(*(this->ctx),
                          A.view(),
                          B.view(),
                          C_ref.view(),
                          {.alpha = ScalarType(1), .beta = ScalarType(1), .transA = Transpose::Trans, .transB = Transpose::Trans});
    }

    this->ctx->wait();

    auto tol = test_utils::tolerance<ScalarType>() * 50;
    ASSERT_TRUE(AssertBatchedMatrixNear(C, C_ref, m, n, batch_size, tol));
}

TYPED_TEST(GemmTest, BatchedGemmForcedSyclTiledVariantLargeTransposed) {
    using ScalarType = typename TestFixture::ScalarType;
    constexpr Backend BackendType = TestFixture::BackendType;

    RunForcedSyclGemmKernelCompare<ScalarType, BackendType>(*(this->ctx), "tiled",
                                                            96, 80, 64, 2,
                                                            Transpose::Trans, Transpose::Trans);
}

TYPED_TEST(GemmTest, BatchedGemmForcedSyclRegister128x32K16TTKernel) {
    using ScalarType = typename TestFixture::ScalarType;
    constexpr Backend BackendType = TestFixture::BackendType;

    if constexpr (!std::is_same_v<ScalarType, float>) {
        GTEST_SKIP() << "128x32x16 TT SYCL register kernel is only selected for float in this slice";
    }

    RunForcedSyclGemmKernelCompare<ScalarType, BackendType>(*(this->ctx), "reg:m=128:n=32:k=16:u=1",
                                                            128, 128, 128, 2,
                                                            Transpose::Trans, Transpose::Trans);
}

TYPED_TEST(GemmTest, BatchedGemmForcedSyclRegister128x32K16TNKernel) {
    using ScalarType = typename TestFixture::ScalarType;
    constexpr Backend BackendType = TestFixture::BackendType;

    if constexpr (!std::is_same_v<ScalarType, float>) {
        GTEST_SKIP() << "128x32x16 TN SYCL register kernel is only selected for float in this slice";
    }

    RunForcedSyclGemmKernelCompare<ScalarType, BackendType>(*(this->ctx), "reg:m=128:n=32:k=16:u=1",
                                                            128, 128, 128, 2,
                                                            Transpose::Trans, Transpose::NoTrans);
}

TYPED_TEST(GemmTest, BatchedGemmForcedSyclRegister128x32K16NTKernel) {
    using ScalarType = typename TestFixture::ScalarType;
    constexpr Backend BackendType = TestFixture::BackendType;

    if constexpr (!std::is_same_v<ScalarType, float>) {
        GTEST_SKIP() << "128x32x16 NT SYCL register kernel is only selected for float in this slice";
    }

    RunForcedSyclGemmKernelCompare<ScalarType, BackendType>(*(this->ctx), "reg:m=128:n=32:k=16:u=1",
                                                            128, 128, 128, 2,
                                                            Transpose::NoTrans, Transpose::Trans);
}

TYPED_TEST(GemmTest, BatchedGemmForcedSyclRegister128x32K32TNKernel) {
    using ScalarType = typename TestFixture::ScalarType;
    constexpr Backend BackendType = TestFixture::BackendType;

    if constexpr (!std::is_same_v<ScalarType, float>) {
        GTEST_SKIP() << "128x32x32 TN SYCL register kernel is only selected for float in this slice";
    }

    RunForcedSyclGemmKernelCompare<ScalarType, BackendType>(*(this->ctx), "reg:m=128:n=32:k=32:u=1",
                                                            128, 128, 128, 2,
                                                            Transpose::Trans, Transpose::NoTrans);
}

TYPED_TEST(GemmTest, BatchedGemmForcedSyclRegister128x32K32NTKernel) {
    using ScalarType = typename TestFixture::ScalarType;
    constexpr Backend BackendType = TestFixture::BackendType;

    if constexpr (!std::is_same_v<ScalarType, float>) {
        GTEST_SKIP() << "128x32x32 NT SYCL register kernel is only selected for float in this slice";
    }

    RunForcedSyclGemmKernelCompare<ScalarType, BackendType>(*(this->ctx), "reg:m=128:n=32:k=32:u=1",
                                                            128, 128, 128, 2,
                                                            Transpose::NoTrans, Transpose::Trans);
}

TYPED_TEST(GemmTest, BatchedGemmForcedSyclRegister32x128K16TNKernel) {
    using ScalarType = typename TestFixture::ScalarType;
    constexpr Backend BackendType = TestFixture::BackendType;

    if constexpr (!std::is_same_v<ScalarType, float>) {
        GTEST_SKIP() << "32x128x16 TN SYCL register kernel is only selected for float in this slice";
    }

    RunForcedSyclGemmKernelCompare<ScalarType, BackendType>(*(this->ctx), "reg:m=32:n=128:k=16:u=1",
                                                            128, 128, 128, 2,
                                                            Transpose::Trans, Transpose::NoTrans);
}

TYPED_TEST(GemmTest, BatchedGemmForcedSyclRegister32x128K16TTKernel) {
    using ScalarType = typename TestFixture::ScalarType;
    constexpr Backend BackendType = TestFixture::BackendType;

    if constexpr (!std::is_same_v<ScalarType, float>) {
        GTEST_SKIP() << "32x128x16 TT SYCL register kernel is only selected for float in this slice";
    }

    RunForcedSyclGemmKernelCompare<ScalarType, BackendType>(*(this->ctx), "reg:m=32:n=128:k=16:u=1",
                                                            128, 128, 128, 2,
                                                            Transpose::Trans, Transpose::Trans);
}

TYPED_TEST(GemmTest, BatchedGemmForcedSyclRegister64x64K16TTKernel) {
    using ScalarType = typename TestFixture::ScalarType;
    constexpr Backend BackendType = TestFixture::BackendType;

    if constexpr (!std::is_same_v<ScalarType, float>) {
        GTEST_SKIP() << "64x64x16 TT SYCL register kernel is only selected for float in this slice";
    }

    RunForcedSyclGemmKernelCompare<ScalarType, BackendType>(*(this->ctx), "reg:m=64:n=64:k=16:u=1",
                                                            128, 128, 128, 2,
                                                            Transpose::Trans, Transpose::Trans);
}

TYPED_TEST(GemmTest, BatchedGemmForcedSyclRegister128x64K16TTKernel) {
    using ScalarType = typename TestFixture::ScalarType;
    constexpr Backend BackendType = TestFixture::BackendType;

    if constexpr (!std::is_same_v<ScalarType, float>) {
        GTEST_SKIP() << "128x64x16 TT SYCL register kernel is only selected for float in this slice";
    }

    RunForcedSyclGemmKernelCompare<ScalarType, BackendType>(*(this->ctx), "reg:m=128:n=64:k=16:u=1",
                                                            128, 128, 128, 2,
                                                            Transpose::Trans, Transpose::Trans);
}

TYPED_TEST(GemmTest, BatchedGemmForcedSyclRegister128x32K32TTKernel) {
    using ScalarType = typename TestFixture::ScalarType;
    constexpr Backend BackendType = TestFixture::BackendType;

    if constexpr (!std::is_same_v<ScalarType, float>) {
        GTEST_SKIP() << "128x32x32 TT SYCL register kernel is only selected for float in this slice";
    }

    RunForcedSyclGemmKernelCompare<ScalarType, BackendType>(*(this->ctx), "reg:m=128:n=32:k=32:u=1",
                                                            128, 128, 128, 2,
                                                            Transpose::Trans, Transpose::Trans);
}

TYPED_TEST(GemmTest, BatchedGemmForcedSyclRegister128x64K32LargeKernel) {
    using ScalarType = typename TestFixture::ScalarType;
    constexpr Backend BackendType = TestFixture::BackendType;

    if constexpr (!std::is_same_v<ScalarType, float>) {
        GTEST_SKIP() << "128x64x32 large SYCL register kernel is only selected for float in this slice";
    }

    RunForcedSyclGemmKernelCompare<ScalarType, BackendType>(*(this->ctx), "reg:m=128:n=64:k=32:u=4",
                                                            256, 256, 256, 2,
                                                            Transpose::NoTrans, Transpose::NoTrans,
                                                            100);
}

TYPED_TEST(GemmTest, BatchedGemmForcedSyclRegister128x64K32LargeU2Kernel) {
    using ScalarType = typename TestFixture::ScalarType;
    constexpr Backend BackendType = TestFixture::BackendType;

    if constexpr (!std::is_same_v<ScalarType, float>) {
        GTEST_SKIP() << "128x64x32 large-u2 SYCL register kernel is only selected for float in this slice";
    }

    RunForcedSyclGemmKernelCompare<ScalarType, BackendType>(*(this->ctx), "reg:m=128:n=64:k=32:u=2",
                                                            256, 256, 256, 2,
                                                            Transpose::NoTrans, Transpose::NoTrans,
                                                            100);
}

// The 128x128x8 kernel has two quite different code paths: an unpredicated one
// for shapes that are exact multiples of the tile with 16-byte-aligned
// operands, and a predicated one that zero-fills the shared tile at the edges.
// Both need covering, and the ragged case is the one that can go wrong
// silently.
TYPED_TEST(GemmTest, BatchedGemmForcedSyclRegister128x128K8KernelAligned) {
    using ScalarType = typename TestFixture::ScalarType;
    constexpr Backend BackendType = TestFixture::BackendType;

    if constexpr (!std::is_same_v<ScalarType, float>) {
        GTEST_SKIP() << "128x128x8 SYCL register kernel is float-only: a 64-accumulator "
                        "thread tile spills for wider scalar types";
    }

    // 256 is an exact multiple of both 128 and 8, so this takes the
    // unpredicated path.
    RunForcedSyclGemmKernelCompare<ScalarType, BackendType>(*(this->ctx), "reg:m=128:n=128:k=8:u=1",
                                                            256, 256, 256, 2,
                                                            Transpose::NoTrans, Transpose::NoTrans,
                                                            100);
}

TYPED_TEST(GemmTest, BatchedGemmForcedSyclRegister128x128K8KernelRagged) {
    using ScalarType = typename TestFixture::ScalarType;
    constexpr Backend BackendType = TestFixture::BackendType;

    if constexpr (!std::is_same_v<ScalarType, float>) {
        GTEST_SKIP() << "128x128x8 SYCL register kernel is float-only: a 64-accumulator "
                        "thread tile spills for wider scalar types";
    }

    // Deliberately ragged in all three dimensions: m and n are not multiples
    // of 128 and k is not a multiple of 8, so every tile edge is predicated
    // and the k loop has a partial final step. The non-trivial alpha/beta also
    // keeps the epilogue from degenerating into a plain accumulate.
    RunForcedSyclGemmKernelCompare<ScalarType, BackendType>(*(this->ctx), "reg:m=128:n=128:k=8:u=1",
                                                            200, 130, 70, 3,
                                                            Transpose::NoTrans, Transpose::NoTrans,
                                                            100, ScalarType(2), ScalarType(-1));
}

// A forced kernel against the vendor, with an optional all-NaN C. The NaN
// check is explicit because AssertBatchedMatrixNear's `abs(a - e) > tol` is
// false for a NaN, so a NaN result would pass it.
template <typename ScalarType>
void Run128x128Compare(Queue& ctx, int m, int n, int k, int batch_size, ScalarType alpha,
                       ScalarType beta, bool nan_c, const char* kernel = "reg:m=128:n=128:k=8:u=1") {
    SKIP_UNLESS_NATIVE(ctx);
    SCOPED_TRACE(::testing::Message() << m << "x" << n << "x" << k << " b" << batch_size
                                      << " beta=" << beta << " nan_c=" << nan_c);
    auto A = Matrix<ScalarType>::Random(m, k, false, batch_size);
    auto B = Matrix<ScalarType>::Random(k, n, false, batch_size);
    auto C = Matrix<ScalarType>::Random(m, n, false, batch_size);
    if (nan_c) {
        auto host = C.data();
        for (size_t i = 0; i < host.size(); ++i) {
            host[i] = std::numeric_limits<ScalarType>::quiet_NaN();
        }
    }
    auto C_ref = C.clone();
    {
        const GemmPin force_kernel(kernel);
        (void)gemm(ctx, A.view(), B.view(), C.view(), {.alpha = alpha, .beta = beta});
    }
    {
        const GemmPin vendor_variant("vendor");
        (void)gemm(ctx, A.view(), B.view(), C_ref.view(), {.alpha = alpha, .beta = beta});
    }
    ctx.wait();
    for (int b = 0; b < batch_size; ++b) {
        for (int col = 0; col < n; ++col) {
            for (int row = 0; row < m; ++row) {
                ASSERT_TRUE(std::isfinite(C(row, col, b))) << "b=" << b << " r=" << row << " c=" << col;
            }
        }
    }
    auto tol = test_utils::tolerance<ScalarType>() * 100;
    ASSERT_TRUE(AssertBatchedMatrixNear(C, C_ref, m, n, batch_size, tol));
}

// The double-buffered k loop is unrolled by two, so an ODD slab count ends in a
// separate tail step that an even k never reaches: 136 = 17 slabs of 8, 128 =
// 16. Batch 256 puts 1024 work-groups in flight, enough for a missing barrier
// to race, which the batch-2 cases above cannot show.
// ARMED BREAK (R9): deleting the tail `compute(0)` in register_128x128.hh.
// EXPECTED: OddSlabs RED on both betas at (0,0), EvenSlabs green. Observed.
// ARMED BREAK (R9): deleting the barrier after `sstore(1)`.
// EXPECTED: OddSlabs and EvenSlabs RED on both betas. Observed.
TYPED_TEST(GemmTest, Forced128x128K8OddSlabsBothBetas) {
    using ScalarType = typename TestFixture::ScalarType;
    if constexpr (!std::is_same_v<ScalarType, float>) {
        GTEST_SKIP() << "128x128x8 SYCL register kernel is float-only";
    } else {
        Run128x128Compare<ScalarType>(*(this->ctx), 256, 256, 136, 256, 2.0f, -1.0f, false);
        Run128x128Compare<ScalarType>(*(this->ctx), 256, 256, 136, 256, 1.0f, 0.0f, false);
    }
}

TYPED_TEST(GemmTest, Forced128x128K8EvenSlabsBothBetas) {
    using ScalarType = typename TestFixture::ScalarType;
    if constexpr (!std::is_same_v<ScalarType, float>) {
        GTEST_SKIP() << "128x128x8 SYCL register kernel is float-only";
    } else {
        Run128x128Compare<ScalarType>(*(this->ctx), 384, 256, 128, 256, 2.0f, -1.0f, false);
        Run128x128Compare<ScalarType>(*(this->ctx), 384, 256, 128, 256, 1.0f, 0.0f, false);
    }
}

// Aligned k <= kStagedEpilogueMaxK takes the staged epilogue: C goes through
// local memory and each warp stores whole columns. k = 24 is an odd slab count
// (its tail runs right before the staging reuses the A tile), 8 a single slab.
// ARMED BREAK (R9): dropping `+ p` from the staged store's column.
// EXPECTED: StagedEpilogue RED on the even k = 64 case too, which no other
// break in this set reaches. Observed.
TYPED_TEST(GemmTest, Forced128x128K8StagedEpilogue) {
    using ScalarType = typename TestFixture::ScalarType;
    if constexpr (!std::is_same_v<ScalarType, float>) {
        GTEST_SKIP() << "128x128x8 SYCL register kernel is float-only";
    } else {
        Run128x128Compare<ScalarType>(*(this->ctx), 256, 384, 24, 256, 2.0f, -1.0f, false);
        Run128x128Compare<ScalarType>(*(this->ctx), 256, 384, 64, 256, 1.0f, 0.0f, false);
        Run128x128Compare<ScalarType>(*(this->ctx), 128, 256, 8, 64, 1.0f, 0.0f, true);
    }
}

// beta == 0 must not read C (BLAS semantics: a NaN in C is overwritten), on
// both legs, plus k below one slab on the predicated leg.
// ARMED BREAK (R9): making the aligned epilogue always take the beta != 0
// branch. EXPECTED: BetaZeroNaNC RED on the 256^3 case, a NaN at (0,0). Observed.
TYPED_TEST(GemmTest, Forced128x128K8BetaZeroNaNC) {
    using ScalarType = typename TestFixture::ScalarType;
    if constexpr (!std::is_same_v<ScalarType, float>) {
        GTEST_SKIP() << "128x128x8 SYCL register kernel is float-only";
    } else {
        Run128x128Compare<ScalarType>(*(this->ctx), 256, 256, 256, 8, 1.5f, 0.0f, true);
        Run128x128Compare<ScalarType>(*(this->ctx), 200, 130, 70, 8, 1.5f, 0.0f, true);
        Run128x128Compare<ScalarType>(*(this->ctx), 130, 131, 5, 8, 1.0f, 0.5f, false);
        Run128x128Compare<ScalarType>(*(this->ctx), 129, 257, 1, 8, 1.0f, 0.0f, true);
    }
}

// The 64x64x16 wide-scalar kernel, which unlike every other register-tiled
// variant serves ALL FOUR scalar types -- that is the whole point of it, so
// there is deliberately no float-only skip here.
//
// The oracle is Tiled16, not the vendor. Every other forced-kernel test in
// this file compares against BATCHLAS_GEMM_ROUTE=vendor, which is the
// stronger oracle where a vendor exists -- but this kernel's entire reason to
// exist is the vendor-free build, and there the forced-vendor arm degrades
// back to a native route. A test whose reference silently becomes the thing
// under test still passes; it just stops testing. Tiled16 is an independent
// implementation (one accumulator per thread, std::complex operator*,
// scalar epilogue) that is present in both builds, so this comparison means
// the same thing either way.
TYPED_TEST(GemmTest, BatchedGemmForcedSyclRegister64x64K16WideAligned) {
    using ScalarType = typename TestFixture::ScalarType;
    SKIP_UNLESS_NATIVE(*this->ctx);
    constexpr Backend BackendType = TestFixture::BackendType;

    // Exact multiple of 64 in m and n and of 16 in k, so the unpredicated
    // fast path is the one that runs.
    constexpr int size = 256;
    constexpr int batch_size = 2;
    auto A = Matrix<ScalarType>::Random(size, size, false, batch_size);
    auto B = Matrix<ScalarType>::Random(size, size, false, batch_size);
    auto C = Matrix<ScalarType>::Random(size, size, false, batch_size);
    auto C_ref = C.clone();

    {
        const GemmPin force_kernel("wide:m=64:n=64:k=16");
        (void)gemm(*(this->ctx), A.view(), B.view(), C.view(),
             {.alpha = ScalarType(2), .beta = ScalarType(-1)});
    }
    {
        const GemmPin force_kernel("tiled");
        (void)gemm(*(this->ctx), A.view(), B.view(), C_ref.view(),
             {.alpha = ScalarType(2), .beta = ScalarType(-1)});
    }
    this->ctx->wait();

    auto tol = test_utils::tolerance<ScalarType>() * 100;
    ASSERT_TRUE(AssertBatchedMatrixNear(C, C_ref, size, size, batch_size, tol));
}

TYPED_TEST(GemmTest, BatchedGemmForcedSyclRegister64x64K16WideRagged) {
    using ScalarType = typename TestFixture::ScalarType;
    SKIP_UNLESS_NATIVE(*this->ctx);
    constexpr Backend BackendType = TestFixture::BackendType;

    // Deliberately ragged in all three dimensions: m and n are not multiples
    // of 64 and k is not a multiple of 16, so every tile edge is predicated
    // and the k loop has a partial final step.
    constexpr int m = 200;
    constexpr int n = 130;
    constexpr int k = 70;
    constexpr int batch_size = 3;
    auto A = Matrix<ScalarType>::Random(m, k, false, batch_size);
    auto B = Matrix<ScalarType>::Random(k, n, false, batch_size);
    auto C = Matrix<ScalarType>::Random(m, n, false, batch_size);
    auto C_ref = C.clone();

    // alpha != 1 and beta != 0 on purpose: a beta == 0 test structurally
    // cannot see an epilogue defect, and the epilogue is where the two paths
    // differ most.
    {
        const GemmPin force_kernel("wide:m=64:n=64:k=16");
        (void)gemm(*(this->ctx), A.view(), B.view(), C.view(),
             {.alpha = ScalarType(2), .beta = ScalarType(-1)});
    }
    {
        const GemmPin force_kernel("tiled");
        (void)gemm(*(this->ctx), A.view(), B.view(), C_ref.view(),
             {.alpha = ScalarType(2), .beta = ScalarType(-1)});
    }
    this->ctx->wait();

    auto tol = test_utils::tolerance<ScalarType>() * 100;
    ASSERT_TRUE(AssertBatchedMatrixNear(C, C_ref, m, n, batch_size, tol));
}

TYPED_TEST(GemmTest, BatchedGemmForcedSyclVariantConjugateTranspose) {
    using ScalarType = typename TestFixture::ScalarType;
    constexpr Backend BackendType = TestFixture::BackendType;

    constexpr int m = 18;
    constexpr int n = 14;
    constexpr int k = 12;
    constexpr int batch_size = 2;

    auto A = Matrix<ScalarType>::Random(k, m, false, batch_size);
    auto B = Matrix<ScalarType>::Random(k, n, false, batch_size);
    auto C = Matrix<ScalarType>::Zeros(m, n, batch_size);
    auto C_ref = Matrix<ScalarType>::Zeros(m, n, batch_size);

    {
        const GemmPin force_variant("native");
        (void)gemm(*(this->ctx),
                          A.view(),
                          B.view(),
                          C.view(),
                          {.alpha = ScalarType(1), .beta = ScalarType(0), .transA = Transpose::ConjTrans});
    }

    {
        const GemmPin vendor_variant("vendor");
        (void)gemm(*(this->ctx),
                          A.view(),
                          B.view(),
                          C_ref.view(),
                          {.alpha = ScalarType(1), .beta = ScalarType(0), .transA = Transpose::ConjTrans});
    }

    this->ctx->wait();

    auto tol = test_utils::tolerance<ScalarType>() * 50;
    ASSERT_TRUE(AssertBatchedMatrixNear(C, C_ref, m, n, batch_size, tol));
}

// ---------------------------------------------------------------------------
// The 128x128x8 kernel on genuine SUB-VIEWS.
//
// Every operand a blocked/panel algorithm hands to gemm is a sub-view carrying
// its PARENT's leading dimension: a 128-row A with lda=512. No other test in
// this file produces one -- they all build standalone Matrix objects, where
// ld == rows by construction. That matters for two separate reasons:
//
//   * can_use_128x128_fast_path (register_128x128.hh:71-91) tests ld % 4 and a
//     16-byte base pointer, NOT contiguity, so the ALIGNED leg can and does
//     fire on a strided sub-view. Nothing has ever checked that it is right
//     there.
//   * MatrixView::operator()(Slice, Slice) offsets the base by
//     c_start*ld + r_start (matrix.hh:1129-1141), so an odd r_start breaks the
//     16-byte alignment and drops the same call onto the PREDICATED leg.
//
// Both legs are exercised below, and the comparison is over the WHOLE parent,
// not just the sub-block, so a write that escapes the view's logical extent
// fails the test rather than going unnoticed.
// ---------------------------------------------------------------------------
TYPED_TEST(GemmTest, BatchedGemmForcedSyclRegister128x128K8SubViewAlignedLeg) {
    using ScalarType = typename TestFixture::ScalarType;
    SKIP_UNLESS_NATIVE(*this->ctx);
    constexpr Backend BackendType = TestFixture::BackendType;

    if constexpr (!std::is_same_v<ScalarType, float>) {
        GTEST_SKIP() << "128x128x8 SYCL register kernel is float-only";
    } else {

    constexpr int P = 512;          // parent order
    constexpr int batch_size = 2;
    constexpr int m = 128, n = 128, k = 128;
    constexpr int r0 = 128;         // multiple of 4: base stays 16-byte aligned

    auto PA = Matrix<ScalarType>::Random(P, P, false, batch_size);
    auto PB = Matrix<ScalarType>::Random(P, P, false, batch_size);
    auto PC = Matrix<ScalarType>::Random(P, P, false, batch_size);
    auto PC_ref = PC.clone();

    auto Asub = [&](Matrix<ScalarType>& M) { return M.view()(Slice(r0, r0 + m), Slice(0, k)); };
    auto Bsub = [&](Matrix<ScalarType>& M) { return M.view()(Slice(0, k), Slice(0, n)); };
    auto Csub = [&](Matrix<ScalarType>& M) { return M.view()(Slice(r0, r0 + m), Slice(0, n)); };

    ASSERT_EQ(Asub(PA).ld(), P);    // the point of the test
    ASSERT_NE(Asub(PA).ld(), Asub(PA).rows());

    {
        const GemmPin force_kernel("reg:m=128:n=128:k=8:u=1");
        (void)gemm(*(this->ctx), Asub(PA), Bsub(PB), Csub(PC),
             {.alpha = ScalarType(2), .beta = ScalarType(-1)});
    }
    {
        const GemmPin vendor_variant("vendor");
        (void)gemm(*(this->ctx), Asub(PA), Bsub(PB), Csub(PC_ref),
             {.alpha = ScalarType(2), .beta = ScalarType(-1)});
    }
    this->ctx->wait();

    auto tol = test_utils::tolerance<ScalarType>() * 100;
    ASSERT_TRUE(AssertBatchedMatrixNear(PC, PC_ref, P, P, batch_size, tol));
    }
}

// The same aligned sub-view at k = 32, which takes the STAGED epilogue: its
// column-per-warp stores are the ones that could leave the view's extent.
// ARMED BREAK (R9): a second, escaping store of each staged column at
// `dst + 128 * ldc`. EXPECTED: RED at parent (128, 128), outside the
// sub-block, while the sub-block itself is right. Observed.
TYPED_TEST(GemmTest, Forced128x128K8SubViewStagedEpilogue) {
    using ScalarType = typename TestFixture::ScalarType;
    SKIP_UNLESS_NATIVE(*this->ctx);
    if constexpr (!std::is_same_v<ScalarType, float>) {
        GTEST_SKIP() << "128x128x8 SYCL register kernel is float-only";
    } else {
        constexpr int P = 512, batch_size = 4, m = 128, n = 128, k = 32, r0 = 128;
        auto PA = Matrix<ScalarType>::Random(P, P, false, batch_size);
        auto PB = Matrix<ScalarType>::Random(P, P, false, batch_size);
        auto PC = Matrix<ScalarType>::Random(P, P, false, batch_size);
        auto PC_ref = PC.clone();
        auto Asub = [&](Matrix<ScalarType>& M) { return M.view()(Slice(r0, r0 + m), Slice(0, k)); };
        auto Bsub = [&](Matrix<ScalarType>& M) { return M.view()(Slice(0, k), Slice(0, n)); };
        auto Csub = [&](Matrix<ScalarType>& M) { return M.view()(Slice(r0, r0 + m), Slice(0, n)); };
        {
            const GemmPin force_kernel("reg:m=128:n=128:k=8:u=1");
            (void)gemm(*(this->ctx), Asub(PA), Bsub(PB), Csub(PC),
                       {.alpha = ScalarType(2), .beta = ScalarType(-1)});
        }
        {
            const GemmPin vendor_variant("vendor");
            (void)gemm(*(this->ctx), Asub(PA), Bsub(PB), Csub(PC_ref),
                       {.alpha = ScalarType(2), .beta = ScalarType(-1)});
        }
        this->ctx->wait();
        auto tol = test_utils::tolerance<ScalarType>() * 100;
        ASSERT_TRUE(AssertBatchedMatrixNear(PC, PC_ref, P, P, batch_size, tol));
    }
}

TYPED_TEST(GemmTest, BatchedGemmForcedSyclRegister128x128K8SubViewPredicatedLeg) {
    using ScalarType = typename TestFixture::ScalarType;
    SKIP_UNLESS_NATIVE(*this->ctx);
    constexpr Backend BackendType = TestFixture::BackendType;

    if constexpr (!std::is_same_v<ScalarType, float>) {
        GTEST_SKIP() << "128x128x8 SYCL register kernel is float-only";
    } else {

    constexpr int P = 512;
    constexpr int batch_size = 2;
    constexpr int m = 200, n = 130, k = 70;   // ragged in all three
    constexpr int r0 = 3;                     // NOT a multiple of 4

    auto PA = Matrix<ScalarType>::Random(P, P, false, batch_size);
    auto PB = Matrix<ScalarType>::Random(P, P, false, batch_size);
    auto PC = Matrix<ScalarType>::Random(P, P, false, batch_size);
    auto PC_ref = PC.clone();

    auto Asub = [&](Matrix<ScalarType>& M) { return M.view()(Slice(r0, r0 + m), Slice(0, k)); };
    auto Bsub = [&](Matrix<ScalarType>& M) { return M.view()(Slice(0, k), Slice(0, n)); };
    auto Csub = [&](Matrix<ScalarType>& M) { return M.view()(Slice(r0, r0 + m), Slice(0, n)); };

    {
        const GemmPin force_kernel("reg:m=128:n=128:k=8:u=1");
        (void)gemm(*(this->ctx), Asub(PA), Bsub(PB), Csub(PC),
             {.alpha = ScalarType(2), .beta = ScalarType(-1)});
    }
    {
        const GemmPin vendor_variant("vendor");
        (void)gemm(*(this->ctx), Asub(PA), Bsub(PB), Csub(PC_ref),
             {.alpha = ScalarType(2), .beta = ScalarType(-1)});
    }
    this->ctx->wait();

    auto tol = test_utils::tolerance<ScalarType>() * 100;
    ASSERT_TRUE(AssertBatchedMatrixNear(PC, PC_ref, P, P, batch_size, tol));
    }
}


// The wide-scalar kernel's predicated leg on a SUB-VIEW: ragged extents AND a
// leading dimension inherited from a 512-wide parent, with a row offset of 3 so
// neither the base pointer nor the ld is 16-byte aligned for any scalar. This
// is the shape a panel update actually hands to gemm, and it is the leg the
// router would have to reach for complex. The Ragged test above is contiguous
// (ld == rows), so it cannot see an ld-dependent staging defect.
TYPED_TEST(GemmTest, BatchedGemmForcedSyclRegister64x64K16WideSubViewPredicatedLeg) {
    using ScalarType = typename TestFixture::ScalarType;
    SKIP_UNLESS_NATIVE(*this->ctx);
    constexpr Backend BackendType = TestFixture::BackendType;

    constexpr int P = 512;
    constexpr int batch_size = 2;
    constexpr int m = 200, n = 130, k = 70;   // ragged in all three
    constexpr int r0 = 3;                     // unaligned base for every scalar

    auto PA = Matrix<ScalarType>::Random(P, P, false, batch_size);
    auto PB = Matrix<ScalarType>::Random(P, P, false, batch_size);
    auto PC = Matrix<ScalarType>::Random(P, P, false, batch_size);
    auto PC_ref = PC.clone();

    auto Asub = [&](Matrix<ScalarType>& M) { return M.view()(Slice(r0, r0 + m), Slice(0, k)); };
    auto Bsub = [&](Matrix<ScalarType>& M) { return M.view()(Slice(0, k), Slice(0, n)); };
    auto Csub = [&](Matrix<ScalarType>& M) { return M.view()(Slice(r0, r0 + m), Slice(0, n)); };

    {
        const GemmPin force_kernel("wide:m=64:n=64:k=16");
        (void)gemm(*(this->ctx), Asub(PA), Bsub(PB), Csub(PC),
             {.alpha = ScalarType(2), .beta = ScalarType(-1)});
    }
    {
        // Reference is Tiled16, NOT the vendor -- matching WideAligned (:2063)
        // and WideRagged (:2094). A vendor reference is INERT in exactly the
        // vendor-free build this kernel exists for: a vendor pin falls back to
        // the automatic choice when no vendor is present, which for this
        // shape is the kernel under test, so the test would compare it against
        // itself and pass over any defect.
        const GemmPin force_kernel("tiled");
        (void)gemm(*(this->ctx), Asub(PA), Bsub(PB), Csub(PC_ref),
             {.alpha = ScalarType(2), .beta = ScalarType(-1)});
    }
    this->ctx->wait();

    auto tol = test_utils::tolerance<ScalarType>() * 100;
    ASSERT_TRUE(AssertBatchedMatrixNear(PC, PC_ref, P, P, batch_size, tol));
}

// ===========================================================================
// P6: the wide-scalar TRANSPOSED register family.
//
// Four variants, forceable by name only -- the selector does not reach them
// yet, deliberately: a selector row without a measured grid behind it is the
// defect this campaign has already shipped once. See
// docs/perf/gemm.md#wide-scalar-transposed-tiles for the window each is
// expected to win and what must be measured before it opens.
//
// THE ORACLE IS Tiled16, NEVER THE VENDOR, for the reason the NN wide tests
// give above: these kernels exist for the vendor-free build, and there a
// forced-vendor arm degrades back to a native route -- which for these shapes
// would be the kernel under test. The test would then compare it against
// itself and pass over any defect. Tiled16 is an independent implementation
// (one accumulator per thread, std::complex operator*, scalar epilogue,
// runtime transpose) present in both builds.
//
// EVERY SHAPE HERE IS RAGGED IN AT LEAST ONE DIMENSION and is taken as a
// SUB-VIEW of a wider parent, so the leading dimension is inherited and the
// base pointer is offset off any 16-byte boundary. That is what a panel update
// actually hands to gemm, and it is the only way the predicated staging leg
// and the predicated epilogue get exercised. The comparison is over the WHOLE
// parent, not the sub-view, so a write past the tile edge is caught rather
// than landing in slack the assertion never reads.
// ===========================================================================

namespace {

// beta is a parameter because a beta == 0 test is structurally blind to the
// epilogue's read-modify-write and a beta != 0 test is blind to an accumulator
// that was never zeroed; both are needed. alpha is always != 1 so that a
// dropped alpha cannot pass.
template <typename ScalarType>
void RunForcedWideTransposedAgainstTiled16(Queue& ctx,
                                           const char* kernel_name,
                                           int m,
                                           int n,
                                           int k,
                                           Transpose transA,
                                           Transpose transB,
                                           ScalarType beta,
                                           int parent = 512,
                                           int row_offset = 3,
                                           int batch_size = 3) {
    SKIP_UNLESS_NATIVE(ctx);
    const int a_rows = transA == Transpose::NoTrans ? m : k;
    const int a_cols = transA == Transpose::NoTrans ? k : m;
    const int b_rows = transB == Transpose::NoTrans ? k : n;
    const int b_cols = transB == Transpose::NoTrans ? n : k;

    auto PA = Matrix<ScalarType>::Random(parent, parent, false, batch_size);
    auto PB = Matrix<ScalarType>::Random(parent, parent, false, batch_size);
    auto PC = Matrix<ScalarType>::Random(parent, parent, false, batch_size);
    auto PC_ref = PC.clone();

    auto Av = [&](Matrix<ScalarType>& M) {
        return M.view()(Slice(row_offset, row_offset + a_rows), Slice(0, a_cols));
    };
    auto Bv = [&](Matrix<ScalarType>& M) {
        return M.view()(Slice(row_offset, row_offset + b_rows), Slice(0, b_cols));
    };
    auto Cv = [&](Matrix<ScalarType>& M) {
        return M.view()(Slice(row_offset, row_offset + m), Slice(0, n));
    };

    {
        const GemmPin force_kernel(kernel_name);
        (void)gemm(ctx, Av(PA), Bv(PB), Cv(PC),
             {.alpha = ScalarType(2), .beta = beta, .transA = transA, .transB = transB});
    }
    {
        const GemmPin force_kernel("tiled");
        (void)gemm(ctx, Av(PA), Bv(PB), Cv(PC_ref),
             {.alpha = ScalarType(2), .beta = beta, .transA = transA, .transB = transB});
    }
    ctx.wait();

    auto tol = test_utils::tolerance<ScalarType>() * 100;
    ASSERT_TRUE(AssertBatchedMatrixNear(PC, PC_ref, parent, parent, batch_size, tol));
}

}  // namespace

// The general square CN tile. ConjTrans on A is the most common transposed
// form in real complex demand.
TYPED_TEST(GemmTest, WideTransposedCN64Ragged) {
    using ScalarType = typename TestFixture::ScalarType;
    RunForcedWideTransposedAgainstTiled16<ScalarType>(
        *(this->ctx), "wide:m=64:n=64:k=16", 100, 70, 90,
        Transpose::ConjTrans, Transpose::NoTrans, ScalarType(-1));
}

TYPED_TEST(GemmTest, WideTransposedCN64BetaZero) {
    using ScalarType = typename TestFixture::ScalarType;
    RunForcedWideTransposedAgainstTiled16<ScalarType>(
        *(this->ctx), "wide:m=64:n=64:k=16", 100, 70, 90,
        Transpose::ConjTrans, Transpose::NoTrans, ScalarType(0));
}

// The small NN wide tiles (32x32 and 16x16 macro tiles). Ragged
// in every dimension, k from 1 to past two staged blocks, both betas, sub-views
// of a wider parent (inherited ld, offset base).
// ARMED BREAK: pass beta = T(0) to the 16x16 tile only. OBSERVED on this branch:
// red only on GemmTest/{4..7} (CUDA), only the 16x16x16wide beta=-1 cases.
TYPED_TEST(GemmTest, SmallWideNNTilesMatchTiled16) {
    using ScalarType = typename TestFixture::ScalarType;
    const char* kernels[] = {"wide:m=32:n=32:k=16", "wide:m=16:n=16:k=16"};
    const int shapes[][3] = {{32, 32, 1}, {32, 32, 70}, {29, 31, 257}, {16, 16, 33},
                             {13, 7, 100}, {32, 17, 8}, {40, 70, 65}, {5, 3, 16}};
    for (const char* kname : kernels) {
        for (const auto& s : shapes) {
            for (ScalarType beta : {ScalarType(-1), ScalarType(0)}) {
                SCOPED_TRACE(std::string(kname) + " m=" + std::to_string(s[0]) + " n=" +
                             std::to_string(s[1]) + " k=" + std::to_string(s[2]));
                RunForcedWideTransposedAgainstTiled16<ScalarType>(
                    *(this->ctx), kname, s[0], s[1], s[2],
                    Transpose::NoTrans, Transpose::NoTrans, beta, 300, 3, 3);
            }
        }
    }
}

// The small wide tiles are NN-only, so can_run refuses every transposed form and
// the pin throws (it used to fall back to Tiled16 silently).
TYPED_TEST(GemmTest, SmallWideTilesRefuseTransposedRequest) {
    using ScalarType = typename TestFixture::ScalarType;
    SKIP_UNLESS_NATIVE(*this->ctx);
    const Transpose pairs[][2] = {{Transpose::ConjTrans, Transpose::NoTrans},
                                  {Transpose::NoTrans, Transpose::Trans},
                                  {Transpose::Trans, Transpose::ConjTrans}};
    for (const char* kname : {"wide:m=32:n=32:k=16", "wide:m=16:n=16:k=16"})
        for (const auto& p : pairs) ExpectPinRefused<ScalarType>(*this->ctx, kname, 29, 21, 70, p[0], p[1]);
}

// The register launchers hard-wire OpA/OpB, so running one on a form it was not
// instantiated for computes the wrong answer (a `Trans` launcher on ConjTrans
// drops the conjugation). Before P3.4, 18 NN-only register variants did that on
// a forced transposed call. can_run now lists each config's forms (every config
// is covered by gemm_candidates_tests.cc TransposedPinOnAMissingInstantiation):
// the TN config serves a float CN call (a real ConjTrans is its Trans) and is
// refused for every other scalar (reg is float-only); NN-only configs are
// refused on NT, CN and TC. The reference is Tiled16, never the vendor.
TYPED_TEST(GemmTest, ForcedTransposedLauncherRejectsMismatchedTransposeForm) {
    using ScalarType = typename TestFixture::ScalarType;
    SKIP_UNLESS_NATIVE(*this->ctx);
    constexpr int m = 96, n = 96, k = 80;  // large enough to reach a 64x64 tile
    const Transpose T = Transpose::Trans, N = Transpose::NoTrans, Cj = Transpose::ConjTrans;
    if constexpr (!std::is_same_v<ScalarType, float>) {
        ExpectPinRefused<ScalarType>(*this->ctx, "reg:m=64:n=64:k=16:u=1", m, n, k, Cj, N);
    } else {
        RunForcedWideTransposedAgainstTiled16<ScalarType>(*this->ctx, "reg:m=64:n=64:k=16:u=1", m, n, k, Cj, N, ScalarType(-1),
                                                          128, 3, 2);
        for (const char* nn_only : {"reg:m=32:n=32:k=8:u=1", "reg:m=64:n=64:k=8:u=1", "reg:m=128:n=128:k=8:u=1", "reg:m=128:n=64:k=32:u=4", "reg:m=128:n=64:k=32:u=2"}) {
            ExpectPinRefused<ScalarType>(*this->ctx, nn_only, m, n, k, N, T);
            ExpectPinRefused<ScalarType>(*this->ctx, nn_only, m, n, k, Cj, N);
            ExpectPinRefused<ScalarType>(*this->ctx, nn_only, m, n, k, T, Cj);
        }
        ExpectPinRefused<ScalarType>(*this->ctx, "reg:m=128:n=64:k=16:u=1", m, n, k, N, N);  // no NN instantiation
        ExpectPinRefused<ScalarType>(*this->ctx, "reg:m=32:n=128:k=16:u=1", m, n, k, N, T);    // no NT instantiation
    }
}

// beta = 0 must not read C: NaN in the C sub-view has to vanish. The reference
// runs tiled16 on a zeroed C, since tiled16 may read C (known-defects.md #11).
// AssertBatchedMatrixNear passes a NaN, hence the explicit finite check.
// ARMED BREAK: `prior = *p` unconditionally in the launch_wide_transposed
// epilogue. OBSERVED: red only on this test, GemmTest/{4..7}.
TYPED_TEST(GemmTest, SmallWideBetaZeroNeverReadsC) {
    using ScalarType = typename TestFixture::ScalarType;
    SKIP_UNLESS_NATIVE(*this->ctx);
    using Real = typename batchlas::base_type<ScalarType>::type;
    constexpr int parent = 300, off = 3, batch = 3, m = 29, n = 21, k = 70;
    for (const char* kname : {"wide:m=32:n=32:k=16", "wide:m=16:n=16:k=16"}) {
        SCOPED_TRACE(kname);
        auto PA = Matrix<ScalarType>::Random(parent, parent, false, batch);
        auto PB = Matrix<ScalarType>::Random(parent, parent, false, batch);
        auto PC = Matrix<ScalarType>::Random(parent, parent, false, batch);
        auto PC_ref = PC.clone();
        for (int b = 0; b < batch; ++b)
            for (int c = 0; c < n; ++c)
                for (int r = off; r < off + m; ++r) {
                    PC(r, c, b) = ScalarType(std::numeric_limits<Real>::quiet_NaN());
                    PC_ref(r, c, b) = ScalarType(0);
                }
        auto sub = [&](Matrix<ScalarType>& M, int rows, int cols) {
            return M.view()(Slice(off, off + rows), Slice(0, cols));
        };
        const GemmOptions<ScalarType> opts{.alpha = ScalarType(2), .beta = ScalarType(0)};
        {
            const GemmPin force_kernel(kname);
            (void)gemm(*(this->ctx), sub(PA, m, k), sub(PB, k, n), sub(PC, m, n), opts);
        }
        {
            const GemmPin force_kernel("tiled");
            (void)gemm(*(this->ctx), sub(PA, m, k), sub(PB, k, n), sub(PC_ref, m, n), opts);
        }
        this->ctx->wait();
        for (int b = 0; b < batch; ++b)
            for (int c = 0; c < n; ++c)
                for (int r = off; r < off + m; ++r)
                    ASSERT_TRUE(std::isfinite(std::abs(PC(r, c, b))))
                        << "b=" << b << " r=" << r << " c=" << c;
        auto tol = test_utils::tolerance<ScalarType>() * 100;
        ASSERT_TRUE(AssertBatchedMatrixNear(PC, PC_ref, parent, parent, batch, tol));
    }
}

// Saturating batch for the shared-memory staged tiles: every item holds the
// same matrices, so every result must be bit-identical to item 0.
TYPED_TEST(GemmTest, SmallWideSaturatingBatchIsBitIdentical) {
    using ScalarType = typename TestFixture::ScalarType;
    SKIP_UNLESS_NATIVE(*this->ctx);
    constexpr int m = 31, n = 29, k = 300, batch = 2048;
    auto A1 = Matrix<ScalarType>::Random(m, k, false, 1);
    auto B1 = Matrix<ScalarType>::Random(k, n, false, 1);
    auto C1 = Matrix<ScalarType>::Random(m, n, false, 1);
    Matrix<ScalarType> A(m, k, batch), B(k, n, batch);
    auto a1 = A1.data(), b1 = B1.data(), c1 = C1.data();
    for (const char* kname : {"wide:m=32:n=32:k=16", "wide:m=16:n=16:k=16"}) {
        Matrix<ScalarType> C(m, n, batch);
        auto a = A.data(), b = B.data(), c = C.data();
        for (int item = 0; item < batch; ++item) {
            for (int i = 0; i < m * k; ++i) a[item * m * k + i] = a1[i];
            for (int i = 0; i < k * n; ++i) b[item * k * n + i] = b1[i];
            for (int i = 0; i < m * n; ++i) c[item * m * n + i] = c1[i];
        }
        {
            const GemmPin force_kernel(kname);
            (void)gemm(*(this->ctx), A.view(), B.view(), C.view(),
                 {.alpha = ScalarType(2), .beta = ScalarType(-1)});
        }
        this->ctx->wait();
        c = C.data();
        int mismatched = 0;
        for (int item = 1; item < batch; ++item) {
            for (int i = 0; i < m * n; ++i) {
                if (std::memcmp(&c[item * m * n + i], &c[i], sizeof(ScalarType)) != 0) {
                    ++mismatched;
                }
            }
        }
        EXPECT_EQ(mismatched, 0) << kname;
    }
}

// The small batched kernel (max(m, n, k) <= 64, real scalars; a complex or a 65 pin
// throws -- it used to fall back to Direct). Ragged edges in every bucket, both transposes of each
// operand, beta = 0 (the C read is skipped) and beta != 0, and batch 67 so the last
// work-group holds a partial set of matrices.
// ARMED BREAK (R9): drop `c < n` from small_batched.hh's epilogue guard.
// EXPECTED: RED on every n that is not a whole bucket.
TYPED_TEST(GemmTest, SmallBatchedMatchesVendorOnRaggedShapes) {
    using ScalarType = typename TestFixture::ScalarType;
    const Transpose ops[] = {Transpose::NoTrans, Transpose::Trans};
    const int shapes[][3] = {{1, 1, 1}, {5, 3, 7}, {8, 8, 8}, {13, 9, 16},
                             {16, 16, 16}, {17, 32, 5}, {31, 29, 23}, {32, 32, 32},
                             {33, 40, 64}, {48, 50, 61}, {64, 64, 64}, {64, 7, 3}};
    SKIP_UNLESS_NATIVE(*this->ctx);
    if constexpr (test_utils::is_complex<ScalarType>::value) {
        ExpectPinRefused<ScalarType>(*this->ctx, "small", 13, 9, 16, Transpose::NoTrans, Transpose::NoTrans);
        return;
    }
    ExpectPinRefused<ScalarType>(*this->ctx, "small", 65, 8, 8, Transpose::NoTrans, Transpose::NoTrans);
    ExpectPinRefused<ScalarType>(*this->ctx, "small", 8, 8, 65, Transpose::Trans, Transpose::NoTrans);
    for (Transpose ta : ops) {
        for (Transpose tb : ops) {
            for (const auto& s : shapes) {
                for (ScalarType beta : {ScalarType(0), ScalarType(-1.5)}) {
                    SCOPED_TRACE(::testing::Message() << "m=" << s[0] << " n=" << s[1]
                                                      << " k=" << s[2]);
                    RunForcedSyclGemmKernelCompare<ScalarType, TestFixture::BackendType>(
                        *(this->ctx), "small", s[0], s[1], s[2], 67, ta, tb, 75,
                        ScalarType(0.5), beta);
                }
            }
        }
    }
}

// The 4x4-tiled leg (float NN, 32 < max(m, n, k) <= 56, buckets 48 and 56):
// ragged in every dimension, k below one 4-step, both betas (two different
// instantiations: beta != 0 prefetches C), NaN C at beta = 0, batch 67.
// ARMED BREAK (R9): `l0 < kp` -> `l0 < kp - 4` in the tiled k loop.
// EXPECTED: SmallTiled RED on every case, and the tiled sub-view cases below;
// SmallBatchedMatchesVendor (NB = 64 and transposed shapes) green. Observed.
// ARMED BREAK (R9): prefetched C read at `Cb[r + c * ldc + 1]`.
// EXPECTED: RED on every beta != 0 case only. Observed.
TYPED_TEST(GemmTest, SmallTiledMatchesVendor) {
    using ScalarType = typename TestFixture::ScalarType;
    if constexpr (!std::is_same_v<ScalarType, float>) {
        GTEST_SKIP() << "the 4x4-tiled leg is float-only";
    } else {
        const int shapes[][3] = {{33, 33, 33}, {40, 40, 40}, {48, 48, 48}, {33, 48, 17},
                                 {48, 36, 3}, {49, 56, 50}, {56, 56, 56}, {56, 33, 41},
                                 {37, 52, 5}, {50, 49, 1}};
        for (const auto& sh : shapes) {
            Run128x128Compare<ScalarType>(*(this->ctx), sh[0], sh[1], sh[2], 67, 0.5f, -1.5f, false, "small");
            Run128x128Compare<ScalarType>(*(this->ctx), sh[0], sh[1], sh[2], 67, 0.5f, 0.0f, true, "small");
        }
    }
}

// A sub-view of a larger parent: ld != m on all three operands, and the WHOLE parent is
// compared, so a store past the view's rows or columns is caught.
TYPED_TEST(GemmTest, SmallBatchedStridedSubviewWritesOnlyItsView) {
    using ScalarType = typename TestFixture::ScalarType;
    if constexpr (test_utils::is_complex<ScalarType>::value) GTEST_SKIP() << "small is real-only (refusal above)";
    RunForcedWideTransposedAgainstTiled16<ScalarType>(
        *(this->ctx), "small", 29, 31, 17, Transpose::NoTrans, Transpose::NoTrans,
        ScalarType(1), /*parent=*/64, /*row_offset=*/3, /*batch_size=*/5);
    RunForcedWideTransposedAgainstTiled16<ScalarType>(
        *(this->ctx), "small", 12, 7, 32, Transpose::Trans, Transpose::Trans,
        ScalarType(0), /*parent=*/64, /*row_offset=*/2, /*batch_size=*/5);
    // The tiled leg (NN, max 33..56), both instantiations.
    RunForcedWideTransposedAgainstTiled16<ScalarType>(
        *(this->ctx), "small", 45, 51, 38, Transpose::NoTrans, Transpose::NoTrans,
        ScalarType(1), /*parent=*/96, /*row_offset=*/3, /*batch_size=*/5);
    RunForcedWideTransposedAgainstTiled16<ScalarType>(
        *(this->ctx), "small", 35, 40, 44, Transpose::NoTrans, Transpose::NoTrans,
        ScalarType(0), /*parent=*/96, /*row_offset=*/1, /*batch_size=*/5);
}

TYPED_TEST(GemmTest, WideTransposedNC64Ragged) {
    using ScalarType = typename TestFixture::ScalarType;
    RunForcedWideTransposedAgainstTiled16<ScalarType>(
        *(this->ctx), "wide:m=64:n=64:k=16", 100, 70, 90,
        Transpose::NoTrans, Transpose::ConjTrans, ScalarType(-1));
}

// The potrf trailing panel verbatim: A22 -= L21 L21^H is m_trailing x W x nb
// with W = 32 and nb = 96 for complex<float>. m is ragged, n is EXACTLY the
// tile width and k an exact multiple of the k step -- the combination the
// driver produces and which no square test reaches.
TYPED_TEST(GemmTest, WideTransposedNC128x32PotrfTrailingShape) {
    using ScalarType = typename TestFixture::ScalarType;
    RunForcedWideTransposedAgainstTiled16<ScalarType>(
        *(this->ctx), "wide:m=128:n=32:k=16", 200, 32, 96,
        Transpose::NoTrans, Transpose::ConjTrans, ScalarType(1));
}

// The other half of the same driver step: the W x W diagonal block, where m is
// 32 against a 128-row macro tile, so three quarters of the tile is predicated
// away. A kernel that mishandles a mostly-empty m tile passes the panel test
// above and fails here.
TYPED_TEST(GemmTest, WideTransposedNC128x32PotrfDiagonalBlockShape) {
    using ScalarType = typename TestFixture::ScalarType;
    RunForcedWideTransposedAgainstTiled16<ScalarType>(
        *(this->ctx), "wide:m=128:n=32:k=16", 32, 32, 96,
        Transpose::NoTrans, Transpose::ConjTrans, ScalarType(0));
}

// The geqrf panel update W1 = V^H A22 verbatim: nb x n2 x m_panel with nb = 32.
// k is large and ragged, so the k loop's partial final step on the TRANSPOSED
// operand is reached here and by none of the potrf shapes.
TYPED_TEST(GemmTest, WideTransposedCN32x128GeqrfPanelShape) {
    using ScalarType = typename TestFixture::ScalarType;
    RunForcedWideTransposedAgainstTiled16<ScalarType>(
        *(this->ctx), "wide:m=32:n=128:k=16", 32, 200, 140,
        Transpose::ConjTrans, Transpose::NoTrans, ScalarType(0));
}

// ---------------------------------------------------------------------------
// THE WIDENING AND THE GUARD, which are one line of can_run read two ways.
//
// wide_form<T> lets ONE ConjTrans instantiation serve a Trans request for a
// REAL scalar, because conj is the identity there -- that is what makes a
// single config able to serve potrf_blocked.cc's kTrailingTransB<T>, which is
// ConjTrans for complex and Trans for real. For a COMPLEX scalar the same
// substitution conjugates an operand that must not be conjugated and returns a
// plausible wrong matrix, so the pin must be refused (it used to fall back).
// Removing the guard leaves real passing and turns complex red.
// ---------------------------------------------------------------------------
TYPED_TEST(GemmTest, WideTransposedRealTransWideningAndComplexRefusal) {
    using ScalarType = typename TestFixture::ScalarType;
    SKIP_UNLESS_NATIVE(*this->ctx);
    if constexpr (test_utils::is_complex<ScalarType>::value)
        ExpectPinRefused<ScalarType>(*this->ctx, "wide:m=128:n=32:k=16", 100, 32, 96, Transpose::NoTrans,
                                     Transpose::Trans);
    else
        RunForcedWideTransposedAgainstTiled16<ScalarType>(*(this->ctx), "wide:m=128:n=32:k=16", 100, 32, 96,
                                                          Transpose::NoTrans, Transpose::Trans, ScalarType(-1));
}

// The mirror of the above on the A leg: a Trans request against a ConjTrans
// instantiation of the CN tile.
TYPED_TEST(GemmTest, WideTransposedRealTransWideningOnALeg) {
    using ScalarType = typename TestFixture::ScalarType;
    SKIP_UNLESS_NATIVE(*this->ctx);
    if constexpr (test_utils::is_complex<ScalarType>::value)
        ExpectPinRefused<ScalarType>(*this->ctx, "wide:m=32:n=128:k=16", 32, 100, 90, Transpose::Trans,
                                     Transpose::NoTrans);
    else
        RunForcedWideTransposedAgainstTiled16<ScalarType>(*(this->ctx), "wide:m=32:n=128:k=16", 32, 100, 90,
                                                          Transpose::Trans, Transpose::NoTrans, ScalarType(-1));
}

// A NoTrans request against a config with only transposing instantiations is
// refused for EVERY type, real included: the widening is Trans <-> ConjTrans
// only. A spelling names a config, and the call's form picks the instantiation,
// so on NN it runs the NN tile.
TYPED_TEST(GemmTest, WideTransposedRefusesNoTransRequest) {
    using ScalarType = typename TestFixture::ScalarType;
    SKIP_UNLESS_NATIVE(*this->ctx);
    ExpectPinRefused<ScalarType>(*this->ctx, "wide:m=128:n=32:k=16", 100, 70, 90, Transpose::NoTrans, Transpose::NoTrans);
    ExpectPinRefused<ScalarType>(*this->ctx, "wide:m=32:n=128:k=16", 100, 70, 90, Transpose::NoTrans, Transpose::NoTrans);
    RunForcedWideTransposedAgainstTiled16<ScalarType>(*(this->ctx), "wide:m=64:n=64:k=16", 100, 70, 90,
                                                      Transpose::NoTrans, Transpose::NoTrans, ScalarType(-1));
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
