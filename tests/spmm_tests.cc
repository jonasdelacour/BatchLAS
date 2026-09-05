#include <gtest/gtest.h>
#include <string_view>
#include <batchlas/backend_config.h>
#include <batchlas/blas/linalg.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/blas/functions/spmm.hh>
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>
#include "test_utils.hh"

using namespace batchlas;

// spmm coverage: all nine (transA, transB) spellings, per-item sparsity
// patterns, heterogeneous nnz, empty/unsorted/duplicate rows, padded leading
// dimensions and strides, and the alpha/beta corners. Fixtures are hand-built:
// convert_to<CSR> is correct only for SQUARE inputs, so a generated rectangular
// fixture would compare the kernel against a garbage A.
// evidence: docs/perf/spmm.md#correctness-findings

template <typename T, Backend B>
struct TestConfig {
    using ScalarType = T;
    static constexpr Backend BackendVal = B;
};

using MyTypes = typename test_utils::backend_types<TestConfig>::type;

namespace {

// Magnitudes in [-1, 1], so a long reduction cannot lose the tolerance.
template <typename T>
T spmm_cov_value(int seed) {
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

template <typename T>
bool spmm_is_finite(const T& v) {
    if constexpr (test_utils::is_complex<T>::value) {
        return std::isfinite(v.real()) && std::isfinite(v.imag());
    } else {
        return std::isfinite(v);
    }
}

// One batch item's CSR structure. row_offsets are ITEM-LOCAL and ro[0] == 0 --
// the convention convert_to<CSR> produces and MatrixView::nnz(b) assumes.
struct SpmmPattern {
    std::vector<int> ro;   // m + 1 entries, ro[0] == 0
    std::vector<int> ci;   // ro.back() entries
};

struct SpmmPatternSpec {
    int m = 0;
    int kA = 0;
    int batch = 1;
    int nnz_per_row = 3;
    // Item-local vs global col_indices base -- an identity at batch 1.
    bool distinct_patterns = false;
    bool heterogeneous_nnz = false;
    bool empty_rows = false;
    bool unsorted_cols = false;
    bool duplicate_cols = false;
};

// Columns are distinct by construction, so the duplicate and unsorted axes
// stay independent.
std::vector<SpmmPattern> spmm_build_pattern(const SpmmPatternSpec& p) {
    std::vector<SpmmPattern> out(static_cast<size_t>(std::max(0, p.batch)));
    for (int b = 0; b < p.batch; ++b) {
        SpmmPattern& it = out[static_cast<size_t>(b)];
        it.ro.assign(static_cast<size_t>(std::max(0, p.m)) + 1, 0);
        for (int i = 0; i < p.m; ++i) {
            int want = p.nnz_per_row;
            if (p.heterogeneous_nnz) want += ((i + 2 * b) % 3) - 1;
            if (p.empty_rows && ((i + b) % 4) == 0) want = 0;
            want = std::clamp(want, 0, p.kA);

            std::vector<int> cols;
            cols.reserve(static_cast<size_t>(want) + 1);
            if (want > 0) {
                // Strictly increasing for want <= kA, so no accidental duplicate.
                const int start = (i * 7 + (p.distinct_patterns ? (b * 5 + 1) : 0)) % p.kA;
                for (int t = 0; t < want; ++t) {
                    const int pos =
                        static_cast<int>(static_cast<int64_t>(t) * p.kA / want);
                    cols.push_back((start + pos) % p.kA);
                }
                std::sort(cols.begin(), cols.end());
                if (p.duplicate_cols && (i % 2) == 0) {
                    // Adjacent, so the row stays non-decreasing: not also unsorted.
                    const int dup = cols.front();
                    cols.insert(cols.begin() + 1, dup);
                }
                if (p.unsorted_cols && cols.size() > 1) {
                    // Neither ascending nor descending -- what convert_to<CSR> emits.
                    std::rotate(cols.begin(), cols.begin() + 1, cols.end());
                }
            }
            it.ro[static_cast<size_t>(i) + 1] =
                it.ro[static_cast<size_t>(i)] + static_cast<int>(cols.size());
            it.ci.insert(it.ci.end(), cols.begin(), cols.end());
        }
    }
    return out;
}

}  // namespace

template <typename Config>
class SpmmCoverageTest : public test_utils::BatchLASTest<Config> {
protected:
    using S = typename Config::ScalarType;
    using R = typename batchlas::base_type<S>::type;
    static constexpr Backend BackendType = Config::BackendVal;

    struct Case {
        // A AS STORED: op(A) is m x kA under NoTrans, kA x m otherwise, so
        // out_rows and red_rows are derived in run_case, never given here.
        int m = 0;
        int kA = 0;
        int nrhs = 1;
        int batch = 1;
        int nnz_per_row = 3;
        Transpose transA = Transpose::NoTrans;
        Transpose transB = Transpose::NoTrans;
        S alpha = S(1);
        S beta = S(0);

        // --- structure axes ---
        bool distinct_patterns = false;
        bool heterogeneous_nnz = false;
        bool empty_rows = false;
        bool unsorted_cols = false;
        bool duplicate_cols = false;

        // --- memory-state axes ---
        // NaN at an out-of-range column (2^30) above each item's own nnz. Live
        // for the NoTrans gather only: under Trans the scatter's range guard
        // discards the entry before the NaN is ever multiplied.
        bool poison_padding = false;
        // A finite sentinel at an IN-RANGE column -- the only poison the
        // transposed scatter accepts, so the only one that makes its nnz bound
        // observable. evidence: docs/perf/spmm.md#the-eleventh-blind-guard
        bool poison_padding_in_range = false;
        // C starts non-finite; only meaningful with beta == 0, where C must not
        // be read at all.
        bool c_starts_nan = false;
        // A's live values start non-finite; only meaningful with alpha == 0.
        bool a_starts_nan = false;
        // One column of B uninitialised and its output discarded -- the lanczos
        // shape. Only expressible with transB == NoTrans.
        int b_nan_col = -1;

        // --- stride and leading-dimension pads ---
        int matrix_stride_pad = 0;   // A: values / col_indices slots per item
        int offset_stride_pad = 0;   // A: row_offsets slots per item
        int b_stride_pad = 0;
        int c_stride_pad = 0;
        int ldb_pad = 0;
        int ldc_pad = 0;
    };

    // Builds the fixture, asserts the decision surface of every view, runs the
    // public spmm, then checks every element and every byte that must not move.
    void run_case(const Case& c) {
        if (!this->ctx) return;

        const bool a_nt = (c.transA == Transpose::NoTrans);
        const bool b_nt = (c.transB == Transpose::NoTrans);
        const int out_rows = a_nt ? c.m : c.kA;     // rows of C
        const int red_rows = a_nt ? c.kA : c.m;     // rows of op(B)
        const int b_rows = b_nt ? red_rows : c.nrhs;
        const int b_cols = b_nt ? c.nrhs : red_rows;
        ASSERT_TRUE(c.b_nan_col < 0 || b_nt)
            << "b_nan_col names a STORED column and is only meaningful for "
               "transB == NoTrans";

        // ---- A, hand-built ---------------------------------------------------
        const std::vector<SpmmPattern> items = spmm_build_pattern(
            SpmmPatternSpec{c.m, c.kA, c.batch, c.nnz_per_row,
                            c.distinct_patterns, c.heterogeneous_nnz,
                            c.empty_rows, c.unsorted_cols, c.duplicate_cols});

        int max_nnz = 0;
        for (int b = 0; b < c.batch; ++b) {
            max_nnz = std::max(
                max_nnz, static_cast<int>(items[static_cast<size_t>(b)].ci.size()));
        }
        // Per-item CAPACITY is the batch maximum, as convert_to<CSR> sizes it.
        const int matrix_stride = std::max(1, max_nnz + c.matrix_stride_pad);
        const int offset_stride = c.m + 1 + c.offset_stride_pad;

        const R poison_r = R(1e3);
        const S poison = static_cast<S>(poison_r);
        const S nan_v = static_cast<S>(std::numeric_limits<R>::quiet_NaN());
        const int idx_poison = 1 << 30;

        // The in-range pad column: a column of A indexes an extent-kA axis either way.
        const int pad_col = std::max(0, c.kA - 1);
        const S pad_sentinel = static_cast<S>(R(8192));
        ASSERT_FALSE(c.poison_padding && c.poison_padding_in_range)
            << "the two padding poisons are alternatives, not a pair: "
               "out-of-range is swallowed by the scatter's range guard and "
               "in-range is what survives it";

        UnifiedVector<S> a_val(
            static_cast<size_t>(matrix_stride) * static_cast<size_t>(c.batch));
        UnifiedVector<int> a_ci(
            static_cast<size_t>(matrix_stride) * static_cast<size_t>(c.batch));
        UnifiedVector<int> a_ro(
            static_cast<size_t>(offset_stride) * static_cast<size_t>(c.batch));

        for (int b = 0; b < c.batch; ++b) {
            const SpmmPattern& it = items[static_cast<size_t>(b)];
            const int nnz_b = static_cast<int>(it.ci.size());
            const size_t ro_base = static_cast<size_t>(b) * offset_stride;
            const size_t v_base = static_cast<size_t>(b) * matrix_stride;
            for (int t = 0; t <= c.m; ++t) {
                a_ro[ro_base + static_cast<size_t>(t)] = it.ro[static_cast<size_t>(t)];
            }
            // The offset_stride pad. Nothing may read past index m of an item.
            for (int t = c.m + 1; t < offset_stride; ++t) {
                a_ro[ro_base + static_cast<size_t>(t)] = idx_poison;
            }
            for (int p = 0; p < nnz_b; ++p) {
                a_ci[v_base + static_cast<size_t>(p)] = it.ci[static_cast<size_t>(p)];
                a_val[v_base + static_cast<size_t>(p)] =
                    c.a_starts_nan ? nan_v
                                   : spmm_cov_value<S>(b * 104729 + p * 31 + 7);
            }
            // Slots above this item's own nnz; the default fill is in range
            // either way, so an over-read is visible.
            for (int p = nnz_b; p < matrix_stride; ++p) {
                int ci_fill = 0;
                S val_fill = poison;
                if (c.poison_padding) {
                    ci_fill = idx_poison;
                    val_fill = nan_v;
                } else if (c.poison_padding_in_range) {
                    ci_fill = pad_col;
                    val_fill = pad_sentinel;
                }
                a_ci[v_base + static_cast<size_t>(p)] = ci_fill;
                a_val[v_base + static_cast<size_t>(p)] = val_fill;
            }
        }

        // ---- B ---------------------------------------------------------------
        const int ldb = std::max(1, b_rows + c.ldb_pad);
        const int str_b = std::max({1, ldb * b_cols + c.b_stride_pad, ldb});
        UnifiedVector<S> b_data(
            static_cast<size_t>(str_b) * static_cast<size_t>(c.batch));
        b_data.fill(poison);   // the ld pad rows AND the stride pad tail
        for (int b = 0; b < c.batch; ++b) {
            for (int col = 0; col < b_cols; ++col) {
                for (int row = 0; row < b_rows; ++row) {
                    const bool dead_col = (c.b_nan_col >= 0 && col == c.b_nan_col);
                    b_data[static_cast<size_t>(b) * str_b +
                           static_cast<size_t>(col) * ldb + static_cast<size_t>(row)] =
                        dead_col ? nan_v
                                 : spmm_cov_value<S>(b * 7919 + col * 131 + row + 3);
                }
            }
        }

        // ---- C, plus a guard band --------------------------------------------
        const int ldc = std::max(1, out_rows + c.ldc_pad);
        const int str_c = std::max({1, ldc * c.nrhs + c.c_stride_pad, ldc});
        // The transposed body writes C through an index taken from col_indices,
        // so a stray write past the last item needs somewhere to land.
        constexpr int kGuard = 64;
        const size_t c_live = static_cast<size_t>(str_c) * static_cast<size_t>(c.batch);
        UnifiedVector<S> c_data(c_live + kGuard);
        std::vector<S> c_initial(c_live);
        for (size_t t = 0; t < c_live; ++t) {
            const S v = c.c_starts_nan
                            ? nan_v
                            : spmm_cov_value<S>(static_cast<int>(t) * 29 + 11);
            c_data[t] = v;
            c_initial[t] = v;
        }
        const S guard_v = static_cast<S>(R(-98765));
        for (int t = 0; t < kGuard; ++t) c_data[c_live + static_cast<size_t>(t)] = guard_v;

        // ---- views, and the assertions that keep this case from being vacuous
        MatrixView<S, MatrixFormat::CSR> A_view(a_val.data(), a_ro.data(), a_ci.data(),
                                                c.m, c.kA, NonZeros{max_nnz},
                                                matrix_stride, offset_stride, c.batch);
        MatrixView<S, MatrixFormat::Dense> B_view(b_data.data(), b_rows, b_cols, ldb,
                                                  str_b, c.batch);
        MatrixView<S, MatrixFormat::Dense> C_view(c_data.data(), out_rows, c.nrhs, ldc,
                                                  str_c, c.batch);

        // matrix_stride and offset_stride are ADJACENT ints among nine positional
        // constructor arguments, so the decision surface is asserted, not trusted.
        ASSERT_EQ(A_view.rows(), c.m);
        ASSERT_EQ(A_view.cols(), c.kA);
        ASSERT_EQ(A_view.batch_size(), c.batch);
        ASSERT_EQ(A_view.matrix_stride(), matrix_stride);
        ASSERT_EQ(A_view.offset_stride(), offset_stride);
        ASSERT_EQ(A_view.nnz(), max_nnz);
        ASSERT_EQ(B_view.ld(), ldb);
        ASSERT_EQ(B_view.stride(), str_b);
        ASSERT_EQ(B_view.rows(), b_rows);
        ASSERT_EQ(B_view.cols(), b_cols);
        ASSERT_EQ(C_view.ld(), ldc);
        ASSERT_EQ(C_view.stride(), str_c);
        ASSERT_EQ(C_view.rows(), out_rows);
        ASSERT_EQ(C_view.cols(), c.nrhs);
        // Row offsets must be ITEM-LOCAL: a global array would turn nnz(b) into
        // a running total.
        for (int b = 0; b < c.batch; ++b) {
            ASSERT_EQ(A_view.nnz(b),
                      static_cast<int>(items[static_cast<size_t>(b)].ci.size()))
                << "item " << b << " row offsets are not item-local";
        }

        // ---- the call --------------------------------------------------------
        const bool transposed = !a_nt || !b_nt;
        // Keyed on a NATIVE pin, not on "a pin exists": a vendor pin can land on
        // a backend that legitimately refuses transposes, turning the
        // pre-existing NETLIB skips into failures.
        const char* const route_pin = std::getenv("BATCHLAS_SPMM_ROUTE");
        const std::string_view pin_text = route_pin ? route_pin : "";
        const bool route_pinned = pin_text.find("native") != std::string_view::npos ||
                                  pin_text == "direct" || pin_text == "cta" ||
                                  pin_text == "blocked";
        UnifiedVector<std::byte> ws;
        try {
            const size_t need = spmm_buffer_size(*(this->ctx), A_view, B_view, C_view,
                                                 c.alpha, c.beta, c.transA, c.transB);
            if (need > 0) ws.resize(need);
            spmm(*(this->ctx), A_view, B_view, C_view, c.alpha, c.beta,
                 c.transA, c.transB, ws);
            this->ctx->wait();
        } catch (const std::exception& e) {
            if (transposed && !route_pinned) {
                GTEST_SKIP() << "backend "
                             << test_utils::backend_to_string(BackendType)
                             << " REFUSED transA=" << static_cast<int>(c.transA)
                             << " transB=" << static_cast<int>(c.transB)
                             << " -- a MISSING ROUTE, not a wrong answer: "
                             << e.what();
            }
            FAIL() << "spmm threw on backend "
                   << test_utils::backend_to_string(BackendType) << ": " << e.what();
        }

        // ---- the reference, written FROM THE DEFINITION ----------------------
        // Accumulated over the NONZEROS so one loop serves both directions and
        // duplicate columns SUM -- MatrixView::at returns only the FIRST match.
        const R tol = test_utils::tolerance<S>();
        for (int b = 0; b < c.batch; ++b) {
            const SpmmPattern& it = items[static_cast<size_t>(b)];
            const size_t n_out = static_cast<size_t>(std::max(1, out_rows * c.nrhs));
            std::vector<S> expect(n_out, S(0));
            std::vector<R> scale(n_out, R(0));

            // alpha == 0 must not read A here either: summing a NaN-filled A
            // would predict NaN and check nothing.
            if (c.alpha != S(0)) {
                for (int i = 0; i < c.m; ++i) {
                    const int rs = it.ro[static_cast<size_t>(i)];
                    const int re = it.ro[static_cast<size_t>(i) + 1];
                    for (int p = rs; p < re; ++p) {
                        S a = a_val[static_cast<size_t>(b) * matrix_stride +
                                    static_cast<size_t>(p)];
                        const int j = it.ci[static_cast<size_t>(p)];
                        // ConjTrans conjugates the SPARSE operand.
                        if constexpr (test_utils::is_complex<S>::value) {
                            if (c.transA == Transpose::ConjTrans) a = std::conj(a);
                        }
                        const int o_row = a_nt ? i : j;
                        const int r_row = a_nt ? j : i;
                        for (int col = 0; col < c.nrhs; ++col) {
                            S bv = b_nt
                                       ? b_data[static_cast<size_t>(b) * str_b +
                                                static_cast<size_t>(col) * ldb +
                                                static_cast<size_t>(r_row)]
                                       : b_data[static_cast<size_t>(b) * str_b +
                                                static_cast<size_t>(r_row) * ldb +
                                                static_cast<size_t>(col)];
                            if constexpr (test_utils::is_complex<S>::value) {
                                if (c.transB == Transpose::ConjTrans) bv = std::conj(bv);
                            }
                            const size_t o = static_cast<size_t>(col) * out_rows +
                                             static_cast<size_t>(o_row);
                            expect[o] += a * bv;
                            scale[o] += std::abs(a) * std::abs(bv);
                        }
                    }
                }
                for (size_t o = 0; o < n_out; ++o) {
                    expect[o] *= c.alpha;
                    scale[o] *= std::abs(c.alpha);
                }
            }
            if (c.beta != S(0)) {
                for (int col = 0; col < c.nrhs; ++col) {
                    for (int o_row = 0; o_row < out_rows; ++o_row) {
                        const S c0 = c_initial[static_cast<size_t>(b) * str_c +
                                               static_cast<size_t>(col) * ldc +
                                               static_cast<size_t>(o_row)];
                        const size_t o = static_cast<size_t>(col) * out_rows +
                                         static_cast<size_t>(o_row);
                        expect[o] += c.beta * c0;
                        scale[o] += std::abs(c.beta) * std::abs(c0);
                    }
                }
            }

            for (int col = 0; col < c.nrhs; ++col) {
                // Uninitialised by construction; the claim is about the others.
                if (col == c.b_nan_col) continue;
                for (int o_row = 0; o_row < out_rows; ++o_row) {
                    const S got = c_data[static_cast<size_t>(b) * str_c +
                                         static_cast<size_t>(col) * ldc +
                                         static_cast<size_t>(o_row)];
                    const size_t o = static_cast<size_t>(col) * out_rows +
                                     static_cast<size_t>(o_row);
                    const S want = expect[o];
                    // A BACKWARD-ERROR denominator, never |expected|: the transposed
                    // body is an atomic scatter, not reproducible run to run.
                    const R denom = std::max(scale[o], R(1));
                    EXPECT_TRUE(spmm_is_finite(got))
                        << "batch " << b << " col " << col << " row " << o_row
                        << " came back non-finite";
                    EXPECT_LE(std::abs(got - want) / denom, tol)
                        << "batch " << b << " col " << col << " row " << o_row
                        << " got " << got << " expected " << want;
                }
            }

            // Every non-live slot of C must be UNTOUCHED: a body that ignored
            // ldc, or derived C's batch stride, passes every check above.
            for (int t = 0; t < str_c; ++t) {
                const int col = t / ldc;
                const int row = t - col * ldc;
                if (col < c.nrhs && row < out_rows) continue;
                const S before = c_initial[static_cast<size_t>(b) * str_c +
                                           static_cast<size_t>(t)];
                const S after = c_data[static_cast<size_t>(b) * str_c +
                                       static_cast<size_t>(t)];
                if (c.c_starts_nan) {
                    EXPECT_TRUE(!spmm_is_finite(after))
                        << "C pad slot " << t << " of batch " << b << " was written";
                } else {
                    EXPECT_EQ(after, before)
                        << "C pad slot " << t << " of batch " << b << " was written";
                }
            }
        }

        for (int t = 0; t < kGuard; ++t) {
            EXPECT_EQ(c_data[c_live + static_cast<size_t>(t)], guard_v)
                << "C guard band element " << t << " past the end of batch "
                << c.batch << " was written";
        }
    }
};

TYPED_TEST_SUITE(SpmmCoverageTest, MyTypes);

// --- 1. Baseline pair: batch 1 and batch > 1 on the SAME uniform pattern ---

TYPED_TEST(SpmmCoverageTest, SingleItemSquareNoTrans) {
    using S = typename TestFixture::ScalarType;
    typename TestFixture::Case c;
    c.m = 64; c.kA = 64; c.nrhs = 3; c.batch = 1; c.nnz_per_row = 4;
    c.alpha = static_cast<S>(1.5); c.beta = static_cast<S>(-0.75);
    this->run_case(c);
}

TYPED_TEST(SpmmCoverageTest, BatchedSquareUniformPatternNoTrans) {
    using S = typename TestFixture::ScalarType;
    typename TestFixture::Case c;
    c.m = 64; c.kA = 64; c.nrhs = 3; c.batch = 4; c.nnz_per_row = 4;
    c.alpha = static_cast<S>(1.5); c.beta = static_cast<S>(-0.75);
    this->run_case(c);
}

// --- 2. Distinct sparsity patterns per batch item ---
// Dropping the b*matrix_stride base is an identity at batch 1 and on a uniform
// batch; only this shape sees it.

TYPED_TEST(SpmmCoverageTest, DistinctPatternsAcrossBatch) {
    using S = typename TestFixture::ScalarType;
    typename TestFixture::Case c;
    c.m = 48; c.kA = 48; c.nrhs = 5; c.batch = 4; c.nnz_per_row = 4;
    c.distinct_patterns = true;
    c.alpha = static_cast<S>(1.25); c.beta = static_cast<S>(0.5);
    this->run_case(c);
}

TYPED_TEST(SpmmCoverageTest, DistinctPatternsAcrossBatchTrans) {
    using S = typename TestFixture::ScalarType;
    typename TestFixture::Case c;
    c.m = 48; c.kA = 48; c.nrhs = 5; c.batch = 4; c.nnz_per_row = 4;
    c.distinct_patterns = true;
    c.transA = Transpose::Trans;
    c.alpha = static_cast<S>(1.25); c.beta = static_cast<S>(0.5);
    this->run_case(c);
}

// --- 3. Heterogeneous nnz, and the uninitialised padding above it ---
// The only legal bound on the nonzero loop is the item's own row_offsets[i+1];
// A.nnz() is the per-item CAPACITY.
// evidence: docs/perf/spmm.md#three-vendor-defects-found-here-and-fixed

TYPED_TEST(SpmmCoverageTest, HeterogeneousNnzAcrossBatch) {
    using S = typename TestFixture::ScalarType;
    typename TestFixture::Case c;
    c.m = 40; c.kA = 40; c.nrhs = 4; c.batch = 4; c.nnz_per_row = 3;
    c.heterogeneous_nnz = true; c.empty_rows = true;
    c.alpha = static_cast<S>(-1.75); c.beta = static_cast<S>(0.25);
    this->run_case(c);
}

// The gather has no range guard on the column index, so the poison is live here.
TYPED_TEST(SpmmCoverageTest, PaddingAboveNnzIsNotRead) {
    using S = typename TestFixture::ScalarType;
    typename TestFixture::Case c;
    c.m = 40; c.kA = 40; c.nrhs = 4; c.batch = 4; c.nnz_per_row = 3;
    c.heterogeneous_nnz = true;
    c.poison_padding = true;         // NaN values, column index 2^30
    c.matrix_stride_pad = 9;         // capacity strictly above the batch maximum
    c.alpha = static_cast<S>(1.0); c.beta = static_cast<S>(0.5);
    this->run_case(c);
}

// The UNPOISONED transposed over-read: the padding carries the default fill.
TYPED_TEST(SpmmCoverageTest, HeterogeneousNnzAcrossBatchTrans) {
    using S = typename TestFixture::ScalarType;
    typename TestFixture::Case c;
    c.m = 40; c.kA = 40; c.nrhs = 4; c.batch = 4; c.nnz_per_row = 3;
    c.heterogeneous_nnz = true; c.empty_rows = true;
    c.transA = Transpose::Trans;
    c.alpha = static_cast<S>(-1.75); c.beta = static_cast<S>(0.25);
    this->run_case(c);
}

// The transposed nnz bound. The poison is IN-RANGE deliberately: an out-of-range
// one is discarded by the scatter's range guard, which made this case vacuous.
// evidence: docs/perf/spmm.md#the-eleventh-blind-guard
TYPED_TEST(SpmmCoverageTest, PaddingAboveNnzIsNotReadTrans) {
    using S = typename TestFixture::ScalarType;
    typename TestFixture::Case c;
    c.m = 40; c.kA = 40; c.nrhs = 4; c.batch = 4; c.nnz_per_row = 3;
    c.heterogeneous_nnz = true;
    c.poison_padding_in_range = true;   // sentinel at column kA-1, NOT 2^30
    c.matrix_stride_pad = 9;            // capacity strictly above the batch maximum
    c.transA = Transpose::Trans;
    c.alpha = static_cast<S>(1.0); c.beta = static_cast<S>(0.5);
    this->run_case(c);
}

// NOT coverage of the transposed bound -- the range guard swallows it -- but the
// range guard's own regression test, and armed against a vendor that over-reads.
TYPED_TEST(SpmmCoverageTest, PaddingAboveNnzOutOfRangeIsNotReadTrans) {
    using S = typename TestFixture::ScalarType;
    typename TestFixture::Case c;
    c.m = 40; c.kA = 40; c.nrhs = 4; c.batch = 4; c.nnz_per_row = 3;
    c.heterogeneous_nnz = true;
    c.poison_padding = true;            // NaN at column 2^30
    c.matrix_stride_pad = 9;
    c.transA = Transpose::Trans;
    c.alpha = static_cast<S>(1.0); c.beta = static_cast<S>(0.5);
    this->run_case(c);
}

// --- 4. All six pads at once: four batch strides, two leading dimensions ---

TYPED_TEST(SpmmCoverageTest, StridePadsAreReadNotDerived) {
    using S = typename TestFixture::ScalarType;
    typename TestFixture::Case c;
    c.m = 37; c.kA = 37; c.nrhs = 6; c.batch = 3; c.nnz_per_row = 4;
    c.matrix_stride_pad = 5;
    c.offset_stride_pad = 3;
    c.b_stride_pad = 17;
    c.c_stride_pad = 23;
    c.ldb_pad = 4;
    c.ldc_pad = 7;
    c.alpha = static_cast<S>(1.5); c.beta = static_cast<S>(-0.5);
    this->run_case(c);
}

TYPED_TEST(SpmmCoverageTest, StridePadsAreReadNotDerivedTrans) {
    using S = typename TestFixture::ScalarType;
    typename TestFixture::Case c;
    c.m = 37; c.kA = 37; c.nrhs = 6; c.batch = 3; c.nnz_per_row = 4;
    c.matrix_stride_pad = 5;
    c.offset_stride_pad = 3;
    c.b_stride_pad = 17;
    c.c_stride_pad = 23;
    c.ldb_pad = 4;
    c.ldc_pad = 7;
    c.transA = Transpose::Trans;
    c.alpha = static_cast<S>(1.5); c.beta = static_cast<S>(-0.5);
    this->run_case(c);
}

// --- 5. The three structure axes RandomSparseHermitian cannot produce ---
// Empty rows must yield exactly beta*C; the same column twice in one row MUST SUM.

TYPED_TEST(SpmmCoverageTest, EmptyRowsContributeOnlyBeta) {
    using S = typename TestFixture::ScalarType;
    typename TestFixture::Case c;
    c.m = 44; c.kA = 44; c.nrhs = 3; c.batch = 3; c.nnz_per_row = 3;
    c.empty_rows = true;
    c.alpha = static_cast<S>(2.0); c.beta = static_cast<S>(-0.25);
    this->run_case(c);
}

TYPED_TEST(SpmmCoverageTest, EmptyRowsContributeOnlyBetaTrans) {
    using S = typename TestFixture::ScalarType;
    typename TestFixture::Case c;
    c.m = 44; c.kA = 44; c.nrhs = 3; c.batch = 3; c.nnz_per_row = 3;
    c.empty_rows = true;
    c.transA = Transpose::Trans;
    c.alpha = static_cast<S>(2.0); c.beta = static_cast<S>(-0.25);
    this->run_case(c);
}

TYPED_TEST(SpmmCoverageTest, UnsortedColumnsWithinARow) {
    using S = typename TestFixture::ScalarType;
    typename TestFixture::Case c;
    c.m = 36; c.kA = 52; c.nrhs = 4; c.batch = 3; c.nnz_per_row = 5;
    c.unsorted_cols = true;
    c.alpha = static_cast<S>(1.25); c.beta = static_cast<S>(0.5);
    this->run_case(c);
}

TYPED_TEST(SpmmCoverageTest, UnsortedColumnsWithinARowTrans) {
    using S = typename TestFixture::ScalarType;
    typename TestFixture::Case c;
    c.m = 36; c.kA = 52; c.nrhs = 4; c.batch = 3; c.nnz_per_row = 5;
    c.unsorted_cols = true;
    c.transA = Transpose::Trans;
    c.alpha = static_cast<S>(1.25); c.beta = static_cast<S>(0.5);
    this->run_case(c);
}

TYPED_TEST(SpmmCoverageTest, DuplicateColumnsSum) {
    using S = typename TestFixture::ScalarType;
    typename TestFixture::Case c;
    c.m = 36; c.kA = 36; c.nrhs = 4; c.batch = 3; c.nnz_per_row = 4;
    c.duplicate_cols = true;
    c.alpha = static_cast<S>(1.0); c.beta = static_cast<S>(0.0);
    this->run_case(c);
}

TYPED_TEST(SpmmCoverageTest, DuplicateColumnsSumTrans) {
    using S = typename TestFixture::ScalarType;
    typename TestFixture::Case c;
    c.m = 36; c.kA = 36; c.nrhs = 4; c.batch = 3; c.nnz_per_row = 4;
    c.duplicate_cols = true;
    c.transA = Transpose::Trans;
    c.alpha = static_cast<S>(1.0); c.beta = static_cast<S>(0.0);
    this->run_case(c);
}

// --- 6. The two scalar guards, made observable ---
// beta == 0 means C IS NOT READ -- callers pass unzeroed BumpAllocator memory.
// alpha == 0 means A is not read and the answer is beta*C, not a quick return.

TYPED_TEST(SpmmCoverageTest, BetaZeroDoesNotReadC) {
    using S = typename TestFixture::ScalarType;
    typename TestFixture::Case c;
    c.m = 40; c.kA = 40; c.nrhs = 4; c.batch = 3; c.nnz_per_row = 4;
    c.ldc_pad = 5; c.c_stride_pad = 11;
    c.alpha = static_cast<S>(1.0); c.beta = static_cast<S>(0.0);
    c.c_starts_nan = true;
    this->run_case(c);
}

TYPED_TEST(SpmmCoverageTest, BetaZeroDoesNotReadCTrans) {
    using S = typename TestFixture::ScalarType;
    typename TestFixture::Case c;
    c.m = 40; c.kA = 40; c.nrhs = 4; c.batch = 3; c.nnz_per_row = 4;
    c.ldc_pad = 5; c.c_stride_pad = 11;
    c.transA = Transpose::Trans;
    c.alpha = static_cast<S>(1.0); c.beta = static_cast<S>(0.0);
    c.c_starts_nan = true;
    this->run_case(c);
}

TYPED_TEST(SpmmCoverageTest, AlphaZeroScalesCAndDoesNotReadA) {
    using S = typename TestFixture::ScalarType;
    typename TestFixture::Case c;
    c.m = 33; c.kA = 33; c.nrhs = 3; c.batch = 3; c.nnz_per_row = 4;
    c.alpha = static_cast<S>(0.0); c.beta = static_cast<S>(0.5);
    c.a_starts_nan = true;
    this->run_case(c);
}

TYPED_TEST(SpmmCoverageTest, AlphaZeroScalesCAndDoesNotReadATrans) {
    using S = typename TestFixture::ScalarType;
    typename TestFixture::Case c;
    c.m = 33; c.kA = 33; c.nrhs = 3; c.batch = 3; c.nnz_per_row = 4;
    c.transA = Transpose::Trans;
    c.alpha = static_cast<S>(0.0); c.beta = static_cast<S>(0.5);
    c.a_starts_nan = true;
    this->run_case(c);
}

// --- 7. The lanczos shape: a column of B that is genuinely uninitialised ---
// lanczos writes column it+1 only after the spmm at iteration it.

TYPED_TEST(SpmmCoverageTest, UninitialisedBColumnDoesNotContaminateTheOthers) {
    using S = typename TestFixture::ScalarType;
    typename TestFixture::Case c;
    c.m = 96; c.kA = 96; c.nrhs = 2; c.batch = 4; c.nnz_per_row = 3;
    c.b_nan_col = 1;
    c.b_stride_pad = 96 * 3;
    c.alpha = static_cast<S>(1.0); c.beta = static_cast<S>(0.0);
    c.c_starts_nan = true;
    this->run_case(c);
}

// --- 8. Rectangular A, both orientations, both directions ---
// m > kA truncates a wrong OUTPUT extent, m < kA a wrong REDUCTION extent.

TYPED_TEST(SpmmCoverageTest, RectangularWideNoTrans) {
    using S = typename TestFixture::ScalarType;
    typename TestFixture::Case c;
    c.m = 37; c.kA = 91; c.nrhs = 4; c.batch = 3; c.nnz_per_row = 5;
    c.alpha = static_cast<S>(1.5); c.beta = static_cast<S>(-0.75);
    this->run_case(c);
}

TYPED_TEST(SpmmCoverageTest, RectangularTallNoTrans) {
    using S = typename TestFixture::ScalarType;
    typename TestFixture::Case c;
    c.m = 91; c.kA = 37; c.nrhs = 4; c.batch = 3; c.nnz_per_row = 5;
    c.alpha = static_cast<S>(1.5); c.beta = static_cast<S>(-0.75);
    this->run_case(c);
}

TYPED_TEST(SpmmCoverageTest, RectangularWideTrans) {
    using S = typename TestFixture::ScalarType;
    typename TestFixture::Case c;
    c.m = 37; c.kA = 91; c.nrhs = 4; c.batch = 3; c.nnz_per_row = 5;
    c.transA = Transpose::Trans;
    c.alpha = static_cast<S>(1.5); c.beta = static_cast<S>(-0.75);
    this->run_case(c);
}

TYPED_TEST(SpmmCoverageTest, RectangularTallTrans) {
    using S = typename TestFixture::ScalarType;
    typename TestFixture::Case c;
    c.m = 91; c.kA = 37; c.nrhs = 4; c.batch = 3; c.nnz_per_row = 5;
    c.transA = Transpose::Trans;
    c.alpha = static_cast<S>(1.5); c.beta = static_cast<S>(-0.75);
    this->run_case(c);
}

// --- 9. All nine transA x transB combinations, on one padded batched fixture ---
// The transA arms are DIFFERENT KERNEL BODIES (a gather for NoTrans, an atomic
// scatter otherwise), so the 3x3 cannot be sampled at its corners. Skips on
// Backend::NETLIB: docs/perf/spmm.md#the-transposed-refusal

namespace {
template <typename Case>
void spmm_fill_nine_shape(Case& c) {
    c.m = 45; c.kA = 61; c.nrhs = 5; c.batch = 3; c.nnz_per_row = 4;
    c.distinct_patterns = true;
    c.ldb_pad = 3; c.ldc_pad = 2;
    c.b_stride_pad = 13; c.c_stride_pad = 9;
    c.matrix_stride_pad = 4; c.offset_stride_pad = 2;
}
}  // namespace

#define SPMM_NINE_CASE(name, ta, tb)                                     \
    TYPED_TEST(SpmmCoverageTest, name) {                                 \
        using S = typename TestFixture::ScalarType;                      \
        typename TestFixture::Case c;                                    \
        spmm_fill_nine_shape(c);                                         \
        c.transA = Transpose::ta;                                        \
        c.transB = Transpose::tb;                                        \
        c.alpha = static_cast<S>(1.25);                                  \
        c.beta = static_cast<S>(-0.5);                                   \
        this->run_case(c);                                               \
    }

SPMM_NINE_CASE(NineNoTransNoTrans,     NoTrans,   NoTrans)
SPMM_NINE_CASE(NineNoTransTrans,       NoTrans,   Trans)
SPMM_NINE_CASE(NineNoTransConjTrans,   NoTrans,   ConjTrans)
SPMM_NINE_CASE(NineTransNoTrans,       Trans,     NoTrans)
SPMM_NINE_CASE(NineTransTrans,         Trans,     Trans)
SPMM_NINE_CASE(NineTransConjTrans,     Trans,     ConjTrans)
SPMM_NINE_CASE(NineConjTransNoTrans,   ConjTrans, NoTrans)
SPMM_NINE_CASE(NineConjTransTrans,     ConjTrans, Trans)
SPMM_NINE_CASE(NineConjTransConjTrans, ConjTrans, ConjTrans)

#undef SPMM_NINE_CASE

// --- 10. The nrhs ladder ---
// The widths the library asks for: 1 (python), 2 (lanczos), 3 (ritz_values),
// 12/25/50 (LOBPCG); 25 and 50 divide no plausible register-block width.

TYPED_TEST(SpmmCoverageTest, NrhsOne) {
    using S = typename TestFixture::ScalarType;
    typename TestFixture::Case c;
    c.m = 64; c.kA = 64; c.nrhs = 1; c.batch = 4; c.nnz_per_row = 3;
    c.distinct_patterns = true;
    c.alpha = static_cast<S>(1.5); c.beta = static_cast<S>(0.25);
    this->run_case(c);
}

TYPED_TEST(SpmmCoverageTest, NrhsTwo) {
    using S = typename TestFixture::ScalarType;
    typename TestFixture::Case c;
    c.m = 64; c.kA = 64; c.nrhs = 2; c.batch = 4; c.nnz_per_row = 3;
    c.distinct_patterns = true;
    c.alpha = static_cast<S>(1.5); c.beta = static_cast<S>(0.25);
    this->run_case(c);
}

TYPED_TEST(SpmmCoverageTest, NrhsThree) {
    using S = typename TestFixture::ScalarType;
    typename TestFixture::Case c;
    c.m = 64; c.kA = 64; c.nrhs = 3; c.batch = 4; c.nnz_per_row = 3;
    c.distinct_patterns = true;
    c.alpha = static_cast<S>(1.5); c.beta = static_cast<S>(0.25);
    this->run_case(c);
}

TYPED_TEST(SpmmCoverageTest, NrhsTwelve) {
    using S = typename TestFixture::ScalarType;
    typename TestFixture::Case c;
    c.m = 64; c.kA = 64; c.nrhs = 12; c.batch = 4; c.nnz_per_row = 3;
    c.distinct_patterns = true;
    c.alpha = static_cast<S>(1.5); c.beta = static_cast<S>(0.25);
    this->run_case(c);
}

TYPED_TEST(SpmmCoverageTest, NrhsTwentyFive) {
    using S = typename TestFixture::ScalarType;
    typename TestFixture::Case c;
    c.m = 64; c.kA = 64; c.nrhs = 25; c.batch = 4; c.nnz_per_row = 3;
    c.distinct_patterns = true;
    c.ldc_pad = 3;
    c.alpha = static_cast<S>(1.5); c.beta = static_cast<S>(0.25);
    this->run_case(c);
}

TYPED_TEST(SpmmCoverageTest, NrhsFifty) {
    using S = typename TestFixture::ScalarType;
    typename TestFixture::Case c;
    c.m = 64; c.kA = 64; c.nrhs = 50; c.batch = 4; c.nnz_per_row = 3;
    c.distinct_patterns = true;
    c.ldc_pad = 3;
    c.alpha = static_cast<S>(1.5); c.beta = static_cast<S>(0.25);
    this->run_case(c);
}

// The transposed arm has its own column-block loop; 50 is not a multiple of 4.
TYPED_TEST(SpmmCoverageTest, NrhsFiftyTrans) {
    using S = typename TestFixture::ScalarType;
    typename TestFixture::Case c;
    c.m = 64; c.kA = 64; c.nrhs = 50; c.batch = 4; c.nnz_per_row = 3;
    c.distinct_patterns = true;
    c.ldc_pad = 3;
    c.transA = Transpose::Trans;
    c.alpha = static_cast<S>(1.5); c.beta = static_cast<S>(0.25);
    this->run_case(c);
}

// --- 11. A complex alpha and beta ---
// Every alpha and beta above is real, so the cross-terms are unchecked until here.

TYPED_TEST(SpmmCoverageTest, ComplexAlphaBetaConjTrans) {
    using S = typename TestFixture::ScalarType;
    typename TestFixture::Case c;
    c.m = 41; c.kA = 41; c.nrhs = 4; c.batch = 3; c.nnz_per_row = 4;
    c.distinct_patterns = true;
    c.transA = Transpose::ConjTrans;
    c.ldc_pad = 2; c.c_stride_pad = 7;
    if constexpr (test_utils::is_complex<S>::value) {
        c.alpha = S(0.5, 1.25);
        c.beta = S(-0.75, 0.5);
    } else {
        c.alpha = static_cast<S>(0.5);
        c.beta = static_cast<S>(-0.75);
    }
    this->run_case(c);
}

// --- 12. A batch large enough that a launch spans several items per group ---
// m*batch = 2048, so an item boundary crossed inside a group is observable.

TYPED_TEST(SpmmCoverageTest, LargeBatchLanczosShape) {
    using S = typename TestFixture::ScalarType;
    typename TestFixture::Case c;
    c.m = 256; c.kA = 256; c.nrhs = 2; c.batch = 8; c.nnz_per_row = 3;
    c.distinct_patterns = true;
    c.b_stride_pad = 256;    // lanczos passes stride = (n+1)*n against cols = 2
    c.alpha = static_cast<S>(1.0); c.beta = static_cast<S>(-1.0);
    this->run_case(c);
}

TYPED_TEST(SpmmCoverageTest, LargeBatchLanczosShapeTrans) {
    using S = typename TestFixture::ScalarType;
    typename TestFixture::Case c;
    c.m = 256; c.kA = 256; c.nrhs = 2; c.batch = 8; c.nnz_per_row = 3;
    c.distinct_patterns = true;
    c.b_stride_pad = 256;
    c.transA = Transpose::Trans;
    c.alpha = static_cast<S>(1.0); c.beta = static_cast<S>(-1.0);
    this->run_case(c);
}

// --- 13. The degenerate extent ---
// nrhs == 0 is legal and C must come back COMPLETELY untouched, not scaled by
// beta. m == 0 is deliberately absent: vacuous under NoTrans.

TYPED_TEST(SpmmCoverageTest, ZeroNrhsLeavesCUntouched) {
    using S = typename TestFixture::ScalarType;
    typename TestFixture::Case c;
    c.m = 32; c.kA = 32; c.nrhs = 0; c.batch = 3; c.nnz_per_row = 3;
    c.ldc_pad = 4; c.c_stride_pad = 8;
    c.alpha = static_cast<S>(2.0); c.beta = static_cast<S>(0.5);
    this->run_case(c);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// The mutation checks that verify this suite: docs/perf/spmm.md#the-eleventh-blind-guard
