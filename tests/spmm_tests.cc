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

// ===========================================================================
// WP8 -- THE FIRST DEDICATED spmm TEST FILE IN THIS TREE.
//
// Before this file, spmm was exercised only as a SUBROUTINE of something else.
// The whole of its existing coverage is two calls in lanczos_tests.cc -- :186
// at batch 1 over a hand-written square CSR fixture, and :266 at batch 2 over
// TriDiagToeplitz(n).convert_to<CSR>() -- both NoTrans/NoTrans, both square,
// both at natural strides, and both checked by comparing an EIGENVALUE rather
// than the product. So every one of the following was unobserved:
//
//   * transA != NoTrans and transB != NoTrans -- EIGHT of the nine
//     combinations. netlib refuses them outright (netlib_lapack.cc:247-250)
//     and nothing in the tree asked cuSPARSE for them either.
//   * batch >= 2 WITH DIFFERENT SPARSITY PATTERNS PER ITEM. This is the axis
//     that separates an ITEM-LOCAL col_indices/row_offsets base (correct) from
//     a global one (wrong), and the distinction IS AN IDENTITY AT BATCH 1 --
//     so a body that confused them passed everything that existed.
//   * non-natural batch strides on B and C, and ld > rows on either. These are
//     the NORM, not the exception, at the real call sites: lanczos passes B
//     with rows=n, cols=2, ld=n and stride=(n+1)*n (lanczos.cc:53, :104), and
//     LOBPCG's X/P/R/AX/AP/AR are column slices of one buffer carrying
//     stride = 3*block_vectors*n against cols = block_vectors
//     (syevx_lobpcg.cc:332-341). WP7 recorded the matching hole for gemv: a
//     kernel that DERIVED the stride passed all 232 cases that existed.
//   * the slots of values/col_indices ABOVE an item's own nnz. convert_to<CSR>
//     sizes every item by the batch MAXIMUM (src/matrix.cc:473-478) and zeroes
//     only row_offsets (:489); UnifiedVector::resize never fills
//     (src/util/sycl-util-impl.cc:71-83). That padding is genuinely
//     uninitialised memory, and a body bounded by A.nnz() -- the per-item
//     CAPACITY, matrix.hh:1069-1073 -- walks straight into it.
//   * beta == 0 against a C that is not finite. Every in-library caller passes
//     beta = 0 into a BumpAllocator-allocated C, and BumpAllocator does not
//     zero (mempool.hh:80-92), so "beta == 0 does not read C" is a live
//     contract and not a nicety.
//
// FIXTURES ARE HAND-BUILT, DELIBERATELY. Matrix<T>::RandomSparseHermitian
// cannot reach four of the axes above: it emits sorted columns with an explicit
// diagonal in every row and an IDENTICAL nnz for every batch item
// (src/matrix.cc:1204-1207, 1249-1252; iluk_tests.cc:573 asserts the sorting),
// so no empty row, no unsorted row, no duplicate column and no heterogeneous
// batch can come out of it. And convert_to<CSR> is correct only for SQUARE
// inputs -- its population kernel derives r = id/rows, c = id%rows over
// id in [0, rows*cols) (src/matrix.cc:536-541) while its counting kernel
// correctly ranges c over cols (:398-413) -- so a RECTANGULAR fixture built
// that way compares the kernel against a garbage A and blames the kernel.
//
// THE REFERENCE IS WRITTEN FROM THE DEFINITION and indexes row_offsets and
// col_indices DIRECTLY. It is not transcribed from netlib (which throws on any
// transpose and applies beta unconditionally), and it never goes through
// MatrixView::at or KernelMatrixView::get: those LINEAR-SEARCH the row and
// return the FIRST entry whose column matches (matrix.hh:157-168), so a
// reference built on them silently drops the second of two duplicate columns
// and cannot test the duplicate-column axis at all.
//
// THE TOLERANCE DENOMINATOR IS A BACKWARD-ERROR SCALE -- sum |a|*|b| over the
// contributions to that output element, floored at 1 -- and NEVER |expected|.
// That is mandatory rather than stylistic here: the transposed path is an
// ATOMIC SCATTER, its summation order follows atomic arrival order, and it is
// therefore not bitwise reproducible from one run to the next. Comparing
// against |expected| is a cancellation detector, not a tolerance.
//
// ONE DELIBERATE WEAKENING, STATED SO IT IS NOT MISTAKEN FOR COVERAGE.
// A transposed case whose backend THROWS is reported as SKIPPED, not failed.
// The reason: netlib's spmm openly refuses every transpose
// (netlib_lapack.cc:247-250) and, because the WP8 preferred() clause admits
// only transA == NoTrans, a VENDOR-PRESENT build still routes every TRANSPOSED
// Backend::NETLIB spmm to netlib -- so an
// unconditional assertion there would be permanently red for a reason that has
// nothing to do with the answer. A WRONG ANSWER IS NEVER SKIPPED, only an
// outright refusal, and:
//   * the skip prints the backend, both transposes and the exception text;
//   * the skip is DISABLED when BATCHLAS_SPMM_ROUTE is set, so the pinned-route
//     runs that the deliberate breaks at the bottom of this file depend on
//     cannot be silently skipped past.
// It follows that a vendor which returns a STATUS CODE instead of throwing is
// NOT covered by this: cusparse.cc:45-87 checks no cuSPARSE status at all, so a
// CUSPARSE_STATUS_NOT_SUPPORTED leaves C untouched and this suite reports it as
// a wrong answer -- which is the correct outcome, and the reason for running
// the suite against the vendor before any native body exists.
// ===========================================================================

template <typename T, Backend B>
struct TestConfig {
    using ScalarType = T;
    static constexpr Backend BackendVal = B;
};

using MyTypes = typename test_utils::backend_types<TestConfig>::type;

namespace {

// Deterministic, reproducible, and -- for complex -- WITH A GENUINELY NON-ZERO
// IMAGINARY PART. Magnitudes stay in [-1, 1] so a reduction of a few hundred
// terms cannot lose the relative tolerance to cancellation. Same generator
// shape as gemv_cov_value (gemv_tests.cc:323-334), for the same reason.
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
// that is the convention convert_to<CSR> produces (a per-item
// joint_inclusive_scan seeded with 0, src/matrix.cc:506-511, pinned at
// matrix_tests.cc:604-609) and the convention MatrixView::nnz(b) assumes
// (matrix.hh:1082-1090 computes offsets[base + rows] - offsets[base]).
struct SpmmPattern {
    std::vector<int> ro;   // m + 1 entries, ro[0] == 0
    std::vector<int> ci;   // ro.back() entries
};

struct SpmmPatternSpec {
    int m = 0;
    int kA = 0;
    int batch = 1;
    int nnz_per_row = 3;
    // Different COLUMNS per batch item. The axis that separates an item-local
    // col_indices base from a global one; an identity at batch 1.
    bool distinct_patterns = false;
    // Different nnz per item, so matrix_stride > nnz(b) for the smaller items
    // and the padding above them is real rather than hypothetical.
    bool heterogeneous_nnz = false;
    bool empty_rows = false;
    bool unsorted_cols = false;
    bool duplicate_cols = false;
};

// The generator RandomSparseHermitian cannot be. Every column set is chosen
// distinct by construction, so a duplicate column only ever appears when
// duplicate_cols asks for one, and an unsorted row only when unsorted_cols
// does -- the axes stay separable rather than arriving together.
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
                // t * kA / want is strictly increasing for want <= kA, so the
                // positions are distinct before the rotation by `start` and
                // distinct after it; no accidental duplicate can appear.
                const int start = (i * 7 + (p.distinct_patterns ? (b * 5 + 1) : 0)) % p.kA;
                for (int t = 0; t < want; ++t) {
                    const int pos =
                        static_cast<int>(static_cast<int64_t>(t) * p.kA / want);
                    cols.push_back((start + pos) % p.kA);
                }
                std::sort(cols.begin(), cols.end());
                if (p.duplicate_cols && (i % 2) == 0) {
                    // Kept adjacent, so the row is still non-decreasing: this is
                    // a duplicate-column row and NOT also an unsorted one.
                    const int dup = cols.front();
                    cols.insert(cols.begin() + 1, dup);
                }
                if (p.unsorted_cols && cols.size() > 1) {
                    // Ascending with the smallest moved to the end: neither
                    // ascending nor descending. This is what convert_to<CSR>
                    // actually emits (relaxed atomic arrival order,
                    // src/matrix.cc:552-566) and what lanczos_tests and
                    // spmm_benchmark already feed spmm today.
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
        // A AS STORED. op(A) is m x kA under NoTrans and kA x m otherwise, so
        // out_rows and red_rows are DERIVED below and never given here.
        int m = 0;
        int kA = 0;
        int nrhs = 1;
        int batch = 1;
        int nnz_per_row = 3;
        Transpose transA = Transpose::NoTrans;
        Transpose transB = Transpose::NoTrans;
        S alpha = S(1);
        S beta = S(0);

        // --- structure axes, none of which RandomSparseHermitian can reach ---
        bool distinct_patterns = false;
        bool heterogeneous_nnz = false;
        bool empty_rows = false;
        bool unsorted_cols = false;
        bool duplicate_cols = false;

        // --- memory-state axes ---
        // Fills values/col_indices ABOVE each item's own nnz with a NaN and an
        // out-of-range column (2^30). That is the REAL state of that memory,
        // not a hypothetical: only row_offsets is zeroed (src/matrix.cc:489)
        // and UnifiedVector::resize never fills.
        //
        // READ THE WARNING BEFORE REUSING THIS ON A TRANSPOSED CASE. Under
        // NoTrans the column index is a READ index into op(B) and the body has
        // no guard on it, so BOTH halves of the poison are live -- but it is
        // the NaN VALUE that arms the assertion (fma_acc(acc, NaN, anything)
        // is NaN whatever the wild read returned), and the out-of-range index
        // is there to make a VENDOR over-read fault loudly rather than to arm
        // anything native. Under Trans the column index is an output row of C
        // and the scatter guards it (spmm_native.cc:541-542, an out-of-range
        // ATOMIC WRITE is heap corruption, so the guard is correct and stays):
        // the guard `continue`s BEFORE `av` is ever multiplied, so it discards
        // the NaN together with the index and the whole poison is swallowed.
        // A transposed case that wants to SEE an over-read must use
        // poison_padding_in_range instead. Measured, not assumed -- see B4 at
        // the bottom of this file.
        bool poison_padding = false;
        // Fills those same slots with a LARGE FINITE sentinel at an IN-RANGE
        // column index, i.e. an entry the kernel will happily accept and
        // accumulate. This is the poison that survives the scatter's range
        // guard, so it is the one that makes the TRANSPOSED nnz bound
        // observable.
        //
        // A LARGE FINITE SENTINEL RATHER THAN A NaN, AND THE REASON IS THE
        // ATOMIC. The transposed body accumulates into C with atomics, so:
        //   * NaN is ABSORBING under addition. One spurious entry and the
        //     element is NaN no matter how many more arrive, so the failure
        //     cannot distinguish an over-read of one slot from a sweep of the
        //     whole slab -- which is exactly the distinction that separates the
        //     sharp B4 break from the too-coarse form of B2.
        //   * NaN can only be caught by the FINITENESS assertion, which is a
        //     boolean and reports nothing about WHERE the extra entries went.
        //     A finite sentinel fails the backward-error comparison instead,
        //     names the (batch, col, row) it landed on -- the poison column, by
        //     construction -- and prints got vs expected, so the magnitude of
        //     the deviation counts the spurious nonzeros.
        //   * `isfinite` is the one assertion a fast-math device build is
        //     entitled to fold away. A magnitude error is not.
        // The sentinel is enormous next to the O(1) live data, and run_case's
        // tolerance denominator is a backward-error scale built ONLY from the
        // live nonzeros, so a single spurious entry misses by ~1e3 tolerances.
        bool poison_padding_in_range = false;
        // C starts non-finite. Only meaningful with beta == 0, where C must not
        // be read at all and the answer must come back FINITE.
        bool c_starts_nan = false;
        // A's live values start non-finite. Only meaningful with alpha == 0,
        // where A must not be read at all. Makes the alpha guard OBSERVABLE
        // rather than the arithmetic identity 0*x == 0 that no break can move.
        bool a_starts_nan = false;
        // One column of B is uninitialised and its output column is discarded.
        // THE LANCZOS SHAPE -- see section 7. Only expressible with
        // transB == NoTrans, where op(B)'s column c is a stored column.
        int b_nan_col = -1;

        // --- FOUR stride pads and TWO leading-dimension pads ---
        int matrix_stride_pad = 0;   // A: values / col_indices slots per item
        int offset_stride_pad = 0;   // A: row_offsets slots per item
        int b_stride_pad = 0;
        int c_stride_pad = 0;
        int ldb_pad = 0;
        int ldc_pad = 0;
    };

    // The one entry point. Builds the fixture, asserts the DECISION SURFACE of
    // every view before the call, runs the public spmm, and then checks every
    // element of every batch item plus every byte that must not have moved.
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
        // matrix_stride is the per-item CAPACITY -- the batch maximum, exactly
        // as convert_to<CSR> sizes it -- so on a heterogeneous batch the
        // smaller items really do carry slots nothing ever wrote.
        const int matrix_stride = std::max(1, max_nnz + c.matrix_stride_pad);
        const int offset_stride = c.m + 1 + c.offset_stride_pad;

        const R poison_r = R(1e3);
        const S poison = static_cast<S>(poison_r);
        const S nan_v = static_cast<S>(std::numeric_limits<R>::quiet_NaN());
        const int idx_poison = 1 << 30;

        // The IN-RANGE pad column, legal in BOTH directions: a column index of
        // A names a ROW of op(B) under NoTrans and an OUTPUT ROW of C under
        // Trans, and both extents are kA. kA - 1 is chosen deliberately over 0 --
        // row 0 is where a dropped batch base and a derived stride both land, so
        // putting the padding somewhere else keeps this axis separable from
        // those. The sentinel is ~1e4 against live data of magnitude <= 1.
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
            // The slots above this item's own nnz. Three fills, and which one
            // is chosen decides whether a transposed over-read is VISIBLE:
            //   default                  -- large value at column 0, in range.
            //                               An over-read corrupts the answer
            //                               detectably without ever leaving the
            //                               buffer, in EITHER direction.
            //   poison_padding           -- NaN at column 2^30. The real state
            //                               of that memory; arms the gather and
            //                               faults a vendor. SWALLOWED WHOLE by
            //                               the scatter's range guard.
            //   poison_padding_in_range  -- the sentinel at pad_col. Survives
            //                               the scatter's guard and lands in C.
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
        // The `, ldb` floor only ever binds when b_cols == 0 (the nrhs == 0
        // case); for any real width ldb*b_cols already exceeds it, so this
        // cannot quietly change a stride the case asked for.
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
        // 64 elements past the last batch item of C. WP7 recorded THREE separate
        // tail-masking breaks that stayed GREEN over 376 gemv cases purely
        // because a write past the end of the allocation landed where nothing
        // was looking (gemv_tests.cc:520-535). The transposed spmm body writes C
        // through an index TAKEN FROM col_indices, which is the strongest reason
        // in this tree to give a stray write somewhere observable to land.
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

        // The CSR constructor takes (data, row_offsets, col_indices, rows, cols,
        // NonZeros, matrix_stride, offset_stride, batch) -- nine positional
        // arguments of which five are plain ints, and matrix_stride and
        // offset_stride are ADJACENT and interchangeable at the call site with
        // no diagnostic. WP7's Trap 5 was exactly this shape on VectorView. So
        // the decision surface is asserted rather than trusted: without these a
        // constructor argument-order slip makes the whole case vacuous.
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
        // And that the row offsets really are ITEM-LOCAL: nnz(b) is
        // offsets[b*offset_stride + m] - offsets[b*offset_stride], so a globally
        // accumulated offset array would make this read a running total instead
        // of this item's count.
        for (int b = 0; b < c.batch; ++b) {
            ASSERT_EQ(A_view.nnz(b),
                      static_cast<int>(items[static_cast<size_t>(b)].ci.size()))
                << "item " << b << " row offsets are not item-local";
        }

        // ---- the call --------------------------------------------------------
        const bool transposed = !a_nt || !b_nt;
        // A NATIVE pin means "the native body must serve this shape", so a refusal
        // is a real failure and the skip below must not fire. Any other state --
        // no pin at all, or a pin naming the vendor -- can still land on a backend
        // that legitimately refuses transposes (netlib_lapack.cc hard-throws on
        // any transpose), so the skip stays armed there. Keying this on "a pin
        // exists" instead turned those 92 pre-existing NETLIB skips into 92
        // failures under BATCHLAS_SPMM_ROUTE=vendor, which looks exactly like a
        // routing regression and is not one.
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
        //
        // C = alpha * op(A) * op(B) + beta * C, accumulated over the NONZEROS
        // rather than over the outputs, so one loop serves both directions:
        //   NoTrans -- the nonzero at (i, j) contributes to output row i and
        //              reads op(B) row j.
        //   Trans   -- op(A)[j, i] IS A[i, j], so the same nonzero contributes
        //              to output row j and reads op(B) row i.
        // Two entries with the SAME column in one row therefore SUM, which is
        // what netlib's loop does (netlib_lapack.cc:253-257) and what any
        // reference built on KernelMatrixView::get would silently lose.
        const R tol = test_utils::tolerance<S>();
        for (int b = 0; b < c.batch; ++b) {
            const SpmmPattern& it = items[static_cast<size_t>(b)];
            const size_t n_out = static_cast<size_t>(std::max(1, out_rows * c.nrhs));
            std::vector<S> expect(n_out, S(0));
            std::vector<R> scale(n_out, R(0));

            // alpha == 0 NEVER READS A, in the reference exactly as in the
            // kernel. A reference that summed a NaN-filled A here would predict
            // NaN and the test would be checking nothing.
            if (c.alpha != S(0)) {
                for (int i = 0; i < c.m; ++i) {
                    const int rs = it.ro[static_cast<size_t>(i)];
                    const int re = it.ro[static_cast<size_t>(i) + 1];
                    for (int p = rs; p < re; ++p) {
                        S a = a_val[static_cast<size_t>(b) * matrix_stride +
                                    static_cast<size_t>(p)];
                        const int j = it.ci[static_cast<size_t>(p)];
                        // ConjTrans conjugates the SPARSE operand. std::conj on
                        // a real scalar returns std::complex, so the branch has
                        // to be compile-time.
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
                // The column fed by a deliberately uninitialised column of B.
                // Its output is discarded by the caller and is expected to be
                // garbage; the CLAIM is about the OTHER columns.
                if (col == c.b_nan_col) continue;
                for (int o_row = 0; o_row < out_rows; ++o_row) {
                    const S got = c_data[static_cast<size_t>(b) * str_c +
                                         static_cast<size_t>(col) * ldc +
                                         static_cast<size_t>(o_row)];
                    const size_t o = static_cast<size_t>(col) * out_rows +
                                     static_cast<size_t>(o_row);
                    const S want = expect[o];
                    // THE BACKWARD-ERROR DENOMINATOR. Not |expected|: the
                    // transposed body is an atomic scatter whose summation order
                    // is nondeterministic, so a correct answer legitimately moves
                    // between runs by O(eps * sum|a||b|). Floored at 1 so a tiny
                    // well-conditioned answer is still held to an absolute
                    // tolerance rather than an unbounded relative one.
                    const R denom = std::max(scale[o], R(1));
                    EXPECT_TRUE(spmm_is_finite(got))
                        << "batch " << b << " col " << col << " row " << o_row
                        << " came back non-finite";
                    EXPECT_LE(std::abs(got - want) / denom, tol)
                        << "batch " << b << " col " << col << " row " << o_row
                        << " got " << got << " expected " << want;
                }
            }

            // EVERY SLOT OF C THAT IS NOT A LIVE ELEMENT MUST BE UNTOUCHED --
            // the ld pad rows (out_rows .. ldc-1 of each column) and the stride
            // pad tail. A body that ignored ldc, or that derived C's batch
            // stride, writes here, and every value check above would still pass.
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

// ---------------------------------------------------------------------------
// 1. THE BASELINE PAIR: batch 1 and batch > 1 on the SAME uniform pattern.
//
// These two exist as a MATCHED PAIR and neither is redundant. Break B1 (see the
// bottom of this file: col_indices read without the per-item base) is an
// IDENTITY for both -- every item's column array is byte-identical -- so they
// are the CONTROL against which DistinctPatternsAcrossBatch is the experiment.
// A single batched test with a per-item pattern could not tell the two apart,
// and would leave "the break is about the batch" as an unexcluded explanation.
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// 2. DISTINCT SPARSITY PATTERNS PER BATCH ITEM.
//
// THE AXIS NOTHING IN THE TREE COVERED. Every existing spmm call runs at batch
// 1 or 2 over a pattern every item SHARES -- lanczos_tests' hand-written
// fixture repeats one row structure across the batch, TriDiagToeplitz repeats
// one matrix, and RandomSparseHermitian gives every item the same nnz -- so
// reading col_indices at `p` instead of `b*matrix_stride + p` produced the
// RIGHT ANSWER everywhere it was ever exercised. It is an identity
// at batch 1 by construction, and an identity at any batch whose items share a
// pattern; only this shape can see it.
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// 3. HETEROGENEOUS nnz, AND THE UNINITIALISED PADDING ABOVE IT.
//
// convert_to<CSR> sizes EVERY item by the batch maximum (src/matrix.cc:473-478)
// and only row_offsets is zeroed (:489); UnifiedVector::resize never fills
// (src/util/sycl-util-impl.cc:71-83). So the slots between an item's own nnz
// and matrix_stride hold whatever the allocator last left there. The only legal
// bound on the nonzero loop is the item's own row_offsets[i+1]; A.nnz() is the
// per-item CAPACITY (matrix.hh:1069-1073), i.e. the batch maximum, and a body
// bounded by it walks into the padding of every smaller item.
//
// WHAT THESE THREE FOUND, AND WHY THEY ARE NOT SKIPPED ANYWHERE.
// They were written against the native bodies and they immediately caught a
// PRE-EXISTING WRONG ANSWER IN THE cuSPARSE ADAPTER, which is the reason this
// note exists rather than a skip. cusparseCreateCsr takes ONE nnz and
// cusparseCsrSetStridedBatch adds only a batch count and two strides -- the
// descriptor has no per-item nnz at all -- and src/backends/backend_handle_impl.hh:63
// was handing it A.nnz(), the CAPACITY. cuSPARSE's CSR contract is
// nnz == rowOffsets[rows], so for every item storing fewer nonzeros than the
// batch maximum the descriptor claimed slots the conversion never wrote, and
// cuSPARSE read them: HeterogeneousNnzAcrossBatch came back wrong in exactly
// the LAST ROW of exactly the SHORT items (the over-read slots belong to the
// item's last row), and PaddingAboveNnzIsNotRead, whose padding carries column
// index 2^30, took the process down with CUDA_ERROR_ILLEGAL_ADDRESS.
//
// THE DISPOSITION IS AN ADAPTER FIX, NOT A VENDOR-LIMITATION SKIP. cuSPARSE
// cannot describe a non-uniform batch in one descriptor, but that is a limit on
// ONE CALL, not on the library: src/backends/cusparse.cc now derives each item's
// nnz from its own row offsets and issues one cusparseSpMM PER ITEM when the
// batch is non-uniform (and one batched call, with the true nnz rather than the
// capacity, when it is uniform). Serialising a non-uniform batch is a real cost
// and it is documented at that site; correct-and-serial beats fast-and-wrong,
// and the native route needs none of it because its loop is bounded by the
// item's own row_offsets[i+1] inside the kernel. So NONE of these three carries
// a skip of its own: all three run live against cuSPARSE and against native,
// and HeterogeneousNnzAcrossBatch runs live against NETLIB too. The only skip
// they can take is the file-wide transposed-refusal skip above, which fires for
// the transposed cases on NETLIB alone and for the documented reason that
// netlib's spmm refuses every transpose.
//
// (PaddingAboveNnzOutOfRangeIsNotReadTrans -- which is this configuration under
// Trans, and which was called PaddingAboveNnzIsNotReadTrans until the deliberate
// break at the bottom of this file proved that name a lie -- crashed the process
// on cuSPARSE as well before the adapter fix; the transposed cuSPARSE path
// over-read the same padding, and the throw that the transposed-refusal skip
// then caught was the SYCL runtime reporting an already-dead CUDA context, not a
// missing route.)
// ---------------------------------------------------------------------------

TYPED_TEST(SpmmCoverageTest, HeterogeneousNnzAcrossBatch) {
    using S = typename TestFixture::ScalarType;
    typename TestFixture::Case c;
    c.m = 40; c.kA = 40; c.nrhs = 4; c.batch = 4; c.nnz_per_row = 3;
    c.heterogeneous_nnz = true; c.empty_rows = true;
    c.alpha = static_cast<S>(-1.75); c.beta = static_cast<S>(0.25);
    this->run_case(c);
}

// THE GATHER KEEPS ITS OUT-OF-RANGE POISON, and unlike the transposed case that
// is not a compromise. The gather has no range guard on the column index -- it
// is a read index into op(B), and a wrong answer there is not heap corruption --
// so nothing discards the entry, and what ARMS this case is the NaN VALUE:
// fma_acc(acc, NaN, whatever the wild read returned) is NaN however that read
// turned out, which is what keeps the assertion from depending on the contents
// of memory the fixture does not own. The out-of-range INDEX is carried on top
// of that as the vendor-fault detector. The in-range counterpart of this axis
// already exists for the gather as HeterogeneousNnzAcrossBatch, so the gather is
// covered by both poisons and only the scatter ever needed the new one.
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

// The transposed twin of HeterogeneousNnzAcrossBatch, with NO poison flag at
// all: the padding carries the default fill, a large value at column 0, which
// is in range in both directions. This is the UNPOISONED form of the transposed
// over-read and before it existed the transposed nnz bound had no coverage of
// any kind -- the poisoned test below it looked like coverage and was not.
TYPED_TEST(SpmmCoverageTest, HeterogeneousNnzAcrossBatchTrans) {
    using S = typename TestFixture::ScalarType;
    typename TestFixture::Case c;
    c.m = 40; c.kA = 40; c.nrhs = 4; c.batch = 4; c.nnz_per_row = 3;
    c.heterogeneous_nnz = true; c.empty_rows = true;
    c.transA = Transpose::Trans;
    c.alpha = static_cast<S>(-1.75); c.beta = static_cast<S>(0.25);
    this->run_case(c);
}

// THE TRANSPOSED nnz BOUND, AND THE ONLY POISON THAT CAN SEE IT.
//
// This test used to carry poison_padding -- NaN at column 2^30 -- and it was
// VACUOUS: breaking the scatter's bound to the per-item capacity left all 352
// cases green. The scatter's range guard (spmm_native.cc:541-542) discards the
// out-of-range column BEFORE the NaN value is ever multiplied, so both halves
// of the poison went in the bin together and a kernel reading uninitialised
// padding on every transposed call shipped undetected. Proven, not argued, by
// two control runs recorded in B4 at the bottom of this file.
//
// The guard is CORRECT and it stays: in the gather a bad column index is an
// out-of-range READ, here it is an out-of-range ATOMIC WRITE, i.e. heap
// corruption. The test is what had to change. The padding now carries an
// IN-RANGE column -- an output row the scatter will happily accumulate into --
// and a large finite sentinel, so an over-read lands in C where the
// backward-error comparison can name it.
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

// THE OUT-OF-RANGE POISON, KEPT, AND HONESTLY LABELLED.
//
// This is the case the one above used to be. It is NOT armed against the native
// transposed bound and it is not pretending to be -- the range guard swallows
// it, which is the whole finding recorded in B4. It is kept because it is armed
// against two OTHER things, both of which have already happened in this tree:
//   * A VENDOR that over-reads the padding. The cuSPARSE adapter did exactly
//     that before this session (it handed cusparseCreateCsr the CAPACITY as the
//     nnz, backend_handle_impl.hh:63) and this configuration took the process
//     down with CUDA_ERROR_ILLEGAL_ADDRESS instead of returning a slightly
//     wrong number. A magnitude error would have been far easier to miss.
//   * A SCATTER THAT LOSES ITS RANGE GUARD. With the guard deleted and the
//     bound left correct the suite is 352/352 green, and with the guard deleted
//     and the bound broken this case SEGFAULTS -- so it is the guard's own
//     regression test, and it is the only case in the file that is.
// Read it as a crash detector, never as coverage of the bound.
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

// ---------------------------------------------------------------------------
// 4. ALL SIX PADS AT ONCE -- four batch strides and two leading dimensions.
//
// Non-natural strides are the NORM at the real call sites, not the exception
// (lanczos.cc:53/:104, syevx_lobpcg.cc:332-341). A body that DERIVES a batch
// stride as ld*cols, or that assumes ld == rows, is correct on every naturally
// packed operand and wrong on every one of those callers. WP7 recorded the same
// hole for gemv over 232 cases before a pad axis existed, which is why these
// are here from the first line rather than added after a bug.
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// 5. THE THREE STRUCTURE AXES RandomSparseHermitian CANNOT PRODUCE.
//
// empty rows   -- re == rs. The output row must be exactly beta*C, so beta != 0
//                 here: at beta == 0 an empty row is indistinguishable from a
//                 row the kernel never wrote at all.
// unsorted     -- convert_to<CSR> emits exactly this (relaxed atomic arrival
//                 order, src/matrix.cc:552-566) and it is what lanczos_tests
//                 and spmm_benchmark already feed spmm today.
// duplicates   -- two entries with the same column in one row MUST SUM. That is
//                 what netlib's loop does and what MatrixView::at /
//                 KernelMatrixView::get do NOT (matrix.hh:157-168 returns the
//                 FIRST match), which is precisely why the reference above
//                 indexes the raw arrays instead.
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// 6. THE TWO SCALAR GUARDS, MADE OBSERVABLE.
//
// beta == 0 MEANS C IS NOT READ. Every in-library caller passes beta = 0 into a
// BumpAllocator-allocated C and BumpAllocator does not zero (mempool.hh:80-92),
// so an implementation that evaluates beta*C unconditionally returns
// 0 * garbage. The in-tree HOST arm does exactly that today
// (netlib_lapack.cc:252 computes `T sum = beta * C_b.at(row, col);` with no
// guard), which is why this test is landed together with the fix to that line:
// without both, the two arms of the library disagree by construction and the
// suite cannot be green on either.
//
// alpha == 0 MEANS A IS NOT READ, and the answer is C = beta*C -- NOT a quick
// return. gemv's quick return also fires at (alpha == 0 && beta == 1) and
// leaves y untouched (gemv_native.cc:473-489); copying that here would be a
// route-dependent WRONG ANSWER, because spmm's answer at alpha == 0 is beta*C.
// A NaN-filled A is what makes the guard observable rather than the arithmetic
// identity 0*x == 0 that no break could ever move.
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// 7. THE LANCZOS SHAPE: A COLUMN OF B THAT IS GENUINELY UNINITIALISED.
//
// lanczos.cc:80-88 writes only the first n entries of each item's 2-column
// basis, and the iteration kernel writes column it+1 only AFTER the spmm at
// iteration it. The buffer comes from BumpAllocator, which does not zero. So at
// EVERY iteration the second column of B is uninitialised memory, its output
// column is discarded, and the contract is that column 0 of C is unharmed by
// it. A register-blocked body that lets one column's accumulator leak into
// another's breaks exactly here and nowhere else in this file.
//
// The strides mirror the call site: ld == rows with a batch stride far above
// ld*cols (lanczos passes stride = (n+1)*n against cols = 2), and C is
// BumpAllocator memory there too, hence c_starts_nan.
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// 8. RECTANGULAR A, BOTH ORIENTATIONS, BOTH DIRECTIONS.
//
// Every fixture above is square, and m and kA are interchangeable in a square
// fixture: a body that used A.rows() where it meant A.cols() is correct on all
// of them. The two orientations fail differently -- with m > kA a wrong OUTPUT
// extent truncates, with m < kA a wrong REDUCTION extent does -- so neither
// alone catches both halves. All four are hand-built, because convert_to<CSR>
// is only correct for square inputs.
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// 9. ALL NINE transA x transB COMBINATIONS, ON ONE RECTANGULAR, PADDED,
//    BATCHED, PER-ITEM-DISTINCT FIXTURE.
//
// Eight of the nine had no coverage anywhere in this tree. They are written out
// one per test rather than looped, so a failure names the pair rather than a
// loop index -- and because the three transA arms are DIFFERENT KERNEL BODIES
// (a gather for NoTrans, a scale followed by an atomic scatter otherwise) while
// transB is a layout choice inside each of them, so the 3x3 is not the product
// of two independent flags and cannot be sampled at its corners.
//
// ConjTrans is not decoration: it is the only spelling under which the SPARSE
// operand is conjugated, and for a real scalar it must be exactly Trans -- a
// body that conjugated a real value would be wrong in a way the complex arms
// cannot see. Running all nine on the real types too is the cheapest way to pin
// that, and it is the same argument WP7 recorded for gemv's ConjTrans arm.
//
// EXPECT SKIPS HERE ON Backend::NETLIB in a vendor-present build: netlib's spmm
// refuses every transpose (netlib_lapack.cc:247-250) and the WP8 preferred()
// clause admits only transA == NoTrans, so that build still routes every
// transposed NETLIB spmm to netlib. See the note at
// the top of this file on why a refusal is a skip and a wrong answer is not.
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// 10. THE nrhs LADDER.
//
// nrhs is the one extent any spmm design register-blocks over, and each block
// width is a separate instantiation whose tail handling is invisible to the
// others. The values are the ones this library actually asks for:
//   1  -- the python API (python/tests/test_batchlas.py:83-95), and what
//         lanczos would ask for if its padding column were ever dropped;
//   2  -- lanczos today (lanczos.cc:53);
//   3  -- ritz_values, via syevx_tests.cc:253;
//   12, 25, 50 -- LOBPCG and syevx_filtered block widths at n = 1024/2048/4096.
// 25 and 50 are deliberately not multiples of any plausible block width.
// ---------------------------------------------------------------------------

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

// The transposed arm carries its OWN column-block loop with its own tail, and
// its width is fixed rather than chosen on nrhs, so the ladder above says
// nothing at all about it. 50 is not a multiple of 4.
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

// ---------------------------------------------------------------------------
// 11. A COMPLEX alpha AND beta.
//
// Every alpha and beta above is real, so their multiplies never mix components
// either: a body that dropped the cross-terms of `alpha * acc` alone would pass
// everything up to here, exactly as WP7 recorded for gemv
// (gemv_tests.cc:648-663).
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// 12. A BATCH LARGE ENOUGH THAT THE LAUNCH SPANS SEVERAL ITEMS PER WORK-GROUP,
//     AT THE lanczos SHAPE.
//
// Everything above runs at batch <= 4 over a few dozen rows, which is a single
// work-group's worth of work on any plausible geometry. This one is m*batch =
// 2048 work units at 3 nnz/row -- the shape the library actually issues most
// often -- so a wrong item stride, or an item boundary crossed inside a group,
// has somewhere to show up.
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// 13. THE DEGENERATE EXTENT.
//
// nrhs == 0 is a legal call with nothing to write, and C must come back
// COMPLETELY untouched -- not scaled by beta, not zeroed. beta is 0.5 and alpha
// is 2.0 here precisely so that "untouched" and "computed" differ: an
// implementation that fell through would write 0.5*C over every element. The
// comparison is on BIT PATTERNS, through run_case's pad check, because the
// claim is "not written" rather than "written with the right value" -- with
// nrhs == 0 every slot of C's stride is a pad slot.
//
// m == 0 is deliberately NOT here. Under NoTrans it makes the launch empty and
// the test vacuous; under Trans it is not a quick return at all (out_rows is
// kA, so the answer is C = beta*C with an empty A), which the ordinary value
// path already covers. The supports() side of both extents belongs to
// route_vocabulary_tests, not to this file.
// ---------------------------------------------------------------------------

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

// ===========================================================================
// THE FOUR DELIBERATE BREAKS.
//
// A test suite that has never been made to FAIL is a hypothesis, not an
// instrument. These four edits to src/sycl/spmm_native.cc are to be applied
// ONE AT A TIME, with BATCHLAS_SPMM_ROUTE=native:direct pinned so the native
// body is what runs (which also disables run_case's refusal-skip), rebuilt,
// run, and REVERTED. Each must turn its named test RED and leave its named
// control GREEN. A break that leaves the suite green means the axis it targets
// is vacuous and must be fixed before the kernel is trusted -- WP7 recorded
// three such breaks coming back green over 376 cases, and this file exists
// partly because of them.
//
// THIS BLOCK IS NOT A PLAN, IT IS A LOG: B4 came back GREEN the first time it
// was run and the axis it names really was vacuous, which is why B2's wording
// below is a correction and why PaddingAboveNnzIsNotReadTrans no longer carries
// the poison it was written with. src/sycl/spmm_native.cc is UNTRACKED, so
// `git diff` reports nothing after a break and cannot be used to confirm the
// revert -- take an md5sum of the pristine file BEFORE the first break and
// compare against it after the last one.
//
// A coverage row cannot substitute for any of this: rows are keyed on a
// power-of-two shape_class and are first-writer-wins
// (VENDOR_FREE_BASELINE.md:255-260), so a row proves that SOME shape resolved
// to a route, never that THIS shape ran THAT body.
//
//   B1 `gatherBase`
//       In body 1 (the NoTrans gather), read the column index as `a_ci[p]`
//       instead of `a_ci[vb + p]` -- i.e. drop the per-item base from
//       col_indices ONLY, leaving the values correctly indexed. Every batch
//       item then reads item 0's sparsity pattern.
//       MUST TURN RED : DistinctPatternsAcrossBatch
//                       (and, as a consequence, every other case built with
//                        distinct_patterns or heterogeneous_nnz:
//                        HeterogeneousNnzAcrossBatch, PaddingAboveNnzIsNotRead,
//                        NrhsOne..NrhsFifty, LargeBatchLanczosShape,
//                        ComplexAlphaBetaConjTrans and the Nine* family)
//       MUST STAY GREEN: SingleItemSquareNoTrans  -- batch 1, an identity
//                        BatchedSquareUniformPatternNoTrans -- batch 4, but
//                        every item's column array is byte-identical, so the
//                        wrong base reads the right numbers. This is the
//                        control that excludes "the break is about the batch"
//                        and pins it to the BASE.
//
//   B2 `gatherBound`  -- THE SHARP FORM. The obvious form is wrong.
//       In body 1 (the NoTrans gather), extend the LAST row's bound only:
//
//           const int re = (i == out_rows - 1) ? a_nnz_cap : a_ro[ro + i + 1];
//
//       capturing `const int a_nnz_cap = A.nnz();` beside the strides. A.nnz()
//       is the per-item CAPACITY, i.e. the batch maximum, and the slots between
//       an item's own nnz and that capacity are the uninitialised padding.
//       MUST TURN RED : PaddingAboveNnzIsNotRead
//                       HeterogeneousNnzAcrossBatch
//                       and EXACTLY those two: they are the only NoTrans cases
//                       whose batch is non-uniform, and the over-read slots
//                       belong to the short items' last row.
//       MUST STAY GREEN: SingleItemSquareNoTrans and every other case with a
//                        homogeneous batch, where ro[m] == A.nnz() for every
//                        item and the two bounds coincide.
//
//       DO NOT WRITE IT AS `const int re = a_nnz_cap;` FOR EVERY ROW. That form
//       was the original wording here and it is far too coarse to prove
//       anything: row 0 then sweeps the item's entire slab, every row after it
//       double-counts, and the break falsifies its OWN named controls -- a
//       homogeneous batch goes red too, so "stays green" identifies nothing and
//       the red set no longer isolates the padding. A break has to be as narrow
//       as the contract it denies.
//
//       B4 is this same defect in the scatter, which carries a SEPARATELY BLIND
//       copy of the bound. It is a different break with a different control set
//       and a different poison; do not treat one as evidence for the other.
//
//   B3 `gatherStride`
//       In body 1, derive B's batch stride as `ldb * nrhs` instead of reading
//       `B_mat.stride()`.
//       MUST TURN RED : StridePadsAreReadNotDerived
//                       UninitialisedBColumnDoesNotContaminateTheOthers
//                       LargeBatchLanczosShape
//                       (all three carry b_stride_pad != 0 at batch >= 2)
//       MUST STAY GREEN: every case with b_stride_pad == 0, where the derived
//                        stride equals the real one -- which is all of sections
//                        1, 2, 3, 5, 6, 8 and 10, and is exactly why this axis
//                        had to be present from the first line rather than
//                        added after a bug.
//       NOTE: at batch 1 the stride is never applied, so this break is an
//       identity for SingleItemSquareNoTrans whatever the pad.
//
//   B4 `scatterBound`  -- THE ONE THAT CAME BACK GREEN.
//       In body 2 (the Trans/ConjTrans scatter), extend the LAST row's bound:
//
//           const int re = (i == rows_in - 1) ? a_nnz_cap : a_ro[ro + i + 1];
//
//       capturing `const int a_nnz_cap = A.nnz();` beside the strides. Note
//       `rows_in`, not `out_rows`: under Trans the loop runs over the rows of
//       the STORED A, which is the reduction extent.
//       MUST TURN RED : HeterogeneousNnzAcrossBatchTrans
//                       PaddingAboveNnzIsNotReadTrans
//                       and EXACTLY those two: they are the only transposed
//                       cases with a non-uniform batch.
//       MUST STAY GREEN: every transposed case with a homogeneous batch --
//                        DistinctPatternsAcrossBatchTrans (different COLUMNS,
//                        identical counts, so it separates this axis from B5's),
//                        EmptyRowsContributeOnlyBetaTrans,
//                        StridePadsAreReadNotDerivedTrans (matrix_stride_pad = 5
//                        and still green, which pins the break to A.nnz() rather
//                        than to the capacity), UnsortedColumnsWithinARowTrans,
//                        DuplicateColumnsSumTrans, RectangularWideTrans,
//                        RectangularTallTrans, LargeBatchLanczosShapeTrans,
//                        NrhsFiftyTrans, ComplexAlphaBetaConjTrans, the Nine*
//                        family, and every NoTrans case (a different body).
//                        ALSO GREEN, AND THAT IS THE POINT:
//                        PaddingAboveNnzOutOfRangeIsNotReadTrans.
//
//       THE HISTORY, BECAUSE IT IS THE REASON THIS ENTRY EXISTS. Run once with
//       the file as originally written, this break left ALL 352 CASES GREEN.
//       The transposed nnz bound was completely unguarded and a kernel reading
//       uninitialised padding on every transposed call would have shipped.
//       PaddingAboveNnzIsNotReadTrans looked like the guard and was not: it
//       poisoned the padding with an OUT-OF-RANGE column (2^30) and a NaN, and
//       the scatter's own range guard (spmm_native.cc:541-542) discards the
//       entry BEFORE `av` is multiplied, so both halves of the poison were
//       swallowed together. The test was green because of a kernel guard, not
//       because of the property it named.
//
//       TWO CONTROL RUNS ESTABLISHED THAT, rather than argument:
//         * broken bound + range guard deleted  -> the case SEGFAULTS
//           (exit 139, on the double instantiation), so the over-read is real
//           and lethal;
//         * correct bound + range guard deleted -> 352/352 green, so deleting
//           the guard is harmless on its own and the crash above is
//           attributable to the bound alone.
//       (The float instantiation survived even the first of those, which is a
//       second reason not to rely on an out-of-range index as a detector: where
//       it lands is not the fixture's to decide.)
//
//       THE FIX WAS TO THE TEST, NOT THE KERNEL. The guard is correct -- in the
//       gather a bad column index is an out-of-range READ, here it is an
//       out-of-range ATOMIC WRITE, i.e. heap corruption -- so it stays, and
//       PaddingAboveNnzIsNotReadTrans now poisons with an IN-RANGE column and a
//       large finite sentinel, an entry the scatter accepts and accumulates, so
//       the over-read lands in C where the backward-error comparison names it.
//       The out-of-range configuration is kept under its own honest name,
//       PaddingAboveNnzOutOfRangeIsNotReadTrans, as a vendor-fault and
//       missing-guard detector that is NOT coverage of this bound.
//       HeterogeneousNnzAcrossBatchTrans was added at the same time: the
//       UNPOISONED form of this over-read had no transposed twin at all.
//
//   THE GENERAL LESSON, WHICH IS WORTH MORE THAN B4 ITSELF. Every contract in
//   this file has TWO independent implementations -- the gather and the scatter
//   -- and a poison tuned to one body's failure mode can be inert against the
//   other's. Before trusting any *Trans twin, ask not "does it exist" but "what
//   does the kernel do with the poison": a defensive predicate between the
//   poison and the assertion makes the case vacuous no matter how carefully it
//   was written.
// ===========================================================================
