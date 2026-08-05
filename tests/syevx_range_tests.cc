// Pure host-side tests for `syevx_resolve_range`.
//
// The resolver is the single place where the three user-facing selectors
// (Extremal, Index, Value) are turned into one internal vocabulary, and EVERY
// solver and EVERY *_buffer_size derives its behaviour from its output rather
// than from SyevxParams directly. That is what makes the solve and the sizing
// call structurally incapable of disagreeing about what was asked for -- so the
// resolver is worth pinning exhaustively and in isolation.
//
// This lives in its own binary rather than in syevx_tests.cc for one reason:
// syevx_tests is labelled `slow` (it runs a few seconds of device work), while
// these need no device at all and run in microseconds. Keeping them separate
// means they run in the default sweep, where a normalization regression should
// be caught, instead of only when someone opts into the slow label.

#include <gtest/gtest.h>
#include <blas/enums.hh>
#include <blas/matrix.hh>
#include <blas/extensions.hh>

#include <string>
#include <vector>

using namespace batchlas;

namespace {

struct ResolveCase {
    const char* name;
    // inputs
    int64_t     n;
    size_t      neigs;
    SyevxSelect select;
    bool        find_largest;
    int64_t     il;
    int64_t     iu;
    SortOrder   order;
    // expectations
    bool        exp_value_range;
    int64_t     exp_il;         // ignored when exp_value_range
    int64_t     exp_iu;         // ignored when exp_value_range
    int64_t     exp_max_count;
    bool        exp_reverse;
};

const ResolveCase kCases[] = {
    // ---- Extremal: the historical contract, which must not move ----------
    // `reverse` comes from find_largest and NEVER from `order`. That is the
    // entire reason Extremal exists as its own selector instead of being spelled
    // as an index block, and it is why the plan's proposed "order contradicts
    // find_largest -> throw" rule was dropped: SortOrder has no unset sentinel,
    // so with `order` defaulting to Ascending and `find_largest` to true, that
    // rule would have rejected the library's own defaults on essentially every
    // existing call in the repo.
    {"ExtremalLargest",           100,   8, SyevxSelect::Extremal, true,  0, -1,
     SortOrder::Ascending,  false, 92, 99, 8, true},
    {"ExtremalSmallest",          100,   8, SyevxSelect::Extremal, false, 0, -1,
     SortOrder::Ascending,  false,  0,  7, 8, false},
    {"ExtremalLargestIgnoresAscendingOrder", 100, 8, SyevxSelect::Extremal, true, 0, -1,
     SortOrder::Descending, false, 92, 99, 8, true},
    {"ExtremalSmallestIgnoresDescendingOrder", 100, 8, SyevxSelect::Extremal, false, 0, -1,
     SortOrder::Descending, false,  0,  7, 8, false},
    // A capacity above n is CLAMPED, not rejected: `neigs` is a capacity now, so
    // asking for more than exists just leaves the tail of W and V unwritten.
    {"ExtremalCapacityAboveN",    100, 200, SyevxSelect::Extremal, true,  0, -1,
     SortOrder::Ascending,  false,  0, 99, 100, true},
    {"ExtremalCapacityAboveNSmallest", 100, 200, SyevxSelect::Extremal, false, 0, -1,
     SortOrder::Ascending,  false,  0, 99, 100, false},
    {"ExtremalWholeSpectrum",     100, 100, SyevxSelect::Extremal, true,  0, -1,
     SortOrder::Ascending,  false,  0, 99, 100, true},
    {"ExtremalSingleton",           1,   1, SyevxSelect::Extremal, true,  0, -1,
     SortOrder::Ascending,  false,  0,  0, 1, true},
    // The empty request, encoded as an empty block (iu < il). Pinned because
    // every consumer has to survive it without an unsigned loop bound.
    {"ExtremalZeroCapacity",      100,   0, SyevxSelect::Extremal, false, 0, -1,
     SortOrder::Ascending,  false,  0, -1, 0, false},

    // ---- Index -----------------------------------------------------------
    {"IndexInterior",             100,   8, SyevxSelect::Index,    true, 20, 27,
     SortOrder::Ascending,  false, 20, 27, 8, false},
    {"IndexInteriorDescending",   100,   8, SyevxSelect::Index,    true, 20, 27,
     SortOrder::Descending, false, 20, 27, 8, true},
    // find_largest must NOT leak into an explicit index block -- `order` alone
    // decides, which is what makes Index behave like LAPACK's range='I'.
    {"IndexIgnoresFindLargest",   100,   8, SyevxSelect::Index,    true, 20, 27,
     SortOrder::Ascending,  false, 20, 27, 8, false},
    // iu < 0 means n-1.
    {"IndexOpenUpperBound",       100, 100, SyevxSelect::Index,    true,  0, -1,
     SortOrder::Descending, false,  0, 99, 100, true},
    {"IndexSingleEigenpair",      100,   1, SyevxSelect::Index,    false, 17, 17,
     SortOrder::Ascending,  false, 17, 17, 1, false},
    {"IndexBottomEnd",            100,   6, SyevxSelect::Index,    false,  0,  5,
     SortOrder::Ascending,  false,  0,  5, 6, false},
    {"IndexTopEnd",               100,   6, SyevxSelect::Index,    false, 94, 99,
     SortOrder::Ascending,  false, 94, 99, 6, false},
    // An index block's max_count comes from the BLOCK, never from `neigs`: the
    // mismatch is the validator's job to reject loudly, and silently clamping it
    // here would turn that loud error into an under-filled output buffer.
    {"IndexCountComesFromTheBlockNotNeigs", 100, 3, SyevxSelect::Index, true, 20, 27,
     SortOrder::Ascending,  false, 20, 27, 8, false},
    // ---- Index, out of range --------------------------------------------
    // The public `syevx` rejects these before resolving, but `syevx_direct` and
    // `syevx_direct_subset` are public entry points of their own and the resolver
    // is documented to CLAMP rather than throw so that it stays usable from a
    // sizing path. Without the clamp, max_count would be the raw iu-il+1 and
    // syevx_direct's selection kernel would index past the n-entry eigenvalue
    // array -- an out-of-bounds device read, not a wrong answer. Every row here
    // exists because that clamp is load-bearing, not decorative.
    {"IndexUpperBoundPastN",       64,   8, SyevxSelect::Index,    true, 60, 67,
     SortOrder::Ascending,  false, 60, 63, 4, false},
    {"IndexLowerBoundBelowZero",   64,   8, SyevxSelect::Index,    true, -3,  4,
     SortOrder::Ascending,  false,  0,  4, 5, false},
    // Entirely past the end: the canonical EMPTY block (il=0, iu=-1), so that
    // `iu - il + 1 == max_count` holds for every resolved range and no consumer
    // sees a negative count.
    {"IndexEntirelyPastN",         64,   8, SyevxSelect::Index,    true, 90, 99,
     SortOrder::Ascending,  false,  0, -1, 0, false},
    {"IndexInvertedBlock",         64,   8, SyevxSelect::Index,    true, 30, 20,
     SortOrder::Ascending,  false,  0, -1, 0, false},

    // ---- Value -----------------------------------------------------------
    // il/iu carry no meaning here (there is no static block), so they are not
    // asserted; max_count is the CAPACITY and the true per-item count may exceed
    // it, which is the whole reason `m` exists.
    {"ValueAscending",            100,  12, SyevxSelect::Value,    true,  0, -1,
     SortOrder::Ascending,  true,   0,  0, 12, false},
    {"ValueDescending",           100,  12, SyevxSelect::Value,    true,  0, -1,
     SortOrder::Descending, true,   0,  0, 12, true},
    {"ValueCapacityAboveN",       100, 500, SyevxSelect::Value,    false, 0, -1,
     SortOrder::Ascending,  true,   0,  0, 100, false},
    {"ValueZeroCapacity",         100,   0, SyevxSelect::Value,    false, 0, -1,
     SortOrder::Ascending,  true,   0,  0, 0, false},
    {"ValueIgnoresFindLargest",   100,  12, SyevxSelect::Value,    true,  0, -1,
     SortOrder::Ascending,  true,   0,  0, 12, false},
};

class SyevxResolveRangeTest : public ::testing::TestWithParam<ResolveCase> {};

} // namespace

TEST_P(SyevxResolveRangeTest, MatchesTheNormalizationTable) {
    const auto& c = GetParam();
    const auto rr = syevx_resolve_range(c.n, c.neigs, c.select, c.find_largest,
                                        c.il, c.iu, c.order);

    EXPECT_EQ(rr.value_range, c.exp_value_range);
    EXPECT_EQ(rr.max_count, c.exp_max_count);
    EXPECT_EQ(rr.reverse, c.exp_reverse);
    if (!c.exp_value_range) {
        EXPECT_EQ(rr.il, c.exp_il);
        EXPECT_EQ(rr.iu, c.exp_iu);
        // The two must stay consistent, or a consumer deriving the count from the
        // block and one deriving it from max_count would silently differ.
        EXPECT_EQ(rr.iu - rr.il + 1, rr.max_count);
        // The block itself must land inside the spectrum whenever it is non-empty.
        // syevx_direct's selection kernel reads lam[il .. iu] out of an n-entry
        // per-item array with no bound of its own, so this is a memory-safety
        // assertion and not a tidiness one. (Asserted for EVERY row rather than
        // only the deliberately out-of-range ones: a clamp that silently stopped
        // applying would otherwise show up nowhere.)
        if (rr.max_count > 0) {
            EXPECT_GE(rr.il, 0);
            EXPECT_LT(rr.iu, c.n);
        }
    }
    // max_count is an upper bound on m[b] and is already clamped to n, so no
    // consumer needs to clamp it again -- several of them assume exactly this.
    EXPECT_GE(rr.max_count, 0);
    EXPECT_LE(rr.max_count, c.n);
}

INSTANTIATE_TEST_SUITE_P(NormalizationTable, SyevxResolveRangeTest,
                         ::testing::ValuesIn(kCases),
                         [](const ::testing::TestParamInfo<ResolveCase>& info) {
                             return std::string(info.param.name);
                         });

// The 3-argument convenience adaptor is distinguished from the 7-argument form by
// arity alone. It must forward every field, in the right order -- a transposed
// pair there would be invisible at the call sites, which all spell it as one
// short call.
TEST(SyevxResolveRangeTest, ParamsAdaptorForwardsEveryField) {
    for (const auto& c : kCases) {
        SCOPED_TRACE(c.name);
        SyevxParams<float> p;
        p.select = c.select;
        p.find_largest = c.find_largest;
        p.il = c.il;
        p.iu = c.iu;
        p.order = c.order;

        const auto direct = syevx_resolve_range(c.n, c.neigs, c.select, c.find_largest,
                                                c.il, c.iu, c.order);
        const auto viaParams = syevx_resolve_range(c.n, c.neigs, p);

        EXPECT_EQ(viaParams.value_range, direct.value_range);
        EXPECT_EQ(viaParams.il, direct.il);
        EXPECT_EQ(viaParams.iu, direct.iu);
        EXPECT_EQ(viaParams.max_count, direct.max_count);
        EXPECT_EQ(viaParams.reverse, direct.reverse);
    }
}

// The defaults of SyevxParams must resolve to precisely the historical
// behaviour: the top `neigs` eigenpairs, returned descending. This is the single
// assertion that stands between "range selection was added" and "every existing
// caller's output changed".
TEST(SyevxResolveRangeTest, DefaultParamsReproduceTheHistoricalContract) {
    const SyevxParams<float> defaults;
    ASSERT_EQ(defaults.select, SyevxSelect::Extremal);
    ASSERT_TRUE(defaults.find_largest);

    const auto rr = syevx_resolve_range(/*n=*/64, /*neigs=*/8, defaults);
    EXPECT_FALSE(rr.value_range);
    EXPECT_EQ(rr.il, 56);
    EXPECT_EQ(rr.iu, 63);
    EXPECT_EQ(rr.max_count, 8);
    EXPECT_TRUE(rr.reverse) << "find_largest must still imply a descending answer";

    SyevxParams<float> smallest;
    smallest.find_largest = false;
    const auto asc = syevx_resolve_range(/*n=*/64, /*neigs=*/8, smallest);
    EXPECT_EQ(asc.il, 0);
    EXPECT_EQ(asc.iu, 7);
    EXPECT_FALSE(asc.reverse);
}

// Extremal and Index are the same thing internally, and the end blocks are where
// they meet. This is the host-side half of the EndBlocksReproduceExtremalBitForBit
// tests in syevx_tests.cc: those assert the two produce identical NUMBERS, this
// asserts they resolve to an identical REQUEST, which is why they must.
TEST(SyevxResolveRangeTest, EndIndexBlocksResolveToTheExtremalRequest) {
    constexpr int64_t n = 64;
    for (const size_t k : {size_t(1), size_t(6), size_t(64)}) {
        SCOPED_TRACE(::testing::Message() << "k=" << k);

        const auto top = syevx_resolve_range(n, k, SyevxSelect::Extremal, true, 0, -1,
                                             SortOrder::Ascending);
        const auto top_idx = syevx_resolve_range(n, k, SyevxSelect::Index, false,
                                                 n - int64_t(k), n - 1,
                                                 SortOrder::Descending);
        EXPECT_EQ(top.il, top_idx.il);
        EXPECT_EQ(top.iu, top_idx.iu);
        EXPECT_EQ(top.max_count, top_idx.max_count);
        EXPECT_EQ(top.reverse, top_idx.reverse);

        const auto bottom = syevx_resolve_range(n, k, SyevxSelect::Extremal, false, 0, -1,
                                                SortOrder::Ascending);
        const auto bottom_idx = syevx_resolve_range(n, k, SyevxSelect::Index, true, 0,
                                                    int64_t(k) - 1, SortOrder::Ascending);
        EXPECT_EQ(bottom.il, bottom_idx.il);
        EXPECT_EQ(bottom.iu, bottom_idx.iu);
        EXPECT_EQ(bottom.max_count, bottom_idx.max_count);
        EXPECT_EQ(bottom.reverse, bottom_idx.reverse);
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
