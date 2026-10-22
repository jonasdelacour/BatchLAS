// SPDX-License-Identifier: MIT
// src/select/select.hh on a synthetic choice variant and synthetic tables (no GPU, no kernel),
// then the batch policy every public entry point shares, on a real queue.
// docs/design/flat-kernel-selection.md §5 and §8.5.

#include <gtest/gtest.h>

#include <batchlas/blas/functions.hh>
#include <batchlas/error.hh>
#include <batchlas/settings.hh>
#include <batchlas/util/env.hh>

#include <src/select/coverage.hh>
#include <src/select/select.hh>
#include <tests/test_utils.hh>

#include <algorithm>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unistd.h>
#include <utility>
#include <variant>
#include <vector>

namespace sel = batchlas::select;
using batchlas::ScopedEnvVar;

namespace {

struct Tiny : sel::NoFields<"tiny"> {};
struct Cta : sel::NoFields<"cta"> {};
struct Blocked : sel::NoFields<"blocked"> {};
struct Vendor : sel::NoFields<"vendor"> {};
struct Lpanel {
    int panel = 8;
    static constexpr std::string_view name = "lpanel";
    static constexpr std::array<std::string_view, 1> fields{"panel"};
    std::array<int, 1> values() const { return {panel}; }
    static Lpanel from(std::array<int, 1> v) { return {v[0]}; }
    bool operator==(const Lpanel&) const = default;
};
struct Wide {  // two fields, never a candidate
    int a = 0, b = 0;
    static constexpr std::string_view name = "wide";
    static constexpr std::array<std::string_view, 2> fields{"a", "b"};
    std::array<int, 2> values() const { return {a, b}; }
    static Wide from(std::array<int, 2> v) { return {v[0], v[1]}; }
    bool operator==(const Wide&) const = default;
};

using C = std::variant<Tiny, Cta, Lpanel, Wide, Blocked, Vendor>;
const std::array<C, 6> kCands{Tiny{}, Cta{}, Lpanel{8}, Lpanel{16}, Blocked{}, Vendor{}};
const std::array<C, 4> kNoVendor{Tiny{}, Cta{}, Lpanel{8}, Blocked{}};
constexpr std::array<std::string_view, 2> kLastResort{"blocked", "vendor"};
const sel::Rules kRules{kLastResort};

using Pred = std::function<bool(const C&)>;
const Pred kAll = [](const C&) { return true; };
Pred only(std::set<std::string> allowed) {
    return [allowed](const C& c) { return allowed.count(sel::to_string(c)) > 0; };
}
Pred all_but(std::set<std::string> banned) {
    return [banned](const C& c) { return banned.count(sel::to_string(c)) == 0; };
}

sel::Key key(const char* uplo, std::int64_t n, std::int64_t batch) {
    return {{"uplo", uplo}, {"n", n}, {"batch", batch}};
}

std::string table(const std::string& device, const std::string& rows, const std::string& op = "synth") {
    return "# op=" + op + " dtype=float device=" + device + " kernels=unknown\n" +
           "# keys: uplo:exact n:log batch:log\n" + rows;
}
// Every row line gets "dtype=float " in front: a tuned/<op>.<device>.txt file keys rows by dtype first.
std::string float_rows(const std::string& rows) {
    std::string out;
    std::size_t a = 0;
    while (a < rows.size()) {
        std::size_t e = rows.find('\n', a);
        if (e == std::string::npos) e = rows.size();
        const std::string line = rows.substr(a, e - a);
        out += (line.empty() || line[0] == '#' ? line : "dtype=float " + line) + (e < rows.size() ? "\n" : "");
        a = e + 1;
    }
    return out;
}
std::pair<std::string, std::string> file(const std::string& device, const std::string& rows,
                                         const std::string& op = "synth") {
    return {op + "." + device + ".txt", "# op=" + op + " device=" + device + " kernels=unknown\n" +
                                            "# keys: dtype:exact uplo:exact n:log batch:log\n" + float_rows(rows)};
}

C choose(const std::string& device, const sel::Key& k, const Pred& ok = kAll, const std::string& op = "synth") {
    return sel::choose(op, "float", sel::device_from_key(device), k, kCands, ok, kRules);
}

std::string S(const C& c) { return sel::to_string(c); }

batchlas::coverage::Shape shape(std::int64_t n, std::int64_t batch) {
    return {.scalar = batchlas::ScalarKind::F32, .backend = batchlas::Backend::CUDA, .m = n, .n = n, .k = n,
            .batch = batch};
}

class Select : public ::testing::Test {
protected:
    void SetUp() override { sel::testing::reset_warnings(); }
    void TearDown() override { sel::testing::use_embedded_tables(); }
};

// ---- spelling -------------------------------------------------------------------------

TEST(SelectSpelling, LongFormRoundTrips) {
    EXPECT_EQ(S(Tiny{}), "tiny");
    EXPECT_EQ(S(Lpanel{8}), "lpanel:panel=8");
    EXPECT_EQ(S(Wide{3, -4}), "wide:a=3:b=-4");
    std::vector<C> all(kCands.begin(), kCands.end());
    all.push_back(Wide{3, -4});
    for (const C& c : all) {
        const auto back = sel::parse<C>(S(c));
        ASSERT_TRUE(back) << S(c);
        EXPECT_EQ(*back, c) << S(c);
    }
    EXPECT_NE(*sel::parse<C>("lpanel:panel=16"), C{Lpanel{8}});
}

TEST(SelectSpelling, PositionalAndNamedFields) {
    EXPECT_EQ(*sel::parse<C>("lpanel:16"), C{Lpanel{16}});
    EXPECT_EQ(*sel::parse<C>("wide:3:4"), (C{Wide{3, 4}}));
    EXPECT_EQ(*sel::parse<C>("wide:b=4:a=3"), (C{Wide{3, 4}}));
    EXPECT_EQ(*sel::parse<C>("wide:3:b=4"), (C{Wide{3, 4}}));
}

TEST(SelectSpelling, EveryRejection) {
    const std::vector<std::pair<std::string, std::string>> bad{
        {"", "unknown family"},          {"huge", "unknown family"},
        {"LPANEL:8", "unknown family"},  {" tiny", "unknown family"},
        {"lpanel:size=8", "unknown field 'size'"},
        {"lpanel", "takes 1 field(s), got 0"}, {"lpanel:8:9", "takes 1 field(s), got 2"},
        {"tiny:1", "takes 0 field(s), got 1"}, {"lpanel:8:", "takes 1 field(s), got 2"},
        {"lpanel:eight", "not an integer"},    {"lpanel:8x", "not an integer"},
        {"lpanel:8 ", "not an integer"},       {"lpanel:panel=", "not an integer"},
        {"lpanel:", "not an integer"},         {"wide:a=1:a=2", "given twice"},
        {"wide:b=4:3", "given twice"},
    };
    for (const auto& [text, why] : bad) {
        std::string err;
        EXPECT_FALSE(sel::parse<C>(text, &err)) << "'" << text << "'";
        EXPECT_NE(err.find(why), std::string::npos) << "'" << text << "' -> " << err;
    }
}

// ---- table text -----------------------------------------------------------------------

TEST(SelectTable, ParsesRowsCommentsAndHeader) {
    const auto t = sel::parse_table(table("sm_120",
        "uplo=L n=64 batch=1024 | lpanel:panel=8 0.302 | vendor 0.49   # noisy\n"
        "\n"
        "# a comment after the rows\n"
        "batch=2048 n=64 uplo=L | cta 1\n"), "synth.sm_120.txt");
    EXPECT_EQ(t.op, "synth");
    EXPECT_EQ(t.device, "sm_120");
    EXPECT_EQ(t.family, "sm");
    EXPECT_EQ(t.arch_number, 120);
    ASSERT_EQ(t.keys.size(), 3u);
    EXPECT_FALSE(t.keys[0].log);
    EXPECT_TRUE(t.keys[2].log);
    ASSERT_EQ(t.rows.size(), 2u);
    EXPECT_EQ(t.rows[0].line, 3);
    ASSERT_EQ(t.rows[0].ranked.size(), 2u);
    EXPECT_EQ(t.rows[0].ranked[1].spelling, "vendor");
    EXPECT_DOUBLE_EQ(t.rows[0].ranked[1].ms, 0.49);
    EXPECT_EQ(t.rows[1].keys[2], "2048");
}

TEST(SelectTable, LogKeyWeightsDefaultToOne) {
    const auto t = sel::parse_table("# dtype=float\n# keys: uplo:exact n:log:3 batch:log m:log:0.5\nuplo=L n=4 batch=1 m=2 | tiny 1\n",
                                    "synth.sm_120.txt");
    ASSERT_EQ(t.keys.size(), 4u);
    EXPECT_DOUBLE_EQ(t.keys[0].weight, 1.0);
    EXPECT_DOUBLE_EQ(t.keys[1].weight, 3.0);
    EXPECT_DOUBLE_EQ(t.keys[2].weight, 1.0);
    EXPECT_DOUBLE_EQ(t.keys[3].weight, 0.5);
    EXPECT_TRUE(t.keys[3].log);
}

TEST(SelectTable, TranscribedRowsAreRankedWithoutTimes) {
    const auto t = sel::parse_table(
        "# op=synth dtype=float device=sm_89 source=transcribed:2b46acab\n"
        "# keys: uplo:exact n:log batch:log\n"
        "uplo=L n=20 batch=8192 | tiny - | cta - | blocked -\n"
        "uplo=L n=64 batch=8192 | blocked - | cta -\n", "synth.sm_89.txt");
    EXPECT_EQ(t.source, "transcribed:2b46acab");
    ASSERT_EQ(t.rows.size(), 2u);
    EXPECT_FALSE(t.rows[0].timed);
    EXPECT_FALSE(t.rows[1].timed);
    ASSERT_EQ(t.rows[0].ranked.size(), 3u);
    EXPECT_EQ(t.rows[0].ranked[2].spelling, "blocked");
    EXPECT_DOUBLE_EQ(t.rows[0].ranked[2].ms, 0.0);
    EXPECT_EQ(t.rows[1].ranked[0].spelling, "blocked");
    const auto timed = sel::parse_table("# op=synth dtype=float device=sm_89 source=sweep.jsonl\n"
                                        "# keys: uplo:exact n:log batch:log\n"
                                        "uplo=L n=64 batch=8192 | blocked 1.5 | cta 2\n", "synth.sm_89.txt");
    EXPECT_TRUE(timed.rows[0].timed);
    EXPECT_DOUBLE_EQ(timed.rows[0].ranked[0].ms, 1.5);
    // A transcribed table may mix row kinds (each row is one kind); the Python --check agrees.
    const auto mixed = sel::parse_table("# op=synth dtype=float device=sm_89 source=transcribed:2b46acab\n"
                                        "# keys: uplo:exact n:log batch:log\n"
                                        "uplo=L n=20 batch=8192 | tiny - | cta -\n"
                                        "uplo=L n=64 batch=8192 | blocked 1.5 | cta 2\n", "synth.sm_89.txt");
    ASSERT_EQ(mixed.rows.size(), 2u);
    EXPECT_FALSE(mixed.rows[0].timed);
    EXPECT_TRUE(mixed.rows[1].timed);
    EXPECT_DOUBLE_EQ(mixed.rows[1].ranked[1].ms, 2.0);
}

TEST(SelectTable, LedgerTableMixesTimedAndTranscribedRows) {
    const auto t = sel::parse_table("# op=synth dtype=float device=sm_89 batchlas=x kernels=y family_kernels=a:b date=d\n"
                                    "# source=ledger:benchmarks/results/tuning/ledger/synth.float.sm_89 tiers=deep:1,coarse:0,preview:0,custom:0,transcribed:1\n"
                                    "# keys: uplo:exact n:log batch:log\n"
                                    "uplo=L n=20 batch=8192 | tiny - | cta - # transcribed\n"
                                    "uplo=L n=64 batch=8192 | blocked 1.5 | cta 2 # deep\n", "synth.sm_89.txt");
    EXPECT_EQ(t.source.rfind("ledger:", 0), 0u);
    ASSERT_EQ(t.rows.size(), 2u);
    EXPECT_FALSE(t.rows[0].timed);
    EXPECT_TRUE(t.rows[1].timed);
    const auto* lo = t.nearest({{"uplo", "L"}, {"n", "16"}, {"batch", "8192"}});
    ASSERT_NE(lo, nullptr);
    EXPECT_EQ(lo->ranked[0].spelling, "tiny");
    const auto* hi = t.nearest({{"uplo", "L"}, {"n", "70"}, {"batch", "8192"}});
    ASSERT_NE(hi, nullptr);
    EXPECT_EQ(hi->ranked[0].spelling, "blocked");
    EXPECT_THROW(sel::parse_table("# op=synth dtype=float device=sm_89 source=sweep.jsonl\n# keys: n:log\nn=4 | tiny -\n",
                                  "synth.sm_89.txt"), batchlas::internal_error);
}

TEST(SelectTable, DeviceFromFileNameWhenHeaderOmitsIt) {
    const auto parts = sel::parse_tables("# keys: dtype:exact uplo:exact n:log\ndtype=cfloat uplo=L n=4 | tiny 1\n",
                                         "dir/op.gfx90a.txt");
    ASSERT_EQ(parts.size(), 1u);
    const sel::Table& t = parts[0];
    EXPECT_EQ(t.file, "op.gfx90a.txt");
    EXPECT_EQ(t.op, "op");
    EXPECT_EQ(t.dtype, "cfloat");
    EXPECT_EQ(t.family, "gfx");
    EXPECT_EQ(t.arch_number, 90);
    EXPECT_THROW(sel::parse_table("# keys: uplo:exact n:log\nuplo=L n=4 | tiny 1\n", "dir/op.cfloat.gfx90a.txt"),
                 batchlas::internal_error) << "an old <op>.<dtype>.<device>.txt name supplies nothing";
}

// tuned/<op>.<device>.txt: dtype is the first key; the file splits into one Table per dtype, each
// without the dtype key, rows in file order, lookups confined to the dtype.
TEST(SelectTable, AnOpDeviceFileSplitsIntoOneTablePerDtype) {
    const std::string text =
        "# op=synth device=sm_120 source=ledger:x.jsonl\n"
        "# keys: dtype:exact uplo:exact n:log batch:log\n"
        "dtype=float uplo=L n=64 batch=1024 | tiny 1 # deep\n"
        "dtype=double uplo=L n=64 batch=1024 | cta 2 # deep\n"
        "dtype=float uplo=L n=512 batch=1024 | vendor 3 # deep\n"
        "dtype=double uplo=L n=8 batch=1024 | blocked - # transcribed\n";
    const auto parts = sel::parse_tables(text, "synth.sm_120.txt");
    ASSERT_EQ(parts.size(), 2u);
    EXPECT_EQ(parts[0].dtype, "float");
    EXPECT_EQ(parts[1].dtype, "double");
    for (const sel::Table& t : parts) {
        EXPECT_EQ(t.op, "synth");
        EXPECT_EQ(t.device, "sm_120");
        EXPECT_EQ(t.source, "ledger:x.jsonl");
        ASSERT_EQ(t.keys.size(), 3u);
        EXPECT_EQ(t.keys[0].name, "uplo");
        ASSERT_EQ(t.rows.size(), 2u);
        for (const auto& r : t.rows) EXPECT_EQ(r.keys.size(), 3u);
    }
    EXPECT_EQ(parts[0].rows[1].line, 5);
    EXPECT_EQ(parts[0].nearest({{"uplo", "L"}, {"n", "8"}, {"batch", "1024"}})->ranked[0].spelling, "tiny");
    EXPECT_FALSE(parts[1].rows[1].timed);
    EXPECT_EQ(parts[1].nearest({{"uplo", "L"}, {"n", "512"}, {"batch", "1024"}})->ranked[0].spelling, "cta");
    const std::vector<std::pair<std::string, std::string>> bad{
        {"# keys: uplo:exact dtype:exact n:log\nuplo=L dtype=float n=4 | tiny 1\n", "the first key must be dtype:exact"},
        {"# keys: dtype:log n:log\ndtype=4 n=4 | tiny 1\n", "the first key must be dtype:exact"},
        {"# op=synth dtype=float\n# keys: dtype:exact n:log\ndtype=float n=4 | tiny 1\n", "the dtype is a key column"},
        {"# keys: dtype:exact n:log\ndtype=half n=4 | tiny 1\n", "unknown dtype 'half'"},
        {"# keys: dtype:exact n:log\ndtype=float n=4 | tiny 1\ndtype=float n=4 | cta 1\n", "duplicate row"},
    };
    for (const auto& [t, why] : bad) {
        try {
            sel::parse_tables(t, "synth.sm_120.txt");
            ADD_FAILURE() << "accepted:\n" << t;
        } catch (const batchlas::internal_error& e) {
            EXPECT_NE(std::string(e.what()).find(why), std::string::npos) << e.what();
        }
    }
}

TEST(SelectTable, EveryParseErrorNamesFileAndLine) {
    const std::string keys = "# keys: uplo:exact n:log batch:log\n";
    const std::string head = "# op=synth dtype=float device=sm_120\n";
    const std::vector<std::tuple<std::string, std::string, std::string>> bad{
        {head + "uplo=L n=1 batch=1 | tiny 1\n", ":2:", "before the '# keys:'"},
        {head + "# keys: uplo n:log\n", ":2:", "needs :exact or :log"},
        {head + "# keys: n:log n:log\n", ":2:", "declared twice"},
        {head + "# keys: n:log:0\n", ":2:", "weight must be a positive number"},
        {head + "# keys: n:log:-1\n", ":2:", "weight must be a positive number"},
        {head + "# keys: n:log:1e3\n", ":2:", "weight must be a positive number"},
        {head + "# keys: n:log:3x\n", ":2:", "weight must be a positive number"},
        {head + "# keys: n:log:1.2.3\n", ":2:", "weight must be a positive number"},
        {head + "# keys: n:log:\n", ":2:", "weight must be a positive number"},
        {head + "# keys: n:log:3:4\n", ":2:", "only a :log key takes a weight"},
        {head + "# keys: uplo:exact:2\n", ":2:", "only a :log key takes a weight"},
        {head + keys + keys, ":3:", "second '# keys:'"},
        {head + keys + "uplo=L n=1 batch=1 m=3 | tiny 1\n", ":3:", "unknown key 'm=3'"},
        {head + keys + "uplo=L n=1 | tiny 1\n", ":3:", "lacks key 'batch'"},
        {head + keys + "uplo=L n=1 n=2 batch=1 | tiny 1\n", ":3:", "given twice"},
        {head + keys + "uplo=L n=0 batch=1 | tiny 1\n", ":3:", "positive integer"},
        {head + keys + "uplo=L n=6.5 batch=1 | tiny 1\n", ":3:", "positive integer"},
        {head + keys + "uplo=L n=1 batch=1 | tiny 1 2\n", ":3:", "is not '<choice> <ms>'"},
        {head + keys + "uplo=L n=1 batch=1 | tiny | cta 1\n", ":3:", "is not '<choice> <ms>'"},
        {head + keys + "uplo=L n=1 batch=1 | tiny 1 | \n", ":3:", "is not '<choice> <ms>'"},
        {head + keys + "uplo=L n=1 batch=1 | tiny fast\n", ":3:", "non-negative number"},
        {head + keys + "uplo=L n=1 batch=1 | tiny -1\n", ":3:", "non-negative number"},
        {head + keys + "uplo=L n=1 batch=1\n", ":3:", "no ranked entries"},
        {head + keys + "uplo=L n=1 batch=1 | tiny 1 | tiny 2\n", ":3:", "ranked twice"},
        {head + keys + "uplo=L n=1 batch=1 | tiny 1\n\nuplo=L batch=1 n=1 | cta 1\n", ":5:", "first at line 3"},
        {"# op=other dtype=float\n" + keys, ":1:", "disagrees with the file name"},
        {head + "# source=transcribed:abc\n" + keys + "uplo=L n=1 batch=1 | tiny - | cta 1\n", ":4:",
         "mixes timed and untimed"},
        {head + "# source=transcribed:abc\n" + keys + "uplo=L n=1 batch=1 | tiny 1 | cta -\n", ":4:",
         "mixes timed and untimed"},
        {head + "# source=transcribed:abc\n" + keys + "uplo=L n=1 batch=1 | tiny -\nuplo=L n=2 batch=1 | tiny 1 | c -\n",
         ":5:", "mixes timed and untimed"},
        {head + "# source=sweep.jsonl\n" + keys + "uplo=L n=1 batch=1 | tiny 1\nuplo=L n=2 batch=1 | tiny -\n",
         ":5:", "needs a 'source=transcribed:<sha>' header"},
        {head + keys + "uplo=L n=1 batch=1 | tiny -\n", ":3:", "needs a 'source=transcribed:<sha>' header"},
        {head + "# source=transcribed:\n" + keys + "uplo=L n=1 batch=1 | tiny -\n", ":2:", "must be a hex sha"},
        {head + "# source=transcribed:HEAD\n" + keys + "uplo=L n=1 batch=1 | tiny -\n", ":2:", "must be a hex sha"},
        {head + "# source=transcribed:abc\n" + keys + "uplo=L n=1 batch=1 | tiny --\n", ":4:", "or '-'"},
        {head + "# source=transcribed:abc\n" + keys + "uplo=L n=1 batch=1 | tiny - -\n", ":4:", "or '<choice> -'"},
    };
    for (const auto& [text, where, why] : bad) {
        try {
            sel::parse_table(text, "synth.sm_120.txt");
            ADD_FAILURE() << "accepted:\n" << text;
        } catch (const batchlas::internal_error& e) {
            const std::string m = e.what();
            EXPECT_EQ(m.find("synth.sm_120.txt" + where), 0u) << m;
            EXPECT_NE(m.find(why), std::string::npos) << m;
        }
    }
    EXPECT_THROW(sel::parse_table(keys, "untitled"), batchlas::internal_error);  // no op/dtype/device at all
}

// ---- nearest ----------------------------------------------------------------------------

// Tie winners sit AFTER the rows they beat, so "first row wins" cannot pass the tie tests.
const char* kGrid =
    "uplo=L n=128 batch=1024 | blocked 1\n"   // line 3
    "uplo=L n=64  batch=4096 | cta 1\n"       // 4
    "uplo=L n=64  batch=1024 | lpanel:8 1\n"  // 5
    "uplo=L n=32  batch=1024 | tiny 1\n"      // 6
    "uplo=U n=32  batch=1024 | vendor 1\n"    // 7
    "uplo=L n=1024 batch=1024 | tiny 2\n"     // 8
    "uplo=L n=256 batch=2048 | cta 2\n"       // 9
    "uplo=L n=512 batch=1024 | vendor 2\n";   // 10

int nearest_line(const sel::Key& k) {
    static const auto t = sel::parse_table(table("sm_120", kGrid), "synth.sm_120.txt");
    const auto* r = t.nearest(k);
    return r ? r->line : -1;
}

TEST(SelectNearest, OnGridAndEitherSideOfTheGeometricMidpoint) {
    EXPECT_EQ(nearest_line(key("L", 64, 1024)), 5);
    EXPECT_EQ(nearest_line(key("L", 90, 1024)), 5);    // midpoint of 64..128 is 90.51
    EXPECT_EQ(nearest_line(key("L", 91, 1024)), 3);
    EXPECT_EQ(nearest_line(key("L", 1, 1024)), 6);     // off the low end
    EXPECT_EQ(nearest_line(key("L", 5000, 1024)), 8);  // off the high end
}

TEST(SelectNearest, ExactKeyFiltersAndIsDroppedWhenNothingMatches) {
    EXPECT_EQ(nearest_line(key("U", 128, 1024)), 7);   // the only U row, though L n=128 is exact in n
    EXPECT_EQ(nearest_line(key("X", 128, 1024)), 3);   // no X rows: the filter is dropped
}

TEST(SelectNearest, LogDistanceSumsOverKeys) {
    EXPECT_EQ(nearest_line(key("L", 300, 1800)), 9);   // .23 + .19 beats every other row
    EXPECT_EQ(nearest_line(key("L", 64, 3000)), 4);    // batch alone separates two n=64 rows
}

TEST(SelectNearest, TiesGoToSmallerNThenSmallerBatch) {
    EXPECT_EQ(nearest_line(key("L", 64, 2048)), 5);    // b=4096 (line 4) and b=1024 tie: smaller batch
    const auto t = sel::parse_table(table("sm_120",
        "uplo=L n=128 batch=1024 | blocked 1\n"
        "uplo=L n=32 batch=1024 | tiny 1\n"
        "uplo=L n=128 batch=256 | cta 1\n"
        "uplo=L n=32 batch=4096 | vendor 1\n"), "synth.sm_120.txt");
    EXPECT_EQ(t.nearest(key("L", 64, 1024))->line, 4);  // n=128 and n=32 tie: smaller n
    EXPECT_EQ(t.nearest(key("L", 64, 512))->line, 4);   // three-way tie at 2: n decides before
                                                        // batch, so (128,256) loses to (32,1024)
}

// From key (32, 1024): row A is one octave off in n, row B two octaves off in batch. Equal
// weights pick A (1 < 2); n:log:3 picks B (3 > 2). The weight is the only difference.
TEST(SelectNearest, KeyWeightScalesThatKeysDistance) {
    const std::string rows = "uplo=L n=16 batch=1024 | tiny 1\n"   // line 3: A
                             "uplo=L n=32 batch=4096 | cta 1\n";   // 4: B
    const std::string head = "# op=synth dtype=float device=sm_120\n";
    const auto plain = sel::parse_table(head + "# keys: uplo:exact n:log batch:log\n" + rows, "synth.sm_120.txt");
    const auto cubed = sel::parse_table(head + "# keys: uplo:exact n:log:3 batch:log\n" + rows, "synth.sm_120.txt");
    const auto light = sel::parse_table(head + "# keys: uplo:exact n:log batch:log:0.4\n" + rows, "synth.sm_120.txt");
    EXPECT_EQ(plain.nearest(key("L", 32, 1024))->line, 3);
    EXPECT_EQ(cubed.nearest(key("L", 32, 1024))->line, 4);
    EXPECT_EQ(light.nearest(key("L", 32, 1024))->line, 4);  // 0.4 * 2 < 1: a fractional weight
    // A weighted tie (3*1 == 3) still goes to the smaller n.
    const auto tie = sel::parse_table(head + "# keys: uplo:exact n:log:3 batch:log\n"
                                      "uplo=L n=32 batch=8192 | cta 1\nuplo=L n=16 batch=1024 | tiny 1\n",
                                      "synth.sm_120.txt");
    EXPECT_EQ(tie.nearest(key("L", 32, 1024))->line, 4);
}

// Two exact keys, (R,T) missing. The (L,T) row is nearer in n, so an all-or-nothing drop
// lands on it and a drop from the left (keeping trans) does too; only dropping trans keeps R.
TEST(SelectNearest, ExactKeysAreDroppedFromTheRightOneAtATime) {
    const std::string rows = "side=L trans=N n=64 batch=1024 | tiny 1\n"     // line 3
                             "side=L trans=T n=64 batch=1024 | cta 1\n"      // 4
                             "side=R trans=N n=512 batch=1024 | blocked 1\n" // 5
                             "side=R trans=N n=8 batch=1024 | vendor 1\n";   // 6
    const std::string head = "# op=synth dtype=float device=sm_120\n";
    auto k = [](const char* side, const char* trans, std::int64_t n) {
        return sel::Key{{"side", side}, {"trans", trans}, {"n", n}, {"batch", 1024}};
    };
    const auto st = sel::parse_table(head + "# keys: side:exact trans:exact n:log batch:log\n" + rows,
                                     "synth.sm_120.txt");
    EXPECT_EQ(st.nearest(k("L", "T", 512))->line, 4);   // both exact keys match
    EXPECT_EQ(st.nearest(k("R", "T", 32))->line, 6);    // trans dropped: side=R kept, 8 beats 512
    EXPECT_EQ(st.nearest(k("R", "T", 300))->line, 5);   // ... and the log distance still decides
    EXPECT_EQ(st.nearest(k("R", "Z", 32))->line, 6);
    EXPECT_EQ(st.nearest(k("X", "T", 64))->line, 3);    // side unmatched: both dropped, pure distance
    // Declared order decides which key is kept: trans first keeps trans=T over side.
    const auto ts = sel::parse_table(head + "# keys: trans:exact n:log side:exact batch:log\n" + rows,
                                     "synth.sm_120.txt");
    EXPECT_EQ(ts.nearest(k("R", "T", 8))->line, 4);
    EXPECT_EQ(ts.nearest(k("R", "N", 8))->line, 6);
    // exact = {0, 2} is not {0, 1}: comparing by prefix position j instead of exact[j] never
    // matches side, drops it, and lets the nearer (L,N,64) row in.
    EXPECT_EQ(ts.nearest(k("R", "N", 32))->line, 6);
}

TEST(SelectNearest, MissingKeyThrows) {
    const auto t = sel::parse_table(table("sm_120", kGrid), "synth.sm_120.txt");
    EXPECT_THROW(t.nearest({{"uplo", "L"}, {"n", 64}}), batchlas::internal_error);
}

// ---- the ranked walk, borrowing and the last resort -------------------------------------

TEST_F(Select, RankedWalkSkipsUnrunnableAndNonCandidates) {
    sel::testing::set_builtin_tables({file("sm_120",
        "uplo=L n=64 batch=1024 | lpanel:panel=32 0.1 | lpanel:8 0.3 | vendor 0.5 | cta 0.6\n")});
    EXPECT_EQ(S(choose("sm_120", key("L", 64, 1024))), "lpanel:panel=8");
    EXPECT_EQ(S(choose("sm_120", key("L", 64, 1024), all_but({"lpanel:panel=8"}))), "vendor");
    EXPECT_EQ(S(choose("sm_120", key("L", 64, 1024), all_but({"lpanel:panel=8", "vendor"}))), "cta");
}

TEST_F(Select, EmbeddedTablesWithBadSpellingsFailLoudly) {
    sel::testing::set_builtin_tables({file("sm_120", "uplo=L n=64 batch=1024 | vendor 1\nuplo=L n=8 batch=1 | lpanel:x 1\n")});
    try {
        choose("sm_120", key("L", 64, 1024));
        ADD_FAILURE() << "accepted";
    } catch (const batchlas::internal_error& e) {
        EXPECT_NE(std::string(e.what()).find("synth.sm_120.txt:4:"), std::string::npos) << e.what();
    }
}

std::vector<std::string> order_for(const std::string& device) {
    std::vector<std::string> out;
    for (const auto* t : sel::tables_in_borrow_order("synth", "float", sel::device_from_key(device)))
        out.push_back(t->device);
    return out;
}

TEST_F(Select, BorrowOrder) {
    const std::string row = "uplo=L n=64 batch=1024 | vendor 1\n";
    sel::testing::set_builtin_tables({file("sm_120", row), file("sm_89", row), file("gfx90a", row),
                                      file("cpu", row), file("intel", row), file("sm_70", row, "other")});
    using V = std::vector<std::string>;
    EXPECT_EQ(order_for("sm_120"), (V{"sm_120", "sm_89", "gfx90a", "intel", "cpu"}));
    EXPECT_EQ(order_for("sm_86"), (V{"sm_89", "sm_120", "gfx90a", "intel", "cpu"}));
    EXPECT_EQ(order_for("sm_100"), (V{"sm_89", "sm_120", "gfx90a", "intel", "cpu"}));
    EXPECT_EQ(order_for("sm_75"), (V{"sm_89", "sm_120", "gfx90a", "intel", "cpu"}));
    EXPECT_EQ(order_for("gfx1100"), (V{"gfx90a", "sm_120", "sm_89", "intel", "cpu"}));
    EXPECT_EQ(order_for("rocm"), (V{"gfx90a", "sm_120", "sm_89", "intel", "cpu"}));
    EXPECT_EQ(order_for("cpu"), (V{"cpu"}));  // §3: the CPU never borrows
    EXPECT_EQ(order_for("intel"), (V{"intel", "sm_120", "sm_89", "gfx90a", "cpu"}));

    sel::testing::set_builtin_tables({file("sm_120", row), file("sm_80", row), file("sm_86", row)});
    EXPECT_EQ(order_for("sm_89"), (V{"sm_86", "sm_80", "sm_120"}));
    EXPECT_EQ(order_for("sm_75"), (V{"sm_80", "sm_86", "sm_120"}));
}

TEST_F(Select, BorrowedChoiceWarnsOncePerOp) {
    sel::testing::set_builtin_tables({
        file("sm_89", "uplo=L n=64 batch=1024 | cta 1 | vendor 2\n"),
        file("sm_120", "uplo=L n=64 batch=1024 | tiny 1 | vendor 2\n"),
        file("sm_89", "uplo=L n=64 batch=1024 | blocked 1\n", "synth2")});
    ::testing::internal::CaptureStderr();
    EXPECT_EQ(S(choose("sm_86", key("L", 64, 1024))), "cta");
    EXPECT_EQ(S(choose("sm_86", key("L", 80, 1024))), "cta");
    EXPECT_EQ(S(choose("sm_120", key("L", 64, 1024))), "tiny");
    EXPECT_EQ(S(choose("sm_86", key("L", 64, 1024), kAll, "synth2")), "blocked");
    const std::string err = ::testing::internal::GetCapturedStderr();
    EXPECT_EQ(err,
              "batchlas: synth has no float table for sm_86; borrowing sm_89 (run tools/tune to tune this device)\n"
              "batchlas: synth2 has no float table for sm_86; borrowing sm_89 (run tools/tune to tune this device)\n");
}

// An Apple GPU is keyed by its chip, from the device name; its family borrows its own chips
// first (nearest at or below), then the sm tables, never ahead of them a CPU table.
TEST(SelectDevice, AppleKeyIsTheChipName) {
    EXPECT_EQ(sel::apple_device_key("Apple M5 Max"), "apple_m5_max");
    EXPECT_EQ(sel::apple_device_key("Apple M2 Ultra"), "apple_m2_ultra");
    EXPECT_EQ(sel::apple_device_key("  Apple   M4 "), "apple_m4");
    EXPECT_EQ(sel::apple_device_key("M3 Pro"), "apple_m3_pro");
    const sel::Device d = sel::device_from_key("apple_m5_max");
    EXPECT_EQ(d.family, "apple");
    EXPECT_EQ(d.arch_number, 5);
    EXPECT_EQ(sel::device_from_key("apple_m12_ultra").arch_number, 12);
    // A table name must survive split_file_name: one dot-free device token.
    EXPECT_EQ(sel::apple_device_key("Apple M5.1 Max").find('.'), std::string::npos);
}

TEST_F(Select, AppleBorrowOrder) {
    const std::string row = "uplo=L n=64 batch=1024 | vendor 1\n";
    sel::testing::set_builtin_tables({file("sm_120", row), file("sm_89", row), file("gfx90a", row),
                                      file("cpu", row), file("intel", row), file("apple_m4", row),
                                      file("apple_m6", row)});
    using V = std::vector<std::string>;
    EXPECT_EQ(order_for("apple_m5_max"), (V{"apple_m4", "apple_m6", "sm_120", "sm_89", "gfx90a", "intel", "cpu"}));
    EXPECT_EQ(order_for("apple_m4"), (V{"apple_m4", "apple_m6", "sm_120", "sm_89", "gfx90a", "intel", "cpu"}));
    EXPECT_EQ(order_for("sm_89"), (V{"sm_89", "sm_120", "gfx90a", "intel", "apple_m6", "apple_m4", "cpu"}));
    sel::testing::set_builtin_tables({file("sm_120", row), file("sm_89", row), file("cpu", row)});
    EXPECT_EQ(order_for("apple_m5_max"), (V{"sm_120", "sm_89", "cpu"}));
}

TEST_F(Select, UntunedAppleGpuBorrowsAndWarnsOncePerOp) {
    sel::testing::set_builtin_tables({file("sm_120", "uplo=L n=64 batch=1024 | tiny 1 | vendor 2\n"),
                                      file("sm_89", "uplo=L n=64 batch=1024 | cta 1 | vendor 2\n")});
    ::testing::internal::CaptureStderr();
    EXPECT_EQ(S(choose("apple_m5_max", key("L", 64, 1024))), "tiny");
    EXPECT_EQ(S(choose("apple_m5_max", key("L", 80, 4096))), "tiny");
    EXPECT_EQ(::testing::internal::GetCapturedStderr(),
              "batchlas: synth has no float table for apple_m5_max; borrowing sm_120 (run tools/tune to tune this "
              "device)\n");
}

// describe() on a real Apple GPU: the chip key, not the generic "gpu" it had before.
TEST(SelectDevice, DescribeKeysAnAppleGpuByItsName) {
    const auto gpus = batchlas::Device::get_devices(batchlas::DeviceType::GPU);
    if (gpus.empty() || gpus.front().get_vendor() != batchlas::Vendor::APPLE) GTEST_SKIP() << "no Apple GPU";
    const sel::Device& d = sel::describe(gpus.front(), batchlas::Backend::SYCL, false);
    EXPECT_EQ(d.key, sel::apple_device_key(gpus.front().get_name()));
    EXPECT_EQ(d.key.rfind("apple_m", 0), 0u) << d.key;
    EXPECT_EQ(d.family, "apple");
    EXPECT_GT(d.arch_number, 0);
    EXPECT_EQ(d.slm_per_item, 8);
}

TEST_F(Select, CpuWithoutItsOwnTableTakesTheLastResortSilently) {
    sel::testing::set_builtin_tables({file("sm_120", "uplo=L n=64 batch=1024 | cta 1 | vendor 2\n")});
    ScopedEnvVar trace("BATCHLAS_SELECT_TRACE", "1");
    ::testing::internal::CaptureStderr();
    {
        const C c = choose("cpu", key("L", 64, 1024));
        EXPECT_EQ(S(c), "blocked");
        sel::TraceScope ts("synth", c, shape(64, 1024));
    }
    EXPECT_EQ(::testing::internal::GetCapturedStderr(), "synth float n=64 batch=1024 -> blocked  [last resort]\n");
}

// The device has a table, so this is not R8's untuned case: no warning, a distinct trace tag,
// and the warn-once is still unspent for a genuinely untuned device.
TEST_F(Select, OwnTableWithNothingRunnableFallsThroughWithoutTheBorrowWarning) {
    sel::testing::set_builtin_tables({file("sm_120", "uplo=L n=64 batch=1024 | vendor 1\n"),
                                      file("sm_89", "uplo=L n=64 batch=1024 | vendor 1 | cta 2\n")});
    ScopedEnvVar trace("BATCHLAS_SELECT_TRACE", "1");
    ::testing::internal::CaptureStderr();
    {
        const C c = choose("sm_120", key("L", 64, 1024), all_but({"vendor"}));
        EXPECT_EQ(S(c), "cta");
        sel::TraceScope ts("synth", c, shape(64, 1024));
    }
    EXPECT_EQ(S(choose("sm_86", key("L", 64, 1024), all_but({"vendor"}))), "cta");
    const std::string err = ::testing::internal::GetCapturedStderr();
    EXPECT_EQ(err,
              "synth float n=64 batch=1024 -> cta  2.00 ms  [sm_89 table, fallthrough from sm_120]\n"
              "batchlas: synth has no float table for sm_86; borrowing sm_89 (run tools/tune to tune this device)\n");
}

TEST_F(Select, LastResortOrderThenThrow) {
    sel::testing::set_builtin_tables({});
    EXPECT_EQ(S(choose("sm_120", key("L", 64, 1024))), "blocked");
    EXPECT_EQ(S(choose("sm_120", key("L", 64, 1024), only({"cta", "vendor"}))), "vendor");
    EXPECT_EQ(S(choose("sm_120", key("L", 64, 1024), only({"lpanel:panel=16", "cta"}))), "cta");
    sel::testing::set_builtin_tables({file("sm_120", "uplo=L n=64 batch=1024 | tiny 1\n")});
    EXPECT_EQ(S(choose("sm_120", key("L", 64, 1024), all_but({"tiny"}))), "blocked");
    try {
        choose("sm_120", key("L", 64, 1024), [](const C&) { return false; });
        ADD_FAILURE() << "no throw";
    } catch (const batchlas::unsupported& e) {
        EXPECT_STREQ(e.what(), "synth: no runnable kernel on sm_120");
    }
}

// ---- pins -------------------------------------------------------------------------------

const char* kPinRow = "uplo=L n=64 batch=1024 | vendor 0.2 | lpanel:8 0.3 | cta 0.6\n";

std::string pin_error(const std::string& pin, const Pred& ok = kAll) {
    sel::ScopedPin<C> p("synth", pin);
    try {
        choose("sm_120", key("L", 64, 1024), ok);
    } catch (const batchlas::invalid_argument& e) {
        return e.what();
    }
    return "<no throw>";
}

TEST_F(Select, BadPinsThrow) {
    sel::testing::set_builtin_tables({file("sm_120", kPinRow)});
    EXPECT_NE(pin_error("huge").find("synth: ScopedPin=\"huge\" is not a valid choice"), std::string::npos);
    EXPECT_NE(pin_error("lpanel:32").find("lpanel:panel=32 is not a compiled synth float candidate"),
              std::string::npos);
    EXPECT_NE(pin_error("wide:1:2").find("not a compiled"), std::string::npos);
    EXPECT_NE(pin_error("cta", all_but({"cta"})).find("cta cannot run this shape on sm_120"), std::string::npos);
    EXPECT_EQ(pin_error("vendor", all_but({"vendor"})), "<no throw>");  // a class word, like native
    // There are no aliases: origin:algorithm spellings are not choices.
    for (const char* old : {"native:tiny", "native:lpanel", "vendor:auto", "auto:auto", "netlib"})
        EXPECT_NE(pin_error(old).find("is not a valid choice: '" + std::string(old) + "'"),
                  std::string::npos)
            << old;
    EXPECT_EQ(pin_error("cta"), "<no throw>");
}

// Bare `vendor` is a class word (§12): with no runnable vendor candidate -- none compiled, or
// a vendor-free build -- it warns once per op and runs the automatic choice instead of throwing.
TEST_F(Select, VendorPinWithNoRunnableVendorFallsBackToAutoAndWarnsOnce) {
    sel::testing::set_builtin_tables({file("sm_120", kPinRow), file("sm_120", kPinRow, "synth2")});
    sel::ScopedPin<C> p("synth", "vendor");
    sel::ScopedPin<C> p2("synth2", "vendor");
    EXPECT_EQ(S(choose("sm_120", key("L", 64, 1024))), "vendor");
    EXPECT_EQ(S(choose("sm_120", key("L", 64, 1024), only({"tiny", "vendor"}))), "vendor");  // pinned over the row
    ::testing::internal::CaptureStderr();
    EXPECT_EQ(S(choose("sm_120", key("L", 64, 1024), all_but({"vendor"}))), "lpanel:panel=8");
    EXPECT_EQ(S(sel::choose("synth", "float", sel::device_from_key("sm_120"), key("L", 64, 1024), kNoVendor, kAll,
                            kRules)),
              "lpanel:panel=8");
    EXPECT_EQ(S(choose("sm_120", key("L", 64, 1024), all_but({"vendor"}), "synth2")), "lpanel:panel=8");
    const std::string err = ::testing::internal::GetCapturedStderr();
    EXPECT_EQ(err, "batchlas: synth pinned \"vendor\", but no vendor candidate can run this shape on sm_120; "
                   "using the automatic choice\n"
                   "batchlas: synth2 pinned \"vendor\", but no vendor candidate can run this shape on sm_120; "
                   "using the automatic choice\n");
}

// The tuner's arms: a strict class word that cannot serve the shape throws instead of timing Auto.
TEST_F(Select, StrictClassWordPinThrowsInsteadOfFallingBack) {
    sel::testing::set_builtin_tables({file("sm_120", kPinRow)});
    auto strict_error = [](const std::string& pin, const Pred& ok) -> std::string {
        sel::ScopedPin<C> p("synth", pin, sel::StrictPin{});
        try {
            return S(choose("sm_120", key("L", 64, 1024), ok));
        } catch (const batchlas::invalid_argument& e) {
            return e.what();
        }
    };
    EXPECT_EQ(strict_error("vendor", kAll), "vendor");
    EXPECT_EQ(strict_error("native", kAll), "lpanel:panel=8");
    EXPECT_EQ(strict_error("cta", kAll), "cta");
    EXPECT_EQ(strict_error("vendor", all_but({"vendor"})),
              "synth: ScopedPin=\"vendor\" (strict): no vendor candidate can run this shape on sm_120");
    EXPECT_EQ(strict_error("native", only({"vendor"})),
              "synth: ScopedPin=\"native\" (strict): no native candidate can run this shape on sm_120");
    EXPECT_EQ(pin_error("vendor", all_but({"vendor"})), "<no throw>") << "an ordinary pin still falls back";
    {
        sel::ScopedPin<C> outer("synth", "vendor", sel::StrictPin{});
        sel::ScopedPin<C> inner("synth", "vendor");
        EXPECT_EQ(S(choose("sm_120", key("L", 64, 1024), all_but({"vendor"}))), "lpanel:panel=8")
            << "strictness belongs to the innermost pin";
    }
}

TEST_F(Select, ConcretePinsAndNormalisation) {
    sel::testing::set_builtin_tables({file("sm_120", kPinRow)});
    EXPECT_EQ(S(choose("sm_120", key("L", 64, 1024))), "vendor");
    const std::vector<std::pair<std::string, std::string>> pins{
        {"tiny", "tiny"}, {"lpanel:16", "lpanel:panel=16"}, {"  LPanel:Panel=16 ", "lpanel:panel=16"},
        {"Blocked", "blocked"}, {"blocked", "blocked"}};
    for (const auto& [pin, want] : pins) {
        sel::ScopedPin<C> p("synth", pin);
        EXPECT_EQ(S(choose("sm_120", key("L", 64, 1024))), want) << pin;
    }
    sel::ScopedPin<C> p("synth", C{Lpanel{16}});
    EXPECT_EQ(S(choose("sm_120", key("L", 64, 1024))), "lpanel:panel=16");
}

TEST_F(Select, NativePinTakesBestRunnableNonVendor) {
    sel::testing::set_builtin_tables({file("sm_120", kPinRow)});
    sel::ScopedPin<C> p("synth", "native");
    EXPECT_EQ(S(choose("sm_120", key("L", 64, 1024))), "lpanel:panel=8");
    EXPECT_EQ(S(choose("sm_120", key("L", 64, 1024), all_but({"lpanel:panel=8"}))), "cta");
    // Nothing native in the row runs: the last resort, still skipping vendor.
    EXPECT_EQ(S(choose("sm_120", key("L", 64, 1024), all_but({"lpanel:panel=8", "cta"}))), "blocked");
}

TEST_F(Select, NativePinWithNoRunnableNativeFallsBackToAutoAndWarnsOnce) {
    sel::testing::set_builtin_tables({file("sm_120", "uplo=L n=64 batch=1024 | lpanel:8 0.1 | vendor 0.2\n")});
    sel::ScopedPin<C> p("synth", "native");
    ::testing::internal::CaptureStderr();
    EXPECT_EQ(S(choose("sm_120", key("L", 64, 1024), only({"vendor"}))), "vendor");
    EXPECT_EQ(S(choose("sm_120", key("L", 64, 1024), only({"vendor"}))), "vendor");
    const std::string err = ::testing::internal::GetCapturedStderr();
    EXPECT_EQ(err, "batchlas: synth pinned \"native\", but no native candidate can run this shape on sm_120; "
                   "using the automatic choice\n");
}

TEST_F(Select, ScopedPinNestsRestoresAndIsPerOp) {
    sel::testing::set_builtin_tables({file("sm_120", kPinRow)});
    {
        sel::ScopedPin<C> outer("synth", C{Tiny{}});
        {
            sel::ScopedPin<C> inner("synth", C{Cta{}});
            sel::ScopedPin<C> other("other_op", C{Blocked{}});
            EXPECT_EQ(S(choose("sm_120", key("L", 64, 1024))), "cta");
        }
        EXPECT_EQ(S(choose("sm_120", key("L", 64, 1024))), "tiny");
    }
    EXPECT_EQ(S(choose("sm_120", key("L", 64, 1024))), "vendor");
}

TEST_F(Select, EnvPinIsReadOnEveryCallAndScopedPinBeatsIt) {
    sel::testing::set_builtin_tables({file("sm_120", kPinRow, "potrf")});
    auto pick = [] { return S(choose("sm_120", key("L", 64, 1024), kAll, "potrf")); };
    ScopedEnvVar clear("BATCHLAS_POTRF_ROUTE", nullptr);
    EXPECT_EQ(pick(), "vendor");
    {
        ScopedEnvVar a("BATCHLAS_POTRF_ROUTE", "cta");
        EXPECT_EQ(pick(), "cta");
        {
            ScopedEnvVar b("BATCHLAS_POTRF_ROUTE", "native");
            EXPECT_EQ(pick(), "lpanel:panel=8");
        }
        EXPECT_EQ(pick(), "cta");
        {
            sel::ScopedPin<C> p("potrf", C{Tiny{}});
            EXPECT_EQ(pick(), "tiny");
        }
        {
            sel::ScopedPin<C> p("potrf", "auto");
            EXPECT_EQ(pick(), "vendor");
        }
        ScopedEnvVar bad("BATCHLAS_POTRF_ROUTE", "regsiter_tiled");
        try {
            pick();
            ADD_FAILURE() << "no throw";
        } catch (const batchlas::invalid_argument& e) {
            EXPECT_NE(std::string(e.what()).find("BATCHLAS_POTRF_ROUTE=\"regsiter_tiled\""), std::string::npos);
        }
    }
    EXPECT_EQ(pick(), "vendor");
}

// ---- BATCHLAS_TUNED_DIR -----------------------------------------------------------------

struct TempDir {
    std::filesystem::path path;
    TempDir() : TempDir("select_tests." + std::to_string(::getpid())) {}
    explicit TempDir(const std::string& name) : path(std::filesystem::temp_directory_path() / name) {
        std::filesystem::create_directories(path);
    }
    ~TempDir() { std::filesystem::remove_all(path); }
    void write(const std::pair<std::string, std::string>& f) const { std::ofstream(path / f.first) << f.second; }
};

TEST_F(Select, TunedDirReplacesSameNamedTablesAndTagsThem) {
    sel::testing::set_builtin_tables({file("sm_120", "uplo=L n=64 batch=1024 | tiny 0.302\n")});
    TempDir dir;
    dir.write(file("sm_120", "uplo=L n=64 batch=1024 | cta 0.25 | tiny 0.302\n"));
    dir.write(file("sm_89", "uplo=L n=64 batch=1024 | blocked 1\n"));
    dir.write({"notes.md", "not a table"});
    dir.write({"README.txt", "prose, which parse_table would reject\n"});
    dir.write({"synth.float.sm_120.bak.txt", table("sm_120", "uplo=L n=64 batch=1024 | vendor 0.01\n")});
    ScopedEnvVar trace("BATCHLAS_SELECT_TRACE", "1");
    EXPECT_EQ(S(choose("sm_120", key("L", 64, 1024))), "tiny");
    ScopedEnvVar over("BATCHLAS_TUNED_DIR", dir.path.c_str());
    ::testing::internal::CaptureStderr();
    {
        const C c = choose("sm_120", key("L", 64, 1024));
        EXPECT_EQ(S(c), "cta");
        sel::TraceScope ts("synth", c, shape(64, 1024));
    }
    {
        const C c = choose("sm_86", key("L", 64, 1024));  // a table only the override dir has
        EXPECT_EQ(S(c), "blocked");
        sel::TraceScope ts("synth", c, shape(64, 1024));
    }
    const std::string err = ::testing::internal::GetCapturedStderr();
    EXPECT_NE(err.find("synth float n=64 batch=1024 -> cta  0.25 ms, next tiny 0.302 ms  [override sm_120]\n"),
              std::string::npos) << err;
    EXPECT_NE(err.find("-> blocked  1.00 ms  [override sm_89, borrowed for sm_86]\n"), std::string::npos) << err;
    for (const char* junk : {"README.txt", "synth.float.sm_120.bak.txt"})
        EXPECT_NE(err.find(std::string("BATCHLAS_TUNED_DIR: skipping ") + junk + " (not <op>.<device>.txt)\n"),
                  std::string::npos) << junk << "\n" << err;
}

TEST_F(Select, TunedDirThatIsNotADirectoryIsIgnoredWithOneWarning) {
    sel::testing::set_builtin_tables({file("sm_120", "uplo=L n=64 batch=1024 | tiny 1\n")});
    ScopedEnvVar over("BATCHLAS_TUNED_DIR", "/nonexistent/select_tests");
    ::testing::internal::CaptureStderr();
    EXPECT_EQ(S(choose("sm_120", key("L", 64, 1024))), "tiny");
    EXPECT_EQ(S(choose("sm_120", key("L", 64, 1024))), "tiny");
    const std::string err = ::testing::internal::GetCapturedStderr();
    EXPECT_EQ(err, "batchlas: BATCHLAS_TUNED_DIR=/nonexistent/select_tests is not a directory; ignoring it\n");
}

// A bad override table is the user's input (invalid_argument); the same text built in is a
// library defect (internal_error, EmbeddedTablesWithBadSpellingsFailLoudly).
TEST_F(Select, TunedDirWithBadTableIsAnInvalidArgument) {
    sel::testing::set_builtin_tables({file("sm_120", "uplo=L n=64 batch=1024 | tiny 1\n")});
    const std::vector<std::pair<std::string, std::string>> bad{
        file("sm_120", "uplo=L n=64 batch=1024 | lpanel:x 1\n"),
        {"synth.sm_120.txt", "# keys: dtype:exact n:log\ndtype=half n=4 | tiny 1\n"},
        {"synth.sm_120.txt", "# keys: dtype:exact side:exact\ndtype=float side=L | tiny 1\n"},
    };
    for (std::size_t i = 0; i < bad.size(); ++i) {
        const auto& f = bad[i];  // a fresh dir each: the loader caches by path
        TempDir dir("select_tests.bad_dir." + std::to_string(::getpid()) + "." + std::to_string(i));
        dir.write(f);
        ScopedEnvVar over("BATCHLAS_TUNED_DIR", dir.path.c_str());
        try {
            choose("sm_120", key("L", 64, 1024));
            ADD_FAILURE() << "accepted:\n" << f.second;
        } catch (const batchlas::invalid_argument& e) {
            EXPECT_EQ(std::string(e.what()).rfind("BATCHLAS_TUNED_DIR: synth.sm_120.txt", 0), 0u) << e.what();
        }
    }
}

// The loader parses each table file once, however many dtypes ask for it (gemm's files are
// ~14 MB), built-in and BATCHLAS_TUNED_DIR alike, and each dtype's table equals a fresh parse.
TEST_F(Select, TableLoaderParsesEachFileOnce) {
    TempDir dir("select_tests.parse_once." + std::to_string(::getpid()));
    std::vector<std::pair<std::string, std::string>> files;
    for (const auto& t : sel::embedded_tables())
        if (t.name.rfind("gemm.", 0) == 0) files.emplace_back(std::string(t.name), std::string(t.text));
    ASSERT_GE(files.size(), 2u);
    for (const auto& f : files) dir.write(f);
    ScopedEnvVar tuned("BATCHLAS_TUNED_DIR", dir.path.string().c_str());
    sel::testing::use_embedded_tables();  // a new generation: nothing cached
    const std::size_t before = sel::testing::parse_count();
    const auto dev = sel::device_from_key(files.front().first.substr(5, files.front().first.size() - 9));
    std::map<std::string, std::vector<sel::Table>> fresh;  // the reference: one direct parse per file
    for (const char* dtype : {"float", "double", "cfloat", "cdouble"}) {
        const auto tables = sel::tables_in_borrow_order("gemm", dtype, dev);
        ASSERT_EQ(tables.size(), files.size()) << dtype;
        for (const sel::Table* t : tables) {
            EXPECT_TRUE(t->is_override) << t->file;
            const auto it = std::find_if(files.begin(), files.end(), [&](const auto& f) { return f.first == t->file; });
            ASSERT_NE(it, files.end()) << t->file;
            auto& ref = fresh[t->file];
            if (ref.empty()) ref = sel::parse_tables(it->second, it->first);
            const auto want = std::find_if(ref.begin(), ref.end(), [&](const sel::Table& x) { return x.dtype == dtype; });
            ASSERT_NE(want, ref.end()) << t->file << " " << dtype;
            ASSERT_EQ(t->rows.size(), want->rows.size()) << t->file << " " << dtype;
            EXPECT_EQ(t->keys.size(), want->keys.size()) << t->file;
            for (std::size_t r = 0; r < t->rows.size(); ++r) {
                ASSERT_EQ(t->rows[r].keys, want->rows[r].keys) << t->file << " " << dtype << " row " << r;
                ASSERT_EQ(t->rows[r].ranked.size(), want->rows[r].ranked.size()) << t->file << " row " << r;
                for (std::size_t e = 0; e < t->rows[r].ranked.size(); ++e)
                    ASSERT_EQ(t->rows[r].ranked[e].spelling, want->rows[r].ranked[e].spelling) << t->file << " row " << r;
            }
        }
    }
    // Every built-in gemm file and its same-named override, once each (the direct reference parses
    // above do not go through the loader, so they are not counted).
    EXPECT_EQ(sel::testing::parse_count() - before, 2 * files.size());
}

TEST(SelectNativeFacts, OverTheCandidateList) {
    const auto f = sel::native_facts(kCands, only({"vendor"}));
    EXPECT_TRUE(f.existed);
    EXPECT_EQ(f.supported, 0);
    EXPECT_EQ(sel::native_facts(kCands, only({"lpanel:panel=16"})).supported, 1);
    const std::array<C, 1> vendor_only{Vendor{}};
    EXPECT_FALSE(sel::native_facts(vendor_only, kAll).existed);
}

// ---- trace ------------------------------------------------------------------------------

TEST_F(Select, TraceLinesAndIndentation) {
    sel::testing::set_builtin_tables({
        file("sm_120", "uplo=L n=64 batch=1024 | lpanel:8 0.302 | cta 0.6 | vendor 0.49\n"
                       "uplo=L n=512 batch=2048 | blocked 14.10 | vendor 13.74 | lpanel:8 14.84\n"),
        file("sm_120", "uplo=L n=64 batch=1024 | cta 0.5\n", "child"),
        file("sm_89", "uplo=L n=64 batch=1024 | lpanel:8 0.302\n")});
    ScopedEnvVar trace("BATCHLAS_SELECT_TRACE", "1");
    ::testing::internal::CaptureStderr();
    {
        const C c = choose("sm_120", key("L", 64, 1024), all_but({"cta"}));
        sel::TraceScope ts("synth", c, shape(64, 1024));
        {
            const C k = choose("sm_120", key("L", 64, 1024), kAll, "child");
            sel::TraceScope tk("child", k, shape(64, 1024));
            const C g = choose("sm_120", key("L", 64, 1024), kAll, "child");
            sel::TraceScope tg("child", g, shape(64, 1024));
        }
    }
    {
        const C c = choose("sm_120", key("L", 512, 2048));
        sel::TraceScope ts("synth", c, shape(512, 2048));
    }
    {
        sel::ScopedPin<C> p("synth", C{Tiny{}});
        const C c = choose("sm_120", key("L", 64, 1024));
        sel::TraceScope ts("synth", c, shape(64, 1024));
    }
    {
        const C c = choose("sm_120", key("L", 64, 1024), only({"tiny", "blocked"}));
        sel::TraceScope ts("synth", c, shape(64, 1024));
    }
    {
        const C c = choose("sm_86", key("L", 64, 1024));
        sel::TraceScope ts("synth", c, shape(64, 1024));
    }
    sel::TraceScope stray("synth", C{Cta{}}, shape(8, 8));
    const std::string err = ::testing::internal::GetCapturedStderr();
    EXPECT_EQ(err,
              "synth float n=64 batch=1024 -> lpanel:panel=8  0.302 ms, next vendor 0.49 ms  [sm_120]\n"
              "  child float n=64 batch=1024 -> cta  0.5 ms  [sm_120]\n"
              "    child float n=64 batch=1024 -> cta  0.5 ms  [sm_120]\n"
              "synth float n=512 batch=2048 -> blocked  14.10 ms, tied with vendor 13.74 ms (3%)  [sm_120]\n"
              "synth float n=64 batch=1024 -> tiny  [pinned]\n"
              "synth float n=64 batch=1024 -> blocked  [last resort]\n"
              "batchlas: synth has no float table for sm_86; borrowing sm_89 (run tools/tune to tune this device)\n"
              "synth float n=64 batch=1024 -> lpanel:panel=8  0.302 ms  [sm_89 table, borrowed for sm_86]\n"
              "synth float n=8 batch=8 -> cta  [untraced]\n");
}

// A transcribed row is walked like a timed one (rank order, can_run, candidates); the trace
// says "transcribed" where a timed row prints its times.
TEST_F(Select, TranscribedRowsWalkInRankOrderAndTraceAsTranscribed) {
    sel::testing::set_builtin_tables({{"synth.sm_89.txt",
        "# op=synth device=sm_89 source=transcribed:2b46acab\n"
        "# keys: dtype:exact uplo:exact n:log batch:log\n"
        "dtype=float uplo=L n=20 batch=8192 | tiny - | cta - | blocked -\n"
        "dtype=float uplo=L n=512 batch=8192 | blocked - | vendor -\n"}});
    ScopedEnvVar trace("BATCHLAS_SELECT_TRACE", "1");
    ::testing::internal::CaptureStderr();
    for (const auto& [n, ok] : std::vector<std::pair<int, Pred>>{{20, kAll}, {20, all_but({"tiny"})}, {512, kAll}}) {
        const C c = choose("sm_89", key("L", n, 8192), ok);
        sel::TraceScope ts("synth", c, shape(n, 8192));
    }
    EXPECT_EQ(::testing::internal::GetCapturedStderr(),
              "synth float n=20 batch=8192 -> tiny  transcribed  [sm_89]\n"
              "synth float n=20 batch=8192 -> cta  transcribed  [sm_89]\n"
              "synth float n=512 batch=8192 -> blocked  transcribed  [sm_89]\n");
}

// describe() keys its memo on the vendor flag, and device_of fills it from the asking op's
// library group: each group's own compile-time predicate, none for an op without a vendor.
TEST(SelectDevice, VendorFlagIsTheOpsLibraryGroupAndPartOfTheMemoKey) {
    const batchlas::Device dev = batchlas::Device::default_device();
#if BATCHLAS_HAS_SYCL_BACKEND && !BATCHLAS_HAS_CUDA_BACKEND
    constexpr auto B = batchlas::Backend::SYCL;  // the Metal build's only GPU backend
#else
    constexpr auto B = batchlas::Backend::CUDA;
#endif
    const sel::Device& without = sel::describe(dev, B, false);
    const sel::Device& with = sel::describe(dev, B, true);
    EXPECT_FALSE(without.has_vendor);
    EXPECT_TRUE(with.has_vendor);
    EXPECT_EQ(&sel::describe(dev, B, true), &with);  // memoized
    batchlas::Queue q(dev, B);
    EXPECT_FALSE(sel::device_of<B>(q).has_vendor);
    EXPECT_EQ(sel::device_of<B>(q, sel::Lib::level3).has_vendor, sel::level3_vendor_available<B>);
    EXPECT_EQ(sel::device_of<B>(q, sel::Lib::factorization).has_vendor, sel::factorization_vendor_available<B>);
    EXPECT_EQ(sel::device_of<B>(q, sel::Lib::solver).has_vendor, sel::solver_vendor_available<B>);
    EXPECT_EQ(sel::device_of<B>(q, sel::Lib::sparse).has_vendor, sel::sparse_vendor_available<B>);
}

// A field-less op's candidates<T>() is all_of: declaration order is its tie-break order.
TEST(SelectDevice, AllOfListsTheVariantInDeclarationOrder) {
    const auto all = sel::all_of<C>();
    ASSERT_EQ(all.size(), std::variant_size_v<C>);
    EXPECT_EQ(S(all.front()), "tiny");
    EXPECT_EQ(S(all.back()), "vendor");
}

// The capture mode of run_factor_grid.sh / route_diff.sh: coverage on, trace off, so no
// decision is noted and the row's scalar/backend/uplo must come from the shape alone.
// threadsafe: the child re-executes the binary instead of forking a process whose SYCL runtime
// already started threads (an AdaptiveCpp child hangs). It re-runs this body, so the path is fixed.
TEST(SelectCoverageDeathTest, RowCarriesScalarBackendAndUploWithTraceOff) {
    ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
    std::filesystem::remove_all(std::filesystem::temp_directory_path() / "select_tests.coverage");
    TempDir dir("select_tests.coverage");
    const std::string out = (dir.path / "cov").string();
    auto child = [&] {
        ScopedEnvVar trace("BATCHLAS_SELECT_TRACE", nullptr);
        ScopedEnvVar cov("BATCHLAS_COVERAGE_OUT", out.c_str());
        batchlas::coverage::g_dynamic_enabled = true;  // latched at static init
        const batchlas::coverage::Shape s{.scalar = batchlas::ScalarKind::C64, .backend = batchlas::Backend::CUDA,
                                          .m = 64, .n = 64, .k = 64, .batch = 1024, .uplo = batchlas::Uplo::Upper};
        { sel::TraceScope ts("potrf", C{Cta{}}, s, sel::NativeFacts{true, 0}); }
        std::exit(0);  // emit() runs from atexit
    };
    EXPECT_EXIT(child(), ::testing::ExitedWithCode(0), "");
    std::vector<std::string> row;
    for (const auto& ent : std::filesystem::directory_iterator(dir.path)) {
        std::ifstream in(ent.path());
        for (std::string line; std::getline(in, line);)
            if (line.rfind("reached,potrf,", 0) == 0) {
                ASSERT_TRUE(row.empty()) << "second potrf row: " << line;
                for (auto f : sel::detail::split(line, ',')) row.emplace_back(f);
            }
    }
    ASSERT_GE(row.size(), 16u);
    EXPECT_EQ(row[2], "complex<double>");
    EXPECT_EQ(row[3], "CUDA");
    EXPECT_EQ(row[6], "64");
    EXPECT_EQ(row[8], "1024");
    EXPECT_EQ(row[9], "native");
    EXPECT_EQ(row[10], "cta");
    EXPECT_EQ(row[12], "1");  // native_route_existed, from the NativeFacts passed in
    EXPECT_EQ(row[13], "0");  // native_route_supported: 0, not the old constant -1
    EXPECT_EQ(row[15], std::to_string(static_cast<int>(batchlas::Uplo::Upper)));
}

TEST_F(Select, TraceIsSilentWhenOff) {
    sel::testing::set_builtin_tables({file("sm_120", "uplo=L n=64 batch=1024 | tiny 1\n")});
    ScopedEnvVar trace("BATCHLAS_SELECT_TRACE", nullptr);
    ::testing::internal::CaptureStderr();
    {
        const C c = choose("sm_120", key("L", 64, 1024));
        sel::TraceScope ts("synth", c, shape(64, 1024));
    }
    EXPECT_EQ(::testing::internal::GetCapturedStderr(), "");
}

}  // namespace

// One batch policy across every public entry point (WP3): an empty batch is a no-op, and a
// heterogeneous batch never reaches a family that reads one (n, ld, stride) for the whole batch.
// These run real calls on a real queue, unlike the synthetic tests above.
namespace {

using batchlas::Backend;
using batchlas::MatrixFormat;
template <class T>
using Dense = batchlas::MatrixView<T, MatrixFormat::Dense>;
template <class T>
using RealOf = typename batchlas::base_type<T>::type;

template <class T, Backend B>
struct PolicyConfig {
    using ScalarType = T;
    static constexpr Backend BackendVal = B;
};
using PolicyTuple = decltype(std::tuple_cat(
#if BATCHLAS_HAS_HOST_BACKEND && BATCHLAS_HAS_CPU_TARGET
    std::tuple<PolicyConfig<float, Backend::NETLIB>, PolicyConfig<std::complex<double>, Backend::NETLIB>>{},
#endif
#if BATCHLAS_HAS_CUDA_BACKEND
    std::tuple<PolicyConfig<float, Backend::CUDA>, PolicyConfig<std::complex<double>, Backend::CUDA>>{},
#endif
#if BATCHLAS_HAS_ROCM_BACKEND
    std::tuple<PolicyConfig<float, Backend::ROCM>, PolicyConfig<std::complex<double>, Backend::ROCM>>{},
#endif
    std::tuple<>{}));
using PolicyTypes = test_utils::tuple_to_types<PolicyTuple>::type;

template <class Config>
class BatchPolicy : public test_utils::BatchLASTest<Config> {};
TYPED_TEST_SUITE(BatchPolicy, PolicyTypes);

// Every op once, by name, on `batch` items of order n at a padded ld, a stride that is not
// ld * n and inc 2; with `het`, A, B and C carry active orders {n, n-3, n, n-5, ...}. Every
// buffer starts as a large finite sentinel, so a write anywhere shows.
template <class T, Backend B>
struct OpBattery {
    static constexpr int n = 8, ld = 11, stride = 11 * 8 + 5;
    batchlas::Queue& q;
    int batch;
    batchlas::Span<std::byte> ws;
    batchlas::UnifiedVector<T> a, b, c, vx, vy, vals, tau;
    batchlas::UnifiedVector<RealOf<T>> w;
    batchlas::UnifiedVector<std::int64_t> piv;
    batchlas::UnifiedVector<std::int32_t> info;
    batchlas::UnifiedVector<int> act, ro, ci;
    batchlas::UnifiedVector<T*> pa, pb, pc;  // the per-item pointer arrays a vendor batched call fills

    static T sentinel() { return Dense<T>::make(RealOf<T>(-999), RealOf<T>(777)); }
    std::size_t items() const { return std::size_t(std::max(batch, 1)); }
    std::size_t len() const { return std::size_t(stride) * items() + 64; }
    OpBattery(batchlas::Queue& queue, int nbatch, batchlas::Span<std::byte> work)
        : q(queue), batch(nbatch), ws(work), a(len(), sentinel()), b(len(), sentinel()), c(len(), sentinel()),
          vx(len(), sentinel()), vy(len(), sentinel()), vals(len(), sentinel()), tau(len(), sentinel()),
          w(len(), RealOf<T>(-999)), piv(len(), -999), info(len(), -999), act(items(), n), ro(len(), 0),
          ci(len(), 0), pa(items(), nullptr), pb(items(), nullptr), pc(items(), nullptr) {
        constexpr int drop[4] = {0, 3, 0, 5};
        for (std::size_t i = 0; i < items(); ++i) act[i] = n - drop[i % 4];
        for (int i = 0; i <= n; ++i) ro[i] = i;
        for (int i = 0; i < n; ++i) ci[i] = (i + 3) % n;
    }
    Dense<T> view(batchlas::UnifiedVector<T>& buf, bool het) {
        T** ptrs = &buf == &a ? pa.data() : (&buf == &b ? pb.data() : pc.data());
        Dense<T> v(buf.data(), n, n, ld, stride, batch, ptrs);
        return het ? v.with_active_dims(act.to_span(), act.to_span()) : v;
    }
    batchlas::VectorView<T> vec(batchlas::UnifiedVector<T>& buf) { return {buf.data(), n, batch, 2, 2 * n + 3}; }
    batchlas::MatrixView<T, MatrixFormat::CSR> csr() {
        return {vals.data(), ro.data(), ci.data(), n, n, batchlas::NonZeros{n}, n + 2, 0, batch};
    }
    batchlas::Span<RealOf<T>> wspan() { return {w.data(), std::size_t(n) * items()}; }
    batchlas::Span<T> tspan() { return {tau.data(), std::size_t(n) * items()}; }
    batchlas::Span<std::int64_t> pspan() { return {piv.data(), std::size_t(n) * items()}; }
    batchlas::Span<std::int32_t> ispan() { return {info.data(), items()}; }

    using Calls = std::vector<std::pair<std::string, std::function<void()>>>;
    Calls calls(bool het) {
        using namespace batchlas;
        const T one = T(1), half = T(0.5);
        const auto A = view(a, het), Bv = view(b, het), Cv = view(c, het);
        const auto L = Uplo::Lower;
        const auto N = Transpose::NoTrans;
        Calls out = {
            {"gemm", [=, this] { gemm<B, T>(q, A, Bv, Cv, one, half, N, N); }},
            {"gemv", [=, this] { gemv<B, T>(q, A, vec(vx), vec(vy), one, half, N); }},
            {"trsm", [=, this] { trsm<B, T>(q, A, Bv, one, Side::Left, L, N, Diag::NonUnit); }},
            {"trmm", [=, this] { trmm<B, T>(q, A, Bv, Cv, one, Side::Left, L, N, Diag::NonUnit); }},
            {"potrf", [=, this] { potrf<B, T>(q, A, L, ws, ispan()); }},
            {"getrf", [=, this] { getrf<B, T>(q, A, pspan(), ws, ispan()); }},
            {"getrs", [=, this] { getrs<B, T>(q, A, Bv, N, pspan(), ws); }},
            {"getri", [=, this] { getri<B, T>(q, A, Cv, pspan(), ws, ispan()); }},
            {"geqrf", [=, this] { geqrf<B, T>(q, A, tspan(), ws); }},
            {"orgqr", [=, this] { orgqr<B, T>(q, A, tspan(), ws); }},
            {"ormqr", [=, this] { ormqr<B, T>(q, A, Cv, Side::Left, N, tspan(), ws); }},
            {"syev", [=, this] { syev<B, T>(q, A, wspan(), JobType::EigenVectors, L, ws, ispan()); }},
            {"gesvd", [=, this] { gesvd<B, T>(q, A, wspan(), Bv, Cv, SvdVectors::All, SvdVectors::All, ws, ispan()); }},
            {"gesv", [=, this] { gesv<B, T>(q, A, Bv, pspan(), ws, ispan()); }},
            {"posv", [=, this] { posv<B, T>(q, A, Bv, L, ws, ispan()); }},
            {"spmm", [=, this] { spmm<B, T, MatrixFormat::CSR>(q, csr(), Bv, Cv, one, half, N, N, ws); }},
        };
        if constexpr (is_std_complex_v<T>) {
            out.emplace_back("hemm", [=, this] { hemm<B, T>(q, A, Bv, Cv, one, half, Side::Left, L); });
            out.emplace_back("herk", [=, this] { herk<B, T>(q, A, Cv, RealOf<T>(1), RealOf<T>(0.5), L, N); });
            out.emplace_back("her2k", [=, this] { her2k<B, T>(q, A, Bv, Cv, one, RealOf<T>(0.5), L, N); });
        } else {
            out.emplace_back("symm", [=, this] { symm<B, T>(q, A, Bv, Cv, one, half, Side::Left, L); });
            out.emplace_back("syrk", [=, this] { syrk<B, T>(q, A, Cv, one, half, L, N); });
            out.emplace_back("syr2k", [=, this] { syr2k<B, T>(q, A, Bv, Cv, one, half, L, N); });
        }
        return out;
    }
    std::vector<std::pair<std::string, std::function<std::size_t()>>> sizes() {
        using namespace batchlas;
        const auto A = view(a, false), Bv = view(b, false), Cv = view(c, false);
        const auto N = Transpose::NoTrans;
        return {
            {"potrf", [=, this] { return potrf_buffer_size<B, T>(q, A, Uplo::Lower); }},
            {"getrf", [=, this] { return getrf_buffer_size<B, T>(q, A); }},
            {"getrs", [=, this] { return getrs_buffer_size<B, T>(q, A, Bv, N); }},
            {"getri", [=, this] { return getri_buffer_size<B, T>(q, A); }},
            {"geqrf", [=, this] { return geqrf_buffer_size<B, T>(q, A, tspan()); }},
            {"orgqr", [=, this] { return orgqr_buffer_size<B, T>(q, A, tspan()); }},
            {"ormqr", [=, this] { return ormqr_buffer_size<B, T>(q, A, Cv, Side::Left, N, tspan()); }},
            {"syev", [=, this] { return syev_buffer_size<B, T>(q, A, wspan(), JobType::EigenVectors, Uplo::Lower); }},
            {"gesvd", [=, this] { return gesvd_buffer_size<B, T>(q, A, wspan(), Bv, Cv, SvdVectors::All, SvdVectors::All); }},
            {"gesv", [=, this] { return gesv_buffer_size<B, T>(q, A, Bv); }},
            {"posv", [=, this] { return posv_buffer_size<B, T>(q, A, Bv, Uplo::Lower); }},
            {"spmm", [=, this] { return spmm_buffer_size<B, T, MatrixFormat::CSR>(q, csr(), Bv, Cv, T(1), T(0.5), N, N); }},
        };
    }
    template <class U>
    static int changed(const batchlas::UnifiedVector<U>& v, U s) {
        return int(std::count_if(v.data(), v.data() + v.size(), [&](const U& x) { return !(x == s); }));
    }
    int touched() const {
        const T s = sentinel();
        return changed(a, s) + changed(b, s) + changed(c, s) + changed(vx, s) + changed(vy, s) + changed(vals, s) +
               changed(tau, s) + changed(w, RealOf<T>(-999)) + changed(piv, std::int64_t(-999)) +
               changed(info, std::int32_t(-999));
    }
};

// Batch 0 through every public entry point and every *_buffer_size, with a 0-byte workspace:
// no throw (vendor-free too: an empty call never reaches selection), 0 bytes, nothing written,
// and no coverage `miss` row.
TYPED_TEST(BatchPolicy, EmptyBatchIsANoOpForEveryOp) {
    using T = typename TestFixture::ScalarType;
    constexpr Backend B = TestFixture::BackendType;
    OpBattery<T, B> ops(*this->ctx, 0, {});
    for (const auto& [name, size] : ops.sizes()) {
        try {
            EXPECT_EQ(size(), 0u) << name << "_buffer_size";
        } catch (const std::exception& e) {
            ADD_FAILURE() << name << "_buffer_size threw on an empty batch: " << e.what();
        }
    }
    for (const auto& [name, call] : ops.calls(false)) {
        try {
            call();
        } catch (const std::exception& e) {
            ADD_FAILURE() << name << " threw on an empty batch: " << e.what();
        }
    }
    this->ctx->wait();
    EXPECT_EQ(ops.touched(), 0);

    // threadsafe: the child re-executes the binary, so the SYCL runtime starts fresh.
    ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
    const auto dir = std::filesystem::temp_directory_path() /
                     ("select_tests.empty." + std::string(sel::dtype_name<T>()) + "." + std::to_string(int(B)));
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    const std::string out = (dir / "cov").string();
    auto child = [&] {
        ScopedEnvVar cov("BATCHLAS_COVERAGE_OUT", out.c_str());
        batchlas::coverage::g_dynamic_enabled = true;
        for (const auto& [name, call] : ops.calls(false)) {
            try {
                call();
            } catch (...) {
            }
        }
        this->ctx->wait();
        std::exit(0);
    };
    EXPECT_EXIT(child(), ::testing::ExitedWithCode(0), "");
    int lines = 0;
    for (const auto& ent : std::filesystem::directory_iterator(dir)) {
        std::ifstream in(ent.path());
        for (std::string line; std::getline(in, line); ++lines)
            EXPECT_NE(line.rfind("miss,", 0), 0u) << "an empty call recorded a routing miss: " << line;
    }
    EXPECT_GT(lines, 0) << "the coverage file was never written, so the miss check saw nothing";
    std::filesystem::remove_all(dir);
}

// gemm with k = 0 is C := beta C in every build (no family takes it; vendor-free it had no
// route): beta 0 writes zeros over a NaN, beta 1 writes nothing, and the padding is never touched.
TYPED_TEST(BatchPolicy, GemmWithZeroInnerExtentScalesC) {
    using T = typename TestFixture::ScalarType;
    constexpr Backend B = TestFixture::BackendType;
    constexpr int m = 7, n = 5, ldc = 9, sc = 9 * 5 + 4, batch = 3;
    const T poison = Dense<T>::make(RealOf<T>(-999), RealOf<T>(777));
    batchlas::UnifiedVector<T> a(64, poison), b(64, poison), c(std::size_t(sc) * batch);
    const Dense<T> A(a.data(), m, 0, m, m, batch), Bv(b.data(), 0, n, 1, 1, batch);
    const Dense<T> Cv(c.data(), m, n, ldc, sc, batch);
    for (const T beta : {T(0), T(1), Dense<T>::make(RealOf<T>(-0.5), RealOf<T>(0.25))}) {
        for (std::size_t e = 0; e < c.size(); ++e) c[e] = Dense<T>::make(RealOf<T>(e % 13) - 6, RealOf<T>(e % 5));
        c[std::size_t(sc) + 2 * ldc + 3] = T(std::numeric_limits<RealOf<T>>::quiet_NaN());
        const std::vector<T> c0(c.data(), c.data() + c.size());
        const auto call = [&] {
            batchlas::gemm<B, T>(*this->ctx, A, Bv, Cv, T(2), beta, batchlas::Transpose::NoTrans,
                                 batchlas::Transpose::NoTrans);
        };
        ASSERT_NO_THROW(call()) << "beta " << beta;
        this->ctx->wait();
        int wrong = 0;
        for (std::size_t e = 0; e < c.size(); ++e) {
            const std::size_t r = e % sc;
            const bool inside = e / sc < std::size_t(batch) && r % ldc < std::size_t(m) && r / ldc < std::size_t(n);
            const T want = !inside || beta == T(1) ? c0[e] : (beta == T(0) ? T(0) : c0[e] * beta);
            const bool same = std::memcmp(&c[e], &want, sizeof(T)) == 0;
            wrong += !same && !(want != want && c[e] != c[e]);
        }
        EXPECT_EQ(wrong, 0) << "beta " << beta;
    }
}

// A batch of 4 with active orders {n, n-3, n, n-5}: every family of these ops launches one
// (n, ld, stride) for the whole batch, so the call must find no route (hemm/herk/her2k, which
// have no selection: an argument error), never return a padded answer. gemm has a per-item loop.
// In a vendor build "no route" is select's "no runnable kernel"; vendor-free it is NoRouteError.
TYPED_TEST(BatchPolicy, HeterogeneousBatchNeverRunsAFamilyThatIgnoresIt) {
    using T = typename TestFixture::ScalarType;
    constexpr Backend B = TestFixture::BackendType;
    batchlas::UnifiedVector<std::byte> work(std::size_t(1) << 24);
    OpBattery<T, B> ops(*this->ctx, 4, {work.data(), work.size()});
    ASSERT_TRUE(ops.view(ops.a, true).is_heterogeneous());
    for (const auto& [name, call] : ops.calls(true)) {
        if (name == "gemm") continue;
        // No selection (hemm/herk/her2k) or a pre-selection refusal (gesv/posv): an argument error.
        const bool refused_upfront =
            name == "hemm" || name == "herk" || name == "her2k" || name == "gesv" || name == "posv";
        try {
            call();
            this->ctx->wait();
            ADD_FAILURE() << name << " ran a heterogeneous batch";
        } catch (const batchlas::NoRouteError&) {
            EXPECT_FALSE(refused_upfront) << name;
        } catch (const std::exception& e) {
            const std::string what = e.what();
            EXPECT_NE(what.find(refused_upfront ? "heterogeneous batch" : "no runnable kernel"), std::string::npos)
                << name << ": " << what;
            EXPECT_TRUE(refused_upfront || dynamic_cast<const batchlas::unsupported*>(&e)) << name << ": " << what;
        }
    }
}

}  // namespace
