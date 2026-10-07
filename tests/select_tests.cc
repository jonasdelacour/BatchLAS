// src/select/select.hh on a synthetic choice variant and synthetic tables: no GPU, no kernel.
// docs/design/flat-kernel-selection.md §5 and §8.5.

#include <gtest/gtest.h>

#include <batchlas/settings.hh>
#include <batchlas/util/env.hh>

#include "../src/select/select.hh"

#include <filesystem>
#include <fstream>
#include <functional>
#include <set>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <variant>

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
std::pair<std::string, std::string> file(const std::string& device, const std::string& rows,
                                         const std::string& op = "synth") {
    return {op + ".float." + device + ".txt", table(device, rows, op)};
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
        "batch=2048 n=64 uplo=L | cta 1\n"), "synth.float.sm_120.txt");
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
    const auto t = sel::parse_table("# keys: uplo:exact n:log:3 batch:log m:log:0.5\nuplo=L n=4 batch=1 m=2 | tiny 1\n",
                                    "synth.float.sm_120.txt");
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
        "uplo=L n=64 batch=8192 | blocked - | cta -\n", "synth.float.sm_89.txt");
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
                                        "uplo=L n=64 batch=8192 | blocked 1.5 | cta 2\n", "synth.float.sm_89.txt");
    EXPECT_TRUE(timed.rows[0].timed);
    EXPECT_DOUBLE_EQ(timed.rows[0].ranked[0].ms, 1.5);
    // A transcribed table may mix row kinds (each row is one kind); the Python --check agrees.
    const auto mixed = sel::parse_table("# op=synth dtype=float device=sm_89 source=transcribed:2b46acab\n"
                                        "# keys: uplo:exact n:log batch:log\n"
                                        "uplo=L n=20 batch=8192 | tiny - | cta -\n"
                                        "uplo=L n=64 batch=8192 | blocked 1.5 | cta 2\n", "synth.float.sm_89.txt");
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
                                    "uplo=L n=64 batch=8192 | blocked 1.5 | cta 2 # deep\n", "synth.float.sm_89.txt");
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
                                  "synth.float.sm_89.txt"), std::runtime_error);
}

TEST(SelectTable, DeviceFromFileNameWhenHeaderOmitsIt) {
    const auto t = sel::parse_table("# keys: uplo:exact n:log\nuplo=L n=4 | tiny 1\n", "dir/op.cfloat.gfx90a.txt");
    EXPECT_EQ(t.file, "op.cfloat.gfx90a.txt");
    EXPECT_EQ(t.dtype, "cfloat");
    EXPECT_EQ(t.family, "gfx");
    EXPECT_EQ(t.arch_number, 90);
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
            sel::parse_table(text, "synth.float.sm_120.txt");
            ADD_FAILURE() << "accepted:\n" << text;
        } catch (const std::runtime_error& e) {
            const std::string m = e.what();
            EXPECT_EQ(m.find("synth.float.sm_120.txt" + where), 0u) << m;
            EXPECT_NE(m.find(why), std::string::npos) << m;
        }
    }
    EXPECT_THROW(sel::parse_table(keys, "untitled"), std::runtime_error);  // no op/dtype/device at all
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
    static const auto t = sel::parse_table(table("sm_120", kGrid), "synth.float.sm_120.txt");
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
        "uplo=L n=32 batch=4096 | vendor 1\n"), "synth.float.sm_120.txt");
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
    const auto plain = sel::parse_table(head + "# keys: uplo:exact n:log batch:log\n" + rows, "synth.float.sm_120.txt");
    const auto cubed = sel::parse_table(head + "# keys: uplo:exact n:log:3 batch:log\n" + rows, "synth.float.sm_120.txt");
    const auto light = sel::parse_table(head + "# keys: uplo:exact n:log batch:log:0.4\n" + rows, "synth.float.sm_120.txt");
    EXPECT_EQ(plain.nearest(key("L", 32, 1024))->line, 3);
    EXPECT_EQ(cubed.nearest(key("L", 32, 1024))->line, 4);
    EXPECT_EQ(light.nearest(key("L", 32, 1024))->line, 4);  // 0.4 * 2 < 1: a fractional weight
    // A weighted tie (3*1 == 3) still goes to the smaller n.
    const auto tie = sel::parse_table(head + "# keys: uplo:exact n:log:3 batch:log\n"
                                      "uplo=L n=32 batch=8192 | cta 1\nuplo=L n=16 batch=1024 | tiny 1\n",
                                      "synth.float.sm_120.txt");
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
                                     "synth.float.sm_120.txt");
    EXPECT_EQ(st.nearest(k("L", "T", 512))->line, 4);   // both exact keys match
    EXPECT_EQ(st.nearest(k("R", "T", 32))->line, 6);    // trans dropped: side=R kept, 8 beats 512
    EXPECT_EQ(st.nearest(k("R", "T", 300))->line, 5);   // ... and the log distance still decides
    EXPECT_EQ(st.nearest(k("R", "Z", 32))->line, 6);
    EXPECT_EQ(st.nearest(k("X", "T", 64))->line, 3);    // side unmatched: both dropped, pure distance
    // Declared order decides which key is kept: trans first keeps trans=T over side.
    const auto ts = sel::parse_table(head + "# keys: trans:exact n:log side:exact batch:log\n" + rows,
                                     "synth.float.sm_120.txt");
    EXPECT_EQ(ts.nearest(k("R", "T", 8))->line, 4);
    EXPECT_EQ(ts.nearest(k("R", "N", 8))->line, 6);
    // exact = {0, 2} is not {0, 1}: comparing by prefix position j instead of exact[j] never
    // matches side, drops it, and lets the nearer (L,N,64) row in.
    EXPECT_EQ(ts.nearest(k("R", "N", 32))->line, 6);
}

TEST(SelectNearest, MissingKeyThrows) {
    const auto t = sel::parse_table(table("sm_120", kGrid), "synth.float.sm_120.txt");
    EXPECT_THROW(t.nearest({{"uplo", "L"}, {"n", 64}}), std::invalid_argument);
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
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("synth.float.sm_120.txt:4:"), std::string::npos) << e.what();
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
    } catch (const std::runtime_error& e) {
        EXPECT_STREQ(e.what(), "synth: no runnable kernel on sm_120");
    }
}

// ---- pins -------------------------------------------------------------------------------

const char* kPinRow = "uplo=L n=64 batch=1024 | vendor 0.2 | lpanel:8 0.3 | cta 0.6\n";

std::string pin_error(const std::string& pin, const Pred& ok = kAll) {
    sel::ScopedPin<C> p("synth", pin);
    try {
        choose("sm_120", key("L", 64, 1024), ok);
    } catch (const std::invalid_argument& e) {
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
    // There are no aliases: the old router's origin:algorithm spellings are not choices.
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
        } catch (const std::invalid_argument& e) {
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
        } catch (const std::invalid_argument& e) {
            EXPECT_NE(std::string(e.what()).find("BATCHLAS_POTRF_ROUTE=\"regsiter_tiled\""), std::string::npos);
        }
    }
    EXPECT_EQ(pick(), "vendor");
}

// ---- BATCHLAS_TUNED_DIR -----------------------------------------------------------------

struct TempDir {
    std::filesystem::path path;
    TempDir() {
        path = std::filesystem::temp_directory_path() / ("select_tests." + std::to_string(::getpid()));
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
        EXPECT_NE(err.find(std::string("BATCHLAS_TUNED_DIR: skipping ") + junk + " (not <op>.<dtype>.<device>.txt)\n"),
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
    sel::testing::set_builtin_tables({{"synth.float.sm_89.txt",
        "# op=synth dtype=float device=sm_89 source=transcribed:2b46acab\n"
        "# keys: uplo:exact n:log batch:log\n"
        "uplo=L n=20 batch=8192 | tiny - | cta - | blocked -\n"
        "uplo=L n=512 batch=8192 | blocked - | vendor -\n"}});
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
    constexpr auto B = batchlas::Backend::CUDA;
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
TEST(SelectCoverageDeathTest, RowCarriesScalarBackendAndUploWithTraceOff) {
    TempDir dir;
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
