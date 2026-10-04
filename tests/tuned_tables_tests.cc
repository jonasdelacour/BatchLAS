// Every table embedded from tuned/*.txt (docs/design/flat-kernel-selection.md §8.4). No GPU.

#include <gtest/gtest.h>

#include <batchlas/util/env.hh>

#include "../src/select/select.hh"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <variant>
#include <vector>

namespace sel = batchlas::select;
using batchlas::ScopedEnvVar;

namespace {

// op -> dtype -> candidate spellings in candidate-list (tie-break) order. Phase 2 replaces
// potrf's literal list with to_string over ops::potrf::candidates<T>(); every migrated op
// adds its own entry here.
std::vector<std::string> candidates(const std::string& op, const std::string& dtype) {
    if (op == "potrf") {
        if (dtype == "float") return {"tiny", "cta", "lpanel:panel=8", "lpanel:panel=16", "blocked", "vendor"};
        return {"tiny", "cta", "lpanel:panel=8", "blocked", "vendor"};
    }
    return {};
}

constexpr double kTie = 0.03;
constexpr double kPrintSlack = 1e-3;  // tables print 4 significant digits

// No skip on an empty set: a broken embed step must fail here, not pass vacuously.
TEST(TunedTables, EmbeddedSetIsExactlyTunedDirByNameAndBytes) {
    std::map<std::string, std::string> want;
    for (const auto& ent : std::filesystem::directory_iterator(BATCHLAS_TUNED_SOURCE_DIR)) {
        if (ent.path().extension() != ".txt") continue;
        std::ifstream in(ent.path(), std::ios::binary);
        std::stringstream buf;
        buf << in.rdbuf();
        want[ent.path().filename().string()] = buf.str();
    }
    ASSERT_FALSE(want.empty()) << BATCHLAS_TUNED_SOURCE_DIR << " has no *.txt";
    std::map<std::string, std::string> got;
    for (const auto& e : sel::embedded_tables()) got[std::string(e.name)] = std::string(e.text);
    ASSERT_EQ(got.size(), want.size());
    for (const auto& [name, text] : want) {
        ASSERT_TRUE(got.count(name)) << name << " is not embedded";
        EXPECT_EQ(got[name], text) << name << " embedded text differs from the file";
    }
}

TEST(TunedTables, EveryEmbeddedTableParsesAndNamesOnlyCandidates) {
    ASSERT_FALSE(sel::embedded_tables().empty());
    for (const auto& e : sel::embedded_tables()) {
        SCOPED_TRACE(std::string(e.name));
        sel::Table t;
        ASSERT_NO_THROW(t = sel::parse_table(e.text, e.name));
        const auto cands = candidates(t.op, t.dtype);
        ASSERT_FALSE(cands.empty()) << "op " << t.op << " has no candidate list in this test's registry";
        EXPECT_FALSE(t.rows.empty());
        for (const auto& row : t.rows)
            for (const auto& entry : row.ranked)
                EXPECT_NE(std::find(cands.begin(), cands.end(), entry.spelling), cands.end())
                    << "line " << row.line << ": " << entry.spelling << " is not a " << t.op << " " << t.dtype
                    << " candidate (spellings must be the long form)";
    }
}

// §6.3: entries within 3% of the best lead, in candidate-list order; the rest ascend in time.
TEST(TunedTables, RankedTimesFollowTheTieRule) {
    ASSERT_FALSE(sel::embedded_tables().empty());
    for (const auto& e : sel::embedded_tables()) {
        SCOPED_TRACE(std::string(e.name));
        const auto t = sel::parse_table(e.text, e.name);
        const auto cands = candidates(t.op, t.dtype);
        auto pos = [&](const std::string& s) { return std::find(cands.begin(), cands.end(), s) - cands.begin(); };
        for (const auto& row : t.rows) {
            double best = row.ranked[0].ms;
            for (const auto& x : row.ranked) best = std::min(best, x.ms);
            const double band = best * (1.0 + kTie);
            std::size_t i = 0;
            for (; i < row.ranked.size() && row.ranked[i].ms <= band * (1.0 + kPrintSlack); ++i)
                if (i > 0) EXPECT_LT(pos(row.ranked[i - 1].spelling), pos(row.ranked[i].spelling))
                               << "line " << row.line << ": tied entries out of candidate order";
            for (std::size_t j = i; j < row.ranked.size(); ++j) {
                EXPECT_GT(row.ranked[j].ms, band * (1.0 - kPrintSlack))
                    << "line " << row.line << ": " << row.ranked[j].spelling << " is tied but ranked late";
                if (j > i) EXPECT_GE(row.ranked[j].ms, row.ranked[j - 1].ms)
                               << "line " << row.line << ": times not ascending after the tie band";
            }
        }
    }
}

// ---- the shipped tables through choose() ------------------------------------------------

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
using C = std::variant<Tiny, Cta, Lpanel, Blocked, Vendor>;
const std::array<C, 6> kFloat{Tiny{}, Cta{}, Lpanel{8}, Lpanel{16}, Blocked{}, Vendor{}};
constexpr std::array<std::string_view, 2> kLastResort{"blocked", "vendor"};

std::string pick(const std::string& device, const char* uplo, std::int64_t n, std::int64_t batch) {
    const sel::Key k{{"uplo", uplo}, {"n", n}, {"batch", batch}};
    return sel::to_string(sel::choose("potrf", "float", sel::device_from_key(device), k, kFloat,
                                      [](const C&) { return true; }, sel::Rules{{}, kLastResort}));
}

TEST(TunedTables, ShippedPotrfRowsAreWhatChooseReturns) {
    ScopedEnvVar dir("BATCHLAS_TUNED_DIR", nullptr);
    ScopedEnvVar pin("BATCHLAS_POTRF_ROUTE", nullptr);
    sel::testing::use_embedded_tables();
    sel::testing::reset_warnings();
    const auto order = sel::tables_in_borrow_order("potrf", "float", sel::device_from_key("sm_120"));
    ASSERT_FALSE(order.empty());
    EXPECT_EQ(order.front()->file, "potrf.float.sm_120.txt");
    EXPECT_FALSE(order.front()->is_override);
    ::testing::internal::CaptureStderr();
    EXPECT_EQ(pick("sm_120", "L", 64, 8192), "lpanel:panel=8");
    EXPECT_EQ(pick("sm_120", "L", 512, 2048), "blocked");  // tied with vendor: candidate order wins
    EXPECT_EQ(pick("sm_120", "U", 64, 8192), "cta");
    // §3/§7.6: no cpu table ships, so the CPU takes the last resort and never borrows.
    EXPECT_EQ(pick("cpu", "L", 64, 8192), "blocked");
    EXPECT_EQ(::testing::internal::GetCapturedStderr(), "");
}

}  // namespace
