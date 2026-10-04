// Every table embedded from tuned/*.txt (docs/design/flat-kernel-selection.md §8.4). No GPU.

#include <gtest/gtest.h>

#include <batchlas/util/env.hh>

#include "../src/ops/potrf/choice.hh"
#include "../src/select/select.hh"

#include <algorithm>
#include <complex>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace sel = batchlas::select;
using batchlas::ScopedEnvVar;

namespace {

template <class Array>
std::vector<std::string> spellings(const Array& cands) {
    std::vector<std::string> out;
    for (const auto& c : cands) out.push_back(sel::to_string(c));
    return out;
}

// op -> dtype -> candidate spellings in candidate-list (tie-break) order, from the op's own
// choice.hh. Every migrated op adds its entry here.
std::vector<std::string> candidates(const std::string& op, const std::string& dtype) {
    namespace potrf = batchlas::ops::potrf;
    if (op == "potrf") {
        if (dtype == "float") return spellings(potrf::candidates<float>());
        if (dtype == "double") return spellings(potrf::candidates<double>());
        if (dtype == "cfloat") return spellings(potrf::candidates<std::complex<float>>());
        if (dtype == "cdouble") return spellings(potrf::candidates<std::complex<double>>());
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

// choice.hh's key_names is the spec every potrf table declares, weights included.
TEST(TunedTables, PotrfTablesDeclareChoiceKeyNames) {
    std::string want = "# keys:";
    for (auto k : batchlas::ops::potrf::key_names) want += " " + std::string(k);
    int seen = 0;
    for (const auto& e : sel::embedded_tables()) {
        if (std::string_view(e.name).rfind("potrf.", 0) != 0) continue;
        EXPECT_NE(std::string(e.text).find("\n" + want + "\n"), std::string::npos) << e.name;
        ++seen;
    }
    EXPECT_GT(seen, 0);
}

const sel::Table& embedded(const std::string& name) {
    static std::map<std::string, sel::Table> cache;
    auto it = cache.find(name);
    if (it == cache.end())
        for (const auto& e : sel::embedded_tables())
            if (e.name == name) it = cache.emplace(name, sel::parse_table(e.text, e.name)).first;
    if (it == cache.end()) throw std::runtime_error(name + " is not embedded");
    return it->second;
}

// The sparse sm_89 tables (final-review finding): with equal weights, float n=24 batch=512
// fell to the n=80 batch=2048 row (vendor; the native tiers are ~2.2x faster there) and
// double n=3 batch=128 to n=256 batch=256. n:log:3 keeps both on their own small-n rows.
TEST(TunedTables, Sm89SmallOrdersStayOnSmallOrderRows) {
    const sel::Key f{{"uplo", "L"}, {"n", 24}, {"batch", 512}};
    const auto* r = embedded("potrf.float.sm_89.txt").nearest(f);
    ASSERT_NE(r, nullptr);
    EXPECT_EQ(r->keys[1], "24") << "line " << r->line;
    EXPECT_EQ(r->ranked.front().spelling, "cta") << "line " << r->line;
    const sel::Key d{{"uplo", "L"}, {"n", 3}, {"batch", 128}};
    r = embedded("potrf.double.sm_89.txt").nearest(d);
    ASSERT_NE(r, nullptr);
    EXPECT_LE(std::stoi(r->keys[1]), 4) << "line " << r->line;
    EXPECT_EQ(r->ranked.front().spelling, "tiny") << "line " << r->line;
    // The same through choose(): what an sm_89 device runs with every candidate runnable.
    ScopedEnvVar dir("BATCHLAS_TUNED_DIR", nullptr);
    ScopedEnvVar pin("BATCHLAS_POTRF_ROUTE", nullptr);
    sel::testing::use_embedded_tables();
    EXPECT_EQ(sel::to_string(sel::choose("potrf", "float", sel::device_from_key("sm_89"), f,
                                         batchlas::ops::potrf::candidates<float>(),
                                         [](const batchlas::ops::potrf::PotrfChoice&) { return true; },
                                         batchlas::ops::potrf::rules)),
              "cta");
}

// ---- the shipped tables through choose() ------------------------------------------------

using C = batchlas::ops::potrf::PotrfChoice;

std::string pick(const std::string& device, const char* uplo, std::int64_t n, std::int64_t batch) {
    const sel::Key k{{"uplo", uplo}, {"n", n}, {"batch", batch}};
    return sel::to_string(sel::choose("potrf", "float", sel::device_from_key(device), k,
                                      batchlas::ops::potrf::candidates<float>(),
                                      [](const C&) { return true; }, batchlas::ops::potrf::rules));
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
