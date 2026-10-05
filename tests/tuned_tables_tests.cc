// Every table embedded from tuned/*.txt (docs/design/flat-kernel-selection.md §8.4). No GPU.

#include <gtest/gtest.h>

#include <batchlas/util/env.hh>

#include "../src/ops/posv/choice.hh"
#include "../src/ops/potrf/choice.hh"
#include "../src/ops/syev/choice.hh"
#include "../src/ops/trsm/choice.hh"
#include "../src/select/select.hh"

#include <algorithm>
#include <complex>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
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
    namespace posv = batchlas::ops::posv;
    if (op == "posv") {
        if (dtype == "float") return spellings(posv::candidates<float>());
        if (dtype == "double") return spellings(posv::candidates<double>());
        if (dtype == "cfloat") return spellings(posv::candidates<std::complex<float>>());
        if (dtype == "cdouble") return spellings(posv::candidates<std::complex<double>>());
    }
    namespace trsm = batchlas::ops::trsm;
    if (op == "trsm") {
        if (dtype == "float") return spellings(trsm::candidates<float>());
        if (dtype == "double") return spellings(trsm::candidates<double>());
        if (dtype == "cfloat") return spellings(trsm::candidates<std::complex<float>>());
        if (dtype == "cdouble") return spellings(trsm::candidates<std::complex<double>>());
    }
    namespace syev = batchlas::ops::syev;
    if (op == "syev") {
        if (dtype == "float") return spellings(syev::candidates<float>());
        if (dtype == "double") return spellings(syev::candidates<double>());
        if (dtype == "cfloat") return spellings(syev::candidates<std::complex<float>>());
        if (dtype == "cdouble") return spellings(syev::candidates<std::complex<double>>());
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

// Every ranked spelling, timed or transcribed, must be a long-form candidate of the op.
std::vector<std::string> spelling_problems(const sel::Table& t) {
    std::vector<std::string> out;
    const auto cands = candidates(t.op, t.dtype);
    if (cands.empty()) return {"op " + t.op + " has no candidate list in this test's registry"};
    for (const auto& row : t.rows)
        for (const auto& entry : row.ranked)
            if (std::find(cands.begin(), cands.end(), entry.spelling) == cands.end())
                out.push_back("line " + std::to_string(row.line) + ": " + entry.spelling + " is not a " + t.op +
                              " " + t.dtype + " candidate (spellings must be the long form)");
    return out;
}

// §6.3: entries within 3% of the best lead, in candidate-list order; the rest ascend in time.
// A transcribed row has no times: its order is the old router's, so it is not checked here.
std::vector<std::string> tie_rule_problems(const sel::Table& t) {
    std::vector<std::string> out;
    const auto cands = candidates(t.op, t.dtype);
    auto pos = [&](const std::string& s) { return std::find(cands.begin(), cands.end(), s) - cands.begin(); };
    for (const auto& row : t.rows) {
        if (!row.timed) continue;
        const std::string at = "line " + std::to_string(row.line) + ": ";
        double best = row.ranked[0].ms;
        for (const auto& x : row.ranked) best = std::min(best, x.ms);
        const double band = best * (1.0 + kTie);
        std::size_t i = 0;
        for (; i < row.ranked.size() && row.ranked[i].ms <= band * (1.0 + kPrintSlack); ++i)
            if (i > 0 && pos(row.ranked[i - 1].spelling) >= pos(row.ranked[i].spelling))
                out.push_back(at + "tied entries out of candidate order");
        for (std::size_t j = i; j < row.ranked.size(); ++j) {
            if (row.ranked[j].ms <= band * (1.0 - kPrintSlack))
                out.push_back(at + row.ranked[j].spelling + " is tied but ranked late");
            if (j > i && row.ranked[j].ms < row.ranked[j - 1].ms)
                out.push_back(at + "times not ascending after the tie band");
        }
    }
    return out;
}

std::string joined(const std::vector<std::string>& v) {
    std::string s;
    for (const auto& x : v) s += x + "\n";
    return s;
}

TEST(TunedTables, EveryEmbeddedTableParsesAndNamesOnlyCandidates) {
    ASSERT_FALSE(sel::embedded_tables().empty());
    for (const auto& e : sel::embedded_tables()) {
        SCOPED_TRACE(std::string(e.name));
        sel::Table t;
        ASSERT_NO_THROW(t = sel::parse_table(e.text, e.name));
        EXPECT_FALSE(t.rows.empty());
        const auto p = spelling_problems(t);
        EXPECT_TRUE(p.empty()) << joined(p);
    }
}

TEST(TunedTables, RankedTimesFollowTheTieRule) {
    ASSERT_FALSE(sel::embedded_tables().empty());
    for (const auto& e : sel::embedded_tables()) {
        SCOPED_TRACE(std::string(e.name));
        const auto p = tie_rule_problems(sel::parse_table(e.text, e.name));
        EXPECT_TRUE(p.empty()) << joined(p);
    }
}

// The two checks above on a synthetic potrf table: a transcribed row out of candidate order
// is exempt from the tie rule, but its spellings are still checked; timed rows still are.
TEST(TunedTables, TranscribedRowsSkipTheTieRuleButNotTheSpellingCheck) {
    const std::string keys = "# keys: uplo:exact n:log:3 batch:log\n";
    const std::string tr = "# op=potrf dtype=float device=sm_89 source=transcribed:2b46acab\n" + keys;
    const std::string sw = "# op=potrf dtype=float device=sm_89 source=sweep.jsonl\n" + keys;
    const auto good = sel::parse_table(tr + "uplo=L n=8 batch=8192 | vendor - | blocked - | tiny -\n"
                                            "uplo=L n=64 batch=8192 | blocked - | vendor -\n",
                                       "potrf.float.sm_89.txt");
    EXPECT_FALSE(good.rows[0].timed);
    EXPECT_TRUE(tie_rule_problems(good).empty()) << joined(tie_rule_problems(good));
    EXPECT_TRUE(spelling_problems(good).empty()) << joined(spelling_problems(good));
    const auto bad = sel::parse_table(tr + "uplo=L n=8 batch=8192 | vendor - | lpanel:8 -\n"
                                           "uplo=L n=64 batch=8192 | vendor - | blocked -\n",
                                      "potrf.float.sm_89.txt");
    EXPECT_EQ(spelling_problems(bad), std::vector<std::string>{"line 3: lpanel:8 is not a potrf float candidate "
                                                               "(spellings must be the long form)"});
    EXPECT_TRUE(tie_rule_problems(bad).empty()) << joined(tie_rule_problems(bad));
    const auto timed = sel::parse_table(sw + "uplo=L n=64 batch=8192 | vendor 2.0 | blocked 1.0\n",
                                        "potrf.float.sm_89.txt");
    EXPECT_EQ(tie_rule_problems(timed), (std::vector<std::string>{"line 3: blocked is tied but ranked late",
                                                                  "line 3: times not ascending after the tie band"}));
}

// choice.hh's key_names is the spec every table of that op declares, weights included.
template <class Names>
void expect_tables_declare(const std::string& op, const Names& names) {
    std::string want = "# keys:";
    for (auto k : names) want += " " + std::string(k);
    int seen = 0;
    for (const auto& e : sel::embedded_tables()) {
        if (std::string_view(e.name).rfind(op + ".", 0) != 0) continue;
        EXPECT_NE(std::string(e.text).find("\n" + want + "\n"), std::string::npos) << e.name;
        ++seen;
    }
    EXPECT_GT(seen, 0) << op;
}

TEST(TunedTables, PotrfTablesDeclareChoiceKeyNames) { expect_tables_declare("potrf", batchlas::ops::potrf::key_names); }
TEST(TunedTables, PosvTablesDeclareChoiceKeyNames) { expect_tables_declare("posv", batchlas::ops::posv::key_names); }
TEST(TunedTables, TrsmTablesDeclareChoiceKeyNames) { expect_tables_declare("trsm", batchlas::ops::trsm::key_names); }
TEST(TunedTables, SyevTablesDeclareChoiceKeyNames) { expect_tables_declare("syev", batchlas::ops::syev::key_names); }

const sel::Table& embedded(const std::string& name) {
    static std::map<std::string, sel::Table> cache;
    auto it = cache.find(name);
    if (it == cache.end())
        for (const auto& e : sel::embedded_tables())
            if (e.name == name) it = cache.emplace(name, sel::parse_table(e.text, e.name)).first;
    if (it == cache.end()) throw std::runtime_error(name + " is not embedded");
    return it->second;
}

// The transcriber spells choice.hh's grid by hand (it builds against the deleted router), so
// each transcribed sm_89 posv table must hold exactly one row per grid cell, both triangles.
TEST(TunedTables, PosvSm89TablesHoldExactlyTheChoiceGrid) {
    namespace posv = batchlas::ops::posv;
    std::set<std::string> want;
    for (const char* u : {"L", "U"})
        for (int n : posv::grid_n)
            for (int r : posv::grid_nrhs)
                for (int b : posv::grid_batch)
                    want.insert(std::string(u) + " " + std::to_string(n) + " " + std::to_string(r) + " " +
                                std::to_string(b));
    for (const char* dt : {"float", "double", "cfloat", "cdouble"}) {
        const sel::Table& t = embedded(std::string("posv.") + dt + ".sm_89.txt");
        std::set<std::string> got;
        for (const auto& row : t.rows)
            got.insert(row.keys[0] + " " + row.keys[1] + " " + row.keys[2] + " " + row.keys[3]);
        EXPECT_EQ(got, want) << dt;
        EXPECT_EQ(t.rows.size(), want.size()) << dt;
    }
}

// Likewise trsm's transcriber: one row per (side, trans, order, q, batch) cell of choice.hh.
TEST(TunedTables, TrsmSm89TablesHoldExactlyTheChoiceGrid) {
    namespace trsm = batchlas::ops::trsm;
    std::set<std::string> want;
    for (const char* s : {"L", "R"})
        for (const char* tr : {"N", "T"})
            for (int o : trsm::grid_order)
                for (int q : trsm::grid_q)
                    for (int b : trsm::grid_batch)
                        want.insert(std::string(s) + " " + tr + " " + std::to_string(o) + " " + std::to_string(q) +
                                    " " + std::to_string(b));
    for (const char* dt : {"float", "double", "cfloat", "cdouble"}) {
        const sel::Table& t = embedded(std::string("trsm.") + dt + ".sm_89.txt");
        std::set<std::string> got;
        for (const auto& row : t.rows)
            got.insert(row.keys[0] + " " + row.keys[1] + " " + row.keys[2] + " " + row.keys[3] + " " + row.keys[4]);
        EXPECT_EQ(got, want) << dt;
        EXPECT_EQ(t.rows.size(), want.size()) << dt;
    }
}

// syev's transcriber spells choice.hh's grid by hand too; one transcription serves both devices.
TEST(TunedTables, SyevTablesHoldExactlyTheChoiceGridOnBothDevices) {
    namespace syev = batchlas::ops::syev;
    std::set<std::string> want;
    for (const char* j : {"N", "V"})
        for (int n : syev::grid_n)
            for (int b : syev::grid_batch) want.insert(std::string(j) + " " + std::to_string(n) + " " + std::to_string(b));
    for (const char* dev : {"sm_89", "sm_120"})
        for (const char* dt : {"float", "double", "cfloat", "cdouble"}) {
            const sel::Table& t = embedded(std::string("syev.") + dt + "." + dev + ".txt");
            std::set<std::string> got;
            for (const auto& row : t.rows) got.insert(row.keys[0] + " " + row.keys[1] + " " + row.keys[2]);
            EXPECT_EQ(got, want) << dt << " " << dev;
            EXPECT_EQ(t.rows.size(), want.size()) << dt << " " << dev;
        }
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
