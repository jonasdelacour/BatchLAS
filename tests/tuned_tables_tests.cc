// Every table embedded from tuned/*.txt (docs/design/flat-kernel-selection.md §8.4). No GPU.

#include <gtest/gtest.h>

#include <batchlas/util/env.hh>

#include "../src/ops/gemm/choice.hh"
#include "../src/ops/geqrf/choice.hh"
#include "../src/ops/orgqr/choice.hh"
#include "../src/ops/getrf/choice.hh"
#include "../src/ops/getrs/choice.hh"
#include "../src/ops/posv/choice.hh"
#include "../src/ops/potrf/choice.hh"
#include "../src/ops/trsm/choice.hh"
#include "../src/ops/gemv/choice.hh"
#include "../src/ops/ormqr/choice.hh"
#include "../src/ops/getri/choice.hh"
#include "../src/ops/gesv/choice.hh"
#include "../src/ops/gesvd/choice.hh"
#include "../src/ops/spmm/choice.hh"
#include "../src/ops/syev/choice.hh"
#include "../src/select/select.hh"

#include <algorithm>
#include <complex>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <tuple>
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
    namespace gemm = batchlas::ops::gemm;
    if (op == "gemm") {
        if (dtype == "float") return spellings(gemm::candidates<float>());
        if (dtype == "double") return spellings(gemm::candidates<double>());
        if (dtype == "cfloat") return spellings(gemm::candidates<std::complex<float>>());
        if (dtype == "cdouble") return spellings(gemm::candidates<std::complex<double>>());
    }
    namespace gemv = batchlas::ops::gemv;
    if (op == "gemv") {
        if (dtype == "float") return spellings(gemv::candidates<float>());
        if (dtype == "double") return spellings(gemv::candidates<double>());
        if (dtype == "cfloat") return spellings(gemv::candidates<std::complex<float>>());
        if (dtype == "cdouble") return spellings(gemv::candidates<std::complex<double>>());
    }
    namespace geqrf = batchlas::ops::geqrf;
    if (op == "geqrf") {
        if (dtype == "float") return spellings(geqrf::candidates<float>());
        if (dtype == "double") return spellings(geqrf::candidates<double>());
        if (dtype == "cfloat") return spellings(geqrf::candidates<std::complex<float>>());
        if (dtype == "cdouble") return spellings(geqrf::candidates<std::complex<double>>());
    }
    namespace orgqr = batchlas::ops::orgqr;
    if (op == "orgqr") {
        if (dtype == "float") return spellings(orgqr::candidates<float>());
        if (dtype == "double") return spellings(orgqr::candidates<double>());
        if (dtype == "cfloat") return spellings(orgqr::candidates<std::complex<float>>());
        if (dtype == "cdouble") return spellings(orgqr::candidates<std::complex<double>>());
    }
    namespace ormqr = batchlas::ops::ormqr;
    if (op == "ormqr") {
        if (dtype == "float") return spellings(ormqr::candidates<float>());
        if (dtype == "double") return spellings(ormqr::candidates<double>());
        if (dtype == "cfloat") return spellings(ormqr::candidates<std::complex<float>>());
        if (dtype == "cdouble") return spellings(ormqr::candidates<std::complex<double>>());
    }
    namespace getrf = batchlas::ops::getrf;
    if (op == "getrf") {
        if (dtype == "float") return spellings(getrf::candidates<float>());
        if (dtype == "double") return spellings(getrf::candidates<double>());
        if (dtype == "cfloat") return spellings(getrf::candidates<std::complex<float>>());
        if (dtype == "cdouble") return spellings(getrf::candidates<std::complex<double>>());
    }
    namespace getrs = batchlas::ops::getrs;
    if (op == "getrs") {
        if (dtype == "float") return spellings(getrs::candidates<float>());
        if (dtype == "double") return spellings(getrs::candidates<double>());
        if (dtype == "cfloat") return spellings(getrs::candidates<std::complex<float>>());
        if (dtype == "cdouble") return spellings(getrs::candidates<std::complex<double>>());
    }
    namespace getri = batchlas::ops::getri;
    if (op == "getri") {
        if (dtype == "float") return spellings(getri::candidates<float>());
        if (dtype == "double") return spellings(getri::candidates<double>());
        if (dtype == "cfloat") return spellings(getri::candidates<std::complex<float>>());
        if (dtype == "cdouble") return spellings(getri::candidates<std::complex<double>>());
    }
    namespace gesv = batchlas::ops::gesv;
    if (op == "gesv") {
        if (dtype == "float") return spellings(gesv::candidates<float>());
        if (dtype == "double") return spellings(gesv::candidates<double>());
        if (dtype == "cfloat") return spellings(gesv::candidates<std::complex<float>>());
        if (dtype == "cdouble") return spellings(gesv::candidates<std::complex<double>>());
    }
    namespace gesvd = batchlas::ops::gesvd;
    if (op == "gesvd") {
        if (dtype == "float") return spellings(gesvd::candidates<float>());
        if (dtype == "double") return spellings(gesvd::candidates<double>());
        if (dtype == "cfloat") return spellings(gesvd::candidates<std::complex<float>>());
        if (dtype == "cdouble") return spellings(gesvd::candidates<std::complex<double>>());
    }
    namespace spmm = batchlas::ops::spmm;
    if (op == "spmm") {
        if (dtype == "float") return spellings(spmm::candidates<float>());
        if (dtype == "double") return spellings(spmm::candidates<double>());
        if (dtype == "cfloat") return spellings(spmm::candidates<std::complex<float>>());
        if (dtype == "cdouble") return spellings(spmm::candidates<std::complex<double>>());
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
            if (i > 0 && pos(row.ranked[i - 1].spelling) >= pos(row.ranked[i].spelling)) {
                // Within print rounding of the band edge the converter may have ranked it untied.
                if (row.ranked[i].ms > band * (1.0 - kPrintSlack)) break;
                out.push_back(at + "tied entries out of candidate order");
            }
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
    // 19.89 / 19.30 prints as 1.0306 but may be <= 1.03 unrounded, so either order passes; a
    // clearly tied pair (19.40) out of candidate order still fails.
    const auto edge = sel::parse_table(sw + "uplo=L n=64 batch=8192 | blocked 19.30 | cta 19.89\n"
                                            "uplo=L n=128 batch=8192 | cta 19.30 | blocked 19.89\n"
                                            "uplo=L n=256 batch=8192 | blocked 19.30 | cta 19.40\n",
                                       "potrf.float.sm_89.txt");
    EXPECT_EQ(tie_rule_problems(edge), std::vector<std::string>{"line 5: tied entries out of candidate order"});
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
TEST(TunedTables, GemmTablesDeclareChoiceKeyNames) { expect_tables_declare("gemm", batchlas::ops::gemm::key_names); }
TEST(TunedTables, GemvTablesDeclareChoiceKeyNames) { expect_tables_declare("gemv", batchlas::ops::gemv::key_names); }
TEST(TunedTables, GeqrfTablesDeclareChoiceKeyNames) { expect_tables_declare("geqrf", batchlas::ops::geqrf::key_names); }
TEST(TunedTables, OrgqrTablesDeclareChoiceKeyNames) { expect_tables_declare("orgqr", batchlas::ops::orgqr::key_names); }
TEST(TunedTables, GetrfTablesDeclareChoiceKeyNames) { expect_tables_declare("getrf", batchlas::ops::getrf::key_names); }
TEST(TunedTables, GetrsTablesDeclareChoiceKeyNames) { expect_tables_declare("getrs", batchlas::ops::getrs::key_names); }

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

// Likewise trsm's transcriber: one row per (side, trans, order, q, batch) cell of choice.hh, on
// sm_89 and on sm_120 where no tuner table exists (complex), with identical rows on both.
TEST(TunedTables, TrsmTranscribedTablesHoldExactlyTheChoiceGrid) {
    namespace trsm = batchlas::ops::trsm;
    std::set<std::string> want;
    for (const char* s : {"L", "R"})
        for (const char* tr : {"N", "T"})
            for (int o : trsm::grid_order)
                for (int q : trsm::grid_q)
                    for (int b : trsm::grid_batch)
                        want.insert(std::string(s) + " " + tr + " " + std::to_string(o) + " " + std::to_string(q) +
                                    " " + std::to_string(b));
    std::map<std::string, std::map<std::string, std::string>> rows_89;
    for (const char* dev : {"sm_89", "sm_120"})
        for (const char* dt : {"float", "double", "cfloat", "cdouble"}) {
            if (std::string(dev) == "sm_120" && dt[0] != 'c') continue;  // tuner tables, below
            const sel::Table& t = embedded(std::string("trsm.") + dt + "." + dev + ".txt");
            EXPECT_EQ(t.source, "transcribed:8b9adeb3") << t.file;
            std::set<std::string> got;
            for (const auto& row : t.rows) {
                const std::string k =
                    row.keys[0] + " " + row.keys[1] + " " + row.keys[2] + " " + row.keys[3] + " " + row.keys[4];
                got.insert(k);
                std::string ranked;
                for (const auto& e : row.ranked) ranked += e.spelling + "|";
                if (std::string(dev) == "sm_89") rows_89[dt][k] = ranked;
                else EXPECT_EQ(ranked, rows_89[dt][k]) << dt << " " << k << ": sm_120 differs from sm_89";
            }
            EXPECT_EQ(got, want) << dt << " " << dev;
            EXPECT_EQ(t.rows.size(), want.size()) << dt << " " << dev;
        }
}

// Coverage of devices is explicit: every migrated op ships a table for every dtype on sm_89 and
// sm_120 (spmm also on the CPU), so no shipped device borrows. Which ones are measured and which
// are transcribed is tuned/README.md's inventory; the measured sm_120 ones are checked below.
TEST(TunedTables, EveryOpShipsATableForEveryDtypeOnEveryShippedDevice) {
    std::set<std::string> names;
    for (const auto& e : sel::embedded_tables()) names.insert(std::string(e.name));
    for (const char* op : {"potrf", "posv", "trsm", "gemm", "gemv", "geqrf", "orgqr", "ormqr", "getrf", "getrs",
                           "getri", "gesv", "gesvd", "spmm", "syev"})
        for (const char* dev : {"sm_89", "sm_120", "cpu"})
            for (const char* dt : {"float", "double", "cfloat", "cdouble"}) {
                if (std::string(dev) == "cpu" && std::string(op) != "spmm") continue;
                EXPECT_TRUE(names.count(std::string(op) + "." + dt + "." + dev + ".txt"))
                    << op << " " << dt << " " << dev;
            }
}

// The sm_120 tables that come from measurements: posv (converted seed sweep) and trsm float and
// double (tuner). Every row timed, and the source names the committed raw data.
TEST(TunedTables, Sm120MeasuredTablesAreTimedAndNameTheirRawData) {
    for (const char* dt : {"float", "double", "cfloat", "cdouble"}) {
        const sel::Table& t = embedded(std::string("posv.") + dt + ".sm_120.txt");
        EXPECT_EQ(t.source.rfind("benchmarks/results/routing/sm120_posv_sweep.jsonl", 0), 0u) << t.file;
        for (const auto& row : t.rows) EXPECT_TRUE(row.timed) << t.file << " line " << row.line;
    }
    for (const char* dt : {"float", "double"}) {
        const sel::Table& t = embedded(std::string("trsm.") + dt + ".sm_120.txt");
        EXPECT_EQ(t.source.rfind(std::string("tuner:benchmarks/results/tuning/trsm.") + dt + ".sm_120.jsonl", 0), 0u)
            << t.file;
        for (const auto& row : t.rows) EXPECT_TRUE(row.timed) << t.file << " line " << row.line;
    }
}

// Likewise gemm's transcriber: one row per demand-grid cell of choice.hh (plan §3): squares for
// every form and layout, panels (packed from m, n >= 128) and skinny shapes for the issued forms;
// plus the edge rows that bracket the old predicate below the grid (the transcriber's header):
// real batch {1, 63, 64}, double k {1, 2} per (form, layout, m, n), float NN extra squares and
// one-axis-off neighbours of the small squares.
// The sm_120 tables are the same transcription (the transcriber read no device fact): row for row.
TEST(TunedTables, GemmTranscribedTablesHoldExactlyTheChoiceGrid) {
    namespace gemm = batchlas::ops::gemm;
    for (const char* dt : {"float", "double", "cfloat", "cdouble"}) {
        const bool cplx = dt[0] == 'c';
        std::vector<std::string_view> forms, panel;
        if (cplx) {
            forms.assign(gemm::grid_complex_forms.begin(), gemm::grid_complex_forms.end());
            panel.assign(gemm::grid_complex_panel_forms.begin(), gemm::grid_complex_panel_forms.end());
        } else {
            forms.assign(gemm::grid_real_forms.begin(), gemm::grid_real_forms.end());
            panel.assign(gemm::grid_real_panel_forms.begin(), gemm::grid_real_panel_forms.end());
        }
        std::set<std::string> want;
        std::set<std::tuple<std::string, std::string, int, int, int>> cells;
        std::vector<int> batches(gemm::grid_batch.begin(), gemm::grid_batch.end());
        if (!cplx) batches.insert(batches.end(), {1, 63, 64});
        auto add = [&](std::string_view f, const char* layout, int m, int n, int k) {
            cells.insert({std::string(f), layout, m, n, k});
            for (int b : batches)
                want.insert(std::string(1, f[0]) + " " + f[1] + " " + layout + " " + std::to_string(m) + " " +
                            std::to_string(n) + " " + std::to_string(k) + " " + std::to_string(b));
        };
        for (std::string_view f : forms) {
            for (int s : gemm::grid_square) {
                add(f, "strided", s, s, s);
                add(f, "packed", s, s, s);
            }
            if (std::find(panel.begin(), panel.end(), f) == panel.end()) continue;
            for (int m : gemm::grid_panel_mn)
                for (int n : gemm::grid_panel_mn)
                    for (int k : gemm::grid_panel_k) {
                        add(f, "strided", m, n, k);
                        if (m >= gemm::grid_packed_panel_min && n >= gemm::grid_packed_panel_min)
                            add(f, "packed", m, n, k);
                    }
            for (int mn : gemm::grid_skinny_mn)
                for (int k : gemm::grid_skinny_k) {
                    add(f, "strided", mn, 32, k);
                    add(f, "strided", 32, mn, k);
                }
        }
        if (std::string(dt) == "double")
            for (const auto& [f, layout, m, n, k] : std::set(cells))
                for (int kk : {1, 2}) add(f, layout.c_str(), m, n, kk);
        if (std::string(dt) == "float")
            for (const char* layout : {"strided", "packed"}) {
                for (int s : {1, 2, 4, 40, 49, 56}) add("NN", layout, s, s, s);
                for (int s : {8, 16, 24, 32, 40, 48})
                    for (int v : {1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64}) {
                        add("NN", layout, v, s, s);
                        add("NN", layout, s, v, s);
                        add("NN", layout, s, s, v);
                    }
            }
        std::map<std::string, std::string> rows_89;
        for (const char* dev : {"sm_89", "sm_120"}) {
            const sel::Table& t = embedded(std::string("gemm.") + dt + "." + dev + ".txt");
            EXPECT_EQ(t.source, "transcribed:424a45bc") << t.file;
            std::set<std::string> got;
            for (const auto& row : t.rows) {
                std::string k = row.keys[0];
                for (std::size_t i = 1; i < row.keys.size(); ++i) k += " " + row.keys[i];
                got.insert(k);
                std::string ranked;
                for (const auto& e : row.ranked) ranked += e.spelling + "|";
                if (std::string(dev) == "sm_89") rows_89[k] = ranked;
                else EXPECT_EQ(ranked, rows_89[k]) << dt << " " << k << ": sm_120 differs from sm_89";
            }
            EXPECT_EQ(got, want) << dt << " " << dev;
            EXPECT_EQ(t.rows.size(), want.size()) << dt << " " << dev;
        }
    }
}

// gemv's transcriber likewise, for both transcribed devices: one row per (trans, out, red,
// batch) cell of choice.hh, and identical rows on sm_89 and sm_120 (the old predicates read no
// architecture).
TEST(TunedTables, GemvTranscribedTablesHoldExactlyTheChoiceGrid) {
    namespace gemv = batchlas::ops::gemv;
    std::set<std::string> want;
    for (const char* tr : {"N", "T"})
        for (int o : gemv::grid_out)
            for (int r : gemv::grid_red)
                for (int b : gemv::grid_batch)
                    want.insert(std::string(tr) + " " + std::to_string(o) + " " + std::to_string(r) + " " +
                                std::to_string(b));
    for (const char* dt : {"float", "double", "cfloat", "cdouble"}) {
        std::map<std::string, std::string> first;
        for (const char* dev : {"sm_89", "sm_120"}) {
            const sel::Table& t = embedded(std::string("gemv.") + dt + "." + dev + ".txt");
            std::set<std::string> got;
            std::map<std::string, std::string> ranked;
            for (const auto& row : t.rows) {
                const std::string k = row.keys[0] + " " + row.keys[1] + " " + row.keys[2] + " " + row.keys[3];
                got.insert(k);
                for (const auto& e : row.ranked) ranked[k] += e.spelling + "|";
            }
            EXPECT_EQ(got, want) << dt << " " << dev;
            EXPECT_EQ(t.rows.size(), want.size()) << dt << " " << dev;
            EXPECT_EQ(t.source.rfind("transcribed:", 0), 0u) << t.file;
            if (first.empty()) first = ranked;
            else EXPECT_EQ(ranked, first) << dt << ": sm_120 rows differ from sm_89";
        }
    }
}

// geqrf's transcriber likewise: one row per choice.hh grid cell (sq x grid_n, tall x grid_n x
// grid_aspect, wide x grid_wide_n x grid_wide_aspect), and identical rows on sm_89 and sm_120.
TEST(TunedTables, GeqrfTablesHoldExactlyTheChoiceGridOnBothDevices) {
    namespace geqrf = batchlas::ops::geqrf;
    std::set<std::string> want;
    for (int n : geqrf::grid_n) want.insert("sq " + std::to_string(n) + " 1");
    for (int n : geqrf::grid_n)
        for (int a : geqrf::grid_aspect) want.insert("tall " + std::to_string(n) + " " + std::to_string(a));
    for (int n : geqrf::grid_wide_n)
        for (int a : geqrf::grid_wide_aspect) want.insert("wide " + std::to_string(n) + " " + std::to_string(a));
    for (const char* dt : {"float", "double", "cfloat", "cdouble"}) {
        std::map<std::string, std::string> rows_89;
        for (const char* dev : {"sm_89", "sm_120"}) {
            const sel::Table& t = embedded(std::string("geqrf.") + dt + "." + dev + ".txt");
            std::set<std::string> got;
            for (const auto& row : t.rows) {
                const std::string key = row.keys[0] + " " + row.keys[1] + " " + row.keys[2];
                got.insert(key);
                std::string ranked;
                for (const auto& e : row.ranked) ranked += e.spelling + "|";
                if (std::string(dev) == "sm_89") rows_89[key] = ranked;
                else EXPECT_EQ(ranked, rows_89[key]) << dt << " " << key << ": sm_120 differs from sm_89";
            }
            EXPECT_EQ(got, want) << dt << " " << dev;
            EXPECT_EQ(t.rows.size(), want.size()) << dt << " " << dev;
        }
    }
}

// orgqr's transcription serves sm_89 and sm_120 alike (the old predicates read no arch): one row
// per n <= m cell of choice.hh's grid in each, and the two tables row-for-row identical.
TEST(TunedTables, OrgqrTablesHoldExactlyTheChoiceGridOnBothDevices) {
    namespace orgqr = batchlas::ops::orgqr;
    std::set<std::string> want;
    for (int m : orgqr::grid)
        for (int n : orgqr::grid)
            if (n <= m) want.insert(std::to_string(m) + " " + std::to_string(n));
    for (const char* dt : {"float", "double", "cfloat", "cdouble"}) {
        const sel::Table& a = embedded(std::string("orgqr.") + dt + ".sm_89.txt");
        const sel::Table& b = embedded(std::string("orgqr.") + dt + ".sm_120.txt");
        std::set<std::string> got;
        for (const auto& row : a.rows) got.insert(row.keys[0] + " " + row.keys[1]);
        EXPECT_EQ(got, want) << dt;
        ASSERT_EQ(a.rows.size(), want.size()) << dt;
        ASSERT_EQ(b.rows.size(), a.rows.size()) << dt;
        for (std::size_t i = 0; i < a.rows.size(); ++i) {
            EXPECT_EQ(a.rows[i].keys, b.rows[i].keys) << dt << " row " << i;
            ASSERT_EQ(a.rows[i].ranked.size(), b.rows[i].ranked.size()) << dt << " row " << i;
            for (std::size_t j = 0; j < a.rows[i].ranked.size(); ++j)
                EXPECT_EQ(a.rows[i].ranked[j].spelling, b.rows[i].ranked[j].spelling) << dt << " row " << i;
        }
    }
}

TEST(TunedTables, OrmqrTablesDeclareChoiceKeyNames) { expect_tables_declare("ormqr", batchlas::ops::ormqr::key_names); }
TEST(TunedTables, GetriTablesDeclareChoiceKeyNames) { expect_tables_declare("getri", batchlas::ops::getri::key_names); }
TEST(TunedTables, GesvTablesDeclareChoiceKeyNames) { expect_tables_declare("gesv", batchlas::ops::gesv::key_names); }
TEST(TunedTables, GesvdTablesDeclareChoiceKeyNames) { expect_tables_declare("gesvd", batchlas::ops::gesvd::key_names); }
TEST(TunedTables, SpmmTablesDeclareChoiceKeyNames) { expect_tables_declare("spmm", batchlas::ops::spmm::key_names); }
TEST(TunedTables, SyevTablesDeclareChoiceKeyNames) { expect_tables_declare("syev", batchlas::ops::syev::key_names); }

// ormqr's transcriber spells choice.hh's grid by hand (k over grid_m up to m), and one
// transcription serves both devices: every table holds exactly that grid, both sides, N/T/C.
TEST(TunedTables, OrmqrTablesHoldExactlyTheChoiceGridOnBothDevices) {
    namespace om = batchlas::ops::ormqr;
    std::set<std::string> want;
    for (const char* s : {"L", "R"})
        for (const char* tr : {"N", "T", "C"})
            for (int m : om::grid_m)
                for (int k : om::grid_m)
                    for (int q : om::grid_q)
                        for (int b : om::grid_batch)
                            if (k <= m)
                                want.insert(std::string(s) + " " + tr + " " + std::to_string(m) + " " +
                                            std::to_string(k) + " " + std::to_string(q) + " " + std::to_string(b));
    for (const char* dev : {"sm_89", "sm_120"})
        for (const char* dt : {"float", "double", "cfloat", "cdouble"}) {
            const sel::Table& t = embedded(std::string("ormqr.") + dt + "." + dev + ".txt");
            std::set<std::string> got;
            for (const auto& row : t.rows)
                got.insert(row.keys[0] + " " + row.keys[1] + " " + row.keys[2] + " " + row.keys[3] + " " +
                           row.keys[4] + " " + row.keys[5]);
            EXPECT_EQ(got, want) << dt << " " << dev;
            EXPECT_EQ(t.rows.size(), want.size()) << dt << " " << dev;
            EXPECT_EQ(t.source.rfind("transcribed:", 0), 0u) << t.file;
        }
}

// getrf's transcriber likewise, for both devices it writes: one row per (n, batch) of choice.hh,
// and the sm_89 and sm_120 rows identical (the old predicates read no architecture).
TEST(TunedTables, GetrfTablesHoldExactlyTheChoiceGridOnBothDevices) {
    namespace getrf = batchlas::ops::getrf;
    std::set<std::string> want;
    for (int n : getrf::grid_n)
        for (int b : getrf::grid_batch) want.insert(std::to_string(n) + " " + std::to_string(b));
    for (const char* dt : {"float", "double", "cfloat", "cdouble"}) {
        std::map<std::string, std::string> rows[2];
        int i = 0;
        for (const char* dev : {"sm_89", "sm_120"}) {
            const sel::Table& t = embedded(std::string("getrf.") + dt + "." + dev + ".txt");
            EXPECT_EQ(t.source.rfind("transcribed:", 0), 0u) << t.file;
            std::set<std::string> got;
            for (const auto& row : t.rows) {
                const std::string key = row.keys[0] + " " + row.keys[1];
                got.insert(key);
                for (const auto& e : row.ranked) rows[i][key] += e.spelling + " ";
            }
            EXPECT_EQ(got, want) << t.file;
            EXPECT_EQ(t.rows.size(), want.size()) << t.file;
            ++i;
        }
        EXPECT_EQ(rows[0], rows[1]) << dt << ": the sm_89 and sm_120 transcriptions differ";
    }
}

// Likewise getrs's transcriber, whose one transcription is written for sm_89 and sm_120 alike.
TEST(TunedTables, GetrsTranscribedTablesHoldExactlyTheChoiceGrid) {
    namespace getrs = batchlas::ops::getrs;
    std::set<std::string> want;
    for (int n : getrs::grid_n)
        for (int r : getrs::grid_nrhs)
            for (int b : getrs::grid_batch) want.insert(std::to_string(n) + " " + std::to_string(r) + " " + std::to_string(b));
    for (const char* dev : {"sm_89", "sm_120"})
        for (const char* dt : {"float", "double", "cfloat", "cdouble"}) {
            const sel::Table& t = embedded(std::string("getrs.") + dt + "." + dev + ".txt");
            EXPECT_EQ(t.source.rfind("transcribed:", 0), 0u) << t.file;
            std::set<std::string> got;
            for (const auto& row : t.rows) got.insert(row.keys[0] + " " + row.keys[1] + " " + row.keys[2]);
            EXPECT_EQ(got, want) << t.file;
            EXPECT_EQ(t.rows.size(), want.size()) << t.file;
        }
}

// Likewise getri's transcriber, whose one transcription is written for sm_89 and sm_120 alike.
TEST(TunedTables, GetriTranscribedTablesHoldExactlyTheChoiceGrid) {
    namespace getri = batchlas::ops::getri;
    std::set<std::string> want;
    for (int n : getri::grid_n)
        for (int b : getri::grid_batch) want.insert(std::to_string(n) + " " + std::to_string(b));
    for (const char* dev : {"sm_89", "sm_120"})
        for (const char* dt : {"float", "double", "cfloat", "cdouble"}) {
            const sel::Table& t = embedded(std::string("getri.") + dt + "." + dev + ".txt");
            std::set<std::string> got;
            for (const auto& row : t.rows) got.insert(row.keys[0] + " " + row.keys[1]);
            EXPECT_EQ(got, want) << dt << " " << dev;
            EXPECT_EQ(t.rows.size(), want.size()) << dt << " " << dev;
            EXPECT_EQ(t.source.rfind("transcribed:", 0), 0u) << t.file;
        }
}

// gesv's transcriber likewise spells choice.hh's grid by hand: one row per (n, nrhs) cell, on
// both transcribed devices.
TEST(TunedTables, GesvTranscribedTablesHoldExactlyTheChoiceGrid) {
    namespace gesv = batchlas::ops::gesv;
    std::set<std::string> want;
    for (int n : gesv::grid_n)
        for (int r : gesv::grid_nrhs) want.insert(std::to_string(n) + " " + std::to_string(r));
    for (const char* dev : {"sm_89", "sm_120"})
        for (const char* dt : {"float", "double", "cfloat", "cdouble"}) {
            const sel::Table& t = embedded(std::string("gesv.") + dt + "." + dev + ".txt");
            std::set<std::string> got;
            for (const auto& row : t.rows) got.insert(row.keys[0] + " " + row.keys[1]);
            EXPECT_EQ(got, want) << dt << " " << dev;
            EXPECT_EQ(t.rows.size(), want.size()) << dt << " " << dev;
        }
}

// gesvd's transcriber: one row per (herm, vec, m, n) cell of choice.hh, and the sm_89 and
// sm_120 transcriptions identical row for row (the old predicates read no architecture).
TEST(TunedTables, GesvdTablesHoldExactlyTheChoiceGridOnBothDevices) {
    namespace gesvd = batchlas::ops::gesvd;
    std::set<std::string> want;
    for (auto h : gesvd::grid_herm)
        for (auto v : gesvd::grid_vec)
            for (int m : gesvd::grid_mn)
                for (int n : gesvd::grid_mn)
                    want.insert(std::string(h) + " " + std::string(v) + " " + std::to_string(m) + " " +
                                std::to_string(n));
    for (const char* dt : {"float", "double", "cfloat", "cdouble"}) {
        std::map<std::string, std::string> rows[2];
        int i = 0;
        for (const char* dev : {"sm_89", "sm_120"}) {
            const sel::Table& t = embedded(std::string("gesvd.") + dt + "." + dev + ".txt");
            EXPECT_EQ(t.source.rfind("transcribed:", 0), 0u) << t.file;
            std::set<std::string> got;
            for (const auto& row : t.rows) {
                const std::string k = row.keys[0] + " " + row.keys[1] + " " + row.keys[2] + " " + row.keys[3];
                got.insert(k);
                for (const auto& e : row.ranked) rows[i][k] += e.spelling + "|";
            }
            EXPECT_EQ(got, want) << t.file;
            EXPECT_EQ(t.rows.size(), want.size()) << t.file;
            ++i;
        }
        EXPECT_EQ(rows[0], rows[1]) << dt << ": sm_89 and sm_120 transcriptions differ";
    }
}

// spmm's transcriber likewise: one row per (transA, transB, m, nrhs, batch) cell of choice.hh, in
// each of the three transcribed devices (the old predicates read no device fact).
TEST(TunedTables, SpmmTranscribedTablesHoldExactlyTheChoiceGrid) {
    namespace sp = batchlas::ops::spmm;
    std::set<std::string> want;
    for (const char* ta : {"N", "T"})
        for (const char* tb : {"N", "T"})
            for (int m : sp::grid_m)
                for (int r : sp::grid_nrhs)
                    for (int b : sp::grid_batch)
                        want.insert(std::string(ta) + " " + tb + " " + std::to_string(m) + " " + std::to_string(r) +
                                    " " + std::to_string(b));
    for (const char* dev : {"sm_89", "sm_120", "cpu"})
        for (const char* dt : {"float", "double", "cfloat", "cdouble"}) {
            const sel::Table& t = embedded(std::string("spmm.") + dt + "." + dev + ".txt");
            std::set<std::string> got;
            for (const auto& row : t.rows)
                got.insert(row.keys[0] + " " + row.keys[1] + " " + row.keys[2] + " " + row.keys[3] + " " +
                           row.keys[4]);
            EXPECT_EQ(got, want) << dt << " " << dev;
            EXPECT_EQ(t.rows.size(), want.size()) << dt << " " << dev;
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
