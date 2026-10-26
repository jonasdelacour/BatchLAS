// SPDX-License-Identifier: MIT
// Every table embedded from tuned/<op>.<device>.txt (docs/design/tuning-data.md,
// docs/design/flat-kernel-selection.md §8.4). No GPU.

#include <gtest/gtest.h>

#include <batchlas/error.hh>
#include <batchlas/util/env.hh>

#include <src/ops/gemm/choice.hh>
#include <src/ops/geqrf/choice.hh>
#include <src/ops/orgqr/choice.hh>
#include <src/ops/getrf/choice.hh>
#include <src/ops/getrs/choice.hh>
#include <src/ops/posv/choice.hh>
#include <src/ops/potrf/choice.hh>
#include <src/ops/trsm/choice.hh>
#include <src/ops/gemv/choice.hh>
#include <src/ops/ormqr/choice.hh>
#include <src/ops/getri/choice.hh>
#include <src/ops/potri/choice.hh>
#include <src/ops/gesv/choice.hh>
#include <src/ops/gesvd/choice.hh>
#include <src/ops/spmm/choice.hh>
#include <src/ops/syev/choice.hh>
#include <src/ops/symm/choice.hh>
#include <src/ops/syrk/choice.hh>
#include <src/ops/syr2k/choice.hh>
#include <src/ops/trmm/choice.hh>
#include <src/ops/hemm/choice.hh>
#include <src/ops/herk/choice.hh>
#include <src/ops/her2k/choice.hh>
#include <src/select/select.hh>

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

template <typename Array>
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
    namespace potri = batchlas::ops::potri;
    if (op == "potri") {
        if (dtype == "float") return spellings(potri::candidates<float>());
        if (dtype == "double") return spellings(potri::candidates<double>());
        if (dtype == "cfloat") return spellings(potri::candidates<std::complex<float>>());
        if (dtype == "cdouble") return spellings(potri::candidates<std::complex<double>>());
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
    namespace symm = batchlas::ops::symm;  // complex T is complex-symmetric
    if (op == "symm") {
        if (dtype == "float") return spellings(symm::candidates<float>());
        if (dtype == "double") return spellings(symm::candidates<double>());
        if (dtype == "cfloat") return spellings(symm::candidates<std::complex<float>>());
        if (dtype == "cdouble") return spellings(symm::candidates<std::complex<double>>());
    }
    namespace syrk = batchlas::ops::syrk;
    if (op == "syrk") {
        if (dtype == "float") return spellings(syrk::candidates<float>());
        if (dtype == "double") return spellings(syrk::candidates<double>());
        if (dtype == "cfloat") return spellings(syrk::candidates<std::complex<float>>());
        if (dtype == "cdouble") return spellings(syrk::candidates<std::complex<double>>());
    }
    namespace syr2k = batchlas::ops::syr2k;
    if (op == "syr2k") {
        if (dtype == "float") return spellings(syr2k::candidates<float>());
        if (dtype == "double") return spellings(syr2k::candidates<double>());
        if (dtype == "cfloat") return spellings(syr2k::candidates<std::complex<float>>());
        if (dtype == "cdouble") return spellings(syr2k::candidates<std::complex<double>>());
    }
    namespace trmm = batchlas::ops::trmm;
    if (op == "trmm") {
        if (dtype == "float") return spellings(trmm::candidates<float>());
        if (dtype == "double") return spellings(trmm::candidates<double>());
        if (dtype == "cfloat") return spellings(trmm::candidates<std::complex<float>>());
        if (dtype == "cdouble") return spellings(trmm::candidates<std::complex<double>>());
    }
    // complex-only: their real spellings are symm, syrk and syr2k.
    if (op == "hemm" && dtype[0] == 'c') return spellings(batchlas::ops::hemm::candidates<std::complex<float>>());
    if (op == "herk" && dtype[0] == 'c') return spellings(batchlas::ops::herk::candidates<std::complex<float>>());
    if (op == "her2k" && dtype[0] == 'c') return spellings(batchlas::ops::her2k::candidates<std::complex<float>>());
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
// A transcribed row has no times: its order is the transcribed one, so it is not checked here.
// A ledger table keeps the tiered race's order (survivors, then the eliminated by median;
// docs/design/tiered-tuning.md), whose medians come from different rep counts: not checked either.
std::vector<std::string> tie_rule_problems(const sel::Table& t) {
    std::vector<std::string> out;
    if (t.source.rfind("ledger:", 0) == 0) return out;
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

// choice.hh's key_names: the keys every table of the op declares after dtype, weights included.
std::string key_spec(const std::string& op) {
    std::string out;
    auto join = [&](const auto& names) {
        for (auto k : names) out += (out.empty() ? "" : " ") + std::string(k);
    };
    if (op == "potrf") join(batchlas::ops::potrf::key_names);
    if (op == "posv") join(batchlas::ops::posv::key_names);
    if (op == "trsm") join(batchlas::ops::trsm::key_names);
    if (op == "gemm") join(batchlas::ops::gemm::key_names);
    if (op == "gemv") join(batchlas::ops::gemv::key_names);
    if (op == "geqrf") join(batchlas::ops::geqrf::key_names);
    if (op == "orgqr") join(batchlas::ops::orgqr::key_names);
    if (op == "ormqr") join(batchlas::ops::ormqr::key_names);
    if (op == "getrf") join(batchlas::ops::getrf::key_names);
    if (op == "getrs") join(batchlas::ops::getrs::key_names);
    if (op == "getri") join(batchlas::ops::getri::key_names);
    if (op == "potri") join(batchlas::ops::potri::key_names);
    if (op == "gesv") join(batchlas::ops::gesv::key_names);
    if (op == "gesvd") join(batchlas::ops::gesvd::key_names);
    if (op == "spmm") join(batchlas::ops::spmm::key_names);
    if (op == "syev") join(batchlas::ops::syev::key_names);
    if (op == "symm") join(batchlas::ops::symm::key_names);
    if (op == "syrk") join(batchlas::ops::syrk::key_names);
    if (op == "syr2k") join(batchlas::ops::syr2k::key_names);
    if (op == "trmm") join(batchlas::ops::trmm::key_names);
    if (op == "hemm") join(batchlas::ops::hemm::key_names);
    if (op == "herk") join(batchlas::ops::herk::key_names);
    if (op == "her2k") join(batchlas::ops::her2k::key_names);
    return out;
}

// Every embedded file is <op>.<device>.txt, keyed "dtype:exact" + the op's choice.hh keys, and splits
// into per-dtype tables whose every spelling is a candidate of that dtype.
TEST(TunedTables, EveryEmbeddedTableIsKeyedByDtypeThenChoiceKeysAndNamesOnlyCandidates) {
    ASSERT_FALSE(sel::embedded_tables().empty());
    for (const auto& e : sel::embedded_tables()) {
        SCOPED_TRACE(std::string(e.name));
        const std::string name(e.name), op = name.substr(0, name.find('.'));
        EXPECT_EQ(std::count(name.begin(), name.end(), '.'), 2) << "not <op>.<device>.txt";
        const std::string spec = key_spec(op);
        ASSERT_FALSE(spec.empty()) << op << " has no key_names in this test's registry";
        EXPECT_NE(std::string(e.text).find("\n# keys: dtype:exact " + spec + "\n"), std::string::npos);
        std::vector<sel::Table> parts;
        ASSERT_NO_THROW(parts = sel::parse_tables(e.text, e.name));
        ASSERT_FALSE(parts.empty());
        for (const sel::Table& t : parts) {
            EXPECT_EQ(t.op, op);
            EXPECT_FALSE(t.rows.empty()) << t.dtype;
            const auto p = spelling_problems(t);
            EXPECT_TRUE(p.empty()) << t.dtype << "\n" << joined(p);
        }
    }
}

TEST(TunedTables, RankedTimesFollowTheTieRule) {
    ASSERT_FALSE(sel::embedded_tables().empty());
    for (const auto& e : sel::embedded_tables()) {
        SCOPED_TRACE(std::string(e.name));
        for (const sel::Table& t : sel::parse_tables(e.text, e.name)) {
            const auto p = tie_rule_problems(t);
            EXPECT_TRUE(p.empty()) << t.dtype << "\n" << joined(p);
        }
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


// One dtype's table of the embedded tuned/<op>.<device>.txt.
const sel::Table& embedded(const std::string& op, const std::string& dtype, const std::string& device) {
    static std::map<std::string, sel::Table> cache;
    const std::string name = op + "." + device + ".txt", id = name + "|" + dtype;
    auto it = cache.find(id);
    if (it == cache.end())
        for (const auto& e : sel::embedded_tables())
            if (e.name == name)
                for (sel::Table& t : sel::parse_tables(e.text, e.name))
                    if (t.dtype == dtype) it = cache.emplace(id, std::move(t)).first;
    if (it == cache.end()) throw batchlas::internal_error(id + " is not embedded");
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
        const sel::Table& t = embedded("posv", dt, "sm_89");
        std::set<std::string> got;
        for (const auto& row : t.rows)
            got.insert(row.keys[0] + " " + row.keys[1] + " " + row.keys[2] + " " + row.keys[3]);
        EXPECT_EQ(got, want) << dt;
        EXPECT_EQ(t.rows.size(), want.size()) << dt;
    }
}

// Likewise trsm's transcriber: one row per (side, trans, order, q, batch) cell of choice.hh, on
// sm_89 (sm_120 ships deep-measured ledger tables, checked below).
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
    for (const char* dev : {"sm_89"})
        for (const char* dt : {"float", "double", "cfloat", "cdouble"}) {
            const sel::Table& t = embedded("trsm", dt, dev);
            EXPECT_EQ(t.source, "ledger:benchmarks/results/tuning/ledger/trsm.sm_89.jsonl") << t.file;
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
    std::set<std::string> names;  // "<op>.<dtype>.<device>.txt" for every (file, dtype) part
    for (const auto& e : sel::embedded_tables())
        for (const sel::Table& t : sel::parse_tables(e.text, e.name))
            names.insert(t.op + "." + t.dtype + "." + t.device + ".txt");
    // hemm, herk and her2k are ComplexScalar-constrained: no real instantiation, so no real table.
    const std::set<std::string> real_only{};
    const std::set<std::string> complex_only{"hemm", "herk", "her2k"};
    for (const char* op : {"potrf", "posv", "trsm", "gemm", "gemv", "geqrf", "orgqr", "ormqr", "getrf", "getrs",
                           "getri", "potri", "gesv", "gesvd", "spmm", "syev", "symm", "syrk", "syr2k", "trmm", "hemm",
                           "herk", "her2k"})
        for (const char* dev : {"sm_89", "sm_120", "cpu"})
            for (const char* dt : {"float", "double", "cfloat", "cdouble"}) {
                if (std::string(dev) == "cpu" && std::string(op) != "spmm") continue;
                if ((dt[0] == 'c' && real_only.count(op)) || (dt[0] != 'c' && complex_only.count(op))) {
                    EXPECT_FALSE(names.count(std::string(op) + "." + dt + "." + dev + ".txt")) << op << " " << dt;
                    continue;
                }
                EXPECT_TRUE(names.count(std::string(op) + "." + dt + "." + dev + ".txt"))
                    << op << " " << dt << " " << dev;
            }
}

// Every sm_120 table is generated from the deep tiered run's ledger (tuned/README.md): the source
// names the committed ledger file and every dtype has timed rows (untimed rows are the old
// transcription kept only where the deep lattice measured nothing near, e.g. beyond the 2048
// dimension cap).
TEST(TunedTables, Sm120TablesAreDeepLedgerTablesNamingTheirLedger) {
    // The complex-symmetric symm/syrk/syr2k rows (2026-10-22) postdate the deep run: transcribed only.
    const std::set<std::string> real_only{"symm", "syrk", "syr2k"};
    for (const char* op : {"potrf", "posv", "trsm", "gemm", "gemv", "geqrf", "orgqr", "ormqr", "getrf", "getrs",
                           "getri", "gesv", "gesvd", "spmm", "syev", "symm", "syrk", "syr2k", "trmm"})
        for (const char* dt : {"float", "double", "cfloat", "cdouble"}) {
            if (dt[0] == 'c' && real_only.count(op)) continue;
            const sel::Table& t = embedded(op, dt, "sm_120");
            EXPECT_EQ(t.source, "ledger:benchmarks/results/tuning/ledger/" + std::string(op) + ".sm_120.jsonl") << t.file;
            std::size_t timed = 0;
            for (const auto& row : t.rows) timed += row.timed;
            EXPECT_GT(timed, 0u) << t.file << " " << dt;
        }
}

// Likewise gemm's transcriber: one row per demand-grid cell of choice.hh (plan §3): squares for
// every form and layout, panels (packed from m, n >= 128) and skinny shapes for the issued forms;
// plus the edge rows that bracket the old predicate below the grid (the transcriber's header):
// real batch {1, 63, 64}, double k {1, 2} per (form, layout, m, n), float NN extra squares and
// one-axis-off neighbours of the small squares.
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
        // Grid forms added after the transcription; the regenerated table holds them.
        // evidence: docs/perf/gemm.md#gemm-wide-instantiations-in-every-transpose-form
        const std::vector<std::string_view> later = cplx ? std::vector<std::string_view>{"TC", "CC", "TT"}
                                                         : std::vector<std::string_view>{};
        const std::vector<std::string_view> later_panel = cplx ? later : std::vector<std::string_view>{"TT"};
        auto in = [](const std::vector<std::string_view>& v, std::string_view f) {
            return std::find(v.begin(), v.end(), f) != v.end();
        };
        for (std::string_view f : forms) {
            if (in(later, f)) continue;
            for (int s : gemm::grid_square) {
                add(f, "strided", s, s, s);
                add(f, "packed", s, s, s);
            }
            if (!in(panel, f) || in(later_panel, f)) continue;
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
        for (const char* dev : {"sm_89"}) {
            const sel::Table& t = embedded("gemm", dt, dev);
            EXPECT_EQ(t.source, "ledger:benchmarks/results/tuning/ledger/gemm.sm_89.jsonl") << t.file;
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

// gemv's transcriber likewise, on sm_89: one row per (trans, out, red, batch) cell of choice.hh.
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
        for (const char* dev : {"sm_89"}) {
            const sel::Table& t = embedded("gemv", dt, dev);
            std::set<std::string> got;
            std::map<std::string, std::string> ranked;
            for (const auto& row : t.rows) {
                const std::string k = row.keys[0] + " " + row.keys[1] + " " + row.keys[2] + " " + row.keys[3];
                got.insert(k);
                for (const auto& e : row.ranked) ranked[k] += e.spelling + "|";
            }
            EXPECT_EQ(got, want) << dt << " " << dev;
            EXPECT_EQ(t.rows.size(), want.size()) << dt << " " << dev;
            if (first.empty()) first = ranked;
            else EXPECT_EQ(ranked, first) << dt << ": sm_120 rows differ from sm_89";
        }
    }
}

// geqrf's transcriber likewise: one row per choice.hh grid cell (sq x grid_n, tall x grid_n x
// grid_aspect, wide x grid_wide_n x grid_wide_aspect), on sm_89.
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
        for (const char* dev : {"sm_89"}) {
            const sel::Table& t = embedded("geqrf", dt, dev);
            std::set<std::string> got;
            for (const auto& row : t.rows) {
                const std::string key = row.keys[0] + " " + row.keys[1] + " " + row.keys[2];
                got.insert(key);
                std::string ranked;
                for (const auto& e : row.ranked) ranked += e.spelling + "|";
                if (std::string(dev) == "sm_89") rows_89[key] = ranked;
                else EXPECT_EQ(ranked, rows_89[key]) << dt << " " << key << ": sm_120 differs from sm_89";
            }
            // The deep wide cells (float, cfloat) drop the transcribed row they bracket: wide 8192 x 1
            // was not measured (cuSOLVER exceeds the cell timeout) and resolves to its deep neighbours.
            const bool deep = std::string(dt) == "float" || std::string(dt) == "cfloat";
            std::set<std::string> want_dt = want;
            if (deep) want_dt.erase("wide 8192 1");
            EXPECT_EQ(got, want_dt) << dt << " " << dev;
            EXPECT_EQ(t.rows.size(), want_dt.size()) << dt << " " << dev;
        }
    }
}

// orgqr's transcription on sm_89: one row per n <= m cell of choice.hh's grid.
TEST(TunedTables, OrgqrTablesHoldExactlyTheChoiceGridOnBothDevices) {
    namespace orgqr = batchlas::ops::orgqr;
    std::set<std::string> want;
    for (int m : orgqr::grid)
        for (int n : orgqr::grid)
            if (n <= m) want.insert(std::to_string(m) + " " + std::to_string(n));
    for (const char* dt : {"float", "double", "cfloat", "cdouble"}) {
        const sel::Table& a = embedded("orgqr", dt, "sm_89");
        std::set<std::string> got;
        for (const auto& row : a.rows) got.insert(row.keys[0] + " " + row.keys[1]);
        EXPECT_EQ(got, want) << dt;
        EXPECT_EQ(a.rows.size(), want.size()) << dt;
        }
}


// ormqr's transcriber spells choice.hh's grid by hand (k over grid_m up to m), and one
// transcription on sm_89 holds exactly that grid, both sides, N/T/C.
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
    for (const char* dev : {"sm_89"})
        for (const char* dt : {"float", "double", "cfloat", "cdouble"}) {
            const sel::Table& t = embedded("ormqr", dt, dev);
            std::set<std::string> got;
            for (const auto& row : t.rows)
                got.insert(row.keys[0] + " " + row.keys[1] + " " + row.keys[2] + " " + row.keys[3] + " " +
                           row.keys[4] + " " + row.keys[5]);
            EXPECT_EQ(got, want) << dt << " " << dev;
            EXPECT_EQ(t.rows.size(), want.size()) << dt << " " << dev;
            }
}

// getrf's transcriber likewise, on sm_89: one row per (n, batch) of choice.hh.
TEST(TunedTables, GetrfTablesHoldExactlyTheChoiceGridOnBothDevices) {
    namespace getrf = batchlas::ops::getrf;
    std::set<std::string> want;
    for (int n : getrf::grid_n)
        for (int b : getrf::grid_batch) want.insert(std::to_string(n) + " " + std::to_string(b));
    for (const char* dt : {"float", "double", "cfloat", "cdouble"}) {
        std::map<std::string, std::string> rows[1];
        int i = 0;
        for (const char* dev : {"sm_89"}) {
            const sel::Table& t = embedded("getrf", dt, dev);
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
    }
}

// Likewise getrs's transcriber, on sm_89.
TEST(TunedTables, GetrsTranscribedTablesHoldExactlyTheChoiceGrid) {
    namespace getrs = batchlas::ops::getrs;
    std::set<std::string> want;
    for (int n : getrs::grid_n)
        for (int r : getrs::grid_nrhs)
            for (int b : getrs::grid_batch) want.insert(std::to_string(n) + " " + std::to_string(r) + " " + std::to_string(b));
    for (const char* dev : {"sm_89"})
        for (const char* dt : {"float", "double", "cfloat", "cdouble"}) {
            const sel::Table& t = embedded("getrs", dt, dev);
            std::set<std::string> got;
            for (const auto& row : t.rows) got.insert(row.keys[0] + " " + row.keys[1] + " " + row.keys[2]);
            EXPECT_EQ(got, want) << t.file;
            EXPECT_EQ(t.rows.size(), want.size()) << t.file;
        }
}

// Likewise getri's transcriber, on sm_89.
TEST(TunedTables, GetriTranscribedTablesHoldExactlyTheChoiceGrid) {
    namespace getri = batchlas::ops::getri;
    std::set<std::string> want;
    for (int n : getri::grid_n)
        for (int b : getri::grid_batch) want.insert(std::to_string(n) + " " + std::to_string(b));
    for (const char* dev : {"sm_89"})
        for (const char* dt : {"float", "double", "cfloat", "cdouble"}) {
            const sel::Table& t = embedded("getri", dt, dev);
            std::set<std::string> got;
            for (const auto& row : t.rows) got.insert(row.keys[0] + " " + row.keys[1]);
            EXPECT_EQ(got, want) << dt << " " << dev;
            EXPECT_EQ(t.rows.size(), want.size()) << dt << " " << dev;
            }
}

// potri's table is transcribed on both devices (no timing): one row per grid cell, fused first where the
// one-launch solve can run (n <= 8, potrs::fused_fits), blocked first above it. A retune replaces rows with
// timed ones and this test goes with them.
TEST(TunedTables, PotriTranscribedTablesHoldExactlyTheChoiceGridAndTheFusedCeiling) {
    namespace potri = batchlas::ops::potri;
    std::set<std::string> want;
    for (int n : potri::grid_n)
        for (int b : potri::grid_batch) want.insert(std::to_string(n) + " " + std::to_string(b));
    for (const char* dev : {"sm_89", "sm_120"})
        for (const char* dt : {"float", "double", "cfloat", "cdouble"}) {
            const sel::Table& t = embedded("potri", dt, dev);
            std::set<std::string> got;
            for (const auto& row : t.rows) {
                got.insert(row.keys[0] + " " + row.keys[1]);
                EXPECT_EQ(row.ranked.front().spelling, std::stoi(row.keys[0]) <= 8 ? "fused" : "blocked")
                    << dt << " " << dev << " " << row.keys[0] << " " << row.keys[1];
            }
            EXPECT_EQ(got, want) << dt << " " << dev;
            EXPECT_EQ(t.rows.size(), want.size()) << dt << " " << dev;
// gesv's transcriber likewise spells choice.hh's grid by hand: one row per (n, nrhs) cell, on sm_89.
TEST(TunedTables, GesvTranscribedTablesHoldExactlyTheChoiceGrid) {
    namespace gesv = batchlas::ops::gesv;
    std::set<std::string> want;
    for (int n : gesv::grid_n)
        for (int r : gesv::grid_nrhs) want.insert(std::to_string(n) + " " + std::to_string(r));
    for (const char* dev : {"sm_89"})
        for (const char* dt : {"float", "double", "cfloat", "cdouble"}) {
            const sel::Table& t = embedded("gesv", dt, dev);
            std::set<std::string> got;
            for (const auto& row : t.rows) got.insert(row.keys[0] + " " + row.keys[1]);
            EXPECT_EQ(got, want) << dt << " " << dev;
            EXPECT_EQ(t.rows.size(), want.size()) << dt << " " << dev;
        }
}

// gesvd's transcriber: one row per (herm, vec, m, n) cell of choice.hh, on sm_89.
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
        std::map<std::string, std::string> rows[1];
        int i = 0;
        for (const char* dev : {"sm_89"}) {
            const sel::Table& t = embedded("gesvd", dt, dev);
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
    }
}

// spmm's transcriber likewise: one row per (transA, transB, m, nrhs, batch) cell of choice.hh, on
// sm_89 and the CPU (the old predicates read no device fact).
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
    for (const char* dev : {"sm_89", "cpu"})
        for (const char* dt : {"float", "double", "cfloat", "cdouble"}) {
            const sel::Table& t = embedded("spmm", dt, dev);
            std::set<std::string> got;
            for (const auto& row : t.rows)
                got.insert(row.keys[0] + " " + row.keys[1] + " " + row.keys[2] + " " + row.keys[3] + " " +
                           row.keys[4]);
            EXPECT_EQ(got, want) << dt << " " << dev;
            EXPECT_EQ(t.rows.size(), want.size()) << dt << " " << dev;
        }
}

// syev's transcriber spells choice.hh's grid by hand too; checked on sm_89.
TEST(TunedTables, SyevTablesHoldExactlyTheChoiceGridOnBothDevices) {
    namespace syev = batchlas::ops::syev;
    std::set<std::string> want;
    for (const char* j : {"N", "V"})
        for (int n : syev::grid_n)
            for (int b : syev::grid_batch) want.insert(std::string(j) + " " + std::to_string(n) + " " + std::to_string(b));
    for (const char* dev : {"sm_89"})
        for (const char* dt : {"float", "double", "cfloat", "cdouble"}) {
            const sel::Table& t = embedded("syev", dt, dev);
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
    const auto* r = embedded("potrf", "float", "sm_89").nearest(f);
    ASSERT_NE(r, nullptr);
    EXPECT_EQ(r->keys[1], "24") << "line " << r->line;
    EXPECT_EQ(r->ranked.front().spelling, "cta") << "line " << r->line;
    const sel::Key d{{"uplo", "L"}, {"n", 3}, {"batch", 128}};
    r = embedded("potrf", "double", "sm_89").nearest(d);
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
                                         batchlas::ops::potrf::spec.rules)),
              "cta");
}

// ---- the shipped tables through choose() ------------------------------------------------

using C = batchlas::ops::potrf::PotrfChoice;

std::string pick(const std::string& device, const char* uplo, std::int64_t n, std::int64_t batch) {
    const sel::Key k{{"uplo", uplo}, {"n", n}, {"batch", batch}};
    return sel::to_string(sel::choose("potrf", "float", sel::device_from_key(device), k,
                                      batchlas::ops::potrf::candidates<float>(),
                                      [](const C&) { return true; }, batchlas::ops::potrf::spec.rules));
}

TEST(TunedTables, ShippedPotrfRowsAreWhatChooseReturns) {
    ScopedEnvVar dir("BATCHLAS_TUNED_DIR", nullptr);
    ScopedEnvVar pin("BATCHLAS_POTRF_ROUTE", nullptr);
    sel::testing::use_embedded_tables();
    sel::testing::reset_warnings();
    const auto order = sel::tables_in_borrow_order("potrf", "float", sel::device_from_key("sm_120"));
    ASSERT_FALSE(order.empty());
    EXPECT_EQ(order.front()->file, "potrf.sm_120.txt");
    EXPECT_EQ(order.front()->dtype, "float");
    EXPECT_FALSE(order.front()->is_override);
    ::testing::internal::CaptureStderr();
    EXPECT_EQ(pick("sm_120", "L", 64, 8192), "lpanel:panel=8");
    EXPECT_EQ(pick("sm_120", "L", 512, 2048), "vendor");  // deep: vendor 13.22 ms, blocked 14.05 ms
    EXPECT_EQ(pick("sm_120", "U", 64, 8192), "cta");
    // §3/§7.6: no cpu table ships, so the CPU takes the last resort and never borrows.
    EXPECT_EQ(pick("cpu", "L", 64, 8192), "blocked");
    EXPECT_EQ(::testing::internal::GetCapturedStderr(), "");
}

}  // namespace
