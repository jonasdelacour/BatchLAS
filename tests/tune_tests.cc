// Host-only tests of the tuner core (tools/tune/tune_core.cc; flat-kernel-selection.md §6):
// the §6.3 tie rule and rotation, the §6.2 bisection, the JSONL records, the §6.4 hash, and
// the round trip through scripts/sweep_to_table.py --tuner. No GPU.

#include "../tools/tune/grid.hh"
#include "../tools/tune/race.hh"
#include "../tools/tune/replay_core.hh"
#include "../tools/tune/tier.hh"
#include "../tools/tune/tune_core.hh"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace batchlas::tune;
namespace fs = std::filesystem;

namespace {

const std::vector<std::string> kPotrfOrder{"tiny", "cta", "lpanel:panel=8", "lpanel:panel=16", "blocked", "vendor"};

std::string read_file(const fs::path& p) {
    std::ifstream f(p);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

std::string run(const std::string& cmd) {
    std::string out;
    if (FILE* p = popen(cmd.c_str(), "r")) {
        char buf[512];
        while (fgets(buf, sizeof(buf), p)) out += buf;
        if (pclose(p) != 0) out = "FAILED: " + out;
    }
    while (!out.empty() && out.back() == '\n') out.pop_back();
    return out;
}

fs::path scratch(const std::string& name) {
    const fs::path d = fs::temp_directory_path() / ("tune_tests_" + std::to_string(getpid())) / name;
    fs::create_directories(d);
    return d;
}

}  // namespace

TEST(TuneTieRule, WithinThreePercentGoesToCandidateListOrder) {
    // flat-kernel-selection.md §6.3's own example: blocked 14.10 vs vendor 13.74 ties, blocked wins.
    EXPECT_EQ(rank({{"vendor", 13.74}, {"blocked", 14.10}, {"lpanel:panel=8", 14.84}}, kPotrfOrder),
              (std::vector<std::string>{"blocked", "vendor", "lpanel:panel=8"}));
    // Exactly 3% is tied; just past it is ranked by time.
    EXPECT_EQ(rank({{"vendor", 1.0}, {"cta", 1.03}}, kPotrfOrder).front(), "cta");
    EXPECT_EQ(rank({{"vendor", 1.0}, {"cta", 1.0301}}, kPotrfOrder).front(), "vendor");
    // Untied entries follow by time, ties among them by list order.
    EXPECT_EQ(rank({{"tiny", 1.0}, {"vendor", 2.0}, {"blocked", 2.0}, {"cta", 1.5}}, kPotrfOrder),
              (std::vector<std::string>{"tiny", "cta", "blocked", "vendor"}));
}

TEST(TuneTieRule, MedianAndRemeasure) {
    EXPECT_DOUBLE_EQ(median({3, 1, 2}), 2);
    EXPECT_DOUBLE_EQ(median({4, 1, 3, 2}), 2.5);
    EXPECT_TRUE(std::isnan(median({})));
    EXPECT_FALSE(needs_remeasure({{"a", {1.0, 1.099}}, {"b", {2.0, 2.0}}}));
    EXPECT_TRUE(needs_remeasure({{"a", {1.0, 1.0}}, {"b", {2.0, 2.21}}}));
    EXPECT_FALSE(needs_remeasure({{"a", {1.0}}}));
}

TEST(TuneRotation, EveryArmVisitsEverySlotOncePerKReps) {
    EXPECT_EQ(rep_order(3, 0, false), (std::vector<std::size_t>{0, 1, 2}));
    EXPECT_EQ(rep_order(3, 1, false), (std::vector<std::size_t>{1, 2, 0}));
    EXPECT_EQ(rep_order(3, 0, true), (std::vector<std::size_t>{2, 1, 0}));
    EXPECT_EQ(rep_order(3, 1, true), (std::vector<std::size_t>{1, 0, 2}));
    for (bool rev : {false, true})
        for (std::size_t k : {1u, 2u, 5u, 6u}) {
            std::vector<std::set<std::size_t>> slots(k);
            for (int r = 0; r < int(k); ++r) {
                const auto o = rep_order(k, r, rev);
                ASSERT_EQ(std::set<std::size_t>(o.begin(), o.end()).size(), k);
                for (std::size_t s = 0; s < k; ++s) slots[s].insert(o[s]);
            }
            for (const auto& s : slots) EXPECT_EQ(s.size(), k) << "k=" << k << " rev=" << rev;
        }
}

TEST(TuneRefine, BisectsUntilTheBracketIsUnderTenPercent) {
    EXPECT_EQ(refine_midpoints({{16, "tiny"}, {32, "cta"}}), (std::vector<std::int64_t>{23}));
    EXPECT_TRUE(refine_midpoints({{16, "cta"}, {32, "cta"}}).empty());
    EXPECT_TRUE(refine_midpoints({{3, "tiny"}, {4, "cta"}}).empty());       // adjacent integers
    EXPECT_TRUE(refine_midpoints({{100, "tiny"}, {109, "cta"}}).empty());   // 1.09 < 1.1
    EXPECT_EQ(refine_midpoints({{100, "tiny"}, {110, "cta"}}).size(), 1u);  // 1.10 is not < 1.1
    EXPECT_EQ(refine_midpoints({{2, "a"}, {4, "b"}}), (std::vector<std::int64_t>{3}));
    // Drive it as the driver does, against a true edge at 187 between grid points 128 and 256.
    for (std::int64_t edge : {129, 187, 255}) {
        auto winner = [&](std::int64_t n) { return n < edge ? std::string("lpanel") : std::string("blocked"); };
        std::vector<LinePoint> line{{64, winner(64)}, {128, winner(128)}, {256, winner(256)}, {512, winner(512)}};
        int rounds = 0;
        for (auto mids = refine_midpoints(line); !mids.empty(); mids = refine_midpoints(line), ++rounds)
            for (std::int64_t m : mids) line.push_back({m, winner(m)});
        std::int64_t lo = 0, hi = 0;
        for (const auto& p : line) {
            if (p.n < edge) lo = std::max(lo, p.n);
            else if (!hi || p.n < hi) hi = p.n;
        }
        EXPECT_LT(double(hi) / double(lo), 1.1) << "edge " << edge;
        EXPECT_LE(rounds, 8) << "edge " << edge;
        std::set<std::int64_t> ns;
        for (const auto& p : line) ns.insert(p.n);
        EXPECT_EQ(ns.size(), line.size()) << "a midpoint was measured twice";
    }
}

TEST(TuneJsonl, RoundTripsEveryValueKind) {
    const CellKey key{{"uplo", "L"}, {"n", "64"}, {"batch", "8192"}};
    const double tricky = 0.1 + 0.2;
    const std::string line = Json().str("s", "a \"q\" \\ \n\t end").num("x", tricky).num("nan", NAN)
                                 .integer("i", -42).boolean("b", true).key(key).line();
    ASSERT_EQ(line.back(), '\n');
    std::string err;
    const auto r = parse_record(line, &err);
    ASSERT_TRUE(r) << err;
    EXPECT_EQ(r->get("s"), "a \"q\" \\ \n\t end");
    EXPECT_EQ(r->number("x"), tricky);  // bit-exact: shortest round-trip text
    EXPECT_TRUE(std::isnan(r->number("nan")));
    EXPECT_EQ(r->get("nan"), "null");
    EXPECT_EQ(r->get("i"), "-42");
    EXPECT_EQ(r->get("b"), "true");
    EXPECT_EQ(r->get("uplo"), "L");
    EXPECT_EQ(r->number("n"), 64);
    EXPECT_NE(line.find("\"n\": 64"), std::string::npos);  // log keys are JSON integers
    EXPECT_EQ(parse_key_arg(key_arg(key)), key);
    EXPECT_EQ(key_text(key), "uplo=L n=64 batch=8192");
    EXPECT_EQ(key_int(key_with(key, "n", "72"), "n"), 72);
    EXPECT_FALSE(parse_record("{\"a\": [1]}"));
    EXPECT_FALSE(parse_record("{\"a\": 1"));
    EXPECT_FALSE(parse_key_arg("n=1,bad"));
}

TEST(TuneHash, Sha256KnownVectorsAndTheKernelManifest) {
    EXPECT_EQ(sha256_hex(""), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    EXPECT_EQ(sha256_hex("abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    EXPECT_EQ(sha256_hex("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
              "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
    EXPECT_EQ(sha256_hex(std::string(1000000, 'a')),
              "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
    const fs::path d = scratch("hash");
    std::ofstream(d / "a.cc") << "one\n";
    std::ofstream(d / "b.hh") << "two";
    const std::string manifest = sha256_hex("one\n") + "  a.cc\n" + sha256_hex("two") + "  b.hh\n";
    EXPECT_EQ(kernel_hash(d.string(), {"a.cc", "b.hh"}), sha256_hex(manifest).substr(0, 8));
    std::string missing;
    EXPECT_FALSE(kernel_hash(d.string(), {"a.cc", "nope.cc"}, &missing));
    EXPECT_EQ(missing, "nope.cc");
    const std::string spec = "#include \"x.hh\"\n// kernel-sources-begin\n  \"a.cc\",\n  \"b.hh\",\n};\n"
                             "// kernel-sources-end\n\"c.cc\"\n";
    EXPECT_EQ(parse_kernel_list(spec), (std::vector<std::string>{"a.cc", "b.hh"}));
}

TEST(TuneHash, EverySpecListsExistingFilesAndAgreesWithTheCiChecker) {
    const std::string repo = BATCHLAS_TUNE_SOURCE_DIR;
    for (const char* op : {"potrf", "posv"}) {
        const auto list = parse_kernel_list(read_file(fs::path(repo) / "tools/tune" / (std::string(op) + "_spec.cc")));
        ASSERT_FALSE(list.empty()) << op;
        std::string missing;
        const auto h = kernel_hash(repo, list, &missing);
        ASSERT_TRUE(h) << op << ": " << missing;
        const std::string py = BATCHLAS_TUNE_PYTHON;
        if (py.empty()) continue;
        const std::string got = run("cd '" + repo + "' && '" + py +
                                    "' -c \"import sys; sys.path.insert(0, '.github/ci'); import check_tuned_tables as c; "
                                    "print(c.kernel_hash('.', c.parse_kernel_list(open('tools/tune/" +
                                    op + "_spec.cc').read()))[0])\"");
        EXPECT_EQ(got, *h) << op;
    }
}

TEST(TuneGate, FailsOnlyWhenBothPassesLose) {
    EXPECT_EQ(gate_verdict(1.06, 1.07, false), "FAIL");
    EXPECT_EQ(gate_verdict(1.06, 1.04, false), "pass");
    EXPECT_EQ(gate_verdict(1.05, 1.05, false), "pass");
    EXPECT_EQ(gate_verdict(0.9, 0.95, true), "BAD_ROW");
    EXPECT_EQ(gate_verdict(1.2, 1.3, true), "FAIL");
}

// The tuner writes JSONL and the converter writes the table (one formatter). The driver's own
// ranking (rank() on the mean of final-attempt pass medians) must agree with what lands there.
TEST(TuneConverter, TunerJsonlRoundTripsThroughSweepToTable) {
    const std::string py = BATCHLAS_TUNE_PYTHON;
    if (py.empty()) GTEST_SKIP() << "no python3";
    const fs::path d = scratch("convert");
    const std::vector<std::string> cands{"tiny", "cta", "lpanel:panel=8", "lpanel:panel=16", "blocked", "vendor"};
    std::ofstream f(d / "potrf.float.sm_999.jsonl");
    f << Json().str("kind", "meta").integer("schema", 1).str("op", "potrf").str("dtype", "float")
             .str("device", "sm_999").str("batchlas", "abc12345").str("kernels", "0badf00d").str("date", "2026-01-02")
             .str("keys", "uplo:exact n:log:3 batch:log").str("candidates", "tiny|cta|lpanel:panel=8|lpanel:panel=16|blocked|vendor")
             .integer("reps", 16).num("warm_s", 1.5).integer("passes", 2).integer("ld_pad", 0).line();
    struct P { std::string n, cand, status; int pass, attempt; double ms; };
    const std::vector<P> rows{
        // n=64: lpanel wins clearly; vendor errored in pass 2 and must be dropped.
        {"64", "lpanel:panel=8", "ok", 1, 0, 0.30}, {"64", "lpanel:panel=8", "ok", 2, 0, 0.31},
        {"64", "cta", "ok", 1, 0, 0.53}, {"64", "cta", "ok", 2, 0, 0.53},
        {"64", "vendor", "ok", 1, 0, 0.49}, {"64", "vendor", "error", 2, 0, 0},
        {"64", "tiny", "skipped", 1, 0, 0}, {"64", "tiny", "skipped", 2, 0, 0},
        // n=512: attempt 0 was noisy, attempt 1 replaces it; blocked and vendor tie (3%).
        {"512", "blocked", "ok", 1, 0, 30.0}, {"512", "blocked", "ok", 2, 0, 14.1},
        {"512", "vendor", "ok", 1, 0, 13.7}, {"512", "vendor", "ok", 2, 0, 13.8},
        {"512", "blocked", "ok", 1, 1, 14.10}, {"512", "blocked", "ok", 2, 1, 14.10},
        {"512", "vendor", "ok", 1, 1, 13.70}, {"512", "vendor", "ok", 2, 1, 13.78},
        // n=96: still noisy after the re-measure: kept, marked # noisy.
        {"96", "cta", "ok", 1, 1, 1.0}, {"96", "cta", "ok", 2, 1, 1.2},
    };
    for (const P& r : rows) {
        f << Json().str("kind", "pass").str("op", "potrf").str("dtype", "float").key({{"uplo", "L"}, {"n", r.n}, {"batch", "8192"}})
                 .integer("pass", r.pass).integer("attempt", r.attempt).str("cand", r.cand).str("status", r.status)
                 .num("median_ms", r.status == "ok" ? r.ms : NAN).line();
        f << Json().str("kind", "rep").str("op", "potrf").key({{"uplo", "L"}, {"n", r.n}, {"batch", "8192"}})
                 .integer("pass", r.pass).str("cand", r.cand).num("ms", 99.0).line();
    }
    f.close();
    const std::string conv = std::string(BATCHLAS_TUNE_SOURCE_DIR) + "/scripts/sweep_to_table.py";
    ASSERT_EQ(run("'" + py + "' '" + conv + "' --tuner '" + (d / "potrf.float.sm_999.jsonl").string() + "' --out '" +
                  d.string() + "'").rfind("FAILED", 0), std::string::npos);
    const std::string table = read_file(d / "potrf.float.sm_999.txt");
    EXPECT_NE(table.find("# op=potrf dtype=float device=sm_999 batchlas=abc12345 kernels=0badf00d date=2026-01-02\n"),
              std::string::npos) << table;
    EXPECT_NE(table.find("# keys: uplo:exact n:log:3 batch:log\n"), std::string::npos);
    EXPECT_NE(table.find("uplo=L n=64 batch=8192 | lpanel:panel=8 0.3050 | cta 0.5300\n"), std::string::npos) << table;
    EXPECT_NE(table.find("uplo=L n=96 batch=8192 | cta 1.100   # noisy\n"), std::string::npos) << table;
    // The C++ rank of the final attempt agrees with the converter's first entry.
    const auto first = rank({{"blocked", 14.10}, {"vendor", 13.74}}, cands).front();
    EXPECT_NE(table.find("uplo=L n=512 batch=8192 | " + first + " 14.10 | vendor 13.74\n"), std::string::npos) << table;
}

// rank() in C++ and in the converter on random times, ties straddled on both sides of 3%.
TEST(TuneConverter, RankAgreesWithTheConverterOnRandomTimes) {
    const std::string py = BATCHLAS_TUNE_PYTHON;
    if (py.empty()) GTEST_SKIP() << "no python3";
    const fs::path d = scratch("rank");
    std::mt19937 gen(20261004);
    std::uniform_real_distribution<double> base(0.05, 20.0), spread(0.0, 0.2);
    const std::vector<double> edges{0.0, 0.01, 0.0299, 0.03, 0.0301, 0.05};
    std::ofstream in(d / "cases.txt");
    std::vector<std::string> want;
    for (int c = 0; c < 400; ++c) {
        std::vector<std::string> pool = kPotrfOrder;
        std::shuffle(pool.begin(), pool.end(), gen);
        pool.resize(1 + gen() % pool.size());
        const double best = base(gen);
        std::map<std::string, double> times;
        for (const auto& cand : pool)
            times[cand] = best * (1.0 + (gen() % 2 ? edges[gen() % edges.size()] : spread(gen)));
        for (const auto& [cand, t] : times) {
            char buf[64];
            std::snprintf(buf, sizeof(buf), "%.17g", t);
            in << cand << "=" << buf << " ";
        }
        in << "\n";
        std::string r;
        for (const auto& s : rank(times, kPotrfOrder)) r += (r.empty() ? "" : "|") + s;
        want.push_back(r);
    }
    in.close();
    std::ofstream(d / "rank.py") << "import sys\nsys.path.insert(0, sys.argv[1] + '/scripts')\n"
                                    "import sweep_to_table as s\norder = s.POTRF.candidate_order\n"
                                    "for line in open(sys.argv[2]):\n"
                                    "    t = {c: float(v) for c, v in (f.rsplit('=', 1) for f in line.split())}\n"
                                    "    print('|'.join(s.rank(t, order)))\n";
    const std::string got = run("'" + py + "' '" + (d / "rank.py").string() + "' '" + BATCHLAS_TUNE_SOURCE_DIR +
                                "' '" + (d / "cases.txt").string() + "'");
    ASSERT_EQ(got.rfind("FAILED", 0), std::string::npos) << got;
    std::istringstream lines(got);
    std::string line;
    for (std::size_t i = 0; i < want.size(); ++i) {
        ASSERT_TRUE(std::getline(lines, line)) << "converter printed " << i << " of " << want.size() << " rankings";
        EXPECT_EQ(line, want[i]) << "case " << i;
    }
}

// The configure-time staleness check computes the same hash as the driver.
TEST(TuneHash, CmakeStalenessHashAgreesWithTheDriver) {
    const std::string repo = BATCHLAS_TUNE_SOURCE_DIR, cmake = BATCHLAS_TUNE_CMAKE;
    const fs::path d = scratch("cmake_hash");
    std::ofstream(d / "hash.cmake") << "include(\"" << repo << "/cmake/BatchLASTunedStaleness.cmake\")\n"
                                    << "_batchlas_tune_kernel_hash(\"${SPEC}\" h)\nmessage(\"HASH=${h}\")\n";
    for (const char* op : {"potrf", "posv"}) {
        const std::string spec = repo + "/tools/tune/" + op + "_spec.cc";
        const std::string out = run("'" + cmake + "' -DPROJECT_SOURCE_DIR='" + repo + "' -DSPEC='" + spec + "' -P '" +
                                    (d / "hash.cmake").string() + "' 2>&1");
        const auto h = kernel_hash(repo, parse_kernel_list(read_file(spec)));
        ASSERT_TRUE(h) << op;
        EXPECT_EQ(out, "HASH=" + *h) << op;
    }
}

TEST(TuneCsv, EmptyFieldsSurviveAndErrorTextIsQuoted) {
    EXPECT_EQ(split_fields("a,,b,", ','), (std::vector<std::string>{"a", "", "b", ""}));
    EXPECT_EQ(split("a,,b,", ','), (std::vector<std::string>{"a", "b"}));
    EXPECT_EQ(parse_csv_line("float,44,\"x,y\",,\"q\"\"r\"\r"),
              (std::vector<std::string>{"float", "44", "x,y", "", "q\"r"}));
    EXPECT_EQ(csv_field("lpanel:panel=8"), "lpanel:panel=8");
    const std::string err = "error:exit 134: a, \"b\"\nc";
    EXPECT_EQ(parse_csv_line("x," + csv_field(err) + ",y"), (std::vector<std::string>{"x", "error:exit 134: a, \"b\" c", "y"}));
}

TEST(TuneOldCsv, ReadsEveryRowOfAGateCsvAndRejectsWhatItCannotRead) {
    // A previous gate's output: 87 of its 120 rows have empty fields.
    std::string err;
    const auto real = parse_old_csv(read_file(fs::path(BATCHLAS_TUNE_SOURCE_DIR) /
                                              "benchmarks/results/routing/sm120_potrf_phase2_gate.csv"),
                                    {"n", "batch"}, {}, &err);
    ASSERT_TRUE(real) << err;
    EXPECT_EQ(real->size(), 120u);
    const std::string head = "dtype,uplo,n,batch,old,new,old_ms_p1,new_ms_p1,ratio_p1,ratio_p2,verdict\n";
    const std::string text = head + "float,L,44,512,lpanel:panel=8,lpanel:panel=8,,,,,same\n\n"
                                    "double,U,72,512,native:lpanel,blocked,1.0,1.1,1.1,1.1,FAIL\n";
    const auto rows = parse_old_csv(text, {"uplo", "n", "batch"}, {}, &err);
    ASSERT_TRUE(rows) << err;
    ASSERT_EQ(rows->size(), 2u);
    EXPECT_EQ((*rows)[1].dtype, "double");
    EXPECT_EQ((*rows)[1].old, "native:lpanel");
    EXPECT_EQ((*rows)[1].key, (CellKey{{"uplo", "U"}, {"n", "72"}, {"batch", "512"}}));
    EXPECT_FALSE(parse_old_csv(text, {"uplo", "n", "batch"}, {"float"}, &err));  // a dtype --dtype leaves out
    EXPECT_NE(err.find("row 3"), std::string::npos) << err;
    EXPECT_FALSE(parse_old_csv(head + "float,L,44,512,tiny,tiny,,,,same\n", {"uplo", "n", "batch"}, {}, &err));
    EXPECT_NE(err.find("10 fields"), std::string::npos) << err;
    EXPECT_FALSE(parse_old_csv(head + "float,L,44,512,,tiny,,,,,same\n", {"uplo", "n", "batch"}, {}, &err));
    EXPECT_FALSE(parse_old_csv(head, {"uplo", "n", "batch"}, {}, &err));
    EXPECT_FALSE(parse_old_csv(text, {"uplo", "n", "nrhs", "batch"}, {}, &err));
    EXPECT_EQ(split_origin("native:lpanel"), (std::pair<std::string, std::string>{"native", "lpanel"}));
    EXPECT_EQ(split_origin("vendor:auto"), (std::pair<std::string, std::string>{"vendor", "auto"}));
    EXPECT_EQ(split_origin("lpanel:panel=8"), (std::pair<std::string, std::string>{"native", "lpanel:panel=8"}));
}

TEST(TuneGate, ExitCodeCannotPassAnIncompleteGate) {
    auto code = [](std::vector<std::string> verdicts) {
        GateCounts c;
        for (const auto& v : verdicts) gate_count(c, v);
        return gate_exit_code(c);
    };
    EXPECT_EQ(code({"same", "same"}), 0);
    EXPECT_EQ(code({"same", "pass", "cap"}), 0);
    EXPECT_EQ(code({"ERROR", "ERROR"}), 3);  // every probe failed: never "0 FAIL, exit 0"
    EXPECT_EQ(code({"same", "BAD_ROW"}), 3);
    EXPECT_EQ(code({}), 3);
    EXPECT_EQ(code({"cap"}), 3);
    EXPECT_EQ(code({"same", "ERROR", "FAIL"}), 1);
    EXPECT_EQ(code({"something new"}), 3);  // an unknown verdict counts as an error
    GateCounts c;
    for (const char* v : {"same", "pass", "FAIL", "BAD_ROW", "ERROR", "cap"}) gate_count(c, v);
    EXPECT_EQ(gate_summary(c), "1 same, 1 timed pass, 1 FAIL, 1 BAD_ROW, 1 ERROR, 1 over the cap");
}

namespace {
PassData pass_of(std::map<std::string, std::pair<std::string, double>> arms) {
    PassData p;
    for (const auto& [cand, sm] : arms) p.arms[cand] = {sm.first, "", sm.second};
    return p;
}
}  // namespace

TEST(TuneAttempts, FinalAttemptNeedsEveryPassAndSurvivesAFailedRemeasure) {
    const std::vector<std::string> cands{"tiny", "cta", "blocked"};
    const Attempt a0{pass_of({{"tiny", {"ok", 1.0}}, {"cta", {"ok", 2.0}}, {"blocked", {"ok", 3.0}}}),
                     pass_of({{"tiny", {"ok", 1.2}}, {"cta", {"ok", 2.0}}, {"blocked", {"error", 0}}})};
    EXPECT_EQ(attempt_times(a0, cands), (std::map<std::string, double>{{"tiny", 1.1}, {"cta", 2.0}}));
    EXPECT_TRUE(attempt_needs_remeasure(a0, cands, 0.10));  // tiny 1.0 vs 1.2
    const Attempt quiet{pass_of({{"tiny", {"ok", 1.0}}, {"blocked", {"ok", 1.0}}}),
                        pass_of({{"tiny", {"ok", 1.05}}, {"blocked", {"error", 9.0}}})};
    EXPECT_FALSE(attempt_needs_remeasure(quiet, cands, 0.10));  // only an arm that failed a pass moved
    const Attempt a1{pass_of({{"tiny", {"ok", 1.0}}, {"cta", {"skipped", 0}}}),
                     pass_of({{"tiny", {"ok", 1.02}}, {"cta", {"skipped", 0}}})};
    const Attempt lost{pass_of({{"tiny", {"error", 0}}, {"cta", {"error", 0}}}),
                       pass_of({{"tiny", {"error", 0}}, {"cta", {"error", 0}}})};
    int used = 9;
    EXPECT_EQ(final_times({a0, a1}, cands, &used), (std::map<std::string, double>{{"tiny", 1.01}}));
    EXPECT_EQ(used, 1);
    EXPECT_EQ(final_times({a0, lost}, cands, &used), attempt_times(a0, cands));
    EXPECT_EQ(used, 0);
    EXPECT_TRUE(final_times({}, cands, &used).empty());
    EXPECT_EQ(used, -1);
}

TEST(TuneRefine, RoundGroupsLinesSkipsKnownCellsAndReportsStalls) {
    const std::vector<std::string> order{"lpanel", "blocked"};
    auto k = [](const char* u, int n, int b) {
        return CellKey{{"uplo", u}, {"n", std::to_string(n)}, {"batch", std::to_string(b)}};
    };
    const std::map<std::string, double> lp{{"lpanel", 1.0}, {"blocked", 2.0}}, bl{{"lpanel", 2.0}, {"blocked", 1.0}};
    const std::map<CellKey, std::map<std::string, double>> measured{
        {k("L", 128, 8192), lp}, {k("L", 256, 8192), bl},                          // edge: bisect at 181
        {k("U", 128, 8192), bl}, {k("U", 256, 8192), bl},                          // no edge
        {k("L", 128, 512), lp},  {k("L", 256, 512), bl},  {k("L", 181, 512), {}},  // 181 known, no winner
        {k("U", 64, 512), {}},   {k("U", 512, 512), lp}};                          // a lone point
    const RefineRound r = refine_round(measured, "n", order, 0.03, 1.1);
    EXPECT_EQ(r.next, (std::vector<CellKey>{k("L", 181, 8192)}));
    ASSERT_EQ(r.stalled.size(), 1u);
    EXPECT_EQ(r.stalled[0], "uplo=L n=* batch=512: edge between n=128 (lpanel) and 256 (blocked) stays wide: "
                            "n=181 has no winner");
}

TEST(TuneCoverage, ReachedRouteFindsColumnsByTheHeader) {
    // The emit() format of src/select/coverage.cc.
    const std::string head = "kind,op,scalar,backend,shape_class,m,n,k,batch,chosen_origin,chosen_algo,calls,"
                             "native_route_existed,native_route_supported,library,uplo,side,diag,transA,transB\n";
    const std::string rows = "reached,trsm,float,CUDA,0,0,64,0,8192,vendor,auto,1,1,1,,0,0,0,0,0\n"
                             "reached,potrf,float,CUDA,0,0,64,0,8192,native,lpanel:panel=8,3,1,1,,0,0,0,0,0\n"
                             "miss,potrf,float,CUDA,,,,,,,,1,0,0,cusolver\n";
    using R = std::pair<std::string, std::string>;
    EXPECT_EQ(reached_route(head + rows, "potrf"), R("native", "lpanel:panel=8"));
    EXPECT_EQ(reached_route(head + rows, "trsm"), R("vendor", "auto"));
    EXPECT_FALSE(reached_route(head + rows, "gemm"));
    EXPECT_FALSE(reached_route(rows, "potrf"));  // no header: no columns to trust
    EXPECT_EQ(reached_route("kind,chosen_algo,op,chosen_origin\nreached,cta,potrf,native\n", "potrf"), R("native", "cta"));
}

TEST(TuneDevices, ExplicitDevicesInsideTheCallersFence) {
    using O = std::optional<std::string>;
    EXPECT_TRUE(devices_problem({}, true, std::nullopt));
    EXPECT_TRUE(devices_problem({1}, false, std::nullopt));  // no default device
    EXPECT_FALSE(devices_problem({1}, true, std::nullopt));
    EXPECT_TRUE(devices_problem({0}, false, O("2")));
    EXPECT_TRUE(devices_problem({0}, true, O("2")));  // outside the fence
    EXPECT_FALSE(devices_problem({2}, true, O("1,2")));
    EXPECT_TRUE(devices_problem({1}, true, O("GPU-5c3e")));
    EXPECT_TRUE(devices_problem({1}, true, O("")));
}

TEST(TuneGuard, EveryOtherLineIsForeignIncludingUnparseableOnes) {
    const AppScan a = scan_compute_apps("123\n 456 \n\n", 123);
    EXPECT_TRUE(a.self);
    EXPECT_EQ(a.foreign, (std::vector<std::string>{"456"}));
    const AppScan b = scan_compute_apps("[N/A]\n[Insufficient Permissions]\n", 7);
    EXPECT_FALSE(b.self);
    EXPECT_EQ(b.foreign.size(), 2u);
    EXPECT_TRUE(scan_compute_apps(" \n", 7).foreign.empty());
}

AppScan foreign(std::vector<std::string> f) { return AppScan{std::move(f), false}; }

TEST(TuneGuard, StrictModeRefusesAnyForeignProcessAndDiscardsOnAny) {
    EXPECT_EQ(guard_before(foreign({"610696"}), 0, 5, false).refuse, "compute processes [610696]");
    EXPECT_EQ(guard_before(foreign({}), 6, 5, false).refuse, "utilization 6%");
    const GuardCheck ok = guard_before(foreign({}), 5, 5, false);
    EXPECT_EQ(ok.refuse, "");
    EXPECT_TRUE(ok.tolerated.empty());
    EXPECT_EQ(guard_new_foreign(foreign({"610696"}), ok.tolerated), (std::vector<std::string>{"610696"}));
    EXPECT_TRUE(guard_new_foreign(foreign({}), ok.tolerated).empty());
}

TEST(TuneGuard, IdleForeignPidsAreToleratedAtStart) {
    const GuardCheck g = guard_before(foreign({"610696", "671125"}), 3, 5, true);
    EXPECT_EQ(g.refuse, "");
    EXPECT_EQ(g.tolerated, (std::vector<std::string>{"610696", "671125"}));
    EXPECT_TRUE(guard_new_foreign(foreign({"671125", "610696"}), g.tolerated).empty());
    EXPECT_TRUE(guard_new_foreign(foreign({"610696"}), g.tolerated).empty());  // one exited
}

TEST(TuneGuard, NewForeignPidDuringTheChildDiscards) {
    const GuardCheck g = guard_before(foreign({"610696"}), 0, 5, true);
    ASSERT_EQ(g.refuse, "");
    EXPECT_EQ(guard_new_foreign(foreign({"610696", "700001"}), g.tolerated), (std::vector<std::string>{"700001"}));
    EXPECT_EQ(guard_new_foreign(foreign({"[N/A]"}), g.tolerated), (std::vector<std::string>{"[N/A]"}));
}

TEST(TuneGuard, BusyUtilizationRefusesEvenWithIdleForeignAllowed) {
    EXPECT_EQ(guard_before(foreign({"610696"}), 57, 5, true).refuse,
              "utilization 57% with compute processes [610696]");
    EXPECT_EQ(guard_before(foreign({}), 6, 5, true).refuse, "utilization 6%");
    EXPECT_TRUE(guard_before(foreign({"610696"}), 57, 5, true).tolerated.empty());
}

TEST(TuneGuard, UnparseableEntriesRefuseEvenWithIdleForeignAllowed) {
    EXPECT_EQ(guard_before(foreign({"610696", "[N/A]"}), 0, 5, true).refuse, "compute processes [610696,[N/A]]");
}

namespace {

// Feeds rounds of {a, b} times from `gen(round)` until race_step stops asking for more.
struct RaceRun {
    RaceState s;
    RaceVerdict v = RaceVerdict::more;
    int rounds = 0;
};
RaceRun run_race(const TierParams& p, const std::function<std::pair<double, double>(int)>& gen) {
    RaceRun r;
    r.s.cands = {"a", "b"};
    r.s.ms.assign(2, {});
    r.s.alive.assign(2, true);
    while (r.v == RaceVerdict::more && r.rounds < 100) {
        const auto [a, b] = gen(r.rounds++);
        r.s.ms[0].push_back(a);
        r.s.ms[1].push_back(b);
        r.v = race_step(r.s, p);
    }
    return r;
}

}  // namespace

TEST(TuneRace, FourPercentSlowerIsEliminatedInDeep) {
    std::mt19937 g(7);
    std::uniform_real_distribution<double> u(-0.005, 0.005);
    const auto r = run_race(params(Tier::deep), [&](int) { return std::pair(1.00 * (1 + u(g)), 1.04 * (1 + u(g))); });
    EXPECT_FALSE(r.s.alive[1]);
    EXPECT_TRUE(r.s.alive[0]);
    EXPECT_EQ(r.v, RaceVerdict::winner);
}

TEST(TuneRace, TwoPercentSlowerIsKeptAsATie) {
    std::mt19937 g(11);
    std::uniform_real_distribution<double> u(-0.005, 0.005);
    const auto r = run_race(params(Tier::deep), [&](int) { return std::pair(1.00 * (1 + u(g)), 1.02 * (1 + u(g))); });
    EXPECT_TRUE(r.s.alive[1]);
    EXPECT_TRUE(r.v == RaceVerdict::tie || r.v == RaceVerdict::cap);
    EXPECT_EQ(race_ranking(r.s, {"b", "a"}), (std::vector<std::string>{"b", "a"}));
    EXPECT_EQ(race_ranking(r.s, {"a", "b"}), (std::vector<std::string>{"a", "b"}));
}

TEST(TuneRace, SingleRoundNeverEliminates) {
    for (Tier t : {Tier::ultra, Tier::coarse, Tier::deep}) {
        const auto& p = params(t);
        const auto r = run_race(p, [&](int) { return std::pair(1.0, 10.0); });
        EXPECT_EQ(r.rounds, p.min_reps) << to_string(t);
        EXPECT_FALSE(r.s.alive[1]) << to_string(t);
        RaceState s;
        s.cands = {"a", "b"};
        s.ms = {{1.0}, {10.0}};
        s.alive = {true, true};
        EXPECT_EQ(race_step(s, p), RaceVerdict::more);
        EXPECT_TRUE(s.alive[1]);
    }
}

TEST(TuneRace, PairedRatiosCancelADriftingClock) {
    std::mt19937 g(3);
    std::uniform_real_distribution<double> u(-0.003, 0.003);
    const auto r = run_race(params(Tier::coarse), [&](int i) {
        const double drift = 1.0 + 0.3 * i / 12.0;
        return std::pair(drift * (1 + u(g)), 1.06 * drift * (1 + u(g)));
    });
    EXPECT_FALSE(r.s.alive[1]);
    EXPECT_EQ(r.v, RaceVerdict::winner);
}

TEST(TuneRace, NaNRoundsAreUnpaired) {
    const auto nan = std::numeric_limits<double>::quiet_NaN();
    RaceState s;
    s.cands = {"a", "b"};
    s.alive = {true, true};
    s.ms = {{1, 1, 1, 1}, {nan, 2, nan, 2}};
    EXPECT_EQ(race_step(s, params(Tier::ultra)), RaceVerdict::more);  // 2 pairs < min_reps 3
    EXPECT_TRUE(s.alive[1]);
    s.ms = {{1, 1, 1, 1, 1}, {nan, 2, nan, 2, 2}};
    EXPECT_EQ(race_step(s, params(Tier::ultra)), RaceVerdict::winner);  // 3 pairs
    EXPECT_FALSE(s.alive[1]);
}

TEST(TuneRace, AllNaNRoundsNeverTie) {
    const auto nan = std::numeric_limits<double>::quiet_NaN();
    const auto& p = params(Tier::ultra);
    RaceState s;
    s.cands = {"a", "b"};
    s.alive = {true, true};
    s.ms = {std::vector<double>(p.max_reps - 1, nan), std::vector<double>(p.max_reps - 1, nan)};
    EXPECT_EQ(race_step(s, p), RaceVerdict::more);
    s.ms = {std::vector<double>(p.max_reps, nan), std::vector<double>(p.max_reps, nan)};
    EXPECT_EQ(race_step(s, p), RaceVerdict::cap);
    EXPECT_TRUE(s.alive[0] && s.alive[1]);
}

TEST(TuneRace, LowerOrderStatMatchesTheBinomial) {
    EXPECT_EQ(lower_order_stat(3, 0.80), 1u);
    EXPECT_EQ(lower_order_stat(6, 0.98), 1u);
    EXPECT_EQ(lower_order_stat(16, 0.98), 4u);
    EXPECT_EQ(lower_order_stat(2, 0.98), 0u);
}

TEST(TuneRace, RankingPutsSurvivorsBeforeTheEliminated) {
    RaceState s;
    s.cands = {"a", "b", "c"};
    s.ms = {{2, 2}, {1, 1}, {1.5, 1.5}};
    s.alive = {true, true, false};
    EXPECT_EQ(race_ranking(s, {"a", "b", "c"}), (std::vector<std::string>{"b", "a", "c"}));
}

TEST(TuneTier, PrecedenceOrderAndRoundTrip) {
    EXPECT_LT(tier_rank(Tier::coarse), tier_rank(Tier::deep));
    EXPECT_LT(tier_rank(Tier::ultra), tier_rank(Tier::coarse));
    EXPECT_LT(tier_rank(Tier::custom), tier_rank(Tier::ultra));
    EXPECT_LT(tier_rank(Tier::transcribed), tier_rank(Tier::custom));
    for (Tier t : {Tier::transcribed, Tier::custom, Tier::ultra, Tier::coarse, Tier::deep})
        EXPECT_EQ(parse_tier(to_string(t)), t);
    EXPECT_FALSE(parse_tier("fast").has_value());
    EXPECT_EQ(params(Tier::ultra).stride, 4);
    EXPECT_EQ(params(Tier::coarse).stride, 2);
    EXPECT_EQ(params(Tier::deep).stride, 1);
    EXPECT_EQ(params(Tier::ultra).refine_ratio, 0.0);
    EXPECT_EQ(params(Tier::coarse).refine_ratio, 1.25);
    EXPECT_EQ(params(Tier::deep).refine_ratio, 1.1);
    EXPECT_TRUE(params(Tier::deep).alternate_reverse);
    EXPECT_FALSE(params(Tier::coarse).alternate_reverse);
}

namespace {

std::vector<AxisSpec> trsm_axes() {
    return axis_specs({"side:exact", "trans:exact", "order:log:2", "q:log", "batch:log"},
                      {{"side", {"L", "R"}},
                       {"trans", {"N", "T"}},
                       {"order", {"1", "2", "4", "8", "12", "16", "24", "32", "48", "64", "96", "128", "192", "256",
                                  "384", "512", "768", "1024"}},
                       {"q", {"1", "2", "4", "8", "16", "32", "64", "128", "256", "512", "1024", "4096"}},
                       {"batch", {"128", "512", "2048", "8192", "32768"}}});
}

std::set<std::string> arg_set(const std::vector<CellKey>& v) {
    std::set<std::string> s;
    for (const auto& k : v) s.insert(key_arg(k));
    return s;
}

CellKey batch_cell(int batch) { return {{"uplo", "L"}, {"n", "64"}, {"batch", std::to_string(batch)}}; }

const std::vector<AxisSpec> kBatchAxes{{"uplo", false, {"L"}}, {"n", true, {"64"}}, {"batch", true, {}}};

}  // namespace

TEST(TuneGrid, LatticesNestAcrossTiers) {
    const auto axes = trsm_axes();
    ASSERT_EQ(axes.size(), 5u);
    EXPECT_FALSE(axes[0].log);
    EXPECT_TRUE(axes[2].log);
    const auto u = arg_set(tier_lattice(axes, Tier::ultra));
    const auto c = arg_set(tier_lattice(axes, Tier::coarse));
    const auto d = arg_set(tier_lattice(axes, Tier::deep));
    EXPECT_EQ(d.size(), 2u * 2 * 18 * 12 * 5);
    EXPECT_TRUE(std::includes(c.begin(), c.end(), u.begin(), u.end()));
    EXPECT_TRUE(std::includes(d.begin(), d.end(), c.begin(), c.end()));
    EXPECT_LT(u.size(), c.size());
    EXPECT_LT(c.size(), d.size());
}

TEST(TuneGrid, EndsAreAlwaysKept) {
    const auto u = tier_lattice(trsm_axes(), Tier::ultra);
    bool n_end = false, b_end = false;
    for (const auto& k : u) {
        n_end |= *key_get(k, "order") == "1024";
        b_end |= *key_get(k, "batch") == "32768";
    }
    EXPECT_TRUE(n_end);
    EXPECT_TRUE(b_end);
}

TEST(TuneGrid, ExactAxesAreNeverSubsampled) {
    std::set<std::string> sides, trans;
    for (const auto& k : tier_lattice(trsm_axes(), Tier::ultra)) {
        sides.insert(*key_get(k, "side"));
        trans.insert(*key_get(k, "trans"));
    }
    EXPECT_EQ(sides, (std::set<std::string>{"L", "R"}));
    EXPECT_EQ(trans, (std::set<std::string>{"N", "T"}));
}

TEST(TuneGrid, HashSubsampleNests) {
    std::vector<CellKey> g;
    for (int m = 1; m <= 47; ++m)
        for (int n = 1; n <= 32; ++n) g.push_back({{"m", std::to_string(m)}, {"n", std::to_string(n)}});
    ASSERT_EQ(g.size(), 1504u);
    const auto u = arg_set(tier_subsample(g, Tier::ultra));
    const auto c = arg_set(tier_subsample(g, Tier::coarse));
    const auto d = arg_set(tier_subsample(g, Tier::deep));
    EXPECT_EQ(d.size(), g.size());
    EXPECT_TRUE(std::includes(c.begin(), c.end(), u.begin(), u.end()));
    EXPECT_NEAR(double(u.size()), 1504 / 4.0, 1504 / 4.0 * 0.1);
    EXPECT_NEAR(double(c.size()), 1504 / 2.0, 1504 / 2.0 * 0.1);
}

TEST(TuneGrid, BisectsBatchToo) {
    const std::map<CellKey, std::vector<std::string>> ranked{{batch_cell(128), {"a", "b"}},
                                                             {batch_cell(8192), {"b", "a"}}};
    const auto next = refine_all_axes(ranked, kBatchAxes, 1.1).next;
    ASSERT_EQ(next.size(), 1u);
    EXPECT_EQ(key_arg(next[0]), "uplo=L,n=64,batch=1024");
    EXPECT_TRUE(refine_all_axes(ranked, kBatchAxes, 0).next.empty());
}

TEST(TuneGrid, AgreeingNeighboursAddNothing) {
    const std::map<CellKey, std::vector<std::string>> ranked{{batch_cell(128), {"a", "b"}}, {batch_cell(8192), {"a", "b"}}};
    const auto r = refine_all_axes(ranked, kBatchAxes, 1.1);
    EXPECT_TRUE(r.next.empty());
    EXPECT_TRUE(r.stalled.empty());
}

TEST(TuneGrid, EmptyRankedCellIsSkippedInItsLine) {
    const std::map<CellKey, std::vector<std::string>> ranked{
        {batch_cell(128), {}}, {batch_cell(512), {"a"}}, {batch_cell(8192), {"b"}}};
    const auto r = refine_all_axes(ranked, kBatchAxes, 1.1);
    ASSERT_EQ(r.next.size(), 1u);
    EXPECT_EQ(key_arg(r.next[0]), "uplo=L,n=64,batch=2048");
    EXPECT_TRUE(r.stalled.empty());
}

TEST(TuneGrid, MidpointWithoutWinnerIsReportedStalled) {
    const std::map<CellKey, std::vector<std::string>> ranked{
        {batch_cell(128), {"a"}}, {batch_cell(1024), {}}, {batch_cell(8192), {"b"}}};
    const auto r = refine_all_axes(ranked, kBatchAxes, 1.1);
    EXPECT_TRUE(r.next.empty());
    ASSERT_EQ(r.stalled.size(), 1u);
    EXPECT_NE(r.stalled[0].find("batch=1024 has no winner"), std::string::npos);
    EXPECT_NE(r.stalled[0].find("batch=128 (a)"), std::string::npos);
}

namespace {

// A raw sweep in the schema-1 layout (tools/tune/README.md "Raw JSONL"): keys "mode:exact n:log:2",
// two passes of 16 reps, a hidden `uplo` field the table does not key on.
struct SynCell {
    int n;
    std::map<std::string, double> ms;  // candidate -> time; absent = error in every pass
    int round = 0;
    std::string status = "ok";
};

std::string write_raw(const std::string& name, const std::vector<SynCell>& cells) {
    const std::string path = (scratch("replay") / name).string();
    std::ofstream f(path);
    f << Json().str("kind", "meta").integer("schema", 1).str("keys", "mode:exact n:log:2")
             .str("candidates", "a|b|c").integer("passes", 2).integer("reps", 16).line();
    for (const SynCell& c : cells) {
        auto base = [&](const char* kind) {
            return Json().str("kind", kind).str("mode", "x").integer("n", c.n).str("uplo", "L");
        };
        for (const char* cand : {"a", "b", "c"}) {
            const bool ok = c.ms.count(cand) != 0;
            for (int pass = 1; pass <= 2; ++pass) {
                std::vector<double> v;
                for (int rep = 0; rep < 16; ++rep) {
                    const double t = ok ? c.ms.at(cand) * (1 + 0.001 * ((rep * 7 + pass * 3) % 5)) : 0;
                    v.push_back(t);
                    if (ok && c.status == "ok")
                        f << base("rep").integer("pass", pass).integer("attempt", 0).str("cand", cand).integer("rep", rep).num("ms", t).line();
                }
                f << base("pass").integer("pass", pass).integer("attempt", 0).str("cand", cand).str("status", ok ? "ok" : "error")
                         .str("reason", "").num("median_ms", ok ? median(v) : NAN).line();
            }
        }
        f << base("cell").integer("round", c.round).str("status", c.status).integer("final_attempt", c.status == "ok" ? 0 : -1).line();
    }
    return path;
}

std::vector<SynCell> flip_cells() {  // a wins up to n=8, b from n=16
    std::vector<SynCell> v;
    for (int n : {1, 2, 4, 8, 16, 32, 64, 128}) v.push_back({n, {{"a", n <= 8 ? 1.0 : 1.5}, {"b", n <= 8 ? 1.5 : 1.0}}});
    return v;
}

}  // namespace

TEST(TuneReplay, LoaderKeepsTableKeysAndOnlyOkCells) {
    auto cells = flip_cells();
    cells.push_back({256, {}, 0, "skipped"});
    ReplayMeta meta;
    const auto rc = load_replay(write_raw("loader.jsonl", cells), &meta);
    ASSERT_EQ(rc.size(), 8u);
    EXPECT_EQ(key_arg(rc.front().key), "mode=x,n=1");
    ASSERT_EQ(rc.front().cands.size(), 2u);
    EXPECT_EQ(rc.front().rounds[0].size(), 32u);
    EXPECT_NEAR(rc.front().exhaustive.at("b"), 1.5, 0.01);
    ASSERT_EQ(meta.axes.size(), 2u);
    EXPECT_EQ(meta.axes[1].values.size(), 9u);  // the skipped round-0 cell stays on the lattice
    EXPECT_EQ(meta.axes[1].values.back(), "256");
    EXPECT_EQ(meta.axes[1].weight, 2.0);
    EXPECT_FALSE(meta.axes[0].log);
}

TEST(TuneReplay, ExhaustiveTierReproducesTheConverterRanking) {
    ReplayMeta meta;
    const auto rc = load_replay(write_raw("exhaustive.jsonl", {{1, {{"a", 1.0}, {"b", 1.5}}}, {2, {{"a", 1.5}, {"b", 1.0}}},
                                                               {3, {{"a", 1.2}, {"b", 1.0}, {"c", 3.0}}}}), &meta);
    TierParams p = params(Tier::deep);
    p.confidence = 1.0;
    p.min_reps = p.max_reps = 32;
    const ReplayReport r = replay(rc, meta.axes, Tier::deep, p);
    EXPECT_EQ(r.cells, 3u);
    EXPECT_EQ(r.cells_measured, 3u);
    EXPECT_EQ(r.reps_fraction, 1.0);
    EXPECT_EQ(r.race_misrank, 0.0);
    EXPECT_EQ(r.table_misrank, 0.0);
    EXPECT_EQ(r.refine_unavailable, 0u) << "no integer between n=1 and n=2";
}

TEST(TuneReplay, BisectionDropsMidpointsThatAreNotInTheRawFile) {
    ReplayMeta meta;
    const auto rc = load_replay(write_raw("bisect.jsonl", {{1, {{"a", 1.0}, {"b", 1.5}}}, {16, {{"a", 1.5}, {"b", 1.0}}}}), &meta);
    const ReplayReport r = replay(rc, meta.axes, Tier::deep);
    EXPECT_EQ(r.cells_measured, 2u);
    EXPECT_GT(r.refine_unavailable, 0u);
    EXPECT_LT(r.reps_fraction, 1.0) << "a 50% gap is eliminated before the cap";
}

TEST(TuneReplay, ShrinkingAnAxisShrinksTheLatticeAndCostsTheDroppedCells) {
    ReplayMeta meta;
    const auto rc = load_replay(write_raw("shrink.jsonl", flip_cells()), &meta);
    auto axes = meta.axes;
    shrink_axis(axes, "n", "2", false);  // 1,4,16,64,128
    EXPECT_EQ(axes[1].values, (std::vector<std::string>{"1", "4", "16", "64", "128"}));
    EXPECT_EQ(replay(rc, axes, Tier::ultra, [] { auto p = params(Tier::ultra); p.stride = 1; return p; }()).cells_measured, 5u);
    axes = meta.axes;
    shrink_axis(axes, "n", "1:8", true);
    EXPECT_EQ(axes[1].values.size(), 2u);
    TierParams p = params(Tier::ultra);
    p.stride = 1;
    const ReplayReport r = replay(rc, axes, Tier::ultra, p);
    EXPECT_EQ(r.cells_measured, 2u);
    EXPECT_GT(r.table_misrank, 0.0) << "n=16 and up read n=8, which loses there";
    EXPECT_THROW(shrink_axis(axes, "n", "3", true), std::invalid_argument);
    EXPECT_THROW(shrink_axis(axes, "zz", "2", false), std::invalid_argument);
    EXPECT_THROW(shrink_axis(axes, "n", "0", false), std::invalid_argument);
}

TEST(TuneReplay, TableMisrankCountsUnmeasuredCells) {
    ReplayMeta meta;
    const auto rc = load_replay(write_raw("flip.jsonl", flip_cells()), &meta);
    const ReplayReport u = replay(rc, meta.axes, Tier::ultra);  // measures n = 1, 16, 128
    EXPECT_EQ(u.cells_measured, 3u);
    EXPECT_EQ(u.race_misrank, 0.0);
    EXPECT_DOUBLE_EQ(u.table_misrank, 1.0 / 8);  // n=8 reads n=16's winner b; n=4 ties and reads n=1
    EXPECT_FALSE(u.worst.empty());
    EXPECT_NEAR(u.mean_loss, 0.5 / 8, 0.01);  // only n=8 loses, by 50%
    EXPECT_NEAR(u.max_loss, 0.5, 0.01);
    EXPECT_NEAR(u.p99_loss, 0.5, 0.01);
    EXPECT_EQ(u.p95_loss, u.max_loss);
    EXPECT_NEAR(u.time_weighted_loss, 0.5 / 8, 0.01);  // chosen 8.5 over a best total of 8
    EXPECT_DOUBLE_EQ(u.table_misrank_lattice, 1.0 / 8);
    EXPECT_EQ(u.unrunnable, 0u);
    EXPECT_EQ(replay(rc, meta.axes, Tier::deep).table_misrank, 0.0);
    const ReplayReport c = replay(rc, meta.axes, Tier::coarse);  // n = 1,4,16,64,128, then bisects to 8
    EXPECT_EQ(c.cells_measured, 6u);
    EXPECT_EQ(c.table_misrank, 0.0);
    EXPECT_GT(c.refine_unavailable, 0u);
}

TEST(TuneReplay, UnrunnableNearestWinnerFallsBackToTheNextEntry) {
    auto make = [](bool c_at_2) {
        std::map<std::string, double> two{{"a", 2.0}, {"b", 3.0}};
        if (c_at_2) two["c"] = 1.0;
        return std::vector<SynCell>{{1, {{"a", 2.0}, {"b", 3.0}, {"c", 1.0}}}, {2, two},
                                    {4, {{"a", 2.0}, {"b", 3.0}, {"c", 1.0}}}, {8, {{"a", 1.0}, {"b", 3.0}, {"c", 5.0}}},
                                    {16, {{"a", 1.0}, {"b", 3.0}, {"c", 5.0}}}};
    };
    ReplayMeta meta;
    auto rc = load_replay(write_raw("unrunnable.jsonl", make(false)), &meta);
    const ReplayReport r = replay(rc, meta.axes, Tier::ultra);
    // n=2 reads n=1's row (c, a, b): c cannot run, so the selector falls back to a (2.0 against a best of 2.0)
    EXPECT_EQ(r.table_misrank, 0.0);
    EXPECT_EQ(r.unrunnable, 0u);
    EXPECT_EQ(r.max_loss, 0.0);
    rc = load_replay(write_raw("runnable.jsonl", make(true)), &meta);
    EXPECT_EQ(replay(rc, meta.axes, Tier::ultra).table_misrank, 0.0);
}

TEST(TuneReplay, FallbackCanLoseAndNoRunnableEntryIsUnrunnable) {
    ReplayMeta meta;
    // n=1 ranks c, a, b. At n=2 c cannot run and a is 3x slower than b: the fallback picks a, a misrank.
    auto rc = load_replay(write_raw("fallback.jsonl", {{1, {{"a", 2.0}, {"b", 3.0}, {"c", 1.0}}}, {2, {{"a", 3.0}, {"b", 1.0}}},
                                                       {4, {{"a", 2.0}, {"b", 3.0}, {"c", 1.0}}}, {8, {{"a", 2.0}, {"b", 3.0}, {"c", 1.0}}},
                                                       {16, {{"a", 2.0}, {"b", 3.0}, {"c", 1.0}}}}), &meta);
    const ReplayReport r = replay(rc, meta.axes, Tier::ultra);  // measures n=1 and n=16; n=2 reads n=1
    EXPECT_EQ(r.unrunnable, 0u);
    EXPECT_DOUBLE_EQ(r.table_misrank, 1.0 / 5);
    EXPECT_NEAR(r.max_loss, 2.0, 0.02);
    // The measured rows list only a; the cells that run only b have no runnable entry.
    rc = load_replay(write_raw("noentry.jsonl", {{1, {{"a", 1.0}}}, {2, {{"b", 1.0}}}, {4, {{"b", 1.0}}}, {8, {{"b", 1.0}}}, {16, {{"a", 1.0}}}}), &meta);
    const ReplayReport u = replay(rc, meta.axes, Tier::ultra);
    EXPECT_EQ(u.unrunnable, 3u);
    EXPECT_DOUBLE_EQ(u.table_misrank, 3.0 / 5);
}

namespace {

// The row nearest_row picks among `rows`, each spelled (s, t, n, b).
std::size_t pick(const std::vector<std::vector<std::string>>& rows, const std::vector<std::string>& key) {
    const auto axes = axis_specs({"s:exact", "t:exact", "n:log:2", "b:log"},
                                 {{"s", {}}, {"t", {}}, {"n", {}}, {"b", {}}});
    auto to_key = [&](const std::vector<std::string>& v) {
        CellKey k;
        for (std::size_t i = 0; i < v.size(); ++i) k.push_back({axes[i].name, v[i]});
        return k;
    };
    std::vector<CellKey> r;
    for (const auto& row : rows) r.push_back(to_key(row));
    return nearest_row(r, to_key(key), axes);
}

}  // namespace

// Hand-computed against nearest() in scripts/sweep_to_table.py.
TEST(TuneReplay, NearestRowMatchesTheConverterRule) {
    // n weighs 2: for key n=8,b=8 row 0 costs 2*1+1 = 3 and row 1 costs 0+2 = 2.
    EXPECT_EQ(pick({{"L", "N", "4", "4"}, {"L", "N", "8", "2"}}, {"L", "N", "8", "8"}), 1u);
    // Exact keys drop from the right: no t=X row, so the pool is the s=L rows. n=4 is 0 away from row 0; n=7
    // costs 2*log2(7/4) = 1.6 from row 0 and 2*log2(8/7) = 0.38 from row 1.
    const std::vector<std::vector<std::string>> rows{{"L", "N", "4", "1"}, {"L", "T", "8", "1"}, {"R", "N", "8", "1"}};
    EXPECT_EQ(pick(rows, {"L", "X", "4", "1"}), 0u);
    EXPECT_EQ(pick(rows, {"L", "X", "7", "1"}), 1u);
    // Geometric midpoint tie (n=4 is 2 octaves from 1 and from 16): the smaller log key wins.
    EXPECT_EQ(pick({{"L", "N", "16", "1"}, {"L", "N", "1", "1"}}, {"L", "N", "4", "1"}), 1u);
    // No s=Z row: the pool is every row; an identical-log tie goes to the earlier row in (exact strings,
    // log integers) order, not to the caller's order.
    EXPECT_EQ(pick({{"R", "N", "4", "1"}, {"L", "N", "4", "1"}}, {"Z", "N", "4", "1"}), 1u);
}

namespace {

const std::vector<AxisSpec> kIndexAxes{{"mode", false, {"x"}}, {"n", true, {"1", "2", "4", "8", "16", "32", "64", "128"}}};

CellKey n_cell(int n) { return {{"mode", "x"}, {"n", std::to_string(n)}}; }

}  // namespace

TEST(TuneGrid, IndexModeRefillsTheLatticeWhereWinnersDiffer) {
    const RefineOpts index{RefineMode::index};
    std::map<CellKey, std::vector<std::string>> ranked;
    for (int n : {1, 16, 128}) ranked[n_cell(n)] = {n <= 8 ? "a" : "b"};
    auto round = refine_all_axes(ranked, kIndexAxes, 1.1, index);
    ASSERT_EQ(round.next.size(), 1u);
    EXPECT_EQ(key_arg(round.next[0]), "mode=x,n=4") << "index 2 between index 0 and 4, not the geometric 4 by luck";
    ranked[n_cell(4)] = {"a"};
    round = refine_all_axes(ranked, kIndexAxes, 1.1, index);
    ASSERT_EQ(round.next.size(), 1u);
    EXPECT_EQ(key_arg(round.next[0]), "mode=x,n=8");
    ranked[n_cell(8)] = {"a"};
    round = refine_all_axes(ranked, kIndexAxes, 1.1, index);  // 8 and 16 are adjacent: the geometric midpoint
    ASSERT_EQ(round.next.size(), 1u);
    EXPECT_EQ(key_arg(round.next[0]), "mode=x,n=11");
    EXPECT_TRUE(refine_all_axes(ranked, kIndexAxes, 0, index).next.size() == 0u) << "adjacent and ratio 0: nothing off the lattice";
}

TEST(TuneGrid, IndexModeSeesAFlipInsideAStrideFourBracket) {
    // Winners a at 1, a at 16: the geometric rule sees agreement. Only the margin trigger refines.
    const std::map<CellKey, std::vector<std::string>> ranked{{n_cell(1), {"a"}}, {n_cell(16), {"a"}}};
    EXPECT_TRUE(refine_all_axes(ranked, kIndexAxes, 1.1, {RefineMode::index}).next.empty());
    std::map<CellKey, double> close{{n_cell(1), 0.02}, {n_cell(16), 0.5}}, far{{n_cell(1), 0.30}, {n_cell(16), 0.5}};
    RefineOpts o{RefineMode::index, 0.05, &close};
    const auto r = refine_all_axes(ranked, kIndexAxes, 1.1, o);
    ASSERT_EQ(r.next.size(), 1u);
    EXPECT_EQ(key_arg(r.next[0]), "mode=x,n=4");
    o.gap = &far;
    EXPECT_TRUE(refine_all_axes(ranked, kIndexAxes, 1.1, o).next.empty());
    o.margin = 0.05, o.gap = &close;
    o.mode = RefineMode::geometric;
    EXPECT_EQ(key_arg(refine_all_axes(ranked, kIndexAxes, 1.1, o).next.at(0)), "mode=x,n=4") << "geometric midpoint of 1 and 16";
}
