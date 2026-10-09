// Host-only tests of the tuner core (tools/tune/tune_core.cc; flat-kernel-selection.md §6):
// the §6.3 tie rule and rotation, the §6.2 bisection, the JSONL records, the §6.4 hash, and
// the round trip through scripts/sweep_to_table.py --tuner. No GPU.

#include "../tools/tune/grid.hh"
#include "../tools/tune/ledger.hh"
#include "../tools/tune/race.hh"
#include "../tools/tune/replay_core.hh"
#include "../tools/tune/schedule.hh"
#include "../tools/tune/tiered_driver.hh"
#include "../tools/tune/tier.hh"
#include "../tools/tune/tune_core.hh"
#include "../tools/tune/worker.hh"

#include <gtest/gtest.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <mutex>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <thread>
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
    for (const char* op : {"potrf", "posv", "getrf", "getrs", "getri", "gesv", "gemv", "trmm", "symm", "syrk", "syr2k", "syev", "gesvd", "spmm"}) {
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

TEST(TuneHash, UnprefixedLinesAreCommon) {
    const std::string spec = "// kernel-sources-begin\n  \"a.cc\",\n  // family: cta\n  \"b.cc\",\n  // common\n"
                             "  \"c.hh\", \"d.hh\",\n  // family: tiny\n  \"e.cc\",\n  // family: cta\n  \"f.cc\",\n"
                             "// kernel-sources-end\n// kernel-deps-begin\n// family: cta \"x.cc\" \"y.cc\"\n"
                             "// family: cta \"z.cc\"\n// kernel-deps-end\n  \"late.cc\",\n";
    const KernelBlock b = parse_kernel_block(spec);
    EXPECT_EQ(b.all, (std::vector<std::string>{"a.cc", "b.cc", "c.hh", "d.hh", "e.cc", "f.cc"}));
    EXPECT_EQ(b.common, (std::vector<std::string>{"a.cc", "c.hh", "d.hh"}));
    EXPECT_EQ(b.family.at("cta"), (std::vector<std::string>{"b.cc", "f.cc"}));
    EXPECT_EQ(b.family.at("tiny"), (std::vector<std::string>{"e.cc"}));
    EXPECT_EQ(b.deps.at("cta"), (std::vector<std::string>{"x.cc", "y.cc", "z.cc"}));
    EXPECT_EQ(parse_kernel_list(spec), b.all);
}

TEST(TuneHash, MarkersAnchorToTheStartOfTheLine) {
    const std::string spec = "// kernel-sources-begin\n  // family: cta\n  \"a.cc\",  // family: tiny\n"
                             "  // common helpers\n  \"b.cc\",\n  // common\n  \"c.cc\",\n"
                             "// kernel-sources-end\n";
    const KernelBlock b = parse_kernel_block(spec);
    EXPECT_EQ(b.family.at("cta"), (std::vector<std::string>{"a.cc", "b.cc"}));
    EXPECT_EQ(b.family.count("tiny"), 0u);
    EXPECT_EQ(b.common, (std::vector<std::string>{"c.cc"}));
}

TEST(TuneHash, KernelBlockFromFileMergesDeps) {
    const fs::path d = scratch("blockfile");
    fs::create_directories(d / "tools/tune");
    std::ofstream(d / "k.hh") << "k";
    std::ofstream(d / "f.cc") << "f";
    std::ofstream(d / "dep.cc") << "dep";
    std::ofstream(d / "tools/tune/x_spec.cc")
        << "// kernel-sources-begin\n\"k.hh\",\n// family: fam\n\"f.cc\",\n// kernel-sources-end\n"
           "// kernel-deps-begin\n// family: fam \"dep.cc\"\n// kernel-deps-end\n";
    const KernelBlock b = kernel_block_from_file(d.string(), "tools/tune/x_spec.cc");
    EXPECT_EQ(b.common, (std::vector<std::string>{"k.hh"}));
    EXPECT_EQ(b.family.at("fam"), (std::vector<std::string>{"f.cc"}));
    EXPECT_EQ(b.deps.at("fam"), (std::vector<std::string>{"dep.cc"}));
    const auto h = family_hashes(d.string(), b, {"fam"});
    EXPECT_EQ(h.at("fam"), *kernel_hash(d.string(), {"k.hh", "f.cc", "dep.cc"}));
    try {
        kernel_block_from_file(d.string(), "tools/tune/nope_spec.cc");
        FAIL() << "expected a throw";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("tools/tune/nope_spec.cc"), std::string::npos);
    }
}

TEST(TuneHash, DepsCommonJoinsEveryFamilyButNotTheOpHash) {
    const fs::path d = scratch("depscommon");
    for (const char* f : {"k.hh", "cta.cc", "op.cc", "dep.cc"}) std::ofstream(d / f) << f;
    const std::string spec = "// kernel-sources-begin\n\"k.hh\",\n// family: cta\n\"cta.cc\",\n// kernel-sources-end\n"
                             "// kernel-deps-begin\n// common \"op.cc\"\n// common helpers \"not.cc\"\n"
                             "// family: cta \"dep.cc\"\n// kernel-deps-end\n";
    const KernelBlock b = parse_kernel_block(spec);
    EXPECT_EQ(b.all, (std::vector<std::string>{"k.hh", "cta.cc"})) << "the deps block never moves the op-level hash";
    EXPECT_EQ(b.deps_common, (std::vector<std::string>{"op.cc"})) << "only a bare '// common' line counts";
    const std::vector<std::string> fams{"cta", "vendor"};
    const auto h0 = family_hashes(d.string(), b, fams);
    EXPECT_EQ(h0.at("cta"), *kernel_hash(d.string(), {"k.hh", "cta.cc", "op.cc", "dep.cc"}));
    EXPECT_EQ(h0.at("vendor"), *kernel_hash(d.string(), {"k.hh", "op.cc"}));
    std::ofstream(d / "op.cc") << "can_run edited";
    const auto h1 = family_hashes(d.string(), b, fams);
    for (const char* f : {"cta", "vendor"}) EXPECT_NE(h1.at(f), h0.at(f)) << f;
}

TEST(TuneHash, EditingOneFamilyFileChangesOnlyThatFamily) {
    const fs::path d = scratch("famhash");
    std::ofstream(d / "common.hh") << "c";
    std::ofstream(d / "cta.cc") << "cta";
    std::ofstream(d / "tiny.cc") << "tiny";
    std::ofstream(d / "dep.cc") << "dep";
    KernelBlock b;
    b.all = {"common.hh", "cta.cc", "tiny.cc"};
    b.common = {"common.hh"};
    b.family = {{"cta", {"cta.cc"}}, {"tiny", {"tiny.cc"}}};
    b.deps = {{"cta", {"dep.cc"}}};
    const std::vector<std::string> fams{"cta", "tiny", "vendor"};
    const auto h0 = family_hashes(d.string(), b, fams);
    ASSERT_EQ(h0.size(), 3u);
    EXPECT_EQ(h0.at("cta"), *kernel_hash(d.string(), {"common.hh", "cta.cc", "dep.cc"}));
    EXPECT_EQ(h0.at("vendor"), *kernel_hash(d.string(), {"common.hh"}));
    std::ofstream(d / "cta.cc") << "cta edited";
    const auto h1 = family_hashes(d.string(), b, fams);
    EXPECT_NE(h1.at("cta"), h0.at("cta"));
    EXPECT_EQ(h1.at("tiny"), h0.at("tiny"));
    EXPECT_EQ(h1.at("vendor"), h0.at("vendor"));
    std::ofstream(d / "dep.cc") << "dep edited";
    const auto h2 = family_hashes(d.string(), b, fams);
    EXPECT_NE(h2.at("cta"), h1.at("cta"));
    EXPECT_EQ(h2.at("tiny"), h0.at("tiny"));
    std::ofstream(d / "common.hh") << "c edited";
    const auto h3 = family_hashes(d.string(), b, fams);
    for (const char* f : {"cta", "tiny", "vendor"}) EXPECT_NE(h3.at(f), h2.at(f)) << f;
    fs::remove(d / "tiny.cc");
    EXPECT_THROW(family_hashes(d.string(), b, fams), std::runtime_error);
}

// Values computed before the family annotations were added: comment markers must not move the
// op-level hash that CMake and CI compare against the table headers.
TEST(TuneHash, OpLevelHashUnchangedByFamilyPrefixes) {
    const std::string repo = BATCHLAS_TUNE_SOURCE_DIR;
    const std::map<std::string, std::string> pinned{
        {"gemm", "b8d9ff55"}, {"potrf", "f4043f40"}, {"posv", "b905a231"}, {"trsm", "6446e809"}};
    for (const auto& [op, want] : pinned) {
        const KernelBlock b = parse_kernel_block(read_file(fs::path(repo) / "tools/tune" / (op + "_spec.cc")));
        EXPECT_EQ(kernel_hash(repo, b.all), want) << op;
        for (const auto& [fam, files] : b.family)
            for (const auto& f : files)
                EXPECT_NE(std::find(b.all.begin(), b.all.end(), f), b.all.end()) << op << " " << fam << " " << f;
        for (const auto& [fam, files] : b.deps) {
            EXPECT_FALSE(files.empty()) << op << " " << fam;
            EXPECT_TRUE(kernel_hash(repo, files)) << op << " " << fam;
        }
    }
}

TEST(TuneHash, SpecsDeclareTheirFamilies) {
    const std::string repo = BATCHLAS_TUNE_SOURCE_DIR;
    const auto block = [&](const char* op) {
        return parse_kernel_block(read_file(fs::path(repo) / "tools/tune" / (std::string(op) + "_spec.cc")));
    };
    // gemm direct lives in gemm_kernels.cc with the shared launch code: common only.
    for (const char* f : {"tiny", "cta", "lpanel", "blocked"}) EXPECT_TRUE(block("potrf").family.count(f)) << f;
    for (const char* f : {"tiled", "small", "reg", "wide"}) EXPECT_TRUE(block("gemm").family.count(f)) << f;
    EXPECT_TRUE(block("trsm").family.count("sg_left"));
    const KernelBlock posv = block("posv");
    for (const char* op : {"gemm", "potrf", "posv", "trsm", "syev", "gesvd", "spmm"})  // can_run lives there
        EXPECT_EQ(block(op).deps_common, std::vector<std::string>{"src/ops/" + std::string(op) + "/" + op + ".cc"}) << op;
    for (const char* f : {"cta", "blocked"}) {
        ASSERT_TRUE(posv.deps.count(f)) << f;
        const auto& d = posv.deps.at(f);
        EXPECT_NE(std::find(d.begin(), d.end(), "src/extensions/potrf_cta.cc"), d.end()) << f;
        EXPECT_NE(std::find(d.begin(), d.end(), "src/sycl/trsm_native.cc"), d.end()) << f;
    }
    // LU: getrf_cta.cc holds cta's kernel and blocked's panel leaf, so it is common, not a family.
    for (const char* op : {"getrf", "getrs", "getri", "gesv"}) {
        const KernelBlock b = block(op);
        EXPECT_EQ(b.deps_common, std::vector<std::string>{"src/ops/" + std::string(op) + "/" + op + ".cc"}) << op;
        for (const auto& [fam, files] : b.deps) {
            EXPECT_FALSE(files.empty()) << op << " " << fam;
            EXPECT_TRUE(kernel_hash(repo, files)) << op << " " << fam;
        }
    }
    for (const char* f : {"tiny", "blocked"}) EXPECT_TRUE(block("getrf").family.count(f)) << f;
    const auto& getrf_common = block("getrf").common;
    EXPECT_NE(std::find(getrf_common.begin(), getrf_common.end(), "src/extensions/getrf_cta.cc"), getrf_common.end());
    for (const char* f : {"cta", "blocked"}) EXPECT_TRUE(block("getrs").family.count(f)) << f;
    EXPECT_TRUE(block("getri").family.count("blocked"));
    EXPECT_TRUE(block("gesv").family.count("tiny"));
    const auto has = [](const KernelBlock& b, const char* fam, const char* file) {
        if (!b.deps.count(fam)) return false;
        const auto& d = b.deps.at(fam);
        return std::find(d.begin(), d.end(), file) != d.end();
    };
    EXPECT_TRUE(has(block("getrf"), "blocked", "src/sycl/trsm_native.cc"));
    EXPECT_TRUE(has(block("getrf"), "blocked", "src/sycl/gemm_kernels.cc"));
    EXPECT_TRUE(has(block("getrs"), "blocked", "src/sycl/trsm_native.cc"));
    EXPECT_TRUE(has(block("getri"), "blocked", "src/sycl/trsm_native.cc"));
    for (const char* file : {"src/extensions/getrf_tiny.cc", "src/extensions/getrf_cta.cc", "src/extensions/getrs_fused.cc",
                             "src/extensions/getrs_native.cc", "src/sycl/trsm_native.cc", "src/sycl/gemm_kernels.cc"})
        EXPECT_TRUE(has(block("gesv"), "blocked", file)) << file;
    // gemv cta and direct share gemv_native.cc: common only, like gemm direct.
    for (const char* op : {"gemv", "trmm", "symm", "syrk", "syr2k"}) {
        const auto dc = block(op).deps_common;
        EXPECT_NE(std::find(dc.begin(), dc.end(), "src/ops/" + std::string(op) + "/" + op + ".cc"), dc.end()) << op;
    }
    for (const char* f : {"triangular", "expand"}) EXPECT_TRUE(block("trmm").family.count(f)) << f;
    EXPECT_TRUE(block("symm").family.count("expand"));
    for (const char* f : {"gram", "triangular"}) EXPECT_TRUE(block("syrk").family.count(f)) << f;
    EXPECT_TRUE(block("syr2k").family.count("triangular"));
    // syrk's two tile kernels share triangular_tiles.hh: common, never one family's.
    const auto syrk_common = block("syrk").common;
    EXPECT_NE(std::find(syrk_common.begin(), syrk_common.end(), "src/backends/triangular_tiles.hh"), syrk_common.end());
    for (const char* op : {"trmm", "symm"}) {  // expand calls the public gemm: gemm's kernels and can_run count
        const KernelBlock b = block(op);
        ASSERT_TRUE(b.deps.count("expand")) << op;
        const auto& d = b.deps.at("expand");
        for (const char* g : {"src/sycl/gemm_kernels.cc", "src/ops/gemm/gemm.cc"})
            EXPECT_NE(std::find(d.begin(), d.end(), g), d.end()) << op << " " << g;
    }
    for (const char* f : {"cta", "cta_fused", "jacobi", "blocked", "two_stage"}) EXPECT_TRUE(block("syev").family.count(f)) << f;
    for (const char* f : {"jacobi", "cta", "blocked"}) EXPECT_TRUE(block("gesvd").family.count(f)) << f;
    EXPECT_TRUE(block("spmm").family.count("direct"));
    // A file two families share is common, never one family's (syev blocked and two_stage both run stedc).
    for (const char* op : {"syev", "gesvd", "spmm"}) {
        const KernelBlock b = block(op);
        std::map<std::string, int> owners;
        for (const auto& [fam, files] : b.family)
            for (const auto& f : files) ++owners[f];
        for (const auto& [f, n] : owners) {
            EXPECT_EQ(n, 1) << op << " " << f;
            EXPECT_EQ(std::find(b.common.begin(), b.common.end(), f), b.common.end()) << op << " " << f;
        }
    }
    const KernelBlock syev = block("syev");
    EXPECT_NE(std::find(syev.common.begin(), syev.common.end(), "src/extensions/stedc.cc"), syev.common.end());
    for (const char* f : {"blocked", "two_stage"}) {  // both call the public gemm: its kernels count too
        ASSERT_TRUE(syev.deps.count(f)) << f;
        const auto& d = syev.deps.at(f);
        EXPECT_NE(std::find(d.begin(), d.end(), "src/sycl/gemm_kernels.cc"), d.end()) << f;
    }
}

// The QR specs: every listed file exists, every family hashes, can_run's <op>.cc is the one common
// dep, and the composed sub-ops are declared (geqrf blocked -> gemm; orgqr blocked -> ormqr -> gemm,
// trmm; ormqr blocked -> gemm, trmm).
TEST(TuneHash, QrSpecsDeclareTheirFamiliesAndCompositions) {
    const std::string repo = BATCHLAS_TUNE_SOURCE_DIR;
    const std::map<std::string, std::vector<std::string>> families{
        {"geqrf", {"tiny", "cta", "blocked", "vendor"}}, {"orgqr", {"blocked", "vendor"}}, {"ormqr", {"blocked", "vendor"}}};
    const std::map<std::string, std::vector<std::string>> blocked_deps{
        {"geqrf", {"src/ops/gemm/gemm.cc", "src/sycl/gemm_kernels.cc"}},
        {"orgqr", {"src/ops/ormqr/ormqr.cc", "src/extensions/ormqr_blocked.cc", "src/sycl/gemm_kernels.cc",
                   "src/ops/trmm/trmm.cc"}},
        {"ormqr", {"src/sycl/gemm_kernels.cc", "src/ops/trmm/trmm.cc", "include/batchlas/tuning_params.hh"}}};
    for (const auto& [op, fams] : families) {
        const std::string spec = "tools/tune/" + op + "_spec.cc";
        const KernelBlock b = kernel_block_from_file(repo, spec);
        std::string missing;
        const auto op_hash = kernel_hash(repo, b.all, &missing);
        ASSERT_TRUE(op_hash) << op << ": " << missing;
        if (const std::string py = BATCHLAS_TUNE_PYTHON; !py.empty())
            EXPECT_EQ(run("cd '" + repo + "' && '" + py +
                          "' -c \"import sys; sys.path.insert(0, '.github/ci'); import check_tuned_tables as c; "
                          "print(c.kernel_hash('.', c.parse_kernel_list(open('" + spec + "').read()))[0])\""),
                      *op_hash) << op;
        EXPECT_EQ(b.deps_common.front(), "src/ops/" + op + "/" + op + ".cc") << op;
        EXPECT_TRUE(b.family.count("blocked")) << op;
        const auto h = family_hashes(repo, b, fams);
        EXPECT_EQ(h.size(), fams.size()) << op;
        const auto& d = b.deps.at("blocked");
        for (const std::string& f : blocked_deps.at(op)) EXPECT_NE(std::find(d.begin(), d.end(), f), d.end()) << op << " " << f;
        for (const auto& [fam, files] : b.family)
            for (const std::string& f : files) EXPECT_EQ(std::count(b.all.begin(), b.all.end(), f), 1) << op << " " << f;
    }
    for (const char* f : {"tiny", "cta"})
        EXPECT_TRUE(kernel_block_from_file(repo, "tools/tune/geqrf_spec.cc").family.count(f)) << f;
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

// A ledger written by LedgerWriter becomes a table: deep and coarse rows kept, a preview row
// inside the deep bracket dropped, the header carries the C++ family hashes (the Python port of
// the kernel-block manifest rule must agree with family_hashes), and --check re-derives it.
TEST(TuneConverter, LedgerRoundTripsThroughSweepToTable) {
    const std::string py = BATCHLAS_TUNE_PYTHON;
    if (py.empty()) GTEST_SKIP() << "no python3";
    const std::string repo = BATCHLAS_TUNE_SOURCE_DIR;
    const KernelBlock block = kernel_block_from_file(repo, "tools/tune/posv_spec.cc");
    const std::vector<std::string> cands{"tiny", "cta", "blocked"};
    const auto fh = family_hashes(repo, block, cands);
    const fs::path d = scratch("ledger");
    fs::remove_all(d / "out");
    const std::string dir = ledger_dir((d / "ledger").string(), "posv", "float", "sm_999");
    fs::remove_all(dir);
    const std::string keys = "uplo:exact n:log:3 nrhs:log batch:log";
    auto write = [&](Tier tier, const std::string& id, const std::string& date, const std::vector<int>& ns) {
        RunMeta m;
        m.run_id = id;
        m.host = "box";
        m.device = "sm_999";
        m.batchlas = "abc12345";
        m.date = date;
        m.tier = tier;
        m.keys = keys;
        m.candidates = "tiny|cta|blocked";
        LedgerWriter w(dir, m);
        for (int n : ns) {
            CellRecord c;
            c.key = {{"uplo", "L"}, {"n", std::to_string(n)}, {"nrhs", "1"}, {"batch", "128"}};
            c.date = date;
            for (const auto& name : cands) {
                CandResult r;
                r.cand = name;
                r.hash = fh.at(name);
                r.status = name == "blocked" ? "skipped" : "ok";
                if (r.status == "ok") r.median_ms = name == "cta" ? 0.5 : 1.25, r.lo = r.median_ms, r.hi = r.median_ms, r.reps = 8;
                c.cands.push_back(r);
            }
            c.ranked = {"cta", "tiny"};
            w.cell(c);
        }
    };
    write(Tier::deep, "20261001T000000-box-1", "2026-10-01", {64});
    write(Tier::preview, "20261002T000000-box-2", "2026-10-02", {8, 16, 32, 128, 512});
    const std::string conv = repo + "/scripts/sweep_to_table.py";
    ASSERT_EQ(run("'" + py + "' '" + conv + "' --ledger '" + dir + "' --out '" + (d / "out").string() + "'")
                  .rfind("FAILED", 0), std::string::npos);
    const std::string table = read_file(d / "out/posv.float.sm_999.txt");
    std::string fam;
    for (const auto& name : cands) fam += (fam.empty() ? "" : ",") + name + ":" + fh.at(name);
    EXPECT_NE(table.find("device=sm_999 batchlas=abc12345 kernels="), std::string::npos) << table;
    EXPECT_NE(table.find(" family_kernels=" + fam + " date=2026-10-02\n"), std::string::npos) << table;
    EXPECT_NE(table.find(" tiers=deep:1,coarse:0,preview:3,custom:0,transcribed:0\n"), std::string::npos) << table;
    EXPECT_NE(table.find("uplo=L n=64 nrhs=1 batch=128 | cta 0.5000 | tiny 1.250 # deep\n"), std::string::npos) << table;
    EXPECT_NE(table.find("n=8 nrhs=1 batch=128 | cta 0.5000 | tiny 1.250 # preview\n"), std::string::npos) << table;
    EXPECT_NE(table.find("n=512 nrhs=1 batch=128 | cta 0.5000 | tiny 1.250 # preview\n"), std::string::npos) << table;
    EXPECT_NE(table.find("n=16 nrhs=1 batch=128 | cta 0.5000 | tiny 1.250 # preview\n"), std::string::npos) << table;
    for (const char* gone : {"n=32 ", "n=128 "})
        EXPECT_EQ(table.find(gone), std::string::npos) << gone << " sits next to the deep row and is dropped:\n" << table;
    const std::string chk = run("'" + py + "' '" + conv + "' --self-test");
    EXPECT_NE(chk.find("--self-test: OK"), std::string::npos) << chk;
    EXPECT_EQ(run("cd '" + repo + "' && '" + py + "' -c \"import sys; sys.path.insert(0, 'scripts'); import sweep_to_table as s; "
                  "b = s.parse_kernel_block(open('tools/tune/posv_spec.cc').read()); "
                  "print(','.join(f + ':' + h for f, h in s.family_hashes(s.REPO, b, ['tiny', 'cta', 'blocked']).items()))\""),
              fam);
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
    for (const char* op : {"potrf", "posv", "getrf", "getrs", "getri", "gesv", "gemv", "trmm", "symm", "syrk", "syr2k"}) {
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

// evidence: docs/design/tiered-tuning.md#engine-guard-retries-a-failed-nvidia-smi-query
TEST(TuneGuard, AFailedQueryRetriesWithDoublingBackoffUntilTheWaitExpires) {
    std::vector<double> slept;
    const auto sleep = [&](double s) { slept.push_back(s); };
    int calls = 0;
    const auto fails_until = [&](int ok_at) {
        return [&calls, ok_at]() -> std::optional<std::string> {
            return ++calls >= ok_at ? std::optional<std::string>("123\n") : std::nullopt;
        };
    };
    EXPECT_EQ(query_with_backoff(fails_until(1), 300, sleep), "123\n");
    EXPECT_TRUE(slept.empty()) << "a working query never sleeps";

    calls = 0;
    EXPECT_EQ(query_with_backoff(fails_until(4), 300, sleep), "123\n") << "a reset that ends within the wait";
    EXPECT_EQ(slept, (std::vector<double>{5, 10, 20}));
    EXPECT_EQ(calls, 4);

    calls = 0, slept.clear();
    EXPECT_EQ(query_with_backoff(fails_until(1000), 300, sleep), std::nullopt);
    EXPECT_EQ(slept, (std::vector<double>{5, 10, 20, 40, 60, 60, 60, 45})) << "capped at 60 s, clipped to the wait";
    EXPECT_EQ(calls, 9) << "one last query once the wait is spent";

    calls = 0, slept.clear();
    EXPECT_EQ(query_with_backoff(fails_until(1000), 0, sleep), std::nullopt);
    EXPECT_TRUE(slept.empty()) << "--guard-wait 0: one try";
    EXPECT_EQ(calls, 1);
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

// Fixed noise within +-0.25%: a 4% gap stays above the 3% tie (worst 1.04 x 0.9975 / 1.0025 = 1.0379)
// and a 2% gap below it (worst 1.02 x 1.0025 / 0.9975 = 1.0251), whatever the library's RNG.
constexpr double kNoise[] = {0.0025, -0.0011, 0.0019, -0.0025, 0.0004, -0.0018, 0.0023, -0.0006,
                             0.0012, -0.0021, 0.0008, -0.0014, 0.0025, -0.0025, 0.0001, -0.0009};
double noise(int i) { return 1 + kNoise[i % 16]; }

}  // namespace

TEST(TuneRace, FourPercentSlowerIsEliminatedInDeep) {
    const auto r = run_race(params(Tier::deep), [&](int i) { return std::pair(1.00 * noise(i), 1.04 * noise(i + 5)); });
    EXPECT_FALSE(r.s.alive[1]);
    EXPECT_TRUE(r.s.alive[0]);
    EXPECT_EQ(r.v, RaceVerdict::winner);
}

TEST(TuneRace, TwoPercentSlowerIsKeptAsATie) {
    const auto r = run_race(params(Tier::deep), [&](int i) { return std::pair(1.00 * noise(i), 1.02 * noise(i + 5)); });
    EXPECT_TRUE(r.s.alive[1]);
    EXPECT_TRUE(r.v == RaceVerdict::tie || r.v == RaceVerdict::cap);
    EXPECT_EQ(race_ranking(r.s, {"b", "a"}), (std::vector<std::string>{"b", "a"}));
    EXPECT_EQ(race_ranking(r.s, {"a", "b"}), (std::vector<std::string>{"a", "b"}));
}

TEST(TuneRace, OneDiesAndTheOtherTwoTieInOneStep) {
    const auto& p = params(Tier::preview);  // min_reps 3, k = 1 at 3 pairs
    RaceState s;
    s.cands = {"a", "b", "c"};
    s.alive = {true, true, true};
    s.ms = {{1.0, 1.0}, {1.10, 1.10}, {1.01, 1.01}};
    EXPECT_EQ(race_step(s, p), RaceVerdict::more);
    s.ms = {{1.0, 1.0, 1.0}, {1.10, 1.10, 1.10}, {1.01, 1.01, 1.01}};
    EXPECT_EQ(race_step(s, p), RaceVerdict::tie) << "b is 10% slower, c within the tie of a";
    EXPECT_EQ(s.alive, (std::vector<bool>{true, false, true}));
    EXPECT_EQ(race_ranking(s, {"a", "b", "c"}), (std::vector<std::string>{"a", "c", "b"}));
}

TEST(TuneRace, AnArmAloneOrBeatenByDefaultStillRunsMinReps) {
    const auto nan = std::numeric_limits<double>::quiet_NaN();
    for (Tier t : {Tier::preview, Tier::coarse, Tier::deep}) {
        const auto& p = params(t);
        RaceState alone;
        alone.cands = {"a"};
        alone.alive = {true};
        alone.ms = {{}};
        int rounds = 0;
        do alone.ms[0].push_back(1.0);
        while (!race_over(race_step(alone, p), ++rounds, p) && rounds < 100);
        EXPECT_EQ(rounds, p.min_reps) << to_string(t) << ": race_step says winner after one round";
        RaceState failed;  // b failed in round 0, as run_race marks it: a wins by default
        failed.cands = {"a", "b"};
        failed.alive = {true, false};
        failed.ms = {{}, {}};
        rounds = 0;
        do failed.ms[0].push_back(1.0), failed.ms[1].push_back(nan);
        while (!race_over(race_step(failed, p), ++rounds, p) && rounds < 100);
        EXPECT_EQ(rounds, p.min_reps) << to_string(t);
    }
    EXPECT_FALSE(race_over(RaceVerdict::more, 50, params(Tier::preview)));
    EXPECT_TRUE(race_over(RaceVerdict::tie, 1, params(Tier::preview)));
    EXPECT_TRUE(race_over(RaceVerdict::cap, 1, params(Tier::preview)));
}

TEST(TuneRace, SingleRoundNeverEliminatesBelowTheGrossLoserRatio) {
    for (Tier t : {Tier::preview, Tier::coarse, Tier::deep}) {
        const auto& p = params(t);
        const auto r = run_race(p, [&](int) { return std::pair(1.0, 3.9); });
        EXPECT_EQ(r.rounds, p.min_reps) << to_string(t);
        EXPECT_FALSE(r.s.alive[1]) << to_string(t);
        RaceState s;
        s.cands = {"a", "b"};
        s.ms = {{1.0}, {3.9}};
        s.alive = {true, true};
        EXPECT_EQ(race_step(s, p), RaceVerdict::more);
        EXPECT_TRUE(s.alive[1]);
    }
}

TEST(TuneRace, GrossLoserDiesAfterTheFirstRoundInEveryTier) {
    for (Tier t : {Tier::preview, Tier::coarse, Tier::deep}) {
        const auto& p = params(t);
        RaceState s;
        s.cands = {"a", "b", "c"};
        s.ms = {{1.0}, {4.1}, {3.9}};
        s.alive = {true, true, true};
        EXPECT_EQ(race_step(s, p), RaceVerdict::more) << to_string(t) << ": c is under the ratio and stays";
        EXPECT_EQ(s.alive, (std::vector<bool>{true, false, true})) << to_string(t);
        EXPECT_EQ(race_ranking(s, {"a", "b", "c"}), (std::vector<std::string>{"a", "c", "b"}));
        RaceState later;  // the median paired ratio decides: one slow round among three is no gross loss
        later.cands = {"a", "b"};
        later.ms = {{1.0, 1.0, 1.0}, {1.5, 9.0, 1.5}};
        later.alive = {true, true};
        (void)race_step(later, p);
        EXPECT_TRUE(later.alive[1] || p.min_reps <= 3) << to_string(t);
        later.ms = {{1.0, 1.0, 1.0}, {4.5, 1.5, 4.5}};
        later.alive = {true, true};
        (void)race_step(later, p);
        EXPECT_FALSE(later.alive[1]) << to_string(t) << ": median paired ratio 4.5";
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
    EXPECT_EQ(race_step(s, params(Tier::preview)), RaceVerdict::more);  // 2 pairs < min_reps 3
    EXPECT_TRUE(s.alive[1]);
    s.ms = {{1, 1, 1, 1, 1}, {nan, 2, nan, 2, 2}};
    EXPECT_EQ(race_step(s, params(Tier::preview)), RaceVerdict::winner);  // 3 pairs
    EXPECT_FALSE(s.alive[1]);
}

TEST(TuneRace, AllNaNRoundsNeverTie) {
    const auto nan = std::numeric_limits<double>::quiet_NaN();
    const auto& p = params(Tier::preview);
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
    EXPECT_LT(tier_rank(Tier::preview), tier_rank(Tier::coarse));
    EXPECT_LT(tier_rank(Tier::custom), tier_rank(Tier::preview));
    EXPECT_LT(tier_rank(Tier::transcribed), tier_rank(Tier::custom));
    for (Tier t : {Tier::transcribed, Tier::custom, Tier::preview, Tier::coarse, Tier::deep})
        EXPECT_EQ(parse_tier(to_string(t)), t);
    EXPECT_FALSE(parse_tier("fast").has_value());
    EXPECT_EQ(params(Tier::preview).stride, 2);
    EXPECT_EQ(params(Tier::coarse).stride, 1);
    EXPECT_EQ(params(Tier::deep).stride, 1);
    for (Tier t : {Tier::preview, Tier::coarse, Tier::deep}) EXPECT_EQ(params(t).refine_ratio, 1.1);
    EXPECT_EQ(params(Tier::preview).refine_mode, RefineMode::index);
    EXPECT_EQ(params(Tier::preview).refine_margin, 0.10);
    EXPECT_EQ(params(Tier::preview).refine_cap_factor, 3.0);
    EXPECT_EQ(params(Tier::coarse).refine_cap_factor, 1.0);
    EXPECT_EQ(params(Tier::deep).refine_cap_factor, 2.0);
    EXPECT_EQ(params(Tier::coarse).refine_mode, RefineMode::geometric);
    EXPECT_EQ(params(Tier::deep).refine_margin, 0.0);
    EXPECT_EQ(params(Tier::preview).max_reps, 6);
    EXPECT_EQ(params(Tier::coarse).confidence, 0.90);
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

// kBatchAxes with batch lattice values: batch only ever refills these.
std::vector<AxisSpec> batch_lattice(const std::vector<int>& batches) {
    std::vector<AxisSpec> axes = kBatchAxes;
    for (int b : batches) axes[2].values.push_back(std::to_string(b));
    return axes;
}

}  // namespace

TEST(TuneGrid, LatticesNestAcrossTiers) {
    const auto axes = trsm_axes();
    ASSERT_EQ(axes.size(), 5u);
    EXPECT_FALSE(axes[0].log);
    EXPECT_TRUE(axes[2].log);
    const auto u = arg_set(tier_lattice(axes, Tier::preview));
    const auto c = arg_set(tier_lattice(axes, Tier::coarse));
    const auto d = arg_set(tier_lattice(axes, Tier::deep));
    EXPECT_EQ(d.size(), 2u * 2 * 18 * 12 * 5);
    EXPECT_TRUE(std::includes(c.begin(), c.end(), u.begin(), u.end()));
    EXPECT_TRUE(std::includes(d.begin(), d.end(), c.begin(), c.end()));
    EXPECT_LT(u.size(), c.size());
    EXPECT_EQ(c, d) << "coarse keeps the full lattice and differs from deep in reps and audit only";
}

TEST(TuneGrid, EndsAreAlwaysKept) {
    const auto u = tier_lattice(trsm_axes(), Tier::preview);
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
    for (const auto& k : tier_lattice(trsm_axes(), Tier::preview)) {
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
    const auto u = arg_set(tier_subsample(g, Tier::preview));
    const auto c = arg_set(tier_subsample(g, Tier::coarse));
    const auto d = arg_set(tier_subsample(g, Tier::deep));
    EXPECT_EQ(d.size(), g.size());
    EXPECT_TRUE(std::includes(c.begin(), c.end(), u.begin(), u.end()));
    EXPECT_NEAR(double(u.size()), 1504 / 2.0, 1504 / 2.0 * 0.1);
    EXPECT_EQ(c.size(), g.size());
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
    const auto r = refine_all_axes(ranked, batch_lattice({128, 512, 2048, 8192}), 1.1);
    ASSERT_EQ(r.next.size(), 1u);
    EXPECT_EQ(key_arg(r.next[0]), "uplo=L,n=64,batch=2048");
    EXPECT_TRUE(r.stalled.empty());
}

TEST(TuneGrid, MidpointWithoutWinnerIsReportedStalled) {
    const std::map<CellKey, std::vector<std::string>> ranked{
        {batch_cell(128), {"a"}}, {batch_cell(1024), {}}, {batch_cell(8192), {"b"}}};
    const auto r = refine_all_axes(ranked, batch_lattice({128, 1024, 8192}), 1.1);
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

namespace {

// The pre-decision ultra protocol: every 4th point, no bisection. The synthetic cases below depend on it.
TierParams sparse() {
    TierParams p = params(Tier::preview);
    p.stride = 4, p.refine_ratio = 0, p.refine_mode = RefineMode::geometric, p.refine_margin = 0;
    return p;
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
    EXPECT_EQ(replay(rc, axes, Tier::preview, [] { auto p = sparse(); p.stride = 1; return p; }()).cells_measured, 5u);
    axes = meta.axes;
    shrink_axis(axes, "n", "1:8", true);
    EXPECT_EQ(axes[1].values.size(), 2u);
    TierParams p = sparse();
    p.stride = 1;
    const ReplayReport r = replay(rc, axes, Tier::preview, p);
    EXPECT_EQ(r.cells_measured, 2u);
    EXPECT_GT(r.table_misrank, 0.0) << "n=16 and up read n=8, which loses there";
    EXPECT_THROW(shrink_axis(axes, "n", "3", true), std::invalid_argument);
    EXPECT_THROW(shrink_axis(axes, "zz", "2", false), std::invalid_argument);
    EXPECT_THROW(shrink_axis(axes, "n", "0", false), std::invalid_argument);
}

TEST(TuneReplay, TableMisrankCountsUnmeasuredCells) {
    ReplayMeta meta;
    const auto rc = load_replay(write_raw("flip.jsonl", flip_cells()), &meta);
    const ReplayReport u = replay(rc, meta.axes, Tier::preview, sparse());  // measures n = 1, 16, 128
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
    TierParams every2 = sparse();
    every2.stride = 2, every2.refine_ratio = 1.25;
    const ReplayReport c = replay(rc, meta.axes, Tier::coarse, every2);  // n = 1,4,16,64,128, then bisects to 8
    EXPECT_EQ(c.cells_measured, 6u);
    EXPECT_EQ(c.table_misrank, 0.0);
    EXPECT_GT(c.refine_unavailable, 0u);
}

namespace {

// One cell n=1 with explicit reps: reps[attempt][cand][pass-1] = value of that (pass, rep) is `f(pass, rep)`; the
// pass record's median is median of those reps. final_attempt says which attempt the converter would use.
struct AttemptData {
    int attempt;
    std::map<std::string, std::function<double(int, int)>> value;
};

std::string write_attempts(const std::string& name, const std::vector<AttemptData>& attempts, int final_attempt) {
    const std::string path = (scratch("replay") / name).string();
    std::ofstream f(path);
    f << Json().str("kind", "meta").integer("schema", 1).str("keys", "mode:exact n:log:2")
             .str("candidates", "a|b").integer("passes", 2).integer("reps", 16).line();
    auto base = [&](const char* kind) { return Json().str("kind", kind).str("mode", "x").integer("n", 1).str("uplo", "L"); };
    for (const AttemptData& a : attempts)
        for (const auto& [cand, val] : a.value)
            for (int pass = 1; pass <= 2; ++pass) {
                std::vector<double> v;
                for (int rep = 0; rep < 16; ++rep) {
                    v.push_back(val(pass, rep));
                    f << base("rep").integer("pass", pass).integer("attempt", a.attempt).str("cand", cand).integer("rep", rep)
                             .num("ms", v.back()).line();
                }
                f << base("pass").integer("pass", pass).integer("attempt", a.attempt).str("cand", cand).str("status", "ok")
                         .str("reason", "").num("median_ms", median(v)).line();
            }
    f << base("cell").integer("round", 0).str("status", "ok").integer("final_attempt", final_attempt).line();
    return path;
}

}  // namespace

TEST(TuneReplay, LoaderUsesTheFinalAttemptAndInterleavesPassesByRep) {
    // Attempt 0 (a 1.0, b 2.0) disagrees with attempt 1, which is final. Attempt 1 values are 10 * pass + rep / 100.
    auto first = [](double base) { return [base](int, int) { return base; }; };
    const std::vector<AttemptData> att{{0, {{"a", first(1.0)}, {"b", first(2.0)}}},
                                       {1, {{"a", [](int pass, int rep) { return 10.0 * pass + rep / 100.0; }},
                                            {"b", [](int pass, int rep) { return 5.0 * pass + rep / 100.0; }}}}};
    const auto rc = load_replay(write_attempts("attempts.jsonl", att, 1));
    ASSERT_EQ(rc.size(), 1u);
    ASSERT_EQ(rc[0].cands, (std::vector<std::string>{"a", "b"}));
    // pass medians of a: 10.075 and 20.075 -> mean 15.075 (attempt 0 would give 1.0)
    EXPECT_NEAR(rc[0].exhaustive.at("a"), 15.075, 1e-9);
    EXPECT_NEAR(rc[0].exhaustive.at("b"), 7.575, 1e-9);
    ASSERT_EQ(rc[0].rounds[0].size(), 32u);
    for (int rep = 0; rep < 16; ++rep)
        for (int pass = 1; pass <= 2; ++pass)
            EXPECT_DOUBLE_EQ(rc[0].rounds[0][std::size_t(rep * 2 + pass - 1)], 10.0 * pass + rep / 100.0) << "p" << pass << "r" << rep;
    EXPECT_EQ(load_replay(write_attempts("attempts0.jsonl", att, 0))[0].exhaustive.at("a"), 1.0);
}

TEST(TuneReplay, HoldoutRacesOnPassOneAndScoresAgainstPassTwo) {
    // Pass 1 says a is faster (1.0 vs 1.5), pass 2 says b is (1.5 vs 1.0): a holdout reference exposes the winner.
    auto pass_ms = [](double p1, double p2) { return [p1, p2](int pass, int) { return pass == 1 ? p1 : p2; }; };
    const auto rc = load_replay(write_attempts("holdout.jsonl", {{0, {{"a", pass_ms(1.0, 1.5)}, {"b", pass_ms(1.5, 1.0)}}}}, 0));
    ReplayMeta meta;
    meta.axes = axis_specs({"mode:exact", "n:log:2"}, {{"mode", {"x"}}, {"n", {"1"}}});
    TierParams p = params(Tier::coarse);
    p.confidence = 1.0;
    p.min_reps = p.max_reps = 16;
    ReplayOpts hold;
    hold.holdout = true;
    const ReplayReport h = replay(rc, meta.axes, Tier::coarse, p, 0.03, hold);
    EXPECT_NEAR(h.measure_s, 16 * (1.0 + 1.5) / 1000, 1e-12) << "16 pass-1 rounds of both candidates, nothing from pass 2";
    EXPECT_EQ(h.race_misrank, 1.0);
    EXPECT_NEAR(h.max_loss, 0.5, 1e-9) << "reference a = 1.5 (pass 2), best b = 1.0";
    EXPECT_NEAR(h.est_gpu_h, (h.measure_s + 2 * (p.warm_topup_s + 0.05)) / 3600, 1e-12);
    p.min_reps = p.max_reps = 32;
    const ReplayReport all = replay(rc, meta.axes, Tier::coarse, p);
    EXPECT_NEAR(all.measure_s, 32 * (1.0 + 1.5) / 1000, 1e-12) << "without holdout every round of both passes is raced";
    EXPECT_EQ(all.race_misrank, 0.0) << "the pass medians tie, so the race and the reference agree by construction";
}

TEST(TuneReplay, NoiseFloorIsTheNoEliminationOracleAndExcessIsTheDifference) {
    const TierParams o = oracle_params();
    EXPECT_EQ(o.confidence, 1.0);
    EXPECT_EQ(o.min_reps, 16);
    EXPECT_EQ(o.max_reps, 16);
    EXPECT_EQ(o.stride, 1);
    auto pass_ms = [](double p1, double p2) { return [p1, p2](int pass, int) { return pass == 1 ? p1 : p2; }; };
    const auto rc = load_replay(write_attempts("floor.jsonl", {{0, {{"a", pass_ms(1.0, 1.5)}, {"b", pass_ms(1.5, 1.0)}}}}, 0));
    const auto axes = axis_specs({"mode:exact", "n:log:2"}, {{"mode", {"x"}}, {"n", {"1"}}});
    ReplayOpts hold;
    hold.holdout = true;
    ReplayOpts all = hold;
    all.all_cells = true;
    const ReplayReport floor = replay(rc, axes, Tier::deep, o, 0.03, all);
    EXPECT_NEAR(floor.measure_s, 16 * (1.0 + 1.5) / 1000, 1e-12) << "every pass-1 rep of every candidate, no elimination";
    EXPECT_EQ(floor.race_misrank, 1.0) << "pass 1 prefers a, the pass-2 reference prefers b: unavoidable";
    ReplayReport r = replay(rc, axes, Tier::deep, params(Tier::deep), 0.03, hold);
    apply_floor(r, floor);
    EXPECT_EQ(r.floor_table, floor.table_misrank);
    EXPECT_DOUBLE_EQ(r.excess_race, r.race_misrank - floor.race_misrank);
    EXPECT_DOUBLE_EQ(r.excess_table, r.table_misrank - floor.table_misrank);
    EXPECT_EQ(r.excess_race, 0.0) << "the race cannot lose more than the oracle here";
}

TEST(TuneReplay, NoiseFloorRacesEveryRawCellAndIgnoresGridShrinking) {
    ReplayMeta meta;
    const auto rc = load_replay(write_raw("floor_all.jsonl", flip_cells()), &meta);
    ReplayOpts all;
    all.holdout = true;
    all.all_cells = true;
    const ReplayReport full = replay(rc, meta.axes, Tier::deep, oracle_params(), 0.03, all);
    EXPECT_EQ(full.cells_measured, 8u) << "every cell, with no lattice walk and no bisection";
    auto shrunk = meta.axes;
    shrink_axis(shrunk, "n", "4", false);
    const ReplayReport cut = replay(rc, shrunk, Tier::deep, oracle_params(), 0.03, all);
    EXPECT_EQ(cut.cells_measured, 8u);
    EXPECT_EQ(cut.table_misrank, full.table_misrank);
    EXPECT_EQ(cut.race_misrank, full.race_misrank);
    ReplayOpts walk;
    walk.holdout = true;
    EXPECT_LT(replay(rc, shrunk, Tier::deep, oracle_params(), 0.03, walk).cells_measured, 8u) << "the lattice walk is what the flag removes";
}

TEST(TuneReplay, TierDefaultsDriveTheLatticeAndTheRefinementMode) {
    ReplayMeta meta;
    const auto rc = load_replay(write_raw("defaults.jsonl", flip_cells()), &meta);
    const ReplayReport p = replay(rc, meta.axes, Tier::preview);  // n = 1,4,16,64,128 + index bisection to 8
    EXPECT_EQ(p.cells_measured, 6u);
    EXPECT_EQ(p.table_misrank, 0.0);
    EXPECT_EQ(replay(rc, meta.axes, Tier::coarse).cells_measured, 8u);
}

TEST(TuneReplay, RefinementCapBoundsTheMeasuredCells) {
    // Decisive winners a,b,b,a,a,b,b,a at n = 1..128: every preview lattice bracket (1,4,16,64,128) flips.
    std::vector<SynCell> v;
    const std::map<int, bool> a_wins{{1, true}, {2, false}, {4, false}, {8, true}, {16, true}, {32, false}, {64, false}, {128, true}};
    for (const auto& [n, a] : a_wins) v.push_back({n, {{"a", a ? 1.0 : 1.5}, {"b", a ? 1.5 : 1.0}}});
    ReplayMeta meta;
    const auto rc = load_replay(write_raw("cap.jsonl", v), &meta);
    TierParams p = params(Tier::preview);
    p.refine_margin = 0;
    const ReplayReport full = replay(rc, meta.axes, Tier::preview, p);
    EXPECT_EQ(full.cells_lattice, 5u);
    EXPECT_EQ(full.cells_measured, 8u);
    EXPECT_EQ(full.refine_capped, 0u);
    p.refine_cap_factor = 0.4;  // floor(0.4 x 5) = 2 refinement cells
    const ReplayReport cut = replay(rc, meta.axes, Tier::preview, p);
    EXPECT_EQ(cut.cells_lattice, 5u);
    EXPECT_EQ(cut.cells_measured, 7u);
    EXPECT_EQ(cut.refine_capped, 1u);
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
    const ReplayReport r = replay(rc, meta.axes, Tier::preview, sparse());
    // n=2 reads n=1's row (c, a, b): c cannot run, so the selector falls back to a (2.0 against a best of 2.0)
    EXPECT_EQ(r.table_misrank, 0.0);
    EXPECT_EQ(r.unrunnable, 0u);
    EXPECT_EQ(r.max_loss, 0.0);
    rc = load_replay(write_raw("runnable.jsonl", make(true)), &meta);
    EXPECT_EQ(replay(rc, meta.axes, Tier::preview, sparse()).table_misrank, 0.0);
}

TEST(TuneReplay, FallbackCanLoseAndNoRunnableEntryIsUnrunnable) {
    ReplayMeta meta;
    // n=1 ranks c, a, b. At n=2 c cannot run and a is 3x slower than b: the fallback picks a, a misrank.
    auto rc = load_replay(write_raw("fallback.jsonl", {{1, {{"a", 2.0}, {"b", 3.0}, {"c", 1.0}}}, {2, {{"a", 3.0}, {"b", 1.0}}},
                                                       {4, {{"a", 2.0}, {"b", 3.0}, {"c", 1.0}}}, {8, {{"a", 2.0}, {"b", 3.0}, {"c", 1.0}}},
                                                       {16, {{"a", 2.0}, {"b", 3.0}, {"c", 1.0}}}}), &meta);
    const ReplayReport r = replay(rc, meta.axes, Tier::preview, sparse());  // measures n=1 and n=16; n=2 reads n=1
    EXPECT_EQ(r.unrunnable, 0u);
    EXPECT_DOUBLE_EQ(r.table_misrank, 1.0 / 5);
    EXPECT_NEAR(r.max_loss, 2.0, 0.02);
    // The measured rows list only a; the cells that run only b have no runnable entry.
    rc = load_replay(write_raw("noentry.jsonl", {{1, {{"a", 1.0}}}, {2, {{"b", 1.0}}}, {4, {{"b", 1.0}}}, {8, {{"b", 1.0}}}, {16, {{"a", 1.0}}}}), &meta);
    const ReplayReport u = replay(rc, meta.axes, Tier::preview, sparse());
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

namespace {

// Timed medians of one cell; `lattice` = a round-0 cell of this run.
RefineCell timed(std::map<std::string, double> ms, bool lattice = true, std::set<std::string> out = {}) {
    return {std::move(ms), std::move(out), lattice};
}

}  // namespace

TEST(TuneGrid, IndexModeSeesAFlipInsideAStrideFourBracket) {
    // Winners a at 1, a at 16: the geometric rule sees agreement. Only the margin trigger refines.
    const std::map<CellKey, std::vector<std::string>> ranked{{n_cell(1), {"a", "b"}}, {n_cell(16), {"a", "b"}}};
    EXPECT_TRUE(refine_all_axes(ranked, kIndexAxes, 1.1, {RefineMode::index}).next.empty());
    std::map<CellKey, RefineCell> close{{n_cell(1), timed({{"a", 1.0}, {"b", 1.02}})}, {n_cell(16), timed({{"a", 1.0}, {"b", 1.5}})}};
    std::map<CellKey, RefineCell> far{{n_cell(1), timed({{"a", 1.0}, {"b", 1.3}})}, {n_cell(16), timed({{"a", 1.0}, {"b", 1.5}})}};
    RefineOpts o{RefineMode::index, 0.05, &close};
    const auto r = refine_all_axes(ranked, kIndexAxes, 1.1, o);
    ASSERT_EQ(r.next.size(), 1u);
    EXPECT_EQ(key_arg(r.next[0]), "mode=x,n=4");
    o.cells = &far;
    EXPECT_TRUE(refine_all_axes(ranked, kIndexAxes, 1.1, o).next.empty());
    o.cells = &close;
    o.mode = RefineMode::geometric;
    EXPECT_EQ(key_arg(refine_all_axes(ranked, kIndexAxes, 1.1, o).next.at(0)), "mode=x,n=4") << "geometric midpoint of 1 and 16";
}

TEST(TuneGrid, NearTieAlternationIsNoFlip) {
    const std::map<CellKey, std::vector<std::string>> ranked{{n_cell(1), {"a", "b"}}, {n_cell(16), {"b", "a"}}};
    EXPECT_EQ(refine_all_axes(ranked, kIndexAxes, 1.1, {RefineMode::index}).next.size(), 1u) << "no times: winners that differ flip";
    std::map<CellKey, RefineCell> tie{{n_cell(1), timed({{"a", 1.0}, {"b", 1.02}})}, {n_cell(16), timed({{"b", 1.0}, {"a", 1.01}})}};
    std::map<CellKey, RefineCell> real{{n_cell(1), timed({{"a", 1.0}, {"b", 1.2}})}, {n_cell(16), timed({{"b", 1.0}, {"a", 1.2}})}};
    std::map<CellKey, RefineCell> one_end{{n_cell(1), timed({{"a", 1.0}, {"b", 1.2}})}, {n_cell(16), timed({{"b", 1.0}, {"a", 1.02}})}};
    for (RefineMode mode : {RefineMode::index, RefineMode::geometric}) {
        EXPECT_TRUE(refine_all_axes(ranked, kIndexAxes, 1.1, {mode, 0, &tie}).next.empty()) << "within the tie at both ends";
        EXPECT_EQ(refine_all_axes(ranked, kIndexAxes, 1.1, {mode, 0, &one_end}).next.size(), 1u) << "decisive at one end: a flip";
        ASSERT_EQ(refine_all_axes(ranked, kIndexAxes, 1.1, {mode, 0, &real}).next.size(), 1u);
        EXPECT_EQ(key_arg(refine_all_axes(ranked, kIndexAxes, 1.1, {mode, 0, &real}).next[0]), "mode=x,n=4");
    }
    std::map<CellKey, RefineCell> edge{{n_cell(1), timed({{"a", 1.0}, {"b", 1.0301}})}, {n_cell(16), timed({{"b", 1.0}, {"a", 1.0301}})}};
    EXPECT_EQ(refine_all_axes(ranked, kIndexAxes, 1.1, {RefineMode::index, 0, &edge}).next.size(), 1u) << "just above the tie";
    edge = {{n_cell(1), timed({{"a", 1.0}, {"b", 1.0299}})}, {n_cell(16), timed({{"b", 1.0}, {"a", 1.0299}})}};
    EXPECT_TRUE(refine_all_axes(ranked, kIndexAxes, 1.1, {RefineMode::index, 0, &edge}).next.empty()) << "just below the tie";
}

TEST(TuneGrid, UntimedOtherWinnerFlipsOnlyWhenItCannotWinThere) {
    const std::map<CellKey, std::vector<std::string>> ranked{{n_cell(1), {"a", "b"}}, {n_cell(16), {"b"}}};
    // n=1 is within the tie, so n=16 decides. There a has no time: not runnable or eliminated is decisive, unknown is not.
    std::map<CellKey, RefineCell> cells{{n_cell(1), timed({{"a", 1.0}, {"b", 1.02}})}, {n_cell(16), timed({{"b", 1.0}}, true, {"a"})}};
    EXPECT_EQ(refine_all_axes(ranked, kIndexAxes, 1.1, {RefineMode::index, 0, &cells}).next.size(), 1u);
    cells[n_cell(16)] = timed({{"b", 1.0}});
    EXPECT_TRUE(refine_all_axes(ranked, kIndexAxes, 1.1, {RefineMode::index, 0, &cells}).next.empty());
}

TEST(TuneGrid, MarginTriggersOnlyBetweenLatticeCells) {
    std::map<CellKey, std::vector<std::string>> ranked{{n_cell(1), {"a", "b"}}, {n_cell(16), {"a", "b"}}};
    std::map<CellKey, RefineCell> cells{{n_cell(1), timed({{"a", 1.0}, {"b", 1.02}})}, {n_cell(16), timed({{"a", 1.0}, {"b", 1.02}})}};
    const RefineOpts o{RefineMode::index, 0.10, &cells};
    EXPECT_EQ(refine_all_axes(ranked, kIndexAxes, 1.1, o).next.size(), 1u);
    cells[n_cell(16)].lattice = false;
    EXPECT_TRUE(refine_all_axes(ranked, kIndexAxes, 1.1, o).next.empty()) << "one end is a refinement cell";
    cells[n_cell(16)].lattice = true;
    // The margin midpoint n=4 is measured and agrees, also within the margin: it re-triggers nothing.
    ranked[n_cell(4)] = {"a", "b"};
    cells[n_cell(4)] = timed({{"a", 1.0}, {"b", 1.02}}, false);
    EXPECT_TRUE(refine_all_axes(ranked, kIndexAxes, 1.1, o).next.empty());
}

TEST(TuneGrid, BatchRefillsOnlyItsLatticeValues) {
    const std::vector<AxisSpec> axes{{"uplo", false, {"L"}}, {"n", true, {"64"}}, {"batch", true, {"128", "512", "2048", "8192"}}};
    std::map<CellKey, std::vector<std::string>> ranked{{batch_cell(128), {"a"}}, {batch_cell(8192), {"b"}}};
    for (RefineMode mode : {RefineMode::index, RefineMode::geometric}) {
        const auto next = refine_all_axes(ranked, axes, 1.1, {mode}).next;
        ASSERT_EQ(next.size(), 1u);
        EXPECT_EQ(key_arg(next[0]), "uplo=L,n=64,batch=512") << "a lattice value, not the geometric 1024";
    }
    ranked = {{batch_cell(128), {"a"}}, {batch_cell(512), {"b"}}};
    EXPECT_TRUE(refine_all_axes(ranked, axes, 1.1, {RefineMode::index}).next.empty()) << "adjacent batch values: no 256";
    EXPECT_TRUE(refine_all_axes(ranked, kBatchAxes, 1.1).next.empty()) << "no lattice values: batch is never bisected";
}

TEST(TuneGrid, FlipMidpointsComeBeforeMarginMidpoints) {
    std::map<CellKey, std::vector<std::string>> ranked{{n_cell(1), {"a", "b"}}, {n_cell(16), {"a", "b"}}, {n_cell(128), {"b", "a"}}};
    std::map<CellKey, RefineCell> cells{{n_cell(1), timed({{"a", 1.0}, {"b", 1.02}})},
                                        {n_cell(16), timed({{"a", 1.0}, {"b", 1.5}})},
                                        {n_cell(128), timed({{"b", 1.0}, {"a", 1.5}})}};
    const auto next = refine_all_axes(ranked, kIndexAxes, 1.1, {RefineMode::index, 0.10, &cells}).next;
    ASSERT_EQ(next.size(), 2u);
    EXPECT_EQ(key_arg(next[0]), "mode=x,n=32") << "the flip 16|128";
    EXPECT_EQ(key_arg(next[1]), "mode=x,n=4") << "the margin bracket 1|16";
}

TEST(TuneGrid, RefineAllowanceIsTheCapFactorTimesTheLattice) {
    EXPECT_EQ(refine_allowance(108, 0, 1.0), 108u);
    EXPECT_EQ(refine_allowance(108, 100, 1.0), 8u);
    EXPECT_EQ(refine_allowance(108, 108, 1.0), 0u);
    EXPECT_EQ(refine_allowance(108, 150, 1.0), 0u);
    EXPECT_EQ(refine_allowance(5, 0, 0.5), 2u);
    EXPECT_EQ(refine_allowance(5, 0, 2.0), 10u);
}

// ---- Ledger (tools/tune/ledger.cc) ----------------------------------------------------------

namespace {

const std::map<std::string, std::string> kHashes{{"lpanel", "h1"}, {"vendor", "v1"}};

struct TempDir {
    fs::path path;
    TempDir() {
        static int n = 0;
        path = fs::temp_directory_path() / ("batchlas_ledger_" + std::to_string(::getpid()) + "_" + std::to_string(n++));
        fs::remove_all(path);
        fs::create_directories(path);
    }
    ~TempDir() { fs::remove_all(path); }
    std::string str() const { return path.string(); }
};

CandResult cres(const std::string& cand, const std::string& hash, const std::string& status = "ok", double ms = 1.0) {
    CandResult c;
    c.cand = cand;
    c.hash = hash;
    c.status = status;
    if (status == "ok") c.median_ms = ms, c.lo = ms * 0.9, c.hi = ms * 1.1, c.reps = 8;
    return c;
}

CellRecord cell_rec(std::vector<CandResult> cands, std::vector<std::string> ranked, int n = 64) {
    CellRecord r;
    r.key = {{"n", std::to_string(n)}, {"batch", "1024"}};
    r.cands = std::move(cands);
    r.ranked = std::move(ranked);
    return r;
}

// Writes one run file holding `r` at `tier`/`date`; run ids are distinct per call.
void write_run(const TempDir& d, Tier tier, const std::string& date, const CellRecord& r, const std::string& id) {
    RunMeta m;
    m.run_id = id;
    m.tier = tier;
    m.date = date;
    LedgerWriter w(d.str(), m);
    CellRecord c = r;
    c.date = date;
    w.cell(c);
}

CellRecord two_family_cell() {
    return cell_rec({cres("lpanel:panel=8", "h1", "ok", 1.0), cres("vendor", "v1", "ok", 2.0)},
                    {"lpanel:panel=8", "vendor"});
}

std::string append_to(const fs::path& p, const std::string& text) {
    std::ofstream(p, std::ios::app) << text;
    return p.string();
}

}  // namespace

// The fixture is shared with sweep_to_table.py's self-test (ledger_all_error_record_counts_as_no_record).
TEST(TuneLedger, AllErrorRecordCountsAsNoRecord) {
    const Ledger l = read_ledger(std::string(BATCHLAS_TUNE_SOURCE_DIR) + "/tests/data/ledger_all_error/posv.float.sm_0");
    ASSERT_EQ(l.cells.size(), 2u);
    ASSERT_TRUE(all_error(l.cells[1]));
    const std::map<std::string, std::string> stored{{"tiny", "htiny"}, {"cta", "hcta"}, {"blocked", "hblocked"}};
    const auto best = best_records(l, stored);
    ASSERT_EQ(best.size(), 1u);
    EXPECT_EQ(best.begin()->second->tier, Tier::preview) << "the newer deep record is a failed child, not a result";
    EXPECT_EQ(best.begin()->second->ranked, (std::vector<std::string>{"tiny", "cta"}));
}

TEST(TuneLedger, AGitLfsPointerIsNamed) {
    TempDir d;
    std::ofstream(d.path / "20261001T000000-a-1.jsonl") << "version https://git-lfs.github.com/spec/v1\noid sha256:00\nsize 9\n";
    try {
        (void)read_ledger(d.str());
        ADD_FAILURE() << "read an LFS pointer as a ledger";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("Git LFS pointer: run git lfs pull"), std::string::npos) << e.what();
    }
}

TEST(TuneLedger, CoarseNeverDisplacesCurrentDeep) {
    TempDir d;
    write_run(d, Tier::deep, "2026-10-01", two_family_cell(), "20261001T000000-a-1");
    write_run(d, Tier::coarse, "2026-10-05", two_family_cell(), "20261005T000000-b-2");
    const Ledger l = read_ledger(d.str());
    ASSERT_EQ(l.cells.size(), 2u);
    const auto best = best_records(l, kHashes);
    ASSERT_EQ(best.size(), 1u);
    EXPECT_EQ(best.begin()->second->tier, Tier::deep);
    EXPECT_EQ(best.begin()->second->run_id, "20261001T000000-a-1");
}

TEST(TuneLedger, CoarseDisplacesStaleDeep) {
    TempDir d;
    write_run(d, Tier::deep, "2026-10-01", two_family_cell(), "20261001T000000-a-1");
    write_run(d, Tier::coarse, "2026-10-05", two_family_cell(), "20261005T000000-b-2");
    auto changed = kHashes;
    changed["lpanel"] = "h2";  // the deep winner's family changed
    const Ledger l = read_ledger(d.str());
    EXPECT_EQ(freshness(l.cells[0], changed), Freshness::stale);
    // The coarse record shares the stale hash for lpanel, so rewrite it against the new hash.
    TempDir d2;
    write_run(d2, Tier::deep, "2026-10-01", two_family_cell(), "20261001T000000-a-1");
    write_run(d2, Tier::coarse, "2026-10-05",
              cell_rec({cres("lpanel:panel=8", "h2", "ok", 1.0), cres("vendor", "v1", "ok", 2.0)}, {"lpanel:panel=8", "vendor"}),
              "20261005T000000-b-2");
    const Ledger l2 = read_ledger(d2.str());
    const auto best = best_records(l2, changed);
    ASSERT_EQ(best.size(), 1u);
    EXPECT_EQ(best.begin()->second->tier, Tier::coarse);
}

TEST(TuneLedger, PartlyStaleKeepsItsTier) {
    TempDir d;
    write_run(d, Tier::deep, "2026-10-01", two_family_cell(), "20261001T000000-a-1");
    write_run(d, Tier::coarse, "2026-10-05",
              cell_rec({cres("lpanel:panel=8", "h1", "ok", 1.0), cres("vendor", "v2", "ok", 2.0)}, {"lpanel:panel=8", "vendor"}),
              "20261005T000000-b-2");
    auto now = kHashes;
    now["vendor"] = "v2";  // the deep record's runner-up changed
    const Ledger l = read_ledger(d.str());
    const CellRecord& deep = l.cells[0].tier == Tier::deep ? l.cells[0] : l.cells[1];
    EXPECT_EQ(freshness(deep, now), Freshness::partly_stale);
    EXPECT_EQ(stale_candidates(deep, now), std::vector<std::string>{"vendor"});
    const auto best = best_records(l, now);
    ASSERT_EQ(best.size(), 1u);
    EXPECT_EQ(best.begin()->second->tier, Tier::deep) << "partly stale still counts at its tier";
}

TEST(TuneLedger, SkippedCandidateHashChangeKeepsTheCellCurrent) {
    // vendor could not run at this cell: an edit to its family cannot change the ranking.
    const CellRecord r = cell_rec({cres("lpanel:panel=8", "h1", "ok", 1.0), cres("vendor", "v1", "skipped")}, {"lpanel:panel=8"});
    auto now = kHashes;
    now["vendor"] = "v2";
    EXPECT_EQ(freshness(r, now), Freshness::current);
    EXPECT_TRUE(stale_candidates(r, now).empty());
    const CellRecord bad = cell_rec({cres("lpanel:panel=8", "h1", "ok", 1.0), cres("vendor", "v1", "bad")}, {"lpanel:panel=8"});
    EXPECT_EQ(freshness(bad, now), Freshness::partly_stale) << "a refused-by-verification candidate did run";
}

TEST(TuneLedger, AddedFamilyIsPartlyStale) {
    const CellRecord r = two_family_cell();
    auto now = kHashes;
    now["wide"] = "w1";
    EXPECT_EQ(freshness(r, kHashes), Freshness::current);
    EXPECT_EQ(freshness(r, now), Freshness::partly_stale);
    EXPECT_EQ(stale_candidates(r, now), std::vector<std::string>{"wide"});
    CellRecord empty = cell_rec({cres("lpanel:panel=8", "h1", "bad"), cres("vendor", "v1", "bad")}, {});
    EXPECT_EQ(freshness(empty, kHashes), Freshness::current);
    EXPECT_EQ(freshness(empty, now), Freshness::partly_stale);
}

TEST(TuneLedger, RemovedWinnerIsStale) {
    const CellRecord r = two_family_cell();
    std::map<std::string, std::string> now{{"vendor", "v1"}};  // lpanel is gone
    EXPECT_EQ(freshness(r, now), Freshness::stale);
    std::map<std::string, std::string> no_runner{{"lpanel", "h1"}};  // only the runner-up is gone
    EXPECT_EQ(freshness(r, no_runner), Freshness::partly_stale);
}

TEST(TuneLedger, TruncatedLastLineIsSkippedWithAWarning) {
    TempDir d;
    write_run(d, Tier::deep, "2026-10-01", two_family_cell(), "20261001T000000-a-1");
    const fs::path f = d.path / "20261001T000000-a-1.jsonl";
    CellRecord second = two_family_cell();
    second.key = {{"n", "128"}, {"batch", "1024"}};
    RunMeta m;
    m.run_id = "20261001T000000-a-1";
    {
        LedgerWriter w(d.str(), m);  // reopen in append mode: adds a second run line, then a cell
        w.cell(second);
    }
    std::string text = [&] { std::ifstream in(f); std::stringstream s; s << in.rdbuf(); return s.str(); }();
    ASSERT_EQ(text.back(), '\n');
    text.resize(text.size() - 25);  // kill mid-write: the last cell line loses its tail
    { std::ofstream out(f, std::ios::trunc); out << text; }
    const Ledger l = read_ledger(d.str());
    ASSERT_EQ(l.cells.size(), 1u) << "only the intact cell survives";
    EXPECT_EQ(key_arg(l.cells[0].key), "n=64,batch=1024");
    ASSERT_EQ(l.warnings.size(), 1u);
    EXPECT_NE(l.warnings[0].find("truncated"), std::string::npos);
}

TEST(TuneLedger, MalformedMiddleLineThrows) {
    TempDir d;
    write_run(d, Tier::deep, "2026-10-01", two_family_cell(), "20261001T000000-a-1");
    append_to(d.path / "20261001T000000-a-1.jsonl", "{\"kind\": \"cell\", \"oops\n");
    append_to(d.path / "20261001T000000-a-1.jsonl", two_family_cell().ranked.empty() ? "" : "{\"kind\": \"audit\"}\n");
    EXPECT_THROW(read_ledger(d.str()), std::runtime_error);
}

TEST(TuneLedger, EmptyRankingIsARecordNotARerun) {
    TempDir d;
    write_run(d, Tier::coarse, "2026-10-01", cell_rec({cres("lpanel:panel=8", "h1", "bad"), cres("vendor", "v1", "skipped")}, {}),
              "20261001T000000-a-1");
    const Ledger l = read_ledger(d.str());
    const auto best = best_records(l, kHashes);
    ASSERT_EQ(best.size(), 1u);
    EXPECT_TRUE(best.begin()->second->ranked.empty());
    EXPECT_EQ(best.begin()->second->cands.size(), 2u);
}

TEST(TuneLedger, TwoRunFilesUnion) {
    TempDir d;
    // Box A: deep on n=64 (older), coarse on n=128. Box B: coarse on n=64 (newer), coarse on n=128 (newer).
    write_run(d, Tier::deep, "2026-10-01", two_family_cell(), "20261001T000000-boxa-1");
    write_run(d, Tier::coarse, "2026-10-01", cell_rec({cres("lpanel:panel=8", "h1"), cres("vendor", "v1", "ok", 2)}, {"lpanel:panel=8", "vendor"}, 128),
              "20261001T000001-boxa-1");
    write_run(d, Tier::coarse, "2026-10-06", two_family_cell(), "20261006T000000-boxb-9");
    write_run(d, Tier::coarse, "2026-10-06", cell_rec({cres("lpanel:panel=8", "h1"), cres("vendor", "v1", "ok", 0.5)}, {"vendor", "lpanel:panel=8"}, 128),
              "20261006T000001-boxb-9");
    const Ledger l = read_ledger(d.str());
    EXPECT_EQ(l.runs.size(), 4u);
    const auto best = best_records(l, kHashes);
    ASSERT_EQ(best.size(), 2u);
    for (const auto& [k, r] : best) {
        if (key_int(k, "n") == 64) EXPECT_EQ(r->tier, Tier::deep);
        else {
            EXPECT_EQ(r->run_id, "20261006T000001-boxb-9") << "same tier: the newer date wins";
            EXPECT_EQ(r->ranked.front(), "vendor");
        }
    }
}

TEST(TuneLedger, EqualTierAndDateTakesTheLargerRunId) {
    TempDir d;
    write_run(d, Tier::coarse, "2026-10-01", cell_rec({cres("vendor", "v1")}, {"vendor"}), "20261001T000000-a-1");
    write_run(d, Tier::coarse, "2026-10-01", cell_rec({cres("vendor", "v1")}, {"vendor"}), "20261001T000000-a-2");
    const Ledger l = read_ledger(d.str());
    const auto best = best_records(l, {{"vendor", "v1"}});
    EXPECT_EQ(best.begin()->second->run_id, "20261001T000000-a-2");
}

TEST(TuneLedger, WriterRoundTripsEveryField) {
    TempDir d;
    RunMeta m;
    m.run_id = make_run_id();
    m.host = "h \"q\"";
    m.device = "sm_89";
    m.device_name = "RTX 4090";
    m.batchlas = "abc1234-dirty";
    m.argv = "--tier coarse --ops trsm";
    m.date = "2026-10-07";
    m.tier = Tier::preview;
    m.keys = "uplo:exact n:log:3 batch:log";
    m.candidates = "tiny|lpanel:panel=8|vendor";
    m.worker_mode = {{"mode", "jit"}, {"warm", "0.2"}};
    CellRecord c = cell_rec({cres("lpanel:panel=8", "h1", "ok", 1.25), cres("vendor", "v1", "eliminated"),
                             cres("tiny", "t1", "error")},
                            {"lpanel:panel=8"});
    c.key = {{"uplo", "L"}, {"n", "64"}};
    c.round = 3;
    c.cands[2].reason = "cuda error, \"bad\"\nline";
    c.cands[0].reps = 12;
    {
        LedgerWriter w(ledger_dir(d.str(), "trsm", "float", "sm_89"), m);
        w.cell(c);
        w.audit(c.key, "pass", 1.5, 1.4);
    }
    const Ledger l = read_ledger(ledger_dir(d.str(), "trsm", "float", "sm_89"));
    ASSERT_EQ(l.runs.size(), 1u);
    const RunMeta& r = l.runs[0];
    EXPECT_EQ(r.run_id, m.run_id);
    EXPECT_EQ(r.host, m.host);
    EXPECT_EQ(r.device, m.device);
    EXPECT_EQ(r.device_name, m.device_name);
    EXPECT_EQ(r.batchlas, m.batchlas);
    EXPECT_EQ(r.argv, m.argv);
    EXPECT_EQ(r.date, m.date);
    EXPECT_EQ(r.keys, m.keys);
    EXPECT_EQ(r.candidates, m.candidates);
    EXPECT_EQ(r.tier, Tier::preview);
    EXPECT_EQ(r.worker_mode, m.worker_mode);
    ASSERT_EQ(l.cells.size(), 1u);
    const CellRecord& g = l.cells[0];
    EXPECT_EQ(g.run_id, m.run_id);
    EXPECT_EQ(g.tier, Tier::preview) << "the writer stamps its own tier";
    EXPECT_EQ(g.key, c.key);
    EXPECT_EQ(g.round, 3);
    EXPECT_EQ(g.date, c.date);
    EXPECT_EQ(g.ranked, c.ranked);
    ASSERT_EQ(g.cands.size(), 3u);
    for (std::size_t i = 0; i < 3; ++i) {
        EXPECT_EQ(g.cands[i].cand, c.cands[i].cand);
        EXPECT_EQ(g.cands[i].hash, c.cands[i].hash);
        EXPECT_EQ(g.cands[i].status, c.cands[i].status);
        EXPECT_EQ(g.cands[i].reason, c.cands[i].reason);
        EXPECT_EQ(g.cands[i].reps, c.cands[i].reps);
        for (auto [a, b] : {std::pair(g.cands[i].median_ms, c.cands[i].median_ms), std::pair(g.cands[i].lo, c.cands[i].lo),
                            std::pair(g.cands[i].hi, c.cands[i].hi)})
            EXPECT_TRUE((std::isnan(a) && std::isnan(b)) || a == b);
    }
    EXPECT_EQ(make_run_id().find('-'), 15u) << "yyyymmddThhmmss";
}

TEST(TuneLedger, ImportSchema1BuildsADeepRunFromPassRecords) {
    TempDir d;
    const std::string raw = (d.path / "raw.jsonl").string();
    {
        std::ofstream o(raw);
        o << Json().str("kind", "meta").integer("schema", 1).str("op", "trsm").str("dtype", "float").str("device", "sm_89")
                 .str("device_name", "RTX").str("batchlas", "abc").str("kernels", "k1").str("date", "2026-09-01")
                 .str("keys", "n:log:3 batch:log").str("candidates", "cta|vendor|blocked:nb=8").integer("passes", 2).line();
        const CellKey k{{"n", "16"}, {"batch", "256"}};
        auto pass = [&](int attempt, int p, const char* cand, const char* st, double ms, int reps) {
            o << Json().str("kind", "pass").key(k).integer("pass", p).integer("attempt", attempt).str("cand", cand)
                     .str("status", st).str("reason", "").num("median_ms", ms).integer("reps", reps).line();
        };
        o << Json().str("kind", "rep").key(k).integer("pass", 1).integer("attempt", 0).str("cand", "cta").num("ms", 9).line();
        pass(0, 1, "cta", "ok", 9.0, 5); pass(0, 2, "cta", "ok", 9.0, 5);
        pass(1, 1, "cta", "ok", 1.0, 5); pass(1, 2, "cta", "ok", 1.2, 6);
        pass(1, 1, "vendor", "ok", 2.0, 5); pass(1, 2, "vendor", "error", 0, 0);
        o << Json().str("kind", "cell").key(k).integer("round", 2).str("status", "ok").str("reason", "").integer("final_attempt", 1)
                 .str("ranked", "cta").line();
        const CellKey k2{{"n", "32"}, {"batch", "256"}};
        o << Json().str("kind", "cell").key(k2).integer("round", 0).str("status", "skipped").str("reason", "over cap")
                 .integer("final_attempt", -1).str("ranked", "").line();
    }
    const std::map<std::string, std::string> now{{"cta", "c9"}, {"vendor", "v9"}, {"blocked", "b9"}};
    import_schema1(raw, d.str(), now, "k1");
    const Ledger l = read_ledger(ledger_dir(d.str(), "trsm", "float", "sm_89"));
    ASSERT_EQ(l.runs.size(), 1u);
    EXPECT_EQ(l.runs[0].tier, Tier::deep);
    ASSERT_EQ(l.cells.size(), 2u);
    const CellRecord* a = l.cells[0].round == 2 ? &l.cells[0] : &l.cells[1];
    const CellRecord* b = a == &l.cells[0] ? &l.cells[1] : &l.cells[0];
    EXPECT_EQ(key_arg(a->key), "n=16,batch=256") << "table keys only, in meta order";
    EXPECT_EQ(a->tier, Tier::deep);
    EXPECT_EQ(a->date, "2026-09-01");
    EXPECT_EQ(a->ranked, std::vector<std::string>{"cta"});
    ASSERT_EQ(a->cands.size(), 3u);
    EXPECT_EQ(a->cands[0].status, "ok");
    EXPECT_DOUBLE_EQ(a->cands[0].median_ms, 1.1);
    EXPECT_DOUBLE_EQ(a->cands[0].lo, 1.0);
    EXPECT_DOUBLE_EQ(a->cands[0].hi, 1.2);
    EXPECT_EQ(a->cands[0].reps, 11);
    EXPECT_EQ(a->cands[0].hash, "c9") << "raw kernels equals the current op hash";
    EXPECT_EQ(a->cands[1].status, "error") << "one pass failed";
    EXPECT_EQ(a->cands[2].status, "skipped") << "never timed";
    EXPECT_EQ(a->cands[2].hash, "b9");
    EXPECT_TRUE(b->ranked.empty());
    EXPECT_EQ(b->cands[0].status, "skipped");
    EXPECT_EQ(b->cands[0].reason, "over cap");
    EXPECT_EQ(freshness(*a, now), Freshness::current);
    const std::string other = d.str() + "/other";
    import_schema1(raw, other, now, "k2");
    const Ledger legacy = read_ledger(ledger_dir(other, "trsm", "float", "sm_89"));
    EXPECT_EQ(legacy.cells[0].cands[0].hash, "legacy:k1");
    EXPECT_EQ(freshness(legacy.cells[0], now), Freshness::stale);
    const std::string custom = d.str() + "/custom";  // a protocol-flag run records itself this way
    import_schema1(raw, custom, now, "k1", Tier::custom);
    const Ledger c = read_ledger(ledger_dir(custom, "trsm", "float", "sm_89"));
    ASSERT_EQ(c.runs.size(), 1u);
    EXPECT_EQ(c.runs[0].tier, Tier::custom);
    EXPECT_EQ(c.runs[0].keys, "n:log:3 batch:log");
    EXPECT_EQ(c.runs[0].candidates, "cta|vendor|blocked:nb=8");
    EXPECT_EQ(c.cells[0].tier, Tier::custom);
}

TEST(TuneLedger, ImportSchema1KeysCellsByTheGridAxesLikeTheDriver) {
    // trsm's uplo/diag are grid axes but not table keys; the driver's cell keys carry them, so an
    // import keyed by the table keys alone would never match a tiered cell.
    TempDir d;
    const std::string raw = (d.path / "raw.jsonl").string();
    {
        std::ofstream o(raw);
        o << Json().str("kind", "meta").integer("schema", 1).str("op", "trsm").str("dtype", "float").str("device", "sm_89")
                 .str("device_name", "RTX").str("batchlas", "abc").str("kernels", "k1").str("date", "2026-09-01")
                 .str("keys", "n:log:3 batch:log").str("candidates", "cta").integer("passes", 1).line();
        const CellKey with_u{{"n", "16"}, {"batch", "256"}, {"uplo", "U"}}, without{{"n", "32"}, {"batch", "256"}};
        for (const CellKey& k : {with_u, without}) {
            o << Json().str("kind", "pass").key(k).integer("pass", 1).integer("attempt", 0).str("cand", "cta")
                     .str("status", "ok").str("reason", "").num("median_ms", 1.0).integer("reps", 4).line();
            o << Json().str("kind", "cell").key(k).integer("round", 0).str("status", "ok").str("reason", "")
                     .integer("final_attempt", 0).str("ranked", "cta").line();
        }
    }
    const std::map<std::string, std::string> now{{"cta", "c9"}};
    const std::vector<std::pair<std::string, std::vector<std::string>>> axes{
        {"uplo", {"L"}}, {"n", {"16", "32"}}, {"batch", {"256"}}};
    import_schema1(raw, d.str(), now, "k1", Tier::deep, axes);
    const Ledger l = read_ledger(ledger_dir(d.str(), "trsm", "float", "sm_89"));
    ASSERT_EQ(l.cells.size(), 2u);
    std::set<std::string> keys;
    for (const CellRecord& c : l.cells) keys.insert(key_arg(c.key));
    EXPECT_EQ(keys, (std::set<std::string>{"uplo=U,n=16,batch=256", "uplo=L,n=32,batch=256"}))
        << "every axis in axis order; a missing fixed axis takes its only value";
    auto with_side = axes;
    with_side.push_back({"side", {"L", "R"}});  // not fixed: a record without it has no key
    EXPECT_THROW(import_schema1(raw, d.str() + "/side", now, "k1", Tier::deep, with_side), std::runtime_error);
}

TEST(TuneLedger, ReopenAfterTornTailStaysReadable) {
    TempDir d;
    write_run(d, Tier::coarse, "2026-10-01", two_family_cell(), "20261001T000000-a-1");
    const fs::path f = d.path / "20261001T000000-a-1.jsonl";
    append_to(f, "{\"kind\": \"cell\", \"run_id\": \"20261001T00");  // killed mid-write
    CellRecord second = two_family_cell();
    second.key = {{"n", "128"}, {"batch", "1024"}};
    RunMeta m;
    m.run_id = "20261001T000000-a-1";
    m.tier = Tier::coarse;
    {
        LedgerWriter w(d.str(), m);
        w.cell(second);
    }
    Ledger l;
    ASSERT_NO_THROW(l = read_ledger(d.str()));
    EXPECT_EQ(l.cells.size(), 2u);
    EXPECT_TRUE(l.warnings.empty());
}

TEST(TuneLedger, ValidLastLineWithoutNewlineParses) {
    TempDir d;
    write_run(d, Tier::coarse, "2026-10-01", two_family_cell(), "20261001T000000-a-1");
    const fs::path f = d.path / "20261001T000000-a-1.jsonl";
    std::string text = [&] { std::ifstream in(f); std::stringstream s; s << in.rdbuf(); return s.str(); }();
    text.pop_back();
    { std::ofstream out(f, std::ios::trunc); out << text; }
    EXPECT_EQ(read_ledger(d.str()).cells.size(), 1u);
    RunMeta m;
    m.run_id = "20261001T000000-a-1";
    m.tier = Tier::coarse;
    CellRecord second = two_family_cell();
    second.key = {{"n", "128"}, {"batch", "1024"}};
    { LedgerWriter w(d.str(), m); w.cell(second); }
    EXPECT_EQ(read_ledger(d.str()).cells.size(), 2u) << "the complete last record was kept";
}

TEST(TuneLedger, LaterRecordInSameRunWins) {
    TempDir d;
    RunMeta m;
    m.run_id = "20261001T000000-a-1";
    m.tier = Tier::coarse;
    {
        LedgerWriter w(d.str(), m);
        CellRecord a = cell_rec({cres("vendor", "v1")}, {"vendor"});
        a.date = "2026-10-01";
        w.cell(a);
        CellRecord b = cell_rec({cres("vendor", "v1")}, {"vendor"});
        b.date = "2026-10-01";
        b.round = 7;
        w.cell(b);
    }
    const Ledger l = read_ledger(d.str());
    const auto best = best_records(l, {{"vendor", "v1"}});
    ASSERT_EQ(best.size(), 1u);
    EXPECT_EQ(best.begin()->second->round, 7);
}

namespace {

CellKey nkey(int n, int batch = 1024) { return {{"n", std::to_string(n)}, {"batch", std::to_string(batch)}}; }

PlanSpec plan_spec(std::vector<std::string> cands) {
    return {std::move(cands), [](const CellKey& k) { return double(key_int(k, "n")) * 1e6; }};
}

const std::vector<std::string> kTwo{"lpanel:panel=8", "vendor"};

ArmOutcome arm(const std::string& name, const std::string& status, std::vector<double> ms = {}) {
    ArmOutcome a;
    a.arm = name;
    a.status = status;
    a.ms = std::move(ms);
    return a;
}

}  // namespace

TEST(TuneSchedule, CurrentDeepCellIsSkippedByCoarse) {
    Ledger l;
    CellRecord r = two_family_cell();
    r.tier = Tier::deep;
    l.cells.push_back(r);
    auto plan = plan_round(plan_spec(kTwo), Tier::coarse, {nkey(64), nkey(128)}, l, kHashes, 4, kChildOverheadS);
    ASSERT_EQ(plan.size(), 2u);
    EXPECT_EQ(plan[0].key, nkey(64));
    EXPECT_EQ(plan[0].reason, "skip:current");
    EXPECT_TRUE(plan[0].arms.empty());
    EXPECT_EQ(plan[0].est_s, 0);
    EXPECT_EQ(plan[1].reason, "");
    EXPECT_EQ(plan[1].arms, kTwo);
    EXPECT_EQ(plan[1].tier, Tier::coarse);
    EXPECT_GT(plan[1].est_s, kChildOverheadS);
    EXPECT_EQ(plan_round(plan_spec(kTwo), Tier::deep, {nkey(64)}, l, kHashes, 4, 0)[0].reason, "skip:current");

    l.cells[0].tier = Tier::coarse;
    plan = plan_round(plan_spec(kTwo), Tier::deep, {nkey(64)}, l, kHashes, 4, 0);
    EXPECT_EQ(plan[0].reason, "") << "a coarse record does not satisfy deep";
    EXPECT_EQ(plan[0].tier, Tier::deep);
    EXPECT_EQ(plan_round(plan_spec(kTwo), Tier::preview, {nkey(64)}, l, kHashes, 4, 0)[0].reason, "skip:current");

    l.cells[0].tier = Tier::deep;
    auto changed = kHashes;
    changed["lpanel"] = "h2";  // the winner's family changed: stale, re-measured in full at the running tier like a missing cell
    plan = plan_round(plan_spec(kTwo), Tier::coarse, {nkey(64)}, l, changed, 4, 0);
    EXPECT_EQ(plan[0].reason, "");
    EXPECT_EQ(plan[0].arms, kTwo);
    EXPECT_EQ(plan[0].tier, Tier::coarse);
}

TEST(TuneSchedule, PartlyStaleRacesOnlyChangedPlusTopTwo) {
    const std::vector<std::string> cands{"lpanel:panel=8", "lpanel:panel=16", "vendor", "wide:m=1", "cta"};
    Ledger l;
    CellRecord r = cell_rec({cres("lpanel:panel=8", "h1", "ok", 1.0), cres("lpanel:panel=16", "h1", "ok", 1.5),
                             cres("vendor", "v1", "ok", 2.0), cres("cta", "c1", "ok", 3.0), cres("old:x", "o1", "ok", 5.0)},
                            {"lpanel:panel=8", "lpanel:panel=16", "vendor", "cta", "old:x"});
    r.tier = Tier::deep;
    l.cells.push_back(r);
    // cta changed, wide was added, old was removed; lpanel and vendor are unchanged.
    const std::map<std::string, std::string> now{{"lpanel", "h1"}, {"vendor", "v1"}, {"cta", "c2"}, {"wide", "w1"}};
    const auto plan = plan_round(plan_spec(cands), Tier::preview, {nkey(64)}, l, now, 4, 0);
    ASSERT_EQ(plan.size(), 1u);
    EXPECT_EQ(plan[0].reason, "partial:cta,wide");
    EXPECT_EQ(plan[0].arms, (std::vector<std::string>{"lpanel:panel=8", "lpanel:panel=16", "wide:m=1", "cta"}))
        << "the stale families plus the stored winner and runner-up; vendor is not re-raced";
    EXPECT_EQ(plan[0].tier, Tier::deep) << "measured at the stored record's tier";
    ASSERT_EQ(plan[0].stored, &l.cells[0]);

    const CellRecord merged = record_from_arms(nkey(64), 3,
                                               {arm("lpanel:panel=8", "ok", {1.1, 1.0, 1.2}), arm("lpanel:panel=16", "ok", {2.5}),
                                                arm("wide:m=1", "ok", {0.5, 0.6}), arm("cta", "bad")},
                                               cands, now, plan[0].stored);
    EXPECT_EQ(merged.round, 3);
    std::vector<std::string> names;
    for (const CandResult& c : merged.cands) names.push_back(c.cand);
    EXPECT_EQ(names, cands) << "unchanged vendor kept, removed old dropped, candidate order";
    ASSERT_EQ(merged.cands.size(), cands.size());
    EXPECT_DOUBLE_EQ(merged.cands[0].median_ms, 1.1);
    EXPECT_DOUBLE_EQ(merged.cands[0].lo, 1.0);
    EXPECT_DOUBLE_EQ(merged.cands[0].hi, 1.2);
    EXPECT_EQ(merged.cands[0].reps, 3);
    EXPECT_DOUBLE_EQ(merged.cands[2].median_ms, 2.0) << "the stored vendor time";
    EXPECT_EQ(merged.cands[3].hash, "w1");
    EXPECT_EQ(merged.cands[4].status, "bad");
    EXPECT_EQ(merged.cands[4].hash, "c2");
    EXPECT_TRUE(std::isnan(merged.cands[4].median_ms));
    EXPECT_EQ(merged.ranked, (std::vector<std::string>{"wide:m=1", "lpanel:panel=8", "lpanel:panel=16", "vendor"}))
        << "the stored vendor (2.0) stays below the re-raced runner-up (2.5)";
    EXPECT_EQ(freshness(merged, now), Freshness::current);
}

// The fixture is shared with sweep_to_table.py's self-test (ledger_dominated_skip_is_partly_stale).
// evidence: docs/design/tiered-tuning.md#engine-dominance-carry-forward-removed
TEST(TuneSchedule, ADominatedSkipIsPartlyStaleAndReRacedAgainstTheTopTwo) {
    const Ledger l = read_ledger(std::string(BATCHLAS_TUNE_SOURCE_DIR) + "/tests/data/ledger_dominated/posv.float.sm_0");
    ASSERT_EQ(l.cells.size(), 2u);
    const std::vector<std::string> cands{"tiny", "cta", "blocked"};
    const std::map<std::string, std::string> fh{{"tiny", "htiny"}, {"cta", "hcta"}, {"blocked", "hblocked"}};
    EXPECT_EQ(stale_candidates(l.cells[0], fh), (std::vector<std::string>{"blocked"}));
    EXPECT_EQ(freshness(l.cells[0], fh), Freshness::partly_stale) << "dominated: never tried there";
    EXPECT_EQ(freshness(l.cells[1], fh), Freshness::current) << "a refused pin could not run there";
    const auto plan = plan_round(plan_spec(cands), Tier::deep, {l.cells[1].key, l.cells[0].key}, l, fh, 4, 0);
    ASSERT_EQ(plan.size(), 2u);
    EXPECT_EQ(plan[0].key, l.cells[0].key);
    EXPECT_EQ(plan[0].reason, "partial:blocked");
    EXPECT_EQ(plan[0].arms, cands) << "the dominated arm plus the stored winner and runner-up";
    EXPECT_EQ(plan[0].tier, Tier::deep);
    EXPECT_EQ(plan[1].reason, "skip:current");
    const CellRecord merged = record_from_arms(l.cells[0].key, 0,
                                               {arm("tiny", "ok", {1.0}), arm("cta", "ok", {2.0}), arm("blocked", "ok", {0.5})},
                                               cands, fh, plan[0].stored);
    EXPECT_EQ(merged.ranked, (std::vector<std::string>{"blocked", "tiny", "cta"}));
    EXPECT_EQ(freshness(merged, fh), Freshness::current);
}

TEST(TuneSchedule, SingleRunnableCandidateIsNotTimed) {
    std::map<std::string, std::string> h;
    for (const char* f : {"tiny", "cta", "lpanel", "blocked", "vendor"}) h[f] = "x";
    const std::map<CellKey, std::vector<std::string>> runnable{
        {nkey(16), {"tiny"}}, {nkey(24), {}}, {nkey(32), {"vendor", "cta"}}};
    const auto plan = plan_round(plan_spec(kPotrfOrder), Tier::preview, {nkey(16), nkey(24), nkey(32), nkey(64)},
                                 Ledger{}, h, 4, kChildOverheadS, &runnable);
    ASSERT_EQ(plan.size(), 4u);
    EXPECT_EQ(plan[0].reason, "skip:single");
    EXPECT_EQ(plan[0].arms, std::vector<std::string>{"tiny"});
    EXPECT_DOUBLE_EQ(plan[0].est_s, kChildOverheadS) << "a probe, nothing timed";
    EXPECT_EQ(plan[1].reason, "skip:single");
    EXPECT_TRUE(plan[1].arms.empty());
    EXPECT_EQ(plan[2].reason, "");
    EXPECT_EQ(plan[2].arms, (std::vector<std::string>{"cta", "vendor"})) << "the runnable arms, candidate order";
    EXPECT_EQ(plan[3].arms, kPotrfOrder) << "no probe result: every candidate";
    EXPECT_EQ(plan_round(plan_spec(kPotrfOrder), Tier::preview, {nkey(16)}, Ledger{}, h, 4, 0)[0].reason, "")
        << "unknown runnable set: measured";

    const CellRecord one = single_record(nkey(16), 0, "tiny", kPotrfOrder, h);
    EXPECT_EQ(one.ranked, std::vector<std::string>{"tiny"});
    ASSERT_EQ(one.cands.size(), kPotrfOrder.size());
    EXPECT_EQ(one.cands[0].status, "ok");
    EXPECT_EQ(one.cands[0].reps, 0);
    EXPECT_TRUE(std::isnan(one.cands[0].median_ms));
    for (std::size_t i = 1; i < one.cands.size(); ++i) EXPECT_EQ(one.cands[i].status, "skipped") << one.cands[i].cand;
    EXPECT_TRUE(single_record(nkey(24), 0, "", kPotrfOrder, h).ranked.empty());
}

TEST(TuneSchedule, AscendingBytesOrder) {
    const double cap_gib = 200e6 / (1024.0 * 1024.0 * 1024.0);
    const auto plan = plan_round(plan_spec(kTwo), Tier::coarse, {nkey(256), nkey(64, 2048), nkey(64), nkey(128)},
                                 Ledger{}, kHashes, cap_gib, 0);
    std::vector<std::string> order;
    for (const PlannedCell& c : plan) order.push_back(key_arg(c.key));
    EXPECT_EQ(order, (std::vector<std::string>{"n=64,batch=2048", "n=64,batch=1024", "n=128,batch=1024",
                                               "n=256,batch=1024"}))
        << "ascending bytes, equal bytes in input order";
    ASSERT_EQ(plan.size(), 4u);
    EXPECT_EQ(plan[3].reason, "skip:cap");
    EXPECT_TRUE(plan[3].arms.empty());
}

TEST(TuneSchedule, PosvAfterPotrfAndTrsm) {
    using V = std::vector<std::string>;
    EXPECT_EQ(op_order({"posv", "gemm", "potrf", "trsm"}), (V{"gemm", "potrf", "trsm", "posv"}));
    EXPECT_EQ(op_order({"potrf", "posv", "trsm"}), (V{"potrf", "trsm", "posv"}));
    EXPECT_EQ(op_order({"trsm", "potrf", "posv", "gemm"}), (V{"trsm", "potrf", "posv", "gemm"}));
    EXPECT_EQ(op_order({"gemm", "posv"}), (V{"gemm", "posv"}));
}

TEST(TuneSchedule, AllNineteenOpsFollowTheirDependencies) {
    using V = std::vector<std::string>;
    const V want{"gemm", "trsm", "syr2k", "gemv", "spmm", "trmm", "symm", "syrk", "potrf", "getrf",
                 "getrs", "geqrf", "ormqr", "getri", "posv", "gesv", "orgqr", "syev", "gesvd"};
    V shuffled = want;
    std::reverse(shuffled.begin(), shuffled.end());
    std::rotate(shuffled.begin(), shuffled.begin() + 7, shuffled.end());
    EXPECT_EQ(op_order(all_ops(shuffled)), want) << "`all` in any registration order";
    EXPECT_EQ(canonical_op_order(), want);
    for (const V& in : {shuffled, want, V(want.rbegin(), want.rend())}) {
        const V out = op_order(in);
        ASSERT_TRUE(out.size() == in.size() && std::is_permutation(out.begin(), out.end(), in.begin()));
        for (const auto& [op, deps] : op_dependencies())
            for (const std::string& d : deps)
                EXPECT_LT(std::find(out.begin(), out.end(), d), std::find(out.begin(), out.end(), op)) << d << " before " << op;
    }
    EXPECT_EQ(op_order({"syev", "orgqr", "ormqr", "trmm", "gemm"}), (V{"gemm", "trmm", "ormqr", "syev", "orgqr"}));
    EXPECT_EQ(op_order({"gesv", "spmm", "getrs", "getrf"}), (V{"spmm", "getrs", "getrf", "gesv"}))
        << "input order breaks ties; a missing dependency (trsm, gemm) does not hold anything back";
    EXPECT_EQ(all_ops({"zeta", "posv", "gemm", "alpha"}), (V{"gemm", "posv", "zeta", "alpha"}));
}

TEST(TuneSchedule, BudgetBelowLatticeWarnsAndStillPlansTheLattice) {
    const std::vector<AxisSpec> axes{{"n", true, {"8", "16", "32", "64", "128", "256", "512"}}, {"batch", false, {"1024"}}};
    const auto lattice = tier_lattice(axes, Tier::preview);
    ASSERT_EQ(lattice.size(), 4u);
    const auto plan = plan_round(plan_spec(kTwo), Tier::preview, lattice, Ledger{}, kHashes, 4, 600);
    ASSERT_EQ(plan.size(), lattice.size()) << "the budget never trims the starting lattice";
    double est = 0;
    for (const PlannedCell& c : plan) {
        EXPECT_EQ(c.reason, "");
        est += c.est_s;
    }
    EXPECT_GT(est, 2400);
    const std::string w = budget_warning(est, 0.5);
    EXPECT_NE(w.find("0.50 h"), std::string::npos) << w;
    char e[32];
    std::snprintf(e, sizeof(e), "%.2f h", est / 3600);
    EXPECT_NE(w.find(e), std::string::npos) << w;
    EXPECT_EQ(budget_warning(est, 1.0), "");
    EXPECT_EQ(budget_warning(est, 0), "") << "no budget";
}

TEST(TuneSchedule, EstimateUsesTheNearestRecordElseTheByteModel) {
    Ledger l;
    EXPECT_DOUBLE_EQ(estimate_ms(l, nkey(64), "vendor", 1e9), 2.0) << "500 GB/s";
    l.cells.push_back(cell_rec({cres("vendor", "v1", "ok", 2.0)}, {"vendor"}, 64));
    l.cells.push_back(cell_rec({cres("vendor", "v1", "ok", 8.0)}, {"vendor"}, 256));
    CellRecord other = cell_rec({cres("vendor", "v1", "ok", 99.0)}, {"vendor"}, 100);
    other.key.insert(other.key.begin(), {"uplo", "U"});
    l.cells.push_back(other);
    EXPECT_DOUBLE_EQ(estimate_ms(l, nkey(100), "vendor", 1e9), 2.0);
    EXPECT_DOUBLE_EQ(estimate_ms(l, nkey(200), "vendor", 1e9), 8.0);
    EXPECT_DOUBLE_EQ(estimate_ms(l, nkey(100), "cta", 1e9), 2.0) << "no record of the candidate: the model";
    CellKey lower = nkey(100);
    lower.insert(lower.begin(), {"uplo", "L"});
    EXPECT_DOUBLE_EQ(estimate_ms(l, lower, "vendor", 1e9), 2.0) << "exact fields must match";
    const TierParams& p = params(Tier::coarse);
    EXPECT_DOUBLE_EQ(cell_estimate_s(l, nkey(64), {"vendor"}, 1e9, p, 0.49),
                     0.49 + p.max_reps * 2.0e-3 + p.warm_topup_s + kVerifyS);
}

TEST(TuneSchedule, RefineCellReadsTheRecord) {
    CellRecord r = cell_rec({cres("a", "1", "ok", 1.0), cres("b", "1", "bad"), cres("c", "1", "skipped"), cres("d", "1", "ok", 1.5)}, {"a", "d"});
    CandResult e = cres("e", "1", "eliminated");
    r.cands.push_back(e);
    e.cand = "f", e.median_ms = 3.0;
    r.cands.push_back(e);
    r.cands.push_back(cres("g", "1", "ok", NAN));
    const RefineCell c = refine_cell(r, true);
    EXPECT_EQ(c.ms, (std::map<std::string, double>{{"a", 1.0}, {"d", 1.5}, {"f", 3.0}}));
    EXPECT_EQ(c.out, (std::set<std::string>{"b", "c", "e"})) << "an untimed ok (skip:single) is runnable";
    EXPECT_TRUE(c.lattice);
    EXPECT_FALSE(refine_cell(r, false).lattice);
}

TEST(TuneSchedule, RefineRatioEstimateReadsTheTiersHistory) {
    Ledger l;
    bool hist = true;
    EXPECT_DOUBLE_EQ(refine_ratio_estimate(l, Tier::preview, 1.0, &hist), 0.5);
    EXPECT_FALSE(hist);
    EXPECT_DOUBLE_EQ(refine_ratio_estimate(l, Tier::deep, 2.0), 1.0);
    for (int round : {0, 0, 0, 0, 1, 2, 3}) {
        CellRecord r = cell_rec({}, {});
        r.tier = Tier::preview, r.round = round;
        l.cells.push_back(r);
    }
    CellRecord deep = cell_rec({}, {});
    deep.tier = Tier::deep, deep.round = 5;
    l.cells.push_back(deep);
    EXPECT_DOUBLE_EQ(refine_ratio_estimate(l, Tier::preview, 1.0, &hist), 0.75) << "3 refined / 4 lattice; the deep record is another tier";
    EXPECT_TRUE(hist);
    EXPECT_DOUBLE_EQ(refine_ratio_estimate(l, Tier::preview, 0.5), 0.5) << "never above the cap";
    EXPECT_DOUBLE_EQ(refine_ratio_estimate(l, Tier::coarse, 1.0, &hist), 0.5);
    EXPECT_FALSE(hist);
}

namespace {

// A host-only op for the tiered driver: winner b up to n=10, a above.
class FakeSpec : public OpSpec {
public:
    std::string op() const override { return "fakeop"; }
    std::vector<std::string> key_names() const override { return {"n:log:3"}; }
    std::vector<std::string> candidates(const std::string&) const override { return {"a", "b"}; }
    std::vector<std::pair<std::string, std::vector<std::string>>> axes() const override {
        return {{"n", {"1", "2", "4", "8", "16", "32", "64"}}};
    }
    double bytes(const std::string&, const CellKey& k) const override { return double(key_int(k, "n")) * 1e3; }
    std::vector<std::string> kernel_sources() const override { return {"k.cc", "b.cc"}; }
    std::string spec_file() const override { return "fake_spec.cc"; }
    std::string normalize_route(const std::string&, const std::string& algo) const override { return algo; }
    std::vector<ArmOutcome> run_cell(const CellRequest&) const override { return {}; }
};

const FakeSpec kFake;

class FakeMeasurer : public CellMeasurer {
public:
    std::vector<std::string> keys;
    int max_reps = 0;
    bool fail = false;  // every child times out
    ArmBatch measure(const CellJob& j) override {
        const CellKey& key = j.key;
        const std::vector<std::string>& arms = j.arms;
        keys.push_back(key_arg(key));
        max_reps = j.p.max_reps;
        ArmBatch b;
        if (fail) return {{}, "timeout"};
        const bool small = key_int(key, "n") <= 10;
        for (const std::string& a : arms) {
            const double t = a == "a" ? 1.0 : small ? 0.5 : 2.0;
            b.arms.push_back(arm(a, "ok", {t, t, t}));
        }
        return b;
    }
};

TieredOpts fake_opts(const TempDir& repo, const TempDir& ledger, Tier tier) {
    if (!find_spec("fakeop")) register_spec(&kFake);
    std::ofstream(repo.path / "k.cc") << "k";
    std::ofstream(repo.path / "b.cc") << "b";
    std::ofstream(repo.path / "fake_spec.cc")
        << "// kernel-sources-begin\n\"k.cc\",\n// family: b\n\"b.cc\",\n// kernel-sources-end\n";
    TieredOpts o;
    o.ops = {"fakeop"};
    o.dtypes = {"float"};
    o.devices = {0};
    o.tier = tier;
    o.repo = repo.str();
    o.ledger_root = ledger.str();
    o.argv = "tune_tests";
    return o;
}

std::vector<std::string> sorted(std::vector<std::string> v) {
    std::sort(v.begin(), v.end());
    return v;
}

}  // namespace

TEST(TuneTieredDriver, RoundsRefineToTheEdgeRecordAndSkipOnRerun) {
    TempDir repo, ledger;
    TieredOpts o = fake_opts(repo, ledger, Tier::preview);
    const fs::path events = ledger.path / "events.jsonl";
    const int fd = ::open(events.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    o.progress_fd = fd;
    FakeMeasurer m;
    ASSERT_EQ(run_tiered(o, {"sm_fake", "Fake"}, &m), 0);
    ::close(fd);
    // Lattice 1,4,16,64; index bisection 8; geometric 11, 9, 10 until the bracket is adjacent.
    EXPECT_EQ(std::vector<std::string>(m.keys.begin(), m.keys.begin() + 4), (std::vector<std::string>{"n=1", "n=4", "n=16", "n=64"}));
    EXPECT_EQ(sorted(m.keys), sorted({"n=1", "n=4", "n=16", "n=64", "n=8", "n=11", "n=9", "n=10"}));
    EXPECT_EQ(m.max_reps, params(Tier::preview).max_reps);
    const Ledger l = read_ledger(ledger_dir(ledger.str(), "fakeop", "float", "sm_fake"));
    ASSERT_EQ(l.runs.size(), 1u);
    EXPECT_EQ(l.runs[0].tier, Tier::preview);
    EXPECT_EQ(l.runs[0].keys, "n:log:3");
    EXPECT_EQ(l.runs[0].candidates, "a|b");
    ASSERT_EQ(l.cells.size(), 8u);
    for (const CellRecord& c : l.cells) {
        EXPECT_EQ(c.ranked.front(), key_int(c.key, "n") <= 10 ? "b" : "a") << key_arg(c.key);
        EXPECT_EQ(c.cands.size(), 2u);
        EXPECT_EQ(c.cands[1].hash, *kernel_hash(repo.str(), {"k.cc", "b.cc"}));
    }
    const std::string ev = read_file(events);
    EXPECT_EQ(ev.rfind("{\"ev\": \"plan\", \"cells\": 4", 0), 0u) << ev;
    EXPECT_NE(ev.find("{\"ev\": \"done\"}"), std::string::npos);
    std::size_t done = 0;
    for (std::size_t p = 0; (p = ev.find("\"cell_done\"", p)) != std::string::npos; ++p) ++done;
    EXPECT_EQ(done, 8u);

    FakeMeasurer again;
    o.progress_fd = -1;
    ASSERT_EQ(run_tiered(o, {"sm_fake", "Fake"}, &again), 0);
    EXPECT_TRUE(again.keys.empty()) << "every cell is current at preview";

    FakeMeasurer coarse;
    o.tier = Tier::coarse;
    ASSERT_EQ(run_tiered(o, {"sm_fake", "Fake"}, &coarse), 0);
    EXPECT_EQ(sorted(coarse.keys), sorted({"n=1", "n=2", "n=4", "n=8", "n=16", "n=32", "n=64", "n=11", "n=9", "n=10"}))
        << "preview records are no bracket ends for coarse: its midpoints are measured again";
}

TEST(TuneTieredDriver, BudgetStopsRefinementButNotTheLattice) {
    TempDir repo, ledger;
    TieredOpts o = fake_opts(repo, ledger, Tier::preview);
    o.budget_h = 1e-12;
    FakeMeasurer m;
    ASSERT_EQ(run_tiered(o, {"sm_fake", "Fake"}, &m), 0);
    EXPECT_EQ(m.keys, (std::vector<std::string>{"n=1", "n=4", "n=16", "n=64"}));
    o.plan = true;
    ASSERT_EQ(run_tiered(o, {"sm_fake", "Fake"}, nullptr), 0) << "--plan needs no measurer";
}

namespace {

// m is derived (4 n), as geqrf's and orgqr's extents are: the keys alone name only n.
class DerivedDimsSpec : public FakeSpec {
public:
    std::string op() const override { return "derivedop"; }
    std::vector<std::int64_t> dims(const std::string&, const CellKey& k) const override { return {4 * key_int(k, "n"), key_int(k, "n")}; }
};

std::int64_t max_of(const std::vector<std::int64_t>& v) { return v.empty() ? 0 : *std::max_element(v.begin(), v.end()); }

}  // namespace

TEST(TuneMaxDim, DefaultDimsTakeEveryIntegerKeyButBatch) {
    const FakeSpec s;
    const CellKey k{{"uplo", "L"}, {"n", "64"}, {"nrhs", "3"}, {"batch", "99999"}, {"form", "tall"}};
    EXPECT_EQ(s.dims("float", k), (std::vector<std::int64_t>{64, 3})) << "words and batch are no extent";
    EXPECT_EQ(max_of(s.dims("float", {{"n", "4096"}, {"batch", "2"}})), 4096);
}

TEST(TuneMaxDim, DerivedDimsCapOnTheDerivedExtent) {
    const DerivedDimsSpec s;
    const CellKey k{{"n", "1024"}, {"batch", "8"}};
    EXPECT_EQ(max_of(s.dims("float", k)), 4096) << "n alone is under 2048, the derived m is not";
    PlanSpec ps = plan_spec(kTwo);
    ps.max_dim = [&](const CellKey& c) { return max_of(s.dims("float", c)); };
    EXPECT_EQ(plan_round(ps, Tier::coarse, {k}, Ledger{}, kHashes, 4, 0, nullptr, 2048)[0].reason, "skip:dim");
    EXPECT_EQ(plan_round(ps, Tier::coarse, {nkey(512)}, Ledger{}, kHashes, 4, 0, nullptr, 2048)[0].reason, "")
        << "4 x 512 = 2048 is allowed";
}

TEST(TuneMaxDim, PlanRoundMarksSkipDimBeforeCapAndLeavesTheRestAlone) {
    PlanSpec ps = plan_spec(kTwo);
    ps.max_dim = [](const CellKey& k) { return key_int(k, "n"); };
    const auto plan = plan_round(ps, Tier::coarse, {nkey(2048), nkey(2049), nkey(64, 1 << 20)}, Ledger{}, kHashes,
                                 1e-9, 0, nullptr, 2048);
    std::map<std::string, std::string> why;
    for (const PlannedCell& c : plan) why[key_arg(c.key)] = c.reason;
    EXPECT_EQ(why["n=2049,batch=1024"], "skip:dim");
    EXPECT_EQ(why["n=2048,batch=1024"], "skip:cap") << "2048 is not above the limit; the byte cap still applies";
    EXPECT_EQ(why["n=64,batch=1048576"], "skip:cap") << "batch is no dimension";
    for (const PlannedCell& c : plan) if (c.reason == "skip:dim") EXPECT_TRUE(c.arms.empty());
    EXPECT_EQ(plan_round(ps, Tier::coarse, {nkey(4096)}, Ledger{}, kHashes, 4, 0)[0].reason, "")
        << "max_dim 0 is off";
}

TEST(TuneMaxDim, TieredRunNeverMeasuresACellOverTheLimit) {
    TempDir repo, ledger;
    TieredOpts o = fake_opts(repo, ledger, Tier::preview);
    o.max_dim = 8;
    FakeMeasurer m;
    ASSERT_EQ(run_tiered(o, {"sm_fake", "Fake"}, &m), 0);
    ASSERT_FALSE(m.keys.empty());
    for (const std::string& k : m.keys) EXPECT_LE(std::stoi(k.substr(2)), 8) << k;
}

TEST(TuneSchedule, AllErrorRecordCountsAsMissing) {
    Ledger l;
    CellRecord failed = cell_rec({cres("lpanel:panel=8", "h1", "error"), cres("vendor", "v1", "error")}, {});
    failed.tier = Tier::deep;
    l.cells.push_back(failed);
    EXPECT_EQ(plan_round(plan_spec(kTwo), Tier::coarse, {nkey(64)}, l, kHashes, 4, 0)[0].reason, "")
        << "a failed child is no result";
    l.cells[0].cands[1].status = "bad";
    EXPECT_EQ(plan_round(plan_spec(kTwo), Tier::coarse, {nkey(64)}, l, kHashes, 4, 0)[0].reason, "skip:current")
        << "a verification failure is a result";
}

TEST(TuneTieredDriver, TransientChildFailureWritesNoRecord) {
    TempDir repo, ledger;
    TieredOpts o = fake_opts(repo, ledger, Tier::preview);
    FakeMeasurer broken;
    broken.fail = true;
    ASSERT_EQ(run_tiered(o, {"sm_fake", "Fake"}, &broken), 0);
    EXPECT_EQ(broken.keys.size(), 4u);
    EXPECT_TRUE(read_ledger(ledger_dir(ledger.str(), "fakeop", "float", "sm_fake")).cells.empty());
    FakeMeasurer m;
    ASSERT_EQ(run_tiered(o, {"sm_fake", "Fake"}, &m), 0);
    EXPECT_EQ(m.keys.size(), 8u) << "the next run measures every cell";
}

// ---- Task 8: race records, the worker wire format, the fresh-process audit --------------------

TEST(TuneSchedule, AnEliminatedArmThatFailsVerificationLeavesTheRow) {
    // run_race verifies eliminated arms too: one that fails comes back "bad", with its race reps.
    const CellRecord r = record_from_arms(nkey(64), 0,
                                          {arm("a", "bad", {0.5, 0.5, 0.5}), arm("b", "ok", {1.0, 1.0, 1.1}),
                                           arm("c", "eliminated", {2.0, 2.1, 2.2})},
                                          {"a", "b", "c"}, {});
    EXPECT_EQ(r.ranked, (std::vector<std::string>{"b", "c"}));
    ASSERT_EQ(r.cands.size(), 3u);
    EXPECT_EQ(r.cands[0].status, "bad");
    EXPECT_TRUE(std::isnan(r.cands[0].median_ms)) << "a wrong answer prints no time";
}

TEST(TuneSchedule, EliminatedArmsKeepTheirMedianAndRankAfterSurvivors) {
    const std::vector<std::string> order{"a", "b", "c"};
    const CellRecord r = record_from_arms(nkey(64), 0,
                                          {arm("a", "eliminated", {3.0, 3.2, 3.1}), arm("b", "ok", {1.0, 1.0, 1.1}),
                                           arm("c", "eliminated", {2.0, 2.1, 2.2})},
                                          order, {});
    EXPECT_EQ(r.ranked, (std::vector<std::string>{"b", "c", "a"}));
    ASSERT_EQ(r.cands.size(), 3u);
    EXPECT_EQ(r.cands[0].status, "eliminated");
    EXPECT_DOUBLE_EQ(r.cands[0].median_ms, 3.1) << "the ledger needs the eliminated median";
    EXPECT_EQ(r.cands[0].reps, 3);
    const CellRecord bad = record_from_arms(nkey(64), 0, {arm("a", "eliminated", {3.0}), arm("b", "bad", {1.0})}, order, {});
    EXPECT_EQ(bad.ranked, (std::vector<std::string>{"a"})) << "a bad survivor drops out; the next arm wins";
}

TEST(TuneSchedule, SeedOrderPutsTheNearestWinnerFirst) {
    std::map<CellKey, CellRecord> done;
    done[nkey(16)] = cell_rec({}, {"c", "a"}, 16);
    done[nkey(512)] = cell_rec({}, {"b"}, 512);
    done[nkey(16)].key = nkey(16);
    done[nkey(512)].key = nkey(512);
    EXPECT_EQ(seed_order(done, nkey(32), {"a", "b", "c"}), (std::vector<std::string>{"c", "a", "b"}));
    EXPECT_EQ(seed_order(done, nkey(400), {"a", "b", "c"}), (std::vector<std::string>{"b", "a", "c"}));
    EXPECT_EQ(seed_order({}, nkey(32), {"a", "b"}), (std::vector<std::string>{"a", "b"}));
    EXPECT_EQ(seed_order(done, nkey(32), {"a", "b"}), (std::vector<std::string>{"a", "b"})) << "winner not raced here";
}

TEST(TuneSchedule, AuditComparesFeasibilityAndTheWinnerBeyondTheTie) {
    const std::vector<std::string> order{"a", "b", "c"};
    const std::vector<ArmOutcome> warm{arm("a", "ok", {1.0}), arm("b", "eliminated", {2.0}), arm("c", "skipped")};
    EXPECT_EQ(audit_compare(warm, warm, order).verdict, "ok");
    const AuditResult same = audit_compare(warm, {arm("a", "ok", {1.1}), arm("b", "ok", {2.0}), arm("c", "skipped")}, order);
    EXPECT_EQ(same.verdict, "ok") << "eliminated and ok are both feasible";
    EXPECT_DOUBLE_EQ(same.warm_ms, 1.0);
    EXPECT_DOUBLE_EQ(same.fresh_ms, 1.1);
    const AuditResult feas = audit_compare(warm, {arm("a", "ok", {1.0}), arm("b", "error"), arm("c", "skipped")}, order);
    EXPECT_TRUE(feas.mismatch()) << feas.verdict;
    EXPECT_NE(feas.verdict.find("b eliminated/error"), std::string::npos) << feas.verdict;
    EXPECT_TRUE(audit_compare(warm, {arm("a", "ok", {1.0}), arm("b", "eliminated", {2.0}), arm("c", "ok", {0.5})}, order)
                    .mismatch()) << "a candidate the warm worker refused runs in a fresh process";
    const std::vector<ArmOutcome> warm_b{arm("a", "eliminated", {2.0}), arm("b", "ok", {1.0}), arm("c", "skipped")};
    EXPECT_EQ(audit_compare(warm_b, {arm("a", "ok", {1.0}), arm("b", "ok", {1.02}), arm("c", "skipped")}, order).verdict, "ok")
        << "winners differ (a by tie order), but the worker's winner b is within 10% in the fresh run";
    EXPECT_EQ(audit_compare(warm_b, {arm("a", "ok", {1.0}), arm("b", "ok", {1.08}), arm("c", "skipped")}, order).verdict, "ok")
        << "a 3-10% swap is noise at the audit's rep counts";
    EXPECT_EQ(audit_compare(warm_b, {arm("a", "ok", {1.0}), arm("b", "ok", {1.12}), arm("c", "skipped")}, order).verdict,
              "mismatch:winner b/a");
    const std::vector<ArmOutcome> warm_elim{arm("a", "ok", {1.0}), arm("b", "eliminated", {2.0}), arm("c", "ok", {3.0})};
    EXPECT_EQ(audit_compare(warm_elim, {arm("a", "ok", {1.0}), arm("b", "bad", {2.0}), arm("c", "ok", {3.0})}, order).verdict,
              "mismatch:feasibility b eliminated/bad") << "run_race verifies eliminated arms too";
    EXPECT_TRUE(audit_compare(warm_elim, {arm("a", "ok", {1.0}), arm("b", "ok", {2.0}), arm("c", "bad", {3.0})}, order)
                    .mismatch()) << "ok in the worker, bad in a fresh process";
    EXPECT_TRUE(audit_compare(warm_elim, {arm("a", "ok", {1.0}), arm("b", "skipped"), arm("c", "ok", {3.0})}, order)
                    .mismatch()) << "eliminated (it launched) vs refused";
    const AuditResult win = audit_compare(warm, {arm("a", "eliminated", {1.2}), arm("b", "ok", {1.0}), arm("c", "skipped")}, order);
    EXPECT_EQ(win.verdict, "mismatch:winner a/b");
    EXPECT_EQ(audit_compare(warm, {arm("a", "error"), arm("b", "error"), arm("c", "error")}, order).verdict, "inconclusive");
}

TEST(TuneSchedule, AuditPickIsAStableHashFraction) {
    int picked = 0;
    for (int n = 1; n <= 2000; ++n) picked += audit_pick("run1", nkey(n), 0.10);
    EXPECT_GT(picked, 140);
    EXPECT_LT(picked, 260);
    EXPECT_EQ(audit_pick("run1", nkey(7), 0.10), audit_pick("run1", nkey(7), 0.10));
    EXPECT_FALSE(audit_pick("run1", nkey(7), 0.0));
    EXPECT_TRUE(audit_pick("run1", nkey(7), 1.0));
}

TEST(TuneLedger, UpdatedRunLineReplacesTheEarlierOne) {
    TempDir d;
    RunMeta m;
    m.run_id = "r1";
    m.tier = Tier::preview;
    m.worker_mode["potrf.float"] = "worker";
    {
        LedgerWriter w(d.str(), m);
        w.cell(cell_rec({cres("a", "1")}, {"a"}));
        m.worker_mode["potrf.float"] = "fresh";
        w.update_run(m);
        w.audit(nkey(64), "mismatch:winner a/b", 1.0, 2.0);
    }
    const Ledger l = read_ledger(d.str());
    ASSERT_EQ(l.runs.size(), 1u);
    EXPECT_EQ(l.runs[0].worker_mode.at("potrf.float"), "fresh");
    EXPECT_EQ(l.cells.size(), 1u);
}

TEST(TuneWorker, RequestAndOutcomeRoundTrip) {
    CellRequest r;
    r.dtype = "cfloat";
    r.key = nkey(32);
    r.arms = {"cta", "lpanel:panel=8"};
    r.mode = "race";
    r.warm_s = 0.2;
    r.ld_pad = 3;
    r.tier = "deep";
    r.min_reps = 6;
    r.max_reps = 16;
    r.confidence = 0.98;
    r.alternate_reverse = true;
    r.seed_order = {"lpanel:panel=8"};
    const auto got = parse_request(request_line("potrf", r));
    ASSERT_TRUE(got);
    EXPECT_EQ(got->first, "potrf");
    const CellRequest& g = got->second;
    EXPECT_EQ(g.dtype, "cfloat");
    EXPECT_EQ(g.key, r.key);
    EXPECT_EQ(g.arms, r.arms);
    EXPECT_EQ(g.mode, "race");
    EXPECT_DOUBLE_EQ(g.warm_s, 0.2);
    EXPECT_EQ(g.ld_pad, 3);
    EXPECT_EQ(g.tier, "deep");
    EXPECT_EQ(g.min_reps, 6);
    EXPECT_EQ(g.max_reps, 16);
    EXPECT_DOUBLE_EQ(g.confidence, 0.98);
    EXPECT_TRUE(g.alternate_reverse);
    EXPECT_EQ(g.seed_order, r.seed_order);
    EXPECT_FALSE(parse_request("{\"op\": \"potrf\"}"));
    EXPECT_FALSE(parse_request("not json"));

    ArmOutcome a = arm("cta", "eliminated", {1.5, 1.25});
    a.slot = {1, 0};
    a.reason = "round 2";
    std::vector<Record> recs;
    for (const std::string& line : split(outcome_text({a, arm("tiny", "skipped")}), '\n')) recs.push_back(*parse_record(line));
    const auto back = outcomes_from_records(recs);
    ASSERT_EQ(back.size(), 2u);
    EXPECT_EQ(back[0].arm, "cta");
    EXPECT_EQ(back[0].status, "eliminated");
    EXPECT_EQ(back[0].reason, "round 2");
    EXPECT_EQ(back[0].ms, a.ms);
    EXPECT_EQ(back[0].slot, a.slot);
    EXPECT_EQ(back[1].status, "skipped");
}

TEST(TuneWorker, ProcessReturnsRecordsUntilDoneThenEofAndTimesOut) {
    WorkerProcess w;
    ASSERT_TRUE(w.start({"/bin/sh", "-c",
                         "read l; echo noise; echo '{\"kind\": \"arm\", \"arm\": \"x\"}'; echo '{\"kind\": \"done\"}'; "
                         "read l; exit 0"},
                        {}, ""));
    ASSERT_TRUE(w.send("{}\n"));
    std::vector<Record> got;
    EXPECT_EQ(w.receive(10, &got), WorkerProcess::Got::done);
    ASSERT_EQ(got.size(), 1u);
    EXPECT_EQ(got[0].get("arm"), "x");
    ASSERT_TRUE(w.send("{}\n"));
    got.clear();
    EXPECT_EQ(w.receive(10, &got), WorkerProcess::Got::eof) << "the worker exited before answering";
    w.stop();
    EXPECT_FALSE(w.alive());
    ASSERT_TRUE(w.start({"/bin/sh", "-c", "exec sleep 30"}, {}, ""));
    EXPECT_EQ(w.receive(0.2, &got), WorkerProcess::Got::timeout);
    w.stop();
}

namespace {

// A persistent measurer whose fresh child disagrees on feasibility at every cell above n=4.
class AuditMeasurer : public FakeMeasurer {
public:
    bool disagree = true;
    int restart_at = -1;  // the worker restarts on the cell with this n
    std::vector<std::string> fresh_keys, worker_keys;
    std::vector<double> footprints;
    bool persistent() const override { return true; }
    ArmBatch measure(const CellJob& j) override {
        worker_keys.push_back(key_arg(j.key));
        footprints.push_back(j.footprint);
        ArmBatch b = FakeMeasurer::measure(j);
        const std::string slower = key_int(j.key, "n") <= 10 ? "a" : "b";
        for (ArmOutcome& a : b.arms)
            if (a.arm == slower) a.status = "eliminated";
        b.worker_restarts = key_int(j.key, "n") == restart_at;
        return b;
    }
    ArmBatch measure_fresh(const CellJob& j) override {
        fresh_keys.push_back(key_arg(j.key));
        ArmBatch b = FakeMeasurer::measure(j);
        for (ArmOutcome& a : b.arms)
            if (disagree && key_int(j.key, "n") > 4 && a.arm == "a") a.status = "error";
        return b;
    }
};

std::size_t count_of(const std::string& hay, const std::string& needle) {
    std::size_t n = 0;
    for (std::size_t p = 0; (p = hay.find(needle, p)) != std::string::npos; ++p) ++n;
    return n;
}

}  // namespace

TEST(TuneTieredDriver, AuditMismatchSendsTheRestOfTheOpDtypeToFreshChildren) {
    TempDir repo, ledger;
    TieredOpts o = fake_opts(repo, ledger, Tier::preview);
    o.audit_fraction = 1.0;
    const fs::path events = ledger.path / "events.jsonl";
    const int fd = ::open(events.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    o.progress_fd = fd;
    AuditMeasurer m;
    ASSERT_EQ(run_tiered(o, {"sm_fake", "Fake"}, &m), 0);
    ::close(fd);
    // Lattice n=1,4,16,64: 1 and 4 agree; 16 is the first mismatch; 64 and every refinement cell run fresh only.
    ASSERT_GE(m.worker_keys.size(), 3u);
    EXPECT_EQ(std::vector<std::string>(m.worker_keys.begin(), m.worker_keys.begin() + 3), (std::vector<std::string>{"n=1", "n=4", "n=16"}));
    EXPECT_EQ(m.worker_keys.size(), 3u) << "no worker measurement after the mismatch";
    ASSERT_GE(m.fresh_keys.size(), 4u);
    EXPECT_EQ(std::vector<std::string>(m.fresh_keys.begin(), m.fresh_keys.begin() + 4),
              (std::vector<std::string>{"n=1", "n=4", "n=16", "n=64"}));
    const std::string dir = ledger_dir(ledger.str(), "fakeop", "float", "sm_fake");
    const Ledger l = read_ledger(dir);
    ASSERT_EQ(l.runs.size(), 1u);
    EXPECT_EQ(l.runs[0].worker_mode.at("fakeop.float"), "fresh");
    std::string text;
    for (const auto& e : fs::directory_iterator(dir)) text += read_file(e.path());
    EXPECT_EQ(count_of(text, "\"kind\": \"audit\""), 3u) << text;
    EXPECT_EQ(count_of(text, "\"verdict\": \"ok\""), 2u);
    EXPECT_NE(text.find("\"wm.fakeop.float\": \"worker\""), std::string::npos) << "the run starts on the worker";
    const std::string ev = read_file(events);
    EXPECT_EQ(count_of(ev, "\"ev\": \"audit\""), 3u) << ev;
    EXPECT_NE(ev.find("\"cand\": \"a\", \"round\": 3"), std::string::npos) << ev;
    EXPECT_EQ(count_of(ev, "\"ev\": \"eliminated\""), m.worker_keys.size()) << "fresh-only cells here race nothing out" << ev;
}

TEST(TuneTieredDriver, AuditMatchKeepsTheWorkerAndReportsRestarts) {
    TempDir repo, ledger;
    TieredOpts o = fake_opts(repo, ledger, Tier::preview);
    o.audit_fraction = 1.0;
    const fs::path events = ledger.path / "events.jsonl";
    const int fd = ::open(events.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    o.progress_fd = fd;
    AuditMeasurer m;
    m.disagree = false;
    m.restart_at = 16;
    ASSERT_EQ(run_tiered(o, {"sm_fake", "Fake"}, &m), 0);
    ::close(fd);
    EXPECT_EQ(m.worker_keys.size(), 8u);
    EXPECT_EQ(m.fresh_keys, m.worker_keys) << "every cell audited, none moved to fresh children";
    EXPECT_EQ(m.footprints, (std::vector<double>{1e3, 4e3, 16e3, 64e3, 8e3, 11e3, 9e3, 10e3})) << "no batch key: bytes";
    const Ledger l = read_ledger(ledger_dir(ledger.str(), "fakeop", "float", "sm_fake"));
    EXPECT_EQ(l.runs.at(0).worker_mode.at("fakeop.float"), "worker");
    const std::string ev = read_file(events);
    EXPECT_EQ(count_of(ev, "\"ev\": \"worker_restart\""), 1u) << ev;
    EXPECT_NE(ev.find("\"restarts\": 1"), std::string::npos) << ev;
}

namespace {

// Thread-safe, several GPUs, uneven per-cell sleeps; float n=`slow_n` is the straggler.
class PipeMeasurer : public CellMeasurer {
public:
    struct Call {
        std::string dtype, key;
        int gpu = 0;
        double footprint = 0, t0 = 0, t1 = 0;
        std::size_t done_before = 0;  // this dtype's cell_done events when the cell started
    };
    fs::path events;
    int slow_n = -1;
    std::mutex mu;
    std::vector<Call> calls;
    ArmBatch measure(const CellJob& j) override {
        Call c{j.dtype, key_arg(j.key), j.gpu, j.footprint, now(), 0, 0};
        if (!events.empty()) c.done_before = count_of(read_file(events), "\"cell_done\", \"op\": \"fakeop\", \"dtype\": \"" + j.dtype + "\"");
        const std::int64_t n = key_int(j.key, "n");
        const bool slow = j.dtype == "float" && n == slow_n;
        std::this_thread::sleep_for(std::chrono::milliseconds(slow ? 400 : 2 + (n * 7 + std::int64_t(j.dtype.size()) * 3) % 5 * 3));
        ArmBatch b;
        for (const std::string& a : j.arms) {
            const double t = a == "a" ? 1.0 : n <= 10 ? 0.5 : 2.0;
            b.arms.push_back(arm(a, "ok", {t, t, t}));
        }
        c.t1 = now();
        std::lock_guard<std::mutex> lock(mu);
        calls.push_back(c);
        return b;
    }

private:
    const std::chrono::steady_clock::time_point t0_ = std::chrono::steady_clock::now();
    double now() const { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_).count(); }
};

// (dtype, key) -> (round, ranked) from a run's ledgers.
std::map<std::pair<std::string, std::string>, std::pair<int, std::string>> records_of(const TempDir& ledger, const std::vector<std::string>& dtypes) {
    std::map<std::pair<std::string, std::string>, std::pair<int, std::string>> out;
    for (const std::string& d : dtypes)
        for (const CellRecord& c : read_ledger(ledger_dir(ledger.str(), "fakeop", d, "sm_fake")).cells) {
            std::string ranked;
            for (const std::string& r : c.ranked) ranked += r + "|";
            out[{d, key_arg(c.key)}] = {c.round, ranked};
        }
    return out;
}

const std::vector<std::string> kPipeDtypes{"float", "double", "cfloat"};

TieredOpts pipe_opts(const TempDir& repo, const TempDir& ledger, std::vector<int> devices, Tier tier = Tier::preview) {
    TieredOpts o = fake_opts(repo, ledger, tier);
    o.dtypes = kPipeDtypes;
    o.devices = std::move(devices);
    return o;
}

}  // namespace

TEST(TuneTieredPipeline, IdleGpusRunLaterJobsWhileAStragglerFinishesItsRound) {
    TempDir repo, ledger;
    TieredOpts o = pipe_opts(repo, ledger, {0, 1, 2, 3});
    const fs::path events = ledger.path / "events.jsonl";
    const int fd = ::open(events.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    o.progress_fd = fd;
    PipeMeasurer m;
    m.events = events;
    m.slow_n = 64;
    ASSERT_EQ(run_tiered(o, {"sm_fake", "Fake"}, &m), 0);
    ::close(fd);
    double slow_end = 0;
    for (const auto& c : m.calls)
        if (c.dtype == "float" && c.key == "n=64") slow_end = c.t1;
    ASSERT_GT(slow_end, 0.3);
    std::size_t later = 0;
    for (const auto& c : m.calls)
        if (c.dtype != "float") {
            ++later;
            EXPECT_LT(c.t1, slow_end) << c.dtype << " " << c.key << ": waited for float's straggler";
        }
    EXPECT_EQ(later, 16u) << "double and cfloat: 4 lattice + 4 refinement cells each";
    const auto recs = records_of(ledger, kPipeDtypes);
    for (const auto& c : m.calls) {
        const int round = recs.at({c.dtype, c.key}).first;
        std::size_t earlier = 0;
        for (const auto& [k, r] : recs) earlier += k.first == c.dtype && r.first < round;
        EXPECT_GE(c.done_before, earlier) << c.dtype << " " << c.key << " round " << round
                                          << " started before every earlier-round cell was recorded";
    }
    EXPECT_NE(read_file(events).find("\"ev\": \"schedule\", \"switches\": "), std::string::npos);
}

TEST(TuneTieredPipeline, EachGpuRunsAscendingFootprintsWithinAJobRound) {
    TempDir repo, ledger;
    TieredOpts o = pipe_opts(repo, ledger, {0, 1}, Tier::coarse);
    const fs::path events = ledger.path / "events.jsonl";
    const int fd = ::open(events.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    o.progress_fd = fd;
    PipeMeasurer m;
    ASSERT_EQ(run_tiered(o, {"sm_fake", "Fake"}, &m), 0);
    ::close(fd);
    const auto recs = records_of(ledger, kPipeDtypes);
    std::map<int, std::vector<const PipeMeasurer::Call*>> per_gpu;
    std::vector<PipeMeasurer::Call> calls = m.calls;
    std::sort(calls.begin(), calls.end(), [](const auto& a, const auto& b) { return a.t0 < b.t0; });
    for (const auto& c : calls) per_gpu[c.gpu].push_back(&c);
    ASSERT_EQ(per_gpu.size(), 2u);
    std::size_t restarts = 0, same = 0;
    for (const auto& [gpu, seq] : per_gpu) {
        double max = 0;
        for (std::size_t i = 0; i < seq.size(); ++i) {
            restarts += seq[i]->footprint < max;
            max = seq[i]->footprint < max ? seq[i]->footprint : std::max(max, seq[i]->footprint);
            if (i == 0 || seq[i]->dtype != seq[i - 1]->dtype) continue;
            const int r0 = recs.at({seq[i - 1]->dtype, seq[i - 1]->key}).first, r1 = recs.at({seq[i]->dtype, seq[i]->key}).first;
            if (r0 != r1) continue;
            ++same;
            EXPECT_GE(seq[i]->footprint, seq[i - 1]->footprint) << "gpu" << gpu << " " << seq[i]->dtype << " round " << r1;
        }
    }
    EXPECT_GE(same, 6u) << "the coarse lattice's 7 cells per dtype split over two GPUs";
    EXPECT_NE(read_file(events).find("\"restarts\": " + std::to_string(restarts) + ","), std::string::npos)
        << "the summary counts the carve-out restarts the pop order costs";
}

TEST(TuneTieredPipeline, RecordsMatchTheSingleGpuRun) {
    TempDir repo, one, four;
    TieredOpts o = pipe_opts(repo, one, {0});
    PipeMeasurer m1;
    ASSERT_EQ(run_tiered(o, {"sm_fake", "Fake"}, &m1), 0);
    o = pipe_opts(repo, four, {0, 1, 2, 3});
    PipeMeasurer m4;
    m4.slow_n = 16;
    ASSERT_EQ(run_tiered(o, {"sm_fake", "Fake"}, &m4), 0);
    const auto a = records_of(one, kPipeDtypes), b = records_of(four, kPipeDtypes);
    EXPECT_EQ(a, b);
    ASSERT_EQ(a.size(), 24u);
    for (const std::string& d : kPipeDtypes)
        for (const char* k : {"n=1", "n=4", "n=16", "n=64", "n=8", "n=11", "n=9", "n=10"}) EXPECT_TRUE(a.count({d, k})) << d << " " << k;
}

TEST(TuneSchedule, WorkerShareRunsInAscendingPerItemFootprint) {
    auto key = [](int n, int batch) { return CellKey{{"n", std::to_string(n)}, {"batch", std::to_string(batch)}}; };
    std::vector<PlannedCell> cells(4);
    cells[0].key = key(64, 128);    // 4096 per item, 524288 bytes
    cells[1].key = key(16, 32768);  // 256 per item, the most bytes
    cells[2].key = key(32, 512);    // 1024 per item
    cells[3].key = key(16, 128);    // 256 per item, fewer bytes
    std::vector<const PlannedCell*> share{&cells[0], &cells[1], &cells[2], &cells[3]};
    sort_for_worker(share, [](const CellKey& k) { return double(key_int(k, "n") * key_int(k, "n") * key_int(k, "batch")); });
    std::vector<std::string> got;
    for (const PlannedCell* c : share) got.push_back(key_arg(c->key));
    EXPECT_EQ(got, (std::vector<std::string>{"n=16,batch=128", "n=16,batch=32768", "n=32,batch=512", "n=64,batch=128"}));
    EXPECT_DOUBLE_EQ(item_footprint(1024, {{"n", "4"}}), 1024) << "no batch key";
}

TEST(TuneWorker, GateRestartsBeforeASmallerFootprintAndSchedulesFullGuards) {
    WorkerGate g(60);
    g.started();
    EXPECT_TRUE(g.full_guard_due(0)) << "first cell after a start";
    g.guarded(0);
    EXPECT_FALSE(g.full_guard_due(30));
    EXPECT_TRUE(g.full_guard_due(61));
    EXPECT_FALSE(g.restart_before(100));
    g.ran(100);
    EXPECT_FALSE(g.restart_before(100)) << "equal footprint: the same launch shape";
    EXPECT_FALSE(g.restart_before(400));
    g.ran(400);
    EXPECT_TRUE(g.restart_before(399)) << "a smaller per-item footprint after a larger launch";
    g.started();
    EXPECT_FALSE(g.restart_before(1));
    EXPECT_TRUE(g.full_guard_due(1)) << "a restarted worker is guarded fully again";
}

TEST(TuneWorker, StickyErrorArmRestartsTheWorkerAndRecordsTheFreshChild) {
    int tries = 0, restarts = 0, fresh = 0;
    auto run = [&](std::vector<WorkerTry> script) {
        tries = restarts = fresh = 0;
        ArmErrors errs;
        return race_on_worker({"a", "b"}, errs, [&](const std::vector<std::string>&) { return script.at(std::size_t(tries++)); },
                              [&] { ++restarts; },
                              [&](const std::vector<std::string>&) {
                                  ++fresh;
                                  return ArmBatch{{arm("a", "ok", {1.0}), arm("b", "ok", {2.0})}, "", 0, false};
                              });
    };
    WorkerTry good{true, false, "", {arm("a", "ok", {1.0}), arm("b", "eliminated", {2.0})}};
    WorkerTry sticky{true, false, "", {arm("a", "ok", {1.0}), arm("b", "error")}};
    WorkerTry died{false, false, "worker exited", {}};
    WorkerTry guard{false, true, "guard", {}};
    ArmBatch b = run({good});
    EXPECT_FALSE(b.fallback);
    EXPECT_EQ(b.arms[1].status, "eliminated");
    EXPECT_EQ(restarts + fresh, 0);
    b = run({sticky});
    EXPECT_TRUE(b.fallback);
    EXPECT_EQ(tries, 1) << "no second worker try: the fresh child decides";
    EXPECT_EQ(restarts, 1);
    EXPECT_EQ(fresh, 1);
    EXPECT_EQ(b.worker_restarts, 1);
    EXPECT_EQ(b.arms[1].status, "ok") << "the fresh child's result is recorded";
    b = run({died, good});
    EXPECT_FALSE(b.fallback);
    EXPECT_EQ(b.worker_restarts, 1);
    b = run({died, died});
    EXPECT_TRUE(b.fallback);
    EXPECT_EQ(restarts, 2);
    b = run({guard, guard});
    EXPECT_EQ(restarts, 0) << "a guard discard is not the worker's fault";
    EXPECT_TRUE(b.fallback);
}

TEST(TuneWorker, TwoConfirmedErrorsTakeAnArmOffTheWorker) {
    ArmErrors errs;
    std::vector<std::vector<std::string>> sent, fresh_sent;
    bool fresh_b_errors = true;
    auto cell = [&] {
        return race_on_worker(
            {"a", "b"}, errs,
            [&](const std::vector<std::string>& arms) {
                sent.push_back(arms);
                WorkerTry t{true, false, "", {}};
                for (const std::string& a : arms) t.arms.push_back(a == "b" ? arm("b", "error") : arm(a, "ok", {1.0}));
                return t;
            },
            [] {},
            [&](const std::vector<std::string>& arms) {
                fresh_sent.push_back(arms);
                ArmBatch b;
                for (const std::string& a : arms)
                    b.arms.push_back(a == "b" && fresh_b_errors ? arm("b", "error") : arm(a, "ok", {a == "b" ? 2.0 : 1.0}));
                return b;
            });
    };
    using V = std::vector<std::vector<std::string>>;
    cell();
    fresh_b_errors = false;
    cell();  // the worker's error is not reproduced: worker poisoning, the streak resets
    fresh_b_errors = true;
    cell();
    EXPECT_EQ(sent, (V{{"a", "b"}, {"a", "b"}, {"a", "b"}}));
    EXPECT_FALSE(errs.benched("b")) << "confirmed, unconfirmed, confirmed: not consecutive";
    ArmBatch b = cell();
    EXPECT_TRUE(b.fallback);
    EXPECT_TRUE(errs.benched("b"));
    EXPECT_FALSE(errs.benched("a"));
    sent.clear(), fresh_sent.clear();
    b = cell();
    EXPECT_EQ(sent, (V{{"a"}})) << "b no longer goes to the worker";
    EXPECT_EQ(fresh_sent, (V{{"b"}})) << "b races alone in a fresh child";
    EXPECT_FALSE(b.fallback);
    EXPECT_EQ(b.worker_restarts, 0);
    ASSERT_EQ(b.arms.size(), 2u);
    EXPECT_EQ(b.arms[0].arm, "a");
    EXPECT_EQ(b.arms[0].status, "ok");
    EXPECT_EQ(b.arms[1].arm, "b");
    EXPECT_EQ(b.arms[1].status, "error");
}

TEST(TuneWorker, AllBenchedArmsAreAFallbackAndMarkedAlone) {
    ArmErrors errs;
    for (int i = 0; i < 2; ++i) errs.note("b", true);
    int tries = 0;
    auto attempt = [&](const std::vector<std::string>& arms) {
        ++tries;
        WorkerTry t{true, false, "", {}};
        for (const std::string& a : arms) t.arms.push_back(arm(a, "ok", {1.0}));
        return t;
    };
    auto fresh = [](const std::vector<std::string>& arms) {
        ArmBatch b;
        for (const std::string& a : arms) b.arms.push_back(arm(a, "ok", {2.0}));
        return b;
    };
    ArmBatch b = race_on_worker({"a", "b"}, errs, attempt, [] {}, fresh);
    EXPECT_FALSE(b.fallback);
    EXPECT_EQ(b.alone, (std::vector<std::string>{"b"}));
    for (int i = 0; i < 2; ++i) errs.note("a", true);
    b = race_on_worker({"a", "b"}, errs, attempt, [] {}, fresh);
    EXPECT_EQ(tries, 1) << "nothing left for the worker";
    EXPECT_TRUE(b.fallback);
    EXPECT_EQ(b.alone, (std::vector<std::string>{"a", "b"}));
    ASSERT_EQ(b.arms.size(), 2u);
}

TEST(TuneWorker, ABenchedArmErringInFiveFreshCellsIsDropped) {
    ArmErrors errs;
    for (int i = 0; i < 2; ++i) errs.note("b", true);
    int b_runs = 0;
    bool b_errors = true;
    auto attempt = [](const std::vector<std::string>& arms) {
        WorkerTry t{true, false, "", {}};
        for (const std::string& a : arms) t.arms.push_back(arm(a, "ok", {1.0}));
        return t;
    };
    auto fresh = [&](const std::vector<std::string>& arms) {
        ArmBatch b;
        for (const std::string& a : arms) {
            b_runs += a == "b";
            b.arms.push_back(a == "b" && b_errors ? arm("b", "error") : arm(a, "ok", {2.0}));
        }
        return b;
    };
    for (int i = 0; i < kDropAfterErrors - 1; ++i) race_on_worker({"a", "b"}, errs, attempt, [] {}, fresh);
    b_errors = false;
    race_on_worker({"a", "b"}, errs, attempt, [] {}, fresh);  // one clean cell resets the streak
    b_errors = true;
    for (int i = 0; i < kDropAfterErrors - 1; ++i) race_on_worker({"a", "b"}, errs, attempt, [] {}, fresh);
    EXPECT_FALSE(errs.dropped("b")) << "4 errors, a clean cell, 4 errors: not consecutive";
    race_on_worker({"a", "b"}, errs, attempt, [] {}, fresh);
    EXPECT_TRUE(errs.dropped("b"));
    EXPECT_FALSE(errs.dropped("a"));
    const int runs = b_runs;
    const ArmBatch b = race_on_worker({"a", "b"}, errs, attempt, [] {}, fresh);
    EXPECT_EQ(b_runs, runs) << "a dropped arm is not run";
    ASSERT_EQ(b.arms.size(), 2u);
    EXPECT_EQ(b.arms[0].status, "ok");
    EXPECT_EQ(b.arms[1].status, "error");
    EXPECT_EQ(b.arms[1].reason, "dropped after 5 consecutive errors");
    EXPECT_EQ(b.alone, (std::vector<std::string>{"b"})) << "kept out of the audit's worker side";
}

TEST(TuneTieredDriver, EveryOpDtypeGetsAtLeastOneAudit) {
    TempDir repo, ledger;
    TieredOpts o = fake_opts(repo, ledger, Tier::preview);
    o.audit_fraction = 0;
    AuditMeasurer m;
    m.disagree = false;
    ASSERT_EQ(run_tiered(o, {"sm_fake", "Fake"}, &m), 0);
    EXPECT_EQ(m.fresh_keys, (std::vector<std::string>{"n=1"})) << "the hash picked nothing: the first worker cell";
}

TEST(TuneTieredDriver, RefinementCapStopsAndReportsIt) {
    TempDir repo, ledger;
    TieredOpts o = fake_opts(repo, ledger, Tier::preview);
    o.refine_cap_factor = 0.5;  // 4 lattice cells: 2 refinement cells
    const fs::path events = ledger.path / "events.jsonl";
    const int fd = ::open(events.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    o.progress_fd = fd;
    FakeMeasurer m;
    testing::internal::CaptureStdout();
    ASSERT_EQ(run_tiered(o, {"sm_fake", "Fake"}, &m), 0);
    const std::string out = testing::internal::GetCapturedStdout();
    ::close(fd);
    EXPECT_EQ(m.keys, (std::vector<std::string>{"n=1", "n=4", "n=16", "n=64", "n=8", "n=11"}));
    const std::string ev = read_file(events);
    EXPECT_NE(ev.find("{\"ev\": \"refine_cap\", \"op\": \"fakeop\", \"dtype\": \"float\", \"lattice\": 4, \"refined\": 2, \"cap\": 2, \"dropped\": 1}"),
              std::string::npos) << ev;
    EXPECT_NE(out.find("refinement cap hit"), std::string::npos) << out;
}

TEST(TuneTieredDriver, PlanEstimateIncludesRefinement) {
    TempDir repo, ledger;
    TieredOpts o = fake_opts(repo, ledger, Tier::preview);
    o.plan = true;
    const fs::path events = ledger.path / "events.jsonl";
    const int fd = ::open(events.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    o.progress_fd = fd;
    testing::internal::CaptureStdout();
    ASSERT_EQ(run_tiered(o, {"sm_fake", "Fake"}, nullptr), 0);
    const std::string out = testing::internal::GetCapturedStdout();
    ::close(fd);
    const std::string ev = read_file(events);
    // No history: refine_cap_factor 3.0 x 0.5 x 4 lattice cells.
    EXPECT_NE(ev.find("\"refine_cells\": 6"), std::string::npos) << ev;
    EXPECT_NE(ev.find("\"est_refine_s\""), std::string::npos) << ev;
    EXPECT_NE(out.find("with refinement"), std::string::npos) << out;
    EXPECT_NE(out.find("lattice only"), std::string::npos) << out;
}

TEST(TuneTieredDriver, AHashPickedFallbackCellDoesNotUseUpTheForcedAudit) {
    // A run id whose 2% hash picks exactly one lattice cell and no refinement cell; that cell falls back.
    const std::vector<int> lattice{1, 4, 16, 64}, refined{8, 11, 9, 10};
    const double fraction = 0.3;
    auto picked = [&](const std::string& id, int n) { return audit_pick(id, {{"n", std::to_string(n)}}, fraction); };
    std::string id;
    int fallback_n = 0;
    for (int i = 0; i < 10000 && id.empty(); ++i) {
        const std::string c = "rid" + std::to_string(i);
        std::vector<int> hit;
        for (int n : lattice)
            if (picked(c, n)) hit.push_back(n);
        if (hit.size() == 1 && std::none_of(refined.begin(), refined.end(), [&](int n) { return picked(c, n); }))
            id = c, fallback_n = hit[0];
    }
    ASSERT_FALSE(id.empty());
    class FallbackMeasurer : public AuditMeasurer {
    public:
        int at = 0;
        ArmBatch measure(const CellJob& j) override {
            ArmBatch b = AuditMeasurer::measure(j);
            b.fallback = key_int(j.key, "n") == at;
            return b;
        }
    };
    TempDir repo, ledger;
    TieredOpts o = fake_opts(repo, ledger, Tier::preview);
    o.audit_fraction = fraction;
    o.run_id = id;
    FallbackMeasurer m;
    m.disagree = false;
    m.at = fallback_n;
    ASSERT_EQ(run_tiered(o, {"sm_fake", "Fake"}, &m), 0);
    EXPECT_EQ(m.fresh_keys.size(), 1u) << "fallback n=" << fallback_n;
    EXPECT_TRUE(std::find(m.fresh_keys.begin(), m.fresh_keys.end(), "n=" + std::to_string(fallback_n)) == m.fresh_keys.end());
}

TEST(TuneTieredDriver, BudgetStoppedRunResumesRefinementWithinTheCap) {
    TempDir repo, ledger;
    TieredOpts o = fake_opts(repo, ledger, Tier::preview);
    o.budget_h = 1e-12;  // run 1: the lattice only, then the budget stops refinement
    FakeMeasurer first;
    ASSERT_EQ(run_tiered(o, {"sm_fake", "Fake"}, &first), 0);
    EXPECT_EQ(first.keys, (std::vector<std::string>{"n=1", "n=4", "n=16", "n=64"}));
    o.budget_h = 0;
    o.refine_cap_factor = 0.5;  // floor(0.5 x 4 stored round-0 records) = 2 refinement cells over all runs
    const fs::path events = ledger.path / "events.jsonl";
    const int fd = ::open(events.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    o.progress_fd = fd;
    FakeMeasurer resumed;
    ASSERT_EQ(run_tiered(o, {"sm_fake", "Fake"}, &resumed), 0);
    ::close(fd);
    EXPECT_EQ(resumed.keys, (std::vector<std::string>{"n=8", "n=11"})) << "the stored lattice is the cap base";
    EXPECT_NE(read_file(events).find("\"lattice\": 4, \"refined\": 2, \"cap\": 2, \"dropped\": 1"), std::string::npos)
        << read_file(events);
    FakeMeasurer third;
    ASSERT_EQ(run_tiered(o, {"sm_fake", "Fake"}, &third), 0);
    EXPECT_TRUE(third.keys.empty()) << "the 2 stored refinement cells used the cap up";
    o.refine_cap_factor = -1;  // preview's 3.0: 12 in all, the bracket closes after 2 more
    FakeMeasurer fourth;
    ASSERT_EQ(run_tiered(o, {"sm_fake", "Fake"}, &fourth), 0);
    EXPECT_EQ(sorted(fourth.keys), sorted({"n=9", "n=10"}));
    const Ledger l = read_ledger(ledger_dir(ledger.str(), "fakeop", "float", "sm_fake"));
    std::size_t lattice = 0, refined = 0;
    for (const CellRecord& c : l.cells) ++(c.round == 0 ? lattice : refined);
    EXPECT_EQ(lattice, 4u);
    EXPECT_LE(refined, std::size_t(3 * lattice));
}

TEST(TuneTieredDriver, AResumedLatticeStillGetsTheMarginHedge) {
    class CloseMeasurer : public FakeMeasurer {
    public:
        ArmBatch measure(const CellJob& j) override {
            keys.push_back(key_arg(j.key));
            ArmBatch b;
            for (const std::string& a : j.arms) b.arms.push_back(arm(a, "ok", {a == "a" ? 1.0 : 1.05}));
            return b;
        }
    };
    TempDir repo, ledger;
    TieredOpts o = fake_opts(repo, ledger, Tier::preview);
    o.budget_h = 1e-12;
    CloseMeasurer first;
    ASSERT_EQ(run_tiered(o, {"sm_fake", "Fake"}, &first), 0);
    EXPECT_EQ(first.keys, (std::vector<std::string>{"n=1", "n=4", "n=16", "n=64"}));
    o.budget_h = 0;
    CloseMeasurer resumed;
    ASSERT_EQ(run_tiered(o, {"sm_fake", "Fake"}, &resumed), 0);
    EXPECT_EQ(sorted(resumed.keys), sorted({"n=2", "n=8", "n=32"})) << "stored round-0 cells are lattice ends";
}

TEST(TuneTieredDriver, MarginRefinesLatticeBracketsOnly) {
    // a wins everywhere, b 5% behind: every lattice bracket is a margin hedge, no midpoint re-triggers.
    class CloseMeasurer : public FakeMeasurer {
    public:
        ArmBatch measure(const CellJob& j) override {
            keys.push_back(key_arg(j.key));
            ArmBatch b;
            for (const std::string& a : j.arms) b.arms.push_back(arm(a, "ok", {a == "a" ? 1.0 : 1.05}));
            return b;
        }
    };
    TempDir repo, ledger;
    TieredOpts o = fake_opts(repo, ledger, Tier::preview);
    CloseMeasurer m;
    ASSERT_EQ(run_tiered(o, {"sm_fake", "Fake"}, &m), 0);
    EXPECT_EQ(sorted(m.keys), sorted({"n=1", "n=4", "n=16", "n=64", "n=2", "n=8", "n=32"}));
}

TEST(TuneTieredDriver, ArmsRacedAloneStayOutOfTheAuditsWarmSide) {
    // Arm a ran alone in a fresh child and errored; the audit's fresh child times it fine.
    class AloneMeasurer : public AuditMeasurer {
    public:
        ArmBatch measure(const CellJob& j) override {
            ArmBatch b = AuditMeasurer::measure(j);
            for (ArmOutcome& a : b.arms)
                if (a.arm == "a") a.status = "error", a.ms.clear();
            b.alone = {"a"};
            return b;
        }
    };
    TempDir repo, ledger;
    TieredOpts o = fake_opts(repo, ledger, Tier::preview);
    o.audit_fraction = 0;
    AloneMeasurer m;
    m.disagree = false;
    ASSERT_EQ(run_tiered(o, {"sm_fake", "Fake"}, &m), 0);
    EXPECT_EQ(m.fresh_keys, (std::vector<std::string>{"n=1"}));
    const Ledger l = read_ledger(ledger_dir(ledger.str(), "fakeop", "float", "sm_fake"));
    EXPECT_EQ(l.runs.back().worker_mode.at("fakeop.float"), "worker");
}

// ---- deep-run readiness: no dominance carry-forward, refused dtype pairs -------------------------

namespace {

// a loses to b by 20x at every cell: an old carry-forward would stop timing a past the first one.
class DomMeasurer : public FakeMeasurer {
public:
    std::vector<std::pair<std::string, std::vector<std::string>>> calls;
    ArmBatch measure(const CellJob& j) override {
        calls.emplace_back(key_arg(j.key), j.arms);
        ArmBatch out;
        for (const std::string& a : j.arms)
            out.arms.push_back(a == "b" ? arm("b", "ok", {1.0, 1.0}) : arm("a", "eliminated", {20.0}));
        return out;
    }
};

}  // namespace

// evidence: docs/design/tiered-tuning.md#engine-dominance-carry-forward-removed
TEST(TuneTieredDriver, ATwentyfoldLoserIsStillTimedAtEveryLargerCell) {
    TempDir repo, ledger;
    TieredOpts o = fake_opts(repo, ledger, Tier::preview);
    DomMeasurer m;
    ASSERT_EQ(run_tiered(o, {"sm_fake", "Fake"}, &m), 0);
    using V = std::vector<std::string>;
    ASSERT_GE(m.calls.size(), 3u);
    for (const auto& [key, arms] : m.calls) EXPECT_EQ(sorted(arms), (V{"a", "b"})) << key;
    const Ledger l = read_ledger(ledger_dir(ledger.str(), "fakeop", "float", "sm_fake"));
    for (const CellRecord& c : l.cells)
        for (const CandResult& r : c.cands) EXPECT_EQ(r.reason.rfind("dominated:", 0), std::string::npos) << key_arg(c.key);
}

namespace {

class FakeRealSpec : public FakeSpec {
public:
    std::string op() const override { return "fakereal"; }
    std::vector<std::string> candidates(const std::string& d) const override {
        if (d != "float") throw std::invalid_argument("fakereal is real-only: no '" + d + "' (float)");
        return {"a", "b"};
    }
};
const FakeRealSpec kFakeReal;

}  // namespace

TEST(TuneTieredDriver, AMultiOpRunSkipsTheDtypesASpecRefusesWithOneNote) {
    TempDir repo, ledger;
    TieredOpts o = fake_opts(repo, ledger, Tier::preview);
    if (!find_spec("fakereal")) register_spec(&kFakeReal);
    o.ops = {"fakeop", "fakereal"};
    o.dtypes = {"float", "cfloat"};
    FakeMeasurer m;
    ::testing::internal::CaptureStdout();
    int rc = -1;
    std::string thrown;
    try {
        rc = run_tiered(o, {"sm_fake", "Fake"}, &m);
    } catch (const std::exception& e) {
        thrown = e.what();
    }
    const std::string out = ::testing::internal::GetCapturedStdout();
    ASSERT_EQ(rc, 0) << thrown << "\n" << out;
    const std::string note = "== note: skipping fakereal cfloat: fakereal is real-only: no 'cfloat' (float)\n";
    EXPECT_NE(out.find(note), std::string::npos) << out;
    EXPECT_EQ(out.find("== note: skipping", out.find(note) + 1), std::string::npos) << "one note: " << out;
    for (const char* dt : {"float", "cfloat"})
        EXPECT_FALSE(read_ledger(ledger_dir(ledger.str(), "fakeop", dt, "sm_fake")).cells.empty()) << dt;
    EXPECT_FALSE(read_ledger(ledger_dir(ledger.str(), "fakereal", "float", "sm_fake")).cells.empty());
    EXPECT_FALSE(fs::exists(ledger_dir(ledger.str(), "fakereal", "cfloat", "sm_fake")));
    o.ops = {"fakereal"};
    EXPECT_THROW(run_tiered(o, {"sm_fake", "Fake"}, &m), std::invalid_argument) << "a single-op run still refuses";
}

// ---- the ld audit (spec.hh kLdAuditPad) ------------------------------------------------------

namespace {

// verify at a padded ld: "a" fails its residual, "b" passes, "c" is refused; race: everything ok but "d".
class LdAuditSpec : public FakeSpec {
public:
    mutable std::vector<CellRequest> calls;
    bool throw_on_verify = false;
    std::string op() const override { return "ldauditop"; }
    std::vector<ArmOutcome> run_cell(const CellRequest& req) const override {
        calls.push_back(req);
        if (req.mode == "verify" && throw_on_verify) throw std::runtime_error("device lost");
        std::vector<ArmOutcome> out;
        for (const std::string& a : req.arms) {
            ArmOutcome o = arm(a, "ok", {1.0});
            if (req.mode == "verify" && req.ld_pad > 0 && a == "a") o.status = "bad", o.reason = "residual", o.residual = 0.5;
            if (req.mode == "verify" && a == "c") o.status = "skipped", o.reason = "pin refused";
            if (req.mode != "verify" && a == "d") o.status = "bad", o.reason = "residual";
            out.push_back(o);
        }
        return out;
    }
};

const ArmOutcome& outcome(const std::vector<ArmOutcome>& v, const std::string& a) {
    return *std::find_if(v.begin(), v.end(), [&](const ArmOutcome& o) { return o.arm == a; });
}

}  // namespace

TEST(TuneLdAudit, ACellReverifiesItsVerifiedAuditArmsAtAPaddedLd) {
    LdAuditSpec s;
    CellRequest r;
    r.dtype = "float";
    r.key = nkey(8);
    r.arms = {"a", "b", "c", "d", "e"};
    r.mode = "race";
    r.ld_pad = 1;
    r.ld_audit = {"a", "b", "c", "d"};
    const auto out = run_cell_audited(s, r);
    ASSERT_EQ(s.calls.size(), 2u);
    EXPECT_EQ(s.calls[1].mode, "verify");
    EXPECT_EQ(s.calls[1].ld_pad, 1 + kLdAuditPad);
    EXPECT_EQ(s.calls[1].arms, (std::vector<std::string>{"a", "b", "c"})) << "d failed at the natural ld; e was not asked";
    EXPECT_EQ(outcome(out, "a").status, "bad");
    EXPECT_EQ(outcome(out, "a").ld_audit, "fail");
    EXPECT_EQ(outcome(out, "a").reason.rfind("ld audit", 0), 0u) << outcome(out, "a").reason;
    EXPECT_DOUBLE_EQ(outcome(out, "a").residual, 0.5);
    EXPECT_EQ(outcome(out, "a").ms, std::vector<double>{1.0}) << "the timing stays; the status makes it unrankable";
    EXPECT_EQ(outcome(out, "b").status, "ok");
    EXPECT_EQ(outcome(out, "b").ld_audit, "pass");
    EXPECT_EQ(outcome(out, "c").status, "ok");
    EXPECT_EQ(outcome(out, "c").ld_audit, "skipped");
    EXPECT_EQ(outcome(out, "d").ld_audit, "");
    EXPECT_EQ(outcome(out, "e").ld_audit, "");

    s.calls.clear();
    r.mode = "jit";
    (void)run_cell_audited(s, r);
    EXPECT_EQ(s.calls.size(), 1u) << "only a timed cell is audited";
    s.calls.clear();
    r.mode = "race";
    r.ld_audit.clear();
    (void)run_cell_audited(s, r);
    EXPECT_EQ(s.calls.size(), 1u) << "nothing asked, nothing audited";

    s.throw_on_verify = true;
    r.ld_audit = {"b"};
    const auto thrown = run_cell_audited(s, r);
    EXPECT_EQ(outcome(thrown, "b").status, "bad");
    EXPECT_EQ(outcome(thrown, "b").ld_audit, "fail");
    EXPECT_NE(outcome(thrown, "b").reason.find("device lost"), std::string::npos);
}

TEST(TuneLdAudit, TheBookAuditsEachArmOnceAndKeepsUnverifiedArmsWanted) {
    LdAuditBook book;
    EXPECT_EQ(book.want({"a", "b", "c"}), (std::vector<std::string>{"a", "b", "c"}));
    std::vector<ArmOutcome> cell{arm("a", "bad"), arm("b", "ok"), arm("c", "skipped")};
    cell[0].ld_audit = "fail", cell[0].reason = "ld audit: residual";
    cell[1].ld_audit = "pass";
    EXPECT_EQ(book.note(nkey(4), cell), std::vector<std::string>{"a"});
    EXPECT_EQ(book.want({"a", "b", "c"}), std::vector<std::string>{"c"}) << "c was never verified, so never audited";
    EXPECT_EQ(book.passed, 1u);
    ASSERT_EQ(book.failed.size(), 1u);
    EXPECT_EQ(book.failed[0], "a @ n=4 batch=1024: ld audit: residual");
    EXPECT_TRUE(book.note(nkey(8), cell).empty()) << "a second verdict for an audited arm is not counted";
    EXPECT_EQ(book.passed, 1u);
    std::vector<ArmOutcome> later{arm("c", "ok")};
    later[0].ld_audit = "skipped";
    (void)book.note(nkey(16), later);
    EXPECT_EQ(book.skipped, 1u);
    EXPECT_TRUE(book.want({"a", "b", "c"}).empty());
}

TEST(TuneLdAudit, RequestsAndOutcomesCarryTheAudit) {
    CellRequest r;
    r.dtype = "float";
    r.key = nkey(8);
    r.arms = {"a", "b"};
    r.mode = "race";
    r.warm_s = 0.1;
    r.ld_audit = {"b", "a"};
    const auto got = parse_request(request_line("fakeop", r));
    ASSERT_TRUE(got);
    EXPECT_EQ(got->second.ld_audit, r.ld_audit);
    r.ld_audit.clear();
    EXPECT_TRUE(parse_request(request_line("fakeop", r))->second.ld_audit.empty());

    std::vector<ArmOutcome> arms{arm("a", "bad", {1.0}), arm("b", "ok", {2.0})};
    arms[0].ld_audit = "fail";
    arms[1].ld_audit = "pass";
    std::vector<Record> recs;
    std::istringstream text(outcome_text(arms));
    for (std::string line; std::getline(text, line);) recs.push_back(*parse_record(line));
    const auto back = outcomes_from_records(recs);
    EXPECT_EQ(outcome(back, "a").ld_audit, "fail");
    EXPECT_EQ(outcome(back, "b").ld_audit, "pass");
}

namespace {

// Honours CellJob::ld_audit as a child would: "a" fails the audit, every other audited arm passes.
class LdAuditMeasurer : public CellMeasurer {
public:
    std::vector<std::vector<std::string>> asked;
    bool worker = false;  // persistent: the first worker cell gets the carve-out audit too
    bool persistent() const override { return worker; }
    ArmBatch measure(const CellJob& j) override {
        asked.push_back(j.ld_audit);
        ArmBatch b;
        for (const std::string& a : j.arms) {
            ArmOutcome o = arm(a, "ok", {a == "a" ? 0.5 : 1.0});
            if (std::find(j.ld_audit.begin(), j.ld_audit.end(), a) != j.ld_audit.end()) {
                o.ld_audit = a == "a" ? "fail" : "pass";
                if (a == "a") o.status = "bad", o.reason = "ld audit: ld_pad +3: bad residual";
            }
            b.arms.push_back(o);
        }
        return b;
    }
};

}  // namespace

TEST(TuneLdAudit, TheTieredDriverAuditsEachCandidatesFirstCellAndReportsTheFailure) {
    TempDir repo, ledger;
    TieredOpts o = fake_opts(repo, ledger, Tier::preview);
    const fs::path events = ledger.path / "events.jsonl";
    const int fd = ::open(events.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    o.progress_fd = fd;
    LdAuditMeasurer m;
    ::testing::internal::CaptureStdout();
    const int rc = run_tiered(o, {"sm_fake", "Fake"}, &m);
    const std::string out = ::testing::internal::GetCapturedStdout();
    ::close(fd);
    ASSERT_EQ(rc, 0) << out;
    ASSERT_FALSE(m.asked.empty());
    EXPECT_EQ(sorted(m.asked[0]), (std::vector<std::string>{"a", "b"}));
    for (std::size_t i = 1; i < m.asked.size(); ++i) EXPECT_TRUE(m.asked[i].empty()) << "cell " << i;
    const Ledger l = read_ledger(ledger_dir(ledger.str(), "fakeop", "float", "sm_fake"));
    std::size_t bad = 0;
    for (const CellRecord& c : l.cells)
        for (const CandResult& r : c.cands)
            if (r.status == "bad") {
                ++bad;
                EXPECT_EQ(r.cand, "a");
                EXPECT_EQ(r.reason.rfind("ld audit", 0), 0u) << r.reason;
                EXPECT_NE(c.ranked.front(), "a") << "a bad arm is never ranked";
            }
    EXPECT_EQ(bad, 1u) << "only the audited cell marks it";
    EXPECT_NE(out.find("ld audit FAILED: a ld audit"), std::string::npos) << out;
    EXPECT_NE(out.find("== summary fakeop float ld audit (ld +3): 1 passed, 0 skipped, 1 failed"), std::string::npos) << out;
    const std::string ev = read_file(events);
    EXPECT_NE(ev.find("\"ev\": \"ld_audit\""), std::string::npos) << ev;
    EXPECT_NE(ev.find("\"ev\": \"ld_audit_summary\""), std::string::npos) << ev;
}

TEST(TuneLdAudit, AnLdAuditFailureIsNotACarveOutMismatch) {
    TempDir repo, ledger;
    TieredOpts o = fake_opts(repo, ledger, Tier::preview);
    LdAuditMeasurer m;
    m.worker = true;
    ::testing::internal::CaptureStdout();
    const int rc = run_tiered(o, {"sm_fake", "Fake"}, &m);
    const std::string out = ::testing::internal::GetCapturedStdout();
    ASSERT_EQ(rc, 0) << out;
    EXPECT_NE(out.find(" audit: ok"), std::string::npos) << out;
    EXPECT_EQ(out.find("fresh children for the rest"), std::string::npos)
        << "the fresh child re-verifies at the natural ld, where the arm passes: no worker defect\n" << out;
}
