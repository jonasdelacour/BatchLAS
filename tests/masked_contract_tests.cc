// SPDX-License-Identifier: MIT
// tests/masked_contract.hh checked on a utility that honours the mask (MatrixView::copy), and
// red, with the named message, on each defect it exists to catch: a write to an inactive item, an
// active item that differs from the unmasked call, a workspace size that follows the mask.
#include <gtest/gtest-spi.h>
#include <gtest/gtest.h>

#include <batchlas/blas/matrix.hh>

#include <tests/masked_contract.hh>
#include <tests/test_support/storage.hh>

#include <complex>
#include <cstdint>
#include <functional>
#include <set>
#include <string>
#include <vector>

namespace {

using T = std::complex<float>;
using batchlas::Span;
namespace mc = masked_contract;

constexpr int kN = 5, kBatch = 16;

struct CopyCase {
    batchlas::Queue q{batchlas::Device::default_device()};
    batchlas::Matrix<T> src = test_support::padded<T>(kN, kN + 1, kBatch);
    batchlas::Matrix<T> dst = test_support::padded<T>(kN, kN + 1, kBatch);

    // How the op under test misbehaves when masked.
    enum class Defect { none, ignores_mask, active_off, sized_by_mask } defect = Defect::none;
    int size_calls = 0;
    std::vector<bool> calls;  // per call: masked
    std::size_t ws_bytes = 0;  // > 0: a workspace that must arrive poisoned, which the call then dirties
    int unpoisoned = 0;

    mc::Case make() {
        mc::Case c;
        c.batch = kBatch;
        c.wg_items = 4;
        c.chunk_items = 5;
        c.prepare = [this] {
            for (std::size_t e = 0; e < std::size_t(src.stride()) * kBatch; ++e)
                src.data_ptr()[e] = T(float(e % 97) - 40.f, float(e % 13) + 0.5f);
            std::fill_n(dst.data_ptr(), std::size_t(dst.stride()) * kBatch, test_support::poison<T>());
        };
        c.call = [this](Span<const std::uint8_t> mask, Span<std::byte> ws) {
            calls.push_back(!mask.empty());
            for (std::byte& b : ws) {
                if (b != std::byte{mc::detail::Workspace::kPoison}) ++unpoisoned;
                b = std::byte{0};
            }
            const auto m = defect == Defect::ignores_mask ? Span<const std::uint8_t>{} : mask;
            batchlas::MatrixView<T>::copy(q, dst.view().with_batch_mask(m), src.view().with_batch_mask(m));
            if (defect == Defect::active_off && !mask.empty())
                batchlas::scale(q, T(2), dst.view().with_batch_mask(mask));
        };
        c.outputs = [this] { return std::vector<mc::Output>{mc::Output::matrix("dst", dst.view())}; };
        if (defect == Defect::sized_by_mask)
            c.buffer_size = [this](Span<const std::uint8_t> mask) { return mask.empty() ? 0 : std::size_t(64 * ++size_calls); };
        else if (ws_bytes > 0)
            c.buffer_size = [this](Span<const std::uint8_t>) { return ws_bytes; };
        return c;
    }
};

TEST(MaskedContract, AnOpThatHonoursTheMaskPasses) {
    CopyCase op;
    mc::check(op.q, op.make());
}

TEST(MaskedContract, PatternsCoverHolesAndChunkEdges) {
    std::vector<std::string> names;
    for (const auto& p : mc::patterns(kBatch, 4, 5)) names.push_back(p.name);
    EXPECT_EQ(names, (std::vector<std::string>{"none", "all-active", "alternating", "single", "wg-slot0-off", "wg-last-only",
                                               "wg-interior-off", "chunk-edges-off", "part-inactive"}));
    const auto part = mc::patterns(kBatch, 4, 5).back();
    EXPECT_EQ(part.chunk_items, 5);
    for (int b = 0; b < kBatch; ++b) EXPECT_EQ(part.active(b), b < 5 || b >= 10) << b;
}

TEST(MaskedContract, ASaturatingCaseRunsTheFourApprovedPatterns) {
    CopyCase op;
    mc::Case c = op.make();
    c.identical_items = true;
    std::vector<std::string> names;
    for (const auto& p : mc::patterns_for(c)) names.push_back(p.name);
    EXPECT_EQ(names, (std::vector<std::string>{"none", "all-active", "alternating", "wg-slot0-off"}));
}

// One call per pattern; the unmasked `none` is the reference (no second run), and a chunk setting
// adds its own reference. Under masked_reference the unmasked call never runs.
TEST(MaskedContract, EachPatternRunsOnceBesideOneReferencePerChunk) {
    CopyCase op;
    mc::check(op.q, op.make());
    EXPECT_EQ(op.calls, (std::vector<bool>{false, true, true, true, true, true, true, false, true, true}));
    CopyCase mr;
    mc::Case c = mr.make();
    c.masked_reference = true;
    mc::check(mr.q, c);
    EXPECT_EQ(mr.calls, std::vector<bool>(9, true));
}

// Every run gets the check's one workspace freshly poisoned, whatever the run before left in it.
TEST(MaskedContract, TheWorkspaceArrivesPoisonedEveryRun) {
    CopyCase op;
    op.ws_bytes = 96;
    mc::check(op.q, op.make());
    EXPECT_EQ(op.calls.size(), 10u);
    EXPECT_EQ(op.unpoisoned, 0);
}

namespace cov = batchlas::coverage;

// gemm, F32, SYCL, Upper, "native": n 33..63 and batch 2..3 share one row key.
void record_gemm(int64_t n, int64_t batch, const char* spelling, bool native_existed, int native_supported) {
    cov::Shape s;
    s.op = batchlas::Op::gemm;
    s.m = s.n = s.k = n;
    s.batch = batch;
    s.uplo = batchlas::Uplo::Upper;
    cov::record_choice(batchlas::Op::gemm, batchlas::ScalarKind::F32, batchlas::Backend::SYCL, s, "native", spelling,
                       native_existed, native_supported);
}

// The process's tables set aside for one test: it starts empty, with the switch as given, and its
// synthetic rows never reach a BATCHLAS_COVERAGE_OUT file.
struct CoverageWindow {
    cov::Snapshot at = cov::snapshot();
    explicit CoverageWindow(bool recording) {
        at.keep = false;
        cov::g_dynamic_enabled = recording;
    }
    ~CoverageWindow() { cov::restore(at); }
};

// The in-process capture (src/select/coverage.hh): only the run's own calls, a nested capture's
// rows merged into the outer one (recording was on under it), and with recording off before, the
// tables and the switch restored after.
TEST(ReachedRows, CaptureSeesOnlyItsRunAndRestoresTheTable) {
    const CoverageWindow window(false);
    record_gemm(40, 3, "tiled", true, 1);  // before the capture: not in it
    std::vector<std::string> inner;
    const auto rows = test_support::reached_rows([&] {
        EXPECT_TRUE(cov::dynamic_enabled());
        record_gemm(40, 3, "tiled", true, 1);
        record_gemm(40, 3, "tiled", true, 1);
        inner = test_support::captured_lines([&] {
            record_gemm(300, 3, "direct", true, 1);
            cov::record_miss(batchlas::Op::trsm, batchlas::ScalarKind::C64, batchlas::Backend::SYCL, "none");
        });
    });
    ASSERT_EQ(inner.size(), 2u);
    EXPECT_EQ(test_support::csv_fields(inner[0]).at(10), "direct");
    EXPECT_EQ(inner[1], "miss,trsm,complex<double>,SYCL,,,,,,,,1,0,0,none\n");
    ASSERT_EQ(rows.size(), 2u) << "the inner capture's restore merges its row into the outer one";
    for (const auto& r : rows) {
        EXPECT_EQ(r.op, "gemm");
        EXPECT_EQ(r.backend, "SYCL");
        EXPECT_EQ(r.uplo, batchlas::Uplo::Upper);
        EXPECT_EQ(r.calls, r.spelling == "tiled" ? 2 : 1) << r.spelling;
    }
    EXPECT_FALSE(cov::dynamic_enabled());
    const auto table = cov::rows_since(window.at);
    ASSERT_EQ(table.size(), 1u) << "recording was off: the capture's rows are dropped";
    EXPECT_EQ(test_support::csv_fields(table[0]).at(11), "1");
    EXPECT_TRUE(test_support::captured_lines([] {}).empty());
}

// A capture while recording is already on (BATCHLAS_COVERAGE_OUT, route_diff.sh) reports its own
// call's shape and native flags, not an earlier call's in the same row key, and restore keeps both.
TEST(ReachedRows, ACaptureUnderRecordingReportsItsOwnCallAndKeepsTheCounts) {
    const CoverageWindow window(true);
    record_gemm(40, 3, "tiled", true, 1);
    record_gemm(40, 3, "tiled", true, 1);
    const auto rows = test_support::reached_rows([] { record_gemm(33, 2, "tiled", false, 0); });
    ASSERT_EQ(rows.size(), 1u);
    EXPECT_EQ(rows[0].m, 33);
    EXPECT_EQ(rows[0].n, 33);
    EXPECT_EQ(rows[0].k, 33);
    EXPECT_EQ(rows[0].batch, 2);
    EXPECT_EQ(rows[0].calls, 1);
    const auto line = test_support::captured_lines([] { record_gemm(33, 2, "tiled", false, 0); });
    ASSERT_EQ(line.size(), 1u);
    EXPECT_EQ(test_support::csv_fields(line[0]).at(12), "0") << "native_route_existed of the in-window call";
    EXPECT_EQ(test_support::csv_fields(line[0]).at(13), "0") << "native_route_supported of the in-window call";
    EXPECT_TRUE(cov::dynamic_enabled());
    const auto table = cov::rows_since(window.at);
    ASSERT_EQ(table.size(), 1u);
    const auto f = test_support::csv_fields(table[0]);
    EXPECT_EQ(f.at(5), "40") << "the earlier call made the row";
    EXPECT_EQ(f.at(8), "3");
    EXPECT_EQ(f.at(11), "4") << "two earlier calls and two captured ones";
    EXPECT_EQ(f.at(12), "1");
    EXPECT_EQ(f.at(13), "1");
}

// The patterns whose failures `run` raised, each failure required to carry `what`.
std::set<std::string> red_patterns(const std::function<void()>& run, const std::string& what) {
    ::testing::TestPartResultArray results;
    {
        const ::testing::ScopedFakeTestPartResultReporter fake(
            ::testing::ScopedFakeTestPartResultReporter::INTERCEPT_ONLY_CURRENT_THREAD, &results);
        run();
    }
    std::set<std::string> red;
    for (int i = 0; i < results.size(); ++i) {
        const std::string msg = results.GetTestPartResult(i).message();
        EXPECT_NE(msg.find(what), std::string::npos) << "a failure outside the defect's red set: " << msg;
        const auto at = msg.find(" pattern ");
        if (at != std::string::npos) red.insert(msg.substr(at + 9, msg.find_first_of("\n", at + 9) - at - 9));
    }
    return red;
}

const std::set<std::string> kWithHoles{"alternating",     "single",          "wg-slot0-off", "wg-last-only",
                                       "wg-interior-off", "chunk-edges-off", "part-inactive"};

TEST(MaskedContract, AWriteToAnInactiveItemIsRed) {
    CopyCase op;
    op.defect = CopyCase::Defect::ignores_mask;
    EXPECT_EQ(red_patterns([&] { mc::check(op.q, op.make()); }, "was written"), kWithHoles);
}

TEST(MaskedContract, AnActiveItemThatDiffersIsRed) {
    CopyCase op;
    op.defect = CopyCase::Defect::active_off;
    auto want = kWithHoles;
    want.insert("all-active");
    EXPECT_EQ(red_patterns([&] { mc::check(op.q, op.make()); }, "differs from the unmasked call"), want);
}

TEST(MaskedContract, AWorkspaceSizedByTheMaskIsRed) {
    CopyCase op;
    op.defect = CopyCase::Defect::sized_by_mask;
    EXPECT_EQ(red_patterns([&] { mc::check(op.q, op.make()); }, "buffer_size depends on the mask's contents"), kWithHoles);
}

TEST(MaskedContract, VerifyModeSeesEveryPatternsActiveItems) {
    CopyCase op;
    mc::Case c = op.make();
    std::vector<std::string> seen;
    c.verify_active = [&](const std::string& pattern, std::span<const int> active) {
        seen.push_back(pattern);
        if (pattern == "alternating") EXPECT_EQ(std::vector<int>(active.begin(), active.end()), (std::vector<int>{1, 3, 5, 7, 9, 11, 13, 15}));
        if (pattern == "single") EXPECT_EQ(std::vector<int>(active.begin(), active.end()), std::vector<int>{kBatch / 2});
    };
    mc::check(op.q, c);
    EXPECT_EQ(seen.size(), mc::patterns(kBatch, 4, 5).size());
}

}  // namespace
