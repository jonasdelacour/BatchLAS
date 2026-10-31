// SPDX-License-Identifier: MIT
#pragma once
// The `reached` rows a callable leaves in the choice-coverage table (linked is not reached: agent
// guide section 9), captured in-process: snapshot, run, diff, restore (src/select/coverage.hh).
#include <batchlas/blas/enums.hh>
#include <batchlas/util/env.hh>

#include <gtest/gtest.h>

#include <src/select/coverage.hh>

#include <algorithm>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace test_support {

/// Set by the binary's first coverage_lines call (inline: one flag per binary).
inline bool& emission_checked() {
    static bool checked = false;
    return checked;
}

/// One `reached` row of src/select/coverage.cc, by column name.
struct ReachedRow {
    std::string op, scalar, backend, origin, spelling;
    long long m = 0, n = 0, k = 0, batch = 0, calls = 0;
    batchlas::Uplo uplo = batchlas::Uplo::Lower;
    batchlas::Side side = batchlas::Side::Left;
    batchlas::Diag diag = batchlas::Diag::NonUnit;
    batchlas::Transpose trans_a = batchlas::Transpose::NoTrans, trans_b = batchlas::Transpose::NoTrans;
};

/// The `reached`, then `miss`, CSV lines (src/select/coverage.cc's emit() format) `run` adds, the
/// calls column counting this run alone. Recording is on for the run; the table and the switch are restored
/// after (an exception in `run` passes through), so BATCHLAS_COVERAGE_OUT never sees the capture.
template <typename Run>
std::vector<std::string> captured_lines(Run&& run) {
    struct Restore {
        batchlas::coverage::Snapshot at = batchlas::coverage::snapshot();
        ~Restore() { batchlas::coverage::restore(at); }
    } guard;
    batchlas::coverage::g_dynamic_enabled = true;
    run();
    return batchlas::coverage::rows_since(guard.at);
}

/// One CSV line split on commas (a coverage line has no quoted fields), its newline dropped.
inline std::vector<std::string> csv_fields(const std::string& line) {
    std::vector<std::string> f;
    std::stringstream ss(line.empty() || line.back() != '\n' ? line : line.substr(0, line.size() - 1));
    for (std::string tok; std::getline(ss, tok, ',');) f.push_back(tok);
    return f;
}

/// captured_lines' `reached` rows, parsed by column name.
template <typename Run>
std::vector<ReachedRow> reached_rows(Run&& run) {
    std::vector<ReachedRow> rows;
    for (const std::string& line : captured_lines(std::forward<Run>(run))) {
        if (line.rfind("reached,", 0) != 0) continue;
        const auto f = csv_fields(line);
        if (f.size() < 20) {
            ADD_FAILURE() << "a short coverage row: " << line;
            continue;
        }
        ReachedRow r{f[1], f[2], f[3], f[9], f[10], std::stoll(f[5]), std::stoll(f[6]), std::stoll(f[7]), std::stoll(f[8]),
                     std::stoll(f[11])};
        r.uplo = static_cast<batchlas::Uplo>(std::stoi(f[15]));
        r.side = static_cast<batchlas::Side>(std::stoi(f[16]));
        r.diag = static_cast<batchlas::Diag>(std::stoi(f[17]));
        r.trans_a = static_cast<batchlas::Transpose>(std::stoi(f[18]));
        r.trans_b = static_cast<batchlas::Transpose>(std::stoi(f[19]));
        rows.push_back(r);
    }
    return rows;
}

/// The CoverageRow cases' capture: captured_lines(run) for every caller, and for the binary's first
/// caller `run` once more, first, in an EXPECT_EXIT child (threadsafe: CUDA starts fresh) whose
/// BATCHLAS_COVERAGE_OUT file, the atexit emission, must hold the same `reached` and `miss` lines.
template <typename Run>
std::vector<std::string> coverage_lines(Run&& run) {
    if (!emission_checked()) {
        emission_checked() = true;
        ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
        const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
        std::string name = std::string(info->test_suite_name()) + "." + info->name();
        for (char& c : name)
            if (c == '/') c = '_';
        const std::string dir = ::testing::TempDir() + "cov." + name;
        std::filesystem::remove_all(dir);
        std::filesystem::create_directories(dir);
        auto child = [&] {
            const batchlas::ScopedEnvVar cov("BATCHLAS_COVERAGE_OUT", (dir + "/cov").c_str());
            batchlas::coverage::g_dynamic_enabled = true;
            try {
                run();
            } catch (const std::exception& e) {
                std::cerr << "coverage_lines child threw: " << e.what() << "\n";
                std::exit(2);
            }
            std::exit(0);
        };
        EXPECT_EXIT(child(), ::testing::ExitedWithCode(0), "");
        std::vector<std::string> emitted;
        for (const auto& ent : std::filesystem::directory_iterator(dir)) {
            std::ifstream in(ent.path());
            for (std::string line; std::getline(in, line);)
                if (line.rfind("reached,", 0) == 0 || line.rfind("miss,", 0) == 0) emitted.push_back(line + "\n");
        }
        std::filesystem::remove_all(dir);
        auto lines = captured_lines(std::forward<Run>(run));
        std::sort(emitted.begin(), emitted.end());
        auto sorted = lines;
        std::sort(sorted.begin(), sorted.end());
        EXPECT_EQ(emitted, sorted) << "the atexit emission differs from the in-process capture";
        return lines;
    }
    return captured_lines(std::forward<Run>(run));
}

}  // namespace test_support
