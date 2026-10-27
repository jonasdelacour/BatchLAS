// SPDX-License-Identifier: MIT
#pragma once
// The `reached` rows a callable leaves in BATCHLAS_COVERAGE_OUT (linked is not reached: agent
// guide section 9). The callable runs in a death-test child so the rows are that process's alone;
// pin routes inside it (or before the call: a threadsafe child re-runs the test body up to here).
#include <batchlas/blas/enums.hh>
#include <batchlas/util/env.hh>

#include <gtest/gtest.h>

#include <src/select/coverage.hh>

#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace test_support {

/// One `reached` row of src/select/coverage.cc, by column name.
struct ReachedRow {
    std::string op, scalar, backend, origin, spelling;
    long long m = 0, n = 0, k = 0, batch = 0, calls = 0;
    batchlas::Uplo uplo = batchlas::Uplo::Lower;
    batchlas::Side side = batchlas::Side::Left;
    batchlas::Diag diag = batchlas::Diag::NonUnit;
    batchlas::Transpose trans_a = batchlas::Transpose::NoTrans, trans_b = batchlas::Transpose::NoTrans;
};

/// Runs `run` in an EXPECT_EXIT child with coverage recording on and returns every `reached`
/// row it wrote (an exception in `run` fails the child). threadsafe: CUDA starts fresh there.
template <typename Run>
std::vector<ReachedRow> reached_rows(Run&& run) {
    ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
    const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
    std::string name = std::string(info->test_suite_name()) + "." + info->name();
    for (char& c : name)
        if (c == '/') c = '_';
    const std::string dir = ::testing::TempDir() + "cov." + name;
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    const std::string out = dir + "/cov";
    auto child = [&] {
        const batchlas::ScopedEnvVar cov("BATCHLAS_COVERAGE_OUT", out.c_str());
        batchlas::coverage::g_dynamic_enabled = true;
        try {
            run();
        } catch (const std::exception& e) {
            std::cerr << "reached_rows child threw: " << e.what() << "\n";
            std::exit(2);
        }
        std::exit(0);
    };
    EXPECT_EXIT(child(), ::testing::ExitedWithCode(0), "");
    std::vector<ReachedRow> rows;
    for (const auto& ent : std::filesystem::directory_iterator(dir)) {
        std::ifstream in(ent.path());
        for (std::string line; std::getline(in, line);) {
            if (line.rfind("reached,", 0) != 0) continue;
            std::vector<std::string> f;
            std::stringstream ss(line);
            for (std::string tok; std::getline(ss, tok, ',');) f.push_back(tok);
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
    }
    std::filesystem::remove_all(dir);
    return rows;
}

}  // namespace test_support
