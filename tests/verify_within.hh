// SPDX-License-Identifier: MIT
// Asserting form of batchlas::verify::pass, for the QR and Cholesky tests.
#pragma once

#include <batchlas/verify/tolerance.hh>

#include <gtest/gtest.h>

#include <limits>

namespace test_utils {

/// pass<T>(kind, n, value), and also value <= keep: a bound the test had before it was migrated.
template <class T>
::testing::AssertionResult within(batchlas::verify::Check kind, int n, double value,
                                  double keep = std::numeric_limits<double>::infinity()) {
    if (!batchlas::verify::pass<T>(kind, n, value))
        return ::testing::AssertionFailure() << "value " << value << " exceeds the batchlas::verify bound "
                                             << batchlas::verify::bound<T>(kind, n) << " (n=" << n << ")";
    if (!(value <= keep))
        return ::testing::AssertionFailure() << "value " << value << " exceeds the test's retained bound " << keep;
    return ::testing::AssertionSuccess();
}

}  // namespace test_utils
