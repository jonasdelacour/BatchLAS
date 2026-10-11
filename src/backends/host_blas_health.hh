// SPDX-License-Identifier: MIT
// The host dgemm health probe and its diagnostic, shared by the NETLIB backend (netlib_lapack.cc) and
// batchlas::verify's double LAPACKE references (src/verify/engine.cc), which call LAPACKE directly.
#pragma once

#include <batchlas/backend_config.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <string>
#include <vector>

namespace batchlas::detail {

#ifndef BATCHLAS_REQUIRED_OPENBLAS_CORETYPE
#define BATCHLAS_REQUIRED_OPENBLAS_CORETYPE ""
#endif

struct HostBlasDoubleHealth {
    bool ok = true;
    int bad_size = 0;
    double worst_error = 0.0;
};

// Port of cmake/BatchLASBlasHealthCheck.cmake on a strided sample of C. The sizes are the ones
// that expose the known defect: n=64 and n=256 happen to be correct there.
template <class Dgemm> HostBlasDoubleHealth probe_dgemm(Dgemm dgemm) {
    HostBlasDoubleHealth health;
    static const int sizes[] = {128, 200, 512};
    for (int n : sizes) {
        const size_t nn = static_cast<size_t>(n) * static_cast<size_t>(n);
        std::vector<double> a(nn), b(nn), c(nn, 0.0);
        for (int col = 0; col < n; ++col) {
            for (int row = 0; row < n; ++row) {
                a[row + static_cast<size_t>(col) * n] = std::sin(0.5 * (row + 1) * (col + 1));
                b[row + static_cast<size_t>(col) * n] = std::cos(0.25 * (row + 1) * (col + 2));
            }
        }

        dgemm(n, a.data(), b.data(), c.data());

        const int step = std::max(1, n / 32);
        double worst = 0.0;
        for (int col = 0; col < n; col += step) {
            for (int row = 0; row < n; row += step) {
                double want = 0.0;
                for (int i = 0; i < n; ++i) {
                    want += a[row + static_cast<size_t>(i) * n] * b[i + static_cast<size_t>(col) * n];
                }
                const double diff = std::abs(want - c[row + static_cast<size_t>(col) * n]);
                if (diff > worst) worst = diff;
            }
        }
        // Rounding differences are ~1e-13 here; a broken kernel is off by O(1)+.
        if (worst > 1e-6) {
            health.ok = false;
            health.bad_size = n;
            health.worst_error = worst;
            return health;
        }
    }
    return health;
}

inline std::string host_blas_double_health_message(const HostBlasDoubleHealth& health) {
    const std::string required = BATCHLAS_REQUIRED_OPENBLAS_CORETYPE;
    // OpenBLAS's own variable: deliberately NOT captured into Settings.
    const char* current_env = std::getenv("OPENBLAS_CORETYPE");
    const std::string current = current_env ? current_env : "";

    std::string msg =
        "BatchLAS: the host BLAS computes dgemm INCORRECTLY on this machine.\n"
        "  probe: max abs error " + std::to_string(health.worst_error) +
        " at n=" + std::to_string(health.bad_size) + " (rounding alone is ~1e-13)\n"
        "  OPENBLAS_CORETYPE is currently " +
        (current.empty() ? std::string("unset") : ("\"" + current + "\"")) + "\n";

    if (!required.empty()) {
        msg += "  this build detected that OPENBLAS_CORETYPE=" + required + " repairs it\n";
    }
    msg += "Double-precision results from the host/NETLIB backend cannot be trusted.\n"
           "OpenBLAS reads OPENBLAS_CORETYPE in its library constructor, before main(),\n"
           "so it must be set in the environment before the process starts - BatchLAS\n"
           "cannot set it for you.\n";

    if (!required.empty() && required != current) {
        msg += "    export OPENBLAS_CORETYPE=" + required + "\n";
    } else if (!required.empty()) {
        msg += "OPENBLAS_CORETYPE=" + required + " is already set and dgemm is still wrong: "
               "this host BLAS is unusable for double precision, upgrade or replace it.\n";
    } else {
        msg += "No working value was recorded at build time. Try, in order:\n"
               "    export OPENBLAS_CORETYPE=SKYLAKEX   (then HASWELL, SANDYBRIDGE, NEHALEM, PRESCOTT)\n";
    }
    msg += "Set BATCHLAS_BLAS_HEALTH=error to turn this into an exception, "
           "or BATCHLAS_BLAS_HEALTH=off to skip the check.\n";
    return msg;
}

}  // namespace batchlas::detail
