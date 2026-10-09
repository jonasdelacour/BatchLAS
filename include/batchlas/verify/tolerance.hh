// SPDX-License-Identifier: MIT
// Pass/fail bounds: c(kind) * n * eps (docs/design/verification.md).
#pragma once

#include <batchlas/verify/scalar.hh>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <mutex>

namespace batchlas::verify {

enum class Check { factorization, solve, blas, orthogonality, eigen_residual, values };

namespace detail {

inline double coefficient(Check kind) {
    switch (kind) {
        case Check::factorization: return 16.0;
        case Check::solve: return 16.0;
        case Check::blas: return 4.0;
        case Check::orthogonality: return 16.0;
        case Check::eigen_residual: return 32.0;
        case Check::values: return 32.0;
    }
    return 16.0;
}

inline const char* kind_name(Check kind) {
    switch (kind) {
        case Check::factorization: return "factorization";
        case Check::solve: return "solve";
        case Check::blas: return "blas";
        case Check::orthogonality: return "orthogonality";
        case Check::eigen_residual: return "eigen_residual";
        case Check::values: return "values";
    }
    return "unknown";
}

template <class T> constexpr const char* dtype_name() {
    if constexpr (is_complex<T>::value) return std::is_same_v<real_t<T>, float> ? "cfloat" : "cdouble";
    else return std::is_same_v<T, float> ? "float" : "double";
}

}  // namespace detail

/// c(kind) * n * eps<T>(); @p n below 1 counts as 1.
template <class T> double bound(Check kind, int n) {
    return detail::coefficient(kind) * double(n < 1 ? 1 : n) * eps<T>();
}

/// True when @p value is not NaN and within bound<T>(kind, n). With BATCHLAS_VERIFY_RECORD=<path>
/// set, appends "kind dtype n value bound" to that file (calibration input).
template <class T> bool pass(Check kind, int n, double value) {
    const double b = bound<T>(kind, n);
    if (const char* path = std::getenv("BATCHLAS_VERIFY_RECORD"); path && *path) {
        static std::mutex mu;
        std::lock_guard<std::mutex> lock(mu);
        if (std::FILE* f = std::fopen(path, "a")) {
            std::fprintf(f, "%s %s %d %.17g %.17g\n", detail::kind_name(kind), detail::dtype_name<T>(), n, value, b);
            std::fclose(f);
        }
    }
    return !std::isnan(value) && value <= b;
}

}  // namespace batchlas::verify
