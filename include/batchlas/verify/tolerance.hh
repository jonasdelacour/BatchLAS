// SPDX-License-Identifier: MIT
// Pass/fail bounds: c(kind) * f(n) * eps, c calibrated (docs/design/verification.md#verification-calibration).
#pragma once

#include <batchlas/verify/scalar.hh>

#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <string>

namespace batchlas::verify {

/// orthogonality: Q from Householder reflectors (geqrf, orgqr, sytrd, gebrd); orthogonality_rotations:
/// eigen- and singular vectors accumulated by an iterative solver (Jacobi, QR iteration, divide and conquer).
enum class Check { factorization, solve, blas, orthogonality, orthogonality_rotations, eigen_residual, values };

namespace detail {

inline double coefficient(Check kind) {
    switch (kind) {
        case Check::factorization: return 16.0;
        case Check::solve: return 16.0;
        case Check::blas: return 8.0;
        case Check::orthogonality: return 32.0;
        case Check::orthogonality_rotations: return 256.0;
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
        case Check::orthogonality_rotations: return "orthogonality_rotations";
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

/// A per-site factor on a kind's bound, in either direction. @p reason is mandatory: every factor
/// is listed by the calibration for a decision (@ref design_verification).
struct Slack {
    double factor;
    const char* reason;
};

namespace detail {

inline void check_slack(const Slack& s) {
    if (!(s.factor > 0) || !std::isfinite(s.factor)) throw std::invalid_argument("batchlas::verify::Slack: factor must be finite and > 0");
    if (!s.reason || !*s.reason) throw std::invalid_argument("batchlas::verify::Slack: a reason is mandatory");
}

// "kind dtype n value bound factor reason"; bound is the kind's, never the slacked one, and value is raw.
inline void record(const char* kind, const char* dtype, int n, double value, double b, double factor, const char* reason) {
    const char* path = std::getenv("BATCHLAS_VERIFY_RECORD");
    if (!path || !*path) return;
    std::string r(reason);
    for (char& c : r)
        if (std::isspace(static_cast<unsigned char>(c))) c = '_';
    static std::mutex mu;
    std::lock_guard<std::mutex> lock(mu);
    if (std::FILE* f = std::fopen(path, "a")) {
        std::fprintf(f, "%s %s %d %.17g %.17g %.17g %s\n", kind, dtype, n, value, b, factor, r.c_str());
        std::fclose(f);
    }
}

}  // namespace detail

/// c(kind) * f(n) * eps<T>(): f(n) = n (below 1 counts as 1), except blas, f(k) = k + 2 (k below 0 counts as 0).
template <class T> double bound(Check kind, int n) {
    const double f = kind == Check::blas ? double(n < 0 ? 0 : n) + 2.0 : double(n < 1 ? 1 : n);
    return detail::coefficient(kind) * f * eps<T>();
}

/// slack.factor * bound<T>(kind, n). @throws std::invalid_argument for an empty reason or a factor <= 0.
template <class T> double bound(Check kind, int n, Slack slack) {
    detail::check_slack(slack);
    return slack.factor * bound<T>(kind, n);
}

/// pass without recording: for predicates evaluated on results expected to be wrong.
template <class T> bool within(Check kind, int n, double value) { return !std::isnan(value) && value <= bound<T>(kind, n); }

template <class T> bool within(Check kind, int n, double value, Slack slack) {
    return !std::isnan(value) && value <= bound<T>(kind, n, slack);
}

/// True when @p value is not NaN and within bound<T>(kind, n). With `BATCHLAS_VERIFY_RECORD=<path>`
/// set, appends "kind dtype n value bound 1 -" to that file (calibration input).
template <class T> bool pass(Check kind, int n, double value) {
    detail::record(detail::kind_name(kind), detail::dtype_name<T>(), n, value, bound<T>(kind, n), 1.0, "-");
    return within<T>(kind, n, value);
}

/// pass against the slacked bound; records the raw value, the kind's bound, the factor and the reason.
template <class T> bool pass(Check kind, int n, double value, Slack slack) {
    detail::check_slack(slack);
    detail::record(detail::kind_name(kind), detail::dtype_name<T>(), n, value, bound<T>(kind, n), slack.factor, slack.reason);
    return within<T>(kind, n, value, slack);
}

/// Bound of pivot_ratio: 1 + 64 eps<T>() (32 machine epsilons), the slack of a cabs1 comparison in T.
template <class T> double pivot_ratio_bound() { return 1.0 + 64.0 * eps<T>(); }

}  // namespace batchlas::verify
