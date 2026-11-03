// SPDX-License-Identifier: MIT
// Pass/fail bounds: c(kind) * f(n) * eps, c calibrated (docs/design/verification.md#verification-calibration).
#pragma once

#include <batchlas/blas/enums.hh>
#include <batchlas/error.hh>

#include <cctype>
#include <cmath>
#include <limits>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <string>

namespace batchlas::verify {

/// Unit roundoff \f$u\f$ of T's real type: \f$2^{-24}\f$ for float, \f$2^{-53}\f$ for double.
/// @return \f$u\f$, half of `std::numeric_limits<float_t<T>>::epsilon()`
template <typename T> constexpr double eps() { return std::numeric_limits<float_t<T>>::epsilon() * 0.5; }

/// \f$\max(a, b)\f$, NaN if either is NaN (std::max drops a NaN second argument, so a poisoned value would pass).
/// @param[in] a  first value
/// @param[in] b  second value
/// @return \f$\max(a, b)\f$, or NaN if @p a or @p b is NaN
inline double nanmax(double a, double b) {
    if (std::isnan(a) || std::isnan(b)) return std::numeric_limits<double>::quiet_NaN();
    return a > b ? a : b;
}

/// orthogonality: Q from Householder reflectors or a direct orthogonalization (geqrf, orgqr, form_q, ortho);
/// orthogonality_rotations: eigen- and singular vectors from an iterative solver (Jacobi, QR iteration,
/// divide and conquer, inverse iteration). \f$n\f$ per kind, as each was calibrated: orthogonality, the rows
/// of Q (Householder error grows with the reflector length); orthogonality_rotations, the number of
/// vectors checked (columns of Q). blas: \f$n\f$ is the inner dimension \f$k\f$
/// (the tuner's componentwise trsm check: \f$k\f$ = the order); trsm_residual is normwise and judged under solve.
enum class Check { factorization, solve, blas, orthogonality, orthogonality_rotations, eigen_residual, values };

namespace detail {

/// @brief The calibrated coefficient \f$c(\mathrm{kind})\f$ of a kind's bound.
/// @param[in] kind  which check the coefficient is for
/// @return \f$c(\mathrm{kind})\f$; 16 for a value that names no Check
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

/// @brief Name of a Check value, as written to the `BATCHLAS_VERIFY_RECORD` file.
/// @param[in] kind  the value to name
/// @return the enumerator's name, or `"unknown"` for a value that names none
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

template <typename T> constexpr const char* dtype_name() {
    if constexpr (is_std_complex_v<T>) return std::is_same_v<float_t<T>, float> ? "cfloat" : "cdouble";
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

/// @brief Rejects a Slack with a non-finite or non-positive factor, or an empty reason.
/// @param[in] s  the slack to validate
/// @throws batchlas::invalid_argument for an empty reason, or a factor that is not both finite and \f$> 0\f$
inline void check_slack(const Slack& s) {
    if (!(s.factor > 0) || !std::isfinite(s.factor)) throw batchlas::invalid_argument("batchlas::verify::Slack: factor must be finite and > 0");
    if (!s.reason || !*s.reason) throw batchlas::invalid_argument("batchlas::verify::Slack: a reason is mandatory");
}

// "kind dtype n value bound factor reason"; bound is the kind's, never the slacked one, and value is raw.
// BATCHLAS_VERIFY_RECORD is a raw getenv on every check, deliberately not the settings snapshot: this
// header is outside the library's Settings snapshot (uninstalled batchlas::verify) and tests setenv it mid-process. The handle stays open while the path is
// unchanged; a new path reopens, unset closes, and a path fopen refuses is retried (nothing recorded).
/// @brief Appends "kind dtype n value bound factor reason" to the `BATCHLAS_VERIFY_RECORD` file, if set.
/// @param[in] kind   name of the check kind, from kind_name()
/// @param[in] dtype   name of the scalar type, from dtype_name()
/// @param[in] n       size the bound scales with
/// @param[in] value   measured error, unslacked
/// @param[in] b       the kind's bound `bound<T>(kind, n)`, without slack
/// @param[in] factor  slack factor, 1 when there is none
/// @param[in] reason  slack reason, `"-"` when there is none; whitespace is written as `_`
inline void record(const char* kind, const char* dtype, int n, double value, double b, double factor, const char* reason) {
    static std::mutex mu;
    static std::string open_path;
    static std::FILE* f = nullptr;
    std::lock_guard<std::mutex> lock(mu);
    const char* path = std::getenv("BATCHLAS_VERIFY_RECORD");
    if (!path || !*path) {
        if (f) std::fclose(f);
        f = nullptr;
        open_path.clear();
        return;
    }
    if (!f || open_path != path) {
        if (f) std::fclose(f);
        f = std::fopen(path, "a");
        open_path = path;
    }
    if (!f) return;
    std::string r(reason);
    for (char& c : r)
        if (std::isspace(static_cast<unsigned char>(c))) c = '_';
    std::fprintf(f, "%s %s %d %.17g %.17g %.17g %s\n", kind, dtype, n, value, b, factor, r.c_str());
    std::fflush(f);  // a test that crashes keeps every line before it
}

}  // namespace detail

/// \f$c(\mathrm{kind}) \cdot f(n) \cdot u\f$ with \f$u\f$ = `eps<T>()`: \f$f(n) = \max(n, 1)\f$, except blas,
/// \f$f(k) = \max(k, 0) + 2\f$.
/// @param[in] kind  which check the bound is for; selects the calibrated coefficient \f$c\f$
/// @param[in] n     size the bound scales with, as defined per kind in Check
/// @return the pass/fail bound
template <typename T> double bound(Check kind, int n) {
    const double f = kind == Check::blas ? double(n < 0 ? 0 : n) + 2.0 : double(n < 1 ? 1 : n);
    return detail::coefficient(kind) * f * eps<T>();
}

/// `slack.factor` \f$\cdot\f$ `bound<T>(kind, n)`.
/// @param[in] kind   which check the bound is for
/// @param[in] n      size the bound scales with, as defined per kind in Check
/// @param[in] slack  per-site factor on the bound, with its mandatory reason
/// @return the slacked bound
/// @throws batchlas::invalid_argument for an empty reason, or a factor that is not both finite and \f$> 0\f$.
template <typename T> double bound(Check kind, int n, Slack slack) {
    detail::check_slack(slack);
    return slack.factor * bound<T>(kind, n);
}

/// pass without recording: for predicates evaluated on results expected to be wrong.
/// @param[in] kind   which check @p value is judged under
/// @param[in] n      size the bound scales with, as defined per kind in Check
/// @param[in] value  measured error
/// @return true when @p value is not NaN and \f$\le\f$ `bound<T>(kind, n)`
template <typename T> bool within(Check kind, int n, double value) { return !std::isnan(value) && value <= bound<T>(kind, n); }

/// within() against the slacked bound, without recording.
/// @param[in] kind   which check @p value is judged under
/// @param[in] n      size the bound scales with, as defined per kind in Check
/// @param[in] value  measured error
/// @param[in] slack  per-site factor on the bound, with its mandatory reason
/// @return true when @p value is not NaN and \f$\le\f$ `bound<T>(kind, n, slack)`
/// @throws batchlas::invalid_argument when @p value is not NaN and the reason is empty or the factor is not both finite and \f$> 0\f$ (a NaN @p value returns false without checking @p slack).
template <typename T> bool within(Check kind, int n, double value, Slack slack) {
    return !std::isnan(value) && value <= bound<T>(kind, n, slack);
}

/// True when @p value is not NaN and within `bound<T>(kind, n)`. With `BATCHLAS_VERIFY_RECORD=<path>`
/// set, appends "kind dtype n value bound 1 -" to that file (calibration input).
/// @param[in] kind   which check @p value is judged under
/// @param[in] n      size the bound scales with, as defined per kind in Check
/// @param[in] value  measured error
/// @return true when the check passes
template <typename T> bool pass(Check kind, int n, double value) {
    detail::record(detail::kind_name(kind), detail::dtype_name<T>(), n, value, bound<T>(kind, n), 1.0, "-");
    return within<T>(kind, n, value);
}

/// pass against the slacked bound; records the raw value, the kind's bound, the factor and the reason.
/// @param[in] kind   which check @p value is judged under
/// @param[in] n      size the bound scales with, as defined per kind in Check
/// @param[in] value  measured error
/// @param[in] slack  per-site factor on the bound, with its mandatory reason
/// @return true when @p value is not NaN and within `bound<T>(kind, n, slack)`
/// @throws batchlas::invalid_argument for an empty reason, or a factor that is not both finite and \f$> 0\f$.
template <typename T> bool pass(Check kind, int n, double value, Slack slack) {
    detail::check_slack(slack);
    detail::record(detail::kind_name(kind), detail::dtype_name<T>(), n, value, bound<T>(kind, n), slack.factor, slack.reason);
    return within<T>(kind, n, value, slack);
}

/// Bound of pivot_ratio: \f$1 + 64\,u\f$ with \f$u\f$ = `eps<T>()` (32 machine epsilons), the slack of a cabs1 comparison in T.
/// @return the bound \f$1 + 64\,u\f$
template <typename T> double pivot_ratio_bound() { return 1.0 + 64.0 * eps<T>(); }

}  // namespace batchlas::verify
