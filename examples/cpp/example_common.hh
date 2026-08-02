#pragma once

// Shared reporting helpers for the C++ examples.
//
// The examples check every result they compute, so a clean run doubles as a
// smoke test. Output mirrors the Python example notebooks:
//
//     [ok  ] gemm error vs reference: 0.000e+00  (tol 1.0e-12)
//     [FAIL] ...
//
// This header is not part of the BatchLAS API.

#include <complex>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

namespace examples {

inline int& failure_count() {
    static int failures = 0;
    return failures;
}

inline void header(const std::string& title) {
    std::cout << "\n" << title << "\n" << std::string(title.size(), '=') << "\n";
}

inline void section(const std::string& title) {
    std::cout << "\n--- " << title << " ---\n";
}

namespace detail {

inline void print(const std::string& status, const std::string& name, const std::string& value) {
    std::cout << "[" << status << "] " << name << ": " << value << "\n";
}

template <typename T>
std::string to_string(const T& value) {
    std::ostringstream os;
    os << value;
    return os.str();
}

inline std::string to_string(bool value) { return value ? "true" : "false"; }

template <typename T>
std::string to_string(const std::complex<T>& value) {
    std::ostringstream os;
    os << value.real() << (value.imag() < T(0) ? "-" : "+") << std::abs(value.imag()) << "i";
    return os.str();
}

inline std::string format_error(double value) {
    std::ostringstream os;
    os << std::scientific << std::setprecision(3) << value;
    return os.str();
}

}  // namespace detail

// Report a value with no pass/fail semantics.
template <typename T>
void report(const std::string& name, const T& value) {
    detail::print("ok  ", name, detail::to_string(value));
}

// Report a boolean check.
inline void report_check(const std::string& name, bool ok) {
    if (!ok) ++failure_count();
    detail::print(ok ? "ok  " : "FAIL", name, ok ? "true" : "false");
}

// Report an error magnitude against a tolerance.
inline void report_error(const std::string& name, double error, double tol) {
    const bool ok = std::isfinite(error) && error <= tol;
    if (!ok) ++failure_count();
    std::ostringstream os;
    os << detail::format_error(error) << "  (tol " << detail::format_error(tol) << ")";
    detail::print(ok ? "ok  " : "FAIL", name, os.str());
}

// Exit code for main(): 0 when every check held.
inline int exit_code() {
    const int failures = failure_count();
    std::cout << "\n" << (failures == 0 ? "All checks passed." : std::to_string(failures) + " check(s) FAILED.")
              << "\n";
    return failures == 0 ? 0 : 1;
}

}  // namespace examples
