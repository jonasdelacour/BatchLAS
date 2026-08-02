#pragma once
#include <cstdlib>
#include <string>

// Single source of truth for reading BatchLAS environment-variable knobs.
//
// Six byte-identical truthiness parsers had accumulated across src/ (queue.hh,
// kernel-trace.hh, syev_cta.cc, sytrd_blocked.cc, gesvd_blocked.cc,
// band_reduction.cc), plus four copy-pasted stoi/try-catch blocks in
// band_reduction.cc. The accepted spellings below are exactly what all of them
// accepted, so consolidating is semantics-preserving.

namespace batchlas {

// Accepts exactly {1, true, TRUE, on, ON}. A null pointer (unset variable) is false.
inline bool env_truthy(const char* v) {
    if (!v) return false;
    const std::string s(v);
    return (s == "1" || s == "true" || s == "TRUE" || s == "on" || s == "ON");
}

// Accepts exactly {0, false, FALSE, off, OFF}. A null pointer (unset variable) is
// false -- note this is NOT !env_truthy: an unset variable is neither truthy nor
// falsy, which is what lets callers distinguish "forced off" from "not specified".
inline bool env_falsy(const char* v) {
    if (!v) return false;
    const std::string s(v);
    return (s == "0" || s == "false" || s == "FALSE" || s == "off" || s == "OFF");
}

inline bool env_truthy(const char* name, bool fallback) {
    const char* v = std::getenv(name);
    if (!v) return fallback;
    return env_truthy(v);
}

// Returns fallback when the variable is unset or does not parse as an integer.
inline int env_int_or(const char* name, int fallback) {
    const char* v = std::getenv(name);
    if (!v) return fallback;
    try {
        return std::stoi(std::string(v));
    } catch (...) {
        return fallback;
    }
}

// Returns fallback when the variable is unset or set to the empty string.
inline std::string env_string_or(const char* name, const std::string& fallback) {
    const char* v = std::getenv(name);
    if (!v || !*v) return fallback;
    return std::string(v);
}

}  // namespace batchlas
