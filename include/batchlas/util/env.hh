#pragma once
#include <batchlas/export.hh>
#include <cstdlib>
#include <string>

/// @file
/// @brief The shared environment-variable parsers and the ScopedEnvVar RAII setter.
///
/// These parsers define the exact spellings that `BATCHLAS_*` knobs accept; batchlas::settings()
/// calls them for every typed field. The accepted sets are the contract: widening one would
/// change how every existing call site reads the same strings.
/// @see @ref design_environment
/// @ingroup config

namespace batchlas {

// Declared, not included: settings.hh pulls in the route vocabulary, and this header is reached
// by nearly every device TU.
namespace detail {
BATCHLAS_API void reload_settings();
}

/// @brief True when `v` is exactly one of `1`, `true`, `TRUE`, `on`, `ON`.
/// @param v the variable's value (`std::getenv(...)`), not its name; `nullptr` means unset
/// @return false for an unset variable and for every other spelling
/// @ingroup config
// Takes the VALUE. A name-taking overload once existed and made env_truthy("BATCHLAS_X") silently
// always false. evidence: docs/design/environment.md#environment-the-shared-parsers-in-envhh
inline bool env_truthy(const char* v) {
    if (!v) return false;
    const std::string s(v);
    return (s == "1" || s == "true" || s == "TRUE" || s == "on" || s == "ON");
}

/// @brief True when `v` is exactly one of `0`, `false`, `FALSE`, `off`, `OFF`.
///
/// Not `!env_truthy(v)`: an unset variable is neither truthy nor falsy, which lets a caller
/// distinguish "forced off" from "not specified".
/// @param v the variable's value, or `nullptr` when unset
/// @return false for an unset variable and for every other spelling
/// @ingroup config
// A knob that wants "Off"/"No" must case-fold its own value (sytrd_sb2st_hh.cc does); do not
// fold that helper back into this one.
inline bool env_falsy(const char* v) {
    if (!v) return false;
    const std::string s(v);
    return (s == "0" || s == "false" || s == "FALSE" || s == "off" || s == "OFF");
}

/// @brief The integer value of environment variable `name`.
/// @param name     the variable's name
/// @param fallback returned when the variable is unset or `std::stoi` rejects it
/// @return the parsed value; note that `std::stoi` accepts a numeric prefix (`"16x"` is 16)
/// @ingroup config
inline int env_int_or(const char* name, int fallback) {
    const char* v = std::getenv(name);
    if (!v) return fallback;
    try {
        return std::stoi(std::string(v));
    } catch (...) {
        return fallback;
    }
}

/// @brief As env_int_or(), but a value <= 0 also yields `fallback`.
///
/// The shape of every kernel-geometry knob: a forced work-group count, tile width or sub-group
/// count is meaningless at zero or below.
/// @param name     the variable's name
/// @param fallback returned when unset, unparseable or non-positive
/// @ingroup config
inline int env_positive_int_or(const char* name, int fallback) {
    const int v = env_int_or(name, fallback);
    return v > 0 ? v : fallback;
}

/// @brief The value of environment variable `name`, or `fallback` when it is unset or empty.
/// @ingroup config
inline std::string env_string_or(const char* name, const std::string& fallback) {
    const char* v = std::getenv(name);
    if (!v || !*v) return fallback;
    return std::string(v);
}

/// @brief Sets (or unsets) an environment variable for a scope and restores it on exit.
///
/// Both the constructor and the destructor call detail::reload_settings(), so the library sees
/// the pinned value immediately and stops seeing it when the scope ends. A raw `::setenv` is
/// invisible to batchlas::settings(); use this instead.
/// @code
/// {
///     ScopedEnvVar pin("BATCHLAS_GEMM_ROUTE", "native");
///     gemm(ctx, a, b, c, GemmOptions<float>{});   // runs the native kernel
/// }
/// @endcode
/// @pre `name` outlives the object (in practice a string literal); it is borrowed, not copied.
/// @warning Not thread-safe, because the process environment is not: construct from a test body
///          or benchmark setup, never inside a parallel region. Do not let one straddle a
///          `*_buffer_size()` query and its solve.
/// @see @ref design_environment
/// @ingroup config
// Name borrowed so no heap allocation lands inside timed benchmark lambdas.
// evidence: docs/design/environment.md#environment-scopedenvvar
class ScopedEnvVar {
public:
    /// @brief Pin `name` to `value` for the scope.
    /// @param name  the variable's name
    /// @param value the value to set, or `nullptr` to unset the variable for the duration (which
    ///              asks for the automatic route regardless of the surrounding environment)
    ScopedEnvVar(const char* name, const char* value) : name_(name) {
        if (const char* old = std::getenv(name_)) {
            old_value_ = old;
            had_old_value_ = true;
        }
        if (value) {
            ::setenv(name_, value, 1);
        } else {
            ::unsetenv(name_);
        }
        detail::reload_settings();
    }

    /// @brief Restore the previous value (or unset state) and reload the settings.
    ~ScopedEnvVar() {
        if (had_old_value_) {
            ::setenv(name_, old_value_.c_str(), 1);
        } else {
            ::unsetenv(name_);
        }
        // Nested instances restore innermost-first, so this reload matches the enclosing scope.
        detail::reload_settings();
    }

    // Copying would restore the same variable twice, the second time from a stale snapshot.
    ScopedEnvVar(const ScopedEnvVar&) = delete;
    ScopedEnvVar& operator=(const ScopedEnvVar&) = delete;

private:
    const char* name_;
    std::string old_value_;
    bool had_old_value_ = false;
};

}  // namespace batchlas
