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

// Declared, not included: <batchlas/settings.hh> pulls in the route vocabulary
// and this header is reached by nearly every device translation unit in the
// tree. A declaration is all ScopedEnvVar's constructor and destructor need, and
// it keeps the include graph of the parsers below exactly as it was.
//
// The definition is in src/util/settings.cc. See <batchlas/settings.hh> for what
// a reload does and, more importantly, for the two cases it does not cover.
namespace detail {
void reload_settings();
}

// NOTE ON THE ARGUMENT: env_truthy/env_falsy take the VALUE of a variable, i.e.
// they are always called as env_truthy(std::getenv("BATCHLAS_X")). The name-taking
// overload `env_truthy(const char* name, bool fallback)` that this header once
// also declared has been removed: it gave one function name two incompatible readings
// of the same `const char*` parameter, so env_truthy("BATCHLAS_DEBUG") -- a
// perfectly natural thing to write once such an overload exists -- silently
// selected the value form and evaluated the *name* as a value, i.e. was always
// false. It had zero call sites, so deleting it was free.
//
// The alternative was to keep both and rename the value forms (env_value_truthy);
// rejected because that churns every existing call site to defend against an
// overload nothing uses. The value-only contract, spelled out here, is enough.

// Accepts exactly {1, true, TRUE, on, ON}. A null pointer (unset variable) is false.
inline bool env_truthy(const char* v) {
    if (!v) return false;
    const std::string s(v);
    return (s == "1" || s == "true" || s == "TRUE" || s == "on" || s == "ON");
}

// Accepts exactly {0, false, FALSE, off, OFF}. A null pointer (unset variable) is
// false -- note this is NOT !env_truthy: an unset variable is neither truthy nor
// falsy, which is what lets callers distinguish "forced off" from "not specified".
//
// The exact-spelling list is the contract, not an oversight: it is what the six
// parsers this replaced accepted, so widening it here would silently change how
// every existing call site reads the same strings. A knob that wants "Off"/"No"
// to work has to case-fold its own value; sytrd_sb2st_hh.cc does exactly that,
// and says why at its call site. Do not fold that helper back into this one.
inline bool env_falsy(const char* v) {
    if (!v) return false;
    const std::string s(v);
    return (s == "0" || s == "false" || s == "FALSE" || s == "off" || s == "OFF");
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

// As env_int_or, but a value that parses to <= 0 is also treated as "unset".
//
// This is the shape every kernel-geometry knob in src/extensions wants: a forced
// work-group count, tile width or sub-group count is meaningless at zero or
// negative, and the call sites all want to fall back to the computed default
// there rather than propagate a nonsense launch geometry. Those sites had each
// grown their own atoi-plus-`> 0` parser; this is that pattern with a name, so
// the clamp stays visible instead of being re-derived (or, worse, dropped) when
// a call site is routed through the plain env_int_or.
inline int env_positive_int_or(const char* name, int fallback) {
    const int v = env_int_or(name, fallback);
    return v > 0 ? v : fallback;
}

// Returns fallback when the variable is unset or set to the empty string.
inline std::string env_string_or(const char* name, const std::string& fallback) {
    const char* v = std::getenv(name);
    if (!v || !*v) return fallback;
    return std::string(v);
}

// ---------------------------------------------------------------------------
// The WRITE side of the same knob contract.
//
// Sixteen near-identical copies of this RAII class had accumulated across
// tests/ and benchmarks/ -- one per op whose route is pinned by an env knob --
// plus a single-key variant in syevx_tests.cc and two hand-rolled
// save/setenv/restore blocks in syev_blocked_tests.cc. They agreed on
// everything except one point: three of them treated a null value as "unset
// for the duration" and the other thirteen passed it straight to setenv, which
// is undefined behaviour. The null-aware reading is a strict superset of the
// other -- nothing that compiled against the thirteen changes meaning -- so it
// is the one kept here.
//
// setenv/unsetenv rather than putenv: putenv keeps the caller's buffer alive
// inside the environment, which is a lifetime trap in a scope that restores on
// exit.
//
// PRECONDITION: `name` must outlive the object -- in practice a string literal,
// which is what every call site passes. It is borrowed, not copied, because two
// of the gemm family benchmarks construct these inside the timed lambda, and a
// std::string member would put a heap allocation per construction into a
// measured region for no benefit any current caller can use. A caller that has
// to compose its key must keep the buffer alive itself.
//
// Not thread-safe, because the process environment is not: construct these
// from a test body or a benchmark setup, never from inside a parallel region.
//
// THE RELOAD. batchlas::settings() reads the environment once, under
// std::call_once, so a setenv on its own is invisible to the library. Both ends
// of this scope therefore call detail::reload_settings(): the constructor so the
// pinned value takes effect, and the destructor so it does not leak into every
// test that runs after this one in the same process. That pair is what lets the
// fifteen test files using this class keep working unchanged, and it is why a
// test that calls ::setenv directly instead of constructing one of these will
// now silently measure the pre-existing value.
class ScopedEnvVar {
public:
    // A null `value` UNSETS the variable for the duration, which is how a
    // caller asks for the automatic route regardless of what the surrounding
    // environment already said.
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

    ~ScopedEnvVar() {
        if (had_old_value_) {
            ::setenv(name_, old_value_.c_str(), 1);
        } else {
            ::unsetenv(name_);
        }
        // Nested ScopedEnvVars restore innermost-first, so each destructor's
        // reload leaves the settings agreeing with the environment the enclosing
        // scope still has pinned.
        detail::reload_settings();
    }

    // Copying would restore the same variable twice, the second time from a
    // snapshot that the first restore has already invalidated.
    ScopedEnvVar(const ScopedEnvVar&) = delete;
    ScopedEnvVar& operator=(const ScopedEnvVar&) = delete;

private:
    const char* name_;
    std::string old_value_;
    bool had_old_value_ = false;
};

}  // namespace batchlas
