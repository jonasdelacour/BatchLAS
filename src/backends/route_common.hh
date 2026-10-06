#pragma once

// Backend-neutral helpers for the level-3 tile dispatchers (symm, syrk, syr2k, trmm):
// the BATCHLAS_<OP>_ROUTE word parse and queue facts. Nothing here names a CUDA type;
// the CUDA-only helpers are in cublasdx_dispatch_common.hh.

#include "../math-helpers.hh"

#include <sycl/sycl.hpp>

#include <batchlas/settings.hh>

#include <algorithm>
#include <cctype>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <string_view>

namespace batchlas::backend::detail {

inline int ceil_div(int value, int divisor) {
    return internal::ceil_div(value, divisor);
}

// The level-3 ops choose by rule, not by table, so they parse BATCHLAS_<OP>_ROUTE
// themselves. Lowercased and trimmed like src/select's pins; unset or empty is Auto,
// and a word the op does not take throws std::invalid_argument (spec R6).
enum class Level3Pin { Auto, Native, Vendor, Cublasdx, Expand, Triangular, Gram };

inline std::string_view level3_pin_word(Level3Pin p) {
    switch (p) {
        case Level3Pin::Auto:       return "auto";
        case Level3Pin::Native:     return "native";
        case Level3Pin::Vendor:     return "vendor";
        case Level3Pin::Cublasdx:   return "cublasdx";
        case Level3Pin::Expand:     return "expand";
        case Level3Pin::Triangular: return "triangular";
        case Level3Pin::Gram:       return "gram";
    }
    return "?";
}

inline std::string level3_upper(std::string_view op) {
    std::string u;
    for (const char c : op) u += static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    return u;
}

inline Level3Pin level3_pin(std::string_view op, std::initializer_list<Level3Pin> accepted) {
    const char* raw = batchlas::settings().routing.route(op).get();
    std::string text = raw ? raw : "";
    text.erase(0, text.find_first_not_of(" \t\n\r"));
    text.erase(text.find_last_not_of(" \t\n\r") + 1);
    std::transform(text.begin(), text.end(), text.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (text.empty() || text == "auto") return Level3Pin::Auto;
    std::string words = "auto";
    for (const Level3Pin p : accepted) {
        if (text == level3_pin_word(p)) return p;
        words += "|" + std::string(level3_pin_word(p));
    }
    throw std::invalid_argument(std::string(op) + ": BATCHLAS_" + level3_upper(op) + "_ROUTE=\"" + text +
                                "\" is not a valid choice: expected " + words);
}

inline bool is_gpu_queue(const Queue& ctx) {
    return ctx.device().type == DeviceType::GPU;
}

// A cublasdx pin the fused kernel cannot serve. MathDx is optional, and without it
// every cublasdx pin lands here.
[[noreturn]] inline void throw_forced_cublasdx_unavailable(std::string_view op,
                                                           const std::string& reason) {
    throw batchlas::unsupported("BATCHLAS_" + level3_upper(op) + "_ROUTE=cublasdx requested, but fused cuBLASDx " +
                                level3_upper(op) + " is unavailable: " + reason);
}

} // namespace batchlas::backend::detail
