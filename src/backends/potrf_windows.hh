#pragma once

// The sm_89-measured hand windows as DATA for WindowChooser: first match wins, so they need
// not be exclusive and no window calls another. Bound only where no CostBook passed its gate.
// evidence: docs/perf/potrf.md#the-measured-lpanel-window

#include "potrf_routes.hh"

namespace batchlas::potrf_routes {

using ds::Question;
using ds::Window;

// The measured types and the measured CTA/LPanel boundary inside them.
template <class T>
inline constexpr bool lpanel_measured = std::is_same_v<T, float> || std::is_same_v<T, std::complex<float>>;
template <class T>
inline constexpr int cta_last = std::is_same_v<T, float> ? 35 : 32;

template <class T>
bool tiny_window(const PotrfShape& s) {
    if (s.n < 1 || s.n > pp::TinyCap<T>::kMaxN || s.n > 32) return false;
    if (s.uplo == Uplo::Upper || lpanel_measured<T>) return true;
    return (s.n >= 2 && s.n <= 8) || (s.n >= 12 && s.n <= 16);   // Lower fp64: fill splits it
}
template <class T>
bool cta_holds(const PotrfShape& s) {
    return s.n <= pp::cta_max_n<T>(resident::device_slm_budget(s.dev.local_mem_bytes));
}
template <class T>
bool lpanel_holds(const PotrfShape& s) {
    return lpanel_measured<T> && s.uplo == Uplo::Lower && s.n > cta_last<T> &&
           s.n <= pp::lpanel_max_n<T>(resident::device_slm_budget(s.dev.local_mem_bytes),
                                      s.dev.max_wg_size);
}
template <class T>
bool lower_measured_in(const PotrfShape& s, int lo, int hi) {
    return lpanel_measured<T> && s.uplo == Uplo::Lower && s.n > lo && s.n <= hi;
}

template <class T>
inline constexpr Window<PotrfShape> kPotrfWindows[] = {
    {"native:tiny", Question::VsVendor, &tiny_window<T>, "#the-tiny-potrf-window"},
    {"native:cta", Question::VsVendor,
     [](const PotrfShape& s) { return !tiny_window<T>(s) && lower_measured_in<T>(s, 32, cta_last<T>); },
     "#the-measured-lpanel-window"},
    {"native:lpanel", Question::VsVendor,
     [](const PotrfShape& s) { return lower_measured_in<T>(s, cta_last<T>, 256) && lpanel_holds<T>(s); },
     "#the-measured-lpanel-window"},
    {"native:tiny", Question::AmongNative, &tiny_window<T>, "#native_tier_preferred"},
    {"native:cta", Question::AmongNative,
     [](const PotrfShape& s) { return !tiny_window<T>(s) && cta_holds<T>(s) && !lpanel_holds<T>(s); },
     "#native_tier_preferred"},
    {"native:lpanel", Question::AmongNative, &lpanel_holds<T>, "#native_tier_preferred"},
    {"native:blocked", Question::AmongNative,
     [](const PotrfShape& s) { return !cta_holds<T>(s) && !lpanel_holds<T>(s); }, "#native_tier_preferred"},
};

template <class T>
std::span<const Window<PotrfShape>> potrf_windows() { return kPotrfWindows<T>; }

// A window naming a route the table does not list is a compile error, not a dead row.
template <class Tbl, class T>
consteval bool windows_name_rows() {
    for (const auto& w : kPotrfWindows<T>) {
        if (Tbl::index_of(w.key) < 0) return false;
    }
    return true;
}

}  // namespace batchlas::potrf_routes
