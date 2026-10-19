#pragma once

/// @file
/// @brief gemm: direct, tiled, small, reg, wide, splitk, vendor. evidence: docs/perf/gemm.md @ingroup api_selection_ops
// Sycl-free on purpose: tests and the tuner include it to name the choices the library runs.

#include <src/select/select.hh>

#include <array>
#include <complex>
#include <cstdint>
#include <string_view>
#include <type_traits>
#include <variant>

namespace batchlas::ops::gemm {

// Fields are the knobs a selector chooses. Derived in the launcher, never fields: the transpose
// instantiation, the aligned vs predicated leg, the small bucket, TR/TC/stages (tables below).
struct Direct : select::NoFields<"direct"> {};  ///< gemm_direct: one work-item per element, any form; max_wg >= 64
struct Tiled : select::NoFields<"tiled"> {};    ///< gemm_tiled: 16x16 shared-memory tile, any form; max_wg >= 256
struct Small : select::NoFields<"small"> {};    ///< gemm_small: several matrices per work-group; see small_fits
/// gemm_reg, float only: m x n x k tile, k unroll u (128x128: u1 two stages, u4 the predicated copy_async pipeline);
/// must name a reg_configs entry instantiating the call's form, and max_wg must cover its threads().
struct Reg {
    int m = 0, n = 0, k = 0, u = 1;
    static constexpr std::string_view name = "reg";
    static constexpr std::array<std::string_view, 4> fields{"m", "n", "k", "u"};
    std::array<int, 4> values() const { return {m, n, k, u}; }
    static Reg from(std::array<int, 4> v) { return {v[0], v[1], v[2], v[3]}; }
    bool operator==(const Reg&) const = default;
};
/// gemm_wide, every scalar: 16-byte-granule m x n x k tiles; must name a wide_configs entry that
/// instantiates the call's form, and max_wg must cover its threads().
struct Wide {
    int m = 0, n = 0, k = 0;
    static constexpr std::string_view name = "wide";
    static constexpr std::array<std::string_view, 3> fields{"m", "n", "k"};
    std::array<int, 3> values() const { return {m, n, k}; }
    static Wide from(std::array<int, 3> v) { return {v[0], v[1], v[2]}; }
    bool operator==(const Wide&) const = default;
};
/// gemm_splitk, every scalar: wide host tile m x n x k (splitk_host) over s k-slices, then one reduction.
struct SplitK {
    int m = 0, n = 0, k = 0, s = 0;
    static constexpr std::string_view name = "splitk";
    static constexpr std::array<std::string_view, 4> fields{"m", "n", "k", "s"};
    std::array<int, 4> values() const { return {m, n, k, s}; }
    static SplitK from(std::array<int, 4> v) { return {v[0], v[1], v[2], v[3]}; }
    bool operator==(const SplitK&) const = default;
};
struct Vendor : select::NoFields<"vendor"> {};  ///< backend::gemm_vendor; needs the level-3 library

using GemmChoice = std::variant<Direct, Tiled, Small, Reg, Wide, SplitK, Vendor>;  ///< natives: GPU or vendor-free host, Default precision

/// Transpose forms an instantiation exists for, after the real-scalar C->T fold.
struct Forms {
    bool nn = false, nt = false, tn = false, tt = false;
};

/// One compiled register config: tr/tc thread tile, stages, forms, and whether NN (every form at
/// 128x128) takes the unpredicated leg when the layout allows (derived per call, never a gate).
struct RegCfg {
    int m, n, k, u, tr, tc, stages;
    Forms forms;
    bool aligned_leg;
    constexpr int threads() const { return (m / tr) * (n / tc); }
    /// Float local memory per work-group (gemm_kernels.cc static_asserts it against each kernel).
    constexpr std::int64_t slm_bytes() const {
        if (m == 128 && n == 128)
            return std::int64_t(4) * ((32 * 132 > stages * k * m ? 32 * 132 : stages * k * m) + stages * k * n);
        return std::int64_t(4) * stages * ((m + 1) * k + (k + 1) * n);
    }
};
inline constexpr std::array<RegCfg, 11> reg_configs{{
    {32, 32, 8, 1, 2, 2, 1, {true, false, false, false}, false},
    {64, 64, 8, 1, 4, 4, 1, {true, false, false, false}, false},
    {64, 64, 16, 1, 4, 4, 1, {true, true, true, true}, false},
    {128, 32, 16, 1, 4, 4, 2, {true, true, true, true}, false},
    {128, 32, 32, 1, 4, 4, 2, {true, true, true, true}, true},
    {128, 64, 16, 1, 4, 4, 1, {false, true, true, true}, false},
    {32, 128, 16, 1, 4, 4, 1, {true, false, true, true}, false},
    {128, 64, 32, 4, 8, 4, 2, {true, false, false, false}, true},
    {128, 64, 32, 2, 8, 4, 2, {true, false, false, false}, true},
    {128, 128, 8, 1, 8, 8, 2, {true, true, true, true}, true},  // its own kernel (register_128x128.hh)
    {128, 128, 8, 4, 8, 8, 4, {true, true, true, true}, true},  // the same; predicated leg: 4 zero-filled copy_async stages
}};

/// A wide form (op(A), op(B)) over {N, T, C} = {0, 1, 2}, as bit 3 op(A) + op(B) of WideCfg::forms.
constexpr unsigned wide_bit(int a, int b) { return 1u << (3 * a + b); }
inline constexpr unsigned kWideNN = wide_bit(0, 0), kWideNT = wide_bit(0, 1), kWideNC = wide_bit(0, 2);
inline constexpr unsigned kWideTN = wide_bit(1, 0), kWideTT = wide_bit(1, 1), kWideTC = wide_bit(1, 2);
inline constexpr unsigned kWideCN = wide_bit(2, 0), kWideCT = wide_bit(2, 1), kWideCC = wide_bit(2, 2);
inline constexpr unsigned kWideAll = 0x1FF;
inline constexpr unsigned kWideSmallTileForms = kWideAll & ~(kWideTT | kWideTC | kWideCC);

/// One compiled wide config and the forms a complex scalar instantiates. A real scalar compiles
/// only the N/C forms: conj is the identity, so its C build serves a real T (wide_serves).
struct WideCfg {
    int m, n, k, ttm, ttn;
    unsigned forms;
    constexpr int threads() const { return (m / ttm) * (n / ttn); }
    constexpr bool has(int a, int b) const { return (forms & wide_bit(a, b)) != 0; }
};
inline constexpr std::array<WideCfg, 8> wide_configs{{
    {64, 64, 16, 4, 4, kWideAll},
    {128, 32, 16, 4, 4, kWideNC | kWideNT | kWideTN | kWideCN},  // the potrf trailing update, W = 32
    {32, 128, 16, 4, 4, kWideCN | kWideTN | kWideNT | kWideNC},  // the geqrf panel update V^H A22, nb = 32
    {32, 32, 16, 4, 4, kWideSmallTileForms},
    {16, 16, 16, 2, 2, kWideSmallTileForms},
    {64, 32, 16, 4, 4, kWideSmallTileForms},
    {32, 64, 16, 4, 4, kWideSmallTileForms},
    {64, 64, 8, 4, 4, kWideAll},  // k <= 8: no zero-filled half slab (wide v2 at TileK = 8)
}};

/// split-k hosts: every wide row launch_wide_transposed runs in all its forms (not 64x64).
constexpr bool splitk_host(const WideCfg& c) { return !(c.m == 64 && c.n == 64); }
inline constexpr std::array<int, 3> kSplitKSlices{2, 4, 8};
/// The partials' column stride: m rounded to 4, so a slice's columns are whole 16-byte granules.
constexpr std::int64_t splitk_ld(std::int64_t m) { return (m + 3) / 4 * 4; }
/// One item's partials (all s slices), bytes; can_run holds them to Device::scratch_budget / batch.
template <class T>
constexpr std::int64_t splitk_item_bytes(std::int64_t m, std::int64_t n, int s) {
    return splitk_ld(m) * n * s * std::int64_t(sizeof(T));
}

template <class T>
inline constexpr bool is_complex_v = !std::is_same_v<T, float> && !std::is_same_v<T, double>;

/// Does config c serve (op(A), op(B)) for scalar T? A real T folds to C (the build it compiles).
template <class T>
constexpr bool wide_serves(const WideCfg& c, int a, int b) {
    if constexpr (is_complex_v<T>) return c.has(a, b);
    else return c.has(a == 0 ? 0 : 2, b == 0 ? 0 : 2);
}

/// Shared stages of a wide config's k x (m + n) tile: two for wide v2 (64x64, every scalar but
/// double) and for a transposed tile other than 64x64 whose two stages fit 40 KiB; else one.
template <class T>
constexpr int wide_stages(const WideCfg& c) {
    if (c.m == 64 && c.n == 64) return std::is_same_v<T, double> ? 1 : 2;
    return 2 * c.k * (c.m + c.n) * std::int64_t(sizeof(T)) <= 40960 ? 2 : 1;
}

/// Local memory a wide config asks for; gemm_kernels.cc static_asserts it against the kernels.
template <class T>
constexpr std::int64_t wide_slm_bytes(const WideCfg& c) {
    return std::int64_t(wide_stages<T>(c)) * c.k * (c.m + c.n) * std::int64_t(sizeof(T));
}

// sycl-free copies of the kernels' limits; gemm.cc static_asserts them against the kernels.
inline constexpr int kSmallMaxDim = 64;
inline constexpr int kSmallWg = 128;
inline constexpr int kSmallTiledMaxDim = 56;  // float NN above 32: one matrix per (NB/4)^2 lanes

/// The work-group `small` launches: the batched kernel's 128 lanes, or the float NN tiled leg's
/// (NB/4)^2 with NB = 48 or 56 (small_batched.hh).
template <class T>
constexpr int small_wg(bool nn, int max_dim) {
    if (!std::is_same_v<T, float> || !nn || max_dim <= 32 || max_dim > kSmallTiledMaxDim) return kSmallWg;
    const int nb = max_dim <= 48 ? 48 : 56;
    return (nb / 4) * (nb / 4);
}
inline constexpr int kDirectWg = 64;
inline constexpr int kTiledWg = 256;
/// direct, tiled, reg and wide put the batch in SYCL dim 0 = CUDA grid z (65535); small is 1-D.
inline constexpr std::int64_t kMaxGridBatch = 65535;

/// small's batched leg is [[sycl::reqd_sub_group_size(32)]]; only the float NN tiled leg is not.
template <class T>
constexpr bool small_needs_sg32(bool nn, int max_dim) {
    return !(std::is_same_v<T, float> && nn && max_dim > 32 && max_dim <= kSmallTiledMaxDim);
}
/// small's ceiling for T: 32 for a 16-byte scalar (small_batched.hh's kSmallMaxDimFor says why).
template <class T>
constexpr int small_max_dim() { return sizeof(T) >= 16 ? 32 : kSmallMaxDim; }
/// Local memory small asks for: 128 / (2 NB) slots of NB x (NB + 1), NB the bucket of max_dim, or
/// the float NN tiled leg's 2 NB (NB + 4) (gemm_kernels.cc static_asserts both).
template <class T>
constexpr std::int64_t small_slm_bytes(bool nn, int max_dim) {
    if (std::is_same_v<T, float> && nn && max_dim > 32 && max_dim <= kSmallTiledMaxDim) {
        const std::int64_t nb = max_dim <= 48 ? 48 : 56;
        return 2 * nb * (nb + 4) * std::int64_t(sizeof(T));
    }
    const std::int64_t nb = max_dim <= 8 ? 8 : max_dim <= 16 ? 16 : max_dim <= 32 ? 32 : 64;
    return (kSmallWg / (2 * nb)) * nb * (nb + 1) * std::int64_t(sizeof(T));
}
/// The shape-and-device half of small's can_run, here so tests can probe synthetic devices.
template <class T>
bool small_fits(const select::Device& d, bool nn, std::int64_t max_dim) {
    if (max_dim < 1 || max_dim > small_max_dim<T>()) return false;
    const int mx = static_cast<int>(max_dim);
    const int wg = small_wg<T>(nn, mx);
    return d.max_wg >= wg && (d.has_sg32 || !small_needs_sg32<T>(nn, mx)) &&
           d.slm_budget_for(wg) >= small_slm_bytes<T>(nn, mx);
}

/// Every compiled choice, once, in tie-break order (§6.3): simpler first, vendor last.
template <class T>
constexpr auto candidates() {
    constexpr std::array<Wide, 8> wides{Wide{64, 64, 16}, Wide{128, 32, 16}, Wide{32, 128, 16}, Wide{32, 32, 16},
                                        Wide{16, 16, 16}, Wide{64, 32, 16}, Wide{32, 64, 16}, Wide{64, 64, 8}};
    constexpr std::size_t hosts = [] {
        std::size_t h = 0;
        for (const WideCfg& c : wide_configs) h += splitk_host(c);
        return h;
    }();
    constexpr std::size_t regs = std::is_same_v<T, float> ? reg_configs.size() : 0;
    std::array<GemmChoice, 4 + regs + wides.size() + hosts * kSplitKSlices.size()> out{};
    std::size_t i = 0;
    out[i++] = Direct{};
    out[i++] = Tiled{};
    out[i++] = Small{};
    if constexpr (regs > 0)
        for (const RegCfg& r : reg_configs) out[i++] = Reg{r.m, r.n, r.k, r.u};
    for (const Wide& w : wides) out[i++] = w;
    for (const WideCfg& c : wide_configs)
        if (splitk_host(c))
            for (int s : kSplitKSlices) out[i++] = SplitK{c.m, c.n, c.k, s};
    out[i++] = Vendor{};
    return out;
}

/// Generality order (§5.5): direct serves every GPU shape, vendor everything else (CPU, precision).
inline constexpr std::array<std::string_view, 2> last_resort{"direct", "vendor"};
inline constexpr select::OpSpec spec{Op::gemm, select::Lib::level3, {last_resort}};  ///< op, vendor library, rules

/// Table keys: ta, tb (N|T|C; C folds to T for a real scalar) and layout (packed = A, B, C
/// contiguous with 16-byte bases) exact; m, n, k, batch log, each weight 1 (work ~ m n k batch).
inline constexpr std::array<std::string_view, 7> key_names{"ta:exact", "tb:exact", "layout:exact", "m:log",
                                                           "n:log",    "k:log",    "batch:log"};

/// The tuner's demand-driven grid; the transcriber (tuned/README.md) spells it again. Squares for
/// every form and both layouts; panels and skinny shapes for the issued forms only, packed panels
/// from m, n >= 128. evidence: docs/design/flat-kernel-selection-phase3-plan.md (§3)
inline constexpr std::array<int, 14> grid_square{8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024};
inline constexpr std::array<int, 7> grid_panel_mn{32, 64, 128, 256, 512, 1024, 2048};
inline constexpr std::array<int, 6> grid_panel_k{8, 16, 32, 64, 96, 128};
inline constexpr std::array<int, 6> grid_skinny_mn{64, 128, 256, 512, 1024, 2048};
inline constexpr std::array<int, 2> grid_skinny_k{256, 1024};
inline constexpr std::array<int, 3> grid_batch{128, 2048, 32768};
/// Large k (not transcribed for sm_89): Gram m = n squares, and 32 x skinny both ways; panel forms, strided.
inline constexpr std::array<int, 3> grid_gram_mn{16, 32, 64};
inline constexpr std::array<int, 3> grid_gram_k{1024, 2048, 4096};
inline constexpr std::array<int, 1> grid_large_skinny_k{2048};
inline constexpr int grid_packed_panel_min = 128;
inline constexpr std::array<std::string_view, 4> grid_real_forms{"NN", "NT", "TN", "TT"};
inline constexpr std::array<std::string_view, 4> grid_real_panel_forms{"NN", "NT", "TN", "TT"};
inline constexpr std::array<std::string_view, 9> grid_complex_forms{"NN", "NT", "TN", "NC", "CN",
                                                                     "CT", "TC", "CC", "TT"};
inline constexpr std::array<std::string_view, 5> grid_complex_panel_forms{"NN", "NT", "NC", "TN", "CN"};

}  // namespace batchlas::ops::gemm
