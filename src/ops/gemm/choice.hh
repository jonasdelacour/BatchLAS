#pragma once

// gemm's selection vocabulary (flat-kernel-selection-phase3-plan.md §1.3), header-only and
// sycl-free, so tests, the tuner and the transcribed tables name the choices the library runs.

#include "../../select/select.hh"

#include <array>
#include <complex>
#include <string_view>
#include <type_traits>
#include <variant>

namespace batchlas::ops::gemm {

// Fields are the knobs a selector chooses. Derived in the launcher, never fields: the transpose
// instantiation, the aligned vs predicated leg, the small bucket, TR/TC/stages (tables below).
struct Direct : select::NoFields<"direct"> {};  // one work-item per element, any form
struct Tiled : select::NoFields<"tiled"> {};    // 16x16 shared-memory tile, any form
struct Small : select::NoFields<"small"> {};    // several matrices per work-group, real, max dim <= 64
struct Reg {                                    // register-tiled, float
    int m = 0, n = 0, k = 0, u = 1;             // macro tile and k unroll
    static constexpr std::string_view name = "reg";
    static constexpr std::array<std::string_view, 4> fields{"m", "n", "k", "u"};
    std::array<int, 4> values() const { return {m, n, k, u}; }
    static Reg from(std::array<int, 4> v) { return {v[0], v[1], v[2], v[3]}; }
    bool operator==(const Reg&) const = default;
};
struct Wide {  // 16-byte-granule tiles for every scalar
    int m = 0, n = 0, k = 0;
    static constexpr std::string_view name = "wide";
    static constexpr std::array<std::string_view, 3> fields{"m", "n", "k"};
    std::array<int, 3> values() const { return {m, n, k}; }
    static Wide from(std::array<int, 3> v) { return {v[0], v[1], v[2]}; }
    bool operator==(const Wide&) const = default;
};
struct Vendor : select::NoFields<"vendor"> {};

using GemmChoice = std::variant<Direct, Tiled, Small, Reg, Wide, Vendor>;

// Transpose forms an instantiation exists for, after the real-scalar C->T fold.
struct Forms {
    bool nn = false, nt = false, tn = false, tt = false;
};

// One compiled register config: tr/tc thread tile, stages, the forms instantiated, and whether
// NN takes the unpredicated leg when the layout allows (derived per call, never a gate).
struct RegCfg {
    int m, n, k, u, tr, tc, stages;
    Forms forms;
    bool aligned_leg;
    constexpr int threads() const { return (m / tr) * (n / tc); }
};
inline constexpr std::array<RegCfg, 10> reg_configs{{
    {32, 32, 8, 1, 2, 2, 1, {true, false, false, false}, false},
    {64, 64, 8, 1, 4, 4, 1, {true, false, false, false}, false},
    {64, 64, 16, 1, 4, 4, 1, {true, true, true, true}, false},
    {128, 32, 16, 1, 4, 4, 2, {true, true, true, true}, false},
    {128, 32, 32, 1, 4, 4, 2, {true, true, true, true}, true},
    {128, 64, 16, 1, 4, 4, 1, {false, true, true, true}, false},
    {32, 128, 16, 1, 4, 4, 1, {true, false, true, true}, false},
    {128, 64, 32, 4, 8, 4, 2, {true, false, false, false}, true},
    {128, 64, 32, 2, 8, 4, 2, {true, false, false, false}, true},
    {128, 128, 8, 1, 8, 8, 2, {true, false, false, false}, true},  // its own kernel (register_128x128.hh)
}};

// One compiled wide config. NN, CN (ConjTrans A) and NC (ConjTrans B) instantiations; a real
// Trans is served by a ConjTrans one (conj is the identity), a complex Trans is not.
struct WideCfg {
    int m, n, k, ttm, ttn;
    bool nn, cn, nc;
    constexpr int threads() const { return (m / ttm) * (n / ttn); }
};
inline constexpr std::array<WideCfg, 5> wide_configs{{
    {64, 64, 16, 4, 4, true, true, true},
    {128, 32, 16, 4, 4, false, false, true},  // the potrf trailing update, W = 32
    {32, 128, 16, 4, 4, false, true, false},  // the geqrf panel update V^H A22, nb = 32
    {32, 32, 16, 4, 4, true, false, false},
    {16, 16, 16, 2, 2, true, false, false},
}};

// sycl-free copies of the kernels' limits; gemm.cc static_asserts them against the kernels.
inline constexpr int kSmallMaxDim = 64;
inline constexpr int kSmallWg = 128;
inline constexpr int kSmallTiledMaxDim = 56;  // float NN above 32: one matrix per (NB/4)^2 lanes

// The work-group `small` launches: the batched kernel's 128 lanes, or the float NN tiled leg's
// (NB/4)^2 with NB = 48 or 56 (small_batched.hh).
template <class T>
constexpr int small_wg(bool nn, int max_dim) {
    if (!std::is_same_v<T, float> || !nn || max_dim <= 32 || max_dim > kSmallTiledMaxDim) return kSmallWg;
    const int nb = max_dim <= 48 ? 48 : 56;
    return (nb / 4) * (nb / 4);
}
inline constexpr int kDirectWg = 64;
inline constexpr int kTiledWg = 256;

template <class T>
inline constexpr bool is_complex_v = !std::is_same_v<T, float> && !std::is_same_v<T, double>;

// Every compiled choice, once, in tie-break order (§6.3): simpler first, vendor last.
template <class T>
constexpr auto candidates() {
    constexpr std::array<Wide, 5> wides{Wide{64, 64, 16}, Wide{128, 32, 16}, Wide{32, 128, 16}, Wide{32, 32, 16},
                                        Wide{16, 16, 16}};
    if constexpr (std::is_same_v<T, float>) {
        std::array<GemmChoice, 19> out{};
        std::size_t i = 0;
        out[i++] = Direct{};
        out[i++] = Tiled{};
        out[i++] = Small{};
        for (const RegCfg& r : reg_configs) out[i++] = Reg{r.m, r.n, r.k, r.u};
        for (const Wide& w : wides) out[i++] = w;
        out[i++] = Vendor{};
        return out;
    } else if constexpr (std::is_same_v<T, double>) {
        return std::array<GemmChoice, 9>{Direct{}, Tiled{}, Small{}, wides[0], wides[1], wides[2], wides[3],
                                         wides[4], Vendor{}};
    } else {
        return std::array<GemmChoice, 8>{Direct{}, Tiled{}, wides[0], wides[1], wides[2], wides[3], wides[4],
                                         Vendor{}};
    }
}

// Legacy BATCHLAS_GEMM_SYCL_KERNEL names (now BATCHLAS_GEMM_ROUTE values) until phase 5. A
// transposed variant's name maps to its config's family spelling; the form is derived per call.
// The deleted variants (S1U1, S2U2, TT8x4/4x8, persistent, split-K, S1U4, Large TT4x8) have no
// alias, so their names throw (R6).
inline constexpr std::array<select::Alias, 102> aliases{{
    {"tiled16", "tiled"}, {"tile16", "tiled"}, {"smallbatched", "small"},
    {"register32", "reg:m=32:n=32:k=8:u=1"}, {"reg32", "reg:m=32:n=32:k=8:u=1"}, {"32x32", "reg:m=32:n=32:k=8:u=1"},
    {"register64", "reg:m=64:n=64:k=8:u=1"}, {"reg64", "reg:m=64:n=64:k=8:u=1"}, {"64x64", "reg:m=64:n=64:k=8:u=1"},
    {"register64k16", "reg:m=64:n=64:k=16:u=1"}, {"reg64k16", "reg:m=64:n=64:k=16:u=1"},
    {"64x64x16", "reg:m=64:n=64:k=16:u=1"}, {"register64k16tn", "reg:m=64:n=64:k=16:u=1"},
    {"reg64k16tn", "reg:m=64:n=64:k=16:u=1"}, {"64x64x16tn", "reg:m=64:n=64:k=16:u=1"},
    {"register64k16nt", "reg:m=64:n=64:k=16:u=1"}, {"reg64k16nt", "reg:m=64:n=64:k=16:u=1"},
    {"64x64x16nt", "reg:m=64:n=64:k=16:u=1"}, {"register64k16tt", "reg:m=64:n=64:k=16:u=1"},
    {"reg64k16tt", "reg:m=64:n=64:k=16:u=1"}, {"64x64x16tt", "reg:m=64:n=64:k=16:u=1"},
    {"register128x32k16", "reg:m=128:n=32:k=16:u=1"}, {"reg128x32k16", "reg:m=128:n=32:k=16:u=1"},
    {"128x32x16", "reg:m=128:n=32:k=16:u=1"}, {"register128x32k16tn", "reg:m=128:n=32:k=16:u=1"},
    {"reg128x32k16tn", "reg:m=128:n=32:k=16:u=1"}, {"128x32x16tn", "reg:m=128:n=32:k=16:u=1"},
    {"register128x32k16nt", "reg:m=128:n=32:k=16:u=1"}, {"reg128x32k16nt", "reg:m=128:n=32:k=16:u=1"},
    {"128x32x16nt", "reg:m=128:n=32:k=16:u=1"}, {"register128x32k16tt", "reg:m=128:n=32:k=16:u=1"},
    {"reg128x32k16tt", "reg:m=128:n=32:k=16:u=1"}, {"128x32x16tt", "reg:m=128:n=32:k=16:u=1"},
    {"register128x32k32tn", "reg:m=128:n=32:k=32:u=1"}, {"reg128x32k32tn", "reg:m=128:n=32:k=32:u=1"},
    {"128x32x32tn", "reg:m=128:n=32:k=32:u=1"}, {"128x32x32_s2_u1_tn", "reg:m=128:n=32:k=32:u=1"},
    {"register128x32k32nt", "reg:m=128:n=32:k=32:u=1"}, {"reg128x32k32nt", "reg:m=128:n=32:k=32:u=1"},
    {"128x32x32nt", "reg:m=128:n=32:k=32:u=1"}, {"128x32x32_s2_u1_nt", "reg:m=128:n=32:k=32:u=1"},
    {"register128x32k32tt", "reg:m=128:n=32:k=32:u=1"}, {"reg128x32k32tt", "reg:m=128:n=32:k=32:u=1"},
    {"128x32x32tt", "reg:m=128:n=32:k=32:u=1"}, {"128x32x32_s2_u1_tt", "reg:m=128:n=32:k=32:u=1"},
    {"register128x32k32", "reg:m=128:n=32:k=32:u=1"}, {"reg128x32k32", "reg:m=128:n=32:k=32:u=1"},
    {"128x32x32", "reg:m=128:n=32:k=32:u=1"}, {"register128x32k32s2u1", "reg:m=128:n=32:k=32:u=1"},
    {"reg128x32k32s2u1", "reg:m=128:n=32:k=32:u=1"}, {"128x32x32_s2_u1", "reg:m=128:n=32:k=32:u=1"},
    {"register128x32k32s2u1aligned", "reg:m=128:n=32:k=32:u=1"}, {"reg128x32k32s2u1aligned", "reg:m=128:n=32:k=32:u=1"},
    {"128x32x32_s2_u1_aligned", "reg:m=128:n=32:k=32:u=1"}, {"register128x32k32s2u1generic", "reg:m=128:n=32:k=32:u=1"},
    {"reg128x32k32s2u1generic", "reg:m=128:n=32:k=32:u=1"}, {"128x32x32_s2_u1_generic", "reg:m=128:n=32:k=32:u=1"},
    {"register128x64k16tn", "reg:m=128:n=64:k=16:u=1"}, {"reg128x64k16tn", "reg:m=128:n=64:k=16:u=1"},
    {"128x64x16tn", "reg:m=128:n=64:k=16:u=1"}, {"register128x64k16nt", "reg:m=128:n=64:k=16:u=1"},
    {"reg128x64k16nt", "reg:m=128:n=64:k=16:u=1"}, {"128x64x16nt", "reg:m=128:n=64:k=16:u=1"},
    {"register128x64k16tt", "reg:m=128:n=64:k=16:u=1"}, {"reg128x64k16tt", "reg:m=128:n=64:k=16:u=1"},
    {"128x64x16tt", "reg:m=128:n=64:k=16:u=1"},
    {"register128x64k32large", "reg:m=128:n=64:k=32:u=4"}, {"reg128x64k32large", "reg:m=128:n=64:k=32:u=4"},
    {"128x64x32large", "reg:m=128:n=64:k=32:u=4"}, {"register128x64k32largeu2", "reg:m=128:n=64:k=32:u=2"},
    {"reg128x64k32largeu2", "reg:m=128:n=64:k=32:u=2"}, {"128x64x32large_u2", "reg:m=128:n=64:k=32:u=2"},
    {"register128x128k8", "reg:m=128:n=128:k=8:u=1"}, {"reg128x128k8", "reg:m=128:n=128:k=8:u=1"},
    {"128x128x8", "reg:m=128:n=128:k=8:u=1"},
    {"register32x128k16", "reg:m=32:n=128:k=16:u=1"}, {"reg32x128k16", "reg:m=32:n=128:k=16:u=1"},
    {"32x128x16", "reg:m=32:n=128:k=16:u=1"}, {"register32x128k16tn", "reg:m=32:n=128:k=16:u=1"},
    {"reg32x128k16tn", "reg:m=32:n=128:k=16:u=1"}, {"32x128x16tn", "reg:m=32:n=128:k=16:u=1"},
    {"register32x128k16tt", "reg:m=32:n=128:k=16:u=1"}, {"reg32x128k16tt", "reg:m=32:n=128:k=16:u=1"},
    {"32x128x16tt", "reg:m=32:n=128:k=16:u=1"},
    {"register64x64k16wide", "wide:m=64:n=64:k=16"}, {"reg64x64k16wide", "wide:m=64:n=64:k=16"},
    {"64x64x16wide", "wide:m=64:n=64:k=16"}, {"register64x64k16widecn", "wide:m=64:n=64:k=16"},
    {"reg64x64k16widecn", "wide:m=64:n=64:k=16"}, {"64x64x16wide_cn", "wide:m=64:n=64:k=16"},
    {"register64x64k16widenc", "wide:m=64:n=64:k=16"}, {"reg64x64k16widenc", "wide:m=64:n=64:k=16"},
    {"64x64x16wide_nc", "wide:m=64:n=64:k=16"},
    {"register128x32k16widenc", "wide:m=128:n=32:k=16"}, {"reg128x32k16widenc", "wide:m=128:n=32:k=16"},
    {"128x32x16wide_nc", "wide:m=128:n=32:k=16"},
    {"register32x128k16widecn", "wide:m=32:n=128:k=16"}, {"reg32x128k16widecn", "wide:m=32:n=128:k=16"},
    {"32x128x16wide_cn", "wide:m=32:n=128:k=16"},
    {"32x32x16wide", "wide:m=32:n=32:k=16"}, {"16x16x16wide", "wide:m=16:n=16:k=16"},
    {"vendor:direct", "vendor"},
}};

// BATCHLAS_GEMM_ROUTE words that meant "the native kernel family" or "the vendor library".
inline constexpr std::array<select::Alias, 7> class_aliases{{
    {"register_tiled", "native"}, {"native:register_tiled", "native"}, {"native:auto", "native"},
    {"sycl", "native"}, {"custom", "native"}, {"vendor:auto", "vendor"}, {"auto:auto", "auto"},
}};
// BATCHLAS_GEMM_VARIANT's own vocabulary: its `native` was the raw cuBLAS call (route_env.hh).
inline constexpr std::array<select::Alias, 7> legacy_aliases{{
    {"native", "vendor"}, {"cuda-native", "vendor"}, {"direct-cuda", "vendor"}, {"cublasdx", "vendor"},
    {"dx", "vendor"}, {"sycl", "native"}, {"custom", "native"},
}};

// Generality order (§5.5): direct serves every GPU shape, vendor everything else (CPU, precision).
inline constexpr std::array<std::string_view, 2> last_resort{"direct", "vendor"};
inline constexpr select::Rules rules{aliases, last_resort, class_aliases, legacy_aliases};

// C folds to T for a real scalar. layout: packed = A, B, C contiguous with 16-byte bases.
// Work ~ m n k batch, so every log key weighs 1.
inline constexpr std::array<std::string_view, 7> key_names{"ta:exact", "tb:exact", "layout:exact", "m:log",
                                                           "n:log",    "k:log",    "batch:log"};

// The tuner's demand-driven grid (plan §3); tools/transcribe/gemm_transcribe.cc spells it again.
// Squares for every form and both layouts; panels and skinny shapes for the issued forms only,
// packed panels from m, n >= 128.
inline constexpr std::array<int, 14> grid_square{8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024};
inline constexpr std::array<int, 7> grid_panel_mn{32, 64, 128, 256, 512, 1024, 2048};
inline constexpr std::array<int, 6> grid_panel_k{8, 16, 32, 64, 96, 128};
inline constexpr std::array<int, 6> grid_skinny_mn{64, 128, 256, 512, 1024, 2048};
inline constexpr std::array<int, 2> grid_skinny_k{256, 1024};
inline constexpr std::array<int, 3> grid_batch{128, 2048, 32768};
inline constexpr int grid_packed_panel_min = 128;
inline constexpr std::array<std::string_view, 4> grid_real_forms{"NN", "NT", "TN", "TT"};
inline constexpr std::array<std::string_view, 3> grid_real_panel_forms{"NN", "NT", "TN"};
inline constexpr std::array<std::string_view, 6> grid_complex_forms{"NN", "NT", "TN", "NC", "CN", "CT"};
inline constexpr std::array<std::string_view, 5> grid_complex_panel_forms{"NN", "NT", "NC", "TN", "CN"};

}  // namespace batchlas::ops::gemm
