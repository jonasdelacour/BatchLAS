// Every gemm candidate, pinned: docs/design/flat-kernel-selection.md §8 and
// flat-kernel-selection-phase3-plan.md §1.3. The oracle for which (config, form) pairs exist is
// the old KernelVariant list, written out here; which kernel ran is read back from the select
// trace, the kernel trace or a bit-for-bit comparison with the family's launcher, never assumed
// from the pin being accepted.
#include <gtest/gtest.h>

#include <batchlas/blas/functions/gemm.hh>
#include <src/select/coverage.hh>
#include <batchlas/no_route.hh>
#include <src/select/vendor.hh>
#include <batchlas/error.hh>
#include <batchlas/util/env.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>
#include <batchlas/util/sycl-vector.hh>
#include <batchlas/util/workspace.hh>

#include <tests/test_utils.hh>
#include <batchlas/verify/tolerance.hh>

#include <src/ops/gemm/choice.hh>
#include <src/ops/gemm/gemm_kernels.hh>
#include <src/util/resident_capacity.hh>
#include <tests/shipped_table_pick.hh>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

using namespace batchlas;

namespace {

namespace og = batchlas::ops::gemm;
using C = og::GemmChoice;
using Pin = select::ScopedPin<C>;
template <typename T>
using MVof = MatrixView<T, MatrixFormat::Dense>;
template <typename T>
using RealOf = typename batchlas::base_type<T>::type;
template <typename T>
constexpr bool kCx = test_utils::is_complex<T>::value;

template <class... F>
struct overloaded : F... { using F::operator()...; };
template <class... F>
overloaded(F...) -> overloaded<F...>;

constexpr Transpose kN = Transpose::NoTrans, kT = Transpose::Trans, kC = Transpose::ConjTrans;
const Transpose kForms[] = {kN, kT, kC};

template <typename T>
bool same_bits(T a, T b) {
    return std::memcmp(&a, &b, sizeof(T)) == 0;
}
const char* tw(Transpose t) { return t == kN ? "N" : (t == kT ? "T" : "C"); }

// ---- the oracle: the compiled kernels as the pre-P3.4 KernelVariant enum named them ----

// slm: the launch's local memory in bytes, set where it exceeds 16 KiB (an Apple GPU's budget is
// ~28 KiB, CUDA's smallest 45,056 B). The 49,920 B rows are what the CUDA driver saw
// (cuLaunchKernel interposer log); the others are the kernels' staged tiles, 4 B x stages x
// ((m+1)k + (k+1)n), 128x128 its own layout.
struct RegRow {
    int m, n, k, u;
    std::vector<std::string> forms;
    std::map<std::string, std::string> trace;
    std::int64_t slm = 0;
};
// The device's resident::device_slm_budget, set by the fixture: a reg tile above it is refused.
// An Apple GPU's runtime takes 8 B per work-item from it, so a group narrower than the device
// maximum stages that much more per missing item.
std::int64_t g_slm_budget = std::numeric_limits<std::int64_t>::max();
std::int64_t g_slm_per_item = 0;
int g_max_wg = 0;
const std::vector<RegRow>& reg_rows() {
    static const std::vector<RegRow> rows{
        {32, 32, 8, 1, {"NN"}, {{"NN", "gemm_sycl_register_32x32"}}},
        {64, 64, 8, 1, {"NN"}, {{"NN", "gemm_sycl_register_64x64"}}},
        {64, 64, 16, 1, {"NN", "NT", "TN", "TT"},
         {{"NN", "gemm_sycl_register_64x64_k16"}, {"NT", "gemm_sycl_register_64x64_k16_nt"},
          {"TN", "gemm_sycl_register_64x64_k16_tn"}, {"TT", "gemm_sycl_register_64x64_k16_tt"}}},
        {128, 32, 16, 1, {"NN", "NT", "TN", "TT"},
         {{"NN", "gemm_sycl_register_128x32_k16"}, {"NT", "gemm_sycl_register_128x32_k16_nt"},
          {"TN", "gemm_sycl_register_128x32_k16_tn"}, {"TT", "gemm_sycl_register_128x32_k16_tt"}},
         20864},
        {128, 32, 32, 1, {"NN", "NT", "TN", "TT"},
         {{"NN", "gemm_sycl_register_128x32_k32_s2_u1_generic"},
          {"NN+aligned", "gemm_sycl_register_128x32_k32_s2_u1_aligned"},
          {"NT", "gemm_sycl_register_128x32_k32_nt"}, {"TN", "gemm_sycl_register_128x32_k32_tn"},
          {"TT", "gemm_sycl_register_128x32_k32_tt"}},
         41472},
        {128, 64, 16, 1, {"NT", "TN", "TT"},
         {{"NT", "gemm_sycl_register_128x64_k16_nt"}, {"TN", "gemm_sycl_register_128x64_k16_tn"},
          {"TT", "gemm_sycl_register_128x64_k16_tt"}}},
        {32, 128, 16, 1, {"NN", "TN", "TT"},
         {{"NN", "gemm_sycl_register_32x128_k16"}, {"TN", "gemm_sycl_register_32x128_k16_tn"},
          {"TT", "gemm_sycl_register_32x128_k16_tt"}}},
        {128, 64, 32, 4, {"NN"},
         {{"NN", "gemm_sycl_register_128x64_k32_large"}, {"NN+aligned", "gemm_sycl_register_128x64_k32_large_aligned"}},
         49920},
        {128, 64, 32, 2, {"NN"},
         {{"NN", "gemm_sycl_register_128x64_k32_large_u2"},
          {"NN+aligned", "gemm_sycl_register_128x64_k32_large_u2_aligned"}},
         49920},
        {128, 128, 8, 1, {"NN", "NT", "TN", "TT"},
         {{"NN", "gemm_sycl_register_128x128_k8"}, {"NN+aligned", "gemm_sycl_register_128x128_k8_aligned"},
          {"NT", "gemm_sycl_register_128x128_k8_nt"}, {"NT+aligned", "gemm_sycl_register_128x128_k8_nt_aligned"},
          {"TN", "gemm_sycl_register_128x128_k8_tn"}, {"TN+aligned", "gemm_sycl_register_128x128_k8_tn_aligned"},
          {"TT", "gemm_sycl_register_128x128_k8_tt"}, {"TT+aligned", "gemm_sycl_register_128x128_k8_tt_aligned"}},
         25088},
        // The copy_async pipeline is the predicated leg only; the aligned leg is u1's two-stage body.
        {128, 128, 8, 4, {"NN", "NT", "TN", "TT"},
         {{"NN", "gemm_sycl_register_128x128_k8_s4"}, {"NN+aligned", "gemm_sycl_register_128x128_k8_aligned"},
          {"NT", "gemm_sycl_register_128x128_k8_s4_nt"}, {"NT+aligned", "gemm_sycl_register_128x128_k8_nt_aligned"},
          {"TN", "gemm_sycl_register_128x128_k8_s4_tn"}, {"TN+aligned", "gemm_sycl_register_128x128_k8_tn_aligned"},
          {"TT", "gemm_sycl_register_128x128_k8_s4_tt"}, {"TT+aligned", "gemm_sycl_register_128x128_k8_tt_aligned"}},
         33792},
    };
    return rows;
}
// Forms are the complex instantiations (op(A) op(B) over N, T, C); a real scalar runs the N/C
// one its T operands fold to.
struct WideRow { int m, n, k; std::map<std::string, std::string> trace; };
const std::vector<WideRow>& wide_rows() {
    static const std::vector<WideRow> rows{
        {64, 64, 16, {{"NN", "gemm_sycl_register_64x64_k16_wide"}, {"NN+aligned", "gemm_sycl_register_64x64_k16_wide_aligned"},
                      {"NT", "gemm_sycl_register_64x64_k16_wide_nt"}, {"NC", "gemm_sycl_register_64x64_k16_wide_nc"},
                      {"TN", "gemm_sycl_register_64x64_k16_wide_tn"}, {"TT", "gemm_sycl_register_64x64_k16_wide_tt"},
                      {"TC", "gemm_sycl_register_64x64_k16_wide_tc"}, {"CN", "gemm_sycl_register_64x64_k16_wide_cn"},
                      {"CT", "gemm_sycl_register_64x64_k16_wide_ct"}, {"CC", "gemm_sycl_register_64x64_k16_wide_cc"},
                      {"NT+aligned", "gemm_sycl_register_64x64_k16_wide_nt_aligned"},
                      {"NC+aligned", "gemm_sycl_register_64x64_k16_wide_nc_aligned"},
                      {"TN+aligned", "gemm_sycl_register_64x64_k16_wide_tn_aligned"},
                      {"TT+aligned", "gemm_sycl_register_64x64_k16_wide_tt_aligned"},
                      {"TC+aligned", "gemm_sycl_register_64x64_k16_wide_tc_aligned"},
                      {"CN+aligned", "gemm_sycl_register_64x64_k16_wide_cn_aligned"},
                      {"CT+aligned", "gemm_sycl_register_64x64_k16_wide_ct_aligned"},
                      {"CC+aligned", "gemm_sycl_register_64x64_k16_wide_cc_aligned"}}},
        {128, 32, 16, {{"NC", "gemm_sycl_register_128x32_k16_wide_nc"}, {"NT", "gemm_sycl_register_128x32_k16_wide_nt"},
                       {"TN", "gemm_sycl_register_128x32_k16_wide_tn"}, {"CN", "gemm_sycl_register_128x32_k16_wide_cn"}}},
        {32, 128, 16, {{"CN", "gemm_sycl_register_32x128_k16_wide_cn"}, {"TN", "gemm_sycl_register_32x128_k16_wide_tn"},
                       {"NT", "gemm_sycl_register_32x128_k16_wide_nt"}, {"NC", "gemm_sycl_register_32x128_k16_wide_nc"}}},
        {32, 32, 16, {{"NN", "gemm_sycl_register_32x32_k16_wide"}, {"NT", "gemm_sycl_register_32x32_k16_wide_nt"},
                      {"NC", "gemm_sycl_register_32x32_k16_wide_nc"}, {"TN", "gemm_sycl_register_32x32_k16_wide_tn"},
                      {"CN", "gemm_sycl_register_32x32_k16_wide_cn"}, {"CT", "gemm_sycl_register_32x32_k16_wide_ct"}}},
        {16, 16, 16, {{"NN", "gemm_sycl_register_16x16_k16_wide"}, {"NT", "gemm_sycl_register_16x16_k16_wide_nt"},
                      {"NC", "gemm_sycl_register_16x16_k16_wide_nc"}, {"TN", "gemm_sycl_register_16x16_k16_wide_tn"},
                      {"CN", "gemm_sycl_register_16x16_k16_wide_cn"}, {"CT", "gemm_sycl_register_16x16_k16_wide_ct"}}},
        {64, 32, 16, {{"NN", "gemm_sycl_register_64x32_k16_wide"}, {"NT", "gemm_sycl_register_64x32_k16_wide_nt"},
                      {"NC", "gemm_sycl_register_64x32_k16_wide_nc"}, {"TN", "gemm_sycl_register_64x32_k16_wide_tn"},
                      {"CN", "gemm_sycl_register_64x32_k16_wide_cn"}, {"CT", "gemm_sycl_register_64x32_k16_wide_ct"}}},
        {32, 64, 16, {{"NN", "gemm_sycl_register_32x64_k16_wide"}, {"NT", "gemm_sycl_register_32x64_k16_wide_nt"},
                      {"NC", "gemm_sycl_register_32x64_k16_wide_nc"}, {"TN", "gemm_sycl_register_32x64_k16_wide_tn"},
                      {"CN", "gemm_sycl_register_32x64_k16_wide_cn"}, {"CT", "gemm_sycl_register_32x64_k16_wide_ct"}}},
        {64, 64, 8, {{"NN", "gemm_sycl_register_64x64_k8_wide"}, {"NT", "gemm_sycl_register_64x64_k8_wide_nt"},
                     {"NC", "gemm_sycl_register_64x64_k8_wide_nc"}, {"TN", "gemm_sycl_register_64x64_k8_wide_tn"},
                     {"TT", "gemm_sycl_register_64x64_k8_wide_tt"}, {"TC", "gemm_sycl_register_64x64_k8_wide_tc"},
                     {"CN", "gemm_sycl_register_64x64_k8_wide_cn"}, {"CT", "gemm_sycl_register_64x64_k8_wide_ct"},
                     {"CC", "gemm_sycl_register_64x64_k8_wide_cc"},
                     {"NN+aligned", "gemm_sycl_register_64x64_k8_wide_aligned"},
                     {"NT+aligned", "gemm_sycl_register_64x64_k8_wide_nt_aligned"},
                     {"NC+aligned", "gemm_sycl_register_64x64_k8_wide_nc_aligned"},
                     {"TN+aligned", "gemm_sycl_register_64x64_k8_wide_tn_aligned"},
                     {"TT+aligned", "gemm_sycl_register_64x64_k8_wide_tt_aligned"},
                     {"TC+aligned", "gemm_sycl_register_64x64_k8_wide_tc_aligned"},
                     {"CN+aligned", "gemm_sycl_register_64x64_k8_wide_cn_aligned"},
                     {"CT+aligned", "gemm_sycl_register_64x64_k8_wide_ct_aligned"},
                     {"CC+aligned", "gemm_sycl_register_64x64_k8_wide_cc_aligned"}}},
    };
    return rows;
}

// A register tile's form: a real ConjTrans is its Trans.
std::string reg_form(Transpose ta, Transpose tb) {
    return std::string(ta == kN ? "N" : "T") + (tb == kN ? "N" : "T");
}
// A wide tile's form: the exact (op(A), op(B)) for a complex scalar; a real one folds T to C
// (conj is the identity, and only the C builds are compiled for it).
template <typename T>
std::string wide_form(Transpose ta, Transpose tb) {
    auto w = [](Transpose t) { return t == kN ? "N" : (kCx<T> && t == kT) ? "T" : "C"; };
    return std::string(w(ta)) + w(tb);
}

const RegRow* reg_row(const og::Reg& r) {
    for (const auto& row : reg_rows())
        if (row.m == r.m && row.n == r.n && row.k == r.k && row.u == r.u) return &row;
    return nullptr;
}
// Whether the row's staged tiles fit the budget of its own launch width.
bool reg_fits(const RegRow& row) {
    int wg = g_max_wg;
    for (const og::RegCfg& c : og::reg_configs)
        if (c.m == row.m && c.n == row.n && c.k == row.k && c.u == row.u) wg = c.threads();
    return row.slm <= g_slm_budget + g_slm_per_item * std::max(0, g_max_wg - wg);
}
const WideRow* wide_row(const og::Wide& w) {
    for (const auto& row : wide_rows())
        if (row.m == w.m && row.n == w.n && row.k == w.k) return &row;
    return nullptr;
}
// The bytes a wide launch gives its two local accessors: one k x (m + n) tile of T, which wide v2
// (64x64, every scalar but double) double-buffers; complex<float> v2 is 32 KiB, a Metal limit, and
// complex<double> 64x64x16 is 64 KiB, over acpp's 48 KiB without the opt-in.
// WP-K: every non-64x64 tile double-buffers too while two stages stay within 40 KiB.
template <typename T>
std::int64_t wide_slm(const WideRow& row) {
    const bool v2 = row.m == 64 && row.n == 64 && !std::is_same_v<T, double>;
    const std::int64_t one = std::int64_t(row.k) * (row.m + row.n) * std::int64_t(sizeof(T));
    const bool two = v2 || ((row.m != 64 || row.n != 64) && 2 * one <= 40960);
    return (two ? 2 : 1) * one;
}
template <typename T>
bool wide_fits(const WideRow& row) {
    int wg = g_max_wg;
    for (const og::WideCfg& c : og::wide_configs)
        if (c.m == row.m && c.n == row.n && c.k == row.k) wg = c.threads();
    return wide_slm<T>(row) <= g_slm_budget + g_slm_per_item * std::max(0, g_max_wg - wg);
}

// split-k's host: a wide row other than 64x64 (launch_wide_transposed in every form), s 2, 4 or 8.
// The oracle has no batch: can_run's batch * s, int group id and scratch budget terms are not modelled
// (SplitKGridCeilingIsBatchTimesSlices covers the first; GridCeilingIsACanRunTerm skips split-k).
const WideRow* splitk_row(const og::SplitK& sk) {
    if ((sk.m == 64 && sk.n == 64) || (sk.s != 2 && sk.s != 4 && sk.s != 8)) return nullptr;
    return wide_row(og::Wide{sk.m, sk.n, sk.k});
}

// small, spelled apart from choice.hh: complex<double> stops at 32 (its 64 bucket is 66.5 KB);
// 128 / (2 NB) slots of NB x (NB + 1) for NB the bucket of max(m, n, k), or the float NN tiled
// leg's (33..56) 2 NB (NB + 4) on (NB/4)^2 lanes, within the device's local memory budget.
template <typename T>
bool small_runs(Transpose ta, Transpose tb, int m, int n, int k) {
    const int mx = std::max({m, n, k});
    if (mx > (sizeof(T) == 16 ? 32 : 64)) return false;
    const bool tiled = std::is_same_v<T, float> && ta == kN && tb == kN && mx > 32 && mx <= 56;
    const int nb = tiled ? (mx <= 48 ? 48 : 56) : mx <= 8 ? 8 : mx <= 16 ? 16 : mx <= 32 ? 32 : 64;
    const int wg = tiled ? (nb / 4) * (nb / 4) : 128;
    const std::int64_t bytes = std::int64_t(tiled ? 2 * nb * (nb + 4) : (128 / (2 * nb)) * nb * (nb + 1)) * sizeof(T);
    return bytes <= g_slm_budget + g_slm_per_item * std::max(0, g_max_wg - wg);
}

// Does this native choice have an instantiation for (ta, tb) at this shape, on a GPU?
template <typename T>
bool oracle_native(const C& c, Transpose ta, Transpose tb, int m, int n, int k) {
    if (m < 1 || n < 1 || k < 1) return false;
    return std::visit(overloaded{
        [](og::Direct) { return true; },
        [](og::Tiled) { return true; },
        [&](og::Small) { return small_runs<T>(ta, tb, m, n, k); },
        [&](const og::Reg& r) {
            const RegRow* row = reg_row(r);
            const std::string f = reg_form(ta, tb);
            return std::is_same_v<T, float> && row && reg_fits(*row) &&
                   std::find(row->forms.begin(), row->forms.end(), f) != row->forms.end();
        },
        [&](const og::Wide& w) {
            const WideRow* row = wide_row(w);
            return row && wide_fits<T>(*row) && row->trace.count(wide_form<T>(ta, tb)) > 0;
        },
        [&](const og::SplitK& sk) {
            const WideRow* row = splitk_row(sk);
            return row && wide_fits<T>(*row) && row->trace.count(wide_form<T>(ta, tb)) > 0;
        },
        [](og::Vendor) { return false; },
    }, c);
}

// The kernel-trace name the pinned choice must launch; "" for the vendor (no SYCL kernel).
// base8: the operands sit 8 bytes past a 16-byte boundary (run_pinned_base8).
template <typename T>
std::string expected_kernel(const C& c, Transpose ta, Transpose tb, bool aligned_layout, bool base8 = false) {
    return std::visit(overloaded{
        [](og::Direct) { return std::string("gemm_sycl_direct"); },
        [](og::Tiled) { return std::string("gemm_sycl_tiled16"); },
        [](og::Small) { return std::string("gemm_sycl_small_batched"); },
        [&](const og::Reg& r) {
            const RegRow* row = reg_row(r);
            const std::string f = reg_form(ta, tb);
            if (aligned_layout && row->trace.count(f + "+aligned")) return row->trace.at(f + "+aligned");
            return row->trace.at(f);
        },
        [&](const og::Wide& w) {
            const WideRow* row = wide_row(w);
            std::string f = wide_form<T>(ta, tb);
            // A complex<double> is one 16-byte vector, so its strided views take the aligned leg too.
            // Wide v2 (64x64, every scalar but double) has the 16-byte leg in every form; of the
            // others only double's NN 64x64x16 kernel has an aligned leg (64x64x8 NN is transposed's).
            const bool aligned = !base8 && (aligned_layout || sizeof(T) == 16);
            const bool v2 = w.m == 64 && w.n == 64 && !std::is_same_v<T, double>;
            const bool nn_leg = f == "NN" && (w.m != 64 || w.n != 64 || w.k == 16);
            if ((nn_leg || v2) && aligned && row->trace.count(f + "+aligned")) f += "+aligned";
            return row->trace.at(f);
        },
        [&](const og::SplitK& sk) { return splitk_row(sk)->trace.at(wide_form<T>(ta, tb)) + "_splitk"; },
        [](og::Vendor) { return std::string(); },
    }, c);
}

// ---- problems --------------------------------------------------------------------------

// One operand's footprint inside the shared buffer: item `it` starts at off + it * stride.
struct Region {
    std::size_t off = 0;
    int rows = 0, cols = 0, ld = 0, stride = 0;
};

enum class Layout { Packed, Strided, Padded, Gap1, OddLd, ItemSkew, Ld2 };

struct Spec {
    Transpose ta = kN, tb = kN;
    int m = 8, n = 8, k = 8, batch = 3;
    Layout layout = Layout::Strided;
    std::optional<Layout> layout_b;  // B's own layout (A and C keep `layout`)
    int period = 0;        // > 0: item it repeats item it % period
    bool beta_zero = false;
    bool nan_c = false;    // C holds NaN (with beta_zero: a kernel that reads C returns NaN)
    bool nan_poison = false;  // the poison is NaN: a gap read times a zero is NaN, not a silent -0
    unsigned seed = 1;
};

// A, B and C live in one buffer of a large finite poison. Packed: contiguous operands with
// 16-byte bases (layout=packed). Strided: ld = rows + 3, an odd batch stride, odd bases. Gap1: the
// tuner's strided cell, ld = rows + 1, odd stride and bases (one poison element per column). Padded:
// 16-byte bases, ld and stride multiples of 4 but neither natural (an aligned-leg view). OddLd:
// every item's base 16-byte aligned, ld odd. ItemSkew: Padded's ld, stride = 1 (mod 4), so item i
// starts i (mod 4) elements past a 16-byte boundary. Ld2: Padded with ld = 2 (mod 4), so
// complex<float> columns alternate 32- and 16-byte aligned and float's ld is not whole granules. The
// panel form carves A, B and C from one parent at its ld and stride, B aliasing A.
template <typename T>
struct Problem {
    Spec s;
    T alpha{}, beta{};
    Region a, b, c;
    UnifiedVector<T> mem;
    UnifiedVector<T> mem0;
    UnifiedVector<T*> pa, pb, pc;

    MVof<T> A() { return {mem.data() + a.off, a.rows, a.cols, a.ld, a.stride, s.batch, pa.data()}; }
    MVof<T> B() { return {mem.data() + b.off, b.rows, b.cols, b.ld, b.stride, s.batch, pb.data()}; }
    MVof<T> Cv() { return {mem.data() + c.off, c.rows, c.cols, c.ld, c.stride, s.batch, pc.data()}; }
    static std::size_t at(const Region& r, int it, int i, int j) {
        return r.off + std::size_t(it) * r.stride + std::size_t(j) * r.ld + i;
    }
};

template <typename T>
T poison() {
    return MatrixView<T, MatrixFormat::Dense>::make(RealOf<T>(-999), RealOf<T>(777));
}

template <typename T>
void init(Problem<T>& p, std::size_t total) {
    using R = RealOf<T>;
    const Spec& s = p.s;
    p.alpha = MatrixView<T, MatrixFormat::Dense>::make(R(1.25), R(-0.5));
    p.beta = s.beta_zero ? T(0) : MatrixView<T, MatrixFormat::Dense>::make(R(-0.75), R(0.25));
    const R nan = std::numeric_limits<R>::quiet_NaN();
    p.mem = UnifiedVector<T>(total, s.nan_poison ? MatrixView<T, MatrixFormat::Dense>::make(nan, nan) : poison<T>());
    p.pa = UnifiedVector<T*>(s.batch, nullptr);
    p.pb = UnifiedVector<T*>(s.batch, nullptr);
    p.pc = UnifiedVector<T*>(s.batch, nullptr);
    std::mt19937 gen(s.seed);
    std::uniform_real_distribution<R> u(R(-1), R(1));
    const int reps = s.period > 0 ? std::min(s.period, s.batch) : s.batch;
    for (int it = 0; it < s.batch; ++it) {
        const int r = it % reps;
        auto put = [&](const Region& g, bool is_c) {
            for (int j = 0; j < g.cols; ++j)
                for (int i = 0; i < g.rows; ++i) {
                    T& v = p.mem[Problem<T>::at(g, it, i, j)];
                    if (it != r) v = p.mem[Problem<T>::at(g, r, i, j)];
                    else if (is_c && s.nan_c) v = MatrixView<T, MatrixFormat::Dense>::make(nan, nan);
                    else {
                        const auto re = u(gen), im = u(gen);
                        v = MatrixView<T, MatrixFormat::Dense>::make(re, im);
                    }
                }
        };
        put(p.a, false);
        if (p.b.off != p.a.off) put(p.b, false);
        put(p.c, true);
    }
    p.mem0 = p.mem;
}

std::size_t up4(std::size_t v) { return (v + 3) / 4 * 4; }

template <typename T>
Problem<T> make_problem(const Spec& s) {
    Problem<T> p;
    p.s = s;
    const int ar = s.ta == kN ? s.m : s.k, ac = s.ta == kN ? s.k : s.m;
    const int br = s.tb == kN ? s.k : s.n, bc = s.tb == kN ? s.n : s.k;
    auto region = [&](std::size_t off, int rows, int cols, Layout lay) {
        if (lay == Layout::Packed) return Region{up4(off), rows, cols, rows, rows * cols};
        const int pad = int(up4(std::size_t(rows))) + 4;
        if (lay == Layout::Padded) return Region{up4(off), rows, cols, pad, pad * cols + 4};
        if (lay == Layout::Ld2) return Region{up4(off), rows, cols, pad + 2, int(up4(std::size_t(pad + 2) * cols))};
        if (lay == Layout::OddLd)
            return Region{up4(off), rows, cols, pad + 1, int(up4(std::size_t(pad + 1) * cols))};
        if (lay == Layout::ItemSkew) return Region{up4(off), rows, cols, pad, pad * cols + 1};
        if (lay == Layout::Gap1) return Region{off | 1, rows, cols, rows + 1, (rows + 1) * cols + 3};
        return Region{off | 1, rows, cols, rows + 3, (rows + 3) * cols + 5};
    };
    auto end = [&](const Region& g) { return g.off + std::size_t(g.stride) * s.batch + 9; };
    p.a = region(8, ar, ac, s.layout);
    p.b = region(end(p.a), br, bc, s.layout_b.value_or(s.layout));
    p.c = region(end(p.b), s.m, s.n, s.layout);
    init(p, end(p.c) + 4);
    return p;
}

template <typename T>
inline constexpr bool kWideV2 = std::is_same_v<T, float> || std::is_same_v<T, std::complex<float>> ||
                                std::is_same_v<T, std::complex<double>>;

// Wide v2's leg against (A layout, B layout): 16-byte loads only when both bases are 16-byte
// aligned and both ld and batch stride are whole granules, never from the shape (160 x 136 x 48
// is ragged in m and n). Each refusing term appears alone on A and on B, so a dropped term (or the
// whole B conjunct) launches the 16-byte leg on a misaligned operand. Ld2 straddles the ld term:
// complex<float> (2 per granule) takes it, float (4) does not. complex<double> is one element per
// granule, so every layout here is whole granules: all twelve take the 16-byte leg.
template <typename T>
std::vector<std::tuple<Layout, Layout, bool>> wide_v2_leg_layouts() {
    using L = Layout;
    const bool cd = std::is_same_v<T, std::complex<double>>;
    const bool cf = cd || std::is_same_v<T, std::complex<float>>;
    return {{L::Padded, L::Padded, true}, {L::Ld2, L::Ld2, cf},       {L::OddLd, L::OddLd, cd},
            {L::ItemSkew, L::ItemSkew, cd}, {L::Padded, L::OddLd, cd}, {L::OddLd, L::Padded, cd},
            {L::Padded, L::ItemSkew, cd}, {L::ItemSkew, L::Padded, cd}, {L::Padded, L::Gap1, cd},
            {L::Gap1, L::Padded, cd},   {L::Padded, L::Ld2, cf},     {L::Ld2, L::Padded, cf}};
}

// A panel update in one parent, all three operands at the parent's ld and stride: potrf's
// A22 -= L21 L21^H (ta = N; B is A's own region, read twice), or a transposed-A shape such as
// geqrf's V^H A22 (A = U12 above A22; tb = N reads U12 again, a transposed tb reads L21).
template <typename T>
Problem<T> make_panel(Spec s, int parent) {
    s.layout = Layout::Strided;
    Problem<T> p;
    p.s = s;
    const int ld = parent + 5, stride = ld * parent + 9, r0 = 3;
    const Region l21{std::size_t(r0 + s.k), s.m, s.k, ld, stride};            // rows r0+k.., cols 0..k
    const Region u12{std::size_t(s.k) * ld + r0, s.k, s.m, ld, stride};       // rows r0..r0+k, cols k..
    p.a = s.ta == kN ? l21 : u12;
    p.b = s.tb == kN ? u12 : l21;
    p.c = {std::size_t(s.k) * ld + r0 + s.k, s.m, s.n, ld, stride};          // A22
    EXPECT_EQ(s.m, s.n) << "a panel update is square";
    init(p, std::size_t(stride) * s.batch);
    return p;
}

// The worst of `items` against the library's componentwise BLAS backward error (k eps units); the
// operands and C0 are read from the pristine copy, so a kernel that wrote into A, B or C0 cannot hide it.
template <typename T>
::testing::AssertionResult items_ok(Queue& ctx, const Problem<T>& p, std::span<const int> items) {
    using MV = MatrixView<T, MatrixFormat::Dense>;
    const Spec& s = p.s;
    auto view0 = [&](const Region& g) { return MV(p.mem0.data() + g.off, g.rows, g.cols, g.ld, g.stride, s.batch); };
    const MV C1(p.mem.data() + p.c.off, p.c.rows, p.c.cols, p.c.ld, p.c.stride, s.batch);
    const double err =
        MV::gemm_backward_error(ctx, view0(p.a), s.ta, view0(p.b), s.tb, view0(p.c), C1, MV::up(p.alpha), MV::up(p.beta), items);
    if (!batchlas::verify::pass<T>(batchlas::verify::Check::blas, s.k, err))
        return ::testing::AssertionFailure() << "worst of " << items.size() << " items: componentwise backward error " << err
                                             << " exceeds " << batchlas::verify::bound<T>(batchlas::verify::Check::blas, s.k);
    return ::testing::AssertionSuccess();
}

// The checked items are right, nothing outside C's footprint moved, and a repeating batch is
// bit-identical to its representatives.
template <typename T>
void expect_gemm(Queue& ctx, const Problem<T>& p, const std::string& what) {
    const Spec& s = p.s;
    std::vector<int> items;
    if (s.period > 0) {
        for (int it = 0; it < std::min(s.period, s.batch); ++it) items.push_back(it);
    } else if (s.batch <= 8) {
        for (int it = 0; it < s.batch; ++it) items.push_back(it);
    } else {
        items = {0, 1, s.batch / 2, s.batch - 1};
    }
    ASSERT_TRUE(items_ok(ctx, p, items)) << what;
    std::vector<char> inc(p.mem.size(), 0);
    for (int it = 0; it < s.batch; ++it)
        for (int j = 0; j < s.n; ++j)
            for (int i = 0; i < s.m; ++i) inc[Problem<T>::at(p.c, it, i, j)] = 1;
    for (std::size_t e = 0; e < p.mem.size(); ++e)
        if (!inc[e]) ASSERT_TRUE(same_bits(p.mem[e], p.mem0[e])) << what << ": wrote outside C at element " << e;
    if (s.period > 0)
        for (int it = s.period; it < s.batch; ++it)
            for (int j = 0; j < s.n; ++j)
                for (int i = 0; i < s.m; ++i)
                    ASSERT_TRUE(same_bits(p.mem[Problem<T>::at(p.c, it, i, j)],
                                          p.mem[Problem<T>::at(p.c, it % s.period, i, j)]))
                        << what << ": item " << it << " differs from its representative at (" << i << "," << j << ")";
}

const char* layout_word(Layout l) {
    return l == Layout::Packed ? " packed"
        : l == Layout::Padded   ? " padded"
        : l == Layout::Gap1     ? " ld+1"
        : l == Layout::OddLd    ? " aligned-odd-ld"
        : l == Layout::ItemSkew ? " item-skewed"
        : l == Layout::Ld2      ? " ld=2(mod 4)"
                                : " strided";
}
std::string label(const Spec& s) {
    const std::string lay = std::string(layout_word(s.layout)) + (s.layout_b ? std::string(" B") + layout_word(*s.layout_b) : "");
    return std::string(tw(s.ta)) + tw(s.tb) + lay +
           " m=" + std::to_string(s.m) + " n=" + std::to_string(s.n) + " k=" + std::to_string(s.k) +
            " batch=" + std::to_string(s.batch) + (s.beta_zero ? " beta=0" : "") + (s.nan_poison ? " NaN-poison" : "");
}
std::string name(const C& c, const Spec& s) { return select::to_string(c) + " " + label(s); }

// Packed, every stored dim a multiple of 128 x 128 x 8, but the effective m (op(A) transposed) or
// n (op(B) transposed) is 72: a leg predicate reading A.rows() / B.cols() as m / n takes the
// aligned leg here and reads past A and writes past C.
Spec ragged_effective(Transpose ta, Transpose tb) {
    return Spec{ta, tb, ta == kN ? 128 : 72, tb == kN ? 128 : 72, 128, 3, Layout::Packed};
}

// The outermost gemm trace line for whatever `run` calls, and the choice it names.
template <class F>
std::string traced_line(F&& run, std::string* all = nullptr) {
    const ScopedEnvVar trace("BATCHLAS_SELECT_TRACE", "1");
    ::testing::internal::CaptureStderr();
    try {
        run();
    } catch (...) {
        (void)::testing::internal::GetCapturedStderr();
        throw;
    }
    const std::string err = ::testing::internal::GetCapturedStderr();
    if (all) *all = err;
    std::istringstream in(err);
    for (std::string line; std::getline(in, line);)
        if (line.rfind("gemm ", 0) == 0 && line.find(" -> ") != std::string::npos) return line;
    return "<no gemm trace line in: " + err + ">";
}
template <class F>
std::string traced_choice(F&& run, std::string* all = nullptr) {
    const std::string line = traced_line(std::forward<F>(run), all);
    const auto arrow = line.find(" -> ");
    if (arrow == std::string::npos) return line;
    const std::string tail = line.substr(arrow + 4);
    return tail.substr(0, tail.find(' '));
}

struct TableGuard {
    ~TableGuard() { select::testing::use_embedded_tables(); }
};

// A synthetic gemm table for (dtype, device), every other embedded table kept.
std::vector<std::pair<std::string, std::string>> with_gemm_table(const std::string& dtype, const std::string& dev,
                                                                 const std::string& rows) {
    std::vector<std::pair<std::string, std::string>> files;
    for (const auto& t : select::embedded_tables())
        if (t.name.rfind("gemm.", 0) != 0) files.emplace_back(std::string(t.name), std::string(t.text));
    files.emplace_back("gemm." + dtype + "." + dev + ".txt",
                       "# op=gemm dtype=" + dtype + " device=" + dev + " kernels=unknown\n"
                       "# keys: ta:exact tb:exact layout:exact m:log n:log k:log batch:log\n" + rows);
    return files;
}

template <typename T, Backend B>
struct Cfg {
    using ScalarType = T;
    static constexpr Backend BackendVal = B;
};
using Types = typename test_utils::backend_types<Cfg>::type;

template <typename Config>
class GemmCandidates : public test_utils::BatchLASTest<Config> {
protected:
    using T = typename Config::ScalarType;
    static constexpr Backend B = Config::BackendVal;
    static constexpr bool kVendor = batchlas::select::level3_vendor_available<B>;

    void SetUp() override {
        test_utils::BatchLASTest<Config>::SetUp();
        if (this->HasFatalFailure() || ::testing::Test::IsSkipped()) return;
        if (!this->ctx) GTEST_SKIP() << "no queue";
        if (this->ctx->device().type != DeviceType::GPU) GTEST_SKIP() << "the native families are GPU kernels here";
        g_slm_budget = static_cast<std::int64_t>(
            resident::device_slm_budget(this->ctx->device().get_property(DeviceProperty::LOCAL_MEM_SIZE)));
        if (g_slm_budget < 49920) EXPECT_TRUE(test_utils::kSlmCappedAt48KiB) << "budget " << g_slm_budget;
        g_slm_per_item = select::device_of<B>(*this->ctx).family == "apple" ? 8 : 0;
        g_max_wg = int(this->ctx->device().get_property(DeviceProperty::MAX_WORK_GROUP_SIZE));
    }

    static bool expect_runs(const C& c, const Spec& s) {
        if (std::holds_alternative<og::Vendor>(c)) return kVendor;
        return oracle_native<T>(c, s.ta, s.tb, s.m, s.n, s.k);
    }
    // The Vendor choice spells the class word `vendor`, which falls back to Auto (with a warning)
    // where no vendor can run, instead of throwing (§5.3).
    static bool vendor_word_falls_back(const C& c) { return std::holds_alternative<og::Vendor>(c) && !kVendor; }

    void run(Problem<T>& p, ComputePrecision prec = ComputePrecision::Default) {
        (void)gemm<B, T>(*this->ctx, p.A(), p.B(), p.Cv(), p.alpha, p.beta, p.s.ta, p.s.tb, prec);
        this->ctx->wait();
    }
    void run_pinned(const C& c, Problem<T>& p) {
        const Pin pin("gemm", c);
        run(p);
    }
    // p's buffer moved one real in (complex<double>: 8 bytes, its alignof), so every base is 8 mod
    // 16 and wide v2 takes its element leg; copied back for expect_gemm.
    void run_pinned_base8(const C& c, Problem<T>& p) {
        static_assert(sizeof(T) == 2 * sizeof(RealOf<T>), "a complex scalar");
        UnifiedVector<RealOf<T>> raw(2 * p.mem.size() + 1, RealOf<T>(0));
        T* base = reinterpret_cast<T*>(raw.data() + 1);
        ASSERT_EQ(reinterpret_cast<std::uintptr_t>(base) % 16u, sizeof(T) / 2);
        std::memcpy(base, p.mem.data(), p.mem.size() * sizeof(T));
        UnifiedVector<T*> pa(p.s.batch, nullptr), pb(p.s.batch, nullptr), pc(p.s.batch, nullptr);
        auto view = [&](const Region& g, UnifiedVector<T*>& ptrs) {
            return MVof<T>(base + g.off, g.rows, g.cols, g.ld, g.stride, p.s.batch, ptrs.data());
        };
        {
            const Pin pin("gemm", c);
            (void)gemm<B, T>(*this->ctx, view(p.a, pa), view(p.b, pb), view(p.c, pc), p.alpha, p.beta, p.s.ta, p.s.tb);
            this->ctx->wait();
        }
        std::memcpy(p.mem.data(), base, p.mem.size() * sizeof(T));
    }
    std::string auto_choice(const Spec& s) {
        const ScopedEnvVar clear("BATCHLAS_GEMM_ROUTE", nullptr);
        auto p = make_problem<T>(s);
        return traced_choice([&] { run(p); });
    }
    // Acceptance is a launch, so `p` is consumed. A launcher refusal after can_run accepted
    // (batchlas::unsupported) is not caught: it is the R3 violation these tests exist for.
    bool pin_accepted(const C& c, Problem<T>& p) {
        const Pin pin("gemm", c);
        try {
            run(p);
            return true;
        } catch (const std::invalid_argument& e) {
            if (std::string(e.what()).find("cannot run this shape") == std::string::npos) throw;
            return false;
        }
    }

    // The arena bytes the next split-k lease on `q` gets, filled with a large finite value: a slice
    // that skips its store (an empty one included) leaves 1e6 in the sum, not fresh-arena zeros or
    // the last call's partials. Returns the capacity; the caller checks the call did not grow it,
    // so the lease came back at the poisoned address.
    static std::size_t poison_partials(Queue& q, const Spec& s, int splits) {
        q.wait();
        const std::int64_t elems = og::splitk_item_bytes<T>(s.m, s.n, splits) / std::int64_t(sizeof(T)) * s.batch + 256;
        auto lease = q.workspace(std::size_t(elems) * sizeof(T));
        const Span<std::byte> w = lease.span();
        MVof<T>(reinterpret_cast<T*>(w.data()), int(elems), 1, int(elems), int(elems), 1).fill(q, T(1.0e6)).wait();
        return q.workspace_capacity();
    }

    // The family's own launcher, called directly; false when it refuses (batchlas::unsupported).
    bool launch(const C& c, Problem<T>& p, std::string* why = nullptr) {
        Queue& q = *this->ctx;
        const MVof<T> A = p.A(), Bm = p.B(), Cm = p.Cv();
        const Transpose ta = p.s.ta, tb = p.s.tb;
        try {
            const bool launched = std::visit(overloaded{
                [&](og::Direct) { (void)sycl_gemm::gemm_direct<T>(q, A, Bm, Cm, p.alpha, p.beta, ta, tb); return true; },
                [&](og::Tiled) { (void)sycl_gemm::gemm_tiled<T>(q, A, Bm, Cm, p.alpha, p.beta, ta, tb); return true; },
                [&](og::Small) { (void)sycl_gemm::gemm_small<T>(q, A, Bm, Cm, p.alpha, p.beta, ta, tb); return true; },
                [&](const og::Reg& r) {
                    if constexpr (!std::is_same_v<T, float>) {
                        if (why) *why = "reg is float only";
                        return false;
                    } else {
                        (void)sycl_gemm::gemm_reg(q, r.m, r.n, r.k, r.u, A, Bm, Cm, p.alpha, p.beta, ta, tb);
                        return true;
                    }
                },
                [&](const og::Wide& w) {
                    (void)sycl_gemm::gemm_wide<T>(q, w.m, w.n, w.k, A, Bm, Cm, p.alpha, p.beta, ta, tb);
                    return true;
                },
                [&](const og::SplitK& sk) {
                    (void)sycl_gemm::gemm_splitk<T>(q, sk.m, sk.n, sk.k, sk.s, A, Bm, Cm, p.alpha, p.beta, ta, tb);
                    return true;
                },
                [&](og::Vendor) {
                    if (why) *why = "the vendor has no SYCL launcher";
                    return false;
                },
            }, c);
            if (launched) q.wait();
            return launched;
        } catch (const batchlas::unsupported& e) {
            if (why) *why = e.what();
            return false;
        }
    }
};

TYPED_TEST_SUITE(GemmCandidates, Types);

// Shapes that are whole multiples of every macro tile (128 x 128 x 64), ragged in all three,
// the small family's buckets (48, 33 x 17 x 5 and its 64 ceiling), the potrf panel (n exactly
// 32), the unit shape, and several 128-tiles in m and n over an odd slab count (300 x 260 x 72
// ragged; 256 x 256 x 72 whole, the 128x128 aligned leg's unstaged epilogue when packed).
const int kShapes[][3] = {{128, 128, 64}, {100, 70, 37}, {48, 48, 48}, {33, 17, 5}, {200, 32, 96}, {1, 1, 1},
                          {300, 260, 72}, {256, 256, 72}};

// §8.1: each candidate on every (ta, tb), both layouts, every shape. Accepted shapes run and
// are right; refused ones throw from the pin. The oracle is the old KernelVariant list.
// can_run's local memory term against the oracle's bytes: every row above 16 KiB agrees with
// RegCfg::slm_bytes, and no other row comes near any device's budget.
TEST(GemmRegSlm, SlmBytesMatchWhatTheDriverWasAskedFor) {
    for (const og::RegCfg& c : og::reg_configs) {
        const auto row = std::find_if(reg_rows().begin(), reg_rows().end(), [&](const RegRow& r) {
            return r.m == c.m && r.n == c.n && r.k == c.k && r.u == c.u;
        });
        ASSERT_NE(row, reg_rows().end()) << c.m << "x" << c.n << "x" << c.k;
        if (row->slm > 0) EXPECT_EQ(c.slm_bytes(), row->slm) << c.m << "x" << c.n << "x" << c.k << " u" << c.u;
        else EXPECT_LE(c.slm_bytes(), 16384) << c.m << "x" << c.n << "x" << c.k << " u" << c.u;
    }
}

// can_run's wide local memory term: literal bytes per row and scalar, so a wide kernel that grows
// its staging must update choice.hh's wide_slm_bytes and this oracle together.
TEST(GemmWideSlm, SlmBytesMatchWhatTheKernelsAllocate) {
    const og::WideCfg& w64 = og::wide_configs[0];
    ASSERT_TRUE(w64.m == 64 && w64.n == 64 && w64.k == 16);
    EXPECT_EQ(og::wide_slm_bytes<float>(w64), 16384);
    EXPECT_EQ(og::wide_slm_bytes<std::complex<float>>(w64), 32768);
    EXPECT_EQ(og::wide_slm_bytes<double>(w64), 16384);
    EXPECT_EQ(og::wide_slm_bytes<std::complex<double>>(w64), 65536);
    const og::WideCfg& w64k8 = og::wide_configs[7];
    ASSERT_TRUE(w64k8.m == 64 && w64k8.n == 64 && w64k8.k == 8);
    EXPECT_EQ(og::wide_slm_bytes<float>(w64k8), 8192);
    EXPECT_EQ(og::wide_slm_bytes<std::complex<float>>(w64k8), 16384);
    EXPECT_EQ(og::wide_slm_bytes<double>(w64k8), 8192);
    EXPECT_EQ(og::wide_slm_bytes<std::complex<double>>(w64k8), 32768);
    // WP-K's transposed tiles: two stages except complex<double> 128x32, 32x128, 64x32 and 32x64.
    const struct { int m, n; std::int64_t f, d, cf, cd; } trans[] = {
        {128, 32, 20480, 40960, 40960, 40960}, {32, 128, 20480, 40960, 40960, 40960},
        {64, 32, 12288, 24576, 24576, 24576},  {32, 64, 12288, 24576, 24576, 24576},
        {32, 32, 8192, 16384, 16384, 32768},   {16, 16, 4096, 8192, 8192, 16384}};
    for (const auto& t : trans) {
        const og::WideCfg c{t.m, t.n, 16, 4, 4, og::kWideAll};
        EXPECT_EQ(og::wide_slm_bytes<float>(c), t.f) << t.m << "x" << t.n;
        EXPECT_EQ(og::wide_slm_bytes<double>(c), t.d) << t.m << "x" << t.n;
        EXPECT_EQ(og::wide_slm_bytes<std::complex<float>>(c), t.cf) << t.m << "x" << t.n;
        EXPECT_EQ(og::wide_slm_bytes<std::complex<double>>(c), t.cd) << t.m << "x" << t.n;
    }
    for (const og::WideCfg& c : og::wide_configs) {
        const WideRow* row = wide_row(og::Wide{c.m, c.n, c.k});
        ASSERT_NE(row, nullptr) << c.m << "x" << c.n;
        EXPECT_EQ(og::wide_slm_bytes<float>(c), wide_slm<float>(*row)) << c.m << "x" << c.n;
        EXPECT_EQ(og::wide_slm_bytes<std::complex<float>>(c), wide_slm<std::complex<float>>(*row)) << c.m << "x" << c.n;
        EXPECT_EQ(og::wide_slm_bytes<double>(c), wide_slm<double>(*row)) << c.m << "x" << c.n;
        EXPECT_EQ(og::wide_slm_bytes<std::complex<double>>(c), wide_slm<std::complex<double>>(*row)) << c.m << "x" << c.n;
    }
    // The M5 Max's selector figures (docs/design/sycl-implementations.md#sycl-impl-metal-launch-limits:
    // 32 KiB less 8 B x 1024 lanes less the 4 KiB reserve, 8 B back per missing lane) refuse only
    // complex<float> v2 among the single-precision 64x64 launches.
    select::Device apple = select::device_from_key("sm_89");
    apple.max_wg = 1024;
    apple.slm_budget = 20480;
    apple.slm_per_item = 8;
    EXPECT_GE(apple.slm_budget_for(w64.threads()), og::wide_slm_bytes<float>(w64));
    EXPECT_LT(apple.slm_budget_for(w64.threads()), og::wide_slm_bytes<std::complex<float>>(w64));
}

TYPED_TEST(GemmCandidates, PinnedCandidatesEveryFormShapeAndLayout) {
    using T = typename TestFixture::T;
    int ran = 0, refused = 0;
    for (const C& c : og::candidates<T>())
        for (Transpose ta : kForms)
            for (Transpose tb : kForms)
                for (const auto& sh : kShapes)
                    for (Layout lay : {Layout::Packed, Layout::Strided}) {
                        Spec s{ta, tb, sh[0], sh[1], sh[2], 3, lay};
                        s.seed = 1000u + 7u * sh[0] + sh[2];
                        auto p = make_problem<T>(s);
                        const std::string what = name(c, s);
                        if (TestFixture::vendor_word_falls_back(c)) {
                            EXPECT_EQ(traced_choice([&] { this->run_pinned(c, p); }), this->auto_choice(s)) << what;
                            expect_gemm(*this->ctx, p, what + " (vendor-free: Auto)");
                            continue;
                        }
                        if (!TestFixture::expect_runs(c, s)) {
                            EXPECT_FALSE(this->pin_accepted(c, p)) << what << " was accepted";
                            ++refused;
                            continue;
                        }
                        ASSERT_TRUE(this->pin_accepted(c, p)) << what << " was refused";
                        expect_gemm(*this->ctx, p, what);
                        if (::testing::Test::HasFatalFailure()) return;
                        ++ran;
                    }
    EXPECT_GT(ran, 0);
    EXPECT_GT(refused, 0);
}

// The (reg, wide) counts of (config, form) pairs the oracle rows list no instantiation for,
// counted from the rows alone (no shape, no device budget), so a loop that skips cases is red.
template <typename T>
std::pair<int, int> missing_instantiations() {
    int reg = 0, wide = 0;
    for (const C& c : og::candidates<T>())
        for (Transpose ta : kForms)
            for (Transpose tb : kForms) {
                if (const auto* r = std::get_if<og::Reg>(&c)) {
                    const RegRow* row = reg_row(*r);
                    reg += !std::is_same_v<T, float> || !row ||
                               std::find(row->forms.begin(), row->forms.end(), reg_form(ta, tb)) == row->forms.end();
                } else if (const auto* w = std::get_if<og::Wide>(&c)) {
                    const WideRow* row = wide_row(*w);
                    wide += !row || row->trace.count(wide_form<T>(ta, tb)) == 0;
                }
            }
    return {reg, wide};
}

// The guard for the 18 register tiles that used to compute NN on a transposed call: every
// (config, form) the oracle has no instantiation for throws from the pin as a can_run refusal
// (not from the launcher), and C is left bit for bit.
TYPED_TEST(GemmCandidates, TransposedPinOnAMissingInstantiationThrows) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    int checked = 0;
    for (const C& c : og::candidates<T>()) {
        if (!std::holds_alternative<og::Reg>(c) && !std::holds_alternative<og::Wide>(c) &&
            !std::holds_alternative<og::SplitK>(c))
            continue;
        for (Transpose ta : kForms)
            for (Transpose tb : kForms) {
                Spec s{ta, tb, 96, 80, 48, 2, Layout::Packed};
                if (oracle_native<T>(c, ta, tb, s.m, s.n, s.k)) continue;
                auto p = make_problem<T>(s);
                const Pin pin("gemm", c);
                try {
                    (void)gemm<B, T>(*this->ctx, p.A(), p.B(), p.Cv(), p.alpha, p.beta, ta, tb);
                    this->ctx->wait();
                    ADD_FAILURE() << name(c, s) << " was accepted";
                } catch (const std::invalid_argument& e) {
                    EXPECT_NE(std::string(e.what()).find("cannot run this shape"), std::string::npos)
                        << name(c, s) << ": " << e.what();
                }
                for (std::size_t e = 0; e < p.mem.size(); ++e)
                    ASSERT_TRUE(same_bits(p.mem[e], p.mem0[e])) << name(c, s) << ": a refused pin wrote element " << e;
                ++checked;
            }
    }
    const auto [reg_missing, wide_missing] = missing_instantiations<T>();
    EXPECT_GE(checked, reg_missing + wide_missing) << "every (config, form) the oracle rows do not list";
    // Pinned apart from the rows: complex 0+5+5+3+3+3+3, real (T folds to C) 0+5+5+4+4+4+4.
    EXPECT_EQ(wide_missing, kCx<T> ? 22 : 26) << "a wide oracle row changed its forms";
    // Float: four NN-only tiles x 8 transposed (ta, tb), 128x64x16's NN, 32x128x16's NT and NC.
    if constexpr (std::is_same_v<T, float>) EXPECT_EQ(reg_missing, 4 * 8 + 1 + 2) << "a reg oracle row changed its forms";
    EXPECT_GT(checked, 0);
}

// The pinned facade runs exactly that family's launcher: bit-identical to the direct call on
// the same input, on a transposed form and the panel layout as well as NN.
TYPED_TEST(GemmCandidates, PinnedRunIsTheLauncherBitForBit) {
    using T = typename TestFixture::T;
    int compared = 0;
    for (const C& c : og::candidates<T>()) {
        if (std::holds_alternative<og::Vendor>(c)) continue;
        for (Transpose ta : kForms)
            for (Transpose tb : {kN, kC}) {
                Spec s{ta, tb, 72, 40, 37, 3, Layout::Strided};
                s.seed = 4242u;
                if (!oracle_native<T>(c, ta, tb, s.m, s.n, s.k)) continue;
                auto pinned = make_problem<T>(s);
                auto direct = make_problem<T>(s);
                this->run_pinned(c, pinned);
                std::string why;
                ASSERT_TRUE(this->launch(c, direct, &why)) << name(c, s) << ": " << why;
                expect_gemm(*this->ctx, pinned, name(c, s));
                for (std::size_t e = 0; e < pinned.mem.size(); ++e)
                    ASSERT_TRUE(same_bits(pinned.mem[e], direct.mem[e]))
                        << name(c, s) << ": the pinned facade did not run this family's launcher; element " << e;
                ++compared;
            }
    }
    EXPECT_GT(compared, 0);
}

// §8.2 (R3): a pin is accepted exactly when the family's own launcher launches, and both agree
// with the oracle, over every form and shapes on both sides of the small ceiling.
TYPED_TEST(GemmCandidates, CanRunEqualsLaunch) {
    using T = typename TestFixture::T;
    int disagreements = 0;
    for (const C& c : og::candidates<T>()) {
        if (std::holds_alternative<og::Vendor>(c)) continue;
        for (Transpose ta : kForms)
            for (Transpose tb : kForms)
                for (auto [m, n, k] : {std::tuple{64, 64, 64}, std::tuple{65, 9, 3}, std::tuple{40, 70, 20}}) {
                    Spec s{ta, tb, m, n, k, 2, Layout::Strided};
                    auto a = make_problem<T>(s);
                    auto b = make_problem<T>(s);
                    const bool pin = this->pin_accepted(c, a);
                    std::string why;
                    const bool run = this->launch(c, b, &why);
                    EXPECT_EQ(pin, run) << name(c, s) << ": can_run says " << pin << ", the launcher "
                                        << (run ? "launches" : "refuses: " + why);
                    EXPECT_EQ(pin, TestFixture::expect_runs(c, s)) << name(c, s) << ": can_run disagrees with the oracle";
                    disagreements += pin != run;
                }
    }
    EXPECT_EQ(disagreements, 0);
}

// A pinned choice launches its own instantiation: the kernel-trace name for every native choice
// on every instantiated form, and the leg of every two-leg config (the NN leg of reg 128x32x32,
// 128x64x32 u4/u2 and wide 64x64; every form of reg 128x128x8 and of wide v2) derived from the layout: aligned
// on a packed tile multiple, predicated on a strided view, so a fast-leg predicate stuck at false
// goes red here. 128x128x8 adds a padded view (aligned) and, per transposed form, a packed shape
// whose stored dims divide the tiles while the effective m or n does not (predicated). Wide v2
// adds every wide_v2_leg_layouts row on a ragged shape (the leg is layout only) and, for
// complex<double>, a padded view at an 8-byte base (its only element-leg layout). A swapped
// launch arm shows here even where both kernels give the same bits. threadsafe: the child
// re-executes the binary with BATCHLAS_KERNEL_TRACE set from the start, so its queue is created with profiling.
TYPED_TEST(GemmCandidates, PinnedChoiceLaunchesItsOwnKernel) {
    using T = typename TestFixture::T;
    ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
    struct Case { C c; Spec s; std::string kernel; bool base8 = false; };
    std::vector<Case> cases;
    for (const C& c : og::candidates<T>()) {
        if (std::holds_alternative<og::Vendor>(c)) continue;
        for (Transpose ta : kForms)
            for (Transpose tb : kForms) {
                const bool small = std::holds_alternative<og::Small>(c);
                for (Layout lay : {Layout::Packed, Layout::Strided}) {
                    Spec s{ta, tb, small ? 48 : 128, small ? 40 : 128, small ? 33 : 64, 2, lay};  // every tile divides
                    if (!oracle_native<T>(c, ta, tb, s.m, s.n, s.k)) continue;
                    cases.push_back({c, s, expected_kernel<T>(c, ta, tb, lay == Layout::Packed)});
                }
                if (const auto* w = std::get_if<og::Wide>(&c);
                    w && w->m == 64 && w->n == 64 && kWideV2<T> && oracle_native<T>(c, ta, tb, 160, 136, 48))
                {
                    for (const auto& [la, lb, vec] : wide_v2_leg_layouts<T>()) {
                        Spec s{ta, tb, 160, 136, 48, 2, la};
                        if (lb != la) s.layout_b = lb;
                        cases.push_back({c, s, expected_kernel<T>(c, ta, tb, vec)});
                    }
                    if (std::is_same_v<T, std::complex<double>>)
                        cases.push_back({c, Spec{ta, tb, 160, 136, 48, 2, Layout::Padded},
                                         expected_kernel<T>(c, ta, tb, true, true), true});
                }
                const auto* r = std::get_if<og::Reg>(&c);
                if (!r || r->m != 128 || r->n != 128 || !oracle_native<T>(c, ta, tb, 128, 128, 64)) continue;
                cases.push_back({c, Spec{ta, tb, 128, 128, 64, 2, Layout::Padded}, expected_kernel<T>(c, ta, tb, true)});
                if (ta != kN || tb != kN)
                    cases.push_back({c, ragged_effective(ta, tb), expected_kernel<T>(c, ta, tb, false)});
            }
    }
    ASSERT_FALSE(cases.empty());
    const std::string path = ::testing::TempDir() + "gemm_ktrace." + std::string(select::dtype_name<T>()) + ".json";
    std::filesystem::remove(path);
    const ScopedEnvVar on("BATCHLAS_KERNEL_TRACE", "1");
    const ScopedEnvVar where("BATCHLAS_KERNEL_TRACE_PATH", path.c_str());
    auto child = [&] {
        for (const auto& k : cases) {
            auto p = make_problem<T>(k.s);
            if constexpr (std::is_same_v<T, std::complex<double>>) {
                if (k.base8) {
                    this->run_pinned_base8(k.c, p);
                    continue;
                }
            }
            this->run_pinned(k.c, p);
        }
        std::exit(0);
    };
    EXPECT_EXIT(child(), ::testing::ExitedWithCode(0), "");
    std::vector<std::string> got;
    std::ifstream in(path);
    for (std::string line; std::getline(in, line);) {
        const auto at = line.find("\"name\":\"gemm_sycl_");
        if (at == std::string::npos) continue;
        const auto b = at + 8;
        got.push_back(line.substr(b, line.find('"', b) - b));
    }
    std::filesystem::remove(path);
    ASSERT_EQ(got.size(), cases.size()) << "one native kernel per pinned call";
    for (std::size_t i = 0; i < cases.size(); ++i)
        EXPECT_EQ(got[i], cases[i].kernel) << name(cases[i].c, cases[i].s);
}

// Sub-views carrying the parent's ld and stride, as potrf's trailing update (A22 -= L21 L21^H, B
// the same memory as A) and geqrf's V^H A22 (CN) issue them, plus CT (and TN for complex), on
// every candidate that runs the form. The whole parent outside A22 stays bit for bit.
TYPED_TEST(GemmCandidates, ParentLdPanelUpdate) {
    using T = typename TestFixture::T;
    int ran = 0;
    // Complex adds TN beside geqrf's CN; for a real scalar the two are one form.
    std::vector<std::pair<Transpose, Transpose>> forms{{kN, kCx<T> ? kC : kT}, {kCx<T> ? kC : kT, kN}, {kC, kT}};
    if constexpr (kCx<T>) forms.emplace_back(kT, kN);
    for (const C& c : og::candidates<T>())
        for (auto [ta, tb] : forms)
            for (auto [mn, k] : {std::pair{96, 32}, std::pair{61, 16}, std::pair{128, 64}}) {
                Spec s{ta, tb, mn, mn, k, 3, Layout::Strided};
                s.seed = 300u + mn;
                auto p = make_panel<T>(s, mn + k + 8);
                if (!TestFixture::expect_runs(c, p.s) || TestFixture::vendor_word_falls_back(c)) continue;
                this->run_pinned(c, p);
                expect_gemm(*this->ctx, p, name(c, p.s) + " (parent-ld panel)");
                if (::testing::Test::HasFatalFailure()) return;
                ++ran;
            }
    EXPECT_GT(ran, 0);
}

// beta = 0 with C full of NaN: BLAS does not read C, so the result must be finite and right.
// Every family, aligned and predicated legs, and the 1x1x1 shape orgqr's blocked apply sends to
// direct (known-defects.md #11: direct, tiled and reg once computed beta * prior unconditionally).
TYPED_TEST(GemmCandidates, BetaZeroDoesNotReadC) {
    using T = typename TestFixture::T;
    int nan_checked = 0;
    const std::pair<Transpose, Transpose> forms[] = {{kN, kN}, {kC, kN}, {kT, kN}, {kC, kT}};
    for (const C& c : og::candidates<T>()) {
        for (auto [ta, tb] : forms)
            for (auto [m, n, k] : {std::tuple{128, 128, 64}, std::tuple{45, 51, 38}, std::tuple{1, 1, 1}}) {
                Spec s{ta, tb, m, n, k, 3, Layout::Strided};
                s.beta_zero = true;
                s.nan_c = true;
                if (!TestFixture::expect_runs(c, s) || TestFixture::vendor_word_falls_back(c)) continue;
                auto p = make_problem<T>(s);
                this->run_pinned(c, p);
                expect_gemm(*this->ctx, p, name(c, s) + " (NaN C)");
                if (::testing::Test::HasFatalFailure()) return;
                ++nan_checked;
            }
    }
    EXPECT_GT(nan_checked, 0);
}

// SLM tiles at a saturating batch: 1024 items repeating 7 distinct problems; each
// representative is right and every other item is bit-identical to its representative.
TYPED_TEST(GemmCandidates, SaturatingBatchIsBitIdenticalToItsRepresentative) {
    using T = typename TestFixture::T;
    constexpr int kBatch = 1024;
    for (const C& c : og::candidates<T>()) {
        if (std::holds_alternative<og::Direct>(c)) continue;
        const bool small = std::holds_alternative<og::Small>(c);
        for (Transpose ta : kForms)
            for (Transpose tb : kForms) {
                Spec s{ta, tb, small ? 45 : 72, small ? 40 : 40, small ? 29 : 37, kBatch, Layout::Strided};
                s.period = 7;
                s.seed = 777u;
                if (!TestFixture::expect_runs(c, s) || TestFixture::vendor_word_falls_back(c)) continue;
                auto p = make_problem<T>(s);
                this->run_pinned(c, p);
                expect_gemm(*this->ctx, p, name(c, s));
                if (::testing::Test::HasFatalFailure()) return;
            }
    }
}

// small's batched kernel in every bucket (NB 8, 16, 32, 64; ragged, k = NB on the first) and all
// nine forms, at batch 1024 repeating 7 problems: complex alpha and beta != 0 with nonzero imaginary
// parts, on odd ld, stride and base (strided) and 16-byte bases with odd ld (aligned-odd-ld). Each
// again at batch 3 with beta = 0 and NaN in C. A transposed A goes through local memory before op(B)
// does, so a dropped conj (C forms only), a swapped stage index (T and C forms of A) or a real-only
// epilogue each go red here. A missing barrier between the two stages races the warps of one slot
// (2 NB lanes), so only NB 32 and 64 can see it, and only when batch 1024 happens to expose it.
// complex<double> above 32 is refused by the pin and leaves C bit for bit.
TYPED_TEST(GemmCandidates, SmallEveryBucketEveryFormAtSaturatingBatch) {
    using T = typename TestFixture::T;
    const int shapes[][3] = {{7, 5, 8}, {13, 16, 11}, {29, 31, 23}, {61, 45, 53}};
    int ran = 0, refused = 0, nan_checked = 0;
    for (const auto& sh : shapes)
        for (Transpose ta : kForms)
            for (Transpose tb : kForms)
                for (Layout lay : {Layout::Strided, Layout::OddLd}) {
                    Spec s{ta, tb, sh[0], sh[1], sh[2], 1024, lay};
                    s.period = 7;
                    s.seed = 5100u + 3u * sh[0] + static_cast<unsigned>(lay);
                    auto p = make_problem<T>(s);
                    const C small{og::Small{}};
                    if (!TestFixture::expect_runs(small, s)) {
                        EXPECT_FALSE(this->pin_accepted(small, p)) << name(small, s) << " was accepted";
                        for (std::size_t e = 0; e < p.mem.size(); ++e)
                            ASSERT_TRUE(same_bits(p.mem[e], p.mem0[e])) << name(small, s) << ": refused but wrote " << e;
                        ++refused;
                        continue;
                    }
                    this->run_pinned(small, p);
                    expect_gemm(*this->ctx, p, name(small, s));
                    if (::testing::Test::HasFatalFailure()) return;
                    ++ran;
                    Spec z = s;
                    z.batch = 3;
                    z.period = 0;
                    z.beta_zero = z.nan_c = true;
                    auto q = make_problem<T>(z);
                    this->run_pinned(small, q);
                    expect_gemm(*this->ctx, q, name(small, z) + " (NaN C)");
                    if (::testing::Test::HasFatalFailure()) return;
                    ++nan_checked;
                }
    // 4 buckets x 9 forms x 2 layouts; complex<double> runs the three up to 32 (a 44 KiB budget
    // holds every bucket of every scalar but its 64).
    EXPECT_EQ(ran + refused, 72);
    EXPECT_GE(ran, sizeof(T) == 16 || g_slm_budget < 33280 ? 54 : 72);
    EXPECT_EQ(nan_checked, ran);
}

// The generic register tiles (2x2, 4x4 and 8x4: every reg row but 128x128) in every form they
// instantiate, at batch 1024 repeating 7 problems, on two layouts: strided (odd ld and base: the
// scalar staging helpers) and padded (ld and stride multiples of 4, neither natural: the packet
// loaders and their b_slot stores, where an ld derived from the rows reads the poison in the gap).
// 136 x 100 x 45 is ragged in every tile and k slab and spans all four 32-column groups of
// b_slot's swizzle; the skinny TN 130 x 33 x 100 is the WP-G target shape. Each form again at
// batch 3 with beta = 0 and NaN in C. Then the aligned NN leg of the rows that have one, packed
// 128 x 128 x 64: b_slot swizzles its store and fragment read on TileN 64. A transposed operand
// staged with the wrong lane map, or a swizzle the store and the read disagree on, is wrong; a
// staging race differs across items.
TYPED_TEST(GemmCandidates, RegScalarStagingEveryFormAtSaturatingBatch) {
    using T = typename TestFixture::T;
    if constexpr (!std::is_same_v<T, float>) {
        GTEST_SKIP() << "reg is float only";
    } else {
        struct Shape { Transpose ta, tb; int m, n, k; Layout layout; };
        std::vector<Shape> shapes;
        for (Layout lay : {Layout::Strided, Layout::Padded}) {
            for (Transpose ta : {kN, kT})
                for (Transpose tb : {kN, kT}) shapes.push_back({ta, tb, 136, 100, 45, lay});
            shapes.push_back({kT, kN, 130, 33, 100, lay});
        }
        int ran = 0, tn_skinny = 0, nan_checked = 0, aligned = 0;
        auto check = [&](const C& c, Spec s) {
            if (!TestFixture::expect_runs(c, s)) return false;
            auto p = make_problem<T>(s);
            this->run_pinned(c, p);
            expect_gemm(*this->ctx, p, name(c, s) + (s.nan_c ? " (NaN C)" : ""));
            return true;
        };
        for (const og::RegCfg& cfg : og::reg_configs) {
            if (cfg.m == 128 && cfg.n == 128) continue;  // its own kernel and test
            const C c{og::Reg{cfg.m, cfg.n, cfg.k, cfg.u}};
            for (const Shape& sh : shapes)
                for (bool nan : {false, true}) {
                    if (nan && sh.n == 33) continue;
                    Spec s{sh.ta, sh.tb, sh.m, sh.n, sh.k, nan ? 3 : 1024, sh.layout};
                    s.period = nan ? 0 : 7;
                    s.beta_zero = s.nan_c = nan;
                    s.seed = 9500u + sh.n;
                    if (!check(c, s)) continue;
                    if (::testing::Test::HasFatalFailure()) return;
                    ++(nan ? nan_checked : ran);
                    tn_skinny += sh.n == 33;
                }
            if (!cfg.aligned_leg) continue;
            Spec s{kN, kN, 128, 128, 64, 1024, Layout::Packed};
            s.period = 7;
            s.seed = 9600u + cfg.u;
            aligned += check(c, s);
            if (::testing::Test::HasFatalFailure()) return;
        }
        // Per layout 22 + 5 (forms: NN 8, NT 4, TN 5, TT 5) and 22 NaN; the aligned rows are
        // 128x32x32 and 128x64x32 u4 and u2. Fewer only where a row is over the budget.
        EXPECT_GE(ran, 40);
        EXPECT_GE(tn_skinny, 8);
        EXPECT_GE(nan_checked, 32);
        EXPECT_GE(aligned, 2);
    }
}

// The 128x128 kernel in every real form (CC folds to TT) at batch 1024, 7 distinct problems: its
// four legs (packed 128 x 256 x 24 aligned + staged over two n tiles, packed 128 x 128 x 72
// aligned + direct epilogue over an odd slab count, strided 200 x 150 x 37 predicated + staged and
// 200 x 150 x 72 predicated + direct over 2 x 2 ragged tiles), each item bit-identical to its
// representative. A per-form loader that reads the
// wrong lane map, or a missing barrier between slabs, differs across items or is wrong.
TYPED_TEST(GemmCandidates, Reg128x128EveryFormEveryLegAtSaturatingBatch) {
    using T = typename TestFixture::T;
    if constexpr (!std::is_same_v<T, float>) {
        GTEST_SKIP() << "reg is float only";
    } else {
        const C c{og::Reg{128, 128, 8, 1}};
        const RegRow* row = reg_row(std::get<og::Reg>(c));
        ASSERT_NE(row, nullptr);
        if (!reg_fits(*row)) GTEST_SKIP() << "128x128 is above this device's local memory budget";
        const std::pair<Transpose, Transpose> forms[] = {{kN, kN}, {kN, kT}, {kT, kN}, {kT, kT}, {kC, kC}};
        struct Leg { int m, n, k; Layout layout; };
        const Leg legs[] = {{128, 256, 24, Layout::Packed}, {128, 128, 72, Layout::Packed},
                            {200, 150, 37, Layout::Strided}, {200, 150, 72, Layout::Strided}};
        int ran = 0;
        for (auto [ta, tb] : forms)
            for (const Leg& leg : legs) {
                Spec s{ta, tb, leg.m, leg.n, leg.k, 1024, leg.layout};
                s.period = 7;
                s.seed = 9100u + leg.k;
                ASSERT_TRUE(TestFixture::expect_runs(c, s)) << name(c, s);
                auto p = make_problem<T>(s);
                this->run_pinned(c, p);
                expect_gemm(*this->ctx, p, name(c, s));
                if (::testing::Test::HasFatalFailure()) return;
                ++ran;
            }
        EXPECT_EQ(ran, 20);
    }
}

// WP-I, the predicated leg's interior loop copy (odd bases, so every leg below is predicated whatever
// its ld). 256 x 256: all four groups interior, k = 72 (nine full slabs, direct epilogue), k = 37 (the
// k tail after full slabs) and k = 5 (one partial slab, the prologue alone). 300 x 200: interior,
// m-edge, n-edge and corner groups in one launch, k = 45 at ld = m + 1 and k = 72 strided. Batch 1024,
// each item bit-identical to its representative. The k-tail legs poison with NaN: a finite poison times
// the other operand's zero is a bit-exact -0, so one operand's dropped k bound would stay green. A wrong
// element in the interior copy turns the 256 legs red; either k-tail bound dropped, the k = 37, 5 and 45
// legs. An interior test that admits an edge group only over-reads (rows and columns past m and n feed
// only discarded accumulators), so no numeric test sees it.
// evidence: docs/perf/gemm.md#gemm-wp-i-an-unpredicated-interior-loop-on-the-128x128-predicated-leg
TYPED_TEST(GemmCandidates, Reg128x128PredicatedInteriorAndEdgeGroupsAtSaturatingBatch) {
    using T = typename TestFixture::T;
    if constexpr (!std::is_same_v<T, float>) {
        GTEST_SKIP() << "reg is float only";
    } else {
        const std::pair<Transpose, Transpose> forms[] = {{kN, kN}, {kN, kT}, {kT, kN}, {kT, kT}, {kC, kC}};
        struct Leg { int m, n, k; Layout layout; bool nan; };
        const Leg legs[] = {{256, 256, 72, Layout::Strided, false}, {256, 256, 37, Layout::Strided, true},
                            {256, 256, 5, Layout::Strided, true}, {300, 200, 45, Layout::Gap1, true},
                            {300, 200, 72, Layout::Strided, false}};
        int ran = 0, rows = 0;
        // u = 4 is WP-N2's zero-filled copy_async pipeline: the same legs, one more prologue depth.
        for (int u : {1, 4}) {
            const C c{og::Reg{128, 128, 8, u}};
            const RegRow* row = reg_row(std::get<og::Reg>(c));
            ASSERT_NE(row, nullptr);
            if (!reg_fits(*row)) continue;
            ++rows;
            for (auto [ta, tb] : forms)
                for (const Leg& leg : legs) {
                    Spec s{ta, tb, leg.m, leg.n, leg.k, 1024, leg.layout};
                    s.period = 7;
                    s.nan_poison = leg.nan;
                    s.seed = 9200u + leg.m + leg.k;
                    ASSERT_TRUE(TestFixture::expect_runs(c, s)) << name(c, s);
                    auto p = make_problem<T>(s);
                    this->run_pinned(c, p);
                    expect_gemm(*this->ctx, p, name(c, s));
                    if (::testing::Test::HasFatalFailure()) return;
                    ++ran;
                }
        }
        if (rows == 0) GTEST_SKIP() << "128x128 is above this device's local memory budget";
        EXPECT_EQ(ran, 25 * rows);
    }
}

// §8.3 on the 128x128 aligned leg in every real form: padded views (ld and stride multiples of 4,
// neither natural), so a loader or epilogue that derives ld or stride from m, n or k reads the
// poison in the gap. Both epilogues (k = 24 staged, k = 72 direct) at batch 1024, then again with
// beta = 0 and NaN in C. Then the ragged-effective packed shapes (PinnedChoiceLaunchesItsOwnKernel
// pins their leg): an aligned leg there writes outside C.
TYPED_TEST(GemmCandidates, Reg128x128AlignedLegReadsLdStrideAndEffectiveDims) {
    using T = typename TestFixture::T;
    if constexpr (!std::is_same_v<T, float>) {
        GTEST_SKIP() << "reg is float only";
    } else {
        const C c{og::Reg{128, 128, 8, 1}};
        const RegRow* row = reg_row(std::get<og::Reg>(c));
        ASSERT_NE(row, nullptr);
        if (!reg_fits(*row)) GTEST_SKIP() << "128x128 is above this device's local memory budget";
        const std::pair<Transpose, Transpose> forms[] = {{kN, kN}, {kN, kT}, {kT, kN}, {kT, kT}, {kC, kC}};
        int ran = 0;
        auto check = [&](Spec s) {
            ASSERT_TRUE(TestFixture::expect_runs(c, s)) << name(c, s);
            auto p = make_problem<T>(s);
            this->run_pinned(c, p);
            expect_gemm(*this->ctx, p, name(c, s) + (s.nan_c ? " (NaN C)" : ""));
            ++ran;
        };
        for (auto [ta, tb] : forms) {
            for (int k : {24, 72})
                for (bool nan : {false, true}) {
                    Spec s{ta, tb, 128, 256, k, nan ? 3 : 1024, Layout::Padded};
                    s.period = nan ? 0 : 7;
                    s.beta_zero = s.nan_c = nan;
                    s.seed = 9300u + k;
                    check(s);
                    if (::testing::Test::HasFatalFailure()) return;
                }
            if (ta == kN && tb == kN) continue;
            check(ragged_effective(ta, tb));
            if (::testing::Test::HasFatalFailure()) return;
        }
        EXPECT_EQ(ran, 24);
    }
}

// WP-N2, the four-stage zero-filled copy_async pipeline of reg 128x128x8 u4 (predicated leg) in every
// real form, batch 1024 repeating 7 problems, each item bit-identical to its representative. Every view
// is predicated: odd bases or odd ld, and one 16-byte-aligned view whose ragged m, n alone refuse the
// aligned leg (an aligned-base shortcut in the pipeline is caught there). Slab counts 1, 2, 3, 4, 5, 8, 9 and 17 straddle the prologue
// depth (3 slabs), the unroll by 4 and the staged/direct epilogue edge (k = 64 / 69); k = 5, 37 and 69
// end in a partial slab and poison with NaN (a dropped k bound times the other operand's zero is a
// silent -0 with finite poison); 300 x 200 adds m-edge, n-edge and corner groups. Then beta = 0 with
// NaN in C. A wait on the wrong group or a refill of a buffer still to be read differs across items or
// is wrong from 2 or 4 slabs on; a lost pointer bump, from 2 slabs in its operand's forms.
// evidence: docs/perf/gemm.md#gemm-wp-n2-the-u4-stack-frame-and-zero-filled-predicated-copies
TYPED_TEST(GemmCandidates, Reg128x128CopyAsyncPipelineEverySlabCountAtSaturatingBatch) {
    using T = typename TestFixture::T;
    if constexpr (!std::is_same_v<T, float>) {
        GTEST_SKIP() << "reg is float only";
    } else {
        const C c{og::Reg{128, 128, 8, 4}};
        const RegRow* row = reg_row(std::get<og::Reg>(c));
        ASSERT_NE(row, nullptr);
        if (!reg_fits(*row)) GTEST_SKIP() << "128x128 u4 is above this device's local memory budget";
        const std::pair<Transpose, Transpose> forms[] = {{kN, kN}, {kN, kT}, {kT, kN}, {kT, kT}, {kC, kC}};
        struct Leg { int m, n, k; Layout layout; bool nan; };
        const Leg legs[] = {{256, 256, 5, Layout::Strided, true},   {128, 256, 8, Layout::Strided, false},
                            {256, 128, 16, Layout::OddLd, false},   {128, 256, 24, Layout::Gap1, false},
                            {256, 128, 32, Layout::Strided, false}, {300, 200, 37, Layout::Gap1, true},
                            {128, 256, 40, Layout::OddLd, false},   {256, 128, 64, Layout::Strided, false},
                            {128, 256, 69, Layout::Gap1, true},     {300, 200, 72, Layout::Strided, false},
                            {128, 128, 136, Layout::OddLd, false},  {300, 200, 40, Layout::Padded, false}};
        const Leg nan_legs[] = {{128, 256, 24, Layout::Gap1, false}, {256, 128, 72, Layout::Strided, false}};
        int ran = 0;
        auto check = [&](const Leg& leg, Transpose ta, Transpose tb, bool nan_c) {
            Spec s{ta, tb, leg.m, leg.n, leg.k, nan_c ? 3 : 1024, leg.layout};
            s.period = nan_c ? 0 : 7;
            s.beta_zero = s.nan_c = nan_c;
            s.nan_poison = leg.nan;
            s.seed = 9700u + leg.m + leg.k;
            ASSERT_TRUE(TestFixture::expect_runs(c, s)) << name(c, s);
            auto p = make_problem<T>(s);
            this->run_pinned(c, p);
            expect_gemm(*this->ctx, p, name(c, s) + (nan_c ? " (NaN C)" : ""));
            ++ran;
        };
        for (auto [ta, tb] : forms) {
            for (const Leg& leg : legs) {
                check(leg, ta, tb, false);
                if (::testing::Test::HasFatalFailure()) return;
            }
            for (const Leg& leg : nan_legs) {
                check(leg, ta, tb, true);
                if (::testing::Test::HasFatalFailure()) return;
            }
        }
        EXPECT_EQ(ran, 70);
    }
}

// WP-B, the C epilogue of every reg and wide tile at a saturating batch, 7 distinct problems, beta != 0
// (C is read). 256 x 256 x 16 strided: every group interior, odd ldc, so the staged store (wide at
// 16-byte scalars stores direct, already coalesced). 200 x 136 x 8 ragged on four layouts: ld = m + 1
// (the tuner's strided cell; the poison element under each column is read or written by an
// off-by-one predicate or an ldc derived from m); padded (16-byte columns: interior groups take
// the band store, edge groups the staged one); 16-byte bases at an odd ld (only the ldc term of the
// band-store test refuses it); and item-skewed (ldc a multiple of 4, every item's base at another
// offset, so only the per-item base term refuses; period 8 keeps an item and its representative on
// one path). Then beta = 0 and NaN in C on the staged and band layouts. A missing barrier around
// the stage, or a stage index the write and read-back disagree on, differs across items or is
// wrong; a band store on an edge group writes outside C, and one at a misaligned base faults.
// evidence: docs/perf/gemm.md#gemm-wp-b-coalesced-c-epilogue-on-the-predicated-legs
TYPED_TEST(GemmCandidates, CoalescedCEpilogueEveryTileAtSaturatingBatch) {
    using T = typename TestFixture::T;
    constexpr bool kWide16 = sizeof(T) == 16;
    struct Shape { int m, n, k; Layout layout; int batch; bool all_forms; };
    std::vector<Shape> shapes;
    if (!kWide16) shapes.push_back({256, 256, 16, Layout::Strided, 1024, false});
    shapes.push_back({200, 136, 8, Layout::Gap1, kWide16 ? 64 : 1024, true});
    shapes.push_back({200, 136, 8, Layout::Padded, kWide16 ? 64 : 128, true});
    shapes.push_back({200, 136, 8, Layout::OddLd, kWide16 ? 64 : 128, true});
    shapes.push_back({200, 136, 8, Layout::ItemSkew, kWide16 ? 64 : 1024, true});
    const std::pair<Transpose, Transpose> forms[] = {{kN, kN}, {kC, kN}, {kN, kC}, {kC, kC}};
    int reg = 0, reg_expected = 0, wide = 0, nan_checked = 0;
    for (const C& c : og::candidates<T>()) {
        const bool is_reg = std::holds_alternative<og::Reg>(c);
        if (!is_reg && !std::holds_alternative<og::Wide>(c)) continue;
        const RegRow* row = is_reg ? reg_row(std::get<og::Reg>(c)) : nullptr;
        if (is_reg && std::is_same_v<T, float>) {
            ASSERT_NE(row, nullptr) << select::to_string(c);
            // Only a row over 32 KiB (128x128 u4, 128x64x32) may be refused, on a device budgeting less.
            if (!reg_fits(*row)) EXPECT_TRUE(row->slm > 32768 && g_slm_budget < row->slm) << select::to_string(c);
        }
        for (const Shape& sh : shapes)
            for (auto [ta, tb] : forms) {
                if (!sh.all_forms && tb != kN) continue;
                Spec s{ta, tb, sh.m, sh.n, sh.k, sh.batch, sh.layout};
                s.period = sh.layout == Layout::ItemSkew ? 8 : 7;
                s.seed = 9700u + sh.m + static_cast<unsigned>(sh.layout);
                if (row && std::is_same_v<T, float> && reg_fits(*row))
                    reg_expected += std::count(row->forms.begin(), row->forms.end(), reg_form(ta, tb)) > 0;
                if (!TestFixture::expect_runs(c, s)) continue;
                auto p = make_problem<T>(s);
                this->run_pinned(c, p);
                expect_gemm(*this->ctx, p, name(c, s));
                if (::testing::Test::HasFatalFailure()) return;
                ++(is_reg ? reg : wide);
            }
        for (Layout lay : {Layout::Gap1, Layout::Padded, Layout::ItemSkew})
            for (auto [ta, tb] : forms) {
                Spec s{ta, tb, 200, 136, 8, 3, lay};
                s.beta_zero = s.nan_c = true;
                if (!TestFixture::expect_runs(c, s)) continue;
                auto p = make_problem<T>(s);
                this->run_pinned(c, p);
                expect_gemm(*this->ctx, p, name(c, s) + " (NaN C)");
                if (::testing::Test::HasFatalFailure()) return;
                ++nan_checked;
            }
    }
    // Float: reg 17 + 4 x 30 (a 48 KiB cap drops 128x64x32 u4 and u2, NN only: 2 x 5; under 33,792 B
    // 128x128 u4 too: 2 + 4 x 4),
    // wide 14 + 4 x 24; double and complex<float> wide 14 + 4 x 24; complex<double> wide 4 x 24.
    if constexpr (std::is_same_v<T, float>) {
        EXPECT_EQ(reg, reg_expected);
        EXPECT_GE(reg, g_slm_budget < 33792 ? 109 : g_slm_budget < 49920 ? 127 : 137);
    }
    EXPECT_GE(wide, kWide16 ? 96 : 110);
    EXPECT_GE(nan_checked, 3 * 24);
}

// WP-E, the skinny wide tiles 64x32 and 32x64 (128 lanes) in every form they instantiate (complex
// six, real five: CT folds to the uncompiled CC). The NN 64x32x256 b128 target and its mirror at
// batch 1024 strided (staged epilogue) and 128 padded (band store), 7 distinct problems; then m and
// n straddling the 32 and 64 tile edges both ways, ragged k on the +1 side, on both layouts. A
// band separation or tile count derived from the other extent is wrong only on a non-square tile;
// an off-by-one edge predicate writes outside C at 33 or 65, or drops the last row at 32 or 64.
// 16-byte scalars keep batch 1024 at k = 64 (four staged blocks), a quarter of the memory.
TYPED_TEST(GemmCandidates, SkinnyWideTilesEveryFormAtSaturatingBatchAndTileEdges) {
    using T = typename TestFixture::T;
    struct Shape { int m, n, k; Layout layout; int batch; };
    const int ks = sizeof(T) == 16 ? 64 : 256;
    std::vector<Shape> shapes{{64, 32, ks, Layout::Strided, 1024}, {32, 64, ks, Layout::Strided, 1024},
                              {64, 32, 256, Layout::Padded, 128},   {32, 64, 256, Layout::Padded, 128}};
    const std::pair<int, int> edges[] = {{32, 64}, {33, 65}, {64, 32}, {65, 33}, {32, 32}, {33, 33}, {64, 64}, {65, 65}};
    for (Layout lay : {Layout::Strided, Layout::Padded})
        for (auto [m, n] : edges) shapes.push_back({m, n, m % 2 ? 17 : 48, lay, 3});
    int ran = 0;
    for (const C c : {C{og::Wide{64, 32, 16}}, C{og::Wide{32, 64, 16}}})
        for (const Shape& sh : shapes)
            for (Transpose ta : kForms)
                for (Transpose tb : kForms) {
                    Spec s{ta, tb, sh.m, sh.n, sh.k, sh.batch, sh.layout};
                    s.period = sh.batch > 3 ? 7 : 0;
                    s.seed = 9800u + sh.m + 3u * sh.n;
                    if (!TestFixture::expect_runs(c, s)) continue;
                    auto p = make_problem<T>(s);
                    this->run_pinned(c, p);
                    expect_gemm(*this->ctx, p, name(c, s));
                    if (::testing::Test::HasFatalFailure()) return;
                    ++ran;
                }
    EXPECT_EQ(ran, 2 * int(shapes.size()) * (kCx<T> ? 6 : 5));
}

// WP-L, split-k on every host tile, s and form. k = 300 slices unevenly (kc = 160, 80, 48: the last
// slice is partial, and at s = 8 the eighth is empty and must store zeros; 16-byte scalars k = 136,
// kc = 80, 48, 32, empty slices at s = 4 and 8); k = 40 leaves slices
// past k at s = 4 and 8; k = 1 only slice 0. m and n one past the host tile (edge groups in both),
// at batch 1024 repeating 7 problems on ld = m + 1 with complex alpha and beta != 0: a slice that
// reads past its k range double-counts valid data, a lost slice drops a k range, a reduction that
// applies beta per slice or reads another item's partials is wrong, and items differ if the slice
// index is derived from anything but the group. Padded views at k = 40, then beta = 0 with NaN in
// C (the reduction must not read C) and a second identical call bit for bit (fixed summation order).
// evidence: docs/perf/gemm.md#gemm-wp-l-global-split-k-over-the-transposed-wide-tiles
TYPED_TEST(GemmCandidates, SplitKEveryHostEveryFormAtSaturatingBatchAndSliceEdges) {
    using T = typename TestFixture::T;
    int ran = 0, nan_checked = 0, repeated = 0;
    const int deep = sizeof(T) == 16 ? 136 : 300;
    for (const C& c : og::candidates<T>()) {
        const auto* sk = std::get_if<og::SplitK>(&c);
        if (!sk) continue;
        struct Shape { int m, n, k; Layout layout; int batch; };
        const Shape shapes[] = {{sk->m + 1, sk->n + 1, deep, Layout::Gap1, 1024},
                                {sk->m, sk->n, 40, Layout::Padded, 3},
                                {sk->m + 1, sk->n - 1, 1, Layout::Strided, 3}};
        for (Transpose ta : kForms)
            for (Transpose tb : kForms) {
                if (!TestFixture::expect_runs(c, Spec{ta, tb, sk->m, sk->n, 40})) continue;
                for (const Shape& sh : shapes) {
                    Spec s{ta, tb, sh.m, sh.n, sh.k, sh.batch, sh.layout};
                    s.period = sh.batch > 3 ? 7 : 0;
                    s.seed = 9500u + sh.m + 3u * sh.n + 11u * sh.k;
                    auto p = make_problem<T>(s);
                    const std::size_t cap = TestFixture::poison_partials(*this->ctx, s, sk->s);
                    this->run_pinned(c, p);
                    ASSERT_EQ(this->ctx->workspace_capacity(), cap) << name(c, s) << ": the lease missed the poison";
                    expect_gemm(*this->ctx, p, name(c, s));
                    if (::testing::Test::HasFatalFailure()) return;
                    ++ran;
                }
                Spec s{ta, tb, sk->m + 1, sk->n + 1, deep, 3, Layout::Gap1};
                s.beta_zero = s.nan_c = true;
                auto p = make_problem<T>(s);
                (void)TestFixture::poison_partials(*this->ctx, s, sk->s);
                this->run_pinned(c, p);
                expect_gemm(*this->ctx, p, name(c, s) + " (NaN C)");
                if (::testing::Test::HasFatalFailure()) return;
                ++nan_checked;
                auto again = make_problem<T>(s);
                this->run_pinned(c, again);
                for (std::size_t e = 0; e < p.mem.size(); ++e)
                    ASSERT_TRUE(same_bits(p.mem[e], again.mem[e])) << name(c, s) << ": a repeated call differs at " << e;
                ++repeated;
            }
    }
    // 6 hosts x 3 slice counts over the (ta, tb) pairs: 128x32 and 32x128 serve four, the four
    // small tiles six (complex) or five (real: T folds to C, so no TT, TC, CT, CC)
    const int forms = kCx<T> ? 3 * (2 * 4 + 4 * 6) : 3 * (2 * 4 + 4 * 5);
    EXPECT_EQ(ran, 3 * forms);
    EXPECT_EQ(nan_checked, forms);
    EXPECT_EQ(repeated, forms);
}

// An out-of-order Queue: the reduction must wait for the slices (gemm_splitk's out->wait()). Poisoned
// partials at batch 1024, s = 8: a reduction that overtakes the slices sums 1e6. A race, so a deleted
// wait is likely, not certainly, red; three repetitions per scalar.
TYPED_TEST(GemmCandidates, SplitKOnAnOutOfOrderQueueReducesAfterTheSlices) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    Queue ooo(*this->ctx, false);
    const C c{og::SplitK{32, 32, 16, 8}};
    Spec s{kN, kN, 33, 33, sizeof(T) == 16 ? 136 : 300, 1024, Layout::Gap1};
    s.period = 7;
    s.seed = 9900u;
    if (!TestFixture::expect_runs(c, s)) GTEST_SKIP() << name(c, s) << " cannot run here";
    for (int rep = 0; rep < 3; ++rep) {
        auto p = make_problem<T>(s);
        const std::size_t cap = TestFixture::poison_partials(ooo, s, 8);
        {
            const Pin pin("gemm", c);
            (void)gemm<B, T>(ooo, p.A(), p.B(), p.Cv(), p.alpha, p.beta, s.ta, s.tb, ComputePrecision::Default);
            ooo.wait();
        }
        ASSERT_EQ(ooo.workspace_capacity(), cap) << name(c, s) << ": the lease missed the poison";
        expect_gemm(*this->ctx, p, name(c, s) + " (out-of-order Queue)");
        if (::testing::Test::HasFatalFailure()) return;
    }
}

// split-k puts batch * s groups in grid z: a pin is a launch at batch * s <= 65535 and a refusal
// one item past it (where the device caps the grid), with C left bit for bit.
TYPED_TEST(GemmCandidates, SplitKGridCeilingIsBatchTimesSlices) {
    using T = typename TestFixture::T;
    const select::Device& dev = select::device_of<TestFixture::B>(*this->ctx);
    const int ceiling = int(test_utils::kCudaGridYZ);
    if (dev.max_groups_yz > ceiling) GTEST_SKIP() << dev.key << " has no grid z cap";
    for (int sl : {2, 8}) {
        const C c{og::SplitK{16, 16, 16, sl}};
        const Spec at{kN, kN, 2, 2, 20, ceiling / sl, Layout::Packed};
        Spec past = at;
        past.batch += 1;
        auto p = make_problem<T>(at);
        ASSERT_TRUE(this->pin_accepted(c, p)) << name(c, at);
        expect_gemm(*this->ctx, p, name(c, at));
        auto q = make_problem<T>(past);
        EXPECT_FALSE(this->pin_accepted(c, q)) << name(c, past);
        for (std::size_t e = 0; e < q.mem.size(); ++e)
            ASSERT_TRUE(same_bits(q.mem[e], q.mem0[e])) << name(c, past) << ": a refused pin wrote element " << e;
    }
}


// WP-K, the pipelined transposed-wide staging (two stages, register prefetch, 16-byte granules)
// on 128x32, 32x128, 64x32, 32x64, 32x32 and 16x16, every form each instantiates. Interior
// 2TM x 2TN at k = 48 padded (the 16-byte leg, three slabs: the last stage read is 0) and k = 64
// at ld = m + 1 (element leg, four slabs, staged epilogue) at a saturating batch, 7 distinct
// problems, each item bit-identical to its representative. Then TM+1 x TN+1 (edge groups) at
// k = 16 (one slab: no prefetch, no flip), 17 and 37 (a predicated k tail) on both legs with NaN
// in the gaps, so one dropped k bound is NaN, not a silent -0. Then the v2 leg layouts at
// 2TM x 2TN x 48. A stage flip the store and compute disagree on is wrong only at k > 16; a
// swizzle the store and fragment read disagree on is wrong in the forms staging that operand
// along k (A transposed, B not).
// evidence: docs/perf/gemm.md#gemm-wp-k-pipelined-staging-for-the-transposed-wide-tiles
TYPED_TEST(GemmCandidates, WideTransposedPipelineEveryTileFormAndLegAtSaturatingBatch) {
    using T = typename TestFixture::T;
    struct Shape { int m, n, k; Layout layout; int batch; bool nan; };
    const int big = sizeof(T) == 16 ? 256 : 1024;
    int ran = 0, expected = 0, leg_ran = 0;
    for (const og::Wide w : {og::Wide{128, 32, 16}, og::Wide{32, 128, 16}, og::Wide{64, 32, 16},
                             og::Wide{32, 64, 16}, og::Wide{32, 32, 16}, og::Wide{16, 16, 16}}) {
        const C c{w};
        if (!wide_fits<T>(*wide_row(w))) continue;
        const std::vector<Shape> shapes{{2 * w.m, 2 * w.n, 48, Layout::Padded, big, false},
                                        {2 * w.m, 2 * w.n, 64, Layout::Gap1, big, false},
                                        {w.m + 1, w.n + 1, 16, Layout::Padded, 3, true},
                                        {w.m + 1, w.n + 1, 17, Layout::Padded, 3, true},
                                        {w.m + 1, w.n + 1, 37, Layout::Gap1, 3, true},
                                        {w.m + 1, w.n + 1, 37, Layout::Padded, 3, true}};
        const bool skinny = w.m == 128 || w.n == 128;
        expected += int(shapes.size()) * (skinny ? 4 : (kCx<T> ? 6 : 5));
        for (Transpose ta : kForms)
            for (Transpose tb : kForms) {
                Spec probe{ta, tb, 2 * w.m, 2 * w.n, 48, 3, Layout::Padded};
                if (!TestFixture::expect_runs(c, probe)) continue;
                for (const Shape& sh : shapes) {
                    Spec s{ta, tb, sh.m, sh.n, sh.k, sh.batch, sh.layout};
                    s.period = sh.batch > 3 ? 7 : 0;
                    s.nan_poison = sh.nan;
                    s.seed = 10100u + sh.m + 3u * sh.n + 7u * sh.k;
                    ASSERT_TRUE(TestFixture::expect_runs(c, s)) << name(c, s);
                    auto p = make_problem<T>(s);
                    this->run_pinned(c, p);
                    expect_gemm(*this->ctx, p, name(c, s));
                    if (::testing::Test::HasFatalFailure()) return;
                    ++ran;
                }
                for (const auto& [la, lb, vec] : wide_v2_leg_layouts<T>()) {
                    static_cast<void>(vec);
                    Spec s{ta, tb, 2 * w.m, 2 * w.n, 48, 4, la};
                    if (lb != la) s.layout_b = lb;
                    s.seed = 10300u + 13u * unsigned(la) + unsigned(lb);
                    auto p = make_problem<T>(s);
                    this->run_pinned(c, p);
                    expect_gemm(*this->ctx, p, name(c, s) + " (leg layout)");
                    if (::testing::Test::HasFatalFailure()) return;
                    ++leg_ran;
                }
            }
    }
    EXPECT_EQ(ran, expected);
    EXPECT_GT(leg_ran, 0);
}


// WP-D, wide v2 (64x64x16: float, complex<float>, since WP-J complex<double>) in all 9 forms at
// batch 1024, 7 distinct problems, each item bit-identical to its representative. Ragged 100 x 70 x 37 at ld = m + 1
// (element loads, edge groups predicated, a k tail, 3 slabs: the odd-count barrier before the
// staged epilogue); ragged 200 x 136 x 40 padded (the 16-byte leg: interior groups unpredicated,
// ld and stride not derivable from the shape); packed 128 x 128 x 64 (every group interior, 4
// slabs, the band store); 130 x 70 x 64 at ld = m + 1 (an even slab count into the staged
// epilogue). Then each form with beta = 0 and NaN in C. A conjugation sign dropped on A or on B
// is wrong exactly in the forms that conjugate that operand; a swizzle the store and the read
// disagree on, or a k-contiguous lane map that puts two lanes on one granule, is wrong in every
// form staging that operand layout (a bit swap is a bijection and stays right); a missing barrier
// between the two buffers differs across items. Then the leg layouts (wide_v2_leg_layouts).
// evidence: docs/perf/gemm.md#gemm-wide-v2-one-double-buffered-64x64-kernel-for-every-form
TYPED_TEST(GemmCandidates, WideV2EveryFormAtSaturatingBatch) {
    using T = typename TestFixture::T;
    if constexpr (!kWideV2<T>) {
        GTEST_SKIP() << "wide v2 serves float, complex<float> and complex<double>";
    } else {
        const C c{og::Wide{64, 64, 16}};
        if (!wide_fits<T>(*wide_row(std::get<og::Wide>(c)))) GTEST_SKIP() << "above this device's local memory budget";
        struct Shape { int m, n, k; Layout layout; };
        const Shape shapes[] = {{100, 70, 37, Layout::Gap1}, {200, 136, 40, Layout::Padded},
                                {128, 128, 64, Layout::Packed}, {130, 70, 64, Layout::Gap1}};
        int ran = 0, nan_checked = 0, leg_ran = 0;
        for (Transpose ta : kForms)
            for (Transpose tb : kForms) {
                for (const Shape& sh : shapes) {
                    Spec s{ta, tb, sh.m, sh.n, sh.k, 1024, sh.layout};
                    s.period = 7;
                    s.seed = 9900u + sh.m + 5u * sh.k;
                    ASSERT_TRUE(TestFixture::expect_runs(c, s)) << name(c, s);
                    auto p = make_problem<T>(s);
                    this->run_pinned(c, p);
                    expect_gemm(*this->ctx, p, name(c, s));
                    if (::testing::Test::HasFatalFailure()) return;
                    ++ran;
                }
                // Every leg layout at 160 x 136 x 48: four interior groups over three full slabs,
                // the only place the 16-byte load runs, so a leg term dropped from
                // wide64_v2_vec_loads issues misaligned 16-byte loads here.
                for (const auto& [la, lb, vec] : wide_v2_leg_layouts<T>()) {
                    Spec s{ta, tb, 160, 136, 48, 4, la};
                    if (lb != la) s.layout_b = lb;
                    s.seed = 9700u + 13u * unsigned(la) + unsigned(lb);
                    auto p = make_problem<T>(s);
                    this->run_pinned(c, p);
                    expect_gemm(*this->ctx, p, name(c, s) + (vec ? " (16-byte leg)" : " (element leg)"));
                    if (::testing::Test::HasFatalFailure()) return;
                    ++leg_ran;
                }
                Spec s{ta, tb, 100, 70, 37, 3, Layout::Gap1};
                s.beta_zero = s.nan_c = true;
                auto p = make_problem<T>(s);
                this->run_pinned(c, p);
                expect_gemm(*this->ctx, p, name(c, s) + " (NaN C)");
                if (::testing::Test::HasFatalFailure()) return;
                ++nan_checked;
            }
        EXPECT_EQ(ran, 9 * 4);
        EXPECT_EQ(leg_ran, 9 * int(wide_v2_leg_layouts<T>().size()));
        EXPECT_EQ(nan_checked, 9);
    }
}

// WP-F, the 64x64x8 slab beside 64x64x16, in all 9 forms, k straddling both slab depths: 8 (one
// k8 slab, half a k16 one), 9 and 17 (a one-deep predicated tail), 16 (two k8 slabs, odd vs even
// count into the staged epilogue). Ragged 100 x 70 at ld = m + 1 (element loads, edge groups) at
// batch 1024, 7 distinct problems, each item bit-identical to its representative; 200 x 136 padded
// (wide v2's 16-byte leg, interior and edge groups; k = 9 again at batch 1024, the vector leg's
// predicated tail slab saturated); packed 128 x 128 x 24 at batch 1024 (all interior, three full
// k8 slabs, the band store). Then beta = 0 and NaN in C. Wide v2 at k = 8
// splits staging: warps 0-3 stage A, 4-7 B, with a KStep and lane map of their own. A B lane
// staging A leaves B unstaged in every form; a k-contiguous map or swizzle copied from k = 16 is
// wrong in the forms with that operand layout; a k-tail `<=` reads the poison past column k.
// evidence: docs/perf/gemm.md#gemm-wp-f-a-k8-slab-for-the-64x64-wide-tile
TYPED_TEST(GemmCandidates, WideK8SlabStraddlesTheSlabDepthAtSaturatingBatch) {
    using T = typename TestFixture::T;
    struct Shape { int m, n, k; Layout layout; int batch; };
    const int big = 1024;
    std::vector<Shape> shapes;
    for (int k : {8, 9, 16, 17}) {
        shapes.push_back({100, 70, k, Layout::Gap1, 1024});
        shapes.push_back({200, 136, k, Layout::Padded, 128});
    }
    shapes.push_back({200, 136, 9, Layout::Padded, big});
    shapes.push_back({128, 128, 24, Layout::Packed, big});
    int ran = 0, nan_checked = 0, tiles = 0;
    for (const C c : {C{og::Wide{64, 64, 8}}, C{og::Wide{64, 64, 16}}}) {
        if (!wide_fits<T>(*wide_row(std::get<og::Wide>(c)))) continue;
        ++tiles;
        for (Transpose ta : kForms)
            for (Transpose tb : kForms) {
                for (const Shape& sh : shapes) {
                    Spec s{ta, tb, sh.m, sh.n, sh.k, sh.batch, sh.layout};
                    s.period = 7;
                    s.seed = 9600u + sh.m + 7u * sh.k;
                    ASSERT_TRUE(TestFixture::expect_runs(c, s)) << name(c, s);
                    auto p = make_problem<T>(s);
                    this->run_pinned(c, p);
                    expect_gemm(*this->ctx, p, name(c, s));
                    if (::testing::Test::HasFatalFailure()) return;
                    ++ran;
                }
                Spec s{ta, tb, 100, 70, 9, 3, Layout::Gap1};
                s.beta_zero = s.nan_c = true;
                auto p = make_problem<T>(s);
                this->run_pinned(c, p);
                expect_gemm(*this->ctx, p, name(c, s) + " (NaN C)");
                if (::testing::Test::HasFatalFailure()) return;
                ++nan_checked;
            }
    }
    EXPECT_GE(tiles, 1) << "64x64x8 asks for half of 64x64x16's local memory";
    EXPECT_EQ(ran, tiles * 9 * int(shapes.size()));
    EXPECT_EQ(nan_checked, tiles * 9);
}

// WP-J, wide v2 for complex<double> (one element per 16-byte granule, its own k-contiguous lane
// map, store and swizzle), with float and complex<float> as the control. Both slab depths, all 9
// forms, m and n straddling the 64 and 128 tile edges both ways, k straddling 8, 16 and 32 both
// ways, every gap NaN (a dropped bound on one operand times the other's zero is NaN, not -0), at
// ld = m + 1 and padded. Then 129 x 65 x 33 at batch 1024, 7 distinct problems, each item
// bit-identical to its representative. A complex<double> store that drops the column offset, a
// swizzle store and read disagree on, or the complex<float> lane map is wrong in every
// complex<double> form with a k-contiguous operand (all but NT and NC) and in no other scalar.
// evidence: docs/perf/gemm.md#gemm-wp-j-complex-double-on-wide-v2
TYPED_TEST(GemmCandidates, WideV2TileAndSlabEdgesUnderNaNPoison) {
    using T = typename TestFixture::T;
    if constexpr (!kWideV2<T>) {
        GTEST_SKIP() << "wide v2 serves float, complex<float> and complex<double>";
    } else {
        const std::pair<int, int> mn[] = {{63, 65}, {64, 64}, {65, 129}, {128, 127}, {129, 63}};
        const int ks[] = {7, 8, 9, 15, 16, 17, 33};
        int ran = 0, saturated = 0, tiles = 0;
        for (const C c : {C{og::Wide{64, 64, 8}}, C{og::Wide{64, 64, 16}}}) {
            if (!wide_fits<T>(*wide_row(std::get<og::Wide>(c)))) continue;
            ++tiles;
            for (Transpose ta : kForms)
                for (Transpose tb : kForms) {
                    for (auto [m, n] : mn)
                        for (int k : ks) {
                            Spec s{ta, tb, m, n, k, 3, (m + k) % 2 ? Layout::Gap1 : Layout::Padded};
                            s.nan_poison = true;
                            s.seed = 9500u + 3u * m + 5u * n + 7u * k;
                            ASSERT_TRUE(TestFixture::expect_runs(c, s)) << name(c, s);
                            auto p = make_problem<T>(s);
                            this->run_pinned(c, p);
                            expect_gemm(*this->ctx, p, name(c, s));
                            if (::testing::Test::HasFatalFailure()) return;
                            ++ran;
                        }
                    Spec s{ta, tb, 129, 65, 33, 1024, Layout::Gap1};
                    s.period = 7;
                    s.nan_poison = true;
                    s.seed = 9400u + 3u * unsigned(ta) + unsigned(tb);
                    auto p = make_problem<T>(s);
                    this->run_pinned(c, p);
                    expect_gemm(*this->ctx, p, name(c, s));
                    if (::testing::Test::HasFatalFailure()) return;
                    ++saturated;
                }
        }
        EXPECT_GE(tiles, 1) << "64x64x8 asks for half of 64x64x16's local memory";
        EXPECT_EQ(ran, tiles * 9 * 5 * 7);
        EXPECT_EQ(saturated, tiles * 9);
    }
}

// WP-J, complex<double> wide v2's element leg (VecLoads = false), reachable only through an 8-byte
// base: every other layout is whole granules. Both slabs, all 9 forms, beta != 0, NaN in every
// gap: 160 x 136 x 48 padded (interior groups, unpredicated full slabs), 100 x 70 x 37 at
// ld = m + 1 (edge groups, the zero-filled k tail), then 160 x 136 x 48 at batch 1024, 7 distinct
// problems, each item bit-identical to its representative. The element load's re-typed zero fill
// or the four-float reassembly is wrong here and in no 16-byte-leg case.
// evidence: docs/perf/gemm.md#gemm-wp-j-complex-double-on-wide-v2
TYPED_TEST(GemmCandidates, WideV2ComplexDoubleElementLegAtAnEightByteBase) {
    using T = typename TestFixture::T;
    if constexpr (!std::is_same_v<T, std::complex<double>>) {
        GTEST_SKIP() << "only complex<double> has an 8-byte base inside its 16-byte granule";
    } else {
        struct Shape { int m, n, k; Layout layout; int batch; };
        const Shape shapes[] = {{160, 136, 48, Layout::Padded, 4}, {100, 70, 37, Layout::Gap1, 4},
                                {160, 136, 48, Layout::Padded, 1024}};
        int ran = 0, tiles = 0;
        for (const C c : {C{og::Wide{64, 64, 8}}, C{og::Wide{64, 64, 16}}}) {
            if (!wide_fits<T>(*wide_row(std::get<og::Wide>(c)))) continue;
            ++tiles;
            for (Transpose ta : kForms)
                for (Transpose tb : kForms)
                    for (const Shape& sh : shapes) {
                        Spec s{ta, tb, sh.m, sh.n, sh.k, sh.batch, sh.layout};
                        s.nan_poison = true;
                        if (sh.batch > 8) s.period = 7;
                        s.seed = 9300u + sh.m + 3u * sh.k + unsigned(sh.batch);
                        ASSERT_TRUE(TestFixture::expect_runs(c, s)) << name(c, s);
                        auto p = make_problem<T>(s);
                        this->run_pinned_base8(c, p);
                        if (::testing::Test::HasFatalFailure()) return;
                        expect_gemm(*this->ctx, p, name(c, s) + " (8-byte base)");
                        if (::testing::Test::HasFatalFailure()) return;
                        ++ran;
                    }
        }
        EXPECT_GE(tiles, 1) << "64x64x8 asks for half of 64x64x16's local memory";
        EXPECT_EQ(ran, tiles * 9 * 3);
    }
}

// validate() refuses mismatched views before choose(), under Auto and every pin; the buffers
// hold every item, so a kernel that ignored the check would stay in bounds.
TYPED_TEST(GemmCandidates, MismatchedViewsThrowUnderEveryPin) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    auto p = make_problem<T>(Spec{kN, kN, 16, 12, 8, 3, Layout::Strided});
    std::vector<std::optional<C>> pins{std::nullopt};
    for (const C& c : og::candidates<T>()) pins.push_back(c);
    // Each case breaks exactly one validate() term; every view stays inside its region. The
    // `< 0` terms are unreachable here: the MatrixView constructor rejects negative extents.
    struct Bad { const char* what; int a_batch, b_batch, b_rows, b_cols, c_rows, c_cols; Transpose tb; };
    const Bad bads[] = {{"A.batch != C.batch", 2, 2, 8, 12, 16, 12, kN}, {"A.batch != B.batch", 3, 2, 8, 12, 16, 12, kN},
                        {"C.rows != m", 3, 3, 8, 12, 15, 12, kN},        {"C.cols != n", 3, 3, 8, 12, 16, 11, kN},
                        {"B.rows != k (NN)", 3, 3, 7, 12, 16, 12, kN},  {"op(B).rows != k (NT)", 3, 3, 12, 7, 16, 12, kT}};
    for (const Bad& bad : bads)
        for (const auto& c : pins) {
            const MVof<T> A(p.mem.data() + p.a.off, 16, 8, p.a.ld, p.a.stride, bad.a_batch);
            const bool nt = bad.tb != kN;  // B is n x k' there: ld 12, inside the region
            const MVof<T> Bm(p.mem.data() + p.b.off, bad.b_rows, bad.b_cols, nt ? 12 : p.b.ld, nt ? 84 : p.b.stride,
                             bad.b_batch);
            const MVof<T> Cm(p.mem.data() + p.c.off, bad.c_rows, bad.c_cols, p.c.ld, p.c.stride, 3);
            std::optional<Pin> pin;
            if (c) pin.emplace("gemm", *c);
            EXPECT_THROW(((void)gemm<B, T>(*this->ctx, A, Bm, Cm, p.alpha, p.beta, kN, bad.tb)), std::invalid_argument)
                << bad.what << " under " << (c ? select::to_string(*c) : std::string("auto"));
        }
    for (std::size_t e = 0; e < p.mem.size(); ++e)
        ASSERT_TRUE(same_bits(p.mem[e], p.mem0[e])) << "a refused call wrote element " << e;
}

// can_run's grid term (R3): direct, tiled, reg and wide put the batch in grid z (<= 65535 on
// CUDA), so a pin is a launch at the ceiling and a refusal one past it; small (1-D) and Auto
// serve the larger batch (real: the transcribed rows' trailing small; complex: the vendor, or
// vendor-free small, the first runnable candidate). Metal has no group-count cap there: one past
// CUDA's ceiling is a launch too.
TYPED_TEST(GemmCandidates, GridCeilingIsACanRunTerm) {
    using T = typename TestFixture::T;
    const select::Device& dev = select::device_of<TestFixture::B>(*this->ctx);
    ASSERT_EQ(dev.max_groups_yz, test_utils::expected_grid_yz(dev.family)) << dev.key;
    const int ceiling = int(test_utils::kCudaGridYZ);
    const bool capped = dev.max_groups_yz <= ceiling;
    const Spec at{kN, kN, 2, 2, 2, ceiling, Layout::Packed};
    const Spec past{kN, kN, 2, 2, 2, ceiling + 1, Layout::Packed};
    for (const C& c : og::candidates<T>()) {
        // split-k's ceiling is batch * s (SplitKGridCeilingIsBatchTimesSlices)
        if (std::holds_alternative<og::Vendor>(c) || std::holds_alternative<og::Small>(c) ||
            std::holds_alternative<og::SplitK>(c))
            continue;
        if (!oracle_native<T>(c, kN, kN, 2, 2, 2)) continue;
        auto p = make_problem<T>(at);
        ASSERT_TRUE(this->pin_accepted(c, p)) << name(c, at);
        expect_gemm(*this->ctx, p, name(c, at));
        auto q = make_problem<T>(past);
        EXPECT_EQ(this->pin_accepted(c, q), !capped) << name(c, past);
        if (!capped) {
            expect_gemm(*this->ctx, q, name(c, past));
            continue;
        }
        for (std::size_t e = 0; e < q.mem.size(); ++e)
            ASSERT_TRUE(same_bits(q.mem[e], q.mem0[e])) << name(c, past) << ": a refused pin wrote element " << e;
    }
    {
        auto p = make_problem<T>(past);
        ASSERT_TRUE(this->pin_accepted(C{og::Small{}}, p)) << "small is a 1-D launch";
        expect_gemm(*this->ctx, p, "small past the grid ceiling");
    }
    if (!capped) return;  // Auto past it is then whatever the table ranks first
    const ScopedEnvVar clear("BATCHLAS_GEMM_ROUTE", nullptr);
    auto p = make_problem<T>(past);
    const std::string got = traced_choice([&] { this->run(p); });
    EXPECT_EQ(got, kCx<T> && TestFixture::kVendor ? "vendor" : "small")
        << label(past) << ": real rows rank small above the vendor; vendor-free complex reaches it last";
    expect_gemm(*this->ctx, p, "auto past the grid ceiling");
}

// An empty batch is a no-op under Auto and every pin, vendor-free too (the old native range
// launched nothing; can_run's batch >= 1 term alone would leave it no route).
TYPED_TEST(GemmCandidates, EmptyBatchIsANoOp) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    auto p = make_problem<T>(Spec{kN, kN, 8, 8, 8, 1, Layout::Packed});
    const MVof<T> A(p.mem.data() + p.a.off, 8, 8, 8, 64, 0), Bm(p.mem.data() + p.b.off, 8, 8, 8, 64, 0),
        Cm(p.mem.data() + p.c.off, 8, 8, 8, 64, 0);
    std::vector<std::optional<C>> pins{std::nullopt};
    for (const C& c : og::candidates<T>()) pins.push_back(c);
    const ScopedEnvVar clear("BATCHLAS_GEMM_ROUTE", nullptr);
    for (const auto& c : pins) {
        std::optional<Pin> pin;
        if (c) pin.emplace("gemm", *c);
        auto call = [&] {
            (void)gemm<B, T>(*this->ctx, A, Bm, Cm, p.alpha, p.beta, kN, kN);
            this->ctx->wait();
        };
        EXPECT_NO_THROW(call()) << (c ? select::to_string(*c) : std::string("auto"));
    }
    for (std::size_t e = 0; e < p.mem.size(); ++e)
        ASSERT_TRUE(same_bits(p.mem[e], p.mem0[e])) << "an empty batch wrote element " << e;
}

// A non-default precision is a correctness condition: every native pin refuses it, and Auto
// takes the vendor (no route at all vendor-free).
TYPED_TEST(GemmCandidates, NonDefaultPrecisionHasNoNativeRoute) {
    using T = typename TestFixture::T;
    const ComputePrecision prec =
        std::is_same_v<RealOf<T>, float> ? ComputePrecision::F32 : ComputePrecision::F64;
    const Spec s{kN, kT, 40, 24, 16, 2, Layout::Strided};
    for (const C& c : og::candidates<T>()) {
        if (std::holds_alternative<og::Vendor>(c) || !oracle_native<T>(c, s.ta, s.tb, s.m, s.n, s.k)) continue;
        auto p = make_problem<T>(s);
        const Pin pin("gemm", c);
        EXPECT_THROW(this->run(p, prec), std::invalid_argument) << select::to_string(c);
    }
    const ScopedEnvVar clear("BATCHLAS_GEMM_ROUTE", nullptr);
    auto p = make_problem<T>(s);
    if constexpr (TestFixture::kVendor) {
        EXPECT_EQ(traced_choice([&] { this->run(p, prec); }), "vendor");
        expect_gemm(*this->ctx, p, "auto, non-default precision");
    } else {
        EXPECT_THROW(this->run(p, prec), batchlas::NoRouteError);
    }
}

// R6: a spelling that names nothing compiled throws instead of meaning Auto, including the
// deleted variants' names, the design doc's never-compiled `tiled:tile=64:k=8`, and the removed
// aliases (pre-selection kernel names and route words), so a resurrected alias goes red.
TYPED_TEST(GemmCandidates, UnknownAndDeletedPinsThrow) {
    using T = typename TestFixture::T;
    const std::vector<std::string> words{
        "bogus", "tiled:tile=64:k=8", "reg:m=1:n=1:k=1:u=1", "wide:m=64:n=64", "native:vendor", "direct:1",
        "128x32x32_s2_u2", "128x32x32_s1_u1", "128x32x32_persistent", "128x32x32_splitk4",
        "128x32x32_s2_u2_tt8x4", "128x32x32_s2_u2_tt4x8", "reg128x64k32largett4x8", "reg128x64k32largett4x8u2",
        std::is_same_v<T, float> ? "reg:m=64:n=64:k=8:u=2" : "reg:m=32:n=32:k=8:u=1",
        "wide:m=16:n=16:k=8",
        "tiled16", "tile16", "smallbatched", "reg32", "reg64", "64x64", "reg64k16tt", "128x128x8",
        "128x32x32_s2_u1", "128x32x32_s2_u1_aligned", "reg128x64k32large", "64x64x16wide", "32x32x16wide",
        "register_tiled", "native:register_tiled", "native:auto", "sycl", "custom", "vendor:auto", "auto:auto",
        "vendor:direct", "cuda-native", "direct-cuda", "cublasdx", "dx"};
    for (const std::string& w : words) {
        auto p = make_problem<T>(Spec{kN, kN, 16, 16, 16, 2, Layout::Packed});
        const Pin pin("gemm", std::string_view(w));
        EXPECT_THROW(this->run(p), std::invalid_argument) << w;
    }
}

// §5.3: spellings (case-folded, trimmed) and the class words, via ScopedPin and via
// BATCHLAS_GEMM_ROUTE. A spelling names a config; the form is the call's. The retired
// BATCHLAS_GEMM_VARIANT and BATCHLAS_GEMM_SYCL_KERNEL are not read.
TYPED_TEST(GemmCandidates, ClassWordsAndSpellings) {
    using T = typename TestFixture::T;
    const ScopedEnvVar clear_route("BATCHLAS_GEMM_ROUTE", nullptr);
    const ScopedEnvVar retired_variant("BATCHLAS_GEMM_VARIANT", "tiled");
    const ScopedEnvVar retired_kernel("BATCHLAS_GEMM_SYCL_KERNEL", "tiled16");
    const Spec nn{kN, kN, 64, 48, 32, 4, Layout::Packed};
    const Spec tt{kT, kT, 64, 48, 32, 4, Layout::Packed};
    const std::string auto_pick = this->auto_choice(nn);
    auto native_best = [&] {
        auto p = make_problem<T>(nn);
        const Pin pin("gemm", std::string_view("native"));
        return traced_choice([&] { this->run(p); });
    }();
    EXPECT_NE(native_best, "vendor");
    const std::string vendor_pick = TestFixture::kVendor ? "vendor" : auto_pick;
    struct Want { const char* word; Spec s; std::string spelling; };
    std::vector<Want> want{
        {"tiled", nn, "tiled"}, {" TILED ", nn, "tiled"}, {"direct", tt, "direct"},
        {"wide:m=64:n=64:k=16", nn, "wide:m=64:n=64:k=16"},
        {"Wide:M=64:N=64:K=16", Spec{kC, kN, 64, 48, 32, 4, Layout::Packed}, "wide:m=64:n=64:k=16"},
        {"wide:128:32:16", Spec{kN, kC, 64, 48, 32, 4, Layout::Packed}, "wide:m=128:n=32:k=16"},
        {"wide:m=32:n=32:k=16", nn, "wide:m=32:n=32:k=16"}, {"wide:m=16:n=16:k=16", nn, "wide:m=16:n=16:k=16"},
        {"wide:m=64:n=32:k=16", nn, "wide:m=64:n=32:k=16"},
        {"wide:32:64:16", Spec{kC, kN, 64, 48, 32, 4, Layout::Packed}, "wide:m=32:n=64:k=16"},
        {"wide:m=64:n=64:k=8", tt, "wide:m=64:n=64:k=8"},
        {"native", nn, native_best}, {"Native", nn, native_best}, {"vendor", nn, vendor_pick},
        {"auto", nn, auto_pick}};
    want.push_back({"small", sizeof(T) == 16 ? Spec{kN, kN, 32, 24, 16, 4, Layout::Packed} : nn, "small"});
    if constexpr (std::is_same_v<T, float>) {
        const std::vector<Want> reg{
            {"reg:m=32:n=32:k=8:u=1", nn, "reg:m=32:n=32:k=8:u=1"}, {"reg:64:64:8:1", nn, "reg:m=64:n=64:k=8:u=1"},
            {"reg:m=64:n=64:k=16:u=1", tt, "reg:m=64:n=64:k=16:u=1"},
            {"reg:m=64:n=64:k=16:u=1", Spec{kT, kN, 64, 48, 32, 4, Layout::Packed}, "reg:m=64:n=64:k=16:u=1"},
            {"reg:m=128:n=32:k=16:u=1", Spec{kN, kT, 64, 48, 32, 4, Layout::Packed}, "reg:m=128:n=32:k=16:u=1"},
            {"reg:m=128:n=32:k=32:u=1", nn, "reg:m=128:n=32:k=32:u=1"},
            {"reg:m=128:n=32:k=32:u=1", Spec{kT, kN, 64, 48, 32, 4, Layout::Packed}, "reg:m=128:n=32:k=32:u=1"},
            {"reg:m=128:n=64:k=16:u=1", tt, "reg:m=128:n=64:k=16:u=1"},
            {"reg:m=32:n=128:k=16:u=1", tt, "reg:m=32:n=128:k=16:u=1"},
            {"reg:m=128:n=64:k=32:u=4", nn, "reg:m=128:n=64:k=32:u=4"},
            {"reg:m=128:n=64:k=32:u=2", nn, "reg:m=128:n=64:k=32:u=2"},
            {"reg:m=128:n=128:k=8:u=1", nn, "reg:m=128:n=128:k=8:u=1"},
            {"reg:m=128:n=128:k=8:u=1", tt, "reg:m=128:n=128:k=8:u=1"},
            {"reg:m=128:n=128:k=8:u=1", Spec{kN, kC, 64, 48, 32, 4, Layout::Packed}, "reg:m=128:n=128:k=8:u=1"},
            {"reg:m=128:n=128:k=8:u=1", Spec{kT, kN, 64, 48, 32, 4, Layout::Packed}, "reg:m=128:n=128:k=8:u=1"}};
        for (const Want& w : reg) {
            const auto c = select::parse<og::GemmChoice>(w.spelling);
            ASSERT_TRUE(c && std::holds_alternative<og::Reg>(*c)) << w.spelling;
            const RegRow* row = reg_row(std::get<og::Reg>(*c));
            if (row && !reg_fits(*row)) {
                // Spelled correctly, refused by the budget: the pin parses and then throws can_run's refusal.
                auto p = make_problem<T>(w.s);
                const Pin pin("gemm", std::string_view(w.word));
                try {
                    this->run(p);
                    ADD_FAILURE() << w.word << " ran above the local memory budget " << g_slm_budget;
                } catch (const std::invalid_argument& e) {
                    EXPECT_NE(std::string(e.what()).find("cannot run this shape"), std::string::npos) << e.what();
                }
                continue;
            }
            want.push_back(w);
        }
    }
    for (const auto& w : want)
        for (bool via_env : {false, true}) {
            auto p = make_problem<T>(w.s);
            select::testing::reset_warnings();
            const std::string what = std::string(w.word) + " " + label(w.s) + (via_env ? " via env" : " via ScopedPin");
            const std::string got = traced_choice([&] {
                const ScopedEnvVar env("BATCHLAS_GEMM_ROUTE", via_env ? w.word : nullptr);
                std::optional<Pin> pin;
                if (!via_env) pin.emplace("gemm", std::string_view(w.word));
                this->run(p);
            });
            EXPECT_EQ(got, w.spelling) << what;
            expect_gemm(*this->ctx, p, what);
        }
}

// §5.3: a ScopedPin wins over BATCHLAS_GEMM_ROUTE, and nested pins restore the outer one.
TYPED_TEST(GemmCandidates, ScopedPinBeatsTheEnvironment) {
    using T = typename TestFixture::T;
    const Spec tt{kT, kT, 40, 30, 20, 2, Layout::Strided};  // a form wide 32x32 does not instantiate
    const ScopedEnvVar env("BATCHLAS_GEMM_ROUTE", "wide:m=32:n=32:k=16");
    {
        auto p = make_problem<T>(tt);
        EXPECT_THROW(this->run(p), std::invalid_argument) << "the environment pin was not read";
    }
    auto p = make_problem<T>(tt);
    EXPECT_EQ(traced_choice([&] {
                  const Pin pin("gemm", C{og::Tiled{}});
                  this->run(p);
              }),
              "tiled");
    expect_gemm(*this->ctx, p, "tiled over the env pin");
    const Pin outer("gemm", C{og::Direct{}});
    {
        const Pin inner("gemm", C{og::Tiled{}});
        auto r = make_problem<T>(tt);
        EXPECT_EQ(traced_choice([&] { this->run(r); }), "tiled");
    }
    auto r = make_problem<T>(tt);
    EXPECT_EQ(traced_choice([&] { this->run(r); }), "direct") << "the inner pin did not restore the outer";
    expect_gemm(*this->ctx, r, "outer direct");
}

// key_of's every field reaches choose(): a synthetic table for this device whose winner changes
// with ta, tb (a real ConjTrans folds to T, a complex one has its own row), layout, m, n, k and
// batch alone. Fixing a field in key_of turns exactly its probe red.
TYPED_TEST(GemmCandidates, AutoReadsEveryKeyField) {
    using T = typename TestFixture::T;
    const ScopedEnvVar clear("BATCHLAS_GEMM_ROUTE", nullptr);
    const std::string dtype(select::dtype_name<T>());
    const std::string dev = select::device_of<TestFixture::B>(*this->ctx).key;
    std::string rows =
        "ta=N tb=N layout=packed m=16 n=16 k=16 batch=128 | direct 1 | tiled 2\n"
        "ta=T tb=N layout=packed m=16 n=16 k=16 batch=128 | tiled 1 | direct 2\n"
        "ta=N tb=T layout=packed m=16 n=16 k=16 batch=128 | tiled 1 | direct 2\n"
        "ta=N tb=N layout=strided m=16 n=16 k=16 batch=128 | tiled 1 | direct 2\n"
        "ta=N tb=N layout=packed m=128 n=16 k=16 batch=128 | tiled 1 | direct 2\n"
        "ta=N tb=N layout=packed m=16 n=128 k=16 batch=128 | tiled 1 | direct 2\n"
        "ta=N tb=N layout=packed m=16 n=16 k=128 batch=128 | tiled 1 | direct 2\n"
        "ta=N tb=N layout=packed m=16 n=16 k=16 batch=4096 | tiled 1 | direct 2\n";
    if constexpr (kCx<T>)
        rows += "ta=C tb=N layout=packed m=16 n=16 k=16 batch=128 | wide:m=64:n=64:k=16 1 | direct 2\n"
                "ta=N tb=C layout=packed m=16 n=16 k=16 batch=128 | wide:m=64:n=64:k=16 1 | direct 2\n";
    const TableGuard restore;
    select::testing::set_builtin_tables(test_utils::dtype_keyed(with_gemm_table(dtype, dev, rows)));
    const std::string conj = kCx<T> ? "wide:m=64:n=64:k=16" : "tiled";
    struct Probe { Spec s; std::string expect; const char* field; };
    const Probe probes[] = {
        {{kN, kN, 16, 16, 16, 128, Layout::Packed}, "direct", "base"},
        {{kT, kN, 16, 16, 16, 128, Layout::Packed}, "tiled", "ta"},
        {{kN, kT, 16, 16, 16, 128, Layout::Packed}, "tiled", "tb"},
        {{kC, kN, 16, 16, 16, 128, Layout::Packed}, conj, "ta=C (real: folds to T)"},
        {{kN, kC, 16, 16, 16, 128, Layout::Packed}, conj, "tb=C (real: folds to T)"},
        {{kN, kN, 16, 16, 16, 128, Layout::Strided}, "tiled", "layout"},
        {{kN, kN, 128, 16, 16, 128, Layout::Packed}, "tiled", "m"},
        {{kN, kN, 16, 128, 16, 128, Layout::Packed}, "tiled", "n"},
        {{kN, kN, 16, 16, 128, 128, Layout::Packed}, "tiled", "k"},
        {{kN, kN, 16, 16, 16, 4096, Layout::Packed}, "tiled", "batch"}};
    for (const auto& k : probes) {
        auto p = make_problem<T>(k.s);
        EXPECT_EQ(traced_choice([&] { this->run(p); }), k.expect) << "the " << k.field << " probe, " << label(k.s);
        expect_gemm(*this->ctx, p, std::string("the ") + k.field + " probe");
    }
}

// The trace key: C folds to T for a real scalar only; layout=packed exactly for contiguous
// operands with 16-byte bases.
TYPED_TEST(GemmCandidates, TraceKeyFoldsAndClassifiesLayout) {
    using T = typename TestFixture::T;
    struct Case { Spec s; std::string key; };
    const std::string cw = kCx<T> ? "C" : "T";
    const Case cases[] = {
        {{kC, kN, 12, 7, 5, 3, Layout::Packed}, cw + " tb=N layout=packed m=12 n=7 k=5 batch=3 ->"},
        {{kN, kC, 12, 7, 5, 1, Layout::Strided}, "N tb=" + cw + " layout=strided m=12 n=7 k=5 batch=1 ->"},
        {{kT, kT, 9, 9, 9, 2, Layout::Packed}, "T tb=T layout=packed m=9 n=9 k=9 batch=2 ->"}};
    for (const auto& k : cases) {
        auto p = make_problem<T>(k.s);
        const std::string line = traced_line([&] { this->run(p); });
        EXPECT_NE(line.find("ta=" + k.key), std::string::npos) << line;
        expect_gemm(*this->ctx, p, line);
    }
    // A at a 16-byte base but ld != rows is strided; so is a contiguous A off a 16-byte base.
    auto p = make_problem<T>(Spec{kN, kN, 8, 8, 8, 1, Layout::Packed});
    std::vector<MVof<T>> views{MVof<T>(p.mem.data() + p.a.off, 8, 8, 9, 72, 1)};
    if constexpr (sizeof(T) < 16) views.emplace_back(p.mem.data() + p.a.off + 1, 8, 8, 8, 64, 1);
    for (const MVof<T>& a : views) {
        const std::string line = traced_line([&] {
            (void)gemm<TestFixture::B, T>(*this->ctx, a, p.B(), p.Cv(), T(1), T(0), kN, kN);
            this->ctx->wait();
        });
        EXPECT_NE(line.find("layout=strided"), std::string::npos) << line;
    }
    // A sub-batch: ld == rows and a 16-byte base, but the batch stride is padded.
    auto q = make_problem<T>(Spec{kN, kN, 8, 8, 8, 2, Layout::Packed});
    const MVof<T> padded(q.mem.data() + q.a.off, 8, 8, 8, 68, 2);
    ASSERT_EQ(reinterpret_cast<std::uintptr_t>(padded.data_ptr()) % 16, 0u);
    const std::string line = traced_line([&] {
        (void)gemm<TestFixture::B, T>(*this->ctx, padded, q.B(), q.Cv(), T(1), T(0), kN, kN);
        this->ctx->wait();
    });
    EXPECT_NE(line.find("layout=strided"), std::string::npos) << line;
}

// Auto against the shipped table this device reads (shipped_table_pick.hh) at the cells the old
// router's windows bracketed: the first entry of the nearest row that runs, the vendor skipped
// vendor-free. A regenerated table changes the expectation, not the test.
TYPED_TEST(GemmCandidates, AutoReadsTheSm89TranscribedTable) {
    using T = typename TestFixture::T;
    const ScopedEnvVar clear("BATCHLAS_GEMM_ROUTE", nullptr);
    std::vector<Spec> rows;
    if constexpr (std::is_same_v<T, float>)
        rows = {{kN, kN, 32, 32, 32, 128, Layout::Packed}, {kN, kN, 128, 128, 128, 128, Layout::Packed},
                {kT, kN, 256, 128, 128, 128, Layout::Strided}, {kN, kT, 256, 32, 96, 2048, Layout::Strided}};
    else if constexpr (std::is_same_v<T, double>)
        rows = {{kN, kN, 16, 16, 16, 128, Layout::Packed}, {kN, kN, 256, 256, 256, 128, Layout::Packed},
                {kN, kC, 256, 32, 96, 2048, Layout::Strided}};
    else
        rows = {{kN, kC, 256, 32, 96, 2048, Layout::Strided}, {kC, kN, 32, 256, 128, 2048, Layout::Strided},
                {kN, kN, 256, 256, 256, 128, Layout::Strided}};
    const auto word = [](Transpose t) { return t == kN ? "N" : (!kCx<T> || t == kT ? "T" : "C"); };
    for (const Spec& s : rows) {
        const select::Key key{{"ta", word(s.ta)}, {"tb", word(s.tb)},
                              {"layout", s.layout == Layout::Packed ? "packed" : "strided"},
                              {"m", s.m}, {"n", s.n}, {"k", s.k}, {"batch", s.batch}};
        const std::string want = test_utils::shipped_table_pick<C>(
            "gemm", select::dtype_name<T>(), select::device_of<TestFixture::B>(*this->ctx), key, [&] {
                auto q = make_problem<T>(s);
                this->run(q);
            });
        ASSERT_NE(want, test_utils::kNoTableEntryRuns) << label(s);
        auto p = make_problem<T>(s);
        EXPECT_EQ(traced_choice([&] { this->run(p); }), want) << label(s);
        expect_gemm(*this->ctx, p, "auto " + label(s));
    }
}

// With a table naming only the vendor, a vendor-free build takes the last resort, direct.
TYPED_TEST(GemmCandidates, VendorFreeLastResortIsDirect) {
    using T = typename TestFixture::T;
    const ScopedEnvVar clear("BATCHLAS_GEMM_ROUTE", nullptr);
    const std::string dtype(select::dtype_name<T>());
    const std::string dev = select::device_of<TestFixture::B>(*this->ctx).key;
    const TableGuard restore;
    select::testing::set_builtin_tables(test_utils::dtype_keyed(
        with_gemm_table(dtype, dev, "ta=N tb=N layout=packed m=64 n=64 k=64 batch=128 | vendor 1\n")));
    for (const Spec& s : {Spec{kN, kN, 64, 64, 64, 4, Layout::Packed}, Spec{kT, kC, 300, 5, 70, 2, Layout::Strided}}) {
        auto p = make_problem<T>(s);
        std::string all;
        EXPECT_EQ(traced_choice([&] { this->run(p); }, &all), TestFixture::kVendor ? "vendor" : "direct") << label(s);
        if (!TestFixture::kVendor) EXPECT_NE(all.find("last resort"), std::string::npos) << all;
        expect_gemm(*this->ctx, p, "vendor-only table, " + label(s));
    }
}

// The coverage row (§5.6): the real backend, gemm's m, n, k, batch, transA and transB, the
// chosen spelling and the native flags. threadsafe: the child re-executes the binary.
TYPED_TEST(GemmCandidates, CoverageRowCarriesBackendKeyAndNativeFlags) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
    const std::string dir = ::testing::TempDir() + "gemm_cov." + std::string(select::dtype_name<T>());
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    const std::string out = dir + "/cov";
    const Spec lo{kN, kC, 24, 10, 6, 2, Layout::Strided};
    const Spec hi{kT, kN, 40, 33, 17, 3, Layout::Packed};
    auto child = [&] {
        const ScopedEnvVar cov("BATCHLAS_COVERAGE_OUT", out.c_str());
        const ScopedEnvVar clear("BATCHLAS_GEMM_ROUTE", nullptr);
        batchlas::coverage::g_dynamic_enabled = true;
        {
            const Pin pin("gemm", C{og::Tiled{}});
            auto p = make_problem<T>(lo);
            this->run(p);
        }
        const Pin pin("gemm", C{og::Direct{}});
        auto p = make_problem<T>(hi);
        this->run(p);
        std::exit(0);
    };
    EXPECT_EXIT(child(), ::testing::ExitedWithCode(0), "");
    std::map<std::string, std::vector<std::string>> rows;  // "m n k" -> fields
    for (const auto& ent : std::filesystem::directory_iterator(dir)) {
        std::ifstream in(ent.path());
        for (std::string line; std::getline(in, line);) {
            if (line.rfind("reached,gemm,", 0) != 0) continue;
            std::vector<std::string> f;
            std::stringstream ss(line);
            for (std::string tok; std::getline(ss, tok, ',');) f.push_back(tok);
            ASSERT_GE(f.size(), 20u) << line;
            rows[f[5] + " " + f[6] + " " + f[7]] = f;
        }
    }
    std::filesystem::remove_all(dir);
    ASSERT_EQ(rows.size(), 2u);
    const std::string backend = B == Backend::CUDA   ? "CUDA"
                                : B == Backend::ROCM ? "ROCM"
                                : B == Backend::SYCL ? "SYCL"
                                                     : "?";
    auto as_int = [](auto e) { return std::to_string(static_cast<int>(e)); };
    for (const auto& [s, algo] : {std::pair{lo, "tiled"}, std::pair{hi, "direct"}}) {
        const std::string key = std::to_string(s.m) + " " + std::to_string(s.n) + " " + std::to_string(s.k);
        ASSERT_TRUE(rows.count(key)) << "no row for " << key;
        const auto& f = rows[key];
        EXPECT_EQ(f[3], backend) << key << ": the backend column";
        EXPECT_EQ(f[8], std::to_string(s.batch)) << key;
        EXPECT_EQ(f[9], "native") << key;
        EXPECT_EQ(f[10], algo) << key;
        EXPECT_EQ(f[12], "1") << key;
        EXPECT_EQ(f[13], "1") << key;
        EXPECT_EQ(f[18], as_int(s.ta)) << key;
        EXPECT_EQ(f[19], as_int(s.tb)) << key;
    }
}

// On a CPU queue the native families run only where there is no host BLAS (a vendor-free
// build), so native pins throw and Auto is the vendor; vendor-free, the last resort is direct.
template <typename Config>
class GemmCandidatesCpu : public test_utils::BatchLASTest<Config> {};
TYPED_TEST_SUITE(GemmCandidatesCpu, Types);

TYPED_TEST(GemmCandidatesCpu, CpuQueueRunsNativeOnlyWithoutAHostBlas) {
    using T = typename TypeParam::ScalarType;
    static constexpr Backend B = TypeParam::BackendVal;
    static constexpr bool kVendor = batchlas::select::level3_vendor_available<B>;
    if (!this->ctx) GTEST_SKIP() << "no queue";
    if (this->ctx->device().type == DeviceType::GPU) GTEST_SKIP() << "a GPU queue";
    const ScopedEnvVar clear("BATCHLAS_GEMM_ROUTE", nullptr);
    const Spec s{kN, kT, 9, 7, 5, 2, Layout::Strided};
    auto call = [&](Problem<T>& p) {
        (void)gemm<B, T>(*this->ctx, p.A(), p.B(), p.Cv(), p.alpha, p.beta, s.ta, s.tb);
        this->ctx->wait();
    };
    for (const C& c : {C{og::Direct{}}, C{og::Tiled{}}}) {
        auto p = make_problem<T>(s);
        const Pin pin("gemm", c);
        if constexpr (kVendor) {
            EXPECT_THROW(call(p), std::invalid_argument) << select::to_string(c);
        } else {
            call(p);
            expect_gemm(*this->ctx, p, "cpu " + select::to_string(c));
        }
    }
    auto p = make_problem<T>(s);
    EXPECT_EQ(traced_choice([&] { call(p); }), kVendor ? "vendor" : "direct");
    expect_gemm(*this->ctx, p, "cpu auto");
}

// The transcribed sm_89 rows, read with Table::nearest directly so every device checks them
// (AutoReadsTheSm89TranscribedTable runs Auto where sm_89 is read): small leads the float NN
// squares to 48 and the vendor the rest of float; double is native first; complex is vendor
// first; a real C folds to T; the old forced-name fallback follows the old kernel.
TEST(GemmTranscribedTable, Sm89RowsHoldTheTranscribedPreference) {
    test_utils::TranscribedRowsHeld held;
    struct Row {
        const char* dtype; const char* ta; const char* tb; const char* layout; int m, n, k, batch;
        std::vector<std::string> ranked;
    };
    const Row rows[] = {
        {"float", "N", "N", "packed", 32, 32, 32, 128, {"small", "direct", "tiled", "vendor"}},
        {"float", "N", "N", "packed", 64, 64, 64, 128, {"vendor", "small", "direct", "tiled"}},
        {"float", "N", "N", "packed", 256, 256, 256, 128, {"vendor", "reg:m=128:n=128:k=8:u=1", "tiled", "direct"}},
        {"float", "T", "N", "strided", 256, 128, 128, 128, {"vendor", "reg:m=128:n=32:k=32:u=1", "tiled", "direct"}},
        {"float", "N", "T", "strided", 256, 32, 96, 2048, {"vendor", "tiled", "direct"}},
        {"double", "N", "N", "packed", 16, 16, 16, 128, {"direct", "tiled", "small", "vendor"}},
        {"double", "N", "N", "packed", 256, 256, 256, 128, {"wide:m=64:n=64:k=16", "tiled", "direct", "vendor"}},
        {"double", "N", "T", "strided", 256, 32, 96, 2048, {"tiled", "direct", "vendor"}},
        {"cfloat", "N", "C", "strided", 256, 32, 96, 2048, {"vendor", "wide:m=128:n=32:k=16", "tiled", "direct"}},
        {"cfloat", "N", "N", "packed", 256, 256, 256, 128, {"vendor", "wide:m=64:n=64:k=16", "tiled", "direct"}},
        {"cdouble", "C", "N", "strided", 32, 256, 128, 2048, {"vendor", "wide:m=32:n=128:k=16", "tiled", "direct"}}};
    for (const Row& r : rows) {
        const auto tables = select::tables_in_borrow_order("gemm", r.dtype, select::device_from_key("sm_89"));
        ASSERT_FALSE(tables.empty()) << r.dtype;
        const select::Table& t = *tables.front();
        ASSERT_EQ(t.device, "sm_89") << r.dtype;
        const select::Key key{{"ta", r.ta}, {"tb", r.tb}, {"layout", r.layout}, {"m", r.m},
                              {"n", r.n},   {"k", r.k},   {"batch", r.batch}};
        const select::TableRow* row = t.nearest(key);
        ASSERT_NE(row, nullptr) << t.file;
        // A measured (timed) row ranks by the tuner, not by the transcription: nothing to hold it to.
        if (held.skip(row->timed)) continue;
        const std::string what = t.file + ":" + std::to_string(row->line);
        std::vector<std::string> got;
        for (const auto& e : row->ranked) got.push_back(e.spelling);
        EXPECT_EQ(got, r.ranked) << what;
    }
    // No real table has a C row: key_of folds it to T before the lookup.
    for (const char* dt : {"float", "double"}) {
        const auto tables = select::tables_in_borrow_order("gemm", dt, select::device_from_key("sm_89"));
        ASSERT_FALSE(tables.empty());
        const select::Table& t = *tables.front();
        ASSERT_GE(t.keys.size(), 2u);
        ASSERT_EQ(t.keys[0].name, "ta");
        ASSERT_EQ(t.keys[1].name, "tb");
        for (const auto& row : t.rows)
            ASSERT_TRUE(row.keys[0] != "C" && row.keys[1] != "C") << t.file << ":" << row.line;
    }
    SKIP_IF_NO_TRANSCRIBED_ROWS(held);
}

// The old preferred() edges below the grid (batch < 64, double k < 2, float native only on NN
// squares to 48) survive nearest lookup: the transcriber brackets each on both sides.
TEST(GemmTranscribedTable, Sm89BracketsTheTranscribedVendorEdges) {
    struct Cell { const char* dtype; const char* ta; const char* tb; const char* layout; int m, n, k, batch; const char* first; };
    const Cell cells[] = {
        {"double", "N", "N", "packed", 256, 256, 256, 8, "vendor"},
        {"double", "N", "N", "packed", 256, 256, 256, 63, "vendor"},
        {"double", "N", "N", "packed", 256, 256, 256, 64, "wide:m=64:n=64:k=16"},
        {"double", "N", "N", "packed", 512, 512, 1, 2048, "vendor"},
        {"double", "N", "N", "packed", 512, 512, 2, 2048, "tiled"},
        {"double", "N", "T", "strided", 300, 300, 1, 1000, "vendor"},
        {"double", "T", "T", "strided", 20, 20, 1, 500, "vendor"},
        {"float", "N", "N", "packed", 52, 52, 52, 2048, "vendor"},
        {"float", "N", "N", "packed", 48, 48, 48, 2048, "small"},
        {"float", "N", "N", "packed", 49, 49, 49, 2048, "vendor"},
        {"float", "N", "N", "packed", 40, 40, 48, 2048, "vendor"},
        {"float", "N", "N", "strided", 40, 40, 40, 2048, "small"},
        {"float", "N", "N", "packed", 32, 32, 1, 2048, "vendor"},
        {"float", "N", "N", "packed", 32, 32, 32, 16, "vendor"},
        {"float", "N", "N", "packed", 32, 32, 32, 64, "small"},
        {"float", "N", "N", "packed", 8, 8, 8, 1, "vendor"},
        {"float", "N", "N", "packed", 2, 2, 2, 2048, "small"},
        {"float", "N", "N", "strided", 2, 8, 8, 2048, "vendor"}};
    for (const Cell& c : cells) {
        const auto tables = select::tables_in_borrow_order("gemm", c.dtype, select::device_from_key("sm_89"));
        ASSERT_FALSE(tables.empty()) << c.dtype;
        const select::Table& t = *tables.front();
        const select::Key key{{"ta", c.ta}, {"tb", c.tb}, {"layout", c.layout}, {"m", c.m},
                              {"n", c.n},   {"k", c.k},   {"batch", c.batch}};
        const select::TableRow* row = t.nearest(key);
        ASSERT_NE(row, nullptr) << t.file;
        EXPECT_EQ(row->ranked.front().spelling, c.first)
            << c.dtype << " " << c.ta << c.tb << " " << c.layout << " " << c.m << "x" << c.n << "x" << c.k
            << " batch=" << c.batch << " -> " << t.file << ":" << row->line;
    }
}

// small's batched leg requires sub-group size 32; only the float NN tiled leg (33..56) does not.
TEST(GemmCanRun, SmallNeedsSubGroup32OutsideTheFloatTiledLeg) {
    select::Device sg = select::device_from_key("sm_89");
    sg.is_gpu = true;
    sg.max_wg = 1024;
    sg.has_sg32 = true;
    sg.slm_budget = 45056;  // CUDA's 48 KiB less the 4 KiB reserve
    select::Device nosg = sg;
    nosg.has_sg32 = false;
    struct Probe { bool is_float, nn; int mx; bool without_sg32; };
    const Probe probes[] = {{true, true, 32, false}, {true, true, 33, true}, {true, true, 56, true},
                            {true, true, 57, false}, {true, false, 40, false}, {false, true, 40, false},
                            {false, false, 8, false}};
    for (const Probe& p : probes) {
        const bool with = p.is_float ? og::small_fits<float>(sg, p.nn, p.mx) : og::small_fits<double>(sg, p.nn, p.mx);
        const bool without =
            p.is_float ? og::small_fits<float>(nosg, p.nn, p.mx) : og::small_fits<double>(nosg, p.nn, p.mx);
        const std::string what = std::string(p.is_float ? "float" : "double") + (p.nn ? " NN" : " NT") +
                                 " max_dim=" + std::to_string(p.mx);
        EXPECT_TRUE(with) << what;
        EXPECT_EQ(without, p.without_sg32) << what;
    }
    EXPECT_FALSE(og::small_fits<float>(sg, true, 65));
    EXPECT_FALSE(og::small_fits<std::complex<float>>(sg, true, 65));
    EXPECT_FALSE(og::small_fits<std::complex<float>>(nosg, true, 40)) << "complex has no tiled leg";
}

// small's ceiling and local memory term (R3), literal per scalar: complex<double> stops at 32, every
// other scalar at 64, and a budget one byte short of a bucket's bytes refuses exactly that bucket.
TEST(GemmCanRun, SmallCeilingAndLocalMemoryPerScalar) {
    select::Device d = select::device_from_key("sm_89");
    d.is_gpu = true;
    d.max_wg = 1024;
    d.has_sg32 = true;
    d.slm_budget = 45056;
    EXPECT_TRUE(og::small_fits<std::complex<double>>(d, false, 32));
    EXPECT_FALSE(og::small_fits<std::complex<double>>(d, false, 33));
    EXPECT_FALSE(og::small_fits<std::complex<double>>(d, true, 64));
    for (bool nn : {false, true}) {
        EXPECT_TRUE(og::small_fits<std::complex<float>>(d, nn, 64));
        EXPECT_TRUE(og::small_fits<double>(d, nn, 64));
        EXPECT_TRUE(og::small_fits<float>(d, nn, 64));
    }
    // 128 / (2 NB) slots of NB x (NB + 1) elements; the float NN tiled leg 2 NB (NB + 4).
    EXPECT_EQ(og::small_slm_bytes<float>(false, 8), 2304);
    EXPECT_EQ(og::small_slm_bytes<std::complex<float>>(false, 16), 8704);
    EXPECT_EQ(og::small_slm_bytes<std::complex<double>>(true, 32), 33792);
    EXPECT_EQ(og::small_slm_bytes<std::complex<float>>(true, 64), 33280);
    EXPECT_EQ(og::small_slm_bytes<double>(false, 33), 33280);
    EXPECT_EQ(og::small_slm_bytes<float>(true, 56), 26880);
    EXPECT_EQ(og::small_slm_bytes<float>(false, 56), 16640);
    struct Edge { const char* what; std::int64_t bytes; bool (*fits)(const select::Device&); };
    const Edge edges[] = {
        {"cfloat NT 64", 33280, [](const select::Device& v) { return og::small_fits<std::complex<float>>(v, false, 64); }},
        {"cdouble NN 32", 33792, [](const select::Device& v) { return og::small_fits<std::complex<double>>(v, true, 32); }},
        {"double NN 40", 33280, [](const select::Device& v) { return og::small_fits<double>(v, true, 40); }},
        {"float NN 56 (tiled)", 26880, [](const select::Device& v) { return og::small_fits<float>(v, true, 56); }}};
    for (const Edge& e : edges) {
        select::Device at = d, short_by_one = d;
        at.slm_budget = e.bytes;
        short_by_one.slm_budget = e.bytes - 1;
        EXPECT_TRUE(e.fits(at)) << e.what;
        EXPECT_FALSE(e.fits(short_by_one)) << e.what;
    }
}

// candidates<T>() holds exactly the oracle's configs: a dropped entry would otherwise just
// shorten every per-candidate loop.
template <typename T>
void expect_candidate_set() {
    std::set<std::string> got, want{"direct", "tiled", "vendor"};
    for (const C& c : og::candidates<T>()) EXPECT_TRUE(got.insert(select::to_string(c)).second) << select::to_string(c);
    want.insert("small");
    if constexpr (std::is_same_v<T, float>)
        for (const auto& r : reg_rows())
            want.insert("reg:m=" + std::to_string(r.m) + ":n=" + std::to_string(r.n) + ":k=" + std::to_string(r.k) +
                        ":u=" + std::to_string(r.u));
    for (const auto& w : wide_rows()) {
        want.insert("wide:m=" + std::to_string(w.m) + ":n=" + std::to_string(w.n) + ":k=" + std::to_string(w.k));
        for (int sl : {2, 4, 8})
            if (splitk_row(og::SplitK{w.m, w.n, w.k, sl}))
                want.insert("splitk:m=" + std::to_string(w.m) + ":n=" + std::to_string(w.n) + ":k=" + std::to_string(w.k) +
                            ":s=" + std::to_string(sl));
    }
    EXPECT_EQ(got, want) << select::dtype_name<T>();
}
TEST(GemmCandidateSet, EqualsTheOracleRows) {
    expect_candidate_set<float>();
    expect_candidate_set<double>();
    expect_candidate_set<std::complex<float>>();
    expect_candidate_set<std::complex<double>>();
}

}  // namespace
