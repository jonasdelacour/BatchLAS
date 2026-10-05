// Every gemm candidate, pinned: docs/design/flat-kernel-selection.md §8 and
// flat-kernel-selection-phase3-plan.md §1.3. The oracle for which (config, form) pairs exist is
// the old KernelVariant list, written out here; which kernel ran is read back from the select
// trace, the kernel trace or a bit-for-bit comparison with the family's launcher, never assumed
// from the pin being accepted.
#include <gtest/gtest.h>

#include <batchlas/blas/functions/gemm.hh>
#include <batchlas/blas/dispatch/coverage.hh>
#include <batchlas/blas/dispatch/no_route.hh>
#include <batchlas/blas/dispatch/vendor_available.hh>
#include <batchlas/error.hh>
#include <batchlas/util/env.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-vector.hh>

#include "test_utils.hh"

#include "../src/ops/gemm/choice.hh"
#include "../src/sycl/gemm_kernels.hh"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <optional>
#include <random>
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
T mk(RealOf<T> r, RealOf<T> i) {
    if constexpr (kCx<T>) return T(r, i);
    else return r;
}
template <typename T>
std::complex<double> up(T v) {
    if constexpr (kCx<T>) return {double(v.real()), double(v.imag())};
    else return {double(v), 0.0};
}
template <typename T>
bool same_bits(T a, T b) {
    return std::memcmp(&a, &b, sizeof(T)) == 0;
}
const char* tw(Transpose t) { return t == kN ? "N" : (t == kT ? "T" : "C"); }

// ---- the oracle: the compiled kernels as the pre-P3.4 KernelVariant enum named them ----

struct RegRow { int m, n, k, u; std::vector<std::string> forms; std::map<std::string, std::string> trace; };
const std::vector<RegRow>& reg_rows() {
    static const std::vector<RegRow> rows{
        {32, 32, 8, 1, {"NN"}, {{"NN", "gemm_sycl_register_32x32"}}},
        {64, 64, 8, 1, {"NN"}, {{"NN", "gemm_sycl_register_64x64"}}},
        {64, 64, 16, 1, {"NN", "NT", "TN", "TT"},
         {{"NN", "gemm_sycl_register_64x64_k16"}, {"NT", "gemm_sycl_register_64x64_k16_nt"},
          {"TN", "gemm_sycl_register_64x64_k16_tn"}, {"TT", "gemm_sycl_register_64x64_k16_tt"}}},
        {128, 32, 16, 1, {"NN", "NT", "TN", "TT"},
         {{"NN", "gemm_sycl_register_128x32_k16"}, {"NT", "gemm_sycl_register_128x32_k16_nt"},
          {"TN", "gemm_sycl_register_128x32_k16_tn"}, {"TT", "gemm_sycl_register_128x32_k16_tt"}}},
        {128, 32, 32, 1, {"NN", "NT", "TN", "TT"},
         {{"NN", "gemm_sycl_register_128x32_k32_s2_u1_generic"},
          {"NN+aligned", "gemm_sycl_register_128x32_k32_s2_u1_aligned"},
          {"NT", "gemm_sycl_register_128x32_k32_nt"}, {"TN", "gemm_sycl_register_128x32_k32_tn"},
          {"TT", "gemm_sycl_register_128x32_k32_tt"}}},
        {128, 64, 16, 1, {"NT", "TN", "TT"},
         {{"NT", "gemm_sycl_register_128x64_k16_nt"}, {"TN", "gemm_sycl_register_128x64_k16_tn"},
          {"TT", "gemm_sycl_register_128x64_k16_tt"}}},
        {32, 128, 16, 1, {"NN", "TN", "TT"},
         {{"NN", "gemm_sycl_register_32x128_k16"}, {"TN", "gemm_sycl_register_32x128_k16_tn"},
          {"TT", "gemm_sycl_register_32x128_k16_tt"}}},
        {128, 64, 32, 4, {"NN"}, {{"NN", "gemm_sycl_register_128x64_k32_large"}}},
        {128, 64, 32, 2, {"NN"}, {{"NN", "gemm_sycl_register_128x64_k32_large_u2"}}},
        {128, 128, 8, 1, {"NN"}, {{"NN", "gemm_sycl_register_128x128_k8"}}},
    };
    return rows;
}
struct WideRow { int m, n, k; std::map<std::string, std::string> trace; };  // forms NN, CN, NC
const std::vector<WideRow>& wide_rows() {
    static const std::vector<WideRow> rows{
        {64, 64, 16, {{"NN", "gemm_sycl_register_64x64_k16_wide"}, {"CN", "gemm_sycl_register_64x64_k16_wide_cn"},
                      {"NC", "gemm_sycl_register_64x64_k16_wide_nc"}}},
        {128, 32, 16, {{"NC", "gemm_sycl_register_128x32_k16_wide_nc"}}},
        {32, 128, 16, {{"CN", "gemm_sycl_register_32x128_k16_wide_cn"}}},
        {32, 32, 16, {{"NN", "gemm_sycl_register_32x32_k16_wide"}}},
        {16, 16, 16, {{"NN", "gemm_sycl_register_16x16_k16_wide"}}},
    };
    return rows;
}

// A register tile's form: a real ConjTrans is its Trans.
std::string reg_form(Transpose ta, Transpose tb) {
    return std::string(ta == kN ? "N" : "T") + (tb == kN ? "N" : "T");
}
// A wide tile's form: its instantiations are NN, ConjTrans-A and ConjTrans-B; a real Trans is a
// ConjTrans, a complex Trans is neither. "" names no instantiation.
template <typename T>
std::string wide_form(Transpose ta, Transpose tb) {
    auto conj_like = [](Transpose t) { return t == kC || (!kCx<T> && t == kT); };
    if (ta == kN && tb == kN) return "NN";
    if (conj_like(ta) && tb == kN) return "CN";
    if (ta == kN && conj_like(tb)) return "NC";
    return "";
}

const RegRow* reg_row(const og::Reg& r) {
    for (const auto& row : reg_rows())
        if (row.m == r.m && row.n == r.n && row.k == r.k && row.u == r.u) return &row;
    return nullptr;
}
const WideRow* wide_row(const og::Wide& w) {
    for (const auto& row : wide_rows())
        if (row.m == w.m && row.n == w.n && row.k == w.k) return &row;
    return nullptr;
}

// Does this native choice have an instantiation for (ta, tb) at this shape, on a GPU?
template <typename T>
bool oracle_native(const C& c, Transpose ta, Transpose tb, int m, int n, int k) {
    if (m < 1 || n < 1 || k < 1) return false;
    return std::visit(overloaded{
        [](og::Direct) { return true; },
        [](og::Tiled) { return true; },
        [&](og::Small) { return !kCx<T> && std::max({m, n, k}) <= 64; },
        [&](const og::Reg& r) {
            const RegRow* row = reg_row(r);
            const std::string f = reg_form(ta, tb);
            return std::is_same_v<T, float> && row &&
                   std::find(row->forms.begin(), row->forms.end(), f) != row->forms.end();
        },
        [&](const og::Wide& w) {
            const WideRow* row = wide_row(w);
            return row && row->trace.count(wide_form<T>(ta, tb)) > 0;
        },
        [](og::Vendor) { return false; },
    }, c);
}

// The kernel-trace name the pinned choice must launch; "" for the vendor (no SYCL kernel).
template <typename T>
std::string expected_kernel(const C& c, Transpose ta, Transpose tb, bool aligned_nn) {
    return std::visit(overloaded{
        [](og::Direct) { return std::string("gemm_sycl_direct"); },
        [](og::Tiled) { return std::string("gemm_sycl_tiled16"); },
        [](og::Small) { return std::string("gemm_sycl_small_batched"); },
        [&](const og::Reg& r) {
            const RegRow* row = reg_row(r);
            std::string f = reg_form(ta, tb);
            if (f == "NN" && aligned_nn && row->trace.count("NN+aligned")) f = "NN+aligned";
            return row->trace.at(f);
        },
        [&](const og::Wide& w) { return wide_row(w)->trace.at(wide_form<T>(ta, tb)); },
        [](og::Vendor) { return std::string(); },
    }, c);
}

// ---- problems --------------------------------------------------------------------------

// One operand's footprint inside the shared buffer: item `it` starts at off + it * stride.
struct Region {
    std::size_t off = 0;
    int rows = 0, cols = 0, ld = 0, stride = 0;
};

enum class Layout { Packed, Strided };

struct Spec {
    Transpose ta = kN, tb = kN;
    int m = 8, n = 8, k = 8, batch = 3;
    Layout layout = Layout::Strided;
    int period = 0;        // > 0: item it repeats item it % period
    bool beta_zero = false;
    bool nan_c = false;    // C holds NaN (with beta_zero: a kernel that reads C returns NaN)
    unsigned seed = 1;
};

// A, B and C live in one buffer of a large finite poison. Packed: contiguous operands with
// 16-byte bases (layout=packed). Strided: ld = rows + 3, an odd batch stride, odd bases. The
// panel form carves A, B and C from one parent at its ld and stride, B aliasing A.
template <typename T>
struct Problem {
    Spec s;
    T alpha{}, beta{};
    Region a, b, c;
    UnifiedVector<T> mem;
    std::vector<T> mem0;
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
    return mk<T>(RealOf<T>(-999), RealOf<T>(777));
}

template <typename T>
void init(Problem<T>& p, std::size_t total) {
    using R = RealOf<T>;
    const Spec& s = p.s;
    p.alpha = mk<T>(R(1.25), R(-0.5));
    p.beta = s.beta_zero ? T(0) : mk<T>(R(-0.75), R(0.25));
    p.mem = UnifiedVector<T>(total, poison<T>());
    p.pa = UnifiedVector<T*>(s.batch, nullptr);
    p.pb = UnifiedVector<T*>(s.batch, nullptr);
    p.pc = UnifiedVector<T*>(s.batch, nullptr);
    std::mt19937 gen(s.seed);
    std::uniform_real_distribution<R> u(R(-1), R(1));
    const int reps = s.period > 0 ? std::min(s.period, s.batch) : s.batch;
    const R nan = std::numeric_limits<R>::quiet_NaN();
    for (int it = 0; it < s.batch; ++it) {
        const int r = it % reps;
        auto put = [&](const Region& g, bool is_c) {
            for (int j = 0; j < g.cols; ++j)
                for (int i = 0; i < g.rows; ++i) {
                    T& v = p.mem[Problem<T>::at(g, it, i, j)];
                    if (it != r) v = p.mem[Problem<T>::at(g, r, i, j)];
                    else v = is_c && s.nan_c ? mk<T>(nan, nan) : mk<T>(u(gen), u(gen));
                }
        };
        put(p.a, false);
        if (p.b.off != p.a.off) put(p.b, false);
        put(p.c, true);
    }
    p.mem0.assign(p.mem.begin(), p.mem.end());
}

std::size_t up4(std::size_t v) { return (v + 3) / 4 * 4; }

template <typename T>
Problem<T> make_problem(const Spec& s) {
    Problem<T> p;
    p.s = s;
    const int ar = s.ta == kN ? s.m : s.k, ac = s.ta == kN ? s.k : s.m;
    const int br = s.tb == kN ? s.k : s.n, bc = s.tb == kN ? s.n : s.k;
    auto region = [&](std::size_t off, int rows, int cols) {
        if (s.layout == Layout::Packed) return Region{up4(off), rows, cols, rows, rows * cols};
        return Region{off | 1, rows, cols, rows + 3, (rows + 3) * cols + 5};
    };
    auto end = [&](const Region& g) { return g.off + std::size_t(g.stride) * s.batch + 9; };
    p.a = region(8, ar, ac);
    p.b = region(end(p.a), br, bc);
    p.c = region(end(p.b), s.m, s.n);
    init(p, end(p.c) + 4);
    return p;
}

// potrf's trailing update in one parent: A22 -= L21 L21^H, so B is A's own region (read twice)
// and C sits below-right of it; all three carry the parent's ld and stride.
template <typename T>
Problem<T> make_panel(Spec s, int parent) {
    s.tb = kCx<T> ? kC : kT;
    s.ta = kN;
    s.layout = Layout::Strided;
    Problem<T> p;
    p.s = s;
    const int ld = parent + 5, stride = ld * parent + 9, r0 = 3;
    p.a = {std::size_t(r0 + s.k), s.m, s.k, ld, stride};                     // L21: rows r0+k.., cols 0..k
    p.b = p.a;                                                                // B = L21 read as ConjTrans
    p.c = {std::size_t(s.k) * ld + r0 + s.k, s.m, s.n, ld, stride};          // A22
    EXPECT_EQ(s.m, s.n) << "a panel update is square";
    init(p, std::size_t(stride) * s.batch);
    return p;
}

template <typename T>
std::complex<double> op_at(const Problem<T>& p, const Region& g, Transpose t, int it, int i, int j) {
    const std::complex<double> v = t == kN ? up(p.mem0[Problem<T>::at(g, it, i, j)]) : up(p.mem0[Problem<T>::at(g, it, j, i)]);
    return t == kC ? std::conj(v) : v;
}

template <typename T>
double eps() {
    return double(std::numeric_limits<RealOf<T>>::epsilon());
}

// Item `it` against a complex<double> reference, componentwise against the BLAS error bound
// (k+2) eps (|alpha| |op(A)| |op(B)| + |beta| |C0|); a C read at beta = 0 is not in the reference.
template <typename T>
::testing::AssertionResult item_ok(const Problem<T>& p, int it) {
    const Spec& s = p.s;
    const std::complex<double> al = up(p.alpha), be = up(p.beta);
    const double tol = 4.0 * (s.k + 2) * eps<T>();
    for (int j = 0; j < s.n; ++j)
        for (int i = 0; i < s.m; ++i) {
            std::complex<double> acc = 0.0;
            double mag = 0;
            for (int l = 0; l < s.k; ++l) {
                const auto x = op_at(p, p.a, s.ta, it, i, l), y = op_at(p, p.b, s.tb, it, l, j);
                acc += x * y;
                mag += std::abs(x) * std::abs(y);
            }
            std::complex<double> ref = al * acc;
            double bound = std::abs(al) * mag;
            if (!s.beta_zero) {
                const auto c0 = up(p.mem0[Problem<T>::at(p.c, it, i, j)]);
                ref += be * c0;
                bound += std::abs(be) * std::abs(c0);
            }
            const auto got = up(p.mem[Problem<T>::at(p.c, it, i, j)]);
            if (!std::isfinite(got.real()) || !std::isfinite(got.imag()) || std::abs(got - ref) > tol * bound + 1e-30)
                return ::testing::AssertionFailure() << "item " << it << " C(" << i << "," << j << ") = " << got
                                                     << ", expected " << ref << " (bound " << tol * bound << ")";
        }
    return ::testing::AssertionSuccess();
}

// The checked items are right, nothing outside C's footprint moved, and a repeating batch is
// bit-identical to its representatives.
template <typename T>
void expect_gemm(const Problem<T>& p, const std::string& what) {
    const Spec& s = p.s;
    std::vector<int> items;
    if (s.period > 0) {
        for (int it = 0; it < std::min(s.period, s.batch); ++it) items.push_back(it);
    } else if (s.batch <= 8) {
        for (int it = 0; it < s.batch; ++it) items.push_back(it);
    } else {
        items = {0, 1, s.batch / 2, s.batch - 1};
    }
    for (int it : items) ASSERT_TRUE(item_ok(p, it)) << what;
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

std::string label(const Spec& s) {
    return std::string(tw(s.ta)) + tw(s.tb) + (s.layout == Layout::Packed ? " packed" : " strided") +
           " m=" + std::to_string(s.m) + " n=" + std::to_string(s.n) + " k=" + std::to_string(s.k) +
           " batch=" + std::to_string(s.batch) + (s.beta_zero ? " beta=0" : "");
}
std::string name(const C& c, const Spec& s) { return select::to_string(c) + " " + label(s); }

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
    static constexpr bool kVendor = dispatch::level3_vendor_available<B>;

    void SetUp() override {
        test_utils::BatchLASTest<Config>::SetUp();
        if (this->HasFatalFailure() || ::testing::Test::IsSkipped()) return;
        if (!this->ctx) GTEST_SKIP() << "no queue";
        if (this->ctx->device().type != DeviceType::GPU) GTEST_SKIP() << "the native families are GPU kernels here";
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

    // The family's own launcher, called directly; false when it refuses (batchlas::unsupported).
    bool launch(const C& c, Problem<T>& p, std::string* why = nullptr) {
        Queue& q = *this->ctx;
        const MVof<T> A = p.A(), Bm = p.B(), Cm = p.Cv();
        const Transpose ta = p.s.ta, tb = p.s.tb;
        try {
            const bool launched = std::visit(overloaded{
                [&](og::Direct) { (void)sycl_gemm::gemm_direct<T>(q, A, Bm, Cm, p.alpha, p.beta, ta, tb); return true; },
                [&](og::Tiled) { (void)sycl_gemm::gemm_tiled<T>(q, A, Bm, Cm, p.alpha, p.beta, ta, tb); return true; },
                [&](og::Small) {
                    if constexpr (kCx<T>) {
                        if (why) *why = "small has no complex instantiation";
                        return false;
                    } else {
                        (void)sycl_gemm::gemm_small<T>(q, A, Bm, Cm, p.alpha, p.beta, ta, tb);
                        return true;
                    }
                },
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
// 32) and the unit shape.
const int kShapes[][3] = {{128, 128, 64}, {100, 70, 37}, {48, 48, 48}, {33, 17, 5}, {200, 32, 96}, {1, 1, 1}};

// §8.1: each candidate on every (ta, tb), both layouts, every shape. Accepted shapes run and
// are right; refused ones throw from the pin. The oracle is the old KernelVariant list.
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
                            expect_gemm(p, what + " (vendor-free: Auto)");
                            continue;
                        }
                        if (!TestFixture::expect_runs(c, s)) {
                            EXPECT_FALSE(this->pin_accepted(c, p)) << what << " was accepted";
                            ++refused;
                            continue;
                        }
                        ASSERT_TRUE(this->pin_accepted(c, p)) << what << " was refused";
                        expect_gemm(p, what);
                        if (::testing::Test::HasFatalFailure()) return;
                        ++ran;
                    }
    EXPECT_GT(ran, 0);
    EXPECT_GT(refused, 0);
}

// The guard for the 18 register tiles that used to compute NN on a transposed call: every
// (config, form) the oracle has no instantiation for throws from the pin as a can_run refusal
// (not from the launcher), and C is left bit for bit.
TYPED_TEST(GemmCandidates, TransposedPinOnAMissingInstantiationThrows) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    int checked = 0;
    for (const C& c : og::candidates<T>()) {
        if (!std::holds_alternative<og::Reg>(c) && !std::holds_alternative<og::Wide>(c)) continue;
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
    if constexpr (std::is_same_v<T, float>) EXPECT_GE(checked, 18 * 4) << "every NN-only reg tile on every transposed form";
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
                expect_gemm(pinned, name(c, s));
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
// on every instantiated form, and the 128x32x32 NN leg derived from the layout (aligned on a
// packed tile multiple, predicated on a strided view). A swapped launch arm shows here even
// where both kernels give the same bits. threadsafe: the child re-executes the binary with
// BATCHLAS_KERNEL_TRACE set from the start, so its queue is created with profiling.
TYPED_TEST(GemmCandidates, PinnedChoiceLaunchesItsOwnKernel) {
    using T = typename TestFixture::T;
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    struct Case { C c; Spec s; std::string kernel; };
    std::vector<Case> cases;
    for (const C& c : og::candidates<T>()) {
        if (std::holds_alternative<og::Vendor>(c)) continue;
        for (Transpose ta : kForms)
            for (Transpose tb : kForms) {
                const bool small = std::holds_alternative<og::Small>(c);
                for (Layout lay : {Layout::Packed, Layout::Strided}) {
                    Spec s{ta, tb, small ? 48 : 128, small ? 40 : 64, small ? 33 : 64, 2, lay};
                    if (!oracle_native<T>(c, ta, tb, s.m, s.n, s.k)) continue;
                    cases.push_back({c, s, expected_kernel<T>(c, ta, tb, lay == Layout::Packed)});
                }
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

// Sub-views carrying the parent's ld and stride, as potrf's trailing update issues them
// (A22 -= L21 L21^H, B the same memory as A), on every candidate that runs the form. The
// whole parent outside A22 stays bit for bit.
TYPED_TEST(GemmCandidates, ParentLdPanelUpdate) {
    using T = typename TestFixture::T;
    int ran = 0;
    for (const C& c : og::candidates<T>())
        for (auto [mn, k] : {std::pair{96, 32}, std::pair{61, 16}, std::pair{128, 64}}) {
            Spec s{kN, kN, mn, mn, k, 3, Layout::Strided};
            s.seed = 300u + mn;
            auto p = make_panel<T>(s, mn + k + 8);
            if (!TestFixture::expect_runs(c, p.s) || TestFixture::vendor_word_falls_back(c)) continue;
            this->run_pinned(c, p);
            expect_gemm(p, name(c, p.s) + " (parent-ld panel)");
            ++ran;
        }
    EXPECT_GT(ran, 0);
}

// beta = 0 with C full of NaN: BLAS does not read C, so the result must be finite and right.
// direct, tiled and the LinearEpilogue register tiles compute beta * prior unconditionally
// (known-defects.md #11, unfixed); they are run on a finite C instead and listed, so a fix
// shows up here as an entry to delete.
TYPED_TEST(GemmCandidates, BetaZeroDoesNotReadC) {
    using T = typename TestFixture::T;
    const std::vector<std::string> reads_c{"direct", "tiled", "reg:m=32:n=32:k=8:u=1", "reg:m=64:n=64:k=8:u=1",
                                           "reg:m=64:n=64:k=16:u=1", "reg:m=128:n=32:k=16:u=1",
                                           "reg:m=128:n=32:k=32:u=1", "reg:m=128:n=64:k=16:u=1",
                                           "reg:m=32:n=128:k=16:u=1", "reg:m=128:n=64:k=32:u=4",
                                           "reg:m=128:n=64:k=32:u=2"};
    int nan_checked = 0;
    for (const C& c : og::candidates<T>()) {
        const std::string sp = select::to_string(c);
        const bool known = std::find(reads_c.begin(), reads_c.end(), sp) != reads_c.end();
        for (Transpose ta : {kN, kC})
            for (auto [m, n, k] : {std::tuple{128, 128, 64}, std::tuple{45, 51, 38}}) {
                Spec s{ta, kN, m, n, k, 3, Layout::Strided};
                s.beta_zero = true;
                s.nan_c = !known;
                if (!TestFixture::expect_runs(c, s) || TestFixture::vendor_word_falls_back(c)) continue;
                auto p = make_problem<T>(s);
                this->run_pinned(c, p);
                expect_gemm(p, name(c, s) + (known ? " (finite C: known-defects #11)" : " (NaN C)"));
                if (::testing::Test::HasFatalFailure()) return;
                nan_checked += !known;
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
        for (Transpose ta : {kN, kT})
            for (Transpose tb : {kN, kC}) {
                Spec s{ta, tb, small ? 45 : 72, small ? 40 : 40, small ? 29 : 37, kBatch, Layout::Strided};
                s.period = 7;
                s.seed = 777u;
                if (!TestFixture::expect_runs(c, s) || TestFixture::vendor_word_falls_back(c)) continue;
                auto p = make_problem<T>(s);
                this->run_pinned(c, p);
                expect_gemm(p, name(c, s));
                if (::testing::Test::HasFatalFailure()) return;
            }
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
    struct Bad { const char* what; int a_batch, c_rows; };
    for (const Bad& bad : {Bad{"A.batch != C.batch", 2, 16}, Bad{"C.rows != m", 3, 15}})
        for (const auto& c : pins) {
            const MVof<T> A(p.mem.data() + p.a.off, 16, 8, p.a.ld, p.a.stride, bad.a_batch);
            const MVof<T> Cm(p.mem.data() + p.c.off, bad.c_rows, 12, p.c.ld, p.c.stride, 3);
            std::optional<Pin> pin;
            if (c) pin.emplace("gemm", *c);
            EXPECT_THROW(((void)gemm<B, T>(*this->ctx, A, p.B(), Cm, p.alpha, p.beta, kN, kN)), std::invalid_argument)
                << bad.what << " under " << (c ? select::to_string(*c) : std::string("auto"));
        }
    for (std::size_t e = 0; e < p.mem.size(); ++e)
        ASSERT_TRUE(same_bits(p.mem[e], p.mem0[e])) << "a refused call wrote element " << e;
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
        expect_gemm(p, "auto, non-default precision");
    } else {
        EXPECT_THROW(this->run(p, prec), dispatch::NoRouteError);
    }
}

// R6: a spelling that names nothing compiled throws instead of meaning Auto, including the
// deleted variants' legacy names and the design doc's never-compiled `tiled:tile=64:k=8`.
TYPED_TEST(GemmCandidates, UnknownAndDeletedPinsThrow) {
    using T = typename TestFixture::T;
    const std::vector<std::string> words{
        "bogus", "tiled:tile=64:k=8", "reg:m=1:n=1:k=1:u=1", "wide:m=64:n=64", "native:vendor", "direct:1",
        "128x32x32_s2_u2", "128x32x32_s1_u1", "128x32x32_persistent", "128x32x32_splitk4",
        "128x32x32_s2_u2_tt8x4", "128x32x32_s2_u2_tt4x8", "reg128x64k32largett4x8", "reg128x64k32largett4x8u2",
        std::is_same_v<T, float> ? "reg:m=64:n=64:k=8:u=2" : "reg:m=32:n=32:k=8:u=1",
        kCx<T> ? "small" : "wide:m=16:n=16:k=8"};
    for (const std::string& w : words) {
        auto p = make_problem<T>(Spec{kN, kN, 16, 16, 16, 2, Layout::Packed});
        const Pin pin("gemm", std::string_view(w));
        EXPECT_THROW(this->run(p), std::invalid_argument) << w;
    }
    // The retired variable fails loudly instead of being ignored by an old script.
    auto p = make_problem<T>(Spec{kN, kN, 16, 16, 16, 2, Layout::Packed});
    const ScopedEnvVar old("BATCHLAS_GEMM_SYCL_KERNEL", "tiled16");
    EXPECT_THROW(this->run(p), std::invalid_argument);
}

// §5.3: the legacy BATCHLAS_GEMM_SYCL_KERNEL names (now BATCHLAS_GEMM_ROUTE aliases) and the
// class words, via ScopedPin and via BATCHLAS_GEMM_ROUTE. A transposed variant's name means its
// config; the form is the call's.
TYPED_TEST(GemmCandidates, LegacyKernelNamesAndClassWords) {
    using T = typename TestFixture::T;
    const ScopedEnvVar clear_route("BATCHLAS_GEMM_ROUTE", nullptr);
    const ScopedEnvVar clear_variant("BATCHLAS_GEMM_VARIANT", nullptr);
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
        {"tiled16", nn, "tiled"}, {"tile16", nn, "tiled"}, {"TILED", nn, "tiled"}, {"native:tiled", nn, "tiled"},
        {"direct", tt, "direct"}, {"64x64x16wide", nn, "wide:m=64:n=64:k=16"},
        {"64x64x16wide_cn", Spec{kC, kN, 64, 48, 32, 4, Layout::Packed}, "wide:m=64:n=64:k=16"},
        {"128x32x16wide_nc", Spec{kN, kC, 64, 48, 32, 4, Layout::Packed}, "wide:m=128:n=32:k=16"},
        {"32x32x16wide", nn, "wide:m=32:n=32:k=16"}, {"16x16x16wide", nn, "wide:m=16:n=16:k=16"},
        {"register_tiled", nn, native_best}, {"native:register_tiled", nn, native_best}, {"sycl", nn, native_best},
        {"custom", nn, native_best}, {"native", nn, native_best}, {"native:auto", nn, native_best},
        {"vendor", nn, vendor_pick}, {"vendor:auto", nn, vendor_pick}, {"auto", nn, auto_pick},
        {"auto:auto", nn, auto_pick}};
    if constexpr (!kCx<T>) want.push_back({"smallbatched", nn, "small"});
    if constexpr (std::is_same_v<T, float>) {
        const std::vector<Want> reg{
            {"reg32", nn, "reg:m=32:n=32:k=8:u=1"}, {"64x64", nn, "reg:m=64:n=64:k=8:u=1"},
            {"reg64k16tt", tt, "reg:m=64:n=64:k=16:u=1"}, {"64x64x16tn", Spec{kT, kN, 64, 48, 32, 4, Layout::Packed},
                                                           "reg:m=64:n=64:k=16:u=1"},
            {"reg128x32k16nt", Spec{kN, kT, 64, 48, 32, 4, Layout::Packed}, "reg:m=128:n=32:k=16:u=1"},
            {"128x32x32_s2_u1_aligned", nn, "reg:m=128:n=32:k=32:u=1"},
            {"128x32x32_s2_u1_generic", nn, "reg:m=128:n=32:k=32:u=1"},
            {"128x32x32_s2_u1_tn", Spec{kT, kN, 64, 48, 32, 4, Layout::Packed}, "reg:m=128:n=32:k=32:u=1"},
            {"reg128x64k16tt", tt, "reg:m=128:n=64:k=16:u=1"}, {"32x128x16tt", tt, "reg:m=32:n=128:k=16:u=1"},
            {"reg128x64k32large", nn, "reg:m=128:n=64:k=32:u=4"},
            {"128x64x32large_u2", nn, "reg:m=128:n=64:k=32:u=2"}, {"128x128x8", nn, "reg:m=128:n=128:k=8:u=1"}};
        want.insert(want.end(), reg.begin(), reg.end());
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
            expect_gemm(p, what);
        }
}

// BATCHLAS_GEMM_VARIANT keeps its own vocabulary, on that variable only: its `native` was the
// raw cuBLAS call (vendor), while BATCHLAS_GEMM_ROUTE=native is the native family; `sycl` and
// `custom` mean native; and a set BATCHLAS_GEMM_ROUTE wins over it.
TYPED_TEST(GemmCandidates, LegacyVariantVocabularyStaysOnItsVariable) {
    using T = typename TestFixture::T;
    const ScopedEnvVar clear_route("BATCHLAS_GEMM_ROUTE", nullptr);
    const Spec s{kN, kN, 64, 48, 32, 4, Layout::Packed};
    const std::string auto_pick = this->auto_choice(s);
    const std::string vendor_pick = TestFixture::kVendor ? "vendor" : auto_pick;
    std::string native_pick;
    {
        auto p = make_problem<T>(s);
        const ScopedEnvVar route("BATCHLAS_GEMM_ROUTE", "native");
        native_pick = traced_choice([&] { this->run(p); });
    }
    EXPECT_NE(native_pick, "vendor") << "BATCHLAS_GEMM_ROUTE=native is the native family";
    struct Case { const char* route; const char* variant; std::string expect; };
    const Case cases[] = {{nullptr, "native", vendor_pick},  {nullptr, "cuda-native", vendor_pick},
                          {nullptr, "cublasdx", vendor_pick}, {nullptr, "vendor", vendor_pick},
                          {nullptr, "sycl", native_pick},     {nullptr, "custom", native_pick},
                          {nullptr, "auto", auto_pick},       {"native", "native", native_pick},
                          {"tiled", "vendor", "tiled"}};
    for (const Case& k : cases) {
        auto p = make_problem<T>(s);
        const ScopedEnvVar route("BATCHLAS_GEMM_ROUTE", k.route);
        const ScopedEnvVar variant("BATCHLAS_GEMM_VARIANT", k.variant);
        EXPECT_EQ(traced_choice([&] { this->run(p); }), k.expect)
            << "ROUTE=" << (k.route ? k.route : "<unset>") << " VARIANT=" << k.variant;
        expect_gemm(p, std::string("VARIANT=") + k.variant);
    }
}

// §5.3: a ScopedPin wins over BATCHLAS_GEMM_ROUTE, and nested pins restore the outer one.
TYPED_TEST(GemmCandidates, ScopedPinBeatsTheEnvironment) {
    using T = typename TestFixture::T;
    const Spec tn{kT, kN, 40, 30, 20, 2, Layout::Strided};
    const ScopedEnvVar env("BATCHLAS_GEMM_ROUTE", "wide:m=32:n=32:k=16");
    {
        auto p = make_problem<T>(tn);
        EXPECT_THROW(this->run(p), std::invalid_argument) << "the environment pin was not read";
    }
    auto p = make_problem<T>(tn);
    EXPECT_EQ(traced_choice([&] {
                  const Pin pin("gemm", C{og::Tiled{}});
                  this->run(p);
              }),
              "tiled");
    expect_gemm(p, "tiled over the env pin");
    const Pin outer("gemm", C{og::Direct{}});
    {
        const Pin inner("gemm", C{og::Tiled{}});
        auto r = make_problem<T>(tn);
        EXPECT_EQ(traced_choice([&] { this->run(r); }), "tiled");
    }
    auto r = make_problem<T>(tn);
    EXPECT_EQ(traced_choice([&] { this->run(r); }), "direct") << "the inner pin did not restore the outer";
    expect_gemm(r, "outer direct");
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
    select::testing::set_builtin_tables(with_gemm_table(dtype, dev, rows));
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
        expect_gemm(p, std::string("the ") + k.field + " probe");
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
        expect_gemm(p, line);
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
}

// Auto against the sm_89 transcribed table on a device that reads it (its own, or borrowed):
// the old router's choice at grid cells, the vendor where it ranks first (the first runnable
// native entry vendor-free).
TYPED_TEST(GemmCandidates, AutoReadsTheSm89TranscribedTable) {
    using T = typename TestFixture::T;
    const ScopedEnvVar clear("BATCHLAS_GEMM_ROUTE", nullptr);
    const auto tables = select::tables_in_borrow_order("gemm", select::dtype_name<T>(),
                                                       select::device_of<TestFixture::B>(*this->ctx));
    if (tables.empty() || tables.front()->device != "sm_89") GTEST_SKIP() << "this device does not read sm_89";
    struct Row { Spec s; std::string with_vendor, without; };
    std::vector<Row> rows;
    if constexpr (std::is_same_v<T, float>)
        rows = {{{kN, kN, 32, 32, 32, 128, Layout::Packed}, "small", "small"},
                {{kN, kN, 128, 128, 128, 128, Layout::Packed}, "vendor", "reg:m=128:n=128:k=8:u=1"},
                {{kT, kN, 256, 128, 128, 128, Layout::Strided}, "vendor", "reg:m=128:n=32:k=32:u=1"},
                {{kN, kT, 256, 32, 96, 2048, Layout::Strided}, "vendor", "tiled"}};
    else if constexpr (std::is_same_v<T, double>)
        rows = {{{kN, kN, 16, 16, 16, 128, Layout::Packed}, "direct", "direct"},
                {{kN, kN, 256, 256, 256, 128, Layout::Packed}, "wide:m=64:n=64:k=16", "wide:m=64:n=64:k=16"},
                {{kN, kC, 256, 32, 96, 2048, Layout::Strided}, "tiled", "tiled"}};
    else
        rows = {{{kN, kC, 256, 32, 96, 2048, Layout::Strided}, "vendor", "wide:m=128:n=32:k=16"},
                {{kC, kN, 32, 256, 128, 2048, Layout::Strided}, "vendor", "wide:m=32:n=128:k=16"},
                {{kN, kN, 256, 256, 256, 128, Layout::Strided}, "vendor", "wide:m=64:n=64:k=16"}};
    for (const Row& r : rows) {
        auto p = make_problem<T>(r.s);
        EXPECT_EQ(traced_choice([&] { this->run(p); }), TestFixture::kVendor ? r.with_vendor : r.without) << label(r.s);
        expect_gemm(p, "auto " + label(r.s));
    }
}

// With a table naming only the vendor, a vendor-free build takes the last resort, direct.
TYPED_TEST(GemmCandidates, VendorFreeLastResortIsDirect) {
    using T = typename TestFixture::T;
    const ScopedEnvVar clear("BATCHLAS_GEMM_ROUTE", nullptr);
    const std::string dtype(select::dtype_name<T>());
    const std::string dev = select::device_of<TestFixture::B>(*this->ctx).key;
    const TableGuard restore;
    select::testing::set_builtin_tables(
        with_gemm_table(dtype, dev, "ta=N tb=N layout=packed m=64 n=64 k=64 batch=128 | vendor 1\n"));
    for (const Spec& s : {Spec{kN, kN, 64, 64, 64, 4, Layout::Packed}, Spec{kT, kC, 300, 5, 70, 2, Layout::Strided}}) {
        auto p = make_problem<T>(s);
        std::string all;
        EXPECT_EQ(traced_choice([&] { this->run(p); }, &all), TestFixture::kVendor ? "vendor" : "direct") << label(s);
        if (!TestFixture::kVendor) EXPECT_NE(all.find("last resort"), std::string::npos) << all;
        expect_gemm(p, "vendor-only table, " + label(s));
    }
}

// The coverage row (§5.6): the real backend, gemm's m, n, k, batch, transA and transB, the
// chosen spelling and the native flags. threadsafe: the child re-executes the binary.
TYPED_TEST(GemmCandidates, CoverageRowCarriesBackendKeyAndNativeFlags) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    const std::string dir = ::testing::TempDir() + "gemm_cov." + std::string(select::dtype_name<T>());
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    const std::string out = dir + "/cov";
    const Spec lo{kN, kC, 24, 10, 6, 2, Layout::Strided};
    const Spec hi{kT, kN, 40, 33, 17, 3, Layout::Packed};
    auto child = [&] {
        const ScopedEnvVar cov("BATCHLAS_COVERAGE_OUT", out.c_str());
        const ScopedEnvVar clear("BATCHLAS_GEMM_ROUTE", nullptr);
        dispatch::coverage::g_dynamic_enabled = true;
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
    const std::string backend = B == Backend::CUDA ? "CUDA" : (B == Backend::ROCM ? "ROCM" : "?");
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
    static constexpr bool kVendor = dispatch::level3_vendor_available<B>;
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
            expect_gemm(p, "cpu " + select::to_string(c));
        }
    }
    auto p = make_problem<T>(s);
    EXPECT_EQ(traced_choice([&] { call(p); }), kVendor ? "vendor" : "direct");
    expect_gemm(p, "cpu auto");
}

}  // namespace
