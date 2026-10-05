// Every gemv candidate, pinned: docs/design/flat-kernel-selection.md §8 and
// docs/design/flat-select-p5/gemv.md. The limit oracle reads the drivers' own rules; which
// kernel ran is read back from the select trace or a bit-for-bit comparison with the direct
// driver, never assumed from the pin being accepted.
#include <gtest/gtest.h>

#include <batchlas/backend_config.h>

#include <batchlas/blas/functions/gemv.hh>
#include "../src/select/coverage.hh"
#include <batchlas/no_route.hh>
#include "../src/select/vendor.hh"
#include <batchlas/util/env.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>
#include <batchlas/util/sycl-vector.hh>

#include "test_utils.hh"

#include "../src/ops/gemv/choice.hh"
#include "../src/sycl/gemv_native.hh"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <cstring>
#include <iostream>
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

namespace gv = batchlas::ops::gemv;
using C = gv::GemvChoice;
using Pin = select::ScopedPin<C>;
template <typename T>
using MVof = MatrixView<T, MatrixFormat::Dense>;

template <typename T>
using RealOf = typename batchlas::base_type<T>::type;
template <typename T>
constexpr bool kCx = test_utils::is_complex<T>::value;

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

const char* trans_s(Transpose t) { return t == Transpose::NoTrans ? "N" : (t == Transpose::Trans ? "T" : "C"); }

struct Spec {
    Transpose trans = Transpose::NoTrans;
    int m = 8, n = 5, batch = 3;  // A's stored extents
    int period = 0;               // > 0: item it repeats item it % period
    bool beta_zero = false;
    unsigned seed = 1;
    int out() const { return trans == Transpose::NoTrans ? m : n; }
    int red() const { return trans == Transpose::NoTrans ? n : m; }
};

// A, x and y in one buffer of a large finite poison, at a padded ld, inc 2 (x) and 3 (y) and
// batch strides that are none of the natural values. Everything gemv may not write holds poison.
template <typename T>
struct Problem {
    Spec s;
    T alpha{}, beta{};
    std::size_t a_off = 0, x_off = 0, y_off = 0;
    int lda = 0, sa = 0, incx = 2, sx = 0, incy = 3, sy = 0;
    UnifiedVector<T> mem;
    std::vector<T> mem0;

    MVof<T> A() { return {mem.data() + a_off, s.m, s.n, lda, sa, s.batch}; }
    VectorView<T> X() { return {mem.data() + x_off, s.red(), s.batch, Inc{incx}, Stride{sx}}; }
    VectorView<T> Y() { return {mem.data() + y_off, s.out(), s.batch, Inc{incy}, Stride{sy}}; }
    std::size_t ai(int it, int i, int j) const { return a_off + std::size_t(it) * sa + std::size_t(j) * lda + i; }
    std::size_t xi(int it, int i) const { return x_off + std::size_t(it) * sx + std::size_t(i) * incx; }
    std::size_t yi(int it, int i) const { return y_off + std::size_t(it) * sy + std::size_t(i) * incy; }
};

template <typename T>
T poison() {
    return mk<T>(RealOf<T>(-999), RealOf<T>(777));
}

template <typename T>
Problem<T> make_problem(const Spec& s) {
    using R = RealOf<T>;
    Problem<T> p;
    p.s = s;
    p.alpha = mk<T>(R(1.5), R(-0.5));
    p.beta = s.beta_zero ? T(0) : mk<T>(R(-0.75), R(0.25));
    p.lda = s.m + 3;
    p.sa = p.lda * std::max(s.n, 1) + 5;
    p.sx = p.incx * std::max(s.red(), 1) + 7;
    p.sy = p.incy * std::max(s.out(), 1) + 11;
    p.x_off = std::size_t(p.sa) * s.batch + 13;
    p.y_off = p.x_off + std::size_t(p.sx) * s.batch + 17;
    p.mem = UnifiedVector<T>(p.y_off + std::size_t(p.sy) * s.batch + 19, poison<T>());
    std::mt19937 gen(s.seed);
    std::uniform_real_distribution<R> u(R(-1), R(1));
    const int reps = s.period > 0 ? std::min(s.period, s.batch) : s.batch;
    for (int it = 0; it < s.batch; ++it) {
        const int r = it % reps;
        for (int j = 0; j < s.n; ++j)
            for (int i = 0; i < s.m; ++i)
                p.mem[p.ai(it, i, j)] = it < reps ? mk<T>(u(gen), u(gen)) : p.mem[p.ai(r, i, j)];
        for (int i = 0; i < s.red(); ++i) p.mem[p.xi(it, i)] = it < reps ? mk<T>(u(gen), u(gen)) : p.mem[p.xi(r, i)];
        // beta == 0 must not read y: its old contents stay poison.
        if (!s.beta_zero)
            for (int i = 0; i < s.out(); ++i)
                p.mem[p.yi(it, i)] = it < reps ? mk<T>(u(gen), u(gen)) : p.mem[p.yi(r, i)];
    }
    p.mem0.assign(p.mem.begin(), p.mem.end());
    return p;
}

template <typename T>
double tol(int red) {
    return 32.0 * std::max(red, 1) * double(std::numeric_limits<RealOf<T>>::epsilon());
}

// y = alpha op(A) x + beta y against a double host reference on the checked items, every element
// outside y's footprint bit for bit, and a repeating batch bit-identical to its representative.
template <typename T>
void expect_gemv(const Problem<T>& p, const std::string& what) {
    const Spec& s = p.s;
    std::vector<int> items;
    if (s.period > 0) for (int it = 0; it < std::min(s.period, s.batch); ++it) items.push_back(it);
    else if (s.batch <= 8) for (int it = 0; it < s.batch; ++it) items.push_back(it);
    else items = {0, 1, s.batch / 2, s.batch - 1};
    const std::complex<double> al = up(p.alpha), be = up(p.beta);
    // Reference-BLAS quick return: m == 0 or n == 0 leaves y untouched, even when out > 0.
    const bool quick = s.m == 0 || s.n == 0;
    for (int it : items)
        if (quick)
            for (int o = 0; o < s.out(); ++o)
                ASSERT_TRUE(same_bits(p.mem[p.yi(it, o)], p.mem0[p.yi(it, o)])) << what << ": quick return wrote y";
    for (int it : quick ? std::vector<int>{} : items)
        for (int o = 0; o < s.out(); ++o) {
            std::complex<double> acc = 0.0;
            double mag = 0.0;
            for (int r = 0; r < s.red(); ++r) {
                std::complex<double> a = s.trans == Transpose::NoTrans ? up(p.mem0[p.ai(it, o, r)])
                                                                       : up(p.mem0[p.ai(it, r, o)]);
                if (s.trans == Transpose::ConjTrans) a = std::conj(a);
                acc += a * up(p.mem0[p.xi(it, r)]);
                mag += std::abs(a) * std::abs(up(p.mem0[p.xi(it, r)]));
            }
            const std::complex<double> y0 = s.beta_zero ? 0.0 : up(p.mem0[p.yi(it, o)]);
            const std::complex<double> want = al * acc + be * y0;
            const double err = std::abs(up(p.mem[p.yi(it, o)]) - want);
            const double scale = std::abs(al) * mag + std::abs(be) * std::abs(y0) + 1e-300;
            ASSERT_TRUE(std::isfinite(err) && err <= tol<T>(s.red()) * std::max(scale, 1.0))
                << what << " item " << it << " y[" << o << "] err " << err << " scale " << scale;
        }
    std::vector<char> iny(p.mem.size(), 0);
    for (int it = 0; it < s.batch; ++it)
        for (int o = 0; o < s.out(); ++o) iny[p.yi(it, o)] = 1;
    for (std::size_t e = 0; e < p.mem.size(); ++e)
        if (!iny[e]) ASSERT_TRUE(same_bits(p.mem[e], p.mem0[e])) << what << ": wrote outside y at element " << e;
    if (s.period > 0)
        for (int it = s.period; it < s.batch; ++it)
            for (int o = 0; o < s.out(); ++o)
                ASSERT_TRUE(same_bits(p.mem[p.yi(it, o)], p.mem[p.yi(it % s.period, o)]))
                    << what << ": y of item " << it << " differs from its representative at " << o;
}

std::string label(const Spec& s) {
    return std::string(trans_s(s.trans)) + " m=" + std::to_string(s.m) + " n=" + std::to_string(s.n) +
           " batch=" + std::to_string(s.batch) + (s.beta_zero ? " beta=0" : "");
}
std::string name(const C& c, const Spec& s) { return select::to_string(c) + " " + label(s); }

// The outermost gemv trace line for whatever `run` calls, and the choice it names.
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
        if (line.rfind("gemv ", 0) == 0 && line.find(" -> ") != std::string::npos) return line;
    return "<no gemv trace line in: " + err + ">";
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

// The device's own gemv table replaced by `rows` under gemv's key line; other tables kept.
template <typename T, Backend B>
void install_gemv_table(Queue& q, const std::string& rows) {
    const std::string dtype(select::dtype_name<T>());
    const std::string dev = select::device_of<B>(q).key;
    std::vector<std::pair<std::string, std::string>> files;
    for (const auto& t : select::embedded_tables())
        if (t.name.rfind("gemv.", 0) != 0) files.emplace_back(std::string(t.name), std::string(t.text));
    files.emplace_back("gemv." + dtype + "." + dev + ".txt", "# op=gemv dtype=" + dtype + " device=" + dev +
                                                                  " kernels=unknown\n"
                                                                  "# keys: trans:exact out:log red:log batch:log\n" +
                                                                  rows);
    select::testing::set_builtin_tables(std::move(files));
}

// known-defects #13: cublasZgemvStridedBatched segfaults inside some BatchLAS processes
// (threadripper02). A child process running GemvVendorProbe.ZgemvRuns answers once whether it
// does here; where it crashes, the cases that would launch that vendor call skip by name.
bool zgemv_vendor_usable() {
    static const bool ok = [] {
        const std::string exe = std::filesystem::read_symlink("/proc/self/exe").string();
        const std::string cmd =
            "BATCHLAS_GEMV_VENDOR_PROBE=1 '" + exe + "' --gtest_filter=GemvVendorProbe.ZgemvRuns > /dev/null 2>&1";
        return std::system(cmd.c_str()) == 0;
    }();
    return ok;
}

TEST(GemvVendorProbe, ZgemvRuns) {
    if (!std::getenv("BATCHLAS_GEMV_VENDOR_PROBE")) GTEST_SKIP() << "the known-defects #13 probe, run as a child";
#if BATCHLAS_HAS_CUDA_BACKEND
    if constexpr (batchlas::select::level3_vendor_available<Backend::CUDA>) {
        using Z = std::complex<double>;
        Queue q(Device("gpu"), Backend::CUDA);
        UnifiedVector<Z> a(12, Z(1, 1)), x(3, Z(1, 0)), y(4, Z(0, 0));
        (void)backend::gemv_vendor<Backend::CUDA, Z>(q, MVof<Z>(a.data(), 4, 3, 4, 12, 1), VectorView<Z>(x.data(), 3, 1),
                                                     VectorView<Z>(y.data(), 4, 1), Z(1), Z(0), Transpose::NoTrans);
        q.wait();
    }
#endif
}

template <typename T, Backend B>
struct Cfg {
    using ScalarType = T;
    static constexpr Backend BackendVal = B;
};
using Types = typename test_utils::backend_types<Cfg>::type;

template <typename Config>
class GemvCandidates : public test_utils::BatchLASTest<Config> {
protected:
    using T = typename Config::ScalarType;
    static constexpr Backend B = Config::BackendVal;
    static constexpr bool kVendor = batchlas::select::level3_vendor_available<B>;

    void SetUp() override {
        test_utils::BatchLASTest<Config>::SetUp();
        if (this->HasFatalFailure() || ::testing::Test::IsSkipped()) return;
        if (!this->ctx) GTEST_SKIP() << "no queue";
        if (this->ctx->device().type != DeviceType::GPU) GTEST_SKIP() << "GPU fixture (GemvCandidatesCpu covers host)";
        if (!this->ctx->device().supports_sub_group_size(32)) GTEST_SKIP() << "no sub-group size 32";
        if (!vendor_launch_ok())
            std::cerr << "note: complex<double> vendor gemv crashes in this process (known-defects #13); "
                         "cases that would launch it are skipped\n";
    }

    // False where a vendor launch would hit known-defects #13 (asked of a child process once).
    static bool vendor_launch_ok() {
        if constexpr (kVendor && B == Backend::CUDA && std::is_same_v<T, std::complex<double>>)
            return zgemv_vendor_usable();
        return true;
    }
    static bool launches_crashing_vendor(const C& c) {
        return std::holds_alternative<gv::Vendor>(c) && kVendor && !vendor_launch_ok();
    }

    // ---- the limit oracle: the drivers' own rules, not gemv.cc's can_run ----
    static bool expect_runs(const C& c, Transpose t) {
        if (std::holds_alternative<gv::Cta>(c)) return t != Transpose::NoTrans;  // the driver throws on N
        if (std::holds_alternative<gv::Direct>(c)) return true;
        return kVendor;
    }
    static bool vendor_word_falls_back(const C& c) { return std::holds_alternative<gv::Vendor>(c) && !kVendor; }

    void run(Problem<T>& p) {
        (void)gemv<B, T>(*this->ctx, p.A(), p.X(), p.Y(), p.alpha, p.beta, p.s.trans);
        this->ctx->wait();
    }
    void run_pinned(const C& c, Problem<T>& p) {
        const Pin pin("gemv", c);
        run(p);
    }
    bool pin_accepted(const C& c, Problem<T>& p) {
        const Pin pin("gemv", c);
        try {
            run(p);
            return true;
        } catch (const std::invalid_argument& e) {
            if (std::string(e.what()).find("cannot run this shape") == std::string::npos) throw;
            return false;
        }
    }
    std::string auto_choice(Problem<T>& p) {
        const ScopedEnvVar clear("BATCHLAS_GEMV_ROUTE", nullptr);
        return traced_choice([&] { run(p); });
    }

    // The family's own driver.
    bool direct(const C& c, Problem<T>& p, std::string* why = nullptr) {
        Queue& q = *this->ctx;
        try {
            if (std::holds_alternative<gv::Cta>(c)) {
                (void)sycl_gemv::gemv_native_cta<T>(q, p.A(), p.X(), p.Y(), p.alpha, p.beta, p.s.trans);
            } else if (std::holds_alternative<gv::Direct>(c)) {
                (void)sycl_gemv::gemv_native_direct<T>(q, p.A(), p.X(), p.Y(), p.alpha, p.beta, p.s.trans);
            } else {
                if constexpr (kVendor) {
                    (void)backend::gemv_vendor<B, T>(q, p.A(), p.X(), p.Y(), p.alpha, p.beta, p.s.trans);
                } else {
                    if (why) *why = "no vendor BLAS in this build";
                    return false;
                }
            }
            q.wait();
            return true;
        } catch (const std::exception& e) {
            if (why) *why = e.what();
            return false;
        }
    }
};

TYPED_TEST_SUITE(GemvCandidates, Types);

// Body-5 reduction gate per type (gemv_native.cc), and a batch whose out*batch clears gate 3's
// 64*CU floor on the largest device here, so both sides of the gate really run body 5 / body 3.
template <typename T>
int seg_trans_gate() {
    if constexpr (std::is_same_v<T, float>) return 32;
    else if constexpr (std::is_same_v<T, std::complex<float>>) return 16;
    else if constexpr (std::is_same_v<T, double>) return 48;
    else return 64;
}

// §8.1: every candidate, every trans, on shapes straddling the drivers' internal splits (body 4
// out <= 16, body 5's red gate), the degenerate extents, and a wide/tall pair. Accepted shapes run
// and are correct; refused ones throw from the pin.
TYPED_TEST(GemvCandidates, PinnedCandidatesStraddleTheirLimits) {
    using T = typename TestFixture::T;
    const int g = seg_trans_gate<T>();
    const std::vector<std::tuple<int, int, int>> shapes{  // (out, red, batch)
        {1, 1, 3},  {16, 9, 3}, {17, 9, 3}, {5, 70, 3}, {70, 5, 3},
        {64, g, 256}, {64, g + 1, 256}, {0, 4, 3}, {4, 0, 3}};
    int ran = 0, refused = 0;
    for (const C& c : gv::candidates<T>())
        for (Transpose t : {Transpose::NoTrans, Transpose::Trans, Transpose::ConjTrans})
            for (const auto& [out, red, batch] : shapes)
                for (bool bz : {false, true}) {
                    Spec s{t, t == Transpose::NoTrans ? out : red, t == Transpose::NoTrans ? red : out, batch};
                    s.beta_zero = bz;
                    s.seed = 1000u + 31u * out + red;
                    auto p = make_problem<T>(s);
                    const std::string what = name(c, s);
                    if (TestFixture::launches_crashing_vendor(c)) continue;
                    if (TestFixture::vendor_word_falls_back(c)) {
                        auto q = make_problem<T>(s);
                        const std::string want = this->auto_choice(q);
                        EXPECT_EQ(traced_choice([&] { this->run_pinned(c, p); }), want) << what;
                        expect_gemv(p, what + " (vendor-free: Auto)");
                        continue;
                    }
                    if (!TestFixture::expect_runs(c, t)) {
                        EXPECT_FALSE(this->pin_accepted(c, p)) << what << " was accepted";
                        ++refused;
                        continue;
                    }
                    ASSERT_TRUE(this->pin_accepted(c, p)) << what << " was refused";
                    expect_gemv(p, what);
                    ++ran;
                }
    EXPECT_GT(ran, 0);
    EXPECT_GT(refused, 0);
}

// The pinned facade runs exactly that family's driver: bit-identical to the direct call.
TYPED_TEST(GemvCandidates, PinnedRunIsTheDirectKernelBitForBit) {
    using T = typename TestFixture::T;
    int compared = 0;
    for (const C& c : gv::candidates<T>())
        for (Transpose t : {Transpose::NoTrans, Transpose::ConjTrans})
            for (auto [m, n] : {std::pair{13, 7}, std::pair{40, 3}, std::pair{9, 300}}) {
                if (!TestFixture::expect_runs(c, t) || TestFixture::launches_crashing_vendor(c)) continue;
                Spec s{t, m, n, 5};
                s.seed = 4242u + m;
                auto pinned = make_problem<T>(s);
                auto direct = make_problem<T>(s);
                const std::string what = name(c, s);
                this->run_pinned(c, pinned);
                std::string why;
                ASSERT_TRUE(this->direct(c, direct, &why)) << what << ": " << why;
                expect_gemv(pinned, what);
                for (std::size_t e = 0; e < pinned.mem.size(); ++e)
                    ASSERT_TRUE(same_bits(pinned.mem[e], direct.mem[e]))
                        << what << ": the pinned facade did not run this family's driver; element " << e;
                ++compared;
            }
    EXPECT_GT(compared, 0);
}

// A saturating batch: 1024 items repeating 7 distinct problems, on the sub-group bodies (3, 4, 5)
// and the one-thread bodies; each representative is correct, every other item bit-identical.
TYPED_TEST(GemvCandidates, SaturatingBatchIsBitIdenticalToItsRepresentative) {
    using T = typename TestFixture::T;
    constexpr int kBatch = 1024;
    struct Case { C c; Transpose t; int m, n; };
    const Case cases[] = {{gv::Cta{}, Transpose::Trans, 12, 40},    {gv::Cta{}, Transpose::ConjTrans, 200, 33},
                          {gv::Direct{}, Transpose::NoTrans, 12, 50}, {gv::Direct{}, Transpose::NoTrans, 70, 9},
                          {gv::Direct{}, Transpose::Trans, 33, 17},   {gv::Vendor{}, Transpose::ConjTrans, 24, 6}};
    for (const auto& k : cases) {
        if (!TestFixture::expect_runs(k.c, k.t) || TestFixture::launches_crashing_vendor(k.c)) continue;
        Spec s{k.t, k.m, k.n, kBatch};
        s.period = 7;
        s.seed = 777u + k.m;
        auto p = make_problem<T>(s);
        this->run_pinned(k.c, p);
        expect_gemv(p, name(k.c, s));
        if (::testing::Test::HasFatalFailure()) return;
    }
}

// §8.2 (R3): on conforming shapes a pin is accepted exactly when the family's own driver launches.
TYPED_TEST(GemvCandidates, CanRunEqualsLaunch) {
    using T = typename TestFixture::T;
    int disagreements = 0;
    for (const C& c : gv::candidates<T>())
        for (Transpose t : {Transpose::NoTrans, Transpose::Trans, Transpose::ConjTrans})
            for (auto [m, n] : {std::pair{1, 1}, std::pair{8, 5}, std::pair{33, 64}, std::pair{0, 3}}) {
                if (TestFixture::vendor_word_falls_back(c) || TestFixture::launches_crashing_vendor(c)) continue;
                Spec s{t, m, n, 2};
                s.seed = 61u + m;
                auto a = make_problem<T>(s);
                auto b = make_problem<T>(s);
                const bool pin = this->pin_accepted(c, a);
                std::string why;
                const bool run = this->direct(c, b, &why);
                EXPECT_EQ(pin, run) << name(c, s) << ": can_run says " << pin << ", the driver "
                                    << (run ? "launches" : "refuses: " + why);
                EXPECT_EQ(pin, TestFixture::expect_runs(c, t)) << name(c, s) << ": can_run disagrees with the oracle";
                disagreements += pin != run;
            }
    EXPECT_EQ(disagreements, 0);
}

// x or y disagreeing with A in batch or length: no native kernel may take it (one launch reads
// A.batch_size() items of every view with one (m, n)), so native pins throw and Auto takes the
// vendor, as the old shape builder did; vendor-free there is no route. The buffers are large
// enough that a kernel ignoring the check would stay in bounds and show as a write.
TYPED_TEST(GemvCandidates, MismatchedViewsHaveNoNativeRoute) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    const ScopedEnvVar clear("BATCHLAS_GEMV_ROUTE", nullptr);
    Spec s{Transpose::Trans, 8, 6, 3};
    auto p = make_problem<T>(s);
    struct Case { const char* what; int xlen, xb, ylen, yb; };
    const Case cases[] = {{"x too short", 7, 3, 6, 3}, {"y too long", 8, 3, 7, 3},
                          {"x batch", 8, 2, 6, 3},     {"y batch", 8, 3, 6, 2}};
    for (const auto& k : cases) {
        const VectorView<T> X(p.mem.data() + p.x_off, k.xlen, k.xb, Inc{p.incx}, Stride{p.sx});
        const VectorView<T> Y(p.mem.data() + p.y_off, k.ylen, k.yb, Inc{p.incy}, Stride{p.sy});
        for (const C& c : gv::candidates<T>()) {
            if (std::holds_alternative<gv::Vendor>(c)) continue;
            const Pin pin("gemv", c);
            try {
                (void)gemv<B, T>(*this->ctx, p.A(), X, Y, p.alpha, p.beta, s.trans);
                this->ctx->wait();
                ADD_FAILURE() << k.what << " " << select::to_string(c) << " was accepted";
            } catch (const std::invalid_argument& e) {
                EXPECT_NE(std::string(e.what()).find("cannot run this shape"), std::string::npos)
                    << k.what << ": " << e.what();
            }
        }
        if constexpr (!TestFixture::kVendor) {
            EXPECT_THROW(((void)gemv<B, T>(*this->ctx, p.A(), X, Y, p.alpha, p.beta, s.trans)), batchlas::NoRouteError)
                << k.what;
        }
    }
    for (std::size_t e = 0; e < p.mem.size(); ++e)
        ASSERT_TRUE(same_bits(p.mem[e], p.mem0[e])) << "a refused call wrote element " << e;
    // Auto: the vendor (known-defects #1's ortho call goes the same way). It indexes by A's
    // extents and batch, which stay inside this buffer.
    if (TestFixture::kVendor && TestFixture::vendor_launch_ok())
        for (const auto& k : cases) {
            const VectorView<T> X(p.mem.data() + p.x_off, k.xlen, k.xb, Inc{p.incx}, Stride{p.sx});
            const VectorView<T> Y(p.mem.data() + p.y_off, k.ylen, k.yb, Inc{p.incy}, Stride{p.sy});
            EXPECT_EQ(traced_choice([&] {
                          (void)gemv<B, T>(*this->ctx, p.A(), X, Y, p.alpha, p.beta, s.trans);
                          this->ctx->wait();
                      }),
                      "vendor")
                << k.what;
        }
}

// R6: a spelling that names nothing compiled throws instead of meaning Auto.
TYPED_TEST(GemvCandidates, UnknownPinsThrow) {
    using T = typename TestFixture::T;
    for (const char* word : {"bogus", "cta:1", "direct:w=2", "native:vendor", "native:segt", "body5", "native:cta:8",
                             // removed aliases (phase 5): each must stay an error
                             "native:cta", "native:direct"}) {
        auto p = make_problem<T>(Spec{});
        const Pin pin("gemv", std::string_view(word));
        EXPECT_THROW(this->run(p), std::invalid_argument) << word;
    }
}

// The named can_run-false cases (absorbs RouteGemv.CtaRequiresTransposedGpuWithAnEnumeratedSubGroup32's
// NoTrans clause and RouteGemv.ZeroExtentIsSupportedButNegativeExtentIsNot's empty batch).
TYPED_TEST(GemvCandidates, CanRunFalsePinsThrow) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    struct Case { C c; Transpose t; int m, n, batch; };
    const Case cases[] = {{gv::Cta{}, Transpose::NoTrans, 8, 5, 2},   {gv::Cta{}, Transpose::NoTrans, 1, 1, 1},
                          {gv::Cta{}, Transpose::Trans, 8, 5, 0},     {gv::Direct{}, Transpose::NoTrans, 8, 5, 0}};
    for (const auto& k : cases) {
        UnifiedVector<T> a(64, T(1)), x(16, T(1)), y(16, T(0));
        const MVof<T> A(a.data(), k.m, k.n, k.m, k.m * k.n, k.batch);
        const int red = k.t == Transpose::NoTrans ? k.n : k.m, out = k.t == Transpose::NoTrans ? k.m : k.n;
        const VectorView<T> X(x.data(), red, k.batch), Y(y.data(), out, k.batch);
        const Pin pin("gemv", k.c);
        const std::string what = name(k.c, Spec{k.t, k.m, k.n, k.batch});
        try {
            (void)gemv<B, T>(*this->ctx, A, X, Y, T(1), T(0), k.t);
            this->ctx->wait();
            ADD_FAILURE() << what << " was accepted";
        } catch (const std::invalid_argument& e) {
            EXPECT_NE(std::string(e.what()).find("cannot run this shape"), std::string::npos) << what << ": " << e.what();
        }
    }
}

// RouteGemv.CtaRequiresTransposedGpuWithAnEnumeratedSubGroup32's device clauses, ported: no test
// device lacks sub-group 32 and the host queue flips is_gpu and has_sg32 together, so each clause
// is checked alone on synthetic devices.
TEST(GemvDeviceAllows, CtaNeedsAGpuWithSubGroup32AndATransposedCall) {
    auto dev = [](bool gpu, bool sg32, bool vendor) {
        select::Device d;
        d.is_gpu = gpu;
        d.has_sg32 = sg32;
        d.has_vendor_blas = vendor;
        return d;
    };
    struct Case { select::Device d; bool transposed; bool cta, direct, vendor; const char* what; };
    const Case cases[] = {
        {dev(true, true, true), true, true, true, true, "gpu sg32 T"},
        {dev(true, false, true), true, false, true, true, "gpu no-sg32 T"},
        {dev(false, true, true), true, false, true, true, "cpu sg32 T"},
        {dev(true, true, true), false, false, true, true, "gpu sg32 N"},
        {dev(true, true, false), true, true, true, false, "gpu sg32 T, no vendor"},
    };
    for (const auto& k : cases) {
        EXPECT_EQ(gv::device_allows(C{gv::Cta{}}, k.d, k.transposed), k.cta) << k.what;
        EXPECT_EQ(gv::device_allows(C{gv::Direct{}}, k.d, k.transposed), k.direct) << k.what;
        EXPECT_EQ(gv::device_allows(C{gv::Vendor{}}, k.d, k.transposed), k.vendor) << k.what;
    }
}

// RouteGemv.HeterogeneousBatchIsRefusedByBothNativeTiers, ported: a heterogeneous A has no
// native family; Auto is the vendor, and vendor-free there is no route.
TYPED_TEST(GemvCandidates, HeterogeneousBatchHasNoNativeRoute) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    const int m = 16, n = 12, batch = 4;
    Matrix<T, MatrixFormat::Dense> A(m, n, batch);
    A.fill(T(1));
    UnifiedVector<T> x(n * batch, T(1)), y(m * batch, T(0));
    UnifiedVector<int> rows(batch), cols(batch);
    for (int b = 0; b < batch; ++b) rows[b] = m - b, cols[b] = n;
    const auto het = A.view().with_active_dims(rows.to_span(), cols.to_span());
    ASSERT_TRUE(het.is_heterogeneous());
    const VectorView<T> X(x.data(), n, batch), Y(y.data(), m, batch);
    for (const C& c : gv::candidates<T>()) {
        if (std::holds_alternative<gv::Vendor>(c)) continue;
        const Pin pin("gemv", c);
        EXPECT_THROW(((void)gemv<B, T>(*this->ctx, het, X, Y, T(1), T(0), Transpose::NoTrans)), std::invalid_argument)
            << select::to_string(c);
    }
    const ScopedEnvVar clear("BATCHLAS_GEMV_ROUTE", nullptr);
    auto call = [&] {
        (void)gemv<B, T>(*this->ctx, het, X, Y, T(1), T(0), Transpose::NoTrans);
        this->ctx->wait();
    };
    if constexpr (TestFixture::kVendor) {
        if (TestFixture::vendor_launch_ok()) EXPECT_EQ(traced_choice(call), "vendor");
    } else {
        EXPECT_THROW(call(), batchlas::NoRouteError);
    }
}

// §5.3: spellings (case-folded) and the class words, via ScopedPin and the environment. Bare
// `native` is the row's best runnable non-vendor: cta under Trans, direct under NoTrans.
TYPED_TEST(GemvCandidates, ClassWordsAndSpellings) {
    using T = typename TestFixture::T;
    const ScopedEnvVar clear("BATCHLAS_GEMV_ROUTE", nullptr);
    for (Transpose t : {Transpose::NoTrans, Transpose::ConjTrans}) {
        const Spec s{t, 24, 10, 4};
        const std::string auto_pick = TestFixture::kVendor ? "vendor" : (t == Transpose::NoTrans ? "direct" : "cta");
        if (TestFixture::vendor_launch_ok()) {
            auto p = make_problem<T>(s);
            EXPECT_EQ(traced_choice([&] { this->run(p); }), auto_pick);
        }
        const std::string native_pick = t == Transpose::NoTrans ? "direct" : "cta";
        std::vector<std::pair<const char*, std::string>> expect{
            {"Direct", "direct"}, {"DIRECT", "direct"}, {"native", native_pick}};
        if (TestFixture::vendor_launch_ok()) {
            expect.emplace_back("vendor", auto_pick);
            expect.emplace_back("auto", auto_pick);
        }
        if (t != Transpose::NoTrans) {
            expect.emplace_back("cta", "cta");
            expect.emplace_back("CTA", "cta");
        }
        for (const auto& [word, spelling] : expect)
            for (bool via_env : {false, true}) {
                auto p = make_problem<T>(s);
                select::testing::reset_warnings();
                std::string err;
                const std::string got = traced_choice([&] {
                    const ScopedEnvVar env("BATCHLAS_GEMV_ROUTE", via_env ? word : nullptr);
                    std::optional<Pin> pin;
                    if (!via_env) pin.emplace("gemv", std::string_view(word));
                    this->run(p);
                }, &err);
                const std::string what = std::string(word) + " " + label(s) + (via_env ? " via env" : " via ScopedPin");
                EXPECT_EQ(got, spelling) << what;
                const bool warns = std::string(word) == "vendor" && !TestFixture::kVendor;
                EXPECT_EQ(err.find("gemv pinned \"vendor\", but no vendor candidate") != std::string::npos, warns)
                    << what << ": " << err;
                expect_gemv(p, what);
            }
    }
}

// §5.3: a ScopedPin wins over BATCHLAS_GEMV_ROUTE, and nested pins restore the outer one. An env
// `cta` on NoTrans throws (R6); the old router fell through silently to the vendor.
TYPED_TEST(GemvCandidates, ScopedPinBeatsTheEnvironment) {
    using T = typename TestFixture::T;
    const ScopedEnvVar env("BATCHLAS_GEMV_ROUTE", "cta");
    {
        auto p = make_problem<T>(Spec{Transpose::NoTrans, 9, 4, 2});
        EXPECT_THROW(this->run(p), std::invalid_argument) << "the environment pin was not read";
    }
    auto p = make_problem<T>(Spec{Transpose::NoTrans, 9, 4, 2});
    EXPECT_EQ(traced_choice([&] {
                  const Pin pin("gemv", C{gv::Direct{}});
                  this->run(p);
              }),
              "direct");
    expect_gemv(p, "direct over env cta");
    const Spec st{Transpose::Trans, 9, 4, 2};
    const Pin outer("gemv", C{gv::Cta{}});
    {
        const Pin inner("gemv", C{gv::Direct{}});
        auto r = make_problem<T>(st);
        EXPECT_EQ(traced_choice([&] { this->run(r); }), "direct");
    }
    auto r = make_problem<T>(st);
    EXPECT_EQ(traced_choice([&] { this->run(r); }), "cta") << "the inner pin did not restore the outer";
    expect_gemv(r, "outer cta");
}

// Auto against the shipped transcribed table (sm_89 and sm_120 hold identical rows; any other
// device borrows one). RouteGemv.CdoubleTransposedBandIsPreferredAndEveryBoundaryIsPinned,
// ported: complex<double> Trans/ConjTrans is cta inside 64 <= red <= 352, out >= 256,
// batch >= 320 and the vendor outside, each edge from both sides, on- and off-grid; every other
// type and NoTrans is the vendor. Vendor-free it is cta for any transposed call, direct for N.
TYPED_TEST(GemvCandidates, AutoReadsTheTranscribedTable) {
    using T = typename TestFixture::T;
    const ScopedEnvVar clear("BATCHLAS_GEMV_ROUTE", nullptr);
    const auto tables =
        select::tables_in_borrow_order("gemv", select::dtype_name<T>(), select::device_of<TestFixture::B>(*this->ctx));
    ASSERT_FALSE(tables.empty());
    ASSERT_EQ(tables.front()->source.rfind("transcribed:", 0), 0u) << tables.front()->file;
    constexpr bool kCd = std::is_same_v<T, std::complex<double>>;
    struct Row { int out, red, batch; bool in; };
    const Row rows[] = {{256, 64, 320, true},   {257, 65, 321, true},   {300, 352, 321, true},
                        {256, 351, 321, true},  {1024, 128, 320, true}, {256, 63, 320, false},
                        {256, 62, 321, false},  {256, 353, 320, false}, {260, 354, 321, false},
                        {255, 128, 320, false}, {254, 100, 321, false}, {512, 128, 319, false},
                        {300, 100, 318, false}, {64, 32, 64, false}};
    for (const Row& r : rows)
        for (Transpose t : {Transpose::NoTrans, Transpose::Trans, Transpose::ConjTrans}) {
            Spec s{t, t == Transpose::NoTrans ? r.out : r.red, t == Transpose::NoTrans ? r.red : r.out, r.batch};
            s.period = 1;  // one random item, the rest copies: the decision, not the data, is under test
            s.seed = 41u + r.red;
            auto p = make_problem<T>(s);
            std::string want;
            if (!TestFixture::kVendor) want = t == Transpose::NoTrans ? "direct" : "cta";
            else want = (kCd && r.in && t != Transpose::NoTrans) ? "cta" : "vendor";
            if (want == "vendor" && !TestFixture::vendor_launch_ok()) continue;  // known-defects #13
            EXPECT_EQ(traced_choice([&] { this->run(p); }), want)
                << label(s) << " out=" << r.out << " red=" << r.red;
            expect_gemv(p, "auto " + label(s));
            if (::testing::Test::HasFatalFailure()) return;
        }
}

// key_of's every field reaches choose(): a synthetic table whose winner changes with trans
// (ConjTrans folding to T), out, red and batch alone; out/red follow trans (NoTrans: out = rows).
TYPED_TEST(GemvCandidates, AutoReadsEveryKeyField) {
    using T = typename TestFixture::T;
    const ScopedEnvVar clear("BATCHLAS_GEMV_ROUTE", nullptr);
    const TableGuard restore;
    install_gemv_table<T, TestFixture::B>(*this->ctx,
                                          "trans=N out=8 red=8 batch=8 | direct 1 | vendor 2\n"
                                          "trans=N out=64 red=8 batch=8 | vendor 1 | direct 2\n"
                                          "trans=T out=8 red=8 batch=8 | cta 1 | direct 2 | vendor 3\n"
                                          "trans=T out=64 red=8 batch=8 | direct 1 | cta 2 | vendor 3\n"
                                          "trans=T out=8 red=64 batch=8 | vendor 1 | cta 2 | direct 3\n"
                                          "trans=T out=8 red=8 batch=64 | direct 1 | vendor 2 | cta 3\n");
    struct Probe { Transpose t; int m, n, batch; std::string expect; const char* field; };
    const Probe probes[] = {
        {Transpose::NoTrans, 8, 8, 8, "direct", "base N"},
        {Transpose::NoTrans, 64, 8, 8, TestFixture::kVendor ? "vendor" : "direct", "out (N: A.rows)"},
        {Transpose::NoTrans, 8, 64, 8, "direct", "red (N: A.cols) has no N row of its own"},
        {Transpose::Trans, 8, 8, 8, "cta", "trans"},
        {Transpose::ConjTrans, 8, 8, 8, "cta", "trans (C folds to T)"},
        {Transpose::Trans, 8, 64, 8, "direct", "out (T: A.cols)"},
        {Transpose::Trans, 64, 8, 8, TestFixture::kVendor ? "vendor" : "cta", "red (T: A.rows)"},
        {Transpose::Trans, 8, 8, 64, "direct", "batch"}};
    for (const auto& k : probes) {
        if (k.expect == "vendor" && !TestFixture::vendor_launch_ok()) continue;  // known-defects #13
        Spec s{k.t, k.m, k.n, k.batch};
        s.seed = 43u;
        auto p = make_problem<T>(s);
        EXPECT_EQ(traced_choice([&] { this->run(p); }), k.expect) << "the " << k.field << " row";
        expect_gemv(p, std::string("the ") + k.field + " row");
    }
}

// RouteGemv.OutLenAndRedLenSwapWithTransA, ported: the trace key's out and red are y's and x's
// lengths, which swap with trans; a stored 64x2048 A is out=64 under N and out=2048 under T/C.
TYPED_TEST(GemvCandidates, TraceKeyOutRedFollowTrans) {
    using T = typename TestFixture::T;
    for (Transpose t : {Transpose::NoTrans, Transpose::Trans, Transpose::ConjTrans}) {
        auto p = make_problem<T>(Spec{t, 64, 48, 5});
        const Pin native("gemv", std::string_view("native"));  // the key does not depend on the choice
        const std::string line = traced_line([&] { this->run(p); });
        const std::string want = t == Transpose::NoTrans ? "trans=N out=64 red=48 batch=5 ->"
                                                         : "trans=T out=48 red=64 batch=5 ->";
        EXPECT_NE(line.find(want), std::string::npos) << line;
        expect_gemv(p, line);
    }
}

// With a table naming only the vendor, a vendor-free build takes the last resort, direct (the
// one native tier with no GPU gate), for both trans; a vendor build takes the vendor.
TYPED_TEST(GemvCandidates, VendorFreeLastResortIsDirect) {
    using T = typename TestFixture::T;
    if (!TestFixture::vendor_launch_ok()) GTEST_SKIP() << "known-defects #13: the vendor arm crashes here";
    const ScopedEnvVar clear("BATCHLAS_GEMV_ROUTE", nullptr);
    const TableGuard restore;
    install_gemv_table<T, TestFixture::B>(*this->ctx, "trans=T out=8 red=8 batch=8 | vendor 1\n");
    for (Transpose t : {Transpose::NoTrans, Transpose::Trans}) {
        auto p = make_problem<T>(Spec{t, 20, 30, 3});
        std::string all;
        const std::string got = traced_choice([&] { this->run(p); }, &all);
        EXPECT_EQ(got, TestFixture::kVendor ? "vendor" : "direct") << trans_s(t);
        if (!TestFixture::kVendor) EXPECT_NE(all.find("last resort"), std::string::npos) << all;
        expect_gemv(p, std::string("vendor-only table ") + trans_s(t));
    }
}

// ortho's CGS calls (ortho.cc): A_i = the first i columns of an m-row A at ld = m, ConjTrans into
// C (length i), then NoTrans with alpha -1, beta 1 back into the column; every candidate.
TYPED_TEST(GemvCandidates, OrthoCallerShapes) {
    using T = typename TestFixture::T;
    int ran = 0;
    for (const C& c : gv::candidates<T>())
        for (int i : {1, 7})
            for (Transpose t : {Transpose::ConjTrans, Transpose::NoTrans}) {
                if (!TestFixture::expect_runs(c, t) || TestFixture::vendor_word_falls_back(c) ||
                    TestFixture::launches_crashing_vendor(c))
                    continue;
                Spec s{t, 40, i, 2};
                s.seed = 500u + i;
                auto p = make_problem<T>(s);
                p.alpha = t == Transpose::NoTrans ? T(-1) : T(1);
                p.beta = t == Transpose::NoTrans ? T(1) : T(0);
                p.s.beta_zero = t != Transpose::NoTrans;
                this->run_pinned(c, p);
                expect_gemv(p, name(c, s) + " (ortho shape)");
                ++ran;
            }
    EXPECT_GT(ran, 0);
}

// The coverage row (§5.6): the backend, gemv's key (m = A.rows, n = A.cols, k = A.rows, transA)
// and the native flags. threadsafe: the child re-executes the binary.
TYPED_TEST(GemvCandidates, CoverageRowCarriesBackendKeyAndNativeFlags) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    const std::string dir = ::testing::TempDir() + "gemv_cov." + std::string(select::dtype_name<T>());
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    const std::string out = dir + "/cov";
    const Spec lo{Transpose::NoTrans, 16, 4, 2};
    const Spec hi{Transpose::ConjTrans, 48, 3, 2};
    auto child = [&] {
        const ScopedEnvVar cov("BATCHLAS_COVERAGE_OUT", out.c_str());
        const ScopedEnvVar clear("BATCHLAS_GEMV_ROUTE", nullptr);
        batchlas::coverage::g_dynamic_enabled = true;
        {
            const Pin pin("gemv", C{gv::Direct{}});
            auto p = make_problem<T>(lo);
            this->run(p);
        }
        const Pin pin("gemv", C{gv::Cta{}});
        auto p = make_problem<T>(hi);
        this->run(p);
        std::exit(0);
    };
    EXPECT_EXIT(child(), ::testing::ExitedWithCode(0), "");
    std::map<std::string, std::vector<std::string>> rows;
    for (const auto& ent : std::filesystem::directory_iterator(dir)) {
        std::ifstream in(ent.path());
        for (std::string line; std::getline(in, line);) {
            if (line.rfind("reached,gemv,", 0) != 0) continue;
            std::vector<std::string> f;
            std::stringstream ss(line);
            for (std::string tok; std::getline(ss, tok, ',');) f.push_back(tok);
            ASSERT_GE(f.size(), 19u) << line;
            rows[f[5] + " " + f[6] + " " + f[7]] = f;
        }
    }
    std::filesystem::remove_all(dir);
    ASSERT_EQ(rows.size(), 2u);
    const std::string klo = "16 4 16", khi = "48 3 48";
    ASSERT_TRUE(rows.count(klo)) << "no 16x4 row";
    ASSERT_TRUE(rows.count(khi)) << "no 48x3 row";
    const std::string backend = B == Backend::CUDA ? "CUDA" : (B == Backend::ROCM ? "ROCM" : "?");
    auto as_int = [](auto e) { return std::to_string(static_cast<int>(e)); };
    for (const auto& [key, s, algo] : {std::tuple{klo, lo, "direct"}, std::tuple{khi, hi, "cta"}}) {
        const auto& f = rows[key];
        EXPECT_EQ(f[3], backend) << key << ": the backend column";
        EXPECT_EQ(f[10], algo) << key;
        EXPECT_EQ(f[12], "1") << key;
        EXPECT_EQ(f[13], "1") << key;
        EXPECT_EQ(f[18], as_int(s.trans)) << key;
    }
}

// RouteGemv.DirectHasNoGpuGateAndThatIsTheWholeWorkPackage, ported: on a host queue there is no
// table, so the last resort decides: the vendor (netlib) where compiled, else direct; cta's
// sub-group body cannot run there, and a direct pin runs and is correct.
template <typename Config>
class GemvCandidatesCpu : public test_utils::BatchLASTest<Config> {};
TYPED_TEST_SUITE(GemvCandidatesCpu, Types);

TYPED_TEST(GemvCandidatesCpu, CpuQueueRunsDirectOrTheVendor) {
    using T = typename TypeParam::ScalarType;
    static constexpr Backend B = TypeParam::BackendVal;
    if (!this->ctx) GTEST_SKIP() << "no queue";
    if (this->ctx->device().type == DeviceType::GPU) GTEST_SKIP() << "a GPU queue";
    const ScopedEnvVar clear("BATCHLAS_GEMV_ROUTE", nullptr);
    for (Transpose t : {Transpose::NoTrans, Transpose::ConjTrans}) {
        const Spec s{t, 11, 6, 3};
        {
            auto p = make_problem<T>(s);
            const Pin pin("gemv", C{gv::Cta{}});
            EXPECT_THROW(((void)gemv<B, T>(*this->ctx, p.A(), p.X(), p.Y(), p.alpha, p.beta, t)), std::invalid_argument);
        }
        auto p = make_problem<T>(s);
        EXPECT_EQ(traced_choice([&] {
                      (void)gemv<B, T>(*this->ctx, p.A(), p.X(), p.Y(), p.alpha, p.beta, t);
                      this->ctx->wait();
                  }),
                  batchlas::select::level3_vendor_available<B> ? "vendor" : "direct")
            << trans_s(t);
        expect_gemv(p, std::string("cpu auto ") + trans_s(t));
        auto d = make_problem<T>(s);
        {
            const Pin pin("gemv", C{gv::Direct{}});
            (void)gemv<B, T>(*this->ctx, d.A(), d.X(), d.Y(), d.alpha, d.beta, t);
            this->ctx->wait();
        }
        expect_gemv(d, std::string("cpu direct ") + trans_s(t));
    }
}

// The transcribed rows of both devices, read with Table::nearest directly so every machine checks
// them: cta | vendor | direct in the complex<double> window, vendor | cta | direct for other
// transposed cells, vendor | direct under N; on-grid and off-grid keys.
TEST(GemvTranscribedTable, RowsHoldTheOldPreference) {
    struct Row { const char* dtype; const char* trans; int out, red, batch; const char* ranked; };
    const Row rows[] = {
        {"cdouble", "T", 256, 64, 320, "cta|vendor|direct"},   {"cdouble", "T", 700, 200, 5000, "cta|vendor|direct"},
        {"cdouble", "T", 256, 352, 32768, "cta|vendor|direct"}, {"cdouble", "T", 256, 63, 320, "vendor|cta|direct"},
        {"cdouble", "T", 256, 353, 320, "vendor|cta|direct"},   {"cdouble", "T", 255, 64, 320, "vendor|cta|direct"},
        {"cdouble", "T", 256, 64, 319, "vendor|cta|direct"},    {"cdouble", "N", 700, 200, 5000, "vendor|direct"},
        {"float", "T", 700, 200, 5000, "vendor|cta|direct"},    {"double", "N", 3, 9000, 2, "vendor|direct"},
        {"cfloat", "T", 256, 64, 320, "vendor|cta|direct"},     {"double", "T", 1, 1, 1, "vendor|cta|direct"}};
    for (const char* dev : {"sm_89", "sm_120"})
        for (const Row& r : rows) {
            const auto tables = select::tables_in_borrow_order("gemv", r.dtype, select::device_from_key(dev));
            ASSERT_FALSE(tables.empty()) << r.dtype;
            const select::Table& t = *tables.front();
            ASSERT_EQ(t.device, dev) << r.dtype;
            EXPECT_EQ(t.source.rfind("transcribed:", 0), 0u) << t.file;
            const select::Key key{{"trans", r.trans}, {"out", r.out}, {"red", r.red}, {"batch", r.batch}};
            const select::TableRow* row = t.nearest(key);
            ASSERT_NE(row, nullptr) << t.file;
            std::string got;
            for (const auto& e : row->ranked) got += (got.empty() ? "" : "|") + e.spelling;
            EXPECT_EQ(got, r.ranked) << t.file << ":" << row->line;
            EXPECT_FALSE(row->timed) << t.file;
        }
}

}  // namespace
