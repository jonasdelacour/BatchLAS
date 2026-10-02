// The descriptor-registry potrf path against the shipped one, in one process on one device.
// T1 equivalence, T2 sizing, T3 execution, T4 pins, T5 saturating Blocked, T6 explain(),
// T7 host cost of one decision.
#include <gtest/gtest.h>

#include "potrf_select.hh"

#include "../../../src/backends/potrf_route.hh"

#include <batchlas/blas/functions/potrf.hh>
#include <batchlas/settings.hh>
#include <batchlas/util/env.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>
#include <batchlas/util/sycl-vector.hh>

#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <complex>
#include <cstdio>
#include <fstream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace batchlas;
namespace v2 = batchlas::potrf_v2;
constexpr Backend kB = Backend::CUDA;

namespace {

std::shared_ptr<Queue> gpu() {
    static std::shared_ptr<Queue>* q = [] {
        auto* p = new std::shared_ptr<Queue>();
        try {
            *p = std::make_shared<Queue>(Device("gpu"), kB, true);
        } catch (...) {
        }
        return p;
    }();
    return *q;
}

std::string old_key(dispatch::Route r) {
    if (dispatch::is_vendor(r)) return "vendor:cusolver";
    return std::string(dispatch::to_string(r.origin)) + ":" + std::string(dispatch::to_string(r.algo));
}

struct Cell { std::string dtype; Uplo uplo; int n; long batch; };

std::vector<Cell> sweep_cells() {
    std::vector<Cell> out;
    std::ifstream f(BATCHLAS_V2_CELLS);
    std::string dt, up;
    int n;
    long b;
    while (f >> dt >> up >> n >> b) out.push_back({dt, up == "upper" ? Uplo::Upper : Uplo::Lower, n, b});
    return out;
}

std::vector<Cell> grid_cells() {   // the offline prototype's grid, every tier boundary
    std::vector<Cell> out;
    for (const char* dt : {"float", "double", "cfloat", "cdouble"})
        for (Uplo up : {Uplo::Lower, Uplo::Upper})
            for (int n : {1, 2, 8, 12, 16, 17, 20, 31, 32, 33, 35, 36, 48, 64, 77, 78, 96, 128, 160,
                          200, 256, 257, 320, 384, 512, 640, 744, 745, 1024, 1280, 1536})
                for (long b : {1L, 16L, 128L, 1024L, 8192L, 32768L}) out.push_back({dt, up, n, b});
    return out;
}

struct Diff { std::string mode, cell, old_route, new_route, why; };

template <class T>
void compare_cell(Queue& q, const Cell& c, const char* mode, std::vector<Diff>& diffs,
                  std::ostream* csv) {
    // Routing reads extents only; no element is touched.
    const MatrixView<T, MatrixFormat::Dense> A(nullptr, c.n, c.n, c.n, c.n * c.n, static_cast<int>(c.batch));
    const auto o = backend::potrf_route<kB, T>(q, A, c.uplo, true);
    const auto ovf = backend::potrf_route<kB, T>(q, A, c.uplo, false);
    const auto s = v2::select<kB, T, true>(q, A, c.uplo, {}, {}, v2::Mode::DecideOnly);
    const auto svf = v2::select<kB, T, false>(q, A, c.uplo, {}, {}, v2::Mode::DecideOnly);
    // The vendor-free shipped walk answers vendor:auto for "nothing native serves".
    const std::string nvf = svf.decision.index < 0 ? "vendor:cusolver" : std::string(svf.key());
    std::ostringstream cell;
    cell << c.dtype << " " << (c.uplo == Uplo::Upper ? "U" : "L") << " n=" << c.n << " b=" << c.batch;
    if (old_key(o) != s.key()) {
        diffs.push_back({std::string(mode) + "/vendor", cell.str(), old_key(o), std::string(s.key()),
                         std::string(s.decision.why)});
    }
    if (old_key(ovf) != nvf) {
        diffs.push_back({std::string(mode) + "/vendor-free", cell.str(), old_key(ovf), nvf,
                         std::string(svf.decision.why)});
    }
    if (csv) {
        *csv << mode << "," << c.dtype << "," << (c.uplo == Uplo::Upper ? "upper" : "lower") << ","
             << c.n << "," << c.batch << "," << old_key(o) << "," << s.key() << ","
             << dispatch::sel::to_cstr(s.decision.by) << "," << old_key(ovf) << "," << nvf << ","
             << dispatch::sel::to_cstr(svf.decision.by) << "\n";
    }
}

}  // namespace

namespace {

void run_equivalence(const std::vector<Cell>& cells, const char* label) {
    auto q = gpu();
    if (!q) GTEST_SKIP() << "no CUDA GPU";
    const ScopedEnvVar no_pin("BATCHLAS_POTRF_ROUTE", nullptr);
    std::unique_ptr<std::ofstream> csv;
    if (const char* out = std::getenv("BATCHLAS_V2_EQUIV_CSV")) {
        csv = std::make_unique<std::ofstream>(std::string(out) + "." + label + ".csv");
        *csv << "mode,dtype,uplo,n,batch,old,new,new_by,old_vf,new_vf,new_vf_by\n";
    }
    std::vector<Diff> diffs;
    long compared = 0;
    for (const char* prof : {"", "sm_89"}) {
        const ScopedEnvVar p("BATCHLAS_ROUTING_PROFILE", *prof ? prof : nullptr);
        const char* mode = *prof ? "sm_89-windows" : "default-model";
        for (const Cell& c : cells) {
            if (c.dtype == "float") compare_cell<float>(*q, c, mode, diffs, csv.get());
            else if (c.dtype == "double") compare_cell<double>(*q, c, mode, diffs, csv.get());
            else if (c.dtype == "cfloat") compare_cell<std::complex<float>>(*q, c, mode, diffs, csv.get());
            else compare_cell<std::complex<double>>(*q, c, mode, diffs, csv.get());
            compared += 2;
        }
    }
    // The ONE intended class: the cta_wg extension tier serves Upper shapes CTA cannot hold,
    // so the vendor-free walk finds a native route where the shipped one finds none.
    long intended = 0;
    for (const Diff& d : diffs) {
        const bool ext = d.mode.ends_with("vendor-free") && d.old_route == "vendor:cusolver" &&
                         d.new_route == "native:cta_wg";
        intended += ext;
        std::printf("%s %-14s %-30s old=%-16s new=%-16s (%s)\n", ext ? "INTENDED" : "MISMATCH",
                    d.mode.c_str(), d.cell.c_str(), d.old_route.c_str(), d.new_route.c_str(),
                    d.why.c_str());
    }
    std::printf("[%s] decisions compared=%ld differences=%zu intended=%ld unintended=%ld\n", label,
                compared, diffs.size(), intended, static_cast<long>(diffs.size()) - intended);
    EXPECT_EQ(static_cast<long>(diffs.size()), intended);
}

}  // namespace

TEST(PotrfSelectV2, T1_EquivalenceSweepCells) { run_equivalence(sweep_cells(), "sweep"); }
TEST(PotrfSelectV2, T1_EquivalenceBoundaryGrid) { run_equivalence(grid_cells(), "grid"); }

// ---------------------------------------------------------------------------------------------
namespace {

template <class T> using RealOf = typename base_type<T>::type;
template <class T> T cj(T v) { if constexpr (std::is_same_v<T, RealOf<T>>) return v; else return std::conj(v); }

template <class T>
void fill_spd(Matrix<T, MatrixFormat::Dense>& A, int n, int batch, unsigned seed, bool same) {
    T* p = A.view().data_ptr();
    const int ld = A.view().ld(), st = A.view().stride();
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(-1, 1);
    std::vector<T> M(static_cast<size_t>(n) * n);
    for (int b = 0; b < batch; ++b) {
        if (b == 0 || !same) {
            for (int j = 0; j < n; ++j)
                for (int i = 0; i <= j; ++i) {
                    T v;
                    if constexpr (std::is_same_v<T, RealOf<T>>) v = T(u(rng));
                    else v = T(u(rng), i == j ? 0.0 : u(rng));
                    if (i == j) v = T(RealOf<T>(n + 2 + std::abs(u(rng))));
                    M[i + static_cast<size_t>(j) * n] = v;
                    M[j + static_cast<size_t>(i) * n] = cj(v);
                }
        }
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) p[static_cast<size_t>(b) * st + i + static_cast<size_t>(j) * ld] = M[i + static_cast<size_t>(j) * n];
    }
}

// ||L L^H - A||_F / ||A||_F for item b, in double.
template <class T>
double residual(const T* F, const T* A0, int n, int ld, Uplo uplo) {
    using C = std::complex<double>;
    auto Lv = [&](int i, int j) -> C {   // L(i,j), i >= j
        const T v = (uplo == Uplo::Lower) ? F[i + static_cast<size_t>(j) * ld] : cj(F[j + static_cast<size_t>(i) * ld]);
        if constexpr (std::is_same_v<T, RealOf<T>>) return C(v, 0); else return C(v.real(), v.imag());
    };
    double num = 0, den = 0;
    for (int j = 0; j < n; ++j)
        for (int i = j; i < n; ++i) {
            C s = 0;
            for (int k = 0; k <= j; ++k) s += Lv(i, k) * std::conj(Lv(j, k));
            const T a = A0[i + static_cast<size_t>(j) * ld];
            C av;
            if constexpr (std::is_same_v<T, RealOf<T>>) av = C(a, 0); else av = C(a.real(), a.imag());
            num += std::norm(s - av) * (i == j ? 1 : 2);
            den += std::norm(av) * (i == j ? 1 : 2);
        }
    return std::sqrt(num / den);
}

template <class T>
struct RunResult { std::string key, text; double worst = 0; std::size_t ws = 0; std::vector<int32_t> info; };

// Factor through the NEW path with EXACTLY the selection's workspace; residual on 3 items.
template <class T>
RunResult<T> run_new(Queue& q, int n, int batch, Uplo uplo, unsigned seed = 7, bool same = false,
                     Matrix<T, MatrixFormat::Dense>* keep = nullptr) {
    Matrix<T, MatrixFormat::Dense> A(n, n, batch);
    fill_spd<T>(A, n, batch, seed, same);
    const int ld = A.view().ld(), st = A.view().stride();
    std::vector<T> A0(A.view().data_ptr(), A.view().data_ptr() + static_cast<size_t>(st) * batch);
    const auto sel = v2::select<kB, T>(q, A.view(), uplo);
    UnifiedVector<std::byte> ws(sel.workspace);
    UnifiedVector<int32_t> info(batch, int32_t(-7));
    (void)v2::run<kB, T>(q, sel, A.view(), uplo, ws.to_span(), info.to_span());
    q.wait();
    RunResult<T> r{std::string(sel.key()), v2::explain<kB, T>(sel), 0, sel.workspace,
                   std::vector<int32_t>(info.begin(), info.end())};
    for (int b : {0, batch / 2, batch - 1}) {
        r.worst = std::max(r.worst, residual<T>(A.view().data_ptr() + static_cast<size_t>(b) * st,
                                                A0.data() + static_cast<size_t>(b) * st, n, ld, uplo));
    }
    if (keep) *keep = std::move(A);
    return r;
}

template <class T> double tol(int n) { return 50.0 * n * std::numeric_limits<RealOf<T>>::epsilon(); }

}  // namespace

// ---- T2 sizing -------------------------------------------------------------------------------
template <class T>
void size_case(Queue& q, int n, int batch, Uplo uplo, const char* name) {
    Matrix<T, MatrixFormat::Dense> A(n, n, batch);
    const std::size_t old_sz = potrf_buffer_size<kB, T>(q, A.view(), uplo);
    const auto sel = v2::select<kB, T>(q, A.view(), uplo);
    std::size_t route_sz = 0;
    const std::string k(sel.key());
    if (k == "native:cta" || k == "native:cta_wg") route_sz = sycl_potrf::potrf_cta_buffer_size<T>(q, A.view());
    else if (k == "native:tiny") route_sz = sycl_potrf::potrf_tiny_buffer_size<T>(q, A.view());
    else if (k == "native:lpanel") route_sz = sycl_potrf::potrf_lpanel_buffer_size<T>(q, A.view());
    else if (k == "vendor:cusolver") route_sz = backend::potrf_vendor_buffer_size<kB, T>(q, A.view(), uplo);
    else route_sz = sel.workspace;   // blocked: the clamped layout, checked by running it (T3)
    std::printf("SIZING %-28s route=%-16s old=%12zu B  new=%12zu B  route-own=%zu B\n", name,
                k.c_str(), old_sz, sel.workspace, route_sz);
    EXPECT_EQ(sel.workspace, route_sz) << name;
    EXPECT_LE(sel.workspace, old_sz) << name;
}

TEST(PotrfSelectV2, T2_Sizing) {
    auto q = gpu();
    if (!q) GTEST_SKIP();
    const ScopedEnvVar no_pin("BATCHLAS_POTRF_ROUTE", nullptr);
    size_case<float>(*q, 36, 8192, Uplo::Lower, "float L n=36 b=8192");
    size_case<float>(*q, 128, 8192, Uplo::Lower, "float L n=128 b=8192");
    size_case<float>(*q, 640, 512, Uplo::Lower, "float L n=640 b=512");
    size_case<std::complex<double>>(*q, 20, 2048, Uplo::Lower, "cdouble L n=20 b=2048");
    size_case<double>(*q, 36, 8192, Uplo::Lower, "double L n=36 b=8192");
    size_case<float>(*q, 128, 8192, Uplo::Upper, "float U n=128 b=8192");
}

// ---- T3 execution: one shape per route, exactly sel.workspace bytes ---------------------------
template <class T>
void exec_case(Queue& q, int n, int batch, Uplo uplo, const char* pin, const char* want) {
    const ScopedEnvVar p("BATCHLAS_POTRF_ROUTE", pin);
    Matrix<T, MatrixFormat::Dense> Anew(1, 1, 1);
    const auto r = run_new<T>(q, n, batch, uplo, 7, false, &Anew);
    // The shipped path, same input, same pin: how many items differ bitwise in the triangle.
    Matrix<T, MatrixFormat::Dense> Aold(n, n, batch);
    fill_spd<T>(Aold, n, batch, 7, false);
    const std::size_t old_ws = potrf_buffer_size<kB, T>(q, Aold.view(), uplo);
    {
        UnifiedVector<std::byte> ws(old_ws);
        UnifiedVector<int32_t> info(batch, 0);
        (void)potrf<kB, T>(q, Aold.view(), uplo, ws.to_span(), info.to_span());
        q.wait();
    }
    const int st = Anew.view().stride(), ld = Anew.view().ld();
    long differ = 0;
    for (int b = 0; b < batch; ++b) {
        for (int j = 0; j < n; ++j) {
            bool d = false;
            for (int i = 0; i < n && !d; ++i) {
                if ((uplo == Uplo::Lower) ? (i < j) : (i > j)) continue;
                const size_t e = static_cast<size_t>(b) * st + i + static_cast<size_t>(j) * ld;
                d = std::memcmp(&Anew.view().data_ptr()[e], &Aold.view().data_ptr()[e], sizeof(T)) != 0;
            }
            if (d) { ++differ; break; }
        }
    }
    std::printf("EXEC %-8s %s n=%-4d b=%-6d pin=%-14s -> %-16s ws=%zu (old %zu) residual=%.3e "
                "items!=shipped=%ld\n%s",
                v2::pr::type_name<T>().c_str(), uplo == Uplo::Lower ? "L" : "U", n, batch,
                pin ? pin : "-", r.key.c_str(), r.ws, old_ws, r.worst, differ, r.text.c_str());
    EXPECT_EQ(r.key, want);
    if (r.key != "native:cta_wg") EXPECT_EQ(differ, 0) << "same route as shipped, different bits";
    EXPECT_LE(r.worst, tol<T>(n)) << want;
    for (int i : {0, batch - 1}) EXPECT_EQ(r.info[i], 0);
}

TEST(PotrfSelectV2, T3_ExecutionPerRoute) {
    auto q = gpu();
    if (!q) GTEST_SKIP();
    exec_case<float>(*q, 16, 8192, Uplo::Lower, nullptr, "native:tiny");
    exec_case<float>(*q, 36, 8192, Uplo::Lower, nullptr, "native:cta");
    exec_case<float>(*q, 128, 8192, Uplo::Lower, nullptr, "native:lpanel");
    exec_case<std::complex<double>>(*q, 20, 2048, Uplo::Lower, nullptr, "native:lpanel");
    exec_case<float>(*q, 640, 512, Uplo::Lower, nullptr, "native:blocked");
    exec_case<double>(*q, 36, 8192, Uplo::Lower, "native:blocked", "native:blocked");
    exec_case<float>(*q, 1024, 128, Uplo::Lower, nullptr, "native:blocked");
    exec_case<std::complex<double>>(*q, 32, 512, Uplo::Lower, nullptr, "vendor:cusolver");
    exec_case<float>(*q, 128, 8192, Uplo::Upper, nullptr, "vendor:cusolver");
    exec_case<std::complex<float>>(*q, 48, 2048, Uplo::Upper, "native:cta_wg", "native:cta_wg");
    exec_case<float>(*q, 128, 512, Uplo::Upper, "native:cta_wg", "native:cta_wg");
}

// ---- T4 pins ---------------------------------------------------------------------------------
TEST(PotrfSelectV2, T4_Pins) {
    auto q = gpu();
    if (!q) GTEST_SKIP();
    Matrix<float, MatrixFormat::Dense> U(128, 128, 64);
    {
        const ScopedEnvVar p("BATCHLAS_POTRF_ROUTE", "native:lpanel");
        const std::size_t before = v2::pin_warnings_emitted();
        const auto s1 = v2::select<kB, float>(*q, U.view(), Uplo::Upper);
        const auto s2 = v2::select<kB, float>(*q, U.view(), Uplo::Upper);
        EXPECT_FALSE(s1.decision.pin_honoured);
        EXPECT_EQ(s1.key(), "vendor:cusolver");
        EXPECT_EQ(s2.key(), "vendor:cusolver");
        EXPECT_EQ(v2::pin_warnings_emitted() - before, 1u) << "warn once per pin, not per call";
        const auto shipped = backend::potrf_route<kB, float>(*q, U.view(), Uplo::Upper, true);
        EXPECT_TRUE(dispatch::is_vendor(shipped)) << "the shipped path silently does the same";
        v2::set_pin_policy(dispatch::sel::PinPolicy::Strict);
        try {
            (void)v2::select<kB, float>(*q, U.view(), Uplo::Upper);
            ADD_FAILURE() << "strict pin did not throw";
        } catch (const std::invalid_argument& e) {
            EXPECT_NE(std::string(e.what()).find("Lower only"), std::string::npos) << e.what();
        }
        v2::set_pin_policy(std::nullopt);
    }
    {
        const ScopedEnvVar p("BATCHLAS_POTRF_ROUTE", "native:cta");
        Matrix<float, MatrixFormat::Dense> L(128, 128, 64);
        const auto s = v2::select<kB, float>(*q, L.view(), Uplo::Lower);
        EXPECT_FALSE(s.decision.pin_honoured);
        const std::string t = v2::explain<kB, float>(s);
        EXPECT_NE(t.find("native:cta       unfit"), std::string::npos) << t;
    }
    {
        const ScopedEnvVar p("BATCHLAS_POTRF_ROUTE", "native:blocked");
        Matrix<float, MatrixFormat::Dense> L(128, 128, 64);
        const auto s = v2::select<kB, float>(*q, L.view(), Uplo::Lower);
        EXPECT_TRUE(s.decision.pin_honoured);
        EXPECT_EQ(s.key(), "native:blocked");
    }
    {   // bare origin: the shipped path has no native for Upper n=128 and runs the vendor
        const ScopedEnvVar p("BATCHLAS_POTRF_ROUTE", "native");
        const auto s = v2::select<kB, float>(*q, U.view(), Uplo::Upper);
        std::printf("bare native, Upper n=128:\n%s", v2::explain<kB, float>(s).c_str());
        EXPECT_TRUE(s.decision.pin_honoured);
        EXPECT_EQ(s.key(), "native:cta_wg");
    }
}

// ---- T5 saturating batch through Blocked, clamped W vs the shipped unclamped layout ----------
TEST(PotrfSelectV2, T5_BlockedSaturatingBatchBitIdentical) {
    auto q = gpu();
    if (!q) GTEST_SKIP();
    using T = double;
    const int n = 36, batch = 8192;
    const ScopedEnvVar p("BATCHLAS_POTRF_ROUTE", "native:blocked");
    Matrix<T, MatrixFormat::Dense> Anew(1, 1, 1);
    const auto r = run_new<T>(*q, n, batch, Uplo::Lower, 99, /*same=*/true, &Anew);
    ASSERT_EQ(r.key, "native:blocked");
    // The shipped path pinned the same way, unclamped W.
    Matrix<T, MatrixFormat::Dense> Aold(n, n, batch);
    fill_spd<T>(Aold, n, batch, 99, true);
    const std::size_t old_ws = potrf_buffer_size<kB, T>(*q, Aold.view(), Uplo::Lower);
    UnifiedVector<std::byte> ws(old_ws);
    UnifiedVector<int32_t> info(batch, 0);
    (void)potrf<kB, T>(*q, Aold.view(), Uplo::Lower, ws.to_span(), info.to_span());
    q->wait();
    const int st = Anew.view().stride();
    long item_mismatch = 0, old_mismatch = 0;
    for (int b = 0; b < batch; ++b) {
        for (int e = 0; e < n * n; ++e) {
            const int i = e % n, j = e / n;
            if (i < j) continue;
            const T v = Anew.view().data_ptr()[static_cast<size_t>(b) * st + e];
            if (std::memcmp(&v, &Anew.view().data_ptr()[e], sizeof(T)) != 0) { ++item_mismatch; break; }
        }
        for (int e = 0; e < n * n; ++e) {
            if (e % n < e / n) continue;
            if (std::memcmp(&Anew.view().data_ptr()[static_cast<size_t>(b) * st + e],
                            &Aold.view().data_ptr()[static_cast<size_t>(b) * st + e], sizeof(T)) != 0) {
                ++old_mismatch;
                break;
            }
        }
    }
    std::printf("T5 double n=36 b=8192 blocked: new ws=%zu B (clamped W), old ws=%zu B; "
                "items != item0: %ld; items != old path: %ld; residual %.3e\n",
                r.ws, old_ws, item_mismatch, old_mismatch, r.worst);
    EXPECT_EQ(item_mismatch, 0);
    EXPECT_EQ(old_mismatch, 0);
    EXPECT_LE(r.worst, tol<T>(n));
}

// ---- T6 explain() ----------------------------------------------------------------------------
TEST(PotrfSelectV2, T6_ExplainF128) {
    auto q = gpu();
    if (!q) GTEST_SKIP();
    const ScopedEnvVar no_pin("BATCHLAS_POTRF_ROUTE", nullptr);
    const MatrixView<float, MatrixFormat::Dense> A(nullptr, 128, 128, 128, 128 * 128, 8192);
    const auto s = v2::select<kB, float>(*q, A, Uplo::Lower, {}, {}, v2::Mode::DecideOnly);
    const std::string t = v2::explain<kB, float>(s);
    std::printf("%s", t.c_str());
    EXPECT_EQ(s.key(), "native:lpanel");
    EXPECT_NE(t.find("by=model"), std::string::npos);
    EXPECT_NE(t.find("PotrfLpanelKernel<float,8>"), std::string::npos);
    EXPECT_NE(t.find("groups=8192 wg=128"), std::string::npos);
    EXPECT_NE(t.find("native:tiny      illegal  n > tiny cap"), std::string::npos);
}

// ---- T7 host cost of one decision, old vs new (reported, not asserted) -----------------------
TEST(PotrfSelectV2, T7_HostOverhead) {
    auto q = gpu();
    if (!q) GTEST_SKIP();
    const ScopedEnvVar no_pin("BATCHLAS_POTRF_ROUTE", nullptr);
    const MatrixView<float, MatrixFormat::Dense> A(nullptr, 128, 128, 128, 128 * 128, 8192);
    constexpr int kReps = 20000;
    auto time = [&](auto&& f) {
        for (int i = 0; i < 200; ++i) f();
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < kReps; ++i) f();
        return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / kReps;
    };
    const double old_us = time([&] { (void)backend::potrf_route<kB, float>(*q, A, Uplo::Lower, true); });
    const double new_us = time([&] {
        (void)v2::select<kB, float>(*q, A, Uplo::Lower, {}, {}, v2::Mode::DecideOnly);
    });
    std::printf("T7 one routing decision, float L n=128 b=8192: shipped potrf_route %.2f us, "
                "new select (decide only) %.2f us\n", old_us, new_us);
}
