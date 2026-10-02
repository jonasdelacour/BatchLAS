// The descriptor-registry potrf path (now the CUDA facade) against the pre-registry one
// (backend::potrf_legacy*), in one process on one device.
// T0 public facade, T1 route + launch-geometry equivalence with an explicit expected-diff list,
// T2 sizing, T3 execution (non-natural ld/stride, poisoned padding, workspace canary), T4 pins,
// T5 saturating batches (Blocked, cta_wg), T6 explain + extrapolation, T7 host cost,
// T8 the cta_wg ceiling launch, T9 heterogeneous batch, T10 an unpriced vendor, T11 run() checks,
// T12 cta_wg vs vendor timing (reported).
#include <gtest/gtest.h>

#include "../../../src/backends/potrf_route.hh"
#include "../../../src/backends/potrf_select.hh"
#include "../../../src/extensions/potrf_native.hh"

#include <batchlas/blas/functions/potrf.hh>
#include <batchlas/settings.hh>
#include <batchlas/util/env.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>
#include <batchlas/util/sycl-vector.hh>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace batchlas;
namespace v2 = batchlas::potrf_v2;
namespace pp = batchlas::potrf_plan;
namespace ds = batchlas::dispatch::sel;
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

template <class T> using RealOf = typename base_type<T>::type;
template <class T> T cj(T v) { if constexpr (std::is_same_v<T, RealOf<T>>) return v; else return std::conj(v); }
template <class T> std::string tname() { return v2::pr::type_name<T>(); }

// ---- a batch with an explicit layout: ld = n + ld_pad, stride = ld*n + stride_pad -----------
// Matrix-owned (the vendor and the batched sub-ops read its per-item pointer array).
template <class T>
struct Mat {
    int n = 0, ld = 0, st = 0, b = 0;
    Matrix<T, MatrixFormat::Dense> M;
    Mat(int n_, int b_, int ld_pad = 0, int st_pad = 0)
        : n(n_), ld(n_ + ld_pad), st((n_ + ld_pad) * n_ + st_pad), b(b_),
          M(n_, n_, b_, n_ + ld_pad, (n_ + ld_pad) * n_ + st_pad) {}
    MatrixView<T, MatrixFormat::Dense> view() const { return M.view(); }
    T* data() const { return M.view().data_ptr(); }
    size_t size() const { return static_cast<size_t>(st) * b; }
    T* item(int i) const { return data() + static_cast<size_t>(i) * st; }
    bool is_pad(size_t e) const {
        const size_t off = e % static_cast<size_t>(st);
        return off >= static_cast<size_t>(ld) * n || static_cast<int>(off % ld) >= n;
    }
};

template <class T> T poison() { return T(RealOf<T>(1.0e6)); }   // in range, finite, large

// Hermitian positive definite, complex off-diagonals with a nonzero imaginary part; the padding
// gets an in-range poison the kernels would happily read.
template <class T>
void fill_spd(Mat<T>& A, unsigned seed, bool same) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(-1, 1);
    const int n = A.n;
    std::vector<T> M(static_cast<size_t>(n) * n);
    for (size_t e = 0; e < A.size(); ++e) A.data()[e] = poison<T>();
    for (int b = 0; b < A.b; ++b) {
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
            for (int i = 0; i < n; ++i) A.item(b)[i + static_cast<size_t>(j) * A.ld] = M[i + static_cast<size_t>(j) * n];
    }
}

// ||L L^H - A||_F / ||A||_F for one item, in double.
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

template <class T> double tol(int n) { return 50.0 * n * std::numeric_limits<RealOf<T>>::epsilon(); }

// Items whose factored triangle differs bitwise between two batches.
template <class T>
long triangle_diff(const Mat<T>& X, const Mat<T>& Y, Uplo uplo, int ref_item = -1) {
    long differ = 0;
    for (int b = 0; b < X.b; ++b) {
        const T* x = X.item(b);
        const T* y = Y.item(ref_item >= 0 ? ref_item : b);
        bool d = false;
        for (int j = 0; j < X.n && !d; ++j)
            for (int i = 0; i < X.n && !d; ++i) {
                if ((uplo == Uplo::Lower) ? (i < j) : (i > j)) continue;
                const size_t e = i + static_cast<size_t>(j) * X.ld;
                d = std::memcmp(&x[e], &y[e], sizeof(T)) != 0;
            }
        differ += d;
    }
    return differ;
}

constexpr std::size_t kCanary = 4096;

struct RunResult {
    std::string key, text;
    double worst = 0;
    std::size_t ws = 0;
    long pad_touched = 0, canary_touched = 0, info_bad = 0;
};

// Factor through the registry path with EXACTLY sel.workspace bytes followed by a canary.
template <class T>
RunResult run_new(Queue& q, Mat<T>& A, Uplo uplo) {
    const std::vector<T> A0(A.data(), A.data() + A.size());
    const auto sel = v2::select<kB, T>(q, A.view(), uplo);
    UnifiedVector<std::byte> ws(sel.workspace + kCanary, std::byte{0xA5});
    UnifiedVector<int32_t> info(A.b, int32_t(-7));
    (void)v2::run<kB, T>(q, sel, A.view(), uplo, Span<std::byte>(ws.data(), sel.workspace), info.to_span());
    q.wait();
    RunResult r{std::string(sel.key()), v2::explain<kB, T>(sel), 0, sel.workspace};
    for (size_t e = sel.workspace; e < ws.size(); ++e) r.canary_touched += ws.data()[e] != std::byte{0xA5};
    for (size_t e = 0; e < A.size(); ++e) {
        if (A.is_pad(e)) r.pad_touched += std::memcmp(&A.data()[e], &A0[e], sizeof(T)) != 0;
    }
    for (int i = 0; i < A.b; ++i) r.info_bad += info.data()[i] != 0;
    for (int b : {0, A.b / 2, A.b - 1}) {
        r.worst = std::max(r.worst, residual<T>(A.item(b), A0.data() + static_cast<size_t>(b) * A.st, A.n, A.ld, uplo));
    }
    return r;
}

template <class T>
void run_legacy(Queue& q, Mat<T>& A, Uplo uplo) {
    UnifiedVector<std::byte> ws(backend::potrf_buffer_size_legacy<kB, T>(q, A.view(), uplo));
    UnifiedVector<int32_t> info(A.b, 0);
    (void)backend::potrf_legacy<kB, T>(q, A.view(), uplo, ws.to_span(), info.to_span());
    q.wait();
}

template <class T> int idx_of(std::string_view k) { return v2::Tbl<kB, T, true>::index_of(k); }

}  // namespace

// ---- T0 the PUBLIC facade runs the registry (first: its stderr checks need a fresh process) --
TEST(PotrfSelectV2, T0_PublicFacade) {
    auto q = gpu();
    if (!q) GTEST_SKIP() << "no CUDA GPU";
    {   // sizing: the chosen route's own bytes, not the max over tiers
        const ScopedEnvVar no_pin("BATCHLAS_POTRF_ROUTE", nullptr);
        const MatrixView<float, MatrixFormat::Dense> A(nullptr, 36, 36, 36, 36 * 36, 8192);
        const std::size_t pub = potrf_buffer_size<kB, float>(*q, A, Uplo::Lower);
        const std::size_t leg = backend::potrf_buffer_size_legacy<kB, float>(*q, A, Uplo::Lower);
        std::printf("T0 float L n=36 b=8192: public potrf_buffer_size=%zu B, legacy=%zu B\n", pub, leg);
        EXPECT_EQ(pub, (v2::select<kB, float>(*q, A, Uplo::Lower).workspace));
        EXPECT_LT(pub, std::size_t{1} << 20);
        EXPECT_GT(leg, std::size_t{100} << 20) << "the defect this fixes is no longer reproduced";
    }
    {   // an unhonoured pin through the public entry point: warned once, then strict throws
        const ScopedEnvVar p("BATCHLAS_POTRF_ROUTE", "lpanel");
        Mat<float> U(64, 128);
        fill_spd(U, 3, false);
        const std::size_t before = v2::pin_warnings_emitted();
        UnifiedVector<std::byte> ws(potrf_buffer_size<kB, float>(*q, U.view(), Uplo::Upper));
        UnifiedVector<int32_t> info(U.b, 0);
        (void)potrf<kB, float>(*q, U.view(), Uplo::Upper, ws.to_span(), info.to_span());
        q->wait();
        EXPECT_EQ(v2::pin_warnings_emitted() - before, 1u) << "once across buffer_size + potrf";
        EXPECT_EQ((v2::select<kB, float>(*q, U.view(), Uplo::Upper).key()), "vendor:cusolver");
        v2::set_pin_policy(ds::PinPolicy::Strict);
        EXPECT_THROW(((void)potrf<kB, float>(*q, U.view(), Uplo::Upper, ws.to_span(), info.to_span())),
                     std::invalid_argument);
        v2::set_pin_policy(std::nullopt);
    }
    {   // the exact key is the pin vocabulary: no "not a recognised route" from the public path
        const ScopedEnvVar p("BATCHLAS_POTRF_ROUTE", "native:cta_wg");
        Mat<std::complex<float>> U(48, 256);
        fill_spd(U, 5, false);
        testing::internal::CaptureStderr();
        UnifiedVector<std::byte> ws(potrf_buffer_size<kB, std::complex<float>>(*q, U.view(), Uplo::Upper));
        UnifiedVector<int32_t> info(U.b, 0);
        (void)potrf<kB, std::complex<float>>(*q, U.view(), Uplo::Upper, ws.to_span(), info.to_span());
        q->wait();
        const std::string err = testing::internal::GetCapturedStderr();
        EXPECT_EQ(err.find("not a recognised route"), std::string::npos) << err;
        EXPECT_EQ(err.find("not honoured"), std::string::npos) << err;
    }
}

// ---- T1 equivalence: route key AND launch geometry, against an explicit expected-diff list ---
namespace {

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

std::vector<Cell> grid_cells() {   // every tier boundary, both sides
    std::vector<Cell> out;
    for (const char* dt : {"float", "double", "cfloat", "cdouble"})
        for (Uplo up : {Uplo::Lower, Uplo::Upper})
            for (int n : {1, 2, 8, 12, 16, 17, 20, 31, 32, 33, 35, 36, 38, 39, 47, 48, 54, 55, 63, 64, 77, 78, 96, 109, 110,
                          128, 129, 155, 156, 200, 256, 257, 320, 384, 512, 640, 744, 745, 1024, 1280, 1536})
                for (long b : {1L, 16L, 128L, 1024L, 8192L, 32768L}) out.push_back({dt, up, n, b});
    return out;
}

// The ONLY intended route change, as data: vendor-free Upper orders that subgroup CTA cannot hold
// and the work-group-scope tier can, so the shipped walk has no native route and the new one picks
// native:cta_wg. Recorded on the RTX PRO 6000 (sm_120) of the README; another device must
// re-derive it (T1 fails there rather than passing an unchecked list).
struct OnlyCtaWg { const char* dtype; int lo, hi; };
constexpr OnlyCtaWg kOnlyCtaWg[] = {
    {"float", 78, 155}, {"double", 55, 109}, {"cfloat", 55, 109}, {"cdouble", 39, 77},
};
constexpr std::size_t kListedLocalMem = 101376;
constexpr int kListedMaxWg = 1024;

bool listed(const Cell& c) {
    if (c.uplo != Uplo::Upper) return false;
    for (const auto& r : kOnlyCtaWg) {
        if (c.dtype == r.dtype && c.n >= r.lo && c.n <= r.hi) return true;
    }
    return false;
}

// The planned launch, encoded the way the shipped launchers' debug hooks report theirs.
template <class Tb>
long long plan_code(const ds::Selection<Tb>& s) {
    if (s.decision.index < 0) return -1;
    return Tb::with_row(s.plan, [&]<class R>(const typename R::Geometry& g) -> long long {
        using G = typename R::Geometry;
        if constexpr (std::is_same_v<G, pp::TinyGeometry>) {
            const int S = g.wg_size / 32;
            return S > 0 ? (static_cast<long long>(S) << 16) | (g.per_wg / S) : 0;
        } else if constexpr (std::is_same_v<G, pp::CtaGeometry>) {
            return (static_cast<long long>(g.L) << 16) | g.G;
        } else if constexpr (std::is_same_v<G, pp::LpanelGeometry>) {
            return (static_cast<long long>(g.nb) << 16) | (g.L << 4) | g.G;
        } else if constexpr (requires { g.p.nb; }) {
            return g.p.nb | (static_cast<long long>(g.p.W) << 16);
        } else {
            return -1;
        }
    });
}

// Groups follow the packing: one per G matrices; cta_wg is G = 1 at work-group scope.
template <class Tb>
bool groups_ok(const ds::Selection<Tb>& s, long batch) {
    if (s.decision.index < 0) return true;
    const bool wg = s.key() == "native:cta_wg";
    return Tb::with_row(s.plan, [&]<class R>(const typename R::Geometry& g) -> bool {
        using G = typename R::Geometry;
        if constexpr (std::is_same_v<G, pp::TinyGeometry>) {
            return g.num_wg == (batch + g.per_wg - 1) / g.per_wg;
        } else if constexpr (std::is_same_v<G, pp::CtaGeometry>) {
            if (wg && (g.G != 1 || g.subgroup_scope || g.wg_size != g.L)) return false;
            return g.num_wg == (batch + g.G - 1) / g.G && g.wg_size == g.G * g.L;
        } else if constexpr (std::is_same_v<G, pp::LpanelGeometry>) {
            return g.num_wg == (batch + g.G - 1) / g.G && g.wg_size == g.G * g.L;
        } else {
            return true;
        }
    });
}

// What the shipped launcher derives for itself (its debug hook), plus Blocked's W clamp.
template <class T>
long long shipped_code(Queue& q, const std::string& key, int n, long batch) {
    if (key == "native:tiny") return sycl_potrf::potrf_tiny_debug_launch<T>(q, n);
    if (key == "native:cta") return sycl_potrf::potrf_cta_debug_launch<T>(q, n, static_cast<int>(batch));
    if (key == "native:lpanel") return sycl_potrf::potrf_lpanel_debug_launch<T>(q, n, static_cast<int>(batch));
    if (key == "native:blocked") {
        const unsigned r = sycl_potrf::potrf_blocked_debug_params<T>(q, n);
        const int nb = static_cast<int>(r & 0xffffu);
        const int W = std::max(1, std::min(static_cast<int>(r >> 16), n - nb));
        return nb | (static_cast<long long>(W) << 16);
    }
    return -1;
}

struct Tally {
    long compared = 0, intended = 0, listed_cells = 0, geometry_checked = 0;
    std::vector<std::string> bad;
};

template <class T>
void compare_cell(Queue& q, const Cell& c, const char* mode, Tally& t, std::ostream* csv) {
    // Routing reads extents only; no element is touched.
    const MatrixView<T, MatrixFormat::Dense> A(nullptr, c.n, c.n, c.n, c.n * c.n, static_cast<int>(c.batch));
    const auto o = backend::potrf_route<kB, T>(q, A, c.uplo, true);
    const auto ovf = backend::potrf_route<kB, T>(q, A, c.uplo, false);
    const auto s = v2::select<kB, T, true>(q, A, c.uplo, v2::Mode::DecideOnly);
    const auto svf = v2::select<kB, T, false>(q, A, c.uplo, v2::Mode::DecideOnly);
    // The vendor-free shipped walk answers vendor:auto for "nothing native serves".
    const std::string nvf = svf.decision.index < 0 ? "vendor:cusolver" : std::string(svf.key());
    std::ostringstream cell;
    cell << c.dtype << " " << (c.uplo == Uplo::Upper ? "U" : "L") << " n=" << c.n << " b=" << c.batch;
    auto bad = [&](const std::string& what) { t.bad.push_back(std::string(mode) + " " + cell.str() + ": " + what); };
    t.compared += 2;
    t.listed_cells += listed(c);
    if (old_key(o) != s.key()) bad("vendor route old=" + old_key(o) + " new=" + std::string(s.key()));
    if (old_key(ovf) != nvf) {
        if (listed(c) && old_key(ovf) == "vendor:cusolver" && nvf == "native:cta_wg") ++t.intended;
        else bad("vendor-free route old=" + old_key(ovf) + " new=" + nvf);
    } else if (listed(c)) {
        bad("listed as cta_wg-only but the routes agree on " + nvf);
    }
    auto check_geometry = [&](const auto& sel, const std::string& k) {
        if (!groups_ok(sel, c.batch)) bad(k + " group count does not follow the packing");
        if (k == "vendor:cusolver" || k == "native:cta_wg" || k.empty()) return;
        ++t.geometry_checked;
        const long long want = shipped_code<T>(q, k, c.n, c.batch), got = plan_code(sel);
        if (want != got) bad(k + " planned geometry " + std::to_string(got) + " != launcher's " + std::to_string(want));
    };
    check_geometry(s, std::string(s.key()));
    check_geometry(svf, std::string(svf.key()));
    if (csv) {
        *csv << mode << "," << c.dtype << "," << (c.uplo == Uplo::Upper ? "upper" : "lower") << ","
             << c.n << "," << c.batch << "," << old_key(o) << "," << s.key() << ","
             << ds::to_cstr(s.decision.by) << "," << old_key(ovf) << "," << nvf << ","
             << ds::to_cstr(svf.decision.by) << "," << plan_code(s) << "\n";
    }
}

void run_equivalence(const std::vector<Cell>& cells, const char* label) {
    auto q = gpu();
    if (!q) GTEST_SKIP() << "no CUDA GPU";
    const auto f = sycl_potrf::potrf_device_facts(q->device());
    ASSERT_TRUE(f.local_mem_bytes == kListedLocalMem && f.max_wg_size == kListedMaxWg)
        << "kOnlyCtaWg was recorded for local_mem=" << kListedLocalMem << " max_wg=" << kListedMaxWg
        << "; this device has " << f.local_mem_bytes << "/" << f.max_wg_size << ": re-derive it";
    const ScopedEnvVar no_pin("BATCHLAS_POTRF_ROUTE", nullptr);
    std::unique_ptr<std::ofstream> csv;
    if (const char* out = std::getenv("BATCHLAS_V2_EQUIV_CSV")) {
        csv = std::make_unique<std::ofstream>(std::string(out) + "." + label + ".csv");
        *csv << "mode,dtype,uplo,n,batch,old,new,new_by,old_vf,new_vf,new_vf_by,plan\n";
    }
    Tally t;
    long listed_total = 0;
    for (const char* prof : {"", "sm_89"}) {
        const ScopedEnvVar p("BATCHLAS_ROUTING_PROFILE", *prof ? prof : nullptr);
        const char* mode = *prof ? "sm_89-windows" : "default-model";
        const long before = t.listed_cells;
        for (const Cell& c : cells) {
            if (c.dtype == "float") compare_cell<float>(*q, c, mode, t, csv.get());
            else if (c.dtype == "double") compare_cell<double>(*q, c, mode, t, csv.get());
            else if (c.dtype == "cfloat") compare_cell<std::complex<float>>(*q, c, mode, t, csv.get());
            else compare_cell<std::complex<double>>(*q, c, mode, t, csv.get());
        }
        listed_total += t.listed_cells - before;
    }
    for (const std::string& b : t.bad) std::printf("MISMATCH %s\n", b.c_str());
    std::printf("[%s] decisions compared=%ld intended(listed cta_wg)=%ld of %ld listed cells; "
                "geometry checked=%ld; unintended=%zu\n",
                label, t.compared, t.intended, listed_total, t.geometry_checked, t.bad.size());
    EXPECT_TRUE(t.bad.empty());
    EXPECT_EQ(t.intended, listed_total) << "every listed cell must move, and only those";
}

}  // namespace

TEST(PotrfSelectV2, T1_EquivalenceSweepCells) { run_equivalence(sweep_cells(), "sweep"); }
TEST(PotrfSelectV2, T1_EquivalenceBoundaryGrid) { run_equivalence(grid_cells(), "grid"); }

// The probe that recorded kOnlyCtaWg (run with --gtest_also_run_disabled_tests).
TEST(PotrfSelectV2, DISABLED_ProbeCtaWgOnlyRanges) {
    auto q = gpu();
    if (!q) GTEST_SKIP();
    const ScopedEnvVar no_pin("BATCHLAS_POTRF_ROUTE", nullptr);
    const auto f = sycl_potrf::potrf_device_facts(q->device());
    std::printf("local_mem=%zu max_wg=%d\n", static_cast<size_t>(f.local_mem_bytes), f.max_wg_size);
    auto probe = [&]<class T>() {
        int lo = -1, hi = -1;
        for (int n = 1; n <= 1600; ++n) {
            const MatrixView<T, MatrixFormat::Dense> A(nullptr, n, n, n, n * n, 1024);
            const auto svf = v2::select<kB, T, false>(*q, A, Uplo::Upper, v2::Mode::DecideOnly);
            const bool old_native = dispatch::is_native(backend::potrf_route<kB, T>(*q, A, Uplo::Upper, false));
            if (!old_native && svf.key() == "native:cta_wg") { if (lo < 0) lo = n; hi = n; }
        }
        std::printf("%s Upper cta_wg-only: %d..%d\n", tname<T>().c_str(), lo, hi);
    };
    probe.template operator()<float>();
    probe.template operator()<double>();
    probe.template operator()<std::complex<float>>();
    probe.template operator()<std::complex<double>>();
}

// ---- T2 sizing: the chosen route's bytes; the Blocked W clamp; never more than the old max ----
TEST(PotrfSelectV2, T2_Sizing) {
    auto q = gpu();
    if (!q) GTEST_SKIP();
    auto row = [&]<class T>(int n, int b, Uplo uplo, const char* pin, const char* name) {
        const ScopedEnvVar p("BATCHLAS_POTRF_ROUTE", pin);
        const MatrixView<T, MatrixFormat::Dense> A(nullptr, n, n, n, n * n, b);
        const std::size_t old_sz = backend::potrf_buffer_size_legacy<kB, T>(*q, A, uplo);
        const auto sel = v2::select<kB, T>(*q, A, uplo);
        std::printf("SIZING %-34s route=%-16s old=%12zu B  new=%12zu B\n", name,
                    std::string(sel.key()).c_str(), old_sz, sel.workspace);
        EXPECT_LE(sel.workspace, old_sz) << name;
        EXPECT_GT(sel.workspace, 0u) << name;
        return std::pair{old_sz, sel};
    };
    {   // the headline defect: CTA at n=36 sized for Blocked's unclamped W*W*batch scratch
        const auto [old_sz, sel] = row.template operator()<float>(36, 8192, Uplo::Lower, nullptr, "float L n=36 b=8192");
        EXPECT_EQ(sel.key(), "native:cta");
        EXPECT_GT(old_sz, std::size_t{100} << 20);
        EXPECT_LT(sel.workspace, std::size_t{1} << 20);
    }
    {   // Blocked itself: the layout allocates W clamped to n - nb, priced as before
        const auto [old_sz, sel] = row.template operator()<double>(36, 8192, Uplo::Lower, "native:blocked",
                                                                   "double L n=36 b=8192 pinned blocked");
        ASSERT_EQ(sel.key(), "native:blocked");
        const unsigned r = sycl_potrf::potrf_blocked_debug_params<double>(*q, 36);
        const int nb = static_cast<int>(r & 0xffffu);
        const pp::BlockedParams want{nb, std::max(1, std::min(static_cast<int>(r >> 16), 36 - nb)),
                                     resident::kMinBlocksPerSm};
        const MatrixView<double, MatrixFormat::Dense> A(nullptr, 36, 36, 36, 36 * 36, 8192);
        EXPECT_EQ(sel.workspace, sycl_potrf::potrf_blocked_buffer_size_params<double>(*q, A, want));
        EXPECT_LT(sel.workspace * 16, old_sz) << "the W clamp no longer shrinks the scratch";
    }
    (void)row.template operator()<float>(128, 8192, Uplo::Lower, nullptr, "float L n=128 b=8192");
    (void)row.template operator()<float>(640, 512, Uplo::Lower, nullptr, "float L n=640 b=512");
    (void)row.template operator()<std::complex<double>>(20, 2048, Uplo::Lower, nullptr, "cdouble L n=20 b=2048");
    (void)row.template operator()<double>(36, 8192, Uplo::Lower, nullptr, "double L n=36 b=8192");
    (void)row.template operator()<float>(128, 8192, Uplo::Upper, nullptr, "float U n=128 b=8192");
}

// ---- T3 execution: one shape per route, exactly sel.workspace bytes + canary, odd layouts ----
template <class T>
void exec_case(Queue& q, int n, int batch, Uplo uplo, const char* pin, const char* want,
               int ld_pad = 0, int st_pad = 0) {
    const ScopedEnvVar p("BATCHLAS_POTRF_ROUTE", pin);
    Mat<T> Anew(n, batch, ld_pad, st_pad);
    fill_spd(Anew, 7, false);
    const RunResult r = run_new(q, Anew, uplo);
    Mat<T> Aold(n, batch, ld_pad, st_pad);
    fill_spd(Aold, 7, false);
    run_legacy(q, Aold, uplo);
    const long differ = triangle_diff(Anew, Aold, uplo);
    std::printf("EXEC %-7s %s n=%-4d b=%-5d ld=%-4d st=%-6d pin=%-14s -> %-16s ws=%zu residual=%.2e "
                "items!=legacy=%ld pad=%ld canary=%ld\n%s",
                tname<T>().c_str(), uplo == Uplo::Lower ? "L" : "U", n, batch, Anew.ld, Anew.st,
                pin ? pin : "-", r.key.c_str(), r.ws, r.worst, differ, r.pad_touched, r.canary_touched,
                r.text.c_str());
    EXPECT_EQ(r.key, want);
    if (r.key != "native:cta_wg") EXPECT_EQ(differ, 0) << "same route as legacy, different bits";
    EXPECT_LE(r.worst, tol<T>(n)) << want;
    EXPECT_EQ(r.info_bad, 0);
    EXPECT_EQ(r.pad_touched, 0) << "wrote outside the n x n block (ld/stride derived, not read?)";
    EXPECT_EQ(r.canary_touched, 0) << "wrote past sel.workspace: under-sized";
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
    exec_case<float>(*q, 128, 512, Uplo::Lower, "native:cta_wg", "native:cta_wg");
}

// Rule 3: ld != n and stride != ld*n, read rather than derived, on every native tier.
TEST(PotrfSelectV2, T3_ExecutionStridedLayouts) {
    auto q = gpu();
    if (!q) GTEST_SKIP();
    exec_case<float>(*q, 16, 4096, Uplo::Lower, nullptr, "native:tiny", 5, 11);
    exec_case<float>(*q, 36, 8192, Uplo::Lower, nullptr, "native:cta", 3, 17);
    exec_case<std::complex<double>>(*q, 20, 2048, Uplo::Lower, nullptr, "native:lpanel", 5, 9);
    exec_case<float>(*q, 640, 256, Uplo::Lower, nullptr, "native:blocked", 1, 64);
    exec_case<double>(*q, 36, 4096, Uplo::Lower, "native:blocked", "native:blocked", 2, 3);
    exec_case<float>(*q, 128, 2048, Uplo::Upper, nullptr, "vendor:cusolver", 7, 1);
    exec_case<std::complex<float>>(*q, 48, 2048, Uplo::Upper, "native:cta_wg", "native:cta_wg", 3, 5);
    exec_case<std::complex<float>>(*q, 96, 1024, Uplo::Lower, "native:cta_wg", "native:cta_wg", 1, 2);
}

// ---- T4 pins ---------------------------------------------------------------------------------
TEST(PotrfSelectV2, T4_Pins) {
    auto q = gpu();
    if (!q) GTEST_SKIP();
    const MatrixView<float, MatrixFormat::Dense> U(nullptr, 128, 128, 128, 128 * 128, 64);
    {
        const ScopedEnvVar p("BATCHLAS_POTRF_ROUTE", "native:lpanel");
        const std::size_t before = v2::pin_warnings_emitted();
        const auto s1 = v2::select<kB, float>(*q, U, Uplo::Upper);
        const auto s2 = v2::select<kB, float>(*q, U, Uplo::Upper);
        EXPECT_FALSE(s1.decision.pin_honoured);
        EXPECT_EQ(s1.key(), "vendor:cusolver");
        EXPECT_EQ(s2.key(), "vendor:cusolver");
        EXPECT_EQ(v2::pin_warnings_emitted() - before, 1u) << "warn once per pin, not per call";
        EXPECT_TRUE(dispatch::is_vendor(backend::potrf_route<kB, float>(*q, U, Uplo::Upper, true)))
            << "the legacy path silently does the same";
        v2::set_pin_policy(ds::PinPolicy::Strict);
        try {
            (void)v2::select<kB, float>(*q, U, Uplo::Upper);
            ADD_FAILURE() << "strict pin did not throw";
        } catch (const std::invalid_argument& e) {
            EXPECT_NE(std::string(e.what()).find("Lower only"), std::string::npos) << e.what();
        }
        v2::set_pin_policy(std::nullopt);
    }
    const MatrixView<float, MatrixFormat::Dense> L(nullptr, 128, 128, 128, 128 * 128, 64);
    {
        const ScopedEnvVar p("BATCHLAS_POTRF_ROUTE", "native:cta");
        const auto s = v2::select<kB, float>(*q, L, Uplo::Lower);
        EXPECT_FALSE(s.decision.pin_honoured);
        const std::string t = v2::explain<kB, float>(s);
        EXPECT_NE(t.find("native:cta       unfit"), std::string::npos) << t;
    }
    {
        const ScopedEnvVar p("BATCHLAS_POTRF_ROUTE", "native:blocked");
        const auto s = v2::select<kB, float>(*q, L, Uplo::Lower);
        EXPECT_TRUE(s.decision.pin_honoured);
        EXPECT_EQ(s.key(), "native:blocked");
    }
    {   // bare origin: legacy has no native for Upper n=128 and runs the vendor; T12 times both
        const ScopedEnvVar p("BATCHLAS_POTRF_ROUTE", "native");
        const auto s = v2::select<kB, float>(*q, U, Uplo::Upper);
        EXPECT_TRUE(s.decision.pin_honoured);
        EXPECT_EQ(s.key(), "native:cta_wg");
    }
}

// ---- T5 saturating batches: every item the same matrix, every result bit-identical to item 0 --
template <class T>
void saturating_case(Queue& q, int n, int batch, Uplo uplo, const char* pin, const char* want,
                     bool compare_legacy) {
    const ScopedEnvVar p("BATCHLAS_POTRF_ROUTE", pin);
    Mat<T> Anew(n, batch);
    fill_spd(Anew, 99, /*same=*/true);
    const RunResult r = run_new(q, Anew, uplo);
    ASSERT_EQ(r.key, want);
    const long vs_item0 = triangle_diff(Anew, Anew, uplo, 0);
    long vs_legacy = 0;
    if (compare_legacy) {
        Mat<T> Aold(n, batch);
        fill_spd(Aold, 99, true);
        run_legacy(q, Aold, uplo);
        vs_legacy = triangle_diff(Anew, Aold, uplo);
    }
    std::printf("T5 %s %s n=%d b=%d %s: ws=%zu items!=item0=%ld items!=legacy=%ld residual=%.2e\n",
                tname<T>().c_str(), uplo == Uplo::Lower ? "L" : "U", n, batch, want, r.ws, vs_item0,
                vs_legacy, r.worst);
    EXPECT_EQ(vs_item0, 0);
    EXPECT_EQ(vs_legacy, 0);
    EXPECT_LE(r.worst, tol<T>(n));
    EXPECT_EQ(r.canary_touched, 0);
}

TEST(PotrfSelectV2, T5_BlockedSaturatingBatchBitIdentical) {
    auto q = gpu();
    if (!q) GTEST_SKIP();
    saturating_case<double>(*q, 36, 8192, Uplo::Lower, "native:blocked", "native:blocked", true);
}

// The new SLM tier: work-group scope, L up to 256, G = 1 (rule 5; wg > 32 so cross-sub-group
// SLM traffic is exercised).
TEST(PotrfSelectV2, T5_CtaWgSaturatingBatchBitIdentical) {
    auto q = gpu();
    if (!q) GTEST_SKIP();
    saturating_case<float>(*q, 128, 2048, Uplo::Upper, "native:cta_wg", "native:cta_wg", false);
    saturating_case<float>(*q, 128, 2048, Uplo::Lower, "native:cta_wg", "native:cta_wg", false);
    saturating_case<std::complex<float>>(*q, 64, 4096, Uplo::Upper, "native:cta_wg", "native:cta_wg", false);
    saturating_case<std::complex<double>>(*q, 48, 4096, Uplo::Lower, "native:cta_wg", "native:cta_wg", false);
}

// ---- T6 explain() and the extrapolation flag, both directions --------------------------------
TEST(PotrfSelectV2, T6_ExplainF128) {
    auto q = gpu();
    if (!q) GTEST_SKIP();
    const ScopedEnvVar no_pin("BATCHLAS_POTRF_ROUTE", nullptr);
    const MatrixView<float, MatrixFormat::Dense> A(nullptr, 128, 128, 128, 128 * 128, 8192);
    const auto s = v2::select<kB, float>(*q, A, Uplo::Lower, v2::Mode::DecideOnly);
    const std::string t = v2::explain<kB, float>(s);
    std::printf("%s", t.c_str());
    EXPECT_EQ(s.key(), "native:lpanel");
    EXPECT_EQ(s.decision.by, ds::Strategy::Model);
    EXPECT_EQ(s.decision.extrapolated, 0) << "f128 b=8192 is inside the fitted box";
    EXPECT_NE(t.find("PotrfLpanelKernel<float,8>"), std::string::npos);
    EXPECT_NE(t.find("groups=8192 wg=128"), std::string::npos);
    EXPECT_NE(t.find("native:tiny      illegal  n > tiny cap"), std::string::npos);
    const MatrixView<float, MatrixFormat::Dense> X(nullptr, 128, 128, 128, 128 * 128, 1 << 22);
    const auto e = v2::select<kB, float>(*q, X, Uplo::Lower, v2::Mode::DecideOnly);
    std::printf("%s", v2::explain<kB, float>(e).c_str());
    EXPECT_EQ(e.decision.by, ds::Strategy::Model);
    EXPECT_EQ(e.decision.extrapolated, 1) << "batch 4M is outside every fitted box";
}

// ---- T7 host cost of one decision, legacy vs new (reported, not asserted) ---------------------
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
    const double new_us = time([&] { (void)v2::select<kB, float>(*q, A, Uplo::Lower, v2::Mode::DecideOnly); });
    const double full_us = time([&] { (void)v2::select<kB, float>(*q, A, Uplo::Lower); });
    std::printf("T7 one routing decision, float L n=128 b=8192: legacy potrf_route %.2f us, "
                "select decide-only %.2f us, select + size %.2f us\n", old_us, new_us, full_us);
}

// ---- T8 rule 9: LAUNCH cta_wg at its advertised ceiling, per dtype; one past it is unfit ------
template <class T>
void ctawg_ceiling(Queue& q) {
    const ScopedEnvVar p("BATCHLAS_POTRF_ROUTE", "native:cta_wg");
    const int row = idx_of<T>("native:cta_wg");
    ASSERT_GE(row, 0);
    constexpr int kScan = 2048;
    int ceiling = 0;
    for (int n = 1; n <= kScan; ++n) {
        const MatrixView<T, MatrixFormat::Dense> A(nullptr, n, n, n, n * n, 256);
        const auto s = v2::select<kB, T, true>(q, A, Uplo::Upper, v2::Mode::DecideOnly);
        if (s.candidates.row[row].eligible()) ceiling = n;
    }
    std::printf("T8 %s cta_wg advertised ceiling n=%d\n", tname<T>().c_str(), ceiling);
    ASSERT_GT(ceiling, 0);
    ASSERT_LT(ceiling, kScan) << "no capacity gate: every order advertised";
    for (Uplo uplo : {Uplo::Upper, Uplo::Lower}) {
        Mat<T> A(ceiling, 256);
        fill_spd(A, 11, false);
        RunResult r;
        ASSERT_NO_THROW(r = run_new(q, A, uplo));
        EXPECT_EQ(r.key, "native:cta_wg");
        EXPECT_LE(r.worst, tol<T>(ceiling));
        EXPECT_EQ(r.info_bad, 0);
    }
}

TEST(PotrfSelectV2, T8_CtaWgCeilingLaunch) {
    auto q = gpu();
    if (!q) GTEST_SKIP();
    ctawg_ceiling<float>(*q);
    ctawg_ceiling<double>(*q);
    ctawg_ceiling<std::complex<float>>(*q);
    ctawg_ceiling<std::complex<double>>(*q);
}

// ---- T9 heterogeneous batch: no native may claim it (a silent wrong answer otherwise) ----------
TEST(PotrfSelectV2, T9_HeterogeneousBatch) {
    auto q = gpu();
    if (!q) GTEST_SKIP();
    const ScopedEnvVar no_pin("BATCHLAS_POTRF_ROUTE", nullptr);
    for (int n : {16, 36, 128}) {
        Matrix<float, MatrixFormat::Dense> A(n, n, 4);
        UnifiedVector<int> act_r(4), act_c(4);
        for (int b = 0; b < 4; ++b) { act_r[b] = n - b; act_c[b] = n - b; }
        const auto V = A.view().with_active_dims(act_r.to_span(), act_c.to_span());
        ASSERT_TRUE(V.is_heterogeneous());
        const auto s = v2::select<kB, float>(*q, V, Uplo::Lower, v2::Mode::DecideOnly);
        const std::string t = v2::explain<kB, float>(s);
        EXPECT_EQ(s.key(), "vendor:cusolver") << t;
        EXPECT_EQ(old_key(backend::potrf_route<kB, float>(*q, V, Uplo::Lower, true)), "vendor:cusolver");
        for (int i = 0; i < s.candidates.size; ++i) {
            if (dispatch::is_native(s.candidates.row[i].route)) {
                EXPECT_FALSE(s.candidates.row[i].legal.ok) << s.candidates.row[i].key << "\n" << t;
            }
        }
        const auto svf = v2::select<kB, float, false>(*q, V, Uplo::Lower, v2::Mode::DecideOnly);
        EXPECT_LT(svf.decision.index, 0) << "vendor-free: no route, not a native one";
    }
}

// ---- T10 the model DECLINES when an eligible vendor has no cost row --------------------------
TEST(PotrfSelectV2, T10_UnpricedVendorDeclinesModel) {
    ds::Candidates c;
    c.size = 2;
    c.dims = 2;
    c.coords = {64, 1024, 0, 0};
    c.row[0] = {dispatch::Route{dispatch::Origin::Native, dispatch::Algorithm::CTA}, "native:cta",
                ds::Verdict::yes(), {}};
    c.row[1] = {dispatch::Route{dispatch::Origin::Vendor, dispatch::Algorithm::Auto}, "vendor:cusolver",
                ds::Verdict::yes(), {}};
    for (auto& r : c.row) { r.lp.fits = true; r.lp.launches = 1; r.lp.groups = 1024; r.lp.wg_size = 32; }
    const launch_plan::CostConstants k{1e-6, 1e-9, 0, 0, 0, 0};
    const ds::CostRow native_only[] = {{"native:cta", 0, k, {1, 4096, 1, 1 << 20}, 10}};
    const ds::CostRow both[] = {{"native:cta", 0, k, {1, 4096, 1, 1 << 20}, 10},
                                {"vendor:cusolver", 0, k, {1, 4096, 1, 1 << 20}, 10}};
    launch_plan::DeviceFacts f;
    f.compute_units = 188;
    const ds::CostBook missing{true, 0.05, native_only}, full{true, 0.05, both};
    EXPECT_FALSE((ds::ModelChooser{&missing, f}.choose(c, 3u).has_value()))
        << "an unpriced vendor must not read as 'no vendor in play'";
    EXPECT_TRUE((ds::ModelChooser{&full, f}.choose(c, 3u).has_value()));
    EXPECT_TRUE((ds::ModelChooser{&missing, f}.choose(c, 1u).has_value())) << "vendor masked out";
}

// ---- T11 run() checks the Selection against the call; settings changes are not an error -----
TEST(PotrfSelectV2, T11_RunValidatesSelection) {
    auto q = gpu();
    if (!q) GTEST_SKIP();
    const ScopedEnvVar no_pin("BATCHLAS_POTRF_ROUTE", nullptr);
    Mat<float> A(36, 64), B(40, 64);
    fill_spd(A, 1, false);
    fill_spd(B, 1, false);
    UnifiedVector<std::byte> ws(std::size_t{1} << 20);
    UnifiedVector<int32_t> info(64, 0);
    const auto decide_only = v2::select<kB, float>(*q, A.view(), Uplo::Lower, v2::Mode::DecideOnly);
    EXPECT_THROW(((void)v2::run<kB, float>(*q, decide_only, A.view(), Uplo::Lower, ws.to_span(), info.to_span())),
                 std::invalid_argument);
    const auto sel = v2::select<kB, float>(*q, A.view(), Uplo::Lower);
    EXPECT_THROW(((void)v2::run<kB, float>(*q, sel, B.view(), Uplo::Lower, ws.to_span(), info.to_span())),
                 std::invalid_argument) << "a selection made for n=36 run on n=40";
    EXPECT_THROW(((void)v2::run<kB, float>(*q, sel, A.view(), Uplo::Upper, ws.to_span(), info.to_span())),
                 std::invalid_argument) << "made for Lower, run as Upper";
    EXPECT_THROW(((void)v2::run<kB, float>(*q, sel, A.view(), Uplo::Lower,
                                           Span<std::byte>(ws.data(), sel.workspace - 1), info.to_span())),
                 std::length_error);
    detail::reload_settings();   // what ScopedEnvVar and configure() do on another thread
    EXPECT_NO_THROW(((void)v2::run<kB, float>(*q, sel, A.view(), Uplo::Lower, ws.to_span(), info.to_span())));
    q->wait();
    EXPECT_EQ(info.data()[0], 0);
}

// ---- T12 the behaviour changes the extension ships, timed (reported; screen, not a verdict) ---
namespace {

template <class T>
double time_route(Queue& q, Mat<T>& A, const Mat<T>& pristine, Uplo uplo, const char* pin,
                  std::string& key) {
    const ScopedEnvVar p("BATCHLAS_POTRF_ROUTE", pin);
    const auto sel = v2::select<kB, T>(q, A.view(), uplo);
    key = std::string(sel.key());
    UnifiedVector<std::byte> ws(sel.workspace);
    UnifiedVector<int32_t> info(A.b, 0);
    std::vector<double> t;
    for (int rep = 0; rep < 13; ++rep) {
        // Fresh SPD input every rep, copied on the device outside the timed region.
        (void)MatrixView<T, MatrixFormat::Dense>::copy(q, A.view(), pristine.view());
        q.wait();
        const auto t0 = std::chrono::steady_clock::now();
        (void)v2::run<kB, T>(q, sel, A.view(), uplo, ws.to_span(), info.to_span());
        q.wait();
        if (rep >= 3) t.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

template <class T>
void time_case(Queue& q, int n, int batch) {
    Mat<T> A(n, batch), P(n, batch);
    fill_spd(P, 3, false);
    std::string kv, kn;
    const double a1 = time_route(q, A, P, Uplo::Upper, "vendor", kv);
    const double b1 = time_route(q, A, P, Uplo::Upper, "native:cta_wg", kn);
    const double b2 = time_route(q, A, P, Uplo::Upper, "native:cta_wg", kn);
    const double a2 = time_route(q, A, P, Uplo::Upper, "vendor", kv);
    const double v = std::min(a1, a2), c = std::min(b1, b2);
    std::printf("T12 %-7s U n=%-4d b=%-5d %s %9.1f us  %s %9.1f us  cta_wg/vendor=%.2f\n",
                tname<T>().c_str(), n, batch, kv.c_str(), v, kn.c_str(), c, c / v);
}

}  // namespace

TEST(PotrfSelectV2, T12_CtaWgVsVendorTiming) {
    auto q = gpu();
    if (!q) GTEST_SKIP();
    for (int b : {1024, 8192}) {
        for (int n : {78, 96, 128, 140}) time_case<float>(*q, n, b);
        for (int n : {64, 96}) time_case<double>(*q, n, b);
        for (int n : {64, 96}) time_case<std::complex<float>>(*q, n, b);
        for (int n : {48, 64}) time_case<std::complex<double>>(*q, n, b);
    }
}
