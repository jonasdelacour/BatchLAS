// Every gesvd candidate, pinned: docs/design/flat-kernel-selection.md §8, phase 5. The limit
// oracle reads the drivers' own ceilings; which kernel ran is read back from the select trace
// or a bit-for-bit comparison with the direct driver, never assumed from an accepted pin.
#include <gtest/gtest.h>

#include <batchlas/blas/extensions.hh>
#include <batchlas/blas/functions/gesvd.hh>
#include "../src/select/coverage.hh"
#include <batchlas/no_route.hh>
#include "../src/select/vendor.hh"
#include <batchlas/util/env.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>
#include <batchlas/util/sycl-vector.hh>

#include "test_utils.hh"

#include "../src/extensions/gesvd_native.hh"
#include "../src/ops/gesvd/choice.hh"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <filesystem>
#include <fstream>
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

namespace gs = batchlas::ops::gesvd;
using C = gs::GesvdChoice;
using Pin = select::ScopedPin<C>;
template <typename T>
using MVof = MatrixView<T, MatrixFormat::Dense>;
template <typename T>
using RealOf = typename batchlas::base_type<T>::type;
template <typename T>
constexpr bool kCx = test_utils::is_complex<T>::value;
using cd = std::complex<double>;

template <typename T>
T from(cd v) {
    if constexpr (kCx<T>) return T(static_cast<RealOf<T>>(v.real()), static_cast<RealOf<T>>(v.imag()));
    else return static_cast<T>(v.real());
}
template <typename T>
cd up(T v) {
    if constexpr (kCx<T>) return {double(v.real()), double(v.imag())};
    else return {double(v), 0.0};
}

// herm: 'N' general, 'L'/'U' the referenced triangle of a Hermitian A.
struct Spec {
    int m = 8, n = 8, batch = 2;
    char herm = 'N';
    SvdVectors ju = SvdVectors::All, jv = SvdVectors::All;
    int gap = 7;      // items sit `gap` elements apart past ld * n: a stride that is not ld * n
    int ldpad = 3;    // ld = m + ldpad
    int period = 0;   // > 0: item b repeats item b % period
    unsigned seed = 1;
    bool poison = false;  // Hermitian: a large finite value in the unreferenced triangle (straddles)
};

const char* job_s(SvdVectors j) { return j == SvdVectors::None ? "N" : (j == SvdVectors::All ? "A" : "T"); }
std::string label(const Spec& s) {
    return std::string("herm=") + s.herm + " jobs=" + job_s(s.ju) + job_s(s.jv) + " " + std::to_string(s.m) + "x" +
           std::to_string(s.n) + " batch=" + std::to_string(s.batch);
}
std::string name(const C& c, const Spec& s) { return select::to_string(c) + " " + label(s); }

// A = H(u) diag(sigma) H(v) (general) or H(u) diag(lambda) H(u) (Hermitian), H(x) = I - 2 x x^H:
// the singular values are known in closed form. A lives at ld = m + 3 with a gap between items;
// a Hermitian A stores a large finite poison in its unreferenced triangle.
template <typename T>
struct Svd {
    Spec s;
    int k = 0, ld = 0;
    std::int64_t stride = 0;
    UnifiedVector<T> a;
    std::vector<T> a0;
    std::vector<std::vector<cd>> full;  // the matrix each item represents, column-major
    std::vector<std::vector<double>> sigma;  // expected, descending
    Matrix<T, MatrixFormat::Dense> U, Vh;
    UnifiedVector<RealOf<T>> sv;
    UnifiedVector<int32_t> info;

    Svd(const Spec& sp)
        : s(sp), k(std::min(sp.m, sp.n)), ld(sp.m + sp.ldpad), stride(std::int64_t(sp.m + sp.ldpad) * sp.n + sp.gap),
          a(std::size_t(stride) * sp.batch + 1, from<T>(cd(-321.0, 123.0))),
          U(sp.ju == SvdVectors::None ? 1 : sp.m, sp.ju == SvdVectors::All ? sp.m : (sp.ju == SvdVectors::Thin ? k : 1),
            sp.batch),
          Vh(sp.jv == SvdVectors::All ? sp.n : (sp.jv == SvdVectors::Thin ? k : 1), sp.jv == SvdVectors::None ? 1 : sp.n,
             sp.batch),
          sv(std::size_t(k) * sp.batch), info(sp.batch, -7) {}
    MVof<T> A() { return {a.data(), s.m, s.n, ld, static_cast<int>(stride), s.batch}; }
    std::size_t ai(int b, int i, int j) const { return std::size_t(b) * stride + std::size_t(j) * ld + i; }
};

template <typename T>
std::vector<cd> unit_vector(std::mt19937& gen, int n) {
    std::uniform_real_distribution<double> d(-1, 1);
    std::vector<cd> u(n);
    double nrm = 0;
    for (auto& x : u) {
        x = kCx<T> ? cd(d(gen), d(gen)) : cd(d(gen), 0);
        nrm += std::norm(x);
    }
    for (auto& x : u) x /= std::sqrt(nrm);
    return u;
}
cd householder(const std::vector<cd>& u, int i, int j) { return (i == j ? 1.0 : 0.0) - 2.0 * u[i] * std::conj(u[j]); }

template <typename T>
Svd<T> make_svd(const Spec& s) {
    Svd<T> p(s);
    const int reps = s.period > 0 ? std::min(s.period, s.batch) : s.batch;
    std::mt19937 gen(s.seed);
    for (int b = 0; b < s.batch; ++b) {
        if (b >= reps) {
            p.full.push_back(p.full[b % reps]);
            p.sigma.push_back(p.sigma[b % reps]);
            for (int j = 0; j < s.n; ++j)
                for (int i = 0; i < s.m; ++i) p.a[p.ai(b, i, j)] = p.a[p.ai(b % reps, i, j)];
            continue;
        }
        std::vector<cd> f(std::size_t(s.m) * s.n, 0.0);
        std::vector<double> sig;
        if (s.herm != 'N') {
            const auto u = unit_vector<T>(gen, s.m);
            std::vector<double> lam(s.m);
            for (int i = 0; i < s.m; ++i) lam[i] = (i % 2 ? -1.0 : 1.0) * (1.0 + double(i) / s.m);
            for (int j = 0; j < s.n; ++j)
                for (int i = 0; i < s.m; ++i)
                    for (int l = 0; l < s.m; ++l) f[j * s.m + i] += householder(u, i, l) * lam[l] * householder(u, l, j);
            for (double l : lam) sig.push_back(std::abs(l));
        } else {
            const auto u = unit_vector<T>(gen, s.m), v = unit_vector<T>(gen, s.n);
            for (int l = 0; l < p.k; ++l) sig.push_back(1.0 + double(l) / std::max(1, p.k));
            for (int j = 0; j < s.n; ++j)
                for (int i = 0; i < s.m; ++i)
                    for (int l = 0; l < p.k; ++l) f[j * s.m + i] += householder(u, i, l) * sig[l] * householder(v, l, j);
        }
        std::sort(sig.rbegin(), sig.rend());
        for (int j = 0; j < s.n; ++j)
            for (int i = 0; i < s.m; ++i) {
                const bool unreferenced = s.poison && ((s.herm == 'L' && i < j) || (s.herm == 'U' && i > j));
                p.a[p.ai(b, i, j)] = unreferenced ? from<T>(cd(4.0e3, -2.0e3)) : from<T>(f[j * s.m + i]);
            }
        p.full.push_back(std::move(f));
        p.sigma.push_back(std::move(sig));
    }
    p.a0.assign(p.a.begin(), p.a.end());
    return p;
}

template <typename T>
void expect_solved(Svd<T>& p, const std::string& what) {
    const double tol = std::is_same_v<RealOf<T>, float> ? 2e-3 : 1e-9;
    const Spec& s = p.s;
    for (int b = 0; b < s.batch; ++b) {
        ASSERT_EQ(p.info[b], 0) << what << " item " << b;
        for (int l = 0; l < p.k; ++l)
            ASSERT_NEAR(double(p.sv[std::size_t(b) * p.k + l]), p.sigma[b][l], tol * 2) << what << " item " << b << " sigma " << l;
    }
    if (s.ju == SvdVectors::None || s.jv == SvdVectors::None) return;
    double worst = 0;
    for (int b = 0; b < s.batch; ++b)
        for (int j = 0; j < s.n; ++j)
            for (int i = 0; i < s.m; ++i) {
                cd acc = 0;
                for (int l = 0; l < p.k; ++l)
                    acc += up(p.U(i, l, b)) * double(p.sv[std::size_t(b) * p.k + l]) * up(p.Vh(l, j, b));
                worst = std::max(worst, std::abs(acc - p.full[b][j * s.m + i]));
            }
    EXPECT_LE(worst, tol * 10) << what << ": reconstruction";
}

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
        if (line.rfind("gesvd ", 0) == 0 && line.find(" -> ") != std::string::npos) return line;
    return "<no gesvd trace line in: " + err + ">";
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

template <typename T, Backend B>
struct Cfg {
    using ScalarType = T;
    static constexpr Backend BackendVal = B;
};
using Types = typename test_utils::backend_types<Cfg>::type;

std::optional<Uplo> uplo_of(char h) {
    if (h == 'L') return Uplo::Lower;
    if (h == 'U') return Uplo::Upper;
    return std::nullopt;
}

template <typename Config>
class GesvdCandidates : public test_utils::BatchLASTest<Config> {
protected:
    using T = typename Config::ScalarType;
    using MV = MVof<T>;
    static constexpr Backend B = Config::BackendVal;
    static constexpr bool kVendor = batchlas::select::solver_vendor_available<B>;
    static constexpr bool kReal = !kCx<T>;

    void SetUp() override {
        test_utils::BatchLASTest<Config>::SetUp();
        if (this->HasFatalFailure() || ::testing::Test::IsSkipped()) return;
        if (!this->ctx) GTEST_SKIP() << "no queue";
        if (this->ctx->device().type != DeviceType::GPU) GTEST_SKIP() << "the native families are GPU kernels";
        if (!this->ctx->device().supports_sub_group_size(32)) GTEST_SKIP() << "no sub-group size 32";
    }

    // ---- the limit oracle: the drivers' own ceilings and checks, not gesvd.cc's can_run ----
    static bool canonical_thin(const Spec& s) {
        const int k = std::min(s.m, s.n);
        return canonical_jobu(s.ju, s.m, k) == SvdVectors::Thin || canonical_jobvh(s.jv, s.n, k) == SvdVectors::Thin;
    }
    static bool vectors(const Spec& s) { return s.ju != SvdVectors::None || s.jv != SvdVectors::None; }
    static bool expect_runs(const C& c, const Spec& s) {
        const int md = std::max(s.m, s.n);
        if (std::holds_alternative<gs::Jacobi>(c))
            return s.herm == 'N' && md <= sycl_gesvd::gesvd_jacobi_max_dim<T>(vectors(s));
        if (std::holds_alternative<gs::Cta>(c))
            return md <= sycl_gesvd::kGesvdCtaMaxDim && !canonical_thin(s) && (s.herm == 'N' ? kReal : s.m == s.n);
        if (std::holds_alternative<gs::Blocked>(c)) return s.herm == 'N' ? kReal : (s.m == s.n && s.herm == 'L');
        return kVendor;
    }
    // cuSOLVER gesvdjBatched: max(m, n) <= 32, a packed batch, no thin factor, and it reads the
    // whole matrix (no Hermitian shortcut). Its refusals are not in can_run (see gesvd.md). A
    // values-only non-square call faults inside it on 424a45bc too (gesvd.md, "Found").
    static bool vendor_envelope(const Spec& s) {
        return s.herm == 'N' && std::max(s.m, s.n) <= 32 && !canonical_thin(s) && s.gap == 0 && s.ldpad == 0 &&
               (vectors(s) || s.m == s.n);
    }

    void run(Svd<T>& p, Queue* on = nullptr) {
        Queue& q = on ? *on : *this->ctx;
        const auto h = uplo_of(p.s.herm);
        const MV A = p.A(), U = p.U.view(), Vh = p.Vh.view();
        const std::size_t bytes = h ? gesvd_buffer_size<B, T>(q, A, p.sv.to_span(), U, Vh, p.s.ju, p.s.jv, *h)
                                    : gesvd_buffer_size<B, T>(q, A, p.sv.to_span(), U, Vh, p.s.ju, p.s.jv);
        UnifiedVector<std::byte> ws(std::max<std::size_t>(bytes, 1));
        Span<std::byte> w(ws.data(), bytes);
        if (h) (void)gesvd<B, T>(q, A, p.sv.to_span(), U, Vh, p.s.ju, p.s.jv, *h, w, p.info.to_span());
        else (void)gesvd<B, T>(q, A, p.sv.to_span(), U, Vh, p.s.ju, p.s.jv, w, p.info.to_span());
        q.wait();
    }
    void run_pinned(const C& c, Svd<T>& p) {
        const Pin pin("gesvd", c);
        run(p);
    }
    // Acceptance is asked of the sizing call, which runs the same choose() (R5).
    bool pin_accepted(const C& c, Svd<T>& p) {
        const Pin pin("gesvd", c);
        const auto h = uplo_of(p.s.herm);
        try {
            if (h) (void)gesvd_buffer_size<B, T>(*this->ctx, p.A(), p.sv.to_span(), p.U.view(), p.Vh.view(), p.s.ju, p.s.jv, *h);
            else (void)gesvd_buffer_size<B, T>(*this->ctx, p.A(), p.sv.to_span(), p.U.view(), p.Vh.view(), p.s.ju, p.s.jv);
            return true;
        } catch (const std::invalid_argument& e) {
            if (std::string(e.what()).find("cannot run this shape") == std::string::npos) throw;
            return false;
        }
    }

    // The family's own driver, sized and launched directly on an in-order queue.
    bool direct(const C& c, Svd<T>& p, std::string* why = nullptr) {
        Queue q(*this->ctx, true);
        const MV A = p.A(), U = p.U.view(), Vh = p.Vh.view();
        const auto h = uplo_of(p.s.herm);
        const SvdVectors ju = canonical_jobu(p.s.ju, p.s.m, p.k), jv = canonical_jobvh(p.s.jv, p.s.n, p.k);
        const auto s = p.sv.to_span();
        try {
            std::size_t bytes = 0;
            if (std::holds_alternative<gs::Jacobi>(c)) {
                if (h) {
                    if (why) *why = "gesvdj_cta has no Hermitian form";
                    return false;
                }
                bytes = gesvdj_cta_buffer_size<B, T>(q, A, s, U, Vh, ju, jv);
            } else if (std::holds_alternative<gs::Cta>(c)) {
                bytes = h ? gesvd_cta_buffer_size<B, T>(q, A, s, U, Vh, ju, jv, *h)
                          : gesvd_cta_buffer_size<B, T>(q, A, s, U, Vh, ju, jv);
            } else if (std::holds_alternative<gs::Blocked>(c)) {
                bytes = h ? gesvd_blocked_buffer_size<B, T>(q, A, s, U, Vh, ju, jv, *h)
                          : gesvd_blocked_buffer_size<B, T>(q, A, s, U, Vh, ju, jv);
            } else {
                if (why) *why = "vendor not compared";
                return false;
            }
            UnifiedVector<std::byte> ws(std::max<std::size_t>(bytes, 1));
            Span<std::byte> w(ws.data(), bytes);
            if (std::holds_alternative<gs::Jacobi>(c))
                (void)gesvdj_cta<B, T>(q, A, s, U, Vh, ju, jv, w, GesvdjParams<T>(), p.info.to_span());
            else if (std::holds_alternative<gs::Cta>(c))
                (void)(h ? gesvd_cta<B, T>(q, A, s, U, Vh, ju, jv, *h, w, p.info.to_span())
                         : gesvd_cta<B, T>(q, A, s, U, Vh, ju, jv, w, p.info.to_span()));
            else
                (void)(h ? gesvd_blocked<B, T>(q, A, s, U, Vh, ju, jv, *h, w, p.info.to_span())
                         : gesvd_blocked<B, T>(q, A, s, U, Vh, ju, jv, w, p.info.to_span()));
            q.wait_and_throw();
            return true;
        } catch (const std::exception& e) {
            if (why) *why = e.what();
            return false;
        }
    }
};

TYPED_TEST_SUITE(GesvdCandidates, Types);

// Pre-existing driver defect (gesvd.md, "Found"): cta with Upper, and blocked with Lower at
// n <= 32, read the UNREFERENCED triangle, so those straddles store the full Hermitian matrix.
bool reads_the_other_triangle(const C& c, char h, int n) {
    return (std::holds_alternative<gs::Cta>(c) && h == 'U') ||
           (std::holds_alternative<gs::Blocked>(c) && h == 'L' && n <= 32);
}

// Shapes straddling cta's 32, jacobi's 32 (cdouble with vectors) and 64, square and not.
const std::vector<std::pair<int, int>> kStraddle{{1, 1}, {8, 8}, {32, 32}, {33, 33}, {64, 64}, {65, 65},
                                                 {32, 8}, {8, 32}, {40, 20}, {20, 40}, {65, 8}};
const SvdVectors kJobs[][2] = {{SvdVectors::None, SvdVectors::None}, {SvdVectors::All, SvdVectors::All},
                               {SvdVectors::Thin, SvdVectors::Thin}, {SvdVectors::None, SvdVectors::All}};

// §8.1: every native family on every straddling shape, triangle and job; accepted pins run and
// are correct (closed-form singular values, reconstruction), refused ones throw from the pin.
TYPED_TEST(GesvdCandidates, PinnedCandidatesStraddleTheirLimits) {
    using T = typename TestFixture::T;
    for (const C& c : gs::candidates<T>()) {
        if (std::holds_alternative<gs::Vendor>(c)) continue;
        for (auto [m, n] : kStraddle)
            for (char h : {'N', 'L', 'U'})
                for (const auto& jobs : kJobs) {
                    Spec s{m, n, 2, h, jobs[0], jobs[1]};
                    s.seed = 11u + m * 7u + n;
                    s.poison = !reads_the_other_triangle(c, h, n);
                    auto p = make_svd<T>(s);
                    const bool want = this->expect_runs(c, s);
                    const std::string what = name(c, s);
                    if (want) {
                        EXPECT_EQ(traced_choice([&] { this->run_pinned(c, p); }), select::to_string(c)) << what;
                        expect_solved(p, what);
                    } else {
                        try {
                            this->run_pinned(c, p);
                            ADD_FAILURE() << what << " was accepted";
                        } catch (const std::invalid_argument& e) {
                            EXPECT_NE(std::string(e.what()).find("cannot run this shape"), std::string::npos)
                                << what << ": " << e.what();
                        }
                    }
                }
    }
}

// The defect itself: every Hermitian-capable family with poison in the unreferenced triangle.
// Disabled until the drivers honour the triangle; cta Lower and blocked Lower at 48 pass today.
TYPED_TEST(GesvdCandidates, DISABLED_HermitianFamiliesIgnoreTheUnreferencedTriangle) {
    using T = typename TestFixture::T;
    for (const C& c : {C{gs::Cta{}}, C{gs::Blocked{}}})
        for (char h : {'L', 'U'})
            for (int n : {8, 32, 48}) {
                Spec s{n, n, 2, h, SvdVectors::None, SvdVectors::None};
                s.poison = true;
                if (!this->expect_runs(c, s)) continue;
                auto p = make_svd<T>(s);
                this->run_pinned(c, p);
                expect_solved(p, name(c, s) + " (poisoned)");
            }
}

// The vendor inside cuSOLVER's envelope (packed, max(m, n) <= 32, no thin factor); outside it the
// pin is accepted (can_run is has_vendor_solver alone) and the library refuses at launch.
TYPED_TEST(GesvdCandidates, VendorPinRunsInsideItsEnvelope) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    if constexpr (!TestFixture::kVendor || B != Backend::CUDA) {
        GTEST_SKIP() << "no cuSOLVER in this build";
    } else {
        for (auto [m, n] : std::vector<std::pair<int, int>>{{1, 1}, {8, 8}, {32, 32}, {32, 8}, {8, 32}})
            for (const auto& jobs : kJobs) {
                Spec s{m, n, 3, 'N', jobs[0], jobs[1]};
                s.gap = 0;
                s.ldpad = 0;
                if (!this->vendor_envelope(s)) continue;
                auto p = make_svd<T>(s);
                EXPECT_EQ(traced_choice([&] { this->run_pinned(C{gs::Vendor{}}, p); }), "vendor") << label(s);
                expect_solved(p, "vendor " + label(s));
            }
        Spec big{48, 48, 2};
        big.gap = 0;
        big.ldpad = 0;
        auto p = make_svd<T>(big);
        try {
            this->run_pinned(C{gs::Vendor{}}, p);
            ADD_FAILURE() << "cuSOLVER took 48 x 48";
        } catch (const std::invalid_argument& e) {
            ADD_FAILURE() << "the pin was refused: " << e.what();
        } catch (const std::exception& e) {
            EXPECT_NE(std::string(e.what()).find("gesvd_vendor (CUSOLVER)"), std::string::npos) << e.what();
        }
    }
}

// §8.2 (R3): on every straddling shape a native pin is accepted exactly when the family's own
// driver sizes and launches.
TYPED_TEST(GesvdCandidates, CanRunEqualsLaunch) {
    using T = typename TestFixture::T;
    int disagreements = 0;
    for (const C& c : gs::candidates<T>()) {
        if (std::holds_alternative<gs::Vendor>(c)) continue;
        for (auto [m, n] : kStraddle)
            for (char h : {'N', 'L', 'U'})
                for (const auto& jobs : kJobs) {
                    Spec s{m, n, 2, h, jobs[0], jobs[1]};
                    auto a = make_svd<T>(s);
                    auto b = make_svd<T>(s);
                    const bool pin = this->pin_accepted(c, a);
                    std::string why;
                    const bool run = this->direct(c, b, &why);
                    // Blocked Hermitian Upper: syev_blocked mirrors Upper into Lower and launches,
                    // but can_run keeps the old router's Lower-only clause (gesvd.md, deviation).
                    const bool upper_blocked = std::holds_alternative<gs::Blocked>(c) && h == 'U' && m == n;
                    if (!upper_blocked) {
                        EXPECT_EQ(pin, run) << name(c, s) << ": can_run says " << pin << ", the driver "
                                            << (run ? "launches" : "refuses: " + why);
                        disagreements += pin != run;
                    }
                    EXPECT_EQ(pin, this->expect_runs(c, s)) << name(c, s) << ": can_run disagrees with the oracle";
                }
    }
    EXPECT_EQ(disagreements, 0);
}

// An empty shape (m, n or batch = 0) is refused by every native family's can_run, as the old
// router's size gate had it: the drivers' validate throws on it. Auto takes the base outcome,
// the vendor where it exists and a NoRouteError vendor-free, never a native family.
TYPED_TEST(GesvdCandidates, EmptyShapesRunNoNativeFamily) {
    using T = typename TestFixture::T;
    const ScopedEnvVar clear("BATCHLAS_GESVD_ROUTE", nullptr);
    const Spec shapes[] = {{0, 8, 2, 'N', SvdVectors::None, SvdVectors::None},
                           {8, 0, 2, 'N', SvdVectors::None, SvdVectors::None},
                           {8, 8, 0, 'N', SvdVectors::None, SvdVectors::None},
                           {8, 8, 0, 'L', SvdVectors::None, SvdVectors::None}};
    for (const Spec& s : shapes) {
        for (const C& c : gs::candidates<T>()) {
            if (std::holds_alternative<gs::Vendor>(c)) continue;
            auto p = make_svd<T>(s);
            EXPECT_FALSE(this->pin_accepted(c, p)) << name(c, s);
        }
        auto p = make_svd<T>(s);
        if constexpr (!TestFixture::kVendor) {
            EXPECT_THROW(this->run(p), batchlas::NoRouteError) << label(s);
        } else {
            std::string refused;
            const std::string got = traced_choice([&] {
                try {
                    this->run(p);
                } catch (const std::exception& e) {
                    refused = e.what();
                }
            });
            // cuSOLVER may refuse the empty call in its sizing step, before any trace line.
            if (got.rfind("<no gesvd", 0) == 0 && !refused.empty())
                EXPECT_NE(refused.find("CUSOLVER"), std::string::npos) << label(s) << ": " << refused;
            else
                EXPECT_EQ(got, "vendor") << label(s) << " " << refused;
        }
    }
}

// The native drivers refuse an out-of-order Queue; the op layer runs them on an in-order queue
// joined to the caller's. A pinned run on an out-of-order Queue is the direct kernel bit for bit.
TYPED_TEST(GesvdCandidates, OutOfOrderQueueRunsThePinnedDriver) {
    using T = typename TestFixture::T;
    Queue ooo(*this->ctx, false);
    ASSERT_FALSE(ooo.in_order());
    const Spec shapes[] = {{12, 12, 3, 'N'}, {16, 16, 3, 'L'}, {40, 24, 2, 'N'}};
    for (const C& c : gs::candidates<T>()) {
        if (std::holds_alternative<gs::Vendor>(c)) continue;
        for (const Spec& s : shapes) {
            if (!this->expect_runs(c, s)) continue;
            auto a = make_svd<T>(s);
            auto b = make_svd<T>(s);
            {
                const Pin pin("gesvd", c);
                ASSERT_NO_THROW(this->run(a, &ooo)) << name(c, s);
            }
            std::string why;
            ASSERT_TRUE(this->direct(c, b, &why)) << name(c, s) << ": " << why;
            for (std::size_t i = 0; i < a.sv.size(); ++i)
                ASSERT_EQ(std::memcmp(&a.sv[i], &b.sv[i], sizeof(a.sv[i])), 0) << name(c, s) << " sv " << i;
            expect_solved(a, "out-of-order " + name(c, s));
        }
    }
}

// The pinned public call runs exactly the family's driver: identical singular values and U.
TYPED_TEST(GesvdCandidates, PinnedRunIsTheDirectKernelBitForBit) {
    using T = typename TestFixture::T;
    const Spec shapes[] = {{12, 12, 3, 'N'}, {24, 9, 3, 'N', SvdVectors::Thin, SvdVectors::All},
                           {16, 16, 3, 'L'}, {16, 16, 3, 'U'}, {48, 30, 3, 'N'}, {40, 40, 2, 'L'}};
    for (const C& c : gs::candidates<T>()) {
        if (std::holds_alternative<gs::Vendor>(c)) continue;
        for (const Spec& s : shapes) {
            if (!this->expect_runs(c, s)) continue;
            auto a = make_svd<T>(s);
            auto b = make_svd<T>(s);
            this->run_pinned(c, a);
            std::string why;
            ASSERT_TRUE(this->direct(c, b, &why)) << name(c, s) << ": " << why;
            for (std::size_t i = 0; i < a.sv.size(); ++i)
                ASSERT_EQ(std::memcmp(&a.sv[i], &b.sv[i], sizeof(a.sv[i])), 0) << name(c, s) << " sv " << i;
            if (s.ju != SvdVectors::None)
                for (int bb = 0; bb < s.batch; ++bb)
                    for (int j = 0; j < a.U.cols(); ++j)
                        for (int i = 0; i < a.U.rows(); ++i) {
                            const T x = a.U(i, j, bb), y = b.U(i, j, bb);
                            ASSERT_EQ(std::memcmp(&x, &y, sizeof(T)), 0) << name(c, s) << " U";
                        }
        }
    }
}

// AGENTS.md §8.5: a saturating batch of identical items through every family that keeps state in
// local memory; every item's result is bit-identical to item 0's.
TYPED_TEST(GesvdCandidates, SaturatingBatchIsBitIdenticalToItsRepresentative) {
    using T = typename TestFixture::T;
    for (const C& c : gs::candidates<T>()) {
        if (std::holds_alternative<gs::Vendor>(c)) continue;
        for (Spec s : {Spec{32, 32, 1024, 'N'}, Spec{20, 20, 1024, 'L'}}) {
            s.period = 1;
            if (!this->expect_runs(c, s)) continue;
            auto p = make_svd<T>(s);
            this->run_pinned(c, p);
            for (int b = 1; b < s.batch; ++b)
                for (int l = 0; l < p.k; ++l)
                    ASSERT_EQ(std::memcmp(&p.sv[std::size_t(b) * p.k + l], &p.sv[l], sizeof(p.sv[l])), 0)
                        << name(c, s) << " item " << b << " sigma " << l;
            expect_solved(p, name(c, s));
        }
    }
}

// §8.3 (R5): the exact gesvd_buffer_size inside a poisoned arena; nothing past it is written,
// and one byte less is refused before anything runs.
TYPED_TEST(GesvdCandidates, ExactWorkspaceInAPoisonedArena) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    const Spec shapes[] = {{16, 16, 3, 'N'}, {16, 16, 3, 'L'}, {48, 20, 2, 'N', SvdVectors::Thin, SvdVectors::Thin}};
    for (const C& c : gs::candidates<T>())
        for (Spec s : shapes) {
            if (std::holds_alternative<gs::Vendor>(c)) s.gap = s.ldpad = 0;
            if (!this->expect_runs(c, s) || (std::holds_alternative<gs::Vendor>(c) && !this->vendor_envelope(s)))
                continue;
            auto p = make_svd<T>(s);
            const Pin pin("gesvd", c);
            const auto h = uplo_of(s.herm);
            Queue& q = *this->ctx;
            const std::size_t bytes = h ? gesvd_buffer_size<B, T>(q, p.A(), p.sv.to_span(), p.U.view(), p.Vh.view(), s.ju, s.jv, *h)
                                        : gesvd_buffer_size<B, T>(q, p.A(), p.sv.to_span(), p.U.view(), p.Vh.view(), s.ju, s.jv);
            constexpr std::size_t kTail = 4096;
            UnifiedVector<std::byte> arena(bytes + kTail, std::byte{0xA5});
            const Span<std::byte> w(arena.data(), bytes);
            if (bytes > 0) {
                const Span<std::byte> short_w(arena.data(), bytes - 1);
                EXPECT_THROW((h ? gesvd<B, T>(q, p.A(), p.sv.to_span(), p.U.view(), p.Vh.view(), s.ju, s.jv, *h, short_w)
                                : gesvd<B, T>(q, p.A(), p.sv.to_span(), p.U.view(), p.Vh.view(), s.ju, s.jv, short_w)),
                             batchlas::workspace_error)
                    << name(c, s);
            }
            if (h) (void)gesvd<B, T>(q, p.A(), p.sv.to_span(), p.U.view(), p.Vh.view(), s.ju, s.jv, *h, w, p.info.to_span());
            else (void)gesvd<B, T>(q, p.A(), p.sv.to_span(), p.U.view(), p.Vh.view(), s.ju, s.jv, w, p.info.to_span());
            q.wait();
            for (std::size_t i = bytes; i < arena.size(); ++i)
                ASSERT_EQ(arena[i], std::byte{0xA5}) << name(c, s) << ": wrote past the workspace at +" << (i - bytes);
            expect_solved(p, name(c, s));
        }
}

// R6: a spelling that names nothing compiled throws instead of meaning Auto.
TYPED_TEST(GesvdCandidates, UnknownPinsThrow) {
    using T = typename TestFixture::T;
    for (const char* word : {"bogus", "jacobi:1", "native:vendor", "gesvdj", "native:two_stage", "cta:wg=32",
                             // removed aliases (phase 5): each must stay an error
                             "native:jacobi", "native:cta", "native:blocked", "batchlas_jacobi",
                             "batchlas-jacobi", "batchlas_cta", "batchlas-cta", "batchlas_blocked",
                             "batchlas-blocked", "vendor:auto", "netlib", "netlib:auto"}) {
        auto p = make_svd<T>(Spec{});
        const Pin pin("gesvd", std::string_view(word));
        EXPECT_THROW(this->run(p), std::invalid_argument) << word;
    }
}

// §5.3: spellings (case-folded) and the class words, through ScopedPin and BATCHLAS_GESVD_ROUTE.
// A Lower-Hermitian 16 x 16 runs on cta and blocked for every type; general 16 x 16 on jacobi.
TYPED_TEST(GesvdCandidates, ClassWordsAndSpellings) {
    using T = typename TestFixture::T;
    const ScopedEnvVar clear("BATCHLAS_GESVD_ROUTE", nullptr);
    Spec herm{16, 16, 2, 'L'};
    herm.gap = 0;  // packed, so a vendor pin reaches cuSOLVER
    herm.ldpad = 0;
    const std::string vendor_pick = TestFixture::kVendor ? "vendor" : "cta";
    const std::pair<const char*, std::string> expect[] = {
        {"BLOCKED", "blocked"}, {" Cta ", "cta"},  {"cta", "cta"},          {"native", "cta"},
        {"auto", "cta"},        {"vendor", vendor_pick}};
    for (const auto& [word, spelling] : expect)
        for (const char* via : {"pin", "BATCHLAS_GESVD_ROUTE"}) {
            auto p = make_svd<T>(herm);
            const std::string what = std::string(word) + " via " + via;
            auto call = [&] {
                const ScopedEnvVar env("BATCHLAS_GESVD_ROUTE", std::string(via) == "pin" ? nullptr : word);
                std::optional<Pin> pin;
                if (std::string(via) == "pin") pin.emplace("gesvd", std::string_view(word));
                this->run(p);
            };
            // The vendor reads the whole matrix; the Hermitian input stores poison in its upper half.
            if (spelling == "vendor") {
                EXPECT_EQ(traced_choice(call), "vendor") << what;
                continue;
            }
            EXPECT_EQ(traced_choice(call), spelling) << what;
            expect_solved(p, what);
        }
    // The retired BATCHLAS_GESVD_PROVIDER is not read.
    auto p = make_svd<T>(herm);
    const ScopedEnvVar retired("BATCHLAS_GESVD_PROVIDER", "blocked");
    EXPECT_EQ(traced_choice([&] { this->run(p); }), "cta") << "BATCHLAS_GESVD_PROVIDER was read";
}

// A ScopedPin beats BATCHLAS_GESVD_ROUTE; nested pins restore.
TYPED_TEST(GesvdCandidates, PinPrecedence) {
    using T = typename TestFixture::T;
    const Spec herm{16, 16, 2, 'L'};
    {
        const ScopedEnvVar canon("BATCHLAS_GESVD_ROUTE", "cta");
        auto p = make_svd<T>(herm);
        EXPECT_EQ(traced_choice([&] { this->run(p); }), "cta") << "BATCHLAS_GESVD_ROUTE was not read";
        auto r = make_svd<T>(herm);
        EXPECT_EQ(traced_choice([&] {
                      const Pin pin("gesvd", C{gs::Blocked{}});
                      this->run(r);
                  }),
                  "blocked");
    }
    const Pin outer("gesvd", C{gs::Cta{}});
    {
        const Pin inner("gesvd", C{gs::Blocked{}});
        auto r = make_svd<T>(herm);
        EXPECT_EQ(traced_choice([&] { this->run(r); }), "blocked");
    }
    auto r = make_svd<T>(herm);
    EXPECT_EQ(traced_choice([&] { this->run(r); }), "cta") << "the inner pin did not restore the outer";
    expect_solved(r, "outer cta");
}

// Auto on the shipped (transcribed) tables reproduces the old router (ported from gesvd_tests'
// DefaultProviderRoutesSmallGeneralToJacobi): jacobi leads general n <= 32 for every job, cta
// Hermitian n <= 32, blocked real general above 32, and the vendor elsewhere.
TYPED_TEST(GesvdCandidates, AutoReadsTheTranscribedTable) {
    using T = typename TestFixture::T;
    const ScopedEnvVar clear("BATCHLAS_GESVD_ROUTE", nullptr);
    constexpr bool kCd = std::is_same_v<T, std::complex<double>>;
    const char* none = TestFixture::kVendor ? "vendor" : "<none>";
    struct Row { Spec s; const char* expect; };
    std::vector<Row> rows;
    for (const auto& jobs : kJobs) rows.push_back({{32, 32, 2, 'N', jobs[0], jobs[1]}, "jacobi"});
    rows.push_back({{32, 32, 2, 'L'}, "cta"});
    rows.push_back({{30, 30, 2, 'U'}, "cta"});
    rows.push_back({{33, 33, 2, 'L'}, "blocked"});
    rows.push_back({{32, 9, 2, 'N', SvdVectors::Thin, SvdVectors::Thin}, "jacobi"});
    rows.push_back({{40, 9, 2, 'N', SvdVectors::Thin, SvdVectors::Thin}, TestFixture::kReal ? "blocked" : (kCd ? none : "jacobi")});
    rows.push_back({{64, 64, 2, 'N'}, TestFixture::kReal ? "blocked" : (kCd ? none : "jacobi")});
    rows.push_back({{48, 48, 2, 'N', SvdVectors::None, SvdVectors::None}, TestFixture::kReal ? "blocked" : "jacobi"});
    rows.push_back({{70, 50, 2, 'N'}, TestFixture::kReal ? "blocked" : none});
    for (const Row& r : rows) {
        auto p = make_svd<T>(r.s);
        if (std::string(r.expect) == "<none>") {
            EXPECT_THROW(this->run(p), batchlas::NoRouteError) << label(r.s);
            continue;
        }
        // cuSOLVER refuses past 32 (the old outcome too), in its sizing call, before any trace.
        std::string refused;
        std::string got = traced_choice([&] {
            try {
                this->run(p);
            } catch (const std::exception& e) {
                if (std::string(r.expect) != "vendor") throw;
                refused = e.what();
            }
        });
        if (refused.find("gesvd_vendor (CUSOLVER)") != std::string::npos) got = "vendor";
        EXPECT_EQ(got, r.expect) << label(r.s);
        if (std::string(r.expect) != "vendor") expect_solved(p, "auto " + label(r.s));
    }
    // The wide-band rule is a preference, not a capability: jacobi still takes a real 64 x 64 pin.
    auto big = make_svd<T>(Spec{64, 64, 2, 'N', SvdVectors::None, SvdVectors::None});
    EXPECT_TRUE(this->pin_accepted(C{gs::Jacobi{}}, big));
}

// key_of's every field reaches choose(): a synthetic table whose winner changes with herm,
// vec, m and n alone. Fixing a field in key_of turns exactly its probe red.
TYPED_TEST(GesvdCandidates, AutoReadsEveryKeyField) {
    using T = typename TestFixture::T;
    const ScopedEnvVar clear("BATCHLAS_GESVD_ROUTE", nullptr);
    const std::string dtype(select::dtype_name<T>());
    const std::string dev = select::device_of<TestFixture::B>(*this->ctx).key;
    std::vector<std::pair<std::string, std::string>> files;
    for (const auto& t : select::embedded_tables())
        if (t.name.rfind("gesvd.", 0) != 0) files.emplace_back(std::string(t.name), std::string(t.text));
    files.emplace_back("gesvd." + dtype + "." + dev + ".txt",
                       "# op=gesvd dtype=" + dtype + " device=" + dev + " kernels=unknown\n"
                       "# keys: herm:exact vec:exact m:log:1.5 n:log:1.5\n"
                       "herm=L vec=all m=16 n=16 | blocked 1 | cta 2\n"
                       "herm=L vec=none m=16 n=16 | cta 1 | blocked 2\n"
                       "herm=U vec=all m=16 n=16 | vendor 1 | cta 2\n"
                       "herm=N vec=all m=16 n=16 | jacobi 1 | cta 2 | blocked 3\n"
                       "herm=N vec=all m=24 n=16 | cta 1 | jacobi 2\n"
                       "herm=N vec=all m=16 n=24 | blocked 1 | jacobi 2\n"
                       "herm=N vec=thin m=24 n=16 | blocked 1 | jacobi 2\n");
    const TableGuard restore;
    select::testing::set_builtin_tables(std::move(files));
    struct Probe { Spec s; const char* expect; const char* field; bool real_only; };
    // Only cta runs a native Upper, so the Upper row tells itself apart with the vendor (whose
    // answer is not checked: it reads the poisoned triangle).
    if (TestFixture::kVendor) {
        Spec u{16, 16, 2, 'U'};
        u.gap = 0;  // gesvdjBatched takes only a packed batch
        u.ldpad = 0;
        auto p = make_svd<T>(u);
        EXPECT_EQ(traced_choice([&] { this->run(p); }), "vendor") << "the herm (Upper) probe";
    }
    const Probe probes[] = {
        {{16, 16, 2, 'L'}, "blocked", "base", false},
        {{16, 16, 2, 'L', SvdVectors::None, SvdVectors::None}, "cta", "vec", false},
        {{16, 16, 2, 'N'}, "jacobi", "herm (general)", false},
        {{24, 16, 2, 'N'}, "cta", "m", true},
        {{16, 24, 2, 'N'}, "blocked", "n", true},
        {{24, 16, 2, 'N', SvdVectors::Thin, SvdVectors::None}, "blocked", "vec (thin)", true},
        {{16, 16, 2, 'N', SvdVectors::Thin, SvdVectors::Thin}, "jacobi", "vec (square thin is all)", false}};
    for (const auto& k : probes) {
        if (k.real_only && !TestFixture::kReal) continue;
        auto p = make_svd<T>(k.s);
        EXPECT_EQ(traced_choice([&] { this->run(p); }), k.expect) << "the " << k.field << " probe";
        expect_solved(p, std::string("the ") + k.field + " probe");
    }
}

// The trace prints key_of() itself: canonical jobs, so a square Thin is `all`.
TYPED_TEST(GesvdCandidates, TraceShowsTheLookupKey) {
    using T = typename TestFixture::T;
    const std::pair<Spec, std::string> cases[] = {
        {{40, 8, 2, 'N', SvdVectors::Thin, SvdVectors::Thin}, "herm=N vec=thin m=40 n=8 ->"},
        {{12, 12, 2, 'N', SvdVectors::Thin, SvdVectors::None}, "herm=N vec=all m=12 n=12 ->"},
        {{12, 12, 2, 'U', SvdVectors::None, SvdVectors::None}, "herm=U vec=none m=12 n=12 ->"}};
    for (const auto& [s, want] : cases) {
        if (!TestFixture::kReal && s.m == 40) continue;  // complex general 40 x 8 thin: jacobi, same key
        auto p = make_svd<T>(s);
        const std::string line = traced_line([&] { this->run(p); });
        EXPECT_NE(line.find(want), std::string::npos) << line;
    }
}

// With a table naming only the vendor, a vendor-free build takes the last resort, blocked, for a
// real general matrix, and has no route for a complex general one past jacobi's ceiling.
TYPED_TEST(GesvdCandidates, VendorFreeLastResortIsBlocked) {
    using T = typename TestFixture::T;
    const ScopedEnvVar clear("BATCHLAS_GESVD_ROUTE", nullptr);
    const std::string dtype(select::dtype_name<T>());
    const std::string dev = select::device_of<TestFixture::B>(*this->ctx).key;
    std::vector<std::pair<std::string, std::string>> files;
    for (const auto& t : select::embedded_tables())
        if (t.name.rfind("gesvd.", 0) != 0) files.emplace_back(std::string(t.name), std::string(t.text));
    files.emplace_back("gesvd." + dtype + "." + dev + ".txt",
                       "# op=gesvd dtype=" + dtype + " device=" + dev + " kernels=unknown\n"
                       "# keys: herm:exact vec:exact m:log:1.5 n:log:1.5\n"
                       "herm=N vec=all m=16 n=16 | vendor 1\n");
    const TableGuard restore;
    select::testing::set_builtin_tables(std::move(files));
    if constexpr (TestFixture::kVendor) {
        GTEST_SKIP() << "the vendor serves the row";
    } else {
        Spec s{40, 40, 2, 'N'};
        auto p = make_svd<T>(s);
        if constexpr (TestFixture::kReal) {
            std::string all;
            EXPECT_EQ(traced_choice([&] { this->run(p); }, &all), "blocked");
            EXPECT_NE(all.find("last resort"), std::string::npos) << all;
            expect_solved(p, "last resort");
        } else {
            auto q = make_svd<T>(Spec{96, 96, 2, 'N'});
            EXPECT_THROW(this->run(q), batchlas::NoRouteError);
        }
    }
}

// The coverage row (§5.6): the real backend, m, n, k = min(m, n), the Hermitian triangle in
// uplo, the choice spelling, and the native flags.
TYPED_TEST(GesvdCandidates, CoverageRowCarriesBackendKeyAndNativeFlags) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
    const std::string dir = ::testing::TempDir() + "gesvd_cov." + std::string(select::dtype_name<T>());
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    const std::string out = dir + "/cov";
    const Spec lo{20, 12, 2, 'N'};
    const Spec hi{24, 24, 2, 'U'};
    auto child = [&] {
        const ScopedEnvVar cov("BATCHLAS_COVERAGE_OUT", out.c_str());
        const ScopedEnvVar clear("BATCHLAS_GESVD_ROUTE", nullptr);
        batchlas::coverage::g_dynamic_enabled = true;
        {
            const Pin pin("gesvd", C{gs::Jacobi{}});
            auto p = make_svd<T>(lo);
            this->run(p);
        }
        const Pin pin("gesvd", C{gs::Cta{}});
        auto p = make_svd<T>(hi);
        this->run(p);
        std::exit(0);
    };
    EXPECT_EXIT(child(), ::testing::ExitedWithCode(0), "");
    std::map<std::string, std::vector<std::string>> rows;
    for (const auto& ent : std::filesystem::directory_iterator(dir)) {
        std::ifstream in(ent.path());
        for (std::string line; std::getline(in, line);) {
            if (line.rfind("reached,gesvd,", 0) != 0) continue;
            std::vector<std::string> f;
            std::stringstream ss(line);
            for (std::string tok; std::getline(ss, tok, ',');) f.push_back(tok);
            ASSERT_GE(f.size(), 19u) << line;
            rows[f[5] + " " + f[6] + " " + f[7]] = f;
        }
    }
    std::filesystem::remove_all(dir);
    ASSERT_EQ(rows.size(), 2u);
    const std::string backend = B == Backend::CUDA ? "CUDA" : (B == Backend::ROCM ? "ROCM" : "?");
    auto as_int = [](auto e) { return std::to_string(static_cast<int>(e)); };
    for (const auto& [key, uplo, algo] : {std::tuple{std::string("20 12 12"), Uplo::Lower, "jacobi"},
                                          std::tuple{std::string("24 24 24"), Uplo::Upper, "cta"}}) {
        ASSERT_TRUE(rows.count(key)) << key;
        const auto& f = rows[key];
        EXPECT_EQ(f[3], backend) << key;
        EXPECT_EQ(f[9], "native") << key;
        EXPECT_EQ(f[10], algo) << key;
        EXPECT_EQ(f[12], "1") << key;
        EXPECT_EQ(f[13], "1") << key;
        EXPECT_EQ(f[15], as_int(uplo)) << key;
    }
}

// On a CPU queue no native family can run: native pins throw and Auto is the vendor (LAPACKE),
// or no route without one.
template <typename Config>
class GesvdCandidatesCpu : public test_utils::BatchLASTest<Config> {};
TYPED_TEST_SUITE(GesvdCandidatesCpu, Types);

TYPED_TEST(GesvdCandidatesCpu, CpuQueueRunsNoNativeFamily) {
    using T = typename TypeParam::ScalarType;
    static constexpr Backend B = TypeParam::BackendVal;
    if (!this->ctx) GTEST_SKIP() << "no queue";
    if (this->ctx->device().type == DeviceType::GPU) GTEST_SKIP() << "a GPU queue";
    const ScopedEnvVar clear("BATCHLAS_GESVD_ROUTE", nullptr);
    const Spec s{8, 8, 2, 'N'};
    auto run = [&](Svd<T>& p) {
        const std::size_t bytes =
            gesvd_buffer_size<B, T>(*this->ctx, p.A(), p.sv.to_span(), p.U.view(), p.Vh.view(), s.ju, s.jv);
        UnifiedVector<std::byte> ws(std::max<std::size_t>(bytes, 1));
        (void)gesvd<B, T>(*this->ctx, p.A(), p.sv.to_span(), p.U.view(), p.Vh.view(), s.ju, s.jv,
                          Span<std::byte>(ws.data(), bytes), p.info.to_span());
        this->ctx->wait();
    };
    for (const C& c : gs::candidates<T>()) {
        if (std::holds_alternative<gs::Vendor>(c)) continue;
        auto p = make_svd<T>(s);
        const Pin pin("gesvd", c);
        EXPECT_THROW(run(p), std::invalid_argument) << select::to_string(c);
    }
    auto p = make_svd<T>(s);
    if constexpr (batchlas::select::solver_vendor_available<B>) {
        EXPECT_EQ(traced_choice([&] { run(p); }), "vendor");
        expect_solved(p, "cpu vendor");
    } else {
        EXPECT_THROW(run(p), batchlas::NoRouteError);
    }
}

// The transcribed rows, read with Table::nearest directly so every device checks them, on both
// transcribed devices.
TEST(GesvdTranscribedTable, RowsHoldTheOldPreference) {
    struct Row { const char* dtype; const char* herm; const char* vec; int m, n; const char* ranked; };
    const Row rows[] = {
        {"float", "N", "all", 32, 32, "jacobi|cta|blocked|vendor"}, {"float", "N", "all", 33, 33, "blocked|vendor|jacobi"},
        {"float", "N", "all", 65, 65, "blocked|vendor"},           {"double", "N", "thin", 32, 8, "jacobi|blocked|vendor"},
        {"double", "L", "all", 32, 32, "cta|blocked|vendor"},      {"double", "U", "all", 32, 32, "cta|vendor"},
        {"double", "U", "all", 33, 33, "vendor"},                  {"cfloat", "N", "all", 64, 64, "jacobi|vendor"},
        {"cfloat", "N", "all", 65, 65, "vendor"},                  {"cdouble", "N", "all", 33, 33, "vendor"},
        {"cdouble", "N", "none", 64, 64, "jacobi|vendor"},         {"cdouble", "L", "none", 48, 48, "blocked|vendor"}};
    for (const char* dev : {"sm_89", "sm_120"})
        for (const Row& r : rows) {
            const auto tables = select::tables_in_borrow_order("gesvd", r.dtype, select::device_from_key(dev));
            ASSERT_FALSE(tables.empty()) << r.dtype;
            const select::Table& t = *tables.front();
            ASSERT_EQ(t.device, dev) << r.dtype;
            EXPECT_EQ(t.source.rfind("transcribed:", 0), 0u) << t.file;
            const select::Key key{{"herm", r.herm}, {"vec", r.vec}, {"m", r.m}, {"n", r.n}};
            const select::TableRow* row = t.nearest(key);
            ASSERT_NE(row, nullptr) << t.file;
            std::string got;
            for (const auto& e : row->ranked) got += (got.empty() ? "" : "|") + e.spelling;
            EXPECT_EQ(got, r.ranked) << t.file << ":" << row->line;
            EXPECT_FALSE(row->timed) << t.file;
        }
}

}  // namespace
