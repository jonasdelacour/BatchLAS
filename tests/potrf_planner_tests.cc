// PlanTree prototype (src/plan/): the new two-phase potrf selection, checked against the
// shipping resolver on the real device, executed through the real kernels, and sized.
// experiments/kernel_selection/planner-tree/README.md records the results.

#include <gtest/gtest.h>

#include <batchlas/blas/functions/potrf.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/env.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>
#include <batchlas/util/sycl-vector.hh>

#include "../src/backends/potrf_route.hh"
#include "../src/plan/potrf_exec.hh"

#include <atomic>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <fstream>
#include <map>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

using namespace batchlas;
namespace pl = batchlas::plan;
using dispatch::Algorithm;
using dispatch::Origin;
using dispatch::Route;

namespace {

constexpr Backend kB = Backend::CUDA;
// A -DBATCHLAS_ENABLE_VENDOR_BLAS=OFF build: unservable shapes plan to `vendor` and throw at run.
constexpr bool kVendor = dispatch::solver_vendor_available<kB>;

std::shared_ptr<Queue> make_queue() {
    static std::shared_ptr<Queue> q = [] {
        try {
            return std::make_shared<Queue>(Device("gpu"), kB, true);
        } catch (...) {
            return std::shared_ptr<Queue>{};
        }
    }();
    return q;
}

struct Cell { std::string dtype; int n; int batch; };

// Distinct (dtype, n, batch) cells of the two sm_120 sweeps (every row is Lower).
std::vector<Cell> sweep_cells() {
    std::set<std::tuple<std::string, int, int>> seen;
    for (const char* f : {"/sm120_potrf_sweep.jsonl", "/sm120_potrf_sweep_edges.jsonl"}) {
        std::ifstream in(std::string(BATCHLAS_ROUTING_RESULTS_DIR) + f);
        std::string line;
        while (std::getline(in, line)) {
            auto field = [&](const std::string& k) {
                const auto p = line.find("\"" + k + "\": ");
                return p == std::string::npos ? std::string() : line.substr(p + k.size() + 4);
            };
            std::string dt = field("dtype");
            dt = dt.substr(1, dt.find('"', 1) - 1);
            seen.emplace(dt, std::stoi(field("n")), std::stoi(field("batch")));
        }
    }
    std::vector<Cell> out;
    for (const auto& [d, n, b] : seen) out.push_back({d, n, b});
    return out;
}

template <class F>
void for_dtype(const std::string& dt, F&& f) {
    if (dt == "float") f.template operator()<float>();
    else if (dt == "double") f.template operator()<double>();
    else if (dt == "cfloat") f.template operator()<std::complex<float>>();
    else f.template operator()<std::complex<double>>();
}

std::string rn(Route r) { return pl::route_name(r); }

struct Tally {
    long cells = 0, mismatches = 0, tu_mismatches = 0, not_honoured = 0;
    std::map<std::string, long> by_tier;
    std::vector<std::string> first;
};

template <typename T>
MatrixView<T, MatrixFormat::Dense> shape_view(int n, int batch) {
    // Routing reads only the extents; the data is never touched.
    return MatrixView<T, MatrixFormat::Dense>(nullptr, n, n, n, n * n, batch);
}

// Tier NAMES are compared, never Routes: two tiers may share a Route (Cta and CtaWg), and the
// shipping resolver's route means that route's OWNER tier.
template <typename T>
void compare_cell(Queue& q, const Cell& c, Uplo uplo, bool vendor_free, const std::string& mode, Tally& t) {
    const auto A = shape_view<T>(c.n, c.batch);
    const Route old_r = backend::potrf_route<kB, T>(q, A, uplo, vendor_free ? false : kVendor);
    const std::string_view want = pl::potrf::Tiers<T>::owner_name(old_r);
    const pl::Node nd = pl::potrf_plan_node<kB, T>(q, A, uplo, vendor_free);
    // The same planner instantiated in THIS TU. Its symbols carry this TU's ABI tag, so a
    // -DBATCHLAS_PLAN_MUTANT build changes this copy only: an independent second check.
    const pl::Node tu = pl::potrf_build<kB, T>(q, A, uplo, pl::request_for_potrf(), vendor_free);
    auto note = [&](const char* tag, std::string_view got) {
        if (t.first.size() < 20) {
            t.first.push_back(std::string(tag) + mode + " " + c.dtype + " n=" + std::to_string(c.n) + " b=" +
                              std::to_string(c.batch) + (uplo == Uplo::Lower ? " L" : " U") + " old=" + rn(old_r) +
                              "(" + std::string(want) + ") new=" + std::string(got));
        }
    };
    ++t.cells;
    if (tu.tier != want) { ++t.tu_mismatches; note("[tu] ", tu.tier); }
    if (!nd.decision.honoured) ++t.not_honoured;
    ++t.by_tier[std::string(nd.tier)];
    if (nd.tier != want || !(nd.route == old_r)) { ++t.mismatches; note("", nd.tier); }
}

void run_grid(Queue& q, const std::vector<Cell>& cells, bool vendor_free, const std::string& mode, Tally& t) {
    for (const Cell& c : cells) {
        for (Uplo u : {Uplo::Lower, Uplo::Upper}) {
            for_dtype(c.dtype, [&]<typename T>() { compare_cell<T>(q, c, u, vendor_free, mode, t); });
        }
    }
}

void report(const std::string& what, const Tally& t) {
    std::printf("[equiv] %-40s cells=%ld mismatches=%ld tu-mismatches=%ld not-honoured=%ld |", what.c_str(), t.cells,
                t.mismatches, t.tu_mismatches, t.not_honoured);
    for (const auto& [r, n] : t.by_tier) std::printf(" %s:%ld", r.c_str(), n);
    std::printf("\n");
    for (const auto& s : t.first) std::printf("    MISMATCH %s\n", s.c_str());
}

class PlannerTest : public ::testing::Test {
protected:
    std::shared_ptr<Queue> q;
    void SetUp() override {
        q = make_queue();
        if (!q || q->device().type != DeviceType::GPU) GTEST_SKIP() << "needs a CUDA GPU";
        pl::cache_clear();
    }
    void TearDown() override {
        pl::set_potrf_planner(-1);
        pl::log_enable(false);
        sycl_potrf::potrf_launch_record_enable(false);
    }
};

// ---- 1. equivalence against the shipping resolver ------------------------------------------

TEST_F(PlannerTest, EquivalenceOnSweepCells) {
    const auto cells = sweep_cells();
    ASSERT_GE(cells.size(), 500u) << "git lfs pull: the sweep files are pointers";
    std::printf("[equiv] %zu distinct (dtype, n, batch) cells x {Lower, Upper}, vendor=%d\n", cells.size(),
                int(kVendor));
    long lib_mm = 0, tu_mm = 0;
    for (const char* prof : {static_cast<const char*>(nullptr), "sm_89"}) {
        ScopedEnvVar p("BATCHLAS_ROUTING_PROFILE", prof);
        const std::string pn = prof ? prof : "device(sm_120)";
        for (bool vf : {false, true}) {
            Tally t;
            run_grid(*q, cells, vf, pn + (vf ? " vendor-free" : " auto"), t);
            report(pn + (vf ? " vendor-free walk" : " auto"), t);
            lib_mm += t.mismatches;
            tu_mm += t.tu_mismatches;
            EXPECT_EQ(t.by_tier.count("native:cta_wg"), 0u) << "Auto picked the unmeasured extension tier";
        }
        for (const char* pin : {"native", "native:tiny", "native:cta", "native:lpanel", "native:blocked", "vendor"}) {
            ScopedEnvVar e("BATCHLAS_POTRF_ROUTE", pin);
            Tally t;
            run_grid(*q, cells, false, pn + " pin=" + pin, t);
            report(pn + " pin=" + pin, t);
            lib_mm += t.mismatches;
            tu_mm += t.tu_mismatches;
            EXPECT_EQ(t.by_tier.count("native:cta_wg"), 0u) << "pin " << pin << " ran the extension tier";
        }
    }
    EXPECT_EQ(lib_mm, 0) << "the LIBRARY's planner differs from the shipping resolver";
    EXPECT_EQ(tu_mm, 0) << "this TU's planner differs from the shipping resolver";
}

// TierList order is the FirstLegal tie order among selectable tiers only. A tier that shares
// a Route (CtaWg) or is measured nowhere can move anywhere without changing any decision.
template <typename T, class Tiers>
std::string_view plan_with(Queue& q, const Cell& c, Uplo u, bool vf) {
    const auto A = shape_view<T>(c.n, c.batch);
    const pl::Request r = pl::request_for_potrf();
    const pl::Facts f = pl::facts_for<kB>(q, vf ? false : kVendor);
    const auto s = pl::potrf_shape<kB, T>(A, u, f, r);
    return pl::plan_op<Tiers>(pl::potrf::context<T>(s, f), s, f, r).tier;
}

TEST_F(PlannerTest, TierOrderOfASharedOrUnmeasuredTierIsNotLive) {
    namespace pt = pl::potrf;
    long diff = 0, picked_ext = 0, n = 0;
    std::vector<std::string> first;
    const auto cells = sweep_cells();
    for (const char* pin : {static_cast<const char*>(nullptr), "native", "native:cta"}) {
        ScopedEnvVar e("BATCHLAS_POTRF_ROUTE", pin);
        for (bool vf : {false, true}) {
            for (const Cell& c : cells) {
                for (Uplo u : {Uplo::Lower, Uplo::Upper}) {
                    for_dtype(c.dtype, [&]<typename T>() {
                        using First = pl::TierList<pt::CtaWg<T>, pt::Tiny<T>, pt::Cta<T>, pt::LPanel<T>, pt::Blocked<T>, pt::Vendor<T>>;
                        using Last = pl::TierList<pt::Tiny<T>, pt::Cta<T>, pt::LPanel<T>, pt::Blocked<T>, pt::Vendor<T>, pt::CtaWg<T>>;
                        const auto ref = plan_with<T, pt::Tiers<T>>(*q, c, u, vf);
                        for (std::string_view got : {plan_with<T, First>(*q, c, u, vf), plan_with<T, Last>(*q, c, u, vf)}) {
                            ++n;
                            if (got == "native:cta_wg") ++picked_ext;
                            if (got != ref) {
                                ++diff;
                                if (first.size() < 10) {
                                    first.push_back(c.dtype + " n=" + std::to_string(c.n) + " b=" + std::to_string(c.batch) +
                                                    " pin=" + (pin ? pin : "auto") + (vf ? " vf" : "") + " ref=" +
                                                    std::string(ref) + " got=" + std::string(got));
                                }
                            }
                        }
                    });
                }
            }
        }
    }
    std::printf("[order] %ld permuted plans: %ld differ, %ld picked native:cta_wg\n", n, diff, picked_ext);
    for (const auto& s : first) std::printf("    %s\n", s.c_str());
    EXPECT_EQ(diff, 0);
    EXPECT_EQ(picked_ext, 0);
}

// The cost model and the hand windows disagree on some cells; the planner makes that a
// reported table instead of a silent divergence (BATCHLAS_STRATEGY overrides the policy).
TEST_F(PlannerTest, StrategyDisagreementIsReported) {
    const auto cells = sweep_cells();
    long diff = 0, n_cells = 0;
    std::map<std::string, long> kinds;
    std::vector<std::string> rows;
    for (const Cell& c : cells) {
        for_dtype(c.dtype, [&]<typename T>() {
            const auto A = shape_view<T>(c.n, c.batch);
            std::string m, w;
            { ScopedEnvVar s("BATCHLAS_STRATEGY", "model"); m = std::string(pl::potrf_plan_node<kB, T>(*q, A, Uplo::Lower, false).tier); }
            { ScopedEnvVar s("BATCHLAS_STRATEGY", "windows"); w = std::string(pl::potrf_plan_node<kB, T>(*q, A, Uplo::Lower, false).tier); }
            ++n_cells;
            if (m != w) {
                ++diff;
                ++kinds[c.dtype + " model=" + m + " windows=" + w];
                rows.push_back(c.dtype + " n=" + std::to_string(c.n) + " b=" + std::to_string(c.batch) + " model=" + m + " windows=" + w);
            }
        });
    }
    std::printf("[strategy] Lower cells=%ld model!=windows on %ld\n", n_cells, diff);
    for (const auto& [k, n] : kinds) std::printf("    %-60s %ld\n", k.c_str(), n);
    if (const char* p = std::getenv("BATCHLAS_PLANNER_DUMP")) {
        std::ofstream o(p);
        for (const auto& r : rows) o << r << "\n";
    }
    if (kVendor) {   // with no vendor, the windows' vs-vendor answer is moot
        const auto cd20 = shape_view<std::complex<double>>(20, 8192);
        ScopedEnvVar s("BATCHLAS_STRATEGY", "windows");
        EXPECT_EQ((pl::potrf_plan_node<kB, std::complex<double>>(*q, cd20, Uplo::Lower, false).tier), "vendor");
    }
}

// ---- 2. sizing -----------------------------------------------------------------------------

template <typename T>
std::pair<std::size_t, std::size_t> sizes(Queue& q, const MatrixView<T, MatrixFormat::Dense>& A, Uplo u) {
    pl::set_potrf_planner(0);
    const std::size_t old_b = potrf_buffer_size<kB, T>(q, A, u);
    pl::set_potrf_planner(1);
    const std::size_t new_b = potrf_buffer_size<kB, T>(q, A, u);
    pl::set_potrf_planner(-1);
    return {old_b, new_b};
}

// Independent of the planner: each legal native tier's own size from the kernels' entry points
// (Blocked at W clamped to n - nb, re-derived here from blocked_params, not read off the node).
template <typename T>
std::size_t independent_cover(Queue& q, const MatrixView<T, MatrixFormat::Dense>& A, const pl::Node& nd) {
    std::size_t need = 0;
    for (const auto& c : nd.decision.candidates) {
        if (!c.legal.ok || !dispatch::is_native(c.route)) continue;
        if (c.name == "native:tiny") need = std::max(need, sycl_potrf::potrf_tiny_buffer_size<T>(q, A));
        if (c.name == "native:cta" || c.name == "native:cta_wg") need = std::max(need, sycl_potrf::potrf_cta_buffer_size<T>(q, A));
        if (c.name == "native:lpanel") need = std::max(need, sycl_potrf::potrf_lpanel_buffer_size<T>(q, A));
        if (c.name == "native:blocked") {
            const int n = static_cast<int>(A.rows());
            const auto hw = pl::hw_facts(q.device());
            auto bp = potrf_plan::blocked_params<T>(n, hw.launch.local_mem_bytes, sycl_trsm::trsm_cta_max_n<T>());
            bp.W = std::max(1, std::min(bp.W, n - bp.nb));
            need = std::max(need, sycl_potrf::potrf_blocked_buffer_size_planned<T>(q, A, bp));
        }
    }
    return need;
}

TEST_F(PlannerTest, SizingCoversEveryLegalNativeTree) {
    struct S { const char* name; std::size_t o, n, chosen, indep; std::string tier; };
    std::vector<S> rows;
    auto add = [&]<typename T>(const char* name, int n, int b) {
        Matrix<T, MatrixFormat::Dense> M(n, n, b);
        const auto [o, nw] = sizes<T>(*q, M.view(), Uplo::Lower);
        const auto nd = pl::potrf_plan_node<kB, T>(*q, M.view(), Uplo::Lower, false);
        rows.push_back({name, o, nw, nd.workspace(), independent_cover<T>(*q, M.view(), nd), std::string(nd.tier)});
    };
    add.operator()<float>("float n=36 b=8192", 36, 8192);
    add.operator()<float>("float n=128 b=8192", 128, 8192);
    add.operator()<float>("float n=640 b=512", 640, 512);
    add.operator()<std::complex<double>>("cdouble n=20 b=8192", 20, 8192);
    add.operator()<double>("double n=1024 b=128", 1024, 128);
    {
        ScopedEnvVar e("BATCHLAS_POTRF_ROUTE", "native:blocked");
        add.operator()<float>("float n=36 b=8192 pin native:blocked", 36, 8192);
    }
    for (const auto& r : rows) {
        std::printf("[size] %-40s old=%zu new=%zu chosen-tree=%zu independent-cover=%zu (%s)\n", r.name, r.o, r.n,
                    r.chosen, r.indep, r.tier.c_str());
        EXPECT_LE(r.n, r.o) << r.name;
        EXPECT_GE(r.n, r.chosen) << r.name;
        EXPECT_GE(r.n, r.indep) << r.name << ": a legal native tier would under-allocate under a changed pin";
    }
    EXPECT_LT(rows[0].n, rows[0].o / 100) << "the 512 MiB unclamped-W over-estimate is back";
    EXPECT_EQ(rows[0].o, 537100288u);
}

// The intended difference, cell by cell: the max over clamped real trees, never more than main.
TEST_F(PlannerTest, SizingAcrossSweepCells) {
    long same = 0, smaller = 0, larger = 0, both_throw = 0, one_throws = 0;
    std::ofstream dump(std::getenv("BATCHLAS_PLANNER_SIZE_DUMP") ? std::getenv("BATCHLAS_PLANNER_SIZE_DUMP") : "/dev/null");
    for (const Cell& c : sweep_cells()) {
        for (Uplo u : {Uplo::Lower, Uplo::Upper}) {
            for_dtype(c.dtype, [&]<typename T>() {
                const auto A = shape_view<T>(c.n, c.batch);
                auto query = [&](int mode) -> std::optional<std::size_t> {
                    pl::set_potrf_planner(mode);
                    try { return potrf_buffer_size<kB, T>(*q, A, u); } catch (const std::exception&) { return std::nullopt; }
                };
                const auto o = query(0), nw = query(1);
                pl::set_potrf_planner(-1);
                if (!o || !nw) { (!o && !nw ? both_throw : one_throws)++; return; }   // unservable vendor-free
                const auto nd = pl::potrf_plan_node<kB, T>(*q, A, u, false);
                (*o == *nw ? same : (*nw < *o ? smaller : larger))++;
                EXPECT_GE(*nw, nd.workspace());
                dump << c.dtype << " n=" << c.n << " b=" << c.batch << (u == Uplo::Lower ? " L " : " U ") << nd.tier
                     << " old=" << *o << " new=" << *nw << " chosen=" << nd.workspace() << "\n";
            });
        }
    }
    std::printf("[size-sweep] cells=%ld same=%ld smaller=%ld larger=%ld unservable(both throw)=%ld one-throws=%ld\n",
                same + smaller + larger + both_throw + one_throws, same, smaller, larger, both_throw, one_throws);
    EXPECT_EQ(larger, 0);
    EXPECT_EQ(one_throws, 0);
}

// ---- 3. execution through the real kernels -------------------------------------------------

template <typename T> using RealOf = typename potrf_plan::RealOf<T>::type;
template <typename T> T conj_(T v) { if constexpr (potrf_plan::kIsComplex<T>) return std::conj(v); else return v; }
template <typename T> RealOf<T> abs2(T v) { if constexpr (potrf_plan::kIsComplex<T>) return std::norm(v); else return v * v; }

template <typename T>
std::vector<T> make_hpd(int n, unsigned seed) {
    std::mt19937 g(seed);
    std::uniform_real_distribution<RealOf<T>> d(-1, 1);
    std::vector<T> M(static_cast<std::size_t>(n) * n), A(M.size(), T{});
    for (auto& v : M) { if constexpr (potrf_plan::kIsComplex<T>) v = T(d(g), d(g)); else v = d(g); }
    for (int i = 0; i < n; ++i)
        for (int j = 0; j <= i; ++j) {
            T acc{};
            for (int k = 0; k < n; ++k) acc += M[i + std::size_t(k) * n] * conj_(M[j + std::size_t(k) * n]);
            acc /= T(RealOf<T>(n));
            if (i == j) acc = T(std::real(acc) + RealOf<T>(2));
            A[i + std::size_t(j) * n] = acc;
            A[j + std::size_t(i) * n] = conj_(acc);
        }
    return A;
}

template <typename T>
double residual(const std::vector<T>& A, const T* F, int n, int ld, Uplo u) {
    auto L = [&](int i, int j) -> T {   // A = L L^H; Upper stores U = L^H
        if (j > i) return T{};
        return u == Uplo::Lower ? F[i + std::size_t(j) * ld] : conj_(F[j + std::size_t(i) * ld]);
    };
    double num = 0, den = 0;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            T acc{};
            for (int k = 0; k <= std::min(i, j); ++k) acc += L(i, k) * conj_(L(j, k));
            num += abs2(acc - A[i + std::size_t(j) * n]);
            den += abs2(A[i + std::size_t(j) * n]);
        }
    return std::sqrt(num / den);
}

struct RunOut { std::string tier; double res = -1; bool identical_items = true; std::vector<pl::LogRow> log; double max_diff_vs_old = -1; bool poison_intact = true; };

// Non-natural ld and stride, one HPD matrix in every item, the other triangle poisoned with a
// large finite value. Runs the planned path, then the shipping path on the same input.
template <typename T>
RunOut run_one(Queue& q, int n, int batch, Uplo u, const char* name_pin = nullptr, const char* route_pin = nullptr) {
    ScopedEnvVar e1("BATCHLAS_ROUTE", name_pin ? (std::string("potrf=") + name_pin).c_str() : nullptr);
    ScopedEnvVar e2("BATCHLAS_POTRF_ROUTE", route_pin);
    const int ld = n + 3, stride = ld * n + 5;
    const auto ref = make_hpd<T>(n, 1234u + n);
    const T poison = T(RealOf<T>(1e6));
    auto load = [&](Matrix<T, MatrixFormat::Dense>& M) {
        T* p = M.view().data_ptr();
        for (int b = 0; b < batch; ++b)
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i) {
                    const bool tri = (u == Uplo::Lower) ? i >= j : i <= j;
                    p[std::size_t(b) * stride + i + std::size_t(j) * ld] = tri ? ref[i + std::size_t(j) * n] : poison;
                }
    };
    RunOut out;
    Matrix<T, MatrixFormat::Dense> A(n, n, batch, ld, stride), B(n, n, batch, ld, stride);
    load(A);
    load(B);
    UnifiedVector<int32_t> info(batch, -7);
    pl::set_potrf_planner(1);
    pl::log_enable(true);
    pl::log_take();
    {
        UnifiedVector<std::byte> ws(potrf_buffer_size<kB, T>(q, A.view(), u));
        (void)potrf<kB, T>(q, A.view(), u, ws.to_span(), info.to_span());
        q.wait();
    }
    out.log = pl::log_take();
    pl::log_enable(false);
    out.tier = out.log.empty() ? "?" : out.log.front().tier;
    for (int b = 0; b < batch; ++b) EXPECT_EQ(info[b], 0) << "item " << b;
    const T* p = A.view().data_ptr();
    for (int b = 1; b < batch && out.identical_items; ++b)
        for (int j = 0; j < n && out.identical_items; ++j)
            for (int i = 0; i < n; ++i)
                if (!(p[std::size_t(b) * stride + i + std::size_t(j) * ld] == p[i + std::size_t(j) * ld])) { out.identical_items = false; break; }
    out.res = std::max({residual<T>(ref, p, n, ld, u), residual<T>(ref, p + std::size_t(batch / 2) * stride, n, ld, u),
                        residual<T>(ref, p + std::size_t(batch - 1) * stride, n, ld, u)});
    pl::set_potrf_planner(0);
    {
        UnifiedVector<std::byte> ws(potrf_buffer_size<kB, T>(q, B.view(), u));
        (void)potrf<kB, T>(q, B.view(), u, ws.to_span(), info.to_span());
        q.wait();
    }
    pl::set_potrf_planner(-1);
    double md = 0;
    const T* r = B.view().data_ptr();
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) {
            const bool tri = (u == Uplo::Lower) ? i >= j : i <= j;
            const std::size_t at = i + std::size_t(j) * ld;
            if (tri) md = std::max(md, double(std::abs(r[at] - p[at])));
            else if (!(p[at] == poison)) out.poison_intact = false;
        }
    out.max_diff_vs_old = md;
    return out;
}

// expect_tier nullptr: the shipping resolver's answer (its route's owner tier) is the reference.
template <typename T>
void exec_case(Queue& q, const char* label, int n, int batch, Uplo u, const char* expect_tier,
               const char* name_pin = nullptr, const char* route_pin = nullptr) {
    std::string want = expect_tier ? expect_tier : "";
    if (!expect_tier) {
        ScopedEnvVar e2("BATCHLAS_POTRF_ROUTE", route_pin);
        want = std::string(pl::potrf::Tiers<T>::owner_name(
            backend::potrf_route<kB, T>(q, shape_view<T>(n, batch), u, kVendor)));
    }
    if (!kVendor && want == "vendor") {
        std::printf("[exec] %-34s skipped: unservable in a vendor-free build\n", label);
        return;
    }
    const RunOut o = run_one<T>(q, n, batch, u, name_pin, route_pin);
    std::printf("[exec] %-34s tier=%-15s residual=%.3e identical_items=%d poison_intact=%d max|new-old|=%.3e log_rows=%zu\n",
                label, o.tier.c_str(), o.res, int(o.identical_items), int(o.poison_intact), o.max_diff_vs_old, o.log.size());
    for (std::size_t i = 1; i < o.log.size() && i < 4; ++i) {
        std::printf("         child %s -> %s backend=%d %lldx%lldx%lld\n", o.log[i].path.c_str(), o.log[i].route.c_str(),
                    int(o.log[i].backend), (long long)o.log[i].m, (long long)o.log[i].n, (long long)o.log[i].k);
    }
    EXPECT_EQ(o.tier, want) << label;
    EXPECT_LT(o.res, 50.0 * n * std::numeric_limits<RealOf<T>>::epsilon()) << label;
    EXPECT_TRUE(o.identical_items) << label;
    if (want != "vendor") EXPECT_TRUE(o.poison_intact) << label;
}

// Hard-coded expectations are the vendor build's measured routing; a vendor-free build checks
// against the shipping resolver instead.
const char* ex(const char* tier) { return kVendor ? tier : nullptr; }

TEST_F(PlannerTest, ExecutesEveryTier) {
    exec_case<float>(*q, "float L n=16 b=4096 (auto)", 16, 4096, Uplo::Lower, ex("native:tiny"));
    exec_case<float>(*q, "float L n=36 b=8192 (auto)", 36, 8192, Uplo::Lower, ex("native:cta"));
    exec_case<float>(*q, "float L n=128 b=8192 (auto)", 128, 8192, Uplo::Lower, ex("native:lpanel"));
    exec_case<std::complex<double>>(*q, "cdouble L n=20 b=8192 (auto)", 20, 8192, Uplo::Lower, ex("native:lpanel"));
    exec_case<float>(*q, "float L n=640 b=512 (auto)", 640, 512, Uplo::Lower, ex("native:blocked"));
    exec_case<double>(*q, "double L n=300 b=256 (pin blocked)", 300, 256, Uplo::Lower, "native:blocked", nullptr, "native:blocked");
    exec_case<float>(*q, "float U n=128 b=1024 (auto)", 128, 1024, Uplo::Upper, ex("vendor"));
    exec_case<std::complex<float>>(*q, "cfloat U n=24 b=1024 (pin cta)", 24, 1024, Uplo::Upper, "native:cta", nullptr, "native:cta");
}

// ---- 3b. plan == launch: the node's geometry against what the kernel ACTUALLY launched -----

std::string kind_of(std::string_view tier) {
    if (tier == "native:tiny") return "tiny";
    if (tier == "native:lpanel") return "lpanel";
    return "cta";   // native:cta, native:cta_wg, and Blocked's leaf
}

// Checks one leaf launch against one plan node (a single-kernel tier, or Blocked's CTA leaf).
void expect_launch_matches(const pl::Node& nd, const sycl_potrf::PotrfLaunchRecord& r, const std::string& label) {
    EXPECT_EQ(std::string(r.kernel), kind_of(nd.tier)) << label;
    EXPECT_EQ(r.n, nd.n) << label;
    EXPECT_EQ(r.batch, nd.batch) << label;
    EXPECT_EQ(r.wg_size, nd.launch.wg_size) << label << ": work-group size";
    EXPECT_EQ(r.num_wg, nd.launch.groups) << label << ": grid";
    const std::string k(nd.kernel);
    if (r.kernel == std::string("tiny")) {
        EXPECT_EQ(r.tparam, nd.param("N")) << label << ": register bucket";
        EXPECT_EQ(r.G, nd.param("per_wg")) << label;
    } else if (r.kernel == std::string("lpanel")) {
        EXPECT_EQ(r.tparam, nd.param("NB")) << label << ": panel width";
        EXPECT_EQ(r.L, nd.param("L")) << label;
        EXPECT_EQ(r.G, nd.param("G")) << label;
    } else {
        EXPECT_EQ(r.subgroup_scope, k.find("SubGroup") != std::string::npos) << label << ": barrier scope";
        EXPECT_EQ(r.L, nd.param("L")) << label;
        EXPECT_EQ(r.G, nd.param("G")) << label;
    }
}

// Runs the call twice: through the public facade (the LIBRARY's Exec) and through this TU's
// potrf_planned (this TU's Exec, which mutants.sh 7/8 break), recording every launch.
template <typename T>
void plan_equals_launch(Queue& q, const char* label, int n, int batch, const char* name_pin = nullptr,
                        const char* lpanel_nb = nullptr) {
    ScopedEnvVar e1("BATCHLAS_ROUTE", name_pin ? (std::string("potrf=") + name_pin).c_str() : nullptr);
    ScopedEnvVar e2("BATCHLAS_POTRF_LPANEL_NB", lpanel_nb);
    Matrix<T, MatrixFormat::Dense> A(n, n, batch, n + 3, (n + 3) * n + 5);
    const auto ref = make_hpd<T>(n, 77u + n);
    const pl::Node nd = pl::potrf_plan_node<kB, T>(q, A.view(), Uplo::Lower, false);
    if (nd.tier == "vendor") { std::printf("[launch] %-30s vendor: no native launch\n", label); return; }
    for (int path = 0; path < 2; ++path) {
        for (int b = 0; b < batch; ++b)
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i) A.view().data_ptr()[std::size_t(b) * A.view().stride() + i + std::size_t(j) * A.view().ld()] = ref[i + std::size_t(j) * n];
        UnifiedVector<int32_t> info(batch, -7);
        UnifiedVector<std::byte> ws(std::max<std::size_t>(1, nd.cover_ws));
        sycl_potrf::potrf_launch_record_take();
        sycl_potrf::potrf_launch_record_enable(true);
        if (path == 0) {
            pl::set_potrf_planner(1);
            (void)potrf<kB, T>(q, A.view(), Uplo::Lower, ws.to_span(), info.to_span());
            pl::set_potrf_planner(-1);
        } else {
            (void)pl::potrf_planned<kB, T>(q, A.view(), Uplo::Lower, ws.to_span(), info.to_span());
        }
        q.wait();
        sycl_potrf::potrf_launch_record_enable(false);
        const auto recs = sycl_potrf::potrf_launch_record_take();
        const std::string lab = std::string(label) + (path == 0 ? " [library exec]" : " [tu exec]");
        for (int b = 0; b < batch; ++b) ASSERT_EQ(info[b], 0) << lab << " item " << b;
        ASSERT_FALSE(recs.empty()) << lab;
        std::printf("[launch] %-44s %s: %zu launch(es), first %s<%d> n=%d wg=%d groups=%lld L=%d G=%d sg=%d\n", lab.c_str(),
                    std::string(nd.tier).c_str(), recs.size(), recs[0].kernel, recs[0].tparam, recs[0].n, recs[0].wg_size,
                    (long long)recs[0].num_wg, recs[0].L, recs[0].G, int(recs[0].subgroup_scope));
        if (nd.tier != "native:blocked") {
            EXPECT_EQ(recs.size(), 1u) << lab;
            expect_launch_matches(nd, recs[0], lab);
            continue;
        }
        // Blocked: one CTA leaf per panel; the full-width leaf is the node's planned child.
        const int nb = static_cast<int>(nd.param("nb"));
        EXPECT_EQ(static_cast<long>(recs.size()), (n + nb - 1) / nb) << lab << ": leaf launches";
        const pl::Node* leaf = nullptr;
        for (const auto& c : nd.children) if (c.op == dispatch::Op::potrf) leaf = &c;
        ASSERT_NE(leaf, nullptr) << lab;
        expect_launch_matches(*leaf, recs[0], lab + " leaf");
    }
}

TEST_F(PlannerTest, PlanEqualsLaunch) {
    plan_equals_launch<float>(*q, "float n=16 b=4096", 16, 4096);
    plan_equals_launch<float>(*q, "float n=36 b=8192", 36, 8192);
    plan_equals_launch<float>(*q, "float n=128 b=8192", 128, 8192);
    plan_equals_launch<float>(*q, "float n=128 b=8192 LPANEL_NB=16", 128, 8192, nullptr, "16");
    plan_equals_launch<std::complex<double>>(*q, "cdouble n=20 b=8192", 20, 8192);
    plan_equals_launch<float>(*q, "float n=20 b=2048 pin cta_wg", 20, 2048, "native:cta_wg");
    plan_equals_launch<float>(*q, "float n=24 b=4096 pin cta", 24, 4096, "native:cta");
    plan_equals_launch<float>(*q, "float n=60 b=1024 pin cta", 60, 1024, "native:cta");
    plan_equals_launch<float>(*q, "float n=640 b=512", 640, 512);
    // Anti-vacuity: the two LPanel plans must differ in the axis the check reads.
    ScopedEnvVar e("BATCHLAS_POTRF_LPANEL_NB", "16");
    const auto nb16 = pl::potrf_plan_node<kB, float>(*q, shape_view<float>(128, 8192), Uplo::Lower, false);
    EXPECT_EQ(nb16.param("NB"), 16);
}

// One planned call for a UR-trace comparison of plan vs launch (BATCHLAS_PLANNER_ONE=dtype,n,batch).
TEST_F(PlannerTest, TraceOne) {
    const char* one = std::getenv("BATCHLAS_PLANNER_ONE");
    if (!one) GTEST_SKIP() << "set BATCHLAS_PLANNER_ONE=float,128,8192";
    std::string s(one);
    const std::string dt = s.substr(0, s.find(','));
    s = s.substr(s.find(',') + 1);
    const int n = std::stoi(s.substr(0, s.find(',')));
    const int b = std::stoi(s.substr(s.find(',') + 1));
    for_dtype(dt, [&]<typename T>() {
        std::printf("%s", pl::explain(pl::potrf_plan_node<kB, T>(*q, shape_view<T>(n, b), Uplo::Lower, false)).c_str());
        const RunOut o = run_one<T>(*q, n, b, Uplo::Lower);
        std::printf("[trace-one] tier=%s residual=%.3e\n", o.tier.c_str(), o.res);
    });
}

// ---- 4. pins: recorded, warned once, strict throws -----------------------------------------

TEST_F(PlannerTest, UnhonourablePinIsRecordedWarnedOnceOrThrows) {
    ScopedEnvVar e("BATCHLAS_POTRF_ROUTE", "native:lpanel");
    Matrix<float, MatrixFormat::Dense> A(96, 96, 64);
    const auto nd = pl::potrf_plan_node<kB, float>(*q, A.view(), Uplo::Upper, false);
    EXPECT_FALSE(nd.decision.honoured);
    EXPECT_EQ(nd.decision.why, pl::Why::PinRejected);
    std::printf("[pin] Upper n=96 pinned native:lpanel -> %s (honoured=%d)\n", std::string(nd.tier).c_str(),
                int(nd.decision.honoured));
    {
        ScopedEnvVar s("BATCHLAS_ROUTE_STRICT", "1");
        EXPECT_THROW((pl::potrf_plan_node<kB, float>(*q, A.view(), Uplo::Upper, false)), invalid_argument);
    }
    // Three unservable calls (three distinct plans) under one name pin: ONE warning.
    ScopedEnvVar n("BATCHLAS_ROUTE", "potrf=native:lpanel");
    testing::internal::CaptureStderr();
    for (int b : {65, 66, 67}) {
        const auto v = shape_view<float>(96, b);
        const auto r = pl::potrf_plan_node<kB, float>(*q, v, Uplo::Upper, false);
        EXPECT_FALSE(r.decision.honoured);
    }
    const std::string err = testing::internal::GetCapturedStderr();
    std::size_t count = 0;
    for (std::size_t p = err.find("cannot serve"); p != std::string::npos; p = err.find("cannot serve", p + 1)) ++count;
    EXPECT_EQ(count, 1u) << err;
}

// A route pin means the route's OWNER: `native:cta` never runs CtaWg, whatever the order.
TEST_F(PlannerTest, RoutePinResolvesToTheOwnerTier) {
    namespace pt = pl::potrf;
    static_assert(pt::Tiers<float>::owner_name(Route{Origin::Native, Algorithm::CTA}) == "native:cta");
    using First = pl::TierList<pt::CtaWg<float>, pt::Tiny<float>, pt::Cta<float>, pt::LPanel<float>, pt::Blocked<float>, pt::Vendor<float>>;
    static_assert(First::owner_name(Route{Origin::Native, Algorithm::CTA}) == "native:cta");
    ScopedEnvVar e("BATCHLAS_POTRF_ROUTE", "native:cta");
    const Cell c{"float", 20, 2048};
    EXPECT_EQ((plan_with<float, First>(*q, c, Uplo::Lower, false)), "native:cta");
    EXPECT_EQ((plan_with<float, pt::Tiers<float>>(*q, c, Uplo::Lower, false)), "native:cta");
}

// ---- 5. EXPLAIN, the cache, and the plan's geometry ----------------------------------------

TEST_F(PlannerTest, ExplainAndCache) {
    const auto A = shape_view<float>(128, 8192);
    const auto nd = pl::potrf_plan_node<kB, float>(*q, A, Uplo::Lower, false);
    const std::string ex = pl::explain(nd);
    std::printf("%s", ex.c_str());
    if (kVendor) {
        EXPECT_NE(ex.find("-> native:lpanel"), std::string::npos);
        EXPECT_NE(ex.find("why: cheapest (chooser cost-model)"), std::string::npos);
    }
    EXPECT_EQ(nd.param("NB"), 8);
    EXPECT_EQ(nd.launch.wg_size, 128);
    EXPECT_EQ(nd.launch.groups, 8192);
    const auto st0 = pl::cache_stats();
    for (int i = 0; i < 3; ++i) (void)pl::potrf_plan_node<kB, float>(*q, A, Uplo::Lower, false);
    // ld and stride are not in the key at batch > 1: the plan does not read them.
    MatrixView<float, MatrixFormat::Dense> padded(nullptr, 128, 128, 131, 131 * 128 + 7, 8192);
    (void)pl::potrf_plan_node<kB, float>(*q, padded, Uplo::Lower, false);
    const auto st1 = pl::cache_stats();
    EXPECT_EQ(st1.misses, st0.misses);
    EXPECT_EQ(st1.hits, st0.hits + 4);
    const auto big = shape_view<float>(640, 512);
    std::printf("%s", pl::explain(pl::potrf_plan_node<kB, float>(*q, big, Uplo::Lower, false)).c_str());

    // Cost per call: uncached plan, cached plan, today's resolve (each sized call does it twice).
    auto time = [](auto&& fn) {
        const int iters = 20000;
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) fn();
        return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / iters;
    };
    const double cached = time([&] { (void)pl::potrf_plan_cached<kB, float>(*q, A, Uplo::Lower); });
    const double uncached = time([&] { (void)pl::potrf_build<kB, float>(*q, A, Uplo::Lower, pl::request_for_potrf(), false); });
    const double today = time([&] { (void)backend::potrf_route<kB, float>(*q, A, Uplo::Lower, true); });
    std::printf("[time] per call: cached plan %.3f us, uncached plan %.3f us, today's potrf_route %.3f us\n",
                cached, uncached, today);
}

// The cache is bounded: distinct shapes past the capacity evict, they do not grow it.
TEST_F(PlannerTest, CacheIsBounded) {
    const int distinct = 1500;   // > the default capacity of 1024
    for (int b = 1; b <= distinct; ++b) (void)pl::potrf_plan_node<kB, float>(*q, shape_view<float>(16, b), Uplo::Lower, false);
    const auto st = pl::cache_stats();
    std::printf("[cache] %d distinct shapes: size=%llu misses=%llu evictions=%llu\n", distinct,
                (unsigned long long)st.size, (unsigned long long)st.misses, (unsigned long long)st.evictions);
    EXPECT_LE(st.size, 1024u);
    EXPECT_EQ(st.misses, static_cast<std::uint64_t>(distinct));
    EXPECT_EQ(st.size + st.evictions, static_cast<std::uint64_t>(distinct));
    // An evicted plan is rebuilt, identically.
    const auto again = pl::potrf_plan_node<kB, float>(*q, shape_view<float>(16, 1), Uplo::Lower, false);
    const auto rebuilt = pl::potrf_build<kB, float>(*q, shape_view<float>(16, 1), Uplo::Lower, pl::request_for_potrf(), false);
    EXPECT_EQ(again.tier, rebuilt.tier);
}

// Concurrent cached lookups vs today's resolver (printed, not asserted: a timing).
TEST_F(PlannerTest, ConcurrentLookupThroughput) {
    const auto A = shape_view<float>(8, 256);
    auto run = [&](int threads, auto&& fn) {
        std::vector<std::thread> ts;
        const auto t0 = std::chrono::steady_clock::now();
        for (int t = 0; t < threads; ++t) ts.emplace_back([&] { for (int i = 0; i < 20000; ++i) fn(); });
        for (auto& t : ts) t.join();
        return threads * 20000.0 / std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / 1e6;
    };
    for (int th : {1, 8}) {
        const double plan = run(th, [&] { (void)pl::potrf_plan_cached<kB, float>(*q, A, Uplo::Lower); });
        const double old = run(th, [&] { (void)backend::potrf_route<kB, float>(*q, A, Uplo::Lower, kVendor); });
        std::printf("[mt] threads=%d cached plan %.2f Mcalls/s, today's potrf_route %.2f Mcalls/s (ratio %.2f)\n", th,
                    plan, old, plan / old);
    }
}

// ---- 6. the extension tier, and the candidacy safety net -----------------------------------

TEST_F(PlannerTest, ExtensionTierRunsOnlyWhenPinned) {
    exec_case<float>(*q, "float L n=20 b=2048 (pin cta_wg)", 20, 2048, Uplo::Lower, "native:cta_wg", "native:cta_wg");
    exec_case<std::complex<double>>(*q, "cdouble U n=24 b=512 (pin cta_wg)", 24, 512, Uplo::Upper, "native:cta_wg",
                                    "native:cta_wg");
    const auto nd = pl::potrf_plan_node<kB, float>(*q, shape_view<float>(20, 2048), Uplo::Lower, false);
    for (const auto& c : nd.decision.candidates) {
        if (c.name == "native:cta_wg") {
            EXPECT_TRUE(c.legal.ok);
            EXPECT_FALSE(c.candidate);
            EXPECT_FALSE(c.selectable);
            EXPECT_LT(c.seconds, 0);
        }
    }
    ScopedEnvVar e("BATCHLAS_ROUTE", "potrf=native:cta_wg");
    const auto pinned = pl::potrf_plan_node<kB, float>(*q, shape_view<float>(20, 2048), Uplo::Lower, false);
    EXPECT_EQ(pinned.kernel, "PotrfCtaKernel<T,NB,TS,WorkGroup>");
    EXPECT_EQ(pinned.param("L"), 64);
}

// The fallback choosers read `selectable` too: a table where the unmeasured tier is the ONLY
// legal native, or the first one, still never yields it.
TEST(PlannerData, ChoosersNeverReturnAnUnselectableTier) {
    std::vector<pl::Candidate> cs(3);
    cs[0].route = {Origin::Native, Algorithm::CTA}; cs[0].name = "native:cta_wg"; cs[0].legal = pl::Legality::yes();
    cs[0].selectable = false;
    cs[1].route = {Origin::Native, Algorithm::CTA}; cs[1].name = "native:cta"; cs[1].legal = pl::Legality::no("n");
    cs[2].route = {Origin::Vendor, Algorithm::Auto}; cs[2].name = "vendor"; cs[2].legal = pl::Legality::no("absent");
    EXPECT_LT(pl::FirstLegalChooser{}.choose(cs).index, 0);
    EXPECT_LT(pl::CostModelChooser{0.1}.choose(cs).index, 0);
    const pl::WindowRow rows[] = {{dispatch::ScalarKind::F32, 0, 1, 64, "native:cta_wg"}};
    EXPECT_LT((pl::WindowChooser{rows, false, false}.choose(cs, dispatch::ScalarKind::F32, 0, 20).index), 0);
    cs[1].legal = pl::Legality::yes();
    EXPECT_EQ(pl::FirstLegalChooser{}.choose(cs).index, 1);
    EXPECT_EQ(pl::CostModelChooser{0.1}.choose(cs).index, 1);
}

// Every tier has an explicit candidacy decision per (arch, dtype, uplo): a profile column, a
// window row, the unmeasured list, or the vendor. A new tier missing from all four fails here.
TEST(PlannerData, EveryTierHasAnExplicitCandidacyDecision) {
    std::vector<std::string_view> names(pl::potrf::Tiers<float>::names.begin(), pl::potrf::Tiers<float>::names.end());
    for (std::string_view t : names) {
        bool decided = (t == "vendor");
        for (auto c : pl::potrf::kProfileColumns) decided |= (c == t);
        for (const auto& w : pl::potrf::kVsVendor) decided |= (w.tier == t);
        for (const auto& w : pl::potrf::kNativeTier) decided |= (w.tier == t);
        for (auto u : pl::potrf::kUnmeasured) decided |= (u == t);
        EXPECT_TRUE(decided) << "tier " << t << " has no candidacy decision";
    }
}

}  // namespace
