#include <batchlas/util/minibench.hh>
#include <batchlas/blas/linalg.hh>
#include <batchlas/blas/dispatch/route_env.hh>
#include <batchlas/backend_config.h>
#include "bench_utils.hh"

#include <cstdio>
#include <string>
#include <vector>
using namespace batchlas;

template <typename T, Backend B>
static void BM_TRSM(minibench::State& state) {
    // SquareBatchSizes emits Args({s, s, bs}): batch is range(2), not range(1).
    const size_t n = state.range(0);
    const size_t batch = state.range(2);

    auto A = Matrix<T>::Triangular(n, Uplo::Lower, T(1), T(0.5), batch);
    auto Bm = Matrix<T>::Random(n, n, false, batch);

    auto q = std::make_shared<Queue>(Device(B == Backend::NETLIB ? "cpu" : "gpu"), B);
    state.SetKernel(q,
                    std::move(A),
                    bench::pristine(Bm),
                    T(1),
                    Side::Left,
                    Uplo::Lower,
                    Transpose::NoTrans,
                    Diag::NonUnit,
                    [](Queue& q, auto&&... xs) {
                        trsm(q, std::forward<decltype(xs)>(xs)...);
                    });
    // TRSM does n^2 * q flops; B is square here, so q == n.
    state.SetMetric("GFLOPS", static_cast<double>(batch) * (1e-9 * n * n * n), minibench::Rate);
    state.SetMetric("Time (µs) / matrix", (1.0 / batch) * 1e6, minibench::Reciprocal);
}

BATCHLAS_REGISTER_BENCHMARK_ALL_TYPES(BM_TRSM, SquareBatchSizes);

// The ortho-shaped grid: the only shape the library issues -- ortho.cc passes a k x k
// Cholesky factor as A and an m x k basis as B, so n is small and q large.
// evidence: docs/perf/trsm.md#the-measured-grid

// The cap is computed for complex<double> and applied to every type, so all four run
// an identical grid. Dropped cells are printed.
static constexpr double kTrsmGridCapBytes = 6.0e9;   // of 24 GB

static inline double trsm_grid_bytes(double n, double q, double batch) {
    // 16 = sizeof(std::complex<double>), the worst type; 2*q*n counts B and the
    // pristine copy the harness keeps so an in-place solve can be re-run.
    return 16.0 * batch * (2.0 * q * n + n * n);
}

template <typename Benchmark>
inline void TrsmOrthoSizes(Benchmark* b) {
    static bool announced = false;
    std::vector<std::string> dropped;
    for (int n : {8, 16, 32, 64, 128, 256}) {
        for (int q : {256, 1024, 4096}) {
            for (int bs : {128, 512, 2048}) {
                const double gb = trsm_grid_bytes(n, q, bs) / 1e9;
                if (gb * 1e9 > kTrsmGridCapBytes) {
                    char buf[128];
                    std::snprintf(buf, sizeof buf, "n=%d q=%d batch=%d (%.1f GB)", n, q, bs, gb);
                    dropped.emplace_back(buf);
                    continue;
                }
                b->Args({n, q, bs});
            }
        }
    }
    if (!announced && !dropped.empty()) {
        announced = true;
        std::fprintf(stderr, "trsm ortho grid: %zu of 54 cells dropped, over the %.1f GB cap:\n",
                     dropped.size(), kTrsmGridCapBytes / 1e9);
        for (const auto& d : dropped) std::fprintf(stderr, "    %s\n", d.c_str());
    }
}

// Profile-only, NOT for ranking: these rows are unsaturated, so a ratio read off them
// is an overhead ratio, not an algorithm one.
template <typename Benchmark>
inline void TrsmOrthoStarvedSizes(Benchmark* b) {
    for (int n : {8, 32, 128}) {
        for (int q : {32, 128}) {
            for (int bs : {1, 8, 32}) {
                b->Args({n, q, bs});
            }
        }
    }
}

// An unrecognised BATCHLAS_TRSM_ROUTE silently measures the default route on both
// sides of an A/B and reports 1.0, so announce the parse once per process.
static void trsm_announce_route_env() {
    static bool done = false;
    if (done) return;
    done = true;
    const auto p = batchlas::dispatch::parse_route_env(batchlas::dispatch::Op::trsm);
    if (p.unparsed) {
        std::fprintf(stderr,
                     "\n*** BATCHLAS_TRSM_ROUTE=\"%s\" WAS NOT UNDERSTOOD. This run measures the\n"
                     "*** default route. Accepted: vendor | native | cta | blocked | native:cta\n\n",
                     p.source.value.c_str());
    } else if (p.found) {
        std::fprintf(stderr, "trsm route forced: %s -> %s:%s\n", p.source.value.c_str(),
                     std::string(batchlas::dispatch::to_string(p.route.origin)).c_str(),
                     std::string(batchlas::dispatch::to_string(p.route.algo)).c_str());
    } else {
        std::fprintf(stderr, "trsm route: unset (resolver's choice)\n");
    }
}

// The two ortho cells, as one body. SD picks which extent of B carries q.
template <typename T, Backend Bk, Side SD>
static void BM_TRSM_OrthoBody(minibench::State& state) {
    trsm_announce_route_env();

    const size_t n     = state.range(0);
    const size_t q     = state.range(1);
    const size_t batch = state.range(2);

    // ConjTrans for complex, Trans for real -- what ortho.cc issues; the resolver keys
    // on the enum, so the two spellings are not interchangeable here.
    constexpr Transpose kTrans =
        batchlas::is_std_complex_v<T> ? Transpose::ConjTrans : Transpose::Trans;
    constexpr Transpose trans = (SD == Side::Right) ? kTrans : Transpose::NoTrans;

    auto A = Matrix<T>::Triangular(n, Uplo::Lower, T(1), T(0.5), batch);
    auto Bm = (SD == Side::Right) ? Matrix<T>::Random(q, n, false, batch)
                                  : Matrix<T>::Random(n, q, false, batch);

    auto qh = std::make_shared<Queue>(Device(Bk == Backend::NETLIB ? "cpu" : "gpu"), Bk);
    state.SetKernel(qh,
                    std::move(A),
                    bench::pristine(Bm),
                    T(1),
                    SD,
                    Uplo::Lower,
                    trans,
                    Diag::NonUnit,
                    [](Queue& qq, auto&&... xs) {
                        trsm(qq, std::forward<decltype(xs)>(xs)...);
                    });

    // Real-arithmetic flop convention for all four types, so complex GFLOPS understates
    // by 4x by construction.
    state.SetMetric("GFLOPS", double(batch) * (1e-9 * double(n) * double(n) * double(q)),
                    minibench::Rate);
    state.SetMetric("Time (us) / matrix", (1.0 / double(batch)) * 1e6, minibench::Reciprocal);
}

template <typename T, Backend Bk>
static void BM_TRSM_OrthoRight(minibench::State& s) { BM_TRSM_OrthoBody<T, Bk, Side::Right>(s); }
template <typename T, Backend Bk>
static void BM_TRSM_OrthoLeft(minibench::State& s)  { BM_TRSM_OrthoBody<T, Bk, Side::Left>(s); }
// NOT named *OrthoRightStarved: --name matches by SUBSTRING, so that spelling would
// fold these profile-only rows into every ranking run.
template <typename T, Backend Bk>
static void BM_TRSM_StarvedRight(minibench::State& s) { BM_TRSM_OrthoBody<T, Bk, Side::Right>(s); }
template <typename T, Backend Bk>
static void BM_TRSM_StarvedLeft(minibench::State& s)  { BM_TRSM_OrthoBody<T, Bk, Side::Left>(s); }

BATCHLAS_BENCH_CUDA_ALL_TYPES(BM_TRSM_OrthoRight, TrsmOrthoSizes);
BATCHLAS_BENCH_CUDA_ALL_TYPES(BM_TRSM_OrthoLeft,  TrsmOrthoSizes);
BATCHLAS_BENCH_CUDA_ALL_TYPES(BM_TRSM_StarvedRight, TrsmOrthoStarvedSizes);
BATCHLAS_BENCH_CUDA_ALL_TYPES(BM_TRSM_StarvedLeft,  TrsmOrthoStarvedSizes);

MINI_BENCHMARK_MAIN();
