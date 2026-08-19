#include <batchlas/util/minibench.hh>
#include <batchlas/blas/linalg.hh>
#include <batchlas/blas/dispatch/route_env.hh>
#include <batchlas/backend_config.h>
#include "bench_utils.hh"

#include <cstdio>
#include <string>
#include <vector>
using namespace batchlas;

// Single TRSM benchmark
template <typename T, Backend B>
static void BM_TRSM(minibench::State& state) {
    // SquareBatchSizes (include/batchlas/util/minibench.hh:790) emits
    // Args({s, s, bs}) -- three arguments, (n, n, batch). This read used to be
    // range(1), i.e. the SECOND n, so `batch` was always equal to `n` and the
    // batch argument was silently ignored: every one of the ten batch rows per
    // size ran the identical problem. It is why the table showed a dead-flat
    // 0.0324 ms from batch 1 to batch 512. gemv_benchmark.cc:11-13 and
    // ormqr_benchmark.cc:11-13 read range(2) and were always correct; this file
    // was the only one that did not.
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
    // TRSM solves op(A) X = alpha B with A n x n and B n x q, which is n^2 * q
    // flops, not n^2. Here B is square so q == n. The old expression omitted q
    // and therefore understated the rate by exactly a factor of n -- 8.3 GFLOP/s
    // reported where the kernel was doing 518.
    state.SetMetric("GFLOPS", static_cast<double>(batch) * (1e-9 * n * n * n), minibench::Rate);
    state.SetMetric("Time (µs) / matrix", (1.0 / batch) * 1e6, minibench::Reciprocal);
}

BATCHLAS_REGISTER_BENCHMARK_ALL_TYPES(BM_TRSM, SquareBatchSizes);

// ===========================================================================
// THE ORTHO-SHAPED GRID (WP3 spec S10).
//
// WHY A SECOND GRID. The benchmark above sweeps a square RHS -- A is n x n and
// B is n x n -- and the library never issues that shape. The only two trsm call
// sites in the tree are src/extensions/ortho.cc:202 and :289, both inside
// chol_alg, and both pass the k x k Cholesky factor as A and the m x k basis as
// B. So the triangular order is the ORTHONORMALISATION BLOCK SIZE and the other
// extent is the VECTOR LENGTH, one to three orders of magnitude larger. A
// coverage capture (.route-diff/wp3-cx.csv) confirms it against what the suite
// actually issues rather than against the source:
//
//   n=10  q=256  batch=1  calls=4880      n=10  q=20  batch=1  calls=4800
//   n=12  q=36   batch=3  calls=2392      n=5   q=64  batch=3  calls=3258
//
// -- every one Side::Right, Uplo::Lower, Transpose::Trans, NonUnit, and every
// one with a triangular order inside V1's CTA capacity of 32. Tuning trsm on
// the square grid would be tuning a shape that has no caller.
//
// The batch counts above are small only because they come from correctness
// tests. Ranking happens at large batch, per this project's standing rule that
// batch=1 is not the regime of interest.
// ===========================================================================

// ---------------------------------------------------------------------------
// The grid, capped so that it fits.
//
// S10 asks for n in {8..256} x q in {256,1024,4096} x batch in {128,512,2048}.
// NINE OF THOSE 54 CELLS DO NOT FIT IN THIS MACHINE'S 24 GB, the largest asking
// for 36.5 GB:
//
//   n=256 q=4096 batch=2048 -> 70.9 GB     n=128 q=4096 batch=2048 -> 34.9 GB
//   n=256 q=1024 batch=2048 -> 19.3 GB     n=256 q=4096 batch=512  -> 17.7 GB
//   n=64  q=4096 batch=2048 -> 17.3 GB     n=128 q=1024 batch=2048 ->  9.1 GB
//   n=128 q=4096 batch=512  ->  8.7 GB     n=32  q=4096 batch=2048 ->  8.6 GB
//   n=256 q=256  batch=2048 ->  6.4 GB
//
// These are the figures trsm_grid_bytes() actually computes and the sizer
// actually prints; do not re-derive them by hand. An earlier draft of this
// table omitted the pristine copy and so understated every row by ~2x, which
// would have put four of these cells back in the grid on paper while the code
// kept dropping them.
//
// (complex<double>, counting A = n*n*batch, B = q*n*batch, and the pristine
// copy of B the harness holds so an in-place solve can be re-run.)
//
// So the grid is capped -- and the dropped cells are PRINTED, not silently
// skipped. A grid that quietly shrinks reads exactly like a grid that covered
// everything, which is how a coverage claim becomes false with nobody editing
// it.
//
// The cap is computed for complex<double> and applied to ALL types, so the four
// types run an identical grid and their rows stay comparable. float could
// afford more; extending it for float alone would make the grid the thing that
// differs between the type columns.
// ---------------------------------------------------------------------------
static constexpr double kTrsmGridCapBytes = 6.0e9;   // of 24 GB, leaving the
                                                     // co-tenant 4090 room

static inline double trsm_grid_bytes(double n, double q, double batch) {
    // 16 = sizeof(std::complex<double>), the worst type. 2*q*n for B and its
    // pristine copy; n*n for A, which trsm does not write and so is not copied.
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

// Profile-only, NOT for ranking. S10: "profiling only at saturation is exactly
// what hid the batch-only-parallelism defect in this repo for months." These
// rows exist so the starvation knee is visible; a ratio read off them is an
// overhead ratio, not an algorithm ratio.
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

// Report, once per process, exactly what BATCHLAS_TRSM_ROUTE was understood to
// mean. The A/B below is driven entirely by that variable, and route_env.hh
// reports `unparsed` for a value it does not recognise -- in which case the run
// silently measures the DEFAULT route twice and reports a ratio of 1.0 as
// though it were a finding. This repo has already lost time to exactly that
// (BATCHLAS_SYEV_PROVIDER=TWOSTAGE parsed as Auto and looked like a null
// result). Printing the parse makes that failure loud instead of plausible.
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

    // ConjTrans for complex, Trans for real -- what ortho.cc issues. Its
    // inv_trans is the conjugate transpose; for a real scalar the two spellings
    // describe the same arithmetic but are NOT the same enum, and the resolver
    // keys on the enum, so writing Trans for complex would benchmark a cell no
    // caller asks for.
    constexpr Transpose kTrans =
        batchlas::is_std_complex_v<T> ? Transpose::ConjTrans : Transpose::Trans;
    constexpr Transpose trans = (SD == Side::Right) ? kTrans : Transpose::NoTrans;

    auto A = Matrix<T>::Triangular(n, Uplo::Lower, T(1), T(0.5), batch);
    // B is q x n for a right-side solve and n x q for a left-side one.
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

    // n^2 * q per matrix. The real-arithmetic convention is kept for all four
    // types so the type columns compare directly; a complex flop is four real
    // ones, so complex GFLOPS here understates by 4x by construction.
    state.SetMetric("GFLOPS", double(batch) * (1e-9 * double(n) * double(n) * double(q)),
                    minibench::Rate);
    state.SetMetric("Time (us) / matrix", (1.0 / double(batch)) * 1e6, minibench::Reciprocal);
}

template <typename T, Backend Bk>
static void BM_TRSM_OrthoRight(minibench::State& s) { BM_TRSM_OrthoBody<T, Bk, Side::Right>(s); }
template <typename T, Backend Bk>
static void BM_TRSM_OrthoLeft(minibench::State& s)  { BM_TRSM_OrthoBody<T, Bk, Side::Left>(s); }
// Separate symbols, not a second registration of the same one: the registry
// keys on the stringified function name, so re-registering BM_TRSM_OrthoRight
// with a second sizer would produce two identically-named row groups and the
// saturated and starved numbers would be indistinguishable in the CSV.
//
// AND THE NAME MUST NOT CONTAIN THE RANKING SYMBOL'S NAME. --name= matches by
// SUBSTRING, so "BM_TRSM_OrthoRightStarved" -- the obvious spelling -- would be
// selected by every --name=BM_TRSM_OrthoRight run, quietly folding starved rows
// into the saturated grid they exist to be excluded from. Hence StarvedRight,
// not OrthoRightStarved.
template <typename T, Backend Bk>
static void BM_TRSM_StarvedRight(minibench::State& s) { BM_TRSM_OrthoBody<T, Bk, Side::Right>(s); }
template <typename T, Backend Bk>
static void BM_TRSM_StarvedLeft(minibench::State& s)  { BM_TRSM_OrthoBody<T, Bk, Side::Left>(s); }

BATCHLAS_BENCH_CUDA_ALL_TYPES(BM_TRSM_OrthoRight, TrsmOrthoSizes);
BATCHLAS_BENCH_CUDA_ALL_TYPES(BM_TRSM_OrthoLeft,  TrsmOrthoSizes);
BATCHLAS_BENCH_CUDA_ALL_TYPES(BM_TRSM_StarvedRight, TrsmOrthoStarvedSizes);
BATCHLAS_BENCH_CUDA_ALL_TYPES(BM_TRSM_StarvedLeft,  TrsmOrthoStarvedSizes);

MINI_BENCHMARK_MAIN();
