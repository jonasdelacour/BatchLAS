// Batched CSR SpMM -- C = alpha * A * op(B) + beta * C -- measured at the shapes
// its real consumers use.
//
// ---------------------------------------------------------------- what this replaces
// The previous version of this file built A as
//     Matrix<T>::Random(m, k, false, batch).convert_to<MatrixFormat::CSR>()
// and swept CubeBatchSizes. Random draws from uniform_real_distribution(-1, 1)
// (src/matrix.cc:1126) and convert_to keeps every entry whose magnitude exceeds a
// default 1e-7 threshold (src/matrix.cc:367, :415-426), so an entry survived with
// probability 1 - 1e-7: nnz was m*k, the "sparse" matrix was 100% dense, and the
// reported 2*nnz*n GFLOPS was a dense GEMM's flop count in a CSR costume. Nothing
// downstream inherits that grid -- there is no spmm CSV in benchmarks/results/.
//
// ---------------------------------------------------------------- the grid
// The two regimes that call spmm in this library:
//   lanczos  (src/extensions/lanczos.cc:110, inside a loop run n times at :175) --
//            nrhs 1-2, few nnz per row, large batch, and n spmm calls per solve, so
//            the per-call HOST cost is a first-class term, not noise.
//   LOBPCG / syevx_filtered (src/extensions/syevx_lobpcg.cc:626, :642, :1215;
//            src/extensions/syevx_filtered.cc:243) -- m 1024-4096, ~16 nnz/row (the
//            constant benchmarks/syevx_benchmark.cc:208 sweeps at), nrhs 12-50,
//            modest batch.
// nnz-per-row is an explicit axis, held CONSTANT as m grows: a fixed density would
// make nnz grow as m^2 and quietly turn a sparse sweep into a dense one, which is
// the same mistake the old file made by a different route.
//
// Three named cells are carried through the whole report:
//   L (lanczos)  m=1024, nnz/row=3,  nrhs=2,  batch=512
//   M (LOBPCG)   m=1024, nnz/row=16, nrhs=12, batch=512
//   S (LOBPCG)   m=2048, nnz/row=16, nrhs=25, batch=128
//
// ---------------------------------------------------------------- what is timed
// The timed region is a loop of K back-to-back spmm calls on one queue, closed by
// a single wait, timed on the host clock. Specifically:
//
//   INSIDE  the K calls themselves, including everything spmm_vendor does per call:
//           handle.setStream (src/backends/cusparse.cc:39), the
//           spmm_vendor_buffer_size re-query it performs on EVERY call (:42, which
//           is a second setStream at :72 and a cusparseSpMM_bufferSize at :75), the
//           BumpAllocator carve, and the cusparseSpMM launch. None of that can be
//           hoisted from the caller's side, so it is charged to both arms alike.
//   OUTSIDE setup, the CSR/dense generation, the one-time spmm_buffer_size query
//           that sizes the workspace, USM prefetch, one full untimed warm call
//           (SetPrepare below) and minibench's own two warmup batches.
//
// Event timing is deliberately NOT installed. bench::make_event_timed_kernel_ms
// costs a recorded ~0.36 ms per call (docs/perf/spmm.md#measurement-harness-and-hygiene); cell L's
// ideal-traffic roof is 22.9 us, so event timing would be 15x the thing measured.
// With no timed-kernel hook installed, minibench's structured path escalates K
// until one batch exceeds 1 ms (include/batchlas/util/minibench.hh:353-368) and
// then times batches of K on the host clock, which is what makes the host clock
// usable here.
//
// ---------------------------------------------------------------- the two arms
// WARM (BM_SPMM_*): the MatrixViews are built ONCE and live for the whole run, so
//   the cuSPARSE descriptors -- created lazily by MatrixView::operator*
//   (src/matrix.cc:2399-2406) -- are built on the first (untimed) call and reused.
//   This is the honest vendor baseline.
// COLD (BM_SPMM_Cold*): the Matrix overload of spmm, which constructs a fresh
//   MatrixView per call (include/batchlas/blas/functions/spmm.hh:47-57). Each fresh
//   view has a null backend handle, so every call pays a cusparseCreateCsr plus two
//   cusparseCreateDnMat -- and LEAKS them: ~BackendMatrixHandle is `= default`
//   (src/backends/backend_handle_impl.hh:25) and no cusparseDestroy*Descr exists
//   anywhere in src/ (only cusparseDestroy, at src/linalg-impl.hh:976). That leak is
//   why the cold arm is kept to a short cell list rather than run over the whole grid.
// COLD minus WARM is the per-call host chain, and it is the term lanczos pays n
// times per solve. Comparing a native kernel against the COLD arm would flatter it
// dishonestly; quote the WARM arm for any kernel-vs-kernel ratio.
//
// ---------------------------------------------------------------- route selection
// spmm takes no Order argument. Select the route with the environment, in separate
// processes, one route per process:
//     BATCHLAS_SPMM_ROUTE=vendor   BATCHLAS_SPMM_ROUTE=direct
// An unrecognised word parses to nothing and falls back SILENTLY (this campaign has
// already lost a measurement to `TWOSTAGE` silently meaning Auto), so confirm the
// route actually taken from the coverage table / scripts/route_diff.sh, never from
// the environment variable alone.
//
// ---------------------------------------------------------------- metrics
// GFLOPS is computed from the REAL per-item non-zero count (row_offsets[m] -
// row_offsets[0] on item 0), never from Matrix::nnz(), which is a per-item
// CAPACITY (include/batchlas/blas/matrix.hh:825-837). "GB/s" is effective
// bandwidth on the ideal-traffic model, which is the metric that matters here:
// SpMM at these densities is roofline-bound at 3.5-5% of FP32 peak, so a low
// GFLOPS number is correct rather than a defect.
// "chk" is the L1 norm of batch item 0 of C after the run. It exists because
// src/backends/cusparse.cc checks no cusparseStatus_t: an unsupported combination
// (transB = Trans is the one to distrust) returns an error, does nothing, and
// reports a flatteringly small time. chk == 0 on a beta = 0 row means the call was
// a no-op and the row must be thrown away.
//
// ---------------------------------------------------------------- arguments
//   arg0 m       rows of A; A is square, so k = m
//   arg1 nnzrow  non-zeros per row, held constant as m grows
//   arg2 nrhs    dense width, = C.cols()
//   arg3 batch
//   arg4 transB  0 = NoTrans (B is k x nrhs), 1 = Trans (B is nrhs x k)
//   arg5 beta    0 or 1
//   arg6 pattern 0 = banded, 1 = random
//   arg7 transA  0 = NoTrans (gather body), 1 = Trans, 2 = ConjTrans (scale +
//                atomic scatter bodies). Absent from every registered sizer, so
//                it defaults to 0 there; supply it from the command line.
//
// transB is the layout lever, measured directly rather than argued about: the dense
// operands are column-major (CUSPARSE_ORDER_COL,
// src/backends/backend_handle_impl.hh:58), so with B stored as nrhs x k and passed
// as Trans, the nrhs values one non-zero needs are CONTIGUOUS instead of ld apart.
// The pattern axis exists for the same reason on the other operand: the banded
// generator gives a tridiagonal-like j in [i - r/2, i + r/2), which is the locality
// lanczos actually has, while the random generator scatters columns across the row
// and is the harder gather LOBPCG actually has.
//
// NETLIB is deliberately not registered. Its spmm is a serial host triple loop that
// throws on any transpose (src/backends/netlib_lapack.cc:248-250), and in a
// vendor-free build the TU is not compiled at all (src/backends/CMakeLists.txt
// gates it on LAPACKE and CBLAS), so the symbol throws NoRouteError and would take
// the whole process -- and the CSV -- down with it.

#include <batchlas/util/minibench.hh>
#include <batchlas/util/bench_structured.hh>

#include <batchlas/blas/linalg.hh>

#include "bench_utils.hh"

#include <batchlas/backend_config.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <complex>
#include <cstdint>
#include <memory>
#include <type_traits>

using namespace batchlas;

namespace {

// arg6, the pattern axis.
constexpr int kBanded = 0;
constexpr int kRandom = 1;

// A deterministic value in (-1, 1) that is never near zero, so a generated matrix
// has no accidental structural zeros and beta = 1 accumulation stays well scaled.
template <typename T>
T spmm_value(std::uint64_t h) {
    const auto unit = [](std::uint64_t x) {
        x ^= x >> 33;
        x *= 0xff51afd7ed558ccdULL;
        x ^= x >> 33;
        x *= 0xc4ceb9fe1a85ec53ULL;
        x ^= x >> 33;
        // Magnitude in [0.25, 0.75], signed on the low bit.
        const double u = static_cast<double>(x >> 11) * (1.0 / 9007199254740992.0);
        const double mag = 0.25 + 0.5 * u;
        return static_cast<batchlas::float_t<T>>((x & 1ull) ? mag : -mag);
    };
    if constexpr (std::is_same_v<T, std::complex<float>> ||
                  std::is_same_v<T, std::complex<double>>) {
        return T(unit(h), unit(h ^ 0x9e3779b97f4a7c15ull));
    } else {
        return T(unit(h));
    }
}

// Banded CSR with EXACTLY nnz_per_row entries in every row, column indices sorted.
// Row i holds columns [start, start + r) with start = clamp(i - r/2, 0, m - r), so
// the boundary rows reuse the first/last band instead of being short: a uniform row
// length keeps nnz exactly r*m and keeps the per-row work uniform, which is what
// the lanczos operator looks like. Written straight into the matrix' USM buffers --
// they are host-writable, and nothing has been enqueued against them yet.
template <typename T>
Matrix<T, MatrixFormat::CSR> make_banded_csr(int m, int nnz_per_row, int batch) {
    const int r = std::clamp(nnz_per_row, 1, m);
    const int nnz = m * r;

    Matrix<T, MatrixFormat::CSR> A(m, m, NonZeros{nnz}, batch);
    auto offsets = A.row_offsets();
    auto cols = A.col_indices();
    auto vals = A.data();

    for (int b = 0; b < batch; ++b) {
        const std::size_t obase = static_cast<std::size_t>(b) * static_cast<std::size_t>(m + 1);
        const std::size_t vbase = static_cast<std::size_t>(b) * static_cast<std::size_t>(nnz);
        for (int i = 0; i <= m; ++i) {
            offsets[obase + static_cast<std::size_t>(i)] = i * r;
        }
        for (int i = 0; i < m; ++i) {
            const int start = std::clamp(i - r / 2, 0, m - r);
            for (int d = 0; d < r; ++d) {
                const std::size_t p = vbase +
                                      static_cast<std::size_t>(i) * static_cast<std::size_t>(r) +
                                      static_cast<std::size_t>(d);
                cols[p] = start + d;
                vals[p] = spmm_value<T>((static_cast<std::uint64_t>(b) << 40) ^
                                        (static_cast<std::uint64_t>(i) << 12) ^
                                        static_cast<std::uint64_t>(start + d));
            }
        }
    }
    return A;
}

// Scattered pattern, via the in-tree generator the sparse syevx benchmark already
// uses (benchmarks/syevx_benchmark.cc:289, :341). density = nnz_per_row / m makes
// csr_random_nnz_per_matrix (src/matrix.cc:128-141) target nnz_per_row * m and land
// within one of it. shared_pattern keeps the column pattern identical across the
// batch, which is what a batched eigensolve on one operator looks like.
template <typename T>
Matrix<T, MatrixFormat::CSR> make_random_csr(int m, int nnz_per_row, int batch) {
    const float density = static_cast<float>(nnz_per_row) / static_cast<float>(m);
    return Matrix<T, MatrixFormat::CSR>::RandomSparseHermitian(
        m, density, batch, /*seed=*/42u, /*diagonal_boost=*/batchlas::float_t<T>(1),
        /*shared_pattern=*/true);
}

template <typename T>
Matrix<T, MatrixFormat::CSR> make_csr(int m, int nnz_per_row, int batch, int pattern) {
    return pattern == kBanded ? make_banded_csr<T>(m, nnz_per_row, batch)
                              : make_random_csr<T>(m, nnz_per_row, batch);
}

// Ideal traffic for one batch item, one pass over A, in bytes:
//   nnz*(s + 4)          values + 32-bit column indices
//   (m + 1)*4            row offsets
//   red_rows*nrhs*s      the B gather, charged at compulsory footprint
//   out_rows*nrhs*s      the C write
//   + out_rows*nrhs*s    the C read, only when beta != 0
// transA is NoTrans throughout, so red_rows = k = m and out_rows = m. This is the
// same model the work-package plan's roofline uses; at cell L it gives 45,060
// B/item, at M 233,476, at S 679,940.
double ideal_bytes_per_item(std::size_t elem_size, int m, int nnz, int nrhs,
                            bool beta_nonzero) {
    const double s = static_cast<double>(elem_size);
    const double dm = static_cast<double>(m);
    const double dn = static_cast<double>(nrhs);
    double bytes = static_cast<double>(nnz) * (s + 4.0) + (dm + 1.0) * 4.0 + 2.0 * dm * dn * s;
    if (beta_nonzero) bytes += dm * dn * s;
    return bytes;
}

// --------------------------------------------------------------------- the runner

enum class Arm { Warm, Cold };

template <typename T, Backend Bk, Arm A>
void run_spmm(minibench::State& state) {
    const int m = state.range(0);
    const int nnz_per_row = state.range(1);
    const int nrhs = state.range(2);
    const int batch = state.range(3);
    const auto transB = state.range(4) == 0 ? Transpose::NoTrans : Transpose::Trans;
    const T beta = T(static_cast<batchlas::float_t<T>>(state.range(5)));
    const int pattern = state.range(6);
    // arg7 -- the transA axis, added for the WP8 vendor bake-off. state.range()
    // returns 0 for an index the sizer never supplied (minibench.hh:159), so
    // every pre-existing sizer above keeps meaning NoTrans and every registered
    // grid is unchanged. A is SQUARE (k = m) throughout this file, so transposing
    // it changes no extent and no metric -- only which of the three native bodies
    // runs: NoTrans is the gather (spmm_native.cc body 1), Trans and ConjTrans are
    // the scale + atomic scatter (bodies 0 and 2).
    const auto transA = state.range(7) == 0   ? Transpose::NoTrans
                      : state.range(7) == 1   ? Transpose::Trans
                                              : Transpose::ConjTrans;
    const T alpha = T(1);

    auto q = std::make_shared<Queue>(Device(Bk == Backend::NETLIB ? "cpu" : "gpu"), Bk);

    auto Amat = std::make_shared<Matrix<T, MatrixFormat::CSR>>(
        make_csr<T>(m, nnz_per_row, batch, pattern));
    // op(B) must be k x nrhs with k = A.cols() = m, so the stored shape flips with
    // transB. Both are column-major; that is the whole point of the axis.
    auto Bmat = std::make_shared<Matrix<T>>(
        transB == Transpose::NoTrans ? Matrix<T>::Random(m, nrhs, false, batch)
                                     : Matrix<T>::Random(nrhs, m, false, batch));
    auto Cmat = std::make_shared<Matrix<T>>(Matrix<T>::Zeros(m, nrhs, batch));

    // The long-lived views. The warm arm hands these same objects to spmm on every
    // call, so the cuSPARSE descriptors are built once, on the untimed warm call.
    auto Av = std::make_shared<MatrixView<T, MatrixFormat::CSR>>(Amat->view());
    auto Bv = std::make_shared<MatrixView<T>>(Bmat->view());
    auto Cv = std::make_shared<MatrixView<T>>(Cmat->view());

    const std::size_t ws_size = spmm_buffer_size<Bk, T, MatrixFormat::CSR>(
        *q, *Av, *Bv, *Cv, alpha, beta, transA, transB);
    auto ws = std::make_shared<UnifiedVector<std::byte>>(ws_size);

    auto kernel_once = [q, Av, Bv, Cv, Amat, Bmat, Cmat, ws, alpha, beta, transA, transB]() {
        if constexpr (A == Arm::Warm) {
            spmm<Bk, T, MatrixFormat::CSR>(*q, *Av, *Bv, *Cv, alpha, beta,
                                           transA, transB, ws->to_span());
        } else {
            // The Matrix overload: a fresh MatrixView, and so a fresh descriptor
            // triple, per call. This is the arm that measures the per-call host
            // chain, and the arm that leaks.
            spmm<Bk, T, MatrixFormat::CSR>(*q, *Amat, *Bmat, *Cmat, alpha, beta,
                                           transA, transB, ws->to_span());
        }
    };

    bench::ManagedInputs managed(q);
    managed.prepare(*Amat).prepare(*Bmat).prepare(*Cmat);

    state.SetPrepare([prep = managed.make_prepare_once(), kernel_once, q]() mutable {
        prep();
        // One full untimed call: JITs any native kernel, builds the descriptors on
        // the long-lived views, and faults the USM pages in. minibench's own warmup
        // batches follow this; both are outside every timed region.
        kernel_once();
        q->wait();

        // ---------------------------------------------------------------
        // A TIME-BUDGETED CLOCK RAMP, because minibench's warm-up is counted
        // in CALLS and the thing being warmed is measured in MILLISECONDS.
        //
        // The SM clock on this box idles at 210 MHz against a measured 2805 MHz
        // ceiling. docs/perf/spmm.md#measurement-harness-and-hygiene shows what that
        // costs: with minibench's default 2 warm-up calls the FIRST row of a
        // process reads 0.1654 ms with rel_sd 0.0495, while the SECOND row of the
        // same process reads 0.1620 ms with rel_sd 0.0019 -- and 250 warm-up
        // calls bring the first row to 0.1618. So the ramp is a per-PROCESS cost
        // that lands entirely on whichever row runs first, and a CALL-counted
        // warm-up cannot price it: 250 calls is 40 ms of ramp on a cheap cell and
        // 13.5 SECONDS of dead time on a 54 ms cell (m=4096, nrhs=50, batch=512,
        // cdouble).
        //
        // So warm for a fixed WALL-CLOCK budget instead, in batches of 8 to keep
        // the sync out of the loop. Default 400 ms; override with
        // BATCHLAS_SPMM_WARM_MS. This is untimed and outside every measured
        // region, exactly like the single call above.
        // ---------------------------------------------------------------
        static const double warm_ms = [] {
            const char* p = std::getenv("BATCHLAS_SPMM_WARM_MS");
            return p ? std::atof(p) : 400.0;
        }();
        const auto t0 = std::chrono::steady_clock::now();
        while (true) {
            const double el = std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - t0).count();
            if (el >= warm_ms) break;
            for (int i = 0; i < 8; ++i) kernel_once();
            q->wait();
        }
    });
    state.SetBeforeEachRun(managed.make_before_each_run());
    state.SetKernel(std::function<void()>(kernel_once));
    state.SetBatchEndWait(q);
    // No SetTimedKernelMs: see the header note on the ~0.36 ms event-timer cost.

    // The real per-item non-zero count, not the capacity.
    const int nnz = Amat->nnz(0);
    constexpr bool complex_type = std::is_same_v<T, std::complex<float>> ||
                                  std::is_same_v<T, std::complex<double>>;
    const double flops = (complex_type ? 8.0 : 2.0) * static_cast<double>(nnz) *
                         static_cast<double>(nrhs);
    const double bytes = ideal_bytes_per_item(sizeof(T), m, nnz, nrhs, state.range(5) != 0);

    state.SetMetric("GFLOPS", static_cast<double>(batch) * 1e-9 * flops, minibench::Rate);
    state.SetMetric("GB/s", static_cast<double>(batch) * 1e-9 * bytes, minibench::Rate);
    state.SetMetric("Time (µs) / matrix",
                    (1.0 / static_cast<double>(batch)) * 1e6,
                    minibench::Reciprocal);
    state.SetMetric("nnz/item", static_cast<double>(nnz), minibench::Normal);
    state.SetMetric("nnz/row", static_cast<double>(nnz) / static_cast<double>(m),
                    minibench::Normal);

    // The no-op detector. cusparseSpMM's status is never checked, so a rejected
    // argument combination is indistinguishable from a fast kernel by time alone.
    state.SetMetricsFunc([q, Cmat, m, nrhs](minibench::Result& res) {
        q->wait();
        auto c = Cmat->data();
        const std::size_t n = static_cast<std::size_t>(m) * static_cast<std::size_t>(nrhs);
        double sum = 0.0;
        for (std::size_t i = 0; i < n; ++i) sum += std::abs(c[i]);
        res.metrics["chk"] = sum;
    });
}

// --------------------------------------------------------------- registered families

template <typename T, Backend Bk>
static void BM_SPMM_Cells(minibench::State& state) { run_spmm<T, Bk, Arm::Warm>(state); }

template <typename T, Backend Bk>
static void BM_SPMM_Grid(minibench::State& state) { run_spmm<T, Bk, Arm::Warm>(state); }

template <typename T, Backend Bk>
static void BM_SPMM_Lanczos(minibench::State& state) { run_spmm<T, Bk, Arm::Warm>(state); }

template <typename T, Backend Bk>
static void BM_SPMM_ColdCells(minibench::State& state) { run_spmm<T, Bk, Arm::Cold>(state); }

template <typename T, Backend Bk>
static void BM_SPMM_ColdLanczos(minibench::State& state) { run_spmm<T, Bk, Arm::Cold>(state); }

// ------------------------------------------------------------------------- sizers

// (m, nnz/row, nrhs, batch) for cells L, M and S.
constexpr int kCells[3][4] = {
    {1024,  3,  2, 512},   // L -- lanczos
    {1024, 16, 12, 512},   // M -- LOBPCG
    {2048, 16, 25, 128},   // S -- LOBPCG
};

// The three cells crossed with both layouts, both beta values and both patterns:
// 24 rows per type. Small enough to run on every pass, which is what makes the
// two-independent-passes rule (cross-pass spread <= 1.01 before quoting a cell)
// cheap to honour.
inline void SpmmCellSizes(minibench::Benchmark* b) {
    for (const auto& c : kCells) {
        for (int pattern : {kBanded, kRandom}) {
            for (int trans_b : {0, 1}) {
                for (int beta : {0, 1}) {
                    b->Args({c[0], c[1], c[2], c[3], trans_b, beta, pattern});
                }
            }
        }
    }
}

// The full four-axis grid, at the settings fixed for it: NoTrans, beta = 0,
// scattered pattern. 384 rows per type -- run one type per process.
//
// The largest row (m = 4096, 32 nnz/row, nrhs = 50, batch = 512) allocates ~4.7 GB
// for complex<double>: A values 1.07 GB, column indices 268 MB, B and C 1.68 GB
// each. That fits a 24 GB card but leaves little room, so do not run two types in
// one process.
inline void SpmmGridSizes(minibench::Benchmark* b) {
    for (int m : {512, 1024, 2048, 4096}) {
        for (int nnzrow : {3, 8, 16, 32}) {
            for (int nrhs : {1, 2, 4, 12, 25, 50}) {
                for (int batch : {8, 64, 128, 512}) {
                    b->Args({m, nnzrow, nrhs, batch, 0, 0, kRandom});
                }
            }
        }
    }
}

// Cell L's shape swept over batch and over nrhs 1 vs 2. lanczos pads its operand
// from one column to two purely to defeat a vendor SpMV fallback -- the source says
// so at src/extensions/lanczos.cc:52-54 -- so nrhs = 1 is measured alongside it to
// price that padding. The batch ladder runs below saturation on purpose: quoting a
// ratio there is not allowed, but measuring only at saturation is exactly what has
// concealed batch-only-parallelism defects in this codebase before.
inline void SpmmLanczosSizes(minibench::Benchmark* b) {
    for (int batch : {8, 32, 64, 128, 256, 512, 1024}) {
        for (int nrhs : {1, 2}) {
            for (int pattern : {kBanded, kRandom}) {
                b->Args({1024, 3, nrhs, batch, 0, 0, pattern});
            }
        }
    }
}

inline void SpmmColdCellSizes(minibench::Benchmark* b) { SpmmCellSizes(b); }
inline void SpmmColdLanczosSizes(minibench::Benchmark* b) { SpmmLanczosSizes(b); }

}  // namespace

// CUDA and ROCM only; see the header note on NETLIB.
BATCHLAS_BENCH_CUDA_ALL_TYPES(BM_SPMM_Cells, SpmmCellSizes)
BATCHLAS_BENCH_ROCM_ALL_TYPES(BM_SPMM_Cells, SpmmCellSizes)

BATCHLAS_BENCH_CUDA_ALL_TYPES(BM_SPMM_Grid, SpmmGridSizes)
BATCHLAS_BENCH_ROCM_ALL_TYPES(BM_SPMM_Grid, SpmmGridSizes)

BATCHLAS_BENCH_CUDA_ALL_TYPES(BM_SPMM_Lanczos, SpmmLanczosSizes)
BATCHLAS_BENCH_ROCM_ALL_TYPES(BM_SPMM_Lanczos, SpmmLanczosSizes)

BATCHLAS_BENCH_CUDA_ALL_TYPES(BM_SPMM_ColdCells, SpmmColdCellSizes)
BATCHLAS_BENCH_ROCM_ALL_TYPES(BM_SPMM_ColdCells, SpmmColdCellSizes)

BATCHLAS_BENCH_CUDA_ALL_TYPES(BM_SPMM_ColdLanczos, SpmmColdLanczosSizes)
BATCHLAS_BENCH_ROCM_ALL_TYPES(BM_SPMM_ColdLanczos, SpmmColdLanczosSizes)

MINI_BENCHMARK_MAIN();
