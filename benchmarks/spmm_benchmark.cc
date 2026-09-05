// Batched CSR SpMM -- C = alpha * A * op(B) + beta * C -- at the shapes lanczos and
// LOBPCG use. Two arms: WARM reuses long-lived MatrixViews (the honest vendor
// baseline); COLD rebuilds a MatrixView per call to price the per-call host chain.
// Route is chosen per process by BATCHLAS_SPMM_ROUTE=vendor|direct; an unrecognised
// word falls back SILENTLY, so confirm the route from the coverage table.
// evidence: docs/perf/spmm.md#measurement-harness-and-hygiene

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

// Deterministic, never near zero: no accidental structural zeros, and beta = 1
// accumulation stays well scaled.
template <typename T>
T spmm_value(std::uint64_t h) {
    const auto unit = [](std::uint64_t x) {
        x ^= x >> 33;
        x *= 0xff51afd7ed558ccdULL;
        x ^= x >> 33;
        x *= 0xc4ceb9fe1a85ec53ULL;
        x ^= x >> 33;
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

// Banded CSR, EXACTLY nnz_per_row sorted entries per row: boundary rows reuse the
// first/last band rather than being short, so nnz stays m*r and per-row work uniform.
// Host-written straight into the USM buffers -- safe only because nothing is enqueued yet.
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

// RandomSparseHermitian targets a density, so nnz per row lands within one of the request.
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

// Ideal traffic for one item, one pass over A: values plus 32-bit column indices, the row
// offsets, and m*nrhs*s each for the B gather, the C write, and the C read when beta != 0.
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
    // state.range() returns 0 for an index no sizer supplied, so arg7 -- which no sizer
    // passes -- stays NoTrans; A is square, so transA changes no extent, only which body runs.
    const auto transA = state.range(7) == 0   ? Transpose::NoTrans
                      : state.range(7) == 1   ? Transpose::Trans
                                              : Transpose::ConjTrans;
    const T alpha = T(1);

    auto q = std::make_shared<Queue>(Device(Bk == Backend::NETLIB ? "cpu" : "gpu"), Bk);

    auto Amat = std::make_shared<Matrix<T, MatrixFormat::CSR>>(
        make_csr<T>(m, nnz_per_row, batch, pattern));
    // op(B) must be k x nrhs with k = m, so the stored shape flips with transB.
    auto Bmat = std::make_shared<Matrix<T>>(
        transB == Transpose::NoTrans ? Matrix<T>::Random(m, nrhs, false, batch)
                                     : Matrix<T>::Random(nrhs, m, false, batch));
    auto Cmat = std::make_shared<Matrix<T>>(Matrix<T>::Zeros(m, nrhs, batch));

    // The warm arm reuses these, so the lazy cuSPARSE descriptors are built once below.
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
            // The Matrix overload builds a fresh MatrixView -- and so a fresh descriptor
            // triple that is never destroyed -- per call. Never compare a native kernel to it.
            spmm<Bk, T, MatrixFormat::CSR>(*q, *Amat, *Bmat, *Cmat, alpha, beta,
                                           transA, transB, ws->to_span());
        }
    };

    bench::ManagedInputs managed(q);
    managed.prepare(*Amat).prepare(*Bmat).prepare(*Cmat);

    state.SetPrepare([prep = managed.make_prepare_once(), kernel_once, q]() mutable {
        prep();
        // Untimed: JITs the kernel, builds the descriptors, faults the USM pages in.
        kernel_once();
        q->wait();

        // Untimed wall-clock warm-up, not a call count: the clock ramp is a per-PROCESS cost.
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
    // No SetTimedKernelMs on purpose: the event timer costs ~15x the cheapest cell's roof.

    // The real per-item non-zero count; Matrix::nnz() with no argument is a capacity.
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

    // The no-op detector: cusparseSpMM's status is never checked, so a rejected argument
    // combination looks like a very fast kernel. chk == 0 on a beta = 0 row means no work.
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

// (m, nnz/row, nrhs, batch) for the three cells carried through docs/perf/spmm.md.
constexpr int kCells[3][4] = {
    {1024,  3,  2, 512},   // L -- lanczos
    {1024, 16, 12, 512},   // M -- LOBPCG
    {2048, 16, 25, 128},   // S -- LOBPCG
};

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

// The largest row allocates ~4.7 GB for complex<double>: run one type per process.
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

// Cell L over batch and over nrhs 1 vs 2: lanczos pads its operand to two columns to
// defeat a vendor SpMV fallback. The batch ladder runs below saturation on purpose.
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

// CUDA and ROCM only: the NETLIB spmm throws on any transpose and is not compiled
// at all in a vendor-free build, so registering it would take the process down.
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
