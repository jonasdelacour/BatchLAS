// Head-to-head: BatchLAS's native small-matrix SVD against cuSOLVER's
// gesvdjBatched, at identical shapes, in one process.
//
// Why a dedicated target rather than extending gesvd_cta_benchmark:
//  - The batch sweep here starts where the GPU saturates. gesvd_cta_benchmark
//    tops out at batch=64, which on a 4090 at n=32 measures launch overhead, not
//    the algorithm (GESVD_PLAN.md defect D).
//  - `--name` is a substring filter, so BM_GESVD_CTA in the existing target would
//    also select anything named BM_GESVD_CTA_*. Keeping the comparison in its own
//    binary makes the filters unambiguous.
//
// The vendor path is called through backend::gesvd_vendor directly rather than
// through gesvd(): the default provider order is CTA -> Blocked -> ... -> Vendor,
// so dispatch never reaches cuSOLVER for real GPU input.
//
// Fairness notes, so the numbers are not over-read:
//  - cuSOLVER returns V; the BatchLAS contract is V^H. The vendor time therefore
//    includes an n x n x batch conj-transpose when jobvh == All. That is the cost
//    of conforming to this API, not a cuSOLVER cost. Compare the jobvh=None rows
//    to see gesvdjBatched unencumbered.
//  - gesvdjBatched is capped at 32x32, so this benchmark stops at n=32.

#include <util/minibench.hh>

#include <blas/enums.hh>
#include <blas/extensions.hh>
#include <blas/functions.hh>

#include "bench_utils.hh"

#include <cstddef>
#include <cstdint>
#include <memory>

using namespace batchlas;

namespace {

template <typename Benchmark>
inline void GesvdVendorBenchSizes(Benchmark* b) {
    // Args: n, batch, jobu_all, jobvh_all
    // Batch starts at 1024: below that the 4090 is not saturated and the ratio
    // between two implementations is an overhead ratio, not an algorithm ratio.
    for (int n : {8, 16, 32}) {
        for (int bs : {1024, 4096, 16384}) {
            for (int jobu : {0, 1}) {
                for (int jobvh : {0, 1}) {
                    b->Args({n, bs, jobu, jobvh});
                }
            }
        }
    }
}

inline SvdVectors parse_job(int v) {
    return (v == 0) ? SvdVectors::None : SvdVectors::All;
}

} // namespace

// cuSOLVER gesvdjBatched.
template <typename T, Backend B>
static void BM_GESVD_CUSOLVER_JACOBI(minibench::State& state) {
    const size_t n = state.range(0);
    const size_t batch = state.range(1);
    const SvdVectors jobu = parse_job(static_cast<int>(state.range(2)));
    const SvdVectors jobvh = parse_job(static_cast<int>(state.range(3)));

    // in-order queue: the native providers require one, so both sides of the
    // comparison are given the same queue semantics.
    auto q = std::make_shared<Queue>(Device("gpu"), B, true);

    auto A = Matrix<T>::Random(n, n, /*hermitian=*/false, batch);
    Matrix<T> U(n, n, batch);
    Matrix<T> Vh(n, n, batch);
    UnifiedVector<typename base_type<T>::type> s(n * batch);

    const size_t ws_size = backend::gesvd_vendor_buffer_size<B, T>(
        *q, A.view(), s.to_span(), U.view(), Vh.view(), jobu, jobvh);
    UnifiedVector<std::byte> workspace(ws_size);

    state.SetKernel(q,
                    bench::pristine(A),
                    std::move(s),
                    std::move(U),
                    std::move(Vh),
                    jobu,
                    jobvh,
                    std::move(workspace),
                    [](Queue& q, auto&& A_, auto&& s_, auto&& U_, auto&& Vh_,
                       auto&& jobu_, auto&& jobvh_, auto&& ws_) {
                        backend::gesvd_vendor<B, T>(q, A_, s_, U_, Vh_, jobu_, jobvh_, ws_);
                    });

    state.SetMetric("Matrices/s", static_cast<double>(batch), minibench::Rate);
    state.SetMetric("Time (µs) / matrix", (1.0 / static_cast<double>(batch)) * 1e6, minibench::Reciprocal);
}

// BatchLAS native CTA path (gebrd_cta -> normal-equation tridiagonal -> steqr_cta).
template <typename T, Backend B>
static void BM_GESVD_BATCHLAS_CTA(minibench::State& state) {
    const size_t n = state.range(0);
    const size_t batch = state.range(1);
    const SvdVectors jobu = parse_job(static_cast<int>(state.range(2)));
    const SvdVectors jobvh = parse_job(static_cast<int>(state.range(3)));

    auto q = std::make_shared<Queue>(Device("gpu"), B, true);

    auto A = Matrix<T>::Random(n, n, /*hermitian=*/false, batch);
    Matrix<T> U(n, n, batch);
    Matrix<T> Vh(n, n, batch);
    UnifiedVector<typename base_type<T>::type> s(n * batch);

    const size_t ws_size = gesvd_cta_buffer_size<B, T>(
        *q, A.view(), s.to_span(), U.view(), Vh.view(), jobu, jobvh);
    UnifiedVector<std::byte> workspace(ws_size);

    state.SetKernel(q,
                    bench::pristine(A),
                    std::move(s),
                    std::move(U),
                    std::move(Vh),
                    jobu,
                    jobvh,
                    std::move(workspace),
                    [](Queue& q, auto&& A_, auto&& s_, auto&& U_, auto&& Vh_,
                       auto&& jobu_, auto&& jobvh_, auto&& ws_) {
                        gesvd_cta<B, T>(q, A_, s_, U_, Vh_, jobu_, jobvh_, ws_);
                    });

    state.SetMetric("Matrices/s", static_cast<double>(batch), minibench::Rate);
    state.SetMetric("Time (µs) / matrix", (1.0 / static_cast<double>(batch)) * 1e6, minibench::Reciprocal);
}

BATCHLAS_BENCH_CUDA(BM_GESVD_CUSOLVER_JACOBI, GesvdVendorBenchSizes)
BATCHLAS_BENCH_CUDA(BM_GESVD_BATCHLAS_CTA, GesvdVendorBenchSizes)

MINI_BENCHMARK_MAIN();
