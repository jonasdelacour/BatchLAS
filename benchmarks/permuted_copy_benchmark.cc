#include <util/minibench.hh>
#include <blas/linalg.hh>
#include <internal/sort.hh>
#include "bench_utils.hh"
#include <batchlas/backend_config.h>
#include <cstdint>

using namespace batchlas;

// Batched permuted copy benchmark
// Measures bandwidth of permuted_copy across batched matrices.
template <typename T, Backend B>
static void BM_PERMUTED_COPY(minibench::State& state) {
    const int rows = state.range(0);
    const int cols = state.range(1);
    const int batch = state.range(2);
    const int n_blocks = state.range(3) > 0 ? state.range(3) : -1;
    const int block_size = state.range(4) > 0 ? state.range(4) : -1;
    //std::cout << "Running BM_PERMUTED_COPY with rows=" << rows << ", cols=" << cols << ", batch=" << batch << ", n_blocks=" << n_blocks << ", block_size=" << block_size << std::endl;
    auto q = std::make_shared<Queue>(B == Backend::NETLIB ? "cpu" : "gpu");

    auto src = Matrix<T>::Random(rows, cols, false, batch);
    auto dst = Matrix<T>::Zeros(rows, cols, batch);

    Vector<int32_t> permutation = Vector<int32_t>::zeros(cols, batch);
    static std::mt19937 rng(12345);
    std::vector<int32_t> indices(cols);
    for (int c = 0; c < cols; ++c) {
        indices[c] = c;
    }
    for (int b = 0; b < batch; ++b) {
        std::shuffle(indices.begin(), indices.end(), rng);
        for (int c = 0; c < cols; ++c) {
            permutation(c, b) = indices[c];
        }
    }
    
    state.SetKernel([=]() {
        permuted_copy(*q, src, dst, permutation, PermutedCopyParams{{n_blocks*block_size, block_size}});
    });
    state.SetBatchEndWait(q);

    const double bytes_moved = static_cast<double>(rows) * cols * batch * sizeof(T) * 2.0;
    state.SetMetric("GB/s", bytes_moved * 1e-9, minibench::Rate);
    state.SetMetric("Time (µs) / Batch", (1.0 / batch) * 1e6, minibench::Reciprocal);
}

BATCHLAS_REGISTER_BENCHMARK(BM_PERMUTED_COPY, SquareBatchSizes);

MINI_BENCHMARK_MAIN();
