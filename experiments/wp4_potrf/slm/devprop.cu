// The hardware side of the occupancy question: per-SM shared pool, per-block cap,
// warp and block limits. SYCL exposes none of these, and the spec's occupancy
// table (spec:284) asserts shared_per_SM ~= 102400 without a source.
#include <cstdio>
#include <cuda_runtime.h>

int main() {
    int n = 0; cudaGetDeviceCount(&n);
    for (int i = 0; i < n; ++i) {
        cudaDeviceProp p{}; cudaGetDeviceProperties(&p, i);
        std::printf("device %d: %s  sm_%d%d\n", i, p.name, p.major, p.minor);
        std::printf("  multiProcessorCount                 = %d\n", p.multiProcessorCount);
        std::printf("  sharedMemPerMultiprocessor          = %zu\n", (size_t)p.sharedMemPerMultiprocessor);
        std::printf("  sharedMemPerBlock                   = %zu\n", (size_t)p.sharedMemPerBlock);
        std::printf("  sharedMemPerBlockOptin              = %zu\n", (size_t)p.sharedMemPerBlockOptin);
        std::printf("  reservedSharedMemPerBlock           = %zu\n", (size_t)p.reservedSharedMemPerBlock);
        std::printf("  regsPerBlock                        = %d\n", p.regsPerBlock);
        std::printf("  regsPerMultiprocessor               = %d\n", p.regsPerMultiprocessor);
        std::printf("  maxThreadsPerMultiProcessor         = %d\n", p.maxThreadsPerMultiProcessor);
        std::printf("  maxThreadsPerBlock                  = %d\n", p.maxThreadsPerBlock);
        std::printf("  maxBlocksPerMultiProcessor          = %d\n", p.maxBlocksPerMultiProcessor);
        std::printf("  warpSize                            = %d\n", p.warpSize);
    }
    return 0;
}
