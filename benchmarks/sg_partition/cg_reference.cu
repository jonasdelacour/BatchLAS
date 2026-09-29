// CUDA cooperative_groups::tiled_partition<P> reference for microbench.cc:
// the same per-chunk work, for SASS and timing comparison.
//
//   nvcc -arch=sm_89 -O3 -std=c++17 benchmarks/sg_partition/cg_reference.cu -o sgp_cg_reference
//   CUDA_VISIBLE_DEVICES=1 ./sgp_cg_reference [log2_lanes=24] [iters=64] [reps=30]

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace cg = cooperative_groups;

enum Op : int { kPerm = 0, kBcast = 1, kShift = 2, kVote = 3, kReduce = 4, kReduceDiv = 5, kSelectExit = 6 };
static const char* op_name(int o) {
    static const char* n[] = {"xor-perm", "select", "shift-left", "any", "reduce", "reduce-div", "select-exit"};
    return n[o];
}

template <int O, unsigned P, typename T>
__global__ void kernel(T* data, const int* trips, int iters) {
    const size_t gid = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    const unsigned lane = threadIdx.x & 31u;
    T v = data[gid];
    if (O == kSelectExit && P < 32 && ((lane / P) & 1u)) return;
    auto tile = cg::tiled_partition<P>(cg::this_thread_block());
    const unsigned lid = tile.thread_rank();
    int n = O == kReduceDiv ? trips[gid / P] : iters;
    const T half = T(0.5), inv = T(1) / T(P);
#pragma unroll 1
    for (int k = 0; k < n; ++k) {
        if (O == kPerm) {
            T r = v;
#pragma unroll
            for (unsigned m = 1; m < P; m <<= 1) r += tile.shfl_xor(r, m);
            v = r * inv + T(1);
        } else if (O == kReduce || O == kReduceDiv) {
            v = cg::reduce(tile, v, cg::plus<T>()) * inv + T(1);
        } else if (O == kBcast || O == kSelectExit) {
            v = tile.shfl(v, ((unsigned)k + lid) & (P - 1u)) * half + v * half;
        } else if (O == kShift) {
            const T s = tile.shfl_down(v, 1u);
            v = (lid + 1u < P ? s : v) * half + T(1);
        } else if (O == kVote) {
            v = tile.any(v > T(3) + T(lid)) ? v * half : v + T(1);
        }
    }
    data[gid] = v;
}

template <int O, unsigned P, typename T>
static double run(size_t n, int iters, int reps, const int* trips) {
    T* d;
    cudaMalloc(&d, n * sizeof(T));
    std::vector<T> init(n);
    for (size_t i = 0; i < n; ++i) init[i] = T((i * 2654435761u) % 1000) / T(250);
    cudaMemcpy(d, init.data(), n * sizeof(T), cudaMemcpyHostToDevice);
    const unsigned blocks = (unsigned)(n / 256);
    kernel<O, P, T><<<blocks, 256>>>(d, trips, iters);
    cudaDeviceSynchronize();
    if (reps <= 0) {
        cudaFree(d);
        return 0.0;
    }
    for (int i = 0; i < 3; ++i) kernel<O, P, T><<<blocks, 256>>>(d, trips, iters);
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    std::vector<double> t;
    for (int i = 0; i < reps; ++i) {
        cudaEventRecord(a);
        kernel<O, P, T><<<blocks, 256>>>(d, trips, iters);
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float ms;
        cudaEventElapsedTime(&ms, a, b);
        t.push_back(ms);
    }
    std::sort(t.begin(), t.end());
    cudaFree(d);
    return t[t.size() / 2];
}

template <unsigned P, typename T>
static void run_p(size_t n, int iters, int reps, const int* trips, const char* tn) {
    double ms[7];
    ms[0] = run<kPerm, P, T>(n, iters, reps, trips);
    ms[1] = run<kBcast, P, T>(n, iters, reps, trips);
    ms[2] = run<kShift, P, T>(n, iters, reps, trips);
    ms[3] = run<kVote, P, T>(n, iters, reps, trips);
    ms[4] = run<kReduce, P, T>(n, iters, reps, trips);
    ms[5] = run<kReduceDiv, P, T>(n, iters, reps, trips);
    ms[6] = run<kSelectExit, P, T>(n, iters, reps, trips);
    for (int o = 0; o < 7; ++o) std::printf("| %2u | %-6s | %-12s | %8.4f |\n", P, tn, op_name(o), ms[o]);
}

int main(int argc, char** argv) {
    const int lg = argc > 1 ? std::atoi(argv[1]) : 24;
    const int iters = argc > 2 ? std::atoi(argv[2]) : 64;
    const int reps = argc > 3 ? std::atoi(argv[3]) : 30;
    const size_t n = size_t(1) << lg;
    std::vector<int> th(n);
    for (size_t i = 0; i < n; ++i) th[i] = iters - 8 * int(i % 4);
    int* trips;
    cudaMalloc(&trips, n * sizeof(int));
    cudaMemcpy(trips, th.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    std::printf("# CUDA CG reference lanes=2^%d iters=%d reps=%d (median ms)\n| P | type | op | cg |\n|---|---|---|---|\n",
                lg, iters, reps);
    run_p<4, float>(n, iters, reps, trips, "float");
    run_p<8, float>(n, iters, reps, trips, "float");
    run_p<16, float>(n, iters, reps, trips, "float");
    run_p<32, float>(n, iters, reps, trips, "float");
    run_p<4, double>(n, iters, reps, trips, "double");
    run_p<8, double>(n, iters, reps, trips, "double");
    run_p<16, double>(n, iters, reps, trips, "double");
    run_p<32, double>(n, iters, reps, trips, "double");
    cudaFree(trips);
    return 0;
}
