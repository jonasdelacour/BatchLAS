// Occupancy steps on the real device, computed by CUDA's own occupancy model
// (cuda_occupancy.h), not by hand. Prints, for each work-group size, the
// register counts at which blocks/SM drops a step.
#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_occupancy.h>

int main() {
    cudaDeviceProp p{};
    if (cudaGetDeviceProperties(&p, 0) != cudaSuccess) { printf("no device\n"); return 1; }
    printf("device: %s cc %d.%d  SMs=%d  regs/SM=%d  regs/block=%d  smem/SM=%zu  smem/block(optin)=%d  maxThreads/SM=%d  warpSize=%d\n",
           p.name, p.major, p.minor, p.multiProcessorCount, p.regsPerMultiprocessor,
           p.regsPerBlock, (size_t)p.sharedMemPerMultiprocessor, p.sharedMemPerBlockOptin,
           p.maxThreadsPerMultiProcessor, p.warpSize);

    cudaOccDeviceProp op(p);
    const int wgs[] = {32, 64, 128, 256, 512, 1024};
    for (int wi = 0; wi < 6; ++wi) {
        int wg = wgs[wi];
        printf("\n== WG=%d ==\n", wg);
        int prev = -1;
        for (int r = 16; r <= 255; ++r) {
            cudaOccFuncAttributes fa{};
            fa.maxThreadsPerBlock = 1024;
            fa.numRegs = r;
            fa.sharedSizeBytes = 0;
            fa.partitionedGCConfig = PARTITIONED_GC_OFF;
            fa.shmemLimitConfig = FUNC_SHMEM_LIMIT_OPTIN;
            fa.maxDynamicSharedSizeBytes = p.sharedMemPerBlockOptin;
            cudaOccDeviceState st{};
            cudaOccResult res{};
            if (cudaOccMaxActiveBlocksPerMultiprocessor(&res, &op, &fa, &st, wg, 0) != CUDA_OCC_SUCCESS) continue;
            int b = res.activeBlocksPerMultiprocessor;
            if (b != prev) { printf("  regs<=%3d ... blocks/SM=%2d  (warps/SM=%3d)\n", r, b, b * wg / 32); prev = b; }
        }
        printf("  (at regs=255: ");
        cudaOccFuncAttributes fa{}; fa.maxThreadsPerBlock = 1024; fa.numRegs = 255;
        fa.partitionedGCConfig = PARTITIONED_GC_OFF; fa.shmemLimitConfig = FUNC_SHMEM_LIMIT_OPTIN;
        fa.maxDynamicSharedSizeBytes = p.sharedMemPerBlockOptin;
        cudaOccDeviceState st{}; cudaOccResult res{};
        cudaOccMaxActiveBlocksPerMultiprocessor(&res, &op, &fa, &st, wg, 0);
        printf("blocks/SM=%d)\n", res.activeBlocksPerMultiprocessor);
    }

    // Shared-memory co-limit at WG=256.
    printf("\n== shared-memory limit at WG=256, 40 regs ==\n");
    for (int s : {0, 8192, 16384, 24576, 32768, 49152, 65536, 71744, 97280, 101376}) {
        cudaOccFuncAttributes fa{}; fa.maxThreadsPerBlock = 1024; fa.numRegs = 40;
        fa.partitionedGCConfig = PARTITIONED_GC_OFF; fa.shmemLimitConfig = FUNC_SHMEM_LIMIT_OPTIN;
        fa.maxDynamicSharedSizeBytes = p.sharedMemPerBlockOptin;
        cudaOccDeviceState st{}; cudaOccResult res{};
        if (cudaOccMaxActiveBlocksPerMultiprocessor(&res, &op, &fa, &st, 256, s) != CUDA_OCC_SUCCESS) {
            printf("  smem=%6d  DOES NOT FIT\n", s); continue;
        }
        printf("  smem=%6d  blocks/SM=%d\n", s, res.activeBlocksPerMultiprocessor);
    }
    return 0;
}
