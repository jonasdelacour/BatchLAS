# BatchLAS STEDC Local Notes

These notes capture a verified local profiling workflow and example findings for `build/benchmarks/stedc_benchmark` on this machine.

## Local Platform

- GPUs: 2 x NVIDIA GeForce RTX 4090
- Driver: 590.48.01
- Compute capability: 8.9
- Profilers found at:
  - `/opt/nvidia/hpc_sdk/Linux_x86_64/2026/compilers/bin/nsys`
  - `/opt/nvidia/hpc_sdk/Linux_x86_64/2026/compilers/bin/ncu`

## Verified Benchmark Command

```bash
build/benchmarks/stedc_benchmark \
  --backend=CUDA \
  --type=float \
  --warmup=5 \
  --min_iters=1 \
  --max_iters=1 \
  256 256 16 0 32 1
```

## Verified Nsight Systems Command

```bash
mkdir -p output/profiling
nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --cpuctxsw=none \
  --stats=true \
  --force-overwrite=true \
  -o output/profiling/stedc_nsys \
  build/benchmarks/stedc_benchmark \
  --backend=CUDA \
  --type=float \
  --warmup=5 \
  --min_iters=1 \
  --max_iters=1 \
  256 256 16 0 32 1
```

Generated artifacts:

- `output/profiling/stedc_nsys.nsys-rep`
- `output/profiling/stedc_nsys.sqlite`

### Nsight Systems Observations

Top GPU kernels by total time from `cuda_gpu_kern_sum`:

- `SteqrCTAKernel<float,16,true>`: 35.7%
- `StedcFusedCtaMerge<Backend=CUDA,float,32>`: 30.3%
- `Matrix::Identity` startup kernel: 13.9%
- `PermutedCopyKernel<float,int>`: 5.4%
- `ampere_sgemm_128x128_nn`: 4.9%

Top CUDA API costs from `cuda_api_sum`:

- `cuEventSynchronize`: 43.7%
- `cuMemAllocManaged`: 18.9%
- `cuLibraryLoadData`: 11.9%
- `cuModuleLoadDataEx`: 9.1%

Memory operation time from `cuda_gpu_mem_time_sum`:

- Device-to-device memcpy: 73.5%
- CUDA memset: 23.3%

Interpretation:

- The benchmark is not dominated by one STEDC kernel alone; `SteqrCTAKernel` and `StedcFusedCtaMerge` both matter.
- Startup and orchestration costs are visible, so `ncu` should be targeted at a real STEDC kernel, not the first launch in the process.

## Verified Nsight Compute Command

```bash
ncu \
  --set basic \
  --kernel-name-base demangled \
  --kernel-name regex:StedcFusedCtaMerge \
  --launch-count 1 \
  --target-processes all \
  --force-overwrite \
  --export output/profiling/stedc_ncu_merge \
  build/benchmarks/stedc_benchmark \
  --backend=CUDA \
  --type=float \
  --warmup=5 \
  --min_iters=1 \
  --max_iters=1 \
  256 256 16 0 32 1
```

Generated artifact:

- `output/profiling/stedc_ncu_merge.ncu-rep`

### Nsight Compute Observations

Imported with:

```bash
ncu --import output/profiling/stedc_ncu_merge.ncu-rep --page details --print-summary per-kernel
```

Key metrics for `StedcFusedCtaMerge<1, float, 32>`:

- Duration: 101.50 us
- Compute (SM) Throughput: 8.80%
- Memory Throughput: 8.80%
- DRAM Throughput: 0.14%
- Registers Per Thread: 78
- Block Size: 32
- Grid Size: 256
- Theoretical Occupancy: 50.00%
- Achieved Occupancy: 3.96%
- Waves Per SM: 0.08

Interpretation:

- This kernel is not DRAM-bandwidth limited.
- Achieved occupancy is extremely low relative to theoretical occupancy.
- The likely first optimization direction is exposing more parallel work or changing launch decomposition, not memory tuning.

## Important Caveat

Both local `ncu` runs generated usable `.ncu-rep` files but the benchmark then aborted with a glibc heap-corruption style exit (`code 6`) after profiler disconnect.

Practical guidance:

- Keep the report if it was written.
- Switch to `ncu --import ...` for analysis instead of rerunning immediately.
- Use the narrowest kernel filter possible.