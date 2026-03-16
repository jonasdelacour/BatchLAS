# BatchLAS Profiling Workflow

## Quick Start

1. Resolve the target:
   `python3 .github/skills/batchlas-function-profiling/scripts/find_batchlas_profile_targets.py <symbol-or-path>`
2. Confirm the benchmark exists:
   `ls build/benchmarks`
3. Print a command instead of hand-writing one:
   `python3 .github/skills/batchlas-function-profiling/scripts/emit_profile_commands.py --benchmark <target> --tool trace -- <args...>`

## Profiling Surfaces

### BatchLAS kernel trace

Use this first when the repo already has `BATCHLAS_KERNEL_TRACE_SCOPE` labels around the function or phase you care about.

- Enable trace:
  `BATCHLAS_KERNEL_TRACE=1`
- Optional output path:
  `BATCHLAS_KERNEL_TRACE_PATH=output/profiling/<name>.trace.json`
- Compatible fallback variables already supported by the repo:
  `BATCHLAS_TRACE_KERNELS=1`
  `BATCHLAS_TRACE_PATH=...`

Relevant code paths:

- `src/util/kernel-trace.hh`
- `src/queue.hh`

Practical note:

- Kernel trace implies queue profiling, so a `Queue` created by the benchmark can emit per-submit timing data into a Chrome trace JSON file.
- This is the best view when you need BatchLAS phase names such as `ormqr_blocked.pack_v_panel` or `sytrd_blocked.update_vw_gemm_vw`.

### Nsight Systems (`nsys`)

Use this when you need a time-sorted summary of GPU kernels, API overhead, memcpy cost, or synchronization.

Recommended capture shape:

```bash
nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --cpuctxsw=none \
  --stats=true \
  --force-overwrite=true \
  -o output/profiling/<name> \
  build/benchmarks/<target> --backend=CUDA --type=float --warmup=5 --min_iters=1 --max_iters=1 --min_time=0 <args...>
```

Useful replay command:

```bash
nsys stats \
  --report cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum \
  output/profiling/<name>.nsys-rep
```

This repo already contains one verified example wrapper:

- `scripts/run_gemm_steady_profile.sh`

### Nsight Compute (`ncu`)

Use this only after `nsys` or BatchLAS trace has identified the hotspot kernel.

Start with:

```bash
ncu \
  --set basic \
  --kernel-name-base demangled \
  --kernel-name regex:<hot-kernel> \
  --launch-count 1 \
  --target-processes all \
  --force-overwrite \
  --export output/profiling/<name> \
  build/benchmarks/<target> --backend=CUDA --type=float --warmup=5 --min_iters=1 --max_iters=1 --min_time=0 <args...>
```

Then inspect the saved report:

```bash
ncu --import output/profiling/<name>.ncu-rep --page details --print-summary per-kernel
```

For deeper diagnosis, the repo already uses this section set in `scripts/run_gemm_steady_profile.sh`:

- `SchedulerStats`
- `WarpStateStats`
- `SourceCounters`
- `MemoryWorkloadAnalysis`
- `ComputeWorkloadAnalysis`

### `nvprof`

Use this only when you are on an older CUDA toolkit where `nvprof` still works. On the current machine, `nvprof --help` exits with an error saying CUDA 13 no longer supports it, so `nsys` and `ncu` are the active path.

If `nvprof` is still available on the target machine, summary mode is the closest direct replacement for a quick time-sorted kernel and API breakdown:

```bash
nvprof --print-gpu-summary --print-api-summary build/benchmarks/<target> ...
```

For trace-style output:

```bash
nvprof --print-gpu-trace --print-api-trace build/benchmarks/<target> ...
```

## Repository-Specific Notes

- Benchmark executables live in `build/benchmarks/`.
- Benchmark CLI options are shared and include `--warmup`, `--min_iters`, `--max_iters`, `--min_time`, `--backend`, `--type`, and `--name`.
- `evaluation/perf_eval.py` supports trace-aware regression cases for `stedc`, `steqr`, `sytrd_cta`, `ormqr_cta`, and `syev_cta`.
- Existing variant selectors matter for reproducibility. For GEMM, common selectors include `BATCHLAS_GEMM_VARIANT` and `BATCHLAS_GEMM_SYCL_KERNEL`.
- Search for `BATCHLAS_KERNEL_TRACE_SCOPE` in the implementation before assuming you need new instrumentation.
