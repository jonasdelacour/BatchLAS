---
name: ptx-codegen-comparison
description: 'Compare cuBLASDx and SYCL-generated device code for the same BatchLAS kernel, extract PTX from build artifacts, identify codegen regressions such as spills or missed unrolling, and validate whether PTX changes correlate with benchmark movement.'
argument-hint: 'Kernel family, benchmark command, and the two variants to compare, for example GEMM cuBLASDx vs SYCL on CUDA backend'
user-invocable: true
disable-model-invocation: false
---

# PTX Codegen Comparison

Use this skill when a BatchLAS CUDA-backend kernel is slower in one implementation path than another and you need to determine whether the gap comes from worse generated device code, a launch/configuration mismatch, or a downstream lowering issue.

This skill is designed around the workflow used to compare cuBLASDx GEMM and SYCL GEMM in BatchLAS, but the same process applies to other kernels as long as you can identify matching variants and extract the relevant device artifacts.

## When To Use

- A cuBLASDx kernel and a SYCL kernel implement the same algorithm but show different runtime.
- You need to answer whether SYCL produced worse PTX from equivalent semantics.
- You need to compare cuBLASDx, SYCL, vendor, or other variant-selected BatchLAS backends on the CUDA device path.
- You suspect spills, missed unrolling, conservative shared-memory staging, or other codegen differences.
- You need a disciplined workflow that ties benchmark deltas to extracted PTX rather than guessing from source similarity.

## Rules

1. Keep the benchmark case fixed. Change only the implementation variant under comparison.
2. For BatchLAS benchmarks, include `--warmup=5`.
3. Verify the exact kernel variant before comparing PTX. Do not compare different tile shapes or aligned vs generic paths by accident.
4. Prefer extracting device code from the exact build artifacts used by the benchmark.
5. On this repo's dpcpp-cuda toolchain, configure PTX-inspection builds with `-DBATCHLAS_KEEP_CUDA_INTERMEDIATES=ON -DBATCHLAS_CPU_TARGET=none`.
6. If final linked artifacts do not expose the SYCL kernel to `cuobjdump`, use the preserved SYCL NVPTX LLVM bitcode and lower it to PTX with `/opt/dpcpp-cuda/bin/llc`.
7. Compare structure, not just top-of-file register declarations. PTX from different pipelines can be at different stages of lowering.
8. Treat PTX as evidence, not the whole answer. If needed, follow up with SASS or Nsight Compute.
9. Fix one codegen issue at a time and re-extract PTX after each source change.
10. Re-run the benchmark after each meaningful PTX improvement so you can tell which source change actually mattered.

## Prerequisites

- A reproducible benchmark command for the kernel family under investigation.
- A way to pin the competing variants, for example `BATCHLAS_GEMM_VARIANT=sycl` vs `BATCHLAS_GEMM_VARIANT=native`.
- CUDA device-code tools such as `cuobjdump`, `c++filt`, and the dpcpp-cuda `llc` backend.
- A build configured with CUDA enabled and benchmarks built.

## Procedure

### 1. Reproduce The Performance Gap

Run the same benchmark case for both variants and record the timings.

For GEMM in this repo:

```bash
BATCHLAS_GEMM_VARIANT=sycl /home/jonaslacour/BatchLAS/build/benchmarks/gemm_benchmark 4096 4096 4096 32 --backend=CUDA --type=float --warmup=5
BATCHLAS_GEMM_VARIANT=native /home/jonaslacour/BatchLAS/build/benchmarks/gemm_benchmark 4096 4096 4096 32 --backend=CUDA --type=float --warmup=5
```

Record:

- exact command
- variant-selection environment
- problem size
- average runtime or throughput
- benchmark target name

### 2. Verify The Intended Kernel Variant

Before reading PTX, verify that both runs are hitting the intended variant.

For GEMM in BatchLAS:

- Variant routing lives in `src/backends/gemm_variant.hh`.
- cuBLASDx kernel selection lives in `src/backends/gemm_cublasdx_dispatch.cc`.
- SYCL register-tiled kernel instantiations live in `src/sycl/gemm/register_tiled_common.hh` and `src/sycl/gemm/register_launchers.hh`.

If needed, use the available environment selectors such as `BATCHLAS_GEMM_CUBLASDX_KERNEL` to pin the cuBLASDx path.

### 3. Configure A PTX Inspection Build

Reconfigure the build so device intermediates are preserved:

```bash
cmake -S . -B build \
  -DBATCHLAS_ENABLE_CUDA=ON \
  -DBATCHLAS_BUILD_BENCHMARKS=ON \
  -DBATCHLAS_KEEP_CUDA_INTERMEDIATES=ON \
  -DBATCHLAS_CPU_TARGET=none
```

Then rebuild.

Why `BATCHLAS_CPU_TARGET=none` matters here:

- On the dpcpp-cuda toolchain, the save-temps link path can fail if `native_cpu` remains enabled.
- For CUDA PTX inspection, the CPU target is not needed.

### 4. Extract CUDA PTX

For native CUDA code in this repo, `cuobjdump` on the linked library is usually the most direct path.

Example:

```bash
cuobjdump -ptx build/src/libbatchlas.so | c++filt > output/device_code/native_full.ptx
cuobjdump -sass build/src/libbatchlas.so | c++filt > output/device_code/native_full.sass
```

If the exact kernel is known, isolate the matching `.entry` body and, when useful, the matching SASS `Function :` body.

### 5. Extract SYCL PTX

There are two cases.

If `cuobjdump` on the final linked artifact exposes the SYCL kernel, use that.

If it does not, use the preserved SYCL NVPTX LLVM bitcode generated by the comparison build and lower it with `llc`:

```bash
/opt/dpcpp-cuda/bin/llc -march=nvptx64 -mcpu=sm_89 \
  -o output/device_code/gemm_kernels_sycl.ptx \
  build/src/gemm_kernels.cc-sycl-nvptx64-nvidia-cuda-sm_89.o

c++filt < output/device_code/gemm_kernels_sycl.ptx > output/device_code/gemm_kernels_sycl.demangled.ptx
```

Use the demangled PTX for inspection.

### 6. Compare High-Signal PTX Structure

Do not start by diffing the whole files. First check a small set of structural signals:

- local-memory use: `.local`, `st.local`, `ld.local`
- vectorized global loads: `ld.global.v4`
- shared-memory staging: `st.shared`, `ld.shared`
- synchronization count: `bar.sync`
- math core shape: `fma.rn.f32`, `mad`, `ffma`
- explicit unroll blockers: `.pragma "nounroll"`

Example commands:

```bash
grep -cE 'ld\.local|st\.local|\.local ' sycl_kernel.ptx
grep -c 'ld.global.v4' sycl_kernel.ptx
grep -c 'st.shared' sycl_kernel.ptx
grep -c 'fma.rn.f32' sycl_kernel.ptx
```

### 8. Identify Root-Cause Patterns

Treat these patterns as especially important:

- Accumulator spills: the SYCL kernel writes the computed tile to local memory and reloads it before final stores.
- Weaker staging unrolling: the native kernel expands many more vector load and shared-store operations in straight-line PTX.
- Similar FMA count but worse staging: the compute core matches, so the gap likely comes from load/store scheduling, spills, or final lowering.
- Similar PTX shape but large runtime gap: move to SASS or Nsight Compute because ptxas may still allocate resources differently.

### 9. Fix One Codegen Problem At A Time

When the PTX points to a concrete source-level issue, apply a narrow change and re-run extraction.

For the BatchLAS GEMM investigation, the successful fix pattern was:

- replace runtime-indexed accumulator storage with compile-time indexed helpers
- make the epilogue consume accumulator values through compile-time indexed callbacks
- make aligned fast-path packet-load loops compile-time bounded so the backend can expand them more aggressively

These changes removed local-memory spills and increased the amount of vectorized staging visible in SYCL PTX.

### 10. Re-Run The Benchmark And Correlate

After each meaningful PTX change, rerun the benchmark case and compare against the same pinned baseline.

Only keep changes that improve both:

- PTX structure in the expected direction
- actual runtime on the target benchmark case

## Decision Points

### If The SYCL Kernel Is Not Visible In The Linked Library

- Do not stop at `cuobjdump` failure.
- Inspect the preserved SYCL NVPTX LLVM bitcode.
- Lower that bitcode to PTX with `llc`.

### If PTX Shows A Clear Spill Or Missed Unroll

- Fix that issue first.
- Re-extract PTX immediately.
- Re-run the same benchmark case.

### If PTX Looks Similar But Runtime Is Still Far Behind

- Move to final SASS comparison or Nsight Compute.
- Check registers, occupancy, local-memory traffic, and issue efficiency.

### If The Native And SYCL Kernels Are Not Actually The Same Variant

- Stop and fix the comparison setup first.
- Do not draw conclusions from mismatched tile shapes or aligned vs generic paths.

## Completion Criteria

Consider the investigation complete when all of the following are true:

- The benchmark gap is reproduced with a fixed command and fixed variant selection.
- Matching kernel variants are identified.
- Native and SYCL PTX are both extracted from real build artifacts.
- The key structural PTX differences are named clearly.
- At least one source-level PTX issue is either fixed or ruled out.
- A benchmark rerun shows whether the PTX change mattered in practice.
- If a large gap remains, the next step is clearly identified as SASS or profiler work rather than more blind source edits.

## BatchLAS-Specific Notes

- `BATCHLAS_KEEP_CUDA_INTERMEDIATES=ON` is intentionally an inspection-only option.
- On this repo's dpcpp-cuda environment, `native_cpu` should be disabled for save-temps CUDA comparison builds.
- Terminal cwd can drift during long sessions. Prefer absolute paths for extraction and benchmark commands when reproducibility matters.
- The helper scripts in `scripts/` are part of this workflow and should be preferred over ad hoc shell pipelines when working on GEMM PTX comparisons.

## Example Prompts

- Compare the PTX for BatchLAS native CUDA GEMM and SYCL GEMM on the CUDA backend for the 4096 cube batch-32 float case.
- Figure out whether the SYCL 128x64x32 GEMM kernel is spilling accumulators compared with the native CUDA kernel.
- Use the PTX comparison workflow to diagnose why the CUDA backend SYCL GEMM path is still slower after a refactor.
- Extract native and SYCL PTX for the same BatchLAS kernel and tell me whether the difference is staging, spills, or math core codegen.