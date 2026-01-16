# CPU Target Detection Implementation

## Overview
Implemented automatic detection and conditional compilation of tests/benchmarks that require CPU SYCL kernel compilation support.

## Problem
When building DPC++ from source for NVIDIA GPUs without CPU backend support, the SYCL compiler cannot compile kernels for CPU devices. Tests and benchmarks that use `Queue("cpu")` or the NETLIB backend fail to compile because no CPU target is specified in the `-fsycl-targets` flags.

## Solution
The build system now intelligently detects whether CPU kernels will actually be compiled by examining the SYCL targets list:

1. **Detection Logic** (CMakeLists.txt:814-845):
   - After GPU architecture detection, checks if any CPU-like targets (native_cpu, spir64*, or anything containing "cpu") are present in `BATCHLAS_SYCL_TARGETS`
   - If no CPU target is in the compilation flags, sets `BATCHLAS_HAS_CPU_TARGET=OFF`
   - Provides status messages explaining the decision

2. **Conditional Compilation**:
   - **Tests** (tests/CMakeLists.txt): Separates tests into base set and CPU-dependent set
   - **Benchmarks** (benchmarks/CMakeLists.txt): Separates benchmarks into base set and CPU-dependent set
   - Only includes CPU-dependent targets when `BATCHLAS_HAS_CPU_TARGET=ON`

3. **Manual Override**:
   - New option: `BATCHLAS_ENABLE_CPU_TESTS` (default: ON)
   - Can be set to OFF to force-disable CPU-dependent tests regardless of detection

## CPU-Dependent Tests
- minibench_cli_tests
- ormqr_cta_tests
- syev_blocked_tests
- syev_cta_tests
- sytrd_cta_tests
- sytrd_blocked_tests

## CPU-Dependent Benchmarks
- geqrf_benchmark
- gemm_benchmark
- trmm_benchmark
- gemv_benchmark
- spmm_benchmark
- trsm_benchmark
- ormqr_benchmark
- ormqr_blocked_benchmark
- ormqr_cta_benchmark
- orgqr_benchmark
- syev_benchmark
- syev_cta_benchmark
- syev_blocked_benchmark
- syevx_benchmark
- lanczos_benchmark
- ortho_benchmark
- steqr_benchmark
- steqr_cta_benchmark
- transpose_benchmark
- permuted_copy_benchmark
- matrix_copy_benchmark
- vector_benchmark
- stehr_benchmark
- stedc_benchmark

## Example Output
```
-- Using SYCL targets: nvidia_gpu_sm_89
-- CPU device detected by sycl-ls, but no CPU target in fsycl-targets
-- CPU kernels will not be compiled - disabling CPU-dependent tests/benchmarks
-- Skipping CPU-dependent tests (no CPU target available)
-- Skipping CPU-dependent benchmarks (no CPU target available)
```

## Benefits
- No manual intervention needed - works automatically
- Based on actual compilation flags, not just device detection
- Clear status messages explain decisions
- Manual override available if needed
- Prevents compilation failures when CPU support is unavailable
