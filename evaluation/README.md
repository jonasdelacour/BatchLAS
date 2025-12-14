# Evaluation

This folder contains lightweight evaluation tooling for BatchLAS.

## Perf regression (CUDA FP32)

The perf runner executes two existing minibench benchmarks:
- `stedc_benchmark` (both recursive and `stedc_flat` variants via the `flat` arg)
- `steqr_benchmark`

It records a JSON baseline and can compare later runs against that baseline.

### Prereqs

- Build the benchmarks (so `build/benchmarks/stedc_benchmark` and `build/benchmarks/steqr_benchmark` exist)
- CUDA backend enabled in your build

### Record a baseline

Run:

- `python3 evaluation/perf_eval.py --record`

This writes the default baseline to:

- `evaluation/baselines/perf_cuda_fp32.json`

### Check for regressions

Run:

- `python3 evaluation/perf_eval.py --check`

By default it fails if any metric regresses by more than 5%.

### Common options

- Change tolerance: `--tolerance 0.10` (10%)
- Use a different build dir: `--build-dir /path/to/build`
- Override cases: `--cases evaluation/perf_cases.json`
- Emit a Chrome trace: `--trace evaluation/trace.json`

### Kernel-level traces (SYCL event profiling)

BatchLAS can emit a kernel/command-level Chrome trace using SYCL event profiling.

Options:

- Via perf runner (recommended):
	- `python3 evaluation/perf_eval.py --check --kernel-trace-dir evaluation/trace/kernels`

- Or manually via env vars when running any executable that uses `Queue`:
	- `BATCHLAS_KERNEL_TRACE=1`
	- `BATCHLAS_KERNEL_TRACE_PATH=path/to/kernels.trace.json`

### Default cases

The default cases are defined in:

- `evaluation/perf_cases.json`

Each case provides the raw positional minibench args passed to the executable.
