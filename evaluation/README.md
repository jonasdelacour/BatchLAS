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

When recording, the runner now snapshots CUDA/NVIDIA toolchain metadata (GPU model,
driver, CUDA toolkit, and NVHPC version when available) into `meta.environment`.

### Check for regressions

Run:

- `python3 evaluation/perf_eval.py --check`

By default it fails if any metric regresses by more than 5%.
It also fails by default if the current CUDA/NVHPC environment does not match
the baseline metadata, to keep comparisons fair after toolkit upgrades.

### Common options

- Change tolerance: `--tolerance 0.10` (10%)
- Use a different build dir: `--build-dir /path/to/build`
- Override cases: `--cases evaluation/perf_cases.json`
- Emit a Chrome trace: `--trace evaluation/trace.json`
- Temporarily ignore toolchain/hardware mismatch: `--allow-env-mismatch`

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

## Accuracy evaluation (STEQR)

The accuracy sampler generates random tridiagonal matrices, compares STEQR/STEQR_CTA
eigenvalues against a NETLIB double reference, and writes one CSV row per matrix.
You can then plot a heatmap of log10(relative error) vs log10(condition number).

### Build the sampler

- Ensure the benchmarks are built (the new target is `steqr_accuracy`)

### Generate a dataset

- Example (CUDA, float, STEQR_CTA):
	- `./build/benchmarks/steqr_accuracy --impl=steqr_cta --backend=CUDA --type=float --n=32 --samples=20000 --batch=256 --log10-cond-min=0 --log10-cond-max=12 --output=output/accuracy/steqr_accuracy.csv`

### Plot the heatmap

- `python3 plotting/steqr_accuracy_heatmap.py --csv output/accuracy/steqr_accuracy.csv --output output/plots/steqr_accuracy_heatmap.png`

### Notes

- The sampler always uses NETLIB double for the reference solve, so the host backend must be enabled.
- Complex types map to their real component for STEQR accuracy (use float/double to be explicit).
