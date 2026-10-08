# Performance regression evaluation {#perf_regression}

Lightweight evaluation tools: a CUDA FP32 perf-regression runner and an STEQR accuracy sampler.

## Perf regression (CUDA FP32)

`evaluation/perf_eval.py` runs `stedc_benchmark` (recursive and `stedc_flat` via the `flat` arg)
and `steqr_benchmark`, records a JSON baseline, and compares later runs against it.

**Prerequisites:** the benchmarks are built (`build/benchmarks/stedc_benchmark` and
`build/benchmarks/steqr_benchmark` exist) and the CUDA backend is enabled.

| Task | Command |
|---|---|
| Record baseline | `python3 evaluation/perf_eval.py --record` |
| Check for regressions | `python3 evaluation/perf_eval.py --check` |

- The baseline is written to `evaluation/baselines/perf_cuda_fp32.json`. It snapshots the GPU
  model, driver, CUDA toolkit and NVHPC version into `meta.environment`.
- `--check` fails if any metric regresses by more than 5 %, or if the CUDA/NVHPC environment
  differs from the baseline metadata.

### Options

| Option | Effect |
|---|---|
| `--tolerance 0.10` | Regression tolerance (default 5 %) |
| `--build-dir <path>` | Build directory to use |
| `--cases evaluation/perf_cases.json` | Override the cases (raw positional minibench args) |
| `--trace evaluation/trace.json` | Emit a Chrome trace |
| `--allow-env-mismatch` | Ignore toolchain/hardware mismatch |
| `--kernel-trace-dir evaluation/trace/kernels` | Kernel-level traces (see below) |

### Kernel-level traces

SYCL event profiling produces a kernel/command-level Chrome trace. Use the runner's option, or set
the variables on any executable that uses `Queue`:

```sh
BATCHLAS_KERNEL_TRACE=1
BATCHLAS_KERNEL_TRACE_PATH=path/to/kernels.trace.json
```

Default cases are in `evaluation/perf_cases.json`.

## Accuracy evaluation (STEQR)

`steqr_accuracy` samples random tridiagonal matrices, compares STEQR and STEQR_CTA eigenvalues
against a NETLIB double reference, and writes one CSV row per matrix. Plot a heatmap of
log10(relative error) against log10(condition number).

Build it with the benchmarks (target `steqr_accuracy`). Example, CUDA float, STEQR_CTA:

```sh
./build/benchmarks/steqr_accuracy --impl=steqr_cta --backend=CUDA --type=float --n=32 \
    --samples=20000 --batch=256 --log10-cond-min=0 --log10-cond-max=12 \
    --output=output/accuracy/steqr_accuracy.csv
```

- The reference solve always uses NETLIB double, so the host backend must be enabled.
- Complex types map to their real component for STEQR accuracy; use float or double explicitly.
