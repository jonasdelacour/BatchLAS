# Performance regression evaluation {#perf_regression}

A CUDA FP32 perf-regression runner and an STEQR accuracy sampler.

## Perf regression (CUDA FP32)

`evaluation/perf_eval.py` runs the cases in `evaluation/perf_cases.json` (`stedc_benchmark` with
the merge variant as its fourth argument, 0 = `Baseline`, 1 = `Fused`; `steqr_benchmark`; the CTA
benchmarks), records a JSON baseline and compares later runs against it. Needs the benchmarks built
and the CUDA backend enabled.

```sh
python3 evaluation/perf_eval.py --record   # write evaluation/baselines/perf_cuda_fp32.json
python3 evaluation/perf_eval.py --check    # fail on > 5% regression or a changed GPU/driver/CUDA/NVHPC
```

| Option | Effect |
|---|---|
| `--tolerance 0.05` | regression tolerance |
| `--build-dir`, `--cases`, `--baseline` | build tree (default `build`), case file (raw positional minibench args), baseline path |
| `--backend CUDA`, `--type float` | minibench filters |
| `--warmup 2`, `--min-time 200`, `--min-iters 5`, `--max-iters 100` | minibench timing |
| `--trace <file>` | Chrome trace of the run |
| `--kernel-trace-dir <dir>` | kernel-level traces (`BATCHLAS_KERNEL_TRACE=1`, `BATCHLAS_KERNEL_TRACE_PATH=<file>` do the same on any `Queue` user) |
| `--allow-env-mismatch` | ignore a toolchain/hardware mismatch |

## Accuracy evaluation (STEQR)

`steqr_accuracy` (benchmarks target) samples random tridiagonal matrices, compares STEQR or
STEQR_CTA eigenvalues with a NETLIB double reference (the host backend must be enabled), and writes
one CSV row per matrix. Complex types map to their real part; pass float or double.

```sh
./build/benchmarks/steqr_accuracy --impl=steqr_cta --backend=CUDA --type=float --n=32 \
    --samples=20000 --batch=256 --log10-cond-min=0 --log10-cond-max=12 \
    --output=output/accuracy/steqr_accuracy.csv
```
