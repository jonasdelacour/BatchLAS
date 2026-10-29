---
name: batchlas-function-profiling
description: Profile BatchLAS functions, benchmark targets, and GPU kernels using the repository's existing benchmarks, SYCL event traces, and NVIDIA CLI profilers. Use when Codex needs to map a source file or function to the right BatchLAS benchmark, capture phase-level traces with BATCHLAS kernel tracing, print time-sorted GPU and API breakdowns with Nsight Systems, inspect hotspot kernels with Nsight Compute, or determine whether older nvprof-based workflows are still usable on the current machine.
---

# BatchLAS Function Profiling

Map the function to an existing benchmark, take one clean baseline outside any profiler, then use the
narrowest profiling surface that answers the question. Reading `ncu` output and kernel tuning: the
`nvidia-kernel-profiling` skill.

## Workflow

1. **Resolve the target.** `python3 .github/skills/batchlas-function-profiling/scripts/find_batchlas_profile_targets.py <symbol-or-path>`
   maps a file, function, benchmark family or trace label to benchmark executables and
   `BATCHLAS_KERNEL_TRACE_SCOPE` labels. Confirm the binary exists in `build/benchmarks/`.
2. **Prefer existing harnesses.** `build/benchmarks/*`; for `stedc`, `steqr`, `sytrd_cta`, `ormqr_cta`
   and `syev_cta` regression cases, `evaluation/perf_eval.py`. Add benchmark code only when nothing
   existing isolates the target.
3. **Baseline.** Run once without a profiler; record the command, backend, type, args and route pins
   (`BATCHLAS_<OP>_ROUTE`, e.g. `BATCHLAS_GEMM_ROUTE=native|vendor|reg:m=128:n=32:k=32:u=1`).
4. **Pick the lightest surface.**
   - Kernel trace for BatchLAS phase names (`ormqr_blocked.pack_v_panel`,
     `sytrd_blocked.update_vw_gemm_vw`): `BATCHLAS_KERNEL_TRACE=1`
     `BATCHLAS_KERNEL_TRACE_PATH=output/profiling/<name>.trace.json` writes a Chrome trace of every
     queue submit (`src/util/kernel-trace.hh`). It does not see vendor (cuBLAS) kernels; `nsys` does.
     Grep for `BATCHLAS_KERNEL_TRACE_SCOPE` before adding instrumentation.
   - `nsys` for a time-sorted kernel/API/memcpy breakdown or to separate orchestration from kernels.
   - `ncu` only after the hot kernel is known, filtered to that kernel.
   - `nvprof` only on pre-CUDA-13 toolkits; CUDA 13 rejects it.
5. **Generate commands, do not hand-assemble them.** `emit_profile_commands.py` prints repo-correct
   trace, `nsys` (+ `nsys stats`), `ncu` (`--ncu-mode basic|detailed`) or `nvprof` commands with short
   runs (`--warmup=5 --min_iters=1 --max_iters=1 --min_time=0`); benchmark args go after `--`, extra
   env with repeatable `--env K=V`. `scripts/run_gemm_steady_profile.sh [build_dir] [out_dir]` is a
   worked nsys+ncu wrapper for `gemm_steady_benchmark`.
6. **Report evidence.** Benchmark, exact command, hottest kernels or trace scopes, first plausible
   bottleneck, and the metric or trace observation behind any recommended change. If profiler output
   conflicts with the baseline, say so before recommending anything.

```bash
S=.github/skills/batchlas-function-profiling/scripts
python3 $S/find_batchlas_profile_targets.py src/ops/ormqr/kernels/ormqr_blocked.cc
python3 $S/emit_profile_commands.py --benchmark gemm_steady_benchmark --tool trace \
  --output-stem gemm_steady_512 -- 512 512 512 512
python3 $S/emit_profile_commands.py --benchmark gemm_steady_benchmark --tool all \
  --kernel-regex GemmRegisterTiledKernel --env BATCHLAS_GEMM_ROUTE=reg:m=128:n=32:k=32:u=1 -- 512 512 512 512
```

Done when the benchmark or harness is identified, the profiling command is reproducible, and the
bottleneck is summarised with evidence from trace scopes, `nsys` summaries or `ncu` metrics.
