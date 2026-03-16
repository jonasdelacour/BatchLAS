---
name: batchlas-function-profiling
description: Profile BatchLAS functions, benchmark targets, and GPU kernels using the repository's existing benchmarks, SYCL event traces, and NVIDIA CLI profilers. Use when Codex needs to map a source file or function to the right BatchLAS benchmark, capture phase-level traces with BATCHLAS kernel tracing, print time-sorted GPU and API breakdowns with Nsight Systems, inspect hotspot kernels with Nsight Compute, or determine whether older nvprof-based workflows are still usable on the current machine.
---

# BatchLAS Function Profiling

Use existing benchmark targets before writing ad hoc drivers. Resolve the function or file to a benchmark, run one clean baseline outside any profiler, then choose the narrowest profiling surface that answers the question.

## Workflow

1. Resolve the target benchmark first.
Use `scripts/find_batchlas_profile_targets.py <symbol-or-path>` to map a source file, function name, or benchmark family to likely benchmark executables and related `BATCHLAS_KERNEL_TRACE_SCOPE` labels.

2. Prefer repository benchmarks and existing evaluation harnesses.
Most profiling work should start from `build/benchmarks/*`. For regression-style cases that already exist in `evaluation/perf_eval.py`, prefer that harness instead of inventing a new runner.

3. Capture a non-profiler baseline.
Run the chosen benchmark once without a profiler and record the exact command, backend, type, args, and any environment selectors such as `BATCHLAS_GEMM_VARIANT` or `BATCHLAS_GEMM_SYCL_KERNEL`.

4. Choose the lightest profiling mode that answers the question.
- Use built-in kernel trace first when you need BatchLAS phase names or function-level scope labels.
- Use `nsys` next when you need a time-sorted kernel/API breakdown or need to separate orchestration from kernel time.
- Use `ncu` only after the hot kernel is known.
- Use `nvprof` only on older CUDA toolkits where it still runs; on this machine it reports that CUDA 13 no longer supports it.

5. Generate commands instead of hand-assembling them.
Use `scripts/emit_profile_commands.py` to print repo-correct commands for trace, `nsys`, `ncu`, or `nvprof`. Pass benchmark args after `--`.

6. Report evidence, not guesses.
Name the benchmark used, the exact profiler command, the hottest kernels or trace scopes, and the first plausible bottleneck. If you recommend a code change, tie it to a concrete metric or trace observation.

## Quick Commands

- Find likely targets for `src/extensions/ormqr_blocked.cc`:
  `python3 .github/skills/batchlas-function-profiling/scripts/find_batchlas_profile_targets.py src/extensions/ormqr_blocked.cc`
- Emit a kernel trace command for a specific benchmark case:
  `python3 .github/skills/batchlas-function-profiling/scripts/emit_profile_commands.py --benchmark gemm_steady_benchmark --tool trace --output-stem gemm_steady_512 -- 512 512 512 512`
- Emit `nsys` and `ncu` commands for a known hotspot kernel:
  `python3 .github/skills/batchlas-function-profiling/scripts/emit_profile_commands.py --benchmark gemm_steady_benchmark --tool all --kernel-regex GemmRegisterTiledKernel --env BATCHLAS_GEMM_VARIANT=sycl --env BATCHLAS_GEMM_SYCL_KERNEL=128x32x32_s2_u2 -- 512 512 512 512`

## Rules

1. Do not add new benchmark code until you have confirmed no existing benchmark or evaluation harness can isolate the target.
2. Keep the benchmark case fixed while comparing variants or code changes.
3. For NVIDIA profiler runs in this repo, use short runs with explicit warmup and usually `--min_iters=1 --max_iters=1 --min_time=0`.
4. Search the implementation for `BATCHLAS_KERNEL_TRACE_SCOPE` before assuming the repo lacks internal instrumentation.
5. Use `nsys` before `ncu`.
6. Filter `ncu` to one kernel as soon as the hotspot is known.
7. Prefer absolute benchmark paths in final commands and reports.
8. If profiler output conflicts with the non-profiler baseline, call that out before recommending optimizations.

## Resources

- Read `references/workflow.md` when you need repo-specific command patterns, profiler selection guidance, or notes about current-toolchain behavior.
- Run `scripts/find_batchlas_profile_targets.py` to map files and symbols to likely benchmarks and trace labels.
- Run `scripts/emit_profile_commands.py` to print ready-to-run commands without hand-editing profiler flags.

## Completion Criteria

Consider the task complete when you have identified the right benchmark or harness, produced a reproducible profiling command, and summarized the bottleneck with evidence from either BatchLAS trace scopes, `nsys` summaries, or `ncu` metrics.
