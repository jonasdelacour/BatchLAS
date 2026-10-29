---
name: nvidia-kernel-profiling
description: 'Profile and tune NVIDIA CUDA kernels during development with Nsight Systems (nsys) and Nsight Compute (ncu) from the CLI. Use for new kernel bring-up, hotspot triage, launch overhead vs kernel time, memcpy and sync analysis, occupancy, warp stall reasons, source-level bottleneck attribution, memory hierarchy throughput, access-pattern analysis, and targeted kernel inspection on BatchLAS benchmarks or custom kernel drivers.'
argument-hint: 'Kernel development target to profile, plus optional kernel regex or benchmark command'
user-invocable: true
disable-model-invocation: false
---

# NVIDIA Kernel Profiling

Find where time goes with `nsys`, then explain one or two hot kernels with `ncu`. Measurement rules
(saturation, warm-up, A/B reps) are agent-guide §10; kernel design facts and measured dead ends are §11
(`python3 scripts/agent/doc_section.py agent-guide §11`). To map a source file to a benchmark or emit
ready-made commands, use the `batchlas-function-profiling` skill.

## Rules

1. `nsys` before `ncu`; `ncu` on a filtered kernel only, never startup kernels.
2. Benchmarks here: `--backend=CUDA` (also for SYCL kernels), `--warmup=5`, and for profiler runs
   `--min_iters=1 --max_iters=1 --min_time=0`. Positional integer args after the options pick one case.
3. `mkdir -p output/profiling` before `-o`/`--export`.
4. One representative case, same command and size for every variant; change one variable at a time
   (launch geometry, layout, algorithm, memory strategy, synchronisation).
5. Record a baseline outside the profiler first: exact command, runtime, size, backend, type, route pins.
6. If `ncu` aborts after disconnect but wrote a `.ncu-rep`, analyse it with `ncu --import`; do not rerun.
7. Advanced `ncu` sections only after the `basic` pass is read; warp stalls are secondary evidence unless
   eligible warps or issue efficiency are low.
8. Keep a change only when the wall-clock result and the profiler evidence agree.

## 1. Nsight Systems triage

```bash
mkdir -p output/profiling
nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none --stats=true \
  --force-overwrite=true -o output/profiling/<run> \
  build/benchmarks/<target> --backend=CUDA --type=float --warmup=5 \
  --min_iters=1 --max_iters=1 --min_time=0 <args...>
nsys stats --report cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum \
  output/profiling/<run>.nsys-rep
```

- Kernels dominate: go to `ncu` on the top one.
- `cuEventSynchronize`, launches, allocations or copies dominate: fix orchestration first.
- Many short kernels: look for fusion, batching or launch overhead.
- Initialisation noise: tighten the case or write a narrower driver before collecting more metrics.

SYCL kernels: add `--cuda-event-trace=true` to get GPU-side kernel records, and `--export=sqlite` to
read launch parameters (registers, grid/block, shared and local memory, duration; no throughput or
stalls):

```sql
SELECT s.value AS kernel, k.registersPerThread, k.gridX, k.gridY, k.gridZ, k.blockX, k.blockY, k.blockZ,
       k.staticSharedMemory, k.dynamicSharedMemory, k.localMemoryPerThread, (k.end - k.start) AS ns
FROM CUPTI_ACTIVITY_KIND_KERNEL AS k JOIN StringIds AS s ON k.demangledName = s.id
WHERE s.value LIKE '%GemmRegisterTiled%' ORDER BY ns DESC LIMIT 5;
```

## 2. Nsight Compute on one kernel

```bash
ncu --set basic --kernel-name-base demangled --kernel-name regex:<kernel_regex> \
  --launch-count 1 --target-processes all --force-overwrite \
  --export output/profiling/<run> <command>
ncu --import output/profiling/<run>.ncu-rep --page details --print-summary per-kernel
```

If it caught the wrong launch, tighten `--kernel-name` or skip launches (`--launch-skip N`).
`--page raw --csv` gives machine-readable output.

Read first: Duration, SM / Memory / DRAM throughput, registers per thread, block and grid size,
theoretical vs achieved occupancy, waves per SM.

| Signal | Likely meaning |
| --- | --- |
| Low achieved occupancy and low memory throughput | too little parallel work or a poor work-to-thread mapping |
| Tiny grid, waves per SM << 1 | launch geometry cannot fill the GPU |
| High registers per thread and low occupancy | register pressure limits residency (gate: §11) |
| High memory, low compute throughput | memory-bound |
| Good occupancy, poor issue efficiency | read scheduler and warp-state sections before changing block size |
| Good SM throughput, poor wall clock | memory hierarchy pressure, barriers or access pattern |
| Many replay passes | metric set too expensive; narrow sections or kernels |

## 3. Escalate with targeted sections

Add `--section` sets to the command above instead of `--set full` (only on a tiny filtered case):

| Question | Sections |
| --- | --- |
| Launch shape, occupancy | `LaunchStats`, `Occupancy` |
| Warp stalls, source lines | `SchedulerStats`, `WarpStateStats`, `SourceCounters` |
| L1/L2/DRAM/shared traffic, coalescing | `MemoryWorkloadAnalysis_Tables` |
| Pipe balance, compute saturation | `SpeedOfLight`, `ComputeWorkloadAnalysis`, `InstructionStats` |
| Behaviour changing over the kernel lifetime | `PmSampling`, `PmSampling_WarpStates` |

Split into two passes (scheduler+source, memory) if one is too heavy. Keep the default cache control
unless you study warmed-cache behaviour. How to read stall reasons, source counters and memory tables:
[references/ncu-metrics.md](./references/ncu-metrics.md).

## 4. Tuning loop

One hypothesis, one change (block or CTA shape, grid decomposition, shared memory, register pressure,
fewer barriers, fusion, coalescing), rebuild with `scripts/b`, rerun the same command, compare baseline
runtime, `nsys` hotspot share and the key `ncu` metrics. Add NVTX ranges if a phase is hard to isolate.

## SYCL traps

- `ncu` replay against a debug-built `libsycl.so` trips runtime assertions (`NDR.LocalSize[0] == 0` in
  `commands.cpp`) and aborts; `--replay-mode application`, single sections and assert-suppressing
  `LD_PRELOAD` do not help. Use a release runtime, the nsys SQLite route above, or a standalone
  nvcc driver that reimplements the kernel without the SYCL runtime.
- Device code is AOT-compiled to an sm_89 cubin at the shared-library device link. Register and spill
  counts: `scripts/register_probe.sh`; PTX: the `ptx-codegen-comparison` skill.

## BatchLAS notes

- Prefer an existing benchmark or a dedicated driver that reuses allocations and keeps setup out of the
  timed region over a full end-to-end sweep.
- STEDC recipe and argument layout: [references/batchlas-stedc.md](./references/batchlas-stedc.md).

## Report

Command; target kind (benchmark, microbenchmark, driver); top `nsys` kernels and whether kernels, API/sync
or copies dominate; key `ncu` metrics and their meaning; stall, source-line and memory findings if
collected; one or two optimisation directions and the next experiment that tests the lead hypothesis.
