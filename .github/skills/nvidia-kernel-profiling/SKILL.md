---
name: nvidia-kernel-profiling
description: 'Profile and tune NVIDIA CUDA kernels during development with Nsight Systems (nsys) and Nsight Compute (ncu) from the CLI. Use for new kernel bring-up, hotspot triage, launch overhead vs kernel time, memcpy and sync analysis, occupancy, warp stall reasons, source-level bottleneck attribution, memory hierarchy throughput, access-pattern analysis, and targeted kernel inspection on BatchLAS benchmarks or custom kernel drivers.'
argument-hint: 'Kernel development target to profile, plus optional kernel regex or benchmark command'
user-invocable: true
disable-model-invocation: false
---

# NVIDIA Kernel Profiling

Use this skill when you are developing, validating, or tuning NVIDIA kernels and need to determine what is actually limiting performance. Start with Nsight Systems to find where time goes at the application level, then switch to Nsight Compute only for one or two specific kernels.

## When To Use

- You are bringing up a new CUDA kernel and need a fast measurement loop.
- You are comparing two kernel variants and need defensible performance evidence.
- A CUDA or BatchLAS benchmark is slower than expected.
- You need to separate kernel time from launch overhead, synchronization, allocations, or memory copies.
- You need launch statistics, occupancy, warp stall reasons, source-level hotspots, or memory hierarchy throughput metrics for a specific kernel.
- You are profiling BatchLAS benchmarks such as `stedc_benchmark`, `steqr_benchmark`, `syev_*`, or `sytrd_*`, or a custom driver program that isolates a kernel under development.

## Rules

1. Use `nsys` before `ncu`.
2. For benchmarks in this repo, include `--warmup=5`.
3. Always create the output directory before using `-o` or `--export`, for example `output/profiling/`.
4. Start with one small representative case, not the full benchmark sweep.
5. After the first `ncu` attempt, filter to a specific kernel. Do not keep profiling startup kernels.
6. If `ncu` perturbs the workload or the process aborts after disconnect, keep the generated `.ncu-rep` if it exists and inspect it with `ncu --import` instead of immediately retrying.
7. When tuning a new kernel, prefer a dedicated microbenchmark or driver that launches the kernel repeatedly with stable inputs.
8. Change one performance variable at a time: launch geometry, data layout, algorithmic structure, memory strategy, or synchronization.
9. Use the same problem size and same profiler command when comparing kernel variants.
10. Collect advanced `ncu` sections only after the hotspot kernel is known and the baseline `basic` pass has been inspected.
11. Treat warp stall reasons as secondary evidence unless scheduler issue efficiency is weak or eligible warps are low.

## Procedure

### 1. Build A Stable Measurement Target

Before profiling, make sure the target is suitable for performance work:

- Use a dedicated benchmark, microbenchmark, or driver program if possible.
- Keep setup and teardown outside the timed region when you can.
- Reuse allocations across iterations if the question is kernel speed rather than end-to-end overhead.
- Pick one representative problem size that is large enough to exercise the kernel but small enough to rerun quickly.
- If you are comparing variants, make sure both variants use the same inputs, same warmup, and same iteration controls.

### 2. Confirm Prerequisites

- Check `which nsys`, `which ncu`, and `nvidia-smi`.
- Ensure the benchmark or driver binary exists.
- Pick a single command line that is representative and quick to rerun.

### 3. Capture A Fast Baseline Outside The Profiler

Run the target normally first and record:

- exact command
- average runtime or throughput
- problem size
- backend and datatype
- any launch parameters under test

Do this before introducing profiler overhead. The point is to preserve a clean baseline for later comparison.

### 4. Run Low-Overhead Nsight Systems Triage

Use a minimal trace first:

```bash
mkdir -p output/profiling
nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --cpuctxsw=none \
  --stats=true \
  --force-overwrite=true \
  -o output/profiling/<run_name> \
  <command>
```

For `stedc_benchmark` in this repo:

```bash
nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --cpuctxsw=none \
  --stats=true \
  --force-overwrite=true \
  -o output/profiling/stedc_nsys \
  build/benchmarks/stedc_benchmark \
  --backend=CUDA \
  --type=float \
  --warmup=5 \
  --min_iters=1 \
  --max_iters=1 \
  256 256 16 0 32 1
```

Focus on these `nsys` reports:

- `cuda_gpu_kern_sum`: top kernels by total GPU time
- `cuda_api_sum`: expensive API, sync, or allocation calls
- `cuda_gpu_mem_time_sum`: memcpy and memset time
- `cuda_gpu_mem_size_sum`: transfer volume

### 5. Decide What Is Actually Dominating

- If GPU kernels dominate, move to `ncu`.
- If `cuEventSynchronize`, launches, allocations, or memcopies dominate, fix orchestration first.
- If several short kernels dominate, look for fusion, batching, or launch overhead.
- If one kernel dominates, target that kernel directly in `ncu`.
- If the target includes significant initialization noise, tighten the benchmark or use a narrower driver before collecting more metrics.

### 6. Run Nsight Compute On One Specific Kernel

Start with the `basic` set and a kernel filter:

```bash
ncu \
  --set basic \
  --kernel-name-base demangled \
  --kernel-name regex:<kernel_regex> \
  --launch-count 1 \
  --target-processes all \
  --force-overwrite \
  --export output/profiling/<run_name> \
  <command>
```

For the STEDC fused merge kernel observed locally:

```bash
ncu \
  --set basic \
  --kernel-name-base demangled \
  --kernel-name regex:StedcFusedCtaMerge \
  --launch-count 1 \
  --target-processes all \
  --force-overwrite \
  --export output/profiling/stedc_ncu_merge \
  build/benchmarks/stedc_benchmark \
  --backend=CUDA \
  --type=float \
  --warmup=5 \
  --min_iters=1 \
  --max_iters=1 \
  256 256 16 0 32 1
```

If the first `ncu` run captures a startup kernel instead of the real hotspot, rerun with a tighter `--kernel-name` filter or with launch skipping.

### 7. Import Saved Nsight Compute Reports Instead Of Reprofiling

```bash
ncu --import output/profiling/<run_name>.ncu-rep --page details
```

For quick summaries, add `--print-summary per-kernel`, or pipe through `head` if the output is long.

### 8. Interpret The First-Pass Metrics

Prioritize these metrics:

- `Duration`
- `Compute (SM) Throughput`
- `Memory Throughput`
- `DRAM Throughput`
- `Registers Per Thread`
- `Block Size` and `Grid Size`
- `Theoretical Occupancy`
- `Achieved Occupancy`
- `Waves Per SM`
- `Compute and Memory throughput breakdowns`

Common interpretations:

- Low achieved occupancy and low memory throughput: the kernel is underfilled, too serialized, or not exposing enough parallel work.
- High memory throughput with low compute throughput: likely memory-bound.
- High registers per thread plus low occupancy: register pressure is limiting residency.
- Tiny grid or very low waves per SM: launch geometry is too small to saturate the GPU.
- Many replay passes: the metric set is expensive; narrow the sections or profile fewer kernels.
- Good occupancy but poor issue efficiency: look at scheduler and warp-state sections before changing block size.
- Good SM throughput but poor wall-clock speed: investigate memory hierarchy pressure, barriers, or poor access patterns.

Development-focused follow-up questions:

- Is the kernel too small, or is the mapping from work to threads poor?
- Is the kernel spending time waiting on dependent work from earlier kernels?
- Is register pressure caused by real algorithmic needs, or by avoidable temporaries and inlining?
- Would a different tile size, block size, or batching strategy increase waves per SM?
- Is end-to-end time dominated by the kernel itself, or by staging data for the kernel?

### 9. Run Advanced Nsight Compute Diagnosis When Basic Metrics Are Not Enough

If the `basic` set shows that the kernel is still the bottleneck but does not explain why, collect a small targeted section set instead of immediately using `--set full`.

Recommended focused sections:

- `SchedulerStats`: active, eligible, and issued warps per scheduler
- `WarpStateStats`: aggregate warp stall state breakdown
- `SourceCounters`: source-level metrics, sampled warp stall reasons, memory traffic by instruction
- `MemoryWorkloadAnalysis` or `MemoryWorkloadAnalysis_Tables`: L1/TEX, L2, DRAM, and shared-memory tables
- `ComputeWorkloadAnalysis`: pipeline utilization and IPC
- `InstructionStats`: instruction mix when pipe balance is suspect
- `SpeedOfLight`: top-level compute and memory throughput breakdowns
- `PmSampling` or `PmSampling_WarpStates`: timeline-like sampling when behavior changes across kernel lifetime

Example advanced command for a single known kernel:

```bash
ncu \
  --section SchedulerStats \
  --section WarpStateStats \
  --section SourceCounters \
  --section MemoryWorkloadAnalysis_Tables \
  --section ComputeWorkloadAnalysis \
  --section SpeedOfLight \
  --kernel-name-base demangled \
  --kernel-name regex:<kernel_regex> \
  --launch-count 1 \
  --target-processes all \
  --force-overwrite \
  --export output/profiling/<run_name>_advanced \
  <command>
```

If this is too heavy or unstable, split it into two passes:

- scheduler plus source analysis
- memory hierarchy analysis

### 10. Read Warp Stall Reasons Correctly

Use `SchedulerStats`, `WarpStateStats`, and `SourceCounters` together.

Prioritize these patterns:

- Low eligible warps or many skipped issue slots: latency hiding is failing.
- High `smsp__pcsamp_warps_issue_stalled_long_scoreboard`: warps are waiting on L1TEX-backed operations, usually global/local/texture data.
- High `smsp__pcsamp_warps_issue_stalled_short_scoreboard`: often shared-memory dependencies, special math, or short-latency MIO dependencies.
- High `smsp__pcsamp_warps_issue_stalled_lg_throttle`: too many local or global memory operations are in flight, often from redundant traffic or spills.
- High `smsp__pcsamp_warps_issue_stalled_mio_throttle`: MIO pressure, often shared-memory traffic, dynamic branches, or special math saturation.
- High `smsp__pcsamp_warps_issue_stalled_math_pipe_throttle`: one math pipeline is oversubscribed; inspect instruction mix and pipe balance.
- High `smsp__pcsamp_warps_issue_stalled_barrier` or `membar`: synchronization is expensive; inspect work imbalance before the barrier.
- High `smsp__pcsamp_warps_issue_stalled_no_instructions`: instruction fetch or tiny-grid effects; inspect code size, divergence, and total waves.

Do not treat the top stall reason as the fix by itself. First ask whether the scheduler had enough eligible warps to issue consistently. If issue efficiency is already high, stall sampling may be descriptive rather than actionable.

### 11. Attribute Bottlenecks To Source Lines And Instructions

Use `SourceCounters` when you need to identify which instructions or source lines generate the most stalls or memory traffic.

Look for these source-level signals:

- `smsp__pcsamp_sample_count`: where warp sampling is concentrated
- sampled warp stall reasons by program counter from the `smsp__pcsamp_warps_issue_stalled_*` metrics
- `memory_l1_tag_requests_global`: global-memory pressure by instruction
- `memory_l2_theoretical_sectors_global` and `_ideal`: whether a global access pattern is generating excess sectors
- `memory_l1_wavefronts_shared` and `_ideal`: whether shared-memory instructions need excess wavefronts
- `derived__memory_l1_conflicts_shared_nway`: shared-memory bank-conflict severity
- `derived__local_spilling_requests` and `derived__local_spilling_requests_pct`: local-memory pressure from register spills
- `sass__inst_executed_register_spilling*`: executed spill loads and stores

Use this to answer questions like:

- Which exact load or store line is driving long scoreboard stalls?
- Which shared-memory instruction is causing bank conflicts or short scoreboard stalls?
- Which instruction stream is creating spill traffic?
- Which branch site or synchronization point dominates sampled stalls?

### 12. Diagnose Access Patterns And Memory Hierarchy Limits

Use `MemoryWorkloadAnalysis` or `MemoryWorkloadAnalysis_Tables` when the kernel appears memory-bound or warp stalls implicate L1TEX, L2, shared memory, or DRAM.

Focus on these table columns and interpretations:

- L1/TEX `Sectors/Req`: high values indicate poor coalescing or scattered access patterns.
- L1/TEX `Hit Rate`: low values increase downstream L2 traffic and latency.
- L1/TEX `% Peak to L2`: high write, atomic, or reduction pressure leaving L1.
- L1/TEX `% Peak to SM`: high return-path pressure, often from scattered reads.
- Shared `Wavefronts` and `Bank Conflicts`: excess wavefronts and bank conflicts indicate inefficient shared-memory layout.
- L2 `Hit Rate`: low values imply more DRAM traffic and longer latency.
- L2 `Throughput` and `% Peak`: whether L2 itself is saturated.
- DRAM `Throughput` and `% Peak`: whether off-chip bandwidth is the real limiter.
- `SpeedOfLight` throughput breakdowns: whether memory or compute units are nearer their sustained peak.

Common interpretations:

- High long scoreboard plus poor L1/L2 hit rates: improve locality or reduce demand before changing math structure.
- High DRAM throughput with modest SM throughput: likely bandwidth-bound.
- High shared-memory bank conflicts plus short scoreboard or MIO throttle: redesign shared-memory layout, access width, or indexing.
- High `Sectors/Req` with low arithmetic intensity: improve coalescing before trying occupancy tweaks.

### 13. Use Timeline Sampling For Phase Changes

If a kernel has distinct phases, tail effects, or changing behavior over time, use `PmSampling` or `PmSampling_WarpStates`.

Use this when:

- one phase is bandwidth-bound and another is latency-bound
- occupancy or throughput collapses near the end of the kernel
- you need to separate startup, steady-state, and tail behavior

This is especially useful for kernels with reductions, merges, or irregular control flow where aggregate counters hide phase changes.

### 14. Iterate Like A Kernel Tuning Loop

After the first `ncu` pass:

1. Form one concrete hypothesis.
2. Change one thing in code or launch parameters.
3. Rebuild and rerun the same command.
4. Compare baseline runtime, `nsys` hotspot share, and the key `ncu` metrics.
5. Keep the change only if the profiler evidence and wall-clock result agree.

Typical changes worth testing:

- block size or CTA shape
- grid decomposition
- shared memory use
- register pressure reduction
- fewer synchronizations
- fused kernels or batched work
- improved memory coalescing or reduced traffic

### 15. Escalate Only When Needed

- Use `--section LaunchStats,Occupancy` for launch-shape and occupancy questions.
- Use `--section SchedulerStats,WarpStateStats,SourceCounters` for warp-stall and source-line diagnosis.
- Use `--section MemoryWorkloadAnalysis_Tables` for L1, L2, DRAM, and shared-memory throughput and access-pattern questions.
- Use `--section SpeedOfLight,ComputeWorkloadAnalysis,InstructionStats` for pipe balance and compute saturation questions.
- Use `--section PmSampling` or `--section PmSampling_WarpStates` when the bottleneck changes over the kernel lifetime.
- Use `--set full` only for a very small filtered case.
- Use `--page raw --csv` when the result needs machine-readable parsing.
- If you need a more detailed memory or pipeline picture, add specific sections rather than jumping straight to a huge report.
- Consider `--cache-control none` only when you need to study warmed-cache behavior inside a larger workflow; otherwise keep the default for reproducibility.
- If a new kernel is still unclear, add or improve NVTX ranges around the phase that launches it so later profiling can isolate the right region faster.

## BatchLAS Notes

- Benchmarks accept positional integer args after options.
- For new kernel development, a custom driver or benchmark target is acceptable and often preferable to a large end-to-end benchmark.
- `stedc_benchmark` locally accepted `256 256 16 0 32 1` for:
  - `n`
  - `batch`
  - `recursion_threshold`
  - `merge_variant`
  - `threads_per_root`
  - `wg_multiplier`
- Representative STEDC findings from this workspace are in [local STEDC notes](./references/batchlas-stedc.md).

## Output Checklist

When reporting findings, include:

1. The exact command used.
2. Whether the target is an end-to-end benchmark, microbenchmark, or custom kernel driver.
2. The top kernels from `nsys`.
3. Whether time is dominated by kernels, API or sync overhead, or copies.
4. The top `ncu` kernel metrics and what they imply.
5. The dominant scheduler or warp-stall signals, if collected.
6. The source lines or instruction classes most associated with stalls, memory traffic, spills, or bank conflicts, if collected.
7. The key L1, L2, DRAM, and shared-memory findings, including whether access patterns look coalesced.
8. One or two concrete optimization directions.
9. The next experiment you would run to validate the leading hypothesis.