# Reading advanced ncu sections

## Warp stalls (`SchedulerStats`, `WarpStateStats`, `SourceCounters`)

First check whether the scheduler had enough eligible warps. If issue efficiency is already high, the
top stall reason is descriptive, not the fix.

| `smsp__pcsamp_warps_issue_stalled_*` | Usual cause |
| --- | --- |
| `long_scoreboard` | waiting on L1TEX: global, local or texture data |
| `short_scoreboard` | shared-memory dependencies, special math, short MIO latency |
| `lg_throttle` | too many local/global ops in flight: redundant traffic or spills |
| `mio_throttle` | shared-memory traffic, dynamic branches, special-math saturation |
| `math_pipe_throttle` | one pipe oversubscribed: check instruction mix |
| `barrier`, `membar` | synchronisation; check work imbalance before the barrier |
| `no_instructions` | instruction fetch or tiny grid: code size, divergence, waves |

## Source attribution (`SourceCounters`)

- `smsp__pcsamp_sample_count` and the per-PC stall metrics: which line stalls.
- `memory_l1_tag_requests_global`: global pressure per instruction.
- `memory_l2_theoretical_sectors_global` vs `_ideal`: excess sectors (poor coalescing).
- `memory_l1_wavefronts_shared` vs `_ideal`, `derived__memory_l1_conflicts_shared_nway`: bank conflicts.
- `derived__local_spilling_requests[_pct]`, `sass__inst_executed_register_spilling*`: spill traffic.

## Memory hierarchy (`MemoryWorkloadAnalysis_Tables`)

| Column | High/low means |
| --- | --- |
| L1/TEX `Sectors/Req` high | uncoalesced or scattered access: fix before occupancy tweaks |
| L1/TEX or L2 `Hit Rate` low | more downstream traffic and latency: improve locality |
| L1/TEX `% Peak to L2` high | write, atomic or reduction pressure |
| L1/TEX `% Peak to SM` high | return-path pressure, often scattered reads |
| Shared `Bank Conflicts`, excess `Wavefronts` | redesign layout, access width or indexing |
| L2 / DRAM `% Peak` high | that level is the limiter (bandwidth-bound if SM throughput is modest) |

Long scoreboard plus poor hit rates: reduce demand or improve locality before changing the math. Bank
conflicts plus short scoreboard or MIO throttle: shared-memory layout.

## Phases (`PmSampling`, `PmSampling_WarpStates`)

Use when one phase is bandwidth-bound and another latency-bound, throughput collapses in the tail, or
aggregate counters hide reductions, merges or irregular control flow.
