# Batch-size sensitivity {#perf_batch_sensitivity}

> **Status:** current · RTX PRO 6000 Blackwell Max-Q (sm_120, 188 SMs, 128 MiB L2, 300 W), threadripper02 GPU 2, CUDA 13.2 · measured 2026-10-20 to 2026-10-21

Why per-item time in the tables and benchmarks depends on batch, ranked by size, and what was done
about it. Most of the "degrades past saturation" signal is the timing harness, not the kernels; the
small-batch half is a real one-work-group-per-item property. Grid-stride loops fix neither.

**Harness.** A standalone sweep linked against the library: `malloc_device` views, 3 warm-up calls, 7 reps per mode, modes interleaved and order alternated per rep.

| Mode | Protocol |
| --- | --- |
| warm | the tuner's: reset copy (`A0 -> A` or `C0 -> C`), `q.wait()`, host clock around call + `q.wait()` |
| cold | warm plus a 1 GiB read scan before the clock; after it a 64 MiB re-read runs at 1545 GB/s instead of 4112 |
| pipe | 16 back-to-back calls, one wait |

Device anchors: L2-resident read 3.9-4.8 TB/s, DRAM read 1.55-1.63
TB/s, the whole 128 MiB usable (128 MiB re-read 4.63 TB/s, 160 MiB 1.59 TB/s). Empty-kernel
submit+wait 5.39 us, pipelined 1.82 us per launch.

## Batch sensitivity: the culprits ranked

| Rank | Culprit | Regime | Kind | Status |
| --- | --- | --- | --- | --- |
| C1 | L2-warm timing in the tuner and minibench | past saturation | harness | open (F1) |
| C2 | one work-group per item: under-fill and wave sawtooth | small batch | kernel | real; F5 prototype failed |
| C3 | ~7 us host floor per call | small batch | harness + library | open (F2, F6) |
| C4 | 300 W power-cap throttling | long large-batch cells | this box | open (F2) |
| C5 | co-tenant time-slicing in ~220 sm_120 cells | isolated spikes | data | open (re-measure) |
| C6 | 65535 grid cap | batch > 65535 | kernel launch | fixed (F4b slices) |
| — | multi-kernel L2 hand-off loss | past L2, blocked drivers | real | fixed (F4a chunking) |
| — | nearest-row routing over a sparse batch ladder | between rows | tables | partly (F3 ladder) |

### Batch sensitivity: C1, the L2-warm timing artifact

`tools/tune/cell_runner.hh` `reset_and_run()` (lines 211-216) runs `p_.reset()`, a device copy plus
`q.wait()`, then starts the clock. Read-only operands stay in L2 from the previous rep and the mutated
ones were just written by the reset; nothing in `tools/`, `benchmarks/` or `include/batchlas/util/`
flushes L2 or rotates buffers. `minibench.hh` is L2-warm too, and pipelined. While batch x per-item
footprint fits in L2, bandwidth-bound cells run from L2; past it they fall to DRAM, so batch stands in
for working set / L2.

Warm vs cold, ns per item:

| Cell | Batch (WS) | Warm | Cold | Cold / warm |
| --- | --- | ---: | ---: | ---: |
| gemm 32^3, Auto (`wide` 32) | 4096 (48 MiB) | 5.39 | 10.16 | 1.88 |
| | 8192 (96 MiB) | 4.54 | 8.82 | 1.94 |
| | 16384 (192 MiB) | 8.05 | 8.37 | 1.04 |
| | 32768 | 10.90 | 10.03 | 0.92 |
| | 65536 | 11.08 | 10.68 | 0.96 |
| gemm 128^3, pinned `reg` | 512 (96 MiB) | 106.6 | 171.9 | 1.61 |
| | >= 2048 | 165-178 | 165-178 | 1.00 |
| gemv 256x1024 | 128 (128.6 MiB) | 30.8 us (4.4 TB/s) | 92.5 us | 3.0 (re-run 2.43) |
| | >= 256 | ~0.65 us/item | same | 1.00 |
| trsm n=32 `sg_left` | <= 256 | | | 1.35-1.40 |
| | 512-8192 | | | 1.15-1.25 |
| | > 192 MiB | | | 0.92-0.99 |
| potrf n=16 / n=64 | across the L2 edge | | | 1.04-1.17 / 1.00-1.06 |
| getrf n=32 / n=128 | across the L2 edge | | | 1.03-1.23 (peak b4096) / 1.00-1.07 |
| syev n=16 | across the L2 edge | | | 1.00-1.02 |

- Warm gemm 32^3 "degrades" 2.4x (4.5 -> 10.9 ns); cold it is flat within ~25%. Factorizations are
  immune (<= 7%, except getrf n=32 at 1.23x).
- **Real residue, output write-back:** cold gemm 32^3 still steps 8.4 -> 10.0 ns at b32768, where the
  128 MiB of C no longer stays dirty in L2 and write-backs land inside the timed call (~20%).
- **Table footprint**: 92.5% of sm_120 gemv cells with L2/2 < WS <= L2
  exceed the 1792 GB/s DRAM peak (max 2.73x), impossible from DRAM. Per-item time at the first batch
  with WS >= 2 L2 over the best in-L2 time: gemm geomean 1.40x (75% of series > 1.15), gemv 1.72x,
  symm 1.14x; getrf, potrf, posv, syev <= 1.0x.

### Batch sensitivity: C2, one work-group per item and the wave sawtooth

Every native launch is `ceil(batch/G)` work-groups, one work-group, sub-group or tile per item, with a
serial per-item critical path (`potrf_cta.cc:121`, `getrf_cta.cc:147`, `getrs_fused.cc:237`). The GPU
holds \f$C = \text{SMs} \times \text{residency} \times \text{items/WG}\f$ items at once, so
\f$T(B) \approx T_0 + \lceil B/C \rceil T_\text{wave}\f$: flat below C (per-item time ~ 1/B), a step at
each multiple of C. Per-item lanes are chosen from n alone; only trsm's V1 ladder and the gemv/spmm
work-group ladders read the CU count.

Sawtooth, cold totals:

| Kernel | C | C -> C+1 | 2C -> 2C+1 | 3C -> 3C+1 | 4C -> 4C+1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| potrf `lpanel:panel=8`, n=128 (64 regs, 8 WG/SM) | 1504 | +42% (191.0 -> 270.3 us) | +16% | +7% | +3% |
| potrf `lpanel:panel=16`, n=128 (100 regs) | 752 | +63% (124.5 -> 202.4 us) | +30% | +18% | +12% |
| gemm `reg:m=128:n=128:k=8:u=1`, 128^3 (1 WG/SM, warm) | 188 | +32% (26.4 -> 34.8 us) | +26% | | +10% |

Per-item time at whole waves is flat to 0.3% (lpanel:8: 127.0 / 126.9 / 127.1 / 126.6 ns).

Under-fill at b128 (nsys):

- potrf n=64 `lpanel:8`: 64 work-groups of 128, kernel 36.0 us (35.7 us at b8); saturated it costs
  25.5 ns/item, so b128 would need 3.3 us. 64 of 1504 resident slots = **4% fill**.
- potrf n=16 `tiny`: kernel flat at 3.81 / 3.87 / 4.00 us at b8 / 128 / 1024; 30 ns/item at b128 against
  0.75 saturated (40x).
- getrf n=32 `tiny` 14.6 / 14.8 / 15.9 us at b8 / 128 / 1024; trsm `sg_left` 8.8 / 9.5 / 15.5 us.

This is what the batch axis genuinely encodes: families with different C and single-item latency swap
winners as B/C crosses ~1. Auto traces: potrf n=64 `lpanel:8` at b <= 256,
`lpanel:16` at b512-1024, `lpanel:8` from b2048; gemm 32^3 vendor at b <= 64, `wide` 16 at b128-512,
`wide` 32 at b1024-32768 (pinned cold: wide16 82.5 vs wide32 91.7 ns at b128, wide32 8.3 vs wide16 10.6
at b16384); syev n=16 vendor to b1024, `cta` from b2048.

### Batch sensitivity: C3, the per-call host floor

Each tuner sample is one synchronous call; launch, host dispatch and the sync return are always inside
it. Host time minus nsys kernel time at b128 is a fixed 6.3-7.8 us:

| Cell (b128) | Host us | Kernel us | Floor share |
| --- | ---: | ---: | ---: |
| potrf n=16 `tiny` | 10.7 | 3.87 | 64% |
| gemm 32^3 `wide` 16 | 10.2 | 2.40 | 76% |
| trsm n=32 `sg_left` | 15.8 | 9.5 | 40% |
| getrf n=32 `tiny` | 21.8 | 14.8 | 32% |
| potrf n=64 `lpanel` | 43.6 | 36.0 | 17% |
| gemm 128^3 `wide` 64 | 22.8 | 15.6 | 32% |

- Pipelined, the library path still costs 5.1-6.1 us per gemm 32^3 call against a 2.4 us kernel: about
  3.5-4 us of host path (selection, arena lease, checks) beyond a raw launch.
- Every batched vendor call pays `MatrixView::init_data_ptr_array` (`src/matrix.cc:2408-2426`), a
  `parallel_for` plus `.wait()`: +5-11 us. Native blocked drivers inherit it when a child routes to the
  vendor at small batch (potrf blocked b128: 2 syncs).
- Multi-launch drivers pay ~1.8-5 us host time per launch: getrf n=128 `blocked` 14 kernels per call,
  syev n=16 `cta` 11.

### Batch sensitivity: C4, power-cap throttling on the Max-Q part

The four sm_120 GPUs are 300 W Max-Q parts (limit fixed, 250-300 W). Short cells boost to 2272-2317
MHz; sustained ones hit throttle reason 0x4 (SW power cap) at 862-1815 MHz.

| Cell | Burst | Sustained | Ratio |
| --- | ---: | ---: | ---: |
| gemm 128^3 `reg` b256 (in L2) | 114.4 ns/item, 2317 MHz | 127.9 ns/item (8000 reps), 1815 MHz | 1.12 |
| gemm 512^3 b128 (compute-bound) | 6.10 us/item | 7.65 us/item, 1230-1245 MHz | 1.25 |
| gemm 128^3 Auto b32768 (DRAM-bound) | 175.1 ns/item cold-only | 212.3 ns/item with pipe reps, median 877 MHz | 1.21 |

Measured inflation is 1.07-1.25x; the 1.3-1.6x from the table survey did not reproduce. It depends on
the duty cycle and on what ran before, and does not carry over to the 4090.

### Batch sensitivity: C5, co-tenant contamination of sm_120 cells

The sm_120 deep run used `--allow-idle-foreign` while another user's process held a context on every
GPU. Time-slicing adds a fixed offset per synchronising call (~170-200 us, or ~2.27-2.29 ms), the same
on every arm, about 2x on vendor arms. About 220 of ~100k sm_120 cells carry it (none on sm_89).

| Table cell | Table | Re-measured alone |
| --- | ---: | ---: |
| getrf float n=32 `tiny` b8 / b256 | 0.0206 / 0.0203 ms | 21.75-21.90 us at b64-256 |
| getrf float n=32 `tiny` b64 / 128 / 255 | 0.214 / 0.217 / 0.206 ms | (10x off) |
| getrf float n=32 b8192 | 0.2342 ms | 0.0532 warm, 0.0629 cold |
| getrf float n=128 `blocked` b64 | 0.3778 ms | 0.2028 |
| symm double m=n=1 b3 and b128, trsm cfloat order 2-4 b2048/8192 | 2.266-2.302 ms | neighbours 0.007-0.03 ms |

A second process looping gemm on the same GPU reproduces it: getrf n=32 goes from 21.8 us to a 2351 us
median at every batch. Routing reads the contaminated getrf rows at b32-128 and b8192-131072.

### Batch sensitivity: C6, the 65535 grid cap

Level-3 natives put batch in CUDA grid y/z. `can_run` refused batch > 65535, so syrk n=k=64 ran `gram` at
44.4 ns/item at b65535 and the per-item vendor loop at 3499 ns/item at b65536: a 79x cliff, outside the
tuned ladder (max 32768). gemm fell to `small` or the vendor (no cliff). Fixed by slicing (F4b, below).
The `int`-linearised expansion limit (`expansion_fits`, 2^31 scratch elements) is the cliff that remains.

### Batch sensitivity: multi-kernel L2 hand-off loss

Blocked drivers hand panels and trailing matrices between kernels through L2. getrf n=128 `blocked`
per item: 348.6 ns at b2048, 394 at 4096, 428 at 8192, 460 at 32768 (+32%, the same warm and cold, so
real). nsys per kernel, b2048 -> b32768: second trailing gemm 9.4 -> 32.4
ns/item, `LaswpGather` 23.7 -> 34.7, `TrsmCta` 19.0 -> 23.7; kernel sums 343 -> 436. Chunking the call
into 2048-item parts by hand recovered it (433.8 -> 358.1 ns). syev n=16 b131072 gained 9% at chunk 16384
and lost 32% at chunk 4096 (more launches, another route). Fixed for getrf and potrf (F4a).

### Batch sensitivity: nearest-row routing across sparse batch rows

The gemm float NN 128^3 table has rows at b1, 128 and 2048 only. Auto: vendor b1-8,
`wide` 64 b16-512, `reg` b1024-16384, `wide` 64 at b32768 (a borrowed row), vendor from b65536. Pinned
cold at b32768: `reg` 178.6 ns against the chosen `wide` 64 at 217 (+22%); `reg` is best or tied from
b1024 to 32768. Inside getrf n=128 the m=96 trailing gemm flips `reg` -> `wide` 64 at b >= 16384 (32.3 ->
64.6 ns/item), about a third of that driver's past-L2 loss.

## Batch sensitivity: what the batch axis buys the tables

Offline, over every `tuned/*.txt` with a batch key, batches >= 128, in-sample best choice per group:

| Batch distinction | Geomean loss | Cells > 10% | Cells > 25% | Worst |
| --- | ---: | ---: | ---: | ---: |
| none (one choice per shape) | 1.42% | 4.0% | 1.6% | 5.36x |
| one L2 bit (WS <= L2, WS > L2) | 0.83% | 2.5% | 1.0% | 4.47x |
| L2 bit plus a launch-floor bit (best t < 20 us) | 0.56% | 1.8% | 0.7% | 4.47x |
| the winner at the largest batch | 4.64% | 10.2% | 5.6% | |

- Largest batch-free losses: syev 3.64%, getri 3.19%, getrs 2.41%, trmm 2.05%, potrf 1.94%, getrf 1.92%;
  ormqr, symm and syr2k 0.00%. sm_89 (potrf only, converted rows) 0.40%.
- gemm series with a winner change across batch: 3676 of 5186; beyond the tie within WS > L2: 129 of 436.
- These losses are measured on warm tables, so they include C1, C4 and C5; the axis carries little that a
  coarse under-filled / one-wave / saturated class would not.

## Batch sensitivity: the shipped fixes

| Fix | Change | Evidence |
| --- | --- | --- |
| F3 ladder | the tuner samples batch at CU multiples (0.25, 2, 16, 128 x CUs: 47, 376, 3008, 24064 on sm_120); key stays `batch:log` | [Engine: the CU-aligned batch ladder](../design/tiered-tuning.md#engine-the-cu-aligned-batch-ladder) |
| F4a L2 chunking | blocked getrf and potrf factor the batch in parts of 75% of L2, rounded up to a CU multiple: getrf n=128 b32768 459.6 -> 383.0 ns (0.833), geomean 0.932 over 31 cells; potrf geomean 0.948 over 21 | [LU: L2 batch chunking](lu.md#lu-l2-batch-chunking), [potrf: L2 batch chunking](potrf.md#potrf-l2-batch-chunking) |
| F4b slices | level-3 and gemm natives run batch > 65535 as slices (`select::over_batch_slices`): syrk n=k=64 b65536 3481 -> 46.0 ns (76x) | [Level-3: the 65535 batch slices](level3.md#level-3-the-65535-batch-slices) |

## Batch sensitivity: measured dead ends

| Alternative | Result | Verdict |
| --- | --- | --- |
| Grid-stride / persistent grids (synthetic one-sub-group-per-item latency chain, persistent 188 x 12 work-groups) | equal to one-WG-per-item within 3% at B <= 2048 (18.7-19.3 vs 19.1-19.5 us); 15% slower at saturation (443 vs 386 us at B=65536), the tail fixed at the grid's granularity. Two sub-groups per item: 1.74x at B <= 1024 (19.1 -> 11.0 us), converging once B >= C | rejected as a general fix: it changes neither under-fill nor wave quantisation. Right only past the 65535 cap (done by slicing) and, untested, for the undersized `steqr_wg` grid |
| Batch-aware k-split leg for potrf `lpanel` (S = 2, 4 slices of the k loop per matrix; branch `bi-f5-lpanel-fix`, not merged) | against the route the table picks (`lpanel:panel=16` at n = 96, 128), best 1.14-1.18x at b128 (n=128 96.5 -> 82.5 us); 1.29-1.30x only against `panel=8` at n=128; n=48/64 3-12%; every split leg loses from B >= 752 (0.19-0.79x), S = 4 from 256-512 | fails the 1.3x gate. The column barrier chain (~29 of 36 us at n=64, three work-group barriers per column) is the lever, at every batch, so it belongs in the kernel; getrf / getrs / geqrf / latrd have the same chain |

## Batch sensitivity: the cold protocol

> **Status:** implemented, not yet measured end to end (work stopped 2026-10-23, threadripper02 under diagnostics).

`reset_and_run` (`tools/tune/cell_runner.hh`) runs reset, `bench::scrub_l2`, then the clock. The scrub is a
read-only scan of min(4 x L2, memory / 16) bytes leased from the Queue's arena (512 MiB, 0.338 ms per run on
sm_120), with a one-word sink. Warm-up, cut and verify runs go through the same path. Ledger records carry
`protocol=lean-cold`; a record of another protocol ranks below a cold one at any tier and is re-measured
(`plan_round`, reason `protocol`). Kernel hashes are untouched. `minibench.hh` structured mode and
`factor_bench` time one synchronous cold run per sample; legacy-mode minibench benchmarks (no `SetKernel`)
are not scrubbed yet.

`TuneRaceGpu.TimedRunsStartWithTheirOperandsOutOfL2` (gemm 32^3, batch 4096, `wide:m=32:n=32:k=16`): tuner
time 1.91-2.23x the back-to-back time; 1.00x with the scrub removed.

**Missing:** the old-vs-new tuner A/B (gemm 32^3 b1024-65536, gemm 128^3 b128-4096, gemv 256x1024
b32-1024) showing per-item time flat past saturation; `--plan` totals with the scrub priced in; the
cold retune (about 29 GPU-h with refinement, float + cfloat, gemm at `--cap-gib 32`).

## Batch sensitivity: timing hygiene

> **Status:** implemented and unit-tested; never run against real NVML in a worker.

- `--allow-idle-foreign` is gone: every GPU in `--devices` must hold zero foreign compute contexts.
- One SM-clock and clock-event-reason sample per timed run; ledger fields `clk_n`, `clk_capped`,
  `clk_mhz`, `throttled`, `throttle_reraces`. A cell with the SW power cap (0x4) active in more than 25%
  of samples is re-raced once.
- `sweep_to_table.py` flags cells whose whole-cell time drops more than 5x as batch grows at fixed other
  keys (`--check`, table writes; `--remeasure-out` writes a `--remeasure-keys` file).
- Not done, by decision: pipelined vs synchronous dual timing (it changes table semantics).

**Missing:** confirm `clk_*` fields in a real ledger.

## Batch sensitivity: open debts

- **Unfinished work (2026-10-23):** cold protocol and hygiene (above), and L2 chunking in syev
  ([syev](syev.md#syev-l2-batch-chunking)), geqrf/orgqr/ormqr ([qr](qr.md#qr-l2-batch-chunking)) and
  gesvd ([gesvd](gesvd.md#gesvd-l2-batch-chunking)): code and tests in, chunk rules unmeasured.
- **stedc deflation scan read device memory on the host** at merge order >= 512 (gesvd n=256 via bdsdc,
  syev n=512) and segfaulted on a `malloc_device` workspace; it now copies the kept widths to the host
  (`StedcTest.DeviceWorkspaceAtTheDeflationScanOrder`). The scan is still batch-global and syncs per merge level.
- **syev two_stage and stedc chunking:** not assessed by measurement. two_stage sizes every per-item array
  for the whole batch; `sytrd_sy2sb` buffer sizing calls `malloc_shared`/`free` per query.
- **The ~220 contaminated cells** (C5) need re-measurement before any batch-axis collapse is fitted.
- **sm_89 tables are untimed transcriptions.** Its 72 MB L2 moves the warm/cold edge to about half the
  sm_120 batch; no sm_120 batch window carries over.
- **Retunes owed:** getrf and potrf `blocked` rows (timed before F4a chunking); level-3 and gemm rows at
  batch >= 65536 (gemm 128^3 past 65535 is 7-12% slower than the vendor it used to fall to: a missing row).
- **F5 generalisation** (batch-aware lanes per item in getrf, getrs, geqrf, latrd) is not pursued.
- **Host floor (F6):** the vendor pointer-array wait and ~3.5 us of per-call host path remain.
