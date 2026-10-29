# Batch-size sensitivity {#perf_batch_sensitivity}

> **Status:** current · RTX PRO 6000 Blackwell Max-Q (sm_120, 188 SMs, 128 MiB L2), threadripper02 GPU 2, CUDA 13.2, 2026-10-20/21; chunk rules and the cold-protocol A/B on RTX 4090 (sm_89), 2026-10-23

Why per-item time depends on batch, and what was done about it. Most of the "degrades past
saturation" signal was the L2-warm timing harness, not the kernels; the small-batch half is a real
one-work-group-per-item property. Grid-stride loops fix neither.

## Batch sensitivity: the culprits ranked

| Culprit | Regime | Status |
| --- | --- | --- |
| L2-warm timing in the tuner and minibench | past saturation | fixed in the harness ([cold protocol](#batch-sensitivity-the-cold-protocol)); no retune run under it |
| one work-group per item: under-fill and wave sawtooth | small batch | real; the batch-aware lanes prototype failed (dead ends) |
| ~6-8 us host floor per synchronous call | small batch | open (vendor pointer-array wait, ~3.5 us host path) |
| 300 W power-cap throttling (sm_120 Max-Q) | long large-batch cells | 1.07-1.25x; timing hygiene re-races capped cells |
| co-tenant time-slicing in ~220 sm_120 cells | isolated spikes | open: re-measure ([C5](#batch-sensitivity-c5-co-tenant)) |
| multi-kernel L2 hand-off loss in blocked drivers | past L2 | fixed by L2 chunking ([chunk rules](#batch-sensitivity-per-op-chunk-rules)) |
| 65535 grid cap on level-3 natives (syrk n=k=64: 79x cliff at b65536) | batch > 65535 | fixed by batch slicing ([gemm batch chunking](gemm.md#gemm-batch-chunking)) |

Under-fill, measured on sm_120 at b128: potrf n=64 `lpanel:8` fills 64 of 1504 resident slots (4%);
potrf n=16 `tiny` runs 30 ns/item against 0.75 saturated. Families with different capacity and
single-item latency therefore swap winners as batch crosses one wave, which is what the batch axis of
the tables genuinely encodes. The tuner samples batch at CU multiples for that reason
([the CU-aligned batch ladder](../design/tiered-tuning.md#engine-the-cu-aligned-batch-ladder)).

## Batch sensitivity: C5, co-tenant contamination of sm_120 cells {#batch-sensitivity-c5-co-tenant}

The sm_120 deep run used the since-removed `--allow-idle-foreign` while another user's process held a
context on every GPU. Time-slicing adds a fixed ~0.17-0.2 ms or ~2.27-2.29 ms per synchronising call,
about 220 of ~100k cells (none on sm_89). Example: getrf float n=32 `tiny` reads 0.214 ms in the
table at b64, 21.8 us re-measured alone; a second process looping gemm reproduces the 2.35 ms median.
Routing reads contaminated getrf rows at b32-128 and b8192-131072.

## Batch sensitivity: per-op chunk rules {#batch-sensitivity-per-op-chunk-rules}

`tuning::blocked_chunk_items` splits a blocked driver's batch into parts of the op's percent of L2
(`*_L2_CHUNK_PERCENT` in `include/batchlas/tuning_params.hh`, rounded up to a CU multiple; 0 = no
chunking); `BATCHLAS_TUNE_L2_CHUNK_PERCENT` and `_ITEMS` override every op. Part loops cap parts at
`max_groups_yz / 2`. Ship criterion: \f$\le 2\%\f$ loss in every measured cell, nested calls included.
RTX 4090, cold, 2026-10-23 (time ratio chunked / unchunked):

| driver | percent | best cell | worst cell at the shipped rule | page |
|---|---:|---|---|---|
| getrf blocked | 75 | float n=128 b16384 0.787 | n=64 b16384 0.907 | [LU](lu.md#lu-l2-batch-chunking) |
| potrf blocked | 75 | cfloat n=128 b4096 0.876 | float n=64 b16384 1.000 | [potrf](potrf.md#potrf-l2-batch-chunking) |
| syev cta | 150 | float n=16 b16384 0.911 | float n=16 b65536 1.026 | [syev](syev.md#syev-l2-batch-chunking) |
| gesvd blocked (real) | 150 | n=256 b1024 0.712 | n=128 values b16384 1.021 | [gesvd](gesvd.md#gesvd-l2-batch-chunking) |
| syev blocked | 0 | (75 %: up to 1.70x slower) | | [syev](syev.md#syev-l2-batch-chunking) |
| geqrf, orgqr blocked | 0 | (75 %: up to 1.87x slower) | | [QR](qr.md#qr-l2-batch-chunking) |
| ormqr blocked | 0 | (150 % alone: 0.83; inside orgqr: 1.48x) | | [QR](qr.md#qr-l2-batch-chunking) |

Harness: one process per (cell, arm), `bench::scrub_l2`, one synchronous call timed, 5-7 reps,
median; 3-5 passes with alternating arm order, pass 0 dropped. No raw data committed.

## Batch sensitivity: measured dead ends

| Alternative | Result | Verdict |
| --- | --- | --- |
| Grid-stride / persistent grids (persistent 188 x 12 work-groups, synthetic per-item latency chain) | within 3% of one-WG-per-item at B <= 2048; 15% slower at B=65536 (443 vs 386 us) | rejected as a general fix: it changes neither under-fill nor wave quantisation. Right only past the 65535 cap (done by slicing) and, untested, for the undersized `steqr_wg` grid |
| Batch-aware k-split leg for potrf `lpanel` (S = 2, 4; branch `bi-f5-lpanel-fix`, not merged) | best 1.14-1.18x at b128 against the tabled route (n=128 96.5 -> 82.5 us); every split leg loses from B >= 752 | fails the 1.3x gate. The column barrier chain (~29 of 36 us at n=64) is the lever at every batch, so it belongs in the kernel; getrf / getrs / geqrf / latrd share it |

## Batch sensitivity: the cold protocol

`reset_and_run` (`tools/tune/cell_runner.hh`) restores the operands, runs `bench::scrub_l2`
(`include/batchlas/util/bench_structured.hh`: a read-only scan of min(4 x L2, memory / 16) bytes
leased from the Queue's arena), then starts the clock. Warm-up, cut and verify runs take the same path.
Ledger records carry `protocol=lean-cold`; a record of another protocol ranks below a cold one and is
re-measured (`plan_round`, reason `protocol`). `minibench.hh` structured mode and `factor_bench` time
one synchronous cold run per sample; legacy-mode minibench benchmarks (no `SetKernel`) are not scrubbed.
`TuneRaceGpu.TimedRunsStartWithTheirOperandsOutOfL2` guards it (tuner time 1.91-2.23x back-to-back
with the scrub, 1.00x without).

Why: warm timing is fastest while batch x footprint fits L2 and "degrades" past it. sm_89, Auto, float,
ns per item, warm (old protocol) -> cold: gemm 32^3 b4096 10.7 -> 16.7 and b65536 14.1 -> 13.8;
gemv 256x256 b128 129.5 -> 353.1 and b1024 291.5 -> 289.0. On sm_120, 92.5% of gemv table cells with
L2/2 < WS <= L2 exceed the DRAM peak (impossible from DRAM). Factorizations are nearly immune (<= 7%,
getrf n=32 up to 1.23x).

**Missing:** `--plan` totals priced from a real cold run; the cold retune (about 29 GPU-h on sm_120,
float + cfloat; commands in `tools/tune/README.md`).

## Batch sensitivity: timing hygiene

> **Status:** implemented and unit-tested; never run against real NVML in a worker.

- Every GPU in `--devices` must hold zero foreign compute contexts (`--allow-idle-foreign` is refused).
- One SM-clock and clock-event-reason sample per timed run; ledger fields `clk_n`, `clk_capped`,
  `clk_mhz`, `throttled`, `throttle_reraces`. A cell with the SW power cap active in more than 25% of
  samples is re-raced once.
- `sweep_to_table.py` flags cells whose time drops more than 5x as batch grows at fixed other keys
  (`--check`; `--remeasure-out` writes a `--remeasure-keys` file).
- Not done, by decision: pipelined vs synchronous dual timing (it changes table semantics).

**Missing:** `clk_*` fields in a real ledger. The 2x4090 box cannot provide one: both GPUs hold
foreign contexts (desktop, another user's job), so the tuner runs only on a headless GPU.

## Batch sensitivity: open debts

- **Chunk rules are sm_89 measurements** (sm_120 only for getrf/potrf). A per-caller ormqr rule
  (chunk only a top-level call) would keep its 0.83-0.95 without orgqr's 1.48x; complex gesvd chunking
  is unmeasured.
- **Range rounding past the grid** ([defect 21](../design/known-defects.md#defect-21-range-rounded-past-the-grid)):
  the part loops are capped; syev two_stage, band_reduction and `transpose` are not.
- **stedc deflation scan** is batch-global and syncs per merge level.
- **syev two_stage and stedc chunking:** not measured. two_stage sizes every per-item array for the
  whole batch; `sytrd_sy2sb` buffer sizing calls `malloc_shared`/`free` per query.
- **The ~220 contaminated cells** (C5) need re-measurement before any batch-axis collapse is fitted.
- **sm_89 tables are untimed transcriptions**; its 72 MB L2 moves the warm/cold edge to about half the
  sm_120 batch.
- **Retunes owed:** getrf and potrf `blocked` rows (timed before chunking); level-3 and gemm rows at
  batch >= 65536 (gemm 128^3 past 65535 is 7-12% slower than the vendor it used to fall to).
- **Host floor:** the vendor pointer-array wait (`MatrixView::init_data_ptr_array`, +5-11 us per
  batched vendor call) and ~3.5 us of per-call host path remain.
