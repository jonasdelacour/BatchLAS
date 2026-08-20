# WP4 / GPU 1 — ncu on the native 128x128 GEMM, ld==rows vs strided ld

Shape: `float`, NN, m=128 n=1024 k=128, batch=512, beta=1, native route
(`GemmRegister128x128Kernel<float, true>`, grid 4096 blocks of 256 threads).
`BATCHLAS_BENCH_LD_PAD=0` (every ld == rows) vs `=384` (every ld == 512, the
leading dimension a trsm sub-view actually carries). All runs on GPU 1 via
`experiments/gpu_guard.sh`, `--launch-skip 3` so the profiled launch is warm.

Scripts: `ncu_full.sh`, `ncu_vendor.sh`, `ncu_bonly.sh` (profiles),
`timing.sh`, `discriminate.sh`, `per_operand.sh`, `b_stride.sh` (timing).

## The headline: the counters do not move. Only the time does.

| | native ld=rows | native ld=512 | cuBLAS ld=rows | cuBLAS ld=512 |
|---|---|---|---|---|
| duration | 917.3 us | **1493.1 us (1.63x)** | 869.4 us | 912.4 us (1.05x) |
| global LD requests | 2,097,152 | **2,097,152** | 6,291,456 | 6,291,456 |
| global LD sectors | 33,554,432 | **33,554,432** | 25,165,824 | 25,165,824 |
| LD sectors/request | 16.00 | **16.00** | 4.00 | 4.00 |
| global ST requests | 524,288 | **524,288** | 2,097,152 | 2,097,152 |
| global ST sectors | 8,388,608 | **8,388,608** | 8,388,608 | 8,388,608 |
| ST sectors/request | 16.00 | **16.00** | 4.00 | 4.00 |
| L1/TEX hit rate | 40.32 % | **40.55 %** | — | — |
| L2 hit rate | 46.49 % | **46.38 %** | 65.59 % | 65.57 % |
| DRAM read sectors | 17,826,616 | **17,826,372** | 17,826,480 | 17,827,000 |
| DRAM write sectors | 7,335,780 | 7,413,692 | 7,293,864 | 7,408,056 |
| shared ST bank conflicts | 4,194,304 | **4,194,304** | 2,097,152 | 2,097,152 |
| shared LD bank conflicts | 0 | **0** | 0 | 0 |
| registers / thread | 119 | 119 | 118 | 118 |
| local ld/st sectors (spills) | 0 | 0 | 0 | 0 |
| achieved occupancy | 31.67 % | 30.57 % | 32.12 % | 32.35 % |
| **DRAM throughput** | **89.28 % of peak** | **55.02 %** | 94.04 % | 90.02 % |
| memory throughput | 877.7 GB/s | 540.9 GB/s | 925 GB/s | 885 GB/s |

A per-SASS-instruction check across all 1000 instructions x 7 traffic counters
(`Instructions Executed`, `L1 Tag Requests Global`, `L2 Theoretical Sectors
Global`, `... Excessive`, `L1 Wavefronts Shared`, `... Excessive`, `L1 Conflicts
Shared N-Way`) found **0 differences** between the two runs.

DRAM traffic is provably minimal in both: 17,826,616 read sectors x 32 B =
570.4 MB, exactly the unique A+B+C footprint (33.5 + 268 + 268 MB) — every byte
is read exactly once, in both runs. 805 MB moved in 917.3 us is 877.8 GB/s,
which ncu puts at 89.28 % of this device's sustained DRAM peak (implied peak
983 GB/s). The strided run moves the same 805 MB in 1493.1 us: 540.9 GB/s,
55.02 %.

**This is not the trsm-symv signature.** There is no over-fetch that grows with
`ld`; nothing about the memory *traffic* changes at all. `ld` costs pure
latency that the kernel cannot hide.

## Where the time goes: warp stall reasons

Warp cycles per issued instruction 13.80 -> 22.89 (+9.09). Per-issue-active
stall ratios:

| stall reason | ld=rows | ld=512 | delta | share of +9.09 |
|---|---|---|---|---|
| **barrier** | 1.552 | **7.703** | +6.151 | **68 %** |
| **long_scoreboard** (L1TEX data) | 8.755 | **11.740** | +2.985 | **33 %** |
| not_selected | 0.949 | 0.798 | -0.151 | |
| short_scoreboard | 0.416 | 0.415 | ~0 | |
| dispatch_stall | 0.354 | 0.322 | -0.033 | |
| wait | 0.233 | 0.231 | ~0 | |
| mio_throttle | 0.110 | 0.069 | -0.041 | |

Eligible warps per scheduler 0.55 -> 0.29; active warps per scheduler
3.83 -> 3.74 (unchanged). Same concurrency, longer latency.

ncu's own verdicts flip accordingly: ld==rows is *"utilizing greater than 80 %
of the available compute or memory performance of this device"*; ld=512 is
*"below 60 % of peak typically indicate latency issues"*.

## SASS attribution: two instructions carry 101 % of the increase

Warp-stall samples, paired by instruction index (addresses differ between runs):

| p0 | p384 | delta | share | instruction |
|---|---|---|---|---|
| 15,404 | 76,706 | +61,302 | **63.7 %** | `LDS.128 R32, [R89.X4+UR8]` — all barrier stall (15,186 -> 76,530) |
| 46,125 | 81,955 | +35,830 | **37.2 %** | `STS.64 [R95.X4+UR8+0x8], R104` — all long_scoreboard (45,885 -> 81,818) |
| 610 | 2,556 | +1,946 | 2.0 % | `STS [R93.X4+UR7], R52` — long_scoreboard |

everything else < 0.5 % each. The k-loop body (idx 150-160) is:

```
150   LDG.E.128 R52, [R52.64]              B  : 16 L1 tag requests, 16 sectors, 0 excessive
151   LDG.E.64  R104, [R40.64+0x8]         A  : packet4 split in two (16 B alignment unprovable)
152   LDG.E.64  R102, [R40.64]             A
153   STS.64 [R95.X4+UR8+0x8], R104        A -> shared, 2-way bank conflict
154   STS.64 [R95.X4+UR8], R102            A -> shared, 2-way bank conflict
155   STS [R93.X4+UR7], R52                B -> shared transpose, stride 0x200, 2-way conflict
156   STS [R92+0x200], R53
157   STS [R92+0x400], R54
158   STS [R92+0x600], R55
159   BAR.SYNC.DEFER_BLOCKING 0x0          register_128x128.hh:227
160   LDS.128 R32, [R89.X4+UR8]            first fragment read; absorbs the deferred barrier wait
```

`DEFER_BLOCKING` means the barrier's stall is charged to the first shared access
after it, so the +61,302 samples on idx 160 *are* the CTA barrier wait. There
are exactly two `BAR.SYNC` per k-iteration (idx 159 and 708 =
`register_128x128.hh:227` and `:247`) and the shared tiles are sized
`TileK * AStride` / `TileK * BStride` only (`:133-134`) — the kernel is
**single-buffered**, so each iteration's global-load latency sits on the
critical path between the two barriers and cannot overlap with the 576 FFMAs.

## Which operand: B, by 96 %

`per_operand.sh`, pad applied one operand at a time (ms, batch=512, beta=1):

| padded | ms | GFLOPS | share of the 0.562 ms penalty |
|---|---|---|---|
| none | 0.9775 | 17,575 | — |
| A only | 0.9816 | 17,502 | 0.7 % |
| C only | 1.0327 | 16,636 | 9.8 % |
| **B only** | **1.5173** | **11,323** | **96.0 %** |
| A+B | 1.5217 | 11,290 | |
| B+C | 1.5380 | 11,170 | |
| A+B+C | 1.5398 | 11,157 | 100 % |

The ncu B-only profile (`full-b1-Bonly384`) reproduces the whole-operand result:
1450.6 us against 1493.1 us for all three, with the same counter signature.

Why B and not A: B is read as `packet4_ref(Bb + (k0+b_k) + (n0+b_n)*ldb)` with
`b_k=(tid%2)*4, b_n=tid/2`, so **one warp reads 32 B from each of 16 different B
columns** — 16 L1 tag requests per warp-instruction (measured: 8,388,608 / 524,288
= 16.0) against 4 for a coalesced load. Those 16 streams are `ldb*4` bytes
apart, so their spread in memory scales directly with `ldb`. A's warp reads 128
consecutive floats and is a single contiguous 512 B run whatever `lda` is.

## It follows the stride, not the footprint

`discriminate.sh` — footprint = allocated bytes, compare GFLOPS not ms:

| pad | batch | allocated | touched | GFLOPS |
|---|---|---|---|---|
| 0 | 512 | 570 MB | 570 MB | 17,571 |
| **0** | **2048** | **2280 MB** | **2280 MB** | **18,001** |
| 384 | 512 | 2280 MB | 570 MB | 11,173 |
| **384** | **128** | **570 MB** | **143 MB** | **11,159** |

4x the footprint at ld==rows costs nothing; the same footprint with a stride
costs the full 1.57x. Rules out TLB / allocation size.

`b_stride.sh` — B's ld alone, A and C at ld==rows (ms):

| padB | 0 | 4 | 8 | 32 | 64 | 128 | 132 | 256 | 384 | 392 | 448 | 896 | 904 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ms | 0.977 | 1.000 | 1.005 | 1.068 | 1.204 | 1.425 | 1.325 | 1.510 | 1.518 | 1.437 | 1.487 | 1.628 | 1.580 |

Monotone and continuous in the stride, from 512 B (padB=0) to 4096 B
(padB=896). The 128/132, 384/392 and 896/904 pairs straddle a power-of-two byte
stride and show only a 5-7 % extra penalty at the power of two — so
set/partition camping is a minor term, not the effect. Confirmed directly:
L2 slice sectors are even in both runs (p0 max/min 926,167/925,013; p384
925,699/922,127) and so are the DRAM channels (`fbpa__dram_sectors` max/min
54.22/54.20 % and 32.46/32.38 %).

## cuBLAS at the same stride

`ampere_sgemm_128x128_nn`: same 128x128 tile, same 4096-block grid, same 256
threads, 118 registers against our 119, 32 % achieved occupancy against our 31 %.
At the same stride it pays 1.05x, we pay 1.63x.

| | ld=rows | ld=512 |
|---|---|---|
| cuBLAS warp cycles / issued inst | 12.06 | 13.20 (+9.5 %) |
| cuBLAS **long_scoreboard** | 6.089 | **6.078 (flat)** |
| cuBLAS barrier | 2.526 | 3.507 (+0.98) |
| native warp cycles / issued inst | 13.80 | 22.89 (+66 %) |
| native long_scoreboard | 8.755 | 11.740 (+34 %) |
| native barrier | 1.552 | 7.703 (+396 %) |

cuBLAS's wait-for-global-data does not grow at all. It uses **17.664 KB shared
per block against our 9.216 KB** (occupancy limited by shared to 3 blocks, by
registers to 2 — so the extra shared memory is free) and issues 3x as many, 4x
smaller global requests. Both are the signature of a double-buffered / prefetched
main loop, which is exactly the latency slack our single-buffered loop lacks.

## ld-independent defects visible here (present equally in both runs)

These cost time but are *not* the `ld` defect — every counter below is identical
at pad 0 and pad 384:

* **A's packet4 global load is split into two `LDG.E.64`** (idx 151/152) because
  16-byte alignment of `Ab + (m0+a_m) + (k0+a_k)*lda` cannot be proven. Doubles
  A's L1 tag requests (8 per warp-instruction instead of 4). No extra DRAM
  traffic — the pair together fetches exactly 512 B per warp. This is the
  `derived__memory_l2_theoretical_sectors_global_excessive` = 8,388,608 sectors
  that ncu's UncoalescedGlobalAccess rule flags (20 % of 41,943,040); it is an
  artifact of the split, not real over-fetch.
* **A's shared staging store is 2-way bank conflicted**: `STS.64` x2, 2,097,152
  wavefronts against 1,048,576 ideal.
* **B's transposing shared store is 2-way bank conflicted**: 4 x 32-bit `STS` at
  0x200 stride, 1,048,576 wavefronts against 524,288 ideal.
* **All 32 `LDS.128` fragment reads are conflict-free** (excessive = 0). The
  `AStride = TileM` / `BStride = TileN` (no odd padding) choice at
  `register_128x128.hh:113-115` works exactly as its header comment claims.

## What could not be established

Why the *latency* of B's loads rises when the sector count, the L1 hit rate, the
L2 hit rate, the L2 slice distribution, the DRAM channel distribution and the
DRAM sector count are all unchanged. The remaining candidate is DRAM row-buffer
locality below L2, and **ncu exposes no row-activate counter**, so this was not
measured. `dram__cycles_active` is unchanged in absolute terms (8,387,465 vs
8,413,355) while `dram__cycles_elapsed` grows 1.63x — the DRAM is simply idle
45 % of the time, delivering the identical sectors.

## Measurement hygiene

All 37 timing cells quoted here have relative standard deviation <= 10% (max
observed 1.7%, `bs-392`); none needed rejecting. Every ncu profile used
`--launch-skip 3` so the profiled launch is past SYCL JIT, and every run went
through `experiments/gpu_guard.sh 1`, which confirmed GPU 1 was exclusive for
the whole run.
