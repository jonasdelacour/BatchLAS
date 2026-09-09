# Native batched factorizations at n = 4..512: research digest and execution plans

Status: research + plan, nothing executed. Written 2026-09-10 against
`readiness/wp3-abi` (aca64f6). Supersedes the WP6 entry of
`docs/design/readiness-remediation-plan.md`, which assumed WP6 was
measurement work; it is not. Every number below that is not marked
"literature" or "arithmetic" was measured on this box (2x RTX 4090, sm_89)
and lives either in `docs/perf/*.md` or at tag `perf-evidence/vendor-independence`
(`git show perf-evidence/vendor-independence:experiments/<path>`).

Ratios are `vendor_ms / native_ms`; **>1 means native wins**. "Large batch"
means the saturating batch for the order (>= 2048 at n <= 64, >= 512 at n = 512).

---

## 0. Verdict in one page

**Where native stands today, n = 4..512, large batch, vendor build:**

| op | float | cfloat | double | cdouble | what the vendor does |
|---|---|---|---|---|---|
| potrf | wins only n <= 48 (2.75x at 8, 1.85x at 32, 1.26x at 48); **0.36-1.00x at 64..256**; unrouted | 1.79x at 8, then 0.40-1.05x | 0.58-1.07x | 0.57-1.47x | cuSOLVER `potrfBatched`, one fused kernel at small n |
| getrf | **0.55x at 32**, 1.61x at 64, 0.73-0.87x at 128, 1.25-2.35x at >= 256 (routed) | 0.60x at 32, 0.83 at 64, 0.38-0.75 at 128, 1.6x at 512 | 0.27-0.75x everywhere | 0.34-0.75x | cuBLAS `getrfBatched`: 1.1 TFLOP/s at float n=32, ~90% of FP64 peak at 512 |
| geqrf | **0.78x at 32**, then 2.0-12x at 64..512 (unrouted) | 0.71x at 32, 2.3-8.9x above | **0.21x at 32, 0.53x at 64**, 1.6-6.8x at >= 128 | 0.33-0.84x to 256, 1.41x at 512 | cuBLAS `geqrfBatched`: ~380 GFLOP/s float at every n, unblocked |
| orgqr | 3.8-123x | 2.3-65x | 6-114x | 1.7-12x | a per-item cuSOLVER loop; not a real comparison |
| getrs (nrhs<=2) | 1.7-2.5x | 1.3-1.6x | 2.0-3.8x | 1.8-2.6x | cuBLAS `getrsBatched`; the fused tier is routed and wins |
| trsm order<=32 | 1.6-3.6x | 1.0-3.8x | 2.4-9.6x | 2.3-14x | complex "vendor" is our own fallback |
| gemm square<=32 | 1.03-1.46x (NN) | no register kernel | 1.2-4.5x | no register kernel | cuBLAS |

**The diagnosis is one sentence.** Every native factorization kernel in the
tree keeps the whole matrix in local memory, one work-group per matrix, with a
runtime `n` and a level-2 (rank-1) recurrence inside; above ~48 KB of local
memory that allows 1-2 resident work-groups per SM (`sm__warps_active` = 8.3%
for potrf at n = 96..155, 2.7% of FP32 peak), and below ~32 it wastes most of
a warp on a matrix that a single sub-group could hold in registers. The
blocked drivers lose at n <= 256 because their leaf is that kernel, not
because of their trailing GEMMs (65-95% of time only at n >= 512).

**What the literature and MAGMA's current source say, unanimously:**
(1) thread-per-row, matrix or panel in **registers** with a compile-time
column count and fully unrolled loops; (2) local memory only as a
communication scratchpad, never the store; (3) **several matrices per
work-group** when n < 32; (4) lazy row swaps by relabelling; (5) a
left-looking, 8-wide fused panel for Cholesky up to n ~ 160-224 with no
BLAS-3 at all; (6) a fused register panel (m x 32, thread-per-row) as the LU
and QR leaf up to m = 512; (7) BLAS-3 blocking only above ~192-256, and there
the trailing GEMM on small-k shapes is where vendor GEMM is weakest.
None of (1)-(6) exists in BatchLAS today. The one experiment that looked like
(3) (`wp6_perf/proto/pivsg.cpp`) kept the matrix in local memory with a
runtime `n` and lost; it did not test (1).

**The ideas, ranked by (expected gain x confidence) / effort:**

| # | idea | band | target (float, vs vendor) | depends on |
|---|---|---|---|---|
| P0 | In-tree harness + baseline + the zero-engineering routing flips | all | realises geqrf 2-12x, orgqr, at n >= 64 today | - |
| P1 | Register-resident sub-group kernels, packed, template N in {8,16,32}: potrf, getrf, geqrf | n <= 32 | potrf >= 3x, geqrf >= 3x, getrf >= 1.3x (vs 0.55x) | P0 |
| P2 | Fused factor+solve for tiny systems (gesv, posv) behind `linalg::solve` | n <= 32, nrhs <= 4 | >= 3x vs getrf+getrs | P1 |
| P3 | Left-looking 8-wide fused Cholesky panel ("lpout") as the potrf leaf, single launch, thread-per-row | 32 < n <= 192 (leaf up to 512 via blocked) | >= 1.2x at 64..256 (vs 0.36-1.00x) | P0 |
| P4 | Register panel LU leaf (m x 32 in registers, lazy swap) + recursive panel + row-parallel laswp | 32 < n <= 512 | >= 1.0x at 64..128 float/cfloat (vs 0.73-0.87x), keep 1.6-2.3x above | P0, P7 |
| P5 | Fused QR panel + fused reflector-apply (no T, no larft) for n <= 128; same kernel is orgqr/ormqr small | 32 < n <= 128, all tall panels | double 0.5x -> >= 1x at 64; larft 20-49% removed | P0 |
| P6 | Transposed and complex wide-scalar register GEMM for the panel-update shapes | 128 < n <= 512 | potrf/geqrf cfloat/cdouble 0.4-0.9x -> >= 1x | P0 |
| P7 | Occupancy-aware capacity rule + G-packing for every CTA kernel + `native_tier_preferred` for potrf | all | removes the 1-2 blocks/SM cliff; cheap | P0 |

Double and cdouble: FP64 runs at 1/64 of FP32 on this card, so any
compute-bound native kernel loses to cuBLAS's 90%-of-peak FP64 code. Double
wins are only possible where the kernel is memory-bound: P1/P2 (n <= 32), the
fused getrs tier (already 2-3.8x). P3-P5 are float/cfloat targets; double is
measured but not gated.

Dependency graph: P0 -> {P1, P3, P5, P6, P7}; P1 -> P2; P7 -> P4; P6 lifts P3/P4/P5 at n >= 256.

---

## 1. Evidence base

### 1.1 Hardware constants used in the arithmetic below (RTX 4090, sm_89)

| quantity | value | source |
|---|---|---|
| SMs | 128 | `docs/perf/README.md` |
| registers / SM; hard launch limit | 65,536; `regs x WG <= 65536` | register probe gate |
| max resident threads / SM; warps; blocks | 1536; 48; 24 | CC 8.9 (literature note; verify with `ncu` occupancy) |
| local memory / work-group (opt-in) | 101,376 B; budget used 97,280 B | `potrf_cta.cc` |
| 48 KB launch hole | dynamic request in (47,104, 49,664] fails cold | `docs/perf/potrf.md` |
| DRAM | ~1008 GB/s theoretical, ~950 achievable | gemv page |
| L2 | 72 MB, ~5 TB/s | chipsandcheese |
| FP32 (measured cuBLAS SGEMM) | ~47 TFLOP/s | gemm page |
| FP64 | ~1.29-1.44 TFLOP/s (1/64) | gemm page |
| device link (the .so is the SYCL link unit) | 117-125 s per touched library | build memory |

Occupancy arithmetic: 100% occupancy needs <= 42 registers/thread; 64
registers -> 1024 threads (67%); 96 -> 640 (42%); 128 -> 512 (33%). A
one-sub-group work-group caps at 24 blocks = 24 warps = 50% occupancy, so
tiny-n work-groups must hold >= 2 sub-groups (G-packing) to reach 100%.

DRAM roofline per matrix, read + write once, float: n=8 512 B, n=16 2 KB,
n=32 8 KB. At 950 GB/s: n=32 float, batch 8192 -> 67 MB -> **70 us**.
Vendor at that cell: cuBLAS getrf 159 us (2.3x above roof), cuBLAS geqrf
772 us (11x above roof), cuSOLVER potrf ~ (CTA is already 1.85x faster
than it). These are the ceilings P1 is measured against.

### 1.2 The existing native kernels (what P1-P5 replace)

| kernel | file | design | ceiling | registers | loss mechanism |
|---|---|---|---|---|---|
| potrf CTA | `src/extensions/potrf_cta.cc`, `potrf_cta_device.hh` | whole matrix in SLM (`lda = n\|1`), NB=8 panel, TS=4 register tiles, L=32..256 work-items, G<=4 matrices per WG only at L=32 | 155/109/109/77 | 64/94/102/128 | SLM footprint: 1-2 blocks/SM above n~90; 8.3% warps active |
| potrf blocked | `potrf_blocked.cc` | right-looking, nb 128/96/96/64, leaf = CTA, routed trsm + gemm (W-panel + fold) | Lower only | - | leaf; complex trailing gemm on `Tiled16` |
| getrf CTA | `getrf_cta.cc`, `getrf_cta_device.hh` | whole matrix in SLM (odd ld), rank-1 right-looking, XOR-butterfly + 32-slot scan argmax, wg 64..512 | 155/109/109/77 | - | 3 WG barriers per column; wg >= 64 for a 32-column matrix; SLM cliff |
| getrf blocked | `getrf_blocked.cc`, `lu_laswp.hh` | nb=32, leaf = same body (resident or global), deferred left gather, routed trsm/gemm | - | - | 4 kernels per block step vs cuBLAS's one fused kernel; double has no register GEMM |
| geqrf CTA | `geqrf_cta.cc`, `geqrf_cta_device.hh` | whole m x n tile in SLM, `?GEQR2` with two `reduce_over_group` per reflector, wg = 32 x teams(n) ignoring m | area 24,320 elems (float) | - | reduction over 256 lanes for a 64-element column (193 idle); static shared -> 48 KB hole pad; FP64 rate |
| geqrf blocked | `geqrf_blocked.cc`, `larft_wy.hh` | right-looking WY, nb 32 (16 double), larft serial on lid 0, three routed gemms | - | - | `V^H A22` on `Tiled16` for complex/transposed float; larft 15-49% |
| orgqr | `orgqr_blocked.cc` | identity fill + ormqr + copy | - | - | 1.5x flops, fill+copy 11-22% |
| getrs fused | `getrs_fused.cc` | one WG per matrix, RHS + nb x nb block resident, sub-group shuffle recurrence, nrhs <= 8 | - | 39-86 | wins; the model for P2 |
| trsm V1 | `src/sycl/trsm_native.cc` | one work-item per solve, `x[N]` registers, N in {8,16,32} compile-time | 32 | 114/153/148/226 | wins; proves the compile-time-N register idiom builds and links here |

Things already measured DEAD that the plans below must not repeat
(full list: `docs/perf/{potrf,lu,qr,trsm}.md`): potrf NB=16 in the
SLM-resident kernel (156-206 registers, 2x slower); `reduce_over_group` over
the work-group anywhere (48 KB hole, 1.5-4.7x slower for FP64); packed
sub-group getrf with the matrix in SLM; cooperative multi-lane trsm solve;
bespoke potrf panel-solve kernel; launch-count fusion (<= 3%); nb=128 for
geqrf; `Tiled16` double-buffering and B-packing for strided ld.

### 1.3 Literature consensus (13 papers, text-extracted; digest in the session transcript)

| band | QR | Cholesky | LU |
|---|---|---|---|
| n <= 16 | fully fused, template N, thread-per-row, **serial** norm/`v^H A` reductions, 4-16 matrices per warp/block (2.0-3.8x from packing alone) | unblocked in registers, thread-per-row, packing (1.2-4.3x) | registers + butterfly argmax + lazy swap (+21-32%), packing (2.0-3.5x) |
| 16 < n <= 32 | one warp per matrix; A in registers, `v^H A` through a padded N x N local scratch (beats shuffles: Q x log2 N shuffle steps vs P local steps) | registers beat local at every n to 32 | same as above; unrolled pivot search worth 2x over a generic one |
| 32 < n <= 128 | fused register panel (m x nb, nb 8-16 compile-time) + fused reflector-by-reflector trailing update, **no T** (removes 14-45% auxiliary kernels) | left-looking blocked: panel N x NB in registers, left stripes streamed; NB = N/2..N/3; fast sqrt/rcp +28% | fused register panel m x nb (1.5-4.8x over a column kernel); memory-bound kernels are >80% of time at n=50 |
| 128 < n <= 512 | LAPACK-style with nested blocking, T via one `V^H V` gemm; GEMM 56-74% of time | right-looking nb=32 outer + recursive left-looking panel; native syrk writing one triangle 4x a gemm-based one | recursive panel (32-wide leaves), row-parallel laswp; vendor GEMM 1.4-3x below a tuned native one on (m x 32)(32 x n) shapes |

Recurring traps: 2-D thread-per-element mappings (~10x slower at small n);
local memory as the store; shuffles for `v^H A`; sub-warp work-groups
without packing; runtime dimensions in the inner loop; power-of-two
staircases (benchmark n = 2^k + 1); mis-aligned sub-matrix views.

Credible speedups on Ada float vs a modern vendor (papers' numbers discounted
for their old cuBLAS baselines): 3-8x at n <= 16, 2-4x at 17..32, 2-4x at
33..64, 1.5-3x at 65..128, 1.3-2x at 129..256 (GEMM-bound), 1.1-1.5x at
257..512.

### 1.4 MAGMA master (2026) as the concrete reference

Read from source at `/home/jonaslacour/.claude/jobs/888a35a4/tmp/magma/`
(fetched from `icl-utk-edu/magma`); the mapping table:

| op | n <= 32 | 32 < n <= crossover | above |
|---|---|---|---|
| potrf | `lpout`: n/8 launches, `threads(n-j, ntcol)`, thread-per-row, panel `(m+8) x 8` in shared, `rp/rA/rC[8]` registers, left-looking update fused with one-step-ahead prefetch; ntcol 8/4 (s), 4/2 (d), 1 (c/z) for rows <= 32/64 | same, to crossover s 192 / d 160 / c 160 / z 224 | block left-looking gemm + recursive panel (nb 64/32 s) + `trsm_recursive` |
| getrf | `smallsq_noshfl<N,pow2>`: `threads(pow2(n), ntcol)`, row in `rA[N]`, `__syncwarp` only, serial argmax over a shared `dsx[N]`, lazy swap via `rowid` | `getf2_fused<32>`: `m` threads, `rA[32]`, shared argmax tree, lazy swap; m <= 512 (s/d), 384 (c), 256 (z); recursive panel nb 128 (s/d/c) 64 (z) -> 32-wide leaves; row-parallel laswp gather | classic column kernels |
| geqrf | `sq1d_reg<N>`: 128-thread block, 128/N matrices, `rA[N]`, `v^H A` via elementwise product in padded shared then one-thread-per-column sums; 4 barriers per column; z limited to N<=16 | `geqr2_fused_reg<M32,N<=16>` thread-per-row, norm by shared tree, `v^H A` by 16-threads-per-column groups; + `unm2r_reg` fused update (no T); nb probed 16,8,4,2,1; up to m=1024 | nb=32 `geqr2` + `larft_sm32x32` + larfb as three gemms |
| gesv nrhs=1 | `gesv_small<N>`: LU + forward substitution fused into elimination + back substitution, all warp-synchronous | shared variant to 53/60 | getrf + laswp + 2 trsm |

Design idioms to carry over verbatim: (1) `rA[N]` thread-per-row with a
`switch(n)` instantiating the template; (2) lazy swap by relabelling
(`rowid`), pivot row broadcast through a 32-element scratch, permutation
applied at write-out; (3) fused left-looking panel with prefetch register
that *becomes* the panel; (4) `ntcol` matrices per block for `m < 32`;
(5) `SLDA` odd/`n+1` padding for the shared scratch; (6) `pivinfo`
composition + one-pass gather laswp; (7) precision divergence only in
constants (`MAX_ROWS`, N ceilings, `ntcol`).

Notable: MAGMA uses **no shuffles** in any factorization kernel; every
reduction goes through shared memory. On Ada with SYCL sub-group
collectives the natural port replaces the shared trees with
`sub_group` reductions plus one shared combine, which is what
`getrf_cta_device.hh` already does for its argmax.

---

## 2. Design rules for every plan below

R1. **Registers are the store; local memory is the mailbox.** A kernel's
local-memory footprint per matrix must be O(N) or O(N x NB), never O(N^2)
for N > 32. Target >= 4 resident work-groups per SM by local memory
(<= 24 KB per work-group) and >= 8 sub-groups per SM.

R2. **Compile-time column count.** The register array is indexed only by
template constants; loops over it carry `#pragma unroll`. Ladder
N in {8, 16, 32} (pad the order up, mask lanes/rows); add 24 only if the
n = 17..24 band shows a cliff. Rows (`m`) stay runtime.

R3. **Sub-group is the synchronisation unit for n <= 32.** One matrix
never spans a sub-group; `group_barrier(sg)` only; G = 32/N_pad matrices per
sub-group at N_pad < 32 and >= 2 sub-groups per work-group. No work-group
barrier in the n <= 32 kernels at all.

R4. **No `reduce_over_group` over a work-group.** Sub-group collectives
(`reduce_over_group(sg, ...)`, `permute_group_by_xor`, `select_from_group`,
`shift_group_left`) plus a scan over one local slot per sub-group. Verify
with ptxas that static shared stays 0 (the 48 KB hole).

R5. **Lazy swap.** LU never moves a row between lanes; `rowid` relabelling,
pivot row through a 32-element local scratch, permutation applied at the
final store. `ipiv` written 1-based LAPACK-style.

R6. **Read once, write once.** Every fused kernel touches global memory
exactly twice per element it owns; trailing panels are streamed, not
re-read per reflector.

R7. **Instantiation budget.** The .so is the device-link unit. Each plan
states its instantiation count (kernels x types) and must measure the
device-link delta of the touched library; > 15% needs a written
justification. Runtime `m`, template only what must be in registers.

R8. **Measure as the repo prescribes** (`docs/perf/README.md`): one harness
on the box (`gpu_guard.sh`, device 1), JIT warmed (>= 1.5 s), interleaved
A/B medians of >= 7 reps, relative sd < 10% or the cell is discarded and
named, host residual on items 0 and batch-1 of every timed row, saturation
ladder with both readings, a bracketing non-winner at every proposed edge,
`route_diff` before/after, ratios in time never GFLOPS, nothing under
`BATCHLAS_KERNEL_TRACE`, bench binaries rebuilt after any `preferred()`
change. A `preferred()` flip needs `t_native <= 0.90 t_vendor` at
saturation, no accuracy regression, and an end-to-end caller A/B.

R9. **Every new test is armed**: the plan lists the deliberate break that
turns it red, and the PR shows it observed red.

R10. **Double is not gated.** Float and cfloat cells gate a plan; double
and cdouble are measured and reported, routed only where they win.

---

## 3. Execution plans

Each plan is written to be executed by a subagent to completion. Common
hand-off rules: work in a worktree; never build with more than `-j4` from
cold; build only the touched library targets plus the tests named; run
`ctest -L blas -R '<op>'` during iteration and the full suite once before
the PR; diff failing names against `tests/known-failures.txt`; run
`scripts/register_probe.sh` (frame 0, spill 0, `regs x WG <= 65536`) and
`scripts/route_diff.sh capture|compare` for any routing change; commit the
sweep CSVs under `benchmarks/results/`; no scratch files under `.claude/`.

### P0. Harness, baseline, and the free routing flips

**Goal.** An in-tree, correct, saturation-aware harness for potrf / getrf /
getrs / geqrf / orgqr, a committed baseline grid, and the `preferred()`
flips the existing evidence already supports.

**Why first.** There is no potrf, getrf or getrs harness in `benchmarks/`;
every LU and Cholesky number came from standalone programs under
`experiments/` at the tag. `benchmarks/geqrf_benchmark.cc` has the flop
count with the wrong sign, float/double only, and no residual check
(`docs/perf/qr.md`). Every later plan needs the same harness and the same
baseline, and P1-P7 are gated on numbers it produces.

**Deliverables.**
1. `benchmarks/factor_bench.cc` -> binary `factor_bench <op> <type> <m> <n> <nrhs> <batch> <reps> [--route=...] [--csv=]`
   with `op in {potrf, getrf, getrs, gesv, geqrf, orgqr, ormqr}` and
   `type in {float, double, cfloat, cdouble}`. Port the proven pieces from
   `experiments/wp4_potrf/phase2_ab/realpotrf.cpp`, `wp6_lu/bench/lubench6.cpp`,
   `wp5_qr/bench/qrbench.cpp` (all at the tag): time-based warm-up
   (`WARM_S`, default 1.5 s), interleaved vendor/native reps in one
   process, medians + rel-sd, host residual on items 0 and batch-1
   (`||A - L L^H||/||A||`, `||P A - L U||/||A||` plus elementwise pivot check
   against LAPACKE, `||A - Q R||/||A||` and `||Q^H Q - I||`), NaN-propagating
   `nanmax`, the resolved route printed per row (read back through
   `resolve_*_route`, never assumed), and a `BAD` flag when any gate fails.
   Inputs: SPD = `A A^H + n I`; LU = diagonally dominant then row-permuted;
   QR = random Gaussian, plus one ill-conditioned column set for the
   orthogonality check. Strided contiguous batches at `ld = m` and one
   `ld = m + 7` row per op (the strided-ld trap).
2. `benchmarks/run_factor_grid.sh <op> <type> <out.csv>` running the
   canonical ladder: orders {4, 8, 9, 16, 17, 24, 32, 33, 48, 64, 65, 96,
   128, 129, 192, 256, 384, 512} x batch schedule from
   `docs/perf/lu.md` (`SAT_LADDER`: 32:[512..16384], 64:[512..16384],
   128:[256..8192], 256:[128..4096], 512:[64..1024]); adds 4..24 at
   [4096..65536]. Wrapped in `experiments/gpu_guard.sh` semantics (port the
   script into `benchmarks/gpu_guard.sh`: wait for idle, pin
   `CUDA_VISIBLE_DEVICES=1`, re-check foreign PIDs, exit 5 on contamination).
3. `benchmarks/results/factor_baseline_<op>_<type>.csv` for all five ops
   and four types, plus `docs/perf/small-n-baseline.md` with the ratio grid
   (this document's section 0 table, re-measured, one file).
4. The flips already supported by evidence, one PR each, each with the
   R8 package (grid CSV, bracketing non-winner, `route_diff`, failing-name
   diff, end-to-end A/B):
   - geqrf `preferred()`: float and cfloat `n >= 64` (CTA to the tier
     crossover, Blocked above); double `n >= 128`; cdouble `n >= 512`.
     End-to-end A/B: `ortho_benchmark` and `syev_benchmark` at n = 512
     (the callers that issue tall panels). Tall-panel cells 128x32, 512x32,
     512x64 are in the grid (float 1.57/2.11/2.80x).
   - orgqr `preferred()`: native everywhere `n <= 512` for every type
     (the vendor arm is a per-item loop; no cell in-window loses).
   - potrf `native_tier_preferred`: declare it (today the static order picks
     CTA below the ceiling on one data point); CTA `n <= 48` float,
     Blocked above, until P3 replaces both.
5. `benchmarks/CMakeLists.txt` registers `factor_bench` in
   `batchlas_benchmarks`; `docs/perf/README.md` gets a "factorization
   harness" paragraph.

**Tests.** `tests/factor_bench_selftest.cc` (label `tools`): feeds a
deliberately corrupted factor (flip one element of L) and asserts the
harness reports `BAD`; feeds a route pin that cannot take and asserts the
printed route is the fallback, not the pin. Armed break: remove the
residual gate; the corrupted-factor test must go red.

**Acceptance.** All 20 baseline CSVs committed with rel-sd < 10% on every
kept row; the ratio grid reproduces the section 0 table within 15% at every
cell that has a prior number (larger drift is investigated, not accepted);
the three flips each meet R8. Device-link delta 0 (no library touched).

**Effort.** 2-3 agent-days (harness 1, grids 1, flips 1). Blocks everything.

---

### P1. Register-resident sub-group kernels for n <= 32 (potrf, getrf, geqrf)

**Goal.** Replace the CTA route below n = 32 for all three factorizations
with kernels that hold one matrix per sub-group partition entirely in
registers, pack G matrices per sub-group and >= 2 sub-groups per
work-group, and synchronise only at sub-group scope.

**Targets** (float, large batch, vs the vendor route): potrf >= 3.0x at
n in {8,16,32} (CTA is 2.75/1.88/1.85x today with the wrong design);
geqrf >= 3.0x at n = 32 (0.78x today; vendor is 11x above the DRAM roof);
getrf >= 1.3x at n = 32 (0.55x today; vendor is 2.3x above roof) and
>= 2x at n <= 16. cfloat: same targets at 0.8x the float ratio. double and
cdouble reported; the tiny band is memory-bound so wins are expected but
not gated.

**Design, shared by the three kernels.**
- Template `<T, int N>` with `N in {8, 16, 32}`; runtime `n <= N`; rows
  `n < N` are padded to the identity (Cholesky, LU: `A(i,i) = 1`, zeros
  elsewhere; QR: zero rows and identity columns) in registers at load, so
  no lane guards exist in the unrolled body. `n = 4..8 -> N = 8`,
  `9..16 -> 16`, `17..32 -> 32`.
- Lane `r` of an `N`-lane partition (`SubGroupPartition<N>` from
  `src/extensions/sg_compat.hh`, already used by every `*_cta` eigen kernel)
  owns row `r`: `T rA[N]` in registers. G = 32 / N matrices per sub-group.
- Work-group = `S` sub-groups, `S = 4` (128 work-items, 4G matrices);
  `nd_range<1>(ceil(batch / (4G)) * 128, 128)`, `[[sycl::reqd_sub_group_size(32)]]`,
  `reqd_work_group_size(128)`. Local memory per work-group: the
  communication scratch only (below), so >= 8 work-groups per SM by memory.
- Load: lane `r` reads `A(r, 0..n-1)` with stride `ld` (column-major, so
  the 32 lanes read 32 consecutive elements of each column: coalesced per
  column). Store the same way. Complex is loaded as two reals per element.
- `info`: per-matrix first failing column, written by lane 0 of the
  partition; every matrix writes.

**potrf (right-looking, unblocked, Lower; Upper by the transposed load
the CTA kernel already uses).** Per column `j` (unrolled): lane `j`
publishes `d = rA[j]` (real part) to local `sx[j]`; `group_barrier(sg)`;
every lane reads `d`, `info` check (`d <= 0 || d != d` -> first failure,
matrix continues with `d = 1` so it stays finite); lanes `r > j`:
`rA[j] *= rsqrt(d)` (`sycl::native::rsqrt` A/B'd against `1/sqrt`), lane `j`:
`rA[j] = sqrt(d)`; all lanes write `sx[r] = rA[j]` (column `j` of L);
barrier; lanes `r > j`: `for k in j+1..r: rA[k] -= rA[j] * conj(sx[k])`.
Two sub-group barriers per column, local scratch `N` elements per matrix.
Registers: `N` for `rA` + ~12; float N=32 ~48 (fits 100% occupancy).
cdouble N=32 is 128 registers for `rA` alone -> instantiate cdouble only
for `N <= 16`, route cdouble n = 17..32 to the existing CTA kernel.

**getrf (partial pivoting, lazy swap; MAGMA `smallsq_noshfl` ported to
sub-group collectives).** State per lane: `rA[N]`, `rowid = r`, `linfo`.
Per column `j`: `a = cabs1(rA[j])` if `rowid >= j` else 0 (eliminated rows
excluded); argmax over the partition by `reduce_over_group(part, packed(a,
rowid), max)` where `packed` is `(float bits of a) << 32 | (N - rowid)` on a
`uint64` (exact; tie -> lowest rowid, LAPACK order) - a sub-group collective,
not a work-group one (R4); `p = rowid of the max`; `sipiv[j] = p` by the
lane that owns it; the owner lane writes `sx[k] = rA[k]` for `k >= j`
(pivot row), barrier; lazy swap: `if rowid == p: rowid = j else if rowid ==
j: rowid = p`; `piv = sx[j]`, `linfo = (piv == 0 && linfo == 0) ? j+1 :
linfo`, `rcp = piv == 0 ? 1 : 1 / piv`; lanes with `rowid > j`: `rA[j] *=
rcp; for k in j+1..N: rA[k] -= rA[j] * sx[k]`; barrier. Store: `A(rowid, k)
= rA[k]`, `ipiv[j] = sipiv[j] + 1`. One collective + two barriers per
column; scratch `N` elements + `N` ints per matrix. The `cabs1` pivot
metric matches LAPACK and cuBLAS (the harness compares pivots
elementwise).

**geqrf (Householder, `m = n` square in P1; tall m in P5).** Per column
`j`: every lane writes `sq[r] = |rA[j]|^2` for `r > j` (else 0) and
`sA(r, j) = rA[j]`; barrier; norm by every lane summing `sq[j+1..N]`
serially from local (N reads; serial beats a tree for N <= 32, literature
and MAGMA), `alpha = sA(j, j)`; LAPACK `larfg` scalars (`beta =
-copysign(norm, re alpha)`, `tau`, `scale = 1/(alpha - beta)`), the
`tau = 0` branch when the sub-column is zero and `im(alpha) = 0`; lanes `r
> j`: `rA[j] *= scale`; lane `j`: `rA[j] = 1`, publishes `tau`; `A(r, j)`
stored now (`beta` on the diagonal, `v` below); then `y = conj(tau) v^H
A(:, j+1..N)`: lane `r` writes `sP(r, k) = conj(rA[j]) * rA[k]` for `k >
j` into an `N x (N+1)`-padded local tile; barrier; lane `k` sums column `k`
of `sP` over rows `j..N` -> `sy[k]`; barrier; lanes: `rA[k] -= rA[j] *
sy[k]` for `k > j` (with `conj(tau)` folded into `sy`). Four barriers per
column; local scratch `N x (N+1) + 2N` elements per matrix (float N=32:
4.4 KB; 4 matrices per WG at N=32 -> 18 KB; at N=8, 16 matrices -> 5 KB).
LAPACK's overflow-safe scaling in `larfg` is kept (`LarfgScalars` in
`geqrf_cta_device.hh`). `tau` is written to the caller's span.

**Files.**
- New `src/extensions/tiny_device.hh`: the load/pad/store helpers, the
  packed argmax, `sx/sq/sP` scratch layout, `SLDA(N) = N + 1`.
- New `src/extensions/potrf_tiny.cc`, `getrf_tiny.cc`, `geqrf_tiny.cc`: the
  three kernels, `switch (N)` dispatch, `*_tiny_dispatch<T>(...)` entry
  points, `*_tiny_max_n<T>()` (32; 16 for cdouble).
- `src/extensions/{potrf,getrf,geqrf}_native.hh`: declare the entry points.
- `include/batchlas/blas/dispatch/route_{potrf,getrf,geqrf}.hh`: add
  `Algorithm::Tiny` to the order arrays **first** (`{Tiny, CTA, Blocked,
  Vendor}`), `supports()` = `n <= tiny_max_n && m == n (geqrf) && has_sg32
  && is_gpu && homogeneous`; `preferred()` windows come from the measured
  grid, not from this plan.
- `src/CMakeLists.txt`: the three TUs join `batchlas_extensions_cta_obj`
  (the library that already carries the sub-group kernels).
- Coverage rows: `Op::potrf/getrf/geqrf x Algorithm::Tiny` in
  `src/dispatch/coverage.cc` so `route_diff` can see the arm.
- Workspace: zero (info/ipiv/tau are caller spans); `*_buffer_size` takes
  the max over supported tiers as today, so nothing changes for callers.

**Instantiations.** 3 kernels x 3 N x 4 types = 36 (33 with cdouble
capped at 16), all in the one library that is rebuilt for any `*_cta`
change. Measure the device-link delta before/after (`time cmake --build
--target batchlas_extensions_cta`); budget 15%.

**Tests** (`tests/{potrf,getrf,geqrf}_tests.cc`, new cases, label `blas`,
GPU-only `SetUp`):
- Residual and (getrf) elementwise pivots vs LAPACKE for every `n in 1..32`,
  every type, both `ld = n` and `ld = n + 5`, batch 1 and batch 4G + 3
  (partial last work-group); geqrf also `||Q^H Q - I||` via `orgqr`.
- Padding is inert: `n = 5` inside `N = 8` gives bit-identical results to
  the same matrix at `n = 5` launched through the CTA route (the CTA route
  stays as the oracle for the transition period).
- `info` for a planted non-PD / singular column at position `k` in item
  `b`, first-failure-wins, other items untouched; for getrf the planted
  zero column must give `info = k+1` **and** the elimination must continue
  finitely (MAGMA's `update = 0` semantics).
- Packed launches: a batch whose items differ only in one element must
  not bleed across partitions (poison neighbours with NaN in unused rows
  and assert no NaN in outputs).
- Sub-group scope: a source check (the `tests/error_model_tests.cc`
  pattern) asserts the three TUs contain no `group_barrier(it.get_group())`.
- Armed breaks, each observed red: (a) drop the `rowid > j` guard in
  getrf's update (pivot row gets scaled twice); (b) off-by-one in the
  padded store (`k < N` instead of `k < n`) writes past `n` columns - the
  `ld = n + 5` poison catches it; (c) publish `rA[j]` before the scale in
  potrf - residual test.

**Measurement (P0 harness).** Grid: n in {4, 8, 9, 16, 17, 24, 32} x batch
in [4096, 8192, 16384, 32768, 65536] x 4 types, arms: `tiny`, `cta`,
`vendor`, interleaved. Report ratio to vendor and to the DRAM roof
(`4 n^2 sizeof(T) batch / 950 GB/s` for a read + write of a square). The
n = 9 and n = 17 rows show the padding cost; if `tiny` at 17 is below `cta`
at 17, add `N = 24`.

**Acceptance.** Float and cfloat targets above met at saturation with a
bracketing loss at the first n where `tiny` stops winning against the
next native tier; register probe passes (frame 0, spill 0) for every
instantiation; ptxas static shared = 0; full-suite failing names ==
ledger; device-link delta <= 15%. `preferred()` for `Tiny` is then set
from the grid (a separate small PR per op, R8 package).

**Kill criteria.** If float getrf `tiny` at n = 32 is below 1.0x of cuBLAS
at batch 16384 after the `rsqrt`/serial-vs-tree/scratch-layout A/Bs listed
in the design, stop getrf-tiny and record the number; potrf and geqrf are
independent and continue.

**Effort.** 5-7 agent-days (potrf 1.5, getrf 2, geqrf 2, tests and
measurement 1.5). Depends on P0.

---

### P2. Fused factor-and-solve for tiny systems (`gesv`, `posv`)

**Goal.** One kernel that factorises and solves `A X = B` for `n <= 32`,
`nrhs <= 4`, from `linalg::solve` (which today is `getrf` then `getrs`;
`include/batchlas/blas/linalg-ops.hh:287-302`). The fused getrs tier already
wins 1.7-3.8x by keeping the RHS resident; fusing the factorization removes
the second launch, the second read of A, and the `ipiv` round trip.

**Targets.** >= 3x vs `getrf + getrs` vendor at n <= 32, nrhs = 1, float
and cfloat; >= 2x at nrhs = 4. For SPD (`posv`) the same against
`potrf + potrs` where `potrs` is composed from trsm today (there is no
potrs op).

**Design.** P1's getrf kernel with `T rB[NR]` per lane (`NR in {1, 4}`,
template; nrhs 2 and 3 use 4 with padding), MAGMA `gesv_batched_small`
structure:
- forward substitution fused into elimination: the pivot-row lane also
  publishes `sb[k] = rB[k]`, and lanes with `rowid > j` do `rB[k] -= rA[j] *
  sb[k]`;
- back substitution after the last column: `sB(rowid, k) = rB[k]`;
  for `i = N-1..0`: lane `i`'s row of U is published (`sx[r] = rA[i]`),
  `x_i = sB(i,k) / sx[i]`, lanes `r < i`: `sB(r,k) -= x_i sx[r]`; one
  barrier per step; store `X(r, k) = sB(r, k)`.
- `posv`: P1's potrf kernel followed by the two triangular solves with L
  held in registers (`L(r, j)` is `rA[j]` on lane `r`): forward `y_i =
  (b_i - sum_{j<i} L(i,j) y_j) / L(i,i)` needs `y_j` broadcast (one local
  write + barrier per step), backward the same through local scratch
  (Kurzak's "store L^T too" trick is unnecessary because every lane holds
  a full row).
- Also emits the factor (in place) and `ipiv` so `solve` stays
  semantically identical to `getrf + getrs`.

**Files.** `src/extensions/gesv_tiny.cc`, `posv_tiny.cc`; new public
ops `gesv` / `posv` in `include/batchlas/blas/functions/{gesv,posv}.hh`
(same signature family as `getrs`, `nrhs` from `B.cols()`), route tables
`route_gesv.hh` / `route_posv.hh` with order `{Tiny, Composed}` where
`Composed` is the existing `getrf + getrs` (or `potrf + trsm x2`) and is the
vendor-free fallback; `linalg::solve` and a new `linalg::solve_spd` call
them. Instantiations: 2 kernels x 3 N x 2 NR x 4 types = 48 -> 40 with
cdouble capped at N = 16.

**Tests.** `tests/gesv_tests.cc`, `posv_tests.cc`: `||A X - B|| / (||A||
||X||)` vs LAPACKE `gesv`/`posv` for n in 1..32, nrhs in 1..4, both ld
paddings; identical factor and `ipiv` to the `getrf` route (bit-exact);
`info` propagation; armed break: skip the last back-substitution step.

**Measurement.** Harness `gesv`/`posv` ops vs `getrf+getrs` (vendor) and
vs `tiny getrf + fused getrs` (native, two launches) - the second arm
quantifies what fusion itself buys. n in {4..32}, nrhs in {1, 2, 4}, batch
[4096..65536].

**Acceptance.** Targets met float/cfloat; `linalg::solve` end-to-end A/B
(the `cond`/`inverse` tests use it) shows the gain; failing names == ledger.

**Kill.** If fused beats the two-launch native arm by < 1.15x at n = 32
nrhs = 1, ship only the `linalg::solve` routing to `tiny getrf + fused
getrs` and drop the fused kernel.

**Effort.** 2-3 agent-days. Depends on P1 (getrf and potrf tiny kernels).

---

### P3. Left-looking fused Cholesky panel ("lpout") as the potrf leaf, 32 < n <= 512

**Goal.** Replace the SLM-resident CTA kernel above n = 32 with a
thread-per-row, 8-wide, left-looking fused panel kernel whose local
footprint is `(n + 8) x 8` elements per matrix, so occupancy no longer
collapses; use it directly to n ~ 192 and as the blocked driver's leaf
above.

**Targets** (float, vs cuSOLVER `potrfBatched`): >= 1.2x at n in {64, 96,
128, 192} (today 1.00 / 0.51 / 0.36 / -), >= 1.0x at 256 and 512 through
the blocked driver (today 0.69 and 0.91 routed). cfloat: >= 1.0x at
64..192 (today 0.40-0.37). This is MAGMA's shipped design for exactly this
band (crossover s 192, c 160), so the risk is in the port, not the idea.

**Design.**
- Kernel `PotrfLpanelKernel<T, NB=8>`: work-group = one matrix, `WG = n`
  work-items (thread `r` owns row `r`), single launch, loop over panels `j
  = 0, 8, 16, ...` inside the kernel (MAGMA `lpin` shape; `lpout`'s
  per-panel relaunch with a shrinking work-group is the A/B). For n <= 64,
  pack `G = 128 / n` matrices per work-group with a per-matrix local slice
  (barriers are uniform because every matrix runs the same panel count).
- Per panel: (1) `rp[8] = A(r, j..j+7)` prefetch; (2) left-looking update
  `rC[8] = sum_{k < j} A(r, k) * conj(A(j..j+7, k))`: loop `k` in steps of
  8, `rA = rp`, threads `r < 8` stage the `8 x 8` block `sB = conj(A(j+r,
  k+i))`, barrier, prefetch `rp = A(r, k+8..k+15)`, 64 FMAs on `rA x sB`,
  barrier - the "prefetch becomes the panel" idiom from
  `zpotf2_devicesfunc.cuh`; after the loop `sA(r, i) = rp[i] - rC[i]` for
  the `(n - j) x 8` panel in local memory; (3) unblocked Cholesky of the
  panel in local: for `i in 0..8`: `d = sA(j+i, j+i)`, info check, scale
  column by `rsqrt(d)`, rank-1 update of the remaining panel columns, 3
  barriers per column; (4) store the panel's lower part `A(r, j+i) = sA(r,
  i)` for `r >= j+i`.
- Threads `r < j` are idle in the panel factorization but must hit the
  barriers; whole sub-groups below `j` skip the FMA loop (uniform branch).
  Above n = 256 the idle fraction averages 50%: that is why the blocked
  driver takes over at the crossover, with this kernel as its leaf on the
  `nb x nb` diagonal block (nb = 128 float).
- `NB = 8` is the baseline (MAGMA), `NB = 16` an A/B for float only
  (register cost `3 x 16` + addressing - not the dead SLM-resident NB=16).
- Complex: real diagonal, `conj` on the staged block; `Cx<R>` POD type from
  `register_64x64_k16_wide.hh` for the FMAs (Annex G trap).
- Local memory per matrix: `(n + 8) * 8 * sizeof(T) + 8 * 8 * sizeof(T)`:
  float n=512 -> 16.9 KB (5 blocks/SM by memory, 3 by threads); cfloat
  n=256 -> 17 KB.
- Upper: same transformed-tile trick as `potrf_cta.cc` (`S(i,c) =
  conj(A(c,i))`), or refuse and let the blocked driver stay Lower-only;
  decide by whether any in-tree caller uses Upper (`grep` first).

**Files.** `src/extensions/potrf_lpanel.cc` (+ `potrf_lpanel_device.hh`);
`route_potrf.hh`: order `{Tiny, LPanel, Blocked, Vendor}` with CTA
retained behind a `BATCHLAS_POTRF_ROUTE=cta` pin for one release then
deleted; `potrf_blocked.cc`: leaf = `LPanel` on the diagonal block,
`PotrfBlockedConst` re-tuned (`nb` sweep {64, 96, 128, 192} per type,
`W` re-checked) after the leaf swap; `potrf_native.hh` declarations;
coverage rows. Instantiations: 1 kernel x {NB 8 (all), NB 16 (float A/B)}
x 4 types = 5.

**Tests** (`tests/potrf_tests.cc`): residual for n in {33, 40, 48, 63, 64,
65, 96, 127, 128, 129, 192, 255, 256, 257, 384, 512}, all types, `ld`
padded, batch 1 and 4G+3; `info` planted at columns crossing panel
boundaries (7, 8, 9, 64, 65); other triangle untouched (poison); packed
launches independent; the blocked driver at n = 512 bit-identical whether
the leaf is `LPanel` or the old CTA (transition oracle). Armed breaks:
drop the last-panel barrier before the store (race -> residual); stage the
block without `conj` (complex residual); off-by-8 in the prefetch bound
(reads past the panel -> poison NaN).

**Measurement.** Grid n in {33, 48, 64, 65, 96, 128, 129, 192, 256, 384,
512} x batch ladder x 4 types; arms `lpanel`, `cta` (n <= ceiling),
`blocked(leaf=lpanel)`, `blocked(leaf=cta)`, `vendor`; nsys attribution
once at float n = 128 and 512. `ncu` `sm__warps_active` at float 128
(gate: >= 40%, vs 8.3% today).

**Acceptance.** Targets; a bracketing loss at the largest n where
`lpanel` alone is behind `blocked` (this sets `native_tier_preferred`);
`preferred()` window written from the grid with the R8 package; register
probe; ledger.

**Kill.** If float n = 128 `lpanel` is < 0.9x of cuSOLVER after the NB and
`lpin`/`lpout` A/Bs, stop and record; P1 and the blocked driver still
benefit from P6.

**Effort.** 4-5 agent-days. Depends on P0; P6 lifts n >= 256 further.

---

### P4. Register-panel LU leaf, recursive panel, row-parallel laswp (32 < n <= 512)

**Goal.** Match cuBLAS `getrfBatched` where native now loses (float n =
128: 0.73-0.87x; cfloat 128: 0.38-0.75x; float 32..64 handled by P1) and
extend the routed window down from 256 to 64.

**Targets** (large batch): float >= 1.0x at 96 and 128, >= 1.3x at 64
(1.61x today via CTA - must not regress), >= 1.5x at 256 (1.57 today);
cfloat >= 0.9x at 128 (0.75), >= 1.3x at 256 (1.19-1.41).

**Design.**
- Kernel `GetrfPanelRegKernel<T, NB=32>`: one work-group per matrix (or G
  per WG when `m <= 64`), `WG = m` work-items (thread-per-row; `m <= 512`
  float/double, 384 cfloat, 256 cdouble as MAGMA's register limits; above
  that the existing global leaf stays), `rA[32]` = row `r` of the `m x 32`
  panel, lazy swap with `rowid`, `linfo` gbstep-relative. Per column `j`:
  `a = cabs1(rA[j])` for `rowid >= j`; argmax = sub-group
  `reduce_over_group(sg, packed(a, rowid), max)` then one local slot per
  sub-group and a serial scan over `ceil(m/32) <= 16` slots by every lane
  (the existing `getrf_cta_device.hh` two-level argmax, now over register
  operands); pivot row through `sx[32]`; relabel; scale; rank-1 on
  registers; 3 work-group barriers per column (MAGMA `getf2_fused` has 4).
  Store `A(rowid, k) = rA[k]`, `ipiv[j] = piv + 1`.
- Driver (`getrf_blocked.cc`): keep the existing nb = 32 loop and swap the
  leaf first (step 1: measure). Step 2: MAGMA's recursive panel - outer
  `nb = 128` (float/cfloat) split into 32-wide register leaves with
  `trsm(Left, Lower, Unit, 32, n2)` + `gemm(m - 32, n2, 32)` between halves,
  so the trailing GEMM outside the panel has `k = 128`
  (`Tiled128x128RegisterK8` territory for float instead of `Tiled16`-class
  small-k shapes) and the number of full-width trailing updates drops 4x.
- laswp: compose the panel's 32 sequential swaps into a permutation on
  device (`setup_pivinfo`) and apply right-of-panel as one gather kernel
  parallel over `batch x ceil(ncols / 32)` work-groups with the `32 x 32`
  tile through local memory (the existing `lu_laswp_deferred_left_launch`
  gather is the left-hand half; this adds the right-hand half, which
  `docs/perf/lu.md` lists as still a `range<2>(batch, ncols)` walk).
- Complex: `Cx<R>` arithmetic; `cabs1`.

**Files.** `src/extensions/getrf_panel_reg.cc` (+ device header);
`getrf_blocked.cc` (leaf swap, recursive panel, `nb` per type);
`lu_laswp.hh` (right-hand gather); `route_getrf.hh` (`Blocked` window
re-measured; CTA retained for `n <= 32` until P1 lands, then deleted);
coverage rows. Instantiations: 1 kernel x NB 32 x 4 types = 4 (+4 if a
`NB = 16` cdouble variant is needed for registers).

**Tests** (`tests/getrf_tests.cc`): residual + elementwise `ipiv` vs
LAPACKE at n in {33, 48, 64, 65, 96, 127, 128, 129, 192, 256, 257, 384,
512}, `m > n` tall panels (`m` in {64, 128, 512} x `n = 32`), all types,
`ld` padded, batch 1 and odd; planted singular column inside and outside
the first panel with `info` gbstep-correct; a matrix that needs a swap
across the panel boundary (row 200 pivoting into column 3); the right-hand
gather bit-identical to the walk on a poisoned trailing block. Armed
breaks: skip `adjust_ipiv` (+ offset) on the second panel; drop the
`rowid >= j` mask in the argmax (already-eliminated row wins); gather
reads `pivinfo` of the wrong panel.

**Measurement.** Grid n in {33, 64, 65, 96, 128, 129, 192, 256, 384, 512}
x ladder x 4 types; arms `blocked(leaf=reg, nb=32)`, `blocked(leaf=reg,
recursive nb=128)`, `blocked(leaf=cta)` (today), `vendor`; nsys at float
128 and 512 (panel / laswp / trsm / gemm split; today 48.5 / 33.6 / 8.8 /
9.2 % at double 128).

**Acceptance.** Targets; window edges bracketed; register probe (the
`rA[32]` cdouble instantiation is the risk: 128 registers for `rA` alone,
so cdouble may need `NB = 16`); `route_diff`; ledger.

**Kill.** If float n = 128 stays < 0.9x after the recursive-panel step,
record it and keep the routed window at 256; the leaf still replaces the
CTA path for tall panels if it wins there.

**Effort.** 5-6 agent-days. Depends on P0 and P7 (packing for m <= 64).

---

### P5. Fused QR panel + fused reflector apply (n <= 128 and every tall panel); orgqr/ormqr small

**Goal.** A thread-per-row register panel kernel `geqr2_fused<T, N in
{8,16}>` for `m x N` panels with runtime `m` (up to 1024), and a fused
"apply N reflectors to C" kernel that needs no `T`, no `larft`, no
`pack_v`, no gemm. Together they are geqrf for `n <= cutoff`, ormqr(Left,
{NoTrans, ConjTrans}) for `k <= cutoff`, and orgqr (apply to an identity
held in registers, backward order). Above the cutoff the existing WY
driver keeps its trailing gemms but takes the fused panel as its leaf and
drops the serial `larft`.

**Targets.** double geqrf n = 64: 0.53x -> >= 1.0x (the panel kernel is
100% of that loss at 14.5% of FP64 peak with 193 of 256 lanes idle);
float/cfloat n = 32..128: keep >= 2x (already there via the weak vendor)
while removing `larft` (15-49% of orgqr/geqrf time); tall panels 128x32
and 512x32 float >= 3x (1.57 / 2.11 today), double >= 1.2x (0.62 / 1.16).
orgqr small `n <= 128`: >= 1.5x over the current identity+ormqr path
(the 1.5x flop and 11-22% fill/copy overheads).

**Design.**
- `GeqrfPanelRegKernel<T, N>`: WG = `roundup(m, 32)` work-items, one panel
  per WG (G-pack for `m <= 64`), `rA[N]` = row `r`, `m` runtime with
  `r >= m` lanes holding zeros. Per column `j`: partial `|rA[j]|^2` for `r
  > j` -> `reduce_over_group(sg, ...)` + one slot per sub-group + serial
  combine (2 barriers) instead of MAGMA's `log2(M32/8)` shared tree;
  `larfg` scalars with LAPACK scaling; scale, store column; `y = conj(tau)
  v^H A(:, j+1..N)`: lane products into a `32 x N`-per-sub-group local
  tile, sub-group `reduce_over_group` per column (N <= 16 collectives of 5
  shuffle steps, or the MAGMA 16-threads-per-column local reduction - A/B
  both), combine across sub-groups through `ceil(m/32) x N` slots; `rA[k]
  -= rA[j] * sy[k]`. Local per WG: `ceil(m/32) x (N + 1)` + `2N` elements
  (float m=512, N=16: ~1.2 KB).
- `LarfApplyRegKernel<T, N>` (the `unm2r` role): WG per matrix, `rC[IB]`
  = row `r` of an `IB`-wide sub-panel of C in registers (`IB = 8`),
  the `m x N` reflector block V in local memory (read once, `SLDA` padded),
  loop over sub-panels of C: for each reflector `j` (forward for `Q^H C`,
  backward for `Q C`): `w = v_j^H C_sub` (same two-level reduction),
  `C_sub -= tau v_j w` (or `conj(tau)`); store the sub-panel. For orgqr:
  `C = I` generated in registers, reflectors applied backward, columns
  `0..n-1` stored - no fill kernel, no copy. Local per WG: `m x (N+1)`
  elements (float m=512 N=16: 34 KB -> 2 blocks/SM; so `N = 8` or
  `m <= 256` for the reg variant, above that the existing WY path).
- Driver: `geqrf_fused_driver`: for `j` in steps of `N`: panel kernel on
  `A(j:, j:j+N)`, apply kernel on `A(j:, j+N:)`; cutoff (largest `n` at
  which this beats the WY driver with the fused leaf) measured per type;
  MAGMA's is a lookup on `(m, batch)`.
- WY driver changes: leaf = `GeqrfPanelRegKernel` (nb 32 = two `N = 16`
  panels with an apply between, or `N = 32` if registers allow for float);
  `larft` via one routed gemm `V^H V` + the `sm32x32` trmv kernel (MAGMA)
  replacing the serial lid-0 recurrence.

**Files.** `src/extensions/geqrf_panel_reg.cc`, `larf_apply_reg.cc`,
`geqrf_fused.cc` (driver), `orgqr_fused.cc`, `ormqr_fused.cc` entry;
`route_geqrf.hh` order `{Tiny, Fused, CTA, Blocked, Vendor}`;
`route_orgqr.hh` / `route_ormqr.hh` gain `Fused`; `geqrf_blocked.cc`
(leaf + larft); `larft_wy.hh`; coverage rows. Instantiations: panel 2 N x
4 types = 8; apply 2 N x {NoTrans, ConjTrans} x 4 = 16; total 24, in the
`extensions_cta` library (measure link delta).

**Tests** (`tests/{geqrf,orgqr,ormqr}_tests.cc`): `||A - QR||`, `||Q^H Q -
I||`, `R` upper-triangular with `beta` sign convention identical to the
CTA route (bit-compare `tau` and `R` on a fixed input between routes: the
transition oracle); tall `m in {33, 64, 100, 128, 256, 512, 513, 1024} x n
in {1, 5, 8, 9, 16, 17, 32, 64, 128}`; orgqr fused vs identity+ormqr
bit-exact; ormqr both trans, Left, `C` wide and narrow; the zero
sub-column (`tau = 0`) case; armed breaks: apply reflectors forward in
orgqr (wrong Q); drop the `r > j` mask in the norm (`beta` wrong sign
magnitude); skip the last sub-panel of C when `n_c % IB != 0`.

**Measurement.** Square n in {33, 48, 64, 96, 128, 192, 256} and tall
{128, 512, 1024} x {16, 32, 64} x ladder x types; arms `fused`, `cta`,
`blocked(leaf=reg)`, `blocked(today)`, `vendor`; end-to-end
`ortho_benchmark` and `syev_benchmark` n = 512 (tall panels are what
`sytrd_sy2sb` and `band_reduction` issue).

**Acceptance.** Targets; cutoff bracketed both sides; register probe;
`route_diff`; ledger; `ortho_benchmark` not slower at any shape.

**Kill.** If the fused apply is slower than WY-with-fused-leaf at every
`n >= 64`, ship only the leaf swap and the larft fix and drop the apply
kernel (orgqr small keeps the register-identity variant only if it wins).

**Effort.** 6-8 agent-days. Depends on P0.

---

### P6. Transposed and complex wide-scalar register GEMM for the panel-update shapes

**Goal.** The trailing updates of P3/P4/P5 and the existing blocked
drivers issue three shape families that today fall to `Tiled16` or the
weak float transposed family: (a) `W1 = V^H A22`: `(nb x m) x (m x n)`,
`ConjTrans/Trans` on A, `k = m` large, output small; (b) `A22 -= L21 U12`
/ `A22 -= L21 L21^H`: `(m x nb) x (nb x n)`, `k = nb in {32, 128}`,
`ConjTrans` on B for potrf; (c) complex NN/CN at every size. `docs/perf/gemm.md`
calls the complex transposed register kernel "the campaign's biggest
prize" (~2.7x on cdouble potrf alone); float transposed is 0.23-0.55x
across 48 of 48 cells.

**Targets.** Float TN/NT at `m, n in [64, 512]`, `k in {32, 128, m}`:
>= 0.9x cuBLAS (from 0.23-0.55); cfloat NN/CN/NC >= 0.9x at `min_dim >=
32` (from 0.14-0.3 on `Tiled16`); cdouble the same at the FP64 ceiling.

**Design.** Port `register_64x64_k16_wide.hh` (4x4 thread tile, `Cx<R>`
POD, 16-byte access granule, 55/72/72/132 registers, zero spill) to (1)
the transposed operand legs by staging the transposed operand through the
existing 16-wide local tile with a transposed store (the trick the float
`128x32K32{TN,NT,TT}` family uses) rather than transposed global reads;
(2) predicated edges (E4's predicated leg for the 128x128 kernel is the
model) so `m, n` need not be multiples of 64; (3) a `k = 32` fast path
that keeps both operand tiles resident (`64 x 32 + 32 x 64` elements) and
skips the k-loop. Selector: `select_kernel_variant` in `gemm_kernels.cc`
gains the new variants with gates `min_dim >= 32 && ctas >= 64`, complex
any transpose; `route_gemm.hh` `preferred()` widens for complex from the
measured grid. The panel-shape (a) with `k = m` gets a split-K A/B
(`ctas` too few when `nb x n` is small): accumulate in a `W x nb x n`
scratch and reduce - measure before shipping.

**Files.** `src/sycl/gemm/register_64x64_k16_wide_{tn,nt,cn,nc}.hh` (or one
header with a layout template), `gemm_kernels.cc` selector, `route_gemm.hh`,
`scripts/gemm_demand.py` (its `preferred()` replica has drifted - refresh
it in the same PR). Instantiations: 4 layouts x 4 types = 16 kernels in
`batchlas_sycl` (the largest library: measure the link delta, budget 15%).

**Tests.** `tests/gemm_tests.cc`: every `(transA, transB)` x type at `m, n,
k in {32, 33, 64, 65, 96, 128, 200, 256}` against a host reference, `ld`
padded, `beta in {0, 1}`, batch odd; a test that asserts on the *selected
variant* (via the coverage row) for the three shape families - today
nothing in ctest asserts on kernel choice and `route_diff` cannot see
selector changes. Armed break: swap the transposed-store index in the
staging tile.

**Measurement.** `gemm_transpose_benchmark` and the archived
`cx_gemm_bench.cpp` shapes; the three families at the shapes P3/P4/P5
actually issue (read them off `BATCHLAS_COVERAGE_OUT` after a potrf /
getrf / geqrf run at n = 256 and 512); end-to-end re-run of the P3/P4/P5
grids at n >= 256.

**Acceptance.** Targets; the end-to-end grids move (record before/after
per op); register probe zero spill for every instantiation; ledger.

**Effort.** 5-7 agent-days. Depends on P0; independent of P1-P5 but
lifts their n >= 256 cells.

---

### P7. Occupancy-aware capacity rule and G-packing for every resident kernel

**Goal.** Cheap, cross-cutting: stop advertising a resident-kernel
capacity that only fits one work-group per SM, pack matrices per
work-group wherever a single sub-group serves a matrix, and declare the
missing `native_tier_preferred` for potrf. This is the fix for the
mechanism behind every mid-band loss, applied to the kernels that stay
(getrs fused, trsm V1, the CTA kernels until P1/P3/P4/P5 retire them, and
the P1-P5 kernels themselves).

**Design.**
- `src/util/resident_capacity.hh`: `resident_max_n(bytes_per_matrix(n),
  local_mem_budget, min_blocks_per_sm = 4)` returning the largest `n` such
  that `bytes(n) <= budget / min_blocks_per_sm` **and** every smaller `n`
  also fits (the non-monotone hole-padding rule from `potrf_cta.cc`);
  replaces the three per-kernel walks (`potrf_cta_max_n_for_slm`,
  `geqrf` area ceiling, `getrf` ceiling). The tier crossover then falls
  out of the capacity instead of being a separately measured table: float
  potrf CTA advertises 77 not 155; geqrf CTA 96 not 155 (the measured
  crossover is 96..112).
- G-packing helper `pack_matrices_per_wg(n, lanes_per_matrix, target_wg =
  128)` used by `getrf_cta.cc` (today one matrix per WG at wg >= 64 for a
  32-column matrix), `geqrf_cta.cc`, and P1/P3/P4/P5.
- `route_potrf.hh`: `native_tier_preferred` declared (from the P0 grid);
  `route_geqrf.hh` capacity uses the new rule; `getrs_fused.cc` and `trsm`
  V1 keep their register tables but gain the blocks-per-SM assertion in
  their tests.
- `device_limits.hh`: the hardcoded 49,152 (2.06x wrong on this box)
  replaced by the runtime query everywhere it is still read
  (`cmake/BatchLASDetectSYCL.cmake:44-45`).

**Tests.** `tests/resident_capacity_tests.cc`: monotone-fit property on a
synthetic non-monotone `bytes(n)`; ceilings pinned per type for potrf /
geqrf / getrf against the budget-parameterised query (the existing
`MeasuredFitCeilings` pattern); a test that packed and solo launches of
`getrf_cta` agree bit-for-bit. Armed break: remove the "every smaller n
fits" clause on the synthetic table.

**Measurement.** `route_diff` before/after (capacity changes move
routes); `ncu sm__warps_active` at the old and new ceilings for potrf and
geqrf; the P0 harness at the affected orders.

**Acceptance.** No cell in the P0 grid gets slower; routes that changed
are listed with the numbers; ledger.

**Effort.** 1.5-2 agent-days. Depends on P0. P4 depends on it.

---

## 4. Sequencing and what to expect

| week | run in parallel | lands |
|---|---|---|
| 1 | P0 | harness, baseline, geqrf/orgqr flips (2-12x at n >= 64 realised) |
| 2-3 | P1, P3, P7 | tiny kernels (n <= 32), lpout potrf, capacity rule |
| 3-4 | P2, P4, P5 | gesv/posv, LU leaf + recursive panel, QR fused panel/apply |
| 4-5 | P6 | transposed/complex register GEMM; re-run P3/P4/P5 grids at n >= 256 |

Expected state after P0-P7 (float, large batch, vs vendor; arithmetic
from the targets, not measurements): potrf >= 3x at n <= 32, >= 1.2x at
64..192, >= 1.0x at 256..512; getrf >= 1.3x at n <= 32, >= 1.0x at
64..128, >= 1.5x at 256..512; geqrf >= 3x at n <= 32, >= 2x at 64..512;
gesv >= 3x at n <= 32; cfloat within 0.8x of the float ratios; double
wins only at n <= 32 and in the solves. Every plan carries its own kill
criterion, and the kill numbers are recorded in the op's `docs/perf`
page as a negative result.

## 5. Decisions taken, and what is still open

Questions 1-3 were put to the repository owner on 2026-09-10 and answered;
the plans above are written to those answers. Recorded here so a subagent
does not re-open them.

**D1. New `Algorithm` enumerators — decided.** `Algorithm::Tiny`,
`Algorithm::LPanel` and `Algorithm::Fused` join `CTA` / `Blocked` /
`Vendor` in `include/batchlas/blas/dispatch/route.hh`, rather than hiding
the new tiers inside the existing `CTA` dispatch functions. The reason is
observability: `scripts/route_diff.sh` and the coverage census key on
`(op, scalar, backend, shape_class) -> Route`, so a tier swap that reuses
`CTA` is invisible to both — exactly the defect `docs/perf/gemm.md`
records for the kernel selector ("`route_diff` cannot see selector
changes"). Cost, accepted: the route vocabulary in
`docs/perf/dispatch.md` gains three words, every `RouteTable` order array
that mentions them is edited, and `Algorithm`'s `to_string` / parse pair
(`route_env.hh`) must round-trip the new names so `BATCHLAS_<OP>_ROUTE=tiny`
pins rather than silently falling through to `automatic()`.

**D2. `gesv` and `posv` are public ops — decided.** P2 ships
`include/batchlas/blas/functions/{gesv,posv}.hh` with their own
`RouteTable`s, `*_buffer_size` functions and test files, shaped like
`getrs`, in addition to routing `linalg::solve` / `linalg::solve_spd` to
them. The v0.2.0 export surface (WP7) therefore grows by two ops; add
them to the surface census before the tag. `posv` also gives the tree its
first `potrs`-shaped entry point, which `docs/perf/potrf.md` lists as
missing.

**D3. `complex<double>` is capped at `N = 16` in P1 — decided.**
`potrf_tiny`, `getrf_tiny` and `geqrf_tiny` instantiate cdouble only for
`N in {8, 16}`; `n = 17..32` keeps the CTA kernel (already 0.76-1.00x of
the vendor there, so little is given up). `rA[32]` for cdouble is 128
registers before anything else, which caps a work-group at 512 work-items
and 33% occupancy — the band is memory-bound, so the instantiation would
probably not pay, and the device-link budget is better spent elsewhere.
Instantiation count for P1 is therefore **33**, not 36. If a measured
cdouble cell at n = 17..32 later looks worth it, the decision is
reversible for one kernel at a time.

Still open, to be settled by measurement or a `grep` rather than by
opinion:

4. P3 `Uplo::Upper`: implement via the transformed tile (`S(i,c) =
   conj(A(c,i))`, as `potrf_cta.cc` already does) or refuse it in
   `supports()` and leave the blocked driver Lower-only. Decide by
   grepping for in-tree `Uplo::Upper` potrf callers first.
5. P5's cutoff between the fused QR driver and the WY driver is a
   `(type, m, batchCount)` lookup table in MAGMA. The plan proposes a
   per-type order threshold measured on square shapes plus the tall-panel
   cells; if the grid shows the cutoff moving with `m` or batch, promote
   it to a 2-D table then, not before.
6. Instantiation budget: P1 (33) + P2 (40) + P5 (24) + P6 (16) ≈ 113 new
   device kernels across three libraries. The build is device-link-bound
   (the `.so` is the SYCL link unit, 117-125 s per touched library), so
   each plan measures the link delta of the library it touches. The 15%
   per-library ceiling is a proposal, not a measured number; the first
   plan to land should replace it with the measured cost per
   instantiation.
