# Small-n factorization plan {#design_small_n_plan}

> **Status:** current, partly shipped · checked 2026-10-21

Decision record for making the native batched factorizations (potrf, getrf, geqrf, gesv, posv,
orgqr) competitive with cuBLAS and cuSOLVER at n = 4..512: the baseline verdict, the design rules,
the decisions, what shipped and what is open. Measured grids: @ref perf_small_n_baseline; per-op
evidence: @ref perf_potrf, @ref perf_lu, @ref perf_qr.

## Small-n: baseline verdict

Ratios are `vendor_ms / native_ms`; above 1 means native wins. "Large batch" is the saturating batch
for the order (>= 2048 at n <= 64, >= 512 at n = 512). RTX 4090 (sm_89), CUDA 13.2, measured
2026-09-10 on the pre-flat-selection tree (`readiness/wp3-abi`, aca64f6).

| op | float | cfloat | double | cdouble | vendor |
|---|---|---|---|---|---|
| potrf | 2.75x at n=8, 1.85x at 32, 1.26x at 48; **0.36-1.00x at 64..256** | 1.79x at 8, then 0.40-1.05x | 0.58-1.07x | 0.57-1.47x | cuSOLVER `potrfBatched` |
| getrf | **0.55x at 32**, 1.61x at 64, 0.73-0.87x at 128, 1.25-2.35x at >= 256 | 0.60x at 32, 0.38-0.75x at 128 | 0.27-0.75x | 0.34-0.75x | cuBLAS `getrfBatched` |
| geqrf | **0.78x at 32**, 2.0-12x at 64..512 | 0.71x at 32, 2.3-8.9x above | **0.21x at 32, 0.53x at 64**, 1.6-6.8x at >= 128 | 0.33-0.84x to 256 | cuBLAS `geqrfBatched`, unblocked |
| getrs (nrhs <= 2) | 1.7-2.5x | 1.3-1.6x | 2.0-3.8x | 1.8-2.6x | `getrsBatched`; fused tier wins |
| orgqr | 3.8-123x | 2.3-65x | 6-114x | 1.7-12x | per-item loop; not a real comparison |

**Diagnosis.** The baseline native factorizations kept the whole matrix in local memory, one
work-group per matrix, with a runtime `n` and a rank-1 recurrence. Above about 48 KB of local memory
that allows 1-2 resident work-groups per SM (`sm__warps_active` = 8.3% for potrf at n = 96..155).
Below about 32 it wastes most of a warp on a matrix one sub-group could hold in registers. The blocked
drivers lose at n <= 256 because their leaf is that kernel; their trailing GEMMs dominate only at
n >= 512.

| kernel | design | loss mechanism |
|---|---|---|
| potrf CTA (`potrf_cta.cc`) | whole matrix in SLM, NB=8, L=32..256 | SLM footprint: 1-2 blocks/SM above n ~90 |
| getrf CTA (`getrf_cta.cc`) | whole matrix in SLM, rank-1 right-looking | 3 work-group barriers per column; wg >= 64 for a 32-column matrix |
| geqrf CTA (`geqrf_cta.cc`) | whole m x n tile in SLM, two `reduce_over_group` per reflector | 256-lane reduction for a 64-element column; 48 KB pad; FP64 rate |
| geqrf blocked (`geqrf_blocked.cc`) | right-looking WY, nb 32 (16 double), serial larft | larft 15-49% of time; `V^H A22` on `Tiled16` for complex |
| getrs fused (`getrs_fused.cc`) | one work-group per matrix, RHS resident | wins; the model for gesv/posv |

**Reference designs.** The literature and MAGMA's source agree on: thread-per-row with the matrix or
panel in registers and a compile-time column count; local memory only as a mailbox; several matrices
per work-group when n < 32; lazy row swaps; a left-looking fused Cholesky panel up to n ~ 160-224; a
fused register panel as the LU and QR leaf up to m = 512; BLAS-3 blocking only above ~192-256.

## Small-n: design rules

- **R1** Registers are the store; local memory is the mailbox. Per-matrix local memory is O(N) or
  O(N x NB) for N > 32. Target >= 4 resident work-groups per SM (`resident::kMinBlocksPerSm`).
- **R2** Compile-time column count with ladder N in {8, 16, 32}; rows stay runtime. N = 24 is not
  available: `SubGroupPartition<P>` (`src/kernels/common/sg_partition/sg_partition.hh`) needs P to
  divide 32.
- **R3** The sub-group is the synchronisation unit for n <= 32. No work-group barrier in those kernels.
- **R4** No `reduce_over_group` over a work-group. Use sub-group collectives plus one slot per
  sub-group. Check that static shared memory is 0 (the 48 KB launch hole).
- **R5** Lazy swap: `rowid` relabelling, pivot row through a 32-element scratch, permutation at store.
- **R6** Read once, write once per owned element.
- **R7** Each plan states its instantiation count and measures the device-link delta of the library it
  touches. The `.so` is the SYCL link unit (117-125 s per touched library).
- **R8** Measure per `docs/perf/README.md`: one process on the box, device 1, JIT warmed, interleaved
  A/B, relative sd < 10%, saturation ladder, bracketing non-winner at every edge. A table flip needs
  `t_native <= 0.90 t_vendor` at saturation, no accuracy regression, and an end-to-end A/B.
- **R8a** Some ops do not saturate inside the memory ceiling (potrf's ratio rises with batch without
  converging). Report the ratio with its direction and batch; do not quote a single limit.
- **R9** Every new test ships with a deliberate break that turns it red, observed.
- **R10** Double and cdouble are measured and reported. They gate nothing: FP64 runs at 1/64 of FP32,
  so only memory-bound double kernels can win.

Hardware anchors (RTX 4090): 128 SMs; 65,536 registers per SM; 101,376 B opt-in local memory per
work-group; a dynamic request in (47,104, 49,664] B fails cold (the 48 KB hole); DRAM ~950 GB/s
achievable; FP64 ~1.3-1.44 TFLOP/s. The DRAM roof for read-plus-write-once float n=32 at batch 8192 is
70 us; cuBLAS getrf there is 159 us and cuBLAS geqrf 772 us.

## Small-n: decisions

- **D1. Tiers are separate families, not hidden inside CTA dispatch**, so `route_diff`, which keys on
  the chosen route, sees a tier swap. Families: potrf `tiny`, `cta`, `lpanel:panel=N`, `blocked`,
  `vendor`; getrf and geqrf `tiny`, `cta`, `blocked`, `vendor`; gesv `tiny`, `blocked`; posv `tiny`,
  `cta`, `blocked` (`src/ops/<op>/choice.hh`).
- **D2. `gesv` and `posv` are public ops.** `linalg::solve_spd` calls `posv`. `linalg::solve` calls
  `gesv` only for `Transpose::NoTrans`, because `gesv` has no transpose parameter; `Trans` and
  `ConjTrans` keep the composed `getrf` then `getrs`.
- **D3. cdouble is capped at N = 16 for the tiny tier.** `rA[32]` in cdouble takes 128 registers before
  anything else, so cdouble n = 17..32 stays on CTA (`getrf_tiny.cc`): 33 tiny-tier instantiations,
  not 36. Reversible one kernel at a time.

## Small-n: plan status

Targets and kill criteria as set (float, large batch, vs vendor):

- P1: potrf >= 3x at n in {8, 16, 32}; geqrf >= 3x at 32; getrf >= 1.3x at 32 and >= 2x at n <= 16.
  Kill: getrf tiny below 1.0x at n = 32, batch 16384.
- P2: fused gesv or posv >= 3x over getrf+getrs at n <= 32 (nrhs 1). Kill: fused below 1.15x over the
  two-launch arm at n = 32, nrhs = 1.
- P3: lpanel >= 1.2x at n in {64, 96, 128, 192}. Kill: below 0.9x at n = 128.
- P4: getrf >= 1.0x at 96 and 128; >= 1.3x at 64; >= 1.5x at 256. Kill: below 0.9x at n = 128 after the
  recursive step.
- P5: tall panels 128x32 and 512x32 >= 3x; double geqrf n = 64 >= 1.0x. Kill: the fused apply slower
  than WY-with-fused-leaf at every n >= 64.

| plan | idea | status |
|---|---|---|
| P0 | harness, baseline, routing flips | **Partly shipped.** `benchmarks/factor_bench.cc`, `run_factor_grid.sh`, `gpu_guard.sh`. 6 of the 20 planned baseline CSVs are in `benchmarks/results/` (`factor_baseline_getrs_*` x4, `_orgqr_double`, `_potrf_float`). |
| P1 | register-resident sub-group kernels, n <= 32 | **Shipped** for potrf, getrf, geqrf (`*_tiny.cc`, `src/kernels/common/tiny_device.hh`, `geqrf_tiny_device.hh`). geqrf departs from the design: two barriers per column, C-derived tile, work-group 64 ([the tiny tier](../perf/qr.md#the-tiny-tier-wp6--p1-square-n--32-in-registers)). |
| P2 | fused factor-and-solve, gesv and posv | **Shipped**: `gesv_tiny.cc`, `posv_tiny.cc`, `linalg::solve_spd`, the `linalg::solve` NoTrans call. |
| P3 | left-looking Cholesky panel (`lpanel`) | **Shipped**, Lower only (`potrf_lpanel.cc`; panel 8, and 16 for float). The sm_120 table ranks it (682 rows); the sm_89 table has no `lpanel` row. |
| P4 | register LU leaf | **Partly shipped.** `getrf_panel_reg.cc` is the default leaf of `blocked` (`BATCHLAS_GETRF_LEAF=slm` selects the SLM panel). The right-hand interchange is a coalesced gather below a crossover height, the per-column walk above it ([the right-hand gather](../perf/lu.md#the-right-hand-gather)). The recursive nb=128 panel is not built. |
| P5 | fused QR panel and reflector apply | **Partly shipped.** `GeqrfPanelRegKernel` is the leaf with `cols = 16` for double only; float, cfloat and cdouble are `cols = 0` (absent) because each adds 39.6-60.5 s of ptxas to the link (`src/ops/geqrf/geqrf_panel_reg_device.hh`). Not built: the fused apply, `geqrf_fused_driver`, `orgqr_fused`, `ormqr_fused`, the larft replacement. |
| P6 | transposed and complex wide-scalar GEMM | **Superseded** by the `wide:m=..:n=..:k=..` gemm family (`src/ops/gemm/choice.hh`). |
| P7 | capacity rule, G-packing | **Shipped.** `src/util/resident_capacity.hh` (`device_slm_budget`, `occupancy_budget`, `resident_max_n`, `pack_matrices_per_wg`) serves the potrf, getrf and geqrf CTA walks, the tiny tiers and `lpanel`. The planned `native_tier_preferred` field does not exist; tables rank tiers instead. |

**Why P5's fused apply is not next.** `geqrf_nb_for_type<T>()` (`geqrf_blocked.cc`) is 32 for float,
cfloat and cdouble, and 16 only for double. For a 32-wide matrix the panel loop returns after one
panel, before `pack_v`, `larft` or the trailing GEMMs that the fused apply would remove. The
"larft is 34% of time" figure came from an unsaturated batch-8 profile; the larger-batch profiles put
larft plus `pack_v` at 6.5-22%. Size any larft work against those.

## Small-n: open items

- Retune sm_89 potrf: the table has no `lpanel` row, although the measured window shows it winning
  there ([the measured LPanel window](../perf/potrf.md#the-measured-lpanel-window)).
- Finish the P0 baseline: 14 of 20 CSVs are missing.
- P2: the end-to-end `linalg::solve` A/B and the fused-versus-two-launch kill test (>= 1.15x at
  n = 32, nrhs = 1) are not run as such. The sm_120 gesv table is deep-measured (877 rows) and ranks
  `tiny` against `blocked` per cell; the sm_89 gesv table is not.
- P5: build the fused apply, or drop it and record the kill. Measure orgqr small (target >= 1.5x over
  identity-plus-ormqr), never measured.
- P4: decide on the recursive panel, or record a kill.
- Confirm the R7 link budget per plan. The 15% per-library figure is a proposal, not measured.
- Upper triangle for `lpanel`: not implemented; the blocked driver stays Lower only.

## Small-n: rejected alternatives

| Alternative | Result | Verdict |
|---|---|---|
| potrf NB=16 in the SLM-resident kernel | 156-206 registers, 2x slower | dead |
| `reduce_over_group` over the work-group | 48 KB hole; 1.5-4.7x slower for FP64 | dead (R4) |
| Packed sub-group getrf with the matrix in SLM | no win | dead |
| Cooperative multi-lane trsm solve | no win | dead |
| Bespoke potrf panel-solve kernel | no win | dead |
| Launch-count fusion alone | <= 3% | dead |
| geqrf nb = 128 | no win | dead |
| `Tiled16` double-buffering and B-packing for strided ld | no gain | dead |
| Pivot-search experiment with SLM matrix and runtime n (`wp6_perf/proto/pivsg.cpp`) | lost; did not test R1 register storage | inconclusive |

## Comment-density gate and its waivers {#small-n-comment-density-burn-down}

Measured numbers live in `docs/perf/<op>.md` behind an `evidence: docs/perf/<page>.md#<anchor>`
pointer; code comments carry correctness rationale only. The gate dates from 2026-09-11, when the
repo aggregate (15.3%) hid 150 over-ceiling files of 424; 21 were brought into band then by moving
measurements into `docs/perf/`.

- `.github/ci/check_comment_density.py` fails any file over 18% comment lines over non-blank lines. It
  runs in `run_local_checks.sh` and the `comment-density` CI job; `--self-test` runs 37 cases.
  Doxygen doc-comment lines in public headers are exempt (5,178 lines on 2026-10-21;
  @ref documentation_conventions).
- On 2026-10-21: 460 files, repo density 10.53% (16,528 / 156,950), 22 waived.
- `.github/ci/comment_density_waivers.txt` holds two file-specific waivers (`src/util/resident_capacity.hh`
  27.00%, `src/ops/geqrf/kernels/geqrf_tiny_device.hh` 21.56%), one for `src/ops/syev/uplo_mirror.hh`,
  and 19 lines of the 2026-09-11 batch block (133 lines originally) that cite this section. Do not add
  to that block. A waiver that matches no file is an error, so delete a line when its file comes into
  band.
- The structural doc-comment headers that were worst before the exemption (`sycl_interop.hh` 86%,
  `settings.hh` 78%) now count 10.00% and 16.50%; `src/util/internal-api.hh` is 16.67%.
