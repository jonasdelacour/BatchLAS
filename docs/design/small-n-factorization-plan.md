# Small-n factorizations {#design_small_n_plan}

> **Status:** current · checked 2026-10-29

Design rules and open debts for the native batched factorizations (potrf, getrf, geqrf, gesv, posv,
orgqr) at n = 4..512. Baseline grids: @ref perf_small_n_baseline; per-op evidence: @ref perf_potrf,
@ref perf_lu, @ref perf_qr.

## Small-n: design rules

The baseline kernels held the whole matrix in local memory, one work-group per matrix: above ~48 KB
that allows 1-2 resident work-groups per SM, and below n ~ 32 it wastes most of a warp. The rules:

- **R1** Registers are the store; local memory is the mailbox (O(N) or O(N x NB) per matrix). Target
  >= 4 resident work-groups per SM (`kMinBlocksPerSm`, `src/util/resident_capacity.hh`).
- **R2** Compile-time column ladder N in {8, 16, 32}; rows stay runtime. N = 24 is impossible:
  `SubGroupPartition<P>` needs P to divide 32.
- **R3** For n <= 32 the sub-group is the synchronisation unit: no work-group barrier.
- **R4** No `reduce_over_group` over a work-group (the 48 KB launch hole; 1.5-4.7x slower in FP64).
- **R5** Lazy swap: `rowid` relabelling, pivot row through a 32-element scratch, permutation at store.

## Small-n: decisions

- **D1** Tiers are separate families, so `route_diff` sees a tier swap: potrf `tiny`, `cta`, `lpanel`,
  `blocked`, `vendor`; getrf and geqrf `tiny`, `cta`, `blocked`, `vendor`; gesv `tiny`, `blocked`; posv
  `tiny`, `cta`, `blocked` (`src/ops/<op>/choice.hh`).
- **D2** `linalg::solve_spd` calls `posv`; `linalg::solve` calls `gesv` only for `Transpose::NoTrans`
  (`gesv` has no transpose), else `getrf` then `getrs` (`include/batchlas/blas/linalg-ops.hh`).
- **D3** cdouble tops out at N = 16 in the tiny tier ([why](../perf/lu.md#d3-why-cdouble-stops-at-n16)).

## Small-n: open items

- sm_89 `tuned/potrf.sm_89.txt` has no `lpanel` row although the measured window shows it winning
  there ([the measured LPanel window](../perf/potrf.md#the-measured-lpanel-window)); sm_120 ranks it.
- `lpanel` is Lower only (`potrf_lpanel.cc` throws on Upper).
- P0 baseline: 6 of 20 planned CSVs exist (`benchmarks/results/factor_baseline_*`,
  `benchmarks/factor_bench.cc`).
- gesv: the sm_89 table is transcribed, not deep-measured; the fused-versus-two-launch kill test
  (>= 1.15x at n = 32, nrhs = 1) and the end-to-end `linalg::solve` A/B are not run.
- geqrf (P5): `GeqrfPanelRegKernel` is built for double only (`cols = 16`; the other types add
  39.6-60.5 s of ptxas each). The fused reflector apply, `orgqr_fused` and `ormqr_fused` are not built;
  kill clause: the fused apply slower than WY-with-fused-leaf at every n >= 64. Below nb (32 except
  double's 16) the blocked loop never reaches `larft`, and larft plus `pack_v` are 6.5-22% at
  saturating batch, so size any larft work against that.
- getrf (P4): the recursive nb = 128 panel is not built; decide or record a kill.

## Small-n: rejected alternatives

| Alternative | Result | Verdict |
|---|---|---|
| potrf NB=16 in the SLM-resident kernel | 156-206 registers, 2x slower | dead |
| Packed sub-group getrf with the matrix in SLM | no win | dead |
| Launch-count fusion alone | <= 3% | dead |
| geqrf nb = 128 | no win | dead |

## Comment-density gate and its waivers {#small-n-comment-density-burn-down}

`.github/ci/check_comment_density.py` fails any file over 18% comment lines (rules: @ref dev_agent_guide
§12, exemptions: @ref documentation_conventions). The 2026-09-11 block in
`.github/ci/comment_density_waivers.txt` lists files that predated the gate; never add to it, and delete
a line when its file comes into band (a waiver that matches no file is an error).
