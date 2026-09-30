# SYEV: routing, tuning and measured results {#perf_syev}

**Covers:** the dense symmetric/Hermitian eigensolver `syev`. That means which provider `Auto`
picks per scalar type and `n` (CTA family, blocked, two-stage, vendor), the small-n kernel
choice, the block-size and stedc constants syev consumes, and the 2026-08 retune and
implementation-plan campaign: its results, rejected ideas, open debts and measurement traps.
**Status:** current. The routing and constants quoted in the first section are what ships.
Later sections are the evidence behind them. Anything a later measurement overturned is
marked *superseded*, with the replacement named.
**Machine:** RTX 4090 (sm_89), device 1 only, CUDA 13.2, cuSOLVER 12.2.0.1, driver 595.84,
unless a table says otherwise.
**Measured:** 2026-08-03 (retune Part 1), 2026-08-04 (Parts 2-4), 2026-08-05 (Part 5),
2026-08-07 (per-type routing, complex small n, constant retune), 2026-08-08 (implementation
plan baselines and results). All times are µs per matrix unless a table says ms.

This page replaces six root-level notes: `SYEV_RETUNE_RESULTS.md`, `SYEV_RETUNE_WORKFLOW_PLAN.md`,
`SYEV_PLAN_RESULTS.md`, `SYEV_PLAN_BASELINES.md`, `SYEV_PERF_IMPLEMENTATION_PLAN.md` and
`SYEV_PERF_IDEATION.md`. It also holds the benchmark grids that used to live as comments in
`include/batchlas/blas/functions/syev.hh`, which were cut in `526601e6`. The research profile
those plans were derived from (`SYEV_PERF_RESEARCH.md`, branch `worktree-syev-perf-research`,
commit `faa4f39`) was never merged; its figures survive here only where a later measurement
reproduced them. `JACOBI_EIGENSOLVER_PLAN.md`, which the sources cite for the Jacobi accuracy
and early speed figures, was never committed to any branch. Its figures are quoted here as the
sources quoted them, and there is no file to follow.

[TOC]

## syev: what ships today

Read the predicate, not this prose. Every line below was checked against the tree when this
page was written.

| decision | predicate | rule |
|---|---|---|
| eigenvectors, n > 32 | `syev_saturated_algorithm_for_n<T>`, `include/batchlas/blas/functions/syev.hh:283` | float: blocked to 448, two-stage 449..1024, vendor above. double: blocked to 448, then vendor. complex<float>: blocked to 512, then vendor. complex<double>: blocked to 256, then vendor |
| eigenvalues only, n > 32 | `syev_saturated_algorithm_for_n_values`, `syev.hh:309` | blocked to 320, two-stage above, every type |
| n ≤ 32 kernel | `syev_choose_small_kernel`, `syev.hh:238` | double: Jacobi. float: Jacobi to 8, `cta_fused` above. complex<float>: `cta_fused` to 8, `cta` above. complex<double>: `cta` |
| n ≤ 32 vendor handover, eigenvectors | `syev_cta_max_n_default_for`, `syev.hh:191` | 24 for complex<double>, 32 (off) otherwise. `BATCHLAS_SYEV_CTA_MAX_N` overrides |
| backend scope | `RouteTable<Op::syev,T>::preferred`, `syev.hh:392` | CUDA only. Other backends keep the historical order walk |
| legacy batch-keyed rule | `syev_prefer_vendor`, `syev.hh:181` | dead for n > 32. Kept so widening the n > 32 branch cannot drop it |
| values-only blocked solve | `syev_blocked.cc:254` and `:414` | `stebz` bisection off the tridiagonal, no `stedc` |
| complex panel width | `sytrd_block_size_default<T>`, `src/extensions/syev_blocked.cc:88` | 32 for complex at 256 < n ≤ 512, harness value otherwise |
| complex trailing update | `rank2k_trailing_update_supported`, `src/extensions/sytrd_blocked.cc:828` | float and complex<float> take the rank-2k (`syr2k`/`her2k`) path. double and complex<double> keep the GEMM pair |
| sb2st back-transform geometry | `sb2st_back_tile_for` / `sb2st_back_subs_for`, `src/extensions/sytrd_sb2st_hh.cc:762,772` | complex<float> at n = 512: tile 2, subs 4. Otherwise the shape-adaptive heuristic |
| block sizes | `include/batchlas/tuning_params.hh:61,131` | ORMQR 16/16/24/48/56, SYTRD 8/8/16/8/48 (buckets n ≤ 64/128/256/512/else) |
| sy2sb WY width | `SY2SB_ORMQR_NB_LARGE = 32`, `tuning_params.hh:128` | n ≤ 512 bucket only. The other buckets use the shape gate in `sytrd_sy2sb.cc` |
| stedc | `tuning_params.hh:149,177,199` | leaf threshold 32, merge variant 2 (FusedCta), threads-per-root 8, wg multiplier 8 |
| stedc leaf cap | `plan_stedc_levels`, `src/extensions/stedc_levels_plan.hh:26` | leaf ≤ threshold, a hard cap |
| two-stage band width | `choose_two_stage_kd`, `src/extensions/two_stage_common.hh:40` | kd = 32 |

## syev: eigenvector routing at saturation

The rule in force is keyed on `n` alone, never on batch. A caller at tiny batch pays launch
overhead whatever is picked, and tuning for that regime costs the saturated regime real
throughput.

### syev: the 2026-08-04 saturated float grid

Float, eigenvectors, median of 3, one process on the device, build `12963a8`. The batch for each
`n` is the largest at which all three providers fit. Blocked and two-stage carry much larger
workspaces than the vendor path.

| n | batch | blocked | vendor | two_stage | winner | margin | old Auto | new Auto |
|---|---|---|---|---|---|---|---|---|
| 64 | 16384 | **1.64** | 2.02 | 2.21 | blocked | 1.23× | vendor ✗ | blocked |
| 128 | 4069 | **6.65** | 7.85 | 10.35 | blocked | 1.18× | vendor ✗ | blocked |
| 256 | 2034 | **36.92** | 37.14 | 57.56 | blocked | 1.01× | vendor | blocked |
| 320 | 1302 | **74.53** | 215.23 | 111.20 | blocked | 1.49× | blocked ✓ | blocked |
| 512 | 508 | 384.27 | 504.69 | **380.02** | two_stage | 1.01× | blocked | two_stage |
| 640 | 651 | 836.30 | 1008.06 | **727.48** | two_stage | 1.15× | blocked ✗ | two_stage |
| 768 | 452 | 1612.98 | 1414.99 | **1184.11** | two_stage | 1.19× | vendor ✗ | two_stage |
| 1024 | 254 | 4089.02 | 2706.84 | **2441.93** | two_stage | 1.11× | vendor ✗ | two_stage |
| 2048 | 64 | 33842.6 | **15019.1** | 24782.3 | vendor | 1.65× | vendor ✓ | vendor |

The old batch-keyed routing was wrong at five of nine sizes, by 1.11×–1.23×. `syev_prefer_vendor`
was taken out of the eigenvector path rather than retuned, because its structure is batch-keyed
and every grid that fed it was built from unsaturated cells. With the rule in place, `Auto`
dispatched to the measured winner at every shape. Ratio of Auto to expected, idle GPU: 1.000,
0.999, 1.001, 1.000, 1.000, 1.002 at n = 64, 128, 320, 640, 1024, 2048. Tests reproduced the
baseline exactly: 16 pass and 3 fail (`lanczos`, `steqr`, `stedc`, all pre-existing then).

For n ≤ 32, at batch 2048, `cta`/Jacobi (n = 16) and `cta`/fused (n = 32) beat every non-CTA
provider by 2.8×–3.0×, so checking the CTA route first is correct.

**Caveats, all still open:**

- n = 2048 was measured at batch 64, below the 128 SMs. It is memory-limited rather than
  saturated, because blocked and two-stage cannot fit a larger batch. The 2048 boundary rests
  on the weakest row in the table.
- This grid is float only. The per-type re-measure below found that the other types differ.

### syev: the blocked-over-cuSOLVER headline measurement

The README's Performance section quotes this measurement. Its grid is 2026-08-07, float,
eigenvectors, median of **5**, harness-default block size (`nb`), one process on the device.
It fills the gap between n = 320 and n = 512 that the 2026-08-04 grid jumped over. Nothing
there had checked where blocked actually stops winning:

| n | batch | blocked | two_stage | vendor | winner | routing before |
|---|---|---|---|---|---|---|
| 320 | 819 | **67.84** | 113.39 | 203.00 | blocked (3.0× over vendor) | blocked, ok |
| 384 | 682 | **120.99** | 179.18 | 309.10 | blocked | two_stage, 1.48× loss |
| 448 | 585 | **195.27** | 262.79 | 400.59 | blocked (2.1× over vendor) | two_stage, 1.35× loss |
| 512 | 512 | 326.71 | 332.55 | 504.40 | tie | two_stage, neutral |
| 640 | 256 | 687.94 | 674.25 | 905.04 | tie | two_stage, ok |
| 768 | 192 | 1241.90 | **1172.90** | 1348.30 | two_stage | two_stage, ok |
| 1024 | 128 | 3296.80 | 2614.70 | 2552.10 | tie (vendor/two-stage) | two_stage, neutral |

So the real float crossover is 448, not 320, and the two rows in between cost 1.35×–1.48×.
Two-stage in eigenvector mode is worth at most 1.06× even for float. It ties blocked at
512/640, wins 1.06× at 768 and is within 1.02× of the vendor at 1024. All of those are neutral,
so the committed two-stage band 449..1024 was kept rather than churned. Commit `73eb5510`
shipped this together with the per-type rule below.

### syev: per-type eigenvector crossovers

The rule had been measured on float and applied to every type. Same method as the headline
grid, 2026-08-07. Blocked is at its own best `nb` (see
[the complex panel width](#syev-complex-panel-width-in-the-256-to-512-bucket)).

complex<float>, before the Annex G fix below:

| n | batch | blocked | two_stage | vendor | winner | routing before |
|---|---|---|---|---|---|---|
| 64 | 4096 | 2.36 | 5.41 | 2.38 | tie | blocked, ok |
| 96 | 2730 | 5.92 | 13.24 | 7.38 | blocked | blocked, ok |
| 128 | 2048 | 11.51 | 25.35 | 11.03 | tie | blocked, ok |
| 192 | 1365 | 33.83 | 72.62 | 78.97 | blocked | blocked, ok |
| 256 | 1024 | 74.13 | 158.23 | 126.88 | blocked | blocked, ok |
| 320 | 819 | 166.27 | 296.94 | 247.59 | blocked | blocked, ok |
| 384 | 682 | 304.06 | 419.48 | 363.64 | blocked | two_stage, 1.38× loss |
| 448 | 585 | 540.06 | 677.67 | 530.16 | tie (blocked/vendor) | two_stage, 1.25× loss |
| 512 | 512 | 802.74 | 973.80 | 707.05 | vendor | two_stage, 1.38× loss |
| 640 | 256 | 1557.10 | 1548.50 | 1265.90 | vendor | two_stage, 1.22× loss |
| 768 | 192 | 2872.50 | 2919.60 | 1957.20 | vendor | two_stage, 1.49× loss |
| 1024 | 128 | 7706.30 | 6816.10 | 4036.20 | vendor | two_stage, 1.69× loss |

Two-stage is never the complex winner at any `n`, in eigenvector mode.

double and complex<double> (n/batch, µs/matrix):

| type | n/batch | blocked | two_stage | vendor | winner |
|---|---|---|---|---|---|
| double | 384/341 | 771.15 | 1012.70 | 874.97 | blocked |
| double | 448/292 | 1193.00 | 1548.80 | 1199.80 | tie (blocked/vendor) |
| double | 512/256 | 1662.80 | 2055.80 | 1617.90 | vendor (two-stage 1.27× loss) |
| double | 640/128 | 3035.10 | 4434.40 | 2860.30 | vendor (two-stage 1.55× loss) |
| double | 768/96 | 5389.50 | 7734.80 | 4544.10 | vendor (two-stage 1.70× loss) |
| double | 1024/64 | 12797.00 | 18128.00 | 9177.70 | vendor (two-stage 1.98× loss) |
| complex<double> | 128/1024 | 108.09 | 201.68 | 156.18 | blocked |
| complex<double> | 192/682 | 342.50 | 625.98 | 385.63 | blocked |
| complex<double> | 224/585 | 489.86 | 946.78 | 549.91 | blocked |
| complex<double> | 256/512 | 649.59 | 1313.40 | 723.36 | blocked |
| complex<double> | 288/455 | 1131.00 | 2031.90 | 974.79 | vendor |
| complex<double> | 320/409 | 1498.50 | 2751.60 | 1258.90 | vendor |
| complex<double> | 448/292 | 3485.50 | 6715.40 | 2825.70 | vendor |

complex<double>'s blocked crossover is 256. The card runs FP64 at 1/64 rate, which penalises
the native panel far more than cuSOLVER. That makes this the most hardware-specific boundary
in the table: re-measure it on a data-center GPU before trusting it there.

After the Annex G fix, complex<double> at n = 288/320/384/448/512 moved from a 1.13×–1.23×
vendor win to a dead heat: blocked 1009.6/1318.9/1961.4/3047.6/3874.7 against vendor
973.9/1258.4/1974.1/2826.4/3929.1. Every cell is inside the 1.10× neutral band, so the boundary
stays at 256, the last `n` with a real margin.

End-to-end gain of old Auto over new Auto, commit `73eb5510`. Eigenvectors, median of 3,
same batch for both:

| type | n=320 | 384 | 448 | 512 | 640 | 768 | 1024 |
|---|---|---|---|---|---|---|---|
| complex<float> | 1.18× | 1.34× | 1.25× | 1.38× | 1.22× | 1.49× | 1.69× |
| complex<double> | 1.22× | 2.17× | 2.38× | 2.30× | - | 2.88× | - |
| double | 1.00× | 1.31× | 1.30× | 1.27× | 1.55× | 1.70× | 1.97× |
| float | 1.00× | 1.50× | 1.35× | 1.00× | 1.00× | 1.00× | 1.00× |

No cell regresses in either mode.

### syev: why complex loses to cuSOLVER above the crossover

The complex path is not broken. The float path is much better optimised than the vendor's, and
complex costs BatchLAS proportionally more than it costs cuSOLVER. n = 512, batch = 512:

| | float | complex | complex/float |
|---|---|---|---|
| blocked, values | 272.61 | 687.27 | 2.52 |
| blocked, vectors | 328.53 | 802.62 | 2.44 |
| two_stage, values | 151.45 | 440.56 | 2.91 |
| two_stage, vectors | 332.26 | 973.60 | 2.93 |
| cuSOLVER, values | 453.54 | 626.90 | 1.38 |
| cuSOLVER, vectors | 503.03 | 707.10 | 1.41 |

At n = 512 blocked was 1.53× faster than cuSOLVER in float and 1.14× slower in complex. The
penalty is the same in both modes, so it is not the back-transform. It is uniform across the
solve. It sits in `sytrd_blocked.panel_only`, which is 68% of the complex solve and 3.85× its
float cost at matched `nb`.

**Partly fixed (`e4186908`).** Inside the panel symv, clang emitted the C99 Annex G complex
multiply (a per-element `isnan` branch plus a `__mulsc3` call). Writing that one loop's
multiply-add out in real arithmetic made the panel kernel 1.22×–1.29× faster. The note on
`mac` in `latrd_lower_panel.cc` records it. Re-measured complex<float>, same method:

| n | batch | blocked | vendor | winner |
|---|---|---|---|---|
| 64 | 4096 | 2.21 | 2.38 | blocked 1.08× |
| 96 | 2730 | 5.55 | 7.40 | blocked 1.33× |
| 128 | 2048 | 10.98 | 11.02 | blocked (was 1.05× behind) |
| 192 | 1365 | 30.60 | 79.41 | blocked 2.59× |
| 256 | 1024 | 66.84 | 126.97 | blocked 1.90× |
| 320 | 819 | 165.54 | 247.37 | blocked 1.49× |
| 384 | 682 | 253.24 | 363.47 | blocked 1.44× |
| 448 | 585 | 424.44 | 529.78 | blocked 1.25× |
| 512 | 512 | 686.19 | 706.96 | blocked 1.03× (crossed over) |
| 640 | 256 | 1357.40 | 1267.80 | vendor 1.07× |
| 768 | 192 | 2335.10 | 1958.40 | vendor 1.19× |
| 1024 | 128 | 5236.90 | 4047.90 | vendor 1.29× |

This is why the complex<float> boundary is 512, not 448. complex<float> now beats cuSOLVER at
every `n` from 4 to 512. The real types are unchanged, because `mac` routes them to a plain
multiply-add. What is left above 512 is the rest of the same gap. The complex panel ratio is now
2.14× float, against the roughly 4× the arithmetic implies. The kernel reaches only about
330 GB/s of this card's roughly 1000 GB/s in *both* types, so it is latency- or
occupancy-bound. See [the open panel symv](#syev-open-debt-the-single-read-panel-symv).

### syev: eigenvalues-only routing

This mode could be measured against the vendor only once `syev_benchmark` grew a `jobz`
argument. The benchmark had hardcoded `JobType::EigenVectors`, so there was no vendor arm to
compare against. Float, 2026-08-04, same method as the saturated eigenvector grid:

| n | batch | blocked | vendor | two_stage | winner | margin |
|---|---|---|---|---|---|---|
| 64 | 16384 | **1.06** | 1.12 | 1.14 | blocked | 1.06× |
| 128 | 4069 | **4.41** | 5.34 | 5.35 | blocked | 1.21× |
| 256 | 2034 | **24.78** | 27.20 | 27.98 | blocked | 1.10× |
| 320 | 2604 | **48.16** | 200.94 | 50.35 | blocked | 1.05× |
| 512 | 1017 | 298.30 | 458.95 | **158.72** | two_stage | 1.88× |
| 640 | 651 | 687.67 | 929.05 | **330.71** | two_stage | 2.08× |
| 768 | 452 | 1362.46 | 1301.59 | **505.29** | two_stage | 2.58× |
| 1024 | 254 | 3537.41 | 2494.44 | **908.32** | two_stage | 2.75× |
| 2048 | 64 | 29547.70 | 13908.80 | **10804.10** | two_stage | 1.29× |

Two-stage keeps winning all the way to 2048, and by 1.88×–2.75× against at most 1.19× with
eigenvectors. With no eigenvectors to produce, two-stage never pays to apply \f$Q_2\f$, so its
cheap band reduction survives into wall-clock time. These figures predate the `stebz` change
([WP1](#syev-stebz-values-only-in-the-blocked-solver-wp1)), which made the blocked column
1.36×–1.87× faster up to n = 320. The 320 | 512 boundary was not re-derived afterwards (see
[open debts](#syev-open-debts-and-unmeasured-items)).

This rule replaces `syev_prefer_two_stage_values`, whose `batch >= 256` term did real damage.
The n = 1024 cell above ran at batch 254, fell through to the vendor and paid 2.75×. Two
callers two matrices apart got different providers for reasons unrelated to which kernel is
better. Adding a batch floor back to this function reopens that hole.

Two-stage is the complex winner in this mode too above n = 320, and beats cuSOLVER by 1.30× at
n ≥ 768 despite the roughly 2.9× complex penalty, because its float baseline is about 3× better
than the vendor's. The per-type split was therefore not applied here.

Pre-saturation record: the first eigenvalues-only table, measured right after the stage-2 chase
fix at `27851a6` and before routing was keyed on `n`:

| n | batch | blocked | two_stage | vendor | two_stage vs vendor |
|---|---|---|---|---|---|
| 256 | 256 | 37.8 | 48.3 | 27.3 | 0.57× |
| 256 | 1024 | 28.1 | 30.0 | 27.5 | 0.92× |
| 512 | 256 | 266.8 | 209.6 | 423.1 | 2.02× |
| 512 | 1024 | 289.0 | 161.4 | 461.1 | 2.86× |
| 1024 | 256 | 3408.5 | 1110.3 | 2505.2 | 2.26× |
| 1024 | 1024 | 3603.1 | 1156.6 | 2626.3 | 2.27× |

At batch 1 the vendor won by 12.5× at n = 1024. That is the batch-starvation defect of both
reductions.

### syev: the superseded batch-keyed eigenvector grids

*Superseded* by the saturated grid above. These grids are kept as a record of the
batch-dependent behaviour. Do not reinstate a batch-keyed rule from them.

The original grid, measured at `27851a6`, float, blocked/vendor (> 1 means the vendor wins):

| n \\ batch | 1 | 8 | 16 | 32 | 64 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|---|---|---|---|
| 64 | 2.06 | 4.38 | 4.44 | - | 3.93 | - | 2.28 | - | - |
| 256 | 4.31 | 2.68 | 2.61 | - | 2.19 | 1.84 | 1.27 | 1.22 | - |
| 320 | - | - | - | 1.34 | 1.06 | 0.79 | 0.69 | - | - |
| 512 | 6.45 | 2.07 | 1.91 | 1.62 | 1.28 | 0.86 | 0.74 | 0.74 | 0.72 |
| 640 | - | - | - | 1.66 | 1.19 | 0.85 | 0.86 | - | - |
| 896 | - | - | - | 1.78 | 1.18 | 1.19 | 1.24 | - | - |
| 1024 | 15.33 | 3.33 | 2.78 | - | 1.35 | 1.44 | 1.46 | 1.46 | 1.51 |

From it came the carve-out "blocked for 320 ≤ n ≤ 640 and batch ≥ 128" (blocked ahead by up
to 1.37×). The carve-out was stated as measured rather than smoothed, because it was not
monotone in `n`. At n = 1024, batch = 1 the panel kernel was 88% of the solve and moved its
roughly 134 MB per panel at about 1/48 of device bandwidth: one work-group per matrix. An earlier
version of this grid was corrupted. `--name=BM_SYEVX_Crossover` is a substring filter, so it
also ran `BM_SYEVX_CrossoverVectors`, and the collector keyed rows by `(n, batch)`, so
eigenvector rows silently overwrote eigenvalues-only rows.

Re-measured 2026-08-04 (build `12963a8`, median of 3) after grid-`latrd` landed (`87f6887`,
defaulted on at n ≥ 768 by `5401f63`):

| n \\ batch | 1 | 8 | 16 | 32 | 64 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|---|---|---|---|
| 64 | 2.08 | 4.43 | 4.48 | 4.32 | 3.93 | 3.22 | 2.29 | 1.55 | 1.11 |
| 128 | 2.00 | 4.16 | 4.09 | 3.79 | 3.39 | 2.71 | 1.72 | 1.22 | 0.98 |
| 256 | 3.96 | 2.67 | 2.59 | 2.44 | 2.19 | 1.87 | 1.26 | 1.21 | 1.08 |
| 320 | 5.00 | 1.23 | 1.35 | 1.24 | 1.03 | 0.79 | 0.68 | 0.47 | 0.36 |
| 512 | 6.42 | 2.07 | 1.93 | 1.62 | 1.27 | 0.86 | 0.73 | 0.74 | - |
| 640 | 8.78 | 2.27 | 2.03 | 1.67 | 1.18 | 0.85 | 0.86 | 0.81 | - |
| 768 | 9.99 | 2.56 | 2.21 | 1.73 | 1.10 | 1.05 | 1.07 | - | - |
| 896 | 10.05 | 2.27 | 1.93 | 1.52 | 1.06 | 1.19 | 1.24 | - | - |
| 1024 | 10.14 | 2.29 | 1.97 | 1.50 | 1.15 | 1.44 | - | - | - |

What it showed correctly: n = 1024 improved a lot (15.33 → 10.14 at batch 1, 3.33 → 2.29 at
batch 8), which is grid-`latrd` doing its job. The blocked win at n = 320 keeps growing with
batch, reaching 0.36 (2.8× faster than cuSOLVER) at batch 1024. What it got wrong: it concluded
that the routing needed no edit. Its ladder stopped at 1024, which for small `n` is nowhere near
saturation. At n = 64 the crossover needs batch around 16384, where blocked wins by 1.23×, while
this grid shows the vendor 1.11× ahead. Small batch flatters the vendor, whose fixed launch cost
is lowest.

### syev: Uplo::Upper by mirroring

`sytrd_blocked` threw on `Uplo::Upper`, and the blocked and two-stage support predicates
rejected it, so every Upper call went to cuSOLVER whatever the shape. Upper also had no test
coverage anywhere in the syev suite. A Hermitian matrix's two triangles carry the same operator,
so mirroring the upper triangle into the lower one lets the Lower pipeline produce identical
results. The mirror is \f$O(n^2)\f$ in front of an \f$O(n^3)\f$ solve (`src/extensions/uplo_mirror.cc`).
Float, 2026-08-04, median of 3, single process:

| n | batch | mode | Auto (Upper) | vendor (Upper) | speedup | Auto (Lower) | Upper/Lower |
|---|---|---|---|---|---|---|---|
| 128 | 4069 | vectors | 6.72 | 8.22 | 1.22× | 6.64 | 1.013× |
| 256 | 2034 | vectors | 37.26 | 37.89 | 1.02× | 36.91 | 1.009× |
| 320 | 1302 | vectors | 75.00 | 221.25 | 2.95× | 74.39 | 1.008× |
| 512 | 508 | vectors | 381.94 | 532.71 | 1.39× | 380.09 | 1.005× |
| 1024 | 254 | vectors | 2459.62 | 3314.37 | 1.35× | 2452.86 | 1.003× |
| 128 | 4069 | values | 4.51 | 5.61 | 1.24× | 4.42 | 1.020× |
| 256 | 2034 | values | 25.19 | 27.85 | 1.11× | 24.87 | 1.013× |
| 320 | 1302 | values | 51.82 | 197.22 | 3.81× | 51.10 | 1.014× |
| 512 | 508 | values | 171.90 | 482.48 | 2.81× | 170.43 | 1.009× |
| 1024 | 254 | values | 915.92 | 3102.78 | 3.39× | 907.97 | 1.009× |

The mirror costs 0.3–2.0%, and Upper now beats the vendor at every measured shape. A native
Upper variant of `sytrd_blocked`, `sytrd_sy2sb` and `sytrd_sb2st` could recover at most that
0.3–2.0%, at the price of an Upper variant of every reduction kernel. It was rejected on these
numbers.

The tests are two cases in `tests/syev_blocked_tests.cc`, one per call site, over all four
types. The obvious test would be vacuous: `Matrix::Random(..., symmetric=true)` is symmetric,
so Upper and Lower are interchangeable. The fixture therefore poisons the strictly-lower
triangle after taking the reference from the upper one. It then asserts that a Lower-read solve
of the same matrix gives a spectrum that differs by more than 1.0, so the test fails with
"fixture is vacuous" if the poisoning ever stops taking effect.

## syev: small-n kernel choice

The three CTA-family kernels are `syev_cta`, `syev_cta_fused` and `syev_jacobi_cta`. Force one
with `BATCHLAS_SYEV_SMALL_KERNEL=cta|fused|jacobi`.

### syev: the 2026-08-03 small-n bake-off

Build `7911847`. Median of 5 process-level repeats with IQR, each cell at its own measured knee
batch, best over `cta_wg_size_multiplier`. A margin below 1.10× or with overlapping IQRs is
*neutral*. `syev_cta_fused_benchmark` and `syev_jacobi_cta_benchmark` each register **two**
benchmarks (`BM_SYEV_CTA_PIPELINED`, `BM_SYEV_CTA_TRIDIAG_REF`), so results are keyed on the
CSV `name` column.

| type | n | mode | winner | wg | µs/matrix | runner-up | ratio | verdict | vs `syev_cta` | vs cuSOLVER | batch |
|---|---|---|---|---|---|---|---|---|---|---|---|
| double | 4 | values | jacobi | 4 | 0.00718 | cta_fused | 3.59× | clean | 3.87× | - | 16384 |
| double | 4 | vectors | jacobi | 2 | 0.01072 | cta_fused | 3.05× | clean | 3.75× | 15.17× | 4096 |
| double | 8 | values | jacobi | 4 | 0.04809 | tridiag_ref | 3.52× | clean | 3.75× | - | 16384 |
| double | 8 | vectors | jacobi | 2 | 0.11924 | cta_fused | 2.83× | clean | 2.91× | 3.70× | 1024 |
| double | 16 | values | jacobi | 1 | 0.33064 | cta_fused | 2.55× | clean | 2.56× | - | 16384 |
| double | 16 | vectors | jacobi | 1 | 0.49690 | cta_fused | 2.49× | clean | 2.50× | 3.29× | 1024 |
| double | 32 | values | jacobi | 4 | 2.35063 | cta_fused | 1.58× | clean | 1.70× | - | 4096 |
| double | 32 | vectors | jacobi | 2 | 3.31753 | vendor | 1.37× | clean | - | 1.37× | 1024 |
| float | 4 | values | jacobi | 2 | 0.00102 | cta_fused | 1.74× | clean | 3.84× | - | 16384 |
| float | 4 | vectors | jacobi | 1 | 0.00104 | cta_fused | 1.71× | clean | 4.55× | 18.57× | 16384 |
| float | 8 | values | jacobi | 2 | 0.00391 | cta_fused | 1.56× | clean | 2.23× | - | 16384 |
| float | 8 | vectors | jacobi | 2 | 0.00478 | cta_fused | 1.37× | clean | 2.55× | 8.42× | 16384 |
| float | 16 | values | cta_fused | 2 | 0.02740 | jacobi | 1.10× | *neutral* | 1.17× | - | 16384 |
| float | 16 | vectors | cta_fused | 1 | 0.02933 | jacobi | 1.22× | clean | 1.25× | 4.10× | 16384 |
| float | 32 | values | cta_fused | 2 | 0.11934 | tridiag_ref | 1.03× | *neutral* | 1.03× | - | 16384 |
| float | 32 | vectors | cta_fused | 2 | 0.28426 | tridiag_ref | 1.11× | clean | 1.12× | 1.79× | 1024 |

`syev_cta`, the kernel `Auto` then sent every n ≤ 32 to, did not win a single cell in either
precision. The two winners were both unreachable from `Auto`. The values-mode cells have no
vendor column because `syev_benchmark` had no `jobz` argument then. Routing these winners made
BatchLAS beat cuSOLVER across the whole n = 4..32 range for the real types, by 1.1×–3.9× over
`syev_cta`. That is now the real branch of `syev_choose_small_kernel`. Double uses Jacobi
everywhere. Float uses Jacobi to n = 8 and `cta_fused` above. The float rule rests on the vector
cells, because the two values-mode cells at n = 16 and 32 are neutral. n = 9..15 was not in this
sweep. The shipped rule gives the whole 9..32 range to `cta_fused`.

A later A/B on real float found that the fused kernel with its tuned multiplier beats Jacobi from
n = 7 up. The 8 | 9 boundary was kept anyway, for Jacobi's relative accuracy on graded input.
See [steqr.md: small-n syev routing](steqr.md#small-n-syev-routing-measured-not-shipped).

**FP64 caveat, load-bearing.** This card runs FP64 at 1/64 rate, which inflates Jacobi's margin
over the tridiagonalising paths. The float column is the better predictor for a 1:2 FP64
datacenter GPU. The double rule should be gated on measured FP64 throughput rather than on the
scalar type, and re-measured before it is trusted on such hardware.

Accuracy is a bonus here, not the justification. The (uncommitted) Jacobi plan's §13.1 records
Jacobi resolving graded-SPD eigenvalues to a relative error of 4.5e-07 where `syev_cta` returns
2.7e+28. Jacobi also wins outright on speed.

*Superseded ideation-era claims.* Before this bake-off, the ideation note (written at `e2ff635`)
quoted the uncommitted Jacobi plan's §13.2 against `syev_cta` alone: double Jacobi faster at every
n ≤ 32 (1.2×–3.8×), float Jacobi winning to n = 16 (1.1×–3.9×) and losing at n = 32 (0.4×–0.6×).
Jacobi then had 30 passing tests and no routed user, and `syev_cta_fused` had no head-to-head
measurement in the tree at all. The bake-off above replaces those figures. It adds `cta_fused`
as a contender, which takes float n = 16 from Jacobi.

### syev: complex small-n kernels

Measured 2026-08-07. The code had previously said "never measured, keep the historical kernel".
complex<float>, eigenvectors, median of 3 at each cell's saturating batch:

| n | batch | cta | cta_fused | jacobi | vendor | winner |
|---|---|---|---|---|---|---|
| 4 | 65536 | 0.0035387 | 0.0014881 | 0.0013689 | 0.052079 | jacobi/fused |
| 5 | 52429 | 0.0067438 | 0.0055953 | 0.0056986 | 0.078717 | fused 1.21× |
| 6 | 43691 | 0.0102170 | 0.0073592 | 0.0078113 | 0.031830 | fused 1.39× |
| 7 | 37449 | 0.0105900 | 0.0084175 | 0.0119040 | 0.037963 | fused 1.26× |
| 8 | 32768 | 0.0132190 | 0.0105830 | 0.0143920 | 0.044342 | fused 1.25× |
| 9 | 29127 | 0.0213110 | 0.0281870 | 0.0544510 | 0.079653 | cta |
| 12 | 21845 | 0.0384370 | 0.0422290 | 0.0967020 | 0.103430 | cta |
| 16 | 16384 | 0.0670010 | 0.0655850 | 0.1857200 | 0.133530 | tie |
| 20 | 13107 | 0.1685500 | 0.2894300 | 0.9536300 | 0.286910 | cta |
| 24 | 10923 | 0.2204500 | 0.3810900 | 1.3444000 | 0.339650 | cta |
| 32 | 8192 | 0.3825800 | 0.5948600 | 2.4824000 | 0.470900 | cta |

`syev_cta` is right for complex from n ≥ 9. The float rule would have been wrong here: Jacobi is
4×–6× off the pace at n ≥ 20 in complex. The gap at n ≤ 8 is worth 1.21×–1.39×, and 2.38× at
n = 4. At n = 4 Jacobi edges fused by 1.09×, which is inside the neutral band, so one threshold
at 8 takes essentially all of it. For complex<double>, fused is ahead of `cta` by only
1.03×–1.08× from n = 4..16, all neutral, so it is not split.

### syev: the complex double vendor handover

`syev_cta_max_n_default_for` returns 24 for complex<double> only. Measured 2026-08-07,
eigenvectors, median of 3:

| n | batch | cta | cta_fused | vendor | winner |
|---|---|---|---|---|---|
| 18 | 14564 | 2.5293 | 2.9254 | 2.7744 | cta |
| 20 | 13107 | 3.0015 | 3.4597 | 3.1085 | cta |
| 22 | 11916 | 3.5318 | 4.0359 | 3.4564 | tie (vendor 1.02×) |
| 24 | 10923 | 4.0806 | 4.6301 | 3.8184 | tie (vendor 1.07×) |
| 26 | 10082 | 5.4993 | 5.3048 | 4.4835 | vendor 1.23× |
| 28 | 9362 | 6.2862 | 5.9845 | 4.9477 | vendor 1.27× |
| 30 | 8738 | 7.1533 | 6.6465 | 5.4089 | vendor 1.32× |
| 32 | 8192 | 8.0253 | 7.4929 | 5.6531 | vendor 1.42× |

This is an FP64-rate artifact and the most machine-specific number here: 1/64-rate FP64
throttles the CTA kernel far harder than it throttles cuSOLVER. complex<float> does not cross
over (`cta` still beats the vendor 1.23× at n = 32 and 1.38× at n = 28), so the rule is
deliberately not "all complex".

### syev: the LOBPCG projected-solve knob

`BATCHLAS_SYEV_CTA_MAX_N` (0..32, default 32 = off) sends small eigenvector solves above the
threshold to the vendor. It was introduced after one point showed the vendor ahead: n = 30,
batch = 8, float, eigenvectors, CTA 229.6 µs/call against cuSOLVER 103.7 (2.21×). nsys
attributed 29.4% of all LOBPCG GPU time to that projected Rayleigh–Ritz solve, roughly 16% end
to end. Measured through LOBPCG, eigenvectors, n = 256, µs/matrix:

| `BATCHLAS_SYEV_CTA_MAX_N` | 32 (off) | 16 | 8 | 0 |
|---|---|---|---|---|
| batch 8 | 15563 | 14211 | 13746 | 13551 |
| batch 64 | 1998.6 | 1890.4 | 1948.0 | 1945.9 |

That is 1.10×–1.15× at batch 8 and 1.03×–1.06× at batch 64, monotone in the threshold. It ships
**off** because moving the projected solve off CTA flips a marginal case in
`ILUKTests.SyevxInstrumentationAndPreconditioner`. The test asserts `lose_count == 0`. At
threshold 16, one case of eight (`d0.06_b0.5_s1234`, `iluk_k2`) crosses to ratio 1.25, at a
point where the baseline has already converged to 4.2e-06. It is a near-tie, but a red test,
and it is someone else's assertion. This is an **owner decision, still open**: either the
assertion tolerates a tie on an already-converged case, or that test pins its projected-solve
provider.

## syev: tuning constants and the shadowed header

### syev: the shadowed tuning header

Two files answer `#include <batchlas/tuning_params.hh>`: the committed
`include/batchlas/tuning_params.hh`, and the one CMake generates into
`${PROJECT_BINARY_DIR}/include/batchlas/` from `cmake/tuning_params.h.in`. `src/CMakeLists.txt`
lists `PROJECT_SOURCE_DIR/include` before `PROJECT_BINARY_DIR/include` (currently lines
154-155 and 443-444), so **the committed header wins on every build**. The CMake defaults in
`cmake/BatchLASGeneratedHeaders.cmake` are dead, and so is any profile-driven retune that
`batchlas_tuning_header` writes to the binary dir.

This was confirmed on 2026-08-03 by running code, not by reading CMake. A throwaway TU compiled
with the real `-I` flags from `flags.make` printed `ormqr_block_size_for_n(512) = 16` and
`sytrd_block_size_for_n(512) = 24`. The generated header would have said 128 and 64.

**Still a live trap.** The include order and the generator path are both unchanged. A retune
takes effect only when its constants are ported by hand into the committed header. The
recommended fix was to keep the committed header as the single source of truth and retarget or
delete the CMake generator path. On 2026-08-03 the profile-driven pipeline was demonstrably
unused (no `build/tuning/profile.json` anywhere, `BATCHLAS_TUNING_PROFILE` empty in the cache).
The 2026-08-07 retune below went through `evaluation/tuning/` and landed as an edit of the
committed `include/batchlas/tuning_params.hh` (`924b3a59`). That has **not** been done. It was option (c) of three: (a)
delete the committed header, which breaks non-CMake consumers; (b) reorder the include dirs,
which silently changes every constant.

The enabling half did land: `7911847` made every accessor runtime-overridable through
`BATCHLAS_TUNE_*`. Overrides are read fresh on every call and feed `*_buffer_size` as well as
the solve, so they must not change between a buffer-size query and its call.

### syev: the ORMQR regression hypothesis, refuted

The 2026-08 retune started from a hypothesis. The shadowing had pinned ORMQR at 16 in every
bucket where the generated header said 16/32/64/128/128, and that loss was supposed to be why
BatchLAS "stopped beating cuSOLVER for n = 32..512". Three independent refutations:

1. **History.** ORMQR had been 16 in every bucket since the header was created (`7363746`,
   2026-03-02) and was never modified. There was never a larger value in the compiled path.
   What did move was SYTRD: 8/16/16/16/16 → 8/8/16/24/32 in `91341c1`, a 33-file refactor with
   no measurement attached.
2. **Direct A/B.** Forcing ORMQR = 128 at n = 512, eigenvectors, batch 128, median of 5
   interleaved rounds: 452.32 µs/matrix (IQR 450.73–454.91) against 301.95 (IQR 300.79–303.75)
   unset. **1.50× slower**, IQRs disjoint.
3. **The sweep below**: 16 was the measured optimum wherever the knob had any effect.

**There was also no baseline to restore to.** An exhaustive search of git history (every CSV,
plot, notebook and markdown table across 400+ revisions) found no artifact anywhere recording
BatchLAS beating cuSOLVER at n = 32..512, or at any `n`. The only in-repo statement of the claim
was the goal line of the retune plan. It may have been true of an uncommitted run. Nothing may
calibrate against it.

The values were never tuned for this solver, though. They came from a standalone
`ormqr_blocked` microbenchmark at batch 8192–512, not from `syev`. At n ≤ 64 the search space
offered only `block_size ∈ [4,8,12,16]`, so 16 was the *ceiling of the search*. And the
generator's `_derive_param_buckets` keeps the parameters of whichever case had the lower
absolute time, rather than the best block size per case.

### syev: the 2026-08-03 block-size sweep in the syev context

*Superseded by the 2026-08-07 retune below.* It is kept because it was the first measurement of
these knobs through `syev` at all. Knobs `BATCHLAS_TUNE_ORMQR_BLOCK_SIZE` and
`BATCHLAS_TUNE_SYTRD_BLOCK_SIZE`, blocked provider, knee batch (capped at 512 because of the
[workspace defect](#syev-workspace-footprint-limits-usable-batch)), float:

| n | mode | knob | best | committed then | verdict | sweep (µs/matrix) |
|---|---|---|---|---|---|---|
| 32 | vectors | ormqr | 16 | 16 | = committed | 8:0.747 **16:0.708** 32:0.851 64:0.844 128:0.847 |
| 64 | vectors | ormqr | 16 | 16 | = committed | 8:2.420 **16:2.297** 32:2.661 64:4.869 128:5.024 |
| 128 | vectors | ormqr | 16 | 16 | = committed | 8:7.861 **16:7.446** 32:8.142 64:12.742 128:22.805 |
| 256 | vectors | ormqr | 32 | 16 | 1.016× neutral | 8:50.756 16:44.615 32:43.932 64:51.573 128:72.105 |
| 512 | vectors | ormqr | 32 | 16 | 1.059× neutral | 8:395.5 16:340.1 32:321.0 64:337.5 128:420.7 |
| 64 | vectors | sytrd | 8 | 8 | = committed | **8:2.298** 16:2.439 24:2.564 32:2.785 48:3.001 64:3.567 |
| 128 | vectors | sytrd | 8 | 8 | = committed | **8:7.456** 16:7.629 24:8.431 32:8.773 48:10.534 64:11.123 |
| 256 | vectors | sytrd | 16 | 16 | = committed | 8:47.055 **16:44.592** 24:45.887 32:48.041 48:53.468 |
| 512 | vectors | sytrd | 24 | 24 | = committed | 8:380.8 16:343.4 **24:339.9** 32:342.6 48:359.3 |
| 512 | values | sytrd | 24 | 24 | = committed | 8:299.2 16:262.5 **24:259.7** 32:261.2 48:277.3 |

Double confirmed float independently:

| n | mode | knob | best | committed then | verdict | sweep (µs/matrix) |
|---|---|---|---|---|---|---|
| 32 | vectors | ormqr | 16 | 16 | = committed | 8:4.877 **16:4.842** 32:5.924 64:5.926 128:5.924 |
| 64 | vectors | ormqr | 16 | 16 | = committed | 8:13.574 **16:12.821** 32:14.809 64:26.680 128:26.567 |
| 128 | vectors | ormqr | 16 | 16 | = committed | 8:50.395 **16:43.639** 32:46.494 64:72.031 128:117.373 |
| 512 | vectors | ormqr | 32 | 16 | 1.053× neutral | 8:2117.0 16:1732.3 **32:1645.1** 64:1814.6 128:2312.2 |
| 64 | vectors | sytrd | 8 | 8 | = committed | **8:12.833** 16:13.328 24:13.726 32:14.402 48:15.046 |
| 128 | vectors | sytrd | 8 | 8 | = committed | **8:43.648** 16:45.102 24:46.910 32:48.398 48:52.887 |
| 512 | values | sytrd | 16 | 24 | 1.066× neutral | 8:1355.2 **16:1202.6** 24:1281.3 32:1255.5 48:1331.7 |
| 512 | vectors | sytrd | 16 | 24 | 1.048× neutral | 8:1805.4 **16:1652.4** 24:1732.0 32:1705.4 48:1782.3 |

ORMQR = 128, the value the hypothesis favoured, cost 2.1× at n = 64 and 3.1× at n = 128 (2.7× in
double). In eigenvalues-only mode the ORMQR sweep was flat to three decimals (n = 128:
7.355/7.355/7.355/7.360/7.352). That is correct, because there is no back-transform, and it
confirms that the override is wired to the path it claims to control. The SYTRD sweep was not
flat, so that knob is live. The consistent sub-threshold candidates at n = 512 were SYTRD = 16 in
double (both modes) and ORMQR = 32 (both precisions). That was more agreement than noise usually
manages, and it pointed at the XLARGE/LARGE buckets.

Coverage gap: n = 256 was missing from this phase entirely, in both modes and both precisions.
`syev_blocked` exceeded the 3 GB per-invocation abort even at batch 476. Absolute µs/matrix at
small `n` in this phase is below saturation. Only the ranking was read.

### syev: the 2026-08-07 constant retune

Commit `924b3a59` retuned on a trimmed space and A/B'd every group end to end at the consumers
before adopting. Net effect, CUDA/float:

| shape | before | after | gain |
|---|---|---|---|
| syev n=64 b=4096 | 1.0798 | 0.9938 | 1.087× |
| syev n=512 b=512 | 166.95 | 151.01 | 1.106× |
| gesvd n=128 b=1024 | 9.2465 | 8.0684 | 1.146× |
| gesvd n=256 b=512 | 94.987 | 92.016 | 1.032× |
| gesvd n=512 b=256 (vectors) | 1143.3 | 1110.3 | 1.030× |

Everything else was within 1%. What changed, and why:

- **ORMQR 16 everywhere → 16/16/24/48/56.** The pre-split aliasing with gebrd had made this
  impossible. gebrd now has its own `GEBRD_BLOCK_SIZE_*`, with a flat curve against ormqr's
  steep one (2.16× between 16 and 56). MEDIUM was checked separately at both consumers and is
  neutral there (within 0.7%).
- **SYTRD 8/8/16/24/32 → 8/8/16/8/48.** The LARGE bucket (256 < n ≤ 512) went to 8. The
  2026-08-03 syev-context sweep above had 24 as the float optimum at n = 512 and 8 as the worst,
  by 1.12×. Whether the trimmed retune measured n = 512 in this bucket is not recorded. This is
  an **open question**; see [open debts](#syev-open-debts-and-unmeasured-items). Complex
  overrides this bucket at the consumer (next section).
- **`SY2SB_ORMQR_NB_LARGE = 32`.** The shape gate in `sytrd_sy2sb.cc` fires only at n ≥ 1024 and
  batch ≥ 32, so n = 512 got no WY hint at all. Worth 1.103× at batch 512 and 1.117× at batch
  1024. With the gate active this knob shadows `ORMQR_BLOCK_SIZE_*` on syev's hot path: changing
  the ormqr constant leaves the kernel trace identical.
- **stedc threads-per-root and wg multiplier are set from syev, overruling stedc's own bench.**
  The stedc sweep picked wg_multiplier 2–4 and threads_per_root 4–8, which cost syev 2.7% at
  n = 256. Measured through syev with everything else fixed:

| syev n | wgm=2 | wgm=4 | wgm=8 | tpr=4 | tpr=8 | tpr=16 |
|---|---|---|---|---|---|---|
| 64 | 0.9987 | 0.9225 | 0.9227 | 0.9937 | 0.9229 | 0.9422 |
| 128 | 4.626 | 4.535 | 4.514 | 4.518 | 4.469 | 4.534 |
| 256 | 23.067 | 22.110 | 21.383 | 21.557 | 21.332 | 21.330 |
| 512 | 151.54 | 151.46 | 151.84 | 151.63 | 151.62 | 151.67 |
| 1024 | 903.40 | 900.12 | 903.01 | 902.35 | 904.74 | 903.35 |

  8/8 wins nearly everywhere (up to 7.6% at n = 64, within noise at n ≥ 512). The isolated
  benchmark and its consumer do not optimise the same thing.
- **`SB2ST_BACK_*` deliberately stay 0.** The geometry spans 3.6× across its grid at n = 1024,
  but the heuristic already picks the winner: n=256 b=1024 0.012428 against tuned (8,4)
  0.012415; n=512 b=512 0.050644 against (8,8) 0.050706; n=1024 b=256 0.20701 against (8,16)
  0.20575. That is within 0.6%, so freezing buckets buys nothing and loses adaptivity. The sb2st
  bench stays in the default space as a tripwire.

Tuning is float-only, and the header is regenerated from a float-only bench. Per-type
corrections therefore live at the consumer, where the next retune cannot overwrite them.

### syev: complex panel width in the 256 to 512 bucket

`sytrd_block_size_default<T>` (`src/extensions/syev_blocked.cc:88`) returns 32 for complex at
256 < n ≤ 512. The harness value of 8 there costs complex 1.16×–1.20×. The other complex buckets
were measured and are not changed. n ≤ 128 prefers 8 (nb 8 against 16: 2.36 vs 2.37 at n = 64,
5.92 vs 6.07 at n = 96, noise). 128 < n ≤ 256 prefers 16 (n = 192: 33.83 at 16 against 36.09 at
8). n > 512 is within 1.03× of the committed 48 at nb = 32, below the neutral band.

### syev: the two-stage band width kd

The prediction was that removing the split-WY penalty (`f7f3c57`, which passes `nb = kd` so
`ormqr` stops chopping a kd = 32 panel into two k = 16 WY blocks, worth 1.19–1.36× at n ≥ 1024,
batch ≥ 32) would move the eigenvector optimum up to 96–128. MAGMA and Gates/Tomov/Dongarra put
it there. **Disproved.** Total ms, eigenvectors, median of 3, 2026-08-04:

| n / batch | kd=16 | kd=32 | kd=48 | kd=64 | kd=96 | kd=128 | blocked |
|---|---|---|---|---|---|---|---|
| 128 / 2048 | 27.6 | 22.0 | 22.0 | 21.0 | 18.5 | **16.9** | **14.6** |
| 256 / 1024 | 69.5 | **61.3** | 64.2 | 65.7 | 93.5 | 75.7 | **40.9** |
| 512 / 512 | 217.7 | **194.2** | 216.8 | 236.1 | 294.9 | - | **191.4** |
| 1024 / 128 | 445.3 | **369.9** | 394.9 | 433.2 | 541.3 | 715.3 | 475.5 |
| 2048 / 32 | 1171 | **1066** | 1160 | 1286 | 1609 | 1943 | **876** |

kd = 32 remains optimal at n ≥ 256. Wide kd wins only at n = 128 (kd = 128 beats 32 by 1.30×),
where two-stage loses to blocked anyway.

The committed table this re-measured (same benchmark, float, eigenvectors, total ms, undated,
before grid-`latrd` and before the kd = 128 column was added):

| n / batch | kd=16 | kd=32 | kd=48 | kd=64 | kd=96 | blocked |
|---|---|---|---|---|---|---|
| 128 / 2048 | 27.8 | 23.4 | 22.8 | 21.9 | 19.6 | 15.0 |
| 256 / 1024 | 78.9 | 65.3 | 66.9 | 66.7 | 72.2 | 42.4 |
| 512 / 512 | 249.5 | 203.9 | 223.1 | 240.0 | 298.6 | 193.3 |
| 1024 / 128 | 500.1 | 425.8 | 443.9 | 470.0 | 546.6 | 481.4 |
| 2048 / 32 | 1275.3 | 1183.4 | 1259.0 | 1353.8 | 1614.0 | 1265.7 |

It already had kd = 32 optimal at every n ≥ 256, and two-stage winning at n = 1024 (1.13×) and
n = 2048 (1.06×). That table itself superseded an earlier 32/64 split measured before the wave
back-transform landed: \f$Q_2\f$ then dominated and got cheaper with kd, which pulled the optimum
up to 64 at large n. With \f$Q_2\f$ about 3× cheaper, stage 1 and the chase
(\f$O(n^2 k_d)\f$ work) set the balance, and they favour a narrow band. The nb-hint A/B at n = 1024, the only shape where `f7f3c57`'s gate fires,
explains why:

| kd | 16 | 32 | 48 | 64 | 96 | 128 |
|---|---|---|---|---|---|---|
| nb hint on (ms) | 445.3 | 369.9 | 394.9 | 433.2 | 541.3 | 715.3 |
| nb hint off (ms) | 453.6 | 391.1 | 420.0 | 448.1 | 531.0 | 662.4 |
| hint gives | 1.019× | 1.057× | 1.064× | 1.034× | 0.981× | **0.926×** |

The hint helps narrow bands and hurts wide ones, as `sytrd_sy2sb.cc` predicts: LARFT work is
\f$O(m k\,n_b)\f$ and doubles with \f$n_b\f$. Removing the split-WY penalty did not free wide kd.
It made wide kd relatively worse.

**"Two-stage wins at n ≥ 1024" is shape-dependent.** Grid-`latrd` sped up the blocked baseline
underneath that comparison:

| n=2048 / batch=32 | blocked | two-stage (kd=32) | winner |
|---|---|---|---|
| latrd legacy (as originally measured) | 1235.1 | 1066 | two-stage 1.16× |
| latrd grid (the default; 1.41× over legacy) | **875.8** | 1066 | **blocked 1.22×** |

The legacy figure reproduces the committed table's 1265.7, so the comparison is like for like.
At n = 1024 / batch = 128 the `latrd` implementation makes no difference (475.5 vs 475.7, since
batch 128 already saturates 128 SMs), and two-stage wins by 1.29×, better than the recorded
1.13×. Two-stage wins where the batch saturates the device and loses where grid-`latrd` rescues
blocked.

### syev: latrd grid gate confirmed in eigenvector mode

`latrd_grid_min_n = 768` was derived in eigenvalues-only mode and is applied in both modes.
legacy/grid ratio (> 1 means grid wins), eigenvectors:

| n | batch 1 | batch 8 | batch 64 |
|---|---|---|---|
| 256 | 0.734 | 0.734 | 0.788 |
| 384 | 0.885 | 0.894 | 0.881 |
| 512 | 1.012 | 1.018 | 0.956 |
| 768 | **1.379** | **1.375** | 1.090 |
| 1024 | **1.843** | **1.781** | - |

The crossover is the same, so the constant does not need splitting per mode. The win shrinks as
batch grows, because once the batch alone saturates the SMs there is no starvation left to absorb.

## syev: the 2026-08-08 implementation plan, measured

The plan turned a profile of float and complex<float> `syev` at saturating batch into work
packages, each with an end-to-end accept gate. A kernel-level win is not a solver-level win:
a 2.16× kernel win had already become an 11% `gesvd` loss in this repo. Rig: `build-cuda`
(`sm_89`, RelWithDebInfo), device 1 of two, one process, idle GPU, clocks warmed, first run of
a fresh process discarded, `--warmup=2 --min_iters=5`. The benchmark positional arguments are
`n batch nb fuse jobz uplo`.

**Rig validation.** At n = 512, batch 512, eigenvectors, two-stage, the rig gave float 334.28
against the research profile's 332.5 (0.5%), and cfloat 971.17 against 972.4 (0.1%). Sweep
standard deviations ran 0.02–0.5%, so differences under about 1% are noise. The research
profile's forced tile=2/subs=4 cfloat figure (855.75) reproduced as 852.77.

The plan's own estimates, for comparison with the table: WP1 up to 1.35×, WP3 1.05×–1.12×,
WP4 1.14× (already measured by the research profile), WP5 1.22×–1.82×, WP6 1.49×, and WP8
(block Jacobi) 1.5×–2.5× at n = 128–256, marked speculative. WP2 had no estimate: it was to
decide its own fate. After the build, `syev_blocked`, `sytrd_blocked`, `sytrd_sb2st` and
`sytrd_sb2st_hh` tests all passed (145 s). The plan required accuracy runs
(`eigensolver_accuracy`, `syev_blocked_acc`) for WP1, WP3 and WP8, which change the algorithm
or the arithmetic, and expected WP4, WP5 and WP6 to be bit-neutral or near it: accuracy moving
there means something is wrong. The results notes do not record those accuracy runs.

| WP | item | gate | measured | verdict |
|---|---|---|---|---|
| WP0 | harness unblockers | cfloat rows appear | rows appear | passed |
| WP1 | A2: `stebz` values-only | ≥ 1.20× float n=256 | **1.44×** | passed, beat its estimate |
| WP4 | A3: per-type tile/subs | ≥ 1.10× cfloat, float unchanged | **1.139×**, float unchanged | passed |
| WP3 | A1: `her2k` trailing update | ≥ 1.05× cfloat | 1.043× | marginal miss, kept |
| WP2 | B3: `cta-large-n` | ≥ 1.15× vs blocked | **0.012× (85× slower)** | rejected |

### syev: stebz values-only in the blocked solver (WP1)

In values mode the blocked solver ran a full eigenvector divide-and-conquer and discarded Z.
At n = 256, batch 1024, that was 28.3% of the float solve. `stedc` cannot simply be asked for
`NoEigenVectors`, because its recursion and merge rely on the eigenvectors. The call therefore
goes away, and `stebz` (bisection, index range 0..n-1, ascending) runs straight off the real
tridiagonal, as the two-stage path already did. There is no `sterf` in the tree. Both `stedc`
(ascending argsort) and `stebz` with `order=Ascending` return ascending order. Commit `f28a3fcd`.
Values-only, `provider=blocked`, batch 1024:

| n | float before | float after | gain | cfloat before | cfloat after | gain |
|---|---|---|---|---|---|---|
| 64 | 1.1614 | 0.62228 | **1.87×** | 1.5055 | 1.0465 | **1.44×** |
| 128 | 4.5231 | 2.9285 | **1.54×** | 7.3078 | 5.6837 | **1.29×** |
| 192 | 11.566 | 7.6783 | **1.51×** | 21.486 | 16.834 | **1.28×** |
| 256 | 20.774 | **14.476** | **1.44×** | 47.218 | **38.250** | **1.23×** |
| 320 | 42.804 | 31.438 | **1.36×** | 101.45 | 87.666 | **1.16×** |

The plan estimated "up to 1.35×", the arithmetic bound was 1.39×, and the gate was ≥ 1.20×. The
win is where theory says it should be. `stedc` runs in real arithmetic, so its absolute cost is
identical for both scalar types. It is 28.3% of the float solve but only about 12% of the cfloat
solve at n = 256 (scaled by 20.774/47.218). Float must therefore gain more than cfloat, and it
does, with the gap narrowing monotonically as the reduction comes to dominate. A result where
cfloat gained as much as float would have meant the saving came from somewhere unintended.
Values mode also stops allocating the eigenvector-only workspace: at n = 256, batch 1024, cfloat,
that is `z_span` 268 MB plus `zc_span` 537 MB, plus the stedc workspace.

### syev: per-type sb2st back-transform geometry (WP4)

`provider=two_stage`, eigenvectors, n = 512, batch 512:

| | shipped constants | forced tile=2 subs=4 (both types) | per-type code (after) |
|---|---|---|---|
| cfloat | 971.17 | 852.77 (1.139× faster) | **852.46 (1.139×)** |
| float | 334.28 | **384.30 (1.15× slower)** | 331.25 (unchanged) |

The global flip would have traded a 1.14× complex win for a 1.15× float loss on the provider float
is routed to at this `n`. The per-type selection (`8c24626f`) captures the whole complex win and
keeps float at its own optimum. The complex optimum is the smaller tile with fewer sub-groups that
the occupancy collapse predicts: back-transform SM throughput 35.3% against float's 93.7%, and
occupancy 49.8% against 82.8%. Only n = 512 was measured, so the rule is `n == 512` exactly. The
instantiated set is tile ∈ {1,2,4,8} × subs ∈ {4,8,16}, and any other pair silently falls through
to the slower tiled kernel. A3 and B2 are **not additive**. A3 is the cheap fraction of the same
occupancy problem, so any B2 estimate must be measured against 852, not 971.

### syev: her2k trailing update for complex float (WP3)

`sytrd_blocked`'s trailing update issued two full \f$n_2 \times n_2\f$ GEMMs for complex, where
float issues one triangle-only `syr2k`. `her2k` goes to a different backend route from `syr2k`:
one batched GEMM into scratch, then a Hermitian fold. That is half the arithmetic of the pair,
because \f$\alpha A B^H\f$ and \f$\bar\alpha B A^H\f$ are conjugate transposes of each other.
Commit `405b4120`.

The primitive was gated first. The panel's update is narrow (\f$k = i_b \in \{16,24,32\}\f$ against
\f$n_2\f$ up to 480), and the fold adds an \f$n_2^2\cdot\text{batch}\f$ write plus read. Measured at
\f$n_2\f$ = 480, batch 512, cfloat:

| k = ib | her2k | one GEMM | the pair | her2k vs pair |
|---|---|---|---|---|
| 16 | 3.2401 ms | 2.1314 ms | 4.2628 ms | 1.32× |
| 24 | 3.2703 ms | 2.1603 ms | 4.3206 ms | 1.32× |
| 32 | 3.3010 ms | 2.1933 ms | 4.3866 ms | 1.33× |

her2k wins, but its time is almost independent of k. It is bound by the \f$n_2^2\f$ product-buffer
traffic, not by arithmetic, so a larger `nb` will not grow the win. Propagating 1.33× through the
phase share predicted about 1.04× end to end. The trailing update is roughly half of the 34.6% of
the cfloat solve at n = 256 that vendor GEMM occupies (14.4% at n = 512).

End to end, `provider=blocked`, eigenvectors, n = 256, batch 1024: cfloat 66.628 → **63.899
(1.043×)**. Float 31.657 → 31.037, unchanged, since it takes the same code path. The gate was
1.05×, a miss by 0.7 of a point, and the miss was predicted by measurement in advance. It was kept
anyway: a real 1.04× at no runtime cost, float untouched, and the single-definition route
prediction it forced in `expansion_budget.hh` is worth having on its own. The gate should read
1.04×, not "met".

**The fallback, quantified.** Forcing the host-loop route (`BATCHLAS_EXPAND_ROUTE=loop`) at
\f$n_2\f$ = 480, k = 32, batch 512 gave 5.3924 ms, 1.23× *slower* than the GEMM pair. The
call-site fit guard is necessary, but the downside is 1.23×, not the 7.8× that the real-`syr2k`
comment in `sytrd_blocked.cc` warns about. That figure was measured in double, where the vendor
loop is far worse. complex<double> is deliberately not admitted: its scratch is 16 bytes per
element, which halves the fit headroom, and it is unmeasured.

Why the guard is needed at all: when the expansion does not fit, `her2k_vendor` silently drops
to a per-item host loop. The budget is `GLOBAL_MEM_SIZE / 4` (`triangular_expand.hh`), about
6 GB on this card, and the scratch is \f$n_2^2\cdot\text{batch}\cdot\text{sizeof}(T)\f$. At the
plan's shapes it fits (cfloat, largest \f$n_2\f$): n = 256 / batch 1024 → \f$n_2\f$ = 224,
411 MB; n = 512 / batch 512 → 480, 944 MB; n = 2048 / batch 64 → 2016, 2.08 GB. The last one
fits but interacts with the unsaturated n = 2048 row. `BATCHLAS_EXPAND_MAX_BYTES=1` is the cheap
way to prove the guard works.

### syev: cta-large-n rejected (WP2)

The unmerged `cta-large-n` branch lifted the CTA eigensolver above n = 32. Its local-memory limits
were n = 128 for float and 64 for cfloat, and it had passing tests (36/36 + 6/6 + 5/5) but no
performance measurement at all. The branch (`04101dc`) was based on `27851a6`, 130 commits behind
main, but only `syev.hh` had moved under it; the branch's own change there was 11 lines in
`syev_supports_cta`, replacing the hard n > 32 rejection with a local-memory partition limit.
Blocked runs at 2.2–11 µs/matrix at these n, so this was a launch-overhead bet. The research profile ranked it **first**, as "the highest value-per-hour item in Tier B,
because the code already exists". Forced `provider=cta` against `blocked`, float, eigenvectors,
batch 256:

| n | CTA | blocked | |
|---|---|---|---|
| 33 | 150.17 | 1.7715 | **85× slower** |
| 64 | 760.68 | 3.6021 | **211× slower** |

At batch 2048, one sweep over n = 33..128 did not complete a single measured iteration in ten
minutes while holding 24 GB. The implementation cost was sunk; the value was not there. At
n ≥ 33 a CTA-resident solve saves about 15 kernel launches but runs an unblocked level-2
algorithm in one work-group per matrix against a blocked level-3 pipeline. Blocked baselines it
had to beat, eigenvectors, batch 2048:

| n | float | cfloat |
|---|---|---|
| 33 | 0.65996 | 0.80367 |
| 48 | 0.97778 | 1.3450 |
| 64 | 1.5123 | 2.2372 |
| 96 | 3.4767 | 5.6691 |
| 128 | 6.5491 | 10.982 |

The port was reverted out of the PR (`14bad498`) and survives on `worktree-wf_3b4af334-426-9`.
Besides the performance result, it also rewrote the live n ≤ 32 CTA kernels (`sytrd_cta.cc`,
`ormqr_cta.cc`, `syev_cta.cc`, and the sub-group-to-work-group partition in `sg_compat.hh`), and
`Auto` routes there. That effect was never measured. If the partition refactor is wanted, it
should come back as its own change with n ≤ 32 numbers attached. A rebase trap also applies:
`syev_cta_max_n_for_vectors` rejects any value above 32, so after a clean rebase `Auto` still
cannot route above 32 and the branch appears to do nothing. Benchmark through the forced
provider. This is now listed among the measured dead ends in `AGENTS.md`.

### syev: harness unblockers (WP0)

`sb2st_hh_benchmark` registered only `float`, so `--type=cfloat` silently yielded zero rows.
After the fix (n = 512, batch 512, kd = 32):

| | float | cfloat | ratio |
|---|---|---|---|
| CHASE | 56.578 | 267.70 | 4.73× |
| BACK | 99.095 | 317.03 | 3.20× |

The CHASE ratio matches the 4.85× the research profile measured through the full solver. BACK
is 3.20× rather than 3.90× because WP4 is active in this build. The float BACK row matches the
roughly 101 µs/matrix derived when the plan's own accept criterion was rejected as a category
error: 972 µs/matrix is a whole two-stage solve, not a back-transform row.

The grid `latrd` path could not be forced. `BATCHLAS_LATRD_GRID_GROUPS` is clamped by the
residency cap `MAX_COMPUTE_UNITS / batch`, so at batch ≥ 128 no value reaches the grid kernel,
and the recorded L2-residency A/B was legacy against legacy. `BATCHLAS_LATRD_GRID_FORCE_UNSAFE`
bypasses the cap. It is deadlock-capable by construction: the software grid barrier hangs, and
does not fail, when the launch exceeds residency, and that looks exactly like slow JIT. Run it
with a timeout. It was added but **never exercised**.

## syev: the stedc merge-variant and leaf-cliff regression

Measured 2026-08-05. PR 55 (`0bb92fb`, the STEDC level-driver rework) cost `syev` 1.05×–3.25× at
every size measured. The vendor column was the control. It reproduced the saturated grid to three
digits at all eight sizes (2.02 / 7.85 / 37.1 / 215.1 / 504.1 / 1006.9 / 1414.6 / 2704.7), so the
deltas are code. Float, eigenvectors:

| n | batch | baseline | after PR 55 | ratio | fixed | fixed vs baseline |
|---|---|---|---|---|---|---|
| 64 | 16384 | 1.64 | 2.25 | 1.37× | **1.70** | 0.96× |
| 128 | 4069 | 6.65 | 8.76 | 1.32× | **6.58** | 1.01× |
| 256 | 2034 | 36.92 | 45.64 | 1.24× | **35.84** | 1.03× |
| 320 | 1302 | 74.53 | 242.43 | **3.25×** | **70.96** | 1.05× |
| 512 | 508 | 380.02 | 411.19 | 1.08× | **372.14** | 1.02× |
| 640 | 651 | 727.48 | 1095.9 | **1.51×** | **714.99** | 1.02× |
| 768 | 452 | 1184.11 | 1258.9 | 1.06× | **1162.3** | 1.02× |
| 1024 | 254 | 2441.93 | 2559.1 | 1.05× | **2394.7** | 1.02× |

**Defect A: the merge-variant flip was backwards.** `3072ea6` had moved `STEDC_MERGE_VARIANT_*`
from FusedCta (2) to Fused (1), claiming that FusedCta was numerically wrong and 2–12% slower.
Neither claim reproduced. With variant 2 forced, all 16 CUDA `stedc_tests` pass in both
precisions, plus `syev_tests` 8/8, `syev_blocked_tests` 44/44 and `syev_two_stage_tests` 20/20.
The NaN had been a symptom of the deadlock that `4bde59a` fixed. On speed, FusedCta is 11–35%
*faster* on `stedc`, on both drivers:

| n | Levels+Fused | Levels+FusedCta | Recursive+Fused | Recursive+FusedCta |
|---|---|---|---|---|
| 64 | 0.672 | **0.499** | 0.640 | **0.500** |
| 128 | 2.236 | **1.752** | 2.249 | **1.736** |
| 256 | 7.387 | **6.257** | 7.620 | **6.233** |
| 512 | 24.744 | **22.228** | 30.897 | **26.505** |

This alone was the entire regression at power-of-two `n`. **Caveat, still open:**
`FusedCtaConditionedHeavyDeflation` asserts only finite-and-sorted, never accuracy. The test was
rewritten in `47a58c8` and its batch halved in `c80fd86`. Strengthen it before trusting variant 2
on hardware it has not been measured on. The stedc page carries the current merge constants and
this history from the stedc side: [stedc: current tuning values](stedc.md#stedc-current-tuning-values).

**Defect B: the level planner chose a leaf past the STEQR cliff.** `plan_stedc_levels` searched
leaves in [threshold/2, threshold·2] and tie-broke toward the leaf nearest the threshold. At
n = 320 and 640 it chose leaf 40 over leaf 20. `steqr` takes the fast `steqr_cta` path only for
n ≤ the sub-group width and throws above it, falling back to `steqr_wg`. `steqr` alone, batch
10416, eigenvectors:

| n | 16 | 20 | 24 | 32 | 36 | 40 | 48 | 64 | 80 |
|---|---|---|---|---|---|---|---|---|---|
| steqr | 0.252 | 0.273 | 0.292 | 0.262 | **3.761** | 4.870 | 16.98 | 36.78 | 54.41 |

A 12.5% width increase costs 14×. That is a code-path cliff, not scaling. Sweeping the threshold
at n = 320 moves the chosen leaf:

| threshold | 16 | 20 | 24 | 32 | 40 | 48 | 64 |
|---|---|---|---|---|---|---|---|
| chosen leaf | 20 | 20 | 20 | **40** | 40 | 40 | 80 |
| stedc | 13.1 | 12.5 | 12.7 | **168.4** | 168.4 | 168.5 | 649.7 |

`STEDC_RECURSION_THRESHOLD = 32` is the sub-group width, an invariant rather than a tuning
constant. The fix hard-caps the leaf at the threshold (`hi = threshold`,
`stedc_levels_plan.hh`). After the cap, the level driver is the faster one at these sizes:
10.8 against 11.1 (Recursive) at n = 320, and 35.6 against 36.2 at n = 640.

It escaped review because `stedc_benchmark` registered only n ∈ {64, 128, 256}, and the rework
was measured at n = 64..512, all powers of two. There the planner lands exactly on leaf 32, so
the cliff could not be sampled. n = 320 and 640 are now in the registered sweep, and the plan
shape is asserted directly in the host-only `StedcLevelPlan.*` tests, because a bad leaf yields
perfectly correct eigenvalues.

## syev: workspace footprint limits usable batch

**Open defect, measured 2026-08-03.** `syev_cta`, n = 32, eigenvectors:

| batch | peak device memory | µs/matrix |
|---|---|---|
| 512 | 445 MiB | - |
| 1024 | 2237 MiB | 0.320 |
| 4096 | 7505 MiB | 0.192 |
| 16384 | **24083 MiB** (whole card) | 0.592, regresses |

That is about 1.8 MB of workspace per 32×32 matrix, roughly 450× the 4 KB of data, scaling
linearly above batch 512. Eigenvalues-only is unaffected (about 343 MiB at batch 16384). The
timing at 16384 goes backwards because of memory pressure, not compute. `syev_blocked` has the
same problem independently: eigenvalues-only at batch 16384 (n = 32), 15258 (n = 64), 3814
(n = 128) and 953 (n = 256) each exceeded 3 GB. The eigenvalues-only half of that predates WP1,
which dropped the eigenvector-only workspace in values mode, and has not been re-measured since.

The consequences: usable batch is capped at about 4096 for n = 32 with vectors on a 24 GB card,
well below where these kernels would otherwise saturate. It also caused a real OOM on a shared
machine, which killed unrelated processes. It is the binding constraint on batch for every
small-n eigenvector solve, and it has not been investigated. It also made n = 2048 unsaturable
([open debts](#syev-open-debts-and-unmeasured-items)).

The same session measured what extra batch buys once past the knee (n = 8, float, `syev_cta`):

| batch | peak device memory | µs/matrix |
|---|---|---|
| 4096 | 521 MiB | 0.020506 |
| 16384 | 509 MiB | 0.0099457 |
| 65536 | **487 MiB** | **0.0075321** |
| 262144 | - | 0.0072411 |
| 1048576 | **24083 MiB** | 0.0071809 |

Going from 65536 to 1048576 improved µs/matrix by 4.7% and cost 50× the memory. Saturation means
the knee, where a 4× batch step buys less than 5%, not the asymptote. The bake-off driver
therefore enforces hard caps: about 1.5 GB of estimated footprint, a 3 GB hard abort per
invocation, and a 120 s timeout. (The workflow plan had asked for ≤ 2 GB peak and ≤ about 2 s of
wall time per invocation; the driver's constants are what actually ran.) A cell that does not
reach its knee within the caps is reported as such, with its curve, and not chased further.
The 1048576 row is the run that caused the OOM: it had climbed to batch \f$2^{20}\f$, used 24 GB of the 24.5 GB card and let the OOM killer take
out unrelated processes, including the session driving the work.

## syev: open debts and unmeasured items

### syev: open debt, the single-read panel symv

**Plan WP5 (B1), the largest item.** It is still open and is quoted in `AGENTS.md` as "the open
2.7x". The `latrd` panel is 35–71% of every blocked solve. Counters on `LatrdLowerPanel`, one
panel, ib = 32, j₀ = 0, as a multiple of ideal traffic:

| | float n=256 b=1024 | float n=512 b=256 | cfloat n=512 b=256 |
|---|---|---|---|
| DRAM | 0.14× | 1.13× | 1.43× |
| L2 | 1.95× | **2.34×** | **2.50×** |
| L1TEX | **11.9×** | **12.2×** | **15.8×** |
| SM throughput | 52.7% | **10.8%** | **10.7%** |

These were re-measured independently on 2026-08-08 with `ncu` on `LatrdLowerPanelKernelLegacy<float, 256, 0>`,
n = 512, batch 256: DRAM 4,845 MB (research 4,850), L2 10,050 MB (10,069), L1TEX **52,499 MB,
identical to the byte**, SM throughput 10.84% (10.8%), achieved occupancy 33.19% (33.2%), and
kernel duration 11.444 ms (11.37). The launch is grid (256,1,1) with 256-thread blocks: one
work-group per matrix.

Mechanism: thread *r* reads \f$A(r,c)\f$ on its row walk, and thread *c* reads the same element as
\f$A(c,r)\f$ on its column walk, which gives 2.3× L2. In the column walk consecutive lanes are
`lda` apart, which gives 12–16× L1. The ceiling from the DRAM floor at n = 512 is **2.7×**.

The design constraint: the symv lives inside the per-reflector loop of a single work-group per
matrix, with `v_local[n]` and `wcol_local[n]` in local memory. The reflectors are sequentially
dependent, so a MAGMA-style multi-block symv is not a drop-in, and any tiling has to happen
inside the work-group. The proposed design stages square off-diagonal tiles (32×32 to start) with a
coalesced load. From that one staged copy it computes both \f$y_R \mathrel{+}= A_{RC} v_C\f$ and
\f$y_C \mathrel{+}= A_{RC}^H v_R\f$: one load, two updates. An earlier rejected attempt noted in the
kernel ("one sub-group per column ... the extra barrier destroys reuse") changed the access pattern
without changing the *number* of reads. Budget: a 32×32 cfloat tile is 8 KB, so n = 512 cfloat
needs 16 KB of local memory in total, which still allows ≥ 2 blocks/SM.

Predicted end to end from the phase shares:

| | at 2.7× | at 2.0× |
|---|---|---|
| cfloat n=512, vectors (panel 71.5%) | 1.82× | 1.56× |
| cfloat n=256, vectors (panel 41.7%) | 1.36× | 1.26× |
| float n=256, vectors (panel 35.5%) | 1.29× | 1.22× |
| float n=256, values (panel 52.7%) | 1.50× | 1.36× |

Gate: ≥ 1.6× on the panel kernel alone at n = 512. The mechanism must be confirmed in `ncu`, not
only in the time: L1TEX should fall from about 12× toward 2×, and L2 from 2.3× toward 1×.

### syev: open debt, complex stage-2 occupancy

**Plan WP6 (B2), open.** At n = 512, batch 512, the float back-transform is saturated (SM 93.7%,
occupancy 82.8%), while the complex one is not (35.3% / 49.8%), and the chase runs 45.0% / 65.5%
float against 29.8% / 45.1% complex. Halved occupancy is a register-pressure signature, so the
complex kernel's 3.90× cost is not the price of complex arithmetic. There are two sub-items:

- a type-aware retiling of `unmqr_hb2st_wave`, measured against the post-WP4 852, not 971;
- the chase runs at 4.85× float, above the roughly 4× arithmetic ratio. The research profile
  recorded `__mulsc3` still present in `libbatchlas_extensions_sytrd.so`. Confirm it is on the
  chase's hot path first (`nm -C ... | grep mulsc3`, then disassemble), then write out the hot
  loop only, not the file.

The prize: if stage 2 came down to the roughly 2.2× complex/float ratio the panel reached after
its own fix, the two stage-2 kernels would fall from 1353 ms to 704 ms. That is 1.49× on the
complex two-stage solve, about 650 µs/matrix against 973.8, ahead of blocked (698) and the vendor
(707) at n = 512. It would close the one region where complex eigenvectors still lose to
cuSOLVER. Any routing change waits for the measurement.

### syev: remaining unmeasured items

- **The n = 2048 eigenvector row** is unsaturated (batch 64) and memory-limited. Re-measure it at
  the largest batch that fits, and report the batch. WP1 removed \f$2n^2\cdot\text{batch}\f$ of
  values-mode scratch, which may change what fits.
- **The values-mode 320 | 512 boundary** was not re-derived after WP1 made blocked
  1.36×–1.87× faster. The plan expected it to move up.
- **One routing sweep per type and mode after WP5.** WP1, WP5 and WP6 each move a boundary in
  `syev.hh`. The plan's rule was to re-run the grid once, after WP5, per scalar type and per job
  mode. The whole class of defect fixed in PR 65 was a constant measured on float and applied to
  every type.
- **SYTRD LARGE = 8 against the syev-context optimum of 24** at n = 512 (see
  [the 2026-08-07 retune](#syev-the-2026-08-07-constant-retune)). Check it with
  `BATCHLAS_TUNE_SYTRD_BLOCK_SIZE` before trusting either.
- **The XLARGE bucket in double**: SYTRD = 16 beat 24 by 1.066×/1.048× and ORMQR = 32 beat 16
  by 1.053×. Both were measured before `924b3a59` changed that bucket to 48/56, and neither has
  been re-measured in double since.
- **Grid residency cap (A5).** The cap could be `SMs × achievable blocks per SM`, computed from
  the kernel's real footprint, which would allow G ≥ 2 up to batch 256–512. It is honestly
  unlikely to pay alone, since there is no starvation at batch ≥ 128. Its value is making the
  L2-residency question answerable. Build it only if the unsafe override shows a win.
- **Profile `stedc` internally (B4) and the grid-barrier port.** Every merge kernel is one
  work-group per matrix (`nd_range(batch*128, 128)`). stedc is 20.7% of the float eigenvector
  solve at n = 256 and has never been profiled internally. WP1 removed it from the values path,
  so the remaining scope is eigenvector mode. The `latrd` grid-barrier machinery (1.9–4.1× at
  n ≥ 1024: sense-reversing barrier, work-group count capped at `MAX_COMPUTE_UNITS`, fixed group
  order for determinism, G = 1 dispatching to legacy bit for bit) is the template. Profile at
  batch 1 first to establish that stedc is the next dominant term. The ideation note framed the
  one-work-group-per-matrix decomposition as the recurring syev defect (four of the six syev
  performance commits before `e2ff635` fixed it in some kernel) and put the stedc merges at about
  a third of the flops of a full eigensolve. Its unmeasured guess was that a stedc grid barrier,
  costing \f$O(\text{nodes})\f$ per merge level rather than \f$O(n)\f$ per panel column, would
  cross over well below `latrd`'s 768. The implementation plan added the caveat that at large
  batch this is not starvation, so the grid-barrier treatment is not obviously right there.
- **A routing-audit benchmark** (`BM_SYEV_RoutingAudit`). It would run `Auto`'s choice and its
  runner-up for every routed shape and report every losing cell, so a stale table becomes a red
  row rather than an archaeology project. It was proposed twice and not built. The winner table
  was also meant to be emitted as generated data rather than hand-typed.
- **A grid-resident whole solve at batch 1** (ideation #8). This would be one persistent kernel
  per matrix across all SMs, for n = 256–1024 at batch 1, where the vendor once led by up to
  15.3×. It is speculative and should wait until the cheap fixes have bounded what is left of
  that gap.
- **Values-mode n ≤ 32** has no vendor comparison. The bake-off predates the `jobz` argument.
- **rocSOLVER** keeps the historical order walk. Nothing was measured there.

## syev: rejected ideas

| idea | why rejected | evidence |
|---|---|---|
| restore ORMQR = 128 (the "regression") | 1.50× slower at n = 512; 2.1×/3.1× slower at n = 64/128 | [refutation](#syev-the-ormqr-regression-hypothesis-refuted) |
| lift the CTA eigensolver above n = 32 (B3) | 85–211× slower than blocked | [WP2](#syev-cta-large-n-rejected-wp2) |
| global sb2st tile=2/subs=4 flip | costs float 1.15× | [WP4](#syev-per-type-sb2st-back-transform-geometry-wp4) |
| wide two-stage kd (96–128) after split-WY fix | kd = 32 still optimal at n ≥ 256; the nb hint hurts wide kd | [kd sweep](#syev-the-two-stage-band-width-kd) |
| freeze `SB2ST_BACK_*` into buckets | heuristic already within 0.6% | [retune](#syev-the-2026-08-07-constant-retune) |
| adopt stedc's own tpr/wgm winners | cost syev 2.7% at n = 256 | [retune](#syev-the-2026-08-07-constant-retune) |
| native Upper reductions | could recover at most the mirror's 0.3–2.0% | [Upper](#syev-uploupper-by-mirroring) |
| the float small-n rule applied to complex | Jacobi is 4–6× off the pace at n ≥ 20 in complex | [complex small n](#syev-complex-small-n-kernels) |
| a batch floor on the values-only rule | sent n = 1024 at batch 254 to the vendor at 2.75× | [values routing](#syev-eigenvalues-only-routing) |
| split `latrd_grid_min_n` per mode | same crossover in both modes | [latrd gate](#syev-latrd-grid-gate-confirmed-in-eigenvector-mode) |
| block Jacobi as a speed path (C1/WP8) | later measured 2–11× slower; accuracy-only opt-in (`AGENTS.md` dead ends; the design was in the uncommitted `JACOBI_EIGENSOLVER_PLAN.md`). The plan's case, now refuted: syev then ran at 1.65–2.1 TFLOP/s (about 3.5% of the ~47 TFLOP/s SGEMM sustains), so a ~10× flop premium (8–10 sweeps × ~4n³) would break even at 17–21 TFLOP/s; `gesvdj_cta` beat the tridiagonalising CTA path 4.1× at n = 16 and 23× at n = 8 | - |
| C5 spectral divide and conquer | its cluster-parallelism argument does not apply at saturating batch | plan, not measured |
| C6 real embedding of the Hermitian problem | 2× the flops and 2× the memory | plan, not measured |

## syev: measurement traps from this campaign

- **Contention inflates the vendor arm too.** A first pass had n = 768 at 0.28 and 0.38
  blocked/vendor for batch 32 and 64, an apparent 3.6× blocked win that would have moved the
  carve-out. Two measuring processes had overlapped. The inflated arm was the **vendor**'s,
  6885 µs/matrix against 1110 when re-run alone. Clean, the row is 1.73 / 1.10 / 1.05, vendor
  throughout.
- **Use physical floors as a check on the measurement.** Upper is Lower plus a mirror pass, so
  Upper/Lower ≥ 1. An early run reported 0.647× at n = 128, which is impossible. It too was two
  benchmark processes sharing the device.
- **JIT.** Discard the first run of a fresh process. SYCL JIT has fabricated a 3.7× loss on this
  box, and contention has produced spurious 3.6× "wins".
- **Provider spellings.** `BATCHLAS_SYEV_PROVIDER` takes `two_stage` or `two-stage`. `TWOSTAGE`
  silently degrades to `Auto`.
- **`--name` is a substring filter.** It corrupted the original eigenvector grid, and two small-n
  benchmark binaries register two benchmarks each. Key on the CSV `name` column.
- **An incremental build is not trustworthy across a revert that changes a widely-included
  struct.** Reverting WP2 removed a `DeviceCaps` member. `cmake --build` succeeded and the tests
  passed, but `two_stage` then segfaulted at n ≥ 256 in `~DeviceCaps` (freeing a `std::string` at
  the wrong offset), because objects built against two layouts were mixed. A clean rebuild fixed
  it. An intermediate spot-check reading float values-only at 15.301 instead of 14.463 was the
  same broken build: a 6% anomaly is exactly the size that gets rationalised as thermal drift.
  (`dispatch/context.hh` and `DeviceCaps` have since been deleted, but the lesson stands.) Every
  WP figure was re-measured on the clean build: 14.476→14.463, 38.250→38.358, 63.899→63.889,
  852.46→852.30, float two-stage 331.25→334.50 (baseline 334.28).
- **An unsampled regime invalidates a table.** This campaign found four: the batch ladder that
  stopped at 1024, "two-stage wins at n ≥ 1024" after grid-`latrd`, the 320 boundary that was
  never measured, and the power-of-two-only stedc sweep. *A benchmark whose parameter grid is all
  one shape measures that shape, not the code.* Routing must be decided at saturation.
- **Isolated benchmarks and their consumers optimise different things.** The stedc bench
  overruled by syev, the ormqr/gebrd aliasing, and a 2.16× kernel win that became an 11% `gesvd`
  loss. Accept end to end.
- **`BATCHLAS_TUNE_*` feeds buffer-size queries too.** Flip it between runs, never between a
  query and its call.

## syev: raw data and provenance

| data | where |
|---|---|
| small-n bake-off (phase A), both precisions | `benchmarks/results/syev_retune_phaseA.jsonl` (Git LFS) |
| syev-context block-size sweep (phase B), float | `benchmarks/results/syev_retune_phaseB_float.jsonl` (Git LFS) |
| the driver that produced both, with its resource caps | `benchmarks/syev_bakeoff_driver.py` |
| retune build | `build/presets/cuda` at `7911847` (Part 1), `12963a8` (Parts 2-4), `0bb92fb` (Part 5) |
| implementation-plan commits | `80737301` (plan), `afc8ae35` (baselines), `c42e5ce9` (WP3 gate, WP5 counters), `8c24626f` (WP4), `405b4120` (WP3), `f28a3fcd` (WP1), `af8768b0` / `be327f5b` (results), merged in PR 68 |
| per-type routing | `73eb5510` (PR 65) |
| constant retune | `924b3a59` |

The phase-A file holds 973 successful measurements. Phase B is complete for both precisions except
n = 256, which is missing. Only the float phase-B file is committed. The double phase-B figures in
the tables above exist only in this page. This checkout holds the LFS pointers. Run `git lfs pull`
to fetch the data (see [the raw-data rules](README.md#new-raw-data-lives-in-benchmarksresults-in-git-lfs)).

Reproduce one shape (positional `n batch nb fuse jobz uplo`; `nb = 0` means the shipped default):

```
CUDA_VISIBLE_DEVICES=1 BATCHLAS_SYEV_PROVIDER=blocked \
  ./build/benchmarks/syev_benchmark --backend=CUDA --type=float,cfloat \
  --warmup=2 --min_iters=5 64,128,192,256,320 1024 0 0 0 0

BATCHLAS_SYEV_PROVIDER=blocked|two_stage|vendor|cta
BATCHLAS_SYEV_SMALL_KERNEL=cta|fused|jacobi
BATCHLAS_LATRD_IMPL=legacy|grid          BATCHLAS_LATRD_GRID_MIN_N=<n>
BATCHLAS_SB2ST_BACK_TILE_W=<1,2,4,8>     BATCHLAS_SB2ST_BACK_SUBS=<4,8,16>
BATCHLAS_EXPAND_MAX_BYTES=<bytes>        # force the her2k host-loop fallback
BATCHLAS_TUNE_ORMQR_BLOCK_SIZE=<n>       BATCHLAS_TUNE_SYTRD_BLOCK_SIZE=<n>
```

Kernel attribution that sees cuBLAS: `nsys profile -t cuda -s none` and then
`nsys stats --report cuda_gpu_kern_sum`. `BATCHLAS_KERNEL_TRACE` does not see cuBLAS. Counters on
the panel kernel:

```
ncu -k regex:LatrdLowerPanel -c 2 --metrics dram__bytes_read.sum,lts__t_bytes.sum,\
l1tex__t_bytes.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active --csv \
    build/benchmarks/latrd_lower_panel_benchmark --backend=CUDA --type=float 512 256 32 0 0
```
