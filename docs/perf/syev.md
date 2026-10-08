# SYEV {#perf_syev}

> **Status:** current · RTX 4090 (sm_89), CUDA 13.2 · checked 2026-10-06

Routing, tuning constants and measurements for the dense symmetric/Hermitian eigensolver `syev`:
which provider `Auto` picks per scalar type, `n` and `jobz`, and the evidence behind each rule.
The measurements date from 2026-08-03 to 2026-08-08. Machine: device 1 only, cuSOLVER 12.2.0.1,
driver 595.84, unless a table says otherwise. Times are µs per matrix unless a table says ms.

[TOC]

## syev: what ships today

The families live in `src/ops/syev/choice.hh`: `cta`, `cta_fused` and `jacobi` (the three n ≤ 32
sub-group solvers), `blocked`, `two_stage` and `vendor`. `select::choose` takes the first runnable
entry of the nearest row of `tuned/syev.<dtype>.<arch>.txt` (arch `sm_89` or `sm_120`), keyed on
`jobz`, `n` and `batch`. `can_run` is correctness only: the native families need a non-NETLIB GPU
backend and a square A with n ≥ 1, and the small three also need n ≤ 32 and a 32-wide sub-group.
Uplo is not a key, because Upper is [mirrored into Lower](#syev-the-upper-to-lower-mirror-for-lower-only-providers).

> **Note:** every syev table (all dtypes, sm_89 and sm_120) is `source=transcribed:424a45bc`: it replays the
> previous router's order per grid cell (`tuned/README.md`, [flat kernel selection](../design/flat-kernel-selection.md)),
> identical across batch. A retune with `tools/tune/` is what would make batch matter.

| decision | where | rule |
|---|---|---|
| eigenvectors, n > 32 | `jobz=V` rows | float: blocked to 448, two-stage 449..1024, vendor above. double: blocked to 448, then vendor. complex<float>: blocked to 512, then vendor. complex<double>: blocked to 256, then vendor |
| eigenvalues only, n > 32 | `jobz=N` rows | blocked to 320, two-stage above, every type |
| n ≤ 32 family | rows with n ≤ 32, both modes | double: `jacobi`. float: `jacobi` to 8, `cta_fused` above. complex<float>: `cta_fused` to 8, `cta` above. complex<double>: `cta` |
| n ≤ 32 vendor handover | `jobz=V`, complex<double>, n = 25..32 | `vendor` first, `cta` second. Every other type stays native to 32 |
| backend scope | `can_run` | any GPU backend with a sub-group of 32. ROCm has no table and borrows the sm tables with a warning (untested) |
| values-only blocked solve | `syev_blocked.stebz_evals` | `stebz` bisection off the tridiagonal, no `stedc` |
| complex panel width | `sytrd_block_size_default<T>`, `src/extensions/syev_blocked.cc` | 32 for complex at 256 < n ≤ 512, harness value otherwise |
| complex trailing update | `rank2k_trailing_update_supported`, `src/extensions/sytrd_blocked.cc` | float and complex<float> use the rank-2k path on the CUDA backend. double, complex<double> and other backends keep the GEMM pair |
| sb2st back-transform | `sb2st_back_tile_for` / `sb2st_back_subs_for`, `src/extensions/sytrd_sb2st_hh.cc` | complex<float> at n = 512: tile 2, subs 4. Otherwise the shape-adaptive heuristic |
| block sizes | `ORMQR_BLOCK_SIZE_*`, `SYTRD_BLOCK_SIZE_*`, `include/batchlas/tuning_params.hh` | ORMQR 16/16/24/48/56, SYTRD 8/8/16/8/48 (buckets n ≤ 64/128/256/512/else) |
| sy2sb WY width | `SY2SB_ORMQR_NB_LARGE = 32` | n ≤ 512 bucket only |
| stedc | `STEDC_*`, `tuning_params.hh` | leaf threshold 32, merge variant 2 (FusedCta), threads-per-root 8, wg multiplier 8 |
| stedc leaf cap | `plan_stedc_levels`, `src/extensions/stedc_levels_plan.hh` | leaf ≤ threshold, a hard cap |
| two-stage band width | `choose_two_stage_kd`, `src/extensions/two_stage_common.hh` | kd = 32 |

**Pinning a family.** `BATCHLAS_SYEV_ROUTE=<family>` (case-folded; also `native`, `vendor`, `auto`)
or a `select::ScopedPin`. A pin that `can_run` refuses, or an unknown word, throws
`invalid_argument`. The retired `BATCHLAS_SYEV_PROVIDER`, `BATCHLAS_SYEV_SMALL_KERNEL` and
`BATCHLAS_SYEV_CTA_MAX_N` are no longer read.


## syev: eigenvector routing at saturation

The rule is keyed on `n` alone, never on batch: a tiny-batch caller pays launch overhead whatever is picked,
and tuning for that regime costs saturated throughput. Auto dispatched to the winner at every shape of the
grids below (Auto/expected 1.000 ± 0.002, idle GPU).

### syev: the 2026-08-04 saturated float grid

Float, eigenvectors, median of 3, build `12963a8`; batch is the largest at which all three providers fit. µs per
matrix.

| n | batch | blocked | vendor | two_stage | winner |
|---|---|---|---|---|---|
| 64 | 16384 | **1.64** | 2.02 | 2.21 | blocked 1.23× |
| 320 | 1302 | **74.53** | 215.23 | 111.20 | blocked 1.49× |
| 512 | 508 | 384.27 | 504.69 | **380.02** | two_stage 1.01× |
| 1024 | 254 | 4089.02 | 2706.84 | **2441.93** | two_stage 1.11× |
| 2048 | 64 | 33842.6 | **15019.1** | 24782.3 | vendor 1.65× |

Also n = 128, 256 (blocked, 1.18× and 1.01×) and 640, 768 (two_stage, 1.15× and 1.19×). The batch-keyed rule this
replaced was wrong at five of nine sizes (1.11×–1.23×). For n ≤ 32 at batch 2048, `cta`/Jacobi (n = 16) and
`cta`/fused (n = 32) beat every non-CTA provider by 2.8×–3.0×. Caveats: n = 2048 was measured at batch 64 (below
the 128 SMs, memory-limited), so that boundary rests on the weakest row; the grid is float only.

### syev: the blocked-over-cuSOLVER headline measurement

Quoted by the README. 2026-08-07, float, eigenvectors, median of 5, default `nb`, fills the 320..512 gap. µs per
matrix.

| n | batch | blocked | two_stage | vendor | winner |
|---|---|---|---|---|---|
| 320 | 819 | **67.84** | 113.39 | 203.00 | blocked (3.0× over vendor) |
| 448 | 585 | **195.27** | 262.79 | 400.59 | blocked (2.1× over vendor) |
| 512 | 512 | 326.71 | 332.55 | 504.40 | tie |
| 768 | 192 | 1241.90 | **1172.90** | 1348.30 | two_stage |
| 1024 | 128 | 3296.80 | 2614.70 | 2552.10 | tie (vendor/two-stage) |

(n = 384: blocked 120.99, two_stage 179.18, vendor 309.10.) The float crossover is 448, not 320 (the former rule
lost 1.35×–1.48× at n = 384 and 448). Two-stage wins by at most 1.06× for float in eigenvector mode, so the
449..1024 band is neutral and was kept.

### syev: per-type eigenvector crossovers

Same method, 2026-08-07; blocked at its own best `nb` (see
[the complex panel width](#syev-complex-panel-width-in-the-256-to-512-bucket)). Two-stage is never the complex
winner in eigenvector mode. µs per matrix (blocked / vendor):

- **complex<float>:** blocked wins n = 64..448 (1.08×–2.59×; 128 a tie), e.g. n = 192: 30.60 / 79.41. n = 512:
  686.19 / 706.96 (1.03×, the crossover). n = 640: 1357.40 / 1267.80 (vendor 1.07×); n = 1024: 5236.90 / 4047.90
  (vendor 1.29×).
- **double:** blocked at n = 384 (771.15 / 874.97), tie at 448 (1193.00 / 1199.80), vendor from 512 (1662.80 /
  1617.90).
- **complex<double>:** blocked at n = 128..256 (n = 256: 649.59 / 723.36), vendor from 288 (1131.00 / 974.79).

Full grids: `benchmarks/results/syev_retune_phaseA.jsonl`.

> **Warning:** the complex<double> boundary (256) is the most hardware-specific number on this page. The card
> runs FP64 at 1/64 rate, which penalises the native panel far more than cuSOLVER. Re-measure on a data-center
> GPU before trusting it. After the Annex G fix, complex<double> at n = 288..512 is a dead heat inside the
> 1.10× neutral band, so the boundary stays at 256.

**Annex G fix (`e4186908`).** Clang emitted the C99 Annex G complex multiply inside the panel symv (an `isnan`
branch plus `__mulsc3`). Writing that multiply-add in real arithmetic made the panel kernel 1.22×–1.29× faster
(real types unchanged: `mac` routes them to a plain multiply-add). At n = 512, batch 512 the complex/float cost
ratio is 2.44–2.93 for blocked and two-stage against 1.38–1.41 for cuSOLVER; the rest of the gap is
`sytrd_blocked.panel_only` (68% of the complex solve), see
[the open panel symv](#syev-open-debt-the-single-read-panel-symv). Old Auto to new Auto (`73eb5510`), n = 320..1024:
complex<float> 1.18×–1.69×, complex<double> 1.22×–2.88×, double 1.00×–1.97×, float 1.00×–1.50×; no cell regresses.

### syev: eigenvalues-only routing

Float, 2026-08-04, same method (`syev_benchmark` gained a `jobz` argument for this; before that the mode had no
vendor arm). µs per matrix.

| n | batch | blocked | vendor | two_stage | winner |
|---|---|---|---|---|---|
| 64 | 16384 | **1.06** | 1.12 | 1.14 | blocked 1.06× |
| 320 | 2604 | **48.16** | 200.94 | 50.35 | blocked 1.05× |
| 512 | 1017 | 298.30 | 458.95 | **158.72** | two_stage 1.88× |
| 1024 | 254 | 3537.41 | 2494.44 | **908.32** | two_stage 2.75× |
| 2048 | 64 | 29547.70 | 13908.80 | **10804.10** | two_stage 1.29× |

Blocked wins n = 64..320, two-stage 512..2048 (n = 128, 256, 640, 768 not shown). With no eigenvectors to apply,
two-stage never pays for \f$Q_2\f$. It is also the complex winner above n = 320 (1.30× over cuSOLVER at
n ≥ 768), so there is no per-type split here.

These figures predate the `stebz` change ([WP1](#syev-stebz-values-only-in-the-blocked-solver-wp1)), which made
blocked 1.36×–1.87× faster up to n = 320; the 320 | 512 boundary has not been re-derived
(see [open debts](#syev-open-debts-and-unmeasured-items)). The rule replaced `syev_prefer_two_stage_values`,
whose `batch >= 256` term sent n = 1024 at batch 254 to the vendor (2.75× loss): a batch floor reopens that hole.
At batch 1 the vendor wins by 12.5× at n = 1024 (batch starvation of both reductions).

### syev: the Upper-to-Lower mirror for Lower-only providers

The Lower-only blocked and two-stage pipelines get Upper input by mirroring the upper triangle into the lower one
(\f$O(n^2)\f$ before an \f$O(n^3)\f$ solve). Float, 2026-08-04, median of 3, n = 128..1024, both modes: Auto (Upper)
beats vendor (Upper) at every shape (n = 320 vectors 2.95×, n = 1024 values 3.39×, n = 256 vectors 1.02×) and
Upper/Lower is 1.003×–1.020×. A native Upper reduction could recover at most that 0.3%–2.0%, so it was rejected.

`mirror_upper_to_lower` (`src/extensions/uplo_mirror.{hh,cc}`) writes \f$A_{ji} := \overline{A_{ij}}\f$ in place
(`syev` overwrites A); the diagonal is left alone and a complex diagonal's imaginary part is not forced to zero, as
the Lower path already assumes. The header is declaration-only with explicit instantiations in `uplo_mirror.cc`
(a SYCL kernel name needs one definition: inline in both `syev_blocked.cc` and `syev_two_stage.cc` gave "definition
with same mangled name"). The obvious test is vacuous because `Matrix::Random(..., symmetric=true)` makes Upper and
Lower interchangeable: the fixture poisons the strictly-lower triangle after taking the reference from the upper
one and asserts a Lower-read solve differs by more than 1.0 ("fixture is vacuous" otherwise). Two cases in
`tests/syev_blocked_tests.cc` cover all four types, one per call site.

## syev: small-n kernel choice

The CTA-family kernels `syev_cta`, `syev_cta_fused`, `syev_jacobi_cta` are the families `cta`, `cta_fused`,
`jacobi` (`BATCHLAS_SYEV_ROUTE=cta|cta_fused|jacobi`). The measurements below used the retired
`BATCHLAS_SYEV_SMALL_KERNEL` knob.

### syev: the 2026-08-03 small-n bake-off

Build `7911847`. Median of 5 process-level repeats with IQR, each cell at its knee batch, best over
`cta_wg_size_multiplier`. A margin below 1.10× or with overlapping IQRs is *neutral*. Key results on the CSV
`name` column (each binary registers two benchmarks). µs per matrix.

| type | n | mode | winner | µs | runner-up | ratio | vs cuSOLVER |
|---|---|---|---|---|---|---|---|
| double | 4 | vectors | jacobi | 0.01072 | cta_fused | 3.05× | 15.17× |
| double | 32 | vectors | jacobi | 3.31753 | vendor | 1.37× | 1.37× |
| float | 8 | vectors | jacobi | 0.00478 | cta_fused | 1.37× | 8.42× |
| float | 16 | vectors | cta_fused | 0.02933 | jacobi | 1.22× | 4.10× |
| float | 16 | values | cta_fused | 0.02740 | jacobi | 1.10× (*neutral*) | - |
| float | 32 | vectors | cta_fused | 0.28426 | tridiag_ref | 1.11× | 1.79× |

Double (n = 4, 8, 16, 32, both modes): Jacobi wins every cell, 1.37×–3.59×. Full grid:
`benchmarks/results/syev_retune_phaseA.jsonl`. `syev_cta`, where Auto sent every n ≤ 32, won no cell. The rule is
the n ≤ 32 rows of `tuned/syev.{float,double}.*.txt`: double Jacobi everywhere; float Jacobi to n = 8, `cta_fused`
above. The float rule rests on the vector cells; n = 9..15 was not swept. A later A/B found fused with its tuned
multiplier beats Jacobi from n = 7; the 8 | 9 boundary stays for Jacobi's relative accuracy on graded input (4.5e-07
against 2.7e+28 for `syev_cta`, quoted from the uncommitted Jacobi plan, see
[the Jacobi design](#syev-the-jacobi-kernel-design-and-its-accuracy-argument)).

> **Warning:** FP64 at 1/64 rate inflates Jacobi's margin over the tridiagonalising paths. The float column
> predicts a 1:2 FP64 datacenter GPU better; gate the double rule on measured FP64 throughput there. See
> [the small-n steqr routing](steqr.md#small-n-syev-routing-measured-not-shipped).

### syev: complex small-n kernels

complex<float>, eigenvectors, 2026-08-07, median of 3 at each cell's saturating batch (µs per matrix):

| n | batch | cta | cta_fused | jacobi | vendor | winner |
|---|---|---|---|---|---|---|
| 8 | 32768 | 0.0132190 | 0.0105830 | 0.0143920 | 0.044342 | fused 1.25× |
| 9 | 29127 | 0.0213110 | 0.0281870 | 0.0544510 | 0.079653 | cta |
| 32 | 8192 | 0.3825800 | 0.5948600 | 2.4824000 | 0.470900 | cta |

(n = 4 jacobi/fused; n = 5..7 fused 1.21×–1.39×; n = 12, 20, 24 cta; n = 16 tie.) `syev_cta` is right for complex
from n ≥ 9; Jacobi is 4×–6× off the pace at n ≥ 20, so the float rule would be wrong here. For complex<double>,
fused leads `cta` by only 1.03×–1.08× at n = 4..16 (neutral), so that range is not split.

### syev: the complex double vendor handover

The complex<double> `jobz=V` rows of `tuned/syev.cdouble.*.txt` put `vendor` first for n = 25..32. 2026-08-07,
median of 3, µs per matrix (cta / cta_fused / vendor): n = 20 3.0015 / 3.4597 / 3.1085 (cta); n = 24 4.0806 /
4.6301 / 3.8184 (tie); n = 26 5.4993 / 5.3048 / 4.4835 (vendor 1.23×); n = 32 8.0253 / 7.4929 / 5.6531 (vendor
1.42×). An FP64-rate artifact and machine-specific. complex<float> does not cross over (`cta` beats the vendor
1.23× at n = 32), so the rule is deliberately not "all complex".

### syev: the fused CTA kernel design

`syev_cta` runs three launches (`sytrd_cta`, `steqr_cta`, `ormqx_cta`) plus two pack kernels and a copy on the
eigenvector path, each stage round-tripping global memory. `syev_cta_fused` keeps one problem resident in one
sub-group partition from load to store (one read of A, one write, against about seven round trips) with the same
device code (`sytrd_cta_device.hh`, `steqr_cta_device.hh`), so a head-to-head measures fusion alone.

- The reduction always runs the Upper path (a Lower input is symmetrised during the tile load). Eigenvalues are
  sorted by rank (each lane writes to its rank slot, ties by index), so no sort kernel.
- Hermitian input reduces to a real tridiagonal by the diagonal unitary similarity \f$T' = S^H T S\f$,
  \f$S_0 = 1\f$, \f$S_{i+1} = S_i\,\bar e_i / |e_i|\f$ (every lane replays the recurrence; S is reapplied before
  the back-transform). It keeps two tiles and applies reflectors afterwards as `ormqx_cta(QL, Left, NoTrans)`.
- Real input follows LAPACK DSYEV: `DORGTR` generates Q explicitly and `DSTEQR` is seeded with it; reflector store
  and rotation accumulator share one tile, and lane c owns column c in registers (no cross-lane reduction).
- The shared tile has leading dimension P + 1: with P = 32 every lane of a column read hits one bank.

### syev: the Jacobi kernel design and its accuracy argument

`syev_jacobi_cta` is partition-resident cyclic two-sided Jacobi: one `SubGroupPartition<P>` owns one problem, A and
(optionally) Z stay in local memory, one launch per decomposition.

- **Threshold.** A rotation is applied only when \f$|a_{pq}| > \mathrm{tol}\cdot\sqrt{|a_{pp}|\,|a_{qq}|}\f$
  (Demmel and Veselić; LAWN 169 Remark 2.2), so eigenvalue error follows the condition number of the
  column-equilibrated matrix and graded input keeps small relative error. The absolute test
  \f$|a_{pq}| \le \mathrm{tol}\cdot\max|a_{kl}|\f$ would forfeit this. The bound is proved for symmetric positive
  definite input; indefinite matrices are handled correctly without it.
- **Schedule.** Round-robin pairing over an index space padded to even size, equivalent to cyclic-by-rows (Hari and
  Begović Kovač, ETNA 46, 2017, Thm 2.11).
- **Measured choices.** Local leading dimension P + 1; rotation pair (c, s) as one `vec<Real, 2>`; round-robin pairs
  precomputed once per work-group in packed 16-bit slots (inline cost three integer modulos per pair per lane per
  phase); `rot_count == 0` is the converged bit.

Results: [the 2026-08-03 bake-off](#syev-the-2026-08-03-small-n-bake-off).

### syev: the LOBPCG projected-solve knob

*Historical; retired 2026-10-05.* `BATCHLAS_SYEV_CTA_MAX_N` (0..32, default 32 = off) sent small eigenvector
solves above the threshold to the vendor, motivated by n = 30, batch 8, float: `cta` 229.6 µs against cuSOLVER
103.7 (2.21×); nsys put 29.4% of LOBPCG GPU time (about 16% end to end) in that projected Rayleigh–Ritz solve.
Through LOBPCG, eigenvectors, n = 256, µs/matrix, value 32 (off) / 16 / 8 / 0: batch 8 15563 / 14211 / 13746 /
13551; batch 64 1998.6 / 1890.4 / 1948.0 / 1945.9 (1.10×–1.15× and 1.03×–1.06×). It shipped **off** because
moving the projected solve off CTA flips a near-tie in `ILUKTests.SyevxInstrumentationAndPreconditioner` (asserts
`lose_count == 0`; at threshold 16 one case crosses to ratio 1.25 at an already-converged point, 4.2e-06).
**Open owner decision:** tolerate a tie on an already-converged case, or pin that test's projected-solve provider.

## syev: tuning constants and the shadowed header

### syev: the shadowed tuning header

Two files answer `#include <batchlas/tuning_params.hh>`: the committed `include/batchlas/tuning_params.hh` and one
CMake generates into `${PROJECT_BINARY_DIR}/include/batchlas/`. `src/CMakeLists.txt` lists the source include
directory first, so **the committed header wins on every build**; the CMake defaults in
`cmake/BatchLASGeneratedHeaders.cmake` and any profile-driven retune written to the binary directory are dead
(2026-08-03: `ormqr_block_size_for_n(512) = 16`, `sytrd_block_size_for_n(512) = 24`; the generated header would say
128 and 64).

> **Warning:** a retune takes effect only when its constants are ported by hand into the committed header (the
> 2026-08-07 retune, `924b3a59`, did). Making the committed header the single source of truth is not done.

Every accessor is overridable through `BATCHLAS_TUNE_*`; overrides are read on every call and feed
`*_buffer_size` as well as the solve, so they must not change between a buffer-size query and its call.

### syev: the ORMQR regression hypothesis, refuted

Hypothesis: shadowing pinned ORMQR at 16 where a generated header said 128, explaining why BatchLAS stopped beating
cuSOLVER at n = 32..512. Refuted: (1) ORMQR has been 16 in every bucket since the header was created (`7363746`,
2026-03-02); what moved was SYTRD (`91341c1`, unmeasured). (2) A/B at n = 512, eigenvectors, batch 128, median of 5
interleaved rounds: ORMQR = 128 gave 452.32 µs/matrix (IQR 450.73–454.91) against 301.95 (IQR 300.79–303.75) unset,
**1.50× slower**. (3) 16 was the measured optimum wherever the knob mattered. No baseline exists to restore: git
history holds no artifact of BatchLAS beating cuSOLVER at n = 32..512, so nothing may calibrate against that claim.
The values came from a standalone `ormqr_blocked` microbenchmark whose n ≤ 64 search offered only block sizes 4–16
(16 was the ceiling), and the generator's `_derive_param_buckets` keeps the parameters of whichever case had the
lower absolute time, not the best block size per case.

### syev: the 2026-08-03 block-size sweep in the syev context

*Superseded by the 2026-08-07 retune.* First measurement of `BATCHLAS_TUNE_ORMQR_BLOCK_SIZE` and
`BATCHLAS_TUNE_SYTRD_BLOCK_SIZE` through `syev`: blocked provider, float, knee batch capped at 512 by the
[workspace limit](#syev-workspace-footprint-limits-usable-batch). ORMQR 16 was optimal at n = 32..128 (n = 128:
7.446 µs against 8.142 at 32 and 22.805 at 128); at n = 256 and 512 ORMQR 32 won by only 1.016× and 1.059×
(neutral). SYTRD matched the committed 8/8/16/24 at n = 64/128/256/512 (n = 512: 24 gives 339.9, 8 gives 380.8).
Double agreed at n ≤ 128 and preferred SYTRD 16 at n = 512 (1.05×–1.07×, neutral). ORMQR = 128 cost 2.1× at n = 64
and 3.1× at n = 128. Gap: n = 256 missing in both modes and precisions (`syev_blocked` exceeded the 3 GB
per-invocation abort even at batch 476). Raw: `benchmarks/results/syev_retune_phaseB_float.jsonl`.

### syev: the 2026-08-07 constant retune

Commit `924b3a59` retuned on a trimmed space and A/B'd each group end to end at the consumers. CUDA/float, µs per
matrix, before to after: syev n=64 b=4096 1.0798 to 0.9938 (1.087×); syev n=512 b=512 166.95 to 151.01 (1.106×);
gesvd n=128 b=1024 9.2465 to 8.0684 (1.146×); gesvd n=256 b=512 1.032×; gesvd n=512 b=256 1.030×. Everything else
was within 1%.

- **ORMQR 16 everywhere to 16/16/24/48/56.** `gebrd` has its own `GEBRD_BLOCK_SIZE_*` (flat curve against ORMQR's
  steep one, 2.16× between 16 and 56). MEDIUM is neutral (within 0.7%).
- **SYTRD 8/8/16/24/32 to 8/8/16/8/48.** The LARGE bucket (256 < n ≤ 512) went to 8 although the 2026-08-03 sweep
  had 24 as the float optimum at n = 512; whether the trimmed retune measured n = 512 is not recorded (**open**).
  Complex overrides this bucket at the consumer (see
  [the complex panel width](#syev-complex-panel-width-in-the-256-to-512-bucket)).
- **`SY2SB_ORMQR_NB_LARGE = 32`.** The shape gate in `sytrd_sy2sb.cc` fires only at n ≥ 1024 and batch ≥ 32, so
  n = 512 got no WY hint. Worth 1.103× at batch 512 and 1.117× at batch 1024. With the gate active it shadows
  `ORMQR_BLOCK_SIZE_*` on syev's hot path: changing the ORMQR constant leaves the kernel trace identical.
- **stedc threads-per-root 8 and wg multiplier 8, set from syev,** overrule stedc's own bench (wgm 2–4, tpr 4–8),
  which cost syev 2.7% at n = 256; 8/8 wins up to 7.6% at n = 64 and is within noise at n ≥ 512.
- **`SB2ST_BACK_*` deliberately stay 0:** the heuristic picks the winner within 0.6% (n=512 b=512 0.050644 against
  tuned (8,8) 0.050706; n=1024 b=256 0.20701 against (8,16) 0.20575).

Tuning is float-only and the header is regenerated from a float-only bench, so per-type corrections live at the
consumer where the next retune cannot overwrite them.

### syev: complex panel width in the 256 to 512 bucket

`sytrd_block_size_default<T>` (`src/extensions/syev_blocked.cc`) returns 32 for complex at 256 < n ≤ 512. The
harness value 8 costs complex 1.16×–1.20×: a complex panel column moves twice the bytes and does four times the
flops per element, so it needs a wider panel to amortise the O(n) barrier-and-reduction chain of
`latrd_lower_panel`. complex<float>, eigenvectors, blocked, µs/matrix, median of 5, device 1, 2026-08-07:

| n / batch | nb=8 | nb=16 | nb=24 | nb=32 | nb=48 | nb=64 |
| --- | --- | --- | --- | --- | --- | --- |
| 320 / 819 | 197.25 | 173.24 | **166.27** | 168.55 | 175.92 | 190.95 |
| 512 / 512 | 942.68 | 838.73 | 815.30 | **802.74** | 811.89 | 827.70 |

(n = 384 and 448 also have nb = 32 optimal.) `nb = 32` is optimal or within 1.5% at every n in the bucket. Float at
n = 512 spreads only 1.04× across nb 8–48 (inside the 1.10× neutral band), so float keeps the harness value. The
override lives at the consumer, not in `tuning_params.hh`, because that header is regenerated from a float-only
bench. Other complex buckets: n ≤ 128 prefers 8; 128 < n ≤ 256 prefers 16 (n = 192: 33.83 against 36.09 at 8);
n > 512 is within 1.03× of the committed 48 at nb = 32.

### syev: the two-stage band width kd

The prediction that removing the split-WY penalty (`f7f3c57`, `nb = kd`) would move the eigenvector optimum to
kd = 96–128 (MAGMA; Gates, Tomov and Dongarra 2018) is **disproved**. Total ms, eigenvectors, median of 3,
2026-08-04:

| n / batch | kd=16 | kd=32 | kd=64 | kd=128 | blocked |
|---|---|---|---|---|---|
| 256 / 1024 | 69.5 | **61.3** | 65.7 | 75.7 | **40.9** |
| 512 / 512 | 217.7 | **194.2** | 236.1 | - | **191.4** |
| 1024 / 128 | 445.3 | **369.9** | 433.2 | 715.3 | 475.5 |

kd = 32 is optimal at every n ≥ 256 (kd = 48 and 96 are worse throughout); wide kd wins only at n = 128 (kd = 128: 16.9 against 22.0 at kd = 32; blocked 14.6), where two-stage loses to
blocked anyway. The nb hint helps narrow bands (1.02×–1.06× for kd ≤ 64) and hurts kd = 128
(0.93×) because LARFT work is \f$O(m k\,n_b)\f$ (see
[sytrd: the dense-to-band block-width hint](sytrd.md#sytrd-the-dense-to-band-ormqr-block-width-hint)).

**"Two-stage wins at n ≥ 1024" is shape-dependent.** At n = 2048 / batch 32 (kd = 32: 1066 ms; kd = 128: 1943), legacy `latrd` gave blocked 1235.1
against two-stage 1066 (two-stage 1.16×); grid `latrd` (the default; 1.41× over legacy) gives blocked **875.8**
(blocked 1.22×). At n = 1024 / batch 128 the batch saturates the 128 SMs and two-stage wins 1.29× either way.

**Mode.** Both modes reduce at a real band width and share `choose_two_stage_kd` (the Householder chase retains
\f$Q_2\f$; see
[sytrd: the Householder chase against the Givens chase](sytrd.md#sytrd-the-householder-chase-against-the-givens-chase)).
The literature puts the eigenvector optimum higher (96/128 on GPU); this table does not reproduce it.

### syev: latrd grid gate confirmed in eigenvector mode

`latrd_grid_min_n = 768` was derived in eigenvalues-only mode and applies in both. legacy/grid ratio in
eigenvector mode (> 1 means grid wins), batch 1 / 8 / 64: n = 512 1.012 / 1.018 / 0.956; n = 768 **1.379** /
**1.375** / 1.090; n = 1024 **1.843** / **1.781** / -; n = 256 0.734 / 0.734 / 0.788 (n = 384: 0.881–0.894). The
crossover is the same in both modes, so the constant is not split. The win shrinks with batch: once the batch
alone saturates the SMs there is no starvation left to absorb.

## syev: the 2026-08-08 implementation plan, measured

The plan turned a profile of float and complex<float> `syev` at saturating batch into work packages with an
end-to-end accept gate (a kernel win is not a solver win: a 2.16× kernel win became an 11% `gesvd` loss). Rig:
`build-cuda` (sm_89, RelWithDebInfo), device 1, one process, idle GPU, `--warmup=2 --min_iters=5`; positional
arguments `n batch nb fuse jobz uplo`. Rig validation at n = 512, batch 512, two-stage, eigenvectors: float 334.28
against the research profile's 332.5, cfloat 971.17 against 972.4; differences under about 1% are noise. Accuracy
runs planned for WP1, WP3, WP8 are not recorded.

| WP | item | gate | measured | verdict |
|---|---|---|---|---|
| WP0 | harness unblockers | cfloat rows appear | rows appear | passed |
| WP1 | A2: `stebz` values-only | ≥ 1.20× float n=256 | **1.44×** | passed |
| WP4 | A3: per-type tile/subs | ≥ 1.10× cfloat, float unchanged | **1.139×**, float unchanged | passed |
| WP3 | A1: `her2k` trailing update | ≥ 1.05× cfloat | 1.043× | marginal miss, kept |
| WP2 | B3: `cta-large-n` | ≥ 1.15× vs blocked | **0.012× (85× slower)** | rejected |

### syev: harness unblockers (WP0)

`sb2st_hh_benchmark` registered only `float`, so `--type=cfloat` silently produced zero rows (fixed; n = 512, batch
512, kd = 32: CHASE float 56.578, cfloat 267.70 ms, 4.73×; BACK 99.095 / 317.03, 3.20×).
`BATCHLAS_LATRD_GRID_GROUPS` cannot force the grid `latrd` path: it is clamped to `MAX_COMPUTE_UNITS / batch`, so at
batch ≥ 128 no value reaches the grid kernel. `BATCHLAS_LATRD_GRID_FORCE_UNSAFE` bypasses the cap but is
deadlock-capable: the software grid barrier hangs, without failing, when the launch exceeds residency (it looks
like slow JIT; run under a timeout). It was added and **never exercised**.

### syev: stebz values-only in the blocked solver (WP1)

In values mode the blocked solver ran a full eigenvector divide-and-conquer and discarded Z (28.3% of the float
solve at n = 256, batch 1024). `stebz` bisection (indices 0..n-1, ascending) now runs straight off the real
tridiagonal, as two-stage already did (`f28a3fcd`). Values only, `provider=blocked`, batch 1024, µs per matrix
(before to after): float n = 64 1.1614 to 0.62228 (**1.87×**), n = 256 20.774 to 14.476 (**1.44×**), n = 320 42.804
to 31.438 (1.36×); cfloat n = 64 1.5055 to 1.0465 (1.44×), n = 256 47.218 to 38.250 (1.23×), n = 320 101.45 to
87.666 (1.16×). The arithmetic bound was 1.39× against a gate of 1.20×. `stedc` runs in real arithmetic, so it is
28.3% of the float solve but about 12% of the cfloat solve at n = 256, hence float gains more. Values mode also
stops allocating eigenvector-only workspace (n = 256, batch 1024, cfloat: `z_span` 268 MB, `zc_span` 537 MB, plus
the stedc workspace).

### syev: per-type sb2st back-transform geometry (WP4)

`provider=two_stage`, eigenvectors, n = 512, batch 512, µs per matrix: cfloat shipped constants 971.17, forced
tile=2 subs=4 852.77 (1.139× faster), per-type code **852.46**; float shipped 334.28, forced tile=2 subs=4 384.30
(**1.15× slower**), per-type code 331.25 (unchanged). A global flip trades a 1.14× complex win for a 1.15× float
loss; the per-type selection (`8c24626f`) keeps the complex win and leaves float at its optimum. The complex
back-transform runs at SM throughput 35.3% and occupancy 49.8% (float 93.7% and 82.8%). Only n = 512 was measured,
so the rule is `n == 512` exactly. The instantiated set is tile ∈ {1,2,4,8} × subs ∈ {4,8,16}; any other pair
silently falls through to the slower tiled kernel. WP4 and WP6 are not additive: measure any complex stage-2
estimate against 852, not 971.

### syev: her2k trailing update for complex float (WP3)

For complex, `sytrd_blocked`'s trailing update issued two full \f$n_2 \times n_2\f$ GEMMs where float issues one
triangle-only `syr2k`. `her2k` does half the arithmetic (\f$\alpha A B^H\f$ and \f$\bar\alpha B A^H\f$ are
conjugate transposes) via one batched GEMM into scratch plus a Hermitian fold (`405b4120`). Primitive,
\f$n_2\f$ = 480, batch 512, cfloat, k = 32: her2k 3.3010 ms against the GEMM pair 4.3866 (1.33×; 1.32× at k = 16,
24), nearly independent of k, so a larger `nb` will not grow the win. End to end, `provider=blocked`, eigenvectors,
n = 256, batch 1024: cfloat 66.628 to **63.899 (1.043×)**; float unchanged. The gate was 1.05×; the miss was
predicted and the change kept (real 1.04× at no runtime cost; it forced a single-definition route prediction in
`expansion_budget.hh`). The gate should read 1.04×, not "met".

- **Fallback.** Forcing the host loop (`BATCHLAS_EXPAND_ROUTE=loop`) at \f$n_2\f$ = 480, k = 32, batch 512 gave
  5.3924 ms, 1.23× slower than the GEMM pair. The 7.8× warned about in the `sytrd_blocked.cc` comment was measured
  in double.
- **Fit guard.** When the expansion does not fit, `her2k_vendor` silently drops to a per-item host loop. Budget
  `GLOBAL_MEM_SIZE / 4` (`triangular_expand.hh`), about 6 GB here; scratch is
  \f$n_2^2\cdot\text{batch}\cdot\text{sizeof}(T)\f$ (n = 512 / batch 512: 944 MB; 2048 / 64: 2.08 GB).
  `BATCHLAS_EXPAND_MAX_BYTES=1` proves the guard works. complex<double> is not admitted (unmeasured).

### syev: cta-large-n rejected (WP2)

The unmerged `cta-large-n` branch lifted the CTA eigensolver above n = 32 (local-memory limits n = 128 float, 64
cfloat). Tests passed but it had no performance measurement. Forced `provider=cta` against `blocked`, float,
eigenvectors, batch 256: n = 33 150.17 against 1.7715 µs (**85× slower**); n = 64 760.68 against 3.6021 (**211×
slower**). At batch 2048 an n = 33..128 sweep finished no iteration in ten minutes while holding 24 GB. A
CTA-resident solve saves about 15 launches but runs an unblocked level-2 algorithm in one work-group per matrix
against a blocked level-3 pipeline. Reverted (`14bad498`; it survives on `worktree-wf_3b4af334-426-9`). The branch
also rewrote the live n ≤ 32 CTA kernels and the sub-group partition in `sg_compat.hh`, never measured; if wanted
it returns as its own change with n ≤ 32 numbers.

> **Note:** benchmark through the forced provider. Under flat selection the cap is `kSmallMaxN = 32` in the small
> families' `can_run` (`src/ops/syev/syev.cc`). A measured dead end in
> [the agent guide](../developer/agent-guide.md).

## syev: the stedc merge-variant and leaf-cliff regression

Measured 2026-08-05. PR 55 (`0bb92fb`, the STEDC level-driver rework) cost `syev` 1.05×–3.25× at every size (the
unchanged vendor column reproduced the grid, so the deltas are code). Float, eigenvectors, baseline / after PR 55 /
fixed, µs per matrix: n = 64 1.64 / 2.25 / 1.70; n = 320 74.53 / 242.43 (**3.25×**) / 70.96; n = 640 727.48 / 1095.9
(**1.51×**) / 714.99; n = 1024 2441.93 / 2559.1 / 2394.7. Fixed is within 0.96×–1.05× of baseline at all eight
sizes.

**Defect A: the merge variant was flipped the wrong way.** `3072ea6` moved `STEDC_MERGE_VARIANT_*` from FusedCta (2)
to Fused (1), claiming FusedCta was numerically wrong and 2–12% slower; neither reproduced. With variant 2 all 16
CUDA `stedc_tests` pass in both precisions, plus `syev_tests` 8/8, `syev_blocked_tests` 44/44 and
`syev_two_stage_tests` 20/20; the NaN was a symptom of the deadlock fixed in `4bde59a`. FusedCta is 11–35% faster on
`stedc` on both drivers (n = 64: Levels+Fused 0.672, Levels+FusedCta 0.499; n = 512: 24.744 against 22.228).

> **Warning:** `FusedCtaConditionedHeavyDeflation` asserts only finite-and-sorted, never accuracy. Strengthen it
> before trusting variant 2 on unmeasured hardware. Current merge constants are on the
> [stedc page](stedc.md#stedc-current-tuning-values).

**Defect B: the level planner chose a leaf past the STEQR cliff.** `plan_stedc_levels` searched leaves in
[threshold/2, threshold·2], tie-breaking toward the leaf nearest the threshold; at n = 320 and 640 it chose leaf 40
over 20. `steqr` takes the fast `steqr_cta` path only for n ≤ the sub-group width and falls back to `steqr_wg`
above it. `steqr` alone, batch 10416, eigenvectors: n = 32 0.262 µs, n = 36 **3.761**, n = 40 4.870, n = 64 36.78. A
12.5% width increase costs 14×: a code-path cliff, not scaling. At n = 320 thresholds 16..24 chose leaf 20 (stedc
12.5–13.1) while threshold 32 chose leaf 40 (**168.4**; 649.7 at threshold 64). `STEDC_RECURSION_THRESHOLD = 32` is
the sub-group width, an invariant. The fix caps the leaf at the threshold (`hi = threshold`, `stedc_levels_plan.hh`);
the level driver is then faster at these sizes (10.8 against 11.1 Recursive at n = 320). It escaped review because
`stedc_benchmark` registered only n ∈ {64, 128, 256} (320 and 640 now are), and the plan shape is asserted directly
in the host-only `StedcLevelPlan.*` tests, since a bad leaf still gives correct eigenvalues.

## syev: workspace footprint limits usable batch

**Open defect, measured 2026-08-03.** `syev_cta`, n = 32, eigenvectors, peak device memory by batch: 512 445 MiB;
1024 2237 MiB (0.320 µs/matrix); 4096 7505 MiB (0.192); 16384 **24083 MiB** (whole card; 0.592, regresses). That is
about 1.8 MB of workspace per 32×32 matrix (roughly 450× the 4 KB of data), linear in batch above 512.
Eigenvalues-only is unaffected (about 343 MiB at batch 16384). `syev_blocked` has the same problem: eigenvalues-only
at batch 16384 (n = 32), 15258 (64), 3814 (128) and 953 (256) each exceeded 3 GB (predates WP1, not re-measured).
Usable batch is capped near 4096 for n = 32 with vectors on a 24 GB card, below saturation; it caused a real OOM on a
shared machine, binds every small-n eigenvector solve, is uninvestigated, and makes n = 2048 unsaturable.

Past the knee extra batch buys little (n = 8, float, `syev_cta`): batch 65536 gives 0.0075321 µs/matrix in 487 MiB;
batch 1048576 gives 0.0071809 (4.7% better) in 24083 MiB (the run that caused the OOM). Saturation therefore means
the knee, where a 4× batch step buys less than 5%. The bake-off driver enforces about 1.5 GB estimated footprint, a
3 GB hard abort per invocation and a 120 s timeout, and reports a cell that does not reach its knee within them.

## syev: open debts and unmeasured items

### syev: open debt, the single-read panel symv

**Plan WP5 (B1), open.** The "open 2.7×" in [the agent guide](../developer/agent-guide.md). The `latrd` panel is
35–71% of every blocked solve. Counters on `LatrdLowerPanel` (ib = 32, j₀ = 0), as a multiple of ideal traffic for
float n=256 b=1024 / float n=512 b=256 / cfloat n=512 b=256: DRAM 0.14× / 1.13× / 1.43×; L2 1.95× / **2.34×** /
**2.50×**; L1TEX **11.9×** / **12.2×** / **15.8×**; SM throughput 52.7% / **10.8%** / **10.7%**. `ncu` on
`LatrdLowerPanelKernelLegacy<float, 256, 0>`, n = 512, batch 256 (2026-08-08): L1TEX 52,499 MB, SM 10.84%,
occupancy 33.19%, 11.444 ms; grid (256,1,1), one work-group per matrix.

**Mechanism.** Thread *r* reads \f$A(r,c)\f$ on its row walk and thread *c* reads the same element as \f$A(c,r)\f$
on its column walk (2.3× L2); on the column walk consecutive lanes are `lda` apart (12–16× L1). The DRAM floor at
n = 512 caps the gain at **2.7×**. The kernel reaches only about 330 GB/s of roughly 1000 GB/s, so it is latency- or
occupancy-bound.

**Design constraint.** The symv sits inside the per-reflector loop of one work-group per matrix, with `v_local[n]`
and `wcol_local[n]` in local memory; reflectors are sequentially dependent, so a MAGMA-style multi-block symv is not
a drop-in. Proposed: stage 32×32 off-diagonal tiles with a coalesced load, then compute
\f$y_R \mathrel{+}= A_{RC} v_C\f$ and \f$y_C \mathrel{+}= A_{RC}^H v_R\f$ from that one copy (a cfloat tile is 8 KB;
n = 512 cfloat needs 16 KB, still ≥ 2 blocks per SM). An earlier "one sub-group per column" attempt changed the
access pattern without reducing reads. Predicted end-to-end gain at 2.7× on the panel: cfloat n=512 vectors 1.82×,
float n=256 vectors 1.29×, float n=256 values 1.50× (1.56×, 1.22×, 1.36× at 2.0×).
**Gate:** ≥ 1.6× on the panel kernel at n = 512, with `ncu` showing L1TEX falling from about 12× toward 2× and L2
from 2.3× toward 1×.

### syev: open debt, complex stage-2 occupancy

**Plan WP6 (B2), open.** At n = 512, batch 512 the float back-transform is saturated (SM 93.7%, occupancy 82.8%)
and the complex one is not (35.3% / 49.8%); the chase runs 45.0% / 65.5% float and 29.8% / 45.1% complex. Halved
occupancy is a register-pressure signature, so the 3.90× complex cost is not the price of complex arithmetic.

- Type-aware retiling of `unmqr_hb2st_wave`, measured against the post-WP4 852, not 971.
- The chase runs at 4.85× float, above the roughly 4× arithmetic ratio. `__mulsc3` is present in
  `libbatchlas_extensions_sytrd.so`; confirm it is on the chase's hot path (`nm -C ... | grep mulsc3`, then
  disassemble), then write out the hot loop only.

Prize: reaching the roughly 2.2× complex/float ratio the panel reached would cut the two stage-2 kernels from
1353 ms to 704 ms, about 1.49× on the complex two-stage solve (about 650 µs/matrix against 973.8), ahead of
blocked (698) and the vendor (707) at n = 512. Any routing change waits for the measurement.

### syev: open debt, stebz reports no convergence status

The values-mode paths of `syev_blocked` and `syev_two_stage` solve the tridiagonal with `stebz` bisection, which
records no status: its three exits (tolerance met, budget exhausted, midpoint stopped advancing) look alike, so both
tiers clear `info` to 0 for every item. `stebz` needs a convergence flag. The secular solver's budget-exhaustion arm
also uses a literal the tuning parameter does not reach (see [stedc: open debts](stedc.md#stedc-open-debts)).

### syev: remaining unmeasured items

- **n = 2048 eigenvector row** (unsaturated at batch 64; WP1 freed 2·n²·batch of values-mode scratch), the
  **values-mode 320 | 512 boundary**, and **one routing sweep per type and mode after WP5** (WP1, WP5, WP6 each move
  a boundary in `syev.hh`; PR 65's defect class was a float-measured constant applied to every type).
- **SYTRD LARGE = 8 against the syev-context optimum of 24** at n = 512 (see
  [the 2026-08-07 retune](#syev-the-2026-08-07-constant-retune)); check with `BATCHLAS_TUNE_SYTRD_BLOCK_SIZE`.
- **XLARGE bucket in double:** SYTRD 16 beat 24 (1.066×/1.048×) and ORMQR 32 beat 16 (1.053×), measured before
  `924b3a59` moved the bucket to 48/56; not re-measured.
- **Grid residency cap (A5):** unlikely to pay (no starvation at batch ≥ 128).
- **Profile `stedc` internally (B4), port the grid barrier:** stedc is 20.7% of the float eigenvector solve at
  n = 256; profile at batch 1 first.
- **Routing-audit benchmark (`BM_SYEV_RoutingAudit`):** not built; the syev tables stay untimed transcriptions until
  `tools/tune/` is run.
- Grid-resident whole solve at batch 1 (speculative; the vendor leads by up to 15.3×); values-mode n ≤ 32 vendor
  comparison; rocSOLVER (nothing measured).

## syev: rejected ideas

| idea | why rejected | evidence |
|---|---|---|
| restore ORMQR = 128 | 1.50× slower at n = 512; 2.1× and 3.1× slower at n = 64, 128 | [refutation](#syev-the-ormqr-regression-hypothesis-refuted) |
| lift the CTA eigensolver above n = 32 (B3) | 85–211× slower than blocked | [WP2](#syev-cta-large-n-rejected-wp2) |
| global sb2st tile=2/subs=4 | costs float 1.15× | [WP4](#syev-per-type-sb2st-back-transform-geometry-wp4) |
| wide two-stage kd (96–128) | kd = 32 still optimal at n ≥ 256 | [kd sweep](#syev-the-two-stage-band-width-kd) |
| freeze `SB2ST_BACK_*`; adopt stedc's own tpr/wgm | heuristic within 0.6%; stedc's winners cost syev 2.7% at n = 256 | [retune](#syev-the-2026-08-07-constant-retune) |
| native Upper reductions | recover at most the mirror's 0.3–2.0% | [Upper](#syev-the-upper-to-lower-mirror-for-lower-only-providers) |
| the float small-n rule applied to complex | Jacobi 4–6× off the pace at n ≥ 20 | [complex small n](#syev-complex-small-n-kernels) |
| a batch floor on the values-only rule | sent n = 1024 at batch 254 to the vendor, 2.75× | [values routing](#syev-eigenvalues-only-routing) |
| split `latrd_grid_min_n` per mode | same crossover in both modes | [latrd gate](#syev-latrd-grid-gate-confirmed-in-eigenvector-mode) |
| block Jacobi as a speed path (C1/WP8) | 2–11× slower; accuracy-only opt-in. At 1.65–2.1 TFLOP/s (about 3.5% of ~47 TFLOP/s SGEMM) a ~10× flop premium needs 17–21 TFLOP/s to break even | [agent guide](../developer/agent-guide.md) |
| C5 spectral divide and conquer; C6 real embedding of the Hermitian problem | cluster parallelism does not apply at saturating batch; 2× flops and memory | plan, not measured |

## syev: measurement traps

- **Contention inflates the vendor arm too.** n = 768 at batch 32 and 64 looked like a 3.6× blocked win; two
  processes had overlapped and the vendor ran 6885 µs/matrix against 1110 alone. Use physical floors as a check:
  Upper is Lower plus a mirror pass, so Upper/Lower ≥ 1 (an early 0.647× was two processes sharing the device).
- **JIT.** Discard the first run of a fresh process; JIT once fabricated a 3.7× loss.
- **`--name` is a substring filter** and corrupted the original eigenvector grid; key on the CSV `name` column.
  `BATCHLAS_SYEV_ROUTE` throws on an unknown word (the retired `_PROVIDER` silently degraded `TWOSTAGE` to Auto).
- **Incremental builds across a widely-included struct change.** Reverting WP2 removed a `DeviceCaps` member; build
  and tests passed but `two_stage` segfaulted at n ≥ 256 in `~DeviceCaps` (mixed layouts) and a 6% anomaly came from
  the same build. A clean rebuild fixed both.
- **An unsampled regime invalidates a table:** the batch ladder that stopped at 1024, "two-stage wins at n ≥ 1024"
  before grid-`latrd`, the never-measured 320 boundary, the power-of-two-only stedc sweep. Decide routing at
  saturation, and accept changes end to end (stedc's bench was overruled by syev; a 2.16× kernel win became an 11%
  `gesvd` loss).
- **`BATCHLAS_TUNE_*` feeds buffer-size queries too.** Flip it between runs, never between a query and its call.

## syev: raw data and provenance

| data | where |
|---|---|
| small-n bake-off (phase A), both precisions (973 measurements) | `benchmarks/results/syev_retune_phaseA.jsonl` (Git LFS) |
| syev-context block-size sweep (phase B), float | `benchmarks/results/syev_retune_phaseB_float.jsonl` (Git LFS) |
| driver for both, with its resource caps | `benchmarks/syev_bakeoff_driver.py` |
| builds | `7911847` (bake-off), `12963a8` (grids), `0bb92fb` (stedc regression), preset `build/presets/cuda` |
| commits | `80737301` plan, `afc8ae35` baselines, `c42e5ce9`, `8c24626f` WP4, `405b4120` WP3, `f28a3fcd` WP1, `af8768b0` / `be327f5b` results (PR 68); `73eb5510` per-type routing (PR 65); `924b3a59` constant retune |

Phase B is complete except n = 256; only the float file is committed (double phase-B figures appear only on this
page). Run `git lfs pull` to fetch the data (see
[the raw-data rules](README.md#new-raw-data-lives-in-benchmarksresults-in-git-lfs)).

Reproduce one shape (positional `n batch nb fuse jobz uplo`; `nb = 0` means the shipped default):

```
CUDA_VISIBLE_DEVICES=1 BATCHLAS_SYEV_ROUTE=blocked \
  ./build/benchmarks/syev_benchmark --backend=CUDA --type=float,cfloat \
  --warmup=2 --min_iters=5 64,128,192,256,320 1024 0 0 0 0

BATCHLAS_SYEV_ROUTE=blocked|two_stage|vendor|cta|cta_fused|jacobi
BATCHLAS_LATRD_IMPL=legacy|grid          BATCHLAS_LATRD_GRID_MIN_N=<n>
BATCHLAS_SB2ST_BACK_TILE_W=<1,2,4,8>     BATCHLAS_SB2ST_BACK_SUBS=<4,8,16>
BATCHLAS_EXPAND_MAX_BYTES=<bytes>        # force the her2k host-loop fallback
BATCHLAS_TUNE_ORMQR_BLOCK_SIZE=<n>       BATCHLAS_TUNE_SYTRD_BLOCK_SIZE=<n>
```

Kernel attribution that sees cuBLAS: `nsys profile -t cuda -s none`, then `nsys stats --report cuda_gpu_kern_sum`.
`BATCHLAS_KERNEL_TRACE` does not see cuBLAS. Counters on the panel kernel:

```
ncu -k regex:LatrdLowerPanel -c 2 --metrics dram__bytes_read.sum,lts__t_bytes.sum,\
l1tex__t_bytes.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active --csv \
    build/benchmarks/latrd_lower_panel_benchmark --backend=CUDA --type=float 512 256 32 0 0
```

