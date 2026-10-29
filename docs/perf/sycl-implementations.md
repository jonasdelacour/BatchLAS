# DPC++ vs AdaptiveCpp: performance and accuracy A/B {#perf_sycl_implementations}

> **Status:** current · RTX PRO 6000 Blackwell Max-Q (sm_120), threadripper02, DPC++ 7.2.0
> (intel/llvm `b87b3566`), AdaptiveCpp 25.10.0 generic SSCP (LLVM 20), CUDA 13.2 · measured 2026-10-09

The same source tree built with DPC++ and with AdaptiveCpp (acpp), measured interleaved on one GPU.
Auto picks the same route in both trees in all 90 Auto cells. Vendor calls and double gemm match
DPC++; float native kernels are slower on acpp, up to 1.70x; the fused small-n syev kernel is faster
on acpp. Every eigenvalue is bit-identical. Gaps, seam, protocol and decisions:
@ref design_sycl_implementations.

## sycl-impl A/B: headline {#sycl-impl-ab-headline}

acpp / DPC++ time per call, geometric mean over the grid's cells. *Auto* is the shipped route,
*native* the pinned BatchLAS families, *vendor* cuBLAS/cuSOLVER through `impl::run_native`.

| op | dtype | Auto | native families (min-max) | vendor |
| --- | --- | --- | --- | --- |
| gemm | float | 1.026 | 1.053 (0.87-1.33) | 1.025 |
| gemm | double | 1.026 | 1.003 (0.97-1.04) | 1.001 |
| potrf | float | 1.076 | 1.175 (0.88-2.30) | 1.028 |
| potrf | double | 1.016 | 1.071 (0.98-1.23) | 1.006 |
| getrf | float | 1.124 | 1.052 (0.61-1.52) | 1.015 |
| getrf | double | 1.017 | 0.935 (0.36-2.47) | 1.018 |
| geqrf | float | 1.263 | 1.188 (0.97-1.68) | 1.036 |
| geqrf | double | 1.052 | 1.076 (0.84-2.66) | 1.012 |
| syev (Auto) | float | 0.950 (0.54-1.37) | | |
| syev (Auto) | double | 1.018 (1.00-1.06) | | |
| steqr_cta | float / double | | 1.217 (1.07-1.41) / 1.113 (1.00-1.21) | |
| syev_cta | float / double | | 1.230 (1.04-1.53) / 1.113 (1.00-1.20) | |
| syev_cta_fused | float / double | | 0.661 (0.54-0.81) / 1.089 (0.98-1.19) | |
| stedc | float / double | | 1.096 (1.05-1.14) / 1.013 (1.00-1.03) | |

The ratios of 2.30-2.66 are the [48 KiB budget](#sycl-impl-ab-the-48-kib-budget-and-the-opt-in-variant);
the float losses are code generation, not the [launch bounds](#sycl-impl-ab-launch-bounds-r10).
Nsight Systems kernel times agree with both harnesses in both trees.

## sycl-impl A/B: setup and protocol {#sycl-impl-ab-setup}

| Item | DPC++ (A) | AdaptiveCpp (B) |
| --- | --- | --- |
| Compiler | `/opt/dpcpp-cuda/bin/clang++` | `/opt/adaptivecpp-25.10.0-cuda13.2/bin/acpp` (25.10.0+git.9f842c7, clang 20.1.2) |
| Tree | `build-dpcpp`, Release, `nvptx64-nvidia-cuda` (sm_120) + `spir64_x86_64` | `build-acpp-rel`, Release, `--acpp-targets=generic` |
| fp | contract `on`, FTZ off | `-ffp-contract=on` (A11), FTZ off |
| Variant | `nolb`: `BATCHLAS_LAUNCH_BOUNDS` empty | `optin`: `-DBATCHLAS_ACPP_SLM_OPTIN=ON` (97,280 B budget) |

Source `acpp-dual-build` at `ddf07744`; acpp runs at `ACPP_ADAPTIVITY_LEVEL=1`,
`ACPP_RT_SCHEDULER=direct`. The protocol on top of agent-guide §10 is the
[design's](../design/sycl-implementations.md#sycl-impl-performance-ab).

- **Harnesses.** `gemm_transpose_benchmark` (NN, TN, `BATCHLAS_BENCH_LD_PAD=3` and 0, beta = 1),
  `factor_bench` (families interleaved, 9 reps), the `steqr`, `syev_cta`, `syev_cta_fused`, `stedc`
  and `syev` (`BATCHLAS_SYEV_ROUTE=auto`) benchmarks; one process per (cell, implementation), under
  a GPU guard, by a campaign driver not kept in the repository.
- **Grid.** float and double; n = 64..1024 gemm, 8..1024 potrf/getrf/geqrf/syev, 5..32 CTA kernels,
  64..1024 stedc. Batch 16384 (n <= 32), 4096 (64), 1024 (128), 512 (256), 256 (512), 128 (1024).
- **Order and statistics.** A throwaway pass, then reps 0-5 with the implementation order rotated
  per cell and rep; reps 6-9 for the 113 cells with 3% <= |ratio - 1| < 15%. Ratio =
  median(acpp) / median(DPC++); most paired bootstrap 95% CIs are within +-1%. Routes from coverage
  `reached` rows; `factor_bench` pins throw rather than fall back.

## sycl-impl A/B: gemm {#sycl-impl-ab-gemm}

- Float Auto is cuBLAS in every cell (1.00-1.09, the [vendor path](#sycl-impl-ab-vendor-calls-through-run-native));
  double Auto is `tiled` (1.02-1.04), `wide:m=64:n=64:k=16` at packed n >= 256 (1.00).
- `tiled` is the one consistent native loss: 1.17-1.19 in float in every form, 1.02-1.04 in
  double. Nsight Systems, `GemmTiled` float n = 1024: 67.4 ms DPC++, 80.9 ms acpp.
- `reg 128x128x8` loses 1.26-1.33 at n = 64, 1.03-1.08 above. `wide 64x64x16` and `reg 64x64x16`
  are 0.95-1.11; `direct` 0.87-1.04. `reg 128x64x32 u4` cannot run on acpp (49,920 B, A2).

## sycl-impl A/B: potrf, getrf and geqrf {#sycl-impl-ab-factorizations}

Auto's route is identical in DPC++, acpp and the opt-in tree in every cell. Largest Auto ratios:

| op, dtype | Auto acpp / DPC++ | Where it loses |
| --- | --- | --- |
| geqrf float | 0.98-1.70 | `blocked` n = 128-1024: 1.35-1.70, same with the opt-in; native panel kernels (gemm sub-routes are cuBLAS in both) |
| getrf float | 1.00-1.49 | `tiny` n = 8-32: 1.16-1.49 ([A10](#sycl-impl-ab-launch-bounds-r10)); `blocked` n >= 256: 1.03-1.09 |
| potrf float | 0.92-1.36 | `tiny` n = 8: 1.20; `lpanel:panel=8` n = 256: 1.36; `blocked` n = 1024: 1.23 (opt-in 1.06) |
| geqrf double | 1.00-1.12 | `blocked` n >= 128: 1.04-1.12 |
| potrf, getrf double | 1.00-1.11 | n = 8: potrf `tiny` 1.04, getrf `vendor` 1.11 |

getrf `blocked` at n = 8-16 runs 0.36-0.67 on acpp in both precisions; no table routes it there.
geqrf `vendor` is not run at n >= 256 (cuSOLVER loops over the batch: 0.42 s per call at float n = 512).

## sycl-impl A/B: the sub-group CTA eigensolvers {#sycl-impl-ab-sub-group-cta-kernels}

The real-kernel A/B for the `SscpBackend` of `sg_partition` ([design: JIT-time target dispatch](../design/sycl-implementations.md#sycl-impl-device-target-dispatch)):
n = 5..32, batch 16384, eigenvectors, each kernel called directly. Paired 95% CIs within +-0.02.

| kernel | dtype | n = 5 | 8 | 12 | 16 | 24 | 32 | DPC++ ms at n = 16 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `steqr_cta` | float | 1.41 | 1.29 | 1.24 | 1.23 | 1.07 | 1.07 | 0.300 |
| `steqr_cta` | double | 1.15 | 1.16 | 1.19 | 1.21 | 1.00 | 1.00 | 7.87 |
| `syev_cta` | float | 1.52 | 1.36 | 1.23 | 1.21 | 1.06 | 1.04 | 0.418 |
| `syev_cta` | double | 1.16 | 1.18 | 1.17 | 1.20 | 1.00 | 1.00 | 8.53 |
| `syev_cta_fused` | float | 0.81 | 0.78 | 0.68 | 0.65 | 0.57 | 0.54 | 0.536 |
| `syev_cta_fused` | double | 1.10 | 1.13 | 1.16 | 1.19 | 0.98 | 0.99 | 8.49 |

- `syev_cta_fused` float is faster on acpp (Nsight Systems, n = 32: 3.30 ms DPC++, 1.78 ms acpp).
  ptxas for `SyevCtaFusedKernel<float, 32, vectors>`: DPC++ 84 registers and a 64 B stack frame,
  acpp 94 registers and no stack ([the JIT-ed PTX check](../design/sycl-implementations.md#sycl-impl-jit-ptx-check)).
- syev Auto inherits these: float n = 16 and 32 take `cta_fused` (0.65, 0.54). stedc inherits
  `steqr_cta`: float 1.05-1.14, double 1.00-1.03. syev Auto float `blocked` (n = 64, 256) is
  1.29-1.36 and `two_stage` (n = 1024) 1.12, the same in the opt-in tree; double syev Auto is
  1.00-1.06.

## sycl-impl A/B: launch bounds (A10) {#sycl-impl-ab-launch-bounds-r10}

acpp drops `[[intel::max_work_group_size, min_work_groups_per_cu]]` (the JIT emits `.maxntid`, no
`.minnctapersm`). The `nolb` DPC++ tree isolates the attribute. Tiny kernels at batch 32768:

| kernel | no bounds / DPC++ (float, cfloat) | acpp / DPC++ float, n = 16 / 24 / 32 | acpp / DPC++ cfloat, n = 16 / 24 / 32 |
| --- | --- | --- | --- |
| getrf `tiny` | 1.00-1.01 | 1.17 / 1.25 / 1.56 | 0.99 / 1.10 / 1.13 |
| gesv `tiny` | 1.00-1.03 | 1.16 / 1.30 / 1.33 | 1.07 / 1.12 / 1.09 |
| posv `tiny` | 1.00-1.03 | 1.15 / 1.29 / 1.34 | 0.98 / 1.10 / 1.11 |
| gemm `reg 128x128x8` (`kMinGroupsPerCu = 2`), float NN, n = 256-1024 | 0.99-1.00 | 0.98-1.08 | |

- **On sm_120 the launch bounds are worth 0-3%.** They were tuned on sm_89
  ([LU](lu.md#the-column-bucket)); not re-measured there.
- **The acpp loss is code generation.** `GetrfTiny<float, 32, 32>`: acpp's JIT-ed PTX has 8,422
  instructions against 6,028, 1,315 branches vs 723, 350 `selp` vs 1,168 (branches where DPC++
  emits selects); 61 vs 80 registers, neither spills.

## sycl-impl A/B: the 48 KiB budget and the opt-in variant {#sycl-impl-ab-the-48-kib-budget-and-the-opt-in-variant}

acpp 25.10 caps a launch at 48 KiB of local memory ([design: the local-memory budget](../design/sycl-implementations.md#sycl-impl-slm-budget)).
`can_run` routes around it (the family is `n/a`) and blocked drivers derive smaller blocks from the
45,056 B budget. The opt-in tree restores 97,280 B; its sub-routes then equal DPC++'s.

| cell | family | acpp / DPC++ | opt-in / DPC++ |
| --- | --- | --- | --- |
| potrf float n = 64 / 128 / 256 / 1024 | `blocked` | 2.30 / 1.45 / 1.61 / 1.22 | 1.03 / 1.20 / 1.07 / 1.06 |
| potrf float n = 1024 | Auto (`blocked`) | 1.23 | 1.06 |
| getrf double n = 32, geqrf double n = 32 | `cta` | 2.47, 2.66 | 1.02, 0.99 |
| potrf/getrf float n = 64, geqrf double n = 64 | `cta` | n/a | 1.02 / 1.01 / 0.91 |
| gemm float NN packed n = 64 / 1024 | `reg 128x64x32 u4` | n/a | 1.19 / 0.77 |

- **Auto is unaffected** except potrf float n = 1024. `reg 128x64x32 u4` never wins Auto here (on
  DPC++ it is 1.9-4.1x slower than the fastest family in these cells).
- **The opt-in costs 4-7% on short launches** (tiny potrf float n = 8: 1.24 acpp, 1.29 opt-in;
  getrf tiny float n = 16: 1.22 vs 1.30): its interposer wraps every `cuLaunchKernel`. This is why
  `BATCHLAS_ACPP_SLM_OPTIN` is not the default ([decisions](../design/sycl-implementations.md#sycl-impl-decisions)).

## sycl-impl A/B: vendor calls through run_native {#sycl-impl-ab-vendor-calls-through-run-native}

Same cuBLAS/cuSOLVER call in both trees; only the submission differs (DPC++ `get_native` +
`create_event_after_external_work`, acpp `AdaptiveCpp_enqueue_custom_operation`). At or above about
1 ms per call the ratio is 1.00 (double gemm, all 15 cells); short calls cost acpp 3-17 us more
(gemm float strided n = 64: 1.09-1.10; factorization `vendor` n = 8: 1.02-1.15). Not investigated:
potrf float `vendor` n = 512 at 1.09, posv cfloat `vendor` at 0.73-0.85 (batch 32768).

## sycl-impl A/B: saturation and adaptivity level 2 {#sycl-impl-ab-saturation-and-adaptivity}

- **Saturation** (18 cells at twice the batch). geqrf float n = 512, potrf float `blocked` n = 512
  and syev float n = 1024 gain over 10% per item, so they are below saturation at the grid's batch;
  geqrf float's ratio shrinks 1.37 -> 1.17. Elsewhere the ratio moves by at most 0.06.
- **Adaptivity level 2** (one shape per process, which flatters acpp). It moves the n = 32
  factorization cells by up to 0.28 either way (potrf `cta` 1.07 -> 0.79, potrf `tiny` 0.92 -> 1.18)
  and the other cells by at most 0.07. Level 1 is the reported setting.

## sycl-impl A/B: per-implementation tables {#sycl-impl-ab-per-implementation-tables}

Tables are keyed by device (`tuned/*.sm_120.txt`), not implementation
([design: the performance protocol](../design/sycl-implementations.md#sycl-impl-performance-ab)). Of
the 78 cell groups with two or more runnable pinned families, the fastest family differs in three:

| group | DPC++ fastest | acpp fastest | DPC++'s pick on acpp | acpp's pick on DPC++ |
| --- | --- | --- | --- | --- |
| gemm double TN ld+3 n = 128 | `tiled` | `wide 64x64x16` | 1.028x slower | 1.004x slower |
| getrf double n = 8 | `vendor` | `tiny` | 1.043x | 1.018x |
| potrf double n = 1024 | `blocked` | `vendor` | 1.095x | 1.003x |

Each is a near-tie on DPC++, and Auto took the same route in all 90 cells: **the data does not
support per-implementation tables.** The acpp losses are per-kernel code generation, not ranking
inversions.

## sycl-impl A/B: accuracy {#sycl-impl-ab-accuracy}

`scripts/impl_diff.py run` (2026-10-09, `build-dpcpp` against `build-acpp-tests`) runs one accuracy
harness in both trees on the same graded inputs (`--seed=1234`, log10 cond 0..10 in 0.1 bins, 128
matrices per bin) and pairs rows per sample; the DPC++ run writes the inputs (`--inputs DIR`) and acpp
reads them back (the device generator differs for double n >= 64). About 3.6 M paired samples, n = 8..256.

| Family (shared inputs) | dtype | Bit-identical | Paired geo-mean B/A | Max paired rel. diff |
| --- | --- | --- | --- | --- |
| eigenvalues: steqr, steqr_cta, stedc, syev_cta, syev_blocked, syevx, syev Auto, cuSOLVER, NETLIB | float, double | 100% | 1 | 0 |
| eigensolver residual R, orthogonality O (eigenvectors of steqr, steqr_cta, stedc, syev); ortho Chol2, Cholesky, ShiftChol3, Householder | double | 100% | 1 | 6.4e-12 |
| same | float | 81-83% | 1 | 2.2e-7 (<= 2 ulp) |
| ortho SVQB, SVQB2 | float, double | 39-50% | 0.89-1.05 | 6.7e3 (past cond ~1/sqrt(eps), chaotic in both) |
| ortho CGS2 | float, double | < 1% | 0.98-1.04 | 2.1 |

- **No accuracy regression on acpp.** No median or paired geo-mean ratio leaves [0.87, 1.09].
- Float R and O differ because the work-group sum is DPC++'s native reduction against acpp's
  `portable::detail::wg_reduce_floating` ([design: the group-algorithm wrappers](../design/sycl-implementations.md#sycl-impl-group-algorithms));
  per-element PTX is the same. Why double R and O are bit-identical is not explained.
- `max_relerr` at cond >= 1e3 measures the dsterf reference, not the solver (a harness property).

## sycl-impl A/B: route parity {#sycl-impl-ab-route-parity}

`scripts/route_diff.sh capture` (`ctest -LE slow`) in `build-dpcpp`, `build-acpp-tests` and the
opt-in `build-acpp-slm-optin`, paired on the full shape key:

| Comparison | Decision keys | Same algo set | Changed | Only DPC++ | Only acpp |
| --- | --- | --- | --- | --- | --- |
| DPC++ vs acpp (45,056 B budget) | 10,303 / 10,236 | 9,844 | 34 | 425 | 358 |
| DPC++ vs acpp + opt-in (97,280 B) | 10,303 / 10,282 | 10,273 | 1 | 29 | 8 |

Every difference is the 48 KiB budget or test scheduling: the opt-in removes 33 of 34 changes and
cuts the one-sided keys from 783 to 37 (tests sized from capacities, budget-derived block sizes,
tests that skip on `kSlmCappedAt48KiB`). Both trees key the tables `[sm_120]` (A4).

## sycl-impl A/B: traps {#sycl-impl-ab-traps}

- **`BATCHLAS_KERNEL_TRACE` durations are not device times under acpp 25.10**: 25-1,700x DPC++'s
  for the same kernels, while Nsight Systems puts them within 0.54-1.53x. minibench's event timing
  (`command_end` differences) matched Nsight Systems. Use Nsight Systems for per-kernel acpp time.
- **A new tree is a new acpp application**: the opt-in variant JIT-compiles again in a shared
  `ACPP_APPDB_DIR` ([protocol](../design/sycl-implementations.md#sycl-impl-performance-ab)).

## sycl-impl A/B: open debts {#sycl-impl-ab-open-debts}

- **Float code generation on acpp.** geqrf blocked (to 1.70), syev float blocked (to 1.36),
  steqr_cta (to 1.41), the tiny LU/Cholesky solvers (to 1.56), `tiled` gemm (1.19). Only getrf tiny's
  PTX is analysed. Untested next steps: an acpp on a newer LLVM, `-O3`-level flags to the SSCP JIT.
- **The `syev_cta_fused` float win** (0.54) points at DPC++'s 64 B stack frame; not confirmed by a
  DPC++ change.
- **Upstream opt-in.** An upstream `cuFuncSetAttribute` in acpp's CUDA backend would remove both the
  48 KiB losses and the interposer's 4-7%.
- **Not measured:** complex types outside A10, the vendor-free acpp tree, coarse-grained events,
  sm_89, and acpp multipass CUDA (`BATCHLAS_ACPP_TARGETS=cuda:sm_89` / `cuda:sm_120`,
  [design](../design/sycl-implementations.md#sycl-impl-cuda-aot)) against DPC++; every row above is
  `generic` (SSCP).
