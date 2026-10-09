# DPC++ vs AdaptiveCpp: performance and accuracy A/B {#perf_sycl_implementations}

> **Status:** current · RTX PRO 6000 Blackwell Max-Q (sm_120), threadripper02, DPC++ 7.2.0
> (intel/llvm `b87b3566`), AdaptiveCpp 25.10.0 generic SSCP (LLVM 20), CUDA 13.2, driver 595.71.05 ·
> measured 2026-10-09

The same source tree built with DPC++ and with AdaptiveCpp (acpp), measured interleaved on one GPU.
Auto picks the same route in both trees in all 90 Auto cells. Vendor calls (geo-mean 1.00-1.04)
and double gemm (1.00) match DPC++. Float native kernels are slower on acpp in most families, up
to 1.70x (geqrf blocked) and 1.56x (getrf tiny); double native kernels lose up to 1.23x outside the
48 KiB cells. The fused small-n syev kernel is faster on acpp, 0.54-0.81x. Every eigenvalue is
bit-identical across the two trees. Design and phases: @ref design_sycl_implementations.

## sycl-impl A/B: headline {#sycl-impl-ab-headline}

acpp / DPC++ time per call, geometric mean over the cells of
[the grid](#sycl-impl-ab-protocol). *Auto* is the shipped route, *native* the pinned BatchLAS
families, *vendor* the cuBLAS/cuSOLVER family through `impl::run_native`.

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

- **Routes.** All 90 Auto cells take the same top-level route in both trees (coverage `reached`
  rows), and every gemm pin ran exactly its pinned family. Sub-op routes differ only where a
  blocked driver derives its block size from the local-memory budget
  ([R2](#sycl-impl-ab-the-48-kib-budget-and-the-opt-in-variant)); the opt-in tree's sub-routes equal
  DPC++'s in every cell.
- **Not the launch bounds.** Removing `BATCHLAS_LAUNCH_BOUNDS` from DPC++ changes nothing
  (0.99-1.03, [R10](#sycl-impl-ab-launch-bounds-r10)). The float losses are code generation:
  the getrf tiny kernel's JIT-ed PTX has 40% more instructions and 1.8x the branches.
- **No per-implementation tables.** The fastest family changes in 3 of 78 cell groups, each by
  at most 9.5% ([the table decision](#sycl-impl-ab-per-implementation-tables)).
- **Device time verified.** Nsight Systems kernel times agree with both harnesses in both trees
  (`syev_cta_fused` float n = 32: 3.30 ms DPC++, 1.78 ms acpp; getrf tiny float n = 32: 0.112 vs
  0.172 ms; tiled gemm float n = 1024: 67.4 vs 80.9 ms).

## sycl-impl A/B: setup {#sycl-impl-ab-setup}

| Item | DPC++ (A) | AdaptiveCpp (B) |
| --- | --- | --- |
| Compiler | `/opt/dpcpp-cuda/bin/clang++`, DPC++ 7.2.0 pre-release, intel/llvm `b87b3566` | `/opt/adaptivecpp-25.10.0-cuda13.2/bin/acpp`, 25.10.0+git.9f842c7, Ubuntu clang 20.1.2 |
| Tree | `build-dpcpp`, Release, targets `nvptx64-nvidia-cuda` (sm_120) + `spir64_x86_64` | `build-acpp-rel`, Release, `--acpp-targets=generic` (SSCP, PTX JIT at first launch) |
| fp | contract `on` (driver default), FTZ off | `-ffp-contract=on` (R11), FTZ off |
| Variant | `nolb`: as A from a source copy whose `BATCHLAS_LAUNCH_BOUNDS` is empty (0 `.minnctapersm` in its PTX; A has 152) | `optin`: as B plus `-DBATCHLAS_ACPP_SLM_OPTIN=ON` (`build-acpp-rel-optin`, budget 97,280 B as on DPC++) |
| Runtime | `LD_LIBRARY_PATH=/opt/dpcpp-cuda/lib:/opt/lib` | same, plus `ACPP_ADAPTIVITY_LEVEL=1`, `ACPP_RT_SCHEDULER=direct`, one `ACPP_APPDB_DIR` per campaign |
| Source | `acpp-dual-build` at `ddf07744`; `src/` and `include/` unchanged since; uncommitted harness changes (`gemm_transpose_benchmark` gained `BATCHLAS_BENCH_LD_PAD`) | same |

Machine: threadripper02, 4x RTX PRO 6000 Blackwell Max-Q (sm_120, 300 W, no display), one NUMA
node, CUDA 13.2 from HPC SDK 26.5. `benchviz info` for the three trees is in
`implab_perf_provenance_sm120.txt`.

## sycl-impl A/B: protocol {#sycl-impl-ab-protocol}

benchviz pins only `native`/`vendor`, so the campaign uses a dedicated driver
(`benchmarks/results/sycl-implementations/perf/harness/driver.py`) over the benchmark binaries:
one process per (cell, implementation), each under a GPU guard, coverage on.

- **Harnesses.** `gemm_transpose_benchmark` (m = n = k, NN and TN, `BATCHLAS_BENCH_LD_PAD=3`
  (strided) and 0 (packed), beta = 1); `factor_bench` for potrf, getrf, geqrf (all pinned families
  interleaved in one process, 9 reps, its own `rel_sd > 0.10` gate; Auto in a second process);
  `steqr_benchmark` (vectors, 50 sweeps), `syev_cta_benchmark`, `syev_cta_fused_benchmark`
  (vectors), `stedc_benchmark` (tuned threshold, Auto merge), `syev_benchmark` with
  `BATCHLAS_SYEV_ROUTE=auto`. minibench cells use `--warmup=3 --min_iters=8 --max_iters=20
  --min_time=200`.
- **Grid.** float and double; n = 64..1024 for gemm, 8..1024 for potrf/getrf/geqrf, 5..32 for the
  CTA kernels, 64..1024 for stedc, 8..1024 for syev. Batch per n: 16384 (n <= 32), 4096 (64),
  1024 (128), 512 (256), 256 (512), 128 (1024). Pins per cell: Auto plus every family listed in the
  tables; a pin that cannot run the shape is `n/a`.
- **Order.** Rep -1 is a throwaway pass over every cell; no measured acpp process printed the
  "new binaries being JIT-compiled" warning. Reps 0-5 then run every cell, rotating the
  implementation order per cell and per rep. 113 cells with 3% <= |acpp/DPC++ - 1| < 15% got reps
  6-9 (DPC++ and acpp only). 5,812 measured processes; 35 of 5,136 `factor_bench` arm samples
  failed its `rel_sd` gate and are dropped (mostly cuSOLVER double geqrf n = 8, bimodal in both
  trees).
- **Statistics.** Per (cell, family, implementation): the median over processes. The ratio is
  median(acpp) / median(DPC++); the raw cells file adds the paired-by-rep bootstrap 95% CI and the
  across-process `rel_sd`. Most CIs are within +-1%.
- **Routes.** Auto's route is read from coverage `reached` rows per process. Pinned gemm cells are
  checked to have reached exactly the pin; `factor_bench` pins throw rather than fall back.

> **Note:** reps 0-2 and the first 140 processes of rep 3 ran on GPU 3. At 21:49 another user's job
> took GPUs 0 and 3 at 300 W, so the rest of reps 3-9 ran on GPU 2 with a guard that tolerates an idle foreign context (`harness/p5guard.sh`). The
> paired acpp/DPC++ ratio moved by a median of 0.000 (p05-p95 0.987-1.018, 425 cells) between the
> two cards; DPC++ absolute times by +0.6%. Other sessions ran ctest on GPUs 0-1 during 2,538 of
> the processes. One process that saw the foreign job start was dropped.

## sycl-impl A/B: gemm {#sycl-impl-ab-gemm}

acpp / DPC++ time per family; `n/a` = acpp cannot run it (49,920 B of local memory, R2); empty =
not measured at that n. Auto's route is the same in both trees.

**float, NN, ld = rows + 3**

| n | batch | Auto | vendor | tiled | reg 128x128x8 | reg 64x64x16 | reg 128x64x32 u4 | wide 64x64x16 | small | direct | DPC++ Auto ms | Auto route |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 64 | 4096 | 1.09 | 1.09 | 1.18 | 1.33 | 1.00 | n/a | 1.11 | 1.10 | 1.00 | 0.1828 | `vendor` |
| 128 | 1024 | 1.03 | 1.04 | 1.17 | 1.08 | 1.01 | n/a | 1.09 |  | 1.04 | 0.2023 | `vendor` |
| 256 | 512 | 1.03 | 1.02 | 1.19 | 1.03 | 0.98 | n/a | 1.03 |  |  | 0.593 | `vendor` |
| 512 | 256 | 1.02 | 1.02 | 1.19 | 1.04 | 1.00 | n/a | 1.02 |  |  | 1.555 | `vendor` |
| 1024 | 128 | 1.00 | 1.00 | 1.19 | 0.98 | 1.02 | n/a | 1.01 |  |  | 6.104 | `vendor` |

**float, NN, ld = rows (packed)**

| n | batch | Auto | vendor | reg 128x128x8 | reg 128x64x32 u4 | wide 64x64x16 | DPC++ Auto ms | Auto route |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 64 | 4096 | 1.01 | 1.01 | 1.26 | n/a | 1.01 | 0.1599 | `vendor` |
| 128 | 1024 | 1.01 | 1.01 | 1.04 | n/a | 1.00 | 0.1795 | `vendor` |
| 256 | 512 | 1.01 | 1.00 | 1.07 | n/a | 1.00 | 0.5043 | `vendor` |
| 512 | 256 | 1.00 | 1.00 | 1.08 | n/a | 1.00 | 1.49 | `vendor` |
| 1024 | 128 | 1.00 | 1.00 | 1.08 | n/a | 1.01 | 5.727 | `vendor` |

**float, TN, ld = rows + 3**

| n | batch | Auto | vendor | tiled | reg 64x64x16 | reg 128x32x32 | wide 64x64x16 | direct | DPC++ Auto ms | Auto route |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 64 | 4096 | 1.09 | 1.10 | 1.17 | 1.02 | 1.02 | 0.99 | 0.87 | 0.1784 | `vendor` |
| 128 | 1024 | 1.04 | 1.03 | 1.18 | 1.04 | 1.00 | 0.96 | 0.94 | 0.2241 | `vendor` |
| 256 | 512 | 1.03 | 1.03 | 1.19 | 1.01 | 1.01 | 0.95 |  | 0.6224 | `vendor` |
| 512 | 256 | 1.02 | 1.02 | 1.19 | 1.03 | 1.02 | 0.95 |  | 1.923 | `vendor` |
| 1024 | 128 | 1.00 | 1.00 | 1.19 | 1.04 | 1.03 | 0.98 |  | 7.693 | `vendor` |

**double** (ld = rows + 3 unless marked)

| n | batch | form | Auto | vendor | tiled | wide 64x64x16 | small | direct | DPC++ Auto ms | Auto route |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 64 | 4096 | NN | 1.03 | 1.00 | 1.03 | 0.98 | 1.01 | 0.97 | 1.366 | `tiled` |
| 128 | 1024 | NN | 1.02 | 1.00 | 1.02 | 0.99 |  | 0.97 | 2.606 | `tiled` |
| 256 | 512 | NN | 1.03 | 1.00 | 1.03 | 1.00 |  |  | 10.81 | `tiled` |
| 512 | 256 | NN | 1.03 | 1.00 | 1.03 | 1.00 |  |  | 43.33 | `tiled` |
| 1024 | 128 | NN | 1.04 | 1.00 | 1.03 | 1.00 |  |  | 171 | `tiled` |
| 64 | 4096 | TN | 1.03 | 1.00 | 1.03 | 0.98 |  | 0.98 | 1.362 | `tiled` |
| 128 | 1024 | TN | 1.02 | 1.00 | 1.02 | 0.99 |  | 0.98 | 2.588 | `tiled` |
| 256 | 512 | TN | 1.03 | 1.00 | 1.03 | 0.99 |  |  | 10.65 | `tiled` |
| 512 | 256 | TN | 1.04 | 1.00 | 1.03 | 1.00 |  |  | 42.53 | `tiled` |
| 1024 | 128 | TN | 1.04 | 1.00 | 1.04 | 1.00 |  |  | 167.8 | `tiled` |
| 64 | 4096 | NN packed | 1.04 | 1.00 |  | 0.99 |  |  | 1.434 | `tiled` |
| 256 | 512 | NN packed | 1.00 | 1.00 |  | 1.00 |  |  | 10.03 | `wide:m=64:n=64:k=16` |
| 1024 | 128 | NN packed | 1.00 | 1.00 |  | 1.00 |  |  | 157.3 | `wide:m=64:n=64:k=16` |

- `tiled` is the one consistent gemm loss: 1.17-1.19 in float in every form, 1.02-1.04 in double.
  Nsight Systems: `GemmTiled` float n = 1024 is 67.4 ms per call on DPC++, 80.9 ms on acpp.
- `reg 128x128x8` (the launch-bounded kernel) loses 1.26-1.33 at n = 64 and 1.03-1.08 above,
  except 0.98 at n = 1024 strided.
- Float Auto is cuBLAS in every cell, so float Auto's 1.00-1.09 is the vendor-call path, below.

## sycl-impl A/B: potrf, getrf and geqrf {#sycl-impl-ab-factorizations}

acpp / DPC++ per pinned family (`·` = runnable in neither tree, `n/a` = DPC++ only). Auto columns:
the DPC++ time, the route (identical in DPC++, acpp and the opt-in tree), and the acpp and opt-in
ratios. geqrf `vendor` is not run at n >= 256 (cuSOLVER loops over the batch: 0.42 s per call at
float n = 512).

**potrf float**

| n | batch | blocked | cta | lpanel 8 | tiny | vendor | Auto DPC++ ms | Auto route | Auto A/D | Auto opt-in/D |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 16384 | 1.25 | 1.27 | 1.01 | 1.24 | 1.08 | 0.01021 | `tiny` | 1.20 | 1.25 |
| 16 | 16384 | 1.27 | 1.29 | 0.93 | 1.03 | 1.04 | 0.02004 | `tiny` | 1.02 | 1.07 |
| 32 | 16384 | 1.06 | 1.06 | 0.88 | 0.93 | 0.99 | 0.09037 | `tiny` | 0.92 | 0.92 |
| 64 | 4096 | 2.30 | n/a | 0.98 | · | 1.04 | 0.1411 | `lpanel:panel=8` | 0.95 | 0.95 |
| 128 | 1024 | 1.45 | · | 1.01 | · | 1.02 | 0.1745 | `lpanel:panel=8` | 1.00 | 1.00 |
| 256 | 512 | 1.61 | · | 1.36 | · | 0.97 | 0.5152 | `lpanel:panel=8` | 1.36 | 1.37 |
| 512 | 256 | 0.99 | · | n/a | · | 1.09 | 1.537 | `vendor` | 1.00 | 1.00 |
| 1024 | 128 | 1.22 | · | · | · | 1.00 | 4.011 | `blocked` | 1.23 | 1.06 |

**potrf double**

| n | batch | blocked | cta | lpanel 8 | tiny | vendor | Auto DPC++ ms | Auto route | Auto A/D | Auto opt-in/D |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 16384 | 1.18 | 1.18 | 1.12 | 1.04 | 1.02 | 0.03919 | `tiny` | 1.04 | 1.04 |
| 16 | 16384 | 1.04 | 1.03 | 1.07 | 1.02 | 1.01 | 0.176 | `tiny` | 1.01 | 1.01 |
| 32 | 16384 | 1.03 | 1.03 | 1.06 | 1.00 | 1.00 | 0.7648 | `lpanel:panel=8` | 1.05 | 1.05 |
| 64 | 4096 | 1.02 | n/a | 1.15 | · | 1.00 | 0.7539 | `vendor` | 1.00 | 1.00 |
| 128 | 1024 | 1.05 | · | 1.23 | · | 1.00 | 0.9247 | `vendor` | 1.00 | 1.00 |
| 256 | 512 | 1.04 | · | n/a | · | 1.00 | 2.957 | `vendor` | 1.00 | 1.00 |
| 512 | 256 | 0.98 | · | · | · | 1.00 | 10.59 | `blocked` | 1.01 | 1.05 |
| 1024 | 128 | 1.10 | · | · | · | 1.00 | 34.03 | `vendor` | 1.00 | 1.00 |

**getrf float**

| n | batch | blocked | cta | tiny | vendor | Auto DPC++ ms | Auto route | Auto A/D | Auto opt-in/D |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 16384 | 0.61 | 1.14 | 1.16 | 1.12 | 0.01236 | `tiny` | 1.16 | 1.16 |
| 16 | 16384 | 0.67 | 1.13 | 1.22 | 1.06 | 0.02504 | `tiny` | 1.23 | 1.23 |
| 32 | 16384 | 1.00 | 1.05 | 1.52 | 1.00 | 0.1202 | `tiny` | 1.49 | 1.49 |
| 64 | 4096 | 1.19 | n/a | · | 0.99 | 0.6445 | `vendor` | 1.00 | 1.00 |
| 128 | 1024 | 1.20 | · | · | 0.99 | 0.9312 | `vendor` | 1.01 | 1.00 |
| 256 | 512 | 1.08 | · | · | 0.96 | 1.378 | `blocked` | 1.09 | 1.08 |
| 512 | 256 | 1.07 | · | · | 1.00 | 5.246 | `blocked` | 1.06 | 1.06 |
| 1024 | 128 | 1.04 | · | · | 1.00 | 20.51 | `blocked` | 1.03 | 1.02 |

**getrf double**

| n | batch | blocked | cta | tiny | vendor | Auto DPC++ ms | Auto route | Auto A/D | Auto opt-in/D |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 16384 | 0.36 | 1.01 | 1.05 | 1.11 | 0.04029 | `vendor` | 1.11 | 1.11 |
| 16 | 16384 | 0.50 | 1.01 | 0.98 | 1.01 | 0.1663 | `vendor` | 1.02 | 1.01 |
| 32 | 16384 | 0.94 | 2.47 | 0.94 | 1.01 | 0.8469 | `vendor` | 1.01 | 1.01 |
| 64 | 4096 | 0.94 | n/a | · | 1.00 | 1.429 | `vendor` | 1.00 | 1.00 |
| 128 | 1024 | 0.95 | · | · | 1.00 | 2.405 | `vendor` | 1.00 | 1.00 |
| 256 | 512 | 1.01 | · | · | 1.00 | 7.244 | `vendor` | 1.00 | 1.00 |
| 512 | 256 | 1.02 | · | · | 1.00 | 31.21 | `vendor` | 1.00 | 1.00 |
| 1024 | 128 | 1.02 | · | · | 1.00 | 108.9 | `vendor` | 1.00 | 1.00 |

**geqrf float**

| n | batch | blocked | cta | tiny | vendor | Auto DPC++ ms | Auto route | Auto A/D | Auto opt-in/D |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 16384 | 1.13 | 1.15 | 1.19 | 1.15 | 0.01392 | `tiny` | 1.18 | 1.25 |
| 16 | 16384 | 1.12 | 1.11 | 1.00 | 1.03 | 0.05019 | `tiny` | 0.98 | 1.00 |
| 32 | 16384 | 0.99 | 0.97 | 1.17 | 1.00 | 0.3906 | `tiny` | 1.13 | 1.13 |
| 64 | 4096 | 1.28 | 1.03 | · | 1.01 | 1.472 | `cta` | 1.03 | 1.04 |
| 128 | 1024 | 1.68 | · | · | 1.00 | 2.16 | `blocked` | 1.70 | 1.70 |
| 256 | 512 | 1.52 | · | · | not run | 4.076 | `blocked` | 1.52 | 1.53 |
| 512 | 256 | 1.37 | · | · | not run | 10.99 | `blocked` | 1.37 | 1.41 |
| 1024 | 128 | 1.35 | · | · | not run | 29.59 | `blocked` | 1.35 | 1.35 |

**geqrf double**

| n | batch | blocked | cta | tiny | vendor | Auto DPC++ ms | Auto route | Auto A/D | Auto opt-in/D |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 16384 | 0.93 | 0.99 | 0.84 | 1.06 | 0.09766 | `vendor` | 1.08 | 1.08 |
| 16 | 16384 | 0.99 | 0.99 | 1.00 | 1.01 | 0.5599 | `vendor` | 1.01 | 1.01 |
| 32 | 16384 | 0.96 | 2.66 | 1.08 | 1.00 | 3.835 | `vendor` | 1.00 | 1.00 |
| 64 | 4096 | 0.97 | n/a | · | 1.00 | 5.499 | `vendor` | 1.02 | 1.00 |
| 128 | 1024 | 1.11 | · | · | 0.98 | 6.288 | `blocked` | 1.11 | 1.11 |
| 256 | 512 | 1.12 | · | · | not run | 15.34 | `blocked` | 1.12 | 1.12 |
| 512 | 256 | 1.04 | · | · | not run | 50.28 | `blocked` | 1.04 | 1.05 |
| 1024 | 128 | 1.05 | · | · | not run | 174.2 | `blocked` | 1.05 | 1.05 |

- **geqrf float blocked** is the largest Auto loss: 1.35-1.70 at n = 128-1024, the same with the
  opt-in budget, so not R2. Its sub-routes (cuBLAS gemm) are identical in both trees; the loss is
  in the native panel kernels.
- **potrf float `lpanel` at n = 256** is 1.36 (Auto takes it), also unchanged by the opt-in.
- The `blocked` families at n = 64-256 and the `cta` cells marked 2.30-2.66 are the 48 KiB budget
  ([below](#sycl-impl-ab-the-48-kib-budget-and-the-opt-in-variant)).
- getrf `blocked` at n = 8-16 runs 0.36-0.67 on acpp in both precisions; no table routes it there.
- The launch-bound cells (float n <= 32 `tiny`) are in [R10](#sycl-impl-ab-launch-bounds-r10).

## sycl-impl A/B: the sub-group CTA eigensolvers {#sycl-impl-ab-sub-group-cta-kernels}

The real-kernel A/B rule for the `SscpBackend` of `sg_partition` (design §5.3): n = 5..32, batch
16384, eigenvectors. Each kernel is called directly, so the route is the function.

| kernel | dtype | n = 5 | 8 | 12 | 16 | 24 | 32 | DPC++ ms at n = 16 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `steqr_cta` | float | 1.41 | 1.29 | 1.24 | 1.23 | 1.07 | 1.07 | 0.300 |
| `steqr_cta` | double | 1.15 | 1.16 | 1.19 | 1.21 | 1.00 | 1.00 | 7.87 |
| `syev_cta` (sytrd_cta + steqr_cta + back-transform) | float | 1.52 | 1.36 | 1.23 | 1.21 | 1.06 | 1.04 | 0.418 |
| `syev_cta` | double | 1.16 | 1.18 | 1.17 | 1.20 | 1.00 | 1.00 | 8.53 |
| `syev_cta_fused` | float | 0.81 | 0.78 | 0.68 | 0.65 | 0.57 | 0.54 | 0.536 |
| `syev_cta_fused` | double | 1.10 | 1.13 | 1.16 | 1.19 | 0.98 | 0.99 | 8.49 |

Paired 95% CIs are within +-0.02 of every ratio.

- **P4 exit criterion: loss recorded.** `steqr_cta` is 1.07-1.41 in float and 1.15-1.21 in double
  at n <= 16, parity at n >= 24 in double. Nsight Systems on float n = 16: `SteqrCTAKernel` 0.215
  ms per call on DPC++, 0.262 ms on acpp.
- **`syev_cta_fused` float is faster on acpp**, 0.54-0.81, and syev's Auto route takes it at n = 16
  and 32 (syev Auto 0.65 and 0.54). Nsight Systems, float n = 32: 3.30 ms DPC++, 1.78 ms acpp per
  call. ptxas resources for `SyevCtaFusedKernel<float, 32, vectors>`: DPC++ 84 registers and a 64 B
  stack frame; acpp 94 registers, no stack (§6.1 of the design lists the same stack arrays).
- `stedc` (float 1.05-1.14, double 1.00-1.03) and syev Auto (below) inherit these kernels.

## sycl-impl A/B: stedc and syev Auto {#sycl-impl-ab-stedc-and-syev}

| op | dtype | n | batch | DPC++ ms | acpp / DPC++ | opt-in / DPC++ | route (both) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| stedc | float | 64 / 128 / 256 / 512 / 1024 | 4096 .. 128 | 1.415 / 1.273 / 2.312 / 4.340 / 9.134 | 1.10 / 1.14 / 1.11 / 1.08 / 1.05 | 1.10 / 1.14 / 1.11 / 1.08 / 1.05 | merge Auto |
| stedc | double | 64 / 128 / 256 / 512 / 1024 | 4096 .. 128 | 32.0 / 21.9 / 35.3 / 52.4 / 110.9 | 1.00 / 1.01 / 1.00 / 1.02 / 1.03 | 1.00 / 1.01 / 1.00 / 1.02 / 1.03 | merge Auto |
| syev Auto | float | 8 / 16 / 32 | 16384 | 0.069 / 0.536 / 3.24 | 1.06 / 0.65 / 0.54 | 1.07 / 0.65 / 0.54 | `jacobi` / `cta_fused` / `cta_fused` |
| syev Auto | float | 64 / 256 | 4096 / 512 | 4.16 / 13.4 | 1.29 / 1.36 | 1.29 / 1.37 | `blocked` |
| syev Auto | float | 1024 | 128 | 280 | 1.12 | 1.12 | `two_stage` |
| syev Auto | double | 8 / 16 / 32 | 16384 | 0.875 / 5.34 / 44.8 | 1.01 / 1.00 / 1.01 | same | `jacobi` |
| syev Auto | double | 64 / 256 | 4096 / 512 | 41.7 / 87.3 | 1.02 / 1.06 | same | `blocked` |
| syev Auto | double | 1024 | 128 | 962 | 1.00 | 1.00 | `vendor` |

The float `blocked` syev (1.29-1.36) and `two_stage` syev (1.12) are the second-largest Auto
losses after geqrf float. In `two_stage` the acpp tree's sub-routes differ only in geqrf `cta` vs
`blocked` panels (the 48 KiB budget); the opt-in tree matches DPC++'s sub-routes and the same ratio.

## sycl-impl A/B: launch bounds (R10) {#sycl-impl-ab-launch-bounds-r10}

acpp drops `[[intel::max_work_group_size, min_work_groups_per_cu]]` (the JIT emits `.maxntid` from
the launch, no `.minnctapersm`). The `nolb` DPC++ tree isolates the attribute: the same compiler
with the macro empty. The tiny kernels run at batch 32768 (the batch their bounds were tuned at).

| kernel | dtype | n = 16 | 24 | 32 | DPC++ ms at n = 32 |
| --- | --- | --- | --- | --- | --- |
| getrf `tiny`, DPC++ no bounds / DPC++ | float / cfloat | 1.01 / 1.00 | 1.00 / 1.01 | 1.00 / 1.00 | 0.238 / 0.624 |
| getrf `tiny`, acpp / DPC++ | float / cfloat | 1.17 / 0.99 | 1.25 / 1.10 | 1.56 / 1.13 | |
| gesv `tiny`, DPC++ no bounds / DPC++ | float / cfloat | 1.03 / 1.00 | 1.00 / 1.00 | 1.01 / 1.00 | 0.330 / 0.834 |
| gesv `tiny`, acpp / DPC++ | float / cfloat | 1.16 / 1.07 | 1.30 / 1.12 | 1.33 / 1.09 | |
| posv `tiny`, DPC++ no bounds / DPC++ | float / cfloat | 1.03 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 0.258 / 0.879 |
| posv `tiny`, acpp / DPC++ | float / cfloat | 1.15 / 0.98 | 1.29 / 1.10 | 1.34 / 1.11 | |

gemm `reg 128x128x8` (`kMinGroupsPerCu = 2`), float NN: DPC++ without bounds / DPC++ = 0.99-1.00
at n = 256, 512, 1024 in both layouts; acpp / DPC++ = 0.98-1.08.

- **On sm_120 the launch bounds are worth 0-3%.** The acpp losses on these kernels (up to 1.56)
  come from elsewhere.
- **Code generation.** `GetrfTiny<float, 32, 32>`: ptxas `-O3` on acpp's JIT-ed PTX uses 61
  registers, DPC++'s cubin 80; neither spills. The acpp PTX has 8,422 instructions against 6,028:
  1,315 branches vs 723, 2,554 `mov` vs 310, 350 `selp` vs 1,168. The same body is compiled to
  branches where DPC++ emits selects.
- The bounds were tuned on sm_89 ([LU](lu.md#the-column-bucket)). This page does not re-measure
  them there.

## sycl-impl A/B: the 48 KiB budget and the opt-in variant {#sycl-impl-ab-the-48-kib-budget-and-the-opt-in-variant}

acpp 25.10 caps a launch at 48 KiB of local memory (design §5.4). `can_run` routes around it, so a
family is `n/a`, and blocked drivers derive smaller blocks from the 45,056 B budget. The opt-in tree
restores the 97,280 B budget; every one of its sub-routes then equals DPC++'s.

| cell | family | acpp / DPC++ | opt-in / DPC++ |
| --- | --- | --- | --- |
| potrf float n = 64 | `blocked` | 2.30 | 1.03 |
| potrf float n = 128 / 256 / 1024 | `blocked` | 1.45 / 1.61 / 1.22 | 1.20 / 1.07 / 1.06 |
| potrf float n = 1024 | Auto (`blocked`) | 1.23 | 1.06 |
| potrf float n = 64 | `cta` | n/a | 1.02 |
| potrf float n = 512, double n = 256 | `lpanel 8` | n/a | 1.19, 1.20 |
| getrf float n = 64 | `cta` | n/a | 1.01 |
| getrf double n = 32 | `cta` | 2.47 | 1.02 |
| geqrf double n = 32 / 64 | `cta` | 2.66 / n/a | 0.99 / 0.91 |
| gemm float NN ld+3, n = 64..1024 | `reg 128x64x32 u4` | n/a | 1.34-1.75 |
| gemm float NN packed, n = 64 / 128 / 256 / 512 / 1024 | `reg 128x64x32 u4` | n/a | 1.19 / 0.98 / 0.86 / 0.80 / 0.77 |

- **Auto is unaffected** except potrf float n = 1024 (`blocked`, 1.23 -> 1.06): no table ranks the
  refused families first in the measured cells.
- The `cta` ratios of 2.47-2.66 at n = 32 are the budget: the opt-in restores 0.99-1.02. How the
  kernel uses the smaller budget is not analysed.
- **The opt-in tree costs 4-7% on short launches**: tiny potrf float n = 8 is 1.24 on acpp and
  1.29 with the opt-in, getrf tiny float n = 16 1.22 vs 1.30, geqrf tiny float n = 8 1.19 vs 1.26,
  potrf `blocked` float n = 512 0.99 vs 1.05. The interposer wraps every `cuLaunchKernel`.
- `reg 128x64x32 u4` never wins Auto here: on DPC++ it is 1.9-4.1x slower than the fastest
  DPC++ family in these cells.

## sycl-impl A/B: vendor calls through run_native {#sycl-impl-ab-vendor-calls-through-run-native}

The vendor family is the same cuBLAS/cuSOLVER call in both trees; only the submission differs
(DPC++ `get_native` + `create_event_after_external_work`, acpp
`AdaptiveCpp_enqueue_custom_operation`).

- At or above about 1 ms per call the ratio is 1.00 (double gemm 1.00 in all 15 cells).
- Short calls cost acpp 3-17 us more: gemm float strided n = 64 (0.18 ms) 1.09-1.10;
  getrf/geqrf/potrf `vendor` n = 8 1.02-1.15.
- potrf float `vendor` n = 512 is 1.09 (+0.13 ms) and posv cfloat `vendor` is 0.73-0.85 at batch
  32768; not investigated.

## sycl-impl A/B: saturation and adaptivity level 2 {#sycl-impl-ab-saturation-and-adaptivity}

**Saturation.** 18 cells re-run at twice the batch (4 reps): per-item time at 2x / 1x.

| cell | family | batch | DPC++ | acpp | acpp/DPC++ at 1x -> 2x |
| --- | --- | --- | --- | --- | --- |
| gemm float NN ld+3 n = 1024 | Auto, `reg 128x128x8`, `tiled` | 128 | 1.10, 1.09, 1.01 | 1.09, 1.07, 1.01 | 1.00 -> 1.00, 0.99 -> 0.96, 1.19 -> 1.19 |
| gemm double n = 1024 | `wide`, `vendor` (TN) | 128 | 1.00, 1.00 | 1.00, 1.00 | 1.00 -> 1.00 |
| potrf float n = 512 | `blocked`, `vendor` | 256 | 0.87, 1.00 | 0.92, 0.96 | 0.99 -> 1.04, 1.09 -> 1.05 |
| potrf double n = 1024 | `blocked`, `vendor` | 128 | 1.03, 0.96 | 0.98, 0.96 | 1.10 -> 1.05, 1.00 -> 1.00 |
| getrf float n = 512 / double n = 1024 | `blocked` | 256 / 128 | 1.04 / 1.03 | 0.98 / 1.02 | 1.07 -> 1.01 / 1.02 -> 1.02 |
| geqrf float n = 512 | `blocked` | 256 | 0.80 | 0.68 | 1.37 -> 1.17 |
| geqrf double n = 1024 | `blocked` | 128 | 0.99 | 0.97 | 1.05 -> 1.03 |
| stedc float n = 1024 / double n = 512 | | 128 / 256 | 0.96 / 0.97 | 0.95 / 0.97 | 1.05 -> 1.03 / 1.02 -> 1.02 |
| syev Auto float n = 1024 / double n = 256 | | 128 / 512 | 0.85 / 0.97 | 0.81 / 0.92 | 1.12 -> 1.07 / 1.06 -> 1.00 |
| steqr_cta float / syev_cta double / fused float, n = 16 | | 16384 | 0.92 / 0.95 / 0.95 | 0.89 / 0.94 / 0.95 | 1.24 -> 1.20 / 1.20 -> 1.19 / 0.65 -> 0.65 |

geqrf float n = 512, potrf float n = 512 (`blocked`) and syev float n = 1024 gain more than 10% per
item at twice the batch, so they are below saturation at the grid's batch. Their acpp losses shrink
with the batch (geqrf 1.37 -> 1.17). Elsewhere the ratio moves by at most 0.06.

**Adaptivity level 2** (`ACPP_ADAPTIVITY_LEVEL=2`, its own JIT database, three throwaway passes;
the JIT warning stopped after the second; 6 reps). Every process reuses one shape, which is what
level 2 specializes for, so this flatters acpp against AOT DPC++.

| cell | family | DPC++ ms | acpp L1 / DPC++ | acpp L2 / DPC++ |
| --- | --- | --- | --- | --- |
| gemm float NN ld+3 n = 256 / 1024 | `reg 128x128x8` | 0.665 / 7.62 | 1.03 / 0.98 | 0.96 / 0.98 |
| gemm double NN ld+3 n = 256 | `wide` | 10.1 | 1.00 | 0.99 |
| potrf float n = 32 | `blocked` / `cta` / `lpanel 8` / `tiny` | 0.182 / 0.166 / 0.131 / 0.077 | 1.06 / 1.07 / 0.88 / 0.92 | 0.82 / 0.79 / 0.79 / 1.18 |
| getrf float n = 32 | `blocked` / `cta` / `tiny` | 0.188 / 0.368 / 0.111 | 0.99 / 1.05 / 1.54 | 0.78 / 0.89 / 1.17 |
| steqr_cta float n = 16 | | 0.294 | 1.26 | 1.21 |
| syev_cta_fused float n = 16 / syev_cta double n = 8 | | 0.538 / 1.52 | 0.65 / 1.18 | 0.62 / 1.18 |
| stedc double n = 256 | | 35.4 | 1.01 | 1.02 |

Level 2 moves the n = 32 factorization cells by up to 0.28 in either direction (potrf `cta` 1.07 ->
0.79, potrf `tiny` 0.92 -> 1.18) and the other cells by at most 0.07. Level 1 is the reported setting.

## sycl-impl A/B: per-implementation tables {#sycl-impl-ab-per-implementation-tables}

The tables are keyed by device (`tuned/*.sm_120.txt`), not implementation (design §7). For each of
the 78 cell groups with two or more runnable pinned families, compare the fastest family per tree:

| group | DPC++ fastest | acpp fastest | DPC++'s pick on acpp | acpp's pick on DPC++ |
| --- | --- | --- | --- | --- |
| gemm double TN ld+3 n = 128 | `tiled` | `wide 64x64x16` | 1.028x slower | 1.004x slower |
| getrf double n = 8 | `vendor` | `tiny` | 1.043x | 1.018x |
| potrf double n = 1024 | `blocked` | `vendor` | 1.095x | 1.003x |

Three changes, each a near-tie on DPC++. Auto took the same route in all 90 cells. **The data does
not support per-implementation tables.** The acpp losses are per-kernel code generation, not
ranking inversions, and the opt-in tree removes the only systematic ranking effect (R2).

## sycl-impl A/B: accuracy {#sycl-impl-ab-accuracy}

Measured 2026-10-09 on GPU 2 (`build-dpcpp` against `build-acpp-tests`, RelWithDebInfo; accuracy
does not depend on the build type). `scripts/impl_diff.py run` runs one accuracy harness in both
trees on the same graded inputs (`--seed=1234`, log10 cond 0..10 in 0.1 bins, 128 matrices per
bin, batch 128) and pairs the rows per sample.

- **Inputs are shared, not regenerated.** The device-side generator is bit-identical across trees
  for float and for double n <= 32. For double n >= 64 it differs in every sample, because its syev
  takes the blocked work-group path. The DPC++ run writes each chunk (`--inputs DIR`) and acpp reads
  it back; the `input_hash` column matches on 100% of shared-input pairs.
- **Grid.** 3 harnesses, float and double, n = 8..256, about 3.6 M paired metric-samples.

| Family (shared inputs) | dtype | n | Bit-identical | B/A median per cond decade | Paired geo-mean B/A | Max paired rel. diff | Flags (t = 2) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `steqr_accuracy`: steqr, steqr_cta, stedc, cuSOLVER syev, NETLIB steqr/sterf/stedc/syev32 (eigenvalues) | float, double | 8, 16, 32 | 100% | 1 | 1 | 0 | 0 |
| `eigensolver_accuracy`: steqr_cta, stedc, syev_cta, syev_blocked, syevx, syev Auto (eigenvalues) | float, double | 8-256 | 100% | 1 | 1 | 0 | 0 |
| same: residual R, orthogonality O | double | 8-256 | 100% (R), 100.0% (O) | 1 | 1 | 1.2e-12 | 0 |
| same: R, O | float | 8-256 | 81% (R), 83% (O) | 1 | 1 | 2.1e-7 (<= 2 ulp) | 0 |
| `orthogonality_accuracy`: eigenvectors of steqr, steqr_cta, stedc, syev | double / float | 16-128 | 100.0% / 83% | 1 | 1 | 1.6e-12 / 2.1e-7 | 0 |
| ortho Chol2, Cholesky, ShiftChol3, Householder | double / float | 16-128 | 100% / 82% | 1 | 1 | 6.4e-12 / 2.2e-7 | 0 |
| ortho SVQB, SVQB2 | double | 16-128 | 50% (100% at n <= 32, 0% at n >= 64) | 0.87-1.09 | 0.93-1.05 | 6.7e3 | 0 |
| ortho SVQB, SVQB2 | float | 16-128 | 39% | 0.87-1.04 | 0.89-1.01 | 5.1e3 | 1: SVQB2 n = 64, 1e5, max only (B/A 2.05) |
| ortho CGS2 | float, double | 16-128 | < 1% | 0.98-1.05 | 0.98-1.04 | 2.1 | 0 |

- **No accuracy regression on acpp.** No median or paired geo-mean ratio leaves [0.87, 1.09].
- **Every eigenvalue is bit-identical**: the CTA kernels, stedc, syev_blocked, syevx, syev Auto,
  cuSOLVER (through `impl::run_native`) and the NETLIB host paths.
- **Float R and O differ by at most 2 ulp** in about 18% of samples. Per-element arithmetic is the
  same PTX in both (`mul.rn`, `add.rn`, `sqrt.approx.f32`); the work-group sum is DPC++'s native
  reduction against acpp's `portable::detail::wg_reduce_floating` (design §5.6). Why double R and O
  are bit-identical over the same grid is not explained.
- **CGS2, SVQB, SVQB2** sum through the §5.6 replacements (and SVQB's blocked syev at n >= 64), so
  bits differ without bias. Past cond ~1/sqrt(eps) SVQB breaks down in both trees and per-sample
  results are chaotic (ratios above 2 and below 0.5 occur 1,483 vs 1,518 times of 12,800 for SVQB2
  float n = 64; geo-mean 0.975). The one flag is a sample maximum in that regime.
- **Own inputs** (each tree generates; eigensolver double n = 32 and 128, ortho double n = 64): n = 32
  stays bit-identical; at n = 64 and 128 no input matches, medians stay within 0.96-1.09
  (eigensolver) and 0.88-1.08 (ortho). These are input effects; none appears with shared inputs.
- `max_relerr` at cond >= 1e3 measures the dsterf reference, not the solver (stedc = syev_blocked =
  syevx to 3 digits within a tree): a harness property.
- acpp repeats byte-identically (eigensolver double n = 32 rerun, `cmp`).

## sycl-impl A/B: route parity {#sycl-impl-ab-route-parity}

`scripts/route_diff.sh capture` (`ctest -LE slow`, GPU 2) in `build-dpcpp`, `build-acpp-tests` and
the opt-in control `build-acpp-slm-optin`; decisions paired on the full shape key (op, scalar, m, n,
k, batch, uplo, side, diag, trans).

| Comparison | Decision keys | Same algo set | Changed | Only DPC++ | Only acpp |
| --- | --- | --- | --- | --- | --- |
| DPC++ vs acpp (45,056 B budget) | 10,303 / 10,236 | 9,844 | 34 | 425 | 358 |
| DPC++ vs acpp + opt-in (97,280 B) | 10,303 / 10,282 | 10,273 | 1 | 29 | 8 |

- **Every difference is R2 or test scheduling.** The opt-in budget removes 33 of 34 changes and
  cuts the one-sided keys from 783 to 37. Both trees key the tables `[sm_120]` (R4 holds).
- The 34 changes: gemm float `reg 128x64x32` u=2/u=4 drop out (9); potrf `cta` above its 45 KiB
  order, `lpanel:panel=8` -> `blocked` (cdouble n = 96, 130), a smaller lpanel:16 window (9); geqrf
  `cta` -> `blocked` (cfloat n = 64-168, float n = 96), `tiny` drops out (7); getrf `cta` ->
  `blocked`/`vendor` (5); getrs (3; `can_run` reads `slm_budget`); gesvd cdouble `jacobi` at
  C = 64 (1).
- One-sided keys: tests sized from capacities (getrf cdouble n = 38 vs 25), blocked drivers' block
  sizes from the budget (potrf cdouble panels k = 32 vs 25), and tests that skip on
  `kSlmCappedAt48KiB`.
- Opt-in residue: `tune_race_gpu_tests` (timing-dependent coverage, batch 512) and NETLIB
  `syev_blocked` shapes missing on DPC++ ("no image" on this box).

## sycl-impl A/B: traps {#sycl-impl-ab-traps}

- **`BATCHLAS_KERNEL_TRACE` durations are not device times under acpp 25.10.** For the same
  kernels its per-kernel `command_start`-to-`command_end` totals were 25-1,700x DPC++'s (syev float
  n = 256: 7.2 s against 73 ms), while Nsight Systems puts the same kernels within 0.54-1.53x of
  DPC++. minibench's event timing reads only `command_end` differences and matched Nsight Systems
  in four cases. Use Nsight Systems for per-kernel acpp time.
- **One measuring process per GPU is not enough on a shared box.** The campaign lost GPU 3
  mid-run to another user's 300 W job; `gpu_guard.sh` refused correctly and the ratios survived the
  move to GPU 2 (see the protocol note), but absolute times moved 0.6%.
- **acpp's JIT database must outlive the warm-up.** One `ACPP_APPDB_DIR` per campaign, shared by the
  throwaway pass and the measured reps; a new tree (the opt-in variant) is a new application
  and JIT-compiles again.

## sycl-impl A/B: open debts {#sycl-impl-ab-open-debts}

- **Float code generation on acpp.** geqrf blocked (1.35-1.70), syev float blocked (1.29-1.36),
  steqr_cta (to 1.41), the tiny LU/Cholesky solvers (to 1.56) and `tiled` gemm (1.19). The PTX of
  getrf tiny shows branches where DPC++ emits selects; the other kernels are not analysed. An acpp
  built on a newer LLVM, or `-O3`-level flags passed to the SSCP JIT, are the untested next steps.
- **The `syev_cta_fused` float win** (0.54) points at DPC++'s 64 B stack frame in that kernel
  (design §6.1); not confirmed by a DPC++ change.
- **Upstream opt-in.** With the 97,280 B budget the R2 cells return to 0.99-1.20; the interposer
  costs 4-7% at launch-bound cells. An upstream `cuFuncSetAttribute` in acpp's CUDA backend would
  remove both.
- **Not measured:** complex types outside R10, the vendor-free acpp tree, coarse-grained events,
  sm_89, and saturation of the cells the table above flags (geqrf float n = 512, potrf float
  `blocked` n = 512, syev float n = 1024).

## sycl-impl A/B: raw data {#sycl-impl-ab-raw-data}

All under `benchmarks/results/sycl-implementations/` (Git LFS).

| File | Content |
| --- | --- |
| `perf/implab_perf_cells_sm120.csv` | Per (cell, family): median ms, across-process `rel_sd`, sample count, routes, `jit` drops for DPC++, acpp, opt-in and no-bounds; ratios and paired bootstrap 95% CIs; sub-route equality |
| `perf/implab_perf_samples_sm120.csv` | One row per (cell, family, implementation, rep): ms, in-process `rel_sd`, residual, SM clock at start |
| `perf/implab_perf_runs_sm120.jsonl.xz` | Every process of the main campaign (warm pass included): argv, env, GPU, guard line, foreign GPU processes, harness rows, coverage `reached` rows |
| `perf/implab_perf_saturation_{cells,runs}_sm120.*`, `perf/implab_perf_adaptivity2_{cells,runs}_sm120.*` | The batch x2 check and the level-2 arm |
| `perf/implab_perf_provenance_sm120.txt` | `benchviz info` for the three trees, compiler and acpp versions, GPUs, `git status` |
| `perf/harness/` | `driver.py` (campaign), `plan.py` (grid), `analyze.py`, `report.py`, `inversions.py`, `consistency.py`, `satcmp.py`, `regs.py`, `ptxhist.py`, `p5guard.sh`, `build.sh`, `nolb.sh` + `patch_nolb.py`; paths are the campaign's job directory |
| `accuracy/` | The accuracy and route-parity evidence, with its own `README.md` |

Command, per campaign (`harness/driver.py --help`):

```sh
python3 driver.py camp1 all --reps 1 --rep0 -1          # throwaway JIT pass
python3 driver.py camp1 all --reps 6 --rep0 0 --gpu 2   # measured reps
python3 analyze.py camp1 --cells cells.csv --samples samples.csv --extend extend.txt
python3 driver.py camp1 all --reps 4 --rep0 6 --cells-file extend.txt --impls dpcpp,acpp
```
