# Blackwell (sm_120) retune

> **Status:** current · RTX PRO 6000 Blackwell Max-Q (sm_120), threadripper02, icpx 2026.0, CUDA 13.2 · measured 2026-09-30

Measurements for sm_120 (188 SMs, 128 MB L2, ~1.8 TB/s GDDR7, 300 W): kernel fixes, tuning
constants and the routing windows that the sm_120 tables in `tuned/` should reproduce. cuBLAS
reaches 43.9-45.8 TFLOP/s SGEMM here, about 43% of the ~106 TFLOP/s FP32 peak (power-limited).

Starting point: benchviz campaign `cuda-20260930-112900` (main @ 4e0a9b7d, float and cfloat). Worst
batchlas/vendor time ratios: cfloat gemm 3.0x geomean (15x at 32x32x1024), float trsm 1.5x (11x at
n=32 q=8), cfloat getrs 1.8x (10x at nrhs=16), cfloat potrf 1.6x (9.5x at n=256), skinny geqrf 2-3.7x.

## What is on main and what sm_120 selects

*measured* = timed by `tools/tune`, *converted* = forced-route sweeps, *transcribed* = an old
router's order per grid cell, untimed (`tuned/README.md`). Main's old routers read no architecture,
so a transcribed sm_120 table holds the sm_89 decisions. "Branch-only" sections were measured on
`worktree-blackwell-tuning` and are not on main.

| Section | On main | sm_120 selection on main |
|---|---|---|
| [icpx fast fp-model](#icpx-fast-fp-model), [device-call guard](#device-call-guard) | yes: `-ffp-model=precise` (`cmake/BatchLASOptions.cmake:294`), `scripts/check_device_calls.py` | n/a |
| [cfloat spmm pair loads](#cfloat-spmm-under-the-precise-fp-model) | yes (`src/sycl/spmm_native.cc`) | `tuned/spmm.*.sm_120.txt`, transcribed: `direct` first for `transA=N` |
| [compute-capability key](#compute-capability-key) | yes | n/a |
| [syev retune](#syev-retune) | no (blocked edge 768, nb 16, WY 32, per-type latrd min-n are branch-only) | `tuned/syev.*.sm_120.txt`, transcribed |
| [potrf LPanel vector sB](#potrf-lpanel-vector-sb) | yes | `tuned/potrf.*.sm_120.txt`, converted from sweeps that include vector sB |
| potrf LPanel auto window, native cap, Blocked nb/W | no | converted rows: float `tiny` to n = 32, `lpanel` first at most cells 36..288, `vendor` or `blocked` above; cfloat `lpanel` first at n = 36 and from 40 to 320 at the larger batches (switch between batch 512 and 8192 depending on n), `vendor` from 384 |
| [posv on sm_120](#posv-on-sm_120) | no posv change | `tuned/posv.*.sm_120.txt`, converted from `sm120_posv_sweep.jsonl` |
| [LU getrs tiny](#lu-getrs-tiny) | no (getrs has `cta`, `blocked`, `vendor`) | `tuned/getrs.*.sm_120.txt`, transcribed (same CSV as sm_89) |
| [LU getrs fused](#lu-getrs-fused), [NR 1](#lu-getrs-fused-nr-1), [sm89 rows](#lu-getrs-fused-sm89-rows) | yes (`c9352406`) | transcribed; the branch's vendor-present sm_120 getrs window is not in the table |
| [getrf windows](#lu-getrf-windows), [native tier](#lu-getrf-native-tier), [gesv](#lu-gesv), [end to end](#lu-end-to-end), [getrs windows](#getrs-windows-after-the-trsm-fix) | no | transcribed (getrf, getrs, gesv) |
| [geqrf skinny panels](#geqrf-skinny-panels) | no (no `geqrf_skinny.cc`) | `tuned/geqrf.*.sm_120.txt`, transcribed (same CSV as sm_89) |
| [trsm V1 ladder cap](#trsm-v1-ladder-cap) | yes (`trsm_v1_ladder_wg`, `src/sycl/trsm_native.cc:97`) | n/a |
| [trsm sub-group Left kernel](#trsm-sub-group-left-kernel), [side right](#trsm-side-right-on-sm120) | kernel yes (candidate `sg_left`); windows no | float and double measured (`tuned/trsm.{float,double}.sm_120.txt`); cfloat and cdouble are the sm_89 transcription, which never ranks `sg_left` |
| [gemm](#gemm-native-register-tiled-selector) | 16x16 and 32x32 wide NN tiles yes; selector windows no | sm_89 transcription: float rows rank `vendor` first on 11154 of 11226 rows and `small` on the other 72; neither small wide tile appears in any sm_120 row, so they run only when pinned |

## Campaign result on the tuning branch

Re-run of the same grid on `worktree-blackwell-tuning` @ ba07a99f (icpx 2026.0 RelWithDebInfo,
precise fp model, 4 GPUs, 2066 arm-cells). Cells are batchlas time / vendor time, with batchlas
pinned to its native walk. Over 1005 cells the geomean fell 0.512 -> 0.291 and cells above 1.1 fell
294 -> 25. Representative rows (geomean before -> after; cells > 1.1 before -> after):

| op | type | cells | geomean | cells > 1.1 |
|---|---|---|---|---|
| gemm | cfloat | 64 | 3.01 -> 0.80 | 55 -> 15 |
| gemm | float | 64 | 1.12 -> 0.82 | 46 -> 1 |
| getrs | cfloat | 40 | 1.76 -> 0.43 | 28 -> 0 |
| gesv | cfloat | 40 | 1.23 -> 0.46 | 18 -> 0 |
| trsm | float | 49 | 1.51 -> 0.52 | 27 -> 0 |
| potrf | cfloat | 8 | 1.56 -> 0.53 | 5 -> 2 |
| syev | float | 8 | 0.66 -> 0.52 | 1 -> 1 |

Other ops (gemv, geqrf, gesvd, getrf, orgqr, ormqr, posv, spmm, syrk, syr2k, trmm, trsm cfloat) end
below 1.0 geomean with at most 4 cells above 1.1. Remaining cells above 1.1:

- cfloat spmm n=16384: 1.11-1.13 in the campaign; the
  [pair-load fix](#cfloat-spmm-under-the-precise-fp-model) brings all four cells to 0.90-0.91 of cuSPARSE.
- cfloat gemm m=n=128..1024: 1.10-1.18 (the large complex tile; not attempted).
- cfloat potrf n=512 (Blocked) 1.48 and n=256 (LPanel) 1.16: the complex trailing update.
- syev cfloat n=1024 b64 1.50; float n=1024 b128 two_stage 1.18 (the sb2st chase is one work-group
  per matrix).
- trmm float n=16 b32768 1.43 and float gemm 8x8x1024 1.12: not addressed.

Full ctest, tuning branch against main: 10 failing suites against 15. `cond_tests`
(`CondTest/2.RandomHermitianTridiagonalLogCondSpectral`, 0.52 decades vs 0.5) and `sytrd_cta_tests`
(`SytrdCtaTest/0.RandomSymmetricUpper`, 2.3e-5 vs 2e-5) newly fail; both fail the same way with
`-ffp-model=precise` alone on main, the fast fp model was hiding them.

## icpx fast fp-model

icpx 2026.0 defaults to `-fp-model=fast`. Device functions then carry a preserve-sign denormal
mode, while the libspirv builtins linked through `-mlink-builtin-bitcode` are IEEE. LLVM does not
inline across that mismatch, so `__spirv_ocl_fma` (every `sycl::fma`), `__spirv_BuiltIn*Id` and
`__spirv_ControlBarrier` become out-of-line calls.

Fix: `-ffp-model=precise` for IntelLLVM in `cmake/BatchLASOptions.cmake`. `-fdenormal-fp-math=ieee`
alone also restores inlining but leaves reassoc and afn on, which
`docs/developer/agent-guide.md` section 11 rules out. The dpcpp-cuda compiler defaults to precise,
which is why the self-built 4090 numbers never showed this.

- `GemmRegister64x64K16WideKernel<complex<float>,true>` SASS: 1031 `CALL.REL` (1024 to a one-FFMA
  stub) under icpx default; 0 calls and 1152 inline FFMA under precise. 4472 of 4539 kernels in the
  icpx images contain calls.
- Same compiler, only the flag toggled (fast -> precise): cfloat wide kernel 128^3 b8192 29.0 -> 3.75 ms;
  512^3 b512 120.8 -> 13.56 ms; batchlas/vendor cfloat gemm 128^3 6.54 -> 0.98, 32x32x1024 15.3 -> 2.03,
  float gemm 1024^3 1.55 -> 1.07; potrf cfloat LPanel n=256 13.87 -> 2.49 ms.

### cfloat spmm under the precise fp model

The flag slows one kernel: the native CSR gather (`SpmmGatherKernel`, `src/sycl/spmm_native.cc`) for
cfloat, 5.51 -> 6.49 ms at n=16384, 16 nnz/row, nrhs=64, batch=32 (cuSPARSE 5.76). The kernel is
L2-bound: each `Cx<float>` element of B was two 4-byte `LDG.E` loads of one sector, and precise
if-converts the 8-column block and issues all 16 halves at once, so both halves of each pair miss
in L1 and send separate L2 requests. ncu, n=16384 cell: fast 5.04 ms / 619M L1 hits / 637M L2
sectors; precise 6.06 ms / 464M / 791M; precise + pair loads 4.83 ms / 23M / 639M (L1 sectors
1256M -> 662M).

Fix: load and store a complex element as one `sycl::vec<R, 2>` (`LDG.E.64` for cfloat, `LDG.E.128`
for cdouble). That needs `2*sizeof(R)` alignment, which `std::complex<R>` does not promise, so the
launcher checks A's values, B and C and falls back to the scalar kernel for a view one R off.
cuSPARSE rejects that case, so the misaligned tests skip unless the native route is pinned. The fix
is not gated by architecture.

Median ms (GPU 0, `spmm_benchmark`, 3 alternating reps; 16 cells): cfloat 16384/64/b32 6.49 -> 5.23 (cuSPARSE 5.74); cfloat 4096/64/b128 5.57 -> 4.63 (5.75); cdouble 16384/64/b16 4.55 -> 3.81 (8.83).

Geomean pair/BASE over 16 cells: cfloat 0.85, cdouble 0.84, float 1.00 (real kernels unchanged).

## Device-call guard

`scripts/check_device_calls.py` (ctest `device_calls_tests`, label `util`; also
`BATCHLAS_BUILD_DIR=build .github/ci/run_local_checks.sh`) guards the fp-model fix. A regression
changes only timings, not test results, so this check is the only signal.

It walks each `build/src/libbatchlas_*.so` for embedded CUDA fatbins (magic `0xBA55ED50`), runs
`cuobjdump -ptx` on each, and fails if any `.entry` calls a `__spirv_*` or `__clc_*` function,
directly or through the local `.func` helpers it calls. It also fails if one of
`GemmRegister64x64K16WideKernel<complex<float>,true>`, `PotrfLpanelKernel` or `TrsmCtaKernel` is
missing. It exits 77 (ctest skip) when there is no NVPTX image or no `cuobjdump`. Scanning 158
images takes 4-7 s.

> **Note:** The check reads PTX, not SASS. In a correct precise build `TrsmCtaKernel` still has 9-33
> SASS `CALL.REL`, which are ptxas's own IEEE division and sqrt slow paths. A PTX `call` names its
> callee, so only PTX is checked.

Proof that it can fail: the same three TUs (`trsm_native.cc`, `potrf_lpanel.cc`, `gemm_kernels.cc`)
device-linked with `-ffp-model=fast` appended (the last flag wins) give 251 kernels, 248 with builtin
calls, 1031 wide-cfloat / 952 lpanel / 8548 trsm-cta call sites, exit 1; as built (precise) 492 kernels,
0 calls, exit 0. The full BASE build (14 libraries, 6277 kernels) has 0 builtin calls; 208 kernels
call non-inlined device functions (`larfg`, `sec_solve_roc`, gesvdj and syrk/trmm tile lambdas), which
are reported but do not fail the check.

Rejected: `__builtin_fma{,f}` in `src/sycl/device_scalar.hh` instead of `sycl::fma`. Under fast it cuts
wide-cfloat calls 1031 -> 7 and lpanel 952 -> 52, but the barrier and id calls remain and the guard stays
red; under precise it changes the SASS of 158 of 6277 kernels. The flag is the fix.

## Compute-capability key

`Device::cuda_compute_capability()` returns `major*10+minor`, parsed from the CUDA device's SYCL
version string and memoized per device. It is the selection device key: `select::describe` names a
CUDA device `sm_<cc>` (`src/select/select.cc:214-215`), which picks `tuned/<op>.<dtype>.sm_<cc>.txt`.
A device with no table of its own borrows the nearest one with a one-time warning. Value 0 means
"not CUDA". All four GPUs here return 120.

`is_sm120_family` (120 <= cc < 130) is no longer a routing predicate. It survives only in the
per-architecture register charge of the fused getrs launch (`src/extensions/getrs_fused.cc:125`).

`tests/util_device_queue_tests.cc` checks the value against the CUDA runtime's
`prop.major*10+prop.minor` for each NVIDIA GPU, paired by ordinal; CPU and non-NVIDIA devices must
return 0. The family window is tested on both sides of each edge (119/120, 129/130). Deliberate
break (cc = major): exactly the two `CudaComputeCapability*` cases go red.

`Device::get_devices()` used to return `idx = 0` for every entry (the cc test checked GPU 0 four
times). `DeviceTest.GetDevices` now requires `devices[i].idx == i`.

## syev retune

Branch-only (SYEV-1/2/3/5), `worktree-blackwell-tuning` @ 43b9f806. icpx 2026.0 precise,
RelWithDebInfo; GPU 1 (headless, one measuring process under a lock). `syev_benchmark`, avg ms,
eigenvectors, Lower, `--warmup=3 --min_iters=8`, alternating A/B order, medians of 2-4 reps.

The branch kept its values in `src/extensions/syev_arch_tuning.hh` and `syev.hh`, applied only to
`is_sm120_family`. Main's sm_120 syev table is transcribed from its architecture-blind router, so
these numbers are the starting point for a measured sm_120 syev table.

### syev float blocked-two-stage edge

The 4090 router sends 449 <= n <= 1024 to two_stage. On sm_120 blocked wins through n = 768 at every
batch, and two_stage wins at saturation from n = 896. The branch's edge is `kSyevSm120FloatBlockedMaxN`
= 768.

| n | batch | blocked | two_stage | winner |
|---|---|---|---|---|
| 768 | 512 | 387.1 | 424.2 | blocked 1.10x |
| 896 | 256 | 315.6 | 347.8 | blocked 1.10x |
| 896 | 512 | 649.4 | 635.5 | two_stage 1.02x |
| 1024 | 128 | 236.3 | 282.7 | blocked 1.20x |
| 1024 | 256 | 494.9 | 485.8 | two_stage 1.02x |
| 1024 | 512 | 990.8 | 893.5 | two_stage 1.11x |

Blocked is saturated by batch 128 at n = 1024; two_stage still gains from batch.

> **Note:** At n = 1024, batch 128, the branch keeps two_stage (282.7 ms against vendor 236.2)
> because the chase kernel runs one work-group per matrix (not addressed). A table keyed on n and
> batch can express the exception.

### syev panel and back-transform blocks

Sweeps of `BATCHLAS_TUNE_SYTRD_BLOCK_SIZE` and `BATCHLAS_TUNE_ORMQR_BLOCK_SIZE`, one knob at a time,
native:blocked, ms:

Sytrd block nb sweep, native:blocked, ms (nb 8 / 16 / 32 / 48 default): cfloat 1024 b64 334.9 / **323.5** / 340.1 / 366.8; float 1024 b128 243.4 / **240.9** / 255.0 / 274.6; float 640 b512 221.7 / **215.2** / 219.6 / 227.6.

ormqr 24-56 shows the same optimum (WY 32: cfloat 1024 b64 326.0 vs 365.4 at the 56 default; float 640 b512 223.6 vs 227.3).

Above n = 512 the branch uses sytrd nb = 16 (`kSm120SytrdBlockXlarge`) and a WY block of 32
(`kSm120OrmqrBlockXlarge`). Main keeps 48 and 56 on every device. `ORMQR_BLOCK_SIZE_XLARGE` also feeds
ormqr, gesvd_blocked and syevx; the branch left that shared constant unchanged. The complex
256 < n <= 512 bucket stays at 32 (nb 16 measured 1.03x, within noise).

### latrd grid min-n

The grid latrd path runs only when `MAX_COMPUTE_UNITS / batch >= 2` (batch <= 94 here). A grid panel
of exactly the threshold n loses to the legacy kernel, so the constant is the largest legacy n
(`kSm120LatrdLegacyMaxN`) and the grid starts one above it. Per-type minimum panel sizes (4090 gate:
768, costing up to 2.1x at float 768 b32): float 321, double 257, complex<float> 257, complex<double>
129. Complex goes lower because a column carries four times the flops, so the grid barrier pays for
itself earlier; complex<double> goes lowest on the 1/64-rate FP64 unit. Brackets from
`BATCHLAS_LATRD_GRID_MIN_N` sweeps (4 alternating reps, spread under 0.3%), ms, winner in bold:

| cell | 256 | 257 | 288 | 320 | 384 | BASE (768) |
|---|---|---|---|---|---|---|
| cfloat 384 b16 | 37.08 | **36.90** | 36.92 | 37.41 | 39.48 | 40.78 |
| cfloat 512 b32 | 58.06 | **57.92** | **57.89** | 58.49 | 60.70 | 70.50 |

| cell | 96 | 128 | 129 | 160 | 192 | 256 | BASE (768) |
|---|---|---|---|---|---|---|---|
| cdouble 128 b8 | 15.17 | 14.86 | **14.77** | 14.78 | 14.77 | 14.77 | 14.77 |
| cdouble 256 b8 | 51.67 | 51.24 | **51.13** | 51.61 | 52.32 | 55.19 | 56.25 |

Double brackets (b16, 224..384) put the best n at 256-257 (e.g. double 256 b16 19.91 at 257 vs 19.99 at 256); float 384-512 b8..16 wins at 321 (float 512 b16 23.39 vs 24.23 at 256).

Final A/B against BASE (Auto unless noted, ms): cfloat 512 b32 70.53 -> 57.93; cdouble 256 b8 56.23 ->
51.11; double 384 b16 48.45 -> 46.22; float 512 b16 41.91 -> 23.31; float 768 b32 (native) 96.28 ->
46.02; float 512 b512 (native) 145.3 -> 102.5. Every small-batch Auto cell still loses to the vendor
by 1.5-3x (cfloat 512 b32 57.93 vs 19.11), so a measured sm_120 syev table should key on batch as well.

Not changed: the grid residency cap (`resident_cap = MAX_COMPUTE_UNITS`, one work-group per SM; G = 1 at
batch 128 on 188 SMs). Raising it needs a blocks-per-SM argument for the spinning grid barrier and a
launch-at-ceiling test; `BATCHLAS_LATRD_GRID_FORCE_UNSAFE` is compiled out, so the gain is unmeasured.

### syev retune result

BASE vs branch, benchviz native arm, ms: float 512 b512 148.1 (two_stage) -> 103.9 (blocked), vendor 193.0; float 768 b256 238.9 -> 183.0 (246.8); float 1024 b128 288.7 -> 283.3 (236.2); cfloat 1024 b64 365.5 -> 264.2 (183.4); float 768 b32 96.3 -> 46.1 (32.9). Benchviz's `sytrd` stage (`benchmarks/benchviz/ops.py`, `sytrd_nb`) still mirrors the sm_89 nb = 48 for n > 512.

## potrf and posv

GPU 2 (no display), `factor_bench` on the branch against BASE 43b9f806 (icpx precise). Each cell runs
in one process with all arms interleaved, 7 timed iterations, 3 reps with the arm order alternated.
Clocks drift up to 20% between processes, so ratios are taken inside one process: `r = vendor / arm`,
above 1 means BatchLAS is faster. Routes were confirmed with `BATCHLAS_COVERAGE_OUT`.

Only the LPanel vector-sB kernel reached main. The branch's LPanel auto-window edges, native cap and
Blocked nb/W did not; main takes the first runnable entry of the nearest row of
`tuned/potrf.<dtype>.sm_120.txt`, converted from the sm_120 forced-route sweeps of `a1063892`
([potrf.md](potrf.md#selection-since-flat-kernel-selection-phase-2)).

### potrf LPanel vector sB

`potrf_lpanel_body`'s left-looking update read its sB operand with one scalar shared load per FMA
(cfloat: generic `LD.E`). It now reads each sB column as 16-byte `sycl::vec<float,4>` loads through a
local-space pointer, and the `sA` and sB stores use local-space views. Each `rS[i]` still accumulates
`kk` in ascending order, so the arithmetic is unchanged. The change is architecture-independent.

LPanel pinned, `r` BASE -> new: float n64 1.80 -> 1.86, n128 1.23 -> 1.36, n256 0.95 -> 1.15, n320 0.84 -> 1.06, n512 0.76 -> 0.90; cfloat n64 1.29 -> 1.49, n128 1.00 -> 1.18, n256 0.78 -> 0.99, n320 0.61 -> 0.72.

Trap: sB is stored as the scalar type and read back as `sycl::vec<float,4>`. This is type punning,
correct only because barrier B1 sits between the stores and the loads. A double-buffered sB that drops
or moves B1 must store through the vector type too.

`VectorSbReadIsBitIdenticalAcrossPackingAndBatch` guards it. Planted breaks: a float vector width for
every type turns red only the LPanel double, cfloat and cdouble instances; sB slots overlapping by
half a slot turns red the cases that pack G > 1 matrices per work-group, for every type.

### potrf LPanel auto window

Branch-only. Measured edges on the new kernel: float edge 320 (Auto r = 1.40 at 128, 1.13 at 256, 1.02
at 288, 1.03 at 320; pinned LPanel won every rep from 224 to 320 and is 0.72 at 352); cfloat edge 128
(Auto 1.51 at 64, 1.19 at 128; pinned LPanel at 144 is 0.98-1.02, 160 1.01, 192 0.78, so 128 is the
conservative edge). sm_89 keeps 256 for both types. On main the converted sm_120 rows carry these edges
(see the table at the top; example `n=128 batch=8192` `lpanel:panel=8` 2.477 ms against `vendor` 3.176).

### potrf LPanel native tier cap

Branch-only. Float LPanel vs Blocked, ms: 288 1.66 vs 1.97; 320 2.16 vs 2.44; 352 1.91 vs 1.55; 384 2.31
vs 1.77; 512 4.47 vs 3.49, so the branch stopped float LPanel at 320. cfloat LPanel matches or beats
Blocked up to its ceiling (256 2.18 vs 3.01; 320 2.60 vs 2.58; 368 1.92 vs 2.11), so it is not capped.
On main a vendor-free build takes the first non-vendor entry of the converted row, so the cap sits
wherever the sweep ranked `blocked` ahead of `lpanel`.

### potrf Blocked nb and W

Branch-only. `BATCHLAS_POTRF_NB` / `_W` overrides on BASE, float blocked vs vendor `r`:

| float n (batch) | 128/128 (sm_89) | 64/64 | 128/64 |
|---|---|---|---|
| 320 (1024) | 0.78 | 0.95 | 0.85 |
| 512 (512) | 0.95-0.98 | 1.07-1.11 | 1.07-1.13 |
| 768 (128) | 1.06-1.07 | 0.90-0.92 | 1.15-1.17 |
| 1024 (128) | 1.26-1.29 | 1.01 | 1.30-1.32 |

W = 64 wins at every order. nb = 64 wins up to 512 and nb = 128 above it (bracketed only by 512 and 768;
576-704 not measured). Main's `potrf_blocked_params` has no architecture term, so every device runs the
sm_89 pair. `PotrfBlockedTest.ResidualAboveTheCtaCeiling` asserts that the large-order pair runs (n = 647
and 768 on sm_120 float). cfloat (96/32) keeps the sm_89 pair: 64/64 is 0.62 vs 0.68 at 256 and 0.89 vs
0.90 at 512. fp64 was not swept.

Not done: float Blocked beats cuSOLVER at n = 640-1024 (1.16-1.42 at b = 128-256), but at n = 512 the
win falls from 1.12 at b=512 to 1.01 at b=2048, and 1536 and 2048 lose (0.89, 0.76); no vendor-present
Blocked window was added. cfloat Blocked stays at 0.67 of cuSOLVER at n = 512-1024.

## posv on sm_120

posv has no routing change of its own; its potrf sub-op resolves through the windows above. Native
walk, `r` BASE -> new: float n=512 nrhs=1 0.78 -> 1.17, nrhs=16 0.72 -> 0.99; n=256 nrhs=1 0.99 -> 1.14;
n=128 nrhs=1 1.28 -> 1.39.

The fused `posv=native:cta` cells that looked slow in the trsm diagnosis (cfloat 32/1 1.35x, 64/4 1.34x)
came from the icpx build; on BASE they are 1.82x and 1.52-1.59x faster than the vendor composition.
Remaining losses are nrhs > 8 at small n (float 32/16 0.52, 64/16 0.68, 128/16 0.79), which run
`potrf=tiny|lpanel` plus the native trsm; `BATCHLAS_TRSM_ROUTE=vendor` makes the same composition
1.26x faster than the vendor arm at 32/16, so the loss is in the trsm kernel (see
[trsm on sm_120 result](#trsm-on-sm_120-result)).

## LU (getrf, getrs, gesv)

GPU 3 (sm_120), icpx 2026.0 precise, RelWithDebInfo, cuSOLVER and cuBLAS as the vendor.
`factor_bench` verifies items 0 and batch-1 of every arm (every row has bad=0). Medians of 7-9 reps,
2 passes with alternating BASE/new order. BASE is 43b9f806. Ratios are time/vendor; below 1 is
faster than the vendor.

### LU getrs tiny

Branch-only. `getrs_tiny.cc` (`Algorithm::Tiny`) uses one sub-group partition per matrix. Lane r holds
row r of op(A) and a chunk of RHS columns in registers; the permutation is a per-lane trace of the
interchange list and both substitutions are shuffle recurrences. No local memory, no barrier. One
reciprocal per lane is broadcast instead of a divide per step: cfloat n=32 went from 255 registers
(17% occupancy, 0.95 ms) to 128 registers (0.39 ms).

Chunk (A stays in registers, so a narrower chunk costs only occupancy): shipped 2 for complex, 4 for
real. Batch 32768, ms:

| cell | chunk 16 | 8 | 4 | 2 | 1 |
|---|---|---|---|---|---|
| cfloat n32 nrhs16 | 2.39 | 2.27 | 1.33 | 0.95 | 0.89 |
| float n32 nrhs16 | 0.84 | 0.50 | 0.47 | 0.43 | 0.46 |
| float n8 nrhs16 | | | 0.047 | 0.057 | 0.067 |

Window (branch): `tiny_window = sm_120 && single precision && n <= tiny_max_n`. Every measured cell
wins for float and cfloat over n in {2, 4, 8, 12, 16, 24, 32}, nrhs in {1, 2, 4, 8, 16, 64}, batch
32768 (8192 for nrhs 64), and over batch 64/256/1024 at six shapes. Tiny time / vendor time at batch
32768 ranges 0.20-0.58 (float) and 0.22-0.55 (cfloat); the closest small-batch cells are cfloat n32
nrhs64 b64 (0.186 vs 0.197 ms) and cfloat n8 nrhs16 b64 (0.0246 vs 0.0271). n=1 and batch < 64 were
admitted unmeasured. fp64 was not measured and keeps the pre-Tiny walk.

### LU getrs fused

`getrs_fused.cc` ran every nb x nb block solve in sub-group 0, one column at a time, while the other
sub-groups waited at the barrier. Columns now go round-robin over the sub-groups
(architecture-independent). CTA arm, before -> after, ms (vendor in brackets):

| cell | nrhs 1 | 4 | 8 |
|---|---|---|---|
| float n64 b16384 | 0.314 -> 0.320 | 0.584 -> 0.449 (0.64) | 1.04 -> 0.77 (0.75) |
| cfloat n64 b16384 | 0.542 -> 0.563 | 1.08 -> 0.93 (0.83) | 1.99 -> 1.56 (1.06) |
| cfloat n512 b256 | 0.578 -> 0.608 | 1.01 -> 0.79 (1.39) | 1.92 -> 1.20 (1.40) |

nrhs=1 was 1-3% slower (one column to spread), so NR == 1 keeps the sub-group-0 form
([NR 1](#lu-getrs-fused-nr-1)).

In-library register counts exceed the sm_89 table: float Trans nrhs 5..8 uses 86 registers, so the
sm_89 row handed out a 768-lane launch that the driver refused (`FusedGetrsLaunchHoleAt48KiB`). On
sm_120 the cap reads a probed row.

On main (`c9352406`): the round-robin kernel, the NR == 1 guard (`src/extensions/getrs_fused.cc:241-242`)
and the sm_120 register rows (`:125`). Branch-only: a vendor-present sm_120 window (float nrhs <= 8 at
n >= 256; cfloat nrhs <= 4 at n >= 128 and nrhs <= 8 at n >= 512), later moved to float n >= 192 and
cfloat n >= 96 / n >= 384 ([getrs windows](#getrs-windows-after-the-trsm-fix)). Main's sm_120 getrs
table is the sm_89 transcription (`cta` first at `n >= 32` with `nrhs <= 2`; float also nrhs 3-4,
[small-n-baseline.md](small-n-baseline.md#getrs)), so nrhs 3-8 cells go to the vendor.

### LU getrs fused NR 1

At NR == 1 the three fused kernels (getrs NoTrans, getrs Trans, potrs) run the block solves in
sub-group 0 only. The guard is `(NR != 1) || sgid == 0`, so each NR keeps its old code. CTA arm
(`getrs=native:cta`, reached in posv too), BASE -> new, ms, nrhs=1: cfloat n64 b16384 0.560 -> 0.541
(vendor 0.70); float n256 b2048 0.572 -> 0.570 (0.88); cfloat n128 b4096 0.484 -> 0.474 (0.70). Every
nrhs > 1 cell is within 0.4%. fp64 is flat (double n128 b4096 0.5586 -> 0.5601), so the gain is a
float/cfloat result; posv is unchanged. Pre-existing: posv double n128 r1 b4096 runs native at 7.14 ms
against vendor 6.14 ms.

Breaks: `c0 = nrhs` at NR == 1 turns red `LuTest/{4..7}.FusedGetrsSolvesEveryTransposeAtEveryInstantiatedWidth`
and `LuTest/{4..7}.FusedGetrsAtBlockBoundariesAndTheNbSwitch` (8 tests) and
`PosvTest/{4..7}.FusedSolveArmSolvesOnBothTriangles` (4 tests, potrs).

> **Warning:** The sub-group-0 guard is not covered by the suite. With `c_active = true` at NR == 1
> (every sub-group solves column 0 at once), all four binaries stay green: the race writes identical
> values. The guard is there because of the data race, not an observed failure.

### LU getrs fused sm89 rows

The round-robin change altered the kernel bodies behind the sm_89 `GetrsFusedRegs` rows, which were
probed on the 4090 and cannot be re-probed here. Standalone sm_89 ptxas counts of `getrs_fused.cc`,
NR > 1, before -> now (icpx 2026.0 / DPC++ 7.2): cdouble NoTrans nrhs 2 50 -> 54 / 50 -> 51; double
NoTrans nrhs 5-8 55 -> 61 / 54 -> 54; double potrs nrhs 5-8 88 -> 92 / 70 -> 71. Every other kernel
moved by 0 or less (NR == 1 identical). potrs now has its own sm_89 row (it was charged the max of
NoTrans and Trans); a `static_assert` keeps that row >= the old max.

Effect on the sm_89 cap (margin +8, allocation rounded to 8): cdouble NoTrans nrhs 2..4 goes from 1024
to 896 lanes, double NoTrans nrhs 5..8 from 896 to 768, and double potrs nrhs 5..8 from 768 to 640. The
fused width reaches a cap only at n >= 1026. This applies to every non-sm_120 device, including
`cuda_cc == 0`. Only the potrs narrowing fixes a launch the driver would refuse: the icpx double potrs
nrhs 5-8 kernel uses 92 registers, and at 768 lanes ceil(24/4) * 32 * 96 = 18432 > 16384.

Not fixed: under icpx 2026.0 some standalone sm_89 counts already exceeded the rows before the change
(double potrs nrhs 5-8 88 vs 72 + 8; cfloat Trans nrhs 5-8 70 vs 56 + 8). An icpx build for a 4090
needs its own probe; the in-library probe printed no ptxas counts, so this evidence is from
standalone TU compiles.

### LU getrf windows

Branch-only. On main the sm_120 getrf table is the sm_89 transcription. BASE build, ms, vendor /
blocked (loss = blocked slower):

Blocked loses at n48-64 b256 (float 0.067 vs 0.095; n64 cfloat 0.106 vs 0.149), n128 b64 (0.25 vs 0.27) and cfloat n48 b1024; it wins at n64 b1024 (0.22 vs 0.15), n128 b256 (0.34 vs 0.29) and n192 b64 (0.69 vs 0.44).

The branch's blocked window was n >= 192, or n >= 128 at batch >= 256, or n >= 48 (float) / 64 (cfloat)
at batch >= 1024. The float n=65 tail-panel dip wins narrowly (3.06 vs 3.16 ms).

The tiny window beats cuSOLVER at every measured order (vendor / tiny, b32768): float n4 0.0160/0.0095,
n32 0.66/0.25; cfloat n4 0.0173/0.011, n32 0.96/0.70. The branch's window was therefore 4..32 for both
types. The sm_89 window excluded cfloat 8 and 25..32 as ties, and the transcribed tables carry that:
at batch 32768 cfloat `tiny` ranks first at n = 16 and 24, and `vendor` at 4, 8, 25 and 32.

### LU getrf native tier

Branch-only (`native_tier_preferred`). On sm_89 CTA wins n=64 by 1.13x. On sm_120 blocked wins from
float n=33 and cfloat n=40 at large batch (ms, CTA / blocked): float n33 1.28/1.06, n64 4.03/1.77 (b8192);
cfloat n33 1.72/1.79, n40 2.29/2.17, n64 3.90/3.50. At batch 256 CTA still wins float n=48
(0.078/0.095), so the CTA ceiling drops to 32 (float) / 39 (cfloat) only at batch >= 1024. On main a
vendor-free sm_120 build takes the first non-vendor entry of the transcribed sm_89 row.

### LU gesv

With tiny getrs routed, the cfloat composition (tiny getrf + tiny getrs) beats the fused gesv tiny
kernel from n = 4 (b32768, ms, fused / composed): n4 r1 0.022/0.019, n16 r4 0.257/0.160, n32 r4
1.27/1.03. It loses at n <= 3, where the composed getrf is the vendor's (n3 r1 0.018/0.022), so the
branch's cfloat fused window on sm_120 is n <= 3. float keeps 32: fused and composed are within about
5% (a screen, not a bracket). In a vendor-free build the composed getrf at n <= 3 is the tiny one,
which was not timed.

### LU end to end

76 LU cells of the campaign loser list, BASE vs branch, time/vendor: Auto geomean 0.990 -> 0.534
(cells above 1.1: 3 -> 0); native-walk geomean 1.178 -> 0.544. Examples (Auto): getrs cfloat n32
nrhs16 1.13 -> 0.45; getrs float n4 nrhs64 1.51 -> 0.32; gesv cfloat n32 nrhs64 1.00 -> 0.28.

Still losing in the native walk (Auto sends these to the vendor): getrs at n >= 64 with nrhs=16 (float
1.32-1.46, cfloat 1.33-2.01) and gesv cfloat n64 nrhs16 (1.33). These are the blocked composition,
bound by TrsmCta; the nrhs >= 16 windows were re-bracketed in
[getrs windows](#getrs-windows-after-the-trsm-fix).

### getrs windows after the trsm fix

Branch-only. Every window here is an sm_120 clause of the deleted route-era getrs router; main's sm_120
getrs table is the sm_89 transcription, so this grid is the evidence a measured sm_120 getrs table
should reproduce. Measured after the trsm V1 ladder cap and the sub-group Left kernel
(`worktree-blackwell-tuning` @ d776064b), GPU 3.

`factor_bench getrs`, arms vendor, auto, `native:blocked`, `native:cta` (nrhs <= 8) and `native:tiny`
(n <= 32) interleaved in one process, 7 in-process reps, 3 passes with alternating arm order. Speed-up
is the median in-process vendor/arm ratio (above 1: BatchLAS faster). Saturated batch is about 3 GiB of
matrices, capped at 32768. Grid: n in {32, 48, 64, 96, 128, 192, 256, 384, 512} by nrhs in {1, 2, 4, 8,
12, 16, 24, 32, 64, 128}, both types, with batch ladders (128 to 16384) at every edge.

**Composition (`native:blocked`).** At saturation it beats cuBLAS at every nrhs >= 16 cell for n =
32..512 in both types; the weakest are cfloat n128 nrhs16 at 1.04, cfloat n256 at 1.16 and float n256 at
1.37. nrhs <= 15 is erratic (float n512 takes 2.04 ms at nrhs 13 and 1.09 ms at nrhs 14). Batch 128 wins
in every probed cell. Edge cells (vendor / blocked ms, speed-up):

| cell | vendor | blocked | speed-up | side |
|---|---|---|---|---|
| float n32 nrhs16 b32768 | 1.270 | 0.613 | 2.07 | in (order edge) |
| float n512 nrhs16 b603 | 1.865 | 1.084 | 1.72 | in |
| float n256 nrhs15 b2374 | 1.669 | 1.713 | 0.97 | out |
| float n512 nrhs13 b605 | 1.659 | 2.037 | 0.81 | out |
| cfloat n48 nrhs32 b4096 | 0.625 | 0.520 | 1.19 | in |
| cfloat n96 nrhs16 b4096 | 0.944 | 0.883 | 1.07 | in (order edge) |
| cfloat n128 nrhs16 b4096 | 1.473 | 1.400 | 1.05 | in |
| cfloat n48 nrhs16 b4096 | 0.299 | 0.330 | 0.90 | out |
| cfloat n48 nrhs16 b8192 | 0.660 | 0.717 | 0.92 | out |
| cfloat n64 nrhs16 b4096 | 0.466 | 0.467 | 1.01 | out (tie) |
| cfloat n128 nrhs15 b4592 | 1.644 | 2.039 | 0.80 | out |

At saturation cfloat n48 nrhs16 wins (5.09 vs 2.63 ms), so the cfloat n < 96 exclusion is a batch-ladder
loss, not a saturation one. Below order 32 the composition also beats cuBLAS, but the tiny tier is first
in the walk and at least as fast, so the clause stops at 32. The cfloat cells at 1.04-1.07 reproduced
across 8 passes, so they are ties or small wins.

**Fused tier (`native:cta`).** The trsm fix does not touch this kernel. Edge cells:

| cell | vendor | cta | speed-up | side |
|---|---|---|---|---|
| float n192 nrhs8 b4096 | 1.144 | 1.089 | 1.05 | in |
| float n192 nrhs5 b4096 | 1.127 | 0.999 | 1.13 | in |
| float n128 nrhs8 b4096 | 0.578 | 0.623 | 0.92 | out |
| float n96 nrhs8 b4096 | 0.403 | 0.404 | 0.99 | out |
| cfloat n96 nrhs4 b4096 | 0.542 | 0.436 | 1.24 | in |
| cfloat n64 nrhs4 b4096 | 0.224 | 0.236 | 0.95 | out |
| cfloat n384 nrhs8 b539 | 1.277 | 0.908 | 1.40 | in |
| cfloat n256 nrhs8 b1206 | 1.046 | 1.429 | 0.73 | out |
| cfloat n256 nrhs6 b1211 | 1.121 | 1.294 | 0.87 | out |

Float nrhs 5..8 wins at saturation from n = 48, but loses between n = 48 and 128 at batch 4096.

**Native tier (`native_tier_preferred`): unchanged.** Over 119 measured cells with n > 32 and nrhs <= 8,
CTA is faster in 111. The composition wins by more than 5% only at float n48..64 nrhs 6..8 with batch
>= 16384 (1.07-1.12) and at cfloat n128 nrhs8 (CTA 1.77 ms, composition 1.31), a cost anomaly of the
fused kernel, not a tier boundary.

**Effect on Auto** (saturated grid, n = 48..512, geomean vendor/Auto): float 1.63 -> 2.04 (25 cells
leave the vendor), cfloat 1.15 -> 1.55 (38 cells). No cell moves onto a losing route. gesv follows,
because its composition calls the routed getrs.

## geqrf skinny panels

GPU 2 (sm_120), icpx 2026.0 precise, RelWithDebInfo, cuSOLVER as the vendor. Harness:
`factor_bench geqrf <T> m n 0 <batch> 10`; a JIT pass, then 3 passes with alternating BASE/new order,
medians. BASE is `worktree-blackwell-tuning` @ 502ad378. Ratios are time/vendor. The campaign losses
were on the native-pinned walk; Auto sent them to the vendor.

Branch-only. Neither the skinny leg (`geqrf_skinny.cc`), the tile-bytes width rule nor the windows are
on main. Main's sm_120 geqrf table is the sm_89 transcription (`geqrf.csv`), and `BATCHLAS_GEQRF_LEAF`
has no `skinny` value. On the branch the changes applied to float and cfloat on sm_120 only; cc 0, cc 89
and fp64 keep BASE behaviour (`SkinnyWindowIsSm120OnlyAndPicksOneTier`, `Sm120LeafWidthFollowsTileBytes`).

### geqrf leaf width from tile bytes

The CTA tier's resident leaf took its work-group width from the column count alone (`geqrf_panel_wg`:
128 threads at n=4, 256 from n=8), and packed one panel per sub-group only at m <= 32. On the branch,
the CTA tier sizes the group by a fixed tile share per work-item (float 72 B, cfloat 96 B), keeps the
old ladder above 16 KiB of tile, and packs panels of up to 4 KiB into one sub-group at any m
(`geqrf_cta.cc`, `geqrf_leaf_launch`). Kernel-only probe, float, batch 32768, ms: 64x8 wg256 0.93 vs
packed 0.21; 128x8 wg256 1.11 vs wg64 0.46; 64x32 wg256 5.25 vs wg64 3.24.

Limits, each measured:

- **Tile <= 16 KiB** (`kGeqrfNarrowMaxBytes`). Above it the group keeps the ladder width; float 24-28 KiB
  panels lost at 128 B per item (native/BASE 1.07-1.09).
- **Columns >= 8.** Below that the ladder already gives 128, and the rule starved tall panels (cfloat
  257x2..384x2 were 1.2-1.4x slower).
- **Rows <= 256.** Above 256 the result was mixed (cfloat 300x8 0.90x, 384x8 1.07x).
- **Batch >= 2048** (`kGeqrfBytesRuleMinBatch`). Narrower groups starve the device at small batch:
  native new/BASE for float 40x32 is 2.46 / 2.15 / 1.16 / 0.76 at b256 / 512 / 1024 / 2048; float 64x32
  1.58 / 1.42 / 0.77 / 0.96.
- **CTA tier only.** With the rule in the blocked driver's panel leaf, float 256x64 was 1.02x BASE;
  without it every blocked cell is 1.00-1.01x.

### geqrf width rule review fixup

Tiles over 16 KiB keep the ladder, and the tile share per work-item is float 72 B, cfloat 96 B (from
128 B). A sweep inside one binary showed that 128 B left 12-15 KiB float panels on wg64 where wg128 is
faster, and 4-7 KiB panels on wg32 where wg64 is faster; the two types disagree at 10 KiB, hence one
share per type. Float wants wg128 from 9 KiB (at 72 B, 9 KiB is exactly 128 items) and not at 8 KiB;
cfloat wants wg64 at 10 KiB and wg128 at 12 KiB (at 96 B, 12 KiB is 128 items).

Final build, batch 32768, native new/BASE (native/vendor): float 40x32 (5 KiB) 0.56 (0.69); float 128x20
(10 KiB) 0.75 (0.75); float 224x16 (14 KiB) 0.87 (0.67); float 128x32 (16 KiB) 0.93 (0.64); float 160x32
(20 KiB) 1.00 (0.57); float 256x32 (32 KiB) 1.00 (0.41); cfloat 48x16 (6 KiB) 0.53 (0.82); cfloat 128x16
(16 KiB) 0.83 (0.68); cfloat 128x20 (20 KiB) 1.00 (0.68); cfloat 256x16 (32 KiB) 1.00 (0.43). Every cell
is at or below 1.00 on both native arms. The 48-cell run (m 32..256, n 16..32, plus float 64x16 and
256x8) has worst native new/BASE 1.00 and worst Auto/vendor 0.90; at batch 2048 the new widths beat
BASE on 11 cells (0.56-0.97).

Tests: `Sm120LeafWidthFollowsTileBytes` checks 16 KiB + 1 row, 24 KiB (ladder) and the 128-item edge per
type. Breaks: ceiling raised to 32 KiB turns it red for float and cfloat; float share 80 B turns the
float instance red.

### geqrf skinny register leg

`geqrf_skinny.cc` and `geqrf_skinny_device.hh`: a second leg of the CTA tier (the route stays
`native:cta`) for n <= 8. The panel lives in one sub-group's registers. P lanes own a matrix (P = 8 or
16 below 17 rows, so 2-4 matrices share a sub-group; otherwise 32). Row r sits on lane r % P in slot
r / P. No local memory, no barrier; the column reductions are independent butterflies.

Instantiations: N in {4, 8} columns and RP in {1, 2, 4, 8, 16} row slots (m <= 512 for float, <= 256
for cfloat; the cfloat 16-slot bucket was slower than the resident leaf at 257x4 and 300x3).
`MinBlocks = 1`, because a tighter launch bound spilled the RP >= 8 buckets. Tau and beta follow the
resident body's LAPACK convention (`geqrf_larfg_scalars`). `geqrf_skinny_preferred` adds a batch gate: a
lane holding >= 512 B of A (cfloat RP 8 with 8 columns, float RP 16 with 8 columns) needs batch >= 1024.

Skinny / resident at b256 / b512 / b1024 (`BATCHLAS_GEQRF_LEAF=resident|skinny` pins the leg on the
branch; a pin that does not fit throws): cfloat 256x8 1.29 / 1.13 / 0.69; float 400x8 1.13 / 1.02 / 0.60;
cfloat 128x8 1.00 / 0.90 / 0.55; float 256x8 0.77 / 0.72 / 0.46; float 512x4 0.69 / 0.68 / 0.68.

### geqrf sm120 tall window

On the branch, the route-era `route_geqrf.hh` had two sm_120-only clauses. Under flat selection the same
decisions are rows of a measured `tuned/geqrf.*.sm_120.txt` that rank `cta` first in these cells:

- `geqrf_skinny_window`: float n <= 8 with m <= 256, or m <= 512 with n >= 2; cfloat n <= 8 with m <= 256;
  strictly tall (m > n). Square shapes stay with the tiny tier.
- `geqrf_sm120_tall_window`: float and cfloat, 9 <= n < m, 32 <= m <= 512; plus cfloat 300 <= m <= 512
  with 4 <= n <= 8.

Brackets (native/vendor at batch 32768):

| edge | inside | outside |
|---|---|---|
| float skinny, m | 512x2 0.80, 512x4 0.52 | 400x1 1.39, 512x1 1.36 |
| tall, m >= 32 | 32x16 float 0.72, cfloat 0.87 | 24x12 float 1.15, cfloat 1.17 |
| tall, m <= 512 | 512x16 float 0.44, cfloat 0.48 | not measured above 512 |
| cfloat band, m | 300x4 0.90, 300x6 0.86, 400x5 0.84 | 290x4 0.91, 260x6 0.98, 260x5 1.14 |
| cfloat band, n | 512x4 0.82 | 300x3 1.10, 512x2 1.13 |
| cfloat skinny, m | 256x4 0.62 | 257x2 1.03, 257x4 0.97 |

cfloat 257x8 (0.73) and 260x8 (0.75) win but stay with the vendor, because 257x6 (0.95) and 260x5 (1.14)
do not, and a per-n edge was not worth a clause. The window is keyed on shape alone and holds at small
batch. Auto, vendor -> new Auto ms at b256: float 40x32 0.319 -> 0.073; float 400x8 0.094 -> 0.030;
cfloat 256x8 0.085 -> 0.029 (b1024: 0.320 -> 0.161, 0.096 -> 0.037, 0.086 -> 0.041). Routes were checked
with `BATCHLAS_COVERAGE_OUT`: cfloat 400x5, float 64x8 and float 40x32 at b256 run `native,cta`; cfloat
290x4 stays `vendor,auto`.

### geqrf skinny panels: result

Campaign loser cells (native-pinned benchviz arm), batch 32768, new native / vendor: float 8x4 0.46,
16x8 0.35, 64x8 0.34, 128x8 0.40, 256x8 0.37, 64x32 0.74, 128x16 0.87; cfloat 8x4 0.50, 32x8 0.43,
64x16 0.79, 128x8 0.58, 256x4 0.62, 512x4 0.82. Raw cells: `benchmarks/results/`.

The 106-cell grid (m 4..512, n 1..128), geomean time/vendor:

| | BASE Auto | new Auto | BASE native | new native |
|---|---|---|---|---|
| float (49 cells) | 0.846 | 0.537 | 1.038 | 0.544 |
| cfloat (57 cells) | 0.867 | 0.652 | 0.904 | 0.622 |

The new Auto column predates the cfloat band; on the final build cfloat Auto/vendor is 300x4 0.89, 300x8
0.66, 384x8 0.61, 400x5 0.84, 512x6 0.64, 512x8 0.48, and 290x4 stays vendor-routed at 1.00. No native
cell is more than 2% slower than BASE on the grid.

Consumers: `syev` two-stage runs geqrf on m x 32 sy2sb panels at batch 64-512 and `ortho` on tall panels;
`syev_benchmark` is unchanged within 0.5% (float 1024 b256 494.3 / 493.8 ms). `sytrd_sy2sb_tests`,
`syev_two_stage_tests` and `orgqr_tests` pass. `ortho_tests` crashes in its complex<double> case, a
pre-existing cuBLAS Zgemv segfault on this box.

Break sweep (`geqrf_tests` filtered to the CTA, blocked, facade and new suites):

| break | red |
|---|---|
| partition butterfly starts at 16 for every P (cross-matrix leak) | SkinnyLegResidualInEveryBucket, SkinnyLegAgreesWithTheResidentLeaf, SkinnyLegIdentityReflectorInOnePartition, SkinnyLegSaturatingBatchIsBitIdentical |
| skinny store uses m instead of ld | the three residual/agreement/identity suites, FacadeReachesTheSkinnyLeg, CtaResidualAndOrthogonality |
| width gate 2048 -> 2049 | Sm120LeafWidthFollowsTileBytes |
| heavy-lane gate 1024 -> 1025, or skinny window without the sm_120 guard | SkinnyWindowIsSm120OnlyAndPicksOneTier |
| no W1/W2 zero fill | BlockedIgnoresAGarbageWorkspace (all four types) |

Not done: no per-n window refinement between 257 and 299 cfloat rows; fp64 has no skinny instantiation,
width rule or window; float 400x1 and 512x1 (1.36-1.39 native/vendor) stay with the vendor; the skinny
N = 16 bucket is not built (64-512 B stack frame at N = 16).

## trsm

GPU 0 (sm_120), icpx 2026.0 precise, RelWithDebInfo, cuBLAS as the vendor. BASE is
`worktree-blackwell-tuning` @ 502ad378. `trsm_benchmark` (`BM_TRSM<`: Side::Left, Lower, NoTrans,
ld = n), `--warmup=3 --min_iters=8`, one process per cell and arm, a JIT pass, then 3 alternating reps,
medians. The vendor arm is `BATCHLAS_TRSM_ROUTE=vendor`; the others pin `native`.

### trsm V1 ladder cap

The V1 CTA kernel gives each lane one right-hand-side column, but lanes without a column still run the
whole N-step recurrence. Its work-group ladder {256, 128, 64, 32} took the first rung with
`batch * ceil(q / wg) >= 4 * CU`; at saturated batch that is always 256, so a q=8 solve ran 248 dead
lanes per group. The ladder now skips a rung more than half of whose lanes would have no column
(`cand > 32 && cand / 2 >= q`, in `trsm_v1_ladder_wg`). The change is architecture-independent and
correctness-neutral (the group barrier stays).

> **Note:** Measured on sm_120 only. On the RTX 4090 (cc 89) small-q V1 work-group geometry changes at
> saturated batch too; its timing is not measured here, so re-measure before quoting a 4090 number.

V1 alone, BASE -> capped, ms (batch 32768): float n32 q8 Left 0.994 -> 0.284; float n16 q8 Left 0.280 ->
0.067; cfloat n32 q8 Left 1.551 -> 0.44; Right side (cuBLAS after the slash): float n32 q8 1.042 ->
0.306 / 0.165; float n32 q32 1.425 -> 0.400 / 0.740; cfloat n32 q8 1.635 -> 0.426 / 1.147. The q >= 256
ortho shapes never reach a capped rung and stayed within 3%.

### trsm sub-group Left kernel

> **On main.** The kernel is `sycl_trsm::trsm_native_sg_left_dispatch` (`src/sycl/trsm_sg_left.cc`), the
> trsm candidate `sg_left` (Left, order <= 32, sub-group 32; `src/ops/trsm/choice.hh:15`). The branch's
> window is not on main. The measured `tuned/trsm.float.sm_120.txt` decides, and it reproduces the
> window's shape: at batch 32768 the `side=L trans=N` rows rank `sg_left` first for q <= 64 at order 8
> and q <= 32 at orders 16 and 32, and `cta` above those q. The cfloat and cdouble sm_120 tables are the
> sm_89 transcription, which never ranks `sg_left`, so complex Left runs it only when pinned
> (`BATCHLAS_TRSM_ROUTE=sg_left`; a pin above order 32 throws).

`trsm_sg_left.cc` handles Side::Left at orders 1..32. Each lane is a (matrix, canonical row r) pair, so a
sub-group holds 32/N matrices for N in {4, 8, 16, 32}. Each lane carries QC right-hand sides: 4 for
q <= 4, 8 for q <= 8, otherwise 16 (complex<double> caps at 8). Lane r holds row r of the canonical Lc
and its QC values of x; step s broadcasts x[s] from lane s with `select_from_group`. No local memory,
no barrier. The 24 (side, uplo, trans, diag) cases use V1's canonical fold (`trsm_canonical.hh`). The
non-finite-reciprocal fallback to division is kept per matrix, as an xor-butterfly over the matrix's N
lanes.

Register decisions (`scripts/register_probe.sh`):

- The step loop stops at the runtime order n, since rows n..N-1 are identity padding. For complex N >= 16
  and for N = 32 with QC = 16 the loop is rolled on purpose (`#pragma unroll 1` with `break`): at cfloat
  N=32 QC=16 the count drops from 168 to 77 registers, and n32 q64 goes from 1.91 to 1.39 ms. Elsewhere
  it is fully unrolled with `continue`. No variant spills.
- The TU is built with `-pragma-unroll-threshold=262144`; without it the unrolled variants put x[] on the stack.

Window data, V1-capped time over sub-group time at batch 32768 (above 1: sub-group faster):

| float n \ q | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|
| 4 | 3.18 | 2.67 | 2.15 | 1.89 | 1.81 |
| 8 | 2.29 | 1.84 | 1.45 | 1.09 | 0.97 |
| 16 | 2.58 | 1.75 | 1.09 | 0.94 | 0.88 |
| 32 | 2.49 | 1.43 | 1.15 | 0.81 | 0.76 |

| cfloat n \ q | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|
| 4 | 2.50 | 1.54 | 1.14 | 1.07 | 0.86 |
| 8 | 1.73 | 1.27 | 1.39 | 0.96 | 0.97 |
| 12 | 1.31 | 0.87 | 0.64 | 0.73 | 0.79 |
| 16 | 1.26 | 1.08 | 1.61 | 1.79 | 1.66 |
| 32 | 1.32 | 1.36 | 1.32 | 1.37 | 1.12 |

The branch's window applied on sm_120 only (cc 0 and 89 kept V1 at every shape): float q <= 128 at n <= 4,
q <= 64 at n <= 8, q <= 32 above; cfloat q <= 64 at n <= 4, q <= 32 at n <= 8, q <= 8 for n in 9..15,
n = 16 at q <= 16 or 32 <= q <= 128, and q <= 128 for n in 17..32. Each edge is bracketed by the next
measured column; ties within 3% go to the side the neighbouring cells favour. The cfloat n=16 island
comes from V1 (V1 at cfloat n16 q64 takes 0.91 ms, against 0.31 ms at n=12 and 1.48 ms at n=24). double
and cdouble were not measured and keep V1. On the branch the blocked driver's diagonal solves used the
same kernel through `trsm_native_v1_buckets`; on main that function runs V1 only
(`src/sycl/trsm_native.cc:312`), so `blocked`'s diagonal solves do not reach `sg_left`.

Tests (`TrsmNativeSgLeft.*`) call the kernel directly with a padded ld, a batch stride and large finite
poison. Planted breaks over all of `trsm_tests`: reading A with ld = n turns 10 tests red; dropping the
store guard `c0 + j < q` turns 13 red before the saturating-batch case aborts with an illegal address;
ending the step loop one row early turns 15 red, all Left (Right-side cases stay green).

### trsm on sm_120: result

`BM_TRSM<`, BASE / new / cuBLAS in ms, batch 32768 unless given (`f_trsm.csv`):

| type | n | q | BASE | new | cuBLAS |
|---|---|---|---|---|---|
| float | 8 | 8 | 0.128 | 0.015 | 0.043 |
| float | 16 | 8 | 0.280 | 0.026 | 0.072 |
| float | 32 | 8 | 0.994 | 0.115 | 0.171 |
| float | 32 | 64 | 1.083 | 0.575 | 1.238 |
| float | 64 | 8 | 2.074 | 0.440 | 0.517 |
| float | 128 | 64 | 7.777 | 5.259 | 9.566 |
| float | 256 | 8 (b8192) | 3.012 | 1.364 | 1.415 |
| float | 512 | 8 (b2048) | 1.893 | 1.091 | 1.125 |
| float | 512 | 512 (b1024) | 7.594 | 7.603 | 12.780 |
| cfloat | 8 | 8 | 0.141 | 0.020 | 0.053 |
| cfloat | 32 | 8 | 1.551 | 0.333 | 1.391 |
| cfloat | 128 | 64 (b16384) | 8.613 | 5.921 | 140.6 |

Over the 62-cell grid (n = 8..512, q in {8, 16, 32, 64, n}): float geomean time ratio against cuBLAS
1.44 -> 0.57, cells above 1: 21 -> 0 (closest: float n=512 q=8/16 and n=256 q=8 at 0.96-0.97, where 15
trailing cuBLAS gemm launches remain); cfloat 0.18 -> 0.09, cells above 1: 5 -> 0. q = n cells at n >= 128
run V1 on uncapped rungs and did not move.

posv (`factor_bench`, native arm = potrf plus two routed trsm; vendor/native, above 1 faster), BASE ->
new: float n4 q16 0.55 -> 5.68, n32 q16 0.51 -> 1.88, n128 q16 0.77 -> 1.28; cfloat n4 q16 0.39 -> 2.53,
n32 q16 1.56 -> 4.89. All 24 cells (n = 4..128, nrhs 16 and 64) beat the vendor composition now; BASE
lost 9.

getrs `native:blocked` (the TrsmCta-bound case), BASE -> new: float n64 q16 0.71 -> 2.17, n256 q16 0.75 ->
1.41, n512 q64 3.10 -> 3.10; cfloat n64 q16 0.53 -> 1.41, n512 q16 1.26 -> 1.52. All 16 cells (n =
64..512, nrhs 16 and 64) now win; BASE lost 6. On main the sm_120 getrs rows (the sm_89 transcription)
still rank `vendor` first at nrhs >= 16.

### trsm side right on sm120

Side::Right has no sub-group kernel, so it keeps the capped V1. Even capped, float V1 loses to cuBLAS
when a solve has few rows (`BM_TRSM_OrthoRight`, `f_right.csv`, ms, native / cuBLAS):

| float | b=2048 | 4096 | 8192 | 32768 |
|---|---|---|---|---|
| n32 q8 | 0.0276 / 0.0298 | 0.047 / 0.039 | 0.070 / 0.053 | 0.305 / 0.164 |
| n20 q8 | | 0.042 / 0.031 | 0.063 / 0.040 | 0.200 / 0.099 |
| n16 q8 | | | | 0.059 / 0.072 |
| n32 q16 | | 0.047 / 0.054 | 0.070 / 0.087 | 0.342 / 0.312 |
| n32 q20 | | 0.047 / 0.075 | 0.071 / 0.124 | 0.360 / 0.497 |

On sm_120 the branch declined native for float Side::Right when order > 16, rows <= 8 and batch >= 4096.
The edges bracket measured non-losers: order 16 (native 0.059 vs 0.072), rows 16 at n=32 (native wins up
to b=16384), and batch 2048 (0.0276 vs 0.0298). cc 0 and 89 routed as before. Auto, BASE -> new, ms
(cuBLAS in brackets): n32 q8 b32768 1.041 -> 0.162 (0.163); n20 q8 b4096 0.107 -> 0.031 (0.031).

On main the window is not code. The measured `tuned/trsm.float.sm_120.txt` decides, and its `side=R trans=N`
rows (key `q` = B's rows) reproduce it: order 16 ranks `cta` first at every q; orders 20 and 24 rank
`vendor` first at q <= 8; order 32 at q = 8 ranks `cta` up to batch 2048 and `vendor` from 8192; `cta`
leads at q = 16 at every batch. The branch's window had no upper order bound; orders above 32 at q <= 8
lose to the vendor at every measured point (native / cuBLAS, ms: n48 q8 b8192 0.141 / 0.082; n64 q8 b32768
1.007 / 0.481; n128 q8 b16384 1.330 / 0.855), and the measured table ranks `vendor` first at orders 48
and 64 with q = 8 at every batch.

Not done: rows 9..16 at order 17..24, and order 32 at batch 32768, still lose 5-18% (n20 q16, n32 q16
b32768). A Side::Right sub-group kernel (each row of B is a Left solve with op(A)^T) would fix both; it
was not written.

## gemm (native register-tiled selector)

GPU 1 (sm_120), icpx 2026.0 precise, RelWithDebInfo, CUDA 13.2 cuBLAS as the vendor. Harness:
`gemm_benchmark` / `gemm_transpose_benchmark` (`--warmup=3 --min_iters=8`), one JIT pass per cell, 3
alternating reps, medians. BASE is 502ad378 (`BATCHLAS_GEMM_ROUTE=native`). Ratios are time/vendor.
Kernel choice was checked with `BATCHLAS_KERNEL_TRACE` on every quoted NN cell.

> **Note:** None of the windows in this section is code on main. `select_kernel_variant` and
> `is_sm120_family` routing are deleted, and gemm chooses from `tuned/gemm.<dtype>.<device>.txt`
> ([gemm.md](gemm.md#choices-flat-selection-p34)). The sm_120 gemm tables are the transcribed sm_89 rows
> (no gemm sweep has run), so Auto on sm_120 runs what the 4090 router chose. The tiles below are ordinary
> candidates (`wide:m=16:n=16:k=16`, `wide:m=32:n=32:k=16`, NN only; `wide:m=64:n=64:k=16`,
> `wide:m=128:n=32:k=16`, `wide:m=32:n=128:k=16` for the transposed fallback;
> `src/ops/gemm/choice.hh:80-86`). The measurements are hypotheses for an sm_120 `tools/tune` gemm sweep.
> The old kernel names (`16x16x16wide`, `32x32x16wide`, ...) are not pin words; they throw.

The branch's windows were keyed on `is_sm120_family(cuda_cc)` in `select_kernel_variant`
(`src/sycl/gemm_kernels.cc`); cc 0 and 89 kept the 4090 ladder. Vendor-build factorizations were
unchanged (potrf, getrf and geqrf in float, cfloat and cdouble at n = 128-512 measured new/base
0.996-1.003).

### gemm small tiles

Two NN instantiations of the wide-scalar template (`launch_wide_transposed`, 64 threads each): 16x16 with a
2x2 thread tile, and 32x32 with a 4x4 thread tile (`Tiled16x16RegisterK16Wide`, `Tiled32x32RegisterK16Wide`
on the branch). On main they run only under the spellings above. On the 4090 the only complex register
kernel was the 64x64 tile, which at m = n = 32 computes 3/4 padding.

Windows (float and complex<float>; `fits16` = max(m,n) <= 16, or min(m,n) <= 8 with max(m,n) <= 32):

- 16x16 tile: `fits16`, at least 1024 tiles in flight, and k >= 32 (float) or max(m,n,k) > 8 (cfloat).
  Float needs 2048 tiles from k = 768 on.
- 32x32 tile: max(m,n) <= 32 and not `fits16`, at least 1024 tiles, and k > 32 (float). Float needs 1536
  tiles from k = 512 on.
- float, 32 < min(m,n), max(m,n) < 128, max(m,n,k) > 56: the 64x64 wide tile when min(m,n) <= 64, otherwise
  128x128x8.
- cfloat 64x64 gate: min(m,n) >= 32 instead of min(m,n,k) >= 32. For k >= 32 this keeps the 4090's 64 CTAs;
  for k < 32 it needs m*n*batch >= 2^19 (gate table below).

Final A/B, NN, time/vendor (BASE -> new), batch 32768 unless given:

| cell | BASE | new | | cell | BASE | new |
|---|---|---|---|---|---|---|
| f 8x8x32 | 1.10 | 0.38 | | c 8x8x256 | 1.19 | 0.58 |
| f 8x8x1024 | 2.11 | 1.10 | | c 16x16x16 | 0.54 | 0.23 |
| f 16x16x32 | 0.99 | 0.51 | | c 16x16x1024 b8192 | 1.38 | 0.79 |
| f 24x24x128 | 2.71 | 1.02 | | c 32x32x16 | 1.64 | 0.72 |
| f 32x32x256 | 1.20 | 0.92 | | c 32x32x1024 b4096 | 2.04 | 1.07 |
| f 40x40x128 | 3.22 | 1.02 | | c 64x64x16 | 2.74 | 0.96 |
| f 64x64x16 | 1.69 | 0.93 | | c 128x128x8 b16384 | 1.57 | 0.99 |
| f 100x100x64 b16384 | 3.21 | 1.28 | | c 1024x1024x16 b256 | 2.47 | 1.00 |

Controls, unchanged (BASE = new): f 8x8x16, 16x16x16 and 32x32x32 (SmallBatched), 48^3, 128x128x32,
128x128x1024 and 256^3; c 8^3 (Direct), 64x64x64 and 256^3 (wide 64x64). With `BATCHLAS_BENCH_LD_PAD=1`:
f 64x64x16 1.62 -> 0.96, f 32x32x256 1.82 -> 0.90, c 16x16x64 1.34 -> 0.60.

Brackets (forced-kernel screens, vendor ratios):

- float 16 tile, k edge: 16x16x16 SmallBatched 0.27 vs 16x16 0.30 (kept); 16x16x32 1.01 vs 0.51. 8x8x16:
  SmallBatched 0.25, Direct 0.22, 16x16 0.25 (kept). At 8x8x32 Direct is 0.36 and the 16 tile 0.39; from
  k = 64 the tile wins (8x8x64 0.57 vs Direct 0.83).
- float 32 tile, k edge: 32x32x32 SmallBatched 0.85 vs 0.89; 32x32x64 1.10 vs 0.93.
- float 33..56 cubes stay on SmallBatched (33^3 0.80 vs 64-wide 1.04; 56^3 0.94 vs 0.95). 40x40x128 goes
  to the 64 tile (1.02 vs 3.22).
- float min(m,n) > 64: 65x65x64 128x128 1.18 vs 64-wide 1.22; 100x100x256 1.06 vs 1.48; 64x96x128:
  64-wide 0.84 vs 128x128 1.02.
- cfloat 16 vs 32 tile at 16x16: 16x16x64 0.51 vs 0.87. At 32x32 (not fits16): 32x32x1024 16 tile 1.38 vs
  32 tile 1.07.

#### Tiles in flight at deep k

One 64-thread group per tile walks k serially, and at 1024 tiles the older kernels (Tiled16 at 16x16,
Tiled32x32Register at 32x32) are not yet wave-limited. Forced-tile screens, GPU 1, beta = 1, 3 reps;
vendor-relative at `LD_PAD=0` / `LD_PAD=1`:

| cell | tiles | BASE | tile | verdict |
|---|---|---|---|---|
| f 16x16x512 | 1024 | 1.64 / 1.45 | 1.43 / 1.41 | tile |
| f 16x16x768 | 1024 | 1.68 / 1.50 | 1.44 / 1.43 | BASE |
| f 16x16x768 | 1536 | 1.08 / 1.01 | 1.11 / 1.11 | BASE |
| f 16x16x1024 | 1024 | 1.61 / 1.21 | 1.71 / 1.77 | BASE |
| f 16x16x1024 | 2048 | 1.11 / 0.98 | 0.92 / 0.95 | tile |
| f 32x32x512 | 1024 | 1.23 / 1.45 | 1.76 / 1.79 | BASE |
| f 32x32x512 | 1536 | 1.45 / 1.71 | 1.31 / 1.32 | tile |
| f 32x32x1024 | 1024 | 1.29 / 1.47 | 1.76 / 1.77 | BASE |
| f 32x32x1024 | 1536 | 1.43 / 1.54 | 1.29 / 1.26 | tile |
| f 15x15x1024 | 1024 | 1.49 / 1.48 | 1.63 / 2.09 | BASE |
| f 15x15x1024 | 2048 | 1.06 / 1.06 | 0.94 / 0.96 | tile |

The tile's time barely moves from 1024 to 2048 tiles (16x16x1024: 0.176 -> 0.265 ms) while BASE doubles, so
the crossover is a tile count that grows with k. The floor is now 2048 tiles for the float 16 tile from
k = 768, and 1536 for the float 32 tile from k = 512. cfloat keeps 1024: at 1024 tiles and k = 1024 it wins
(16x16 1.47 -> 1.02; 32x32 1.72 -> 1.14). After the fix, 16x16x1024 b1024 / b2048 and 32x32x512 b1024 /
b1536 match the better of BASE and the tile (kernel checked by trace).

The rule forfeits gains on shapes that fill the tile badly, where the older kernel is slow: at 1024 tiles
the tile wins f 8x8x1024 (2.93 vs 2.33), 24x24x512 (2.30 vs 1.11) and 32x8x1024 (2.56 vs 1.76), which now
stay on BASE until 1536 or 2048 tiles. No single fill threshold separates the measured cells: 15x15 (fill
0.88) loses and 31x31 (fill 0.94 of Tiled16) wins.

Rejected: k-split (2-4 slices of k with an SLM reduction) was never better than the unsplit 16/32 tiles
(float 32x32x1024 ks2 0.91 vs 0.98 was the only win; cfloat 32x32x* ks2 ran 1.07-1.31). A 128x64 tile and a
64x64 tile with K = 32 for large cfloat were 1.28-1.55 vs 1.11-1.17 for the 64x64 K = 16 tile.

cfloat k < 32 gate (launch-bound cells, 10-20 us): the 64x64 wide tile wins from m*n*batch >= 2^19 (64x64x8 b128 1.16 -> 1.09; 48x48x16 b256 1.63 -> 1.02) and loses below it (33x33x8 b256 0.98 vs 1.01-1.03; 128x128x8 b16 1.02 vs 1.11; 256x256x16 b4 is a win given up). A CTA count does not separate these cells (33x33x8 b256 loses at 256 CTAs, 64x64x8 b256 wins); the output element count does.

### gemm complex double

complex<double> NN on sm_120 goes to the 16x16 tile whenever max(m,n,k) > 8. At FP64's 1/64 rate the kernel
is compute-bound, and the tile that puts the most groups in flight won or tied at every shape and batch
measured: 16x16 >= 32x32 >= 64x64, and all three beat Tiled16 and Direct. The exception is 8^3, where
Direct (0.22) beats the 16 tile (0.38). BASE -> new, vendor-relative: 8x8x256 b16384 1.54 -> 0.44; 16x16x16
b32768 1.54 -> 0.41; 32x32x256 b8192 3.45 -> 0.87; 64x64x16 b16384 3.21 -> 0.82; 100x100x16 b8192 2.44 ->
0.71; 64x64x64 b16384 0.89 -> 0.86; 256x256x256 b512 0.89 -> 0.89 (min_dim >= 256 keeps the 64 tile);
1024x1024x16 b64 2.83 -> 0.83.

Brackets: 64x64x16 b16384 on the 16, 32 and 64 tiles is 0.82, 0.81 and 0.90; 100x100x16 is 0.71, 0.90 and
0.90. 16x16x64 b512 on the 64 tile is 4.77, so the tile must not grow with the shape. 8^3 b64: Direct 0.46,
16 tile 0.60. Real double was screened too: the 16 tile gains 5-25% (16x16x16 0.30 -> 0.22), but double
already beats cuBLAS 1.1-4x there, so it is left alone.

### gemm transposed fallback

float and complex<float> one-transposed forms (TN/NT; complex only with ConjTrans) that would fall to Tiled16
or Direct take a wide transposed tile when min(m,n) >= 32 and there are at least 128 64x64 CTAs. The tile is
64x64, except for float panel shapes: 128x32 NC when n < 64 and m >= 128, and 32x128 CN when m < 64 and
n >= 128. Unchanged: the float K32 family (m >= 128, n >= 32, k >= 128) and float SmallBatched (max <= 32).
On the 4090 the real-scalar tile tied Tiled16 ([gemm.md](gemm.md#wide-scalar-transposed-tiles)); here
Tiled16 is 2-5x behind cuBLAS on these forms.

Final A/B, `LD_PAD=1`, beta=1, BASE -> new, vendor-relative: f NT 256x256x32 b2048 3.53 -> 1.49; f NT
512x512x64 b512 5.72 -> 1.64; f NT 64x64x16 b16384 1.73 -> 0.98; f TN 48x48x48 b16384 2.27 -> 1.08; c NC
64x64x64 b8192 2.96 -> 1.02; c CN 64x64x64 b8192 2.50 -> 0.86; c CN 32x32x256 b8192 2.26 -> 1.83.

Brackets. The 128-CTA floor counts the CTAs of the tile actually launched. f NT 256x32x96 pad 0 on Tiled16
vs 128x32 NC: b32 (64 panel CTAs) 1.51 vs 1.85; b64 (128) 2.17 vs 1.81; b16 stays on Tiled16. f NT 64x64x64
b64 (64 CTAs) is 1.34 on both; b128 is 1.66 vs 1.48. 128x128x32 b16 is 1.04 on Tiled16 vs 1.38 on the wide
tile, and b64 is 1.65 vs 1.39. The min(m,n) >= 32 floor: f NT 16x16x256 is 0.62 on Tiled16 vs 1.39; c CN
16x16x256 is 0.93 vs 2.48. The panel-shaped tile: f NT 256x32x96 is 1.69 on 128x32 vs 2.07 on 64x64.

Still behind cuBLAS: float NT/TN at a strided ld (1.2-1.7; NT 256x40x64 1.78 and 256x32x96 1.51, both from
4.5-5x on BASE); panel cells near the 128-CTA floor (256x32x96 b64 1.81, where the vendor is launch-bound at
14 us); the float K32 NT kernel at `LD_PAD=1` (256x32x256 NT 2.02); and cfloat 32x32 transposed, which has no
32x32 transposed tile (1.45-1.91).

### gemm result

Tests are in `tests/gemm_tests.cc`: `SmallWideNNTilesMatchTiled16` (ragged m, n, k from 1 to 257, sub-view
ld, both betas), `SelectorChosenSmallTilesMatchTiled16` (auto-selected at batch 1024),
`SmallWideSaturatingBatchIsBitIdentical` (batch 2048), `Sm120TransposedFallbackMatchesTiled16`, and the two
selector straddle tests.

Armed breaks: dropping the epilogue's `col >= n` guard turns exactly the shapes with a partial n tile red.
Staging k one past its end turns exactly the k % 16 != 0 shapes red. Moving the transposed fallback's floor
to `ctas > 128` and the float 16-tile deep-k edge to `k >= 769` (one build) turns red only the six
exact-edge assertions of the two straddle tests.

Still losing in the NN native walk: float 8x8x1024 1.10; float 100x100x64 1.28 (128x128 at 3/4 fill); float
65x65x8 1.29 (BASE 1.48); float 64x100x512 1.22 (BASE 1.59); the deep-k float small cells below 1536/2048
tiles (1.2-2.9); cfloat 32x32x1024 1.07; cfloat 256^3 1.13; and the other large cfloat squares (1.11-1.17,
not addressed; the large complex tile).
