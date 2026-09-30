# Blackwell (sm_120) retune

Machine: threadripper02, 4x NVIDIA RTX PRO 6000 Blackwell Max-Q (CC 12.0, 188 SMs,
128 MB L2, ~1.8 TB/s GDDR7, 300 W). cuBLAS reaches 43.9-45.8 TFLOP/s SGEMM on this
part (power-limited: ~43% of the ~106 TFLOP/s FP32 peak). CUDA 13.2 (HPC SDK 26.5).

Starting point: benchviz saturation campaign `cuda-20260930-112900` (main @ 4e0a9b7d,
float + cfloat). Worst batchlas/vendor time ratios: cfloat gemm 3.0x geomean (15x at
32x32x1024), trsm float 1.5x (11x at n=32 q=8), getrs cfloat 1.8x (10x at nrhs=16),
potrf cfloat 1.6x (9.5x at n=256), geqrf skinny 2-3.7x.

## icpx fast fp-model

The campaign build used oneAPI icpx 2026.0, whose default is `-fp-model=fast`. Device
functions then carry a preserve-sign denormal mode, while the libspirv builtins linked
through `-mlink-builtin-bitcode` are IEEE. LLVM does not inline across that attribute
mismatch, so `__spirv_ocl_fma` (every `sycl::fma`), `__spirv_BuiltIn*Id` and
`__spirv_ControlBarrier` become out-of-line `.func` calls, and the fma stub itself runs
as `fma.rn.ftz.f32`.

- SASS of `GemmRegister64x64K16WideKernel<complex<float>,true>`: 1031 `CALL.REL` (1024 to
  a one-FFMA stub) under icpx default; 0 calls and 1152 inline FFMA under dpcpp-cuda
  (which defaults to precise) or icpx `-ffp-model=precise`.
- 4472 of 4539 kernels in the icpx images contained calls.
- Same-compiler probe, only `-ffp-model=precise` toggled: cfloat wide kernel 128^3
  b8192 29.0 -> 3.75 ms, 512^3 b512 120.8 -> 13.56 ms.
- Whole-library A/B (batchlas/vendor): cfloat gemm 128^3 6.54 -> 0.98, 32x32x1024
  15.3 -> 2.03; float gemm 1024^3 1.55 -> 1.07; potrf cfloat lpanel n=256 13.87 ->
  2.49 ms; getrs cfloat n=32 nrhs=16 3.6x; trsm float 2x.

The 4090 numbers in this directory were all taken with self-built dpcpp-cuda, which is
why this never appeared there. Fix: `-ffp-model=precise` for IntelLLVM in
`cmake/BatchLASOptions.cmake`. `-fdenormal-fp-math=ieee` alone also restores inlining
but leaves reassoc/afn on, which section 11 of AGENTS.md rules out.

## syev retune

Work package SYEV-1/2/3/5. Machine as above, GPU 1 (headless, one measuring process
under a lock), build: icpx 2026.0 `-ffp-model=precise`, RelWithDebInfo, sm_120. BASE is
branch `worktree-blackwell-tuning` @ 43b9f806. `syev_benchmark` avg ms, eigenvectors,
Lower, `--warmup=3 --min_iters=8`, alternating A/B order, medians of 2-4 reps (per-rep
spread under 1% unless noted). Every per-architecture value lives in
`src/extensions/syev_arch_tuning.hh` or `syev.hh` and applies only to
`is_sm120_family(cuda_cc)`; cc 0 and 89 keep the 4090 values.

### syev float blocked-two-stage edge

`syev_saturated_algorithm_for_n<float>` sent 449 <= n <= 1024 to two_stage (4090:
two_stage 1.01x at 512, 1.15x at 640). On sm_120, with the panel and back-transform
constants below, blocked wins through n = 768 at every batch, and two_stage wins at
saturation from 896:

| n | batch | blocked | two_stage | per-matrix winner |
|---|---|---|---|---|
| 768 | 256 | 183.3 | 235.4 | blocked 1.28x |
| 768 | 512 | 387.1 | 424.2 | blocked 1.10x |
| 896 | 256 | 315.6 | 347.8 | blocked 1.10x |
| 896 | 512 | 649.4 | 635.5 | two_stage 1.02x |
| 1024 | 128 | 236.3 | 282.7 | blocked 1.20x |
| 1024 | 256 | 494.9 | 485.8 | two_stage 1.02x |
| 1024 | 512 | 990.8 | 893.5 | two_stage 1.11x |

Blocked is saturated by batch 128 at n = 1024 (1.85 / 1.93 / 1.94 ms per matrix at
128 / 256 / 512); two_stage still gains from batch (2.21 / 1.90 / 1.75), so the
saturated comparison favours two_stage from 896 and the edge is n = 768
(`kSyevSm120FloatBlockedMaxN`). BASE constants gave the same picture (blocked 1.43x at
512, 1.19x at 640, 1.17x/1.06x at 768 b256/b512; two_stage 1.05x at 896 b512).

The n = 1024 b = 128 campaign cell stays on two_stage (282.7 ms vs vendor 236.2): the
chase kernel runs one work-group per matrix (SYEV-4, not addressed), so two_stage is
below saturation at batch 128. Blocked would tie the vendor there, but the window is
keyed on n alone.

### syev panel and back-transform blocks

`BATCHLAS_TUNE_SYTRD_BLOCK_SIZE` / `BATCHLAS_TUNE_ORMQR_BLOCK_SIZE` sweeps on BASE,
native:blocked unless noted, one knob at a time (the other at its default):

| cell | nb 8 | 16 | 24 | 32 | 48 (default) |
|---|---|---|---|---|---|
| cfloat 1024 b64 | 334.9 | **323.5** | 332.1 | 340.1 | 366.8 |
| float 1024 b128 | 243.4 | **240.9** | 249.6 | 255.0 | 274.6 |
| float 768 b256 | 191.3 | **188.4** | 191.4 | 194.7 | 204.6 |
| float 640 b512 | 221.7 | **215.2** | 217.0 | 219.6 | 227.6 |
| cfloat 768 b128 | 250.4 | **236.1** | 237.7 | 243.2 | 259.5 |

| cell | ormqr 24 | 32 | 40 | 56 (default) |
|---|---|---|---|---|
| cfloat 1024 b64 | **322.2** | 326.0 | 339.5 | 365.4 |
| float 1024 b128 | 273.7 | **269.9** | 270.2 | 273.8 |
| float 768 b256 | 204.1 | **200.2** | 201.7 | 204.7 |
| float 640 b512 | 229.5 | **223.6** | 225.5 | 227.3 |
| cfloat 768 b128 | 235.0 | **232.7** | 241.4 | 259.1 |
| float 1024 b128 two_stage | 287.6 | **282.6** | - | 287.1 |

So above n = 512, sm_120 uses nb = 16 (`kSm120SytrdBlockXlarge`) and a WY block of 32
(`kSm120OrmqrBlockXlarge`). The only other consumer of `SYTRD_BLOCK_SIZE_XLARGE` is
syev_blocked itself (the public `sytrd_blocked` defaults to `SYTRD_BLOCK_SIZE_MEDIUM`).
`ORMQR_BLOCK_SIZE_XLARGE` also feeds the ormqr op, gesvd_blocked and syevx; rather than
re-measure those, syev reads its own copy and the shared constant is unchanged (the
campaign has no ormqr/orgqr cell above n = 512, and orgqr_blocked never read it).
Not changed: the complex 256 < n <= 512 bucket (32; the diagnosis saw 1.03x for 16, at
the noise edge).

### latrd grid min-n

The grid latrd path only runs when `MAX_COMPUTE_UNITS / batch >= 2` (batch <= 94
here), so this matters for small and moderate batches. `BATCHLAS_LATRD_GRID_MIN_N`
sweep, native:blocked on the new constants, 3 reps:

| cell | 128 | 192 | 256 | 320 | 384 | 448 | 512 | 768 |
|---|---|---|---|---|---|---|---|---|
| cfloat 256 b8 | 13.17 | 12.03 | 11.47 | - | 11.41 | - | 11.41 | 11.41 |
| cfloat 384 b16 | 38.55 | 37.68 | **37.06** | - | - | - | - | - |
| cfloat 512 b32 | 59.54 | 58.75 | **58.07** | - | 60.67 | - | 68.07 | 70.40 |
| cfloat 1024 b64 | 265.3 | 264.7 | **264.0** | - | 266.8 | - | 270.3 | 283.4 |
| float 256 b32 | - | - | 6.41 | - | **6.16** | - | 6.16 | 6.16 |
| float 384 b8 | - | - | 15.25 | **14.73** | 14.93 | 14.97 | 14.97 | 15.00 |
| float 448 b16 | - | - | - | **19.01** | 19.24 | 19.93 | - | - |
| float 512 b16 | - | - | 24.23 | **23.40** | 23.42 | 24.14 | 25.20 | 25.34 |
| float 768 b32 | - | - | 46.74 | **46.11** | 46.24 | 46.90 | 47.88 | 62.90 |
| float 1024 b64 | - | - | 120.0 | **119.3** | 119.4 | 119.8 | 120.8 | 126.8 |

(Columns 256/384/512/768 and 128/192/256 and 320/384/448 come from three separate
sweeps; overlapping cells agree within 0.5%.) A complex column carries four times the
flops, so the grid barrier pays for itself earlier. The 4090 gate of 768 costs up to
2.1x at float 768 b32 here.

#### latrd grid min-n: per-type brackets

A second pass (review follow-up) re-bracketed every type under Auto, which is what ships
at these small batches, 4 alternating reps after a discarded pass, spread under 0.3%.
It found one systematic effect: a grid panel of *exactly* the threshold n loses to the
legacy kernel (cfloat 256 b8 with min-n 256: 11.48 ms against 11.41 ms at 257, 288 or
BASE, in every rep), while the next panel size up already wins. So the constant is the
largest n that stays legacy, and the grid path starts one above it
(`kSm120LatrdLegacyMaxN`, grid for n > 320 / 256 / 256 / 128). Values below are ms,
the winner in bold; min-n = X means the grid runs for panels with n >= X:

| cell | 256 | 257 | 288 | 320 | 384 | BASE (768) |
|---|---|---|---|---|---|---|
| cfloat 256 b8 | 11.48 | **11.41** | **11.41** | 11.41 | 11.41 | 11.41 |
| cfloat 288 b8 | 24.24 | **24.10** | **24.12** | 24.61 | 24.63 | 24.62 |
| cfloat 320 b16 | 28.03 | **27.91** | **27.90** | 28.46 | 29.43 | 29.42 |
| cfloat 384 b16 | 37.08 | **36.90** | 36.92 | 37.41 | 39.48 | 40.78 |
| cfloat 512 b32 | 58.06 | **57.92** | **57.89** | 58.49 | 60.70 | 70.50 |

| cell | 96 | 128 | 129 | 160 | 192 | 256 | BASE (768) |
|---|---|---|---|---|---|---|---|
| cdouble 128 b8 | 15.17 | 14.86 | **14.77** | 14.78 | 14.77 | 14.77 | 14.77 |
| cdouble 160 b8 | 25.53 | 25.27 | **25.14** | 25.40 | 25.68 | 25.68 | 25.66 |
| cdouble 192 b8 | 32.91 | 32.64 | **32.53** | 32.79 | 33.50 | 33.98 | 33.95 |
| cdouble 224 b8 | 40.16 | **39.91** | - | 40.04 | 40.76 | 42.62 | 42.62 |
| cdouble 256 b8 | 51.67 | 51.24 | **51.13** | 51.61 | 52.32 | 55.19 | 56.25 |
| cdouble 192 b32 | 36.28 | **36.02** | - | 36.18 | 36.88 | 37.33 | 37.34 |

| cell | 160 | 192 | 224 | 256 | 257 | 320 | 321 | BASE (768) |
|---|---|---|---|---|---|---|---|---|
| double 192 b16 | 12.68 | 12.48 | **12.38** | 12.39 | - | **12.38** | - | 12.41 |
| double 224 b16 | 16.28 | 16.07 | 15.91 | **15.82** | - | **15.82** | - | 15.82 |
| double 256 b16 | 20.57 | 20.35 | 20.17 | 19.99 | **19.91** | **19.91** | - | 19.91 |
| double 288 b16 | 29.86 | 29.57 | 29.31 | 29.05 | **28.99** | 29.28 | - | 29.27 |
| double 320 b16 | 34.39 | 34.11 | 33.85 | 33.55 | **33.50** | 34.21 | - | 34.37 |
| double 384 b16 | 47.14 | 46.85 | 46.57 | 46.29 | **46.21** | 46.91 | - | 48.47 |
| double 256 b64 | 23.78 | 23.54 | 23.35 | 23.18 | **23.10** | 23.10 | - | 23.10 |
| float 320 b16 | - | - | - | - | - | 10.71 | **10.69** | - |
| float 384 b8 | - | - | - | 15.25 | - | 14.73 | **14.70** | 15.00 |
| float 448 b16 | - | - | - | - | - | 19.02 | **19.01** | - |
| float 512 b16 | - | - | - | 24.23 | - | 23.46 | **23.39** | 25.34 |

So: float 321 (256 is 4% slower at float 256 b32, 384 and 448 lose at 384/448), double
257 (224 and 320 lose on both sides), complex<float> 257 (256 loses at 256; 320 loses 2%
at 288 and 320; 257 and 288 tie within 0.1%), complex<double> 129 (128 loses at 128; 96
and 160 lose on both sides). Complex<double> goes lowest because a column carries four
times the flops of a real one at a 1/64-rate FP64 unit. The first pass shipped
320 / 320 / 256 / 256; the double values there were copied from the float siblings. The
review re-measured those on the first-pass build and every shipped Auto cell improved
over BASE (double 384 b16 48.46 -> 46.92, double 448 b32 71.53 -> 67.93, cdouble 256 b8
56.24 -> 55.19), and the per-type values above improve on that again (46.21 and 51.13 ms).

Final A/B, three arms in one alternating run, 4 reps after a discarded pass: BASE (the
BASE build's libraries), first pass (min-n 320/320/256/256), and the per-type values. ms:

| cell | route | BASE | first pass | per-type | vendor |
|---|---|---|---|---|---|
| cfloat 256 b8 | Auto | 11.41 | 11.48 | **11.41** | 5.95 |
| cfloat 288 b8 | Auto | 24.60 | 24.26 | **24.09** | 9.10 |
| cfloat 384 b16 | Auto | 40.77 | 37.06 | **36.90** | 11.05 |
| cfloat 512 b32 | Auto | 70.53 | 58.08 | **57.93** | 19.11 |
| cdouble 160 b8 | Auto | 25.67 | 25.67 | **25.14** | 8.54 |
| cdouble 256 b8 | Auto | 56.23 | 55.17 | **51.11** | 19.46 |
| double 256 b16 | Auto | 19.91 | 19.91 | 19.91 | 12.58 |
| double 384 b16 | Auto | 48.45 | 46.91 | **46.22** | 24.14 |
| double 448 b32 | Auto | 71.53 | 67.93 | **67.13** | 42.65 |
| float 384 b32 | Auto | 15.49 | 15.20 | **15.17** | 9.74 |
| float 512 b16 | Auto | 41.91 | 23.40 | **23.31** | 13.20 |
| float 768 b32 | native | 96.28 | 46.12 | 46.02 | 32.89 |
| float 512 b512 | native | 145.3 | 102.5 | 102.5 | 189.5 |
| float 768 b256 | native | 234.2 | 181.5 | 181.5 | 242.6 |
| float 1024 b128 | native | 286.4 | 281.0 | 281.0 | 232.7 |
| cfloat 1024 b64 | native | 365.7 | 264.5 | 264.4 | 181.7 |

The saturated cells do not reach the grid path (batch > 94), and they are unchanged
between the two passes, as expected.

All of these small-batch Auto cells are native and all still lose to the vendor by
1.5-3x (vendor: cdouble 256 b8 19.5, cdouble 192 b8 10.7, double 384 b16 24.2, double
256 b16 12.6, float 384 b32 9.7, cfloat 256 b8 6.0 ms). The native/vendor window is
keyed on n alone and was chosen at saturation, so these cells are not addressed here.

Not changed: the grid residency cap (`resident_cap = MAX_COMPUTE_UNITS`, one work-group
per SM). At batch 128 on 188 SMs it gives G = 1. Raising it needs a blocks-per-SM
argument for the spinning grid barrier plus a launch-at-ceiling test, and
`BATCHLAS_LATRD_GRID_FORCE_UNSAFE` is compiled out, so the gain could not be measured.

### syev retune: result

BASE vs new, benchviz's batchlas arm (`BATCHLAS_SYEV_ROUTE=native`), campaign batches
plus three small-batch cells, 4 alternating reps (3 for the small-batch cells), ms:

| cell | BASE | new | speed-up | vendor | new / vendor |
|---|---|---|---|---|---|
| float 512 b512 | 148.1 (two_stage) | 103.9 (blocked) | 1.43x | 193.0 | 0.54 |
| float 640 b512 | 269.6 | 211.0 | 1.28x | 356.5 | 0.59 |
| float 768 b256 | 238.9 | 183.0 | 1.31x | 246.8 | 0.74 |
| float 1024 b128 | 288.7 | 283.3 | 1.02x | 236.2 | 1.20 |
| float 1024 b256 | 487.3 | 488.2 | 1.00x | 503.7 | 0.97 |
| cfloat 768 b128 | 258.8 | 211.5 | 1.22x | 175.0 | 1.21 |
| cfloat 1024 b64 | 365.5 | 264.2 | 1.38x | 183.4 | 1.44 |
| float 512 b16 | 42.0 | 23.5 | 1.79x | 13.2 | 1.78 |
| float 768 b32 | 96.3 | 46.1 | 2.09x | 32.9 | 1.40 |
| cfloat 512 b32 | 70.6 | 58.1 | 1.21x | 19.2 | 3.03 |

Not changed by this package:

* benchviz's `sytrd` stage (`benchmarks/benchviz/ops.py`, `sytrd_nb`) still mirrors the
  sm_89 nb = 48 for n > 512. On sm_120 it therefore benchmarks a panel width that syev no
  longer uses; benchviz has no architecture input to key it on.
* Small-batch Auto cells where the vendor is 1.5-3x faster (see the per-type brackets
  above): the native/vendor window is keyed on n alone.

Unchanged within 0.2% (the new code does not reach them): float 64/128/256 and cfloat
64/128/256/512 at their campaign batches. Auto on cfloat 1024 b64 is the vendor both
before and after (181.1 ms). `sytrd_blocked_benchmark` (the public sytrd op, which also
takes the latrd threshold): cfloat 1024 b64 nb48 256.0 -> 232.7, cfloat 512 b32 43.1 ->
30.6, float 512 b16 14.8 -> 12.8; float 512 b512, float 1024 b128 and cfloat 512 b256
unchanged. Routes checked with `BATCHLAS_COVERAGE_OUT`: float 512 native is `two_stage`
on BASE and `blocked` now.
