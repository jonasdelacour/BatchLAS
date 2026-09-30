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
| cfloat 256 b8 | 13.17 | 12.03 | **11.47** | - | 11.41 | - | 11.41 | 11.41 |
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
flops, so the grid barrier pays for itself earlier: sm_120 uses 256 for complex, bracketed
below by 192 (4.8% slower at cfloat 256 b8), and 320 for real, bracketed by 256 (4% slower
at float 256 b32) and 448. The 4090 gate of 768 costs up to 2.1x at float 768 b32 here.
The double types follow their float sibling without a measurement.

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

Unchanged within 0.2% (the new code does not reach them): float 64/128/256 and cfloat
64/128/256/512 at their campaign batches. Auto on cfloat 1024 b64 is the vendor both
before and after (181.1 ms). `sytrd_blocked_benchmark` (the public sytrd op, which also
takes the latrd threshold): cfloat 1024 b64 nb48 256.0 -> 232.7, cfloat 512 b32 43.1 ->
30.6, float 512 b16 14.8 -> 12.8; float 512 b512, float 1024 b128 and cfloat 512 b256
unchanged. Routes checked with `BATCHLAS_COVERAGE_OUT`: float 512 native is `two_stage`
on BASE and `blocked` now.
