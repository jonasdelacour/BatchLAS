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

## Device-call guard

Machine: threadripper02, RTX PRO 6000 Blackwell Max-Q (sm_120), icpx 2026.0, CUDA 13.2.
The `-ffp-model=precise` fix above has no functional symptom when it regresses: every
test stays green and only the timings move. `scripts/check_device_calls.py` (ctest
`device_calls_tests`, label `util`; also `BATCHLAS_BUILD_DIR=build
.github/ci/run_local_checks.sh`) guards it. It walks each `build/src/libbatchlas_*.so`
for embedded CUDA fatbins (magic `0xBA55ED50`), runs `cuobjdump -ptx` on each, and
fails if any `.entry` makes a `call` to a `__spirv_*`/`__clc_*` function, directly or
through the local `.func` helpers it calls (followed transitively; body-less
prototypes are skipped, bodies are found by brace matching), or if
one of the representative kernels (`GemmRegister64x64K16WideKernel<complex<float>,true>`,
`PotrfLpanelKernel`, `TrsmCtaKernel`) is missing. It exits 77 (ctest skip) when
there is no NVPTX image or no `cuobjdump`.

It reads PTX, not SASS: in a correct precise build, `TrsmCtaKernel` still has 9-33
SASS `CALL.REL`, which are ptxas's own IEEE division/sqrt slow-path subroutines. A
SASS `CALL.REL` count cannot tell those from a libspirv stub. A PTX `call` names its
callee.

Proof that it can fail: the same three TUs (`trsm_native.cc`, `potrf_lpanel.cc`,
`gemm_kernels.cc`) compiled with BASE's `flags.make` and device-linked into a scratch
library, with and without `-ffp-model=fast` appended (the last flag wins):

| build | kernels | with builtin calls | wide cfloat | potrf lpanel | trsm cta | exit |
|---|---|---|---|---|---|---|
| `-ffp-model=fast` | 251 | 248 | 1031 calls | 952 | 8548 | 1 |
| as built (precise) | 492 | 0 | 0 | 0 | 0 | 0 |
| BASE build, 14 libraries | 6277 | 0 | 0 | 0 | 0 | 0 |

208 BASE kernels do call something: device functions LLVM chose not to inline
(`larfg`, `sec_solve_roc`, gesvdj and syrk/trmm tile lambdas). Those are reported but
do not fail the check, and following them transitively finds no builtin call inside
them either. Transitive counts on the two real full-library builds (the rows above
count direct call sites only):

| build | kernels | with builtin calls | wide cfloat | potrf lpanel | trsm cta | exit |
|---|---|---|---|---|---|---|
| BASE (precise), 14 libraries | 6277 | 0 | 0 | 0 | 0 | 0 |
| campaign build `~/BatchLAS/build` (icpx default fast) | 4539 | 4472 (direct only: 4452) | 1031 | 952 | 9576 (direct: 8548) | 1 |

Scanning 158 images takes 4-7 s.

### `__builtin_fma` in `device_scalar.hh`: not adopted

`fma_acc` and the complex helpers in `src/sycl/device_scalar.hh` could call
`__builtin_fma{,f}` instead of `sycl::fma`. That spelling inlines even under the fast
model: in the fast scratch build it cut the wide cfloat kernel from 1031 calls to 7,
lpanel from 952 to 52 and trsm cta from 8548 to 672. The barrier and id calls remain,
so the guard stays red there, as it should. Under precise it is **not** a no-op,
though. SASS of all 14 libraries against BASE: 158 of 6277 kernels differ, in
`extensions_cta` (126), `sycl` (16) and `extensions_factorization` (16). Examples are
GetrsFusedN<float,1> (1104 -> 896 instructions, LDG 40 -> 22), GesvTiny<cdouble>,
GetrfTiny<cfloat>, Geqrf/GetrfPanel* and complex TrsmCta n=32. The differences are
in unrolling and scheduling (`llvm.fma` is visible to the cost models;
the libspirv/NVVM fma is not). No kernel's FMA fusion changed that we could see, but
neither bit-identity nor timing was measured for those 158 kernels. Since the flag
is the real fix and the guard now enforces it, the switch was left out. The
tiny-probe result (identical SASS for one loop, `tmp/gemm/probe`) does not carry
over to the library.

## Compute-capability key

Machine: threadripper02, 4x RTX PRO 6000 Blackwell Max-Q (sm_120), CUDA 13.2.
Per-architecture windows key on `Device::cuda_compute_capability()` (`major*10+minor`
parsed from the CUDA device's SYCL version string, memoized per device) through
`dispatch::is_sm120_family(cc)` (120 <= cc < 130). A value of 0 means "not a CUDA
device", and 0 and 89 keep the sm_89 windows. `tests/util_device_queue_tests.cc`
checks it against an independent oracle: every NVIDIA SYCL GPU is paired with its
CUDA ordinal (the device name must match) and its value must equal the CUDA runtime's
`prop.major*10+prop.minor`. CPU and non-NVIDIA devices must return 0. The family window
is tested on both sides of each edge (119/120, 129/130) and at 0, 89 and 100.
Deliberate break (non-CUDA default 89, `cc = major`): exactly the two
`CudaComputeCapability*` cases go red. Returns 120 on all four GPUs here.

Wiring the per-device pairing exposed an architecture-independent defect:
`Device::get_devices()` returned `idx = 0` for every entry (its generator never
incremented), so the cc test checked GPU 0 four times and `gpus.at(1)` in
`linalg_layer_tests` was GPU 0. Fixed; `DeviceTest.GetDevices` now requires
`devices[i].idx == i`. Breaks, all four GPUs visible: the old generator turns only
`GetDevices` red (idx 1/2/3 read 0); forcing cc = 89 on device 1 turns only
`CudaComputeCapabilityOnNvidiaGpu` red ("CUDA ordinal 1"). Before the fix the
second break was invisible, because no Device had idx 1. With one GPU visible
both pass, as expected.

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

## potrf and posv

Machine as above, GPU 2 (no display), `factor_bench` from this branch against the BASE
build of 43b9f806 (icpx with `-ffp-model=precise`). Every cell is one process with all
arms interleaved, 7 timed iterations after warm-up, 3 reps with the arm order and the
BASE/new process order alternated, medians. The card's clocks drift up to 20% between
processes, so ratios are taken inside one process: `r = vendor / arm`, above 1 means
BatchLAS is faster. Batches are the benchviz campaign's (float n=64 b=32768 down to
n=512 b=512; cfloat half that). Routes were confirmed with `BATCHLAS_COVERAGE_OUT`
`reached` rows at every edge below. Raw CSVs and scripts:
`~/.claude/jobs/698ef31c/tmp/wp-potrf/` (`lp_ab_v1.csv`, `v2_ab.csv`, `final_ab.csv`,
`nbw_base.csv`, `nbw2_base.csv`, `bw_new.csv`, `posv_base.csv`).

### potrf LPanel vector sB

`potrf_lpanel_body`'s left-looking update read its lane-uniform sB operand with one
scalar shared load per FMA (cfloat: generic `LD.E`, the address space was lost). It now
reads each sB column as 16-byte `sycl::vec<float,4>` loads through a local-space pointer
(the block accessor is declared as 16-byte vectors for the alignment), and the panel
`sA` and the sB stores go through local-space views. Each `rS[i]` still accumulates `kk`
in ascending order, so the arithmetic is unchanged. Architecture-independent, so it is
unconditional.

LPanel pinned, `r` BASE -> new:

| n | float | cfloat |
|---|---|---|
| 48 | | 1.03 -> 1.25 |
| 64 | 1.80 -> 1.86 | 1.29 -> 1.49 |
| 96 | 1.28 -> 1.39 | 0.97 -> 1.17 |
| 128 | 1.23 -> 1.36 | 1.00 -> 1.18 |
| 160 | 1.11 -> 1.27 | 0.86 -> 1.01 |
| 192 | 0.90 -> 1.00 (b=2048); 1.08 -> 1.23 (b=4096) | 0.71 -> 0.78 |
| 256 | 0.95 -> 1.15 | 0.78 -> 0.99 |
| 320 | 0.84 -> 1.06 | 0.61 -> 0.72 |
| 384 | 0.70 -> 0.77 | |
| 512 | 0.76 -> 0.90 | |

The vector sB reads alone (`lp_ab_v1.csv`, 4 reps) gave float n=256 0.95 -> 1.11 and
cfloat n=256 0.71 -> 0.89; the local-space `sA` views took cfloat n=256 on to 0.99.
Not done: double-buffering sB to drop the second barrier (ncu attributed 54% of stall
cycles to barriers before this change).
Trap: sB is stored as the scalar type and read back as `sycl::vec<float,4>`, which is
type punning. It is correct only because barrier B1 sits between the stores and the
loads; a double-buffered sB that drops or moves B1 must do the stores through the vector
type as well.

`VectorSbReadIsBitIdenticalAcrossPackingAndBatch` arms it. Planted breaks, each rebuilt
and run over all of `potrf_tests`: a float vector width for every type turns red only the
LPanel suite's double, cfloat and cdouble instances; sB slots overlapping by half a slot
turns red only the cases that pack G > 1 matrices per work-group (this test at n=41
among them), every type.

### potrf LPanel auto window

With a vendor present, `preferred()` sends 32 < n <= edge (Lower) to LPanel. On sm_120,
measured on the new kernel, final routing:

- float edge 320: the Auto arm is r = 1.40 (128), 1.13 (256), 1.02 (288), 1.03 (320);
  pinned LPanel won in every rep from 224 to 320; at 352 LPanel is 0.72 (non-winner). n=192 at b=2048 ties (1.00, cuSOLVER is
  unusually fast at that batch; 1.23 at b=4096).
- cfloat edge 128: Auto 1.51 (64), 1.19 (128); pinned LPanel at 144 is 0.98-1.02, 160 1.01, 192 0.78 and
  256 0.99 (non-winners). 144 and 160 are ties, not losses: the edge could sit anywhere in 128-160,
  and 128 is the conservative choice.

sm_89 (and `cuda_cc` 0) keeps 256 for both types.

### potrf LPanel native tier cap

The vendor-free walk (`native_tier_preferred`) had no LPanel order cap. On sm_120 float
LPanel vs Blocked (retuned constants, ms): 288 1.66 vs 1.97, 320 2.16 vs 2.44, 352 1.91
vs 1.55, 384 2.31 vs 1.77, 512 4.47 vs 3.49. So float LPanel stops at 320, and only when
Blocked can take the shape. cfloat LPanel matches or beats Blocked to its ceiling (256
2.18 vs 3.01, 288 2.09 vs 2.14, 320 2.60 vs 2.58, 368 1.92 vs 2.11), so it is not
capped. The margin is real at 256 and 368 only; 288 and 320 are ties within noise,
so "not capped" rests on LPanel never losing, not on it winning everywhere. This
reverses the diagnosis, which was measured on the scalar kernel.

Native arm, `r` BASE -> new: float 352 0.62 -> 0.88, 384 0.66 -> 0.99, 512 0.73 ->
1.12; cfloat 368 0.54 -> 0.72.

### potrf Blocked nb W

The `BATCHLAS_POTRF_NB`/`_W` overrides on the BASE build, blocked vs vendor `r`:

| float n (batch) | 128/128 (sm_89) | 64/64 | 96/64 | 128/64 |
|---|---|---|---|---|
| 288 (1024) | 0.56 | 0.82 | | |
| 320 (1024) | 0.78 | 0.95 | 0.95 | 0.85 |
| 384 (512) | 0.75-0.81 | 0.97-0.99 | 0.95 | 0.83-0.93 |
| 512 (512) | 0.95-0.98 | 1.07-1.11 | 1.11 | 1.07-1.13 |
| 768 (128) | 1.06-1.07 | 0.90-0.92 | | 1.15-1.17 |
| 1024 (128) | 1.26-1.29 | 1.01 | | 1.30-1.32 |

W=64 wins at every order; nb=64 wins up to 512 and nb=128 above it, so sm_120 float uses
nb 64 through n=512 and 128 above, W 64. The switch point is bracketed only by 512
(64 wins) and 768 (128 wins); 576-704 were not measured, so where in (512, 768) the
crossover really lies is unknown and 512 is a guess. `PotrfBlockedTest.ResidualAboveTheCtaCeiling` judges
each order against the blocking the driver uses at that order and asserts that the
large-order pair ran (on sm_120 float, n=647 and 768 run 128/64); before, every size it
chose was at most 512 and so ran 64/64. cfloat (96/32) did not move: 64/64 was 0.62 vs
0.68 at 256, 0.75 vs 0.79 at 320, 0.89 vs 0.90 at 512, so it keeps the sm_89 pair. fp64
was not swept and is unchanged.

Not done: Blocked float beats cuSOLVER at n = 640-1024 (1.16-1.42 at b = 128-256), but
at n=512 the win drops from 1.12 at b=512 to 1.01 at b=2048, so it is not established at
saturation, and 1536/2048 lose (0.89/0.76). No vendor-present Blocked window was added.
cfloat Blocked stays at 0.67 of cuSOLVER at n = 512-1024, which is the complex trailing
update (the gemm package).

### posv on sm_120

posv has no routing change of its own: its potrf sub-op resolves through the table
above. Native walk, `r` BASE -> new: float n=512 nrhs=1 0.78 -> 1.17, nrhs=4 0.80 ->
1.16, nrhs=16 0.72 -> 0.99; n=256 nrhs=1 0.99 -> 1.14; n=128 nrhs=1 1.28 -> 1.39.

The fused `posv=native:cta` cells in the trsm diagnosis (cfloat 32/1 1.35x and 64/4
1.34x slower) were the icpx build: on BASE they are 1.82x and 1.52-1.59x faster than
the vendor composition, so the fused kernel needs nothing.

The remaining posv losses are nrhs > 8 at small n (float 32/16 0.52, 64/16 0.68, 128/16
0.79), which run `potrf=tiny|lpanel` plus the native trsm. Pinning only
`BATCHLAS_TRSM_ROUTE=vendor` makes the same composition 1.26x faster than the vendor
arm at 32/16 and 1.09x at 128/16, so the whole loss is the trsm kernel, which the trsm
package owns. posv routing is left unchanged.

## LU (getrf, getrs, gesv)

Machine: threadripper02, GPU 3 (RTX PRO 6000 Blackwell Max-Q, sm_120), icpx 2026.0
with `-ffp-model=precise`, RelWithDebInfo, CUDA 13.2 cuSOLVER/cuBLAS as the vendor.
Harness: `factor_bench` (verifies items 0 and batch-1 of every arm; every quoted row
is bad=0), reps 7-9, medians, 2 passes with alternating BASE/new order. BASE is
43b9f806. Ratios are time/vendor, so below 1 means faster than the vendor.

### LU getrs tiny

`Algorithm::Tiny` (src/extensions/getrs_tiny.cc): one sub-group partition per matrix,
lane r holds row r of op(A) and a chunk of RHS columns in registers. The permutation is
a per-lane trace of the interchange list, and both substitutions are shuffle
recurrences. There is no local memory and no work-group barrier. Two things decided
its shape:

- Divides. The first version divided by U(i,i) at every step. The compiler hoisted the
  N per-step divides out of the RHS chunk loop as loop invariants: cfloat n=32 used 255
  registers at 17% occupancy, 0.95 ms. Taking one reciprocal per lane and broadcasting
  the scaled value brought it to 128 registers, 0.39 ms (ncu).
- The chunk. A stays in registers, so a narrower chunk costs no extra traffic, only
  occupancy (batch 32768, ms):

  | cell | chunk 16 | 8 | 4 | 2 | 1 |
  |---|---|---|---|---|---|
  | cfloat n32 nrhs16 | 2.39 | 2.27 | 1.33 | 0.95 | 0.89 |
  | cfloat n16 nrhs8  | 0.27 | | 0.15 | 0.13 | 0.14 |
  | cfloat n8 nrhs16  | | | 0.074 | 0.072 | 0.085 |
  | float n32 nrhs16  | 0.84 | 0.50 | 0.47 | 0.43 | 0.46 |
  | float n8 nrhs16   | | | 0.047 | 0.057 | 0.067 |

  Shipped: a chunk of 2 for complex and 4 for real.

Window: every measured cell wins, for float and cfloat, over n in {2,4,8,12,16,24,32},
nrhs in {1,2,4,8,16,64} at batch 32768 (8192 for nrhs 64), and over batch 64/256/1024
at six shapes. Tiny time / vendor time at batch 32768:

| n | float r1 | r4 | r16 | r64 | cfloat r1 | r4 | r16 | r64 |
|---|---|---|---|---|---|---|---|---|
| 4  | 0.28 | 0.22 | 0.35 | 0.44 | 0.31 | 0.30 | 0.41 | 0.55 |
| 8  | 0.28 | 0.20 | 0.34 | 0.43 | 0.29 | 0.28 | 0.37 | 0.46 |
| 16 | 0.34 | 0.30 | 0.42 | 0.58 | 0.22 | 0.36 | 0.40 | 0.39 |
| 32 | 0.22 | 0.32 | 0.36 | 0.38 | 0.45 | 0.53 | 0.51 | 0.50 |

The closest small-batch cells are cfloat n32 nrhs64 b64 (0.186 vs 0.197 ms) and cfloat
n8 nrhs16 b64 (0.0246 vs 0.0271). The window is therefore the tier's whole fit:
`tiny_window = tiny_native = sm_120 && single precision && n <= tiny_max_n`. That is
wider than the measured grid: n=1 and batch < 64 were never timed and are admitted
unmeasured (batch 1 is outside this project's scope, and n=1 is one division). fp64 was
not measured, and it keeps the pre-Tiny walk.

### LU getrs fused

`getrs_fused.cc` ran every nb x nb block solve in sub-group 0, one column at a time,
while the other sub-groups waited at the barrier. Columns now go round-robin over the
sub-groups. This is an architecture-independent defect fix. BASE -> new, CTA arm, ms
(vendor in brackets):

| cell | nrhs 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| float n64 b16384  | 0.314 -> 0.320 | 0.378 -> 0.353 | 0.584 -> 0.449 (0.64) | 1.04 -> 0.77 (0.75) |
| float n512 b256   | 0.451 -> 0.459 | 0.403 -> 0.317 | 0.620 -> 0.434 | 1.18 -> 0.67 (1.02) |
| cfloat n64 b16384 | 0.542 -> 0.563 | 0.695 -> 0.595 | 1.08 -> 0.93 (0.83) | 1.99 -> 1.56 (1.06) |
| cfloat n128 b4096 | 0.475 -> 0.488 | 0.570 -> 0.488 | 0.80 -> 0.69 (0.84) | 1.72 -> 1.43 (0.88) |
| cfloat n512 b256  | 0.578 -> 0.608 | 0.672 -> 0.504 | 1.01 -> 0.79 (1.39) | 1.92 -> 1.20 (1.40) |

nrhs=1 was 1-3% slower, since it has only one column to spread, so NR == 1 now keeps
the sub-group-0 form (next section). On sm_120 the
in-library register counts exceed the sm_89 table: float Trans nrhs 5..8 uses 86
registers, so the sm_89 row handed out a 768-lane launch that the driver refused
(`FusedGetrsLaunchHoleAt48KiB`). On sm_120 the cap reads a probed row. The sm_89 rows
are covered under [sm89 rows](#lu-getrs-fused-sm89-rows).

Vendor-present window, added on sm_120 only: float nrhs <= 8 at n >= 256 (the
brackets are n=128 nrhs 8 at 0.63 vs 0.57 ms, a loss, and n=256 at 0.47 vs 0.66, a
win); cfloat nrhs <= 4 at n >= 128 (n=64 0.93 vs 0.85 loses, n=128 0.69 vs 0.86 wins)
and nrhs <= 8 at n >= 512 (n=256 1.38 vs 0.93 loses, n=512 1.20 vs 1.40 wins).

### LU getrs fused NR 1

threadripper02, GPU 3 (RTX PRO 6000 Blackwell Max-Q, sm_120), icpx 2026.0 RelWithDebInfo.
At NR == 1 the three fused kernels (getrs NoTrans, getrs Trans, potrs) again run the block
solves in sub-group 0 only. The round-robin `c = sgid; c += nsg` form is unchanged for
NR > 1. The guard is `(NR != 1) || sgid == 0` on the block, so each NR folds to its old
code. Standalone sm_89 ptxas counts are identical per kernel: NR == 1 matches the pre-round-robin TU,
and NR > 1 matches BASE, under both icpx 2026.0 and DPC++ 7.2.

BASE vs new, ms: `factor_bench` CTA arm (`getrs=native:cta`, and posv reached
`native:cta`, both confirmed with `BATCHLAS_COVERAGE_OUT`), 9 in-process reps, median of
4 alternating passes after a warm pass. Vendor is in brackets.

| cell | nrhs 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| cfloat n64 b16384 | 0.560 -> 0.541 (0.70) | 0.592 -> 0.592 | 0.927 -> 0.928 | 1.560 -> 1.556 |
| float n256 b2048  | 0.572 -> 0.570 (0.88) | 0.503 -> 0.504 | 0.644 -> 0.645 | 0.986 -> 0.982 |
| cfloat n128 b4096 | 0.484 -> 0.474 (0.70) | 0.490 -> 0.489 | 0.687 -> 0.686 | 1.416 -> 1.415 |

nrhs=1 recovers 3.5% and 2.2% on the cfloat cells, and new beats BASE in 4 of 4 passes.
float n256 is flat (0.3%). Every nrhs>1 cell is within 0.4%, with no consistent sign.
posv (native, potrf + fused potrs) is unchanged: float n256 r1 b1024 1.443 -> 1.443,
cfloat n128 r1 b4096 1.597 -> 1.597, float n256 r4 1.392 -> 1.391. potrf dominates it.

Deliberate break: `c0 = nrhs` at NR == 1, i.e. no sub-group solves the single column.
It was applied to one kernel at a time, with getrf/gesv/posv/potrf_tests run on GPU 3:

- NoTrans kernel: `LuTest/{4..7}.FusedGetrsSolvesEveryTransposeAtEveryInstantiatedWidth`
  and `LuTest/{4..7}.FusedGetrsAtBlockBoundariesAndTheNbSwitch`, 8 red. The other 200
  getrf tests and all gesv/posv/potrf tests stayed green.
- Trans kernel: the same 8 names. Both tests sweep both transposes.
- potrs kernel: `PosvTest/{4..7}.FusedSolveArmSolvesOnBothTriangles`, 4 red. Everything
  else stayed green.

The restored file was md5-identical to the pre-break copy.

### LU getrs fused sm89 rows

The round-robin commit changed the kernel bodies under the sm_89 `GetrsFusedRegs`
rows. Those rows were probed on the 4090, which cannot be re-probed from here. Standalone sm_89
ptxas counts of `getrs_fused.cc`, pre-round-robin -> now, NR > 1 (NR == 1 is identical):

| kernel, bucket | icpx 2026.0 | DPC++ 7.2 |
|---|---|---|
| cdouble NoTrans nrhs 2   | 50 -> 54 (+4) | 50 -> 51 (+1) |
| cdouble NoTrans nrhs 3-4 | 52 -> 56 (+4) | 56 -> 56 |
| cdouble Trans nrhs 3-4   | 62 -> 64 (+2) | 50 -> 50 |
| cfloat Trans nrhs 2      | 44 -> 46 (+2) | 40 -> 39 |
| double NoTrans nrhs 5-8  | 55 -> 61 (+6) | 54 -> 54 |
| double Trans nrhs 3-4    | 50 -> 54 (+4) | 40 -> 40 |
| double potrs nrhs 2      | 48 -> 50 (+2) | 48 -> 48 |
| double potrs nrhs 3-4    | 60 -> 62 (+2) | 46 -> 46 |
| double potrs nrhs 5-8    | 88 -> 92 (+4) | 70 -> 71 (+1) |
| cdouble potrs nrhs 2     | 52 -> 54 (+2) | 60 -> 54 |

Every other kernel moved by <= 0. Each row that moved up now carries the larger of the two
deltas: cdouble notrans {56,56} -> {60,60} at nrhs 2 and 3-4, cdouble trans 58 -> 60 at
3-4, cfloat trans 43 -> 45 at 2, double notrans 61 -> 67 at 5-8, double trans 51 -> 55
at 3-4. potrs used to be charged max(notrans, trans). It now has its own sm_89 row,
equal to that max except double nrhs 2 (52 -> 54) and nrhs 5-8 (72 -> 76), where the
potrs kernel moved past it. A static_assert keeps the potrs row >= the old max.

Effect on the sm_89 cap. The margin is +8 and allocation is rounded up to 8, so
only three caps move, all toward a narrower launch: cdouble NoTrans nrhs 2..4
1024 -> 896 lanes, double NoTrans nrhs 5..8 896 -> 768, and double potrs nrhs 5..8
768 -> 640. The fused width only reaches the cap at n >= 1026 (wg = 1024), so no smaller order is
touched. Those cells were not timed, because this machine has no sm_89. The in-library probe
(`probe89.sh`, the link line re-targeted to sm_89) printed no ptxas counts for this
library, in the LU round or this one. The evidence is therefore standalone TU compiles.

Not fixed: these rows bound DPC++-built sm_89 kernels. Under icpx 2026.0 the standalone
sm_89 counts already exceeded some rows before the round-robin, e.g. double potrs
nrhs 5-8 88 vs 72 + 8, and cfloat Trans nrhs 5-8 70 vs 56 + 8. An icpx build for a
4090 needs its own probe.

### LU getrf windows

BASE build, 3 passes, ms, vendor / blocked (native tier in the next section):

| cell | float | cfloat |
|---|---|---|
| n40 b32768  | 2.38 / 2.45 (loss) | |
| n48 b256    | 0.067 / 0.095 (loss) | |
| n48 b1024   | 0.136 / 0.122 | 0.166 / 0.180 (loss) |
| n48 b16384  | 1.60 / 1.37 | 2.37 / 2.56 (loss) |
| n64 b256    | 0.098 / 0.117 (loss) | 0.106 / 0.149 (loss) |
| n64 b1024   | 0.22 / 0.15 | 0.26 / 0.23 |
| n96 b256    | 0.19 / 0.20 (loss) | 0.23 / 0.26 (loss) |
| n96 b1024   | 0.52 / 0.29 | 0.73 / 0.52 |
| n128 b64    | 0.25 / 0.27 (loss) | 0.30 / 0.34 (loss) |
| n128 b256   | 0.34 / 0.29 | 0.41 / 0.39 |
| n192 b64    | 0.69 / 0.44 | 0.83 / 0.55 |
| n256 b64    | 1.33 / 0.62 | 1.53 / 0.82 |

The new window is blocked for n >= 192, or n >= 128 at batch >= 256, or n >= 48
(float) / 64 (cfloat) at batch >= 1024. The float n=65 tail-panel dip wins narrowly:
3.06 vs 3.16 ms.

The tiny window beats cuSOLVER at every measured order (vendor / tiny, b32768):
float n4 0.0160/0.0095, n32 0.66/0.25; cfloat n4 0.0173/0.011, n8 0.034/0.024,
n24 0.55/0.37, n25 0.60/0.41, n32 0.96/0.70, and cfloat n32 b1024 0.051/0.042. The
window is therefore 4..32 for both types. The sm_89 window excluded cfloat 8 and 25..32
as ties.

### LU getrf native tier

This is `native_tier_preferred`, the vendor-free walk. On sm_89 CTA won n=64 by 1.13x
(lu.md). On sm_120 blocked wins from float n=33 and cfloat n=40 at large batch (ms,
CTA / blocked): float n33 1.28/1.06, n48 2.38/1.37, n64 4.03/1.77, n72 8.75/3.30,
n77 4.88/1.80 (b8192); cfloat n33 1.72/1.79 (CTA wins), n40 2.29/2.17, n48 3.08/2.56,
n64 3.90/3.50. At batch 256 CTA still wins float n=48 (0.078/0.095) and ties at n=64,
so the CTA ceiling drops to 32 (float) / 39 (cfloat) only at batch >= 1024.

### LU gesv

With the tiny getrs routed, the cfloat composition (tiny getrf + tiny getrs) beats the
fused gesv tiny kernel from n=4 (b32768, ms, fused / composed): n4 r1 0.022/0.019,
n16 r4 0.257/0.160, n24 r4 0.827/0.654, n32 r1 1.02/0.89, n32 r4 1.27/1.03. It loses at
n<=3, where the composed getrf is the vendor's (n3 r1 0.018/0.022, n2 r4 0.018/0.023).
cfloat's fused window on sm_120 is therefore n <= 3. This supersedes the "widen to 32"
proposal (LU-6d), which was measured before the tiny getrs existed. float keeps 32:
fused and composed are within about 5% of each other, with the fused kernel ahead at
nrhs=4. That is a screen, not a bracket: the float edge was not re-bracketed against
the tiny-getrs composition. The cfloat n <= 3 edge is bracketed only in the vendor
build. In a vendor-free build the composed getrf at n <= 3 would be the tiny one, not
cuSOLVER's, and that composition was not timed.

### LU end to end

The 76 LU cells of the campaign loser list, 2 passes, BASE vs new, Auto route (the
shipped vendor-present route) and the pinned native walk, time/vendor:

- Auto geomean: 0.990 -> 0.534. Cells above 1.1x: 3 -> 0.
- Native-walk geomean: 1.178 -> 0.544.
- Examples (Auto): getrs cfloat n32 nrhs16 1.13 -> 0.45, getrs float n4 nrhs64
  1.51 -> 0.32 (BASE Auto took the float nrhs >= 64 blocked window at n=4), gesv
  cfloat n32 nrhs64 1.00 -> 0.28, getrf float n64 b32768 1.00 -> 0.64, getrf cfloat
  n32 1.00 -> 0.74.

Still losing in the native walk (Auto sends these to the vendor): getrs at n >= 64 with
nrhs=16 (float 1.32-1.46, cfloat 1.33-2.01) and gesv cfloat n64 nrhs16 (1.33). These
are the blocked composition, which is TrsmCta-bound (the LU-2 work-group ladder). The
getrs vendor-present composition windows for nrhs >= 16 at n > 32 (LU-3) are left for
re-bracketing after the trsm package lands. getrs cfloat n64 nrhs4 in the native walk
is 1.05.
