# Blackwell (sm_120) retune

Machine: threadripper02, 4x NVIDIA RTX PRO 6000 Blackwell Max-Q (CC 12.0, 188 SMs,
128 MB L2, ~1.8 TB/s GDDR7, 300 W). cuBLAS reaches 43.9-45.8 TFLOP/s SGEMM on this
part (power-limited: ~43% of the ~106 TFLOP/s FP32 peak). CUDA 13.2 (HPC SDK 26.5).

Starting point: benchviz saturation campaign `cuda-20260930-112900` (main @ 4e0a9b7d,
float + cfloat). Worst batchlas/vendor time ratios: cfloat gemm 3.0x geomean (15x at
32x32x1024), trsm float 1.5x (11x at n=32 q=8), getrs cfloat 1.8x (10x at nrhs=16),
potrf cfloat 1.6x (9.5x at n=256), geqrf skinny 2-3.7x.

## Result

Same campaign re-run on the tuned branch: `cuda-blackwell-tuned` (worktree-blackwell-tuning
@ ba07a99f, icpx 2026.0 RelWithDebInfo, fp model precise, 4 GPUs, 2066 arm-cells, same
grid). Cells are batchlas time / vendor time with batchlas pinned to its native walk
(benchviz's batchlas arm), so cells that Auto routes to the vendor still count here.

| op | type | cells | geomean before | geomean after | cells > 1.1 before | after |
|---|---|---|---|---|---|---|
| gemm | cfloat | 64 | 3.01 | 0.80 | 55 | 15 |
| gemm | float | 64 | 1.12 | 0.82 | 46 | 1 |
| gemv | cfloat | 64 | 1.02 | 1.00 | 1 | 0 |
| gemv | float | 64 | 1.01 | 0.97 | 5 | 0 |
| geqrf | cfloat | 35 | 0.77 | 0.44 | 14 | 0 |
| geqrf | float | 36 | 0.71 | 0.39 | 12 | 0 |
| gesv | cfloat | 40 | 1.23 | 0.46 | 18 | 0 |
| gesv | float | 40 | 0.65 | 0.34 | 9 | 0 |
| gesvd | float | 5 | 0.24 | 0.20 | 0 | 0 |
| getrf | cfloat | 8 | 1.04 | 0.63 | 3 | 0 |
| getrf | float | 8 | 0.61 | 0.47 | 1 | 0 |
| getrs | cfloat | 40 | 1.76 | 0.43 | 28 | 0 |
| getrs | float | 40 | 0.94 | 0.34 | 17 | 0 |
| orgqr | cfloat | 36 | 0.05 | 0.05 | 0 | 0 |
| orgqr | float | 34 | 0.04 | 0.04 | 0 | 0 |
| ormqr | float | 28 | 0.01 | 0.01 | 0 | 0 |
| posv | cfloat | 40 | 0.57 | 0.19 | 11 | 0 |
| posv | float | 40 | 1.05 | 0.44 | 23 | 0 |
| potrf | cfloat | 8 | 1.56 | 0.53 | 5 | 2 |
| potrf | float | 8 | 0.66 | 0.39 | 3 | 0 |
| spmm | cfloat | 42 | 0.48 | 0.44 | 0 | 4 |
| spmm | float | 42 | 0.33 | 0.26 | 0 | 0 |
| syev | cfloat | 8 | 0.69 | 0.64 | 1 | 1 |
| syev | float | 8 | 0.66 | 0.52 | 1 | 1 |
| syr2k | float | 49 | 0.07 | 0.06 | 0 | 0 |
| syrk | float | 49 | 0.07 | 0.06 | 0 | 0 |
| trmm | float | 7 | 0.79 | 0.77 | 1 | 1 |
| trsm | cfloat | 49 | 0.46 | 0.10 | 13 | 0 |
| trsm | float | 49 | 1.51 | 0.52 | 27 | 0 |
| **all** | | 1005 | 0.512 | 0.291 | 294 | 25 |

Remaining losers (> 1.1): 25 cells in this campaign, 21 after the spmm fix below.

- cfloat spmm n=16384: 1.11-1.13 in the campaign. The precise fp model caused it (the flag
  alone takes 5.51 -> 6.49 ms), and the pair-load fix in
  [#cfloat-spmm-under-the-precise-fp-model](#cfloat-spmm-under-the-precise-fp-model), merged
  after the campaign, makes all four cells 0.90-0.91 of cuSPARSE (re-measured on the
  merged build, GPU 0, 3 alternating reps: nrhs 16/32/64/128 at 5.45/5.37/5.31/5.28 ms
  against 6.05/5.92/5.84/5.87 ms). The cfloat spmm row in the table above predates the fix.
- cfloat gemm m=n=128..1024: 1.10-1.18 (gemm-7, the large-tile residual; not attempted).
- cfloat potrf n=512 (Blocked) 1.48 and n=256 (LPanel) 1.16: the complex trailing update.
  Under Auto, n=256 goes to the vendor.
- syev cfloat n=1024 b64 1.50 (Auto already ships cuSOLVER) and float n=1024 b128
  two_stage 1.18 (SYEV-4: the sb2st chase is one work-group per matrix).
- trmm float n=16 b32768 1.43 and float gemm 8x8x1024 1.12: not addressed.

Full ctest, tuned branch against the main build it replaces (both on this machine): 10
failing suites against 15. bdsdc, gesvd, stedc, syev_jacobi_cta, syev_two_stage, syevx and
sytrd_blocked now pass. `cond_tests` (CondTest/2.RandomHermitianTridiagonalLogCondSpectral,
0.52 decades vs a 0.5 tolerance) and `sytrd_cta_tests` (SytrdCtaTest/0.RandomSymmetricUpper,
2.3e-5 vs 2e-5) newly fail. Both fail identically at 43b9f806, which is main plus only the
`-ffp-model=precise` flag, and on dpcpp builds of main: the fast fp model was hiding them.

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

### cfloat spmm under the precise fp model

The flag made one kernel slower: the native CSR gather (`SpmmGatherKernel`,
`src/sycl/spmm_native.cc`) for cfloat, 5.51 -> 6.49 ms at n=16384, 16 nnz/row,
nrhs=64, batch=32 (cuSPARSE 5.76). Not occupancy: 74 vs 80 registers both round to
80, 3 blocks of 256, 24 warps/SM, no spills. Not instruction count either: precise
issues 414M instructions against fast's 1199M (its 100 `CALL.REL` are gone).

The kernel is L2-bound (L2 throughput 98-99%, DRAM 6%: B is L2-resident and the
gather is random). Each `Cx<float>` element of B was two 4-byte `LDG.E` of the same
sector. ncu on that cell, one launch:

| libs | time ms | L1 sectors | L1 hit | L2 sectors read |
|---|---|---|---|---|
| fast (out-of-line fma) | 5.04 | 1256M | 619M | 637M |
| precise | 6.06 | 1256M | 464M | 791M |
| precise + pair loads | 4.83 | 662M | 23M | 639M |

The 619M fast-build hits are the second half of each pair finding its sector
already in L1: there is almost no real reuse (23M once the pair is one load). The
fast build issued one column's pair, then called the fma stub, so the first half
had landed before the second was issued. Precise if-converts the 8-column block and
issues all 16 halves at once, so both halves of a pair miss and each sends its own
L2 request: 24% more L2 sectors, 20% more time.

Fix: load and store a complex element as one `sycl::vec<R, 2>` (`LDG.E.64` for
cfloat, `LDG.E.128` for cdouble). That needs `2*sizeof(R)` alignment, which
`std::complex<R>` does not promise, so the launcher checks A's values, B and C and
falls back to the scalar kernel for a view that starts one R off. cuSPARSE does not
accept that case at all ("matB had an illegal value: NULL pointer", status
unchecked by the vendor arm), so the misaligned tests skip unless the native
route is pinned. It is not an sm_120 constant: the halves-miss-twice effect is
how L1 handles two in-flight misses on one sector, so it is not gated by arch.

Median ms, GPU 0 only, main's `spmm_benchmark` with the libraries swapped through
`LD_LIBRARY_PATH`, `BATCHLAS_SPMM_ROUTE=native`, 16 nnz/row random pattern, benchviz
batches (~3 GiB). One warm pass discarded, then 3 reps in alternating arm order (5
for the headline, whose per-rep ranges do not overlap: fast 5.46-5.54, BASE
6.44-6.54, pair 5.18-5.26, cuSPARSE 5.67-5.78). BASE is
`worktree-blackwell-tuning` @ ba07a99f, whose kernel SASS is identical to the
flag-only build's.

| type | n | nrhs | batch | fast | precise BASE | pair | cuSPARSE | pair/BASE |
|---|---|---|---|---|---|---|---|---|
| cfloat | 1024 | 4 | 4096 | 2.12 | 1.68 | 1.62 | 9.18 | 0.97 |
| cfloat | 1024 | 16 | 2048 | 3.48 | 2.85 | 2.55 | 5.21 | 0.90 |
| cfloat | 1024 | 64 | 512 | 3.35 | 2.73 | 2.43 | 7.33 | 0.89 |
| cfloat | 1024 | 128 | 256 | 3.32 | 2.69 | 2.40 | 8.54 | 0.89 |
| cfloat | 4096 | 4 | 1024 | 3.08 | 3.35 | 2.86 | 10.87 | 0.85 |
| cfloat | 4096 | 16 | 512 | 5.08 | 5.73 | 4.77 | 5.96 | 0.83 |
| cfloat | 4096 | 64 | 128 | 4.94 | 5.57 | 4.63 | 5.75 | 0.83 |
| cfloat | 4096 | 128 | 64 | 4.92 | 5.54 | 4.61 | 5.82 | 0.83 |
| cfloat | 16384 | 4 | 256 | 3.56 | 4.16 | 3.41 | 11.03 | 0.82 |
| cfloat | 16384 | 16 | 128 | 5.85 | 6.87 | 5.56 | 6.15 | 0.81 |
| cfloat | 16384 | 64 | 32 | 5.49 | 6.49 | 5.23 | 5.74 | 0.81 |
| cfloat | 16384 | 128 | 16 | 5.65 | 6.66 | 5.37 | 5.97 | 0.81 |
| cdouble | 4096 | 16 | 256 | 4.02 | 4.36 | 3.68 | 8.31 | 0.84 |
| cdouble | 4096 | 64 | 64 | 3.93 | 4.25 | 3.59 | 8.29 | 0.84 |
| cdouble | 16384 | 16 | 64 | 4.13 | 4.67 | 3.92 | 8.85 | 0.84 |
| cdouble | 16384 | 64 | 16 | 4.01 | 4.55 | 3.81 | 8.83 | 0.84 |

Geomean pair/BASE: cfloat 0.85, cdouble 0.84, float 1.00 (same 12 n x nrhs cells,
every cell 0.99-1.00; the real kernels are unchanged). The pair build is faster
than the fast build on every complex cell, and than cuSPARSE on every cell here.

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

**potrf routing below is historical.** Flat selection deleted `route_potrf.hh`, its sm_120 edges
and its `native_tier_preferred` cap. potrf now takes the first runnable entry of the nearest row of
`tuned/potrf.<dtype>.sm_120.txt`, converted from the sm_120 route sweeps
([potrf.md](potrf.md#selection-since-flat-kernel-selection-phase-2)). The kernel and tuning-constant
results stand. posv's `RouteTable` windows are deleted too: posv reads `tuned/posv.<dtype>.<device>.txt`
([potrf.md](potrf.md#posv-selection-since-flat-kernel-selection-phase-3)); sm_120 has no posv table yet and borrows
the transcribed sm_89 one until its sweep is converted.

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
package owns. posv routing was left unchanged here (it has since moved to flat selection, see above).

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
These edges were later moved to float n >= 192 and cfloat n >= 96 / n >= 384, see
[getrs windows after the trsm fix](#getrs-windows-after-the-trsm-fix).

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

The NR == 1 gain is a float/cfloat result, not a general one. An independent review re-run
(same harness, same card) reproduced the float and cfloat wins (float n64 b16384
0.3256 -> 0.3205, also 4 of 4) and found fp64 flat: double n128 b4096 0.5586 -> 0.5601
(+0.3%, new faster in 1 of 4 passes; vendor 1.75), cdouble n64 b16384 2.1045 -> 2.1052
(vendor 5.52). Pre-existing and unchanged by this commit: posv double n128 r1 b4096 runs
native 7.14 ms against vendor 6.14 ms, in BASE as well.

Deliberate break: `c0 = nrhs` at NR == 1, i.e. no sub-group solves the single column.
It was applied to one kernel at a time, with getrf/gesv/posv/potrf_tests run on GPU 3:

- NoTrans kernel: `LuTest/{4..7}.FusedGetrsSolvesEveryTransposeAtEveryInstantiatedWidth`
  and `LuTest/{4..7}.FusedGetrsAtBlockBoundariesAndTheNbSwitch`, 8 red. The other 200
  getrf tests and all gesv/posv/potrf tests stayed green.
- Trans kernel: the same 8 names. Both tests sweep both transposes.
- potrs kernel: `PosvTest/{4..7}.FusedSolveArmSolvesOnBothTriangles`, 4 red. Everything
  else stayed green.

The restored file was md5-identical to the pre-break copy.

That break proves the NR == 1 path is reached. It does not prove the sub-group-0 guard is
needed. A second break tests the guard: `c_active = true` at NR == 1 while keeping
c0 = 0 and cs = 1, so every sub-group solves column 0 at the same time. It was applied to
each kernel in turn (NoTrans, Trans, potrs), with all four binaries rebuilt and run on
GPU 3. Every result was **green**: getrf 208, gesv 40, posv 40, potrf 209 passed. The
race writes identical values. A wrong answer needs one sub-group to load its column only
after another has stored the solved one, which is a full jb-step recurrence later, and
that never happened in these runs (the review also ran the fused tests 20 times with
`--gtest_repeat`, all green). So the suite does not guard against this race. It is
timing-dependent, and no deterministic test for it was written. The guard is there
because of the data race, not because of an observed failure.

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
touched. Those cells were not timed, because this machine has no sm_89.

This is a deliberate change of 4090 behaviour, and it applies to `cuda_cc == 0` as well:
every non-sm_120 device (a 4090, and any non-CUDA GPU) reads these sm_89 rows. It is the
one place in this package where cc 89 / cc 0 behaviour moves. Of the three caps, only one
fixes a launch the driver would refuse: the icpx-built double potrs nrhs 5-8 kernel uses
92 registers (88 before the round-robin), and at 768 lanes that is
ceil(24/4) * 32 * 96 = 18432 > 16384, an abort. The other two narrowings restore the +8
margin policy for launches that are legal today: cdouble NoTrans nrhs 2-4 at 54/56
registers fits 1024 lanes, and double NoTrans nrhs 5-8 at 61 fits 1024. They are
conservative, and they cost the 4090 some width at n >= 1026 for those two kernels.

The in-library probe
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
re-bracketing after the trsm package lands ([done](#getrs-windows-after-the-trsm-fix)). getrs cfloat n64 nrhs4 in the native walk
is 1.05.

### getrs windows after the trsm fix

LU-3, measured after the trsm V1 ladder cap and the sub-group Left kernel landed
(worktree-blackwell-tuning @ d776064b). Same machine, GPU 3, same build flags.
`factor_bench getrs` with arms vendor, auto, `native:blocked`, `native:cta` (nrhs <= 8)
and `native:tiny` (n <= 32) interleaved in one process, 7 in-process reps. One warm pass,
then 3 passes with the arm order reversed on alternate passes. Values are medians. The
"speedup" column is the median of the in-process vendor/arm ratios, so above 1 means
BatchLAS is faster. Saturated batch is the benchviz rule (about 3 GiB of matrices,
capped at 32768). The grid is n in {32,48,64,96,128,192,256,384,512} by nrhs in
{1,2,4,8,12,16,24,32,64,128}, both types. Batch ladders (128 to 16384) were added at
every edge. The edge cells were re-run as two-arm processes (vendor plus one native
arm, 5 passes). Those agree with the multi-arm runs to within 0.02, so arm position
does not move these ratios. Raw data: `~/.claude/jobs/698ef31c/tmp/wp3-lu3/`
(`grid.csv`, `edge.csv`, `mid.csv`, `lo.csv`, `conf.csv`, `gesv.csv`).

**Composition (`native:blocked`), vendor-present window.** It now beats cuBLAS at every
saturated nrhs >= 16 cell for n = 32..512 in both types. The weakest cells are cfloat
n128 nrhs16 at 1.04 and n256 at 1.16, and float n256 at 1.37. nrhs <= 15 is erratic and
loses at several orders. The composition's cost is not monotone in nrhs: float n512
takes 2.04 ms at nrhs 13 and 1.09 ms at nrhs 14. The cfloat cells n = 48..64 with
nrhs 16..24 dip below the vendor mid-ladder, at batch 4096-8192. Brackets, ms:

| cell | vendor | blocked | speedup | side |
|---|---|---|---|---|
| float n32 nrhs16 b32768 | 1.270 | 0.613 | 2.07 | in (order edge) |
| float n48 nrhs16 b4096 | 0.205 | 0.140 | 1.46 | in |
| float n128 nrhs16 b4096 | 0.994 | 0.704 | 1.42 | in |
| float n512 nrhs16 b603 | 1.865 | 1.084 | 1.72 | in |
| float n256 nrhs15 b2374 | 1.669 | 1.713 | 0.97 | out |
| float n512 nrhs13 b605 | 1.659 | 2.037 | 0.81 | out |
| float n256 nrhs12 b2390 | 1.529 | 1.587 | 0.96 | out |
| float n512 nrhs12 b605 | 1.588 | 1.920 | 0.83 | out |
| cfloat n48 nrhs32 b4096 | 0.625 | 0.520 | 1.19 | in |
| cfloat n48 nrhs32 b128 | 0.189 | 0.071 | 2.67 | in |
| cfloat n96 nrhs16 b4096 | 0.944 | 0.883 | 1.07 | in (order edge) |
| cfloat n96 nrhs16 b128 | 0.238 | 0.121 | 1.96 | in |
| cfloat n128 nrhs16 b4096 | 1.473 | 1.400 | 1.05 | in |
| cfloat n256 nrhs16 b1024 | 1.323 | 1.121 | 1.18 | in |
| cfloat n48 nrhs16 b4096 | 0.299 | 0.330 | 0.90 | out |
| cfloat n48 nrhs16 b8192 | 0.660 | 0.717 | 0.92 | out |
| cfloat n48 nrhs24 b4096 | 0.462 | 0.507 | 0.90 | out |
| cfloat n64 nrhs16 b4096 | 0.466 | 0.467 | 1.01 | out (tie) |
| cfloat n64 nrhs15 b17236 | 2.927 | 3.018 | 0.97 | out |
| cfloat n128 nrhs15 b4592 | 1.644 | 2.039 | 0.80 | out |
| cfloat n512 nrhs15 b301 | 1.910 | 3.099 | 0.62 | out |

At saturation the cfloat n48 nrhs16 cell wins (5.09 vs 2.63 ms), so the cfloat n < 96
exclusion is a batch-ladder loss and not a saturation one. The batch floor stays at 128.
Batch 128 wins in every probed cell, and smaller batches were not timed. Below order 32
the composition also beats cuBLAS (float n8..24 nrhs 16/64 1.41-3.47, cfloat
1.36-3.51), but the tiny tier is as fast or faster at every one of those cells and is first in the
walk, so the clause stops at 32. The cfloat cells at 1.04-1.07 sit inside the 5% band,
where AGENTS.md section 10 asks for 14-16 reps. They were reproduced in 3 multi-arm
and 5 two-arm passes, so they are ties or small wins, not losses.

**Fused tier (`native:cta`), vendor-present window.** The trsm fix does not touch this
kernel. The grid adds orders the earlier brackets did not have (96, 192, 384), and
the batch ladders move three edges down:

| cell | vendor | cta | speedup | side |
|---|---|---|---|---|
| float n192 nrhs8 b4096 | 1.144 | 1.089 | 1.05 | in |
| float n192 nrhs8 b256 | 0.281 | 0.172 | 1.63 | in |
| float n192 nrhs5 b4096 | 1.127 | 0.999 | 1.13 | in |
| float n128 nrhs8 b4096 | 0.578 | 0.623 | 0.92 | out |
| float n96 nrhs8 b4096 | 0.403 | 0.404 | 0.99 | out |
| cfloat n96 nrhs4 b4096 | 0.542 | 0.436 | 1.24 | in |
| cfloat n96 nrhs4 b256 | 0.164 | 0.109 | 1.51 | in |
| cfloat n96 nrhs3 b4096 | 0.542 | 0.407 | 1.33 | in |
| cfloat n64 nrhs4 b4096 | 0.224 | 0.236 | 0.95 | out |
| cfloat n64 nrhs4 b18950 | 0.974 | 1.057 | 0.92 | out |
| cfloat n384 nrhs8 b539 | 1.277 | 0.908 | 1.40 | in |
| cfloat n384 nrhs8 b256 | 0.983 | 0.801 | 1.23 | in |
| cfloat n384 nrhs5 b541 | 1.369 | 0.818 | 1.68 | in |
| cfloat n256 nrhs8 b1206 | 1.046 | 1.429 | 0.73 | out |
| cfloat n256 nrhs6 b1211 | 1.121 | 1.294 | 0.87 | out |

float nrhs 5..8 wins at saturation from n=48, but between n=48 and n=128 it loses at
batch 4096. That is the same mid-ladder shape the earlier n=128 bracket recorded.

**Native tier (`native_tier_preferred`): unchanged.** The LU-3 diagnosis proposed CTA only
for n >= 32 with nrhs <= 4, which would send nrhs 5..8 to the composition. Over the 119
measured cells with n > 32 and nrhs <= 8, CTA is faster in 111. The composition wins by
more than 5% only at float n48..64 nrhs 6..8 with batch >= 16384 (1.07-1.12) and at
cfloat n128 nrhs8 (CTA 1.77 ms, composition 1.31). At cfloat nrhs 8 the CTA times are
1.61 / 1.77 / 1.22 / 1.30 ms at n = 112 / 128 / 144 / 160. So n=128 is a cost anomaly in
the fused kernel, not a tier boundary, and both tiers lose to the vendor there anyway.
Elsewhere the composition is far behind at nrhs 8, for example cfloat n384 CTA 0.90 ms
against composition 2.29 ms.

**Effect on Auto** (saturated grid, n = 48..512, geomean vendor/Auto): float
1.63 -> 2.04 (25 cells move off the vendor, the smallest gain 1.08), cfloat
1.15 -> 1.55 (38 cells, the smallest gain 1.04). No cell moves onto a losing route.
gesv follows, because its composition calls the routed getrs. At saturated batch, gesv
Auto vs the all-vendor composition goes cfloat n128 nrhs16 1.24, n512 nrhs16 2.03, float
n48 nrhs16 1.66 and n512 nrhs64 2.94. With getrs pinned to the composition, cfloat n48
nrhs16 would reach 1.29. The window leaves that to the vendor because of the mid-ladder
loss above. `cuda_cc` 0 / 89 and fp64 route exactly as before.

## geqrf skinny panels

Machine: threadripper02, GPU 2 (RTX PRO 6000 Blackwell Max-Q, sm_120), icpx 2026.0 with
`-ffp-model=precise`, RelWithDebInfo, CUDA 13.2 cuSOLVER as the vendor. Harness:
`factor_bench geqrf <T> m n 0 <batch> 10` (every quoted row is `ok`: verified, rel_sd
within the harness gate), one throwaway JIT pass, 3 passes with the BASE/new order
alternated, medians. BASE is `worktree-blackwell-tuning` @ 502ad378. Ratios are
time/vendor, so below 1 means faster than cuSOLVER. The campaign losses were all on
the native-pinned walk (benchviz's batchlas arm); Auto sent every one of them to the
vendor, so the vendor build lost nothing before this package.

Everything below applies only to `is_sm120_family(cuda_cc)` and to float and cfloat.
cc 0 and 89, and fp64, keep BASE's behaviour; `SkinnyWindowIsSm120OnlyAndPicksOneTier`
and `Sm120LeafWidthFollowsTileBytes` assert that.

### geqrf: leaf width from tile bytes

The CTA tier's resident leaf took its work-group width from the column count alone
(`geqrf_panel_wg`: 128 threads at n=4, 256 from n=8) and packed one panel per sub-group
only at m <= 32. A 64x8 float panel (a 2 KiB tile) therefore ran on 8 warps behind three
work-group barriers per reflector. On sm_120 the CTA tier now sizes the group at a fixed
tile share per work-item (float 72 B, cfloat 96 B; 128 B for both before the review fixup,
see [the fixup](#geqrf-width-rule-review-fixup)), capped by the old ladder, keeps the
ladder above 16 KiB of tile, and packs panels of up to 4 KiB
into one sub-group at any m (`geqrf_cta.cc`, `geqrf_leaf_launch`). The investigation's
kernel-only probe (tmp probe of the shipped device body, float, batch 32768, ms):
64x8 wg256 0.93 vs packed 0.21; 128x8 wg256 1.11 vs wg64 0.46; 256x8 wg256 1.16 vs wg128
0.81; 64x32 wg256 5.25 vs wg64 3.24. cfloat 512x4 was already best at the ladder's 128.

Four limits, each measured:

- **Tile <= 16 KiB** (`kGeqrfNarrowMaxBytes`). Above it the group keeps the ladder's
  width. Found by the review: at 128 B an item, float 24-28 KiB panels ran wg128 and lost
  (native/BASE 256x24 1.08, 256x28 1.09, 192x32 1.07, 224x32 1.09, Auto too on the last
  two). Brackets after the fixup are in the fixup section.

- **Columns >= 8.** Below that the ladder already gives 128 and the rule starved tall
  panels (cfloat 257x2..384x2 1.2-1.4x slower in the first cut).
- **Rows <= 256.** Above 256 it was mixed at batch 32768: cfloat 300x8 0.90x, 384x8 1.07x
  (native, new/BASE). Both are 1.00 with the limit.
- **Batch >= 2048** (`kGeqrfBytesRuleMinBatch`). Narrower groups starve the device when
  the batch is small. Native new/BASE with the rule at every batch:

  | cell | b256 | b512 | b1024 | b2048 |
  |---|---|---|---|---|
  | float 40x32 | 2.46 | 2.15 | 1.16 | 0.76 |
  | float 64x32 | 1.58 | 1.42 | 0.77 | 0.96 |
  | float 128x16 | 1.45 | 1.34 | 0.73 | 0.92 |
  | float 128x32 | 1.19 | 1.05 | 1.06 | 1.00 |
  | cfloat 64x32 | 1.13 | 0.97 | 1.00 | 0.96 |

- **CTA tier only.** The blocked driver's panel leaf keeps the ladder. With the rule in
  the blocked leaf, float 256x64 and 256x128 were 1.02x BASE and 128x128 0.97x; without
  it every blocked cell is 1.00-1.01x (the remaining ~0.5% is the W1/W2 zero fill, see
  [known-defects #11](../design/known-defects.md#11-native-gemm-reads-c-at-beta-zero)).

### geqrf: width rule review fixup

Same machine and GPU 2, BASE build at d776064b (502ad378 plus gemm and trsm; the CTA tier
calls no gemm), fixup build at 1114fbc1. `factor_bench geqrf`, a JIT pass, then 3 passes
with the BASE/new order alternated, medians, ms, every row `ok`.

Two changes. (1) Tiles over 16 KiB keep the ladder (the regression above). (2) The tile
share per work-item became float 72 B, cfloat 96 B. A sweep of the share inside one
binary (a temporary env knob, not shipped; native, batch 32768, 4 rotated passes) showed
that 128 B left 12-15 KiB float panels on wg64 where wg128 is faster, and 4-7 KiB panels on
wg32 where wg64 is faster. The two types disagree at 10 KiB, which is why each type gets
its own share. Time at each share over time at 128 B:

| cell | KiB | 112 B | 96 B | 80 B | 64 B |
|---|---|---|---|---|---|
| float 48x32 | 6 | | 0.85 | 0.85 | 0.85 |
| float 64x32 | 8 | | 1.00 | 1.00 | 1.01 |
| float 128x16 | 8 | 1.00 | 1.00 | 1.00 | 1.01 |
| float 96x24 | 9 | | 1.00 | 1.00 | 0.96 |
| float 128x20 | 10 | 1.00 | 1.00 | 0.96 | 0.96 |
| float 192x16 | 12 | 1.00 | 0.92 | | 0.91 |
| float 224x16 | 14 | 0.85 | 0.85 | | 0.85 |
| float 128x32 | 16 | 1.00 | 1.00 | | 1.07 |
| cfloat 32x24 | 6 | | 0.93 | 0.93 | 0.93 |
| cfloat 64x16 | 8 | | 1.00 | 1.00 | 1.11 |
| cfloat 64x20 | 10 | | 1.00 | 1.03 | 1.02 |
| cfloat 80x16 | 10 | | 1.00 | 1.02 | 1.02 |
| cfloat 96x16 | 12 | 1.00 | 0.97 | | 0.97 |
| cfloat 112x16 | 14 | 0.95 | 0.95 | 0.95 | 0.95 |
| cfloat 128x16 | 16 | 1.00 | 1.00 | | 1.21 |

So float wants wg128 from 9 KiB (72 B: 9 KiB is exactly 128 items) and not at 8 KiB;
cfloat wants wg64 at 10 KiB and wg128 at 12 KiB (96 B: 12 KiB is 128 items).

Final build against BASE, batch 32768 (native new/BASE; new native/vendor; new
Auto/vendor). Every cell is at or below 1.00 on both arms:

| cell | KiB | nat new/BASE | nat/vendor | Auto/vendor |
|---|---|---|---|---|
| float 40x32 | 5 | 0.56 | 0.69 | 0.69 |
| float 64x28 | 7 | 0.63 | 0.72 | 0.72 |
| float 96x24 | 9 | 0.73 | 0.79 | 0.79 |
| float 128x20 | 10 | 0.75 | 0.75 | 0.76 |
| float 128x28 | 14 | 0.86 | 0.65 | 0.65 |
| float 224x16 | 14 | 0.87 | 0.67 | 0.67 |
| float 128x32 | 16 | 0.93 | 0.64 | 0.64 |
| float 256x16 | 16 | 0.95 | 0.64 | 0.64 |
| float 160x32 | 20 | 1.00 | 0.57 | 0.57 |
| float 256x20 | 20 | 1.00 | 0.57 | 0.57 |
| float 224x24 | 21 | 1.00 | 0.54 | 0.54 |
| float 192x32 | 24 | 1.00 | 0.52 | 0.52 |
| float 256x24 | 24 | 1.00 | 0.52 | 0.52 |
| float 256x28 | 28 | 1.00 | 0.45 | 0.45 |
| float 224x32 | 28 | 1.00 | 0.45 | 0.45 |
| float 256x32 | 32 | 1.00 | 0.41 | 0.41 |
| cfloat 48x16 | 6 | 0.53 | 0.82 | 0.83 |
| cfloat 40x32 | 10 | 0.72 | 0.81 | 0.81 |
| cfloat 64x20 | 10 | 0.68 | 0.78 | 0.78 |
| cfloat 112x16 | 14 | 0.77 | 0.70 | 0.70 |
| cfloat 128x16 | 16 | 0.83 | 0.68 | 0.68 |
| cfloat 128x20 | 20 | 1.00 | 0.68 | 0.68 |
| cfloat 192x16 | 24 | 1.00 | 0.56 | 0.56 |
| cfloat 256x12 | 24 | 1.00 | 0.57 | 0.57 |
| cfloat 256x16 | 32 | 1.00 | 0.43 | 0.43 |

The 48-cell run (m 32..256, n 16..32 plus float 64x16, 256x8) has its worst native
new/BASE at 1.00 (float 192x24, 224x24, 256x20..32, cfloat 32x32, 128x20, 192x16, 256x12,
256x16) and its worst Auto/vendor at 0.90 (cfloat 32x24). At the batch gate (batch 2048)
the new widths are also faster than BASE: native new/BASE 0.56-0.97 on 11 cells
(float 40x32 0.59, 224x16 0.82; cfloat 48x16 0.56, 64x32 0.97), Auto/vendor 0.36-0.60.

Tests: `Sm120LeafWidthFollowsTileBytes` now also checks 16 KiB + 1 row and 24 KiB (ladder)
and the 128-item edge per type (float 72x32 / 71x32, cfloat 48x32 / 47x32). Breaks: ceiling
raised to 32 KiB turns red `Sm120LeafWidthFollowsTileBytes` for float and cfloat only; float
share 80 B turns red the float instance only (50 suites otherwise green).

### geqrf: the skinny register leg

`geqrf_skinny.cc` / `geqrf_skinny_device.hh`: a second leg of the CTA tier (the route is
still `native:cta`) for n <= 8. The panel lives in one sub-group's registers: P lanes own a
matrix (P = 8 or 16 below 17 rows, so 2-4 matrices share a sub-group; else 32), row r sits
on lane r % P in slot r / P, and there is no local memory and no barrier. The n-1-j column
reductions of each reflector are independent butterflies instead of one SLM reduction per
column. The Householder scalars are `geqrf_larfg_scalars`, so tau and beta follow the
resident body's LAPACK convention. Instantiations: N in {4, 8} columns, RP in
{1, 2, 4, 8, 16} row slots (m <= 512 for float, <= 256 for cfloat: its 16-slot bucket
measured slower than the resident leaf at 257x4 and 300x3). It is a functor, so the launch
bound can be spelled; MinBlocks = 1, because a tighter cap spilled the RP >= 8 buckets and
bought nothing measurable elsewhere (float 8x4: MB1 0.0118, MB4 0.0108, MB8 0.0107 ms;
64x8 0.090/0.089/0.089).

The leg is chosen inside the CTA launcher by `geqrf_skinny_preferred`, which is the route
window below plus a batch gate: a lane holding >= 512 B of A (cfloat RP 8 with 8 columns,
float RP 16 with 8 columns) needs batch >= 1024, below which the resident leaf wins.
Skinny/resident, pinned with `BATCHLAS_GEQRF_LEAF`:

| cell | b256 | b512 | b1024 |
|---|---|---|---|
| cfloat 160x8 (heavy) | 1.41 | 1.25 | 0.75 |
| cfloat 256x8 (heavy) | 1.29 | 1.13 | 0.69 |
| cfloat 256x6 (heavy) | 1.36 | 1.24 | 0.77 |
| float 400x8 (heavy) | 1.13 | 1.02 | 0.60 |
| cfloat 128x8 | 1.00 | 0.90 | 0.55 |
| cfloat 256x4 | 0.90 | 0.86 | 0.87 |
| float 256x8 | 0.77 | 0.72 | 0.46 |
| float 512x8 | 0.96 | 0.86 | 0.52 |
| float 512x4 | 0.69 | 0.68 | 0.68 |

`BATCHLAS_GEQRF_LEAF=resident|skinny` pins the leg (a pin that does not fit throws).

### geqrf: the sm120 tall window

`route_geqrf.hh` gains two sm_120-only clauses in `preferred()`, both before the order
floor, both answering for exactly one native tier (`best_native_tier`):

- `geqrf_skinny_window`: float n <= 8 with m <= 256, or m <= 512 with n >= 2; cfloat
  n <= 8 with m <= 256; strictly tall (m > n). Square shapes stay with the tiny tier.
- `geqrf_sm120_tall_window`: float and cfloat, 9 <= n < m, 32 <= m <= 512; plus cfloat
  300 <= m <= 512 with 4 <= n <= 8 (resident leaf, above the skinny leg's rows).

Brackets (native/vendor at batch 32768; excluded cells in the right column):

| edge | inside | outside |
|---|---|---|
| float skinny, m | 512x2 0.80, 512x4 0.52 | 400x1 1.39, 512x1 1.36 |
| tall, m >= 32 | 32x16 float 0.72, cfloat 0.87 | 24x12 float 1.15, cfloat 1.17 |
| tall, m <= 512 | 512x16 float 0.44, cfloat 0.48 | not measured above 512 |
| cfloat band, m | 300x4 0.90, 300x6 0.86, 400x5 0.84 | 290x4 0.91, 260x6 0.98, 260x5 1.14 |
| cfloat band, n | 512x4 0.82 | 300x3 1.10, 512x2 1.13 |
| cfloat skinny, m | 256x4 0.62 | 257x2 1.03, 257x4 0.97 |

cfloat 257x8 (0.73) and 260x8 (0.75) win but sit between the skinny rows and the band;
they were left to the vendor because 257x6 (0.95) and 260x5 (1.14) do not, and a
per-n edge was not worth the extra clause.

The window is keyed on shape alone, and it holds at small batch as well. Final build,
BASE vs new, ms: native is identical to BASE at b256 and b1024 (the batch gates) except
where the skinny leg runs (b1024 float 400x8 0.061 -> 0.037, cfloat 256x8 0.058 -> 0.041),
and Auto moves from the vendor to native:

| cell | b256 vendor | BASE Auto | new Auto | b1024 vendor | BASE Auto | new Auto |
|---|---|---|---|---|---|---|
| float 40x32 | 0.319 | 0.319 | 0.073 | 0.320 | 0.320 | 0.161 |
| float 64x32 | 0.379 | 0.379 | 0.078 | 0.382 | 0.382 | 0.173 |
| float 128x16 | 0.136 | 0.136 | 0.041 | 0.137 | 0.137 | 0.088 |
| float 400x8 | 0.094 | 0.094 | 0.030 | 0.096 | 0.096 | 0.037 |
| cfloat 256x8 | 0.085 | 0.085 | 0.029 | 0.086 | 0.086 | 0.041 |
| cfloat 64x32 | | | 0.096 | 0.499 | 0.499 | 0.219 |

Routes checked with `BATCHLAS_COVERAGE_OUT` (Auto, `reached` rows): cfloat 400x5, float
64x8 and float 40x32 at b256 are `native,cta`; cfloat 290x4 is `vendor,auto`.

### geqrf skinny panels: result

Campaign loser cells (the native-pinned benchviz arm), batch 32768, ms:

| cell | vendor | BASE native | new native | new/vendor |
|---|---|---|---|---|
| float 8x4 | 0.0283 | 0.0534 | 0.0129 | 0.46 |
| float 16x4 | 0.0412 | 0.0585 | 0.0184 | 0.45 |
| float 16x8 | 0.1085 | 0.1672 | 0.0382 | 0.35 |
| float 64x4 | 0.0920 | 0.1961 | 0.0371 | 0.40 |
| float 64x8 | 0.309 | 0.833 | 0.104 | 0.34 |
| float 64x16 | 1.145 | 1.919 | 0.982 | 0.86 |
| float 64x32 | 4.487 | 4.684 | 3.313 | 0.74 |
| float 128x4 | 0.133 | 0.270 | 0.070 | 0.53 |
| float 128x8 | 0.479 | 1.031 | 0.194 | 0.40 |
| float 128x16 | 1.893 | 2.357 | 1.643 | 0.87 |
| float 256x4 | 0.287 | 0.414 | 0.163 | 0.57 |
| float 256x8 | 1.106 | 1.359 | 0.406 | 0.37 |
| cfloat 8x4 | 0.0321 | 0.0680 | 0.0160 | 0.50 |
| cfloat 16x4 | 0.0499 | 0.0773 | 0.0252 | 0.50 |
| cfloat 16x8 | 0.146 | 0.229 | 0.064 | 0.44 |
| cfloat 32x4 | 0.105 | 0.097 | 0.047 | 0.45 |
| cfloat 32x8 | 0.313 | 0.266 | 0.133 | 0.43 |
| cfloat 32x16 | 1.221 | 1.066 | 1.068 | 0.87 |
| cfloat 64x4 | 0.132 | 0.282 | 0.071 | 0.54 |
| cfloat 64x16 | 1.984 | 2.660 | 1.558 | 0.79 |
| cfloat 64x32 | 8.577 | 6.832 | 5.711 | 0.67 |
| cfloat 128x4 | 0.250 | 0.380 | 0.172 | 0.69 |
| cfloat 128x8 | 0.926 | 1.405 | 0.542 | 0.58 |
| cfloat 128x16 | 4.053 | 3.349 | 2.779 | 0.69 |
| cfloat 256x4 | 0.620 | 0.611 | 0.387 | 0.62 |
| cfloat 512x4 | 1.465 | 1.203 | 1.205 | 0.82 |

The 106-cell grid (m 4..512, n 1..128, the benchviz geqrf cells plus the brackets),
geomean time/vendor:

| | BASE Auto | new Auto | BASE native | new native |
|---|---|---|---|---|
| float (49 cells) | 0.846 | 0.537 | 1.038 | 0.544 |
| cfloat (57 cells) | 0.867 | 0.652 | 0.904 | 0.622 |

(new Auto here predates the cfloat band. Re-measured on the final build, Auto/vendor:
300x4 0.89, 300x8 0.66, 384x8 0.61, 400x5 0.84, 512x6 0.64, 512x8 0.48; 290x4 stays
vendor-routed at 1.00.) Worst new Auto cell: 1.02 (vendor-routed
noise). Native cells slower than BASE by more than 2%: none in this grid (the first
cut's cfloat 384x8 1.07 is fixed by the 256-row limit, and cfloat 257x2..384x2 by the
8-column limit). The grid had no float cell between 16 and 32 KiB of tile at n >= 8, which
is where the review found 1.07-1.09x losses; see
[the fixup](#geqrf-width-rule-review-fixup). This grid predates the fixup and was not
re-run; the fixup leaves every campaign cell's width unchanged.

Consumers: `syev` two-stage runs geqrf on m x 32 sy2sb panels at batch 64-512, and
`ortho` (Householder) on tall panels. syev_benchmark, BASE vs final build, native, ms, 4
alternating reps: float 1024 b256 494.3 / 493.8, float 1024 b128 283.9 / 283.0, float 896
b512 648.2 / 647.4, float 512 b512 two_stage 151.4 / 150.6, cfloat 512 b128 two_stage
129.9 / 128.7, cfloat 1024 b64 264.8 / 264.1. (An earlier cut without the batch gates
read float 1024 b256 529.7 / 557.7 and b128 291.8 / 310.9; the gates removed that.)
`sytrd_sy2sb_tests`, `syev_two_stage_tests` and `orgqr_tests` pass; `ortho_tests` passes
through OrthoMatrixTest/6 and crashes in /7 (complex<double>, the pre-existing cuBLAS
Zgemv segfault on this box; no double path changed). `orgqr`, `ormqr` and `gesvd` do not
call geqrf's leaves.

Break sweep (each break alone, `geqrf_tests` filtered to the CTA, blocked, facade and new
suites; red set per break, float = /4, cfloat = /6):

| break | red |
|---|---|
| partition butterfly starts at 16 for every P (cross-matrix leak) | SkinnyLegResidualInEveryBucket, SkinnyLegAgreesWithTheResidentLeaf, SkinnyLegIdentityReflectorInOnePartition, SkinnyLegSaturatingBatchIsBitIdentical (float, cfloat) |
| skinny store uses m instead of ld | the three residual/agreement/identity suites, FacadeReachesTheSkinnyLeg, CtaResidualAndOrthogonality (float, cfloat) |
| width gate 2048 -> 2049 | Sm120LeafWidthFollowsTileBytes (float, cfloat) |
| heavy-lane gate 1024 -> 1025 | SkinnyWindowIsSm120OnlyAndPicksOneTier (float, cfloat) |
| skinny window without the sm_120 guard | SkinnyWindowIsSm120OnlyAndPicksOneTier (float, cfloat) |
| no W1/W2 zero fill | BlockedIgnoresAGarbageWorkspace (all four types) |

`CtaResidualAndOrthogonality` going red under the ld break shows the pre-existing CTA suite
reaches the skinny leg through the ordinary route on this device.

Not done:

- The multi-matrix packing for m <= 16 exists (P = 8/16) but no per-n window refinement
  between 257 and 299 cfloat rows (see the bracket note above).
- fp64: no skinny instantiation, no width rule, no window. Not measured.
- float 400x1 / 512x1 (1.36-1.39 native/vendor): n = 1 above 256 rows stays with the vendor.
- The skinny N = 16 bucket (the probe showed a 64-512 B stack frame at N = 16).

## trsm

Machine: threadripper02, GPU 0 (RTX PRO 6000 Blackwell Max-Q, sm_120), icpx 2026.0 with
`-ffp-model=precise`, RelWithDebInfo, CUDA 13.2 cuBLAS as the vendor. BASE is
worktree-blackwell-tuning @ 502ad378. `trsm_benchmark` (`BM_TRSM<`: Side::Left, Lower,
NoTrans, ld = n) with `--warmup=3 --min_iters=8`, one process per (cell, arm), a
throwaway JIT pass first, then 3 reps with the arm order alternated, medians. The
vendor arm is `BATCHLAS_TRSM_ROUTE=vendor`; the others pin `native`. Batches are the
benchviz campaign's. Raw logs, CSVs and the drivers (`ab.py`, `fb.py`) are in
`~/.claude/jobs/698ef31c/tmp/wp2-trsm/` (`f_trsm.csv`, `f_win.csv`, `f_posv.csv`,
`f_getrs.csv`, `r_edge.csv`, `r_right.csv`, `f_right.csv`).

### trsm V1 ladder cap

The V1 CTA kernel gives each lane one right-hand-side column, and lanes without a
column still run the whole N-step recurrence. Its work-group ladder over
{256, 128, 64, 32} took the first rung with `batch * ceil(q / wg) >= 4 * CU`. At any
saturated batch that is 256, so a q=8 solve ran 248 dead lanes per group. The ladder
now skips a rung more than half of whose lanes would have no column
(`cand > 32 && cand / 2 >= q`, in `trsm_v1_ladder_wg`). This is an
architecture-independent defect fix, so it is unconditional. The A/B is against the
BASE build; the development knob that restored the old ladder was removed.

It was measured on sm_120 only. On the RTX 4090 (cc 89) V1's work-group geometry also
changes for q < 128 at saturated batch, and routing there is otherwise unchanged. The
change is correctness-neutral: the group barrier stays and the work-group only shrinks.
Its sm_89 timing is not measured. Re-measure small-q V1 cells there before quoting a
4090 number.

V1 alone, BASE -> capped, ms (batch 32768): float n32 q8 0.994 -> 0.284, n16 q8
0.280 -> 0.067, n32 q32 1.105 -> 0.43; cfloat n32 q8 1.551 -> 0.44, n16 q16
0.479 -> 0.162. The cap also applies to Side::Right (`r_right.csv`,
`BM_TRSM_OrthoRight`, BASE -> new, with cuBLAS after the slash): float n32 q8
1.042 -> 0.306 / 0.165, n32 q32 1.425 -> 0.400 / 0.740, n16 q16 0.277 -> 0.060 / 0.125,
cfloat n32 q8 1.635 -> 0.426 / 1.147. The q >= 256 ortho shapes never reach a capped
rung and stayed within 3% (`~/.claude/jobs/698ef31c/tmp/wp-trsm/r_ortho.log`).

### trsm sub-group Left kernel

`trsm_sg_left.cc` handles Side::Left at orders 1..32. Each lane is a (matrix, canonical
row r) pair, so a sub-group holds 32/N matrices for the buckets N in {4, 8, 16, 32}.
Each lane carries QC right-hand sides: 4 for q <= 4, 8 for q <= 8, else 16
(complex<double> caps at 8). Lane r holds row r of the canonical Lc and its QC values
of x, and step s broadcasts x[s] from lane s with `select_from_group`. There is no
local memory and no barrier, and a lane exists only for a real (row, rhs-chunk) pair.
The 24 (side, uplo, trans, diag) cases use V1's canonical fold (`trsm_canonical.hh`).
The non-finite-reciprocal fallback to division is kept per matrix, as an xor-butterfly
over the matrix's N lanes.

Two register decisions, from `scripts/register_probe.sh` on the sm_120 link:

- The step loop stops at the runtime order n, since rows n..N-1 are identity padding.
  A `break` in the loop stops LLVM from unrolling it, and nL[] (the row of Lc) then
  lands in local memory. That is slower at small QC (float n32 q8 0.116 -> 0.143 ms).
  It is faster where the unrolled form's registers limit occupancy: at cfloat N=32
  QC=16 the count drops from 168 to 77 registers, and n32 q64 goes from 1.91 to 1.39
  ms. So the loop is rolled on purpose (`#pragma unroll 1` + `break`) for complex
  N >= 16 and for N=32 QC=16, and fully unrolled with a `continue` everywhere else.
  No variant spills.
- The kernel's TU is built with `-pragma-unroll-threshold=262144`. Without it, the
  unrolled variants put x[] on the stack.

Kernel choice (`trsm_left_use_sg`): V1-capped time over sub-group time at batch 32768,
so above 1 the sub-group kernel is faster (`f_win.csv`):

| float n \ q | 8 | 16 | 24 | 32 | 48 | 64 | 128 |
|---|---|---|---|---|---|---|---|
| 4  | 3.18 | 2.67 | 2.22 | 2.15 | 2.03 | 1.89 | 1.81 |
| 8  | 2.29 | 1.84 | 1.53 | 1.45 | 1.25 | 1.09 | 0.97 |
| 12 | 2.76 | 1.89 | 1.31 | 1.28 | 1.20 | 0.92 | 0.91 |
| 16 | 2.58 | 1.75 | 1.17 | 1.09 | 0.98 | 0.94 | 0.88 |
| 24 | 3.18 | 1.51 | 1.09 | 1.20 | 1.01 | 0.87 | 0.84 |
| 32 | 2.49 | 1.43 | 1.10 | 1.15 | 0.92 | 0.81 | 0.76 |

| cfloat n \ q | 8 | 16 | 24 | 32 | 48 | 64 | 128 |
|---|---|---|---|---|---|---|---|
| 4  | 2.50 | 1.54 | 1.21 | 1.14 | 1.20 | 1.07 | 0.86 |
| 8  | 1.73 | 1.27 | 0.99 | 1.39 | 0.88 | 0.96 | 0.97 |
| 12 | 1.31 | 0.87 | 0.57 | 0.64 | 0.71 | 0.73 | 0.79 |
| 16 | 1.26 | 1.08 | 0.88 | 1.61 | 1.47 | 1.79 | 1.66 |
| 24 | 1.31 | 1.28 | 1.01 | 1.33 | 1.08 | 1.28 | 1.06 |
| 32 | 1.32 | 1.36 | 0.99 | 1.32 | 1.26 | 1.37 | 1.12 |

The shipped window applies on sm_120 only; `cuda_cc` 0 and 89 keep V1 at every shape.

- float: q <= 128 at n <= 4, q <= 64 at n <= 8, q <= 32 above.
- cfloat: q <= 64 at n <= 4, q <= 32 at n <= 8, q <= 8 for n in 9..15, n = 16 at
  q <= 16 or 32 <= q <= 128, and q <= 128 for n in 17..32.

Each edge is bracketed by the next measured column. Ties within 3% (float n8 q128,
cfloat n8 q24/q64, cfloat n24/n32 q24) go to whichever side the neighbouring cells
favour. The cfloat n=16 island comes from V1, not from the new kernel: V1 at cfloat
n16 q64 takes 0.91 ms, against 0.31 ms at n=12 and 1.48 ms at n=24 (same bucket,
power-of-two ld). double and complex<double> were not measured and keep V1. The
blocked driver's diagonal solves are 32-wide Left CTA calls, so they take the same
kernel through `trsm_native_v1_buckets`.

Tests (`TrsmNativeSgLeft.*`) call the kernel directly with a padded ld and batch
stride and large finite poison. Three planted breaks were each rebuilt and run over all
of `trsm_tests`. Reading A with ld = n turned red the three direct strided suites plus
the routed Left orders and the blocked suites (10 tests). Dropping the store's
`c0 + j < q` guard turned red 13 tests before the run aborted with an illegal address
in the saturating-batch case: `TrsmNativeCta.Complex{CanonicalCrossProductFloat,
AlphaHasImaginaryPart, PartialBucketAndRaggedRhs}`, `TrsmNativeBlocked.{FloatAndRaggedRhs,
ComplexCrossoverAndAlpha, TwoLevelFloatAndComplex}`, the four `TrsmFloatLeftOrders` Left
tests, and the three strided/edge `TrsmNativeSgLeft` tests (`break2.log`). Ending the
step loop one row early turned red every test that reached the kernel (15), while the
Right-side cases stayed green.

These red sets were recorded before `RunTrsmNative` ran each Left case twice. It now
runs V1 pinned (`allow_sg = false`) and then with the kernel choice. On sm_120 the
choice sends small-q Left to this kernel, so V1's Left path stays covered on this box.
`TrsmNativeCta.LadderRungsAreCappedByRhsCount` checks the ladder against literal
(max_wg, CU, q, batch) cases.

### trsm on sm_120: result

`BM_TRSM<`, BASE / new / cuBLAS in ms, plus the time ratios against cuBLAS
(`f_trsm.csv`):

| type | n | q | batch | BASE | new | cuBLAS | BASE/cuBLAS | new/cuBLAS |
|---|---|---|---|---|---|---|---|---|
| float | 8 | 8 | 32768 | 0.128 | 0.015 | 0.043 | 2.99 | 0.34 |
| float | 8 | 32 | 32768 | 0.132 | 0.029 | 0.116 | 1.14 | 0.25 |
| float | 16 | 8 | 32768 | 0.280 | 0.026 | 0.072 | 3.86 | 0.36 |
| float | 16 | 64 | 32768 | 0.387 | 0.228 | 0.530 | 0.73 | 0.43 |
| float | 32 | 8 | 32768 | 0.994 | 0.115 | 0.171 | 5.83 | 0.67 |
| float | 32 | 16 | 32768 | 1.025 | 0.230 | 0.324 | 3.16 | 0.71 |
| float | 32 | 32 | 32768 | 1.105 | 0.376 | 0.683 | 1.62 | 0.55 |
| float | 32 | 64 | 32768 | 1.083 | 0.575 | 1.238 | 0.87 | 0.46 |
| float | 64 | 8 | 32768 | 2.074 | 0.440 | 0.517 | 4.01 | 0.85 |
| float | 64 | 16 | 32768 | 2.166 | 0.713 | 1.026 | 2.11 | 0.69 |
| float | 128 | 8 | 32768 | 4.861 | 1.448 | 1.667 | 2.92 | 0.87 |
| float | 128 | 64 | 32768 | 7.777 | 5.259 | 9.566 | 0.81 | 0.55 |
| float | 256 | 8 | 8192 | 3.012 | 1.364 | 1.415 | 2.13 | 0.96 |
| float | 256 | 256 | 4096 | 6.618 | 6.616 | 10.878 | 0.61 | 0.61 |
| float | 512 | 8 | 2048 | 1.893 | 1.091 | 1.125 | 1.68 | 0.97 |
| float | 512 | 16 | 2048 | 2.026 | 1.317 | 1.353 | 1.50 | 0.97 |
| float | 512 | 512 | 1024 | 7.594 | 7.603 | 12.780 | 0.59 | 0.59 |
| cfloat | 8 | 8 | 32768 | 0.141 | 0.020 | 0.053 | 2.67 | 0.39 |
| cfloat | 16 | 8 | 32768 | 0.427 | 0.077 | 0.342 | 1.25 | 0.23 |
| cfloat | 32 | 8 | 32768 | 1.551 | 0.333 | 1.391 | 1.12 | 0.24 |
| cfloat | 32 | 64 | 32768 | 2.483 | 1.380 | 10.523 | 0.24 | 0.13 |
| cfloat | 64 | 8 | 32768 | 3.588 | 1.131 | 7.382 | 0.49 | 0.15 |
| cfloat | 128 | 64 | 16384 | 8.613 | 5.921 | 140.6 | 0.06 | 0.04 |
| cfloat | 512 | 512 | 512 | 12.095 | 12.106 | 568.8 | 0.02 | 0.02 |

Over the whole 62-cell grid (n = 8..512, q in {8, 16, 32, 64, n}):

- float: the geomean time ratio against cuBLAS went from 1.44 to 0.57, and the cells
  above 1 went from 21 to 0. The closest cells are float n=512 q=8/16 and n=256 q=8
  at 0.96-0.97, where 15 trailing cuBLAS gemm launches remain.
- cfloat: the geomean went from 0.18 to 0.09, and the cells above 1 from 5 to 0.

The q = n cells at n >= 128 run V1 on uncapped rungs and did not move.

The fixup that removed the development knobs was re-measured on the review's 23 Left
and Right cells, 6 Auto cells, and 7 posv/getrs cells, against BASE and cuBLAS
(`wp2-trsm/fx/m_*.csv`). Every cell matched the review within about 2%, for example
float n32 q8 at 0.117 ms (cuBLAS 0.169) and cfloat n32 q8 at 0.335 ms (1.409). No cell
moved against BASE in the wrong direction.

`preferred()` is unchanged for Side::Left (Side::Right: next section). With the new
kernel, native wins every measured Side::Left cell, so the diagnosis's interim small-q vendor window (trsm-interim-route-small-q) is
not needed.

posv (`factor_bench`): the native arm is the shipped native walk, which at these nrhs
is potrf plus two routed trsm. Ratios are vendor time over native time in one process,
so above 1 BatchLAS is faster. BASE -> new:

- float: n4 q16 0.55 -> 5.68, n32 q16 0.51 -> 1.88, n64 q16 0.68 -> 1.61, n128 q16
  0.77 -> 1.28, n128 q64 1.28 -> 1.74.
- cfloat: n4 q16 0.39 -> 2.53, n8 q16 0.87 -> 3.43, n32 q16 1.56 -> 4.89, n128 q16
  2.70 -> 4.02.

All 24 cells (n = 4..128, nrhs 16/64) now beat the vendor composition; BASE lost 9 of
them.

getrs `native:blocked` is the TrsmCta-bound case (the LU-2 issue). Same ratios, BASE ->
new:

- float: n64 q16 0.71 -> 2.17, n128 q16 0.67 -> 1.63, n256 q16 0.75 -> 1.41, n512 q16
  1.31 -> 1.89, n512 q64 3.10 -> 3.10.
- cfloat: n64 q16 0.53 -> 1.41, n128 q16 0.51 -> 1.04, n256 q16 0.75 -> 1.17, n512 q16
  1.26 -> 1.52.

All 16 cells (n = 64..512, nrhs 16/64) now win; BASE lost 6. The getrs vendor-present
windows for nrhs >= 16 at n > 32 (LU-3) still send these shapes to the vendor. They
belong to the LU package, to re-bracket against these numbers.

### trsm side right on sm120

Side::Right has no sub-group kernel, so it keeps the capped V1. Even capped, float V1
loses to cuBLAS when a solve has few rows (`BM_TRSM_OrthoRight`, `f_right.csv`, ms,
native / cuBLAS):

| float | b=2048 | 4096 | 8192 | 16384 | 32768 |
|---|---|---|---|---|---|
| n32 q8  | 0.0276 / 0.0298 | 0.047 / 0.039 | 0.070 / 0.053 | 0.128 / 0.087 | 0.305 / 0.164 |
| n20 q8  | | 0.042 / 0.031 | 0.063 / 0.040 | 0.112 / 0.059 | 0.200 / 0.099 |
| n16 q8  | | | | | 0.059 / 0.072 |
| n32 q16 | | 0.047 / 0.054 | 0.070 / 0.087 | 0.134 / 0.155 | 0.342 / 0.312 |
| n20 q16 | | 0.042 / 0.040 | 0.063 / 0.059 | 0.112 / 0.099 | 0.210 / 0.178 |
| n32 q20 | | 0.047 / 0.075 | 0.071 / 0.124 | 0.150 / 0.230 | 0.360 / 0.497 |

On sm_120, `preferred()` therefore declines native for float Side::Right when
order > 16, rows <= 8 and batch >= 4096. The edges are brackets against measured
non-losers: order 16 (native 0.059 vs 0.072), rows 16 at n=32 (native wins up to b=16384),
and batch 2048 (0.0276 vs 0.0298). `cuda_cc` 0 and 89 route as before. Auto, BASE ->
new, ms (`f_rauto.csv`, cuBLAS in brackets): n32 q8 b32768 1.041 -> 0.162 (0.163),
n32 q8 b4096 0.122 -> 0.039 (0.039), n20 q8 b4096 0.107 -> 0.031 (0.031), n16 q8
b32768 0.277 -> 0.059 (0.072).

The window has no upper order bound. The review re-measured orders above 32 at q <= 8
and found that the vendor wins at every one, native / cuBLAS in ms
(`~/.claude/jobs/698ef31c/tmp/review2-trsm/r_rbig.csv`):

- n48 q8 b8192: 0.141 / 0.082
- n64 q8 b32768: 1.007 / 0.481
- n128 q8 b16384: 1.330 / 0.855
- n256 q4 b4096: 1.072 / 0.683

Auto matched the vendor at each of them.

Not done: rows 9..16 at order 17..24, and at order 32 with batch 32768, still lose
5-18% (n20 q16, n32 q16 b32768). A Side::Right sub-group kernel (each row of B is a
Left solve with op(A)^T, so the canonical fold carries over with the B access
transposed) would fix both, but it was not written.

## gemm (native register-tiled selector)

Machine: threadripper02, GPU 1 (RTX PRO 6000 Blackwell Max-Q, sm_120), icpx 2026.0
with `-ffp-model=precise`, RelWithDebInfo, CUDA 13.2 cuBLAS as the vendor. Harness:
`gemm_benchmark` / `gemm_transpose_benchmark` (`--warmup=3 --min_iters=8`), one
throwaway JIT pass per cell, 3 reps alternating arm order, medians. BASE is
502ad378 (its libraries, `BATCHLAS_GEMM_ROUTE=native`); new is the same binary
against the new libraries. Ratios are time/vendor. Kernel choice was checked with
`BATCHLAS_KERNEL_TRACE` on every quoted NN cell.

Everything below is keyed on `is_sm120_family(cuda_cc)` in `select_kernel_variant`
(`src/sycl/gemm_kernels.cc`). cc 0 and cc 89 keep the 4090 ladder, which
`GemmDispatchPolicyTest.Sm120SmallTilesAndTheirEdges` and
`Sm120TransposedFallbackAndItsEdges` assert shape by shape. None of it changes
`preferred()`: in a vendor build, Auto still sends complex and transposed float to
cuBLAS. The consumers are the native walk, vendor-free builds and direct
`gemm_custom` callers. The vendor-build factorizations are unchanged: potrf, getrf
and geqrf in float, cfloat and cdouble at n=128-512 measured new/base 0.996-1.003.

### gemm small tiles

There are two new NN instantiations of the wide-scalar template
(`launch_wide_transposed`, 64 threads each): 16x16 with a 2x2 thread tile and 32x32
with a 4x4 thread tile (`Tiled16x16RegisterK16Wide`, `Tiled32x32RegisterK16Wide`).
They are forceable as `16x16x16wide` and `32x32x16wide`. On the 4090 the only complex
register kernel was the 64x64 tile, which at m=n=32 computes 3/4 padding.

Windows (float and complex<float>; `fits16` = max(m,n) <= 16, or min(m,n) <= 8 with
max(m,n) <= 32):

- 16x16 tile: `fits16`, at least 1024 tiles in flight, and k >= 32 (float) or
  max(m,n,k) > 8 (cfloat). float needs 2048 tiles from k = 768 on.
- 32x32 tile: max(m,n) <= 32 and not `fits16`, at least 1024 tiles, and k > 32 (float).
  float needs 1536 tiles from k = 512 on (see "Tiles in flight at deep k" below).
- float, 32 < min(m,n), max(m,n) < 128, max(m,n,k) > 56: the 64x64 wide tile when
  min(m,n) <= 64, otherwise 128x128x8 (gemm-3).
- cfloat 64x64 wide gate: min(m,n) >= 32 instead of min(m,n,k) >= 32 (gemm-2).
  For k >= 32 it keeps the 4090's 64 CTAs. For k < 32 it needs m*n*batch >= 2^19
  (see the gate table below).

Final A/B, NN, time/vendor (BASE -> new):

| cell | BASE | new | | cell | BASE | new |
|---|---|---|---|---|---|---|
| f 8x8x32 b32768 | 1.10 | 0.38 | | c 8x8x256 b32768 | 1.19 | 0.58 |
| f 8x8x128 | 1.35 | 0.71 | | c 8x8x1024 b16384 | 1.36 | 0.58 |
| f 8x8x1024 | 2.11 | 1.10 | | c 16x16x16 | 0.54 | 0.23 |
| f 16x16x32 | 0.99 | 0.51 | | c 16x16x64 | 1.31 | 0.51 |
| f 16x16x1024 b16384 | 1.33 | 0.85 | | c 16x16x1024 b8192 | 1.38 | 0.79 |
| f 24x24x128 | 2.71 | 1.02 | | c 32x32x8 | 1.23 | 0.76 |
| f 32x32x64 | 1.08 | 0.93 | | c 32x32x16 | 1.64 | 0.72 |
| f 32x32x256 | 1.20 | 0.92 | | c 32x32x64 | 1.48 | 0.95 |
| f 32x32x1024 b8192 | 1.34 | 0.99 | | c 32x32x1024 b4096 | 2.04 | 1.07 |
| f 40x40x128 | 3.22 | 1.02 | | c 64x64x8 | 1.65 | 0.96 |
| f 64x64x8 | 1.35 | 0.93 | | c 64x64x16 | 2.74 | 0.96 |
| f 64x64x16 | 1.69 | 0.93 | | c 128x128x8 b16384 | 1.57 | 0.99 |
| f 64x64x1024 b4096 | 1.16 | 0.89 | | c 1024x1024x16 b256 | 2.47 | 1.00 |
| f 96x96x32 | 1.52 | 1.02 | | | | |
| f 100x100x64 b16384 | 3.21 | 1.28 | | | | |
| f 127x127x16 b16384 | 2.09 | 1.02 | | | | |

Batch is 32768 unless given. Controls, unchanged by construction and by measurement
(BASE = new): f 8x8x16, 16x16x16 and 32x32x32 (SmallBatched), 48^3, 128x128x32,
128x128x1024 and 256^3; c 8^3 (Direct), 64x64x64 and 256^3 (wide 64x64). With
`BATCHLAS_BENCH_LD_PAD=1`: f 64x64x16 1.62 -> 0.96, f 32x32x256 1.82 -> 0.90,
f 100x100x64 3.12 -> 1.32, c 16x16x64 1.34 -> 0.60, c 64x64x16 2.65 -> 0.99.

Brackets (forced-kernel screens on the same card, as /vendor):

- k edge of the float 16 tile: 16x16x16 SmallBatched 0.27 vs 16x16 0.30 (kept), and
  16x16x32 1.01 vs 0.51. 8x8x16: SmallBatched 0.25, Direct 0.22, 16x16 0.25 (kept).
  This supersedes gemm-4. Direct beats SmallBatched at 8x8x32 (0.36), but the 16x16
  tile is as good (0.39) and wins from k=64 up (8x8x64 0.57 vs Direct 0.83).
- k edge of the float 32 tile: 32x32x32 SmallBatched 0.85 vs 0.89, and 32x32x64 1.10
  vs 0.93.
- Tiles in flight (k = 256): float 16x16x256 b512 is 1.77 on the 16 tile vs 1.25 on
  BASE, and b1024 1.22 vs 1.35. Float 32x32x256 b512 1.85 vs 1.20, and b1024 1.08 vs
  1.09. cfloat 32x32x64 b1024 1.05 vs 1.69. The 1024 floor holds only up to moderate k;
  see the next subsection.

#### Tiles in flight at deep k

Review found the 1024-tile floor losing at deep k: one 64-thread group per tile walks k
serially, and at 1024 tiles the older kernels (Tiled16 at 16x16, Tiled32x32Register at
32x32) are not yet wave-limited. Forced-tile screens, GPU 1, beta = 1, 3 reps, BASE vs
the tile, /vendor at `LD_PAD=0` / `LD_PAD=1`:

| cell | tiles | BASE | tile | verdict |
|---|---|---|---|---|
| f 16x16x384 | 1024 | 1.49 / 1.32 | 1.34 / 1.33 | tile |
| f 16x16x512 | 1024 | 1.64 / 1.45 | 1.43 / 1.41 | tile |
| f 16x16x512 | 1536 | 1.62 / 1.46 | 1.10 / 1.11 | tile |
| f 16x16x768 | 1024 | 1.68 / 1.50 | 1.44 / 1.43 | BASE (forfeit) |
| f 16x16x768 | 1536 | 1.08 / 1.01 | 1.11 / 1.11 | BASE |
| f 16x16x1024 | 1024 | 1.61 / 1.21 | 1.71 / 1.77 | BASE |
| f 16x16x1024 | 1536 | 1.13 / 1.03 | 1.12 / 1.12 | BASE |
| f 16x16x1024 | 2048 | 1.11 / 0.98 | 0.92 / 0.95 | tile |
| f 16x16x2048 | 2048 | 1.23 / 1.10 | 0.96 / 0.97 | tile |
| f 15x15x1024 | 1024 | 1.49 / 1.48 | 1.63 / 2.09 | BASE |
| f 15x15x1024 | 2048 | 1.06 / 1.06 | 0.94 / 0.96 | tile |
| f 32x32x384 | 1024 | 1.03 / 1.05 | 1.01 / 1.04 | tile (tie) |
| f 32x32x512 | 1024 | 1.23 / 1.45 | 1.76 / 1.79 | BASE |
| f 32x32x512 | 1536 | 1.45 / 1.71 | 1.31 / 1.32 | tile |
| f 32x32x768 | 1024 | 1.27 / 1.55 | 1.66 / 1.73 | BASE |
| f 32x32x768 | 1536 | 1.45 / 1.66 | 1.23 / 1.25 | tile |
| f 32x32x1024 | 1024 | 1.29 / 1.47 | 1.76 / 1.77 | BASE |
| f 32x32x1024 | 1536 | 1.43 / 1.54 | 1.29 / 1.26 | tile |
| f 32x32x2048 | 1536 | 1.33 / 1.41 | 1.25 / 1.21 | tile |
| f 32x32x2048 | 2048 | 1.23 / 1.30 | 1.01 / 1.00 | tile |
| f 16x32x512 | 1024 | 1.25 / 1.05 | 1.16 / 1.09 | BASE |
| f 16x32x1024 | 1024 | 1.32 / 1.19 | 1.83 / 1.86 | BASE |

The tile's time barely moves from 1024 to 2048 tiles (16x16x1024: 0.176 -> 0.265 ms)
while BASE doubles, so the crossover is a tile count that grows with k. The floor is
now 2048 tiles for the float 16 tile from k = 768 and 1536 for the float 32 tile from
k = 512. cfloat keeps 1024: at 1024 tiles and k = 1024 it wins (16x16 1.47 -> 1.02,
32x32 1.72 -> 1.14).

Confirmation after the fix, one run with four arms (BASE, the pre-fix head 6b6aa30b,
the fix, vendor), 3 reps, /vendor at pad 0 / pad 1, kernel checked by trace:

| cell | BASE | pre-fix | fix |
|---|---|---|---|
| f 16x16x1024 b1024 | 1.55 / 1.20 | 1.64 / 1.76 | 1.56 / 1.20 |
| f 16x16x1024 b2048 | 1.12 / 0.99 | 0.92 / 0.94 | 0.93 / 0.94 |
| f 16x16x768 b2048 | 1.05 / 0.94 | 0.92 / 0.93 | 0.92 / 0.93 |
| f 32x32x512 b1024 | 1.22 / 1.38 | 1.75 / 1.76 | 1.22 / 1.48 |
| f 32x32x512 b1536 | 1.44 / 1.71 | 1.31 / 1.32 | 1.31 / 1.32 |
| f 32x32x1024 b1024 | 1.29 / 1.49 | 1.77 / 1.76 | 1.29 / 1.47 |
| f 32x32x1024 b1536 | 1.43 / 1.55 | 1.28 / 1.26 | 1.28 / 1.26 |
| f 32x32x1024 b2048 | 1.33 / 1.43 | 1.10 / 1.08 | 1.10 / 1.07 |

32x32x512 b1024 at pad 1 runs the same kernel as BASE; a 7-rep recheck gives 1.47 vs
1.41. Unchanged cells: f 16x16x32, 32x32x1024 b8192, 40x40x128, cfloat 16x16x1024 and
32x32x1024 b1024.

The rule forfeits gains on shapes that fill the tile badly, where the older kernel is
slow: at 1024 tiles the tile wins f 8x8x1024 (2.93 vs 2.33), 12x12x1024 (2.22 vs
1.85), 24x24x512 (2.30 vs 1.11), 24x24x1024 (2.48 vs 1.68), 20x20, 28x28 and 31x31
x1024 (2.40-2.47 vs 1.66-1.75) and 32x8x1024 (2.56 vs 1.76), and these now stay on
BASE until 1536/2048 tiles. A fill-based exception was not adopted: 15x15 (fill 0.88)
loses and 31x31 (fill 0.94 of Tiled16) wins, so no single fill threshold separates
the measured cells.
- float 33..56 cubes stay on SmallBatched: 33^3 0.80 vs 64-wide 1.04, 56^3 0.94 vs
  0.95. 40x40x128 (max 128) goes to the 64 tile, 1.02 vs 3.22.
- float min(m,n) > 64: 65x65x64 128x128 1.18 vs 64-wide 1.22, 100x100x256 1.06 vs
  1.48, 127x127x16 1.03 vs 1.46. 64x96x128 (min 64): 64-wide 0.84 vs 128x128 1.02.
- cfloat 16 vs 32 tile at 16x16: 16x16x64 0.51 vs 0.87. At 32x32 (not fits16):
  32x32x1024 16 tile 1.38 vs 32 tile 1.07.

Two things tried and not adopted. k-split (2-4 slices of k per group with an SLM
reduction) was never better than the unsplit 16/32 tiles: float 32x32x1024 ks2 0.91
vs 0.98 was the only win, and cfloat 32x32x* ks2 ran 1.07-1.31. A 128x64 tile and a
64x64 tile with K=32 for large cfloat were 1.28-1.55 vs 1.11-1.17 for the 64x64 K=16
tile.

cfloat k < 32 gate, BASE -> new (these cells are launch-bound, around 10-20 us):

| cell | m*n*batch | BASE | 64 wide | shipped |
|---|---|---|---|---|
| 33x33x8 b64 | 70k | 0.86 | 1.02 | BASE |
| 33x33x8 b256 | 279k | 0.98 | 1.01-1.03 | BASE |
| 64x64x8 b64 | 262k | 0.87 | 1.00 | BASE |
| 128x128x8 b16 | 262k | 1.02 | 1.11 | BASE |
| 256x256x16 b4 | 262k | 1.10 | 0.90 | BASE (win given up) |
| 64x64x8 b128 | 524k | 1.16 | 1.09 | wide |
| 128x128x8 b32 | 524k | 1.25 | 1.10 | wide |
| 48x48x16 b256 | 590k | 1.63 | 1.02 | wide |
| 33x33x16 b512 | 558k | 1.39 | 0.86 | wide |

A CTA count does not separate these cells: 33x33x8 b256 loses at 256 CTAs while
64x64x8 b256 wins. The output element count does.

### gemm complex double

complex<double> NN on sm_120 goes to the 16x16 tile whenever max(m,n,k) > 8. At
FP64's 1/64 rate the kernel is compute-bound, and the tile that puts the most groups
in flight won or tied at every shape and batch measured: 16x16 >= 32x32 >= 64x64,
and all three beat Tiled16 and Direct. The exception is 8^3, where Direct is 0.22 vs
0.38. Final A/B (BASE -> new, /vendor):

| cell | BASE | new |
|---|---|---|
| 8x8x16 b32768 | 0.40 | 0.37 |
| 8x8x256 b16384 | 1.54 | 0.44 |
| 16x16x16 b32768 | 1.54 | 0.41 |
| 24x24x64 b32768 | 1.96 | 0.85 |
| 32x32x8 b32768 | 1.72 | 0.81 |
| 32x32x256 b8192 | 3.45 | 0.87 |
| 48x48x16 b16384 | 2.41 | 0.62 |
| 64x64x16 b16384 | 3.21 | 0.82 |
| 64x64x64 b16384 | 0.89 | 0.86 |
| 100x100x16 b8192 | 2.44 | 0.71 |
| 128x128x128 b2048 | 0.89 | 0.87 |
| 256x256x256 b512 | 0.89 | 0.89 (min_dim >= 256 keeps the 64 tile) |
| 1024x1024x16 b64 | 2.83 | 0.83 |
| 16x16x64 b16 | 0.61 | 0.60 |
| 32x32x64 b64 | 1.29 | 0.60 |
| 64x64x16 b16 | 1.14 | 0.66 |

Brackets: 64x64x16 b16384 on the 16, 32 and 64 tiles is 0.82, 0.81 and 0.90.
100x100x16 is 0.71, 0.90 and 0.90. 16x16x64 b512 on the 64 tile is 4.77, so the tile
must not grow with the shape. 8^3 b64: Direct 0.46, 16 tile 0.60. double (real) was
screened too. The 16 tile gains 5-25% (16x16x16 0.30 -> 0.22), but double already
beats cuBLAS 1.1-4x there, so it is left alone.

### gemm transposed fallback

float and complex<float> one-transposed forms (TN/NT; complex only with ConjTrans)
that would fall to Tiled16 or Direct now take a wide transposed tile when
min(m,n) >= 32 and there are >= 128 64x64 CTAs. The tile is 64x64, except for float
with a panel shape: 128x32 NC when n < 64 and m >= 128, 32x128 CN when m < 64 and
n >= 128. Two things are unchanged: the float K32 family (m >= 128, n >= 32,
k >= 128) and float SmallBatched (max <= 32). On the 4090 the real-scalar tile was a
tie with Tiled16 (docs/perf/gemm.md#wide-scalar-transposed-tiles). Here Tiled16 is
2-5x behind cuBLAS on these forms.

Final A/B, `LD_PAD=1`, beta=1, BASE -> new, /vendor:

| form | cell | BASE | new |
|---|---|---|---|
| f NT | 256x256x32 b2048 | 3.53 | 1.49 |
| f NT | 512x512x64 b512 | 5.72 | 1.64 |
| f NT | 64x64x16 b16384 | 1.73 | 0.98 |
| f NT | 256x32x96 b1024 | 4.60 | 1.73 |
| f NT | 32x256x256 b2048 | 3.94 | 1.64 |
| f NT | 64x64x64 b128 | 1.66 | 1.48 |
| f TN | 256x256x32 b2048 | 3.37 | 1.48 |
| f TN | 32x256x256 b2048 | 3.12 | 1.20 |
| f TN | 48x48x48 b16384 | 2.27 | 1.08 |
| c NC | 64x64x64 b8192 | 2.96 | 1.02 |
| c NC | 32x256x256 b2048 | 4.17 | 1.71 |
| c NC | 32x32x32 b16384 | 2.55 | 1.45 |
| c CN | 64x64x64 b8192 | 2.50 | 0.86 |
| c CN | 256x32x256 b2048 | 3.29 | 1.44 |
| c CN | 32x32x256 b8192 | 2.26 | 1.83 |

Brackets. The 128-CTA floor counts CTAs of the tile actually launched (review fix:
it used to count 64x64 CTAs for the panel tiles too). f NT 256x32x96 pad 0 on Tiled16
vs 128x32 NC: b32 (64 panel CTAs, 128 64x64 ones) 1.51 vs 1.85, b64 (128) 2.17 vs
1.81; b16 stays on Tiled16. f NT 64x64x64 b64 (64 CTAs) is 1.27 on Tiled16 vs 1.52
on 64 NC, and b128 is 1.66 vs 1.48 (re-measured: b64 1.34 both, b128 1.78 vs 1.40). 128x128x32 b16 is 1.04 vs 1.38, and b64 is 1.65
vs 1.39. The min(m,n) >= 32 floor: f NT 16x16x256 is 0.62 on Tiled16 vs 1.39, and
c CN 16x16x256 is 0.93 vs 2.48. The panel-shaped tile: f NT 256x32x96 is 1.69 on 128x32
vs 2.07 on 64x64, and 256x256x32 (pad 0) 1.28 vs 1.15. Cells that do not move: f NT and
TN 256x32x256 (K32), c NC 256x32x256 and c CN 32x256x256 (existing 128x32 / 32x128
selector), and f 16x16x256.

Still behind cuBLAS: float NT/TN at a strided ld (1.2-1.7; e.g. NT 256x40x64 1.78 and
256x32x96 1.51, both from 4.5-5x on BASE), and panel cells near the 128-CTA floor
(256x32x96 b64 1.81, where the vendor is launch-bound at 14 us). The float K32 NT kernel
at `LD_PAD=1` (256x32x256 NT 2.02; 128x32 NC measured 1.35 there, pad 0 a tie 0.69 vs
0.71). cfloat 32x32 transposed, which has no 32x32 transposed tile (1.45-1.91).

### gemm result

Selector-reached kernels were confirmed per cell by `BATCHLAS_KERNEL_TRACE`. Tests
are in tests/gemm_tests.cc: `SmallWideNNTilesMatchTiled16` (ragged m, n, k from 1 to
257, sub-view ld, both betas), `SelectorChosenSmallTilesMatchTiled16` (auto-selected
at batch 1024), `SmallWideSaturatingBatchIsBitIdentical` (batch 2048),
`Sm120TransposedFallbackMatchesTiled16`, and the two selector straddle tests. Armed
breaks: dropping the epilogue's `col >= n` guard turns exactly the shapes with a
partial n tile red (whole-tile shapes stay green). Staging k one past its end turns
exactly the k % 16 != 0 shapes red. Moving the transposed fallback's floor to
`ctas > 128` and the float 16-tile deep-k edge to `k >= 769` (one build) turns red
only the six exact-edge assertions of the two straddle tests (f NC/CN panel at b64,
f/c 64x64x16 at b128, f 16x16x768 b1024); every correctness test stays green.

Still losing in the NN native walk: float 8x8x1024 1.10, float 100x100x64 1.28
(128x128 at 3/4 fill; no better tile measured), float 65x65x8 1.29 (128x128K8, BASE
1.48), float 64x100x512 1.22 (64 wide tile, BASE 1.59), the deep-k float small cells
below 1536/2048 tiles (1.2-2.9, see "Tiles in flight at deep k"), cfloat 32x32x1024 1.07, cfloat 256^3
1.13 and the other large cfloat squares (1.11-1.17, not touched: gemm-7 territory).
