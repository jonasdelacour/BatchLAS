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
`tiny_window = tiny_native = sm_120 && single precision && n <= tiny_max_n`. fp64 was
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

nrhs=1 is 1-3% slower, since it has only one column to spread. On sm_120 the
in-library register counts exceed the sm_89 table: float Trans nrhs 5..8 uses 86
registers, so the sm_89 row handed out a 768-lane launch that the driver refused
(`FusedGetrsLaunchHoleAt48KiB`). On sm_120 the cap reads a probed row. An sm_89
compile of the TU moves no kernel past its row.

Vendor-present window, added on sm_120 only: float nrhs <= 8 at n >= 256 (the
brackets are n=128 nrhs 8 at 0.63 vs 0.57 ms, a loss, and n=256 at 0.47 vs 0.66, a
win); cfloat nrhs <= 4 at n >= 128 (n=64 0.93 vs 0.85 loses, n=128 0.69 vs 0.86 wins)
and nrhs <= 8 at n >= 512 (n=256 1.38 vs 0.93 loses, n=512 1.20 vs 1.40 wins).

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
nrhs=4.

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
