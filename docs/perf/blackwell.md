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
  256 0.99 (non-winners).

sm_89 (and `cuda_cc` 0) keeps 256 for both types.

### potrf LPanel native tier cap

The vendor-free walk (`native_tier_preferred`) had no LPanel order cap. On sm_120 float
LPanel vs Blocked (retuned constants, ms): 288 1.66 vs 1.97, 320 2.16 vs 2.44, 352 1.91
vs 1.55, 384 2.31 vs 1.77, 512 4.47 vs 3.49. So float LPanel stops at 320, and only when
Blocked can take the shape. cfloat LPanel matches or beats Blocked to its ceiling (256
2.18 vs 3.01, 288 2.09 vs 2.14, 320 2.60 vs 2.58, 368 1.92 vs 2.11), so it is not
capped. This reverses the diagnosis, which was measured on the scalar kernel.

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
nb 64 through n=512 and 128 above, W 64. cfloat (96/32) did not move: 64/64 was 0.62 vs
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
