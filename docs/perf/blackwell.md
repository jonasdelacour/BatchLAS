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
