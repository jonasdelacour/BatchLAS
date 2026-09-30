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
