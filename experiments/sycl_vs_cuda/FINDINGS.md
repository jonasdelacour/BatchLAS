# Can SYCL match CUDA for batched GEMM?

Yes. Measured on this machine, a single SGEMM kernel body compiled by both
nvcc and DPC++ runs within 1.3% either way, and within 1% of cuBLAS's own
FP32 SGEMM.

The long-standing belief that "SYCL kernels cannot match the same kernel
written in CUDA" does not survive a direct test. It was never directly tested:
every in-tree comparison pitted a *BatchLAS SYCL kernel* against *cuBLAS*,
which confounds two independent variables -- the language and the kernel
design. This experiment separates them.

## Method

`sgemm_body.h` contains a canonical SGEMM as a macro: 128x128 block tile,
K-step 8, 256 threads, 8x8 accumulators per thread, with the standard split
row/column bands (`ty*4` and `64+ty*4`) that make the shared-memory fragment
loads bank-conflict free.

`sgemm_cuda.cu` and `sgemm_sycl.cpp` both `#include` that header and expand
the same macro. The only differences are how a thread learns its coordinates
(`threadIdx` vs `nd_item::get_local_id`), how it reaches shared memory
(`__shared__` vs `local_accessor`), and how it barriers. Every arithmetic
operation, every shared-memory index, and every load width is textually
identical.

Both programs print a checksum of C computed the same way. They agree exactly
(`85879.051714`), so the two builds demonstrably compute the same thing.

Hardware: RTX 4090 (sm_89), CUDA 13.2, DPC++ from `/opt/dpcpp-cuda`
(`clang 22.0.0git`, intel/llvm). All runs pinned to one GPU.

## Result 1: the generated code is the same

Both toolchains compile the kernel to the same SASS inner loop:

| | CUDA (nvcc -O3) | SYCL (DPC++ -O3, AOT sm_89) |
| --- | ---: | ---: |
| FFMA in main loop | 512 | 512 |
| `LDS.128` | 32 | 32 |
| `BAR.SYNC` | 2 | 2 |
| FFMA per `LDS.128` | 16 | 16 |
| registers | 115 | 113 |
| spill stores / loads | 0 / 0 | 0 / 0 |

The one difference is in the global-load prologue: nvcc emits 18 `LDG.E.128`
where DPC++ emits 1 `LDG.E.128` plus 2 narrower `LDG.E`, i.e. nvcc unrolls and
vectorizes the global staging more aggressively. At these shapes the kernel is
compute-bound with a high L2 hit rate, so this does not show up in the timings
below -- but it is a real difference and is the first place to look if a
future kernel is global-load bound.

## Result 2: the runtimes are the same

TFLOP/s, `2*m*n*k*batch`, timed over 30 iterations after 10 warmup.

| Shape | CUDA | SYCL | cuBLAS FP32 (pedantic) | cuBLAS TF32 |
| --- | ---: | ---: | ---: | ---: |
| 512^3, batch 512 | 43.73 | **43.58** | 43.88 | 78.04 |
| 1024^3, batch 64 | 47.10 | **47.16** | 47.50 | 84.11 |
| 256^3, batch 1024 | 32.19 | **31.78** | 39.22 | 39.44 |

SYCL lands at 99.3%, 100.1% and 98.7% of the CUDA build. Both hand-written
kernels are within 1% of cuBLAS at 512^3 and 1024^3.

At 256^3 both hand-written kernels fall to ~82% of cuBLAS. That is a
tile-selection problem, not a language problem: a 128x128 block tile leaves
only 2x2 tiles per 256x256 matrix, so the tail is coarse. cuBLAS switches to a
smaller tile. The SYCL and CUDA builds degrade *identically*, which is the
point.

## Result 3: the 80 TFLOP/s target is a TF32 number, not an FP32 one

This reframes the whole effort. The RTX 4090's ~82.6 TFLOP/s FP32 figure is an
FMA-issue peak that no real GEMM sustains. cuBLAS's own strict-FP32 SGEMM tops
out at 43.9-47.5 TFLOP/s here, and `cublasGetMathMode` returns
`CUBLAS_DEFAULT_MATH` (0) -- the default path is *not* using tensor cores, and
default and pedantic timings are identical.

The ~78-84 TFLOP/s that looks like "peak" is cuBLAS with
`CUBLAS_TF32_TENSOR_OP_MATH`: tensor cores at reduced precision (tf32 keeps 10
explicit mantissa bits against fp32's 23).

So there is no 80 TFLOP/s to chase with FFMA in any language. The honest FP32
ceiling is ~44-47 TFLOP/s, and the hand-written kernel already reaches it.

## Result 4: tensor cores are reachable from portable SYCL

`tf32_smoke.cpp` compiles `joint_matrix` with `precision::tf32` for sm_89. The
generated PTX contains 64 real
`mma.sync.aligned.row.row.m16n16k8.f32.tf32.tf32.f32` instructions and the
results are correct.

That probe does no shared-memory staging and has no data reuse, so its
throughput would be meaningless and is not reported. It establishes
reachability only. **Whether a *tuned* SYCL joint_matrix GEMM can reach
cuBLAS's ~78 TFLOP/s is not yet measured** and is the obvious next experiment.

## A measurement hazard

Summing SYCL event-profiling intervals over *queued* submissions does not
measure kernel time. With 30 submissions in flight, the summed
`command_start`..`command_end` interval reported **19.836 ms** for a kernel
whose true time is **3.15 ms** -- a 6.3x inflation, because the interval
includes time the command spent waiting in the queue. With a single submission
in flight the same measurement reports 3.261 ms and agrees with wall clock.

Any benchmark that enqueues many iterations and sums SYCL event intervals will
understate SYCL throughput badly. This is worth auditing wherever in-tree
numbers compare a SYCL path timed this way against a vendor path timed with
CUDA events.

## What this means for BatchLAS

BatchLAS's best in-tree SYCL GEMM reaches ~17-21 TFLOP/s at 512^3 batch 512
(`output/gemm_steady_phase*.md`). The kernel here reaches 43.6 on the same
shape, in the same language, on the same hardware, in about 200 lines.

The gap is therefore in the kernel design, not in SYCL. Reproduce with:

```
./build.sh
CUDA_VISIBLE_DEVICES=0 ./build/sgemm_cuda  --m 512 --n 512 --k 512 --batch 512
CUDA_VISIBLE_DEVICES=0 ./build/sgemm_sycl  --m 512 --n 512 --k 512 --batch 512 --iters 1 --warmup 20
```
