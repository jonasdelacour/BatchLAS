---
name: ptx-codegen-comparison
description: 'Compare cuBLASDx and SYCL-generated device code for the same BatchLAS kernel, extract PTX from build artifacts, identify codegen regressions such as spills or missed unrolling, and validate whether PTX changes correlate with benchmark movement.'
argument-hint: 'Kernel family, benchmark command, and the two variants to compare, for example GEMM cuBLASDx vs SYCL on CUDA backend'
user-invocable: true
disable-model-invocation: false
---

# PTX Codegen Comparison

cuBLASDx was deleted from the tree (last at commit `eeacaaa9`), so compare two variants you can pin:
a native family against `vendor`, or two native spellings via `BATCHLAS_<OP>_ROUTE=<spelling>`. The
question is whether a runtime gap comes from worse generated code, a launch/configuration mismatch, or
later lowering (ptxas).

## Rules

1. Fixed benchmark case, `--warmup=5`; change only the variant. Measure at saturation (agent-guide §10).
2. Confirm both runs hit the intended kernel (same tile shape, same aligned vs generic path) with
   `BATCHLAS_SELECT_TRACE=1` before reading PTX. Mismatched variants: fix the setup, conclude nothing.
3. Extract device code from the artifacts the benchmark actually ran; use absolute paths.
4. Compare structure, not only register declarations: PTX from different pipelines can be at
   different lowering stages. PTX is evidence, not the whole answer.
5. Fix one codegen issue at a time; re-extract PTX and rerun the benchmark after each change. Keep a
   change only if both the PTX and the runtime move the right way.

## Procedure

1. **Reproduce the gap**, recording command, route pins, size and time:

   ```bash
   BATCHLAS_GEMM_ROUTE=native build/benchmarks/gemm_benchmark 4096 4096 4096 32 --backend=CUDA --type=float --warmup=5
   BATCHLAS_GEMM_ROUTE=vendor build/benchmarks/gemm_benchmark 4096 4096 4096 32 --backend=CUDA --type=float --warmup=5
   ```

   GEMM selection is `src/ops/gemm/{choice.hh,gemm.cc}`; register-tiled kernels are
   `src/ops/gemm/kernels/register_tiled_common.hh` and `register_launchers.hh`. Pin one kernel with
   e.g. `reg:m=128:n=32:k=32:u=1`, `wide:m=64:n=64:k=16`, `tiled`, `vendor`.

2. **Inspection build** (a separate tree; GPU-only trees are never the gate):

   ```bash
   cmake -S . -B build-ptx -DBATCHLAS_ENABLE_CUDA=ON -DBATCHLAS_BUILD_BENCHMARKS=ON \
     -DBATCHLAS_KEEP_CUDA_INTERMEDIATES=ON -DBATCHLAS_CPU_TARGET=none
   ```

   `BATCHLAS_KEEP_CUDA_INTERMEDIATES` adds `--save-temps` and `-Xcuda-ptxas -v` to the SYCL CUDA
   device link (`cmake/BatchLASDetectSYCL.cmake`).

3. **Extract PTX.** SYCL device code is AOT-compiled to sm_89 at the shared-library device link and
   embedded as CUDA fatbins in `build/src/libbatchlas*.so`. `scripts/check_device_calls.py` shows the
   extraction: find each fatbin by its magic header (`fatbins()`), write it out, `cuobjdump -ptx` it
   (`ptx_of()`); pipe through `c++filt`. Fallback: lower the preserved NVPTX bitcode from step 2
   (`find build-ptx -name '*sycl-nvptx64-nvidia-cuda*'`) with
   `/opt/dpcpp-cuda/bin/llc -march=nvptx64 -mcpu=sm_89 -o out.ptx <file>`. Isolate the matching
   `.entry` body before comparing.

4. **Compare high-signal structure** before any full diff:

   ```bash
   grep -cE 'ld\.local|st\.local|\.local ' kernel.ptx   # spills / local arrays
   grep -c 'ld.global.v4' kernel.ptx                    # vectorised global loads
   grep -cE 'st\.shared|ld\.shared' kernel.ptx          # shared staging
   grep -c 'bar.sync' kernel.ptx
   grep -c 'fma.rn.f32' kernel.ptx                      # math core
   grep -c 'nounroll' kernel.ptx                        # unroll blockers
   ```

   Registers and spills per kernel at the real link: `scripts/register_probe.sh` (gate it with the
   per-sub-partition register rule, agent-guide §11).

5. **Diagnose.**

   | PTX finding | Next step |
   | --- | --- |
   | Accumulator tile written to `.local` and reloaded | compile-time indexed accumulators; return by value, never `T&` out-params (§11) |
   | Fewer straight-line vector loads / shared stores | make aligned packet-load loops compile-time bounded so they unroll |
   | Same FMA count, worse staging | load/store scheduling, spills or final lowering |
   | Similar PTX, large runtime gap | SASS (`cuobjdump -sass`) or `ncu` (the `nvidia-kernel-profiling` skill): registers, occupancy, local traffic, issue efficiency |

## Done when

The gap reproduces with fixed command and pins; matching variants are confirmed; PTX for both comes
from real artifacts; the structural differences are named; at least one source-level issue is fixed or
ruled out and a rerun shows whether it mattered; a remaining gap has SASS or profiler work named as the
next step.
