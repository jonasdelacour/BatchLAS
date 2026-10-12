# Local AdaptiveCpp patches for the Metal build

`../build_adaptivecpp_metal.sh` applies these with `git am` on top of AdaptiveCpp
`develop` at `9ec394ef`, on the local branch `batchlas-metal`. They are local only and have
not been sent upstream. Regenerate them from that branch with
`git format-patch 9ec394ef..batchlas-metal -o <this directory>`.

| Patch | Bug fixed | Evidence on the M5 Max (applegpu_g17s) |
| --- | --- | --- |
| 0001 barrier fences | `group_barrier` emitted `threadgroup_barrier(mem_flags::mem_threadgroup)`, which does not order global memory. A work-group barrier with `sub_group` fence scope also emitted a `simdgroup_barrier`, which does not synchronize the work-group. | Now emits `threadgroup_barrier(mem_flags::mem_threadgroup \| mem_flags::mem_device)`, checked with `__ACPP_PRINT_METAL_CODE=1`. A global-exchange stress test reads no stale values either before or after the patch: the old code was racy by the MSL spec, but this hardware did not show the race. |
| 0002 safe math | `MTLCompileOptions` defaults to fast math. | Before: `7.0f/3.0f` gave 2.33333349 and `exp(20.5f)` was 3 ulp off. After: correctly rounded. `ACPP_METAL_FAST_MATH=1` brings back the old behaviour. The library optimization level also changes from `Size` to `Default`. |
| 0003 launch limits | A dispatch above the pipeline's `maxTotalThreadsPerThreadgroup`, or above `maxThreadgroupMemoryLength`, ran silently with undefined results. | 30000 B of local memory at wg=1024 (plus 8192 B of runtime scratch) now raises an error naming the kernel and the byte counts, and wg=2048 raises one naming the 1024 limit. The error is asynchronous: `queue::wait_and_throw()` delivers it to the queue's async handler. |
| 0004 narrow signed i8/i16 | An `sdiv`, `srem` or `ashr` on i8 or i16 was emitted as `as_type<uchar>(int)`: MSL promotes the operands to `int`, and the Metal compiler rejects the size-mismatched cast. LLVM narrows small-range arithmetic to i8 by itself (`lane % 16 / 8`). | The float `small` gemm kernels (`launch_small_batched`) failed MSL compilation with `as_type cast from 'int' to 'uchar' is not allowed`; after the patch all 8 build a pipeline (offline: `llvm-to-metal-tool` + `newLibraryWithSource`). |
| 0005 cross-address-space memmove | MemCpyOpt folds a global->private->local copy pair into one `memmove`; its loop expansion compares a `device` with a `threadgroup` pointer, which MSL rejects. A memmove between distinct non-generic address spaces is now lowered as a memcpy. | The `reg:` gemm kernel with a trivially copyable 16-byte struct failed with `comparison of distinct pointer types ('device void *' and 'threadgroup void *')`; after the patch all 6 build. BatchLAS's `Vec4` copies element-wise anyway, so it does not depend on this patch. |
| 0006 no vectorizers | A `PassBuilder` without a `TargetMachine` runs LoopVectorize at O3, and MetalEmitter has no vector types; the Metal translator now optimizes with `LoopVectorization` and `SLPVectorization` off. | The float `syrk`/`syr2k` triangular-tile kernels failed the JIT with `Unknown LLVM instruction: insertelement`: the correction loop in `triangular_tile_decode` was early-exit vectorized (`<4 x i32>`, `llvm.experimental.cttz.elts`). After the patch all 28 build a pipeline. |
| 0007 `get_num_subgroups` | The Metal libkernel did not define `__acpp_sscp_get_num_subgroups` (`sub_group::get_group_range()`), so the MSL named an undeclared identifier. It is now computed from the work-group size, as in the PTX libkernel. | 8 float `geqrf_cta` kernels failed MSL compilation with `use of undeclared identifier '__acpp_sscp_get_num_subgroups'`. |
| 0008 trunc to i1 | `trunc iN %x to i1` was emitted as a C cast to `bool`, which tests the whole value rather than bit 0: `trunc i64 2` came out true. LLVM forms such truncs from `x & 1` parity tests. | The float `reg:m=128:n=32:k=16:u=1` gemm (the only two-stage `reg` tile that fits Apple's 32 KiB) gave wrong results for every element once k spanned three or more tiles: the stage select `tile_idx & 1` became `(bool)tile_idx`, so tile 2 read the wrong buffer. k=16 and k=32 were exact; after the patch k=48..200 match a double host product to 1e-5. |
| 0009 convergent SIMD-group intrinsics | The public sub-group builtins are `convergent`, but the `__acpp_sscp_metal_{shuffle,broadcast,reduce,scan,collpredicate}_*` declarations they inline to were not. LLVM then unswitched loops on per-lane conditions and ran a `simd_shuffle` on part of the SIMD-group, which reads inactive lanes. | The float tiny `getrf` at n=5..8 (lane bucket 8, shuffle broadcast) returned wrong factors: the rank-1 update's broadcasts sat inside `if (rowid <= j)` in the MSL. After the patch every n=1..32 matches the CTA route. |

Every launch also reserves `work-group size * 8` bytes of threadgroup memory as runtime
scratch, so a kernel can use only `32768 - 8 * wg` bytes of local memory. Every pipeline
measured so far reports `maxTotalThreadsPerThreadgroup = 1024`, including kernels with 512
live floats per thread, which the compiler spills.

The build script also pins the installed `acpp`, `syclcc` and `syclcc-clang` drivers to
Homebrew's `python3`. Their `#!/usr/bin/env python3` otherwise resolves to the Command Line
Tools shim whenever `/usr/bin` precedes Homebrew on `PATH`, and that shim exports
`CPATH=/usr/local/include` and `LIBRARY_PATH=/usr/local/lib` to every compile: a GoogleTest
installed under `/usr/local` then shadows the one BatchLAS fetches, and the tests fail to link.
`make install` rewrites the shebang, so re-run the script (or the `sed` in it) after a reinstall.
