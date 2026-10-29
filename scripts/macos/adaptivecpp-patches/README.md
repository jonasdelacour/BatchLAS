# Local AdaptiveCpp patches for the Metal build

`../build_adaptivecpp_metal.sh` applies these with `git am` on AdaptiveCpp `develop` at `9ec394ef`,
on the local branch `batchlas-metal` (not sent upstream). Regenerate with
`git format-patch 9ec394ef..batchlas-metal -o <this directory>`. The script reuses an existing branch
only if it sits on `9ec394ef` and carries exactly these diffs (`git patch-id --stable`);
`ACPP_REAPPLY=1` recreates it.

Evidence is on the M5 Max (applegpu_g17s).

| Patch | Bug fixed | Reproducer / evidence |
| --- | --- | --- |
| 0001 barrier fences | `group_barrier` fenced only threadgroup memory (and a `sub_group`-scope fence emitted `simdgroup_barrier`) | now `mem_threadgroup \| mem_device`, checked with `__ACPP_PRINT_METAL_CODE=1`; racy by the MSL spec, not observed on this hardware |
| 0002 safe math | `MTLCompileOptions` defaulted to fast math | `7.0f/3.0f` and `exp(20.5f)` now correctly rounded; `ACPP_METAL_FAST_MATH=1` restores fast math. Library opt level `Size` -> `Default` |
| 0003 launch limits | dispatch over `maxTotalThreadsPerThreadgroup` or `maxThreadgroupMemoryLength` ran with undefined results | now an async error (via `wait_and_throw()`) naming kernel and byte counts |
| 0004 narrow signed i8/i16 | `sdiv`/`srem`/`ashr` on i8/i16 emitted a rejected `as_type<uchar>(int)` | float `small` gemm kernels failed MSL compilation; all 8 build |
| 0005 cross-address-space memmove | a `device`<->`threadgroup` `memmove` loop compared distinct pointer types | `reg:` gemm with a 16-byte struct failed; all 6 build (BatchLAS's `Vec4` does not depend on it) |
| 0006 no vectorizers | LoopVectorize/SLP produced vector IR MetalEmitter cannot emit | float `syrk`/`syr2k` triangular tiles failed with `Unknown LLVM instruction: insertelement`; all 28 build |
| 0007 `get_num_subgroups` | `__acpp_sscp_get_num_subgroups` undefined in the Metal libkernel | 8 float `geqrf_cta` kernels failed MSL compilation |
| 0008 trunc to i1 | `trunc iN to i1` emitted as `(bool)x`, testing the whole value | float `reg:m=128:n=32:k=16:u=1` gemm wrong once k spans >= 3 tiles; k=48..200 now match to 1e-5 |
| 0009 convergent SIMD-group intrinsics | `__acpp_sscp_metal_{shuffle,broadcast,reduce,scan,collpredicate}_*` not `convergent`, so loops were unswitched on per-lane conditions | float tiny `getrf` n=5..8 gave wrong factors; n=1..32 now match the CTA route |
| 0010 linear translation time | super-linear StructurizeCFG PHI repair, per-value slot renumbering, unbounded local names | offline `llvm-to-metal-tool` on float `posv_tiny` NC=28: ~90 s -> 1.25 s; `TinyFactorIsBitIdenticalToPotrf` cold 227 s -> 0.7 s. IR comments only with `__ACPP_PRINT_METAL_CODE` or `__ACPP_METAL_IR_COMMENTS` |
| 0011 no re-advise in `alloc_at` | every split re-`madvise`d the (10x RAM) tail as `MADV_FREE_REUSABLE` | `malloc_shared`+`free` 1 KiB: 74 us -> 6 us; `steqr_tests` `MixedDirectionLockstep` passes in 75 s |
| 0012 float memory through `uint` | a float stored as `i32` and reloaded as `float` broke MSL strict aliasing (stale value forwarded) | float tiny `geqrf` left R(j,j) stale; `geqrf_tests_native`, `geqrf_candidates_tests`, `ormqr_tests`, `orgqr_candidates_tests`, `syev_two_stage_tests`, `syev_candidates_tests` now pass |
| 0013 instruction in emitter errors | after 0010, unknown-instruction / wide-integer / indirect-call errors named no instruction | diagnostics only; MSL unchanged |

- Every launch reserves `8 * wg` bytes of threadgroup memory as scratch: a kernel gets
  `32768 - 8 * wg` bytes of local memory. Every pipeline measured reports
  `maxTotalThreadsPerThreadgroup = 1024`.
- The script pins the installed `acpp`, `syclcc`, `syclcc-clang` shebangs to Homebrew's `python3`;
  the Command Line Tools shim would export `CPATH=/usr/local/include` and
  `LIBRARY_PATH=/usr/local/lib`, shadowing BatchLAS's GoogleTest. `make install` rewrites the
  shebang: re-run the script after a reinstall.
