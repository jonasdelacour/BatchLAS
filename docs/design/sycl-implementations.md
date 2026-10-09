# Two SYCL implementations: DPC++ and AdaptiveCpp {#design_sycl_implementations}

> **Status:** plan · 2026-10-08 · probe run on threadripper02 (4x RTX PRO 6000, sm_120, CUDA 13.2,
> AdaptiveCpp 25.10.0 at `/opt/adaptivecpp-25.10.0-cuda13.2`, LLVM 20 plugin). Nothing is
> implemented yet.

Goal: the same source tree configures and builds with either intel/llvm DPC++ (`/opt/dpcpp-cuda`)
or AdaptiveCpp (`acpp`). Two trees on one box then A/B the same tests and benchmarks for
correctness and speed. Other SYCL implementations are out of scope. They should not be
precluded either.

## 1. Probe result (2026-10-08)

A scratch copy of `1f0b6d84` was compiled with `acpp --acpp-targets=generic -O2 -std=c++20`, using
the DPC++ tree's `-I`/`-D` flags. The patches were regex shims applied to the copy, not real fixes.

| Stage | Result |
| --- | --- |
| CMake configure, `CMAKE_CXX_COMPILER=acpp` | rc=0 but unusable: `-fsycl` is still added and acpp rejects it fatally. CUDA detection goes through DPC++'s `sycl-ls`. |
| Library TUs, pristine source | 23 of 109 compile. |
| Library TUs after shim layers L0+L1 | 107 of 109. Open: `ormqr_cta.cc` (`sycl::vec<std::complex>`) and `ritz_values.cc` (`joint_reduce` of `std::complex`). |
| Test TUs | 82 of 86 compile. |
| `cuda:sm_120` (SMCP) flow | Unusable: clang 20's CUDA wrapper rejects the CUDA 13.2 headers, even on an empty file. **Generic (SSCP) is the only flow on this install.** |
| Compile wall, -P32, no ccache | acpp 50 s vs DPC++ 62 s. The acpp `.so` link takes 1.1 s because there is no device link; kernels are JIT-compiled at first launch instead. |
| `examples/consumer` gemm | Max error 0 under Auto and under every gemm pin (`tiled`, `direct`, `small`, `reg:m=128:n=128:k=8:u=1`, `vendor`). SSCP JIT and the cuBLAS stream interop both work. |
| Test binaries | 39 of 77 fully green (29 before `-fcx-limited-range`). The failure causes are listed in §4. |

## 2. What is DPC++-specific today

Most of the DPC++ dependence already sits behind a handful of choke points. `sycl::ext::*` is almost
unused (one benchmark uses `chunked_partition`). There is no `host_task`, no `interop_handle`, no
specialization constants and no `sycl::stream`.

| Construct | Where | Count |
| --- | --- | --- |
| `-fsycl*`, `-Xsycl-target-*`, `sycl-ls` parsing, the alias probe, `native_cpu`/`spir64_x86_64` | `cmake/BatchLASDetectSYCL.cmake` (772 lines), `BatchLASCompilerBootstrap.cmake`, `BatchLASOptions.cmake` | 1 module + 2 |
| Flags reach targets through three helpers | `batchlas_apply_object_options`, `batchlas_configure_binary_target` (`cmake/BatchLASTargetHelpers.cmake`), `batchlas_configure_component` (`src/CMakeLists.txt`) | 3 choke points |
| `queue::get_backend()`, `backend::ext_oneapi_cuda/hip`, `get_native(queue)` | `src/queue.hh` (`Queue::native_handle`), `src/linalg-impl.hh` (`LinalgHandle::setStream`, about 46 call sites through one helper), `src/util/queue-impl.cc` | 3 files; 86 TUs fail on them via the header |
| `ext_oneapi_submit_barrier`, `ext_oneapi_memcpy2d` | `queue-impl.cc`, `netlib_lapack.cc`, `matrix.cc` | already macro-guarded, with fallback |
| `mem_advise` with UR flags | `sycl-util-impl.cc` | already guarded (`SYCL_IMPLEMENTATION_ONEAPI`) |
| `kernel_bundle` / `get_kernel_id` | `get_kernel_max_wg_size` (`linalg-impl.hh`), `stedc_merge_cta.cc` | 2 helpers |
| `__SYCL_DEVICE_ONLY__` + `__NVPTX__` dispatch, `__nvvm_*` / `__spirv_*` / AMDGCN intrinsics | `src/extensions/sg_partition/` (4 backends) | 1 directory, used by 17 files |
| `[[intel::max_work_group_size, min_work_groups_per_cu]]` | `BATCHLAS_LAUNCH_BOUNDS` (`src/queue.hh`), used in 4 kernels | 1 macro |
| Inline PTX `asm` (L2 prefetch loads) | `src/sycl/gemm/register_128x128.hh` | 1 file, `#if`-guarded |
| `[[sycl::reqd_sub_group_size(32)]]` | spelled directly | 26 files (acpp warns and ignores it) |
| oneDPL `<oneapi/dpl/random>` in kernels | `matrix.cc`, `lanczos.cc`, `syevx_lobpcg.cc` | done (C10): `src/util/philox.hh`, oneDPL is no longer a dependency |
| `info::device::version` parsed as `"8.9"` | `Device::cuda_compute_capability` (`queue-impl.cc`); feeds the table key `sm_<cc>` in `select::describe` | 1 site |
| PTX inspection: fatbin magic, `cuobjdump`, banned `__spirv_` callees | `scripts/check_device_calls.py`, `register_probe.sh`, `.github/skills/ptx-codegen-comparison` | 3 tools |
| `/opt/dpcpp-cuda`, `libsycl.so.9`, `-fsycl` | `examples/consumer/`, `consumer_test.sh`, CI, `scripts/rocm_syntax_check.sh` | about 6 files |
| Compiler provenance | `benchmarks/benchviz/store.py` (warns on `IntelLLVM`, greps `nvptx`), `compare.py` drops `compiler_id` | 2 files |

## 3. Compile-level differences (fix once, both compilers)

All of these are DPC++ extensions or acpp gaps where a spelling legal in both exists.

| # | Construct | acpp error | Portable form |
| --- | --- | --- | --- |
| C1 | `q.get_backend()` | no member | `q.get_device().get_backend()` |
| C2 | `backend::ext_oneapi_cuda`, `get_native(queue)` | no member | `interop::` helper, see §5.2 |
| C3 | `get_kernel_id` / `kernel_bundle` | no member | helper: DPC++ keeps the query; acpp returns the device `max_work_group_size` |
| C4 | `joint_reduce(g, acc.begin(), acc.end(), …)` | no `__acpp_leader_reduce` overload for accessor iterators | pass raw pointers `&acc[0]` (6 TUs) |
| C5 | `group_broadcast` of a struct (`larfg_result`, `Cx<double>`) | SSCP broadcasts scalars only | one `bl::group_broadcast` that splits a trivially copyable struct into words; 22 call sites, most behind `include/batchlas/util/group-invoke.hh` |
| C6 | ternary between an accessor iterator and `T*` | incompatible operands | `&smem[0]` (1 TU) |
| C7 | `sycl::vec<std::complex<T>, N>` | `vec` static assert | plain arrays (`ormqr_cta.cc`) |
| C8 | `joint_reduce` / `reduce_over_group` of `std::complex` with `sycl::plus` | no overload | reduce `real` and `imag` separately (`ritz_values.cc`; also check `math-helpers.hh:127`, `sycl::vec<R,3>`) |
| C9 | `select_from_group` of a non-scalar | no overload | same word-split helper as C5 (`device_blas_tests`) |
| C10 | oneDPL random needs `sycl::isequal`, `sycl::tanpi` | missing in acpp | **done**: `src/util/philox.hh` (Philox4x32-10) serves the 3 sites; the oneDPL CMake search and `ONEDPL_ROOT` are gone |
| C11 | `ext_oneapi_submit_barrier` in `tests/linalg_layer_tests.cc` | no member | guard like the library sites |

Exit criterion: all of `src/`, `tests/`, `benchmarks/` and `tools/` compile under both compilers.
The DPC++ PTX-call gate (`device_calls_tests`) stays green, and the DPC++ route diff is empty.

## 4. Runtime differences (behaviour, not spelling)

| # | Symptom on acpp (probe) | Cause | Plan |
| --- | --- | --- | --- |
| R1 | 17 binaries: `ptxas fatal: Unresolved extern function '__mulsc3'/'__muldc3'` | `std::complex` `*` lowers to Annex G libcalls; SSCP ships no device definition | Ship device definitions of `__mulsc3`/`__muldc3`/`__divsc3`/`__divdc3` for the acpp build, so both builds keep identical Annex G semantics. `-fcx-limited-range` cleared all 17, but changes NaN/Inf results and would make the A/B unequal. Keep it only as a measured variant. |
| R2 | gemm, trmm/symm/syrk candidates, herk: launch fails, CUDA error 1 | acpp reports `local_mem_size` = 48 KiB and never calls `cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES)` | Every `can_run` that admits more than 48 KiB must read the device SLM budget (`select::describe`'s `slm_budget`), not a constant. Then the acpp build routes around large-SLM families instead of crashing. Separately, ask upstream for the opt-in (patch the CUDA backend's launch). |
| R3 | One bad launch kills the whole gtest binary | acpp's default async handler calls `std::terminate` | Install an async handler on `QueueImpl` that rethrows on `wait_and_throw` (both builds). |
| R4 | `cuda_compute_capability()` is 0, so the table key `sm_120` becomes `gpu` (borrowed correctly by luck) | acpp's `info::device::version` is `"sm_120"`, not `"12.0"` | Parse both forms, or ask CUDA directly (`cuDeviceGetAttribute` on the native device ordinal). |
| R5 | Hangs: `SgPartitionDivergence/0.MaskedSG32`, bdsdc, syev_cta, syev_blocked, syev_two_stage, gesvd, sytrd_blocked | Generic SSCP is single-pass, so `__SYCL_DEVICE_ONLY__ && __NVPTX__` is never true and `sg_partition` drops to the unmasked `GenericBackend`, which deadlocks under divergence | §5.3: an SSCP backend for `sg_partition`. |
| R6 | Out-of-order queues segfault (ormqr/orgqr candidates, symm) | probe shim took the stream from the in-order executor | §5.2: vendor calls go through `AdaptiveCpp_enqueue_custom_operation`. |
| R7 | `Queue::native_handle()` returns null | guarded by `SYCL_EXT_ONEAPI_BACKEND_CUDA` | §5.2 |
| R8 | `select_tests` hangs in a death test | `fork()` after the acpp runtime started its threads | Use `GTEST_FLAG(death_test_style)="threadsafe"`, or skip death tests on acpp. |
| R9 | Wrong answers: cdouble potrf/getrf CTA, orgqr orthogonality 5.8e-6 vs 3.8e-6, norm, cond, lanczos, syr2k_candidates, geqrf | not attributed: some fail on DPC++ on this box too, and there was no same-box DPC++ baseline | Attribute only after R1-R8, against a same-box DPC++ run, comparing failing names. |
| R10 | Silent: `BATCHLAS_LAUNCH_BOUNDS` empty; the `register_128x128` PTX prefetch path is compiled out; `reqd_sub_group_size` ignored | single-pass SSCP has no NVPTX device pass | §5.3 for the prefetch. No min-blocks-per-SM equivalent exists: the JIT emits `maxntid` from the launch size but no `minnctapersm`. Accept it, measure the loss (§7), and record it as an implementation difference, not a bug. |
| R11 | Floating point: acpp rounds `sqrt` and division correctly and uses `-ffp-contract=fast`; DPC++ here is `-ffp-model=precise` | different defaults | Match explicitly in both trees and record the flags in benchviz provenance. Any accuracy delta must survive matched flags. |

## 5. Design

### 5.1 Build: one cache variable, two branches

- `BATCHLAS_SYCL_IMPL = AUTO | DPCPP | ACPP`. AUTO picks ACPP when the compiler is the `acpp`
  driver (or `find_package(AdaptiveCpp)` succeeds with `CMAKE_CXX_COMPILER` unset), and DPCPP
  otherwise. The bootstrap stops forcing a DPC++ `clang++` when ACPP is selected.
- Split `BatchLASDetectSYCL.cmake` into `BatchLASSyclDpcpp.cmake` (today's file, unchanged in
  behaviour) and `BatchLASSyclAcpp.cmake`, which:
  - sets `ACPP_TARGETS=generic` by default, with no `-fsycl*` and no `--cuda-path`;
  - gets NVIDIA presence and architecture from `nvidia-smi` or `acpp-info`, not `sycl-ls`;
  - sets `BATCHLAS_HAS_CPU_TARGET` from the OpenMP backend;
  - makes acpp the compiler for every SYCL target. Use `acpp` as `CMAKE_CXX_COMPILER` plus
    `--acpp-targets=generic` on the same INTERFACE option libraries, rather than
    `add_sycl_to_target`. The probe shows this works, and it keeps the object-library and
    `$<TARGET_OBJECTS>` model intact.
- The three target helpers are the only call sites that change.
- The device-link options (`BATCHLAS_SYCL_LINK_JOBS`, FTZ, line info, keep-intermediates, the
  native_cpu flags) become DPC++-only.
- `-mllvm -pragma-unroll-threshold` is accepted by acpp for stage 1. Whether the SSCP JIT honours
  unrolling must be checked on the JIT-ed PTX.
- Generated `backend_config.h` gets `BATCHLAS_SYCL_IMPL_DPCPP` / `_ACPP` (`#cmakedefine01`). Source
  code tests those, never `__ACPP__` or `SYCL_IMPLEMENTATION_ONEAPI` directly.
- Package config: record `BatchLAS_SYCL_IMPL` and the targets.
  `BATCHLAS_REQUIRE_MATCHING_COMPILER` stays: an acpp library cannot be consumed by a DPC++
  project and vice versa. `examples/consumer` and `consumer_test.sh` take their flags from the
  package (`BatchLAS_SYCL_COMPILE_OPTIONS`) instead of hard-coding `-fsycl` and `libsycl.so.9`.
- Presets: `acpp` and `acpp-tests`, mirroring `dev-tests` with
  `CMAKE_CXX_COMPILER=/opt/adaptivecpp/bin/acpp`.
- ccache: the launcher works unchanged. Measure the hit rate on a second acpp tree before claiming it.

### 5.2 Runtime: one interop seam {#sycl-impl-runtime-interop-seam}

One new internal header, `src/sycl/impl.hh`, holds every implementation difference above the
kernel level. Nothing outside it names an implementation. It provides:

| Helper | DPC++ | acpp |
| --- | --- | --- |
| `impl::backend_of(const sycl::queue&)` | `q.get_backend()` | `q.get_device().get_backend()` |
| `impl::is_cuda(backend)` | `ext_oneapi_cuda` | `cuda` |
| `impl::run_native(Queue&, F(cudaStream_t))` | today's pattern: `get_native<ext_oneapi_cuda>(q)`, host call, `create_event_after_external_work` | `q.submit([&](handler& h){ h.AdaptiveCpp_enqueue_custom_operation([&](interop_handle& ih){ f(ih.get_native_queue<backend::cuda>()); }); })` |
| `impl::kernel_max_wg_size<K>(dev)` | kernel bundle query | device `max_work_group_size` |
| `impl::cuda_cc(dev)` | parse `"8.9"` | parse `"sm_89"`, or the driver attribute |
| `impl::submit_barrier(q)` | `ext_oneapi_submit_barrier` | in-order: last event; out-of-order: `ACPP_EXT_QUEUE_WAIT_LIST` |

- `LinalgHandle::setStream` moves inside `run_native`: every vendor call becomes
  `impl::run_native(ctx, [&](auto s){ handle.setStream(s); cublasXxx(...); })`. That covers about
  46 call sites, mechanical but touching every vendor entry point. The DPC++ instantiation must
  stay byte-for-byte the same pattern (route diff plus timing spot check).
- The NETLIB host path needs only a CPU `sycl::device`, and acpp's OpenMP backend provides one.
  `submit_host_task` is already synchronous and needs no change beyond `impl::submit_barrier`.

### 5.3 Device: target dispatch at JIT time {#sycl-impl-device-target-dispatch}

- SSCP compiles device code once, target-agnostically. The NVPTX choice moves from the
  preprocessor to the JIT:
  `AdaptiveCpp_jit::compile_if(reflect<compiler_backend>() == compiler_backend::ptx, …)`
  (`hipSYCL/glue/llvm-sscp/jit-reflection/queries.hpp`).
- `sg_partition` gains a fifth backend, `SscpBackend`. Its `NvptxBackend` member functions are
  reached through `compile_if`, and the `GenericBackend` is kept only for the host pass.
- **Open, settle first (spike S1):** whether `__nvvm_shfl_sync_*`, `__nvvm_vote_ballot_sync` and
  `__nvvm_activemask` survive the stage-1 frontend when the target is not nvptx. If they do not:
  - declare them as `extern "C"` and resolve them in a small NVVM bitcode library linked at JIT
    time (AdaptiveCpp's own `libkernel/sscp/builtins/ptx` does this); or
  - fall back to a fully converged formulation that uses only SYCL collectives on the whole sub-group.

  The second option is correct but must be judged by the real-kernel A/B rule (steqr/syev_cta,
  n = 5..16, batch 16384), never by microbenchmark.
- The same mechanism carries the `register_128x128` L2-prefetch `asm` and, if possible, launch
  bounds. If neither is expressible, it remains an implementation difference recorded in §7.
- `BATCHLAS_LAUNCH_BOUNDS`, `reqd_sub_group_size` and the device-only macros go into one
  `src/sycl/kernel_attrs.hh`, so that the 26 files using `reqd_sub_group_size` spell a macro.
  The acpp build defines it empty and asserts sub-group size 32 at queue creation.

## 6. Correctness A/B

- Each implementation has its own failure ledger: `tests/known-failures.txt` (DPC++, unchanged)
  and `tests/known-failures-acpp.txt`. CI and `scripts/ctest_gpus.sh` select the ledger from
  `BATCHLAS_SYCL_IMPL`. Compare failing **names**, as for DPC++.
- **Gate for "acpp supported":** the acpp tree's failing set is a subset of the DPC++ failing set
  on the same box, plus entries in the acpp ledger, each with a cause from §4.
- **Cross-implementation accuracy:** the accuracy harnesses (`steqr_accuracy`,
  `eigensolver_accuracy`, `orthogonality_accuracy`) run in both trees on identical seeds and graded
  inputs, with FP flags matched (R11). A new `scripts/impl_diff.py` reports per-cell residual ratios.
  This complements the unit tests, which cannot see a 1.5x residual change.
- Route parity: `scripts/route_diff.sh` across the two trees. A route that differs is either an SLM
  budget difference (R2, expected) or a bug.
- `device_calls_tests` is DPC++-only. The acpp equivalent dumps the JIT-ed PTX
  (`ACPP_S2_DUMP_IR_FINAL`, or the CUDA driver cache) and runs the same banned-callee and
  required-kernel checks. It also covers R1-style libcalls. Not needed before phase 4.

## 7. Performance A/B protocol

On top of the measurement rules in `docs/developer/agent-guide.md` §10:

- **JIT and adaptivity.**
  - `ACPP_APPDB_DIR` is set per campaign and cleared on driver or acpp change.
  - `ACPP_ADAPTIVITY_LEVEL` is pinned and recorded. Report level 1; level 2 is a separate arm.
  - Do throwaway passes until acpp stops printing its "new binaries JIT-compiled" warning.
  - Level 2 specializes invariant kernel arguments after about 1024 launches. Benchmarks that
    reuse one shape would flatter acpp against AOT DPC++, so record the invocation count.
- **Launch path.** In-order queues (already the default), `ACPP_RT_SCHEDULER=direct`, and coarse-grained events as a
  recorded variant. Launch latency matters below saturation; compare algorithms at saturation only.
- **Provenance.** benchviz `provenance.builds[]` gains `sycl_impl`, `sycl_impl_version`, the
  compiler `--version` string and the acpp env knobs above. `compare.py` must carry
  `compiler_id`, `sycl_impl`, `fp_model` and `sycl_targets` into comparisons; today it drops them.
  The IntelLLVM `-ffp-model` warning in `store.py` gets an acpp counterpart (flags in R11).
- **Tables.** Both trees read the same `tuned/*.sm_120.txt` at first. Tables are keyed by device,
  not implementation. If the A/B shows rankings that invert per implementation (expected near
  launch-bound and SLM-bound cells), add an optional implementation suffix to the table key with
  the same borrowing rule. Decide only after phase 5 data.
- **First comparison set:** gemm (float and double, NN/TN, strided `ld`, beta=1), potrf, getrf,
  geqrf, syev_cta, steqr_cta and stedc at batch >= 128, n from the CTA range up to 1024. This
  covers the register-tile kernels, the sub-group CTA kernels (R5 and R10) and the SLM-heavy
  kernels (R2).

## 8. Phases

| Phase | Content | Exit criterion | Size |
| --- | --- | --- | --- |
| S1 spike (first) | Can `__nvvm_*` be reached from SSCP stage 1, via `compile_if` or a bitcode library? Do `__mulsc3` definitions resolve at JIT? Does SSCP honour `#pragma unroll` at the 262144 threshold? | Three yes/no answers with a tiny kernel each; decides §5.3 | 1 day |
| P1 build | §5.1: `BATCHLAS_SYCL_IMPL`, split detect module, three helpers, presets, package config, consumer | `cmake --preset acpp-tests` configures; DPC++ configure output is identical before and after (diff the flags) | about 500 CMake lines |
| P2 compile | §3 C1-C11, `impl.hh`, `kernel_attrs.hh`, oneDPL RNG replaced | everything compiles under both; DPC++ route diff empty; `device_calls_tests` green | about 40 files |
| P3 runtime | §4 R1-R4, R6-R8 | acpp: no aborts or segfaults; failing names attributed | — |
| P4 sub-groups | §5.3 `SscpBackend`, prefetch path | `sg_partition_tests` green on acpp; no hangs in CTA eig tests; steqr/syev_cta A/B within the noise of the DPC++ build, or the loss recorded | the riskiest phase |
| P5 A/B | §6 + §7: ledgers, `impl_diff.py`, benchviz provenance, first comparison set | a `docs/perf/sycl-implementations.md` page with saturated grids for both | — |
| P6 CI | GPU-less acpp configure and compile job (`--acpp-targets=generic` needs no GPU at build time); full gate stays local | acpp build is red in CI when a DPC++-only construct leaks in | small |

Each phase is its own PR. P1 and P2 must leave the DPC++ build bit-identical in behaviour: the
route diff is empty and the kernel PTX hashes are unchanged for every TU that P2 did not touch.

## 9. Decisions needed from the maintainer

1. **Complex multiply semantics (R1).** Recommended: ship device `__mulsc3`-family definitions so
   both builds keep Annex G semantics. The alternative, `-fcx-limited-range` in both trees, is a
   numerics change to the DPC++ build too.
2. **SLM above 48 KiB on acpp (R2).** Recommended: route around it through `can_run` now and
   pursue the upstream opt-in in parallel. Patching our acpp install would make results
   unreproducible elsewhere.
3. **Vendor libraries in the acpp build.** Recommended: keep cuBLAS/cuSOLVER through `run_native`,
   so that vendor-routed cells A/B the native kernels identically. The vendor-free acpp tree
   (`-DBATCHLAS_ENABLE_VENDOR_BLAS=OFF`) is the native-only comparison.
4. **Per-implementation tuning tables.** Defer until the phase 5 data shows inversions.
