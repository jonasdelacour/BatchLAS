# Two SYCL implementations: DPC++ and AdaptiveCpp {#design_sycl_implementations}

> **Status:** current · checked 2026-10-21 · Linux acpp measured 2026-10-09 on threadripper02
> (4x RTX PRO 6000, sm_120, CUDA 13.2, AdaptiveCpp 25.10.0 at `/opt/adaptivecpp-25.10.0-cuda13.2`,
> LLVM 20 plugin); Metal measured 2026-10-13 on an Apple M5 Max.

One source tree builds with intel/llvm DPC++ (`/opt/dpcpp-cuda`) or AdaptiveCpp (`acpp`, generic
SSCP target, kernels JIT-compiled at first launch); a third configuration is acpp with its Metal
backend on macOS ([AdaptiveCpp Metal](#sycl-impl-metal)). Measurements: @ref perf_sycl_implementations.
Other SYCL implementations are out of scope but not precluded.

## Design {#sycl-impl-design}

### Build: one cache variable, two detect modules {#sycl-impl-build}

- `BATCHLAS_SYCL_IMPL = AUTO | DPCPP | ACPP` (`cmake/BatchLASOptions.cmake:13-51`). AUTO picks ACPP
  when the compiler is named `acpp`/`syclcc`/`syclcc-clang` or `--acpp-version` names AdaptiveCpp,
  DPCPP otherwise. `DPCPP` with the acpp driver is a configure error.
- `ACPP` with another compiler: `cmake/BatchLASCompilerBootstrap.cmake` looks for `acpp` on `PATH`,
  `/opt/adaptivecpp/bin` and `~/opt/adaptivecpp/bin`, swaps the cached compiler (CMake resets the
  cache and reconfigures), and errors only when none is found. A bare configure on macOS looks for
  `acpp` too.
- The resolved value is cached as `BATCHLAS_SYCL_IMPL_RESOLVED`; `compare_failures.py --build-dir`
  reads it to pick the ledger.
- `CMakeLists.txt` includes `cmake/BatchLASSyclAcpp.cmake` for ACPP and
  `cmake/BatchLASDetectSYCL.cmake` for DPCPP. The acpp module includes the DPC++ one for its
  functions only (`BATCHLAS_DETECT_SYCL_FUNCTIONS_ONLY`) and sets every variable the rest of the
  build reads:
  - `--acpp-targets=${BATCHLAS_ACPP_TARGETS}` (default `generic`; one target set, no `;`) on the
    same INTERFACE option libraries the DPC++ flags use, so the object-library and
    `$<TARGET_OBJECTS>` model is unchanged; no `-fsycl*`, no `--cuda-path`;
  - devices from `acpp-info`, falling back to the runtime backends in the install
    (`librt-backend-omp`, `-metal` on macOS, `-cuda` plus `nvidia-smi`); `sycl-ls` belongs to DPC++;
  - `-include src/sycl/annexg_complex.hh` (A1) and `-ffp-contract=on` (A11) in every TU.
- The device-link options (`BATCHLAS_SYCL_LINK_JOBS`, FTZ, line info, keep-intermediates, the
  `native_cpu` flags) exist only in the DPC++ module. On acpp, `BATCHLAS_CPU_TARGET` takes only
  `auto` or `none` (@ref design_cpu_target_detection).
- The generated `backend_config.h` carries `BATCHLAS_SYCL_IMPL_DPCPP` / `_ACPP`
  (`#cmakedefine01`). acpp's runtime names (`AdaptiveCpp_*`) appear only in `src/sycl/impl.hh` and
  `src/sycl/sscp_target.hh`; `__acpp_if_target_sscp` also wraps the intrinsic and `asm` sites
  (`backend_sscp.hh`, `register_128x128.hh`).
- Package: `BatchLASConfig.cmake` records `BatchLAS_SYCL_IMPL`, and a consumer must use the same
  implementation. An acpp install ships `include/batchlas/acpp/annexg_complex.hh` and appends
  `-include` for it to `BatchLAS_SYCL_COMPILE_OPTIONS`. `examples/consumer` takes its flags from
  `BatchLAS_SYCL_COMPILE_OPTIONS` / `BatchLAS_SYCL_LINK_OPTIONS`.
- Presets (`CMakePresets.json`): `acpp` inherits `dev` with
  `CMAKE_CXX_COMPILER=/opt/adaptivecpp-25.10.0-cuda13.2/bin/acpp` and `BATCHLAS_SYCL_IMPL=ACPP`
  (tree `build-acpp`); `acpp-tests` adds tests (`build-acpp-tests`). Metal: `acpp-metal`,
  `acpp-metal-tests` ([build shape](#sycl-impl-metal-build-shape)).

### Runtime: one interop seam {#sycl-impl-runtime-interop-seam}

`src/sycl/impl.hh` (namespace `batchlas::impl`) holds every implementation difference above the
kernel level. On DPC++ each helper expands to the plain DPC++ call.

| Helper | DPC++ | acpp |
| --- | --- | --- |
| `kCudaBackend`, `backend_v<N>`, `is_cuda`, `is_hip` | `backend::ext_oneapi_cuda` / `_hip` | `backend::cuda` / `hip` |
| `is_metal`, `acpp_adaptivity_level` | `false`, `-1` | `backend::metal`; the resolved level |
| `backend_of(queue)` | `q.get_backend()` | `q.get_device().get_backend()` |
| `cuda_cc(device)` | parse `"8.9"` | parse `"sm_89"` |
| `native_stream(queue)`, `query_stream<N>(q)` | `get_native` | the in-order executor's stream, possibly `nullptr` |
| `run_native<N>(q, f)` | `f(get_native(q))` inline | `f` inside `AdaptiveCpp_enqueue_custom_operation`, submitted untraced (`submit_and_record(..., traced = false)`) |
| `submit_barrier(q[, wait_list])` | `ext_oneapi_submit_barrier` | last event (in-order) or a wait-list barrier |
| `kernel_max_wg_size<K>` | kernel-bundle query | `min(max_work_group_size, 256)`: the device limit aborts a register-heavy kernel |
| `parallel_for(handler, nd_range, ...)` | the plain launch | a no-op custom operation for an empty range |
| `on_async_errors`, `throw_async_errors`, `throw_launch_errors`, `wait_and_throw` | rethrowing handler; the waits throw | the error sink ([asynchronous errors](#sycl-impl-asynchronous-errors)) |
| `check_group_count(q, nd_range)`, `check_global_range` | empty | CUDA grid-limit check before the launch; on Metal the 2^32 - 1 global-range check |

Kernel-level differences have their own headers: `src/sycl/sscp_target.hh` (the target as a
value), `src/sycl/kernel_attrs.hh` (attributes and loop pragmas), `src/sycl/local_mem.hh` (the
local-memory budget) and `include/batchlas/util/group-collectives.hh` (the group algorithms).
Sites outside them that test `BATCHLAS_SYCL_IMPL_ACPP` directly:

| Site | What differs on acpp |
| --- | --- |
| `src/queue.hh` (`QueueImpl`) | Metal detection, coarse-grained events in the queue properties, the empty-`nd_range` no-op launch |
| `src/util/queue-impl.cc` | `Queue::wait` throws what a wait registered; `Device::supports_sub_group_size` |
| `src/kernels/common/sg_partition/` | backend choice (`SscpBackend`) |
| `src/ops/gemm/kernels/register_128x128.hh` | L2-prefetch asm through `on_ptx()` |
| `register_64x64_k16_wide.hh`, `register_wide_transposed.hh` | the cp.async stage-3 leg is DPC++ only (A12) |
| `src/ops/syev/tridiag/tridiag_solver.cc` | the loop condition is broadcast for NETLIB (OpenMP device) |
| `tests/test_utils.hh` | `kSlmCappedAt48KiB` |

- Every vendor call runs as `impl::run_native<impl::Native::Cuda>(*ctx, [=](auto s){
  handle.setStream(s); ... })`.
- The NETLIB host path needs only a CPU `sycl::device`; acpp's OpenMP backend provides one.
- `mem_advise` with UR flags is DPC++ only (`src/util/sycl-util-impl.cc`, `BATCHLAS_SYCL_IMPL_DPCPP`);
  `ext_oneapi_memcpy2d` is used only under `SYCL_EXT_ONEAPI_MEMCPY2D` (`src/matrix.cc`), otherwise
  the copy kernel runs.

### Device: target dispatch at JIT time {#sycl-impl-device-target-dispatch}

- `src/sycl/sscp_target.hh`: `sycl_impl::on_ptx()`, `on_metal()` and `target_arch()`. On DPC++ they
  are `constexpr` from `__SYCL_DEVICE_ONLY__`, `__NVPTX__` and `__SYCL_CUDA_ARCH__`. On acpp
  generic (SSCP), which compiles one IR for every target and defines neither macro, they are
  JIT-reflection queries (`AdaptiveCpp_jit::reflect<compiler_backend>()`, `<target_arch>()`) inside
  `__acpp_if_target_sscp`, and the JIT deletes the dead branch before lowering.
- **Trap:** a target intrinsic or inline `asm` must also sit inside `__acpp_if_target_sscp(...)`;
  `on_ptx()` alone does not keep it out of the x86 host object, where instruction selection
  rejects it.
- `sg_partition` has a fifth backend, `SscpBackend` (`src/kernels/common/sg_partition/backend_sscp.hh`).
  Its PTX leg is `NvptxBackend` unchanged; the `__nvvm_*` names it calls are defined over the LLVM
  intrinsics through asm labels (`llvm.nvvm.shfl.sync.*`, `vote.ballot.sync`, `activemask`,
  `redux.sync.*`), so they keep the intrinsics' `convergent` attribute. Every other JIT target takes
  `GenericBackend`. No bitcode library and no converged-only formulation is needed.
- `src/sycl/kernel_attrs.hh`: `BATCHLAS_REQD_SG_SIZE`, `BATCHLAS_LAUNCH_BOUNDS` and
  `BATCHLAS_UNROLL_FULL`. On acpp the first two are empty (no SSCP equivalent) and the unroll is
  `clang loop unroll(full)`, because the SSCP JIT unrolls a bare `#pragma unroll` only partially.
  On acpp `Device::supports_sub_group_size(n)` is true only when `n` is the device's single
  sub-group size, so every `can_run` that requires `has_sg32` refuses any other device.
- The `register_128x128` L2-prefetch load is reached through `on_ptx()` plus
  `__acpp_if_target_sscp`. Launch bounds stay an implementation difference (A10).

### Local memory: the launch budget {#sycl-impl-slm-budget}

| | DPC++ (CUDA) | acpp 25.10 (CUDA) | acpp + `BATCHLAS_ACPP_SLM_OPTIN` |
| --- | --- | --- | --- |
| `info::device::local_mem_size` | 101,376 B (the opt-in maximum) | 49,152 B | 49,152 B |
| Launches above 48 KiB | opted in by the runtime | fail, `CU:1` | opted in by the interposer |
| Budget (`impl::local_mem_bytes` less the 4 KiB reserve) | 97,280 B | 45,056 B | 97,280 B |

- **One seam.** `impl::local_mem_bytes` (`src/sycl/local_mem.hh`) is the only reader of
  `local_mem_size`; `DeviceProperty::LOCAL_MEM_SIZE` and `select::Device::slm_budget` go through it.
  On DPC++ it is exactly `get_info<local_mem_size>()`.
- **Every `can_run` that can exceed 48 KiB reads the budget.** At 97,280 B each term admits what a
  constant admitted, so DPC++ routes do not move. The families above 48 KiB:

  | Family | Largest launch | Term |
  | --- | --- | --- |
  | gemm `reg:m=128:n=64:k=32` (u=4, u=2) | 49,920 B | `RegCfg::slm_bytes()` in `can_run`; `gemm_reg` refuses too |
  | herk `gram` | 65,536 B (cdouble, n = 128) | `gram_slm_bytes` in herk's `can_run` |
  | gesvd `jacobi` / `gesvdj_cta`, C = 64 | 71,752 B (double), 71,488 B (cfloat) | `gesvd_jacobi_max_dim(vectors, budget)` |
  | `syev_cta_fused` with an explicit multiplier | 54,272 B (cdouble, n = 16, x4) | multiplier clamp, below |

- **The CTA multiplier clamp counts whole steps.** One multiplier step of `syev_cta_fused`,
  `syev_jacobi_cta`, `sytrd_cta`, `ormqr_cta` and `gebrd_cta` is lcm(P, 32) lanes, 32/P problems.
  `resident::cta_fit_wg_multiplier` counts whole steps, and the kernels throw `unsupported` when
  one step does not fit. A clamp by problem count admits up to 32/P times the bytes for P < 32: on
  DPC++ at 9f551ab0, `syev_cta_fused_benchmark` double n = 16 multiplier 21 and cdouble n = 16
  multiplier 8 aborted with "Excessive allocation of local memory". The result is unchanged
  wherever the old clamp fitted.
- **Tests.** Capacities derive from the device budget. A test whose premise is the 99 KiB opt-in
  skips on `test_utils::kSlmCappedAt48KiB` (acpp without the option) and asserts unchanged
  everywhere else: CTA orders >= 32/34 for cdouble potrf/getrf, the geqrf 48 KiB launch hole, and
  the shipped sm_120 potrf rows.
- **The opt-in variant.** `-DBATCHLAS_ACPP_SLM_OPTIN=ON` (acpp only; a configure error on DPC++ and
  on macOS) builds `libbatchlas_acpp_slm_optin.so` (`src/sycl/acpp_slm_optin.cc`), DT_NEEDED by
  every component. It defines `cuLaunchKernel`, calls
  `cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES)` once per function above 48 KiB, and reports
  `CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN`. It is a recorded A/B variant, not the
  default.
- **Upstream (AdaptiveCpp v25.10.0 source tree).** `src/runtime/cuda/cuda_queue.cpp` lines 187-201
  launch without the attribute. `src/runtime/cuda/cuda_hardware_manager.cpp` lines 418-419 report
  `sharedMemPerBlock`, not `sharedMemPerBlockOptin`. The host device reports `SIZE_MAX`
  (`src/runtime/omp/omp_hardware_manager.cpp` lines 280-281), so `describe` clamps the budget to
  `int64_t`. The upstream opt-in request is open.

### Asynchronous errors and launch failures {#sycl-impl-asynchronous-errors}

| | DPC++ | acpp 25.10 |
| --- | --- | --- |
| Failed launch (grid over 65,535 in y/z, unservable local memory) | `sycl::exception` from the submit | registered in ONE process-wide error list; the submit returns |
| `queue::wait()` after a device fault | reports nothing to the handler | registers the fault in the same list |
| `~queue()` | no handler call | `throw_asynchronous()` over the whole list, through that queue's handler |
| Empty `nd_range` | no-op | launched as a 0-block grid, `CU:1` |

A rethrowing handler on acpp turns any unconsumed launch error into `std::terminate` from the next
`sycl::queue` destructor, any copy included (11 test binaries aborted this way).

- **Handler (acpp).** `impl::on_async_errors` parks errors in a leaked process-wide sink;
  `impl::throw_async_errors(q)` collects and rethrows the first, dropping the rest as DPC++'s
  rethrowing handler does. No handler throws, so no destructor terminates. DPC++ keeps the
  rethrowing handler.
- **Where acpp errors surface.** `QueueImpl::submit_and_record` calls
  `impl::throw_launch_errors` after the submit (acpp launches inside `submit`), so a failed launch
  throws from the launching call as on DPC++. `Queue::wait()`, `Queue::wait_and_throw()` and
  `impl::wait_and_throw(event, q)` throw what a wait registered. Destructors only park.
- **Grid limit (acpp).** `impl::check_group_count` refuses an `nd_range` over the CUDA grid limit
  before the launch, with DPC++'s `errc::nd_range` message ("Number of work-groups exceed limit
  ..."). It runs in `QueueImpl::parallel_for(nd_range)` and in the five tile launchers that submit
  through a command group (`syrk_gram_tiles`, `syrk_triangular_tiles`, `syr2k_triangular_tiles`,
  `trmm_triangular_tiles`, `expand_mirrored`). On DPC++ it is empty.
- **Empty `nd_range` (acpp).** Every command-group `nd_range` launch in `src/` goes through
  `impl::parallel_for(handler, nd_range, ...)`, which gives the handler a no-op custom operation
  for an empty range (an empty command group is an error there). `QueueImpl::parallel_for(nd_range)`
  submits that same no-op command group, so the event it returns stays ordered after earlier work
  on an in-order queue.
- **Vendor calls in the trace.** `impl::run_native` submits through an untraced
  `QueueImpl::submit_and_record`: the custom operation updates the last event but is not a
  kernel-trace record, matching DPC++, where vendor work never appears in `BATCHLAS_KERNEL_TRACE`.
- **Tests.** `util_device_queue_tests` `QueueAsyncErrors.*`: an over-limit grid, 1 MiB of local
  memory, and an error collected by a queue copy's destructor. Each throws a `sycl::exception` at
  the launch or the next wait, and the queue then runs a kernel. Three deliberate breaks each turn
  exactly one case red: no post-submit check, no drain in `Queue::wait`, a rethrowing handler
  (that one aborts).
- **Upstream (AdaptiveCpp v25.10.0 source tree).**
  - `include/hipSYCL/sycl/queue.hpp` lines 288-290: `~queue` calls `throw_asynchronous()`.
  - `include/hipSYCL/glue/error.hpp` line 145: the list it drains is the global `rt::application::errors()`.
  - `queue.hpp:371-374`: `wait()` registers a stream error instead of throwing it.
  - `include/hipSYCL/sycl/handler.hpp` line 379: the `nd_range` `parallel_for` has no empty-range
    guard. The `range` overloads at `:300, :320, :341, :362` have one.
  - `include/hipSYCL/sycl/event.hpp` lines 36-40: the constructor drops its `handler` argument, so
    `event::wait_and_throw()` (`:90-93`) calls an empty `std::function` (`std::bad_function_call`)
    when an error is pending.

### Group algorithms: the portable wrappers {#sycl-impl-group-algorithms}

`include/batchlas/util/group-collectives.hh` (`batchlas::portable::`). Under DPC++ every name is a
using-declaration of the `sycl::` function, so the converted call sites compile to md5-identical
objects. Under acpp they replace four broken `sycl::` group algorithms:

| acpp 25.10 defect (SSCP) | Effect in BatchLAS | `portable::` replacement |
| --- | --- | --- |
| Work-group `reduce_over_group` of a float/double rounds the result to ~16/~42 mantissa bits: the final broadcast bit-casts it to an integer and stores that into the FLOAT scratch (`sscp/builtins/detail/reduction.hpp:101-103`, `detail/broadcast.hpp:29`). Sub-group and integer reduces are exact. | norm, geqrf/orgqr (~300 eps at m >= 32), ortho, symv, sytrd_blocked, steqr, syevx; bdsdc's `best == best_all` never matched, zeroing a vector | sub-group reduce, then a typed 32-slot local scratch across sub-groups (`detail::wg_reduce_floating`); `sycl::vec` component-wise |
| `joint_reduce(maximum)` pads idle items with `numeric_limits<T>::min()`, the smallest positive value (`sscp/group_functions.hpp:347`) | wrong maximum of an all-negative range | starts from `sycl::known_identity_v` |
| In-place `joint_exclusive_scan` writes `result[i + 1]` in the chunk that reads `first[i]` (`group_functions.hpp:968-992`); with one element it drops `init` (`:745-748`) | stedc deflation lost an eigenvalue for every merge larger than 129; through bdsdc, gesvd blocked | chunked scan carrying the running total; reads and writes index i in one chunk |
| Host `joint_reduce` without `init` on a `const T*` does not compile (`libkernel/host/group_functions.hpp:290`) | — | overload without `init`, keeping SYCL's item-to-element mapping (stedc fills its scratch per item with no barrier before the call) |

Every floating `reduce_over_group`/`joint_reduce`/`joint_exclusive_scan` in `src/` goes through
`portable::`. Guards: `DeviceBlasTest.PortableGroupReductionsAreExact` and
`StedcDeflationScan.PortableJointExclusiveScanInPlaceAcrossChunks` (GPU and CPU device; each
deliberate break, i.e. acpp's own call, turns exactly that test red).

Two BatchLAS bugs that acpp exposed, fixed on both builds:

- **`Queue::get_event()` covers USM copies.** `QueueImpl` records USM `memcpy`/`memset`/`fill` as
  well as `submit`/`parallel_for`; without them a caller's wait after a trailing copy returned
  early (syev_blocked NETLIB segfaulted on DPC++ and acpp). Guard:
  `QueueTest.GetEventCoversUsmMemcpyMemsetAndFill`.
- **cuSOLVER `potrf` workspace.** `Lwork` is a count of elements; allocating it as bytes gives a
  workspace `sizeof(T)` times too small (`CUSOLVER error: 7`, cfloat n = 164, cdouble n = 78).

## Compile-level differences: portable spellings {#sycl-impl-compile-differences}

DPC++ extensions or acpp gaps where a spelling legal in both exists. Use the portable form.

| # | Construct | acpp error | Portable form |
| --- | --- | --- | --- |
| C1 | `q.get_backend()` | no member | `q.get_device().get_backend()` (`impl::backend_of`) |
| C2 | `backend::ext_oneapi_cuda`, `get_native(queue)` | no member | `impl::` helper, see [the interop seam](#sycl-impl-runtime-interop-seam) |
| C3 | `get_kernel_id` / `kernel_bundle` | no member | `impl::kernel_max_wg_size`: DPC++ keeps the query; acpp returns the device `max_work_group_size` capped at 256 |
| C4 | `joint_reduce(g, acc.begin(), acc.end(), …)` | no `__acpp_leader_reduce` overload for accessor iterators | pass raw pointers `&acc[0]` (6 TUs at the port) |
| C5 | `group_broadcast` of a struct (`larfg_result`, `Cx<double>`) | SSCP broadcasts scalars only | one `bl::group_broadcast` that splits a trivially copyable struct into words; 22 call sites at the port, most behind `include/batchlas/util/group-invoke.hh` |
| C6 | ternary between an accessor iterator and `T*` | incompatible operands | `&smem[0]` |
| C7 | `sycl::vec<std::complex<T>, N>` | `vec` static assert | plain arrays (`ormqr_cta.cc`) |
| C8 | `joint_reduce` / `reduce_over_group` of `std::complex` with `sycl::plus` | no overload | reduce `real` and `imag` separately (`ritz_values.cc`); a `sycl::vec<R,3>` reduce goes through `portable::` component-wise (`math-helpers.hh`) |
| C9 | `select_from_group` of a non-scalar | no overload | the word-split helper of C5 |
| C10 | oneDPL random (needs `sycl::isequal`, `sycl::tanpi`) | missing in acpp | `src/util/philox.hh` (Philox4x32-10); oneDPL is not a dependency |
| C11 | `ext_oneapi_submit_barrier` | no member | `impl::submit_barrier` |
| C12 | `[[sycl::reqd_sub_group_size(32)]]`, `[[intel::...]]` launch bounds | ignored with a warning | `BATCHLAS_REQD_SG_SIZE`, `BATCHLAS_LAUNCH_BOUNDS` (`src/sycl/kernel_attrs.hh`) |

Rule: `src/`, `tests/`, `benchmarks/` and `tools/` compile under both compilers, the DPC++
PTX-call gate (`device_calls_tests`) stays green, and a seam change leaves the DPC++ route diff
empty. CI's `acpp-compile` job enforces the first part ([CI coverage](#sycl-impl-phases)).

## Implementation differences at run time {#sycl-impl-runtime-differences}

`tests/known-failures-acpp.txt` cites these rows (A1-A12).

| # | Behaviour on acpp | Cause | Resolution and cost |
| --- | --- | --- | --- |
| A1 | `ptxas fatal: Unresolved extern function '__mulsc3'/'__muldc3'` (17 binaries without the fix) | `std::complex` `*` lowers to Annex G libcalls; SSCP ships no device definition | Device definitions of `__mulsc3`/`__muldc3`/`__divsc3`/`__divdc3` in `src/sycl/annexg_complex.hh`, force-included in every TU, so both builds keep Annex G semantics. `-fcx-limited-range` is rejected ([alternatives](#sycl-impl-decisions)). `acpp_jit_tests` reports any unresolved libcall. |
| A2 | Launches above 48 KiB of local memory fail, CUDA error 1 (gemm, trmm/symm/syrk candidates, herk) | acpp reports `local_mem_size` = 48 KiB and never calls `cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES)` | Every `can_run` reads the device budget ([the local-memory budget](#sycl-impl-slm-budget)), so acpp routes around large-SLM families. Auto routes are unchanged; budget-derived block sizes cost 1.22-2.66 in the affected pinned cells (potrf `blocked`, getrf/geqrf `cta`), 0.99-1.20 in the opt-in tree, which also runs the refused families (0.91-1.20; gemm `reg 128x64x32 u4` 0.77-1.75) at 4-7% extra on short launches ([perf](../perf/sycl-implementations.md#sycl-impl-ab-the-48-kib-budget-and-the-opt-in-variant)). Upstream request open. |
| A3 | One bad launch would kill the whole process | acpp's default async handler calls `std::terminate` | An async handler on `QueueImpl` that never throws from a destructor: [asynchronous errors](#sycl-impl-asynchronous-errors). |
| A4 | `info::device::version` is `"sm_120"`, not `"12.0"`; a DPC++-only parser returns 0 and keys the tables `gpu` | different version string | `impl::cuda_cc` parses both; both trees key the tables `[sm_120]`. |
| A5 | Hangs under divergence (`SgPartitionDivergence/0.MaskedSG32`, bdsdc, syev_cta, syev_blocked, syev_two_stage, gesvd, sytrd_blocked) without `SscpBackend` | SSCP is single-pass, so `__SYCL_DEVICE_ONLY__ && __NVPTX__` is never true and `sg_partition` drops to the unmasked `GenericBackend` | [JIT-time target dispatch](#sycl-impl-device-target-dispatch): `SscpBackend`; no CTA test hangs. Speed (n = 5..32, batch 16384, vs DPC++): `steqr_cta` 1.07-1.41 float and 1.00-1.21 double, `syev_cta_fused` float 0.54-0.81 ([perf](../perf/sycl-implementations.md#sycl-impl-ab-sub-group-cta-kernels)). |
| A6 | Out-of-order queues segfault in vendor calls when the stream is taken from the in-order executor | the stream is not ordered with the queue | [the interop seam](#sycl-impl-runtime-interop-seam): vendor calls go through `AdaptiveCpp_enqueue_custom_operation`. Cost: 3-17 us more per short vendor call; 1.00 at 1 ms and above. |
| A7 | `Queue::native_handle()` returns null when guarded by `SYCL_EXT_ONEAPI_BACKEND_CUDA` | DPC++-only macro | [the interop seam](#sycl-impl-runtime-interop-seam). |
| A8 | A death test hangs | `fork()` after the acpp runtime started its threads | `GTEST_FLAG(death_test_style)="threadsafe"` in `select_tests` and the candidate tests. |
| A9 | Wrong answers (cdouble potrf/getrf CTA, orgqr orthogonality 5.8e-6 vs 3.8e-6, norm, cond, lanczos, syr2k_candidates, geqrf) | acpp group-algorithm bugs, the 48 KiB budget, two BatchLAS bugs on both builds | [the group-algorithm wrappers](#sycl-impl-group-algorithms), [the local-memory budget](#sycl-impl-slm-budget). Residue in `tests/known-failures-acpp.txt`. |
| A10 | `BATCHLAS_LAUNCH_BOUNDS` empty; `reqd_sub_group_size` ignored | single-pass SSCP has no NVPTX device pass; the JIT emits `maxntid` from the launch size but no `minnctapersm` | Accepted as an implementation difference. A DPC++ tree with the bounds removed runs the bounded kernels at 0.99-1.03 on sm_120, so the bounds are not what acpp loses; the acpp tiny-kernel losses (to 1.56) are code generation ([perf](../perf/sycl-implementations.md#sycl-impl-ab-launch-bounds-r10)). The `register_128x128` prefetch is reached through [JIT-time target dispatch](#sycl-impl-device-target-dispatch). |
| A11 | Contraction differed: the acpp driver adds `-ffp-contract=fast` at -O2+, DPC++ (clang) uses `on`. Both use `sqrt.approx.f32`/`rsqrt.approx.f32` (bit-identical on 4M floats) and `div.rn`, no FTZ | different driver defaults | `-ffp-contract=on` in `BatchLASSyclAcpp.cmake`. Upstream: acpp's `nvvm-reflect-prec-sqrt` flag (`LLVMToPtx.cpp:86,163`) is ignored by LLVM 20's NVVMReflect, so precise sqrt is unreachable; harmless because it matches DPC++'s default. |
| A12 | The cp.async (sm_80+) gemm pipeline is compiled out: `copy_async.hh` falls back to plain copies, and the wide tiles skip their stage-3 leg (`!BATCHLAS_SYCL_IMPL_ACPP`, `register_64x64_k16_wide.hh`, `register_wide_transposed.hh`) | `BATCHLAS_GEMM_CP_ASYNC` needs `__NVPTX__` and `__SYCL_CUDA_ARCH__`, which SSCP never defines | Correct but slower kernels; the cost on acpp is not measured. |

## Correctness A/B {#sycl-impl-correctness-ab}

- Each implementation has its own self-contained failure ledger: `tests/known-failures.txt`
  (DPC++) and `tests/known-failures-acpp.txt`; defects shared by both are listed in both. A Metal
  tree has a third, `tests/known-failures-acpp-metal.txt` ([Metal ledger](#sycl-impl-metal-failure-ledger)).
  `.github/ci/compare_failures.py` picks the ledger with `--sycl-impl` (plus `--gpu-family METAL`)
  or `--build-dir <tree>` (the cached `BATCHLAS_SYCL_IMPL_RESOLVED`). Compare failing **names**.
- **Gate for "acpp supported":** the acpp tree's failing set is a subset of the DPC++ failing set
  on the same box, plus entries in the acpp ledger, each with a cause from
  [runtime differences](#sycl-impl-runtime-differences). At 2026-10-09: 8 of 103 tests failed, all
  in the ledgers or box-environment.
- **Cross-implementation accuracy:** the accuracy harnesses (`steqr_accuracy`,
  `eigensolver_accuracy`, `orthogonality_accuracy`) run in both trees on identical seeds and graded
  inputs, with FP flags matched (A11). `scripts/impl_diff.py` reports per-cell residual ratios.
  This complements the unit tests, which cannot see a 1.5x residual change.
- Route parity: `scripts/route_diff.sh` across the two trees. A route that differs is either an SLM
  budget difference (A2, expected) or a bug.
- `device_calls_tests` is DPC++-only (it skips on acpp: no fatbin). The acpp counterpart is
  `acpp_jit_tests`, below.

### The JIT-ed PTX check {#sycl-impl-jit-ptx-check}

`scripts/check_acpp_jit.py`, ctest `acpp_jit_tests` (label `util`; registered only in an acpp tree
with the CUDA backend, one GPU slot, skips with 77 when no PTX appears).

- **Source.** acpp keeps the PTX it hands the CUDA driver in
  `<ACPP_APPDB_DIR>/apps/<app>-<hash>/jit-cache/<kernel>.<config>.jit`: plain PTX, one `.entry`
  per file, one file per specialization and application (a full-suite appdb, 2026-10-09: 42,015
  files, 1,569 kernels). Host (OpenMP) kernels sit beside them as ELF objects and are ignored. Only
  launched kernels appear, so the test runs CUDA cases against a fresh appdb first:
  `gemm_candidates_tests` (float pins), `gesv_tests` (double), `potrf_tests` CTA and lpanel
  (complex), `trsm_tests` `TrsmNativeSgLeft.*`, `syev_cta_fused_tests` (complex) and
  `getrf_tests` `RegPanel*`. 148 s on one RTX PRO 6000; 520 PTX files, 206 kernels.
- **calls.** Any `call` or `.extern .func` fails. SSCP inlines every callee: none in the 42,015
  files of the full-suite appdb. A kernel that ptxas rejects is still cached, so the A1 case is
  visible: a `std::complex<float>` multiply built without `annexg_complex.hh` fails to load
  (`ptxas fatal: Unresolved extern function`) and the check reports `-> __mulsc3`.
- **required.** Nine families (gemm 128x128 and register tiled, getrf and gesv tiny, potrf CTA
  and lpanel, trsm `sg_left`, steqr CTA, syev CTA fused) must be JIT-ed at least once.
- **stack.** 21 register-resident families (the `BATCHLAS_UNROLL_FULL` sites: register gemm
  tiles, tiny solvers, CTA kernels, level-3 tiles) must have no `__local_depot`. Accesses are
  counted both as `ld/st.local` and through the generic `%SP`. Known cases carry a byte ceiling
  and a reason; growing past the ceiling fails.

Known stack arrays (full-suite appdb; DPC++ column from `cuobjdump -ptx` of the DPC++ tree):

| Family | acpp | DPC++ | Note |
| --- | --- | --- | --- |
| `getrf_panel_reg<cdouble, 32>` | 512 B, ~600 ld + 650 st.local | 0 B | acpp only. `rA[32]` stays a stack array although its loads are unrolled to constant offsets. Cause not investigated |
| getrs fused, cdouble | 16 B | 0 B | acpp only: the temporary of a `std::complex<double>` row swap in local memory, generic-addressed, in some specializations |
| gemm wide transposed, cdouble | 16 B | 0 B | acpp only: the `std::complex<double>` staged from global to local memory in the tile load |
| gemm register tiled `128x64x32` u=4, u=2 | 128 B | 128 B | both, same two tiles |
| trsm `sg_left` | 128-512 B | 128-512 B | both, same kernels, same bytes |
| syev CTA fused, complex | up to 512 B | up to 272 B | both; by order cdouble n=32 512 vs 272 B, n=4 64 vs 136 B. DPC++ also 16-72 B for real types, where acpp has none |
| gesvdj CTA | up to 256 B | up to 736 B | both; acpp never larger per kernel |

Outside the gated families, acpp also keeps 16 B depots in the cdouble laswp, `getrf_panel_resident`
and `getrf_panel_global` kernels (DPC++ 0 B), 88 B in `potrf_blocked_panel_fixup<cdouble>` (DPC++
88 B), and 32-64 B in the stedc merge kernels.

Deliberate breaks, each against the real appdb: lowering the `getrf_panel_reg` ceiling to 256 B,
dropping the trsm `sg_left` entry, and renaming the potrf lpanel pattern each turn exactly that
family red.

Every JIT run prints `'+ptx88' is not a recognized feature for this target`: acpp 25.10 asks LLVM
20 for PTX ISA 8.8 (CUDA 13.2), LLVM 20 ignores it and emits `.version 8.7`. Harmless here.

## Performance A/B protocol {#sycl-impl-performance-ab}

On top of the measurement rules in `docs/developer/agent-guide.md` §10:

- **JIT and adaptivity.**
  - `ACPP_APPDB_DIR` is set per campaign and cleared on driver or acpp change.
  - `ACPP_ADAPTIVITY_LEVEL` is pinned and recorded. Report level 1; level 2 is a separate arm.
  - Do throwaway passes until acpp stops printing its "new binaries JIT-compiled" warning.
  - Level 2 specializes invariant kernel arguments after about 1024 launches. Benchmarks that
    reuse one shape would flatter acpp against AOT DPC++, so record the invocation count.
- **Launch path.** In-order queues (the default), `ACPP_RT_SCHEDULER=direct`, and coarse-grained
  events as a recorded variant. Launch latency matters below saturation; compare algorithms at
  saturation only.
- **Provenance** (benchviz README, "Toolchain and SYCL implementation" and "Implementation A/B"):
  - `provenance.builds[]` records `sycl_impl`, `sycl_impl_version`, the compiler `--version` head,
    the fp flags with the effective fp contract, and for acpp the `ACPP_*` knobs its runs got.
  - benchviz sets `ACPP_ADAPTIVITY_LEVEL=1`, `ACPP_RT_SCHEDULER=direct` and a per-campaign
    `ACPP_APPDB_DIR` unless the environment does. Each row records `sycl_impl` and `acpp_jit`.
  - `compare` carries each side's toolchain, labels the arms `DPC++` / `AdaptiveCpp`, and warns on
    any other difference. `store.py` warns on an acpp build whose effective fp contract is not
    `on` (A11), and on an `IntelLLVM` build.
  - `benchviz ab` runs two builds interleaved, alternating which goes first per cell.
  - Coarse-grained events are a queue property, not an environment knob; BatchLAS sets it only on
    Metal queues ([coarse events](#sycl-impl-metal-coarse-events)), and `builds[]` records that.
- **Tables.** Both trees read the same `tuned/*.sm_120.txt`. Tables are keyed by device, not
  implementation ([alternatives](#sycl-impl-decisions)).
- **Comparison set** (@ref perf_sycl_implementations): gemm (float and double, NN/TN, strided
  `ld`, beta=1), potrf, getrf, geqrf, syev_cta, steqr_cta and stedc at batch >= 128, n from the CTA
  range up to 1024, plus syev Auto, a no-launch-bounds DPC++ arm, the opt-in arm, a batch x2
  saturation check and an adaptivity-level-2 arm. It covers the register-tile kernels, the
  sub-group CTA kernels (A5, A10) and the SLM-heavy kernels (A2). benchviz pins only
  `native`/`vendor`, so that campaign used its own driver.

## CI coverage of the acpp build {#sycl-impl-phases}

- Hosted job `acpp-compile` ([CI](../ci.md#the-acpp-compile-job), `.github/ci/build_adaptivecpp.sh`):
  builds AdaptiveCpp 25.10 (cached), configures BatchLAS with `--acpp-targets=generic` and CUDA
  off, and compiles every target; it runs nothing. It turns red when a DPC++-only construct leaks
  in. Green on GitHub (run 39564010759, 2026-10-21).
- The full acpp gate stays local: an `acpp-tests` tree, gated against
  `tests/known-failures-acpp.txt`, plus `acpp_jit_tests` ([the JIT-ed PTX check](#sycl-impl-jit-ptx-check)).
- A seam change must leave the DPC++ build bit-identical in behaviour: the route diff is empty and
  the kernel PTX hashes are unchanged for every TU the change did not touch.

## Rejected alternatives {#sycl-impl-decisions}

| Alternative | Result | Verdict |
| --- | --- | --- |
| `-fcx-limited-range` instead of Annex G definitions (A1) | Cleared all 17 failing binaries (39 of 77 green vs 29 at the probe), but changes NaN/Inf results, so the A/B would be unequal | Rejected; a measured variant only |
| Opt in above 48 KiB by default on acpp (A2) | Interposer: affected pinned cells 0.99-1.20 vs 1.22-2.66 without it; one Auto cell (potrf float n = 1024) 1.06 vs 1.23 | Kept as `BATCHLAS_ACPP_SLM_OPTIN`, not the default; route around it instead |
| Drop vendor libraries from the acpp build | Through `run_native` vendor cells run the same call: 1.00 at 1 ms and above, 3-17 us more per short call | Rejected: vendor libraries kept. The vendor-free acpp tree is not measured |
| Per-implementation tuning tables (an implementation suffix on the table key) | Fastest pinned family changes in 3 of 78 cell groups, each a near-tie on DPC++; Auto takes the same route in all 90 Auto cells ([perf](../perf/sycl-implementations.md#sycl-impl-ab-per-implementation-tables)) | Not added; revisit with a newer acpp LLVM or another device |
| acpp `cuda:sm_120` (SMCP) flow | clang 20's CUDA wrapper rejects the CUDA 13.2 headers, even on an empty file | Unusable on this install; generic SSCP is the only flow |
| A bitcode library or `compile_if` for `__nvvm_*` | Not needed: asm labels on the LLVM intrinsics are reachable from SSCP stage 1 | Rejected |
| Bare `#pragma unroll` under SSCP | The JIT unrolls it only partially | `clang loop unroll(full)` (`BATCHLAS_UNROLL_FULL`) |

## Probe baseline {#sycl-impl-probe-result}

The starting point (2026-10-08): a scratch copy of `1f0b6d84` compiled with
`acpp --acpp-targets=generic -O2 -std=c++20` and the DPC++ tree's `-I`/`-D` flags, patched by
regex shims, not real fixes.

| Stage | Result |
| --- | --- |
| CMake configure, `CMAKE_CXX_COMPILER=acpp` | rc=0 but unusable: `-fsycl` was still added and acpp rejects it fatally; CUDA detection went through DPC++'s `sycl-ls` |
| Library TUs, pristine source | 23 of 109 compiled |
| Library TUs after shim layers L0+L1 | 107 of 109 (`ormqr_cta.cc`: C7; `ritz_values.cc`: C8) |
| Test TUs | 82 of 86 compiled |
| Compile wall, -P32, no ccache | acpp 50 s vs DPC++ 62 s. The acpp `.so` link takes 1.1 s because there is no device link; kernels are JIT-compiled at first launch instead |
| `examples/consumer` gemm | Max error 0 under Auto and under every gemm pin (`tiled`, `direct`, `small`, `reg:m=128:n=128:k=8:u=1`, `vendor`); SSCP JIT and the cuBLAS stream interop both worked |
| Test binaries | 39 of 77 fully green (29 before `-fcx-limited-range`); causes in [runtime differences](#sycl-impl-runtime-differences) |

## Where DPC++-only code remains {#sycl-impl-dpcpp-specific}

`sycl::ext::*` is not used in code; there is no `host_task`, no `interop_handle`, no
specialization constants and no `sycl::stream`.

| Construct | Where |
| --- | --- |
| `-fsycl*`, `-Xsycl-target-*`, `sycl-ls` parsing, the alias probe, `native_cpu`/`spir64_x86_64` | `cmake/BatchLASDetectSYCL.cmake`, `BatchLASCompilerBootstrap.cmake`, `BatchLASOptions.cmake` |
| Flags reach targets through three helpers | `batchlas_apply_object_options`, `batchlas_configure_binary_target` (`cmake/BatchLASTargetHelpers.cmake`), `batchlas_configure_component` (`src/CMakeLists.txt`) |
| Backend enums, `get_native`, kernel bundles, barriers | `src/sycl/impl.hh` only ([the interop seam](#sycl-impl-runtime-interop-seam)) |
| `__nvvm_*` / `__spirv_*` / AMDGCN intrinsics, inline PTX `asm` | `src/kernels/common/sg_partition/`, `src/ops/gemm/kernels/` (`register_128x128.hh`, `register_256x128.hh`, `copy_async.hh`, the wide tiles), each `#if`-guarded |
| PTX inspection: fatbin magic, `cuobjdump`, banned `__spirv_` callees | `scripts/check_device_calls.py`, `register_probe.sh`, `.github/skills/ptx-codegen-comparison`; acpp counterpart `scripts/check_acpp_jit.py` |
| `/opt/dpcpp-cuda`, `libsycl.so.9`, `-fsycl` | `examples/consumer/`, `consumer_test.sh`, CI, `scripts/rocm_syntax_check.sh` |

## AdaptiveCpp Metal {#sycl-impl-metal}

Machine: Apple M5 Max (`applegpu_g17s`), macOS 27.0.1, AdaptiveCpp `develop` `9ec394ef` plus local
patches 0001-0013 at `~/opt/adaptivecpp`, Homebrew `llvm@21` 21.1.8, OpenBLAS 0.3.34, measured
2026-10-13. Build recipe: [macOS](../guide/build.md#macos-apple-gpu-through-adaptivecpp-metal).

A third configuration of the acpp build: the Apple GPU through AdaptiveCpp's Metal backend. It is
vendor-free, has no device fp64, and every table it reads was measured on NVIDIA. Nothing here
changes the DPC++ or Linux acpp builds: each switch is either a configure result that is false
there (`BATCHLAS_HAS_SYCL_BACKEND`, `BATCHLAS_DEVICE_NO_VECTOR_TYPES`) or a runtime test that is
false off Metal (`impl::is_metal`, `sycl_impl::on_metal()`).

### Metal: the SSCP-to-MSL pipeline and its cache {#sycl-impl-metal-pipeline}

- **Compile.** `--acpp-targets=generic`, as on Linux: one SSCP IR image per TU, no device link.
- **First launch.** AdaptiveCpp's `llvm-to-metal` translator lowers the kernel's IR to Metal
  Shading Language, and `newLibraryWithSource` compiles that MSL. Both happen in-process at the
  first launch of each kernel; a translation or MSL compile error is reported at that launch.
- **Cache.** Translated binaries go to `~/.acpp/apps` (`ACPP_APPDB_DIR` overrides). The cache is
  keyed by the kernel, not by the translator, so **clear it after reinstalling AdaptiveCpp**, or
  kernels keep the code the old translator emitted.
- **Configure.** `BatchLASSyclAcpp.cmake` parses `acpp-info`, including its `Vendor:` field. A
  `Metal` GPU sets `BATCHLAS_DETECTED_APPLE_GPU`, arch `apple_gpu_<arch>` (`apple_gpu_g17s` here),
  and, unless `BATCHLAS_ENABLE_METAL=OFF`, `BATCHLAS_HAS_SYCL_BACKEND` and
  `BATCHLAS_DEVICE_NO_VECTOR_TYPES`. `BATCHLAS_ENABLE_METAL` is `AUTO|ON|OFF` like
  `BATCHLAS_ENABLE_CUDA`; `ON` needs ACPP and a Metal GPU. `OFF` while the GPU is visible is a
  `FATAL_ERROR` (the guards would be compiled out under queues that still reach it): hide it with
  `ACPP_VISIBILITY_MASK=omp`. `Level Zero`/`OpenCL` GPUs count as Intel only when the vendor says
  so. Without `acpp-info`, a `librt-backend-metal.dylib` in the install stands for the GPU.
- **The OpenMP device is mandatory.** Double and the verification layer run there, so a Metal
  configure without `BATCHLAS_HAS_CPU_TARGET` is a `FATAL_ERROR`. `ACPP_VISIBILITY_MASK=omp;metal`
  in the presets keeps both devices visible.

### Metal: Backend::SYCL and the no-fp64 policy {#sycl-impl-metal-fp64-policy}

MSL has no `double`. AdaptiveCpp's Metal emitter rejects any double type, so a double kernel cannot
be JIT-compiled for the GPU at all.

- **`Backend::SYCL`** (`include/batchlas/blas/enums.hh`): native SYCL kernels on a GPU with no
  vendor library. `Queue::backend()` resolves an Apple GPU to it. `select::native_gpu_backend<B>`
  (`src/select/vendor.hh`) is CUDA or SYCL, so every native GPU family gated on CUDA runs here too;
  ROCm and the host keep their vendor routes.
- **Instantiation.** `BATCHLAS_INSTANTIATE_*_ALL_BACKENDS` (`src/util/template-instantiations.hh`)
  give `Backend::SYCL` float and complex<float> only, plus the double pair under
  `BATCHLAS_HAS_DEVICE_FP64` (`BATCHLAS_DEVICE_FP64=AUTO|ON|OFF`; AUTO is `OFF` on Metal, and `ON`
  there is a configure error). No `<SYCL, double>` symbol exists, so a stray reference is a link
  error, not a JIT error.
- **The refusal.** The deducing entry points (`BATCHLAS_DISPATCH_ON_QUEUE` and the `with_backend`
  lambdas in `options.hh`/`extensions.hh`) go through `detail::with_backend_gated<NeedsFp64>`
  (`include/batchlas/blas/queue-dispatch.hh`). `detail::needs_fp64_v<Args...>` is true when any
  container argument (`Matrix`, `MatrixView`, `Vector`, `VectorView`, `Span`, `UnifiedVector`, or a
  one-parameter template over the scalar such as `StedcParams<double>`) holds double or
  complex<double>. Scalars do not count: a float op may take a double knob. Then the SYCL branch is
  discarded at compile time and replaced by `batchlas::unsupported` naming the op, the device and
  the fix (a CPU `Queue`). `with_backend_for<T>(ctx, op, f)` is the same gate for a caller that
  knows its element type; plain `with_backend` has no gate.
- **No silent reroute.** A double call on the GPU throws. Double work runs on a CPU `Queue`, whose
  backend is NETLIB (OpenBLAS plus the OpenMP device). `Queue::set_backend(NETLIB)` on the Apple GPU
  throws, because NETLIB still launches its native kernels, double ones included, on the queue's
  device.
- **No internal double on Metal.** `lartg` for complex<float> rounds in float on the Metal device
  only (chosen at JIT time, `src/math-helpers.hh`); `random_hermitian_tridiagonal_with_log10_cond_metric<SYCL, float>`
  builds the spectral case on a leaked host NETLIB queue; the queue-less `Matrix` helpers use
  `Device::default_device_for<T>()`, the CPU for double; `batchlas::verify` checks on a CPU queue
  ([verification](verification.md#verification-no-fp64-device)).
- **Tests.** `backend_dispatch_tests` `NoDeviceFp64.*` call gemm, potrf and syev with double through
  the ordinary spelling and expect the throw.

### Metal: the ACPP_ADAPTIVITY_LEVEL=0 guard {#sycl-impl-metal-adaptivity-guard}

At adaptivity level 0 AdaptiveCpp JITs every kernel of a TU together at the first launch. Every
library TU instantiates double kernels for the host backend, so on Metal one double kernel fails the
whole batch and every float launch in that TU fails with it. The `Queue` constructors therefore
throw `batchlas::unsupported` on an Apple GPU when the resolved level is 0
(`require_metal_adaptivity`, `src/util/queue-impl.cc`; the level is read from AdaptiveCpp's
settings, environment or config file, `impl::acpp_adaptivity_level`). Level 1 (the default) and
above JIT per kernel. `backend_dispatch_tests_adaptivity0` runs `MetalAdaptivityGuard.*` with
`ACPP_ADAPTIVITY_LEVEL=0`; it is added only in a Metal tree.

### Metal: the local AdaptiveCpp patches {#sycl-impl-metal-acpp-patches}

Thirteen patches on AdaptiveCpp `develop` `9ec394ef` (`ACPP_BASE`), applied by
`scripts/macos/build_adaptivecpp_metal.sh`. They are local and not sent upstream. The evidence for
each, with its reproducer, is in `scripts/macos/adaptivecpp-patches/README.md`.

| Patch | Upstream defect | What it broke in BatchLAS |
| --- | --- | --- |
| 0001 barrier fences | `group_barrier` fenced threadgroup memory only; a work-group barrier with sub-group fence scope emitted a `simdgroup_barrier` | Every kernel that exchanges global data across a barrier was racy by the MSL spec (about 384 barrier sites). Not observed on this GPU |
| 0002 safe math | `MTLCompileOptions` default to fast math | `7.0f/3.0f` and `exp` off by up to 3 ulp; NaN/Inf guards (Annex G complex, secular-equation safeguards) foldable. `ACPP_METAL_FAST_MATH=1` restores fast math |
| 0003 launch limits | a launch over the pipeline's thread or threadgroup-memory limit ran with undefined results | Silent wrong answers; now an asynchronous error naming the kernel and byte counts |
| 0004 narrow i8/i16 | signed i8/i16 `sdiv`/`srem`/`ashr` emitted a size-mismatched `as_type` | float `small` gemm kernels did not compile |
| 0005 memmove | a `device`-to-`threadgroup` memmove compared distinct pointer types | `reg:` gemm with a 16-byte struct did not compile (BatchLAS's `Vec4` copies element-wise anyway) |
| 0006 no vectorizers | LoopVectorize/SLP produced vector IR the emitter cannot print | syrk/syr2k triangular-tile kernels failed the JIT (`insertelement`) |
| 0007 `get_num_subgroups` | undefined in the Metal libkernel | 8 float `geqrf_cta` kernels did not compile |
| 0008 trunc to i1 | `trunc iN to i1` emitted as `(bool)x`, not bit 0 | `reg:m=128:n=32:k=16:u=1` gemm wrong once k spans three tiles |
| 0009 convergent SIMD-group intrinsics | shuffles/broadcasts not `convergent`, so LLVM sank them under per-lane branches | float tiny `getrf` n = 5..8 wrong |
| 0010 linear translation time | StructurizeCFG's per-PHI `SSAUpdater`, name renumbering and long names made translation super-linear | `posv_tiny` NC = 28 took about 90 s to translate (1.25 s after) |
| 0011 no re-advise in `alloc_at` | every shared/host allocation re-`madvise`d most of a 10x-RAM region (about 70 us per call) | `steqr_tests` timed out at 1800 s; passes in 335 s after |
| 0012 float memory through `uint` | stores of an `i32` into a float object and float loads of it did not alias under MSL strict aliasing | float tiny `geqrf` left R(j,j) stale; geqrf, ormqr, orgqr and the two-stage syev failed downstream |
| 0013 instruction in emitter errors | 0010's IR-comment switch also emptied the instruction text of three JIT error messages | diagnostics only; the MSL is unchanged |

With 0001-0012 the AdaptiveCpp `sycl_tests` pass on Metal (792 of 793 cases, 1 failed warning).

### Metal: launch limits {#sycl-impl-metal-launch-limits}

Measured on the M5 Max with patch 0003.

| Limit | Value | Where BatchLAS models it |
| --- | --- | --- |
| Threadgroup memory reported (`local_mem_size`) | 32,768 B | `cmake/BatchLASDetectSYCL.cmake` table row `apple_gpu_*` (`batchlas_architecture_table_local_mem_bytes`; a hypothesis, not a measured budget) |
| Hidden per-lane scratch | AdaptiveCpp adds `8 * wg` bytes of group-algorithm scratch to every launch (`metal_queue.cpp`) | `impl::local_mem_scratch_per_item` = 8; `impl::local_mem_bytes(d, wg)` = reported - 8 wg; `local_mem_bytes(d)` takes the device's largest wg (24,576 B at wg = 1024), so every older caller is safe at any width |
| Selector budget | `slm_budget` (at max wg, less the 4 KiB reserve) = 20,480 B | `select::Device::slm_per_item` and `slm_budget_for(wg)`; gemm `reg` reads the per-width figure |
| Threads per threadgroup | 1024 for every pipeline measured, including 512 live floats per thread: the compiler spills rather than lowering the cap | the getrs_fused sm_89 register cap is skipped on Metal |
| Grid | **no 65,535 cap** on the work-group count, in any dimension; thread positions are 32-bit, so a dimension of 2^32 work-items or more runs with wrong ids and no error (2^32 - 1024 is exact) | `impl::check_global_range` (every `parallel_for`, and `check_group_count` on Metal) throws `errc::nd_range` above 2^32 - 1; `select::Device::max_groups_yz` = 2^32 - 1, read by the gemm, syrk, syr2k, trmm and symm `can_run` instead of 65,535 |
| Co-residency | not promised; MSL atomics are relaxed only | the latrd grid-barrier path is off on Metal, even when forced ([sytrd](../perf/sytrd.md#sytrd-the-latrd-grid-kernel-and-its-co-residency-cap)) |
| Sub-groups per work-group (safe) | 1 | `cmake/BatchLASDetectSYCL.cmake`: the unrecognised-GPU default, a hypothesis |

Guards: `resident_capacity_tests` launches at the advertised budget and at the raw per-width figure,
refuses one 16-byte granule past it, checks `slm_budget` against the launch budget, runs 65,537
groups per dimension and 2^32 - 1024 work-items, and serves batch 65,537 with five gemm pins.
`select::describe` hard-codes 8 and 2^32 - 1 (`select.cc` cannot include `src/sycl/`); these tests
hold the two copies equal.

### Metal: coarse-grained events and error delivery {#sycl-impl-metal-coarse-events}

- **Coarse events by default.** AdaptiveCpp commits one Metal command buffer per operation unless
  the queue has `AdaptiveCpp_coarse_grained_events`. `QueueImpl::make_queue_properties` adds it on
  Metal queues unless profiling is on or `BATCHLAS_METAL_FLUSH_EACH_OP` is set
  ([environment](environment.md)). Such an event completes with the whole queue: over-synchronised,
  never early, and without timestamps.
- **Measured** (6-7 reps per mode, alternating, first pass dropped): back-to-back calls with no host
  wait, flush-each-op over coarse time: gemm 1.64 (n = 8, batch 1024); potrf 1.33 (n = 8/1024), 1.39
  (n = 16/4096), 1.72 (n = 32/128); neutral at n = 64/512. With a host wait after every call, coarse
  is neutral, and 4-8% slower at n = 64/batch 512. Results are bit-identical between modes.
- **`~QueueImpl` always drains on acpp.** acpp's `~queue` does not wait, and with coarse events the
  command buffer stays open, so a derived queue's work could outlive the caller's wait. DPC++
  drains only when the arena holds blocks.
- **Errors are asynchronous.** A command-buffer failure (GPU timeout, a patch-0003 launch refusal) is
  registered from a completion handler, so it surfaces only at `Queue::wait()` or the next
  `impl::throw_launch_errors` ([asynchronous errors](#sycl-impl-asynchronous-errors)). Under coarse
  events one error can cover many operations: set `BATCHLAS_METAL_FLUSH_EACH_OP=1` to attribute it.
  The macOS GPU watchdog also applies to the GPU that drives the display.

### Metal: device-code rules {#sycl-impl-metal-device-code-rules}

| Metal fact | Consequence | Where |
| --- | --- | --- |
| No generic address space: a pointer chosen between local and global memory at run time is rejected ("Conflicting address space information in PHI node") | the body is instantiated once per storage combination under `on_metal()`; elsewhere the pointers are selected once | sb2st chase (`src/ops/syev/sytrd/sytrd_sb2st_cta.cc`), tridiagonal solve (`src/ops/syev/tridiag/tridiag_solver.cc`) |
| No floating-point atomics on threadgroup memory | One and Inf norms give each column or row sum to one work-item on Metal | `src/util/norm.cc` |
| No static (program-scope) local memory | `wg_reduce_floating` uses AdaptiveCpp's own reduce on Metal; its typed scratch has no rounding defect ([group algorithms](#sycl-impl-group-algorithms)) | `include/batchlas/util/group-collectives.hh` |
| No vector types in the emitter | under `BATCHLAS_DEVICE_NO_VECTOR_TYPES`: `Vec4` and `tiny_r4` are 16-byte structs, `Raw16` is an `unsigned __int128` (emitted as `uint4`). CUDA codegen keeps the vectors | `src/ops/gemm/kernels/vec4.hh`, `src/kernels/common/tiny_device.hh`, `src/ops/gemm/kernels/register_64x64_k16_wide.hh` |
| No double | see [the no-fp64 policy](#sycl-impl-metal-fp64-policy) | |
| Float subnormals flush to zero, in safe math too; `info::device` still reports denorm support | a test that needs a subnormal must probe the device | `TrsmNativeSgLeft.NonFiniteReciprocalFallsBackToDivision` probes and skips |
| `logb(0)` and `logb(inf)` are finite (`(float)ilogb(fabs(x))` in the Metal libkernel) | Annex G complex division of zero, inf or NaN may give wrong special values; normal-range results are unaffected. Not tested | `src/sycl/annexg_complex.hh` (`annexg::div`) |
| An arm64 host rejects the `"l"` asm constraint | the PTX L2-prefetch asm is compiled only on x86 hosts | `src/ops/gemm/kernels/register_128x128.hh` |

`sycl_impl::on_metal()` (`src/sycl/sscp_target.hh`) is JIT-time reflection on the compiler backend
(spelled as the number 4, since acpp 25.10 has no `metal` enumerator); a private copy sits in
`group-collectives.hh` because an installed header cannot include `src/`. It folds to a constant in
each backend's IR, so the other branch is dropped before emission; on DPC++ it is `constexpr false`.

### Metal: the apple_m5_max table key {#sycl-impl-metal-select-key}

`select::describe` keys an Apple GPU by its device name (`select::apple_device_key`: "Apple M5 Max"
becomes `apple_m5_max`), family `apple`, arch number 5. The family ranks after `sm`, `gfx` and
`intel`, before unknown families and `cpu`. No `tuned/*.apple_*.txt` exists, so each op borrows the
nearest table and warns once (on this tree: "gemm has no float table for apple_m5_max; borrowing
sm_120"). Every route on the Apple GPU is therefore an NVIDIA measurement applied to a different
GPU; tune before quoting a ratio. `select_tests` cover the key, the borrow order and the once-per-op
warning.

### Metal: build shape on macOS {#sycl-impl-metal-build-shape}

- **One dylib.** ld64 rejects undefined symbols in a dylib, and the component libraries reference
  each other in cycles, so macOS requires `BATCHLAS_MONOLITHIC_LIBRARY=ON` (the default there; `OFF`
  is a configure error, `cmake/BatchLASOptions.cmake:273-294`). `BATCHLAS_HIDDEN_VISIBILITY=OFF`
  (set by the presets, and the default whenever tests are built): the tests reach about 1,000
  unexported internal symbols.
- **OpenBLAS as LAPACKE and CBLAS.** Homebrew's keg-only OpenBLAS has no `liblapacke`; LAPACKE and
  CBLAS are inside `libopenblas.dylib`. `BATCHLAS_OPENBLAS_ROOT` points the search there, frameworks
  are off for the search so the SDK's vecLib `cblas.h`/`lapack.h` never pair with it, and the
  prefix's include directory goes on the library's include path.
- **`<float.h>` shim.** The macOS 27 SDK's `<math.h>` requests only `INFINITY`/`NAN` from
  `<float.h>` under modules, which libc++'s include guard then blocks, so `FLT_MAX` is undefined
  after `<cmath>`. A configure check enables `cmake/compat/float-h` only when that happens
  (`cmake/BatchLASDependencies.cmake`).
- `BATCHLAS_ACPP_SLM_OPTIN` (a CUDA interposer) is a configure error on macOS.

### Metal: tests and the failure ledger {#sycl-impl-metal-failure-ledger}

- Native candidate and pin suites have `Backend::SYCL` arms (float, complex<float>); the double
  configurations run as NETLIB on the host. Grid-ceiling oracles read the device's own ceiling.
- `tests/known-failures-acpp-metal.txt` is a third self-contained ledger, picked by
  `compare_failures.py` for an ACPP tree whose cache has `BATCHLAS_GPU_FAMILY_RESOLVED=METAL` (or
  `--gpu-family METAL`). Its `type_param` strings are macOS libc++ spellings
  (`std::__1::complex<float>`). Entries:
  - hemm, herk, her2k complex<float>: vendor-only rules, `NoRouteError` in a vendor-free build
    ([flat selection §11](flat-kernel-selection.md)).
  - `posv` pinned blocked, Upper, n = 33, complex<float>: no vendor-free potrf route
    ([known defects](known-defects.md#defect-19-upper-potrf-beyond-the-cta-ceiling-has-no-vendor-free-route)).
  - lanczos: [defect 3](known-defects.md#defect-3-lanczos-issues-a-two-column-multiply-and-consumes-one-column),
    shared with the other trees.
- `device_calls_tests` exits 77 (skip): there is no NVPTX image.
- Run ctest serially (`ctest --preset acpp-metal-tests` sets `-j1`).

## Open items {#sycl-impl-open-items}

| Item | Configuration | What would settle it |
| --- | --- | --- |
| Upstream opt-in above 48 KiB (A2) not requested | Linux acpp | An AdaptiveCpp issue or patch for `cuda_queue.cpp` / `sharedMemPerBlockOptin` |
| Three cells (geqrf, potrf `blocked`, syev; float n >= 512) still gain over 10% per item at twice the batch | Linux acpp A/B | Re-measure at a larger batch |
| Vendor-free acpp tree not measured | Linux acpp | A `-DBATCHLAS_ENABLE_VENDOR_BLAS=OFF` acpp A/B |
| cp.async gemm pipeline absent (A12), cost unknown | Linux acpp | A/B of the affected gemm tiles |
| `getrf_panel_reg<cdouble, 32>` stack array on acpp, cause not investigated | Linux acpp | PTX/IR inspection |
| No Apple tuning tables; the CMake local-memory row and the safe sub-group count are hypotheses | Metal | Tables tuned on the M5 Max |
| The patches are not upstream; each AdaptiveCpp bump must re-apply them and re-run the `sycl_tests` and the BatchLAS suite | Metal | Upstream submission |
| Annex G special values under the Metal `logb` ([device-code rules](#sycl-impl-metal-device-code-rules)) are untested | Metal | A special-value test |
| No performance A/B against another implementation on the Apple GPU | Metal | — |
