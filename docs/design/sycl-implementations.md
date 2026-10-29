# Two SYCL implementations: DPC++ and AdaptiveCpp {#design_sycl_implementations}

> **Status:** current · checked 2026-10-29 · Linux acpp: AdaptiveCpp 25.10.0, threadripper02
> (sm_120, CUDA 13.2), 2026-10-09; Metal: Apple M5 Max, 2026-10-13.

One source tree builds with intel/llvm DPC++ or AdaptiveCpp (`acpp`, generic SSCP target, kernels
JIT-compiled at first launch); a third configuration is acpp with its Metal backend on macOS
([AdaptiveCpp Metal](#sycl-impl-metal)). Options and presets: @ref guide_build. Measurements:
@ref perf_sycl_implementations.

## Design {#sycl-impl-design}

### Build: one cache variable, two detect modules {#sycl-impl-build}

- `BATCHLAS_SYCL_IMPL = AUTO | DPCPP | ACPP` (`cmake/BatchLASOptions.cmake`). AUTO picks ACPP when
  the compiler is `acpp`/`syclcc`/`syclcc-clang` or `--acpp-version` names AdaptiveCpp. `ACPP` with
  another compiler swaps in `acpp` from `PATH`, `/opt/adaptivecpp/bin` or `~/opt/adaptivecpp/bin`
  (`cmake/BatchLASCompilerBootstrap.cmake`). The result is cached as `BATCHLAS_SYCL_IMPL_RESOLVED`,
  which `compare_failures.py --build-dir` reads to pick the ledger.
- `CMakeLists.txt` includes `cmake/BatchLASSyclAcpp.cmake` for ACPP, `cmake/BatchLASDetectSYCL.cmake`
  for DPCPP. The acpp module reuses the DPC++ one for its functions only and sets every variable the
  build reads: `--acpp-targets=${BATCHLAS_ACPP_TARGETS}` on the same INTERFACE option libraries (no
  `-fsycl*`), devices from `acpp-info` (else the installed runtime backends), and in every TU
  `-include src/sycl/annexg_complex.hh` (A1) and `-ffp-contract` (A11).
- Device-link options (`BATCHLAS_SYCL_LINK_JOBS`, FTZ, `native_cpu`) are DPC++ only. On acpp
  `BATCHLAS_CPU_TARGET` takes only `auto` or `none` (@ref design_cpu_target_detection).
- `backend_config.h` carries `BATCHLAS_SYCL_IMPL_DPCPP` / `_ACPP`. acpp runtime names appear only in
  `src/sycl/impl.hh` and `src/sycl/sscp_target.hh`.
- A consumer must use the same implementation (`BatchLASConfig.cmake` records `BatchLAS_SYCL_IMPL`);
  an acpp install ships `include/batchlas/acpp/annexg_complex.hh` and `-include`s it through
  `BatchLAS_SYCL_COMPILE_OPTIONS`.

### Runtime: one interop seam {#sycl-impl-runtime-interop-seam}

`src/sycl/impl.hh` (`batchlas::impl`) holds every implementation difference above the kernel level;
on DPC++ each helper is the plain DPC++ call.

| Helper | acpp |
| --- | --- |
| `kCudaBackend`, `backend_v<N>`, `is_cuda`, `is_hip`, `is_metal` | `backend::cuda` / `hip` / `metal` |
| `backend_of(queue)` | `q.get_device().get_backend()` |
| `cuda_cc(device)` | parses `"sm_89"` (DPC++: `"8.9"`) |
| `native_stream`, `query_stream<N>` | the in-order executor's stream, possibly `nullptr` |
| `run_native<N>(q, f)` | `f` inside `AdaptiveCpp_enqueue_custom_operation`, submitted untraced |
| `submit_barrier(q[, wait_list])` | last event (in-order) or a wait-list barrier |
| `kernel_max_wg_size<K>` | `min(max_work_group_size, 256)`: the device limit aborts a register-heavy kernel |
| `parallel_for(handler, nd_range, ...)` | a no-op custom operation for an empty range |
| `on_async_errors`, `throw_async_errors`, `throw_launch_errors`, `wait_and_throw` | the error sink ([asynchronous errors](#sycl-impl-asynchronous-errors)) |
| `check_group_count`, `check_global_range` | CUDA grid-limit check; on Metal the 2^32 - 1 global-range check |

Kernel-level differences live in `src/sycl/sscp_target.hh`, `src/sycl/kernel_attrs.hh`,
`src/sycl/local_mem.hh` and `include/batchlas/util/group-collectives.hh`; DPC++-only intrinsics and
PTX `asm` are `#if`-guarded in `sg_partition/` and `src/ops/gemm/kernels/`. Every vendor call runs as
`impl::run_native<impl::Native::Cuda>(*ctx, [=](auto s){ handle.setStream(s); ... })`.

### Device: target dispatch at JIT time {#sycl-impl-device-target-dispatch}

- `sycl_impl::on_ptx()`, `on_metal()`, `target_arch()` (`src/sycl/sscp_target.hh`): `constexpr` from
  `__SYCL_DEVICE_ONLY__`/`__NVPTX__`/`__SYCL_CUDA_ARCH__` on DPC++; on acpp generic, which compiles
  one IR for every target and defines neither macro, JIT-reflection queries inside
  `__acpp_if_target_sscp`, and the JIT deletes the dead branch.
- **Trap:** a target intrinsic or inline `asm` must also sit inside `__acpp_if_target_sscp(...)`;
  `on_ptx()` alone does not keep it out of the x86 host object, where instruction selection rejects it.
- `sg_partition`'s `SscpBackend` (`backend_sscp.hh`) reuses `NvptxBackend` on PTX; its `__nvvm_*`
  names are asm labels on the LLVM intrinsics, so they keep `convergent`. Other targets take
  `GenericBackend`.
- `src/sycl/kernel_attrs.hh`: on acpp `BATCHLAS_REQD_SG_SIZE` and `BATCHLAS_LAUNCH_BOUNDS` are empty
  and `BATCHLAS_UNROLL_FULL` is `clang loop unroll(full)` (SSCP unrolls a bare `#pragma unroll` only
  partially). `Device::supports_sub_group_size(n)` is true only for the device's single sub-group
  size, so `can_run`s that need sg 32 refuse other devices.

### Multipass CUDA: --acpp-targets=cuda:sm_XX {#sycl-impl-cuda-aot}

acpp's ahead-of-time CUDA flow: a host pass plus an NVPTX device pass by acpp's clang (LLVM 20), no JIT.

- **Toolkit.** Clang 20 rejects CUDA 13 headers: the flow needs CUDA 12.x (`BATCHLAS_ACPP_CUDA_PATH`,
  configure fails on 13.x), whose vendor libraries are linked. The flags go into
  `BatchLAS_SYCL_COMPILE_OPTIONS`. Configure ignores a `$CUDACXX` outside the chosen toolkit, seeds
  `CUDAToolkit_ROOT` from acpp's `default-cuda-path`, and fails when FindCUDAToolkit's toolkit is not
  the one acpp compiles against (stale cache in an old tree).
- **No GPU at configure:** an explicit `cuda:sm_XX` builds the CUDA backend from `librt-backend-cuda`.
- **No `SYCL_EXTERNAL`:** the secular-equation root solvers (`stedc_secular_roots.hh`) are `inline`
  where it is undefined. Device code may not name a `__host__`-only function.
- **Target dispatch:** `on_ptx()` is true and `target_arch()` is `__CUDA_ARCH__ / 10` in the device
  pass; `redux` keys on `BATCHLAS_NVPTX_ARCH`.
- **Work-group reduce.** `portable::reduce_over_group` takes `detail::wg_reduce_floating` in the CUDA
  and HIP device passes too (`kExactWgReducePass`) only because acpp's multipass reduce, exact in
  probes, reads as racy (static scratch returned after the last barrier,
  `generic/hiplike/group_functions.hpp:36-71`); `DeviceBlasTest.PortableWorkGroupDoubleSumIsExact`
  guards the replacement.
- **Contraction:** `BATCHLAS_FP_CONTRACT` (default `on`) over the driver's `fast`
  ([build guide](../guide/build.md#guide_build_fp_contract)).
- **Tests** (sm_89): failures are ledgered by cause in `tests/known-failures-acpp-cuda.txt`. The
  getrf register-panel leaf is capped at 256 rows on acpp
  ([Defect 22](known-defects.md#getrf-register-panel-unlaunched-on-acpp-multipass)). No CPU target
  exists in this flow; `device_calls_tests` and `acpp_jit_tests` are not registered.

### Local memory: the launch budget {#sycl-impl-slm-budget}

| | DPC++ (CUDA) | acpp 25.10 (CUDA) | acpp + `BATCHLAS_ACPP_SLM_OPTIN` |
| --- | --- | --- | --- |
| `info::device::local_mem_size` | 101,376 B (opt-in maximum) | 49,152 B | 49,152 B |
| Launches above 48 KiB | opted in by the runtime | fail, `CU:1` | opted in by the interposer |
| Budget (less the 4 KiB reserve) | 97,280 B | 45,056 B | 97,280 B |

- **One seam.** `impl::local_mem_bytes` (`src/sycl/local_mem.hh`) is the only reader of
  `local_mem_size`; `DeviceProperty::LOCAL_MEM_SIZE` and `select::Device::slm_budget` go through it.
- **Every `can_run` that can exceed 48 KiB reads the budget**, so DPC++ routes do not move. Above
  48 KiB: gemm `reg:m=128:n=64:k=32` (49,920 B, `RegCfg::slm_bytes()`), herk `gram` (65,536 B,
  `gram_slm_bytes`), gesvd `jacobi` C = 64 (71,752 B, `gesvd_jacobi_max_dim`), `syev_cta_fused` with
  an explicit multiplier (54,272 B).
- **The CTA multiplier clamp counts whole steps** (`resident::cta_fit_wg_multiplier`): one step is
  lcm(P, 32) lanes, and the kernels throw `unsupported` when one step does not fit. A clamp by
  problem count over-admitted by up to 32/P and aborted on DPC++ ("Excessive allocation of local memory").
- Tests whose premise is the 99 KiB opt-in skip on `test_utils::kSlmCappedAt48KiB`.
- **Opt-in variant.** `-DBATCHLAS_ACPP_SLM_OPTIN=ON` (acpp + CUDA only) builds
  `libbatchlas_acpp_slm_optin.so` (`src/sycl/acpp_slm_optin.cc`), which interposes `cuLaunchKernel`,
  sets `MAX_DYNAMIC_SHARED_SIZE_BYTES` once per function above 48 KiB and reports the opt-in maximum.
  A recorded A/B variant, not the default.
- **Upstream** (v25.10.0): `src/runtime/cuda/cuda_queue.cpp:187-201` launches without the attribute;
  `cuda_hardware_manager.cpp:418-419` reports `sharedMemPerBlock`. The host device reports `SIZE_MAX`,
  so `describe` clamps the budget to `int64_t`.

### Asynchronous errors and launch failures {#sycl-impl-asynchronous-errors}

| | DPC++ | acpp 25.10 |
| --- | --- | --- |
| Failed launch (grid over 65,535 in y/z, unservable local memory) | `sycl::exception` from the submit | registered in one process-wide error list; the submit returns |
| `queue::wait()` after a device fault | reports nothing | registers the fault in that list |
| `~queue()` | no handler call | `throw_asynchronous()` over the whole list |
| Empty `nd_range` | no-op | a 0-block grid, `CU:1` |

So a rethrowing handler on acpp turns any unconsumed error into `std::terminate` at the next
`sycl::queue` destructor, copies included.

- **Handler (acpp).** `impl::on_async_errors` parks errors in a leaked process-wide sink and never
  throws; `impl::throw_async_errors(q)` rethrows the first. DPC++ keeps the rethrowing handler.
- **Where errors surface.** `QueueImpl::submit_and_record` calls `impl::throw_launch_errors` after
  the submit, so a failed launch throws from the launching call as on DPC++; `Queue::wait()`,
  `wait_and_throw()` and `impl::wait_and_throw(event, q)` throw what a wait registered.
- **Grid limit.** `impl::check_group_count` refuses an over-limit `nd_range` before launch with
  DPC++'s `errc::nd_range` message, in `QueueImpl::parallel_for(nd_range)` and the five command-group
  tile launchers (syrk gram/triangular, syr2k, trmm, `expand_mirrored`).
- **Empty `nd_range`.** Every command-group launch goes through `impl::parallel_for(handler, ...)`,
  which submits a no-op custom operation, keeping in-order event ordering.
- **Vendor calls** submit untraced, so they never appear in `BATCHLAS_KERNEL_TRACE`, as on DPC++.
  Tests: `util_device_queue_tests` `QueueAsyncErrors.*`.
- **Upstream** (v25.10.0): `sycl/queue.hpp:288-290` (`~queue`), `glue/error.hpp:145` (global list),
  `queue.hpp:371-374` (`wait`), `sycl/handler.hpp:379` (no empty-range guard on `nd_range`),
  `sycl/event.hpp:36-40` (dropped handler, so `event::wait_and_throw()` throws `std::bad_function_call`).

### Group algorithms: the portable wrappers {#sycl-impl-group-algorithms}

`include/batchlas/util/group-collectives.hh` (`batchlas::portable::`): under DPC++ each name is a
using-declaration of the `sycl::` function (md5-identical objects); under acpp it replaces a broken one.

| acpp 25.10 defect (SSCP) | Effect | Replacement |
| --- | --- | --- |
| Work-group `reduce_over_group` of float/double rounds to ~16/~42 mantissa bits (the broadcast bit-casts through the float scratch, `sscp/builtins/detail/reduction.hpp:101-103`) | norm, geqrf/orgqr, symv, sytrd, steqr, syevx, bdsdc wrong | sub-group reduce, then a typed local scratch (`detail::wg_reduce_floating`) |
| `joint_reduce(maximum)` pads with `numeric_limits<T>::min()` | wrong max of an all-negative range | start from `sycl::known_identity_v` |
| In-place `joint_exclusive_scan` overwrites unread input across chunks; one element drops `init` | stedc lost eigenvalues for merges > 129 | chunked scan carrying the running total |
| Host `joint_reduce` without `init` on `const T*` does not compile | — | overload keeping SYCL's item-to-element mapping |

Every floating `reduce_over_group`/`joint_reduce`/`joint_exclusive_scan` in `src/` goes through
`portable::`. Guards: `DeviceBlasTest.PortableGroupReductionsAreExact`,
`StedcDeflationScan.PortableJointExclusiveScanInPlaceAcrossChunks`.

## Compile-level differences: portable spellings {#sycl-impl-compile-differences}

DPC++ extensions or acpp gaps with a spelling legal in both. Use the portable form.

| # | Construct | Portable form |
| --- | --- | --- |
| C1, C2, C3, C11 | `q.get_backend()`, `backend::ext_oneapi_cuda`, `get_native`, `kernel_bundle`, `ext_oneapi_submit_barrier` | `impl::` helpers ([the interop seam](#sycl-impl-runtime-interop-seam)) |
| C4, C6 | `joint_reduce` over accessor iterators; a ternary between an accessor iterator and `T*` | raw pointers `&acc[0]` |
| C5, C9 | `group_broadcast` / `select_from_group` of a struct | `bl::group_broadcast` (word split; `group-invoke.hh`) |
| C7 | `sycl::vec<std::complex<T>, N>` | plain arrays |
| C8 | group reduce of `std::complex` | reduce `real` and `imag` separately |
| C10 | oneDPL random | `src/util/philox.hh` |
| C12 | `[[sycl::reqd_sub_group_size]]`, `[[intel::...]]` launch bounds | `BATCHLAS_REQD_SG_SIZE`, `BATCHLAS_LAUNCH_BOUNDS` |

Rule: `src/`, `tests/`, `benchmarks/` and `tools/` compile under both compilers
([CI coverage](#sycl-impl-phases)), and a seam change leaves the DPC++ route diff empty.

## Implementation differences at run time {#sycl-impl-runtime-differences}

`tests/known-failures-acpp.txt` cites these rows. Costs: @ref perf_sycl_implementations.

| # | Behaviour on acpp | Cause | Resolution |
| --- | --- | --- | --- |
| A1 | `ptxas fatal: Unresolved extern function '__mulsc3'` | `std::complex` `*` lowers to Annex G libcalls with no SSCP device definition | device `__mulsc3`/`__muldc3`/`__divsc3`/`__divdc3` in `src/sycl/annexg_complex.hh`, force-included; `acpp_jit_tests` catches regressions |
| A2 | Launches above 48 KiB fail, CUDA error 1 | acpp reports 48 KiB and never opts in | `can_run`s read the budget ([local-memory budget](#sycl-impl-slm-budget)); Auto routes unchanged, budget-derived block sizes cost 1.22-2.66 in pinned cells ([perf](../perf/sycl-implementations.md#sycl-impl-ab-the-48-kib-budget-and-the-opt-in-variant)) |
| A3 | One bad launch would kill the process | default async handler terminates | [asynchronous errors](#sycl-impl-asynchronous-errors) |
| A4 | `info::device::version` is `"sm_120"` | different version string | `impl::cuda_cc` parses both; tables key `[sm_120]` on both |
| A5 | Hangs under divergence without `SscpBackend` | SSCP never sees `__NVPTX__`, so `sg_partition` was unmasked | [JIT-time target dispatch](#sycl-impl-device-target-dispatch) ([perf](../perf/sycl-implementations.md#sycl-impl-ab-sub-group-cta-kernels)) |
| A6 | Out-of-order queues segfault in vendor calls | stream not ordered with the queue | `run_native` through `AdaptiveCpp_enqueue_custom_operation`; 3-17 us per short call |
| A7 | `Queue::native_handle()` null under `SYCL_EXT_ONEAPI_BACKEND_CUDA` | DPC++-only macro | [the interop seam](#sycl-impl-runtime-interop-seam) |
| A8 | Death tests hang | `fork()` after the runtime's threads start | `death_test_style="threadsafe"` |
| A9 | Wrong answers (CTA potrf/getrf cdouble, orgqr, norm, geqrf, ...) | group-algorithm bugs, the 48 KiB budget | [group algorithms](#sycl-impl-group-algorithms), [budget](#sycl-impl-slm-budget); residue in the ledger |
| A10 | `BATCHLAS_LAUNCH_BOUNDS` empty; `reqd_sub_group_size` ignored | single-pass SSCP emits no `minnctapersm` | accepted: removing the bounds on DPC++ costs 0.99-1.03, so acpp's tiny-kernel losses are codegen ([perf](../perf/sycl-implementations.md#sycl-impl-ab-launch-bounds-r10)) |
| A11 | Contraction differed (acpp `fast` at -O2, DPC++ `on`) | driver defaults | `-ffp-contract=on` in `BatchLASSyclAcpp.cmake`; other FP settings already match |
| A12 | cp.async gemm pipeline compiled out | `BATCHLAS_GEMM_CP_ASYNC` needs `__NVPTX__`/`__SYCL_CUDA_ARCH__` | correct, slower; cost not measured |

## Correctness A/B {#sycl-impl-correctness-ab}

- Each tree has its own self-contained ledger (`tests/known-failures{,-acpp,-acpp-cuda,-acpp-metal}.txt`);
  `compare_failures.py --sycl-impl` / `--build-dir` picks it.
- **"acpp supported"** means the acpp tree's failing set is the DPC++ set on the same box plus acpp
  ledger entries, each with a cause from [runtime differences](#sycl-impl-runtime-differences).
- **Accuracy:** `steqr_accuracy`, `eigensolver_accuracy` and `orthogonality_accuracy` run in both
  trees on identical seeds with FP flags matched (A11); `scripts/impl_diff.py` reports per-cell
  residual ratios.
- **Route parity:** `scripts/route_diff.sh` across the trees; a difference is A2 or a bug.

### The JIT-ed PTX check {#sycl-impl-jit-ptx-check}

`scripts/check_acpp_jit.py`, ctest `acpp_jit_tests` (an acpp tree with the CUDA backend; exit 77
when no PTX appears) is the acpp counterpart of the DPC++-only `device_calls_tests`.

- acpp caches the PTX it hands the driver in `<ACPP_APPDB_DIR>/apps/<app>-<hash>/jit-cache/*.jit`,
  one `.entry` per file. Only launched kernels appear, so the test first runs a set of CUDA cases
  against a fresh appdb.
- **calls:** any `call` or `.extern .func` fails (SSCP inlines every callee); this shows A1 as
  `-> __mulsc3`.
- **required:** the `REQUIRED` families must be JIT-ed at least once.
- **stack:** the register-resident families (`RESIDENT`) must have no `__local_depot`; known depots,
  with byte ceilings, DPC++ comparison and reasons, are `STACK_KNOWN` in the script.

The JIT's `'+ptx88' is not a recognized feature` warning is harmless (LLVM 20 emits `.version 8.7`).

## Performance A/B protocol {#sycl-impl-performance-ab}

On top of @ref dev_agent_guide §10:

- **JIT.** A per-campaign `ACPP_APPDB_DIR`, cleared on driver or acpp change; throwaway passes until
  the "new binaries JIT-compiled" warning stops. Pin and record `ACPP_ADAPTIVITY_LEVEL` (report 1;
  level 2 specializes invariant arguments after ~1024 launches, so it is a separate arm).
- **Launch path.** In-order queues, `ACPP_RT_SCHEDULER=direct`; compare at saturation only.
- **Provenance.** benchviz records `sycl_impl` and the `ACPP_*` knobs per build and sets them unless
  the environment does; `store.py` warns on an acpp fp contract other than `on`; `benchviz ab`
  interleaves two builds.
- **Tables** are keyed by device, not implementation ([alternatives](#sycl-impl-decisions)).

## CI coverage of the acpp build {#sycl-impl-phases}

- Hosted job `acpp-compile` ([CI](../ci.md#the-acpp-compile-job), `.github/ci/build_adaptivecpp.sh`)
  builds AdaptiveCpp 25.10 (cached) and compiles every target with `--acpp-targets=generic`, CUDA
  off; it runs nothing and turns red when a DPC++-only construct leaks in.
- The full acpp gate is local: an `acpp-tests` tree against its ledger, plus `acpp_jit_tests`.

## Rejected alternatives {#sycl-impl-decisions}

| Alternative | Result | Verdict |
| --- | --- | --- |
| `-fcx-limited-range` instead of Annex G definitions (A1) | fixes the link, changes NaN/Inf results | rejected |
| Opt in above 48 KiB by default (A2) | pinned cells 0.99-1.20 vs 1.22-2.66 without | kept as `BATCHLAS_ACPP_SLM_OPTIN`, not default |
| Drop vendor libraries from the acpp build | `run_native` vendor cells 1.00 at >= 1 ms | rejected |
| Per-implementation tuning tables | Auto takes the same route in all 90 cells ([perf](../perf/sycl-implementations.md#sycl-impl-ab-per-implementation-tables)) | not added; revisit with a newer acpp LLVM |

## AdaptiveCpp Metal {#sycl-impl-metal}

The Apple GPU through AdaptiveCpp's Metal backend (AdaptiveCpp `develop` `9ec394ef` plus local
patches, M5 Max): vendor-free, no device fp64, and every table it reads was measured on NVIDIA.
Recipe and traps: [macOS build](../guide/build.md#macos-apple-gpu-through-adaptivecpp-metal),
@ref dev_agent_guide "Known Pitfalls". Each switch below is a configure result false elsewhere
(`BATCHLAS_HAS_SYCL_BACKEND`, `BATCHLAS_DEVICE_NO_VECTOR_TYPES`) or a runtime test false off Metal
(`impl::is_metal`, `sycl_impl::on_metal()`). The local AdaptiveCpp patches, each with its reproducer
and what it broke in BatchLAS, are in `scripts/macos/adaptivecpp-patches/README.md`.

### Metal: the SSCP-to-MSL pipeline and configure {#sycl-impl-metal-pipeline}

- At each kernel's first launch `llvm-to-metal` lowers the generic IR to MSL and
  `newLibraryWithSource` compiles it; errors surface at that launch.
- **Configure.** A `Metal` GPU in `acpp-info` sets `BATCHLAS_DETECTED_APPLE_GPU`, arch
  `apple_gpu_<arch>` and, unless `BATCHLAS_ENABLE_METAL=OFF`, `BATCHLAS_HAS_SYCL_BACKEND` and
  `BATCHLAS_DEVICE_NO_VECTOR_TYPES`. The OpenMP device is mandatory (double and verification run
  there): no `BATCHLAS_HAS_CPU_TARGET` is a configure error.

### Metal: Backend::SYCL and the no-fp64 policy {#sycl-impl-metal-fp64-policy}

MSL has no `double`, so a double kernel cannot be JIT-compiled for the GPU.

- **`Backend::SYCL`** (`include/batchlas/blas/enums.hh`): native kernels on a GPU with no vendor
  library; `Queue::backend()` resolves an Apple GPU to it, and `select::native_gpu_backend<B>` is
  CUDA or SYCL, so every native GPU family runs here too.
- **Instantiation.** `Backend::SYCL` gets float and complex<float>, plus double only under
  `BATCHLAS_HAS_DEVICE_FP64` (off on Metal), so a stray `<SYCL, double>` is a link error.
- **The refusal.** The deducing entry points go through `detail::with_backend_gated<NeedsFp64>`
  (`queue-dispatch.hh`): when any container argument holds double or complex<double>
  (`detail::needs_fp64_v`; scalars do not count), the SYCL branch is compiled out and throws
  `batchlas::unsupported` naming the op and the fix (a CPU `Queue`). `with_backend_for<T>` is the
  same gate; plain `with_backend` has none.
- **No silent reroute.** Double runs on a CPU `Queue` (NETLIB: OpenBLAS plus the OpenMP device).
  `Queue::set_backend(NETLIB)` on the Apple GPU throws, since NETLIB still launches native kernels on
  the queue's device.
- **No internal double on Metal:** complex<float> `lartg` rounds in float on Metal; queue-less
  `Matrix` helpers use `Device::default_device_for<T>()`; `batchlas::verify` checks on a CPU queue
  ([verification](verification.md#verification-no-fp64-device)). Tests: `backend_dispatch_tests`
  `NoDeviceFp64.*`.
- **`ACPP_ADAPTIVITY_LEVEL=0` is refused:** level 0 JITs a TU's
  kernels together, host double ones included, so every float launch in the TU fails. `Queue`
  constructors throw `unsupported` on an Apple GPU at level 0 (`require_metal_adaptivity`,
  `src/util/queue-impl.cc`; test `backend_dispatch_tests_adaptivity0`).

### Metal: launch limits {#sycl-impl-metal-launch-limits}

Measured on the M5 Max with patch 0003.

| Limit | Value | Where BatchLAS models it |
| --- | --- | --- |
| Threadgroup memory reported | 32,768 B | the `apple_gpu_*` row in `cmake/BatchLASDetectSYCL.cmake` (a hypothesis) |
| Hidden per-lane scratch | AdaptiveCpp adds `8 * wg` bytes to every launch | `impl::local_mem_scratch_per_item` = 8; `local_mem_bytes(d)` assumes the largest wg (24,576 B at 1024) |
| Selector budget | 20,480 B at max wg (less the 4 KiB reserve) | `select::Device::slm_per_item`, `slm_budget_for(wg)`; gemm `reg` reads the per-width figure |
| Threads per threadgroup | 1024 for every pipeline (the compiler spills) | the getrs_fused register cap is skipped on Metal |
| Grid | no 65,535 cap; a dimension of 2^32 work-items or more runs with wrong ids, no error | `impl::check_global_range` throws above 2^32 - 1; `select::Device::max_groups_yz` = 2^32 - 1 in the gemm, syrk, syr2k, trmm, symm `can_run` |
| Co-residency | not promised; atomics relaxed only | the latrd grid-barrier path is off on Metal ([sytrd](../perf/sytrd.md#sytrd-the-latrd-grid-kernel-and-its-co-residency-cap)) |
| Sub-groups per work-group (safe) | 1 | the unrecognised-GPU default (a hypothesis) |

Guards: `resident_capacity_tests` launches at and one granule past the budget, 65,537 groups per
dimension and 2^32 - 1024 work-items. `select::describe` hard-codes 8 and 2^32 - 1 (it cannot include
`src/sycl/`); these tests hold the copies equal.

### Metal: coarse-grained events and error delivery {#sycl-impl-metal-coarse-events}

- **Coarse events by default.** `QueueImpl::make_queue_properties` adds
  `AdaptiveCpp_coarse_grained_events` on Metal unless profiling is on or
  `BATCHLAS_METAL_FLUSH_EACH_OP` is set; otherwise AdaptiveCpp commits one command buffer per op.
  Such an event completes with the whole queue (never early, no timestamps). Flushing each op cost
  1.33-1.72x on back-to-back small potrf/gemm calls, neutral with a host wait per call (2026-10-13).
- **`~QueueImpl` always drains on Metal**: coarse events leave the command buffer open, so a derived
  queue's work could outlive the caller's wait. Other builds drain only when the arena holds blocks.
- **Errors are asynchronous**: a command-buffer failure surfaces at `Queue::wait()` or the next
  `impl::throw_launch_errors`, and under coarse events may cover many ops
  (`BATCHLAS_METAL_FLUSH_EACH_OP=1` attributes it).

### Metal: device-code rules {#sycl-impl-metal-device-code-rules}

| Metal fact | Consequence | Where |
| --- | --- | --- |
| No generic address space (a run-time local/global pointer choice is rejected) | one body per storage combination under `on_metal()` | `sytrd_sb2st_cta.cc`, `tridiag_solver.cc` |
| No floating-point atomics on threadgroup memory | One/Inf norms give each sum to one work-item | `src/util/norm.cc` |
| No static local memory | `wg_reduce_floating` uses AdaptiveCpp's reduce on Metal | `group-collectives.hh` |
| No vector types in the emitter | `Vec4`, `tiny_r4` are structs, `Raw16` an `unsigned __int128` under `BATCHLAS_DEVICE_NO_VECTOR_TYPES` | `vec4.hh`, `tiny_device.hh`, `register_64x64_k16_wide.hh` |
| Float subnormals flush to zero; `info::device` claims denorms | tests needing a subnormal probe the device | `TrsmNativeSgLeft.NonFiniteReciprocalFallsBackToDivision` |
| `logb(0)`, `logb(inf)` are finite | Annex G division special values may be wrong; untested | `annexg::div` |
| arm64 hosts reject the `"l"` asm constraint | PTX L2-prefetch asm only on x86 hosts | `register_128x128.hh` |

`on_metal()` is JIT-time reflection (the backend spelled as 4: acpp 25.10 has no `metal`
enumerator); `group-collectives.hh` keeps a private copy because an installed header cannot include `src/`.

### Metal: table key, build shape and failure ledger {#sycl-impl-metal-failure-ledger}

- `select::apple_device_key` keys "Apple M5 Max" as `apple_m5_max` (family `apple`); with no
  `tuned/*.apple_*.txt` each op borrows the nearest table and warns once (`select_tests`).
- ld64 rejects the components' cyclic undefined symbols, so macOS requires
  `BATCHLAS_MONOLITHIC_LIBRARY=ON`. LAPACKE and CBLAS come from Homebrew's `libopenblas.dylib`
  (`BATCHLAS_OPENBLAS_ROOT`, frameworks off so vecLib headers never pair with it). A configure check
  enables the `cmake/compat/float-h` shim when the macOS 27 SDK leaves `FLT_MAX` undefined.
- Native suites have `Backend::SYCL` arms (float, complex<float>); double runs as NETLIB on the host.
  `tests/known-failures-acpp-metal.txt` (libc++ `std::__1::` spellings) holds hemm/herk/her2k
  complex<float> (vendor-only, `NoRouteError`), Upper `posv` n = 33 complex<float>
  ([Defect 19](known-defects.md#defect-19-upper-potrf-beyond-the-cta-ceiling-has-no-vendor-free-route))
  and lanczos ([Defect 3](known-defects.md#defect-3-lanczos-issues-a-two-column-multiply-and-consumes-one-column)).

## Open items {#sycl-impl-open-items}

| Item | Configuration | What would settle it |
| --- | --- | --- |
| Upstream opt-in above 48 KiB (A2) not requested | Linux acpp | an AdaptiveCpp issue or patch |
| Three cells (geqrf, potrf `blocked`, syev; float n >= 512) gain over 10% per item at twice the batch | Linux acpp | re-measure at a larger batch |
| Vendor-free acpp tree not measured | Linux acpp | a `-DBATCHLAS_ENABLE_VENDOR_BLAS=OFF` A/B |
| cp.async gemm pipeline absent (A12), cost unknown | Linux acpp | A/B of the affected tiles |
| `getrf_panel_reg<cdouble, 32>` stack array, cause not investigated | Linux acpp | PTX/IR inspection |
| No Apple tuning tables; the CMake local-memory row and safe sub-group count are hypotheses | Metal | tables tuned on the M5 Max |
| Patches not upstream; each AdaptiveCpp bump must re-apply them and re-run the suites | Metal | upstream submission |
| Annex G special values under Metal `logb` untested | Metal | a special-value test |
