# CPU target detection {#design_cpu_target_detection}

> **Status:** current · checked 2026-10-21

Configure decides whether CPU device code is compiled (`BATCHLAS_HAS_CPU_TARGET`) and drops what
depends on it. Logic: `cmake/BatchLASDetectSYCL.cmake` (DPC++) and `cmake/BatchLASSyclAcpp.cmake`
(AdaptiveCpp). Option: `BATCHLAS_CPU_TARGET` (`cmake/BatchLASOptions.cmake:206`, default `auto`).

A GPU-only DPC++ may have no CPU backend. Without a CPU entry in `-fsycl-targets`, code that submits
kernels to a CPU device (the `NETLIB` SYCL helpers, `Queue("cpu")` work that is not host LAPACK) fails
to compile or finds no kernel at run time.

## Two stages

DPC++ only. AdaptiveCpp: [below](#cpu-target-adaptivecpp).

**Stage 1: `detect_sycl_cpu_target()`** (`cmake/BatchLASDetectSYCL.cmake:395-449`) may append a CPU
target to `BATCHLAS_SYCL_TARGETS`:

| `BATCHLAS_CPU_TARGET` | Effect |
| --- | --- |
| `none` | No CPU target; `OFF`. |
| `native_cpu` or `spir64_x86_64` | Appended; `ON`. |
| `auto` (default) | `sycl-ls` lists `[opencl:cpu]` or `[host:cpu]`: append `spir64_x86_64`. Lists `[native_cpu:cpu]`: append `native_cpu`. Otherwise, or if `sycl-ls` is missing or fails: `OFF`. |
| Anything else | Configure `WARNING`, then treated as `auto`. |

**Stage 2: the final gate** (`cmake/BatchLASDetectSYCL.cmake:560-578`) runs after
`-- Using SYCL targets: ...` is printed. It decides from the compile flags, not from device detection.

1. An explicit `native_cpu` or `spir64_x86_64` forces `ON`.
2. If no entry of `BATCHLAS_SYCL_TARGETS` matches `cpu`, `spir64` or `native_cpu`, the result is `OFF`,
   whatever `sycl-ls` said. Configure then prints:
   ```
   -- CPU device detected by sycl-ls, but no CPU target in fsycl-targets
   -- CPU kernels will not be compiled - disabling CPU-dependent tests/benchmarks
   ```
3. `BATCHLAS_CPU_TARGET=none` forces `OFF` (and drops the CPU target itself).

The result is exported as `BATCHLAS_HAS_CPU_TARGET` (`cmake/backend_config.h.in`, `#cmakedefine01`) and
to Python as `compiled_features()["has_cpu_target"]` (`python/batchlas/bindings/support.hh`).

`batchlas_extensions_cta` is built `NO_CPU_TARGETS` (`src/CMakeLists.txt:348`): its objects use
`BATCHLAS_SYCL_TARGETS_NO_CPU` (the target list without `cpu`/`spir64` entries) whatever this gate
says.

## AdaptiveCpp {#cpu-target-adaptivecpp}

acpp compiles every backend from one target set, so there is no CPU-free variant of the flags
(`cmake/BatchLASSyclAcpp.cmake:204-240`):

| Condition | `BATCHLAS_HAS_CPU_TARGET` |
| --- | --- |
| `BATCHLAS_CPU_TARGET=none` | `OFF` |
| an OpenMP device is visible and `BATCHLAS_ACPP_TARGETS` starts with `generic` or `omp` | `ON` |
| otherwise | `OFF` |

`native_cpu` and `spir64_x86_64` are DPC++ targets: under acpp they print a `WARNING` and act as
`auto`. A Metal configure without a CPU target is a `FATAL_ERROR`, because double and the
verification layer run on the OpenMP device (@ref design_sycl_implementations).

## What `BATCHLAS_HAS_CPU_TARGET=OFF` turns off

- `minibench_cli_tests` is not built. It is the only member of `CPU_SYCL_DEPENDENT_TESTS`
  (`tests/CMakeLists.txt:138-148`).
- Every NETLIB instantiation of every typed suite is compiled out. `backend_types`,
  `backend_types_filtered` and `backend_types_complex` (`tests/test_utils.hh:132-213`) add the NETLIB
  block only when `BATCHLAS_HAS_HOST_BACKEND && BATCHLAS_HAS_CPU_TARGET`. Any NETLIB configuration
  still reached is skipped in `BatchLASTest::SetUp()` (`tests/test_utils.hh:365`).

> **Note:** Other test binaries still build (for example `ormqr_cta_tests`, `syev_cta_tests`,
> `sytrd_cta_tests`), because their GPU kernels compile and host LAPACK needs no SYCL kernel. Their
> NETLIB cases are absent. Benchmarks have no base and CPU-dependent split.

## Traps

- **Typed-test indices shift.** Without the NETLIB block, `SteqrTest/0` is float/CUDA, not
  float/NETLIB. `tests/known-failures.txt` therefore pins each typed entry with `type_param=`
  (a mismatch is a hard error in `compare_failures.py`), and its entries are verified under the
  `dev-tests` preset only.
- **`dev-gpu` and `dev-gpu-tests` set `BATCHLAS_CPU_TARGET=none`.** A cold build is 13-25% faster
  (248 s against 284-334 s full-target, threadripper02; agent guide §7). The preset description
  claims about 30% off every compile and device link; that figure has no recorded measurement.
  Coverage is reduced, and `ctest` still reports green: never the pre-push gate.
- **`native_cpu` has sub-group size 1.** A CPU target does not make CTA kernels runnable. See
  [Running the BatchLAS tests](../../tests/README.md).
- **Stage 2 matches by substring.** Any target name containing `cpu` counts.
