# CPU target detection {#design_cpu_target_detection}

> **Status:** current · line numbers checked 2026-10-08.

Configure decides whether CPU device code is compiled (`BATCHLAS_HAS_CPU_TARGET`) and drops what
depends on it. Logic: `cmake/BatchLASDetectSYCL.cmake`. Options: `cmake/BatchLASOptions.cmake`.

A GPU-only DPC++ may have no CPU backend. Without a CPU entry in `-fsycl-targets`, code that submits
kernels to a CPU device (the `NETLIB` SYCL helpers, `Queue("cpu")` work that is not host LAPACK) fails
to compile or finds no kernel at run time.

## Two stages

**Stage 1: `detect_sycl_cpu_target()`** (`cmake/BatchLASDetectSYCL.cmake:477`) may append a CPU target
to `BATCHLAS_SYCL_TARGETS`:

| `BATCHLAS_CPU_TARGET` | Effect |
| --- | --- |
| `none` | No CPU target; `OFF`. |
| `native_cpu` or `spir64_x86_64` | Appended; `ON`. |
| `auto` (default, `cmake/BatchLASOptions.cmake:117`) | `sycl-ls` lists `[opencl:cpu]` or `[host:cpu]`: append `spir64_x86_64`. Lists `[native_cpu:cpu]`: append `native_cpu`. Otherwise, or if `sycl-ls` is missing or fails: `OFF`. |
| Anything else | Configure `WARNING`, then treated as `auto`. |

**Stage 2: the final gate** (`cmake/BatchLASDetectSYCL.cmake:637-660`) runs after
`-- Using SYCL targets: ...` is printed. It decides from the compile flags, not from device detection.

1. An explicit `native_cpu` or `spir64_x86_64` forces `ON`.
2. If no entry of `BATCHLAS_SYCL_TARGETS` matches `cpu`, `spir64` or `native_cpu`, the result is `OFF`,
   whatever `sycl-ls` said. Configure then prints:
   ```
   -- CPU device detected by sycl-ls, but no CPU target in fsycl-targets
   -- CPU kernels will not be compiled - disabling CPU-dependent tests/benchmarks
   ```
3. `BATCHLAS_ENABLE_CPU_TESTS=OFF` (`cmake/BatchLASOptions.cmake:72`, default `ON`) forces `OFF`.

The result is exported as `BATCHLAS_HAS_CPU_TARGET` (`cmake/backend_config.h.in`, `#cmakedefine01`) and
to Python as `compiled_features()["has_cpu_target"]`.

## What `BATCHLAS_HAS_CPU_TARGET=OFF` turns off

- `minibench_cli_tests` is not built. It is the only member of `CPU_SYCL_DEPENDENT_TESTS`
  (`tests/CMakeLists.txt:135-137`).
- Every NETLIB instantiation of every typed suite is compiled out. `backend_types`,
  `backend_types_filtered` and `backend_types_complex` (`tests/test_utils.hh`) add the NETLIB block only
  when `BATCHLAS_HAS_HOST_BACKEND && BATCHLAS_HAS_CPU_TARGET`. Any NETLIB configuration still reached is
  skipped in `BatchLASTest::SetUp()`.

> **Note:** Other test binaries still build (for example `ormqr_cta_tests`, `syev_cta_tests`,
> `sytrd_cta_tests`), because their GPU kernels compile and host LAPACK needs no SYCL kernel. Their
> NETLIB cases are absent. Benchmarks have no base and CPU-dependent split.

## Traps

- **Typed-test indices shift.** Without the NETLIB block, `SteqrTest/0` is float/CUDA, not
  float/NETLIB. `tests/known-failures.txt` therefore pins each entry with `type_param=`, and applies
  those entries only to the `dev-tests` preset.
- **`dev-gpu` and `dev-gpu-tests` set `BATCHLAS_CPU_TARGET=none`.** This saves about 30% per compile and
  device link. Their coverage is reduced, and `ctest` still reports green.
- **`native_cpu` has sub-group size 1.** A CPU target does not make CTA kernels runnable. See
  [Running the BatchLAS tests](../../tests/README.md).
- **Stage 2 matches by substring.** Any target name containing `cpu` counts.
