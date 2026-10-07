# Build: CPU SYCL target detection {#design_cpu_target_detection}

> **Covers:** how configure decides whether CPU device code is compiled
> (`BATCHLAS_HAS_CPU_TARGET`), and what that switches on and off in the tests.
> **Status:** current. Logic in `cmake/BatchLASDetectSYCL.cmake`, options in
> `cmake/BatchLASOptions.cmake`; line numbers verified 2026-09-30.
> Migrated from the root `IMPLEMENTATION_NOTES.md`, whose `CMakeLists.txt:814-845`
> citation predates the split of the top-level CMake into `cmake/`.

## The problem: a GPU-only compiler cannot build CPU kernels

A DPC++ built from source for NVIDIA GPUs may have no CPU backend at all. With
no CPU entry in `-fsycl-targets`, a translation unit that instantiates a kernel
for a CPU device (the `NETLIB` backend's SYCL helpers, or anything submitted to
`Queue("cpu")` that is not a host LAPACK call) does not compile, or compiles and
then finds no kernel image at run time. Configure therefore has to know whether
CPU device code will actually be built, and drop what cannot work.

## Two stages: probe the runtime, then trust only the target list

**Stage 1, `detect_sycl_cpu_target()`** (`cmake/BatchLASDetectSYCL.cmake:495`)
chooses a CPU target and may append it to `BATCHLAS_SYCL_TARGETS`:

| `BATCHLAS_CPU_TARGET` | effect |
| --- | --- |
| `none` | no CPU target; `BATCHLAS_HAS_CPU_TARGET=OFF` |
| `native_cpu` / `spir64_x86_64` | that target is appended; `ON` |
| `auto` (default, `cmake/BatchLASOptions.cmake:152`) | runs `sycl-ls`: `[opencl:cpu]` or `[host:cpu]` appends `spir64_x86_64`; `[native_cpu:cpu]` appends `native_cpu`; neither, or no `sycl-ls`, or `sycl-ls` failing, gives `OFF` |
| anything else | a configure `WARNING`, then treated as `auto` |

**Stage 2, the final gate** (`cmake/BatchLASDetectSYCL.cmake:655-678`), runs
after the target list is printed (`-- Using SYCL targets: ...`). It decides on
the *compilation flags*, not on device detection:

1. An explicit `native_cpu` / `spir64_x86_64` override forces `ON`.
2. If no entry of `BATCHLAS_SYCL_TARGETS` matches `cpu`, `spir64` or
   `native_cpu`, the result is `OFF`, whatever `sycl-ls` said. When stage 1 had
   said `ON`, configure explains itself:
   ```
   -- CPU device detected by sycl-ls, but no CPU target in fsycl-targets
   -- CPU kernels will not be compiled - disabling CPU-dependent tests/benchmarks
   ```
3. `BATCHLAS_ENABLE_CPU_TESTS=OFF` (`cmake/BatchLASOptions.cmake:104`, default
   `ON`) forces `OFF` regardless:
   `-- CPU-dependent tests and benchmarks manually disabled`.

The result is exported to C++ as `BATCHLAS_HAS_CPU_TARGET` through
`cmake/backend_config.h.in` (`#cmakedefine01`), and to Python as
`compiled_features()["has_cpu_target"]`.

## What `BATCHLAS_HAS_CPU_TARGET=OFF` turns off

- **One test binary is not built:** `minibench_cli_tests`, the only member of
  `CPU_SYCL_DEPENDENT_TESTS` (`tests/CMakeLists.txt:115-125`). Configure prints
  `-- Skipping CPU SYCL-dependent tests (no CPU target in fsycl-targets): minibench_cli_tests`.
- **Every NETLIB instantiation of every typed suite is compiled out.**
  `test_utils::backend_types`, `backend_types_filtered` and
  `backend_types_complex` (`tests/test_utils.hh`) include the NETLIB block only
  under `BATCHLAS_HAS_HOST_BACKEND && BATCHLAS_HAS_CPU_TARGET`, and
  `BatchLASTest::SetUp()` `GTEST_SKIP`s a NETLIB config that is still reached.

> **Superseded claim.** `IMPLEMENTATION_NOTES.md` said that tests which only
> create `Queue("cpu")` for NETLIB reference comparisons "are included even
> without CPU targets". The *binaries* are still built (`ormqr_cta_tests`,
> `syev_cta_tests`, `sytrd_cta_tests`, ...), because their GPU kernels compile
> and host LAPACK needs no SYCL kernel; but their NETLIB *cases* no longer
> exist in such a build, because the typed-suite lists above now gate on
> `BATCHLAS_HAS_CPU_TARGET`. It also said benchmarks are split into a base set
> and a CPU-dependent set; `benchmarks/CMakeLists.txt` has no such split today.

## Traps

- **Typed-test indices shift.** With the NETLIB block compiled out,
  `SteqrTest/0` stops meaning float/NETLIB and becomes float/CUDA. That is why
  `tests/known-failures.txt` pins each entry with a `type_param=` field and why
  its entries are only claimed for the `dev-tests` preset.
- **`dev-gpu` / `dev-gpu-tests` set `BATCHLAS_CPU_TARGET=none` on purpose**
  (about 30% off every compile and device link), so they carry exactly this
  reduced coverage while `ctest` still reports green.
- **`native_cpu` has sub-group size 1.** A CPU target being present does not
  make CTA kernels runnable on it; see [Running the BatchLAS tests](../../tests/README.md).
- **Stage 2 matches by substring.** Any target whose name contains `cpu` counts
  as a CPU target.
