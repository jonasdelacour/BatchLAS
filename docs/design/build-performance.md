# Build performance {#design_build_performance}

> **Status:** current · device-link numbers measured 2026-10-24 on the 2x RTX 4090 box (DPC++ at
> `/opt/dpcpp-cuda`, `spir64_x86_64,nvidia_gpu_sm_89`, -j8)

The build is SYCL device-link-bound: the shared library is the device-link unit, so one changed
object redoes `sycl-post-link`, ptxas and CPU AOT for its whole `.so`, and `-j` cannot shorten
that. Working rules (`-j8`, ccache, GPU-only trees): @ref dev_agent_guide §7; measured dead ends
are the §11 table.

## build performance: split components and the monolithic library {#build-performance-split-components-and-the-monolithic-library}

| `BATCHLAS_MONOLITHIC_LIBRARY` | Shape | Use |
| --- | --- | --- |
| `OFF` (default except macOS) | 16 component `.so` plus `batchlas_backends_cuda`/`_rocm` when built | development; an edit relinks only its component |
| `ON` (required on macOS, `release` preset) | one `batchlas` `.so` | release: one SONAME, one export set, no cross-`.so` cycles |

Before the split a one-line edit cost 216 s (198 s one relink). A monolithic `--target batchlas
-j16` build takes 11m38s, 575 s of it the single link, so the `cuda` pre-push preset stays split.
A from-scratch `release` build is OOM-killed at `--parallel 8` on the 125 GB box; `--parallel 4`
completes in 728 s. `-fvisibility=hidden` applies only to a monolithic tree
(@ref design_symbol_visibility).

## build performance: the umbrella header excludes device code {#build-performance-the-umbrella-header-excludes-device-code}

`<batchlas/blas/linalg.hh>` (and so `<batchlas.hh>`) includes neither `<batchlas/blas/device.hh>`
(@ref design_device_group_blas) nor `<sycl/sycl.hpp>`; together they cost about 4.1 s per consumer
TU (measured before 2026-09-30, not re-measured).

- A consumer that needs `batchlas::device::*` includes `<batchlas/blas/device.hh>` and builds with `-fsycl`.
- A consumer that passes a `sycl::queue` or `sycl::event` across the boundary includes
  `<batchlas/sycl_interop.hh>`, the one public non-device header that includes `<sycl/sycl.hpp>`.

**Both edges must stay cut.** The headers form a cycle, so restoring either include brings back the
whole umbrella. Consumers must configure with the same SYCL compiler (`docs/cpp-api.md`, "Building
against BatchLAS").

## build performance: device-link mechanisms

- **One module is the link floor.** `sycl-post-link -split=auto` cuts a library into modules;
  llvm-foreach runs them in batches of `BATCHLAS_SYCL_LINK_JOBS` (default `auto`, the logical cores
  clamped to 4..32), waiting for each batch's slowest. A module is never split, so the largest module
  floors every relink of its library.
- **Global-offset twins are off.** DPC++ gives every kernel a `<name>_with_offset` twin that nothing
  launches (2,619 of 5,305 entry points). `BATCHLAS_SYCL_GLOBAL_OFFSET=OFF` (default, probed at
  configure) passes `-mllvm -enable-global-offset=false`: posv_tiny's module went 319 s -> 164 s. The
  cost: a kernel using a SYCL 1.2.1 global offset (`parallel_for(range, id, k)`, `get_offset()`) would
  not see it; nothing in the tree does.
- **No CPU AOT for GPU-only code.** `opencl-aot` was most of each full-target link (942 of 1,186
  CPU-seconds in `libbatchlas_sycl`). The native level-3 families require `d.is_gpu` in `can_run`, so
  `batchlas_extensions_level3` (the seven level-3 `<op>.cc` and the trsm kernels),
  `batchlas_extensions_posv` and `batchlas_extensions_gesv` are `NO_CPU_TARGETS` libraries, each a
  device-code cluster with no `SYCL_EXTERNAL` crossing it. gemm's CPU image cannot be dropped: a
  vendor-free CPU queue runs it.
- **gemm shards.** gemm's reg and wide config rows are generated shard TUs (`BATCHLAS_GEMM_SHARDS`,
  `gemm_shard.cc.in` in `src/ops/CMakeLists.txt`); a `static_assert` in `gemm_kernels.cc` fails on a
  row with no shard. A `gemm_kernels.cc` edit went 1024 s -> 213 s, a sycl relink 763 s -> 194 s.
- **No executable relinks.** `BATCHLAS_LINK_DEPENDS_NO_SHARED=ON` sets `CMAKE_LINK_DEPENDS_NO_SHARED`
  (about 150 executables, 2.5 min, relinked on every library change before) and links with
  `-Wl,-z,now`, so a stale executable missing a removed symbol fails at load. `scripts/gate.sh` turns
  it off.
- **Private TMPDIR.** A killed or failed DPC++ link leaves `<tu>-sm_89-<hash>.*` in `$TMPDIR`; on the
  box's tmpfs `/tmp` that reached 38-43 GiB of RAM. `BATCHLAS_PRIVATE_TMPDIR=ON` runs every compile
  and link through `<build>/batchlas-tmpdir`, a fresh directory under `<build>/.tmp` removed on exit.
  TMPDIR is not in ccache's hash.
- **`tuning_params.hh` fan-out.** Public-header default arguments are `0`, resolved to the tuned
  constant inside the `.cc`, so a header edit recompiles 11 objects, not 128. A/B a constant with its
  `BATCHLAS_TUNE_*` override.

## build performance: what is still slow

- The sycl relink (194 s): many modules under 60 s, so more link jobs help only on a bigger machine.
- The posv and gesv links (about 169 s each): one module each, the floor.
- Header edits: about 280 of 318 objects include `matrix.hh`, `settings`, `error`, `enums` or
  `workspace`.
- A new tree without a warm cache (no cold build of the current tree measured). Seed it instead
  (`scripts/seed_build_tree.sh`, 9 s) or warm the cache with `scripts/prewarm_cache.sh`.
- `gemm_candidates_tests_heavy` (351 s) is the longest ctest entry; the first `scripts/gate.sh` run
  takes 67 min.
