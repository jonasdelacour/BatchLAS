# Build performance {#design_build_performance}

> **Status:** current · checked 2026-10-21

The build is SYCL device-link-bound. The library shape sets the relink unit; the public headers set
the per-TU front-end cost of consumer and test translation units. Working rules (`-j8`, ccache,
GPU-only trees, measured dead ends): `docs/developer/agent-guide.md` §7.

## build performance: split components and the monolithic library {#build-performance-split-components-and-the-monolithic-library}

The unit of SYCL device linking is the shared library: one changed object redoes `sycl-post-link`,
ptxas and CPU AOT for every object in its `.so`, single-threaded, so `-j` cannot shorten it.

| `BATCHLAS_MONOLITHIC_LIBRARY` | Shape | Use |
| --- | --- | --- |
| `OFF` (default except macOS) | 12 component `.so` plus `batchlas_backends_cuda`/`_rocm` when built (`src/CMakeLists.txt:265-376`) | development; one edit relinks only its component |
| `ON` (default on macOS, `release` preset) | one `batchlas` `.so` from every object library | release and packaging: one SONAME, one export set, no cross-`.so` symbol cycles |

Measured on the 2x RTX 4090 box (DPC++ at `/opt/dpcpp-cuda`, RelWithDebInfo,
`nvidia_gpu_sm_89,native_cpu`; `cmake/BatchLASOptions.cmake:199-222`, `CMakePresets.json` `release`):

- Before the split, a one-line edit cost 216 s, of which 198 s was a single relink.
- Monolithic: `--target batchlas -j16` from scratch took 11m38s, of which the one
  `Linking CXX shared library libbatchlas.so` step was 575 s; the product is a 273 MiB `.so`.
- A from-scratch `release` build was OOM-killed at `--parallel 20` and at `--parallel 8` (125 GB
  box, 115 GB free) during compilation; `--parallel 4` completes in 728 s wall.
- The `cuda` pre-push preset stays split, so the 575 s link never sits in the iteration loop.
  `-fvisibility=hidden` applies only to a monolithic tree (@ref design_symbol_visibility).

## build performance: the umbrella header excludes device code {#build-performance-the-umbrella-header-excludes-device-code}

`<batchlas/blas/linalg.hh>`, which `<batchlas.hh>` includes, does not include:

- `<batchlas/blas/device.hh>`, the device-side group BLAS templates (@ref design_device_group_blas), or
- `<sycl/sycl.hpp>`. Only `<batchlas/sycl_interop.hh>` and the device-side headers (`group-invoke.hh`,
  `group-collectives.hh`, `sycl-local-accessor-helpers.hh`, `blas/device/`) include it.

Together they cost about 4.1 s per consumer TU, and 71 of the 114 test and benchmark TUs use
neither (measured before 2026-09-30; stale, not re-measured).

- A consumer that needs `batchlas::device::*` includes `<batchlas/blas/device.hh>` and builds with `-fsycl`.
- A consumer that passes a `sycl::queue` or `sycl::event` across the boundary includes
  `<batchlas/sycl_interop.hh>`, the one public non-device header that includes `<sycl/sycl.hpp>`.

**Both edges must stay cut.** The headers form a cycle. Restoring either include brings back the whole
umbrella, so cutting one edge gains nothing.

Code sites: `include/batchlas/blas/linalg.hh:14` and `include/batchlas/sycl_interop.hh:18`. Consumers
must configure with the same SYCL compiler; see `docs/cpp-api.md`, "Building against BatchLAS".
