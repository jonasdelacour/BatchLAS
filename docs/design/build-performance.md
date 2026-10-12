# Build performance {#design_build_performance}

> **Status:** current. Build-time facts (device-link-bound builds, `dev-gpu` presets, measured dead
> ends) are in section 7 of `docs/developer/agent-guide.md`.

The public headers keep consumer and test translation units cheap to compile. Headers cannot shorten
the device link, so they control only the per-TU front-end cost.

## build performance: the umbrella header excludes device code

`<batchlas/blas/linalg.hh>`, which `<batchlas.hh>` includes, does not include:

- `<batchlas/blas/device.hh>`, the device-side group BLAS templates (@ref design_device_group_blas), or
- `<sycl/sycl.hpp>`, which `<batchlas/blas/functions.hh>` would otherwise bring in.

Together they cost about 4.1 s per consumer TU. 71 of the 114 test and benchmark TUs use neither.
These figures predate 2026-09-30 and have not been re-measured.

- A consumer that needs `batchlas::device::*` includes `<batchlas/blas/device.hh>` and builds with `-fsycl`.
- A consumer that passes a `sycl::queue` or `sycl::event` across the boundary includes
  `<batchlas/sycl_interop.hh>`, the one public header that includes `<sycl/sycl.hpp>` directly.

**Both edges must stay cut.** The headers form a cycle. Restoring either include brings back the whole
umbrella, so cutting one edge gains nothing.

Code sites: `include/batchlas/blas/linalg.hh` (above `functions.hh`) and `include/batchlas/sycl_interop.hh`
(above its `<sycl/sycl.hpp>` include). Consumers must configure with the same SYCL compiler; see
`docs/cpp-api.md`, "Building against BatchLAS".
