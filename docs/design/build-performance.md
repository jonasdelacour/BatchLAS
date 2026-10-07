# Build performance: header structure decisions {#design_build_performance}

> **Covers:** design decisions in the public headers that exist to keep consumer and test
> translation units cheap to compile.
> **Status:** current. The general build-time facts (device-link-bound builds, the `dev-gpu`
> presets, the measured dead ends such as Ninja, lld, PCH and unity builds) are in section 7 of
> `docs/developer/agent-guide.md` and are not repeated here.

The BatchLAS build is SYCL device-link-bound, not compile-bound (`docs/developer/agent-guide.md` section 7): the
shared library is the unit of device linking, so most build-time work happens in the link of
each component library, and `-j` cannot shorten it. What the headers can still control is the
front-end cost each consumer translation unit pays, and that is what this page records.

## build performance: the umbrella header excludes device code

`<batchlas/blas/linalg.hh>`, which is what `<batchlas.hh>` includes, deliberately pulls in
neither of two headers:

- `<batchlas/blas/device.hh>`, the device-side group BLAS kernel templates
  (@ref design_device_group_blas), and
- `<sycl/sycl.hpp>`, which would otherwise arrive through `<batchlas/blas/functions.hh>`.

Together they cost about 4.1 s per consumer translation unit, and 71 of the 114 test and
benchmark translation units used neither. (Both figures are as recorded in the header comment
before 2026-09-30; they have not been re-measured since.) A consumer that needs
`batchlas::device::*` includes `<batchlas/blas/device.hh>` itself and compiles with `-fsycl`; a
consumer that needs to move a `sycl::queue` or `sycl::event` across the boundary includes
`<batchlas/sycl_interop.hh>`, which is the one public header that includes `<sycl/sycl.hpp>`
directly (the device-side group BLAS headers under `include/batchlas/blas/device/` and the
in-kernel utilities `util/group-invoke.hh` and `util/sycl-local-accessor-helpers.hh` include it
too, and are reachable only through `device.hh`).

**Both edges have to stay cut.** These headers form a cycle, so restoring either include
re-pulls the whole umbrella and the saving vanishes. This is the same cycle that `docs/developer/agent-guide.md`
section 7 lists among the measured dead ends ("cutting one edge of a header cycle"): cutting
*one* edge buys nothing, which is why the umbrella cuts both.

Code sites that depend on this: `include/batchlas/blas/linalg.hh` (above the
`functions.hh` include) and `include/batchlas/sycl_interop.hh` (above its `<sycl/sycl.hpp>`
include). The consumer-side rule (configure the whole consuming project with the same SYCL
compiler) is in the top-level `README.md`, "Consuming BatchLAS from CMake".
