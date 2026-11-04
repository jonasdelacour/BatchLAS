// SPDX-License-Identifier: MIT
#pragma once

#include <src/sycl/device_scalar.hh>

#include <batchlas/device.hh>
#include <batchlas/device/lapack/geqrf.hh>

#include <sycl/sycl.hpp>

namespace batchlas::geqrf_native {

// The panel kernels' scope tag; the body is device::geqr2. evidence: docs/perf/qr.md#the-48-kib-launch-hole
enum class GeqrfScope { SubGroup, WorkGroup };

}  // namespace batchlas::geqrf_native
