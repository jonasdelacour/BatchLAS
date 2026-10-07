#pragma once
#include <complex>

/// @file
/// @brief Everything `<batchlas.hh>` provides: containers, entry points, extensions and the linalg layer.
///
/// Does not include `<batchlas/blas/device.hh>` (in-kernel group BLAS) or
/// `<sycl/sycl.hpp>`; include those yourself where needed.
/// @ingroup api

#include <batchlas/blas/enums.hh>
#include <batchlas/blas/matrix.hh>
// Never device.hh or <sycl/sycl.hpp>; keep both header-cycle edges cut. evidence: docs/design/build-performance.md#build-performance-the-umbrella-header-excludes-device-code
#include <batchlas/blas/functions.hh>
#include <batchlas/blas/extra.hh>
#include <batchlas/blas/extensions.hh>
#include <batchlas/blas/csr_generators.hh>
#include <batchlas/blas/linalg-ops.hh>
