#ifndef BATCHLAS_HH
#define BATCHLAS_HH

/// @file
/// @brief The umbrella header: `#include <batchlas.hh>` gives the whole public API.
///
/// Everything is in namespace `batchlas`. Every entry point enqueues on a
/// batchlas::Queue and returns a batchlas::Event; nothing is readable until the
/// queue or the event is waited on. Two opt-in headers are not included:
/// `<batchlas/blas/device.hh>` (in-kernel group BLAS) and `<batchlas/sycl_interop.hh>`
/// (SYCL types); both need `-fsycl`.
/// @ingroup api_reference

#include <batchlas/blas/linalg.hh>

#endif