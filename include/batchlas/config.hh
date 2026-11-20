// SPDX-License-Identifier: MIT
#pragma once
/// @file
/// @brief The build configuration every public header reads: the generated backend_config.h and
///        export.hh of a build tree or an install, else header-only defaults.
///
/// Header-only use (`-I <checkout>/include`, nothing configured or linked) gets defaults from the
/// SYCL implementation's own headers. Each macro can be overridden with `-D`; Metal needs
/// `-DBATCHLAS_DEVICE_NO_VECTOR_TYPES=1`. @see @ref guide_cpp_api_device_header_only
/// @ingroup api_config

#if __has_include(<batchlas/backend_config.h>)

#include <batchlas/backend_config.h>
#include <batchlas/export.hh>

#else

// The implementation is only recognisable from its headers (neither driver predefines a name).
#include <sycl/sycl.hpp>

#define BATCHLAS_HEADER_ONLY 1

#ifndef BATCHLAS_SYCL_IMPL_ACPP
#if defined(ACPP_VERSION_MAJOR) || defined(HIPSYCL_VERSION_MAJOR)
#define BATCHLAS_SYCL_IMPL_ACPP 1
#else
#define BATCHLAS_SYCL_IMPL_ACPP 0
#endif
#endif
#ifndef BATCHLAS_SYCL_IMPL_DPCPP
#define BATCHLAS_SYCL_IMPL_DPCPP (!BATCHLAS_SYCL_IMPL_ACPP)
#endif

// No library to dispatch to: only the native SYCL kernels in these headers exist.
#ifndef BATCHLAS_HAS_SYCL_BACKEND
#define BATCHLAS_HAS_SYCL_BACKEND 1
#endif
#ifndef BATCHLAS_HAS_HOST_BACKEND
#define BATCHLAS_HAS_HOST_BACKEND 0
#endif
#ifndef BATCHLAS_HAS_CUDA_BACKEND
#define BATCHLAS_HAS_CUDA_BACKEND 0
#endif
#ifndef BATCHLAS_HAS_ROCM_BACKEND
#define BATCHLAS_HAS_ROCM_BACKEND 0
#endif
#ifndef BATCHLAS_HAS_DEVICE_FP64
#define BATCHLAS_HAS_DEVICE_FP64 1
#endif
// 1 only for AdaptiveCpp's Metal emitter, which has no vector types; a run-time target, so a flag.
#ifndef BATCHLAS_DEVICE_NO_VECTOR_TYPES
#define BATCHLAS_DEVICE_NO_VECTOR_TYPES 0
#endif
// Library builds probe whether this AdaptiveCpp takes an SLM opt-in; off is always correct.
#ifndef BATCHLAS_ACPP_SLM_OPTIN
#define BATCHLAS_ACPP_SLM_OPTIN 0
#endif

#ifndef BATCHLAS_API
#if defined(_WIN32)
#define BATCHLAS_API
#else
#define BATCHLAS_API __attribute__((visibility("default")))
#endif
#endif
#ifndef BATCHLAS_NO_EXPORT
#define BATCHLAS_NO_EXPORT
#endif

#endif
