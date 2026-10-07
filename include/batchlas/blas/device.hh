#pragma once

/// @file
/// @brief Umbrella header for the in-kernel group BLAS (`namespace batchlas::device`).
///
/// Opt-in: not included by `<batchlas.hh>`, and it needs `-fsycl`. The calling
/// contract (executors, workspace protocol, traps) is on
/// @ref design_device_group_blas.
/// @ingroup device

#include <batchlas/blas/device/group_blas.hh>