// SPDX-License-Identifier: MIT
#pragma once
// The stedc kernels' short names for helpers that live with the public merge routines:
// uniform (lapack/laed4.hh) and fold (lapack/laed2.hh). Tests: tests/device_lapack_stedc_suite.cc.

#include <batchlas/device.hh>
#include <batchlas/device/lapack/laed2.hh>
#include <batchlas/device/lapack/laed4.hh>

namespace batchlas::stedc_v2 {


using batchlas::device::detail::fold;
using batchlas::device::detail::is_group_v;
using batchlas::device::detail::uniform;

}  // namespace batchlas::stedc_v2
