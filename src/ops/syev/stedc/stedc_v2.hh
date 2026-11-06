// SPDX-License-Identifier: MIT
#pragma once
// The stedc kernels' short names for helpers that live with the public merge routines:
// uniform (lapack/laed4.hh) and fold (lapack/laed2.hh). Tests: tests/device_blas_v2_stedc_tests.cc.

#include <batchlas/device.hh>
#include <batchlas/device/lapack/laed2.hh>
#include <batchlas/device/lapack/laed4.hh>

namespace batchlas::stedc_v2 {

namespace v2 = batchlas::device;

using batchlas::device::laed2_detail::fold;
using batchlas::device::laed4_detail::is_group_v;
using batchlas::device::laed4_detail::uniform;

}  // namespace batchlas::stedc_v2
