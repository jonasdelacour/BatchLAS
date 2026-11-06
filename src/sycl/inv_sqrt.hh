// SPDX-License-Identifier: MIT
#pragma once

#include <batchlas/device/scalar.hh>

namespace batchlas::sycl_device {  // core's inv_sqrt (device/scalar.hh) under the src spelling, until its callers move

using InvSqrtEdge = ::batchlas::device::InvSqrtEdge;
using ::batchlas::device::inv_sqrt;

}  // namespace batchlas::sycl_device
