// SPDX-License-Identifier: MIT
#pragma once

#include <batchlas/device/lapack/getrf.hh>

namespace batchlas::getrf_native {

inline constexpr int kLuRedSlots = ::batchlas::device::detail::kRedSlots;  // argmax slots: ONE constant

}  // namespace batchlas::getrf_native
