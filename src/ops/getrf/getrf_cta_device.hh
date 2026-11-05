// SPDX-License-Identifier: MIT
#pragma once

#include <batchlas/device/lapack/getrf.hh>

namespace batchlas::getrf_native {

inline constexpr int kLuRedSlots = ::batchlas::device::getrf_detail::kRedSlots;  // argmax slots: ONE constant

// SubGroup scope makes G-packing legal; WorkGroup on a packed launch is a RACE: derive both together.
enum class LuScope { SubGroup, WorkGroup };

}  // namespace batchlas::getrf_native
