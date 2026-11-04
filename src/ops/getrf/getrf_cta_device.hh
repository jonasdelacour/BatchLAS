// SPDX-License-Identifier: MIT
#pragma once

namespace batchlas::getrf_native {

inline constexpr int kLuRedSlots = 32;  // argmax slots: a CONSTANT, so query, fit and launcher agree

// SubGroup scope makes G-packing legal; WorkGroup on a packed launch is a RACE: derive both together.
enum class LuScope { SubGroup, WorkGroup };

}  // namespace batchlas::getrf_native
