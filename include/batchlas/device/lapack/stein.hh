// SPDX-License-Identifier: MIT
#pragma once
// xSTEIN's own pieces (inverse iteration on a symmetric tridiagonal); factor / solve: lagtf / lagts.

#include <cstdint>

#include <batchlas/device/core.hh>
#include <batchlas/device/lapack/lagtf.hh>
#include <batchlas/device/lapack/lagts.hh>

namespace batchlas::device::inline v2 {

namespace stein_detail {
/// The start vector's generator (dstein uses dlarnv): a 32-bit LCG, so a fixed per-(item,
/// vector) seed reproduces a run bit for bit. Advances state and returns it.
BATCHLAS_DLA_HELPER std::uint32_t lcg(std::uint32_t& state) {
    state = state * 1664525u + 1013904223u;
    return state;
}
}  // namespace stein_detail

}  // namespace batchlas::device::inline v2
