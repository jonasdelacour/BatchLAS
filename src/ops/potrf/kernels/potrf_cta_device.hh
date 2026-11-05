// SPDX-License-Identifier: MIT
#pragma once

// The CTA kernel's body is device::potrf's tier::Cta body (lapack/potrf.hh, potrf_detail::cta_body).

#include <batchlas/device/core.hh>
#include <batchlas/device/lapack/potrf.hh>

namespace batchlas::potrf_native {

enum class PotrfScope { SubGroup, WorkGroup };  // part of the kernel name PotrfCtaKernel: stays here

template <PotrfScope SC>
inline constexpr ::batchlas::device::Scope potrf_body_scope =
    SC == PotrfScope::SubGroup ? ::batchlas::device::Scope::Sub : ::batchlas::device::Scope::Group;

}  // namespace batchlas::potrf_native
