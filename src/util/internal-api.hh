// SPDX-License-Identifier: MIT
#pragma once

// Exported but NOT ABI: private symbols only in-tree tests link. evidence: docs/design/runtime-internals.md#runtime-internals-symbol-visibility-for-private-headers

#include <batchlas/export.hh>

#include <string_view>

#ifndef BATCHLAS_INTERNAL_API
#define BATCHLAS_INTERNAL_API BATCHLAS_API
#endif

namespace batchlas::detail {

// The one channel for user-facing warnings ("batchlas: <text>" on stderr; not env-gated debug traces).
BATCHLAS_INTERNAL_API void warn(std::string_view text);
BATCHLAS_INTERNAL_API bool warn_once(std::string_view tag, std::string_view text);  // true if it printed
BATCHLAS_INTERNAL_API void reset_warnings();  // forget every tag (tests)

}  // namespace batchlas::detail
