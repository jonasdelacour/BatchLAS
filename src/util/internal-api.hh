#pragma once

// Exported but NOT ABI: private symbols only in-tree tests link. evidence: docs/design/runtime-internals.md#runtime-internals-symbol-visibility-for-private-headers

#include <batchlas/export.hh>

#ifndef BATCHLAS_INTERNAL_API
#define BATCHLAS_INTERNAL_API BATCHLAS_API
#endif
