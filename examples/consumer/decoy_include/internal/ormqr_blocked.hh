#pragma once
//
// DECOY. This is not a BatchLAS header and it is not meant to compile.
// See decoy_include/util/workspace.hh for the full explanation.
//
// This one covers the `internal/` name -- the worst of the three to squat on,
// since "internal" is exactly what a consumer calls its own private headers.
// internal/ormqr_blocked.hh is the internal/ header that the consumer's TU
// actually reaches: <batchlas/blas/linalg.hh> -> blas/functions.hh ->
// blas/functions/ormqr.hh -> <batchlas/internal/ormqr_blocked.hh>.
//
#error "BATCHLAS_DECOY_ORMQR_BLOCKED_SHADOWED: a consumer-owned internal/ormqr_blocked.hh was picked up ahead of BatchLAS's own"
