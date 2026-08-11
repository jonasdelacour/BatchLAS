#pragma once
//
// DECOY. This is not a BatchLAS header and it is not meant to compile.
// See decoy_include/util/workspace.hh for the full explanation.
//
// This one covers the `blas/` name. blas/enums.hh is on the consumer's real
// include path: <batchlas/blas/linalg.hh> includes <batchlas/blas/enums.hh>
// directly, and so does <batchlas/blas/functions/ormqr.hh>. If either ever
// regressed to the unprefixed spelling, this file would be found first.
//
// blas/linalg.hh would have been the tempting name to decoy instead -- and the
// wrong one: after the move nothing spells <blas/linalg.hh>, not even the
// consumer, so that decoy could never fire and the probe would pass vacuously.
//
#error "BATCHLAS_DECOY_ENUMS_SHADOWED: a consumer-owned blas/enums.hh was picked up ahead of BatchLAS's own"
