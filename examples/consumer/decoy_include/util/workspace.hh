#pragma once
//
// DECOY. This is not a BatchLAS header and it is not meant to compile.
//
// It stands in for something entirely ordinary: a consumer project that already
// has a header called util/workspace.hh of its own. BatchLAS installs its
// headers into generic top-level directories (blas/, util/, internal/) and its
// own 578 cross-includes spell them unprefixed (`#include <util/workspace.hh>`),
// so the two namespaces collide in the consumer's include root. CMake makes the
// collision the default outcome: a target's own include directories are ordered
// before anything propagated from a linked target, and imported targets
// propagate theirs as -isystem, which is searched last. The consumer's file
// wins, inside BatchLAS's own headers, and the error message names files the
// consumer has never heard of.
//
// consumer_test.sh builds the example with this directory on the include path
// and classifies the outcome by the sentinel below:
//
//   builds clean        -> BatchLAS's public headers no longer reach for an
//                          unprefixed <util/...> spelling. The collision class
//                          is closed; make this a hard assertion.
//   sentinel in the log -> the known state: the install no longer squats the
//                          include ROOT (that part IS asserted, separately and
//                          hard, against the install tree), but the internal
//                          spellings are still unprefixed, so a consumer header
//                          of the same name still shadows. Closing this needs
//                          include/{blas,util,internal} moved under
//                          include/batchlas/ and the include sites rewritten.
//   any other failure   -> a real regression; the test fails.
//
#error "BATCHLAS_DECOY_WORKSPACE_SHADOWED: a consumer-owned util/workspace.hh was picked up ahead of BatchLAS's own"
