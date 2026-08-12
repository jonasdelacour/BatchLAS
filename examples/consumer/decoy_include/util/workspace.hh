#pragma once
//
// DECOY. This is not a BatchLAS header and it is not meant to compile.
//
// It stands in for something entirely ordinary: a consumer project that already
// has a header called util/workspace.hh of its own. This directory is placed on
// the include path where a consumer's own headers go -- i.e. AHEAD of anything
// an imported target propagates, because CMake orders a target's own include
// directories first and imported targets propagate theirs as -isystem, which is
// searched last. So if BatchLAS ever spells one of its own includes
// `<util/workspace.hh>`, THIS file is what its headers get.
//
// That used to be the state of the world: the public headers lived in
// include/{blas,util,internal} and cross-included each other unprefixed. They
// now live in include/batchlas/ and are spelled <batchlas/util/workspace.hh>,
// so this file must be unreachable.
//
// consumer_test.sh builds the example with this directory on the include path.
// There are exactly two acceptable outcomes:
//
//   builds clean        -> correct. BatchLAS is reachable only as <batchlas/...>
//                          and shadows nothing in the consumer's include root.
//   sentinel in the log -> REGRESSION. Some installed BatchLAS header spells an
//                          unprefixed <util/...> include again. The test fails.
//
// Any other build failure is also a test failure -- it means the probe stopped
// probing. See also decoy_include/blas/enums.hh and
// decoy_include/internal/ormqr_blocked.hh, which cover the other two names.
//
#error "BATCHLAS_DECOY_WORKSPACE_SHADOWED: a consumer-owned util/workspace.hh was picked up ahead of BatchLAS's own"
