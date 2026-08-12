#pragma once
#include <complex>
// Main include file for the BatchLAS library that includes all components

// Include the enum declarations
#include <batchlas/blas/enums.hh>
#include <batchlas/blas/matrix.hh>
// <batchlas/blas/device.hh> (the device-side group-BLAS kernel templates) and, through
// <batchlas/blas/functions.hh>, <sycl/sycl.hpp> are deliberately NOT pulled in here.
// Together they cost ~4.1 s per consumer TU and 71 of the 114 test/benchmark
// TUs use neither. If you need batchlas::device::*, include <batchlas/blas/device.hh>
// yourself. Note that both edges have to stay cut: the headers form a cycle,
// so restoring either one re-pulls the whole umbrella and the saving vanishes.
#include <batchlas/blas/functions.hh>
#include <batchlas/blas/extra.hh>
#include <batchlas/blas/extensions.hh>
#include <batchlas/blas/csr_generators.hh>

// The batchlas::linalg convenience layer.
#include <batchlas/blas/linalg-ops.hh>
