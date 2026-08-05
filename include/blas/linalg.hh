#pragma once
#include <complex>
// Main include file for the BatchLAS library that includes all components

// Include the enum declarations
#include <blas/enums.hh>
#include <blas/matrix.hh>
// <blas/device.hh> (the device-side group-BLAS kernel templates) and, through
// <blas/functions.hh>, <sycl/sycl.hpp> are deliberately NOT pulled in here.
// Together they cost ~4.1 s per consumer TU and 71 of the 114 test/benchmark
// TUs use neither. If you need batchlas::device::*, include <blas/device.hh>
// yourself. Note that both edges have to stay cut: the headers form a cycle,
// so restoring either one re-pulls the whole umbrella and the saving vanishes.
#include <blas/functions.hh>
#include <blas/extra.hh>
#include <blas/extensions.hh>
#include <blas/csr_generators.hh>

// The batchlas::linalg convenience layer.
#include <blas/linalg-ops.hh>
