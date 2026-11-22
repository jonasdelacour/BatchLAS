// SPDX-License-Identifier: MIT
#pragma once
/// @file
/// @brief The device umbrella: in-kernel BLAS and the launch frame of batchlas::device in one
/// include.
/// @ingroup api_device
///
/// It includes the core vocabulary, arithmetic, scalar kernels, executors, operands, views,
/// interop, reductions, BLAS levels 1 to 3 (the register tier of level 3), the LAPACK
/// auxiliaries, tiers, the launch frame and the workspace protocol. The LAPACK-level routines are
/// in `<batchlas/device/lapack.hh>`, the in-group level-3 forms in
/// `<batchlas/device/level3/group.hh>` and the sparse routines in `<batchlas/device/sparse.hh>`.
/// The reference is @ref api_device.

#include <batchlas/device/core.hh>
#include <batchlas/device/arith.hh>
#include <batchlas/device/scalar.hh>
#include <batchlas/device/exec.hh>
#include <batchlas/device/operand.hh>
#include <batchlas/device/level1.hh>
#include <batchlas/device/level2.hh>
#include <batchlas/device/lapack_aux.hh>
#include <batchlas/device/interop.hh>
#include <batchlas/device/grid.hh>
#include <batchlas/device/reduce.hh>
#include <batchlas/device/view.hh>
#include <batchlas/device/level3.hh>
#include <batchlas/device/workspace.hh>
#include <batchlas/device/tier.hh>
#include <batchlas/device/launch.hh>
#include <batchlas/device/sg_partition.hh>
