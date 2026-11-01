// SPDX-License-Identifier: MIT
#pragma once
// v2::Partition over SubGroupPartition (all backends, Masked chunks); include/ cannot name src/.

#include <batchlas/blas/device/v2/exec.hh>
#include <src/kernels/common/sg_partition/sg_partition.hh>

namespace batchlas::device::v2 {

template <int P, bool Masked = sgp::kMaskedByDefault>
using SgpPartition = Partition<P, Masked, SubGroupPartition<static_cast<size_t>(P), Masked>>;

}  // namespace batchlas::device::v2
