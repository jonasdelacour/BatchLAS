// SPDX-License-Identifier: MIT
#pragma once
// device::Partition over SubGroupPartition (detail/sg_partition/: all backends, Masked chunks).

#include <batchlas/device/exec.hh>
#include <batchlas/device/detail/sg_partition/sg_partition.hh>

namespace batchlas::device {

template <int P, bool Masked = sgp::kMaskedByDefault>
using SgpPartition = Partition<P, Masked, SubGroupPartition<static_cast<size_t>(P), Masked>>;

// The device executor over a SubGroupPartition a kernel already holds.
template <size_t P, bool Masked>
BATCHLAS_DEVICE_INLINE SgpPartition<static_cast<int>(P), Masked> wrap(SubGroupPartition<P, Masked> part) {
    return {typename SgpPartition<static_cast<int>(P), Masked>::Wrap{}, part};
}

}  // namespace batchlas::device
