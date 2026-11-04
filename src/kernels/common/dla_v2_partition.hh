// SPDX-License-Identifier: MIT
#pragma once
// v2::Partition over SubGroupPartition (all backends, Masked chunks); include/ cannot name src/.

#include <batchlas/device/exec.hh>
#include <batchlas/device/detail/sg_partition/sg_partition.hh>

namespace batchlas::device::inline v2 {

template <int P, bool Masked = sgp::kMaskedByDefault>
using SgpPartition = Partition<P, Masked, SubGroupPartition<static_cast<size_t>(P), Masked>>;

// The v2 executor over a SubGroupPartition a kernel already holds.
template <size_t P, bool Masked>
BATCHLAS_DLA_INLINE SgpPartition<static_cast<int>(P), Masked> wrap(SubGroupPartition<P, Masked> part) {
    return {typename SgpPartition<static_cast<int>(P), Masked>::Wrap{}, part};
}

}  // namespace batchlas::device
