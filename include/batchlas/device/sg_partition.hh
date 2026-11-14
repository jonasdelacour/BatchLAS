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

// The executor of this item's P-lane chunk of sg: SgpPartition<P, Masked>{it} from a sub-group.
template <int P, bool Masked = sgp::kMaskedByDefault>
BATCHLAS_DEVICE_INLINE SgpPartition<P, Masked> make_partition(sycl::sub_group sg) {
    return wrap(batchlas::make_partition<static_cast<size_t>(P), Masked>(sg));
}

}  // namespace batchlas::device
