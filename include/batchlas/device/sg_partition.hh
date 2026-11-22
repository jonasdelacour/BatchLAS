// SPDX-License-Identifier: MIT
#pragma once
/** @file
 * @brief The Partition executor over a SubGroupPartition: masked P-lane chunks of a sub-group on
 * every backend.
 * @ingroup api_device_exec
 */

#include <batchlas/device/exec.hh>
#include <batchlas/device/detail/sg_partition/sg_partition.hh>

namespace batchlas::device {

/// @brief A Partition whose partition object is a `SubGroupPartition<P, Masked>`.
/// @ingroup api_device_exec
///
/// A masked partition's lanes may diverge from the other partitions of the sub-group. The default
/// for Masked is the backend's (`sgp::kMaskedByDefault`).
/// @tparam P the partition size, a power of two up to the sub-group size
/// @tparam Masked true for a masked partition
template <int P, bool Masked = sgp::kMaskedByDefault>
using SgpPartition = Partition<P, Masked, SubGroupPartition<static_cast<size_t>(P), Masked>>;

/// @brief The device executor over a SubGroupPartition a kernel already holds.
/// @ingroup api_device_exec
/// @tparam P the partition size
/// @tparam Masked whether the partition is masked
/// @param part the partition object
/// @return the SgpPartition executor that adopts it
template <size_t P, bool Masked>
BATCHLAS_DEVICE_INLINE SgpPartition<static_cast<int>(P), Masked> wrap(SubGroupPartition<P, Masked> part) {
    return {typename SgpPartition<static_cast<int>(P), Masked>::Wrap{}, part};
}

/// @brief The executor of this item's P-lane chunk of sg.
/// @ingroup api_device_exec
/// @tparam P the partition size, a power of two up to the sub-group size
/// @tparam Masked true for a masked partition (default: the backend's)
/// @param sg the item's sub-group
/// @return `SgpPartition<P, Masked>` over this item's chunk
template <int P, bool Masked = sgp::kMaskedByDefault>
BATCHLAS_DEVICE_INLINE SgpPartition<P, Masked> make_partition(sycl::sub_group sg) {
    return wrap(batchlas::make_partition<static_cast<size_t>(P), Masked>(sg));
}

}  // namespace batchlas::device
