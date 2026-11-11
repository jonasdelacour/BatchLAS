// SPDX-License-Identifier: MIT
#pragma once

// Host-side launch helpers the native launchers share; no kernel compiles differently through it.

#include <batchlas/blas/matrix.hh>
#include <batchlas/error.hh>
#include <batchlas/util/sycl-device-queue.hh>

#include <string>

namespace batchlas::native {

struct Need {
    bool square = false;  // A.rows() == A.cols()
    bool extents = true;  // A has rows, cols and batch >= 1
    int sub_group = 32;   // a sub-group size the device must ENUMERATE; 0 = none
    int min_wg = 0;       // the work-group the route launches; 0 = none
};

// can_run's launch gates, re-applied where a caller reaches a native route without choose(): A's
// shape, every operand homogeneous (one launch reads one extent tuple), a GPU queue, the sub-group
// size from the enumerated list (MAX_SUB_GROUP_SIZE reports entry [0], so it accepts a {64}-only
// device and the launch aborts), and the launched work-group. An operand or the queue type throws
// invalid_argument, a missing device feature unsupported; every message opens with `who`.
template <class T, MatrixFormat MF, class... Ops>
Device require_launchable(Queue& ctx, const char* who, const MatrixView<T, MF>& A, Need need, const Ops&... ops) {
    const std::string w(who);
    if (need.square && A.rows() != A.cols()) throw batchlas::invalid_argument(w + ": A must be square");
    if (need.extents && (A.rows() < 1 || A.cols() < 1 || A.batch_size() < 1))
        throw batchlas::invalid_argument(w + ": degenerate extents");
    if (A.is_heterogeneous() || (ops.is_heterogeneous() || ...))
        throw batchlas::invalid_argument(w + ": heterogeneous batch is not supported");
    const Device dev = ctx.device();
    if (dev.type != DeviceType::GPU) throw batchlas::invalid_argument(w + ": GPU queues only");
    if (need.sub_group > 0 && !dev.supports_sub_group_size(static_cast<std::size_t>(need.sub_group)))
        throw batchlas::unsupported(w + ": device does not offer sub-group size " + std::to_string(need.sub_group) +
                                    ", which the kernel requires");
    if (need.min_wg > 0 && static_cast<int>(dev.get_property(DeviceProperty::MAX_WORK_GROUP_SIZE)) < need.min_wg)
        throw batchlas::unsupported(w + ": the route launches a work-group of " + std::to_string(need.min_wg) +
                                    ", which exceeds this device's maximum");
    return dev;
}

}  // namespace batchlas::native
