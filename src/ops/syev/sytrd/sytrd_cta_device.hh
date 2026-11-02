// SPDX-License-Identifier: MIT
#pragma once

// Device-side CTA SYTRD, shared by sytrd_cta.cc and syev_cta_fused.cc: one definition keeps
// the fused and partitioned syev numerically identical, so comparing them measures fusion only.

#include <batchlas/blas/matrix.hh>
#include <batchlas/blas/device/v2/ext/sytrd-cta-blocked-band.hh>
#include <batchlas/util/group-invoke.hh>
#include <src/kernels/common/sg_partition/sg_partition.hh>
#include <src/math-helpers.hh>

#include <cstdint>
#include <batchlas/blas/device/detail/group_blas_common.hh>

namespace batchlas {

    // v2::sytd2<Upper> on a partition-resident tile (ld LDA) holding the full Hermitian matrix,
    // zero-padded past n; V_local/W_local hold n. Returns tau(lane) (zero for lane >= n-1).
    template <typename T, int32_t LDA, typename LocalPtr, typename Partition>
    inline T sytd2_cta_upper_partition(const Partition& part,
                                       LocalPtr A_local,
                                       LocalPtr V_local,
                                       LocalPtr W_local,
                                       int32_t n,
                                       int32_t lane) {
        namespace v2 = batchlas::device::v2;
        (void)lane;  // the executor's lid()
        using Tile = v2::Mat<T, v2::Dyn, v2::Dyn, v2::Ext<LDA>, v2::Mem::Local>;
        return v2::sytd2<Uplo::Upper>(v2::adopt_partition(part), Tile{A_local, v2::Dyn{n}, v2::Dyn{n}, {}},
                                      v2::vec<v2::Mem::Local>(V_local, n), v2::vec<v2::Mem::Local>(W_local, n), n);
    }

} // namespace batchlas
