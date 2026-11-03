// SPDX-License-Identifier: MIT
#pragma once

// Device-side CTA SYTRD, shared by sytrd_cta.cc and syev_cta_fused.cc: one definition keeps
// the fused and partitioned syev numerically identical, so comparing them measures fusion only.

#include <batchlas/blas/matrix.hh>
#include <batchlas/util/group-invoke.hh>
#include <src/kernels/common/dla_v2_partition.hh>
#include <src/kernels/common/sg_partition/sg_partition.hh>
#include <src/math-helpers.hh>
#include <src/ops/syev/sytrd/sytrd_v2.hh>

#include <cstdint>
#include <batchlas/blas/device/detail/group_blas_common.hh>

namespace batchlas {

    // sytrd_v2::sytd2<Upper> on the full Hermitian tile (ld LDA, zero past n) at A_local[base_a];
    // returns tau(lane), 0 for lane >= n-1. Accessor-based views: &A_local[base] is generic.
    template <typename T, int32_t LDA, size_t P, bool M>
    inline T sytd2_cta_upper_partition(const SubGroupPartition<P, M>& part,
                                       const sycl::local_accessor<T, 1>& A_local, int32_t base_a,
                                       const sycl::local_accessor<T, 1>& V_local, int32_t base_v,
                                       const sycl::local_accessor<T, 1>& W_local, int32_t base_w,
                                       int32_t n) {
        namespace v2 = batchlas::device::v2;
        using Local = sycl::local_accessor<T, 1>;
        using Tile = v2::Mat<T, v2::Dyn, v2::Dyn, v2::Ext<LDA>, v2::Mem::Local, v2::Cyclic, Local>;
        return sytrd_v2::sytd2<Uplo::Upper>(v2::wrap(part), Tile{A_local, v2::Dyn{n}, v2::Dyn{n}, {}, base_a},
                                            v2::vec(V_local, n, base_v), v2::vec(W_local, n, base_w), n);
    }

} // namespace batchlas
