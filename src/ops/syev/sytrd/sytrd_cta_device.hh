// SPDX-License-Identifier: MIT
#pragma once

// Device-side CTA SYTRD for the fused syev kernel: one definition (device/lapack/sytrd.hh)
// keeps the fused and partitioned syev numerically identical, so comparing them measures fusion.

#include <batchlas/blas/matrix.hh>
#include <batchlas/util/group-invoke.hh>
#include <batchlas/device/sg_partition.hh>
#include <batchlas/device/detail/sg_partition/sg_partition.hh>
#include <src/math-helpers.hh>
#include <batchlas/device/lapack/sytrd.hh>

#include <cstdint>

namespace batchlas {

    // device::sytrd_detail::sytd2_upper_tile on the accessors' decorated pointers.
    template <typename T, int32_t LDA, size_t P, bool M>
    BATCHLAS_DLA_INLINE T sytd2_cta_upper_partition(const SubGroupPartition<P, M>& part,
                                                    const sycl::local_accessor<T, 1>& A_local, int32_t base_a,
                                                    const sycl::local_accessor<T, 1>& V_local, int32_t base_v,
                                                    const sycl::local_accessor<T, 1>& W_local, int32_t base_w,
                                                    int32_t n) {
        namespace sd = device::sytrd_detail;
        return sd::sytd2_upper_tile<T, LDA>(part, sd::decorated_ptr(A_local), base_a, sd::decorated_ptr(V_local), base_v,
                                            sd::decorated_ptr(W_local), base_w, n);
    }

} // namespace batchlas
