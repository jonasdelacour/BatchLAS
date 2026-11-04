// SPDX-License-Identifier: MIT
#pragma once

// Device-side CTA SYTRD, shared by sytrd_cta.cc and syev_cta_fused.cc: one definition keeps
// the fused and partitioned syev numerically identical, so comparing them measures fusion only.

#include <batchlas/blas/matrix.hh>
#include <batchlas/util/group-invoke.hh>
#include <batchlas/device/sg_partition.hh>
#include <batchlas/device/detail/sg_partition/sg_partition.hh>
#include <src/math-helpers.hh>
#include <batchlas/device/lapack/sytrd.hh>

#include <cstdint>

namespace batchlas {

    template <typename T>
    using LocalDecorated = typename sycl::multi_ptr<T, sycl::access::address_space::local_space,
                                                    sycl::access::decorated::yes>::pointer;

    // device::sytd2<Upper> on the full Hermitian tile (ld LDA, zero past n) at A_local[base_a];
    // returns tau(lane), 0 for lane >= n-1. The inliner declines this body for complex T, so the
    // call carries decorated local pointers by value: an accessor taken by reference is copied
    // to the stack (cf/cd Upper STACK 0->344), and &A_local[base] is generic.
    template <typename T, int32_t LDA, size_t P, bool M>
    inline T sytd2_cta_upper_tile(const SubGroupPartition<P, M>& part, LocalDecorated<T> A_local, int32_t base_a,
                                  LocalDecorated<T> V_local, int32_t base_v, LocalDecorated<T> W_local,
                                  int32_t base_w, int32_t n) {
        namespace v2 = batchlas::device;
        using LP = LocalDecorated<T>;
        using Tile = v2::Mat<T, v2::Dyn, v2::Dyn, v2::Ext<LDA>, v2::Mem::Local, v2::Cyclic, LP>;
        using Col = v2::Vec<T, v2::Dyn, v2::Unit, v2::Mem::Local, v2::Cyclic, LP>;
        return v2::sytd2<Uplo::Upper>(v2::wrap(part), Tile{A_local, v2::Dyn{n}, v2::Dyn{n}, {}, base_a},
                                      Col{V_local, v2::Dyn{n}, {}, base_v}, Col{W_local, v2::Dyn{n}, {}, base_w}, n);
    }

    template <typename T, int32_t LDA, size_t P, bool M>
    BATCHLAS_DLA_INLINE T sytd2_cta_upper_partition(const SubGroupPartition<P, M>& part,
                                                    const sycl::local_accessor<T, 1>& A_local, int32_t base_a,
                                                    const sycl::local_accessor<T, 1>& V_local, int32_t base_v,
                                                    const sycl::local_accessor<T, 1>& W_local, int32_t base_w,
                                                    int32_t n) {
        constexpr auto yes = sycl::access::decorated::yes;
        return sytd2_cta_upper_tile<T, LDA>(part, A_local.template get_multi_ptr<yes>().get(), base_a,
                                            V_local.template get_multi_ptr<yes>().get(), base_v,
                                            W_local.template get_multi_ptr<yes>().get(), base_w, n);
    }

} // namespace batchlas
