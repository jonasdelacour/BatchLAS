// SPDX-License-Identifier: MIT
#pragma once

#include <batchlas/blas/matrix.hh>

#include <src/math-helpers.hh>
#include <src/queue.hh>
#include <src/ops/syev/sytrd/sytrd_sb2st_cta_instantiations.hh>
#include <batchlas/device/lapack/sb2st.hh>

#include <sycl/sycl.hpp>

#include <complex>
#include <cstdint>

namespace batchlas {

namespace detail {

template <typename T>
Event btrd_lower_inplace_subgroup_dispatch(Queue& q,
                                          KernelMatrixView<T, MatrixFormat::Dense> ab,
                                          int n,
                                          int kd,
                                          VectorView<typename base_type<T>::type> c,
                                          VectorView<T> work,
                                          VectorView<typename base_type<T>::type> d,
                                          VectorView<typename base_type<T>::type> e,
                                          VectorView<T> tau,
                                          const std::uint8_t* mask);

// Explicit instantiations live in sytrd_sb2st_cta.cc.
#define BTRD_LOWER_INPLACE_SUBGROUP_DISPATCH_EXTERN(fp, real_fp) \
extern template Event btrd_lower_inplace_subgroup_dispatch<BATCHLAS_UNPAREN fp>(Queue&, \
                                                                                 KernelMatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>, \
                                                                                 int, \
                                                                                 int, \
                                                                                 VectorView<BATCHLAS_UNPAREN real_fp>, \
                                                                                 VectorView<BATCHLAS_UNPAREN fp>, \
                                                                                 VectorView<BATCHLAS_UNPAREN real_fp>, \
                                                                                 VectorView<BATCHLAS_UNPAREN real_fp>, \
                                                                                 VectorView<BATCHLAS_UNPAREN fp>, \
                                                                                 const std::uint8_t*);

BATCHLAS_FOR_EACH_SYTRD_SB2ST_CTA_DISPATCH_TYPE(BTRD_LOWER_INPLACE_SUBGROUP_DISPATCH_EXTERN)

#undef BTRD_LOWER_INPLACE_SUBGROUP_DISPATCH_EXTERN

} // namespace detail

} // namespace batchlas
