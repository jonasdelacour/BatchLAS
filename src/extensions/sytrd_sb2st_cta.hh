#pragma once

#include <blas/matrix.hh>

#include "../queue.hh"

#include <complex>
#include <cstdint>

namespace batchlas {

namespace internal {

template <typename T>
Event btrd_lower_inplace_subgroup_dispatch(Queue& q,
                                          KernelMatrixView<T, MatrixFormat::Dense> ab,
                                          int n,
                                          int kd,
                                          VectorView<typename base_type<T>::type> c,
                                          VectorView<T> work,
                                          VectorView<typename base_type<T>::type> d,
                                          VectorView<typename base_type<T>::type> e,
                                          VectorView<T> tau);

// Explicit instantiations live in sytrd_sb2st_cta.cc.
extern template Event btrd_lower_inplace_subgroup_dispatch<float>(Queue&,
                                                                  KernelMatrixView<float, MatrixFormat::Dense>,
                                                                  int,
                                                                  int,
                                                                  VectorView<float>,
                                                                  VectorView<float>,
                                                                  VectorView<float>,
                                                                  VectorView<float>,
                                                                  VectorView<float>);
extern template Event btrd_lower_inplace_subgroup_dispatch<double>(Queue&,
                                                                   KernelMatrixView<double, MatrixFormat::Dense>,
                                                                   int,
                                                                   int,
                                                                   VectorView<double>,
                                                                   VectorView<double>,
                                                                   VectorView<double>,
                                                                   VectorView<double>,
                                                                   VectorView<double>);
extern template Event btrd_lower_inplace_subgroup_dispatch<std::complex<float>>(
    Queue&,
    KernelMatrixView<std::complex<float>, MatrixFormat::Dense>,
    int,
    int,
    VectorView<float>,
    VectorView<std::complex<float>>,
    VectorView<float>,
    VectorView<float>,
    VectorView<std::complex<float>>);
extern template Event btrd_lower_inplace_subgroup_dispatch<std::complex<double>>(
    Queue&,
    KernelMatrixView<std::complex<double>, MatrixFormat::Dense>,
    int,
    int,
    VectorView<double>,
    VectorView<std::complex<double>>,
    VectorView<double>,
    VectorView<double>,
    VectorView<std::complex<double>>);

} // namespace internal

} // namespace batchlas
