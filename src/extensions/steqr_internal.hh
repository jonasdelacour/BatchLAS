#pragma once

#include <blas/extensions.hh>

namespace batchlas {

template <Backend B, typename T>
Event steqr_legacy(Queue& ctx,
                   const VectorView<T>& d_in,
                   const VectorView<T>& e_in,
                   const VectorView<T>& eigenvalues,
                   const Span<std::byte>& ws,
                   JobType jobz,
                   SteqrParams<T> params,
                   const MatrixView<T, MatrixFormat::Dense>& eigvects);

template <typename T>
size_t steqr_legacy_buffer_size(Queue& ctx,
                                const VectorView<T>& d,
                                const VectorView<T>& e,
                                const VectorView<T>& eigenvalues,
                                JobType jobz,
                                SteqrParams<T> params);

template <Backend B, typename T>
Event steqr_wg(Queue& ctx,
               const VectorView<T>& d_in,
               const VectorView<T>& e_in,
               const VectorView<T>& eigenvalues,
               const Span<std::byte>& ws,
               JobType jobz,
               SteqrParams<T> params,
               const MatrixView<T, MatrixFormat::Dense>& eigvects);

template <typename T>
size_t steqr_wg_buffer_size(Queue& ctx,
                            const VectorView<T>& d,
                            const VectorView<T>& e,
                            const VectorView<T>& eigenvalues,
                            JobType jobz,
                            SteqrParams<T> params);

} // namespace batchlas
