#pragma once

#include <batchlas/blas/extensions.hh>

namespace batchlas {

template <Backend B, typename T>
Event steqr_legacy(Queue& ctx,
                   const VectorView<T>& d_in,
                   const VectorView<T>& e_in,
                   const VectorView<T>& eigenvalues,
                   const Span<std::byte>& ws,
                   JobType jobz,
                   SteqrParams<T> params,
                   const MatrixView<T, MatrixFormat::Dense>& eigvects,
                   Span<int32_t> info = Span<int32_t>());

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
               const MatrixView<T, MatrixFormat::Dense>& eigvects,
               Span<int32_t> info = Span<int32_t>());

template <typename T>
size_t steqr_wg_buffer_size(Queue& ctx,
                            const VectorView<T>& d,
                            const VectorView<T>& e,
                            const VectorView<T>& eigenvalues,
                            JobType jobz,
                            SteqrParams<T> params);

// steqr's route choice WITHOUT the `info` clear.
//
// The public `steqr` is exactly `detail::info_clear` followed by this, and the two
// tiers below it only ever RAISE a status (info_span.hh). stedc's leaves call this
// one: its recursive driver solves the two halves of the SAME batch items with two
// separate leaf solves, so a leaf that cleared would erase the first half's failure
// and stedc would report success on an item that did not converge.
template <Backend B, typename T>
Event steqr_dispatch(Queue& ctx,
                     const VectorView<T>& d_in,
                     const VectorView<T>& e_in,
                     const VectorView<T>& eigenvalues,
                     const Span<std::byte>& ws,
                     JobType jobz,
                     SteqrParams<T> params,
                     const MatrixView<T, MatrixFormat::Dense>& eigvects,
                     Span<int32_t> info = Span<int32_t>());

} // namespace batchlas
