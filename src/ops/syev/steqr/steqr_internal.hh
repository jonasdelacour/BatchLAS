// SPDX-License-Identifier: MIT
#pragma once

#include <batchlas/blas/extensions.hh>

#include <src/math-helpers.hh>

#include <array>

namespace batchlas {

template <Backend B, typename Real>
Event steqr_legacy(Queue& ctx,
                   const VectorView<Real>& d_in,
                   const VectorView<Real>& e_in,
                   const VectorView<Real>& eigenvalues,
                   const Span<std::byte>& ws,
                   JobType jobz,
                   SteqrParams<Real> params,
                   const MatrixView<Real, MatrixFormat::Dense>& eigvects,
                   Span<int32_t> info = Span<int32_t>());

template <typename Real>
size_t steqr_legacy_buffer_size(Queue& ctx,
                                const VectorView<Real>& d,
                                const VectorView<Real>& e,
                                const VectorView<Real>& eigenvalues,
                                JobType jobz,
                                SteqrParams<Real> params);

template <Backend B, typename Real>
Event steqr_wg(Queue& ctx,
               const VectorView<Real>& d_in,
               const VectorView<Real>& e_in,
               const VectorView<Real>& eigenvalues,
               const Span<std::byte>& ws,
               JobType jobz,
               SteqrParams<Real> params,
               const MatrixView<Real, MatrixFormat::Dense>& eigvects,
               Span<int32_t> info = Span<int32_t>());

template <typename Real>
size_t steqr_wg_buffer_size(Queue& ctx,
                            const VectorView<Real>& d,
                            const VectorView<Real>& e,
                            const VectorView<Real>& eigenvalues,
                            JobType jobz,
                            SteqrParams<Real> params);

// steqr's route choice WITHOUT the `info` clear.
//
// The public `steqr` is exactly `detail::info_clear` followed by this, and the two
// tiers below it only ever RAISE a status (info_span.hh). stedc's leaves call this
// one: its recursive driver solves the two halves of the SAME batch items with two
// separate leaf solves, so a leaf that cleared would erase the first half's failure
// and stedc would report success on an item that did not converge.
template <Backend B, typename Real>
Event steqr_dispatch(Queue& ctx,
                     const VectorView<Real>& d_in,
                     const VectorView<Real>& e_in,
                     const VectorView<Real>& eigenvalues,
                     const Span<std::byte>& ws,
                     JobType jobz,
                     SteqrParams<Real> params,
                     const MatrixView<Real, MatrixFormat::Dense>& eigvects,
                     Span<int32_t> info = Span<int32_t>());

}  // namespace batchlas
