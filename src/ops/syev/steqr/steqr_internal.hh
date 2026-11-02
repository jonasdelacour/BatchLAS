// SPDX-License-Identifier: MIT
#pragma once

#include <batchlas/blas/extensions.hh>

#include <src/select/select.hh>

#include <cstdint>

namespace batchlas {

// Correctness only: a GPU with 32-wide sub-groups and 1 <= n <= 32. The routers also exclude
// ROCm (wrong eigenvalues on gfx1200). For 1 <= n <= 32, steqr_cta falls back to steqr_wg when this
// is false; for n outside [1, 32] it throws invalid_argument (the routers check this first).
inline bool steqr_cta_can_run(const select::Device& d, int64_t n) {
    return d.is_gpu && d.has_sg32 && n >= 1 && n <= 32;
}

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

// steqr's route choice WITHOUT the `info` clear (steqr = info_clear + this; the tiers only
// raise, info_span.hh). stedc's leaves call this: its recursive driver solves two halves of the
// SAME items separately, so a leaf that cleared would erase the first half's failure.
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
