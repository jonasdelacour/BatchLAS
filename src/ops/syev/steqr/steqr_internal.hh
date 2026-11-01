// SPDX-License-Identifier: MIT
#pragma once

#include <batchlas/blas/extensions.hh>

#include <src/select/select.hh>

#include <cstdint>

namespace batchlas {

// An unreduced block (D = d[0, len), E = e[0, len-1), physical) in its sweep's orientation: QL is
// QR on J T J, so reversed maps virtual k to len-1-k. The sweep converges the virtual end len-1.
template <typename T> struct OrientedBlock {
    VectorView<T> D, E;
    int32_t len;
    bool reversed;

    T& d(int32_t k) const { return D(reversed ? len - 1 - k : k); }
    T& e(int32_t k) const { return E(reversed ? len - 2 - k : k); }
};

// Correctness only: a GPU with 32-wide sub-groups and 1 <= n <= 32. The routers also exclude
// ROCm (wrong eigenvalues on gfx1200); steqr_cta falls back to steqr_wg when this is false.
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

// NETLIB stedc's solver until WP5: legacy gave steqr_wg's bits, so it forwards (size kept below).
template <Backend B, typename T>
inline Event steqr_legacy(Queue& ctx,
                          const VectorView<T>& d_in,
                          const VectorView<T>& e_in,
                          const VectorView<T>& eigenvalues,
                          const Span<std::byte>& ws,
                          JobType jobz,
                          SteqrParams<T> params,
                          const MatrixView<T, MatrixFormat::Dense>& eigvects,
                          Span<int32_t> info = Span<int32_t>()) {
    return steqr_wg<B, T>(ctx, d_in, e_in, eigenvalues, ws, jobz, params, eigvects, info);
}

// steqr_wg's size plus the two buffers only legacy carved: NETLIB stedc's workspace is unchanged.
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
