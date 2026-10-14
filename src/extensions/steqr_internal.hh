#pragma once

#include <batchlas/blas/extensions.hh>

#include "../math-helpers.hh"

#include <array>

namespace batchlas {

namespace internal {

// {c, -s} of LAPACK lartg(a, b): the rotation convention of the work-group bulge chase.
template <typename T>
inline std::array<T, 2> givens_rotation(const T& a, const T& b) {
    auto [c_l, s_l, r] = internal::lartg(a, b);
    (void)r;
    return std::array<T, 2>{c_l, -s_l};
}

// Similarity transform of rows/cols i, j of the tridiagonal (d, e); returns the new bulge.
// QR == true indexes top-down (virtual k == physical k), false bottom-up (k -> n-1-k), so one
// bulge chase implements both the QR and the QL iteration.
template <typename T>
inline T apply_givens_rotation(const VectorView<T>& d,
                               const VectorView<T>& e,
                               const T& prev_bulge,
                               size_t i,
                               size_t j,
                               const std::array<T, 2>& givens,
                               bool QR) {
    const T c = givens[0];
    const T s = givens[1];

    const size_t n = d.size();
    const size_t ne = e.size();

    auto d_get = [&](size_t k) -> T {
        return QR ? d(k) : d(n - 1 - k);
    };
    auto d_set = [&](size_t k, T val) {
        if (QR) {
            d(k) = val;
        } else {
            d(n - 1 - k) = val;
        }
    };
    auto e_get = [&](size_t k) -> T {
        return QR ? e(k) : e(ne - 1 - k);
    };
    auto e_set = [&](size_t k, T val) {
        if (QR) {
            e(k) = val;
        } else {
            e(ne - 1 - k) = val;
        }
    };

    T di = d_get(i);
    T dj = d_get(j);
    T ei = e_get(i);
    T ej = (j < ne) ? e_get(j) : T(0);

    T di_new = c * (c * di - ei * s) - s * (ei * c - s * dj);
    T dj_new = c * (c * dj + ei * s) + s * (ei * c + s * di);
    d_set(i, di_new);
    d_set(j, dj_new);

    if (i > 0) {
        T e_im1 = e_get(i - 1);
        e_set(i - 1, e_im1 * c - prev_bulge * s);
    }

    T ei_new = c * (c * ei + s * di) - s * (c * dj + s * ei);
    e_set(i, ei_new);

    if (j < ne) {
        e_set(j, c * ej);
    }

    return -ej * s;
}

} // namespace internal

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
