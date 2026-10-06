#pragma once

// herk's and her2k's `fold` (src/ops/{herk,her2k}): portable SYCL, so it runs vendor-free.

#include "triangular_expand.hh"

#include "../queue.hh"

#include <batchlas/blas/enums.hh>
#include <batchlas/blas/matrix.hh>

#include <sycl/sycl.hpp>

#include <cstddef>

namespace batchlas::backend::detail {

// Fold a dense rank-k product into the referenced triangle of a Hermitian C:
// C = product + beta * C, or, for TwoSided, C = product + product^H + beta * C.
//
// TwoSided is how HER2K gets its second term for free. The two terms
// alpha * A * B^H and conj(alpha) * B * A^H are conjugate transposes of one
// another, so one GEMM produces both and the mirrored read below adds them.
//
// The unreferenced triangle is neither read nor written, and the diagonal is
// real on exit (C = C^H), whatever the caller left in its imaginary part.
// Batch in SYCL dim 0 (grid z, <= 65535 groups); groups of 256 (expand_group_shape).
template <typename T, bool TwoSided>
Event accumulate_hermitian(Queue& ctx,
                           const MatrixView<T, MatrixFormat::Dense>& C,
                           const MatrixView<T, MatrixFormat::Dense>& product,
                           float_t<T> beta,
                           Uplo uplo) {
    using real_t = float_t<T>;
    const int n = C.rows();
    const int batch = C.batch_size();
    const bool lower = uplo == Uplo::Lower;

    T* dst = C.data_ptr();
    const T* src = product.data_ptr();
    const int ldc = C.ld();
    const int ldp = product.ld();
    const std::size_t stride_c = static_cast<std::size_t>(C.stride());
    const std::size_t stride_p = static_cast<std::size_t>(product.stride());

    const auto shape = expand_group_shape(n);
    const sycl::range<3> global(static_cast<std::size_t>(batch),
                                static_cast<std::size_t>(ceil_div(n, shape.cols) * shape.cols),
                                static_cast<std::size_t>(ceil_div(n, shape.rows) * shape.rows));
    const sycl::range<3> local(1,
                               static_cast<std::size_t>(shape.cols),
                               static_cast<std::size_t>(shape.rows));

    ctx->parallel_for(sycl::nd_range<3>(global, local), [=](sycl::nd_item<3> item) {
        const int i = static_cast<int>(item.get_global_id(2));
        const int j = static_cast<int>(item.get_global_id(1));
        if (i >= n || j >= n || (lower ? (i < j) : (i > j))) {
            return;
        }
        const int b = static_cast<int>(item.get_group(0));

        const std::size_t p_base = static_cast<std::size_t>(b) * stride_p;
        T value = src[p_base + static_cast<std::size_t>(j) * ldp + i];
        if constexpr (TwoSided) {
            const T mirrored = src[p_base + static_cast<std::size_t>(i) * ldp + j];
            value = T(value.real() + mirrored.real(), value.imag() - mirrored.imag());
        }

        T* c = dst + static_cast<std::size_t>(b) * stride_c +
               static_cast<std::size_t>(j) * ldc + i;
        if (beta != real_t(0)) {
            // The diagonal's imaginary part is not an input (C is Hermitian, beta real).
            const T prev = *c;
            value = T(value.real() + beta * prev.real(),
                      i == j ? value.imag() : value.imag() + beta * prev.imag());
        }
        *c = i == j ? T(value.real(), real_t(0)) : value;
    });

    return ctx.get_event();
}

}  // namespace batchlas::backend::detail
