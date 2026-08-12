#pragma once

// SYRK and SYR2K name one triangle of C; the other half is the caller's storage
// and must come out of the call exactly as it went in.
//
// The generic fallbacks in this directory (used by the backends that have no
// native ?syrk/?syr2k here -- MKL) decompose the routine into GEMM, and a GEMM
// computes the *whole* symmetric product. It therefore cannot be aimed at C: it
// would write both triangles. The same reasoning as the CUDA HERK path in
// src/backends/cublas.cc ("The GEMM cannot be pointed at C ... HERK owns only one
// of them"). So the product goes to scratch and only the named triangle is folded
// back into C:
//
//     C(i, j) := product(i, j) + beta * C(i, j)   for (i, j) in the uplo triangle
//
// with beta == 0 meaning "C is not read", as in BLAS -- so an uninitialised or
// poisoned C cannot turn the result into NaN.

#include <batchlas/blas/matrix.hh>
#include <batchlas/util/kernel-heuristics.hh>

#include "../queue.hh"

#include <sycl/sycl.hpp>

namespace batchlas {
namespace detail {

template <typename T>
Event fold_symmetric_product_into_triangle(Queue& ctx,
                                           const MatrixView<T, MatrixFormat::Dense>& C,
                                           const MatrixView<T, MatrixFormat::Dense>& product,
                                           T beta,
                                           Uplo uplo) {
    const size_t n = static_cast<size_t>(C.rows());
    const size_t batch_size = static_cast<size_t>(C.batch_size());
    const size_t total_elements = batch_size * n * n;
    if (total_elements == 0) {
        return ctx.get_event();
    }

    T* c_ptr = C.data_ptr();
    const T* p_ptr = product.data_ptr();
    const size_t c_ld = static_cast<size_t>(C.ld());
    const size_t c_stride = static_cast<size_t>(C.stride());
    const size_t p_ld = static_cast<size_t>(product.ld());
    const size_t p_stride = static_cast<size_t>(product.stride());
    const bool upper = uplo == Uplo::Upper;
    const bool ignore_c = beta == T(0);

    auto [global_size, local_size] = compute_nd_range_sizes(
        total_elements, ctx.device(), KernelType::ELEMENTWISE);

    ctx->parallel_for(sycl::nd_range<1>(global_size, local_size), [=](sycl::nd_item<1> item) {
        const size_t flat_idx = item.get_global_id(0);
        if (flat_idx >= total_elements) return;

        const size_t b = flat_idx / (n * n);
        const size_t remainder = flat_idx % (n * n);
        const size_t i = remainder / n;  // row
        const size_t j = remainder % n;  // column

        const bool in_triangle = upper ? (i <= j) : (i >= j);
        if (!in_triangle) return;

        const size_t c_idx = b * c_stride + j * c_ld + i;
        const size_t p_idx = b * p_stride + j * p_ld + i;
        c_ptr[c_idx] = ignore_c ? p_ptr[p_idx] : p_ptr[p_idx] + beta * c_ptr[c_idx];
    });

    return ctx.get_event();
}

}  // namespace detail
}  // namespace batchlas
