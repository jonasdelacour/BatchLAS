#pragma once

#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/blas/enums.hh>
#include <batchlas/blas/queue-dispatch.hh>

namespace batchlas {

// Signature aliases for explicit instantiation; see BATCHLAS_INSTANTIATE in
// src/util/template-instantiations.hh. Keep in sync with the declarations below.
namespace sig {
template <typename T>
using her2k = Event(Queue&,
                    const MatrixView<T, MatrixFormat::Dense>&,
                    const MatrixView<T, MatrixFormat::Dense>&,
                    const MatrixView<T, MatrixFormat::Dense>&,
                    T, float_t<T>, Uplo, Transpose);

// backend::her2k_vendor's signature. NOT an alias for sig::her2k: the vendor
// parameter order can differ from the public one -- trsm's alpha moves to
// the end -- so each is spelled out from the definition it describes.
template <typename T>
using her2k_vendor = Event(Queue&,
                          const MatrixView<T, MatrixFormat::Dense>&,
                          const MatrixView<T, MatrixFormat::Dense>&,
                          const MatrixView<T, MatrixFormat::Dense>&,
                          T,
                          float_t<T>,
                          Uplo,
                          Transpose);
}  // namespace sig


// C = alpha * A * B^H + conj(alpha) * B * A^H + beta * C (Transpose::NoTrans,
// A and B are n x k) or
// C = alpha * A^H * B + conj(alpha) * B^H * A + beta * C (Transpose::ConjTrans,
// A and B are k x n), with C Hermitian n x n: only the triangle named by `uplo`
// is written, and the diagonal comes out real.
//
// The conjugate on the second term is what makes the sum Hermitian, and is the
// whole of the difference from syr2k -- the second term is the conjugate
// transpose of the first, not a copy of it with the operands swapped. It is
// also why alpha may be complex here while herk's must be real: the pair
// alpha * A * B^H and its own conjugate transpose is Hermitian for any alpha.
// beta scales an already-Hermitian C and so is still real.
template <Backend Ba, ComplexScalar T>
Event her2k(Queue& ctx,
            const MatrixView<T, MatrixFormat::Dense>& A,
            const MatrixView<T, MatrixFormat::Dense>& B,
            const MatrixView<T, MatrixFormat::Dense>& C,
            T alpha,
            float_t<T> beta,
            Uplo uplo,
            Transpose transA);

template <Backend Ba, ComplexScalar T>
inline Event her2k(Queue& ctx,
                   const Matrix<T, MatrixFormat::Dense>& A,
                   const Matrix<T, MatrixFormat::Dense>& Bmat,
                   const Matrix<T, MatrixFormat::Dense>& Cmat,
                   T alpha,
                   float_t<T> beta,
                   Uplo uplo,
                   Transpose transA) {
    return her2k<Ba, T>(ctx,
                        MatrixView<T, MatrixFormat::Dense>(A),
                        MatrixView<T, MatrixFormat::Dense>(Bmat),
                        MatrixView<T, MatrixFormat::Dense>(Cmat),
                        alpha,
                        beta,
                        uplo,
                        transA);
}

} // namespace batchlas


namespace batchlas::backend {

// The vendor path for her2k.
//
// DECLARATION ONLY. The public `her2k<Back, T>` used to be DEFINED inside each
// vendor TU, so dropping a vendor library dropped the public entry point along
// with the vendor path. WP0 S5 moves that definition to
// src/dispatch/entry_points/level3.cc; what stays behind is the vendor
// implementation, named as such. Each vendor wrapper TU defines this primary
// template for its own Backend value and instantiates it there.
template <Backend Back, ComplexScalar T>
Event her2k_vendor(Queue& ctx,
                   const MatrixView<T, MatrixFormat::Dense>& A,
                   const MatrixView<T, MatrixFormat::Dense>& B,
                   const MatrixView<T, MatrixFormat::Dense>& C,
                   T alpha,
                   float_t<T> beta,
                   Uplo uplo,
                   Transpose transA);

}  // namespace batchlas::backend

namespace batchlas {

// Backend-deducing overloads: `f(ctx, ...)` uses ctx.backend().
// See BATCHLAS_DISPATCH_ON_QUEUE in blas/queue-dispatch.hh.

BATCHLAS_DISPATCH_ON_QUEUE(her2k)

}  // namespace batchlas
