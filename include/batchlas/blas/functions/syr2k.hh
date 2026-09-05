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
using syr2k = Event(Queue&,
                    const MatrixView<T, MatrixFormat::Dense>&,
                    const MatrixView<T, MatrixFormat::Dense>&,
                    const MatrixView<T, MatrixFormat::Dense>&,
                    T, T, Uplo, Transpose);

// backend::syr2k_vendor's signature. NOT an alias for sig::syr2k: the vendor
// parameter order can differ from the public one -- trsm's alpha moves to
// the end -- so each is spelled out from the definition it describes.
template <typename T>
using syr2k_vendor = Event(Queue&,
                          const MatrixView<T, MatrixFormat::Dense>&,
                          const MatrixView<T, MatrixFormat::Dense>&,
                          const MatrixView<T, MatrixFormat::Dense>&,
                          T,
                          T,
                          Uplo,
                          Transpose);
}  // namespace sig


template <Backend Ba, RealScalar T>
Event syr2k(Queue& ctx,
            const MatrixView<T, MatrixFormat::Dense>& A,
            const MatrixView<T, MatrixFormat::Dense>& B,
            const MatrixView<T, MatrixFormat::Dense>& C,
            T alpha,
            T beta,
            Uplo uplo,
            Transpose transA);

template <Backend Ba, RealScalar T>
inline Event syr2k(Queue& ctx,
                   const Matrix<T, MatrixFormat::Dense>& A,
                   const Matrix<T, MatrixFormat::Dense>& Bmat,
                   const Matrix<T, MatrixFormat::Dense>& Cmat,
                   T alpha,
                   T beta,
                   Uplo uplo,
                   Transpose transA) {
    return syr2k<Ba, T>(ctx,
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

// The vendor path for syr2k.
//
// DECLARATION ONLY. The public `syr2k<Back, T>` used to be DEFINED inside each
// vendor TU, so dropping a vendor library dropped the public entry point along
// with the vendor path. WP0 S5 moves that definition to
// src/dispatch/entry_points/level3.cc; what stays behind is the vendor
// implementation, named as such. Each vendor wrapper TU defines this primary
// template for its own Backend value and instantiates it there.
template <Backend Back, RealScalar T>
Event syr2k_vendor(Queue& ctx,
                   const MatrixView<T, MatrixFormat::Dense>& A,
                   const MatrixView<T, MatrixFormat::Dense>& B,
                   const MatrixView<T, MatrixFormat::Dense>& C,
                   T alpha,
                   T beta,
                   Uplo uplo,
                   Transpose transA);

}  // namespace batchlas::backend

namespace batchlas {

// Backend-deducing overloads: `f(ctx, ...)` uses ctx.backend().
// See BATCHLAS_DISPATCH_ON_QUEUE in blas/queue-dispatch.hh.

BATCHLAS_DISPATCH_ON_QUEUE(syr2k)

}  // namespace batchlas
