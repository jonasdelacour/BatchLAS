#pragma once

/// @file
/// @brief Apply Q or P from a bidiagonal reduction (ormbr). Kernel helper, not API.
///
/// Installed with the rest of `include/batchlas` (the install rule copies the
/// tree wholesale); no public header includes it. Not a stable interface.
/// The declarations here repeat the documented ones in `batchlas/blas/extensions.hh`
/// (batchlas::ormbr, batchlas::ormbr_buffer_size) without default arguments.
///
/// Contract, as LAPACK `?ormbr` / `?unmbr`: `vect = 'Q'` applies the left
/// reflectors stored in `a` (Q of order `a.rows()`), `'P'` the right reflectors
/// (P of order `a.cols()`); C is overwritten with \f$ \mathrm{op}(X) C \f$ or
/// \f$ C\, \mathrm{op}(X) \f$ according to `side` and `trans`. `tau` must be
/// unit-stride and packed by batch. Throws batchlas::invalid_argument on
/// mismatched batch sizes or orders, a bad `vect` or a short / strided `tau`,
/// and batchlas::unsupported for `Transpose::Trans` with complex T and `'P'`
/// (use `ConjTrans`).
/// @ingroup internal_helpers

#include <batchlas/export.hh>
#include <batchlas/blas/enums.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>

namespace batchlas {

template <Backend B, typename T>
BATCHLAS_API Event ormbr(Queue& ctx,
                         const MatrixView<T, MatrixFormat::Dense>& a,
                         const VectorView<T>& tau,
                         const MatrixView<T, MatrixFormat::Dense>& c,
                         char vect,
                         Side side,
                         Transpose trans,
                         const Span<std::byte>& ws,
                         int32_t block_size);

template <Backend B, typename T>
BATCHLAS_API size_t ormbr_buffer_size(Queue& ctx,
                                      const MatrixView<T, MatrixFormat::Dense>& a,
                                      const VectorView<T>& tau,
                                      const MatrixView<T, MatrixFormat::Dense>& c,
                                      char vect,
                                      Side side,
                                      Transpose trans,
                                      int32_t block_size);

} // namespace batchlas
