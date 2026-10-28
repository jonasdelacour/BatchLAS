// sygst's TwoSided family (sygst_native.hh): mirror A, then apply the factor from both sides with two
// full trsm (itype 1) or trmm (itypes 2, 3). evidence: docs/perf/sygvd.md#sygvd-routes

#include <batchlas/backend_config.h>

#include <batchlas/blas/functions/trmm.hh>
#include <batchlas/blas/functions/trsm.hh>

#include <batchlas/util/mempool.hh>

#include <src/ops/sygst/sygst_native.hh>
#include <src/util/batch_mask.hh>
#include <src/util/template-instantiations.hh>

#include <complex>
#include <cstddef>

namespace batchlas::ops::sygst {

// itypes 2 and 3 form one product out of place: an n x n buffer and its pointer array per item.
template <class T>
std::size_t twosided_workspace(Queue& q, const MV<T>& A, int itype) {
    if (itype == 1) return 0;
    auto pool = BumpAllocator::measuring();
    (void)pool.allocate<T>(q, static_cast<std::size_t>(A.rows()) * A.rows() * A.batch_size());
    (void)pool.allocate<T*>(q, static_cast<std::size_t>(A.batch_size()));
    return pool.required_bytes();
}

template <Backend B, class T>
Event twosided(Queue& q, const MV<T>& A_in, const MV<T>& Bf, int itype, Uplo uplo, Span<std::byte> ws) {
    const bool lower = uplo == Uplo::Lower;
    const T one(1);
    // A takes the call's mask even when only B carries it: hermitize must skip the inactive items too.
    const Span<const std::uint8_t> mask = ::batchlas::detail::call_mask("sygst", A_in, Bf);
    const MV<T> A = A_in.with_batch_mask(mask);
    (void)A.hermitize(q, uplo);
    if (itype == 1) {  // Lower: L^-1 A L^-H. Upper: U^-H A U^-1.
        (void)::batchlas::trsm<B, T>(q, Bf, A, one, Side::Left, uplo, lower ? Transpose::NoTrans : kH<T>, Diag::NonUnit);
        return ::batchlas::trsm<B, T>(q, Bf, A, one, Side::Right, uplo, lower ? kH<T> : Transpose::NoTrans, Diag::NonUnit);
    }
    const int n = A.rows(), batch = A.batch_size();
    BumpAllocator pool(ws);
    const Span<T> buf = pool.allocate<T>(q, static_cast<std::size_t>(n) * n * batch);
    const Span<T*> ptr = pool.allocate<T*>(q, static_cast<std::size_t>(batch));
    const MV<T> W = MV<T>(buf.data(), n, n, n, n * n, batch, ptr.data()).with_batch_mask(mask);
    // Lower: W := L^H A, A := W L. Upper: W := U A, A := W U^H.
    (void)::batchlas::trmm<B, T>(q, Bf, A, W, one, Side::Left, uplo, lower ? kH<T> : Transpose::NoTrans, Diag::NonUnit);
    return ::batchlas::trmm<B, T>(q, Bf, W, A, one, Side::Right, uplo, lower ? Transpose::NoTrans : kH<T>, Diag::NonUnit);
}

#define SYGST_TWOSIDED_INSTANTIATE(B_, fp)                                                                    \
    template Event twosided<B_, BATCHLAS_UNPAREN fp>(Queue&, const MV<BATCHLAS_UNPAREN fp>&,                    \
                                                     const MV<BATCHLAS_UNPAREN fp>&, int, Uplo, Span<std::byte>);
BATCHLAS_INSTANTIATE_SCALAR_ALL_BACKENDS(SYGST_TWOSIDED_INSTANTIATE)
#undef SYGST_TWOSIDED_INSTANTIATE

template std::size_t twosided_workspace<float>(Queue&, const MV<float>&, int);
template std::size_t twosided_workspace<double>(Queue&, const MV<double>&, int);
template std::size_t twosided_workspace<std::complex<float>>(Queue&, const MV<std::complex<float>>&, int);
template std::size_t twosided_workspace<std::complex<double>>(Queue&, const MV<std::complex<double>>&, int);

}  // namespace batchlas::ops::sygst
