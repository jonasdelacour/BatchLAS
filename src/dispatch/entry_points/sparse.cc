// The public sparse entry points, defined once, outside every vendor TU.
//
// Same move and same reason as entry_points/level3.cc: spmm was DEFINED in
// cusparse.cc, netlib_lapack.cc and rocsparse.cc, so dropping a vendor library
// dropped the public entry point along with the vendor path.
//
// spmm carries a third template parameter (MatrixFormat), which is why its
// instantiations are spelled out here rather than going through the shared
// per-type macros the dense facades use.

#include <batchlas/backend_config.h>

#include <batchlas/blas/functions/spmm.hh>

#include <batchlas/blas/dispatch/no_route.hh>
#include <batchlas/blas/dispatch/vendor_available.hh>

#include "../../util/template-instantiations.hh"

#include <complex>

namespace batchlas {

template <Backend B, typename T, MatrixFormat MFormat>
Event spmm(Queue& ctx,
           const MatrixView<T, MFormat>& A,
           const MatrixView<T, MatrixFormat::Dense>& B_mat,
           const MatrixView<T, MatrixFormat::Dense>& C,
           T alpha,
           T beta,
           Transpose transA,
           Transpose transB,
           Span<std::byte> workspace) {
    if constexpr (!dispatch::sparse_vendor_available<B>) {
        dispatch::throw_no_vendor_route<T>(
            dispatch::Op::spmm, B, dispatch::kSparseLibrary<B>);
    } else {
        return backend::spmm_vendor<B, T, MFormat>(ctx, A, B_mat, C, alpha, beta, transA, transB, workspace);
    }
}

template <Backend B, typename T, MatrixFormat MFormat>
size_t spmm_buffer_size(Queue& ctx,
                        const MatrixView<T, MFormat>& A,
                        const MatrixView<T, MatrixFormat::Dense>& B_mat,
                        const MatrixView<T, MatrixFormat::Dense>& C,
                        T alpha,
                        T beta,
                        Transpose transA,
                        Transpose transB) {
    if constexpr (!dispatch::sparse_vendor_available<B>) {
        dispatch::throw_no_vendor_route<T>(
            dispatch::Op::spmm, B, dispatch::kSparseLibrary<B>);
    } else {
        return backend::spmm_vendor_buffer_size<B, T, MFormat>(ctx, A, B_mat, C, alpha, beta, transA, transB);
    }
}

// ---------------------------------------------------------------------------
// Explicit instantiations, one block per backend whose vendor TU is compiled.
// ---------------------------------------------------------------------------

#define SPMM_ONE(B_, fp, F)                                            \
    BATCHLAS_INSTANTIATE(sig::spmm<fp BATCHLAS_COMMA F>, spmm, B_, fp, F) \
    BATCHLAS_INSTANTIATE(sig::spmm_buffer_size<fp BATCHLAS_COMMA F>, spmm_buffer_size, B_, fp, F)

// CSR is the only sparse format any backend instantiates today.
#define SPMM_ALL(B_)                                    \
    SPMM_ONE(B_, float, MatrixFormat::CSR)              \
    SPMM_ONE(B_, double, MatrixFormat::CSR)             \
    SPMM_ONE(B_, std::complex<float>, MatrixFormat::CSR)\
    SPMM_ONE(B_, std::complex<double>, MatrixFormat::CSR)

// Keyed on the DEVICE FAMILY, not on the vendor library. The bodies above
// compile to a throw when the library is absent, so the public entry point
// exists as a symbol in every build that has the device -- which is exactly what
// stopped being true when the definitions lived in the vendor TUs.
#if BATCHLAS_HAS_CUDA_BACKEND
SPMM_ALL(Backend::CUDA)
#endif

#if BATCHLAS_HAS_ROCM_BACKEND
SPMM_ALL(Backend::ROCM)
#endif

#if BATCHLAS_HAS_HOST_BACKEND
SPMM_ALL(Backend::NETLIB)
#endif

#undef SPMM_ALL
#undef SPMM_ONE

}  // namespace batchlas
