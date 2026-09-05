// The public factorization entry points, defined once, outside every vendor TU.
//
// Same move, and same reason, as entry_points/level3.cc: geqrf/orgqr/getrf/
// getrs/getri were DEFINED in cublas.cc, netlib_lapack.cc and rocsolver.cc, and
// potrf in cusolver.cc, netlib_lapack.cc and rocsolver.cc -- so dropping a
// vendor library dropped the public entry point along with the vendor path.
//
// Each op moves TOGETHER WITH ITS BUFFER-SIZE QUERY. Splitting them would let
// the two resolve differently, which is the defect class S4d found in ormqr
// (buffer size 2560 bytes, call demanded 276480).

#include <batchlas/backend_config.h>

#include <batchlas/blas/functions/geqrf.hh>
#include <batchlas/blas/functions/orgqr.hh>
#include <batchlas/blas/functions/getrf.hh>
#include <batchlas/blas/functions/getrs.hh>
#include <batchlas/blas/functions/getri.hh>
#include <batchlas/blas/functions/potrf.hh>

#include <batchlas/blas/dispatch/no_route.hh>
#include <batchlas/blas/dispatch/vendor_available.hh>

#include "../../util/template-instantiations.hh"

#include <complex>

namespace batchlas {

template <Backend B, typename T>
Event geqrf(Queue& ctx,
            const MatrixView<T,MatrixFormat::Dense>& A,
            Span<T> tau,
            Span<std::byte> work_space) {
    if constexpr (!dispatch::factorization_vendor_available<B>) {
        dispatch::throw_no_vendor_route<T>(
            dispatch::Op::geqrf, B, dispatch::kFactorizationLibrary<B>);
    } else {
        return backend::geqrf_vendor<B, T>(ctx, A, tau, work_space);
    }
}

template <Backend B, typename T>
size_t geqrf_buffer_size(Queue& ctx,
                         const MatrixView<T,MatrixFormat::Dense>& A,
                         Span<T> tau) {
    if constexpr (!dispatch::factorization_vendor_available<B>) {
        dispatch::throw_no_vendor_route<T>(
            dispatch::Op::geqrf, B, dispatch::kFactorizationLibrary<B>);
    } else {
        return backend::geqrf_vendor_buffer_size<B, T>(ctx, A, tau);
    }
}

template <Backend B, typename T>
Event orgqr(Queue& ctx,
            const MatrixView<T, MatrixFormat::Dense>& A,
            Span<T> tau,
            Span<std::byte> workspace) {
    if constexpr (!dispatch::factorization_vendor_available<B>) {
        dispatch::throw_no_vendor_route<T>(
            dispatch::Op::orgqr, B, dispatch::kFactorizationLibrary<B>);
    } else {
        return backend::orgqr_vendor<B, T>(ctx, A, tau, workspace);
    }
}

template <Backend B, typename T>
size_t orgqr_buffer_size(Queue& ctx,
                         const MatrixView<T, MatrixFormat::Dense>& A,
                         Span<T> tau) {
    if constexpr (!dispatch::factorization_vendor_available<B>) {
        dispatch::throw_no_vendor_route<T>(
            dispatch::Op::orgqr, B, dispatch::kFactorizationLibrary<B>);
    } else {
        return backend::orgqr_vendor_buffer_size<B, T>(ctx, A, tau);
    }
}

template <Backend B, typename T>
Event getrf(Queue& ctx,
            const MatrixView<T, MatrixFormat::Dense>& A,
            Span<int64_t> pivots,
            Span<std::byte> work_space,
            Span<int32_t> info) {
    if constexpr (!dispatch::factorization_vendor_available<B>) {
        dispatch::throw_no_vendor_route<T>(
            dispatch::Op::getrf, B, dispatch::kFactorizationLibrary<B>);
    } else {
        return backend::getrf_vendor<B, T>(ctx, A, pivots, work_space, info);
    }
}

template <Backend B, typename T>
size_t getrf_buffer_size(Queue& ctx,
                         const MatrixView<T, MatrixFormat::Dense>& A) {
    if constexpr (!dispatch::factorization_vendor_available<B>) {
        dispatch::throw_no_vendor_route<T>(
            dispatch::Op::getrf, B, dispatch::kFactorizationLibrary<B>);
    } else {
        return backend::getrf_vendor_buffer_size<B, T>(ctx, A);
    }
}

template <Backend Back, typename T>
Event getrs(Queue& ctx,
            const MatrixView<T,MatrixFormat::Dense>& A,
            const MatrixView<T,MatrixFormat::Dense>& B,
            Transpose transA,
            Span<int64_t> pivots,
            Span<std::byte> work_space) {
    if constexpr (!dispatch::factorization_vendor_available<Back>) {
        dispatch::throw_no_vendor_route<T>(
            dispatch::Op::getrs, Back, dispatch::kFactorizationLibrary<Back>);
    } else {
        return backend::getrs_vendor<Back, T>(ctx, A, B, transA, pivots, work_space);
    }
}

template <Backend Back, typename T>
size_t getrs_buffer_size(Queue& ctx,
                         const MatrixView<T,MatrixFormat::Dense>& A,
                         const MatrixView<T,MatrixFormat::Dense>& B,
                         Transpose transA) {
    if constexpr (!dispatch::factorization_vendor_available<Back>) {
        dispatch::throw_no_vendor_route<T>(
            dispatch::Op::getrs, Back, dispatch::kFactorizationLibrary<Back>);
    } else {
        return backend::getrs_vendor_buffer_size<Back, T>(ctx, A, B, transA);
    }
}

template <Backend B, typename T>
Event getri(Queue& ctx,
            const MatrixView<T, MatrixFormat::Dense>& A,
            const MatrixView<T, MatrixFormat::Dense>& C,
            Span<int64_t> pivots,
            Span<std::byte> work_space,
            Span<int32_t> info) {
    if constexpr (!dispatch::factorization_vendor_available<B>) {
        dispatch::throw_no_vendor_route<T>(
            dispatch::Op::getri, B, dispatch::kFactorizationLibrary<B>);
    } else {
        return backend::getri_vendor<B, T>(ctx, A, C, pivots, work_space, info);
    }
}

template <Backend B, typename T>
size_t getri_buffer_size(Queue& ctx,
                         const MatrixView<T, MatrixFormat::Dense>& A) {
    if constexpr (!dispatch::factorization_vendor_available<B>) {
        dispatch::throw_no_vendor_route<T>(
            dispatch::Op::getri, B, dispatch::kFactorizationLibrary<B>);
    } else {
        return backend::getri_vendor_buffer_size<B, T>(ctx, A);
    }
}

template <Backend B, typename T>
Event potrf(Queue& ctx,
                const MatrixView<T, MatrixFormat::Dense>& descrA,
                Uplo uplo,
                Span<std::byte> workspace,
                Span<int32_t> info_out) {
    if constexpr (!dispatch::solver_vendor_available<B>) {
        dispatch::throw_no_vendor_route<T>(
            dispatch::Op::potrf, B, dispatch::kSolverLibrary<B>);
    } else {
        return backend::potrf_vendor<B, T>(ctx, descrA, uplo, workspace, info_out);
    }
}

template <Backend B, typename T>
size_t potrf_buffer_size(Queue& ctx,
                        const MatrixView<T,MatrixFormat::Dense>& A,
                        Uplo uplo) {
    if constexpr (!dispatch::solver_vendor_available<B>) {
        dispatch::throw_no_vendor_route<T>(
            dispatch::Op::potrf, B, dispatch::kSolverLibrary<B>);
    } else {
        return backend::potrf_vendor_buffer_size<B, T>(ctx, A, uplo);
    }
}

// ---------------------------------------------------------------------------
// Explicit instantiations, one block per backend whose vendor TU is compiled.
// ---------------------------------------------------------------------------

#define OP_INSTANTIATE(OP, B_, fp) BATCHLAS_INSTANTIATE(sig::OP<fp>, OP, B_, fp)

#define FACTORIZATION_ONE(B_, fp)              \
    OP_INSTANTIATE(geqrf, B_, fp)              \
    OP_INSTANTIATE(geqrf_buffer_size, B_, fp)  \
    OP_INSTANTIATE(orgqr, B_, fp)              \
    OP_INSTANTIATE(orgqr_buffer_size, B_, fp)  \
    OP_INSTANTIATE(getrf, B_, fp)              \
    OP_INSTANTIATE(getrf_buffer_size, B_, fp)  \
    OP_INSTANTIATE(getrs, B_, fp)              \
    OP_INSTANTIATE(getrs_buffer_size, B_, fp)  \
    OP_INSTANTIATE(getri, B_, fp)              \
    OP_INSTANTIATE(getri_buffer_size, B_, fp)

#define FACTORIZATION_ALL(B_)                       \
    FACTORIZATION_ONE(B_, float)                    \
    FACTORIZATION_ONE(B_, double)                   \
    FACTORIZATION_ONE(B_, std::complex<float>)      \
    FACTORIZATION_ONE(B_, std::complex<double>)

#define POTRF_ALL(B_)                               \
    OP_INSTANTIATE(potrf, B_, float)                \
    OP_INSTANTIATE(potrf, B_, double)               \
    OP_INSTANTIATE(potrf, B_, std::complex<float>)  \
    OP_INSTANTIATE(potrf, B_, std::complex<double>) \
    OP_INSTANTIATE(potrf_buffer_size, B_, float)                \
    OP_INSTANTIATE(potrf_buffer_size, B_, double)               \
    OP_INSTANTIATE(potrf_buffer_size, B_, std::complex<float>)  \
    OP_INSTANTIATE(potrf_buffer_size, B_, std::complex<double>)

// geqrf/orgqr/getrf/getrs/getri come from cuBLAS on NVIDIA; potrf from cuSOLVER.
// Keyed on the DEVICE FAMILY, not on the vendor library. The bodies above
// compile to a throw when the library is absent, so the public entry point
// exists as a symbol in every build that has the device -- which is exactly what
// stopped being true when the definitions lived in the vendor TUs.
#if BATCHLAS_HAS_CUDA_BACKEND
FACTORIZATION_ALL(Backend::CUDA)
POTRF_ALL(Backend::CUDA)
#endif

// On ROCm all of them come from rocSOLVER.
#if BATCHLAS_HAS_ROCM_BACKEND
FACTORIZATION_ALL(Backend::ROCM)
POTRF_ALL(Backend::ROCM)
#endif

#if BATCHLAS_HAS_HOST_BACKEND
FACTORIZATION_ALL(Backend::NETLIB)
POTRF_ALL(Backend::NETLIB)
#endif

#undef POTRF_ALL
#undef FACTORIZATION_ALL
#undef FACTORIZATION_ONE
#undef OP_INSTANTIATE

}  // namespace batchlas
