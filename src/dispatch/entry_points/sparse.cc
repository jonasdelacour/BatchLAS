// The public sparse entry points, defined once outside every vendor TU so that
// dropping a vendor library cannot drop the public entry point with it.
// See docs/design/vendor-independence.md#the-entry-point-facade.

#include <batchlas/backend_config.h>

#include <batchlas/blas/functions/spmm.hh>

#include <batchlas/blas/dispatch/no_route.hh>
#include <batchlas/blas/dispatch/vendor_available.hh>

#include <batchlas/blas/dispatch/route_spmm.hh>

#include "../../backends/spmm_route.hh"
#include "../../sycl/spmm_native.hh"

#include "../../util/template-instantiations.hh"

#include <algorithm>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace batchlas {

// Throws rather than falling through to the vendor: a fall-through would
// silently take the vendor the moment a native capability comes off zero.
template <typename T>
[[noreturn]] inline void spmm_throw_native_unimplemented(dispatch::Route route,
                                                         const char* who) {
    throw std::logic_error(
        std::string(who) + ": resolved to a native route (" +
        std::string(dispatch::to_string(route.origin)) + ":" +
        std::string(dispatch::to_string(route.algo)) +
        ") but no native spmm kernel is linked into this build. "
        "sycl_spmm::spmm_gather_available / spmm_scatter_available reported a "
        "capability the facade cannot service.");
}

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
    // Deliberately no validate_params here: the native kernel must accept
    // exactly what the vendor does, so bad shapes resolve to the vendor.
    const dispatch::Route route = backend::spmm_route<B, T, MFormat>(
        ctx, A, B_mat, C, transA, transB,
        /*vendor_available=*/dispatch::sparse_vendor_available<B>);

    // preferred() is false for every spmm route, shape and type: forced only.
    // evidence: docs/perf/spmm.md#the-preferred-window-as-implemented
    if (dispatch::is_native(route)) {
        // supports() refuses every non-CSR format, forced routes included.
        if constexpr (MFormat == MatrixFormat::CSR) {
            if (route.algo == dispatch::Algorithm::Direct) {
                return sycl_spmm::spmm_native_csr<T>(ctx, A, B_mat, C, alpha,
                                                     beta, transA, transB);
            }
        }
        spmm_throw_native_unimplemented<T>(route, "spmm");
    }

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
    // Must resolve to the same Route as spmm(), or the allocation undershoots.
    const dispatch::Route route = backend::spmm_route<B, T, MFormat>(
        ctx, A, B_mat, C, transA, transB,
        /*vendor_available=*/dispatch::sparse_vendor_available<B>);

    // max() over every supported native tier, not the resolved one, so a
    // query/call disagreement over- rather than under-allocates. `native_fired`
    // cannot be `native_need != 0`: the native need is exactly zero. Nothing
    // here may touch device memory: row_offsets()/nnz() are not host-reachable.
    std::size_t native_need = 0;
    bool native_fired = false;
    if (dispatch::is_native(route)) {
        if constexpr (MFormat == MatrixFormat::CSR) {
            const auto shape = backend::spmm_op_shape<B, T, MFormat>(
                ctx, A, B_mat, C, transA, transB);
            using Tbl = dispatch::RouteTable<dispatch::Op::spmm, T>;
            if (shape && Tbl::supports({dispatch::Origin::Native,
                                        dispatch::Algorithm::Direct}, *shape)) {
                constexpr std::size_t kSpmmNativeDirectNeed = 0;
                native_need = std::max(native_need, kSpmmNativeDirectNeed);
                native_fired = true;
            }
        }
        if (!native_fired) {
            spmm_throw_native_unimplemented<T>(route, "spmm_buffer_size");
        }
    }

    if constexpr (!dispatch::sparse_vendor_available<B>) {
        if (!native_fired) {
            dispatch::throw_no_vendor_route<T>(
                dispatch::Op::spmm, B, dispatch::kSparseLibrary<B>);
        }
        return native_need;
    } else {
        return std::max(native_need,
                        backend::spmm_vendor_buffer_size<B, T, MFormat>(ctx, A, B_mat, C, alpha, beta, transA, transB));
    }
}

#define SPMM_ONE(B_, fp, F)                                            \
    BATCHLAS_INSTANTIATE(sig::spmm<fp BATCHLAS_COMMA F>, spmm, B_, fp, F) \
    BATCHLAS_INSTANTIATE(sig::spmm_buffer_size<fp BATCHLAS_COMMA F>, spmm_buffer_size, B_, fp, F)

#define SPMM_ALL(B_)                                    \
    SPMM_ONE(B_, float, MatrixFormat::CSR)              \
    SPMM_ONE(B_, double, MatrixFormat::CSR)             \
    SPMM_ONE(B_, std::complex<float>, MatrixFormat::CSR)\
    SPMM_ONE(B_, std::complex<double>, MatrixFormat::CSR)

// Keyed on the device family, not the vendor library: the bodies above compile
// to a throw when the library is absent, so the symbol exists in every build.
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
