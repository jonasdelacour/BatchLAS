// geqrf: the whole selection path (docs/design/flat-kernel-selection.md §4.3, rule R1;
// docs/design/flat-select-p5/geqrf.md). public geqrf() -> choose() -> std::visit -> launch.
// The kernel for a shape is the first runnable entry of the nearest row in
// tuned/geqrf.<dtype>.<device>.txt; can_run() (can_run.hh) only removes entries that cannot run.
// Tiny factors a square n <= 32 in registers, Cta holds the whole panel in local memory, and
// Blocked factors nb-wide panels with Cta's device function and updates the trailing matrix
// with the public gemm, which picks its own kernel.

#include <batchlas/backend_config.h>

#include <batchlas/blas/functions/gemm.hh>
#include <batchlas/blas/functions/geqrf.hh>
#include <batchlas/blas/dispatch/no_route.hh>
#include <batchlas/blas/dispatch/vendor_available.hh>

#include "can_run.hh"
#include "choice.hh"
#include "geqrf.hh"
#include "../../select/select.hh"
#include "../../extensions/geqrf_native.hh"
#include "../../util/template-instantiations.hh"

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <variant>

namespace batchlas {
namespace ops::geqrf {

template <class T>
select::Key key_of(const MV<T>& A) {
    const std::int64_t m = A.rows(), n = A.cols();
    return {{"form", m == n ? "sq" : (m > n ? "tall" : "wide")}, {"n", n}, {"aspect", aspect_of(m, n)}};
}

// geqrf's vendor group is cuBLAS + cuSOLVER on CUDA (factorization_vendor_available), carried in
// has_vendor_solver; select's default for that flag is cuSOLVER alone.
template <Backend B>
const select::Device& device(Queue& q) {
    return select::device_of<B>(q, dispatch::factorization_vendor_available<B>);
}

template <Backend B, class T>
GeqrfChoice choose(Queue& q, const MV<T>& A) {
    const select::Device& d = device<B>(q);
    auto ok = [&](const GeqrfChoice& c) { return can_run<T>(c, d, A); };
    try {
        return select::choose("geqrf", select::dtype_name<T>(), d, key_of<T>(A), candidates<T>(), ok, rules);
    } catch (const std::runtime_error&) {
        // Vendor-free burn-down reads coverage `miss` rows; a plain runtime_error records none.
        const auto all = candidates<T>();
        if (!dispatch::factorization_vendor_available<B> && std::none_of(all.begin(), all.end(), ok))
            dispatch::throw_no_vendor_route<T>(dispatch::Op::geqrf, B, dispatch::kFactorizationLibrary<B>);
        throw;
    }
}

// The coverage row's native flags (§5.6): computed only when coverage records a row.
template <Backend B, class T>
select::NativeFacts native_facts(Queue& q, const MV<T>& A) {
    if (!dispatch::coverage::dynamic_enabled()) return {};
    const select::Device& d = device<B>(q);
    return select::native_facts(candidates<T>(), [&](const GeqrfChoice& c) { return can_run<T>(c, d, A); });
}

template <Backend B, class T>
Event launch(Queue& q, const GeqrfChoice& c, const MV<T>& A, Span<T> tau, Span<std::byte> ws) {
    return std::visit(overloaded{
        [&](Tiny) { return sycl_geqrf::geqrf_tiny_dispatch<T>(q, A, tau, ws); },
        [&](Cta) { return sycl_geqrf::geqrf_cta_dispatch<T>(q, A, tau, ws); },
        [&](Blocked) {
            // The public gemm, not gemm_custom: the trailing updates are TN/NN panel products the
            // native GEMM loses on some shapes, and the gemm table knows which.
            return sycl_geqrf::geqrf_blocked_dispatch<T>(
                q, A, tau, ws,
                [](Queue& c2, const MV<T>& ga, const MV<T>& gb, const MV<T>& gc, T galpha, T gbeta, Transpose gta,
                   Transpose gtb, ComputePrecision gp) {
                    return gemm<B, T>(c2, ga, gb, gc, galpha, gbeta, gta, gtb, gp);
                });
        },
        [&](Vendor) -> Event {
            if constexpr (dispatch::factorization_vendor_available<B>)
                return backend::geqrf_vendor<B, T>(q, A, tau, ws);
            else
                dispatch::throw_no_vendor_route<T>(dispatch::Op::geqrf, B, dispatch::kFactorizationLibrary<B>);
        },
    }, c);
}

// Exactly the chosen family's need (R5). Never dereferences A or tau: band_reduction.cc sizes
// through a null-data view (see geqrf_buffer_size_bound).
template <Backend B, class T>
std::size_t workspace(Queue& q, const GeqrfChoice& c, const MV<T>& A, Span<T> tau) {
    return std::visit(overloaded{
        [&](Tiny) { return sycl_geqrf::geqrf_tiny_buffer_size<T>(q, A); },
        [&](Cta) { return sycl_geqrf::geqrf_cta_buffer_size<T>(q, A); },
        [&](Blocked) { return sycl_geqrf::geqrf_blocked_buffer_size<T>(q, A); },
        [&](Vendor) -> std::size_t {
            if constexpr (dispatch::factorization_vendor_available<B>)
                return backend::geqrf_vendor_buffer_size<B, T>(q, A, tau);
            else
                dispatch::throw_no_vendor_route<T>(dispatch::Op::geqrf, B, dispatch::kFactorizationLibrary<B>);
        },
    }, c);
}

}  // namespace ops::geqrf

template <Backend Back, typename T>
Event geqrf(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, Span<T> tau, Span<std::byte> work_space) {
    geqrf_validate_params<T>(A);
    const auto c = ops::geqrf::choose<Back, T>(ctx, A);
    // The coverage row's key: m, n and k = min(m, n), as the old shape builder set them.
    auto shape = select::square_shape<Back, T>(A.cols(), A.batch_size());
    shape.m = A.rows();
    shape.k = std::min(A.rows(), A.cols());
    const select::Key trace_key = ops::geqrf::key_of<T>(A);
    select::TraceScope trace("geqrf", c, shape, ops::geqrf::native_facts<Back, T>(ctx, A), trace_key);
    return ops::geqrf::launch<Back, T>(ctx, c, A, tau, work_space);
}

template <Backend Back, typename T>
size_t geqrf_buffer_size(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, Span<T> tau) {
    geqrf_validate_params<T>(A);
    return ops::geqrf::workspace<Back, T>(ctx, ops::geqrf::choose<Back, T>(ctx, A), A, tau);
}

template <Backend Back, typename T>
std::size_t geqrf_buffer_size_bound(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, Span<T> tau) {
    geqrf_validate_params<T>(A);
    const select::Device& d = ops::geqrf::device<Back>(ctx);
    std::size_t need = 0;
    for (const auto& c : ops::geqrf::candidates<T>()) {
        const bool vendor = select::family_of(c) == "vendor";
        if (vendor ? d.has_vendor_solver : d.is_gpu)
            need = std::max(need, ops::geqrf::workspace<Back, T>(ctx, c, A, tau));
    }
    return need;
}

#define GEQRF_INSTANTIATE(B_, fp)                                               \
    BATCHLAS_INSTANTIATE(sig::geqrf<fp>, geqrf, B_, fp)                         \
    BATCHLAS_INSTANTIATE(sig::geqrf_buffer_size<fp>, geqrf_buffer_size, B_, fp) \
    template std::size_t geqrf_buffer_size_bound<B_, fp>(Queue&, const MatrixView<fp, MatrixFormat::Dense>&, Span<fp>);

#define GEQRF_ALL(B_)                          \
    GEQRF_INSTANTIATE(B_, float)               \
    GEQRF_INSTANTIATE(B_, double)              \
    GEQRF_INSTANTIATE(B_, std::complex<float>) \
    GEQRF_INSTANTIATE(B_, std::complex<double>)

// Keyed on the device family, not the vendor library: without the library the Vendor arm
// compiles to a throw, so the symbol exists in every build with the device.
#if BATCHLAS_HAS_CUDA_BACKEND
GEQRF_ALL(Backend::CUDA)
#endif
#if BATCHLAS_HAS_ROCM_BACKEND
GEQRF_ALL(Backend::ROCM)
#endif
#if BATCHLAS_HAS_HOST_BACKEND
GEQRF_ALL(Backend::NETLIB)
#endif

#undef GEQRF_ALL
#undef GEQRF_INSTANTIATE

}  // namespace batchlas
