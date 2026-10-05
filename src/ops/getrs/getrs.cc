// getrs: the whole selection path (docs/design/flat-kernel-selection.md §4.3, rule R1).
// public getrs() -> choose() -> std::visit -> launch.
// The kernel for a shape is the first runnable entry of the nearest row in
// tuned/getrs.<dtype>.<device>.txt; can_run() below only removes entries that cannot run.
// Cta permutes and solves both triangles in one kernel with the right-hand sides resident in
// local memory; Blocked applies the interchanges and calls the public trsm twice.

#include <batchlas/backend_config.h>

#include <batchlas/blas/functions/getrs.hh>
#include <batchlas/blas/functions/trsm.hh>
#include <batchlas/blas/dispatch/no_route.hh>
#include <batchlas/blas/dispatch/vendor_available.hh>

#include "choice.hh"
#include "../../select/select.hh"
#include "../../extensions/getrs_native.hh"
#include "../../util/template-instantiations.hh"

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <variant>

namespace batchlas {
namespace ops::getrs {

template <class... F>
struct overloaded : F... { using F::operator()...; };
template <class... F>
overloaded(F...) -> overloaded<F...>;

template <class T>
using MV = MatrixView<T, MatrixFormat::Dense>;

template <class T>
select::Key key_of(const MV<T>& A, const MV<T>& B) {
    return {{"n", A.rows()}, {"nrhs", B.cols()}, {"batch", A.batch_size()}};
}

// getrs's vendor is the factorization library (cuBLAS + cuSOLVER on CUDA), carried in
// Device::has_vendor_solver for this op.
template <Backend B>
const select::Device& device_for(Queue& q) {
    return select::device_of<B>(q, dispatch::factorization_vendor_available<B>);
}

// Correctness only (R3): false means the driver would throw or answer wrongly. Both natives
// re-check exactly this at their entry. A non-conforming pair (A not square, rows or batch
// disagreeing) is left to the vendor, as before. NETLIB writes true int64 pivots where the
// natives read packed 1-based int32, a silent wrong answer, so no native runs under it.
template <Backend B, class T>
bool can_run(const GetrsChoice& c, const select::Device& d, const MV<T>& A, const MV<T>& Bm) {
    const std::int64_t n = A.rows(), nrhs = Bm.cols();
    const bool conforming = A.cols() == n && Bm.rows() == n && Bm.batch_size() == A.batch_size();
    const bool native = B != Backend::NETLIB && d.is_gpu && d.has_sg32 && conforming && !A.is_heterogeneous() &&
                        !Bm.is_heterogeneous() && n >= 1 && nrhs >= 1 && A.batch_size() >= 1;
    return std::visit(overloaded{
        [&](Cta) {
            // The fused work-group never drops below 32 lanes (getrs_fused_wg).
            const auto cap = sycl_getrs::getrs_fused_max_rhs_elems<T>(static_cast<std::size_t>(d.slm_budget));
            return native && sycl_getrs::getrs_fused_available<T>() && d.max_wg >= 32 &&
                   nrhs <= sycl_getrs::kGetrsFusedMaxRhs &&
                   static_cast<std::size_t>(n) * static_cast<std::size_t>(nrhs) <= cap;
        },
        [&](Blocked) { return native && sycl_getrs::getrs_blocked_available<T>(); },
        [&](Vendor) { return d.has_vendor_solver; },
    }, c);
}

template <Backend B, class T>
GetrsChoice choose(Queue& q, const MV<T>& A, const MV<T>& Bm) {
    const select::Device& d = device_for<B>(q);
    auto ok = [&](const GetrsChoice& c) { return can_run<B, T>(c, d, A, Bm); };
    try {
        return select::choose("getrs", select::dtype_name<T>(), d, key_of<T>(A, Bm), candidates<T>(), ok, rules);
    } catch (const std::runtime_error&) {
        // Vendor-free burn-down reads coverage `miss` rows; a plain runtime_error records none.
        const auto all = candidates<T>();
        if (!dispatch::factorization_vendor_available<B> && std::none_of(all.begin(), all.end(), ok))
            dispatch::throw_no_vendor_route<T>(dispatch::Op::getrs, B, dispatch::kFactorizationLibrary<B>);
        throw;
    }
}

// The coverage row's native flags (§5.6): computed only when coverage records a row.
template <Backend B, class T>
select::NativeFacts native_facts(Queue& q, const MV<T>& A, const MV<T>& Bm) {
    if (!dispatch::coverage::dynamic_enabled()) return {};
    const select::Device& d = device_for<B>(q);
    return select::native_facts(candidates<T>(), [&](const GetrsChoice& c) { return can_run<B, T>(c, d, A, Bm); });
}

template <Backend B, class T>
Event launch(Queue& q, const GetrsChoice& c, const MV<T>& A, const MV<T>& Bm, Transpose transA,
             Span<std::int64_t> pivots, Span<std::byte> ws) {
    return std::visit(overloaded{
        [&](Cta) { return sycl_getrs::getrs_fused_dispatch<T>(q, A, Bm, transA, pivots, ws); },
        [&](Blocked) {
            return sycl_getrs::getrs_blocked_dispatch<T>(
                q, A, Bm, transA, pivots, ws,
                [](Queue& c2, const MV<T>& ta, const MV<T>& tb, T talpha, Side tside, Uplo tuplo, Transpose ttrans,
                   Diag tdiag) { return trsm<B, T>(c2, ta, tb, talpha, tside, tuplo, ttrans, tdiag); });
        },
        [&](Vendor) -> Event {
            if constexpr (dispatch::factorization_vendor_available<B>)
                return backend::getrs_vendor<B, T>(q, A, Bm, transA, pivots, ws);
            else
                dispatch::throw_no_vendor_route<T>(dispatch::Op::getrs, B, dispatch::kFactorizationLibrary<B>);
        },
    }, c);
}

// Exactly the chosen family's need (R5). Nothing here dereferences A or B: a measuring pass
// hands null data pointers.
template <Backend B, class T>
std::size_t workspace(Queue& q, const GetrsChoice& c, const MV<T>& A, const MV<T>& Bm, Transpose transA) {
    return std::visit(overloaded{
        [&](Cta) { return sycl_getrs::getrs_fused_buffer_size<T>(q, A, Bm, transA); },
        [&](Blocked) { return sycl_getrs::getrs_blocked_buffer_size<T>(q, A, Bm, transA); },
        [&](Vendor) -> std::size_t {
            if constexpr (dispatch::factorization_vendor_available<B>)
                return backend::getrs_vendor_buffer_size<B, T>(q, A, Bm, transA);
            else
                dispatch::throw_no_vendor_route<T>(dispatch::Op::getrs, B, dispatch::kFactorizationLibrary<B>);
        },
    }, c);
}

}  // namespace ops::getrs

template <Backend Back, typename T>
Event getrs(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, const MatrixView<T, MatrixFormat::Dense>& B,
            Transpose transA, Span<int64_t> pivots, Span<std::byte> work_space) {
    getrs_validate_params<T>(A, B);
    const auto c = ops::getrs::choose<Back, T>(ctx, A, B);
    // The coverage row's key: m = k = order, n = nrhs, and transA (the field separating rows).
    auto shape = select::square_shape<Back, T>(A.rows(), A.batch_size());
    shape.n = B.cols();
    shape.transA = transA;
    const select::Key trace_key = ops::getrs::key_of<T>(A, B);
    select::TraceScope trace("getrs", c, shape, ops::getrs::native_facts<Back, T>(ctx, A, B), trace_key);
    return ops::getrs::launch<Back, T>(ctx, c, A, B, transA, pivots, work_space);
}

template <Backend Back, typename T>
size_t getrs_buffer_size(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A,
                         const MatrixView<T, MatrixFormat::Dense>& B, Transpose transA) {
    getrs_validate_params<T>(A, B);
    return ops::getrs::workspace<Back, T>(ctx, ops::getrs::choose<Back, T>(ctx, A, B), A, B, transA);
}

#define GETRS_INSTANTIATE(B_, fp)                                   \
    BATCHLAS_INSTANTIATE(sig::getrs<fp>, getrs, B_, fp)             \
    BATCHLAS_INSTANTIATE(sig::getrs_buffer_size<fp>, getrs_buffer_size, B_, fp)

#define GETRS_ALL(B_)                          \
    GETRS_INSTANTIATE(B_, float)               \
    GETRS_INSTANTIATE(B_, double)              \
    GETRS_INSTANTIATE(B_, std::complex<float>) \
    GETRS_INSTANTIATE(B_, std::complex<double>)

// Keyed on the device family, not the vendor library: without the library the Vendor arm
// compiles to a throw, so the symbol exists in every build with the device.
#if BATCHLAS_HAS_CUDA_BACKEND
GETRS_ALL(Backend::CUDA)
#endif
#if BATCHLAS_HAS_ROCM_BACKEND
GETRS_ALL(Backend::ROCM)
#endif
#if BATCHLAS_HAS_HOST_BACKEND
GETRS_ALL(Backend::NETLIB)
#endif

#undef GETRS_ALL
#undef GETRS_INSTANTIATE

}  // namespace batchlas
