// getri: the whole selection path (docs/design/flat-kernel-selection.md §4.3, rule R1).
// public getri() -> choose() -> std::visit -> launch. The kernel for a shape is the first
// runnable entry of the nearest row in tuned/getri.<dtype>.<device>.txt; can_run() below only
// removes entries that cannot run. Blocked writes P into C and solves with two public trsm calls
// (each picks its own kernel); Vendor is cublas<t>getriBatched / rocsolver / LAPACKE.
//
// getri_buffer_size has no C, so the key and can_run read A alone and both entry points reach the
// same choice (R5). Sizing runs under BumpAllocator::measuring() (inv.cc): nothing here touches
// the workspace or dereferences A.data_ptr().

#include <batchlas/backend_config.h>

#include <batchlas/blas/functions/getri.hh>
#include <batchlas/blas/functions/trsm.hh>
#include <batchlas/no_route.hh>
#include "../../select/vendor.hh"

#include "choice.hh"
#include "../../select/select.hh"
#include "../../extensions/getri_native.hh"
#include "../../util/template-instantiations.hh"

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <variant>

namespace batchlas {
namespace ops::getri {

template <class... F>
struct overloaded : F... { using F::operator()...; };
template <class... F>
overloaded(F...) -> overloaded<F...>;

template <class T>
using MV = MatrixView<T, MatrixFormat::Dense>;

template <class T>
select::Key key_of(const MV<T>& A) {
    return {{"n", A.rows()}, {"batch", A.batch_size()}};
}

// Correctness only (R3): false means the driver would throw or answer wrongly. Blocked's clauses
// are getri_blocked_dispatch's own checks on A (C, the pivot span and aliasing are argument
// errors the driver reports itself), plus the pivot format: GPU backends pack 1-based int32 into
// the int64 span and netlib writes genuine int64, so a NETLIB backend on a GPU queue would read
// netlib's pivots wrongly. evidence: docs/perf/lu.md#correctness-findings
// Vendor: exactly the launch's own guard. Not d.has_vendor_solver && d.has_vendor_blas: on ROCm
// rocSOLVER and rocBLAS are separate options, and getri_vendor<ROCM> needs only rocSOLVER.
template <Backend B, class T>
bool can_run(const GetriChoice& c, const select::Device& d, const MV<T>& A) {
    return std::visit(overloaded{
        [&](Blocked) {
            return d.is_gpu && d.has_sg32 && B != Backend::NETLIB && sycl_getri::getri_blocked_available<T>() &&
                   A.rows() == A.cols() && A.rows() >= 1 && A.batch_size() >= 1 && !A.is_heterogeneous();
        },
        [&](Vendor) { return select::factorization_vendor_available<B>; },
    }, c);
}

template <Backend B, class T>
GetriChoice choose(Queue& q, const MV<T>& A) {
    const select::Device& d = select::device_of<B>(q);
    auto ok = [&](const GetriChoice& c) { return can_run<B, T>(c, d, A); };
    try {
        return select::choose("getri", select::dtype_name<T>(), d, key_of<T>(A), candidates<T>(), ok, rules);
    } catch (const std::runtime_error&) {
        // Vendor-free burn-down reads coverage `miss` rows; a plain runtime_error records none.
        const auto all = candidates<T>();
        if (!select::factorization_vendor_available<B> && std::none_of(all.begin(), all.end(), ok))
            select::throw_no_vendor_route<T>(Op::getri, B, select::kFactorizationLibrary<B>);
        throw;
    }
}

// The coverage row's native flags (§5.6): computed only when coverage records a row.
template <Backend B, class T>
select::NativeFacts native_facts(Queue& q, const MV<T>& A) {
    if (!coverage::dynamic_enabled()) return {};
    const select::Device& d = select::device_of<B>(q);
    return select::native_facts(candidates<T>(), [&](const GetriChoice& c) { return can_run<B, T>(c, d, A); });
}

template <Backend B, class T>
Event launch(Queue& q, const GetriChoice& c, const MV<T>& A, const MV<T>& C, Span<int64_t> pivots,
             Span<std::byte> ws, Span<int32_t> info) {
    return std::visit(overloaded{
        [&](Blocked) {
            return sycl_getri::getri_blocked_dispatch<T>(
                q, A, C, pivots, ws, info,
                [](Queue& c2, const MV<T>& ta, const MV<T>& tb, T talpha, Side tside, Uplo tuplo, Transpose ttrans,
                   Diag tdiag) { return trsm<B, T>(c2, ta, tb, talpha, tside, tuplo, ttrans, tdiag); });
        },
        [&](Vendor) -> Event {
            if constexpr (select::factorization_vendor_available<B>)
                return backend::getri_vendor<B, T>(q, A, C, pivots, ws, info);
            else
                select::throw_no_vendor_route<T>(Op::getri, B, select::kFactorizationLibrary<B>);
        },
    }, c);
}

// Exactly the chosen family's need (R5); Blocked's trsm calls take no workspace.
template <Backend B, class T>
std::size_t workspace(Queue& q, const GetriChoice& c, const MV<T>& A) {
    return std::visit(overloaded{
        [&](Blocked) { return sycl_getri::getri_blocked_buffer_size<T>(q, A); },
        [&](Vendor) -> std::size_t {
            if constexpr (select::factorization_vendor_available<B>)
                return backend::getri_vendor_buffer_size<B, T>(q, A);
            else
                select::throw_no_vendor_route<T>(Op::getri, B, select::kFactorizationLibrary<B>);
        },
    }, c);
}

}  // namespace ops::getri

template <Backend Back, typename T>
Event getri(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, const MatrixView<T, MatrixFormat::Dense>& C,
            Span<int64_t> pivots, Span<std::byte> work_space, Span<int32_t> info) {
    getri_validate_params<T>(A, C);
    const auto c = ops::getri::choose<Back, T>(ctx, A);
    // The coverage row's key: m = k = order, n = A.cols (equal unless the vendor takes a non-square A).
    auto shape = select::square_shape<Back, T>(A.rows(), A.batch_size());
    shape.n = A.cols();
    select::TraceScope trace("getri", c, shape, ops::getri::native_facts<Back, T>(ctx, A), ops::getri::key_of<T>(A));
    return ops::getri::launch<Back, T>(ctx, c, A, C, pivots, work_space, info);
}

template <Backend Back, typename T>
size_t getri_buffer_size(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A) {
    getri_validate_params<T>(A);
    return ops::getri::workspace<Back, T>(ctx, ops::getri::choose<Back, T>(ctx, A), A);
}

#define GETRI_INSTANTIATE(B_, fp)                      \
    BATCHLAS_INSTANTIATE(sig::getri<fp>, getri, B_, fp) \
    BATCHLAS_INSTANTIATE(sig::getri_buffer_size<fp>, getri_buffer_size, B_, fp)

#define GETRI_ALL(B_)                          \
    GETRI_INSTANTIATE(B_, float)               \
    GETRI_INSTANTIATE(B_, double)              \
    GETRI_INSTANTIATE(B_, std::complex<float>) \
    GETRI_INSTANTIATE(B_, std::complex<double>)

// Keyed on the device family, not the vendor library: without the library the Vendor arm
// compiles to a throw, so the symbol exists in every build with the device.
#if BATCHLAS_HAS_CUDA_BACKEND
GETRI_ALL(Backend::CUDA)
#endif
#if BATCHLAS_HAS_ROCM_BACKEND
GETRI_ALL(Backend::ROCM)
#endif
#if BATCHLAS_HAS_HOST_BACKEND
GETRI_ALL(Backend::NETLIB)
#endif

#undef GETRI_ALL
#undef GETRI_INSTANTIATE

}  // namespace batchlas
