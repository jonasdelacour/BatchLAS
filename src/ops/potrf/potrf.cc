// potrf: the whole selection path (docs/design/flat-kernel-selection.md §4.3, rule R1).
// public potrf() -> choose() -> std::visit -> launch. Which kernel runs for a shape is the
// first runnable entry of the nearest row in tuned/potrf.<dtype>.<device>.txt; can_run()
// below only removes entries that cannot run at all. Kernel bodies live in
// src/extensions/potrf_{tiny,cta,lpanel,blocked}.cc.

#include <batchlas/backend_config.h>

#include <batchlas/blas/functions/gemm.hh>
#include <batchlas/blas/functions/potrf.hh>
#include <batchlas/blas/functions/trsm.hh>
#include <batchlas/blas/dispatch/no_route.hh>
#include <batchlas/blas/dispatch/vendor_available.hh>

#include "choice.hh"
#include "../../select/select.hh"
#include "../../extensions/potrf_native.hh"
#include "../../util/template-instantiations.hh"

#include <algorithm>
#include <complex>
#include <cstddef>
#include <variant>

namespace batchlas {
namespace ops::potrf {

template <class... F>
struct overloaded : F... { using F::operator()...; };
template <class... F>
overloaded(F...) -> overloaded<F...>;

template <class T>
select::Key key_of(const MatrixView<T, MatrixFormat::Dense>& A, Uplo uplo) {
    return {{"uplo", uplo == Uplo::Lower ? "L" : "U"}, {"n", A.rows()}, {"batch", A.batch_size()}};
}

// Correctness only (R3): false means the driver would throw or answer wrongly. Each clause
// is the argument check at the top of that driver's *_dispatch.
template <class T>
bool can_run(const PotrfChoice& c, const select::Device& d, const MatrixView<T, MatrixFormat::Dense>& A,
             Uplo uplo) {
    const int64_t n = A.rows();
    const auto budget = static_cast<std::size_t>(d.slm_budget);
    const bool native = d.is_gpu && d.has_sg32 && !A.is_heterogeneous() && n >= 1 && A.batch_size() >= 1;
    return std::visit(overloaded{
        [&](Tiny) {
            return native && n <= sycl_potrf::potrf_tiny_max_n<T>() &&
                   d.max_wg >= sycl_potrf::kPotrfTinyWgSize;
        },
        [&](Cta) { return native && n <= sycl_potrf::potrf_cta_max_n_for_slm<T>(budget); },
        [&](const Lpanel& l) {
            // 0 for a panel width this scalar type does not instantiate.
            return native && uplo == Uplo::Lower &&
                   n <= sycl_potrf::potrf_lpanel_max_n_for_slm<T>(budget, d.max_wg,
                                                                  resident::kMinBlocksPerSm, l.panel);
        },
        [&](Blocked) {
            // Lower-only; its leaf is the CTA kernel, so CTA must hold at least order 1.
            return native && uplo == Uplo::Lower && sycl_potrf::potrf_blocked_available<T>() &&
                   sycl_potrf::potrf_cta_max_n_for_slm<T>(budget) >= 1;
        },
        [&](Vendor) { return d.has_vendor_solver; },
    }, c);
}

template <Backend B, class T>
PotrfChoice choose(Queue& q, const MatrixView<T, MatrixFormat::Dense>& A, Uplo uplo) {
    const select::Device& d = select::device_of<B>(q);
    auto ok = [&](const PotrfChoice& c) { return can_run<T>(c, d, A, uplo); };
    try {
        return select::choose("potrf", select::dtype_name<T>(), d, key_of(A, uplo), candidates<T>(), ok, rules);
    } catch (const std::runtime_error&) {
        // Vendor-free burn-down reads coverage `miss` rows; a plain runtime_error records none.
        const auto all = candidates<T>();
        if (!dispatch::solver_vendor_available<B> && std::none_of(all.begin(), all.end(), ok))
            dispatch::throw_no_vendor_route<T>(dispatch::Op::potrf, B, dispatch::kSolverLibrary<B>);
        throw;
    }
}

// The coverage row's native flags (§5.6): computed only when coverage records a row.
template <Backend B, class T>
select::NativeFacts native_facts(Queue& q, const MatrixView<T, MatrixFormat::Dense>& A, Uplo uplo) {
    if (!dispatch::coverage::dynamic_enabled()) return {};
    const select::Device& d = select::device_of<B>(q);
    return select::native_facts(candidates<T>(), [&](const PotrfChoice& c) { return can_run<T>(c, d, A, uplo); });
}

template <Backend B, class T>
Event launch(Queue& q, const PotrfChoice& c, const MatrixView<T, MatrixFormat::Dense>& A, Uplo uplo,
             Span<std::byte> ws, Span<int32_t> info) {
    using MV = MatrixView<T, MatrixFormat::Dense>;
    return std::visit(overloaded{
        [&](Tiny) { return sycl_potrf::potrf_tiny_dispatch<T>(q, A, uplo, ws, info); },
        [&](Cta) { return sycl_potrf::potrf_cta_dispatch<T>(q, A, uplo, ws, info); },
        [&](const Lpanel& l) {
            return sycl_potrf::potrf_lpanel_dispatch<T>(q, A, uplo, ws, info, resident::kMinBlocksPerSm,
                                                        l.panel);
        },
        [&](Blocked) {
            // The public gemm and trsm: each child decides its own kernel.
            return sycl_potrf::potrf_blocked_dispatch<T>(
                q, A, uplo, ws, info,
                [](Queue& c, const MV& a, const MV& b, const MV& r, T alpha, T beta, Transpose ta,
                   Transpose tb, ComputePrecision p) { return gemm<B, T>(c, a, b, r, alpha, beta, ta, tb, p); },
                [](Queue& c, const MV& a, const MV& b, T alpha, Side s, Uplo u, Transpose t, Diag dg) {
                    return trsm<B, T>(c, a, b, alpha, s, u, t, dg);
                });
        },
        [&](Vendor) -> Event {
            if constexpr (dispatch::solver_vendor_available<B>)
                return backend::potrf_vendor<B, T>(q, A, uplo, ws, info);
            else
                dispatch::throw_no_vendor_route<T>(dispatch::Op::potrf, B, dispatch::kSolverLibrary<B>);
        },
    }, c);
}

// Exactly the chosen family's need (R5). Pure: no launch, no data read.
template <Backend B, class T>
std::size_t workspace(Queue& q, const PotrfChoice& c, const MatrixView<T, MatrixFormat::Dense>& A, Uplo uplo) {
    return std::visit(overloaded{
        [&](Tiny) { return sycl_potrf::potrf_tiny_buffer_size<T>(q, A); },
        [&](Cta) { return sycl_potrf::potrf_cta_buffer_size<T>(q, A); },
        [&](const Lpanel&) { return sycl_potrf::potrf_lpanel_buffer_size<T>(q, A); },
        [&](Blocked) { return sycl_potrf::potrf_blocked_buffer_size<T>(q, A, uplo); },
        [&](Vendor) -> std::size_t {
            if constexpr (dispatch::solver_vendor_available<B>)
                return backend::potrf_vendor_buffer_size<B, T>(q, A, uplo);
            else
                dispatch::throw_no_vendor_route<T>(dispatch::Op::potrf, B, dispatch::kSolverLibrary<B>);
        },
    }, c);
}

}  // namespace ops::potrf

template <Backend B, typename T>
Event potrf(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, Uplo uplo, Span<std::byte> workspace,
            Span<int32_t> info) {
    potrf_validate_params<T>(A, uplo);
    const auto c = ops::potrf::choose<B, T>(ctx, A, uplo);
    auto shape = select::square_shape<B, T>(A.rows(), A.batch_size());
    shape.uplo = uplo;  // part of the coverage key; never inferred
    select::TraceScope trace("potrf", c, shape, ops::potrf::native_facts<B, T>(ctx, A, uplo));
    return ops::potrf::launch<B, T>(ctx, c, A, uplo, workspace, info);
}

template <Backend B, typename T>
size_t potrf_buffer_size(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, Uplo uplo) {
    potrf_validate_params<T>(A, uplo);
    return ops::potrf::workspace<B, T>(ctx, ops::potrf::choose<B, T>(ctx, A, uplo), A, uplo);
}

#define POTRF_INSTANTIATE(B_, fp)                                         \
    BATCHLAS_INSTANTIATE(sig::potrf<fp>, potrf, B_, fp)                   \
    BATCHLAS_INSTANTIATE(sig::potrf_buffer_size<fp>, potrf_buffer_size, B_, fp)

#define POTRF_ALL(B_)                          \
    POTRF_INSTANTIATE(B_, float)               \
    POTRF_INSTANTIATE(B_, double)              \
    POTRF_INSTANTIATE(B_, std::complex<float>) \
    POTRF_INSTANTIATE(B_, std::complex<double>)

// Keyed on the device family, not the vendor library: without the library the Vendor arms
// compile to a throw, so the symbol exists in every build with the device.
#if BATCHLAS_HAS_CUDA_BACKEND
POTRF_ALL(Backend::CUDA)
#endif
#if BATCHLAS_HAS_ROCM_BACKEND
POTRF_ALL(Backend::ROCM)
#endif
#if BATCHLAS_HAS_HOST_BACKEND
POTRF_ALL(Backend::NETLIB)
#endif

#undef POTRF_ALL
#undef POTRF_INSTANTIATE

}  // namespace batchlas
