// trsm: the whole selection path (docs/design/flat-kernel-selection.md §4.3, rule R1;
// flat-kernel-selection-phase3-plan.md §1.2). public trsm() -> choose() -> std::visit -> launch.
// The kernel for a shape is the first runnable entry of the nearest row in
// tuned/trsm.<dtype>.<device>.txt; can_run() below only removes entries that cannot run.
// Cta and SgLeft are register-resident solvers for order <= 32; Blocked solves 32-wide diagonal
// blocks with Cta and updates the rest with the public gemm, which picks its own kernel.

#include <batchlas/backend_config.h>

#include <batchlas/blas/functions/gemm.hh>
#include <batchlas/blas/functions/trsm.hh>
#include <batchlas/no_route.hh>
#include "../../select/vendor.hh"

#include "choice.hh"
#include "../../select/select.hh"
#include "../../sycl/trsm_native.hh"
#include "../../util/template-instantiations.hh"

#include <algorithm>
#include <complex>
#include <cstdint>
#include <variant>

namespace batchlas {
namespace ops::trsm {

template <class... F>
struct overloaded : F... { using F::operator()...; };
template <class... F>
overloaded(F...) -> overloaded<F...>;

template <class T>
using MV = MatrixView<T, MatrixFormat::Dense>;

// The right-hand-side count: the extent of B the triangular order does not occupy.
template <class T>
std::int64_t rhs_count(const MV<T>& B, Side side) {
    return side == Side::Left ? B.cols() : B.rows();
}

// ConjTrans folds to T: the kernels differ from Trans only in a conjugation (trsm_canonical.hh).
template <class T>
select::Key key_of(const MV<T>& A, const MV<T>& B, Side side, Transpose transA) {
    return {{"side", side == Side::Left ? "L" : "R"}, {"trans", transA == Transpose::NoTrans ? "N" : "T"},
            {"order", A.rows()}, {"q", rhs_count<T>(B, side)}, {"batch", A.batch_size()}};
}

// Correctness only (R3): false means the driver would throw or answer wrongly. One launch covers
// the batch with a single (order, q, ld, stride), so no native family takes a heterogeneous
// batch. The vendor accepts one and runs at the full storage order (known-defects #12), as before.
template <class T>
bool can_run(const TrsmChoice& c, const select::Device& d, const MV<T>& A, const MV<T>& B, Side side) {
    const std::int64_t order = A.rows(), q = rhs_count<T>(B, side);
    const bool native = d.is_gpu && !A.is_heterogeneous() && !B.is_heterogeneous() && order >= 1 && q >= 1 &&
                        A.batch_size() >= 1;
    const int cta_max = sycl_trsm::trsm_cta_max_n<T>();
    // V1's work-group ladder bottoms out at 32 lanes whatever max_wg is (trsm_v1_ladder_wg).
    const bool v1_fits = d.max_wg >= 32;
    return std::visit(overloaded{
        [&](Cta) { return native && v1_fits && order <= cta_max; },
        [&](SgLeft) {
            return native && side == Side::Left && d.has_sg32 && order <= sycl_trsm::kTrsmSgLeftMaxN &&
                   d.max_wg >= sycl_trsm::kTrsmSgLeftWgSize;
        },
        [&](Blocked) { return native && v1_fits && sycl_trsm::trsm_blocked_available<T>() && cta_max >= 1; },
        [&](Vendor) { return d.has_vendor_blas; },
    }, c);
}

template <Backend B, class T>
TrsmChoice choose(Queue& q, const MV<T>& A, const MV<T>& Bm, Side side, Transpose transA) {
    const select::Device& d = select::device_of<B>(q);
    auto ok = [&](const TrsmChoice& c) { return can_run<T>(c, d, A, Bm, side); };
    try {
        return select::choose("trsm", select::dtype_name<T>(), d, key_of<T>(A, Bm, side, transA), candidates<T>(),
                              ok, rules);
    } catch (const std::runtime_error&) {
        // Vendor-free burn-down reads coverage `miss` rows; a plain runtime_error records none.
        const auto all = candidates<T>();
        if (!select::level3_vendor_available<B> && std::none_of(all.begin(), all.end(), ok))
            select::throw_no_vendor_route<T>(Op::trsm, B, select::kLevel3Library<B>);
        throw;
    }
}

// The coverage row's native flags (§5.6): computed only when coverage records a row.
template <Backend B, class T>
select::NativeFacts native_facts(Queue& q, const MV<T>& A, const MV<T>& Bm, Side side) {
    if (!coverage::dynamic_enabled()) return {};
    const select::Device& d = select::device_of<B>(q);
    return select::native_facts(candidates<T>(), [&](const TrsmChoice& c) { return can_run<T>(c, d, A, Bm, side); });
}

template <Backend B, class T>
Event launch(Queue& q, const TrsmChoice& c, const MV<T>& A, const MV<T>& Bm, T alpha, Side side, Uplo uplo,
             Transpose transA, Diag diag) {
    return std::visit(overloaded{
        [&](Cta) { return sycl_trsm::trsm_native_v1_dispatch<T>(q, A, Bm, alpha, side, uplo, transA, diag); },
        [&](SgLeft) { return sycl_trsm::trsm_native_sg_left_dispatch<T>(q, A, Bm, alpha, uplo, transA, diag); },
        [&](Blocked) {
            // The public gemm, not gemm_custom: trsm's operands are sub-views carrying the
            // parent's ld, where the native GEMM collapses.
            // evidence: docs/perf/gemm.md#the-strided-ld-defect-and-the-routing-fix
            return sycl_trsm::trsm_native_blocked<T>(
                q, A, Bm, alpha, side, uplo, transA, diag,
                [](Queue& c2, const MV<T>& ga, const MV<T>& gb, const MV<T>& gc, T galpha, T gbeta, Transpose gta,
                   Transpose gtb, ComputePrecision gp) {
                    return gemm<B, T>(c2, ga, gb, gc, galpha, gbeta, gta, gtb, gp);
                });
        },
        [&](Vendor) -> Event {
            if constexpr (select::level3_vendor_available<B>)
                return backend::trsm_vendor<B, T>(q, A, Bm, side, uplo, transA, diag, alpha);
            else
                select::throw_no_vendor_route<T>(Op::trsm, B, select::kLevel3Library<B>);
        },
    }, c);
}

}  // namespace ops::trsm

// trsm takes no workspace (no trsm_buffer_size), so R5 has nothing to size.
template <Backend Back, typename T>
Event trsm(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, const MatrixView<T, MatrixFormat::Dense>& B,
           T alpha, Side side, Uplo uplo, Transpose transA, Diag diag) {
    trsm_validate_params(A, B, side, uplo, transA, diag);
    const auto c = ops::trsm::choose<Back, T>(ctx, A, B, side, transA);
    // The coverage row's key: m, n are B's extents and k the triangular order, as before.
    auto shape = select::square_shape<Back, T>(A.rows(), A.batch_size());
    shape.m = B.rows();
    shape.n = B.cols();
    shape.side = side;
    shape.uplo = uplo;
    shape.transA = transA;
    shape.diag = diag;
    const select::Key trace_key = ops::trsm::key_of<T>(A, B, side, transA);
    select::TraceScope trace("trsm", c, shape, ops::trsm::native_facts<Back, T>(ctx, A, B, side), trace_key);
    return ops::trsm::launch<Back, T>(ctx, c, A, B, alpha, side, uplo, transA, diag);
}

#define TRSM_INSTANTIATE(B_, fp) BATCHLAS_INSTANTIATE(sig::trsm<fp>, trsm, B_, fp)

#define TRSM_ALL(B_)                          \
    TRSM_INSTANTIATE(B_, float)               \
    TRSM_INSTANTIATE(B_, double)              \
    TRSM_INSTANTIATE(B_, std::complex<float>) \
    TRSM_INSTANTIATE(B_, std::complex<double>)

// Keyed on the device family, not the vendor library: without the library the Vendor arm
// compiles to a throw, so the symbol exists in every build with the device.
#if BATCHLAS_HAS_CUDA_BACKEND
TRSM_ALL(Backend::CUDA)
#endif
#if BATCHLAS_HAS_ROCM_BACKEND
TRSM_ALL(Backend::ROCM)
#endif
#if BATCHLAS_HAS_HOST_BACKEND
TRSM_ALL(Backend::NETLIB)
#endif

#undef TRSM_ALL
#undef TRSM_INSTANTIATE

}  // namespace batchlas
