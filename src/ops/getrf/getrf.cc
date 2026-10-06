// getrf (flat-kernel-selection.md §4.3, R1; phase 5): select::run takes the first entry of the nearest
// tuned/getrf.<dtype>.<device>.txt row that can_run() admits. Tiny holds the whole matrix in registers,
// Cta in local memory; Blocked factorises Cta-sized panels and updates the rest with the public trsm and
// gemm, each of which picks its own kernel. PIVOTS: every family packs 1-based int32 into the int64 span's
// first half (getrf_native.hh), except NETLIB's LAPACKE, which writes int64 -- so no native family runs on
// that backend.

#include <batchlas/backend_config.h>

#include <batchlas/blas/functions/gemm.hh>
#include <batchlas/blas/functions/getrf.hh>
#include <batchlas/blas/functions/trsm.hh>

#include "choice.hh"
#include "../../select/select.hh"
#include "../../extensions/getrf_native.hh"
#include "../../util/template-instantiations.hh"

#include <complex>
#include <cstddef>
#include <cstdint>
#include <variant>

namespace batchlas {
namespace ops::getrf {

using select::overloaded;

template <class T>
using MV = MatrixView<T, MatrixFormat::Dense>;

template <class T>
select::Key key_of(const MV<T>& A) {
    return {{"n", A.rows()}, {"batch", A.batch_size()}};
}

// Correctness only (R3): false means the driver would throw or answer wrongly. Each clause is the
// argument check at the top of that driver's *_dispatch. One launch covers the batch with a single
// (n, ld, stride), so no native family takes a heterogeneous batch; the vendor serves everything.
template <Backend B, class T>
bool can_run(const GetrfChoice& c, const select::Device& d, const MV<T>& A) {
    const std::int64_t n = A.rows();
    const bool native = B != Backend::NETLIB && d.is_gpu && d.has_sg32 && !A.is_heterogeneous() &&
                        A.rows() == A.cols() && n >= 1 && A.batch_size() >= 1;
    const auto budget = static_cast<std::size_t>(d.slm_budget);
    return std::visit(overloaded{
        [&](Tiny) {
            return native && n <= sycl_getrf::getrf_tiny_max_n<T>() && d.max_wg >= sycl_getrf::kGetrfTinyWgSize;
        },
        [&](Cta) { return native && n <= sycl_getrf::getrf_cta_max_n_for_slm<T>(budget); },
        [&](Blocked) {
            // min_blocks 1: the driver asks whether the panel leaf's argmax slots fit at all.
            return native && sycl_getrf::getrf_blocked_available<T>() &&
                   sycl_getrf::getrf_cta_max_n_for_slm<T>(budget, 1) >= 1;
        },
        [&](Vendor) { return d.has_vendor; },
    }, c);
}

template <Backend B, class T>
Event launch(Queue& q, const GetrfChoice& c, const MV<T>& A, Span<int64_t> pivots, Span<std::byte> ws,
             Span<int32_t> info) {
    return std::visit(overloaded{
        [&](Tiny) { return sycl_getrf::getrf_tiny_dispatch<T>(q, A, pivots, ws, info); },
        [&](Cta) { return sycl_getrf::getrf_cta_dispatch<T>(q, A, pivots, ws, info); },
        [&](Blocked) {
            // The public gemm and trsm, not gemm_custom: the operands are sub-views carrying the
            // parent's ld, where the native GEMM collapses.
            // evidence: docs/perf/gemm.md#the-strided-ld-defect-and-the-routing-fix
            return sycl_getrf::getrf_blocked_dispatch<T>(
                q, A, pivots, ws, info,
                [](Queue& c2, const MV<T>& ga, const MV<T>& gb, const MV<T>& gc, T galpha, T gbeta, Transpose gta,
                   Transpose gtb, ComputePrecision gp) {
                    return gemm<B, T>(c2, ga, gb, gc, galpha, gbeta, gta, gtb, gp);
                },
                [](Queue& c2, const MV<T>& ta, const MV<T>& tb, T talpha, Side tside, Uplo tuplo, Transpose ttrans,
                   Diag tdiag) { return trsm<B, T>(c2, ta, tb, talpha, tside, tuplo, ttrans, tdiag); });
        },
        [&](Vendor) -> Event {
            if constexpr (select::has_library<B>(spec.vendor))
                return backend::getrf_vendor<B, T>(q, A, pivots, ws, info);
            else select::no_vendor<B, T>(spec);
        },
    }, c);
}

// Exactly the chosen family's need (R5); Blocked's gemm and trsm take no workspace. None of these
// may read A.data_ptr(): inv.cc sizes getrf from a layout pass with a null data pointer.
template <Backend B, class T>
std::size_t workspace(Queue& q, const GetrfChoice& c, const MV<T>& A) {
    return std::visit(overloaded{
        [&](Tiny) { return sycl_getrf::getrf_tiny_buffer_size<T>(q, A); },
        [&](Cta) { return sycl_getrf::getrf_cta_buffer_size<T>(q, A); },
        [&](Blocked) { return sycl_getrf::getrf_blocked_buffer_size<T>(q, A); },
        [&](Vendor) -> std::size_t {
            if constexpr (select::has_library<B>(spec.vendor))
                return backend::getrf_vendor_buffer_size<B, T>(q, A);
            else select::no_vendor<B, T>(spec);
        },
    }, c);
}

}  // namespace ops::getrf

template <Backend B, typename T>
Event getrf(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, Span<int64_t> pivots,
            Span<std::byte> work_space, Span<int32_t> info) {
    namespace o = ops::getrf;
    getrf_validate_params<T>(A);
    // The coverage row's key: m, n are A's extents (a non-square call reaches the vendor), k = m.
    const coverage::Shape shape{.m = A.rows(), .n = A.cols(), .k = A.rows(), .batch = A.batch_size()};
    const select::Key key = o::key_of<T>(A);
    return select::run<B, T>(
        o::spec, ctx, key, o::candidates<T>(), [&](const auto& c, const auto& d) { return o::can_run<B, T>(c, d, A); },
        shape, key, [&](const auto& c) { return o::launch<B, T>(ctx, c, A, pivots, work_space, info); });
}

template <Backend B, typename T>
size_t getrf_buffer_size(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A) {
    namespace o = ops::getrf;
    getrf_validate_params<T>(A);
    const auto c = select::pick<B, T>(o::spec, ctx, o::key_of<T>(A), o::candidates<T>(),
                                      [&](const auto& k, const auto& d) { return o::can_run<B, T>(k, d, A); });
    return o::workspace<B, T>(ctx, c, A);
}

#define GETRF_INSTANTIATE(B_, fp) \
    BATCHLAS_INSTANTIATE_OP(B_, fp, getrf) BATCHLAS_INSTANTIATE_OP(B_, fp, getrf_buffer_size)
BATCHLAS_INSTANTIATE_SCALAR_ALL_BACKENDS(GETRF_INSTANTIATE)
#undef GETRF_INSTANTIATE

}  // namespace batchlas
