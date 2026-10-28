// SPDX-License-Identifier: MIT
// posv (flat-kernel-selection.md §4.3, R1; flat-kernel-selection-phase3-plan.md §1.1): select::run takes
// the first entry of the nearest tuned/posv.<device>.txt row that can_run() admits. Tiny is the
// fused factor-and-solve kernel; Cta and Blocked compose the public potrf with one fused solve (Cta) or
// two public trsm calls (Blocked), and each child picks its own kernel.

#include <batchlas/backend_config.h>

#include <batchlas/blas/functions/posv.hh>
#include <batchlas/blas/functions/potrf.hh>
#include <batchlas/blas/functions/trsm.hh>

#include <src/ops/posv/choice.hh>
#include <src/ops/potrs/potrs.hh>
#include <src/select/select.hh>
#include <src/util/batch_mask.hh>
#include <src/ops/getrs/getrs_native.hh>
#include <src/kernels/common/solve_native.hh>
#include <src/util/template-instantiations.hh>

#include <complex>
#include <cstddef>
#include <string>
#include <type_traits>
#include <variant>

namespace batchlas {
namespace ops::posv {

using select::overloaded;

template <typename T>
select::Key key_of(const MatrixView<T, MatrixFormat::Dense>& A, const MatrixView<T, MatrixFormat::Dense>& B,
                   Uplo uplo) {
    return {{"uplo", uplo == Uplo::Lower ? "L" : "U"}, {"n", A.rows()}, {"nrhs", B.cols()},
            {"batch", A.batch_size()}};
}

// Correctness only (R3): false means the driver would throw. Each clause is the argument
// check at the top of that driver's *_dispatch; Blocked's children decide for themselves,
// except on a heterogeneous batch: vendor potrf and trsm accept it and solve at the full
// storage order, a silent wrong answer, so no family may take one.
template <typename T>
bool can_run(const PosvChoice& c, const select::Device& d, const MatrixView<T, MatrixFormat::Dense>& A,
             const MatrixView<T, MatrixFormat::Dense>& B) {
    // tiny honours a batch mask; cta and blocked run their children (potrf, the solve) on the masked views.
    const int64_t n = A.rows(), nrhs = B.cols();
    const bool homogeneous = !A.is_heterogeneous() && !B.is_heterogeneous();
    const bool native = d.is_gpu && d.has_sg32 && homogeneous && n >= 1 && nrhs >= 1 && A.batch_size() >= 1;
    return std::visit(overloaded{
        [&](Tiny) {
            return native && n <= sycl_posv::posv_tiny_max_n<T>() && nrhs <= sycl_posv::kPosvTinyMaxRhs &&
                   d.max_wg >= sycl_posv::kPosvTinyWgSize;
        },
        [&](Cta) { return native && ops::potrs::fused_fits<T>(d, n, nrhs); },
        [&](Blocked) { return homogeneous; },
    }, c);
}

template <Backend B, typename T>
Event launch(Queue& q, const PosvChoice& c, const MatrixView<T, MatrixFormat::Dense>& A,
             const MatrixView<T, MatrixFormat::Dense>& Bm, Uplo uplo, Span<std::byte> ws, Span<int32_t> info) {
    return std::visit(overloaded{
        [&](Tiny) { return sycl_posv::posv_tiny_dispatch<T>(q, A, Bm, uplo, ws, info); },
        [&](Cta) {
            // The public potrf, then BOTH triangular solves in one kernel: two nrhs = 1 trsm
            // launches stream L twice at a fraction of the bandwidth one fused pass gets.
            (void)potrf<B, T>(q, A, uplo, ws, info);
            return ops::potrs::solve<B, T>(q, A, Bm, uplo, true);
        },
        [&](Blocked) {
            // No split: the solve takes no workspace, so potrf owns the whole span.
            (void)potrf<B, T>(q, A, uplo, ws, info);
            return ops::potrs::solve<B, T>(q, A, Bm, uplo, false);
        },
    }, c);
}

// Exactly the chosen family's need (R5); neither solve of Cta or Blocked takes workspace.
template <Backend B, typename T>
std::size_t workspace(Queue& q, const PosvChoice& c, const MatrixView<T, MatrixFormat::Dense>& A,
                      const MatrixView<T, MatrixFormat::Dense>& Bm, Uplo uplo) {
    return std::visit(overloaded{
        [&](Tiny) { return sycl_posv::posv_tiny_buffer_size<T>(q, A, Bm); },
        [&](Cta) { return potrf_buffer_size<B, T>(q, A, uplo); },
        [&](Blocked) { return potrf_buffer_size<B, T>(q, A, uplo); },
    }, c);
}

// n or nrhs 0 (batch 0 returned earlier, a no-op) or a heterogeneous batch has no kernel, so
// both throw before choose(): neither Auto nor a pin can take them.
template <typename T>
void throw_if_unservable(const MatrixView<T, MatrixFormat::Dense>& A, const MatrixView<T, MatrixFormat::Dense>& Bm,
                         const char* who) {
    if (A.is_heterogeneous() || Bm.is_heterogeneous())
        throw batchlas::internal_error(std::string(who) + ": a heterogeneous batch (per-item active dims on " +
                                       (A.is_heterogeneous() ? "A" : "B") + "), which no posv kernel serves.");
    if (A.rows() >= 1 && Bm.cols() >= 1 && A.batch_size() >= 1) return;
    throw batchlas::internal_error(std::string(who) + ": n = " + std::to_string(A.rows()) + ", nrhs = " +
                                   std::to_string(Bm.cols()) + ", batch = " + std::to_string(A.batch_size()) +
                                   " is an empty problem, which no posv kernel serves.");
}

}  // namespace ops::posv

template <Backend Back, typename T>
Event posv(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, const MatrixView<T, MatrixFormat::Dense>& B,
           Uplo uplo, Span<std::byte> workspace, Span<int32_t> info) {
    posv_validate_params<T>(A, B, uplo);
    (void)detail::call_mask("posv", A, B);
    // An empty batch is a no-op under any pin, as gemm's is: it never reaches selection.
    if (A.batch_size() == 0) return ctx.create_event_after_external_work();
    ops::posv::throw_if_unservable<T>(A, B, "posv");
    // The coverage row's key: m = k = order, n = nrhs, and uplo (the only field separating rows).
    const coverage::Shape shape{.m = A.rows(), .n = B.cols(), .k = A.rows(), .batch = A.batch_size(), .uplo = uplo};
    return select::run<Back, T>(
        ops::posv::spec, ctx, ops::posv::key_of(A, B, uplo), ops::posv::candidates<T>(),
        [&](const auto& c, const auto& d) { return ops::posv::can_run<T>(c, d, A, B); }, shape,
        {{"n", A.rows()}, {"nrhs", B.cols()}, {"batch", A.batch_size()}},
        [&](const auto& c) {
            // Cta and Blocked chain potrf into the solves, so they need an in-order queue.
            return select::on_in_order_queue(ctx, [&](Queue& q) {
                return ops::posv::launch<Back, T>(q, c, A, B, uplo, workspace, info);
            });
        });
}

template <Backend Back, typename T>
size_t posv_buffer_size(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A,
                        const MatrixView<T, MatrixFormat::Dense>& B, Uplo uplo) {
    posv_validate_params<T>(A, B, uplo);
    (void)detail::call_mask("posv_buffer_size", A, B);
    if (A.batch_size() == 0) return 0;
    ops::posv::throw_if_unservable<T>(A, B, "posv_buffer_size");
    const auto c = select::pick<Back, T>(
        ops::posv::spec, ctx, ops::posv::key_of(A, B, uplo), ops::posv::candidates<T>(),
        [&](const auto& k, const auto& d) { return ops::posv::can_run<T>(k, d, A, B); });
    return ops::posv::workspace<Back, T>(ctx, c, A, B, uplo);
}

// Keyed on the device family: posv has no vendor arm, so every build with the device has it.
#define POSV_INSTANTIATE(B_, fp) BATCHLAS_INSTANTIATE_OP(B_, fp, posv) BATCHLAS_INSTANTIATE_OP(B_, fp, posv_buffer_size)
BATCHLAS_INSTANTIATE_SCALAR_ALL_BACKENDS(POSV_INSTANTIATE)
#undef POSV_INSTANTIATE

}  // namespace batchlas
