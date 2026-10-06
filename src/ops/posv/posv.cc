// posv (flat-kernel-selection.md §4.3, R1; flat-kernel-selection-phase3-plan.md §1.1): select::run takes
// the first entry of the nearest tuned/posv.<dtype>.<device>.txt row that can_run() admits. Tiny is the
// fused factor-and-solve kernel; Cta and Blocked compose the public potrf with one fused solve (Cta) or
// two public trsm calls (Blocked), and each child picks its own kernel.

#include <batchlas/backend_config.h>

#include <batchlas/blas/functions/posv.hh>
#include <batchlas/blas/functions/potrf.hh>
#include <batchlas/blas/functions/trsm.hh>

#include "choice.hh"
#include "../../select/select.hh"
#include "../../extensions/getrs_native.hh"
#include "../../extensions/solve_native.hh"
#include "../../util/template-instantiations.hh"

#include <complex>
#include <cstddef>
#include <string>
#include <type_traits>
#include <variant>

namespace batchlas {
namespace ops::posv {

using select::overloaded;

template <class T>
select::Key key_of(const MatrixView<T, MatrixFormat::Dense>& A, const MatrixView<T, MatrixFormat::Dense>& B,
                   Uplo uplo) {
    return {{"uplo", uplo == Uplo::Lower ? "L" : "U"}, {"n", A.rows()}, {"nrhs", B.cols()},
            {"batch", A.batch_size()}};
}

// Correctness only (R3): false means the driver would throw. Each clause is the argument
// check at the top of that driver's *_dispatch; Blocked's children decide for themselves,
// except on a heterogeneous batch: vendor potrf and trsm accept it and solve at the full
// storage order, a silent wrong answer, so no family may take one.
template <class T>
bool can_run(const PosvChoice& c, const select::Device& d, const MatrixView<T, MatrixFormat::Dense>& A,
             const MatrixView<T, MatrixFormat::Dense>& B) {
    const int64_t n = A.rows(), nrhs = B.cols();
    const bool homogeneous = !A.is_heterogeneous() && !B.is_heterogeneous();
    const bool native = d.is_gpu && d.has_sg32 && homogeneous && n >= 1 && nrhs >= 1 && A.batch_size() >= 1;
    return std::visit(overloaded{
        [&](Tiny) {
            return native && n <= sycl_posv::posv_tiny_max_n<T>() && nrhs <= sycl_posv::kPosvTinyMaxRhs &&
                   d.max_wg >= sycl_posv::kPosvTinyWgSize;
        },
        [&](Cta) {
            const auto cap = sycl_getrs::getrs_fused_max_rhs_elems<T>(static_cast<std::size_t>(d.slm_budget));
            return native && nrhs <= sycl_getrs::kGetrsFusedMaxRhs &&
                   static_cast<std::size_t>(n) * static_cast<std::size_t>(nrhs) <= cap;
        },
        [&](Blocked) { return homogeneous; },
    }, c);
}

template <Backend B, class T>
Event launch(Queue& q, const PosvChoice& c, const MatrixView<T, MatrixFormat::Dense>& A,
             const MatrixView<T, MatrixFormat::Dense>& Bm, Uplo uplo, Span<std::byte> ws, Span<int32_t> info) {
    return std::visit(overloaded{
        [&](Tiny) { return sycl_posv::posv_tiny_dispatch<T>(q, A, Bm, uplo, ws, info); },
        [&](Cta) {
            // The public potrf, then BOTH triangular solves in one kernel: two nrhs = 1 trsm
            // launches stream L twice at a fraction of the bandwidth one fused pass gets.
            (void)potrf<B, T>(q, A, uplo, ws, info);
            return sycl_getrs::potrs_fused_dispatch<T>(q, A, Bm, uplo);
        },
        [&](Blocked) {
            // No split: trsm takes no workspace, so potrf owns the whole span.
            (void)potrf<B, T>(q, A, uplo, ws, info);
            // ConjTrans only for a complex scalar: a real backend may refuse ConjTrans, and
            // for a real matrix Trans is the same operation. For a real T, base_type<T> IS T.
            constexpr bool kReal = std::is_same_v<T, typename base_type<T>::type>;
            constexpr Transpose kAdj = kReal ? Transpose::Trans : Transpose::ConjTrans;
            const T one = T(1);
            if (uplo == Uplo::Lower) {
                (void)trsm<B, T>(q, A, Bm, one, Side::Left, Uplo::Lower, Transpose::NoTrans, Diag::NonUnit);
                return trsm<B, T>(q, A, Bm, one, Side::Left, Uplo::Lower, kAdj, Diag::NonUnit);
            }
            (void)trsm<B, T>(q, A, Bm, one, Side::Left, Uplo::Upper, kAdj, Diag::NonUnit);
            return trsm<B, T>(q, A, Bm, one, Side::Left, Uplo::Upper, Transpose::NoTrans, Diag::NonUnit);
        },
    }, c);
}

// Exactly the chosen family's need (R5); neither solve of Cta or Blocked takes workspace.
template <Backend B, class T>
std::size_t workspace(Queue& q, const PosvChoice& c, const MatrixView<T, MatrixFormat::Dense>& A,
                      const MatrixView<T, MatrixFormat::Dense>& Bm, Uplo uplo) {
    return std::visit(overloaded{
        [&](Tiny) { return sycl_posv::posv_tiny_buffer_size<T>(q, A, Bm); },
        [&](Cta) { return potrf_buffer_size<B, T>(q, A, uplo); },
        [&](Blocked) { return potrf_buffer_size<B, T>(q, A, uplo); },
    }, c);
}

// An empty problem (n, nrhs or batch 0) or a heterogeneous batch has no kernel; the old router
// refused both the same way. Thrown before choose() so a pin cannot take them either.
template <class T>
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
           Uplo uplo, Span<std::byte> work_space, Span<int32_t> info) {
    namespace o = ops::posv;
    posv_validate_params<T>(A, B, uplo);
    o::throw_if_unservable<T>(A, B, "posv");
    // The coverage row's key: m = k = order, n = nrhs, and uplo (the only field separating rows).
    const coverage::Shape shape{.m = A.rows(), .n = B.cols(), .k = A.rows(), .batch = A.batch_size(), .uplo = uplo};
    return select::run<Back, T>(
        o::spec, ctx, o::key_of(A, B, uplo), o::candidates<T>(),
        [&](const auto& c, const auto& d) { return o::can_run<T>(c, d, A, B); }, shape,
        {{"n", A.rows()}, {"nrhs", B.cols()}, {"batch", A.batch_size()}},
        [&](const auto& c) { return o::launch<Back, T>(ctx, c, A, B, uplo, work_space, info); });
}

template <Backend Back, typename T>
size_t posv_buffer_size(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A,
                        const MatrixView<T, MatrixFormat::Dense>& B, Uplo uplo) {
    namespace o = ops::posv;
    posv_validate_params<T>(A, B, uplo);
    o::throw_if_unservable<T>(A, B, "posv_buffer_size");
    const auto c = select::pick<Back, T>(o::spec, ctx, o::key_of(A, B, uplo), o::candidates<T>(),
                                         [&](const auto& k, const auto& d) { return o::can_run<T>(k, d, A, B); });
    return o::workspace<Back, T>(ctx, c, A, B, uplo);
}

// Keyed on the device family: posv has no vendor arm, so every build with the device has it.
#define POSV_INSTANTIATE(B_, fp) BATCHLAS_INSTANTIATE_OP(B_, fp, posv) BATCHLAS_INSTANTIATE_OP(B_, fp, posv_buffer_size)
BATCHLAS_INSTANTIATE_SCALAR_ALL_BACKENDS(POSV_INSTANTIATE)
#undef POSV_INSTANTIATE

}  // namespace batchlas
