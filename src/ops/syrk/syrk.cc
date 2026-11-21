// SPDX-License-Identifier: MIT
// syrk (flat-kernel-selection.md §4.3, R1; §12): select::run takes the first entry of the nearest
// tuned/syrk.<device>.txt row that can_run() admits. Gram puts all of C in one tile sized to n
// (n <= 128) and stages A once; Triangular walks 128-wide tiles of the requested triangle only.

#include <batchlas/backend_config.h>

#include <batchlas/blas/functions/syrk.hh>
#include <batchlas/error.hh>

#include <src/ops/level3/batched.hh>
#include <src/ops/syrk/choice.hh>
#include <src/backends/level3_shape.hh>
#include <src/ops/syrk/kernels/syrk_gram_tiles.hh>
#include <src/ops/syrk/kernels/syrk_triangular_tiles.hh>
#include <src/select/select.hh>
#include <src/util/batch_mask.hh>
#include <src/util/template-instantiations.hh>

#include <complex>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <variant>

namespace batchlas {
namespace ops::syrk {

using select::overloaded;

template <typename Scalar>
using MV = MatrixView<Scalar, MatrixFormat::Dense>;

// op(A)'s inner extent: the reduction length of C = op(A) op(A)^T.
template <typename Scalar>
std::int64_t inner(const MV<Scalar>& A, Transpose transA) {
    return transA == Transpose::NoTrans ? A.cols() : A.rows();
}

template <typename Scalar>
select::Key key_of(const MV<Scalar>& A, const MV<Scalar>& C, Transpose transA) {
    const std::int64_t n = C.rows(), k = inner<Scalar>(A, transA);
    const char* trans = transA == Transpose::NoTrans ? "N" : (transA == Transpose::Trans ? "T" : "C");
    return {{"form", form_of(n, k)}, {"trans", trans}, {"n", n}, {"k", k}, {"batch", C.batch_size()}};
}

// Correctness only (R3): false means the kernel would throw or answer wrongly. The tile kernels
// are portable SYCL wired for CUDA, SYCL and a vendor-free host (NETLIB), take one
// (n, k, ld, stride) per launch (no heterogeneous batch) and put the batch (chunked by the launch) and triangular's tile count
// (not chunkable) in grid dimensions capped at 65535 on CUDA. syrk is real-only, so ConjTrans is Trans to both kernels (the table, not
// can_run, keeps it on the vendor). Every vendor loop (cuBLAS, rocBLAS, netlib) runs each item
// at the top-level (n, k), a wrong answer for a heterogeneous batch, so no family takes one.
// Shapes are validated before this runs.
template <Backend B, typename Scalar>
bool can_run(const SyrkChoice& c, const select::Device& d, const MV<Scalar>& A, const MV<Scalar>& C, Transpose transA) {
    // Gram and Triangular honour a batch mask; the vendor's per-item loop cannot skip an item.
    const bool masked = batchlas::detail::any_batch_mask(A, C);
    const std::int64_t n = C.rows(), k = inner<Scalar>(A, transA), batch = C.batch_size();
    const Transpose real_trans = transA == Transpose::NoTrans ? Transpose::NoTrans : Transpose::Trans;
    const bool homogeneous = !A.is_heterogeneous() && !C.is_heterogeneous();
    // The worked example of a host route: native_backend here and in launch(), natives_run_on here.
    // evidence: docs/design/vendor-independence.md#the-host-native-gate
    const bool native = select::native_backend<B> && select::natives_run_on(d) && homogeneous && n >= 1 && k >= 1 &&
                        batch >= 1;
    return std::visit(overloaded{
        [&](Gram) {
            return native && backend::detail::syrk_gram_supported<Scalar>(A, C, real_trans, false) &&
                   d.max_wg >= gram_threads(n) && d.slm_budget >= gram_slm_bytes<Scalar>(n);
        },
        [&](Triangular) {
            return native && d.max_wg >= kTriangularWg && d.slm_budget >= triangular_slm_bytes<Scalar>() &&
                   triangular_groups(n, backend::detail::kTriangularTile) <= d.max_groups_yz;
        },
        [&](Vendor) { return vendor_wired<B, Scalar> && d.has_vendor && homogeneous && !masked; },
    }, c);
}

// The native arms compile for CUDA, SYCL and a vendor-free NETLIB (native_backend); can_run never
// admits the other instantiations.
template <Backend B, typename Scalar>
Event launch(Queue& q, const SyrkChoice& c, const MV<Scalar>& A, const MV<Scalar>& C, Scalar alpha, Scalar beta,
             Uplo uplo,
             Transpose transA) {
    return std::visit(overloaded{
        [&](Gram) -> Event {
            if constexpr (select::native_backend<B>)
                return level3::for_each_chunk(q, C.batch_size(), level3::grid_chunk<B>(q), [&](int f, int n) {
                    return backend::detail::syrk_gram_tiles<Scalar, false>(q, level3::batch_range(A, f, n),
                                                                           level3::batch_range(C, f, n), alpha, beta,
                                                                           uplo, transA);
                });
            else
                throw batchlas::internal_error("syrk: gram is wired for native backends only");
        },
        [&](Triangular) -> Event {
            if constexpr (select::native_backend<B> && !std::is_same_v<Scalar, std::complex<double>>)
                return level3::for_each_chunk(q, C.batch_size(), level3::grid_chunk<B>(q), [&](int f, int n) {
                    return backend::detail::syrk_triangular_tiles<Scalar>(q, level3::batch_range(A, f, n),
                                                                              level3::batch_range(C, f, n), alpha,
                                                                              beta, uplo, transA);
                });
            else
                throw batchlas::internal_error("syrk: triangular is wired for native backends only");
        },
        [&](Vendor) -> Event {
            if constexpr (vendor_wired<B, Scalar>)
                return backend::syrk_vendor<B, Scalar>(q, A, C, alpha, beta, uplo, transA);
            else select::no_vendor<B, Scalar>(spec);
        },
    }, c);
}

}  // namespace ops::syrk

// syrk takes no workspace (no syrk_buffer_size), so R5 has nothing to size.
template <Backend Back, FloatingOrComplexScalar T>
Event syrk(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, const MatrixView<T, MatrixFormat::Dense>& C,
           T alpha, T beta, Uplo uplo, Transpose transA) {
    (void)batchlas::detail::call_mask("syrk", A, C);
    (void)backend::shape::validate_rank_k<batchlas::invalid_argument>("syrk", A, C, transA, /*hermitian=*/false);
    if (ComplexScalar<T> && transA == Transpose::ConjTrans)
        throw batchlas::invalid_argument("syrk: ConjTrans with a complex T is herk's A^H A, not a symmetric update");
    // An empty batch launches nothing under any pin, as gemm's does.
    if (C.batch_size() == 0) return ctx.create_event_after_external_work();
    // The coverage row's key: m = n = C's order, k = op(A)'s inner extent.
    const coverage::Shape shape{.m = C.rows(), .n = C.rows(), .k = ops::syrk::inner<T>(A, transA),
                                .batch = A.batch_size(), .transA = transA, .uplo = uplo};
    const select::Key key = ops::syrk::key_of<T>(A, C, transA);
    return select::run<Back, T>(
        ops::syrk::spec, ctx, key, ops::syrk::candidates<T>(),
        [&](const auto& c, const auto& d) { return ops::syrk::can_run<Back, T>(c, d, A, C, transA); }, shape, key,
        [&](const auto& c) { return ops::syrk::launch<Back, T>(ctx, c, A, C, alpha, beta, uplo, transA); });
}

#define SYRK_INSTANTIATE(B_, fp) BATCHLAS_INSTANTIATE_OP(B_, fp, syrk)
BATCHLAS_INSTANTIATE_SCALAR_ALL_BACKENDS(SYRK_INSTANTIATE)
#undef SYRK_INSTANTIATE

}  // namespace batchlas
