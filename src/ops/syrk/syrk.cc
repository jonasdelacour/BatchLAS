// syrk (flat-kernel-selection.md §4.3, R1; §12): select::run takes the first entry of the nearest
// tuned/syrk.<dtype>.<device>.txt row that can_run() admits. Gram puts all of C in one tile sized to n
// (n <= 128) and stages A once; Triangular walks 128-wide tiles of the requested triangle only (float).

#include <batchlas/backend_config.h>

#include <batchlas/blas/functions/syrk.hh>

#include "choice.hh"
#include "../../backends/level3_shape.hh"
#include "../../backends/syrk_gram_tiles.hh"
#include "../../backends/syrk_triangular_tiles.hh"
#include "../../select/select.hh"
#include "../../util/template-instantiations.hh"

#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <variant>

namespace batchlas {
namespace ops::syrk {

using select::overloaded;

template <class T>
using MV = MatrixView<T, MatrixFormat::Dense>;

// op(A)'s inner extent: the reduction length of C = op(A) op(A)^T.
template <class T>
std::int64_t inner(const MV<T>& A, Transpose transA) {
    return transA == Transpose::NoTrans ? A.cols() : A.rows();
}

template <class T>
select::Key key_of(const MV<T>& A, const MV<T>& C, Transpose transA) {
    const std::int64_t n = C.rows(), k = inner<T>(A, transA);
    const char* trans = transA == Transpose::NoTrans ? "N" : (transA == Transpose::Trans ? "T" : "C");
    return {{"form", form_of(n, k)}, {"trans", trans}, {"n", n}, {"k", k}, {"batch", C.batch_size()}};
}

// Correctness only (R3): false means the kernel would throw or answer wrongly. The tile kernels
// are portable SYCL wired only for CUDA (the old reach), take one (n, k, ld, stride) per launch
// (no heterogeneous batch) and put the batch, and triangular its tile count, in grid dimensions
// capped at 65535. syrk is real-only, so ConjTrans is Trans to both kernels (the table, not
// can_run, keeps it on the vendor). Every vendor loop (cuBLAS, rocBLAS, netlib) runs each item
// at the top-level (n, k), a wrong answer for a heterogeneous batch, so no family takes one.
// Shapes are validated before this runs.
template <Backend B, class T>
bool can_run(const SyrkChoice& c, const select::Device& d, const MV<T>& A, const MV<T>& C, Transpose transA) {
    const std::int64_t n = C.rows(), k = inner<T>(A, transA), batch = C.batch_size();
    const Transpose real_trans = transA == Transpose::NoTrans ? Transpose::NoTrans : Transpose::Trans;
    const bool homogeneous = !A.is_heterogeneous() && !C.is_heterogeneous();
    const bool native = B == Backend::CUDA && d.is_gpu && homogeneous && n >= 1 && k >= 1 && batch >= 1 &&
                        batch <= kMaxGridBatch;
    return std::visit(overloaded{
        [&](Gram) {
            return native && backend::detail::syrk_gram_supported<T>(A, C, real_trans, false) &&
                   d.max_wg >= gram_threads(n) && d.slm_budget >= gram_slm_bytes<T>(n);
        },
        [&](Triangular) {
            return native && std::is_same_v<T, float> && d.max_wg >= kTriangularWg &&
                   triangular_groups(n, backend::detail::kTriangularTile) <= kMaxGridTiles;
        },
        [&](Vendor) { return d.has_vendor && homogeneous; },
    }, c);
}

// The native arms compile only for CUDA (the kernels are not instantiated anywhere else), and
// Triangular only for float; can_run never admits the other instantiations.
template <Backend B, class T>
Event launch(Queue& q, const SyrkChoice& c, const MV<T>& A, const MV<T>& C, T alpha, T beta, Uplo uplo,
             Transpose transA) {
    return std::visit(overloaded{
        [&](Gram) -> Event {
            if constexpr (B == Backend::CUDA)
                return backend::detail::syrk_gram_tiles<T, false>(q, A, C, alpha, beta, uplo, transA);
            else
                throw std::logic_error("syrk: gram is wired for CUDA only");
        },
        [&](Triangular) -> Event {
            if constexpr (B == Backend::CUDA && std::is_same_v<T, float>)
                return backend::detail::syrk_triangular_tiles<float>(q, A, C, alpha, beta, uplo, transA);
            else
                throw std::logic_error("syrk: triangular is a float CUDA kernel");
        },
        [&](Vendor) -> Event {
            if constexpr (select::has_library<B>(spec.vendor))
                return backend::syrk_vendor<B, T>(q, A, C, alpha, beta, uplo, transA);
            else select::no_vendor<B, T>(spec);
        },
    }, c);
}

}  // namespace ops::syrk

// syrk takes no workspace (no syrk_buffer_size), so R5 has nothing to size.
template <Backend Back, RealScalar T>
Event syrk(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, const MatrixView<T, MatrixFormat::Dense>& C,
           T alpha, T beta, Uplo uplo, Transpose transA) {
    (void)backend::shape::validate_rank_k<std::invalid_argument>("SYRK", A, C, transA, /*hermitian=*/false);
    // An empty batch launches nothing under any pin, as gemm's does.
    if (C.batch_size() == 0) return ctx.create_event_after_external_work();
    // The coverage row's key, as the old router recorded it: m = n = C's order, k = op(A)'s inner extent.
    const coverage::Shape shape{.m = C.rows(), .n = C.rows(), .k = ops::syrk::inner<T>(A, transA),
                                .batch = A.batch_size(), .transA = transA, .uplo = uplo};
    const select::Key key = ops::syrk::key_of<T>(A, C, transA);
    return select::run<Back, T>(
        ops::syrk::spec, ctx, key, ops::syrk::candidates<T>(),
        [&](const auto& c, const auto& d) { return ops::syrk::can_run<Back, T>(c, d, A, C, transA); }, shape, key,
        [&](const auto& c) { return ops::syrk::launch<Back, T>(ctx, c, A, C, alpha, beta, uplo, transA); });
}

#define SYRK_INSTANTIATE(B_, fp) BATCHLAS_INSTANTIATE_OP(B_, fp, syrk)
BATCHLAS_INSTANTIATE_REAL_ALL_BACKENDS(SYRK_INSTANTIATE)
#undef SYRK_INSTANTIATE

}  // namespace batchlas
