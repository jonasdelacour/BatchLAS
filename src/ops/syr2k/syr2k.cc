// syr2k (flat-kernel-selection.md §4.3, R1; §12): select::run takes the first entry of the nearest
// tuned/syr2k.<device>.txt row that can_run() admits. Triangular is the 128x128x8 tile-masked
// kernel that writes only the requested half of C.

#include <batchlas/backend_config.h>

#include <batchlas/blas/functions/syr2k.hh>

#include <src/ops/level3/batched.hh>
#include <src/ops/syr2k/choice.hh>
#include <src/select/select.hh>
#include <src/backends/level3_shape.hh>
#include <src/ops/syr2k/kernels/syr2k_triangular_tiles.hh>
#include <src/util/template-instantiations.hh>

#include <complex>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <variant>

namespace batchlas {
namespace ops::syr2k {

using select::overloaded;

template <class T>
using MV = MatrixView<T, MatrixFormat::Dense>;

// op(A)'s inner extent: the k of C = alpha (op(A) op(B)^T + op(B) op(A)^T) + beta C.
template <class T>
std::int64_t inner(const MV<T>& A, Transpose transA) {
    return transA == Transpose::NoTrans ? A.cols() : A.rows();
}

template <class T>
select::Key key_of(const MV<T>& A, const MV<T>& C, Transpose transA) {
    return {{"n", C.rows()}, {"k", inner<T>(A, transA)}, {"batch", C.batch_size()}};
}

// Correctness only (R3). The kernel is portable SYCL wired for CUDA and SYCL (design D6), any real T. One launch covers the batch with a single (n, k, ld, stride), so no
// heterogeneous operand. A real ConjTrans is Trans, folded by the launch, so it is no term. The
// launch chunks the batch (grid z); the tile-count term is grid y's own limit. Every vendor loop (cuBLAS,
// rocBLAS, netlib) runs each item at the top-level (n, k), a wrong answer for a heterogeneous
// batch, so the vendor is refused one too.
template <Backend B, class T>
bool can_run(const Syr2kChoice& c, const select::Device& d, const MV<T>& A, const MV<T>& Bm, const MV<T>& C,
             Transpose transA) {
    const std::int64_t n = C.rows(), k = inner<T>(A, transA), batch = C.batch_size();
    const std::int64_t side = (n + backend::detail::kTriangularTile - 1) / backend::detail::kTriangularTile;
    const bool homogeneous = !A.is_heterogeneous() && !Bm.is_heterogeneous() && !C.is_heterogeneous();
    return std::visit(overloaded{
        [&](Triangular) {
            return select::native_gpu_backend<B> && d.is_gpu && d.max_wg >= 256 &&
                   d.slm_budget >= triangular_slm_bytes<T>() &&
                   homogeneous && n >= 1 && k >= 1 && batch >= 1 &&
                   side * (side + 1) / 2 <= d.max_groups_yz;
        },
        [&](Vendor) { return vendor_wired<B, T> && d.has_vendor && homogeneous; },
    }, c);
}

template <Backend B, class T>
Event launch(Queue& q, const Syr2kChoice& c, const MV<T>& A, const MV<T>& Bm, const MV<T>& C, T alpha, T beta,
             Uplo uplo, Transpose transA) {
    return std::visit(overloaded{
        [&](Triangular) -> Event {
            // Instantiated only where can_run can admit it, so no other (B, T) compiles the kernel.
            if constexpr (select::native_gpu_backend<B> && !std::is_same_v<T, std::complex<double>>) {
                const Transpose t = transA == Transpose::ConjTrans ? Transpose::Trans : transA;  // real T
                return level3::for_each_chunk(q, C.batch_size(), level3::grid_chunk<B>(q), [&](int f, int n) {
                    return backend::detail::syr2k_triangular_tiles<T>(q, level3::batch_range(A, f, n),
                                                                          level3::batch_range(Bm, f, n),
                                                                          level3::batch_range(C, f, n), alpha,
                                                                          beta, uplo, t);
                });
            }
            else
                throw std::logic_error("syr2k: triangular reached for a (backend, dtype) can_run refuses");
        },
        [&](Vendor) -> Event {
            if constexpr (vendor_wired<B, T>)
                return backend::syr2k_vendor<B, T>(q, A, Bm, C, alpha, beta, uplo, transA);
            else select::no_vendor<B, T>(spec);
        },
    }, c);
}

}  // namespace ops::syr2k

// syr2k takes no workspace (no syr2k_buffer_size), so R5 has nothing to size.
template <Backend Back, FloatingOrComplexScalar T>
Event syr2k(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, const MatrixView<T, MatrixFormat::Dense>& B,
            const MatrixView<T, MatrixFormat::Dense>& C, T alpha, T beta, Uplo uplo, Transpose transA) {
    backend::shape::validate_rank_2k<std::invalid_argument>("SYR2K", A, B, C, transA, /*hermitian=*/false);
    if (ComplexScalar<T> && transA == Transpose::ConjTrans)
        throw std::invalid_argument("SYR2K: ConjTrans with a complex T is her2k's, not a symmetric update");
    // An empty batch launches nothing under any pin, as syrk's does (the cuBLAS loop faults on one).
    if (C.batch_size() == 0) return ctx.create_event_after_external_work();
    // The coverage row's key: m = n = C's order, k = op(A)'s inner extent.
    const coverage::Shape shape{.m = C.rows(), .n = C.rows(), .k = ops::syr2k::inner<T>(A, transA),
                                .batch = C.batch_size(), .transA = transA, .uplo = uplo};
    const select::Key key = ops::syr2k::key_of<T>(A, C, transA);
    return select::run<Back, T>(
        ops::syr2k::spec, ctx, key, ops::syr2k::candidates<T>(),
        [&](const auto& c, const auto& d) { return ops::syr2k::can_run<Back, T>(c, d, A, B, C, transA); },
        shape, key,
        [&](const auto& c) { return ops::syr2k::launch<Back, T>(ctx, c, A, B, C, alpha, beta, uplo, transA); });
}

#define SYR2K_INSTANTIATE(B_, fp) BATCHLAS_INSTANTIATE_OP(B_, fp, syr2k)
BATCHLAS_INSTANTIATE_SCALAR_ALL_BACKENDS(SYR2K_INSTANTIATE)
#undef SYR2K_INSTANTIATE

}  // namespace batchlas
