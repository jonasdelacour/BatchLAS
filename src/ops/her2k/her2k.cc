// her2k (flat-kernel-selection.md §4.3, §12): select::run takes the first entry of the nearest
// tuned/her2k.<device>.txt row that can_run() admits (src/ops/her2k/route.hh). Fold is one gemm of
// alpha op(A) op(B)^H over the whole n x n product, folded with its own conjugate transpose into the
// uplo triangle; Vendor is the per-item cublas?her2k / cblas_?her2k loop.

#include <batchlas/backend_config.h>

#include <batchlas/blas/functions/her2k.hh>

#include <src/backends/level3_shape.hh>
#include <src/ops/her2k/route.hh>
#include <src/ops/level3/batched.hh>
#include <src/ops/syr2k/kernels/syr2k_triangular_tiles.hh>
#include <src/select/select.hh>
#include <src/util/template-instantiations.hh>

#include <complex>
#include <stdexcept>
#include <type_traits>
#include <variant>

namespace batchlas {
namespace ops::her2k {

template <Backend B, class T>
Event launch(Queue& q, const Her2kChoice& c, const MV<T>& A, const MV<T>& Bm, const MV<T>& C, T alpha,
             float_t<T> beta, Uplo uplo, Transpose transA) {
    return std::visit(select::overloaded{
        [&](Triangular) -> Event {
            if constexpr (select::native_gpu_backend<B> && std::is_same_v<T, std::complex<float>>)
                return level3::for_each_chunk(q, C.batch_size(), level3::grid_chunk<B>(q), [&](int f, int n) {
                    return backend::detail::syr2k_triangular_tiles<T, /*Conjugate=*/true>(
                        q, level3::batch_range(A, f, n), level3::batch_range(Bm, f, n), level3::batch_range(C, f, n),
                        alpha, T(beta), uplo, transA);
                });
            else
                throw std::logic_error("her2k: triangular is a complex<float> native-GPU kernel");  // can_run refuses
        },
        [&](Fold) -> Event {
            if constexpr (select::native_gpu_backend<B> || B == Backend::ROCM)
                return level3::gemm_fold<B, T, /*TwoSided=*/true>(q, A, Bm, C, alpha, beta, uplo, transA);
            else
                throw std::logic_error("her2k: fold is wired for GPU backends only");  // can_run refuses it
        },
        [&](Vendor) -> Event {
            if constexpr (B != Backend::ROCM && select::has_library<B>(spec.vendor))
                return backend::her2k_vendor<B, T>(q, A, Bm, C, alpha, beta, uplo, transA);
            else select::no_vendor<B, T>(spec);
        },
    }, c);
}

}  // namespace ops::her2k

// her2k takes no caller workspace; Fold leases its scratch from the queue's arena.
template <Backend Back, ComplexScalar T>
Event her2k(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, const MatrixView<T, MatrixFormat::Dense>& B,
            const MatrixView<T, MatrixFormat::Dense>& C, T alpha, float_t<T> beta, Uplo uplo, Transpose transA) {
    backend::shape::validate_rank_2k<std::invalid_argument>("HER2K", A, B, C, transA, /*hermitian=*/true);
    if (C.batch_size() == 0 || C.rows() == 0) return ctx.create_event_after_external_work();
    const coverage::Shape shape{.m = C.rows(), .n = C.rows(), .k = ops::her2k::inner<T>(A, transA),
                                .batch = C.batch_size(), .transA = transA, .uplo = uplo};
    const select::Key key = ops::her2k::key_of<T>(A, C, transA);
    return select::run<Back, T>(
        ops::her2k::spec, ctx, key, ops::her2k::candidates<T>(),
        [&](const auto& c, const auto& d) { return ops::her2k::can_run<Back, T>(c, d, ctx, A, B, C); }, shape,
        key, [&](const auto& c) { return ops::her2k::launch<Back, T>(ctx, c, A, B, C, alpha, beta, uplo, transA); });
}

#define HER2K_INSTANTIATE(B_, fp) BATCHLAS_INSTANTIATE_OP(B_, fp, her2k)
BATCHLAS_IF_CUDA(BATCHLAS_FOR_EACH_COMPLEX_TYPE_1(HER2K_INSTANTIATE, Backend::CUDA))
BATCHLAS_IF_SYCL(BATCHLAS_FOR_EACH_DEVICE_COMPLEX_TYPE_1(HER2K_INSTANTIATE, Backend::SYCL))
BATCHLAS_IF_ROCM(BATCHLAS_FOR_EACH_COMPLEX_TYPE_1(HER2K_INSTANTIATE, Backend::ROCM))
BATCHLAS_IF_HOST(BATCHLAS_FOR_EACH_COMPLEX_TYPE_1(HER2K_INSTANTIATE, Backend::NETLIB))
#undef HER2K_INSTANTIATE

}  // namespace batchlas
