// syr2k: the whole selection path (docs/design/flat-kernel-selection.md §4.3, rule R1;
// docs/design/flat-kernel-selection.md §12). public syr2k() -> choose() -> std::visit -> launch.
// The kernel for a shape is the first runnable entry of the nearest row in
// tuned/syr2k.<dtype>.<device>.txt; can_run() below only removes entries that cannot run.
// Triangular is the 128x128x8 tile-masked kernel that writes only the requested half of C.

#include <batchlas/backend_config.h>

#include <batchlas/blas/functions/syr2k.hh>
#include <batchlas/no_route.hh>
#include "../../select/vendor.hh"

#include "choice.hh"
#include "../../select/select.hh"
#include "../../backends/level3_shape.hh"
#include "../../backends/syr2k_triangular_tiles.hh"
#include "../../util/template-instantiations.hh"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <variant>

namespace batchlas {
namespace ops::syr2k {

template <class... F>
struct overloaded : F... { using F::operator()...; };
template <class... F>
overloaded(F...) -> overloaded<F...>;

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

// Correctness only (R3). The kernel is wired for CUDA (the old reach, design D6) and
// instantiated for float. One launch covers the batch with a single (n, k, ld, stride), so no
// heterogeneous operand. Real ConjTrans stays with the vendor, as before (the kernel would read
// it as Trans). The grid terms are the launch's own limits (GridCeilingsAreCanRunTerms). Every
// vendor loop (cuBLAS, rocBLAS, netlib) runs each item at the top-level (n, k), a wrong answer for
// a heterogeneous batch, so the vendor is refused one too.
template <Backend B, class T>
bool can_run(const Syr2kChoice& c, const select::Device& d, const MV<T>& A, const MV<T>& Bm, const MV<T>& C,
             Transpose transA) {
    const std::int64_t n = C.rows(), k = inner<T>(A, transA), batch = C.batch_size();
    const std::int64_t side = (n + backend::detail::kTriangularTile - 1) / backend::detail::kTriangularTile;
    const bool homogeneous = !A.is_heterogeneous() && !Bm.is_heterogeneous() && !C.is_heterogeneous();
    return std::visit(overloaded{
        [&](Triangular) {
            return B == Backend::CUDA && std::is_same_v<T, float> && d.is_gpu && d.max_wg >= 256 &&
                   transA != Transpose::ConjTrans && homogeneous && n >= 1 && k >= 1 && batch >= 1 &&
                   batch <= kMaxGridBatch && side * (side + 1) / 2 <= kMaxGridTiles;
        },
        [&](Vendor) { return d.has_vendor_blas && homogeneous; },
    }, c);
}

template <Backend B, class T>
Syr2kChoice choose(Queue& q, const MV<T>& A, const MV<T>& Bm, const MV<T>& C, Transpose transA) {
    const select::Device& d = select::device_of<B>(q);
    auto ok = [&](const Syr2kChoice& c) { return can_run<B, T>(c, d, A, Bm, C, transA); };
    try {
        return select::choose("syr2k", select::dtype_name<T>(), d, key_of<T>(A, C, transA), candidates<T>(), ok,
                              rules);
    } catch (const std::runtime_error&) {
        // Vendor-free burn-down reads coverage `miss` rows; a plain runtime_error records none.
        const auto all = candidates<T>();
        if (!select::level3_vendor_available<B> && std::none_of(all.begin(), all.end(), ok))
            select::throw_no_vendor_route<T>(Op::syr2k, B, select::kLevel3Library<B>);
        throw;
    }
}

// The coverage row's native flags (§5.6): computed only when coverage records a row.
template <Backend B, class T>
select::NativeFacts native_facts(Queue& q, const MV<T>& A, const MV<T>& Bm, const MV<T>& C, Transpose transA) {
    if (!coverage::dynamic_enabled()) return {};
    const select::Device& d = select::device_of<B>(q);
    return select::native_facts(candidates<T>(),
                                [&](const Syr2kChoice& c) { return can_run<B, T>(c, d, A, Bm, C, transA); });
}

template <Backend B, class T>
Event launch(Queue& q, const Syr2kChoice& c, const MV<T>& A, const MV<T>& Bm, const MV<T>& C, T alpha, T beta,
             Uplo uplo, Transpose transA) {
    return std::visit(overloaded{
        [&](Triangular) -> Event {
            // Instantiated only where can_run can admit it, so no other (B, T) compiles the kernel.
            if constexpr (B == Backend::CUDA && std::is_same_v<T, float>)
                return backend::detail::syr2k_triangular_tiles<float>(q, A, Bm, C, alpha, beta, uplo, transA);
            else
                throw std::logic_error("syr2k: triangular reached for a (backend, dtype) can_run refuses");
        },
        [&](Vendor) -> Event {
            if constexpr (select::level3_vendor_available<B>)
                return backend::syr2k_vendor<B, T>(q, A, Bm, C, alpha, beta, uplo, transA);
            else
                select::throw_no_vendor_route<T>(Op::syr2k, B, select::kLevel3Library<B>);
        },
    }, c);
}

}  // namespace ops::syr2k

// syr2k takes no workspace (no syr2k_buffer_size), so R5 has nothing to size.
template <Backend Back, RealScalar T>
Event syr2k(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, const MatrixView<T, MatrixFormat::Dense>& B,
            const MatrixView<T, MatrixFormat::Dense>& C, T alpha, T beta, Uplo uplo, Transpose transA) {
    backend::shape::validate_rank_2k<std::invalid_argument>("SYR2K", A, B, C, transA, /*hermitian=*/false);
    // An empty batch launches nothing under any pin, as syrk's does (the cuBLAS loop faults on one).
    if (C.batch_size() == 0) return ctx.create_event_after_external_work();
    const auto c = ops::syr2k::choose<Back, T>(ctx, A, B, C, transA);
    // The coverage key the old level-3 recorder wrote: m = n = C's order, k = op(A)'s inner extent.
    auto shape = select::square_shape<Back, T>(C.rows(), C.batch_size());
    shape.k = ops::syr2k::inner<T>(A, transA);
    shape.uplo = uplo;
    shape.side = Side::Left;
    shape.diag = Diag::NonUnit;
    shape.transA = transA;
    const select::Key trace_key = ops::syr2k::key_of<T>(A, C, transA);
    select::TraceScope trace("syr2k", c, shape, ops::syr2k::native_facts<Back, T>(ctx, A, B, C, transA), trace_key);
    return ops::syr2k::launch<Back, T>(ctx, c, A, B, C, alpha, beta, uplo, transA);
}

#define SYR2K_INSTANTIATE(B_, fp) BATCHLAS_INSTANTIATE(sig::syr2k<fp>, syr2k, B_, fp)

// Keyed on the device family, not the vendor library: without the library the Vendor arm
// compiles to a throw, so the symbol exists in every build with the device. Real types only.
#if BATCHLAS_HAS_CUDA_BACKEND
SYR2K_INSTANTIATE(Backend::CUDA, float)
SYR2K_INSTANTIATE(Backend::CUDA, double)
#endif
#if BATCHLAS_HAS_ROCM_BACKEND
SYR2K_INSTANTIATE(Backend::ROCM, float)
SYR2K_INSTANTIATE(Backend::ROCM, double)
#endif
#if BATCHLAS_HAS_HOST_BACKEND
SYR2K_INSTANTIATE(Backend::NETLIB, float)
SYR2K_INSTANTIATE(Backend::NETLIB, double)
#endif

#undef SYR2K_INSTANTIATE

}  // namespace batchlas
