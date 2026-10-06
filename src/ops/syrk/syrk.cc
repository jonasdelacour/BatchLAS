// syrk: the whole selection path (docs/design/flat-kernel-selection.md §4.3, rule R1).
// public syrk() -> choose() -> std::visit -> launch. The kernel for a shape is the first
// runnable entry of the nearest row in tuned/syrk.<dtype>.<device>.txt; can_run() below only
// removes entries that cannot run. Gram puts all of C in one tile sized to n (n <= 128) and
// stages A once; Triangular walks 128-wide tiles of the requested triangle only (float).

#include <batchlas/backend_config.h>

#include <batchlas/blas/functions/syrk.hh>
#include <batchlas/no_route.hh>
#include "../../select/vendor.hh"

#include "choice.hh"
#include "../../backends/level3_shape.hh"
#include "../../backends/syrk_gram_tiles.hh"
#include "../../backends/syrk_triangular_tiles.hh"
#include "../../select/select.hh"
#include "../../util/template-instantiations.hh"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <variant>

namespace batchlas {
namespace ops::syrk {

template <class... F>
struct overloaded : F... { using F::operator()...; };
template <class... F>
overloaded(F...) -> overloaded<F...>;

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
    return {{"form", form_of(n, k)}, {"n", n}, {"k", k}, {"batch", C.batch_size()}};
}

// Correctness only (R3): false means the kernel would throw or answer wrongly. The tile kernels
// are portable SYCL wired only for CUDA (the old reach), take one (n, k, ld, stride) per launch
// (no heterogeneous batch), put the batch in a grid dimension capped at 65535, and spell
// A A^T, so a ConjTrans goes to the vendor as before. Shapes are validated before this runs.
template <Backend B, class T>
bool can_run(const SyrkChoice& c, const select::Device& d, const MV<T>& A, const MV<T>& C, Transpose transA) {
    const std::int64_t n = C.rows(), k = inner<T>(A, transA), batch = C.batch_size();
    const bool native = B == Backend::CUDA && d.is_gpu && !A.is_heterogeneous() && !C.is_heterogeneous() &&
                        transA != Transpose::ConjTrans && n >= 1 && k >= 1 && batch >= 1 &&
                        batch <= kMaxGridBatch;
    return std::visit(overloaded{
        [&](Gram) {
            return native && backend::detail::syrk_gram_supported<T>(A, C, transA, false) &&
                   d.max_wg >= gram_threads(n) && d.slm_budget >= gram_slm_bytes<T>(n);
        },
        [&](Triangular) { return native && std::is_same_v<T, float>; },
        [&](Vendor) { return d.has_vendor_blas; },
    }, c);
}

template <Backend B, class T>
SyrkChoice choose(Queue& q, const MV<T>& A, const MV<T>& C, Transpose transA) {
    const select::Device& d = select::device_of<B>(q);
    auto ok = [&](const SyrkChoice& c) { return can_run<B, T>(c, d, A, C, transA); };
    try {
        return select::choose("syrk", select::dtype_name<T>(), d, key_of<T>(A, C, transA), candidates<T>(), ok,
                              rules);
    } catch (const std::runtime_error&) {
        // Vendor-free burn-down reads coverage `miss` rows; a plain runtime_error records none.
        const auto all = candidates<T>();
        if (!select::level3_vendor_available<B> && std::none_of(all.begin(), all.end(), ok))
            select::throw_no_vendor_route<T>(Op::syrk, B, select::kLevel3Library<B>);
        throw;
    }
}

// The coverage row's native flags (§5.6): computed only when coverage records a row.
template <Backend B, class T>
select::NativeFacts native_facts(Queue& q, const MV<T>& A, const MV<T>& C, Transpose transA) {
    if (!coverage::dynamic_enabled()) return {};
    const select::Device& d = select::device_of<B>(q);
    return select::native_facts(candidates<T>(),
                                [&](const SyrkChoice& c) { return can_run<B, T>(c, d, A, C, transA); });
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
            if constexpr (select::level3_vendor_available<B>)
                return backend::syrk_vendor<B, T>(q, A, C, alpha, beta, uplo, transA);
            else
                select::throw_no_vendor_route<T>(Op::syrk, B, select::kLevel3Library<B>);
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
    const auto c = ops::syrk::choose<Back, T>(ctx, A, C, transA);
    // The coverage row's key, as the old router recorded it: m = n = C's order, k = op(A)'s inner extent.
    auto shape = select::square_shape<Back, T>(C.rows(), A.batch_size());
    shape.k = ops::syrk::inner<T>(A, transA);
    shape.uplo = uplo;
    shape.transA = transA;
    const select::Key trace_key = ops::syrk::key_of<T>(A, C, transA);
    select::TraceScope trace("syrk", c, shape, ops::syrk::native_facts<Back, T>(ctx, A, C, transA), trace_key);
    return ops::syrk::launch<Back, T>(ctx, c, A, C, alpha, beta, uplo, transA);
}

#define SYRK_INSTANTIATE(B_, fp) BATCHLAS_INSTANTIATE(sig::syrk<fp>, syrk, B_, fp)

#define SYRK_ALL(B_)            \
    SYRK_INSTANTIATE(B_, float) \
    SYRK_INSTANTIATE(B_, double)

// Keyed on the device family, not the vendor library: without the library the Vendor arm
// compiles to a throw, so the symbol exists in every build with the device.
#if BATCHLAS_HAS_CUDA_BACKEND
SYRK_ALL(Backend::CUDA)
#endif
#if BATCHLAS_HAS_ROCM_BACKEND
SYRK_ALL(Backend::ROCM)
#endif
#if BATCHLAS_HAS_HOST_BACKEND
SYRK_ALL(Backend::NETLIB)
#endif

#undef SYRK_ALL
#undef SYRK_INSTANTIATE

}  // namespace batchlas
