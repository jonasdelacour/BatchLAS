// gemv: the whole selection path (docs/design/flat-kernel-selection.md §4.3, rule R1;
// docs/design/flat-select-p5/gemv.md). public gemv() -> choose() -> std::visit -> launch.
// The kernel for a shape is the first runnable entry of the nearest row in
// tuned/gemv.<dtype>.<device>.txt; can_run() below only removes entries that cannot run.
// Direct is one work-item per output (bodies 1/2/4), Cta one sub-group per output (bodies 3/5);
// which body runs is derived inside the driver. evidence: docs/perf/gemv.md

#include <batchlas/backend_config.h>

#include <batchlas/blas/functions/gemv.hh>
#include <batchlas/blas/dispatch/no_route.hh>
#include <batchlas/blas/dispatch/vendor_available.hh>

#include "choice.hh"
#include "../../select/select.hh"
#include "../../sycl/gemv_native.hh"
#include "../../util/template-instantiations.hh"

#include <algorithm>
#include <complex>
#include <cstdint>
#include <variant>

namespace batchlas {
namespace ops::gemv {

template <class... F>
struct overloaded : F... { using F::operator()...; };
template <class... F>
overloaded(F...) -> overloaded<F...>;

template <class T>
using MV = MatrixView<T, MatrixFormat::Dense>;

// y's length (out) and x's length (red) swap with transA; A's extents are as stored.
template <class T>
std::int64_t out_len(const MV<T>& A, Transpose t) {
    return t == Transpose::NoTrans ? A.rows() : A.cols();
}
template <class T>
std::int64_t red_len(const MV<T>& A, Transpose t) {
    return t == Transpose::NoTrans ? A.cols() : A.rows();
}

// ConjTrans folds to T: the transposed bodies differ only in a conjugation.
template <class T>
select::Key key_of(const MV<T>& A, Transpose transA) {
    return {{"trans", transA == Transpose::NoTrans ? "N" : "T"},
            {"out", out_len<T>(A, transA)},
            {"red", red_len<T>(A, transA)},
            {"batch", A.batch_size()}};
}

// Correctness only (R3). There is no gemv validator, deliberately (known-defects #1: a throw
// would turn a live silent misuse into a crash), so the native terms carry the agreement checks:
// one launch reads A.batch_size() items of all three views with one (m, n), so x and y must
// match A in batch and length or a native kernel indexes past them. The vendor takes any call,
// as before. m == 0 or n == 0 is legal; the drivers quick-return.
template <class T>
bool can_run(const GemvChoice& c, const select::Device& d, const MV<T>& A, const VectorView<T>& X,
             const VectorView<T>& Y, Transpose transA) {
    const bool native = !A.is_heterogeneous() && A.rows() >= 0 && A.cols() >= 0 && A.batch_size() >= 1 &&
                        X.batch_size() == A.batch_size() && Y.batch_size() == A.batch_size() &&
                        X.size() == red_len<T>(A, transA) && Y.size() == out_len<T>(A, transA);
    if (!device_allows(c, d, transA != Transpose::NoTrans)) return false;
    return std::visit(overloaded{
        [&](Cta) { return native && sycl_gemv::gemv_cta_available<T>(); },
        [&](Direct) { return native && sycl_gemv::gemv_direct_available<T>(); },
        [&](Vendor) { return true; },
    }, c);
}

template <Backend B, class T>
GemvChoice choose(Queue& q, const MV<T>& A, const VectorView<T>& X, const VectorView<T>& Y, Transpose transA) {
    const select::Device& d = select::device_of<B>(q);
    auto ok = [&](const GemvChoice& c) { return can_run<T>(c, d, A, X, Y, transA); };
    try {
        return select::choose("gemv", select::dtype_name<T>(), d, key_of<T>(A, transA), candidates<T>(), ok, rules);
    } catch (const std::runtime_error&) {
        // Vendor-free burn-down reads coverage `miss` rows; a plain runtime_error records none.
        const auto all = candidates<T>();
        if (!dispatch::level3_vendor_available<B> && std::none_of(all.begin(), all.end(), ok))
            dispatch::throw_no_vendor_route<T>(dispatch::Op::gemv, B, dispatch::kLevel3Library<B>);
        throw;
    }
}

// The coverage row's native flags (§5.6): computed only when coverage records a row.
template <Backend B, class T>
select::NativeFacts native_facts(Queue& q, const MV<T>& A, const VectorView<T>& X, const VectorView<T>& Y,
                                 Transpose transA) {
    if (!dispatch::coverage::dynamic_enabled()) return {};
    const select::Device& d = select::device_of<B>(q);
    return select::native_facts(candidates<T>(),
                                [&](const GemvChoice& c) { return can_run<T>(c, d, A, X, Y, transA); });
}

template <Backend B, class T>
Event launch(Queue& q, const GemvChoice& c, const MV<T>& A, const VectorView<T>& X, const VectorView<T>& Y,
             T alpha, T beta, Transpose transA) {
    return std::visit(overloaded{
        [&](Cta) { return sycl_gemv::gemv_native_cta<T>(q, A, X, Y, alpha, beta, transA); },
        [&](Direct) { return sycl_gemv::gemv_native_direct<T>(q, A, X, Y, alpha, beta, transA); },
        [&](Vendor) -> Event {
            if constexpr (dispatch::level3_vendor_available<B>)
                return backend::gemv_vendor<B, T>(q, A, X, Y, alpha, beta, transA);
            else
                dispatch::throw_no_vendor_route<T>(dispatch::Op::gemv, B, dispatch::kLevel3Library<B>);
        },
    }, c);
}

}  // namespace ops::gemv

// gemv takes no workspace (no gemv_buffer_size), so R5 has nothing to size.
template <Backend Back, typename T>
Event gemv(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, const VectorView<T>& X, const VectorView<T>& Y,
           T alpha, T beta, Transpose transA) {
    const auto c = ops::gemv::choose<Back, T>(ctx, A, X, Y, transA);
    // The coverage row's key, as before: m, n are A's stored extents, k repeats m.
    auto shape = select::square_shape<Back, T>(A.rows(), A.batch_size());
    shape.n = A.cols();
    shape.transA = transA;
    shape.is_gpu = ctx.device().type == DeviceType::GPU;
    shape.heterogeneous_batch = A.is_heterogeneous();
    const select::Key trace_key = ops::gemv::key_of<T>(A, transA);
    select::TraceScope trace("gemv", c, shape, ops::gemv::native_facts<Back, T>(ctx, A, X, Y, transA), trace_key);
    return ops::gemv::launch<Back, T>(ctx, c, A, X, Y, alpha, beta, transA);
}

#define GEMV_INSTANTIATE(B_, fp) BATCHLAS_INSTANTIATE(sig::gemv<fp>, gemv, B_, fp)

#define GEMV_ALL(B_)                          \
    GEMV_INSTANTIATE(B_, float)               \
    GEMV_INSTANTIATE(B_, double)              \
    GEMV_INSTANTIATE(B_, std::complex<float>) \
    GEMV_INSTANTIATE(B_, std::complex<double>)

// Keyed on the device family, not the vendor library: without the library the Vendor arm
// compiles to a throw, so the symbol exists in every build with the device.
#if BATCHLAS_HAS_CUDA_BACKEND
GEMV_ALL(Backend::CUDA)
#endif
#if BATCHLAS_HAS_ROCM_BACKEND
GEMV_ALL(Backend::ROCM)
#endif
#if BATCHLAS_HAS_HOST_BACKEND
GEMV_ALL(Backend::NETLIB)
#endif

#undef GEMV_ALL
#undef GEMV_INSTANTIATE

}  // namespace batchlas
