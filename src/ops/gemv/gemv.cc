// gemv (flat-kernel-selection.md §4.3, R1; docs/design/flat-kernel-selection.md#phase-5-gemv): select::run
// takes the first entry of the nearest tuned/gemv.<dtype>.<device>.txt row that can_run() admits. Direct
// is one work-item per output (bodies 1/2/4), Cta one sub-group per output (bodies 3/5); which body runs
// is derived inside the driver. evidence: docs/perf/gemv.md

#include <batchlas/backend_config.h>

#include <batchlas/blas/functions/gemv.hh>

#include "choice.hh"
#include "../../select/select.hh"
#include "../../sycl/gemv_native.hh"
#include "../../util/template-instantiations.hh"

#include <complex>
#include <cstdint>
#include <variant>

namespace batchlas {
namespace ops::gemv {

using select::overloaded;

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
Event launch(Queue& q, const GemvChoice& c, const MV<T>& A, const VectorView<T>& X, const VectorView<T>& Y,
             T alpha, T beta, Transpose transA) {
    return std::visit(overloaded{
        [&](Cta) { return sycl_gemv::gemv_native_cta<T>(q, A, X, Y, alpha, beta, transA); },
        [&](Direct) { return sycl_gemv::gemv_native_direct<T>(q, A, X, Y, alpha, beta, transA); },
        [&](Vendor) -> Event {
            if constexpr (select::has_library<B>(spec.vendor))
                return backend::gemv_vendor<B, T>(q, A, X, Y, alpha, beta, transA);
            else select::no_vendor<B, T>(spec);
        },
    }, c);
}

}  // namespace ops::gemv

// gemv takes no workspace (no gemv_buffer_size), so R5 has nothing to size.
template <Backend Back, typename T>
Event gemv(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, const VectorView<T>& X, const VectorView<T>& Y,
           T alpha, T beta, Transpose transA) {
    namespace o = ops::gemv;
    // The coverage row's key, as before: m, n are A's stored extents, k repeats m.
    const coverage::Shape shape{.m = A.rows(), .n = A.cols(), .k = A.rows(), .batch = A.batch_size(),
                                .transA = transA};
    const select::Key key = o::key_of<T>(A, transA);
    return select::run<Back, T>(
        o::spec, ctx, key, o::candidates<T>(),
        [&](const auto& c, const auto& d) { return o::can_run<T>(c, d, A, X, Y, transA); }, shape, key,
        [&](const auto& c) { return o::launch<Back, T>(ctx, c, A, X, Y, alpha, beta, transA); });
}

#define GEMV_INSTANTIATE(B_, fp) BATCHLAS_INSTANTIATE_OP(B_, fp, gemv)
BATCHLAS_INSTANTIATE_SCALAR_ALL_BACKENDS(GEMV_INSTANTIATE)
#undef GEMV_INSTANTIATE

}  // namespace batchlas
