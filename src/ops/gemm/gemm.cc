// gemm (flat-kernel-selection.md §4.3, R1; flat-kernel-selection-phase3-plan.md §1.3): select::run takes
// the first entry of the nearest tuned/gemm.<dtype>.<device>.txt row that can_run() admits. A
// heterogeneous batch never reaches choose(): it is split into homogeneous items first, and each item
// makes its own choice.

#include <batchlas/backend_config.h>

#include <batchlas/blas/functions/gemm.hh>
#include <batchlas/settings.hh>

#include "choice.hh"
#include "../../select/select.hh"
#include "../../backends/gemm_heterogeneous.hh"
#include "../../sycl/gemm_kernels.hh"
#include "../../util/template-instantiations.hh"

#include <algorithm>
#include <complex>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <variant>

namespace batchlas {
namespace ops::gemm {

using select::overloaded;

template <class T>
using MV = MatrixView<T, MatrixFormat::Dense>;

struct Dims {
    std::int64_t m, n, k;
};

template <class T>
Dims dims_of(const MV<T>& A, const MV<T>& B, Transpose ta, Transpose tb) {
    const auto [m, k] = get_effective_dims(A, ta);
    const auto [kb, n] = get_effective_dims(B, tb);
    static_cast<void>(kb);
    return {m, n, k};
}

// One launch covers the batch with one (m, n, k, ld, stride), so the three views must agree.
template <class T>
void validate(const MV<T>& A, const MV<T>& B, const MV<T>& C, Transpose ta, Transpose tb) {
    const auto [m, k] = get_effective_dims(A, ta);
    const auto [kb, n] = get_effective_dims(B, tb);
    if (A.batch_size() != B.batch_size() || A.batch_size() != C.batch_size() || k != kb || C.rows() != m ||
        C.cols() != n || m < 0 || n < 0 || k < 0)
        throw batchlas::invalid_argument("GEMM: incompatible matrix dimensions");
}

template <class T>
bool contiguous16(const MV<T>& M) {
    const bool aligned = reinterpret_cast<std::uintptr_t>(M.data_ptr()) % 16 == 0;
    return aligned && M.ld() == M.rows() && (M.batch_size() == 1 || M.stride() == M.ld() * M.cols());
}

// C folds to T for a real scalar (conj is the identity): one row serves both.
template <class T>
std::string_view trans_word(Transpose t) {
    if (t == Transpose::NoTrans) return "N";
    return (!is_complex_v<T> || t == Transpose::Trans) ? "T" : "C";
}

template <class T>
select::Key key_of(const MV<T>& A, const MV<T>& B, const MV<T>& C, Transpose ta, Transpose tb) {
    const Dims d = dims_of<T>(A, B, ta, tb);
    const bool packed = contiguous16<T>(A) && contiguous16<T>(B) && contiguous16<T>(C);
    return {{"ta", trans_word<T>(ta)}, {"tb", trans_word<T>(tb)}, {"layout", packed ? "packed" : "strided"},
            {"m", d.m}, {"n", d.n}, {"k", d.k}, {"batch", A.batch_size()}};
}

// Does a wide config instantiate this form? A real Trans is served by a ConjTrans instantiation.
template <class T>
bool wide_form(const WideCfg& c, Transpose ta, Transpose tb) {
    auto conj_like = [](Transpose t) { return t == Transpose::ConjTrans || (!is_complex_v<T> && t == Transpose::Trans); };
    const bool na = ta == Transpose::NoTrans, nb = tb == Transpose::NoTrans;
    return (c.nn && na && nb) || (c.cn && conj_like(ta) && nb) || (c.nc && na && conj_like(tb));
}

inline bool reg_form(const RegCfg& c, Transpose ta, Transpose tb) {
    const bool a = ta != Transpose::NoTrans, b = tb != Transpose::NoTrans;  // float: C is T
    return (!a && !b && c.forms.nn) || (!a && b && c.forms.nt) || (a && !b && c.forms.tn) || (a && b && c.forms.tt);
}

// Correctness only (R3): false means the launcher would throw or answer wrongly. Every register
// and wide tile hard-wires its transpose form, so a form it does not instantiate is refused
// here; before this, 18 NN-only variants silently computed NN on a transposed call.
template <class T>
bool can_run(const GemmChoice& c, const select::Device& d, const MV<T>& A, const MV<T>& B, const MV<T>& C,
             Transpose ta, Transpose tb, ComputePrecision precision) {
    const Dims s = dims_of<T>(A, B, ta, tb);
    // The kernels also run on a host SYCL device, but a CPU with a host BLAS keeps it (maintainer
    // decision, plan §13); without one (a vendor-free host queue) they are its only gemm, as before.
    const bool device = d.is_gpu || !d.has_vendor;
    const bool native = device && precision == ComputePrecision::Default && s.m > 0 && s.n > 0 && s.k > 0 &&
                        A.batch_size() >= 1 && !A.is_heterogeneous() && !B.is_heterogeneous() &&
                        !C.is_heterogeneous();
    const bool grid = native && A.batch_size() <= kMaxGridBatch;  // every 3-D launch but small's
    return std::visit(overloaded{
        [&](Direct) { return grid && d.max_wg >= kDirectWg; },
        [&](Tiled) { return grid && d.max_wg >= kTiledWg; },
        [&](Small) {
            const bool nn = ta == Transpose::NoTrans && tb == Transpose::NoTrans;
            return native && small_fits<T>(d, nn, std::max({s.m, s.n, s.k}));
        },
        [&](const Reg& r) {
            if constexpr (!std::is_same_v<T, float>) return false;
            const auto* cfg = std::find_if(reg_configs.begin(), reg_configs.end(), [&](const RegCfg& g) {
                return g.m == r.m && g.n == r.n && g.k == r.k && g.u == r.u;
            });
            return grid && cfg != reg_configs.end() && reg_form(*cfg, ta, tb) && d.max_wg >= cfg->threads();
        },
        [&](const Wide& w) {
            const auto* cfg = std::find_if(wide_configs.begin(), wide_configs.end(),
                                           [&](const WideCfg& g) { return g.m == w.m && g.n == w.n && g.k == w.k; });
            return grid && cfg != wide_configs.end() && wide_form<T>(*cfg, ta, tb) && d.max_wg >= cfg->threads();
        },
        [&](Vendor) { return d.has_vendor; },
    }, c);
}

template <Backend B, class T>
Event launch(Queue& q, const GemmChoice& c, const MV<T>& A, const MV<T>& Bm, const MV<T>& C, T alpha, T beta,
             Transpose ta, Transpose tb, ComputePrecision precision) {
    return std::visit(overloaded{
        [&](Direct) { return sycl_gemm::gemm_direct<T>(q, A, Bm, C, alpha, beta, ta, tb); },
        [&](Tiled) { return sycl_gemm::gemm_tiled<T>(q, A, Bm, C, alpha, beta, ta, tb); },
        [&](Small) -> Event {
            if constexpr (!is_complex_v<T>) return sycl_gemm::gemm_small<T>(q, A, Bm, C, alpha, beta, ta, tb);
            else throw batchlas::unsupported("gemm: small has no complex instantiation");
        },
        [&](const Reg& r) -> Event {
            if constexpr (std::is_same_v<T, float>)
                return sycl_gemm::gemm_reg(q, r.m, r.n, r.k, r.u, A, Bm, C, alpha, beta, ta, tb);
            else throw batchlas::unsupported("gemm: reg is float only");
        },
        [&](const Wide& w) { return sycl_gemm::gemm_wide<T>(q, w.m, w.n, w.k, A, Bm, C, alpha, beta, ta, tb); },
        [&](Vendor) -> Event {
            if constexpr (select::has_library<B>(spec.vendor))
                return backend::gemm_vendor<B, T>(q, A, Bm, C, alpha, beta, ta, tb, precision);
            else select::no_vendor<B, T>(spec);
        },
    }, c);
}

}  // namespace ops::gemm

// gemm takes no workspace (no gemm_buffer_size), so R5 has nothing to size.
template <Backend Back, typename T>
Event gemm(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, const MatrixView<T, MatrixFormat::Dense>& B,
           const MatrixView<T, MatrixFormat::Dense>& C, T alpha, T beta, Transpose transA, Transpose transB,
           ComputePrecision precision) {
    namespace o = ops::gemm;
    if (A.is_heterogeneous() || B.is_heterogeneous() || C.is_heterogeneous()) {
        // Items are homogeneous by construction, so the recursion is one level deep.
        return backend::detail::gemm_heterogeneous_loop<T>(
            ctx, A, B, C, beta, transA, transB,
            [&](const MatrixView<T, MatrixFormat::Dense>& a, const MatrixView<T, MatrixFormat::Dense>& b,
                const MatrixView<T, MatrixFormat::Dense>& c) {
                return gemm<Back, T>(ctx, a, b, c, alpha, beta, transA, transB, precision);
            });
    }
    o::validate<T>(A, B, C, transA, transB);
    // An empty batch launches nothing under any pin, as the old native range did.
    if (A.batch_size() == 0) return ctx.create_event_after_external_work();
    const auto d = o::dims_of<T>(A, B, transA, transB);
    const coverage::Shape shape{.m = d.m, .n = d.n, .k = d.k, .batch = A.batch_size(), .transA = transA,
                                .transB = transB};
    const select::Key key = o::key_of<T>(A, B, C, transA, transB);
    return select::run<Back, T>(
        o::spec, ctx, key, o::candidates<T>(),
        [&](const auto& c, const auto& dev) { return o::can_run<T>(c, dev, A, B, C, transA, transB, precision); },
        shape, key,
        [&](const auto& c) { return o::launch<Back, T>(ctx, c, A, B, C, alpha, beta, transA, transB, precision); });
}

#define GEMM_INSTANTIATE(B_, fp) BATCHLAS_INSTANTIATE_OP(B_, fp, gemm)
BATCHLAS_INSTANTIATE_SCALAR_ALL_BACKENDS(GEMM_INSTANTIATE)
#undef GEMM_INSTANTIATE

}  // namespace batchlas
