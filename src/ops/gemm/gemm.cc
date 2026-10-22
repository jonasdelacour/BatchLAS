// gemm (flat-kernel-selection.md §4.3, R1; flat-kernel-selection-phase3-plan.md §1.3): select::run takes
// the first entry of the nearest tuned/gemm.<device>.txt row that can_run() admits. A
// heterogeneous batch never reaches choose(): it is split into homogeneous items first, and each item
// makes its own choice.

#include <batchlas/backend_config.h>

#include <batchlas/blas/functions/gemm.hh>
#include <batchlas/settings.hh>

#include <src/ops/gemm/choice.hh>
#include <src/select/select.hh>
#include <src/backends/gemm_heterogeneous.hh>
#include <src/ops/gemm/gemm_kernels.hh>
#include <src/util/template-instantiations.hh>

#include <algorithm>
#include <complex>
#include <cstdint>
#include <limits>
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
    auto op = [](Transpose t) { return t == Transpose::NoTrans ? 0 : t == Transpose::Trans ? 1 : 2; };
    return wide_serves<T>(c, op(ta), op(tb));
}

inline bool reg_form(const RegCfg& c, Transpose ta, Transpose tb) {
    const bool a = ta != Transpose::NoTrans, b = tb != Transpose::NoTrans;  // float: C is T
    return (!a && !b && c.forms.nn) || (!a && b && c.forms.nt) || (a && !b && c.forms.tn) || (a && b && c.forms.tt);
}

// Correctness only (R3): false means the launcher would throw or answer wrongly. Every register
// and wide tile hard-wires its transpose form, so a form it does not instantiate is refused
// here (a silent wrong answer otherwise). evidence: docs/perf/gemm.md#choices-flat-selection-p34
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
    // A batch past the grid runs as slices (launch), so the batch is no term of any family.
    return std::visit(overloaded{
        [&](Direct) { return native && d.max_wg >= kDirectWg; },
        [&](Tiled) { return native && d.max_wg >= kTiledWg; },
        [&](Small) {
            const bool nn = ta == Transpose::NoTrans && tb == Transpose::NoTrans;
            return native && small_fits<T>(d, nn, std::max({s.m, s.n, s.k}));
        },
        [&](const Reg& r) {
            if constexpr (!std::is_same_v<T, float>) return false;
            const auto* cfg = std::find_if(reg_configs.begin(), reg_configs.end(), [&](const RegCfg& g) {
                return g.m == r.m && g.n == r.n && g.k == r.k && g.u == r.u;
            });
            return native && cfg != reg_configs.end() && reg_form(*cfg, ta, tb) && d.max_wg >= cfg->threads() &&
                   d.slm_budget_for(cfg->threads()) >= cfg->slm_bytes();
        },
        [&](const Wide& w) {
            const auto* cfg = std::find_if(wide_configs.begin(), wide_configs.end(),
                                           [&](const WideCfg& g) { return g.m == w.m && g.n == w.n && g.k == w.k; });
            return native && cfg != wide_configs.end() && wide_form<T>(*cfg, ta, tb) && d.max_wg >= cfg->threads() &&
                   d.slm_budget_for(cfg->threads()) >= wide_slm_bytes<T>(*cfg);
        },
        [&](const SplitK& sk) {
            // A launch slice puts slice * s in grid z (an int group id), slice = min(batch, max_groups_yz / s);
            // one item's partials are int-indexed; each slice's partials lease must fit the scratch budget
            // (past it the arena throws std::bad_alloc).
            const auto* cfg = std::find_if(wide_configs.begin(), wide_configs.end(),
                                           [&](const WideCfg& g) { return g.m == sk.m && g.n == sk.n && g.k == sk.k; });
            const bool slices = std::find(kSplitKSlices.begin(), kSplitKSlices.end(), sk.s) != kSplitKSlices.end();
            const std::int64_t slice = std::min<std::int64_t>(A.batch_size(), d.max_groups_yz / std::max(1, sk.s));
            return native && slices && slice >= 1 && slice * sk.s <= std::numeric_limits<int>::max() &&
                   cfg != wide_configs.end() && splitk_host(*cfg) && wide_form<T>(*cfg, ta, tb) &&
                   d.max_wg >= cfg->threads() && d.slm_budget_for(cfg->threads()) >= wide_slm_bytes<T>(*cfg) &&
                   splitk_ld(s.m) * s.n <= std::numeric_limits<int>::max() &&
                   splitk_item_bytes<T>(s.m, s.n, sk.s) <= d.scratch_budget / slice;
        },
        [&](Vendor) { return d.has_vendor; },
    }, c);
}

template <Backend B, class T>
Event launch_slice(Queue& q, const GemmChoice& c, const MV<T>& A, const MV<T>& Bm, const MV<T>& C, T alpha, T beta,
                   Transpose ta, Transpose tb, ComputePrecision precision) {
    return std::visit(overloaded{
        [&](Direct) { return sycl_gemm::gemm_direct<T>(q, A, Bm, C, alpha, beta, ta, tb); },
        [&](Tiled) { return sycl_gemm::gemm_tiled<T>(q, A, Bm, C, alpha, beta, ta, tb); },
        [&](Small) { return sycl_gemm::gemm_small<T>(q, A, Bm, C, alpha, beta, ta, tb); },
        [&](const Reg& r) -> Event {
            if constexpr (std::is_same_v<T, float>)
                return sycl_gemm::gemm_reg(q, r.m, r.n, r.k, r.u, A, Bm, C, alpha, beta, ta, tb);
            else throw batchlas::unsupported("gemm: reg is float only");
        },
        [&](const Wide& w) { return sycl_gemm::gemm_wide<T>(q, w.m, w.n, w.k, A, Bm, C, alpha, beta, ta, tb); },
        [&](const SplitK& sk) {
            return sycl_gemm::gemm_splitk<T>(q, sk.m, sk.n, sk.k, sk.s, A, Bm, C, alpha, beta, ta, tb);
        },
        [&](Vendor) -> Event {
            if constexpr (select::has_library<B>(spec.vendor))
                return backend::gemm_vendor<B, T>(q, A, Bm, C, alpha, beta, ta, tb, precision);
            else select::no_vendor<B, T>(spec);
        },
    }, c);
}

// direct, tiled, reg and wide put the batch in grid z, splitk batch * s: a batch past the grid
// runs as slices. small is 1-D and the vendor loops itself, so both take the whole batch.
template <Backend B, class T>
Event launch(Queue& q, const GemmChoice& c, const MV<T>& A, const MV<T>& Bm, const MV<T>& C, T alpha, T beta,
             Transpose ta, Transpose tb, ComputePrecision precision) {
    if (std::holds_alternative<Small>(c) || std::holds_alternative<Vendor>(c))
        return launch_slice<B, T>(q, c, A, Bm, C, alpha, beta, ta, tb, precision);
    const std::int64_t groups_per_item = std::holds_alternative<SplitK>(c) ? std::get<SplitK>(c).s : 1;
    return select::over_batch_slices(
        q, A.batch_size(),
        [&](Queue& sq, int first, int count) {
            return launch_slice<B, T>(sq, c, A.batch_range(first, count), Bm.batch_range(first, count),
                                      C.batch_range(first, count), alpha, beta, ta, tb, precision);
        },
        groups_per_item);
}

}  // namespace ops::gemm

// gemm takes no caller workspace (no gemm_buffer_size): splitk leases its partials from the Queue.
template <Backend Back, typename T>
Event gemm(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, const MatrixView<T, MatrixFormat::Dense>& B,
           const MatrixView<T, MatrixFormat::Dense>& C, T alpha, T beta, Transpose transA, Transpose transB,
           ComputePrecision precision) {
    if (A.is_heterogeneous() || B.is_heterogeneous() || C.is_heterogeneous()) {
        // Items are homogeneous by construction, so the recursion is one level deep.
        return backend::detail::gemm_heterogeneous_loop<T>(
            ctx, A, B, C, beta, transA, transB,
            [&](const MatrixView<T, MatrixFormat::Dense>& a, const MatrixView<T, MatrixFormat::Dense>& b,
                const MatrixView<T, MatrixFormat::Dense>& c) {
                return gemm<Back, T>(ctx, a, b, c, alpha, beta, transA, transB, precision);
            });
    }
    ops::gemm::validate<T>(A, B, C, transA, transB);
    // An empty batch launches nothing under any pin.
    if (A.batch_size() == 0) return ctx.create_event_after_external_work();
    const auto d = ops::gemm::dims_of<T>(A, B, transA, transB);
    // No family takes a zero extent (vendor-free there was no route): an empty C is a no-op and
    // k = 0 is C := beta C, written as zeros at beta 0 so a NaN in C does not survive.
    if (d.m == 0 || d.n == 0 || (d.k == 0 && beta == T(1))) return ctx.create_event_after_external_work();
    if (d.k == 0) return beta == T(0) ? C.fill_zeros(ctx) : scale(ctx, beta, C);
    const coverage::Shape shape{.m = d.m, .n = d.n, .k = d.k, .batch = A.batch_size(), .transA = transA,
                                .transB = transB};
    const select::Key key = ops::gemm::key_of<T>(A, B, C, transA, transB);
    return select::run<Back, T>(
        ops::gemm::spec, ctx, key, ops::gemm::candidates<T>(),
        [&](const auto& c, const auto& dev) {
            return ops::gemm::can_run<T>(c, dev, A, B, C, transA, transB, precision);
        },
        shape, key,
        [&](const auto& c) {
            return ops::gemm::launch<Back, T>(ctx, c, A, B, C, alpha, beta, transA, transB, precision);
        });
}

#define GEMM_INSTANTIATE(B_, fp) BATCHLAS_INSTANTIATE_OP(B_, fp, gemm)
BATCHLAS_INSTANTIATE_SCALAR_ALL_BACKENDS(GEMM_INSTANTIATE)
#undef GEMM_INSTANTIATE

}  // namespace batchlas
