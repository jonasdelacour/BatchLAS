// gemm: the whole selection path (docs/design/flat-kernel-selection.md §4.3, rule R1;
// flat-kernel-selection-phase3-plan.md §1.3). public gemm() -> choose() -> std::visit -> launch.
// The kernel for a shape is the first runnable entry of the nearest row in
// tuned/gemm.<dtype>.<device>.txt; can_run() below only removes entries that cannot run.
// A heterogeneous batch never reaches choose(): it is split into homogeneous items first, and
// each item makes its own choice.

#include <batchlas/backend_config.h>

#include <batchlas/blas/functions/gemm.hh>
#include <batchlas/blas/dispatch/no_route.hh>
#include <batchlas/blas/dispatch/vendor_available.hh>
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

template <class... F>
struct overloaded : F... { using F::operator()...; };
template <class... F>
overloaded(F...) -> overloaded<F...>;

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
    const bool device = d.is_gpu || !d.has_vendor_blas;
    const bool native = device && precision == ComputePrecision::Default && s.m > 0 && s.n > 0 && s.k > 0 &&
                        A.batch_size() >= 1 && !A.is_heterogeneous() && !B.is_heterogeneous() &&
                        !C.is_heterogeneous();
    return std::visit(overloaded{
        [&](Direct) { return native && d.max_wg >= kDirectWg; },
        [&](Tiled) { return native && d.max_wg >= kTiledWg; },
        [&](Small) {
            const auto mx = std::max({s.m, s.n, s.k});
            const bool nn = ta == Transpose::NoTrans && tb == Transpose::NoTrans;
            return native && !is_complex_v<T> && mx <= kSmallMaxDim && d.max_wg >= small_wg<T>(nn, int(mx));
        },
        [&](const Reg& r) {
            if constexpr (!std::is_same_v<T, float>) return false;
            const auto* cfg = std::find_if(reg_configs.begin(), reg_configs.end(), [&](const RegCfg& g) {
                return g.m == r.m && g.n == r.n && g.k == r.k && g.u == r.u;
            });
            return native && cfg != reg_configs.end() && reg_form(*cfg, ta, tb) && d.max_wg >= cfg->threads();
        },
        [&](const Wide& w) {
            const auto* cfg = std::find_if(wide_configs.begin(), wide_configs.end(),
                                           [&](const WideCfg& g) { return g.m == w.m && g.n == w.n && g.k == w.k; });
            return native && cfg != wide_configs.end() && wide_form<T>(*cfg, ta, tb) && d.max_wg >= cfg->threads();
        },
        [&](Vendor) { return d.has_vendor_blas; },
    }, c);
}

template <Backend B, class T>
GemmChoice choose(Queue& q, const MV<T>& A, const MV<T>& Bm, const MV<T>& C, Transpose ta, Transpose tb,
                  ComputePrecision precision) {
    // A retired variable fails loudly: an old script setting it would otherwise time Auto.
    if (const char* old = settings().selection.gemm_sycl_kernel.get(); old && *old)
        throw std::invalid_argument(std::string("gemm: BATCHLAS_GEMM_SYCL_KERNEL=\"") + old +
                                    "\" is retired; set BATCHLAS_GEMM_ROUTE (its names are aliases there)");
    const select::Device& d = select::device_of<B>(q);
    auto ok = [&](const GemmChoice& c) { return can_run<T>(c, d, A, Bm, C, ta, tb, precision); };
    try {
        return select::choose("gemm", select::dtype_name<T>(), d, key_of<T>(A, Bm, C, ta, tb), candidates<T>(), ok,
                              rules);
    } catch (const std::runtime_error&) {
        // Vendor-free burn-down reads coverage `miss` rows; a plain runtime_error records none.
        const auto all = candidates<T>();
        if (!dispatch::level3_vendor_available<B> && std::none_of(all.begin(), all.end(), ok))
            dispatch::throw_no_vendor_route<T>(dispatch::Op::gemm, B, dispatch::kLevel3Library<B>);
        throw;
    }
}

// The coverage row's native flags (§5.6): computed only when coverage records a row.
template <Backend B, class T>
select::NativeFacts native_facts(Queue& q, const MV<T>& A, const MV<T>& Bm, const MV<T>& C, Transpose ta,
                                 Transpose tb, ComputePrecision precision) {
    if (!dispatch::coverage::dynamic_enabled()) return {};
    const select::Device& d = select::device_of<B>(q);
    return select::native_facts(candidates<T>(),
                                [&](const GemmChoice& c) { return can_run<T>(c, d, A, Bm, C, ta, tb, precision); });
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
            if constexpr (dispatch::level3_vendor_available<B>)
                return backend::gemm_vendor<B, T>(q, A, Bm, C, alpha, beta, ta, tb, precision);
            else
                dispatch::throw_no_vendor_route<T>(dispatch::Op::gemm, B, dispatch::kLevel3Library<B>);
        },
    }, c);
}

}  // namespace ops::gemm

// gemm takes no workspace (no gemm_buffer_size), so R5 has nothing to size.
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
    const auto c = ops::gemm::choose<Back, T>(ctx, A, B, C, transA, transB, precision);
    const auto d = ops::gemm::dims_of<T>(A, B, transA, transB);
    auto shape = select::square_shape<Back, T>(d.m, A.batch_size());
    shape.n = d.n;
    shape.k = d.k;
    shape.transA = transA;
    shape.transB = transB;
    shape.precision = precision;
    const select::Key trace_key = ops::gemm::key_of<T>(A, B, C, transA, transB);
    select::TraceScope trace("gemm", c, shape,
                             ops::gemm::native_facts<Back, T>(ctx, A, B, C, transA, transB, precision), trace_key);
    return ops::gemm::launch<Back, T>(ctx, c, A, B, C, alpha, beta, transA, transB, precision);
}

#define GEMM_INSTANTIATE(B_, fp) BATCHLAS_INSTANTIATE(sig::gemm<fp>, gemm, B_, fp)

#define GEMM_ALL(B_)                          \
    GEMM_INSTANTIATE(B_, float)               \
    GEMM_INSTANTIATE(B_, double)              \
    GEMM_INSTANTIATE(B_, std::complex<float>) \
    GEMM_INSTANTIATE(B_, std::complex<double>)

// Keyed on the device family, not the vendor library: without the library the Vendor arm
// compiles to a throw, so the symbol exists in every build with the device.
#if BATCHLAS_HAS_CUDA_BACKEND
GEMM_ALL(Backend::CUDA)
#endif
#if BATCHLAS_HAS_ROCM_BACKEND
GEMM_ALL(Backend::ROCM)
#endif
#if BATCHLAS_HAS_HOST_BACKEND
GEMM_ALL(Backend::NETLIB)
#endif

#undef GEMM_ALL
#undef GEMM_INSTANTIATE

}  // namespace batchlas
