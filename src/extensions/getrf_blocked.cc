// Native batched GETRF, BLOCKED tier: right-looking driver. Per panel -- (P)
// factorise the diagonal panel via getrf_panel_factorize, (S) apply its
// interchanges left and right, (T) solve L11 \ A12, (G) update A22 -= L21 U12.
// This TU must stay in EXTENSIONS_CTA_SOURCES: (P) calls a device symbol from
// getrf_cta.cc, so the two must share one device-code cluster.
// evidence: docs/perf/lu.md#getrf-window-evidence

#include "getrf_native.hh"
#include "lu_laswp.hh"

#include "../sycl/gemm_kernels.hh"
#include "../queue.hh"
#include "../util/template-instantiations.hh"

#include <batchlas/util/mempool.hh>

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace batchlas {
namespace sycl_getrf {

namespace {

// Per-TU tag: gives this cluster its own instantiation of the shared LASWP kernel.
struct GetrfBlockedLaswpTag {};

// Block width, shared by the driver, the workspace query and the debug query. It
// IS the trailing GEMM's k: a multiple of 16, and never below 32 (the wide-scalar
// complex gemm kernel is gated on min_dim >= 32).
template <typename T>
constexpr int getrf_nb_for_type() {
    return 32;
}

template <typename T>
inline int getrf_blocked_nb(int n) {
    return std::max(1, std::min(getrf_nb_for_type<T>(), n));
}

// Three spellings of one left-hand interchange; DeferGather ships. The knob's
// presence latches in a static, but its value is re-read per call, so a harness
// can swap arms mid-run.
enum class LeftLaswp { InLoop, DeferWalk, DeferGather };

inline LeftLaswp getrf_left_laswp_mode() {
    static const bool present = (std::getenv("BATCHLAS_GETRF_LASWP") != nullptr);
    if (!present) return LeftLaswp::DeferGather;
    const char* s = std::getenv("BATCHLAS_GETRF_LASWP");
    if (s == nullptr) return LeftLaswp::DeferGather;
    if (std::strcmp(s, "inloop") == 0) return LeftLaswp::InLoop;
    if (std::strcmp(s, "defer_walk") == 0) return LeftLaswp::DeferWalk;
    return LeftLaswp::DeferGather;
}

// Workspace layout, replayed by both the query and the call. No matrix scratch:
// panel, interchange, solve and update work in place on A. ONE POINTER ARRAY PER
// ROLE, never nullptr and never shared -- init_data_ptr_array rebases from each
// view's own data_ptr()/stride, so a shared array loses the first view's bases.
template <typename T>
struct GetrfBlockedWs {
    Span<int32_t> info;
    Span<T*> p11;
    Span<T*> p12;
    Span<T*> p21;
    Span<T*> p22;
};

template <typename T>
GetrfBlockedWs<T> getrf_blocked_layout(Queue& ctx, BumpAllocator& pool, int batch) {
    const std::size_t b = static_cast<std::size_t>(batch);
    GetrfBlockedWs<T> ws;
    ws.info = pool.allocate<int32_t>(ctx, b);
    ws.p11 = pool.allocate<T*>(ctx, b);
    ws.p12 = pool.allocate<T*>(ctx, b);
    ws.p21 = pool.allocate<T*>(ctx, b);
    ws.p22 = pool.allocate<T*>(ctx, b);
    return ws;
}

}  // namespace

// RouteTable<Op::getrf,T>::preferred() is false everywhere: only a vendor-free
// build or a forced route reaches this driver.
template <> bool getrf_blocked_available<float>()                { return true; }
template <> bool getrf_blocked_available<double>()               { return true; }
template <> bool getrf_blocked_available<std::complex<float>>()  { return true; }
template <> bool getrf_blocked_available<std::complex<double>>() { return true; }

template <typename T>
std::size_t getrf_blocked_buffer_size(Queue& ctx,
                                      const MatrixView<T, MatrixFormat::Dense>& A) {
    const int batch = static_cast<int>(A.batch_size());
    if (batch < 1) return 0;
    return workspace_bytes([&](BumpAllocator& pool) {
        return getrf_blocked_layout<T>(ctx, pool, batch);
    });
}

// Blocking query. Low 16 bits: block width; bits 16-23: the leading panel's leaf
// (1 = local-memory resident, 2 = global); bits 24+: the resolved LeftLaswp mode;
// 0 means absent or degenerate.
template <typename T>
unsigned getrf_blocked_debug_params(Queue& ctx, int n) {
    if (n < 1) return 0u;
    const int nb = getrf_blocked_nb<T>(n);

    const auto dev = ctx.device();
    const std::size_t local_mem = dev.get_property(DeviceProperty::LOCAL_MEM_SIZE);
    const std::size_t budget = (local_mem > 4096) ? (local_mem - 4096) : 0;
    const int ib0 = std::min(nb, n);
    const unsigned leaf = getrf_leaf_fits<T>(n, ib0, budget) ? 1u : 2u;
    const unsigned lmode = static_cast<unsigned>(getrf_left_laswp_mode());

    return (lmode << 24) | (leaf << 16) | static_cast<unsigned>(nb);
}

template <typename T>
Event getrf_blocked_dispatch(Queue& ctx,
                             const MatrixView<T, MatrixFormat::Dense>& A,
                             Span<int64_t> pivots,
                             Span<std::byte> workspace,
                             Span<int32_t> info_out,
                             GetrfTrailingGemm<T> trailing_gemm,
                             GetrfPanelSolveTrsm<T> panel_trsm) {
    // Defaults to the native kernel so a direct caller needs no dispatch
    // dependency; the facade injects the ROUTED gemm instead.
    if (!trailing_gemm) {
        trailing_gemm = [](Queue& c,
                           const MatrixView<T, MatrixFormat::Dense>& ga,
                           const MatrixView<T, MatrixFormat::Dense>& gb,
                           const MatrixView<T, MatrixFormat::Dense>& gc,
                           T galpha, T gbeta, Transpose gta, Transpose gtb,
                           ComputePrecision gp) {
            return sycl_gemm::gemm_custom<T>(c, ga, gb, gc, galpha, gbeta, gta, gtb, gp);
        };
    }

    const int m = static_cast<int>(A.rows());
    const int n = static_cast<int>(A.cols());
    const int batch = static_cast<int>(A.batch_size());

    // Every RouteTable<Op::getrf,T>::supports() gate, re-applied because this entry
    // point is reachable without the table: a forced route the table refuses falls
    // through to automatic(), so a wrong gate here silently measures cuBLAS.
    if (m < 1 || n < 1 || batch < 1) {
        throw std::invalid_argument("getrf_blocked: degenerate extents");
    }
    if (m != n) {
        throw std::invalid_argument(
            "getrf_blocked: A must be square (route_getrf.hh's supports() refuses m != n)");
    }
    if (A.is_heterogeneous()) {
        throw std::invalid_argument("getrf_blocked: heterogeneous batch is not supported");
    }
    const auto dev = ctx.device();
    if (dev.type != DeviceType::GPU) {
        throw std::invalid_argument("getrf_blocked: GPU queues only");
    }
    if (!dev.supports_sub_group_size(32)) {
        throw std::runtime_error(
            "getrf_blocked: device does not offer sub-group size 32, which the panel leaf "
            "requires");
    }
    if (pivots.size() < static_cast<std::size_t>(n) * static_cast<std::size_t>(batch)) {
        throw std::invalid_argument("getrf_blocked: pivot span is shorter than n * batch");
    }
    {
        const std::size_t local_mem = dev.get_property(DeviceProperty::LOCAL_MEM_SIZE);
        const std::size_t budget = (local_mem > 4096) ? (local_mem - 4096) : 0;
        if (getrf_cta_max_n_for_slm<T>(budget) < 1) {
            throw std::runtime_error(
                "getrf_blocked: this device's local-memory budget cannot host the panel "
                "leaf's argmax slots, so the tier is unavailable (route_getrf.hh's "
                "supports() refuses the Blocked arm when cta_max_n is 0)");
        }
    }
    if (!panel_trsm) {
        throw std::invalid_argument(
            "getrf_blocked: the panel-solve trsm seam is empty. Inject the ROUTED "
            "batchlas::trsm (the facade does; a direct caller must too) -- this driver "
            "deliberately has no native fallback for it, so that the router, and not this "
            "file, chooses the trsm arm.");
    }

    const int nb = getrf_blocked_nb<T>(n);
    const LeftLaswp mode = getrf_left_laswp_mode();
    const std::size_t local_mem_all = dev.get_property(DeviceProperty::LOCAL_MEM_SIZE);
    const std::size_t slm_budget = (local_mem_all > 4096) ? (local_mem_all - 4096) : 0;
    const int max_wg = static_cast<int>(dev.get_property(DeviceProperty::MAX_WORK_GROUP_SIZE));

    BumpAllocator pool(workspace);
    auto ws = getrf_blocked_layout<T>(ctx, pool, batch);
    // detail::info_target's rule, inlined: an empty OR SHORT caller span means "not
    // requested" and falls back to pool scratch, and it is THAT span that is zeroed.
    Span<int32_t> info = (info_out.size() >= static_cast<std::size_t>(batch))
                             ? info_out
                             : ws.info;

    // The fill belongs HERE, not after the first panel: getf2_panel_device READS
    // info[b], so this is a real read-after-write. Unguarded, the first panel reads
    // the caller's garbage back and reports a false singularity.
    ctx->fill(info.data(), int32_t(0), static_cast<std::size_t>(batch));
    if (!ctx.in_order()) ctx.wait();

    // PACKED 1-BASED int32, matching the vendor arms bit for bit (getrf_native.hh's
    // PIVOT CONTRACT); the stride is the ORDER, cublas?getrfBatched's PivotArray layout.
    auto piv_i32 = pivots.as_span<int>();
    int* const piv_ptr = piv_i32.data();

    const int ld = A.ld();
    const int stride = A.stride();
    T* const a_ptr = A.data_ptr();

    // The explicit 6-arg constructor, never operator()(Slice, Slice), which would
    // propagate the parent's pointer array; each role passes its own, never nullptr.
    auto sub = [&](int r0, int nr, int c0, int nc, Span<T*> ptrs) {
        return MatrixView<T, MatrixFormat::Dense>(
            a_ptr + static_cast<std::ptrdiff_t>(c0) * ld + r0,
            nr, nc, ld, stride, batch, ptrs.data());
    };

    for (int j0 = 0; j0 < n; j0 += nb) {
        const int ib = std::min(nb, n - j0);   // THE SHORT FINAL PANEL, here only
        const int mp = n - j0;                 // panel height, and A22's height + ib
        const int j2 = j0 + ib;
        const int n2 = n - j2;                 // trailing columns; ZERO on the last panel
        const int m2 = mp - ib;                // trailing rows;    ZERO on the last panel

        // (P) piv_stride is n and piv_base is j0, which is what makes ipiv global
        // 1-based and info a global column index with no fix-up.
        (void)getrf_panel_factorize<T>(ctx,
                                       a_ptr + static_cast<std::ptrdiff_t>(j0) * ld + j0,
                                       ld, stride, mp, ib, batch,
                                       piv_ptr, n, j0, info.data(), nullptr);

        // A caller may build an out-of-order queue, so every dependent edge guards itself.
        if (!ctx.in_order()) ctx.wait();

        // (S-left) The panel's interchanges on the finished columns [0, j0). Skip it
        // and P A = L U silently stops holding: L's finished columns must travel too.
        if (mode == LeftLaswp::InLoop && j0 > 0) {
            (void)lu_native::lu_laswp_launch<GetrfBlockedLaswpTag, T>(
                ctx, a_ptr, ld, stride, /*ncols=*/j0, batch,
                piv_ptr, /*piv_stride=*/n, /*k0=*/j0, /*k1=*/j2, /*forward=*/true);
            if (!ctx.in_order()) ctx.wait();
        }

        if (n2 <= 0) break;   // the short final panel: no trailing work at all

        // (S-right) The same interchanges on the trailing columns, before (T) reads them.
        (void)lu_native::lu_laswp_launch<GetrfBlockedLaswpTag, T>(
            ctx, a_ptr + static_cast<std::ptrdiff_t>(j2) * ld, ld, stride,
            /*ncols=*/n2, batch,
            piv_ptr, /*piv_stride=*/n, /*k0=*/j0, /*k1=*/j2, /*forward=*/true);
        if (!ctx.in_order()) ctx.wait();

        // (T) U12 := L11 \ A12. L11 is unit lower, so the diagonal it holds (which
        // is U's) is not read. alpha comes THIRD in the public trsm.
        const auto L11 = sub(j0, ib, j0, ib, ws.p11);
        const auto A12 = sub(j0, ib, j2, n2, ws.p12);
        (void)panel_trsm(ctx, L11, A12, T(1), Side::Left, Uplo::Lower,
                         Transpose::NoTrans, Diag::Unit);
        if (!ctx.in_order()) ctx.wait();

        if (m2 > 0) {
            // (G) A22 -= L21 U12. Every GEMM here loads the prior value whatever
            // beta is, so A22 must already hold the trailing block -- it does.
            const auto L21 = sub(j2, m2, j0, ib, ws.p21);
            const auto A22 = sub(j2, m2, j2, n2, ws.p22);
            (void)trailing_gemm(ctx, L21, A12, A22, T(-1), T(1),
                                Transpose::NoTrans, Transpose::NoTrans,
                                ComputePrecision::Default);
            if (!ctx.in_order()) ctx.wait();
        }
    }

    // (S-left), DEFERRED. Block r receives the suffix [j0_{r+1}, n) in increasing k.
    // Extents come from ib and j0, never nb: a loop written from the block COUNT
    // reads past the pivot list at n = 129.
    if (mode != LeftLaswp::InLoop) {
        bool done = false;
        if (mode == LeftLaswp::DeferGather) {
            done = lu_native::lu_laswp_deferred_left_launch<GetrfBlockedLaswpTag, T>(
                ctx, a_ptr, ld, stride, batch, piv_ptr, /*piv_stride=*/n,
                n, nb, slm_budget, max_wg);
            if (done && !ctx.in_order()) ctx.wait();
        }
        if (!done) {
            // Fallback, never a throw: taken when the staging tile will not fit SLM.
            for (int c0 = 0; c0 < n; c0 += nb) {
                const int ib = std::min(nb, n - c0);
                const int k0 = c0 + ib;
                if (k0 >= n) break;              // the last block: nothing deferred
                (void)lu_native::lu_laswp_launch<GetrfBlockedLaswpTag, T>(
                    ctx, a_ptr + static_cast<std::ptrdiff_t>(c0) * ld, ld, stride,
                    /*ncols=*/ib, batch,
                    piv_ptr, /*piv_stride=*/n, /*k0=*/k0, /*k1=*/n, /*forward=*/true);
                if (!ctx.in_order()) ctx.wait();
            }
        }
    }

    return ctx.get_event();
}

#define BATCHLAS_GETRF_BLOCKED_INSTANTIATE(T)                                                 \
    template std::size_t getrf_blocked_buffer_size<T>(                                        \
        Queue&, const MatrixView<T, MatrixFormat::Dense>&);                                   \
    template unsigned getrf_blocked_debug_params<T>(Queue&, int);                             \
    template Event getrf_blocked_dispatch<T>(Queue&,                                          \
                                             const MatrixView<T, MatrixFormat::Dense>&,       \
                                             Span<int64_t>, Span<std::byte>, Span<int32_t>,   \
                                             GetrfTrailingGemm<T>, GetrfPanelSolveTrsm<T>);

BATCHLAS_GETRF_BLOCKED_INSTANTIATE(float)
BATCHLAS_GETRF_BLOCKED_INSTANTIATE(double)
BATCHLAS_GETRF_BLOCKED_INSTANTIATE(std::complex<float>)
BATCHLAS_GETRF_BLOCKED_INSTANTIATE(std::complex<double>)

#undef BATCHLAS_GETRF_BLOCKED_INSTANTIATE

}  // namespace sycl_getrf
}  // namespace batchlas
