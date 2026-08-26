// Native batched GETRF, BLOCKED tier -- the right-looking driver.
//
// LAPACK ?GETRF's schedule, verbatim, per panel:
//   (P) factorise the diagonal panel A(j0:n, j0:j0+ib) with the CTA DEVICE
//       FUNCTION -- resident if it fits local memory, streamed from global if
//       not (getrf_panel_factorize is the one decision site);
//   (S) apply that panel's row interchanges to the columns LEFT of it, [0, j0),
//       and to the columns RIGHT of it, [j0+ib, n);
//   (T) solve L11 \ A12 with the ROUTED trsm;
//   (G) update A22 -= L21 U12 with the ROUTED gemm.
//
// WHY THIS FILE IS IN EXTENSIONS_CTA_SOURCES rather than
// EXTENSIONS_FACTORIZATION_SOURCES: (P) calls getrf_panel_factorize, a device
// symbol defined in getrf_cta.cc, so the two sources must sit in ONE device-code
// cluster (src/extensions/CMakeLists.txt:29-42: "a source must sit with the
// sources whose device symbols it calls"). getrs_native.cc and getri_blocked.cc
// do NOT, because neither touches the getrf device body -- they follow
// orgqr_blocked.cc. A wrong grouping is a hard `ptxas fatal: Unresolved extern
// function`, never a silent miscompile.
//
// THE CAPABILITY FLAG LIVES HERE, beside the driver, and that placement is
// load-bearing (potrf_native.hh:81-92, geqrf_native.hh:114-121): these are full
// explicit specialisations, so they link from wherever they sit -- and sitting
// anywhere but beside the driver would let a build advertise the tier while THIS
// FILE is absent from EXTENSIONS_CTA_SOURCES or #if 0'd out. Co-located, "the
// flag is true" and "the file is compiled" are the same fact.
//
// ===========================================================================
// THE SHORT FINAL PANEL, EXPLICITLY, because this family has produced exactly
// that failure before with a green suite (the sy2sb stage-1 short-final-panel
// bug). It is carried by ONE std::min -- `ib = min(nb, n - j0)` -- and by the
// rule that EVERY extent below is derived from `ib`, `mp` or `n2` and NEVER from
// `nb`. Specifically:
//   * the panel leaf is handed (mp, ib), so its own kmax = min(mp, ib) = ib;
//   * the interchange sub-list is [j0, j0+ib), not [j0, j0+nb);
//   * n2 = n - j0 - ib is ZERO on the last panel, which skips (S-right), (T) and
//     (G) entirely -- the `if (n2 <= 0) break;` below;
//   * m2 = mp - ib is likewise zero there.
// A test that must straddle the boundary can find it through
// getrf_blocked_debug_params, which reports nb from the SAME pure function this
// driver calls.
//
// ===========================================================================
// SUB-VIEWS ARE BUILT EXPLICITLY, 6-ARG, WITH THE PARENT ld AND stride AND
// batch, and NEVER with operator()(Slice, Slice) -- which propagates the PARENT
// pointer array (matrix.hh:1140, a known open bug, deliberately untouched:
// WP4_POTRF_SPEC_CORRECTIONS.md:1126). The constructor also DEFAULTS stride to
// ld*cols when 0 is passed (matrix.cc:1839-1842), so passing (ptr, ld, stride)
// explicitly is what removes both traps.
//
// AND THE POINTER ARRAY IS A FRESH PER-ROLE ONE FROM THE WORKSPACE, never the
// parent's (which points at the UNSLICED base addresses) and never nullptr.
// geqrf_blocked.cc passes nullptr and survives; this driver does not, and the
// difference is measured rather than argued -- see the layout note below.
//
// ===========================================================================
// WHAT THE TRAILING GEMM WILL GET, measured at the REAL batch and stride
// (experiments/wp6_lu/baseline/routeq_lu_*.csv), because a probe that shrinks
// the batch to save memory cannot ask this question -- gemm_kernels.cc:695-707's
// CTA-count gate multiplies by A.batch_size() and
// can_use_64x64_k16_wide_fast_path reads data_ptr(), ld() AND stride(). For this
// exact shape class (NN, k = nb), vendor-free:
//     float             -> Tiled128x128RegisterK8 at m,n >= 128
//     cfloat, cdouble   -> Tiled64x64RegisterK16Wide at every cell but the
//                          N=2048 tail panel
//     double            -> Tiled16 at ALL 13 shapes
// DOUBLE is the type with no register kernel on this path, INVERTING the
// prediction WP6 inherited, and it is STRUCTURAL: the wide-scalar CTA-count
// relaxation is `if constexpr (is_std_complex_v<T>)`, complex only, and the only
// other wide-scalar door (gemm_kernels.cc:642) needs min_dim >= 256, which
// k = nb can never satisfy. The deficit is bounded (gemm_kernels.cc:606-616
// measures double at 1.01-1.08x of Tiled16, itself ~92% of the 4090's FP64
// ceiling) but there is no WP6-local fix: it needs a transposed/predicated
// wide-scalar kernel and belongs to GEMM.

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

// The per-TU tag that gives this cluster its own instantiation of the shared
// LASWP kernel -- see lu_laswp.hh's opening note on why the kernel is a template
// on a tag rather than a linked symbol.
struct GetrfBlockedLaswpTag {};

// ---------------------------------------------------------------------------
// THE BLOCK WIDTH. ONE pure function, called by the driver, by the workspace
// query and by the debug query, so none of the three can describe a blocking the
// call does not use.
//
// 32 FOR EVERY TYPE, and the constraints that pin it are the trailing GEMM's,
// not the panel's:
//   * A MULTIPLE OF 16, because the trailing update's k IS the block width and
//     every register tile in gemm_kernels.cc steps k in multiples of 8 or 16.
//   * NEVER BELOW 32 FOR COMPLEX. gemm_kernels.cc:700 gates the complex
//     wide-scalar kernel on min_dim >= 32, and min_dim of the trailing NN update
//     IS the block width -- geqrf measured 1.72-2.30x lost at nb = 24
//     (geqrf_blocked.cc:150-160).
//   * NOT WIDER WITHOUT A MEASUREMENT. A wider block shortens the O(n^2)
//     interchange traffic and lengthens the SERIAL panel, which is the part
//     with no batch-independent parallelism to spend; geqrf measured nb = 128 as
//     the WORST width it tested end to end in both builds despite the trailing
//     GEMMs alone preferring it.
//
// Keyed on the TYPE and clamped to the order, NOT keyed on rows:
// sytrd_sy2sb.cc:57-90 is an entire workaround for ormqr keying its table on the
// panel HEIGHT when the dimension that matters is the block count.
//
// IT IS NOT A TUNED NUMBER. preferred() is false, so nothing routes here yet; an
// nb sweep is part of the routing step, and it is one function to change.
template <typename T>
constexpr int getrf_nb_for_type() {
    return 32;
}

template <typename T>
inline int getrf_blocked_nb(int n) {
    return std::max(1, std::min(getrf_nb_for_type<T>(), n));
}

// ---------------------------------------------------------------------------
// HOW THE LEFT-HAND INTERCHANGE IS SCHEDULED. Three spellings of ONE
// composition -- see lu_laswp.hh's identity note -- kept selectable so the A/B
// that chose between them can be re-run rather than re-derived.
//
//   InLoop      LAPACK's own schedule: (S-left) inside the block loop, P-1
//               launches of the per-column walk. What WP6 shipped.
//   DeferWalk   the same walk, re-scheduled to ONE pass after the loop, P-1
//               launches with the suffix list. IDENTICAL TRAFFIC BY
//               CONSTRUCTION -- the control that separates the SCHEDULE change
//               from the KERNEL change, and it is expected to measure 1.00x.
//   DeferGather the deferred pass served by the SLM-staged gather, one launch.
//               The shipped arm.
//
// THE ENVIRONMENT READ COSTS NOTHING WHEN THE KNOB IS ABSENT: the presence test
// is a function-local static, so a production process pays one getenv for its
// lifetime and none per call. When the knob IS set the value is re-read on every
// call, deliberately -- that is what lets an A/B harness flip arms BETWEEN
// INTERLEAVED REPS INSIDE ONE PROCESS, which is the only way two driver
// spellings can be interleaved at all (they are not two routes and not two
// builds).
// ---------------------------------------------------------------------------
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

// The workspace layout. Described once and replayed by both the query and the
// call, per mempool.hh:165-190.
//
// NO MATRIX SCRATCH AT ALL. The panel, the interchange, the solve and the update
// ALL work in place on A: the leaf's tile is local memory, LASWP is an in-place
// exchange, and trsm/gemm write A's own sub-views. So unlike geqrf there is no
// V, T, W1 or W2, and nothing here scales with n.
//
// TWO TERMS, BOTH PER-ITEM AND TINY:
//
//   info      -- the fallback status span for a caller that supplied none, which
//                src/extensions/inv.cc:48 is (it passes no info at all).
//                detail::info_target's rule.
//
//   ptr rolls -- ONE POINTER ARRAY PER ROLE for the four sub-views handed to the
//                routed trsm and gemm. THIS IS A MEASURED REQUIREMENT, NOT
//                DEFENSIVE PROGRAMMING: a MatrixView built by the 6-arg
//                constructor has an EMPTY data_ptrs_ span, and every vendor
//                batched call dereferences A.data_ptrs(ctx)
//                (matrix.hh:1101 -> matrix.cc:2364-2371), which then throws
//                "data_ptrs target is null" -- an abort, not a wrong answer.
//                Found by this driver aborting at float and double n >= 33 while
//                BOTH COMPLEX TYPES PASSED, because at those shapes
//                RouteTable<Op::gemm> sends float/double to the VENDOR and
//                complex to a native tile. The driver cannot know which arm the
//                injected gemm will resolve to, so all four roles get an array
//                unconditionally, at 8 bytes per item per role.
//
//                PER ROLE, never one array shared between two different views:
//                init_data_ptr_array recomputes the target from the view's OWN
//                data_ptr()/stride on every call (matrix.cc:2364-2383), so two
//                roles sharing one array would have the second call rewrite the
//                first's bases. potrf_blocked.cc:333-347 records the same trap.
//                A12 appears in two calls but as the SAME view, so it keeps one
//                array; L11, L21 and A22 get their own.
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

// ---------------------------------------------------------------------------
// THE CAPABILITY FLAG. TRUE for all four types.
//
// That does NOT move any vendor-present traffic:
// RouteTable<Op::getrf,T>::preferred() is still false everywhere, so a
// vendor-present build keeps taking cuBLAS and only a vendor-free build (or an
// explicit BATCHLAS_GETRF_ROUTE) reaches this driver -- route_resolve.hh:60-63.
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// THE BLOCKING QUERY. Low 16 bits: the block width. High 16 bits: WHICH LEAF the
// LEADING panel takes -- 1 = the local-memory-resident leaf, 2 = the global one.
// 0 for the whole word means the driver is absent or the order is degenerate.
//
// Both halves come from the SAME functions the driver calls (getrf_blocked_nb
// and getrf_leaf_fits at the same runtime budget), so this cannot report a
// blocking or a residency the call does not use -- the potrf_cta_launch_params
// discipline. It is NOT optional scaffolding: a test that must straddle a block
// boundary cannot see where the boundary is, and a test that hardcodes the width
// keeps passing after any of its inputs moves while silently no longer testing a
// short final panel.
// ---------------------------------------------------------------------------
template <typename T>
unsigned getrf_blocked_debug_params(Queue& ctx, int n) {
    if (n < 1) return 0u;
    const int nb = getrf_blocked_nb<T>(n);

    const auto dev = ctx.device();
    const std::size_t local_mem = dev.get_property(DeviceProperty::LOCAL_MEM_SIZE);
    const std::size_t budget = (local_mem > 4096) ? (local_mem - 4096) : 0;
    const int ib0 = std::min(nb, n);
    const unsigned leaf = getrf_leaf_fits<T>(n, ib0, budget) ? 1u : 2u;
    // Bits 24+: the LEFT-HAND INTERCHANGE SPELLING this call would resolve, from
    // the SAME function the driver calls. Without it a test that sets
    // BATCHLAS_GETRF_LASWP after the presence flag has already latched would run
    // the DEFAULT arm and pass -- the eleventh blind guard, pre-empted.
    const unsigned lmode = static_cast<unsigned>(getrf_left_laswp_mode());

    return (lmode << 24) | (leaf << 16) | static_cast<unsigned>(nb);
}

// ---------------------------------------------------------------------------
// THE DRIVER.
// ---------------------------------------------------------------------------
template <typename T>
Event getrf_blocked_dispatch(Queue& ctx,
                             const MatrixView<T, MatrixFormat::Dense>& A,
                             Span<int64_t> pivots,
                             Span<std::byte> workspace,
                             Span<int32_t> info_out,
                             GetrfTrailingGemm<T> trailing_gemm,
                             GetrfPanelSolveTrsm<T> panel_trsm) {
    // The GEMM seam defaults to the NATIVE kernel so this TU stands alone: a
    // direct caller (a test, or a harness that must not be silently served by a
    // vendor) gets the native path with no dispatch dependency. The FACADE
    // injects the ROUTED gemm instead -- calling sycl_gemm::gemm_custom from a
    // driver TU unconditionally is WP3 step 16's recorded defect
    // (trsm_native.hh:82-104, fix at level3.cc:186-231): it bypasses
    // RouteTable<Op::gemm> entirely and takes the native kernel even on shapes
    // WP2 measured it losing.
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

    // Every gate RouteTable<Op::getrf,T>::supports() applies to the Blocked arm,
    // re-applied because this entry point is reachable without the table -- and
    // it must be, for potrf_native.hh:126-141's reason: route_resolve.hh:165
    // falls through to automatic() when a forced route is unsupported, so a
    // pinned-route test that is wrong about one gate silently measures cuBLAS.
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
    // Also a supports() gate: the table refuses the Blocked arm unless
    // cta_max_n >= 1, because the panel leaf IS the CTA device function. A device
    // with no usable local memory has no blocked driver either.
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
        // NOT defaulted to a native trsm entry point, unlike the gemm seam above.
        // Picking between trsm's V1 and V2 arms here would re-implement
        // RouteTable<Op::trsm>'s capacity decision in a second place, and doing
        // it unconditionally would be WP3 step 16's defect re-created. A direct
        // caller injects trsm<Backend::CUDA, T> itself, which is still a call no
        // vendor getrf can serve.
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
    // detail::info_target's rule, inlined so this TU does not include
    // src/linalg-impl.hh (potrf_blocked.cc:594-616 is the same inlining): an
    // empty OR SHORT caller span means "not requested" and falls back to pool
    // scratch, and it is THAT span which is zeroed, never info_out. Note the
    // direction -- supplying a span only ever REMOVES a use of the pool draw,
    // never adds one -- so the size above is right in both modes and the draw is
    // UNCONDITIONAL, which is what keeps the query independent of info_out (an
    // argument it is never given).
    Span<int32_t> info = (info_out.size() >= static_cast<std::size_t>(batch))
                             ? info_out
                             : ws.info;

    // THE ZERO PRE-PASS IS NOT OPTIONAL, and for the blocked driver it is what
    // makes first-failure-wins mean anything: getf2_panel_device READS info[b] to
    // decide whether an earlier panel already failed, and a caller's span arrives
    // with garbage (options_api_tests.cc:498,509 seeds -12345).
    //
    // THE GUARD BELONGS HERE, NOT AFTER THE FIRST PANEL. The loop's own guard at
    // the end of step (P) orders the panel ahead of the interchange; it is the
    // fill -> panel edge that was unguarded, and that edge is a genuine
    // read-after-write. On an out-of-order queue the first panel read the
    // caller's pre-call garbage instead of the zero -- 3,743 wrong items of
    // 983,040 measured -- and then wrote it back, so a singular item reported the
    // caller's own sentinel and a caller testing `info[b] != 0` saw a false
    // singularity. In-order queues (the default, sycl-device-queue.hh:254) pay
    // nothing here.
    ctx->fill(info.data(), int32_t(0), static_cast<std::size_t>(batch));
    if (!ctx.in_order()) ctx.wait();

    // PACKED 1-BASED int32, matching the CUDA/ROCm vendor arms bit for bit --
    // see getrf_native.hh's PIVOT CONTRACT. The stride is the ORDER of the whole
    // matrix, which is cublas?getrfBatched's PivotArray layout.
    auto piv_i32 = pivots.as_span<int>();
    int* const piv_ptr = piv_i32.data();

    const int ld = A.ld();
    const int stride = A.stride();
    T* const a_ptr = A.data_ptr();

    // THE POINTER ARRAY IS PASSED, NOT nullptr -- see the layout note. geqrf's
    // sub() passes nullptr and survives only because its trailing GEMM shapes
    // happen to resolve native; this driver's do not, and the failure is an
    // abort at the first vendor-routed call.
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

        // (P) THE PANEL. One device body, two residencies, chosen by capacity.
        // piv_stride is n -- the WHOLE matrix's pivot count, which is the vendor's
        // published layout -- and piv_base is j0, which is what makes the values
        // GLOBAL 1-based ipiv and info the GLOBAL column index without a fix-up.
        (void)getrf_panel_factorize<T>(ctx,
                                       a_ptr + static_cast<std::ptrdiff_t>(j0) * ld + j0,
                                       ld, stride, mp, ib, batch,
                                       piv_ptr, n, j0, info.data(), nullptr);

        // In-order queues give the dependent ordering for free; a caller may
        // construct an out-of-order one (sycl-device-queue.hh:254 defaults
        // in_order = true but it is a parameter), so every dependent boundary in
        // this schedule carries its own guard.
        if (!ctx.in_order()) ctx.wait();

        // (S-left) The panel's interchanges applied to the columns ALREADY
        // FACTORISED, [0, j0). ?GETRF does this (DLASWP(J-1, A, LDA, J, J+JB-1))
        // and skipping it is the classic silently-wrong blocked LU: L's rows are
        // permuted by every later pivot, so P A = L U only holds if the finished
        // columns of L travel with the exchange. It is NOT covered by the leaf,
        // whose tile starts at column j0.
        if (mode == LeftLaswp::InLoop && j0 > 0) {
            (void)lu_native::lu_laswp_launch<GetrfBlockedLaswpTag, T>(
                ctx, a_ptr, ld, stride, /*ncols=*/j0, batch,
                piv_ptr, /*piv_stride=*/n, /*k0=*/j0, /*k1=*/j2, /*forward=*/true);
            if (!ctx.in_order()) ctx.wait();
        }

        if (n2 <= 0) break;   // the short final panel: no trailing work at all

        // (S-right) The same interchanges applied to the trailing columns,
        // BEFORE the solve reads them.
        (void)lu_native::lu_laswp_launch<GetrfBlockedLaswpTag, T>(
            ctx, a_ptr + static_cast<std::ptrdiff_t>(j2) * ld, ld, stride,
            /*ncols=*/n2, batch,
            piv_ptr, /*piv_stride=*/n, /*k0=*/j0, /*k1=*/j2, /*forward=*/true);
        if (!ctx.in_order()) ctx.wait();

        // (T) U12 := L11 \ A12. L11 is the panel's ib x ib diagonal block, UNIT
        // lower -- Diag::Unit, so the diagonal it holds (which is U's) is not
        // read. Routed, injected; note alpha comes THIRD in the public trsm
        // (functions/trsm.hh:100-108), the old spelling being a deleted overload
        // at :121-138 precisely so a stale call cannot silently compile.
        const auto L11 = sub(j0, ib, j0, ib, ws.p11);
        const auto A12 = sub(j0, ib, j2, n2, ws.p12);
        (void)panel_trsm(ctx, L11, A12, T(1), Side::Left, Uplo::Lower,
                         Transpose::NoTrans, Diag::Unit);
        if (!ctx.in_order()) ctx.wait();

        if (m2 > 0) {
            // (G) A22 -= L21 U12. beta = 1, which every GEMM in this tree reads
            // unconditionally (LinearEpilogue::apply is alpha*accum + beta*prior
            // with prior loaded whatever beta is), so A22 must already hold the
            // trailing block -- it does, in place.
            const auto L21 = sub(j2, m2, j0, ib, ws.p21);
            const auto A22 = sub(j2, m2, j2, n2, ws.p22);
            (void)trailing_gemm(ctx, L21, A12, A22, T(-1), T(1),
                                Transpose::NoTrans, Transpose::NoTrans,
                                ComputePrecision::Default);
            if (!ctx.in_order()) ctx.wait();
        }
    }

    // -----------------------------------------------------------------------
    // (S-left), DEFERRED. Column block r receives the transposition suffix
    // [j0_{r+1}, n) in INCREASING k -- exactly the concatenation, in order, of
    // the lists it would have received one panel at a time inside the loop. The
    // identity, and the proof that no later step reads a column below j0, are in
    // lu_laswp.hh's note; the last block receives nothing.
    //
    // EVERY EXTENT IS DERIVED FROM ib AND j0, NEVER FROM nb -- the short final
    // panel rule, restated here because this loop is the one place the driver
    // walks the block list a second time and a `for (r) { ... r*nb + nb ... }`
    // written from the block COUNT would read past the pivot list at n = 129.
    // -----------------------------------------------------------------------
    if (mode != LeftLaswp::InLoop) {
        bool done = false;
        if (mode == LeftLaswp::DeferGather) {
            done = lu_native::lu_laswp_deferred_left_launch<GetrfBlockedLaswpTag, T>(
                ctx, a_ptr, ld, stride, batch, piv_ptr, /*piv_stride=*/n,
                n, nb, slm_budget, max_wg);
            if (done && !ctx.in_order()) ctx.wait();
        }
        if (!done) {
            // The FALLBACK, never a throw: the same composition spelled with the
            // ordinary walk, one launch per column block. Reached when the
            // staging tile will not fit local memory (n above ~6,000 for float,
            // ~1,500 for cdouble on this box) and whenever the control arm is
            // selected.
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

// ---------------------------------------------------------------------------
// Instantiation: PER SCALAR TYPE ONLY, no Backend cross-product, exactly as
// potrf_blocked.cc:838-846 and geqrf_blocked.cc:494-507. Everything that needs a
// Backend arrives injected.
// ---------------------------------------------------------------------------
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
