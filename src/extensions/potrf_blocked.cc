// Native batched POTRF, WP4 Phase 2: the BLOCKED DRIVER.
//
// Read docs/perf/potrf.md; where the original spec and its corrections
// disagree the corrections win. This file is the "revised implementation order"
// rows 2.1 and 2.2.
//
// THE SCHEDULE -- right-looking, Uplo::Lower, for j = 0, nb, 2nb, ... with
// ib = min(nb, n-j) and m2 = n - j - ib:
//
//   (1) LEAF     A(j:j+ib, j:j+ib) = L11 L11^H, by the Phase 1 CTA kernel,
//                handed a SUB-VIEW. potrf_cta_dispatch reads A.ld() and
//                A.stride() (potrf_cta.cc:543-544) and touches only [0,ib)^2 of
//                the view it is given (potrf_cta_device.hh:472, :523-524), so a
//                sub-view is exactly as correct as a whole matrix -- and, unlike
//                every GEMM operand below, it is ld-INSENSITIVE: consecutive
//                lanes hold consecutive rows at a fixed column, so the global
//                access is unit-stride at any ld.
//   (2) PANEL    L21 = A21 * L11^{-H}, i.e.
//                trsm(Right, Lower, ConjTrans, NonUnit) on the m2 x ib panel,
//                through the INJECTED (routed) trsm. See PotrfPanelSolve in
//                potrf_native.hh for the measurement that chose the routed trsm
//                over a bespoke panel kernel.
//   (3) TRAILING A22 -= L21 L21^H, through the INJECTED (routed) gemm, split so
//                that the upper triangle of A is never written. See the
//                trailing-update block below.
//
// WHY NOT herk/syrk FOR (3). herk (level3.cc:295-306) has NO native arm at all:
// in a vendor-free build it calls throw_no_vendor_route, which would defeat the
// entire point of WP4. syrk has a custom arm, but its "cublasdx" route is
// silently a fallback that WRITES BOTH TRIANGLES (a recorded bug). So the
// trailing update is gemm + an explicit triangular fold.
//
// ---------------------------------------------------------------------------
// WHAT THIS FILE DELIBERATELY DOES NOT DO
// ---------------------------------------------------------------------------
// * It does not flip RouteTable<Op::potrf,T>::preferred(). This phase ships
//   ROUTE-NEUTRAL: reachable only when a caller pins BATCHLAS_POTRF_ROUTE, or in
//   a vendor-free build where route_resolve.hh:60-63 hands over any SUPPORTED
//   native route. The fit split between CTA and Blocked, and the vendor
//   crossover, are a separate measured grid.
// * It does not touch potrf_cta.cc's launcher. The obvious cheaper spelling of
//   the info merge is a `j_offset` + first-wins flag on the leaf itself (three
//   lines at potrf_cta.cc:615, removing one kernel launch per panel). It is not
//   taken because this driver needs a per-panel kernel ANYWAY -- the quench in
//   potrf_blocked_panel_fixup below -- so the merge rides along on a launch that
//   already exists and the marginal cost of doing it in the driver is zero.
//   Phase 1 stays untouched, and every test that pins the leaf's unconditional
//   write (EmptyInfoSpanStillFactorises and the four direct-call info tests,
//   which never pre-zero) keeps its meaning.
// * No Backend cross-product, exactly as potrf_cta.cc:706-726. Everything that
//   needs a Backend arrives through the two injection seams.

#include "potrf_native.hh"
#include "symmetric_product_fold.hh"

#include "../sycl/gemm_kernels.hh"
#include "../sycl/trsm_native.hh"

#include "../queue.hh"
#include "../util/template-instantiations.hh"

#include <batchlas/util/mempool.hh>

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <sycl/sycl.hpp>

namespace batchlas {
namespace sycl_potrf {

namespace {

// ---------------------------------------------------------------------------
// Type helpers.
// ---------------------------------------------------------------------------
template <typename T> struct PotrfIsComplex : std::false_type {};
template <> struct PotrfIsComplex<std::complex<float>>  : std::true_type {};
template <> struct PotrfIsComplex<std::complex<double>> : std::true_type {};

// The transpose the trailing update puts on the SECOND operand.
//
// A FREE 1.81x FOR float FROM ONE ENUM VALUE, and it is measured, not reasoned.
// ConjTrans and Trans denote the same operation for a real scalar, but they do
// not reach the same kernel: gemm_kernels.cc:470's transposed short-circuit
// sends every ConjTrans to Tiled16, while the three float exceptions at :472-480
// require transB == Transpose::Trans EXACTLY and reach
// Tiled128x32RegisterK32NT. Kernel level that is 1.77-1.86x; end to end on
// correct answers at n=1024 it is 16.629 -> 9.180 ms, i.e. 1.81x. The win is
// worth 1.00x at nb <= 96 and 1.81x at nb = 128 -- exactly that kernel's
// k >= 128 gate, since the trailing update's k IS nb. It is also part of why
// nb = 128 is the measured choice for float below.
//
// IT MUST NOT BE DONE FOR COMPLEX. The A/B harness's PHASE2_BREAK=conj is
// precisely this substitution and the residual goes 4.0e-07 -> 1.9e-02 for
// cfloat and cdouble. A22 -= L21 L21^H is a HERMITIAN update; L21 L21^T is a
// different matrix.
template <typename T>
constexpr Transpose kTrailingTransB =
    PotrfIsComplex<T>::value ? Transpose::ConjTrans : Transpose::Trans;

// ---------------------------------------------------------------------------
// THE BLOCK WIDTHS. MEASURED, and they are NOT potrf_cta_max_n<T>().
// ---------------------------------------------------------------------------
//
// nb -- the diagonal block order, hence the leaf's order, hence the trailing
// update's k. Measured over the whole blocked driver at batch 128, n in
// {512, 1024}, worst relative sd 1.3%:
//
//   float   n=512 total ms by nb 32/48/64/96/128/155:
//             default 2.272/1.845/1.507/1.306/1.114/1.234
//             native  3.754/3.194/2.929/2.634/2.462/2.524
//           and n=1024 with a correct (vendor) panel solve, nb 64/96/128/155:
//             19.993/17.699/9.180/11.604            -> 128 wins everywhere
//   double  n=512 native by nb 32/48/64/80/96/109:
//             8.594/7.814/7.399/7.190/7.019/7.258
//           n=1024: 59.343/55.456/53.338/53.538/50.980/53.682  -> 96 at both
//   cfloat  n=512 leaf+panel+trail 7.38/6.32/5.75/5.47/5.22/5.40
//           n=1024 total 43.4/42.8/36.7/38.1/34.1/37.9         -> 96 at both
//   cdouble n=512 total by nb 32/48/64/77: 81.3/78.6/76.2/79.5
//           n=1024 native 576/749/550/579, default 205/202/190/203 -> 64
//
// The float mechanism is sharp and named above: k == nb, and float's only
// transposed register kernel needs k >= 128, so nb < 128 cannot reach it while
// nb > 128 only makes the leaf slower (leaf 0.335 ms at nb=128 against 0.391 at
// 155, n=512). NONE of the four is potrf_cta_max_n<T>() = {155,109,109,77}; all
// four are strictly below it. Sizing nb from the fit ceiling -- the obvious
// thing, and what the spec does -- is measurably wrong for every type.
//
// W -- the width of the column panel the trailing update is cut into. Measured
// trailing-stage ms at n=512 on a NATIVE gemm, W = 16/32/64/96/128:
//   float   1.693/1.491/1.569/1.706/1.860
//   double  3.942/3.885/4.197/4.497/4.908
//   cfloat  3.127/2.839/3.109/3.377/3.819
//   cdouble 52.1  /54.2 /59.2 /63.5 /68.3
// Monotonic above 32 for three types and above 16 for cdouble. That is the
// wasted-work fraction showing up directly: the W x W diagonal block computes a
// full square where only a triangle is wanted, so waste = W/m2, LINEAR in W.
// The spec's W = 128 was chosen against a 12.5% figure the corrections doc had
// already refuted (the real figure is 25% at the spec's own shapes).
//
// AND FOR float THAT TABLE IS WRONG, WHICH WP4 PHASE 2 TRIAGE RE-MEASURED.
// Those cells were trailing-stage timings at n=512 only, where the effect below
// is smallest, and a caveat was attached saying the vendor gemm wanted 96-128
// and that no single number could serve both. Re-measured END TO END on a
// CORRECT factorisation (the panel solve was returning wrong answers at large
// batch when that table was taken -- see the trsm barrier note in
// potrf_blocked_params below), interleaved, 3 passes x 2 reps, worst rel sd
// 4.9%, ms by W = 16/32/64/96/128:
//
//   float, both seams NATIVE (the vendor-free build):
//     n=512  b=256   4.422 / 3.449 / 3.774 / 4.271 / 3.403   -> 128 by 1.01x
//     n=1024 b=256  28.286 /17.863 /18.654 /20.495 /16.785   -> 128 by 1.06x
//     n=2048 b=128 107.046 /52.937 /50.879 /54.936 /46.510   -> 128 by 1.14x
//   float, both seams VENDOR:
//     n=1024 b=256  29.330 /19.657 /15.619 /15.124 /15.122   -> 128 by 1.30x
//
// So 128 wins on BOTH routes, and by more as n grows. The mechanism is the same
// one that fixes nb at 128 for float: the W x W DIAGONAL-block gemm has
// m = n = W and k = nb, and gemm_kernels.cc:472-480 gives float's transposed
// register kernel only at m >= 128 && n >= 32 && k >= 128. At W = 32 (or 64, or
// 96) that gemm can never reach it and lands on Tiled16; at W = 128 it does.
// Note the curve is NON-MONOTONIC -- 96 is worse than 64 -- which is the
// signature of a kernel-selection cliff rather than of the linear waste term,
// and is why interpolating this table is unsafe.
//
// The per-route W the review asked for is therefore NOT needed: one constant is
// better than the old one on both routes, and no dispatch fact enters this TU.
//
// The other three types are unchanged and were re-measured at the same point,
// n=1024 b=256 native, ms by W = 16/32/64/96/128:
//   double  77.73 / 78.58 / 81.94 / 85.11 / 89.17   (16 by 1.1% over the shipped
//                                                    32 -- inside the noise the
//                                                    n=512 table was taken at,
//                                                    so 32 stays)
//   cfloat  54.63 / 54.00 / 57.40 / 59.95 / 63.45   -> 32, as shipped
// cdouble was not re-swept; its k = nb = 64 cannot reach any register kernel at
// any W, and both its n=512 table and the double/cfloat trend say smaller wins.
template <typename T> struct PotrfBlockedConst;
template <> struct PotrfBlockedConst<float>                { static constexpr int NB = 128; static constexpr int W = 128; };
template <> struct PotrfBlockedConst<double>               { static constexpr int NB = 96;  static constexpr int W = 32; };
template <> struct PotrfBlockedConst<std::complex<float>>  { static constexpr int NB = 96;  static constexpr int W = 32; };
template <> struct PotrfBlockedConst<std::complex<double>> { static constexpr int NB = 64;  static constexpr int W = 16; };

// Tuning overrides, read ONCE per process.
//
// TUNING, not routing: they never change which route is chosen, only how the
// driver blocks once chosen, so they do not belong in the Route vocabulary
// (trsm_native.cc:630-632 states the same rule for BATCHLAS_TRSM_OUTER_NB).
//
// Read once into a function-local static, and that is a correctness property
// rather than a micro-optimisation: potrf_buffer_size and potrf resolve
// SEPARATELY (options.hh:546-552 calls the query at :550 and the call at :551),
// so an env re-read between them could size one layout and build another. A
// static cannot.
inline int potrf_env_int(const char* name) {
    const char* raw = std::getenv(name);
    if (!raw || !*raw) return 0;
    const int v = std::atoi(raw);
    return v > 0 ? v : 0;
}
inline int potrf_nb_env() { static const int v = potrf_env_int("BATCHLAS_POTRF_NB"); return v; }
inline int potrf_w_env()  { static const int v = potrf_env_int("BATCHLAS_POTRF_W");  return v; }

// ---------------------------------------------------------------------------
// THE BLOCK WIDTHS, AS ONE PURE FUNCTION.
//
// Called by potrf_blocked_buffer_size AND by potrf_blocked_dispatch, which is
// what makes it impossible for the query to size a W the call does not use. The
// analogous split in Phase 1 -- a RAW slm figure in the capability query against
// a PADDED one in the launcher -- made supports() promise a call the launcher
// then threw on (potrf_cta.cc:442-454). Do not let either number become a
// literal in two places.
// ---------------------------------------------------------------------------
struct PotrfBlockedParams {
    int nb;  // diagonal block order == the leaf's order == the trailing update's k
    int W;   // trailing-update column-panel width
};

template <typename T>
PotrfBlockedParams potrf_blocked_params(Queue& ctx, int n) {
    using C = PotrfBlockedConst<T>;

    // THE CEILING IS ASKED OF THIS DEVICE, never of potrf_cta_max_n<T>().
    // The no-argument overload answers at the hardcoded kPotrfReferenceSlmBudget
    // (potrf_cta.cc:181, :476-478) while potrf_cta_dispatch gates on the RUNTIME
    // budget (:672-673). They coincide on this box and diverge on any device
    // with less local memory -- where sizing nb from the constant would pick a
    // block the leaf then refuses with std::invalid_argument. Same recipe, same
    // reserve, as the shape builder (potrf_route.hh:114-115).
    const std::size_t local_mem =
        static_cast<std::size_t>(ctx.device().get_property(DeviceProperty::LOCAL_MEM_SIZE));
    const int ceiling = potrf_cta_max_n_for_slm<T>(local_mem > 4096 ? local_mem - 4096 : 0);

    const int want = potrf_nb_env() ? potrf_nb_env() : C::NB;
    int nb = std::min(want, std::max(ceiling, 1));
    if (n > 0) nb = std::min(nb, n);

    // ROUNDED DOWN TO A WHOLE NUMBER OF trsm_cta_max_n<T>() BLOCKS.
    //
    // WHY IT WAS ADDED, AND WHY THAT REASON IS NOW GONE. The measure phase found
    // (while looking for something else) that the panel solve returned a WRONG
    // ANSWER -- host residual 1e+04..1e+20 against the vendor's 1e-07 -- on the
    // DEFAULT route, at triangular orders 48, 77, 80 and 109 but not at 32, 64,
    // 96 or 128, and only above roughly q*batch = 65k. Keeping nb a multiple of
    // the V1 capacity kept the final V1 block at full width and out of the
    // failing bucket, so this line was shipped as a containment.
    //
    // WP4 Phase 2 TRIAGE FOUND THE ACTUAL DEFECT and fixed it at source: the V1
    // CTA kernel staged its triangle into local memory and then read the
    // diagonal back with NO group barrier in between (src/sycl/trsm_native.cc,
    // the barrier now immediately after the staging loop). q*batch above ~65k is
    // exactly where the launcher's work-group ladder leaves wg = 32 -- one
    // sub-group, where the race cannot express itself -- and goes to 128 or 256.
    // Post-fix, a direct trsm(Right, Lower, ConjTrans, NonUnit) sweep over
    // orders {16,32,48,64,77,80,96,109,128,155} at q = 896, batch = 256 agrees
    // with a host reference to the same relative error as cuBLAS at every order,
    // float and double; nothing resembling 1e+04 appears anywhere.
    //
    // THE ROUNDING IS KEPT ANYWAY, and deliberately, because it costs nothing:
    // all four shipped NB values are already multiples of 32, so it is the
    // identity on every default path. What it now buys is that a hand-set
    // BATCHLAS_POTRF_NB cannot wander into an order whose V1 block structure has
    // never been measured. If a future tuning pass wants 109 or 155 it should
    // delete this line and re-measure, not fight it.
    //
    // AND IT APPLIES TO THE ENV OVERRIDE TOO -- a sweep of
    // BATCHLAS_POTRF_NB = 48/80/109/155 collapses onto 32/64/96/128 and produces
    // four cells identical to their neighbours. The nb table recorded above was
    // gathered with those raw values as labels; read it as 32/64/96/128, and do
    // not expect e.g. "nb=155" to reproduce as anything but nb=128.
    //
    // Note what makes the whole question containable: the panel solve is only
    // ever issued at the FULL block width. A short final block has m2 == 0 by
    // construction (ib < nb only when j + ib == n), so it issues no trsm and no
    // trailing update at all -- see the loop in potrf_blocked_dispatch.
    //
    // The rounding is skipped when the device ceiling is itself below the V1
    // capacity, because there is then no legal multiple to round to.
    const int leaf_trsm = sycl_trsm::trsm_cta_max_n<T>();
    if (leaf_trsm > 0 && nb >= leaf_trsm) {
        nb = (nb / leaf_trsm) * leaf_trsm;
    }
    if (nb < 1) nb = 1;

    int W = potrf_w_env() ? potrf_w_env() : C::W;
    if (W < 1) W = 1;

    return {nb, W};
}

// ---------------------------------------------------------------------------
// THE WORKSPACE, described exactly once.
// ---------------------------------------------------------------------------
template <typename T>
struct PotrfBlockedWs {
    Span<int32_t> info;       // the driver's own info fallback
    Span<int32_t> leaf_info;  // what the leaf writes, per panel, before merging
    Span<T*>      a11_ptrs;   // pointer array for the L11 role
    Span<T*>      a21_ptrs;   // pointer array for the panel role
    Span<T>       product;    // W x W x batch scratch for the diagonal-block gemm
    Span<std::byte> leaf_ws;  // handed straight to potrf_cta_dispatch
};

template <typename T>
PotrfBlockedWs<T> potrf_blocked_layout(Queue& ctx, BumpAllocator& pool,
                                       int n, int nb, int batch, int W,
                                       std::size_t leaf_bytes) {
    PotrfBlockedWs<T> ws;
    const std::size_t b = static_cast<std::size_t>(batch);

    // The info fallback, drawn unconditionally even though the caller usually
    // supplies a span. potrf_cta_buffer_size does the same, and it has to: the
    // query cannot see info_out, so a conditional draw would make the reported
    // size depend on an argument the query is never given.
    ws.info = pool.allocate<int32_t>(ctx, b);

    // A SEPARATE span for the leaf, and this is the whole of the info fix.
    // potrf_cta.cc:615 writes info UNCONDITIONALLY (`if (tid == 0)
    // info_ptr[matrix_id] = *fl;`) and re-zeroes its flag on every launch
    // (potrf_cta_device.hh:488), so pointing every panel's leaf at the driver's
    // persistent info gives LAST-PANEL-WINS -- the exact inversion of LAPACK's
    // first-failure rule, and worse, a successful later panel overwrites a real
    // failure with 0 and the call reports success on a non-PD matrix.
    ws.leaf_info = pool.allocate<int32_t>(ctx, b);

    // ONE POINTER ARRAY PER ROLE. The vendor batched trsm calls A.data_ptrs(ctx)
    // (cublas.cc:1220), and a MatrixView built by the 6-arg constructor has an
    // empty data_ptrs_ span, so it throws "data_ptrs target is null"
    // (matrix.cc:2369) -- an abort, not a wrong answer. The driver cannot know
    // whether the injected trsm will resolve to a vendor, so both roles get an
    // array unconditionally; they cost 8 bytes per item.
    //
    // PER ROLE, never one array shared: init_data_ptr_array recomputes the
    // target from the view's OWN data_ptr()/stride on every call
    // (matrix.cc:2364-2383), so two roles sharing one array would have the
    // second call rewrite the first's bases. That is [FIX-B-trap] with the
    // pointer array in place of the pointer.
    ws.a11_ptrs = pool.allocate<T*>(ctx, b);
    ws.a21_ptrs = pool.allocate<T*>(ctx, b);

    // The diagonal-block product. fold_symmetric_product_into_triangle computes
    // C = product + beta*C (symmetric_product_fold.hh:29-34, :68) and has no
    // alpha, so the product must be a real allocated matrix that the feeding
    // gemm writes with alpha = -1, beta = 0.
    //
    // NOT DRAWN AT ALL when the whole matrix is one block. nb is already
    // min(want, ceiling, n) (potrf_blocked_params), so `n <= nb` is exactly the
    // single-block case, in which m2 == 0 at the only j and the trailing loop
    // never runs. It is worth the branch because src/extensions/ortho.cc:78 is a
    // real caller of the public query at k = 5..256 and this term is the whole
    // of the difference between 512 B and 64 KiB there. The condition is
    // evaluated from the same (n, nb) in the query and in the driver, so the two
    // cannot disagree about whether the buffer exists.
    ws.product = (n > nb) ? pool.allocate<T>(ctx, static_cast<std::size_t>(W) *
                                                  static_cast<std::size_t>(W) * b)
                          : Span<T>{};

    // An EXPLICIT draw, never pool.remaining(): mempool.hh:121-128 THROWS in
    // measuring mode ("A sizing pool has no tail to hand out"), so a layout that
    // handed the tail to the leaf could not be sized at all. The leaf never
    // touches this in practice -- it falls back to pool scratch only when the
    // info span it is given is shorter than batch, and ws.leaf_info never is --
    // but sizing it costs one int32 per item and removes the dependency on that
    // contract staying true.
    ws.leaf_ws = pool.allocate<std::byte>(ctx, leaf_bytes);

    return ws;
}

}  // namespace

template <typename T>
std::size_t potrf_blocked_buffer_size(Queue& ctx,
                                      const MatrixView<T, MatrixFormat::Dense>& A,
                                      Uplo uplo) {
    // The layout is uplo-independent; the driver, not the query, carries the
    // Uplo::Lower gate. The parameter is kept so the query's signature matches
    // the call's, which is what stops the two from being given different
    // arguments (entry_points/factorization.cc:8-10).
    static_cast<void>(uplo);

    const int batch = static_cast<int>(A.batch_size());
    const int n = static_cast<int>(A.rows());
    if (batch < 1 || n < 1) return 0;

    const auto p = potrf_blocked_params<T>(ctx, n);

    // ASKED ABOUT THE CALLER'S VIEW, never a workspace-derived one
    // (mempool.hh:179-184). potrf_cta_buffer_size reads only A.batch_size(), and
    // every leaf sub-view carries the parent batch, so the parent is both the
    // correct and the only safe thing to ask about.
    const std::size_t leaf_bytes = potrf_cta_buffer_size<T>(ctx, A);

    return workspace_bytes([&](BumpAllocator& pool) {
        return potrf_blocked_layout<T>(ctx, pool, n, p.nb, batch, p.W, leaf_bytes);
    });
}

// The blocking, as the tests must be able to see it. potrf_native.hh carries
// why this exists at all; the point here is that it answers from
// potrf_blocked_params -- the SAME pure function the driver and the buffer-size
// query call -- so it cannot report a blocking the call does not use.
template <typename T>
unsigned potrf_blocked_debug_params(Queue& ctx, int n) {
    const auto p = potrf_blocked_params<T>(ctx, n);
    return (static_cast<unsigned>(p.W) << 16) | static_cast<unsigned>(p.nb);
}

// ---------------------------------------------------------------------------
// The per-panel fixup kernel: INFO MERGE + FAILED-ITEM QUENCH, one launch.
// ---------------------------------------------------------------------------
namespace {

template <typename T> class PotrfBlockedFixupKernel;

// Runs after the leaf and BEFORE the panel solve. Two jobs, fused because the
// second needs the first's answer and because a launch is a launch.
//
// (1) INFO MERGE, LOCAL -> GLOBAL, FIRST FAILURE WINS.
//     The leaf reports `j_local + k + 1`, 1-based and local to the sub-view it
//     was handed (potrf_cta_device.hh:195-199). LAPACK's blocked ?POTRF does
//     `INFO = INFO + J - 1` and then stops; a batched driver cannot stop, so it
//     masks per item instead:
//         if (info[b] == 0 && leaf[b] != 0) info[b] = j + leaf[b];
//     That index is a TRUE updated-Schur minor order, not an approximation: the
//     right-looking recurrence leaves A(j:j+ib, j:j+ib) fully updated before the
//     leaf runs, so `j + local` is exactly LAPACK's leading-minor order.
//
// (2) QUENCH, so a failed item stays FINITE through the rest of the schedule.
//     A failed leaf leaves S(j+k, j+k) unpublished -- zero or negative -- and
//     the panel trsm then divides A21 by it, giving Inf/NaN which the trailing
//     gemm smears across all of A22 and every later panel. Finiteness is not a
//     contract claim (LAPACK leaves a failed A undefined, and
//     tests/potrf_tests.cc:723-726 says so in as many words), but it IS a
//     property the CTA route has, and losing it silently at the tier boundary
//     would be a surprise. The quench restores it: force the lower triangle of
//     the diagonal block to the identity and zero the panel below it. Then trsm
//     gives L21 = 0, the trailing update is a no-op, and A22 is left exactly as
//     it was.
//
//     ZEROING THE PANEL ALONE IS NOT ENOUGH, which is where the spec's naive
//     form (perf-evidence/vendor-independence:WP4_POTRF_SPEC.md:387) is wrong: with a zero diagonal in L11 the
//     solve computes 0/0 = NaN. The unit diagonal is the load-bearing half.
//
//     The UPPER triangle of the diagonal block is NOT written. LAPACK potrf
//     (Lower) must leave it exactly as the caller passed it, on failed items as
//     much as on healthy ones.
//
// COST WHEN NOTHING FAILS, which is the case that matters: one work-group per
// batch item, every work-item reads two int32 and returns. n/nb launches per
// factorisation, ~8-16 at the shapes this driver serves.
template <typename T>
Event potrf_blocked_panel_fixup(Queue& ctx,
                                T* a_ptr, int ld, int stride,
                                int j, int ib, int m2, int batch,
                                int32_t* info_ptr, const int32_t* leaf_ptr,
                                int wg) {
    const int rows = ib + m2;          // the whole column panel A(j:n, j:j+ib)
    const T one = T(1);
    const T zero = T(0);

    ctx->submit([&](sycl::handler& h) {
        h.parallel_for<PotrfBlockedFixupKernel<T>>(
            sycl::nd_range<1>(sycl::range<1>(static_cast<std::size_t>(batch) *
                                             static_cast<std::size_t>(wg)),
                              sycl::range<1>(static_cast<std::size_t>(wg))),
            [=](sycl::nd_item<1> it) {
                const int b = static_cast<int>(it.get_group_linear_id());
                const int tid = static_cast<int>(it.get_local_linear_id());

                // EVERY work-item reads info[b] BEFORE any work-item writes it,
                // and the barrier is what makes that true rather than probable.
                // Without it the merge is a read/write race on one word inside
                // one work-group.
                const int32_t prev = info_ptr[b];
                const int32_t li = leaf_ptr[b];
                sycl::group_barrier(it.get_group());
                if (tid == 0 && prev == 0 && li != 0) {
                    info_ptr[b] = j + li;
                }

                const bool dead = (prev != 0) || (li != 0);
                if (!dead) return;

                T* base = a_ptr + static_cast<std::ptrdiff_t>(b) * stride +
                          static_cast<std::ptrdiff_t>(j) * ld + j;
                const std::size_t total = static_cast<std::size_t>(rows) *
                                          static_cast<std::size_t>(ib);
                for (std::size_t e = static_cast<std::size_t>(tid); e < total;
                     e += static_cast<std::size_t>(wg)) {
                    const int c = static_cast<int>(e / static_cast<std::size_t>(rows));
                    const int r = static_cast<int>(e % static_cast<std::size_t>(rows));
                    if (r < ib) {
                        if (r < c) continue;                 // upper triangle: untouched
                        base[static_cast<std::ptrdiff_t>(c) * ld + r] = (r == c) ? one : zero;
                    } else {
                        base[static_cast<std::ptrdiff_t>(c) * ld + r] = zero;
                    }
                }
            });
    });
    return ctx.get_event();
}

}  // namespace

// ---------------------------------------------------------------------------
// The driver.
// ---------------------------------------------------------------------------
template <typename T>
Event potrf_blocked_dispatch(Queue& ctx,
                             const MatrixView<T, MatrixFormat::Dense>& A,
                             Uplo uplo,
                             Span<std::byte> workspace,
                             Span<int32_t> info_out,
                             PotrfTrailingGemm<T> trailing_gemm,
                             PotrfPanelSolve<T> panel_solve) {
    // Default both seams to the NATIVE kernels so this TU stands alone: a direct
    // caller (the tests, and any harness that must not be silently served by a
    // vendor) gets the native path with no dispatch dependency. The facade
    // passes the routed gemm and the routed trsm instead --
    // entry_points/factorization.cc, modelled on level3.cc:186-231.
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
    if (!panel_solve) {
        panel_solve = [](Queue& c,
                         const MatrixView<T, MatrixFormat::Dense>& ta,
                         const MatrixView<T, MatrixFormat::Dense>& tb,
                         T talpha, Side tside, Uplo tuplo, Transpose ttrans, Diag tdiag) {
            // V2 rather than V1: it degenerates to a single V1 solve when the
            // triangular order fits the CTA capacity (trsm_native.cc:780-802
            // clamps both loop bounds), so one spelling covers every nb.
            return sycl_trsm::trsm_native_blocked<T>(c, ta, tb, talpha, tside, tuplo,
                                                     ttrans, tdiag);
        };
    }

    const int n = static_cast<int>(A.rows());
    const int batch = static_cast<int>(A.batch_size());

    // Every gate RouteTable<Op::potrf,T>::supports() applies to the Blocked arm,
    // re-applied here because this entry point is reachable without the table --
    // and it MUST be, for the reason in potrf_native.hh: a forced route the
    // table rejects silently falls back to the vendor.
    if (A.rows() != A.cols()) {
        throw std::invalid_argument("potrf_blocked: A must be square");
    }
    if (n < 1 || batch < 1) {
        throw std::invalid_argument("potrf_blocked: degenerate extents");
    }
    if (uplo != Uplo::Lower) {
        // CORRECTNESS, not fit. The schedule below is Lower-shaped end to end;
        // handed an Upper view it would read and overwrite the wrong triangle
        // and return a plausible wrong answer. route_potrf.hh:278 carries the
        // matching supports() gate.
        throw std::invalid_argument(
            "potrf_blocked: Uplo::Upper is not implemented; the driver factors the "
            "lower triangle only (route_potrf.hh:270-278)");
    }
    if (A.is_heterogeneous()) {
        // Same reason as the leaf (potrf_cta.cc:648-656): one schedule covers the
        // batch with a single (order, ld, stride) tuple, so per-item active dims
        // would factorise the wrong order in place for every item after the first.
        throw std::invalid_argument("potrf_blocked: heterogeneous batch is not supported");
    }
    const auto dev = ctx.device();
    if (dev.type != DeviceType::GPU) {
        throw std::invalid_argument("potrf_blocked: GPU queues only");
    }

    const auto p = potrf_blocked_params<T>(ctx, n);
    const int nb = p.nb;
    const int W = p.W;

    const std::size_t leaf_bytes = potrf_cta_buffer_size<T>(ctx, A);
    BumpAllocator pool(workspace);
    auto ws = potrf_blocked_layout<T>(ctx, pool, n, nb, batch, W, leaf_bytes);

    // detail::info_target's rule, inlined so this TU does not include
    // src/linalg-impl.hh, and identical to the leaf's (potrf_cta.cc:686-694): an
    // empty OR SHORT caller span means "not requested" and falls back to pool
    // scratch. The trap the corrections doc records (:938-943) is to zero
    // info_out instead of THIS span -- a short non-empty caller span silently
    // becomes pool scratch, and zeroing the caller's would leave the span the
    // driver actually reads full of whatever the pool last held.
    Span<int32_t> info = (info_out.size() >= static_cast<std::size_t>(batch))
                             ? info_out
                             : ws.info;

    // THE ZERO PRE-PASS, and it is not optional. The merge in panel_fixup reads
    // info[b] to decide whether an earlier panel already failed; info_out arrives
    // with caller garbage (options_api_tests.cc:498,509 seeds -12345), so without
    // this every item looks already-failed, every failure is discarded, and every
    // item is quenched to the identity -- a silent wrong answer with info
    // unchanged and no error raised.
    // NO .wait() HERE. It used to be `...fill(...).wait()`, which on the DEFAULT
    // in-order queue (sycl-device-queue.hh:254) drains everything already
    // enqueued before the first leaf is even submitted -- a full host round trip
    // per potrf call, paid by every pipelined caller (ortho.cc:78 issues one
    // potrf per CholQR iteration). In-order already orders the fill ahead of the
    // fixup that reads it; out-of-order is handled by the explicit guard below,
    // which is the same idiom the panel loop uses. It is also the only
    // `fill(...).wait()` anywhere in src/ -- steqr_cta.cc:158 and
    // latrd_lower_panel.cc:669 both zero device state fire-and-forget.
    ctx->fill(info.data(), int32_t(0), static_cast<std::size_t>(batch));

    // THE SCRATCH ZERO PRE-PASS, and it is a WRONG-ANSWER fix, not hygiene.
    //
    // The diagonal-block gemm below is issued with beta = T(0), and the comment
    // that used to sit there claimed beta = 0 meant the scratch was never read.
    // THAT IS TRUE OF THE FOLD (symmetric_product_fold.hh:49, :68 branch on
    // `ignore_c`) AND FALSE OF EVERY GEMM IN THIS TREE: LinearEpilogue::apply is
    // `alpha*accum + beta*prior` with `prior = c_ptr[...]` read unconditionally
    // (gemm/epilogue_linear.hh:7-9, tiled_general.hh:79-81,
    // register_tiled_common.hh:598,613), so 0 * NaN = NaN.
    //
    // REPRODUCED through the ordinary public API, not a contrived buffer: an
    // earlier `ctx.workspace()` lease leaves poison in the arena bytes this
    // scratch is served from (options.hh:550-551 leases the same arena), and
    // float n=256 batch=8 on a well-conditioned SPD input then returned
    // info != 0 for 8/8 items with relative residual 9.941e-01 under
    // BATCHLAS_POTRF_ROUTE=blocked BATCHLAS_GEMM_ROUTE=native -- i.e. the
    // vendor-free configuration this work package exists for. With cuBLAS
    // injected it happened to survive, because cuBLAS honours beta == 0.
    // Every Phase 2 test missed it because each allocates a FRESH
    // UnifiedVector<std::byte>, which the CUDA driver hands back zeroed.
    //
    // ONE fill, not one per column panel. After the first diagonal gemm the
    // scratch holds a finite product, and 0 * finite == 0; the only way a later
    // read can be non-finite is if the product itself overflowed, which means
    // the rectangle gemm has already written the same magnitudes into A and the
    // answer is garbage either way. Per-panel zeroing would add one launch per
    // column panel (112 of them at n=1024) to a stage that is launch-bound.
    //
    // src/extensions/syrk.cc:51 -- the only other caller of this fold helper --
    // allocates its product scratch with Matrix<T,Dense>::Zeros for exactly this
    // reason. This driver is now consistent with it.
    if (!ws.product.empty()) {
        ctx->fill(ws.product.data(), T(0), ws.product.size());
    }

    // Both fills are READ by the first panel's kernels. In-order gives that
    // ordering for free; out-of-order does not.
    if (!ctx.in_order()) ctx.wait();

    const int ld = A.ld();
    const int stride = A.stride();
    T* const a_ptr = A.data_ptr();

    // SUB-VIEWS ARE BUILT BY THE EXPLICIT 6-ARG CONSTRUCTOR, never by
    // operator()(Slice,Slice), and the parent's ld AND stride AND batch are
    // passed at every call. Two independent traps, both verified in source:
    // matrix.hh:1136-1140 carries a comment saying a slice must not propagate the
    // parent pointer array and the very next line does; and the constructor
    // DEFAULTS stride to ld*cols when 0 is passed (matrix.cc:1839-1842), so a
    // sub-view of ib columns built without an explicit stride silently gets
    // stride = ld*ib and every batch item after the first reads the wrong matrix.
    // The A/B harness's PHASE2_BREAK=stride is exactly that second one, and it
    // turns the residual red for all four types.
    //
    // `ptrs` is an optional per-ROLE pointer array (see the layout above), not
    // the parent's.
    auto sub = [&](int r0, int nr, int c0, int nc, T** ptrs) {
        return MatrixView<T, MatrixFormat::Dense>(
            a_ptr + static_cast<std::ptrdiff_t>(c0) * ld + r0,
            nr, nc, ld, stride, batch, ptrs);
    };

    // The scratch product, W x W per item, tightly packed. Its ld stays W even
    // when the final column panel is short (w < W), which the fold honours: it
    // reads C.ld()/C.stride() and product.ld()/product.stride() independently
    // (symmetric_product_fold.hh:44-47, :66-67).
    T* const prod_ptr = ws.product.data();

    const int fixup_wg = std::min<int>(
        128, std::max<int>(32, static_cast<int>(
                                   dev.get_property(DeviceProperty::MAX_WORK_GROUP_SIZE))));

    for (int j = 0; j < n; j += nb) {
        // THE SHORT FINAL BLOCK is carried by this one std::min and nothing else,
        // exactly as the leaf's own panel loop carries its ragged last panel
        // (potrf_cta.cc:39-41). Every derived extent below comes from `ib` and
        // `m2`, never from `nb`, so nothing else has to special-case it. Note the
        // consequence relied on in potrf_blocked_params: ib < nb IMPLIES
        // j + ib == n implies m2 == 0, so a short block issues no panel solve and
        // no trailing update -- the trsm and the gemm always see the full block
        // width.
        const int ib = std::min(nb, n - j);
        const int m2 = n - j - ib;

        // (1) THE LEAF, on a sub-view. It re-checks its own fit and THROWS if the
        // order does not fit this device (potrf_cta.cc:677-684) rather than
        // factoring a leading submatrix -- the discipline WP3 had to learn the
        // hard way (trsm_native.cc:96-115: a 33-order solve silently solved the
        // leading 32x32 system and left the last row untouched). ib <= nb <= the
        // device ceiling by construction, so it cannot fire here.
        const auto A11 = sub(j, ib, j, ib, ws.a11_ptrs.data());
        potrf_cta_dispatch<T>(ctx, A11, Uplo::Lower, ws.leaf_ws, ws.leaf_info);

        // EVERY dependent boundary in this schedule carries its own guard, not
        // just the two at the bottom of the loop. The leaf WRITES ws.leaf_info
        // and A11; the fixup READS ws.leaf_info and rewrites A11/A21; the panel
        // solve READS A11. On an out-of-order queue an unguarded fixup can see
        // the PREVIOUS panel's leaf_info -- reporting a wrong failing column, or
        // quenching a healthy item to the identity -- and an unguarded solve can
        // divide by a pivot the quench has not yet replaced. Public API:
        // Queue(Device, /*in_order=*/false), sycl-device-queue.hh:254.
        if (!ctx.in_order()) ctx.wait();

        // Local -> global info, first-failure-wins, plus the finiteness quench.
        // It must run before the panel solve, which is what would otherwise
        // divide by the failed pivot.
        potrf_blocked_panel_fixup<T>(ctx, a_ptr, ld, stride, j, ib, m2, batch,
                                     info.data(), ws.leaf_info.data(), fixup_wg);

        if (m2 == 0) break;

        if (!ctx.in_order()) ctx.wait();

        // (2) THE PANEL SOLVE. L21 = A21 * L11^{-H}, in place.
        const auto A21 = sub(j + ib, m2, j, ib, ws.a21_ptrs.data());
        panel_solve(ctx, A11, A21, T(1), Side::Right, Uplo::Lower,
                    Transpose::ConjTrans, Diag::NonUnit);

        // The trailing update READS the L21 the solve just wrote.
        if (!ctx.in_order()) ctx.wait();

        // (3) THE TRAILING UPDATE, A22 -= L21 L21^H, cut into column panels of
        // width W.
        //
        // A PLAIN SQUARE GEMM OVER ALL OF A22 IS WRONG. It would write the upper
        // triangle, which LAPACK potrf(Lower) must leave exactly as the caller
        // passed it. So each column panel splits in two:
        //
        //   * the W x W DIAGONAL block, whose product is symmetric/Hermitian and
        //     therefore cannot be aimed at A. It goes to scratch with
        //     alpha = -1, beta = 0 and is then FOLDED into the named triangle
        //     with beta = 1. The fold has NO alpha (symmetric_product_fold.hh has
        //     no such parameter), which is why the sign lives on the gemm.
        //     Copying the rectangle's (alpha=-1, beta=1) onto this gemm instead
        //     would double-count the scratch's previous contents.
        //
        //     BETA = 0 ON THAT GEMM DOES NOT MEAN THE SCRATCH IS UNREAD. An
        //     earlier revision of this comment claimed it did, citing
        //     symmetric_product_fold.hh:49/:68 -- which are the FOLD's lines,
        //     not the gemm's. Every gemm epilogue in this tree reads C
        //     unconditionally and multiplies by beta, so poison in the scratch
        //     DOES become NaN. The scratch is therefore zeroed once, before the
        //     panel loop; see the fill above for the reproduction.
        //   * the strictly BELOW-DIAGONAL rectangle, which is a plain gemm
        //     straight into A with alpha = -1, beta = 1. No scratch, no fold.
        //
        // THE BLIND-GUARD WARNING THAT BELONGS HERE, because it has already
        // caught someone inside this work package: the A/B harness's
        // PHASE2_BREAK=nofold deletes the fold and writes the diagonal block's
        // product straight into A, clobbering the upper triangle -- and the
        // residual stayed GREEN for all four types, because a residual computed
        // over the lower triangle cannot see it. It is also 11% CHEAPER. Nothing
        // residual-based will ever notice this fold being removed; only a test
        // that poisons the opposite triangle and bit-compares it afterwards will.
        for (int c = 0; c < m2; c += W) {
            const int w = std::min(W, m2 - c);

            const auto Lrow = sub(j + ib + c, w, j, ib, nullptr);
            const auto Cd = sub(j + ib + c, w, j + ib + c, w, nullptr);
            const MatrixView<T, MatrixFormat::Dense> Sc(prod_ptr, w, w, W, W * W, batch);

            trailing_gemm(ctx, Lrow, Lrow, Sc, T(-1), T(0),
                          Transpose::NoTrans, kTrailingTransB<T>,
                          ComputePrecision::Default);

            // RAW ON THE SCRATCH, and it is the sharpest of the four edges: the
            // fold reads exactly what the gemm above writes, and it is a much
            // smaller kernel (batch*w*w elementwise against a k=ib gemm), so on
            // an out-of-order queue it can finish first and fold stale scratch
            // into A22 with beta = 1 -- a silently wrong factor with info == 0.
            // The guard at the bottom of this loop is a WAR guard for the NEXT
            // iteration's gemm; it does not cover this.
            if (!ctx.in_order()) ctx.wait();

            ::batchlas::detail::fold_symmetric_product_into_triangle<T>(
                ctx, Cd, Sc, T(1), Uplo::Lower);

            const int mr = m2 - c - w;
            if (mr > 0) {
                const auto Lr = sub(j + ib + c + w, mr, j, ib, nullptr);
                const auto Cr = sub(j + ib + c + w, mr, j + ib + c, w, nullptr);
                trailing_gemm(ctx, Lr, Lrow, Cr, T(-1), T(1),
                              Transpose::NoTrans, kTrailingTransB<T>,
                              ComputePrecision::Default);
            }

            // The next column panel's gemm OVERWRITES the scratch this panel's
            // fold is still reading. An in-order queue gives that ordering for
            // free; an out-of-order one does not, and a caller may construct
            // either (sycl-device-queue.hh:239 defaults in_order=true but it is a
            // parameter). This is a correctness requirement, not a tuning choice
            // -- the same line, for the same reason, as trsm_native.cc:799.
            if (!ctx.in_order()) ctx.wait();
        }

        // The next panel's leaf reads what this trailing update just wrote.
        if (!ctx.in_order()) ctx.wait();
    }

    return ctx.get_event();
}

// ---------------------------------------------------------------------------
// The capability flag. DEFINED HERE, beside the driver, and deleted from
// potrf_cta.cc when Phase 2 landed -- see potrf_native.hh for why the placement
// is load-bearing rather than tidy.
// ---------------------------------------------------------------------------
template <> bool potrf_blocked_available<float>()                { return true; }
template <> bool potrf_blocked_available<double>()               { return true; }
template <> bool potrf_blocked_available<std::complex<float>>()  { return true; }
template <> bool potrf_blocked_available<std::complex<double>>() { return true; }

// ---------------------------------------------------------------------------
// Instantiation: PER SCALAR TYPE ONLY, no Backend cross-product, exactly as
// potrf_cta.cc:706-726. Everything that needs a Backend arrives injected.
// ---------------------------------------------------------------------------
#define BATCHLAS_POTRF_BLOCKED_INSTANTIATE(T)                                                 \
    template unsigned potrf_blocked_debug_params<T>(Queue&, int);                              \
    template std::size_t potrf_blocked_buffer_size<T>(                                        \
        Queue&, const MatrixView<T, MatrixFormat::Dense>&, Uplo);                             \
    template Event potrf_blocked_dispatch<T>(                                                 \
        Queue&, const MatrixView<T, MatrixFormat::Dense>&, Uplo, Span<std::byte>,             \
        Span<int32_t>, PotrfTrailingGemm<T>, PotrfPanelSolve<T>);

BATCHLAS_POTRF_BLOCKED_INSTANTIATE(float)
BATCHLAS_POTRF_BLOCKED_INSTANTIATE(double)
BATCHLAS_POTRF_BLOCKED_INSTANTIATE(std::complex<float>)
BATCHLAS_POTRF_BLOCKED_INSTANTIATE(std::complex<double>)

#undef BATCHLAS_POTRF_BLOCKED_INSTANTIATE

}  // namespace sycl_potrf
}  // namespace batchlas
