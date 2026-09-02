// Native batched GEQRF -- the BLOCKED tier.
//
// A right-looking blocked Householder QR: factor a column panel, form its WY
// block reflector, apply it to the trailing block with three GEMMs, advance.
//
//   for j0 = 0, nb, 2nb, ...
//       ib  = min(nb, k - j0)                    <- THE short final panel
//       (P) A(j0:m, j0:j0+ib) = geqr2, writing tau[j0 : j0+ib]
//       (L) V  = unit-lower pack of that panel   (pack_v)
//           T  = larft(V, tau[j0:j0+ib])         (ib x ib, upper triangular)
//       (G1) W1 = V^H A22           (ib x n2)
//       (G2) W2 = T^H W1            (ib x n2)
//       (G3) A22 -= V  W2           ((m-j0) x n2)
//
// with A22 = A(j0:m, j0+ib:n). ROWS START AT j0, NOT AT j0+ib: the block
// reflector acts on every row the panel spans, and V is (m - j0) x ib.
//
// WHY THE UPDATE IS (I - V T^H V^H) AND NOT (I - V T V^H). Q_block =
// H_1 H_2 ... H_ib = I - V T V^H is what larft's forward/columnwise convention
// builds, and the FACTORISATION applies its conjugate transpose:
// A22 := H_ib^H ... H_1^H A22 = Q_block^H A22 = (I - V T^H V^H) A22. This is the
// same t_eff = ConjTrans that ormqr_blocked.cc:534 uses for
// (Side::Left, transpose_apply), and it is invisible for real scalars -- i.e.
// getting it wrong passes every float and double test and only breaks complex.
//
// ---------------------------------------------------------------------------
// WHAT WP5's SCAFFOLDING GOT WRONG, RECORDED HERE RATHER THAN QUIETLY FIXED
// ---------------------------------------------------------------------------
// geqrf_native.hh and route_geqrf.hh both say "the blocked driver's panel leaf
// IS the CTA device function, so the crossover between the two tiers is a
// capacity and not a tuned guess". Half of that is true and half of it cannot
// be. The panel leaf IS the same DEVICE FUNCTION -- geqr2_panel_device, one
// body, shared through geqrf_cta_device.hh. It is NOT always the same
// RESIDENCY: a blocked panel is (m - j0) x nb with m unbounded, and a 1024 x 32
// float panel is 128 KB against a 97 KB local-memory budget. So the leaf has two
// accessors, chooses between them per panel by the same geqrf_cta_fits predicate
// the route table's capacity uses, and reports which it took through
// geqrf_blocked_debug_params' high half. The alternative -- refusing to route
// any matrix whose leading panel is not resident -- would have made the blocked
// tier serve nothing it was written for.
//
// ---------------------------------------------------------------------------
// THE SHORT FINAL PANEL, and why the obvious test for it is vacuous here
// ---------------------------------------------------------------------------
// `ib = std::min(nb, k - j0)` is the ONLY place the ragged last panel is
// handled, and every derived extent below comes from `ib` and `n2`, never from
// `nb` -- the discipline potrf_blocked.cc:695 states and the sy2sb stage-1 bug
// (commit ec1a178) violated by deriving a trailing-update range from the block
// width instead of the actual panel width.
//
// A geqrf-SPECIFIC FACT THAT MAKES THAT ERROR CLASS SMALLER HERE, and it is
// worth stating so nobody tests for the wrong thing: supports() requires m >= n,
// so k = min(m, n) = n, so the LAST panel's trailing block is EMPTY
// (n2 = n - j0 - ib = 0). There is no trailing update after a short panel to get
// wrong. What a short panel still exercises is the leaf (ib columns, not nb),
// pack_v/larft at ib, and -- for m > n -- reflectors of length m - j0 that
// outrun the panel's own width. Hence:
//
//   * a short-final-panel test MUST use m > n, a middle panel, or complex.
//     WP5's baseline measured that on a SQUARE REAL matrix, deleting the last
//     reflector leaves the residual BIT-IDENTICAL (4.072e-07 float /
//     1.615e-15 double) because larfg returns tau = 0 for a 1x1 real trailing
//     reflector; the same break turns complex red at 2.137e-02.
//   * the residues that matter are n % nb in {0, 1, 2, ...}, not just
//     {0, nonzero}: ec1a178's failure needed n % kd >= 2.
//
// ---------------------------------------------------------------------------
// SUB-VIEWS
// ---------------------------------------------------------------------------
// Every operand handed to the GEMM is built by the EXPLICIT 6-argument
// constructor with the parent's ld AND stride AND batch. Never
// operator()(Slice, Slice): matrix.hh:1136-1140 carries a comment saying a slice
// must not propagate the parent pointer array and the very next line does, and
// the constructor DEFAULTS stride to ld*cols when 0 is passed
// (matrix.cc:1839-1842), so a sub-view of ib columns built without an explicit
// stride silently gets stride = ld*ib and every batch item after the first reads
// the wrong matrix. sytrd_blocked.cc and gebrd_blocked.cc do use the slice
// spelling and get away with it only because their GEMMs route to
// strided-batched vendor kernels; WP3 and WP4 both worked around it instead.
//
// ---------------------------------------------------------------------------
// PERFORMANCE, INHERITED AND NOT RE-DERIVED (docs/perf/qr.md#the-vendor-baseline)
// ---------------------------------------------------------------------------
//  * EFFORT GOES TO THE PANEL. Both WY trailing GEMMs summed over all 18 panels
//    of a real N=1024, nb=56, batch=128 factorisation cost 33.40 ms vendor-free
//    for float against cuSOLVER's 2109.8 ms for the WHOLE call -- 63.2x
//    headroom, 4.3x in the worst type (cdouble).
//  * THE COMPLEX DEFICIT IS NOT WP5's TO FIX. A vendor-free build pays 2.55x
//    (float), 1.00x (double), 2.61x (cfloat), 2.01x (cdouble) on the BLAS-3
//    core, essentially all of it in G1 (4.81x / 1.00x / 4.99x / 3.12x) because
//    gemm_kernels.cc:470-482 short-circuits every transposed form to
//    Direct/Tiled16 and route_gemm.hh:113-114 refuses complex outright.
//    Recorded and moved past, as WP4 did.
//  * G3 IS AT PARITY WITH cuBLAS FOR EVERY TYPE ONLY IF nb >= 32:
//    gemm_kernels.cc:700 gates the complex wide-scalar kernel on min_dim >= 32
//    and min_dim of the NN update IS the block width. At nb = 24 complex G3
//    falls to Tiled16 and costs 1.72-2.30x. That makes "nb >= 32 for complex" a
//    MECHANICAL requirement, not a preference.

#include "geqrf_native.hh"
#include "larft_wy.hh"

#include "../sycl/gemm_kernels.hh"
#include "../queue.hh"
#include "../util/template-instantiations.hh"

#include <batchlas/util/mempool.hh>

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace batchlas {

// The caller tag naming this TU's copies of the shared WY kernels. At namespace
// scope for the reason larft_wy.hh gives: these are the SYCL kernel names, and
// ormqr_blocked.cc (a different device-code cluster) instantiates the same
// templates with its own tag.
struct GeqrfWyTag;

namespace sycl_geqrf {

namespace {

// ConjTrans for complex, Trans for real. NOT ConjTrans unconditionally: they are
// mathematically identical on a real scalar, but gemm_kernels.cc:472 gates the
// float transposed-register escape on `transA == Transpose::Trans`, so spelling
// it ConjTrans for float would refuse a kernel for nothing. ormqr_blocked.cc:526
// spells it ConjTrans for both and pays that.
template <typename T>
inline constexpr Transpose kConjT =
    batchlas::internal::is_complex<T>::value ? Transpose::ConjTrans : Transpose::Trans;

// THE BLOCK WIDTH. ONE pure function, called by the driver, by the buffer-size
// query and by geqrf_blocked_debug_params -- so a test cannot be told a blocking
// the call does not use, and the query cannot size for one width while the call
// runs another (potrf_native.hh:230-236 records what that costs: an unhandled
// throw on a call the route table had promised).
//
// MEASURED, and deliberately NOT tuning::ormqr_block_size_for_n
// (docs/perf/qr.md#block-width-evidence). That ladder's 16/16/24/48/56 by
// A.rows() was tuned on CUDA/float ONLY -- evaluation/tuning/tune.py:494 takes a
// single --type for a whole run and the ormqr_blocked space has no type axis --
// and even in a VENDOR-PRESENT build the shipped width costs double 1.32-1.41x
// and cdouble 1.46-1.47x. Three constraints, each measured:
//
//   * A MULTIPLE OF 16. G1's m IS the block width and G1 is Tiled16 for every
//     type in a vendor-free build; 24 and 56 lose everywhere.
//   * NEVER BELOW 32 FOR COMPLEX. gemm_kernels.cc:700 gates the complex
//     wide-scalar kernel on min_dim >= 32, and min_dim of G3 is the block width.
//   * NOT WIDER THAN 32, despite what the trailing GEMMs alone say. Measuring
//     G1+G3 in isolation shows a 2.7x float cliff at nb = 128 (the TN register
//     kernel becoming reachable at m >= 128); end to end nb = 128 is the WORST
//     width tested in BOTH builds, 83.0 ms against 36.8 at nb = 32, because the
//     panel cost a per-gemm probe cannot see dominates.
//
// It is keyed on the TYPE and clamped to k, NOT keyed on rows: sytrd_sy2sb.cc:
// 57-90 is an entire workaround for ormqr keying its table on the panel HEIGHT
// when the dimension that matters is the reflector count. Do not repeat it.
template <typename T>
constexpr int geqrf_nb_for_type() {
    // double wants 16 and everything else wants 32, measured in both builds at
    // both n = 256 and n = 1024. A single bucket table keyed only on n cannot
    // express that, which is the structural note in the baseline write-up.
    if constexpr (std::is_same_v<T, double>) {
        return 16;
    } else {
        return 32;
    }
}

template <typename T>
inline int geqrf_blocked_nb(int m, int n) {
    const int k = std::min(m, n);
    return std::max(1, std::min(geqrf_nb_for_type<T>(), k));
}

// ---------------------------------------------------------------------------
// WORKSPACE. Described ONCE, here, and replayed by both the query and the call
// (mempool.hh:165-190). Never hand-summed: mempool.hh:82-86 checks capacity from
// the UNALIGNED cursor while :118-120 advances only by the data extent, so an
// "exactly computed" figure fails the allocator's own check.
//
// MONOTONE NON-DECREASING IN (rows, cols, batch), which is a geqrf-ONLY contract
// and not a nicety: src/extensions/band_reduction.cc:1041-1044 sizes sytrd's
// band reduction against a (m_max x nb_max) NULL view and then CALLS geqrf at
// :595 with `Bsub`, a smaller sub-view. max() over routes at one shape says
// nothing about that. Every term below is a product of m, n, nb and batch, and
// nb is itself monotone in k = min(m, n), so the sum is monotone in all three.
// It also reads no element of A and no element of tau.
//
// V is sized at the FULL m, not at m - j0: the leading panel is the tallest and
// bounds every later one, which is the same argument sytrd_sy2sb.cc:329 makes
// and the same one that lets band_reduction size from a bigger shape than it
// calls with.
// ---------------------------------------------------------------------------
template <typename T>
struct GeqrfBlockedWs {
    Span<T> v;
    Span<T> t;
    Span<T> w1;
    Span<T> w2;
};

template <typename T>
GeqrfBlockedWs<T> geqrf_blocked_layout(Queue& ctx, BumpAllocator& pool,
                                       int m, int n, int nb, int batch) {
    GeqrfBlockedWs<T> ws;
    const std::size_t b = static_cast<std::size_t>(batch);

    // W1/W2 ARE SIZED ON THE WIDEST TRAILING BLOCK, n - nb, NOT ON n.
    //
    // The first panel's trailing block is A(:, nb:n) and every later one is
    // narrower, so n - nb bounds all of them; sizing on n over-allocates by a
    // third of this layout, which is not academic -- the facade's
    // max(native, vendor) means a caller at n = 64, batch = 8192 pays this
    // figure even when the CTA tier (workspace 0) is the route it takes, and
    // the cdouble cell there is a nine-figure number. It is still MONOTONE
    // NON-DECREASING in (rows, cols, batch), which is what
    // band_reduction.cc:1041-1044 requires: nb = min(nb_type, min(m,n)) so
    // n - nb is 0 for every n <= nb_type and then rises with n, and it does not
    // depend on m at all.
    //
    // The driver builds W1/W2 with THIS batch stride (nb * n2max), from the same
    // expression, so a view can never describe a shape the pool did not reserve.
    const std::size_t n2max = static_cast<std::size_t>(std::max(0, n - nb));

    ws.v = pool.allocate<T>(ctx, static_cast<std::size_t>(m) *
                                     static_cast<std::size_t>(nb) * b);
    ws.t = pool.allocate<T>(ctx, static_cast<std::size_t>(nb) *
                                     static_cast<std::size_t>(nb) * b);
    // max(1, ...) so a single-panel shape (n <= nb, no trailing update at all)
    // still gets a valid span rather than a zero-length allocation.
    ws.w1 = pool.allocate<T>(ctx, std::max<std::size_t>(
                                      1, static_cast<std::size_t>(nb) * n2max * b));
    ws.w2 = pool.allocate<T>(ctx, std::max<std::size_t>(
                                      1, static_cast<std::size_t>(nb) * n2max * b));
    return ws;
}

}  // namespace

// ---------------------------------------------------------------------------
// THE CAPABILITY FLAG. DEFINED HERE, beside the driver, for
// potrf_native.hh:81-92's reason: co-located, "the flag is true" and "the file
// is compiled" are the same fact, and no build can advertise a tier whose TU is
// missing from EXTENSIONS_CTA_SOURCES.
//
// TRUE for all four types as of WP5. That does NOT move any vendor-present
// traffic: RouteTable<Op::geqrf,T>::preferred() is still false everywhere, so a
// vendor-present build keeps taking cuSOLVER and only a vendor-free build (or an
// explicit BATCHLAS_GEQRF_ROUTE) reaches this driver -- route_resolve.hh:60-63.
// ---------------------------------------------------------------------------
template <> bool geqrf_blocked_available<float>()                { return true; }
template <> bool geqrf_blocked_available<double>()               { return true; }
template <> bool geqrf_blocked_available<std::complex<float>>()  { return true; }
template <> bool geqrf_blocked_available<std::complex<double>>() { return true; }

template <typename T>
std::size_t geqrf_blocked_buffer_size(Queue& ctx,
                                      const MatrixView<T, MatrixFormat::Dense>& A) {
    const int m = static_cast<int>(A.rows());
    const int n = static_cast<int>(A.cols());
    const int batch = static_cast<int>(A.batch_size());
    if (m < 1 || n < 1 || batch < 1) return 0;

    const int nb = geqrf_blocked_nb<T>(m, n);
    return workspace_bytes([&](BumpAllocator& pool) {
        return geqrf_blocked_layout<T>(ctx, pool, m, n, nb, batch);
    });
}

template <typename T>
unsigned geqrf_blocked_debug_params(Queue& ctx, int m, int n) {
    if (m < 1 || n < 1) return 0u;
    const int nb = geqrf_blocked_nb<T>(m, n);

    // WHICH LEAF THE LEADING PANEL TAKES, asked through the SAME predicate the
    // driver asks (geqrf_cta_fits) at the SAME budget, so this cannot report a
    // residency the call does not use.
    const auto dev = ctx.device();
    const std::size_t local_mem = dev.get_property(DeviceProperty::LOCAL_MEM_SIZE);
    const std::size_t budget = (local_mem > 4096) ? (local_mem - 4096) : 0;
    const int ib0 = std::min(nb, std::min(m, n));
    const unsigned leaf = geqrf_cta_fits<T>(m, ib0, budget) ? 1u : 2u;

    return (leaf << 16) | static_cast<unsigned>(nb);
}

// ---------------------------------------------------------------------------
// THE DRIVER.
// ---------------------------------------------------------------------------
template <typename T>
Event geqrf_blocked_dispatch(Queue& ctx,
                             const MatrixView<T, MatrixFormat::Dense>& A,
                             Span<T> tau,
                             Span<std::byte> workspace,
                             GeqrfTrailingGemm<T> trailing_gemm) {
    // Default the seam to the NATIVE kernel so this TU stands alone: a direct
    // caller (the tests, and any harness that must not be silently served by a
    // vendor) gets the native path with no dispatch dependency. The facade
    // injects the ROUTED gemm instead -- calling sycl_gemm::gemm_custom from a
    // driver TU unconditionally is WP3 step 16's recorded defect
    // (trsm_native.hh:82-104): it bypasses RouteTable<Op::gemm> entirely and
    // takes the native kernel even on shapes WP2 measured it losing.
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
    const int k = std::min(m, n);

    // Every gate RouteTable<Op::geqrf,T>::supports() applies to the Blocked arm,
    // re-applied because this entry point is reachable without the table -- and
    // it must be, for potrf_native.hh:126-141's reason: route_resolve.hh:101
    // falls through to automatic() when a forced route is unsupported, so a
    // pinned-route test that is wrong about one gate silently measures cuSOLVER.
    if (m < 1 || n < 1 || batch < 1) {
        throw std::invalid_argument("geqrf_blocked: degenerate extents");
    }
    if (m < n) {
        throw std::invalid_argument(
            "geqrf_blocked: m < n is not supported (route_geqrf.hh's supports() refuses it)");
    }
    if (A.is_heterogeneous()) {
        throw std::invalid_argument("geqrf_blocked: heterogeneous batch is not supported");
    }
    const auto dev = ctx.device();
    if (dev.type != DeviceType::GPU) {
        throw std::invalid_argument("geqrf_blocked: GPU queues only");
    }
    if (!dev.supports_sub_group_size(32)) {
        throw std::runtime_error(
            "geqrf_blocked: device does not offer sub-group size 32, which the panel leaf "
            "requires");
    }
    if (tau.size() < static_cast<std::size_t>(k) * static_cast<std::size_t>(batch)) {
        throw std::invalid_argument("geqrf_blocked: tau span is shorter than k * batch");
    }

    const int nb = geqrf_blocked_nb<T>(m, n);
    // The widest trailing block, and therefore the batch stride W1/W2 carry.
    // The SAME expression geqrf_blocked_layout uses -- one source, so the views
    // below cannot describe a shape the allocator did not reserve.
    const int n2max = std::max(0, n - nb);

    BumpAllocator pool(workspace);
    auto ws = geqrf_blocked_layout<T>(ctx, pool, m, n, nb, batch);

    const int ld = A.ld();
    const int stride = A.stride();
    T* const a_ptr = A.data_ptr();
    T* const tau_ptr = tau.data();

    // THE EXPLICIT 6-ARG SUB-VIEW. See the note at the top of this file for both
    // traps it removes. `ptrs = nullptr` deliberately: a null pointer array lets
    // a backend regenerate the correct one for this view, whereas the parent's
    // array points at the UNSLICED base addresses.
    auto sub = [&](int r0, int nr, int c0, int nc) {
        return MatrixView<T, MatrixFormat::Dense>(
            a_ptr + static_cast<std::ptrdiff_t>(c0) * ld + r0,
            nr, nc, ld, stride, batch, nullptr);
    };

    for (int j0 = 0; j0 < k; j0 += nb) {
        // THE SHORT FINAL PANEL is carried by this one std::min and nothing
        // else. Every extent below comes from `ib` and `n2`, never from `nb`.
        const int ib = std::min(nb, k - j0);
        const int j2 = j0 + ib;
        const int mp = m - j0;          // panel / V height, and A22's height
        const int n2 = n - j2;          // trailing columns; ZERO on the last panel

        // (P) THE PANEL. One device body, two residencies, chosen by capacity --
        // see geqrf_panel_factorize. tau's batch stride is k (the WHOLE matrix's
        // reflector count, which is geqrf's published contract, cublas.cc:1259)
        // and the offset is j0; deriving the stride from the panel's own
        // min(mp, ib) would scatter tau across the wrong slots for every item
        // after the first.
        (void)geqrf_panel_factorize<T>(ctx,
                                       a_ptr + static_cast<std::ptrdiff_t>(j0) * ld + j0,
                                       ld, stride, mp, ib, batch,
                                       tau_ptr, k, j0, nullptr);

        if (n2 <= 0) break;

        // The WY construction READS the panel the leaf just wrote. In-order
        // queues give that ordering for free; a caller may construct an
        // out-of-order one (sycl-device-queue.hh:254 defaults in_order = true but
        // it is a parameter), and every dependent boundary in this schedule
        // carries its own guard rather than relying on the two at the bottom.
        if (!ctx.in_order()) ctx.wait();

        // (L) V and T. V is packed into scratch rather than expressed as a
        // trmm-plus-gemm pair over the triangular top block: one m*ib copy per
        // panel against three more BLAS-3 calls and a Diag::Unit trmm, on a
        // schedule whose trailing GEMMs are already only 1.6% of the vendor's
        // total time.
        //
        // V IS PACKED CONTIGUOUSLY AT ld = mp, NOT AT THE PARENT ld = m. V does
        // not live in A -- it is our own scratch, allocated m*nb*batch at
        // ws.v -- so nothing forces it to carry a parent leading dimension, and
        // performance fact 1 (build every SUB-VIEW of A with the parent ld) does
        // not apply to a buffer this driver owns outright. Handing it out at
        // ld = m gave both trailing GEMMs a SHORT operand with a LONG stride,
        // which is the recorded "native GEMM collapses on strided ld" shape: at
        // j0 = 992 of a 1024-column factorisation the panel is 32 rows tall and
        // its columns were 4 KB apart. Over the second half of the panels
        // mp <= m/2, so the majority of the driver's V traffic was in that
        // regime. mp*ib <= m*nb, so this always fits the same allocation.
        //
        // ALL THREE SITES MUST AGREE -- the pack that writes V, the larft that
        // reads it, and the view the GEMMs read. They are kept adjacent and
        // share ld_v/stride_v below for exactly that reason.
        const int ld_v = mp;
        const int stride_v = mp * ib;

        MatrixView<T, MatrixFormat::Dense> Vblk(ws.v.data(), mp, ib, ld_v,
                                                stride_v, batch);
        (void)wy::pack_v_panel_batched<GeqrfWyTag, T>(
            ctx, ws.v.data(), ld_v, stride_v, A, j0, ib, m);

        if (!ctx.in_order()) ctx.wait();

        MatrixView<T, MatrixFormat::Dense> Tblk(ws.t.data(), ib, ib, nb,
                                                nb * nb, batch);
        // THE COMPILE-TIME `false` FORM. Passing a literal to the runtime
        // wrapper instantiated the device-BLAS larft for GeqrfWyTag as well --
        // 32 entry functions that can never launch, in the slowest-linking
        // library in the tree. See larft_wy.hh.
        (void)wy::larft_forward_columnwise_batched_t<GeqrfWyTag, T, false>(
            ctx, ws.t.data(), nb, nb * nb,
            ws.v.data(), ld_v, stride_v,
            mp, ib,
            tau_ptr, /*tau_stride=*/k, /*tau_offset=*/j0, batch);

        if (!ctx.in_order()) ctx.wait();

        const auto A22 = sub(j0, mp, j2, n2);
        // THE BATCH STRIDE IS THE LAYOUT'S, nb * n2max, NOT THIS PANEL'S
        // nb * n2. W1 and W2 are private scratch, so any stride the three GEMMs
        // AGREE on is arithmetically fine -- which is why a kernel break that
        // changed it to nb*n2 turned nothing red (recorded in
        // docs/perf/qr.md#break-sweeps as a break that did not
        // discriminate, and why). It must nonetheless match what
        // geqrf_blocked_layout reserved, or a later panel walks off the end of
        // the allocation: n2 SHRINKS as j0 advances, so a per-panel stride would
        // fit while a fixed one derived from a later panel would not.
        MatrixView<T, MatrixFormat::Dense> W1(ws.w1.data(), ib, n2, nb,
                                              nb * n2max, batch);
        MatrixView<T, MatrixFormat::Dense> W2(ws.w2.data(), ib, n2, nb,
                                              nb * n2max, batch);

        // (G1) W1 = V^H A22.
        trailing_gemm(ctx, Vblk, A22, W1, T(1), T(0), kConjT<T>, Transpose::NoTrans,
                      ComputePrecision::Default);
        if (!ctx.in_order()) ctx.wait();

        // (G2) W2 = T^H W1. See the derivation at the top of this file for why
        // the conjugate transpose and not T itself.
        trailing_gemm(ctx, Tblk, W1, W2, T(1), T(0), kConjT<T>, Transpose::NoTrans,
                      ComputePrecision::Default);
        if (!ctx.in_order()) ctx.wait();

        // (G3) A22 -= V W2.
        trailing_gemm(ctx, Vblk, W2, A22, T(-1), T(1), Transpose::NoTrans,
                      Transpose::NoTrans, ComputePrecision::Default);

        // The next panel's leaf reads what this update just wrote, and the next
        // pack_v OVERWRITES the V this update is still reading.
        if (!ctx.in_order()) ctx.wait();
    }

    return ctx.get_event();
}

// ---------------------------------------------------------------------------
// Instantiation: PER SCALAR TYPE ONLY, no Backend cross-product, exactly as
// potrf_blocked.cc:838-846. Everything that needs a Backend arrives injected.
// ---------------------------------------------------------------------------
#define BATCHLAS_GEQRF_BLOCKED_INSTANTIATE(T)                                                 \
    template std::size_t geqrf_blocked_buffer_size<T>(                                        \
        Queue&, const MatrixView<T, MatrixFormat::Dense>&);                                   \
    template unsigned geqrf_blocked_debug_params<T>(Queue&, int, int);                        \
    template Event geqrf_blocked_dispatch<T>(Queue&,                                          \
                                             const MatrixView<T, MatrixFormat::Dense>&,       \
                                             Span<T>, Span<std::byte>, GeqrfTrailingGemm<T>);

BATCHLAS_GEQRF_BLOCKED_INSTANTIATE(float)
BATCHLAS_GEQRF_BLOCKED_INSTANTIATE(double)
BATCHLAS_GEQRF_BLOCKED_INSTANTIATE(std::complex<float>)
BATCHLAS_GEQRF_BLOCKED_INSTANTIATE(std::complex<double>)

#undef BATCHLAS_GEQRF_BLOCKED_INSTANTIATE

}  // namespace sycl_geqrf
}  // namespace batchlas
