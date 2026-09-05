// Native batched GETRS -- the row interchange plus two ROUTED triangular solves.
//
// This is the small file. Both solves are the routed trsm, injected from the
// facade; the only kernel getrs owns is the interchange, and even that is the
// shared lu_laswp.hh template.
//
// getrs_native.cc sits in EXTENSIONS_FACTORIZATION_SOURCES and NOT in
// EXTENSIONS_CTA_SOURCES: it shares no device symbol with the getrf pair
// (src/extensions/CMakeLists.txt:77-89's cluster rule, orgqr_blocked.cc's
// precedent). lu_laswp.hh is a TEMPLATE ON A TAG precisely so that this stays
// true -- see its opening note.
//
// ===========================================================================
// THE MEASUREMENT THAT GOVERNS THIS FILE IS A NEGATIVE RESULT, AND IT IS NOT
// PAPERED OVER.
//
// Composed "row permutation + two routed trsm" against cublas?getrsBatched, at
// saturating batch, in process, against a host oracle
// (experiments/wp6_lu/baseline/grid_norm.csv, summary.txt):
//
//   nrhs = 1  : GEOMEAN 0.36x over 28 cells (4 types x 7 orders). 25 LOSSES.
//               n(batch)   float double cfloat cdouble
//                32(8192)   0.20   0.19   0.10   0.09
//               128(4096)   0.41   0.23   0.34   0.14
//               512(512)    0.66   0.32   0.59   0.26
//              2048(32)     0.94   1.14   0.87   1.07
//               The only wins are at n=2048, against a vendor which is NOT
//               SATURATED there (64x the work for 1.52x the time, batch 1 to 64).
//   nrhs = 64 : geomean 1.17x with this interchange walk, 1.55x with the
//               permutation collapsed to a gather. 20 and 25 wins of 28.
//
// THE nrhs=1 LOSS IS STRUCTURAL: at one right-hand side the permutation is a
// rounding error and the whole loss is in the triangular solves, because trsm's
// blocked driver amortises a panel over many columns and one column gives it
// nothing to amortise. The permutation strategy is irrelevant to it -- 0.36x
// either way. So this arm SHIPS ROUTE-NEUTRAL (preferred() is false for every
// shape), which is a legitimate outcome under the campaign's gate: "a native
// kernel that is correct but slower than the vendor ships route-neutral, it does
// not become the default". It exists so that a vendor-free build HAS a getrs,
// and getrs has no internal consumer at all -- src/extensions/inv.cc:48-49 calls
// getrf then getri, and the public layer reaches getrs only through
// linalg::solve (linalg-ops.hh:343-344), at nrhs = 1.
//
// WHY THE INTERCHANGE WALK AND NOT THE GATHER, stated as a decision rather than
// made by accident. The gather is worth +0.38 of geomean at nrhs = 64 and
// NOTHING at nrhs = 1, and it costs an OUT-OF-PLACE RHS plus an int32[n] per item
// for the collapsed permutation.
//
// THE BUFFER FIGURE MUST BE READ AT THE nrhs THAT DECIDES, and the one this note
// used to quote was not. 67,371,008 B is the buffer at n=2048, nrhs=64,
// batch=32 -- but nrhs=64 is the case where the gather WINS, so it never carried
// the decision. At nrhs=1, the only nrhs the library actually issues, the same
// buffer is n*batch*sizeof(T) = 262,144 B, i.e. 257x smaller, and the argument
// against buying it is NOT the memory. It is that at nrhs=1 the gather buys
// nothing measurable: the loss there is in the two triangular solves (0.36x
// either way), not in the interchange. getri gets the same collapse for FREE (it
// writes P straight into C, no permutation kernel and no workspace at all) and
// takes it; getrs would have to buy it, at the one nrhs where the library
// actually calls it the purchase buys nothing, and the workspace would enter
// the facade's max(native, vendor) for every getrs in the process. The walk
// therefore ships, the buffer-size query stays 0, and the gather is left to the
// routing step -- where a preferred() window on GetrsShape::nrhs() is the thing
// that would justify paying for it.

#include "getrs_native.hh"
#include "lu_laswp.hh"

#include "../queue.hh"
#include "../util/template-instantiations.hh"

#include <batchlas/util/mempool.hh>

#include <complex>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace batchlas {
namespace sycl_getrs {

namespace {

// The per-TU tag that gives this cluster its own instantiation of the shared
// LASWP kernel. See lu_laswp.hh.
struct GetrsLaswpTag {};

}  // namespace

// ---------------------------------------------------------------------------
// THE CAPABILITY FLAG. TRUE for all four types.
//
// It moves no vendor-present traffic: RouteTable<Op::getrs,T>::preferred() is
// false everywhere, so a vendor-present build keeps taking cublas?getrsBatched
// and this driver is reachable only through BATCHLAS_GETRS_ROUTE, through the
// direct entry point, or in a vendor-free build (route_resolve.hh:60-63).
//
// DEFINED HERE, beside the driver, for potrf_native.hh:81-92's reason: these are
// full explicit specialisations and link from wherever they sit, so co-locating
// them is what makes "the flag is true" and "the file is compiled" the same fact.
// ---------------------------------------------------------------------------
template <> bool getrs_blocked_available<float>()                { return true; }
template <> bool getrs_blocked_available<double>()               { return true; }
template <> bool getrs_blocked_available<std::complex<float>>()  { return true; }
template <> bool getrs_blocked_available<std::complex<double>>() { return true; }

// ---------------------------------------------------------------------------
// WORKSPACE. ZERO, in every mode, and that is a consequence of the strategy
// decision above rather than a coincidence: the interchange is applied IN PLACE
// to the caller's B, and the routed trsm takes no workspace at all (verified
// against functions/trsm.hh -- there is no trsm_buffer_size anywhere in this
// tree, which is also why no buffer-size twin is injected alongside the solve
// seam). getrs has no `info` argument, so unlike getrf there is not even a
// fallback status span to reserve.
//
// It still takes B and transA, because the SIGNATURE is the contract: a later
// gather strategy scales with nrhs and applies its permutation to a different
// buffer depending on transA, and a query that could not see either would be
// right only for the strategy that shipped.
//
// Zero is returned directly rather than through a BumpAllocator::measuring()
// replay because there is no layout to replay; 0 is trivially an alignment
// multiple, which is the property the facade's max(native, vendor) needs
// (mempool.hh:45-58). Any term added later must arrive through workspace_bytes.
//
// It dereferences nothing: A and B arrive with null data pointers from a
// measuring pass.
// ---------------------------------------------------------------------------
template <typename T>
std::size_t getrs_blocked_buffer_size(Queue&,
                                      const MatrixView<T, MatrixFormat::Dense>&,
                                      const MatrixView<T, MatrixFormat::Dense>&,
                                      Transpose) {
    return 0;
}

// ---------------------------------------------------------------------------
// THE DRIVER.
//
// A = F^{-1} L U, where F is the interchange sequence applied FORWARDS (F v
// permutes v by ipiv, in order) -- that is LAPACK's convention: ?GETRF returns
// ipiv such that applying it to A yields L U.
//
//   NoTrans   : A x = b  <=>  L U x = F b
//               apply F to B, solve L (unit lower), solve U (non-unit upper).
//
//   Trans     : A^T = (F^{-1} L U)^T = U^T L^T F^{-T} = U^T L^T F, because a
//               permutation matrix is orthogonal, so F^{-T} = F.
//               A^T x = b  <=>  U^T L^T (F x) = b
//               solve U^T, solve L^T, then x = F^{-1} w -- and F^{-1} is the
//               SAME list walked BACKWARDS, since each transposition is its own
//               inverse and only the order reverses.
//
//   ConjTrans : identical, with H for T; F is real so it is unchanged.
//
// THE TWO SOLVES SWAP ORDER **AND** THE PERMUTATION MOVES TO THE OUTPUT, IN
// REVERSE. Getting either half wrong is a silently wrong answer that no NoTrans
// test can see -- and the scaffolding's break B8 measured that no test in this
// suite issues a Trans getrs at all today, so the guard for this has to be
// written with the kernel.
// ---------------------------------------------------------------------------
template <typename T>
Event getrs_blocked_dispatch(Queue& ctx,
                             const MatrixView<T, MatrixFormat::Dense>& A,
                             const MatrixView<T, MatrixFormat::Dense>& B,
                             Transpose transA,
                             Span<int64_t> pivots,
                             Span<std::byte> workspace,
                             GetrsSolveTrsm<T> solve_trsm) {
    static_cast<void>(workspace);   // this arm needs none; see the query above

    const int n = static_cast<int>(A.rows());
    const int nrhs = static_cast<int>(B.cols());
    const int batch = static_cast<int>(A.batch_size());

    // Every gate RouteTable<Op::getrs,T>::supports() applies, re-applied because
    // this entry point is reachable WITHOUT the table -- and it must be, for
    // potrf_native.hh:126-141's reason: route_resolve.hh:165 falls through to
    // automatic() when a forced route is unsupported, so a pinned-route test that
    // is wrong about one gate silently measures cuBLAS and passes green.
    if (n < 1 || nrhs < 1 || batch < 1) {
        throw std::invalid_argument("getrs_blocked: degenerate extents");
    }
    if (A.rows() != A.cols()) {
        throw std::invalid_argument("getrs_blocked: A must be square");
    }
    if (A.rows() != B.rows()) {
        throw std::invalid_argument("getrs_blocked: B must have A.rows() rows");
    }
    if (A.batch_size() != B.batch_size()) {
        throw std::invalid_argument("getrs_blocked: A and B must agree on batch size");
    }
    if (A.is_heterogeneous() || B.is_heterogeneous()) {
        throw std::invalid_argument("getrs_blocked: heterogeneous batch is not supported");
    }
    const auto dev = ctx.device();
    if (dev.type != DeviceType::GPU) {
        throw std::invalid_argument("getrs_blocked: GPU queues only");
    }
    if (!dev.supports_sub_group_size(32)) {
        // Enumerated, never get_property(MAX_SUB_GROUP_SIZE) >= 32 -- that
        // property returns sub_group_sizes()[0] (queue-impl.cc:325), so the weak
        // test refuses a {8,16,32} device and ACCEPTS a {64} one. The gate is
        // transcribed from supports() and must match it exactly even though this
        // file's own kernel carries no sub-group requirement: the routed trsm it
        // calls does.
        throw std::runtime_error(
            "getrs_blocked: device does not offer sub-group size 32");
    }
    if (pivots.size() < static_cast<std::size_t>(n) * static_cast<std::size_t>(batch)) {
        throw std::invalid_argument("getrs_blocked: pivot span is shorter than n * batch");
    }
    if (!solve_trsm) {
        // AN ABSENT INJECTION THROWS rather than silently reaching for a native
        // trsm entry point: the two solves ARE this op, and picking trsm's arm
        // here would be WP3 step 16's defect re-created (trsm_native.hh:82-104,
        // fix at level3.cc:186-231). A direct caller injects
        // trsm<Backend::CUDA, T> itself, which is still a call no vendor getrs
        // can serve.
        throw std::invalid_argument(
            "getrs_blocked: the solve seam is empty. Inject the ROUTED batchlas::trsm "
            "(the facade does; a direct caller must too) -- this driver deliberately has "
            "no native fallback, so that the router, and not this file, chooses the trsm "
            "arm.");
    }

    // PACKED 1-BASED int32 -- the format cublas.cc:1476 and rocsolver.cc:227 both
    // read through pivots.as_span<int>(), and the one a native getrf writes. See
    // getrf_native.hh's PIVOT CONTRACT: the two ops have independent env
    // variables and independent preferred() windows, so every mixture of native
    // and vendor arms is reachable and they must agree bit for bit.
    auto piv_i32 = pivots.as_span<int>();

    if (transA == Transpose::NoTrans) {
        (void)lu_native::lu_laswp_launch<GetrsLaswpTag, T>(
            ctx, B.data_ptr(), B.ld(), B.stride(), nrhs, batch,
            piv_i32.data(), /*piv_stride=*/n, /*k0=*/0, /*k1=*/n, /*forward=*/true);
        // In-order queues give the ordering for free; an out-of-order one does
        // not, and every dependent boundary in this schedule carries its guard.
        if (!ctx.in_order()) ctx.wait();

        // alpha comes THIRD in the public trsm (functions/trsm.hh:100-108); the
        // old spelling is a DELETED overload at :121-138 so a stale call cannot
        // silently compile into a wrong answer.
        (void)solve_trsm(ctx, A, B, T(1), Side::Left, Uplo::Lower,
                         Transpose::NoTrans, Diag::Unit);
        if (!ctx.in_order()) ctx.wait();
        return solve_trsm(ctx, A, B, T(1), Side::Left, Uplo::Upper,
                          Transpose::NoTrans, Diag::NonUnit);
    }

    // Trans and ConjTrans. The transpose flag is passed THROUGH to both solves --
    // ConjTrans on a real type is Trans, which the trsm layer already handles.
    (void)solve_trsm(ctx, A, B, T(1), Side::Left, Uplo::Upper, transA, Diag::NonUnit);
    if (!ctx.in_order()) ctx.wait();
    (void)solve_trsm(ctx, A, B, T(1), Side::Left, Uplo::Lower, transA, Diag::Unit);
    if (!ctx.in_order()) ctx.wait();

    // F^{-1}: the SAME list, walked BACKWARDS. This is the half of the transposed
    // case that a NoTrans test cannot see.
    return lu_native::lu_laswp_launch<GetrsLaswpTag, T>(
        ctx, B.data_ptr(), B.ld(), B.stride(), nrhs, batch,
        piv_i32.data(), /*piv_stride=*/n, /*k0=*/0, /*k1=*/n, /*forward=*/false);
}

// ---------------------------------------------------------------------------
// Instantiation: PER SCALAR TYPE ONLY, no Backend cross-product. Everything that
// needs a Backend arrives injected.
// ---------------------------------------------------------------------------
#define BATCHLAS_GETRS_INSTANTIATE(T)                                                      \
    template std::size_t getrs_blocked_buffer_size<T>(                                     \
        Queue&, const MatrixView<T, MatrixFormat::Dense>&,                                 \
        const MatrixView<T, MatrixFormat::Dense>&, Transpose);                             \
    template Event getrs_blocked_dispatch<T>(                                              \
        Queue&, const MatrixView<T, MatrixFormat::Dense>&,                                 \
        const MatrixView<T, MatrixFormat::Dense>&, Transpose,                              \
        Span<int64_t>, Span<std::byte>, GetrsSolveTrsm<T>);

BATCHLAS_GETRS_INSTANTIATE(float)
BATCHLAS_GETRS_INSTANTIATE(double)
BATCHLAS_GETRS_INSTANTIATE(std::complex<float>)
BATCHLAS_GETRS_INSTANTIATE(std::complex<double>)

#undef BATCHLAS_GETRS_INSTANTIATE

}  // namespace sycl_getrs
}  // namespace batchlas
