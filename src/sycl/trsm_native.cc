// Native batched TRSM — the kernel translation unit.
//
// WP3 steps 3-4: V1, both sides, real types. See WP3_TRSM_SPEC_CORRECTIONS.md
// first, then WP3_TRSM_SPEC.md §2-§3.
//
// NOT ROUTED. trsm_cta_max_n<T>() still returns 0 for every type, so
// RouteTable<Op::trsm,T>::supports() reports both native routes unsupported and
// nothing in the library can reach this code. It is exercised by a direct call
// from tests. The capacities become non-zero only once the register probe has
// read the instantiations — that is the spec's own gate and it is the point of
// keeping this step unrouted.
//
// THE DECOMPOSITION. One work-group per (matrix, block of independent solves);
// one thread per INDEPENDENT SOLVE. The solution vector lives in that thread's
// registers as `T x[N]` with N a COMPILE-TIME bucket >= n, and both loops are
// fully unrolled so every register index is a compile-time constant. That is
// not a preference: a per-thread array indexed by a runtime induction variable
// is placed in .local by ptxas, which turns a DRAM-bound kernel into an
// L1-bound one and voids the design. Rows n..N-1 are zero-padded during staging
// (Lc(s,t)=0, Lc(s,s)=1, rd[s]=1) so the unrolled tail computes zeros rather
// than branching — the sytrd_cta idiom.
//
// THE TWO SIDES DIFFER IN EXACTLY THREE PLACES, and the kernel is templated on
// Side rather than duplicated so they cannot drift apart:
//
//   1. q            Left: B.cols()          Right: B.rows()
//   2. Lc(s,t)      Left: opA(rho(s),rho(t))  Right: opA(rho(t),rho(s))
//                   -- THE OPERAND ORDER IS SWAPPED. Invisible on a symmetric
//                      triangle, wrong on every other one.
//   3. the RHS accessor stride pair (ds, du):
//                   Left:  b0 = fwd?0:(n-1),      ds = +-1,   du = ldb
//                   Right: b0 = fwd?0:(n-1)*ldb,  ds = +-ldb, du = 1
//
// Right went first because its du == 1 makes lanes touch consecutive addresses,
// so the register question was answered without the coalescing question in the
// way. Left has du == ldb, i.e. lanes stride by ldb, and §3.4 specifies an SLM
// transpose staging tile to fix that.
//
// THAT TILE IS DELIBERATELY NOT IN THIS STEP. It is a performance mitigation
// for a cost the spec PREDICTS ("8x over-fetch") and has never measured, and
// its own sizing formula in §4.1 is off by a factor that writes 127 elements
// out of bounds (WP3_TRSM_SPEC_CORRECTIONS.md finding 4). Correctness first,
// then measure the over-fetch, then add the tile if the measurement asks for
// it. Landing an unmeasured optimisation alongside a new kernel would make both
// unattributable.

#include "trsm_native.hh"

#include "../linalg-impl.hh"
#include "device_scalar.hh"
#include "gemm_kernels.hh"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <complex>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace batchlas::sycl_trsm {

namespace {

// ---------------------------------------------------------------------------
// Canonicalisation — WP3_TRSM_SPEC.md §2.1, transcribed once.
//
// The 24 (side, uplo, transA, diag) combinations fold into ONE recurrence over a
// canonical unit-lower factor Lc and a canonical RHS accessor. Both in-tree
// references perform the same fold (netlib_lapack.cc:445-449,
// cublas.cc:1134-1137) — which is exactly why a test that transcribes either of
// them proves nothing, and why the test oracle is an independent multiply-back.
// ---------------------------------------------------------------------------
struct Canonical {
    bool do_trans;
    bool do_conj;
    bool op_is_lower;
    bool unit;
    bool fwd;
};

inline Canonical canonicalise(Side side, Uplo uplo, Transpose transA, Diag diag) {
    Canonical c{};
    c.do_trans = (transA != Transpose::NoTrans);
    c.do_conj = (transA == Transpose::ConjTrans);
    c.op_is_lower = (uplo == Uplo::Lower) ? !c.do_trans : c.do_trans;
    c.unit = (diag == Diag::Unit);
    // fwd is the direction the canonical recurrence marches. Getting this
    // backwards is silent: it solves a different triangle and still returns.
    c.fwd = (side == Side::Left) ? c.op_is_lower : !c.op_is_lower;
    return c;
}

// The smallest compile-time bucket >= n, or 0 if there is none.
//
// RETURNING 0 RATHER THAN THE NEXT POWER OF TWO IS THE POINT. This used to
// return 64 for any n > 32, and the dispatch switch below collapsed 64 onto the
// N=32 instantiation via its `default:` label -- so a 33-order solve ran with
// N=32 and silently solved the leading 32x32 system, leaving the last row of B
// untouched. Nothing caught it: the staging pad test (s >= n) cannot fire when
// N < n, the recurrence simply stops early, and the store loop writes only the
// rows it computed. It was unreachable through the facade because supports(CTA)
// caps the order at trsm_cta_max_n, but the direct entry is exactly what V2
// calls on its diagonal blocks.
//
// There is no N=64 bucket by measurement, not by omission: the register probe
// put x[64] in local memory for both real types (256 B / 512 B stack frame,
// zero spill), which voids V1's register residency. n > 32 is V2's job.
inline int smallest_bucket_ge(int n) {
    if (n <= 8) return 8;
    if (n <= 16) return 16;
    if (n <= 32) return 32;
    return 0;
}

// THE CAP STAYS AT 32, AND N=64 WAS RE-TESTED RATHER THAN ASSUMED.
//
// The original N=64 rejection predates the Side::Left staging tile, which cut
// float Side::Left from 114 registers to 53 at N=32 -- so the arithmetic that
// killed it no longer described the kernel and it was worth re-measuring. It
// still fails, and by more than before (scripts/register_probe.sh, float):
//
//   N=64 Left    72 registers, 456 B stack frame, 0 B spill   *** FAIL
//   N=64 Right  119 registers, 256 B stack frame, 0 B spill   *** FAIL
//
// Zero spill with a non-zero frame is the signature of x[] living in local
// memory, which voids the design. Left is WORSE than Right because the staging
// tile adds live state of its own, so the tile does not pay for the bigger
// accumulator -- it competes with it.
//
// This matters beyond the kernel: nb is what V2 blocks on, and the traffic
// model (B elements touched per batch item, units of q, at n=512) is
//   NB=32 -> 5824    NB=128,nb=32 -> 4096    NB=128,nb=64 -> 3328
//   NB=128,nb=128 -> 2560, against an ideal of 1024.
// So the remaining large-order gap is bounded by the CTA capacity, and closing
// it needs an nb of 64+ that a one-solve-per-work-item design cannot hold. The
// route to it is a cooperative solve (W work-items per solve exchanging x_s by
// sub-group broadcast, so each holds nb/W elements), not a bigger array.
template <typename D>
constexpr int trsm_max_bucket() {
    return 32;
}

// Packed lower-triangle index, row-major by s: N(N+1)/2 elements.
// All threads read the same Lc(s,t) at the same step, so this is an SLM
// BROADCAST — bank layout is irrelevant to conflicts here.
constexpr int tri_idx(int s, int t) { return s * (s + 1) / 2 + t; }

// ---------------------------------------------------------------------------
// Side::Left staging tile height, in ELEMENTS (spec S3.4).
//
// WHY THIS EXISTS. B is column-major, and for Side::Left thread u owns COLUMN u
// -- so at step s the lanes of a warp read B(rho(s), u0+lane), addresses ldb
// apart. Measured with ncu on float, n=32, q=1024, batch=512:
//
//        load sectors/request   store sectors/request   dram vs floor   time
//   Left       31.39                   32.00                0.85x      0.517 ms
//   Right       5.13                    4.00                0.75x      0.141 ms
//
// A fully coalesced 32-lane float load moves 128 B = 4 sectors in one request,
// so 31.4 is 7.85x over-fetch -- almost exactly the 8x the spec predicted.
//
// BUT NOT AT THE LEVEL THE SPEC SAID. S3.4 calls it "8x over-fetch on both the
// read and the write-allocate", which reads as DRAM traffic; DRAM is measured
// at 0.75-0.85x of the analytic floor 2*q*n*sizeof(T)*batch, i.e. BELOW it. The
// bytes lane u skips at step s are the bytes lane u wants at steps s+1..s+7,
// and they are still in cache when it gets there. The defect is entirely LSU/L1
// transaction COUNT, and the fix is the same one either way -- but "we are
// short of DRAM bandwidth" would have been the wrong thing to optimise, and the
// same misreading has already cost this repo one panel kernel.
//
// HOW BIG. One 32 B sector holds 32/sizeof(T) elements, so a tile that is at
// least that tall makes each lane-group's read exactly fill its sectors. That
// is 8 for float and only 2 for complex<double> -- which is itself the reason
// float is the ONLY type that loses this race: a 16-byte scalar is already
// 2-lanes-per-sector and can over-fetch by at most 2x.
//
// The value is one past that, 2 sectors' worth, because 64 B contiguous per
// lane-group also fills half a 128 B cache line and costs nothing extra in SLM
// terms at these sizes. SLM is (rows+1) * wg * sizeof(T), and only the two real
// types stage (see trsm_stage_left below): 17.4 KB for float and 18.4 KB for
// double at wg=256. Neither is the binding occupancy limit -- that stays
// registers, see the static_assert in the launcher.
template <typename D>
constexpr int trsm_stage_rows() {
    return sizeof(D) <= 4 ? 16 : 8;
}

// WHICH TYPES STAGE, and this is a MEASURED exclusion, not a guess.
//
// Staging costs the complex instantiations their register residency. The round
// structure makes x[] live across a work-group barrier and indexed by
// s = k*STEP + j, and for the wide bodies the compiler stops fully unrolling
// the nested loop, so the array goes to local memory. From
// scripts/register_probe.sh, Side::Left, with staging applied to all four:
//
//   type              N=8        N=16       N=32          verdict
//   float           27 / 0 B   36 / 0 B   53 /   0 B      fine (was 114 regs)
//   double          40 / 0 B   60 / 0 B   90 /   0 B      fine (was 153 regs)
//   complex<float>  40 / 0 B   56 / 0 B   72 / 464 B      *** x[] IN LOCAL MEM
//   complex<double> 70 / 16 B 104 / 16 B 170 / 232 B      *** x[] IN LOCAL MEM
//
// (registers / stack frame; spill stores and loads are zero in every row, which
// is exactly why the frame and not the spill counter is the gate -- a frame
// with no spill IS the accumulator array sitting in local memory.)
//
// And they do not need it. The over-fetch factor is 32/sizeof(T) lanes per
// sector, so a 16-byte scalar can be at most 2x off; the step-9 grid has
// complex winning Side::Left at every order (2.27-21.91x) and double winning at
// every order too. float is the only type that loses, and it loses precisely
// because 32/4 = 8.
template <typename D>
constexpr bool trsm_stage_left() {
    return sizeof(D) <= 8 && !sycl_device::dev_is_complex_v<D>;
}

template <typename T>
inline bool finite_recip(T d, T& out) {
    const T r = T(1) / d;
    out = r;
    return sycl::isfinite(r);
}

template <typename T, int N, Side SideV>
class TrsmCtaKernel;

}  // namespace

// ---------------------------------------------------------------------------
// V1 launcher. Direct-call only at this step; nothing routes here.
// ---------------------------------------------------------------------------
template <typename T, int N, Side SideV>
Event trsm_native_v1(Queue& ctx,
                     const MatrixView<T, MatrixFormat::Dense>& A,
                     const MatrixView<T, MatrixFormat::Dense>& B,
                     T alpha,
                     Uplo uplo,
                     Transpose transA,
                     Diag diag) {
    // The whole kernel runs on the POD device scalar. std::complex is re-typed
    // here, at the pointer boundary, and never crosses into the kernel body --
    // including alpha, which is reinterpreted exactly as the operands are.
    using D = typename sycl_device::DevMap<T>::type;
    static_assert(sizeof(D) == sizeof(T), "device scalar must be layout-compatible");

    const Canonical can = canonicalise(SideV, uplo, transA, diag);

    const int n = static_cast<int>(A.rows());
    const int q = static_cast<int>(SideV == Side::Left ? B.cols() : B.rows());
    const int bs = static_cast<int>(A.batch_size());

    const auto dev = ctx.device();
    const int max_wg = static_cast<int>(dev.get_property(DeviceProperty::MAX_WORK_GROUP_SIZE));
    const int cu = static_cast<int>(dev.get_property(DeviceProperty::MAX_COMPUTE_UNITS));

    // THE LADDER MUST NOT GO ABOVE 256. The register probe measures the worst
    // instantiation (complex<double>, N=32) at 226 registers per thread, so
    // 226 * 256 = 57,856 against the hard 65,536-registers-per-BLOCK limit --
    // 12% of headroom. At 512 it would be 115,712 and the launch would abort.
    // This is the constraint that decides the ladder's top, not occupancy.
    static_assert(256 * 226 <= 65536,
                  "the work-group ceiling is set by registers per block, not by occupancy; "
                  "re-run scripts/register_probe.sh before raising it");
    int wg = 32;
    for (int cand : {256, 128, 64, 32}) {
        if (cand > max_wg) continue;
        wg = cand;
        const int64_t groups_c = (q + cand - 1) / cand;
        // >= 4*CU work-groups keeps the machine fed. This is the guard against
        // the repeated BatchLAS defect of a kernel parallel over batch ONLY:
        // the grid is batch * ceil(q/WG), never batch alone.
        if (static_cast<int64_t>(bs) * groups_c >= static_cast<int64_t>(4) * cu) break;
    }

    const int groups = (q + wg - 1) / wg;
    const size_t tri_elems = static_cast<size_t>(N) * (N + 1) / 2;

    ctx->submit([&](sycl::handler& h) {
        sycl::local_accessor<D, 1> lc(sycl::range<1>(tri_elems), h);
        sycl::local_accessor<D, 1> rd(sycl::range<1>(N), h);
        sycl::local_accessor<int, 1> use_div(sycl::range<1>(1), h);

        // The Side::Left staging tile, and NOTHING for Side::Right -- its reads
        // already measure 5.13 sectors/request against a coalesced floor of 4,
        // so staging there would buy nothing and spend SLM.
        //
        // Row stride is NB_STAGE + 1, not NB_STAGE, and that padding is the
        // whole reason the tile is conflict-free on the way OUT: thread `lane`
        // reads sTile[lane*(NB+1) + j] for its own column, so at fixed j the 32
        // lanes are 17 (or 9) words apart. gcd(odd, 32) == 1, so the 32 lanes
        // land in 32 distinct banks. With a stride of NB itself they would land
        // in gcd(16,32)=16 banks and every read would be 2-way conflicted.
        // Never taller than the system itself: an N=8 bucket staging 16 rows
        // would allocate twice the SLM it can ever fill.
        constexpr bool kStageLeft = (SideV == Side::Left) && trsm_stage_left<D>();
        constexpr int NB_STAGE  = trsm_stage_rows<D>();
        constexpr int TILE_ROWS = (NB_STAGE < N) ? NB_STAGE : N;
        sycl::local_accessor<D, 1> tile(
            sycl::range<1>(kStageLeft
                               ? static_cast<size_t>(TILE_ROWS + 1) * wg
                               : size_t{0}),
            h);

        const D* a_ptr = reinterpret_cast<const D*>(A.data_ptr());
        D* b_ptr = reinterpret_cast<D*>(B.data_ptr());
        const int lda = static_cast<int>(A.ld());
        const int ldb = static_cast<int>(B.ld());
        const int stride_a = static_cast<int>(A.stride());
        const int stride_b = static_cast<int>(B.stride());

        const bool do_trans = can.do_trans;
        const bool do_conj = can.do_conj;
        const bool fwd = can.fwd;
        const bool unit = can.unit;

        D alpha_d;
        __builtin_memcpy(&alpha_d, &alpha, sizeof(D));

        h.parallel_for<TrsmCtaKernel<T, N, SideV>>(
            sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(bs) * groups * wg),
                              sycl::range<1>(wg)),
            [=](sycl::nd_item<1> it) {
                const int wg_id = static_cast<int>(it.get_group_linear_id());
                const int b = wg_id / groups;
                const int lane = static_cast<int>(it.get_local_linear_id());
                const int u = (wg_id % groups) * wg + lane;
                const bool live = (u < q);

                const D* Ab = a_ptr + static_cast<std::ptrdiff_t>(b) * stride_a;
                D* Bb = b_ptr + static_cast<std::ptrdiff_t>(b) * stride_b;

                D* sLc = lc.template get_multi_ptr<sycl::access::decorated::no>().get();
                D* sRd = rd.template get_multi_ptr<sycl::access::decorated::no>().get();
                int* sDiv = use_div.template get_multi_ptr<sycl::access::decorated::no>().get();
                D* sTile = tile.template get_multi_ptr<sycl::access::decorated::no>().get();
                const int u0 = (wg_id % groups) * wg;   // this group's first column

                if (lane == 0) sDiv[0] = 0;

                // ---- Cooperative staging of the canonical triangle ---------
                // rho(s) = fwd ? s : n-1-s maps canonical to stored index.
                // Lc(s,t) = opA(rho(s), rho(t)) for Left, opA(rho(t), rho(s))
                // for Right -- THE OPERAND ORDER IS SWAPPED between them, and
                // is invisible on a symmetric triangle.
                //
                // CONJUGATION. opA(r,c) = do_trans ? conj_if(A(c,r)) : A(r,c),
                // and do_conj implies do_trans, so the rule is simply: conjugate
                // iff transA == ConjTrans. It applies to EVERY staged element
                // INCLUDING THE DIAGONAL -- opA(r,r) = conj(A(r,r)) -- so the
                // reciprocal below is taken of the conjugated value. alpha and B
                // are never conjugated. For a real scalar do_conj is dead, which
                // is why this had no effect until complex arrived.
                for (size_t idx = lane; idx < tri_elems; idx += static_cast<size_t>(wg)) {
                    int s = 0;
                    while (tri_idx(s + 1, 0) <= static_cast<int>(idx)) ++s;
                    const int t = static_cast<int>(idx) - tri_idx(s, 0);

                    D v;
                    if (s >= n || t >= n) {
                        // Zero padding with a unit diagonal, so the unrolled
                        // tail computes zeros instead of branching.
                        v = (s == t) ? sycl_device::dev_one<D>() : D{};
                    } else {
                        const int rs = fwd ? s : (n - 1 - s);
                        const int rt = fwd ? t : (n - 1 - t);
                        const int r = (SideV == Side::Left) ? rs : rt;
                        const int c = (SideV == Side::Left) ? rt : rs;
                        v = do_trans
                                ? Ab[c + static_cast<std::ptrdiff_t>(r) * lda]   // A(c,r)
                                : Ab[r + static_cast<std::ptrdiff_t>(c) * lda];  // A(r,c)
                        if (do_conj) v = sycl_device::dev_conj(v);
                    }
                    sLc[idx] = v;
                }

                // THE BARRIER THAT WAS MISSING, and it is a WRONG ANSWER, not a
                // tuning nicety. Two independent read-after-writes cross this
                // line, and until WP4 Phase 2 triage neither was ordered:
                //
                //  (1) sLc. The staging loop above is strided by `lane`, so
                //      element `idx` is written by lane `idx % wg`. The
                //      reciprocal loop below has lane `s` READ sLc[tri_idx(s,s)]
                //      == sLc[s*(s+1)/2 + s], which is a different lane's write
                //      for every s where (s*(s+1)/2 + s) % wg != s -- i.e. for
                //      nearly all of them. Reading an unwritten local word gives
                //      a garbage diagonal, hence a garbage reciprocal, hence a
                //      wrong solve.
                //  (2) sDiv[0]. Lane 0 zeroes it at :339; every lane may
                //      atomically store 1 into it in the loop below. Unordered,
                //      lane 0's zero can land after the store and silently
                //      discard the revert-to-division flag.
                //
                // WHY IT SURVIVED WP3'S TEST SUITE AND ITS BENCHMARKS. wg is
                // chosen at :266-274 as the FIRST candidate in {256,128,64,32}
                // that keeps bs*ceil(q/wg) >= 4*CU, so it is 32 -- a single
                // sub-group, executing the two loops in lock step, where the
                // race cannot express itself -- for every shape below roughly
                // q*bs = 65k. Every trsm test and every WP3 A/B cell sits in
                // that regime. The blocked POTRF panel solve does not: at
                // n=1024, batch=256 the first panel has q = 896, wg = 256, eight
                // sub-groups, and the race fires.
                //
                // MEASURED, vendor-free potrf (build-novendor, no env), float
                // and double, n=1024 batch=256, a diagonally dominant SPD input
                // that cuSOLVER factors to 1e-8/1e-16: before this barrier
                // 61-75 of 256 items came back info != 0, non-deterministically,
                // and the reported failing column was always == 1 (mod nb) --
                // the first column of a panel, i.e. a diagonal block that the
                // previous panel's bad L21 had already destroyed. After it,
                // 0/256 over every rep, both types, at batch up to 1024.
                sycl::group_barrier(it.get_group());

                // ---- Diagonal reciprocals, guarded -------------------------
                // The recurrence multiplies by rd[s] = 1/Lc(s,s) rather than
                // dividing, which is the only arithmetic deviation from the
                // reference loop nest. It is unsafe in exactly one place: if the
                // reciprocal is not finite the multiply produces inf where a
                // division would have produced a finite number. So it is
                // CHECKED, and any thread seeing a non-finite one flips a
                // work-group-uniform flag reverting the whole group to division.
                //
                // For complex the reciprocal is Smith's algorithm, not
                // conj(d)/|d|^2: the textbook form squares the components and
                // so overflows to 0 for inputs whose true reciprocal is
                // perfectly representable. See src/sycl/device_scalar.hh.
                // BOTH components are tested, since either can go non-finite
                // independently.
                for (int s = lane; s < N; s += wg) {
                    D r = sycl_device::dev_one<D>();
                    if (s < n && !unit) {
                        const D d = sLc[tri_idx(s, s)];
                        r = sycl_device::dev_recip(d);
                        if (!sycl_device::dev_isfinite(r)) {
                            sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                             sycl::memory_scope::work_group,
                                             sycl::access::address_space::local_space>(sDiv[0])
                                .store(1);
                            r = sycl_device::dev_one<D>();
                        }
                    }
                    sRd[s] = r;
                }

                sycl::group_barrier(it.get_group());

                const bool divide = (sDiv[0] != 0);

                // ---- The recurrence, fully unrolled ------------------------
                // Canonical RHS accessor, spec section 2.1:
                //   Left : b0 = fwd?0:(n-1),      ds = +-1,   du = ldb
                //   Right: b0 = fwd?0:(n-1)*ldb,  ds = +-ldb, du = 1
                const std::ptrdiff_t unit_s =
                    (SideV == Side::Left) ? 1 : static_cast<std::ptrdiff_t>(ldb);
                const std::ptrdiff_t du =
                    (SideV == Side::Left) ? static_cast<std::ptrdiff_t>(ldb) : 1;
                const std::ptrdiff_t b0 = fwd ? 0 : static_cast<std::ptrdiff_t>(n - 1) * unit_s;
                const std::ptrdiff_t ds = fwd ? unit_s : -unit_s;

                // The address of canonical step s for column `col`. For
                // Side::Left the row is rho(s) and the column is col, so a
                // ROUND of consecutive canonical steps is a contiguous run of
                // rows -- ascending when fwd, descending when not, and the
                // coalescer does not care which.
                const auto left_addr = [&](int s_can, int col) -> std::ptrdiff_t {
                    const int row = fwd ? s_can : (n - 1 - s_can);
                    return static_cast<std::ptrdiff_t>(row) +
                           static_cast<std::ptrdiff_t>(col) * ldb;
                };

                // Rounds. Side::Right keeps ONE round covering every step and
                // reads global memory directly, exactly as before -- the
                // staging code below is all inside `if constexpr`, so its
                // barriers and SLM traffic do not exist in that instantiation.
                constexpr int STEP   = kStageLeft ? TILE_ROWS : N;
                constexpr int ROUNDS = (N + STEP - 1) / STEP;

                D x[N];
#pragma unroll
                for (int k = 0; k < ROUNDS; ++k) {
                    if constexpr (kStageLeft) {
                        // Barrier BEFORE overwriting the tile: the previous
                        // round's per-thread reads must have retired. Both
                        // barriers are outside every `live` test, because a
                        // work-group barrier that only some lanes reach is
                        // undefined -- and lanes with u >= q are exactly the
                        // ones a `live` guard would drop.
                        sycl::group_barrier(it.get_group());
                        // COALESCED FILL. i runs over NB_STAGE*wg elements with
                        // r = i % TILE_ROWS varying fastest, so consecutive
                        // lanes read consecutive ROWS of one column: 16 floats
                        // = 64 B = two fully-used sectors per lane-group.
                        for (int i = lane; i < TILE_ROWS * wg; i += wg) {
                            const int r = i % TILE_ROWS;
                            const int c = i / TILE_ROWS;
                            const int s_can = k * TILE_ROWS + r;
                            const int col = u0 + c;
                            D v{};
                            if (s_can < n && col < q) v = Bb[left_addr(s_can, col)];
                            sTile[c * (TILE_ROWS + 1) + r] = v;
                        }
                        sycl::group_barrier(it.get_group());
                    }

#pragma unroll
                    for (int j = 0; j < STEP; ++j) {
                        const int s = k * STEP + j;
                        if (s >= N) continue;    // folds away when STEP divides N
                        D acc = D{};
#pragma unroll
                        for (int t = 0; t < N; ++t) {
                            if (t < s) sycl_device::fma_acc(acc, sLc[tri_idx(s, t)], x[t]);
                        }
                        D rhs = D{};
                        if (live && s < n) {
                            if constexpr (kStageLeft) {
                                rhs = sycl_device::dev_mul(
                                    alpha_d, sTile[lane * (TILE_ROWS + 1) + j]);
                            } else {
                                rhs = sycl_device::dev_mul(
                                    alpha_d,
                                    Bb[b0 + static_cast<std::ptrdiff_t>(s) * ds + u * du]);
                            }
                        }
                        D v = sycl_device::dev_sub(rhs, acc);
                        if (!unit) {
                            v = divide ? sycl_device::dev_div(v, sLc[tri_idx(s, s)])
                                       : sycl_device::dev_mul(v, sRd[s]);
                        }
                        x[s] = v;
                    }
                }

                // ---- Write-back, the mirror image --------------------------
                if constexpr (kStageLeft) {
#pragma unroll
                    for (int k = 0; k < ROUNDS; ++k) {
                        sycl::group_barrier(it.get_group());
                        // Each thread drops its own round of results into its
                        // own tile column; the guard is on the STEP, not on
                        // `live`, because a non-live lane's column is one the
                        // coalesced store below already refuses to write.
#pragma unroll
                        for (int j = 0; j < TILE_ROWS; ++j) {
                            const int s = k * TILE_ROWS + j;
                            if (s < N) sTile[lane * (TILE_ROWS + 1) + j] = x[s];
                        }
                        sycl::group_barrier(it.get_group());
                        for (int i = lane; i < TILE_ROWS * wg; i += wg) {
                            const int r = i % TILE_ROWS;
                            const int c = i / TILE_ROWS;
                            const int s_can = k * TILE_ROWS + r;
                            const int col = u0 + c;
                            if (s_can < n && col < q) {
                                Bb[left_addr(s_can, col)] = sTile[c * (TILE_ROWS + 1) + r];
                            }
                        }
                    }
                } else {
#pragma unroll
                    for (int s = 0; s < N; ++s) {
                        if (live && s < n) {
                            Bb[b0 + static_cast<std::ptrdiff_t>(s) * ds + u * du] = x[s];
                        }
                    }
                }
            });
    });

    return ctx.get_event();
}

// Runtime bucket dispatch. Direct-call entry used by tests at this step.
template <typename T, Side SideV>
Event trsm_native_v1_buckets(Queue& ctx,
                             const MatrixView<T, MatrixFormat::Dense>& A,
                             const MatrixView<T, MatrixFormat::Dense>& B,
                             T alpha, Uplo uplo, Transpose transA, Diag diag) {
    using D_ = typename sycl_device::DevMap<T>::type;
    switch (smallest_bucket_ge(static_cast<int>(A.rows()))) {
        case 8:  return trsm_native_v1<T, 8, SideV>(ctx, A, B, alpha, uplo, transA, diag);
        case 16: return trsm_native_v1<T, 16, SideV>(ctx, A, B, alpha, uplo, transA, diag);
        case 32: return trsm_native_v1<T, 32, SideV>(ctx, A, B, alpha, uplo, transA, diag);
        default: break;
    }
    {
            // ENFORCED, not assumed. The router already caps the order via
            // supports(CTA), so reaching here means a direct caller (V2, or a
            // test) exceeded the contract -- and the alternative to throwing is
            // returning a wrong answer for the rows that do not fit.
            throw std::runtime_error(
                "BatchLAS: trsm_native_v1 called with triangular order " +
                std::to_string(A.rows()) +
                ", which exceeds this scalar's CTA register capacity of " +
                std::to_string(trsm_max_bucket<D_>()) + ". Orders above the "
                "capacity are the blocked driver's (V2's) job; the CTA kernel cannot "
                "serve them and must not silently solve a leading submatrix.");
    }
}

// ---------------------------------------------------------------------------
// V2 -- the host-blocked driver, for orders above V1's register capacity.
//
// Canonical block i covers s in [i*nb, min(n,(i+1)*nb)). Because rho is a
// BIJECTION on [0,n), both the block R_i and the already-solved set S_i are
// contiguous runs in STORED indices:
//
//        r0 (start of R_i)      s0 (start of S_i)     m = hi-lo   k = lo
//   fwd  lo                     0                     block rows  solved rows
//  !fwd  n-hi                   n-lo                  block rows  solved rows
//
// so fwd enters only through two scalars and all four (side, fwd) cases share
// one code path.
//
// THE ALPHA CONTRACT, which is the one thing here that is silently wrong if
// mis-stated. alpha is applied EXACTLY ONCE per block, by one of two routes:
//   * block 0        -- no trailing update exists, so V1 applies it (alpha_eff = alpha)
//   * blocks i > 0   -- the trailing GEMM applies it as its BETA (beta = alpha),
//                       computing B_i := alpha*B_i - op(A_off)*X_prev, and V1
//                       then runs with alpha_eff = 1
// Never both, never neither. Writing the natural beta = 1 on that GEMM computes
// B_i - sum(...) where alpha*B_i - sum(...) is required: a wrong answer for
// every alpha != 1 at every block i > 0, which compiles and passes any alpha = 1
// test. The existing suite uses alpha = 1 throughout, so this would have been
// invisible without a test that varies it.
//
// SUB-VIEWS ARE BUILT BY THE EXPLICIT 6-ARG CONSTRUCTOR, never by
// operator()(Slice,Slice). Two reasons, both verified in source. First, that
// operator passes the parent's pointer array into the child despite a comment
// directly above it saying it must not (matrix.hh:1140), and any later
// data_ptrs() call on the slice would rewrite the parent's per-batch bases.
// Second, and the trap that actually bites here: the constructor DEFAULTS
// stride to ld*cols when 0 is passed (src/matrix.cc:1839-1842), so a sub-view
// of k columns built without an explicit stride silently gets stride = ld*k and
// every batch item after the first reads the wrong matrix. The parent's ld AND
// stride are passed explicitly at every call below.
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// V2's OUTER block width, which is NOT the CTA capacity.
//
// It used to be. `nb = trsm_cta_max_n<T>()` tied the trailing-update block to
// the register capacity of the diagonal solver, and that single line is what
// made large orders lose: at n=512 it gives 16 blocks, so
//
//   * every trailing GEMM has one dimension pinned at 32. GEMM arithmetic
//     intensity with one dimension pinned at w tends to 2*w/sizeof(T) flop per
//     byte -- 16 for float at w=32, against an RTX 4090 machine balance of
//     ~40 TFLOP/s / ~950 GB/s = 42. So 93.75% of the solve's flops (the GEMM
//     share is 1 - nb/n) ran in a regime that is bandwidth-bound BY
//     CONSTRUCTION, on a problem that at n=512 is intrinsically compute-bound
//     (51 flop/byte). At w=128 the intensity is 64, above the balance.
//
//   * the driver is left-looking, so the already-solved prefix of B is re-read
//     by every later trailing GEMM. The re-read factor is (p-1)/2 for p = n/nb
//     blocks: 7.5x at n=512, nb=32. Total DRAM amplification over an ideal
//     trsm works out at 4.75x, and 2.20x at nb=128.
//
// Measured consequence before this change (float, Side::Left, vendor/native,
// >1 means native wins): 1.13x at order 128, 0.74-0.93x at 256, 0.58-0.69x at
// 512 -- monotonically worse, which is the signature of a per-block cost that
// grows with the block count rather than a fixed inefficiency.
//
// So the outer width is decoupled: the trailing update runs at OUTER_NB, and
// each OUTER_NB-wide panel is then solved by the old nb = cta_max_n loop
// against its own, much shorter, prefix. Two levels, same arithmetic.
//
// BATCHLAS_TRSM_OUTER_NB overrides it for tuning sweeps. It is a TUNING knob,
// not a routing one: it never changes which route is chosen, only how V2
// blocks once chosen, so it does not belong in the Route vocabulary.
inline int trsm_outer_block_default() { return 128; }

// AND IT DEPENDS ON THE SIDE, which is measured, not aesthetic. Sweeping
// OUTER_NB over {32,64,128,256} on float (worst cell per order, vendor/native):
//
//            Side::Left                     Side::Right
//   order   nb32  nb64 nb128 nb256    nb32  nb64 nb128 nb256
//     128   1.18  1.10  1.20  1.17    1.00  0.96  1.00  0.98
//     256   0.75  0.78  0.87  0.75    1.01  0.94  0.83  1.01
//     512   0.58  0.74  0.76  0.75    1.07  0.92  0.82  0.91
//
// Widening helps Left at every large order and HURTS Right at every large
// order, turning two Right cells that won into losses. The two sides put the
// block width on different GEMM dimensions -- Left's trailing update is
// C(nb x q), Right's is C(q x nb) -- and they therefore land in different
// clauses of select_kernel_variant. Widening also shortens the inner updates'
// k, and float's transposed fast paths in gemm_kernels.cc require k >= 128, so
// for Right the inner GEMMs drop below the gate and fall to Tiled16.
//
// So Right keeps the old single-level schedule and Left gets the wide one. One
// number for both sides would have to be 32, which throws away everything the
// two-level driver buys at order 256 and 512.
inline int trsm_outer_block(int cta_nb, Side side) {
    static const int env = [] {
        const char* raw = std::getenv("BATCHLAS_TRSM_OUTER_NB");
        if (!raw || !*raw) return 0;
        const int v = std::atoi(raw);
        return v > 0 ? v : 0;
    }();
    const int want = env ? env : (side == Side::Left ? trsm_outer_block_default() : cta_nb);
    // Must be a whole number of CTA blocks, and at least one.
    const int rounded = (want / cta_nb) * cta_nb;
    return rounded >= cta_nb ? rounded : cta_nb;
}

template <typename T>
Event trsm_native_blocked(Queue& ctx,
                          const MatrixView<T, MatrixFormat::Dense>& A,
                          const MatrixView<T, MatrixFormat::Dense>& B,
                          T alpha,
                          Side side,
                          Uplo uplo,
                          Transpose transA,
                          Diag diag,
                          TrsmTrailingGemm<T> trailing_gemm) {
    // Default to the native kernel so this TU stands alone: a direct caller
    // (the tests) gets gemm_custom with no dispatch dependency. The facade
    // passes the ROUTED gemm instead -- see the note on TrsmTrailingGemm in
    // trsm_native.hh for the measurement that motivates it.
    if (!trailing_gemm) {
        trailing_gemm = [](Queue& c,
                           const MatrixView<T, MatrixFormat::Dense>& ga,
                           const MatrixView<T, MatrixFormat::Dense>& gb,
                           const MatrixView<T, MatrixFormat::Dense>& gc,
                           T galpha, T gbeta, Transpose gta, Transpose gtb,
                           ComputePrecision gp) {
            return sycl_gemm::gemm_custom<T>(c, ga, gb, gc, galpha, gbeta,
                                             gta, gtb, gp);
        };
    }
    const Canonical can = canonicalise(side, uplo, transA, diag);
    const int n = static_cast<int>(A.rows());
    const int q = static_cast<int>(side == Side::Left ? B.cols() : B.rows());
    const int nb = trsm_cta_max_n<T>();          // the CTA capacity: the INNER block
    const int outer_nb = trsm_outer_block(nb, side);  // the trailing-update block; see above

    const int lda = A.ld(), ldb = B.ld();
    const int sa = A.stride(), sb = B.stride();
    const int bs = A.batch_size();

    auto sub = [](const MatrixView<T, MatrixFormat::Dense>& V,
                  int r0, int nr, int c0, int nc, int ld, int stride, int batch) {
        // Column-major: offset = c0*ld + r0, the repo's own dense-slice form.
        return MatrixView<T, MatrixFormat::Dense>(
            V.data_ptr() + static_cast<std::ptrdiff_t>(c0) * ld + r0,
            nr, nc, ld, stride, batch);
    };

    // Canonical range [a, b) -> the stored row offset of that range. rho(s) is
    // fwd ? s : n-1-s, so a canonical range maps to [a,b) when fwd and to
    // [n-b, n-a) when not. Every offset below goes through this, which is what
    // lets the two levels share one index convention -- the old code inlined
    // the two special cases (prefix [0,lo) and block [lo,hi)) and there are now
    // four.
    auto stored_off = [&](int a, int b) { return can.fwd ? a : (n - b); };

    // Apply the already-solved canonical range [p_lo, p_hi) to the target range
    // [t_lo, t_hi):  C := -op(A_off) * X + beta*C.
    auto apply_update = [&](int t_lo, int t_hi, int p_lo, int p_hi, T beta) {
        const int m = t_hi - t_lo;
        const int k = p_hi - p_lo;
        const int r0 = stored_off(t_lo, t_hi);
        const int s0 = stored_off(p_lo, p_hi);

        const auto C = (side == Side::Left) ? sub(B, r0, m, 0, q, ldb, sb, bs)
                                            : sub(B, 0, q, r0, m, ldb, sb, bs);
        const auto X = (side == Side::Left) ? sub(B, s0, k, 0, q, ldb, sb, bs)
                                            : sub(B, 0, q, s0, k, ldb, sb, bs);
        // The A block is chosen so that op() lands on the required sub-block,
        // which is why transA is passed through unchanged.
        const auto Aoff =
            (side == Side::Left)
                ? (can.do_trans ? sub(A, s0, k, r0, m, lda, sa, bs)
                                : sub(A, r0, m, s0, k, lda, sa, bs))
                : (can.do_trans ? sub(A, r0, m, s0, k, lda, sa, bs)
                                : sub(A, s0, k, r0, m, lda, sa, bs));

        if (side == Side::Left) {
            // C(m x q) := -op(Aoff)(m x k) * X(k x q) + beta*C
            trailing_gemm(ctx, Aoff, X, C, T(-1), beta,
                          transA, Transpose::NoTrans,
                          ComputePrecision::Default);
        } else {
            // C(q x m) := -X(q x k) * op(Aoff)(k x m) + beta*C.
            // X GOES IN THE A POSITION. The obvious single form with the A
            // block first produces a C of at most nb rows against the required
            // q and does not conform for any transpose.
            trailing_gemm(ctx, X, Aoff, C, T(-1), beta,
                          Transpose::NoTrans, transA,
                          ComputePrecision::Default);
        }
    };

    auto solve_diag = [&](int lo, int hi, T alpha_eff) {
        const int m = hi - lo;
        const int r0 = stored_off(lo, hi);
        const auto Adiag = sub(A, r0, m, r0, m, lda, sa, bs);
        const auto Bblk = (side == Side::Left) ? sub(B, r0, m, 0, q, ldb, sb, bs)
                                               : sub(B, 0, q, r0, m, ldb, sb, bs);
        trsm_native_v1_dispatch<T>(ctx, Adiag, Bblk, alpha_eff, side, uplo, transA, diag);
    };

    // TWO LEVELS. The outer one applies the whole solved prefix to a panel in a
    // SINGLE fat GEMM; the inner one is the old right-looking loop, but its
    // prefix is now at most OUTER_NB - nb wide instead of the whole matrix.
    //
    // ALPHA IS APPLIED EXACTLY ONCE PER ELEMENT OF B, on that element's first
    // touch, and which operation that is depends on where the block sits:
    //   panel 0, block 0      -- nothing has touched it, so the CTA solve
    //                            carries alpha.
    //   panel 0, block > 0    -- the inner GEMM is the first touch: beta=alpha.
    //   panel > 0, any block  -- the OUTER GEMM already touched the whole
    //                            panel with beta=alpha, so the inner GEMM uses
    //                            beta=1 and the solve uses alpha=1.
    // Getting this wrong scales part of B twice, which no shape-only test would
    // catch; tests/trsm_tests.cc drives alpha != 1 through every canonical cell
    // for exactly this reason.
    for (int LO = 0; LO < n; LO += outer_nb) {
        const int HI = std::min(n, LO + outer_nb);

        if (LO > 0) apply_update(LO, HI, 0, LO, alpha);

        for (int lo = LO; lo < HI; lo += nb) {
            const int hi = std::min(HI, lo + nb);
            const bool first_touch_is_here = (LO == 0);
            if (lo > LO) {
                apply_update(lo, hi, LO, lo, first_touch_is_here ? alpha : T(1));
            }
            const T alpha_eff = (LO == 0 && lo == 0) ? alpha : T(1);
            solve_diag(lo, hi, alpha_eff);

            // Block i+1's GEMM reads what block i's solve just wrote. An
            // in-order queue gives that for free; an out-of-order one does not,
            // and a caller may construct either (sycl-device-queue.hh:239
            // defaults in_order=true but it is a parameter). This is a
            // correctness requirement, not a tuning choice.
            if (!ctx.in_order()) ctx.wait();
        }
    }

    return ctx.get_event();
}

template <typename T>
Event trsm_native_v1_dispatch(Queue& ctx,
                              const MatrixView<T, MatrixFormat::Dense>& A,
                              const MatrixView<T, MatrixFormat::Dense>& B,
                              T alpha,
                              Side side,
                              Uplo uplo,
                              Transpose transA,
                              Diag diag) {
    return (side == Side::Left)
               ? trsm_native_v1_buckets<T, Side::Left>(ctx, A, B, alpha, uplo, transA, diag)
               : trsm_native_v1_buckets<T, Side::Right>(ctx, A, B, alpha, uplo, transA, diag);
}

template Event trsm_native_v1_dispatch<float>(
    Queue&, const MatrixView<float, MatrixFormat::Dense>&,
    const MatrixView<float, MatrixFormat::Dense>&, float, Side, Uplo, Transpose, Diag);
// double is instantiated deliberately, as the FALSIFICATION PROBE for the
// spec's n_cta(double) = 32. That number comes from a "256 B/thread register
// cliff" which gemm_kernels.cc:725-735 records as measured false, so N=64 double
// -- 64 doubles of accumulator per thread -- is exactly the configuration the
// hypothesis says must spill. The register probe decides it, not the spec.
template Event trsm_native_v1_dispatch<double>(
    Queue&, const MatrixView<double, MatrixFormat::Dense>&,
    const MatrixView<double, MatrixFormat::Dense>&, double, Side, Uplo, Transpose, Diag);
template Event trsm_native_v1_dispatch<std::complex<float>>(
    Queue&, const MatrixView<std::complex<float>, MatrixFormat::Dense>&,
    const MatrixView<std::complex<float>, MatrixFormat::Dense>&, std::complex<float>,
    Side, Uplo, Transpose, Diag);
template Event trsm_native_v1_dispatch<std::complex<double>>(
    Queue&, const MatrixView<std::complex<double>, MatrixFormat::Dense>&,
    const MatrixView<std::complex<double>, MatrixFormat::Dense>&, std::complex<double>,
    Side, Uplo, Transpose, Diag);

// THE REGISTER GATE HAS RUN. scripts/register_probe.sh, sm_89, this TU:
//
//   type    N    registers   stack frame   spill
//   float    8      42            0          0
//   float   16      76            0          0
//   float   32     114            0          0
//   float   64     119          256 B        0      <-- x[64] is NOT in registers
//   double   8      59            0          0
//   double  16     100            0          0
//   double  32     153            0          0
//   double  64     145          512 B        0      <-- x[64] is NOT in registers
//
// READ THE STACK-FRAME COLUMN, NOT THE SPILL COLUMN. Nothing spills anywhere,
// including double N=64 -- so the spec's "256 B/thread register cliff", which
// predicts exactly that configuration must spill, is FALSIFIED, as
// WP3_TRSM_SPEC_CORRECTIONS.md expected.
//
// But the design still fails at N=64, and it fails in the column the
// corrections document told the implementer to ignore. 256 B is 64 floats; 512 B
// is 64 doubles. Those are x[] itself, placed in local memory rather than
// promoted to registers. ptxas reports that as a STACK FRAME, not as a spill,
// because the array was never in registers to be spilled out of -- and
// register residency is the entire thesis of V1.
//
// So the gate this file is measured against is:
//     stack frame == 0  AND  0 spill bytes  AND  registers x WG <= 65536
// The corrections document's "gate on spill bytes, not stack frame" was right
// about the GEMM kernels it was derived from (220 of 376 entry functions there
// carry a benign non-zero frame) and WRONG here, because in THIS kernel the only
// thing that can be on the stack is the accumulator array. Both documents have
// been amended.
//
// MEASURED CAPACITY: n_cta(float) = 32, n_cta(double) = 32. The spec predicted
// float 64. Its own step-2 instruction -- "if x[64] spills, reduce n_cta(float)
// to 32 before writing anything else" -- reached the right answer by the wrong
// mechanism, which is why the gate had to be run rather than reasoned about.
//
// STEP 4 re-ran the gate with Side::Left added. All 24 trsm kernels (2 types x
// 3 buckets x 2 sides, each in its plain and _with_offset flavour) report
// `0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads`, and
// Side::Left matches Side::Right register-for-register (float N=32: 114 both).
// The N=64 buckets are gone, so the one configuration that failed the gate is
// no longer built.
//
// So the capacities are now the measured ones. Real types only: complex still
// returns 0 because it needs a POD device scalar and a guarded complex
// RECIPROCAL, and GEMM's wide-scalar helpers provide multiply but no division.
//
// COMPLEX MEASURED TOO, and the prediction going in was wrong. complex<double>
// at N=32 holds 32 complex doubles -- 512 bytes of accumulator, the same size
// that put double N=64 in local memory -- so it was expected to fail the gate.
// It does not: 0 bytes stack frame, 0 spill, 226 registers. All 24 kernels
// (4 types x 3 buckets x 2 sides) pass, so n_cta = 32 for every type.
//
//   type              N=8   N=16   N=32     regs*256 at N=32
//   float              44     76    114           29,184
//   double             59    101    153           39,168
//   complex<float>     50     86    148           37,888
//   complex<double>    74    138    226           57,856   <- worst, 12% headroom
//
// The binding constraint is registers per BLOCK, not occupancy, and it is what
// caps the work-group ladder at 256; see the static_assert in the launcher.
template <> int trsm_cta_max_n<float>()                { return 32; }
template <> int trsm_cta_max_n<double>()               { return 32; }
template <> int trsm_cta_max_n<std::complex<float>>()  { return 32; }
template <> int trsm_cta_max_n<std::complex<double>>() { return 32; }

// V2 does not exist yet, for any type. Until it does, an order above
// trsm_cta_max_n has NO native route, and RouteTable<Op::trsm,T>::supports()
// must say so -- otherwise a vendor-free caller at n > 32 is handed a Blocked
// route the facade cannot service and the call dies further downstream with a
// message that blames the wrong thing.
template Event trsm_native_blocked<float>(
    Queue&, const MatrixView<float, MatrixFormat::Dense>&,
    const MatrixView<float, MatrixFormat::Dense>&, float, Side, Uplo, Transpose, Diag, TrsmTrailingGemm<float>);
template Event trsm_native_blocked<double>(
    Queue&, const MatrixView<double, MatrixFormat::Dense>&,
    const MatrixView<double, MatrixFormat::Dense>&, double, Side, Uplo, Transpose, Diag, TrsmTrailingGemm<double>);
template Event trsm_native_blocked<std::complex<float>>(
    Queue&, const MatrixView<std::complex<float>, MatrixFormat::Dense>&,
    const MatrixView<std::complex<float>, MatrixFormat::Dense>&, std::complex<float>,
    Side, Uplo, Transpose, Diag, TrsmTrailingGemm<std::complex<float>>);
template Event trsm_native_blocked<std::complex<double>>(
    Queue&, const MatrixView<std::complex<double>, MatrixFormat::Dense>&,
    const MatrixView<std::complex<double>, MatrixFormat::Dense>&, std::complex<double>,
    Side, Uplo, Transpose, Diag, TrsmTrailingGemm<std::complex<double>>);

template <> bool trsm_blocked_available<float>()                { return true; }
template <> bool trsm_blocked_available<double>()               { return true; }
template <> bool trsm_blocked_available<std::complex<float>>()  { return true; }
template <> bool trsm_blocked_available<std::complex<double>>() { return true; }

}  // namespace batchlas::sycl_trsm
