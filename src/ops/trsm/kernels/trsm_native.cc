// SPDX-License-Identifier: MIT
// Native batched TRSM: the register-resident CTA solver (V1) and the
// host-blocked driver (V2). See docs/perf/trsm.md.
//
// V1 puts one thread on each independent solve; x[N] must stay in registers, so N
// is a compile-time bucket >= n and the loops are fully unrolled (indexing x by a
// runtime variable moves it to local memory). Rows n..N-1 pad with Lc(s,s)=1.

#include <src/ops/trsm/trsm_native.hh>
#include <src/ops/trsm/trsm_canonical.hh>
#include <src/ops/trsm/trsm_v2.hh>

#include <src/linalg-impl.hh>
#include <src/util/resident_capacity.hh>
#include <src/sycl/device_scalar.hh>
#include <batchlas/device/detail/kernel_attrs.hh>
#include <src/util/batch_mask.hh>
#include <batchlas/device/arith.hh>

#include <sycl/sycl.hpp>

#include <algorithm>
#include <complex>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <batchlas/device.hh>
#include <batchlas/settings.hh>

namespace batchlas::sycl_trsm {

namespace v2 = batchlas::device;

namespace {

// Smallest compile-time bucket >= n, or 0 for none: a narrower bucket would
// silently solve the leading NxN system. evidence: docs/perf/trsm.md#the-bucket-ladder-that-truncated
inline int smallest_bucket_ge(int n) {
    if (n <= 8) return 8;
    if (n <= 16) return 16;
    if (n <= 32) return 32;
    return 0;
}

// N=64 is rejected by measurement, not omitted: x[] lands in local memory for
// both sides. evidence: docs/perf/trsm.md#rejected-the-n64-cta-bucket
template <typename D>
constexpr int trsm_max_bucket() {
    return 32;
}

// Side::Left staging tile height, in ELEMENTS: two 32 B sectors' worth, so a
// lane-group's column read fills them. evidence: docs/perf/trsm.md#the-sideleft-staging-tile
template <typename D>
constexpr int trsm_stage_rows() {
    return sizeof(D) <= 4 ? 16 : 8;
}

// Only the real types stage: staging would move complex's x[] into local memory.
template <typename D>
constexpr bool trsm_stage_left() {
    return sizeof(D) <= 8 && !sycl_device::dev_is_complex_v<D>;
}

template <typename T, int N, Side SideV>
class TrsmCtaKernel;

}  // namespace

template <typename T, int N, Side SideV>
Event trsm_native_v1(Queue& ctx,
                     const MatrixView<T, MatrixFormat::Dense>& A,
                     const MatrixView<T, MatrixFormat::Dense>& B,
                     T alpha,
                     Uplo uplo,
                     Transpose transA,
                     Diag diag) {
    using D = typename sycl_device::DevMap<T>::type;
    static_assert(sizeof(D) == sizeof(T), "device scalar must be layout-compatible");

    const Canonical can = canonicalise(SideV, uplo, transA, diag);

    const int n = static_cast<int>(A.rows());
    const int q = static_cast<int>(SideV == Side::Left ? B.cols() : B.rows());
    const int bs = static_cast<int>(A.batch_size());

    const auto dev = ctx.device();
    const int max_wg = static_cast<int>(dev.get_property(DeviceProperty::MAX_WORK_GROUP_SIZE));
    const int cu = static_cast<int>(dev.get_property(DeviceProperty::MAX_COMPUTE_UNITS));

    // Both operands are named so the assert is driven by the ladder it guards:
    // adding a rung above kMaxWg now fails to compile instead of aborting at launch.
    constexpr int kMaxWg = kTrsmV1MaxWg;
    constexpr int kWorstRegsPerThread = 226;   // complex<double>, N=32
    // 256 lanes is 8 warps, 2 per sub-partition: 2 x 32 x ceil8(226) = 14,848 of 16,384.
    // evidence: docs/perf/lu.md#the-register-cap-that-binds-is-per-sub-partition
    static_assert(resident::sm89_fits(kWorstRegsPerThread, kMaxWg),
                  "the work-group ceiling is set by registers per sub-partition, not by "
                  "occupancy; re-run scripts/register_probe.sh before raising it");
    const int wg = trsm_v1_ladder_wg(max_wg, cu, q, bs);

    const int groups = (q + wg - 1) / wg;
    const size_t tri_elems = static_cast<size_t>(N) * (N + 1) / 2;

    ctx->submit([&](sycl::handler& h) {
        sycl::local_accessor<D, 1> lc(sycl::range<1>(tri_elems), h);
        sycl::local_accessor<D, 1> rd(sycl::range<1>(N), h);
        sycl::local_accessor<int, 1> use_div(sycl::range<1>(1), h);

        // Nothing is staged for Side::Right; its reads are already coalesced.
        // Row stride is TILE_ROWS + 1, not TILE_ROWS: the odd stride puts the 32
        // lanes' column reads in 32 distinct banks instead of 2-way conflicts.
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
        const std::uint8_t* mask = batchlas::detail::mask_ptr(batchlas::detail::call_mask("trsm", A, B));

        const bool do_trans = can.do_trans;
        const bool do_conj = can.do_conj;
        const bool fwd = can.fwd;
        const bool unit = can.unit;

        D alpha_d;
        __builtin_memcpy(&alpha_d, &alpha, sizeof(D));

        batchlas::impl::parallel_for<TrsmCtaKernel<T, N, SideV>>(h,
            sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(bs) * groups * wg),
                              sycl::range<1>(wg)),
            [=](sycl::nd_item<1> it) {
                using Ar = v2::arith_t<v2::arith::Fma, D>;
                constexpr bool kLeft = (SideV == Side::Left);
                const int wg_id = static_cast<int>(it.get_group_linear_id());
                const int b = wg_id / groups;
                // A work-group serves one item (I1): uniform, before the first barrier.
                if (mask != nullptr && mask[b] == 0) return;
                const v2::RtGroup<> ex{it};
                const int lane = ex.lid();
                const int u0 = (wg_id % groups) * wg;   // this group's first column
                const int u = u0 + lane;
                const bool live = (u < q);

                D* Bb = b_ptr + static_cast<std::ptrdiff_t>(b) * stride_b;
                // ConjTrans conjugates INCLUDING THE DIAGONAL, so rd below inverts conj(d).
                const dla::CanonTri<const D, kLeft> L{a_ptr + static_cast<std::ptrdiff_t>(b) * stride_a, lda, n,
                                                      fwd, do_trans, do_conj};

                D* sLc = lc.template get_multi_ptr<sycl::access::decorated::no>().get();
                D* sRd = rd.template get_multi_ptr<sycl::access::decorated::no>().get();
                int* sDiv = use_div.template get_multi_ptr<sycl::access::decorated::no>().get();
                D* sTile = tile.template get_multi_ptr<sycl::access::decorated::no>().get();

                if (lane == 0) sDiv[0] = 0;
                dla::tp_pack_lower<N>(ex, L, n, sLc);
                // REQUIRED; without it the solve returns WRONG ANSWERS: sLc is
                // written strided by lane and read by a different lane, and lane 0
                // zeroes sDiv[0] while any lane may store 1. The race cannot show
                // at wg == 32, i.e. below roughly q*batch = 65k -- the whole suite.
                // evidence: docs/perf/trsm.md#the-missing-group-barrier
                // Mem::Global fences both spaces, as the legacy sycl::group_barrier did.
                ex.barrier(v2::Mem::Global);
                // A non-finite 1/Lc(s,s) flips the group-uniform flag back to division.
                dla::tp_diag_recip<N, Ar>(ex, sLc, n, unit, sRd, sDiv);
                ex.barrier(v2::Mem::Global);
                const dla::TpDiag<D> dg{sLc, sRd, unit, sDiv[0] != 0};

                v2::Frag<D, N> x;
                if constexpr (kStageLeft) {
                    constexpr int ROUNDS = (N + TILE_ROWS - 1) / TILE_ROWS;
                    const dla::FlipRows<D> Bc{Bb, ldb, n, q, fwd};
                    D* my_col = sTile + lane * (TILE_ROWS + 1);
BATCHLAS_UNROLL_FULL
                    for (int k = 0; k < ROUNDS; ++k) {
                        // Barrier BEFORE overwriting the tile (last round's reads); no
                        // barrier sits under `live`: one only some lanes reach is undefined.
                        ex.barrier(v2::Mem::Global);
                        dla::lacpy_tile_in<TILE_ROWS, TILE_ROWS + 1>(ex, Bc, k * TILE_ROWS, u0, sTile);
                        ex.barrier(v2::Mem::Global);
                        dla::trsv_lower_rows<N, TILE_ROWS, Ar>(x, k * TILE_ROWS, sLc, dg, [&](int s) {
                            D rhs{};
                            if (live && s < n) rhs = Ar::mul(alpha_d, my_col[s - k * TILE_ROWS]);
                            return rhs;
                        });
                    }
BATCHLAS_UNROLL_FULL
                    for (int k = 0; k < ROUNDS; ++k) {
                        // A non-live lane's column is one lacpy_tile_out refuses to write.
                        ex.barrier(v2::Mem::Global);
                        dla::frag_copy_out<TILE_ROWS>(x, k * TILE_ROWS, my_col);
                        ex.barrier(v2::Mem::Global);
                        dla::lacpy_tile_out<TILE_ROWS, TILE_ROWS + 1>(ex, sTile, Bc, k * TILE_ROWS, u0);
                    }
                } else {
                    const auto bu = dla::CanonRhs<D, kLeft>(Bb, n, ldb, fwd).rhs(u);
                    dla::trsv_lower_rows<N, N, Ar>(x, 0, sLc, dg, [&](int s) {
                        D rhs{};
                        if (live && s < n) rhs = Ar::mul(alpha_d, bu(s));
                        return rhs;
                    });
                    v2::store(x, bu, live ? n : 0);
                }
            });
    });

    return ctx.get_event();
}

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
            throw batchlas::unsupported(
                "BatchLAS: trsm_native_v1 called with triangular order " +
                std::to_string(A.rows()) +
                ", which exceeds this scalar's CTA register capacity of " +
                std::to_string(trsm_max_bucket<D_>()) + ". Orders above the "
                "capacity are the blocked driver's (V2's) job; the CTA kernel cannot "
                "serve them and must not silently solve a leading submatrix.");
    }
}

// V2 -- the host-blocked driver, for orders above V1's register capacity. rho is
// a BIJECTION on [0,n), so a canonical block and the already-solved set are both
// contiguous runs in STORED indices; fwd enters only through stored_off.
//
// ALPHA IS APPLIED EXACTLY ONCE per element of B: by the V1 solve on the first
// block, or as the trailing GEMM's BETA on every later one. beta=1 there is wrong
// for every alpha != 1 and still passes any alpha == 1 test. Sub-views pass ld AND
// stride explicitly; the 6-arg constructor defaults stride to ld*cols when 0.
// The outer block width is deliberately NOT the CTA capacity, which would pin one
// GEMM dimension at 32. evidence: docs/perf/trsm.md#the-two-level-blocked-driver
inline int trsm_outer_block_default() { return 128; }

// Widening helps Side::Left and HURTS Side::Right, whose trailing update puts the
// width on the other GEMM dimension. evidence: docs/perf/trsm.md#rejected-outer_nb-of-128-for-sideright
inline int trsm_outer_block(int cta_nb, Side side) {
    // Read per call, NEVER latch in a static: an A/B would measure one arm twice.
    // 0 = unset; the side-dependent default applies on the next line.
    // evidence: docs/perf/trsm.md#tuning-knobs-and-environment
    const int env = batchlas::settings().geometry.trsm_outer_nb;
    const int want = env ? env : (side == Side::Left ? trsm_outer_block_default() : cta_nb);
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
    if (!trailing_gemm) {
        throw batchlas::invalid_argument("trsm_native_blocked: trailing_gemm is required (pass the public gemm)");
    }
    const Canonical can = canonicalise(side, uplo, transA, diag);
    const int n = static_cast<int>(A.rows());
    const int q = static_cast<int>(side == Side::Left ? B.cols() : B.rows());
    const int nb = trsm_cta_max_n<T>();          // the CTA capacity: the INNER block
    const int outer_nb = trsm_outer_block(nb, side);  // the trailing-update block

    const int lda = A.ld(), ldb = B.ld();
    const int sa = A.stride(), sb = B.stride();
    const int bs = A.batch_size();

    // Every block re-attaches the call's batch mask, which the raw-pointer view drops.
    const auto mask = batchlas::detail::call_mask("trsm", A, B);
    auto sub = [&mask](const MatrixView<T, MatrixFormat::Dense>& V,
                       int r0, int nr, int c0, int nc, int ld, int stride, int batch) {
        return MatrixView<T, MatrixFormat::Dense>(
            V.data_ptr() + static_cast<std::ptrdiff_t>(c0) * ld + r0,
            nr, nc, ld, stride, batch).with_batch_mask(mask);
    };

    // Canonical range [a, b) -> the stored row offset: [a,b) when fwd,
    // [n-b, n-a) when not. Both levels go through this, sharing one convention.
    auto stored_off = [&](int a, int b) { return can.fwd ? a : (n - b); };

    auto apply_update = [&](int t_lo, int t_hi, int p_lo, int p_hi, T beta) {
        const int m = t_hi - t_lo;
        const int k = p_hi - p_lo;
        const int r0 = stored_off(t_lo, t_hi);
        const int s0 = stored_off(p_lo, p_hi);

        const auto C = (side == Side::Left) ? sub(B, r0, m, 0, q, ldb, sb, bs)
                                            : sub(B, 0, q, r0, m, ldb, sb, bs);
        const auto X = (side == Side::Left) ? sub(B, s0, k, 0, q, ldb, sb, bs)
                                            : sub(B, 0, q, s0, k, ldb, sb, bs);
        // The A block is chosen so op() lands on the required sub-block, which
        // is why transA is passed through unchanged.
        const auto Aoff =
            (side == Side::Left)
                ? (can.do_trans ? sub(A, s0, k, r0, m, lda, sa, bs)
                                : sub(A, r0, m, s0, k, lda, sa, bs))
                : (can.do_trans ? sub(A, r0, m, s0, k, lda, sa, bs)
                                : sub(A, s0, k, r0, m, lda, sa, bs));

        if (side == Side::Left) {
            // C(m x q) := -op(Aoff)(m x k) * X(k x q) + beta*C
            (void)trailing_gemm(ctx, Aoff, X, C, T(-1), beta,
                          transA, Transpose::NoTrans,
                          ComputePrecision::Default);
        } else {
            // C(q x m) := -X(q x k) * op(Aoff)(k x m) + beta*C. X GOES IN THE A
            // POSITION: with the A block first, C would have at most nb rows
            // against the required q.
            (void)trailing_gemm(ctx, X, Aoff, C, T(-1), beta,
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
        // (void) on an Event: deliberate. This Queue is in-order, so the next submission
        // is already ordered after this one and the Event carries nothing the caller needs.
        (void)trsm_native_v1_dispatch<T>(ctx, Adiag, Bblk, alpha_eff, side, uplo, transA, diag);
    };

    // TWO LEVELS: the outer applies the whole solved prefix to a panel in one fat
    // GEMM; the inner is a right-looking loop against a prefix < OUTER_NB wide.
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

            // Block i+1's GEMM reads what block i's solve just wrote; an
            // out-of-order queue does not give that for free.
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

// Measured CTA capacity per type; the gate is stack frame == 0, zero spill and
// resident::sm89_fits at the widest rung of the ladder above -- registers per SUB-PARTITION,
// not the per-block 65,536. evidence: docs/perf/trsm.md#the-register-gate-and-the-cta-capacity
template <> int trsm_cta_max_n<float>()                { return 32; }
template <> int trsm_cta_max_n<double>()               { return 32; }
template <> int trsm_cta_max_n<std::complex<float>>()  { return 32; }
template <> int trsm_cta_max_n<std::complex<double>>() { return 32; }

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
