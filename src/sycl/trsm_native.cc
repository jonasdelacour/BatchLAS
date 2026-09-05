// Native batched TRSM: the register-resident CTA solver (V1) and the
// host-blocked driver (V2). See docs/perf/trsm.md.
//
// V1 puts one thread on each independent solve; x[N] must stay in registers, so N
// is a compile-time bucket >= n and the loops are fully unrolled (indexing x by a
// runtime variable moves it to local memory). Rows n..N-1 pad with Lc(s,s)=1.

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

// The 24 (side, uplo, transA, diag) combinations fold into ONE recurrence over a
// canonical unit-lower Lc. evidence: docs/perf/trsm.md#design-v1-v2-and-the-canonical-fold
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

// Packed lower triangle, row-major by s: N(N+1)/2 elements. All threads read the
// same Lc(s,t) at a step, so this is an SLM broadcast and bank layout is moot.
constexpr int tri_idx(int s, int t) { return s * (s + 1) / 2 + t; }

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

template <typename T>
inline bool finite_recip(T d, T& out) {
    const T r = T(1) / d;
    out = r;
    return sycl::isfinite(r);
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

    static_assert(256 * 226 <= 65536,
                  "the work-group ceiling is set by registers per block, not by occupancy; "
                  "re-run scripts/register_probe.sh before raising it");
    int wg = 32;
    for (int cand : {256, 128, 64, 32}) {
        if (cand > max_wg) continue;
        wg = cand;
        const int64_t groups_c = (q + cand - 1) / cand;
        if (static_cast<int64_t>(bs) * groups_c >= static_cast<int64_t>(4) * cu) break;
    }

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

                // rho(s) = fwd ? s : n-1-s maps canonical to stored index, and the
                // operand order of Lc is swapped between the sides. ConjTrans
                // conjugates INCLUDING THE DIAGONAL, so rd below inverts conj(d).
                for (size_t idx = lane; idx < tri_elems; idx += static_cast<size_t>(wg)) {
                    int s = 0;
                    while (tri_idx(s + 1, 0) <= static_cast<int>(idx)) ++s;
                    const int t = static_cast<int>(idx) - tri_idx(s, 0);

                    D v;
                    if (s >= n || t >= n) {
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

                // REQUIRED; without it the solve returns WRONG ANSWERS: sLc is
                // written strided by lane and read by a different lane, and lane 0
                // zeroes sDiv[0] while any lane may store 1. The race cannot show
                // at wg == 32, i.e. below roughly q*batch = 65k -- the whole suite.
                // evidence: docs/perf/trsm.md#the-missing-group-barrier
                sycl::group_barrier(it.get_group());

                // The recurrence multiplies by rd[s] = 1/Lc(s,s), which is inf
                // where a division would stay finite; a thread seeing a non-finite
                // reciprocal flips a group-uniform flag back to division.
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

                const std::ptrdiff_t unit_s =
                    (SideV == Side::Left) ? 1 : static_cast<std::ptrdiff_t>(ldb);
                const std::ptrdiff_t du =
                    (SideV == Side::Left) ? static_cast<std::ptrdiff_t>(ldb) : 1;
                const std::ptrdiff_t b0 = fwd ? 0 : static_cast<std::ptrdiff_t>(n - 1) * unit_s;
                const std::ptrdiff_t ds = fwd ? unit_s : -unit_s;

                const auto left_addr = [&](int s_can, int col) -> std::ptrdiff_t {
                    const int row = fwd ? s_can : (n - 1 - s_can);
                    return static_cast<std::ptrdiff_t>(row) +
                           static_cast<std::ptrdiff_t>(col) * ldb;
                };

                constexpr int STEP   = kStageLeft ? TILE_ROWS : N;
                constexpr int ROUNDS = (N + STEP - 1) / STEP;

                D x[N];
#pragma unroll
                for (int k = 0; k < ROUNDS; ++k) {
                    if constexpr (kStageLeft) {
                        // Barrier BEFORE overwriting the tile: last round's reads
                        // must have retired. Both barriers sit outside every `live`
                        // test -- a barrier only some lanes reach is undefined.
                        sycl::group_barrier(it.get_group());
                        // r varies fastest: lanes read consecutive ROWS.
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

                if constexpr (kStageLeft) {
#pragma unroll
                    for (int k = 0; k < ROUNDS; ++k) {
                        sycl::group_barrier(it.get_group());
                        // The guard is on the STEP, not on `live`: a non-live
                        // lane's column is one the store below refuses to write.
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
            throw std::runtime_error(
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
    static const int env = [] {
        const char* raw = std::getenv("BATCHLAS_TRSM_OUTER_NB");
        if (!raw || !*raw) return 0;
        const int v = std::atoi(raw);
        return v > 0 ? v : 0;
    }();
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
    // Default to the native kernel so this TU stands alone; the facade passes the ROUTED gemm.
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
    const int outer_nb = trsm_outer_block(nb, side);  // the trailing-update block

    const int lda = A.ld(), ldb = B.ld();
    const int sa = A.stride(), sb = B.stride();
    const int bs = A.batch_size();

    auto sub = [](const MatrixView<T, MatrixFormat::Dense>& V,
                  int r0, int nr, int c0, int nc, int ld, int stride, int batch) {
        return MatrixView<T, MatrixFormat::Dense>(
            V.data_ptr() + static_cast<std::ptrdiff_t>(c0) * ld + r0,
            nr, nc, ld, stride, batch);
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
            trailing_gemm(ctx, Aoff, X, C, T(-1), beta,
                          transA, Transpose::NoTrans,
                          ComputePrecision::Default);
        } else {
            // C(q x m) := -X(q x k) * op(Aoff)(k x m) + beta*C. X GOES IN THE A
            // POSITION: with the A block first, C would have at most nb rows
            // against the required q.
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
// registers x work-group <= 65536. evidence: docs/perf/trsm.md#the-register-gate-and-the-cta-capacity
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
