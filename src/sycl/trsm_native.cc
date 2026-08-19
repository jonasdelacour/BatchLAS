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

// Packed lower-triangle index, row-major by s: N(N+1)/2 elements.
// All threads read the same Lc(s,t) at the same step, so this is an SLM
// BROADCAST — bank layout is irrelevant to conflicts here.
constexpr int tri_idx(int s, int t) { return s * (s + 1) / 2 + t; }

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

                D x[N];
#pragma unroll
                for (int s = 0; s < N; ++s) {
                    D acc = D{};
#pragma unroll
                    for (int t = 0; t < N; ++t) {
                        if (t < s) sycl_device::fma_acc(acc, sLc[tri_idx(s, t)], x[t]);
                    }
                    D rhs = D{};
                    if (live && s < n) {
                        rhs = sycl_device::dev_mul(
                            alpha_d, Bb[b0 + static_cast<std::ptrdiff_t>(s) * ds + u * du]);
                    }
                    D v = sycl_device::dev_sub(rhs, acc);
                    if (!unit) {
                        v = divide ? sycl_device::dev_div(v, sLc[tri_idx(s, s)])
                                   : sycl_device::dev_mul(v, sRd[s]);
                    }
                    x[s] = v;
                }

#pragma unroll
                for (int s = 0; s < N; ++s) {
                    if (live && s < n) {
                        Bb[b0 + static_cast<std::ptrdiff_t>(s) * ds + u * du] = x[s];
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
    switch (smallest_bucket_ge(static_cast<int>(A.rows()))) {
        case 8:  return trsm_native_v1<T, 8, SideV>(ctx, A, B, alpha, uplo, transA, diag);
        case 16: return trsm_native_v1<T, 16, SideV>(ctx, A, B, alpha, uplo, transA, diag);
        case 32: return trsm_native_v1<T, 32, SideV>(ctx, A, B, alpha, uplo, transA, diag);
        default:
            // ENFORCED, not assumed. The router already caps the order via
            // supports(CTA), so reaching here means a direct caller (V2, or a
            // test) exceeded the contract -- and the alternative to throwing is
            // returning a wrong answer for the rows that do not fit.
            throw std::runtime_error(
                "BatchLAS: trsm_native_v1 called with triangular order " +
                std::to_string(A.rows()) +
                ", which exceeds the CTA register capacity of 32. Orders above the "
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
template <typename T>
Event trsm_native_blocked(Queue& ctx,
                          const MatrixView<T, MatrixFormat::Dense>& A,
                          const MatrixView<T, MatrixFormat::Dense>& B,
                          T alpha,
                          Side side,
                          Uplo uplo,
                          Transpose transA,
                          Diag diag) {
    const Canonical can = canonicalise(side, uplo, transA, diag);
    const int n = static_cast<int>(A.rows());
    const int q = static_cast<int>(side == Side::Left ? B.cols() : B.rows());
    const int nb = trsm_cta_max_n<T>();          // the CTA capacity IS the block size

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

    for (int lo = 0; lo < n; lo += nb) {
        const int hi = std::min(n, lo + nb);
        const int m = hi - lo;                    // rows of this block
        const int k = lo;                         // rows already solved
        const int r0 = can.fwd ? lo : (n - hi);
        const int s0 = can.fwd ? 0 : (n - lo);

        if (k > 0) {
            // Trailing update. beta = alpha carries the scaling; see the
            // contract above.
            const auto C = (side == Side::Left)
                               ? sub(B, r0, m, 0, q, ldb, sb, bs)
                               : sub(B, 0, q, r0, m, ldb, sb, bs);
            const auto X = (side == Side::Left)
                               ? sub(B, s0, k, 0, q, ldb, sb, bs)
                               : sub(B, 0, q, s0, k, ldb, sb, bs);
            // The A block is chosen so that op() lands on the required
            // sub-block, which is why transA is passed through unchanged.
            const auto Aoff =
                (side == Side::Left)
                    ? (can.do_trans ? sub(A, s0, k, r0, m, lda, sa, bs)
                                    : sub(A, r0, m, s0, k, lda, sa, bs))
                    : (can.do_trans ? sub(A, r0, m, s0, k, lda, sa, bs)
                                    : sub(A, s0, k, r0, m, lda, sa, bs));

            if (side == Side::Left) {
                // C(m x q) := -op(Aoff)(m x k) * X(k x q) + alpha*C
                sycl_gemm::gemm_custom<T>(ctx, Aoff, X, C, T(-1), alpha,
                                          transA, Transpose::NoTrans,
                                          ComputePrecision::Default);
            } else {
                // C(q x m) := -X(q x k) * op(Aoff)(k x m) + alpha*C.
                // X GOES IN THE A POSITION. The obvious single form with the A
                // block first produces a C of at most nb rows against the
                // required q and does not conform for any transpose.
                sycl_gemm::gemm_custom<T>(ctx, X, Aoff, C, T(-1), alpha,
                                          Transpose::NoTrans, transA,
                                          ComputePrecision::Default);
            }
        }

        const auto Adiag = sub(A, r0, m, r0, m, lda, sa, bs);
        const auto Bblk = (side == Side::Left) ? sub(B, r0, m, 0, q, ldb, sb, bs)
                                               : sub(B, 0, q, r0, m, ldb, sb, bs);
        const T alpha_eff = (k == 0) ? alpha : T(1);
        trsm_native_v1_dispatch<T>(ctx, Adiag, Bblk, alpha_eff, side, uplo, transA, diag);

        // Block i+1's GEMM reads what block i's solve just wrote. An in-order
        // queue gives that for free; an out-of-order one does not, and a caller
        // may construct either (sycl-device-queue.hh:239 defaults in_order=true
        // but it is a parameter). This is a correctness requirement, not a
        // tuning choice, and it costs nothing on the default path.
        if (!ctx.in_order()) ctx.wait();
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
    const MatrixView<float, MatrixFormat::Dense>&, float, Side, Uplo, Transpose, Diag);
template Event trsm_native_blocked<double>(
    Queue&, const MatrixView<double, MatrixFormat::Dense>&,
    const MatrixView<double, MatrixFormat::Dense>&, double, Side, Uplo, Transpose, Diag);
template Event trsm_native_blocked<std::complex<float>>(
    Queue&, const MatrixView<std::complex<float>, MatrixFormat::Dense>&,
    const MatrixView<std::complex<float>, MatrixFormat::Dense>&, std::complex<float>,
    Side, Uplo, Transpose, Diag);
template Event trsm_native_blocked<std::complex<double>>(
    Queue&, const MatrixView<std::complex<double>, MatrixFormat::Dense>&,
    const MatrixView<std::complex<double>, MatrixFormat::Dense>&, std::complex<double>,
    Side, Uplo, Transpose, Diag);

template <> bool trsm_blocked_available<float>()                { return true; }
template <> bool trsm_blocked_available<double>()               { return true; }
template <> bool trsm_blocked_available<std::complex<float>>()  { return true; }
template <> bool trsm_blocked_available<std::complex<double>>() { return true; }

}  // namespace batchlas::sycl_trsm
