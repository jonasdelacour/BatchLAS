// Native batched TRSM — the kernel translation unit.
//
// WP3 step 3: V1, Side::Right, real types. See WP3_TRSM_SPEC_CORRECTIONS.md
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
// WHY Side::Right FIRST. The canonical RHS accessor for Right has du == 1, so
// threads differing in u touch consecutive addresses and both the load and the
// store are coalesced. Left has du == ldb and needs the transpose staging tile
// of §3.4; it is a separate step so that this one has zero barriers after the
// single staging barrier and the register question is answered on its own.

#include "trsm_native.hh"

#include "../linalg-impl.hh"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <complex>
#include <cstdlib>
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

// The smallest compile-time bucket >= n. Buckets are powers of two up to 64;
// the router guarantees n <= trsm_cta_max_n<T>() before this is consulted.
inline int smallest_bucket_ge(int n) {
    if (n <= 8) return 8;
    if (n <= 16) return 16;
    if (n <= 32) return 32;
    return 64;
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
template <typename T, int N>
Event trsm_native_v1_right(Queue& ctx,
                           const MatrixView<T, MatrixFormat::Dense>& A,
                           const MatrixView<T, MatrixFormat::Dense>& B,
                           T alpha,
                           Uplo uplo,
                           Transpose transA,
                           Diag diag) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "WP3 step 3 is real types only; complex needs a POD device scalar and a "
                  "guarded complex reciprocal, which GEMM's wide-scalar helpers do not "
                  "provide (they have no division).");

    const Canonical can = canonicalise(Side::Right, uplo, transA, diag);

    const int n = static_cast<int>(A.rows());
    const int q = static_cast<int>(B.rows());   // Side::Right -> q = B.rows()
    const int bs = static_cast<int>(A.batch_size());

    // Work-group sizing. Descending, because a larger work-group amortises the
    // triangle staging over more solves: A traffic / B traffic = n / (4*WG).
    const auto dev = ctx.device();
    const int max_wg = static_cast<int>(dev.get_property(DeviceProperty::MAX_WORK_GROUP_SIZE));
    const int cu = static_cast<int>(dev.get_property(DeviceProperty::MAX_COMPUTE_UNITS));

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
        sycl::local_accessor<T, 1> lc(sycl::range<1>(tri_elems), h);
        sycl::local_accessor<T, 1> rd(sycl::range<1>(N), h);
        // Work-group-uniform flag: does any diagonal reciprocal overflow? If so
        // the whole group reverts to division. See the guard note below.
        sycl::local_accessor<int, 1> use_div(sycl::range<1>(1), h);

        const T* a_ptr = A.data_ptr();
        T* b_ptr = B.data_ptr();
        const int lda = static_cast<int>(A.ld());
        const int ldb = static_cast<int>(B.ld());
        const int stride_a = static_cast<int>(A.stride());
        const int stride_b = static_cast<int>(B.stride());

        const bool do_trans = can.do_trans;
        const bool fwd = can.fwd;
        const bool unit = can.unit;

        h.parallel_for<TrsmCtaKernel<T, N, Side::Right>>(
            sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(bs) * groups * wg),
                              sycl::range<1>(wg)),
            [=](sycl::nd_item<1> it) {
                const int wg_id = static_cast<int>(it.get_group_linear_id());
                const int b = wg_id / groups;
                const int lane = static_cast<int>(it.get_local_linear_id());
                const int u = (wg_id % groups) * wg + lane;
                const bool live = (u < q);

                const T* Ab = a_ptr + static_cast<std::ptrdiff_t>(b) * stride_a;
                T* Bb = b_ptr + static_cast<std::ptrdiff_t>(b) * stride_b;

                T* sLc = lc.template get_multi_ptr<sycl::access::decorated::no>().get();
                T* sRd = rd.template get_multi_ptr<sycl::access::decorated::no>().get();
                int* sDiv = use_div.template get_multi_ptr<sycl::access::decorated::no>().get();

                if (lane == 0) sDiv[0] = 0;

                // ---- Cooperative staging of the canonical triangle ---------
                // rho(s) = fwd ? s : n-1-s maps canonical index to stored index.
                // Lc(s,t) = opA(rho(t), rho(s)) for Side::Right -- note the
                // OPERAND ORDER IS SWAPPED relative to Side::Left. Getting that
                // backwards is invisible on the symmetric cells and wrong on the
                // rest, which is the sort of defect a residual test catches only
                // if the test uses a non-symmetric triangle.
                for (size_t idx = lane; idx < tri_elems; idx += static_cast<size_t>(wg)) {
                    // Recover (s,t) from the packed index.
                    int s = 0;
                    while (tri_idx(s + 1, 0) <= static_cast<int>(idx)) ++s;
                    const int t = static_cast<int>(idx) - tri_idx(s, 0);

                    T v;
                    if (s >= n || t >= n) {
                        // Zero padding, with a unit diagonal, so the unrolled
                        // tail computes zeros instead of branching.
                        v = (s == t) ? T(1) : T(0);
                    } else {
                        const int rs = fwd ? s : (n - 1 - s);
                        const int rt = fwd ? t : (n - 1 - t);
                        // opA(r,c) with (r,c) = (rho(t), rho(s)) for Right.
                        const int r = rt;
                        const int c = rs;
                        v = do_trans
                                ? Ab[c + static_cast<std::ptrdiff_t>(r) * lda]   // A(c,r)
                                : Ab[r + static_cast<std::ptrdiff_t>(c) * lda];  // A(r,c)
                    }
                    sLc[idx] = v;
                }

                // ---- Diagonal reciprocals, guarded -------------------------
                // The recurrence multiplies by rd[s] = 1/Lc(s,s) instead of
                // dividing, which is the only arithmetic deviation from the
                // reference loop nest. It is unsafe in exactly one place: if
                // Lc(s,s) is small enough that 1/Lc(s,s) overflows, the multiply
                // produces inf where the division would have produced a finite
                // number. So the reciprocal is CHECKED, and any thread that sees
                // a non-finite one flips a work-group-uniform flag that reverts
                // the whole group to division. BATCHLAS_TRSM_DIAG=div forces
                // that path unconditionally, for A/B-ing the accuracy claim.
                for (int s = lane; s < N; s += wg) {
                    T r = T(1);
                    if (s < n && !unit) {
                        const T d = sLc[tri_idx(s, s)];
                        if (!finite_recip(d, r)) {
                            sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                             sycl::memory_scope::work_group,
                                             sycl::access::address_space::local_space>(sDiv[0])
                                .store(1);
                            r = T(1);
                        }
                    }
                    sRd[s] = r;
                }

                sycl::group_barrier(it.get_group());

                const bool divide = (sDiv[0] != 0);

                // ---- The recurrence, fully unrolled ------------------------
                // Canonical RHS accessor for Side::Right:
                //   b0 = fwd ? 0 : (n-1)*ldb ; ds = fwd ? +ldb : -ldb ; du = 1
                // du == 1 is why Right goes first: lanes differ in u, so the
                // load and the store are both coalesced.
                const std::ptrdiff_t b0 =
                    fwd ? 0 : static_cast<std::ptrdiff_t>(n - 1) * ldb;
                const std::ptrdiff_t ds =
                    fwd ? static_cast<std::ptrdiff_t>(ldb) : -static_cast<std::ptrdiff_t>(ldb);

                T x[N];
#pragma unroll
                for (int s = 0; s < N; ++s) {
                    T acc = T(0);
#pragma unroll
                    for (int t = 0; t < N; ++t) {
                        if (t < s) acc += sLc[tri_idx(s, t)] * x[t];
                    }
                    T rhs = T(0);
                    if (live && s < n) {
                        rhs = alpha * Bb[b0 + static_cast<std::ptrdiff_t>(s) * ds + u];
                    }
                    T v = rhs - acc;
                    if (!unit) {
                        v = divide ? (v / sLc[tri_idx(s, s)]) : (v * sRd[s]);
                    }
                    x[s] = v;
                }

#pragma unroll
                for (int s = 0; s < N; ++s) {
                    if (live && s < n) {
                        Bb[b0 + static_cast<std::ptrdiff_t>(s) * ds + u] = x[s];
                    }
                }
            });
    });

    return ctx.get_event();
}

// Runtime bucket dispatch. Direct-call entry used by tests at this step.
template <typename T>
Event trsm_native_v1_right_dispatch(Queue& ctx,
                                    const MatrixView<T, MatrixFormat::Dense>& A,
                                    const MatrixView<T, MatrixFormat::Dense>& B,
                                    T alpha,
                                    Uplo uplo,
                                    Transpose transA,
                                    Diag diag) {
    switch (smallest_bucket_ge(static_cast<int>(A.rows()))) {
        case 8:  return trsm_native_v1_right<T, 8>(ctx, A, B, alpha, uplo, transA, diag);
        case 16: return trsm_native_v1_right<T, 16>(ctx, A, B, alpha, uplo, transA, diag);
        case 32: return trsm_native_v1_right<T, 32>(ctx, A, B, alpha, uplo, transA, diag);
        default: return trsm_native_v1_right<T, 64>(ctx, A, B, alpha, uplo, transA, diag);
    }
}

template Event trsm_native_v1_right_dispatch<float>(
    Queue&, const MatrixView<float, MatrixFormat::Dense>&,
    const MatrixView<float, MatrixFormat::Dense>&, float, Uplo, Transpose, Diag);
// double is instantiated deliberately, as the FALSIFICATION PROBE for the
// spec's n_cta(double) = 32. That number comes from a "256 B/thread register
// cliff" which gemm_kernels.cc:725-735 records as measured false, so N=64 double
// -- 64 doubles of accumulator per thread -- is exactly the configuration the
// hypothesis says must spill. The register probe decides it, not the spec.
template Event trsm_native_v1_right_dispatch<double>(
    Queue&, const MatrixView<double, MatrixFormat::Dense>&,
    const MatrixView<double, MatrixFormat::Dense>&, double, Uplo, Transpose, Diag);

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
// STILL ZERO, THOUGH, and that is not the register result. Only Side::Right
// exists in this step; a non-zero capacity would make supports() accept
// Side::Left too, for which there is no kernel. The capacities become
// {32, 32, ...} in the step that adds Side::Right's counterpart, together with a
// side gate in RouteTable<Op::trsm,T>::supports().
template <> int trsm_cta_max_n<float>()                { return 0; }
template <> int trsm_cta_max_n<double>()               { return 0; }
template <> int trsm_cta_max_n<std::complex<float>>()  { return 0; }
template <> int trsm_cta_max_n<std::complex<double>>() { return 0; }

}  // namespace batchlas::sycl_trsm
