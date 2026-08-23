// Native batched GETRI -- write the permutation straight into C, then two
// ROUTED triangular solves.
//
// getri takes the FACTORED A (from getrf) and writes A^-1 into C
// (functions/getri.hh:55-61). With A = F^{-1} L U, where F is the interchange
// sequence applied FORWARDS:
//
//     A^-1 = U^-1 L^-1 F
//
// so the whole op is: set C := F, solve L (unit lower) against it, solve U
// (non-unit upper) against it. TWO trsm calls and NO permutation kernel.
//
// This file sits in EXTENSIONS_FACTORIZATION_SOURCES and not in
// EXTENSIONS_CTA_SOURCES: it shares no device symbol with the getrf pair
// (src/extensions/CMakeLists.txt:77-89's cluster rule, orgqr_blocked.cc's
// precedent). It does not include getrf_cta_device.hh.
//
// ===========================================================================
// THE MEASUREMENT, AND ITS CROSSOVER. Composed against cublas<t>getriBatched at
// saturating batch, in process, host oracle (experiments/wp6_lu/baseline/):
//
//     n(batch)    float   double  cfloat  cdouble
//      32(8192)    0.54    0.23    0.23    0.23
//      64(8192)    0.83    0.53    0.35    0.54
//     128(4096)    1.32    0.90    1.06    0.89
//     256(2048)    3.89    1.16    2.05    1.04
//     512(512)     5.75    1.28    3.01    1.02
//    1024(128)    15.66    1.16    6.05    1.11
//    2048(32)     74.87    3.93   25.88    4.30
//
// Geomean 1.60x over 28 cells, 18 wins, worst 0.23x, best 74.9x. Crossover
// n ~ 128 for float/cfloat and n ~ 256 for double/cdouble; BELOW it cuBLAS's
// small-n getriBatched path wins by up to 4.3x.
//
// TWO HONESTY CONSTRAINTS ON THOSE NUMBERS, stated rather than silently
// corrected: (a) EVERY n >= 512 CELL IS AGAINST AN UNSATURATED VENDOR -- cuBLAS
// getrf at float n=1024 is still falling 10% from batch 128 to 256, and cdouble
// n=2048 does 64x the work for 1.03x the time from batch 1 to 64, so the 74.9x
// is a comparison against a routine barely using the GPU at that batch;
// (b) the grid's batch schedule penalises the vendor at getri float n=256, which
// is best at batch 256 (13.85 us/item) and degrades to 20.38 at 2048, so that
// cell carries ~1.47x of pessimism.
//
// preferred() is FALSE for every shape regardless, so none of this routes yet.
// The crossover above is the shape of the window a later routing step would
// write, and it is a per-type window -- which is precisely why it is not being
// guessed at here.
//
// ===========================================================================
// THE WIN IS MOSTLY THE PERMUTATION, AND getri GETS IT FOR FREE.
//
// A LAPACK-faithful laswp is 51% of the composed call at n=128 (measured by a
// deliberate break: getri_trsm 0.4580 -> 0.2251 ms without the row exchange).
// It is structurally slow: ipiv is a SEQUENCE so it must be walked in order, one
// work-item per column, and in column-major consecutive work-items land ld apart
// -- 32 transactions per warp access. Collapsing it to a gather turns the
// composition's geomean from 0.97x into 1.60x, and getri needs NO kernel and NO
// workspace to get that: it writes F straight into C instead of writing the
// identity and then permuting it. Same store count, one kernel.
//
// HOW F IS COMPUTED WITHOUT A PERMUTATION ARRAY, which is the part that keeps
// the workspace at zero. F = S_{n-1} ... S_0 and (F v)_i = v_{perm[i]}, so
// F[i, perm[i]] = 1. perm[i] is obtained by tracing position i BACKWARDS through
// the list -- r = i; for k = n-1 .. 0: r = S_k(r) -- which each work-item does
// INDEPENDENTLY for its own i, in registers, reading ipiv from global where
// every work-item of the group wants the same element at the same instant (an
// L1 broadcast, not DRAM traffic).
//
// THE REJECTED ALTERNATIVES, and why:
//   * a forward walk into a shared perm[] array needs n ints of local memory per
//     work-group -- 8 KB at n=2048 -- which would put a CAPACITY on an op whose
//     route table has no field to advertise one, i.e. a throw on a call the
//     table had promised;
//   * the same array in global memory is workspace (262,144 B at n=2048
//     batch=32, 1,048,576 B at n=32 batch=8192) for no benefit;
//   * writing the identity and PERMUTING it is the 0.97x arm this design exists
//     to avoid.
// The backward trace costs O(n) per work-item, i.e. O(n^2) per matrix -- the
// same order as the zero-fill of C that has to happen anyway, and three orders
// below the 2n^3 of the solves.
//
// ===========================================================================
// TWO CONTRACT FACTS, BOTH MEASURED, BOTH HONOURED HERE:
//   (a) A IS NOT WRITTEN. cuBLAS's prototype takes `const T* const A[]`
//       (cublas_api.h:5568-5576) and measured max|A_after - A_factored| == 0 for
//       all four types. Nothing below writes A. cuBLAS also does not support
//       in-place (A == C), and neither does this arm -- it is refused explicitly
//       rather than left to produce garbage, because C is zeroed before A's
//       triangles are read.
//   (b) info IS EXACT-ZERO SEMANTICS, NOT A TOLERANCE (getrf_native.hh's PIVOT
//       CONTRACT note 4): ?GETRI reports the first i with U(i,i) exactly zero,
//       1-based. A kernel that flags |U(i,i)| < eps reports non-zero where the
//       vendor reports zero, and that divergence is invisible to any
//       native-vs-native test. potrf's first-failure masking and its quench are
//       NOT copied: LAPACK's LU family does neither and neither does cuBLAS.

#include "getri_native.hh"

#include "../sycl/device_scalar.hh"
#include "../queue.hh"
#include "../util/template-instantiations.hh"

#include <batchlas/util/mempool.hh>

#include <sycl/sycl.hpp>

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace batchlas {
namespace sycl_getri {

namespace {

template <typename T> class GetriZeroKernel;
template <typename T> class GetriPermKernel;

// C := 0, over the LOGICAL rows x cols x batch region only.
//
// NOT ctx->fill over stride*batch: C may be a view into a larger buffer, and
// writing its padding (or a neighbouring view's rows) would be a silent
// corruption that no residual on C could see. The range is (batch, rows*cols)
// with the element index row-fastest, so consecutive work-items write
// consecutive rows of one column -- coalesced in column-major.
//
// It is fully parallel over batch AND elements, not over batch alone: at the
// smallest interesting cell (n=32, batch=8192) that is 8.4M work-items and at
// the largest (n=2048, batch=32) it is 134M, against 128 SMs.
//
// THE 2-D RANGE AND ITS TWO 64-BIT DIVISIONS ARE MEASURED, NOT ACCIDENTAL. The
// obvious cleanup -- range<3>(batch, n, n), which removes both divisions and keeps
// the row in the fastest-varying dimension, so the coalescing above is unchanged --
// was built and swept, and it is NOT faster: geomean 1.000 over the 78-cell getri
// saturation sweep, and 0.874x at the largest fill in it (float n=512, batch=1024:
// 39.31 ms -> 45.00 ms, relsd 7%). The divisions are evidently not what this kernel
// is bound by. Left as it is, with the negative result recorded so it is not
// re-attempted as an obvious win.
template <typename T>
Event getri_zero_c_launch(Queue& ctx, T* c_ptr, int ldc, int stride_c,
                          int n, int batch) {
    using DM = sycl_device::DevMap<T>;
    using D = typename DM::type;
    static_assert(sizeof(D) == sizeof(T), "device scalar must be layout-compatible");

    D* const cp = reinterpret_cast<D*>(c_ptr);
    const std::size_t elems = static_cast<std::size_t>(n) * static_cast<std::size_t>(n);

    ctx->submit([&](sycl::handler& h) {
        h.parallel_for<GetriZeroKernel<T>>(
            sycl::range<2>(static_cast<std::size_t>(batch), elems),
            [=](sycl::item<2> it) {
                const int b = static_cast<int>(it.get_id(0));
                const std::size_t e = it.get_id(1);
                const int r = static_cast<int>(e % static_cast<std::size_t>(n));
                const int c = static_cast<int>(e / static_cast<std::size_t>(n));
                cp[static_cast<std::ptrdiff_t>(b) * stride_c +
                   static_cast<std::ptrdiff_t>(c) * ldc + r] = D{};
            });
    });
    return ctx.get_event();
}

// C := F (the ones of the permutation matrix) and, if requested, info.
//
// ONE WORK-GROUP PER MATRIX, because info is a MINIMUM over the diagonal and a
// reduction needs somewhere to land. The dominant term -- the backward traces --
// is spread over the whole work-group, so this is not the "parallel over batch
// only" defect; it is nonetheless the one launch in this file whose work-group
// COUNT is the batch, which at batch=32 leaves most of a 128-SM device idle. The
// work there is ~n^2 = 4.2M trace steps per matrix, about 10 us in total against
// a call that takes hundreds of milliseconds, so it is stated rather than fixed.
template <typename T>
Event getri_perm_launch(Queue& ctx,
                        const T* a_ptr, int lda, int stride_a,
                        T* c_ptr, int ldc, int stride_c,
                        int n, int batch,
                        const int* piv, int piv_stride,
                        int32_t* info_ptr, bool want_info,
                        int wg) {
    using DM = sycl_device::DevMap<T>;
    using D = typename DM::type;
    static_assert(sizeof(D) == sizeof(T), "device scalar must be layout-compatible");

    const D* const ap = reinterpret_cast<const D*>(a_ptr);
    D* const cp = reinterpret_cast<D*>(c_ptr);

    ctx->submit([&](sycl::handler& h) {
        sycl::local_accessor<int, 1> slot(sycl::range<1>(1), h);
        h.parallel_for<GetriPermKernel<T>>(
            sycl::nd_range<1>(sycl::range<1>(static_cast<std::size_t>(batch) *
                                             static_cast<std::size_t>(wg)),
                              sycl::range<1>(static_cast<std::size_t>(wg))),
            [=](sycl::nd_item<1> it) {
                const int b = static_cast<int>(it.get_group_linear_id());
                const int tid = static_cast<int>(it.get_local_linear_id());
                const int lwg = static_cast<int>(it.get_local_range(0));

                if (tid == 0) slot[0] = std::numeric_limits<int>::max();
                sycl::group_barrier(it.get_group());

                const int* const ip = piv + static_cast<std::ptrdiff_t>(b) * piv_stride;
                const D* const Ab = ap + static_cast<std::ptrdiff_t>(b) * stride_a;
                D* const Cb = cp + static_cast<std::ptrdiff_t>(b) * stride_c;

                for (int i = tid; i < n; i += lwg) {
                    // THE BACKWARD TRACE. perm[i] = S_0(S_1(...S_{n-1}(i)...)),
                    // i.e. follow position i back through the interchange list
                    // from the LAST transposition to the first. Each is its own
                    // inverse, so only the ORDER reverses -- the same fact that
                    // makes getrs's transposed permutation a reversed walk.
                    int r = i;
                    for (int k = n - 1; k >= 0; --k) {
                        const int p = ip[k] - 1;        // 1-BASED on the wire
                        if (r == k) {
                            r = p;
                        } else if (r == p) {
                            r = k;
                        }
                    }
                    // F[i, perm[i]] = 1, column-major.
                    Cb[static_cast<std::ptrdiff_t>(r) * ldc + i] =
                        sycl_device::dev_one<D>();

                    if (want_info) {
                        // EXACT zero, no epsilon -- see the header note.
                        const D u = Ab[static_cast<std::ptrdiff_t>(i) * lda + i];
                        if (sycl_device::dev_is_zero(u)) {
                            sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                             sycl::memory_scope::work_group,
                                             sycl::access::address_space::local_space>(
                                slot[0])
                                .fetch_min(i + 1);      // 1-BASED, LAPACK info
                        }
                    }
                }

                if (want_info) {
                    sycl::group_barrier(it.get_group());
                    if (tid == 0) {
                        const int v = slot[0];
                        info_ptr[b] =
                            (v == std::numeric_limits<int>::max()) ? 0 : static_cast<int32_t>(v);
                    }
                }
            });
    });
    return ctx.get_event();
}

}  // namespace

// ---------------------------------------------------------------------------
// THE CAPABILITY FLAG. TRUE for all four types. It moves no vendor-present
// traffic: preferred() is false everywhere (route_resolve.hh:60-63).
//
// DEFINED HERE, beside the driver, for potrf_native.hh:81-92's reason: full
// explicit specialisations link from wherever they sit, so co-location is what
// makes "the flag is true" and "the file is compiled" the same fact.
// ---------------------------------------------------------------------------
template <> bool getri_blocked_available<float>()                { return true; }
template <> bool getri_blocked_available<double>()               { return true; }
template <> bool getri_blocked_available<std::complex<float>>()  { return true; }
template <> bool getri_blocked_available<std::complex<double>>() { return true; }

// ---------------------------------------------------------------------------
// WORKSPACE. ZERO, and it is a consequence of the design rather than a
// coincidence: the permutation is a STORE PATTERN rather than a buffer (see the
// backward-trace note above), and the routed trsm takes no workspace at all.
//
// EVEN THE `info` FALLBACK IS FREE HERE, unlike getrf's. getrf's device body
// READS info to keep first-failure-wins across panels, so it needs a target even
// when the caller supplied none; getri's info is a pure output, so a caller that
// does not ask for it -- and src/extensions/inv.cc:48-49 is exactly such a
// caller -- simply does not get the reduction computed. `want_info` in the
// launcher above is that decision.
//
// So the facade's max(native, vendor) for getri will always be the VENDOR term
// (getri_vendor_buffer_size is BumpAllocator::allocation_size<int>(batch), a
// per-item info array, cublas.cc:1552 -- 512 B at n=2048 batch=32), and the LU
// family is never the workspace hazard WP5's ormqr was.
//
// It is called from INSIDE a layout function under measuring() (inv.cc:35,
// reached from inv_buffer_size at :54-57), so per mempool.hh:180-186 it must be
// PURE WITH RESPECT TO THE WORKSPACE and must not dereference A.data_ptr(). It
// reads nothing at all.
// ---------------------------------------------------------------------------
template <typename T>
std::size_t getri_blocked_buffer_size(Queue&, const MatrixView<T, MatrixFormat::Dense>&) {
    return 0;
}

// ---------------------------------------------------------------------------
// THE DRIVER.
// ---------------------------------------------------------------------------
template <typename T>
Event getri_blocked_dispatch(Queue& ctx,
                             const MatrixView<T, MatrixFormat::Dense>& A,
                             const MatrixView<T, MatrixFormat::Dense>& C,
                             Span<int64_t> pivots,
                             Span<std::byte> workspace,
                             Span<int32_t> info_out,
                             GetriSolveTrsm<T> solve_trsm) {
    static_cast<void>(workspace);   // this arm needs none; see the query above

    const int n = static_cast<int>(A.rows());
    const int batch = static_cast<int>(A.batch_size());

    // Every gate RouteTable<Op::getri,T>::supports() applies, re-applied because
    // this entry point is reachable WITHOUT the table -- and it must be, for
    // potrf_native.hh:126-141's reason: route_resolve.hh:165 falls through to
    // automatic() when a forced route is unsupported, so a pinned-route test that
    // is wrong about one gate silently measures cuBLAS and passes green.
    if (n < 1 || batch < 1) {
        throw std::invalid_argument("getri_blocked: degenerate extents");
    }
    if (A.rows() != A.cols()) {
        throw std::invalid_argument("getri_blocked: A must be square");
    }
    if (A.is_heterogeneous() || C.is_heterogeneous()) {
        throw std::invalid_argument("getri_blocked: heterogeneous batch is not supported");
    }
    const auto dev = ctx.device();
    if (dev.type != DeviceType::GPU) {
        throw std::invalid_argument("getri_blocked: GPU queues only");
    }
    if (!dev.supports_sub_group_size(32)) {
        // Transcribed from supports() and matching it exactly. This file's own
        // kernels carry no sub-group requirement; the routed trsm they call does.
        throw std::runtime_error("getri_blocked: device does not offer sub-group size 32");
    }

    // THESE THREE ARE NOT EXPRESSIBLE IN GetriShape and therefore cannot gate the
    // route -- src/backends/getri_route.hh records why: getri_buffer_size(ctx, A)
    // has no C, so the shape builder is a function of A ALONE and the query and
    // the call must be built from identical arguments. They are checked here, on
    // the actual spellings, which is the only place they can be.
    if (C.rows() != n || C.cols() != n) {
        throw std::invalid_argument("getri_blocked: C must be square of A's order");
    }
    if (C.batch_size() != A.batch_size()) {
        throw std::invalid_argument("getri_blocked: A and C must agree on batch size");
    }
    if (C.data_ptr() == A.data_ptr()) {
        // IN-PLACE IS REFUSED, matching cuBLAS, and here it is a hard requirement
        // rather than a convention: C is zeroed before A's triangles are read by
        // the solves, so an aliased pair would destroy the factorisation first.
        throw std::invalid_argument(
            "getri_blocked: in-place (C aliasing A) is not supported, as it is not by "
            "cublas<t>getriBatched");
    }

    if (pivots.size() < static_cast<std::size_t>(n) * static_cast<std::size_t>(batch)) {
        throw std::invalid_argument("getri_blocked: pivot span is shorter than n * batch");
    }
    if (!solve_trsm) {
        // AN ABSENT INJECTION THROWS rather than silently reaching for a native
        // trsm entry point: the two solves ARE this op, and picking trsm's arm
        // here would be WP3 step 16's defect re-created (trsm_native.hh:82-104,
        // fix at level3.cc:186-231). A direct caller injects
        // trsm<Backend::CUDA, T> itself, which is still a call no vendor getri
        // can serve.
        throw std::invalid_argument(
            "getri_blocked: the solve seam is empty. Inject the ROUTED batchlas::trsm "
            "(the facade does; a direct caller must too) -- this driver deliberately has "
            "no native fallback, so that the router, and not this file, chooses the trsm "
            "arm.");
    }

    // PACKED 1-BASED int32, the format cublas.cc:1537 and rocsolver.cc:227 read
    // through pivots.as_span<int>() and the one a native getrf writes. See
    // getrf_native.hh's PIVOT CONTRACT: the ops have independent env variables
    // and independent preferred() windows, so every mixture of native and vendor
    // arms is reachable and they must agree bit for bit.
    auto piv_i32 = pivots.as_span<int>();

    const bool want_info = info_out.size() >= static_cast<std::size_t>(batch);

    // THE WORK-GROUP WIDTH, DERIVED FROM n RATHER THAN HARDCODED -- AND IT BUYS
    // NOTHING MEASURABLE, which is stated rather than implied. The permutation
    // kernel's only loop is `for (i = tid; i < n; i += lwg)`, so a work-item with
    // tid >= n does the slot init, two barriers and nothing else: at float n=32,
    // batch=8192 the old constant 256 launched 2,097,152 work-items of which
    // 262,144 -- 12.5% -- had a row to trace. Narrowing the group to n is
    // therefore obviously right in DESCRIPTION, and over the 78-cell getri
    // saturation sweep it measures 0.9999 geomean (spread 0.982-1.023, i.e.
    // noise): this kernel is not what those cells are bound by. Kept because it
    // costs nothing and because a device with a small MAX_WORK_GROUP_SIZE should
    // not be handed a width chosen for this one; NOT kept as a performance claim.
    // Rounding up to a power of two and capping at 256 reproduces the old width
    // for every n >= 256 and narrows it below. The floor is 32 (one sub-group),
    // and like getrf_leaf_wg the device MAX_WORK_GROUP_SIZE is a downward clamp.
    const int max_wg = static_cast<int>(dev.get_property(DeviceProperty::MAX_WORK_GROUP_SIZE));
    int wg = 32;
    while (wg < n && wg < 256) wg <<= 1;
    while (wg > max_wg && wg > 32) wg >>= 1;

    (void)getri_zero_c_launch<T>(ctx, C.data_ptr(), C.ld(), C.stride(), n, batch);
    // In-order queues give the ordering for free; an out-of-order one does not.
    if (!ctx.in_order()) ctx.wait();

    (void)getri_perm_launch<T>(ctx, A.data_ptr(), A.ld(), A.stride(),
                               C.data_ptr(), C.ld(), C.stride(),
                               n, batch, piv_i32.data(), /*piv_stride=*/n,
                               want_info ? info_out.data() : nullptr, want_info, wg);
    if (!ctx.in_order()) ctx.wait();

    // C := L^-1 C, then C := U^-1 C. alpha comes THIRD in the public trsm
    // (functions/trsm.hh:100-108); the old spelling is a DELETED overload at
    // :121-138 so a stale call cannot silently compile into a wrong answer.
    (void)solve_trsm(ctx, A, C, T(1), Side::Left, Uplo::Lower,
                     Transpose::NoTrans, Diag::Unit);
    if (!ctx.in_order()) ctx.wait();
    return solve_trsm(ctx, A, C, T(1), Side::Left, Uplo::Upper,
                      Transpose::NoTrans, Diag::NonUnit);
}

// ---------------------------------------------------------------------------
// Instantiation: PER SCALAR TYPE ONLY, no Backend cross-product. Everything that
// needs a Backend arrives injected.
// ---------------------------------------------------------------------------
#define BATCHLAS_GETRI_INSTANTIATE(T)                                                      \
    template std::size_t getri_blocked_buffer_size<T>(                                     \
        Queue&, const MatrixView<T, MatrixFormat::Dense>&);                                \
    template Event getri_blocked_dispatch<T>(                                              \
        Queue&, const MatrixView<T, MatrixFormat::Dense>&,                                 \
        const MatrixView<T, MatrixFormat::Dense>&,                                         \
        Span<int64_t>, Span<std::byte>, Span<int32_t>, GetriSolveTrsm<T>);

BATCHLAS_GETRI_INSTANTIATE(float)
BATCHLAS_GETRI_INSTANTIATE(double)
BATCHLAS_GETRI_INSTANTIATE(std::complex<float>)
BATCHLAS_GETRI_INSTANTIATE(std::complex<double>)

#undef BATCHLAS_GETRI_INSTANTIATE

}  // namespace sycl_getri
}  // namespace batchlas
