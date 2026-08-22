// Native batched ORGQR -- the blocked tier, which is ORMQR APPLIED TO AN
// IDENTITY.
//
//     Q = H_1 H_2 ... H_k I_{m x n} = ormqr(A, I, Side::Left, Transpose::NoTrans)
//
// and that is the whole algorithm. WP5's baseline settled it by measurement
// rather than by argument: the Q this produces is ELEMENTWISE identical to
// cuSOLVER's orgqr to 1.4e-15 (double/cdouble) and 6.9e-07 (float/cfloat) across
// all 24 (type, n) cells, with independent ||Q^H Q - I|| and ||QR - A|| probes on
// both. Do not specialise before a cell measures it necessary: the ENTIRE
// theoretical prize is the 1.5x flop ratio of applying Q to I (2n^3 against
// 4n^3/3), against a 2.3-111x margin over the vendor across most of the range.
//
// THE VENDOR THIS REPLACES IS A PER-ITEM LOOP, not a batched kernel:
// cublas.cc:1413-1420 opens an out-of-order sub-queue and calls
// cusolverDnXorgqr once per batch member, and cublas.cc:1447 sizes its workspace
// as single_ws * batch -- 1164 MB for float n=64 b=8192 and 4644 MB for cdouble
// at the same shape, against 104 MB / 416 MB for routed ormqr-on-an-identity. So
// a native orgqr closes a MEMORY HAZARD as well as a speed gap, and any win
// reported from it should be phrased as "beats the per-item loop", never as
// "beats cuSOLVER".
//
// THIS FILE IS IN EXTENSIONS_FACTORIZATION_SOURCES, NOT EXTENSIONS_CTA_SOURCES.
// The grouping rule is the device-code cluster, not the topic
// (src/extensions/CMakeLists.txt:70-85): this driver shares device symbols with
// neither geqrf TU -- its own device code is an identity fill and a copy-back --
// and it is built on ormqr_blocked.cc, which lives here.
//
// ---------------------------------------------------------------------------
// TWO DEPARTURES FROM THE WP5 SCAFFOLDING, both recorded rather than papered over
// ---------------------------------------------------------------------------
// 1. orgqr_blocked_buffer_size NOW TAKES tau AND THE INJECTED SIZE QUERY.
//    The scaffolding declared it as (Queue&, A) and asked that the apply's
//    workspace "be computed from ONE pure function the driver also calls". With
//    a two-argument signature it cannot be: the apply goes through the ROUTED
//    ormqr, whose workspace depends on a resolution only the facade can perform,
//    and hand-rolling the native formula here would be exactly the drift the
//    request exists to prevent. So the facade injects the routed
//    ormqr_buffer_size into BOTH the query and the call, from the same
//    resolution, and the two agree by construction rather than because getenv
//    returned the same thing twice -- which is the ormqr_buffer_size_dispatch
//    anti-pattern (ormqr.hh:281-303) the scaffolding names.
//
// 2. THE APPLY IS REQUIRED, NOT DEFAULTED. The scaffolding says an empty
//    OrgqrApplyQ means "use ormqr_blocked directly". It cannot: ormqr_blocked is
//    template <Backend B, typename T> (internal/ormqr_blocked.hh:23-39) and this
//    driver is instantiated per scalar type with NO Backend, which is the whole
//    reason the seam exists. Defaulting it would mean naming a Backend here --
//    hardcoding Backend::CUDA in a file that also has to build for ROCm. An
//    empty function therefore THROWS, naming the requirement. A direct caller
//    (a test) injects `ormqr<Backend::CUDA, T>` itself, which is still a call no
//    vendor orgqr can serve, so the pinned-route argument
//    (tests/potrf_tests.cc:6-25) is intact.

#include "orgqr_native.hh"

#include "../queue.hh"
#include "../util/template-instantiations.hh"

#include <batchlas/util/mempool.hh>

#include <sycl/sycl.hpp>

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace batchlas {
namespace sycl_orgqr {

namespace {

template <typename T> class OrgqrIdentityKernel;
template <typename T> class OrgqrCopyBackKernel;

// The WY block width handed to the apply.
//
// MEASURED (experiments/wp5_qr/baseline/summary_nb.txt), and deliberately NOT
// tuning::ormqr_block_size_for_n, which is what resolve_ormqr_block_size returns
// when no hint is given (ormqr.hh:220-228). That ladder's 16/16/24/48/56 by
// A.rows() was tuned on CUDA/float ONLY -- evaluation/tuning/tune.py:494 takes a
// single --type per run and the ormqr_blocked space has no type axis -- and even
// in a VENDOR-PRESENT build the shipped width costs double 1.32-1.41x and
// cdouble 1.46-1.47x, while float at n=256 is exactly 1.00x, the one cell it was
// tuned at.
//
// Multiple of 16 (G1's m IS the block width and G1 is Tiled16 for every type),
// never below 32 for complex (gemm_kernels.cc:700's min_dim >= 32 wide-scalar
// gate), 16 for double. resolve_ormqr_block_size clamps a hint to k, so a short
// matrix degrades gracefully without this function having to.
template <typename T>
constexpr int32_t orgqr_nb_for_type() {
    if constexpr (std::is_same_v<T, double>) {
        return 16;
    } else {
        return 32;
    }
}

template <typename T>
inline int32_t orgqr_nb(int m, int n) {
    const int32_t k = static_cast<int32_t>(std::min(m, n));
    return std::max<int32_t>(1, std::min<int32_t>(orgqr_nb_for_type<T>(), std::max(1, k)));
}

// The view the apply writes Q into. Built over a caller-supplied pointer so the
// SIZE query and the CALL describe the SAME C -- the query passes nullptr, which
// is safe because a workspace query may read a view's metadata but never its
// data (the rule band_reduction.cc:1041-1044 forces on geqrf, kept here because
// the two now share a code path in the facade).
template <typename T>
inline MatrixView<T, MatrixFormat::Dense> orgqr_c_view(T* p, int m, int n, int batch) {
    return MatrixView<T, MatrixFormat::Dense>(p, m, n, /*ld=*/m,
                                              /*stride=*/static_cast<int>(
                                                  static_cast<std::size_t>(m) *
                                                  static_cast<std::size_t>(n)),
                                              batch, nullptr);
}

// The workspace, described ONCE and replayed by both the query and the call
// (mempool.hh:165-190). `apply_bytes` is computed OUTSIDE this function, against
// the CALLER's views, because mempool.hh:179-184 forbids asking a nested size
// query about workspace-derived views.
template <typename T>
struct OrgqrWs {
    Span<T> c;
    Span<std::byte> apply;
};

template <typename T>
OrgqrWs<T> orgqr_blocked_layout(Queue& ctx, BumpAllocator& pool,
                                int m, int n, int batch, std::size_t apply_bytes) {
    OrgqrWs<T> ws;
    ws.c = pool.allocate<T>(ctx, static_cast<std::size_t>(m) *
                                     static_cast<std::size_t>(n) *
                                     static_cast<std::size_t>(batch));
    ws.apply = pool.allocate<std::byte>(ctx, apply_bytes);
    return ws;
}

// The apply's workspace requirement, asked exactly once per entry point and fed
// into the layout above. Injected, for the reason at the top of this file: only
// the facade can resolve RouteTable<Op::ormqr> and only the resolved route knows
// its own size.
template <typename T>
std::size_t orgqr_apply_bytes(Queue& ctx,
                              const MatrixView<T, MatrixFormat::Dense>& A,
                              Span<T> tau,
                              int m, int n, int batch,
                              const OrgqrApplyQBufferSize<T>& q) {
    if (!q) {
        throw std::logic_error(
            "sycl_orgqr: the apply-Q workspace query was not injected. orgqr's native arm "
            "is ormqr applied to an identity, and only the facade can name the ROUTED "
            "ormqr_buffer_size -- this driver is instantiated per scalar type with no "
            "Backend parameter (see the note at the top of orgqr_blocked.cc).");
    }
    // C is described by metadata alone; nullptr is deliberate. See orgqr_c_view.
    const auto C = orgqr_c_view<T>(nullptr, m, n, batch);
    return q(ctx, A, C, Side::Left, Transpose::NoTrans, tau, orgqr_nb<T>(m, n));
}

}  // namespace

// ---------------------------------------------------------------------------
// THE CAPABILITY FLAG. DEFINED HERE, beside the driver (potrf_native.hh:81-92).
//
// TRUE for all four types as of WP5. It moves no vendor-present traffic:
// RouteTable<Op::orgqr,T>::preferred() is still false, so only a vendor-free
// build or an explicit BATCHLAS_ORGQR_ROUTE reaches this driver.
//
// It is NOT "is ormqr_blocked compiled" -- that was already true, and answering
// this question with it would make the route table hand a vendor-free caller a
// route the facade could not service.
// ---------------------------------------------------------------------------
template <> bool orgqr_blocked_available<float>()                { return true; }
template <> bool orgqr_blocked_available<double>()               { return true; }
template <> bool orgqr_blocked_available<std::complex<float>>()  { return true; }
template <> bool orgqr_blocked_available<std::complex<double>>() { return true; }

template <typename T>
std::size_t orgqr_blocked_buffer_size(Queue& ctx,
                                      const MatrixView<T, MatrixFormat::Dense>& A,
                                      Span<T> tau,
                                      OrgqrApplyQBufferSize<T> apply_q_buffer_size) {
    const int m = static_cast<int>(A.rows());
    const int n = static_cast<int>(A.cols());
    const int batch = static_cast<int>(A.batch_size());
    if (m < 1 || n < 1 || batch < 1) return 0;

    const std::size_t apply_bytes =
        orgqr_apply_bytes<T>(ctx, A, tau, m, n, batch, apply_q_buffer_size);

    return workspace_bytes([&](BumpAllocator& pool) {
        return orgqr_blocked_layout<T>(ctx, pool, m, n, batch, apply_bytes);
    });
}

template <typename T>
int orgqr_blocked_debug_block_size(Queue& ctx, int m, int n) {
    static_cast<void>(ctx);
    if (m < 1 || n < 1) return 0;
    return static_cast<int>(orgqr_nb<T>(m, n));
}

template <typename T>
Event orgqr_blocked_dispatch(Queue& ctx,
                             const MatrixView<T, MatrixFormat::Dense>& A,
                             Span<T> tau,
                             Span<std::byte> workspace,
                             OrgqrApplyQ<T> apply_q,
                             OrgqrApplyQBufferSize<T> apply_q_buffer_size) {
    const int m = static_cast<int>(A.rows());
    const int n = static_cast<int>(A.cols());
    const int batch = static_cast<int>(A.batch_size());
    const int k = std::min(m, n);

    // Every gate RouteTable<Op::orgqr,T>::supports() applies, re-applied because
    // this entry point is reachable without the table -- route_resolve.hh:101
    // falls through to automatic() when a forced route is unsupported, so a
    // pinned-route test that is wrong about one gate silently measures the
    // vendor.
    if (m < 1 || n < 1 || batch < 1) {
        throw std::invalid_argument("orgqr_blocked: degenerate extents");
    }
    if (m < n) {
        // Q's columns live in R^m, so n > m has no meaning; supports() refuses it
        // and the two must agree or a forced route reaches a shape the table
        // promised the vendor.
        throw std::invalid_argument(
            "orgqr_blocked: n > m is not supported (route_orgqr.hh's supports() refuses it)");
    }
    if (A.is_heterogeneous()) {
        throw std::invalid_argument("orgqr_blocked: heterogeneous batch is not supported");
    }
    if (ctx.device().type != DeviceType::GPU) {
        throw std::invalid_argument("orgqr_blocked: GPU queues only");
    }
    if (tau.size() < static_cast<std::size_t>(k) * static_cast<std::size_t>(batch)) {
        throw std::invalid_argument("orgqr_blocked: tau span is shorter than k * batch");
    }
    if (!apply_q) {
        throw std::logic_error(
            "sycl_orgqr::orgqr_blocked_dispatch: the apply-Q seam was not injected. orgqr's "
            "native arm is ormqr applied to an identity, and only a layer that can name a "
            "Backend can reach the ROUTED ormqr (see the note at the top of "
            "orgqr_blocked.cc).");
    }

    // ONE resolution of the block width and ONE of the apply's size, both shared
    // with the query above through the same pure functions.
    const int32_t nb = orgqr_nb<T>(m, n);
    const std::size_t apply_bytes =
        orgqr_apply_bytes<T>(ctx, A, tau, m, n, batch, apply_q_buffer_size);

    BumpAllocator pool(workspace);
    auto ws = orgqr_blocked_layout<T>(ctx, pool, m, n, batch, apply_bytes);

    const auto C = orgqr_c_view<T>(ws.c.data(), m, n, batch);

    // (1) C := I_{m x n}. Parallel over (batch, rows, cols) -- NOT over batch
    // alone, which is this repository's recurring performance defect
    // (gebrd.cc:45 is it in its purest form).
    {
        T* const cp = ws.c.data();
        const std::size_t stride_c =
            static_cast<std::size_t>(m) * static_cast<std::size_t>(n);
        // DIM 2 IS THE ROW, NOT THE COLUMN, and that is the whole performance
        // content of this launch. sycl::id<3> makes dim 2 the fastest-varying
        // index, C is COLUMN-MAJOR with leading dimension m, so putting the
        // column there made lanes 0..31 of a warp write addresses m*sizeof(T)
        // apart -- 32 sectors per warp instead of 4. Measured before the swap:
        // 3.459 ms for a 537 MB pure write (float m=n=1024, batch=128) = 155
        // GB/s on a card that does ~800-900 GB/s coalesced. The repository's own
        // fast kernels already use this convention (src/matrix.cc:400 launches
        // range<3>(batch, cols, wg) and walks ROWS in dim 2;
        // src/sycl/gemm/register_128x128.hh:127 the same).
        ctx->parallel_for<OrgqrIdentityKernel<T>>(
            sycl::range<3>(static_cast<std::size_t>(batch), static_cast<std::size_t>(n),
                           static_cast<std::size_t>(m)),
            [=](sycl::id<3> idx) {
                const std::size_t b = idx[0];
                const int c = static_cast<int>(idx[1]);
                const int r = static_cast<int>(idx[2]);
                cp[b * stride_c + static_cast<std::size_t>(r) +
                   static_cast<std::size_t>(c) * static_cast<std::size_t>(m)] =
                    (r == c) ? T(1) : T(0);
            });
    }

    // The apply READS the identity this fill just wrote. In-order queues order it
    // for free; a caller may construct an out-of-order one
    // (sycl-device-queue.hh:254).
    if (!ctx.in_order()) ctx.wait();

    // (2) C := Q C. Through the injected, ROUTED ormqr, so it honours
    // BATCHLAS_ORMQR_ROUTE -- calling ormqr_blocked from here would be WP3 step
    // 16's recorded defect one level up. Argument order is the POSITIONAL entry
    // point's (ormqr.hh:311-320), which is W13's trap: an option struct orders
    // its fields differently.
    (void)apply_q(ctx, A, C, Side::Left, Transpose::NoTrans, tau, ws.apply, nb);

    if (!ctx.in_order()) ctx.wait();

    // (3) A := C. orgqr overwrites its input; ormqr writes into a separate C, so
    // the copy is the price of the reuse and is exactly the m*n*batch scratch the
    // baseline priced against the vendor's single_ws*batch.
    {
        const T* const cp = ws.c.data();
        T* const ap = A.data_ptr();
        const int lda = A.ld();
        const int stride_a = A.stride();
        const std::size_t stride_c =
            static_cast<std::size_t>(m) * static_cast<std::size_t>(n);
        // DIM 2 IS THE ROW -- same reason as the identity fill above. This one
        // is uncoalesced on BOTH sides (it reads C at ld m and writes A at ld
        // lda), and measured 4.674 ms for 1.07 GB of traffic = 230 GB/s.
        ctx->parallel_for<OrgqrCopyBackKernel<T>>(
            sycl::range<3>(static_cast<std::size_t>(batch), static_cast<std::size_t>(n),
                           static_cast<std::size_t>(m)),
            [=](sycl::id<3> idx) {
                const std::size_t b = idx[0];
                const int c = static_cast<int>(idx[1]);
                const int r = static_cast<int>(idx[2]);
                ap[static_cast<std::ptrdiff_t>(b) * stride_a +
                   static_cast<std::ptrdiff_t>(r) +
                   static_cast<std::ptrdiff_t>(c) * lda] =
                    cp[b * stride_c + static_cast<std::size_t>(r) +
                       static_cast<std::size_t>(c) * static_cast<std::size_t>(m)];
            });
    }

    return ctx.get_event();
}

// ---------------------------------------------------------------------------
// Instantiation: PER SCALAR TYPE ONLY, no Backend cross-product. This is the
// point of the injected seams -- ormqr_blocked itself is template <Backend B,
// typename T> (internal/ormqr_blocked.hh:23-39, a 4x4 cross-product) and building
// on it the naive way would inherit that in a build that is device-link-bound.
// ---------------------------------------------------------------------------
#define BATCHLAS_ORGQR_BLOCKED_INSTANTIATE(T)                                                 \
    template std::size_t orgqr_blocked_buffer_size<T>(                                        \
        Queue&, const MatrixView<T, MatrixFormat::Dense>&, Span<T>,                           \
        OrgqrApplyQBufferSize<T>);                                                            \
    template int orgqr_blocked_debug_block_size<T>(Queue&, int, int);                         \
    template Event orgqr_blocked_dispatch<T>(Queue&,                                          \
                                             const MatrixView<T, MatrixFormat::Dense>&,       \
                                             Span<T>, Span<std::byte>, OrgqrApplyQ<T>,        \
                                             OrgqrApplyQBufferSize<T>);

BATCHLAS_ORGQR_BLOCKED_INSTANTIATE(float)
BATCHLAS_ORGQR_BLOCKED_INSTANTIATE(double)
BATCHLAS_ORGQR_BLOCKED_INSTANTIATE(std::complex<float>)
BATCHLAS_ORGQR_BLOCKED_INSTANTIATE(std::complex<double>)

#undef BATCHLAS_ORGQR_BLOCKED_INSTANTIATE

}  // namespace sycl_orgqr
}  // namespace batchlas
