// Native batched GEQRF -- the BLOCKED tier: a right-looking blocked Householder QR.
// Per panel, with ib = min(nb, k - j0), A22 = A(j0:m, j0+ib:n) and V = (m-j0) x ib:
// geqr2 leaf, pack V, larft T, then W1 = V^H A22; W2 = T^H W1; A22 -= V W2.
// The update is (I - V T^H V^H), not (I - V T V^H): the factorisation applies
// Q_block^H. Identical for real scalars, so the wrong one only breaks complex.
// Design and evidence: docs/perf/qr.md

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

// Kernel-name tag for this TU's copies of the WY kernels; ormqr_blocked.cc has its own.
struct GeqrfWyTag;

namespace sycl_geqrf {

namespace {

// ConjTrans for complex, Trans for real: identical on a real scalar, but the native
// GEMM gates its float transposed-register kernel on Transpose::Trans.
template <typename T>
inline constexpr Transpose kConjT =
    batchlas::internal::is_complex<T>::value ? Transpose::ConjTrans : Transpose::Trans;

// Block width: 16 for double, 32 otherwise, clamped to k. One pure function so the
// driver, the size query and debug_params cannot disagree. Keep it a multiple of 16
// and never below 32 for complex: the complex wide-scalar GEMM gates on min_dim >= 32
// and min_dim of the trailing NN update IS the block width. docs/perf/qr.md#block-width-evidence
template <typename T>
constexpr int geqrf_nb_for_type() {
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

// Described once here and replayed by both the query and the call. The total must be
// monotone non-decreasing in (rows, cols, batch) and read no element of A or tau:
// band_reduction.cc sizes against a null view, then calls geqrf with a sub-view of it.
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

    // W1/W2 are sized on the widest trailing block, n - nb, which is also their stride.
    const std::size_t n2max = static_cast<std::size_t>(std::max(0, n - nb));

    ws.v = pool.allocate<T>(ctx, static_cast<std::size_t>(m) *
                                     static_cast<std::size_t>(nb) * b);
    ws.t = pool.allocate<T>(ctx, static_cast<std::size_t>(nb) *
                                     static_cast<std::size_t>(nb) * b);
    ws.w1 = pool.allocate<T>(ctx, std::max<std::size_t>(
                                      1, static_cast<std::size_t>(nb) * n2max * b));
    ws.w2 = pool.allocate<T>(ctx, std::max<std::size_t>(
                                      1, static_cast<std::size_t>(nb) * n2max * b));
    return ws;
}

}  // namespace

// Co-located with the driver so "the flag is true" and "this TU is compiled" are one
// fact. It moves no vendor-present traffic: RouteTable<Op::geqrf,T>::preferred() is
// still false everywhere. evidence: docs/perf/qr.md#route-arms
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

    const auto dev = ctx.device();
    const std::size_t local_mem = dev.get_property(DeviceProperty::LOCAL_MEM_SIZE);
    const std::size_t budget = (local_mem > 4096) ? (local_mem - 4096) : 0;
    const int ib0 = std::min(nb, std::min(m, n));
    const unsigned leaf = geqrf_cta_fits<T>(m, ib0, budget) ? 1u : 2u;

    return (leaf << 16) | static_cast<unsigned>(nb);
}

template <typename T>
Event geqrf_blocked_dispatch(Queue& ctx,
                             const MatrixView<T, MatrixFormat::Dense>& A,
                             Span<T> tau,
                             Span<std::byte> workspace,
                             GeqrfTrailingGemm<T> trailing_gemm) {
    // Default the seam to the native kernel so this TU stands alone; the facade injects
    // the ROUTED gemm. Calling gemm_custom here unconditionally bypasses the route table.
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

    // Re-applies every gate RouteTable<Op::geqrf,T>::supports() applies: this entry point
    // is reachable without the table, and route resolution falls back to automatic() on an
    // unsupported forced route, so a wrong gate here silently measures the vendor.
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
    const int n2max = std::max(0, n - nb);

    BumpAllocator pool(workspace);
    auto ws = geqrf_blocked_layout<T>(ctx, pool, m, n, nb, batch);

    const int ld = A.ld();
    const int stride = A.stride();
    T* const a_ptr = A.data_ptr();
    T* const tau_ptr = tau.data();

    // Explicit 6-argument construction, never operator()(Slice, Slice): a slice propagates
    // the parent's pointer array, and stride defaults to ld*cols when 0 is passed, so an
    // ib-column sub-view silently gets stride = ld*ib. `ptrs = nullptr` so a backend
    // regenerates the array for this view.
    auto sub = [&](int r0, int nr, int c0, int nc) {
        return MatrixView<T, MatrixFormat::Dense>(
            a_ptr + static_cast<std::ptrdiff_t>(c0) * ld + r0,
            nr, nc, ld, stride, batch, nullptr);
    };

    for (int j0 = 0; j0 < k; j0 += nb) {
        // The short final panel: every extent below derives from ib and n2, never nb.
        const int ib = std::min(nb, k - j0);
        const int j2 = j0 + ib;
        const int mp = m - j0;          // panel / V height, and A22's height
        const int n2 = n - j2;          // trailing columns; ZERO on the last panel

        // tau's batch stride is k, the whole matrix's reflector count (geqrf's contract),
        // at offset j0; the panel's own min(mp, ib) would scatter tau for every item but one.
        (void)geqrf_panel_factorize<T>(ctx,
                                       a_ptr + static_cast<std::ptrdiff_t>(j0) * ld + j0,
                                       ld, stride, mp, ib, batch,
                                       tau_ptr, k, j0, nullptr);

        if (n2 <= 0) break;

        // Each dependent boundary guards itself; a caller may pass an out-of-order queue.
        if (!ctx.in_order()) ctx.wait();

        // V is packed contiguously at ld = mp, not the parent ld = m: at ld = m both
        // trailing GEMMs get a short operand with a long stride, the shape the native GEMM
        // collapses on (docs/perf/gemm.md#the-strided-ld-defect-and-the-routing-fix).
        // mp*ib <= m*nb always fits ws.v; pack, larft and both GEMM views must agree here.
        const int ld_v = mp;
        const int stride_v = mp * ib;

        MatrixView<T, MatrixFormat::Dense> Vblk(ws.v.data(), mp, ib, ld_v,
                                                stride_v, batch);
        (void)wy::pack_v_panel_batched<GeqrfWyTag, T>(
            ctx, ws.v.data(), ld_v, stride_v, A, j0, ib, m);

        if (!ctx.in_order()) ctx.wait();

        MatrixView<T, MatrixFormat::Dense> Tblk(ws.t.data(), ib, ib, nb,
                                                nb * nb, batch);
        // The compile-time `false` form: a runtime literal also instantiates the
        // device-BLAS larft for GeqrfWyTag, 32 entry functions that can never launch.
        (void)wy::larft_forward_columnwise_batched_t<GeqrfWyTag, T, false>(
            ctx, ws.t.data(), nb, nb * nb,
            ws.v.data(), ld_v, stride_v,
            mp, ib,
            tau_ptr, /*tau_stride=*/k, /*tau_offset=*/j0, batch);

        if (!ctx.in_order()) ctx.wait();

        const auto A22 = sub(j0, mp, j2, n2);
        // Batch stride is the layout's nb * n2max, not this panel's nb * n2: the GEMMs agree
        // either way, so a wrong stride is silent, but only n2max matches what was reserved.
        MatrixView<T, MatrixFormat::Dense> W1(ws.w1.data(), ib, n2, nb,
                                              nb * n2max, batch);
        MatrixView<T, MatrixFormat::Dense> W2(ws.w2.data(), ib, n2, nb,
                                              nb * n2max, batch);

        trailing_gemm(ctx, Vblk, A22, W1, T(1), T(0), kConjT<T>, Transpose::NoTrans,
                      ComputePrecision::Default);
        if (!ctx.in_order()) ctx.wait();

        trailing_gemm(ctx, Tblk, W1, W2, T(1), T(0), kConjT<T>, Transpose::NoTrans,
                      ComputePrecision::Default);
        if (!ctx.in_order()) ctx.wait();

        trailing_gemm(ctx, Vblk, W2, A22, T(-1), T(1), Transpose::NoTrans,
                      Transpose::NoTrans, ComputePrecision::Default);

        // The next pack_v OVERWRITES the V this update is still reading.
        if (!ctx.in_order()) ctx.wait();
    }

    return ctx.get_event();
}

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
