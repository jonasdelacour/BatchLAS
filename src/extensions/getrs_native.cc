// Native batched GETRS: the row interchange plus two ROUTED trsm solves the
// facade injects. Ships ROUTE-NEUTRAL (preferred() is false at every shape) so a
// vendor-free build has a getrs. This TU must share no device symbol with the
// getrf pair -- hence EXTENSIONS_FACTORIZATION_SOURCES and lu_laswp.hh's tag.
// evidence: docs/perf/lu.md#getrs-composition-window-evidence

#include "getrs_native.hh"
#include "lu_laswp.hh"

#include "../queue.hh"
#include "../util/template-instantiations.hh"

#include <batchlas/util/mempool.hh>

#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace batchlas {
namespace sycl_getrs {

namespace {

// Per-TU tag: this cluster's own instantiation of the shared LASWP kernel.
struct GetrsLaswpTag {};

template <typename T> class GetrsPermGatherKernel;

// Collapsed permutation: apply the transposition list once to an identity index
// array, after which dst[i] = src[idxs[i]] is coalesced on both sides. It stages
// in LOCAL memory and writes back to B's own addresses, so
// getrs_blocked_buffer_size stays 0 at every shape and width.
// evidence: docs/perf/lu.md#getrs-collapsed-permutation

constexpr std::size_t kGetrsPermTileCap = 24576;

// The 48 KB launch hole. evidence: docs/perf/lu.md#the-48-kb-launch-hole
constexpr std::size_t kGetrsPermHoleLo = 47104;
constexpr std::size_t kGetrsPermHoleHi = 49664;
constexpr std::size_t kGetrsPermHolePadTo = 49920;

constexpr std::size_t getrs_perm_hole_padded(std::size_t bytes) {
    return (bytes > kGetrsPermHoleLo && bytes <= kGetrsPermHoleHi) ? kGetrsPermHolePadTo
                                                                  : bytes;
}

template <typename T>
bool getrs_perm_gather_fits(int n, std::size_t slm_budget) {
    if (n <= 0) return false;
    const std::size_t int_bytes = 2u * static_cast<std::size_t>(n) * sizeof(int);
    if (slm_budget <= int_bytes) return false;
    const std::size_t col_bytes =
        static_cast<std::size_t>(n | 1) * sizeof(typename sycl_device::DevMap<T>::type);
    return (slm_budget - int_bytes) >= col_bytes;
}

template <typename T>
bool getrs_perm_gather_launch(Queue& ctx,
                              T* base, int ld, int stride, int nrhs, int batch,
                              const int* piv, int piv_stride, int n,
                              bool forward,
                              std::size_t slm_budget, int max_wg) {
    if (nrhs <= 0 || batch <= 0 || n <= 0) return true;

    using DM = sycl_device::DevMap<T>;
    using D = typename DM::type;
    static_assert(sizeof(D) == sizeof(T), "device scalar must be layout-compatible");

    // ODD ldt: the permuted read is random in the row index, so an even ldt
    // would put a whole column in one bank.
    const int ldt = n | 1;
    const std::size_t int_bytes = 2u * static_cast<std::size_t>(n) * sizeof(int);
    if (!getrs_perm_gather_fits<T>(n, slm_budget)) return false;

    const std::size_t col_bytes = static_cast<std::size_t>(ldt) * sizeof(D);
    std::size_t data_budget = slm_budget - int_bytes;
    if (data_budget > kGetrsPermTileCap) data_budget = kGetrsPermTileCap;
    std::size_t cs = data_budget / col_bytes;
    if (cs == 0) {
        cs = (slm_budget - int_bytes) / col_bytes;
        if (cs == 0) return false;
    }
    if (cs > static_cast<std::size_t>(nrhs)) cs = static_cast<std::size_t>(nrhs);
    const int Cs = static_cast<int>(cs);

    std::size_t tile_elems = static_cast<std::size_t>(Cs) * static_cast<std::size_t>(ldt);
    const std::size_t raw = int_bytes + tile_elems * sizeof(D);
    const std::size_t padded = getrs_perm_hole_padded(raw);
    if (padded > raw && padded <= slm_budget) {
        tile_elems = (padded - int_bytes + sizeof(D) - 1) / sizeof(D);
    }

    int wg = (max_wg < 256) ? max_wg : 256;
    if (wg < 32) wg = 32;

    D* const bp = reinterpret_cast<D*>(base);

    ctx->submit([&](sycl::handler& h) {
        sycl::local_accessor<int, 1> ints(
            sycl::range<1>(2u * static_cast<std::size_t>(n)), h);
        sycl::local_accessor<D, 1> tile(sycl::range<1>(tile_elems), h);

        h.parallel_for<GetrsPermGatherKernel<T>>(
            sycl::nd_range<1>(sycl::range<1>(static_cast<std::size_t>(batch) *
                                             static_cast<std::size_t>(wg)),
                              sycl::range<1>(static_cast<std::size_t>(wg))),
            [=](sycl::nd_item<1> it) {
                const auto grp = it.get_group();
                const int b = static_cast<int>(it.get_group(0));
                const int lid = static_cast<int>(it.get_local_id(0));

                int* const idxs = &ints[0];
                int* const ips = &ints[static_cast<std::size_t>(n)];

                D* const Bb = bp + static_cast<std::ptrdiff_t>(b) * stride;
                const int* const ip = piv + static_cast<std::ptrdiff_t>(b) * piv_stride;

                for (int i = lid; i < n; i += wg) {
                    int p = ip[i] - 1;          // GLOBAL 1-BASED on the wire
                    // ?GETRF guarantees p in [i, n); clamped anyway, since a
                    // bad value corrupts the index array for the WHOLE item.
                    if (p < 0 || p >= n) p = i;
                    ips[i] = p;
                    idxs[i] = i;
                }
                sycl::group_barrier(grp);

                if (lid == 0) {
                    if (forward) {
                        for (int k = 0; k < n; ++k) {
                            const int p = ips[k];
                            if (p != k) { const int t = idxs[k]; idxs[k] = idxs[p]; idxs[p] = t; }
                        }
                    } else {
                        // REVERSE ORDER: forwards, this list builds P, not
                        // P^T; each transposition is its own inverse.
                        for (int k = n - 1; k >= 0; --k) {
                            const int p = ips[k];
                            if (p != k) { const int t = idxs[k]; idxs[k] = idxs[p]; idxs[p] = t; }
                        }
                    }
                }
                sycl::group_barrier(grp);

                for (int cb = 0; cb < nrhs; cb += Cs) {
                    const int cw = ((nrhs - cb) < Cs) ? (nrhs - cb) : Cs;

                    int col = lid / n;
                    int row = lid - col * n;
                    while (col < cw) {
                        tile[static_cast<std::size_t>(col) * ldt + row] =
                            Bb[static_cast<std::ptrdiff_t>(cb + col) * ld + row];
                        row += wg;
                        while (row >= n) { row -= n; ++col; }
                    }
                    sycl::group_barrier(grp);

                    col = lid / n;
                    row = lid - col * n;
                    while (col < cw) {
                        Bb[static_cast<std::ptrdiff_t>(cb + col) * ld + row] =
                            tile[static_cast<std::size_t>(col) * ldt + idxs[row]];
                        row += wg;
                        while (row >= n) { row -= n; ++col; }
                    }
                    // The write-back must finish before the next chunk overwrites the tile.
                    sycl::group_barrier(grp);
                }
            });
    });
    return true;
}

// BATCHLAS_GETRS_LASWP=walk|gather: the only way to reach the walk once the
// gather is the default.
enum class PermSpelling { kDefault, kWalk, kGather };

// Deliberately NOT latched: once a presence check latches false, a later setenv
// is invisible and the test silently runs the default arm and passes.
PermSpelling perm_spelling() {
    const char* const s = std::getenv("BATCHLAS_GETRS_LASWP");
    if (s == nullptr) return PermSpelling::kDefault;
    if (std::strcmp(s, "walk") == 0) return PermSpelling::kWalk;
    if (std::strcmp(s, "gather") == 0) return PermSpelling::kGather;
    return PermSpelling::kDefault;
}

bool getrs_perm_use_gather(PermSpelling sp, int nrhs) {
    if (sp == PermSpelling::kWalk) return false;
    if (sp == PermSpelling::kGather) return true;
    return nrhs >= kGetrsPermGatherMinNrhs;
}

}  // namespace

// Which spelling this call resolves, through the SAME functions the driver uses.
// Tests only: on capacity the gather silently falls back to the walk. 1 = gather.
template <typename T>
int getrs_perm_spelling_debug(Queue& ctx, int n, int nrhs) {
    if (!getrs_perm_use_gather(perm_spelling(), nrhs)) return 0;
    const auto dev = ctx.device();
    if (dev.type != DeviceType::GPU) return 0;
    const std::size_t lm = dev.get_property(DeviceProperty::LOCAL_MEM_SIZE);
    const std::size_t budget = (lm > 4096) ? (lm - 4096) : 0;
    return getrs_perm_gather_fits<T>(n, budget) ? 1 : 0;
}

// Capability flag, true for all four types; preferred() is false at every shape,
// so a vendor-present build still takes cublas?getrsBatched.
template <> bool getrs_blocked_available<float>()                { return true; }
template <> bool getrs_blocked_available<double>()               { return true; }
template <> bool getrs_blocked_available<std::complex<float>>()  { return true; }
template <> bool getrs_blocked_available<std::complex<double>>() { return true; }

// Workspace: ZERO in every mode -- the interchange is in place and the routed
// trsm takes none. It dereferences nothing: A and B arrive null when measuring.
template <typename T>
std::size_t getrs_blocked_buffer_size(Queue&,
                                      const MatrixView<T, MatrixFormat::Dense>&,
                                      const MatrixView<T, MatrixFormat::Dense>&,
                                      Transpose) {
    return 0;
}

// ?GETRF returns ipiv such that F A = L U, with F applied FORWARDS. NoTrans
// permutes B, then solves L (unit lower) and U. Trans/ConjTrans solve U then L
// and apply the SAME list REVERSED to the output; either half wrong is a
// silently wrong Trans answer that no NoTrans test can see.
template <typename T>
Event getrs_blocked_dispatch(Queue& ctx,
                             const MatrixView<T, MatrixFormat::Dense>& A,
                             const MatrixView<T, MatrixFormat::Dense>& B,
                             Transpose transA,
                             Span<int64_t> pivots,
                             Span<std::byte> workspace,
                             GetrsSolveTrsm<T> solve_trsm) {
    static_cast<void>(workspace);   // this arm needs none

    const int n = static_cast<int>(A.rows());
    const int nrhs = static_cast<int>(B.cols());
    const int batch = static_cast<int>(A.batch_size());

    // supports()'s gates, re-applied: this entry point is reachable WITHOUT the
    // table, and an unsupported forced route falls through to automatic().
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
        // Enumerated, never MAX_SUB_GROUP_SIZE >= 32: that property returns
        // sub_group_sizes()[0], so it would ACCEPT a {64}-only device.
        throw std::runtime_error(
            "getrs_blocked: device does not offer sub-group size 32");
    }
    if (pivots.size() < static_cast<std::size_t>(n) * static_cast<std::size_t>(batch)) {
        throw std::invalid_argument("getrs_blocked: pivot span is shorter than n * batch");
    }
    if (!solve_trsm) {
        throw std::invalid_argument(
            "getrs_blocked: the solve seam is empty. Inject the ROUTED batchlas::trsm "
            "(the facade does; a direct caller must too) -- this driver deliberately has "
            "no native fallback, so that the router, and not this file, chooses the trsm "
            "arm.");
    }

    // PACKED 1-BASED int32, the format the vendor paths read and a native getrf
    // writes; native and vendor arms mix freely and must agree bit for bit.
    auto piv_i32 = pivots.as_span<int>();

    const std::size_t local_mem_all = dev.get_property(DeviceProperty::LOCAL_MEM_SIZE);
    const std::size_t slm_budget = (local_mem_all > 4096) ? (local_mem_all - 4096) : 0;
    const int max_wg = static_cast<int>(dev.get_property(DeviceProperty::MAX_WORK_GROUP_SIZE));
    const bool want_gather = getrs_perm_use_gather(perm_spelling(), nrhs);

    auto apply_perm = [&](bool forward) -> Event {
        if (want_gather &&
            getrs_perm_gather_launch<T>(ctx, B.data_ptr(), B.ld(), B.stride(), nrhs,
                                        batch, piv_i32.data(), /*piv_stride=*/n, n,
                                        forward, slm_budget, max_wg)) {
            return ctx.get_event();
        }
        // FALLBACK, not a throw: the gather refuses when a column of B plus the
        // two int arrays will not fit local memory; the walk is identical.
        return lu_native::lu_laswp_launch<GetrsLaswpTag, T>(
            ctx, B.data_ptr(), B.ld(), B.stride(), nrhs, batch,
            piv_i32.data(), /*piv_stride=*/n, /*k0=*/0, /*k1=*/n, forward);
    };

    if (transA == Transpose::NoTrans) {
        (void)apply_perm(/*forward=*/true);
        // An out-of-order queue does not give this ordering for free.
        if (!ctx.in_order()) ctx.wait();

        (void)solve_trsm(ctx, A, B, T(1), Side::Left, Uplo::Lower,
                         Transpose::NoTrans, Diag::Unit);
        if (!ctx.in_order()) ctx.wait();
        return solve_trsm(ctx, A, B, T(1), Side::Left, Uplo::Upper,
                          Transpose::NoTrans, Diag::NonUnit);
    }

    (void)solve_trsm(ctx, A, B, T(1), Side::Left, Uplo::Upper, transA, Diag::NonUnit);
    if (!ctx.in_order()) ctx.wait();
    (void)solve_trsm(ctx, A, B, T(1), Side::Left, Uplo::Lower, transA, Diag::Unit);
    if (!ctx.in_order()) ctx.wait();

    return apply_perm(/*forward=*/false);
}

#define BATCHLAS_GETRS_INSTANTIATE(T)                                                      \
    template std::size_t getrs_blocked_buffer_size<T>(                                     \
        Queue&, const MatrixView<T, MatrixFormat::Dense>&,                                 \
        const MatrixView<T, MatrixFormat::Dense>&, Transpose);                             \
    template Event getrs_blocked_dispatch<T>(                                              \
        Queue&, const MatrixView<T, MatrixFormat::Dense>&,                                 \
        const MatrixView<T, MatrixFormat::Dense>&, Transpose,                              \
        Span<int64_t>, Span<std::byte>, GetrsSolveTrsm<T>);                                \
    template int getrs_perm_spelling_debug<T>(Queue&, int, int);

BATCHLAS_GETRS_INSTANTIATE(float)
BATCHLAS_GETRS_INSTANTIATE(double)
BATCHLAS_GETRS_INSTANTIATE(std::complex<float>)
BATCHLAS_GETRS_INSTANTIATE(std::complex<double>)

#undef BATCHLAS_GETRS_INSTANTIATE

}  // namespace sycl_getrs
}  // namespace batchlas
