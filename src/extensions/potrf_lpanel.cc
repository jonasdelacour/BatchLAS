// Native batched POTRF: the LPANEL tier's launcher and capability surface; the device code
// is in potrf_lpanel_device.hh. This TU must stay in EXTENSIONS_CTA_SOURCES next to
// potrf_cta.cc and potrf_blocked.cc -- one device-code cluster, because the blocked driver's
// leaf and this kernel are meant to become interchangeable and the tests compare the two.
// evidence: docs/perf/potrf.md#the-lpanel-tier

#include "potrf_native.hh"
#include "potrf_lpanel_device.hh"
#include "potrf_slm_hole.hh"

#include "../queue.hh"
#include "../util/resident_capacity.hh"
#include "../util/template-instantiations.hh"

#include <batchlas/util/mempool.hh>

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>

namespace batchlas {

// Kernel name tag; outside the anonymous namespace so it names no internal-linkage entity.
template <typename T, int NB>
class PotrfLpanelKernel;

namespace sycl_potrf {

namespace {

using potrf_native::potrf_hole_padded;

// The panel width, and the length of the rp[]/rS[]/rA[] register arrays: 3 * NB scalars
// per lane before addressing. evidence: docs/perf/potrf.md#lpanel-register-gate
template <typename T>
struct PotrfLpanelConst { static constexpr int NB = 8; };

// Which NB values this build actually instantiates. A hint the type cannot honour is an
// ERROR rather than a silent downgrade: the NB A/B would otherwise measure NB = 8 twice.
template <typename T>
constexpr bool potrf_lpanel_nb_is_built(int nb) {
    return nb == PotrfLpanelConst<T>::NB || (std::is_same_v<T, float> && nb == 16);
}

template <typename T>
constexpr int potrf_lpanel_nb_for(int hint) {
    return (hint == 0) ? PotrfLpanelConst<T>::NB : hint;
}

// Called by BOTH the capability query and the launcher, so the ceiling supports() advertises
// cannot disagree with what the kernel allocates. sA is the n x NB panel at ld = n -- no odd
// padding, because lane `row` reads sA[row + i*n], already stride 1 across lanes -- sB is the
// NB x NB broadcast block, and the 256 over-covers *fail plus alignment slack.
constexpr std::size_t potrf_lpanel_slm_per_matrix(int n, int nb, std::size_t sz_d) {
    return (static_cast<std::size_t>(n) * static_cast<std::size_t>(nb) +
            static_cast<std::size_t>(nb) * static_cast<std::size_t>(nb)) * sz_d + 256;
}

// Lanes per matrix. `L >= n` is a CORRECTNESS requirement of the body, not a tuning choice:
// lane `row` carries rp/rS across the whole k loop. Rounded to a whole sub-group so a packed
// work-group stays a multiple of 32.
constexpr int potrf_lpanel_lanes(int n) {
    return ((n + 31) / 32) * 32;
}

struct PotrfLpanelLaunch {
    int L = 32;              // lanes per matrix, >= n
    int G = 1;               // matrices per work-group
    int wg_size = 32;
    int num_wg = 0;
    int slda = 1;
    std::size_t slm_per_matrix = 0;
    std::size_t slm_total = 0;   // G * slm_per_matrix, after the hole pad
    bool fits = false;
};

// G > 1 is legal here and NOT in potrf_cta.cc: every barrier in potrf_lpanel_body is a
// WORK-GROUP barrier and the schedule depends only on (n, NB), which the packed matrices
// share, so the extra matrices are synchronised with each other harmlessly rather than
// raced. evidence: docs/perf/potrf.md#why-lpanel-may-pack-on-work-group-barriers
PotrfLpanelLaunch potrf_lpanel_launch_params(int n, int nb, int batch, std::size_t sz_d,
                                             std::size_t slm_budget, int max_wg) {
    PotrfLpanelLaunch p;
    p.slda = n;
    p.L = potrf_lpanel_lanes(n);
    p.slm_per_matrix = potrf_lpanel_slm_per_matrix(n, nb, sz_d);

    p.G = resident::pack_matrices_per_wg(p.slm_per_matrix, p.L, slm_budget, max_wg);
    // The pad is applied to the TOTAL, so a G the byte test passed can still overflow it.
    while (p.G > 1 &&
           potrf_hole_padded(static_cast<std::size_t>(p.G) * p.slm_per_matrix) > slm_budget) {
        p.G >>= 1;
    }

    p.wg_size = p.G * p.L;
    p.num_wg = (batch + p.G - 1) / p.G;
    p.slm_total = potrf_hole_padded(static_cast<std::size_t>(p.G) * p.slm_per_matrix);
    p.fits = (p.L >= n) && (p.slm_total <= slm_budget) && (p.wg_size <= max_wg);
    return p;
}

}  // namespace

template <typename T>
int potrf_lpanel_max_n_for_slm(std::size_t slm_budget_bytes, int max_wg_size,
                               int min_blocks_per_sm, int nb_hint) {
    const int nb = potrf_lpanel_nb_for<T>(nb_hint);
    if (!potrf_lpanel_nb_is_built<T>(nb)) return 0;

    // Two independent caps, and the work-group one is NOT slack: the body requires L >= n.
    const int by_wg = (max_wg_size / 32) * 32;
    const int by_slm = resident::resident_max_n(
        [nb](int n) {
            using DM = sycl_device::DevMap<T>;
            return potrf_hole_padded(
                potrf_lpanel_slm_per_matrix(n, nb, sizeof(typename DM::type)));
        },
        slm_budget_bytes, min_blocks_per_sm);
    return std::min(by_wg, by_slm);
}

namespace {

template <typename T>
Span<int32_t> potrf_lpanel_layout(Queue& ctx, BumpAllocator& pool, int batch) {
    return pool.allocate<int32_t>(ctx, static_cast<std::size_t>(batch));
}

}  // namespace

template <typename T>
std::size_t potrf_lpanel_buffer_size(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A) {
    const int batch = A.batch_size();
    return workspace_bytes([&](BumpAllocator& p) {
        return potrf_lpanel_layout<T>(ctx, p, batch);
    });
}

// The launch geometry, for tests; see potrf_native.hh for the encoding.
template <typename T>
unsigned potrf_lpanel_debug_launch(Queue& ctx, int n, int batch, int min_blocks_per_sm,
                                   int nb_hint) {
    using DM = sycl_device::DevMap<T>;
    const int nb = potrf_lpanel_nb_for<T>(nb_hint);
    if (!potrf_lpanel_nb_is_built<T>(nb)) return 0u;

    const auto dev = ctx.device();
    const std::size_t budget = resident::device_slm_budget(
        dev.get_property(DeviceProperty::LOCAL_MEM_SIZE));
    const int max_wg = static_cast<int>(dev.get_property(DeviceProperty::MAX_WORK_GROUP_SIZE));
    const auto p = potrf_lpanel_launch_params(
        n, nb, batch, sizeof(typename DM::type),
        resident::occupancy_budget(budget, min_blocks_per_sm), max_wg);
    if (!p.fits) return 0u;
    return (static_cast<unsigned>(nb) << 16) | (static_cast<unsigned>(p.L) << 4) |
           static_cast<unsigned>(p.G);
}

namespace {

template <typename T, int NB>
Event potrf_lpanel_launch(Queue& ctx,
                          const MatrixView<T, MatrixFormat::Dense>& A,
                          Span<int32_t> info,
                          const PotrfLpanelLaunch& p,
                          int n, int batch) {
    // std::complex is re-typed to the POD device scalar at the pointer boundary: its
    // Annex-G operator* costs an isnan branch and a __mulsc3 call in device code.
    using DM = sycl_device::DevMap<T>;
    using D = typename DM::type;
    using R = typename DM::real;
    static_assert(sizeof(D) == sizeof(T), "device scalar must be layout-compatible");

    D* a_ptr = reinterpret_cast<D*>(A.data_ptr());
    const int ldg = A.ld();
    const int stride_a = A.stride();
    const int slda = p.slda;
    const int G = p.G;
    const int L = p.L;
    const int wg_size = p.wg_size;
    const int num_wg = p.num_wg;
    int32_t* info_ptr = info.data();

    // Padded into the panel accessor, as potrf_cta.cc does. `natural` counts what the THREE
    // accessors below actually request, not slm_per_matrix's rounded-up figure: the pad has
    // to land the real request outside the hole, and the two differ by the 256 of slack.
    const std::size_t panel_used = static_cast<std::size_t>(G) *
                                   static_cast<std::size_t>(slda) * NB;
    const std::size_t natural = panel_used * sizeof(D)
                              + static_cast<std::size_t>(G) * NB * NB * sizeof(D)
                              + static_cast<std::size_t>(G) * sizeof(int);
    const std::size_t pad_bytes = (p.slm_total > natural) ? (p.slm_total - natural) : 0;
    const std::size_t panel_elems = panel_used + (pad_bytes + sizeof(D) - 1) / sizeof(D);

    ctx->submit([&](sycl::handler& h) {
        sycl::local_accessor<D, 1> panel(sycl::range<1>(panel_elems), h);
        sycl::local_accessor<D, 1> block(
            sycl::range<1>(static_cast<std::size_t>(G) * NB * NB), h);
        sycl::local_accessor<int, 1> fail(sycl::range<1>(static_cast<std::size_t>(G)), h);

        h.parallel_for<PotrfLpanelKernel<T, NB>>(
            sycl::nd_range<1>(sycl::range<1>(static_cast<std::size_t>(num_wg) * wg_size),
                              sycl::range<1>(wg_size)),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                const int wg_id = static_cast<int>(it.get_group_linear_id());
                const int lid = static_cast<int>(it.get_local_linear_id());
                const int slot = lid / L;
                const int tid = lid - slot * L;
                const int matrix_id = wg_id * G + slot;

                // NO EARLY RETURN, and no shortened `n` either: every barrier below is a
                // work-group barrier, so a lane that leaves early -- or runs fewer panels --
                // hangs the group. A dead slot runs the whole body against matrix 0 with
                // its stores suppressed.
                const bool live = (matrix_id < batch);

                D* sA = &panel[0] + static_cast<std::ptrdiff_t>(slot) * slda * NB;
                D* sB = &block[0] + static_cast<std::ptrdiff_t>(slot) * NB * NB;
                int* fl = &fail[0] + slot;

                // Built from data_ptr() + b*stride, never MatrixView::operator()(Slice,Slice):
                // its 6-arg ctor defaults stride to ld*cols when 0 is passed, after which
                // every batch item but the first reads the wrong matrix.
                D* Ag = a_ptr + static_cast<std::ptrdiff_t>(live ? matrix_id : 0) * stride_a;

                potrf_native::potrf_lpanel_body<D, R, NB>(it, tid, L, live, sA, slda, sB, fl,
                                                          Ag, ldg, n);

                // One writer per matrix; `fail` is published by B6 of the last panel.
                if (live && tid == 0) info_ptr[matrix_id] = *fl;
            });
    });

    return ctx.get_event();
}

}  // namespace

template <typename T>
Event potrf_lpanel_dispatch(Queue& ctx,
                            const MatrixView<T, MatrixFormat::Dense>& A,
                            Uplo uplo,
                            Span<std::byte> workspace,
                            Span<int32_t> info_out,
                            int min_blocks_per_sm,
                            int nb_hint) {
    using DM = sycl_device::DevMap<T>;
    constexpr std::size_t sz_d = sizeof(typename DM::type);

    const int n = static_cast<int>(A.rows());
    const int batch = static_cast<int>(A.batch_size());

    // supports()'s gates, re-applied: this entry point is reachable without the table.
    if (A.rows() != A.cols()) {
        throw batchlas::invalid_argument("potrf_lpanel: A must be square");
    }
    if (n < 1 || batch < 1) {
        throw batchlas::invalid_argument("potrf_lpanel: degenerate extents");
    }
    if (uplo != Uplo::Lower) {
        // The left-looking recurrence reads A(j+i, k) with k < j -- the LOWER triangle --
        // so Upper would need the transformed-tile trick potrf_cta.cc uses. No in-tree
        // caller asks for it (ortho.cc and linalg::cholesky both pass Lower) and the
        // blocked driver already refuses it, so refusing here preserves the status quo.
        throw batchlas::invalid_argument(
            "potrf_lpanel: Uplo::Upper is not implemented; see "
            "RouteTable<Op::potrf, T>::supports, LPanel arm");
    }
    if (A.is_heterogeneous()) {
        throw batchlas::invalid_argument("potrf_lpanel: heterogeneous batch is not supported");
    }
    const auto dev = ctx.device();
    if (dev.type != DeviceType::GPU) {
        throw batchlas::invalid_argument("potrf_lpanel: GPU queues only");
    }
    if (!dev.supports_sub_group_size(32)) {
        // Enumerated, never get_property(MAX_SUB_GROUP_SIZE) >= 32: that returns the first
        // supported size, so it accepts a {64} device -- a launch abort under reqd_sub_group_size(32).
        throw batchlas::unsupported(
            "potrf_lpanel: device does not offer sub-group size 32, which the kernel requires");
    }

    const int nb = potrf_lpanel_nb_for<T>(nb_hint);
    if (!potrf_lpanel_nb_is_built<T>(nb)) {
        throw batchlas::unsupported("potrf_lpanel: NB = " + std::to_string(nb) +
                                    " is not instantiated for this scalar type");
    }

    const std::size_t device_budget = resident::device_slm_budget(
        dev.get_property(DeviceProperty::LOCAL_MEM_SIZE));
    const std::size_t budget = resident::occupancy_budget(device_budget, min_blocks_per_sm);
    const int max_wg = static_cast<int>(dev.get_property(DeviceProperty::MAX_WORK_GROUP_SIZE));

    const auto p = potrf_lpanel_launch_params(n, nb, batch, sz_d, budget, max_wg);
    if (!p.fits) {
        throw batchlas::invalid_argument(
            "potrf_lpanel: order " + std::to_string(n) +
            " does not fit this device (needs " + std::to_string(p.slm_total) + " B of " +
            std::to_string(budget) + " B, and " + std::to_string(p.L) +
            " work-items); the ceiling for this type is " +
            std::to_string(potrf_lpanel_max_n_for_slm<T>(device_budget, max_wg,
                                                         min_blocks_per_sm, nb)));
    }

    BumpAllocator pool(workspace);
    // detail::info_target's rule, inlined to avoid including src/linalg-impl.hh: an empty
    // or short caller span means "not requested" and draws pool scratch instead.
    Span<int32_t> info = (info_out.size() >= static_cast<std::size_t>(batch))
                             ? info_out
                             : potrf_lpanel_layout<T>(ctx, pool, batch);

    if constexpr (std::is_same_v<T, float>) {
        if (nb == 16) {
            return potrf_lpanel_launch<T, 16>(ctx, A, info, p, n, batch);
        }
    }
    return potrf_lpanel_launch<T, PotrfLpanelConst<T>::NB>(ctx, A, info, p, n, batch);
}

// Per scalar type only, no Backend cross-product: the kernel has no Backend parameter.
#define BATCHLAS_POTRF_LPANEL_INSTANTIATE(T)                                                   \
    template int potrf_lpanel_max_n_for_slm<T>(std::size_t, int, int, int);                    \
    template unsigned potrf_lpanel_debug_launch<T>(Queue&, int, int, int, int);                \
    template std::size_t potrf_lpanel_buffer_size<T>(                                          \
        Queue&, const MatrixView<T, MatrixFormat::Dense>&);                                    \
    template Event potrf_lpanel_dispatch<T>(Queue&, const MatrixView<T, MatrixFormat::Dense>&, \
                                            Uplo, Span<std::byte>, Span<int32_t>, int, int);

BATCHLAS_POTRF_LPANEL_INSTANTIATE(float)
BATCHLAS_POTRF_LPANEL_INSTANTIATE(double)
BATCHLAS_POTRF_LPANEL_INSTANTIATE(std::complex<float>)
BATCHLAS_POTRF_LPANEL_INSTANTIATE(std::complex<double>)

#undef BATCHLAS_POTRF_LPANEL_INSTANTIATE

}  // namespace sycl_potrf
}  // namespace batchlas
