// P4: the REGISTER-RESIDENT GETRF panel leaf, a DROP-IN for getrf_panel_factorize and not
// a tier -- same ipiv contract, same read-modify-written `info`, same cabs1 pivot metric,
// so the two must agree on ipiv EXACTLY. BATCHLAS_GETRF_LEAF picks the arm and the default
// is still the local-memory leaf: no grid has been measured yet. Same device-code cluster
// as getrf_cta.cc (EXTENSIONS_CTA_SOURCES), which is where lu_cabs1 and lu_zero live.
// evidence: docs/perf/lu.md#the-register-panel-leaf

#include "getrf_native.hh"
#include "getrf_panel_reg_device.hh"

#include "../queue.hh"
#include "../util/template-instantiations.hh"

#include <batchlas/error.hh>

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>

namespace batchlas {
namespace sycl_getrf {

// At TU namespace scope: a kernel name must not name an internal-linkage entity.
template <typename T, int NB> class GetrfPanelRegKernel;

namespace {

namespace gn = ::batchlas::getrf_native;

// A launch ABORT, not a slowdown, so every cap below is compile-time. The per-BLOCK file
// is not what binds on sm_89: registers are owned per sub-partition, and the whole-block
// spelling accepts launches the driver refuses.
// evidence: docs/perf/lu.md#the-register-cap-that-binds-is-per-sub-partition
constexpr int kRegsPerBlock = 65536;
constexpr int kRegsPerPartition = 16384;
constexpr int kPartitionsPerBlock = 4;

constexpr int kPanelRegWgQuantum = 32;   // one row per work-item, rounded to a sub-group
constexpr int kPanelRegMaxRows = 512;    // the tallest panel this kernel is written for

// MEASURED ON THIS KERNEL and not inherited from the tiny tier, every cell at 0 stack
// frame and 0 spill -- and cdouble at NB = 32 is PRESENT, against the plan's assumption
// that it would need NB = 16. evidence: docs/perf/lu.md#the-register-panel-leaf-register-probe
constexpr int kPanelRegHeadroomPct = 15;

template <typename T>
constexpr int panel_reg_probe_regs() {
    if constexpr (std::is_same_v<T, float>) return 64;
    else if constexpr (std::is_same_v<T, double>) return 104;
    else if constexpr (std::is_same_v<T, std::complex<float>>) return 96;
    else if constexpr (std::is_same_v<T, std::complex<double>>) return 176;
    else return 0;
}

// Headroom, then ptxas's ALLOCATION GRANULARITY: the count the gates below divide into is
// the ALLOCATED one. evidence: docs/perf/lu.md#the-register-panel-leaf-register-probe
constexpr int kPanelRegAllocGranularity = 8;

template <typename T>
constexpr int panel_reg_budget_regs() {
    const int with_headroom = panel_reg_probe_regs<T>() * (100 + kPanelRegHeadroomPct) / 100;
    return (with_headroom + kPanelRegAllocGranularity - 1) / kPanelRegAllocGranularity *
           kPanelRegAllocGranularity;
}

// The tallest work-group this type's register demand can actually be LAUNCHED at: the
// most warps one sub-partition may hold, times the four partitions a block is dealt over.
constexpr int panel_reg_wg_ceiling(int regs) {
    const int warps_per_partition = kRegsPerPartition / (kPanelRegWgQuantum * regs);
    return warps_per_partition * kPartitionsPerBlock * kPanelRegWgQuantum;
}

// 0 spells "this type has no register panel", exactly as getrf_cta_max_n_for_slm's 0 does.
template <typename T>
constexpr int panel_reg_max_m() {
    constexpr int regs = panel_reg_budget_regs<T>();
    if constexpr (regs < 1) {
        return 0;
    } else {
        constexpr int by_regs =
            panel_reg_wg_ceiling(regs) / kPanelRegWgQuantum * kPanelRegWgQuantum;
        return (by_regs < kPanelRegMaxRows) ? by_regs : kPanelRegMaxRows;
    }
}

// The launch-abort gate for every shipped type, in BOTH spellings: the per-block file and
// the per-sub-partition one that actually binds.
#define BATCHLAS_PANEL_REG_LAUNCH_GATE(T)                                                   \
    static_assert(panel_reg_max_m<T>() * panel_reg_budget_regs<T>() <= kRegsPerBlock);      \
    static_assert(panel_reg_max_m<T>() <= panel_reg_wg_ceiling(panel_reg_budget_regs<T>()));

BATCHLAS_PANEL_REG_LAUNCH_GATE(float)
BATCHLAS_PANEL_REG_LAUNCH_GATE(double)
BATCHLAS_PANEL_REG_LAUNCH_GATE(std::complex<float>)
BATCHLAS_PANEL_REG_LAUNCH_GATE(std::complex<double>)
#undef BATCHLAS_PANEL_REG_LAUNCH_GATE
static_assert(panel_reg_max_m<std::complex<double>>() > 32,
              "the register panel must at least cover a full nb = 32 block's height");

constexpr int panel_reg_wg(int m) {
    return (m + kPanelRegWgQuantum - 1) / kPanelRegWgQuantum * kPanelRegWgQuantum;
}

template <typename T, int NB>
Event getrf_panel_reg_launch(Queue& ctx, T* a_ptr, int ld, int stride,
                             int m, int ncols, int batch,
                             int* piv_ptr, int piv_stride, int piv_base,
                             int32_t* info_ptr, int wg) {
    // std::complex is re-typed HERE, at the pointer boundary: its Annex-G operator* costs
    // an isnan branch and a libcall in device code.
    using DM = sycl_device::DevMap<T>;
    using D = typename DM::type;
    using R = typename DM::real;
    static_assert(sizeof(D) == sizeof(T), "device scalar must be layout-compatible");

    D* const ap = reinterpret_cast<D*>(a_ptr);
    const int kmax = std::min(m, ncols);
    const std::ptrdiff_t strp = static_cast<std::ptrdiff_t>(stride);

    ctx->submit([&](sycl::handler& h) {
        // Under a kilobyte for every instantiation: the 48 KB hole cannot be reached.
        sycl::local_accessor<D, 1> sx(sycl::range<1>(static_cast<std::size_t>(NB)), h);
        sycl::local_accessor<R, 1> rval(
            sycl::range<1>(static_cast<std::size_t>(gn::kLuRedSlots)), h);
        sycl::local_accessor<int, 1> ridx(
            sycl::range<1>(static_cast<std::size_t>(gn::kLuRedSlots)), h);

        h.parallel_for<GetrfPanelRegKernel<T, NB>>(
            sycl::nd_range<1>(sycl::range<1>(static_cast<std::size_t>(batch) *
                                             static_cast<std::size_t>(wg)),
                              sycl::range<1>(static_cast<std::size_t>(wg))),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                const int b = static_cast<int>(it.get_group_linear_id());
                gn::getf2_panel_reg_device<D, NB>(
                    it, ap + static_cast<std::ptrdiff_t>(b) * strp, ld,
                    m, ncols, kmax,
                    piv_ptr + static_cast<std::ptrdiff_t>(b) * piv_stride + piv_base,
                    piv_base,
                    info_ptr + b,
                    sx, rval, ridx);
            });
    });
    return ctx.get_event();
}

}  // namespace

// The ONE place the panel width is spelled, and it is NOT the driver's block width: a
// SHORT final panel arrives here with ncols < NB.
template <typename T>
int getrf_panel_reg_nb() {
    return (panel_reg_max_m<T>() > 0) ? 32 : 0;
}

template <typename T>
int getrf_panel_reg_max_m() {
    return panel_reg_max_m<T>();
}

// The ONE admission test, shared by the driver, the debug hook and the entry point.
template <typename T>
bool getrf_panel_reg_fits(int m, int n, int max_wg) {
    if (m < 1 || n < 1) return false;
    const int nb = getrf_panel_reg_nb<T>();
    if (nb < 1 || n > nb) return false;
    if (m > getrf_panel_reg_max_m<T>()) return false;
    const int wg = panel_reg_wg(m);
    // kLuRedSlots argmax slots is the cross-sub-group scan's whole capacity.
    if (wg > kPanelRegWgQuantum * gn::kLuRedSlots) return false;
    return wg <= max_wg;
}

// Test hook: the work-group this panel would launch, 0 when the panel is not eligible.
template <typename T>
unsigned getrf_panel_reg_debug_launch(Queue& ctx, int m, int n) {
    const int max_wg =
        static_cast<int>(ctx.device().get_property(DeviceProperty::MAX_WORK_GROUP_SIZE));
    if (!getrf_panel_reg_fits<T>(m, n, max_wg)) return 0u;
    return static_cast<unsigned>(panel_reg_wg(m));
}

// `piv_stride` is the matrix ORDER, never the panel width; `info_ptr` is READ as well as
// written, so the caller zeroes it before panel 0.
template <typename T>
Event getrf_panel_reg_factorize(Queue& ctx,
                                T* a_ptr, int ld, int stride,
                                int m, int n, int batch,
                                int* piv_ptr, int piv_stride, int piv_base,
                                int32_t* info_ptr) {
    const auto dev = ctx.device();
    if (m < 1 || n < 1 || batch < 1) {
        throw batchlas::invalid_argument("getrf_panel_reg: degenerate extents");
    }
    if (dev.type != DeviceType::GPU) {
        throw batchlas::invalid_argument("getrf_panel_reg: GPU queues only");
    }
    if (!dev.supports_sub_group_size(32)) {
        // ENUMERATED, never MAX_SUB_GROUP_SIZE >= 32: that property reports entry [0].
        throw batchlas::unsupported(
            "getrf_panel_reg: device does not offer sub-group size 32, which the kernel "
            "requires");
    }
    const int max_wg = static_cast<int>(dev.get_property(DeviceProperty::MAX_WORK_GROUP_SIZE));
    if (!getrf_panel_reg_fits<T>(m, n, max_wg)) {
        // Thrown, never a silent fallback: a leaf that quietly becomes the other leaf is
        // exactly what a pinned A/B cannot see.
        throw batchlas::unsupported(
            "getrf_panel_reg: a " + std::to_string(m) + "x" + std::to_string(n) +
            " panel is outside this type's register-resident leaf (width <= " +
            std::to_string(getrf_panel_reg_nb<T>()) + ", height <= " +
            std::to_string(getrf_panel_reg_max_m<T>()) + ", work-group <= " +
            std::to_string(max_wg) + ")");
    }

    // `if constexpr`, not the runtime guard above: a plain call INSTANTIATES the kernel
    // even for a type whose table says absent. The `else` is unreachable.
    if constexpr (panel_reg_max_m<T>() > 0) {
        constexpr int NB = 32;
        return getrf_panel_reg_launch<T, NB>(ctx, a_ptr, ld, stride, m, n, batch,
                                             piv_ptr, piv_stride, piv_base, info_ptr,
                                             panel_reg_wg(m));
    } else {
        throw batchlas::unsupported(
            "getrf_panel_reg: this scalar type has no register-resident panel leaf");
    }
}

// Per scalar type only, no Backend cross-product: this build is device-link-bound.
#define BATCHLAS_GETRF_PANEL_REG_INSTANTIATE(T)                                             \
    template int getrf_panel_reg_nb<T>();                                                   \
    template int getrf_panel_reg_max_m<T>();                                                \
    template bool getrf_panel_reg_fits<T>(int, int, int);                                   \
    template unsigned getrf_panel_reg_debug_launch<T>(Queue&, int, int);                    \
    template Event getrf_panel_reg_factorize<T>(Queue&, T*, int, int, int, int, int, int*,  \
                                                int, int, int32_t*);

BATCHLAS_GETRF_PANEL_REG_INSTANTIATE(float)
BATCHLAS_GETRF_PANEL_REG_INSTANTIATE(double)
BATCHLAS_GETRF_PANEL_REG_INSTANTIATE(std::complex<float>)
BATCHLAS_GETRF_PANEL_REG_INSTANTIATE(std::complex<double>)

#undef BATCHLAS_GETRF_PANEL_REG_INSTANTIATE

}  // namespace sycl_getrf
}  // namespace batchlas
