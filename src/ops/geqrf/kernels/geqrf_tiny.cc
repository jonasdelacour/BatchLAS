// SPDX-License-Identifier: MIT
// Native batched GEQRF: the TINY tier -- square m == n <= 32, one matrix per
// SubGroupPartition<N>, the whole matrix in registers between the load and the store. A thin
// launcher: the body is device::geqrf<tier::Reg<N>> (include/batchlas/device/lapack/geqrf.hh).
// evidence: docs/perf/qr.md#the-tiny-tier-wp6--p1-square-n--32-in-registers

#include <src/ops/geqrf/geqrf_native.hh>
#include <src/ops/geqrf/kernels/geqrf_tiny_device.hh>
#include <src/kernels/common/tiny_device.hh>
#include <batchlas/device/sg_partition.hh>
#include <batchlas/device/lapack/geqrf.hh>

#include <src/queue.hh>
#include <src/util/resident_capacity.hh>
#include <src/util/template-instantiations.hh>
#include <batchlas/device/detail/kernel_attrs.hh>
#include <src/util/batch_mask.hh>

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>

// Outside the anonymous namespace: a kernel name may not name an internal-linkage entity.
namespace batchlas {
template <typename T, int N, int C, bool Masked> class GeqrfTinyKernel;
}

namespace batchlas {
namespace sycl_geqrf {

namespace {

namespace gn = ::batchlas::geqrf_native;
namespace tn = ::batchlas::tiny_native;
namespace v2 = ::batchlas::device;

// THE LAUNCH GATES. The register table lives next to the chunk rule that reads it; a second
// copy here let a MODEL choose the chunk width while a PROBE gated the launch. evidence: docs/perf/qr.md#the-tiny-tier-register-table
template <typename T> using TinyDev = typename sycl_device::DevMap<T>::type;

constexpr int geqrf_tiny_worst_regs() {
    int worst = 0;
    for (const int* t : {gn::GeqrfTinyRegs<float>::at, gn::GeqrfTinyRegs<double>::at,
                         gn::GeqrfTinyRegs<sycl_device::Cx<float>>::at,
                         gn::GeqrfTinyRegs<sycl_device::Cx<double>>::at}) {
        for (int i = 0; i < 3; ++i) worst = (t[i] > worst) ? t[i] : worst;
    }
    return worst + gn::kGeqrfTinyRegMargin;
}

// GATE 1, the HARD one: a work-group whose warps overflow a register SUB-PARTITION ABORTS
// the launch -- 64 lanes is 2 warps in one of the four, 1 x 32 x ceil8(226 + 8) = 7,680 of
// its 16,384. Both operands are named so raising either is deliberate; GATE 2 still binds.
constexpr int kGeqrfTinyMaxWg = gn::kGeqrfTinyWg;
constexpr int kGeqrfTinyWorstRegs = geqrf_tiny_worst_regs();
static_assert(resident::sm89_fits(kGeqrfTinyWorstRegs, kGeqrfTinyMaxWg),
              "geqrf tiny: regs x wg overflows a register sub-partition -- re-run "
              "scripts/register_probe.sh before raising either operand");

// GATE 2: every SHIPPED cell keeps R1's resident::kMinBlocksPerSm blocks per SM (a regression
// fails to COMPILE). GATE 3: each row's probed chunk width still re-derives from that row.
#define BATCHLAS_GEQRF_TINY_CELL_ASSERT(D, N)                                               \
    static_assert(gn::geqrf_tiny_cell_is_resident<D, N>(),                                  \
                  "geqrf tiny: this cell falls below R1's four-blocks-per-SM floor");       \
    static_assert(gn::geqrf_tiny_chunk_is_fixed_point<D, N>(),                              \
                  "geqrf tiny: the register row was probed at a different chunk width "     \
                  "than the rule now derives -- re-probe at the derived C and update the "  \
                  "row, or pin the cell");

BATCHLAS_GEQRF_TINY_CELL_ASSERT(float, 8)
BATCHLAS_GEQRF_TINY_CELL_ASSERT(float, 16)
BATCHLAS_GEQRF_TINY_CELL_ASSERT(float, 32)
BATCHLAS_GEQRF_TINY_CELL_ASSERT(double, 8)
BATCHLAS_GEQRF_TINY_CELL_ASSERT(double, 16)
BATCHLAS_GEQRF_TINY_CELL_ASSERT(double, 32)
BATCHLAS_GEQRF_TINY_CELL_ASSERT(sycl_device::Cx<float>, 8)
BATCHLAS_GEQRF_TINY_CELL_ASSERT(sycl_device::Cx<float>, 16)
BATCHLAS_GEQRF_TINY_CELL_ASSERT(sycl_device::Cx<float>, 32)
BATCHLAS_GEQRF_TINY_CELL_ASSERT(sycl_device::Cx<double>, 8)
BATCHLAS_GEQRF_TINY_CELL_ASSERT(sycl_device::Cx<double>, 16)
#undef BATCHLAS_GEQRF_TINY_CELL_ASSERT

// cdouble's N = 32 cell is excluded by INSTANTIATION BUDGET, not by physics.
// evidence: docs/perf/qr.md#the-cdouble-n32-cell
template <typename T>
constexpr int geqrf_tiny_type_ceiling() {
    return std::is_same_v<T, std::complex<double>> ? 16 : 32;
}

// Local memory ONE work-group asks for, at the derived chunk width.
template <typename T, int N>
constexpr std::size_t geqrf_tiny_wg_bytes() {
    constexpr int C = gn::geqrf_tiny_chunk<TinyDev<T>, N>();
    return static_cast<std::size_t>(gn::geqrf_tiny_matrices_per_wg<N>()) *
           gn::geqrf_tiny_slm_elems(N, C) * sizeof(T);
}

// No request lands near the (47104, 49664] launch hole, so geqrf_cta.cc's padding is not carried.
static_assert(geqrf_tiny_wg_bytes<std::complex<double>, 8>() < 47104,
              "tiny tile approaching the 48 KB launch hole");
static_assert(geqrf_tiny_wg_bytes<double, 32>() < 47104,
              "tiny tile approaching the 48 KB launch hole");

// Masked: the mask folds into `live` (I2), a template so the unmasked kernel is unchanged (the fold
// cost potrf tiny 2-4.5%). evidence: docs/design/masked-batches.md#masked-batches-kernel-contract
template <typename T, int N, bool Masked>
Event geqrf_tiny_launch(Queue& ctx, T* a_ptr, int ld, int stride, int n, int batch,
                        T* tau_ptr, const std::uint8_t* mask) {
    // Re-typed HERE: std::complex's Annex-G operator* is an isnan branch plus a libcall.
    using DM = sycl_device::DevMap<T>;
    using D = typename DM::type;
    static_assert(sizeof(D) == sizeof(T), "device scalar must be layout-compatible");

    constexpr int C = gn::geqrf_tiny_chunk<D, N>();
    constexpr int M = gn::geqrf_tiny_matrices_per_wg<N>();
    constexpr int WG = M * N;
    constexpr std::size_t kPerMatrix = gn::geqrf_tiny_slm_elems(N, C);
    using Tier = v2::tier::Reg<N>;
    // The public plan carves this launcher's per-matrix slot: tile at 0, y right after it.
    static_assert(v2::geqrf_buffer_size<Tier, D, C>(N).local_bytes == kPerMatrix * sizeof(D),
                  "geqrf tiny: the device plan and the launcher's slot stride disagree");
    static_assert(gn::geqrf_tiny_slda(C) == C + 1, "geqrf tiny: the device body's tile stride is C + 1");

    static_assert(WG == gn::kGeqrfTinyWg, "the work-group must stay a whole number of warps");
    static_assert(WG % tn::kTinySubGroupSize == 0, "the work-group must be whole sub-groups");
    static_assert(tn::tiny_n_is_legal(N), "N must divide the sub-group");

    D* const ap = reinterpret_cast<D*>(a_ptr);
    D* const tp = reinterpret_cast<D*>(tau_ptr);
    const int num_wg = (batch + M - 1) / M;

    ctx->submit([&](sycl::handler& h) {
        sycl::local_accessor<D, 1> scratch(
            sycl::range<1>(static_cast<std::size_t>(M) * kPerMatrix), h);

        batchlas::impl::parallel_for<GeqrfTinyKernel<T, N, C, Masked>>(h,
            sycl::nd_range<1>(sycl::range<1>(static_cast<std::size_t>(num_wg) *
                                             static_cast<std::size_t>(WG)),
                              sycl::range<1>(static_cast<std::size_t>(WG))),
            [=](sycl::nd_item<1> it) BATCHLAS_REQD_SG_SIZE(32) {
                const auto sg = it.get_sub_group();
                const v2::SgpPartition<N> ex{sg};
                const auto& part = ex.part;
                const int slot = tn::tiny_partition_id(sg, part);
                const int prob = static_cast<int>(it.get_group_linear_id()) * M + slot;

                // NO EARLY RETURN. `if (prob >= batch) return;` is partition-uniform but NOT
                // sub-group-uniform, so at N < 32 a later group_barrier(sg) is undefined. A
                // dead partition carries the identity and suppresses its store. Guarded by
                // source text (GeqrfTinySource.KernelBodyHasNoEarlyReturn), not numerically.
                // First uses in this order (batch, mask, n, stride, ap, ld, scratch, tp): [=]
                // orders the kernel arguments by them, as the inline body did.
                bool live = prob < batch;
                if constexpr (Masked) live = live && mask[prob] != 0;  // inactive: untouched A and tau
                const int order = n;
                const std::ptrdiff_t off =
                    live ? static_cast<std::ptrdiff_t>(prob) * stride : std::ptrdiff_t(0);
                D* const a0 = ap + off;
                const int lda = ld;

                // The body is the public device::geqrf<tier::Reg<N>> (lapack/geqrf.hh); the
                // per-matrix slot of the work-group's local memory is its Workspace.
                D* const sP = &scratch[0] + static_cast<std::ptrdiff_t>(slot) *
                                                static_cast<std::ptrdiff_t>(kPerMatrix);
                const v2::Workspace ws{
                    sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::yes>(
                        reinterpret_cast<std::byte*>(sP))};
                v2::geqrf<Tier, C>(ex, v2::mat_wide(a0, order, order, lda),
                                   v2::vec_wide(tp + static_cast<std::ptrdiff_t>(prob) * order, order), ws, live);
            });
    });
    return ctx.get_event();
}

}  // namespace

template <typename T>
int geqrf_tiny_max_n_for_slm(std::size_t slm_budget_bytes) {
    // A walk with a break, not a max: per-work-group bytes are NOT monotone in N, and
    // can_run advertises the contiguous range n <= max_n, so the ceiling must be the
    // largest N at which every smaller bucket also fits.
    const std::size_t slice = resident::occupancy_budget(slm_budget_bytes);
    constexpr int ceiling = geqrf_tiny_type_ceiling<T>();

    if (geqrf_tiny_wg_bytes<T, 8>() > slice) return 0;
    if (ceiling < 16 || geqrf_tiny_wg_bytes<T, 16>() > slice) return 8;
    if (ceiling < 32 || geqrf_tiny_wg_bytes<T, 32>() > slice) return 16;
    return 32;
}

template <typename T>
std::size_t geqrf_tiny_buffer_size(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A) {
    // Zero, and monotone in (rows, cols, batch) by being constant. Must not dereference
    // A.data_ptr(): band_reduction.cc sizes against a null-data dummy view.
    static_cast<void>(ctx);
    static_cast<void>(A);
    return 0;
}

template <typename T>
unsigned geqrf_tiny_debug_launch(Queue& ctx, int n) {
    const auto dev = ctx.device();
    if (dev.type != DeviceType::GPU || !dev.supports_sub_group_size(32)) return 0u;
    const std::size_t budget =
        resident::device_slm_budget(dev.get_property(DeviceProperty::LOCAL_MEM_SIZE));
    if (n < 1 || n > geqrf_tiny_max_n_for_slm<T>(budget)) return 0u;
    switch (tn::tiny_bucket_ge(n)) {
        case 8:  return (8u << 16)  | unsigned(gn::geqrf_tiny_matrices_per_wg<8>());
        case 16: return (16u << 16) | unsigned(gn::geqrf_tiny_matrices_per_wg<16>());
        case 32: return (32u << 16) | unsigned(gn::geqrf_tiny_matrices_per_wg<32>());
        default: return 0u;
    }
}

template <typename T>
Event geqrf_tiny_dispatch(Queue& ctx,
                          const MatrixView<T, MatrixFormat::Dense>& A,
                          Span<T> tau,
                          Span<std::byte> workspace) {
    static_cast<void>(workspace);

    const int m = static_cast<int>(A.rows());
    const int n = static_cast<int>(A.cols());
    const int batch = static_cast<int>(A.batch_size());

    // Direct callers reach this without the selector, so every gate can_run applies is
    // RE-APPLIED here and throws (a pin the selector refuses throws there instead).
    if (m < 1 || n < 1 || batch < 1) {
        throw batchlas::invalid_argument("geqrf_tiny: degenerate extents");
    }
    if (m != n) {
        throw batchlas::invalid_argument(
            "geqrf_tiny: the tiny tier factors SQUARE matrices only (m == n); a tall panel "
            "is the CTA tier's");
    }
    if (A.is_heterogeneous()) {
        throw batchlas::invalid_argument("geqrf_tiny: heterogeneous batch is not supported");
    }
    const auto dev = ctx.device();
    if (dev.type != DeviceType::GPU) {
        throw batchlas::invalid_argument("geqrf_tiny: GPU queues only");
    }
    if (!dev.supports_sub_group_size(32)) {
        // ENUMERATED, not MAX_SUB_GROUP_SIZE >= 32: that returns the FIRST supported
        // size, so the weak test accepts a {64} device and the launch aborts.
        throw batchlas::unsupported(
            "geqrf_tiny: device does not offer sub-group size 32, which the kernel requires");
    }
    if (tau.size() < static_cast<std::size_t>(n) * static_cast<std::size_t>(batch)) {
        throw batchlas::invalid_argument("geqrf_tiny: tau span is shorter than n * batch");
    }

    const std::size_t budget =
        resident::device_slm_budget(dev.get_property(DeviceProperty::LOCAL_MEM_SIZE));
    const int max_n = geqrf_tiny_max_n_for_slm<T>(budget);
    if (n > max_n) {
        throw batchlas::invalid_argument(
            "geqrf_tiny: order " + std::to_string(n) + " is above this type's tiny ceiling (" +
            std::to_string(max_n) + ") on this device's local-memory budget");
    }

    T* const a_ptr = A.data_ptr();
    T* const tau_ptr = tau.data();
    const int ld = static_cast<int>(A.ld());
    const int stride = static_cast<int>(A.stride());
    const std::uint8_t* const mask = ::batchlas::detail::mask_ptr(::batchlas::detail::call_mask("geqrf_tiny", A));
    const auto launch = [&]<int NB>(std::integral_constant<int, NB>) {
        return mask ? geqrf_tiny_launch<T, NB, true>(ctx, a_ptr, ld, stride, n, batch, tau_ptr, mask)
                    : geqrf_tiny_launch<T, NB, false>(ctx, a_ptr, ld, stride, n, batch, tau_ptr, nullptr);
    };

    // Explicit throw, never a silent leading-submatrix solve. The N = 32 arm sits under
    // `if constexpr` so a ceiling-16 type does not INSTANTIATE an unreachable kernel.
    const int bucket = tn::tiny_bucket_ge(n);
    if (bucket == 8) {
        return launch(std::integral_constant<int, 8>{});
    }
    if (bucket == 16) {
        return launch(std::integral_constant<int, 16>{});
    }
    if constexpr (geqrf_tiny_type_ceiling<T>() >= 32) {
        if (bucket == 32) {
            return launch(std::integral_constant<int, 32>{});
        }
    }
    throw batchlas::unsupported(
        "geqrf_tiny: order " + std::to_string(n) +
        " has no compile-time bucket for this scalar type; the ladder is {8, 16, 32}");
}

// Per scalar type only; the ladder above pulls the <T, N, C> cross-product in implicitly.
#define BATCHLAS_GEQRF_TINY_INSTANTIATE(T)                                                     \
    template int geqrf_tiny_max_n_for_slm<T>(std::size_t);                                     \
    template unsigned geqrf_tiny_debug_launch<T>(Queue&, int);                                 \
    template std::size_t geqrf_tiny_buffer_size<T>(                                            \
        Queue&, const MatrixView<T, MatrixFormat::Dense>&);                                    \
    template Event geqrf_tiny_dispatch<T>(Queue&, const MatrixView<T, MatrixFormat::Dense>&,   \
                                          Span<T>, Span<std::byte>);

BATCHLAS_GEQRF_TINY_INSTANTIATE(float)
BATCHLAS_GEQRF_TINY_INSTANTIATE(double)
BATCHLAS_GEQRF_TINY_INSTANTIATE(std::complex<float>)
BATCHLAS_GEQRF_TINY_INSTANTIATE(std::complex<double>)

#undef BATCHLAS_GEQRF_TINY_INSTANTIATE

}  // namespace sycl_geqrf
}  // namespace batchlas
