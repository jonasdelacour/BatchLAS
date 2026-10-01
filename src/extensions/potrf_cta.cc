// Native batched POTRF: the CTA kernel's launcher and capability surface; the
// device code is in potrf_cta_device.hh. This TU must stay in EXTENSIONS_CTA_SOURCES
// next to potrf_blocked.cc, whose diagonal leaf is potrf_cta_body -- splitting a
// device-code cluster across libraries is a `ptxas fatal: Unresolved extern function`.
// evidence: docs/perf/potrf.md

#include "potrf_launch_plan.hh"
#include "potrf_native.hh"
#include "potrf_cta_device.hh"
#include "potrf_slm_hole.hh"

#include "../queue.hh"
#include "../util/resident_capacity.hh"
#include "../util/template-instantiations.hh"

#include <batchlas/util/mempool.hh>

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace batchlas {

// Kernel name tag; outside the anonymous namespace so it names no internal-linkage entity.
template <typename T, int NB, int TS, potrf_native::PotrfScope SC>
class PotrfCtaKernel;

namespace sycl_potrf {

namespace {

using potrf_native::PotrfScope;

// Geometry, capacity and the L ladder live in potrf_launch_plan.hh: the launcher, the
// ceiling supports() advertises and the cost model read that one definition.
template <typename T>
using PotrfCtaConst = potrf_plan::CtaConst<T>;

// Convenience overloads only; every real decision re-reads the device. NOT device_limits.hh's
// constant, which is hardcoded per architecture and never queried.
// evidence: docs/perf/potrf.md#the-slm-budget-and-the-fit-ceilings
constexpr std::size_t kPotrfReferenceSlmBudget = 97280;

using PotrfCtaLaunch = potrf_plan::CtaGeometry;

// Under Scope::WorkGroup the phase barriers are work-group barriers, which is correct only
// when the work-group holds exactly one matrix.
inline void potrf_cta_check_scope(const PotrfCtaLaunch& p) {
    if (!p.subgroup_scope && p.G != 1) {
        throw batchlas::internal_error("potrf_cta: Scope::WorkGroup with G != 1 is a race by construction");
    }
}

}  // namespace

template <typename T>
int potrf_cta_max_n_for_slm(std::size_t slm_budget_bytes, int min_blocks_per_sm) {
    return potrf_plan::cta_max_n<T>(slm_budget_bytes, min_blocks_per_sm);
}

template <typename T>
int potrf_cta_max_n() {
    return potrf_cta_max_n_for_slm<T>(kPotrfReferenceSlmBudget);
}

namespace {

template <typename T>
Span<int32_t> potrf_cta_layout(Queue& ctx, BumpAllocator& pool, int batch) {
    return pool.allocate<int32_t>(ctx, static_cast<std::size_t>(batch));
}

}  // namespace

template <typename T>
std::size_t potrf_cta_buffer_size(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A) {
    const int batch = A.batch_size();
    return workspace_bytes([&](BumpAllocator& p) {
        return potrf_cta_layout<T>(ctx, p, batch);
    });
}

// The launch geometry, for tests; see potrf_native.hh.
template <typename T>
unsigned potrf_cta_debug_launch(Queue& ctx, int n, int batch, int min_blocks_per_sm) {
    const auto p = potrf_plan::cta_geometry<T>(n, batch, potrf_device_facts(ctx.device()),
                                               min_blocks_per_sm);
    potrf_cta_check_scope(p);
    if (!p.fits) return 0u;
    return (static_cast<unsigned>(p.L) << 16) | static_cast<unsigned>(p.G);
}

namespace {

template <typename T, int NB, int TS, PotrfScope SC>
Event potrf_cta_launch(Queue& ctx,
                       const MatrixView<T, MatrixFormat::Dense>& A,
                       bool upper,
                       Span<int32_t> info,
                       const PotrfCtaLaunch& p,
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
    const int lda = p.lda;
    const int Rt0 = p.Rt0;
    const int G = p.G;
    const int L = p.L;
    const int wg_size = p.wg_size;
    const std::int64_t num_wg = p.num_wg;
    int32_t* info_ptr = info.data();

    // Padded into the TILE accessor: a fifth, unused local_accessor is plausibly eliminated.
    const std::size_t tile_elems_used = static_cast<std::size_t>(G) *
                                        static_cast<std::size_t>(lda) * static_cast<std::size_t>(n);
    const std::size_t natural = static_cast<std::size_t>(G) * p.slm_per_matrix;
    const std::size_t pad_bytes = (p.slm_total > natural) ? (p.slm_total - natural) : 0;
    const std::size_t tile_elems = tile_elems_used + (pad_bytes + sizeof(D) - 1) / sizeof(D);

    ctx->submit([&](sycl::handler& h) {
        sycl::local_accessor<D, 1> tile(sycl::range<1>(tile_elems), h);
        sycl::local_accessor<R, 1> diag(sycl::range<1>(static_cast<std::size_t>(G) * NB), h);
        sycl::local_accessor<int, 1> fail(sycl::range<1>(static_cast<std::size_t>(G)), h);
        sycl::local_accessor<int, 1> off(
            sycl::range<1>(static_cast<std::size_t>(G) * static_cast<std::size_t>(Rt0 + 1)), h);

        h.parallel_for<PotrfCtaKernel<T, NB, TS, SC>>(
            sycl::nd_range<1>(sycl::range<1>(static_cast<std::size_t>(num_wg) * wg_size),
                              sycl::range<1>(wg_size)),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                const auto sg = it.get_sub_group();
                const int wg_id = static_cast<int>(it.get_group_linear_id());

                int matrix_id;
                int slot;
                int tid;
                bool p1_active;
                if constexpr (SC == PotrfScope::SubGroup) {
                    const int sg_id = static_cast<int>(sg.get_group_linear_id());
                    matrix_id = wg_id * G + sg_id;
                    slot = sg_id;
                    tid = static_cast<int>(sg.get_local_linear_id());
                    p1_active = true;
                    // Sub-group-uniform, and this scope uses only sub-group barriers.
                    if (matrix_id >= batch) return;
                } else {
                    matrix_id = wg_id;   // G == 1 => num_wg == batch, cannot exceed
                    slot = 0;
                    tid = static_cast<int>(it.get_local_linear_id());
                    p1_active = (sg.get_group_linear_id() == 0);
                }

                D* S = &tile[0] + static_cast<std::ptrdiff_t>(slot) * lda * n;
                R* dg = &diag[0] + static_cast<std::ptrdiff_t>(slot) * NB;
                int* fl = &fail[0] + slot;
                int* of = &off[0] + static_cast<std::ptrdiff_t>(slot) * (Rt0 + 1);

                // Built from data_ptr() + b*stride, never MatrixView::operator()(Slice,Slice):
                // its 6-arg ctor defaults stride to ld*cols when 0 is passed, after which
                // every batch item but the first reads the wrong matrix.
                D* Ag = a_ptr + static_cast<std::ptrdiff_t>(matrix_id) * stride_a;

                potrf_native::potrf_cta_body<D, R, NB, TS, SC>(
                    it, sg, tid, L, p1_active, S, lda, dg, fl, of, Ag, ldg, n, upper);

                // One writer per matrix; `fail` is published by B3 of the last panel.
                if (tid == 0) info_ptr[matrix_id] = *fl;
            });
    });

    return ctx.get_event();
}

}  // namespace

template <typename T>
Event potrf_cta_dispatch(Queue& ctx,
                         const MatrixView<T, MatrixFormat::Dense>& A,
                         Uplo uplo,
                         Span<std::byte> workspace,
                         Span<int32_t> info_out,
                         int min_blocks_per_sm) {
    using C = PotrfCtaConst<T>;

    const int n = static_cast<int>(A.rows());
    const int batch = static_cast<int>(A.batch_size());

    // supports()'s gates, re-applied: this entry point is reachable without the table.
    if (A.rows() != A.cols()) {
        throw batchlas::invalid_argument("potrf_cta: A must be square");
    }
    if (n < 1 || batch < 1) {
        throw batchlas::invalid_argument("potrf_cta: degenerate extents");
    }
    if (A.is_heterogeneous()) {
        // One launch, one (order, ld, stride) tuple read from the capacity extents.
        throw batchlas::invalid_argument("potrf_cta: heterogeneous batch is not supported");
    }
    const auto dev = ctx.device();
    if (dev.type != DeviceType::GPU) {
        throw batchlas::invalid_argument("potrf_cta: GPU queues only");
    }
    if (!dev.supports_sub_group_size(32)) {
        // Enumerated, never get_property(MAX_SUB_GROUP_SIZE) >= 32: that returns the first
        // supported size, so it accepts a {64} device -- a launch abort under reqd_sub_group_size(32).
        throw batchlas::unsupported(
            "potrf_cta: device does not offer sub-group size 32, which the kernel requires");
    }

    // The OCCUPANCY slice, not the whole budget: at the default target this gate is the
    // same predicate supports() advertises, so a routed order cannot fail at enqueue. The
    // blocked driver passes the target ITS block width was clamped against, which above
    // potrf_plan::kOccupancyNbMaxOrder is 1 -- the leaf's order and its gate stay one decision.
    const auto facts = potrf_device_facts(dev);
    const std::size_t device_budget = resident::device_slm_budget(facts.local_mem_bytes);
    const std::size_t budget = resident::occupancy_budget(device_budget, min_blocks_per_sm);

    const auto p = potrf_plan::cta_geometry<T>(n, batch, facts, min_blocks_per_sm);
    potrf_cta_check_scope(p);
    if (!p.fits) {
        throw batchlas::invalid_argument(
            "potrf_cta: order " + std::to_string(n) +
            " does not fit this device's local memory (needs " +
            std::to_string(p.slm_total) + " B of " + std::to_string(budget) +
            " B); the ceiling for this type is " +
            std::to_string(potrf_cta_max_n_for_slm<T>(device_budget, min_blocks_per_sm)));
    }

    BumpAllocator pool(workspace);
    // detail::info_target's rule, inlined to avoid including src/linalg-impl.hh: an empty
    // or short caller span means "not requested" and draws pool scratch instead.
    Span<int32_t> info = (info_out.size() >= static_cast<std::size_t>(batch))
                             ? info_out
                             : potrf_cta_layout<T>(ctx, pool, batch);

    const bool upper = (uplo == Uplo::Upper);

    if (p.subgroup_scope) {
        return potrf_cta_launch<T, C::NB, C::TS, PotrfScope::SubGroup>(
            ctx, A, upper, info, p, n, batch);
    }
    return potrf_cta_launch<T, C::NB, C::TS, PotrfScope::WorkGroup>(
        ctx, A, upper, info, p, n, batch);
}

// Per scalar type only, no Backend cross-product: the kernel has no Backend parameter.
#define BATCHLAS_POTRF_CTA_INSTANTIATE(T)                                                   \
    template int potrf_cta_max_n_for_slm<T>(std::size_t, int);                              \
    template int potrf_cta_max_n<T>();                                                      \
    template unsigned potrf_cta_debug_launch<T>(Queue&, int, int, int);                     \
    template std::size_t potrf_cta_buffer_size<T>(Queue&,                                   \
                                                  const MatrixView<T, MatrixFormat::Dense>&); \
    template Event potrf_cta_dispatch<T>(Queue&, const MatrixView<T, MatrixFormat::Dense>&, \
                                         Uplo, Span<std::byte>, Span<int32_t>, int);

BATCHLAS_POTRF_CTA_INSTANTIATE(float)
BATCHLAS_POTRF_CTA_INSTANTIATE(double)
BATCHLAS_POTRF_CTA_INSTANTIATE(std::complex<float>)
BATCHLAS_POTRF_CTA_INSTANTIATE(std::complex<double>)

#undef BATCHLAS_POTRF_CTA_INSTANTIATE

}  // namespace sycl_potrf
}  // namespace batchlas
