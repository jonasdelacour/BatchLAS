// Native batched POTRF, Phase 1: the CTA kernel's launcher and capability
// surface; all device code is in potrf_cta_device.hh. This TU must stay in
// EXTENSIONS_CTA_SOURCES next to potrf_blocked.cc, whose diagonal leaf is
// potrf_cta_body -- splitting a device-code cluster across libraries is a
// `ptxas fatal: Unresolved extern function`.
// evidence: docs/perf/potrf.md

#include "potrf_native.hh"
#include "potrf_cta_device.hh"

#include "../queue.hh"
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

// The (NB, TS) ladder per scalar type: NB is the panel width and the length of
// the d[]/x[] register arrays, TS the (P3) thread tile. Both are measured, and
// NB = 16 is slower. evidence: docs/perf/potrf.md#register-gate
template <typename T>
struct PotrfCtaConst;
template <> struct PotrfCtaConst<float>                { static constexpr int NB = 8;  static constexpr int TS = 4; };
template <> struct PotrfCtaConst<double>               { static constexpr int NB = 8;  static constexpr int TS = 4; };
template <> struct PotrfCtaConst<std::complex<float>>  { static constexpr int NB = 8;  static constexpr int TS = 4; };
template <> struct PotrfCtaConst<std::complex<double>> { static constexpr int NB = 8;  static constexpr int TS = 2; };

// Runtime local_mem_size minus the 4096 B reserve every other device-BLAS sizing
// decision here applies. NOT device_limits.hh's 49152, which is wrong here by 2.06x.
constexpr std::size_t kPotrfReferenceSlmBudget = 97280;

// Soft occupancy target for matrices per work-group: 4 resident blocks/SM on sm_89.
constexpr std::size_t kPotrfSlmSoftTarget = 24576;

// The L ladder's two knobs; shared memory, not registers or threads, binds these launches.
constexpr int kPotrfMaxL = 256;
constexpr int kPotrfElemsPerItem = 24;

// THE SLM FORMULA, called by BOTH the capability query and the launcher, so the
// ceiling supports() advertises and the allocation the kernel makes cannot
// disagree. lda = n | 1 is odd, so a stride-lda row read is conflict-free; the
// 256 covers *fail plus inter-accessor alignment slack and is a deliberate
// over-estimate -- at 64 it fell 60 B short at the float ceiling, and short here
// means supports() advertises an order whose launch fails at enqueue.
// evidence: docs/perf/potrf.md#the-slm-budget-and-the-fit-ceilings
constexpr std::size_t potrf_slm_per_matrix(int n, int NB, int TS,
                                           std::size_t sz_d, std::size_t sz_r) {
    const std::size_t lda = static_cast<std::size_t>(n | 1);
    const int m2_0 = (n > NB) ? (n - NB) : 0;
    const int Rt0 = (m2_0 + TS - 1) / TS;
    return lda * static_cast<std::size_t>(n) * sz_d
         + static_cast<std::size_t>(NB) * sz_r
         + 256
         + 4 * static_cast<std::size_t>(Rt0 + 1);
}

// The 48 KB launch hole: a dynamic local-memory request in
// (49152 - static_shared, 49152] fails at enqueue with CUDA_ERROR_INVALID_VALUE,
// and only on a process's first launch. Inert while these kernels have zero
// static shared, but one group algorithm in the body reintroduces it.
// evidence: docs/perf/potrf.md#the-48-kb-launch-hole
constexpr std::size_t kPotrfHoleLo = 47104;
constexpr std::size_t kPotrfHoleHi = 49664;
constexpr std::size_t kPotrfHolePadTo = 49920;

constexpr std::size_t potrf_hole_padded(std::size_t bytes) {
    return (bytes > kPotrfHoleLo && bytes <= kPotrfHoleHi) ? kPotrfHolePadTo : bytes;
}

inline int prev_pow2(int v) {
    int r = 1;
    while ((r << 1) <= v) r <<= 1;
    return r;
}

// Everything the launch needs, computed once. Scope is DERIVED here and nowhere
// else, so no caller can assert one the L ladder disagrees with.
struct PotrfCtaLaunch {
    int L = 32;               // work-items per matrix
    int G = 1;                // matrices per work-group; > 1 only when L == 32
    int wg_size = 32;
    int num_wg = 0;
    int lda = 1;
    int Rt0 = 0;
    std::size_t slm_per_matrix = 0;
    std::size_t slm_total = 0;   // G * slm_per_matrix, after the hole pad
    PotrfScope scope = PotrfScope::SubGroup;
    bool fits = false;
};

template <int NB, int TS>
PotrfCtaLaunch potrf_cta_launch_params(int n, int batch, std::size_t sz_d, std::size_t sz_r,
                                       std::size_t slm_budget, int max_wg) {
    PotrfCtaLaunch p;
    p.lda = n | 1;
    const int m2_0 = (n > NB) ? (n - NB) : 0;
    p.Rt0 = (m2_0 + TS - 1) / TS;
    const long long Ntiles_0 = static_cast<long long>(p.Rt0) * (p.Rt0 + 1) / 2;

    // L is derived from m2_0 = n - NB, the FIRST trailing update, not from n:
    // ceil(n/TS) counts a triangle that is never updated. The ladder counts
    // ELEMENTS per work-item, not tiles, since TS varies across the type ladder;
    // kPotrfElemsPerItem is fitted. evidence: docs/perf/potrf.md#the-l-ladder
    {
        const long long work_elems = Ntiles_0 * static_cast<long long>(TS) * TS;
        int want = 32;
        while (want < kPotrfMaxL &&
               static_cast<long long>(want) * kPotrfElemsPerItem < work_elems) {
            want <<= 1;
        }
        p.L = want;
    }
    while (p.L > 32 && p.L > max_wg) p.L >>= 1;

    p.slm_per_matrix = potrf_slm_per_matrix(n, NB, TS, sz_d, sz_r);

    // G > 1 only at L == 32, capped so wg_size <= 128; wg 256 is register-limited.
    if (p.L == 32 && p.slm_per_matrix > 0) {
        const std::size_t target = std::min(kPotrfSlmSoftTarget, slm_budget);
        const int by_slm = static_cast<int>(target / p.slm_per_matrix);
        p.G = std::clamp(prev_pow2(std::max(1, by_slm)), 1, 4);
        while (p.G > 1 && (p.G * p.L > max_wg ||
                           static_cast<std::size_t>(p.G) * p.slm_per_matrix > slm_budget)) {
            p.G >>= 1;
        }
    } else {
        p.G = 1;
    }

    p.wg_size = p.G * p.L;
    p.num_wg = (batch + p.G - 1) / p.G;
    p.scope = (p.L == 32) ? PotrfScope::SubGroup : PotrfScope::WorkGroup;
    p.slm_total = potrf_hole_padded(static_cast<std::size_t>(p.G) * p.slm_per_matrix);
    p.fits = (p.slm_total <= slm_budget) && (p.wg_size <= max_wg);

    // Under Scope::WorkGroup the phase barriers are work-group barriers, which is
    // correct only when the work-group holds exactly one matrix.
    if (p.scope == PotrfScope::WorkGroup && p.G != 1) {
        throw std::logic_error("potrf_cta: Scope::WorkGroup with G != 1 is a race by construction");
    }
    return p;
}

}  // namespace

template <typename T>
int potrf_cta_max_n_for_slm(std::size_t slm_budget_bytes) {
    using C = PotrfCtaConst<T>;
    using DM = sycl_device::DevMap<T>;
    constexpr std::size_t sz_d = sizeof(typename DM::type);
    constexpr std::size_t sz_r = sizeof(typename DM::real);

    // Monotone in n, so a linear walk is exact; 4096 is a bound, not a capability.
    // The pad is applied HERE too, so this ceiling and the launcher's p.fits test
    // are one predicate. The `break` is load-bearing: potrf_hole_padded is NOT
    // monotone, and supports() advertises `order <= cta_max_n`, a contiguous range.
    int best = 0;
    for (int n = 1; n <= 4096; ++n) {
        if (potrf_hole_padded(potrf_slm_per_matrix(n, C::NB, C::TS, sz_d, sz_r))
            > slm_budget_bytes) break;
        best = n;
    }
    return best;
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

// The launch geometry, for tests. See potrf_native.hh for why this exists.
template <typename T>
unsigned potrf_cta_debug_launch(Queue& ctx, int n, int batch) {
    using C = PotrfCtaConst<T>;
    using DM = sycl_device::DevMap<T>;
    const auto dev = ctx.device();
    const std::size_t local_mem = dev.get_property(DeviceProperty::LOCAL_MEM_SIZE);
    const std::size_t budget = (local_mem > 4096) ? (local_mem - 4096) : 0;
    const int max_wg = static_cast<int>(dev.get_property(DeviceProperty::MAX_WORK_GROUP_SIZE));
    const auto p = potrf_cta_launch_params<C::NB, C::TS>(
        n, batch, sizeof(typename DM::type), sizeof(typename DM::real), budget, max_wg);
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
    // std::complex is re-typed to the POD device scalar HERE, at the pointer
    // boundary: its Annex-G operator* costs an isnan branch and a __mulsc3 call.
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
    const int num_wg = p.num_wg;
    int32_t* info_ptr = info.data();

    // Pad the TILE accessor rather than adding a fifth, unused one: an unused
    // local_accessor is a plausible dead-code elimination.
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
                    // Sub-group-uniform, and this scope uses only the
                    // sub-group's own barriers, so returning here strands nobody.
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

                // Built EXPLICITLY from data_ptr() + b*stride, never MatrixView::operator()(Slice,Slice):
                // its 6-arg constructor defaults stride to ld*cols when 0 is passed, after which
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
                         Span<int32_t> info_out) {
    using C = PotrfCtaConst<T>;
    using DM = sycl_device::DevMap<T>;
    constexpr std::size_t sz_d = sizeof(typename DM::type);
    constexpr std::size_t sz_r = sizeof(typename DM::real);

    const int n = static_cast<int>(A.rows());
    const int batch = static_cast<int>(A.batch_size());

    // supports()'s gates, re-applied: this entry point is reachable without the table.
    if (A.rows() != A.cols()) {
        throw std::invalid_argument("potrf_cta: A must be square");
    }
    if (n < 1 || batch < 1) {
        throw std::invalid_argument("potrf_cta: degenerate extents");
    }
    if (A.is_heterogeneous()) {
        // One launch covers the batch with one (order, ld, stride) tuple and reads
        // the CAPACITY extents, so per-item active dims would factorise the wrong order.
        throw std::invalid_argument("potrf_cta: heterogeneous batch is not supported");
    }
    const auto dev = ctx.device();
    if (dev.type != DeviceType::GPU) {
        throw std::invalid_argument("potrf_cta: GPU queues only");
    }
    if (!dev.supports_sub_group_size(32)) {
        // ENUMERATED, never get_property(MAX_SUB_GROUP_SIZE) >= 32: that returns the
        // FIRST supported size, so the weak test refuses a {8,16,32} device and
        // ACCEPTS a {64} one -- a launch abort under reqd_sub_group_size(32).
        throw std::runtime_error(
            "potrf_cta: device does not offer sub-group size 32, which the kernel requires");
    }

    const std::size_t local_mem = dev.get_property(DeviceProperty::LOCAL_MEM_SIZE);
    const std::size_t budget = (local_mem > 4096) ? (local_mem - 4096) : 0;
    const int max_wg = static_cast<int>(dev.get_property(DeviceProperty::MAX_WORK_GROUP_SIZE));

    const auto p = potrf_cta_launch_params<C::NB, C::TS>(n, batch, sz_d, sz_r, budget, max_wg);
    if (!p.fits) {
        throw std::invalid_argument(
            "potrf_cta: order " + std::to_string(n) +
            " does not fit this device's local memory (needs " +
            std::to_string(p.slm_total) + " B of " + std::to_string(budget) +
            " B); the ceiling for this type is " +
            std::to_string(potrf_cta_max_n_for_slm<T>(budget)));
    }

    BumpAllocator pool(workspace);
    // detail::info_target's rule, inlined so this TU need not include
    // src/linalg-impl.hh: an empty or SHORT caller span means "not requested" and
    // draws pool scratch instead, which keeps potrf_cta_buffer_size correct in both.
    Span<int32_t> info = (info_out.size() >= static_cast<std::size_t>(batch))
                             ? info_out
                             : potrf_cta_layout<T>(ctx, pool, batch);

    const bool upper = (uplo == Uplo::Upper);

    if (p.scope == PotrfScope::SubGroup) {
        return potrf_cta_launch<T, C::NB, C::TS, PotrfScope::SubGroup>(
            ctx, A, upper, info, p, n, batch);
    }
    return potrf_cta_launch<T, C::NB, C::TS, PotrfScope::WorkGroup>(
        ctx, A, upper, info, p, n, batch);
}

// Instantiation is per scalar type only, no Backend cross-product: the kernel has
// no Backend parameter, and a 3x device-compiled family is pure cost here.
#define BATCHLAS_POTRF_CTA_INSTANTIATE(T)                                                   \
    template int potrf_cta_max_n_for_slm<T>(std::size_t);                                   \
    template int potrf_cta_max_n<T>();                                                      \
    template unsigned potrf_cta_debug_launch<T>(Queue&, int, int);                          \
    template std::size_t potrf_cta_buffer_size<T>(Queue&,                                   \
                                                  const MatrixView<T, MatrixFormat::Dense>&); \
    template Event potrf_cta_dispatch<T>(Queue&, const MatrixView<T, MatrixFormat::Dense>&, \
                                         Uplo, Span<std::byte>, Span<int32_t>);

BATCHLAS_POTRF_CTA_INSTANTIATE(float)
BATCHLAS_POTRF_CTA_INSTANTIATE(double)
BATCHLAS_POTRF_CTA_INSTANTIATE(std::complex<float>)
BATCHLAS_POTRF_CTA_INSTANTIATE(std::complex<double>)

#undef BATCHLAS_POTRF_CTA_INSTANTIATE

}  // namespace sycl_potrf
}  // namespace batchlas
