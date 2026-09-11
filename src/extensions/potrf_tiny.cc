// Native batched POTRF, the TINY tier: right-looking unblocked Cholesky with one matrix
// per SubGroupPartition<N>, lane r owning row r in registers. Zero local memory, zero
// barriers, every cross-lane value a sub-group shuffle; the shared load/pad/store helpers
// are in tiny_device.hh. It stays in EXTENSIONS_CTA_SOURCES with potrf_cta.cc, the same
// device-code cluster. evidence: docs/perf/potrf.md#the-tiny-tier
//
// Why no local-memory scratch. Each of the two things a column publishes -- the pivot and
// column j of L -- has exactly ONE producer lane, so an indexed shuffle delivers it. The
// decisive argument is portability: sg_compat.hh's group_barrier(part) expands to NOTHING
// off NVPTX, so a publish/read pair ordered only by it is silently unordered elsewhere.

#include "potrf_native.hh"
#include "tiny_device.hh"

#include "../queue.hh"
#include "../util/resident_capacity.hh"
#include "../util/template-instantiations.hh"

#include <batchlas/util/mempool.hh>

#include <complex>
#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>

namespace batchlas {

// Kernel name tag; outside the anonymous namespace so it names no internal-linkage entity.
template <typename T, int N>
class PotrfTinyKernel;

namespace sycl_potrf {

namespace {

using tiny_native::kTinySubGroupSize;
using tiny_native::kTinySubGroups;
using tiny_native::kTinyWgSize;

// A flat compile-time constant, not a budget walk: the tier owns no local memory.
// evidence: docs/perf/potrf.md#the-tiny-tier
template <typename T>
struct PotrfTinyCap { static constexpr int kMaxN = 32; };
template <>
struct PotrfTinyCap<std::complex<double>> { static constexpr int kMaxN = 16; };

// A LAUNCH gate: violating it aborts the enqueue (trsm_native.cc:113). The gate that
// actually bites here is the probe's STACK FRAME column. evidence: docs/perf/potrf.md#the-tiny-tier
constexpr int kTinyWorstProbedRegs = 176;
static_assert(kTinyWgSize * kTinyWorstProbedRegs <= 65536,
              "kTinyWgSize x kTinyWorstProbedRegs exceeds the per-block register file; "
              "re-run scripts/register_probe.sh before raising either");

// Matrices per work-group, through the shared helper. The byte argument is a nominal 1
// against an unbounded budget because this tier owns no local memory.
inline int potrf_tiny_matrices_per_wg(int N, int max_wg) {
    return resident::pack_matrices_per_wg(/*bytes_per_matrix=*/1, /*lanes_per_matrix=*/N,
                                          /*wg_slm_budget_bytes=*/~std::size_t(0), max_wg,
                                          /*target_wg_size=*/kTinyWgSize,
                                          /*max_pack=*/kTinyWgSize / N);
}

template <typename T, int N>
Event potrf_tiny_launch(Queue& ctx,
                        const MatrixView<T, MatrixFormat::Dense>& A,
                        bool upper,
                        Span<int32_t> info,
                        int n, int batch, int per_wg) {
    // std::complex is re-typed to the POD device scalar at the pointer boundary: its
    // Annex-G operator* costs an isnan branch and a __mulsc3 call in device code.
    using DM = sycl_device::DevMap<T>;
    using D = typename DM::type;
    using R = typename DM::real;
    static_assert(sizeof(D) == sizeof(T), "device scalar must be layout-compatible");
    static_assert(tiny_native::tiny_n_is_legal(N),
                  "N must divide the sub-group: SubGroupPartition<P> bases a chunk at "
                  "(lane/P)*P, so a P that does not divide 32 shuffles off the end");

    D* a_ptr = reinterpret_cast<D*>(A.data_ptr());
    const int ld = A.ld();
    const int stride_a = A.stride();
    int32_t* info_ptr = info.data();

    const int wg_size = per_wg * N;
    const int num_wg = (batch + per_wg - 1) / per_wg;

    // A Hermitian diagonal's imaginary part is contractually ignored; for a real type
    // the transform is the identity.
    constexpr bool real_diag = DM::is_complex;

    ctx->submit([&](sycl::handler& h) {
        h.parallel_for<PotrfTinyKernel<T, N>>(
            sycl::nd_range<1>(sycl::range<1>(static_cast<std::size_t>(num_wg) * wg_size),
                              sycl::range<1>(wg_size)),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kTinySubGroupSize)]] {
                const auto sg = it.get_sub_group();
                const auto part = make_partition<N>(sg);
                const int lane = static_cast<int>(part.get_local_linear_id());
                const int wg_id = static_cast<int>(it.get_group_linear_id());
                const int prob_id = wg_id * per_wg + tiny_native::tiny_partition_id<N>(sg, part);

                // NO EARLY RETURN -- the in-tree `if (prob_id >= nb) return;` idiom is
                // not copied here; see tiny_device.hh's third invariant.
                const bool live = (prob_id < batch);
                const bool row_live = live && (lane < n);

                const D* __restrict Ag =
                    a_ptr + static_cast<std::ptrdiff_t>(live ? prob_id : 0) * stride_a;

                D rA[N];
                if (upper) {
                    tiny_native::tiny_load_upper<D, N>(rA, Ag, ld, lane, row_live, real_diag);
                } else {
                    tiny_native::tiny_load_lower<D, N>(rA, Ag, ld, lane, row_live, real_diag);
                }

                // `linfo` is partition-uniform because `akk` is a broadcast.
                bool alive = live;
                int32_t linfo = 0;

                // The order guard is a PREDICATE, never a `break`: a runtime break
                // defeats the unroll at N >= 16, after which rA[j] is a dynamic index.
                // evidence: docs/perf/potrf.md#the-tiny-tier
#pragma unroll
                for (int j = 0; j < N; ++j) {
                    const bool col_live = (j < n);

                    const R akk = select_from_group(part, sycl_device::dev_real(rA[j]),
                                                    static_cast<uint32_t>(j));

                    // `!(akk > 0)`, not `akk <= 0`, so NaN is rejected too, as LAPACK does.
                    const bool bad = !(akk > R(0));
                    if (alive && col_live && bad) {
                        linfo = j + 1;   // 1-based; sticky, so FIRST FAILURE WINS
                        alive = false;
                    }

                    // EVERY failure effect below is a value substitution, never control
                    // flow: `alive` is partition-uniform but NOT sub-group-uniform, so an
                    // `if (alive) { ...shuffle... }` would wrap a collective in divergence.
                    const bool act = alive && col_live;
                    const R dkk = act ? sycl::sqrt(akk) : R(1);

                    // NOT rsqrt: rsqrt.approx is not the reference.
                    const R rinv = R(1) / dkk;

                    // tiny_select, not `?:` -- see tiny_device.hh on why a ternary over a
                    // 16-byte aggregate stops the array being promotable.
                    rA[j] = tiny_native::tiny_select(
                        act,
                        tiny_native::tiny_select(lane == j,
                                                 sycl_device::dev_from_real<D>(dkk),
                                                 sycl_device::dev_mul_real(rA[j], rinv)),
                        rA[j]);

#pragma unroll
                    for (int k = j + 1; k < N; ++k) {
                        // The lane guard is INSIDE the collective, never around it.
                        const D vk = tiny_native::tiny_bcast<D>(part, rA[j],
                                                                static_cast<uint32_t>(k));
                        const D upd = sycl_device::dev_sub(
                            rA[k],
                            sycl_device::dev_mul(rA[j], sycl_device::dev_conj(vk)));

                        // `k <= lane` keeps rA[k] for k > lane exactly zero for the
                        // kernel's whole life, which is what makes the padding inert.
                        rA[k] = tiny_native::tiny_select(act && k < n && k <= lane,
                                                         upd, rA[k]);
                    }
                }

                D* __restrict Aout =
                    a_ptr + static_cast<std::ptrdiff_t>(live ? prob_id : 0) * stride_a;
                if (upper) {
                    tiny_native::tiny_store_upper<D, N>(rA, Aout, ld, lane, row_live);
                } else {
                    tiny_native::tiny_store_lower<D, N>(rA, Aout, ld, lane, row_live);
                }

                if (live && part.leader()) info_ptr[prob_id] = linfo;
            });
    });

    return ctx.get_event();
}

template <typename T>
Span<int32_t> potrf_tiny_layout(Queue& ctx, BumpAllocator& pool, int batch) {
    return pool.allocate<int32_t>(ctx, static_cast<std::size_t>(batch));
}

}  // namespace

template <typename T>
int potrf_tiny_max_n() {
    return PotrfTinyCap<T>::kMaxN;
}

template <typename T>
std::size_t potrf_tiny_buffer_size(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A) {
    // NOT zero: an empty or short caller `info` span draws `batch` int32s of pool
    // scratch, and potrf_buffer_size diagnoses "unimplemented" by native_need == 0.
    const int batch = A.batch_size();
    return workspace_bytes([&](BumpAllocator& p) {
        return potrf_tiny_layout<T>(ctx, p, batch);
    });
}

template <typename T>
unsigned potrf_tiny_debug_launch(Queue& ctx, int n) {
    const auto dev = ctx.device();
    if (dev.type != DeviceType::GPU) return 0u;
    const int N = tiny_native::tiny_bucket_ge(n);
    if (N == 0 || n > potrf_tiny_max_n<T>()) return 0u;
    const int max_wg = static_cast<int>(dev.get_property(DeviceProperty::MAX_WORK_GROUP_SIZE));
    const int per_wg = potrf_tiny_matrices_per_wg(N, max_wg);
    const int wg = per_wg * N;
    if (wg % kTinySubGroupSize != 0) return 0u;
    const int S = wg / kTinySubGroupSize;                 // sub-groups per work-group
    const int G = per_wg / S;                             // partitions per sub-group
    return (static_cast<unsigned>(S) << 16) | static_cast<unsigned>(G);
}

template <typename T>
Event potrf_tiny_dispatch(Queue& ctx,
                          const MatrixView<T, MatrixFormat::Dense>& A,
                          Uplo uplo,
                          Span<std::byte> workspace,
                          Span<int32_t> info_out) {
    const int n = static_cast<int>(A.rows());
    const int batch = static_cast<int>(A.batch_size());

    // supports()'s gates, re-applied: a forced route that supports() rejects falls back
    // to automatic() and silently runs the vendor, so this entry point throws instead.
    if (A.rows() != A.cols()) {
        throw batchlas::invalid_argument("potrf_tiny: A must be square");
    }
    if (n < 1 || batch < 1) {
        throw batchlas::invalid_argument("potrf_tiny: degenerate extents");
    }
    if (A.is_heterogeneous()) {
        throw batchlas::invalid_argument("potrf_tiny: heterogeneous batch is not supported");
    }
    const auto dev = ctx.device();
    if (dev.type != DeviceType::GPU) {
        throw batchlas::invalid_argument("potrf_tiny: GPU queues only");
    }
    if (!dev.supports_sub_group_size(32)) {
        // Enumerated, never MAX_SUB_GROUP_SIZE >= 32, which reports entry [0].
        throw batchlas::unsupported(
            "potrf_tiny: device does not offer sub-group size 32, which the kernel requires");
    }
    const int cap = potrf_tiny_max_n<T>();
    if (n > cap) {
        throw batchlas::invalid_argument(
            "potrf_tiny: order " + std::to_string(n) +
            " is above this type's register ceiling of " + std::to_string(cap));
    }
    const int max_wg = static_cast<int>(dev.get_property(DeviceProperty::MAX_WORK_GROUP_SIZE));
    if (max_wg < kTinyWgSize) {
        throw batchlas::unsupported(
            "potrf_tiny: the device's maximum work-group size is below the tier's " +
            std::to_string(kTinyWgSize));
    }

    BumpAllocator pool(workspace);
    // detail::info_target's rule, inlined to avoid including src/linalg-impl.hh: an
    // empty or short caller span means "not requested" and draws pool scratch instead.
    Span<int32_t> info = (info_out.size() >= static_cast<std::size_t>(batch))
                             ? info_out
                             : potrf_tiny_layout<T>(ctx, pool, batch);

    const bool upper = (uplo == Uplo::Upper);
    const int N = tiny_native::tiny_bucket_ge(n);

    // Only the per-type OUTER entry points are explicitly instantiated; this switch
    // pulls the three N in (trsm_native.cc's discipline).
    switch (N) {
        case 8:
            return potrf_tiny_launch<T, 8>(ctx, A, upper, info, n, batch,
                                           potrf_tiny_matrices_per_wg(8, max_wg));
        case 16:
            return potrf_tiny_launch<T, 16>(ctx, A, upper, info, n, batch,
                                            potrf_tiny_matrices_per_wg(16, max_wg));
        case 32:
            if constexpr (PotrfTinyCap<T>::kMaxN >= 32) {
                return potrf_tiny_launch<T, 32>(ctx, A, upper, info, n, batch,
                                                potrf_tiny_matrices_per_wg(32, max_wg));
            }
            break;
        default:
            break;
    }
    throw batchlas::invalid_argument(
        "potrf_tiny: order " + std::to_string(n) + " has no register bucket for this type");
}

// Per scalar type only, no Backend cross-product: the kernel has no Backend parameter.
#define BATCHLAS_POTRF_TINY_INSTANTIATE(T)                                                   \
    template int potrf_tiny_max_n<T>();                                                      \
    template unsigned potrf_tiny_debug_launch<T>(Queue&, int);                               \
    template std::size_t potrf_tiny_buffer_size<T>(Queue&,                                   \
                                                   const MatrixView<T, MatrixFormat::Dense>&); \
    template Event potrf_tiny_dispatch<T>(Queue&, const MatrixView<T, MatrixFormat::Dense>&, \
                                          Uplo, Span<std::byte>, Span<int32_t>);

BATCHLAS_POTRF_TINY_INSTANTIATE(float)
BATCHLAS_POTRF_TINY_INSTANTIATE(double)
BATCHLAS_POTRF_TINY_INSTANTIATE(std::complex<float>)
BATCHLAS_POTRF_TINY_INSTANTIATE(std::complex<double>)

#undef BATCHLAS_POTRF_TINY_INSTANTIATE

}  // namespace sycl_potrf
}  // namespace batchlas
