// Native batched GEQRF: the TINY tier -- square m == n <= 32, one matrix per
// SubGroupPartition<N>, the whole matrix in registers between the load and the store.
// evidence: docs/perf/qr.md#the-tiny-tier-wp6--p1-square-n--32-in-registers

#include "geqrf_native.hh"
#include "geqrf_tiny_device.hh"
#include "tiny_device.hh"

#include "../queue.hh"
#include "../util/resident_capacity.hh"
#include "../util/template-instantiations.hh"

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>

// Outside the anonymous namespace: a kernel name may not name an internal-linkage entity.
namespace batchlas {
template <typename T, int N, int C> class GeqrfTinyKernel;
}

namespace batchlas {
namespace sycl_geqrf {

namespace {

namespace gn = ::batchlas::geqrf_native;
namespace tn = ::batchlas::tiny_native;

// THE LAUNCH GATES. The register table itself lives next to the chunk rule that reads it;
// a second copy here is how a MODEL came to choose the chunk width while a PROBE gated the
// launch. evidence: docs/perf/qr.md#the-tiny-tier-register-table
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

// GATE 1, the HARD one: regs x work-group size over the per-block register file ABORTS the
// launch. Both operands are named so raising either is deliberate; GATE 2 is what binds.
constexpr int kGeqrfTinyMaxWg = gn::kGeqrfTinyWg;
constexpr int kGeqrfTinyWorstRegs = geqrf_tiny_worst_regs();
static_assert(kGeqrfTinyMaxWg * kGeqrfTinyWorstRegs <= 65536,
              "geqrf tiny: regs x wg exceeds the per-block register file -- re-run "
              "scripts/register_probe.sh before raising either operand");

// GATE 2: every SHIPPED cell keeps R1's resident::kMinBlocksPerSm blocks per SM, so a
// regression fails to COMPILE rather than quietly costing occupancy. GATE 3, in the same
// breath: the chunk width each row was probed at still re-derives from that row.
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

// No request lands near the (47104, 49664] launch hole, so none of geqrf_cta.cc's
// hole-padding machinery is carried here; these keep that true.
static_assert(geqrf_tiny_wg_bytes<std::complex<double>, 8>() < 47104,
              "tiny tile approaching the 48 KB launch hole");
static_assert(geqrf_tiny_wg_bytes<double, 32>() < 47104,
              "tiny tile approaching the 48 KB launch hole");

template <typename T, int N>
Event geqrf_tiny_launch(Queue& ctx, T* a_ptr, int ld, int stride, int n, int batch,
                        T* tau_ptr) {
    // Re-typed HERE: std::complex's Annex-G operator* is an isnan branch plus a libcall.
    using DM = sycl_device::DevMap<T>;
    using D = typename DM::type;
    using R = gn::real_of<D>;
    static_assert(sizeof(D) == sizeof(T), "device scalar must be layout-compatible");

    constexpr int C = gn::geqrf_tiny_chunk<D, N>();
    constexpr int SLDA = gn::geqrf_tiny_slda(C);
    constexpr int M = gn::geqrf_tiny_matrices_per_wg<N>();
    constexpr int WG = M * N;
    constexpr std::size_t kPerMatrix = gn::geqrf_tiny_slm_elems(N, C);

    static_assert(WG == gn::kGeqrfTinyWg, "the work-group must stay a whole number of warps");
    static_assert(WG % tn::kTinySubGroupSize == 0, "the work-group must be whole sub-groups");
    static_assert(tn::tiny_n_is_legal(N), "N must divide the sub-group");

    D* const ap = reinterpret_cast<D*>(a_ptr);
    D* const tp = reinterpret_cast<D*>(tau_ptr);
    const int num_wg = (batch + M - 1) / M;

    ctx->submit([&](sycl::handler& h) {
        sycl::local_accessor<D, 1> scratch(
            sycl::range<1>(static_cast<std::size_t>(M) * kPerMatrix), h);

        h.parallel_for<GeqrfTinyKernel<T, N, C>>(
            sycl::nd_range<1>(sycl::range<1>(static_cast<std::size_t>(num_wg) *
                                             static_cast<std::size_t>(WG)),
                              sycl::range<1>(static_cast<std::size_t>(WG))),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                const auto sg = it.get_sub_group();
                const auto part = make_partition<N>(sg);
                const int lane = static_cast<int>(part.get_local_linear_id());
                const int slot = tn::tiny_partition_id(sg, part);
                const int prob = static_cast<int>(it.get_group_linear_id()) * M + slot;

                // NO EARLY RETURN. `if (prob >= batch) return;` is partition-uniform but NOT
                // sub-group-uniform, so at N < 32 a later group_barrier(sg) is undefined. A
                // dead partition carries the identity and suppresses its store. Guarded by
                // source text (GeqrfTinySource.KernelBodyHasNoEarlyReturn), not numerically.
                const bool live = prob < batch;
                const bool row_live = live && lane < n;
                const std::ptrdiff_t off =
                    live ? static_cast<std::ptrdiff_t>(prob) * stride : std::ptrdiff_t(0);

                D rA[N];  // top level, never a parameter: tiny_device.hh invariant 1
                tn::tiny_load_full<D, N>(rA, ap + off, ld, lane, n, row_live);

                D* const sP = &scratch[0] + static_cast<std::ptrdiff_t>(slot) *
                                                static_cast<std::ptrdiff_t>(kPerMatrix);
                D* const sy = sP + static_cast<std::ptrdiff_t>(N) * SLDA;

                D my_tau = D{};  // one register, stored once N-wide rather than N times

                // unroll(FULL), not the bare hint, on EVERY loop whose index reaches rA: the
                // hint loses to a code-size threshold at the widest cells, rA[j] becomes a
                // DYNAMIC index, and the array leaves the register file with ZERO spill.
                // evidence: docs/perf/qr.md#the-stack-frame-is-the-gate-for-this-kernel-not-the-spill-counter
#pragma clang loop unroll(full)
                for (int j = 0; j < N; ++j) {
                    // n is one scalar for the whole batch (supports() refuses a heterogeneous
                    // view), so every predicate below is uniform and so is the barrier sequence.
                    if (j >= n) continue;

                    const D alpha = tn::tiny_bcast<D, N>(part, rA[j], static_cast<uint32_t>(j));

                    // Order-symmetric, so every lane leaves with a BIT-IDENTICAL smax and ssq
                    // and needs no publish/broadcast pair. Widening either leaks across
                    // partitions. evidence: docs/perf/qr.md#the-two-partition-butterflies
                    R smax = (lane >= j) ? gn::dev_absmax<D>(rA[j]) : R(0);
                    smax = gn::geqrf_tiny_reduce_fmax<N>(part, smax);

                    R ssq = R(0);
                    if (lane > j && smax > R(0)) ssq = gn::dev_abs2_scaled<D>(rA[j], smax);
                    ssq = gn::geqrf_tiny_reduce_sum<N>(part, ssq);

                    const gn::LarfgScalars<D> hh =
                        gn::geqrf_larfg_scalars<D>(alpha, smax, ssq);

                    // NO `if (hh.identity) continue;`: the identity case is arithmetically
                    // inert (tau = 0, beta = alpha, vfactor = 1), and omitting the branch is
                    // what keeps the barrier sequence sub-group-uniform.
                    if (lane == j) {
                        rA[j] = hh.beta;
                        my_tau = hh.tau;
                    } else if (lane > j) {
                        rA[j] = hh.use_mul
                                    ? sycl_device::dev_mul(rA[j], hh.vfactor)
                                    : sycl_device::dev_div(rA[j], hh.vfactor);
                    }

                    // One select drives the whole apply: it zeroes eliminated rows, supplies
                    // the implicit v(j) = 1, and removes every lane guard below. Dropping the
                    // `lane < j` term lets eliminated rows re-enter the reflector.
                    const D vr = (lane < j)
                                     ? D{}
                                     : ((lane == j) ? sycl_device::dev_one<D>() : rA[j]);
                    const D cvr = sycl_device::dev_conj(vr);
                    // CONJ(tau): zgeqr2's convention, the one ormqr/orgqr/ormbr/sy2sb read.
                    // evidence: docs/perf/qr.md#a-residual-test-cannot-guard-a-convention
                    const D ctau = sycl_device::dev_conj(hh.tau);

#pragma clang loop unroll(full)
                    for (int k0 = 0; k0 < N; k0 += C) {
                        if (k0 + C <= j + 1) continue;  // wholly at or below the diagonal
                        if (k0 >= n) continue;          // wholly past the runtime cap

                        // P1: the elementwise product into the tile.
#pragma clang loop unroll(full)
                        for (int kk = 0; kk < C; ++kk) {
                            const int k = k0 + kk;
                            const bool act = (k > j) && (k < n);
                            sP[lane * SLDA + kk] =
                                act ? sycl_device::dev_mul(cvr, rA[k]) : D{};
                        }
                        sycl::group_barrier(sg);

                        // P2: lane l sums column l in ROW order -- LAPACK's association.
                        if (lane < C) {
                            D acc = D{};
#pragma unroll
                            for (int r = 0; r < N; ++r) {
                                if (r >= j) acc = gn::dev_add(acc, sP[r * SLDA + lane]);
                            }
                            sy[lane] = sycl_device::dev_mul(ctau, acc);
                        }
                        sycl::group_barrier(sg);

                        // P3: one address per column, a broadcast. The two barriers above
                        // cover all three hazards ONLY because sy is a separate array; their
                        // count and placement are guarded by source text, not numerically.
                        // evidence: docs/perf/qr.md#break-sweeps-the-tiny-tier
#pragma clang loop unroll(full)
                        for (int kk = 0; kk < C; ++kk) {
                            const int k = k0 + kk;
                            if ((k > j) && (k < n)) {
                                rA[k] = sycl_device::dev_sub(
                                    rA[k], sycl_device::dev_mul(vr, sy[kk]));
                            }
                        }
                    }
                    // No barrier between columns: column j+1's butterflies read rA[j+1], which
                    // this lane updated itself in P3 -- a register-private dependence.
                }

                tn::tiny_store_full<D, N>(rA, ap + off, ld, lane, n, row_live);
                if (row_live) {
                    tp[static_cast<std::ptrdiff_t>(prob) * n + lane] = my_tau;
                }
            });
    });
    return ctx.get_event();
}

}  // namespace

template <typename T>
int geqrf_tiny_max_n_for_slm(std::size_t slm_budget_bytes) {
    // A walk with a break, not a max: per-work-group bytes are NOT monotone in N, and
    // supports() advertises the contiguous range n <= max_n, so the ceiling must be the
    // largest N at which every smaller bucket also fits.
    const std::size_t slice = resident::occupancy_budget(slm_budget_bytes);
    constexpr int ceiling = geqrf_tiny_type_ceiling<T>();

    if (geqrf_tiny_wg_bytes<T, 8>() > slice) return 0;
    if (ceiling < 16 || geqrf_tiny_wg_bytes<T, 16>() > slice) return 8;
    if (ceiling < 32 || geqrf_tiny_wg_bytes<T, 32>() > slice) return 16;
    return 32;
}

template <typename T>
int geqrf_tiny_max_n() {
    return geqrf_tiny_max_n_for_slm<T>(gn::kGeqrfTinyReferenceSlm);
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

    // A forced route that supports() rejects falls through to automatic() and silently
    // runs the vendor, so every gate supports() applies is RE-APPLIED here and throws.
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

    // Explicit throw, never a silent leading-submatrix solve. The N = 32 arm sits under
    // `if constexpr` so a ceiling-16 type does not INSTANTIATE an unreachable kernel.
    const int bucket = tn::tiny_bucket_ge(n);
    if (bucket == 8) {
        return geqrf_tiny_launch<T, 8>(ctx, a_ptr, ld, stride, n, batch, tau_ptr);
    }
    if (bucket == 16) {
        return geqrf_tiny_launch<T, 16>(ctx, a_ptr, ld, stride, n, batch, tau_ptr);
    }
    if constexpr (geqrf_tiny_type_ceiling<T>() >= 32) {
        if (bucket == 32) {
            return geqrf_tiny_launch<T, 32>(ctx, a_ptr, ld, stride, n, batch, tau_ptr);
        }
    }
    throw batchlas::unsupported(
        "geqrf_tiny: order " + std::to_string(n) +
        " has no compile-time bucket for this scalar type; the ladder is {8, 16, 32}");
}

// Per scalar type only; the ladder above pulls the <T, N, C> cross-product in implicitly.
#define BATCHLAS_GEQRF_TINY_INSTANTIATE(T)                                                     \
    template int geqrf_tiny_max_n_for_slm<T>(std::size_t);                                     \
    template int geqrf_tiny_max_n<T>();                                                        \
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
