// potri's local-memory-resident kernel: one work-group per order-m block, LAPACK's trti2 and lauu2
// on a tile padded to a multiple of TS. Its lower triangle holds M = L diag(L)^-1 (the multipliers),
// its upper one P = diag(L) L^-1 transposed, so neither overwrites the other; Upper runs the same
// code on L = U^H. Both O(m^3) phases accumulate TS x TS register tiles (one or two per work-item)
// and read 2 TS local values per TS^2 multiply-adds. evidence: docs/perf/potri.md#potri-the-algorithm
// The body is written on device LA v2.1 core (include/batchlas/device.hh) and the
// potri-private tile ops of src/ops/potri/potri_v2.hh.
//
// Invariants: acc[] tiles are indexed only by unrolled constants (a dynamic index moves them to
// local memory); every barrier is at the top level or under a work-group-uniform condition.

#include <src/ops/potri/potri_native.hh>
#include <batchlas/error.hh>

#include <src/sycl/device_scalar.hh>
#include <src/sycl/impl.hh>
#include <batchlas/device/detail/kernel_attrs.hh>
#include <batchlas/device/arith.hh>
#include <src/ops/potri/potri_v2.hh>
#include <src/queue.hh>
#include <src/util/resident_capacity.hh>
#include <src/util/batch_mask.hh>

#include <sycl/sycl.hpp>

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <string>

namespace batchlas::sycl_potri {

namespace {

namespace v2 = batchlas::device;

template <typename T> class PotriCtaKernel;

}  // namespace

template <typename T>
Event potri_cta(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, Uplo uplo, CtaMode mode, int nb,
                bool full, Span<int32_t> info, Stash<T> stash) {
    using DM = sycl_device::DevMap<T>;
    using D = typename DM::type;
    using R = typename DM::real;
    static_assert(sizeof(D) == sizeof(T), "device scalar must be layout-compatible");
    constexpr int TS = kCtaTs<T>;
    constexpr int NT = kCtaTiles;
    const int n = A.rows();
    const int batch = A.batch_size();
    if (n < 1 || batch < 1 || A.cols() != n || A.is_heterogeneous())
        throw batchlas::internal_error("potri_cta: a homogeneous batch of square blocks is required");
    const int bn = mode == CtaMode::Trtri ? std::min(std::max(nb, 1), n) : n;
    const int nblk = (n + bn - 1) / bn;
    const auto dev = ctx.device();
    const std::size_t budget = resident::device_slm_budget(dev.get_property(DeviceProperty::LOCAL_MEM_SIZE));
    // The kernel's own launch limit (its registers), not only the device's: at 181 registers
    // (complex<float>, sm_89) a block holds 352 work-items, so the ceiling's 384 would not launch.
    // Each item holds up to kCtaTiles tiles, so the smaller block still covers the ceiling.
    int max_wg = static_cast<int>(dev.get_property(DeviceProperty::MAX_WORK_GROUP_SIZE));
    try {
        const int kmax = static_cast<int>(impl::kernel_max_wg_size<PotriCtaKernel<T>>(ctx->get_context(), ctx->get_device()));
        if (kmax >= 32) max_wg = std::min(max_wg, kmax / 32 * 32);
    } catch (const sycl::exception&) {
    }
    if (!cta_fits(bn, TS, sizeof(D), sizeof(R), budget, max_wg))
        throw batchlas::internal_error("potri_cta: order " + std::to_string(bn) + " needs " +
                                       std::to_string(cta_slm_bytes(bn, TS, sizeof(D), sizeof(R))) +
                                       " B of local memory and " + std::to_string(cta_tiles(bn, TS)) +
                                       " tiles; the ceiling here is " + std::to_string(cta_max_n<T>(budget, max_wg)));
    const bool want_info = mode == CtaMode::Potri && !info.empty();
    if (want_info && info.size() < static_cast<std::size_t>(batch))
        throw batchlas::internal_error("potri_cta: info span shorter than the batch");
    const int wg = cta_wg(bn, TS, max_wg);
    const int mp = (bn + TS - 1) / TS * TS;
    const int lds = mp + 1;
    const std::size_t bytes = cta_slm_bytes(bn, TS, sizeof(D), sizeof(R));
    const std::size_t tile = (bytes - static_cast<std::size_t>(mp) * sizeof(R)) / sizeof(D);

    D* const a = reinterpret_cast<D*>(A.data_ptr());
    const std::ptrdiff_t lda = A.ld(), stride = A.stride();
    int32_t* const info_ptr = want_info ? info.data() : nullptr;
    const bool lower = uplo == Uplo::Lower;
    const int md = static_cast<int>(mode);
    const std::size_t groups = static_cast<std::size_t>(batch) * static_cast<std::size_t>(nblk);
    const bool stash_on = stash.slots != nullptr;
    if (stash_on && (mode == CtaMode::Potri || !stash.flags ||
                     (mode == CtaMode::Trtri ? stash.origin != 0 || stash.n != n
                                             : stash.origin < 0 || stash.origin + n != stash.n)))
        throw batchlas::internal_error("potri_cta: a stash needs flags, Trtri over whole items or Lauum on the last block");
    D* const slots = reinterpret_cast<D*>(stash.slots);
    int32_t* const flags = stash.flags;
    const int sn = stash.n, origin = stash.origin;
    // A's batch mask (the blocked driver's views carry their part's slice): one item per group, I1.
    const std::uint8_t* const mask = detail::mask_ptr(detail::call_mask("potri", A));
    const std::ptrdiff_t tri = static_cast<std::ptrdiff_t>(sn) * (sn + 1) / 2;

    ctx->submit([&](sycl::handler& h) {
        sycl::local_accessor<D, 1> St(sycl::range<1>(tile), h);
        sycl::local_accessor<R, 1> rdt(sycl::range<1>(static_cast<std::size_t>(mp)), h);
        impl::parallel_for<PotriCtaKernel<T>>(h, sycl::nd_range<1>(groups * wg, wg), [=](sycl::nd_item<1> it) {
            using namespace sycl_device;
            const v2::RtGroup<> ex(it);
            const v2::Thread th{};
            const int tid = ex.lid();
            const std::size_t g = it.get_group_linear_id();
            const std::ptrdiff_t item = static_cast<std::ptrdiff_t>(g / nblk);
            if (mask != nullptr && mask[item] == 0) return; // work-group uniform, before any collective
            const int j0 = static_cast<int>(g % nblk) * bn;
            const int m = sycl::min(bn, n - j0);
            const int mpw = (m + TS - 1) / TS * TS;
            const int ntile = cta_tiles(m, TS);
            D* const Ag = a + item * stride + static_cast<std::ptrdiff_t>(j0) * (lda + 1);

            if (stash_on) {
                // A zero-diagonal item is parked in its slot as packed uplo (Trtri) and put back (Lauum).
                D* const Ai = a + item * stride - static_cast<std::ptrdiff_t>(origin) * (lda + 1);
                auto Af = [&](int r, int c) -> D& { return Ai[r + static_cast<std::ptrdiff_t>(c) * lda]; };
                int z = -1;
                if (md == 1) {
                    z = v2::find_first(
                        ex, sn, [&](int i) { return dev_is_zero(Ai[static_cast<std::ptrdiff_t>(i) * (lda + 1)]); });
                    if (j0 == 0 && tid == 0) flags[item] = z + 1;
                } else if (flags[item] != 0) {
                    z = 0;
                }
                if (z >= 0) { // work-group uniform; no other group of this launch writes these columns
                    tpcopy(ex, lower, md == 1, sn, Af, slots + item * tri, md == 1 ? j0 : 0, md == 1 ? j0 + m : sn);
                    return;
                }
            }
            D* const S = &St[0];
            R* const rd = &rdt[0];
            auto s = [&](int r, int c) -> D& { return S[r + static_cast<std::ptrdiff_t>(c) * lds]; };
            auto g_at = [&](int r, int c) -> D& { return Ag[r + static_cast<std::ptrdiff_t>(c) * lda]; };

            // Stage the uplo triangle, read as L ((i, j) = the L coordinates), 4 positions per round
            // so their loads overlap: M and 1 / L(i, i) (Potri, Trtri) or X^T (Lauum); the rest := 0.
            struct Staged {
                int i, j;
                D v, dg;
            };
            for_each_owned_2d_batched<4>(
                ex, mpw, mpw,
                [&](bool live, int r, int c) {
                    Staged e;
                    e.i = live ? (lower ? r : c) : -1;
                    e.j = live ? (lower ? c : r) : 0;
                    const bool in = live && e.i >= e.j && e.i < m;
                    e.v = in ? g_at(r, c) : D{};
                    e.dg = in && md != 2 && e.i > e.j ? g_at(e.j, e.j) : dev_one<D>();
                    return e;
                },
                [&](const Staged& e) {
                    const int i = e.i, j = e.j;
                    if (i < j) return;
                    const bool in = i < m;
                    const D lv = lower ? e.v : dev_conj(e.v);
                    if (md == 2) {
                        s(j, i) = lv; // lauum's X, transposed into the upper triangle
                        if (i != j) s(i, j) = D{};
                    } else if (i == j) {
                        rd[i] = in ? R(1) / dev_real(lv) : R(0);
                        s(i, i) = in ? dev_one<D>() : D{}; // P(k, k) = 1, read as row k's last entry
                    } else {
                        s(i, j) = dev_div_real(lv, dev_real(e.dg)); // M(i, j) = L(i, j) / L(j, j)
                        s(j, i) = D{};
                    }
                });
            ex.barrier();

            int zero = -1;
            if (md == 0) {
                zero = v2::find_first(ex, m, [&](int i) { return dev_is_zero(g_at(i, i)); });
                if (ex.leader() && info_ptr) info_ptr[item] = zero + 1;
            }
            if (zero >= 0) return; // work-group uniform: no barrier is left unmatched

            int ti[NT], tj[NT];
            bool has[NT];
            BATCHLAS_UNROLL_FULL
            for (int q = 0; q < NT; ++q) {
                const int t = tid + q * wg;
                has[q] = t < ntile;
                const RowCol rc = tri_index(t);
                ti[q] = rc.r * TS;
                tj[q] = rc.c * TS;
            }
            v2::Tile<D, TS, TS> acc[NT];

            if (md != 2) {
                // trti2, right-looking by TS pivots: tile row K solves with M(K, K) in registers and
                // publishes P(K, j) at s(j, K); the tiles below take acc -= M(rows, K) P(K, cols).
                BATCHLAS_UNROLL_FULL
                for (int q = 0; q < NT; ++q)
                    v2::laset(acc[q], v2::Zero{}, v2::One{}, tj[q] - ti[q]);  // 1 on the global diagonal
                for (int kb = 0; kb < mpw; kb += TS) {
                    BATCHLAS_UNROLL_FULL
                    for (int q = 0; q < NT; ++q) {
                        if (!has[q] || ti[q] != kb) continue;
                        reg::trsm_llnu(th, [&](int x, int c) { return s(kb + x, kb + c); }, acc[q]);
                        reg::for_each(th, acc[q], kb, tj[q], [&](int i, int j, const D& v) {
                            if (j < i) s(j, i) = v;
                        });
                    }
                    ex.barrier();
                    BATCHLAS_UNROLL_FULL
                    for (int q = 0; q < NT; ++q) {
                        if (!has[q] || ti[q] <= kb || tj[q] > kb) continue;
                        const bool diag = tj[q] == kb; // P(K, K) is unit lower: zero above, one on it
                        reg::gemm_sub<TS>(
                            th, [&](int x, int c) { return s(ti[q] + x, kb + c); },
                            [&](int c, int y) { return !diag || y <= c ? s(tj[q] + y, kb + c) : D{}; }, acc[q]);
                    }
                }
                // X = diag(L)^-1 P as X^T in the upper triangle; the strict lower one (M) := 0.
                BATCHLAS_UNROLL_FULL
                for (int q = 0; q < NT; ++q) {
                    if (!has[q]) continue;
                    reg::for_each(th, acc[q], ti[q], tj[q], [&](int i, int j, const D& v) {
                        if (i > j) {
                            s(j, i) = dev_mul_real(v, rd[i]);
                            s(i, j) = D{};
                        } else if (i == j) {
                            s(i, i) = dev_from_real<D>(rd[i]);
                        }
                    });
                }
                ex.barrier();
                if (md == 1) {
                    v2::for_each_owned_2d(ex, m, m, [&](int r, int c) {
                        if (lower ? r < c : r > c) return;
                        // X(r, c) sits at s(c, r); Upper wants (L^-1)^H.
                        g_at(r, c) = lower ? s(c, r) : dev_conj(s(r, c));
                    });
                    return;
                }
            }

            // lauu2: Y = X^H X as rank-1 steps, Y(i, j) = sum_k conj(X(k, i)) X(k, j); X(k, x) = s(x, k),
            // zero for x > k.
            BATCHLAS_UNROLL_FULL
            for (int q = 0; q < NT; ++q) {
                v2::fill(acc[q], D{});
                if (!has[q]) continue;
                for (int k = ti[q]; k < mpw; ++k) {
                    v2::Frag<D, TS> xi, xj;
                    v2::load(xi, [&](int x) { return dev_conj(s(ti[q] + x, k)); });
                    v2::load(xj, [&](int y) { return s(tj[q] + y, k); });
                    reg::ger(th, xi, xj, acc[q]);
                }
            }
            ex.barrier();
            // Y into the strict lower triangle (no longer read) and rd (its real diagonal).
            BATCHLAS_UNROLL_FULL
            for (int q = 0; q < NT; ++q) {
                if (!has[q]) continue;
                reg::for_each(th, acc[q], ti[q], tj[q], [&](int i, int j, const D& v) {
                    if (i > j) s(i, j) = v;
                    else if (i == j) rd[i] = dev_real(v);
                });
            }
            ex.barrier();

            // Y is A^-1 (Potri) or the product (Lauum) for either uplo; full fills the other triangle.
            const bool mirror = full && md == 0;
            v2::for_each_owned_2d(ex, m, m, [&](int r, int c) {
                const bool owned = lower ? r >= c : r <= c;
                if (!owned && !mirror) return;
                g_at(r, c) = r > c ? s(r, c) : r == c ? dev_from_real<D>(rd[r]) : dev_conj(s(c, r));
            });
        });
    });
    return ctx.get_event();
}

#define BATCHLAS_POTRI_CTA_INSTANTIATE(T)                                                                    \
    template Event potri_cta<T>(Queue&, const MatrixView<T, MatrixFormat::Dense>&, Uplo, CtaMode, int, bool, \
                                Span<int32_t>, Stash<T>);

BATCHLAS_POTRI_CTA_INSTANTIATE(float)
BATCHLAS_POTRI_CTA_INSTANTIATE(double)
BATCHLAS_POTRI_CTA_INSTANTIATE(std::complex<float>)
BATCHLAS_POTRI_CTA_INSTANTIATE(std::complex<double>)

#undef BATCHLAS_POTRI_CTA_INSTANTIATE

}  // namespace batchlas::sycl_potri
