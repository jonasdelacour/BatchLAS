// potri's local-memory-resident kernel: one work-group per order-m block, LAPACK's trti2 and lauu2
// on a tile padded to a multiple of TS. Its lower triangle holds M = L diag(L)^-1 (the multipliers),
// its upper one P = diag(L) L^-1 transposed, so neither overwrites the other; Upper runs the same
// code on L = U^H. Both O(m^3) phases accumulate TS x TS register tiles (one or two per work-item)
// and read 2 TS local values per TS^2 multiply-adds. evidence: docs/perf/potri.md#potri-the-algorithm
//
// Invariants: acc[][][] is indexed only by unrolled constants (a dynamic index moves it to local
// memory); every barrier is at the top level or under a work-group-uniform condition.

#include <src/ops/potri/potri_native.hh>
#include <batchlas/error.hh>

#include <src/sycl/device_scalar.hh>
#include <src/sycl/impl.hh>
#include <src/sycl/kernel_attrs.hh>
#include <src/queue.hh>
#include <src/util/resident_capacity.hh>

#include <sycl/sycl.hpp>

#include <algorithm>
#include <climits>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <string>

namespace batchlas::sycl_potri {

namespace {

template <typename T> class PotriCtaKernel;

// (e % m, e / m) for e = start, start + step, ... without a division per step.
struct Walk {
    int r, c, dr, dc, m;
    Walk(int start, int step, int m_) : r(start % m_), c(start / m_), dr(step % m_), dc(step / m_), m(m_) {}
    void next() {
        r += dr;
        c += dc;
        if (r >= m) { r -= m; ++c; }
    }
};

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
    const int max_wg = static_cast<int>(dev.get_property(DeviceProperty::MAX_WORK_GROUP_SIZE));
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
    const std::ptrdiff_t tri = static_cast<std::ptrdiff_t>(sn) * (sn + 1) / 2;

    ctx->submit([&](sycl::handler& h) {
        sycl::local_accessor<D, 1> St(sycl::range<1>(tile), h);
        sycl::local_accessor<R, 1> rdt(sycl::range<1>(static_cast<std::size_t>(mp)), h);
        impl::parallel_for<PotriCtaKernel<T>>(h, sycl::nd_range<1>(groups * wg, wg), [=](sycl::nd_item<1> it) {
            using namespace sycl_device;
            const auto grp = it.get_group();
            const int tid = static_cast<int>(it.get_local_linear_id());
            const std::size_t g = it.get_group_linear_id();
            const std::ptrdiff_t item = static_cast<std::ptrdiff_t>(g / nblk);
            const int j0 = static_cast<int>(g % nblk) * bn;
            const int m = sycl::min(bn, n - j0);
            const int mpw = (m + TS - 1) / TS * TS;
            const int ntile = cta_tiles(m, TS);
            D* const Ag = a + item * stride + static_cast<std::ptrdiff_t>(j0) * (lda + 1);

            if (stash_on) {
                D* const Ai = a + item * stride - static_cast<std::ptrdiff_t>(origin) * (lda + 1);
                int z = INT_MAX;
                if (md == 1) {
                    for (int i = tid; i < sn; i += wg)
                        if (dev_is_zero(Ai[static_cast<std::ptrdiff_t>(i) * (lda + 1)])) z = sycl::min(z, i);
                    z = sycl::reduce_over_group(grp, z, sycl::minimum<int>());
                    if (j0 == 0 && tid == 0) flags[item] = z == INT_MAX ? 0 : z + 1;
                } else if (flags[item] != 0) {
                    z = 0;
                }
                if (z != INT_MAX) {  // work-group uniform; no other group of this launch writes these columns
                    D* const slot = slots + item * tri;
                    for (int c = md == 1 ? j0 : 0; c < (md == 1 ? j0 + m : sn); ++c) {
                        const std::ptrdiff_t cc = c;
                        const std::ptrdiff_t off = lower ? cc * sn - cc * (cc - 1) / 2 - cc : cc * (cc + 1) / 2;
                        for (int r = (lower ? c : 0) + tid; r < (lower ? sn : c + 1); r += wg) {
                            D& g = Ai[r + cc * lda];
                            if (md == 1) slot[off + r] = g;
                            else g = slot[off + r];
                        }
                    }
                    return;
                }
            }
            D* const S = &St[0];
            R* const rd = &rdt[0];
            auto s = [&](int r, int c) -> D& { return S[r + static_cast<std::ptrdiff_t>(c) * lds]; };
            auto g_at = [&](int r, int c) -> D& { return Ag[r + static_cast<std::ptrdiff_t>(c) * lda]; };

            // Global (r, c) walks column-major for coalescing, kU positions per round so their loads
            // overlap; (i, j) = the L coordinates. Only the uplo triangle is read, the rest is zeroed.
            constexpr int kU = 4;
            for (Walk w(tid, wg, mpw); w.c < mpw;) {
                int wi[kU], wj[kU];
                D v[kU], dg[kU];
                BATCHLAS_UNROLL_FULL
                for (int u = 0; u < kU; ++u) {
                    const bool live = w.c < mpw;
                    wi[u] = live ? (lower ? w.r : w.c) : -1;
                    wj[u] = live ? (lower ? w.c : w.r) : 0;
                    const bool in = live && wi[u] >= wj[u] && wi[u] < m;
                    v[u] = in ? g_at(w.r, w.c) : D{};
                    dg[u] = in && md != 2 && wi[u] > wj[u] ? g_at(wj[u], wj[u]) : dev_one<D>();
                    if (live) w.next();
                }
                BATCHLAS_UNROLL_FULL
                for (int u = 0; u < kU; ++u) {
                    const int i = wi[u], j = wj[u];
                    if (i < j) continue;
                    const bool in = i < m;
                    const D lv = lower ? v[u] : dev_conj(v[u]);
                    if (md == 2) {
                        s(j, i) = lv;  // lauum's X, transposed into the upper triangle
                        if (i != j) s(i, j) = D{};
                    } else if (i == j) {
                        rd[i] = in ? R(1) / dev_real(lv) : R(0);
                        s(i, i) = in ? dev_one<D>() : D{};  // P(k, k) = 1, read as row k's last entry
                    } else {
                        s(i, j) = dev_div_real(lv, dev_real(dg[u]));  // M(i, j) = L(i, j) / L(j, j)
                        s(j, i) = D{};
                    }
                }
            }
            sycl::group_barrier(grp);

            int zero = INT_MAX;
            if (md == 0) {
                for (int i = tid; i < m; i += wg)
                    if (dev_is_zero(g_at(i, i))) zero = sycl::min(zero, i);
                zero = sycl::reduce_over_group(grp, zero, sycl::minimum<int>());
                if (tid == 0 && info_ptr) info_ptr[item] = zero == INT_MAX ? 0 : zero + 1;
            }
            if (zero != INT_MAX) return;  // work-group uniform: no barrier is left unmatched

            int ti[NT], tj[NT];
            bool has[NT];
            BATCHLAS_UNROLL_FULL
            for (int q = 0; q < NT; ++q) {
                const int t = tid + q * wg;
                has[q] = t < ntile;
                int r = static_cast<int>((sycl::sqrt(8.0f * static_cast<float>(t) + 1.0f) - 1.0f) * 0.5f);
                while ((r + 1) * (r + 2) / 2 <= t) ++r;
                while (r * (r + 1) / 2 > t) --r;
                ti[q] = r * TS;
                tj[q] = (t - r * (r + 1) / 2) * TS;
            }
            D acc[NT][TS][TS];

            if (md != 2) {
                // trti2, right-looking by TS pivots: tile row K substitutes within its block rows in
                // registers and publishes them (P(k, j) at s(j, k)); the tiles below take the rank-TS
                // update acc -= M(rows, K) P(K, cols). One barrier per tile row.
                BATCHLAS_UNROLL_FULL
                for (int q = 0; q < NT; ++q)
                    BATCHLAS_UNROLL_FULL
                    for (int x = 0; x < TS; ++x)
                        BATCHLAS_UNROLL_FULL
                        for (int y = 0; y < TS; ++y) acc[q][x][y] = ti[q] + x == tj[q] + y ? dev_one<D>() : D{};
                for (int kb = 0; kb < mpw; kb += TS) {
                    BATCHLAS_UNROLL_FULL
                    for (int q = 0; q < NT; ++q) {
                        if (!has[q] || ti[q] != kb) continue;
                        BATCHLAS_UNROLL_FULL
                        for (int x = 1; x < TS; ++x)
                            BATCHLAS_UNROLL_FULL
                            for (int c = 0; c < x; ++c) {
                                const D mxc = s(kb + x, kb + c);
                                BATCHLAS_UNROLL_FULL
                                for (int y = 0; y < TS; ++y)
                                    acc[q][x][y] = dev_sub(acc[q][x][y], dev_mul(mxc, acc[q][c][y]));
                            }
                        BATCHLAS_UNROLL_FULL
                        for (int x = 0; x < TS; ++x)
                            BATCHLAS_UNROLL_FULL
                            for (int y = 0; y < TS; ++y)
                                if (tj[q] + y < kb + x) s(tj[q] + y, kb + x) = acc[q][x][y];
                    }
                    sycl::group_barrier(grp);
                    BATCHLAS_UNROLL_FULL
                    for (int q = 0; q < NT; ++q) {
                        if (!has[q] || ti[q] <= kb || tj[q] > kb) continue;
                        const bool diag = tj[q] == kb;  // P(K, K) is unit lower: zero above, one on it
                        D mt[TS][TS], pt[TS][TS];
                        BATCHLAS_UNROLL_FULL
                        for (int c = 0; c < TS; ++c) {
                            BATCHLAS_UNROLL_FULL
                            for (int x = 0; x < TS; ++x) mt[x][c] = s(ti[q] + x, kb + c);
                            BATCHLAS_UNROLL_FULL
                            for (int y = 0; y < TS; ++y) pt[c][y] = !diag || y <= c ? s(tj[q] + y, kb + c) : D{};
                        }
                        BATCHLAS_UNROLL_FULL
                        for (int c = 0; c < TS; ++c)
                            BATCHLAS_UNROLL_FULL
                            for (int x = 0; x < TS; ++x)
                                BATCHLAS_UNROLL_FULL
                                for (int y = 0; y < TS; ++y)
                                    acc[q][x][y] = dev_sub(acc[q][x][y], dev_mul(mt[x][c], pt[c][y]));
                    }
                }
                // X = diag(L)^-1 P as X^T in the upper triangle; the strict lower one (M) := 0.
                BATCHLAS_UNROLL_FULL
                for (int q = 0; q < NT; ++q) {
                    if (!has[q]) continue;
                    BATCHLAS_UNROLL_FULL
                    for (int x = 0; x < TS; ++x)
                        BATCHLAS_UNROLL_FULL
                        for (int y = 0; y < TS; ++y) {
                            const int i = ti[q] + x, j = tj[q] + y;
                            if (i > j) {
                                s(j, i) = dev_mul_real(acc[q][x][y], rd[i]);
                                s(i, j) = D{};
                            } else if (i == j) {
                                s(i, i) = dev_from_real<D>(rd[i]);
                            }
                        }
                }
                sycl::group_barrier(grp);
                if (md == 1) {
                    for (Walk w(tid, wg, m); w.c < m; w.next()) {
                        if (lower ? w.r < w.c : w.r > w.c) continue;
                        // X(r, c) sits at s(c, r); Upper wants (L^-1)^H.
                        g_at(w.r, w.c) = lower ? s(w.c, w.r) : dev_conj(s(w.r, w.c));
                    }
                    return;
                }
            }

            // lauu2: Y = X^H X, Y(i, j) = sum_k conj(X(k, i)) X(k, j); X(k, x) = s(x, k), zero for x > k.
            BATCHLAS_UNROLL_FULL
            for (int q = 0; q < NT; ++q) {
                BATCHLAS_UNROLL_FULL
                for (int x = 0; x < TS; ++x)
                    BATCHLAS_UNROLL_FULL
                    for (int y = 0; y < TS; ++y) acc[q][x][y] = D{};
                if (!has[q]) continue;
                for (int k = ti[q]; k < mpw; ++k) {
                    D xi[TS], xj[TS];
                    BATCHLAS_UNROLL_FULL
                    for (int x = 0; x < TS; ++x) xi[x] = dev_conj(s(ti[q] + x, k));
                    BATCHLAS_UNROLL_FULL
                    for (int y = 0; y < TS; ++y) xj[y] = s(tj[q] + y, k);
                    BATCHLAS_UNROLL_FULL
                    for (int x = 0; x < TS; ++x)
                        BATCHLAS_UNROLL_FULL
                        for (int y = 0; y < TS; ++y) fma_acc(acc[q][x][y], xi[x], xj[y]);
                }
            }
            sycl::group_barrier(grp);
            // Y into the strict lower triangle (no longer read) and rd (its real diagonal).
            BATCHLAS_UNROLL_FULL
            for (int q = 0; q < NT; ++q) {
                if (!has[q]) continue;
                BATCHLAS_UNROLL_FULL
                for (int x = 0; x < TS; ++x)
                    BATCHLAS_UNROLL_FULL
                    for (int y = 0; y < TS; ++y) {
                        const int i = ti[q] + x, j = tj[q] + y;
                        if (i > j) s(i, j) = acc[q][x][y];
                        else if (i == j) rd[i] = dev_real(acc[q][x][y]);
                    }
            }
            sycl::group_barrier(grp);

            // Y is A^-1 (Potri) or the product (Lauum) for either uplo; full fills the other triangle.
            const bool mirror = full && md == 0;
            for (Walk w(tid, wg, m); w.c < m; w.next()) {
                const int r = w.r, c = w.c;
                const bool owned = lower ? r >= c : r <= c;
                if (!owned && !mirror) continue;
                g_at(r, c) = r > c ? s(r, c) : r == c ? dev_from_real<D>(rd[r]) : dev_conj(s(c, r));
            }
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
