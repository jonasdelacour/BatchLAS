// SPDX-License-Identifier: MIT
// Device LA v2 level-3 tile components (include/batchlas/blas/device/v2/ext/l3-triangular-tiles.hh)
// against long-double host references, each composed the way a tuned kernel composes it:
//   BandTile    PacketStager (both panels, Trans x Aligned) + frag_ld4<Bands4<2,64,4>> + ger_acc +
//               epilogue_tri_packets over a triangular tile grid (tri_tile_decode): syrk / herk.
//   Trmm        stage_swz_kc / stage_swz_cc<PacketStore> + trmat_op_at + Swizzle + epilogue_cm(Zero).
//   Gram        stage_swz_kc / stage_swz_cc + contiguous Bands4 + TriMask<Herm> epilogue_cm.
//   Elements    sym_at<Herm>, trmat_op_at and tri_tile_decode element by element.
// Guards (agent-guide section 8): non-natural ld with NaN in the padding rows (a read past the
// shape poisons the result), complex data and conjugation, beta != 0 and a runtime beta == 0 over
// NaN, a sentinel unread triangle that must survive bit for bit, NaN in the unnamed triangle of A
// and on a unit diagonal, edge tiles (m, n not multiples of the tile), batch 1024 with period-3
// data where every item must be bit-identical to its representative, a 256-item launch.
#include <gtest/gtest.h>
#include <tests/test_utils.hh>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include <batchlas/blas/device/v2/ext/l3-triangular-tiles.hh>
#include <src/queue.hh>
#include <src/sycl/kernel_attrs.hh>

using namespace batchlas;
namespace v2 = batchlas::device::v2;
using batchlas::verify::Check;

namespace {

template <class T> using real_of = typename v2::Arith<T>::real_t;
template <class T> constexpr bool is_cplx = v2::Arith<T>::is_complex;

constexpr int kPeriod = 3;

std::uint64_t mix(std::uint64_t x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}
template <class R> R uval(std::uint64_t i) { return static_cast<R>(static_cast<double>(mix(i) >> 11) * 0x1.0p-53 * 2.0 - 1.0); }
template <class T> T val(std::uint64_t i) {
    if constexpr (is_cplx<T>) return T(uval<real_of<T>>(2 * i), uval<real_of<T>>(2 * i + 1));
    else return uval<T>(i);
}
template <class T> T nan_v() { return T(std::numeric_limits<real_of<T>>::quiet_NaN()); }

// kPeriod representatives (edited by fix(item pointer)), copied so item b repeats item b % kPeriod.
template <class T, class Fix> std::vector<T> batched(int per_item, int items, std::uint64_t seed, Fix fix) {
    std::vector<T> h(static_cast<std::size_t>(items) * per_item);
    for (int b = 0; b < kPeriod; ++b) {
        for (int i = 0; i < per_item; ++i) h[std::size_t(b) * per_item + i] = val<T>(seed + std::uint64_t(b) * 1000003u + i);
        fix(h.data() + std::size_t(b) * per_item);
    }
    for (int b = kPeriod; b < items; ++b)
        std::copy_n(h.data() + std::size_t(b % kPeriod) * per_item, per_item, h.data() + std::size_t(b) * per_item);
    return h;
}

template <class T> bool same_bits(const T& a, const T& b) { return std::memcmp(&a, &b, sizeof(T)) == 0; }

template <class T> using LD = std::conditional_t<is_cplx<T>, std::complex<long double>, long double>;
template <class T> LD<T> wide(T v) { return LD<T>(v); }
template <class T> LD<T> cj(LD<T> v, bool c) {
    if constexpr (is_cplx<T>) return c ? std::conj(v) : v;
    else return v;
}
template <class T> double rel(T got, LD<T> ref, long double scale) {
    const long double d = std::abs(LD<T>(got) - ref);
    if (std::isnan(d)) return std::numeric_limits<double>::infinity();
    return static_cast<double>(d / std::max<long double>(scale, std::numeric_limits<real_of<T>>::min()));
}

struct Dev {
    Queue ctx{Device::default_device()};
    sycl::queue& q() { return sycl_queue(ctx); }
    template <class T> T* up(const std::vector<T>& h) {
        T* p = sycl::malloc_device<T>(h.size(), q());
        q().memcpy(p, h.data(), h.size() * sizeof(T)).wait();
        ptrs.push_back(p);
        return p;
    }
    template <class T> std::vector<T> down(const T* p, std::size_t n) {
        std::vector<T> h(n);
        q().memcpy(h.data(), p, n * sizeof(T)).wait();
        return h;
    }
    // groups work-groups of WG items; f(item, shared T*, with local_elems T, packet-aligned).
    template <int WG, class T, class F> void run(int groups, int local_elems, F f) {
        q().submit([&](sycl::handler& h) {
             sycl::local_accessor<v2::Packet4<T>, 1> lm(sycl::range<1>(std::max(local_elems / 4, 1)), h);
             h.parallel_for(sycl::nd_range<1>(std::size_t(groups) * WG, WG), [=](sycl::nd_item<1> it) {
                 v2::Packet4<T>* base = lm.template get_multi_ptr<sycl::access::decorated::no>().get();
                 f(it, reinterpret_cast<T*>(base));
             });
         }).wait_and_throw();
    }
    ~Dev() {
        for (void* p : ptrs) sycl::free(p, q());
    }
    std::vector<void*> ptrs;
};

template <class T> class DeviceBlasV2L3TilesTest : public ::testing::Test {
protected:
    Dev dev;
};

using Types = ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(DeviceBlasV2L3TilesTest, Types);

// ---- BandTile: C(lower) := alpha opX opY^T + beta C on the 128 x 128 x 8 band tile ------------

struct BandCase {
    int n, k, ldx, ldc, batch;
    bool beta_zero;
};

// X and Y are n x k (Trans: k x n) with ld ldx; ConjR / ConjC conjugate the row / column factor;
// Herm drops Im(C(i, i)). Returns {worst relative error, items differing from their twin,
// upper-triangle sentinels changed}.
template <class T, bool Trans, bool Aligned, bool ConjR, bool ConjC, bool Herm>
void run_band_tile(Dev& dev, const BandCase& bc, double& worst, int& bitdiff, int& touched) {
    constexpr int Rows = 128, K = 8, WG = 256;
    using L = v2::Bands4<2, 64, 4>;
    using St = v2::PacketStager<Rows, K, WG, Trans, Aligned>;
    const int n = bc.n, k = bc.k, ldx = bc.ldx, ldc = bc.ldc, batch = bc.batch;
    const int xrows = Trans ? k : n, xcols = Trans ? n : k;
    const int per_x = ldx * xcols, per_c = ldc * n;
    const T sentinel = T(7);
    auto pad_x = [&](T* p) {
        for (int c = 0; c < xcols; ++c)
            for (int r = xrows; r < ldx; ++r) p[r + c * ldx] = nan_v<T>();
    };
    auto prep_c = [&](T* p) {
        for (int c = 0; c < n; ++c)
            for (int r = 0; r < ldc; ++r) {
                if (r >= n) p[r + c * ldc] = nan_v<T>();
                else if (r < c) p[r + c * ldc] = sentinel;
                else if (bc.beta_zero) p[r + c * ldc] = nan_v<T>();
            }
    };
    const auto hx = batched<T>(per_x, batch, 11, pad_x);
    const auto hy = batched<T>(per_x, batch, 12, pad_x);
    const auto hc = batched<T>(per_c, batch, 13, prep_c);
    const T* x = dev.up(hx);
    const T* y = dev.up(hy);
    T* c = dev.up(hc);
    const int tiles = v2::tri_tile_count((n + Rows - 1) / Rows);
    const T alpha = val<T>(21), beta = bc.beta_zero ? T(0) : val<T>(22);

    dev.template run<WG, T>(batch * tiles, 2 * Rows * K, [=](sycl::nd_item<1> it, T* sh) {
        const int g = static_cast<int>(it.get_group_linear_id());
        const int b = g / tiles, t = g % tiles;
        const int lid = static_cast<int>(it.get_local_linear_id());
        const int ty = lid % 16, tx = lid / 16;
        int bi = 0, bj = 0;
        v2::tri_tile_decode(t, bi, bj);
        const int m0 = bi * Rows, n0 = bj * Rows;
        T* sa = sh;
        T* sb = sh + Rows * K;
        v2::RegTile<T, 8, 8> acc;
        v2::fill_tile(acc, T(0));
        const St stage(lid);
        const v2::Panel<T> px{x + std::ptrdiff_t(b) * per_x, ldx, sa, m0};
        const v2::Panel<T> py{y + std::ptrdiff_t(b) * per_x, ldx, sb, n0};
        for (int k0 = 0; k0 < k; k0 += K) {
            stage.run(k0, n, k, px, py);
            it.barrier(sycl::access::fence_space::local_space);
#pragma unroll
            for (int kk = 0; kk < K; ++kk) {
                v2::Frag<T, 8> fr;
                v2::Frag<T, 8> fc;
                v2::frag_ld4<L, ConjR>(fr, sa + kk * Rows, ty);
                v2::frag_ld4<L, ConjC>(fc, sb + kk * Rows, tx);
                v2::ger_acc(acc, fr, fc);
            }
            it.barrier(sycl::access::fence_space::local_space);
        }
        v2::epilogue_tri_packets<Aligned, L, L>(v2::TriMask<Herm>{bi == bj, true}, acc, c + std::ptrdiff_t(b) * per_c, ldc,
                                                m0, n0, ty, tx, n, alpha, beta);
    });

    const auto got = dev.down(c, hc.size());
    for (int b = 0; b < batch; ++b) {
        const T* gb = got.data() + std::size_t(b) * per_c;
        const T* g0 = got.data() + std::size_t(b % kPeriod) * per_c;
        for (int i = 0; i < per_c; ++i) bitdiff += !same_bits(gb[i], g0[i]);
    }
    for (int b = 0; b < kPeriod; ++b) {
        const T* xb = hx.data() + std::size_t(b) * per_x;
        const T* yb = hy.data() + std::size_t(b) * per_x;
        const T* c0 = hc.data() + std::size_t(b) * per_c;
        const T* gb = got.data() + std::size_t(b) * per_c;
        auto opx = [&](const T* m, int i, int l) { return wide(Trans ? m[l + i * ldx] : m[i + l * ldx]); };
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) {
                if (i < j) {
                    touched += !same_bits(gb[i + j * ldc], sentinel);
                    continue;
                }
                LD<T> s = 0;
                long double scale = 0;
                for (int l = 0; l < k; ++l) {
                    const LD<T> a = cj<T>(opx(xb, i, l), ConjR), d = cj<T>(opx(yb, j, l), ConjC);
                    s += a * d;
                    scale += std::abs(a) * std::abs(d);
                }
                LD<T> ref = wide(alpha) * s;
                scale *= std::abs(wide(alpha));
                if (!bc.beta_zero) {
                    ref += wide(beta) * wide(c0[i + j * ldc]);
                    scale += std::abs(wide(beta)) * std::abs(wide(c0[i + j * ldc]));
                }
                if constexpr (Herm && is_cplx<T>) {
                    if (i == j) ref = LD<T>(ref.real(), 0);
                }
                worst = std::max(worst, rel(gb[i + j * ldc], ref, scale));
            }
    }
}

// Unaligned: n = 37 (one diagonal tile, all edges), ld + 3, batch 1024. Aligned: n = 256 (three
// tiles, one off the diagonal through the packet path), ld + 4 keeps the 16-byte packets.
TYPED_TEST(DeviceBlasV2L3TilesTest, BandTileSyrkHerk) {
    using T = TypeParam;
    constexpr bool C = is_cplx<T>;
    double worst = 0;
    int bitdiff = 0, touched = 0;
    run_band_tile<T, false, false, false, C, C>(this->dev, {37, 13, 40, 40, 1024, false}, worst, bitdiff, touched);
    run_band_tile<T, true, false, C, false, C>(this->dev, {37, 13, 16, 41, 1024, true}, worst, bitdiff, touched);
    run_band_tile<T, false, true, false, false, false>(this->dev, {256, 16, 260, 260, 6, false}, worst, bitdiff, touched);
    run_band_tile<T, true, true, C, false, C>(this->dev, {256, 16, 20, 260, 6, true}, worst, bitdiff, touched);
    EXPECT_EQ(bitdiff, 0) << "items differ from their representative";
    EXPECT_EQ(touched, 0) << "the unnamed triangle was written";
    EXPECT_VERIFY(T, Check::blas, 16, worst);
}

// ---- Trmm: C := alpha op(A) B, A triangular, swizzled staging, column-major epilogue ----------

template <class T, bool Trans, bool Unit, bool Lower> void run_trmm(Dev& dev, double& worst, int& bitdiff) {
    constexpr int TileM = 32, TileN = 32, TileK = 16, LR = 8, LCn = 8, WG = LR * LCn;
    using RB = v2::Bands4<1, TileM, 4>;
    using CB = v2::Bands4<1, TileN, 4>;
    using SwA = v2::Swizzle<TileM / 4>;
    using SwB = v2::Swizzle<TileN / 4>;
    constexpr int m = 45, n = 21, lda = m + 3, ldb = m + 5, ldc = m + 2, batch = 1024;
    constexpr bool conj = Trans && is_cplx<T>;
    constexpr bool lower_eff = Lower != Trans;
    const int tiles_m = (m + TileM - 1) / TileM, tiles_n = (n + TileN - 1) / TileN, tiles = tiles_m * tiles_n;
    auto fix_a = [&](T* p) {  // the other triangle, the padding and (Unit) the diagonal are never read
        for (int c = 0; c < m; ++c)
            for (int r = 0; r < lda; ++r)
                if (r >= m || (Lower ? r < c : r > c) || (Unit && r == c)) p[r + c * lda] = nan_v<T>();
    };
    auto fix_b = [&](T* p) {
        for (int c = 0; c < n; ++c)
            for (int r = m; r < ldb; ++r) p[r + c * ldb] = nan_v<T>();
    };
    auto fix_c = [&](T* p) {
        for (int i = 0; i < ldc * n; ++i) p[i] = nan_v<T>();
    };
    const auto ha = batched<T>(lda * m, batch, 31, fix_a);
    const auto hb = batched<T>(ldb * n, batch, 32, fix_b);
    const auto hc = batched<T>(ldc * n, batch, 33, fix_c);
    const T* a = dev.up(ha);
    const T* bm = dev.up(hb);
    T* c = dev.up(hc);
    const T alpha = val<T>(34);

    dev.template run<WG, T>(batch * tiles, TileK * (TileM + TileN), [=](sycl::nd_item<1> it, T* sh) {
        const int g = static_cast<int>(it.get_group_linear_id());
        const int b = g / tiles, t = g % tiles;
        const int m0 = (t / tiles_n) * TileM, n0 = (t % tiles_n) * TileN;
        const int tid = static_cast<int>(it.get_local_linear_id());
        const int tr = tid % LR, tc = tid / LR;
        const T* ab = a + std::ptrdiff_t(b) * lda * m;
        const T* bb = bm + std::ptrdiff_t(b) * ldb * n;
        T* sa = sh;
        T* sb = sh + TileK * TileM;
        v2::RegTile<T, 4, 4> acc;
        v2::fill_tile(acc, T(0));
        for (int p0 = 0; p0 < m; p0 += TileK) {
            if constexpr (Trans) {
                v2::stage_swz_kc<TileM, TileK, WG>(sa, tid, m0, p0, [&](int i, int p) {
                    return v2::trmat_op_at<true>(ab, lda, m, i, p, lower_eff, Unit, conj);
                });
            } else {
                v2::stage_swz_cc<TileM, TileK, WG, true>(sa, tid, m0, p0, [&](int i, int p) {
                    return v2::trmat_op_at<false>(ab, lda, m, i, p, lower_eff, Unit, conj);
                });
            }
            v2::stage_swz_kc<TileN, TileK, WG>(sb, tid, n0, p0, [&](int j, int p) {
                return (j < n && p < m) ? bb[p + std::ptrdiff_t(j) * ldb] : T(0);
            });
            it.barrier(sycl::access::fence_space::local_space);
            for (int p = 0; p < TileK; ++p) {
                v2::Frag<T, 4> fa;
                v2::Frag<T, 4> fb;
                v2::frag_ld4<RB, false, SwA>(fa, sa + p * TileM, tr, p);
                v2::frag_ld4<CB, false, SwB>(fb, sb + p * TileN, tc, p);
                v2::ger_acc(acc, fa, fb);
            }
            it.barrier(sycl::access::fence_space::local_space);
        }
        v2::epilogue_cm<RB, CB>(v2::FullMask{}, acc, c + std::ptrdiff_t(b) * ldc * n, ldc, m0, n0, tr, tc, m, n, alpha,
                                v2::Zero{});
    });

    const auto got = dev.down(c, hc.size());
    const int per_c = ldc * n;
    for (int b = 0; b < batch; ++b)
        for (int i = 0; i < per_c; ++i) bitdiff += !same_bits(got[std::size_t(b) * per_c + i], got[std::size_t(b % kPeriod) * per_c + i]);
    for (int b = 0; b < kPeriod; ++b) {
        const T* ab = ha.data() + std::size_t(b) * lda * m;
        const T* bb = hb.data() + std::size_t(b) * ldb * n;
        const T* gb = got.data() + std::size_t(b) * per_c;
        auto op_a = [&](int i, int p) -> LD<T> {  // op(A)(i, p) from the stored triangle only
            const int r = Trans ? p : i, q = Trans ? i : p;
            if (r == q) return Unit ? LD<T>(1) : cj<T>(wide(ab[r + q * lda]), conj);
            if (Lower ? r < q : r > q) return LD<T>(0);
            return cj<T>(wide(ab[r + q * lda]), conj);
        };
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < m; ++i) {
                LD<T> s = 0;
                long double scale = 0;
                for (int p = 0; p < m; ++p) {
                    const LD<T> x = op_a(i, p), y = wide(bb[p + j * ldb]);
                    s += x * y;
                    scale += std::abs(x) * std::abs(y);
                }
                worst = std::max(worst, rel(gb[i + j * ldc], wide(alpha) * s, scale * std::abs(wide(alpha))));
            }
    }
}

TYPED_TEST(DeviceBlasV2L3TilesTest, TrmmSwizzledTiles) {
    using T = TypeParam;
    double worst = 0;
    int bitdiff = 0;
    run_trmm<T, false, false, true>(this->dev, worst, bitdiff);
    run_trmm<T, false, true, false>(this->dev, worst, bitdiff);
    run_trmm<T, true, false, false>(this->dev, worst, bitdiff);
    run_trmm<T, true, true, true>(this->dev, worst, bitdiff);
    EXPECT_EQ(bitdiff, 0) << "items differ from their representative";
    EXPECT_VERIFY(T, Check::blas, 45, worst);
}

// ---- Gram: one NTile = 32 tile, contiguous 4-wide thread tiles over the triangle ----------------

template <class T, bool Trans, bool Herm> void run_gram(Dev& dev, bool lower, double& worst, int& bitdiff, int& touched) {
    constexpr int NTile = 32, TT = 4, KC = 32, Lanes = NTile / TT, TriTiles = Lanes * (Lanes + 1) / 2;
    constexpr int WG = ((TriTiles + 31) / 32) * 32;
    using Band = v2::Bands4<TT / 4, 4, TT>;
    using Sw = v2::Swizzle<NTile / 4>;
    constexpr bool ConjR = Herm && Trans, ConjC = Herm && !Trans;
    constexpr int n = 29, k = 45, batch = 1024;
    const int ldx = Trans ? k + 2 : n + 3, xrows = Trans ? k : n, xcols = Trans ? n : k, ldc = n + 4;
    const int per_x = ldx * xcols, per_c = ldc * n;
    const T sentinel = T(7);
    auto fix_x = [&](T* p) {
        for (int c = 0; c < xcols; ++c)
            for (int r = xrows; r < ldx; ++r) p[r + c * ldx] = nan_v<T>();
    };
    auto fix_c = [&](T* p) {
        for (int c = 0; c < n; ++c)
            for (int r = 0; r < ldc; ++r)
                if (r >= n) p[r + c * ldc] = nan_v<T>();
                else if (lower ? r < c : r > c) p[r + c * ldc] = sentinel;
    };
    const auto hx = batched<T>(per_x, batch, 41, fix_x);
    const auto hc = batched<T>(per_c, batch, 42, fix_c);
    const T* x = dev.up(hx);
    T* c = dev.up(hc);
    const T alpha = val<T>(43), beta = val<T>(44);

    dev.template run<WG, T>(batch, KC * NTile, [=](sycl::nd_item<1> it, T* sh) {
        const int b = static_cast<int>(it.get_group_linear_id());
        const int tid = static_cast<int>(it.get_local_linear_id());
        const bool active = tid < TriTiles;
        int bi = 0, bj = 0;
        v2::tri_tile_decode(active ? tid : 0, bi, bj);
        const int tr = lower ? bi : bj, tc = lower ? bj : bi;
        const T* xb = x + std::ptrdiff_t(b) * per_x;
        v2::RegTile<T, TT, TT> acc;
        v2::fill_tile(acc, T(0));
        for (int k0 = 0; k0 < k; k0 += KC) {
            if constexpr (Trans) {
                v2::stage_swz_kc<NTile, KC, WG>(sh, tid, 0, k0, [&](int col, int gk) {
                    return (col < n && gk < k) ? xb[gk + std::ptrdiff_t(col) * ldx] : T(0);
                });
            } else {
                v2::stage_swz_cc<NTile, KC, WG, false>(sh, tid, 0, k0, [&](int col, int gk) {
                    return (gk < k && col < n) ? xb[col + std::ptrdiff_t(gk) * ldx] : T(0);
                });
            }
            it.barrier(sycl::access::fence_space::local_space);
            if (active) {
                for (int p = 0; p < KC; ++p) {
                    v2::Frag<T, TT> fa;
                    v2::Frag<T, TT> fb;
                    v2::frag_ld4<Band, ConjR, Sw>(fa, sh + p * NTile, tr, p);
                    v2::frag_ld4<Band, ConjC, Sw>(fb, sh + p * NTile, tc, p);
                    v2::ger_acc(acc, fa, fb);
                }
            }
            it.barrier(sycl::access::fence_space::local_space);
        }
        if (!active) return;
        v2::epilogue_cm<Band, Band>(v2::TriMask<Herm>{bi == bj, lower}, acc, c + std::ptrdiff_t(b) * per_c, ldc, 0, 0, tr, tc,
                                    n, n, alpha, beta);
    });

    const auto got = dev.down(c, hc.size());
    for (int b = 0; b < batch; ++b)
        for (int i = 0; i < per_c; ++i) bitdiff += !same_bits(got[std::size_t(b) * per_c + i], got[std::size_t(b % kPeriod) * per_c + i]);
    for (int b = 0; b < kPeriod; ++b) {
        const T* xb = hx.data() + std::size_t(b) * per_x;
        const T* c0 = hc.data() + std::size_t(b) * per_c;
        const T* gb = got.data() + std::size_t(b) * per_c;
        auto opx = [&](int i, int l) { return wide(Trans ? xb[l + i * ldx] : xb[i + l * ldx]); };
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) {
                if (lower ? i < j : i > j) {
                    touched += !same_bits(gb[i + j * ldc], sentinel);
                    continue;
                }
                LD<T> s = 0;
                long double scale = 0;
                for (int l = 0; l < k; ++l) {
                    const LD<T> u = cj<T>(opx(i, l), ConjR), v = cj<T>(opx(j, l), ConjC);
                    s += u * v;
                    scale += std::abs(u) * std::abs(v);
                }
                LD<T> ref = wide(alpha) * s + wide(beta) * wide(c0[i + j * ldc]);
                scale = scale * std::abs(wide(alpha)) + std::abs(wide(beta)) * std::abs(wide(c0[i + j * ldc]));
                if constexpr (Herm && is_cplx<T>) {
                    if (i == j) ref = LD<T>(ref.real(), 0);
                }
                worst = std::max(worst, rel(gb[i + j * ldc], ref, scale));
            }
    }
}

TYPED_TEST(DeviceBlasV2L3TilesTest, GramContiguousTiles) {
    using T = TypeParam;
    constexpr bool C = is_cplx<T>;
    double worst = 0;
    int bitdiff = 0, touched = 0;
    run_gram<T, false, C>(this->dev, true, worst, bitdiff, touched);
    run_gram<T, true, C>(this->dev, false, worst, bitdiff, touched);
    run_gram<T, true, false>(this->dev, true, worst, bitdiff, touched);
    EXPECT_EQ(bitdiff, 0) << "items differ from their representative";
    EXPECT_EQ(touched, 0) << "the unnamed triangle was written";
    EXPECT_VERIFY(T, Check::blas, 45, worst);
}

// ---- Elements: sym_at, trmat_op_at, tri_tile_decode ------------------------------------------

TYPED_TEST(DeviceBlasV2L3TilesTest, ElementReads) {
    using T = TypeParam;
    constexpr int n = 23, lda = n + 3, WG = 64, kDecode = 4096;
    // Slot layout per element (r, c): sym lower, sym upper, herm lower, herm upper, op(A) for
    // (Trans, lower_eff, unit, conj) = (0,1,0,0), (1,0,1,1).
    constexpr int kSlots = 6;
    auto fix = [&](T* p) {
        for (int r = n; r < lda; ++r)
            for (int c = 0; c < n; ++c) p[r + c * lda] = nan_v<T>();
    };
    const auto ha = batched<T>(lda * n, kPeriod, 51, fix);
    // Lower-stored and upper-stored copies with the other half NaN (a mirror read must not see it).
    std::vector<T> hl(ha), hu(ha);
    for (int b = 0; b < kPeriod; ++b)
        for (int c = 0; c < n; ++c)
            for (int r = 0; r < n; ++r) {
                T* pl = hl.data() + std::size_t(b) * lda * n;
                T* pu = hu.data() + std::size_t(b) * lda * n;
                if (r < c) pl[r + c * lda] = nan_v<T>();
                if (r > c) pu[r + c * lda] = nan_v<T>();
            }
    const T* al = this->dev.up(hl);
    const T* au = this->dev.up(hu);
    std::vector<T> ho(std::size_t(kPeriod) * kSlots * n * n, nan_v<T>());
    T* o = this->dev.up(ho);
    std::vector<int> hd(2 * kDecode, -1);
    int* d = this->dev.up(hd);

    this->dev.template run<WG, T>(kPeriod, 4, [=](sycl::nd_item<1> it, T*) {
        const int b = static_cast<int>(it.get_group_linear_id());
        const int lid = static_cast<int>(it.get_local_linear_id());
        const T* lb = al + std::ptrdiff_t(b) * lda * n;
        const T* ub = au + std::ptrdiff_t(b) * lda * n;
        T* ob = o + std::ptrdiff_t(b) * kSlots * n * n;
        for (int e = lid; e < n * n; e += WG) {
            const int r = e % n, c = e / n;
            ob[0 * n * n + e] = v2::sym_at(lb, lda, r, c, true);
            ob[1 * n * n + e] = v2::sym_at(ub, lda, r, c, false);
            ob[2 * n * n + e] = v2::sym_at<true>(lb, lda, r, c, true);
            ob[3 * n * n + e] = v2::sym_at<true>(ub, lda, r, c, false);
            ob[4 * n * n + e] = v2::trmat_op_at<false>(lb, lda, n, r, c, true, false, false);
            ob[5 * n * n + e] = v2::trmat_op_at<true>(lb, lda, n, r, c, false, true, true);
        }
        if (b == 0)
            for (int t = lid; t < kDecode; t += WG) v2::tri_tile_decode(t, d[2 * t], d[2 * t + 1]);
    });

    const auto got = this->dev.down(o, ho.size());
    const auto dec = this->dev.down(d, hd.size());
    int bad = 0;
    for (int b = 0; b < kPeriod; ++b) {
        const T* A = ha.data() + std::size_t(b) * lda * n;
        const T* g = got.data() + std::size_t(b) * kSlots * n * n;
        auto at = [&](int r, int c) { return A[r + c * lda]; };
        auto cjt = [&](T v) {
            if constexpr (is_cplx<T>) return std::conj(v);
            else return v;
        };
        for (int c = 0; c < n; ++c)
            for (int r = 0; r < n; ++r) {
                const int e = r + c * n;
                const T sl = r >= c ? at(r, c) : at(c, r);
                const T su = r <= c ? at(r, c) : at(c, r);
                T hl_ = r >= c ? at(r, c) : cjt(at(c, r));
                T hu_ = r <= c ? at(r, c) : cjt(at(c, r));
                if constexpr (is_cplx<T>) {
                    if (r == c) hl_ = hu_ = T(at(r, c).real(), 0);
                }
                const T t0 = r == c ? at(r, r) : (c < r ? at(r, c) : T(0));        // NoTrans, lower
                const T t1 = r == c ? T(1) : (c > r ? cjt(at(c, r)) : T(0));       // ConjTrans of lower, unit
                bad += !same_bits(g[e], sl) + !same_bits(g[n * n + e], su) + !same_bits(g[2 * n * n + e], hl_) +
                       !same_bits(g[3 * n * n + e], hu_) + !same_bits(g[4 * n * n + e], t0) + !same_bits(g[5 * n * n + e], t1);
            }
    }
    for (int t = 0, bi = 0; t < kDecode; ++t) {
        while ((bi + 1) * (bi + 2) / 2 <= t) ++bi;
        bad += dec[2 * t] != bi || dec[2 * t + 1] != t - bi * (bi + 1) / 2;
    }
    EXPECT_EQ(bad, 0);
}

}  // namespace
