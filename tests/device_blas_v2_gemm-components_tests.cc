// SPDX-License-Identifier: MIT
// Device LA v2 gemm components (include/batchlas/blas/device/v2/ext/gemm-components.hh) against
// host references: each test composes the ops into the kernel shape a tuned gemm uses.
// Guards (docs/developer/agent-guide.md section 8): non-natural ld (m + 3, k + 5, 16-aligned
// m + 7 for the packet stagers), m, n, k off every tile multiple and exact multiples for the
// unpredicated forms, NoTrans / Trans / ConjTrans with complex data, beta == 0 over NaN-poisoned C
// (never read) and beta != 0, batch 1024 at period 3 with every item bit-identical to its
// representative (all SLM paths), odd and even slab counts for both pipelines, -0 through the
// beta == 0 epilogue and sum_seq, fma bit-exactness against the host fma.
#include <gtest/gtest.h>
#include <tests/test_utils.hh>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include <batchlas/blas/device/v2/ext/gemm-components.hh>
#include <batchlas/util/sycl-local-accessor-helpers.hh>
#include <src/queue.hh>
#include <src/sycl/kernel_attrs.hh>

using namespace batchlas;
namespace v2 = batchlas::device::v2;
using batchlas::verify::Check;

namespace {

template <class T> using real_of = typename v2::Arith<T>::real_t;
template <class T> constexpr bool is_cplx = v2::Arith<T>::is_complex;

constexpr int kBatch = 1024;
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
// kBatch items of per_item values; item b repeats item b % kPeriod.
template <class T> std::vector<T> batched(int per_item, std::uint64_t seed) {
    std::vector<T> h(std::size_t(kBatch) * per_item);
    for (int b = 0; b < kPeriod; ++b)
        for (int i = 0; i < per_item; ++i) h[std::size_t(b) * per_item + i] = val<T>(seed + std::uint64_t(b) * 1000003u + i);
    for (int b = kPeriod; b < kBatch; ++b)
        std::copy_n(h.data() + std::size_t(b % kPeriod) * per_item, per_item, h.data() + std::size_t(b) * per_item);
    return h;
}
template <class T> bool same_bits(const T& a, const T& b) { return std::memcmp(&a, &b, sizeof(T)) == 0; }
template <class T> T nan_of() { return T(std::numeric_limits<real_of<T>>::quiet_NaN()); }

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
    template <int WG, class F> void run(int groups, std::size_t local_bytes, F f) {
        q().submit([&](sycl::handler& h) {
             sycl::local_accessor<unsigned char, 1> lm(sycl::range<1>(std::max<std::size_t>(local_bytes, 16)), h);
             h.parallel_for(sycl::nd_range<1>(std::size_t(groups) * WG, WG),
                            [=](sycl::nd_item<1> it) BATCHLAS_REQD_SG_SIZE(32) { f(it, batchlas::util::get_raw_ptr(lm)); });
         }).wait_and_throw();
    }
    ~Dev() {
        for (void* p : ptrs) sycl::free(p, q());
    }
    std::vector<void*> ptrs;
};

template <class T> class DeviceBlasV2GemmComponentsTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!dev.ctx.device().supports_sub_group_size(32)) GTEST_SKIP() << "v2 tests launch sub-group size 32";
    }
    Dev dev;
};

using Types = ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(DeviceBlasV2GemmComponentsTest, Types);

template <class T> using LD = std::conditional_t<is_cplx<T>, std::complex<long double>, long double>;
template <class T> T conj_h(T v) {
    if constexpr (is_cplx<T>) return std::conj(v);
    else return v;
}
// op(X)(i, j) of a column-major X on the host.
template <class T> T op_h(const T* x, int ld, Transpose op, int i, int j) {
    if (op == Transpose::NoTrans) return x[i + std::size_t(j) * ld];
    const T v = x[j + std::size_t(i) * ld];
    return op == Transpose::ConjTrans ? conj_h(v) : v;
}
template <class T> double rel(T got, LD<T> ref, long double scale) {
    const long double d = std::abs(LD<T>(got) - ref);
    if (std::isnan(d)) return std::numeric_limits<double>::infinity();
    return static_cast<double>(d / std::max<long double>(scale, std::numeric_limits<real_of<T>>::min()));
}

struct Shape {
    int m, n, k, lda, ldb, ldc;
};

// The host check shared by the gemm-shaped tests: alpha op(A) op(B) + beta C0 per item (beta ==
// 0 ignores C0, which is NaN there), items bit-identical to their representative.
template <class T>
void check_gemm(const Shape& s, Transpose ta, Transpose tb, T alpha, T beta, const std::vector<T>& A, const std::vector<T>& B,
                const std::vector<T>& C0, const std::vector<T>& got, std::size_t sa, std::size_t sb, std::size_t sc,
                const char* where) {
    double worst = 0;
    int bitdiff = 0;
    for (int b = 0; b < kBatch; ++b) {
        for (int j = 0; j < s.n; ++j)
            for (int i = 0; i < s.m; ++i)
                bitdiff += !same_bits(got[b * sc + i + std::size_t(j) * s.ldc], got[(b % kPeriod) * sc + i + std::size_t(j) * s.ldc]);
        if (b >= kPeriod) continue;
        for (int j = 0; j < s.n; ++j)
            for (int i = 0; i < s.m; ++i) {
                LD<T> acc = 0;
                long double scale = 0;
                for (int l = 0; l < s.k; ++l) {
                    const LD<T> x = LD<T>(op_h(A.data() + b * sa, s.lda, ta, i, l)), y = LD<T>(op_h(B.data() + b * sb, s.ldb, tb, l, j));
                    acc += x * y;
                    scale += std::abs(x) * std::abs(y);
                }
                LD<T> ref = LD<T>(alpha) * acc;
                scale *= std::abs(LD<T>(alpha));
                if (beta != T(0)) {
                    const LD<T> c = LD<T>(C0[b * sc + i + std::size_t(j) * s.ldc]);
                    ref += LD<T>(beta) * c;
                    scale += std::abs(LD<T>(beta)) * std::abs(c);
                }
                worst = std::max(worst, rel(got[b * sc + i + std::size_t(j) * s.ldc], ref, scale));
            }
    }
    EXPECT_EQ(bitdiff, 0) << where << ": items differ from their representative";
    EXPECT_VERIFY(T, Check::blas, s.k, worst) << where;
}

// ---- stager + kloop + mma_slab + store_tile_staged: the register-tiled gemm ----------------------

constexpr int TM = 16, TN = 32, TK = 8, TR = 2, TC = 4, LCOLS = TN / TC, THR = (TM / TR) * LCOLS;
constexpr int SA = (TM + 1) * TK, SB = (TK + 1) * TN;  // one stage; the epilogue takes two passes at Stages 1

enum class Stager { Scalar, Packet, PacketAligned };

template <Transpose OpA, Transpose OpB, int Stages, bool Exact, Stager St, class T>
void run_register_tiled(Dev& dev, const Shape& s, T alpha, T beta, const char* where) {
    const int gm = (s.m + TM - 1) / TM, gn = (s.n + TN - 1) / TN;
    const int ar = OpA == Transpose::NoTrans ? s.m : s.k, ac = OpA == Transpose::NoTrans ? s.k : s.m;
    const int br = OpB == Transpose::NoTrans ? s.k : s.n, bc = OpB == Transpose::NoTrans ? s.n : s.k;
    const std::size_t sa = std::size_t(s.lda) * ac, sb = std::size_t(s.ldb) * bc, sc = std::size_t(s.ldc) * s.n;
    static_cast<void>(ar);
    static_cast<void>(br);
    const auto hA = batched<T>(int(sa), 11), hB = batched<T>(int(sb), 12);
    auto hC = batched<T>(int(sc), 13);
    if (beta == T(0)) std::fill(hC.begin(), hC.end(), nan_of<T>());
    const T* A = dev.up(hA);
    const T* B = dev.up(hB);
    T* C = dev.up(hC);
    const int m = s.m, n = s.n, k = s.k, lda = s.lda, ldb = s.ldb, ldc = s.ldc;
    dev.template run<THR>(kBatch * gm * gn, sizeof(T) * Stages * (SA + SB), [=](sycl::nd_item<1> it, unsigned char* lm) {
        const int g = static_cast<int>(it.get_group_linear_id());
        const int b = g / (gm * gn), rem = g % (gm * gn);
        const int group_row = rem % gm, group_col = rem / gm;
        const int lid = static_cast<int>(it.get_local_linear_id());
        const int local_row = lid / LCOLS, local_col = lid % LCOLS;
        T* const sa_ = reinterpret_cast<T*>(lm);
        T* const sb_ = sa_ + Stages * SA;
        const T* a = A + std::ptrdiff_t(b) * sa;
        const T* bm = B + std::ptrdiff_t(b) * sb;
        T* c = C + std::ptrdiff_t(b) * sc;
        const v2::WorkGroupNd<THR, 1> ex{it};
        T acc[TR][TC];
        v2::frag_fill(acc, T(0));
        auto load = [&](int kk0, int stage) {
            T* ta = sa_ + stage * SA;
            T* tb = sb_ + stage * SB;
            auto dst_a = [&](int r, int cc) -> T& { return ta[cc * (TM + 1) + r]; };
            auto dst_b = [&](int r, int cc) -> T& { return tb[cc * (TK + 1) + r]; };
            const int m0 = group_row * TM, n0 = group_col * TN;
            if constexpr (St == Stager::PacketAligned) {
                v2::stage_block_packet_aligned<TM, TK, 4, THR>(lid, a, lda, m0, kk0, dst_a);
                v2::stage_block_packet_aligned<TK, TN, 4, THR>(lid, bm, ldb, kk0, n0, dst_b);
            } else if constexpr (St == Stager::Packet) {
                v2::stage_block_packet<OpA, TM, TK, 4, THR>(lid, a, lda, m0, kk0, m, k, dst_a);
                v2::stage_block_packet<OpB, TK, TN, 4, THR>(lid, bm, ldb, kk0, n0, k, n, dst_b);
            } else {
                v2::stage_block<OpA, TM, TK, THR>(lid, a, lda, m0, kk0, m, k, dst_a);
                v2::stage_block<OpB, TK, TN, THR>(lid, bm, ldb, kk0, n0, k, n, dst_b);
            }
        };
        auto compute = [&](int stage, int kk0) {
            const T* ta = sa_ + stage * SA;
            const T* tb = sb_ + stage * SB;
            v2::mma_slab<TK, 2, Exact>(
                acc, kk0, k, [&](int t, int i) { return ta[t * (TM + 1) + local_row * TR + i]; },
                [&](int t, int j) { return tb[(local_col * TC + j) * (TK + 1) + t]; });
        };
        const int tiles = Exact ? k / TK : (k + TK - 1) / TK;
        v2::kloop<Stages, Exact, TK>(ex, tiles, load, compute);
        v2::store_tile_staged<TM, TN, TR, TC, THR, Stages * SA, Stages * SB>(ex, lid, local_row, local_col, acc, sa_, sb_, c,
                                                                          ldc, group_row * TM, group_col * TN, m, n, alpha, beta);
    });
    check_gemm(s, OpA, OpB, alpha, beta, hA, hB, hC, dev.down(C, hC.size()), sa, sb, sc, where);
}

TYPED_TEST(DeviceBlasV2GemmComponentsTest, RegisterTiledGemmFromComponents) {
    using T = TypeParam;
    using enum Transpose;
    const T alpha = val<T>(7), beta = val<T>(8);
    const Shape odd{37, 45, 29, 37 + 3, 29 + 5, 37 + 2};       // every bound live, non-natural ld
    const Shape odd_t{37, 45, 29, 29 + 5, 45 + 3, 37 + 2};     // the same for transposed operands
    run_register_tiled<NoTrans, NoTrans, 1, false, Stager::Scalar>(this->dev, odd, alpha, beta, "NN s1 scalar");
    run_register_tiled<NoTrans, NoTrans, 1, false, Stager::Scalar>(this->dev, odd, alpha, T(0), "NN s1 scalar beta=0");
    run_register_tiled<Trans, ConjTrans, 2, false, Stager::Scalar>(this->dev, odd_t, alpha, beta, "TC s2 scalar");
    run_register_tiled<ConjTrans, Trans, 2, false, Stager::Scalar>(this->dev, odd_t, alpha, T(0), "CT s2 scalar beta=0");
    if constexpr (!is_cplx<T>) {
        // Packets need 16-byte aligned columns: ld a multiple of 4 that is still not the natural one.
        const Shape pk{37, 45, 29, 44, 36, 39};
        const Shape pk_t{37, 45, 29, 36, 48, 39};
        run_register_tiled<NoTrans, NoTrans, 2, false, Stager::Packet>(this->dev, pk, alpha, beta, "NN s2 packet");
        run_register_tiled<Trans, Trans, 1, false, Stager::Packet>(this->dev, pk_t, alpha, T(0), "TT s1 packet beta=0");
        const Shape exact{32, 64, 24, 36, 28, 35};  // whole tiles, k % TK == 0: the unpredicated forms
        run_register_tiled<NoTrans, NoTrans, 2, true, Stager::PacketAligned>(this->dev, exact, alpha, beta, "NN s2 aligned");
        run_register_tiled<NoTrans, NoTrans, 1, true, Stager::PacketAligned>(this->dev, exact, alpha, T(0), "NN s1 aligned beta=0");
    }
}

// ---- stage_storage + frag_load + gemv_t_frag + epilogue_store_fused: the small batched gemm -------

template <int NB, int Chunk, class T>
void run_small(Dev& dev, const Shape& s, Transpose ta_, Transpose tb_, T alpha, T beta, const char* where) {
    constexpr int kTpm = 2 * NB, kCpt = NB / 2, kLdb = NB + 1;
    const bool ta = ta_ != Transpose::NoTrans, tb = tb_ != Transpose::NoTrans;
    const bool ca = is_cplx<T> && ta_ == Transpose::ConjTrans, cb = is_cplx<T> && tb_ == Transpose::ConjTrans;
    const std::size_t sa = std::size_t(s.lda) * (ta ? s.m : s.k), sb = std::size_t(s.ldb) * (tb ? s.k : s.n);
    const std::size_t sc = std::size_t(s.ldc) * s.n;
    const auto hA = batched<T>(int(sa), 21), hB = batched<T>(int(sb), 22);
    auto hC = batched<T>(int(sc), 23);
    const bool beta_zero = beta == T(0);
    if (beta_zero) std::fill(hC.begin(), hC.end(), nan_of<T>());
    const T* A = dev.up(hA);
    const T* B = dev.up(hB);
    T* C = dev.up(hC);
    const int m = s.m, n = s.n, k = s.k;
    const std::ptrdiff_t lda = s.lda, ldb = s.ldb, ldc = s.ldc;
    dev.template run<kTpm>(kBatch, sizeof(T) * NB * kLdb, [=](sycl::nd_item<1> it, unsigned char* lm) {
        const int b = static_cast<int>(it.get_group_linear_id());
        const int tt = static_cast<int>(it.get_local_linear_id());
        const int r = tt % NB, c0 = (tt / NB) * kCpt;
        T* const sB = reinterpret_cast<T*>(lm);
        const T* Ab = A + std::ptrdiff_t(b) * sa;
        const T* Bb = B + std::ptrdiff_t(b) * sb;
        T* Cb = C + std::ptrdiff_t(b) * sc;
        const v2::WorkGroupNd<kTpm, 1> ex{it};
        using Ar = v2::FmaArith<T>;
        T a[NB];
        if (ta) {
            v2::stage_storage<kTpm>(tt, v2::Ext<NB>{}, v2::Ext<NB>{}, true, Ab, lda, false, ca, k, m,
                                    [&](int l, int rr) -> T& { return sB[rr + l * kLdb]; });
            ex.barrier();
            v2::frag_load(a, [&](int l) { return sB[r + l * kLdb]; });
            ex.barrier();
        } else {
            v2::frag_load(a, [&](int l) {
                T v{};
                if (r < m && l < k) v = Ab[r + l * lda];
                return v;
            });
        }
        v2::stage_storage<kTpm>(tt, v2::Ext<NB>{}, v2::Ext<NB>{}, true, Bb, ldb, tb, cb, k, n,
                                [&](int l, int c) -> T& { return sB[l + c * kLdb]; });
        ex.barrier();
        T acc[kCpt];
        v2::frag_fill(acc, T{});
        v2::gemv_t_frag<Chunk, Ar>(acc, a, [&](int l, int j) { return sB[l + (c0 + j) * kLdb]; }, k);
        if (r < m) {
            for (int j = 0; j < kCpt; ++j) {
                const int c = c0 + j;
                if (c < n) v2::epilogue_store_fused<Ar>(Cb + r + c * ldc, alpha, beta, beta_zero, acc[j]);
            }
        }
    });
    check_gemm(s, ta_, tb_, alpha, beta, hA, hB, hC, dev.down(C, hC.size()), sa, sb, sc, where);
}

TYPED_TEST(DeviceBlasV2GemmComponentsTest, SmallBatchedGemmFromComponents) {
    using T = TypeParam;
    using enum Transpose;
    const T alpha = val<T>(31), beta = val<T>(32);
    const Shape s16{13, 11, 15, 13 + 3, 15 + 5, 13 + 1};      // NB 16: four gemv_t_frag chunks of 4
    const Shape s16t{13, 11, 15, 15 + 3, 11 + 5, 13 + 1};
    run_small<16, 4>(this->dev, s16, NoTrans, NoTrans, alpha, beta, "NB16 NN");
    run_small<16, 4>(this->dev, s16t, ConjTrans, Trans, alpha, T(0), "NB16 CT beta=0");
    run_small<16, 4>(this->dev, s16t, Trans, ConjTrans, alpha, beta, "NB16 TC");
    const Shape s8{7, 8, 5, 7 + 3, 5 + 2, 7 + 4};             // NB 8: one chunk, k < NB
    run_small<8, 8>(this->dev, s8, NoTrans, NoTrans, alpha, T(0), "NB8 NN beta=0");
    const Shape s8t{7, 8, 5, 5 + 3, 8 + 2, 7 + 4};
    run_small<8, 8>(this->dev, s8t, ConjTrans, ConjTrans, alpha, beta, "NB8 CC");
}

// ---- the tiled_general shape: op_elem stage, dot_slab with OpArith, epilogue_store -----------------

template <Transpose OpA, Transpose OpB, class T>
void run_tiled_general(Dev& dev, const Shape& s, T alpha, T beta, const char* where) {
    constexpr int Tile = 8;
    const int gm = (s.m + Tile - 1) / Tile, gn = (s.n + Tile - 1) / Tile;
    const std::size_t sa = std::size_t(s.lda) * (OpA == Transpose::NoTrans ? s.k : s.m);
    const std::size_t sb = std::size_t(s.ldb) * (OpB == Transpose::NoTrans ? s.n : s.k), sc = std::size_t(s.ldc) * s.n;
    const auto hA = batched<T>(int(sa), 41), hB = batched<T>(int(sb), 42);
    auto hC = batched<T>(int(sc), 43);
    if (beta == T(0)) std::fill(hC.begin(), hC.end(), nan_of<T>());
    const T* A = dev.up(hA);
    const T* B = dev.up(hB);
    T* C = dev.up(hC);
    const int m = s.m, n = s.n, k = s.k, lda = s.lda, ldb = s.ldb, ldc = s.ldc;
    dev.template run<Tile * Tile>(kBatch * gm * gn, sizeof(T) * 2 * Tile * Tile, [=](sycl::nd_item<1> it, unsigned char* lm) {
        const int g = static_cast<int>(it.get_group_linear_id());
        const int b = g / (gm * gn), rem = g % (gm * gn);
        const int lid = static_cast<int>(it.get_local_linear_id());
        const int local_row = lid / Tile, local_col = lid % Tile;
        const int row = (rem % gm) * Tile + local_row, col = (rem / gm) * Tile + local_col;
        T* const ta = reinterpret_cast<T*>(lm);
        T* const tb = ta + Tile * Tile;
        const T* a = A + std::ptrdiff_t(b) * sa;
        const T* bm = B + std::ptrdiff_t(b) * sb;
        const v2::WorkGroupNd<Tile * Tile, 1> ex{it};
        const v2::Vec<T, v2::Ext<Tile>, v2::Unit, v2::Mem::Local> a_row{ta + local_row * Tile, {}};
        const v2::Vec<T, v2::Ext<Tile>, v2::Ext<Tile>, v2::Mem::Local> b_col{tb + local_col, {}};
        T sum = T(0);
        for (int kk0 = 0; kk0 < k; kk0 += Tile) {
            ta[local_row * Tile + local_col] = (row < m && kk0 + local_col < k) ? v2::op_elem<OpA>(a, lda, row, kk0 + local_col) : T(0);
            tb[local_row * Tile + local_col] = (col < n && kk0 + local_row < k) ? v2::op_elem<OpB>(bm, ldb, kk0 + local_row, col) : T(0);
            ex.barrier();
            v2::dot_slab<Tile, v2::OpArith<T>>(sum, a_row, b_col, kk0, k);
            ex.barrier();
        }
        if (row < m && col < n) v2::epilogue_store(C + std::ptrdiff_t(b) * sc + col * ldc + row, alpha, beta, sum);
    });
    check_gemm(s, OpA, OpB, alpha, beta, hA, hB, hC, dev.down(C, hC.size()), sa, sb, sc, where);
}

TYPED_TEST(DeviceBlasV2GemmComponentsTest, TiledGeneralGemmFromComponents) {
    using T = TypeParam;
    using enum Transpose;
    const T alpha = val<T>(51), beta = val<T>(52);
    run_tiled_general<NoTrans, ConjTrans>(this->dev, Shape{19, 13, 21, 19 + 3, 13 + 4, 19 + 2}, alpha, beta, "NC");
    run_tiled_general<ConjTrans, NoTrans>(this->dev, Shape{19, 13, 21, 21 + 3, 21 + 4, 19 + 2}, alpha, T(0), "CN beta=0");
    run_tiled_general<Trans, Trans>(this->dev, Shape{16, 24, 16, 16 + 1, 24 + 1, 16 + 1}, alpha, beta, "TT whole tiles");
}

// ---- kloop_prefetch2 and kloop_async: every slab once, in order, behind the right barrier ---------

// Lane l sums X(l + 1 mod 32, 0 : 4 * slabs) slab by slab (zero past K), reading its neighbour's
// staged values, so a missing barrier, a skipped or doubled slab or a wrong buffer shows. Exact:
// the host adds the same values in the same order.
template <bool Async, class T> void run_pipeline(Dev& dev, int slabs, const char* where) {
    constexpr int WG = 32, KT = 4, S = 4, ldx = WG + 5;
    const int K = KT * slabs - 1;
    const auto hX = batched<T>(ldx * K, 61 + slabs);
    const T* X = dev.up(hX);
    std::vector<T> hout(std::size_t(kBatch) * WG, nan_of<T>());
    T* out = dev.up(hout);
    const std::size_t sx = std::size_t(ldx) * K;
    dev.template run<WG>(kBatch, sizeof(T) * S * WG * KT, [=](sycl::nd_item<1> it, unsigned char* lm) {
        using A = v2::Arith<T>;
        const int b = static_cast<int>(it.get_group_linear_id());
        const int lane = static_cast<int>(it.get_local_linear_id());
        const T* x = X + std::ptrdiff_t(b) * sx;
        T* const sm = reinterpret_cast<T*>(lm);
        const v2::WorkGroupNd<WG, 1> ex{it};
        T s = A::zero();
        auto compute = [&](int buf) {
            for (int j = 0; j < KT; ++j) s = A::add(s, sm[buf * WG * KT + ((lane + 1) % WG) * KT + j]);
        };
        auto fetch = [&](int k0, int j) { return k0 + j < K ? x[lane + std::ptrdiff_t(k0 + j) * ldx] : A::zero(); };
        if constexpr (Async) {
            auto issue = [&](int k0, auto buf) {
                for (int j = 0; j < KT; ++j) sm[decltype(buf)::value * WG * KT + lane * KT + j] = fetch(k0, j);
            };
            v2::kloop_async<S, KT>(ex, slabs, issue, [] {}, [] {}, [] {}, compute);
        } else {
            T reg[KT];
            auto gload = [&](int k0) {
                for (int j = 0; j < KT; ++j) reg[j] = fetch(k0, j);
            };
            auto sstore = [&](int buf) {
                for (int j = 0; j < KT; ++j) sm[buf * WG * KT + lane * KT + j] = reg[j];
            };
            v2::kloop_prefetch2<KT>(ex, slabs, gload, sstore, compute);
        }
        out[std::ptrdiff_t(b) * WG + lane] = s;
    });
    const auto got = dev.down(out, hout.size());
    int bad = 0;
    for (int b = 0; b < kBatch; ++b)
        for (int lane = 0; lane < WG; ++lane) {
            const int src = (lane + 1) % WG;
            T ref = T(0);
            for (int kk = 0; kk < KT * slabs; ++kk)
                ref = ref + (kk < K ? hX[(b % kPeriod) * sx + src + std::size_t(kk) * ldx] : T(0));
            bad += !same_bits(got[std::size_t(b) * WG + lane], ref);
        }
    EXPECT_EQ(bad, 0) << where << " slabs " << slabs;
}

TYPED_TEST(DeviceBlasV2GemmComponentsTest, PipelinesRunEverySlabOnce) {
    using T = TypeParam;
    for (int slabs : {1, 2, 3, 4, 5, 8, 9}) {
        run_pipeline<false, T>(this->dev, slabs, "kloop_prefetch2");
        run_pipeline<true, T>(this->dev, slabs, "kloop_async<4>");
    }
}

// ---- elementwise: op_elem, epilogues, axpby_value, sum_seq, outer_mac, FmaArith -------------------

enum Slot { kOpN, kOpT, kOpC, kEpiZero, kEpiStore, kEpiStoreZero, kAxpby, kFusedZero, kFused, kSum, kNegZeroEpi, kNegZeroSum,
            kFma, kMul, kOuter0, kSlots = kOuter0 + 15 };

TYPED_TEST(DeviceBlasV2GemmComponentsTest, ElementwiseOpsMatchTheHost) {
    using T = TypeParam;
    constexpr int ld = 5, nsum = 7;
    std::vector<T> hx(3 * ld), hs(nsum), ha(3), hb(5);
    for (int i = 0; i < 3 * ld; ++i) hx[i] = val<T>(100 + i);
    for (int i = 0; i < nsum; ++i) hs[i] = val<T>(200 + i);
    for (int i = 0; i < 3; ++i) ha[i] = val<T>(300 + i);
    for (int i = 0; i < 5; ++i) hb[i] = val<T>(400 + i);
    const T alpha = val<T>(501), beta = val<T>(502), acc = val<T>(503), prior = val<T>(504);
    const T* x = this->dev.up(hx);
    const T* sv = this->dev.up(hs);
    const T* av = this->dev.up(ha);
    const T* bv = this->dev.up(hb);
    std::vector<T> hout(kSlots + 2, nan_of<T>());
    T* out = this->dev.up(hout);
    this->dev.template run<32>(1, 0, [=](sycl::nd_item<1> it, unsigned char*) {
        if (it.get_local_linear_id() != 0) return;
        using enum Transpose;
        out[kOpN] = v2::op_elem<NoTrans>(x, ld, 1, 2);
        out[kOpT] = v2::op_elem<Trans>(x, ld, 2, 1);
        out[kOpC] = v2::op_elem<ConjTrans>(x, ld, 2, 1);
        out[kEpiZero] = v2::epilogue_apply(alpha, T(0), acc, out[kSlots]);       // prior NaN, never read
        out[kEpiStore] = prior;
        v2::epilogue_store(out + kEpiStore, alpha, beta, acc);
        v2::epilogue_store(out + kEpiStoreZero, alpha, T(0), acc);                 // NaN destination
        out[kAxpby] = v2::axpby_value(alpha, acc, beta, prior);
        v2::epilogue_store_fused<v2::FmaArith<T>>(out + kFusedZero, alpha, beta, true, acc);  // NaN destination
        out[kFused] = prior;
        v2::epilogue_store_fused<v2::FmaArith<T>>(out + kFused, alpha, beta, false, acc);
        out[kSum] = v2::sum_seq(sv, 3, 2);                                          // sv[0] + sv[2] + sv[4]
        const T nz = -T(0);
        out[kNegZeroEpi] = v2::epilogue_apply(T(1), T(0), nz, out[kSlots]);        // -0 survives beta == 0
        const T negz[2] = {nz, nz};
        out[kNegZeroSum] = v2::sum_seq(negz, 2, 1);                                 // -0 + -0 = -0 (a +0 start gives +0)
        T f = prior;
        v2::FmaArith<T>::mac(f, alpha, acc);
        out[kFma] = f;
        out[kMul] = v2::FmaArith<T>::mul(alpha, acc);
        T a3[3], b5[5], o[3][5];
        for (int i = 0; i < 3; ++i) a3[i] = av[i];
        for (int j = 0; j < 5; ++j) b5[j] = bv[j];
        v2::frag_fill(o, prior);
        v2::outer_mac(o, a3, b5);
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 5; ++j) out[kOuter0 + i * 5 + j] = o[i][j];
    });
    const auto g = this->dev.down(out, hout.size());
    const auto bits = [&](int slot, T ref) { EXPECT_TRUE(same_bits(g[slot], ref)) << "slot " << slot; };
    bits(kOpN, hx[1 + 2 * ld]);
    bits(kOpT, hx[1 + 2 * ld]);
    bits(kOpC, conj_h(hx[1 + 2 * ld]));
    using RR = real_of<T>;
    auto hfma = [](T a, T b, T c) -> T {  // host fma in the device order (bit-exact: both correctly rounded)
        if constexpr (is_cplx<T>) {
            RR re = std::fma(a.real(), b.real(), c.real());
            re = std::fma(-a.imag(), b.imag(), re);
            RR im = std::fma(a.real(), b.imag(), c.imag());
            im = std::fma(a.imag(), b.real(), im);
            return T(re, im);
        } else {
            return std::fma(a, b, c);
        }
    };
    bits(kFma, hfma(alpha, acc, prior));
    if constexpr (is_cplx<T>)
        bits(kMul, T(std::fma(alpha.real(), acc.real(), -alpha.imag() * acc.imag()),
                     std::fma(alpha.real(), acc.imag(), alpha.imag() * acc.real())));
    else bits(kMul, alpha * acc);
    // Rounding-level checks for the operator forms (contraction may fuse them).
    const auto near = [&](int slot, LD<T> ref, long double scale) { EXPECT_VERIFY(T, Check::blas, 2, rel(g[slot], ref, scale)) << "slot " << slot; };
    const LD<T> A = alpha, B = beta, X = acc, P = prior;
    const long double sc = std::abs(A) * std::abs(X) + std::abs(B) * std::abs(P);
    if constexpr (is_cplx<T>) {  // the fused complex epilogue: mul, then one fma-chain mac, bit for bit
        const T fused = T(std::fma(alpha.real(), acc.real(), -alpha.imag() * acc.imag()),
                          std::fma(alpha.real(), acc.imag(), alpha.imag() * acc.real()));
        bits(kFusedZero, fused);
        bits(kFused, hfma(beta, prior, fused));
    } else {  // a real scalar takes the operators
        bits(kFusedZero, alpha * acc);
        near(kFused, A * X + B * P, sc);
        EXPECT_TRUE(std::signbit(g[kNegZeroEpi])) << "beta == 0 must keep -0";  // complex 1 * -0 is +0 (Annex G)
    }
    EXPECT_TRUE(std::signbit(v2::Arith<T>::re(g[kNegZeroSum]))) << "sum_seq must start from x(0)";
    near(kEpiZero, A * X, sc);
    near(kEpiStore, A * X + B * P, sc);
    near(kEpiStoreZero, A * X, sc);
    near(kAxpby, A * X + B * P, sc);
    near(kSum, LD<T>(hs[0]) + LD<T>(hs[2]) + LD<T>(hs[4]), std::abs(LD<T>(hs[0])) + std::abs(LD<T>(hs[2])) + std::abs(LD<T>(hs[4])));
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 5; ++j)
            near(kOuter0 + i * 5 + j, P + LD<T>(ha[i]) * LD<T>(hb[j]), std::abs(P) + std::abs(LD<T>(ha[i])) * std::abs(LD<T>(hb[j])));
}

}  // namespace
