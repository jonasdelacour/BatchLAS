// SPDX-License-Identifier: MIT
// Device LA v2, symmetric-indefinite ops (include/batchlas/blas/device/v2/ext/sytrf-sytrs.hh)
// against long-double host references.
// Guards (docs/developer/agent-guide.md section 8): non-natural ld, the index-reversed view
// (rev = true, the Upper route), complex data with nonzero imaginary parts, the unread triangle
// and the trsv diagonal NaN-poisoned, batch 1024 with period-3 data where every item must be
// bit-identical to its representative, arg-max ties inside one owner and across owners and
// sub-groups at widths 32 / 128 / 256 and the sub-group-only form, NaN keys, an empty range,
// the Bunch-Kaufman thresholds straddled by one ulp, a 2 x 2 block across the lane-31 / 32
// register boundary.
#include <gtest/gtest.h>
#include <tests/test_utils.hh>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include <batchlas/blas/device/v2/dla.hh>
#include <batchlas/blas/device/v2/ext/sytrf-sytrs.hh>
#include <batchlas/util/sycl-local-accessor-helpers.hh>
#include <src/kernels/common/dla_v2_scalar.hh>
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
template <class T> T nanv() { return T(std::numeric_limits<real_of<T>>::quiet_NaN()); }

// kPeriod representatives (edited by fix(item pointer)), copied to every item b from b % kPeriod.
template <class T, class Fix> std::vector<T> batched(int per_item, std::uint64_t seed, Fix fix, int items = kBatch) {
    std::vector<T> h(static_cast<std::size_t>(items) * per_item);
    for (int b = 0; b < std::min(kPeriod, items); ++b) {
        for (int i = 0; i < per_item; ++i) h[std::size_t(b) * per_item + i] = val<T>(seed + std::uint64_t(b) * 1000003u + i);
        fix(h.data() + std::size_t(b) * per_item);
    }
    for (int b = kPeriod; b < items; ++b)
        std::copy_n(h.data() + std::size_t(b % kPeriod) * per_item, per_item, h.data() + std::size_t(b) * per_item);
    return h;
}

template <class T> bool same_bits(const T& a, const T& b) { return std::memcmp(&a, &b, sizeof(T)) == 0; }

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
    template <int WG, class F> void run(int batch, std::size_t local_bytes, F f) {
        q().submit([&](sycl::handler& h) {
             sycl::local_accessor<unsigned char, 1> lm(sycl::range<1>(std::max<std::size_t>(local_bytes, 16)), h);
             h.parallel_for(sycl::nd_range<1>(std::size_t(batch) * WG, WG),
                            [=](sycl::nd_item<1> it) BATCHLAS_REQD_SG_SIZE(32) { f(it, batchlas::util::get_raw_ptr(lm)); });
         }).wait_and_throw();
    }
    ~Dev() {
        for (void* p : ptrs) sycl::free(p, q());
    }
    std::vector<void*> ptrs;
};

template <class T> class DeviceBlasV2SytrfTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!dev.ctx.device().supports_sub_group_size(32)) GTEST_SKIP() << "v2 tests launch sub-group size 32";
    }
    Dev dev;
};

using Types = ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(DeviceBlasV2SytrfTest, Types);

template <class T> using LD = std::conditional_t<is_cplx<T>, std::complex<long double>, long double>;
template <class T> LD<T> wide(T v) { return LD<T>(v); }

template <class T> double rel(T got, LD<T> ref, long double scale) {
    const long double d = std::abs(LD<T>(got) - ref);
    if (std::isnan(d)) return std::numeric_limits<double>::infinity();
    return static_cast<double>(d / std::max<long double>(scale, std::numeric_limits<real_of<T>>::min()));
}

// Physical offset of logical (i, j) in an m x n RevMat with leading dimension ld.
inline std::size_t phys(int i, int j, int m, int n, int ld, bool rev) {
    const int pi = rev ? m - 1 - i : i, pj = rev ? n - 1 - j : j;
    return std::size_t(pi) + std::size_t(pj) * ld;
}

// Every item equal to its representative, bit for bit.
template <class T> int twins_differ(const std::vector<T>& got, std::size_t per_item) {
    int bad = 0;
    for (int b = kPeriod; b < kBatch; ++b)
        for (std::size_t e = 0; e < per_item; ++e) bad += !same_bits(got[b * per_item + e], got[(b % kPeriod) * per_item + e]);
    return bad;
}

// ---- level 1 over index ranges on the reversed tile ---------------------------------------

// swap (column against row: ?SYTF2's interchange), copy, scal, axpy / axpy2 (Minus), fill and
// map_range on rev = true with ld = n + 5; a barrier between ops. Element scales are tracked
// alongside the long-double reference.
TYPED_TEST(DeviceBlasV2SytrfTest, RangeOpsOnReversedTile) {
    using T = TypeParam;
    constexpr int WG = 64, n = 37, ld = n + 5;
    const std::size_t per = std::size_t(ld) * n;
    const T a = val<T>(71), c = val<T>(72), v = val<T>(73);
    auto hA = batched<T>(int(per), 11, [](T*) {});
    T* A = this->dev.up(hA);
    this->dev.template run<WG>(kBatch, 0, [=](sycl::nd_item<1> it, unsigned char*) {
        const auto b = static_cast<std::ptrdiff_t>(it.get_group_linear_id());
        const v2::RtGroup g{it, static_cast<int>(it.get_local_linear_id()), WG};
        const auto S = v2::rev_mat(A + b * std::ptrdiff_t(per), ld, n, true);
        v2::swap(g, v2::col_of(S, 3), v2::row_of(S, 9), {4, 9});
        g.barrier();
        v2::copy(g, v2::col_of(S, 5), v2::col_of(S, 7), {8, n});
        g.barrier();
        v2::scal(g, a, v2::col_of(S, 2), {3, n});
        g.barrier();
        v2::axpy(g, v2::Minus{c}, v2::col_of(S, 0), v2::col_of(S, 1), {2, n});
        g.barrier();
        v2::axpy2(g, v2::Minus{a}, v2::col_of(S, 10), v2::Minus{c}, v2::col_of(S, 11), v2::col_of(S, 12), {13, n});
        g.barrier();
        v2::fill(g, v, v2::col_of(S, 20), {21, 30});
        g.barrier();
        v2::map_range(g, v2::col_of(S, 25), {26, n}, [&](int i) { return (i & 1) ? v2::sfp::mul(a, static_cast<T>(S(i, 24))) : S(i, 24); });
    });
    const auto got = this->dev.down(A, hA.size());
    EXPECT_EQ(twins_differ(got, per), 0) << "items differ from their representative";
    double worst = 0;
    for (int b = 0; b < kPeriod; ++b) {
        std::vector<LD<T>> r(per);
        std::vector<long double> s(per);
        for (std::size_t e = 0; e < per; ++e) r[e] = wide(hA[b * per + e]), s[e] = std::abs(r[e]);
        auto at = [&](int i, int j) { return phys(i, j, n, n, ld, true); };
        const LD<T> A_ = wide(a), C_ = wide(c), V_ = wide(v);
        for (int i = 4; i < 9; ++i) std::swap(r[at(i, 3)], r[at(9, i)]), std::swap(s[at(i, 3)], s[at(9, i)]);
        for (int i = 8; i < n; ++i) r[at(i, 7)] = r[at(i, 5)], s[at(i, 7)] = s[at(i, 5)];
        for (int i = 3; i < n; ++i) r[at(i, 2)] *= A_, s[at(i, 2)] *= std::abs(A_);
        for (int i = 2; i < n; ++i) r[at(i, 1)] -= r[at(i, 0)] * C_, s[at(i, 1)] += s[at(i, 0)] * std::abs(C_);
        for (int i = 13; i < n; ++i) {
            r[at(i, 12)] -= r[at(i, 10)] * A_ + r[at(i, 11)] * C_;
            s[at(i, 12)] += s[at(i, 10)] * std::abs(A_) + s[at(i, 11)] * std::abs(C_);
        }
        for (int i = 21; i < 30; ++i) r[at(i, 20)] = V_, s[at(i, 20)] = std::abs(V_);
        for (int i = 26; i < n; ++i) {
            const bool odd = i & 1;
            r[at(i, 25)] = odd ? A_ * r[at(i, 24)] : r[at(i, 24)];
            s[at(i, 25)] = odd ? std::abs(A_) * s[at(i, 24)] : s[at(i, 24)];
        }
        for (std::size_t e = 0; e < per; ++e) worst = std::max(worst, rel(got[b * per + e], r[e], s[e]));
    }
    EXPECT_VERIFY(T, Check::blas, 2, worst);
}

// ---- rank-1 / rank-2 trailing updates and the 2 x 2 pivot block ----------------------------

// syr (alpha x x^T), then ?SYTF2's rank-2 step: gemmt2_sub with w = [x0 x1] D^-1 computed on the
// fly, then bk_apply2 in place, on the logical lower triangle; the strict upper triangle and the
// padding rows are NaN and must stay NaN (never written) without reaching any result (never read).
TYPED_TEST(DeviceBlasV2SytrfTest, SyrGemmt2Apply2LowerOnly) {
    using T = TypeParam;
    using R = real_of<T>;
    constexpr int WG = 128, n = 45, ld = n + 3;
    const std::size_t per = std::size_t(ld) * n;
    const T alpha = val<T>(81);
    for (const bool rev : {false, true}) {
        auto hA = batched<T>(int(per), 21 + rev, [&](T* p) {
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < ld; ++i) {
                    const int li = rev ? n - 1 - i : i, lj = rev ? n - 1 - j : j;  // logical of physical (i, j)
                    if (i >= n || li < lj) p[std::size_t(i) + std::size_t(j) * ld] = nanv<T>();
                }
            p[phys(1, 1, n, n, ld, rev)] *= R(0.1);  // a well-conditioned 2 x 2 pivot: small diagonal, large off-diagonal
            p[phys(2, 2, n, n, ld, rev)] *= R(0.1);
            p[phys(2, 1, n, n, ld, rev)] = T(R(2.5)) + p[phys(2, 1, n, n, ld, rev)] * R(0.25);
        });
        T* A = this->dev.up(hA);
        this->dev.template run<WG>(kBatch, 0, [=](sycl::nd_item<1> it, unsigned char*) {
            const auto b = static_cast<std::ptrdiff_t>(it.get_group_linear_id());
            const v2::RtGroup g{it, static_cast<int>(it.get_local_linear_id()), WG};
            const auto S = v2::rev_mat(A + b * std::ptrdiff_t(per), ld, n, rev);
            v2::syr<Uplo::Lower>(g, alpha, v2::col_of(S, 0), S, {3, n});
            g.barrier();
            const auto blk = v2::bk_block2(static_cast<T>(S(1, 1)), static_cast<T>(S(2, 1)), static_cast<T>(S(2, 2)));
            const auto c0 = v2::col_of(S, 1), c1 = v2::col_of(S, 2);
            v2::gemmt2_sub<Uplo::Lower>(g, c0, [&](int j) { return blk.first(c0(j), c1(j)); }, c1,
                                        [&](int j) { return blk.second(c0(j), c1(j)); }, S, {3, n});
            g.barrier();
            v2::bk_apply2(g, blk, c0, c1, c0, c1, {3, n});
        });
        const auto got = this->dev.down(A, hA.size());
        EXPECT_EQ(twins_differ(got, per), 0) << "rev " << rev << ": items differ from their representative";
        double worst = 0;
        int poison = 0;
        for (int b = 0; b < kPeriod; ++b) {
            const T* h = hA.data() + b * per;
            const T* gb = got.data() + b * per;
            auto L = [&](int i, int j) { return wide(h[phys(i, j, n, n, ld, rev)]); };
            const LD<T> Al = wide(alpha), d21 = L(2, 1), d11 = L(2, 2) / d21, d22 = L(1, 1) / d21;
            const LD<T> e21 = (LD<T>(1) / (d11 * d22 - LD<T>(1))) / d21;
            const long double ae = std::abs(e21);
            std::vector<LD<T>> w0(n), w1(n);
            std::vector<long double> s0(n), s1(n);
            for (int j = 3; j < n; ++j) {
                w0[j] = e21 * (d11 * L(j, 1) - L(j, 2)), w1[j] = e21 * (d22 * L(j, 2) - L(j, 1));
                s0[j] = ae * (std::abs(d11) * std::abs(L(j, 1)) + std::abs(L(j, 2)));
                s1[j] = ae * (std::abs(d22) * std::abs(L(j, 2)) + std::abs(L(j, 1)));
            }
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i) {
                    const T gv = gb[phys(i, j, n, n, ld, rev)];
                    if (i < j) {
                        poison += !std::isnan(std::real(gv));
                        continue;
                    }
                    LD<T> ref = L(i, j);
                    long double sc = std::abs(ref);
                    if (j >= 3) {
                        ref += Al * L(i, 0) * L(j, 0);
                        sc += std::abs(Al) * std::abs(L(i, 0)) * std::abs(L(j, 0));
                        ref -= L(i, 1) * w0[j] + L(i, 2) * w1[j];
                        sc += std::abs(L(i, 1)) * s0[j] + std::abs(L(i, 2)) * s1[j];
                    } else if (i >= 3 && (j == 1 || j == 2)) {
                        ref = j == 1 ? w0[i] : w1[i];
                        sc = j == 1 ? s0[i] : s1[i];
                    }
                    worst = std::max(worst, rel(gv, ref, sc));
                }
            for (int j = 0; j < n; ++j)
                for (int i = n; i < ld; ++i) poison += !std::isnan(std::real(gb[std::size_t(i) + std::size_t(j) * ld]));
        }
        EXPECT_EQ(poison, 0) << "rev " << rev << ": a poisoned (unreferenced) element was written";
        EXPECT_VERIFY(T, Check::blas, 6, worst) << "rev " << rev;
    }
}

// ---- gemv_sub, gemmt_block_sub, lacpy_flat, fill_rows -----------------------------------------

// ?LASYF's column update (gemv_sub: b == y in place, and b != y into W) and the blocked trailing
// update of one diagonal block (gemmt_block_sub, rank 6), on reversed S (ld n + 2) and reversed W
// (n x 6, ld n + 1); then the lower triangle copied flat (NonUnit / Unit, int / ptrdiff index) into
// NaN-poisoned buffers, and fill_rows.
TYPED_TEST(DeviceBlasV2SytrfTest, GemvSubGemmtBlockLacpy) {
    using T = TypeParam;
    constexpr int WG = 64, n = 40, ldS = n + 2, nw = 6, ldW = n + 1, ldB = n + 4;
    const std::size_t pS = std::size_t(ldS) * n, pW = std::size_t(ldW) * nw, pB = std::size_t(ldB) * n;
    const T fv = val<T>(91);
    auto hS = batched<T>(int(pS), 31, [](T*) {});
    auto hW = batched<T>(int(pW), 32, [](T*) {});
    std::vector<T> hB(kBatch * pB, nanv<T>()), hC(kBatch * pB, nanv<T>());
    T* S_ = this->dev.up(hS);
    T* W_ = this->dev.up(hW);
    T* B_ = this->dev.up(hB);
    T* C_ = this->dev.up(hC);
    this->dev.template run<WG>(kBatch, 0, [=](sycl::nd_item<1> it, unsigned char*) {
        const auto b = static_cast<std::ptrdiff_t>(it.get_group_linear_id());
        const v2::RtGroup g{it, static_cast<int>(it.get_local_linear_id()), WG};
        const auto S = v2::rev_mat(S_ + b * std::ptrdiff_t(pS), ldS, n, true);
        const auto W = v2::rev_mat(W_ + b * std::ptrdiff_t(pW), ldW, n, nw, true);
        const auto L = v2::cols_from(S, 2);
        v2::gemv_sub(g, L, v2::row_of(W, 30), v2::col_of(S, 30), v2::col_of(S, 30), {30, n}, 1, 5);
        v2::gemv_sub(g, L, v2::row_of(W, 30), v2::col_of(S, 31), v2::col_of(W, 0), {31, n}, 0, nw);
        v2::gemmt_block_sub<Uplo::Lower>(g, L, W, S, 10, 12, nw);
        g.barrier(v2::Mem::Global);
        v2::lacpy_flat<Uplo::Lower>(g, S, v2::rev_mat(B_ + b * std::ptrdiff_t(pB), ldB, n, false), n);
        v2::lacpy_flat<Uplo::Lower, Diag::Unit, std::ptrdiff_t>(g, S, v2::rev_mat(C_ + b * std::ptrdiff_t(pB), ldB, n, false), n);
        v2::fill_rows(g, fv, W, {0, 5}, 3);
    });
    const auto gS = this->dev.down(S_, hS.size()), gW = this->dev.down(W_, hW.size());
    const auto gB = this->dev.down(B_, hB.size()), gC = this->dev.down(C_, hC.size());
    EXPECT_EQ(twins_differ(gS, pS) + twins_differ(gW, pW) + twins_differ(gB, pB) + twins_differ(gC, pB), 0)
        << "items differ from their representative";
    double worst = 0;
    int copy_bad = 0;
    for (int b = 0; b < kPeriod; ++b) {
        auto Sat = [&](int i, int j) { return phys(i, j, n, n, ldS, true); };
        auto Wat = [&](int i, int c) { return phys(i, c, n, nw, ldW, true); };
        const T* hs = hS.data() + b * pS;
        const T* hw = hW.data() + b * pW;
        std::vector<LD<T>> rS(pS), rW(pW);
        std::vector<long double> sS(pS), sW(pW);
        for (std::size_t e = 0; e < pS; ++e) rS[e] = wide(hs[e]), sS[e] = std::abs(rS[e]);
        for (std::size_t e = 0; e < pW; ++e) rW[e] = wide(hw[e]), sW[e] = std::abs(rW[e]);
        for (int i = 30; i < n; ++i)
            for (int l = 1; l < 5; ++l) {
                rS[Sat(i, 30)] -= wide(hs[Sat(i, 2 + l)]) * wide(hw[Wat(30, l)]);
                sS[Sat(i, 30)] += std::abs(wide(hs[Sat(i, 2 + l)])) * std::abs(wide(hw[Wat(30, l)]));
            }
        for (int i = 31; i < n; ++i) {
            LD<T> acc = wide(hs[Sat(i, 31)]);
            long double sc = std::abs(acc);
            for (int l = 0; l < nw; ++l) {
                acc -= wide(hs[Sat(i, 2 + l)]) * wide(hw[Wat(30, l)]);
                sc += std::abs(wide(hs[Sat(i, 2 + l)])) * std::abs(wide(hw[Wat(30, l)]));
            }
            rW[Wat(i, 0)] = acc, sW[Wat(i, 0)] = sc;
        }
        for (int j = 10; j < 22; ++j)
            for (int i = j; i < 22; ++i)
                for (int l = 0; l < nw; ++l) {
                    rS[Sat(i, j)] -= wide(hs[Sat(i, 2 + l)]) * wide(hw[Wat(j, l)]);
                    sS[Sat(i, j)] += std::abs(wide(hs[Sat(i, 2 + l)])) * std::abs(wide(hw[Wat(j, l)]));
                }
        for (int i = 0; i < 5; ++i)
            for (int c = 0; c < 3; ++c) rW[Wat(i, c)] = wide(fv), sW[Wat(i, c)] = std::abs(wide(fv));
        for (std::size_t e = 0; e < pS; ++e) worst = std::max(worst, rel(gS[b * pS + e], rS[e], sS[e]));
        for (std::size_t e = 0; e < pW; ++e) worst = std::max(worst, rel(gW[b * pW + e], rW[e], sW[e]));
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < ldB; ++i) {
                const T gb = gB[b * pB + std::size_t(i) + std::size_t(j) * ldB], gc = gC[b * pB + std::size_t(i) + std::size_t(j) * ldB];
                const bool in = i < n;
                copy_bad += in && i >= j ? !same_bits(gb, gS[b * pS + Sat(i, j)]) : !std::isnan(std::real(gb));
                copy_bad += in && i > j ? !same_bits(gc, gS[b * pS + Sat(i, j)]) : !std::isnan(std::real(gc));
            }
    }
    EXPECT_EQ(copy_bad, 0) << "lacpy_flat wrote outside its triangle or changed a value";
    EXPECT_VERIFY(T, Check::blas, nw, worst);
}

// ---- the deterministic arg-max and sums --------------------------------------------------------

enum IamaxSlot { kLow, kHigh, kSkip, kEmpty, kNanLow, kNanHigh, kISlots };
enum SumSlot { kVal, kEmptyVal, kS0, kS1, kS1Off, kSSlots };

// x: three tied maxima (|re| + |im| = 4) at 10, 42, 138 (one owner at width 32; 10 / 138 one owner
// at 128); y: NaN at 20 (real part) and 77 (imaginary part), 1e30 at 50; dotu2_sum on x, z.
template <class T, int WG, bool SgOnly>
void run_iamax(Dev& dev, int n, const std::vector<T>& hx, const std::vector<T>& hy, const std::vector<T>& hz, std::vector<int>& oi,
               std::vector<T>& ov) {
    using R = real_of<T>;
    T* x = dev.up(hx);
    T* y = dev.up(hy);
    T* z = dev.up(hz);
    std::vector<int> hoi(std::size_t(kBatch) * kISlots, -7);
    std::vector<T> hov(std::size_t(kBatch) * kSSlots, nanv<T>());
    int* oip = dev.up(hoi);
    T* ovp = dev.up(hov);
    dev.template run<WG>(kBatch, (2 * sizeof(R) + sizeof(int)) * 32, [=](sycl::nd_item<1> it, unsigned char* lm) {
        const auto b = static_cast<std::ptrdiff_t>(it.get_group_linear_id());
        const auto sg = it.get_sub_group();
        if (SgOnly && sg.get_group_linear_id() != 0) return;
        const int lid = SgOnly ? static_cast<int>(sg.get_local_linear_id()) : static_cast<int>(it.get_local_linear_id());
        const v2::RtGroup g{it, lid, SgOnly ? 32 : WG, SgOnly};
        R* sv = reinterpret_cast<R*>(lm);
        const v2::ArgMaxScratch<R> am{sv, reinterpret_cast<int*>(sv + 64)};
        const v2::SumScratch<R> ss{sv, sv + 32};
        const auto xv = v2::vec(x + b * n, n), yv = v2::vec(y + b * n, n), zv = v2::vec(z + b * n, n);
        const auto lo = v2::iamax_bk(g, am, xv, {3, n}, true);
        const auto hi = v2::iamax_bk(g, am, xv, {3, n}, false);
        const auto sk = v2::iamax_bk(g, am, xv, {0, n}, true, v2::SkipAt{lo.i});
        const auto em = v2::iamax_bk(g, am, xv, {n, n}, true);
        const auto nl = v2::iamax_bk(g, am, yv, {0, n}, true);
        const auto nh = v2::iamax_bk(g, am, yv, {0, n}, false);
        const auto s2 = v2::dotu2_sum<T>(g, ss, true, xv, zv, zv, {60, n});
        const auto s1 = v2::dotu2_sum<T>(g, ss, false, zv, xv, zv, {1, n});
        if (g.leader()) {
            int* o = oip + b * kISlots;
            o[kLow] = lo.i, o[kHigh] = hi.i, o[kSkip] = sk.i, o[kEmpty] = em.i, o[kNanLow] = nl.i, o[kNanHigh] = nh.i;
            T* q = ovp + b * kSSlots;
            q[kVal] = T(lo.v), q[kEmptyVal] = T(em.v), q[kS0] = s2[0], q[kS1] = s2[1], q[kS1Off] = s1[1];
        }
    });
    oi = dev.down(oip, hoi.size());
    ov = dev.down(ovp, hov.size());
}

template <class T> void check_iamax(int n, const std::vector<T>& hx, const std::vector<T>& hz,
                                    const std::vector<int>& oi, const std::vector<T>& ov, const char* where) {
    int bad = 0;
    for (int b = 0; b < kBatch; ++b) {
        for (int k = 0; k < kISlots; ++k) bad += oi[std::size_t(b) * kISlots + k] != oi[std::size_t(b % kPeriod) * kISlots + k];
        for (int k = 0; k < kSSlots; ++k) bad += !same_bits(ov[std::size_t(b) * kSSlots + k], ov[std::size_t(b % kPeriod) * kSSlots + k]);
    }
    EXPECT_EQ(bad, 0) << where << ": items differ from their representative";
    for (int b = 0; b < kPeriod; ++b) {
        const int* o = oi.data() + std::size_t(b) * kISlots;
        const T* q = ov.data() + std::size_t(b) * kSSlots;
        EXPECT_EQ(o[kLow], 10) << where;
        EXPECT_EQ(o[kHigh], 138) << where;
        EXPECT_EQ(o[kSkip], 42) << where;
        EXPECT_EQ(o[kEmpty], -1) << where;
        EXPECT_EQ(o[kNanLow], 20) << where << " (NaN ranks as Inf)";
        EXPECT_EQ(o[kNanHigh], 77) << where;
        EXPECT_EQ(std::real(q[kVal]), real_of<T>(4)) << where;
        EXPECT_EQ(std::real(q[kEmptyVal]), real_of<T>(-1)) << where;
        EXPECT_TRUE(same_bits(q[kS1Off], T(0))) << where << " two == false gives an exact zero";
        LD<T> d0 = 0, d1 = 0;
        long double a0 = 0, a1 = 0;
        for (int i = 60; i < n; ++i) {
            const LD<T> xi = wide(hx[std::size_t(b) * n + i]), zi = wide(hz[std::size_t(b) * n + i]);
            d0 += xi * zi, d1 += zi * zi;
            a0 += std::abs(xi) * std::abs(zi), a1 += std::abs(zi) * std::abs(zi);
        }
        EXPECT_VERIFY(T, Check::blas, n, std::max(rel(q[kS0], d0, a0), rel(q[kS1], d1, a1))) << where << " dotu2_sum";
    }
}

TYPED_TEST(DeviceBlasV2SytrfTest, IamaxBkTiesNanAndSums) {
    using T = TypeParam;
    using R = real_of<T>;
    constexpr int n = 150;
    auto mk = [](R re, R im) {
        if constexpr (is_cplx<T>) return T(re, im);
        else return T(re + (re < 0 ? -std::abs(im) : std::abs(im)));
    };
    const auto hx = batched<T>(n, 41, [&](T* p) {
        p[10] = mk(R(3), R(-1)), p[42] = mk(R(-1), R(3)), p[138] = mk(R(-2), R(2));
    });
    const auto hy = batched<T>(n, 42, [&](T* p) {
        p[20] = T(std::numeric_limits<R>::quiet_NaN());
        if constexpr (is_cplx<T>) p[77] = T(R(0.5), std::numeric_limits<R>::quiet_NaN());
        else p[77] = std::numeric_limits<R>::quiet_NaN();
        p[50] = T(R(1e30));
    });
    const auto hz = batched<T>(n, 43, [](T*) {});
    std::vector<int> oi;
    std::vector<T> ov;
    run_iamax<T, 32, false>(this->dev, n, hx, hy, hz, oi, ov);
    check_iamax(n, hx, hz, oi, ov, "width 32");
    run_iamax<T, 128, false>(this->dev, n, hx, hy, hz, oi, ov);
    check_iamax(n, hx, hz, oi, ov, "width 128");
    run_iamax<T, 256, false>(this->dev, n, hx, hy, hz, oi, ov);
    check_iamax(n, hx, hz, oi, ov, "width 256");
    run_iamax<T, 64, true>(this->dev, n, hx, hy, hz, oi, ov);
    check_iamax(n, hx, hz, oi, ov, "sub-group 0 of 64 (sg_only)");
}

// ---- Bunch-Kaufman decisions and the 2 x 2 blocks -------------------------------------------

// bk_stage / bk_choice over cases straddling each threshold by one ulp (LAPACK ?SYTF2's tests
// written out here); the result is item 0's broadcast.
TYPED_TEST(DeviceBlasV2SytrfTest, BkStageAndChoiceThresholds) {
    using T = TypeParam;
    using R = real_of<T>;
    // The thresholds are built on the device's alpha (a float sqrt need not be correctly rounded).
    std::vector<R> ha(1, R(0));
    R* ap = this->dev.up(ha);
    this->dev.template run<32>(1, 0, [=](sycl::nd_item<1> it, unsigned char*) {
        if (it.get_local_linear_id() == 0) ap[0] = v2::bk_alpha<R>();
    });
    const R alpha = this->dev.down(ap, 1)[0];
    EXPECT_NEAR(alpha, (R(1) + std::sqrt(R(17))) / R(8), 2 * std::numeric_limits<R>::epsilon());
    const R nan = std::numeric_limits<R>::quiet_NaN();
    auto below = [](R x) { return std::nextafter(x, R(0)); };
    auto above = [](R x) { return std::nextafter(x, R(1e30)); };
    // {absakk, colmax, rowmax, absimax}
    std::vector<std::array<R, 4>> cases = {
        {0, 0, 1, 1}, {nan, 1, 1, 1}, {nan, 0, 1, 1}, {1, 0, 1, 1},                              // zero / NaN / keep
        {alpha * 2, 2, 1, 1}, {below(alpha * 2), 2, 3, 1},                                       // keep at the line, row search below it
        {R(0.5), 1, 1, 1}, {R(0.1), 1, 1, alpha}, {R(0.1), 1, 1, below(alpha)},                   // absimax straddles alpha rowmax
        {R(0.1), 1, R(0.2), R(0.01)}, {above(alpha / 4), 1, 4, 0}, {alpha / 4, 1, 4, 0}, {below(alpha / 4), 1, 4, 0}};
    const int nc = static_cast<int>(cases.size());
    std::vector<R> hin;
    for (const auto& c : cases) hin.insert(hin.end(), c.begin(), c.end());
    std::vector<int> hout(std::size_t(nc) * 3, -9);
    R* in = this->dev.up(hin);
    int* out = this->dev.up(hout);
    this->dev.template run<32>(nc, 0, [=](sycl::nd_item<1> it, unsigned char*) {
        const int c = static_cast<int>(it.get_group_linear_id());
        const v2::RtGroup g{it, static_cast<int>(it.get_local_linear_id()), 32};
        const R* p = in + 4 * c;
        // only item 0 sees the real inputs: the verdict must be its broadcast
        const bool l0 = g.leader();
        const R absakk = l0 ? p[0] : R(-5), colmax = l0 ? p[1] : R(7), rowmax = l0 ? p[2] : R(1), absimax = l0 ? p[3] : R(9);
        const R al = v2::bk_alpha<R>();
        const int stage = v2::bk_stage(g, absakk, colmax, al);
        int kp = 3, kstep = 1;
        if (stage == v2::kBkNeedRow) v2::bk_choice(g, absakk, colmax, rowmax, absimax, al, 3, 8, kp, kstep);
        int* o = out + 3 * c;
        if (it.get_local_linear_id() == 31) o[0] = stage, o[1] = kp, o[2] = kstep;
    });
    const auto got = this->dev.down(out, hout.size());
    for (int c = 0; c < nc; ++c) {
        const auto [akk, cmax, rmax, aimax] = cases[c];
        int stage = v2::kBkNeedRow, kp = 3, kstep = 1;
        if (std::max(akk, cmax) == R(0) || std::isnan(akk)) stage = v2::kBkZero;
        else if (akk >= alpha * cmax) stage = v2::kBkKeep;
        else if (akk >= alpha * cmax * (cmax / rmax)) kp = 3;
        else if (aimax >= alpha * rmax) kp = 8;
        else kp = 8, kstep = 2;
        EXPECT_EQ(got[3 * c], stage) << "case " << c;
        EXPECT_EQ(got[3 * c + 1], kp) << "case " << c;
        EXPECT_EQ(got[3 * c + 2], kstep) << "case " << c;
    }
}

// The 2 x 2 D solve three ways: LdlD2 (?SYTRS), ldl_dsolve_row per row (must agree with LdlD2 bit
// for bit: the fused tiers mix the two), BkBlock2 ([b0 b1] D^-1, the factor's form), against
// Cramer in long double; and the 1 x 1 row.
TYPED_TEST(DeviceBlasV2SytrfTest, LdlBlockSolves) {
    using T = TypeParam;
    using R = real_of<T>;
    constexpr int per = 5;  // dkk, dk1k1, off, b0, b1
    auto h = batched<T>(per, 51, [](T* p) {
        p[0] *= R(0.2), p[1] *= R(0.2), p[2] = T(R(-2)) + p[2] * R(0.5);
    });
    std::vector<T> ho(std::size_t(kBatch) * 7, nanv<T>());
    T* in = this->dev.up(h);
    T* out = this->dev.up(ho);
    this->dev.template run<32>(kBatch, 0, [=](sycl::nd_item<1> it, unsigned char*) {
        if (it.get_local_linear_id() != 0) return;
        const auto b = static_cast<std::ptrdiff_t>(it.get_group_linear_id());
        const T* p = in + b * per;
        T* o = out + b * 7;
        const T dkk = p[0], dk1 = p[1], off = p[2], b0 = p[3], b1 = p[4];
        v2::ldl_d2(off, dkk, dk1).solve(b0, b1, o[0], o[1]);
        o[2] = v2::ldl_dsolve_row(1, b0, b1, dkk, dk1, off);
        o[3] = v2::ldl_dsolve_row(2, b1, b0, dk1, dkk, off);
        const auto blk = v2::bk_block2(dkk, off, dk1);
        o[4] = blk.first(b0, b1);
        o[5] = blk.second(b0, b1);
        o[6] = v2::ldl_dsolve_row(0, b0, b0, dkk, dkk, dkk);
    });
    const auto got = this->dev.down(out, ho.size());
    EXPECT_EQ(twins_differ(got, 7), 0);
    double worst = 0;
    int disagree = 0;
    for (int b = 0; b < kPeriod; ++b) {
        const T* p = h.data() + b * per;
        const T* o = got.data() + b * 7;
        const LD<T> a = wide(p[0]), d = wide(p[1]), e = wide(p[2]), x0 = wide(p[3]), x1 = wide(p[4]);
        const LD<T> det = a * d - e * e;
        const LD<T> r0 = (d * x0 - e * x1) / det, r1 = (a * x1 - e * x0) / det;
        const long double s0 = (std::abs(d) * std::abs(x0) + std::abs(e) * std::abs(x1)) / std::abs(det);
        const long double s1 = (std::abs(a) * std::abs(x1) + std::abs(e) * std::abs(x0)) / std::abs(det);
        worst = std::max({worst, rel(o[0], r0, s0), rel(o[1], r1, s1), rel(o[4], r0, s0), rel(o[5], r1, s1),
                          rel(o[6], x0 / a, std::abs(x0 / a))});
        disagree += !same_bits(o[0], o[2]) + !same_bits(o[1], o[3]);
    }
    EXPECT_EQ(disagree, 0) << "ldl_dsolve_row and LdlD2 must round identically";
    EXPECT_VERIFY(T, Check::blas, 8, worst);
}

// ---- register-distributed sweeps --------------------------------------------------------------

// trsv_frag NoTrans then Trans (unit lower; the diagonal and the strict upper triangle are NaN),
// and ldl_dsolve_frag with a 2 x 2 block on rows 31 / 32 (across the register boundary), n = 50,
// R = 2, one sub-group per item, ld n + 3.
TYPED_TEST(DeviceBlasV2SytrfTest, TrsvFragAndDsolveFrag) {
    using T = TypeParam;
    using R = real_of<T>;
    constexpr int n = 50, ld = n + 3, RR = 2;
    const std::size_t pL = std::size_t(ld) * n;
    auto hL = batched<T>(int(pL), 61, [](T* p) {
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < ld; ++i) {
                T& v = p[std::size_t(i) + std::size_t(j) * ld];
                v = i > j && i < n ? v * R(0.1) : nanv<T>();
            }
    });
    // D: kind by t % 5 = {0: 1 x 1, 1 / 2 and 3 / 4: 2 x 2}; t = 31 opens a block (31 % 5 == 1)
    auto hD = batched<T>(int(pL), 62, [](T* p) {
        for (int t = 0; t < n; ++t) {
            T& v = p[std::size_t(t) * (ld + 1)];
            v = t % 5 == 0 ? T(R(2)) + v * R(0.5) : v * R(0.2);
        }
    });
    std::vector<int> hk(n);
    for (int t = 0; t < n; ++t) hk[t] = t % 5 == 0 ? 0 : (t % 5) % 2 == 1 ? 1 : 2;
    auto he = batched<T>(n, 63, [](T* p) {
        for (int t = 0; t < n; ++t) p[t] = T(R(2.5)) + p[t] * R(0.5);
    });
    const auto hb = batched<T>(n, 64, [](T*) {});
    std::vector<T> ho(std::size_t(kBatch) * 3 * n, nanv<T>());
    T* Lp = this->dev.up(hL);
    T* Dp = this->dev.up(hD);
    int* kp = this->dev.up(hk);
    T* ep = this->dev.up(he);
    T* bp = this->dev.up(hb);
    T* op = this->dev.up(ho);
    this->dev.template run<32>(kBatch, 0, [=](sycl::nd_item<1> it, unsigned char*) {
        const auto b = static_cast<std::ptrdiff_t>(it.get_group_linear_id());
        const auto sg = it.get_sub_group();
        const int lane = static_cast<int>(sg.get_local_linear_id());
        const auto L = v2::mat(Lp + b * std::ptrdiff_t(pL), n, n, ld);
        const auto Dm = v2::mat(Dp + b * std::ptrdiff_t(pL), n, n, ld);
        const T* bb = bp + b * n;
        T* o = op + b * 3 * n;
        T x[RR], y[RR];
        for (int r = 0; r < RR; ++r) {
            const int t = lane + 32 * r;
            x[r] = t < n ? bb[t] : T(0);
            y[r] = x[r];
        }
        v2::trsv_frag<Uplo::Lower, Transpose::NoTrans, Diag::Unit>(sg, lane, L, n, x);
        for (int r = 0; r < RR; ++r)
            if (lane + 32 * r < n) o[lane + 32 * r] = x[r];
        v2::trsv_frag<Uplo::Lower, Transpose::Trans, Diag::Unit>(sg, lane, L, n, x);
        for (int r = 0; r < RR; ++r)
            if (lane + 32 * r < n) o[n + lane + 32 * r] = x[r];
        v2::ldl_dsolve_frag<RR>(sg, lane, Dm, kp, ep + b * n, n, y);
        for (int r = 0; r < RR; ++r)
            if (lane + 32 * r < n) o[2 * n + lane + 32 * r] = y[r];
    });
    const auto got = this->dev.down(op, ho.size());
    EXPECT_EQ(twins_differ(got, 3 * std::size_t(n)), 0) << "items differ from their representative";
    double worst = 0;
    for (int b = 0; b < kPeriod; ++b) {
        const T* l = hL.data() + b * pL;
        const T* dd = hD.data() + b * pL;
        const T* e = he.data() + b * n;
        const T* o = got.data() + b * 3 * n;
        std::vector<LD<T>> y(n), z(n);
        std::vector<long double> sy(n), sz(n);
        for (int t = 0; t < n; ++t) {
            y[t] = wide(hb[b * n + t]), sy[t] = std::abs(y[t]);
            for (int k = 0; k < t; ++k) {
                y[t] -= wide(l[t + std::size_t(k) * ld]) * y[k];
                sy[t] += std::abs(wide(l[t + std::size_t(k) * ld])) * sy[k];
            }
        }
        for (int t = n - 1; t >= 0; --t) {
            z[t] = y[t], sz[t] = sy[t];
            for (int i = t + 1; i < n; ++i) {
                z[t] -= wide(l[i + std::size_t(t) * ld]) * z[i];
                sz[t] += std::abs(wide(l[i + std::size_t(t) * ld])) * sz[i];
            }
        }
        for (int t = 0; t < n; ++t) worst = std::max({worst, rel(o[t], y[t], sy[t]), rel(o[n + t], z[t], sz[t])});
        for (int t = 0; t < n;) {
            const LD<T> x0 = wide(hb[b * n + t]);
            if (hk[t] == 0) {
                const LD<T> d0 = wide(dd[std::size_t(t) * (ld + 1)]);
                worst = std::max(worst, rel(o[2 * n + t], x0 / d0, std::abs(x0 / d0)));
                t += 1;
                continue;
            }
            const LD<T> x1 = wide(hb[b * n + t + 1]), a = wide(dd[std::size_t(t) * (ld + 1)]),
                        d = wide(dd[std::size_t(t + 1) * (ld + 1)]), off = wide(e[t]);
            const LD<T> det = a * d - off * off;
            const long double s0 = (std::abs(d) * std::abs(x0) + std::abs(off) * std::abs(x1)) / std::abs(det);
            const long double s1 = (std::abs(a) * std::abs(x1) + std::abs(off) * std::abs(x0)) / std::abs(det);
            worst = std::max({worst, rel(o[2 * n + t], (d * x0 - off * x1) / det, s0), rel(o[2 * n + t + 1], (a * x1 - off * x0) / det, s1)});
            t += 2;
        }
    }
    EXPECT_VERIFY(T, Check::blas, n, worst);
}

// ---- ?SYCONV on one column ----------------------------------------------------------------------

// Logical ipiv {3, -5, -5, 5, 5} on 5 x 5, one column per item of a width-32 RtGroup. Expected by
// hand: column 0 gets rows (2 4) then (3 4): [a0 a1 a4 a2 a3]; column 1 (the 2 x 2 block at 1)
// moves L(2, 1) to e[1], zeroes it, then (3 4); column 2 gets (3 4); columns 3 and 4 unchanged.
TYPED_TEST(DeviceBlasV2SytrfTest, SyconvColumnByHand) {
    using T = TypeParam;
    constexpr int n = 5, ld = 7;
    std::vector<T> h(std::size_t(ld) * n);
    for (std::size_t e = 0; e < h.size(); ++e) h[e] = val<T>(700 + e);
    std::vector<T> he(n, nanv<T>());
    const std::vector<int> ps = {3, -5, -5, 5, 5};
    T* L = this->dev.up(h);
    T* E = this->dev.up(he);
    const int* P = this->dev.up(ps);
    this->dev.template run<32>(1, 0, [=](sycl::nd_item<1> it, unsigned char*) {
        const v2::RtGroup g{it, static_cast<int>(it.get_local_linear_id()), 32};
        const auto M = v2::mat(L, n, n, ld);
        v2::for_range(g, {0, n}, [&](int j) { v2::syconv_col<T>(M, P, n, j, E); });
    });
    const auto got = this->dev.down(L, h.size());
    const auto ge = this->dev.down(E, he.size());
    auto H = [&](int i, int j) { return h[std::size_t(i) + std::size_t(j) * ld]; };
    std::vector<T> want = h;
    auto W = [&](int i, int j) -> T& { return want[std::size_t(i) + std::size_t(j) * ld]; };
    W(2, 0) = H(4, 0), W(3, 0) = H(2, 0), W(4, 0) = H(3, 0);
    W(2, 1) = T(0), W(3, 1) = H(4, 1), W(4, 1) = H(3, 1);
    W(3, 2) = H(4, 2), W(4, 2) = H(3, 2);
    int bad = 0;
    for (std::size_t e = 0; e < h.size(); ++e) bad += !same_bits(got[e], want[e]);
    EXPECT_EQ(bad, 0);
    EXPECT_TRUE(same_bits(ge[1], H(2, 1))) << "the 2 x 2 off-diagonal moves to e[1]";
    for (int j : {0, 2, 3, 4}) EXPECT_TRUE(std::isnan(std::real(ge[j]))) << "e[" << j << "] is written only for a block start";
}

}  // namespace
