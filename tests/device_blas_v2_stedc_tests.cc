// SPDX-License-Identifier: MIT
// stedc-private v2 helpers (src/ops/syev/stedc/stedc_v2.hh: uniform, fold) and the core ops that
// replaced its copies (prod_scaled, sqrt_scaled, the work-group sync<scope::Group>(ex, Mem::Global)
// stedc uses as its barrier) against host references. The shared ops the stedc kernels use
// (RtGroup, nrm2<alg::Blue>, rot, allreduce, for_each_owned_2d, ...) are covered by
// device_blas_v2_tests. Executors: RtGroup at a non-power-of-two width (96) and at the 1024
// ceiling, WorkGroup<128>, Partition<8> (lockstep), SgpPartition<8, Masked>, Thread.
// Guards (docs/developer/agent-guide.md section 8): batch 1024 at period 3 or 4 with every item
// bit-identical to its representative, folds over sub-views with an offset (tail, then head),
// an all-negative max fold (the identity), the owner mapping fold relies on (no barrier after
// an owned write), scaled products leaving the floating range where the naive product must
// fail, uniform on a bool and on a lane-dependent value, the work-group barrier ordering local
// and global memory, output slots pre-poisoned.
#include <gtest/gtest.h>
#include <tests/test_utils.hh>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <batchlas/device.hh>
#include <batchlas/util/sycl-local-accessor-helpers.hh>
#include <batchlas/device/sg_partition.hh>
#include <src/ops/syev/stedc/stedc_v2.hh>
#include <src/queue.hh>
#include <batchlas/device/detail/kernel_attrs.hh>

using namespace batchlas;
namespace v2 = batchlas::device;
namespace sv = batchlas::stedc_v2;
using batchlas::verify::Check;

namespace {

constexpr int kBatch = 1024;

std::uint64_t mix(std::uint64_t x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}
template <class R> R uval(std::uint64_t i) { return static_cast<R>(static_cast<double>(mix(i) >> 11) * 0x1.0p-53 * 2.0 - 1.0); }
template <class T> bool same_bits(const T& a, const T& b) { return std::memcmp(&a, &b, sizeof(T)) == 0; }

// |got - ref| / scale, the blas relative error that EXPECT_VERIFY bounds.
template <class T> double rel(T got, long double ref, long double scale) {
    const long double d = std::fabs(static_cast<long double>(got) - ref);
    if (std::isnan(d)) return std::numeric_limits<double>::infinity();
    return static_cast<double>(d / std::max<long double>(scale, std::numeric_limits<T>::min()));
}

struct Dev {
    std::optional<Queue> ctx = test_utils::sycl_device_visible()
                                   ? std::optional<Queue>(std::in_place, Device::default_device())
                                   : std::nullopt;
    sycl::queue& q() { return sycl_queue(*ctx); }
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
    // Run-time local range: RtGroup launches are not templated on it.
    template <class F> void run(int wg, int groups, std::size_t local_bytes, F f) {
        q().submit([&](sycl::handler& h) {
             sycl::local_accessor<unsigned char, 1> lm(sycl::range<1>(std::max<std::size_t>(local_bytes, 16)), h);
             h.parallel_for(sycl::nd_range<1>(std::size_t(groups) * wg, std::size_t(wg)),
                            [=](sycl::nd_item<1> it) BATCHLAS_REQD_SG_SIZE(32) {
                                f(it, batchlas::util::get_raw_ptr(lm));
                            });
         }).wait_and_throw();
    }
    ~Dev() {
        for (void* p : ptrs) sycl::free(p, q());
    }
    std::vector<void*> ptrs;
};

// Item b is served by `lanes` consecutive work-items, `per_group` items per work-group of `wg`.
struct Rt96 {
    static constexpr const char* name = "RtGroup(96)";
    static constexpr int wg = 96, per_group = 1, lanes = 96;
    static auto make(sycl::nd_item<1> it) { return v2::RtGroup<>{it}; }
};
struct Rt1024 {
    static constexpr const char* name = "RtGroup(1024)";
    static constexpr int wg = 1024, per_group = 1, lanes = 1024;
    static auto make(sycl::nd_item<1> it) { return v2::RtGroup<>{it}; }
};
struct WG128 {
    static constexpr const char* name = "WorkGroup<128>";
    static constexpr int wg = 128, per_group = 1, lanes = 128;
    static auto make(sycl::nd_item<1> it) { return v2::WorkGroup<128>{it}; }
};
struct Part8 {
    static constexpr const char* name = "Partition<8>";
    static constexpr int wg = 32, per_group = 4, lanes = 8;
    static auto make(sycl::nd_item<1> it) { return v2::Partition<8>{it.get_sub_group()}; }
};
struct SgpPart8 {
    static constexpr const char* name = "SgpPartition<8, Masked>";
    static constexpr int wg = 32, per_group = 4, lanes = 8;
    static auto make(sycl::nd_item<1> it) { return v2::SgpPartition<8, true>{it.get_sub_group()}; }
};
struct Thread1 {
    static constexpr const char* name = "Thread";
    static constexpr int wg = 32, per_group = 32, lanes = 1;
    static auto make(sycl::nd_item<1>) { return v2::Thread{}; }
};

template <class Spec, class Body> void launch(Dev& dev, int items, std::size_t local_bytes, Body body) {
    dev.run(Spec::wg, items / Spec::per_group, local_bytes, [=](sycl::nd_item<1> it, unsigned char* lm) {
        const auto ex = Spec::make(it);
        const int b = static_cast<int>(it.get_group_linear_id()) * Spec::per_group +
                      static_cast<int>(it.get_local_linear_id()) / Spec::lanes;
        body(ex, b, lm);
    });
}

template <class F> void for_each_group_spec(F f) {
    f(Rt96{});
    f(Rt1024{});
    f(WG128{});
}
template <class F> void for_each_spec(F f) {
    for_each_group_spec(f);
    f(Part8{});
    f(SgpPart8{});
    f(Thread1{});
}

// Empty when the device can run these launches, else why not.
std::string skip_reason(Dev& dev) {
    if (!dev.ctx) return "no SYCL device visible";
    if (!dev.ctx->device().supports_sub_group_size(32)) return "v2 tests launch sub-group size 32";
    return {};
}

template <class T> class DeviceBlasV2StedcTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (const auto why = skip_reason(dev); !why.empty()) GTEST_SKIP() << why;
        if (test_utils::lacks_fp64<T>(dev.q().get_device())) GTEST_SKIP() << "device has no fp64";
    }
    Dev dev;
};
using RealTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(DeviceBlasV2StedcTest, RealTypes);

// ---- prod_scaled / sqrt_scaled ----------------------------------------------------------------

// Factors per case: 0 in range with one negative factor; 1 overflows R (the square root does
// not); 2 underflows R in float (0.6^1024); 3 holds an exact zero.
TYPED_TEST(DeviceBlasV2StedcTest, ScaledProductLeavesTheFloatingRange) {
    using T = TypeParam;
    constexpr int n = 1024, ncase = 4, nslot = 4;
    const int ebig = std::is_same_v<T, float> ? 3 : 20;  // 64 factors of ~2^(ebig + 0.5) leave range
    std::vector<T> hf(std::size_t(ncase) * n);
    std::vector<int> hn(kBatch);
    for (int cs = 0; cs < ncase; ++cs)
        for (int i = 0; i < n; ++i) {
            const T u = uval<T>(std::uint64_t(cs) * 104729u + i);
            T f = T(1) + T(0.1) * u;
            if (cs == 0 && i == 17) f = -f;
            if (cs == 1) f = std::ldexp(T(1.5) + T(0.4) * u, ebig);
            if (cs == 2) f = T(0.6);
            if (cs == 3 && i == 50) f = T(0);
            hf[std::size_t(cs) * n + i] = f;
        }
    for (int b = 0; b < kBatch; ++b) hn[b] = b % ncase == 0 ? 200 : (b % ncase == 1 ? 64 : (b % ncase == 2 ? n : 100));
    for_each_spec([&](auto spec) {
        using Spec = decltype(spec);
        Dev& dev = this->dev;
        const T* f = dev.up(hf);
        const int* nn = dev.up(hn);
        T* out = dev.up(std::vector<T>(std::size_t(kBatch) * nslot, T(-7)));
        int* expo = dev.up(std::vector<int>(kBatch, -7777));
        launch<Spec>(dev, kBatch, 0, [=](const auto& ex, int b, unsigned char*) {
            const T* fb = f + std::ptrdiff_t(b % ncase) * n;
            const auto q = v2::prod_scaled(ex, nn[b], [&](int i) { return fb[i]; });
            T acc = T(1);
            v2::for_each_owned(ex, nn[b], [&](int i) { acc *= fb[i]; });
            const T naive = v2::allreduce(ex, acc, sycl::multiplies<T>());
            if (ex.leader()) {
                T* o = out + std::ptrdiff_t(b) * nslot;
                o[0] = q.mant, o[1] = v2::sqrt_scaled(q), o[2] = naive, o[3] = T(v2::detail::size_of(ex));
                expo[b] = q.expo;
            }
        });
        const auto got = dev.down(out, std::size_t(kBatch) * nslot);
        const auto ge = dev.down(expo, kBatch);
        int bitdiff = 0;
        for (int b = ncase; b < kBatch; ++b) {
            bitdiff += ge[b] != ge[b % ncase];
            for (int k = 0; k < nslot; ++k) bitdiff += !same_bits(got[std::size_t(b) * nslot + k], got[std::size_t(b % ncase) * nslot + k]);
        }
        EXPECT_EQ(bitdiff, 0) << Spec::name << ": items differ from their representative";
        for (int b = 0; b < ncase; ++b) {
            const T* o = got.data() + std::size_t(b) * nslot;
            long double lg = 0;  // log2 |prod|
            int neg = 0, zero = 0;
            for (int i = 0; i < hn[b]; ++i) {
                const T v = hf[std::size_t(b) * n + i];
                zero += v == T(0), neg += v < T(0);
                if (v != T(0)) lg += std::log2(std::fabs(static_cast<long double>(v)));
            }
            EXPECT_EQ(o[3], T(Spec::lanes)) << Spec::name << " size_of";
            if (zero) {
                EXPECT_EQ(o[0], T(0)) << Spec::name << " zero factor";
                EXPECT_EQ(o[1], T(0)) << Spec::name << " zero factor, sqrt";
                continue;
            }
            EXPECT_TRUE(std::fabs(o[0]) >= T(0.5) && std::fabs(o[0]) < T(1)) << Spec::name << " case " << b << " mantissa " << o[0];
            EXPECT_EQ(o[0] < T(0), neg % 2 == 1) << Spec::name << " case " << b << " sign";
            const long double got_lg = std::log2(std::fabs(static_cast<long double>(o[0]))) + ge[b];
            // log2 error ~ relative error / ln 2: verify it as a relative error of the product.
            EXPECT_VERIFY(T, Check::blas, hn[b], static_cast<double>(std::fabs(got_lg - lg) * std::log(2.0L)))
                << Spec::name << " case " << b << " log2 " << got_lg << " vs " << lg;
            const long double half = std::exp2(lg / 2);
            EXPECT_VERIFY(T, Check::blas, hn[b], rel(o[1], half, half)) << Spec::name << " case " << b << " sqrt_scaled";
            const bool in_range = std::fabs(lg) < std::numeric_limits<T>::max_exponent - 2 && lg > std::numeric_limits<T>::min_exponent + 2;
            if (!in_range) {
                // The case must leave range: the naive product overflows or underflows. Gradual
                // underflow: 0.6 x the least subnormal rounds back to it, so a one-item product
                // stops at 2^-149, not 0.
                EXPECT_TRUE(std::fabs(o[2]) < std::numeric_limits<T>::min() || std::isinf(o[2]))
                    << Spec::name << " case " << b << " naive " << o[2];
            } else if (o[3] <= T(32)) {
                // One round of at most 32 items: the scaled product IS the naive one, bit for bit.
                EXPECT_TRUE(same_bits(std::ldexp(o[0], ge[b]), o[2])) << Spec::name << " case " << b;
            }
        }
    });
}

// ---- fold ------------------------------------------------------------------------------------

TYPED_TEST(DeviceBlasV2StedcTest, FoldMatchesHost) {
    using T = TypeParam;
    constexpr int n = 333, nslot = 6, kPeriod = 3, lo = 5, len = n - 11;
    // All negative: a max fold that starts from 0 instead of the identity returns 0.
    std::vector<T> hx(std::size_t(kBatch) * n);
    for (int b = 0; b < kBatch; ++b)
        for (int i = 0; i < n; ++i) hx[std::size_t(b) * n + i] = -T(1) - std::fabs(uval<T>(std::uint64_t(b % kPeriod) * 7919u + i));
    for_each_group_spec([&](auto spec) {
        using Spec = decltype(spec);
        Dev& dev = this->dev;
        const T* x = dev.up(hx);
        T* out = dev.up(std::vector<T>(std::size_t(kBatch) * nslot, T(-7)));
        launch<Spec>(dev, kBatch, sizeof(T) * n, [=](const auto& ex, int b, unsigned char* lm) {
            const auto xv = v2::vec(x + std::ptrdiff_t(b) * n, n);
            // Staged through local memory with no barrier: fold reads each element on its owner.
            const auto s = v2::vec<v2::Mem::Local>(reinterpret_cast<T*>(lm), n);
            v2::map(ex, s, [](T v) { return -v; }, xv);
            const T staged = sv::fold(ex, s, sycl::plus<T>());
            const T mx = sv::fold(ex, xv, sycl::maximum<T>());
            const T sm = sv::fold(ex, xv, sycl::plus<T>());
            // plus: an empty sycl::joint_reduce gives Op's identity (-inf for maximum), T{} on acpp.
            const T em = sv::fold(ex, xv.head(0), sycl::plus<T>());
            const T win = sv::fold(ex, xv.tail(lo).head(len), sycl::plus<T>());  // off = lo, not at the start
            const T wmx = sv::fold(ex, xv.tail(lo).head(len), sycl::minimum<T>());
            if (ex.leader()) {
                T* o = out + std::ptrdiff_t(b) * nslot;
                o[0] = mx, o[1] = sm, o[2] = em, o[3] = win, o[4] = wmx, o[5] = staged;
            }
        });
        const auto got = dev.down(out, std::size_t(kBatch) * nslot);
        int bitdiff = 0;
        for (int b = kPeriod; b < kBatch; ++b)
            for (int k = 0; k < nslot; ++k) bitdiff += !same_bits(got[std::size_t(b) * nslot + k], got[std::size_t(b % kPeriod) * nslot + k]);
        EXPECT_EQ(bitdiff, 0) << Spec::name << ": items differ from their representative";
        double worst = 0;
        for (int b = 0; b < kPeriod; ++b) {
            const T* xb = hx.data() + std::size_t(b) * n;
            const T* o = got.data() + std::size_t(b) * nslot;
            long double sm = 0, sa = 0, sw = 0, swa = 0;
            for (int i = 0; i < n; ++i) sm += xb[i], sa += std::fabs(xb[i]);
            for (int i = lo; i < lo + len; ++i) sw += xb[i], swa += std::fabs(xb[i]);
            EXPECT_EQ(o[0], *std::max_element(xb, xb + n)) << Spec::name << " fold max";
            EXPECT_EQ(o[2], T(0)) << Spec::name << " empty fold is T{}";
            EXPECT_EQ(o[4], *std::min_element(xb + lo, xb + lo + len)) << Spec::name << " fold min over the window";
            worst = std::max({worst, rel(o[1], sm, sa), rel(o[3], sw, swa), rel(o[5], -sm, sa)});
        }
        EXPECT_VERIFY(T, Check::blas, n, worst) << Spec::name << " fold sum";
    });
}

// ---- uniform, the stedc barrier ----------------------------------------------------------------

TEST(DeviceBlasV2StedcHelpers, UniformAndGroupSync) {
    Dev dev;
    if (const auto why = skip_reason(dev); !why.empty()) GTEST_SKIP() << why;
    constexpr int items = kBatch;
    for_each_spec([&](auto spec) {
        using Spec = decltype(spec);
        using Ex = decltype(Spec::make(std::declval<sycl::nd_item<1>>()));
        constexpr int L = Spec::lanes;
        float* uni = dev.up(std::vector<float>(std::size_t(items) * L, -7.f));
        int* flag = dev.up(std::vector<int>(std::size_t(items) * L, -7));
        int* syn = dev.up(std::vector<int>(std::size_t(items) * L, -7));
        int* gsyn = dev.up(std::vector<int>(std::size_t(items) * L, -7));
        int* gbuf = dev.up(std::vector<int>(std::size_t(items) * L, -7));
        launch<Spec>(dev, items, sizeof(int) * Spec::wg, [=](const auto& ex, int b, unsigned char* lm) {
            const int lid = ex.lid();
            uni[std::ptrdiff_t(b) * L + lid] = sv::uniform(ex, static_cast<float>(lid * 7 + 3 + b));
            flag[std::ptrdiff_t(b) * L + lid] = sv::uniform(ex, lid == 0) ? 1 : 0;
            if constexpr (sv::is_group_v<Ex>) {
                int* buf = reinterpret_cast<int*>(lm);
                const int src = (lid + 33) % L;
                buf[lid] = lid + 1 + b;
                gbuf[std::ptrdiff_t(b) * L + lid] = 2 * lid + b;
                v2::sync<v2::scope::Group>(ex, v2::Mem::Global);
                syn[std::ptrdiff_t(b) * L + lid] = buf[src];
                gsyn[std::ptrdiff_t(b) * L + lid] = gbuf[std::ptrdiff_t(b) * L + src];
            }
        });
        const auto gu = dev.down(uni, std::size_t(items) * L);
        const auto gf = dev.down(flag, std::size_t(items) * L);
        const auto gs = dev.down(syn, std::size_t(items) * L), gg = dev.down(gsyn, std::size_t(items) * L);
        int bad_u = 0, bad_f = 0, bad_s = 0, bad_g = 0;
        for (int b = 0; b < items; ++b)
            for (int l = 0; l < L; ++l) {
                const std::size_t k = std::size_t(b) * L + l;
                bad_u += gu[k] != static_cast<float>(3 + b);
                bad_f += gf[k] != 1;
                if constexpr (sv::is_group_v<Ex>) {
                    bad_s += gs[k] != ((l + 33) % L) + 1 + b;
                    bad_g += gg[k] != 2 * ((l + 33) % L) + b;
                } else {
                    bad_s += gs[k] != -7, bad_g += gg[k] != -7;  // no barrier branch on a team
                }
            }
        EXPECT_EQ(bad_u, 0) << Spec::name << " uniform is the leader's value";
        EXPECT_EQ(bad_f, 0) << Spec::name << " uniform(bool) is the leader's flag";
        EXPECT_EQ(bad_s, 0) << Spec::name << " sync<Group> orders the group's local memory";
        EXPECT_EQ(bad_g, 0) << Spec::name << " sync<Group> orders the group's global memory";
    });
}

// Which executors stedc barriers and folds on as a work-group.
static_assert(sv::is_group_v<v2::RtGroup<>> && sv::is_group_v<v2::WorkGroup<64>>);
static_assert(!sv::is_group_v<v2::RtGroup<v2::Scope::Sub>> && !sv::is_group_v<v2::Partition<8>> &&
              !sv::is_group_v<v2::SgpPartition<8, true>> && !sv::is_group_v<v2::Thread>);

}  // namespace
