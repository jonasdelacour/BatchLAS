// SPDX-License-Identifier: MIT
// device_core_tier_tests: tier.hh and workspace.hh (design C.4, C.5) on a mock routine with three
// tiers (Reg<16>, Cta<64>, Blocked<32>).
// Guards (agent-guide section 8): with_tier at every tier boundary (n = 16, 17, 64, 65); each
// filter on its own (local memory at a 32 KB Metal-like limit straddled, no fp64 for double, a
// missing sub-group size, a work-group ceiling); the throw names the routine and n; upto<N>
// resolves to the smallest admitting tier at every boundary; Plan's carve offsets equal its
// size and are aligned; a device kernel writes and reads back through carve_local / carve_global
// for 1024 matrices, each slice disjoint.
// Red set (each break was made once, 2026-11-04, and this suite failed): the fp64 filter skipped;
// global_slice striding by local_bytes (first missed while local and global sizes were equal: the
// plan now has a global extra and the caller's buffer layout is checked byte for byte).
#include <tests/device_core_common.hh>

#include <batchlas/device/tier.hh>
#include <batchlas/device/workspace.hh>

#include <string>

using namespace device_core_test;

namespace mockdev {
struct mock {
    static constexpr const char* name = "mockroutine";
};
}  // namespace mockdev

namespace batchlas::device {
template <class T> struct tiers<mockdev::mock, T> {
    using type = tier::list<tier::Reg<16>, tier::Cta<64>, tier::Blocked<32>>;
};
template <class T> struct tier_info<mockdev::mock, tier::Reg<16>, T> {
    static constexpr int wg_size = 32, sg_size = 32;
    static constexpr bool admits(int n) { return n >= 0; }
    static constexpr WorkspaceSize buffer_size(int) { return {}; }
};
template <class T> struct tier_info<mockdev::mock, tier::Cta<64>, T> {
    static constexpr int wg_size = 128, sg_size = 32;
    static constexpr bool admits(int n) { return n >= 0; }
    static constexpr WorkspaceSize buffer_size(int n) {
        Plan p;
        p.take<T>(WsSpace::Local, std::size_t(n) * n);
        return p.size();
    }
};
template <class T> struct tier_info<mockdev::mock, tier::Blocked<32>, T> {
    static constexpr int wg_size = 256, sg_size = 16;
    static constexpr bool admits(int n) { return n >= 1; }
    static constexpr WorkspaceSize buffer_size(int n) {
        Plan p;
        p.take<T>(WsSpace::Local, 32 * 32);
        p.take<T>(WsSpace::Global, std::size_t(n) * 32);
        return p.size();
    }
};
}  // namespace batchlas::device

namespace {

using device::tier::Blocked;
using device::tier::Cta;
using device::tier::Reg;

static_assert(std::is_same_v<device::resolve_tier_t<mockdev::mock, float, device::tier::upto<16>>, Reg<16>>);
static_assert(std::is_same_v<device::resolve_tier_t<mockdev::mock, float, device::tier::upto<17>>, Cta<64>>);
static_assert(std::is_same_v<device::resolve_tier_t<mockdev::mock, float, device::tier::upto<64>>, Cta<64>>);
static_assert(std::is_same_v<device::resolve_tier_t<mockdev::mock, float, device::tier::upto<65>>, Blocked<32>>);
static_assert(std::is_same_v<device::resolve_tier_t<mockdev::mock, float, Cta<64>>, Cta<64>>);

device::DeviceLimits big() { return {std::size_t(100) << 10, {16, 32}, 1024, true}; }

template <class T> std::string pick(const device::DeviceLimits& lim, int n) {
    std::string got = "none";
    try {
        device::with_tier<mockdev::mock, T>(lim, n, [&]<class Tier>(Tier) {
            if constexpr (std::is_same_v<Tier, Reg<16>>) got = "reg";
            else if constexpr (std::is_same_v<Tier, Cta<64>>) got = "cta";
            else got = "blocked";
        });
    } catch (const batchlas::invalid_argument& e) {
        got = std::string("throw:") + e.what();
    }
    return got;
}

TEST(DeviceCoreTierHost, WithTierVisitsTheFirstAdmittingTierAtEveryBoundary) {
    EXPECT_EQ(pick<float>(big(), 0), "reg");
    EXPECT_EQ(pick<float>(big(), 16), "reg");
    EXPECT_EQ(pick<float>(big(), 17), "cta");
    EXPECT_EQ(pick<float>(big(), 64), "cta");
    EXPECT_EQ(pick<float>(big(), 65), "blocked");
    EXPECT_EQ(pick<float>(big(), 5000), "blocked");
}

TEST(DeviceCoreTierHost, EachDeviceFilterRemovesItsTier) {
    // Cta<64> double at n = 64 needs 32 KB of local memory: fits 32 KB exactly, not 32 KB - 1.
    device::DeviceLimits metal = big();
    metal.local_mem_bytes = 32 << 10;
    EXPECT_EQ(pick<double>(metal, 64), "cta");
    metal.local_mem_bytes = (32 << 10) - 1;
    EXPECT_EQ(pick<double>(metal, 64), "blocked");
    // No fp64: no double tier at all, float unaffected.
    device::DeviceLimits nofp64 = big();
    nofp64.fp64 = false;
    EXPECT_EQ(pick<float>(nofp64, 10), "reg");
    EXPECT_EQ(pick<double>(nofp64, 10).rfind("throw:", 0), 0u);
    // Sub-group size 16 missing: Blocked (sg 16) is filtered.
    device::DeviceLimits sg32 = big();
    sg32.sub_group_sizes = {32};
    EXPECT_EQ(pick<float>(sg32, 100).rfind("throw:", 0), 0u);
    // A 64-item work-group ceiling leaves only Reg (wg 32).
    device::DeviceLimits small = big();
    small.max_work_group_size = 64;
    EXPECT_EQ(pick<float>(small, 16), "reg");
    EXPECT_EQ(pick<float>(small, 17).rfind("throw:", 0), 0u);
}

TEST(DeviceCoreTierHost, NoTierThrowsNamingTheRoutineAndN) {
    device::DeviceLimits none = big();
    none.sub_group_sizes = {8};
    EXPECT_EQ(pick<float>(none, 77), "throw:mockroutine: no device tier admits n=77");
}

TEST(DeviceCoreTierHost, PlanOffsetsAreAlignedAndSumToItsSize) {
    device::Plan p(16);
    const auto a = p.take<char>(device::WsSpace::Local, 3);
    const auto b = p.take<double>(device::WsSpace::Local, 5);
    const auto c = p.take<std::complex<double>>(device::WsSpace::Global, 7);
    const auto e = p.take<float>(device::WsSpace::Global, 1);
    EXPECT_EQ(a.offset, 0u);
    EXPECT_EQ(b.offset, 16u);
    EXPECT_EQ(c.offset, 0u);
    EXPECT_EQ(e.offset, 7u * 16u);
    const auto sz = p.size();
    EXPECT_EQ(sz.local_bytes, 16u + 5u * 8u + 8u);  // 56 rounded to 16
    EXPECT_EQ(sz.global_bytes, 7u * 16u + 16u);
    EXPECT_EQ(sz.align, 16u);
    constexpr auto cta = device::tier_info<mockdev::mock, Cta<64>, float>::buffer_size(64);
    static_assert(cta.local_bytes == 64 * 64 * 4 && cta.global_bytes == 0);
}

// Each of 1024 work-groups carves its local and global slots from one plan and round-trips data
// through both; the global slices of neighbours must not overlap.
TYPED_TEST_SUITE(DeviceCoreTest, AllTypes);
TYPED_TEST(DeviceCoreTest, CarveRoundTripsThroughLocalAndGlobalSlices) {
    using T = TypeParam;
    auto& d = this->dev;
    constexpr int items = 1024, n = 37;
    device::Plan plan;
    const auto ls = plan.take<T>(device::WsSpace::Local, n);
    plan.take<T>(device::WsSpace::Global, 3 * n);  // a global extra ahead: global_bytes != local_bytes
    const auto gs = plan.take<T>(device::WsSpace::Global, n);
    const auto sz = plan.size();
    ASSERT_NE(sz.global_bytes, sz.local_bytes);
    const auto x_h = vals<T>(std::size_t(items) * n, 81);
    T* x = d.up(x_h);
    std::byte* gbuf = d.template alloc<std::byte>(sz.global_bytes * items, std::byte{0});
    T* out = d.template alloc<T>(x_h.size(), T(0));
    d.template run<32>(items, sz.local_bytes, [=](sycl::nd_item<1> it, unsigned char* lm) {
        const int b = static_cast<int>(it.get_group_linear_id());
        const int l = static_cast<int>(it.get_local_linear_id());
        const device::Workspace w{sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::yes>(
                                  reinterpret_cast<std::byte*>(lm)),
                              device::global_slice(gbuf, sz, b)};
        auto L = device::carve_local(w, ls);
        T* G = device::carve_global(w, gs);
        for (int i = l; i < n; i += 32) L[i] = x[b * n + i];
        it.barrier(sycl::access::fence_space::local_space);
        for (int i = l; i < n; i += 32) G[i] = L[n - 1 - i];
        sycl::group_barrier(it.get_group(), sycl::memory_scope::work_group);
        for (int i = l; i < n; i += 32) out[b * n + i] = G[i];
    });
    const auto got = d.down(out, x_h.size());
    for (int b = 0; b < items; ++b)
        for (int i = 0; i < n; ++i) ASSERT_TRUE(same_bits(got[b * n + i], x_h[b * n + (n - 1 - i)])) << b << " " << i;
    // Matrix b's slot sits at b * global_bytes + offset in the caller's buffer (the layout a host
    // launcher allocates), not merely somewhere disjoint.
    const auto raw = d.down(gbuf, sz.global_bytes * items);
    for (int b = 0; b < items; b += 101)
        for (int i = 0; i < n; ++i) {
            T v;
            std::memcpy(&v, raw.data() + b * sz.global_bytes + gs.offset + i * sizeof(T), sizeof(T));
            ASSERT_TRUE(same_bits(v, x_h[b * n + (n - 1 - i)])) << "slice of item " << b << " at " << i;
        }
}

}  // namespace
