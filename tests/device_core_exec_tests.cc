// SPDX-License-Identifier: MIT
// device_core_exec_tests: exec.hh additions (design A.2) against host references.
// Guards (agent-guide section 8): complex data with nonzero imaginary parts through every lane
// move (bit-identical: a move is exact); executors whose team is a strict slice of a sub-group
// (Teams P = 8, Partition P = 8) so a move that crosses its block is visible; per-lane (not
// uniform) shuffle sources; a WG = 1024 sync<scope::Group> exchange where every item reads an
// item of another sub-group; the index-only executors at their guards (n < size, n == size,
// offsets). Red set (each break made once, 2026-11-04, and this suite failed): Teams::bcast
// addressing the whole sub-group (block_select<SG>) instead of its P-lane block.
#include <tests/device_core_common.hh>
#include <batchlas/device/exec.hh>
#include <batchlas/device/level1.hh>

using namespace device_core_test;

namespace {

template <class T> class DeviceCoreExecTest : public DeviceCoreTest<T> {};
TYPED_TEST_SUITE(DeviceCoreExecTest, AllTypes);

// Each executor's bcast / uniform / shuffle / shift on one WG of 64 items. Host model: team =
// aligned block of `team` lanes inside the sub-group of 32; bcast is over the executor (`scope`
// items), shuffle and shift inside the team.
template <class T, class MakeEx> void check_moves(Dev& d, int team, int scope, MakeEx make) {
    constexpr int WG = 64;
    const auto x_h = vals<T>(WG, 7 + team);
    T* x = d.up(x_h);
    T* bc = d.template alloc<T>(WG, T(0));
    T* un = d.template alloc<T>(WG, T(0));
    T* sh = d.template alloc<T>(WG, T(0));
    T* sf = d.template alloc<T>(WG, T(0));
    const int src = 5 % scope;
    d.template run<WG>(1, 0, [=](sycl::nd_item<1> it, unsigned char*) {
        const auto ex = make(it);
        const int g = static_cast<int>(it.get_local_linear_id());
        const int lane = g % team;
        bc[g] = ex.bcast(x[g], src);
        un[g] = ex.uniform(x[g]);
        sh[g] = ex.shuffle(x[g], (lane * 5 + 3) % team);
        sf[g] = ex.shift(x[g], 1);
    });
    const auto b = d.down(bc, WG), u = d.down(un, WG), s = d.down(sh, WG), f = d.down(sf, WG);
    for (int g = 0; g < WG; ++g) {
        const int base_scope = g - g % scope, base_team = g - g % team, lane = g % team;
        EXPECT_TRUE(same_bits(b[g], x_h[base_scope + src])) << "bcast item " << g << " team " << team;
        EXPECT_TRUE(same_bits(u[g], x_h[base_scope])) << "uniform item " << g;
        EXPECT_TRUE(same_bits(s[g], x_h[base_team + (lane * 5 + 3) % team])) << "shuffle item " << g;
        if (lane + 1 < team) EXPECT_TRUE(same_bits(f[g], x_h[g + 1])) << "shift item " << g;
    }
}

TYPED_TEST(DeviceCoreExecTest, LaneMovesStayInsideTheirTeam) {
    using T = TypeParam;
    auto& d = this->dev;
    check_moves<T>(d, 32, 64, [](sycl::nd_item<1> it) { return batchlas::device::WorkGroup<64, 32>{it}; });
    check_moves<T>(d, 8, 8, [](sycl::nd_item<1> it) { return batchlas::device::Teams<64, 8>{it}; });
    check_moves<T>(d, 32, 32, [](sycl::nd_item<1> it) { return batchlas::device::SubGroup<32>{it.get_sub_group()}; });
    check_moves<T>(d, 8, 8, [](sycl::nd_item<1> it) { return batchlas::device::Partition<8>{it.get_sub_group()}; });
    check_moves<T>(d, 32, 32, [](sycl::nd_item<1> it) { return batchlas::device::RtGroup<batchlas::device::Scope::Sub>{it}; });
    check_moves<T>(d, 32, 32, [](sycl::nd_item<1> it) { return batchlas::device::Fit{batchlas::device::SubGroup<32>{it.get_sub_group()}}; });
}

TYPED_TEST(DeviceCoreExecTest, ThreadMovesAreTheIdentity) {
    using T = TypeParam;
    const T v = val<T>(3);
    const batchlas::device::Thread th;
    EXPECT_TRUE(same_bits(th.bcast(v, 7), v));
    EXPECT_TRUE(same_bits(th.uniform(v), v));
    EXPECT_TRUE(same_bits(th.shuffle(v, 3), v));
    EXPECT_TRUE(same_bits(th.shift(v, 1), v));
}

// sync<scope::Group> on a 1024-item work-group orders a local-memory exchange across sub-groups;
// sync<scope::Sub> one within a sub-group; sync<scope::Part> one within a Partition.
TYPED_TEST(DeviceCoreExecTest, SyncScopesOrderTheirExchange) {
    using T = TypeParam;
    auto& d = this->dev;
    constexpr int WG = 1024;
    const auto x_h = vals<T>(WG, 41);
    T* x = d.up(x_h);
    T* g_out = d.template alloc<T>(WG, T(0));
    T* s_out = d.template alloc<T>(WG, T(0));
    T* p_out = d.template alloc<T>(WG, T(0));
    d.template run<WG>(1, 3 * WG * sizeof(T), [=](sycl::nd_item<1> it, unsigned char* lm) {
        T* slm = reinterpret_cast<T*>(lm);
        const batchlas::device::WorkGroup<WG, 32> wg{it};
        const batchlas::device::Partition<8> part{it.get_sub_group()};
        const int g = wg.lid();
        slm[g] = x[g];
        slm[WG + g] = x[g];
        slm[2 * WG + g] = x[g];
        batchlas::device::sync<batchlas::device::scope::Group>(wg);
        g_out[g] = slm[(g + 517) % WG];
        batchlas::device::sync<batchlas::device::scope::Sub>(wg);
        const int sb = g - g % 32;
        s_out[g] = slm[WG + sb + (g % 32 + 13) % 32];
        batchlas::device::sync<batchlas::device::scope::Part>(part);
        const int pb = g - g % 8;
        p_out[g] = slm[2 * WG + pb + (g % 8 + 3) % 8];
    });
    const auto go = d.down(g_out, WG), so = d.down(s_out, WG), po = d.down(p_out, WG);
    for (int g = 0; g < WG; ++g) {
        EXPECT_TRUE(same_bits(go[g], x_h[(g + 517) % WG])) << "group " << g;
        EXPECT_TRUE(same_bits(so[g], x_h[g - g % 32 + (g % 32 + 13) % 32])) << "sub " << g;
        EXPECT_TRUE(same_bits(po[g], x_h[g - g % 8 + (g % 8 + 3) % 8])) << "part " << g;
    }
}

// Index-only executors through for_each_owned / for_each_owned_2d (host: plain index logic).
TEST(DeviceCoreExecIndex, SlotFitGridAndElemVisitTheirIndices) {
    auto visits = [](const auto& ex, int n) {
        std::vector<int> v;
        batchlas::device::for_each_owned(ex, n, [&](int i) { v.push_back(i); });
        return v;
    };
    EXPECT_EQ(visits(batchlas::device::Slot<8>{3}, 30), (std::vector<int>{3, 11, 19, 27}));
    EXPECT_EQ(visits(batchlas::device::Slot<>{2, 5}, 13), (std::vector<int>{2, 7, 12}));
    EXPECT_EQ(visits(batchlas::device::Fit{batchlas::device::Slot<8>{5}}, 6), (std::vector<int>{5}));
    EXPECT_EQ(visits(batchlas::device::Fit{batchlas::device::Slot<8>{6}}, 6), (std::vector<int>{}));
    EXPECT_EQ(visits(batchlas::device::Grid1D{9}, 9), (std::vector<int>{}));
    EXPECT_EQ(visits(batchlas::device::Grid1D<true>{9}, 9), (std::vector<int>{9}));  // Exact: no test
    std::vector<int> r;
    batchlas::device::for_each_owned(batchlas::device::Fit{batchlas::device::Slot<8>{2}}, batchlas::device::Range{4, 7}, [&](int i) { r.push_back(i); });
    EXPECT_EQ(r, (std::vector<int>{6}));
    auto visits2 = [](const auto& ex, int m, int n) {
        std::vector<std::pair<int, int>> v;
        batchlas::device::for_each_owned_2d(ex, m, n, [&](int i, int j) { v.emplace_back(i, j); });
        return v;
    };
    using P = std::vector<std::pair<int, int>>;
    EXPECT_EQ(visits2(batchlas::device::ElemAt{3, 4}, 4, 5), (P{{3, 4}}));
    EXPECT_EQ(visits2(batchlas::device::ElemAt{3, 5}, 4, 5), (P{}));
    EXPECT_EQ(visits2(batchlas::device::ElemAt<true>{3, 5}, 4, 5), (P{{3, 5}}));
    EXPECT_EQ(visits2(batchlas::device::ElemAt{5, 6}.sub(2, 4), 4, 3), (P{{3, 2}}));
    EXPECT_EQ(visits2(batchlas::device::ElemAt{5, 3}.sub(2, 4), 4, 3), (P{}));  // left of the block
    EXPECT_EQ(visits2(batchlas::device::ElemAt{6, 6}.sub(2, 4), 4, 3), (P{}));  // below the block
    const batchlas::device::bcast_policy::Slm<4, int*> slm{nullptr, 2};
    EXPECT_EQ(slm.buffer(0, 3), 6);
    EXPECT_EQ(slm.buffer(5, 3), 18);  // odd column: the second buffer half
}

}  // namespace
