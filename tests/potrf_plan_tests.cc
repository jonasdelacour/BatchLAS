// potrf launch plans (src/extensions/potrf_launch_plan.hh) and the cost they feed
// (src/util/launch_plan.hh). Host-only: no queue, no kernel.
//
// The geometry oracle is potrf_plan_golden.inc, recorded from the launchers BEFORE they were
// moved onto the plan functions, at every order where a launch code changes (both sides).
// evidence: docs/perf/potrf.md#launch-plans
#include <gtest/gtest.h>

#include "../src/extensions/potrf_launch_plan.hh"

#include <complex>
#include <cstdint>
#include <string>

using namespace batchlas;
using launch_plan::CostConstants;
using launch_plan::DeviceFacts;
using launch_plan::LaunchPlan;

namespace {

struct GoldenRow {
    char type;
    int tier;   // 0 tiny, 1 cta, 2 lpanel, 3 blocked
    int n;
    int batch;
    int min_blocks;
    int nb_hint;
    unsigned code;
};

const GoldenRow kGolden[] = {
#include "potrf_plan_golden.inc"
};

struct CeilRow {
    char type;
    std::size_t budget;
    int min_blocks;
    int cta, lpanel, lpanel16;
};

const CeilRow kCeil[] = {
#include "potrf_plan_golden_ceil.inc"
};

// The device the golden table was recorded on; a 4090 reports the same two values.
constexpr DeviceFacts kRec{101376, 1024, 188, 1536, 32};
constexpr int kTrsmLeaf = 32;   // sycl_trsm::trsm_cta_max_n<T>() for every type

// The *_debug_* encodings, so the comparison is code-for-code with the recording.
template <typename T>
unsigned encode(int tier, int n, int batch, int mb, int nb_hint) {
    switch (tier) {
        case 0: {
            const auto g = potrf_plan::tiny_geometry<T>(n, batch, kRec.max_wg_size);
            if (!g.fits) return 0u;
            const int S = g.wg_size / tiny_native::kTinySubGroupSize;
            return (static_cast<unsigned>(S) << 16) | static_cast<unsigned>(g.per_wg / S);
        }
        case 1: {
            const auto g = potrf_plan::cta_geometry<T>(n, batch, kRec, mb);
            return g.fits ? (static_cast<unsigned>(g.L) << 16) | static_cast<unsigned>(g.G) : 0u;
        }
        case 2: {
            const auto g = potrf_plan::lpanel_geometry<T>(n, batch, kRec, mb, nb_hint);
            return g.fits ? (static_cast<unsigned>(g.nb) << 16) |
                                (static_cast<unsigned>(g.L) << 4) | static_cast<unsigned>(g.G)
                          : 0u;
        }
        default: {
            const auto p = potrf_plan::blocked_params<T>(n, kRec.local_mem_bytes, kTrsmLeaf);
            return (static_cast<unsigned>(p.W) << 16) | static_cast<unsigned>(p.nb);
        }
    }
}

unsigned encode_any(const GoldenRow& r) {
    switch (r.type) {
        case 's': return encode<float>(r.tier, r.n, r.batch, r.min_blocks, r.nb_hint);
        case 'd': return encode<double>(r.tier, r.n, r.batch, r.min_blocks, r.nb_hint);
        case 'c': return encode<std::complex<float>>(r.tier, r.n, r.batch, r.min_blocks, r.nb_hint);
        default: return encode<std::complex<double>>(r.tier, r.n, r.batch, r.min_blocks, r.nb_hint);
    }
}

TEST(PotrfPlanGeometry, MatchesThePreRefactorLaunchersAtEveryEdge) {
    const char* tier_name[] = {"tiny", "cta", "lpanel", "blocked"};
    int per_tier[4] = {0, 0, 0, 0};
    for (const auto& r : kGolden) {
        ++per_tier[r.tier];
        EXPECT_EQ(encode_any(r), r.code)
            << r.type << " " << tier_name[r.tier] << " n=" << r.n << " batch=" << r.batch
            << " min_blocks=" << r.min_blocks << " nb_hint=" << r.nb_hint;
    }
    // Anti-vacuity: every tier has edges in the table.
    for (int t = 0; t < 4; ++t) EXPECT_GT(per_tier[t], 8) << tier_name[t];
}

template <typename T>
void expect_ceil(const CeilRow& r) {
    EXPECT_EQ(potrf_plan::cta_max_n<T>(r.budget, r.min_blocks), r.cta)
        << r.type << " budget=" << r.budget << " mb=" << r.min_blocks;
    EXPECT_EQ(potrf_plan::lpanel_max_n<T>(r.budget, 1024, r.min_blocks, 0), r.lpanel)
        << r.type << " budget=" << r.budget << " mb=" << r.min_blocks;
    EXPECT_EQ(potrf_plan::lpanel_max_n<T>(r.budget, 1024, r.min_blocks, 16), r.lpanel16)
        << r.type << " budget=" << r.budget << " mb=" << r.min_blocks << " nb16";
}

TEST(PotrfPlanGeometry, CeilingsMatchThePreRefactorQueries) {
    for (const auto& r : kCeil) {
        switch (r.type) {
            case 's': expect_ceil<float>(r); break;
            case 'd': expect_ceil<double>(r); break;
            case 'c': expect_ceil<std::complex<float>>(r); break;
            default: expect_ceil<std::complex<double>>(r); break;
        }
    }
}

// A plan that does not fit says so, rather than reporting a geometry.
TEST(PotrfPlan, UnfitShapesHaveNoPlan) {
    EXPECT_FALSE(potrf_plan::tiny_plan<std::complex<double>>(17, 64, kRec).fits);
    EXPECT_TRUE(potrf_plan::tiny_plan<std::complex<double>>(16, 64, kRec).fits);
    EXPECT_FALSE(potrf_plan::cta_plan<float>(78, 64, kRec).fits);
    EXPECT_TRUE(potrf_plan::cta_plan<float>(77, 64, kRec).fits);
    EXPECT_FALSE(potrf_plan::lpanel_plan<float>(745, 64, kRec).fits);
    EXPECT_TRUE(potrf_plan::lpanel_plan<float>(744, 64, kRec).fits);
}

// Hand-counted: float n=300 is nb=128, W=128 -> panels (128, m2=172), (128, m2=44), (44, 0).
// Updates: panel 0 has w=128 (mr=44) and w=44 (mr=0); panel 1 has w=44 (mr=0).
// Launches: 2 fills + [2 leaf/fixup + 1 solve + 3 + 2] + [2 + 1 + 2] + [2] = 17.
TEST(PotrfPlan, BlockedScheduleIsHandCounted) {
    const LaunchPlan p = potrf_plan::blocked_plan<float>(300, 1000, kRec, kTrsmLeaf);
    ASSERT_TRUE(p.fits);
    EXPECT_EQ(p.launches, 17);
    EXPECT_EQ(p.wave_launches, 3);
    EXPECT_EQ(p.serial_steps, 300);
    // The leaf at nb=128 > the occupancy-scaled CTA ceiling would not fit; above order 256 the
    // leaf is clamped against the RESIDENT ceiling (155), so nb stays 128 and one matrix per group.
    EXPECT_EQ(p.groups, 1000);
    const double flops = 128.0 * 128 * 128 / 3 + 128.0 * 128 / 2 + 128.0 / 6     // leaf 0
                       + 172.0 * 128 * 128                                        // solve 0
                       + 2.0 * 128 * 128 * 128 + 2.0 * 44 * 128 * 128             // w=128
                       + 2.0 * 44 * 44 * 128                                      // w=44
                       + 128.0 * 128 * 128 / 3 + 128.0 * 128 / 2 + 128.0 / 6     // leaf 1
                       + 44.0 * 128 * 128 + 2.0 * 44 * 44 * 128                   // solve 1, w=44
                       + 44.0 * 44 * 44 / 3 + 44.0 * 44 / 2 + 44.0 / 6;          // leaf 2
    EXPECT_DOUBLE_EQ(p.flops, flops * 1000);
}

TEST(PotrfPlan, ComplexCountsFourRealFlopsPerMultiplyAdd) {
    const auto r = potrf_plan::cta_plan<float>(40, 100, kRec);
    const auto c = potrf_plan::cta_plan<std::complex<float>>(40, 100, kRec);
    EXPECT_DOUBLE_EQ(c.useful_flops, 4 * r.useful_flops);
    EXPECT_DOUBLE_EQ(r.useful_flops, (40.0 * 40 * 40 / 3 + 40.0 * 40 / 2 + 40.0 / 6) * 100);
    EXPECT_DOUBLE_EQ(c.bytes, (2 * (40.0 * 41 / 2) * 8 + 4) * 100);
}

// The tiny tier runs the whole register bucket: n=9 executes the N=16 recurrence.
TEST(PotrfPlan, TinyExecutesTheBucket) {
    const auto p = potrf_plan::tiny_plan<float>(9, 64, kRec);
    EXPECT_EQ(p.serial_steps, 16);
    EXPECT_DOUBLE_EQ(p.flops, potrf_plan::useful_flops<float>(16, 64));
    EXPECT_DOUBLE_EQ(p.useful_flops, potrf_plan::useful_flops<float>(9, 64));
    // No register data: threads (1536/64 = 24) and the group cap (32) bind -> 24.
    EXPECT_EQ(p.regs_per_item, 0);
    EXPECT_EQ(p.resident_groups_per_cu, 24);
    // The bucket's register count binds: 104 regs -> 4 warps per sub-partition, 16 per CU,
    // 2 warps per group -> 8. The N=8 and N=32 slots must not be read for n=9.
    potrf_plan::KernelRegs r;
    r.tiny[0] = 1;
    r.tiny[1] = 104;
    r.tiny[2] = 1;
    const auto q = potrf_plan::tiny_plan<float>(9, 64, kRec, r);
    EXPECT_EQ(q.regs_per_item, 104);
    EXPECT_EQ(q.resident_groups_per_cu, 8);
}

// CTA reads the register count of the scope its L selects; the leaf of Blocked reads CTA's.
TEST(PotrfPlan, CtaRegistersFollowTheScope) {
    potrf_plan::KernelRegs r;
    r.cta_sg = 71;
    r.cta_wg = 56;
    EXPECT_EQ(potrf_plan::cta_plan<float>(10, 64, kRec, r).regs_per_item, 71);    // L == 32
    EXPECT_EQ(potrf_plan::cta_plan<float>(77, 64, kRec, r).regs_per_item, 56);    // L > 32
    EXPECT_EQ(potrf_plan::blocked_plan<float>(300, 64, kRec, kTrsmLeaf, r).regs_per_item, 56);
}

// The CTA body walks whole NB = 8 panels; LPanel's chain is 3n + (sizeof(T)/2)*P(P-1)/2.
TEST(PotrfPlan, SerialChainsAreHandCounted) {
    EXPECT_EQ(potrf_plan::cta_plan<float>(13, 64, kRec).serial_steps, 16);
    EXPECT_EQ(potrf_plan::cta_plan<float>(16, 64, kRec).serial_steps, 16);
    const auto lp = potrf_plan::lpanel_plan<float>(20, 64, kRec);    // P = 3
    EXPECT_EQ(lp.serial_steps, 3 * 20 + 2 * 3);
    EXPECT_DOUBLE_EQ(lp.slot_chain, 66.0);
    const auto lz = potrf_plan::lpanel_plan<std::complex<double>>(20, 64, kRec);
    EXPECT_DOUBLE_EQ(lz.slot_chain, 3 * 20 + 8 * 3);
    EXPECT_EQ(potrf_plan::cta_plan<float>(20, 64, kRec).slot_chain, 0.0);
}

// Blocked's sub-op work is batch-wide, and the W x W scratch fill is counted when n > nb.
TEST(PotrfPlan, BlockedWorkIsBatchWideWithTheScratchFill) {
    const auto p = potrf_plan::blocked_plan<float>(300, 1000, kRec, kTrsmLeaf);
    EXPECT_TRUE(p.batch_wide_work);
    const auto t = launch_plan::cost_terms(p, kRec);
    EXPECT_DOUBLE_EQ(t.flop, p.flops);
    EXPECT_DOUBLE_EQ(t.byte, p.bytes);
    // n = 20 is one 20 x 20 leaf (nb = n); n = 300 has nb = 128 < n, so the fill is counted.
    const auto small = potrf_plan::blocked_plan<float>(20, 1000, kRec, kTrsmLeaf);
    EXPECT_DOUBLE_EQ(small.bytes, (2 * (20.0 * 21 / 2) * 4 + 8.0) * 1000);
    EXPECT_GE(p.bytes, 128.0 * 128 * 4 * 1000);
}

TEST(LaunchPlanOccupancy, TightestKnownCapBinds) {
    DeviceFacts d{101376, 1024, 0, 0, 0};
    EXPECT_EQ(launch_plan::resident_groups_per_cu(d, 0, 64, 0), 1);       // nothing known
    EXPECT_EQ(launch_plan::resident_groups_per_cu(d, 20000, 64, 0), 4);   // 97280 / 20000
    d.max_threads_per_cu = 1536;
    EXPECT_EQ(launch_plan::resident_groups_per_cu(d, 1000, 512, 0), 3);   // threads bind
    d.max_groups_per_cu = 2;
    EXPECT_EQ(launch_plan::resident_groups_per_cu(d, 1000, 512, 0), 2);   // group cap binds
}

TEST(LaunchPlanCost, WavesAndTermsAreHandComputed) {
    LaunchPlan p;
    p.launches = 3;
    p.wave_launches = 1;
    p.groups = 1000;
    p.resident_groups_per_cu = 4;
    p.flops = 2.0e9;
    p.bytes = 4.0e6;
    p.serial_steps = 50;
    const DeviceFacts d{101376, 1024, 188, 0, 0};
    p.wg_size = 64;
    p.slot_chain = 10;
    const auto t = launch_plan::cost_terms(p, d);
    // Throughput: ceil(1000/188) = 6 groups per CU, one group's share 2e6 flops / 4000 bytes.
    // Latency: 1000 groups over 752 resident slots = 2 waves.
    EXPECT_DOUBLE_EQ(t.launch, 3);
    EXPECT_DOUBLE_EQ(t.flop, 6 * 2.0e6);
    EXPECT_DOUBLE_EQ(t.byte, 6 * 4000.0);
    EXPECT_DOUBLE_EQ(t.step, 2 * 50.0);
    EXPECT_DOUBLE_EQ(t.slot, 6 * 2 * 10.0);
    const CostConstants c{1e-6, 1e-9, 1e-6, 1e-7, 1e-6};
    // The byte side (2.4e-2) binds over the flop (1.2e-2) and slot (1.2e-4) sides.
    EXPECT_DOUBLE_EQ(launch_plan::cost(p, d, c), 3e-6 + 2.4e-2 + 1e-5);
    static_assert(launch_plan::combine({1, 2, 3, 4}, {1, 1, 1, 1}) == 1 + 3 + 4);
}

// evaluation/routing/fit.py re-states combine() in one line; this is the same arithmetic on
// the values tests/test_fit.py pins, so the two cannot drift silently.
TEST(LaunchPlanCost, CombineMatchesTheFitterPin) {
    const launch_plan::CostTerms t{2, 1e6, 3e5, 40, 5e5};
    const CostConstants c{5e-6, 2e-12, 1e-11, 3e-8, 1e-11};
    EXPECT_DOUBLE_EQ(launch_plan::combine(t, c), 1e-5 + 5e-6 + 1.2e-6);
}

TEST(LaunchPlanCost, PredictionsOutsideTheMeasuredBoxAreFlagged) {
    const launch_plan::SupportRegion s{8, 256, 512, 32768, 40};
    const LaunchPlan p = potrf_plan::cta_plan<float>(64, 1024, kRec);
    const CostConstants c{1e-6, 1e-11, 1e-9, 1e-8};
    EXPECT_FALSE(launch_plan::predict(p, kRec, c, s, 64, 1024).extrapolated);
    EXPECT_TRUE(launch_plan::predict(p, kRec, c, s, 7, 1024).extrapolated);
    EXPECT_TRUE(launch_plan::predict(p, kRec, c, s, 257, 1024).extrapolated);
    EXPECT_TRUE(launch_plan::predict(p, kRec, c, s, 64, 511).extrapolated);
    EXPECT_TRUE(launch_plan::predict(p, kRec, c, s, 64, 32769).extrapolated);
    EXPECT_TRUE(launch_plan::extrapolated(launch_plan::SupportRegion{}, 64, 1024));  // fallback
    EXPECT_DOUBLE_EQ(launch_plan::predict(p, kRec, c, s, 64, 1024).seconds,
                     launch_plan::cost(p, kRec, c));
}

}  // namespace
