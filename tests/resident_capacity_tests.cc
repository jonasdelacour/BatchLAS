// The occupancy rule shared by every local-memory-resident kernel: the capacity walk in
// src/util/resident_capacity.hh, the per-type ceilings it produces for potrf / getrf /
// geqrf, the G-packing helper, and the one launch property that packing can break.
// evidence: docs/perf/potrf.md#the-occupancy-rule
#include <gtest/gtest.h>

#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>
#include <batchlas/util/sycl-vector.hh>

#include "test_utils.hh"

#include "../src/util/resident_capacity.hh"
#include "../src/extensions/potrf_native.hh"
#include "../src/extensions/getrf_native.hh"
#include "../src/extensions/geqrf_native.hh"

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <vector>

using namespace batchlas;

namespace {

// 1. THE WALK -- on a SYNTHETIC non-monotone bytes(n), not on a device.

// The library's real pad, restated. Band and pad must stay byte-identical to potrf_cta.cc,
// getrf_cta.cc and geqrf_cta.cc, or the table below stops modelling the shipped shape.
constexpr std::size_t kHoleLo = 47104, kHoleHi = 49664, kHolePadTo = 49920;

constexpr std::size_t hole_padded(std::size_t bytes) {
    return (bytes > kHoleLo && bytes <= kHoleHi) ? kHolePadTo : bytes;
}

// A deliberately non-monotone footprint: 128 B per unit of n, then the library's pad, so
// that bytes(389) < bytes(388). No budget this box reports reaches the band where a
// continue-walk and a break-walk disagree, which is why this table is synthetic.
// evidence: docs/perf/potrf.md#the-contiguity-rule-and-the-synthetic-table
constexpr std::size_t synth_bytes(int n) {
    return hole_padded(static_cast<std::size_t>(n) * 128u);
}

TEST(ResidentCapacity, SyntheticFootprintIsActuallyNonMonotone) {
    // ANTI-VACUITY: were this table monotone, break and continue would agree below.
    ASSERT_GT(synth_bytes(388), synth_bytes(389))
        << "the synthetic footprint is monotone; the contiguity rule is untestable on it";
    EXPECT_EQ(synth_bytes(368), 47104u);
    EXPECT_EQ(synth_bytes(369), kHolePadTo);
    EXPECT_EQ(synth_bytes(388), kHolePadTo);
    EXPECT_EQ(synth_bytes(389), 49792u);
}

// THE ARMED PROPERTY. supports() spells capacity as the contiguous `order <= cta_max_n`, so
// the ceiling must be the largest n at which EVERY order up to n fits -- not the largest n
// that happens to fit. A walk that skipped a miss advertises orders that cannot launch.
TEST(ResidentCapacity, CeilingIsContiguousAcrossTheNonMonotoneHole) {
    // A budget that admits n <= 368 and n = 389 but refuses everything between.
    const std::size_t tight = 49850;

    const int cap = resident::resident_max_n(synth_bytes, tight, 1, 4096);
    EXPECT_EQ(cap, 368) << "the walk did not stop at the first miss: it advertised a "
                           "contiguous range containing orders that cannot launch";

    // The two anti-vacuity halves, stated as facts about the table rather than the walk.
    ASSERT_GT(synth_bytes(369), tight) << "369 fits after all; nothing is skipped";
    ASSERT_LE(synth_bytes(389), tight) << "389 does not fit either, so a continue-walk "
                                          "would agree with a break-walk here";

    // Contiguity of the answer: every order inside the advertised range really launches.
    for (int n = 1; n <= cap; ++n) {
        EXPECT_LE(synth_bytes(n), tight)
            << "order " << n << " is inside the advertised range but does not fit";
    }
    EXPECT_GT(synth_bytes(cap + 1), tight);

    // Both edges of the band.
    EXPECT_EQ(resident::resident_max_n(synth_bytes, 49920, 1, 4096), 390);
    EXPECT_EQ(resident::resident_max_n(synth_bytes, 47104, 1, 4096), 368);
}

TEST(ResidentCapacity, OccupancyTargetDividesTheBudget) {
    EXPECT_EQ(resident::occupancy_budget(97280, 1), 97280u);
    EXPECT_EQ(resident::occupancy_budget(97280, 4), 24320u);
    EXPECT_EQ(resident::occupancy_budget(97280, 0), 97280u);   // 0 is not a divisor
    EXPECT_EQ(resident::occupancy_budget(97280, -3), 97280u);

    // And the walk really consumes it.
    EXPECT_EQ(resident::resident_max_n([](int n) { return std::size_t(n) * 1024u; },
                                       97280, 1),
              95);
    EXPECT_EQ(resident::resident_max_n([](int n) { return std::size_t(n) * 1024u; },
                                       97280, 4),
              23);
}

TEST(ResidentCapacity, DeviceBudgetSubtractsTheRuntimeReserve) {
    EXPECT_EQ(resident::device_slm_budget(101376), 97280u);
    EXPECT_EQ(resident::device_slm_budget(49152), 45056u);
    EXPECT_EQ(resident::device_slm_budget(4096), 0u);      // no room at all
    EXPECT_EQ(resident::device_slm_budget(1024), 0u);      // and no underflow
}

// 2. G-PACKING

TEST(ResidentCapacity, PackingIsBoundedByAllThreeLimits) {
    // Local memory, the work-group width and the max_pack cap each bind in turn. The
    // 8-lane rows are load-bearing: at 32 lanes the cap of 4 and the 128-wide target
    // agree, so a helper that ignored max_pack would answer the same on every other row.
    // evidence: docs/perf/potrf.md#the-g-packing-helper-and-its-three-limits
    EXPECT_EQ(resident::pack_matrices_per_wg(4096, 32, 24320, 1024), 4);
    EXPECT_EQ(resident::pack_matrices_per_wg(8192, 32, 24320, 1024), 2);
    EXPECT_EQ(resident::pack_matrices_per_wg(20000, 32, 24320, 1024), 1);
    EXPECT_EQ(resident::pack_matrices_per_wg(4096, 32, 24320, 64), 2);
    EXPECT_EQ(resident::pack_matrices_per_wg(4096, 32, 24320, 32), 1);
    EXPECT_EQ(resident::pack_matrices_per_wg(1024, 256, 24320, 1024), 1);
    EXPECT_EQ(resident::pack_matrices_per_wg(64, 8, 24320, 1024), 4);
    EXPECT_EQ(resident::pack_matrices_per_wg(64, 8, 24320, 1024, 128, 16), 16);
    EXPECT_EQ(resident::pack_matrices_per_wg(64, 8, 24320, 1024, 32, 16), 4);
    EXPECT_EQ(resident::pack_matrices_per_wg(64, 8, 24320, 64, 128, 16), 8);
    EXPECT_EQ(resident::pack_matrices_per_wg(6144, 8, 24320, 1024, 128, 16), 2);

    // THE RESULT AGAINST THE THREE LIMITS, over a grid. g >= 1 and "g is a power of two"
    // are structural -- the walk starts at 1 and only doubles -- so they are not asserted.
    for (int lanes : {8, 32, 64}) {
        for (std::size_t bytes : {std::size_t(1), std::size_t(999), std::size_t(4096),
                                  std::size_t(1u << 20)}) {
            for (int max_wg : {32, 64, 1024}) {
                const int g = resident::pack_matrices_per_wg(bytes, lanes, 24320, max_wg);
                ASSERT_GE(g, 1) << "lanes=" << lanes << " bytes=" << bytes;
                EXPECT_LE(g, 4) << "G exceeds max_pack at lanes=" << lanes
                                << " bytes=" << bytes << " max_wg=" << max_wg;
                if (g > 1) {   // G == 1 is the honest answer even when one matrix
                               // does not fit, so the bounds bite only above it.
                    EXPECT_LE(g * lanes, std::min(max_wg, 128))
                        << "G lanes exceed the work-group width at lanes=" << lanes;
                    EXPECT_LE(static_cast<std::size_t>(g) * bytes, std::size_t(24320))
                        << "G matrices exceed the local-memory slice at bytes=" << bytes;
                }
            }
        }
    }
    // Degenerate inputs answer 1 rather than dividing by zero.
    EXPECT_EQ(resident::pack_matrices_per_wg(0, 32, 24320, 1024), 1);
    EXPECT_EQ(resident::pack_matrices_per_wg(4096, 0, 24320, 1024), 1);
}

// 3. THE SHIPPED CEILINGS, PER TYPE. Pinned in BOTH scales: the occupancy-scaled figure is
// what supports() advertises, the unscaled one is what a blocked driver's panel can be held
// at. evidence: docs/perf/potrf.md#the-shipped-ceilings-pinned-in-both-scales

template <typename T> struct Ceilings;
template <> struct Ceilings<float> {
    static constexpr int potrf_advertised = 77,  potrf_resident = 155;
    static constexpr int getrf_advertised = 77,  getrf_resident = 155;
    static constexpr int64_t geqrf_advertised = 11776, geqrf_resident = 24320;
};
template <> struct Ceilings<double> {
    static constexpr int potrf_advertised = 54,  potrf_resident = 109;
    static constexpr int getrf_advertised = 54,  getrf_resident = 109;
    static constexpr int64_t geqrf_advertised = 5888, geqrf_resident = 12160;
};
template <> struct Ceilings<std::complex<float>> {
    static constexpr int potrf_advertised = 54,  potrf_resident = 109;
    static constexpr int getrf_advertised = 54,  getrf_resident = 109;
    static constexpr int64_t geqrf_advertised = 5888, geqrf_resident = 12160;
};
template <> struct Ceilings<std::complex<double>> {
    static constexpr int potrf_advertised = 38,  potrf_resident = 77;
    static constexpr int getrf_advertised = 38,  getrf_resident = 77;
    static constexpr int64_t geqrf_advertised = 2944, geqrf_resident = 6080;
};

template <typename T>
class CapacityCeilingTest : public ::testing::Test {};

using CeilingTypes = ::testing::Types<float, double, std::complex<float>,
                                      std::complex<double>>;
TYPED_TEST_SUITE(CapacityCeilingTest, CeilingTypes);

TYPED_TEST(CapacityCeilingTest, ShippedCeilingsAtTheReferenceBudget) {
    using T = TypeParam;
    using C = Ceilings<T>;
    constexpr std::size_t kRef = 97280;   // 101,376 B of local memory less the reserve

    EXPECT_EQ(sycl_potrf::potrf_cta_max_n_for_slm<T>(kRef), C::potrf_advertised);
    EXPECT_EQ(sycl_potrf::potrf_cta_max_n_for_slm<T>(kRef, 1), C::potrf_resident);

    EXPECT_EQ(sycl_getrf::getrf_cta_max_n_for_slm<T>(kRef), C::getrf_advertised);
    EXPECT_EQ(sycl_getrf::getrf_cta_max_n_for_slm<T>(kRef, 1), C::getrf_resident);

    EXPECT_EQ(sycl_geqrf::geqrf_cta_max_elems_for_slm<T>(kRef), C::geqrf_advertised);
    EXPECT_EQ(sycl_geqrf::geqrf_cta_max_elems_for_slm<T>(kRef, 1), C::geqrf_resident);
}

// The advertised capacity must be a strict subset of what a work-group can hold, or the
// occupancy target is not being applied; and the fit predicates must agree with their own
// ceilings, or supports() and the entry-point gate can disagree.
TYPED_TEST(CapacityCeilingTest, AdvertisedIsStrictlyInsideResident) {
    using T = TypeParam;
    constexpr std::size_t kRef = 97280;

    const int potrf_a = sycl_potrf::potrf_cta_max_n_for_slm<T>(kRef);
    const int getrf_a = sycl_getrf::getrf_cta_max_n_for_slm<T>(kRef);
    EXPECT_LT(potrf_a, sycl_potrf::potrf_cta_max_n_for_slm<T>(kRef, 1));
    EXPECT_LT(getrf_a, sycl_getrf::getrf_cta_max_n_for_slm<T>(kRef, 1));
    EXPECT_LT(sycl_geqrf::geqrf_cta_max_elems_for_slm<T>(kRef),
              sycl_geqrf::geqrf_cta_max_elems_for_slm<T>(kRef, 1));

    // getrf: the ceiling and the predicate are one statement.
    EXPECT_TRUE(sycl_getrf::getrf_cta_fits<T>(getrf_a, kRef));
    EXPECT_FALSE(sycl_getrf::getrf_cta_fits<T>(getrf_a + 1, kRef));
    // ... and the residency predicate is strictly wider.
    EXPECT_TRUE(sycl_getrf::getrf_leaf_fits<T>(getrf_a + 1, getrf_a + 1, kRef));

    // The residency ceiling needs its OWN tight pair. Probing one order above the
    // ADVERTISED ceiling passes with the whole occupancy factor as slack.
    const int getrf_r = sycl_getrf::getrf_cta_max_n_for_slm<T>(kRef, 1);
    EXPECT_TRUE(sycl_getrf::getrf_leaf_fits<T>(getrf_r, getrf_r, kRef))
        << "the residency ceiling names an order the residency predicate refuses";
    EXPECT_FALSE(sycl_getrf::getrf_leaf_fits<T>(getrf_r + 1, getrf_r + 1, kRef))
        << "one order past the residency ceiling still fits: the ceiling is not the "
           "predicate's own boundary";

    // geqrf: an AREA, so both a square at the ceiling and a tall panel of the same area.
    const int64_t area = sycl_geqrf::geqrf_cta_max_elems_for_slm<T>(kRef);
    EXPECT_TRUE(sycl_geqrf::geqrf_cta_fits<T>(static_cast<int>(area / 8), 8, kRef));
    EXPECT_FALSE(sycl_geqrf::geqrf_cta_fits<T>(static_cast<int>(area / 8) + 1, 8, kRef));
    EXPECT_TRUE(sycl_geqrf::geqrf_leaf_fits<T>(static_cast<int>(area / 8) + 1, 8, kRef));

    // ... and the same tight pair at geqrf's residency ceiling, same 8-column panel shape.
    const int64_t area_r = sycl_geqrf::geqrf_cta_max_elems_for_slm<T>(kRef, 1);
    EXPECT_TRUE(sycl_geqrf::geqrf_leaf_fits<T>(static_cast<int>(area_r / 8), 8, kRef))
        << "the residency area ceiling names a panel the residency predicate refuses";
    EXPECT_FALSE(sycl_geqrf::geqrf_leaf_fits<T>(static_cast<int>(area_r / 8) + 1, 8, kRef))
        << "a panel one row past the residency area ceiling still fits";
}

// A bigger budget must admit more of everything: the scaling is a division, not a table.
TYPED_TEST(CapacityCeilingTest, CeilingsRiseWithTheBudget) {
    using T = TypeParam;
    EXPECT_LT(sycl_potrf::potrf_cta_max_n_for_slm<T>(97280),
              sycl_potrf::potrf_cta_max_n_for_slm<T>(101376));
    EXPECT_LT(sycl_getrf::getrf_cta_max_n_for_slm<T>(97280),
              sycl_getrf::getrf_cta_max_n_for_slm<T>(101376));
    EXPECT_LT(sycl_geqrf::geqrf_cta_max_elems_for_slm<T>(97280),
              sycl_geqrf::geqrf_cta_max_elems_for_slm<T>(101376));
}

// 4. PACKED VS SOLO, BIT FOR BIT -- getrf_cta's resident leaf

template <typename T, Backend B>
struct CapConfig {
    using ScalarType = T;
    static constexpr Backend BackendVal = B;
};

template <typename T> T mkv(double re, double im);
template <> float mkv<float>(double re, double) { return static_cast<float>(re); }
template <> double mkv<double>(double re, double) { return re; }
template <> std::complex<float> mkv<std::complex<float>>(double re, double im) {
    return {static_cast<float>(re), static_cast<float>(im)};
}
template <> std::complex<double> mkv<std::complex<double>>(double re, double im) {
    return {re, im};
}

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed * 6364136223846793005ULL + 1442695040888963407ULL) {}
    double next() {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return double(int32_t(uint32_t(s >> 32))) / 2147483648.0;
    }
};

template <typename Config>
class PackedLeafTest : public test_utils::BatchLASTest<Config> {
protected:
    using T = typename Config::ScalarType;

    void SetUp() override {
        test_utils::BatchLASTest<Config>::SetUp();
        if (this->HasFatalFailure() || ::testing::Test::IsSkipped()) return;
        if (!this->ctx) GTEST_SKIP() << "no queue";
        if (this->ctx->device().type != DeviceType::GPU)
            GTEST_SKIP() << "the resident leaf is a GPU kernel";
        if (!this->ctx->device().supports_sub_group_size(32))
            GTEST_SKIP() << "device does not offer sub-group size 32";
    }
};

using PackedLeafTypes = typename test_utils::backend_types<CapConfig>::type;
TYPED_TEST_SUITE(PackedLeafTest, PackedLeafTypes);

// A packed launch (G > 1 panels per work-group, sub-group-scoped barriers) must agree BIT FOR
// BIT with the same matrices run one to a work-group. The break this catches is a barrier at
// the wrong scope, or a slot offset letting one sub-group's rank-1 update reach its
// neighbour's tile: both leave a plausible factorisation, so a residual bound cannot see
// them. Each matrix is distinctly scaled, so a cross-matrix write changes a value.
// evidence: docs/perf/lu.md#packed-resident-leaves
TYPED_TEST(PackedLeafTest, PackedLeafMatchesSoloBitForBit) {
    using T = typename TestFixture::T;

    int packed_ns = 0;
    for (int n : {8, 12, 16, 24, 32}) {
        const unsigned geom = sycl_getrf::getrf_cta_debug_launch<T>(*this->ctx, n, n);
        ASSERT_NE(geom, 0u) << "n=" << n << " is not resident, which no device should hit";
        const int G = static_cast<int>(geom & 0xffffu);
        if (G <= 1) continue;      // this type does not pack at this order
        ++packed_ns;

        const int batch = 4 * G + 1;   // deliberately NOT a multiple of G: the last
                                       // work-group is ragged and its idle sub-groups
                                       // must take the early-out without hanging.
        const int ld = n + 3;
        const int stride = ld * n + 5;

        UnifiedVector<T> packed(static_cast<size_t>(stride) * batch, mkv<T>(-9.75e3, 4.5e3));
        Rng rg(4242u + 13u * unsigned(n));
        for (int b = 0; b < batch; ++b) {
            const double scale = 1.0 + b;
            for (int j = 0; j < n; ++j) {
                for (int i = 0; i < n; ++i) {
                    packed[size_t(b) * stride + size_t(j) * ld + i] =
                        mkv<T>(scale * rg.next(), scale * rg.next());
                }
            }
        }
        std::vector<T> a0(packed.begin(), packed.end());

        UnifiedVector<int> piv_p(static_cast<size_t>(n) * batch, -12345);
        UnifiedVector<int32_t> info_p(static_cast<size_t>(batch), 0);
        bool resident = false;
        ASSERT_NO_THROW(sycl_getrf::getrf_panel_factorize<T>(
            *this->ctx, packed.data(), ld, stride, n, n, batch, piv_p.data(), n, 0,
            info_p.data(), &resident));
        this->ctx->wait();
        ASSERT_TRUE(resident) << "n=" << n << " did not take the resident leaf";

        // LIVENESS, before any packed-vs-solo comparison. Both sides are seeded
        // identically -- info 0, ipiv at the -12345 poison, the factor compared against the
        // OTHER run -- so a kernel that wrote nothing would satisfy every equality below.
        for (int b = 0; b < batch; ++b) {
            ASSERT_EQ(info_p[b], 0)
                << "packed item " << b << " at n=" << n << " reported info = "
                << info_p[b] << "; the fixture's matrices are nonsingular";
            for (int k = 0; k < n; ++k) {
                const int piv = piv_p[size_t(b) * n + k];
                ASSERT_GE(piv, k + 1) << "ipiv[" << k << "] = " << piv << " at n=" << n
                                      << " b=" << b << " is below the diagonal row "
                                         "(1-based), or was never written";
                ASSERT_LE(piv, n) << "ipiv[" << k << "] = " << piv << " at n=" << n
                                  << " b=" << b << " names a row outside the matrix";
            }
            int changed = 0;
            for (int j = 0; j < n; ++j) {
                for (int i = 0; i < n; ++i) {
                    const size_t o = size_t(b) * stride + size_t(j) * ld + i;
                    if (packed[o] != a0[o]) ++changed;
                }
            }
            ASSERT_GT(changed, 0)
                << "packed item " << b << " at n=" << n << " is bit-identical to its "
                   "input: the leaf wrote no factor at all";
        }

        // Solo: one matrix per launch, so the same code path runs with G == 1 panels in
        // flight per work-group even though the scope is unchanged.
        for (int b = 0; b < batch; ++b) {
            UnifiedVector<T> solo(static_cast<size_t>(stride), mkv<T>(-9.75e3, 4.5e3));
            std::copy(a0.begin() + size_t(b) * stride,
                      a0.begin() + size_t(b + 1) * stride, solo.begin());
            UnifiedVector<int> piv_s(static_cast<size_t>(n), -12345);
            UnifiedVector<int32_t> info_s(1, 0);
            bool res_s = false;
            ASSERT_NO_THROW(sycl_getrf::getrf_panel_factorize<T>(
                *this->ctx, solo.data(), ld, stride, n, n, 1, piv_s.data(), n, 0,
                info_s.data(), &res_s));
            this->ctx->wait();
            ASSERT_TRUE(res_s);

            EXPECT_EQ(info_p[b], info_s[0]) << "n=" << n << " b=" << b;
            for (int k = 0; k < n; ++k) {
                ASSERT_EQ(piv_p[size_t(b) * n + k], piv_s[k])
                    << "packed and solo pivoted differently at n=" << n << " b=" << b
                    << " k=" << k;
            }
            for (int j = 0; j < n; ++j) {
                for (int i = 0; i < n; ++i) {
                    const T pv = packed[size_t(b) * stride + size_t(j) * ld + i];
                    const T sv = solo[size_t(j) * ld + i];
                    ASSERT_EQ(pv, sv) << "packed vs solo differ at n=" << n << " b=" << b
                                      << " (" << i << "," << j << ")";
                }
            }
        }
    }
    // ANTI-VACUITY: without a G > 1 launch every assertion above compared a solo run
    // against another solo run.
    ASSERT_GT(packed_ns, 0)
        << "no order in the sweep packed more than one matrix per work-group for this "
           "type; this test proved nothing";
}

}  // namespace
