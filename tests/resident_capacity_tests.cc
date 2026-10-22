// SPDX-License-Identifier: MIT
// The occupancy rule shared by every local-memory-resident kernel: the capacity walk in
// src/util/resident_capacity.hh, the per-type ceilings it produces for potrf / getrf /
// geqrf, the G-packing helper, and the one launch property that packing can break.
// evidence: docs/perf/potrf.md#potrf-the-occupancy-rule
#include <gtest/gtest.h>

#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>
#include <batchlas/util/sycl-vector.hh>

#include <tests/test_utils.hh>

#include <batchlas/blas/functions/gemm.hh>
#include <batchlas/sycl_interop.hh>
#include <batchlas/util/env.hh>

#include <src/util/resident_capacity.hh>
#include <src/queue.hh>
#include <src/select/select.hh>
#include <src/sycl/local_mem.hh>
#include <src/ops/potrf/potrf_native.hh>
#include <src/ops/getrf/getrf_native.hh>
#include <src/ops/geqrf/geqrf_native.hh>

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
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

// THE ARMED PROPERTY. can_run spells capacity as the contiguous `order <= cta_max_n`, so
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

// The sub-group CTA kernels' multiplier clamp counts whole steps of 32/P problems. syev_cta_fused
// complex<double> P = 16 holds 6,784 B a problem, 13,568 B a step.
TEST(ResidentCapacity, CtaMultiplierClampCountsWholeSteps) {
    EXPECT_EQ(resident::cta_fit_wg_multiplier(8, 2, 6784, 101376), 7);  // the problem count said 14
    EXPECT_EQ(resident::cta_fit_wg_multiplier(8, 2, 6784, 49152), 3);
    EXPECT_EQ(resident::cta_fit_wg_multiplier(2, 2, 6784, 101376), 2);  // a fitting request is kept
    EXPECT_EQ(resident::cta_fit_wg_multiplier(7, 2, 6784, 7 * 13568), 7);  // exactly full
    EXPECT_EQ(resident::cta_fit_wg_multiplier(8, 2, 6784, 7 * 13568 - 1), 6);
    EXPECT_EQ(resident::cta_fit_wg_multiplier(4, 1, 71752, 49152), 0);  // not one step: callers throw
    EXPECT_EQ(resident::cta_fit_wg_multiplier(4, 1, 0, 0), 4);  // nothing staged
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
// what can_run advertises, the unscaled one is what a blocked driver's panel can be held
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
// ceilings, or can_run and the entry-point gate can disagree.
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

        using MV = MatrixView<T, MatrixFormat::Dense>;
        UnifiedVector<T> packed(static_cast<size_t>(stride) * batch, MV::make(-9.75e3, 4.5e3));
        const Matrix<T> rnd = Matrix<T>::Random(n, n, false, batch, 4242u + 13u * unsigned(n));
        for (int b = 0; b < batch; ++b) {
            const double scale = 1.0 + b;
            for (int j = 0; j < n; ++j) {
                for (int i = 0; i < n; ++i) {
                    const T r = rnd(i, j, b);
                    packed[size_t(b) * stride + size_t(j) * ld + i] =
                        MV::make(scale * double(std::real(r)), scale * double(std::imag(r)));
                }
            }
        }
        std::vector<T> a0(packed.begin(), packed.end());

        UnifiedVector<int> piv_p(static_cast<size_t>(n) * batch, -12345);
        UnifiedVector<int32_t> info_p(static_cast<size_t>(batch), 0);
        bool resident = false;
        ASSERT_NO_THROW((void)sycl_getrf::getrf_panel_factorize<T>(
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
            UnifiedVector<T> solo(static_cast<size_t>(stride), MV::make(-9.75e3, 4.5e3));
            std::copy(a0.begin() + size_t(b) * stride,
                      a0.begin() + size_t(b + 1) * stride, solo.begin());
            UnifiedVector<int> piv_s(static_cast<size_t>(n), -12345);
            UnifiedVector<int32_t> info_s(1, 0);
            bool res_s = false;
            ASSERT_NO_THROW((void)sycl_getrf::getrf_panel_factorize<T>(
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

// 5. THE LAUNCH CEILINGS THE BUDGETS ADVERTISE -- local memory per work-group size, the
// selector's copy of it, and the grid. Rule 9: a launch AT each ceiling, not a re-derivation.

class LocalMemCeilingKernel;
class GroupCountKernel;
class GlobalRangeKernel;

std::optional<Queue> gpu_queue() {
    for (const Device& d : Device::get_devices(DeviceType::GPU)) return Queue(d);
    return std::nullopt;
}

bool on_metal(Queue& q) { return impl::is_metal(impl::backend_of(sycl_queue(q))); }

// Every lane writes a strided share of `words` local words, then reads the mirror-image share,
// which other lanes wrote: the whole request is touched and the barrier is load-bearing.
std::uint32_t slm_word(std::size_t i, std::size_t g) { return static_cast<std::uint32_t>(i * 2654435761u ^ g); }

void launch_touching_slm(Queue& q, std::size_t wg, std::size_t words, std::uint32_t* out, int groups) {
    q->submit([&](sycl::handler& h) {
        sycl::local_accessor<std::uint32_t, 1> slm(sycl::range<1>(words), h);
        h.parallel_for<LocalMemCeilingKernel>(
            sycl::nd_range<1>(wg * groups, wg), [=](sycl::nd_item<1> it) {
                const std::size_t l = it.get_local_id(0), g = it.get_group(0);
                for (std::size_t i = l; i < words; i += wg) slm[i] = slm_word(i, g);
                sycl::group_barrier(it.get_group());
                std::uint32_t acc = 0;
                for (std::size_t i = l; i < words; i += wg) acc += slm[words - 1 - i];
                out[g * wg + l] = acc;
            });
    });
    q.wait();
}

TEST(LaunchCeiling, LocalMemoryAtTheAdvertisedBudgetLaunches) {
    auto q = gpu_queue();
    if (!q) GTEST_SKIP() << "no GPU";
    const sycl::device dev = sycl_queue(*q).get_device();
    const std::size_t max_wg = dev.get_info<sycl::info::device::max_work_group_size>();
    constexpr int kGroups = 3;
    for (const std::size_t wg : {std::size_t{32}, std::size_t{256}, max_wg}) {
        // What callers launch: the budget at this width. On Metal also the raw per-width
        // figure, where the runtime's scratch leaves no slack at all.
        std::vector<std::size_t> sizes{resident::device_slm_budget(impl::local_mem_bytes(dev, wg))};
        if (on_metal(*q)) sizes.push_back(impl::local_mem_bytes(dev, wg));
        for (const std::size_t bytes : sizes) {
            const std::size_t words = bytes / sizeof(std::uint32_t);
            ASSERT_GT(words, wg) << "wg=" << wg;
            UnifiedVector<std::uint32_t> out(wg * kGroups, 0xdeadbeefu);
            ASSERT_NO_THROW(launch_touching_slm(*q, wg, words, out.data(), kGroups))
                << "wg=" << wg << ": " << bytes << " B is advertised but does not launch";
            for (int g = 0; g < kGroups; ++g)
                for (std::size_t l = 0; l < wg; ++l) {
                    std::uint32_t want = 0;
                    for (std::size_t i = l; i < words; i += wg) want += slm_word(words - 1 - i, g);
                    ASSERT_EQ(out[g * wg + l], want) << "wg=" << wg << " bytes=" << bytes << " g=" << g << " l=" << l;
                }
        }
    }
}

// The other half of the tight pair: one 16-byte granule past the per-width figure is refused
// (acpp patch 0003's check), so local_mem_bytes(dev, wg) is the boundary, not merely below it.
TEST(LaunchCeiling, MetalRefusesOneGranulePastThePerWidthFigure) {
    auto q = gpu_queue();
    if (!q || !on_metal(*q)) GTEST_SKIP() << "the per-work-item scratch is acpp Metal's";
    const sycl::device dev = sycl_queue(*q).get_device();
    const std::size_t max_wg = dev.get_info<sycl::info::device::max_work_group_size>();
    for (const std::size_t wg : {std::size_t{32}, max_wg}) {
        const std::size_t words = impl::local_mem_bytes(dev, wg) / sizeof(std::uint32_t) + 4;
        UnifiedVector<std::uint32_t> out(wg, 0u);
        EXPECT_ANY_THROW(launch_touching_slm(*q, wg, words, out.data(), 1)) << "wg=" << wg;
        UnifiedVector<std::uint32_t> ok(wg, 0u);
        EXPECT_NO_THROW(launch_touching_slm(*q, wg, words - 4, ok.data(), 1)) << "the queue did not survive";
    }
}

// select::Device restates the launch budget without SYCL (select.cc hard-codes Metal's 8 B and
// 32-bit grid); a drift there lets can_run admit what the launcher refuses.
TEST(LaunchCeiling, SelectorBudgetIsTheLaunchBudget) {
    auto q = gpu_queue();
    if (!q) GTEST_SKIP() << "no GPU";
    const sycl::device dev = sycl_queue(*q).get_device();
    const select::Device& d = select::describe(q->device(), q->backend(), false);
    const int max_wg = static_cast<int>(dev.get_info<sycl::info::device::max_work_group_size>());
    EXPECT_EQ(d.slm_budget, std::int64_t(resident::device_slm_budget(impl::local_mem_bytes(dev))));
    for (const int wg : {1, 32, 256, 512, max_wg})
        EXPECT_EQ(d.slm_budget_for(wg), std::int64_t(resident::device_slm_budget(impl::local_mem_bytes(dev, wg))))
            << "wg=" << wg;
    if (on_metal(*q)) {
        EXPECT_EQ(d.max_groups_yz, std::int64_t(impl::kMetalMaxGlobalRange));
        EXPECT_LT(d.slm_budget, d.slm_budget_for(256)) << "Metal's budget does not depend on the width";
    } else {
        EXPECT_EQ(d.max_groups_yz, 65535);
    }
}

// Metal has no 65535 cap on the work-group count in any dimension (CUDA has, in y and z); its
// limit is a 32-bit work-item count per dimension, past which ids are silently wrong.
TEST(LaunchCeiling, MetalGridRunsPastTheCudaGroupCapAndStopsAt32Bits) {
    auto q = gpu_queue();
    if (!q || !on_metal(*q)) GTEST_SKIP() << "CUDA's caps are util_device_queue_tests'";
    constexpr std::size_t kGroups = 65537;
    for (int dim = 0; dim < 3; ++dim) {
        UnifiedVector<std::uint32_t> cnt(kGroups, 0u);
        std::uint32_t* c = cnt.data();
        sycl::range<3> global(1, 1, 1);
        global[dim] = kGroups;
        (*q)->parallel_for<GroupCountKernel>(sycl::nd_range<3>(global, sycl::range<3>(1, 1, 1)),
                                             [=](sycl::nd_item<3> it) {
                                                 const std::size_t g = it.get_group(0) + it.get_group(1) + it.get_group(2);
                                                 sycl::atomic_ref<std::uint32_t, sycl::memory_order::relaxed,
                                                                  sycl::memory_scope::device> r(c[g]);
                                                 r.fetch_add(1u);
                                             });
        q->wait();
        for (std::size_t g = 0; g < kGroups; ++g) ASSERT_EQ(cnt[g], 1u) << "dim=" << dim << " group " << g;
    }
    // At the 32-bit ceiling every group runs once; one work-group more is refused before launch.
    constexpr std::size_t kWg = 1024;
    const std::size_t at = (std::size_t{1} << 22) - 1;
    UnifiedVector<std::uint32_t> cnt(at + 1, 0u);
    std::uint32_t* c = cnt.data();
    auto launch = [&](std::size_t groups) {
        (*q)->parallel_for<GlobalRangeKernel>(sycl::nd_range<1>(groups * kWg, kWg), [=](sycl::nd_item<1> it) {
            if (it.get_local_id(0) != 0) return;
            sycl::atomic_ref<std::uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device> r(c[it.get_group(0)]);
            r.fetch_add(1u);
        });
        q->wait();
    };
    ASSERT_LE(at * kWg, impl::kMetalMaxGlobalRange);
    launch(at);
    for (std::size_t g = 0; g < at; ++g) ASSERT_EQ(cnt[g], 1u) << "group " << g;
    try {
        launch(at + 1);
        ADD_FAILURE() << "a 2^32 work-item dimension launched";
    } catch (const sycl::exception& e) {
        EXPECT_EQ(e.code(), sycl::errc::nd_range) << e.what();
    }
    q->wait();
    for (std::size_t g = 0; g < at; ++g) ASSERT_EQ(cnt[g], 1u) << "the refused launch ran: group " << g;
}

#if BATCHLAS_HAS_SYCL_BACKEND
// A saturating batch past 65535 on the grid gemms (batch in SYCL dim 0). Items differ only in
// A(0,0), small integers keep every product exact, so a wrapped or clamped batch index shows as
// a wrong row 0; ld/stride padding and beta = 1 catch a derived accessor.
TEST(LaunchCeiling, MetalGemmPinsServeABatchPastTheCudaGrid) {
    auto q = gpu_queue();
    if (!q || !on_metal(*q)) GTEST_SKIP() << "CUDA refuses this batch (gemm_candidates_tests)";
    constexpr int n = 16, ld = n + 3, stride = ld * n + 5, batch = 65537;
    auto a0 = [](int i, int l) { return float((i * 7 + l * 3) % 9 - 4); };
    auto bv = [](int l, int j) { return float((l * 5 + j * 11) % 7 - 3); };
    auto v = [](int b) { return float(b % 13 - 6 + ((b >> 16) & 1) * 3); };  // item 0 != item 65536
    auto cin = [](int b) { return float(b % 3); };
    std::vector<float> base(n * n, 0.0f);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
            for (int l = 0; l < n; ++l) base[i + j * n] += a0(i, l) * bv(l, j);
    const std::size_t total = std::size_t(stride) * batch;
    UnifiedVector<float> A(total, 777.0f), Bm(total, 777.0f), C(total, 777.0f);
    for (int b = 0; b < batch; ++b)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) {
                const std::size_t o = std::size_t(b) * stride + std::size_t(j) * ld + i;
                A[o] = (i == 0 && j == 0) ? v(b) : a0(i, j);
                Bm[o] = bv(i, j);
            }
    const MatrixView<float> Av(A.data(), n, n, ld, stride, batch), Bv(Bm.data(), n, n, ld, stride, batch),
        Cv(C.data(), n, n, ld, stride, batch);
    for (const char* pin : {"direct", "tiled", "reg:m=32:n=32:k=8:u=1", "reg:m=128:n=128:k=8:u=1", "wide:m=16:n=16:k=16"}) {
        for (int b = 0; b < batch; ++b)
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i) C[std::size_t(b) * stride + std::size_t(j) * ld + i] = cin(b);
        const ScopedEnvVar route("BATCHLAS_GEMM_ROUTE", pin);
        auto run = [&] {
            (void)gemm<Backend::SYCL, float>(*q, Av, Bv, Cv, 1.0f, 1.0f, Transpose::NoTrans, Transpose::NoTrans);
        };
        ASSERT_NO_THROW(run()) << pin << " refused a batch of " << batch;
        q->wait();
        for (int b = 0; b < batch; ++b)
            for (int j = 0; j < ld * n + (b + 1 < batch ? 5 : 0); ++j) {
                const std::size_t o = std::size_t(b) * stride + j;
                const int col = j / ld, row = j % ld;
                if (col >= n || row >= n) {
                    ASSERT_EQ(C[o], 777.0f) << pin << ": padding written at item " << b << " offset " << j;
                    continue;
                }
                const float want = base[row + col * n] + (row == 0 ? (v(b) - a0(0, 0)) * bv(0, col) : 0.0f) + cin(b);
                ASSERT_EQ(C[o], want) << pin << " item " << b << " (" << row << "," << col << ")";
            }
    }
}
#endif

}  // namespace
