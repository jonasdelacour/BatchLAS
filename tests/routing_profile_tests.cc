// The routing-profile plumbing: the nearest-profile map, BATCHLAS_ROUTING_PROFILE, and
// the guarantee that EVERY OpShape builder fills the device facts through one helper.
//
// The map and the override need no GPU. The builder half does: it calls each builder
// on a real GPU queue and requires the facts that builder recorded to be the device's.

#include <gtest/gtest.h>

#include <batchlas/arch/arch_key.hh>
#include <batchlas/blas/dispatch/device_facts.hh>
#include <batchlas/blas/functions/gesvd.hh>
#include <batchlas/blas/functions/ormqr.hh>
#include <batchlas/blas/functions/syev.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/error.hh>
#include <batchlas/util/env.hh>
#include <batchlas/util/sycl-device-queue.hh>

#include "../src/backends/gemm_variant.hh"
#include "../src/backends/gemv_route.hh"
#include "../src/backends/geqrf_route.hh"
#include "../src/backends/gesv_route.hh"
#include "../src/backends/getrf_route.hh"
#include "../src/backends/getri_route.hh"
#include "../src/backends/getrs_route.hh"
#include "../src/backends/level3_coverage.hh"
#include "../src/backends/orgqr_route.hh"
#include "../src/backends/posv_route.hh"
#include "../src/backends/potrf_route.hh"
#include "../src/backends/spmm_route.hh"
#include "../src/backends/trsm_route.hh"

#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

using namespace batchlas;
using arch::ArchKey;
using arch::ArchVendor;
using arch::ProfileChoice;
using arch::RoutingProfile;

namespace {

constexpr ArchKey nv(int cc) { return {ArchVendor::NVIDIA, cc}; }
constexpr ProfileChoice exact(RoutingProfile p) { return {p, false}; }
constexpr ProfileChoice near(RoutingProfile p) { return {p, true}; }

// The map is constexpr, so its contract is checked at compile time as well.
static_assert(arch::nearest_profile(nv(89)) == exact(RoutingProfile::sm_89));
static_assert(arch::nearest_profile(nv(120)) == exact(RoutingProfile::sm_120));

TEST(RoutingProfileMap, StraddlesEveryEdge) {
    struct Case { int cc; ProfileChoice want; };
    const Case cases[] = {
        {0,   near(RoutingProfile::sm_89)},
        {75,  near(RoutingProfile::sm_89)},
        {80,  near(RoutingProfile::sm_89)},
        {86,  near(RoutingProfile::sm_89)},
        {87,  near(RoutingProfile::sm_89)},
        {88,  near(RoutingProfile::sm_89)},
        {89,  exact(RoutingProfile::sm_89)},
        {90,  near(RoutingProfile::sm_89)},
        {99,  near(RoutingProfile::sm_89)},
        {100, near(RoutingProfile::sm_120)},
        {103, near(RoutingProfile::sm_120)},
        {119, near(RoutingProfile::sm_120)},
        {120, exact(RoutingProfile::sm_120)},
        {121, near(RoutingProfile::sm_120)},
        {122, near(RoutingProfile::sm_89)},
        {130, near(RoutingProfile::sm_89)},
    };
    for (const auto& c : cases) {
        const ProfileChoice got = arch::nearest_profile(nv(c.cc));
        EXPECT_EQ(got.profile, c.want.profile) << "cc=" << c.cc;
        EXPECT_EQ(got.nearest, c.want.nearest) << "cc=" << c.cc;
    }
}

TEST(RoutingProfileMap, NonCudaDevicesBorrowSm89) {
    for (ArchVendor v : {ArchVendor::Other, ArchVendor::AMD, ArchVendor::Intel, ArchVendor::CPU}) {
        // A stray cc on a non-NVIDIA key must not select a CUDA profile.
        for (int cc : {0, 89, 120}) {
            EXPECT_EQ(arch::nearest_profile({v, cc}), near(RoutingProfile::sm_89))
                << "vendor=" << static_cast<int>(v) << " cc=" << cc;
        }
    }
}

TEST(RoutingProfileMap, ParseIsExact) {
    EXPECT_EQ(arch::parse_routing_profile("sm_89"), RoutingProfile::sm_89);
    EXPECT_EQ(arch::parse_routing_profile("sm_120"), RoutingProfile::sm_120);
    for (const char* bad : {"", "SM_89", "sm89", "sm_100", "auto", "unset", "sm_120 ", "89"}) {
        EXPECT_FALSE(arch::parse_routing_profile(bad).has_value()) << '"' << bad << '"';
    }
    EXPECT_EQ(arch::to_string(RoutingProfile::sm_89), "sm_89");
    EXPECT_EQ(arch::to_string(RoutingProfile::sm_120), "sm_120");
    EXPECT_EQ(arch::to_string(RoutingProfile::Unset), "unset");
}

TEST(RoutingProfileMap, ForcedProfileIsNearestUnlessExact) {
    EXPECT_EQ(arch::select_profile(nv(89), std::nullopt), exact(RoutingProfile::sm_89));
    EXPECT_EQ(arch::select_profile(nv(89), RoutingProfile::sm_120), near(RoutingProfile::sm_120));
    EXPECT_EQ(arch::select_profile(nv(120), RoutingProfile::sm_120), exact(RoutingProfile::sm_120));
    EXPECT_EQ(arch::select_profile(nv(120), RoutingProfile::sm_89), near(RoutingProfile::sm_89));
    // Forcing the profile an unmeasured device already borrows is still a substitution.
    EXPECT_EQ(arch::select_profile(nv(90), RoutingProfile::sm_89), near(RoutingProfile::sm_89));
    EXPECT_EQ(arch::select_profile(nv(120), RoutingProfile::Unset), exact(RoutingProfile::sm_120));
}

TEST(RoutingProfileOverride, UnsetAndEmptyMeanNoOverride) {
    {
        ScopedEnvVar e("BATCHLAS_ROUTING_PROFILE", nullptr);
        EXPECT_FALSE(dispatch::routing_profile_override().has_value());
    }
    {
        ScopedEnvVar e("BATCHLAS_ROUTING_PROFILE", "");
        EXPECT_FALSE(dispatch::routing_profile_override().has_value());
    }
}

TEST(RoutingProfileOverride, RecognisedValuesAreReturned) {
    {
        ScopedEnvVar e("BATCHLAS_ROUTING_PROFILE", "sm_120");
        EXPECT_EQ(dispatch::routing_profile_override(), RoutingProfile::sm_120);
    }
    {
        ScopedEnvVar e("BATCHLAS_ROUTING_PROFILE", "sm_89");
        EXPECT_EQ(dispatch::routing_profile_override(), RoutingProfile::sm_89);
    }
}

TEST(RoutingProfileOverride, UnrecognisedValueThrowsAndNamesTheVariable) {
    ScopedEnvVar e("BATCHLAS_ROUTING_PROFILE", "sm_100");
    try {
        (void)dispatch::routing_profile_override();
        FAIL() << "an unrecognised profile must not silently mean automatic";
    } catch (const batchlas::invalid_argument& ex) {
        const std::string what = ex.what();
        EXPECT_NE(what.find("BATCHLAS_ROUTING_PROFILE"), std::string::npos) << what;
        EXPECT_NE(what.find("sm_100"), std::string::npos) << what;
    }
}

// ---------------------------------------------------------------------------
// Every builder, on a GPU queue.

using Probe = std::function<dispatch::OpShape(const Queue&)>;

struct Buffers {
    static constexpr int n = 8, batch = 2;
    std::vector<float> a = std::vector<float>(n * n * batch, 1.0f);
    std::vector<float> b = std::vector<float>(n * n * batch, 1.0f);
    std::vector<float> c = std::vector<float>(n * n * batch, 1.0f);
    std::vector<float> x = std::vector<float>(n * batch, 1.0f);
    std::vector<float> y = std::vector<float>(n * batch, 1.0f);
    std::vector<int> ro = std::vector<int>((n + 1) * batch, 0);
    std::vector<int> ci = std::vector<int>(n * batch, 0);

    MatrixView<float, MatrixFormat::Dense> A() { return {a.data(), n, n, n, n * n, batch}; }
    MatrixView<float, MatrixFormat::Dense> B() { return {b.data(), n, n, n, n * n, batch}; }
    MatrixView<float, MatrixFormat::Dense> C() { return {c.data(), n, n, n, n * n, batch}; }
    VectorView<float> X() { return {x.data(), n, batch, 1, n}; }
    VectorView<float> Y() { return {y.data(), n, batch, 1, n}; }
    MatrixView<float, MatrixFormat::CSR> S() {
        return {a.data(), ro.data(), ci.data(), n, n, NonZeros{n}, n, n + 1, batch};
    }
};

// Ops with no shape builder: they never reach the resolver or coverage, so there is
// nothing to fill. Adding a builder for one of them means moving it into probes().
const std::set<dispatch::Op> kNoBuilder = {
    dispatch::Op::hemm, dispatch::Op::herk, dispatch::Op::her2k, dispatch::Op::iluk,
};

template <typename Opt>
dispatch::OpShape take(const Opt& s) {
    if (!s) throw std::runtime_error("builder declined a valid shape");
    return static_cast<const dispatch::OpShape&>(*s);
}

std::map<dispatch::Op, Probe> probes(const std::shared_ptr<Buffers>& m) {
    using dispatch::Op;
    constexpr Backend CU = Backend::CUDA;
    namespace bd = batchlas::backend;
    namespace dd = batchlas::blas::dispatch::detail;
    const auto l3 = [](Op op) {
        return [op](const Queue& q) {
            return bd::detail::level3_op_shape(q, op, Buffers::n, Buffers::n, Buffers::n,
                                               Buffers::batch);
        };
    };
    return {
        {Op::gemm, [m](const Queue& q) {
             return take(bd::gemm_op_shape<float>(q, m->A(), m->B(), m->C(), Transpose::NoTrans,
                                                  Transpose::NoTrans, ComputePrecision::Default));
         }},
        {Op::gemv, [m](const Queue& q) {
             return take(bd::gemv_op_shape<CU, float>(q, m->A(), m->X(), m->Y(), Transpose::NoTrans));
         }},
        {Op::trsm, [m](const Queue& q) {
             return take(bd::trsm_op_shape<float>(q, m->A(), m->B(), Side::Left, Uplo::Lower,
                                                  Transpose::NoTrans, Diag::NonUnit));
         }},
        {Op::trmm, l3(Op::trmm)},
        {Op::symm, l3(Op::symm)},
        {Op::syrk, l3(Op::syrk)},
        {Op::syr2k, l3(Op::syr2k)},
        {Op::potrf, [m](const Queue& q) {
             return take(bd::potrf_op_shape<CU, float>(q, m->A(), Uplo::Lower));
         }},
        {Op::getrf, [m](const Queue& q) { return take(bd::getrf_op_shape<CU, float>(q, m->A())); }},
        {Op::getrs, [m](const Queue& q) {
             return take(bd::getrs_op_shape<CU, float>(q, m->A(), m->B(), Transpose::NoTrans));
         }},
        {Op::getri, [m](const Queue& q) { return take(bd::getri_op_shape<CU, float>(q, m->A())); }},
        {Op::geqrf, [m](const Queue& q) { return take(bd::geqrf_op_shape<CU, float>(q, m->A())); }},
        {Op::orgqr, [m](const Queue& q) { return take(bd::orgqr_op_shape<CU, float>(q, m->A())); }},
        {Op::ormqr, [m](const Queue& q) {
             return dd::ormqr_op_shape<float>(q, m->A(), Side::Left, Transpose::NoTrans);
         }},
        {Op::syev, [m](const Queue& q) {
             return static_cast<dispatch::OpShape>(
                 dd::syev_op_shape<float>(q, CU, m->A(), Uplo::Lower, JobType::EigenVectors));
         }},
        {Op::gesvd, [m](const Queue& q) {
             return static_cast<dispatch::OpShape>(dd::gesvd_op_shape<float>(
                 q, m->A(), SvdVectors::Thin, SvdVectors::Thin, std::nullopt));
         }},
        {Op::spmm, [m](const Queue& q) {
             return take(bd::spmm_op_shape<CU, float, MatrixFormat::CSR>(
                 q, m->S(), m->B(), m->C(), Transpose::NoTrans, Transpose::NoTrans));
         }},
        {Op::gesv, [m](const Queue& q) { return take(bd::gesv_op_shape<CU, float>(q, m->A(), m->B())); }},
        {Op::posv, [m](const Queue& q) {
             return take(bd::posv_op_shape<CU, float>(q, m->A(), m->B(), Uplo::Lower));
         }},
    };
}

std::unique_ptr<Queue> gpu_queue() {
    try {
        if (Device::get_devices(DeviceType::GPU).empty()) return nullptr;
        return std::make_unique<Queue>(Device("gpu"));
    } catch (...) {
        return nullptr;
    }
}

TEST(RoutingProfileBuilders, EveryOpHasAProbeOrAnExemption) {
    const auto table = probes(std::make_shared<Buffers>());
    for (int i = 0; i < static_cast<int>(dispatch::Op::COUNT); ++i) {
        const auto op = static_cast<dispatch::Op>(i);
        const bool probed = table.count(op) != 0;
        const bool exempt = kNoBuilder.count(op) != 0;
        EXPECT_NE(probed, exempt) << dispatch::op_name(op)
                                  << ": every Op needs exactly one of a probe or an exemption";
    }
}

TEST(RoutingProfileBuilders, EveryBuilderFillsTheDeviceFacts) {
    const auto q = gpu_queue();
    if (!q) GTEST_SKIP() << "no GPU queue";
    ScopedEnvVar unset("BATCHLAS_ROUTING_PROFILE", nullptr);

    const dispatch::DeviceFacts f = dispatch::device_facts(q->device());
    const ProfileChoice want = dispatch::routing_profile(q->device());
    ASSERT_NE(want.profile, RoutingProfile::Unset);
    ASSERT_TRUE(f.is_gpu);
    EXPECT_GT(f.compute_units, 0);
    EXPECT_GT(f.max_sub_group, 0);
    EXPECT_EQ(f.key.cuda_cc, q->device().cuda_compute_capability());
    if (f.key.cuda_cc > 0) EXPECT_EQ(f.key.vendor, ArchVendor::NVIDIA);

    for (const auto& [op, probe] : probes(std::make_shared<Buffers>())) {
        SCOPED_TRACE(std::string(dispatch::op_name(op)));
        const dispatch::OpShape s = probe(*q);
        EXPECT_EQ(s.op, op);
        EXPECT_EQ(s.is_gpu, f.is_gpu);
        EXPECT_EQ(s.max_sub_group, f.max_sub_group);
        EXPECT_EQ(s.compute_units, f.compute_units);
        EXPECT_EQ(s.cuda_cc, f.key.cuda_cc);
        EXPECT_EQ(s.profile, want.profile);
        EXPECT_EQ(s.profile_nearest, want.nearest);
    }
}

TEST(RoutingProfileBuilders, EveryBuilderSeesTheOverride) {
    const auto q = gpu_queue();
    if (!q) GTEST_SKIP() << "no GPU queue";
    const auto key = dispatch::device_facts(q->device()).key;
    for (RoutingProfile forced : {RoutingProfile::sm_89, RoutingProfile::sm_120}) {
        ScopedEnvVar e("BATCHLAS_ROUTING_PROFILE", std::string(arch::to_string(forced)).c_str());
        const ProfileChoice want = arch::select_profile(key, forced);
        for (const auto& [op, probe] : probes(std::make_shared<Buffers>())) {
            SCOPED_TRACE(std::string(dispatch::op_name(op)) + " forced " +
                         std::string(arch::to_string(forced)));
            const dispatch::OpShape s = probe(*q);
            EXPECT_EQ(s.profile, forced);
            EXPECT_EQ(s.profile_nearest, want.nearest);
        }
    }
}

TEST(RoutingProfileBuilders, EveryBuilderRejectsAnUnrecognisedOverride) {
    const auto q = gpu_queue();
    if (!q) GTEST_SKIP() << "no GPU queue";
    ScopedEnvVar e("BATCHLAS_ROUTING_PROFILE", "sm_8.9");
    for (const auto& [op, probe] : probes(std::make_shared<Buffers>())) {
        EXPECT_THROW((void)probe(*q), batchlas::invalid_argument) << dispatch::op_name(op);
    }
}

} // namespace
