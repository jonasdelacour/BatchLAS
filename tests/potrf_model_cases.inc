// The cost-model route choice (route_potrf.hh model_pick, potrf_route.hh potrf_price_routes,
// the generated src/backends/potrf_profile_constants.hh). The pinned picks are fit.py's
// full-model choices from evaluation/routing/profiles/sm_120.md, NOT re-derived here.
// evidence: docs/perf/potrf.md#the-cost-model-route-choice

constexpr launch_plan::DeviceFacts kSm120Facts{101376, 1024, 188, 0, 0};

template <typename T>
dispatch::PotrfShape model_shape(int n, int batch, Uplo uplo, arch::RoutingProfile p) {
    auto s = backend::potrf_op_shape_from_facts<Backend::CUDA, T>(
        kSm120Facts, n, n, batch, uplo, true, true, false, true);
    s.compute_units = kSm120Facts.compute_units;
    s.profile = p;
    backend::potrf_price_routes<T>(s, kSm120Facts, /*leaf_trsm=*/32, 0, 0);
    return s;
}

template <typename T>
std::string model_route(int n, int batch, Uplo uplo, arch::RoutingProfile p, bool vendor = true) {
    const auto s = model_shape<T>(n, batch, uplo, p);
    const auto r = dispatch::resolve_route_uninstrumented<dispatch::Op::potrf, T>(
        dispatch::Route{}, s, vendor);
    return std::string(dispatch::to_string(r.origin)) + ":" + std::string(dispatch::to_string(r.algo));
}

struct PinnedPick {
    char type;
    int n, batch;
    Uplo uplo;
    const char* model;   // sm_120 full-fit pick
    const char* windows; // today's hand windows
};

// Route-diff rows of profiles/sm_120.md (model != windows), plus four it leaves alone.
const PinnedPick kSm120Picks[] = {
    {'z', 3, 512, Uplo::Lower, "native:lpanel", "native:tiny"},
    {'z', 20, 8192, Uplo::Lower, "native:lpanel", "vendor:auto"},
    {'c', 20, 8192, Uplo::Lower, "native:lpanel", "native:tiny"},
    {'c', 64, 128, Uplo::Lower, "vendor:auto", "native:lpanel"},
    {'c', 224, 128, Uplo::Lower, "vendor:auto", "native:lpanel"},
    {'d', 1, 8192, Uplo::Lower, "native:lpanel", "vendor:auto"},
    {'d', 12, 128, Uplo::Lower, "vendor:auto", "native:tiny"},
    {'d', 28, 2048, Uplo::Lower, "native:lpanel", "vendor:auto"},
    {'s', 288, 8192, Uplo::Lower, "native:lpanel", "vendor:auto"},
    {'s', 1280, 512, Uplo::Lower, "native:blocked", "vendor:auto"},
    {'s', 8, 32768, Uplo::Lower, "native:tiny", "native:tiny"},
    {'s', 64, 8192, Uplo::Lower, "native:lpanel", "native:lpanel"},
    {'z', 128, 2048, Uplo::Lower, "vendor:auto", "vendor:auto"},
    {'s', 16, 8192, Uplo::Upper, "native:tiny", "native:tiny"},
};

std::string pick_any(const PinnedPick& p, arch::RoutingProfile prof) {
    switch (p.type) {
        case 's': return model_route<float>(p.n, p.batch, p.uplo, prof);
        case 'd': return model_route<double>(p.n, p.batch, p.uplo, prof);
        case 'c': return model_route<std::complex<float>>(p.n, p.batch, p.uplo, prof);
        default: return model_route<std::complex<double>>(p.n, p.batch, p.uplo, prof);
    }
}

TEST(PotrfCostModel, Sm120PicksMatchTheFit) {
    for (const auto& p : kSm120Picks) {
        EXPECT_EQ(pick_any(p, arch::RoutingProfile::sm_120), p.model)
            << p.type << " n=" << p.n << " batch=" << p.batch;
    }
}

// sm_89's gate failed, so its profile must route exactly as the hand windows do.
TEST(PotrfCostModel, GateFailedProfileKeepsTheWindows) {
    for (const auto& p : kSm120Picks) {
        EXPECT_EQ(pick_any(p, arch::RoutingProfile::sm_89), p.windows)
            << p.type << " n=" << p.n << " batch=" << p.batch;
        EXPECT_EQ(pick_any(p, arch::RoutingProfile::Unset), p.windows)
            << p.type << " n=" << p.n << " batch=" << p.batch;
    }
    EXPECT_FALSE(model_shape<float>(64, 8192, Uplo::Lower, arch::RoutingProfile::sm_89).model_enabled);
}

// Synthetic costs on a supported shape: the margin, the vendor-free walk, and supports().
TEST(PotrfCostModel, MarginVendorFreeAndSupports) {
    using Tbl = dispatch::RouteTable<dispatch::Op::potrf, float>;
    auto s = model_shape<float>(64, 8192, Uplo::Lower, arch::RoutingProfile::sm_120);
    ASSERT_TRUE(s.model_enabled);
    const auto pick = [&](bool vendor) {
        return dispatch::resolve_route_uninstrumented<dispatch::Op::potrf, float>(
            dispatch::Route{}, s, vendor);
    };
    for (double& c : s.model_cost) c = -1;
    s.model_cost[4] = 1.0;     // vendor
    s.model_cost[3] = 0.96;    // blocked: within the 5% margin -> vendor keeps it
    EXPECT_TRUE(dispatch::is_vendor(pick(true)));
    EXPECT_EQ(pick(false).algo, dispatch::Algorithm::Blocked);   // vendor-free: argmin native
    s.model_cost[3] = 0.94;    // beyond the margin -> native
    EXPECT_EQ(pick(true).algo, dispatch::Algorithm::Blocked);
    s.model_cost[0] = 0.01;    // Tiny is cheapest but supports() refuses n = 64
    EXPECT_FALSE(Tbl::supports(dispatch::kPotrfOrder[0], s));
    EXPECT_EQ(pick(true).algo, dispatch::Algorithm::Blocked);
    s.model_cost[2] = 0.5;     // LPanel is supported and cheaper
    EXPECT_EQ(pick(true).algo, dispatch::Algorithm::LPanel);
    s.model_extrapolated[2] = true;
    EXPECT_EQ(Tbl::cost_extrapolated(pick(true), s), 1);
    s.model_enabled = false;   // the windows again
    EXPECT_EQ(Tbl::cost_extrapolated(pick(true), s), -1);
}

// On this box's real queue: the library's route follows the device's profile.
TEST(PotrfCostModel, DeviceRouteFollowsTheProfile) {
    std::unique_ptr<Queue> q;
    try {
        q = std::make_unique<Queue>(Device("gpu"), Backend::CUDA, true);
    } catch (...) {
        GTEST_SKIP() << "no CUDA GPU";
    }
    if (q->device().cuda_compute_capability() != 120) GTEST_SKIP() << "pins are sm_120 picks";
    const ScopedEnvVar clear_route("BATCHLAS_POTRF_ROUTE", "");
    const ScopedEnvVar natural("BATCHLAS_ROUTING_PROFILE", "");   // the device's own profile
    const MatrixView<double, MatrixFormat::Dense> A(nullptr, 28, 28, 28, 28 * 28, 2048);
    const auto modelled = backend::potrf_route<Backend::CUDA, double>(*q, A, Uplo::Lower, true);
    EXPECT_EQ(modelled.algo, dispatch::Algorithm::LPanel);
    const ScopedEnvVar windows_profile("BATCHLAS_ROUTING_PROFILE", "sm_89");
    const auto windowed = backend::potrf_route<Backend::CUDA, double>(*q, A, Uplo::Lower, true);
    EXPECT_TRUE(dispatch::is_vendor(windowed));
}
