// The dispatch vocabulary: Route parsing, the legacy environment spellings, and
// the per-op RouteTable supports()/preferred() windows.
//
// The legacy spellings appear in committed benchmark scripts and in the provenance
// of recorded results, so their mapping is pinned here rather than reimplemented.
// evidence: docs/perf/dispatch.md

#include <gtest/gtest.h>

#include <batchlas/blas/dispatch/route.hh>
#include <batchlas/blas/dispatch/route_env.hh>
#include <batchlas/blas/dispatch/route_trsm.hh>
#include <batchlas/blas/dispatch/route_potrf.hh>
#include <batchlas/blas/dispatch/route_geqrf.hh>
#include <batchlas/blas/dispatch/route_orgqr.hh>
#include <batchlas/blas/dispatch/route_getrf.hh>
#include <batchlas/blas/dispatch/route_getrs.hh>
#include <batchlas/blas/dispatch/route_getri.hh>
#include <batchlas/blas/dispatch/route_gemv.hh>
#include <batchlas/blas/dispatch/route_spmm.hh>

#include <complex>
#include <cstdlib>
#include <string>

using namespace batchlas;
using namespace batchlas::dispatch;

namespace {

// Sets an env var for the object's lifetime and restores it, so cases cannot leak.
class ScopedEnv {
public:
    ScopedEnv(std::string key, const char* value) : key_(std::move(key)) {
        if (const char* old = std::getenv(key_.c_str())) {
            had_ = true;
            old_ = old;
        }
        if (value) {
            ::setenv(key_.c_str(), value, 1);
        } else {
            ::unsetenv(key_.c_str());
        }
    }
    ~ScopedEnv() {
        if (had_) {
            ::setenv(key_.c_str(), old_.c_str(), 1);
        } else {
            ::unsetenv(key_.c_str());
        }
    }
private:
    std::string key_;
    std::string old_;
    bool had_ = false;
};

// Clears both spellings so a case starts from a known state.
struct ClearRouteEnv {
    explicit ClearRouteEnv(Op op)
        : canonical_("BATCHLAS_" + op_env_stem(op) + "_ROUTE", nullptr),
          legacy_(std::string(legacy_variable_for(op)).empty()
                      ? std::string("BATCHLAS_UNUSED_ROUTE_KEY")
                      : std::string(legacy_variable_for(op)),
                  nullptr) {}
    ScopedEnv canonical_;
    ScopedEnv legacy_;
};

} // namespace

// --- the two axes are actually separate -----------------------------------

TEST(RouteVocabulary, OriginAndAlgorithmAreIndependent) {
    const Route a{Origin::Native, Algorithm::Auto};
    const Route b{Origin::Native, Algorithm::CTA};
    const Route c{Origin::Vendor, Algorithm::CTA};

    EXPECT_NE(a, b) << "same origin, different algorithm must differ";
    EXPECT_NE(b, c) << "same algorithm, different origin must differ";
    EXPECT_TRUE(is_native(a));
    EXPECT_TRUE(is_vendor(c));
    EXPECT_FALSE(is_vendor(b));
}

TEST(RouteVocabulary, LibraryIsOutputNotIdentity) {
    // `library` is filled in by the resolver on the way out. Two requests that
    // differ only in it are the same request.
    Route a{Origin::Vendor, Algorithm::Auto};
    Route b{Origin::Vendor, Algorithm::Auto};
    a.library = BackendLibrary::CUBLAS;
    a.library_valid = true;
    b.library = BackendLibrary::ROCBLAS;
    b.library_valid = true;
    EXPECT_EQ(a, b);
}

TEST(RouteVocabulary, VendorPredicateCoversNetlib) {
    EXPECT_TRUE(is_vendor(*parse_origin_word("netlib")));
    EXPECT_TRUE(is_vendor(*parse_origin_word("vendor")));
    EXPECT_FALSE(is_vendor(*parse_origin_word("native")));
}

// --- canonical spelling ----------------------------------------------------

TEST(RouteVocabulary, ParsesOriginOnly) {
    const auto r = parse_route_value("native");
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(r->origin, Origin::Native);
    EXPECT_EQ(r->algo, Algorithm::Auto) << "a bare origin must leave the algorithm free";
}

TEST(RouteVocabulary, ParsesOriginAndAlgorithmPair) {
    const auto r = parse_route_value("native:register_tiled");
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(r->origin, Origin::Native);
    EXPECT_EQ(r->algo, Algorithm::RegisterTiled);
}

TEST(RouteVocabulary, BareAlgorithmImpliesNative) {
    const auto r = parse_route_value("cta");
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(r->origin, Origin::Native);
    EXPECT_EQ(r->algo, Algorithm::CTA);
}

TEST(RouteVocabulary, DeviceLibraryAlgorithmImpliesVendor) {
    // cuBLASDx is NVIDIA's source compiled into our .so; naming it must not claim Native.
    const auto r = parse_route_value("cublasdx");
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(r->algo, Algorithm::FusedDevice);
    EXPECT_TRUE(is_vendor(*r)) << "a device-library route is vendor code";
}

TEST(RouteVocabulary, UnknownValueIsRejectedNotSilentlyAuto) {
    EXPECT_FALSE(parse_route_value("nonsense").has_value());
    EXPECT_FALSE(parse_route_value("native:nonsense").has_value());
    EXPECT_FALSE(parse_route_value("").has_value());
}

// --- legacy spellings must keep working ------------------------------------

TEST(RouteVocabulary, LegacyGemmVariantSyclSelectsRegisterTiled) {
    ClearRouteEnv clear(Op::gemm);
    ScopedEnv set("BATCHLAS_GEMM_VARIANT", "sycl");

    const auto parsed = parse_route_env(Op::gemm);
    ASSERT_TRUE(parsed.found);
    EXPECT_EQ(parsed.route.algo, Algorithm::RegisterTiled);
    EXPECT_TRUE(is_native(parsed.route));
    EXPECT_TRUE(parsed.source.legacy);
    EXPECT_EQ(parsed.source.variable, "BATCHLAS_GEMM_VARIANT")
        << "a diagnostic must be able to quote the spelling the user typed";
}

TEST(RouteVocabulary, LegacyGemmNativeMeansRawCudaNotBatchLAS) {
    // THE TRAP: legacy BATCHLAS_GEMM_VARIANT=native names the RAW CUDA path, so it
    // must map to Origin::Vendor -- the opposite of what "native" means in the
    // canonical vocabulary. evidence: docs/perf/dispatch.md#the-environment-vocabulary
    for (const char* spelling : {"native", "cuda-native", "direct-cuda"}) {
        const auto legacy = parse_legacy_route_value(Op::gemm, spelling);
        ASSERT_TRUE(legacy.has_value()) << spelling;
        EXPECT_TRUE(is_vendor(*legacy))
            << "legacy GEMM '" << spelling << "' means the raw CUDA path, i.e. vendor";
    }

    // The canonical spelling is unaffected: there, native means native.
    const auto canonical = parse_route_value("native");
    ASSERT_TRUE(canonical.has_value());
    EXPECT_TRUE(is_native(*canonical));

    // And the collision is GEMM-specific -- no other op had that alias.
    const auto other = parse_legacy_route_value(Op::syev, "native");
    ASSERT_TRUE(other.has_value());
    EXPECT_TRUE(is_native(*other));
}

TEST(RouteVocabulary, LegacyLevel3CustomMeansTheFusedKernelNotRegisterTiled) {
    // THE SECOND COLLISION: in symm/syrk/syr2k/trmm the legacy "custom" names the
    // FUSED cuBLASDx kernel, while canonical "custom" is our register-tiled family.
    // Different kernel, same word, so these ops must not use the generic parser.
    for (Op op : {Op::symm, Op::syrk, Op::syr2k, Op::trmm}) {
        const auto legacy = parse_legacy_route_value(op, "custom");
        ASSERT_TRUE(legacy.has_value()) << op_env_stem(op);
        EXPECT_EQ(legacy->algo, Algorithm::FusedDevice) << op_env_stem(op);
        EXPECT_TRUE(is_vendor(*legacy)) << "the fused kernel is NVIDIA's source";
    }

    // The canonical spelling is unaffected, and still names our own kernel.
    const auto canonical = parse_route_value("custom");
    ASSERT_TRUE(canonical.has_value());
    EXPECT_EQ(canonical->algo, Algorithm::RegisterTiled);
    EXPECT_TRUE(is_native(*canonical));

    const auto gemm_custom = parse_legacy_route_value(Op::gemm, "custom");
    ASSERT_TRUE(gemm_custom.has_value());
    EXPECT_EQ(gemm_custom->algo, Algorithm::RegisterTiled);
    EXPECT_TRUE(is_native(*gemm_custom));
}

TEST(RouteVocabulary, LegacyLevel3TileSpellingsSurvive) {
    // `tiles` and `narrow` existed only in the ops' private parsers and have no
    // canonical spelling, so nothing but the alias table can carry them.
    for (Op op : {Op::syrk, Op::syr2k, Op::trmm}) {
        for (const char* spelling : {"triangular", "tiles"}) {
            const auto r = parse_legacy_route_value(op, spelling);
            ASSERT_TRUE(r.has_value()) << op_env_stem(op) << " " << spelling;
            EXPECT_EQ(r->algo, Algorithm::TriangularTiles)
                << op_env_stem(op) << " " << spelling;
        }
    }
    for (const char* spelling : {"gram", "narrow"}) {
        const auto r = parse_legacy_route_value(Op::syrk, spelling);
        ASSERT_TRUE(r.has_value()) << spelling;
        EXPECT_EQ(r->algo, Algorithm::GramTiles) << spelling;
    }
}

TEST(RouteVocabulary, LegacyLevel3GemmIsTheVendorMeasurementRoute) {
    // syrk/syr2k's `gemm` is the deliberately WRONG route kept for measurement:
    // it computes both triangles, and it runs through gemm_cublasdx. The
    // bare-algorithm rule would otherwise call it Native.
    for (Op op : {Op::syrk, Op::syr2k}) {
        const auto r = parse_legacy_route_value(op, "gemm");
        ASSERT_TRUE(r.has_value()) << op_env_stem(op);
        EXPECT_EQ(r->algo, Algorithm::DiagFullGemm) << op_env_stem(op);
        EXPECT_TRUE(is_vendor(*r)) << op_env_stem(op);
    }
}

TEST(RouteVocabulary, LegacyTrmmTriangularIsOneValueNotTwoReadings) {
    ClearRouteEnv clear(Op::trmm);
    ScopedEnv set("BATCHLAS_TRMM_VARIANT", "triangular");

    const auto parsed = parse_route_env(Op::trmm);
    ASSERT_TRUE(parsed.found) << "it is an opinion, not the absence of one";
    EXPECT_EQ(parsed.route.algo, Algorithm::TriangularTiles);
    EXPECT_TRUE(is_native(parsed.route));
}

TEST(RouteVocabulary, LegacyVendorSpellingMapsToVendorOrigin) {
    ClearRouteEnv clear(Op::trmm);
    ScopedEnv set("BATCHLAS_TRMM_VARIANT", "vendor");

    const auto parsed = parse_route_env(Op::trmm);
    ASSERT_TRUE(parsed.found);
    EXPECT_TRUE(is_vendor(parsed.route));
}

TEST(RouteVocabulary, LegacySyrkTriangularAndGramSurvive) {
    {
        ClearRouteEnv clear(Op::syrk);
        ScopedEnv set("BATCHLAS_SYRK_VARIANT", "triangular");
        const auto parsed = parse_route_env(Op::syrk);
        ASSERT_TRUE(parsed.found);
        EXPECT_EQ(parsed.route.algo, Algorithm::TriangularTiles);
    }
    {
        ClearRouteEnv clear(Op::syrk);
        ScopedEnv set("BATCHLAS_SYRK_VARIANT", "gram");
        const auto parsed = parse_route_env(Op::syrk);
        ASSERT_TRUE(parsed.found);
        EXPECT_EQ(parsed.route.algo, Algorithm::GramTiles);
    }
}

TEST(RouteVocabulary, LegacyProviderSpellingsSurvive) {
    ClearRouteEnv clear(Op::syev);
    ScopedEnv set("BATCHLAS_SYEV_PROVIDER", "two_stage");

    const auto parsed = parse_route_env(Op::syev);
    ASSERT_TRUE(parsed.found);
    EXPECT_EQ(parsed.route.algo, Algorithm::TwoStage);
    EXPECT_TRUE(is_native(parsed.route));
}

TEST(RouteVocabulary, CanonicalSpellingWinsOverLegacy) {
    ClearRouteEnv clear(Op::gemm);
    ScopedEnv legacy("BATCHLAS_GEMM_VARIANT", "vendor");
    ScopedEnv canonical("BATCHLAS_GEMM_ROUTE", "native:register_tiled");

    const auto parsed = parse_route_env(Op::gemm);
    ASSERT_TRUE(parsed.found);
    EXPECT_TRUE(is_native(parsed.route));
    EXPECT_FALSE(parsed.source.legacy);
}

// --- the unset default -----------------------------------------------------

TEST(RouteVocabulary, UnsetDefaultsAreAutoForEveryOp) {
    // Auto for every op. GEMM's unset default was Vendor until its window was
    // measured, and this one line changes the route of every GEMM call.
    // evidence: docs/perf/gemm.md#the-auto-flip
    EXPECT_EQ(legacy_unset_default(Op::gemm).origin, Origin::Auto);
    EXPECT_EQ(legacy_unset_default(Op::syrk).origin, Origin::Auto);
    EXPECT_EQ(legacy_unset_default(Op::symm).origin, Origin::Auto);
    EXPECT_EQ(legacy_unset_default(Op::trmm).origin, Origin::Auto);

    // Auto is not "always native": it defers to preferred(), and a named route
    // still wins. Covered in tests/route_gemm_equivalence_tests.cc.
    EXPECT_EQ(legacy_unset_default(Op::gemm).algo, Algorithm::Auto);
}

TEST(RouteVocabulary, NothingSetReportsNotFound) {
    ClearRouteEnv clear(Op::gemm);
    const auto parsed = parse_route_env(Op::gemm);
    EXPECT_FALSE(parsed.found);
    EXPECT_FALSE(parsed.unparsed);
}

TEST(RouteVocabulary, SetButUnparsedIsDistinguishableFromUnset) {
    ClearRouteEnv clear(Op::gemm);
    ScopedEnv set("BATCHLAS_GEMM_ROUTE", "not-a-route");
    const auto parsed = parse_route_env(Op::gemm);
    EXPECT_FALSE(parsed.found);
    EXPECT_TRUE(parsed.unparsed) << "a typo must be reportable, not silently Auto";
    EXPECT_EQ(parsed.source.value, "not-a-route");
}

// --- shape bucketing -------------------------------------------------------

TEST(RouteVocabulary, ShapeClassCollapsesIterationsButNotRegimes) {
    OpShape a; a.m = a.n = a.k = 512; a.batch = 128;
    OpShape b; b.m = b.n = b.k = 513; b.batch = 130;   // same power-of-two buckets
    OpShape c; c.m = c.n = c.k = 2048; c.batch = 128;  // different regime

    EXPECT_EQ(a.shape_class(), b.shape_class());
    EXPECT_NE(a.shape_class(), c.shape_class());
    EXPECT_EQ(a.max_dim(), 512);
}


// ---------------------------------------------------------------------------
// RouteTable<Op::trsm, T>. A route may be SUPPORTED while not PREFERRED: the
// vendor-off fallback re-walks the candidate order testing supports() ALONE, so
// a speed threshold placed in supports() leaves trsm with no route at all in a
// vendor-free build. evidence: docs/perf/trsm.md#what-the-spec-got-wrong
// ---------------------------------------------------------------------------
namespace {

TrsmShape trsm_shape(int64_t tri_order, int64_t q, int64_t batch, int cta_max,
                     Side side = Side::Left) {
    TrsmShape s;
    s.op = Op::trsm;
    s.scalar = ScalarKind::F32;
    s.k = tri_order;                       // the triangular order n
    s.m = tri_order;
    s.n = q;                               // Side::Left -> q = B.cols()
    if (side == Side::Right) { s.m = q; s.n = tri_order; }
    s.side = side;
    s.batch = batch;
    s.is_gpu = true;
    s.cta_max_n = cta_max;
    s.blocked_available = (cta_max > 0);
    return s;
}

using TrsmTable = RouteTable<Op::trsm, float>;
constexpr Route kCta{Origin::Native, Algorithm::CTA};
constexpr Route kBlocked{Origin::Native, Algorithm::Blocked};
constexpr Route kAuto{Origin::Auto, Algorithm::Auto};

} // namespace

TEST(RouteTrsm, SupportedButNotPreferredIsTheWholePoint) {
    // A cell that is supported and un-preferred for a NON-speed reason: batch below
    // the measured floor. evidence: docs/perf/trsm.md#the-batch-floor
    const auto s = trsm_shape(/*tri_order=*/32, /*q=*/1024, /*batch=*/1, /*cta_max=*/32,
                              Side::Left);
    EXPECT_TRUE(TrsmTable::supports(kCta, s))
        << "batch size is a speed question; it must not gate CORRECTNESS";
    EXPECT_FALSE(TrsmTable::preferred(kCta, s))
        << "batch=1 measured 0.40-0.86x and must not be preferred";
    EXPECT_TRUE(is_native(resolve_trsm_route<float>(kAuto, s, /*vendor_available=*/false)))
        << "un-preferred must never mean unroutable when there is no vendor";
    EXPECT_TRUE(is_vendor(resolve_trsm_route<float>(kAuto, s, /*vendor_available=*/true)))
        << "and with a vendor present it must take it";
}

// ---------------------------------------------------------------------------
// The shipped preferred() window. evidence: docs/perf/trsm.md#the-measured-grid
// ---------------------------------------------------------------------------

TEST(RouteTrsm, FloatLeftIsPreferredAtEveryOrder) {
    // float Side::Left is preferred at every order: CTA up to the CTA capacity,
    // the blocked driver above it.
    for (int64_t order : {8, 16, 32, 64, 128}) {
        const Route r = (order <= 32) ? kCta : kBlocked;
        EXPECT_TRUE(TrsmTable::preferred(r, trsm_shape(order, 1024, 2048, 32, Side::Left)))
            << "float Side::Left order " << order << " wins after the staging tile";
    }
    EXPECT_TRUE(TrsmTable::preferred(kBlocked, trsm_shape(256, 1024, 512, 32, Side::Left)))
        << "float Side::Left order 256 at q*batch=524288 now measures 1.32x";
    EXPECT_TRUE(TrsmTable::preferred(kBlocked, trsm_shape(512, 1024, 512, 32, Side::Left)))
        << "float Side::Left order 512 at q*batch=524288 now measures 1.28x";
    EXPECT_TRUE(TrsmTable::preferred(kBlocked, trsm_shape(512, 256, 128, 32, Side::Left)))
        << "and the small-work cells it always won stay won";

    // Side::Right is preferred at every order the CTA arm serves.
    for (int64_t order : {8, 16, 32}) {
        EXPECT_TRUE(TrsmTable::preferred(kCta, trsm_shape(order, 1024, 2048, 32, Side::Right)))
            << "float Side::Right order " << order << " measured 1.54-4.59x";
    }
}

TEST(RouteTrsm, BatchFloorIsSpeedNotCorrectness) {
    // The floor is batch 8, and it lives in preferred(): a vendor-free build at
    // batch=1 still routes native rather than throwing.
    const auto tiny = trsm_shape(16, 1024, 1, 32, Side::Right);
    EXPECT_FALSE(TrsmTable::preferred(kCta, tiny));
    EXPECT_TRUE(TrsmTable::supports(kCta, tiny));
    EXPECT_TRUE(is_native(resolve_trsm_route<float>(kAuto, tiny, /*vendor_available=*/false)));
    EXPECT_TRUE(is_vendor(resolve_trsm_route<float>(kAuto, tiny, /*vendor_available=*/true)));

    EXPECT_TRUE(TrsmTable::preferred(kCta, trsm_shape(16, 1024, 8, 32, Side::Right)));
}

TEST(RouteTrsm, DoubleAndComplexWinEveryMeasuredCell) {
    // preferred() switches on the TABLE's T, never on s.scalar, so each type needs
    // its own instantiation -- a sweep over s.scalar would test float three times.
    for (int64_t order : {8, 32}) {
        for (Side sd : {Side::Left, Side::Right}) {
            const auto s = trsm_shape(order, 1024, 2048, 32, sd);
            EXPECT_TRUE((RouteTable<Op::trsm, double>::preferred(kCta, s)))
                << "double order " << order << " worst measured cell is 1.39x";
            EXPECT_TRUE((RouteTable<Op::trsm, std::complex<float>>::preferred(kCta, s)));
            EXPECT_TRUE((RouteTable<Op::trsm, std::complex<double>>::preferred(kCta, s)));
        }
    }
    // Above the CTA capacity the blocked driver carries the same verdict.
    const auto big = trsm_shape(256, 1024, 512, 32, Side::Left);
    EXPECT_TRUE((RouteTable<Op::trsm, double>::preferred(kBlocked, big)));
    EXPECT_TRUE((RouteTable<Op::trsm, std::complex<float>>::preferred(kBlocked, big)))
        << "complex<float> order 256 Side::Left measured 21.1x";
}

TEST(RouteTrsm, VendorIsNeverItselfPreferred) {
    // The vendor route is LAST in kTrsmOrder, so a true here is indistinguishable
    // from falling through -- until someone reorders the list.
    const auto s = trsm_shape(16, 1024, 2048, 32, Side::Right);
    EXPECT_FALSE(TrsmTable::preferred(Route{Origin::Vendor, Algorithm::Auto}, s));
}

TEST(RouteTrsm, VendorFreeStillFindsANativeRouteAtEveryOrder) {
    // Below the CTA capacity -> CTA. Above it -> Blocked. Never "no route".
    const auto small = trsm_shape(32, 128, 4096, 64);
    const auto big   = trsm_shape(4096, 128, 64, 64);

    const Route rs = resolve_trsm_route<float>(kAuto, small, /*vendor_available=*/false);
    EXPECT_TRUE(is_native(rs));
    EXPECT_EQ(rs.algo, Algorithm::CTA);

    const Route rb = resolve_trsm_route<float>(kAuto, big, /*vendor_available=*/false);
    EXPECT_TRUE(is_native(rb));
    EXPECT_EQ(rb.algo, Algorithm::Blocked)
        << "an order above the register capacity must fall to the blocked driver, not vanish";
}

TEST(RouteTrsm, AbsentKernelIsUnsupportedRatherThanSelectable) {
    // cta_max_n == 0 is what a build without the kernel reports: both native routes
    // must be UNSUPPORTED, never selectable.
    const auto s = trsm_shape(32, 128, 4096, /*cta_max=*/0);
    EXPECT_FALSE(TrsmTable::supports(kCta, s));
    EXPECT_FALSE(TrsmTable::supports(kBlocked, s));
    EXPECT_TRUE(is_vendor(resolve_trsm_route<float>(kAuto, s, /*vendor_available=*/true)));
}

TEST(RouteTrsm, CorrectnessGatesAreNotSpeedGates) {
    // Every false below must be "would compute a wrong answer", so each has a
    // structural reason. Large batch and large q are NOT among them.
    const auto ok = trsm_shape(32, 128, 4096, 64);
    EXPECT_TRUE(TrsmTable::supports(kCta, ok));

    auto cpu = ok;  cpu.is_gpu = false;
    EXPECT_FALSE(TrsmTable::supports(kCta, cpu));

    auto het = ok;  het.heterogeneous_batch = true;
    EXPECT_FALSE(TrsmTable::supports(kCta, het));

    auto empty_tri = ok;  empty_tri.k = 0;
    EXPECT_FALSE(TrsmTable::supports(kCta, empty_tri));

    auto over = ok;  over.k = 65;   // one past the capacity
    EXPECT_FALSE(TrsmTable::supports(kCta, over));
    EXPECT_TRUE(TrsmTable::supports(kBlocked, over))
        << "the blocked driver splits the order itself, so the cap does not apply to it";

    // ...but only when it exists. A table that advertises a tier the build does
    // not contain hands the vendor-free caller a route nothing can service.
    auto no_v2 = over;  no_v2.blocked_available = false;
    EXPECT_FALSE(TrsmTable::supports(kBlocked, no_v2));

    // A tiny batch is slow, not wrong: it must stay SUPPORTED.
    auto tiny_batch = ok;  tiny_batch.batch = 1;
    EXPECT_TRUE(TrsmTable::supports(kCta, tiny_batch))
        << "batch size is a speed question; putting it in supports() breaks vendor-free trsm";
}

TEST(RouteTrsm, RhsCountFollowsSide) {
    const auto left  = trsm_shape(32, 128, 16, 64, Side::Left);
    const auto right = trsm_shape(32, 128, 16, 64, Side::Right);
    EXPECT_EQ(left.tri_order(), 32);
    EXPECT_EQ(right.tri_order(), 32);
    EXPECT_EQ(left.rhs_count(), 128) << "Side::Left  -> q = B.cols()";
    EXPECT_EQ(right.rhs_count(), 128) << "Side::Right -> q = B.rows()";
}

// ---------------------------------------------------------------------------
// POTRF's table. Same split as RouteTrsm above: a batch threshold in supports()
// would remove potrf's vendor-free route entirely, and a forced route bypasses
// preferred() but NEVER supports(). evidence: docs/perf/potrf.md
// ---------------------------------------------------------------------------
namespace {

// PERMISSIVE DEFAULTS, one hostile field per case: with cta_max_n at 0 or has_sg32
// false, every "supports() is false" case below would pass for the wrong reason.
PotrfShape potrf_shape(int64_t order, int64_t batch, int cta_max,
                       Uplo uplo = Uplo::Lower) {
    PotrfShape s;
    s.op = Op::potrf;
    s.scalar = ScalarKind::F32;
    // AUTO, deliberately: resolve_potrf_route is the INSTRUMENTED entry point, so
    // every shape built here lands in the coverage table. The real builder sets
    // s.backend = B, so AUTO keeps a synthetic row distinguishable from a real call.
    s.backend = Backend::AUTO;
    s.m = order;                 // square: m == n == k == the order
    s.n = order;
    s.k = order;
    s.batch = batch;
    s.uplo = uplo;
    s.is_gpu = true;
    s.has_sg32 = true;
    s.cta_max_n = cta_max;
    s.blocked_available = (cta_max > 0);
    return s;
}

using PotrfTable = RouteTable<Op::potrf, float>;
constexpr Route kPotrfCta{Origin::Native, Algorithm::CTA};
constexpr Route kPotrfBlocked{Origin::Native, Algorithm::Blocked};
constexpr Route kPotrfNativeBare{Origin::Native, Algorithm::Auto};
constexpr Route kPotrfAuto{Origin::Auto, Algorithm::Auto};

} // namespace

TEST(RoutePotrf, SupportedButNotPreferredIsTheWholePoint) {
    // 155 is the measured float CTA fit ceiling, and batch=1 is exactly the shape a
    // spec-faithful supports() would have made UNSUPPORTED.
    // evidence: docs/perf/potrf.md#the-slm-budget-and-the-fit-ceilings
    const auto s = potrf_shape(/*order=*/128, /*batch=*/1, /*cta_max=*/155);

    EXPECT_TRUE(PotrfTable::supports(kPotrfCta, s))
        << "batch size is a speed question; it must not gate CORRECTNESS";
    EXPECT_FALSE(PotrfTable::preferred(kPotrfCta, s))
        << "nothing about potrf has been measured yet";
    EXPECT_TRUE(is_native(resolve_potrf_route<float>(kPotrfAuto, s,
                                                     /*vendor_available=*/false)))
        << "un-preferred must never mean unroutable when there is no vendor";
    EXPECT_TRUE(is_vendor(resolve_potrf_route<float>(kPotrfAuto, s,
                                                     /*vendor_available=*/true)))
        << "and with a vendor present it must take it -- WP4 step 0.7 is a "
           "zero-behaviour-change gate";
}

TEST(RoutePotrf, VendorFreeFallbackPicksTheNativeRoute) {
    // Below the CTA capacity -> CTA. Above it -> Blocked. Never "no route".
    const auto small = potrf_shape(/*order=*/64,  /*batch=*/256, /*cta_max=*/155);
    const auto big   = potrf_shape(/*order=*/512, /*batch=*/64,  /*cta_max=*/155);

    const Route rs = resolve_potrf_route<float>(kPotrfAuto, small,
                                                /*vendor_available=*/false);
    EXPECT_TRUE(is_native(rs));
    EXPECT_EQ(rs.algo, Algorithm::CTA);

    const Route rb = resolve_potrf_route<float>(kPotrfAuto, big,
                                                /*vendor_available=*/false);
    EXPECT_TRUE(is_native(rb));
    EXPECT_EQ(rb.algo, Algorithm::Blocked)
        << "an order above the SLM capacity must fall to the blocked driver, "
           "not vanish";

    // A BARE ORIGIN must resolve to a specific algorithm: potrf has two native
    // routes, so {Native, Auto} names neither and no dispatch tail can map it.
    const Route rbare = resolve_potrf_route<float>(kPotrfNativeBare, small,
                                                   /*vendor_available=*/true);
    EXPECT_EQ(rbare.origin, Origin::Native);
    EXPECT_EQ(rbare.algo, Algorithm::CTA);
}

TEST(RoutePotrf, PreferredIsFalseEverywhere) {
    // preferred() is all-false for potrf, so Origin::Auto takes the vendor for every
    // shape. Replace this with clauses citing cells when a grid is measured.
    for (int64_t order : {1, 8, 63, 64, 77, 109, 155, 156, 512, 4096}) {
        for (int64_t batch : {1, 8, 128, 2048}) {
            for (Uplo up : {Uplo::Lower, Uplo::Upper}) {
                const auto s = potrf_shape(order, batch, 155, up);
                EXPECT_FALSE(PotrfTable::preferred(kPotrfCta, s));
                EXPECT_FALSE(PotrfTable::preferred(kPotrfBlocked, s));
                EXPECT_FALSE(PotrfTable::preferred(Route{Origin::Vendor, Algorithm::Auto}, s))
                    << "the vendor is where the walk ENDS, never itself preferred";
                EXPECT_TRUE(is_vendor(resolve_potrf_route<float>(kPotrfAuto, s, true)))
                    << "order " << order << " batch " << batch;
            }
        }
    }
    // ...and the other three scalar types, spelled out: preferred() reads the
    // table's T, so a loop varying s.scalar would test float three times.
    const auto s = potrf_shape(64, 512, 155);
    EXPECT_FALSE((RouteTable<Op::potrf, double>::preferred(kPotrfCta, s)));
    EXPECT_FALSE((RouteTable<Op::potrf, std::complex<float>>::preferred(kPotrfCta, s)));
    EXPECT_FALSE((RouteTable<Op::potrf, std::complex<double>>::preferred(kPotrfCta, s)));
}

TEST(RoutePotrf, CorrectnessGatesAreNotSpeedGates) {
    const auto ok = potrf_shape(/*order=*/64, /*batch=*/256, /*cta_max=*/155);
    ASSERT_TRUE(PotrfTable::supports(kPotrfCta, ok))
        << "guard: the permissive shape must be supported, or every EXPECT_FALSE "
           "below passes for the wrong reason";

    // Each false below is "would compute a WRONG ANSWER or could not launch".
    auto nonsquare = ok;  nonsquare.n = 65;     // m != n
    EXPECT_FALSE(PotrfTable::supports(kPotrfCta, nonsquare))
        << "there is no Cholesky factor of a non-square view";

    auto cpu = ok;  cpu.is_gpu = false;
    EXPECT_FALSE(PotrfTable::supports(kPotrfCta, cpu));

    auto het = ok;  het.heterogeneous_batch = true;
    EXPECT_FALSE(PotrfTable::supports(kPotrfCta, het))
        << "one launch, one (order, ld, stride) tuple, no batch walker";

    auto empty = ok;  empty.k = 0;  empty.m = 0;  empty.n = 0;
    EXPECT_FALSE(PotrfTable::supports(kPotrfCta, empty));

    auto no_batch = ok;  no_batch.batch = 0;
    EXPECT_FALSE(PotrfTable::supports(kPotrfCta, no_batch));

    auto over = ok;  over.k = 156;  over.m = 156;  over.n = 156;
    EXPECT_FALSE(PotrfTable::supports(kPotrfCta, over))
        << "156 is one past the measured float SLM fit ceiling of 155";
    EXPECT_TRUE(PotrfTable::supports(kPotrfBlocked, over))
        << "the blocked driver splits the order itself, so the cap is not its cap";

    // ...but only when it exists.
    auto no_blocked = over;  no_blocked.blocked_available = false;
    EXPECT_FALSE(PotrfTable::supports(kPotrfBlocked, no_blocked));

    // THE THREE THINGS THAT ARE *NOT* CORRECTNESS GATES; each must stay SUPPORTED.
    auto tiny_batch = ok;  tiny_batch.batch = 1;
    EXPECT_TRUE(PotrfTable::supports(kPotrfCta, tiny_batch))
        << "spec:559's kPotrfCtaMinBatch belongs in preferred()";
    auto huge_batch = ok;  huge_batch.batch = 1 << 20;
    EXPECT_TRUE(PotrfTable::supports(kPotrfCta, huge_batch));
    const auto tiny_order = potrf_shape(1, 256, 155);
    EXPECT_TRUE(PotrfTable::supports(kPotrfBlocked, tiny_order))
        << "spec:567's `n <= cta_max -> unsupported` would make a forced "
           "`blocked` at small n silently measure the VENDOR";
}

TEST(RoutePotrf, Sg32GatesBothNativeArms) {
    // A device whose sub_group_sizes lack 32 REJECTS the launch of a kernel carrying
    // [[sycl::reqd_sub_group_size(32)]], and the blocked driver's diagonal leaf IS
    // that same device function, so one missing capability must close BOTH arms. The
    // order is above the CTA ceiling so the blocked arm is the one under test.
    auto s = potrf_shape(/*order=*/200, /*batch=*/256, /*cta_max=*/155);
    ASSERT_TRUE(PotrfTable::supports(kPotrfBlocked, s))
        << "guard: the blocked arm must be OPEN before we close it, or the "
           "next assertion cannot fail";

    s.has_sg32 = false;
    EXPECT_FALSE(PotrfTable::supports(kPotrfBlocked, s))
        << "the blocked driver's leaf is the same reqd_sub_group_size(32) "
           "device function";

    auto small = potrf_shape(/*order=*/64, /*batch=*/256, /*cta_max=*/155);
    ASSERT_TRUE(PotrfTable::supports(kPotrfCta, small));
    small.has_sg32 = false;
    EXPECT_FALSE(PotrfTable::supports(kPotrfCta, small));

    // And a vendor-free build must then say "needs a vendor" rather than
    // handing back a route whose launch the device would reject.
    EXPECT_TRUE(is_vendor(resolve_potrf_route<float>(kPotrfAuto, small, false)));
}

TEST(RoutePotrf, UpperIsUnsupportedByTheBlockedDriver) {
    // Uplo::Upper on the blocked arm is a CORRECTNESS gate: the driver implements
    // the Lower recurrence only and would overwrite the wrong triangle. Contrast
    // syev, whose blocked arms accept Upper because they MIRROR first.
    const auto lower = potrf_shape(/*order=*/512, /*batch=*/64, /*cta_max=*/155,
                                   Uplo::Lower);
    const auto upper = potrf_shape(/*order=*/512, /*batch=*/64, /*cta_max=*/155,
                                   Uplo::Upper);
    ASSERT_TRUE(PotrfTable::supports(kPotrfBlocked, lower));
    EXPECT_FALSE(PotrfTable::supports(kPotrfBlocked, upper));

    EXPECT_TRUE(is_vendor(resolve_potrf_route<float>(kPotrfAuto, upper,
                                                     /*vendor_available=*/false)))
        << "no native route can serve it; the honest answer is 'needs a vendor'";
}

TEST(RoutePotrf, AbsentKernelIsUnsupportedRatherThanSelectable) {
    // cta_max_n == 0 / blocked_available == false is what a build WITHOUT the kernel
    // reports. Both native routes must then be UNSUPPORTED, so an absent capability
    // can never select a launch that is not there.
    const auto s = potrf_shape(/*order=*/64, /*batch=*/256, /*cta_max=*/0);
    EXPECT_FALSE(PotrfTable::supports(kPotrfCta, s));
    EXPECT_FALSE(PotrfTable::supports(kPotrfBlocked, s));
    EXPECT_TRUE(is_vendor(resolve_potrf_route<float>(kPotrfAuto, s, true)));
    EXPECT_TRUE(is_vendor(resolve_potrf_route<float>(kPotrfAuto, s, false)))
        << "vendor-free with nothing supported must say 'needs a vendor', not "
           "invent a native route";

    // AND FORCING MUST NOT ESCAPE IT: a forced route is gated on supports() and
    // falls through to automatic(), so a green forced-route test proves nothing.
    EXPECT_TRUE(is_vendor(resolve_potrf_route<float>(kPotrfCta, s, true)));
    EXPECT_TRUE(is_vendor(resolve_potrf_route<float>(kPotrfNativeBare, s, true)));
}

TEST(RoutePotrf, BatchlasPotrfRouteIsActuallyRead) {
    // No registry entry is needed: parse_route_env synthesises
    // "BATCHLAS_" + op_env_stem(op) + "_ROUTE".
    ClearRouteEnv clear(Op::potrf);

    EXPECT_EQ(op_env_stem(Op::potrf), "POTRF");
    EXPECT_TRUE(std::string(legacy_variable_for(Op::potrf)).empty())
        << "no legacy potrf variable ever shipped; a case in legacy_variable_for "
           "would INVENT one";

    {
        const auto unset = parse_route_env(Op::potrf);
        EXPECT_FALSE(unset.found);
        EXPECT_FALSE(unset.unparsed);
        EXPECT_EQ(legacy_unset_default(Op::potrf).origin, Origin::Auto);
    }
    {
        ScopedEnv e("BATCHLAS_POTRF_ROUTE", "cta");
        const auto p = parse_route_env(Op::potrf);
        ASSERT_TRUE(p.found) << "BATCHLAS_POTRF_ROUTE was not read at all";
        EXPECT_EQ(p.route, (Route{Origin::Native, Algorithm::CTA}))
            << "a bare algorithm implies Origin::Native";
        EXPECT_EQ(p.source.variable, "BATCHLAS_POTRF_ROUTE");
        EXPECT_FALSE(p.source.legacy);
    }
    {
        ScopedEnv e("BATCHLAS_POTRF_ROUTE", "vendor");
        const auto p = parse_route_env(Op::potrf);
        ASSERT_TRUE(p.found);
        EXPECT_EQ(p.route, (Route{Origin::Vendor, Algorithm::Auto}));
    }
    {
        ScopedEnv e("BATCHLAS_POTRF_ROUTE", "native:blocked");
        const auto p = parse_route_env(Op::potrf);
        ASSERT_TRUE(p.found);
        EXPECT_EQ(p.route, (Route{Origin::Native, Algorithm::Blocked}));
    }
    {
        ScopedEnv e("BATCHLAS_POTRF_ROUTE", "not-a-route");
        const auto p = parse_route_env(Op::potrf);
        EXPECT_FALSE(p.found);
        EXPECT_TRUE(p.unparsed) << "a typo must be reported, not silently Auto";
    }
}

// ---------------------------------------------------------------------------
// GEQRF's table. Same split as the RoutePotrf block, plus two gates with no potrf
// analogue: the CTA capacity is an AREA and a height, and copying potrf's
// `m == n` gate here would strip geqrf of rectangular A, which is the point of
// the op. evidence: docs/perf/qr.md
// ---------------------------------------------------------------------------
namespace {

// PERMISSIVE DEFAULTS, one hostile field per case: with the capacities at 0 or
// has_sg32 false, every "supports() is false" case below passes for the wrong reason.
GeqrfShape geqrf_shape(int64_t rows, int64_t cols, int64_t batch,
                       int cta_max_m, int64_t cta_max_elems) {
    GeqrfShape s;
    s.op = Op::geqrf;
    s.scalar = ScalarKind::F32;
    // AUTO, deliberately -- the same reason as potrf_shape's: it keeps a synthetic
    // unit-test coverage row distinguishable from one a library call produced.
    s.backend = Backend::AUTO;
    s.m = rows;
    s.n = cols;
    s.k = rows < cols ? rows : cols;   // the REFLECTOR COUNT, not an order
    s.batch = batch;
    s.is_gpu = true;
    s.has_sg32 = true;
    s.cta_max_m = cta_max_m;
    s.cta_max_elems = cta_max_elems;
    s.blocked_available = (cta_max_m > 0 && cta_max_elems > 0);
    return s;
}

using GeqrfTable = RouteTable<Op::geqrf, float>;
constexpr Route kGeqrfCta{Origin::Native, Algorithm::CTA};
constexpr Route kGeqrfBlocked{Origin::Native, Algorithm::Blocked};
constexpr Route kGeqrfNativeBare{Origin::Native, Algorithm::Auto};
constexpr Route kGeqrfAuto{Origin::Auto, Algorithm::Auto};

} // namespace

TEST(RouteGeqrf, VendorFreeFallbackHandsOverTheNativeRoute) {
    // THE TEST THAT FAILS IF A SPEED THRESHOLD EVER LANDS IN supports(): the
    // vendor-free walk tests supports() alone, so such a gate removes the route.
    const auto s = geqrf_shape(/*rows=*/64, /*cols=*/16, /*batch=*/1,
                               /*cta_max_m=*/256, /*cta_max_elems=*/8192);

    EXPECT_TRUE(GeqrfTable::supports(kGeqrfCta, s))
        << "batch size and panel size are speed questions; neither may gate "
           "CORRECTNESS";
    EXPECT_FALSE(GeqrfTable::preferred(kGeqrfCta, s))
        << "nothing native about geqrf has been measured -- there is no kernel";

    EXPECT_TRUE(is_native(resolve_geqrf_route<float>(kGeqrfAuto, s,
                                                     /*vendor_available=*/false)))
        << "un-preferred must never mean unroutable when there is no vendor";
    EXPECT_TRUE(is_vendor(resolve_geqrf_route<float>(kGeqrfAuto, s,
                                                     /*vendor_available=*/true)))
        << "and with a vendor present it must take it -- the WP5 scaffolding "
           "gate is zero behaviour change";
}

TEST(RouteGeqrf, RectangularIsSupportedAndSquarenessIsNotAGate) {
    // The most likely wrong edit here is copying potrf's `if (s.m != s.n)`: a tall
    // panel is what band_reduction.cc and sytrd_sy2sb.cc actually issue.
    const auto tall   = geqrf_shape(1024, 32, 128, 2048, 1 << 20);
    const auto square = geqrf_shape(64, 64, 128, 2048, 1 << 20);
    EXPECT_TRUE(GeqrfTable::supports(kGeqrfCta, tall))
        << "rectangular A is the entire point of geqrf (options.hh:727-730)";
    EXPECT_TRUE(GeqrfTable::supports(kGeqrfCta, square));

    // ...and k is the REFLECTOR COUNT, not either extent. A predicate that read
    // s.n where it meant min(m,n) would silently disagree with tau's length.
    EXPECT_EQ(tall.reflectors(), 32);
    EXPECT_EQ(square.reflectors(), 64);
    EXPECT_EQ(tall.rows(), 1024);
    EXPECT_EQ(tall.cols(), 32);
}

TEST(RouteGeqrf, WideIsUnsupportedByEveryNativeArm) {
    // m < n IS a correctness gate: both drivers are panel-oriented right-looking
    // schedules over columns, and a wide view walks the trailing update past the
    // bottom of the panel.
    const auto wide = geqrf_shape(/*rows=*/32, /*cols=*/1024, /*batch=*/128,
                                  /*cta_max_m=*/2048, /*cta_max_elems=*/1 << 20);
    EXPECT_FALSE(GeqrfTable::supports(kGeqrfCta, wide));
    EXPECT_FALSE(GeqrfTable::supports(kGeqrfBlocked, wide));
    EXPECT_TRUE(is_vendor(resolve_geqrf_route<float>(kGeqrfAuto, wide,
                                                     /*vendor_available=*/false)))
        << "no native route can serve it; the honest answer is 'needs a vendor'";

    // GUARD AGAINST VACUITY: the same extents the other way round must be
    // supported, or the assertions above pass for the wrong reason.
    const auto tall = geqrf_shape(1024, 32, 128, 2048, 1 << 20);
    ASSERT_TRUE(GeqrfTable::supports(kGeqrfCta, tall));
}

TEST(RouteGeqrf, CtaCapacityIsAnAreaAndAHeightNotTwoExtentBounds) {
    // The CTA tile holds the whole m x n panel, so the fit is governed by m*n; a
    // table checking only per-extent ceilings would accept an unlaunchable panel.
    // cta_max_m = 256, cta_max_elems = 8192.
    const auto fits = geqrf_shape(/*rows=*/256, /*cols=*/32, /*batch=*/64, 256, 8192);
    ASSERT_TRUE(GeqrfTable::supports(kGeqrfCta, fits))
        << "guard: 256*32 == 8192 is exactly the area budget";

    // Within BOTH per-extent bounds, over the AREA.
    auto over_area = geqrf_shape(/*rows=*/256, /*cols=*/64, /*batch=*/64, 256, 8192);
    EXPECT_FALSE(GeqrfTable::supports(kGeqrfCta, over_area))
        << "256*64 == 16384 does not fit an 8192-scalar tile, even though "
           "256 <= cta_max_m";

    // Within the AREA, over the HEIGHT.
    auto over_height = geqrf_shape(/*rows=*/512, /*cols=*/8, /*batch=*/64, 256, 8192);
    EXPECT_FALSE(GeqrfTable::supports(kGeqrfCta, over_height))
        << "512*8 == 4096 fits the tile, but one Householder vector spans 512 "
           "rows and the reduction has its own ceiling";

    // The BLOCKED arm inherits the PRESENCE of the leaf but not its capacity -- it
    // splits the panel itself.
    EXPECT_TRUE(GeqrfTable::supports(kGeqrfBlocked, over_area));
    EXPECT_TRUE(GeqrfTable::supports(kGeqrfBlocked, over_height));

    // ...but only when it exists.
    auto no_blocked = over_area;
    no_blocked.blocked_available = false;
    EXPECT_FALSE(GeqrfTable::supports(kGeqrfBlocked, no_blocked));
}

TEST(RouteGeqrf, CorrectnessGatesAreNotSpeedGates) {
    const auto ok = geqrf_shape(/*rows=*/128, /*cols=*/64, /*batch=*/256, 256, 8192);
    ASSERT_TRUE(GeqrfTable::supports(kGeqrfCta, ok))
        << "guard: the permissive shape must be supported, or every EXPECT_FALSE "
           "below passes for the wrong reason";

    auto cpu = ok;  cpu.is_gpu = false;
    EXPECT_FALSE(GeqrfTable::supports(kGeqrfCta, cpu));
    EXPECT_FALSE(GeqrfTable::supports(kGeqrfBlocked, cpu));

    auto het = ok;  het.heterogeneous_batch = true;
    EXPECT_FALSE(GeqrfTable::supports(kGeqrfCta, het))
        << "one launch, one (m, n, ld, stride) tuple, no batch walker -- and "
           "netlib's geqrf hoists m and n outside its loop too "
           "(netlib_lapack.cc:1406-1417), so nothing in this tree serves it";
    EXPECT_FALSE(GeqrfTable::supports(kGeqrfBlocked, het));

    auto empty_cols = ok;  empty_cols.n = 0;  empty_cols.k = 0;
    EXPECT_FALSE(GeqrfTable::supports(kGeqrfCta, empty_cols));

    auto empty_rows = ok;  empty_rows.m = 0;  empty_rows.k = 0;
    EXPECT_FALSE(GeqrfTable::supports(kGeqrfCta, empty_rows));

    auto no_batch = ok;  no_batch.batch = 0;
    EXPECT_FALSE(GeqrfTable::supports(kGeqrfCta, no_batch));

    // THE THINGS THAT ARE *NOT* CORRECTNESS GATES; each must stay SUPPORTED.
    auto tiny_batch = ok;  tiny_batch.batch = 1;
    EXPECT_TRUE(GeqrfTable::supports(kGeqrfCta, tiny_batch))
        << "a minimum-batch threshold belongs in preferred()";
    auto huge_batch = ok;  huge_batch.batch = 1 << 20;
    EXPECT_TRUE(GeqrfTable::supports(kGeqrfCta, huge_batch));
    const auto tiny = geqrf_shape(1, 1, 256, 256, 8192);
    EXPECT_TRUE(GeqrfTable::supports(kGeqrfBlocked, tiny))
        << "`the panel is small so blocked should be false` is a FIT judgement "
           "between two native routes; with it here a forced `blocked` at small "
           "n silently measures the VENDOR (route_resolve.hh:101, :111)";
}

TEST(RouteGeqrf, Sg32GatesBothNativeArms) {
    // The blocked driver's panel leaf IS the reqd_sub_group_size(32) device function,
    // so one missing capability must close BOTH arms. The panel is above the CTA
    // area so the blocked arm is the one under test.
    auto big = geqrf_shape(/*rows=*/1024, /*cols=*/1024, /*batch=*/64, 256, 8192);
    ASSERT_FALSE(GeqrfTable::supports(kGeqrfCta, big))
        << "guard: this panel must NOT fit the CTA tile, or the next assertions "
           "test the wrong arm";
    ASSERT_TRUE(GeqrfTable::supports(kGeqrfBlocked, big))
        << "guard: the blocked arm must be OPEN before we close it";

    big.has_sg32 = false;
    EXPECT_FALSE(GeqrfTable::supports(kGeqrfBlocked, big));

    auto small = geqrf_shape(/*rows=*/128, /*cols=*/64, /*batch=*/64, 256, 8192);
    ASSERT_TRUE(GeqrfTable::supports(kGeqrfCta, small));
    small.has_sg32 = false;
    EXPECT_FALSE(GeqrfTable::supports(kGeqrfCta, small));

    EXPECT_TRUE(is_vendor(resolve_geqrf_route<float>(kGeqrfAuto, small, false)));
}

TEST(RouteGeqrf, PreferredIsFalseEverywhere) {
    // preferred() is all-false for geqrf, so Origin::Auto takes the vendor for every
    // shape. Replace with clauses citing cells when a measured window lands.
    for (int64_t rows : {1, 32, 128, 512, 1024, 4096}) {
        for (int64_t cols : {1, 16, 32, 128, 512}) {
            if (cols > rows) continue;
            for (int64_t batch : {1, 8, 128, 2048}) {
                const auto s = geqrf_shape(rows, cols, batch, 4096, 1 << 24);
                EXPECT_FALSE(GeqrfTable::preferred(kGeqrfCta, s));
                EXPECT_FALSE(GeqrfTable::preferred(kGeqrfBlocked, s));
                EXPECT_FALSE(GeqrfTable::preferred(Route{Origin::Vendor, Algorithm::Auto}, s))
                    << "the vendor is where the walk ENDS, never itself preferred";
                EXPECT_TRUE(is_vendor(resolve_geqrf_route<float>(kGeqrfAuto, s, true)))
                    << "rows " << rows << " cols " << cols << " batch " << batch;
            }
        }
    }
    // ...and the other three scalar types, spelled out: preferred() reads the
    // table's T, so a loop varying s.scalar would test float three times.
    const auto s = geqrf_shape(256, 64, 512, 4096, 1 << 24);
    EXPECT_FALSE((RouteTable<Op::geqrf, double>::preferred(kGeqrfCta, s)));
    EXPECT_FALSE((RouteTable<Op::geqrf, std::complex<float>>::preferred(kGeqrfCta, s)));
    EXPECT_FALSE((RouteTable<Op::geqrf, std::complex<double>>::preferred(kGeqrfCta, s)));
    EXPECT_FALSE((RouteTable<Op::geqrf, double>::preferred(kGeqrfBlocked, s)));
    EXPECT_FALSE((RouteTable<Op::geqrf, std::complex<float>>::preferred(kGeqrfBlocked, s)));
    EXPECT_FALSE((RouteTable<Op::geqrf, std::complex<double>>::preferred(kGeqrfBlocked, s)));
}

TEST(RouteGeqrf, BareOriginResolvesToASpecificAlgorithm) {
    // geqrf has TWO native routes, so {Native, Auto} names neither. Below the CTA
    // capacity -> CTA; above it -> Blocked. Never "no route".
    const auto small = geqrf_shape(128, 64, 256, 256, 8192);
    const auto big   = geqrf_shape(1024, 1024, 64, 256, 8192);

    const Route rs = resolve_geqrf_route<float>(kGeqrfNativeBare, small,
                                                /*vendor_available=*/true);
    EXPECT_EQ(rs.origin, Origin::Native);
    EXPECT_EQ(rs.algo, Algorithm::CTA);

    const Route rb = resolve_geqrf_route<float>(kGeqrfAuto, big,
                                                /*vendor_available=*/false);
    EXPECT_TRUE(is_native(rb));
    EXPECT_EQ(rb.algo, Algorithm::Blocked)
        << "a panel above the tile capacity must fall to the blocked driver, "
           "not vanish";
}

TEST(RouteGeqrf, AbsentKernelIsUnsupportedRatherThanSelectable) {
    // Zero capacity / blocked_available == false is what a build without the kernels
    // reports: both native routes must then be UNSUPPORTED.
    const auto s = geqrf_shape(/*rows=*/128, /*cols=*/64, /*batch=*/256,
                               /*cta_max_m=*/0, /*cta_max_elems=*/0);
    EXPECT_FALSE(GeqrfTable::supports(kGeqrfCta, s));
    EXPECT_FALSE(GeqrfTable::supports(kGeqrfBlocked, s));
    EXPECT_TRUE(is_vendor(resolve_geqrf_route<float>(kGeqrfAuto, s, true)));
    EXPECT_TRUE(is_vendor(resolve_geqrf_route<float>(kGeqrfAuto, s, false)))
        << "vendor-free with nothing supported must say 'needs a vendor', not "
           "invent a native route";

    // AND FORCING MUST NOT ESCAPE IT: a forced route is gated on supports(), so a
    // green forced-route test is not evidence that a native kernel ran.
    EXPECT_TRUE(is_vendor(resolve_geqrf_route<float>(kGeqrfCta, s, true)));
    EXPECT_TRUE(is_vendor(resolve_geqrf_route<float>(kGeqrfBlocked, s, true)));
    EXPECT_TRUE(is_vendor(resolve_geqrf_route<float>(kGeqrfNativeBare, s, true)));
    EXPECT_TRUE(is_vendor(resolve_geqrf_route<float>(kGeqrfCta, s, false)));

    // Half a capability is still absent: a build that reported a height but no
    // area (or the reverse) must not select either arm.
    auto half_a = geqrf_shape(128, 64, 256, /*cta_max_m=*/256, /*cta_max_elems=*/0);
    EXPECT_FALSE(GeqrfTable::supports(kGeqrfCta, half_a));
    auto half_b = geqrf_shape(128, 64, 256, /*cta_max_m=*/0, /*cta_max_elems=*/8192);
    half_b.blocked_available = true;
    EXPECT_FALSE(GeqrfTable::supports(kGeqrfCta, half_b));
    EXPECT_FALSE(GeqrfTable::supports(kGeqrfBlocked, half_b))
        << "the blocked driver's panel leaf IS the CTA device function, so it "
           "inherits the presence gate";
}

TEST(RouteGeqrf, BatchlasGeqrfRouteIsActuallyRead) {
    // The canonical spelling needs no registry entry -- parse_route_env synthesises
    // it -- but nothing had exercised that path for geqrf.
    ClearRouteEnv clear(Op::geqrf);

    EXPECT_EQ(op_env_stem(Op::geqrf), "GEQRF");
    EXPECT_TRUE(std::string(legacy_variable_for(Op::geqrf)).empty())
        << "no legacy geqrf variable ever shipped; a case in legacy_variable_for "
           "would INVENT a legacy spelling. Note that Op::ormqr DOES have one "
           "(route_env.hh:118) -- that is not a precedent for this op";

    {
        const auto unset = parse_route_env(Op::geqrf);
        EXPECT_FALSE(unset.found);
        EXPECT_FALSE(unset.unparsed);
        EXPECT_EQ(legacy_unset_default(Op::geqrf).origin, Origin::Auto);
    }
    {
        ScopedEnv e("BATCHLAS_GEQRF_ROUTE", "cta");
        const auto p = parse_route_env(Op::geqrf);
        ASSERT_TRUE(p.found) << "BATCHLAS_GEQRF_ROUTE was not read at all";
        EXPECT_EQ(p.route, (Route{Origin::Native, Algorithm::CTA}))
            << "a bare algorithm implies Origin::Native";
        EXPECT_EQ(p.source.variable, "BATCHLAS_GEQRF_ROUTE");
        EXPECT_FALSE(p.source.legacy);
    }
    {
        ScopedEnv e("BATCHLAS_GEQRF_ROUTE", "vendor");
        const auto p = parse_route_env(Op::geqrf);
        ASSERT_TRUE(p.found);
        EXPECT_EQ(p.route, (Route{Origin::Vendor, Algorithm::Auto}));
    }
    {
        ScopedEnv e("BATCHLAS_GEQRF_ROUTE", "native:blocked");
        const auto p = parse_route_env(Op::geqrf);
        ASSERT_TRUE(p.found);
        EXPECT_EQ(p.route, (Route{Origin::Native, Algorithm::Blocked}));
    }
    {
        // AN UNRECOGNISED VALUE IS SILENTLY {Auto, Auto}, WHICH IS THE VENDOR: a
        // "native" run that looks identical to the vendor probably IS the vendor.
        ScopedEnv e("BATCHLAS_GEQRF_ROUTE", "not-a-route");
        const auto p = parse_route_env(Op::geqrf);
        EXPECT_FALSE(p.found);
        EXPECT_TRUE(p.unparsed) << "a typo must be reported, not silently Auto";
    }
}

// ---------------------------------------------------------------------------
// ORGQR's table. orgqr ships as ORMQR APPLIED TO AN IDENTITY, so its supports()
// TRANSCRIBES RouteTable<Op::ormqr,T>'s gates plus its own -- that table is what
// serves the call, and omitting an inherited gate is the wrong-answer class.
// ---------------------------------------------------------------------------
namespace {

OrgqrShape orgqr_shape(int64_t rows, int64_t cols, int64_t batch,
                       bool blocked_available = true) {
    OrgqrShape s;
    s.op = Op::orgqr;
    s.scalar = ScalarKind::F32;
    s.backend = Backend::AUTO;   // same reason as potrf_shape / geqrf_shape
    s.m = rows;
    s.n = cols;
    s.k = rows < cols ? rows : cols;
    s.batch = batch;
    s.is_gpu = true;
    // The builder pins the apply at (Left, NoTrans); mirror it, or the inherited
    // complex-Trans case below would be testing a shape the builder never makes.
    s.side = Side::Left;
    s.transA = Transpose::NoTrans;
    s.blocked_available = blocked_available;
    return s;
}

using OrgqrTable = RouteTable<Op::orgqr, float>;
using OrgqrTableC = RouteTable<Op::orgqr, std::complex<float>>;
constexpr Route kOrgqrBlocked{Origin::Native, Algorithm::Blocked};
constexpr Route kOrgqrNativeBare{Origin::Native, Algorithm::Auto};
constexpr Route kOrgqrAuto{Origin::Auto, Algorithm::Auto};

} // namespace

TEST(RouteOrgqr, VendorFreeFallbackHandsOverTheNativeRoute) {
    // The same speed-threshold guard as geqrf's, and it matters more: the vendor
    // orgqr is a per-item loop, so a gate back to it also restores a workspace of
    // single_ws * batch.
    const auto s = orgqr_shape(/*rows=*/64, /*cols=*/64, /*batch=*/1);

    EXPECT_TRUE(OrgqrTable::supports(kOrgqrBlocked, s));
    EXPECT_FALSE(OrgqrTable::preferred(kOrgqrBlocked, s));
    EXPECT_TRUE(is_native(resolve_orgqr_route<float>(kOrgqrAuto, s,
                                                     /*vendor_available=*/false)));
    EXPECT_TRUE(is_vendor(resolve_orgqr_route<float>(kOrgqrAuto, s,
                                                     /*vendor_available=*/true)));
}

TEST(RouteOrgqr, PreferredIsFalseEverywhere) {
    // NOT `is_native(r) && supports(r, s)`: that spelling would make native the
    // default on every supported shape, and cfloat n=2048 is a measured loss.
    for (int64_t n : {1, 32, 64, 256, 1024, 2048}) {
        for (int64_t batch : {1, 8, 128, 2048}) {
            const auto s = orgqr_shape(n, n, batch);
            EXPECT_FALSE(OrgqrTable::preferred(kOrgqrBlocked, s));
            EXPECT_FALSE(OrgqrTable::preferred(Route{Origin::Vendor, Algorithm::Auto}, s))
                << "the vendor is where the walk ENDS, never itself preferred";
            EXPECT_TRUE(is_vendor(resolve_orgqr_route<float>(kOrgqrAuto, s, true)))
                << "n " << n << " batch " << batch;
        }
    }
    const auto s = orgqr_shape(256, 256, 512);
    EXPECT_FALSE((RouteTable<Op::orgqr, double>::preferred(kOrgqrBlocked, s)));
    EXPECT_FALSE((RouteTable<Op::orgqr, std::complex<float>>::preferred(kOrgqrBlocked, s)));
    EXPECT_FALSE((RouteTable<Op::orgqr, std::complex<double>>::preferred(kOrgqrBlocked, s)));
}

TEST(RouteOrgqr, CorrectnessGatesIncludeTheOnesInheritedFromOrmqr) {
    const auto ok = orgqr_shape(/*rows=*/256, /*cols=*/128, /*batch=*/256);
    ASSERT_TRUE(OrgqrTable::supports(kOrgqrBlocked, ok))
        << "guard: the permissive shape must be supported, or every EXPECT_FALSE "
           "below passes for the wrong reason";

    // INHERITED from route_ormqr.hh -- the native apply is GPU-only.
    auto cpu = ok;  cpu.is_gpu = false;
    EXPECT_FALSE(OrgqrTable::supports(kOrgqrBlocked, cpu));

    // INHERITED from route_ormqr.hh -- complex with a plain Trans. It cannot fire
    // through the builder, but the gate is transcribed rather than dropped.
    auto complex_trans = ok;  complex_trans.transA = Transpose::Trans;
    EXPECT_FALSE(OrgqrTableC::supports(kOrgqrBlocked, complex_trans))
        << "route_ormqr.hh:63-66's exclusion must be inherited, not silently "
           "dropped; cuSOLVER refuses the same combination (CUSOLVER error: 3)";
    EXPECT_TRUE(OrgqrTable::supports(kOrgqrBlocked, complex_trans))
        << "and it must apply to COMPLEX only -- a real scalar with Trans is "
           "served by the native ormqr and refused by cuSOLVER, the opposite "
           "asymmetry";

    // orgqr's OWN gates.
    auto wide = ok;  wide.n = 512;  // n > m: more orthonormal columns than rows
    EXPECT_FALSE(OrgqrTable::supports(kOrgqrBlocked, wide))
        << "Q's columns live in R^m; n > m asks for a basis that does not exist";

    auto het = ok;  het.heterogeneous_batch = true;
    EXPECT_FALSE(OrgqrTable::supports(kOrgqrBlocked, het))
        << "one identity, one apply, one (m, n, ld, stride) tuple -- and note "
           "RouteTable<Op::ormqr> has no such gate and its builder never sets "
           "the field, so ormqr's routing is blind to this today";

    auto empty = ok;  empty.n = 0;  empty.k = 0;
    EXPECT_FALSE(OrgqrTable::supports(kOrgqrBlocked, empty));
    auto no_batch = ok;  no_batch.batch = 0;
    EXPECT_FALSE(OrgqrTable::supports(kOrgqrBlocked, no_batch));

    // NOT correctness gates.
    auto tiny_batch = ok;  tiny_batch.batch = 1;
    EXPECT_TRUE(OrgqrTable::supports(kOrgqrBlocked, tiny_batch));
    auto huge = ok;  huge.m = 8192;  huge.n = 8192;  huge.k = 8192;
    EXPECT_TRUE(OrgqrTable::supports(kOrgqrBlocked, huge))
        << "n=2048 measuring slower than the vendor is a preferred() clause, "
           "never a supports() gate";
}

TEST(RouteOrgqr, AbsentDriverIsUnsupportedRatherThanSelectable) {
    // blocked_available is orgqr's OWN driver flag, not "is ormqr_blocked compiled";
    // answering with the latter hands back a route the facade cannot service.
    const auto s = orgqr_shape(/*rows=*/256, /*cols=*/256, /*batch=*/128,
                               /*blocked_available=*/false);
    EXPECT_FALSE(OrgqrTable::supports(kOrgqrBlocked, s));
    EXPECT_TRUE(is_vendor(resolve_orgqr_route<float>(kOrgqrAuto, s, true)));
    EXPECT_TRUE(is_vendor(resolve_orgqr_route<float>(kOrgqrAuto, s, false)));
    EXPECT_TRUE(is_vendor(resolve_orgqr_route<float>(kOrgqrBlocked, s, true)))
        << "forcing must not escape supports() (route_resolve.hh:101, :111)";
    EXPECT_TRUE(is_vendor(resolve_orgqr_route<float>(kOrgqrNativeBare, s, false)));

    // Guard against vacuity: with the driver present the same shape IS native
    // in a vendor-free build.
    const auto present = orgqr_shape(256, 256, 128, /*blocked_available=*/true);
    EXPECT_TRUE(is_native(resolve_orgqr_route<float>(kOrgqrAuto, present, false)));
}

TEST(RouteOrgqr, BareOriginResolvesToASpecificAlgorithm) {
    // orgqr has ONE native route, but {Native, Auto} must still come back as
    // {Native, Blocked} -- no dispatch tail can map an Auto algorithm to a kernel.
    // Note the departure from route_ormqr.hh, which accepts Auto inside supports().
    const auto s = orgqr_shape(256, 256, 128);
    const Route r = resolve_orgqr_route<float>(kOrgqrNativeBare, s,
                                               /*vendor_available=*/true);
    EXPECT_EQ(r.origin, Origin::Native);
    EXPECT_EQ(r.algo, Algorithm::Blocked);
    EXPECT_FALSE(OrgqrTable::supports(kOrgqrNativeBare, s))
        << "a bare {Native, Auto} names no algorithm and must not be reported "
           "as supported; the origin-restricted walk is what resolves it";
}

TEST(RouteOrgqr, BatchlasOrgqrRouteIsActuallyRead) {
    ClearRouteEnv clear(Op::orgqr);

    EXPECT_EQ(op_env_stem(Op::orgqr), "ORGQR");
    EXPECT_TRUE(std::string(legacy_variable_for(Op::orgqr)).empty())
        << "no legacy orgqr variable ever shipped; a case in legacy_variable_for "
           "would INVENT a legacy spelling";

    // AND THE OP IT DELEGATES TO IS PINNED BY A DIFFERENT VARIABLE: orgqr's native
    // arm re-enters routed ormqr, which reads its own canonical and legacy names.
    EXPECT_EQ(legacy_variable_for(Op::ormqr), "BATCHLAS_ORMQR_PROVIDER");

    {
        const auto unset = parse_route_env(Op::orgqr);
        EXPECT_FALSE(unset.found);
        EXPECT_FALSE(unset.unparsed);
        EXPECT_EQ(legacy_unset_default(Op::orgqr).origin, Origin::Auto);
    }
    {
        ScopedEnv e("BATCHLAS_ORGQR_ROUTE", "blocked");
        const auto p = parse_route_env(Op::orgqr);
        ASSERT_TRUE(p.found) << "BATCHLAS_ORGQR_ROUTE was not read at all";
        EXPECT_EQ(p.route, (Route{Origin::Native, Algorithm::Blocked}));
        EXPECT_EQ(p.source.variable, "BATCHLAS_ORGQR_ROUTE");
        EXPECT_FALSE(p.source.legacy);
    }
    {
        ScopedEnv e("BATCHLAS_ORGQR_ROUTE", "vendor");
        const auto p = parse_route_env(Op::orgqr);
        ASSERT_TRUE(p.found);
        EXPECT_EQ(p.route, (Route{Origin::Vendor, Algorithm::Auto}));
    }
    {
        ScopedEnv e("BATCHLAS_ORGQR_ROUTE", "not-a-route");
        const auto p = parse_route_env(Op::orgqr);
        EXPECT_FALSE(p.found);
        EXPECT_TRUE(p.unparsed) << "a typo must be reported, not silently Auto";
    }
}

// ---------------------------------------------------------------------------
// The LU family: getrf, getrs, getri. These cases are SYNTHETIC -- they call
// supports()/preferred() on hand-built shapes and never reach a kernel;
// tests/getrf_tests.cc is where the real device shapes are asserted. getrf and
// getri DO take potrf's `m == n` gate, unlike geqrf. evidence: docs/perf/lu.md
// ---------------------------------------------------------------------------
namespace {

// PERMISSIVE DEFAULTS, one hostile field per case: with cta_max_n at 0 or has_sg32
// false, every "supports() is false" case below passes for the wrong reason.
GetrfShape getrf_shape(int64_t order, int64_t batch, int cta_max_n) {
    GetrfShape s;
    s.op = Op::getrf;
    s.scalar = ScalarKind::F32;
    // AUTO, deliberately -- the same reason as potrf_shape's and geqrf_shape's: it
    // keeps a synthetic coverage row distinguishable from one a library call made.
    s.backend = Backend::AUTO;
    s.m = order;
    s.n = order;
    s.k = order;          // THE ORDER, potrf's mapping and not geqrf's
    s.batch = batch;
    s.is_gpu = true;
    s.has_sg32 = true;
    s.cta_max_n = cta_max_n;
    s.blocked_available = (cta_max_n > 0);
    return s;
}

// THE FUSED TIER'S TWO CAPACITIES, AND THEY DEFAULT TO PRESENT. A helper that left
// them at 0 makes RouteTable<getrs>::supports({Native, CTA}, s) false on every
// shape here, and every getrs assertion below then holds vacuously whatever the
// table says. The values are what this box reports for a 4-byte scalar;
// getrf_tests.cc is what asks the real builder on the real device whether it agrees.
constexpr int64_t kFusedMaxElemsF32 = 23264;
constexpr int64_t kFusedMaxNrhs     = 8;

GetrsShape getrs_shape(int64_t order, int64_t nrhs, int64_t batch,
                       bool blocked_available = true,
                       Transpose transA = Transpose::NoTrans,
                       int64_t fused_max_elems = kFusedMaxElemsF32,
                       int64_t fused_max_nrhs = kFusedMaxNrhs) {
    GetrsShape s;
    s.op = Op::getrs;
    s.scalar = ScalarKind::F32;
    s.backend = Backend::AUTO;
    s.m = order;
    s.n = nrhs;
    s.k = order;
    s.batch = batch;
    s.transA = transA;
    s.is_gpu = true;
    s.has_sg32 = true;
    s.blocked_available = blocked_available;
    s.fused_max_elems = fused_max_elems;
    s.fused_max_nrhs = fused_max_nrhs;
    return s;
}

GetriShape getri_shape(int64_t order, int64_t batch,
                       bool blocked_available = true) {
    GetriShape s;
    s.op = Op::getri;
    s.scalar = ScalarKind::F32;
    s.backend = Backend::AUTO;
    s.m = order;
    s.n = order;
    s.k = order;
    s.batch = batch;
    s.is_gpu = true;
    s.has_sg32 = true;
    s.blocked_available = blocked_available;
    return s;
}

// "Does this table DECLARE the optional third predicate?" -- the same detection
// route_resolve.hh performs. IT HAS TO BE A TEMPLATE: written against a concrete
// table the name lookup is a hard error rather than a substitution failure.
template <typename Tbl, typename Shape>
inline constexpr bool declares_native_tier_preferred =
    requires(Route r, const Shape& s) { Tbl::native_tier_preferred(r, s); };

using GetrfTable = RouteTable<Op::getrf, float>;
constexpr Route kGetrfCta{Origin::Native, Algorithm::CTA};
constexpr Route kGetrfBlocked{Origin::Native, Algorithm::Blocked};
constexpr Route kGetrfNativeBare{Origin::Native, Algorithm::Auto};
constexpr Route kGetrfAuto{Origin::Auto, Algorithm::Auto};

using GetrsTable = RouteTable<Op::getrs, float>;
constexpr Route kGetrsCta{Origin::Native, Algorithm::CTA};
constexpr Route kGetrsBlocked{Origin::Native, Algorithm::Blocked};
constexpr Route kGetrsNativeBare{Origin::Native, Algorithm::Auto};
constexpr Route kGetrsAuto{Origin::Auto, Algorithm::Auto};

using GetriTable = RouteTable<Op::getri, float>;
constexpr Route kGetriBlocked{Origin::Native, Algorithm::Blocked};
constexpr Route kGetriNativeBare{Origin::Native, Algorithm::Auto};
constexpr Route kGetriAuto{Origin::Auto, Algorithm::Auto};

constexpr Route kVendorAuto{Origin::Vendor, Algorithm::Auto};

// ---- THE OTHER THREE SCALAR TYPES, NAMED ONCE ------------------------------
// preferred() decides on the TABLE's T and NOT on s.scalar, so a per-type clause
// has its cfloat, double and cdouble boundaries checked by nothing unless the
// table is instantiated at those types.
using GetrfTableD  = RouteTable<Op::getrf, double>;
using GetrfTableCF = RouteTable<Op::getrf, std::complex<float>>;
using GetrfTableCD = RouteTable<Op::getrf, std::complex<double>>;
using GetrsTableD  = RouteTable<Op::getrs, double>;
using GetrsTableCF = RouteTable<Op::getrs, std::complex<float>>;
using GetrsTableCD = RouteTable<Op::getrs, std::complex<double>>;
using GetriTableD  = RouteTable<Op::getri, double>;
using GetriTableCF = RouteTable<Op::getri, std::complex<float>>;
using GetriTableCD = RouteTable<Op::getri, std::complex<double>>;

} // namespace

TEST(RouteGetrf, VendorFreeFallbackHandsOverTheNativeRoute) {
    // THE TEST THAT FAILS IF A SPEED THRESHOLD EVER LANDS IN supports(): the
    // vendor-free walk tests supports() ALONE. order=40 and batch=2 are
    // inverse_tests' own extents, so that suite closes only if they stay supported.
    const auto s = getrf_shape(/*order=*/40, /*batch=*/2, /*cta_max_n=*/128);

    EXPECT_TRUE(GetrfTable::supports(kGetrfCta, s))
        << "batch size and order are speed questions; neither may gate CORRECTNESS "
           "-- and these are inverse_tests' own extents";
    EXPECT_TRUE(GetrfTable::supports(kGetrfBlocked, s));
    EXPECT_FALSE(GetrfTable::preferred(kGetrfCta, s))
        << "getrf's preferred() is all-false BY DECISION, not by absence: both "
           "native arms exist and are measured, and the window is withheld "
           "because the crossover moves with batch as much as with order "
           "(docs/perf/lu.md#the-vendor-baseline-and-saturation). Flip this only together with "
           "a measured grid";

    EXPECT_TRUE(is_native(resolve_getrf_route<float>(kGetrfAuto, s,
                                                     /*vendor_available=*/false)))
        << "un-preferred must never mean unroutable when there is no vendor";
    EXPECT_TRUE(is_vendor(resolve_getrf_route<float>(kGetrfAuto, s,
                                                     /*vendor_available=*/true)))
        << "and with a vendor present it must take it -- the WP6 scaffolding gate "
           "is zero behaviour change";
}

TEST(RouteGetrf, SquarenessIsAGateHereAndDeliberatelyNotInGeqrf) {
    // getrf takes potrf's `m != n` line and geqrf refuses it; the two are one edit
    // apart, so both halves are pinned here. The justification is that BatchLAS's
    // public getrf is square -- widening it is an API change, not a routing one.
    auto wide = getrf_shape(/*order=*/64, /*batch=*/128, /*cta_max_n=*/256);
    wide.n = 1024;                    // m=64, n=1024, k=64
    EXPECT_FALSE(GetrfTable::supports(kGetrfCta, wide));
    EXPECT_FALSE(GetrfTable::supports(kGetrfBlocked, wide));

    auto tall = getrf_shape(/*order=*/1024, /*batch=*/128, /*cta_max_n=*/2048);
    tall.n = 32;                      // m=1024, n=32, k=1024
    EXPECT_FALSE(GetrfTable::supports(kGetrfCta, tall));
    EXPECT_FALSE(GetrfTable::supports(kGetrfBlocked, tall));

    // GUARD AGAINST VACUITY: the square shape at the same extents must be
    // supported, or both assertions above pass for the wrong reason.
    const auto square = getrf_shape(64, 128, 256);
    ASSERT_TRUE(GetrfTable::supports(kGetrfCta, square));

    // ...and the sibling table must still NOT have the gate: deleting geqrf's
    // rectangular support is the recorded wrong edit in the other direction.
    GeqrfShape g;
    g.op = Op::geqrf;
    g.scalar = ScalarKind::F32;
    g.backend = Backend::AUTO;
    g.m = 1024; g.n = 32; g.k = 32;
    g.batch = 128;
    g.is_gpu = true;
    g.has_sg32 = true;
    g.cta_max_m = 2048;
    g.cta_max_elems = 1 << 20;
    g.blocked_available = true;
    EXPECT_TRUE((RouteTable<Op::geqrf, float>::supports(
        Route{Origin::Native, Algorithm::CTA}, g)))
        << "rectangular A is the entire point of geqrf (options.hh:727-730); "
           "getrf's squareness gate must not migrate into it";
}

TEST(RouteGetrf, CtaCapacityIsTheOrderAndBlockedInheritsOnlyThePresence) {
    // The CTA tile holds the whole n x n matrix PLUS the pivot-search scratch, so
    // the capacity is a hard launch limit and not a tuning knob.
    const auto fits = getrf_shape(/*order=*/128, /*batch=*/64, /*cta_max_n=*/128);
    ASSERT_TRUE(GetrfTable::supports(kGetrfCta, fits))
        << "guard: order 128 is exactly the capacity";

    const auto over = getrf_shape(/*order=*/129, /*batch=*/64, /*cta_max_n=*/128);
    EXPECT_FALSE(GetrfTable::supports(kGetrfCta, over));

    // The BLOCKED arm inherits the PRESENCE of the leaf but not its capacity -- it
    // splits the matrix into panels the leaf can hold.
    EXPECT_TRUE(GetrfTable::supports(kGetrfBlocked, over));

    // ...but only when it exists.
    auto no_blocked = over;
    no_blocked.blocked_available = false;
    EXPECT_FALSE(GetrfTable::supports(kGetrfBlocked, no_blocked));

    // AND THE BLOCKED ARM CARRIES NO LOWER BOUND: "order <= the CTA capacity so
    // blocked should be false" is a fit judgement between two native routes, and in
    // supports() it makes a forced `blocked` fall through and measure the vendor.
    const auto tiny = getrf_shape(/*order=*/1, /*batch=*/256, /*cta_max_n=*/128);
    EXPECT_TRUE(GetrfTable::supports(kGetrfBlocked, tiny));
}

TEST(RouteGetrf, CorrectnessGatesAreNotSpeedGates) {
    const auto ok = getrf_shape(/*order=*/64, /*batch=*/256, /*cta_max_n=*/128);
    ASSERT_TRUE(GetrfTable::supports(kGetrfCta, ok))
        << "guard: the permissive shape must be supported, or every EXPECT_FALSE "
           "below passes for the wrong reason";

    auto cpu = ok;  cpu.is_gpu = false;
    EXPECT_FALSE(GetrfTable::supports(kGetrfCta, cpu));
    EXPECT_FALSE(GetrfTable::supports(kGetrfBlocked, cpu));

    auto het = ok;  het.heterogeneous_batch = true;
    EXPECT_FALSE(GetrfTable::supports(kGetrfCta, het))
        << "one launch, one (order, ld, stride) tuple, no batch walker -- and "
           "netlib's getrf hoists n outside its loop too (netlib_lapack.cc:1291), "
           "so nothing in this tree serves a heterogeneous LU";
    EXPECT_FALSE(GetrfTable::supports(kGetrfBlocked, het));

    auto empty = ok;  empty.m = 0; empty.n = 0; empty.k = 0;
    EXPECT_FALSE(GetrfTable::supports(kGetrfCta, empty));
    EXPECT_FALSE(GetrfTable::supports(kGetrfBlocked, empty));

    auto no_batch = ok;  no_batch.batch = 0;
    EXPECT_FALSE(GetrfTable::supports(kGetrfCta, no_batch));
    EXPECT_FALSE(GetrfTable::supports(kGetrfBlocked, no_batch));

    // THE THINGS THAT ARE *NOT* CORRECTNESS GATES; each must stay SUPPORTED.
    auto tiny_batch = ok;  tiny_batch.batch = 1;
    EXPECT_TRUE(GetrfTable::supports(kGetrfCta, tiny_batch))
        << "a minimum-batch threshold belongs in preferred()";
    auto two_batch = ok;  two_batch.batch = 2;
    EXPECT_TRUE(GetrfTable::supports(kGetrfCta, two_batch))
        << "inverse_tests runs at batch 2; a batch floor here keeps the one suite "
           "WP6 can close red however good the kernel is";
    auto huge_batch = ok;  huge_batch.batch = 1 << 20;
    EXPECT_TRUE(GetrfTable::supports(kGetrfCta, huge_batch));
}

TEST(RouteGetrf, Sg32GatesBothNativeArms) {
    // The blocked driver's diagonal-panel leaf IS the reqd_sub_group_size(32) device
    // function, so one missing capability must close BOTH arms. The order is above
    // the CTA capacity so the blocked arm is the one under test.
    auto big = getrf_shape(/*order=*/1024, /*batch=*/64, /*cta_max_n=*/128);
    ASSERT_FALSE(GetrfTable::supports(kGetrfCta, big))
        << "guard: this order must NOT fit the CTA tile, or the next assertions "
           "test the wrong arm";
    ASSERT_TRUE(GetrfTable::supports(kGetrfBlocked, big))
        << "guard: the blocked arm must be OPEN before we close it";

    big.has_sg32 = false;
    EXPECT_FALSE(GetrfTable::supports(kGetrfBlocked, big));

    auto small = getrf_shape(/*order=*/64, /*batch=*/64, /*cta_max_n=*/128);
    ASSERT_TRUE(GetrfTable::supports(kGetrfCta, small));
    small.has_sg32 = false;
    EXPECT_FALSE(GetrfTable::supports(kGetrfCta, small));

    EXPECT_TRUE(is_vendor(resolve_getrf_route<float>(kGetrfAuto, small, false)));
}

// ---------------------------------------------------------------------------
// THE PIVOT-FORMAT GATE, the one route test here with a BACKEND axis. The native
// kernels write PACKED 1-based int32 into the first half of the caller's int64
// pivot span; netlib writes genuine int64. On a GPU queue built with
// Backend::NETLIB the two arms silently disagree and getri returns wrong numbers
// with info == 0, so this is a CORRECTNESS gate and lives in supports().
// evidence: docs/perf/lu.md#correctness-findings
// ---------------------------------------------------------------------------
TEST(RouteLuPivotFormat, NetlibOnAGpuQueueIsNotANativeShape) {
    // --- getrf, both tiers ---------------------------------------------
    auto f = getrf_shape(/*order=*/40, /*batch=*/2, /*cta_max_n=*/128);
    f.backend = Backend::CUDA;
    ASSERT_TRUE(GetrfTable::supports(kGetrfCta, f))
        << "guard: this shape must be OPEN at a packed-int32 backend, or the "
           "NETLIB assertion below passes for the wrong reason";
    ASSERT_TRUE(GetrfTable::supports(kGetrfBlocked, f));

    f.backend = Backend::NETLIB;
    EXPECT_FALSE(GetrfTable::supports(kGetrfCta, f))
        << "the native kernel writes packed int32; netlib's getri/getrs read "
           "genuine int64 out of the same span";
    EXPECT_FALSE(GetrfTable::supports(kGetrfBlocked, f))
        << "both tiers write the same format, so one gate must close both";
    EXPECT_TRUE(GetrfTable::supports(kVendorAuto, f))
        << "the vendor arm is exactly what must still serve this configuration";

    // ROCm packs int32 like CUDA (rocsolver.cc:227), so the gate must NOT be an
    // allow-list of one backend.
    f.backend = Backend::ROCM;
    EXPECT_TRUE(GetrfTable::supports(kGetrfCta, f));

    // A forced route must be REFUSED, not silently honoured.
    f.backend = Backend::NETLIB;
    EXPECT_TRUE(is_vendor(resolve_getrf_route<float>(kGetrfBlocked, f, true)))
        << "route_resolve.hh:165 honours a forced route only if supports() says "
           "yes; this is the clause that makes the env var safe";

    // --- getrs ---------------------------------------------------------
    auto rs = getrs_shape(/*order=*/40, /*nrhs=*/3, /*batch=*/2);
    rs.backend = Backend::CUDA;
    ASSERT_TRUE(GetrsTable::supports(kGetrsBlocked, rs)) << "guard";
    rs.backend = Backend::NETLIB;
    EXPECT_FALSE(GetrsTable::supports(kGetrsBlocked, rs))
        << "the native getrs READS packed int32; a netlib getrf wrote int64";
    EXPECT_TRUE(is_vendor(resolve_getrs_route<float>(kGetrsBlocked, rs, true)));

    // --- getri ---------------------------------------------------------
    auto ri = getri_shape(/*order=*/40, /*batch=*/2);
    ri.backend = Backend::CUDA;
    ASSERT_TRUE(GetriTable::supports(kGetriBlocked, ri)) << "guard";
    ri.backend = Backend::NETLIB;
    EXPECT_FALSE(GetriTable::supports(kGetriBlocked, ri))
        << "the native getri READS packed int32; a netlib getrf wrote int64";
    EXPECT_TRUE(is_vendor(resolve_getri_route<float>(kGetriBlocked, ri, true)));

    // With no vendor at all there is no route, and "throws NoRouteError" is the
    // honest answer for a pivot format the native kernel cannot serve.
    EXPECT_FALSE(is_native(resolve_getrf_route<float>(kGetrfAuto, f, false)));
}

// THE MEASURED ORDER WINDOW. The clause names Algorithm::Blocked, NOT "native":
// cta_max_n is passed large below so CTA is SUPPORTED at every order in the loop
// and a clause that forgot the algo test is caught. The boundary is per type.
// evidence: docs/perf/lu.md#getrf-window-evidence
TEST(RouteGetrf, PreferredIsTheMeasuredOrderWindowPerTypeAndBlockedOnly) {
    for (int64_t batch : {1, 2, 128, 8192}) {
        for (int64_t order : {256, 257, 512, 2048}) {
            const auto s = getrf_shape(order, batch, /*cta_max_n=*/4096);
            EXPECT_TRUE(GetrfTable::preferred(kGetrfBlocked, s))
                << "float order " << order << " batch " << batch;
            EXPECT_FALSE(GetrfTable::preferred(kGetrfCta, s))
                << "the CTA arm is NOT in the window: float n=128, where it "
                   "serves, reads 0.825/0.773/0.872 at batch 256/512/1024";
            EXPECT_FALSE(GetrfTable::preferred(kVendorAuto, s))
                << "the vendor is where the walk ENDS, never itself preferred";
            const Route r = resolve_getrf_route<float>(kGetrfAuto, s, true);
            EXPECT_TRUE(is_native(r) && r.algo == Algorithm::Blocked)
                << "float order " << order << " batch " << batch
                << ": CTA is supported here (cta_max_n = 4096) and must still not "
                   "be selected -- the clause names Blocked";
        }
        for (int64_t order : {1, 32, 40, 128, 255}) {
            const auto s = getrf_shape(order, batch, 4096);
            EXPECT_FALSE(GetrfTable::preferred(kGetrfBlocked, s)) << "float order " << order;
            EXPECT_FALSE(GetrfTable::preferred(kGetrfCta, s));
            EXPECT_TRUE(is_vendor(resolve_getrf_route<float>(kGetrfAuto, s, true)))
                << "order " << order << " batch " << batch;
        }
        // cfloat's boundary is 512, not float's 256.
        for (int64_t order : {512, 513, 2048}) {
            const auto s = getrf_shape(order, batch, 4096);
            EXPECT_TRUE(GetrfTableCF::preferred(kGetrfBlocked, s)) << "cfloat order " << order;
            const Route r = resolve_getrf_route<std::complex<float>>(kGetrfAuto, s, true);
            EXPECT_TRUE(is_native(r) && r.algo == Algorithm::Blocked);
        }
        for (int64_t order : {1, 128, 256, 511}) {
            const auto s = getrf_shape(order, batch, 4096);
            EXPECT_FALSE(GetrfTableCF::preferred(kGetrfBlocked, s))
                << "cfloat order " << order << ": n=256 batch=128 is 0.884";
            EXPECT_TRUE(is_vendor(
                resolve_getrf_route<std::complex<float>>(kGetrfAuto, s, true)));
        }
        // double and cdouble: nothing, at any order.
        for (int64_t order : {1, 128, 256, 512, 1024, 2048}) {
            const auto s = getrf_shape(order, batch, 4096);
            EXPECT_FALSE(GetrfTableD::preferred(kGetrfBlocked, s)) << "double order " << order;
            EXPECT_FALSE(GetrfTableCD::preferred(kGetrfBlocked, s)) << "cdouble order " << order;
            EXPECT_TRUE(is_vendor(resolve_getrf_route<double>(kGetrfAuto, s, true)));
            EXPECT_TRUE(is_vendor(
                resolve_getrf_route<std::complex<double>>(kGetrfAuto, s, true)));
        }
    }

    // THE WINDOW IS NOT A CORRECTNESS GATE. Inside it, on a build with no
    // blocked driver, preferred() must still say yes and supports() must say no.
    const auto absent = getrf_shape(512, 256, /*cta_max_n=*/0);
    EXPECT_TRUE(GetrfTable::preferred(kGetrfBlocked, absent));
    EXPECT_FALSE(GetrfTable::supports(kGetrfBlocked, absent));
    EXPECT_TRUE(is_vendor(resolve_getrf_route<float>(kGetrfAuto, absent, true)));
}

TEST(RouteGetrf, BareOriginResolvesToASpecificAlgorithm) {
    // getrf has TWO native routes, so {Native, Auto} names neither. Below the CTA
    // capacity -> CTA; above it -> Blocked. Never "no route".
    const auto small = getrf_shape(64, 256, 128);
    const auto big   = getrf_shape(1024, 64, 128);

    const Route rs = resolve_getrf_route<float>(kGetrfNativeBare, small,
                                                /*vendor_available=*/true);
    EXPECT_EQ(rs.origin, Origin::Native);
    EXPECT_EQ(rs.algo, Algorithm::CTA);

    const Route rb = resolve_getrf_route<float>(kGetrfAuto, big,
                                                /*vendor_available=*/false);
    EXPECT_TRUE(is_native(rb));
    EXPECT_EQ(rb.algo, Algorithm::Blocked)
        << "an order above the tile capacity must fall to the blocked driver, not "
           "vanish";
}

TEST(RouteGetrf, AbsentKernelIsUnsupportedRatherThanSelectable) {
    // Zero capacity / blocked_available == false is what a build without the kernels
    // reports: both native routes must then be UNSUPPORTED.
    const auto s = getrf_shape(/*order=*/64, /*batch=*/256, /*cta_max_n=*/0);
    EXPECT_FALSE(GetrfTable::supports(kGetrfCta, s));
    EXPECT_FALSE(GetrfTable::supports(kGetrfBlocked, s));
    EXPECT_TRUE(is_vendor(resolve_getrf_route<float>(kGetrfAuto, s, true)));
    EXPECT_TRUE(is_vendor(resolve_getrf_route<float>(kGetrfAuto, s, false)))
        << "vendor-free with nothing supported must say 'needs a vendor', not "
           "invent a native route";

    // AND FORCING MUST NOT ESCAPE IT: a forced route is gated on supports(), so a
    // green forced-route test is not evidence that a native kernel ran.
    EXPECT_TRUE(is_vendor(resolve_getrf_route<float>(kGetrfCta, s, true)));
    EXPECT_TRUE(is_vendor(resolve_getrf_route<float>(kGetrfBlocked, s, true)));
    EXPECT_TRUE(is_vendor(resolve_getrf_route<float>(kGetrfNativeBare, s, true)));
    EXPECT_TRUE(is_vendor(resolve_getrf_route<float>(kGetrfCta, s, false)));

    // Half a capability is still absent: the blocked driver's diagonal-panel leaf IS
    // the CTA device function, so it inherits the presence gate.
    auto half = getrf_shape(64, 256, /*cta_max_n=*/0);
    half.blocked_available = true;
    EXPECT_FALSE(GetrfTable::supports(kGetrfBlocked, half))
        << "the blocked driver's diagonal-panel leaf IS the CTA device function, "
           "so it inherits the presence gate";
}

TEST(RouteGetrf, NativeTierPreferredIsDeclaredAndPinsTheMeasuredTierChoice) {
    // native_tier_preferred() is the third predicate, consulted ONLY in the
    // vendor-free branch. DOUBLE alone prefers the blocked driver below its own CTA
    // ceiling. evidence: docs/perf/lu.md#native_tier_preferred
    EXPECT_TRUE((declares_native_tier_preferred<GetrfTable, GetrfShape>))
        << "the tier sweep has run; an undeclared hook now costs 1.18-1.29x at "
           "double n=76..96 in the vendor-free build";

    // float: CTA below the capacity ceiling, from the hook and not the order array.
    const auto small_f = getrf_shape(64, 8192, 128);
    EXPECT_TRUE((GetrfTable::native_tier_preferred(kGetrfCta, small_f)));
    EXPECT_FALSE((GetrfTable::native_tier_preferred(kGetrfBlocked, small_f)));
    EXPECT_EQ(resolve_getrf_route<float>(kGetrfAuto, small_f,
                                         /*vendor_available=*/false).algo,
              Algorithm::CTA);

    // double: the ONE type where the hook overrides kGetrfOrder. Without it the
    // vendor-free walk would return CTA here purely because CTA is listed first.
    using GetrfTableD = RouteTable<Op::getrf, double>;
    auto small_d = getrf_shape(64, 8192, 128);
    small_d.scalar = ScalarKind::F64;
    EXPECT_FALSE((GetrfTableD::native_tier_preferred(kGetrfCta, small_d)));
    EXPECT_TRUE((GetrfTableD::native_tier_preferred(kGetrfBlocked, small_d)));
    EXPECT_EQ(resolve_getrf_route<double>(kGetrfAuto, small_d,
                                          /*vendor_available=*/false).algo,
              Algorithm::Blocked)
        << "double's vendor-free tier choice must come from the measured hook, "
           "not from kGetrfOrder's CTA-first ladder";

    // ...and at n <= 32 double goes back to CTA: the blocked driver runs ONE panel
    // whose leaf IS the CTA device function, so CTA is the cheaper spelling of it.
    auto tiny_d = getrf_shape(32, 8192, 128);
    tiny_d.scalar = ScalarKind::F64;
    EXPECT_TRUE((GetrfTableD::native_tier_preferred(kGetrfCta, tiny_d)));
    EXPECT_EQ(resolve_getrf_route<double>(kGetrfAuto, tiny_d,
                                          /*vendor_available=*/false).algo,
              Algorithm::CTA);

    // IT IS NOT A CORRECTNESS GATE: both arms stay supports()-able wherever the
    // window moves, so a pinned `cta` still runs CTA instead of falling through.
    EXPECT_TRUE((GetrfTableD::supports(kGetrfCta, small_d)));
    EXPECT_TRUE((GetrfTableD::supports(kGetrfBlocked, small_d)));
    EXPECT_EQ(resolve_getrf_route<double>(kGetrfCta, small_d,
                                          /*vendor_available=*/false).algo,
              Algorithm::CTA);

    // AND IT MOVES NOTHING IN A VENDOR-PRESENT BUILD: the hook is consulted only
    // inside the `!vendor_available` branch.
    EXPECT_TRUE(is_vendor(resolve_getrf_route<double>(kGetrfAuto, small_d,
                                                      /*vendor_available=*/true)));

    // For contrast, geqrf declares it too.
    EXPECT_TRUE((declares_native_tier_preferred<RouteTable<Op::geqrf, float>, GeqrfShape>))
        << "geqrf's measured tier window must not be deleted by a WP6 copy-paste";

    // getrs declares it too: kGetrsOrder holds the fused {Native, CTA} ahead of the
    // composition, and the two are 7.9x apart at nrhs = 1.
    EXPECT_TRUE((declares_native_tier_preferred<GetrsTable, GetrsShape>))
        << "getrs has two native tiers; the order array alone cannot follow a "
           "crossover between them";

    // getri is single-arm and must NOT declare it: with one native route there is no
    // native-vs-native question.
    EXPECT_FALSE((declares_native_tier_preferred<GetriTable, GetriShape>));
}

TEST(RouteGetrf, BatchlasGetrfRouteIsActuallyRead) {
    // The canonical spelling needs no registry entry; parse_route_env synthesises it.
    ClearRouteEnv clear(Op::getrf);

    EXPECT_EQ(op_env_stem(Op::getrf), "GETRF");
    EXPECT_TRUE(std::string(legacy_variable_for(Op::getrf)).empty())
        << "no legacy getrf variable ever shipped; a case in legacy_variable_for "
           "would INVENT a legacy spelling. Note that Op::ormqr DOES have one "
           "(route_env.hh:118) -- that is not a precedent for this op";

    {
        const auto unset = parse_route_env(Op::getrf);
        EXPECT_FALSE(unset.found);
        EXPECT_FALSE(unset.unparsed);
        EXPECT_EQ(legacy_unset_default(Op::getrf).origin, Origin::Auto);
    }
    {
        ScopedEnv e("BATCHLAS_GETRF_ROUTE", "cta");
        const auto p = parse_route_env(Op::getrf);
        ASSERT_TRUE(p.found) << "BATCHLAS_GETRF_ROUTE was not read at all";
        EXPECT_EQ(p.route, (Route{Origin::Native, Algorithm::CTA}))
            << "a bare algorithm implies Origin::Native";
        EXPECT_EQ(p.source.variable, "BATCHLAS_GETRF_ROUTE");
        EXPECT_FALSE(p.source.legacy);
    }
    {
        ScopedEnv e("BATCHLAS_GETRF_ROUTE", "native:blocked");
        const auto p = parse_route_env(Op::getrf);
        ASSERT_TRUE(p.found);
        EXPECT_EQ(p.route, (Route{Origin::Native, Algorithm::Blocked}));
    }
    {
        ScopedEnv e("BATCHLAS_GETRF_ROUTE", "vendor");
        const auto p = parse_route_env(Op::getrf);
        ASSERT_TRUE(p.found);
        EXPECT_EQ(p.route, (Route{Origin::Vendor, Algorithm::Auto}));
    }
    {
        // AN UNRECOGNISED VALUE IS SILENTLY {Auto, Auto}, WHICH IS THE VENDOR: a
        // "native" run that looks identical to the vendor probably IS the vendor.
        ScopedEnv e("BATCHLAS_GETRF_ROUTE", "not-a-route");
        const auto p = parse_route_env(Op::getrf);
        EXPECT_FALSE(p.found);
        EXPECT_TRUE(p.unparsed) << "a typo must be reported, not silently Auto";
    }
}

// ---------------------------------------------------------------------------
// GETRS: the fused narrow-RHS CTA tier and the composition over the routed trsm.
// ---------------------------------------------------------------------------

TEST(RouteGetrs, VendorFreeFallbackHandsOverTheNativeRoute) {
    // The speed-threshold guard again, and here the temptation is concrete: the
    // composed arm is a measured loss at nrhs=1. That threshold belongs in
    // preferred() -- in supports() it removes the vendor-free route.
    const auto s = getrs_shape(/*order=*/32, /*nrhs=*/1, /*batch=*/1);

    EXPECT_TRUE(GetrsTable::supports(kGetrsBlocked, s))
        << "nrhs and batch are speed questions; neither may gate CORRECTNESS, even "
           "though nrhs=1 is measured 0.36x geomean";
    EXPECT_FALSE(GetrsTable::preferred(kGetrsBlocked, s))
        << "the COMPOSITION is never preferred at any width the fused tier serves; "
           "it is 0.36x geomean here and the window belongs to CTA alone";

    EXPECT_TRUE(is_native(resolve_getrs_route<float>(kGetrsAuto, s, false)));

    // With a vendor present this shape is native too: nrhs = 1 is inside the
    // measured window, and supports() is unchanged by the window.
    const Route with_vendor = resolve_getrs_route<float>(kGetrsAuto, s, true);
    EXPECT_TRUE(is_native(with_vendor));
    EXPECT_EQ(with_vendor.algo, Algorithm::CTA);

    // The width just outside clause A for a NON-float type is the other half of
    // the window and must still take the vendor.
    const auto wide = getrs_shape(/*order=*/32, /*nrhs=*/4, /*batch=*/1);
    EXPECT_TRUE(is_vendor((resolve_route<Op::getrs, double>(kGetrsAuto, wide, true))))
        << "double at nrhs = 4 is OUTSIDE the measured window (its n=128 ladder dips "
           "to 0.940x mid-ladder); routing it native would ship a measured loss";
}

TEST(RouteGetrs, AllThreeTransposeModesAreSupportedAndTransAReachesTheShape) {
    // transA is a LIVE routing input for this op and a genuine algorithm fork:
    // NoTrans applies P first and solves L then U, while Trans/ConjTrans solve
    // U^T/U^H then L^T/L^H and apply P^T LAST, on the output, in reverse. All three
    // must be SUPPORTED.
    for (Transpose t : {Transpose::NoTrans, Transpose::Trans, Transpose::ConjTrans}) {
        const auto s = getrs_shape(/*order=*/64, /*nrhs=*/8, /*batch=*/128,
                                   /*blocked_available=*/true, t);
        EXPECT_TRUE(GetrsTable::supports(kGetrsBlocked, s))
            << "transpose mode " << static_cast<int>(t);
        EXPECT_EQ(s.transA, t)
            << "the shape must CARRY transA -- it is what makes getrs's coverage "
               "rows separable at all";
    }
}

TEST(RouteGetrs, CorrectnessGatesAreNotSpeedGates) {
    const auto ok = getrs_shape(/*order=*/64, /*nrhs=*/8, /*batch=*/256);
    ASSERT_TRUE(GetrsTable::supports(kGetrsBlocked, ok))
        << "guard: the permissive shape must be supported, or every EXPECT_FALSE "
           "below passes for the wrong reason";

    auto cpu = ok;  cpu.is_gpu = false;
    EXPECT_FALSE(GetrsTable::supports(kGetrsBlocked, cpu));

    auto nosg = ok;  nosg.has_sg32 = false;
    EXPECT_FALSE(GetrsTable::supports(kGetrsBlocked, nosg));

    auto het = ok;  het.heterogeneous_batch = true;
    EXPECT_FALSE(GetrsTable::supports(kGetrsBlocked, het))
        << "one launch, one (order, nrhs, ld, stride) tuple, and the pivot list is "
           "read at b*order + k with a single order";

    auto no_rhs = ok;  no_rhs.n = 0;
    EXPECT_FALSE(GetrsTable::supports(kGetrsBlocked, no_rhs));

    auto empty = ok;  empty.m = 0; empty.k = 0;
    EXPECT_FALSE(GetrsTable::supports(kGetrsBlocked, empty));

    auto no_batch = ok;  no_batch.batch = 0;
    EXPECT_FALSE(GetrsTable::supports(kGetrsBlocked, no_batch));

    // NOT correctness gates.
    auto one_rhs = ok;  one_rhs.n = 1;
    EXPECT_TRUE(GetrsTable::supports(kGetrsBlocked, one_rhs))
        << "nrhs=1 is where the composition LOSES 0.36x geomean -- that belongs in "
           "preferred(), and putting it here would delete the vendor-free route";
    auto tiny_batch = ok;  tiny_batch.batch = 1;
    EXPECT_TRUE(GetrsTable::supports(kGetrsBlocked, tiny_batch));
    auto huge = ok;  huge.m = 1 << 20; huge.k = 1 << 20;
    EXPECT_TRUE(GetrsTable::supports(kGetrsBlocked, huge))
        << "the two solves are the ROUTED trsm, whose blocked tier carries no upper "
           "bound on the order; a transcribed ceiling here could not fire and would "
           "read as live";
}

// THE MEASURED nrhs WINDOW, pinned from BOTH sides:
//     nrhs <= 2  for every type and order   -- clause A
//   + nrhs <= 4  for float only             -- clause B
// plus clause C for the composition below. The composition is never preferred at
// any width the fused tier serves.
// evidence: docs/perf/lu.md#getrs-fused-window-evidence
TEST(RouteGetrs, PreferredIsTheMeasuredNrhsWindowAndNothingWider) {
    // ---- clause A: every type, every order, nrhs <= 2 ----------------------
    for (int64_t order : {1, 32, 128, 2048}) {
        for (int64_t nrhs : {int64_t(1), int64_t(2)}) {
            for (int64_t batch : {1, 128, 8192}) {
                const auto s = getrs_shape(order, nrhs, batch);
                EXPECT_TRUE(GetrsTable::preferred(kGetrsCta, s))
                    << "clause A: order " << order << " nrhs " << nrhs;
                EXPECT_FALSE(GetrsTable::preferred(kGetrsBlocked, s))
                    << "the COMPOSITION must never be preferred: it is the arm the "
                       "fused tier replaces and it loses to the vendor at every width "
                       "the fused tier serves";
                EXPECT_FALSE(GetrsTable::preferred(kVendorAuto, s))
                    << "preferred() is asked only of NATIVE routes; a true here would "
                       "make the vendor win the first walk for the wrong reason";
                const Route r = resolve_getrs_route<float>(kGetrsAuto, s, true);
                EXPECT_TRUE(is_native(r) && r.algo == Algorithm::CTA)
                    << "order " << order << " nrhs " << nrhs << " batch " << batch;
                // ... and every type, not just float.
                EXPECT_TRUE((RouteTable<Op::getrs, double>::preferred(kGetrsCta, s)));
                EXPECT_TRUE((RouteTable<Op::getrs, std::complex<float>>::preferred(kGetrsCta, s)));
                EXPECT_TRUE((RouteTable<Op::getrs, std::complex<double>>::preferred(kGetrsCta, s)));
            }
        }
    }

    // ---- clause B is FLOAT ONLY, the half most likely to be widened by someone
    // who reads "nrhs <= 4" and drops the type test ------------------------
    for (int64_t order : {32, 128, 1024, 2048}) {
        const auto s = getrs_shape(order, /*nrhs=*/4, /*batch=*/256);
        EXPECT_TRUE(GetrsTable::preferred(kGetrsCta, s))
            << "clause B: float nrhs=4 at order " << order;
        EXPECT_FALSE((RouteTable<Op::getrs, double>::preferred(kGetrsCta, s)))
            << "double nrhs=4 is OUTSIDE the window: its n=128 ladder dips to 0.940x "
               "at batch 2048, MID-LADDER, where no boundary in n or batch reaches it";
        EXPECT_FALSE((RouteTable<Op::getrs, std::complex<float>>::preferred(kGetrsCta, s)))
            << "cfloat nrhs=4 dips to 0.976x at n=1024 batch 16";
        EXPECT_FALSE((RouteTable<Op::getrs, std::complex<double>>::preferred(kGetrsCta, s)))
            << "cdouble nrhs=4 is 0.577x at n=32 and dips mid-ladder at n=128 and 1024";
    }

    // ---- outside EVERY window, EVERY type takes the vendor ------------------
    for (int64_t nrhs : {5, 8, 16, 32, 63}) {
        const auto s = getrs_shape(/*order=*/256, nrhs, /*batch=*/512);
        EXPECT_FALSE(GetrsTable::preferred(kGetrsCta, s)) << "float nrhs " << nrhs;
        EXPECT_FALSE(GetrsTable::preferred(kGetrsBlocked, s))
            << "float nrhs " << nrhs << " is BELOW clause C's boundary of 64; the "
               "measured cell just under it is 0.9069 at n=64 nrhs=32 batch=4096";
        EXPECT_TRUE(is_vendor(resolve_getrs_route<float>(kGetrsAuto, s, true)))
            << "nrhs " << nrhs << " is outside the window and must take the vendor";
        EXPECT_FALSE((GetrsTableD::preferred(kGetrsCta, s)));
        EXPECT_FALSE((GetrsTableCD::preferred(kGetrsCta, s)));
    }

    // ---- CLAUSE C: THE WIDE-nrhs COMPOSITION WINDOW ------------------------
    // AXIS: GetrsShape::nrhs(), which is B.cols(), NOT order(). A predicate on the
    // wrong extent inverts the window, and the order loop below is what proves it is
    // not read. The boundary is per type, and clause C carries a batch floor that
    // clauses A and B do not -- deliberately conservative, and pinned on both sides.
    // evidence: docs/perf/lu.md#getrs-composition-window-evidence
    for (int64_t order : {32, 64, 128, 512, 1024, 2048}) {
        for (int64_t batch : {128, 129, 4096}) {
            // float: IN at 64, OUT at 63.
            const auto f_in  = getrs_shape(order, /*nrhs=*/64, batch);
            const auto f_out = getrs_shape(order, /*nrhs=*/63, batch);
            EXPECT_TRUE(GetrsTable::preferred(kGetrsBlocked, f_in))
                << "clause C float, order " << order << " batch " << batch;
            EXPECT_FALSE(GetrsTable::preferred(kGetrsBlocked, f_out));
            EXPECT_FALSE(GetrsTable::preferred(kGetrsCta, f_in))
                << "the FUSED tier must stay unpreferred at nrhs 64; it cannot "
                   "even serve it (kGetrsFusedMaxRhs = 8) and a true here would "
                   "make the walk stop on a route supports() then refuses";
            const Route r = resolve_getrs_route<float>(kGetrsAuto, f_in, true);
            EXPECT_TRUE(is_native(r) && r.algo == Algorithm::Blocked)
                << "float nrhs=64 order " << order << " batch " << batch;
            EXPECT_TRUE(is_vendor(resolve_getrs_route<float>(kGetrsAuto, f_out, true)));

            // double: IN at 128, OUT at 127 AND at 64 -- the half most likely to be
            // widened by someone who reads "nrhs >= 64" and drops the type test.
            const auto d_in  = getrs_shape(order, /*nrhs=*/128, batch);
            const auto d_127 = getrs_shape(order, /*nrhs=*/127, batch);
            const auto d_64  = getrs_shape(order, /*nrhs=*/64,  batch);
            EXPECT_TRUE(GetrsTableD::preferred(kGetrsBlocked, d_in));
            EXPECT_FALSE(GetrsTableD::preferred(kGetrsBlocked, d_127));
            EXPECT_FALSE(GetrsTableD::preferred(kGetrsBlocked, d_64));
            EXPECT_TRUE(is_native(resolve_getrs_route<double>(kGetrsAuto, d_in, true)));
            EXPECT_TRUE(is_vendor(resolve_getrs_route<double>(kGetrsAuto, d_64, true)));

            // cfloat and cdouble: NOTHING, at any width -- each has a mid-ladder dip
            // that no boundary in batch, order or nrhs can exclude.
            for (int64_t q : {64, 128, 256}) {
                const auto s = getrs_shape(order, q, batch);
                EXPECT_FALSE(GetrsTableCF::preferred(kGetrsBlocked, s))
                    << "cfloat nrhs " << q << ": mid-ladder dip at n=64 b=1024";
                EXPECT_FALSE(GetrsTableCD::preferred(kGetrsBlocked, s))
                    << "cdouble nrhs " << q << ": 0.9238 at n=128 nrhs=128 b=1024, "
                       "and 12 losses of 13 at nrhs 64";
                EXPECT_TRUE(is_vendor(resolve_getrs_route<std::complex<float>>(
                    kGetrsAuto, s, true)));
                EXPECT_TRUE(is_vendor(resolve_getrs_route<std::complex<double>>(
                    kGetrsAuto, s, true)));
            }
        }
    }

    // THE CLAUSE IS ON nrhs AND NOT ON order, PROVED BY CONSTRUCTION: hold nrhs and
    // sweep order over four decades; the answer may not move.
    {
        bool in_all = true, out_all = false;
        for (int64_t order : {1, 2, 8, 63, 64, 65, 1000, 100000}) {
            in_all  &= GetrsTable::preferred(kGetrsBlocked, getrs_shape(order, 64, 512));
            out_all |= GetrsTable::preferred(kGetrsBlocked, getrs_shape(order, 63, 512));
        }
        EXPECT_TRUE(in_all)  << "clause C must admit nrhs=64 at EVERY order";
        // ...and the batch floor, from both sides, at a width and an order that
        // are both well inside the window.
        EXPECT_TRUE (GetrsTable::preferred(kGetrsBlocked, getrs_shape(512, 128, 128)));
        EXPECT_FALSE(GetrsTable::preferred(kGetrsBlocked, getrs_shape(512, 128, 127)));
        EXPECT_FALSE(GetrsTable::preferred(kGetrsBlocked, getrs_shape(512, 128, 1)))
            << "clause C must not route batch 1: the low end is ragged and the "
               "only readings there came from a contaminated sweep";
        EXPECT_TRUE (GetrsTableD::preferred(kGetrsBlocked, getrs_shape(512, 128, 128)));
        EXPECT_FALSE(GetrsTableD::preferred(kGetrsBlocked, getrs_shape(512, 128, 127)));
        EXPECT_TRUE(is_vendor(resolve_getrs_route<float>(
            kGetrsAuto, getrs_shape(512, 128, 127), true)));
        EXPECT_FALSE(out_all) << "clause C must refuse nrhs=63 at EVERY order -- a "
                                 "true here means the predicate is reading order()";
    }

    // THE WINDOW IS NOT A CORRECTNESS GATE: at nrhs = 8 the fused tier is not
    // preferred but IS supported, so a pinned native:cta still runs it there.
    {
        const auto s = getrs_shape(/*order=*/256, /*nrhs=*/8, /*batch=*/512);
        EXPECT_TRUE(GetrsTable::supports(kGetrsCta, s));
        EXPECT_FALSE(GetrsTable::preferred(kGetrsCta, s));
        const Route pinned = resolve_getrs_route<float>(kGetrsCta, s, true);
        EXPECT_TRUE(is_native(pinned) && pinned.algo == Algorithm::CTA);
    }

    // ---- and the window may not outrun the CAPACITY: inside by nrhs, outside by
    // elements, supports() refuses so preferred() cannot select an absent launch.
    {
        auto s = getrs_shape(/*order=*/kFusedMaxElemsF32 + 1, /*nrhs=*/1, /*batch=*/8);
        EXPECT_TRUE(GetrsTable::preferred(kGetrsCta, s))
            << "guard: preferred() must NOT repeat the capacity test, or a pinned "
               "native:cta above the ceiling would silently resolve elsewhere";
        EXPECT_FALSE(GetrsTable::supports(kGetrsCta, s));
        const Route r = resolve_getrs_route<float>(kGetrsAuto, s, true);
        EXPECT_TRUE(is_vendor(r)) << "above the resident-RHS ceiling the vendor takes it";
        const Route rf = resolve_getrs_route<float>(kGetrsAuto, s, false);
        EXPECT_TRUE(is_native(rf) && rf.algo == Algorithm::Blocked)
            << "a vendor-free build above the ceiling must fall to the COMPOSITION";
    }

    // ---- ABSENT TIERS. Each capability is independent -----------------------
    // (a) the fused tier absent, the composition present.
    {
        const auto s = getrs_shape(64, 1, 256, /*blocked_available=*/true,
                                   Transpose::NoTrans, /*fused_max_elems=*/0,
                                   /*fused_max_nrhs=*/0);
        EXPECT_FALSE(GetrsTable::supports(kGetrsCta, s));
        EXPECT_TRUE(GetrsTable::supports(kGetrsBlocked, s));
        EXPECT_TRUE(is_vendor(resolve_getrs_route<float>(kGetrsAuto, s, true)))
            << "with the fused tier absent, preferred() selects nothing and the "
               "vendor takes it -- the pre-WP6-PERF behaviour, exactly";
        const Route rf = resolve_getrs_route<float>(kGetrsAuto, s, false);
        EXPECT_TRUE(is_native(rf) && rf.algo == Algorithm::Blocked);
        EXPECT_TRUE(is_native(resolve_getrs_route<float>(kGetrsCta, s, false)))
            << "a forced native:cta the build cannot serve falls to automatic(), "
               "which in a vendor-free build is the composition";
    }
    // (b) the composition absent, the fused tier present.
    {
        const auto s = getrs_shape(64, 1, 256, /*blocked_available=*/false);
        EXPECT_FALSE(GetrsTable::supports(kGetrsBlocked, s));
        EXPECT_TRUE(GetrsTable::supports(kGetrsCta, s));
        const Route r = resolve_getrs_route<float>(kGetrsAuto, s, false);
        EXPECT_TRUE(is_native(r) && r.algo == Algorithm::CTA);
    }
    // (c) BOTH absent -- the original assertion, unchanged in meaning.
    {
        const auto absent = getrs_shape(64, 8, 256, /*blocked_available=*/false,
                                        Transpose::NoTrans, 0, 0);
        EXPECT_FALSE(GetrsTable::supports(kGetrsBlocked, absent));
        EXPECT_FALSE(GetrsTable::supports(kGetrsCta, absent));
        EXPECT_TRUE(is_vendor(resolve_getrs_route<float>(kGetrsAuto, absent, true)));
        EXPECT_TRUE(is_vendor(resolve_getrs_route<float>(kGetrsAuto, absent, false)));
        EXPECT_TRUE(is_vendor(resolve_getrs_route<float>(kGetrsBlocked, absent, true)));
        EXPECT_TRUE(is_vendor(resolve_getrs_route<float>(kGetrsNativeBare, absent, true)));
    }
}

TEST(RouteGetrs, BareOriginResolvesToASpecificAlgorithm) {
    // {Native, Auto} must not come back verbatim: no dispatch tail can map it to a
    // driver. AND THE ANSWER CHANGED WHEN THE FUSED TIER LANDED -- a bare `native`
    // pin used to mean the composition and now means the fused kernel, so any
    // baseline recorded with one is measuring a different getrs today.
    const auto s = getrs_shape(64, 8, 256);
    ASSERT_TRUE(GetrsTable::supports(kGetrsCta, s))
        << "guard: 64 x 8 = 512 elements is well inside the capacity, so the "
           "assertion below must be about the ORDER and not about a refusal";
    const Route r = resolve_getrs_route<float>(kGetrsNativeBare, s,
                                               /*vendor_available=*/true);
    EXPECT_EQ(r.origin, Origin::Native);
    EXPECT_EQ(r.algo, Algorithm::CTA)
        << "a bare `native` origin must resolve to the FIRST supported route in "
           "kGetrsOrder, which is now the fused tier";
    EXPECT_FALSE(GetrsTable::supports(kGetrsNativeBare, s))
        << "{Native, Auto} itself must never be reported supported";

    // Above the fused tier's width it still resolves, to the composition.
    const auto wide = getrs_shape(64, 64, 256);
    EXPECT_FALSE(GetrsTable::supports(kGetrsCta, wide));
    const Route rw = resolve_getrs_route<float>(kGetrsNativeBare, wide, true);
    EXPECT_EQ(rw.origin, Origin::Native);
    EXPECT_EQ(rw.algo, Algorithm::Blocked);
}

TEST(RouteGetrs, BatchlasGetrsRouteIsActuallyRead) {
    ClearRouteEnv clear(Op::getrs);

    EXPECT_EQ(op_env_stem(Op::getrs), "GETRS");
    EXPECT_TRUE(std::string(legacy_variable_for(Op::getrs)).empty())
        << "no legacy getrs variable ever shipped; a case in legacy_variable_for "
           "would INVENT a legacy spelling";

    {
        const auto unset = parse_route_env(Op::getrs);
        EXPECT_FALSE(unset.found);
        EXPECT_EQ(legacy_unset_default(Op::getrs).origin, Origin::Auto);
    }
    {
        ScopedEnv e("BATCHLAS_GETRS_ROUTE", "blocked");
        const auto p = parse_route_env(Op::getrs);
        ASSERT_TRUE(p.found) << "BATCHLAS_GETRS_ROUTE was not read at all";
        EXPECT_EQ(p.route, (Route{Origin::Native, Algorithm::Blocked}));
        EXPECT_EQ(p.source.variable, "BATCHLAS_GETRS_ROUTE");
        EXPECT_FALSE(p.source.legacy);
    }
    {
        ScopedEnv e("BATCHLAS_GETRS_ROUTE", "vendor");
        const auto p = parse_route_env(Op::getrs);
        ASSERT_TRUE(p.found);
        EXPECT_EQ(p.route, (Route{Origin::Vendor, Algorithm::Auto}));
    }
    {
        ScopedEnv e("BATCHLAS_GETRS_ROUTE", "not-a-route");
        const auto p = parse_route_env(Op::getrs);
        EXPECT_FALSE(p.found);
        EXPECT_TRUE(p.unparsed) << "a typo must be reported, not silently Auto";
    }
}

// ---------------------------------------------------------------------------
// GETRI. One native arm: a composition over the routed trsm.
// ---------------------------------------------------------------------------

TEST(RouteGetri, VendorFreeFallbackHandsOverTheNativeRoute) {
    // n=40, batch=2 are inverse_tests' actual extents, and getri is the first LU op
    // inv.cc sizes; that suite closes only if these extents stay supported.
    const auto s = getri_shape(/*order=*/40, /*batch=*/2);

    EXPECT_TRUE(GetriTable::supports(kGetriBlocked, s))
        << "batch size and order are speed questions; neither may gate CORRECTNESS "
           "-- and these are inverse_tests' own extents";
    EXPECT_FALSE(GetriTable::preferred(kGetriBlocked, s));

    EXPECT_TRUE(is_native(resolve_getri_route<float>(kGetriAuto, s, false)));
    EXPECT_TRUE(is_vendor(resolve_getri_route<float>(kGetriAuto, s, true)));
}

TEST(RouteGetri, CorrectnessGatesIncludeTheOnesInheritedFromTrsm) {
    // getri's native arm is a composition over the ROUTED trsm, so trsm's structural
    // gates are TRANSCRIBED here; omitting one is the wrong-answer class.
    const auto ok = getri_shape(/*order=*/64, /*batch=*/256);
    ASSERT_TRUE(GetriTable::supports(kGetriBlocked, ok))
        << "guard: the permissive shape must be supported, or every EXPECT_FALSE "
           "below passes for the wrong reason";

    auto cpu = ok;  cpu.is_gpu = false;
    EXPECT_FALSE(GetriTable::supports(kGetriBlocked, cpu))
        << "INHERITED from route_trsm.hh:138-142";

    auto het = ok;  het.heterogeneous_batch = true;
    EXPECT_FALSE(GetriTable::supports(kGetriBlocked, het))
        << "INHERITED from route_trsm.hh:151-154, and getri's own besides -- the "
           "pivot list is read at b*order + k with a single order";

    auto nosg = ok;  nosg.has_sg32 = false;
    EXPECT_FALSE(GetriTable::supports(kGetriBlocked, nosg));

    auto wide = ok;  wide.n = 1024;
    EXPECT_FALSE(GetriTable::supports(kGetriBlocked, wide))
        << "getri's operand is square (options.hh:687-690)";

    auto empty = ok;  empty.m = 0; empty.n = 0; empty.k = 0;
    EXPECT_FALSE(GetriTable::supports(kGetriBlocked, empty));

    auto no_batch = ok;  no_batch.batch = 0;
    EXPECT_FALSE(GetriTable::supports(kGetriBlocked, no_batch));

    // NOT correctness gates.
    auto tiny_batch = ok;  tiny_batch.batch = 1;
    EXPECT_TRUE(GetriTable::supports(kGetriBlocked, tiny_batch));
    auto two = ok;  two.batch = 2;
    EXPECT_TRUE(GetriTable::supports(kGetriBlocked, two))
        << "inverse_tests runs at batch 2";
    auto small = getri_shape(32, 8192);
    EXPECT_TRUE(GetriTable::supports(kGetriBlocked, small))
        << "n=32 is where the composition LOSES 0.23-0.54x; that is preferred()'s "
           "business, not supports()'";
    auto huge = ok;  huge.m = 1 << 20; huge.n = 1 << 20; huge.k = 1 << 20;
    EXPECT_TRUE(GetriTable::supports(kGetriBlocked, huge))
        << "the routed trsm's blocked tier carries no upper bound on the order; a "
           "transcribed ceiling here could not fire and would read as live";
}

// THE MEASURED ORDER WINDOW, per type. THE AXIS IS GetriShape::order(), which is
// `k`, and there is NO batch term: at batch 1..32 the native driver beats cuBLAS
// everywhere because cuBLAS's batched getri is a per-item loop there.
// evidence: docs/perf/lu.md#getri-window-evidence
TEST(RouteGetri, PreferredIsTheMeasuredOrderWindowPerType) {
    for (int64_t batch : {1, 2, 4, 128, 8192}) {
        // ---- float: IN at 128, OUT at 127 and at 64 -----------------------
        for (int64_t order : {128, 129, 256, 512, 2048}) {
            const auto s = getri_shape(order, batch);
            EXPECT_TRUE(GetriTable::preferred(kGetriBlocked, s))
                << "float order " << order << " batch " << batch;
            EXPECT_FALSE(GetriTable::preferred(kVendorAuto, s))
                << "preferred() is asked only of NATIVE routes";
            const Route r = resolve_getri_route<float>(kGetriAuto, s, true);
            EXPECT_TRUE(is_native(r) && r.algo == Algorithm::Blocked)
                << "float order " << order << " batch " << batch;
        }
        for (int64_t order : {1, 32, 40, 64, 127}) {
            const auto s = getri_shape(order, batch);
            EXPECT_FALSE(GetriTable::preferred(kGetriBlocked, s))
                << "float order " << order << ": n=64 LOSES at 0.856 (batch 8192) "
                   "and 0.853 (batch 16384)";
            EXPECT_TRUE(is_vendor(resolve_getri_route<float>(kGetriAuto, s, true)));
        }

        // ---- cfloat: IN at 256, OUT at 255 and at 128 ---------------------
        for (int64_t order : {256, 257, 512, 2048}) {
            const auto s = getri_shape(order, batch);
            EXPECT_TRUE(GetriTableCF::preferred(kGetriBlocked, s))
                << "cfloat order " << order << " batch " << batch;
            const Route r = resolve_getri_route<std::complex<float>>(kGetriAuto, s, true);
            EXPECT_TRUE(is_native(r) && r.algo == Algorithm::Blocked);
        }
        for (int64_t order : {1, 64, 128, 129, 255}) {
            const auto s = getri_shape(order, batch);
            EXPECT_FALSE(GetriTableCF::preferred(kGetriBlocked, s))
                << "cfloat order " << order << ": n=128 is 0.71 at batch 512";
            EXPECT_TRUE(is_vendor(
                resolve_getri_route<std::complex<float>>(kGetriAuto, s, true)));
        }

        // ---- double and cdouble: NOTHING, at any order --------------------
        for (int64_t order : {1, 64, 128, 256, 512, 1024, 2048}) {
            const auto s = getri_shape(order, batch);
            EXPECT_FALSE(GetriTableD::preferred(kGetriBlocked, s))
                << "double order " << order << " earned no window";
            EXPECT_FALSE(GetriTableCD::preferred(kGetriBlocked, s))
                << "cdouble order " << order << " earned no window";
            EXPECT_TRUE(is_vendor(resolve_getri_route<double>(kGetriAuto, s, true)));
            EXPECT_TRUE(is_vendor(
                resolve_getri_route<std::complex<double>>(kGetriAuto, s, true)));
        }
    }
}

TEST(RouteGetri, AbsentDriverIsUnsupported) {
    // ABSENT DRIVER -- what this build reports today.
    const auto absent = getri_shape(64, 256, /*blocked_available=*/false);
    EXPECT_FALSE(GetriTable::supports(kGetriBlocked, absent));
    EXPECT_TRUE(is_vendor(resolve_getri_route<float>(kGetriAuto, absent, true)));
    EXPECT_TRUE(is_vendor(resolve_getri_route<float>(kGetriAuto, absent, false)));
    EXPECT_TRUE(is_vendor(resolve_getri_route<float>(kGetriBlocked, absent, true)));
    EXPECT_TRUE(is_vendor(resolve_getri_route<float>(kGetriNativeBare, absent, true)));

    // ...AND INSIDE THE WINDOW: preferred() must still say yes -- it is a SPEED
    // predicate and must not repeat the capability test -- while supports() says no.
    const auto in_window_absent = getri_shape(512, 256, /*blocked_available=*/false);
    EXPECT_TRUE(GetriTable::preferred(kGetriBlocked, in_window_absent));
    EXPECT_FALSE(GetriTable::supports(kGetriBlocked, in_window_absent));
    EXPECT_TRUE(is_vendor(resolve_getri_route<float>(kGetriAuto, in_window_absent, true)));
    EXPECT_TRUE(is_vendor(resolve_getri_route<float>(kGetriAuto, in_window_absent, false)))
        << "vendor-free with no driver must say 'needs a vendor', not invent a route";

    // A NETLIB queue is a CORRECTNESS refusal (the pivot format disagrees), and
    // the window must not override it.
    auto netlib = getri_shape(512, 256);
    netlib.backend = Backend::NETLIB;
    EXPECT_TRUE(GetriTable::preferred(kGetriBlocked, netlib));
    EXPECT_FALSE(GetriTable::supports(kGetriBlocked, netlib));
    EXPECT_TRUE(is_vendor(resolve_getri_route<float>(kGetriAuto, netlib, true)));
}

TEST(RouteGetri, BareOriginResolvesToASpecificAlgorithm) {
    const auto s = getri_shape(64, 256);
    const Route r = resolve_getri_route<float>(kGetriNativeBare, s,
                                               /*vendor_available=*/true);
    EXPECT_EQ(r.origin, Origin::Native);
    EXPECT_EQ(r.algo, Algorithm::Blocked);
    EXPECT_FALSE(GetriTable::supports(kGetriNativeBare, s))
        << "{Native, Auto} itself must never be reported supported";
}

TEST(RouteGetri, BatchlasGetriRouteIsActuallyRead) {
    ClearRouteEnv clear(Op::getri);

    EXPECT_EQ(op_env_stem(Op::getri), "GETRI");
    EXPECT_TRUE(std::string(legacy_variable_for(Op::getri)).empty())
        << "no legacy getri variable ever shipped; a case in legacy_variable_for "
           "would INVENT a legacy spelling";

    {
        const auto unset = parse_route_env(Op::getri);
        EXPECT_FALSE(unset.found);
        EXPECT_EQ(legacy_unset_default(Op::getri).origin, Origin::Auto);
    }
    {
        ScopedEnv e("BATCHLAS_GETRI_ROUTE", "blocked");
        const auto p = parse_route_env(Op::getri);
        ASSERT_TRUE(p.found) << "BATCHLAS_GETRI_ROUTE was not read at all";
        EXPECT_EQ(p.route, (Route{Origin::Native, Algorithm::Blocked}));
        EXPECT_EQ(p.source.variable, "BATCHLAS_GETRI_ROUTE");
        EXPECT_FALSE(p.source.legacy);
    }
    {
        ScopedEnv e("BATCHLAS_GETRI_ROUTE", "vendor");
        const auto p = parse_route_env(Op::getri);
        ASSERT_TRUE(p.found);
        EXPECT_EQ(p.route, (Route{Origin::Vendor, Algorithm::Auto}));
    }
    {
        ScopedEnv e("BATCHLAS_GETRI_ROUTE", "not-a-route");
        const auto p = parse_route_env(Op::getri);
        EXPECT_FALSE(p.found);
        EXPECT_TRUE(p.unparsed) << "a typo must be reported, not silently Auto";
    }
}

// ---------------------------------------------------------------------------
// THE THREE LU OPS ARE PINNED BY THREE INDEPENDENT VARIABLES, which is the
// silent-wrong-answer channel the pivot contract has to close.
// ---------------------------------------------------------------------------
TEST(RouteLuFamily, TheThreeOpsResolveIndependentlyAndThatIsThePivotHazard) {
    // The physical pivot format is BACKEND-DEPENDENT: the vendors store PACKED
    // 1-BASED INT32 in the first half of the caller's int64 buffer, while netlib
    // widens an int scratch into genuine int64. A native getrf must agree with
    // WHATEVER SERVES getri on the same call, and the mixture is reachable through
    // ordinary configuration -- three variables, three tables, no shape field able
    // to express "the op downstream of me resolved differently".
    ClearRouteEnv clear_f(Op::getrf);
    ClearRouteEnv clear_s(Op::getrs);
    ClearRouteEnv clear_i(Op::getri);

    ScopedEnv ef("BATCHLAS_GETRF_ROUTE", "cta");
    ScopedEnv ei("BATCHLAS_GETRI_ROUTE", "vendor");

    EXPECT_EQ(parse_route_env(Op::getrf).route, (Route{Origin::Native, Algorithm::CTA}));
    EXPECT_EQ(parse_route_env(Op::getri).route, (Route{Origin::Vendor, Algorithm::Auto}));
    EXPECT_FALSE(parse_route_env(Op::getrs).found)
        << "and the third is untouched -- the three do not share a variable";

    // With capabilities present that pin produces a mixed pair; today both resolve to
    // the vendor, which is why this asserts on the PARSED routes and on supports().
    const auto fs = getrf_shape(/*order=*/64, /*batch=*/128, /*cta_max_n=*/128);
    const auto is_ = getri_shape(/*order=*/64, /*batch=*/128);
    EXPECT_TRUE(GetrfTable::supports(kGetrfCta, fs));
    EXPECT_TRUE(GetriTable::supports(kGetriBlocked, is_));
    EXPECT_TRUE(is_vendor(resolve_getri_route<float>(
        Route{Origin::Vendor, Algorithm::Auto}, is_, /*vendor_available=*/true)))
        << "a pinned vendor getri reading a natively-written pivot buffer is the "
           "channel getrf_native.hh's PIVOT CONTRACT section exists to close; it "
           "needs a CROSS-OP test with the kernel, which no pure-layer case can be";
}

// ===========================================================================
// gemv. The Direct arm has NO is_gpu clause, so a native_cpu queue can take it;
// without that the vendor-free walk finds no route for a CPU queue at all.
//
// gemv_shape() sets direct_available, cta_available, has_sg32 AND is_gpu: leave
// any of them at its default and supports() is false on every shape, so every
// assertion here holds vacuously. RouteGemv.HelperIsArmed is what checks that.
// evidence: docs/perf/gemv.md
// ===========================================================================

namespace {

GemvShape gemv_shape(int64_t m, int64_t n, int64_t batch,
                     Transpose transA = Transpose::NoTrans,
                     bool is_gpu = true,
                     bool has_sg32 = true,
                     bool direct_available = true,
                     bool cta_available = true,
                     bool heterogeneous = false) {
    GemvShape s;
    s.op = Op::gemv;
    s.scalar = ScalarKind::F32;
    s.backend = Backend::AUTO;
    s.m = m;
    s.n = n;
    s.k = m;
    s.batch = batch;
    s.transA = transA;
    s.is_gpu = is_gpu;
    s.heterogeneous_batch = heterogeneous;
    s.has_sg32 = has_sg32;
    s.direct_available = direct_available;
    s.cta_available = cta_available;
    return s;
}

using GemvTable = RouteTable<Op::gemv, float>;
constexpr Route kGemvCta{Origin::Native, Algorithm::CTA};
constexpr Route kGemvDirect{Origin::Native, Algorithm::Direct};
constexpr Route kGemvNativeBare{Origin::Native, Algorithm::Auto};
constexpr Route kGemvAuto{Origin::Auto, Algorithm::Auto};

} // namespace

// THE HELPER IS ARMED. Run this first: if it fails, every other gemv assertion
// in this file is vacuous and none of them means anything.
TEST(RouteGemv, HelperIsArmed) {
    const auto on = gemv_shape(/*m=*/256, /*n=*/256, /*batch=*/512, Transpose::Trans);
    EXPECT_TRUE(GemvTable::supports(kGemvCta, on));
    EXPECT_TRUE(GemvTable::supports(kGemvDirect, on));

    // Now take each capability away in turn and require the answer to MOVE.
    const auto no_kernels = gemv_shape(256, 256, 512, Transpose::Trans,
                                       /*is_gpu=*/true, /*has_sg32=*/true,
                                       /*direct_available=*/false,
                                       /*cta_available=*/false);
    EXPECT_FALSE(GemvTable::supports(kGemvCta, no_kernels))
        << "cta_available is not reaching supports(): every CTA assertion below "
           "would hold vacuously, which is how getrs's 78/78 survived a flip";
    EXPECT_FALSE(GemvTable::supports(kGemvDirect, no_kernels))
        << "direct_available is not reaching supports()";
}

// THE DELIVERABLE, AS AN ASSERTION: adding `if (!s.is_gpu) return false;` to the
// Direct arm turns this red, and the Backend::NETLIB rows of gemv_tests.cc with it.
TEST(RouteGemv, DirectHasNoGpuGateAndThatIsTheWholeWorkPackage) {
    for (Transpose t : {Transpose::NoTrans, Transpose::Trans, Transpose::ConjTrans}) {
        const auto cpu = gemv_shape(64, 48, 6, t, /*is_gpu=*/false, /*has_sg32=*/false);
        EXPECT_TRUE(GemvTable::supports(kGemvDirect, cpu))
            << "Direct must serve a CPU device: bodies 1, 2 and 4 are serial dot "
               "products with no work-group collective and no required sub-group "
               "size. This is the line that closes half of gemv_tests.cc.";
        EXPECT_TRUE(is_native(resolve_gemv_route<float>(kGemvAuto, cpu,
                                                        /*vendor_available=*/false)))
            << "vendor-free, a CPU gemv must still find a route";
        EXPECT_EQ(resolve_gemv_route<float>(kGemvAuto, cpu, false).algo,
                  Algorithm::Direct);
    }
}

// CTA's THREE CORRECTNESS GATES, each taken away on its own so that a single
// dropped clause cannot hide behind another.
TEST(RouteGemv, CtaRequiresTransposedGpuWithAnEnumeratedSubGroup32) {
    // 1. NoTrans has no CTA body at all -- gemv_native_cta throws on it.
    EXPECT_FALSE(GemvTable::supports(
        kGemvCta, gemv_shape(256, 256, 512, Transpose::NoTrans)));
    EXPECT_TRUE(GemvTable::supports(
        kGemvCta, gemv_shape(256, 256, 512, Transpose::Trans)));
    EXPECT_TRUE(GemvTable::supports(
        kGemvCta, gemv_shape(256, 256, 512, Transpose::ConjTrans)));

    // 2. A CPU device has no sub-group to reduce over.
    EXPECT_FALSE(GemvTable::supports(
        kGemvCta, gemv_shape(256, 256, 512, Transpose::Trans, /*is_gpu=*/false,
                             /*has_sg32=*/true)));

    // 3. The body carries [[sycl::reqd_sub_group_size(32)]]; a device that does
    //    not ENUMERATE 32 aborts the launch. has_sg32 must come from
    //    sub_group_sizes, never from MAX_SUB_GROUP_SIZE.
    EXPECT_FALSE(GemvTable::supports(
        kGemvCta, gemv_shape(256, 256, 512, Transpose::Trans, /*is_gpu=*/true,
                             /*has_sg32=*/false)));
}

// A HETEROGENEOUS BATCH IS A CORRECTNESS GATE FOR GEMV: one launch covers the
// batch with a single (m, n, ld, stride) tuple, and VectorView has no active-size
// concept, so there is nothing to walk on the x and y side. Both tiers refuse it.
TEST(RouteGemv, HeterogeneousBatchIsRefusedByBothNativeTiers) {
    const auto het = gemv_shape(256, 256, 512, Transpose::Trans,
                                /*is_gpu=*/true, /*has_sg32=*/true,
                                /*direct_available=*/true, /*cta_available=*/true,
                                /*heterogeneous=*/true);
    EXPECT_FALSE(GemvTable::supports(kGemvCta, het));
    EXPECT_FALSE(GemvTable::supports(kGemvDirect, het));
    EXPECT_TRUE(is_vendor(resolve_gemv_route<float>(kGemvAuto, het, true)));
}

// float, double and cfloat earn NO window: cuBLAS gemvStridedBatched sits at
// 94-105% of the achievable DRAM roof. This case pins that the complex<double>
// clause below did not leak into the other three types -- GemvTable is the float
// table and cannot see it. evidence: docs/perf/gemv.md
TEST(RouteGemv, PreferredIsAllFalseForTheThreeTypesThatDidNotEarnAWindow) {
    using GemvTableD  = RouteTable<Op::gemv, double>;
    using GemvTableCF = RouteTable<Op::gemv, std::complex<float>>;
    const Transpose ts[3] = {Transpose::NoTrans, Transpose::Trans, Transpose::ConjTrans};
    for (Transpose t : ts) {
        for (int64_t m : {1, 4, 32, 64, 128, 256, 320, 352, 384, 1024}) {
            for (int64_t n : {8, 64, 256, 512, 2048}) {
                for (int64_t batch : {1, 128, 320, 512, 4096}) {
                    const auto s = gemv_shape(m, n, batch, t);
                    EXPECT_FALSE(GemvTable::preferred(kGemvCta, s))
                        << "float m " << m << " n " << n << " batch " << batch
                        << ": refuted at out_len 256, red_len 128, batch 512 (0.9340)";
                    EXPECT_FALSE(GemvTable::preferred(kGemvDirect, s));
                    EXPECT_FALSE(GemvTableD::preferred(kGemvCta, s))
                        << "double: refuted at out_len 512, red_len 128, batch 1024 (0.9722)";
                    EXPECT_FALSE(GemvTableCF::preferred(kGemvCta, s))
                        << "cfloat: refuted at out_len 256, red_len 48, batch 512 (0.6644)";
                }
            }
        }
    }
}

// ---- THE complex<double> TRANSPOSED WINDOW ---------------------------------
// THE AXES ARE out_len() AND red_len(), AND THEY SWAP WITH transA. gemv_shape()
// takes (m, n), so gemv_band() converts ONCE, here: under Trans and ConjTrans,
// out_len == n == cols and red_len == m == rows.
// evidence: docs/perf/gemv.md#the-cdouble-window-boundaries
namespace {
GemvShape gemv_band(int64_t out_len, int64_t red_len, int64_t batch, Transpose t) {
    return gemv_shape(/*m=*/red_len, /*n=*/out_len, batch, t);
}
using GemvTableCD = RouteTable<Op::gemv, std::complex<double>>;
} // namespace

TEST(RouteGemv, CdoubleTransposedBandIsPreferredAndEveryBoundaryIsPinned) {
    const Transpose trs[2] = {Transpose::Trans, Transpose::ConjTrans};

    // INSIDE, both transposed spellings: ortho.cc issues ConjTrans for every complex
    // type, so pinning only Trans would guard the wrong half of the clause.
    for (Transpose t : trs) {
        for (int64_t out : {256, 512, 1024, 4096}) {
            for (int64_t red : {64, 128, 256, 352}) {
                for (int64_t b : {320, 512, 4096}) {
                    const auto s = gemv_band(out, red, b, t);
                    EXPECT_TRUE(GemvTableCD::preferred(kGemvCta, s))
                        << "out_len " << out << " red_len " << red << " batch " << b;
                    EXPECT_FALSE(GemvTableCD::preferred(kGemvDirect, s))
                        << "the Direct tier must never be preferred by this clause: "
                           "the measurement is of the CTA kernel";
                    EXPECT_FALSE(GemvTableCD::preferred(kVendorAuto, s));
                    const Route r = resolve_gemv_route<std::complex<double>>(
                        kGemvAuto, s, /*vendor_available=*/true);
                    EXPECT_TRUE(is_native(r) && r.algo == Algorithm::CTA)
                        << "out_len " << out << " red_len " << red << " batch " << b;
                }
            }
        }
    }

    // ---- EVERY BOUNDARY, FROM BOTH SIDES, EACH WITH ITS MEASURED CELL ------
    for (Transpose t : trs) {
        // red_len lower edge: 64 in, 63 out.
        EXPECT_TRUE (GemvTableCD::preferred(kGemvCta, gemv_band(512,  64, 512, t)));
        EXPECT_FALSE(GemvTableCD::preferred(kGemvCta, gemv_band(512,  63, 512, t)));
        EXPECT_FALSE(GemvTableCD::preferred(kGemvCta, gemv_band(512,  48, 512, t)));
        EXPECT_FALSE(GemvTableCD::preferred(kGemvCta, gemv_band(512,  32, 512, t)));
        // red_len upper edge: 352 in, 353 out.
        EXPECT_TRUE (GemvTableCD::preferred(kGemvCta, gemv_band(512, 352, 512, t)));
        EXPECT_FALSE(GemvTableCD::preferred(kGemvCta, gemv_band(512, 353, 512, t)));
        EXPECT_FALSE(GemvTableCD::preferred(kGemvCta, gemv_band(512, 384, 512, t)));
        // out_len lower edge: 256 in, 255 out.
        EXPECT_TRUE (GemvTableCD::preferred(kGemvCta, gemv_band(256, 128, 512, t)));
        EXPECT_FALSE(GemvTableCD::preferred(kGemvCta, gemv_band(255, 128, 512, t)));
        EXPECT_FALSE(GemvTableCD::preferred(kGemvCta, gemv_band(192, 128, 512, t)));
        // batch floor: 320 in, 319 out -- cuBLAS's own kernel-selection threshold,
        // not a fitted constant.
        EXPECT_TRUE (GemvTableCD::preferred(kGemvCta, gemv_band(512, 128, 320, t)));
        EXPECT_FALSE(GemvTableCD::preferred(kGemvCta, gemv_band(512, 128, 319, t)));
        EXPECT_FALSE(GemvTableCD::preferred(kGemvCta, gemv_band(512, 128, 256, t)));
        EXPECT_FALSE(GemvTableCD::preferred(kGemvCta, gemv_band(512, 128, 128, t)));
        for (int64_t b : {1, 2, 128, 256, 319}) {
            EXPECT_TRUE(is_vendor(resolve_gemv_route<std::complex<double>>(
                kGemvAuto, gemv_band(512, 128, b, t), true)))
                << "below the batch floor the vendor must still take it, batch " << b;
        }
    }

    // NoTrans IS EXCLUDED by the clause itself, not only by supports(): under NoTrans
    // out_len and red_len SWAP, so the same extents are a different shape entirely.
    for (int64_t out : {256, 512, 1024}) {
        for (int64_t red : {64, 128, 256}) {
            const auto s = gemv_shape(/*m=*/out, /*n=*/red, 512, Transpose::NoTrans);
            EXPECT_FALSE(GemvTableCD::preferred(kGemvCta, s));
            EXPECT_FALSE(GemvTableCD::preferred(kGemvDirect, s));
            EXPECT_TRUE(is_vendor(resolve_gemv_route<std::complex<double>>(
                kGemvAuto, s, true)));
        }
    }
}

// THE CLAUSE READS red_len() AND out_len(), NOT m AND n -- PROVED BY TRANSPOSING A
// SINGLE SHAPE AND REQUIRING THE ANSWER TO MOVE. A predicate spelled on m and n
// returns the same answer for both and this case would stay green.
TEST(RouteGemv, TheBandIsOnRedLenAndInvertsUnderNoTrans) {
    auto A = gemv_shape(/*m=*/128, /*n=*/512, /*batch=*/512, Transpose::Trans);
    EXPECT_EQ(A.red_len(), 128);
    EXPECT_EQ(A.out_len(), 512);
    EXPECT_TRUE(GemvTableCD::preferred(kGemvCta, A));

    auto B = A; B.transA = Transpose::NoTrans;
    EXPECT_EQ(B.red_len(), 512);
    EXPECT_EQ(B.out_len(), 128);
    EXPECT_FALSE(GemvTableCD::preferred(kGemvCta, B));

    // And the mirror image: 512 rows by 128 columns is OUTSIDE transposed although
    // its m and n are exactly A's swapped.
    auto C = gemv_shape(/*m=*/512, /*n=*/128, /*batch=*/512, Transpose::Trans);
    EXPECT_EQ(C.red_len(), 512);
    EXPECT_EQ(C.out_len(), 128);
    EXPECT_FALSE(GemvTableCD::preferred(kGemvCta, C));
}

// THE WINDOW IS NOT A CORRECTNESS GATE, AND A CAPABILITY THE BUILD DOES NOT HAVE
// STILL WINS OVER IT: inside the band on a device that does not enumerate 32, or
// with no CTA kernel linked, supports() must refuse and the vendor must take it.
TEST(RouteGemv, TheWindowNeverOutrunsTheCapability) {
    const auto ok = gemv_band(512, 128, 512, Transpose::Trans);
    EXPECT_TRUE(GemvTableCD::preferred(kGemvCta, ok));
    EXPECT_TRUE(GemvTableCD::supports(kGemvCta, ok));

    auto nosg = gemv_shape(128, 512, 512, Transpose::Trans, /*is_gpu=*/true,
                           /*has_sg32=*/false);
    EXPECT_TRUE(GemvTableCD::preferred(kGemvCta, nosg))
        << "preferred() must NOT repeat the capability test -- a pinned "
           "native:cta on such a device has to fall through supports(), not "
           "silently resolve elsewhere for the wrong reason";
    EXPECT_FALSE(GemvTableCD::supports(kGemvCta, nosg));
    EXPECT_TRUE(is_vendor(resolve_gemv_route<std::complex<double>>(kGemvAuto, nosg, true)));

    auto nocta = gemv_shape(128, 512, 512, Transpose::Trans, true, true,
                            /*direct_available=*/true, /*cta_available=*/false);
    EXPECT_FALSE(GemvTableCD::supports(kGemvCta, nocta));
    EXPECT_TRUE(is_vendor(resolve_gemv_route<std::complex<double>>(kGemvAuto, nocta, true)));
    // ...and vendor-free it must still find the Direct arm rather than nothing.
    const Route f = resolve_gemv_route<std::complex<double>>(kGemvAuto, nocta, false);
    EXPECT_TRUE(is_native(f) && f.algo == Algorithm::Direct);
}

// AN AUTO GEMV IS THE VENDOR WHEREVER THERE IS ONE AND THE NATIVE LADDER WHEREVER
// THERE IS NOT. Both halves matter: route-neutrality, and the vendor-free burn-down.
TEST(RouteGemv, AutoTakesTheVendorWhenPresentAndTheLadderWhenNot) {
    const auto tr = gemv_shape(256, 256, 512, Transpose::Trans);
    const auto no = gemv_shape(256, 256, 512, Transpose::NoTrans);

    EXPECT_TRUE(is_vendor(resolve_gemv_route<float>(kGemvAuto, tr, true)));
    EXPECT_TRUE(is_vendor(resolve_gemv_route<float>(kGemvAuto, no, true)));

    // Vendor-free: the ladder is CTA, then Direct, then the vendor -- tightest first,
    // so a transposed GPU shape gets the coalesced body and everything else Direct.
    EXPECT_EQ(resolve_gemv_route<float>(kGemvAuto, tr, false).algo, Algorithm::CTA);
    EXPECT_EQ(resolve_gemv_route<float>(kGemvAuto, no, false).algo, Algorithm::Direct);
}

// A BARE `native` DOES NOT MEAN CTA: the walk picks the first SUPPORTED native
// route, so a bare pin lands on Direct for NoTrans, for a CPU device and for a GPU
// without an enumerated 32.
TEST(RouteGemv, BareNativeResolvesToTheFirstSUPPORTEDRouteNotToCta) {
    EXPECT_EQ(resolve_gemv_route<float>(
                  kGemvNativeBare, gemv_shape(256, 256, 512, Transpose::Trans), true).algo,
              Algorithm::CTA);
    EXPECT_EQ(resolve_gemv_route<float>(
                  kGemvNativeBare, gemv_shape(256, 256, 512, Transpose::NoTrans), true).algo,
              Algorithm::Direct);
    EXPECT_EQ(resolve_gemv_route<float>(
                  kGemvNativeBare,
                  gemv_shape(256, 256, 512, Transpose::Trans, /*is_gpu=*/false,
                             /*has_sg32=*/false), true).algo,
              Algorithm::Direct);
    EXPECT_EQ(resolve_gemv_route<float>(
                  kGemvNativeBare,
                  gemv_shape(256, 256, 512, Transpose::Trans, /*is_gpu=*/true,
                             /*has_sg32=*/false), true).algo,
              Algorithm::Direct);
}

// PINNING A ROUTE THE SHAPE CANNOT TAKE IS SILENT, AND ITS OUTCOME DEPENDS ON THE
// BUILD. Not a bug asserted as correct -- it is the standing fall-through
// behaviour -- but it has cost time twice, so it is written down here.
TEST(RouteGemv, PinningCtaOnAShapeCtaCannotServeFallsThroughSilently) {
    const auto no = gemv_shape(256, 256, 512, Transpose::NoTrans);

    const Route with_vendor = resolve_gemv_route<float>(kGemvCta, no, true);
    EXPECT_TRUE(is_vendor(with_vendor))
        << "vendor-present, a pin CTA cannot serve resolves to the VENDOR -- not "
           "to native:direct, and with no diagnostic";

    const Route without_vendor = resolve_gemv_route<float>(kGemvCta, no, false);
    EXPECT_TRUE(is_native(without_vendor));
    EXPECT_EQ(without_vendor.algo, Algorithm::Direct)
        << "vendor-free, the SAME pin lands on native:direct: the outcome of a "
           "pin is build-dependent, so only the resolved-route column can tell "
           "you which arm actually ran";
}

// out_len() AND red_len() SWAP WITH transA. The one measured cuBLAS slow region is
// a band on **m**, which under a transposed transA is red_len(), NOT out_len(); a
// predicate written on out_len() would test n and invert the window.
TEST(RouteGemv, OutLenAndRedLenSwapWithTransA) {
    const auto no = gemv_shape(/*m=*/64, /*n=*/2048, /*batch=*/1, Transpose::NoTrans);
    EXPECT_EQ(no.out_len(), 64);
    EXPECT_EQ(no.red_len(), 2048);
    for (Transpose t : {Transpose::Trans, Transpose::ConjTrans}) {
        const auto tr = gemv_shape(/*m=*/64, /*n=*/2048, /*batch=*/1, t);
        EXPECT_EQ(tr.out_len(), 2048);
        EXPECT_EQ(tr.red_len(), 64)
            << "under a transposed transA the reduction runs over m; a clause "
               "written on out_len() tests n and inverts the measured window";
    }
}

// GemvShape must not re-declare transA or is_gpu: resolve_route SLICES it to
// OpShape on the way into the coverage table, so a shadowing member would be
// written by the builder and then not copied, collapsing the two arms -- different
// KERNELS, not different flags -- into one first-writer-wins row.
TEST(RouteGemv, ShapeDoesNotShadowOpShapeFields) {
    GemvShape s = gemv_shape(64, 48, 6, Transpose::ConjTrans, /*is_gpu=*/true);
    const OpShape& sliced = static_cast<const OpShape&>(s);
    EXPECT_EQ(&s.transA, &sliced.transA)
        << "GemvShape re-declares transA: every gemv coverage row would report "
           "NoTrans and the two arms would collapse into one row";
    EXPECT_EQ(&s.is_gpu, &sliced.is_gpu);
    EXPECT_EQ(sliced.transA, Transpose::ConjTrans);
    EXPECT_TRUE(sliced.is_gpu);
}

// DEGENERATE EXTENTS. m == 0 or n == 0 is a LEGAL call the native kernel serves by
// quick-returning without touching y, so it must stay SUPPORTED. A negative extent
// or an empty batch has no launch geometry and goes to the vendor.
TEST(RouteGemv, ZeroExtentIsSupportedButNegativeExtentIsNot) {
    EXPECT_TRUE(GemvTable::supports(kGemvDirect, gemv_shape(0, 6, 3, Transpose::Trans)));
    EXPECT_TRUE(GemvTable::supports(kGemvDirect, gemv_shape(5, 0, 3, Transpose::NoTrans)));
    EXPECT_FALSE(GemvTable::supports(kGemvDirect, gemv_shape(-1, 6, 3)));
    EXPECT_FALSE(GemvTable::supports(kGemvDirect, gemv_shape(6, -1, 3)));
    EXPECT_FALSE(GemvTable::supports(kGemvDirect, gemv_shape(6, 6, 0)));
    EXPECT_TRUE(is_vendor(resolve_gemv_route<float>(kGemvAuto, gemv_shape(6, 6, 0), true)));
}

// Algorithm::Auto IS NOT A NATIVE GEMV ROUTE: gemv has two native tiers, so a bare
// "native" names neither and supports() must say false, or the walk would stop on
// a route with no kernel behind it.
TEST(RouteGemv, NativeAutoIsNotItselfASupportedRoute) {
    const auto s = gemv_shape(256, 256, 512, Transpose::Trans);
    EXPECT_FALSE(GemvTable::supports(kGemvNativeBare, s));
    EXPECT_FALSE(GemvTable::supports(Route{Origin::Native, Algorithm::Blocked}, s));
    EXPECT_TRUE(GemvTable::supports(kVendorAuto, s));
}

// THE ORDER ARRAY IS A CAPABILITY LADDER, TIGHTEST FIRST, and it is the only thing
// that decides the vendor-free outcome while preferred() is all-false.
TEST(RouteGemv, OrderIsCtaThenDirectThenVendor) {
    ASSERT_EQ(GemvTable::order_end() - GemvTable::order_begin(), 3);
    EXPECT_EQ(GemvTable::order_begin()[0].origin, Origin::Native);
    EXPECT_EQ(GemvTable::order_begin()[0].algo, Algorithm::CTA);
    EXPECT_EQ(GemvTable::order_begin()[1].origin, Origin::Native);
    EXPECT_EQ(GemvTable::order_begin()[1].algo, Algorithm::Direct);
    EXPECT_TRUE(is_vendor(GemvTable::order_begin()[2]));
}

// ===========================================================================
// spmm. The Direct arm has NO is_gpu clause, so a native_cpu queue can take the
// route -- build-novendor's Backend::NETLIB rows depend on it.
//
// TWO CAPABILITY FLAGS, NOT ONE, and they are not interchangeable: transA ==
// NoTrans is served by the gather body, transA != NoTrans by the scale+scatter
// PAIR. They are separate kernels, so a build can have one and not the other, and
// supports() consults exactly the flag for the body that would actually run.
//
// spmm_shape() sets format, gather_available AND scatter_available: leave any at
// its default and supports() is false on every shape, so every assertion here
// holds vacuously. RouteSpmm.HelperIsArmed is what checks that.
// evidence: docs/perf/spmm.md
// ===========================================================================

namespace {

SpmmShape spmm_shape(int64_t m, int64_t k, int64_t nrhs, int64_t batch,
                     Transpose transA = Transpose::NoTrans,
                     Transpose transB = Transpose::NoTrans,
                     MatrixFormat format = MatrixFormat::CSR,
                     bool is_gpu = true,
                     bool gather_available = true,
                     bool scatter_available = true,
                     bool heterogeneous = false) {
    SpmmShape s;
    s.op = Op::spmm;
    s.scalar = ScalarKind::F32;
    s.backend = Backend::AUTO;
    // THE FIELD MAPPING, ONCE, HERE: m = A.rows(), k = A.cols(), n = C.cols(), i.e.
    // nrhs. Which of m and k is the OUTPUT extent swaps with transA, which is why
    // the shape carries out_rows() and red_rows().
    s.m = m;
    s.k = k;
    s.n = nrhs;
    s.batch = batch;
    s.transA = transA;
    s.transB = transB;
    s.is_gpu = is_gpu;
    s.heterogeneous_batch = heterogeneous;
    s.format = format;
    s.gather_available = gather_available;
    s.scatter_available = scatter_available;
    return s;
}

using SpmmTable = RouteTable<Op::spmm, float>;
constexpr Route kSpmmDirect{Origin::Native, Algorithm::Direct};
constexpr Route kSpmmNativeBare{Origin::Native, Algorithm::Auto};
constexpr Route kSpmmCta{Origin::Native, Algorithm::CTA};
constexpr Route kSpmmAuto{Origin::Auto, Algorithm::Auto};

constexpr Transpose kAllTrans[3] = {Transpose::NoTrans, Transpose::Trans,
                                    Transpose::ConjTrans};

} // namespace

// THE HELPER IS ARMED. RUN THIS FIRST: if it fails, every other spmm assertion in
// this file is vacuous. Four gates, taken away ONE AT A TIME, each required to MOVE
// the answer -- taking them away together would not distinguish which one reaches
// supports().
TEST(RouteSpmm, HelperIsArmed) {
    const auto gather  = spmm_shape(/*m=*/4096, /*k=*/4096, /*nrhs=*/25, /*batch=*/64);
    const auto scatter = spmm_shape(4096, 4096, 25, 64, Transpose::Trans);
    ASSERT_TRUE(SpmmTable::supports(kSpmmDirect, gather))
        << "the baseline shape is not even supported: nothing below can be "
           "distinguished from a table that refuses everything";
    ASSERT_TRUE(SpmmTable::supports(kSpmmDirect, scatter));

    // 1. gather_available, which serves transA == NoTrans and nothing else.
    const auto no_gather = spmm_shape(4096, 4096, 25, 64, Transpose::NoTrans,
                                      Transpose::NoTrans, MatrixFormat::CSR,
                                      /*is_gpu=*/true, /*gather_available=*/false,
                                      /*scatter_available=*/true);
    EXPECT_FALSE(SpmmTable::supports(kSpmmDirect, no_gather))
        << "gather_available is not reaching supports(): every NoTrans assertion "
           "below would hold for the wrong reason, which is how getrs's 78/78 "
           "survived a capability flip";

    // 2. scatter_available, which serves transA != NoTrans and nothing else.
    const auto no_scatter = spmm_shape(4096, 4096, 25, 64, Transpose::Trans,
                                       Transpose::NoTrans, MatrixFormat::CSR,
                                       /*is_gpu=*/true, /*gather_available=*/true,
                                       /*scatter_available=*/false);
    EXPECT_FALSE(SpmmTable::supports(kSpmmDirect, no_scatter))
        << "scatter_available is not reaching supports()";

    // 3. format. The helper defaults to CSR; if it did not, or if the gate were
    //    dropped, a Dense view would reach a CSR kernel -- a wrong answer.
    auto dense = gather;
    dense.format = MatrixFormat::Dense;
    EXPECT_FALSE(SpmmTable::supports(kSpmmDirect, dense))
        << "format is not reaching supports()";

    // 4. heterogeneous_batch, which OpShape carries and the helper writes.
    auto het = gather;
    het.heterogeneous_batch = true;
    EXPECT_FALSE(SpmmTable::supports(kSpmmDirect, het))
        << "heterogeneous_batch is not reaching supports()";
}

// THE DELIVERABLE, AS AN ASSERTION: adding `if (!s.is_gpu) return false;` to the
// Direct arm turns this red. It is asserted for all nine transpose pairs because
// all three bodies are plain loops -- no local memory, no group collective, no
// required sub-group size -- so a gate added to only one half would still be fatal.
TEST(RouteSpmm, NoGpuGateOnDirect) {
    for (Transpose ta : kAllTrans) {
        for (Transpose tb : kAllTrans) {
            const auto cpu = spmm_shape(/*m=*/512, /*k=*/512, /*nrhs=*/2, /*batch=*/8,
                                        ta, tb, MatrixFormat::CSR, /*is_gpu=*/false);
            EXPECT_TRUE(SpmmTable::supports(kSpmmDirect, cpu))
                << "Direct must serve a CPU device: the gather, scale and scatter "
                   "bodies use no local memory and no group collective. This is "
                   "the line that closes the Backend::NETLIB half of the "
                   "burn-down, where the spmm symbol exists and throws today.";
            const Route r = resolve_spmm_route<float>(kSpmmAuto, cpu,
                                                      /*vendor_available=*/false);
            EXPECT_TRUE(is_native(r)) << "vendor-free, a CPU spmm must still find a route";
            EXPECT_EQ(r.algo, Algorithm::Direct);
        }
    }
}

// ALL NINE (transA, transB) COMBINATIONS ARE SERVED, which keeps the transB ==
// Trans layout lever available: a caller holding B in the other layout can pass it
// transposed instead of materialising a copy.
TEST(RouteSpmm, AllNineTransposeCombinationsSupported) {
    for (Transpose ta : kAllTrans) {
        for (Transpose tb : kAllTrans) {
            const auto s = spmm_shape(4096, 2048, 25, 128, ta, tb);
            EXPECT_TRUE(SpmmTable::supports(kSpmmDirect, s))
                << "transA " << static_cast<int>(ta) << " transB "
                << static_cast<int>(tb);
            EXPECT_EQ(resolve_spmm_route<float>(kSpmmAuto, s,
                                                /*vendor_available=*/false).algo,
                      Algorithm::Direct)
                << "transA " << static_cast<int>(ta) << " transB "
                << static_cast<int>(tb);
        }
    }
}

// THE TWO FLAGS ARE INDEPENDENT AND SERVE DISJOINT HALVES OF THE transA AXIS. A
// table that ORed them would pass a shape to a kernel this build does not contain:
// selectable-but-unimplemented rather than unsupported. transB is swept inside both
// halves because it must NOT influence the choice.
TEST(RouteSpmm, GatherAndScatterUseDifferentCapabilities) {
    for (Transpose tb : kAllTrans) {
        // Gather only: NoTrans is served, the two transposed spellings are not.
        const auto g_no = spmm_shape(1024, 1024, 12, 64, Transpose::NoTrans, tb,
                                     MatrixFormat::CSR, /*is_gpu=*/true,
                                     /*gather_available=*/true,
                                     /*scatter_available=*/false);
        EXPECT_TRUE(SpmmTable::supports(kSpmmDirect, g_no));
        for (Transpose ta : {Transpose::Trans, Transpose::ConjTrans}) {
            auto g_tr = g_no; g_tr.transA = ta;
            EXPECT_FALSE(SpmmTable::supports(kSpmmDirect, g_tr))
                << "with no scatter body linked, a transposed spmm must be "
                   "UNSUPPORTED rather than selected and then absent";
            EXPECT_TRUE(is_vendor(resolve_spmm_route<float>(kSpmmAuto, g_tr, true)));
        }

        // Scatter only: exactly the inverse.
        const auto s_no = spmm_shape(1024, 1024, 12, 64, Transpose::NoTrans, tb,
                                     MatrixFormat::CSR, /*is_gpu=*/true,
                                     /*gather_available=*/false,
                                     /*scatter_available=*/true);
        EXPECT_FALSE(SpmmTable::supports(kSpmmDirect, s_no))
            << "with no gather body linked, NoTrans must be UNSUPPORTED -- the "
               "scatter flag says nothing about the NoTrans body";
        for (Transpose ta : {Transpose::Trans, Transpose::ConjTrans}) {
            auto s_tr = s_no; s_tr.transA = ta;
            EXPECT_TRUE(SpmmTable::supports(kSpmmDirect, s_tr));
        }
    }
}

// ONLY CSR HAS BODIES, and this is a correctness gate: a Dense or COO view reaching
// a CSR kernel reads row offsets that are not there, which is a wrong answer or a
// fault, not a slow route.
TEST(RouteSpmm, NonCsrFormatRefused) {
    for (MatrixFormat f : {MatrixFormat::Dense, MatrixFormat::CSC, MatrixFormat::COO,
                           MatrixFormat::SELL, MatrixFormat::BSR,
                           MatrixFormat::BLOCKED_ELL}) {
        for (Transpose ta : kAllTrans) {
            const auto s = spmm_shape(1024, 1024, 12, 64, ta, Transpose::NoTrans, f);
            EXPECT_FALSE(SpmmTable::supports(kSpmmDirect, s))
                << "format " << static_cast<int>(f);
            EXPECT_TRUE(SpmmTable::supports(kVendorAuto, s));
            EXPECT_TRUE(is_vendor(resolve_spmm_route<float>(kSpmmAuto, s, true)));
            // And vendor-free there is nothing to fall back to, so the resolver
            // returns the vendor as its honest "this needs one" signal.
            EXPECT_TRUE(is_vendor(resolve_spmm_route<float>(kSpmmAuto, s, false)));
        }
    }
    // The CSR control, so this case cannot pass by refusing everything.
    EXPECT_TRUE(SpmmTable::supports(
        kSpmmDirect, spmm_shape(1024, 1024, 12, 64, Transpose::NoTrans,
                                Transpose::NoTrans, MatrixFormat::CSR)));
}

// A HETEROGENEOUS BATCH IS A CORRECTNESS GATE: one launch covers the batch with a
// single (ld, stride) tuple per DENSE operand. Per-item variation on the SPARSE
// side is expressible only as nnz(b), which every body handles through the
// row-offset array, so this gate can only ever fire on the dense operands.
TEST(RouteSpmm, HeterogeneousBatchRefused) {
    for (Transpose ta : kAllTrans) {
        const auto het = spmm_shape(1024, 1024, 12, 64, ta, Transpose::NoTrans,
                                    MatrixFormat::CSR, /*is_gpu=*/true,
                                    /*gather_available=*/true,
                                    /*scatter_available=*/true,
                                    /*heterogeneous=*/true);
        EXPECT_FALSE(SpmmTable::supports(kSpmmDirect, het));
        EXPECT_TRUE(is_vendor(resolve_spmm_route<float>(kSpmmAuto, het, true)));
    }
}

// DEGENERATE EXTENTS, AND HALF OF A CONTRACT WITH THE LAUNCHER. m == 0 and n == 0
// are LEGAL calls that stay SUPPORTED, so spmm_native_csr MUST quick-return on the
// HOST -- before any submit -- when out_rows == 0 || nrhs == 0 || batch <= 0. A
// NEGATIVE extent or an empty batch has no launch geometry and is refused.
TEST(RouteSpmm, ZeroExtentsAreSupportedNegativeAreNot) {
    EXPECT_TRUE(SpmmTable::supports(kSpmmDirect, spmm_shape(0, 512, 12, 8)));
    EXPECT_TRUE(SpmmTable::supports(kSpmmDirect, spmm_shape(512, 512, 0, 8)));
    EXPECT_TRUE(SpmmTable::supports(kSpmmDirect, spmm_shape(512, 0, 12, 8)));
    // ...and under a transposed transA, where m and k swap roles, both still
    // stand: out_rows() is the one that is zero in the second call.
    EXPECT_TRUE(SpmmTable::supports(
        kSpmmDirect, spmm_shape(0, 512, 12, 8, Transpose::Trans)));
    EXPECT_TRUE(SpmmTable::supports(
        kSpmmDirect, spmm_shape(512, 0, 12, 8, Transpose::Trans)));

    EXPECT_FALSE(SpmmTable::supports(kSpmmDirect, spmm_shape(-1, 512, 12, 8)));
    EXPECT_FALSE(SpmmTable::supports(kSpmmDirect, spmm_shape(512, -1, 12, 8)));
    EXPECT_FALSE(SpmmTable::supports(kSpmmDirect, spmm_shape(512, 512, -1, 8)));
    EXPECT_FALSE(SpmmTable::supports(kSpmmDirect, spmm_shape(512, 512, 12, 0)));
    EXPECT_FALSE(SpmmTable::supports(kSpmmDirect, spmm_shape(512, 512, 12, -1)));
    EXPECT_TRUE(is_vendor(
        resolve_spmm_route<float>(kSpmmAuto, spmm_shape(512, 512, 12, 0), true)));
}

// Algorithm::Auto IS NOT ITSELF A NATIVE SPMM ROUTE: a bare `native` names no body,
// so supports() must say false. It still RESOLVES, to the one native route there
// is, and both halves are asserted.
TEST(RouteSpmm, BareNativeAutoIsNotSupported) {
    const auto s = spmm_shape(4096, 4096, 25, 64);
    EXPECT_FALSE(SpmmTable::supports(kSpmmNativeBare, s));
    EXPECT_FALSE(SpmmTable::supports(kSpmmCta, s))
        << "there is no CTA body; a pin naming one must be unsupported, not "
           "selectable";
    EXPECT_FALSE(SpmmTable::supports(Route{Origin::Native, Algorithm::Blocked}, s));
    EXPECT_TRUE(SpmmTable::supports(kVendorAuto, s));

    for (Transpose ta : kAllTrans) {
        auto t = s; t.transA = ta;
        const Route r = resolve_spmm_route<float>(kSpmmNativeBare, t, true);
        EXPECT_TRUE(is_native(r));
        EXPECT_EQ(r.algo, Algorithm::Direct)
            << "a bare `native` must land on the one route that has a body";
    }
}

// ===========================================================================
// THE SHIPPED preferred() CLAUSE, in words:
//
//     preferred(r, s) ==   is_native(r)
//                       && r.algo == Algorithm::Direct
//                       && s.format == MatrixFormat::CSR
//                       && s.transA == Transpose::NoTrans
//                       && !(T is complex<float> && s.transB != NoTrans)
//
// and NOTHING else -- no batch, extent, is_gpu or nnz term. Each case below pins
// the predicate AND the resolved route, because the two come apart: a preferred()
// that answered true for {Vendor, Auto} would make the native route unreachable
// while every predicate-only assertion still passed.
// evidence: docs/perf/spmm.md#the-evidence-for-each-boundary
// ===========================================================================

namespace {

// The three sibling tables, named once. float is `SpmmTable` above.
using SpmmTableD  = RouteTable<Op::spmm, double>;
using SpmmTableCF = RouteTable<Op::spmm, std::complex<float>>;
using SpmmTableCD = RouteTable<Op::spmm, std::complex<double>>;

constexpr Route kSpmmBlocked{Origin::Native, Algorithm::Blocked};

} // namespace

// THE CLAUSE ACCEPTS THE GATHER FOR ALL FOUR TYPES, across the measured grid and
// beyond it: the clause carries no extent or batch term, so a future one must turn
// this red rather than pass unnoticed.
TEST(RouteSpmm, PreferredAcceptsTheGatherForEveryType) {
    for (int64_t m : {1, 64, 512, 1024, 2048, 4096, 65536}) {
        for (int64_t nrhs : {1, 2, 3, 12, 25, 50}) {
            for (int64_t batch : {1, 2, 4, 8, 64, 128, 512, 4096}) {
                const auto s = spmm_shape(m, m, nrhs, batch);
                EXPECT_TRUE(SpmmTable::preferred(kSpmmDirect, s))
                    << "float m " << m << " nrhs " << nrhs << " batch " << batch;
                EXPECT_TRUE(SpmmTableD::preferred(kSpmmDirect, s));
                EXPECT_TRUE(SpmmTableCD::preferred(kSpmmDirect, s));
                EXPECT_TRUE(SpmmTableCF::preferred(kSpmmDirect, s))
                    << "complex<float> is refused only with transB != NoTrans";

                // ...and the decision that follows, WITH a vendor present.
                EXPECT_TRUE(is_native(resolve_spmm_route<float>(kSpmmAuto, s, true)))
                    << "float m " << m << " nrhs " << nrhs << " batch " << batch
                    << ": the gather must now win against a present vendor";
                EXPECT_EQ(resolve_spmm_route<float>(kSpmmAuto, s, true).algo,
                          Algorithm::Direct);
            }
        }
    }

    // A RECTANGULAR SPREAD TOO, because m == k above would hide a clause written
    // on one extent and read on the other.
    for (int64_t m : {512, 4096}) {
        for (int64_t k : {64, 8192}) {
            const auto s = spmm_shape(m, k, 25, 128);
            EXPECT_TRUE(SpmmTable::preferred(kSpmmDirect, s)) << "m " << m << " k " << k;
            EXPECT_TRUE(SpmmTableD::preferred(kSpmmDirect, s));
            EXPECT_TRUE(SpmmTableCD::preferred(kSpmmDirect, s));
        }
    }
}

// THE BATCH AXIS HAS NO FLOOR, AND ITS ABSENCE IS A MEASURED DECISION: no cell at
// batch <= 64 exceeds the 1.10 gate, so a floor has no measured non-winner to
// bracket it. evidence: docs/perf/spmm.md#the-batch-axis-has-no-floor
TEST(RouteSpmm, PreferredHasNoBatchFloor) {
    for (int64_t batch : {1, 2, 4, 8, 16, 32, 64, 127, 128, 129, 512, 4096}) {
        const auto s = spmm_shape(/*m=*/4096, /*k=*/4096, /*nrhs=*/50, batch);
        EXPECT_TRUE(SpmmTable::preferred(kSpmmDirect, s))
            << "batch " << batch << ": the clause carries NO batch term. If a "
               "floor was just added, it needs a measured non-winner outside "
               "the 1.10 gate to bracket it -- docs/perf/spmm.md#raw-evidence"
               "smallbatch.txt has none at any rung, worst 1.078 at batch 4";
        EXPECT_TRUE(SpmmTableCD::preferred(kSpmmDirect, s)) << "batch " << batch;
        EXPECT_TRUE(is_native(resolve_spmm_route<float>(kSpmmAuto, s, true)))
            << "batch " << batch;
    }
}

// THE TRANSPOSED REFUSAL, WHICH IS MEASURED AND NOT AN OMISSION, and every narrower
// candidate was refuted too. Both sides of the boundary are asserted on the SAME
// shape, so this cannot pass by refusing everything.
// evidence: docs/perf/spmm.md#the-gather-window
TEST(RouteSpmm, PreferredRefusesEveryTransposedA) {
    for (int64_t nrhs : {1, 2, 4, 12, 25, 50}) {
        for (int64_t batch : {8, 128, 512, 1024}) {
            const auto gather = spmm_shape(2048, 2048, nrhs, batch);
            ASSERT_TRUE(SpmmTable::preferred(kSpmmDirect, gather))
                << "the NoTrans control must be preferred or this case is "
                   "passing by refusing everything";

            for (Transpose ta : {Transpose::Trans, Transpose::ConjTrans}) {
                for (Transpose tb : kAllTrans) {
                    const auto s = spmm_shape(2048, 2048, nrhs, batch, ta, tb);
                    EXPECT_TRUE(SpmmTable::supports(kSpmmDirect, s))
                        << "the scatter stays SUPPORTED -- the refusal is a "
                           "speed decision, not a correctness one, and "
                           "BATCHLAS_SPMM_ROUTE=native must still reach it";
                    EXPECT_FALSE(SpmmTable::preferred(kSpmmDirect, s))
                        << "transA " << static_cast<int>(ta) << " nrhs " << nrhs
                        << " batch " << batch;
                    EXPECT_FALSE(SpmmTableD::preferred(kSpmmDirect, s));
                    EXPECT_FALSE(SpmmTableCF::preferred(kSpmmDirect, s));
                    EXPECT_FALSE(SpmmTableCD::preferred(kSpmmDirect, s))
                        << "complex<double> is the WORST scatter cell measured "
                           "(3.011 at m=4096 nnz/row=16 nrhs=50 b=512)";
                    EXPECT_TRUE(is_vendor(resolve_spmm_route<float>(kSpmmAuto, s, true)))
                        << "vendor-present, a transposed spmm must still go to "
                           "the vendor";
                    // ...and vendor-FREE it must still reach the native bodies:
                    // un-preferred is not unsupported.
                    const Route free_route =
                        resolve_spmm_route<float>(kSpmmAuto, s, false);
                    EXPECT_TRUE(is_native(free_route));
                    EXPECT_EQ(free_route.algo, Algorithm::Direct);
                }
            }
        }
    }
}

// THE ONE TYPE-CONDITIONAL BOUNDARY, ASSERTED FROM ALL FOUR SIDES. The exclusion is
// (type AND transB) TOGETHER -- drop either half of the conjunction and one of these
// goes red. It is deliberately NOT narrowed by nrhs: the threshold is a property of
// the BANDED column pattern, which SpmmShape has no field for and cannot acquire.
// evidence: docs/perf/spmm.md#the-cfloat-transb-exclusion
TEST(RouteSpmm, PreferredRefusesComplexFloatWithTransposedB) {
    for (int64_t nrhs : {1, 2, 8, 12, 16, 17, 25, 32, 50}) {
        for (int64_t batch : {1, 4, 128, 512}) {
            for (Transpose tb : {Transpose::Trans, Transpose::ConjTrans}) {
                const auto s = spmm_shape(2048, 2048, nrhs, batch,
                                          Transpose::NoTrans, tb);
                EXPECT_FALSE(SpmmTableCF::preferred(kSpmmDirect, s))
                    << "complex<float> transB " << static_cast<int>(tb)
                    << " nrhs " << nrhs << " batch " << batch
                    << ": refused WHOLE, not by nrhs -- the boundary rides on "
                       "the column pattern, which SpmmShape cannot see";
                EXPECT_TRUE(SpmmTable::preferred(kSpmmDirect, s))
                    << "float on the identical cell: 0.36-0.94, never loses";
                EXPECT_TRUE(SpmmTableD::preferred(kSpmmDirect, s));
                EXPECT_TRUE(SpmmTableCD::preferred(kSpmmDirect, s))
                    << "complex<double> on the identical cells: 0.66-0.69";
            }

            // The other side of the type-conditional: same type, transB NoTrans.
            const auto ok = spmm_shape(2048, 2048, nrhs, batch);
            EXPECT_TRUE(SpmmTableCF::preferred(kSpmmDirect, ok))
                << "complex<float> with transB == NoTrans is IN the window "
                   "(nrhs " << nrhs << " batch " << batch << ")";
        }
    }

    // And the resolved decision, both ways, with a vendor present -- the level a
    // reversed kSpmmOrder or a dropped conjunct actually shows up at.
    const auto cf_tb = spmm_shape(2048, 2048, 25, 512, Transpose::NoTrans,
                                  Transpose::Trans);
    EXPECT_TRUE(is_vendor(
        resolve_spmm_route<std::complex<float>>(kSpmmAuto, cf_tb, true)));
    EXPECT_TRUE(is_native(
        resolve_spmm_route<std::complex<double>>(kSpmmAuto, cf_tb, true)));
    EXPECT_TRUE(is_native(resolve_spmm_route<float>(kSpmmAuto, cf_tb, true)));
    // Vendor-FREE, even the refused complex<float> cell takes the native body:
    // un-preferred is not unsupported, and this is the burn-down row.
    EXPECT_TRUE(is_native(
        resolve_spmm_route<std::complex<float>>(kSpmmAuto, cf_tb, false)));
}

// THE CLAUSE SPEAKS ONLY FOR {Native, Direct} AND ONLY FOR CSR. The route half
// matters because preferred() is asked about EVERY entry in kSpmmOrder: a clause
// answering true for {Vendor, Auto} would pin the vendor as "preferred" and make
// the native route unreachable whatever the order says.
TEST(RouteSpmm, PreferredIsFalseForEveryOtherRouteAndFormat) {
    const auto s = spmm_shape(4096, 4096, 25, 512);
    ASSERT_TRUE(SpmmTable::preferred(kSpmmDirect, s))
        << "the control must be preferred or this case refuses everything";

    for (const Route r : {kVendorAuto, kSpmmNativeBare, kSpmmCta, kSpmmBlocked}) {
        EXPECT_FALSE(SpmmTable::preferred(r, s))
            << "origin " << static_cast<int>(r.origin) << " algo "
            << static_cast<int>(r.algo);
        EXPECT_FALSE(SpmmTableD::preferred(r, s));
        EXPECT_FALSE(SpmmTableCF::preferred(r, s));
        EXPECT_FALSE(SpmmTableCD::preferred(r, s));
    }

    for (MatrixFormat f : {MatrixFormat::Dense, MatrixFormat::CSC, MatrixFormat::COO,
                           MatrixFormat::SELL, MatrixFormat::BSR,
                           MatrixFormat::BLOCKED_ELL}) {
        const auto ns = spmm_shape(4096, 4096, 25, 512, Transpose::NoTrans,
                                   Transpose::NoTrans, f);
        EXPECT_FALSE(SpmmTable::preferred(kSpmmDirect, ns))
            << "format " << static_cast<int>(f);
        EXPECT_FALSE(SpmmTableCD::preferred(kSpmmDirect, ns));
    }
}

// THE CLAUSE HAS NO is_gpu TERM EITHER. supports() has no GPU gate; if preferred()
// acquired one, a native_cpu queue would still be served in a vendor-free build but
// would go back to netlib -- which refuses every transpose -- in a vendor-present
// one, silently.
TEST(RouteSpmm, PreferredHasNoGpuTerm) {
    for (int64_t batch : {1, 8, 128, 512}) {
        const auto cpu = spmm_shape(1024, 1024, 12, batch, Transpose::NoTrans,
                                    Transpose::NoTrans, MatrixFormat::CSR,
                                    /*is_gpu=*/false);
        EXPECT_TRUE(SpmmTable::preferred(kSpmmDirect, cpu)) << "batch " << batch;
        EXPECT_TRUE(SpmmTableCD::preferred(kSpmmDirect, cpu));
        EXPECT_TRUE(is_native(resolve_spmm_route<float>(kSpmmAuto, cpu, true)))
            << "a CPU queue's NoTrans spmm must take the native gather even "
               "with a vendor present";
    }
}

// UN-PREFERRED IS NOT UNSUPPORTED, AND A PIN MUST STILL REACH THE SCATTER. Had the
// refusal been written into supports() instead, the pin would fall through to the
// vendor with no diagnostic and every transposed measurement would be cuSPARSE.
TEST(RouteSpmm, ForcedNativeStillReachesTheRefusedScatter) {
    for (Transpose ta : {Transpose::Trans, Transpose::ConjTrans}) {
        const auto s = spmm_shape(2048, 2048, 50, 512, ta);
        ASSERT_FALSE(SpmmTable::preferred(kSpmmDirect, s));
        for (bool vendor : {true, false}) {
            const Route pinned = resolve_spmm_route<float>(kSpmmDirect, s, vendor);
            EXPECT_TRUE(is_native(pinned))
                << "transA " << static_cast<int>(ta) << " vendor " << vendor
                << ": a forced native:direct bypasses preferred() and must "
                   "reach the scatter";
            EXPECT_EQ(pinned.algo, Algorithm::Direct);
            const Route bare = resolve_spmm_route<float>(kSpmmNativeBare, s, vendor);
            EXPECT_TRUE(is_native(bare));
            EXPECT_EQ(bare.algo, Algorithm::Direct);
        }
    }

    // And the complex<float> cell the clause refuses by type, likewise.
    const auto cf = spmm_shape(2048, 2048, 25, 512, Transpose::NoTrans,
                               Transpose::Trans);
    ASSERT_FALSE(SpmmTableCF::preferred(kSpmmDirect, cf));
    const Route pinned =
        resolve_spmm_route<std::complex<float>>(kSpmmDirect, cf, true);
    EXPECT_TRUE(is_native(pinned));
    EXPECT_EQ(pinned.algo, Algorithm::Direct);
}

// AN AUTO SPMM IS THE NATIVE ROUTE WHEREVER THE CLAUSE FIRES, THE VENDOR WHEREVER
// IT DOES NOT, AND THE NATIVE ROUTE EVERYWHERE ONCE THE VENDOR IS GONE. The moved
// decisions are enumerated by scripts/route_diff.sh.
TEST(RouteSpmm, AutoTakesNativeWhereTheClauseFiresAndVendorWhereItDoesNot) {
    for (Transpose ta : kAllTrans) {
        for (bool gpu : {true, false}) {
            const auto s = spmm_shape(4096, 4096, 25, 128, ta, Transpose::NoTrans,
                                      MatrixFormat::CSR, gpu);
            const Route with_vendor = resolve_spmm_route<float>(kSpmmAuto, s, true);
            if (ta == Transpose::NoTrans) {
                EXPECT_TRUE(is_native(with_vendor))
                    << "the gather is the measured window (worst-of-two 0.968, "
                       "median 0.445 over 176 saturated cells) and must take "
                       "native:direct even with cuSPARSE present";
                EXPECT_EQ(with_vendor.algo, Algorithm::Direct);
            } else {
                EXPECT_TRUE(is_vendor(with_vendor))
                    << "the scatter LOSES (169 of 458 saturated cells over the "
                       "1.10 gate, worst 3.011) and must stay on the vendor";
            }
            const Route without_vendor = resolve_spmm_route<float>(kSpmmAuto, s, false);
            EXPECT_TRUE(is_native(without_vendor));
            EXPECT_EQ(without_vendor.algo, Algorithm::Direct);
        }
    }
}

// PINNING A ROUTE THE TABLE CANNOT SERVE IS SILENT, AND ITS OUTCOME DEPENDS ON THE
// BUILD. BATCHLAS_SPMM_ROUTE=cta parses fine, supports() rejects it because there
// is no CTA body, and the run then measures cuSPARSE while the operator believes it
// measured the native kernel; a MISSPELLED value behaves the same way, because
// `unparsed` is discarded. Inside the preferred window the same pin lands on
// native:direct instead -- one trap with two outcomes.
TEST(RouteSpmm, SilentPinFallThrough) {
    const auto s = spmm_shape(4096, 4096, 25, 128);
    ASSERT_TRUE(SpmmTable::supports(kSpmmDirect, s))
        << "the shape must be one Direct CAN serve, or this tests nothing";

    const Route without_vendor = resolve_spmm_route<float>(kSpmmCta, s,
                                                           /*vendor_available=*/false);
    EXPECT_TRUE(is_native(without_vendor));
    EXPECT_EQ(without_vendor.algo, Algorithm::Direct)
        << "vendor-free, a pin CTA cannot serve lands on native:direct -- NOT a "
           "throw, and not nothing";

    // OUTSIDE the preferred window -- transposed, which the measurement refused
    // -- the original behaviour is unchanged: the pin silently becomes cuSPARSE.
    const auto scatter = spmm_shape(4096, 4096, 25, 128, Transpose::Trans);
    ASSERT_TRUE(SpmmTable::supports(kSpmmDirect, scatter));
    ASSERT_FALSE(SpmmTable::preferred(kSpmmDirect, scatter));
    const Route with_vendor = resolve_spmm_route<float>(kSpmmCta, scatter,
                                                        /*vendor_available=*/true);
    EXPECT_TRUE(is_vendor(with_vendor))
        << "vendor-present and outside the preferred window, the SAME pin "
           "resolves to the VENDOR, with no diagnostic: the outcome of a pin is "
           "build-dependent, so only the resolved-route column can tell you "
           "which arm actually ran";

    // INSIDE it, the same unserviceable pin lands on the NATIVE gather -- still
    // silently, and still not what was asked for.
    const Route inside = resolve_spmm_route<float>(kSpmmCta, s,
                                                   /*vendor_available=*/true);
    EXPECT_TRUE(is_native(inside));
    EXPECT_EQ(inside.algo, Algorithm::Direct);
}

// SpmmShape MUST NOT RE-DECLARE ANY OpShape FIELD. resolve_route SLICES it to
// OpShape on the way into the coverage table, so a shadowing member would be
// written by the builder and then NOT copied: the gather and scatter arms -- which
// are different KERNELS, not different flags -- would collapse into ONE
// first-writer-wins row while the table itself still behaved correctly.
TEST(RouteSpmm, ShapeDoesNotShadowOpShapeFields) {
    SpmmShape s = spmm_shape(/*m=*/4096, /*k=*/2048, /*nrhs=*/25, /*batch=*/64,
                             Transpose::ConjTrans, Transpose::Trans,
                             MatrixFormat::CSR, /*is_gpu=*/false);
    const OpShape& sliced = static_cast<const OpShape&>(s);
    EXPECT_EQ(&s.transA, &sliced.transA)
        << "SpmmShape re-declares transA: every spmm coverage row would report "
           "NoTrans and the gather and scatter arms would collapse into one row";
    EXPECT_EQ(&s.transB, &sliced.transB)
        << "SpmmShape re-declares transB: the transB layout lever would be "
           "invisible in every route_diff";
    EXPECT_EQ(&s.m, &sliced.m)
        << "SpmmShape re-declares m: shape_class would bucket the default";
    EXPECT_EQ(&s.is_gpu, &sliced.is_gpu);
    EXPECT_EQ(&s.batch, &sliced.batch);
    EXPECT_EQ(sliced.transA, Transpose::ConjTrans);
    EXPECT_EQ(sliced.transB, Transpose::Trans);
    EXPECT_EQ(sliced.m, 4096);
    EXPECT_EQ(sliced.k, 2048);
    EXPECT_EQ(sliced.n, 25);
    EXPECT_FALSE(sliced.is_gpu)
        << "is_gpu is recorded for the coverage row and deliberately never read "
           "by supports(); it still has to SURVIVE the slice";
}

// out_rows() AND red_rows() SWAP WITH transA. The shipped clause reads NEITHER, so
// this is here to make the mapping wrong-proof for whoever adds the first extent
// clause or staged tier: the design review rejected one whose staged B slab was
// sized over `m` when B has red_rows() rows.
TEST(RouteSpmm, OutRowsAndRedRowsSwapWithTransA) {
    const auto no = spmm_shape(/*m=*/4096, /*k=*/64, /*nrhs=*/25, /*batch=*/8);
    EXPECT_EQ(no.out_rows(), 4096);
    EXPECT_EQ(no.red_rows(), 64);
    EXPECT_EQ(no.nrhs(), 25);
    for (Transpose t : {Transpose::Trans, Transpose::ConjTrans}) {
        auto tr = no; tr.transA = t;
        EXPECT_EQ(tr.out_rows(), 64)
            << "under a transposed transA the output extent is A.cols(); a "
               "predicate spelled `s.m` tests the reduction instead";
        EXPECT_EQ(tr.red_rows(), 4096);
        EXPECT_EQ(tr.nrhs(), 25) << "nrhs is C.cols() and does NOT swap";
    }
}

// THE ORDER ARRAY HAS EXACTLY TWO ENTRIES, ASSERTED RATHER THAN ASSUMED -- and this
// case is the ONLY one that reversing the array turns red, because preferred() is
// false for the vendor entry and the vendor-free walk has exactly one native
// candidate. A third entry would mean a second native tier, which would also need
// native_tier_preferred() to arbitrate it.
TEST(RouteSpmm, OrderIsExactlyTwoEntries) {
    ASSERT_EQ(SpmmTable::order_end() - SpmmTable::order_begin(), 2);
    EXPECT_EQ(SpmmTable::order_begin()[0].origin, Origin::Native);
    EXPECT_EQ(SpmmTable::order_begin()[0].algo, Algorithm::Direct);
    EXPECT_EQ(SpmmTable::order_begin()[1].origin, Origin::Vendor);
    EXPECT_EQ(SpmmTable::order_begin()[1].algo, Algorithm::Auto);
    EXPECT_TRUE(is_vendor(SpmmTable::order_begin()[1]));
}

// THE ENV VARIABLE EXISTS WITHOUT A LINE OF route_env.hh CHANGING: parse_route_env
// synthesises the name from op_env_stem(Op::spmm).
TEST(RouteSpmm, BatchlasSpmmRouteIsActuallyRead) {
    ClearRouteEnv clear(Op::spmm);

    EXPECT_EQ(op_env_stem(Op::spmm), "SPMM");
    EXPECT_TRUE(std::string(legacy_variable_for(Op::spmm)).empty())
        << "no legacy spmm variable ever shipped";

    {
        const auto unset = parse_route_env(Op::spmm);
        EXPECT_FALSE(unset.found);
        EXPECT_EQ(legacy_unset_default(Op::spmm).origin, Origin::Auto);
        EXPECT_EQ(legacy_unset_default(Op::spmm).algo, Algorithm::Auto);
    }
    {
        // A bare algorithm implies Origin::Native.
        ScopedEnv e("BATCHLAS_SPMM_ROUTE", "direct");
        const auto p = parse_route_env(Op::spmm);
        ASSERT_TRUE(p.found) << "BATCHLAS_SPMM_ROUTE was not read at all";
        EXPECT_EQ(p.route, (Route{Origin::Native, Algorithm::Direct}));
        EXPECT_EQ(p.source.variable, "BATCHLAS_SPMM_ROUTE");
        EXPECT_FALSE(p.source.legacy);
    }
    {
        ScopedEnv e("BATCHLAS_SPMM_ROUTE", "native:direct");
        const auto p = parse_route_env(Op::spmm);
        ASSERT_TRUE(p.found);
        EXPECT_EQ(p.route, (Route{Origin::Native, Algorithm::Direct}));
    }
    {
        // A bare origin leaves the algorithm free; the resolver picks the body.
        ScopedEnv e("BATCHLAS_SPMM_ROUTE", "native");
        const auto p = parse_route_env(Op::spmm);
        ASSERT_TRUE(p.found);
        EXPECT_EQ(p.route, (Route{Origin::Native, Algorithm::Auto}));
        EXPECT_EQ(resolve_spmm_route<float>(p.route,
                                            spmm_shape(4096, 4096, 25, 64), true).algo,
                  Algorithm::Direct);
    }
    {
        ScopedEnv e("BATCHLAS_SPMM_ROUTE", "vendor");
        const auto p = parse_route_env(Op::spmm);
        ASSERT_TRUE(p.found);
        EXPECT_EQ(p.route, (Route{Origin::Vendor, Algorithm::Auto}));
    }
    {
        // THE TYPO PATH: parse_route_env reports it, and every adapter in the tree
        // then DISCARDS `unparsed` and uses the unset default, so the run goes to the
        // vendor with no message.
        ScopedEnv e("BATCHLAS_SPMM_ROUTE", "not-a-route");
        const auto p = parse_route_env(Op::spmm);
        EXPECT_FALSE(p.found);
        EXPECT_TRUE(p.unparsed) << "a typo must be reported, not silently Auto";
        EXPECT_EQ(legacy_unset_default(Op::spmm), (Route{Origin::Auto, Algorithm::Auto}))
            << "and the value spmm_route.hh substitutes for it is plain Auto";
    }
}
