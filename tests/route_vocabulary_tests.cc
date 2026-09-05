// The dispatch vocabulary, and the legacy environment spellings that must keep
// meaning what they meant.
//
// The legacy variables appear in committed benchmark scripts and in the
// provenance of recorded results under output/. If BATCHLAS_GEMM_VARIANT=sycl
// silently stopped selecting the register-tiled kernel, every number recorded
// with it would quietly become uncomparable. These tests pin the mapping.

#include <gtest/gtest.h>

#include <batchlas/blas/dispatch/route.hh>
#include <batchlas/blas/dispatch/route_env.hh>
#include <batchlas/blas/dispatch/route_trsm.hh>
#include <batchlas/blas/dispatch/route_potrf.hh>

#include <complex>
#include <cstdlib>
#include <string>

using namespace batchlas;
using namespace batchlas::dispatch;

namespace {

// Sets an env var for the lifetime of the object and restores it after, so the
// cases cannot leak into each other.
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
    // The defect this whole change addresses: Provider could not say "native"
    // without also naming an algorithm, nor name an algorithm without implying
    // an origin.
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
    // Provider::Netlib was an ORIGIN masquerading as a peer of the algorithm
    // values, which is why three separate normalize_*_vendor_like() helpers
    // existed. The gate must catch it without enumerating names.
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
    // cuBLASDx compiles into our .so but is NVIDIA's source and NVIDIA-only, so
    // it can never be the portable path. Naming it must not claim Native.
    const auto r = parse_route_value("cublasdx");
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(r->algo, Algorithm::FusedDevice);
    EXPECT_TRUE(is_vendor(*r)) << "a device-library route is vendor code";
}

TEST(RouteVocabulary, UnknownValueIsRejectedNotSilentlyAuto) {
    // The legacy parsers returned Auto for anything they did not recognise,
    // which turned a typo into a silent routing change.
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
    // THE TRAP. BATCHLAS_GEMM_VARIANT=native is gemm_variant.hh's alias for
    // cuda-native / direct-cuda -- the RAW CUDA path. GemmVariantRequest::Native
    // is consumed purely as an exclusion (both gemm_use_sycl_custom and
    // gemm_use_cublasdx_custom return false for it), so the call lands in
    // gemm_vendor_impl. The same word means the opposite thing in the canonical
    // vocabulary, where "native" is BatchLAS's own kernel.
    //
    // Routing it through the canonical parser would silently flip GEMM from
    // vendor to native for anyone who had set it. Caught by the route
    // equivalence diff; pinned here so it cannot come back.
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
    // THE SECOND COLLISION. In symm/syrk/syr2k/trmm, "custom" was
    // parse_cublasdx_variant_request's custom_variant -- the FUSED cuBLASDx
    // kernel. The canonical parser reads "custom" as an alias for the
    // register-tiled GEMM family. Different kernel, same word, so the level-3
    // ops must not go through the generic parser for it.
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

    // ...and the collision does not leak into gemm, whose "custom" really is
    // the register-tiled kernel.
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
    // BATCHLAS_TRMM_VARIANT was read by two parsers that disagreed about its
    // vocabulary: one understood vendor/cublasdx/auto and answered Auto for
    // anything else, the other looked only for triangular|tiles. So this value
    // was simultaneously "no opinion" and "pin the tile kernel", and the pair
    // had to be consulted together to mean anything.
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

// --- the unset default, which used to be asymmetric ------------------------

TEST(RouteVocabulary, UnsetDefaultsAreAutoForEveryOp) {
    // This test used to be called UnsetDefaultsDifferBetweenGemmAndLevel3 and
    // asserted the opposite for GEMM: gemm_variant_request() returned Vendor
    // when its variable was unset while parse_cublasdx_variant_request()
    // returned Auto, which is why the level-3 native tile kernels ran by
    // default and GEMM's never did.
    //
    // WP2 E6 removed that asymmetry, after E3 and E4 measured every window
    // preferred() claims. Keeping the assertion and inverting it -- rather than
    // deleting it -- is deliberate: an absent assertion cannot detect a silent
    // revert, and this particular default is one line that changes the route of
    // every GEMM call in the library.
    EXPECT_EQ(legacy_unset_default(Op::gemm).origin, Origin::Auto);
    EXPECT_EQ(legacy_unset_default(Op::syrk).origin, Origin::Auto);
    EXPECT_EQ(legacy_unset_default(Op::symm).origin, Origin::Auto);
    EXPECT_EQ(legacy_unset_default(Op::trmm).origin, Origin::Auto);

    // Auto is not "always native": it defers to preferred(), and an explicitly
    // named route still wins. Those are covered in
    // tests/route_gemm_equivalence_tests.cc.
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
// WP3 step 1 -- RouteTable<Op::trsm, T>.
//
// These pin the one property the vendor-free build depends on and which a
// single-boolean predicate cannot express: a route may be SUPPORTED while not
// being PREFERRED. resolve_route (route_resolve.hh:60-63) implements the
// vendor-off fallback by re-walking the candidate order testing supports()
// ALONE, so if a speed threshold ever migrates into supports(), trsm stops
// having any route at all without a vendor and level3.cc throws. That is the
// exact regression WP3_TRSM_SPEC_CORRECTIONS.md finding 3 is about, and it
// would otherwise only show up as a vendor-free suite going red much later.
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
    // The supports/preferred split must stay visible even now that float wins
    // everywhere measured, so this names a cell that is structurally supported
    // and un-preferred for a NON-speed reason: batch below the measured floor.
    // At batch=1 the native kernel measured 0.40-0.86x (experiments/wp3_s9),
    // and that floor lives in preferred(), so a vendor-free build still routes
    // it natively rather than throwing.
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
// The measured window (WP3 step 9). These replace a pair of tests that pinned
// preferred() as all-false -- correct while nothing had been measured, and
// actively wrong now: the point of step 9 was to move traffic, so a test
// asserting that no traffic moves would have to be deleted or the flip
// abandoned. What is pinned instead is the SHAPE of the measurement, so that a
// later edit widening a window has to come with its own numbers.
// ---------------------------------------------------------------------------

TEST(RouteTrsm, FloatLeftIsPreferredAtEveryOrder) {
    // This test used to pin the boundary at 16, which is where it sat while
    // Side::Left read B's columns one scattered lane at a time. WP3 step 12
    // built the SLM staging tile and the boundary moved to 128: orders 32, 64
    // and 128 went from 0.70-0.79x to 1.19-3.20x. Order 256 did NOT follow
    // (0.76-0.93x), so there is still a window and it still has a top.
    for (int64_t order : {8, 16, 32, 64, 128}) {
        const Route r = (order <= 32) ? kCta : kBlocked;
        EXPECT_TRUE(TrsmTable::preferred(r, trsm_shape(order, 1024, 2048, 32, Side::Left)))
            << "float Side::Left order " << order << " wins after the staging tile";
    }
    // THE WORK THRESHOLD IS GONE (WP3 step 16). It existed because the large
    // cells lost, and they lost on the trailing-update GEMM rather than on the
    // solve; routing that GEMM turned 0.76-0.92x into 1.21-1.32x, so Side::Left
    // is preferred at every order and every size. These two cells are the ones
    // the old threshold excluded.
    EXPECT_TRUE(TrsmTable::preferred(kBlocked, trsm_shape(256, 1024, 512, 32, Side::Left)))
        << "float Side::Left order 256 at q*batch=524288 now measures 1.32x";
    EXPECT_TRUE(TrsmTable::preferred(kBlocked, trsm_shape(512, 1024, 512, 32, Side::Left)))
        << "float Side::Left order 512 at q*batch=524288 now measures 1.28x";
    EXPECT_TRUE(TrsmTable::preferred(kBlocked, trsm_shape(512, 256, 128, 32, Side::Left)))
        << "and the small-work cells it always won stay won";

    // Same orders, other side: all preferred, and untouched by the tile --
    // Side::Right never stages, and its register counts are byte-identical
    // before and after (114/153/144/226 at N=32).
    for (int64_t order : {8, 16, 32}) {
        EXPECT_TRUE(TrsmTable::preferred(kCta, trsm_shape(order, 1024, 2048, 32, Side::Right)))
            << "float Side::Right order " << order << " measured 1.54-4.59x";
    }
}

TEST(RouteTrsm, BatchFloorIsSpeedNotCorrectness) {
    // batch=1 measured 0.40-0.86x at order >= 32; batch=8 already wins. The
    // floor therefore sits at 8 -- and it lives in preferred(), so a vendor-free
    // build at batch=1 still routes native rather than throwing.
    const auto tiny = trsm_shape(16, 1024, 1, 32, Side::Right);
    EXPECT_FALSE(TrsmTable::preferred(kCta, tiny));
    EXPECT_TRUE(TrsmTable::supports(kCta, tiny));
    EXPECT_TRUE(is_native(resolve_trsm_route<float>(kAuto, tiny, /*vendor_available=*/false)));
    EXPECT_TRUE(is_vendor(resolve_trsm_route<float>(kAuto, tiny, /*vendor_available=*/true)));

    EXPECT_TRUE(TrsmTable::preferred(kCta, trsm_shape(16, 1024, 8, 32, Side::Right)));
}

TEST(RouteTrsm, DoubleAndComplexWinEveryMeasuredCell) {
    // 32/32, 30/30 and 30/30 cells respectively, both sides, order 8..256.
    // Spelled as three separate instantiations because the predicate is
    // `if constexpr` on T -- s.scalar is NOT what it reads, so a test that only
    // varied s.scalar would pass while testing float three times.
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
    // The vendor route is LAST in kTrsmOrder, so returning true for it would be
    // indistinguishable from falling through -- until someone reorders the list
    // and every native cell silently loses.
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
    // cta_max_n == 0 is the state until the kernel TU lands. Both native routes
    // must report UNSUPPORTED, so this table can be merged before the kernel
    // exists without ever selecting a launch that is not there.
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
// POTRF's table (WP4 step 0.4). Same discipline as the RouteTrsm block above,
// and for the same reason: WP4_POTRF_SPEC.md:559/:567 put two BATCH thresholds
// inside supports() and then claimed at :574 that the native routes would be
// "reachable only by force". Both halves are wrong, and both are load-bearing:
//
//   * a batch threshold in supports() removes potrf's vendor-free route
//     entirely (route_resolve.hh:60-63 re-walks the order testing supports()
//     ALONE), which is the failure WP4 exists to remove; and
//   * a forced route bypasses preferred() but NEVER supports()
//     (route_resolve.hh:101), so "reachable only by force" would silently run
//     cuSOLVER and pass green over an untested kernel.
//
// These cases pin the SPLIT, not any numbers -- there are no measured numbers
// for potrf yet, which is what PotrfPreferredIsFalseEverywhere says out loud.
// ---------------------------------------------------------------------------
namespace {

// PERMISSIVE DEFAULTS, one hostile field per case. If the fixture left
// cta_max_n at 0 or has_sg32 at false, every "supports() is false" case below
// would pass for the wrong reason -- the "test that cannot fail by
// construction" family this repo has hit three times (trmm uplo/diag; a
// conjugation test blind by construction; a ConjTrans test too small to reach
// the tile it guarded).
PotrfShape potrf_shape(int64_t order, int64_t batch, int cta_max,
                       Uplo uplo = Uplo::Lower) {
    PotrfShape s;
    s.op = Op::potrf;
    s.scalar = ScalarKind::F32;
    // AUTO, deliberately. resolve_potrf_route is the INSTRUMENTED entry point
    // (route_resolve.hh:130-152), so every shape built here lands in the
    // coverage table and shows up in a scripts/route_diff.sh capture. The real
    // builder sets s.backend = B (potrf_route.hh), so leaving this at AUTO is
    // what keeps a synthetic unit-test row distinguishable from a row a library
    // call actually produced -- otherwise the burn-down would read as though
    // potrf had reached a native route on CUDA, which it has not.
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
    // The one case the whole split exists for. 155 is the CTA fit ceiling WP4
    // step 0.2 MEASURED for float (experiments/wp4_potrf/slm/maxn_fitcheck.csv;
    // the spec's 105 came from a 45,056 B budget that the runtime query --
    // sycl::info::device::local_mem_size == 101,376 B -- refutes), and batch=1
    // is exactly the shape a spec-faithful supports() would have made
    // UNSUPPORTED.
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

    // A BARE ORIGIN must also resolve to a specific algorithm: potrf has two
    // native routes, so {Native, Auto} names neither and no dispatch tail can
    // map it to a kernel (route_resolve.hh:87-98).
    const Route rbare = resolve_potrf_route<float>(kPotrfNativeBare, small,
                                                   /*vendor_available=*/true);
    EXPECT_EQ(rbare.origin, Origin::Native);
    EXPECT_EQ(rbare.algo, Algorithm::CTA);
}

TEST(RoutePotrf, PreferredIsFalseEverywhere) {
    // The merge state, asserted rather than assumed. This is what makes step
    // 0.7's route_diff a real gate: with preferred() all-false, Origin::Auto
    // takes the vendor for every shape, so no existing decision can move.
    // Delete this test when the spec 10.3 grid is measured -- and replace it
    // with clauses citing the cells, as route_trsm.hh:188-325 does.
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
    // ...and the other three scalar types, spelled out: were the table ever to
    // become `if constexpr (is_same_v<T, float>)`, a loop that only varied
    // s.scalar would test float three times.
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

    // THE THREE THINGS THAT ARE *NOT* CORRECTNESS GATES. Each is what the spec
    // put in supports(); each must stay SUPPORTED.
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
    // The gate trsm does not have, so it cannot arrive by copying trsm's table.
    // A device whose sub_group_sizes lack 32 REJECTS the launch of a kernel
    // carrying [[sycl::reqd_sub_group_size(32)]], and the blocked driver's
    // diagonal leaf IS that same device function -- so one missing capability
    // must close BOTH arms, not just the CTA one.
    //
    // The order is above the CTA ceiling so that the blocked arm is the one
    // actually under test; at order 64 the CTA arm would answer first and the
    // second assertion could not distinguish the two.
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
    // Uplo::Upper on the blocked arm is a CORRECTNESS gate, not a preference:
    // the driver implements the Lower recurrence only and would read and
    // overwrite the wrong triangle. Contrast syev, whose blocked arms accept
    // Upper because they MIRROR first (syev.hh:840-847, uplo_mirror.hh).
    const auto lower = potrf_shape(/*order=*/512, /*batch=*/64, /*cta_max=*/155,
                                   Uplo::Lower);
    const auto upper = potrf_shape(/*order=*/512, /*batch=*/64, /*cta_max=*/155,
                                   Uplo::Upper);
    ASSERT_TRUE(PotrfTable::supports(kPotrfBlocked, lower));
    EXPECT_FALSE(PotrfTable::supports(kPotrfBlocked, upper));

    // ...so a vendor-free build has no native route for an Upper order above
    // the CTA ceiling, and must say so.
    EXPECT_TRUE(is_vendor(resolve_potrf_route<float>(kPotrfAuto, upper,
                                                     /*vendor_available=*/false)))
        << "no native route can serve it; the honest answer is 'needs a vendor'";
}

TEST(RoutePotrf, AbsentKernelIsUnsupportedRatherThanSelectable) {
    // cta_max_n == 0 / blocked_available == false is what a build WITHOUT the
    // kernel reports. blocked_available is exactly that state today
    // (src/extensions/potrf_cta.cc returns false for every type -- the driver is
    // Phase 2), and cta_max_n is that state on any device whose local memory
    // cannot hold even a 1x1 tile. Both native routes must then report
    // UNSUPPORTED, so a capability that is absent can never select a launch that
    // is not there. This is what let the table merge ahead of the kernel.
    const auto s = potrf_shape(/*order=*/64, /*batch=*/256, /*cta_max=*/0);
    EXPECT_FALSE(PotrfTable::supports(kPotrfCta, s));
    EXPECT_FALSE(PotrfTable::supports(kPotrfBlocked, s));
    EXPECT_TRUE(is_vendor(resolve_potrf_route<float>(kPotrfAuto, s, true)));
    EXPECT_TRUE(is_vendor(resolve_potrf_route<float>(kPotrfAuto, s, false)))
        << "vendor-free with nothing supported must say 'needs a vendor', not "
           "invent a native route";

    // AND FORCING MUST NOT ESCAPE IT. route_resolve.hh:101 gates the forced
    // route on supports() and falls through to automatic() -- which is why a
    // green forced-route test is not by itself evidence that a native kernel
    // ran.
    EXPECT_TRUE(is_vendor(resolve_potrf_route<float>(kPotrfCta, s, true)));
    EXPECT_TRUE(is_vendor(resolve_potrf_route<float>(kPotrfNativeBare, s, true)));
}

TEST(RoutePotrf, BatchlasPotrfRouteIsActuallyRead) {
    // The spec's BATCHLAS_POTRF_PROVIDER is read by NOTHING in this tree, so
    // "pin the native path with it" would have pinned nothing. The canonical
    // spelling needs no registry entry -- parse_route_env synthesises
    // "BATCHLAS_" + op_env_stem(op) + "_ROUTE" (route_env.hh:214-217) -- but
    // that is a claim about a code path nobody had exercised for potrf.
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
