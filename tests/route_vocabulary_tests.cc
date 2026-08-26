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
#include <batchlas/blas/dispatch/route_geqrf.hh>
#include <batchlas/blas/dispatch/route_orgqr.hh>
#include <batchlas/blas/dispatch/route_getrf.hh>
#include <batchlas/blas/dispatch/route_getrs.hh>
#include <batchlas/blas/dispatch/route_getri.hh>
#include <batchlas/blas/dispatch/route_gemv.hh>

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

// ---------------------------------------------------------------------------
// GEQRF's table (WP5 scaffolding). Same discipline as the RoutePotrf block
// above, plus gates that have no potrf analogue at all.
//
// WHY THESE CASES EXIST AT A POINT WHERE NO KERNEL DOES. Every one of them pins
// a property that is invisible while the capabilities report absent and becomes
// load-bearing the instant one comes off zero:
//
//   * a speed threshold in supports() would remove geqrf's vendor-free route
//     entirely (route_resolve.hh:60-63 re-walks the order testing supports()
//     ALONE), and geqrf is the op the vendor-free burn-down is blocked on;
//   * a forced route bypasses preferred() but NEVER supports()
//     (route_resolve.hh:101), so a table with one gate wrong makes
//     BATCHLAS_GEQRF_ROUTE=cta silently run cuSOLVER and pass green;
//   * copying potrf's `m == n` gate here would strip geqrf of rectangular A,
//     which is the entire point of the op (options.hh:727-730).
//
// THE BREAKS THAT WERE RUN AGAINST THESE CASES, AND WHAT EACH DID, recorded in
// the shape of tests/potrf_tests.cc:28-70 -- because this repository has now
// shipped FIVE tests that could not fail by construction, the most recent
// written in the same change as the fix it guarded. Every break below was
// applied to the source, REBUILT, and run; the two that turned nothing red are
// reported, not hidden.
//
//   B1  a speed threshold in supports() (`if (s.batch < 64) return false;`)
//         -> RED: VendorFreeFallbackHandsOverTheNativeRoute,
//                 CorrectnessGatesAreNotSpeedGates
//   B2  preferred() replaced by route_ormqr.hh:78-79's
//       `is_native(r) && supports(r, s)`
//         -> RED: PreferredIsFalseEverywhere,
//                 VendorFreeFallbackHandsOverTheNativeRoute
//   B3  the `m >= n` gate deleted
//         -> RED: WideIsUnsupportedByEveryNativeArm
//   B4  the CTA area test replaced by a per-extent bound
//       (`s.n <= s.cta_max_elems` instead of `s.m * s.n <= s.cta_max_elems`)
//         -> RED: CtaCapacityIsAnAreaAndAHeightNotTwoExtentBounds
//   B5  the `cta_max_* < 1` absent-kernel guards deleted from both arms
//         -> RED: AbsentKernelIsUnsupportedRatherThanSelectable
//   B6  `case Op::geqrf/orgqr:` added to legacy_variable_for (route_env.hh:109)
//         -> RED: BatchlasGeqrfRouteIsActuallyRead,
//                 BatchlasOrgqrRouteIsActuallyRead
//   B7  route_orgqr.hh's INHERITED complex + Transpose::Trans gate deleted
//         -> RED: CorrectnessGatesIncludeTheOnesInheritedFromOrmqr
//
// AND THE TWO THAT NOTHING HERE CAN SEE, which is the more useful half:
//
//   B8  geqrf_cta_max_m_for_slm/geqrf_cta_max_elems_for_slm made non-zero with
//       NO kernel behind them
//         -> NOTHING in this file turned red, and nothing could: every case
//            above builds its own GeqrfShape and never asks the build what its
//            capabilities are. It was verified at the FACADE instead --
//            build-novendor's orgqr_tests then failed with
//            "geqrf_buffer_size: resolved to a native route (native:cta) but no
//            native geqrf kernel is linked into this build", i.e. the whole
//            chain (env read -> builder -> table -> vendor-free fallback ->
//            facade native arm -> internal-consistency throw) is live.
//   B9  B8 plus the facade's geqrf native arm and its buffer-size native terms
//       deleted
//         -> NOTHING turned red anywhere. build-novendor went straight back to
//            the ordinary "no route for geqrf<float> ... built without cuBLAS"
//            NoRouteError. THAT is the defect this scaffolding exists to
//            prevent: without the native arm, a capability coming off zero is
//            absorbed in silence, and in a vendor-PRESENT build the same
//            deletion would quietly hand every call to cuSOLVER
//            (route_compiled.hh:1-24). No unit test can catch it; only writing
//            the arm before the kernel can.
//
// UPDATE, WP5 KERNELS: the kernels have since landed
// (src/extensions/geqrf_cta_device.hh + geqrf_cta.cc + geqrf_blocked.cc, and
// orgqr_blocked.cc), so the capacities now answer from the device's real
// local_mem_size and geqrf_blocked_available/orgqr_blocked_available are true.
// NONE of the cases below changed: each builds its own GeqrfShape/OrgqrShape by
// hand and never asks the build what its capabilities are -- which is exactly
// what B8 records. The kernels' own correctness is verified in
// experiments/wp5_qr/kernels/ (a residual + orthogonality + ELEMENTWISE-vs-vendor
// harness, five reference breaks and seven kernel breaks). preferred() is still
// false for both ops, so PreferredIsFalseEverywhere is still the merge state and
// still the thing to delete when a measured grid says otherwise.
//
// ONE PROPERTY DELIBERATELY LEFT UNGUARDED, stated so it is not mistaken for
// covered: geqrf_op_shape/orgqr_op_shape set `s.backend = B` (the line trsm's
// and ormqr's builders omit, which is why their coverage rows all read
// Backend::AUTO). Deleting it turns nothing red here, because these cases build
// their shapes by hand. It is only observable in a -DBATCHLAS_ENABLE_COVERAGE
// build's CSV.
// ---------------------------------------------------------------------------
namespace {

// PERMISSIVE DEFAULTS, one hostile field per case. If the fixture left the
// capacities at 0 or has_sg32 at false, every "supports() is false" case below
// would pass for the wrong reason -- the "test that cannot fail by construction"
// family this repo has hit five times.
GeqrfShape geqrf_shape(int64_t rows, int64_t cols, int64_t batch,
                       int cta_max_m, int64_t cta_max_elems) {
    GeqrfShape s;
    s.op = Op::geqrf;
    s.scalar = ScalarKind::F32;
    // AUTO, deliberately -- the same reason as potrf_shape's: resolve_geqrf_route
    // is the INSTRUMENTED entry point (route_resolve.hh:130-152), so every shape
    // built here lands in the coverage table and shows up in a route_diff
    // capture. The real builder sets s.backend = B (src/backends/geqrf_route.hh),
    // so leaving this at AUTO keeps a synthetic unit-test row distinguishable
    // from a row a library call actually produced.
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
    // THE TEST THAT FAILS IF A SPEED THRESHOLD EVER LANDS IN supports(). batch=1
    // and a tiny panel are exactly the shapes a "minimum batch" or "minimum n"
    // gate would make UNSUPPORTED, and route_resolve.hh:60-63 tests supports()
    // alone -- so such a gate turns a vendor-free geqrf back into the
    // NoRouteError that four ormqr/orgqr suites already die on.
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
    // The single most likely wrong edit in this file is copying
    // route_potrf.hh:213's `if (s.m != s.n) return false;`. A tall panel is what
    // band_reduction.cc:595 and sytrd_sy2sb.cc:504 actually issue.
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
    // m < n IS a correctness gate: both planned drivers are panel-oriented
    // right-looking schedules over columns, and a wide view walks the trailing
    // update past the bottom of the panel. It is also the conservative direction
    // -- a superfluous gate sends the call to the vendor (loud in a vendor-free
    // build), a missing one returns wrong numbers quietly.
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
    // The CTA tile holds the whole m x n panel, so what fits is governed by m*n.
    // A table that checked only per-extent ceilings would accept a panel whose
    // tile is many times the budget -- an unlaunchable route, which is exactly
    // what supports() exists to exclude.
    //
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

    // The BLOCKED arm inherits the PRESENCE of the leaf but not its capacity --
    // it splits the panel itself. That is what makes the two-tier ladder a
    // capability ladder rather than a tuned guess.
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

    // THE THINGS THAT ARE *NOT* CORRECTNESS GATES. Each must stay SUPPORTED, and
    // each is what a spec-shaped supports() would have refused.
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
    // A device whose sub_group_sizes lack 32 REJECTS the launch of a kernel
    // carrying [[sycl::reqd_sub_group_size(32)]], and the blocked driver's panel
    // leaf IS that same device function -- so one missing capability must close
    // BOTH arms, not just the CTA one.
    //
    // The panel is chosen ABOVE the CTA area so the blocked arm is the one
    // actually under test; at a fitting size the CTA arm would answer first.
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

    // And a vendor-free build must then say "needs a vendor" rather than handing
    // back a route whose launch the device would reject.
    EXPECT_TRUE(is_vendor(resolve_geqrf_route<float>(kGeqrfAuto, small, false)));
}

TEST(RouteGeqrf, PreferredIsFalseEverywhere) {
    // The merge state, asserted rather than assumed. With preferred() all-false,
    // Origin::Auto takes the vendor for every shape, so no existing decision can
    // move -- which is what makes the scaffolding gate ("same passes, same
    // failures, same messages") a real gate. Delete this test when a measured
    // window lands, and replace it with clauses citing the cells.
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
    // ...and the other three scalar types, spelled out: were the table ever to
    // become `if constexpr (is_same_v<T, float>)`, a loop that only varied
    // s.scalar would test float three times.
    const auto s = geqrf_shape(256, 64, 512, 4096, 1 << 24);
    EXPECT_FALSE((RouteTable<Op::geqrf, double>::preferred(kGeqrfCta, s)));
    EXPECT_FALSE((RouteTable<Op::geqrf, std::complex<float>>::preferred(kGeqrfCta, s)));
    EXPECT_FALSE((RouteTable<Op::geqrf, std::complex<double>>::preferred(kGeqrfCta, s)));
    EXPECT_FALSE((RouteTable<Op::geqrf, double>::preferred(kGeqrfBlocked, s)));
    EXPECT_FALSE((RouteTable<Op::geqrf, std::complex<float>>::preferred(kGeqrfBlocked, s)));
    EXPECT_FALSE((RouteTable<Op::geqrf, std::complex<double>>::preferred(kGeqrfBlocked, s)));
}

TEST(RouteGeqrf, BareOriginResolvesToASpecificAlgorithm) {
    // geqrf has TWO native routes, so {Native, Auto} names neither and no
    // dispatch tail can map it to a kernel (route_resolve.hh:87-98). Below the
    // CTA capacity -> CTA; above it -> Blocked. Never {Native, Auto}, never "no
    // route".
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
    // ZERO CAPACITY / blocked_available == false is what THIS BUILD reports
    // today: src/extensions/geqrf_cta.cc returns 0 from both capacity functions
    // and geqrf_blocked.cc returns false for every type. Both native routes must
    // then be UNSUPPORTED, so a capability that is absent can never select a
    // launch that is not there. This is what lets the tables merge ahead of the
    // kernels.
    const auto s = geqrf_shape(/*rows=*/128, /*cols=*/64, /*batch=*/256,
                               /*cta_max_m=*/0, /*cta_max_elems=*/0);
    EXPECT_FALSE(GeqrfTable::supports(kGeqrfCta, s));
    EXPECT_FALSE(GeqrfTable::supports(kGeqrfBlocked, s));
    EXPECT_TRUE(is_vendor(resolve_geqrf_route<float>(kGeqrfAuto, s, true)));
    EXPECT_TRUE(is_vendor(resolve_geqrf_route<float>(kGeqrfAuto, s, false)))
        << "vendor-free with nothing supported must say 'needs a vendor', not "
           "invent a native route";

    // AND FORCING MUST NOT ESCAPE IT. route_resolve.hh:101 gates the forced route
    // on supports() and falls through to automatic() -- which is why a green
    // forced-route test is not by itself evidence that a native kernel ran.
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
    // geqrf had NO route resolution at all before WP5 -- the facade was four
    // `if constexpr (!vendor) throw; else vendor;` bodies -- so "pin the native
    // path with BATCHLAS_GEQRF_ROUTE" was a claim about a code path nobody had
    // exercised for this op. The canonical spelling needs no registry entry
    // (parse_route_env synthesises it, route_env.hh:214-217), but that is exactly
    // the sort of claim that turns out to be false.
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
        // AN UNRECOGNISED VALUE IS SILENTLY {Auto, Auto}, WHICH IS THE VENDOR.
        // This is the measurement trap the campaign has hit before: a "native"
        // run that looks identical to the vendor probably IS the vendor.
        ScopedEnv e("BATCHLAS_GEQRF_ROUTE", "not-a-route");
        const auto p = parse_route_env(Op::geqrf);
        EXPECT_FALSE(p.found);
        EXPECT_TRUE(p.unparsed) << "a typo must be reported, not silently Auto";
    }
}

// ---------------------------------------------------------------------------
// ORGQR's table (WP5 scaffolding).
//
// orgqr ships as ORMQR APPLIED TO AN IDENTITY, so its supports() is
// RouteTable<Op::ormqr,T>::supports()' gates TRANSCRIBED plus orgqr's own --
// because that table is what will actually serve the call, and silently omitting
// an inherited gate is the wrong-answer class. These cases pin the transcription
// as well as the split.
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
    // The same speed-threshold guard as geqrf's. It matters more here than the
    // ratios suggest: the vendor orgqr is NOT batched (cublas.cc:1413-1420 loops
    // cusolverDnXorgqr per batch item), so a gate that pushed orgqr back to the
    // vendor would also restore a workspace of single_ws * batch
    // (cublas.cc:1447).
    const auto s = orgqr_shape(/*rows=*/64, /*cols=*/64, /*batch=*/1);

    EXPECT_TRUE(OrgqrTable::supports(kOrgqrBlocked, s));
    EXPECT_FALSE(OrgqrTable::preferred(kOrgqrBlocked, s));
    EXPECT_TRUE(is_native(resolve_orgqr_route<float>(kOrgqrAuto, s,
                                                     /*vendor_available=*/false)));
    EXPECT_TRUE(is_vendor(resolve_orgqr_route<float>(kOrgqrAuto, s,
                                                     /*vendor_available=*/true)));
}

TEST(RouteOrgqr, PreferredIsFalseEverywhere) {
    // NOT route_ormqr.hh:78-79's `is_native(r) && supports(r, s)`. Copying that
    // spelling would make native the default on every supported shape with no
    // measured window at all -- and there IS a measured losing cell to respect
    // (cfloat n=2048 loses at every batch that fits in 24 GB).
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

    // INHERITED from route_ormqr.hh:59 -- the native apply is GPU-only, which is
    // also why the four ormqr/orgqr suites' Backend::NETLIB rows are NOT closable
    // by WP5 (test_utils::backend_types instantiates them against Device("cpu")).
    auto cpu = ok;  cpu.is_gpu = false;
    EXPECT_FALSE(OrgqrTable::supports(kOrgqrBlocked, cpu));

    // INHERITED from route_ormqr.hh:63-66 -- complex with a plain Trans. It
    // cannot fire through the builder, which pins transA to NoTrans, but the gate
    // is transcribed rather than dropped, and this case is what proves the
    // transcription is present and applies only to complex.
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
    // blocked_available == false is what THIS BUILD reports today
    // (src/extensions/orgqr_blocked.cc). It is NOT "is ormqr_blocked compiled" --
    // that is already true, and answering with it would hand a vendor-free caller
    // a route the facade cannot service.
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
    // {Native, Blocked} rather than verbatim -- no dispatch tail can map an Auto
    // algorithm to a kernel. Note the departure from route_ormqr.hh:57, which
    // accepts Algorithm::Auto inside supports() as if it were Blocked.
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

    // AND THE OP IT DELEGATES TO IS PINNED BY A DIFFERENT VARIABLE. orgqr's
    // native arm re-enters routed ormqr, which reads its own canonical spelling
    // AND a legacy one. Pinning one and not the other is how a measurement ends
    // up describing something other than what was intended.
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
// WP6 -- the LU family: getrf, getrs, getri.
//
// These cases are SYNTHETIC: they call supports()/preferred() on hand-built
// shapes, so they exercise the tables and not the kernels -- which is why they
// keep working now that all three ops have live native arms. tests/getrf_tests.cc
// is where the real device shapes are asserted.
//
// Same discipline as the RoutePotrf and RouteGeqrf blocks above. Every case here
// pins a property that was invisible while the capabilities reported absent and
// is load-bearing now that they do not:
//
//   * a speed threshold in supports() would remove the vendor-free route
//     entirely (route_resolve.hh:113-127 re-walks the order testing supports()
//     ALONE), and for getrf/getri that is the route inverse_tests needs;
//   * a forced route bypasses preferred() but NEVER supports()
//     (route_resolve.hh:165), so a table with one gate wrong makes
//     BATCHLAS_GETRF_ROUTE=cta silently run cuBLAS and pass green;
//   * getrf/getri DO take potrf's `m == n` gate, unlike geqrf where copying it was
//     the recorded wrong edit -- and a case below pins that the two families
//     differ on purpose rather than by accident;
//   * getrs's transA is a LIVE routing input and the only field in this family
//     that separates a coverage row from its neighbour.
//
// THE BREAKS THAT WERE RUN AGAINST THESE CASES, AND WHAT EACH DID, recorded in the
// shape of the RouteGeqrf block above -- because this repository has now shipped
// SIX tests that could not fail by construction, one of them written in the same
// change as the fix it guarded. Every break below was applied to the source,
// REBUILT, and run; the ones that turned nothing red are reported, not hidden.
// The results are in the WP6 record.
// ---------------------------------------------------------------------------
namespace {

// PERMISSIVE DEFAULTS, one hostile field per case. If the fixture left cta_max_n
// at 0 or has_sg32 at false, every "supports() is false" case below would pass for
// the wrong reason -- the "test that cannot fail by construction" family this repo
// has now hit six times.
GetrfShape getrf_shape(int64_t order, int64_t batch, int cta_max_n) {
    GetrfShape s;
    s.op = Op::getrf;
    s.scalar = ScalarKind::F32;
    // AUTO, deliberately -- the same reason as potrf_shape's and geqrf_shape's:
    // resolve_getrf_route is the INSTRUMENTED entry point
    // (route_resolve.hh:178-217), so every shape built here lands in the coverage
    // table and shows up in a route_diff capture. The real builder sets
    // s.backend = B (src/backends/getrf_route.hh), so leaving this at AUTO keeps a
    // synthetic unit-test row distinguishable from one a library call produced.
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

// THE FUSED TIER'S TWO CAPACITIES, AND THEY DEFAULT TO PRESENT.
//
// This is a REPAIR, and the defect it repairs is the exact shape this file exists
// to catch. The helper below used to set neither field, so both defaulted to 0,
// so RouteTable<getrs>::supports({Native, CTA}, s) returned false on EVERY shape
// in this file -- which meant every `is_vendor(resolve(...))` assertion about
// getrs held no matter what preferred() said, and the whole fused tier plus its
// measured nrhs window could land, or be inverted, with this suite reporting
// 78/78 either way. A helper that quietly describes a device on which the tier
// under test cannot run is a BLIND GUARD, and the campaign has now recorded eight.
//
// The values are what this box actually reports for a 4-byte scalar: local memory
// 101,376 B, less the 4 KB the launcher holds back, less the largest diagonal
// block the tier ever charges (32 x 33 floats = 4,224 B), over sizeof(float) --
// 23,264 elements -- and kGetrsFusedMaxRhs = 8. They are CONSTANTS here rather
// than a device query because this file is the PURE layer: getrf_tests.cc's
// RouteTableAndTheVendorFreeFallback is what asks the real builder on the real
// device whether it agrees. Two files, two halves of the same claim.
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
// route_resolve.hh:76-83 performs, spelled once here.
//
// IT HAS TO BE A TEMPLATE. Written inline against a concrete table
// (`requires { GetrfTable::native_tier_preferred(r, s); }`) the name lookup is a
// HARD ERROR rather than a substitution failure, because nothing is dependent --
// which is exactly how route_resolve.hh gets it right: its check sits inside a
// template whose `Table` is a parameter.
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
//
// WHY THIS BLOCK EXISTS, AND IT IS A REPAIR OF THE SAME CLASS THE FILE ALREADY
// RECORDS TWICE. Every LU helper above builds a shape with `s.scalar =
// ScalarKind::F32` and every alias above instantiates the table at `float`.
// preferred() decides on the TABLE's T (`if constexpr (std::is_same_v<T, ...>)`)
// and NOT on s.scalar, so a per-type clause -- and WP8 lands three of them --
// has its cfloat, double and cdouble boundaries checked by NOTHING unless the
// table is instantiated at those types. A sweep that varies s.scalar tests float
// three times and reports green through an inverted double clause.
//
// NOTE that the helpers still build shapes carrying `scalar = F32`. That is
// harmless for the DECISION -- preferred() reads the table's T and never
// s.scalar -- but it means a route_diff capture reads F32 on every synthetic row
// in this file, whatever type the table was instantiated at. Filter those rows
// on `backend == AUTO` and read them as "the pure layer ran", not as "the
// library routed a float".
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
    // THE TEST THAT FAILS IF A SPEED THRESHOLD EVER LANDS IN supports(). batch=2
    // and a small order are exactly the shapes a "minimum batch" or "minimum n"
    // gate would make UNSUPPORTED, and route_resolve.hh:113-127 tests supports()
    // ALONE -- so such a gate turns a vendor-free getrf back into the NoRouteError
    // that inverse_tests dies on today.
    //
    // batch=2 AND order=40 ARE inverse_tests' ACTUAL EXTENTS
    // (tests/inverse_tests.cc:10-39). That suite is the one WP6 can close
    // outright, and it closes if and only if these values stay supported.
    const auto s = getrf_shape(/*order=*/40, /*batch=*/2, /*cta_max_n=*/128);

    EXPECT_TRUE(GetrfTable::supports(kGetrfCta, s))
        << "batch size and order are speed questions; neither may gate CORRECTNESS "
           "-- and these are inverse_tests' own extents";
    EXPECT_TRUE(GetrfTable::supports(kGetrfBlocked, s));
    EXPECT_FALSE(GetrfTable::preferred(kGetrfCta, s))
        << "getrf's preferred() is all-false BY DECISION, not by absence: both "
           "native arms exist and are measured, and the window is withheld "
           "because the crossover moves with batch as much as with order "
           "(experiments/wp6_lu/bench/README.md). Flip this only together with "
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
    // getrf takes route_potrf.hh:213's `m != n` line and geqrf refuses it, and the
    // two are one edit apart. This case pins BOTH halves so that copying either
    // file's supports() into the other turns something red.
    //
    // The justification is not "LU is square" in the abstract -- LAPACK's xGETRF is
    // defined for rectangular A. It is that BatchLAS's public getrf is square:
    // options.hh:615 calls require_square on every arena spelling and cuBLAS's
    // getrfBatched takes one `n`. Widening it is an API change, not a routing
    // decision.
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

    // ...and the sibling table must still NOT have the gate. A wrong edit in the
    // other direction -- deleting geqrf's rectangular support -- is the recorded
    // one (route_geqrf.hh:55-64), so it is pinned from here too.
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
    // the capacity is a hard launch limit and not a tuning knob. A single number
    // suffices here where geqrf needs two, because the operand is square.
    const auto fits = getrf_shape(/*order=*/128, /*batch=*/64, /*cta_max_n=*/128);
    ASSERT_TRUE(GetrfTable::supports(kGetrfCta, fits))
        << "guard: order 128 is exactly the capacity";

    const auto over = getrf_shape(/*order=*/129, /*batch=*/64, /*cta_max_n=*/128);
    EXPECT_FALSE(GetrfTable::supports(kGetrfCta, over));

    // The BLOCKED arm inherits the PRESENCE of the leaf but not its capacity -- it
    // splits the matrix into panels the leaf can hold itself. That is what makes
    // the two-tier ladder a capability ladder rather than a tuned guess.
    EXPECT_TRUE(GetrfTable::supports(kGetrfBlocked, over));

    // ...but only when it exists.
    auto no_blocked = over;
    no_blocked.blocked_available = false;
    EXPECT_FALSE(GetrfTable::supports(kGetrfBlocked, no_blocked));

    // AND THE BLOCKED ARM CARRIES NO LOWER BOUND. "order <= the CTA capacity so
    // blocked should be false" is a FIT judgement between two native routes, and
    // route_potrf.hh:284-296 records what putting it in supports() costs: per
    // route_resolve.hh:165 a forced `blocked` at a small order falls through to
    // automatic() at :175, which at merge returns {Vendor, Auto} -- so the test
    // that pinned the blocked driver measures cuBLAS and passes green.
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

    // THE THINGS THAT ARE *NOT* CORRECTNESS GATES. Each must stay SUPPORTED, and
    // each is what a spec-shaped supports() would have refused.
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
    // A device whose sub_group_sizes lack 32 REJECTS the launch of a kernel
    // carrying [[sycl::reqd_sub_group_size(32)]], and the blocked driver's
    // diagonal-panel leaf IS that same device function -- so one missing capability
    // must close BOTH arms, not just the CTA one.
    //
    // The order is chosen ABOVE the CTA capacity so the blocked arm is the one
    // actually under test; at a fitting size the CTA arm would answer first.
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

    // And a vendor-free build must then say "needs a vendor" rather than handing
    // back a route whose launch the device would reject.
    EXPECT_TRUE(is_vendor(resolve_getrf_route<float>(kGetrfAuto, small, false)));
}

// ---------------------------------------------------------------------------
// THE PIVOT-FORMAT GATE. This is the one route test in the file with a BACKEND
// axis, and it exists because the omission it guards was a SILENT WRONG ANSWER
// that shipped: supports() gated on s.is_gpu and never on s.backend, so on a GPU
// queue constructed with Backend::NETLIB the native getrf was selectable, wrote
// PACKED 1-based int32 into the first half of the caller's int64 pivot span, and
// netlib's getri/getrs read the same bytes as GENUINE int64
// (netlib_lapack.cc:1235, :1312-1320, :1361). Measured before the gate:
// ||A*C - I||_F / n = 5.32e-01 with getri info == 0, against 5.15e-07 when both
// arms agreed. Nothing threw and nothing in the suite fired -- tests/
// getrf_tests.cc skips every NETLIB row because its fixture queue is a CPU
// queue, and every other case in THIS file leaves s.backend at AUTO, which is
// exactly why the defect was invisible from both sides.
//
// THE ASSERTIONS ARE ANTI-VACUOUS BY CONSTRUCTION: each op first ASSERTs that
// the same shape with the SAME extents is supported at Backend::CUDA, so
// "supported becomes false" cannot pass because the shape was unsupported for an
// unrelated reason. It is a CORRECTNESS gate, so it belongs in supports() and
// not preferred(): a forced route bypasses preferred() and never supports()
// (route_resolve.hh:8-10, :101, :165).
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

    // A forced route must be REFUSED, not silently honoured. This is the path
    // that actually produced the wrong answer: BATCHLAS_GETRF_ROUTE=blocked with
    // getri left to resolve on its own.
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

    // AND THE GATE MUST NOT SWALLOW THE VENDOR-FREE FALLBACK BY ACCIDENT. With
    // no vendor at all there is nothing to disagree with -- but there is also no
    // route, and "throws NoRouteError" is the honest answer for a configuration
    // whose pivot format the native kernel cannot serve. Asserted so that a
    // later widening of this gate is a deliberate change and not a discovery.
    EXPECT_FALSE(is_native(resolve_getrf_route<float>(kGetrfAuto, f, false)));
}

// THE MEASURED ORDER WINDOW (WP8 routing pass), replacing the all-false sweep.
// Its own note said "delete this test when a measured window lands, and replace
// it with clauses citing the cells"; this is that replacement, and the cells are
// in route_getrf.hh beside the clause.
//
// TWO STRUCTURAL POINTS THIS CASE PINS, BOTH OF WHICH A SIMPLER SWEEP MISSES.
//  1. THE CLAUSE NAMES Algorithm::Blocked, NOT "native". The CTA arm is a
//     different kernel and it LOSES where it serves (float n=128 reads
//     0.773-0.872 of cuBLAS). cta_max_n is passed large here on purpose, so CTA
//     is SUPPORTED at every order in the loop and a clause that forgot the algo
//     test would be caught by the resolved-route assertion rather than hidden by
//     supports().
//  2. IT IS PER TYPE. float's boundary is 256 and cfloat's is 512; a loop that
//     only varied s.scalar would test float four times, because preferred()
//     reads the TABLE's T.
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
        // cfloat: 512, not 256. I1's grid puts cfloat n=256 batch=128 at 0.884,
        // which is why the two boundaries differ by one grid step.
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
        // double and cdouble: nothing, at any order. double's best cell anywhere
        // is 1.067 and cdouble's is 1.012; `double order >= 512` would admit
        // 0.933 / 0.813 / 0.743 at batch 256 / 512 / 1024.
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
    // getrf has TWO native routes, so {Native, Auto} names neither and no dispatch
    // tail can map it to a kernel (route_resolve.hh:146-163). Below the CTA
    // capacity -> CTA; above it -> Blocked. Never {Native, Auto}, never "no route".
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
    // ZERO CAPACITY / blocked_available == false is what THIS BUILD reports today:
    // src/extensions/getrf_cta.cc returns 0 from the capacity function and
    // getrf_blocked.cc returns false for every type. Both native routes must then
    // be UNSUPPORTED, so a capability that is absent can never select a launch that
    // is not there. This is what lets the tables merge ahead of the kernels.
    const auto s = getrf_shape(/*order=*/64, /*batch=*/256, /*cta_max_n=*/0);
    EXPECT_FALSE(GetrfTable::supports(kGetrfCta, s));
    EXPECT_FALSE(GetrfTable::supports(kGetrfBlocked, s));
    EXPECT_TRUE(is_vendor(resolve_getrf_route<float>(kGetrfAuto, s, true)));
    EXPECT_TRUE(is_vendor(resolve_getrf_route<float>(kGetrfAuto, s, false)))
        << "vendor-free with nothing supported must say 'needs a vendor', not "
           "invent a native route";

    // AND FORCING MUST NOT ESCAPE IT. route_resolve.hh:165 gates the forced route
    // on supports() and falls through to automatic() -- which is why a green
    // forced-route test is not by itself evidence that a native kernel ran.
    EXPECT_TRUE(is_vendor(resolve_getrf_route<float>(kGetrfCta, s, true)));
    EXPECT_TRUE(is_vendor(resolve_getrf_route<float>(kGetrfBlocked, s, true)));
    EXPECT_TRUE(is_vendor(resolve_getrf_route<float>(kGetrfNativeBare, s, true)));
    EXPECT_TRUE(is_vendor(resolve_getrf_route<float>(kGetrfCta, s, false)));

    // Half a capability is still absent: a build reporting the blocked driver but
    // no CTA leaf must not select the blocked arm, because the leaf IS the CTA
    // device function.
    auto half = getrf_shape(64, 256, /*cta_max_n=*/0);
    half.blocked_available = true;
    EXPECT_FALSE(GetrfTable::supports(kGetrfBlocked, half))
        << "the blocked driver's diagonal-panel leaf IS the CTA device function, "
           "so it inherits the presence gate";
}

TEST(RouteGetrf, NativeTierPreferredIsDeclaredAndPinsTheMeasuredTierChoice) {
    // The scaffolding pinned this hook's DELIBERATE ABSENCE and said in as many
    // words: "Delete this case when the tier sweep lands and the predicate is
    // declared; replace it with one that pins the measured crossover." The sweep
    // has landed (experiments/wp6_lu/kernels/tier.txt, run_tier.sh: both arms
    // pinned, every pin verified to have taken, double re-run across four
    // batches), so this is that replacement.
    //
    // THE MEASURED ANSWER, blocked_ms / cta_ms, so > 1 means CTA is ahead:
    //   float   n=64 1.74  n=76 1.48  n=96 1.49  n=100 1.68  n=128 1.13
    //   cfloat  n=64 1.39  n=76 1.59  n=96 1.30  n=100 1.33
    //   cdouble n=64 1.37  n=76 1.09
    //   double  n=64 0.98  n=76 0.85  n=96 0.77  n=100 1.00
    // i.e. DOUBLE alone prefers the blocked driver below its own CTA ceiling.
    EXPECT_TRUE((declares_native_tier_preferred<GetrfTable, GetrfShape>))
        << "the tier sweep has run; an undeclared hook now costs 1.18-1.29x at "
           "double n=76..96 in the vendor-free build";

    // float: CTA below the capacity ceiling, from the hook rather than from the
    // order array -- the two are only distinguishable on the type where they
    // disagree, which is why the double case below is the load-bearing one.
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

    // ...and at n <= 32 double goes back to CTA, because there the blocked
    // driver's nb is min(32, n) = n and it runs ONE panel whose leaf IS the CTA
    // device function -- the same code, measured identical (1.8126 vs 1.8113 ms
    // at n=32 batch 8192), so CTA is the cheaper spelling of it.
    auto tiny_d = getrf_shape(32, 8192, 128);
    tiny_d.scalar = ScalarKind::F64;
    EXPECT_TRUE((GetrfTableD::native_tier_preferred(kGetrfCta, tiny_d)));
    EXPECT_EQ(resolve_getrf_route<double>(kGetrfAuto, tiny_d,
                                          /*vendor_available=*/false).algo,
              Algorithm::CTA);

    // IT IS NOT A CORRECTNESS GATE. Both arms stay supports()-able at every shape
    // the window moves, which is what keeps a pinned `cta` at double n=64 running
    // CTA instead of falling through to automatic() (route_resolve.hh:165 -> :175)
    // and measuring the very arm it was pinned away from.
    EXPECT_TRUE((GetrfTableD::supports(kGetrfCta, small_d)));
    EXPECT_TRUE((GetrfTableD::supports(kGetrfBlocked, small_d)));
    EXPECT_EQ(resolve_getrf_route<double>(kGetrfCta, small_d,
                                          /*vendor_available=*/false).algo,
              Algorithm::CTA);

    // AND IT MOVES NOTHING IN A VENDOR-PRESENT BUILD, which is the whole reason
    // this is the third predicate and not a preferred() window: the hook is
    // consulted only inside the `!vendor_available` branch (route_resolve.hh:
    // 119-127).
    EXPECT_TRUE(is_vendor(resolve_getrf_route<double>(kGetrfAuto, small_d,
                                                      /*vendor_available=*/true)));

    // For contrast, geqrf declares it too -- so a copy-paste that dropped geqrf's
    // predicate moves this assertion and not the ones above.
    EXPECT_TRUE((declares_native_tier_preferred<RouteTable<Op::geqrf, float>, GeqrfShape>))
        << "geqrf's measured tier window must not be deleted by a WP6 copy-paste";

    // getrs DECLARES IT NOW, and that assertion flipped when the op gained a
    // second native arm. It used to read EXPECT_FALSE on the ground that "with one
    // native route there is no native-vs-native question" -- true while
    // kGetrsOrder held a single {Native, Blocked}. It now holds {Native, CTA}
    // (the FUSED narrow-RHS kernel, src/extensions/getrs_fused.cc) ahead of it,
    // and the two are 7.9x apart at nrhs = 1, so the hook is the ONLY instrument
    // that can carry that window without also dragging vendor-present traffic
    // (route_resolve.hh:40-70).
    EXPECT_TRUE((declares_native_tier_preferred<GetrsTable, GetrsShape>))
        << "getrs has two native tiers; the order array alone cannot follow a "
           "crossover between them";

    // getri is still single-arm and must NOT declare it: with one native route
    // there is no native-vs-native question, and route_orgqr.hh sets the precedent
    // of simply not having the member.
    EXPECT_FALSE((declares_native_tier_preferred<GetriTable, GetriShape>));
}

TEST(RouteGetrf, BatchlasGetrfRouteIsActuallyRead) {
    // getrf had NO route resolution at all before WP6 -- the facade was six
    // `if constexpr (!vendor) throw; else vendor;` bodies
    // (factorization.cc:464-541) -- so "pin the native path with
    // BATCHLAS_GETRF_ROUTE" was a claim about a code path nobody had exercised for
    // this op. The canonical spelling needs no registry entry (parse_route_env
    // synthesises it, route_env.hh:205-217), but that is exactly the sort of claim
    // that turns out to be false.
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
        // AN UNRECOGNISED VALUE IS SILENTLY {Auto, Auto}, WHICH IS THE VENDOR.
        // This is the measurement trap the campaign has hit before: a "native" run
        // that looks identical to the vendor probably IS the vendor.
        ScopedEnv e("BATCHLAS_GETRF_ROUTE", "not-a-route");
        const auto p = parse_route_env(Op::getrf);
        EXPECT_FALSE(p.found);
        EXPECT_TRUE(p.unparsed) << "a typo must be reported, not silently Auto";
    }
}

// ---------------------------------------------------------------------------
// GETRS. One native arm, and the one op in this family with a live variant.
// ---------------------------------------------------------------------------

TEST(RouteGetrs, VendorFreeFallbackHandsOverTheNativeRoute) {
    // The speed-threshold guard again, and it matters here for a reason getrf's
    // does not: getrs's composed arm is a MEASURED LOSS at nrhs=1 (geomean 0.36x
    // over 28 cells, 25 losses). The temptation to write `if (s.nrhs() < 8) return
    // false;` in supports() is therefore concrete rather than hypothetical -- and
    // it would remove the vendor-free route at exactly the shape a vendor-free
    // build has no alternative for. That threshold belongs in preferred().
    const auto s = getrs_shape(/*order=*/32, /*nrhs=*/1, /*batch=*/1);

    EXPECT_TRUE(GetrsTable::supports(kGetrsBlocked, s))
        << "nrhs and batch are speed questions; neither may gate CORRECTNESS, even "
           "though nrhs=1 is measured 0.36x geomean";
    EXPECT_FALSE(GetrsTable::preferred(kGetrsBlocked, s))
        << "the COMPOSITION is never preferred at any width the fused tier serves; "
           "it is 0.36x geomean here and the window belongs to CTA alone";

    EXPECT_TRUE(is_native(resolve_getrs_route<float>(kGetrsAuto, s, false)));

    // AND WITH A VENDOR PRESENT IT IS NOW NATIVE TOO, at this shape, because
    // nrhs = 1 is inside the measured window (2.26x geomean over 111 cells, min
    // 1.24x, flat across every batch ladder at five orders). This assertion used
    // to read is_vendor and it was RIGHT to, at the time: preferred() was all
    // false. It is written the other way round now, and the pairing is the point
    // -- supports() is unchanged by the window, so the vendor-free line above
    // still passes for its own reason.
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
    // transA is a LIVE routing input for this op -- the only field in the LU family
    // that separates a coverage row from its neighbour (coverage.cc:52-58's
    // variant_key carries transA; getrf and getri set none of variant_key's fields
    // at all). It is also a genuine algorithm fork: NoTrans applies P first and
    // solves L then U, while Trans/ConjTrans solves U^T/U^H then L^T/L^H and
    // applies P^T LAST, on the output, in reverse.
    //
    // All three must be SUPPORTED. The natural wrong edit is to refuse
    // Trans/ConjTrans "until the reversed path is written", which is a gate that
    // goes stale silently the moment it IS written -- and the vendor is measured
    // correct in all three modes, so a native arm that cannot serve one must say so
    // with a test that fails without it, at the moment it lands.
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

// THE MEASURED nrhs WINDOW, both sides of it, and the ABSENT DRIVER.
//
// REPLACES PreferredIsFalseEverywhereAndAbsentDriverIsUnsupported, which asserted
// that preferred() is false for every route at every shape. That was true when it
// was written and is now the OPPOSITE of the shipped table, so it is rewritten
// rather than deleted: the property it guarded -- "a preferred() window may not
// appear without the grid that justifies it" -- still needs a guard, and the way
// to guard a window is to pin BOTH of its sides.
//
// THE WINDOW (experiments/wp6_perf/bench/README.md, 354 + 134 measured cells):
//     nrhs <= 2  for every type and order   -- clause A
//   + nrhs <= 4  for float only             -- clause B
// and NOTHING else. Everything wider is a measured loss somewhere on its batch
// ladder, and the composition is never preferred at any width at all.
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

    // ---- clause B is FLOAT ONLY, and that is the half most likely to be
    // widened by someone who reads "nrhs <= 4" and drops the type test ------
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
    // nrhs 64 USED TO BE IN THIS LOOP and is not any more: WP8 lands clause C,
    // which prefers the COMPOSITION for float at nrhs >= 64. That is the whole
    // point of an armed guard -- this loop went red when the clause landed and
    // had to be moved, rather than the clause landing unnoticed.
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

    // ---- CLAUSE C: THE WIDE-nrhs COMPOSITION WINDOW (WP8 routing pass) ------
    //
    // AXIS: GetrsShape::nrhs(), which is B.cols(). NOT order(). A predicate on
    // the wrong extent inverts this window -- it would admit the wide-ORDER
    // narrow-RHS calls where the composition measures 0.09-0.36x of the vendor
    // -- and that error was caught twice in WP7. The order loop below exists to
    // prove the clause does NOT read order: every order gives the same answer.
    //
    // THE BOUNDARY IS PER TYPE and comes from experiments/wp8_getrs/
    // clause_summary.txt, two native passes and two vendor passes, quoted at the
    // worse pass: float >= 64 (30 cells, geomean 2.837, min 1.765) and double
    // >= 128 (15 cells, geomean 1.865, min 1.279). cfloat and cdouble earn
    // nothing at any width and each has its refuting cell in route_getrs.hh.
    //
    // AND CLAUSE C CARRIES A BATCH FLOOR, WHICH CLAUSES A AND B DO NOT. It is
    // CONSERVATIVE and known to be: at nrhs = 128 the composition still wins by
    // 3.56x-5.96x at batch 32 and 64, so the floor gives up measured wins rather
    // than excluding measured losses. It is set at the campaign's saturation
    // rung because below 32 the region is ragged and the only readings there
    // came from a contaminated sweep. Both sides of it are pinned below.
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

            // double: IN at 128, OUT at 127 AND OUT at 64 -- the half most
            // likely to be widened by someone who reads "nrhs >= 64" and drops
            // the type test. Its refuting cell is 0.9984 at n=64 nrhs=64 b=2048.
            const auto d_in  = getrs_shape(order, /*nrhs=*/128, batch);
            const auto d_127 = getrs_shape(order, /*nrhs=*/127, batch);
            const auto d_64  = getrs_shape(order, /*nrhs=*/64,  batch);
            EXPECT_TRUE(GetrsTableD::preferred(kGetrsBlocked, d_in));
            EXPECT_FALSE(GetrsTableD::preferred(kGetrsBlocked, d_127));
            EXPECT_FALSE(GetrsTableD::preferred(kGetrsBlocked, d_64));
            EXPECT_TRUE(is_native(resolve_getrs_route<double>(kGetrsAuto, d_in, true)));
            EXPECT_TRUE(is_vendor(resolve_getrs_route<double>(kGetrsAuto, d_64, true)));

            // cfloat and cdouble: NOTHING, at any width. cfloat scored a clean
            // 15 cells / geomean 1.974 / min 1.482 on the directly measured
            // rungs and was in the clause until the gap sweep found 0.9944 at
            // n=64 nrhs=128 b=1024 -- a dip in the MIDDLE of its ladder, with
            // 1.2901 at b=512 and 1.4824 at b=2048 on either side of it, which
            // no boundary in batch, order or nrhs can exclude.
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

    // THE CLAUSE IS ON nrhs AND NOT ON order, PROVED BY CONSTRUCTION. Hold nrhs
    // and sweep order over four decades: the answer may not move. If the
    // predicate were spelled `s.order() >= 64` this loop is what goes red.
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

    // THE WINDOW IS NOT A CORRECTNESS GATE, and this is the pairing that keeps a
    // pinned route honest: at nrhs = 8 the fused tier is NOT preferred but IS
    // supported, so `BATCHLAS_GETRS_ROUTE=native:cta` still runs the fused kernel
    // there rather than falling through automatic() to cuBLAS and measuring the
    // route it was pinned away from (route_resolve.hh:165 -> :175).
    {
        const auto s = getrs_shape(/*order=*/256, /*nrhs=*/8, /*batch=*/512);
        EXPECT_TRUE(GetrsTable::supports(kGetrsCta, s));
        EXPECT_FALSE(GetrsTable::preferred(kGetrsCta, s));
        const Route pinned = resolve_getrs_route<float>(kGetrsCta, s, true);
        EXPECT_TRUE(is_native(pinned) && pinned.algo == Algorithm::CTA);
    }

    // ---- and the window may not outrun the CAPACITY -----------------------
    // Inside the window by nrhs, outside it by elements: supports() refuses, so
    // preferred() never gets to select an unlaunchable route.
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
    // driver (route_resolve.hh:146-163).
    //
    // AND THE ANSWER CHANGED WHEN THE FUSED TIER LANDED, which is a fact about the
    // ENV VARIABLE and not only about this function: `BATCHLAS_GETRS_ROUTE=native`
    // used to mean the composition, because it was the only native route in
    // kGetrsOrder; CTA is now listed first, so the bare origin means the FUSED
    // KERNEL wherever the right-hand side is resident. Any baseline recorded with
    // a bare `native` pin -- experiments/wp6_lu/bench/run_cells.sh:37 and
    // kernels/run_grid.sh:39 export it into all three LU variables at once -- is
    // measuring a different getrs today than when it was recorded. That is a
    // measurement-comparability trap, not a correctness one, and this test is
    // where it is written down.
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
// GETRI. One native arm, a composition over the routed trsm, and the LU op with a
// measured win to go and get.
// ---------------------------------------------------------------------------

TEST(RouteGetri, VendorFreeFallbackHandsOverTheNativeRoute) {
    // n=40, batch=2 are inverse_tests' ACTUAL extents (tests/inverse_tests.cc:10-39),
    // and inverse_tests fails today on "no route for getri<float>" -- getri is the
    // first LU op inv.cc's layout asks about (inv.cc:35 sizes getri before :36 sizes
    // getrf). That suite closes if and only if these extents stay supported.
    const auto s = getri_shape(/*order=*/40, /*batch=*/2);

    EXPECT_TRUE(GetriTable::supports(kGetriBlocked, s))
        << "batch size and order are speed questions; neither may gate CORRECTNESS "
           "-- and these are inverse_tests' own extents";
    EXPECT_FALSE(GetriTable::preferred(kGetriBlocked, s));

    EXPECT_TRUE(is_native(resolve_getri_route<float>(kGetriAuto, s, false)));
    EXPECT_TRUE(is_vendor(resolve_getri_route<float>(kGetriAuto, s, true)));
}

TEST(RouteGetri, CorrectnessGatesIncludeTheOnesInheritedFromTrsm) {
    // getri's native arm is a composition over the ROUTED trsm, so
    // RouteTable<Op::trsm,T>::supports()' structural gates (route_trsm.hh:132-160)
    // are TRANSCRIBED here -- silently omitting an inherited gate is the
    // wrong-answer class route_orgqr.hh:41-49 records.
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

// THE MEASURED ORDER WINDOW (WP8 routing pass), replacing the all-false sweep
// this case used to be. The old sweep varied order x batch for SCALAR FLOAT ONLY
// and checked the other three types at ONE shape; a per-type clause needs a
// per-type sweep or its cfloat and double boundaries are guarded by nothing.
//
// THE AXIS IS GetriShape::order(), which is `k`. There is NO batch term, and
// that is measured rather than omitted: at batch 1..32 the native driver beats
// cuBLAS by 1.7x-28x for every type at every order measured, because cuBLAS's
// batched getri is a per-item loop there.
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
        // The old header put the crossover at 128 for cfloat as well as float.
        // It is wrong: cfloat n=128 is an OUTRIGHT LOSS at 0.71 in the middle of
        // its own ladder, which is why the boundary is per type and not shared.
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
        // double is the near-miss and the one most likely to be widened: n=512
        // is clean at 1.296-1.311, but n=256 is 1.12-1.16 (below the bar without
        // losing), n=1024 is 1.1551-1.1619 and FALLING with the last rung
        // measured being the lowest, and n=2048 is 1.085 outright. A clause
        // admitting exactly one order is the leg predicate this campaign has
        // already found twice. cdouble loses at n=512 (0.954) and tops out at
        // 1.136 anywhere.
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

    // ...AND INSIDE THE WINDOW, WHICH IS THE HALF THAT ONLY MATTERS NOW THAT
    // THERE IS ONE. preferred() must still say yes -- it is a SPEED predicate and
    // repeating the capability test in it is the defect route_resolve.hh:60-70
    // records -- while supports() says no and the vendor takes the shape.
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
// THE THREE LU OPS ARE PINNED BY THREE INDEPENDENT VARIABLES, and that is the
// silent-wrong-answer channel WP6's kernel step has to close.
// ---------------------------------------------------------------------------
TEST(RouteLuFamily, TheThreeOpsResolveIndependentlyAndThatIsThePivotHazard) {
    // Measured, through the public API against a host LAPACKE oracle: the physical
    // pivot format is BACKEND-DEPENDENT. cublas.cc:1508 / rocsolver.cc:227 do
    // pivots.as_span<int>() and store PACKED 1-BASED INT32 in the first half of the
    // caller's int64 buffer (0/18 mismatches against LAPACKE read that way, 18/18
    // read as int64), while netlib_lapack.cc:1312-1320 WIDENS an int scratch into
    // genuine int64.
    //
    // A native getrf must therefore agree with WHATEVER SERVES getri on the same
    // call -- and this case exists to show that the mixture is reachable through
    // ordinary configuration, not through misuse. Three independent variables,
    // three independent tables, no shape field able to express "the op downstream
    // of me resolved differently".
    ClearRouteEnv clear_f(Op::getrf);
    ClearRouteEnv clear_s(Op::getrs);
    ClearRouteEnv clear_i(Op::getri);

    ScopedEnv ef("BATCHLAS_GETRF_ROUTE", "cta");
    ScopedEnv ei("BATCHLAS_GETRI_ROUTE", "vendor");

    EXPECT_EQ(parse_route_env(Op::getrf).route, (Route{Origin::Native, Algorithm::CTA}));
    EXPECT_EQ(parse_route_env(Op::getri).route, (Route{Origin::Vendor, Algorithm::Auto}));
    EXPECT_FALSE(parse_route_env(Op::getrs).found)
        << "and the third is untouched -- the three do not share a variable";

    // With capabilities present, that pin really does produce a mixed pair. (Today
    // both resolve to the vendor because nothing is linked, which is why this
    // asserts on the PARSED routes and on supports(), not on the resolved pair.)
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
// WP7 -- gemv. THE ROUTE TABLE'S SELECTION BEHAVIOUR, WHICH HAD NO TEST AT ALL.
//
// WP7 landed a native gemv, a route table and 232 kernel-level cases, and added
// ZERO assertions here. That is the same class of hole as the getrs blind guard
// recorded above, one step earlier: not a vacuous assertion, but no assertion.
//
// AND THE HOLE WAS PROVED NON-INVASIVELY BEFORE THESE WERE WRITTEN, from two
// directions:
//   * VENDOR-PRESENT: all 114 gemv decisions in a full `ctest -LE slow` capture
//     resolve to vendor:auto. No vendor-present test can observe supports(),
//     preferred() or kGemvOrder at all.
//   * VENDOR-FREE: `BATCHLAS_GEMV_ROUTE=native:direct ./tests/gemv_tests` takes
//     the CTA kernel out of EVERY decision -- zero cta rows in the coverage
//     dump -- and all 232 cases still pass. Nothing asserted WHICH native route
//     was chosen.
//
// THE ONE STRUCTURAL PROPERTY WP7 EXISTS FOR -- Direct having NO is_gpu clause,
// so a native_cpu queue can take it -- was already guarded behaviourally, by
// the 20 Backend::NETLIB rows of gemv_tests.cc going red if it were added. It is
// stated here as an ASSERTION as well, because a behavioural guard in another
// binary is not where someone editing this table will look.
//
// THE V1 TRAP, AVOIDED DELIBERATELY. gemv_shape() below sets direct_available,
// cta_available, has_sg32 AND is_gpu. Leave any of them at its default and
// supports() is false on every shape, every assertion in this section holds
// vacuously, and the section reports green through a flip and its inverse --
// which is exactly how getrs's 78/78 survived. That the helper is ARMED is
// itself checked, by RouteGemv.HelperIsArmed below: it forces the capabilities
// false and requires the supports() answers to CHANGE.
//
// AND THE ARMING WAS MEASURED, NOT ASSERTED. Seven breaks, each applied to the
// named file, rebuilt and run, then reverted (91 cases total):
//
//   break      file            red   named cases turned red
//   --------------------------------------------------------------------------
//   unarmed    this file        7    the helper stops setting direct_available
//                                    and cta_available -- i.e. THE V1 TRAP
//                                    ITSELF. HelperIsArmed catches it, and so do
//                                    six others, so the section cannot go
//                                    vacuous quietly.
//   nosg32     this file        4    the helper stops setting has_sg32
//   gpugate    route_gemv.hh    2    `if (!s.is_gpu) return false;` added to the
//                                    Direct arm -- THE WP7 DELIVERABLE FLIPPED.
//                                    DirectHasNoGpuGate... goes red here, and in
//                                    the real build the 20 Backend::NETLIB rows
//                                    of gemv_tests.cc go red with it.
//   order      route_gemv.hh    3    kGemvOrder reordered to Direct-before-CTA
//   clause     route_gemv.hh    2    a preferred() clause lands (the exact one
//                                    the perf audit refuted)
//   sg32gate   route_gemv.hh    2    CTA stops requiring an enumerated 32
//   zeroext    route_gemv.hh    1    m == 0 / n == 0 refused instead of
//                                    quick-returned -- ZeroExtentIsSupported...
//                                    ALONE, which is what a single-purpose case
//                                    should do
//
// A NOTE FOR WHOEVER READS THE NEXT route_diff. The cases below call
// `resolve_gemv_route`, which is the INSTRUMENTED resolver, so each one writes a
// row into the coverage table with `backend = AUTO` -- six of them. They are
// fabricated shapes from a pure-layer test, not decisions the library made, and
// three of them read `native:direct` even in a vendor-present capture because
// they pass `vendor_available = false` explicitly. This is the file's existing
// practice, not a gemv invention: the capture already carries AUTO rows for
// eleven ops (gemm 2542, ormqr 152, trsm 93, potrf 59 ...). Filter on
// `backend != AUTO` before concluding anything about what the library routes.
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

// THE WP7 DELIVERABLE, AS AN ASSERTION. Adding `if (!s.is_gpu) return false;` to
// the Direct arm turns this red -- and, in the real build, turns the 20
// Backend::NETLIB rows of gemv_tests.cc red with it, because the vendor-free
// walk then finds NO route for a native_cpu queue and the facade throws.
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
    //    sub_group_sizes, never from MAX_SUB_GROUP_SIZE -- that returns
    //    sub_group_sizes()[0] and is wrong in both directions.
    EXPECT_FALSE(GemvTable::supports(
        kGemvCta, gemv_shape(256, 256, 512, Transpose::Trans, /*is_gpu=*/true,
                             /*has_sg32=*/false)));
}

// A HETEROGENEOUS BATCH IS A CORRECTNESS GATE FOR GEMV, not merely un-preferred:
// one launch covers the whole batch with a single (m, n, ld, stride) tuple, and
// VectorView has no active-size concept at all, so there is nothing to walk on
// the x and y side. It must be refused for BOTH native tiers.
TEST(RouteGemv, HeterogeneousBatchIsRefusedByBothNativeTiers) {
    const auto het = gemv_shape(256, 256, 512, Transpose::Trans,
                                /*is_gpu=*/true, /*has_sg32=*/true,
                                /*direct_available=*/true, /*cta_available=*/true,
                                /*heterogeneous=*/true);
    EXPECT_FALSE(GemvTable::supports(kGemvCta, het));
    EXPECT_FALSE(GemvTable::supports(kGemvDirect, het));
    EXPECT_TRUE(is_vendor(resolve_gemv_route<float>(kGemvAuto, het, true)));
}

// preferred() SHIPS ALL-FALSE, AND THAT IS A MEASURED RESULT RATHER THAN AN
// OMISSION -- cuBLAS gemvStridedBatched is at 94-105% of the achievable DRAM
// roof on 90 of 92 reproducing cells. This case is what makes a future clause
// VISIBLE: land one and this goes red, and whoever lands it has to come here and
// say which cells it is for.
// THE float TABLE IS STILL ALL-FALSE EVERYWHERE, AND THAT IS NOW A CLAUSE OF ITS
// OWN RATHER THAN THE DEFAULT. WP8 lands a complex<double> window; this case
// pins that it did not leak into the other three types. It cannot see the window
// at all -- GemvTable is RouteTable<Op::gemv, float> -- which is exactly why the
// cdouble cases below had to be written separately, and why a suite that only
// counts 91/91 is not evidence about a per-type clause.
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

// ---- THE complex<double> TRANSPOSED WINDOW (WP8 routing pass) ---------------
//
// THE AXES ARE out_len() AND red_len(), AND THE WHOLE POINT OF SPELLING THEM
// THAT WAY IS THAT THEY SWAP WITH transA. gemv_shape() takes (m, n), so every
// case below converts ONCE, here, and never reasons in m and n again:
// under Trans and ConjTrans, out_len == n == cols and red_len == m == rows.
namespace {
GemvShape gemv_band(int64_t out_len, int64_t red_len, int64_t batch, Transpose t) {
    return gemv_shape(/*m=*/red_len, /*n=*/out_len, batch, t);
}
using GemvTableCD = RouteTable<Op::gemv, std::complex<double>>;
} // namespace

TEST(RouteGemv, CdoubleTransposedBandIsPreferredAndEveryBoundaryIsPinned) {
    const Transpose trs[2] = {Transpose::Trans, Transpose::ConjTrans};

    // INSIDE, both transposed spellings. ortho.cc issues ConjTrans for every
    // complex type, so C is the LIVE path and pinning only T would guard the
    // wrong half of a clause that says `transA != NoTrans`.
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
        // red_len lower edge: 64 in, 63 out. Below it the vendor is only
        // PARTIALLY dipped and the margin falls under the 1.15x bar
        // (1.1028 at red_len 40) or inverts (0.9515 at red_len 32).
        EXPECT_TRUE (GemvTableCD::preferred(kGemvCta, gemv_band(512,  64, 512, t)));
        EXPECT_FALSE(GemvTableCD::preferred(kGemvCta, gemv_band(512,  63, 512, t)));
        EXPECT_FALSE(GemvTableCD::preferred(kGemvCta, gemv_band(512,  48, 512, t)));
        EXPECT_FALSE(GemvTableCD::preferred(kGemvCta, gemv_band(512,  32, 512, t)));
        // red_len upper edge: 352 in, 353 out. cuBLAS is back at the roof
        // (901-906 GB/s) at 384 and the ratio there is 1.0304 / 1.0314.
        EXPECT_TRUE (GemvTableCD::preferred(kGemvCta, gemv_band(512, 352, 512, t)));
        EXPECT_FALSE(GemvTableCD::preferred(kGemvCta, gemv_band(512, 353, 512, t)));
        EXPECT_FALSE(GemvTableCD::preferred(kGemvCta, gemv_band(512, 384, 512, t)));
        // out_len lower edge: 256 in, 255 out. At out_len 192 the ratio is
        // 0.9988; at 64 and 32 cuBLAS reads 1777 and 1597 GB/s, above this
        // device's DRAM peak, because it is converting L2 residency into
        // bandwidth -- the family route_gemv.hh declines to chase.
        EXPECT_TRUE (GemvTableCD::preferred(kGemvCta, gemv_band(256, 128, 512, t)));
        EXPECT_FALSE(GemvTableCD::preferred(kGemvCta, gemv_band(255, 128, 512, t)));
        EXPECT_FALSE(GemvTableCD::preferred(kGemvCta, gemv_band(192, 128, 512, t)));
        // batch floor: 320 in, 319 out. This is cuBLAS's own kernel-selection
        // threshold, not a fitted constant: at out_len 512 red_len 128 it runs
        // 930.1 GB/s at batch 256 and 360.4 at batch 320.
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

    // NoTrans IS EXCLUDED, and by the clause itself rather than only by
    // supports(). Under NoTrans out_len and red_len SWAP, so a shape that is
    // inside the band transposed is a different shape entirely untransposed --
    // this is the pairing that catches a predicate written on the wrong extent.
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

// THE CLAUSE READS red_len() AND out_len(), NOT m AND n -- PROVED BY TRANSPOSING
// A SINGLE SHAPE AND REQUIRING THE ANSWER TO MOVE.
//
// This is the case that would have caught the WP7 defect twice over. Take the
// physical matrix A with 128 rows and 512 columns. Transposed it is red_len 128,
// out_len 512 -- INSIDE the band. Untransposed it is out_len 128, red_len 512 --
// outside on BOTH terms. A predicate spelled `s.m >= 64 && s.m <= 352 &&
// s.n >= 256` returns the same answer for both and this case stays green; the
// clause as written returns different answers, which is the whole content of
// "name the axis explicitly".
TEST(RouteGemv, TheBandIsOnRedLenAndInvertsUnderNoTrans) {
    auto A = gemv_shape(/*m=*/128, /*n=*/512, /*batch=*/512, Transpose::Trans);
    EXPECT_EQ(A.red_len(), 128);
    EXPECT_EQ(A.out_len(), 512);
    EXPECT_TRUE(GemvTableCD::preferred(kGemvCta, A));

    auto B = A; B.transA = Transpose::NoTrans;
    EXPECT_EQ(B.red_len(), 512);
    EXPECT_EQ(B.out_len(), 128);
    EXPECT_FALSE(GemvTableCD::preferred(kGemvCta, B));

    // And the mirror image: 512 rows by 128 columns is OUTSIDE transposed
    // (red_len 512 > 352, out_len 128 < 256) although its m and n are exactly
    // A's swapped.
    auto C = gemv_shape(/*m=*/512, /*n=*/128, /*batch=*/512, Transpose::Trans);
    EXPECT_EQ(C.red_len(), 512);
    EXPECT_EQ(C.out_len(), 128);
    EXPECT_FALSE(GemvTableCD::preferred(kGemvCta, C));
}

// THE WINDOW IS NOT A CORRECTNESS GATE, AND A CAPABILITY THE BUILD DOES NOT HAVE
// STILL WINS OVER IT. Inside the band but on a device that does not enumerate a
// sub-group size of 32, or in a build with no CTA kernel linked, supports() must
// refuse and the vendor must take it -- otherwise the clause selects a launch
// that is not there.
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

// WITH preferred() ALL-FALSE, AN AUTO GEMV IS THE VENDOR WHEREVER THERE IS ONE
// AND THE NATIVE LADDER WHEREVER THERE IS NOT. Both halves matter: the first is
// WP7's route-neutrality claim in the vendor-present build (0 moved decisions in
// route_diff), the second is the burn-down.
TEST(RouteGemv, AutoTakesTheVendorWhenPresentAndTheLadderWhenNot) {
    const auto tr = gemv_shape(256, 256, 512, Transpose::Trans);
    const auto no = gemv_shape(256, 256, 512, Transpose::NoTrans);

    EXPECT_TRUE(is_vendor(resolve_gemv_route<float>(kGemvAuto, tr, true)));
    EXPECT_TRUE(is_vendor(resolve_gemv_route<float>(kGemvAuto, no, true)));

    // Vendor-free: the ladder is CTA, then Direct, then the vendor -- tighter
    // first, so a transposed GPU shape gets the coalesced body and everything
    // else gets Direct.
    EXPECT_EQ(resolve_gemv_route<float>(kGemvAuto, tr, false).algo, Algorithm::CTA);
    EXPECT_EQ(resolve_gemv_route<float>(kGemvAuto, no, false).algo, Algorithm::Direct);
}

// A BARE `native` DOES NOT MEAN CTA. route_gemv.hh's env note said it did, and
// measurement disagreed: the walk picks the first SUPPORTED native route, so a
// bare pin lands on Direct for NoTrans, for a CPU device and for a GPU without
// an enumerated 32 -- 76 of 104 decisions in gemv_tests. This case is the note's
// guard, so the two cannot drift apart again.
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

// PINNING A ROUTE THE SHAPE CANNOT TAKE IS SILENT, AND ITS OUTCOME DEPENDS ON
// THE BUILD. This is not a bug being asserted as correct -- it is the campaign's
// standing fall-through behaviour -- but it has cost time twice and it is worth
// having in a test so nobody re-derives it from a benchmark that lies.
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

// out_len() AND red_len() SWAP WITH transA, AND THAT IS THE TRAP B3 NAMED. The
// one measured cuBLAS slow region is a band on **m**, which under a transposed
// transA is red_len(), NOT out_len(). A predicate written on out_len() would test
// n and invert the window. Nothing in the shipped table reads them -- preferred()
// is all-false -- so this is here to make the mapping WRONG-PROOF for whoever
// adds the first clause.
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

// B2, IN THE PURE LAYER. GemvShape must not re-declare transA or is_gpu:
// resolve_route SLICES it to OpShape on the way into the coverage table, so a
// shadowing member would be written by the shape builder and then not copied,
// every gemv coverage row would report NoTrans, and gemv's two arms -- which are
// different KERNELS, not different flags -- would collapse into one
// first-writer-wins row.
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

// DEGENERATE EXTENTS. m == 0 or n == 0 is a LEGAL call that the native kernel
// serves by quick-returning without touching y, exactly as reference ?GEMV and
// both vendors do -- so it must stay SUPPORTED. A negative extent or an empty
// batch has no launch geometry and goes to the vendor.
TEST(RouteGemv, ZeroExtentIsSupportedButNegativeExtentIsNot) {
    EXPECT_TRUE(GemvTable::supports(kGemvDirect, gemv_shape(0, 6, 3, Transpose::Trans)));
    EXPECT_TRUE(GemvTable::supports(kGemvDirect, gemv_shape(5, 0, 3, Transpose::NoTrans)));
    EXPECT_FALSE(GemvTable::supports(kGemvDirect, gemv_shape(-1, 6, 3)));
    EXPECT_FALSE(GemvTable::supports(kGemvDirect, gemv_shape(6, -1, 3)));
    EXPECT_FALSE(GemvTable::supports(kGemvDirect, gemv_shape(6, 6, 0)));
    EXPECT_TRUE(is_vendor(resolve_gemv_route<float>(kGemvAuto, gemv_shape(6, 6, 0), true)));
}

// Algorithm::Auto IS NOT A NATIVE GEMV ROUTE. gemv has two native tiers, so a
// bare "native" names neither; resolve_route walks the order restricted to the
// origin to pick one. supports() must say false, or the walk would stop on a
// route with no kernel behind it.
TEST(RouteGemv, NativeAutoIsNotItselfASupportedRoute) {
    const auto s = gemv_shape(256, 256, 512, Transpose::Trans);
    EXPECT_FALSE(GemvTable::supports(kGemvNativeBare, s));
    EXPECT_FALSE(GemvTable::supports(Route{Origin::Native, Algorithm::Blocked}, s));
    EXPECT_TRUE(GemvTable::supports(kVendorAuto, s));
}

// THE ORDER ARRAY IS A CAPABILITY LADDER, TIGHTEST FIRST. It is asserted rather
// than assumed because it is the only thing that decides the vendor-free
// outcome: with preferred() all-false, the vendor-off walk takes the FIRST
// supported route in this order and nothing else influences it.
TEST(RouteGemv, OrderIsCtaThenDirectThenVendor) {
    ASSERT_EQ(GemvTable::order_end() - GemvTable::order_begin(), 3);
    EXPECT_EQ(GemvTable::order_begin()[0].origin, Origin::Native);
    EXPECT_EQ(GemvTable::order_begin()[0].algo, Algorithm::CTA);
    EXPECT_EQ(GemvTable::order_begin()[1].origin, Origin::Native);
    EXPECT_EQ(GemvTable::order_begin()[1].algo, Algorithm::Direct);
    EXPECT_TRUE(is_vendor(GemvTable::order_begin()[2]));
}
