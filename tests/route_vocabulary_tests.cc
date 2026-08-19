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
    // float, Side::Left, order 32 is a real cell of this shape, not a contrived
    // one: WP3 step 9 measures it at 0.71-0.87x, so it is a CORRECTNESS yes and
    // a SPEED no. If the two ever collapse into one boolean, this is the cell
    // that silently loses its vendor-free route.
    const auto s = trsm_shape(/*tri_order=*/32, /*q=*/128, /*batch=*/4096, /*cta_max=*/64);
    EXPECT_TRUE(TrsmTable::supports(kCta, s))
        << "a 32-order solve is inside a 64 capacity; this must be a CORRECTNESS yes";
    EXPECT_FALSE(TrsmTable::preferred(kCta, s))
        << "float Side::Left at order 32 measured 0.71-0.87x; it must not be preferred";
    EXPECT_TRUE(is_native(resolve_trsm_route<float>(kAuto, s, /*vendor_available=*/false)))
        << "un-preferred must never mean unroutable when there is no vendor";
}

// ---------------------------------------------------------------------------
// The measured window (WP3 step 9). These replace a pair of tests that pinned
// preferred() as all-false -- correct while nothing had been measured, and
// actively wrong now: the point of step 9 was to move traffic, so a test
// asserting that no traffic moves would have to be deleted or the flip
// abandoned. What is pinned instead is the SHAPE of the measurement, so that a
// later edit widening a window has to come with its own numbers.
// ---------------------------------------------------------------------------

TEST(RouteTrsm, FloatLeftCliffAtOrderSixteen) {
    // The sharpest boundary in the grid, and the only per-side split any type
    // has: 3.58x at order 8, 1.47x at 16, then 0.73x at 32 and never recovering
    // (0.61x at 256). Side::Right at the same orders wins throughout, so this
    // must NOT be spelled as a plain order cap.
    for (int64_t order : {8, 16}) {
        EXPECT_TRUE(TrsmTable::preferred(kCta, trsm_shape(order, 1024, 2048, 32, Side::Left)))
            << "float Side::Left order " << order << " measured a win";
    }
    for (int64_t order : {32, 64}) {
        const auto s = trsm_shape(order, 1024, 2048, 32, Side::Left);
        const Route r = (order <= 32) ? kCta : kBlocked;
        EXPECT_FALSE(TrsmTable::preferred(r, s))
            << "float Side::Left order " << order << " measured a LOSS";
    }
    // Same orders, other side: all preferred.
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
