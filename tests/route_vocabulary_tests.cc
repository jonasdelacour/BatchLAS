// The dispatch vocabulary: Route parsing, the legacy environment spellings, and
// the per-op RouteTable supports()/preferred() windows.
// The legacy spellings appear in committed benchmark scripts and in the provenance of
// recorded results, so their mapping is pinned here. evidence: docs/perf/dispatch.md

#include <gtest/gtest.h>

#include <batchlas/blas/dispatch/route.hh>
#include <batchlas/blas/dispatch/route_env.hh>
#include <batchlas/util/env.hh>

#include <complex>
#include <string>

using namespace batchlas;
using namespace batchlas::dispatch;

namespace {

// The private ScopedEnv this file used to carry is gone. It saved, set and
// restored exactly as batchlas::ScopedEnvVar does, but a raw ::setenv is no
// longer enough: parse_route_env reads its two values from the batchlas::settings()
// snapshot (route_env.hh), which is loaded once before main() -- src/dispatch/
// coverage.cc takes it in a namespace-scope initialiser in an always-linked TU.
// ScopedEnvVar reloads that snapshot at both ends of its scope, so the cases
// below assert on the value they just pinned instead of on the ambient one.

// Clears both spellings so a case starts from a known state.
//
// ScopedEnvVar BORROWS its name rather than copying it, and neither key here is a
// string literal -- the canonical one is composed from op_env_stem, the legacy one
// from a std::string_view. The two buffers are therefore members, declared BEFORE
// the guards that point at them: members are initialised in declaration order and
// destroyed in reverse, so each name is alive before its guard is built and still
// alive when that guard restores.
struct ClearRouteEnv {
    explicit ClearRouteEnv(Op op)
        : canonical_key_("BATCHLAS_" + op_env_stem(op) + "_ROUTE"),
          legacy_key_(std::string(legacy_variable_for(op)).empty()
                          ? std::string("BATCHLAS_UNUSED_ROUTE_KEY")
                          : std::string(legacy_variable_for(op))),
          canonical_(canonical_key_.c_str(), nullptr),
          legacy_(legacy_key_.c_str(), nullptr) {}

    std::string canonical_key_;
    std::string legacy_key_;
    ScopedEnvVar canonical_;
    ScopedEnvVar legacy_;
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
    // `library` is an output the resolver fills in; two requests differing only in it are equal.
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

// gemm's legacy BATCHLAS_GEMM_VARIANT words (`native` = the vendor, `sycl`/`custom` = native)
// are flat-selection aliases now: src/ops/gemm/choice.hh legacy_aliases.

TEST(RouteVocabulary, LegacyLevel3CustomMeansTheFusedKernelNotRegisterTiled) {
    // THE SECOND COLLISION: in symm/syrk/syr2k/trmm the legacy "custom" names the
    // FUSED cuBLASDx kernel, while canonical "custom" is our register-tiled family.
    for (Op op : {Op::symm, Op::syrk, Op::syr2k, Op::trmm}) {
        const auto legacy = parse_legacy_route_value(op, "custom");
        ASSERT_TRUE(legacy.has_value()) << op_env_stem(op);
        EXPECT_EQ(legacy->algo, Algorithm::FusedDevice) << op_env_stem(op);
        EXPECT_TRUE(is_vendor(*legacy)) << "the fused kernel is NVIDIA's source";
    }

    const auto canonical = parse_route_value("custom");
    ASSERT_TRUE(canonical.has_value());
    EXPECT_EQ(canonical->algo, Algorithm::RegisterTiled);
    EXPECT_TRUE(is_native(*canonical));

}

TEST(RouteVocabulary, LegacyLevel3TileSpellingsSurvive) {
    // `tiles` and `narrow` have no canonical spelling; only the alias table can carry them.
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
    // syrk/syr2k's `gemm` is the deliberately WRONG route kept for measurement: it
    // computes both triangles and runs through gemm_cublasdx, so it is vendor, not native.
    for (Op op : {Op::syrk, Op::syr2k}) {
        const auto r = parse_legacy_route_value(op, "gemm");
        ASSERT_TRUE(r.has_value()) << op_env_stem(op);
        EXPECT_EQ(r->algo, Algorithm::DiagFullGemm) << op_env_stem(op);
        EXPECT_TRUE(is_vendor(*r)) << op_env_stem(op);
    }
}

TEST(RouteVocabulary, LegacyTrmmTriangularIsOneValueNotTwoReadings) {
    ClearRouteEnv clear(Op::trmm);
    ScopedEnvVar set("BATCHLAS_TRMM_VARIANT", "triangular");

    const auto parsed = parse_route_env(Op::trmm);
    ASSERT_TRUE(parsed.found) << "it is an opinion, not the absence of one";
    EXPECT_EQ(parsed.route.algo, Algorithm::TriangularTiles);
    EXPECT_TRUE(is_native(parsed.route));
}

TEST(RouteVocabulary, LegacyVendorSpellingMapsToVendorOrigin) {
    ClearRouteEnv clear(Op::trmm);
    ScopedEnvVar set("BATCHLAS_TRMM_VARIANT", "vendor");

    const auto parsed = parse_route_env(Op::trmm);
    ASSERT_TRUE(parsed.found);
    EXPECT_TRUE(is_vendor(parsed.route));
}

TEST(RouteVocabulary, LegacySyrkTriangularAndGramSurvive) {
    {
        ClearRouteEnv clear(Op::syrk);
        ScopedEnvVar set("BATCHLAS_SYRK_VARIANT", "triangular");
        const auto parsed = parse_route_env(Op::syrk);
        ASSERT_TRUE(parsed.found);
        EXPECT_EQ(parsed.route.algo, Algorithm::TriangularTiles);
    }
    {
        ClearRouteEnv clear(Op::syrk);
        ScopedEnvVar set("BATCHLAS_SYRK_VARIANT", "gram");
        const auto parsed = parse_route_env(Op::syrk);
        ASSERT_TRUE(parsed.found);
        EXPECT_EQ(parsed.route.algo, Algorithm::GramTiles);
    }
}

TEST(RouteVocabulary, LegacyProviderSpellingsSurvive) {
    ClearRouteEnv clear(Op::syev);
    ScopedEnvVar set("BATCHLAS_SYEV_PROVIDER", "two_stage");

    const auto parsed = parse_route_env(Op::syev);
    ASSERT_TRUE(parsed.found);
    EXPECT_EQ(parsed.route.algo, Algorithm::TwoStage);
    EXPECT_TRUE(is_native(parsed.route));
}

TEST(RouteVocabulary, CanonicalSpellingWinsOverLegacy) {
    ClearRouteEnv clear(Op::gemm);
    ScopedEnvVar legacy("BATCHLAS_GEMM_VARIANT", "vendor");
    ScopedEnvVar canonical("BATCHLAS_GEMM_ROUTE", "native:register_tiled");

    const auto parsed = parse_route_env(Op::gemm);
    ASSERT_TRUE(parsed.found);
    EXPECT_TRUE(is_native(parsed.route));
    EXPECT_FALSE(parsed.source.legacy);
}

// --- the unset default -----------------------------------------------------

TEST(RouteVocabulary, UnsetDefaultsAreAutoForEveryOp) {
    // Auto for every op; GEMM's unset default was Vendor until its window was measured.
    // evidence: docs/perf/gemm.md#the-auto-flip
    EXPECT_EQ(legacy_unset_default(Op::gemm).origin, Origin::Auto);
    EXPECT_EQ(legacy_unset_default(Op::syrk).origin, Origin::Auto);
    EXPECT_EQ(legacy_unset_default(Op::symm).origin, Origin::Auto);
    EXPECT_EQ(legacy_unset_default(Op::trmm).origin, Origin::Auto);

    // Auto defers to preferred(); a named route still wins.
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
    ScopedEnvVar set("BATCHLAS_GEMM_ROUTE", "not-a-route");
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


TEST(RouteVocabulary, AlgorithmEnumeratorValuesAreAbi) {
    auto value = [](Algorithm a) { return static_cast<int>(a); };
    EXPECT_EQ(value(Algorithm::Auto), 0);
    EXPECT_EQ(value(Algorithm::Direct), 1);
    EXPECT_EQ(value(Algorithm::CTA), 2);
    EXPECT_EQ(value(Algorithm::Blocked), 3);
    EXPECT_EQ(value(Algorithm::TwoStage), 4);
    EXPECT_EQ(value(Algorithm::Jacobi), 5);
    EXPECT_EQ(value(Algorithm::RegisterTiled), 6);
    EXPECT_EQ(value(Algorithm::SplitK), 7);
    EXPECT_EQ(value(Algorithm::ExpandGemm), 8);
    EXPECT_EQ(value(Algorithm::TriangularTiles), 9);
    EXPECT_EQ(value(Algorithm::GramTiles), 10);
    EXPECT_EQ(value(Algorithm::FusedDevice), 11);
    EXPECT_EQ(value(Algorithm::DiagFullGemm), 12);

    // Appended after the frozen block, not inserted into it.
    EXPECT_EQ(value(Algorithm::Tiny), 13);

    // Origin is installed too, and Vendor's value reaches is_vendor() in every table.
    EXPECT_EQ(static_cast<int>(Origin::Auto), 0);
    EXPECT_EQ(static_cast<int>(Origin::Native), 1);
    EXPECT_EQ(static_cast<int>(Origin::Vendor), 2);
}

// Algorithm::Tiny is INVISIBLE until the enum, to_string (the coverage CSV's chosen_algo
// column) and parse_algorithm_word all know it. A missing parse case is SILENT.
TEST(RouteVocabulary, TinyVocabularyRoundTrip) {
    EXPECT_EQ(to_string(Algorithm::Tiny), "tiny");
    ASSERT_TRUE(parse_algorithm_word("tiny").has_value());
    EXPECT_EQ(*parse_algorithm_word("tiny"), Algorithm::Tiny);
}

// GEQRF's table moved to flat selection (src/ops/geqrf/); its RouteGeqrf tests are
// ported to tests/geqrf_candidates_tests.cc.

// The LU family (getrf, getrs, getri) moved to flat selection: its route tests are ported to
// tests/get{rf,rs,ri}_candidates_tests.cc.

// spmm moved to flat selection: its route tests are ported to tests/spmm_candidates_tests.cc.
