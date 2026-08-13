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

// --- the unset-default asymmetry, which is real and easy to get wrong ------

TEST(RouteVocabulary, UnsetDefaultsDifferBetweenGemmAndLevel3) {
    // gemm_variant_request() returns Vendor when its variable is unset, while
    // parse_cublasdx_variant_request() returns Auto. That asymmetry is why the
    // level-3 native tile kernels run by default today and GEMM's does not.
    EXPECT_TRUE(is_vendor(legacy_unset_default(Op::gemm)));
    EXPECT_EQ(legacy_unset_default(Op::syrk).origin, Origin::Auto);
    EXPECT_EQ(legacy_unset_default(Op::symm).origin, Origin::Auto);
    EXPECT_EQ(legacy_unset_default(Op::trmm).origin, Origin::Auto);
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
