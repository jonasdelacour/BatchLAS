// The dispatch vocabulary: Route parsing, the legacy environment spellings, and
// the per-op RouteTable supports()/preferred() windows.
// The legacy spellings appear in committed benchmark scripts and in the provenance of
// recorded results, so their mapping is pinned here. evidence: docs/perf/dispatch.md

#include <gtest/gtest.h>

#include <batchlas/blas/dispatch/route.hh>
#include <batchlas/blas/dispatch/route_env.hh>
#include <batchlas/blas/dispatch/route_orgqr.hh>
#include <batchlas/blas/dispatch/route_getrf.hh>
#include <batchlas/blas/dispatch/route_getrs.hh>
#include <batchlas/blas/dispatch/route_getri.hh>
#include <batchlas/blas/dispatch/route_spmm.hh>
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

// ---------------------------------------------------------------------------
// ORGQR's table. orgqr ships as ORMQR APPLIED TO AN IDENTITY, so its supports()
// TRANSCRIBES RouteTable<Op::ormqr,T>'s gates plus its own; omitting an inherited
// gate is the wrong-answer class. evidence: docs/perf/qr.md
// ---------------------------------------------------------------------------
namespace {

OrgqrShape orgqr_shape(int64_t rows, int64_t cols, int64_t batch,
                       bool blocked_available = true) {
    OrgqrShape s;
    s.op = Op::orgqr;
    s.scalar = ScalarKind::F32;
    s.backend = Backend::AUTO;   // same reason as geqrf_shape
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
    // orgqr is a per-item loop, so a gate back to it also restores a batch-sized workspace.
    const auto s = orgqr_shape(/*rows=*/64, /*cols=*/64, /*batch=*/1);

    EXPECT_TRUE(OrgqrTable::supports(kOrgqrBlocked, s));
    EXPECT_TRUE(OrgqrTable::preferred(kOrgqrBlocked, s))
        << "n = 64 is inside the measured window (27.10x float against the per-item "
           "vendor loop, docs/perf/small-n-baseline.md#orgqr)";
    EXPECT_TRUE(is_native(resolve_orgqr_route<float>(kOrgqrAuto, s,
                                                     /*vendor_available=*/false)));
    EXPECT_TRUE(is_native(resolve_orgqr_route<float>(kOrgqrAuto, s,
                                                     /*vendor_available=*/true)))
        << "with a vendor present n = 64 is native too -- the window covers it";

    // AND THE FALLBACK MUST STILL WORK ABOVE THE CEILING, which is the half a window in
    // supports() would destroy: n = 1024 is a recorded loss and takes the vendor, but a
    // vendor-free build has to reach the native arm there all the same.
    // evidence: docs/perf/qr.md#orgqr-grid
    const auto big = orgqr_shape(/*rows=*/1024, /*cols=*/1024, /*batch=*/1);
    EXPECT_TRUE(OrgqrTable::supports(kOrgqrBlocked, big))
        << "the ceiling is a SPEED bound; putting it in supports() would delete the "
           "vendor-free route above n = 512";
    EXPECT_FALSE(OrgqrTable::preferred(kOrgqrBlocked, big));
    EXPECT_TRUE(is_native(resolve_orgqr_route<float>(kOrgqrAuto, big, false)));
    EXPECT_TRUE(is_vendor(resolve_orgqr_route<float>(kOrgqrAuto, big, true)));
}

// THE MEASURED CEILING, and it is the whole window: native to n = 512, vendor above. NOT
// `is_native && supports` -- there is no losing cell INSIDE the measured range, so the
// bound comes from the cells above it. evidence: docs/perf/qr.md#the-shipped-orgqr-ceiling
TEST(RouteOrgqr, PreferredIsNativeUpToTheMeasuredCeiling) {
    // ---- INSIDE: native at every order the grid covers, every type, every batch.
    for (int64_t n : {1, 16, 32, 64, 256, 512}) {
        for (int64_t batch : {1, 8, 128, 2048}) {
            const auto s = orgqr_shape(n, n, batch);
            EXPECT_TRUE(OrgqrTable::preferred(kOrgqrBlocked, s))
                << "n " << n << " batch " << batch;
            EXPECT_FALSE(OrgqrTable::preferred(Route{Origin::Vendor, Algorithm::Auto}, s))
                << "the vendor is where the walk ENDS, never itself preferred";
            const Route r = resolve_orgqr_route<float>(kOrgqrAuto, s, true);
            EXPECT_TRUE(is_native(r) && r.algo == Algorithm::Blocked)
                << "n " << n << " batch " << batch;
        }
    }

    // ---- OUTSIDE: the recorded losses. docs/perf/qr.md#the-shipped-orgqr-ceiling
    for (int64_t n : {1024, 2048}) {
        for (int64_t batch : {1, 8, 128, 2048}) {
            const auto s = orgqr_shape(n, n, batch);
            EXPECT_FALSE(OrgqrTable::preferred(kOrgqrBlocked, s))
                << "n " << n << " batch " << batch << " is a recorded LOSS "
                   "(1024: cfloat 0.82x / cdouble 0.78x; 2048: float 0.41x, "
                   "cfloat 0.31x, cdouble 0.46x)";
            EXPECT_TRUE(is_vendor(resolve_orgqr_route<float>(kOrgqrAuto, s, true)))
                << "n " << n << " batch " << batch;
        }
    }

    // The ceiling has NO batch term on purpose: exactly ONE type recovers with batch,
    // and one type crossing is not a window. A batch clause needs its own measurement.
    // evidence: docs/perf/qr.md#the-shipped-orgqr-ceiling
    EXPECT_FALSE(OrgqrTable::preferred(kOrgqrBlocked, orgqr_shape(1024, 1024, 256)));

    // ---- THE EDGE, from both sides and on BOTH extents. Q's columns live in C^m,
    // so a 1024x512 view is 512 reflectors against 1024 rows: the m cost is real and
    // the bound is not a cols()-only test.
    EXPECT_TRUE (OrgqrTable::preferred(kOrgqrBlocked, orgqr_shape(512, 512, 1024)));
    EXPECT_FALSE(OrgqrTable::preferred(kOrgqrBlocked, orgqr_shape(513, 513, 1024)));
    EXPECT_TRUE (OrgqrTable::preferred(kOrgqrBlocked, orgqr_shape(512, 256, 1024)));
    EXPECT_FALSE(OrgqrTable::preferred(kOrgqrBlocked, orgqr_shape(1024, 512, 1024)))
        << "rows = 1024 is outside the measured range whatever the column count";

    // ---- ALL FOUR TYPES, on both sides. The grid is complete per type, so this is
    // not float's window generalised.
    {
        const auto in  = orgqr_shape(512, 512, 1024);
        const auto out = orgqr_shape(1024, 1024, 1024);
        EXPECT_TRUE((RouteTable<Op::orgqr, double>::preferred(kOrgqrBlocked, in)));
        EXPECT_TRUE((RouteTable<Op::orgqr, std::complex<float>>::preferred(kOrgqrBlocked, in)));
        EXPECT_TRUE((RouteTable<Op::orgqr, std::complex<double>>::preferred(kOrgqrBlocked, in)));
        EXPECT_FALSE((RouteTable<Op::orgqr, double>::preferred(kOrgqrBlocked, out)));
        EXPECT_FALSE((RouteTable<Op::orgqr, std::complex<float>>::preferred(kOrgqrBlocked, out)));
        EXPECT_FALSE((RouteTable<Op::orgqr, std::complex<double>>::preferred(kOrgqrBlocked, out)));
    }
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

    // Guard against vacuity: with the driver present the same shape IS native.
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

    // orgqr's native arm re-enters routed ormqr, which reads its own variable.
    EXPECT_EQ(legacy_variable_for(Op::ormqr), "BATCHLAS_ORMQR_PROVIDER");

    {
        const auto unset = parse_route_env(Op::orgqr);
        EXPECT_FALSE(unset.found);
        EXPECT_FALSE(unset.unparsed);
        EXPECT_EQ(legacy_unset_default(Op::orgqr).origin, Origin::Auto);
    }
    {
        ScopedEnvVar e("BATCHLAS_ORGQR_ROUTE", "blocked");
        const auto p = parse_route_env(Op::orgqr);
        ASSERT_TRUE(p.found) << "BATCHLAS_ORGQR_ROUTE was not read at all";
        EXPECT_EQ(p.route, (Route{Origin::Native, Algorithm::Blocked}));
        EXPECT_EQ(p.source.variable, "BATCHLAS_ORGQR_ROUTE");
        EXPECT_FALSE(p.source.legacy);
    }
    {
        ScopedEnvVar e("BATCHLAS_ORGQR_ROUTE", "vendor");
        const auto p = parse_route_env(Op::orgqr);
        ASSERT_TRUE(p.found);
        EXPECT_EQ(p.route, (Route{Origin::Vendor, Algorithm::Auto}));
    }
    {
        ScopedEnvVar e("BATCHLAS_ORGQR_ROUTE", "not-a-route");
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

// PERMISSIVE DEFAULTS, one hostile field per case -- as in geqrf_shape above.
GetrfShape getrf_shape(int64_t order, int64_t batch, int cta_max_n, int tiny_max = 0) {
    GetrfShape s;
    s.op = Op::getrf;
    s.scalar = ScalarKind::F32;
    // AUTO, deliberately -- the same reason as geqrf_shape's.
    s.backend = Backend::AUTO;
    s.m = order;
    s.n = order;
    s.k = order;          // THE ORDER, potrf's mapping and not geqrf's
    s.batch = batch;
    s.is_gpu = true;
    s.has_sg32 = true;
    s.cta_max_n = cta_max_n;
    // 0 means "the register tier is absent from this build". It must be set explicitly by
    // any case that asserts on the tiny window, or that case holds vacuously.
    s.tiny_max_n = tiny_max;
    s.blocked_available = (cta_max_n > 0);
    return s;
}

// THE FUSED TIER'S TWO CAPACITIES, AND THEY DEFAULT TO PRESENT: a helper leaving them
// at 0 makes RouteTable<getrs>::supports({Native, CTA}, s) false on every shape here,
// and every getrs assertion below then holds vacuously whatever the table says.
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
// preferred() decides on the TABLE's T and NOT on s.scalar.
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

TEST(RouteGetrf, TheMeasuredTinyWindowAndNothingElse) {
    // Tiny: float 5..32, cfloat 5..7 and 9..24. Every edge is a MEASURED non-winner.
    // evidence: docs/perf/lu.md#the-column-bucket
    // ARMED BREAK (R9): cfloat tiny_window upper edge 24 -> 16. EXPECTED: RED at 17, 20, 24.
    using F   = RouteTable<Op::getrf, float>;
    using CF  = RouteTable<Op::getrf, std::complex<float>>;
    using D   = RouteTable<Op::getrf, double>;
    using CD  = RouteTable<Op::getrf, std::complex<double>>;
    constexpr Route kTiny{Origin::Native, Algorithm::Tiny};
    constexpr Route kCta{Origin::Native, Algorithm::CTA};
    constexpr Route kAuto{Origin::Auto, Algorithm::Auto};

    auto hits = [](auto tbl, const GetrfShape& sh) {
        using Tbl = decltype(tbl);
        int n = 0;
        for (const Route* it = Tbl::order_begin(); it != Tbl::order_end(); ++it)
            if (Tbl::preferred(*it, sh)) ++n;
        return n;
    };

    // IN the window, exactly ONE tier answers (R8b) and Auto takes a native route.
    for (int64_t n : {5, 7, 8, 9, 12, 16, 17, 20, 22, 24, 32}) {
        const auto sh = getrf_shape(n, 16384, /*cta_max_n=*/128, /*tiny_max=*/32);
        EXPECT_TRUE(F::preferred(kTiny, sh)) << "float n=" << n;
        EXPECT_EQ(hits(F{}, sh), 1) << "float n=" << n;
        EXPECT_TRUE(is_native(resolve_getrf_route<float>(kAuto, sh, true))) << "float n=" << n;
    }
    for (int64_t n : {17, 20, 22}) {                    // the retired CTA band: tiny 1.36-1.53x
        const auto sh = getrf_shape(n, 16384, 128, 32);
        EXPECT_FALSE(F::preferred(kCta, sh)) << "float n=" << n;
        EXPECT_EQ(resolve_getrf_route<float>(kAuto, sh, true), kTiny) << "float n=" << n;
    }
    for (int64_t n : {5, 7, 9, 12, 16, 17, 20, 24}) {   // the column bucket: 17..24 1.15-1.45x
        const auto sh = getrf_shape(n, 16384, 128, 32);
        EXPECT_TRUE(CF::preferred(kTiny, sh)) << "cfloat n=" << n;
        EXPECT_EQ(hits(CF{}, sh), 1) << "cfloat n=" << n;
    }

    // The MEASURED brackets. Each of these is a cell that lost or tied, not a guess.
    for (int64_t n : {1, 4}) {                          // float n=4 ties at the DRAM roof
        const auto sh = getrf_shape(n, 65536, 128, 32);
        EXPECT_FALSE(F::preferred(kTiny, sh)) << "float n=" << n << " measured 0.99x at n=4";
        EXPECT_TRUE(is_vendor(resolve_getrf_route<float>(kAuto, sh, true)));
        // ...but the vendor-FREE walk still takes Tiny: ~3x the CTA tier here.
        EXPECT_EQ(resolve_getrf_route<float>(kAuto, sh, false), kTiny) << "float n=" << n;
    }
    for (int64_t n : {4, 8}) {                          // cfloat's two ties
        const auto sh = getrf_shape(n, 65536, 128, 32);
        EXPECT_FALSE(CF::preferred(kTiny, sh))
            << "cfloat n=" << n << ": n=8 falls to 1.086x one doubling deeper";
        EXPECT_EQ(resolve_getrf_route<std::complex<float>>(kAuto, sh, false), kTiny)
            << "cfloat n=" << n;
    }
    for (int64_t n : {25, 28, 32}) {                    // cfloat ties above 24: 0.97-1.10x
        const auto sh = getrf_shape(n, 65536, 128, 32);
        EXPECT_FALSE(CF::preferred(kTiny, sh)) << "cfloat n=" << n << " ties the vendor";
        EXPECT_TRUE(is_vendor(resolve_getrf_route<std::complex<float>>(kAuto, sh, true)));
    }

    // fp64 is unrouted here on purpose: 1/64 rate on this part, and no grid.
    for (int64_t n : {8, 12, 16, 32}) {
        const auto sh = getrf_shape(n, 16384, 128, 32);
        EXPECT_FALSE(D::preferred(kTiny, sh));
        EXPECT_FALSE(CD::preferred(kTiny, sh));
    }

    // A build WITHOUT the tier must not fire the window, or Auto sends the shape to a
    // tier that cannot serve it and the walk falls through to an unmeasured arm.
    const auto absent = getrf_shape(12, 16384, /*cta_max_n=*/128, /*tiny_max=*/0);
    EXPECT_FALSE(F::preferred(kTiny, absent));
    EXPECT_EQ(hits(F{}, absent), 0);
    EXPECT_TRUE(is_vendor(resolve_getrf_route<float>(kAuto, absent, true)));

    // The tier ceiling still gates: n beyond tiny_max_n is refused even inside 8..32.
    const auto capped = getrf_shape(24, 16384, /*cta_max_n=*/128, /*tiny_max=*/16);
    EXPECT_FALSE(F::preferred(kTiny, capped));

    // The Blocked window is untouched and still disjoint from this one.
    const auto big = getrf_shape(256, 16384, 128, 32);
    EXPECT_TRUE(F::preferred(Route{Origin::Native, Algorithm::Blocked}, big));
    EXPECT_EQ(hits(F{}, big), 1);
}

TEST(RouteGetrf, VendorFreeFallbackHandsOverTheNativeRoute) {
    // THE TEST THAT FAILS IF A SPEED THRESHOLD EVER LANDS IN supports(): the
    // vendor-free walk tests supports() ALONE. order=40, batch=2 are inverse_tests' extents.
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
    // apart, so both halves are pinned here. BatchLAS's public getrf is square.
    auto wide = getrf_shape(/*order=*/64, /*batch=*/128, /*cta_max_n=*/256);
    wide.n = 1024;                    // m=64, n=1024, k=64
    EXPECT_FALSE(GetrfTable::supports(kGetrfCta, wide));
    EXPECT_FALSE(GetrfTable::supports(kGetrfBlocked, wide));

    auto tall = getrf_shape(/*order=*/1024, /*batch=*/128, /*cta_max_n=*/2048);
    tall.n = 32;                      // m=1024, n=32, k=1024
    EXPECT_FALSE(GetrfTable::supports(kGetrfCta, tall));
    EXPECT_FALSE(GetrfTable::supports(kGetrfBlocked, tall));

    // GUARD AGAINST VACUITY: the square shape at the same extents must be supported.
    const auto square = getrf_shape(64, 128, 256);
    ASSERT_TRUE(GetrfTable::supports(kGetrfCta, square));

    // geqrf's half (rectangular A stays native) is geqrf_candidates_tests'
    // CanRunEqualsLaunch on tall shapes.
}

TEST(RouteGetrf, CtaCapacityIsTheOrderAndBlockedInheritsOnlyThePresence) {
    // The CTA tile holds the whole n x n matrix PLUS the pivot-search scratch: a hard
    // launch limit, not a tuning knob.
    const auto fits = getrf_shape(/*order=*/128, /*batch=*/64, /*cta_max_n=*/128);
    ASSERT_TRUE(GetrfTable::supports(kGetrfCta, fits))
        << "guard: order 128 is exactly the capacity";

    const auto over = getrf_shape(/*order=*/129, /*batch=*/64, /*cta_max_n=*/128);
    EXPECT_FALSE(GetrfTable::supports(kGetrfCta, over));

    // The BLOCKED arm inherits the PRESENCE of the leaf but not its capacity.
    EXPECT_TRUE(GetrfTable::supports(kGetrfBlocked, over));

    // ...but only when it exists.
    auto no_blocked = over;
    no_blocked.blocked_available = false;
    EXPECT_FALSE(GetrfTable::supports(kGetrfBlocked, no_blocked));

    // AND THE BLOCKED ARM CARRIES NO LOWER BOUND: in supports(), "order <= the CTA
    // capacity so blocked should be false" makes a forced `blocked` measure the vendor.
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
    // function, so one missing capability must close BOTH arms.
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

    // ROCm packs int32 like CUDA, so the gate must NOT be an allow-list of one backend.
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

    // With no vendor at all there is no route, which is the honest answer here.
    EXPECT_FALSE(is_native(resolve_getrf_route<float>(kGetrfAuto, f, false)));
}

// THE MEASURED ORDER WINDOW, per type. The clause names Algorithm::Blocked, NOT
// "native": cta_max_n is passed large below so CTA is SUPPORTED at every order in
// the loop and a clause that forgot the algo test is caught.
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
        // cfloat since P4: >= 512 at any batch, and 256..511 ONLY at batch >= 256.
        // The register panel leaf is what makes 256 clear the gate at all, and the
        // batch term is a measured loss at batch 64..128, not caution.
        // evidence: docs/perf/lu.md#the-cfloat-window-moves-to-256
        for (int64_t order : {512, 513, 2048}) {
            const auto s = getrf_shape(order, batch, 4096);
            EXPECT_TRUE(GetrfTableCF::preferred(kGetrfBlocked, s)) << "cfloat order " << order;
            const Route r = resolve_getrf_route<std::complex<float>>(kGetrfAuto, s, true);
            EXPECT_TRUE(is_native(r) && r.algo == Algorithm::Blocked);
        }
        for (int64_t order : {256, 257, 511}) {
            const auto s = getrf_shape(order, batch, 4096);
            EXPECT_EQ(GetrfTableCF::preferred(kGetrfBlocked, s), batch >= 256)
                << "cfloat order " << order << " batch " << batch
                << ": the 256..511 band is batch-gated -- 0.92 at batch 128, 1.285 at 256";
            const Route r = resolve_getrf_route<std::complex<float>>(kGetrfAuto, s, true);
            EXPECT_EQ(is_native(r) && r.algo == Algorithm::Blocked, batch >= 256);
        }
        for (int64_t order : {1, 128, 192, 255}) {
            const auto s = getrf_shape(order, batch, 4096);
            EXPECT_FALSE(GetrfTableCF::preferred(kGetrfBlocked, s))
                << "cfloat order " << order << ": n=192 is 1.101/0.915/0.859 up the "
                   "batch ladder, below the gate at every saturating rung";
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

    // THE WINDOW IS NOT A CORRECTNESS GATE: preferred() must not repeat the capability test.
    const auto absent = getrf_shape(512, 256, /*cta_max_n=*/0);
    EXPECT_TRUE(GetrfTable::preferred(kGetrfBlocked, absent));
    EXPECT_FALSE(GetrfTable::supports(kGetrfBlocked, absent));
    EXPECT_TRUE(is_vendor(resolve_getrf_route<float>(kGetrfAuto, absent, true)));
}

TEST(RouteGetrf, BareOriginResolvesToASpecificAlgorithm) {
    // getrf has TWO native routes, so {Native, Auto} names neither.
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
    // Zero capacity / blocked_available == false is what a build without the kernels reports.
    const auto s = getrf_shape(/*order=*/64, /*batch=*/256, /*cta_max_n=*/0);
    EXPECT_FALSE(GetrfTable::supports(kGetrfCta, s));
    EXPECT_FALSE(GetrfTable::supports(kGetrfBlocked, s));
    EXPECT_TRUE(is_vendor(resolve_getrf_route<float>(kGetrfAuto, s, true)));
    EXPECT_TRUE(is_vendor(resolve_getrf_route<float>(kGetrfAuto, s, false)))
        << "vendor-free with nothing supported must say 'needs a vendor', not "
           "invent a native route";

    // AND FORCING MUST NOT ESCAPE IT: a forced route is gated on supports().
    EXPECT_TRUE(is_vendor(resolve_getrf_route<float>(kGetrfCta, s, true)));
    EXPECT_TRUE(is_vendor(resolve_getrf_route<float>(kGetrfBlocked, s, true)));
    EXPECT_TRUE(is_vendor(resolve_getrf_route<float>(kGetrfNativeBare, s, true)));
    EXPECT_TRUE(is_vendor(resolve_getrf_route<float>(kGetrfCta, s, false)));

    // Half a capability is still absent: the blocked leaf IS the CTA device function.
    auto half = getrf_shape(64, 256, /*cta_max_n=*/0);
    half.blocked_available = true;
    EXPECT_FALSE(GetrfTable::supports(kGetrfBlocked, half))
        << "the blocked driver's diagonal-panel leaf IS the CTA device function, "
           "so it inherits the presence gate";
}

TEST(RouteGetrf, NativeTierPreferredIsDeclaredAndPinsTheMeasuredTierChoice) {
    // native_tier_preferred() is the third predicate, consulted ONLY in the vendor-free
    // branch. DOUBLE alone prefers the blocked driver below its own CTA ceiling.
    // evidence: docs/perf/lu.md#native_tier_preferred
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

    // double: the ONE type where the hook overrides kGetrfOrder's CTA-first ladder.
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

    // IT IS NOT A CORRECTNESS GATE: both arms stay supports()-able wherever the window moves.
    EXPECT_TRUE((GetrfTableD::supports(kGetrfCta, small_d)));
    EXPECT_TRUE((GetrfTableD::supports(kGetrfBlocked, small_d)));
    EXPECT_EQ(resolve_getrf_route<double>(kGetrfCta, small_d,
                                          /*vendor_available=*/false).algo,
              Algorithm::CTA);

    // The hook is consulted only inside the `!vendor_available` branch.
    EXPECT_TRUE(is_vendor(resolve_getrf_route<double>(kGetrfAuto, small_d,
                                                      /*vendor_available=*/true)));

    // getrs declares it too: it has two native tiers, which the order array alone cannot follow.
    EXPECT_TRUE((declares_native_tier_preferred<GetrsTable, GetrsShape>))
        << "getrs has two native tiers; the order array alone cannot follow a "
           "crossover between them";

    // getri is single-arm and must NOT declare it.
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
        ScopedEnvVar e("BATCHLAS_GETRF_ROUTE", "cta");
        const auto p = parse_route_env(Op::getrf);
        ASSERT_TRUE(p.found) << "BATCHLAS_GETRF_ROUTE was not read at all";
        EXPECT_EQ(p.route, (Route{Origin::Native, Algorithm::CTA}))
            << "a bare algorithm implies Origin::Native";
        EXPECT_EQ(p.source.variable, "BATCHLAS_GETRF_ROUTE");
        EXPECT_FALSE(p.source.legacy);
    }
    {
        ScopedEnvVar e("BATCHLAS_GETRF_ROUTE", "native:blocked");
        const auto p = parse_route_env(Op::getrf);
        ASSERT_TRUE(p.found);
        EXPECT_EQ(p.route, (Route{Origin::Native, Algorithm::Blocked}));
    }
    {
        ScopedEnvVar e("BATCHLAS_GETRF_ROUTE", "vendor");
        const auto p = parse_route_env(Op::getrf);
        ASSERT_TRUE(p.found);
        EXPECT_EQ(p.route, (Route{Origin::Vendor, Algorithm::Auto}));
    }
    {
        ScopedEnvVar e("BATCHLAS_GETRF_ROUTE", "not-a-route");
        const auto p = parse_route_env(Op::getrf);
        EXPECT_FALSE(p.found);
        EXPECT_TRUE(p.unparsed) << "a typo must be reported, not silently Auto";
    }
}

// ---------------------------------------------------------------------------
// GETRS: the fused narrow-RHS CTA tier and the composition over the routed trsm.
// ---------------------------------------------------------------------------

TEST(RouteGetrs, VendorFreeFallbackHandsOverTheNativeRoute) {
    // The speed-threshold guard again, and here the temptation is concrete: the composed
    // arm is a measured loss at nrhs=1, which belongs in preferred() and not supports().
    const auto s = getrs_shape(/*order=*/32, /*nrhs=*/1, /*batch=*/1);

    EXPECT_TRUE(GetrsTable::supports(kGetrsBlocked, s))
        << "nrhs and batch are speed questions; neither may gate CORRECTNESS, even "
           "though nrhs=1 is measured 0.36x geomean";
    EXPECT_FALSE(GetrsTable::preferred(kGetrsBlocked, s))
        << "the COMPOSITION is never preferred at any width the fused tier serves; "
           "it is 0.36x geomean here and the window belongs to CTA alone";

    EXPECT_TRUE(is_native(resolve_getrs_route<float>(kGetrsAuto, s, false)));

    // With a vendor present this shape is native too: nrhs = 1 is inside the window.
    const Route with_vendor = resolve_getrs_route<float>(kGetrsAuto, s, true);
    EXPECT_TRUE(is_native(with_vendor));
    EXPECT_EQ(with_vendor.algo, Algorithm::CTA);

    // The width just outside clause A for a NON-float type must still take the vendor.
    const auto wide = getrs_shape(/*order=*/32, /*nrhs=*/4, /*batch=*/1);
    EXPECT_TRUE(is_vendor((resolve_route<Op::getrs, double>(kGetrsAuto, wide, true))))
        << "double at nrhs = 4 is OUTSIDE the measured window (its n=128 ladder dips "
           "to 0.940x mid-ladder); routing it native would ship a measured loss";
}

TEST(RouteGetrs, AllThreeTransposeModesAreSupportedAndTransAReachesTheShape) {
    // transA is a LIVE routing input and a genuine algorithm fork: NoTrans applies P
    // first and solves L then U, while Trans/ConjTrans solve U^T/U^H then L^T/L^H and
    // apply P^T LAST, on the output, in reverse.
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

// THE MEASURED WINDOW, pinned from BOTH sides: order >= 32 with nrhs <= 2 (every type,
// clause A) and nrhs <= 4 (float, clause B), plus clause C for the composition.
// evidence: docs/perf/lu.md#getrs-fused-window-evidence, #getrs-order-floor-evidence
TEST(RouteGetrs, PreferredIsTheMeasuredNrhsWindowAndNothingWider) {
    // ---- THE ORDER FLOOR, from both sides. Below it the work-group IS the cost.
    //      evidence: docs/perf/lu.md#getrs-order-floor-evidence
    for (int64_t order : {1, 4, 8, 16, 17, 24, 31}) {
        for (int64_t batch : {1, 128, 8192}) {
            for (int64_t nrhs : {int64_t(1), int64_t(2)}) {
                const auto s = getrs_shape(order, nrhs, batch);
                EXPECT_FALSE(GetrsTable::preferred(kGetrsCta, s))
                    << "float order " << order << " nrhs " << nrhs << " batch " << batch;
                EXPECT_FALSE((GetrsTableD::preferred(kGetrsCta, s)));
                EXPECT_FALSE((GetrsTableCF::preferred(kGetrsCta, s)));
                EXPECT_FALSE((GetrsTableCD::preferred(kGetrsCta, s)));
                EXPECT_TRUE(is_vendor(resolve_getrs_route<float>(kGetrsAuto, s, true)))
                    << "order " << order << " nrhs " << nrhs << " batch " << batch
                    << " is below the order floor and must take the vendor";
                // NOT a correctness gate: the fused arm stays selectable when forced,
                // and a vendor-free build must still reach a native route.
                EXPECT_TRUE(GetrsTable::supports(kGetrsCta, s));
                EXPECT_TRUE(is_native(resolve_getrs_route<float>(kGetrsAuto, s, false)));
            }
            // clause B's float-only width takes the SAME floor, off the same grid.
            // evidence: docs/perf/lu.md#getrs-order-floor-evidence
            EXPECT_FALSE(GetrsTable::preferred(kGetrsCta, getrs_shape(order, 4, batch)))
                << "clause B, float order " << order;
        }
    }
    // ...and the first order INSIDE it, so the floor is bracketed rather than asserted.
    for (int64_t batch : {1, 128, 8192}) {
        for (int64_t nrhs : {int64_t(1), int64_t(2)}) {
            const auto s = getrs_shape(32, nrhs, batch);
            EXPECT_TRUE(GetrsTable::preferred(kGetrsCta, s))
                << "order 32 is the first order where all four types clear the flip "
                   "gate AND stay clear above it (float 2.29, cfloat 1.40, double "
                   "3.77, cdouble 2.13 at nrhs 1)";
            EXPECT_TRUE((GetrsTableD::preferred(kGetrsCta, s)));
            EXPECT_TRUE((GetrsTableCF::preferred(kGetrsCta, s)));
            EXPECT_TRUE((GetrsTableCD::preferred(kGetrsCta, s)));
            const Route r = resolve_getrs_route<float>(kGetrsAuto, s, true);
            EXPECT_TRUE(is_native(r) && r.algo == Algorithm::CTA)
                << "order 32 nrhs " << nrhs << " batch " << batch;
        }
        EXPECT_TRUE(GetrsTable::preferred(kGetrsCta, getrs_shape(32, 4, batch)))
            << "clause B at the floor: float nrhs=4 clears only from 32 (1.30)";
        EXPECT_FALSE((GetrsTableD::preferred(kGetrsCta, getrs_shape(32, 4, batch))))
            << "and the floor must not widen clause B beyond float";
    }
    // The floor is on order(), not on nrhs() or batch, PROVED BY CONSTRUCTION: at a
    // fixed nrhs and batch, order alone flips the answer at 32.
    EXPECT_FALSE(GetrsTable::preferred(kGetrsCta, getrs_shape(31, 1, 8192)));
    EXPECT_TRUE (GetrsTable::preferred(kGetrsCta, getrs_shape(32, 1, 8192)));

    // ---- clause A: every type, every order AT OR ABOVE THE FLOOR, nrhs <= 2 --
    for (int64_t order : {32, 128, 2048}) {
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

    // ---- clause B is FLOAT ONLY --------------------------------------------
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
    // AXIS: GetrsShape::nrhs(), which is B.cols(), NOT order(). The boundary is per
    // type, and clause C carries a batch floor that clauses A and B do not.
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

            // double: IN at 128, OUT at 127 AND at 64.
            const auto d_in  = getrs_shape(order, /*nrhs=*/128, batch);
            const auto d_127 = getrs_shape(order, /*nrhs=*/127, batch);
            const auto d_64  = getrs_shape(order, /*nrhs=*/64,  batch);
            EXPECT_TRUE(GetrsTableD::preferred(kGetrsBlocked, d_in));
            EXPECT_FALSE(GetrsTableD::preferred(kGetrsBlocked, d_127));
            EXPECT_FALSE(GetrsTableD::preferred(kGetrsBlocked, d_64));
            EXPECT_TRUE(is_native(resolve_getrs_route<double>(kGetrsAuto, d_in, true)));
            EXPECT_TRUE(is_vendor(resolve_getrs_route<double>(kGetrsAuto, d_64, true)));

            // cfloat and cdouble: NOTHING, at any width.
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

    // THE CLAUSE IS ON nrhs AND NOT ON order, PROVED BY CONSTRUCTION.
    {
        bool in_all = true, out_all = false;
        for (int64_t order : {1, 2, 8, 63, 64, 65, 1000, 100000}) {
            in_all  &= GetrsTable::preferred(kGetrsBlocked, getrs_shape(order, 64, 512));
            out_all |= GetrsTable::preferred(kGetrsBlocked, getrs_shape(order, 63, 512));
        }
        EXPECT_TRUE(in_all)  << "clause C must admit nrhs=64 at EVERY order";
        // ...and the batch floor, from both sides.
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

    // THE WINDOW IS NOT A CORRECTNESS GATE: at nrhs = 8 the fused tier is supported.
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
    // (c) BOTH absent.
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
    // driver. AND THE ANSWER CHANGED WHEN THE FUSED TIER LANDED -- a bare `native` pin
    // used to mean the composition and now means the fused kernel.
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
        ScopedEnvVar e("BATCHLAS_GETRS_ROUTE", "blocked");
        const auto p = parse_route_env(Op::getrs);
        ASSERT_TRUE(p.found) << "BATCHLAS_GETRS_ROUTE was not read at all";
        EXPECT_EQ(p.route, (Route{Origin::Native, Algorithm::Blocked}));
        EXPECT_EQ(p.source.variable, "BATCHLAS_GETRS_ROUTE");
        EXPECT_FALSE(p.source.legacy);
    }
    {
        ScopedEnvVar e("BATCHLAS_GETRS_ROUTE", "vendor");
        const auto p = parse_route_env(Op::getrs);
        ASSERT_TRUE(p.found);
        EXPECT_EQ(p.route, (Route{Origin::Vendor, Algorithm::Auto}));
    }
    {
        ScopedEnvVar e("BATCHLAS_GETRS_ROUTE", "not-a-route");
        const auto p = parse_route_env(Op::getrs);
        EXPECT_FALSE(p.found);
        EXPECT_TRUE(p.unparsed) << "a typo must be reported, not silently Auto";
    }
}

// ---------------------------------------------------------------------------
// GETRI. One native arm: a composition over the routed trsm.
// ---------------------------------------------------------------------------

TEST(RouteGetri, VendorFreeFallbackHandsOverTheNativeRoute) {
    // n=40, batch=2 are inverse_tests' actual extents.
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
        << "INHERITED from trsm's can_run (src/ops/trsm/trsm.cc: native needs is_gpu)";

    auto het = ok;  het.heterogeneous_batch = true;
    EXPECT_FALSE(GetriTable::supports(kGetriBlocked, het))
        << "INHERITED from trsm's can_run (no native heterogeneous batch), and getri's own besides -- the "
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
// `k`, and there is NO batch term.
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

    // ...AND INSIDE THE WINDOW: preferred() must still say yes while supports() says no.
    const auto in_window_absent = getri_shape(512, 256, /*blocked_available=*/false);
    EXPECT_TRUE(GetriTable::preferred(kGetriBlocked, in_window_absent));
    EXPECT_FALSE(GetriTable::supports(kGetriBlocked, in_window_absent));
    EXPECT_TRUE(is_vendor(resolve_getri_route<float>(kGetriAuto, in_window_absent, true)));
    EXPECT_TRUE(is_vendor(resolve_getri_route<float>(kGetriAuto, in_window_absent, false)))
        << "vendor-free with no driver must say 'needs a vendor', not invent a route";

    // A NETLIB queue is a CORRECTNESS refusal (the pivot format disagrees), and the
    // window must not override it.
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
        ScopedEnvVar e("BATCHLAS_GETRI_ROUTE", "blocked");
        const auto p = parse_route_env(Op::getri);
        ASSERT_TRUE(p.found) << "BATCHLAS_GETRI_ROUTE was not read at all";
        EXPECT_EQ(p.route, (Route{Origin::Native, Algorithm::Blocked}));
        EXPECT_EQ(p.source.variable, "BATCHLAS_GETRI_ROUTE");
        EXPECT_FALSE(p.source.legacy);
    }
    {
        ScopedEnvVar e("BATCHLAS_GETRI_ROUTE", "vendor");
        const auto p = parse_route_env(Op::getri);
        ASSERT_TRUE(p.found);
        EXPECT_EQ(p.route, (Route{Origin::Vendor, Algorithm::Auto}));
    }
    {
        ScopedEnvVar e("BATCHLAS_GETRI_ROUTE", "not-a-route");
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
    // The physical pivot format is BACKEND-DEPENDENT: the vendors store PACKED 1-BASED
    // INT32 in the first half of the caller's int64 buffer, netlib genuine int64. A
    // native getrf must agree with WHATEVER SERVES getri on the same call, and no shape
    // field can express "the op downstream of me resolved differently".
    ClearRouteEnv clear_f(Op::getrf);
    ClearRouteEnv clear_s(Op::getrs);
    ClearRouteEnv clear_i(Op::getri);

    ScopedEnvVar ef("BATCHLAS_GETRF_ROUTE", "cta");
    ScopedEnvVar ei("BATCHLAS_GETRI_ROUTE", "vendor");

    EXPECT_EQ(parse_route_env(Op::getrf).route, (Route{Origin::Native, Algorithm::CTA}));
    EXPECT_EQ(parse_route_env(Op::getri).route, (Route{Origin::Vendor, Algorithm::Auto}));
    EXPECT_FALSE(parse_route_env(Op::getrs).found)
        << "and the third is untouched -- the three do not share a variable";

    // Today both resolve to the vendor, which is why this asserts on the PARSED routes.
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
// spmm. The Direct arm has NO is_gpu clause, so a native_cpu queue can take the
// route -- build-novendor's Backend::NETLIB rows depend on it.
//
// TWO CAPABILITY FLAGS, NOT ONE, and they are not interchangeable: transA ==
// NoTrans is served by the gather body, transA != NoTrans by the scale+scatter
// PAIR. They are separate kernels, so a build can have one and not the other, and
// supports() consults exactly the flag for the body that would actually run.
//
// spmm_shape() sets format, gather_available AND scatter_available: leave any at its
// default and supports() is false on every shape, so every assertion here holds
// vacuously. RouteSpmm.HelperIsArmed is what checks that. evidence: docs/perf/spmm.md
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

// THE HELPER IS ARMED. RUN THIS FIRST: if it fails, every other spmm assertion in this
// file is vacuous. Four gates, taken away ONE AT A TIME, each required to MOVE the answer.
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

// Adding `if (!s.is_gpu) return false;` to the Direct arm turns this red. All nine
// transpose pairs are asserted because all three bodies are plain loops -- no local
// memory, no group collective, no required sub-group size.
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

// ALL NINE (transA, transB) COMBINATIONS ARE SERVED, which keeps the transB == Trans
// layout lever available instead of materialising a transposed copy.
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

// THE TWO FLAGS ARE INDEPENDENT AND SERVE DISJOINT HALVES OF THE transA AXIS. A table
// that ORed them would pass a shape to a kernel this build does not contain. transB is
// swept inside both halves because it must NOT influence the choice.
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

// ONLY CSR HAS BODIES, and this is a correctness gate: a Dense or COO view reaching a
// CSR kernel reads row offsets that are not there.
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
            // Vendor-free there is nothing to fall back to, so the resolver returns
            // the vendor as its honest "this needs one" signal.
            EXPECT_TRUE(is_vendor(resolve_spmm_route<float>(kSpmmAuto, s, false)));
        }
    }
    // The CSR control, so this case cannot pass by refusing everything.
    EXPECT_TRUE(SpmmTable::supports(
        kSpmmDirect, spmm_shape(1024, 1024, 12, 64, Transpose::NoTrans,
                                Transpose::NoTrans, MatrixFormat::CSR)));
}

// A HETEROGENEOUS BATCH IS A CORRECTNESS GATE: one launch covers the batch with a
// single (ld, stride) tuple per DENSE operand. Per-item variation on the sparse side
// is expressible only as nnz(b), which every body already handles.
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
    // ...and under a transposed transA, where m and k swap roles, both still stand.
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

// Algorithm::Auto IS NOT ITSELF A NATIVE SPMM ROUTE: a bare `native` names no body, so
// supports() must say false. It still RESOLVES, to the one native route there is.
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
// and NOTHING else -- no batch, extent, is_gpu or nnz term.
// evidence: docs/perf/spmm.md#the-evidence-for-each-boundary
// ===========================================================================

namespace {

// The three sibling tables, named once. float is `SpmmTable` above.
using SpmmTableD  = RouteTable<Op::spmm, double>;
using SpmmTableCF = RouteTable<Op::spmm, std::complex<float>>;
using SpmmTableCD = RouteTable<Op::spmm, std::complex<double>>;

constexpr Route kSpmmBlocked{Origin::Native, Algorithm::Blocked};

} // namespace

// The clause carries no extent or batch term, so a future one must turn this red.
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

    // A RECTANGULAR SPREAD TOO: m == k above would hide a clause read on the other extent.
    for (int64_t m : {512, 4096}) {
        for (int64_t k : {64, 8192}) {
            const auto s = spmm_shape(m, k, 25, 128);
            EXPECT_TRUE(SpmmTable::preferred(kSpmmDirect, s)) << "m " << m << " k " << k;
            EXPECT_TRUE(SpmmTableD::preferred(kSpmmDirect, s));
            EXPECT_TRUE(SpmmTableCD::preferred(kSpmmDirect, s));
        }
    }
}

// THE BATCH AXIS HAS NO FLOOR, AND ITS ABSENCE IS A MEASURED DECISION.
// evidence: docs/perf/spmm.md#the-batch-axis-has-no-floor
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

// THE TRANSPOSED REFUSAL IS MEASURED, NOT AN OMISSION. Both sides of the boundary are
// asserted on the SAME shape. evidence: docs/perf/spmm.md#the-gather-window
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
                    // ...and vendor-FREE it must still reach the native bodies.
                    const Route free_route =
                        resolve_spmm_route<float>(kSpmmAuto, s, false);
                    EXPECT_TRUE(is_native(free_route));
                    EXPECT_EQ(free_route.algo, Algorithm::Direct);
                }
            }
        }
    }
}

// THE ONE TYPE-CONDITIONAL BOUNDARY, ASSERTED FROM ALL FOUR SIDES: the exclusion is
// (type AND transB) TOGETHER -- drop either half of the conjunction and one of these
// goes red. It is deliberately NOT narrowed by nrhs, because the threshold rides on
// the banded column pattern, which SpmmShape has no field for.
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

    // And the resolved decision, both ways, with a vendor present.
    const auto cf_tb = spmm_shape(2048, 2048, 25, 512, Transpose::NoTrans,
                                  Transpose::Trans);
    EXPECT_TRUE(is_vendor(
        resolve_spmm_route<std::complex<float>>(kSpmmAuto, cf_tb, true)));
    EXPECT_TRUE(is_native(
        resolve_spmm_route<std::complex<double>>(kSpmmAuto, cf_tb, true)));
    EXPECT_TRUE(is_native(resolve_spmm_route<float>(kSpmmAuto, cf_tb, true)));
    // Vendor-FREE, even the refused complex<float> cell takes the native body.
    EXPECT_TRUE(is_native(
        resolve_spmm_route<std::complex<float>>(kSpmmAuto, cf_tb, false)));
}

// THE CLAUSE SPEAKS ONLY FOR {Native, Direct} AND ONLY FOR CSR. preferred() is asked
// about EVERY entry in kSpmmOrder, so a clause answering true for {Vendor, Auto} would
// pin the vendor as "preferred" and make the native route unreachable.
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
// acquired one, a native_cpu queue would silently go back to netlib -- which refuses
// every transpose -- in a vendor-present build.
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

// UN-PREFERRED IS NOT UNSUPPORTED, AND A PIN MUST STILL REACH THE SCATTER. In
// supports() the refusal would make every transposed measurement cuSPARSE instead.
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

// AN AUTO SPMM IS NATIVE WHEREVER THE CLAUSE FIRES, THE VENDOR WHEREVER IT DOES NOT,
// AND NATIVE EVERYWHERE ONCE THE VENDOR IS GONE.
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
// BUILD: BATCHLAS_SPMM_ROUTE=cta parses fine, supports() rejects it because there is no
// CTA body, and the run then measures cuSPARSE; a MISSPELLED value behaves the same way
// because `unparsed` is discarded.
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

    // OUTSIDE the preferred window the pin silently becomes cuSPARSE instead.
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

    // INSIDE it, the same unserviceable pin lands on the NATIVE gather -- still silently.
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

// out_rows() AND red_rows() SWAP WITH transA. The shipped clause reads NEITHER, so this
// is here to make the mapping wrong-proof for whoever adds the first extent clause.
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

// THE ORDER ARRAY HAS EXACTLY TWO ENTRIES, ASSERTED RATHER THAN ASSUMED -- and this is
// the ONLY case that reversing the array turns red. A third entry would mean a second
// native tier, which would also need native_tier_preferred() to arbitrate it.
TEST(RouteSpmm, OrderIsExactlyTwoEntries) {
    ASSERT_EQ(SpmmTable::order_end() - SpmmTable::order_begin(), 2);
    EXPECT_EQ(SpmmTable::order_begin()[0].origin, Origin::Native);
    EXPECT_EQ(SpmmTable::order_begin()[0].algo, Algorithm::Direct);
    EXPECT_EQ(SpmmTable::order_begin()[1].origin, Origin::Vendor);
    EXPECT_EQ(SpmmTable::order_begin()[1].algo, Algorithm::Auto);
    EXPECT_TRUE(is_vendor(SpmmTable::order_begin()[1]));
}

// The env variable exists without a line of route_env.hh changing: parse_route_env
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
        ScopedEnvVar e("BATCHLAS_SPMM_ROUTE", "direct");
        const auto p = parse_route_env(Op::spmm);
        ASSERT_TRUE(p.found) << "BATCHLAS_SPMM_ROUTE was not read at all";
        EXPECT_EQ(p.route, (Route{Origin::Native, Algorithm::Direct}));
        EXPECT_EQ(p.source.variable, "BATCHLAS_SPMM_ROUTE");
        EXPECT_FALSE(p.source.legacy);
    }
    {
        ScopedEnvVar e("BATCHLAS_SPMM_ROUTE", "native:direct");
        const auto p = parse_route_env(Op::spmm);
        ASSERT_TRUE(p.found);
        EXPECT_EQ(p.route, (Route{Origin::Native, Algorithm::Direct}));
    }
    {
        // A bare origin leaves the algorithm free; the resolver picks the body.
        ScopedEnvVar e("BATCHLAS_SPMM_ROUTE", "native");
        const auto p = parse_route_env(Op::spmm);
        ASSERT_TRUE(p.found);
        EXPECT_EQ(p.route, (Route{Origin::Native, Algorithm::Auto}));
        EXPECT_EQ(resolve_spmm_route<float>(p.route,
                                            spmm_shape(4096, 4096, 25, 64), true).algo,
                  Algorithm::Direct);
    }
    {
        ScopedEnvVar e("BATCHLAS_SPMM_ROUTE", "vendor");
        const auto p = parse_route_env(Op::spmm);
        ASSERT_TRUE(p.found);
        EXPECT_EQ(p.route, (Route{Origin::Vendor, Algorithm::Auto}));
    }
    {
        // THE TYPO PATH: parse_route_env reports it, and every adapter in the tree then
        // DISCARDS `unparsed` and uses the unset default, so the run goes to the vendor.
        ScopedEnvVar e("BATCHLAS_SPMM_ROUTE", "not-a-route");
        const auto p = parse_route_env(Op::spmm);
        EXPECT_FALSE(p.found);
        EXPECT_TRUE(p.unparsed) << "a typo must be reported, not silently Auto";
        EXPECT_EQ(legacy_unset_default(Op::spmm), (Route{Origin::Auto, Algorithm::Auto}))
            << "and the value spmm_route.hh substitutes for it is plain Auto";
    }
}

// The vendor-free getrf walk for cfloat: Tiny wherever it fits (1.2-1.5x over the better of
// CTA and Blocked at 25..32), CTA above -- each the measured best native tier.
// evidence: docs/perf/lu.md#the-column-bucket
// ARMED BREAK (R9): restore tiny_native's `n <= 8` clause in place of `tiny_fits`.
// EXPECTED: RED at 25, 28 and 32 (the walk leaves Tiny).
TEST(RouteGetrf, CfloatNativeWalkTakesTinyWhereverItFits) {
    constexpr Route kTiny{Origin::Native, Algorithm::Tiny};
    constexpr Route kCta{Origin::Native, Algorithm::CTA};
    constexpr Route kAuto{Origin::Auto, Algorithm::Auto};
    using CF = std::complex<float>;
    for (int64_t n : {17, 24, 25, 28, 32}) {
        const auto sh = getrf_shape(n, 32768, /*cta_max_n=*/128, /*tiny_max=*/32);
        EXPECT_EQ(resolve_getrf_route<CF>(kAuto, sh, false), kTiny) << "cfloat n=" << n;
    }
    for (int64_t n : {25, 28, 32}) {
        // Vendor-present Auto is unchanged: native against native only.
        const auto sh = getrf_shape(n, 32768, 128, 32);
        EXPECT_TRUE(is_vendor(resolve_getrf_route<CF>(kAuto, sh, true))) << "cfloat n=" << n;
    }
    const auto above = getrf_shape(40, 32768, 128, 32);
    EXPECT_EQ(resolve_getrf_route<CF>(kAuto, above, false), kCta) << "cfloat n=40";
}
