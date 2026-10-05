// The dispatch vocabulary: Route parsing, the legacy environment spellings, and
// the per-op RouteTable supports()/preferred() windows.
// The legacy spellings appear in committed benchmark scripts and in the provenance of
// recorded results, so their mapping is pinned here. evidence: docs/perf/dispatch.md

#include <gtest/gtest.h>

#include <batchlas/blas/dispatch/route.hh>
#include <batchlas/blas/dispatch/route_env.hh>
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

// The LU family (getrf, getrs, getri) moved to flat selection: its route tests are ported to
// tests/get{rf,rs,ri}_candidates_tests.cc.

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

constexpr Route kVendorAuto{Origin::Vendor, Algorithm::Auto};

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

