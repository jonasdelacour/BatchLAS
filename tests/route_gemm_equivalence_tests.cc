// Does the Route-based GEMM split choose the same thing the current code does?
//
// This is the acceptance test the WP0 spec demands for its Risk 2: diff the
// CHOSEN ROUTE, not the timing, across the whole input space. Any differing
// case is a bug, not a tuning question.
//
// `gemm_use_sycl_custom` is replicated below rather than called, deliberately.
// It lives in src/backends/gemm_variant.hh, takes MatrixViews and a live Queue,
// and reads getenv internally -- none of which a pure decision test should need.
// The replica is transcribed from that source; `ReplicaIsFaithful` below pins
// the transcription itself against the handful of behaviours that are easy to
// get wrong, so a drift in the replica cannot silently make the diff pass.

#include <gtest/gtest.h>

#include <batchlas/blas/dispatch/route.hh>
#include <batchlas/blas/dispatch/route_env.hh>
#include <batchlas/blas/dispatch/route_gemm.hh>

#include <complex>
#include <string>
#include <vector>

using namespace batchlas;
using namespace batchlas::dispatch;

namespace {

// ---------------------------------------------------------------------------
// Faithful replica of the CURRENT decision, from src/backends/gemm_variant.hh.
// ---------------------------------------------------------------------------

enum class LegacyRequest { Vendor, Sycl, Native, CuBLASDx, Auto };

// gemm_variant_request(): note the default when the variable is UNSET is
// Vendor, not Auto. That asymmetry against the level-3 ops is load-bearing.
LegacyRequest legacy_request_from(const char* raw) {
    if (!raw) return LegacyRequest::Vendor;
    std::string v(raw);
    for (char& c : v) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (v == "sycl" || v == "custom") return LegacyRequest::Sycl;
    if (v == "native" || v == "cuda-native" || v == "direct-cuda") return LegacyRequest::Native;
    if (v == "cublasdx" || v == "dx") return LegacyRequest::CuBLASDx;
    if (v == "auto") return LegacyRequest::Auto;
    return LegacyRequest::Vendor;
}

// gemm_custom_problem_supported()
bool legacy_problem_supported(const OpShape& s) {
    if (s.precision != ComputePrecision::Default) return false;
    if (s.heterogeneous_batch) return false;
    return s.m > 0 && s.n > 0 && s.k > 0;
}

// gemm_use_sycl_custom(): true means "run the native SYCL kernel".
template <typename T>
bool legacy_use_sycl_custom(LegacyRequest request, const OpShape& s) {
    if (request == LegacyRequest::Vendor || request == LegacyRequest::Native ||
        request == LegacyRequest::CuBLASDx) {
        return false;
    }
    if (!legacy_problem_supported(s)) return false;
    if (request == LegacyRequest::Sycl) return true;

    if (!s.is_gpu) return false;

    if constexpr (is_std_complex_v<T>) {
        return false;
    } else {
        const int64_t max_dim = s.max_dim();
        if (s.m != s.n || s.n != s.k || s.batch < 64) return false;

        if constexpr (std::is_same_v<T, float>) {
            if (s.transA != Transpose::NoTrans || s.transB != Transpose::NoTrans) {
                if (s.transA == Transpose::ConjTrans || s.transB == Transpose::ConjTrans) {
                    return false;
                }
                return s.batch >= 128 && max_dim >= 128 && max_dim <= 512;
            }
            if (max_dim <= 32) return true;
            return max_dim >= 128 && max_dim <= 512;
        } else if constexpr (std::is_same_v<T, double>) {
            return max_dim <= 512;
        } else {
            return false;
        }
    }
}

// ---------------------------------------------------------------------------
// The new decision, expressed as the same boolean so the two are comparable.
// ---------------------------------------------------------------------------
template <typename T>
bool new_uses_native(const char* env_value, const OpShape& s) {
    Route forced = legacy_unset_default(Op::gemm);   // Auto since WP2 E6
    if (env_value) {
        // The legacy parser: these values arrive through
        // BATCHLAS_GEMM_VARIANT, where "native" means the raw CUDA path rather
        // than a BatchLAS kernel.
        if (const auto parsed = parse_legacy_route_value(Op::gemm, env_value)) {
            forced = *parsed;
        } else {
            forced = legacy_unset_default(Op::gemm);
        }
    }
    const Route chosen = resolve_gemm_route<T>(forced, s);
    return is_native(chosen);
}

// The env spellings that appear in benchmark scripts and recorded provenance,
// plus unset.
const std::vector<const char*> kEnvValues = {
    nullptr, "vendor", "sycl", "custom", "auto", "cublasdx", "dx",
    "native", "cuda-native", "direct-cuda",
};

std::vector<OpShape> shape_grid(ScalarKind scalar) {
    std::vector<OpShape> out;
    const int64_t dims[] = {8, 16, 32, 33, 64, 127, 128, 256, 512, 513, 1024, 2048};
    const int64_t batches[] = {1, 8, 63, 64, 127, 128, 512, 2048};
    const Transpose trans[] = {Transpose::NoTrans, Transpose::Trans, Transpose::ConjTrans};

    for (int64_t d : dims) {
        for (int64_t b : batches) {
            for (Transpose ta : trans) {
                for (Transpose tb : trans) {
                    for (bool gpu : {true, false}) {
                        OpShape s;
                        s.op = Op::gemm;
                        s.scalar = scalar;
                        s.m = s.n = s.k = d;
                        s.batch = b;
                        s.transA = ta;
                        s.transB = tb;
                        s.is_gpu = gpu;
                        out.push_back(s);
                    }
                }
            }
        }
    }
    // Non-square, heterogeneous and non-default-precision cells, which is where
    // a correctness gate and a speed window are most easily confused.
    for (int64_t d : dims) {
        OpShape s;
        s.op = Op::gemm; s.scalar = scalar;
        s.m = d; s.n = d * 2; s.k = d / 2 ? d / 2 : 1;
        s.batch = 256; s.is_gpu = true;
        out.push_back(s);

        OpShape h = s; h.m = h.n = h.k = d; h.heterogeneous_batch = true;
        out.push_back(h);

        OpShape p = s; p.m = p.n = p.k = d; p.precision = ComputePrecision::F32;
        out.push_back(p);

        OpShape z = s; z.m = z.n = z.k = 0;
        out.push_back(z);
    }
    return out;
}

// THE INTENDED DIVERGENCES FROM THE LEGACY DECISION.
//
// There are two. They are classified and COUNTED SEPARATELY below, so that
// neither can quietly stop being reached: a single boolean "is this expected?"
// would let one divergence's cells disappear from the grid while the other
// kept the count non-zero, and the test would still pass.
//
// (1) WP2 C2 -- heterogeneous batch, a WIDENING.
// (2) WP2 E4 -- float's measured window, a NARROWING.
// (3) WP2 E6 -- the UNSET DEFAULT flipped from Vendor to Auto.
//
// (3) is different in kind from the other two: it is not a change to which
// shapes are preferred, it is a change to what happens when nobody asked. It
// therefore only ever fires for env == nullptr, and excusing it wholesale would
// blind the test at that one env value -- which is the value ordinary calls
// use, so that would be the worst place to lose coverage. Instead the exception
// is narrow (unset only, and only in the vendor -> native direction) and
// `UnsetNowMeansAuto` below asserts POSITIVELY that unset and "auto" now agree
// on every shape in the grid. An excuse plus a positive assertion of what
// replaced it beats an excuse alone.
//
// The legacy predicate rejected a heterogeneous batch as UNSUPPORTED, so a
// forced native request on one fell through to the vendor. That was a
// correctness statement and it is no longer true: the facade now walks the
// batch (src/backends/gemm_heterogeneous.hh) and each member is homogeneous by
// construction, so a native route can serve it. supports() says so.
//
// This is deliberately an EXCEPTION LIST rather than a change to the replica.
// The replica's job is to model the legacy decision faithfully; editing it to
// match the new behaviour would make the two agree by construction and the test
// would stop being able to detect anything. Every future intended divergence
// should be added here, with its reason, and the count below asserted -- an
// exception list that grows silently is just a disabled test.
//
// Note this cell does NOT change the production route on a vendor-present box:
// backend::gemm_vendor (src/backends/cublas.cc) tests
// gemm_has_heterogeneous_batch BEFORE consulting gemm_use_sycl_custom, so the
// heterogeneous loop is entered either way. The divergence is visible to this
// test because it calls the predicate directly.
enum class Divergence { None, HeterogeneousWidened, FloatWindowNarrowed, DefaultFlipped };

// (2) WP2 E4. The legacy predicate preferred native for float on two windows
// that measurement says it loses: every transposed cell (0.34-0.55x of cuBLAS
// across TN/NT/TT at n=128..512) and NN at 128 <= max_dim <= 512 (0.40-0.98x).
// preferred() no longer claims either. This costs a vendor-free build nothing --
// resolve_route still falls back to any supported native route when there is no
// vendor -- so the divergence is a pure vendor-present improvement.
// Numbers and conditions in experiments/wp2_e4/.
template <typename T>
Divergence classify_divergence(const char* env, const OpShape& s,
                               bool old_native, bool new_native) {
    const bool forced_native = env != nullptr &&
        (std::string_view(env) == "sycl" || std::string_view(env) == "custom");
    if (s.heterogeneous_batch && forced_native && !old_native && new_native) {
        return Divergence::HeterogeneousWidened;
    }
    // (3) WP2 E6. Only for an UNSET environment, and only in the direction the
    // flip goes: the legacy default was a forced Vendor, so it never chose
    // native; Auto now defers to preferred(). Anything else at env == nullptr
    // is still compared.
    if (env == nullptr && !old_native && new_native) {
        return Divergence::DefaultFlipped;
    }
    if constexpr (std::is_same_v<T, float>) {
        if (old_native && !new_native) {
            const bool transposed =
                s.transA != Transpose::NoTrans || s.transB != Transpose::NoTrans;
            const int64_t max_dim = s.max_dim();
            if (transposed || (max_dim >= 128 && max_dim <= 512)) {
                return Divergence::FloatWindowNarrowed;
            }
        }
    }
    return Divergence::None;
}

template <typename T>
void expect_equivalent(ScalarKind kind, const char* type_name) {
    size_t compared = 0, native_cases = 0, het_widened = 0, float_narrowed = 0,
           default_flipped = 0;
    for (const char* env : kEnvValues) {
        for (const OpShape& s : shape_grid(kind)) {
            const bool old_native = legacy_use_sycl_custom<T>(legacy_request_from(env), s);
            const bool new_native = new_uses_native<T>(env, s);
            ++compared;
            if (old_native) ++native_cases;

            const Divergence d = classify_divergence<T>(env, s, old_native, new_native);
            if (d == Divergence::HeterogeneousWidened) { ++het_widened; continue; }
            if (d == Divergence::FloatWindowNarrowed)  { ++float_narrowed; continue; }
            if (d == Divergence::DefaultFlipped)       { ++default_flipped; continue; }

            ASSERT_EQ(old_native, new_native)
                << "route diverged for " << type_name
                << "  env=" << (env ? env : "<unset>")
                << "  " << s.describe()
                << "  transA=" << static_cast<int>(s.transA)
                << " transB=" << static_cast<int>(s.transB)
                << "  gpu=" << s.is_gpu
                << "  het=" << s.heterogeneous_batch
                << "  prec=" << static_cast<int>(s.precision);
        }
    }
    EXPECT_GT(compared, 1000u) << "grid too small to be meaningful";
    // Guards against the degenerate pass where the new code never picks native
    // and the old code never did either because the grid missed the window.
    EXPECT_GT(native_cases, 0u) << "grid never exercised the native route for " << type_name;

    // The exception list is ASSERTED, not merely allowed. An exception list
    // that can grow without anyone noticing is a disabled test, and a count of
    // zero would mean the grid stopped reaching the heterogeneous cells and the
    // exception silently stopped guarding anything.
    EXPECT_GT(het_widened, 0u)
        << "grid no longer reaches the intended heterogeneous divergence for "
        << type_name << " -- the exception is now vacuous";

    // Counted separately from the heterogeneous one on purpose: a single total
    // would let this one vanish from the grid while the other kept it non-zero.
    if constexpr (std::is_same_v<T, float>) {
        EXPECT_GT(float_narrowed, 0u)
            << "grid no longer reaches the WP2 E4 float narrowing -- the "
               "exception is now vacuous";
    } else {
        EXPECT_EQ(float_narrowed, 0u)
            << "the E4 narrowing is float-only but fired for " << type_name;
    }

    // The flip only produces divergences for types preferred() actually accepts.
    // complex is refused outright, so zero there is the correct answer and a
    // NON-zero one would mean complex had started routing native by default --
    // exactly the regression WP2_GEMM_SPEC warns about.
    if constexpr (is_std_complex_v<T>) {
        EXPECT_EQ(default_flipped, 0u)
            << "complex must not route native by default: preferred() refuses it, "
               "and the register ladder for complex needs min_dim >= 256 and an "
               "aligned NN shape (see WP2_GEMM_SPEC.md)";
    } else {
        EXPECT_GT(default_flipped, 0u)
            << "grid no longer reaches the WP2 E6 default flip for " << type_name
            << " -- the exception is now vacuous";
    }
}

// What the E6 exception above excuses, asserted positively: after the flip an
// UNSET environment must decide exactly as an explicit "auto" does, on every
// shape in the grid. Without this the exception would merely hide the change.
template <typename T>
void expect_unset_equals_auto(ScalarKind kind, const char* type_name) {
    size_t compared = 0, native_cases = 0;
    for (const OpShape& s : shape_grid(kind)) {
        const bool unset = new_uses_native<T>(nullptr, s);
        const bool automatic = new_uses_native<T>("auto", s);
        ++compared;
        if (unset) ++native_cases;
        ASSERT_EQ(unset, automatic)
            << "unset and \"auto\" disagree for " << type_name << "  " << s.describe()
            << "  transA=" << static_cast<int>(s.transA)
            << " transB=" << static_cast<int>(s.transB)
            << "  gpu=" << s.is_gpu;
    }
    EXPECT_GT(compared, 100u) << "grid too small to be meaningful";
    // Guard against the degenerate agreement where neither ever picks native.
    if constexpr (!is_std_complex_v<T>) {
        EXPECT_GT(native_cases, 0u)
            << "unset never chose native for " << type_name
            << " -- the flip would be a no-op and this test vacuous";
    }
}

} // namespace

TEST(RouteGemmEquivalence, Float)         { expect_equivalent<float>(ScalarKind::F32, "float"); }
TEST(RouteGemmEquivalence, Double)        { expect_equivalent<double>(ScalarKind::F64, "double"); }
TEST(RouteGemmEquivalence, ComplexFloat)  { expect_equivalent<std::complex<float>>(ScalarKind::C32, "complex<float>"); }
TEST(RouteGemmEquivalence, ComplexDouble) { expect_equivalent<std::complex<double>>(ScalarKind::C64, "complex<double>"); }

TEST(RouteGemmEquivalence, UnsetNowMeansAutoFloat)         { expect_unset_equals_auto<float>(ScalarKind::F32, "float"); }
TEST(RouteGemmEquivalence, UnsetNowMeansAutoDouble)        { expect_unset_equals_auto<double>(ScalarKind::F64, "double"); }
TEST(RouteGemmEquivalence, UnsetNowMeansAutoComplexFloat)  { expect_unset_equals_auto<std::complex<float>>(ScalarKind::C32, "complex<float>"); }
TEST(RouteGemmEquivalence, UnsetNowMeansAutoComplexDouble) { expect_unset_equals_auto<std::complex<double>>(ScalarKind::C64, "complex<double>"); }

// --- the replica itself must be faithful ----------------------------------

TEST(RouteGemmEquivalence, ReplicaIsFaithful) {
    // Behaviours transcribed from gemm_variant.hh that are easy to get wrong.
    // If the replica drifts, the diff above would pass vacuously.
    OpShape s; s.op = Op::gemm; s.scalar = ScalarKind::F32;
    s.m = s.n = s.k = 256; s.batch = 512; s.is_gpu = true;

    // UNSET means Vendor for gemm -- not Auto.
    EXPECT_FALSE(legacy_use_sycl_custom<float>(legacy_request_from(nullptr), s));
    // ...and "auto" does reach the window.
    EXPECT_TRUE(legacy_use_sycl_custom<float>(legacy_request_from("auto"), s));
    // "sycl" bypasses the GPU check and the window entirely.
    OpShape cpu = s; cpu.is_gpu = false; cpu.m = cpu.n = cpu.k = 4096;
    EXPECT_TRUE(legacy_use_sycl_custom<float>(legacy_request_from("sycl"), cpu));
    // ...but never bypasses the correctness gate.
    OpShape het = s; het.heterogeneous_batch = true;
    EXPECT_FALSE(legacy_use_sycl_custom<float>(legacy_request_from("sycl"), het));
    // "native" and "cublasdx" both mean "not the SYCL kernel".
    EXPECT_FALSE(legacy_use_sycl_custom<float>(legacy_request_from("native"), s));
    EXPECT_FALSE(legacy_use_sycl_custom<float>(legacy_request_from("cublasdx"), s));
    // complex is excluded outright even inside the window.
    EXPECT_FALSE(legacy_use_sycl_custom<std::complex<float>>(legacy_request_from("auto"), s));
    // double has a different window: no lower bound, no transpose arm.
    OpShape d = s; d.scalar = ScalarKind::F64; d.m = d.n = d.k = 64;
    EXPECT_TRUE(legacy_use_sycl_custom<double>(legacy_request_from("auto"), d));
}

// --- the split itself is meaningful ---------------------------------------

TEST(RouteGemmEquivalence, SupportsIsCorrectnessOnlyNotSpeed) {
    // The trap the spec names: if the measured window leaked into supports(), a
    // large float GEMM would have NO supported native route, and vendor-off
    // would break an op that works today.
    using Table = RouteTable<Op::gemm, float>;
    OpShape big; big.op = Op::gemm; big.scalar = ScalarKind::F32;
    big.m = big.n = big.k = 1024; big.batch = 256; big.is_gpu = true;

    const Route native{Origin::Native, Algorithm::RegisterTiled};
    EXPECT_TRUE(Table::supports(native, big))
        << "1024^3 batch 256 must remain SUPPORTED even though it is outside the "
           "measured window -- otherwise there is no native route at all";
    EXPECT_FALSE(Table::preferred(native, big))
        << "...but it is not preferred, because the window stops at 512";
}

TEST(RouteGemmEquivalence, CorrectnessGateSurvivesForcing) {
    using Table = RouteTable<Op::gemm, float>;

    // The rule under test is "forcing cannot select a route that computes the
    // WRONG ANSWER" -- forcing bypasses preferred(), never supports().
    //
    // The example was a heterogeneous batch until WP2 C2 made that supported.
    // Swapped for a non-Default ComputePrecision, which no native kernel serves:
    // select_kernel_variant has no TF32 path at all, so this is a correctness
    // gate and not a speed one. Picking a still-unsupported example matters more
    // than the specific shape -- a test whose premise has quietly become false
    // passes for the wrong reason.
    OpShape unsupported; unsupported.op = Op::gemm; unsupported.scalar = ScalarKind::F32;
    unsupported.m = unsupported.n = unsupported.k = 256;
    unsupported.batch = 512; unsupported.is_gpu = true;
    unsupported.precision = ComputePrecision::F32;

    const Route native{Origin::Native, Algorithm::RegisterTiled};
    EXPECT_FALSE(Table::supports(native, unsupported));
    const Route chosen = resolve_gemm_route<float>(native, unsupported);
    EXPECT_TRUE(is_vendor(chosen));

    // And the retired example, asserted in its new direction so the change is
    // pinned rather than merely absent.
    OpShape het; het.op = Op::gemm; het.scalar = ScalarKind::F32;
    het.m = het.n = het.k = 256; het.batch = 512; het.is_gpu = true;
    het.heterogeneous_batch = true;
    EXPECT_TRUE(Table::supports(native, het))
        << "WP2 C2: the facade walks a heterogeneous batch, so native supports it";
    EXPECT_FALSE(Table::preferred(native, het))
        << "...but a per-item loop is a cost, so it is never the preferred choice";
}

TEST(RouteGemmEquivalence, SupportedButUnpreferredIsReachedOnlyWithoutVendor) {
    // The bug this test caught on first run. 8x8x8 at batch 1 is SUPPORTED by
    // the native kernel (it computes the right answer) but is far outside the
    // measured window. Taking "the first supported route" would pick native --
    // the order lists it first -- and silently invert GEMM's vendor-by-default.
    OpShape tiny; tiny.op = Op::gemm; tiny.scalar = ScalarKind::F32;
    tiny.m = tiny.n = tiny.k = 8; tiny.batch = 1; tiny.is_gpu = true;
    const Route automatic{Origin::Auto, Algorithm::Auto};

    EXPECT_TRUE(is_vendor(resolve_gemm_route<float>(automatic, tiny, /*vendor_available=*/true)))
        << "with a vendor present, an unpreferred native route must not be chosen";

    // With no vendor compiled in, correctness beats preference: a supported
    // route is better than no route at all. This is the vendor-off
    // configuration the work package is building toward.
    const Route chosen = resolve_gemm_route<float>(automatic, tiny, /*vendor_available=*/false);
    EXPECT_TRUE(is_native(chosen))
        << "with no vendor available, the supported native route must be chosen";

    // ...but only when it is actually supported: vendor-off must not
    // manufacture a wrong answer.
    //
    // This used to use a heterogeneous batch as the unsupported example. WP2 C2
    // made heterogeneous SUPPORTED -- the facade walks the batch and each member
    // is homogeneous by construction -- so that example stopped testing
    // anything, and the assertion failed rather than silently passing, which is
    // the behaviour worth having. Substituted with a genuinely unsupported
    // shape: a non-Default ComputePrecision, which no native kernel serves
    // (route_gemm.hh supports(), and select_kernel_variant has no TF32 path).
    OpShape unsupported = tiny;
    unsupported.precision = ComputePrecision::F32;
    // Aliased because the comma in RouteTable<Op::gemm, float> is a macro
    // argument separator to the preprocessor, not a template argument.
    using GemmTable = RouteTable<Op::gemm, float>;
    ASSERT_FALSE(GemmTable::supports(
                     Route{Origin::Native, Algorithm::RegisterTiled}, unsupported))
        << "the premise: this shape must actually be unsupported";
    EXPECT_TRUE(is_vendor(resolve_gemm_route<float>(automatic, unsupported, /*vendor_available=*/false)))
        << "an unsupported route must never be selected, even with no alternative";

    // And the retired example now behaves the other way, on purpose.
    OpShape het = tiny; het.heterogeneous_batch = true;
    EXPECT_TRUE(is_native(resolve_gemm_route<float>(automatic, het, /*vendor_available=*/false)))
        << "WP2 C2: heterogeneous batch is supported natively via the facade loop";
}

TEST(RouteGemmEquivalence, ForcedVendorStillDegradesWhenThereIsNoVendor) {
    // This test used to be called DefaultedVendorAlsoDegradesWhenThereIsNoVendor
    // and its premise was that GEMM's unset default IS Vendor, so an ordinary
    // call arrived as a forced Vendor request. WP2 E6 flipped that default to
    // Auto, so the premise is now false -- but the behaviour it guards is not:
    // an EXPLICITLY forced Vendor must still degrade to native where no vendor
    // was compiled in, and that path is now reached only by explicit forcing.
    OpShape tiny; tiny.op = Op::gemm; tiny.scalar = ScalarKind::F32;
    tiny.m = tiny.n = tiny.k = 8; tiny.batch = 1; tiny.is_gpu = true;

    // Pin the new default in its own right rather than merely dropping the old
    // assertion: a deleted assertion cannot detect a silent revert.
    const Route defaulted = legacy_unset_default(Op::gemm);
    EXPECT_EQ(defaulted.origin, Origin::Auto)
        << "WP2 E6: GEMM's unset default is Auto, like every other op";

    // batch 1 and 8x8 is outside every measured window, so Auto defers to the
    // vendor when there is one -- the flip did not make Auto mean "always
    // native".
    EXPECT_TRUE(is_vendor(resolve_gemm_route<float>(defaulted, tiny, /*vendor_available=*/true)));
    EXPECT_TRUE(is_native(resolve_gemm_route<float>(defaulted, tiny, /*vendor_available=*/false)))
        << "a vendor that is not compiled in cannot be the answer";

    // And an explicitly forced vendor degrades the same way. Silently running
    // native is the better failure than dispatching to a library that is not
    // there; announcing it is the job of the S6 gate, not of the resolver.
    const Route explicit_vendor{Origin::Vendor, Algorithm::Auto};
    EXPECT_TRUE(is_native(resolve_gemm_route<float>(explicit_vendor, tiny, /*vendor_available=*/false)));

    // Still never at the cost of correctness -- with an example that is still
    // unsupported after WP2 C2 (see the note in the test above).
    OpShape unsupported = tiny;
    unsupported.precision = ComputePrecision::F32;
    EXPECT_TRUE(is_vendor(resolve_gemm_route<float>(defaulted, unsupported, /*vendor_available=*/false)));
}

TEST(RouteGemmEquivalence, ResolutionIsPureAndRepeatable) {
    // gemm and gemm_buffer_size must reach the same route by construction. A
    // pure resolver gives that for free; the current code relies on a comment.
    OpShape s; s.op = Op::gemm; s.scalar = ScalarKind::F32;
    s.m = s.n = s.k = 256; s.batch = 512; s.is_gpu = true;
    const Route a = resolve_gemm_route<float>(Route{Origin::Auto, Algorithm::Auto}, s);
    const Route b = resolve_gemm_route<float>(Route{Origin::Auto, Algorithm::Auto}, s);
    EXPECT_EQ(a, b);
}
