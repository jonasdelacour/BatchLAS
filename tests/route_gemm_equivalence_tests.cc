// Equivalence test for the Route-based GEMM dispatch: it diffs the CHOSEN ROUTE,
// not the timing, against a replica of the legacy gemm_use_sycl_custom() decision,
// and pins the intended divergences.
//
// The replica is transcribed from src/backends/gemm_variant.hh rather than called;
// `ReplicaIsFaithful` pins it, so a drift in the replica cannot make the diff pass
// vacuously. Window and evidence: docs/perf/gemm.md#the-preferred-window-as-implemented

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

// Replica of the legacy decision, transcribed from src/backends/gemm_variant.hh.

enum class LegacyRequest { Vendor, Sycl, Native, CuBLASDx, Auto };

// gemm_variant_request(): UNSET means Vendor for gemm, not Auto.
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

// The new decision, expressed as the same boolean so the two are comparable.
template <typename T>
bool new_uses_native(const char* env_value, const OpShape& s) {
    Route forced = legacy_unset_default(Op::gemm);   // Auto, not the legacy Vendor
    if (env_value) {
        if (const auto parsed = parse_legacy_route_value(Op::gemm, env_value)) {
            forced = *parsed;
        } else {
            forced = legacy_unset_default(Op::gemm);
        }
    }
    const Route chosen = resolve_gemm_route<T>(forced, s);
    return is_native(chosen);
}

// The env spellings benchmark scripts use; nullptr is unset.
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
    // Cells where a correctness gate and a speed window are most easily confused.
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

// The intended divergences. Deliberately an exception list rather than an edit to
// the replica: editing the replica would make the two agree by construction. Each
// divergence is counted separately below so none can quietly stop being reached.
// evidence: docs/perf/gemm.md#evidence-for-each-boundary
enum class Divergence { None, HeterogeneousWidened, FloatWindowNarrowed, DefaultFlipped,
                        DoubleWindowWidened };

template <typename T>
Divergence classify_divergence(const char* env, const OpShape& s,
                               bool old_native, bool new_native) {
    const bool forced_native = env != nullptr &&
        (std::string_view(env) == "sycl" || std::string_view(env) == "custom");
    if (s.heterogeneous_batch && forced_native && !old_native && new_native) {
        return Divergence::HeterogeneousWidened;
    }
    // Narrow on purpose: only an unset env, and only in the flip's direction.
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
    if constexpr (std::is_same_v<T, double>) {
        if (!old_native && new_native) {
            const bool non_square = (s.m != s.n || s.n != s.k);
            if (non_square || s.max_dim() > 512) {
                return Divergence::DoubleWindowWidened;
            }
        }
    }
    return Divergence::None;
}

template <typename T>
void expect_equivalent(ScalarKind kind, const char* type_name) {
    size_t compared = 0, native_cases = 0, het_widened = 0, float_narrowed = 0,
           default_flipped = 0, double_widened = 0;
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
            if (d == Divergence::DoubleWindowWidened)  { ++double_widened; continue; }

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
    EXPECT_GT(native_cases, 0u) << "grid never exercised the native route for " << type_name;

    EXPECT_GT(het_widened, 0u)
        << "grid no longer reaches the intended heterogeneous divergence for "
        << type_name << " -- the exception is now vacuous";

    if constexpr (std::is_same_v<T, float>) {
        EXPECT_GT(float_narrowed, 0u)
            << "grid no longer reaches the WP2 E4 float narrowing -- the "
               "exception is now vacuous";
    } else {
        EXPECT_EQ(float_narrowed, 0u)
            << "the E4 narrowing is float-only but fired for " << type_name;
    }

    if constexpr (is_std_complex_v<T>) {
        EXPECT_EQ(default_flipped, 0u)
            << "complex must not route native by default: preferred() refuses it, "
               "and the register ladder for complex needs min_dim >= 256 and an "
               "aligned NN shape (see docs/perf/gemm.md#evidence-for-each-boundary)";
    } else {
        EXPECT_GT(default_flipped, 0u)
            << "grid no longer reaches the WP2 E6 default flip for " << type_name
            << " -- the exception is now vacuous";
    }

    if constexpr (std::is_same_v<T, double>) {
        EXPECT_GT(double_widened, 0u)
            << "grid no longer reaches the WP2 E5 double widening -- the "
               "exception is now vacuous";
    } else {
        EXPECT_EQ(double_widened, 0u)
            << "the E5 widening is double-only but fired for " << type_name;
    }
}

// The positive form of the DefaultFlipped exception: unset must decide as "auto".
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

TEST(RouteGemmEquivalence, ReplicaIsFaithful) {
    // Behaviours transcribed from gemm_variant.hh that are easy to get wrong.
    OpShape s; s.op = Op::gemm; s.scalar = ScalarKind::F32;
    s.m = s.n = s.k = 256; s.batch = 512; s.is_gpu = true;

    // UNSET means Vendor for gemm -- not Auto.
    EXPECT_FALSE(legacy_use_sycl_custom<float>(legacy_request_from(nullptr), s));
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

TEST(RouteGemmEquivalence, SupportsIsCorrectnessOnlyNotSpeed) {
    // If the measured window leaked into supports(), a large float GEMM would have
    // NO supported native route and a vendor-off build would break a working op.
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

    // Forcing bypasses preferred(), never supports(), so it cannot select a route
    // that computes the WRONG ANSWER. Here the unsupported shape is a non-Default
    // ComputePrecision: select_kernel_variant has no TF32 path.
    OpShape unsupported; unsupported.op = Op::gemm; unsupported.scalar = ScalarKind::F32;
    unsupported.m = unsupported.n = unsupported.k = 256;
    unsupported.batch = 512; unsupported.is_gpu = true;
    unsupported.precision = ComputePrecision::F32;

    const Route native{Origin::Native, Algorithm::RegisterTiled};
    EXPECT_FALSE(Table::supports(native, unsupported));
    const Route chosen = resolve_gemm_route<float>(native, unsupported);
    EXPECT_TRUE(is_vendor(chosen));

    OpShape het; het.op = Op::gemm; het.scalar = ScalarKind::F32;
    het.m = het.n = het.k = 256; het.batch = 512; het.is_gpu = true;
    het.heterogeneous_batch = true;
    EXPECT_TRUE(Table::supports(native, het))
        << "WP2 C2: the facade walks a heterogeneous batch, so native supports it";
    EXPECT_FALSE(Table::preferred(native, het))
        << "...but a per-item loop is a cost, so it is never the preferred choice";
}

TEST(RouteGemmEquivalence, SupportedButUnpreferredIsReachedOnlyWithoutVendor) {
    // 8x8x8 at batch 1 is SUPPORTED but far outside the measured window. Taking
    // "the first supported route" would pick native and invert vendor-by-default.
    OpShape tiny; tiny.op = Op::gemm; tiny.scalar = ScalarKind::F32;
    tiny.m = tiny.n = tiny.k = 8; tiny.batch = 1; tiny.is_gpu = true;
    const Route automatic{Origin::Auto, Algorithm::Auto};

    EXPECT_TRUE(is_vendor(resolve_gemm_route<float>(automatic, tiny, /*vendor_available=*/true)))
        << "with a vendor present, an unpreferred native route must not be chosen";

    const Route chosen = resolve_gemm_route<float>(automatic, tiny, /*vendor_available=*/false);
    EXPECT_TRUE(is_native(chosen))
        << "with no vendor available, the supported native route must be chosen";

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

    OpShape het = tiny; het.heterogeneous_batch = true;
    EXPECT_TRUE(is_native(resolve_gemm_route<float>(automatic, het, /*vendor_available=*/false)))
        << "WP2 C2: heterogeneous batch is supported natively via the facade loop";
}

TEST(RouteGemmEquivalence, ForcedVendorStillDegradesWhenThereIsNoVendor) {
    OpShape tiny; tiny.op = Op::gemm; tiny.scalar = ScalarKind::F32;
    tiny.m = tiny.n = tiny.k = 8; tiny.batch = 1; tiny.is_gpu = true;

    const Route defaulted = legacy_unset_default(Op::gemm);
    EXPECT_EQ(defaulted.origin, Origin::Auto)
        << "WP2 E6: GEMM's unset default is Auto, like every other op";

    // 8x8 at batch 1 is outside every measured window, so Auto defers to a vendor.
    EXPECT_TRUE(is_vendor(resolve_gemm_route<float>(defaulted, tiny, /*vendor_available=*/true)));
    EXPECT_TRUE(is_native(resolve_gemm_route<float>(defaulted, tiny, /*vendor_available=*/false)))
        << "a vendor that is not compiled in cannot be the answer";

    // A forced vendor degrades the same way: running native is a better failure than
    // dispatching to a library that is not there.
    const Route explicit_vendor{Origin::Vendor, Algorithm::Auto};
    EXPECT_TRUE(is_native(resolve_gemm_route<float>(explicit_vendor, tiny, /*vendor_available=*/false)));

    // Still never at the cost of correctness.
    OpShape unsupported = tiny;
    unsupported.precision = ComputePrecision::F32;
    EXPECT_TRUE(is_vendor(resolve_gemm_route<float>(defaulted, unsupported, /*vendor_available=*/false)));
}

TEST(RouteGemmEquivalence, ResolutionIsPureAndRepeatable) {
    // gemm and gemm_buffer_size must reach the same route by construction.
    OpShape s; s.op = Op::gemm; s.scalar = ScalarKind::F32;
    s.m = s.n = s.k = 256; s.batch = 512; s.is_gpu = true;
    const Route a = resolve_gemm_route<float>(Route{Origin::Auto, Algorithm::Auto}, s);
    const Route b = resolve_gemm_route<float>(Route{Origin::Auto, Algorithm::Auto}, s);
    EXPECT_EQ(a, b);
}
