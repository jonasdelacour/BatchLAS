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
    Route forced = legacy_unset_default(Op::gemm);   // Vendor when unset
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

template <typename T>
void expect_equivalent(ScalarKind kind, const char* type_name) {
    size_t compared = 0, native_cases = 0;
    for (const char* env : kEnvValues) {
        for (const OpShape& s : shape_grid(kind)) {
            const bool old_native = legacy_use_sycl_custom<T>(legacy_request_from(env), s);
            const bool new_native = new_uses_native<T>(env, s);
            ++compared;
            if (old_native) ++native_cases;

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
}

} // namespace

TEST(RouteGemmEquivalence, Float)         { expect_equivalent<float>(ScalarKind::F32, "float"); }
TEST(RouteGemmEquivalence, Double)        { expect_equivalent<double>(ScalarKind::F64, "double"); }
TEST(RouteGemmEquivalence, ComplexFloat)  { expect_equivalent<std::complex<float>>(ScalarKind::C32, "complex<float>"); }
TEST(RouteGemmEquivalence, ComplexDouble) { expect_equivalent<std::complex<double>>(ScalarKind::C64, "complex<double>"); }

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
    OpShape het; het.op = Op::gemm; het.scalar = ScalarKind::F32;
    het.m = het.n = het.k = 256; het.batch = 512; het.is_gpu = true;
    het.heterogeneous_batch = true;

    const Route native{Origin::Native, Algorithm::RegisterTiled};
    EXPECT_FALSE(Table::supports(native, het));
    // Forcing must not be able to select a route that computes the wrong answer.
    const Route chosen = resolve_gemm_route<float>(native, het);
    EXPECT_TRUE(is_vendor(chosen));
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

    // ...but only when it is actually supported. A heterogeneous batch is a
    // correctness failure, so vendor-off must not manufacture a wrong answer.
    OpShape het = tiny; het.heterogeneous_batch = true;
    EXPECT_TRUE(is_vendor(resolve_gemm_route<float>(automatic, het, /*vendor_available=*/false)))
        << "an unsupported route must never be selected, even with no alternative";
}

TEST(RouteGemmEquivalence, DefaultedVendorAlsoDegradesWhenThereIsNoVendor) {
    // The gap the test above did NOT close, found only once the adapter was
    // wired up: it passes Origin::Auto, which the real call path never produces
    // for an unset environment. GEMM's unset default is Vendor
    // (legacy_unset_default), so an ordinary call arrives as a FORCED Vendor
    // request and used to be returned verbatim -- making the degradation above
    // unreachable outside this file.
    OpShape tiny; tiny.op = Op::gemm; tiny.scalar = ScalarKind::F32;
    tiny.m = tiny.n = tiny.k = 8; tiny.batch = 1; tiny.is_gpu = true;
    const Route defaulted = legacy_unset_default(Op::gemm);
    ASSERT_TRUE(is_vendor(defaulted)) << "the premise of this test";

    EXPECT_TRUE(is_vendor(resolve_gemm_route<float>(defaulted, tiny, /*vendor_available=*/true)));
    EXPECT_TRUE(is_native(resolve_gemm_route<float>(defaulted, tiny, /*vendor_available=*/false)))
        << "a vendor that is not compiled in cannot be the answer";

    // And an explicitly forced vendor degrades the same way. Silently running
    // native is the better failure than dispatching to a library that is not
    // there; announcing it is the job of the S6 gate, not of the resolver.
    const Route explicit_vendor{Origin::Vendor, Algorithm::Auto};
    EXPECT_TRUE(is_native(resolve_gemm_route<float>(explicit_vendor, tiny, /*vendor_available=*/false)));

    // Still never at the cost of correctness.
    OpShape het = tiny; het.heterogeneous_batch = true;
    EXPECT_TRUE(is_vendor(resolve_gemm_route<float>(defaulted, het, /*vendor_available=*/false)));
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
