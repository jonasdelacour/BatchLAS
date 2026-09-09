// batchlas::Settings -- readiness audit A-3. ~106 BATCHLAS_* variables were read at
// ~77 scattered getenv sites, so ambient process state silently chose which kernel ran
// and whether arguments were validated, with no programmatic equivalent and no way
// for an embedding application to lock it down. The full surface is documented in
// docs/cpp-api.md#configuration and on the fields of <batchlas/settings.hh>; what
// this file pins is the five behaviours the header promises:
//
//   settings()                  reads the environment once, under call_once
//   configure(const Settings&)  programmatic override, the last word over the
//                               environment; std::runtime_error once a Queue exists
//   detail::reload_settings()   re-read, called from BOTH ends of ScopedEnvVar
//   the unsafe gate             BATCHLAS_ALLOW_UNSAFE_ENV, default OFF
//   one field, one reader       BATCHLAS_GEMM_VARIANT had two
//
// ORDERING CONTRACT: configure() is permitted only until the first Queue exists, so
// the two SettingsConfigure cases must run in definition order and before any case
// that builds one. GoogleTest preserves definition order unless --gtest_shuffle is
// passed, and nothing here passes it. The first case asserts on
// detail::queue_constructed() rather than skipping if it finds the latch already
// closed -- a case that quietly does nothing is the failure mode this repository has
// documented twelve times over.

#include <gtest/gtest.h>

#include <batchlas/backend_config.h>
#include <batchlas/settings.hh>
#include <batchlas/blas/dispatch/route.hh>
#include <batchlas/blas/dispatch/route_env.hh>
#include <batchlas/blas/functions.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/blas/options.hh>
#include <batchlas/blas/queue-dispatch.hh>
#include <batchlas/util/env.hh>
#include <batchlas/util/sycl-device-queue.hh>

// The second reader of BATCHLAS_GEMM_VARIANT, and not on the public path.
#include "../src/backends/gemm_variant.hh"

#include <stdexcept>
#include <string>
#include <vector>

// #cmakedefine01: always defined, to 0 or 1, so `#if` is the correct test. Drop the
// header and `#if BATCHLAS_ALLOW_UNSAFE_ENV` reads as 0, asserting the wrong arm
// while still reporting green. Fail the compile instead.
// Deliberately no `#ifndef ... #define 0` fallback here, unlike src/util/settings.cc:
// this file must not be able to assert one arm while the library was built for the
// other. A build tree configured before the option existed hits this and has to be
// re-configured, which is the loud failure we want.
#if !defined(BATCHLAS_ALLOW_UNSAFE_ENV)
#error "BATCHLAS_ALLOW_UNSAFE_ENV is undefined: batchlas/backend_config.h was not \
reached, or this build tree predates the option. Re-run cmake."
#endif

using namespace batchlas;
using batchlas::dispatch::Op;
using batchlas::dispatch::Origin;
using batchlas::dispatch::parse_route_env;

namespace {

constexpr size_t kGemm = static_cast<size_t>(Op::gemm);

const EnvValue& gemm_route() { return settings().routing.canonical_route(Op::gemm); }
const EnvValue& gemm_legacy() { return settings().routing.legacy_route(Op::gemm); }

// Both spellings cleared, so a case starts from a known state whatever the shell or
// the ctest ENVIRONMENT property said -- tests/CMakeLists.txt pins
// BATCHLAS_GEMM_ROUTE=native for one rerun, and a bisect leaves one exported.
struct ClearGemmRoute {
    ScopedEnvVar canonical{"BATCHLAS_GEMM_ROUTE", nullptr};
    ScopedEnvVar legacy{"BATCHLAS_GEMM_VARIANT", nullptr};
};

}  // namespace

// (a) An explicit configure() is the last word over the environment. Before it, the
// only way to route gemm through the native kernel was to reach into the process
// environment, which every other component of the process also sees.
TEST(SettingsConfigure, ConfigureBeatsTheEnvironment) {
    ASSERT_FALSE(detail::queue_constructed())
        << "a Queue already exists, so this case ran after one that builds one; "
           "configure() is closed. See the ordering contract at the top of this file.";

    const Settings original = settings();

    // Prove the environment is being honoured first, or the assertion below proves
    // nothing: a configure() that "wins" against a variable nobody read is vacuous.
    ScopedEnvVar pin("BATCHLAS_GEMM_ROUTE", "vendor");
    ASSERT_EQ(gemm_route().value(), "vendor");
    ASSERT_EQ(parse_route_env(Op::gemm).route.origin, Origin::Vendor);

    Settings s = settings();
    s.routing.canonical[kGemm] = EnvValue::of("native");
    ASSERT_NO_THROW(configure(s));

    EXPECT_EQ(gemm_route().value(), "native");
    EXPECT_EQ(parse_route_env(Op::gemm).route.origin, Origin::Native)
        << "the route adapters still read the environment directly rather than settings()";

    // Restore, while configure() is still permitted, so the cases below start clean.
    // (pin's destructor reloads from the environment on the way out anyway; this is
    // belt and braces for the case where the assertions above fail early.)
    configure(original);
}

// (b) configure() after a Queue exists throws. A route changed halfway through a run
// makes two calls in one process disagree about which kernel they used, and worse,
// several of these knobs are read by a *_buffer_size() query as well as by the
// matching solve -- so a mid-run change under-sizes a workspace the caller has
// already allocated. Queue's constructor is the latch because it is the earliest
// point at which a dispatch decision can already have been made. THIS CASE CLOSES
// THE LATCH for the rest of the binary: nothing after it may expect configure() to
// succeed.
TEST(SettingsConfigure, ConfigureAfterAQueueExistsThrows) {
    ClearGemmRoute clear;   // ...so the "nothing was applied" check below cannot be
                            // satisfied by an ambient BATCHLAS_GEMM_ROUTE=vendor.
    Queue q;
    EXPECT_TRUE(detail::queue_constructed());

    Settings s = settings();
    s.routing.canonical[kGemm] = EnvValue::of("vendor");
    EXPECT_THROW(configure(s), std::runtime_error);

    // ...and the refusal is a refusal, not a partial application.
    EXPECT_FALSE(gemm_route().is_set());
}

// (c) ScopedEnvVar still works, in both directions -- THE regression guard here.
// Fifteen test files mutate the environment through it and then expect the new value
// to take effect; a call_once settings() on its own makes every one of them silently
// measure the default arm and pass. reload_settings() from both the constructor and
// the destructor is what keeps them working unchanged. Observed through
// parse_route_env as well as through the field, because parse_route_env is what
// those fifteen files actually reach.
TEST(SettingsEnvironment, ScopedEnvVarIsStillObservedAndStillReverts) {
    ClearGemmRoute clear;

    ASSERT_FALSE(parse_route_env(Op::gemm).found);
    ASSERT_FALSE(gemm_route().is_set());

    {
        ScopedEnvVar pin("BATCHLAS_GEMM_ROUTE", "native");

        EXPECT_EQ(gemm_route().value(), "native")
            << "settings() cached the environment and never re-read it";

        const auto parsed = parse_route_env(Op::gemm);
        EXPECT_TRUE(parsed.found);
        EXPECT_EQ(parsed.route.origin, Origin::Native);
        EXPECT_EQ(parsed.source.variable, "BATCHLAS_GEMM_ROUTE");
        EXPECT_FALSE(parsed.source.legacy);
    }

    EXPECT_FALSE(gemm_route().is_set())
        << "ScopedEnvVar's destructor did not reload; a pin leaked into the next case";
    EXPECT_FALSE(parse_route_env(Op::gemm).found);

    // Nested, as the route suites write it: the inner pin wins, then hands the
    // outer one back rather than unsetting it.
    {
        ScopedEnvVar outer("BATCHLAS_GEMM_ROUTE", "vendor");
        {
            ScopedEnvVar inner("BATCHLAS_GEMM_ROUTE", "native");
            EXPECT_EQ(gemm_route().value(), "native");
        }
        EXPECT_EQ(gemm_route().value(), "vendor");
    }
}

// (d) BATCHLAS_SKIP_POINTER_CHECKS is gated by the build option.
//
// WHICH ARM RUNS WHERE, because a guarded case that compiles to nothing in the
// preset everybody uses is worse than no case at all:
//
//   BATCHLAS_ALLOW_UNSAFE_ENV=OFF -> the `#else` arm.  Presets: cuda. Also any
//       plain `cmake -B build` and every install/release build, since OFF is the
//       option's default. This is the arm that proves the lock works.
//   BATCHLAS_ALLOW_UNSAFE_ENV=ON  -> the `#if` arm.   Presets: dev, dev-tests,
//       fast-dev, dev-gpu, dev-gpu-tests, benchmarks. dev-tests is the preset most
//       developers run ctest from, so THIS is the arm you will normally see; the
//       locked arm is covered by the cuda preset, which is the pre-push gate.
//
// The ON arm deliberately does not make the poisoned call. With the checks genuinely
// disabled, host memory reaches the device as a wild address and the process dies of
// SIGABRT from inside the CUDA runtime during teardown, which no catch block and no
// test framework can intercept. It asserts on the gate instead.
TEST(SettingsUnsafe, SkipPointerChecksHonoursTheBuildOption) {
    ScopedEnvVar skip("BATCHLAS_SKIP_POINTER_CHECKS", "1");

    // One variable, two readers -- the Settings field and the predicate every entry
    // point consults. They must not be able to disagree.
    EXPECT_EQ(settings().unsafe.skip_pointer_checks, !detail::pointer_checks_enabled());

#if BATCHLAS_ALLOW_UNSAFE_ENV
    EXPECT_TRUE(settings().unsafe.skip_pointer_checks)
        << "this build allows the unsafe group, so the variable must take effect";
    EXPECT_FALSE(detail::pointer_checks_enabled());
#else
    EXPECT_FALSE(settings().unsafe.skip_pointer_checks)
        << "BATCHLAS_ALLOW_UNSAFE_ENV is OFF, so the environment must not disarm the "
           "USM check";
    EXPECT_TRUE(detail::pointer_checks_enabled());

    // The check is not merely still enabled in Settings; it still fires. Ordinary
    // host memory handed to a GPU queue must be rejected host-side, by name, before
    // any device work is enqueued.
    Queue q;
    if (q.device().type == DeviceType::CPU) {
        GTEST_SKIP() << "host memory is legitimately device-accessible on a CPU device; "
                        "the gate assertions above already ran";
    }
    constexpr int n = 4;
    std::vector<float> host(n * n, 1.0f);
    MatrixView<float, MatrixFormat::Dense> poisoned(host.data(), n, n);
    Matrix<float, MatrixFormat::Dense> b(n, n, 1), c(n, n, 1);
    EXPECT_THROW(gemm(q, poisoned, b.view(), c.view(), GemmOptions<float>{}),
                 std::invalid_argument);
#endif
}

// (e) The two readers of BATCHLAS_GEMM_VARIANT now read the same string:
// parse_route_env(Op::gemm) via legacy_variable_for, and gemm_variant_request(),
// which had its own tolower parser and its own getenv.
//
// The SOURCE is what is pinned, not the semantics. Their unset defaults differ on
// purpose -- Auto against Vendor, recorded at route_gemm_equivalence_tests.cc:28 --
// and unifying that would be a behaviour change deciding which kernel a bare gemm()
// runs. The later blocks are the load-bearing ones: one call_once read with the
// reload hook missing gives a first answer that is right and every one after it
// stale.
TEST(SettingsRouting, BothReadersOfGemmVariantSeeTheSameValue) {
    ScopedEnvVar clear_canonical("BATCHLAS_GEMM_ROUTE", nullptr);

    {
        ScopedEnvVar v("BATCHLAS_GEMM_VARIANT", "sycl");
        const auto parsed = parse_route_env(Op::gemm);
        EXPECT_TRUE(parsed.found);
        EXPECT_TRUE(parsed.source.legacy);
        EXPECT_EQ(parsed.source.variable, "BATCHLAS_GEMM_VARIANT");
        EXPECT_EQ(parsed.source.value, "sycl");
        EXPECT_EQ(gemm_legacy().value(), "sycl");
        EXPECT_EQ(backend::gemm_variant_request(), backend::GemmVariantRequest::Sycl);
    }
    {
        ScopedEnvVar v("BATCHLAS_GEMM_VARIANT", "cublasdx");
        EXPECT_EQ(parse_route_env(Op::gemm).source.value, "cublasdx");
        EXPECT_EQ(gemm_legacy().value(), "cublasdx");
        EXPECT_EQ(backend::gemm_variant_request(), backend::GemmVariantRequest::CuBLASDx);
    }
    {
        ScopedEnvVar v("BATCHLAS_GEMM_VARIANT", nullptr);
        EXPECT_FALSE(parse_route_env(Op::gemm).found);
        EXPECT_FALSE(gemm_legacy().is_set());
        // The documented asymmetry, pinned so a later "cleanup" has to argue with it.
        EXPECT_EQ(backend::gemm_variant_request(), backend::GemmVariantRequest::Vendor);
    }
}
