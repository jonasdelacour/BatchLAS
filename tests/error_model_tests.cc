// What BatchLAS throws, as a type a consumer can catch.
//
// Before include/batchlas/error.hh there were 573 throw sites in include/ + src/
// and not one BatchLAS type among them: 321 std::invalid_argument, 231
// std::runtime_error, 13 std::logic_error, 5 std::out_of_range, 3
// std::bad_alloc. A consumer could not write `catch (const batchlas::error&)`,
// and could not tell a shape mismatch from an unsupported route from a
// workspace shortfall from a device failure -- which in a batched solver is the
// difference between "retry this batch smaller" and "abort the run".
//
// THREE PROPERTIES, ASSERTED SEPARATELY AT EVERY SITE
//
//   1. the SPECIFIC batchlas type is caught -- otherwise the type is decorative
//      and a consumer still cannot discriminate;
//   2. `catch (const std::exception&)` still catches it -- 43 existing catch
//      sites across tests/, benchmarks/ and examples/consumer/main.cc depend on
//      this, as does every `catch (const std::exception& e)` in user code;
//   3. `catch (const batchlas::exception&)` catches it, and `e.message()` still
//      reads the text -- the point of the tag base, and the property with no
//      fallback if it breaks.
//
// They are three separate try/catch blocks, never one block with three
// handlers: a single block only ever proves which handler comes FIRST, and what
// is under test is that all three match independently.
//
// WHY THE HIERARCHY RULES ARE STATIC_ASSERTS
//
// Two of the three ways to get this hierarchy wrong produce a program that
// compiles clean and calls std::terminate at the first throw (error.hh states
// both, and this file guards both):
//
//   * the tag base deriving from std::exception -- every leaf then has an
//     ambiguous std::exception subobject and property 2 silently disappears;
//   * the tag base inherited non-virtually -- a future two-arm class then has
//     two tag subobjects and property 3 silently disappears for it.
//
// EVIDENCE. include/batchlas/error.hh includes only <stdexcept> and <string>,
// so it compiles with host g++ -- no SYCL, no build of this project. Compiling
// it with `g++ -std=c++20 -I include` and probing every class:
//
//   invalid_argument   specific=1 std=1 tag=1 message="potrf: A must be square, got 100x50"
//   out_of_range       specific=1 std=1 tag=1 message="Matrix indices out of range"
//   error/unsupported/device_error/workspace_error/convergence_error/
//   internal_error/api_misuse    all  specific=1 std=1 tag=1, message preserved
//   REVERTED(runtime)  specific=0 std=1 tag=0     <- see "the regression" below
//   REVERTED(invalid)  specific=0 std=1 tag=0
//   unsupported as batchlas::error = 1, invalid_argument as batchlas::error = 0
//
// WHAT IS DELIBERATELY OUTSIDE THE HIERARCHY, and so is NOT caught by
// `catch (const batchlas::exception&)`:
//
//   * std::bad_alloc from the three sites where a sycl::malloc_* returned null.
//     It is the standard type for that and is what pybind11 maps to MemoryError.
//   * sycl::exception, raised by the SYCL runtime itself -- including everything
//     the device reports asynchronously at ctx.wait_and_throw().
//
// Registered with the `util` label: nothing here launches a kernel.

#include <gtest/gtest.h>

#include <batchlas/error.hh>

#include <batchlas/blas/matrix.hh>
#include <batchlas/util/mempool.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>

#include <cctype>
#include <cstddef>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

using namespace batchlas;

namespace {

// ---------------------------------------------------------------------------
// The compile-time contract
// ---------------------------------------------------------------------------

// Every class carries the tag.
static_assert(std::is_base_of_v<batchlas::exception, batchlas::invalid_argument>);
static_assert(std::is_base_of_v<batchlas::exception, batchlas::out_of_range>);
static_assert(std::is_base_of_v<batchlas::exception, batchlas::error>);
static_assert(std::is_base_of_v<batchlas::exception, batchlas::unsupported>);
static_assert(std::is_base_of_v<batchlas::exception, batchlas::device_error>);
static_assert(std::is_base_of_v<batchlas::exception, batchlas::workspace_error>);
static_assert(std::is_base_of_v<batchlas::exception, batchlas::convergence_error>);
static_assert(std::is_base_of_v<batchlas::exception, batchlas::internal_error>);
static_assert(std::is_base_of_v<batchlas::exception, batchlas::api_misuse>);

// Every class keeps the standard base its sites had before the migration. That
// is what makes this change type-only for existing catch sites -- and, through
// pybind11's default translator (python/ registers none of its own), what keeps
// the Python exception each call raises unchanged.
static_assert(std::is_base_of_v<std::invalid_argument, batchlas::invalid_argument>);
static_assert(std::is_base_of_v<std::out_of_range, batchlas::out_of_range>);
static_assert(std::is_base_of_v<std::runtime_error, batchlas::error>);
static_assert(std::is_base_of_v<std::runtime_error, batchlas::unsupported>);
static_assert(std::is_base_of_v<std::runtime_error, batchlas::device_error>);
static_assert(std::is_base_of_v<std::runtime_error, batchlas::workspace_error>);
static_assert(std::is_base_of_v<std::runtime_error, batchlas::convergence_error>);
static_assert(std::is_base_of_v<std::runtime_error, batchlas::internal_error>);
// api_misuse and internal_error stay under std::runtime_error deliberately.
// tests/mempool_tests.cc:1115 and :1131 catch std::runtime_error around
// src/queue.hh:53 (a Queue used from a foreign thread, an api_misuse site), and
// the second does so inside a std::thread lambda -- if that throw stops matching
// there it does not fail an assertion, it terminates the test binary.
static_assert(std::is_base_of_v<std::runtime_error, batchlas::api_misuse>);

// Rule 1: the tag must not derive from std::exception.
static_assert(!std::is_base_of_v<std::exception, batchlas::exception>,
              "batchlas::exception must NOT derive from std::exception: every leaf would then "
              "have an ambiguous std::exception base, catch(const std::exception&) would stop "
              "matching, and the first throw would reach std::terminate -- with no diagnostic.");

// Rule 2: the tag must be a VIRTUAL base.
//
// The obvious guard -- define a class deriving from both arms and check it
// converts to the tag -- cannot be written against this header: two
// exception_bridge bases both override the tag's message(), so such a class has
// no unique final overrider and is rejected outright ("no unique final overrider
// for 'virtual const char* batchlas::exception::message() const'"). This
// detector needs no such class and is exact: a static_cast FROM a base TO a
// derived type is ill-formed if and only if the base is virtual.
template <typename Derived, typename = void>
struct DowncastFromTagCompiles : std::false_type {};
template <typename Derived>
struct DowncastFromTagCompiles<
    Derived, std::void_t<decltype(static_cast<Derived*>(std::declval<batchlas::exception*>()))>>
    : std::true_type {};

// The control: a plainly non-virtual base DOES permit the downcast, so the
// detector below is armed in both directions and is not vacuously true.
struct NonVirtualBase {};
struct NonVirtualDerived : NonVirtualBase {};
template <typename Derived, typename = void>
struct DowncastFromNonVirtualCompiles : std::false_type {};
template <typename Derived>
struct DowncastFromNonVirtualCompiles<
    Derived, std::void_t<decltype(static_cast<Derived*>(std::declval<NonVirtualBase*>()))>>
    : std::true_type {};
static_assert(DowncastFromNonVirtualCompiles<NonVirtualDerived>::value,
              "the virtual-base detector is broken: it reports a non-virtual base as virtual");

static_assert(!DowncastFromTagCompiles<batchlas::error>::value,
              "batchlas::exception must be a VIRTUAL base of batchlas::error: a class deriving "
              "from two arms otherwise gets two tag subobjects and catch(const "
              "batchlas::exception&) silently stops matching it.");
static_assert(!DowncastFromTagCompiles<batchlas::invalid_argument>::value,
              "batchlas::exception must be a VIRTUAL base of batchlas::invalid_argument.");
static_assert(!DowncastFromTagCompiles<batchlas::out_of_range>::value,
              "batchlas::exception must be a VIRTUAL base of batchlas::out_of_range.");

// ---------------------------------------------------------------------------
// The three-way catch check
// ---------------------------------------------------------------------------

struct CatchResult {
    bool specific = false;
    bool std_exception = false;
    bool batchlas_tag = false;
    std::string via_what;
    std::string via_message;
};

template <typename Specific, typename Thrower>
CatchResult ProbeCatches(Thrower&& thrower) {
    CatchResult r;
    try { thrower(); } catch (const Specific&) { r.specific = true; } catch (...) {}
    try {
        thrower();
    } catch (const std::exception& e) {
        r.std_exception = true;
        r.via_what = e.what();
    } catch (...) {}
    try {
        thrower();
    } catch (const batchlas::exception& e) {
        r.batchlas_tag = true;
        r.via_message = e.message();
    } catch (...) {}
    return r;
}

// `needle` must appear in the text read BOTH ways: what() through the std::
// base, and message() through the tag. They are different members reaching the
// same string, and a hierarchy that broke one while keeping the other would look
// healthy from whichever side a test happened to check.
template <typename Specific, typename Thrower>
void ExpectThreeWayCatch(Thrower&& thrower, const char* needle) {
    const CatchResult r = ProbeCatches<Specific>(std::forward<Thrower>(thrower));
    EXPECT_TRUE(r.specific) << "the specific batchlas type did not catch it";
    EXPECT_TRUE(r.std_exception) << "catch(const std::exception&) stopped matching; "
                                    "43 existing catch sites depend on it";
    EXPECT_TRUE(r.batchlas_tag) << "catch(const batchlas::exception&) did not match; "
                                   "check that the tag base is inherited virtually";
    EXPECT_NE(r.via_what.find(needle), std::string::npos)
        << "message lost through what(): \"" << r.via_what << "\"";
    EXPECT_NE(r.via_message.find(needle), std::string::npos)
        << "message lost through batchlas::exception::message(): \"" << r.via_message << "\"";
}

}  // namespace

// ---------------------------------------------------------------------------
// One directly-thrown representative per class.
//
// These do not go through a call site, on purpose: they isolate the HIERARCHY
// from the CLASSIFICATION, so a site later re-adjudicated into a different class
// cannot make a hierarchy defect look like a classification change. The
// call-site tests further down are the other half.
//
// It is also the only way to cover internal_error: all 13 sites it replaces are
// internal invariants -- nine are [[noreturn]] "resolved to a native route with
// no linked kernel" helpers, two are commented "Unreachable" fit-check
// disagreements, two are un-injected internal seams -- so by construction none
// of them is reachable from a public call.
// ---------------------------------------------------------------------------

TEST(ErrorModelHierarchy, InvalidArgumentIsCaughtThreeWays) {
    ExpectThreeWayCatch<batchlas::invalid_argument>(
        [] { throw batchlas::invalid_argument("potrf: A must be square, got 100x50"); },
        "must be square");
}

TEST(ErrorModelHierarchy, OutOfRangeIsCaughtThreeWays) {
    // A separate class rather than a fold into invalid_argument: pybind11's
    // default translator sends std::out_of_range to IndexError and
    // std::invalid_argument to ValueError, and element access raising ValueError
    // would be a Python-visible regression.
    ExpectThreeWayCatch<batchlas::out_of_range>(
        [] { throw batchlas::out_of_range("Matrix indices out of range"); }, "out of range");
}

TEST(ErrorModelHierarchy, ErrorIsCaughtThreeWays) {
    ExpectThreeWayCatch<batchlas::error>(
        [] { throw batchlas::error("ILU(k): symbolic phase produced a row without diagonal"); },
        "symbolic phase");
}

TEST(ErrorModelHierarchy, UnsupportedIsCaughtThreeWays) {
    ExpectThreeWayCatch<batchlas::unsupported>(
        [] { throw batchlas::unsupported("BatchLAS: no route for gesvd<c64> on this backend"); },
        "no route");
}

TEST(ErrorModelHierarchy, DeviceErrorIsCaughtThreeWays) {
    ExpectThreeWayCatch<batchlas::device_error>(
        [] { throw batchlas::device_error("No GPU device available"); }, "GPU device");
}

TEST(ErrorModelHierarchy, WorkspaceErrorIsCaughtThreeWays) {
    ExpectThreeWayCatch<batchlas::workspace_error>(
        [] { throw batchlas::workspace_error("syev: insufficient workspace for chosen provider"); },
        "insufficient workspace");
}

TEST(ErrorModelHierarchy, ConvergenceErrorIsCaughtThreeWays) {
    ExpectThreeWayCatch<batchlas::convergence_error>(
        [] { throw batchlas::convergence_error("bdsqr: item did not converge"); },
        "did not converge");
}

TEST(ErrorModelHierarchy, InternalErrorIsCaughtThreeWays) {
    ExpectThreeWayCatch<batchlas::internal_error>(
        [] { throw batchlas::internal_error("syev: resolver returned a route with no dispatch arm"); },
        "no dispatch arm");
}

TEST(ErrorModelHierarchy, ApiMisuseIsCaughtThreeWays) {
    ExpectThreeWayCatch<batchlas::api_misuse>(
        [] { throw batchlas::api_misuse("Queue used from a thread other than its owner"); },
        "thread");
}

// The children of `error` must be catchable AS `error`: that is the coarse
// handler a consumer writes for "anything BatchLAS reports at runtime" without
// enumerating the leaves.
TEST(ErrorModelHierarchy, EveryRuntimeLeafIsCaughtAsBatchlasError) {
    auto as_error = [](auto&& thrower) {
        try { thrower(); } catch (const batchlas::error&) { return true; } catch (...) {}
        return false;
    };
    EXPECT_TRUE(as_error([] { throw batchlas::unsupported("u"); }));
    EXPECT_TRUE(as_error([] { throw batchlas::device_error("d"); }));
    EXPECT_TRUE(as_error([] { throw batchlas::workspace_error("w"); }));
    EXPECT_TRUE(as_error([] { throw batchlas::convergence_error("c"); }));
    EXPECT_TRUE(as_error([] { throw batchlas::internal_error("i"); }));
    EXPECT_TRUE(as_error([] { throw batchlas::api_misuse("m"); }));

    // ...and the argument arm must NOT be, or the discrimination this hierarchy
    // exists for is gone: "the caller passed something bad" and "the run failed"
    // would land in the same handler. This half is what stops the whole test
    // from passing under a flattened hierarchy where everything derives from
    // everything.
    EXPECT_FALSE(as_error([] { throw batchlas::invalid_argument("a"); }));
    EXPECT_FALSE(as_error([] { throw batchlas::out_of_range("o"); }));
}

// ---------------------------------------------------------------------------
// Real call sites.
//
// Each reaches a throw that exists in src/ or include/, with no kernel, no
// queue and no device allocation, so the outcome does not vary with the build
// configuration or with which device the machine has.
//
// THE REGRESSION THESE CATCH -- the proof.
//
// Take MatrixViewRejectsShortLeadingDimension below and revert its site
// (src/matrix.cc:1880) to `std::invalid_argument`, the pre-migration spelling.
// Reading the assertions in ExpectThreeWayCatch against that:
//
//   * r.specific -- the handler is `catch (const batchlas::invalid_argument&)`
//     and the object thrown is a std::invalid_argument, which is that type's
//     BASE. A handler for a derived type does not match a base object, so the
//     catch does not fire and EXPECT_TRUE(r.specific) FAILS. This is the
//     assertion that catches the regression.
//   * r.batchlas_tag -- std::invalid_argument does not carry the tag, so
//     `catch (const batchlas::exception&)` does not fire either: that assertion
//     FAILS too, and so does the message() one behind it.
//   * r.std_exception -- std::invalid_argument still derives from
//     std::exception, so this one PASSES. It is blind to the regression by
//     construction, which is exactly why it is not the only assertion here.
//
// Verified by compiling include/batchlas/error.hh with host g++ -std=c++20 and
// probing both spellings of one site:
//     workspace_error    specific=1 std=1 tag=1
//     REVERTED(runtime)  specific=0 std=1 tag=0
//     invalid_argument   specific=1 std=1 tag=1
//     REVERTED(invalid)  specific=0 std=1 tag=0
// Two of the three assertions flip; the std::exception one does not move.
// ---------------------------------------------------------------------------

TEST(ErrorModelSites, MatrixViewRejectsShortLeadingDimension) {
    // src/matrix.cc:1880 -- "leading dimension N is smaller than the row count M".
    // Pure host validation inside the MatrixView constructor: the buffer is
    // described, never read.
    std::vector<float> buffer(64, 0.0f);
    ExpectThreeWayCatch<batchlas::invalid_argument>(
        [&] {
            MatrixView<float, MatrixFormat::Dense> bad(buffer.data(), /*rows=*/8, /*cols=*/8,
                                                       /*ld=*/2, /*stride=*/64, /*batch_size=*/1);
            (void)bad;
        },
        "leading dimension");
}

TEST(ErrorModelSites, MatrixViewElementAccessOutOfRange) {
    // src/matrix.cc:1984 / :2006 -- MatrixView::at(row, col, batch), reached
    // through operator(). Two of the 5 sites that stay on std::out_of_range's
    // side of the hierarchy so Python element access keeps raising IndexError.
    std::vector<float> buffer(64, 0.0f);
    const MatrixView<float, MatrixFormat::Dense> view(buffer.data(), 8, 8, 8, 64, 1);

    ExpectThreeWayCatch<batchlas::out_of_range>(
        [&] { (void)view(/*row=*/9, /*col=*/0, /*batch=*/0); }, "out of range");
    // The batch index is checked first and by a separate statement; both arms of
    // the same site have to carry the type, not just whichever a test picked.
    ExpectThreeWayCatch<batchlas::out_of_range>(
        [&] { (void)view(/*row=*/0, /*col=*/0, /*batch=*/7); }, "out of range");
}

TEST(ErrorModelSites, BumpAllocatorOverflowIsAWorkspaceError) {
    // include/batchlas/util/mempool.hh:86 -- "Attempted to allocate N bytes from
    // a BumpAllocator with only M bytes remaining." The canonical workspace
    // shortfall: a *_buffer_size that under-counted, surfacing where the bytes
    // run out instead of as a silent heap overflow.
    //
    // tests/mempool_tests.cc:193 catches this same site as std::runtime_error,
    // which is one reason workspace_error must stay inside that arm.
    Device device = Device::default_device();
    const size_t align = BumpAllocator::alignment<int32_t>(device);
    std::vector<std::byte> storage(align * 2);
    // The pool is given `align` bytes; the request below needs 4 * align.
    BumpAllocator pool(storage.data(), align);

    ExpectThreeWayCatch<batchlas::workspace_error>(
        [&] {
            auto span = pool.allocate<int32_t>(device, align);
            (void)span;
        },
        "BumpAllocator");
}

TEST(ErrorModelSites, SizingPoolRejectsRemainingAsApiMisuse) {
    // include/batchlas/util/mempool.hh:129 -- "BumpAllocator::remaining() is not
    // available in sizing mode." Wrong-MODE API misuse, not exhaustion: a sizing
    // pool's extent is fictitious, so a callee that sized itself against
    // remaining().size() would size against a number that means nothing.
    //
    // It is deliberately NOT workspace_error. A caller told to "retry smaller"
    // on this one would loop forever: no buffer size makes a sizing-mode query
    // legal.
    BumpAllocator measuring = BumpAllocator::measuring();
    ExpectThreeWayCatch<batchlas::api_misuse>([&] { (void)measuring.remaining(); }, "sizing mode");
}

// ---------------------------------------------------------------------------
// Message text other tests match on.
//
// Four live catch sites assert on the TEXT of a message, not only its type
// (tests/matrix_tests.cc:1076, tests/options_api_tests.cc:592,
// tests/syr2k_tests.cc:189, tests/trmm_tests.cc:126). A type migration must not
// reword them. Only the one needing no queue and no device is re-checked here;
// the other three are guarded where they live.
// ---------------------------------------------------------------------------

TEST(ErrorModelSites, MessagesSurviveTheTypeChange) {
    std::vector<float> buffer(64, 0.0f);
    try {
        MatrixView<float, MatrixFormat::Dense> bad(buffer.data(), 8, 8, 2, 64, 1);
        (void)bad;
        FAIL() << "expected a throw";
    } catch (const std::exception& e) {
        const std::string what = e.what();
        // The numbers are what make this message actionable; a type-only change
        // keeps every one of them.
        EXPECT_NE(what.find("MatrixView"), std::string::npos) << what;
        EXPECT_NE(what.find("leading dimension 2"), std::string::npos) << what;
        EXPECT_NE(what.find("row count 8"), std::string::npos) << what;
    }
}

// ---------------------------------------------------------------------------
// No installed public header may terminate the host process.
//
// include/batchlas/util/miniacc.hh:675 used to call exit(0) from its --help
// handling. A library must never terminate its caller's process, and that file
// IS installed, so it shipped that behaviour to every consumer.
//
// The list below is exactly the installed set. minibench.hh, minibench_structured.hh
// and bench_structured.hh also call exit(0), and are deliberately NOT checked:
// cmake/BatchLASPackaging.cmake:70-72 EXCLUDEs all three from the install
// precisely because minibench.hh defines MINI_BENCHMARK_MAIN() -> int main().
// They are a benchmark harness, not public headers, and a --help handler ending
// its own process is what a harness should do. Adding one of them here would be
// asserting the library contract against a file the library does not ship.
//
// A source check rather than a behavioural one, because calling the site would
// end this test binary -- which IS the defect. The pattern is deliberately
// looser than "std::exit" because the offending spelling was unqualified.
// ---------------------------------------------------------------------------

#ifdef BATCHLAS_INCLUDE_DIR_PATH
namespace {

// Returns the first line of `path` whose CODE (comments stripped) calls exit(),
// or an empty string. `_exit(` and `quick_exit(` are excluded only so a future
// deliberate spelling can be argued about separately rather than silently
// matched here.
std::string FirstProcessExitCall(const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "r");
    if (!f) return {};
    char line[8192];
    std::string found;
    while (std::fgets(line, sizeof(line), f)) {
        const std::string text(line);
        const size_t comment = text.find("//");
        const std::string code = comment == std::string::npos ? text : text.substr(0, comment);
        const size_t at = code.find("exit(");
        if (at == std::string::npos) continue;
        // Reject identifier characters immediately before "exit(" so that
        // _exit(, quick_exit( and any *_exit( helper do not match.
        if (at > 0) {
            const char prev = code[at - 1];
            if (prev == '_' || std::isalnum(static_cast<unsigned char>(prev))) continue;
        }
        found = text;
        break;
    }
    std::fclose(f);
    return found;
}

}  // namespace

TEST(ErrorModelHeaders, InstalledHeadersDoNotCallExit) {
    const std::string root = BATCHLAS_INCLUDE_DIR_PATH;
    for (const char* header : {"util/miniacc.hh"}) {
        const std::string offending = FirstProcessExitCall(root + "/" + header);
        EXPECT_TRUE(offending.empty())
            << header << " calls exit() in an installed public header; a library must not "
            << "terminate its caller: " << offending;
    }
}
#endif  // BATCHLAS_INCLUDE_DIR_PATH
