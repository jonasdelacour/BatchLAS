#!/usr/bin/env python3
"""Gate a ctest run against tests/known-failures.txt, so a not-green tree can still have CI.

THE DEFECT THIS EXISTS FOR
    `main` fails four tests and has for months (two in lanczos_tests, two in
    steqr_tests). A CI job that fails on a non-zero ctest exit can therefore
    never go green, and a job that never goes green is switched off within a
    week -- which is how this project ended up with .github/workflows/ci.yml
    running three text scanners and nothing else. Compile errors, link errors,
    ABI breakage and wrong numbers all pass that workflow. This script is what
    lets a real ctest run be a gate: it fails on any failure that is NOT in the
    ledger, and it warns when a ledger entry starts passing, so the ledger
    cannot quietly become a place to hide regressions.

    Nothing else can do this. ctest's exit code is one bit and cannot express
    "these four are expected"; `-E` would silently suppress the other 40-odd
    cases in the same two binaries; and ctest exits 0 on a SKIP, so it cannot
    tell a run that tested packaging from one whose packaging test found no
    device and gave up.

GRANULARITY: GTEST CASE, NOT CTEST TEST
    The ledger is keyed `<ctest-test>::<Suite>.<Case>`, read from per-binary
    gtest XML. Keying it on the ctest test name instead would mask every case in
    lanczos_tests and all 10 TYPED_TESTs x 4 instantiations of steqr_tests --
    about 40 cases -- in order to excuse 4. ctest's own junit is kept as a
    second layer rather than replaced, for two things it alone can see:
    consumer_package_tests is a bash script that writes no gtest XML at all, and
    a test binary that was never built appears only there.

    Get the gtest layer by exporting GTEST_OUTPUT before the run; no CMake change
    is needed, because CMake's per-test ENVIRONMENT property is additive:

        mkdir -p build/presets/dev-tests/gtest-xml
        GTEST_OUTPUT=xml:build/presets/dev-tests/gtest-xml/ \
            ctest --preset dev-tests --output-junit ctest.xml
        python3 .github/ci/compare_failures.py --junit ctest.xml \
            --gtest-dir build/presets/dev-tests/gtest-xml \
            --require consumer_package_tests --expected-tests 70

    The trailing slash on GTEST_OUTPUT is required -- it selects directory mode.

THE TWO XML SCHEMAS DISAGREE ABOUT WHICH ATTRIBUTE CARRIES THE OUTCOME
    In ctest's junit, `status` is the outcome: run / fail / notrun / disabled.
    In gtest's XML, `status` only says whether the case was attempted and
    `result` carries the outcome (completed / skipped / suppressed) -- a
    *failing* gtest case is status="run" result="completed" with a <failure>
    child. Reusing one parser's logic on the other file reads every skip as a
    pass. They are parsed separately here, on purpose.

WHAT COUNTS AS A FAILURE, AND THE JUNIT BLIND SPOT
    A ctest test whose executable does not exist emits the SAME status="notrun"
    and the SAME <skipped> child as a legitimate SKIP_RETURN_CODE=77 skip,
    differing only in the message text -- and ctest folds it into the file's
    `skipped=` count while its own exit code calls it a failure. So the summary
    attributes on the root element are not trusted here at all; every case is
    classified individually, and a <skipped message> that is not one of ctest's
    two known skip reasons is treated as a HARD FAILURE rather than defaulted to
    benign. If a CMake upgrade rewords those strings this script starts failing
    loudly, which is the correct direction to be wrong in.

WHAT THIS DOES NOT CATCH
    * A vacuous green run. GTEST_SKIP inside BatchLASTest::SetUp exits 0, and
      should_run_backend treats an unrecognised BATCHLAS_TEST_BACKEND value as
      "skip everything", so one typo in a CI env var yields 70/70 green with
      zero assertions. The SKIPPED section below reports every binary in which
      no case executed, but reporting is all it does -- CI must also simply not
      set BATCHLAS_TEST_BACKEND or BATCHLAS_TEST_FLOAT_TYPE.
    * A shrunken run, unless you pass --expected-tests. Thirty missing
      executables is thirty extra "skips" as far as the file is concerned.
    * Anything about the build. This reads reports; it never compiles.
    * Flakiness. One run in, one verdict out.

Usage:
    python3 .github/ci/compare_failures.py --junit FILE [--gtest-dir DIR]
                                           [--known FILE] [--require NAME]...
                                           [--expected-tests N]
Exit code 0 = gated clean (warnings are still possible), 1 = gate failed.
"""

import argparse
import os
import re
import sys
import xml.etree.ElementTree as ET

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_KNOWN = os.path.join(REPO, "tests", "known-failures.txt")

# The only two <skipped message> values ctest 3.28 emits for a real skip. A
# third value means something else went wrong -- "Unable to find executable"
# above all -- and must not be waved through as a skip.
CTEST_REAL_SKIP = re.compile(r"^(SKIP_RETURN_CODE=\d+|SKIP_REGULAR_EXPRESSION_MATCHED)$")

# GTEST_OUTPUT directory mode names the file after the EXECUTABLE, so the eight
# route-pinned reruns collide with their unpinned twin and gtest disambiguates
# with a counter: gemm_tests.xml and gemm_tests_1.xml. Which one holds the
# pinned run is decided by ctest execution order, not by anything declared.
RERUN_SUFFIX = re.compile(r"^(.*)_(\d+)$")

# ctest's own words for "the process died" rather than "a test case reported a
# failure". gtest writes its XML only at the end of a run, so any of these means
# there IS no case-level evidence -- and a ledger entry naming one case must not
# be allowed to excuse the whole binary dying. This repo aborts on SYCL
# assertions in exactly the two suites the ledger lists, so it is not a
# hypothetical.
_PROCESS_DEATH = ("segfault", "sigsegv", "sigabrt", "aborted", "timeout", "timed out",
                  "exception", "illegal", "bus error", "child aborted", "numerical")


def _is_process_death(detail):
    d = (detail or "").lower()
    return any(w in d for w in _PROCESS_DEATH)

PASSED, FAILED, SKIPPED, DISABLED = "passed", "failed", "skipped", "disabled"


class Result(object):
    """One outcome, from either layer, under one ledger-shaped identity."""

    def __init__(self, ident, outcome, detail="", type_param=None, source="", ran=True):
        self.ident = ident
        self.outcome = outcome
        self.detail = detail
        self.type_param = type_param
        self.source = source
        # False only for "ctest called this a skip but the test never executed"
        # -- a missing executable, above all. Such a result is a failure that the
        # ledger must NOT be allowed to absorb: a build that produced no
        # steqr_tests binary would otherwise be reported as the two steqr
        # failures we already know about, which is the exact blind spot this
        # script's header claims to close.
        self.ran = ran


def read_known(path):
    """Parse the ledger. Returns ([(ident, type_param_or_None, lineno)], findings)."""
    entries, findings = [], []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            # ';' is the field separator because it is the one punctuation mark
            # that cannot occur in a C++ template-id, and a type_param is a
            # template-id: 'SteqrConfig<double, (batchlas::Backend)6>' carries
            # commas, spaces, angle brackets, colons and parentheses.
            ident, type_param = line, None
            if ";" in line:
                ident, _, rest = line.partition(";")
                ident = ident.strip()
                key, _, value = rest.partition("=")
                if key.strip() != "type_param":
                    findings.append("%s:%d: error: unknown field %r "
                                    "(only `;type_param=<...>` is understood)"
                                    % (os.path.relpath(path, REPO), lineno, key.strip()))
                    continue
                type_param = value.strip()
            entries.append((ident, type_param, lineno))
    return entries, findings


def read_junit(path):
    """ctest --output-junit. `status` is the outcome here; the summary counts are not trusted."""
    results = []
    root = ET.parse(path).getroot()
    for tc in root.iter("testcase"):
        # name and classname are both the ctest test name; either will do.
        name = tc.get("name") or tc.get("classname") or "(unnamed)"
        status = tc.get("status") or ""
        skipped = tc.find("skipped")
        if status == "run":
            results.append(Result(name, PASSED, source=path))
        elif status == "fail":
            failure = tc.find("failure")
            detail = (failure.get("message") if failure is not None else "") or "Failed"
            results.append(Result(name, FAILED, detail, source=path))
        elif status == "disabled":
            results.append(Result(name, DISABLED, "DISABLED property", source=path))
        elif status == "notrun":
            message = (skipped.get("message") if skipped is not None else "") or ""
            if CTEST_REAL_SKIP.match(message):
                results.append(Result(name, SKIPPED, message, source=path))
            else:
                # The blind spot: a never-built executable lands here, looking
                # exactly like a skip apart from this string.
                results.append(Result(name, FAILED,
                                      "not run (%s) -- ctest records this as a skip, but the "
                                      "test did not execute" % (message or "no reason given"),
                                      source=path, ran=False))
        else:
            results.append(Result(name, FAILED, "unrecognised ctest status %r" % status,
                                  source=path))
    return results


def read_gtest_dir(path, junit_names):
    """Per-binary gtest XML. `result` is the outcome here, NOT `status`.

    Returns (results, binaries, ambiguous). `binaries` is every executable that
    wrote a file -- needed to tell "this case passed" from "this binary produced
    no XML at all", which are very different things for a ledger entry.
    """
    results, binaries, ambiguous = [], set(), set()
    names = sorted(n for n in os.listdir(path) if n.endswith(".xml"))
    stems = set(n[:-len(".xml")] for n in names)
    for name in names:
        binary = name[:-len(".xml")]
        # Fold gemm_tests_1.xml back onto gemm_tests, but remember that we did:
        # a failure reported out of one of these cannot be attributed to the
        # pinned or the unpinned run with any confidence.
        # Route-pinned reruns (tests/CMakeLists.txt registers 8 of them: gemm,
        # gemv, trsm, potrf, geqrf, orgqr, getrf, spmm) share an executable with
        # their unpinned twin, so gemm_tests_1.xml and gemm_tests.xml describe
        # DIFFERENT runs of the same cases.
        #
        # This used to fold `_1` back onto the bare name. That did not merely
        # lose attribution, it erased failures: files are read in sorted order
        # and reconcile() assigns by identity, so the passing `_1` result
        # overwrote a genuine failure from the unpinned run and the failure
        # appeared in no section at all. `_1` is the NATIVE rerun, so the
        # direction erased was a default-route (vendor) regression -- 16 of the
        # 70 ctest tests, and the half more likely to matter.
        #
        # Each file now keeps its own identity. A pinned rerun's failure is
        # reported against `gemm_tests_1::Suite.Case`, which is exactly what a
        # ledger entry would have to name to excuse it.
        m = RERUN_SUFFIX.match(binary)
        if m and (m.group(1) in junit_names or m.group(1) in stems):
            ambiguous.add(binary)
        binaries.add(binary)
        full = os.path.join(path, name)
        for tc in ET.parse(full).getroot().iter("testcase"):
            ident = "%s::%s.%s" % (binary, tc.get("classname") or "", tc.get("name") or "")
            result = tc.get("result")
            failure = tc.find("failure")
            if failure is None:
                failure = tc.find("error")
            if result == "skipped":
                outcome, detail = SKIPPED, "GTEST_SKIP"
            elif result == "suppressed":
                outcome, detail = DISABLED, "DISABLED_ prefix"
            elif failure is not None:
                head = (failure.get("message") or "").strip().splitlines()
                outcome, detail = FAILED, head[0] if head else "failed"
            elif result in (None, "completed"):
                # Older gtest omits `result`; fall back to `status`, where
                # notrun means a DISABLED_ case.
                outcome = DISABLED if tc.get("status") == "notrun" else PASSED
                detail = ""
            else:
                outcome, detail = FAILED, "unrecognised gtest result %r" % result
            results.append(Result(ident, outcome, detail, tc.get("type_param"), source=full))
    return results, binaries, ambiguous


def reconcile(junit_results, gtest_results, gtest_binaries):
    """Fold the two layers into one identity space.

    A ctest test that failed is attributed to its failing gtest CASES where we
    have them; where we do not -- a bash script, a binary that crashed before
    writing XML, a run with no --gtest-dir -- the ctest test name is itself the
    identity. That one rule is what lets consumer_package_tests and a typed
    gtest case live in the same ledger.
    """
    by_ident = {}
    failing_binaries = set()
    for r in gtest_results:
        # Worst outcome wins. Two files can still land on one identity (a rerun
        # whose stem is not matched, a hand-assembled report), and a PASSED must
        # never overwrite a FAILED -- that is how a real failure disappeared.
        prev = by_ident.get(r.ident)
        if prev is None or (prev.outcome != FAILED and r.outcome == FAILED):
            by_ident[r.ident] = r
        if r.outcome == FAILED:
            failing_binaries.add(r.ident.split("::", 1)[0])
    for r in junit_results:
        if r.outcome == FAILED and r.ident in failing_binaries:
            continue  # already accounted for, case by case
        if r.outcome in (PASSED, SKIPPED, DISABLED) and r.ident in gtest_binaries:
            continue  # the gtest layer is strictly more precise
        by_ident[r.ident] = r
    return by_ident


def section(title, rows):
    print("")
    print("=== %s (%d) ===" % (title, len(rows)))
    for text in rows:
        print("  %s" % text)
    if not rows:
        print("  (none)")


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--junit", metavar="FILE",
                        help="ctest --output-junit report")
    parser.add_argument("--gtest-dir", metavar="DIR",
                        help="directory written by GTEST_OUTPUT=xml:DIR/ (one file per binary)")
    parser.add_argument("--known", metavar="FILE", default=DEFAULT_KNOWN,
                        help="the ledger (default: tests/known-failures.txt)")
    parser.add_argument("--require", metavar="NAME", action="append", default=[],
                        help="test that MUST have actually run; repeatable, or comma-separated")
    parser.add_argument("--expected-tests", metavar="N", type=int,
                        help="fail unless the junit holds exactly N testcases")
    args = parser.parse_args(argv[1:])

    if not args.junit and not args.gtest_dir:
        print("compare_failures: nothing to read -- pass --junit and/or --gtest-dir")
        return 1

    errors = []
    junit_results = []
    if args.junit:
        if not os.path.isfile(args.junit):
            print("compare_failures: no such junit file: %s" % args.junit)
            return 1
        junit_results = read_junit(args.junit)
    junit_names = set(r.ident for r in junit_results)

    gtest_results, gtest_binaries, ambiguous = [], set(), set()
    gtest_dir_given = bool(args.gtest_dir)
    if args.gtest_dir:
        if not os.path.isdir(args.gtest_dir):
            print("compare_failures: no such gtest XML directory: %s" % args.gtest_dir)
            return 1
        gtest_results, gtest_binaries, ambiguous = read_gtest_dir(args.gtest_dir, junit_names)
        # A --gtest-dir that yields nothing is not "no gtest layer requested",
        # it is a broken one, and every ledger entry would then fall through to
        # the coarse binary-level branch. That silently downgrades the gate to
        # exactly the granularity tests/known-failures.txt exists to avoid.
        if not gtest_binaries:
            print("compare_failures: --gtest-dir %s contains no .xml files. The gate would "
                  "degrade to binary level, which the ledger forbids: two of the four listed "
                  "failures share a binary with passing cases." % args.gtest_dir)
            return 1

    ledger = os.path.relpath(args.known, REPO)
    if ledger.startswith(".."):                    # ledger outside the repo (a test fixture)
        ledger = args.known
    entries, ledger_findings = read_known(args.known)
    errors += ledger_findings
    by_ident = reconcile(junit_results, gtest_results, gtest_binaries)

    known_failures, newly_passing, listed_skipped, not_reported = [], [], [], []
    warnings, consumed = [], set()
    new_failures, skipped, vacuous = [], [], []

    for ident, type_param, lineno in entries:
        binary = ident.split("::", 1)[0]
        result = by_ident.get(ident)
        if result is not None and not result.ran:
            continue  # never executed; not eligible for the ledger, see Result.ran
        if result is None and "::" in ident and binary not in gtest_binaries:
            # No gtest XML for this binary, so a failure could only have been
            # recorded at ctest-test level. Match coarsely rather than calling a
            # long-known failure new -- but say so, every time. `ran` is what
            # keeps a missing executable out of this branch.
            #
            # TWO THINGS THIS BRANCH MUST NOT EXCUSE, both proven to pass green
            # before they were excluded:
            #
            #  * A PROCESS-LEVEL DEATH. gtest writes its XML at the END of a
            #    run, so a binary that segfaults, aborts on a SYCL assertion or
            #    hits its timeout writes nothing -- and then looks exactly like
            #    "no gtest layer for this binary". A ledger entry naming one
            #    case cannot excuse the whole binary dying; the failure is not
            #    the known one. See _PROCESS_DEATH.
            #  * A LOST GTEST LAYER. If --gtest-dir was supplied and this binary
            #    still produced no XML, the layer did not degrade by design, it
            #    broke. Coarse-matching there silently turns a case-level gate
            #    into a binary-level one, which is what this ledger exists to
            #    prevent. Handled at the call site, where args are in scope.
            coarse = by_ident.get(binary)
            if coarse is not None and coarse.outcome == FAILED and coarse.ran \
                    and _is_process_death(coarse.detail):
                new_failures.append(
                    "%s  [the ctest test %s died at process level (%s); gtest wrote no XML, so "
                    "no ledger entry can excuse it -- this is not the known case failing]"
                    % (ident, binary, coarse.detail))
                consumed.add(binary)
                continue
            if coarse is not None and coarse.outcome == FAILED and coarse.ran \
                    and gtest_dir_given:
                errors.append(
                    "compare_failures: error: %s failed but wrote no gtest XML, although "
                    "--gtest-dir was given. It crashed, timed out, or was never built. A ledger "
                    "entry cannot excuse a run that did not happen." % binary)
                consumed.add(binary)
                continue
            if coarse is not None and coarse.outcome == FAILED and coarse.ran:
                consumed.add(binary)
                known_failures.append("%s  [COARSE: matched the ctest test %s; no gtest XML for "
                                      "this binary, so the specific case is unverified]"
                                      % (ident, binary))
                continue
        if result is None:
            not_reported.append("%s  (%s:%d)" % (ident, ledger, lineno))
            continue
        # The typed-test index trap: honouring an index-keyed entry under a
        # different configure preset would suppress a different test entirely.
        # Deliberately NOT consumed, so the outcome it failed to excuse still
        # shows up below under its own heading rather than vanishing.
        if type_param is not None and result.type_param is not None \
                and type_param != result.type_param:
            errors.append("%s:%d: error: %s is pinned to type_param=%r but "
                          "this run reports type_param=%r -- the typed-test index has shifted, so "
                          "this entry names a different test"
                          % (ledger, lineno, ident, type_param, result.type_param))
            continue
        consumed.add(ident)
        if result.outcome == FAILED:
            known_failures.append("%s%s" % (ident, "  (%s)" % result.detail if result.detail else ""))
            if type_param is None and result.type_param is not None:
                warnings.append("%s:%d: %s is a typed test with no "
                                "`;type_param=` pin -- paste in the observed spelling: "
                                ";type_param=%s" % (ledger, lineno, ident, result.type_param))
        elif result.outcome == PASSED:
            newly_passing.append("%s  (%s:%d)" % (ident, ledger, lineno))
        else:
            listed_skipped.append("%s  [%s: %s] -- listed as a known failure but did not run, so "
                                  "it is neither confirmed nor fixed"
                                  % (ident, result.outcome, result.detail))

    # Everything the ledger did not account for. `consumed` holds the identities
    # an entry matched -- including a coarse ctest-level match, which is why the
    # parent test name can be in there alongside gtest case names.
    for ident in sorted(by_ident):
        result = by_ident[ident]
        if ident in consumed:
            continue
        if result.outcome == FAILED:
            note = ("  [from a route-pinned rerun pair; see the header]"
                    if ident.split("::", 1)[0] in ambiguous else "")
            new_failures.append("%s  (%s)%s" % (ident, result.detail or "failed", note))
        elif result.outcome in (SKIPPED, DISABLED):
            skipped.append("%s  [%s: %s]" % (ident, result.outcome, result.detail))

    # A binary in which nothing executed is the shape of the vacuous-green
    # failure mode: a BATCHLAS_TEST_BACKEND typo skips every case and exits 0.
    for binary in sorted(gtest_binaries):
        cases = [r for r in gtest_results if r.ident.split("::", 1)[0] == binary]
        if cases and not [r for r in cases if r.outcome in (PASSED, FAILED)]:
            vacuous.append("%s: all %d case(s) skipped -- this binary asserted nothing"
                           % (binary, len(cases)))

    # --require is the answer to "ctest exits 0 on a skip". A GPU job whose
    # packaging test silently skipped has not tested packaging.
    required = []
    for item in args.require:
        required += [n.strip() for n in item.split(",") if n.strip()]
    for name in required:
        result = by_ident.get(name)
        ran = [r for r in gtest_results
               if r.ident.split("::", 1)[0] == name and r.outcome in (PASSED, FAILED)]
        if result is None and not ran:
            errors.append("compare_failures: required test %r is absent from this report -- it was "
                          "never registered or never reached, so the run proves less than it "
                          "appears to" % name)
        elif not ran and result is not None and result.outcome in (SKIPPED, DISABLED):
            errors.append("compare_failures: required test %r %s (%s) -- the runner lacked a "
                          "prerequisite, so the run proves less than it appears to; this run must "
                          "not be reported as having tested %s"
                          % (name, result.outcome, result.detail, name))

    # Counted off the junit only: the gtest layer has no idea how many ctest
    # tests were supposed to exist, and a binary that was never built writes no
    # gtest XML to be missed.
    if args.expected_tests is not None and args.junit and len(junit_results) != args.expected_tests:
        errors.append("compare_failures: junit holds %d test(s), expected %d -- the run was "
                      "truncated or the suite shrank"
                      % (len(junit_results), args.expected_tests))

    # Diff-shaped, NEW FAILURES first, and every section prints even when empty:
    # a human scanning a CI log needs "0 new failures" to be visible, not
    # inferred from the absence of a heading.
    section("NEW FAILURES", new_failures)
    section("KNOWN FAILURES", known_failures)
    section("NEWLY PASSING", newly_passing)
    section("SKIPPED", skipped + listed_skipped + vacuous)
    if not_reported:
        section("LISTED BUT NOT IN THIS REPORT", not_reported)
        print("  (expected when the run was filtered, e.g. `ctest -LE slow`; also what a GPU-less "
              "build looks like, since lanczos_tests then compiles to zero tests)")

    print("")
    for text in warnings:
        print("compare_failures: warning: %s" % text)
    if newly_passing:
        print("compare_failures: %d listed failure(s) now PASS. Someone fixed them. Delete those "
              "lines from tests/known-failures.txt -- while they stay, the ledger hides the "
              "regression that re-breaks them.\n"
              "  This FAILS the gate rather than warning, so that deleting the line is "
              "unavoidable rather than optional. A warning inside a green tick is how the "
              "SytrdBlockedLatrdGridCudaTest entry went stale in the first place: the GPU job "
              "uploads its report only on failure, so on a green run the warning exists nowhere "
              "but in scrollback." % len(newly_passing))
    for text in errors:
        print(text)

    print("compare_failures: %d new failure(s), %d known, %d newly passing, %d skipped"
          % (len(new_failures), len(known_failures), len(newly_passing),
             len(skipped) + len(listed_skipped)))
    return 1 if (new_failures or errors or newly_passing) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
