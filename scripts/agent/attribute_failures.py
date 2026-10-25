#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Is a failure mine or the base's? Run the named ctest tests in this branch's tree and in a
tree of the base revision, N times each, and compare failing gtest CASE names.

    python3 scripts/agent/attribute_failures.py [--tree build-gate] [--base REV] [--reps 2] TEST...

TEST is a ctest name as the gate prints it (gesvd_tests, syev_blocked_tests_heavy, ...).
The base source is a detached worktree under ~/.cache/batchlas-attr/<sha> (kept, so the next
attribution against the same base is incremental); its build tree is seeded from --tree with
scripts/seed_build_tree.sh, so it has the same configure options, and only the executables
behind TEST are built (scripts/b: lock, -j cap). Before reaching for this, check the ledgers:
tests/known-failures.txt and tests/known-failures-host-<host>.txt already name what fails on
main here.

Verdicts per case: PRE-EXISTING (fails on base too), NEW (fails only here, every rep),
FLAKY (fails in some reps), FIXED (fails only on base). A process-level death (no gtest XML) is
the pseudo-case <test>::(process). Exit 1 when anything is NEW.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = os.environ.get("BATCHLAS_ATTR_DIR", os.path.expanduser("~/.cache/batchlas-attr"))


def run(cmd, **kw):
    print("+ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, **kw)


def out(cmd, cwd=REPO):
    return subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=cwd).stdout.strip()


def executables(tree, tests):
    """ctest name -> executable target name, from ctest's own JSON listing."""
    info = json.loads(out(["ctest", "--test-dir", tree, "--show-only=json-v1"]))
    result = {}
    for t in info.get("tests", []):
        if t["name"] in tests and t.get("command"):
            # Tests run as `sh tests/ctest_gpu_env.sh <exe> args`: the target is the first
            # argument that lives in the build tree.
            exe = next((c for c in t["command"] if c.startswith(tree + os.sep)), t["command"][0])
            result[t["name"]] = os.path.basename(exe)
    missing = set(tests) - set(result)
    if missing:
        sys.exit("attribute: not registered in %s: %s" % (tree, " ".join(sorted(missing))))
    return result


def device_selector():
    if "ONEAPI_DEVICE_SELECTOR" in os.environ:
        return os.environ["ONEAPI_DEVICE_SELECTOR"]
    try:
        devs = subprocess.run(["sycl-ls"], capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    # The gate's rule: with both CPU devices, CPU queues must not land on native_cpu.
    return "cuda:*;opencl:cpu" if "[native_cpu:" in devs and "[opencl:cpu]" in devs else None


def failing_cases(tree, tests, scratch):
    """One ctest run of TESTS in TREE -> set of failing case names."""
    shutil.rmtree(scratch, ignore_errors=True)
    gx = os.path.join(scratch, "gtest") + "/"
    os.makedirs(gx)
    junit = os.path.join(scratch, "junit.xml")
    env = dict(os.environ, GTEST_OUTPUT="xml:" + gx)
    sel = device_selector()
    if sel:
        env["ONEAPI_DEVICE_SELECTOR"] = sel
    regex = "^(%s)$" % "|".join(sorted(tests))
    run(["ctest", "--test-dir", tree, "-j", "1", "-R", regex, "--output-junit", junit],
        env=env, stdout=subprocess.DEVNULL)
    failed = set()
    for name in os.listdir(gx):
        for case in ET.parse(os.path.join(gx, name)).getroot().iter("testcase"):
            if case.find("failure") is not None:
                suite = case.get("classname")
                failed.add("%s::%s.%s" % (name[:-4], suite, case.get("name")))
    if os.path.exists(junit):
        written = {n[:-4] for n in os.listdir(gx)}
        for tc in ET.parse(junit).getroot().iter("testcase"):
            if tc.get("status") == "fail" and tc.get("name") not in written:
                failed.add("%s::(process)" % tc.get("name"))
    return failed


def base_tree(base, tree, targets):
    sha = out(["git", "rev-parse", base])
    src = os.path.join(CACHE, sha[:12])
    if not os.path.isdir(src):
        os.makedirs(CACHE, exist_ok=True)
        run(["git", "-C", REPO, "worktree", "add", "--detach", src, sha], check=True)
        if out(["git", "lfs", "version"]):
            run(["git", "-C", src, "lfs", "pull"], check=False)
    build = os.path.join(src, "build-attr")
    if not os.path.isfile(os.path.join(build, "CMakeCache.txt")):
        run([os.path.join(REPO, "scripts", "seed_build_tree.sh"), "--dst-src", src, "--allow-dirty",
             os.path.abspath(tree), build], check=True)
    rc = run([os.path.join(src, "scripts", "b") if os.path.exists(os.path.join(src, "scripts", "b"))
              else os.path.join(REPO, "scripts", "b"), "-B", build] + sorted(set(targets))).returncode
    if rc != 0:
        sys.exit("attribute: base build failed (exit %d); the base itself may not build here" % rc)
    return build


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("tests", nargs="+")
    ap.add_argument("--tree", default="build-gate", help="this branch's built tree")
    ap.add_argument("--base", default=None, help="base revision (default: merge-base with origin/main)")
    ap.add_argument("--reps", type=int, default=2)
    args = ap.parse_args()
    tree = os.path.abspath(os.path.join(REPO, args.tree))
    base = args.base or out(["git", "merge-base", "origin/main", "HEAD"])
    exes = executables(tree, set(args.tests))
    btree = base_tree(base, tree, exes.values())
    run([os.path.join(REPO, "scripts", "b"), "-B", tree] + sorted(set(exes.values())), check=True)

    scratch = os.path.join(os.environ.get("CLAUDE_JOB_DIR", "/tmp"), "tmp", "attribute")
    counts = {"head": defaultdict(int), "base": defaultdict(int)}
    for rep in range(args.reps):
        # Alternate the order so a warm-device or JIT effect cannot side with one tree.
        order = [("head", tree), ("base", btree)] if rep % 2 == 0 else [("base", btree), ("head", tree)]
        for label, t in order:
            for case in failing_cases(t, set(args.tests), os.path.join(scratch, label)):
                counts[label][case] += 1

    cases = sorted(set(counts["head"]) | set(counts["base"]))
    new = 0
    print("\nattribution vs base %s, %d rep(s) each (head fails / base fails):" % (base[:10], args.reps))
    for c in cases:
        h, b = counts["head"][c], counts["base"][c]
        if h and b:
            verdict = "PRE-EXISTING"
        elif h == args.reps:
            verdict, new = "NEW", new + 1
        elif h:
            verdict = "FLAKY"
        else:
            verdict = "FIXED"
        print("  %-12s %d/%d  %s" % (verdict, h, b, c))
    if not cases:
        print("  nothing failed in either tree")
    return 1 if new else 0


if __name__ == "__main__":
    sys.exit(main())
