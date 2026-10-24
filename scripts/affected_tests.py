#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Map a diff to the test binaries it can break: build targets and an anchored ctest regex.

    scripts/affected_tests.py                      # summary for merge-base(HEAD, origin/main)
    scripts/affected_tests.py --print-build-cmd --print-ctest-cmd --build-dir build
    scripts/affected_tests.py --paths src/ops/trmm/trmm.cc   # explicit paths instead of git
    scripts/affected_tests.py --json

Changed paths come from `git diff --name-only <base>` (committed, staged and unstaged) plus
untracked files. src/ops/<op>/** and include/batchlas/blas/functions/<op>.hh map to the op's
test binaries (tests named <op>_*, plus aliases below), then to every op whose sources call
it, transitively: sub-ops call the public gemm, so a gemm change also runs potrf, geqrf,
syev and the rest. Shared code (Queue, Matrix, mempool, sg_partition, include/batchlas/util,
src/select, src/sycl, src/util, cmake/, ...) falls back to the full build and the quick tier
`ctest -LE 'slow|heavy'`. A changed header (src/ops/**, include/**, tests/*.hh) also selects
every test whose .cc includes it, directly or through a tests/*.hh. The regex names the quick
entries and their route-pinned `_native` reruns; --heavy adds the `<name>_heavy` case tiers
(tests/CMakeLists.txt, BATCHLAS_HEAVY_TESTS). Builds go through scripts/b.
This is an iteration aid: the pre-push gate is still the full run (agent guide section 8).
"""

import argparse
import json
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS_CMAKE = os.path.join(REPO, "tests", "CMakeLists.txt")

# Op directories whose tests are not named <op>_*.
ALIASES = {
    "level3": ["hemm", "herk", "her2k"],
    "hemm": ["hermitian_candidates_tests"],
    "herk": ["hermitian_candidates_tests"],
    "her2k": ["hermitian_candidates_tests"],
    "inv": ["inverse_tests"],
    "gesvd": ["gesvdj_cta_tests", "bdsqr", "bdsdc"],
    "tridiag": ["stebz", "stein", "steqr", "stedc"],
    "lascl": ["backend_dispatch_tests"],
}
SHARED_PREFIXES = (
    "cmake/", "CMakeLists.txt", "CMakePresets.json", "src/select/", "src/sycl/", "src/util/",
    "src/backends/", "src/kernels/", "src/verify/", "src/CMakeLists.txt", "src/ops/CMakeLists.txt",
    "include/batchlas/util/", "include/batchlas/verify/", "include/batchlas/blas/device",
    "tests/test_utils.hh", "tests/CMakeLists.txt", "tests/cmake/",
)
IGNORED_PREFIXES = ("docs/", "benchmarks/", "python/", "examples/", "evaluation/", ".github/",
                    ".claude/", "scripts/", "plotting/")


def cmake_list(text, name):
    """Words of `set(<name> ...)` in a CMakeLists, comments stripped (first occurrence)."""
    m = re.search(r"^\s*set\(\s*%s\b(.*?)\)" % re.escape(name),
                  "\n".join(line.split("#", 1)[0] for line in text.splitlines()), re.M | re.S)
    return m.group(1).split() if m else []


def read_registry():
    text = open(TESTS_CMAKE, encoding="utf-8").read()
    targets = cmake_list(text, "TEST_TARGETS") + cmake_list(text, "CPU_SYCL_DEPENDENT_TESTS")
    pinned = [w for w in cmake_list(text, "BATCHLAS_ROUTE_PINNED_TESTS") if not w.startswith('"')]
    return {"targets": sorted(set(targets)), "heavy": set(cmake_list(text, "BATCHLAS_HEAVY_TESTS")),
            "pinned": set(w for w in pinned if w.endswith("_tests")),
            "slow": set(cmake_list(text, "BATCHLAS_SLOW_TESTS"))}


def git(*args):
    out = subprocess.run(["git", "-C", REPO] + list(args), capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit("affected_tests: git %s failed: %s" % (" ".join(args), out.stderr.strip()))
    return out.stdout


def changed_paths(base):
    if base is None:
        base = git("merge-base", "HEAD", "origin/main").strip()
    paths = set(git("diff", "--name-only", base).split())
    paths |= set(git("ls-files", "--others", "--exclude-standard").split())
    return base, sorted(paths)


def op_dirs():
    root = os.path.join(REPO, "src", "ops")
    return sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))


def read_sources(root):
    chunks = []
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.endswith((".cc", ".hh")):
                chunks.append(open(os.path.join(dirpath, f), encoding="utf-8", errors="replace").read())
    return "\n".join(chunks)


def callers_graph(ops):
    """unit -> units whose sources call it by name (over-inclusive on purpose).

    A unit is an op directory or one of its sub-op directories (syev/sytrd, gesvd/bdsqr):
    the sub-op unit lets a gemm change reach sytrd_blocked_tests, not only syev_*.
    """
    texts = {}
    for op in ops:
        root = os.path.join(REPO, "src", "ops", op)
        texts[op] = read_sources(root)
        for sub in os.listdir(root):
            if sub != "kernels" and os.path.isdir(os.path.join(root, sub)):
                texts.setdefault(sub, read_sources(os.path.join(root, sub)))
    graph = {unit: set() for unit in texts}
    for unit in texts:
        call = re.compile(r"\b%s\s*(<[^;{}()]*>)?\s*\(" % re.escape(unit))
        for other, text in texts.items():
            if other != unit and call.search(text):
                graph[unit].add(other)
    return graph


def tests_for(name, targets, seen=None):
    seen = set() if seen is None else seen
    if name in seen:
        return set()
    seen.add(name)
    found = set()
    for word in [name] + ALIASES.get(name, []):
        if word in targets:
            found.add(word)
        elif word != name and word in ALIASES:
            found |= tests_for(word, targets, seen)
        else:
            found |= set(t for t in targets if t.startswith(word + "_"))
    return found


def test_target(stem, targets):
    stem = "verify_tests" if stem == "verify_residuals_tests" else stem
    return stem if stem in targets else None


def include_spellings(path):
    """Root-based spellings of PATH in an #include (agent guide section 12)."""
    names = {path}
    if path.startswith("include/"):
        names.add(path[len("include/"):])
    return names


def test_includers(headers, targets):
    """Registered tests whose tests/*.cc include one of HEADERS, also through tests/*.hh."""
    tdir = os.path.join(REPO, "tests")
    texts = {}
    for f in os.listdir(tdir):
        if f.endswith((".cc", ".hh")):
            with open(os.path.join(tdir, f), encoding="utf-8", errors="replace") as fh:
                texts["tests/" + f] = fh.read()
    want, todo = set(), list(headers)
    while todo:
        h = todo.pop()
        if h in want:
            continue
        want.add(h)
        pat = re.compile(r"^\s*#\s*include\s*[<\"](%s)[>\"]"
                         % "|".join(re.escape(n) for n in include_spellings(h)), re.M)
        for name, text in texts.items():
            if name.endswith(".hh") and name not in want and pat.search(text):
                todo.append(name)
    found = set()
    for name, text in texts.items():
        if not name.endswith(".cc"):
            continue
        for h in want:
            pat = r"^\s*#\s*include\s*[<\"](%s)[>\"]" % "|".join(re.escape(n) for n in include_spellings(h))
            if re.search(pat, text, re.M):
                t = test_target(os.path.basename(name)[:-len(".cc")], targets)
                if t:
                    found.add(t)
                break
    return found


def classify(path, ops, targets):
    """('shared', why) | ('ops', {op or sub-op names}) | ('tests', {binaries}) | ('ignored', why)."""
    if path.startswith(IGNORED_PREFIXES) or path.endswith(".md"):
        return "ignored", "not compiled into a test"
    if path.startswith(SHARED_PREFIXES):
        return "shared", "shared code"
    m = re.match(r"src/ops/([^/]+)/(?:([^/]+)/)?", path)
    if m and m.group(1) in ops:
        names = {m.group(1)}
        if m.group(2) and m.group(2) not in ("kernels",) and os.path.isdir(
                os.path.join(REPO, "src", "ops", m.group(1), m.group(2))):
            names.add(m.group(2))
        return "ops", names
    m = re.match(r"include/batchlas/blas/functions/([^/.]+)\.hh$", path)
    if m:
        return "ops", {m.group(1)}
    if path == "include/batchlas/internal/ormqr_blocked.hh":
        return "ops", {"ormqr"}
    m = re.match(r"tests/([^/]+)\.cc$", path)
    if m:
        stem = test_target(m.group(1), targets)
        return ("tests", {stem}) if stem else ("ignored", "not a registered test")
    if re.match(r"tests/[^/]+\.hh$", path):
        return "tests", test_includers([path], targets)
    m = re.match(r"tuned/([a-z0-9_]+)\.", path)
    if m:
        return "ops", {m.group(1), "tuned_tables"}
    if path.startswith("tools/tune/"):
        return "tests", {"tune_tests", "tune_race_gpu_tests"} & set(targets)
    if path.startswith(("src/", "include/")):
        return "shared", "unmapped library path"
    return "ignored", "outside src/, include/, tests/ and tuned/"


def affected(paths, heavy):
    reg = read_registry()
    targets, ops = reg["targets"], op_dirs()
    graph = callers_graph(ops)
    shared, ignored, binaries, touched_ops = [], [], set(), set()
    for path in paths:
        kind, what = classify(path, ops, targets)
        if kind == "shared":
            shared.append(path)
        elif kind == "ignored":
            ignored.append(path)
        elif kind == "tests":
            binaries |= set(what)
        else:
            touched_ops |= set(what)
            if path.endswith((".hh", ".h")):
                binaries |= test_includers([path], targets)
    closure, todo = set(touched_ops), list(touched_ops)
    while todo:
        for caller in graph.get(todo.pop(), ()):
            if caller not in closure:
                closure.add(caller)
                todo.append(caller)
    for name in touched_ops:
        binaries |= tests_for(name, targets)
    # Callers reached only through the closure: their quick entries, not the slow binaries.
    slow_skipped = set()
    for name in closure - touched_ops:
        found = tests_for(name, targets) - binaries
        slow_skipped |= found & reg["slow"]
        binaries |= found - reg["slow"]
    binaries &= set(targets)
    entries = set(binaries)
    entries |= set(b + "_native" for b in binaries if b in reg["pinned"])
    if heavy:
        entries |= set(b + "_heavy" for b in binaries if b in reg["heavy"])
    return {"shared": shared, "ignored": ignored, "ops": sorted(touched_ops),
            "ops_with_callers": sorted(closure), "targets": sorted(binaries),
            "slow_callers_skipped": sorted(slow_skipped),
            "ctest_entries": sorted(entries),
            "ctest_regex": "^(%s)$" % "|".join(sorted(entries)) if entries else ""}


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", help="git ref to diff against (default: merge-base HEAD origin/main)")
    ap.add_argument("--paths", nargs="+", help="use these repo-relative paths instead of git")
    ap.add_argument("--build-dir", default="build")
    ap.add_argument("--heavy", action="store_true", help="also run the <name>_heavy case tiers")
    ap.add_argument("--print-build-cmd", action="store_true")
    ap.add_argument("--print-ctest-cmd", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    base, paths = (None, sorted(args.paths)) if args.paths else changed_paths(args.base)
    res = affected(paths, args.heavy)
    res["base"], res["paths"] = base, paths
    b = args.build_dir
    if res["shared"]:
        res["build_cmd"] = "BATCHLAS_FULL_BUILD=1 scripts/b -B %s" % b
        res["ctest_cmd"] = "ctest --test-dir %s -LE %s" % (b, "slow" if args.heavy else "'slow|heavy'")
    elif res["targets"]:
        res["build_cmd"] = "scripts/b -B %s %s" % (b, " ".join(res["targets"]))
        res["ctest_cmd"] = "ctest --test-dir %s -R '%s' --output-on-failure" % (b, res["ctest_regex"])
    else:
        res["build_cmd"] = res["ctest_cmd"] = ""
    if args.json:
        print(json.dumps(res, indent=2))
        return 0
    if args.print_build_cmd or args.print_ctest_cmd:
        for flag, key in ((args.print_build_cmd, "build_cmd"), (args.print_ctest_cmd, "ctest_cmd")):
            if flag and res[key]:
                print(res[key])
        return 0
    print("base: %s, %d changed path(s)" % (base or "(--paths)", len(paths)))
    if res["shared"]:
        print("shared code changed (%s%s): default target and the quick tier"
              % (", ".join(res["shared"][:5]), ", ..." if len(res["shared"]) > 5 else ""))
    elif not res["targets"]:
        print("no test binary is affected")
    else:
        print("ops: %s (with callers: %s)" % (" ".join(res["ops"]) or "-", " ".join(res["ops_with_callers"]) or "-"))
        print("targets: %s" % " ".join(res["targets"]))
        print("ctest -R: %s" % res["ctest_regex"])
        if res["slow_callers_skipped"]:
            print("slow caller binaries left out: %s" % " ".join(res["slow_callers_skipped"]))
    if res["build_cmd"]:
        print(res["build_cmd"])
        print(res["ctest_cmd"])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
