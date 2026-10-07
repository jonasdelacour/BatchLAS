#!/usr/bin/env python3
"""Fail when a Markdown file lives outside docs/.

WHY THIS EXISTS
---------------
The documentation site (docs/developer/documentation.md) is the project's
database of results, rationale and design decisions. It only works if that
material is where Doxygen reads it: docs/. Plans, result write-ups and
implementation notes used to accumulate as root-level *.md files that nothing
published, nothing cross-checked and nothing linked back to, until their
conclusions were quoted from memory. This gate stops the next one.

WHAT IS ALLOWED OUTSIDE docs/
-----------------------------
    README.md        in any directory (a directory's own entry point)
    AGENTS.md        repository root only
    CLAUDE.md        repository root only
    LICENSE.md       repository root only
    .github/**       prompts, skills and templates GitHub or agents read in place

Everything else goes to docs/<area>/<topic>.md; the table of areas is in
docs/developer/documentation.md#where-things-go.

Files are what git knows about: tracked files plus untracked files that are
not ignored (a brand-new plan is the most likely offender), so build trees and
anything in .gitignore are never scanned. A path deleted from the working tree
but still in the index is skipped.

USAGE
-----
    python3 .github/ci/check_markdown_locations.py              # the gate
    python3 .github/ci/check_markdown_locations.py --tracked-only
    python3 .github/ci/check_markdown_locations.py --self-test  # arm it both ways

Exit code 0 = every Markdown file is where it belongs, 1 = offenders (or a
failed self-test), 2 = git could not list the tree.
"""

import argparse
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SUFFIXES = (".md", ".markdown")
ROOT_ONLY = ("AGENTS.md", "CLAUDE.md", "LICENSE.md")
ANYWHERE = ("README.md",)
ALLOWED_TREES = ("docs/", ".github/")


def allowed(rel):
    """True when a Markdown path may live where it is."""
    rel = rel.replace(os.sep, "/")
    if rel.startswith(ALLOWED_TREES):
        return True
    name = rel.rsplit("/", 1)[-1]
    if name in ANYWHERE:
        return True
    return "/" not in rel and name in ROOT_ONLY


def list_markdown(root, tracked_only):
    args = ["git", "-C", root, "ls-files", "-z", "--cached"]
    if not tracked_only:
        args += ["--others", "--exclude-standard"]
    try:
        out = subprocess.run(args, capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        print("check_markdown_locations: cannot list files: %s" % exc, file=sys.stderr)
        return None
    names = sorted({p.decode("utf-8", "replace") for p in out.split(b"\0") if p})
    return [n for n in names
            if n.lower().endswith(SUFFIXES) and os.path.isfile(os.path.join(root, n))]


# (path, expected allowed, why)
SELF_TEST_CASES = [
    ("docs/perf/syev.md", True, "an evidence page"),
    ("docs/ci.md", True, "a top-level docs page"),
    ("README.md", True, "the root README"),
    ("examples/consumer/README.md", True, "a directory README anywhere"),
    ("AGENTS.md", True, "root AGENTS.md"),
    ("CLAUDE.md", True, "root CLAUDE.md"),
    ("LICENSE.md", True, "root LICENSE.md"),
    (".github/skills/x/SKILL.md", True, "agent skills are read in place"),
    ("SYEVX_PLAN.md", False, "a root plan file is exactly what this gate stops"),
    ("src/NOTES.md", False, "notes beside the code"),
    ("tests/AGENTS.md", False, "AGENTS.md is allowed at the root only"),
    ("benchmarks/LICENSE.md", False, "LICENSE.md is allowed at the root only"),
    ("docsx/page.md", False, "a lookalike of docs/ is not docs/"),
    ("src/docs/page.md", False, "a nested docs/ is not the site"),
    ("notes/readme.md", False, "README matching is exact, not case-folded"),
    ("PLAN.markdown", False, "the .markdown spelling is Markdown too"),
]


def self_test():
    print("== check_markdown_locations self-test (expected vs observed)")
    bad = 0
    for rel, expect, why in SELF_TEST_CASES:
        got = allowed(rel)
        ok = got == expect
        bad += 0 if ok else 1
        print("  %-4s %-30s expected %-7s observed %-7s  %s"
              % ("ok" if ok else "FAIL", rel, "allowed" if expect else "flagged",
                 "allowed" if got else "flagged", why))
    print("check_markdown_locations self-test: %d case(s), %d failure(s)"
          % (len(SELF_TEST_CASES), bad))
    return 1 if bad else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="Fail when a Markdown file lives outside docs/.")
    ap.add_argument("--root", default=REPO, help="repository root")
    ap.add_argument("--tracked-only", action="store_true",
                    help="skip untracked (not ignored) files")
    ap.add_argument("--self-test", action="store_true",
                    help="run the path rule against fixtures and exit")
    args = ap.parse_args(argv)
    if args.self_test:
        return self_test()

    files = list_markdown(os.path.abspath(args.root), args.tracked_only)
    if files is None:
        return 2
    offenders = [f for f in files if not allowed(f)]
    for rel in offenders:
        print("%s:1: error: Markdown outside docs/. Move it to docs/<area>/<topic>.md "
              "(see docs/developer/documentation.md#where-things-go); only README.md, "
              "root AGENTS.md/CLAUDE.md/LICENSE.md and .github/ are exempt." % rel)
    print("check_markdown_locations: %d Markdown file(s), %d outside docs/"
          % (len(files), len(offenders)))
    return 1 if offenders else 0


if __name__ == "__main__":
    sys.exit(main())
