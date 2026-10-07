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
    README.md          in any directory but the root (a directory's own entry point)
    .claude/CLAUDE.md  agent memory; it only imports docs/developer/agent-guide.md
    .github/**         the project README, skills and templates read in place

Nothing at all at the repository root: the project README is .github/README.md,
which GitHub renders on the repository page.

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
ANYWHERE_BUT_ROOT = ("README.md",)
ALLOWED_FILES = (".claude/CLAUDE.md",)
ALLOWED_TREES = ("docs/", ".github/")


def allowed(rel):
    """True when a Markdown path may live where it is."""
    rel = rel.replace(os.sep, "/")
    if rel.startswith(ALLOWED_TREES) or rel in ALLOWED_FILES:
        return True
    return "/" in rel and rel.rsplit("/", 1)[-1] in ANYWHERE_BUT_ROOT


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
    (".github/README.md", True, "the project README GitHub renders"),
    ("examples/consumer/README.md", True, "a directory README"),
    (".claude/CLAUDE.md", True, "agent memory that imports the guide"),
    (".github/skills/x/SKILL.md", True, "agent skills are read in place"),
    ("README.md", False, "no Markdown at the root, README included"),
    ("AGENTS.md", False, "the agent guide lives in docs/developer/"),
    ("CLAUDE.md", False, "root CLAUDE.md: use .claude/CLAUDE.md"),
    ("LICENSE.md", False, "no Markdown at the root"),
    (".claude/notes.md", False, "only .claude/CLAUDE.md is exempt"),
    ("SYEVX_PLAN.md", False, "a root plan file is exactly what this gate stops"),
    ("src/NOTES.md", False, "notes beside the code"),
    ("tests/AGENTS.md", False, "an AGENTS.md is not exempt anywhere"),
    ("tests/CLAUDE.md", False, "only .claude/CLAUDE.md is exempt"),
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
              "(see docs/developer/documentation.md#where-things-go); only a non-root "
              "README.md, .claude/CLAUDE.md and .github/ are exempt." % rel)
    print("check_markdown_locations: %d Markdown file(s), %d outside docs/"
          % (len(files), len(offenders)))
    return 1 if offenders else 0


if __name__ == "__main__":
    sys.exit(main())
