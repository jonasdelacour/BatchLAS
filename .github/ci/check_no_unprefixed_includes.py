#!/usr/bin/env python3
"""Check that nothing spells a BatchLAS header without the `batchlas/` prefix.

Every public header lives under include/batchlas/ and installs to
<prefix>/include/batchlas/, so `batchlas` is the only name BatchLAS claims in a
consumer's include root. That guarantee is only worth something if no header we
ship spells its own includes `<blas/...>`, `<util/...>` or `<internal/...>`: an
installed header that does is resolved against the consumer's include path
first, so a consumer that owns a file named util/workspace.hh silently wins --
or, worse, gets our header where it wanted its own.

Nothing else catches a regression here quickly. The in-tree build cannot: it
puts -I<source>/include on the line, and after the move an unprefixed spelling
is simply a missing file, which is fine until someone re-adds include/util/.
The only other guard is consumer_package_tests, which is `slow`-labelled and
needs a full install.

Usage:
    python3 .github/ci/check_no_unprefixed_includes.py [repo-root]
Exit code 0 = clean, 1 = findings.
"""

import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SEARCH_DIRS = ("include", "src", "tests", "benchmarks", "python", "examples")
SUFFIXES = (".h", ".hh", ".hpp", ".hxx", ".inc", ".ipp", ".c", ".cc", ".cpp", ".cxx", ".cu")

BAD_INCLUDE = re.compile(r'^\s*#\s*include\s*<(blas|util|internal)/')

# examples/consumer/decoy_include/ deliberately contains headers named
# blas/enums.hh, util/workspace.hh and internal/ormqr_blocked.hh. They exist to
# be put ahead of the installed tree on the include path and to #error if they
# are ever reached; they are not compiled by any normal build.
SKIP_DIRS = (os.path.join("examples", "consumer", "decoy_include"),)


def main(argv):
    root = os.path.abspath(argv[1]) if len(argv) > 1 else REPO
    findings = []
    checked = 0

    for top in SEARCH_DIRS:
        base = os.path.join(root, top)
        if not os.path.isdir(base):
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d != "__pycache__"]
            rel_dir = os.path.relpath(dirpath, root)
            if any(rel_dir == s or rel_dir.startswith(s + os.sep) for s in SKIP_DIRS):
                dirnames[:] = []
                continue
            for name in sorted(filenames):
                if not name.endswith(SUFFIXES):
                    continue
                path = os.path.join(dirpath, name)
                checked += 1
                with open(path, "r", encoding="utf-8", errors="replace") as fh:
                    for lineno, line in enumerate(fh, 1):
                        if BAD_INCLUDE.match(line):
                            findings.append((os.path.relpath(path, root), lineno, line.strip()))

    for rel, lineno, text in findings:
        print("%s:%d: error: unprefixed include `%s` -- spell it <batchlas/...>"
              % (rel, lineno, text))
    print("check_no_unprefixed_includes: %d file(s) checked, %d problem(s)"
          % (checked, len(findings)))
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
