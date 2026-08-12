#!/usr/bin/env python3
"""Check that public headers are self-contained within the installed include tree.

`include/` is what gets installed; `src/` is not. A public header that says
`#include "../../src/queue.hh"` compiles in-tree and then fails for every
consumer with

    fatal error: '../../src/queue.hh' file not found

which is exactly what happened to include/batchlas/internal/sort.hh and
include/batchlas/util/kernel-heuristics.hh. The rule enforced here: a quoted include in a
public header must resolve to something that stays inside the include tree
(either the checked-in one or the generated one under <build>/include).

Usage:
    python3 .github/ci/check_public_headers.py [include-dir]
Exit code 0 = clean, 1 = findings.
"""

import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INCLUDE_ROOT = os.path.join(REPO, "include")

HEADER_SUFFIXES = (".h", ".hh", ".hpp", ".hxx", ".inc", ".ipp")
QUOTED_INCLUDE = re.compile(r'^\s*#\s*include\s*"([^"]+)"', re.MULTILINE)
# Generated headers live in <build>/include and are installed alongside; they
# are never reachable from the source tree, so do not try to resolve them.
GENERATED = re.compile(r"^batchlas/(backend_config\.h|device_limits\.hh|tuning_params\.hh)$")


def main(argv):
    root = os.path.abspath(argv[1]) if len(argv) > 1 else INCLUDE_ROOT
    if not os.path.isdir(root):
        print("check_public_headers: no such directory: %s" % root)
        return 1

    findings = []
    checked = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for name in sorted(filenames):
            if not name.endswith(HEADER_SUFFIXES):
                continue
            path = os.path.join(dirpath, name)
            rel = os.path.relpath(path, REPO)
            checked += 1
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                text = fh.read()
            for m in QUOTED_INCLUDE.finditer(text):
                target = m.group(1)
                line = text.count("\n", 0, m.start()) + 1
                if GENERATED.match(target):
                    continue
                resolved = os.path.normpath(os.path.join(dirpath, target))
                inside = os.path.relpath(resolved, root)
                if inside.startswith(".."):
                    findings.append((rel, line,
                                     '#include "%s" escapes the installed include tree '
                                     "(resolves to %s, which is not installed)"
                                     % (target, os.path.relpath(resolved, REPO))))
                elif not os.path.exists(resolved):
                    findings.append((rel, line,
                                     '#include "%s" does not exist under %s'
                                     % (target, os.path.relpath(root, REPO))))

    for rel, line, message in findings:
        print("%s:%d: error: %s" % (rel, line, message))
    print("check_public_headers: %d header(s) checked, %d problem(s)" % (checked, len(findings)))
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
