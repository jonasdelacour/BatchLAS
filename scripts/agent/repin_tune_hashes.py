#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Recompute the op-level kernel-hash pins in tests/tune_tests.cc
(TuneHash.OpLevelHashUnchangedByFamilyPrefixes) from the working tree.

    python3 scripts/agent/repin_tune_hashes.py [--check]

A pin changes whenever a listed kernel source changes for real, so every kernel PR and every
merge of main touches it. On a merge conflict in that map, take either side and re-run this:
never pick hashes by hand. --check exits 1 when a pin is stale, without writing.
"""

import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import sweep_to_table as stt  # noqa: E402

TEST = os.path.join(REPO, "tests", "tune_tests.cc")
PIN_MAP = re.compile(r'(const std::map<std::string, std::string> pinned\{)(.*?)(\};)', re.S)
PIN = re.compile(r'\{"(\w+)", "([0-9a-f]{8})"\}')


def main():
    check = sys.argv[1:] == ["--check"]
    text = open(TEST).read()
    m = PIN_MAP.search(text)
    if not m:
        print("repin_tune_hashes: no `pinned{...}` map in %s" % TEST)
        return 2
    body, stale = m.group(2), []
    for op, old in PIN.findall(body):
        new = stt.kernel_hash(stt.parse_kernel_block(stt.read_spec_source(op))["all"])
        if new != old:
            stale.append(op)
            body = body.replace('{"%s", "%s"}' % (op, old), '{"%s", "%s"}' % (op, new))
        print("  %-6s %s%s" % (op, new, "" if new == old else "  (was %s)" % old))
    if not stale:
        print("repin_tune_hashes: all pins current")
        return 0
    if check:
        print("repin_tune_hashes: stale pins: %s" % " ".join(stale))
        return 1
    open(TEST, "w").write(text[:m.start(2)] + body + text[m.end(2):])
    print("repin_tune_hashes: rewrote %s in %s" % (" ".join(stale), os.path.relpath(TEST, REPO)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
