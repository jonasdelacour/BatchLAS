#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Self-test for scripts/agent/doc_section.py against a fixture page.

    python3 scripts/tests/test_doc_section.py
"""

import os
import subprocess
import sys
import tempfile

TOOL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent", "doc_section.py")

PAGE = """# Fixture page {#fixture_page}

> **Status:** test

## 1. First part

one
```bash
# not a heading
```

### 1a. Sub part

sub body

## 2. Second part {#pinned-second}

two body

## Second thoughts

tail body
"""

# (args, exit code, substrings that must appear, substrings that must not)
CASES = [
    (["§1"], 0, ["## 1. First part", "# not a heading", "sub body"], ["two body"]),
    (["1", "--no-sub"], 0, ["one"], ["sub body"]),
    (["1a"], 0, ["sub body"], ["one\n"]),
    (["pinned-second"], 0, ["two body"], ["tail body"]),
    (["second-thoughts"], 0, ["tail body"], ["two body"]),
    (["Second"], 1, ["ambiguous"], ["two body"]),
    (["nothing-here"], 1, ["no section"], []),
    (["§2", "--max", "20"], 0, ["cut at"], ["two body"]),
    ([], 0, ["[#1-first-part]", "[#pinned-second]", "[#1a-sub-part]"], ["not a heading"]),
    (["--depth", "2"], 0, ["[#1-first-part]"], ["[#1a-sub-part]"]),
]


def main():
    fails = 0
    with tempfile.TemporaryDirectory() as d:
        page = os.path.join(d, "fixture.md")
        with open(page, "w") as f:
            f.write(PAGE)
        for args, code, want, avoid in CASES:
            r = subprocess.run([sys.executable, TOOL, page] + args, capture_output=True, text=True)
            out = r.stdout + r.stderr
            bad = [w for w in want if w not in out] + ["!" + a for a in avoid if a in out]
            if r.returncode != code or bad:
                fails += 1
                print("FAIL %s: exit %d (want %d) %s\n%s" % (args, r.returncode, code, bad, out))
    print("doc_section: %d/%d cases pass" % (len(CASES) - fails, len(CASES)))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
