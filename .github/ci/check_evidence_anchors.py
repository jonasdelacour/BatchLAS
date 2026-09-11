#!/usr/bin/env python3
"""Check that every `evidence: docs/...#anchor` pointer resolves to a real heading.

The house rule is that measured numbers live in docs/perf/<op>.md and code
comments carry correctness rationale plus a pointer at the evidence. That
pointer is the only thing tying a load-bearing claim to the measurement behind
it, and it is exactly the kind of thing that rots silently: renaming a heading
while editing a doc breaks every citation of it, and nothing in a build, a test
or a review reliably notices. A dangling pointer is worse than no pointer --
it reads as "this was measured" while leading nowhere.

So this resolves each one. A reference names a file, optionally with a
`#anchor`; the file must exist, and the anchor must match the GitHub slug of
some heading in it. The slug rules are GitHub's: lowercase, drop everything
that is not a letter, digit, space, hyphen or underscore, then spaces to
hyphens, with `-1`, `-2` ... appended to repeats in document order.

Usage:
    python3 .github/ci/check_evidence_anchors.py [repo-root]
    python3 .github/ci/check_evidence_anchors.py --self-test
Exit code 0 = clean, 1 = findings.
"""

import os
import re
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Trailing punctuation is stripped separately: a citation often ends a sentence
# or sits inside parentheses, and `.` is also legal inside the path.
REFERENCE = re.compile(r"evidence:\s*(docs/[^\s,;)\]\"'`]+)")

SKIP_DIRS = {".git", "build", "__pycache__", "node_modules", ".venv", "venv"}

# The documentation of the convention itself, which spells the shape rather
# than citing a page. Anything matching this is a template, not a pointer.
PLACEHOLDER = re.compile(r"<[^>]+>")

# Read-as-text extensions plus the extensionless names we actually carry.
TEXT_SUFFIXES = (
    ".h", ".hh", ".hpp", ".hxx", ".inc", ".ipp", ".c", ".cc", ".cpp", ".cxx",
    ".cu", ".md", ".py", ".sh", ".cmake", ".in", ".txt", ".yml", ".yaml",
    ".json", ".toml", ".cfg",
)
TEXT_NAMES = ("CMakeLists.txt", "AGENTS.md", "Makefile")

MAX_BYTES = 4 * 1024 * 1024

# This file's own docstring and self-test fixtures contain deliberately dangling
# pointers -- they are the arming plants. Scanning them would make the checker
# permanently red, so it skips itself; nothing else is exempt.
SELF = os.path.relpath(os.path.abspath(__file__), REPO)

FENCE = re.compile(r"^\s*(```|~~~)")
HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")


def slugify(text):
    """GitHub's heading-to-anchor transform, less the duplicate suffix."""
    # Inline code, emphasis and links contribute their text, not their markup.
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = text.replace("*", "").replace("_", "_")
    text = text.lower()
    text = "".join(ch for ch in text if ch.isalnum() or ch in " -_")
    return text.strip().replace(" ", "-")


def heading_anchors(path):
    """Every anchor a GitHub-rendered view of `path` would expose."""
    anchors = set()
    seen = {}
    fence = None
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            marker = FENCE.match(line)
            if marker:
                if fence is None:
                    fence = marker.group(1)
                elif line.strip().startswith(fence):
                    fence = None
                continue
            if fence is not None:
                continue
            match = HEADING.match(line)
            if not match:
                continue
            base = slugify(match.group(2))
            if not base:
                continue
            count = seen.get(base, 0)
            seen[base] = count + 1
            anchors.add(base if count == 0 else "%s-%d" % (base, count))
    return anchors


def is_text(name):
    return name.endswith(TEXT_SUFFIXES) or name in TEXT_NAMES


def check(root):
    findings = []
    refs = 0
    anchor_cache = {}

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for name in sorted(filenames):
            if not is_text(name):
                continue
            path = os.path.join(dirpath, name)
            try:
                if os.path.getsize(path) > MAX_BYTES:
                    continue
                with open(path, "r", encoding="utf-8", errors="replace") as fh:
                    lines = fh.readlines()
            except OSError:
                continue
            rel = os.path.relpath(path, root)
            if rel == SELF:
                continue
            for lineno, line in enumerate(lines, 1):
                for target in REFERENCE.findall(line):
                    target = target.rstrip(".,;:")
                    if PLACEHOLDER.search(target):
                        continue
                    refs += 1
                    doc, _, anchor = target.partition("#")
                    doc_path = os.path.join(root, doc)
                    if not os.path.isfile(doc_path):
                        findings.append((rel, lineno, target, "no such file"))
                        continue
                    if not anchor:
                        continue
                    if doc not in anchor_cache:
                        anchor_cache[doc] = heading_anchors(doc_path)
                    if anchor not in anchor_cache[doc]:
                        findings.append((rel, lineno, target, "no heading in %s has that anchor" % doc))
    return refs, findings


SELF_TEST_DOC = """# A page

## The launch shape: 64 work-items, not 128

## The work-group A/B

```
### Inside a fence
```
"""

SELF_TEST_SRC = """// good: evidence: docs/perf/p.md#the-launch-shape-64-work-items-not-128
// good: evidence: docs/perf/p.md#the-work-group-ab
// good, no anchor: evidence: docs/perf/p.md
// template, skipped: evidence: docs/perf/<page>.md#<anchor>
// bad anchor: evidence: docs/perf/p.md#the-tiny-tier-launch-shape
// bad file: evidence: docs/perf/missing.md#a-page
// fenced heading is not an anchor: evidence: docs/perf/p.md#inside-a-fence
"""


def self_test():
    """The checker must flag the bad pointers and only the bad pointers.

    A scan that reports nothing is indistinguishable from a scan that cannot
    read the tree, so this is armed in both directions: it asserts the three
    plants are found AND that the four good references are not.
    """
    with tempfile.TemporaryDirectory() as root:
        os.makedirs(os.path.join(root, "docs", "perf"))
        os.makedirs(os.path.join(root, "src"))
        with open(os.path.join(root, "docs", "perf", "p.md"), "w") as fh:
            fh.write(SELF_TEST_DOC)
        with open(os.path.join(root, "src", "a.cc"), "w") as fh:
            fh.write(SELF_TEST_SRC)
        refs, findings = check(root)
        got = sorted(f[2] for f in findings)
        want = sorted([
            "docs/perf/missing.md#a-page",
            "docs/perf/p.md#the-tiny-tier-launch-shape",
            "docs/perf/p.md#inside-a-fence",
        ])
        if refs != 6:
            print("check_evidence_anchors --self-test: FAILED, counted %d references, expected 6" % refs)
            return 1
        if got != want:
            print("check_evidence_anchors --self-test: FAILED")
            print("  flagged:  %s" % got)
            print("  expected: %s" % want)
            return 1
    print("check_evidence_anchors --self-test: ok, 6 references, 3 plants caught, 3 good ones passed")
    return 0


def main(argv):
    if "--self-test" in argv[1:]:
        return self_test()
    root = os.path.abspath(argv[1]) if len(argv) > 1 else REPO
    refs, findings = check(root)
    for rel, lineno, target, why in findings:
        print("%s:%d: error: dangling evidence pointer `%s` -- %s" % (rel, lineno, target, why))
    print("check_evidence_anchors: %d reference(s) resolved, %d dangling" % (refs, len(findings)))
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
