#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Merge-conflict triage without dumping whole diffs into context.

    python3 scripts/agent/conflicts.py                       # every conflicted file, per-hunk summary
    python3 scripts/agent/conflicts.py show FILE [IDX]       # one hunk (or all of FILE) in full
    python3 scripts/agent/conflicts.py take FILE IDX|all ours|theirs|both|base

IDX counts conflict hunks from 0 in file order and is recomputed on every call, so after a
`take` the remaining hunks renumber. `both` keeps ours then theirs. diff3/zdiff3 markers
(|||||||) are understood; `base` needs them. A file whose last hunk is resolved is NOT staged:
build or read it first, then `git add` it.
Generated files are never hand-merged: tests/tune_tests.cc pins -> scripts/agent/repin_tune_hashes.py;
tuned/*.txt -> scripts/sweep_to_table.py over the merged ledgers.
"""

import os
import subprocess
import sys

REPO = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True,
                      check=True).stdout.strip()
GENERATED = {"tests/tune_tests.cc": "then run scripts/agent/repin_tune_hashes.py",
             "tuned/": "regenerate with scripts/sweep_to_table.py; never hand-merge a table"}


def hunks(lines):
    """[(start, mid_base_or_None, mid, end)] line indices of each <<<<<<< ... >>>>>>> block."""
    out, i = [], 0
    while i < len(lines):
        if lines[i].startswith("<<<<<<< "):
            start, base, mid = i, None, None
            j = i + 1
            while j < len(lines) and not lines[j].startswith(">>>>>>> "):
                if lines[j].startswith("||||||| ") and mid is None:
                    base = j
                elif lines[j].startswith("=======") and mid is None:
                    mid = j
                j += 1
            out.append((start, base, mid, j))
            i = j
        i += 1
    return out


def sides(lines, h):
    start, base, mid, end = h
    ours = lines[start + 1:base if base is not None else mid]
    basel = lines[base + 1:mid] if base is not None else None
    return ours, basel, lines[mid + 1:end]


def read(path):
    with open(os.path.join(REPO, path), encoding="utf-8", errors="replace") as f:
        return f.read().split("\n")


def preview(block, n=3):
    shown = [l.strip()[:110] for l in block if l.strip()][:n]
    more = len([l for l in block if l.strip()]) - len(shown)
    return " | ".join(shown) + (" (+%d)" % more if more > 0 else "")


def note_for(path):
    for key, note in GENERATED.items():
        if path == key or path.startswith(key):
            return "  ** generated: " + note
    return ""


def summary():
    files = subprocess.run(["git", "diff", "--name-only", "--diff-filter=U"], capture_output=True,
                           text=True, cwd=REPO).stdout.split()
    if not files:
        print("no conflicted files")
        return 0
    total = 0
    for path in files:
        lines = read(path)
        hs = hunks(lines)
        total += len(hs)
        print("%s: %d hunk(s)%s" % (path, len(hs), note_for(path)))
        for k, h in enumerate(hs):
            ours, _, theirs = sides(lines, h)
            print("  [%d] line %d  ours %d / theirs %d lines" % (k, h[0] + 1, len(ours), len(theirs)))
            print("      ours:   " + (preview(ours) or "(empty)"))
            print("      theirs: " + (preview(theirs) or "(empty)"))
    print("%d file(s), %d hunk(s). Resolve: conflicts.py take FILE IDX ours|theirs|both" % (len(files), total))
    return 1


def show(path, idx=None):
    lines = read(path)
    for k, h in enumerate(hunks(lines)):
        if idx is not None and k != idx:
            continue
        print("--- hunk %d (line %d)" % (k, h[0] + 1))
        print("\n".join(lines[h[0]:h[3] + 1]))
    return 0


def take(path, which_idx, side):
    lines = read(path)
    hs = hunks(lines)
    targets = range(len(hs)) if which_idx == "all" else [int(which_idx)]
    if which_idx != "all" and not 0 <= int(which_idx) < len(hs):
        sys.exit("conflicts: %s has %d hunk(s) left; indices renumber after each take" % (path, len(hs)))
    for k in sorted(targets, reverse=True):
        h = hs[k]
        ours, basel, theirs = sides(lines, h)
        if side == "base" and basel is None:
            sys.exit("conflicts: no base section (set merge.conflictStyle=zdiff3 and re-merge)")
        repl = {"ours": ours, "theirs": theirs, "both": ours + theirs, "base": basel}[side]
        lines[h[0]:h[3] + 1] = repl
    with open(os.path.join(REPO, path), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    left = len(hunks(lines))
    print("%s: took %s for %s; %d hunk(s) left%s" % (path, side, which_idx, left,
                                                    "" if left else " (build it, then git add)"))
    return 0


def main(argv):
    if not argv:
        return summary()
    if argv[0] == "show" and len(argv) in (2, 3):
        return show(argv[1], int(argv[2]) if len(argv) == 3 else None)
    if argv[0] == "take" and len(argv) == 4 and argv[3] in ("ours", "theirs", "both", "base"):
        return take(argv[1], argv[2], argv[3])
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
