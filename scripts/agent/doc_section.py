#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Read one section of a docs page instead of the whole page (docs/perf/gemm.md is 200 KB).

    python3 scripts/agent/doc_section.py PAGE [--depth N]     # outline: line, size, anchor, heading
    python3 scripts/agent/doc_section.py PAGE SEL [SEL...]    # print those sections
    python3 scripts/agent/doc_section.py --find REGEX [DIR]   # which sections mention REGEX

PAGE is a path (docs/perf/gemm.md), a page label (perf_gemm, dev_agent_guide) or a bare page
name (gemm, agent-guide). SEL is a section number (§8, 8, 2b), an anchor or {#id}
(batch-sensitivity-the-cold-protocol), or a case-insensitive heading substring; an ambiguous
substring lists its matches and prints nothing. A section runs to the next heading of the same
or higher level (--no-sub stops at any heading). Output is capped by --max (default 12000
characters) so a top-level selector cannot dump a whole page.
"""

import argparse
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
EXPLICIT_ID = re.compile(r"\s*\{#([A-Za-z0-9_.-]+)\}\s*$")


def slug(text):
    text = re.sub(r"`|\*\*|\[|\]\([^)]*\)", "", text).strip().lower()
    return re.sub(r"\s", "-", re.sub(r"[^\w\s-]", "", text))


def headings(lines):
    out, fenced = [], False
    for i, line in enumerate(lines):
        if line.lstrip().startswith("```"):
            fenced = not fenced
            continue
        m = None if fenced else HEADING.match(line)
        if not m:
            continue
        title = m.group(2)
        ident = EXPLICIT_ID.search(title)
        title = EXPLICIT_ID.sub("", title)
        num = re.match(r"(\d+[a-z]?(?:\.\d+)*)\.?\s", title)
        out.append({"line": i, "level": len(m.group(1)), "title": title,
                    "anchor": ident.group(1) if ident else slug(title),
                    "num": num.group(1) if num else None})
    return out


def doc_files(base):
    for d, _, files in os.walk(base):
        for f in sorted(files):
            if f.endswith(".md"):
                yield os.path.join(d, f)


def resolve_page(page):
    for cand in (page, os.path.join(ROOT, page)):
        if os.path.isfile(cand):
            return cand
    label, hits = "{#%s}" % page, []
    for path in doc_files(os.path.join(ROOT, "docs")):
        with open(path) as f:
            first = f.readline()
        if label in first or os.path.basename(path) in (page, page + ".md"):
            hits.append(path)
    if len(hits) == 1:
        return hits[0]
    sys.exit("doc_section: %s for page %r%s" % ("ambiguous" if hits else "no match", page,
             "".join("\n  " + os.path.relpath(h, ROOT) for h in hits)))


def section_end(hs, k, total, no_sub):
    for h in hs[k + 1:]:
        if no_sub or h["level"] <= hs[k]["level"]:
            return h["line"]
    return total


def select(hs, sel):
    key = sel.lstrip("§").strip()
    for field in ("num", "anchor"):
        exact = [k for k, h in enumerate(hs) if h[field] and h[field].lower() == key.lower()]
        if exact:
            return exact[:1]
    return [k for k, h in enumerate(hs) if sel.lower() in h["title"].lower()]


def outline(path, lines, hs, depth):
    print("%s: %d lines, %d chars" % (os.path.relpath(path, ROOT), len(lines), sum(map(len, lines))))
    for k, h in enumerate(hs):
        if h["level"] > depth:
            continue
        end = section_end(hs, k, len(lines), False)
        size = sum(len(l) + 1 for l in lines[h["line"]:end])
        print("%5d %6d  %s%s  [#%s]" % (h["line"] + 1, size, "  " * (h["level"] - 1), h["title"], h["anchor"]))


def find(pattern, base):
    rx = re.compile(pattern, re.I)
    for path in doc_files(base):
        with open(path) as f:
            lines = f.read().split("\n")
        hs, counts = headings(lines), {}
        for i, line in enumerate(lines):
            if rx.search(line):
                owner = max((k for k, h in enumerate(hs) if h["line"] <= i), default=None)
                counts[owner] = counts.get(owner, 0) + 1
        for owner, n in counts.items():
            title = hs[owner]["title"] if owner is not None else "(preamble)"
            line = hs[owner]["line"] + 1 if owner is not None else 1
            print("%s:%d  %3d hit(s)  %s" % (os.path.relpath(path, ROOT), line, n, title))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("page", nargs="?")
    ap.add_argument("sel", nargs="*")
    ap.add_argument("--find", metavar="REGEX")
    ap.add_argument("--no-sub", action="store_true")
    ap.add_argument("--max", type=int, default=12000)
    ap.add_argument("--depth", type=int, default=6, help="outline: deepest heading level shown")
    a = ap.parse_args()
    if a.find:
        find(a.find, a.page or os.path.join(ROOT, "docs"))
        return 0
    if not a.page:
        ap.error("PAGE or --find is required")
    path = resolve_page(a.page)
    with open(path) as f:
        lines = f.read().split("\n")
    hs = headings(lines)
    if not a.sel:
        outline(path, lines, hs, a.depth)
        return 0
    rc = 0
    for sel in a.sel:
        ks = select(hs, sel)
        if len(ks) != 1:
            rc = 1
            print("doc_section: %s for %r%s" % ("ambiguous" if ks else "no section", sel,
                  "".join("\n  %d  %s  [#%s]" % (hs[k]["line"] + 1, hs[k]["title"], hs[k]["anchor"]) for k in ks)),
                  file=sys.stderr)
            continue
        k = ks[0]
        text = "\n".join(lines[hs[k]["line"]:section_end(hs, k, len(lines), a.no_sub)]).rstrip()
        print("%s:%d" % (os.path.relpath(path, ROOT), hs[k]["line"] + 1))
        if len(text) > a.max:
            cut = text[:a.max].rsplit("\n", 1)[0]
            print(cut + "\n[doc_section: cut at %d of %d chars; select a subsection or pass --max]" % (len(cut), len(text)))
        else:
            print(text)
    return rc


if __name__ == "__main__":
    sys.exit(main())
