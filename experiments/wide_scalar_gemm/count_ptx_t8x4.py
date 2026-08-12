#!/usr/bin/env python3
"""Per-entry instruction census of the PTX emitted for tile-128x64-k8-t8x4.

Checks the two things that cannot be verified without a GPU:
  * the fragment loads from shared really are vectorized (property 2 and 4),
  * the complex inner loop really is 4 fma per MAC with no __mulsc3 call
    and no isnan branch (the Annex-G trap).
"""
import re
import sys
import collections

PATTERNS = [
    ("ld.shared.v4", r"ld\.shared\.v4"),
    ("ld.shared.v2", r"ld\.shared\.v2"),
    ("ld.shared.scalar", r"ld\.shared\.(?!v[24])"),
    ("st.shared.v4", r"st\.shared\.v4"),
    ("st.shared.v2", r"st\.shared\.v2"),
    ("st.shared.scalar", r"st\.shared\.(?!v[24])"),
    ("fma.rn.f32", r"fma\.rn\.f32"),
    ("fma.rn.f64", r"fma\.rn\.f64"),
    ("ld.global.v4", r"ld\.global\.(nc\.)?v4"),
    ("ld.global.v2", r"ld\.global\.(nc\.)?v2"),
    ("st.global.v4", r"st\.global\.v4"),
    ("st.global.v2", r"st\.global\.v2"),
    ("__mulsc3/__muldc3", r"__mul[sd]c3"),
    ("call", r"^\s*call"),
]


def census(path):
    text = open(path).read()
    # Split into entry bodies.
    parts = re.split(r"\.(?:visible\s+)?entry\s+", text)
    out = collections.OrderedDict()
    for part in parts[1:]:
        name = part.split("(")[0].split()[0].strip()
        if name.endswith("_with_offset"):
            continue
        counts = {}
        for label, pat in PATTERNS:
            counts[label] = len(re.findall(pat, part, re.M))
        out[name] = counts
    return out


def main():
    total = collections.OrderedDict()
    for path in sys.argv[1:]:
        total.update(census(path))
    labels = [p[0] for p in PATTERNS]
    width = max(len(n) for n in total)
    for name, counts in total.items():
        print(name.ljust(width), end="  ")
        print("  ".join("%s=%d" % (l, counts[l]) for l in labels if counts[l]))


if __name__ == "__main__":
    main()
