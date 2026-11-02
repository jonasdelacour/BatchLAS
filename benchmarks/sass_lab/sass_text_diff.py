#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Per-kernel SASS text identity between two lab dumps (cuobjdump -sass, addresses stripped).

  sass_text_diff.py BASE.sass CAND.sass [--match REGEX]

Prints one line per kernel present in both whose instruction text differs, then a summary
"identical I/N"; exit 1 when any kernel differs or a BASE kernel is missing from CAND.
"""
import argparse
import re
import sys

FUNC = re.compile(r"Function : (\S+)")
INSN = re.compile(r"/\*[0-9a-f]{4,}\*/\s+(.*?)\s*;")


def kernels(path):
    out, name = {}, None
    with open(path) as f:
        for line in f:
            m = FUNC.search(line)
            if m:
                name = m.group(1)
                out[name] = []
                continue
            if name is None:
                continue
            m = INSN.search(line)
            if m:
                out[name].append(re.sub(r"`\(\S+\)", "`(L)", m.group(1)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("cand")
    ap.add_argument("--match", default=".")
    a = ap.parse_args()
    rx = re.compile(a.match)
    b, c = kernels(a.base), kernels(a.cand)
    names = sorted(k for k in b if rx.search(k))
    same, rc = 0, 0
    for k in names:
        if k not in c:
            print(f"MISSING {k}")
            rc = 1
        elif b[k] == c[k]:
            same += 1
        else:
            print(f"DIFF {k}: {len(b[k])} -> {len(c[k])} instructions")
            rc = 1
    print(f"identical {same}/{len(names)}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
