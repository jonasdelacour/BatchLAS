#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Split a Workflow (or background-task) output file into small files and print an outline, so
the parent Reads only the sections it needs instead of a 35-50k character blob.

    python3 scripts/agent/wf_dump.py <tasks/ID.output> [OUT_DIR] [--min-chars 400]

Every string in `result` longer than --min-chars goes to its own file, named by its JSON path
(result.reports.2.summary.md); the outline shows the remaining structure, each file's size, and
the first line of each long string. `logs` go to logs.txt; summary, agent count and token totals
are printed. A non-JSON file (a plain agent transcript) is refused, not parsed.
"""

import json
import os
import re
import sys


def safe(path):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", path)[:120]


def walk(node, path, out_dir, min_chars, lines, depth=0):
    pad = "  " * depth
    if isinstance(node, dict):
        for k, v in node.items():
            walk(v, "%s.%s" % (path, k), out_dir, min_chars, lines, depth + (0 if depth == 0 and not lines else 1))
    elif isinstance(node, list):
        lines.append("%s%s: list[%d]" % (pad, path, len(node)))
        for i, v in enumerate(node):
            walk(v, "%s.%d" % (path, i), out_dir, min_chars, lines, depth + 1)
    elif isinstance(node, str) and len(node) >= min_chars:
        name = safe(path) + ".md"
        with open(os.path.join(out_dir, name), "w") as f:
            f.write(node)
        first = next((l.strip() for l in node.splitlines() if l.strip()), "")[:100]
        lines.append("%s%s -> %s (%d chars) %s" % (pad, path, name, len(node), first))
    else:
        text = json.dumps(node)
        lines.append("%s%s = %s" % (pad, path, text if len(text) < 140 else text[:140] + "..."))


def main(argv):
    args = [a for a in argv if not a.startswith("--")]
    min_chars = 400
    if "--min-chars" in argv:
        min_chars = int(argv[argv.index("--min-chars") + 1])
        args.remove(str(min_chars))
    if not args:
        print(__doc__)
        return 2
    src = args[0]
    out_dir = args[1] if len(args) > 1 else os.path.join(
        os.environ.get("CLAUDE_JOB_DIR", "/tmp"), "tmp", "wf-" + os.path.basename(src).split(".")[0])
    try:
        with open(src) as f:
            data = json.load(f)
    except ValueError:
        print("wf_dump: %s is not one JSON document (an agent transcript?); not reading it" % src)
        return 2
    os.makedirs(out_dir, exist_ok=True)
    if isinstance(data, dict):
        for key in ("summary", "agentCount", "totalTokens", "totalToolCalls"):
            if key in data:
                print("%s: %s" % (key, data[key]))
        if data.get("logs"):
            with open(os.path.join(out_dir, "logs.txt"), "w") as f:
                f.write("\n".join(map(str, data["logs"])))
            print("logs: %d line(s) -> logs.txt" % len(data["logs"]))
    lines = []
    walk(data.get("result", data) if isinstance(data, dict) else data, "result", out_dir, min_chars, lines)
    print("outline (files in %s):" % out_dir)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
