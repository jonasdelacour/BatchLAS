#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Claude Code PreToolUse hook (matcher Agent|Task|Workflow): every subagent picks a model tier.

Blocks (exit 2, reason on stderr):
  * an Agent/Task call whose subagent_type is not a project tier agent (.claude/agents/*.md) or a
    built-in with its own fixed definition (FIXED), unless it passes both `model` and `effort`;
  * a Workflow script (inline `script`, or the file at `scriptPath`) with an agent() call whose
    options set neither `agentType: '<tier>'` nor both `model` and `effort`, or name an
    agentType that is neither a tier nor FIXED.
Options built from a variable or a spread (`{...TIER.scout, schema}`) are trusted. A saved
workflow run by `name` is not inspected. A script line `// tier-guard: off <reason>` disables
the check for that script; the reason shows in the permission dialog.
Anything this script fails to parse passes (exit 0).
evidence: docs/developer/agent-core.md#core-subagent-and-workflow-tiers
"""

import json
import os
import re
import sys

FIXED = {"fork", "claude-code-guide", "statusline-setup"}
IDENT = re.compile(r"[A-Za-z0-9_$.]")


def tiers(root):
    out = {}
    d = os.path.join(root, ".claude", "agents")
    for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        if not f.endswith(".md"):
            continue
        with open(os.path.join(d, f)) as fh:
            head = fh.read(2000)
        m = re.search(r"^name:\s*(\S+)", head, re.M)
        desc = re.search(r"^description:\s*(.+)$", head, re.M)
        if m:
            out[m.group(1)] = (desc.group(1).strip().strip('"')[:110] if desc else "")
    return out


def roster(ts):
    return "\n".join("  %-10s %s" % (k, v) for k, v in ts.items()) or "  (no .claude/agents/*.md found)"


def blank(src):
    """Same-length copy of src with comments and string bodies replaced by spaces."""
    out, i, n = list(src), 0, len(src)
    while i < n:
        c = src[i]
        if src.startswith("//", i) or src.startswith("/*", i):
            end = src.find("\n" if src[i + 1] == "/" else "*/", i + 2)
            end = n if end < 0 else end + (0 if src[i + 1] == "/" else 2)
            out[i:end] = " " * (end - i)
            i = end
        elif c in "'\"`":
            j = i + 1
            while j < n and src[j] != c:
                j += 2 if src[j] == "\\" else 1
            out[i + 1:j] = " " * (min(j, n) - i - 1)
            i = j + 1
        else:
            i += 1
    return "".join(out)


def call_args(code, start):
    """Top-level argument spans of the call whose '(' is at start."""
    depth, spans, a = 0, [], start + 1
    for j in range(start, len(code)):
        ch = code[j]
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
            if depth == 0:
                spans.append((a, j))
                return spans
        elif ch == "," and depth == 1:
            spans.append((a, j))
            a = j + 1
    return spans


def check_script(src, ts):
    if re.search(r"^\s*//\s*tier-guard:\s*off\b", src, re.M):
        return []
    code, bad = blank(src), []
    for m in re.finditer(r"\bagent\s*\(", code):
        if m.start() > 0 and IDENT.match(code[m.start() - 1]):
            continue
        line = src.count("\n", 0, m.start()) + 1
        spans = call_args(code, m.end() - 1)
        opts_code = code[spans[1][0]:spans[1][1]].strip() if len(spans) > 1 else ""
        opts = src[spans[1][0]:spans[1][1]] if len(spans) > 1 else ""
        if not opts_code:
            bad.append("line %d: agent() without options" % line)
            continue
        if not opts_code.startswith("{") or "..." in opts_code:
            continue
        at = re.search(r"\bagentType\s*:\s*(['\"`])([\w:-]+)\1", opts)
        if at:
            if at.group(2) not in ts and at.group(2) not in FIXED:
                bad.append("line %d: agentType '%s' is not a tier agent" % (line, at.group(2)))
            continue
        if re.search(r"\bagentType\s*:", opts_code):
            continue
        has = [k for k in ("model", "effort") if re.search(r"\b%s\s*:" % k, opts_code)]
        if len(has) < 2:
            bad.append("line %d: agent() sets %s" % (line, " and ".join(has) + " only" if has else "no tier"))
    return bad


def main():
    try:
        payload = json.load(sys.stdin)
        tool, inp = payload.get("tool_name", ""), payload.get("tool_input") or {}
        root = os.environ.get("CLAUDE_PROJECT_DIR") or payload.get("cwd") or os.getcwd()
        ts = tiers(root)
        if tool in ("Agent", "Task"):
            st = inp.get("subagent_type") or "general-purpose"
            if st in ts or st in FIXED or (inp.get("model") and inp.get("effort")):
                return 0
            reason = ("subagent_type '%s' without both `model` and `effort`. Pick the tier agent that fits "
                      "the task as subagent_type:\n%s\nor pass model + effort explicitly" % (st, roster(ts)))
        elif tool == "Workflow":
            src = inp.get("script")
            if not src and inp.get("scriptPath"):
                with open(inp["scriptPath"]) as fh:
                    src = fh.read()
            bad = check_script(src or "", ts)
            if not bad:
                return 0
            reason = ("workflow agent() calls without a tier:\n  %s\nGive each call `agentType: '<tier>'` "
                      "(tiers below) or both `model` and `effort`; cheap mechanical stages take the "
                      "cheapest tier that can do them:\n%s" % ("\n  ".join(bad), roster(ts)))
        else:
            return 0
    except Exception:
        return 0
    sys.stderr.write("BLOCKED by scripts/agent_tier_guard.py: %s.\nTier rules: docs/developer/agent-core.md "
                     "(\"Core: subagent and workflow tiers\").\n" % reason)
    return 2


if __name__ == "__main__":
    sys.exit(main())
