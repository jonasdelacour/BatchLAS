#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Self-test for scripts/agent_tier_guard.py: runs the hook as Claude Code does (JSON on stdin).

    python3 scripts/tests/test_agent_tier_guard.py
"""

import json
import os
import subprocess
import sys
import tempfile

HOOK = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent_tier_guard.py")
ROOT = os.path.dirname(os.path.dirname(HOOK))

BLOCK, PASS = 2, 0


def agent(**inp):
    return {"tool_name": "Agent", "tool_input": dict(prompt="p", description="d", **inp)}


def wf(script):
    return {"tool_name": "Workflow", "tool_input": {"script": "export const meta = {name: 'x', description: 'y'}\n" + script}}


CASES = [
    # Agent tool
    (agent(subagent_type="scout"), PASS, "a tier agent"),
    (agent(subagent_type="architect"), PASS, "the top tier"),
    (agent(subagent_type="fork"), PASS, "fork keeps the parent's model"),
    (agent(subagent_type="claude-code-guide"), PASS, "built-in with a fixed definition"),
    (agent(), BLOCK, "no subagent_type means general-purpose on the session model"),
    (agent(subagent_type="general-purpose"), BLOCK, "general-purpose untiered"),
    (agent(subagent_type="Explore"), BLOCK, "Explore untiered"),
    (agent(subagent_type="general-purpose", model="sonnet"), BLOCK, "model without effort"),
    (agent(subagent_type="general-purpose", effort="low"), BLOCK, "effort without model"),
    (agent(subagent_type="Explore", model="haiku", effort="low"), PASS, "explicit model and effort"),
    (agent(subagent_type="scouts"), BLOCK, "a lookalike of a tier name"),
    ({"tool_name": "Task", "tool_input": {"subagent_type": "general-purpose"}}, BLOCK, "the Task spelling"),
    # Workflow scripts
    (wf("await agent('scan', {agentType: 'scout', label: 's'})"), PASS, "agentType tier"),
    (wf("await agent(`scan ${x}`, {schema: S, model: 'haiku', effort: 'low'})"), PASS, "model and effort"),
    (wf("await agent('scan')"), BLOCK, "no options"),
    (wf("await agent('scan', {schema: S})"), BLOCK, "options without a tier"),
    (wf("await agent('scan', {model: 'opus'})"), BLOCK, "model only"),
    (wf("await agent('scan', {agentType: 'general-purpose'})"), BLOCK, "agentType not a tier"),
    (wf("await agent('scan', {...TIER.scout, schema: S})"), PASS, "spread options are trusted"),
    (wf("await agent('scan', OPTS)"), PASS, "options from a variable are trusted"),
    (wf("const r = await parallel(xs.map(x => () => agent(x.p, {agentType: 'mechanic'})))\n"
        "await agent('judge', {schema: V})"), BLOCK, "one untiered call among tiered ones"),
    (wf("await agent('write an agent(x) call', {agentType: 'scout'})"), PASS, "agent( inside a string is text"),
    (wf("// call agent('x') later\nawait agent('y', {agentType: 'verifier'})"), PASS, "agent( inside a comment"),
    (wf("await myagent('x')\nawait obj.agent('y')"), PASS, "other functions named *agent"),
    (wf("// tier-guard: off one-off probe\nawait agent('scan')"), PASS, "explicit opt-out line"),
    (wf("await agent('a', {agentType: 'scout', schema: {type: 'object', properties: {p: {type: 'string'}}}})"),
     PASS, "nested braces in options"),
    # never block on garbage or other tools
    ({"tool_name": "Workflow", "tool_input": {"name": "saved-flow"}}, PASS, "saved workflow by name"),
    ({"tool_name": "Bash", "tool_input": {"command": "ls"}}, PASS, "another tool"),
    ({"tool_name": "Agent", "tool_input": None}, BLOCK, "null input is an untiered general-purpose call"),
]


def run(payload, env):
    stdin = payload if isinstance(payload, str) else json.dumps(payload)
    p = subprocess.run([sys.executable, HOOK], input=stdin, capture_output=True, text=True, env=env, cwd=ROOT)
    return p.returncode, p.stderr


def main():
    env = dict(os.environ, CLAUDE_PROJECT_DIR=ROOT)
    cases = list(CASES) + [("not json", PASS, "unparsable stdin")]
    bad = 0
    for payload, want, label in cases:
        rc, err = run(payload, env)
        ok = rc == want and (rc != 2 or "scout" in err)
        bad += not ok
        print("  %-4s %s  %s" % ("ok" if ok else "FAIL", "block" if want else "pass ", label))
        if not ok:
            print("       exit %d, stderr: %s" % (rc, err.strip()))
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
        f.write("await agent('scan', {schema: S})\n")
    rc, _ = run({"tool_name": "Workflow", "tool_input": {"scriptPath": f.name}}, env)
    os.unlink(f.name)
    bad += rc != BLOCK
    print("  %-4s block  an untiered call in a scriptPath file" % ("ok" if rc == BLOCK else "FAIL"))
    print("agent_tier_guard self-test: %d failure(s) of %d" % (bad, len(cases) + 1))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
