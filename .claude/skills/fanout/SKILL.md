---
name: fanout
description: Brief, launch and integrate parallel subagents or a Workflow on BatchLAS (several ops, work packages or investigations at once). Pre-flight the base, the brief clauses every agent needs, integration and verification of their branches, digesting workflow output cheaply.
---

# Fan work out to subagents or a workflow

Past fan-outs lost 94-147 minutes per agent to a base that did not link. The usual further costs were hand-edited generated files, fixes broadcast before they ran, and parents reading 35-50k-character reports whole.

## Before launching anything
- **Pre-flight the base.**
  - Run `scripts/agent/gpu_health.sh`. If it fails, tell the user now; do not start "compile-only" work.
  - Build the targets every agent will need on the base commit (`scripts/b -B <tree> <targets>`) and run one smoke test.
  - A base that does not build here is the first work package. Do not hand it to N agents.
- **Size it.** Give each agent one op or work package with a disjoint file set. Name the shared files (`tests/tune_tests.cc`, `tuned/`, `include/batchlas/tuning_params.hh`, `src/ops/CMakeLists.txt`) and give each one a single owner.
- **Tier every agent** (`.claude/agents/`, table in `docs/developer/agent-core.md`): `subagent_type: "<tier>"` on the Agent tool, `agentType: '<tier>'` in a workflow `agent()`. scout for locating, mechanic for specified edits, engineer for work packages, verifier for refuting, architect only for design. `scripts/agent_tier_guard.py` blocks an untiered launch.
- **Timebox.** Pick a check-in point (about 2-3 h of wall time). Tell the user the plan, the agent count and when you will report.

## Brief clauses (paste into every brief, after the task-specific part)
```
Context: BatchLAS, rules in docs/developer/agent-guide.md (read sections 7-9 only if you build/test/route).
Base: <sha>, verified to build and pass <smoke test> on this box at <time>.
Build only named targets with scripts/b -B <your tree> <targets>; it holds the box-wide lock and
the -j8 cap. Set TMPDIR to a disk path ($CLAUDE_JOB_DIR/tmp), never tmpfs /tmp.
Never hand-edit tuned/*.txt (regenerate with scripts/sweep_to_table.py from the ledger) or the
TuneHash pins (scripts/agent/repin_tune_hashes.py; <owner> owns them).
Wait on background work with scripts/agent/wait_for.sh run in the background, never sleep loops.
Report: what changed (files), what you ran and its result, and for every failure whether you
compared against base (scripts/agent/attribute_failures.py) -- say "not compared" if you did not.
Keep the report under 600 words; put long evidence in files and give their paths.
```

## Integrating
- Merge agent branches one at a time into the integration branch, build the union, and run the union's tests. An agent's "tests pass" covers its own tree only.
- **A fix you broadcast must have run, not just linked:** run the affected test before sending it to other agents.
- **Workflow output:** `python3 scripts/agent/wf_dump.py <tasks/ID.output>` writes every long string to its own file and prints an outline with sizes. Read only the sections you need.
- Treat agent claims as hypotheses. Re-run the decisive command yourself before reporting it to the user.
- Finish with the `/prepush` skill, then open the PR.
