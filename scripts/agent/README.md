# Agent loop tools

Each script replaces a procedure that agent sessions hand-wrote, often several times, in PRs #187-#196. Every one runs as a single plain command, which the worktree Bash guard accepts. The project skills in `.claude/skills/` (`/sync-main`, `/fanout`, `/tune`, `/prepush`) say when to use which. The rules are in `docs/developer/agent-guide.md` §14.

| Script | Replaces |
| --- | --- |
| `session_start.sh` | SessionStart hook: warns when the checkout is behind origin/main, shows a held build lock and the GPU state |
| `gpu_health.sh` | re-typed `nvidia-smi`/`sycl-ls`/driver-mismatch checks; exit 1 means stop GPU work and tell the user |
| `wait_for.sh` | `until ...; sleep` polls that hit the Bash timeout, and `pgrep -f` waiters that match themselves; `--stall-min` catches a hung cell |
| `prepush.sh` | CI's checks locally: LFS ledgers, hash pins, strict docs build with CI's Doxygen (`get_doxygen.sh`), `run_local_checks.sh`, `pr_scope.py` |
| `pr_scope.py` | the unreviewed `git diff --stat` before a push: large files, campaign data, new headers, loosened tolerances, ledger edits |
| `attribute_failures.py` | hand-built "does it fail on main too" trees: a cached base worktree, a seeded tree, N reps, per-case verdicts |
| `conflicts.py` | whole conflict diffs dumped into context, and ad-hoc per-hunk resolvers |
| `repin_tune_hashes.py` | hand-`sed` of the TuneHash pins in `tests/tune_tests.cc` |
| `ci_failures.sh` / `pr_edit.sh` | empty `gh run view --log`, `gh pr edit` failing on the Projects-classic GraphQL error |
| `doc_section.py` | whole-page Reads and `sed -n` ranges of 70-200 KB `docs/perf` pages and of the agent guide: prints an outline, one section by §N/anchor/title, or which sections mention a term |
| `wf_dump.py` | ad-hoc JSON unwrapping of Workflow output and reading 50k-character results whole |
| `tune_procs.sh` | finding the tuner that holds a GPU lock, orphaned workers, killing a run tree |
| `vendor_gaps.py` | re-written `losses.py` scripts: where the vendor still beats native, from the tuned tables |
| `dispatch.sh` | hand-typed `ssh host 'cd ...; claude -p ...'` launches: a worktree and headless session on jrhnbr, tr02 (CPU/build only, GPUs hidden) or the Mac, with per-host rules, `hosts`/`list`/`logs`/`report`/`stop`/`rm` |
| `prune_worktrees.sh` | 60+ stale worktrees; removes only clean, merged ones (`--apply`) |
| `claude-settings.json` | template for `.claude/settings.json`: registers `session_start.sh`, the build guard `scripts/agent_build_guard.py`, the tier guard `scripts/agent_tier_guard.py`, and read-only allow rules |
| `../agent_tier_guard.py` | subagents and workflow stages launched on the session's model and effort whatever the task: blocks an Agent call or Workflow `agent()` that picks no tier from `.claude/agents/` (scout, mechanic, engineer, verifier, architect) |

The per-host failure ledger `tests/known-failures-host-<host>.txt` is read by `.github/ci/compare_failures.py` on top of `tests/known-failures.txt`, so `scripts/gate.sh` no longer reports this box's known crashes as new.
