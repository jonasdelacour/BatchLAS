# Agent loop tools

Single plain commands the worktree Bash guard accepts. The skills in `.claude/skills/` (`/sync-main`,
`/fanout`, `/tune`, `/prepush`) say when to use which; rules in `docs/developer/agent-guide.md` §14.

| Script | Does |
| --- | --- |
| `session_start.sh` | SessionStart hook: checkout behind origin/main, held build lock, GPU state |
| `gpu_health.sh` | `nvidia-smi`/`sycl-ls`/driver-mismatch check; exit 1 means stop GPU work and tell the user |
| `wait_for.sh` | wait on a pid/log in the background; `--stall-min` catches a hung cell |
| `prepush.sh` | CI's checks locally: LFS ledgers, hash pins, strict docs build with CI's Doxygen (`get_doxygen.sh`), `run_local_checks.sh`, `pr_scope.py` |
| `pr_scope.py` | pre-push scope review: large files, campaign data, new headers, loosened tolerances, ledger edits |
| `attribute_failures.py` | "fails on main too?": cached base worktree, seeded tree, N reps, per-case verdicts |
| `conflicts.py` | per-hunk conflict previews and `take FILE IDX ours\|theirs\|both` |
| `repin_tune_hashes.py` | regenerate the TuneHash pins in `tests/tune_tests.cc` |
| `ci_failures.sh` / `pr_edit.sh` | CI verdict for a PR (exit 2 = pending); PR edits where `gh pr edit` fails |
| `doc_section.py` | page outline, one section by §N/anchor/title, or `--find` a term |
| `wf_dump.py` | split Workflow output into files and print an outline |
| `tune_procs.sh` | tuner holding a GPU lock, orphaned workers; `--kill PID` takes the tree |
| `vendor_gaps.py` | where the vendor still beats native, from the tuned tables |
| `dispatch.sh` | worktree + headless session on jrhnbr, tr02 (CPU/build only, GPUs hidden) or the Mac; `hosts`/`list`/`logs`/`report`/`stop`/`rm` |
| `prune_worktrees.sh` | remove clean, merged worktrees (`--apply`) |
| `claude-settings.json` | template for `.claude/settings.json`: `session_start.sh`, `scripts/agent_build_guard.py`, `scripts/agent_tier_guard.py`, read-only allow rules |
| `../agent_tier_guard.py` | blocks an Agent call or Workflow `agent()` that picks no tier from `.claude/agents/` |

`.github/ci/compare_failures.py` reads `tests/known-failures-host-<host>.txt` on top of
`tests/known-failures.txt`, so `scripts/gate.sh` does not report a box's known crashes as new.
