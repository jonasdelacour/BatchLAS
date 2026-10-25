---
name: sync-main
description: Merge the current origin/main into a PR branch and get it green again ("make PR N compatible with main", "finish PR N on the updated main", "combine PRs"). Conflict triage without dumping diffs, regenerated hash pins and tables, scope review before push.
---

# Sync a PR branch with main

The goal is the PR as the user wrote it, on today's main. It is not a chance to port, tidy or widen it.

1. **Start state.** Run `git fetch origin` and confirm the branch. Note the PR's own scope first with `python3 scripts/agent/pr_scope.py`, so you can tell later what the merge added. If that report already flags raw results data, new helper headers or tolerance edits, ask the user about them before merging.
2. **Merge, don't rebase** (other agents and the user may have the branch): `git merge origin/main`.
3. **Triage.** Run `python3 scripts/agent/conflicts.py`: one line per hunk with a 3-line preview per side. Never `git diff` a conflicted file whole (each dump cost 20-30k characters in past sessions).
   - Resolve with `conflicts.py take FILE IDX ours|theirs|both`. Use `show FILE IDX` only for a hunk the preview cannot decide.
   - For a hunk that needs both sides edited, use Edit on the marked region.
   - **Generated files are never hand-merged:**
     - `tests/tune_tests.cc` pins: take either side, then `python3 scripts/agent/repin_tune_hashes.py`.
     - `tuned/*.txt`: take main's, then regenerate from the merged ledgers with `scripts/sweep_to_table.py --op <op> --device <dev>`. Run `git lfs pull` first if the ledgers are pointers.
     - `tests/known-failures*.txt`: union, then let the gate tell you which entries now pass.
4. **Semantic clashes.** Build first the test targets whose sources *both* sides touched. Both past merges failed 40 minutes into a full build on a helper that one side renamed or removed (`PaddedBatch`, `test_utils::tolerance`). Find them with `git diff --name-only origin/main...HEAD -- tests/` intersected with `git diff --name-only HEAD...origin/main -- tests/`. Build them with `scripts/b -B <tree> <targets>`. `python3 scripts/affected_tests.py --print-build-cmd` gives the rest.
5. **Do not widen the PR.**
   - Do not port main's new code into the PR's style, loosen a tolerance, delete a ledger section, or add helper headers or classes. If one of these seems necessary, stop and ask.
   - Prefer the repo's established classes over custom behaviour in a benchmark or test.
6. **Gate.**
   - Run `scripts/gate.sh --only full` in the background (`run_in_background`).
   - Attribute every new failure. First check `tests/known-failures-host-$(hostname -s).txt`, then run `python3 scripts/agent/attribute_failures.py <tests>`. Never rebuild main by hand.
7. **Before push:** run `scripts/agent/prepush.sh` and paste its REVIEW lines into your report to the user.
8. **After push:** `scripts/agent/ci_failures.sh <pr>` gives the verdict (exit 2 means still pending). Update the PR text with `scripts/agent/pr_edit.sh`; `gh pr edit` fails on this repo.
