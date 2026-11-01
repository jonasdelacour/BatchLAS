---
name: dead-code
description: Find BatchLAS code nothing uses any more (unreferenced functions, files, exports, CMake options, BATCHLAS_* env flags, scripts, routes) with static analysis plus a dynamic check where the build and tests allow it, then open ONE draft PR that deletes it and says why each item is safe to remove. Runs weekly from launchd (~/.dead-code/run.sh); also invocable by hand ("dead code sweep", "find unused code").
---

# Dead-code sweep

Goal: one draft PR that only deletes. Every deleted item carries its evidence. When in doubt, leave it
out of the PR and list it under "Candidates kept" instead. A missed deletion costs nothing; a wrong one
breaks a downstream user or another backend.

## 0. Set up (unattended-safe)

- `git fetch origin`; work in a fresh worktree from `origin/main` (EnterWorktree, name
  `dead-code-<YYYY-MM-DD>`), then `scripts/seed_build_tree.sh ~/BatchLAS/build build`.
- **Do not duplicate:** `gh pr list --state open --search 'dead code in:title' --json number,title,headRefName`.
  If an open dead-code PR exists, do not open another: report its number and stop.
  Also drop any candidate already touched by another open PR (`gh pr list --json files` or
  `scripts/agent/conflicts.py`).
- Run `scripts/agent/gpu_health.sh` only if you will run GPU tests; on this Mac there is no CUDA, so the
  dynamic check uses the Metal tree (`BATCHLAS_GATE_CUDA=OFF`, `BATCHLAS_GATE_PRESET=acpp-metal-tests`).
- Hand noisy sweeps to `scout` subagents (one per category below), keep only their candidate lists.
  Persist raw lists to `$CLAUDE_JOB_DIR/tmp/dead-code/` (or `~/.dead-code/work/` when unset).

## 1. Static candidates

For each category produce `name | defining file:line | reference count outside its definition | how checked`.
Search the **whole tree** with `git grep -nw` (src, include, tests, benchmarks, examples, python, tools,
playground, scripts, cmake, CMakeLists.txt, CMakePresets.json, .github, docs, tuned), not just `src/`.

1. **Functions / classes / templates** in `src/` and `include/`: symbols with zero references besides their
   declaration and definition. Cross-check with the compiler where possible: build with
   `-Wunused-function -Wunused-template -Wunused-member-function -Wunused-private-field`
   (clang) into a scratch tree, and with `nm -C` on the built `.so` vs. `git grep`.
2. **Files**: `.cc/.hh` not in any `CMakeLists.txt` source list and not `#include`d; headers nobody
   includes; scripts in `scripts/` and `tools/` not called from any script, CMake, CI workflow, skill,
   agent file or doc.
3. **Exports**: entries in `include/` public headers and `python/` bindings. Public API is consumed
   **outside this repo**: an unreferenced public symbol is a candidate only if it is also undocumented in
   `docs/` and not part of a documented op family. Otherwise list it under "Candidates kept".
4. **Flags**: CMake `option()`/cache variables never read by any `if()`/generator expression;
   `BATCHLAS_*` env vars (`src/util/settings.cc` and `getenv` sites) never read, or read but never set
   or documented; `BATCHLAS_TUNE_*` knobs whose default is the only value any table uses.
5. **Routes / kernels**: `choice.hh` spellings whose `can_run()` is always false, or that no
   `tuned/<op>.<device>.txt` on any device selects and no test pins via `BATCHLAS_<OP>_ROUTE`.
   A route reachable only through a pin is still alive if a test or doc pins it.

## 2. Traps (read before marking anything dead)

- **Other backends.** This Mac builds only Metal. CUDA/ROCm/oneMKL/vendor-BLAS code is not compiled
  here but is live. Never conclude "unused" from a Metal build; check `#if`/`BATCHLAS_HAS_*` guards and
  grep both arms. Vendor wrappers stay unless unreferenced in every configuration.
- **Indirect references.** String-built names (route spellings, `BATCHLAS_` + op name, macro token
  pasting, X-macro lists, explicit template instantiation files, registration tables, Python
  `getattr`, CMake `foreach` over names). Grep the stem, not only the full identifier.
- **Docs anchors and evidence.** Cited heading slugs are an interface
  (`.github/ci/check_evidence_anchors.py`); files cited from `docs/` or ledgers are not dead.
- **Kernel-hash pins and tables.** Deleting a kernel or route can change hashes: rerun
  `scripts/agent/repin_tune_hashes.py` and `scripts/sweep_to_table.py` as the repo requires, or drop the item.
- **Ledgers** (`tests/known-failures*.txt`) and committed campaign data: never edit them here.
- **Tests are not users.** A symbol referenced only by its own unit test is a candidate; delete the
  test with it and say so. A test helper used by other tests is alive.

## 3. Dynamic check (where the build and tests allow)

- Coverage of routes: run the affected test binaries with `BATCHLAS_COVERAGE_OUT=$W/cov` and merge with
  `scripts/coverage_merge.sh $W/cov`; a route `linked` but never `reached` supports, but does not prove,
  deadness (other devices may reach it). `scripts/route_diff.sh` compares two trees.
- Deletion check: apply the deletions, build with `scripts/b <targets>` (from
  `scripts/affected_tests.py --print-build-cmd --print-ctest-cmd`), run those ctests, then the
  `*_heavy` variant. Any new compile, link or test failure means the item is not dead: restore it and
  move it to "Candidates kept" with the error.
- Python: `python -c 'import batchlas'` and the python tests if bindings changed.
- Not verifiable on this box (CUDA-only code)? Then it is not deleted this week; say so.
- Refute before you commit: give the final list (items + evidence, not your conclusion) to one
  `verifier` subagent asked to find a live use of each item. Drop every item it finds a use for.

## 4. The PR

- Keep it the size of the ask: deletions only, no refactors, no new helpers, no tolerance changes.
  Cap at about 40 items or 1500 deleted lines; leave the rest for next week's run.
- Commit per category. Run the `/prepush` skill (gate on the Metal tree, `scripts/agent/prepush.sh`,
  `pr_scope.py` REVIEW lines). If prepush flags anything in the scope list, drop those items rather
  than asking: the run is unattended.
- Title `Remove dead code (<YYYY-MM-DD> sweep)`. Write the body to a file and
  `gh pr create --draft --body-file F`. Body:

  ```text
  ## Removed
  | Item | Kind | Where | Why it is safe (static evidence) | Dynamic check |
  ## Candidates kept (and why)
  ## Verification
  <build targets, ctest commands and results, gate summary, REVIEW lines>
  ```

  Every "Why it is safe" cell names the search that came back empty (e.g. `git grep -nw foo -- .` -> 0
  hits outside its definition) and the other configurations considered.
- Then `scripts/agent/ci_failures.sh <pr>` in the background until it stops returning 2; fix failures
  that the deletions caused by restoring the item.
- Nothing safe to remove? Open no PR; report "no dead code found" with the candidates kept.

## 5. Report

Finish with the PR URL (or "no PR" and why), counts removed/kept per category, and the log path.
