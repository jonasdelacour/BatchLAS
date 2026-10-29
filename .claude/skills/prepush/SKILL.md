---
name: prepush
description: Everything to do before pushing a BatchLAS branch or opening/updating a PR - the GPU gate, attributing failures to base, CI's own checks locally (strict docs build, tuned tables, LFS), and the scope review the user sees. Use before any push.
---

# Before push

1. **Gate** (when C++, CMake, tests or tables changed).
   - Run `scripts/gate.sh --only full` with `run_in_background`; it takes about 40 min and holds the box-wide build lock.
   - Add `--only vf`, or nothing (both trees), when routing, vendor wrappers or `can_run` changed.
   - New tree: pass `--seed-from <an existing built tree>` (the configure takes about 2 min instead of 15).
   - The verdict is in `build-gate/gate-summary.txt`.
2. **Attribute new failures. Never excuse them by count.**
   - `tests/known-failures-host-<host>.txt` already lists what fails on main on this box. `compare_failures.py` reads it automatically and marks those entries `[host]`.
   - For anything else run `python3 scripts/agent/attribute_failures.py --tree build-gate <ctest names>`. Its verdicts are PRE-EXISTING, NEW, FLAKY and FIXED, against a cached base tree.
   - Fix NEW failures. Report PRE-EXISTING ones; add a host-ledger entry, with date and cause, only with the user's OK.
3. **CI's checks:** run `scripts/agent/prepush.sh` (about 1 min, no GPU). It covers:
   - LFS ledgers, the kernel-hash pins, the strict docs build with CI's pinned Doxygen (installed to `~/.local/opt` once), and every `run_local_checks.sh` gate.
   - The **scope review**.
   - `--only docs|checks|scope|pins` reruns one part.
4. **Show the user the scope review.** Paste every `REVIEW` line into your report; before pushing, ask when it flags raw results data, large or binary files, new headers, a loosened tolerance, or an added or removed ledger entry.
5. **Push and open the PR.**
   - Write the body to a file, then `gh pr create --draft --body-file F`. Edit it later with `scripts/agent/pr_edit.sh`.
   - Then run `scripts/agent/ci_failures.sh <pr>` in the background until it stops returning 2 (pending). Report the verdict, and fix failures before the user finds them.
