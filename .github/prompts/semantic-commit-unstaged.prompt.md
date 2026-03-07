---
name: Semantic Commit Unstaged Changes
description: "Inspect unstaged repository changes, group them into honest semantic commits, and stage and create those commits with concise factual messages."
argument-hint: "Optional constraints for grouping or commit wording"
agent: "agent"
---
Review the repository's current unstaged changes and turn them into semantically related commits.

Follow this process:

1. Inspect the unstaged diff before changing the index.
2. Propose a commit plan that groups changed files and hunks by actual purpose, not by directory alone.
3. If the grouping is ambiguous, risky, or mixes unrelated work, stop and ask a short clarifying question before staging anything.
4. Otherwise, stage one commit at a time, create the commit, then continue with the remaining unstaged changes until all intended changes are committed.
5. After each commit, verify what remains unstaged so the next commit stays clean.

Commit rules:

- Keep commit messages concise: 1 or 2 sentences maximum.
- Be strictly factual. Do not exaggerate neatness, performance, maintainability, correctness, or scope.
- Describe what changed and, only when justified by the diff, why.
- Do not claim improvements that are not clearly supported by the changes.
- Do not rewrite, revert, or discard user changes unless explicitly asked.
- Do not combine unrelated edits just to reduce commit count.
- Prefer a small number of coherent commits over many tiny commits, but split distinct concerns when the diff supports it.

Before creating commits, show the planned commit list with a short title per commit. Then perform the work.

At the end, report:

- The commits you created in order
- Any changes intentionally left unstaged or uncommitted
- Any uncertainty you encountered while grouping changes

If the user supplied extra instructions as an argument, apply them as long as they do not conflict with the rules above.