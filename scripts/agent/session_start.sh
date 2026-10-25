#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Claude Code SessionStart hook (.claude/settings.json): its stdout becomes session context.
# Keep it under ~10 lines and ~10 s. It exists because sessions ran for days on a checkout
# 131 commits behind origin/main and never saw the tooling the guide they loaded predates.
set -u
cd "${CLAUDE_PROJECT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}" 2>/dev/null || exit 0

timeout 8 git fetch -q origin main 2>/dev/null
branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null)
behind=$(git rev-list --count HEAD..origin/main 2>/dev/null || echo "?")
echo "BatchLAS session context ($(hostname -s), $(date '+%F %H:%M')):"
if [ "$behind" != 0 ]; then
    echo "- WARNING: $branch is $behind commit(s) behind origin/main. The agent guide, scripts and ledgers you see may be stale;"
    echo "  read new tooling with \`git show origin/main:<path>\`, and branch new work from origin/main."
else
    echo "- $branch is up to date with origin/main."
fi
dirty=$(git status --porcelain 2>/dev/null | grep -vc '^??' || true)
[ "${dirty:-0}" -gt 0 ] && echo "- $dirty tracked file(s) modified in this checkout."

lock=/tmp/batchlas-build.lock
if [ -e "$lock" ] && ! flock -n "$lock" true 2>/dev/null; then
    echo "- Build lock held: $(cat "$lock.info" 2>/dev/null || echo unknown). scripts/b waits for it."
fi
gpus=$(timeout 5 nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>&1)
if [ $? -ne 0 ]; then
    echo "- GPU: nvidia-smi FAILED ($(head -n 1 <<<"$gpus")). Run scripts/agent/gpu_health.sh; do not start GPU work."
else
    echo "- GPU (index, util, mem): $(tr '\n' ';' <<<"$gpus")"
fi
echo "- Agent tools: scripts/agent/README.md (wait_for, gpu_health, attribute_failures, prepush, conflicts, ci_failures). Skills: /sync-main /fanout /tune /prepush."
exit 0
