#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# What failed in a PR's CI, compactly: failing jobs, their error lines, and the Doxygen warning
# log (which the docs job writes to a file, so the job log alone never shows it).
#
#   scripts/agent/ci_failures.sh <pr-number> [--full]
#   scripts/agent/ci_failures.sh --sha <commit> [--full]     (an older push of the PR)
#
# Works around two gh traps on this repo: `gh run view --log[-failed]` returns nothing, and
# `gh pr checks --watch` exits while checks are still pending. Exit 0 all green, 1 failures,
# 2 still pending.
set -u
repo=$(gh repo view --json nameWithOwner -q .nameWithOwner)
if [ "${1:-}" = --sha ]; then
    sha=${2:?--sha needs a commit}; pr=${sha:0:12}; shift 2
else
    pr=${1:?usage: ci_failures.sh <pr-number>|--sha <commit> [--full]}; shift
    sha=$(gh api "repos/$repo/pulls/$pr" -q .head.sha)
fi
full=${1:-}
out=${CLAUDE_JOB_DIR:-/tmp}/tmp/ci-pr$pr
mkdir -p "$out"
echo "PR/commit $pr head ${sha:0:12} ($repo)"
gh api "repos/$repo/commits/$sha/check-runs?per_page=100" \
    -q '.check_runs[] | [.id, .status, (.conclusion // "-"), .name] | @tsv' > "$out/checks.tsv"
pending=$(awk -F'\t' '$2 != "completed"' "$out/checks.tsv" | wc -l)
failed=$(awk -F'\t' '$2 == "completed" && $3 != "success" && $3 != "skipped" && $3 != "neutral"' "$out/checks.tsv")
echo "checks: $(wc -l < "$out/checks.tsv") total, $pending pending, $(grep -c . <<<"$failed") failed"
awk -F'\t' '{printf "  %-10s %-10s %s\n", $2, $3, $4}' "$out/checks.tsv"

while IFS=$'\t' read -r id _status concl name; do
    [ -n "$id" ] || continue
    log="$out/job-$id.log"
    gh api "repos/$repo/actions/jobs/$id/logs" > "$log" 2>/dev/null || true
    echo
    echo "=== FAILED: $name ($concl), log $log"
    if [ "$full" = --full ]; then tail -n 120 "$log"
    else
        # The step output just before each ##[error] is the cause; strip the timestamps.
        grep -B 15 '##\[error\]' "$log" | sed 's/^[0-9T:.-]*Z //' | grep -v '^--$' | head -n 60
        grep -nE 'error:|FAILED|NEW FAILURES|Traceback' "$log" | sed 's/[0-9T:.-]*Z //' | head -n 20
    fi
done <<<"$failed"

if grep -qiE 'docs|documentation' <<<"$failed"; then
    run=$(gh api "repos/$repo/actions/runs?head_sha=$sha&per_page=20" -q '.workflow_runs[0].id')
    rm -rf "$out/docs-html"
    if gh run download "$run" -n docs-html -D "$out/docs-html" >/dev/null 2>&1; then
        echo
        echo "=== doxygen-warnings.log ($out/docs-html)"
        find "$out/docs-html" -name doxygen-warnings.log -exec head -n 40 {} \;
        find "$out/docs-html" -name doxygen-warnings.log -size 0 | grep -q . &&
            echo "(empty: the failure is not in the log; reproduce with scripts/agent/prepush.sh --only docs)"
    else
        echo "(no docs-html artifact for run $run)"
    fi
fi
[ -n "$failed" ] && exit 1
[ "$pending" -gt 0 ] && exit 2
exit 0
