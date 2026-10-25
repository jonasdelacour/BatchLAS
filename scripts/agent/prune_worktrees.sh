#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# List the agent worktrees under .claude/worktrees and remove the ones that are finished.
#
#   scripts/agent/prune_worktrees.sh            # dry run: one line per worktree, verdict and size
#   scripts/agent/prune_worktrees.sh --apply    # remove every PRUNE worktree
#
# PRUNE = no uncommitted or untracked files (build trees aside), the branch is merged
# (its tip is an ancestor of origin/main, or its PR is MERGED/CLOSED), and nothing holds the
# directory (no process has its cwd inside). Branch refs are kept; only the checkout and its
# build trees go. Everything else is KEEP with the reason.
set -u
apply=0; [ "${1:-}" = --apply ] && apply=1
top=$(git rev-parse --path-format=absolute --git-common-dir | sed 's#/\.git$##')
base=$top/.claude/worktrees
git -C "$top" fetch -q origin main 2>/dev/null
declare -A pr_state
while IFS=$'\t' read -r head state; do pr_state[$head]=$state; done < <(
    gh pr list --state all --limit 300 --json headRefName,state -q '.[] | [.headRefName, .state] | @tsv' 2>/dev/null)
busy=$(for p in /proc/[0-9]*; do readlink "$p/cwd" 2>/dev/null; done | grep "^$base/" | cut -d/ -f1-"$(( $(tr -cd / <<<"$base" | wc -c) + 2 ))" | sort -u)

freed=0
for wt in "$base"/*/; do
    wt=${wt%/}
    [ -e "$wt/.git" ] || continue
    name=$(basename "$wt")
    branch=$(git -C "$wt" rev-parse --abbrev-ref HEAD 2>/dev/null)
    size=$(du -sm "$wt" 2>/dev/null | cut -f1)
    dirty=$(git -C "$wt" status --porcelain --untracked-files=normal 2>/dev/null \
        | grep -vE '^\?\? (build[^/]*|benchviz_runs)/?$' | wc -l)
    state=${pr_state[$branch]:-none}
    if git -C "$wt" merge-base --is-ancestor HEAD origin/main 2>/dev/null; then merged=yes
    elif [ "$state" = MERGED ] || [ "$state" = CLOSED ]; then merged=yes
    else merged=no; fi
    reason=""
    [ "$dirty" -gt 0 ] && reason="$dirty uncommitted file(s)"
    [ "$merged" = no ] && reason="${reason:+$reason; }branch not merged (PR: $state)"
    grep -qx "$wt" <<<"$busy" && reason="${reason:+$reason; }in use by a process"
    if [ -z "$reason" ]; then
        printf 'PRUNE  %-34s %6s MB  %s (PR %s)\n' "$name" "$size" "$branch" "$state"
        if [ "$apply" = 1 ]; then
            git -C "$top" worktree remove --force "$wt" && freed=$((freed + size))
        fi
    else
        printf 'KEEP   %-34s %6s MB  %s: %s\n' "$name" "$size" "$branch" "$reason"
    fi
done
[ "$apply" = 1 ] && { git -C "$top" worktree prune; echo "removed; ~${freed} MB freed"; }
[ "$apply" = 1 ] || echo "(dry run; --apply removes the PRUNE lines)"
