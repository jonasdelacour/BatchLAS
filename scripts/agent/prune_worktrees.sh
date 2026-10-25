#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# List the agent worktrees under .claude/worktrees and remove the ones that are finished.
#
#   scripts/agent/prune_worktrees.sh [--stale-days N]            # dry run: verdict and size each
#   scripts/agent/prune_worktrees.sh [--stale-days N] --apply    # remove every PRUNE worktree
#
# PRUNE = no uncommitted or untracked files (build trees aside), the branch is merged
# (its tip is an ancestor of origin/main, or its PR is MERGED/CLOSED), and nothing holds the
# directory (no process has its cwd inside). Branch refs are kept; only the checkout and its
# build trees go. Everything else is KEEP with the reason.
# --stale-days N also prunes (STALE) a worktree with no OPEN PR whose newest activity -- last
# commit, or the mtime of any uncommitted file -- is older than N days, merged or not and DIRTY
# OR NOT: the user's rule is that unpublished work older than a week is obsolete. Its
# uncommitted changes are lost; committed work stays reachable from the kept branch ref.
set -u
apply=0 stale_days=0
while [ "$#" -gt 0 ]; do
    case $1 in
        --apply) apply=1; shift ;;
        --stale-days) stale_days=$2; shift 2 ;;
        *) echo "prune_worktrees: unknown argument $1" >&2; exit 2 ;;
    esac
done
now=$(date +%s)
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
    verdict=PRUNE
    if [ -n "$reason" ] && [ "$stale_days" -gt 0 ] && [ "$state" != OPEN ]; then
        last=$(git -C "$wt" log -1 --format=%ct HEAD 2>/dev/null || echo 0)
        while IFS= read -r f; do
            m=$(stat -c %Y "$wt/$f" 2>/dev/null || echo 0)
            [ "$m" -gt "$last" ] && last=$m
        done < <(git -C "$wt" status --porcelain --untracked-files=normal 2>/dev/null \
                 | grep -vE '^\?\? (build[^/]*|benchviz_runs)/?$' | cut -c4-)
        age=$(( (now - last) / 86400 ))
        if [ "$age" -ge "$stale_days" ]; then
            verdict=STALE; reason=""
            [ "$dirty" -gt 0 ] && state="$state, idle ${age}d, DROPS $dirty uncommitted file(s)" \
                               || state="$state, idle ${age}d"
        fi
    fi
    grep -qx "$wt" <<<"$busy" && { verdict=PRUNE; reason="${reason:+$reason; }in use by a process"; }
    if [ -z "$reason" ]; then
        printf '%-6s %-34s %6s MB  %s (PR %s)\n' "$verdict" "$name" "$size" "$branch" "$state"
        if [ "$apply" = 1 ]; then
            git -C "$top" worktree remove --force "$wt" && freed=$((freed + size))
        fi
    else
        printf 'KEEP   %-34s %6s MB  %s: %s\n' "$name" "$size" "$branch" "$reason"
    fi
done
[ "$apply" = 1 ] && { git -C "$top" worktree prune; echo "removed; ~${freed} MB freed"; }
[ "$apply" = 1 ] || echo "(dry run; --apply removes the PRUNE and STALE lines)"
