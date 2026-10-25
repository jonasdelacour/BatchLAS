#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Set a PR's title and/or body through the REST API. `gh pr edit` fails on this repo with the
# GraphQL "Projects (classic) is being deprecated" error.
#
#   scripts/agent/pr_edit.sh <pr-number> [--title TEXT] [--body-file FILE]
set -eu
pr=${1:?usage: pr_edit.sh <pr-number> [--title TEXT] [--body-file FILE]}
shift
args=()
while [ "$#" -gt 0 ]; do
    case $1 in
        --title) args+=(-f "title=$2"); shift 2 ;;
        --body-file) args+=(-F "body=@$2"); shift 2 ;;
        *) echo "pr_edit: unknown argument $1" >&2; exit 2 ;;
    esac
done
[ "${#args[@]}" -gt 0 ] || { echo "pr_edit: nothing to change" >&2; exit 2; }
repo=$(gh repo view --json nameWithOwner -q .nameWithOwner)
gh api -X PATCH "repos/$repo/pulls/$pr" "${args[@]}" -q '"updated #\(.number): \(.title)"'
