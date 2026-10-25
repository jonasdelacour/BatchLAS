#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Who is tuning, building or benchmarking on this box, and who holds which lock. Run it when a
# tune or build "does nothing": the usual cause is a queue behind a lock held by a run started
# days ago from another worktree, or by an orphaned worker.
#
#   scripts/agent/tune_procs.sh                 # list
#   scripts/agent/tune_procs.sh --kill PID      # TERM PID and every descendant, then KILL leftovers
#
# --kill only acts on your own processes; ask the user before killing a run you did not start.
set -u
me=$(id -un)

if [ "${1:-}" = --kill ]; then
    root=${2:?--kill needs a pid}
    [ "$(ps -o user= -p "$root" 2>/dev/null | tr -d ' ')" = "$me" ] || { echo "tune_procs: $root is not yours" >&2; exit 1; }
    tree() { local p; for p in $(pgrep -P "$1"); do tree "$p"; echo "$p"; done; }
    victims="$(tree "$root") $root"
    echo "tune_procs: TERM $victims"
    kill -TERM $victims 2>/dev/null
    for _ in 1 2 3 4 5 6 7 8 9 10; do
        alive=""; for p in $victims; do kill -0 "$p" 2>/dev/null && alive="$alive $p"; done
        [ -z "$alive" ] && { echo "tune_procs: all gone"; exit 0; }
        sleep 1
    done
    echo "tune_procs: KILL$alive"; kill -KILL $alive 2>/dev/null
    exit 0
fi

echo "== tuner, benchmark, build and test processes"
printf '%-8s %-8s %-10s %-11s %-s\n' PID PPID USER ELAPSED "COMMAND (cwd)"
ps -eo pid=,ppid=,user=,etime=,args= | grep -E 'batchlas_tune|_benchmark|benchviz|ctest|cmake --build|scripts/b |gate.sh|make -j' \
    | grep -v grep | while read -r pid ppid user etime args; do
        cwd=$(readlink "/proc/$pid/cwd" 2>/dev/null || echo '?')
        orphan=""; [ "$ppid" = 1 ] && orphan=" [ORPHAN]"
        printf '%-8s %-8s %-10s %-11s %s (%s)%s\n' "$pid" "$ppid" "$user" "$etime" "${args:0:110}" "$cwd" "$orphan"
    done

echo "== locks"
for lk in /tmp/batchlas-build.lock /tmp/batchlas_tune_gpu*.lock; do
    [ -e "$lk" ] || continue
    ino=$(stat -c %i "$lk")
    holders=$(awk -v ino="$ino" '{split($6, a, ":"); if (a[3] == ino) print $5}' /proc/locks | sort -u | tr '\n' ' ')
    if [ -n "$holders" ]; then
        echo "$lk HELD by pid(s) $holders$( [ -f "$lk.info" ] && echo "; info: $(cat "$lk.info")")"
        for p in $holders; do ps -o pid=,etime=,args= -p "$p" 2>/dev/null | cut -c1-140 | sed 's/^/    /'; done
    else
        echo "$lk free"
    fi
done
