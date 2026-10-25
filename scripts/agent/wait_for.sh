#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Block until a background job finishes, then print its verdict and log tail. Run it with the
# harness's run_in_background so its exit is the notification; never poll in a sleep loop.
#
#   scripts/agent/wait_for.sh --pid PID [--log FILE]
#   scripts/agent/wait_for.sh --log FILE --marker REGEX
#   common: [--stall-min M] [--timeout-min T] [--tail N]
#
# --pid waits on kill -0 (never pgrep -f: it matches its own command line).
# --marker waits for a line matching REGEX in --log (e.g. '^rc=' or 'verdict:').
# --stall-min exits 3 when --log has not grown for M minutes: a hung tuner cell or a stuck
#   kernel then reaches the session in M minutes instead of when the user asks for status.
# Exit: 0 done, 2 usage, 3 stalled, 4 timed out.
set -u

pid="" log="" marker="" stall=0 timeout=0 tail_n=25
while [ "$#" -gt 0 ]; do
    case $1 in
        --pid) pid=$2; shift 2 ;;
        --log) log=$2; shift 2 ;;
        --marker) marker=$2; shift 2 ;;
        --stall-min) stall=$2; shift 2 ;;
        --timeout-min) timeout=$2; shift 2 ;;
        --tail) tail_n=$2; shift 2 ;;
        -h|--help) sed -n '3,15p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "wait_for: unknown argument $1" >&2; exit 2 ;;
    esac
done
if [ -z "$pid" ] && { [ -z "$log" ] || [ -z "$marker" ]; }; then
    echo "wait_for: need --pid, or --log with --marker" >&2; exit 2
fi
if [ "$stall" != 0 ] && [ -z "$log" ]; then
    echo "wait_for: --stall-min needs --log" >&2; exit 2
fi

start=$(date +%s)
last_size=-1
last_change=$start
finish() {
    local code=$1 why=$2
    echo "wait_for: $why after $(( ($(date +%s) - start) / 60 )) min"
    if [ -n "$log" ] && [ -f "$log" ]; then
        echo "--- last $tail_n lines of $log"
        tail -n "$tail_n" "$log"
    fi
    exit "$code"
}

while :; do
    if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
        finish 0 "pid $pid exited"
    fi
    if [ -n "$marker" ] && [ -f "$log" ] && grep -Eq -- "$marker" "$log"; then
        finish 0 "marker /$marker/ seen"
    fi
    now=$(date +%s)
    if [ -n "$log" ] && [ -f "$log" ]; then
        size=$(stat -c %s "$log" 2>/dev/null || echo 0)
        if [ "$size" != "$last_size" ]; then last_size=$size; last_change=$now; fi
    fi
    if [ "$stall" != 0 ] && [ $(( now - last_change )) -ge $(( stall * 60 )) ]; then
        finish 3 "STALLED: $log unchanged for $stall min"
    fi
    if [ "$timeout" != 0 ] && [ $(( now - start )) -ge $(( timeout * 60 )) ]; then
        finish 4 "TIMED OUT at $timeout min"
    fi
    sleep 15
done
