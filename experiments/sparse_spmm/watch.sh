#!/usr/bin/env bash
# Emit one line per milestone of a sweep pass, and one line when it ends either
# way -- a filter that only matched the happy path would be silent through a
# crash, which looks identical to "still running".
D="$(cd "$(dirname "$0")" && pwd)"
PASS="${1:?pass}"
LOG="$D/$PASS.log"
last=0
while true; do
  n=$(ls "$D/$PASS" 2>/dev/null | grep -v routes | grep -c csv || true)
  if [ "$n" -ge $((last + 20)) ]; then echo "$PASS: $n csvs written"; last=$n; fi
  if grep -q "PASS $PASS complete" "$LOG" 2>/dev/null; then
    echo "$PASS COMPLETE ($n csvs)"; break
  fi
  if grep -qE "REFUSING|Error|error:|Aborted|terminate called|what\(\):" "$LOG" 2>/dev/null; then
    echo "$PASS ERROR SIGNATURE in log ($n csvs): $(grep -m1 -E 'REFUSING|Error|error:|Aborted|terminate called|what\(\):' "$LOG")"; break
  fi
  if ! pgrep -f "run_all.sh $PASS" > /dev/null; then
    echo "$PASS RUNNER EXITED without completion marker ($n csvs)"; break
  fi
  sleep 30
done
