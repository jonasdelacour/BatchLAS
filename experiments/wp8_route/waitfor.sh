#!/usr/bin/env bash
# waitfor.sh <file> <min-lines> [label] -- block until a sweep's CSV reaches a
# line count, then print one line. One notification, no polling in the chat.
set -u
F="$1"; N="$2"; L="${3:-DONE}"
while true; do
  if [ -f "$F" ]; then
    c=$(grep -c . "$F" 2>/dev/null | head -1)
  else
    c=0
  fi
  [ "${c:-0}" -ge "$N" ] && break
  sleep 30
done
echo "$L (${c} lines)"
