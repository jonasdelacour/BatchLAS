#!/usr/bin/env bash
# Block until $1 has at least $2 lines. Used to sequence a long sweep against
# analysis work without a bare sleep.
set -u
f="$1"; want="$2"
while true; do
  if [ -f "$f" ] && [ "$(wc -l < "$f")" -ge "$want" ]; then break; fi
  sleep 20
done
echo "READY $f $(wc -l < "$f") lines"
