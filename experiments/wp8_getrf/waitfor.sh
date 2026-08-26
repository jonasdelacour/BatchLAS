#!/usr/bin/env bash
# Block until $1 has at least $2 non-empty lines. Used to sequence sweeps
# without polling.
set -u
f="$1"; want="$2"
while true; do
  c=$(grep -c . "$f" 2>/dev/null || echo 0)
  if [ "$c" -ge "$want" ]; then echo "$f complete: $c rows"; exit 0; fi
  sleep 30
done
