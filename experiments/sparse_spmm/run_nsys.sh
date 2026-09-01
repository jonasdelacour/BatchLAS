#!/usr/bin/env bash
set -eu
D="$(cd "$(dirname "$0")" && pwd)"
OUT="$D/nsys"
mkdir -p "$OUT"
# cell L (lanczos, L2-resident), cell L at DRAM residency, cell M and cell S,
# plus the transposed LOBPCG cell the native arm LOSES -- both routes each.
run() { "$D/nsys_split.sh" "$OUT" "$1" "$2" "${@:3}"; }
for route in vendor native:direct; do
  run float "$route" 1024 3  2  512  0 0 1 0
  run float "$route" 1024 3  2  4096 0 0 1 0
  run float "$route" 1024 16 12 512  0 0 1 0
  run float "$route" 2048 16 25 128  0 0 1 0
  run float "$route" 2048 16 25 128  0 0 1 1
done
echo "NSYS complete"
