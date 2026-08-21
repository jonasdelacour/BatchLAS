#!/usr/bin/env bash
# Generalisation of scripts/register_probe.sh to ANY of the 14 device-link units.
#
# WHY THIS EXISTS. scripts/register_probe.sh hardcodes
# build/src/CMakeFiles/batchlas_sycl.dir/link.txt (its line 41), and
# build/src/CMakeFiles/batchlas_sycl.dir/link.txt links exactly two objects:
# sycl/gemm_kernels.cc.o and sycl/trsm_native.cc.o. Per WP4_POTRF_SPEC_CORRECTIONS.md
# W12 and its revised step 1.2, potrf_cta.cc / potrf_blocked.cc land in
# src/extensions/ and therefore in a DIFFERENT shared library -- so the stock
# probe would report "0 kernels with non-zero spill" for a tree whose potrf
# kernel it never compiled. That is exactly the phantom measurement the
# corrections document warns about.
#
# Same mechanism as the stock script: replay <target>'s link.txt verbatim with one
# extra -Xsycl-target-backend/-Xcuda-ptxas -v pair and the -o redirected out of the
# build tree, so no reconfigure is needed and the flags stay what the real build uses.
#
# Usage: regprobe_any.sh <cmake-target> <out.log>
#   e.g. regprobe_any.sh batchlas_extensions_cta /path/to/cta.log
set -uo pipefail
ROOT=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan
TGT="${1:?usage: regprobe_any.sh <cmake-target> <out.log>}"
OUT="${2:?usage: regprobe_any.sh <cmake-target> <out.log>}"
SCRATCH=/home/jonaslacour/.claude/jobs/20812aa0/tmp
mkdir -p "$SCRATCH"
cd "$ROOT/build/src" || exit 1

LT="CMakeFiles/$TGT.dir/link.txt"
[ -f "$LT" ] || { echo "no link.txt for target '$TGT'"; exit 2; }
LINE=$(cat "$LT")
SO="lib$TGT.so"
LINE=${LINE/-o $SO/-o $SCRATCH\/regprobe_$TGT.so}
LINE="$LINE -Xsycl-target-backend=nvptx64-nvidia-cuda -Xcuda-ptxas -v"

/usr/bin/time -f "LINK %e s real, %U s user" bash -c "$LINE" > "$OUT" 2>&1
rc=$?
echo "target=$TGT exit=$rc log=$OUT"
grep -c 'Compiling entry function' "$OUT" | sed 's/^/entry functions: /'
echo -n "kernels with non-zero spill: "
grep -E 'spill (stores|loads)' "$OUT" | grep -vcE '0 bytes spill stores, 0 bytes spill loads'
grep 'LINK ' "$OUT" || true
