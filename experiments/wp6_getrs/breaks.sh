#!/usr/bin/env bash
# DELIBERATE BREAKS. Each one disables ONE guarded property of the fused tier,
# rebuilds the vendor-free library, and re-runs the host-oracle harness. A break
# that leaves the harness GREEN is a BLIND GUARD and is reported as such -- this
# repository has seven recorded ones, including a test written in the same change
# as its fix.
#
# The file is backed up and restored by COPY, not by git: src/extensions/
# getrs_fused.cc is a new, untracked file, so `git checkout --` cannot restore it
# and would report a spurious "did not match" instead.
#
# usage: breaks.sh <break-name>
#   B1  transposed path walks the interchange list FORWARDS instead of backwards
#   B3  NoTrans path drops the interchange walk entirely
#   B4  transposed path drops the permutation from the output altogether
#   B5  the diagonal block's leading dimension loses its +1 bank-conflict pad
#   B6  ConjTrans stops conjugating
#   B7  the register cap on the work-group width is removed
set -uo pipefail
W=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan
F="$W/src/extensions/getrs_fused.cc"
BAK="/home/jonaslacour/.claude/jobs/20812aa0/tmp/getrs_fused.cc.orig"
B="${1:?usage: breaks.sh <B1|B3|B4|B5|B6|B7>}"

[ -f "$BAK" ] || cp "$F" "$BAK"
cp "$BAK" "$F"

case "$B" in
  B1) perl -0pi -e 's/for \(int k = n - 1; k >= 0; --k\) \{\s*\n(\s*)const int p = pv\[k\] - 1;/for (int k = 0; k < n; ++k) {\n$1const int p = pv[k] - 1;/' "$F" ;;
  B3) perl -0pi -e 's/interchange list, walked FORWARDS, in LOCAL memory\n(\s*)if \(tid < nrhs\)/interchange list -- BROKEN\n$1if (false)/' "$F" ;;
  B4) perl -0pi -e 's/if \(tid < nrhs\) \{\n(\s*)D\* const yc = y \+ static_cast<std::size_t>\(tid\) \* static_cast<std::size_t>\(n\);\n(\s*)for \(int k = n - 1;/if (false) {\n$1D* const yc = y + static_cast<std::size_t>(tid) * static_cast<std::size_t>(n);\n$2for (int k = n - 1;/' "$F" ;;
  B5) sed -i 's|inline int getrs_fused_blk_ld(int nb) { return nb + 1; }|inline int getrs_fused_blk_ld(int nb) { return nb; }|' "$F" ;;
  B6) sed -i 's|return conj ? dev_conj(a) : a;|return a;|' "$F" ;;
  B7) sed -i 's|    if (wg > cap) wg = cap;|    /* BROKEN: register cap removed */|' "$F" ;;
  *) echo "unknown break $B"; exit 2 ;;
esac

if cmp -s "$BAK" "$F"; then
  echo "BREAK $B DID NOT APPLY -- the pattern did not match. Aborting."
  cp "$BAK" "$F"; exit 3
fi
echo "=== BREAK $B applied ($(diff "$BAK" "$F" | grep -c '^[<>]') changed lines); rebuilding ==="
if ! cmake --build "$W/build-novendor" -j 32 --target batchlas_extensions_factorization > /dev/null 2>&1; then
  echo "BREAK $B: BUILD FAILED (that is itself a red)"; cp "$BAK" "$F"; exit 0
fi
bash "$W/experiments/wp6_lu/bench/build_nv.sh" > /dev/null 2>&1
echo "=== host oracle, all three transA, vendor-free ==="
if [ "$B" = "B7" ]; then
  CUDA_VISIBLE_DEVICES="${GPU:-1}" WARM_S=0.05 NTRANS=3 NPROBE=1 \
    "$W/experiments/wp6_lu/bench/lubench6_nv" getrs float 2048 8 8 1 2>&1 | tail -5
else
  for t in float cdouble; do
    for n in 64 129 512; do
      CUDA_VISIBLE_DEVICES="${GPU:-1}" WARM_S=0.05 NTRANS=3 NPROBE=2 \
        "$W/experiments/wp6_lu/bench/lubench6_nv" getrs "$t" "$n" 1 64 2 2>&1 | grep -E "resid|^getrs"
    done
  done
fi
cp "$BAK" "$F"
echo "=== BREAK $B reverted ==="
