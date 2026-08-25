#!/usr/bin/env bash
# WP7 R4 follow-up 2. Pin the cdouble+Trans slow region down:
#   (a) where its m boundary is, at finer granularity than the octave grid;
#   (b) whether it is an ld == m alignment effect (LD=m+1 keeps the shape and
#       the traffic identical and only breaks the alignment) -- if padding ld
#       fixes it, WP7's fix is a routing/pad decision, not a new kernel;
#   (c) whether it depends on batch, i.e. on the CTA count.
set -uo pipefail
W=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan
D="$W/experiments/wp7_gemv/baseline"
BIN="$D/gemvbase_v"
OUT="${OUT:-$D/refine.csv}"
export CUDA_VISIBLE_DEVICES=0
export OPENBLAS_CORETYPE=SKYLAKEX
export WARM_S=0.6
R=11
GB=$((1024*1024*1024))

echo "tag,type,m,n,batch,transA,median_ms,mean_ms,rel_sd,GBs,frac_of_900,relerr,ld" > "$OUT"
emit() { tag="$1"; shift; ldv="$1"; shift
  LD="$ldv" "$BIN" "$@" 2>>"$D/refine_err.txt" | sed "s/^/$tag,/" >> "$OUT"; }

# (a) m boundary, n fixed at 128 and 256, ~1 GB of A each.
for n in 128 256; do
  for m in 48 64 80 96 112 128 144 160 192 224 256 288 320 384 448 512; do
    b=$(( GB / (m*n*16) )); [ "$b" -lt 32 ] && b=32
    emit mboundary 0 cdouble "$m" "$n" "$b" T "$R"
  done
done

# (b) ld sensitivity at the two worst cells from the main sweep.
for ldv in 0 257 264 320; do
  emit ldtest "$ldv" cdouble 256 256 1024 T "$R"
done
for ldv in 0 65 72 128; do
  emit ldtest "$ldv" cdouble 64 2048 512 T "$R"
done
# control: the same ld experiment on a cell that is already AT the roof, so a
# "padding helps" reading cannot be an artefact of padding itself.
for ldv in 0 513 520; do
  emit ldctrl "$ldv" cdouble 512 512 256 T "$R"
done

# (c) batch dependence of the worst cell.
for b in 64 128 256 512 1024 2048 4096; do
  emit batchdep 0 cdouble 256 256 "$b" T "$R"
done
# and its NoTrans twin, which the main sweep put at the roof.
for b in 64 256 1024 4096; do
  emit batchdep 0 cdouble 256 256 "$b" N "$R"
done
echo "wrote $OUT"
