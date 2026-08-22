#!/usr/bin/env bash
# ANTI-VACUITY. Damage the checker's REFERENCE six ways and record what each one
# does to every residual it guards. A break that does NOT turn red is reported,
# not hidden -- BREAK=1 and BREAK=4 each have a documented null result.
#
#   0  control (no damage)
#   1  drop the LAST reflector              (sy2sb short-final-panel class)
#   2  apply the reflectors in reversed (WY) order
#   3  drop the last COLUMN of the explicit Q
#   4  conjugate tau                        (complex phase-convention class)
#   5  drop a MIDDLE reflector
#
# Columns of the CSV, per emitted row:
#   op,type,n,batch,med_ms,mean_ms,rel_sd,GFLOPs,geqrf_res,ortho,recon,ws[,route,nb,dQ]
set -uo pipefail
D=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan/experiments/wp5_qr/baseline
export CUDA_VISIBLE_DEVICES=1 WARM_S=0.2
echo "break,type,rows"
for t in float double cfloat cdouble; do
  for b in 0 1 2 3 4 5; do
    out="$(BREAK=$b "$D/wp5qr_v" qcheck "$t" 96 8 2 2>&1 | tr '\n' '|')"
    echo "$b,$t,$out"
  done
done
