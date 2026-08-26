#!/usr/bin/env bash
cd /home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan
export CUDA_VISIBLE_DEVICES=0
scripts/route_diff.sh capture build          wp8-before-v
scripts/route_diff.sh capture build-novendor wp8-before-nv
scripts/route_diff.sh compare wp7-repair-v  wp8-before-v
scripts/route_diff.sh compare wp7-repair-nv wp8-before-nv
echo DONE_BEFORE
