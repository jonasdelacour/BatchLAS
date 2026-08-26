#!/usr/bin/env bash
# route_diff AFTER captures, plus the health checks and the diff against the
# BEFORE pair taken at the same tree minus the four preferred() clauses.
#
# THE INTENDED MOVES ARE ENUMERATED IN ADVANCE (D4 part C3): anything outside
# this list is a defect until argued otherwise.
#
#   getrf   float order >= 256, cfloat order >= 512     vendor:auto -> native:blocked
#   getri   float order >= 128, cfloat order >= 256     vendor:auto -> native:blocked
#   getrs   float nrhs >= 64, double nrhs >= 128,       vendor:auto -> native:blocked
#           both AND batch >= 128
#           (this line read "float/double nrhs >= 128" while the SHIPPED clause
#           is float nrhs >= 64 -- a script whose whole purpose is catching
#           unintended route moves must not describe a different clause than the
#           one that landed. The float boundary was narrowed to 128 during the
#           contaminated sweep and reverted to 64 once the box was idle; this
#           comment was not reverted with it.)
#   gemv    cdouble, transA != NoTrans, red_len 64..352,
#           out_len >= 256, batch >= 320                vendor:auto -> native:cta
#
# PLUS ADDED ROWS from the new pure-layer cases in route_vocabulary_tests.cc,
# which call the INSTRUMENTED resolver and therefore write Backend::AUTO rows.
# Those are fabricated shapes, not library decisions; filter on backend != AUTO
# before concluding anything about what the library routes.
set -u
cd /home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan
export CUDA_VISIBLE_DEVICES=0
scripts/route_diff.sh capture build          wp8-after-v
scripts/route_diff.sh capture build-novendor wp8-after-nv
bash experiments/wp8_route/health.sh wp8-after-v wp8-after-nv
echo "=== DIFF v ==="
scripts/route_diff.sh compare wp8-before-v  wp8-after-v  || true
echo "=== DIFF nv ==="
scripts/route_diff.sh compare wp8-before-nv wp8-after-nv || true
echo AFTER_DONE
