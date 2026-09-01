#!/usr/bin/env bash
# The three remaining sweeps, strictly sequential, one harness at a time.
set -eu
D="$(cd "$(dirname "$0")" && pwd)"
"$D/run_cfloat_edge.sh" cfedge1
"$D/run_cfloat_edge.sh" cfedge2
"$D/run_satext.sh" sat1
"$D/run_satext.sh" sat2
echo "TAIL SWEEPS complete"
