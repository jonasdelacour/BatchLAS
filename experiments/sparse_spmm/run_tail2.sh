#!/usr/bin/env bash
set -eu
D="$(cd "$(dirname "$0")" && pwd)"
"$D/run_scatter_ladder.sh" scl1
"$D/run_scatter_ladder.sh" scl2
echo "SCATTER LADDER PASSES complete"
