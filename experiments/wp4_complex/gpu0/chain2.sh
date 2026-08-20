#!/usr/bin/env bash
# Runs after chain.sh (sweep1 -> sweep2 -> confirm_kernels). Serialised: two
# benchmark processes on one GPU would poison both.
set -uo pipefail
cd "$(dirname "$0")/../../.."
while pgrep -f "wp4_complex/gpu0/chain.sh" > /dev/null; do sleep 20; done
echo "chain1 finished; starting sweep3"
./experiments/wp4_complex/gpu0/sweep3.sh
echo "CHAIN2_DONE"
