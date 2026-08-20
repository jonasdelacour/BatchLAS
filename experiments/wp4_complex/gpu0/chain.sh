#!/usr/bin/env bash
# Wait for sweep.sh to finish, then run sweep2.sh and the kernel-identity
# confirmation. Serialised on purpose: two benchmark processes on one GPU would
# poison both.
set -uo pipefail
cd "$(dirname "$0")/../../.."
while pgrep -f "wp4_complex/gpu0/sweep.sh" > /dev/null; do sleep 20; done
echo "sweep1 finished; starting sweep2"
./experiments/wp4_complex/gpu0/sweep2.sh
echo "sweep2 finished; confirming kernel identities"
./experiments/wp4_complex/gpu0/confirm_kernels.sh
echo "CHAIN_DONE"
