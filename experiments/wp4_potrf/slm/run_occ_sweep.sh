#!/usr/bin/env bash
# blocks/SM as a function of the SLM request, MEASURED by ncu rather than derived
# from spec:284's assumed 102400 pool. One ncu run per (bytes, wg) so each launch
# is unambiguous.
#
# Register limits here are NOT predictive of the real potrf CTA kernel (this probe
# uses 38 regs/thread); launch__occupancy_limit_shared_mem is, since it depends
# only on the shared request and the carveout the driver picks.
set -euo pipefail
W=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan
D="$W/experiments/wp4_potrf/slm"
NCU=/usr/local/cuda-13.2/bin/ncu
out="$D/occ_sweep.csv"
echo "bytes,wg,limit_shared_blocks,limit_reg_blocks,limit_warp_blocks,limit_blocks,regs,static_shmem,dynamic_shmem,config_size,max_warps_pct" > "$out"

M=launch__occupancy_limit_shared_mem,launch__occupancy_limit_registers,launch__occupancy_limit_warps,launch__occupancy_limit_blocks,launch__registers_per_thread,launch__shared_mem_per_block_static,launch__shared_mem_per_block_dynamic,launch__shared_mem_config_size,sm__maximum_warps_per_active_cycle_pct

for BYTES in 8192 12800 17066 25600 32768 45056 49408 65536 81920 97280 101120; do
  for WG in 32 64 128 256; do
    csv=$("$W/experiments/gpu_guard.sh" 0 "$NCU" --metrics "$M" --csv "$D/slm_occ" "$BYTES" "$WG" 512 2>/dev/null | grep '^"0"' || true)
    if [ -z "$csv" ]; then echo "$BYTES,$WG,NA,NA,NA,NA,NA,NA,NA,NA,NA" >> "$out"; continue; fi
    get() { echo "$csv" | grep "\"$1\"" | sed 's/.*,"\([^"]*\)"$/\1/' | tr -d ','; }
    echo "$BYTES,$WG,$(get launch__occupancy_limit_shared_mem),$(get launch__occupancy_limit_registers),$(get launch__occupancy_limit_warps),$(get launch__occupancy_limit_blocks),$(get launch__registers_per_thread),$(get launch__shared_mem_per_block_static),$(get launch__shared_mem_per_block_dynamic),$(get launch__shared_mem_config_size),$(get sm__maximum_warps_per_active_cycle_pct)" >> "$out"
  done
done
echo "wrote $out"
